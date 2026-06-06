"""
因子研究完整流水线 — 4 阶段断点续跑架构。

用法: python run_factor_research.py

阶段:
  Stage 1 — 预取数据: 日线+财务数据缓存到本地 (~40min 首次, 后续秒过)
  Stage 2 — 构建面板: 从缓存计算 12 因子 → panel.parquet (~30s)
  Stage 3 — 预处理: MAD 去极值 + Z-score 标准化 → preprocessed.parquet (~10s)
  Stage 4 — 分析: Rank IC + 分层回测 + 多因子合成 → output/ (~5s)

中断恢复: Ctrl+C 或超时中断后, 重新运行 python run_factor_research.py,
已完成的阶段自动跳过, 从断点继续。只有 Stage 1 内的财务数据获取比较慢,
但已缓存的股票不会重复请求。

状态文件: .pipeline_state.json (自动生成)
中间产物: output/panel.parquet, output/preprocessed.parquet
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# ─── 核心模块 ─────────────────────────────────────────
from data.fetcher import Fetcher
from data.cleaner import winsorize_mad, standardize_cross_section

from factor_lib.momentum import (
    compute_momentum_1m, compute_momentum_3m,
    compute_momentum_6m, compute_momentum_12m_1m,
)
from factor_lib.volatility import compute_volatility_20d, compute_volatility_60d, compute_beta
from factor_lib.growth import compute_revenue_growth, compute_earnings_growth
from factor_lib.technical import compute_volume_20d_change, compute_price_ma_deviation

from factor_research.ic_analysis import compute_rank_ic, compute_ic_summary
from factor_research.group_backtest import (
    assign_quantile_groups, compute_group_returns, compute_long_short,
)
from factor_research.backtest_engine import combine_factors, compute_performance
from factor_research.report import factor_summary_table
from data.cleaner import neutralize_industry_market_cap

# ─── 板块分类 ─────────────────────────────────────────

def _classify_board(symbol: str) -> str:
    """按股票代码前缀分类到 5 大板块。"""
    sym = str(symbol)
    if sym.startswith("688"):
        return "科创板"
    if sym.startswith("300") or sym.startswith("301"):
        return "创业板"
    if sym.startswith("002"):
        return "深市中小板"
    if sym.startswith("600") or sym.startswith("601") or sym.startswith("603") or sym.startswith("605"):
        return "沪市主板"
    if sym.startswith("000") or sym.startswith("001"):
        return "深市主板"
    return "其他"

# ─── 全局配置 ─────────────────────────────────────────
STOCK_POOL = "000906"         # 中证 800
START_DATE = "20170101"
END_DATE   = "20241231"
MAX_STOCKS = 300              # 扩大到 300 只 (全量 700 需较长时间)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

STATE_FILE      = Path(".pipeline_state.json")
PANEL_FILE      = OUTPUT_DIR / "panel.parquet"
PREPROCESSED_FILE = OUTPUT_DIR / "preprocessed.parquet"

# 12 个因子列表
FACTOR_COLS = [
    # 动量
    "Mom_1M", "Mom_3M", "Mom_6M", "Mom_12M_1M",
    # 波动率
    "Vol_20D", "Vol_60D", "Beta",
    # 价值
    "BP", "EP",
    # 质量
    "ROE", "Debt_Ratio", "Net_Profit_Margin",
    # 成长
    "RevGrowth_YoY", "ProfitGrowth_YoY",
    # 技术面
    "VolChg_20D", "PriceDev_20D",
]


# ═══════════════════════════════════════════════════════
# 状态管理
# ═══════════════════════════════════════════════════════

def load_state() -> dict:
    """读取流水线进度。"""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"stage": 0}


def save_state(state: dict) -> None:
    """写入流水线进度。"""
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def is_stage_done(stage: int) -> bool:
    """检查某阶段是否已完成。"""
    state = load_state()
    return state.get("stage", 0) >= stage


# ═══════════════════════════════════════════════════════
# Stage 1 — 预取数据
# ═══════════════════════════════════════════════════════

def stage1_prefetch(f: Fetcher, symbols: list[str]) -> None:
    """
    将所有股票的日线和财务数据缓存到本地。

    这是唯一慢的阶段（首次 ~40min, 财务每只 ~10s）。
    每 5 只保存一次进度。中断后重跑自动跳过已缓存的股票。
    """
    if is_stage_done(1):
        print("[Stage 1/4] 已完成, 跳过\n")
        return

    print("=" * 60)
    print("[Stage 1/4] 预取数据到本地缓存")
    print(f"           股票池: {len(symbols)} 只")
    print("=" * 60)

    state = load_state()
    done_count = state.get("stage1_done", 0)
    total = len(symbols)

    # 统计已有缓存
    cached_daily = len(list(Path("data/raw").glob("daily_*.csv")))
    cached_fin   = len(list(Path("data/raw").glob("financial_*.csv")))
    print(f"已有缓存: 日线 {cached_daily} / 财务 {cached_fin}\n")

    pbar = tqdm(total=total, desc="预取", initial=done_count)
    new_fin = 0

    for i, sym in enumerate(symbols):
        if i < done_count:
            pbar.update(1)
            continue

        # --- 日线 (缓存内很快) ---
        try:
            f.get_daily(sym, START_DATE, END_DATE)
        except Exception as e:
            pbar.write(f"  [日线] {sym}: {e}")

        # --- 财务 (慢, 但 get_financial_history 内部检查缓存) ---
        try:
            fin_cache = Path("data/raw") / f"financial_{sym}_ths_history.csv"
            if not fin_cache.exists():
                f.get_financial_history(sym)
                new_fin += 1
        except Exception as e:
            pbar.write(f"  [财务] {sym}: {e}")

        # 每 5 只存盘一次进度
        if (i + 1) % 5 == 0 or (i + 1) == total:
            state["stage1_done"] = i + 1
            save_state(state)

        pbar.update(1)

    pbar.close()

    # 标记 Stage 1 完成
    state["stage"] = 1
    state.pop("stage1_done", None)
    save_state(state)

    total_fin = len(list(Path("data/raw").glob("financial_*.csv")))
    print(f"[Stage 1/4] 完成: 本次新增财务缓存 {new_fin} 只, "
          f"总计财务缓存 {total_fin} 只\n")


# ═══════════════════════════════════════════════════════
# Stage 2 — 构建因子面板
# ═══════════════════════════════════════════════════════

def _build_panel(f: Fetcher, symbols: list[str]) -> pd.DataFrame:
    """
    从本地缓存构建因子面板。

    1. 读取每只股票的日线缓存 → 月频采样
    2. 读取每只股票的财务缓存 → per-share 估值因子
    3. 计算行情因子（动量/波动率）
    4. 合并为一张面板
    """
    # ── 第一遍: 月频行情 ──
    all_daily = []

    for sym in tqdm(symbols, desc="读取缓存(日线)"):
        # 日线 → 月频
        try:
            daily = f.get_daily(sym, START_DATE, END_DATE)  # 走缓存, 快
            daily = daily[daily["日期"] >= START_DATE].copy()
            daily["month"] = daily["日期"].dt.to_period("M")
            month_end = daily.groupby("month").tail(1).copy()
            month_end["symbol"] = sym
            all_daily.append(month_end)
        except Exception:
            continue

    if not all_daily:
        raise RuntimeError("未能获取任何日线数据")

    daily_panel = pd.concat(all_daily, ignore_index=True)
    daily_panel = daily_panel.rename(columns={"日期": "date"})

    # ── 第二遍: 全量日线（用于因子计算） ──
    daily_full = []
    for sym in tqdm(symbols, desc="日度行情(因子计算)"):
        try:
            d = f.get_daily(sym, START_DATE, END_DATE)
            d["symbol"] = sym
            d = d.rename(columns={"日期": "date"})
            daily_full.append(d[["date", "symbol", "收盘", "成交量", "成交额"]])
        except Exception:
            continue

    if daily_full:
        daily_all = pd.concat(daily_full, ignore_index=True)
        daily_all = daily_all.rename(columns={
            "收盘": "close", "成交量": "volume", "成交额": "amount"
        })
        # 安全去重: 防止同一只股票被重复加载 (例如指数成分股去重失败)
        daily_all = daily_all.drop_duplicates(subset=["date", "symbol"])
    else:
        daily_all = daily_panel[["date", "symbol"]].copy()
        daily_all["close"] = 0.0
        daily_all["volume"] = 0.0
        daily_all["amount"] = 0.0

    # 动量因子
    mom_1m   = compute_momentum_1m(daily_all)
    mom_3m   = compute_momentum_3m(daily_all)
    mom_6m   = compute_momentum_6m(daily_all)
    mom_12_1 = compute_momentum_12m_1m(daily_all)

    # 波动率因子
    vol_20 = compute_volatility_20d(daily_all)
    vol_60 = compute_volatility_60d(daily_all)
    beta   = compute_beta(daily_all)

    for fdf in [mom_1m, mom_3m, mom_6m, mom_12_1, vol_20, vol_60, beta]:
        if fdf is not None and not fdf.empty:
            daily_panel = daily_panel.merge(fdf, on=["date", "symbol"], how="left")

    # 技术面因子 (成交量变化 + 均线偏离)
    vol_chg = compute_volume_20d_change(daily_all)
    price_dev = compute_price_ma_deviation(daily_all, window=20)

    for fdf in [vol_chg, price_dev]:
        if fdf is not None and not fdf.empty:
            daily_panel = daily_panel.merge(fdf, on=["date", "symbol"], how="left")

    # ── 财务因子 (Point-in-Time 对齐) ──
    # 加载所有股票的完整财务历史
    fin_frames = []
    for sym in tqdm(symbols, desc="加载财务历史"):
        try:
            hist = f.get_financial_history(sym)
            if not hist.empty:
                fin_frames.append(hist)
        except Exception:
            continue

    if fin_frames:
        fin_all = pd.concat(fin_frames, ignore_index=True)
        fin_all = fin_all.dropna(subset=["symbol", "report_date"])

        # ── 成长因子 (基于财务历史, PIT 合并前计算) ──
        growth_rev = compute_revenue_growth(fin_all)
        growth_earn = compute_earnings_growth(fin_all)
        for gdf in [growth_rev, growth_earn]:
            if gdf is not None and not gdf.empty:
                fin_all = fin_all.merge(gdf, on=["symbol", "report_date"], how="left")

        # PIT 对齐: 对每个 symbol 组内, 取 report_date <= date 的最新报告
        # 用 groupby-apply 避免 merge_asof 跨组排序问题
        def _pit_merge(group: pd.DataFrame) -> pd.DataFrame:
            sym = group.name
            fin_sym = fin_all[fin_all["symbol"] == sym]
            if fin_sym.empty:
                return group
            group = group.sort_values("date")
            fin_sym = fin_sym.sort_values("report_date")
            return pd.merge_asof(
                group,
                fin_sym,
                left_on="date",
                right_on="report_date",
                direction="backward",
            )

        daily_panel = (
            daily_panel
            .groupby("symbol", group_keys=False)
            .apply(_pit_merge)
            .reset_index(drop=True)
        )

        # 统一列名: 中文 → 英文
        daily_panel = daily_panel.rename(columns={
            "销售净利率": "Net_Profit_Margin",
            "Rev_Growth_YoY": "RevGrowth_YoY",
            "Earnings_Growth": "ProfitGrowth_YoY",
        })

        daily_panel["股价"] = daily_panel["收盘"].astype(float)

        # BP = 每股净资产 / 股价 (现在每股净资产随报告期变化)
        if "每股净资产" in daily_panel.columns:
            daily_panel["BP"] = (
                daily_panel["每股净资产"].astype(float)
                / daily_panel["股价"].replace(0, float("nan"))
            )

        # EP = 每股收益 / 股价
        if "每股收益" in daily_panel.columns:
            daily_panel["EP"] = (
                daily_panel["每股收益"].astype(float)
                / daily_panel["股价"].replace(0, float("nan"))
            )

        # ROE / Debt_Ratio / Net_Profit_Margin 直接从 merge 后的列取
        # (merge_asof 自动带过来, 已统一为英文列名)

    return daily_panel


def stage2_build_panel(f: Fetcher, symbols: list[str]) -> pd.DataFrame:
    """构建因子面板, 落盘为 parquet。"""
    if is_stage_done(2) and PANEL_FILE.exists():
        print("[Stage 2/4] 已完成, 从 parquet 加载\n")
        return pd.read_parquet(PANEL_FILE)

    print("=" * 60)
    print("[Stage 2/4] 构建因子面板 (从缓存计算)")
    print("=" * 60)

    panel = _build_panel(f, symbols)

    # 添加板块分类 (用于后续行业中性化)
    panel["board"] = panel["symbol"].apply(_classify_board)

    panel.to_parquet(PANEL_FILE, index=False)

    state = load_state()
    state["stage"] = 2
    save_state(state)

    # 检查有多少因子列存在
    available = [c for c in FACTOR_COLS if c in panel.columns]
    print(f"[Stage 2/4] 完成: {panel.shape[0]} 行 × {panel.shape[1]} 列")
    print(f"          可用因子 ({len(available)}/12): {available}\n")
    return panel


# ═══════════════════════════════════════════════════════
# Stage 3 — 预处理
# ═══════════════════════════════════════════════════════

def stage3_preprocess(panel: pd.DataFrame) -> pd.DataFrame:
    """MAD 去极值 + 行业中性化 + Z-score 标准化, 落盘为 parquet。"""
    if is_stage_done(3) and PREPROCESSED_FILE.exists():
        print("[Stage 3/4] 已完成, 从 parquet 加载\n")
        return pd.read_parquet(PREPROCESSED_FILE)

    print("=" * 60)
    print("[Stage 3/4] 预处理 (MAD 去极值 + 行业中性化 + Z-score 标准化)")
    print("=" * 60)

    available = [c for c in FACTOR_COLS if c in panel.columns]
    print(f"处理 {len(available)} 个因子: {available}")

    # 确保板块列存在 (从 Stage 2 来)
    if "board" not in panel.columns:
        panel["board"] = panel["symbol"].apply(_classify_board)

    for col in tqdm(available, desc="预处理因子"):
        # 1. MAD 去极值
        panel[col] = panel.groupby("date")[col].transform(
            lambda x: winsorize_mad(x, n_mad=3.0)
        )
        # 2. 行业中性化 (板块作为行业代理; 市值数据暂不可用)
        panel = neutralize_industry_market_cap(
            panel, factor_col=col, industry_col="board", date_col="date"
        )
        # 3. Z-score 标准化 (在中性化后的值上做)
        panel = standardize_cross_section(
            panel, factor_col=f"{col}_neutral", date_col="date"
        )

    panel.to_parquet(PREPROCESSED_FILE, index=False)

    state = load_state()
    state["stage"] = 3
    save_state(state)

    neutral_z_cols = [c for c in panel.columns if c.endswith("_neutral_z")]
    print(f"[Stage 3/4] 完成: 生成 {len(neutral_z_cols)} 个中性化+标准化因子\n")
    return panel


# ═══════════════════════════════════════════════════════
# Stage 4 — 分析
# ═══════════════════════════════════════════════════════

def stage4_analyze(panel: pd.DataFrame) -> None:
    """
    IC 分析 + 分层回测 + 多因子合成。
    结果写入 output/ 目录。
    """
    print("=" * 60)
    print("[Stage 4/4] 因子分析")
    print("=" * 60)

    # ── 构造下期收益 ──
    panel = panel.sort_values(["symbol", "date"]).copy()
    panel["next_close"] = panel.groupby("symbol")["收盘"].shift(-1)
    panel["forward_return_1m"] = (
        panel["next_close"] - panel["收盘"].astype(float)
    ) / panel["收盘"].astype(float)
    panel = panel.dropna(subset=["forward_return_1m"])

    # ── IC 分析 ──
    factor_z_cols = [c for c in panel.columns if c.endswith("_neutral_z")]
    # 回退: 如果没有 neutral_z 列 (旧版 panel), 使用 _z 列
    if not factor_z_cols:
        factor_z_cols = [c for c in panel.columns if c.endswith("_z")]
    print(f"\nIC 分析: {len(factor_z_cols)} 个标准化因子\n")

    ic_results = {}
    for col in factor_z_cols:
        ic = compute_rank_ic(
            panel, factor_col=col, return_col="forward_return_1m", date_col="date"
        )
        summary = compute_ic_summary(ic)
        if summary:
            # 去掉 _neutral_z 或 _z 后缀显示因子名
            name = col.replace("_neutral_z", "").replace("_z", "")
            ic_results[name] = summary
            print(
                f"  {name:20s}  "
                f"IC_Mean={summary.get('IC_Mean', 0):+.4f}  "
                f"IC_IR={summary.get('IC_IR', 0):+.4f}  "
                f"Win={summary.get('IC_Win_Rate', 0):.1%}"
            )

    # 汇总表
    table = factor_summary_table(ic_results)
    table.to_csv(
        OUTPUT_DIR / "factor_ic_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n  → IC 汇总已保存到 output/factor_ic_summary.csv")

    # ── 分层回测 + 多因子合成 ──
    if table is not None and not table.empty and "IC_IR" in table.columns:
        best_factor = table.iloc[0]["因子"]
        best_col = f"{best_factor}_z"

        # 优先找 neutral_z 后缀, 回退找 _z 后缀
        if f"{best_factor}_neutral_z" in panel.columns:
            best_col = f"{best_factor}_neutral_z"
        elif f"{best_factor}_z" in panel.columns:
            best_col = f"{best_factor}_z"

        if best_col in panel.columns:
            print(f"\n分层回测 (最佳单因子: {best_factor})")
            assigned = assign_quantile_groups(
                panel, factor_col=best_col, n_groups=5, date_col="date"
            )
            g_rets = compute_group_returns(
                assigned, return_col="forward_return_1m", date_col="date"
            )
            ls = compute_long_short(g_rets)
            if not ls.empty:
                perf = compute_performance(ls["long_short_return"])
                print(
                    f"  单因子多空: Sharpe={perf['Sharpe_Ratio']:.3f}  "
                    f"AnnRet={perf['Annualized_Return']:.2%}  "
                    f"MaxDD={perf['Max_Drawdown']:.1%}"
                )

            # 多因子合成 — IC_IR 加权 + 符号翻转 + 去冗余
            print(f"\n多因子合成 (IC_IR 加权, 翻转负IC因子, 去冗余 |r|>0.7, {len(factor_z_cols)} 个因子)")
            combined = combine_factors(
                panel,
                factor_cols=factor_z_cols,
                method="ic_weighted",
                return_col="forward_return_1m",
                date_col="date",
                max_correlation=0.7,
                flip_sign=True,
            )
            assigned_m = assign_quantile_groups(
                combined, factor_col="composite_factor", n_groups=5, date_col="date"
            )
            g_rets_m = compute_group_returns(
                assigned_m, return_col="forward_return_1m", date_col="date"
            )
            ls_m = compute_long_short(g_rets_m)
            if not ls_m.empty:
                perf_m = compute_performance(ls_m["long_short_return"])
                print(
                    f"  复合因子多空: Sharpe={perf_m['Sharpe_Ratio']:.3f}  "
                    f"AnnRet={perf_m['Annualized_Return']:.2%}  "
                    f"MaxDD={perf_m['Max_Drawdown']:.1%}"
                )

    # 标记完成
    state = load_state()
    state["stage"] = 4
    save_state(state)

    print(f"\n{'=' * 60}")
    print("流水线全部完成。结果在 output/ 目录。")
    print("=" * 60)


# ═══════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("因子研究流水线 (4 阶段断点续跑)")
    print(f"股票池: {STOCK_POOL} | 区间: {START_DATE}-{END_DATE}")
    print("=" * 60)

    f = Fetcher()

    # ── 获取成分股 ──
    print("\n获取成分股...")
    try:
        symbols = f.get_index_members(STOCK_POOL)
    except Exception:
        print(f"  无法获取 {STOCK_POOL} 成分股, 使用预设列表")
        symbols = [
            "000001", "000002", "600036", "600519", "000858",
            "002415", "300750", "601318", "600276", "000333",
        ]

    if MAX_STOCKS and len(symbols) > MAX_STOCKS:
        # 随机采样以避免排序偏差 (指数 API 返回顺序不按市值)
        import random
        random.seed(42)
        symbols = random.sample(symbols, MAX_STOCKS)
    print(f"  股票池: {len(symbols)} 只\n")

    # ── 四阶段依次执行 ──
    stage1_prefetch(f, symbols)          # 预取 → 本地缓存
    panel = stage2_build_panel(f, symbols)  # 构建面板 → panel.parquet
    panel = stage3_preprocess(panel)        # 预处理 → preprocessed.parquet
    stage4_analyze(panel)                   # 分析 → output/


if __name__ == "__main__":
    main()
