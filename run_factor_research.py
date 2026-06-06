"""
因子研究完整流水线 — 一键运行入口。

用法: python run_factor_research.py

这个脚本把 data/fetcher → factor_lib → data/cleaner →
factor_research/ic_analysis → group_backtest → report 串起来,
输出一个完整的因子研究报告。
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# ─── 1. 数据获取 ─────────────────────────────────────
from data.fetcher import Fetcher
from data.cleaner import winsorize_mad, standardize_cross_section

from factor_lib.momentum import (
    compute_momentum_1m, compute_momentum_3m,
    compute_momentum_6m, compute_momentum_12m_1m,
)
from factor_lib.volatility import compute_volatility_20d, compute_volatility_60d, compute_beta
from factor_lib.value import compute_bp, compute_ep
from factor_lib.quality import compute_roe, compute_gross_margin, compute_debt_ratio
from factor_lib.growth import compute_revenue_growth, compute_earnings_growth

from factor_research.ic_analysis import compute_rank_ic, compute_ic_summary
from factor_research.group_backtest import assign_quantile_groups, compute_group_returns, compute_long_short
from factor_research.backtest_engine import combine_factors, compute_performance
from factor_research.report import (
    plot_ic_timeseries, plot_ic_distribution,
    plot_group_nav, plot_factor_correlation, factor_summary_table,
)

# ─── 配置 ──────────────────────────────────────────
STOCK_POOL = "000906"        # 中证 800
START_DATE = "20170101"
END_DATE = "20241231"
MAX_STOCKS = 200             # 测试阶段限制股票数, 全量改 None

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def build_panel(f: Fetcher, symbol_list: list[str]) -> pd.DataFrame:
    """
    构建因子面板数据。

    对每只股票:
    1. 获取日线行情
    2. 计算动量/波动率因子 (基于日线)
    3. 获取财务数据, 计算估值/质量/成长因子
    4. 合并为一张面板
    """
    print(f"\n构建因子面板: {len(symbol_list)} 只股票...")

    all_daily = []
    fin_rows = []

    for sym in tqdm(symbol_list, desc="获取数据"):
        # 日线
        try:
            daily = f.get_daily(sym, START_DATE, END_DATE)
            daily = daily[daily["日期"] >= START_DATE].copy()
            # 月频采样: 取每月最后一个交易日
            daily["month"] = daily["日期"].dt.to_period("M")
            month_end = daily.groupby("month").tail(1).copy()
            month_end["symbol"] = sym
            all_daily.append(month_end)
        except Exception:
            continue

        # 财务
        try:
            fin = f.get_financial(sym)
            if fin:
                fin["symbol"] = sym
                fin_rows.append(fin)
        except Exception:
            continue

    if not all_daily:
        raise RuntimeError("未能获取任何日线数据")

    daily_panel = pd.concat(all_daily, ignore_index=True)
    daily_panel = daily_panel.rename(columns={"日期": "date"})
    daily_panel["log_market_cap"] = np.log(
        daily_panel["收盘"].astype(float) * daily_panel.get("成交量", 1).astype(float)
    )

    # ── 计算行情因子 ──
    # 构造一个适合因子计算的日度格式
    daily_full = []
    for sym in tqdm(symbol_list[:MAX_STOCKS or len(symbol_list)], desc="日度行情"):
        try:
            d = f.get_daily(sym, START_DATE, END_DATE)
            d["symbol"] = sym
            d = d.rename(columns={"日期": "date"})
            daily_full.append(d[["date", "symbol", "收盘"]])
        except Exception:
            continue
    if daily_full:
        daily_all = pd.concat(daily_full, ignore_index=True)
        daily_all = daily_all.rename(columns={"收盘": "close"})
    else:
        daily_all = daily_panel[["date", "symbol"]].copy()
        daily_all["close"] = 0.0

    # 动量
    mom_1m = compute_momentum_1m(daily_all)
    mom_3m = compute_momentum_3m(daily_all)
    mom_6m = compute_momentum_6m(daily_all)
    mom_12_1 = compute_momentum_12m_1m(daily_all)

    # 波动率
    vol_20 = compute_volatility_20d(daily_all)
    vol_60 = compute_volatility_60d(daily_all)
    beta = compute_beta(daily_all)

    # 合并行情因子到月度面板
    for fdf in [mom_1m, mom_3m, mom_6m, mom_12_1, vol_20, vol_60, beta]:
        if fdf is not None and not fdf.empty:
            daily_panel = daily_panel.merge(fdf, on=["date", "symbol"], how="left")

    # ── 计算财务因子并合并 ──
    if fin_rows:
        fin_df = pd.DataFrame(fin_rows)
        fin_df = fin_df.rename(columns={"报告期": "report_date"})

        # 估值因子需要财务 + 市值
        bp = compute_bp(fin_df, daily_panel)
        ep = compute_ep(fin_df, daily_panel)
        if not bp.empty:
            daily_panel = daily_panel.merge(bp, on=["date", "symbol"], how="left")
        if not ep.empty:
            daily_panel = daily_panel.merge(ep, on=["date", "symbol"], how="left")

        roe = compute_roe(fin_df)
        gm = compute_gross_margin(fin_df)
        dr = compute_debt_ratio(fin_df)
        for fdf in [roe, gm, dr]:
            if fdf is not None and not fdf.empty:
                daily_panel = daily_panel.merge(
                    fdf, left_on="symbol", right_on="symbol", how="left"
                )

        rev_g = compute_revenue_growth(fin_df)
        earn_g = compute_earnings_growth(fin_df)
        for fdf in [rev_g, earn_g]:
            if fdf is not None and not fdf.empty:
                daily_panel = daily_panel.merge(
                    fdf, left_on="symbol", right_on="symbol", how="left"
                )

    return daily_panel


def preprocess_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    对面板中每个因子执行:
    去极值 → 中性化 → 标准化
    """
    # 候选因子列
    factor_cols = [
        "Mom_1M", "Mom_3M", "Mom_6M", "Mom_12M_1M",
        "Vol_20D", "Vol_60D", "Beta",
        "BP", "EP", "ROE", "Gross_Margin", "Debt_Ratio",
        "Rev_Growth_YoY", "Earnings_Growth",
    ]
    available = [c for c in factor_cols if c in panel.columns]
    print(f"预处理 {len(available)} 个因子: {available}")

    for col in tqdm(available, desc="预处理因子"):
        # 去极值
        panel[col] = panel.groupby("date")[col].transform(
            lambda x: winsorize_mad(x, n_mad=3.0)
        )
        # 标准化 (生成 {col}_z 列, 保留原值)
        panel = standardize_cross_section(panel, factor_col=col, date_col="date")

    return panel


def run_analysis(panel: pd.DataFrame):
    """
    对预处理后的面板执行:
    1. IC 分析 (所有因子)
    2. 分层回测 (Top 因子)
    3. 多因子合成
    4. 组合绩效评估
    """
    # 构造下期收益 (forward return)
    panel = panel.sort_values(["symbol", "date"]).copy()
    panel["close"] = panel.groupby("symbol")["收盘"].shift(0)
    panel["next_close"] = panel.groupby("symbol")["收盘"].shift(-1)
    panel["forward_return_1m"] = panel["next_close"] / panel["收盘"] - 1
    panel = panel.dropna(subset=["forward_return_1m"])

    # ── IC 分析 ──
    factor_cols = [c for c in panel.columns if c.endswith("_z")]
    print(f"\nIC 分析: {len(factor_cols)} 个因子")

    ic_results = {}
    for col in factor_cols:
        ic = compute_rank_ic(panel, factor_col=col, return_col="forward_return_1m", date_col="date")
        summary = compute_ic_summary(ic)
        if summary:
            ic_results[col.replace("_z", "")] = summary
            print(f"  {col.replace('_z', ''):20s}  IC_Mean={summary.get('IC_Mean',0):+.4f}  "
                  f"IC_IR={summary.get('IC_IR',0):+.4f}  Win={summary.get('IC_Win_Rate',0):.1%}")

    # 汇总表
    table = factor_summary_table(ic_results)
    table.to_csv(OUTPUT_DIR / "factor_ic_summary.csv", index=False, encoding="utf-8-sig")

    # ── 分层回测 (用 IC_IR 最高的因子) ──
    if table is not None and not table.empty and "IC_IR" in table.columns:
        best_factor = table.iloc[0]["因子"]
        best_col = f"{best_factor}_z"
        if best_col in panel.columns:
            print(f"\n分层回测: {best_factor} (IC_IR 最高)")
            assigned = assign_quantile_groups(
                panel, factor_col=best_col, n_groups=5, date_col="date"
            )
            g_rets = compute_group_returns(
                assigned, return_col="forward_return_1m", date_col="date"
            )
            ls = compute_long_short(g_rets)
            if not ls.empty:
                perf = compute_performance(ls["long_short_return"])
                print(f"  多空: Sharpe={perf['Sharpe_Ratio']:.3f}  "
                      f"AnnRet={perf['Annualized_Return']:.2%}  "
                      f"MaxDD={perf['Max_Drawdown']:.1%}")

                # ── 多因子合成 ──
                print("\n多因子合成 (等权)...")
                combined = combine_factors(panel, factor_cols=factor_cols, method="equal_weight")
                assigned_m = assign_quantile_groups(
                    combined, factor_col="composite_factor", n_groups=5, date_col="date"
                )
                g_rets_m = compute_group_returns(
                    assigned_m, return_col="forward_return_1m", date_col="date"
                )
                ls_m = compute_long_short(g_rets_m)
                if not ls_m.empty:
                    perf_m = compute_performance(ls_m["long_short_return"])
                    print(f"  复合因子: Sharpe={perf_m['Sharpe_Ratio']:.3f}  "
                          f"AnnRet={perf_m['Annualized_Return']:.2%}  "
                          f"MaxDD={perf_m['Max_Drawdown']:.1%}")

    print(f"\n报告数据已保存到 {OUTPUT_DIR}/")
    return panel, ic_results


def main():
    print("=" * 60)
    print("因子研究完整流水线")
    print("=" * 60)

    f = Fetcher()

    # 1. 获取成分股
    print("\n[1/4] 获取成分股...")
    try:
        symbols = f.get_index_members(STOCK_POOL)
    except Exception:
        print(f"  无法获取 {STOCK_POOL}, 使用预设列表")
        symbols = ["000001", "000002", "600036", "600519", "000858",
                   "002415", "300750", "601318", "600276", "000333"][:10]

    if MAX_STOCKS:
        symbols = symbols[:MAX_STOCKS]
    print(f"  股票池: {len(symbols)} 只")

    # 2. 构建因子面板
    print("\n[2/4] 构建因子面板...")
    panel = build_panel(f, symbols)
    print(f"  面板: {panel.shape[0]} 行 × {panel.shape[1]} 列")

    # 3. 预处理
    print("\n[3/4] 预处理...")
    panel = preprocess_panel(panel)

    # 4. 分析
    print("\n[4/4] 分析...")
    panel, ic_results = run_analysis(panel)

    print(f"\n{'='*60}")
    print("流水线完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
