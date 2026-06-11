"""
Split-Universe 双模型系统 — 运行脚本。

从现有 panel.parquet 出发, 执行市值分层双模型分析,
并与原全市场统一模型进行对比回测。

用法:
  python run_split_universe.py
"""

import logging
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import ConstantInputWarning
warnings.filterwarnings("ignore", category=ConstantInputWarning)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

from factor_research.split_universe import (
    SplitUniverseModel, run_split_universe_analysis,
)
from factor_research.backtest_engine import (
    combine_factors, compute_performance,
)
from factor_research.group_backtest import (
    assign_quantile_groups, compute_group_returns, compute_long_short,
)
from factor_research.ic_analysis import compute_rank_ic, compute_ic_summary


# ─── 因子列表 (16 个) ────────────────────────────────────
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

OUTPUT_DIR = Path("output")


def load_panel() -> pd.DataFrame:
    """
    加载现有数据。

    优先使用 preprocessed.parquet (含 _neutral_z 标准化因子列 +
    换手率/成交额用于市值估计), 回退使用 panel.parquet。
    """
    preprocessed_path = OUTPUT_DIR / "preprocessed.parquet"
    panel_path = OUTPUT_DIR / "panel.parquet"

    if preprocessed_path.exists():
        print(f"  加载 preprocessed: {preprocessed_path}")
        df = pd.read_parquet(preprocessed_path)
    elif panel_path.exists():
        print(f"  加载 panel: {panel_path}")
        df = pd.read_parquet(panel_path)
    else:
        raise FileNotFoundError(
            "未找到 preprocessed.parquet 或 panel.parquet。"
            "请先运行 run_factor_research.py 的 Stage 1-3。"
        )

    print(f"  数据: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    print(f"  股票数: {df['symbol'].nunique()}")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    return df


def run_baseline_backtest(panel: pd.DataFrame) -> dict:
    """运行全市场统一模型作为对比基准。"""
    print("\n" + "=" * 60)
    print("Baseline: 全市场统一模型 (单域)")
    print("=" * 60)

    # 标准化因子列 (preprocessed panel 已有 _neutral_z)
    factor_z_cols = [c for c in panel.columns if c.endswith("_neutral_z")]
    if not factor_z_cols:
        factor_z_cols = [c for c in panel.columns if c.endswith("_z")]

    if not factor_z_cols:
        print("  [WARN] 未找到标准化因子列, 跳过 baseline")
        return {}

    # 计算 forward return
    panel = panel.sort_values(["symbol", "date"]).copy()
    if "forward_return_1m" not in panel.columns:
        print("  计算 forward_return_1m...")
        close_col = "收盘" if "收盘" in panel.columns else "close"
        panel["next_close"] = panel.groupby("symbol")[close_col].shift(-1)
        panel["forward_return_1m"] = (
            panel["next_close"] - panel[close_col].astype(float)
        ) / panel[close_col].astype(float)

    # IC
    print(f"  IC 分析: {len(factor_z_cols)} 个因子")
    ic_results = {}
    for col in factor_z_cols:
        ic = compute_rank_ic(panel, factor_col=col, return_col="forward_return_1m", date_col="date")
        summary = compute_ic_summary(ic)
        if summary:
            name = col.replace("_neutral_z", "").replace("_z", "")
            ic_results[name] = summary
            print(f"    {name:20s} IC_IR={summary.get('IC_IR', 0):+.4f}")

    # 复合因子 — Gram-Schmidt 正交化 + 滚动 IC_IR 加权
    print(f"\n  多因子合成 Gram-Schmidt 正交化 (滚动24月IC_IR)")
    combined = combine_factors(
        panel, factor_cols=factor_z_cols, method="ic_weighted",
        return_col="forward_return_1m", date_col="date",
        flip_sign=True, orthogonalize=True, rolling_window=24,
    )

    # 分层回测
    assigned = assign_quantile_groups(
        combined, factor_col="composite_factor", n_groups=5, date_col="date"
    )
    g_rets = compute_group_returns(assigned, return_col="forward_return_1m", date_col="date")
    ls = compute_long_short(g_rets)
    if ls.empty:
        print("  [WARN] 多空组合为空")
        return {}

    perf = compute_performance(ls["long_short_return"])

    print(f"\n  Baseline 结果:")
    print(f"    Sharpe:     {perf['Sharpe_Ratio']:.4f}")
    print(f"    年化收益:    {perf['Annualized_Return']:.2%}")
    print(f"    年化波动:    {perf['Volatility']:.2%}")
    print(f"    最大回撤:    {perf['Max_Drawdown']:.1%}")
    print(f"    Calmar:     {perf['Calmar_Ratio']:.4f}")
    print(f"    胜率:       {perf['Win_Rate']:.1%}")

    return perf


def run_split_universe_backtest(
    blended_panel: pd.DataFrame,
    panel_all: pd.DataFrame,
) -> dict:
    """对 Split-Universe 拼接后的 alpha_signal 做分层回测。"""
    print("\n" + "=" * 60)
    print("Split-Universe: 双模型拼接信号回测")
    print("=" * 60)

    if "alpha_signal" not in blended_panel.columns:
        print("  [WARN] blended_panel 缺少 alpha_signal 列")
        return {}

    # 需要 forward_return_1m 来做回测
    if "forward_return_1m" not in blended_panel.columns:
        # 从全市场 panel 合并收益
        close_col = "收盘" if "收盘" in panel_all.columns else "close"
        fwd = panel_all[["date", "symbol"]].copy()
        panel_all_sorted = panel_all.sort_values(["symbol", "date"])
        panel_all_sorted["forward_return_1m"] = (
            panel_all_sorted.groupby("symbol")[close_col]
            .transform(lambda x: x.shift(-1) / x - 1)
        )
        fwd = panel_all_sorted[["date", "symbol", "forward_return_1m"]].dropna()
        blended_panel = blended_panel.merge(
            fwd, on=["date", "symbol"], how="left"
        )

    blend_with_ret = blended_panel.dropna(subset=["forward_return_1m"])
    print(f"  有效样本: {len(blend_with_ret):,} 行")

    # IC of alpha_signal
    ic = compute_rank_ic(
        blend_with_ret, factor_col="alpha_signal",
        return_col="forward_return_1m", date_col="date",
    )
    summary = compute_ic_summary(ic)
    if summary:
        print(f"  alpha_signal IC_IR = {summary.get('IC_IR', 0):+.4f}  "
              f"IC_Mean = {summary.get('IC_Mean', 0):+.4f}")

    # 分层回测
    assigned = assign_quantile_groups(
        blend_with_ret, factor_col="alpha_signal", n_groups=5, date_col="date",
    )
    g_rets = compute_group_returns(
        assigned, return_col="forward_return_1m", date_col="date",
    )
    ls = compute_long_short(g_rets)
    if ls.empty:
        print("  [WARN] 多空组合为空")
        return {}

    perf = compute_performance(ls["long_short_return"])

    print(f"\n  Split-Universe 结果:")
    print(f"    Sharpe:     {perf['Sharpe_Ratio']:.4f}")
    print(f"    年化收益:    {perf['Annualized_Return']:.2%}")
    print(f"    年化波动:    {perf['Volatility']:.2%}")
    print(f"    最大回撤:    {perf['Max_Drawdown']:.1%}")
    print(f"    Calmar:     {perf['Calmar_Ratio']:.4f}")
    print(f"    胜率:       {perf['Win_Rate']:.1%}")

    # ── 分域回测: 分别看大盘和小盘池内 alpha 表现 ──
    for uni_name in ["大盘", "小盘"]:
        uni_data = blend_with_ret[blend_with_ret["universe"] == uni_name]
        if len(uni_data) < 100:
            continue
        uni_ic = compute_rank_ic(
            uni_data, factor_col="alpha_signal",
            return_col="forward_return_1m", date_col="date",
        )
        uni_summary = compute_ic_summary(uni_ic)

        uni_assigned = assign_quantile_groups(
            uni_data, factor_col="alpha_signal", n_groups=5, date_col="date",
        )
        uni_g_rets = compute_group_returns(
            uni_assigned, return_col="forward_return_1m", date_col="date",
        )
        uni_ls = compute_long_short(uni_g_rets)
        if not uni_ls.empty:
            uni_perf = compute_performance(uni_ls["long_short_return"])
            print(f"\n  [{uni_name}池内回测]")
            print(f"    alpha IC_IR: {uni_summary.get('IC_IR', 0):+.4f}")
            print(f"    Sharpe:      {uni_perf['Sharpe_Ratio']:.4f}")
            print(f"    年化收益:     {uni_perf['Annualized_Return']:.2%}")
            print(f"    最大回撤:     {uni_perf['Max_Drawdown']:.1%}")

    return perf


def main():
    print("=" * 60)
    print("Split-Universe 双模型协同系统")
    print("=" * 60)

    # 1. 加载数据
    panel = load_panel()

    # 只保留存在的因子
    available_factors = [f for f in FACTOR_COLS if f in panel.columns]
    if len(available_factors) < len(FACTOR_COLS):
        missing = set(FACTOR_COLS) - set(available_factors)
        print(f"  [WARN] 缺失因子: {missing}")
    print(f"  可用因子: {len(available_factors)}/{len(FACTOR_COLS)}")

    # 2. Baseline 回测 (全市场统一模型)
    baseline_perf = run_baseline_backtest(panel)

    # 3. Split-Universe 分析
    print("\n" + "=" * 60)
    print("执行 Split-Universe 分析...")
    print("=" * 60)

    result = run_split_universe_analysis(
        panel=panel,
        factor_cols=available_factors,
        percentile=0.5,
        output_dir="output",
    )

    # 4. Split-Universe 回测
    split_perf = run_split_universe_backtest(
        blended_panel=result.blended_panel,
        panel_all=panel,
    )

    # 5. 对比总结
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)

    if baseline_perf and split_perf:
        print(f"\n  {'指标':<18} {'Baseline(统一)':>15} {'Split-Universe':>15} {'变化':>10}")
        print("  " + "-" * 60)
        for key, label in [
            ("Sharpe_Ratio", "Sharpe"),
            ("Annualized_Return", "年化收益"),
            ("Max_Drawdown", "最大回撤"),
            ("Calmar_Ratio", "Calmar"),
            ("Win_Rate", "胜率"),
        ]:
            b_val = baseline_perf.get(key, 0)
            s_val = split_perf.get(key, 0)
            delta = s_val - b_val
            if key in ("Annualized_Return", "Max_Drawdown", "Win_Rate"):
                print(f"  {label:<18} {b_val:>14.2%}  {s_val:>14.2%}  {delta:>+9.2%}")
            else:
                print(f"  {label:<18} {b_val:>14.4f}  {s_val:>14.4f}  {delta:>+9.4f}")

    # 6. 因子归属总结
    if result.comparison_table is not None:
        print(f"\n  因子归属分布:")
        print(result.comparison_table[["因子", "归属", "大盘优势"]].to_string(index=False))

    print(f"\n所有输出文件在 output/ 目录:")
    print(f"  - split_universe_ic_comparison.csv  (三域 IC_IR 对比)")
    print(f"  - split_universe_blended.parquet    (拼接 Alpha 信号)")
    print(f"  - split_universe_ic_curve.png       (累积 IC 净值图)")
    print("=" * 60)


if __name__ == "__main__":
    main()
