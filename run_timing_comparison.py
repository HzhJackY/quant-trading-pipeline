"""
择时对比回测 — 有择时 vs 无择时完整绩效对比。

在 Split-Universe 扣费回测基础上, 增加择时乘数干预:
  - 正常状态: multiplier = 1.0  (100% 满仓)
  - 触发状态: multiplier = 0.3  (30% 仓位, 70% 现金)

对比维度:
  - Net Sharpe / MaxDD / Calmar / Annual Return
  - 换手率、平均成本
  - NAV 曲线 (有/无择时叠加)
  - 择时信号触发统计

用法:
  python run_timing_comparison.py

前置条件:
  - 已运行 run_split_universe.py (生成 output/split_universe_blended.parquet)
  - 已有 output/preprocessed.parquet
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_timing_comparison")

from factor_research.backtest_engine import (
    run_backtest_with_costs,
    generate_comparison_table,
)
from factor_research.transaction_cost import TieredCostModel, UniverseCostConfig
from factor_research.market_timing import (
    fetch_csi500,
    prepare_timing_multipliers,
)

OUTPUT_DIR = Path("output")


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_data():
    preprocessed_path = OUTPUT_DIR / "preprocessed.parquet"
    blended_path = OUTPUT_DIR / "split_universe_blended.parquet"

    for p in [preprocessed_path, blended_path]:
        if not p.exists():
            raise FileNotFoundError(f"未找到 {p}。请先运行前置 Pipeline。")

    print(f"加载: {preprocessed_path}")
    panel = pd.read_parquet(preprocessed_path)
    print(f"  面板: {panel.shape[0]:,} 行 | {panel['symbol'].nunique()} 只股票")
    print(f"  日期: {panel['date'].min()} ~ {panel['date'].max()}")

    print(f"\n加载: {blended_path}")
    blended = pd.read_parquet(blended_path)
    print(f"  面板: {blended.shape[0]:,} 行")
    print(f"  大盘: {(blended['universe'] == '大盘').sum()}")
    print(f"  小盘: {(blended['universe'] == '小盘').sum()}")

    return panel, blended


# ═══════════════════════════════════════════════════════════════
# 对比主流程
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("择时对比回测 — Market Timing On/Off 绩效对比")
    print("=" * 64)

    # ── 参数 ─────────────────────────────────────────
    AUM = 50_000_000          # 5000 万
    TOP_QUANTILE = 0.3        # 前 30%
    MIN_STOCKS = 5

    # ── 1. 加载数据 ───────────────────────────────
    panel, blended = load_data()
    rebalance_dates = sorted(panel["date"].unique())
    print(f"\n调仓截面数: {len(rebalance_dates)}")

    # ── 2. 初始化成本模型 ─────────────────────────
    large_config = UniverseCostConfig(
        commission_bps=2.5, stamp_duty_bps=5.0, transfer_fee_bps=0.1,
        base_slippage_bps=5.0, impact_gamma=0.5, impact_eta=1.0,
    )
    small_config = UniverseCostConfig(
        commission_bps=2.5, stamp_duty_bps=5.0, transfer_fee_bps=0.1,
        base_slippage_bps=15.0, impact_gamma=0.65, impact_eta=1.5,
    )
    cost_model = TieredCostModel(
        aum=AUM, large_cap_config=large_config, small_cap_config=small_config,
    )

    # ── 3. 准备择时乘数 ────────────────────────────
    print(f"\n{'─' * 48}")
    print("准备择时乘数...")
    print(f"{'─' * 48}")

    try:
        # 需要覆盖回测全区间 (2017-2024) + 60 日均线前视
        index_df = fetch_csi500(start_date="2016-01-01", use_cache=False)
        timing_multipliers = prepare_timing_multipliers(index_df, rebalance_dates)

        n_trig = sum(1 for v in timing_multipliers.values() if v < 1.0)
        n_total = len(timing_multipliers)
        print(f"  择时覆盖: {n_total}/{len(rebalance_dates)} 期")
        print(f"  触发减仓: {n_trig}/{n_total} 期 ({100*n_trig/max(n_total,1):.1f}%)")

        # 按年统计触发率
        trig_by_year = {}
        for dt, mult in sorted(timing_multipliers.items()):
            yr = dt.year
            if yr not in trig_by_year:
                trig_by_year[yr] = {"total": 0, "triggered": 0}
            trig_by_year[yr]["total"] += 1
            if mult < 1.0:
                trig_by_year[yr]["triggered"] += 1

        print(f"\n  年度触发率:")
        for yr in sorted(trig_by_year):
            d = trig_by_year[yr]
            print(f"    {yr}: {d['triggered']}/{d['total']} ({100*d['triggered']/d['total']:.1f}%)")

    except Exception as e:
        print(f"  ⚠ 择时数据准备失败: {e}")
        print(f"  将继续无择时回测 (仅 Baseline)")
        timing_multipliers = None

    # ── 4. 回测: 无择时 (Baseline) ────────────────
    print(f"\n{'─' * 48}")
    print("回测 A: 无择时 (满仓 Baseline)")
    print(f"{'─' * 48}")

    result_no_timing = run_backtest_with_costs(
        panel=panel, blended=blended, cost_model=cost_model,
        top_quantile=TOP_QUANTILE, min_stocks_per_universe=MIN_STOCKS,
        timing_multipliers=None,
    )

    # ── 5. 回测: 有择时 ────────────────────────────
    if timing_multipliers is not None:
        print(f"\n{'─' * 48}")
        print("回测 B: 有择时 (MA20/60 死叉 + 波动率 80% 分位)")
        print(f"{'─' * 48}")

        result_with_timing = run_backtest_with_costs(
            panel=panel, blended=blended, cost_model=cost_model,
            top_quantile=TOP_QUANTILE, min_stocks_per_universe=MIN_STOCKS,
            timing_multipliers=timing_multipliers,
        )
    else:
        result_with_timing = None
        print("  ⚠ 无择时数据, 跳过")

    # ── 6. 绩效对比 ───────────────────────────────
    print(f"\n{'=' * 64}")
    print("绩效对比: 有择时 vs 无择时")
    print(f"{'=' * 64}")

    nm_no = result_no_timing.get("net_metrics") or {}
    gm_no = result_no_timing.get("gross_metrics") or {}

    if result_with_timing is not None:
        nm_with = result_with_timing.get("net_metrics") or {}
        gm_with = result_with_timing.get("gross_metrics") or {}
    else:
        nm_with, gm_with = {}, {}

    # 打印对比表
    metrics_list = [
        ("Net Sharpe",         "sharpe",       "{:.2f}"),
        ("Annual Return (%)",  "ann_return",   "{:.2f}"),
        ("Annual Vol (%)",     "ann_vol",      "{:.2f}"),
        ("Max Drawdown (%)",   "max_dd",       "{:.2f}"),
        ("Calmar Ratio",       "calmar",       "{:.2f}"),
        ("Win Rate (%)",       "win_rate",     "{:.2f}"),
        ("Avg Monthly Ret (%)","avg_ret",      "{:.3f}"),
    ]

    # Try to find the keys in net_metrics
    key_map = {
        "sharpe": "Sharpe_Ratio",
        "ann_return": "Annualized_Return",
        "ann_vol": "Annualized_Volatility",
        "max_dd": "Max_Drawdown",
        "calmar": "Calmar_Ratio",
        "win_rate": "Win_Rate",
        "avg_ret": "Average_Return",
    }

    print(f"\n{'指标':<25} {'无择时':>12} {'有择时':>12} {'变化':>12}")
    print(f"{'─' * 25} {'─' * 12} {'─' * 12} {'─' * 12}")

    comparison_data = {}
    for label, short_key, fmt in metrics_list:
        actual_key = key_map.get(short_key, short_key)
        v_no = nm_no.get(actual_key, np.nan)
        v_with = nm_with.get(actual_key, np.nan) if nm_with else np.nan

        # Convert percentage-style values
        if short_key in ("ann_return", "ann_vol", "max_dd", "win_rate", "avg_ret"):
            v_no = v_no * 100 if pd.notna(v_no) else np.nan
            v_with = v_with * 100 if pd.notna(v_with) else np.nan

        if pd.notna(v_no) and pd.notna(v_with):
            delta = v_with - v_no
            delta_str = f"{delta:+.2f}"
        else:
            delta_str = "N/A"

        v_no_str = fmt.format(v_no) if pd.notna(v_no) else "N/A"
        v_with_str = fmt.format(v_with) if pd.notna(v_with) else "N/A"

        print(f"{label:<25} {v_no_str:>12} {v_with_str:>12} {delta_str:>12}")
        comparison_data[label] = {"no_timing": v_no, "with_timing": v_with, "delta": delta_str}

    # 换手率 + 成本
    to_no = result_no_timing.get("avg_turnover", 0) or 0
    cost_no = result_no_timing.get("avg_cost_bps", 0) or 0

    if result_with_timing is not None:
        to_with = result_with_timing.get("avg_turnover", 0) or 0
        cost_with = result_with_timing.get("avg_cost_bps", 0) or 0
    else:
        to_with, cost_with = np.nan, np.nan

    to_no_pct = to_no * 100
    to_with_pct = to_with * 100 if pd.notna(to_with) else np.nan

    print(f"{'Avg Turnover (%)':<25} {to_no_pct:>12.1f} {to_with_pct:>12.1f} {to_with_pct - to_no_pct:>+12.1f}" if pd.notna(to_with) else f"{'Avg Turnover (%)':<25} {to_no_pct:>12.1f} {'N/A':>12} {'N/A':>12}")
    print(f"{'Avg Monthly Cost (bps)':<25} {cost_no:>12.1f} {cost_with:>12.1f} {cost_with - cost_no:>+12.1f}" if pd.notna(cost_with) else f"{'Avg Monthly Cost (bps)':<25} {cost_no:>12.1f} {'N/A':>12} {'N/A':>12}")

    # ── 7. NAV 曲线图 ─────────────────────────────
    print(f"\n{'─' * 48}")
    print("生成 NAV 对比图...")
    print(f"{'─' * 48}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    net_nav_no = result_no_timing["net_nav"]
    gross_nav_no = result_no_timing["gross_nav"]

    # 上: 有/无择时 NAV 对比
    ax1 = axes[0]
    ax1.plot(net_nav_no.index, net_nav_no.values, linewidth=1.5, label="No Timing (Net)", color="steelblue")
    ax1.plot(gross_nav_no.index, gross_nav_no.values, linewidth=0.8, alpha=0.4, label="No Timing (Gross)", color="steelblue", linestyle=":")

    if result_with_timing is not None:
        net_nav_with = result_with_timing["net_nav"]
        gross_nav_with = result_with_timing["gross_nav"]
        ax1.plot(net_nav_with.index, net_nav_with.values, linewidth=1.5, label="With Timing (Net)", color="darkorange")
        ax1.plot(gross_nav_with.index, gross_nav_with.values, linewidth=0.8, alpha=0.4, label="With Timing (Gross)", color="darkorange", linestyle=":")

    ax1.set_title("NAV Comparison: Market Timing On vs Off")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("NAV")

    # 下: 择时乘数时间序列
    ax2 = axes[1]
    if timing_multipliers is not None:
        mult_dates = sorted(timing_multipliers.keys())
        mult_vals = [timing_multipliers[d] for d in mult_dates]
        ax2.fill_between(mult_dates, 0.3, 1.0, step="pre", color="green", alpha=0.08)
        ax2.step(mult_dates, mult_vals, where="post", color="green", linewidth=1.5)
        ax2.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label="Trigger (0.3)")
        ax2.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Normal (1.0)")
        ax2.set_ylim(0, 1.3)
        ax2.set_ylabel("Position Multiplier")
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        n_trig_dates = sum(1 for v in mult_vals if v < 1.0)
        ax2.set_title(f"Market Timing Multiplier ({n_trig_dates}/{len(mult_vals)} periods triggered, {100*n_trig_dates/len(mult_vals):.1f}%)")
    else:
        ax2.text(0.5, 0.5, "No timing data available", ha="center", va="center", transform=ax2.transAxes, fontsize=14)

    ax2.set_xlabel("Date")
    fig.autofmt_xdate()
    plt.tight_layout()

    fig_path = OUTPUT_DIR / "timing_comparison_nav.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  图表已保存: {fig_path}")
    plt.close(fig)

    # ── 8. 保存数据文件 ────────────────────────────
    print(f"\n{'─' * 48}")
    print("保存数据文件...")
    print(f"{'─' * 48}")

    # 8a. NAV 对比 CSV
    common_dates = net_nav_no.index
    nav_comparison = pd.DataFrame({
        "date": common_dates,
        "nav_no_timing": net_nav_no.reindex(common_dates).values,
        "nav_with_timing": (
            result_with_timing["net_nav"].reindex(common_dates).values
            if result_with_timing is not None
            else np.nan
        ),
    })

    # 添加择时乘数列 (如果日期匹配)
    if timing_multipliers is not None:
        nav_comparison["timing_multiplier"] = nav_comparison["date"].map(
            lambda d: timing_multipliers.get(pd.Timestamp(d), np.nan)
        )

    nav_path = OUTPUT_DIR / "timing_nav_comparison.csv"
    nav_comparison.to_csv(nav_path, index=False, encoding="utf-8-sig")
    print(f"  NAV 对比: {nav_path} ({len(nav_comparison)} 行)")

    # 8b. 逐月收益对比
    ret_no = result_no_timing["net_returns"]
    returns_comparison = pd.DataFrame({
        "date": ret_no.index,
        "net_ret_no_timing": ret_no.values,
        "net_ret_with_timing": (
            result_with_timing["net_returns"].reindex(ret_no.index).values
            if result_with_timing is not None
            else np.nan
        ),
    })
    if timing_multipliers is not None:
        returns_comparison["timing_multiplier"] = returns_comparison["date"].map(
            lambda d: timing_multipliers.get(pd.Timestamp(d), np.nan)
        )

    ret_path = OUTPUT_DIR / "timing_returns_comparison.csv"
    returns_comparison.to_csv(ret_path, index=False, encoding="utf-8-sig")
    print(f"  收益对比: {ret_path} ({len(returns_comparison)} 行)")

    # 8c. 择时信号明细
    if timing_multipliers is not None:
        signal_df = pd.DataFrame(
            [(d, timing_multipliers[d]) for d in sorted(timing_multipliers.keys())],
            columns=["date", "multiplier"],
        )
        signal_df["is_triggered"] = signal_df["multiplier"] < 1.0
        signal_path = OUTPUT_DIR / "timing_signal_log.csv"
        signal_df.to_csv(signal_path, index=False, encoding="utf-8-sig")
        print(f"  择时信号: {signal_path} ({len(signal_df)} 行")

    # ── 9. 总结 ───────────────────────────────────
    print(f"\n{'=' * 64}")
    print("择时对比回测完成")
    print(f"{'=' * 64}")

    sharpe_no = nm_no.get("Sharpe_Ratio", "N/A")
    if nm_with:
        sharpe_with = nm_with.get("Sharpe_Ratio", "N/A")
        print(f"  Net Sharpe:       {sharpe_no} → {sharpe_with}")
    else:
        print(f"  Net Sharpe:       {sharpe_no}")

    dd_no = nm_no.get("Max_Drawdown", "N/A")
    if nm_with:
        dd_with = nm_with.get("Max_Drawdown", "N/A")
        print(f"  Max Drawdown:     {dd_no} → {dd_with}")
    else:
        print(f"  Max Drawdown:     {dd_no}")

    print(f"\n输出文件:")
    print(f"  - {fig_path}")
    print(f"  - {nav_path}")
    print(f"  - {ret_path}")
    print(f"  - output/timing_signal_log.csv")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
