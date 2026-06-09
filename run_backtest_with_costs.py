"""
交易成本感知回测 — 运行脚本。

将 transaction_cost.py 的 TieredCostModel 接入 backtest_engine.py,
执行持仓级成本扣除, 产出机构级 Net Sharpe Baseline。

核心流程:
  1. 加载 preprocessed.parquet + split_universe_blended.parquet
  2. 初始化 TieredCostModel (AUM=5000万)
  3. 执行持仓级回测 (含价格漂移修正 + 分域成本扣除)
  4. 输出零摩擦 vs 扣费对比表

用法:
  python run_backtest_with_costs.py

前置条件:
  - 已运行 run_split_universe.py (生成 output/split_universe_blended.parquet)
  - 已有 output/preprocessed.parquet (含 成交额, Vol_20D, 收盘 等列)
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_backtest_with_costs")

from factor_research.backtest_engine import (
    run_backtest_with_costs,
    generate_comparison_table,
    compute_nav,
    compute_performance,
)
from factor_research.transaction_cost import TieredCostModel, UniverseCostConfig

OUTPUT_DIR = Path("output")


def load_data():
    """加载数据。"""
    preprocessed_path = OUTPUT_DIR / "preprocessed.parquet"
    blended_path = OUTPUT_DIR / "split_universe_blended.parquet"

    for p in [preprocessed_path, blended_path]:
        if not p.exists():
            raise FileNotFoundError(f"未找到 {p}。请先运行前置 Pipeline。")

    print(f"加载: {preprocessed_path}")
    panel = pd.read_parquet(preprocessed_path)
    print(f"  面板: {panel.shape[0]:,} 行 × {panel.shape[1]} 列")
    print(f"  股票数: {panel['symbol'].nunique()}")
    print(f"  日期范围: {panel['date'].min()} ~ {panel['date'].max()}")

    print(f"\n加载: {blended_path}")
    blended = pd.read_parquet(blended_path)
    print(f"  拼接面板: {blended.shape[0]:,} 行 × {blended.shape[1]} 列")
    print(f"  大盘行数: {(blended['universe'] == '大盘').sum()}")
    print(f"  小盘行数: {(blended['universe'] == '小盘').sum()}")

    return panel, blended


def print_markdown_table(result: dict, aum: float, title: str = ""):
    """打印对比表到控制台。"""
    md = generate_comparison_table(result, aum=aum)
    if title:
        print(f"\n{'=' * 64}")
        print(f"  {title}")
        print(f"{'=' * 64}")
    print(md)


def main():
    print("=" * 64)
    print("交易成本感知回测 — 机构级 Net Sharpe Baseline")
    print("=" * 64)

    # ── 参数 ─────────────────────────────────────────
    AUM = 50_000_000          # 5000 万
    TOP_QUANTILE = 0.3        # 前 30%
    MIN_STOCKS = 5            # 最少持仓数

    # ── 1. 加载数据 ───────────────────────────────
    panel, blended = load_data()

    # ── 2. 初始化成本模型 ─────────────────────────
    # 大盘: 标准配置 (平方根冲击 γ=0.5, 低滑点)
    large_config = UniverseCostConfig(
        commission_bps=2.5,
        stamp_duty_bps=5.0,
        transfer_fee_bps=0.1,
        base_slippage_bps=5.0,      # 大盘 bid-ask 窄
        impact_gamma=0.5,            # 平方根法则
        impact_eta=1.0,
    )
    # 小盘: 高冲击配置 (更陡 γ=0.65, 更高滑点)
    small_config = UniverseCostConfig(
        commission_bps=2.5,
        stamp_duty_bps=5.0,
        transfer_fee_bps=0.1,
        base_slippage_bps=15.0,      # 小盘 bid-ask 宽
        impact_gamma=0.65,           # 流动性折价, 更陡
        impact_eta=1.5,              # 冲击放大
    )

    cost_model = TieredCostModel(
        aum=AUM,
        large_cap_config=large_config,
        small_cap_config=small_config,
    )

    print(f"\n成本模型配置:")
    print(f"  AUM: {AUM/1e4:.0f} 万")
    print(f"  大盘: commission={large_config.commission_bps}bps "
          f"slip={large_config.base_slippage_bps}bps "
          f"γ={large_config.impact_gamma}")
    print(f"  小盘: commission={small_config.commission_bps}bps "
          f"slip={small_config.base_slippage_bps}bps "
          f"γ={small_config.impact_gamma}")

    # ── 3. 执行回测 ───────────────────────────────
    print(f"\n{'─' * 48}")
    print(f"执行持仓级成本感知回测...")
    print(f"{'─' * 48}")

    result = run_backtest_with_costs(
        panel=panel,
        blended=blended,
        cost_model=cost_model,
        top_quantile=TOP_QUANTILE,
        min_stocks_per_universe=MIN_STOCKS,
    )

    # ── 4. 输出对比表 ─────────────────────────────
    # 4a. 表格
    print_markdown_table(result, aum=AUM, title="Split-Universe 扣费回测")

    # 4b. 成本分解详情
    cost_df = result["cost_breakdown"]
    if not cost_df.empty and "total_cost_bps" in cost_df.columns:
        valid_costs = cost_df.dropna(subset=["total_cost_bps"])
        if not valid_costs.empty:
            print(f"\n{'─' * 48}")
            print("成本分解统计 (bps):")
            print(f"{'─' * 48}")
            for col, label in [
                ("total_cost_bps", "总成本"),
                ("large_cost_bps", "大盘成本"),
                ("small_cost_bps", "小盘成本"),
            ]:
                if col in valid_costs.columns:
                    s = valid_costs[col]
                    print(f"  {label:>10}: 均值={s.mean():.1f}  "
                          f"中位数={s.median():.1f}  "
                          f"最大={s.max():.1f}")

    # 4c. 换手率统计
    to_series = result["turnovers"].dropna()
    if len(to_series) > 0:
        print(f"\n换手率统计:")
        print(f"  平均单边换手率: {to_series.mean()*100:.1f}%")
        print(f"  中位数: {to_series.median()*100:.1f}%")
        print(f"  最大值: {to_series.max()*100:.1f}%")

    # ── 5. 保存结果 ───────────────────────────────
    print(f"\n{'─' * 48}")
    print("保存结果...")
    print(f"{'─' * 48}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 收益序列
    returns_df = pd.DataFrame({
        "date": result["gross_returns"].index,
        "gross_return": result["gross_returns"].values,
        "net_return": result["net_returns"].values,
        "turnover": result["turnovers"].values,
    })
    returns_df.to_csv(
        OUTPUT_DIR / "backtest_net_returns.csv",
        index=False, encoding="utf-8-sig",
    )
    print(f"  收益序列: output/backtest_net_returns.csv ({len(returns_df)} 行)")

    # 成本明细
    result["cost_breakdown"].to_csv(
        OUTPUT_DIR / "backtest_cost_breakdown.csv",
        encoding="utf-8-sig",
    )
    print(f"  成本明细: output/backtest_cost_breakdown.csv")

    # NAV (已对齐, 共享同一 date index)
    gross_nav = result["gross_nav"]
    net_nav = result["net_nav"]
    common_dates = gross_nav.index.intersection(net_nav.index)
    nav_df = pd.DataFrame({
        "date": common_dates,
        "gross_nav": gross_nav.reindex(common_dates).values,
        "net_nav": net_nav.reindex(common_dates).values,
    })
    nav_df.to_csv(
        OUTPUT_DIR / "backtest_nav_comparison.csv",
        index=False, encoding="utf-8-sig",
    )
    print(f"  NAV 对比: output/backtest_nav_comparison.csv ({len(nav_df)} 期)")

    # ── 6. 总结 ───────────────────────────────────
    gm = result.get("gross_metrics") or {}
    nm = result.get("net_metrics") or {}

    print(f"\n{'=' * 64}")
    print(f"回测完成")
    print(f"{'=' * 64}")
    print(f"  AUM: {AUM/1e4:.0f} 万")
    print(f"  零摩擦 Sharpe: {gm.get('Sharpe_Ratio', 'N/A')}")
    print(f"  扣费后 Sharpe: {nm.get('Sharpe_Ratio', 'N/A')}")
    print(f"  Sharpe 衰减: {(gm.get('Sharpe_Ratio', 0) or 0) - (nm.get('Sharpe_Ratio', 0) or 0):.4f}")
    print(f"  平均每期成本: {result.get('avg_cost_bps', 0):.1f} bps")
    print(f"  平均单边换手率: {result.get('avg_turnover', 0)*100:.1f}%")
    print(f"\n输出文件:")
    print(f"  - output/backtest_net_returns.csv")
    print(f"  - output/backtest_cost_breakdown.csv")
    print(f"  - output/backtest_nav_comparison.csv")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
