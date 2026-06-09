"""
LightGBM ML 选股回测 — 运行脚本。

将 LightGBMAlphaEngine 训练的 ml_signal 接入交易成本感知回测引擎,
对比 ML 模型 vs 线性 alpha_signal 的扣费后绩效。

核心流程:
  1. 加载 preprocessed.parquet + split_universe_blended.parquet
  2. 训练 LightGBMAlphaEngine (36M train + 6M val + 1M test Walk-Forward)
  3. 将 ml_signal 合并到 blended panel
  4. 使用相同参数运行两次带摩擦回测:
     a. 线性 alpha_signal (Baseline, 零摩擦 Sharpe 1.176, 扣费后 1.135)
     b. ML ml_signal (对比组)
  5. 输出 ML vs Linear 对比表

用法:
  python run_ml_backtest.py

前置条件:
  - 已运行 run_split_universe.py (生成 output/split_universe_blended.parquet)
  - 已有 output/preprocessed.parquet
  - 已有 output/backtest_net_returns.csv (线性 baseline, 可选)
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ml_backtest")

from factor_research.ml_engine import LightGBMAlphaEngine, MLConfig
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
    print(f"  面板: {panel.shape[0]:,} 行 × {panel.shape[1]} 列 | "
          f"{panel['symbol'].nunique()} 只股票 | "
          f"{panel['date'].nunique()} 个截面")

    print(f"加载: {blended_path}")
    blended = pd.read_parquet(blended_path)
    print(f"  拼接面板: {blended.shape[0]:,} 行 | "
          f"{blended['date'].nunique()} 个截面 | "
          f"大盘:{(blended['universe']=='大盘').sum()} | "
          f"小盘:{(blended['universe']=='小盘').sum()}")

    return panel, blended


def build_cost_model(aum: float = 50_000_000) -> TieredCostModel:
    """构造 AUM 感知分层成本模型。"""
    large_config = UniverseCostConfig(
        commission_bps=2.5,
        stamp_duty_bps=5.0,
        transfer_fee_bps=0.1,
        base_slippage_bps=5.0,
        impact_gamma=0.5,
        impact_eta=1.0,
    )
    small_config = UniverseCostConfig(
        commission_bps=2.5,
        stamp_duty_bps=5.0,
        transfer_fee_bps=0.1,
        base_slippage_bps=15.0,
        impact_gamma=0.65,
        impact_eta=1.5,
    )
    return TieredCostModel(aum=aum, large_cap_config=large_config, small_cap_config=small_config)


def print_metrics_table(title: str, result: dict):
    """打印单组绩效指标。"""
    m = result.get("net_metrics") or result.get("gross_metrics") or {}
    to_avg = result.get("avg_turnover", 0.0)
    cost_avg = result.get("avg_cost_bps", 0.0)
    print(f"\n  {title}:")
    print(f"    年化收益: {m.get('Annualized_Return', 0)*100:.2f}%")
    print(f"    Sharpe:    {m.get('Sharpe_Ratio', 0):.4f}")
    print(f"    最大回撤:  {m.get('Max_Drawdown', 0)*100:.2f}%")
    print(f"    月胜率:    {m.get('Win_Rate', 0)*100:.1f}%")
    print(f"    平均换手:  {to_avg*100:.1f}%")
    print(f"    平均成本:  {cost_avg:.1f} bps/期")


def main():
    print("=" * 64)
    print("LightGBM ML 选股回测 — Stage 3")
    print("=" * 64)

    # ── 参数 ─────────────────────────────────────────
    AUM = 50_000_000
    TOP_QUANTILE = 0.3
    MIN_STOCKS = 5

    # ── 1. 加载数据 ───────────────────────────────
    panel, blended = load_data()

    # ── 2. 训练 LightGBM Alpha Engine ──────────────
    print(f"\n{'─' * 48}")
    print("Phase 1: LightGBM Walk-Forward 训练")
    print(f"{'─' * 48}")

    ml_config = MLConfig(
        train_months=36,
        val_months=6,
        test_months=1,
        label_method="rank",
        feature_method="rank",
        max_depth=4,
        num_leaves=24,
        learning_rate=0.02,
        n_estimators=2000,
        subsample=0.70,
        colsample_bytree=0.70,
        min_child_samples=100,
        reg_alpha=0.10,
        reg_lambda=0.10,
        early_stopping_rounds=50,
    )

    engine = LightGBMAlphaEngine(config=ml_config)
    ml_predictions = engine.run(panel)

    # 保存 ML 预测
    ml_pred_path = OUTPUT_DIR / "ml_signal_predictions.parquet"
    ml_predictions.to_parquet(ml_pred_path, index=False)
    print(f"\n  ML 预测已保存: {ml_pred_path}")

    # 保存训练报告
    report = engine.to_markdown_report()
    report_path = OUTPUT_DIR / "ml_training_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  训练报告已保存: {report_path}")

    # ── 3. 构建两组对比面板 ────────────────────────
    print(f"\n{'─' * 48}")
    print("Phase 2: 构造对比面板 & 回测")
    print(f"{'─' * 48}")

    # 线性 baseline: 使用 blended panel 自带的 alpha_signal
    linear_blended = blended.copy()

    # ML 对比: merge ml_signal 到 blended panel
    ml_blended = blended.merge(
        ml_predictions[["date", "symbol", "ml_signal"]],
        on=["date", "symbol"],
        how="left",
    )

    # 对于 ML 未覆盖的日期, 使用 0.5 (中性) 填充
    ml_blended["ml_signal"] = ml_blended["ml_signal"].fillna(0.5)
    ml_coverage = (ml_blended["ml_signal"] != 0.5).sum()
    print(f"  ML 信号覆盖率: {ml_coverage}/{len(ml_blended)} "
          f"({100*ml_coverage/len(ml_blended):.1f}%)")

    # ── 4. 成本模型 ────────────────────────────────
    cost_model = build_cost_model(aum=AUM)
    print(f"  成本模型: AUM={AUM/1e4:.0f}万")

    # ── 5. 回测: 线性 baseline ─────────────────────
    print(f"\n  [1/2] 线性 alpha_signal 回测 (Baseline)...")
    linear_result = run_backtest_with_costs(
        panel=panel,
        blended=linear_blended,
        cost_model=cost_model,
        top_quantile=TOP_QUANTILE,
        min_stocks_per_universe=MIN_STOCKS,
        alpha_col="alpha_signal",
    )

    # ── 6. 回测: ML ml_signal ───────────────────────
    print(f"\n  [2/2] ML ml_signal 回测...")
    ml_result = run_backtest_with_costs(
        panel=panel,
        blended=ml_blended,
        cost_model=cost_model,
        top_quantile=TOP_QUANTILE,
        min_stocks_per_universe=MIN_STOCKS,
        alpha_col="ml_signal",
    )

    # ── 7. 对比输出 ────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"Phase 3: ML vs Linear 扣费后绩效对比")
    print(f"{'=' * 64}")

    lm = linear_result.get("net_metrics") or {}
    mm = ml_result.get("net_metrics") or {}

    def fmt_pct(v: float | None) -> str:
        if v is None:
            return "N/A"
        return f"{v * 100:.2f}%"

    def fmt_num(v: float | None, decimals: int = 4) -> str:
        if v is None:
            return "N/A"
        return f"{v:.{decimals}f}"

    delta_sharpe = (mm.get("Sharpe_Ratio", 0) or 0) - (lm.get("Sharpe_Ratio", 0) or 0)
    delta_ret = (mm.get("Annualized_Return", 0) or 0) - (lm.get("Annualized_Return", 0) or 0)

    comparison_md = [
        "\n## ML vs Linear — 扣费后绩效对比 (AUM 5000万)",
        "",
        "| 指标 | Linear alpha_signal | LightGBM ml_signal | Δ |",
        "|------|--------------------|--------------------|---|",
        f"| 年化收益 | {fmt_pct(lm.get('Annualized_Return'))} | "
        f"{fmt_pct(mm.get('Annualized_Return'))} | "
        f"{'+' if delta_ret >= 0 else ''}{delta_ret*100:.2f}% |",
        f"| 年化波动率 | {fmt_pct(lm.get('Volatility'))} | "
        f"{fmt_pct(mm.get('Volatility'))} | — |",
        f"| **Sharpe Ratio** | **{fmt_num(lm.get('Sharpe_Ratio'))}** | "
        f"**{fmt_num(mm.get('Sharpe_Ratio'))}** | "
        f"**{'+' if delta_sharpe >= 0 else ''}{delta_sharpe:.4f}** |",
        f"| 最大回撤 | {fmt_pct(lm.get('Max_Drawdown'))} | "
        f"{fmt_pct(mm.get('Max_Drawdown'))} | — |",
        f"| Calmar Ratio | {fmt_num(lm.get('Calmar_Ratio'))} | "
        f"{fmt_num(mm.get('Calmar_Ratio'))} | — |",
        f"| 月胜率 | {fmt_pct(lm.get('Win_Rate'))} | "
        f"{fmt_pct(mm.get('Win_Rate'))} | — |",
        f"| 平均单边换手率 | {linear_result.get('avg_turnover', 0)*100:.1f}% | "
        f"{ml_result.get('avg_turnover', 0)*100:.1f}% | — |",
        f"| 平均每期成本 | {linear_result.get('avg_cost_bps', 0):.1f} bps | "
        f"{ml_result.get('avg_cost_bps', 0):.1f} bps | — |",
        f"| 回测期数 | {lm.get('Periods', 'N/A')} | "
        f"{mm.get('Periods', 'N/A')} | — |",
    ]
    md_str = "\n".join(comparison_md)
    print(md_str)

    # 保存对比表
    (OUTPUT_DIR / "ml_vs_linear_comparison.md").write_text(md_str, encoding="utf-8")

    # ── 8. 保存 ML 回测结果 ─────────────────────────
    ml_returns = pd.DataFrame({
        "date": ml_result["gross_returns"].index,
        "gross_return": ml_result["gross_returns"].values,
        "net_return": ml_result["net_returns"].values,
        "turnover": ml_result["turnovers"].values,
    })
    ml_returns.to_csv(
        OUTPUT_DIR / "ml_backtest_net_returns.csv",
        index=False, encoding="utf-8-sig",
    )
    ml_result["cost_breakdown"].to_csv(
        OUTPUT_DIR / "ml_backtest_cost_breakdown.csv",
        encoding="utf-8-sig",
    )

    # NAV
    gross_nav = ml_result["gross_nav"]
    net_nav = ml_result["net_nav"]
    common = gross_nav.index.intersection(net_nav.index)
    nav_df = pd.DataFrame({
        "date": common,
        "gross_nav": gross_nav.reindex(common).values,
        "net_nav": net_nav.reindex(common).values,
    })
    nav_df.to_csv(
        OUTPUT_DIR / "ml_backtest_nav_comparison.csv",
        index=False, encoding="utf-8-sig",
    )

    # ── 9. 总结 ────────────────────────────────────
    print(f"\n{'=' * 64}")
    print(f"ML 回测完成")
    print(f"{'=' * 64}")
    print(f"  Linear 扣费后 Sharpe: {lm.get('Sharpe_Ratio', 'N/A')}")
    print(f"  ML     扣费后 Sharpe: {mm.get('Sharpe_Ratio', 'N/A')}")
    print(f"  Sharpe 提升: {delta_sharpe:+.4f}")
    print(f"  年化收益提升: {delta_ret*100:+.2f}%")
    print(f"\n输出文件:")
    print(f"  - output/ml_signal_predictions.parquet")
    print(f"  - output/ml_training_report.md")
    print(f"  - output/ml_vs_linear_comparison.md")
    print(f"  - output/ml_backtest_net_returns.csv")
    print(f"  - output/ml_backtest_cost_breakdown.csv")
    print(f"  - output/ml_backtest_nav_comparison.csv")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
