# ═══════════════════════════════════════════════════════════════
# ARCHIVED: LambdaRank 回测 runner
# LightGBMAlphaEngineV2 已被 V7 取代, 此脚本不再活跃
# 保留作为实验记录 — 不建议用于新工作
# ═══════════════════════════════════════════════════════════════
"""
LambdaRank 排序学习回测 — 运行脚本 (Stage 4).

将 LightGBMAlphaEngineV2 (LambdaRank) 训练的 ml_rank_signal 接入
交易成本感知回测, 对比:
  - V0: 线性 alpha_signal (Baseline)
  - V1: L2 LightGBM (原版)
  - V4: LambdaRank + 3M label + Delta特征 + Categorical (新版)

核心改进:
  - Objective: lambdarank → 直接优化截面排序
  - Label: 3M forward return → 内生降换手
  - Feature: +Δ1M/Δ3M + board_cat + mcap_bin
  - Gap: train↔val 之间 3 个月盲区 → 杜绝标签泄漏

用法:
  python run_ml_lambdarank.py
  python run_ml_lambdarank.py --skip-baselines  # 仅运行 V4
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ml_lambdarank")

from factor_research.ml_engine_v2 import LightGBMAlphaEngineV2, MLConfigV2
from factor_research.ml_engine import LightGBMAlphaEngine, MLConfig
from factor_research.backtest_engine import (
    run_backtest_with_costs,
    compute_nav,
    compute_performance,
)
from factor_research.transaction_cost import TieredCostModel, UniverseCostConfig

OUTPUT_DIR = Path("output")


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════


def load_data():
    preprocessed_path = OUTPUT_DIR / "preprocessed.parquet"
    blended_path = OUTPUT_DIR / "split_universe_blended.parquet"

    for p in [preprocessed_path, blended_path]:
        if not p.exists():
            raise FileNotFoundError(f"未找到 {p}。请先运行前置 Pipeline。")

    panel = pd.read_parquet(preprocessed_path)
    blended = pd.read_parquet(blended_path)
    print(f"加载: panel {panel.shape[0]:,} 行 × {panel.shape[1]} 列 | "
          f"{panel['date'].nunique()} 截面")
    print(f"加载: blended {blended.shape[0]:,} 行")
    return panel, blended


def build_cost_model(aum: float = 50_000_000) -> TieredCostModel:
    return TieredCostModel(
        aum=aum,
        large_cap_config=UniverseCostConfig(
            commission_bps=2.5, stamp_duty_bps=5.0, transfer_fee_bps=0.1,
            base_slippage_bps=5.0, impact_gamma=0.5, impact_eta=1.0,
        ),
        small_cap_config=UniverseCostConfig(
            commission_bps=2.5, stamp_duty_bps=5.0, transfer_fee_bps=0.1,
            base_slippage_bps=15.0, impact_gamma=0.65, impact_eta=1.5,
        ),
    )


def fmt_pct(v, decimals=2):
    if v is None or np.isnan(v):
        return "N/A"
    return f"{v * 100:.{decimals}f}%"

def fmt_num(v, decimals=4):
    if v is None or np.isnan(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def run_backtest_for_signal(
    panel: pd.DataFrame,
    blended: pd.DataFrame,
    cost_model: TieredCostModel,
    signal_df: pd.DataFrame | None,
    alpha_col: str,
    label: str,
    top_quantile: float = 0.3,
    min_stocks: int = 5,
) -> dict:
    """对单个信号执行回测, 返回结果 dict。"""
    t0 = time.perf_counter()

    if signal_df is not None:
        bt_blended = blended.merge(
            signal_df[["date", "symbol", alpha_col]],
            on=["date", "symbol"], how="left",
        )
        bt_blended[alpha_col] = bt_blended[alpha_col].fillna(0.5)
    else:
        bt_blended = blended

    result = run_backtest_with_costs(
        panel=panel,
        blended=bt_blended,
        cost_model=cost_model,
        top_quantile=top_quantile,
        min_stocks_per_universe=min_stocks,
        alpha_col=alpha_col,
    )

    elapsed = time.perf_counter() - t0
    nm = result.get("net_metrics") or {}
    print(f"  [{label}] {elapsed:.0f}s | "
          f"Sharpe={nm.get('Sharpe_Ratio', 0):.4f} | "
          f"TO={result.get('avg_turnover', 0)*100:.1f}% | "
          f"Cost={result.get('avg_cost_bps', 0):.1f}bps")
    return {"result": result, "wall_time_sec": elapsed}


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="LambdaRank 回测")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="跳过 V0, V1 (仅运行 V4)")
    parser.add_argument("--aum", type=float, default=50_000_000)
    parser.add_argument("--top", type=float, default=0.3)
    args = parser.parse_args()

    print("=" * 64)
    print("LambdaRank 排序学习回测 — Stage 4")
    print("=" * 64)

    panel, blended = load_data()
    cost_model = build_cost_model(aum=args.aum)

    results: dict[str, dict] = {}

    # ── V0: Linear Baseline ──
    if not args.skip_baselines:
        print(f"\n[V0] 线性 alpha_signal (Baseline)")
        results["V0_Linear"] = run_backtest_for_signal(
            panel, blended, cost_model, None, "alpha_signal", "V0",
            top_quantile=args.top,
        )

    # ── V1: L2 LightGBM (cached or re-run) ──
    if not args.skip_baselines:
        cached_pred = OUTPUT_DIR / "ml_signal_predictions.parquet"
        if cached_pred.exists():
            print(f"\n[V1] L2 LightGBM (cached)")
            preds = pd.read_parquet(cached_pred)
            results["V1_L2"] = run_backtest_for_signal(
                panel, blended, cost_model, preds, "ml_signal", "V1",
                top_quantile=args.top,
            )
        else:
            print(f"\n[V1] L2 LightGBM (training...)")
            engine_v1 = LightGBMAlphaEngine(config=MLConfig(seeds=[42]))
            preds = engine_v1.run(panel)
            results["V1_L2"] = run_backtest_for_signal(
                panel, blended, cost_model, preds, "ml_signal", "V1",
                top_quantile=args.top,
            )

    # ── V4: LambdaRank ──
    print(f"\n{'─' * 48}")
    print("[V4] LambdaRank + 3M Label + Delta + Categorical")
    print(f"{'─' * 48}")

    config_v4 = MLConfigV2(
        train_months=36,
        val_months=6,
        test_months=1,
        label_horizon=3,           # 3M forward return
        objective="lambdarank",
        metric="ndcg",
        eval_at=(10, 30),
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
        use_delta_features=True,
        use_categorical_features=True,
    )

    engine_v4 = LightGBMAlphaEngineV2(config=config_v4)
    preds_v4 = engine_v4.run(panel, blended=blended)

    # 保存预测
    pred_path = OUTPUT_DIR / "ml_lambdarank_predictions.parquet"
    preds_v4.to_parquet(pred_path, index=False)
    print(f"  LambdaRank 预测保存: {pred_path}")

    # 保存训练报告
    report = engine_v4.to_markdown_report()
    (OUTPUT_DIR / "ml_lambdarank_report.md").write_text(report, encoding="utf-8")

    results["V4_LambdaRank"] = run_backtest_for_signal(
        panel, blended, cost_model, preds_v4, "ml_rank_signal", "V4",
        top_quantile=args.top,
    )

    # ── 对比表 ──
    print(f"\n{'=' * 64}")
    print("LambdaRank vs Baseline 扣费后绩效对比")
    print(f"{'=' * 64}")

    header = [
        "\n## LambdaRank V4 — 扣费后绩效对比 (AUM 5000万)",
        "",
        "| 指标 | V0: 线性 Baseline | V1: L2 LightGBM | V4: LambdaRank |",
        "|------|:---:|:---:|:---:|",
    ]

    for label, key in [
        ("年化收益", "Annualized_Return"),
        ("年化波动率", "Volatility"),
        ("**Sharpe Ratio**", "Sharpe_Ratio"),
        ("最大回撤", "Max_Drawdown"),
        ("Calmar Ratio", "Calmar_Ratio"),
        ("月胜率", "Win_Rate"),
    ]:
        vals = []
        for eid in ["V0_Linear", "V1_L2", "V4_LambdaRank"]:
            r = results.get(eid, {}).get("result", {})
            nm = r.get("net_metrics") or {}
            v = nm.get(key)
            if "Return" in key or "Volatility" in key or "Drawdown" in key or "Win" in key:
                vals.append(fmt_pct(v))
            else:
                vals.append(fmt_num(v))
        header.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")

    # 交易特征
    header.append("")
    header.append("### 交易特征")
    header.append("")
    header.append("| 指标 | V0: 线性 | V1: L2 | V4: LambdaRank |")
    header.append("|------|:---:|:---:|:---:|")

    for label, key in [("月均换手率", "avg_turnover"), ("月均成本(bps)", "avg_cost_bps")]:
        vals = []
        for eid in ["V0_Linear", "V1_L2", "V4_LambdaRank"]:
            r = results.get(eid, {}).get("result", {})
            v = r.get(key, 0)
            if "turnover" in key:
                vals.append(f"{v*100:.1f}%")
            else:
                vals.append(f"{v:.1f}")
        header.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")

    md_str = "\n".join(header)
    print(md_str)
    (OUTPUT_DIR / "ml_lambdarank_comparison.md").write_text(md_str, encoding="utf-8")

    # ── 总结 ──
    print(f"\n{'=' * 64}")
    print("LambdaRank 回测完成")
    print(f"{'=' * 64}")
    for eid, r in results.items():
        nm = (r.get("result", {}).get("net_metrics") or {})
        print(f"  {eid}: Sharpe={nm.get('Sharpe_Ratio', 0):.4f} | "
              f"Ret={nm.get('Annualized_Return', 0)*100:.2f}% | "
              f"TO={r.get('result', {}).get('avg_turnover', 0)*100:.1f}%")
    print(f"\n输出:")
    print(f"  - output/ml_lambdarank_predictions.parquet")
    print(f"  - output/ml_lambdarank_report.md")
    print(f"  - output/ml_lambdarank_comparison.md")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
