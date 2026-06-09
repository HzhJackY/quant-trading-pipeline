"""
V6 Label Blending + Time-Decay 回测 — 运行脚本 (Stage 6).

Executes V6 training (blended label + time-decay weights + turnover-aware
objective) and compares against V0 (Linear) and V5 (lambda=2.0) baselines.

Key Innovations (V6):
  1. Label Blending:
     y_target = 0.4 * forward_return_1m + 0.6 * forward_return_3m
     -> cross-sectional rank [0,1]
     Injects short-term "crisis sensitivity" while keeping 3M anchor.

  2. Time-Decay Sample Weighting:
     w_i = exp(-dt * ln(2) / H)  with H = 12 months
     Recent samples weight higher; stale history decays exponentially.
     Injected via lgb.Dataset(..., weight=sample_weights).

  3. Inherited from V5:
     - Turnover-Aware Custom Objective (lambda=2.0)
     - 3M Gap (prevents leakage through 3M label component)
     - prev_signal = alpha_signal_{t-1} anchor via closure
     - subsample=1.0 (closure alignment)

Ablation Comparison:
  - V0: Linear alpha_signal (Baseline)
  - V5: Turnover-Aware L2, lambda=2.0 (best from lambda sweep)
  - V6: V5 + Blended Label + Time-Decay Weights

Goal: MaxDD compressed from -27.12% to < -20%, Net Sharpe > 1.0

Usage:
  python run_ml_v6.py
  python run_ml_v6.py --skip-baselines     # V6 only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ml_v6")

from factor_research.ml_engine_v6 import LightGBMAlphaEngineV6, MLConfigV6
from factor_research.ml_engine_v5 import LightGBMAlphaEngineV5, MLConfigV5
from factor_research.ml_engine import LightGBMAlphaEngine, MLConfig
from factor_research.backtest_engine import (
    run_backtest_with_costs,
)
from factor_research.transaction_cost import TieredCostModel, UniverseCostConfig

OUTPUT_DIR = Path("output")


# ═══════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════


def load_data():
    preprocessed_path = OUTPUT_DIR / "preprocessed.parquet"
    blended_path = OUTPUT_DIR / "split_universe_blended.parquet"

    for p in [preprocessed_path, blended_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run upstream pipeline first.")

    panel = pd.read_parquet(preprocessed_path)
    blended = pd.read_parquet(blended_path)
    print(f"Loaded: panel {panel.shape[0]:,} rows x {panel.shape[1]} cols | "
          f"{panel['date'].nunique()} cross-sections")
    print(f"Loaded: blended {blended.shape[0]:,} rows")
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
    """Run cost-aware backtest for a single signal and return results dict."""
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
# V6 Training + Backtest
# ═══════════════════════════════════════════════════════════


def train_and_backtest_v6(
    panel: pd.DataFrame,
    blended: pd.DataFrame,
    cost_model: TieredCostModel,
    blend_alpha: float = 0.4,
    half_life: int = 12,
    lambda_turnover: float = 2.0,
    top_quantile: float = 0.3,
    min_stocks: int = 5,
) -> dict:
    """
    Train V6 model (blended label + time-decay + turnover-aware) and backtest.
    """
    print(f"\n{'─' * 64}")
    print(f"[V6] Blended Label + Time-Decay + Turnover-Aware (lambda={lambda_turnover})")
    print(f"{'─' * 64}")
    print(f"  Label: {blend_alpha*100:.0f}% * ret_1m + {(1-blend_alpha)*100:.0f}% * ret_3m")
    print(f"  Time-Decay: half_life = {half_life}M | w = exp(-dt * ln(2) / {half_life})")
    print(f"  Turnover Penalty: lambda = {lambda_turnover}")
    print(f"  Gap: 3M | subsample: 1.0 (closure alignment)")

    # ── Config ──
    config = MLConfigV6(
        train_months=36,
        val_months=6,
        test_months=1,
        label_horizon=3,
        blend_alpha=blend_alpha,
        half_life=half_life,
        lambda_turnover=lambda_turnover,
        max_depth=4,
        num_leaves=24,
        learning_rate=0.02,
        n_estimators=2000,
        subsample=1.0,           # closure alignment
        colsample_bytree=0.70,
        min_child_samples=100,
        reg_alpha=0.10,
        reg_lambda=0.10,
        early_stopping_rounds=50,
    )

    # ── Train ──
    engine = LightGBMAlphaEngineV6(config=config)
    t0 = time.perf_counter()
    predictions = engine.run(panel, blended=blended)
    train_time = time.perf_counter() - t0

    print(f"  Training time: {train_time:.0f}s | "
          f"Predictions: {len(predictions)} rows | "
          f"{predictions['date'].nunique()} cross-sections")

    # ── Save predictions ──
    pred_path = OUTPUT_DIR / "ml_v6_predictions.parquet"
    predictions.to_parquet(pred_path, index=False)
    print(f"  Predictions saved: {pred_path}")

    # ── Save training report ──
    report = engine.to_markdown_report()
    (OUTPUT_DIR / "ml_v6_report.md").write_text(report, encoding="utf-8")

    # ── Backtest ──
    result = run_backtest_for_signal(
        panel, blended, cost_model, predictions,
        alpha_col="v6_ml_signal",
        label="V6",
        top_quantile=top_quantile,
        min_stocks=min_stocks,
    )
    result["train_time_sec"] = train_time
    result["predictions"] = predictions
    result["feature_importance"] = engine.get_feature_importance()

    return result


# ═══════════════════════════════════════════════════════════
# Comparison Table Generation
# ═══════════════════════════════════════════════════════════


def generate_comparison_table(
    results: dict[str, dict],
    aum: float,
) -> str:
    """
    Generate V0 vs V5 vs V6 ablation comparison table.
    Focus on MaxDD improvement and Net Sharpe.
    """

    def get_metric(exp_id, key):
        r = results.get(exp_id, {}).get("result", {})
        nm = r.get("net_metrics") or {}
        return nm.get(key)

    def get_to(exp_id):
        r = results.get(exp_id, {}).get("result", {})
        return r.get("avg_turnover", 0)

    def get_cost(exp_id):
        r = results.get(exp_id, {}).get("result", {})
        return r.get("avg_cost_bps", 0)

    exp_ids = ["V0_Linear", "V5_Turnover", "V6_BlendedDecay"]
    exp_labels = ["V0: Linear", "V5: TO-Aware (lambda=2.0)", "V6: V5 + Blend + Decay"]

    header = [
        f"## V6 Label Blending + Time-Decay — Ablation Study",
        f"",
        f"- **AUM:** {aum/1e4:.0f} 万",
        f"- **Stock Selection:** Top 30% split-universe equal-weight",
        f"- **Cost Model:** Almgren-Chriss impact + tiered fees",
        f"",
        f"### V6 Innovations",
        f"",
        f"| Innovation | Detail |",
        f"|------------|--------|",
        f"| **Label Blending** | y = 0.4 * ret_1m + 0.6 * ret_3m -> rank [0,1] |",
        f"| **Time-Decay Weights** | w = exp(-dt * ln(2) / 12), via lgb.Dataset(weight=...)  |",
        f"| **Turnover Penalty** | Custom objective L = 0.5*(p-y)^2 + 2.0*0.5*(p-prev)^2 |",
        f"| **Gap**              | 3M (prevents leakage through 3M label component) |",
        f"",
        f"### Performance Comparison (Net of Costs)",
        f"",
        f"| Metric | {' | '.join(exp_labels)} |",
        f"|--------|{'|'.join([':---:' for _ in exp_ids])}|",
    ]

    # ── Performance metrics ──
    metrics = [
        ("Annualized Return", "Annualized_Return", fmt_pct),
        ("Annualized Volatility", "Volatility", fmt_pct),
        ("**Sharpe Ratio**", "Sharpe_Ratio", fmt_num),
        ("**Max Drawdown**", "Max_Drawdown", fmt_pct),
        ("Calmar Ratio", "Calmar_Ratio", fmt_num),
        ("Monthly Win Rate", "Win_Rate", fmt_pct),
    ]

    for label, key, formatter in metrics:
        vals = [formatter(get_metric(eid, key)) for eid in exp_ids]
        header.append(f"| {label} | {' | '.join(vals)} |")

    # ── Trading characteristics ──
    header.append("")
    header.append("### Trading Characteristics")
    header.append("")
    header.append(f"| Metric | {' | '.join(exp_labels)} |")
    header.append(f"|--------|{'|'.join([':---:' for _ in exp_ids])}|")

    for label, key in [
        ("Monthly One-Way Turnover", "avg_turnover"),
        ("Monthly Avg Cost (bps)", "avg_cost_bps"),
    ]:
        vals = []
        for eid in exp_ids:
            r = results.get(eid, {}).get("result", {})
            v = r.get(key, 0)
            if key == "avg_turnover":
                vals.append(f"{v*100:.1f}%")
            else:
                vals.append(f"{v:.1f}")
        header.append(f"| {label} | {' | '.join(vals)} |")

    # ── V6 vs V5 Improvement Analysis ──
    header.append("")
    header.append("### V6 vs V5 — Improvement Analysis")
    header.append("")
    header.append("| Metric | V5 (lambda=2.0) | V6 | Delta | % Change |")
    header.append("|--------|:---:|:---:|:---:|:---:|")

    v5_sr = get_metric("V5_Turnover", "Sharpe_Ratio") or 0
    v6_sr = get_metric("V6_BlendedDecay", "Sharpe_Ratio") or 0
    v5_dd = get_metric("V5_Turnover", "Max_Drawdown") or 0
    v6_dd = get_metric("V6_BlendedDecay", "Max_Drawdown") or 0
    v5_ret = get_metric("V5_Turnover", "Annualized_Return") or 0
    v6_ret = get_metric("V6_BlendedDecay", "Annualized_Return") or 0
    v5_to = get_to("V5_Turnover") * 100
    v6_to = get_to("V6_BlendedDecay") * 100
    v5_cost = get_cost("V5_Turnover")
    v6_cost = get_cost("V6_BlendedDecay")
    v5_cal = get_metric("V5_Turnover", "Calmar_Ratio") or 0
    v6_cal = get_metric("V6_BlendedDecay", "Calmar_Ratio") or 0

    improvements = [
        ("Sharpe Ratio", v5_sr, v6_sr, False),
        ("Annualized Return", v5_ret, v6_ret, True),
        ("Max Drawdown", v5_dd, v6_dd, True),
        ("Calmar Ratio", v5_cal, v6_cal, False),
        ("Monthly Turnover", v5_to / 100, v6_to / 100, True),
        ("Monthly Cost (bps)", v5_cost, v6_cost, False),
    ]

    for name, base, new, is_pct in improvements:
        delta = new - base
        if is_pct:
            pct_change = f"{delta/base*100:+.1f}%" if base != 0 else "N/A"
            header.append(
                f"| {name} | {fmt_pct(base) if abs(base)<1 else fmt_num(base)} | "
                f"{fmt_pct(new) if abs(new)<1 else fmt_num(new)} | "
                f"{fmt_pct(delta) if abs(delta)<1 else ('%+.4f' % delta)} | {pct_change} |"
            )
        else:
            pct_change = f"{delta/abs(base)*100:+.1f}%" if base != 0 else "N/A"
            header.append(
                f"| {name} | {fmt_num(base)} | {fmt_num(new)} | "
                f"{fmt_num(delta) if abs(delta)<0.01 else '%+.4f' % delta} | {pct_change} |"
            )

    # ── Training time ──
    header.append("")
    header.append("### Training Time")
    header.append("")
    header.append("| Config | Wall Time | Train Time |")
    header.append("|--------|-----------|------------|")
    for eid, label in zip(exp_ids, exp_labels):
        r = results.get(eid, {})
        wt = r.get("wall_time_sec", 0)
        tt = r.get("train_time_sec", 0)
        header.append(f"| {label} | {wt:.0f}s | {tt:.0f}s |")

    return "\n".join(header)


# ═══════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="V6 Label Blending + Time-Decay Backtest")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip V0 and V5 (V6 only)")
    parser.add_argument("--aum", type=float, default=50_000_000)
    parser.add_argument("--top", type=float, default=0.3)
    parser.add_argument("--min-stocks", type=int, default=5)
    parser.add_argument("--blend-alpha", type=float, default=0.4,
                        help="Weight of 1M return in blended label (default 0.4)")
    parser.add_argument("--half-life", type=int, default=12,
                        help="Half-life in months for time-decay (default 12)")
    parser.add_argument("--turnover-lambda", type=float, default=2.0,
                        dest="turnover_lambda",
                        help="Turnover penalty coefficient (default 2.0)")
    args = parser.parse_args()

    lam = args.turnover_lambda
    print("=" * 64)
    print("V6: Label Blending + Time-Decay + Turnover-Aware Objective")
    print("=" * 64)
    print()
    print(f"Label:    y = {args.blend_alpha} * ret_1m + "
          f"{1-args.blend_alpha} * ret_3m -> rank")
    print(f"Decay:    w = exp(-dt * ln(2) / {args.half_life}) "
          f"(half-life = {args.half_life}M)")
    print(f"Penalty:  lambda = {lam} "
          f"(L = 0.5*(p-y)^2 + {lam}*0.5*(p-prev)^2)")
    print(f"Injection: closure -> LightGBM custom objective + Dataset weight")
    print("=" * 64)

    # ── Load data ──
    panel, blended = load_data()
    cost_model = build_cost_model(aum=args.aum)

    results: dict[str, dict] = {}

    # ── V0: Linear Baseline ──
    if not args.skip_baselines:
        print(f"\n{'─' * 56}")
        print("[V0] Linear alpha_signal (Baseline)")
        print(f"{'─' * 56}")
        results["V0_Linear"] = run_backtest_for_signal(
            panel, blended, cost_model, None, "alpha_signal", "V0",
            top_quantile=args.top,
            min_stocks=args.min_stocks,
        )

    # ── V5: Turnover-Aware (lambda=2.0, cached if available) ──
    if not args.skip_baselines:
        v5_cached = OUTPUT_DIR / "ml_v5_predictions_lambda2.0.parquet"
        if v5_cached.exists():
            print(f"\n{'─' * 56}")
            print("[V5] Turnover-Aware L2, lambda=2.0 (cached)")
            print(f"{'─' * 56}")
            preds_v5 = pd.read_parquet(v5_cached)
            results["V5_Turnover"] = run_backtest_for_signal(
                panel, blended, cost_model, preds_v5,
                alpha_col="inertia_ml_signal",
                label="V5",
                top_quantile=args.top,
                min_stocks=args.min_stocks,
            )
        else:
            print(f"\n{'─' * 56}")
            print("[V5] Turnover-Aware L2, lambda=2.0 (training...)")
            print(f"{'─' * 56}")
            config_v5 = MLConfigV5(
                train_months=36, val_months=6, test_months=1,
                label_horizon=3, lambda_turnover=2.0,
                max_depth=4, num_leaves=24, learning_rate=0.02,
                n_estimators=2000, subsample=1.0, colsample_bytree=0.70,
                min_child_samples=100, reg_alpha=0.10, reg_lambda=0.10,
                early_stopping_rounds=50,
            )
            engine_v5 = LightGBMAlphaEngineV5(config=config_v5)
            t0 = time.perf_counter()
            preds_v5 = engine_v5.run(panel, blended=blended)
            train_time = time.perf_counter() - t0
            preds_v5.to_parquet(v5_cached, index=False)
            engine_v5.to_markdown_report()
            results["V5_Turnover"] = run_backtest_for_signal(
                panel, blended, cost_model, preds_v5,
                alpha_col="inertia_ml_signal",
                label="V5",
                top_quantile=args.top,
                min_stocks=args.min_stocks,
            )
            results["V5_Turnover"]["train_time_sec"] = train_time

    # ── V6: Blended Label + Time-Decay + Turnover-Aware ──
    results["V6_BlendedDecay"] = train_and_backtest_v6(
        panel=panel,
        blended=blended,
        cost_model=cost_model,
        blend_alpha=args.blend_alpha,
        half_life=args.half_life,
        lambda_turnover=args.turnover_lambda,
        top_quantile=args.top,
        min_stocks=args.min_stocks,
    )

    # ── Comparison table ──
    print(f"\n{'=' * 64}")
    print("V6 Ablation Report")
    print(f"{'=' * 64}")

    table_md = generate_comparison_table(results, aum=args.aum)
    print(table_md)

    # Save report
    report_path = OUTPUT_DIR / "ml_v6_ablation_report.md"
    report_path.write_text(table_md, encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    # ── Save backtest returns ──
    for eid, key in [("V6_BlendedDecay", "V6")]:
        r = results.get(eid, {})
        bt_result = r.get("result", {})
        if bt_result:
            ret_path = OUTPUT_DIR / f"ml_v6_backtest_returns.csv"
            ret_df = pd.DataFrame({
                "date": bt_result.get("net_returns", pd.Series()).index,
                "net_return": bt_result.get("net_returns", pd.Series()).values,
                "turnover": bt_result.get("turnovers", pd.Series()).values,
            })
            if not ret_df.empty:
                ret_df.to_csv(ret_path, index=False, encoding="utf-8-sig")
                print(f"Backtest returns saved: {ret_path}")

    # ── Summary ──
    print(f"\n{'=' * 64}")
    print("V6 Ablation Study Complete")
    print(f"{'=' * 64}")

    for eid, label in [
        ("V0_Linear", "V0: Linear"),
        ("V5_Turnover", "V5: TO-Aware lambda=2.0"),
        ("V6_BlendedDecay", "V6: Blend+Decay+TO"),
    ]:
        nm = (results.get(eid, {}).get("result", {}).get("net_metrics") or {})
        sr = nm.get("Sharpe_Ratio", 0)
        dd = nm.get("Max_Drawdown", 0)
        ret = nm.get("Annualized_Return", 0)
        to = results.get(eid, {}).get("result", {}).get("avg_turnover", 0)
        cost = results.get(eid, {}).get("result", {}).get("avg_cost_bps", 0)
        wt = results.get(eid, {}).get("wall_time_sec",
               results.get(eid, {}).get("train_time_sec", 0))
        print(f"  {label}: "
              f"Sharpe={sr:.4f} | "
              f"Ret={ret*100:.2f}% | "
              f"MaxDD={dd*100:.2f}% | "
              f"TO={to*100:.1f}% | "
              f"Cost={cost:.1f}bps | "
              f"Time={wt:.0f}s")

    print(f"\nOutput files:")
    print(f"  - output/ml_v6_predictions.parquet")
    print(f"  - output/ml_v6_report.md")
    print(f"  - output/ml_v6_ablation_report.md")
    print(f"  - output/ml_v6_backtest_returns.csv")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
