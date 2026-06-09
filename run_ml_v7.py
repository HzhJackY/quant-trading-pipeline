"""
V7: 1M Labels + 0M Gap + Turnover-Aware Objective — Runner (Stage 7).

The definitive test: does removing the 3M gap fix MaxDD while the
turnover penalty (lambda=2.0) independently controls turnover?

Comparison:
  - V0: Linear alpha_signal (1M horizon, no gap, 23.7% TO)
  - V5: 3M label + 3M gap + TO lambda=2.0 (12.9% TO, -27.12% MaxDD)
  - V7: 1M label + 0M gap + TO lambda=2.0 (target: < -20% MaxDD, > 1.0 Sharpe)

Usage:
  python run_ml_v7.py
  python run_ml_v7.py --skip-baselines
"""

from __future__ import annotations

import argparse, logging, time
from pathlib import Path
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("run_ml_v7")

from factor_research.ml_engine_v7 import LightGBMAlphaEngineV7, MLConfigV7
from factor_research.backtest_engine import run_backtest_with_costs
from factor_research.transaction_cost import TieredCostModel, UniverseCostConfig

OUTPUT_DIR = Path("output")

def load_data():
    for p in [OUTPUT_DIR/"preprocessed.parquet", OUTPUT_DIR/"split_universe_blended.parquet"]:
        if not p.exists(): raise FileNotFoundError(f"Missing {p}")
    panel = pd.read_parquet(OUTPUT_DIR/"preprocessed.parquet")
    blended = pd.read_parquet(OUTPUT_DIR/"split_universe_blended.parquet")
    print(f"Loaded: panel {panel.shape[0]:,} rows, blended {blended.shape[0]:,} rows")
    return panel, blended

def build_cost_model(aum=50_000_000):
    return TieredCostModel(aum=aum,
        large_cap_config=UniverseCostConfig(commission_bps=2.5, stamp_duty_bps=5.0, transfer_fee_bps=0.1, base_slippage_bps=5.0, impact_gamma=0.5, impact_eta=1.0),
        small_cap_config=UniverseCostConfig(commission_bps=2.5, stamp_duty_bps=5.0, transfer_fee_bps=0.1, base_slippage_bps=15.0, impact_gamma=0.65, impact_eta=1.5))

def fmt_pct(v, d=2): return "N/A" if v is None or np.isnan(v) else f"{v*100:.{d}f}%"
def fmt_num(v, d=4): return "N/A" if v is None or np.isnan(v) else f"{v:.{d}f}"

def run_bt(panel, blended, cost_model, signal_df, alpha_col, label, top=0.3, min_s=5):
    t0 = time.perf_counter()
    if signal_df is not None:
        bt = blended.merge(signal_df[["date","symbol",alpha_col]], on=["date","symbol"], how="left")
        bt[alpha_col] = bt[alpha_col].fillna(0.5)
    else:
        bt = blended
    r = run_backtest_with_costs(panel=panel, blended=bt, cost_model=cost_model, top_quantile=top, min_stocks_per_universe=min_s, alpha_col=alpha_col)
    nm = r.get("net_metrics") or {}
    dt = time.perf_counter() - t0
    print(f"  [{label}] {dt:.0f}s | Sharpe={nm.get('Sharpe_Ratio',0):.4f} | TO={r.get('avg_turnover',0)*100:.1f}% | Cost={r.get('avg_cost_bps',0):.1f}bps")
    return {"result": r, "wall_time_sec": dt}

def main():
    p = argparse.ArgumentParser(description="V7: 1M + 0Gap + TO")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--aum", type=float, default=50_000_000)
    p.add_argument("--top", type=float, default=0.3)
    p.add_argument("--turnover-lambda", type=float, default=2.0, dest="lam")
    args = p.parse_args()

    lam = args.lam
    print("=" * 64)
    print("V7: 1M Labels + 0M Gap + Turnover-Aware Objective")
    print("=" * 64)
    print(f"Label: forward_return_1m -> rank [0,1] | Gap: 0M (no blind zone)")
    print(f"Loss: L = 0.5*(p-y)^2 + {lam}*0.5*(p-prev)^2")
    print("=" * 64)

    panel, blended = load_data()
    cost_model = build_cost_model(args.aum)
    results = {}

    # V0
    if not args.skip_baselines:
        print("\n" + "-"*56 + "\n[V0] Linear alpha_signal\n" + "-"*56)
        results["V0"] = run_bt(panel, blended, cost_model, None, "alpha_signal", "V0", top=args.top)

    # V5 (cached)
    if not args.skip_baselines:
        v5_path = OUTPUT_DIR/"ml_v5_predictions_lambda2.0.parquet"
        if v5_path.exists():
            print("\n" + "-"*56 + "\n[V5] TO-Aware lambda=2.0, 3M gap (cached)\n" + "-"*56)
            results["V5"] = run_bt(panel, blended, cost_model, pd.read_parquet(v5_path), "inertia_ml_signal", "V5", top=args.top)
        else:
            print("\n[V5] Not cached — training required. Run run_ml_turnover_aware.py first.")
            return

    # V7
    print("\n" + "-"*64)
    print(f"[V7] 1M Label + 0M Gap + TO lambda={lam}")
    print("-"*64)
    config = MLConfigV7(train_months=36, val_months=6, test_months=1, label_horizon=1, lambda_turnover=lam, max_depth=4, num_leaves=24, learning_rate=0.02, n_estimators=2000, subsample=1.0, colsample_bytree=0.70, min_child_samples=100, reg_alpha=0.10, reg_lambda=0.10, early_stopping_rounds=50)
    engine = LightGBMAlphaEngineV7(config)
    t0 = time.perf_counter()
    preds = engine.run(panel, blended=blended)
    train_t = time.perf_counter() - t0
    preds.to_parquet(OUTPUT_DIR/"ml_v7_predictions.parquet", index=False)
    (OUTPUT_DIR/"ml_v7_report.md").write_text(engine.to_markdown_report(), encoding="utf-8")
    print(f"  Train: {train_t:.0f}s | Preds: {len(preds)} rows | {preds['date'].nunique()} cross-sections")
    results["V7"] = run_bt(panel, blended, cost_model, preds, "v7_ml_signal", "V7", top=args.top)
    results["V7"]["train_time_sec"] = train_t
    results["V7"]["predictions"] = preds

    # Comparison table
    print("\n" + "="*64 + "\nV7 Final Comparison\n" + "="*64)

    def gm(eid, key):
        r = results.get(eid,{}).get("result",{})
        return (r.get("net_metrics") or {}).get(key)
    def gto(eid): return results.get(eid,{}).get("result",{}).get("avg_turnover",0)
    def gcost(eid): return results.get(eid,{}).get("result",{}).get("avg_cost_bps",0)

    eids = ["V0","V5","V7"]
    labels = ["V0: Linear", "V5: 3M+gap+TO L2", "V7: 1M+0gap+TO L2"]

    lines = [
        f"## V7 Final Ablation — 1M + 0M Gap + Turnover-Aware",
        f"",
        f"| Metric | {' | '.join(labels)} |",
        f"|--------|{'|'.join([':---:' for _ in eids])}|",
    ]

    for m, k, f in [("Annualized Return","Annualized_Return",fmt_pct), ("Volatility","Volatility",fmt_pct), ("**Sharpe Ratio**","Sharpe_Ratio",fmt_num), ("**Max Drawdown**","Max_Drawdown",fmt_pct), ("Calmar Ratio","Calmar_Ratio",fmt_num), ("Win Rate","Win_Rate",fmt_pct)]:
        lines.append(f"| {m} | {' | '.join(f(gm(e,k)) for e in eids)} |")

    lines += ["", "### Trading Characteristics", "",
              f"| Metric | {' | '.join(labels)} |",
              f"|--------|{'|'.join([':---:' for _ in eids])}|"]

    for m, k in [("Monthly Turnover","avg_turnover"), ("Monthly Cost (bps)","avg_cost_bps")]:
        vals = []
        for e in eids:
            v = results.get(e,{}).get("result",{}).get(k,0)
            vals.append(f"{v*100:.1f}%" if "turnover" in k else f"{v:.1f}")
        lines.append(f"| {m} | {' | '.join(vals)} |")

    # V7 vs V5 delta
    lines += ["", "### V7 vs V5 — Key Improvements", "",
              "| Metric | V5 | V7 | Delta |",
              "|--------|:---:|:---:|:---:|"]
    for name, key, is_pct in [("Sharpe","Sharpe_Ratio",False), ("MaxDD","Max_Drawdown",True), ("Turnover","avg_turnover",True), ("Cost","avg_cost_bps",False)]:
        if key == "avg_turnover":
            v5, v7 = gto("V5"), gto("V7")
        elif key == "avg_cost_bps":
            v5, v7 = gcost("V5"), gcost("V7")
        else:
            v5, v7 = gm("V5",key), gm("V7",key)
        d = v7 - v5
        if is_pct: lines.append(f"| {name} | {fmt_pct(v5) if abs(v5)<1 else fmt_num(v5)} | {fmt_pct(v7) if abs(v7)<1 else fmt_num(v7)} | {fmt_pct(d) if abs(d)<1 else ('%+.4f'%d)} |")
        else: lines.append(f"| {name} | {fmt_num(v5)} | {fmt_num(v7)} | {'%+.4f'%d} |")

    # Verdict
    v7_sr = gm("V7","Sharpe_Ratio") or 0
    v7_dd = gm("V7","Max_Drawdown") or 0
    v7_to = gto("V7") * 100
    lines += ["", "### Production Readiness Assessment", ""]
    checks = []
    checks.append((f"Sharpe > 1.0", v7_sr > 1.0, f"{v7_sr:.4f}"))
    checks.append((f"MaxDD < -20%", abs(v7_dd) < 0.20, f"{v7_dd*100:.2f}%"))
    checks.append((f"Turnover < 25%", v7_to < 25, f"{v7_to:.1f}%"))
    lines.append("| Criterion | Met? | Value |")
    lines.append("|-----------|------|-------|")
    passed = 0
    for crit, met, val in checks:
        lines.append(f"| {crit} | {'YES' if met else 'NO'} | {val} |")
        if met: passed += 1
    lines += ["", f"**{passed}/3 criteria met.**"]
    if passed == 3:
        lines += ["", "**VERDICT: READY FOR PRODUCTION.** "
                       "V7 clears all deployment gates. "
                       "Recommend AUM scaling test as next step."]

    md = "\n".join(lines)
    print(md)
    (OUTPUT_DIR/"ml_v7_final_report.md").write_text(md, encoding="utf-8")

    # Save returns
    r7 = results.get("V7",{}).get("result",{})
    if r7:
        pd.DataFrame({"date": r7.get("net_returns",pd.Series()).index, "net_return": r7.get("net_returns",pd.Series()).values, "turnover": r7.get("turnovers",pd.Series()).values}).to_csv(OUTPUT_DIR/"ml_v7_backtest_returns.csv", index=False, encoding="utf-8-sig")

    print(f"\nOutput: output/ml_v7_predictions.parquet | ml_v7_report.md | ml_v7_final_report.md")
    for eid, lab in [("V0","V0"),("V5","V5"),("V7","V7")]:
        nm = results.get(eid,{}).get("result",{}).get("net_metrics") or {}
        print(f"  {lab}: Sharpe={nm.get('Sharpe_Ratio',0):.4f} | MaxDD={nm.get('Max_Drawdown',0)*100:.2f}% | TO={results.get(eid,{}).get('result',{}).get('avg_turnover',0)*100:.1f}%")
    print("="*64)

if __name__ == "__main__":
    main()
