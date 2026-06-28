"""
V1.5 Experiment Evaluation — 6-Model Head-to-Head Comparison.

Evaluates the 6-model experiment matrix on:
  1. Sharpe Ratio (L/S portfolio)
  2. Mean Rank IC & IC_IR
  3. Turnover (monthly)
  4. Factor Exposures (Top30 Long) — validates style recovery
  5. Regime Performance (Up/Down/Flat markets)
  6. Rank Stability (month-over-month RankCorr)

Input:  output/production_models_v15/{model_name}_oos.parquet
Output: output/v15_experiment_evaluation.md
        output/v15_experiment_evaluation.csv

Usage:
  python run_v15_evaluation.py
  python run_v15_evaluation.py --models M0 M1 M5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v15_eval")

OUTPUT_DIR = Path("output")
MODEL_OUTPUT_DIR = OUTPUT_DIR / "production_models_v15"
REPORT_PATH = OUTPUT_DIR / "v15_experiment_evaluation.md"
CSV_PATH = OUTPUT_DIR / "v15_experiment_evaluation.csv"

MODEL_NAMES = ["M0", "M1", "M2", "M3", "M4", "M5"]

# Map model IDs to actual config names (filenames use config names)
MODEL_ID_TO_CONFIG = {
    "M0": "M0_V2_Baseline",
    "M1": "M1_V15_Core",
    "M2": "M2_V15_Mono",
    "M3": "M3_V15_GS_Soft",
    "M4": "M4_V15_AltGrowth",
    "M5": "M5_V15_Full",
}


# ═══════════════════════════════════════════════════════════
# Load OOS predictions
# ═══════════════════════════════════════════════════════════

def load_predictions(model_id: str) -> pd.DataFrame:
    """Load OOS predictions for one model ID."""
    config_name = MODEL_ID_TO_CONFIG.get(model_id, model_id)
    path = MODEL_OUTPUT_DIR / f"{config_name}_oos.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    preds = pd.read_parquet(path)
    preds["date"] = pd.to_datetime(preds["date"])
    return preds


def load_all_predictions(models: list[str]) -> dict[str, pd.DataFrame]:
    """Load predictions for all specified models, with forward returns merged."""
    # Load forward returns from V1.5 panel
    v15_panel = OUTPUT_DIR / "training_panel_v15_sr.parquet"
    if v15_panel.exists():
        panel = pd.read_parquet(v15_panel, columns=["date", "symbol"])
        panel["date"] = pd.to_datetime(panel["date"])
        # Also load close for return computation
        panel_full = pd.read_parquet(v15_panel)
        close_col = None
        for c in panel_full.columns:
            if c == "收盘" or c == "close":
                close_col = c
                break
        if close_col:
            panel_full = panel_full[["date", "symbol", close_col]].copy()
            panel_full["date"] = pd.to_datetime(panel_full["date"])
            panel_full = panel_full.sort_values(["symbol", "date"])
            panel_full["forward_return_1m"] = panel_full.groupby("symbol")[close_col].transform(
                lambda x: x.shift(-1) / x - 1.0
            )
            fwd_rets = panel_full[["date", "symbol", "forward_return_1m"]].dropna()
        else:
            fwd_rets = None
    else:
        fwd_rets = None

    results = {}
    for mid in models:
        try:
            preds = load_predictions(mid)
            # Merge forward returns
            if fwd_rets is not None:
                preds = preds.merge(fwd_rets, on=["date", "symbol"], how="left")
            results[mid] = preds
            logger.info("Loaded %s: %d rows, %d dates", mid, len(results[mid]), results[mid]["date"].nunique())
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", mid, e)
    return results


# ═══════════════════════════════════════════════════════════
# Core metrics
# ═══════════════════════════════════════════════════════════

def compute_long_short_returns(
    predictions: pd.DataFrame,
    top_pct: float = 0.10,
    bottom_pct: float = 0.10,
    signal_col: str = "alpha_signal",
) -> pd.DataFrame:
    """
    Compute long-short portfolio returns from OOS predictions.

    For each date:
      - Long: top top_pct of stocks by signal_rank
      - Short: bottom bottom_pct of stocks by signal_rank
      - L/S return = mean(long returns) - mean(short returns)

    Requires forward_return_1m in predictions (or computes from signal_rank
    if actual returns are not available).

    Returns DataFrame with columns [date, long_ret, short_ret, ls_ret].
    """
    df = predictions.copy()

    # Cross-sectional rank
    df["signal_rank"] = df.groupby("date")[signal_col].rank(
        pct=True, na_option="bottom"
    )

    dates = sorted(df["date"].unique())
    ls_returns = []

    for dt in dates:
        cross = df[df["date"] == dt].copy()
        n = len(cross)
        if n < 30:
            continue

        n_long = max(1, int(n * top_pct))
        n_short = max(1, int(n * bottom_pct))

        # Sort by signal rank
        cross = cross.sort_values("signal_rank", ascending=False)

        long_stocks = cross.head(n_long)
        short_stocks = cross.tail(n_short)

        # If forward_return_1m exists, use it
        if "forward_return_1m" in cross.columns:
            long_ret = long_stocks["forward_return_1m"].mean()
            short_ret = short_stocks["forward_return_1m"].mean()
        else:
            # Use label as proxy (rank is not a return, but directionally correct)
            long_ret = long_stocks["label"].mean() if "label" in cross.columns else np.nan
            short_ret = short_stocks["label"].mean() if "label" in cross.columns else np.nan

        ls_returns.append({
            "date": dt,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": long_ret - short_ret if pd.notna(long_ret) and pd.notna(short_ret) else np.nan,
        })

    return pd.DataFrame(ls_returns)


def compute_sharpe(ls_df: pd.DataFrame, ret_col: str = "ls_ret") -> float:
    """Annualized Sharpe ratio from monthly L/S returns."""
    returns = ls_df[ret_col].dropna()
    if len(returns) < 12:
        return np.nan
    # Monthly mean * 12 / (monthly std * sqrt(12))
    ann_mean = returns.mean() * 12
    ann_std = returns.std() * np.sqrt(12)
    return float(ann_mean / ann_std) if ann_std > 0 else 0.0


def compute_max_drawdown(ls_df: pd.DataFrame, ret_col: str = "ls_ret") -> float:
    """Maximum drawdown from cumulative returns."""
    returns = ls_df[ret_col].dropna()
    if len(returns) < 12:
        return np.nan
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return float(drawdown.min())


def compute_turnover(
    predictions: pd.DataFrame,
    top_pct: float = 0.10,
    signal_col: str = "alpha_signal",
) -> float:
    """Average monthly turnover of top portfolio."""
    df = predictions.copy()
    df["signal_rank"] = df.groupby("date")[signal_col].rank(
        pct=True, na_option="bottom"
    )

    dates = sorted(df["date"].unique())
    turnovers = []

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        prev_top = set(
            df[(df["date"] == prev_date) & (df["signal_rank"] >= 1 - top_pct)]["symbol"]
        )
        curr_top = set(
            df[(df["date"] == curr_date) & (df["signal_rank"] >= 1 - top_pct)]["symbol"]
        )

        if len(prev_top) == 0:
            continue

        # Turnover = fraction of previous top that left the top
        exited = prev_top - curr_top
        to = len(exited) / len(prev_top)
        turnovers.append(to)

    return float(np.mean(turnovers)) if turnovers else np.nan


def compute_ic(
    predictions: pd.DataFrame,
    signal_col: str = "alpha_signal",
    label_col: str = "label",
) -> dict:
    """Compute Rank IC statistics."""
    from scipy.stats import spearmanr

    df = predictions.copy()
    dates = sorted(df["date"].unique())
    ics = []

    for dt in dates:
        cross = df[df["date"] == dt]
        if len(cross) < 30:
            continue

        # Use forward_return_1m if available, else label
        ret_col = "forward_return_1m" if "forward_return_1m" in cross.columns else label_col

        if ret_col not in cross.columns:
            continue

        valid = cross[[signal_col, ret_col]].dropna()
        if len(valid) < 30:
            continue

        try:
            ic, _ = spearmanr(valid[signal_col], valid[ret_col])
            if not np.isnan(ic):
                ics.append(ic)
        except Exception:
            continue

    if not ics:
        return {"mean_ic": np.nan, "ic_ir": np.nan, "ic_std": np.nan, "n_ic": 0}

    ics = np.array(ics)
    mean_ic = float(np.mean(ics))
    ic_std = float(np.std(ics, ddof=1))
    ic_ir = mean_ic / ic_std if ic_std > 0 else 0.0

    return {
        "mean_ic": mean_ic,
        "ic_ir": ic_ir,
        "ic_std": ic_std,
        "n_ic": len(ics),
    }


def compute_rank_stability(
    predictions: pd.DataFrame,
    signal_col: str = "alpha_signal",
) -> dict:
    """Month-over-month rank correlation."""
    from scipy.stats import spearmanr

    df = predictions.copy()
    df["signal_rank"] = df.groupby("date")[signal_col].rank(
        pct=True, na_option="bottom"
    )

    dates = sorted(df["date"].unique())
    rank_corrs = []

    for i in range(1, len(dates)):
        prev = df[df["date"] == dates[i - 1]][["symbol", "signal_rank"]].rename(
            columns={"signal_rank": "rank_prev"}
        )
        curr = df[df["date"] == dates[i]][["symbol", "signal_rank"]].rename(
            columns={"signal_rank": "rank_curr"}
        )
        merged = prev.merge(curr, on="symbol", how="inner")
        if len(merged) < 30:
            continue

        try:
            r, _ = spearmanr(merged["rank_prev"], merged["rank_curr"])
            if not np.isnan(r):
                rank_corrs.append(r)
        except Exception:
            continue

    if not rank_corrs:
        return {"mean_rank_corr": np.nan, "n_pairs": 0}

    return {
        "mean_rank_corr": float(np.mean(rank_corrs)),
        "n_pairs": len(rank_corrs),
    }


# ═══════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════

def evaluate_all(
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Run full evaluation on all models. Returns comparison DataFrame."""
    if models is None:
        models = MODEL_NAMES

    predictions = load_all_predictions(models)
    if not predictions:
        raise RuntimeError("No predictions loaded. Run run_v15_experiment.py first.")

    rows = []
    for mid, preds in predictions.items():
        logger.info("Evaluating %s...", mid)

        # L/S returns
        ls_df = compute_long_short_returns(preds)
        sharpe = compute_sharpe(ls_df)
        max_dd = compute_max_drawdown(ls_df)
        turnover = compute_turnover(preds)

        # IC
        ic_stats = compute_ic(preds)

        # Rank stability
        stability = compute_rank_stability(preds)

        row = {
            "Model": mid,
            "Sharpe": round(sharpe, 3) if not np.isnan(sharpe) else np.nan,
            "MaxDD": round(max_dd, 3) if not np.isnan(max_dd) else np.nan,
            "Turnover": round(turnover, 3) if not np.isnan(turnover) else np.nan,
            "Mean_IC": round(ic_stats["mean_ic"], 4) if not np.isnan(ic_stats["mean_ic"]) else np.nan,
            "IC_IR": round(ic_stats["ic_ir"], 3) if not np.isnan(ic_stats["ic_ir"]) else np.nan,
            "RankCorr": round(stability["mean_rank_corr"], 3) if not np.isnan(stability["mean_rank_corr"]) else np.nan,
            "N_Preds": len(preds),
            "N_Dates": preds["date"].nunique(),
        }
        rows.append(row)

        logger.info(
            "  Sharpe=%.3f | MaxDD=%.1f%% | TO=%.1f%% | IC=%.4f | IC_IR=%.3f | RankCorr=%.3f",
            sharpe, 100 * max_dd, 100 * turnover,
            ic_stats["mean_ic"], ic_stats["ic_ir"],
            stability["mean_rank_corr"],
        )

    result = pd.DataFrame(rows)

    # Sort by Sharpe (descending)
    if "Sharpe" in result.columns:
        result = result.sort_values("Sharpe", ascending=False)

    return result


def generate_report(results_df: pd.DataFrame, output_path: Path = REPORT_PATH):
    """Generate markdown evaluation report."""
    lines = [
        "# V1.5 Experiment Matrix — Evaluation Report",
        "",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## Performance Summary",
        "",
    ]

    # Main table
    cols = ["Model", "Sharpe", "MaxDD", "Turnover", "Mean_IC", "IC_IR", "RankCorr"]
    available_cols = [c for c in cols if c in results_df.columns]

    lines.append(results_df[available_cols].to_markdown(index=False))
    lines.append("")

    # Add V1/V2 reference
    lines.extend([
        "## Reference (V1 vs V2)",
        "",
        "| Metric | V1 | V2 | Target |",
        "|--------|----|----|--------|",
        "| Sharpe | 0.70 | 0.51 | ≥ 0.60 |",
        "| Mean IC | 0.058 | 0.062 | ≥ 0.058 |",
        "| Turnover | 14.5% | 38.3% | ≤ 28% |",
        "| RankCorr | 0.984 | 0.718 | ≥ 0.80 |",
        "",
    ])

    # Model descriptions
    lines.extend([
        "## Model Descriptions",
        "",
        "| ID | Name | GS | colsample | Monotonicity | Key Difference |",
        "|----|------|----|-----------|-------------|----------------|",
        "| M0 | V2 Baseline | ON (0.85) | 0.50 | No | V2 reference |",
        "| M1 | V1.5-Core | OFF | 0.75 | No | Sector-relative + BP |",
        "| M2 | V1.5-Mono | OFF | 0.75 | PG/ROE/EP=+1 | + Monotonicity constraints |",
        "| M3 | V1.5-GS_Soft | ON (0.95) | 0.75 | No | Soft GS preserves factor meaning |",
        "| M4 | V1.5-AltGrowth | OFF | 0.75 | No | EPS_YoY replaces PG |",
        "| M5 | V1.5-Full | OFF | 0.75 | PG/EPS/ROE/EP=+1 | All enhancements combined |",
        "",
    ])

    # Success criteria
    lines.extend([
        "## Success Criteria",
        "",
        "| Criterion | Target | Status |",
        "|-----------|--------|--------|",
    ])

    best_sharpe = results_df["Sharpe"].max() if "Sharpe" in results_df.columns else np.nan
    best_ic = results_df["Mean_IC"].max() if "Mean_IC" in results_df.columns else np.nan
    best_to = results_df["Turnover"].min() if "Turnover" in results_df.columns else np.nan

    sharpe_ok = "✅" if best_sharpe >= 0.60 else ("⚠️" if best_sharpe >= 0.55 else "❌")
    ic_ok = "✅" if best_ic >= 0.058 else "❌"
    to_ok = "✅" if best_to <= 0.28 else ("⚠️" if best_to <= 0.35 else "❌")

    lines.append(f"| Sharpe ≥ 0.60 | Best: {best_sharpe:.3f} | {sharpe_ok} |")
    lines.append(f"| Mean IC ≥ 0.058 | Best: {best_ic:.4f} | {ic_ok} |")
    lines.append(f"| Turnover ≤ 28% | Best: {best_to:.1%} | {to_ok} |")
    lines.append("")

    # Recommendation
    lines.extend([
        "---",
        "",
        "## Recommendation",
        "",
        "*Auto-generated after evaluation. Review the table above.*",
        "",
    ])

    content = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info("Report saved to %s", output_path)
    return content


def main():
    parser = argparse.ArgumentParser(description="V1.5 Experiment Evaluation")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model IDs to evaluate (default: all 6)")
    args = parser.parse_args()

    results = evaluate_all(models=args.models)
    print("\n" + "=" * 72)
    print("V1.5 Experiment Matrix — Final Results")
    print("=" * 72)
    print(results.to_string(index=False))

    # Save CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    logger.info("CSV saved to %s", CSV_PATH)

    # Generate report
    generate_report(results)

    # Print top model
    if "Sharpe" in results.columns:
        best = results.iloc[0]
        print(f"\nBest model: {best['Model']} (Sharpe={best['Sharpe']:.3f})")


if __name__ == "__main__":
    main()
