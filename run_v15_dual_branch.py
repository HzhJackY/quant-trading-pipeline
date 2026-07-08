"""
V1.5 Dual-Branch Ensemble.

Trains independent fundamental and technical LightGBM branches, blends their
cross-sectional OOS ranks, and applies gap-aware EMA smoothing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from run_v15_experiment import (
    ExperimentConfig,
    V15_PANEL_PATH,
    prepare_panel_for_config,
    train_single_model,
)

logger = logging.getLogger("v15_dual_branch")

OUTPUT_DIR = Path("output")
DUAL_MODEL_DIR = OUTPUT_DIR / "production_models_v15_dual"
FINAL_PREDICTIONS_PATH = DUAL_MODEL_DIR / "Dual_Final_oos.parquet"
REPORT_CSV_PATH = OUTPUT_DIR / "v15_dual_branch_evaluation.csv"
REPORT_MD_PATH = OUTPUT_DIR / "v15_dual_branch_evaluation.md"

FEATURES_FUNDA = [
    "EP_neutral_z",
    "BP_raw_neutral_z",
    "SR_ROE_neutral_z",
    "Net_Profit_Margin_neutral_z",
    "Operating_Margin_neutral_z",
    "CFO_to_Earnings_neutral_z",
    "EPS_YoY_neutral_z",
    "SR_ProfitGrowth_YoY_neutral_z",
    "SR_RevGrowth_YoY_neutral_z",
    "ProfitGrowth_YoY_neutral_z",
    "RevGrowth_YoY_neutral_z",
    "Debt_Ratio_neutral_z",
    "Current_Ratio_neutral_z",
    "Quick_Ratio_neutral_z",
    "Equity_Multiplier_neutral_z",
]

FEATURES_TECH = [
    "Mom_1M_neutral_z",
    "Mom_3M_neutral_z",
    "Mom_6M_neutral_z",
    "Mom_12M_1M_neutral_z",
    "RSI_14_neutral_z",
    "Vol_20D_neutral_z",
    "Vol_60D_neutral_z",
    "Vol_120D_neutral_z",
    "Beta_neutral_z",
    "Skewness_60D_neutral_z",
    "MaxDD_60D_neutral_z",
    "High_Low_Range_20D_neutral_z",
    "Amihud_Illiquidity_neutral_z",
    "Dollar_Volume_20D_neutral_z",
    "Turnover_Volatility_20D_neutral_z",
    "PriceDev_20D_neutral_z",
    "VolChg_20D_neutral_z",
]

MODEL_F_CONFIG = ExperimentConfig(
    name="Model_F",
    description="Fundamental-only Quality+Value branch",
    panel_path=V15_PANEL_PATH,
    feature_neutral_z=FEATURES_FUNDA,
    gs_enabled=False,
    colsample_bytree=0.75,
    learning_rate=0.05,
    monotone_constraints={
        "EP_neutral_z": 1,
        "SR_ROE_neutral_z": 1,
        "SR_ProfitGrowth_YoY_neutral_z": 1,
    },
)

MODEL_T_CONFIG = ExperimentConfig(
    name="Model_T",
    description="Technical-only branch",
    panel_path=V15_PANEL_PATH,
    feature_neutral_z=FEATURES_TECH,
    gs_enabled=False,
    colsample_bytree=0.75,
    learning_rate=0.05,
)


def _validate_weights(weight_f: float, weight_t: float) -> None:
    if weight_f < 0 or weight_t < 0:
        raise ValueError("Blend weights must be non-negative")
    if not np.isclose(weight_f + weight_t, 1.0):
        raise ValueError("Blend weights must sum to 1")


def blend_oos_predictions(
    pred_f: pd.DataFrame,
    pred_t: pd.DataFrame,
    weight_f: float = 0.5,
    weight_t: float = 0.5,
    min_stocks: int = 30,
) -> pd.DataFrame:
    """Cross-sectionally rank two branch predictions and blend them."""
    _validate_weights(weight_f, weight_t)

    f = pred_f[["date", "symbol", "alpha_signal"]].copy()
    t = pred_t[["date", "symbol", "alpha_signal"]].copy()
    f["date"] = pd.to_datetime(f["date"])
    t["date"] = pd.to_datetime(t["date"])

    dates_f = set(f["date"].unique())
    dates_t = set(t["date"].unique())
    if dates_f != dates_t:
        raise ValueError("Branch OOS date sets do not match")

    merged = f.rename(columns={"alpha_signal": "pred_f"}).merge(
        t.rename(columns={"alpha_signal": "pred_t"}),
        on=["date", "symbol"],
        how="inner",
        validate="one_to_one",
    )

    counts = merged.groupby("date")["symbol"].nunique()
    if (counts < min_stocks).any():
        bad_date = counts[counts < min_stocks].index[0]
        raise ValueError(
            f"Branch intersection below {min_stocks} stocks on {bad_date}"
        )

    merged["rank_f"] = merged.groupby("date")["pred_f"].rank(
        pct=True, na_option="bottom"
    )
    merged["rank_t"] = merged.groupby("date")["pred_t"].rank(
        pct=True, na_option="bottom"
    )
    merged["raw_blend_pred"] = (
        weight_f * merged["rank_f"] + weight_t * merged["rank_t"]
    )
    return merged.sort_values(["date", "symbol"]).reset_index(drop=True)


def apply_gap_aware_ema(
    predictions: pd.DataFrame,
    alpha: float = 0.6,
) -> pd.DataFrame:
    """
    Apply stock-level EMA only across consecutive global OOS dates.

    If a stock misses one or more rebalance dates, its state is reset when it
    reappears.
    """
    if not 0 < alpha <= 1:
        raise ValueError("EMA alpha must be in (0, 1]")

    result = predictions.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["date", "symbol"]).reset_index(drop=True)

    global_dates = sorted(result["date"].dropna().unique())
    date_ord = {pd.Timestamp(dt): i for i, dt in enumerate(global_dates)}
    state: dict[str, tuple[int, float]] = {}
    final_values: list[float] = []

    for row in result.itertuples(index=False):
        current_ord = date_ord[pd.Timestamp(row.date)]
        raw = getattr(row, "raw_blend_pred")
        previous = state.get(row.symbol)
        is_consecutive = previous is not None and previous[0] == current_ord - 1

        if np.isfinite(raw):
            final = (
                alpha * float(raw) + (1 - alpha) * previous[1]
                if is_consecutive
                else float(raw)
            )
        elif is_consecutive:
            final = previous[1]
        else:
            final = np.nan

        if np.isfinite(final):
            state[row.symbol] = (current_ord, final)
        else:
            state.pop(row.symbol, None)
        final_values.append(final)

    result["final_pred"] = final_values
    result["alpha_signal"] = result["final_pred"]
    return result


EXPOSURE_COLUMNS = {
    "ROE": "SR_ROE_neutral_z",
    "ProfitGrowth": "SR_ProfitGrowth_YoY_neutral_z",
    "EP": "EP_neutral_z",
    "BP": "BP_raw_neutral_z",
}


def _merge_predictions_panel(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    signal_col: str,
) -> pd.DataFrame:
    required = {"date", "symbol", signal_col}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"Prediction columns missing: {sorted(missing)}")

    pred = predictions.copy()
    pnl = panel.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pnl["date"] = pd.to_datetime(pnl["date"])
    return pred.merge(pnl, on=["date", "symbol"], how="inner")


def _select_top_n(
    merged: pd.DataFrame,
    signal_col: str,
    n_positions: int,
) -> pd.DataFrame:
    selected = []
    for _, cross in merged.groupby("date", sort=True):
        valid = cross.dropna(subset=[signal_col]).sort_values(
            [signal_col, "symbol"], ascending=[False, True]
        )
        if len(valid) < n_positions:
            continue
        selected.append(valid.head(n_positions))
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def _annualized_sharpe(returns: pd.Series) -> float:
    values = returns.dropna()
    if len(values) < 2:
        return np.nan
    std = values.std(ddof=1)
    return float(values.mean() / std * np.sqrt(12)) if std > 0 else 0.0


def _max_drawdown(returns: pd.Series) -> float:
    values = returns.dropna()
    if values.empty:
        return np.nan
    nav = (1 + values).cumprod()
    return float((nav / nav.cummax() - 1).min())


def evaluate_top30(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    signal_col: str = "alpha_signal",
    n_positions: int = 30,
) -> dict:
    """Evaluate a fixed-size long-only Top-N portfolio."""
    merged = _merge_predictions_panel(predictions, panel, signal_col)
    if "forward_return_1m" not in merged.columns:
        raise KeyError("panel must contain forward_return_1m")

    top = _select_top_n(merged, signal_col, n_positions)
    if top.empty:
        raise ValueError("No dates have enough stocks for Top-N evaluation")

    monthly_all = top.groupby("date")["forward_return_1m"].mean().sort_index()
    monthly = monthly_all.dropna()
    dates = sorted(top["date"].unique())
    holdings = {
        dt: set(top[top["date"] == dt]["symbol"])
        for dt in dates
    }
    turnovers = []
    for previous_date, current_date in zip(dates, dates[1:]):
        previous = holdings[previous_date]
        current = holdings[current_date]
        turnovers.append(len(previous - current) / len(previous))

    # Top30/Bottom30 L/S diagnostic.
    ls_returns = []
    for dt, cross in merged.groupby("date", sort=True):
        valid = cross.dropna(subset=[signal_col, "forward_return_1m"]).sort_values(
            [signal_col, "symbol"], ascending=[False, True]
        )
        if len(valid) < 2 * n_positions:
            continue
        ls_returns.append(
            valid.head(n_positions)["forward_return_1m"].mean()
            - valid.tail(n_positions)["forward_return_1m"].mean()
        )

    return {
        "sharpe": _annualized_sharpe(monthly),
        "max_drawdown": _max_drawdown(monthly),
        "turnover": float(np.mean(turnovers)) if turnovers else np.nan,
        "mean_monthly_return": float(monthly.mean()),
        "ls_sharpe": _annualized_sharpe(pd.Series(ls_returns, dtype=float)),
        "n_months": int(monthly.notna().sum()),
        "n_positions": int(n_positions),
    }


def compute_top30_exposures(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    signal_col: str = "alpha_signal",
    n_positions: int = 30,
) -> dict[str, float]:
    """Mean monthly factor exposures of the fixed Top-N portfolio."""
    merged = _merge_predictions_panel(predictions, panel, signal_col)
    top = _select_top_n(merged, signal_col, n_positions)
    if top.empty:
        raise ValueError("No dates have enough stocks for exposure evaluation")

    exposures = {}
    for name, column in EXPOSURE_COLUMNS.items():
        if column not in top.columns:
            exposures[name] = np.nan
            continue
        monthly = top.groupby("date")[column].mean()
        exposures[name] = float(monthly.mean())
    return exposures


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-f", type=float, default=0.5)
    parser.add_argument("--weight-t", type=float, default=0.5)
    parser.add_argument("--ema-alpha", type=float, default=0.6)
    parser.add_argument("--output-dir", type=Path, default=DUAL_MODEL_DIR)
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Reuse existing Model_F and Model_T OOS prediction files",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_evaluation_panel() -> pd.DataFrame:
    panel = pd.read_parquet(V15_PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    close_col = next(
        (column for column in ("收盘", "close") if column in panel.columns),
        None,
    )
    if close_col is None:
        raise KeyError("No close price column found in V1.5 panel")
    panel = panel.sort_values(["symbol", "date"]).copy()
    panel["forward_return_1m"] = panel.groupby("symbol")[close_col].transform(
        lambda values: values.shift(-1) / values - 1.0
    )
    required = [
        "date",
        "symbol",
        "forward_return_1m",
        *EXPOSURE_COLUMNS.values(),
    ]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"Evaluation panel missing columns: {missing}")
    return panel[required].copy()


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    result = pd.read_parquet(path)
    result["date"] = pd.to_datetime(result["date"])
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    return result


def _build_comparison(
    predictions_by_name: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for name, predictions in predictions_by_name.items():
        metrics = evaluate_top30(predictions, panel)
        exposures = compute_top30_exposures(predictions, panel)
        rows.append({
            "Model": name,
            "Sharpe": metrics["sharpe"],
            "MaxDD": metrics["max_drawdown"],
            "Turnover": metrics["turnover"],
            "MeanMonthlyReturn": metrics["mean_monthly_return"],
            "LS_Sharpe": metrics["ls_sharpe"],
            "N_Months": metrics["n_months"],
            "ROE": exposures["ROE"],
            "ProfitGrowth": exposures["ProfitGrowth"],
            "EP": exposures["EP"],
            "BP": exposures["BP"],
        })
    return pd.DataFrame(rows)


def _write_report(
    comparison: pd.DataFrame,
    csv_path: Path,
    markdown_path: Path,
    weight_f: float,
    weight_t: float,
    ema_alpha: float,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(csv_path, index=False, encoding="utf-8-sig")

    final = comparison[comparison["Model"] == "Dual_Final"].iloc[0]
    turnover_pass = bool(final["Turnover"] < 0.30)
    roe_pass = bool(final["ROE"] > 0)
    pg_pass = bool(final["ProfitGrowth"] > 0)

    lines = [
        "# V1.5 Dual-Branch Ensemble Evaluation",
        "",
        f"- Fundamental/technical weights: {weight_f:.2f}/{weight_t:.2f}",
        f"- EMA alpha: {ema_alpha:.2f}",
        "- Portfolio: fixed Top 30 long-only",
        "",
        "## Comparison",
        "",
        comparison.round(4).to_markdown(index=False),
        "",
        "## Target Verdicts",
        "",
        f"- Turnover < 30%: **{turnover_pass}** ({final['Turnover']:.1%})",
        f"- ROE exposure > 0: **{roe_pass}** ({final['ROE']:.4f})",
        (
            "- ProfitGrowth exposure > 0: "
            f"**{pg_pass}** ({final['ProfitGrowth']:.4f})"
        ),
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    weight_f: float = 0.5,
    weight_t: float = 0.5,
    ema_alpha: float = 0.6,
    output_dir: Path = DUAL_MODEL_DIR,
    skip_training: bool = False,
) -> pd.DataFrame:
    """Train both branches, blend/smooth OOS predictions, and evaluate."""
    _validate_weights(weight_f, weight_t)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not skip_training:
        panel_f = prepare_panel_for_config(MODEL_F_CONFIG)
        panel_t = prepare_panel_for_config(MODEL_T_CONFIG)
        train_single_model(MODEL_F_CONFIG, panel_f, output_dir)
        train_single_model(MODEL_T_CONFIG, panel_t, output_dir)

    pred_f = _load_predictions(output_dir / "Model_F_oos.parquet")
    pred_t = _load_predictions(output_dir / "Model_T_oos.parquet")
    raw_blend = blend_oos_predictions(
        pred_f,
        pred_t,
        weight_f=weight_f,
        weight_t=weight_t,
    )
    final = apply_gap_aware_ema(raw_blend, alpha=ema_alpha)
    final.to_parquet(output_dir / "Dual_Final_oos.parquet", index=False)

    single = _load_predictions(
        OUTPUT_DIR / "production_models_v15" / "M5_V15_Full_oos.parquet"
    )
    evaluation_panel = _load_evaluation_panel()
    comparison = _build_comparison(
        {
            "Single_033": single,
            "Branch_F": pred_f,
            "Branch_T": pred_t,
            "Dual_Final": final,
        },
        evaluation_panel,
    )
    _write_report(
        comparison,
        REPORT_CSV_PATH,
        REPORT_MD_PATH,
        weight_f,
        weight_t,
        ema_alpha,
    )
    return comparison


def main() -> None:
    args = build_arg_parser().parse_args()
    _validate_weights(args.weight_f, args.weight_t)
    if args.dry_run:
        print(
            f"Model_F={len(FEATURES_FUNDA)} features | "
            f"Model_T={len(FEATURES_TECH)} features | "
            f"weights={args.weight_f:.2f}/{args.weight_t:.2f} | "
            f"ema_alpha={args.ema_alpha:.2f} | output={args.output_dir}"
        )
        return
    comparison = run_pipeline(
        weight_f=args.weight_f,
        weight_t=args.weight_t,
        ema_alpha=args.ema_alpha,
        output_dir=args.output_dir,
        skip_training=args.skip_training,
    )
    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
