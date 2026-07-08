"""
V1.5 compact monthly factor experiment.

Compares a pure fundamental model with two small extensions containing only
medium-horizon momentum and volatility risk proxies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_v15_dual_branch import (
    EXPOSURE_COLUMNS,
    FEATURES_FUNDA,
    _load_evaluation_panel,
    _load_predictions,
    compute_top30_exposures,
    evaluate_top30,
)
from run_v15_experiment import (
    ExperimentConfig,
    V15_PANEL_PATH,
    prepare_panel_for_config,
    train_single_model,
)

OUTPUT_DIR = Path("output")
COMPACT_MODEL_DIR = OUTPUT_DIR / "production_models_v15_compact"
REPORT_CSV_PATH = OUTPUT_DIR / "v15_compact_evaluation.csv"
REPORT_MD_PATH = OUTPUT_DIR / "v15_compact_evaluation.md"
YEARLY_CSV_PATH = OUTPUT_DIR / "v15_compact_yearly_sharpe.csv"

MONOTONE_CONSTRAINTS = {
    "EP_neutral_z": 1,
    "SR_ROE_neutral_z": 1,
    "SR_ProfitGrowth_YoY_neutral_z": 1,
}


def _compact_config(name: str, description: str, features: list[str]) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        description=description,
        panel_path=V15_PANEL_PATH,
        feature_neutral_z=features,
        gs_enabled=False,
        colsample_bytree=0.75,
        learning_rate=0.05,
        reg_alpha=0.10,
        monotone_constraints=MONOTONE_CONSTRAINTS.copy(),
        seeds=[42],
        lambda_turnover=2.0,
        train_months=36,
        val_months=6,
        test_months=1,
    )


COMPACT_F_CONFIG = _compact_config(
    "Compact_F",
    "Fundamental-only compact monthly model",
    FEATURES_FUNDA.copy(),
)
COMPACT_FT_CONFIG = _compact_config(
    "Compact_FT",
    "Fundamental core plus Mom_3M and Vol_60D",
    FEATURES_FUNDA + ["Mom_3M_neutral_z", "Vol_60D_neutral_z"],
)
COMPACT_FT3_CONFIG = _compact_config(
    "Compact_FT3",
    "Fundamental core plus Mom_3M, Vol_60D, and Mom_6M",
    FEATURES_FUNDA
    + ["Mom_3M_neutral_z", "Vol_60D_neutral_z", "Mom_6M_neutral_z"],
)

COMPACT_CONFIGS = {
    "Compact_F": COMPACT_F_CONFIG,
    "Compact_FT": COMPACT_FT_CONFIG,
    "Compact_FT3": COMPACT_FT3_CONFIG,
}


def _annualized_sharpe(returns: pd.Series) -> float:
    values = returns.dropna()
    if len(values) < 2:
        return np.nan
    std = values.std(ddof=1)
    return float(values.mean() / std * np.sqrt(12)) if std > 0 else 0.0


def monthly_top30_returns(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    signal_col: str = "alpha_signal",
) -> pd.Series:
    """Return the realized monthly returns of a fixed Top-30 portfolio."""
    pred = predictions[["date", "symbol", signal_col]].copy()
    pred["date"] = pd.to_datetime(pred["date"])
    merged = pred.merge(
        panel[["date", "symbol", "forward_return_1m"]],
        on=["date", "symbol"],
        how="inner",
    )
    rows = {}
    for date, cross in merged.groupby("date", sort=True):
        valid = cross.dropna(subset=[signal_col]).sort_values(
            [signal_col, "symbol"],
            ascending=[False, True],
        )
        if len(valid) < 30:
            continue
        rows[pd.Timestamp(date)] = valid.head(30)["forward_return_1m"].mean()
    return pd.Series(rows, dtype=float).sort_index()


def yearly_top30_sharpe(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    monthly = monthly_top30_returns(predictions, panel).dropna()
    if monthly.empty:
        return pd.DataFrame(columns=["year", "sharpe", "n_months"])
    frame = monthly.rename("return").reset_index(name="return")
    frame = frame.rename(columns={"index": "date"})
    frame["year"] = frame["date"].dt.year
    rows = []
    for year, group in frame.groupby("year"):
        rows.append({
            "year": int(year),
            "sharpe": _annualized_sharpe(group["return"]),
            "n_months": int(len(group)),
        })
    return pd.DataFrame(rows)


def mixed_model_passes(
    candidate: dict,
    baseline: dict,
    yearly_comparison: pd.DataFrame,
) -> bool:
    """Apply all approved compact-model selection rules."""
    annual = yearly_comparison.dropna(
        subset=["baseline_sharpe", "candidate_sharpe"]
    )
    positive_years = int(
        (annual["candidate_sharpe"] > annual["baseline_sharpe"]).sum()
    )
    return bool(
        candidate["Sharpe"] > baseline["Sharpe"]
        and candidate["MaxDD"] >= baseline["MaxDD"] - 0.05
        and candidate["Turnover"] <= baseline["Turnover"] + 0.10
        and candidate["ROE"] > 0
        and candidate["ProfitGrowth"] > 0
        and positive_years >= 2
    )


def _validate_oos_dates(predictions: dict[str, pd.DataFrame]) -> None:
    date_sets = {
        name: set(pd.to_datetime(frame["date"]).unique())
        for name, frame in predictions.items()
    }
    reference_name, reference_dates = next(iter(date_sets.items()))
    mismatches = [
        name for name, dates in date_sets.items()
        if dates != reference_dates
    ]
    if mismatches:
        raise ValueError(
            f"OOS date sets differ from {reference_name}: {mismatches}"
        )


def evaluate_compact_predictions(
    predictions: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_oos_dates(predictions)
    rows = []
    yearly_by_model = {}
    for name, frame in predictions.items():
        metrics = evaluate_top30(frame, panel)
        exposures = compute_top30_exposures(frame, panel)
        yearly = yearly_top30_sharpe(frame, panel).rename(
            columns={"sharpe": name, "n_months": f"{name}_n_months"}
        )
        yearly_by_model[name] = yearly
        rows.append({
            "Model": name,
            "Sharpe": metrics["sharpe"],
            "MaxDD": metrics["max_drawdown"],
            "Turnover": metrics["turnover"],
            "MeanMonthlyReturn": metrics["mean_monthly_return"],
            "LS_Sharpe": metrics["ls_sharpe"],
            "N_Months": metrics["n_months"],
            **exposures,
        })

    yearly_result = None
    for yearly in yearly_by_model.values():
        yearly_result = (
            yearly
            if yearly_result is None
            else yearly_result.merge(yearly, on="year", how="outer")
        )
    return pd.DataFrame(rows), yearly_result.sort_values("year")


def _add_selection_verdicts(
    comparison: pd.DataFrame,
    yearly: pd.DataFrame,
) -> pd.DataFrame:
    result = comparison.copy()
    result["PassAllRules"] = False
    baseline = result[result["Model"] == "Compact_F"].iloc[0].to_dict()
    for name in ("Compact_FT", "Compact_FT3"):
        candidate = result[result["Model"] == name].iloc[0].to_dict()
        yearly_comparison = yearly[["year", "Compact_F", name]].rename(
            columns={
                "Compact_F": "baseline_sharpe",
                name: "candidate_sharpe",
            }
        )
        result.loc[result["Model"] == name, "PassAllRules"] = mixed_model_passes(
            candidate,
            baseline,
            yearly_comparison,
        )
    return result


def _write_report(comparison: pd.DataFrame, yearly: pd.DataFrame) -> None:
    comparison.to_csv(REPORT_CSV_PATH, index=False, encoding="utf-8-sig")
    yearly.to_csv(YEARLY_CSV_PATH, index=False, encoding="utf-8-sig")

    passing = comparison.loc[comparison["PassAllRules"], "Model"].tolist()
    recommendation = passing[0] if passing else "Compact_F"
    lines = [
        "# V1.5 Compact Factor Experiment",
        "",
        "- Frequency: monthly",
        "- Portfolio: fixed Top 30 long-only",
        "- No EMA smoothing",
        "",
        "## Comparison",
        "",
        comparison.round(4).to_markdown(index=False),
        "",
        "## Yearly Top-30 Sharpe",
        "",
        yearly.round(4).to_markdown(index=False),
        "",
        "## Decision",
        "",
        f"- Recommended model under approved rules: **{recommendation}**",
        (
            "- Mixed models passing every rule: "
            + (", ".join(passing) if passing else "None")
        ),
        "",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_compact_experiment(
    output_dir: Path = COMPACT_MODEL_DIR,
    skip_training: bool = False,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not skip_training:
        for config in COMPACT_CONFIGS.values():
            panel = prepare_panel_for_config(config)
            train_single_model(config, panel, output_dir)

    predictions = {
        name: _load_predictions(output_dir / f"{name}_oos.parquet")
        for name in COMPACT_CONFIGS
    }
    panel = _load_evaluation_panel()
    comparison, yearly = evaluate_compact_predictions(predictions, panel)
    comparison = _add_selection_verdicts(comparison, yearly)
    _write_report(comparison, yearly)
    return comparison


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=COMPACT_MODEL_DIR)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.dry_run:
        for name, config in COMPACT_CONFIGS.items():
            print(
                f"{name}: {len(config.feature_neutral_z)} features | "
                f"colsample={config.colsample_bytree:.2f} | "
                f"GS={'ON' if config.gs_enabled else 'OFF'}"
            )
        print(f"output={args.output_dir}")
        return
    result = run_compact_experiment(
        output_dir=args.output_dir,
        skip_training=args.skip_training,
    )
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
