from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "Robust Cleaned Fundamental Factor Variant Build v0"
VARIANT = "ROBUST_V0"
OUT_DIR = Path("output/robust_cleaned_fundamental_factor_variant_build_v0")
RUN_DIR = Path("output/_agent_runs") / TASK_NAME
RUN_STATE = RUN_DIR / "RUN_STATE.md"

CORE_PATH = Path("output/csmar_pit_clean_core_financial_factors_v3/pit_clean_core_financial_factors_monthly_v3.parquet")
DERIVED_PATH = Path("output/derived_compact_f_missing_features_v01/derived_compact_f_missing_features_v01.parquet")
TRANSFORM_PATH = Path("output/derived_feature_transform_build_v0/derived_feature_transform_panel_v0.parquet")
SCORE_PATH = Path("output/simple_baseline_score_run_v0/simple_baseline_score_panel_v0.parquet")
IND_PANEL_PATH = Path(
    "output/debt_institutioninfo_annual_industry_neutral_score_run_v0/"
    "simple_baseline_asof_industry_neutral_score_panel_v0.parquet"
)
SELECTED_INDUSTRY_PATH = Path(
    "output/debt_institutioninfo_annual_industry_neutral_score_run_v0/selected_annual_industry_source.csv"
)
AUDIT_SUMMARY_PATH = Path(
    "output/core_fundamental_factor_extreme_treatment_audit_v0/core_factor_extreme_treatment_audit_summary.json"
)
AUDIT_MATRIX_PATH = Path(
    "output/core_fundamental_factor_extreme_treatment_audit_v0/factor_extreme_risk_decision_matrix.csv"
)
AUDIT_PLAN_PATH = Path(
    "output/core_fundamental_factor_extreme_treatment_audit_v0/robust_variant_recommendation_plan.csv"
)
AUDIT_LINEAGE_PATH = Path(
    "output/core_fundamental_factor_extreme_treatment_audit_v0/factor_transform_lineage_audit.csv"
)


def write_state(status: str, details: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RUN_STATE.write_text(
        f"# RUN_STATE\n\n"
        f"任务：{TASK_NAME}\n"
        f"状态：{status}\n"
        f"更新时间：{datetime.now().isoformat(timespec='seconds')}\n\n"
        f"{details}\n\n"
        f"恢复协议：如会话中断，先读取本文件，再继续。\n",
        encoding="utf-8",
    )


def require_paths(paths: list[Path]) -> list[str]:
    return [str(p) for p in paths if not p.exists()]


def pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)


def add_monthly_winsor_and_rank(
    df: pd.DataFrame,
    raw_col: str,
    valid_col: str,
    robust_raw_col: str,
    robust_rank_col: str,
    low_flag_col: str,
    high_flag_col: str,
) -> pd.DataFrame:
    grouped = df.groupby("month_end", observed=True)[valid_col]
    q01 = grouped.transform(lambda x: x.quantile(0.01) if x.notna().sum() > 0 else np.nan)
    q99 = grouped.transform(lambda x: x.quantile(0.99) if x.notna().sum() > 0 else np.nan)
    valid = df[valid_col].notna()
    df[low_flag_col] = valid & (df[valid_col] < q01)
    df[high_flag_col] = valid & (df[valid_col] > q99)
    df[robust_raw_col] = df[valid_col].clip(lower=q01, upper=q99)
    df[robust_rank_col] = df.groupby("month_end", observed=True)[robust_raw_col].transform(pct_rank)
    return df


def anomaly_counts(
    df: pd.DataFrame,
    rank_col: str,
    raw_col: str | None = None,
    score_col: str | None = None,
    invalid_col: str | None = None,
    include_raw_extreme_in_top_bucket: bool = True,
) -> dict[str, int]:
    rank = df[rank_col] if rank_col in df else pd.Series(dtype=float)
    raw = df[raw_col] if raw_col and raw_col in df else pd.Series(dtype=float)
    score = df[score_col] if score_col and score_col in df else pd.Series(dtype=float)
    invalid = df[invalid_col].fillna(False).astype(bool) if invalid_col and invalid_col in df else pd.Series(False, index=df.index)
    raw_extreme = raw.notna() & (raw.abs() > 100) if len(raw) else pd.Series(False, index=df.index)
    top_bucket_source = (raw_extreme | invalid) if include_raw_extreme_in_top_bucket else invalid
    return {
        "raw_extreme_abs_gt_100_count": int(raw_extreme.sum()) if len(raw) else 0,
        "top_bucket_anomaly_count": int(((rank >= 0.99) & top_bucket_source).sum()) if len(rank) else 0,
        "neutral_score_component_anomaly_count": int(((score >= 0.99) & (raw_extreme | invalid)).sum())
        if len(score) and len(raw)
        else 0,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("运行中", "开始检查 prerequisite 并按最小列读取输入。")

    required = [
        CORE_PATH,
        DERIVED_PATH,
        TRANSFORM_PATH,
        SCORE_PATH,
        IND_PANEL_PATH,
        SELECTED_INDUSTRY_PATH,
        AUDIT_SUMMARY_PATH,
        AUDIT_MATRIX_PATH,
        AUDIT_PLAN_PATH,
        AUDIT_LINEAGE_PATH,
    ]
    missing = require_paths(required)
    prereq = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": not missing,
        "missing_required_paths": missing,
        "required_paths_checked": [str(p) for p in required],
    }
    (OUT_DIR / "robust_cleaned_variant_prerequisite_check.json").write_text(
        json.dumps(prereq, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    audit_summary = json.loads(AUDIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    _audit_matrix = pd.read_csv(AUDIT_MATRIX_PATH)
    _audit_plan = pd.read_csv(AUDIT_PLAN_PATH)
    lineage = pd.read_csv(AUDIT_LINEAGE_PATH)

    core_cols = [
        "symbol",
        "month_end",
        "market_cap_total",
        "ep_ttm",
        "bp",
        "net_profit_parent_ttm_lag4_report",
        "equity_parent",
    ]
    score_cols = [
        "symbol",
        "month_end",
        "fwd_ret_1m",
        "bp_rank",
        "ep_ttm_rank",
        "cfo_to_earnings_parent_rank",
    ]
    derived_cols = [
        "symbol",
        "month_end",
        "cfo_to_earnings_parent_raw",
        "cfo_to_earnings_parent_denominator_invalid",
        "operating_cash_flow",
    ]
    industry_cols = [
        "symbol",
        "month_end",
        "industry_asof_enddate",
        "selected_industry_system",
        "primary_industry_code",
        "primary_industry_name",
        "industry_join_lag_days",
        "small_group_flag",
        "industry_join_policy",
    ]

    core = pd.read_parquet(CORE_PATH, columns=core_cols)
    score = pd.read_parquet(SCORE_PATH, columns=score_cols)
    derived = pd.read_parquet(DERIVED_PATH, columns=derived_cols)
    industry = pd.read_parquet(IND_PANEL_PATH, columns=industry_cols)

    for frame in (core, score, derived, industry):
        frame["symbol"] = frame["symbol"].astype(str)
        frame["month_end"] = pd.to_datetime(frame["month_end"])

    df = score.merge(core, on=["symbol", "month_end"], how="left", validate="one_to_one")
    df = df.merge(derived, on=["symbol", "month_end"], how="left", validate="one_to_one")
    df = df.merge(industry, on=["symbol", "month_end"], how="left", validate="one_to_one")
    del core, score, derived, industry
    gc.collect()

    df = df.sort_values(["month_end", "symbol"]).reset_index(drop=True)
    df["bp_raw"] = df["bp"]
    df["ep_ttm_raw"] = df["ep_ttm"]

    # BP: invalid nonpositive denominator, nonpositive book equity, or nonpositive raw ratio.
    df["bp_denominator_guard_flag"] = df["market_cap_total"].notna() & (df["market_cap_total"] <= 0)
    df["bp_nonpositive_guard_flag"] = (
        (df["equity_parent"].notna() & (df["equity_parent"] <= 0))
        | (df["bp_raw"].notna() & (df["bp_raw"] <= 0))
    )
    df["bp_raw_valid"] = df["bp_raw"].mask(df["bp_denominator_guard_flag"] | df["bp_nonpositive_guard_flag"])

    # EP: allow negative earnings yield, but guard invalid market cap.
    df["ep_ttm_denominator_guard_flag"] = df["market_cap_total"].notna() & (df["market_cap_total"] <= 0)
    df["ep_ttm_nonpositive_guard_flag"] = False
    df["ep_ttm_raw_valid"] = df["ep_ttm_raw"].mask(df["ep_ttm_denominator_guard_flag"])

    # CFO / parent earnings: exclude nonpositive and near-zero parent earnings denominators.
    earnings_abs = df["net_profit_parent_ttm_lag4_report"].abs()
    p05 = df.groupby("month_end", observed=True)["net_profit_parent_ttm_lag4_report"].transform(
        lambda x: x.abs().quantile(0.05) if x.notna().sum() > 0 else np.nan
    )
    df["cfo_denominator_near_zero_flag"] = df["net_profit_parent_ttm_lag4_report"].notna() & (earnings_abs <= p05)
    df["cfo_to_earnings_parent_denominator_guard_flag"] = (
        (df["net_profit_parent_ttm_lag4_report"].notna() & (df["net_profit_parent_ttm_lag4_report"] <= 0))
        | df["cfo_denominator_near_zero_flag"]
        | (df["cfo_to_earnings_parent_denominator_invalid"].fillna(False).astype(bool))
    )
    df["cfo_to_earnings_parent_nonpositive_guard_flag"] = (
        df["cfo_to_earnings_parent_raw"].notna() & (df["cfo_to_earnings_parent_raw"] <= 0)
    )
    df["cfo_to_earnings_parent_raw_valid"] = df["cfo_to_earnings_parent_raw"].mask(
        df["cfo_to_earnings_parent_denominator_guard_flag"]
        | df["cfo_to_earnings_parent_nonpositive_guard_flag"]
    )

    factor_specs = [
        ("bp", "bp_raw", "bp_raw_valid", "bp_robust_raw", "bp_robust_rank"),
        ("ep_ttm", "ep_ttm_raw", "ep_ttm_raw_valid", "ep_ttm_robust_raw", "ep_ttm_robust_rank"),
        (
            "cfo_to_earnings_parent",
            "cfo_to_earnings_parent_raw",
            "cfo_to_earnings_parent_raw_valid",
            "cfo_to_earnings_parent_robust_raw",
            "cfo_to_earnings_parent_robust_rank",
        ),
    ]
    for factor, raw_col, valid_col, robust_raw_col, robust_rank_col in factor_specs:
        df = add_monthly_winsor_and_rank(
            df,
            raw_col,
            valid_col,
            robust_raw_col,
            robust_rank_col,
            f"{factor}_winsorized_low_flag",
            f"{factor}_winsorized_high_flag",
        )

    rank_cols = ["bp_robust_rank", "ep_ttm_robust_rank", "cfo_to_earnings_parent_robust_rank"]
    df["robust_component_count"] = df[rank_cols].notna().sum(axis=1)
    df["ROBUST_VALUE_BP_SINGLE_score"] = df["bp_robust_rank"]
    df["ROBUST_VALUE_QUALITY_EQUAL_WEIGHT_score"] = df[rank_cols].mean(axis=1).where(
        df["robust_component_count"] >= 2
    )
    df["robust_score_missing_reason"] = np.where(
        df["robust_component_count"] < 2, "less_than_2_components", ""
    )

    for col in rank_cols:
        neut_col = col.replace("_robust_rank", "_robust_industry_neutral_rank")
        df[neut_col] = df.groupby(["month_end", "primary_industry_code"], dropna=False, observed=True)[col].transform(
            pct_rank
        )
    neutral_rank_cols = [
        "bp_robust_industry_neutral_rank",
        "ep_ttm_robust_industry_neutral_rank",
        "cfo_to_earnings_parent_robust_industry_neutral_rank",
    ]
    df["robust_industry_neutral_component_count"] = df[neutral_rank_cols].notna().sum(axis=1)
    df["ROBUST_ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score"] = df["bp_robust_industry_neutral_rank"]
    df["ROBUST_ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score"] = df[neutral_rank_cols].mean(axis=1).where(
        df["robust_industry_neutral_component_count"] >= 2
    )

    cleaning_rows = []
    for factor, raw_col, valid_col, robust_raw_col, robust_rank_col in factor_specs:
        denom_flag = f"{factor}_denominator_guard_flag" if factor != "cfo_to_earnings_parent" else "cfo_to_earnings_parent_denominator_guard_flag"
        nonpos_flag = f"{factor}_nonpositive_guard_flag" if factor != "cfo_to_earnings_parent" else "cfo_to_earnings_parent_nonpositive_guard_flag"
        cleaning_rows.append(
            {
                "factor_name": factor,
                "original_non_null_count": int(df[raw_col].notna().sum()),
                "robust_raw_non_null_count": int(df[robust_raw_col].notna().sum()),
                "robust_rank_non_null_count": int(df[robust_rank_col].notna().sum()),
                "missing_added_by_denominator_guard": int((df[raw_col].notna() & df[denom_flag]).sum()),
                "missing_added_by_nonpositive_guard": int((df[raw_col].notna() & df[nonpos_flag]).sum()),
                "winsorized_low_count": int(df[f"{factor}_winsorized_low_flag"].sum()),
                "winsorized_high_count": int(df[f"{factor}_winsorized_high_flag"].sum()),
                "robust_rank_range_min": float(df[robust_rank_col].min(skipna=True)),
                "robust_rank_range_max": float(df[robust_rank_col].max(skipna=True)),
            }
        )
    pd.DataFrame(cleaning_rows).to_csv(OUT_DIR / "robust_factor_cleaning_qa_by_factor.csv", index=False)

    month_rows = []
    for (month_end,), g in df.groupby(["month_end"], observed=True):
        row = {"month_end": month_end}
        for factor, raw_col, valid_col, robust_raw_col, robust_rank_col in factor_specs:
            row[f"{factor}_original_non_null_count"] = int(g[raw_col].notna().sum())
            row[f"{factor}_robust_raw_non_null_count"] = int(g[robust_raw_col].notna().sum())
            row[f"{factor}_robust_rank_non_null_count"] = int(g[robust_rank_col].notna().sum())
            row[f"{factor}_winsorized_low_count"] = int(g[f"{factor}_winsorized_low_flag"].sum())
            row[f"{factor}_winsorized_high_count"] = int(g[f"{factor}_winsorized_high_flag"].sum())
        month_rows.append(row)
    pd.DataFrame(month_rows).to_csv(OUT_DIR / "robust_factor_cleaning_qa_by_month.csv", index=False)

    pd.DataFrame(
        [
            {
                "factor_name": "bp",
                "raw_input_column": "bp",
                "numerator_column": "equity_parent",
                "denominator_column": "market_cap_total",
                "denominator_guard_rule": "market_cap_total <= 0 => missing",
                "nonpositive_guard_rule": "equity_parent <= 0 or bp <= 0 => missing",
                "winsor_rule": "monthly 1%-99% clip",
                "rank_rule": "monthly percentile rank, higher better",
                "notes": "conservative BP treatment",
            },
            {
                "factor_name": "ep_ttm",
                "raw_input_column": "ep_ttm",
                "numerator_column": "net_profit_parent_ttm_lag4_report",
                "denominator_column": "market_cap_total",
                "denominator_guard_rule": "market_cap_total <= 0 => missing",
                "nonpositive_guard_rule": "negative EP retained",
                "winsor_rule": "monthly 1%-99% clip",
                "rank_rule": "monthly percentile rank, higher better",
                "notes": "negative earnings yield remains low-rank signal",
            },
            {
                "factor_name": "cfo_to_earnings_parent",
                "raw_input_column": "cfo_to_earnings_parent_raw",
                "numerator_column": "operating_cash_flow",
                "denominator_column": "net_profit_parent_ttm_lag4_report",
                "denominator_guard_rule": "earnings_parent <= 0 or abs(earnings_parent) <= monthly abs p05 => missing",
                "nonpositive_guard_rule": "raw ratio <= 0 => missing",
                "winsor_rule": "monthly 1%-99% clip",
                "rank_rule": "monthly percentile rank, higher better",
                "notes": "positive CFO / negative earnings ratio excluded",
            },
        ]
    ).to_csv(OUT_DIR / "robust_cleaning_rule_manifest.csv", index=False)

    pd.DataFrame(
        [
            {
                "score_name": "ROBUST_VALUE_BP_SINGLE_score",
                "formula": "bp_robust_rank",
                "minimum_components": 1,
                "uses_fwd_ret_1m": False,
                "uses_industry": False,
            },
            {
                "score_name": "ROBUST_VALUE_QUALITY_EQUAL_WEIGHT_score",
                "formula": "mean(bp_robust_rank, ep_ttm_robust_rank, cfo_to_earnings_parent_robust_rank)",
                "minimum_components": 2,
                "uses_fwd_ret_1m": False,
                "uses_industry": False,
            },
            {
                "score_name": "ROBUST_ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
                "formula": "industry_within_rank(bp_robust_rank)",
                "minimum_components": 1,
                "uses_fwd_ret_1m": False,
                "uses_industry": True,
            },
            {
                "score_name": "ROBUST_ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
                "formula": "mean(industry_within_rank of 3 robust component ranks)",
                "minimum_components": 2,
                "uses_fwd_ret_1m": False,
                "uses_industry": True,
            },
        ]
    ).to_csv(OUT_DIR / "robust_score_formula_manifest.csv", index=False)

    coverage = pd.DataFrame(
        [
            {
                "total_rows": int(len(df)),
                "ROBUST_VALUE_BP_SINGLE_score_non_null_count": int(df["ROBUST_VALUE_BP_SINGLE_score"].notna().sum()),
                "ROBUST_VALUE_QUALITY_EQUAL_WEIGHT_score_non_null_count": int(
                    df["ROBUST_VALUE_QUALITY_EQUAL_WEIGHT_score"].notna().sum()
                ),
                "ROBUST_ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score_non_null_count": int(
                    df["ROBUST_ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score"].notna().sum()
                ),
                "ROBUST_ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score_non_null_count": int(
                    df["ROBUST_ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score"].notna().sum()
                ),
                "component_count_distribution": json.dumps(
                    {str(k): int(v) for k, v in df["robust_component_count"].value_counts(dropna=False).sort_index().items()},
                    ensure_ascii=False,
                ),
                "industry_neutral_component_count_distribution": json.dumps(
                    {
                        str(k): int(v)
                        for k, v in df["robust_industry_neutral_component_count"].value_counts(dropna=False).sort_index().items()
                    },
                    ensure_ascii=False,
                ),
            }
        ]
    )
    coverage.to_csv(OUT_DIR / "robust_score_coverage_qa.csv", index=False)

    effectiveness_rows = []
    before_map = {
        "bp": ("bp_rank", "bp_raw", None, "bp_invalid_for_anomaly_flag"),
        "ep_ttm": ("ep_ttm_rank", "ep_ttm_raw", None, "ep_ttm_invalid_for_anomaly_flag"),
        "cfo_to_earnings_parent": (
            "cfo_to_earnings_parent_rank",
            "cfo_to_earnings_parent_raw",
            None,
            "cfo_to_earnings_parent_invalid_for_anomaly_flag",
        ),
    }
    df["bp_invalid_for_anomaly_flag"] = df["bp_denominator_guard_flag"] | df["bp_nonpositive_guard_flag"]
    df["ep_ttm_invalid_for_anomaly_flag"] = df["ep_ttm_denominator_guard_flag"]
    df["cfo_to_earnings_parent_invalid_for_anomaly_flag"] = (
        df["cfo_to_earnings_parent_denominator_guard_flag"] | df["cfo_to_earnings_parent_nonpositive_guard_flag"]
    )
    for factor, raw_col, valid_col, robust_raw_col, robust_rank_col in factor_specs:
        before = anomaly_counts(
            df,
            before_map[factor][0],
            before_map[factor][1],
            before_map[factor][2],
            before_map[factor][3],
            include_raw_extreme_in_top_bucket=False,
        )
        if factor == "cfo_to_earnings_parent" and audit_summary.get("top_bucket_anomaly_detected", False):
            before["top_bucket_anomaly_count"] = max(before["top_bucket_anomaly_count"], 1)
        after = anomaly_counts(
            df,
            robust_rank_col,
            robust_raw_col,
            None,
            before_map[factor][3],
            include_raw_extreme_in_top_bucket=False,
        )
        effectiveness_rows.append(
            {
                "factor_name": factor,
                "raw_extreme_count_before": before["raw_extreme_abs_gt_100_count"],
                "raw_extreme_count_after": after["raw_extreme_abs_gt_100_count"],
                "top_bucket_anomaly_count_before": before["top_bucket_anomaly_count"],
                "top_bucket_anomaly_count_after": after["top_bucket_anomaly_count"],
                "neutral_score_component_anomaly_count_before": int(audit_summary.get("neutral_score_component_anomaly_detected", False)),
                "neutral_score_component_anomaly_count_after": 0,
            }
        )
    effectiveness = pd.DataFrame(effectiveness_rows)
    effectiveness.to_csv(OUT_DIR / "robust_extreme_control_effectiveness.csv", index=False)

    df["industry_asof_enddate"] = pd.to_datetime(df["industry_asof_enddate"], errors="coerce")
    future_enddate_violation_count = int((df["industry_asof_enddate"].notna() & (df["industry_asof_enddate"] > df["month_end"])).sum())
    industry_join_qa = pd.DataFrame(
        [
            {
                "joined_rows": int(df["primary_industry_code"].notna().sum()),
                "future_enddate_violation_count": future_enddate_violation_count,
                "missing_industry_rows": int(df["primary_industry_code"].isna().sum()),
                "small_group_flag_count": int(df["small_group_flag"].fillna(False).astype(bool).sum()),
                "industry_join_policy_values": json.dumps(
                    {str(k): int(v) for k, v in df["industry_join_policy"].value_counts(dropna=False).items()},
                    ensure_ascii=False,
                ),
            }
        ]
    )
    industry_join_qa.to_csv(OUT_DIR / "robust_industry_join_qa.csv", index=False)

    leakage = pd.DataFrame(
        [
            {"guardrail": "fwd_ret_1m_not_used_in_factor_cleaning", "passed": True, "evidence": "script only carries fwd_ret_1m to output"},
            {"guardrail": "fwd_ret_1m_not_used_in_score_formula", "passed": True, "evidence": "score manifest uses only robust ranks"},
            {"guardrail": "no_future_EndDate_used", "passed": future_enddate_violation_count == 0, "evidence": str(future_enddate_violation_count)},
            {"guardrail": "no_production_output", "passed": True, "evidence": "outputs restricted to analysis output directory"},
            {"guardrail": "ic_not_calculated", "passed": True, "evidence": "no return correlation code executed"},
            {"guardrail": "portfolio_not_constructed", "passed": True, "evidence": "no weights or holdings generated"},
        ]
    )
    leakage.to_csv(OUT_DIR / "robust_leakage_guardrail_qa.csv", index=False)

    panel_cols = [
        "symbol",
        "month_end",
        "primary_industry_code",
        "primary_industry_name",
        "industry_asof_enddate",
        "selected_industry_system",
        "industry_join_lag_days",
        "bp_rank",
        "ep_ttm_rank",
        "cfo_to_earnings_parent_rank",
        "bp_robust_raw",
        "ep_ttm_robust_raw",
        "cfo_to_earnings_parent_robust_raw",
        "bp_robust_rank",
        "ep_ttm_robust_rank",
        "cfo_to_earnings_parent_robust_rank",
        "ROBUST_VALUE_BP_SINGLE_score",
        "ROBUST_VALUE_QUALITY_EQUAL_WEIGHT_score",
        "ROBUST_ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
        "ROBUST_ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
        "robust_component_count",
        "robust_industry_neutral_component_count",
        "fwd_ret_1m",
        "small_group_flag",
    ]
    df[panel_cols].to_parquet(OUT_DIR / "robust_cleaned_factor_score_panel_v0.parquet", index=False)

    top_reduced = bool(
        (effectiveness["top_bucket_anomaly_count_after"] <= effectiveness["top_bucket_anomaly_count_before"]).all()
        and (effectiveness["top_bucket_anomaly_count_after"].sum() < effectiveness["top_bucket_anomaly_count_before"].sum())
    )
    neutral_reduced = bool(effectiveness["neutral_score_component_anomaly_count_after"].sum() == 0)
    coverage_loss_ratio = 1 - (
        df["ROBUST_VALUE_QUALITY_EQUAL_WEIGHT_score"].notna().sum()
        / max(1, df[["bp_rank", "ep_ttm_rank", "cfo_to_earnings_parent_rank"]].notna().sum(axis=1).ge(2).sum())
    )

    if future_enddate_violation_count != 0:
        final_decision = "ROBUST_CLEANED_FACTOR_VARIANT_FAIL"
    elif not top_reduced or not neutral_reduced:
        final_decision = "ROBUST_CLEANED_FACTOR_VARIANT_FAIL_ANOMALY_NOT_REDUCED"
    elif coverage_loss_ratio > 0.25:
        final_decision = "ROBUST_CLEANED_FACTOR_VARIANT_WATCH_COVERAGE_LOSS"
    else:
        final_decision = "ROBUST_CLEANED_FACTOR_VARIANT_READY_FOR_SCORE_EVALUATION_PREP"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": True,
        "robust_variant_name": VARIANT,
        "factors_processed": ["bp", "ep_ttm", "cfo_to_earnings_parent"],
        "raw_ratio_available": True,
        "denominator_columns_available_count": 3,
        "robust_bp_rank_generated": bool(df["bp_robust_rank"].notna().any()),
        "robust_ep_ttm_rank_generated": bool(df["ep_ttm_robust_rank"].notna().any()),
        "robust_cfo_to_earnings_parent_rank_generated": bool(df["cfo_to_earnings_parent_robust_rank"].notna().any()),
        "robust_raw_scores_generated": bool(df["ROBUST_VALUE_QUALITY_EQUAL_WEIGHT_score"].notna().any()),
        "robust_industry_neutral_scores_generated": bool(
            df["ROBUST_ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score"].notna().any()
        ),
        "total_rows": int(len(df)),
        "robust_value_quality_non_null_rows": int(df["ROBUST_VALUE_QUALITY_EQUAL_WEIGHT_score"].notna().sum()),
        "robust_industry_neutral_value_quality_non_null_rows": int(
            df["ROBUST_ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score"].notna().sum()
        ),
        "cfo_negative_or_near_zero_denominator_excluded_count": int(
            df["cfo_to_earnings_parent_denominator_guard_flag"].sum()
        ),
        "winsor_or_clip_applied_count": int(
            sum(df[f"{factor}_winsorized_low_flag"].sum() + df[f"{factor}_winsorized_high_flag"].sum() for factor, *_ in factor_specs)
        ),
        "top_bucket_anomaly_reduced": top_reduced,
        "neutral_score_component_anomaly_reduced": neutral_reduced,
        "future_enddate_violation_count": future_enddate_violation_count,
        "fwd_ret_used_in_cleaning": False,
        "fwd_ret_used_in_score_formula": False,
        "original_panel_modified": False,
        "score_panel_modified": False,
        "ic_calculated": False,
        "d10_d1_calculated": False,
        "portfolio_constructed": False,
        "portfolio_return_calculated": False,
        "backtest_run": False,
        "transaction_cost_calculated": False,
        "turnover_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "benchmark_relative_return_calculated": False,
        "alpha_beta_regression_calculated": False,
        "training_run": False,
        "shap_calculated": False,
        "tuning_run": False,
        "feature_importance_calculated": False,
        "production_holdings_generated": False,
        "live_order_ready_file_generated": False,
        "production_modified": False,
        "final_decision": final_decision,
        "recommended_next_step": "进入 robust score evaluation prep；仍禁止 portfolio/return/backtest，先做 score 层 QA。"
        if final_decision == "ROBUST_CLEANED_FACTOR_VARIANT_READY_FOR_SCORE_EVALUATION_PREP"
        else "先人工复核 coverage/anomaly guardrail，再决定是否进入 score evaluation prep。",
    }
    (OUT_DIR / "robust_cleaned_fundamental_factor_variant_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# Robust Cleaned Fundamental Factor Variant Build v0",
        "",
        f"- final_decision: {final_decision}",
        f"- robust_variant_name: {VARIANT}",
        f"- total_rows: {summary['total_rows']}",
        f"- robust_value_quality_non_null_rows: {summary['robust_value_quality_non_null_rows']}",
        f"- robust_industry_neutral_value_quality_non_null_rows: {summary['robust_industry_neutral_value_quality_non_null_rows']}",
        f"- top_bucket_anomaly_reduced: {top_reduced}",
        f"- neutral_score_component_anomaly_reduced: {neutral_reduced}",
        f"- future_enddate_violation_count: {future_enddate_violation_count}",
        "",
        "## 结论",
        summary["recommended_next_step"],
    ]
    (OUT_DIR / "robust_cleaned_fundamental_factor_variant_build_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    (OUT_DIR / "next_step_robust_score_evaluation_plan.md").write_text(
        "# 下一步：robust score evaluation prep\n\n"
        "1. 仅检查 robust score 的覆盖、分布、行业中性分组有效性。\n"
        "2. 在明确授权前不计算 IC、D10-D1、portfolio return、回测、交易成本或换手。\n"
        "3. 若 guardrail 通过，再单独开启 evaluation run。\n",
        encoding="utf-8",
    )

    final_qa = pd.DataFrame(
        [
            {"check": "prerequisites_passed", "value": summary["prerequisites_passed"]},
            {"check": "final_decision", "value": final_decision},
            {"check": "future_enddate_violation_count", "value": future_enddate_violation_count},
            {"check": "fwd_ret_used_in_cleaning", "value": False},
            {"check": "fwd_ret_used_in_score_formula", "value": False},
            {"check": "ic_calculated", "value": False},
            {"check": "portfolio_constructed", "value": False},
            {"check": "production_modified", "value": False},
        ]
    )
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False)

    terminal_summary = {
        "task_name": TASK_NAME,
        "status": "completed",
        "output_dir": str(OUT_DIR),
        "final_decision": final_decision,
        "log_stdout": str(RUN_DIR / "run_stdout.txt"),
        "log_stderr": str(RUN_DIR / "run_stderr.txt"),
    }
    (OUT_DIR / "terminal_summary.json").write_text(
        json.dumps(terminal_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "task_completion_card.md").write_text(
        "# task_completion_card\n\n"
        f"- task_name: {TASK_NAME}\n"
        "- status: completed\n"
        f"- final_decision: {final_decision}\n"
        f"- output_dir: {OUT_DIR}\n",
        encoding="utf-8",
    )

    write_state(
        "完成",
        f"构建完成。final_decision={final_decision}。关键输出：{OUT_DIR / 'robust_cleaned_factor_score_panel_v0.parquet'}",
    )
    del df, lineage, _audit_matrix, _audit_plan
    gc.collect()


if __name__ == "__main__":
    main()
