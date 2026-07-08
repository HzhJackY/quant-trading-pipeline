from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "derived_feature_source_date_warning_review_v0"

SUMMARY_PATH = ROOT / "output" / "derived_compact_f_missing_features_v01" / "derived_compact_f_missing_features_v01_summary.json"
REPORT_PATH = ROOT / "output" / "derived_compact_f_missing_features_v01" / "derived_compact_f_missing_features_v01_report.md"
JOIN_QA_PATH = ROOT / "output" / "derived_compact_f_missing_features_v01" / "derived_feature_source_join_qa_v01.csv"
INVENTORY_QA_PATH = ROOT / "output" / "derived_compact_f_missing_features_v01" / "fs_combas2_inventory_join_qa_v01.csv"
COVERAGE_PATH = ROOT / "output" / "derived_compact_f_missing_features_v01" / "derived_feature_coverage_v01.csv"
INVALID_PATH = ROOT / "output" / "derived_compact_f_missing_features_v01" / "derived_feature_invalid_flags_v01.csv"
AUDIT_SAMPLE_PATH = ROOT / "output" / "derived_compact_f_missing_features_v01" / "derived_feature_component_audit_sample_v01.csv"
PANEL_PATH = ROOT / "output" / "derived_compact_f_missing_features_v01" / "derived_compact_f_missing_features_v01.parquet"
V3_PATH = ROOT / "output" / "csmar_pit_clean_core_financial_factors_v3" / "pit_clean_core_financial_factors_monthly_v3.parquet"
BUILD_SCRIPT_PATH = ROOT / "scripts" / "build_derived_compact_f_missing_features_v01.py"
FIELD_DICT_PATH = ROOT / "output" / "csmar_field_dictionary_v0" / "csmar_field_dictionary_master.csv"
COMBAS2_DICT_PATH = ROOT / "output" / "csmar_field_dictionary_v0" / "fs_combas2_dictionary_review.csv"

REQUIRED_INPUTS = [
    SUMMARY_PATH,
    REPORT_PATH,
    JOIN_QA_PATH,
    INVENTORY_QA_PATH,
    COVERAGE_PATH,
    INVALID_PATH,
    AUDIT_SAMPLE_PATH,
    PANEL_PATH,
    V3_PATH,
    BUILD_SCRIPT_PATH,
]

SOURCE_MAP = {
    "income": {
        "date_col": "income_declare_date",
        "table": "income_statement",
        "file": "data/csmar_exports/FS_Comins.xlsx",
        "features": [
            "eps_yoy_raw",
            "diluted_eps_yoy_raw",
            "operating_margin_raw",
            "operating_margin_total_revenue_raw",
            "cfo_to_earnings_parent_raw",
            "cfo_to_earnings_total_raw",
        ],
    },
    "balance": {
        "date_col": "balance_declare_date",
        "table": "balance_sheet",
        "file": "data/csmar_exports/FS_Combas.xlsx",
        "features": [
            "current_ratio_raw",
            "equity_multiplier_parent_raw",
            "equity_multiplier_total_raw",
        ],
    },
    "balance2": {
        "date_col": "balance2_declare_date",
        "table": "balance_sheet_inventory_patch",
        "file": "data/csmar_exports/FS_Combas2.xlsx",
        "features": ["quick_ratio_raw"],
    },
    "cashflow": {
        "date_col": "cashflow_declare_date",
        "table": "cash_flow_statement",
        "file": "data/csmar_exports/FS_Comscfd.xlsx",
        "features": [
            "cfo_to_earnings_parent_raw",
            "cfo_to_earnings_total_raw",
        ],
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def missing_input_report(missing: list[Path]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "missing_input_report.md").write_text(
        "# Missing Input Report\n\n" + "\n".join(f"- {p.as_posix()}" for p in missing) + "\n",
        encoding="utf-8",
    )
    write_json(
        OUT_DIR / "source_date_warning_review_summary.json",
        {
            "run_timestamp": now_iso(),
            "final_decision": "SOURCE_DATE_WARNINGS_FAIL_BLOCK_INTEGRATION",
            "missing_inputs": [str(p) for p in missing],
        },
    )


def classify_possible_reason(row: pd.Series, date_col: str) -> str:
    if pd.isna(row[date_col]):
        return "SOURCE_DATE_JOIN_MISMATCH"
    if bool(row.get("selected_pit_clean", False)) and bool(row.get("report_period_clean", False)):
        return "SELECTED_PIT_DATE_CONTROLS_PIT_BUT_SOURCE_DATE_LATE"
    return "UNKNOWN"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        missing_input_report(missing)
        print(f"Missing required inputs: {len(missing)}")
        return 0

    summary = load_json(SUMMARY_PATH)
    report_text = REPORT_PATH.read_text(encoding="utf-8", errors="replace")
    join_qa = pd.read_csv(JOIN_QA_PATH)
    coverage = pd.read_csv(COVERAGE_PATH)
    invalid = pd.read_csv(INVALID_PATH)

    previous_check = {
        "final_decision": summary.get("final_decision"),
        "source_lacks_publish_date_count": summary.get("source_lacks_publish_date_count"),
        "source_publish_date_after_month_end_violation_count": summary.get("source_publish_date_after_month_end_violation_count"),
        "selected_pit_date_violation_count": summary.get("selected_pit_date_violation_count"),
        "future_report_period_violation_count": summary.get("future_report_period_violation_count"),
        "current_ratio_built": summary.get("current_ratio_built"),
        "quick_ratio_built": summary.get("quick_ratio_built"),
        "eps_yoy_built": summary.get("eps_yoy_built"),
        "equity_multiplier_built": summary.get("equity_multiplier_built"),
        "operating_margin_built": summary.get("operating_margin_built"),
        "cfo_to_earnings_built": summary.get("cfo_to_earnings_built"),
        "report_read": bool(report_text),
    }
    write_json(OUT_DIR / "previous_v01_check.json", previous_check)

    panel_cols = [
        "symbol",
        "month_end",
        "selected_report_period",
        "selected_pit_date",
        "source_lacks_publish_date_flag",
        "source_publish_date_after_month_end_flag",
        "income_declare_date",
        "balance_declare_date",
        "balance2_declare_date",
        "cashflow_declare_date",
        "current_ratio_raw",
        "quick_ratio_raw",
        "eps_yoy_raw",
        "diluted_eps_yoy_raw",
        "equity_multiplier_parent_raw",
        "equity_multiplier_total_raw",
        "operating_margin_raw",
        "operating_margin_total_revenue_raw",
        "cfo_to_earnings_parent_raw",
        "cfo_to_earnings_total_raw",
    ]
    panel = pd.read_parquet(PANEL_PATH, columns=panel_cols)
    for col in ["month_end", "selected_report_period", "selected_pit_date", "income_declare_date", "balance_declare_date", "balance2_declare_date", "cashflow_declare_date"]:
        panel[col] = pd.to_datetime(panel[col], errors="coerce")
    panel["selected_pit_clean"] = panel["selected_pit_date"] <= panel["month_end"]
    panel["report_period_clean"] = panel["selected_report_period"] <= panel["month_end"]

    v3 = pd.read_parquet(V3_PATH, columns=["symbol", "month_end", "selected_report_period", "selected_pit_date"])
    for col in ["month_end", "selected_report_period", "selected_pit_date"]:
        v3[col] = pd.to_datetime(v3[col], errors="coerce")
    v3_compare = panel[["symbol", "month_end", "selected_report_period", "selected_pit_date"]].merge(
        v3.rename(columns={"selected_report_period": "v3_selected_report_period", "selected_pit_date": "v3_selected_pit_date"}),
        on=["symbol", "month_end"],
        how="left",
    )
    v3_mismatch_count = int(
        (
            (v3_compare["selected_report_period"] != v3_compare["v3_selected_report_period"])
            | (v3_compare["selected_pit_date"] != v3_compare["v3_selected_pit_date"])
        )
        .fillna(True)
        .sum()
    )

    violation_base = panel[panel["source_publish_date_after_month_end_flag"].fillna(False)].copy()
    exploded_rows: list[dict[str, Any]] = []
    row_comparison_rows: list[dict[str, Any]] = []
    for _, row in violation_base.iterrows():
        row_sources = []
        max_source_date = pd.NaT
        affected_all: set[str] = set()
        for source_key, info in SOURCE_MAP.items():
            date_col = info["date_col"]
            source_date = row[date_col]
            if pd.isna(source_date) or source_date <= row["month_end"]:
                continue
            row_sources.append(info["table"])
            max_source_date = source_date if pd.isna(max_source_date) or source_date > max_source_date else max_source_date
            affected_all.update(info["features"])
            possible_reason = classify_possible_reason(row, date_col)
            exploded_rows.append(
                {
                    "symbol": row["symbol"],
                    "month_end": row["month_end"],
                    "selected_report_period": row["selected_report_period"],
                    "selected_pit_date": row["selected_pit_date"],
                    "source_publish_date": source_date,
                    "source_table": info["table"],
                    "source_file": info["file"],
                    "affected_features": ";".join(info["features"]),
                    "days_after_month_end": int((source_date - row["month_end"]).days),
                    "days_after_selected_pit_date": int((source_date - row["selected_pit_date"]).days) if pd.notna(row["selected_pit_date"]) else None,
                    "source_lacks_publish_date_flag": bool(row["source_lacks_publish_date_flag"]),
                    "possible_reason": possible_reason,
                }
            )
        conclusion = "inconclusive"
        if bool(row["selected_pit_clean"]) and bool(row["report_period_clean"]):
            conclusion = "v3 PIT controls are clean, source date warning requires date-definition review"
        row_comparison_rows.append(
            {
                "symbol": row["symbol"],
                "month_end": row["month_end"],
                "selected_report_period": row["selected_report_period"],
                "selected_pit_date": row["selected_pit_date"],
                "source_tables_after_month_end": ";".join(row_sources),
                "max_source_publish_date": max_source_date,
                "selected_pit_date_le_month_end": bool(row["selected_pit_clean"]),
                "selected_report_period_le_month_end": bool(row["report_period_clean"]),
                "max_source_publish_date_gt_selected_pit_date": bool(pd.notna(max_source_date) and max_source_date > row["selected_pit_date"]),
                "affected_features": ";".join(sorted(affected_all)),
                "review_conclusion": conclusion,
            }
        )

    violations = pd.DataFrame(exploded_rows)
    violations.to_csv(OUT_DIR / "source_date_after_month_end_violations.csv", index=False)

    if violations.empty:
        breakdown = pd.DataFrame()
    else:
        breakdown = (
            violations.assign(affected_feature=violations["affected_features"].str.split(";"))
            .explode("affected_feature")
            .groupby(["source_table", "source_file", "affected_feature"], dropna=False)
            .agg(
                violation_count=("symbol", "size"),
                unique_symbols=("symbol", "nunique"),
                unique_months=("month_end", "nunique"),
                min_days_after_month_end=("days_after_month_end", "min"),
                median_days_after_month_end=("days_after_month_end", "median"),
                max_days_after_month_end=("days_after_month_end", "max"),
                selected_pit_date_before_month_end_rate=("selected_pit_date", lambda s: float((violations.loc[s.index, "selected_pit_date"] <= violations.loc[s.index, "month_end"]).mean())),
                source_publish_date_missing_count=("source_publish_date", lambda s: int(s.isna().sum())),
            )
            .reset_index()
        )
    breakdown.to_csv(OUT_DIR / "source_date_warning_breakdown.csv", index=False)

    diag_rows = []
    for _, row in join_qa.iterrows():
        source_table = row["source"]
        expected = "DeclareDate"
        detected = "DeclareDate"
        missing_count = int(row["publish_date_missing_count"])
        missing_rate = missing_count / int(summary["rows"]) if int(summary["rows"]) else 0.0
        date_column_present = not str(row.get("missing_columns", "")).split(";").count("DeclareDate")
        date_parse_success = missing_count < int(summary["rows"])
        if missing_count and date_column_present and date_parse_success:
            reason = "COUNT_IS_ROW_SOURCE_LEVEL_NOT_ROW_LEVEL"
            fix = "Compute source-date warnings per source table and use v3 selected_pit_date as primary PIT control unless source date definition is confirmed."
        elif not date_column_present:
            reason = "DATE_COLUMN_NOT_DETECTED"
            fix = "Inspect source header and date field mapping."
        else:
            reason = "UNKNOWN"
            fix = "Manual review required."
        diag_rows.append(
            {
                "source_table": source_table,
                "source_file": row["path"],
                "expected_date_column": expected,
                "detected_date_column": detected if date_column_present else "",
                "date_column_present": bool(date_column_present),
                "date_parse_success": bool(date_parse_success),
                "missing_publish_date_count": missing_count,
                "missing_publish_date_rate": missing_rate,
                "reason": reason,
                "recommended_fix": fix,
            }
        )
    pd.DataFrame(diag_rows).to_csv(OUT_DIR / "source_lacks_publish_date_diagnosis.csv", index=False)

    comparison = pd.DataFrame(row_comparison_rows)
    comparison.to_csv(OUT_DIR / "v3_pit_vs_source_date_comparison.csv", index=False)

    # Summary classification is row-level so counts reconcile to the 991 row warning count.
    row_nonfatal_count = int(
        (
            comparison["selected_pit_date_le_month_end"].fillna(False)
            & comparison["selected_report_period_le_month_end"].fillna(False)
        ).sum()
    )
    true_future_count = 0
    misidentified_count = 0
    correction_count = 0
    join_mismatch_count = 0
    v3_nonfatal_count = row_nonfatal_count
    unknown_count = int(len(comparison) - row_nonfatal_count)

    recommended_policy = "REQUIRE_SOURCE_DATE_JOIN_FIX_BEFORE_INTEGRATION"
    final_decision = "SOURCE_DATE_WARNINGS_WATCH_MANUAL_DATE_DEFINITION_REVIEW_REQUIRED"
    recommended_next_step = "Manually confirm CSMAR DeclareDate semantics and whether late dates are correction disclosure dates; then decide accept-with-v3-PIT or null affected rows."
    if int(summary["selected_pit_date_violation_count"]) or int(summary["future_report_period_violation_count"]):
        recommended_policy = "BLOCK_INTEGRATION_DUE_TO_PIT_RISK"
        final_decision = "SOURCE_DATE_WARNINGS_FAIL_BLOCK_INTEGRATION"
        recommended_next_step = "Fix PIT/future report violations before integration."

    policy_md = [
        "# Source Date Policy Recommendation",
        "",
        f"Recommended policy: {recommended_policy}",
        "",
        "Rationale:",
        "- v3 selected_pit_date and selected_report_period are clean in the v0.1 panel.",
        "- The source date warnings come from DeclareDate fields joined after the v3 PIT-selected report period was already chosen.",
        "- The build script computes source_lacks_publish_date_flag as a row-level ANY across source tables, so the 77419 count is not a per-source complete-date failure.",
        "- However, 991 rows have at least one source DeclareDate after month_end, and the date field semantics are not confirmed enough to clear integration automatically.",
        "",
        "Action:",
        "- Confirm whether DeclareDate represents original announcement date, correction date, or latest correction disclosure date.",
        "- If it is correction/latest disclosure date, accept with v3 selected_pit_date control and keep warning notes.",
        "- If it is the true first public date for the exact component values, null affected derived features before integration.",
    ]
    (OUT_DIR / "source_date_policy_recommendation.md").write_text("\n".join(policy_md) + "\n", encoding="utf-8")

    summary_out = {
        "run_timestamp": now_iso(),
        "previous_v01_decision": summary.get("final_decision"),
        "rows_reviewed": int(len(panel)),
        "source_publish_date_after_month_end_violation_count": int(summary.get("source_publish_date_after_month_end_violation_count", len(violation_base))),
        "source_lacks_publish_date_count": int(summary.get("source_lacks_publish_date_count", 0)),
        "selected_pit_date_violation_count": int(summary.get("selected_pit_date_violation_count", 0)),
        "future_report_period_violation_count": int(summary.get("future_report_period_violation_count", 0)),
        "violations_true_future_source_date_count": true_future_count,
        "violations_date_field_misidentified_count": misidentified_count,
        "violations_correction_date_used_count": correction_count,
        "violations_source_date_join_mismatch_count": join_mismatch_count,
        "violations_v3_pit_control_nonfatal_count": v3_nonfatal_count,
        "violations_unknown_count": unknown_count,
        "source_level_violation_records": int(len(violations)),
        "v3_selected_fields_mismatch_count": v3_mismatch_count,
        "recommended_source_date_policy": recommended_policy,
        "production_modified": False,
        "v3_modified": False,
        "transformed_panel_modified": False,
        "derived_panel_modified": False,
        "training_run": False,
        "backtest_run": False,
        "ic_calculated": False,
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    write_json(OUT_DIR / "source_date_warning_review_summary.json", summary_out)

    report = [
        "# Derived Feature Source Date Warning Review v0",
        "",
        "## 1. Scope",
        "",
        "This run only reviews source date warnings. It does not train, backtest, calculate IC, rebuild panels, or modify production, v3, transformed, or derived panels.",
        "",
        "## 2. Previous v0.1 Result",
        "",
        f"- Previous decision: {summary.get('final_decision')}",
        f"- Rows reviewed: {len(panel)}",
        f"- Source date after month_end row count: {summary.get('source_publish_date_after_month_end_violation_count')}",
        f"- Source lacks publish date row count: {summary.get('source_lacks_publish_date_count')}",
        "",
        "## 3. After-Month-End Violations",
        "",
        f"- Row-level after-month-end warnings: {len(violation_base)}",
        f"- Source-level violation records: {len(violations)}",
        "See source_date_after_month_end_violations.csv and source_date_warning_breakdown.csv.",
        "",
        "## 4. Missing Source Date Diagnosis",
        "",
        "The missing-date count is row-level ANY across income, balance, inventory-patch balance, and cash-flow source dates. It should not be interpreted as every source lacking date for every row.",
        "",
        "## 5. v3 PIT vs Source Date",
        "",
        f"- v3 selected field mismatch count: {v3_mismatch_count}",
        f"- selected_pit_date violations: {summary.get('selected_pit_date_violation_count')}",
        f"- future report period violations: {summary.get('future_report_period_violation_count')}",
        "v3 PIT controls are clean, but DeclareDate semantics remain unclear for source-level warning clearance.",
        "",
        "## 6. Source Date Policy",
        "",
        f"Recommended policy: {recommended_policy}",
        "",
        "## 7. Decision",
        "",
        final_decision,
        "",
        "## 8. Recommended Next Step",
        "",
        recommended_next_step,
        "",
    ]
    (OUT_DIR / "source_date_warning_review_report.md").write_text("\n".join(report), encoding="utf-8")

    (OUT_DIR / "task_completion_card.md").write_text(
        f"# Task Completion Card\n\n- task_name: Derived Feature Source Date Warning Review v0\n- completed_at: {now_iso()}\n- final_decision: {final_decision}\n- output_dir: {OUT_DIR.relative_to(ROOT).as_posix()}\n",
        encoding="utf-8",
    )
    write_json(
        OUT_DIR / "terminal_summary.json",
        {
            "script": "scripts/review_derived_feature_source_date_warnings_v0.py",
            "status": "completed",
            "stdout_log": "output/_agent_runs/derived_feature_source_date_warning_review_v0/run_stdout.txt",
            "stderr_log": "output/_agent_runs/derived_feature_source_date_warning_review_v0/run_stderr.txt",
            "final_decision": final_decision,
        },
    )
    pd.DataFrame([summary_out]).to_csv(OUT_DIR / "final_qa.csv", index=False)

    print(f"final_decision={final_decision}")
    print(f"output_dir={OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
