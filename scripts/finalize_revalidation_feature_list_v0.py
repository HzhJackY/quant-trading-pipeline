from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QA_DIR = ROOT / "output" / "transformed_panel_qa_review_v0"
BUILD_DIR = ROOT / "output" / "transformed_training_panel_v0"
OUT_DIR = ROOT / "output" / "finalized_revalidation_feature_list_v0"

QA_SUMMARY_PATH = QA_DIR / "transformed_panel_qa_review_summary.json"
WATCH_REVIEW_PATH = QA_DIR / "watch_feature_review.csv"
CONSTANT_REVIEW_PATH = QA_DIR / "constant_feature_review.csv"
COVERAGE_REVIEW_PATH = QA_DIR / "feature_coverage_review.csv"
LEAKAGE_REVIEW_PATH = QA_DIR / "feature_leakage_review.csv"
QA_FEATURE_LIST_CSV = QA_DIR / "model_feature_list_for_revalidation_v0.csv"
QA_FEATURE_LIST_JSON = QA_DIR / "model_feature_list_for_revalidation_v0.json"
QA_REPORT_PATH = QA_DIR / "transformed_panel_qa_review_report.md"
BUILD_FEATURE_LIST_CSV = BUILD_DIR / "model_feature_list_v0.csv"
BUILD_FEATURE_LIST_JSON = BUILD_DIR / "model_feature_list_v0.json"
BUILD_SUMMARY_PATH = BUILD_DIR / "transformed_training_panel_summary.json"

REQUIRED_INPUTS = [
    QA_SUMMARY_PATH,
    WATCH_REVIEW_PATH,
    CONSTANT_REVIEW_PATH,
    COVERAGE_REVIEW_PATH,
    LEAKAGE_REVIEW_PATH,
    QA_FEATURE_LIST_CSV,
    QA_FEATURE_LIST_JSON,
    QA_REPORT_PATH,
    BUILD_FEATURE_LIST_CSV,
    BUILD_FEATURE_LIST_JSON,
    BUILD_SUMMARY_PATH,
]

MANUAL_EXCLUDED_FEATURES = ["trading_status_z"]
MANUAL_REASON = (
    "manually excluded; trading_status is categorical/audit-like and zscore is not "
    "economically meaningful as default alpha feature"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def missing_input_report(missing: list[Path]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Missing Input Report",
        "",
        "The finalized feature list was not generated because required whitelisted inputs are missing.",
        "",
        "## Missing files",
        "",
    ]
    lines.extend(f"- {p.as_posix()}" for p in missing)
    (OUT_DIR / "missing_input_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "run_timestamp": now_iso(),
        "final_decision": "FINALIZED_FEATURE_LIST_FAIL_BLOCK_REVALIDATION",
        "missing_inputs": [str(p) for p in missing],
    }
    write_json(OUT_DIR / "finalized_feature_list_summary.json", summary)


def parse_offenders(leakage_df: pd.DataFrame) -> set[str]:
    offenders: set[str] = set()
    if "offending_features" not in leakage_df.columns:
        return offenders
    for value in leakage_df["offending_features"].dropna().astype(str):
        for item in value.split(";"):
            item = item.strip()
            if item:
                offenders.add(item)
    return offenders


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        missing_input_report(missing)
        print(f"Missing required inputs: {len(missing)}")
        return 0

    qa_summary = load_json(QA_SUMMARY_PATH)
    build_summary = load_json(BUILD_SUMMARY_PATH)
    qa_df = pd.read_csv(QA_FEATURE_LIST_CSV)
    build_df = pd.read_csv(BUILD_FEATURE_LIST_CSV)
    watch_df = pd.read_csv(WATCH_REVIEW_PATH)
    constant_df = pd.read_csv(CONSTANT_REVIEW_PATH)
    leakage_df = pd.read_csv(LEAKAGE_REVIEW_PATH)

    required_cols = {
        "feature_name",
        "source_factor",
        "transform_type",
        "factor_group",
        "review_status",
        "qa_recommendation",
        "included_by_default",
        "reason",
    }
    missing_cols = sorted(required_cols - set(qa_df.columns))
    if missing_cols:
        raise ValueError(f"QA feature list missing columns: {missing_cols}")

    manual_watch = set(
        watch_df.loc[
            watch_df.get("recommended_action", pd.Series(dtype=str)).astype(str).eq("NEED_MANUAL_REVIEW"),
            "feature_name",
        ].astype(str)
    )

    previous_final_decision_ok = (
        qa_summary.get("final_decision") == "TRANSFORMED_PANEL_QA_WATCH_MANUAL_FEATURE_REVIEW_REQUIRED"
    )
    qa_preconditions_ok = all(
        [
            previous_final_decision_ok,
            qa_summary.get("leakage_detected") is False,
            qa_summary.get("severe_leakage_detected") is False,
            int(qa_summary.get("rank_range_violation_count", -1)) == 0,
            int(qa_summary.get("infinite_value_count", -1)) == 0,
            "trading_status_z" in manual_watch,
        ]
    )

    build_meta = build_df[["feature_name"]].copy()
    if "direction" in build_df.columns:
        build_meta["direction"] = build_df["direction"]
    else:
        build_meta["direction"] = ""
    if "notes" in build_df.columns:
        build_meta["notes"] = build_df["notes"]
    else:
        build_meta["notes"] = ""

    finalized = qa_df.merge(build_meta, on="feature_name", how="left")
    for col in ["direction", "notes"]:
        if col not in finalized.columns:
            finalized[col] = ""
        finalized[col] = finalized[col].fillna("")

    previous_default_feature_count = int(finalized["included_by_default"].apply(normalize_bool).sum())

    excluded_decisions: list[dict[str, Any]] = []
    for feature in MANUAL_EXCLUDED_FEATURES:
        mask = finalized["feature_name"].astype(str).eq(feature)
        if mask.any():
            previous_status = ";".join(finalized.loc[mask, "qa_recommendation"].astype(str).unique())
            finalized.loc[mask, "included_by_default"] = False
            finalized.loc[mask, "qa_recommendation"] = "EXCLUDE_FROM_REVALIDATION"
            finalized.loc[mask, "reason"] = MANUAL_REASON
            excluded_decisions.append(
                {
                    "feature_name": feature,
                    "previous_status": previous_status,
                    "final_action": "EXCLUDE_FROM_REVALIDATION",
                    "reason": MANUAL_REASON,
                    "manual_decision": True,
                }
            )
        else:
            excluded_decisions.append(
                {
                    "feature_name": feature,
                    "previous_status": "MISSING_FROM_INPUT_FEATURE_LIST",
                    "final_action": "NEED_MANUAL_REVIEW",
                    "reason": "manual exclusion target was not present in input feature list",
                    "manual_decision": True,
                }
            )

    finalized["included_by_default"] = finalized["included_by_default"].apply(normalize_bool)

    constant_features = set(constant_df.get("feature_name", pd.Series(dtype=str)).dropna().astype(str))
    leakage_features = parse_offenders(leakage_df)
    forbidden_exact = {
        "selected_report_period",
        "selected_pit_date",
        "market_cap_trade_date",
    }
    forbidden_substrings = [
        "selected_report_period",
        "selected_pit_date",
        "market_cap_trade_date",
        "pit_date",
        "production",
        "live",
        "holding",
        "prediction",
        "pred_",
    ]

    default_df = finalized[finalized["included_by_default"]].copy()
    default_features = set(default_df["feature_name"].astype(str))
    default_lower = {f: f.lower() for f in default_features}

    constant_in_default = sorted(default_features & constant_features)
    leakage_in_default = sorted(default_features & leakage_features)
    excluded_status_in_default = sorted(
        default_df.loc[default_df["review_status"].astype(str).str.upper().eq("EXCLUDE"), "feature_name"].astype(str)
    )
    raw_component_in_default = sorted(
        default_df.loc[
            default_df["review_status"].astype(str).str.upper().isin(["RAW_COMPONENT_ONLY", "RAW_COMPONENT"]),
            "feature_name",
        ].astype(str)
    )
    forbidden_in_default = sorted(
        f
        for f, low in default_lower.items()
        if f in forbidden_exact or any(part in low for part in forbidden_substrings)
    )
    trading_status_z_excluded = "trading_status_z" not in default_features

    safety_failures = {
        "constant_in_default": constant_in_default,
        "leakage_in_default": leakage_in_default,
        "excluded_status_in_default": excluded_status_in_default,
        "raw_component_in_default": raw_component_in_default,
        "forbidden_in_default": forbidden_in_default,
    }
    has_safety_failures = any(bool(v) for v in safety_failures.values())

    finalized_default_feature_count = int(finalized["included_by_default"].sum())

    columns = [
        "feature_name",
        "source_factor",
        "transform_type",
        "factor_group",
        "review_status",
        "qa_recommendation",
        "included_by_default",
        "direction",
        "reason",
        "notes",
    ]
    finalized[columns].to_csv(OUT_DIR / "finalized_model_feature_list_v0.csv", index=False)
    write_json(OUT_DIR / "finalized_model_feature_list_v0.json", finalized[columns].to_dict(orient="records"))

    excluded_df = pd.DataFrame(excluded_decisions)
    excluded_df.to_csv(OUT_DIR / "excluded_feature_decisions_v0.csv", index=False)

    leakage_detected = bool(qa_summary.get("leakage_detected"))
    severe_leakage_detected = bool(qa_summary.get("severe_leakage_detected"))
    if (
        qa_preconditions_ok
        and trading_status_z_excluded
        and not leakage_detected
        and not severe_leakage_detected
        and not has_safety_failures
    ):
        final_decision = "FINALIZED_FEATURE_LIST_READY_FOR_COMPACT_F_REVALIDATION_PREP"
        recommended_next_step = "Compact-F Revalidation Prep."
    elif not trading_status_z_excluded or has_safety_failures or leakage_detected or severe_leakage_detected:
        final_decision = "FINALIZED_FEATURE_LIST_FAIL_BLOCK_REVALIDATION"
        recommended_next_step = "Fix finalized feature list safety failures before revalidation prep."
    else:
        final_decision = "FINALIZED_FEATURE_LIST_WATCH_REVIEW_REQUIRED"
        recommended_next_step = "Review QA preconditions before Compact-F Revalidation Prep."

    summary = {
        "run_timestamp": now_iso(),
        "input_feature_list_used": str(QA_FEATURE_LIST_CSV.relative_to(ROOT)),
        "finalized_feature_list_path": str((OUT_DIR / "finalized_model_feature_list_v0.csv").relative_to(ROOT)),
        "previous_default_feature_count": previous_default_feature_count,
        "finalized_default_feature_count": finalized_default_feature_count,
        "manually_excluded_features": MANUAL_EXCLUDED_FEATURES,
        "trading_status_z_excluded": trading_status_z_excluded,
        "constant_features_excluded_from_default": len(constant_in_default) == 0,
        "leakage_detected": leakage_detected,
        "severe_leakage_detected": severe_leakage_detected,
        "production_modified": False,
        "v3_modified": False,
        "transformed_panel_modified": False,
        "training_run": False,
        "backtest_run": False,
        "ic_calculated": False,
        "neutralization_executed": False,
        "qa_preconditions_ok": qa_preconditions_ok,
        "safety_failures": safety_failures,
        "source_build_training_run": bool(build_summary.get("training_run")),
        "source_build_backtest_run": bool(build_summary.get("backtest_run")),
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    write_json(OUT_DIR / "finalized_feature_list_summary.json", summary)

    excluded_lines = []
    for item in excluded_decisions:
        excluded_lines.append(f"- {item['feature_name']}: {item['final_action']} - {item['reason']}")
    if not excluded_lines:
        excluded_lines.append("- None")

    report_lines = [
        "# Finalized Revalidation Feature List v0",
        "",
        "## 1. Scope",
        "",
        "This run only finalizes the revalidation feature list. It does not train, backtest, calculate IC, modify production, modify v3, rebuild the transformed panel, or execute neutralization.",
        "",
        "## 2. Previous QA Result",
        "",
        f"- Previous QA decision: {qa_summary.get('final_decision')}",
        f"- Previous default feature count: {previous_default_feature_count}",
        f"- Leakage detected: {leakage_detected}",
        f"- Severe leakage detected: {severe_leakage_detected}",
        "",
        "## 3. Manual Decision",
        "",
        f"- trading_status_z is excluded from the default revalidation feature list.",
        f"- Reason: {MANUAL_REASON}",
        "",
        "## 4. Final Feature List",
        "",
        f"- Finalized default feature count: {finalized_default_feature_count}",
        "",
        "## 5. Excluded Features",
        "",
        *excluded_lines,
        "",
        "## 6. Safety Checks",
        "",
        f"- trading_status_z excluded: {trading_status_z_excluded}",
        f"- Constant features excluded from default: {len(constant_in_default) == 0}",
        f"- Leakage/audit/forbidden features in default: {has_safety_failures}",
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
    (OUT_DIR / "finalized_feature_list_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    task_card = [
        "# Task Completion Card",
        "",
        "- task_name: Finalize Revalidation Feature List v0",
        f"- completed_at: {now_iso()}",
        f"- final_decision: {final_decision}",
        f"- output_dir: {OUT_DIR.relative_to(ROOT).as_posix()}",
    ]
    (OUT_DIR / "task_completion_card.md").write_text("\n".join(task_card) + "\n", encoding="utf-8")
    terminal_summary = {
        "script": "scripts/finalize_revalidation_feature_list_v0.py",
        "status": "completed",
        "stdout_log": "output/_agent_runs/finalize_revalidation_feature_list_v0/run_stdout.txt",
        "stderr_log": "output/_agent_runs/finalize_revalidation_feature_list_v0/run_stderr.txt",
        "final_decision": final_decision,
    }
    write_json(OUT_DIR / "terminal_summary.json", terminal_summary)
    pd.DataFrame([summary]).to_csv(OUT_DIR / "final_qa.csv", index=False)

    del qa_df, build_df, watch_df, constant_df, leakage_df, finalized
    gc.collect()
    print(f"final_decision={final_decision}")
    print(f"output_dir={OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
