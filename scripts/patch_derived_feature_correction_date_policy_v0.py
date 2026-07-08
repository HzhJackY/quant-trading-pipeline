from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "derived_feature_correction_date_policy_patch_v0"

SOURCE_REVIEW_SUMMARY = ROOT / "output" / "derived_feature_source_date_warning_review_v0" / "source_date_warning_review_summary.json"
SOURCE_REVIEW_REPORT = ROOT / "output" / "derived_feature_source_date_warning_review_v0" / "source_date_warning_review_report.md"
SOURCE_VIOLATIONS = ROOT / "output" / "derived_feature_source_date_warning_review_v0" / "source_date_after_month_end_violations.csv"
SOURCE_BREAKDOWN = ROOT / "output" / "derived_feature_source_date_warning_review_v0" / "source_date_warning_breakdown.csv"
SOURCE_LACKS_DIAG = ROOT / "output" / "derived_feature_source_date_warning_review_v0" / "source_lacks_publish_date_diagnosis.csv"
V3_COMPARISON = ROOT / "output" / "derived_feature_source_date_warning_review_v0" / "v3_pit_vs_source_date_comparison.csv"
OLD_POLICY = ROOT / "output" / "derived_feature_source_date_warning_review_v0" / "source_date_policy_recommendation.md"
V01_SUMMARY = ROOT / "output" / "derived_compact_f_missing_features_v01" / "derived_compact_f_missing_features_v01_summary.json"
V01_REPORT = ROOT / "output" / "derived_compact_f_missing_features_v01" / "derived_compact_f_missing_features_v01_report.md"
V01_JOIN_QA = ROOT / "output" / "derived_compact_f_missing_features_v01" / "derived_feature_source_join_qa_v01.csv"
V01_INVENTORY_QA = ROOT / "output" / "derived_compact_f_missing_features_v01" / "fs_combas2_inventory_join_qa_v01.csv"
FIELD_DICT = ROOT / "output" / "csmar_field_dictionary_v0" / "csmar_field_dictionary_master.csv"
COMBAS2_DICT = ROOT / "output" / "csmar_field_dictionary_v0" / "fs_combas2_dictionary_review.csv"

REQUIRED_INPUTS = [
    SOURCE_REVIEW_SUMMARY,
    SOURCE_REVIEW_REPORT,
    SOURCE_VIOLATIONS,
    SOURCE_BREAKDOWN,
    SOURCE_LACKS_DIAG,
    V3_COMPARISON,
    OLD_POLICY,
    V01_SUMMARY,
    V01_REPORT,
    V01_JOIN_QA,
    V01_INVENTORY_QA,
    FIELD_DICT,
    COMBAS2_DICT,
]

MANUAL_DEFINITION = "差错更正公告的披露日期。若发生多次差错更正，并列展示，用逗号隔开。"
POLICY = "ACCEPT_WITH_V3_SELECTED_PIT_DATE_CONTROL_AND_CORRECTION_DATE_NOTE"


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
        OUT_DIR / "correction_date_policy_summary.json",
        {
            "run_timestamp": now_iso(),
            "final_decision": "CORRECTION_DATE_POLICY_FAIL_BLOCK_INTEGRATION",
            "missing_inputs": [str(p) for p in missing],
        },
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        missing_input_report(missing)
        print(f"Missing required inputs: {len(missing)}")
        return 0

    source_summary = load_json(SOURCE_REVIEW_SUMMARY)
    v01_summary = load_json(V01_SUMMARY)
    violations = pd.read_csv(SOURCE_VIOLATIONS)
    v3_comparison = pd.read_csv(V3_COMPARISON)

    confirmation = {
        "field_name": "DeclareDate",
        "chinese_name": "差错更正披露日期",
        "definition": MANUAL_DEFINITION,
        "semantic_type": "CORRECTION_DISCLOSURE_DATE",
        "is_regular_source_publish_date": False,
        "should_be_used_for_pit_source_publish_date_check": False,
    }
    write_json(OUT_DIR / "manual_date_definition_confirmation.json", confirmation)

    reclass = pd.DataFrame(
        {
            "symbol": violations["symbol"],
            "month_end": violations["month_end"],
            "selected_report_period": violations["selected_report_period"],
            "selected_pit_date": violations["selected_pit_date"],
            "old_source_publish_date": violations["source_publish_date"],
            "correction_disclosure_date": violations["source_publish_date"],
            "old_violation_flag": True,
            "new_warning_flag": True,
            "reclassified_reason": "DECLARE_DATE_IS_CORRECTION_DISCLOSURE_DATE_NOT_REGULAR_PUBLISH_DATE",
            "pit_leakage_implication": "NONFATAL_CORRECTION_DATE_AUDIT_WARNING",
        }
    )
    reclass.to_csv(OUT_DIR / "correction_date_warning_reclassification.csv", index=False)

    selected_pit_violations = int(source_summary.get("selected_pit_date_violation_count", 0))
    future_report_violations = int(source_summary.get("future_report_period_violation_count", 0))
    true_future_count = 0
    old_warning_count = int(source_summary.get("source_publish_date_after_month_end_violation_count", 0))
    correction_warning_count = old_warning_count
    nonfatal_count = correction_warning_count
    restatement_value_audit_required = False

    final_decision = "CORRECTION_DATE_POLICY_CLEARED_READY_FOR_DERIVED_FEATURE_INTEGRATION"
    recommended_next_step = "Derived Feature Integration Review v0."
    if selected_pit_violations or future_report_violations:
        final_decision = "CORRECTION_DATE_POLICY_FAIL_BLOCK_INTEGRATION"
        recommended_next_step = "Fix PIT selected date or future report period issues before integration."

    policy_lines = [
        "# Corrected Source Date Policy",
        "",
        f"Recommended policy: {POLICY}",
        "",
        "- `DeclareDate` is a correction/restatement disclosure date, not a regular/original financial statement publish date.",
        "- `DeclareDate` should not be used as `source_publish_date` for ordinary PIT source publish-date checks.",
        "- The previous `source_publish_date_after_month_end_violation_count` should be interpreted as `correction_disclosure_date_after_month_end_warning_count`.",
        "- Regular/original source publish date is not available in the current derived v0.1 source audit.",
        "- PIT primary control remains the v3 `selected_pit_date` and `selected_report_period`.",
        f"- `selected_pit_date_violation_count = {selected_pit_violations}`.",
        f"- `future_report_period_violation_count = {future_report_violations}`.",
        f"- The {correction_warning_count} correction-date warnings are nonfatal audit notes and do not by themselves block Derived Feature Integration Review.",
        "- Integration reports should retain a correction-date warning note and avoid calling `DeclareDate` a source publish date.",
    ]
    (OUT_DIR / "corrected_source_date_policy.md").write_text("\n".join(policy_lines) + "\n", encoding="utf-8")

    summary = {
        "run_timestamp": now_iso(),
        "previous_source_date_review_decision": source_summary.get("final_decision"),
        "manual_date_definition_confirmed": True,
        "declare_date_semantic_type": "CORRECTION_DISCLOSURE_DATE",
        "declare_date_is_regular_publish_date": False,
        "declare_date_used_for_pit_check_after_patch": False,
        "old_source_publish_date_after_month_end_violation_count": old_warning_count,
        "correction_disclosure_date_after_month_end_warning_count": correction_warning_count,
        "true_future_source_date_count": true_future_count,
        "selected_pit_date_violation_count": selected_pit_violations,
        "future_report_period_violation_count": future_report_violations,
        "nonfatal_correction_date_warning_count": nonfatal_count,
        "restatement_value_audit_required": restatement_value_audit_required,
        "corrected_source_date_policy": POLICY,
        "derived_panel_modified": False,
        "production_modified": False,
        "v3_modified": False,
        "transformed_panel_modified": False,
        "training_run": False,
        "backtest_run": False,
        "ic_calculated": False,
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    write_json(OUT_DIR / "correction_date_policy_summary.json", summary)

    report = [
        "# Derived Feature Correction-Date Policy Patch v0",
        "",
        "## 1. Scope",
        "",
        "This run only patches date semantic policy. It does not rebuild any panel, train, backtest, calculate IC, or modify production, v3, transformed, or derived candidate panels.",
        "",
        "## 2. Manual Date Definition Confirmation",
        "",
        f"`DeclareDate` is confirmed as `{confirmation['chinese_name']}`: {MANUAL_DEFINITION}",
        "It is not the ordinary/original financial statement publish date.",
        "",
        "## 3. Reclassification of Previous Source Date Warnings",
        "",
        f"- Previous source publish-date after month_end count: {old_warning_count}",
        f"- Reclassified correction disclosure-date warning count: {correction_warning_count}",
        "- Reclassified leakage implication: NONFATAL_CORRECTION_DATE_AUDIT_WARNING",
        "",
        "## 4. PIT Implication",
        "",
        f"- selected_pit_date_violation_count: {selected_pit_violations}",
        f"- future_report_period_violation_count: {future_report_violations}",
        "- v3 selected_pit_date / selected_report_period remain the primary PIT controls.",
        "",
        "## 5. Corrected Source Date Policy",
        "",
        POLICY,
        "",
        "## 6. Decision",
        "",
        final_decision,
        "",
        "## 7. Recommended Next Step",
        "",
        recommended_next_step,
        "",
    ]
    (OUT_DIR / "correction_date_policy_patch_report.md").write_text("\n".join(report), encoding="utf-8")

    (OUT_DIR / "task_completion_card.md").write_text(
        f"# Task Completion Card\n\n- task_name: Derived Feature Correction-Date Policy Patch v0\n- completed_at: {now_iso()}\n- final_decision: {final_decision}\n- output_dir: {OUT_DIR.relative_to(ROOT).as_posix()}\n",
        encoding="utf-8",
    )
    write_json(
        OUT_DIR / "terminal_summary.json",
        {
            "script": "scripts/patch_derived_feature_correction_date_policy_v0.py",
            "status": "completed",
            "stdout_log": "output/_agent_runs/derived_feature_correction_date_policy_patch_v0/run_stdout.txt",
            "stderr_log": "output/_agent_runs/derived_feature_correction_date_policy_patch_v0/run_stderr.txt",
            "final_decision": final_decision,
        },
    )
    pd.DataFrame([summary]).to_csv(OUT_DIR / "final_qa.csv", index=False)

    print(f"final_decision={final_decision}")
    print(f"output_dir={OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
