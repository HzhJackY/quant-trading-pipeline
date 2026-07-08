from __future__ import annotations

import csv
import gc
import json
from datetime import datetime
from pathlib import Path


TASK_NAME = "simple_fundamental_baseline_prep_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

STOP_PIVOT_DIR = ROOT / "output" / "compact_f_v3_full_stop_pivot_decision_memo_v0"
PRICE_LABEL_DIR = ROOT / "output" / "compact_f_v3_full_training_panel_price_label_v0"
SPLIT_DIR = ROOT / "output" / "compact_f_v3_full_training_config_freeze_v0"

INPUTS = {
    "stop_pivot_summary": STOP_PIVOT_DIR / "compact_f_v3_full_stop_pivot_decision_summary.json",
    "seed_list": STOP_PIVOT_DIR / "simple_baseline_candidate_seed_list.csv",
    "do_not_continue": STOP_PIVOT_DIR / "compact_f_v3_full_do_not_continue_list.csv",
    "stop_pivot_memo": STOP_PIVOT_DIR / "stop_pivot_decision_memo.md",
    "price_label_summary": PRICE_LABEL_DIR / "compact_f_v3_full_training_panel_price_label_summary.json",
    "frozen_split_config": SPLIT_DIR / "frozen_split_config.csv",
    "training_panel_future_path": PRICE_LABEL_DIR / "compact_f_v3_full_training_panel_price_label_unique13_v0.parquet",
}

BASELINE_FEATURES = [
    "bp_rank",
    "ep_ttm_rank",
    "cfo_to_earnings_parent_rank",
    "roe_ttm_rank",
    "profit_growth_yoy_rank",
]

EXPECTED_DIRECTIONS = {feature: "POSITIVE" for feature in BASELINE_FEATURES}

REQUIRED_PROHIBITIONS = [
    "continue Compact-F-v3-full rescue",
    "sign-flip production/backtest",
    "duplicate-weighted logical15",
    "current_ratio / quick_ratio as core alpha without separate validation",
    "direct backtest prep from failed branch",
    "LightGBM first",
    "SHAP first",
    "tuning first",
]

ALLOWED_NEXT_STEPS = [
    "Single-factor evaluation prep",
    "Single-factor IC / Rank IC run",
    "Single-factor decile diagnostics",
    "Simple equal-weight score prep",
    "Manual baseline candidate selection",
]

FINAL_READY = "SIMPLE_FUNDAMENTAL_BASELINE_PREP_READY_FOR_SINGLE_FACTOR_EVALUATION_PREP"
FINAL_WATCH = "SIMPLE_FUNDAMENTAL_BASELINE_PREP_WATCH_REVIEW_REQUIRED"
FINAL_FAIL = "SIMPLE_FUNDAMENTAL_BASELINE_PREP_FAIL"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def bool_str(value: bool) -> str:
    return "true" if value else "false"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    input_status = {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "is_file": path.is_file(),
        }
        for name, path in INPUTS.items()
    }
    missing_inputs = [name for name, status in input_status.items() if not status["is_file"]]

    if missing_inputs:
        final_decision = FINAL_FAIL
        prerequisite_checks: list[dict] = []
        stop_summary = {}
        price_summary = {}
        seed_rows = []
        do_not_rows = []
        split_rows = []
        memo_text = ""
    else:
        stop_summary = read_json(INPUTS["stop_pivot_summary"])
        seed_rows = read_csv_rows(INPUTS["seed_list"])
        do_not_rows = read_csv_rows(INPUTS["do_not_continue"])
        memo_text = read_text(INPUTS["stop_pivot_memo"])
        price_summary = read_json(INPUTS["price_label_summary"])
        split_rows = read_csv_rows(INPUTS["frozen_split_config"])

        memo_features = stop_summary.get("candidate_baseline_features", [])
        seed_features = [row.get("feature_name", "") for row in seed_rows]
        expected_from_seed = {row.get("feature_name", ""): row.get("expected_direction", "") for row in seed_rows}

        branch_stopped = bool(stop_summary.get("branch_stopped")) and "formally stopped" in memo_text
        pivot_recommended = bool(stop_summary.get("pivot_baseline_recommended"))
        stopped_compact_f_branch_confirmed = branch_stopped and not bool(stop_summary.get("branch_continue_recommended"))
        feature_seed_match = memo_features == BASELINE_FEATURES and seed_features == BASELINE_FEATURES
        direction_match = all(expected_from_seed.get(feature) == EXPECTED_DIRECTIONS[feature] for feature in BASELINE_FEATURES)
        leakage_pass = price_summary.get("leakage_guard_status") == "PASS"
        target_window_pass = price_summary.get("target_window_status") == "PASS"
        split_config_present = len(split_rows) > 0
        parquet_path_recorded_only = input_status["training_panel_future_path"]["is_file"]
        imported_block_count = len(do_not_rows)
        prohibited_continuations_imported = imported_block_count > 0

        prerequisite_checks = [
            {
                "check": "all_required_input_files_present",
                "status": "PASS",
                "detail": "All required small memo/config files and future parquet path exist.",
            },
            {
                "check": "stop_pivot_branch_stopped",
                "status": "PASS" if stopped_compact_f_branch_confirmed else "FAIL",
                "detail": "Compact-F-v3-full branch_stopped=true and branch_continue_recommended=false.",
            },
            {
                "check": "pivot_baseline_recommended",
                "status": "PASS" if pivot_recommended else "FAIL",
                "detail": "Stop/Pivot memo recommends simple baseline prep.",
            },
            {
                "check": "baseline_seed_features_match_policy",
                "status": "PASS" if feature_seed_match else "FAIL",
                "detail": ",".join(BASELINE_FEATURES),
            },
            {
                "check": "expected_directions_match_policy",
                "status": "PASS" if direction_match else "FAIL",
                "detail": "All seed features are POSITIVE expected direction.",
            },
            {
                "check": "price_label_leakage_guard_pass",
                "status": "PASS" if leakage_pass else "FAIL",
                "detail": str(price_summary.get("leakage_guard_status")),
            },
            {
                "check": "price_label_target_window_pass",
                "status": "PASS" if target_window_pass else "FAIL",
                "detail": str(price_summary.get("target_window_status")),
            },
            {
                "check": "frozen_split_config_present",
                "status": "PASS" if split_config_present else "FAIL",
                "detail": f"split_rows={len(split_rows)}",
            },
            {
                "check": "training_panel_parquet_recorded_not_read",
                "status": "PASS" if parquet_path_recorded_only else "FAIL",
                "detail": str(INPUTS["training_panel_future_path"]),
            },
            {
                "check": "prohibited_continuations_imported",
                "status": "PASS" if prohibited_continuations_imported else "FAIL",
                "detail": f"imported_rows={imported_block_count}; task_policy_blocks={len(REQUIRED_PROHIBITIONS)}",
            },
        ]

        prerequisites_passed = all(row["status"] == "PASS" for row in prerequisite_checks)
        final_decision = FINAL_READY if prerequisites_passed else FINAL_WATCH

        del memo_features, seed_features, expected_from_seed
        gc.collect()

    training_panel_stat = {}
    parquet_path = INPUTS["training_panel_future_path"]
    if parquet_path.exists():
        stat = parquet_path.stat()
        training_panel_stat = {
            "path": str(parquet_path),
            "exists": True,
            "size_bytes": stat.st_size,
            "last_write_time": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
            "content_read": False,
        }
    else:
        training_panel_stat = {"path": str(parquet_path), "exists": False, "content_read": False}

    manifest_rows = []
    seed_reason_by_feature = {row.get("feature_name", ""): row.get("reason", "") for row in seed_rows}
    seed_source_by_feature = {row.get("feature_name", ""): row.get("source_from_smoke_audit", "") for row in seed_rows}
    for idx, feature in enumerate(BASELINE_FEATURES, start=1):
        manifest_rows.append(
            {
                "priority": idx,
                "feature_name": feature,
                "expected_direction": EXPECTED_DIRECTIONS[feature],
                "reason": seed_reason_by_feature.get(feature, "Simple interpretable fundamental baseline candidate."),
                "source": seed_source_by_feature.get(feature, "task_policy"),
                "include_in_first_baseline_candidate": "True",
                "training_panel_path_for_future_use_only": str(parquet_path),
                "training_panel_content_read": "False",
            }
        )
    write_csv(
        OUT_DIR / "simple_baseline_feature_seed_manifest.csv",
        manifest_rows,
        [
            "priority",
            "feature_name",
            "expected_direction",
            "reason",
            "source",
            "include_in_first_baseline_candidate",
            "training_panel_path_for_future_use_only",
            "training_panel_content_read",
        ],
    )

    direction_rows = [
        {
            "feature_name": feature,
            "expected_direction": EXPECTED_DIRECTIONS[feature],
            "score_usage_policy": "higher_rank_is_better",
            "sign_flip_allowed": "False",
            "requires_separate_validation_before_production": "True",
        }
        for feature in BASELINE_FEATURES
    ]
    write_csv(
        OUT_DIR / "simple_baseline_expected_direction_policy.csv",
        direction_rows,
        [
            "feature_name",
            "expected_direction",
            "score_usage_policy",
            "sign_flip_allowed",
            "requires_separate_validation_before_production",
        ],
    )

    eval_rows = [
        {
            "step_order": idx,
            "allowed_future_step": step,
            "status_in_this_prep_task": "NOT_RUN",
            "permitted_after_this_task": "True",
            "notes": "Prep manifest only; execute in a separate explicitly scoped task.",
        }
        for idx, step in enumerate(ALLOWED_NEXT_STEPS, start=1)
    ]
    write_csv(
        OUT_DIR / "simple_baseline_evaluation_plan.csv",
        eval_rows,
        ["step_order", "allowed_future_step", "status_in_this_prep_task", "permitted_after_this_task", "notes"],
    )

    imported_blocks = [row.get("prohibited_continuation", "") for row in do_not_rows]
    guardrail_rows = []
    for item in REQUIRED_PROHIBITIONS:
        source = "imported_or_task_policy"
        matched_import = any(item.lower().split(" ")[0] in block.lower() or block.lower() in item.lower() for block in imported_blocks)
        if item in {"LightGBM first", "SHAP first", "tuning first"}:
            matched_import = False
            source = "task_policy_extension"
        guardrail_rows.append(
            {
                "guardrail": item,
                "status": "BLOCK",
                "source": source,
                "imported_from_stop_pivot": bool_str(matched_import),
                "passed_in_this_prep": "True",
            }
        )
    no_run_flags = {
        "training run?": False,
        "backtest run?": False,
        "IC calculated?": False,
        "SHAP calculated?": False,
        "tuning run?": False,
        "holdings generated?": False,
        "production modified?": False,
    }
    for label, value in no_run_flags.items():
        guardrail_rows.append(
            {
                "guardrail": label,
                "status": "PASS" if value is False else "FAIL",
                "source": "task_execution_guardrail",
                "imported_from_stop_pivot": "false",
                "passed_in_this_prep": bool_str(value is False),
            }
        )
    write_csv(
        OUT_DIR / "simple_baseline_guardrail_checklist.csv",
        guardrail_rows,
        ["guardrail", "status", "source", "imported_from_stop_pivot", "passed_in_this_prep"],
    )

    prerequisites_passed = final_decision != FINAL_FAIL and all(row["status"] == "PASS" for row in prerequisite_checks)
    guardrail_checklist_passed = all(row["passed_in_this_prep"].lower() == "true" for row in guardrail_rows)
    stopped_compact_f_branch_confirmed = any(
        row["check"] == "stop_pivot_branch_stopped" and row["status"] == "PASS" for row in prerequisite_checks
    )
    prohibited_continuations_imported = any(
        row["check"] == "prohibited_continuations_imported" and row["status"] == "PASS" for row in prerequisite_checks
    )

    prerequisite_payload = {
        "run_timestamp": run_timestamp,
        "task_name": TASK_NAME,
        "input_status": input_status,
        "training_panel_future_path": training_panel_stat,
        "prerequisite_checks": prerequisite_checks,
        "prerequisites_passed": prerequisites_passed,
        "stopped_compact_f_branch_confirmed": stopped_compact_f_branch_confirmed,
        "prohibited_continuations_imported": prohibited_continuations_imported,
        "training_panel_parquet_content_read": False,
    }
    write_json(OUT_DIR / "simple_baseline_prep_prerequisite_check.json", prerequisite_payload)

    next_step_md = "\n".join(
        [
            "# Simple Fundamental Baseline Prep v0 - Next Step Plan",
            "",
            f"Final decision: {final_decision}",
            "",
            "## Allowed Future Sequence",
            "",
            "1. Single-factor evaluation prep",
            "2. Single-factor IC / Rank IC run",
            "3. Single-factor decile diagnostics",
            "4. Simple equal-weight score prep",
            "5. Manual baseline candidate selection",
            "",
            "## Current Task Boundary",
            "",
            "- No parquet content was read; the training panel path was recorded for future use only.",
            "- No training, backtest, IC, SHAP, tuning, holdings generation, or production modification was performed.",
            "- Compact-F-v3-full rescue and direct backtest prep from the failed branch remain blocked.",
            "",
        ]
    )
    write_text(OUT_DIR / "simple_baseline_next_step_plan.md", next_step_md)

    summary_payload = {
        "run_timestamp": run_timestamp,
        "task_name": TASK_NAME,
        "final_decision": final_decision,
        "prerequisites_passed": prerequisites_passed,
        "stopped_compact_f_branch_confirmed": stopped_compact_f_branch_confirmed,
        "candidate_feature_count": len(BASELINE_FEATURES),
        "candidate_features": BASELINE_FEATURES,
        "expected_direction_policy_generated": True,
        "evaluation_plan_generated": True,
        "guardrail_checklist_passed": guardrail_checklist_passed,
        "prohibited_continuations_imported": prohibited_continuations_imported,
        "training_run": False,
        "backtest_run": False,
        "ic_calculated": False,
        "shap_calculated": False,
        "tuning_run": False,
        "holdings_generated": False,
        "production_modified": False,
        "training_panel_parquet_content_read": False,
        "training_panel_path_for_future_use_only": str(parquet_path),
        "allowed_next_steps": ALLOWED_NEXT_STEPS,
        "required_prohibitions": REQUIRED_PROHIBITIONS,
    }
    write_json(OUT_DIR / "simple_fundamental_baseline_prep_summary.json", summary_payload)

    report_md = "\n".join(
        [
            "# Simple Fundamental Baseline Prep v0",
            "",
            f"Final decision: {final_decision}",
            "",
            "## Candidate Features",
            "",
            *[f"- {feature}: POSITIVE" for feature in BASELINE_FEATURES],
            "",
            "## Guardrails",
            "",
            "- Compact-F-v3-full rescue remains stopped.",
            "- Sign-flip production/backtest remains blocked.",
            "- Duplicate-weighted logical15, direct backtest prep, LightGBM-first, SHAP-first, and tuning-first paths remain blocked.",
            "- current_ratio / quick_ratio are not core alpha candidates without separate validation.",
            "",
            "## Execution Boundary",
            "",
            "- Training run: false",
            "- Backtest run: false",
            "- IC calculated: false",
            "- SHAP calculated: false",
            "- Tuning run: false",
            "- Holdings generated: false",
            "- Production modified: false",
            "- Training panel parquet content read: false",
            "",
        ]
    )
    write_text(OUT_DIR / "simple_fundamental_baseline_prep_report.md", report_md)

    qa_rows = [
        {"check": "required_outputs_generated", "status": "PASS", "detail": "All requested prep outputs were written."},
        {"check": "training_panel_parquet_not_read", "status": "PASS", "detail": str(parquet_path)},
        {"check": "no_training_backtest_ic_shap_tuning", "status": "PASS", "detail": "All execution flags false."},
        {"check": "production_not_modified", "status": "PASS", "detail": "No production path touched by this script."},
        {"check": "guardrails_passed", "status": "PASS" if guardrail_checklist_passed else "FAIL", "detail": str(guardrail_checklist_passed)},
        {"check": "final_decision_allowed_value", "status": "PASS", "detail": final_decision},
    ]
    write_csv(OUT_DIR / "final_qa.csv", qa_rows, ["check", "status", "detail"])

    completion_card = "\n".join(
        [
            "# Task Completion Card",
            "",
            f"Task: {TASK_NAME}",
            f"Final decision: {final_decision}",
            f"Prerequisites passed: {bool_str(prerequisites_passed)}",
            f"Candidate feature count: {len(BASELINE_FEATURES)}",
            "Training/backtest/IC/SHAP/tuning/holdings/production: all false",
            "Training panel parquet content read: false",
            "",
        ]
    )
    write_text(OUT_DIR / "task_completion_card.md", completion_card)

    terminal_summary = {
        "task_name": TASK_NAME,
        "run_timestamp": run_timestamp,
        "script": str(Path(__file__).resolve()),
        "stdout_log": str(RUN_DIR / "run_stdout.txt"),
        "stderr_log": str(RUN_DIR / "run_stderr.txt"),
        "final_decision": final_decision,
        "outputs_dir": str(OUT_DIR),
        "no_heavy_operations": True,
    }
    write_json(OUT_DIR / "terminal_summary.json", terminal_summary)

    run_state = "\n".join(
        [
            "# RUN_STATE.md",
            "",
            f"Task: {TASK_NAME}",
            "Status: completed",
            f"Final decision: {final_decision}",
            "",
            "Completed artifacts:",
            "- simple_baseline_prep_prerequisite_check.json",
            "- simple_baseline_feature_seed_manifest.csv",
            "- simple_baseline_expected_direction_policy.csv",
            "- simple_baseline_evaluation_plan.csv",
            "- simple_baseline_guardrail_checklist.csv",
            "- simple_baseline_next_step_plan.md",
            "- simple_fundamental_baseline_prep_summary.json",
            "- simple_fundamental_baseline_prep_report.md",
            "- task_completion_card.md",
            "- terminal_summary.json",
            "- final_qa.csv",
            "",
            "Resume note:",
            "- No parquet content was read.",
            "- No training/backtest/IC/SHAP/tuning/holdings/production actions were run.",
            "",
        ]
    )
    write_text(RUN_DIR / "RUN_STATE.md", run_state)

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    del seed_rows, do_not_rows, split_rows, manifest_rows, direction_rows, eval_rows, guardrail_rows
    gc.collect()
    return 0 if final_decision != FINAL_FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
