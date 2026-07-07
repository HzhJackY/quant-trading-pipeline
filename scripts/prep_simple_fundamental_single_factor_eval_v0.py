from __future__ import annotations

import csv
import gc
import json
from datetime import datetime
from pathlib import Path


TASK_NAME = "simple_fundamental_single_factor_eval_prep_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

BASELINE_DIR = ROOT / "output" / "simple_fundamental_baseline_prep_v0"
STOP_DIR = ROOT / "output" / "compact_f_v3_full_stop_pivot_decision_memo_v0"
SPLIT_CONFIG_PATH = ROOT / "output" / "compact_f_v3_full_training_config_freeze_v0" / "frozen_split_config.csv"
INPUT_PANEL_PATH = (
    ROOT
    / "output"
    / "compact_f_v3_full_training_panel_price_label_v0"
    / "compact_f_v3_full_training_panel_price_label_unique13_v0.parquet"
)

INPUTS = {
    "baseline_summary": BASELINE_DIR / "simple_fundamental_baseline_prep_summary.json",
    "feature_seed_manifest": BASELINE_DIR / "simple_baseline_feature_seed_manifest.csv",
    "expected_direction_policy": BASELINE_DIR / "simple_baseline_expected_direction_policy.csv",
    "baseline_evaluation_plan": BASELINE_DIR / "simple_baseline_evaluation_plan.csv",
    "baseline_guardrail_checklist": BASELINE_DIR / "simple_baseline_guardrail_checklist.csv",
    "branch_stop_record": STOP_DIR / "compact_f_v3_full_branch_stop_record.json",
    "do_not_continue_list": STOP_DIR / "compact_f_v3_full_do_not_continue_list.csv",
    "frozen_split_config": SPLIT_CONFIG_PATH,
    "input_panel_future_path": INPUT_PANEL_PATH,
}

CANDIDATE_FEATURES = [
    "bp_rank",
    "ep_ttm_rank",
    "cfo_to_earnings_parent_rank",
    "roe_ttm_rank",
    "profit_growth_yoy_rank",
]
EXPECTED_DIRECTIONS = {feature: "POSITIVE" for feature in CANDIDATE_FEATURES}
TARGET_COLUMN = "fwd_ret_1m"
TARGET_DEFINITION = "RAW_CLOSE_TO_CLOSE_MONTHLY_FORWARD_RETURN"

METRICS = [
    ("IC", "monthly Pearson IC", "validation,test", "Allowed only in next evaluation run."),
    ("Rank IC", "monthly Spearman Rank IC", "validation,test", "Allowed only in next evaluation run."),
    ("IC summary", "IC summary", "validation,test", "Allowed only in next evaluation run."),
    ("Decile", "single-factor decile diagnostics", "train,validation,test", "Train is diagnostics only."),
    ("Decile", "D10-D1 spread", "validation,test", "Final conclusion uses validation and test only."),
    ("Aggregate", "validation/test aggregate summary", "validation,test", "Final conclusion uses validation and test only."),
]

PROHIBITIONS = [
    "read training panel parquet content",
    "training model",
    "backtest",
    "calculate IC",
    "calculate Rank IC",
    "calculate decile return",
    "tuning",
    "SHAP",
    "feature importance",
    "holdings generation",
    "production write",
    "continue Compact-F-v3-full rescue",
    "sign-flip production/backtest",
    "LightGBM-first",
    "duplicate-weighted logical15",
    "current_ratio / quick_ratio as core alpha without separate validation",
]

FINAL_READY = "SIMPLE_FUNDAMENTAL_SINGLE_FACTOR_EVAL_PREP_READY_FOR_RUN"
FINAL_WATCH = "SIMPLE_FUNDAMENTAL_SINGLE_FACTOR_EVAL_PREP_WATCH_REVIEW_REQUIRED"
FINAL_FAIL = "SIMPLE_FUNDAMENTAL_SINGLE_FACTOR_EVAL_PREP_FAIL"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    input_status = {
        name: {"path": str(path), "exists": path.exists(), "is_file": path.is_file()}
        for name, path in INPUTS.items()
    }
    missing_inputs = [name for name, status in input_status.items() if not status["is_file"]]

    baseline_summary = {}
    seed_rows: list[dict] = []
    direction_rows: list[dict] = []
    eval_plan_rows: list[dict] = []
    baseline_guard_rows: list[dict] = []
    stop_record = {}
    do_not_rows: list[dict] = []
    split_rows: list[dict] = []

    if not missing_inputs:
        baseline_summary = read_json(INPUTS["baseline_summary"])
        seed_rows = read_csv_rows(INPUTS["feature_seed_manifest"])
        direction_rows = read_csv_rows(INPUTS["expected_direction_policy"])
        eval_plan_rows = read_csv_rows(INPUTS["baseline_evaluation_plan"])
        baseline_guard_rows = read_csv_rows(INPUTS["baseline_guardrail_checklist"])
        stop_record = read_json(INPUTS["branch_stop_record"])
        do_not_rows = read_csv_rows(INPUTS["do_not_continue_list"])
        split_rows = read_csv_rows(INPUTS["frozen_split_config"])

    seed_features = [row.get("feature_name", "") for row in seed_rows]
    direction_map = {row.get("feature_name", ""): row.get("expected_direction", "") for row in direction_rows}
    expected_direction_all_positive = all(direction_map.get(feature) == "POSITIVE" for feature in CANDIDATE_FEATURES)
    candidate_features_match = seed_features == CANDIDATE_FEATURES
    baseline_ready = baseline_summary.get("final_decision") == "SIMPLE_FUNDAMENTAL_BASELINE_PREP_READY_FOR_SINGLE_FACTOR_EVALUATION_PREP"
    baseline_prereq = truthy(baseline_summary.get("prerequisites_passed"))
    baseline_no_parquet_read = not truthy(baseline_summary.get("training_panel_parquet_content_read"))
    branch_stopped = truthy(stop_record.get("branch_stopped"))
    split_config_present = len(split_rows) > 0
    input_panel_path_recorded = INPUT_PANEL_PATH.exists()
    compact_f_rescue_blocked = branch_stopped and any("LightGBM rescue" in row.get("prohibited_continuation", "") for row in do_not_rows)
    sign_flip_production_blocked = any("sign-flip as production" in row.get("prohibited_continuation", "") for row in do_not_rows)
    lightgbm_first_blocked = compact_f_rescue_blocked

    prerequisite_checks = [
        {"check": "all_required_inputs_present", "status": "PASS" if not missing_inputs else "FAIL", "detail": ",".join(missing_inputs)},
        {"check": "baseline_prep_ready", "status": "PASS" if baseline_ready and baseline_prereq else "FAIL", "detail": str(baseline_summary.get("final_decision"))},
        {"check": "candidate_features_frozen", "status": "PASS" if candidate_features_match else "FAIL", "detail": ",".join(CANDIDATE_FEATURES)},
        {"check": "expected_direction_all_positive", "status": "PASS" if expected_direction_all_positive else "FAIL", "detail": "POSITIVE"},
        {"check": "input_panel_path_recorded_only", "status": "PASS" if input_panel_path_recorded and baseline_no_parquet_read else "FAIL", "detail": str(INPUT_PANEL_PATH)},
        {"check": "split_config_present", "status": "PASS" if split_config_present else "FAIL", "detail": str(SPLIT_CONFIG_PATH)},
        {"check": "compact_f_rescue_blocked", "status": "PASS" if compact_f_rescue_blocked else "FAIL", "detail": str(branch_stopped)},
        {"check": "sign_flip_production_blocked", "status": "PASS" if sign_flip_production_blocked else "FAIL", "detail": str(sign_flip_production_blocked)},
        {"check": "baseline_guardrails_passed", "status": "PASS" if all(truthy(row.get("passed_in_this_prep")) for row in baseline_guard_rows) else "FAIL", "detail": f"rows={len(baseline_guard_rows)}"},
        {"check": "baseline_eval_plan_present", "status": "PASS" if len(eval_plan_rows) > 0 else "FAIL", "detail": f"rows={len(eval_plan_rows)}"},
    ]
    prerequisites_passed = all(row["status"] == "PASS" for row in prerequisite_checks)
    final_decision = FINAL_READY if prerequisites_passed else (FINAL_FAIL if missing_inputs else FINAL_WATCH)

    input_panel_stat = {"path": str(INPUT_PANEL_PATH), "exists": INPUT_PANEL_PATH.exists(), "content_read": False}
    if INPUT_PANEL_PATH.exists():
        stat = INPUT_PANEL_PATH.stat()
        input_panel_stat.update(
            {
                "size_bytes": stat.st_size,
                "last_write_time": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
            }
        )

    write_json(
        OUT_DIR / "single_factor_eval_prep_prerequisite_check.json",
        {
            "run_timestamp": run_timestamp,
            "task_name": TASK_NAME,
            "input_status": input_status,
            "input_panel_path": input_panel_stat,
            "prerequisite_checks": prerequisite_checks,
            "prerequisites_passed": prerequisites_passed,
            "training_panel_parquet_read": False,
        },
    )

    reason_map = {row.get("feature_name", ""): row.get("reason", "") for row in seed_rows}
    manifest_rows = [
        {
            "feature_name": feature,
            "expected_direction": "POSITIVE",
            "target_column": TARGET_COLUMN,
            "include_in_eval_run": "True",
            "source_reason": reason_map.get(feature, "Frozen from simple baseline prep."),
            "notes": "Candidate frozen for next single-factor evaluation run; not evaluated in this prep task.",
        }
        for feature in CANDIDATE_FEATURES
    ]
    write_csv(
        OUT_DIR / "single_factor_candidate_manifest.csv",
        manifest_rows,
        ["feature_name", "expected_direction", "target_column", "include_in_eval_run", "source_reason", "notes"],
    )

    sample_policy = {
        "train": "diagnostics only",
        "validation": "validation evaluation",
        "test": "out-of-sample evaluation",
        "final_conclusion": "validation + test only",
        "allowed_sample_roles": ["train", "validation", "test"],
        "train_role_usage": "diagnostics_only",
        "final_conclusion_roles": ["validation", "test"],
    }
    write_json(OUT_DIR / "single_factor_eval_sample_policy.json", sample_policy)

    metric_rows = [
        {
            "metric_group": group,
            "metric_name": name,
            "allowed_in_next_run": "True",
            "calculated_in_this_task": "False",
            "sample_roles_allowed": roles,
            "notes": notes,
        }
        for group, name, roles, notes in METRICS
    ]
    write_csv(
        OUT_DIR / "single_factor_metric_plan.csv",
        metric_rows,
        ["metric_group", "metric_name", "allowed_in_next_run", "calculated_in_this_task", "sample_roles_allowed", "notes"],
    )

    run_config = {
        "input_panel_path": str(INPUT_PANEL_PATH),
        "target_column": TARGET_COLUMN,
        "target_definition": TARGET_DEFINITION,
        "candidate_features": CANDIDATE_FEATURES,
        "expected_directions": EXPECTED_DIRECTIONS,
        "split_config_path": str(SPLIT_CONFIG_PATH),
        "allowed_sample_roles": ["train", "validation", "test"],
        "train_role_usage": "diagnostics_only",
        "final_conclusion_roles": ["validation", "test"],
        "output_directory_for_next_run": str(ROOT / "output" / "simple_fundamental_single_factor_eval_run_v0"),
        "calculated_now": False,
    }
    write_json(OUT_DIR / "single_factor_eval_run_config_draft.json", run_config)

    no_run_flags = {
        "training run?": False,
        "backtest run?": False,
        "IC calculated?": False,
        "Rank IC calculated?": False,
        "decile calculated?": False,
        "SHAP calculated?": False,
        "tuning run?": False,
        "holdings generated?": False,
        "production modified?": False,
        "training panel parquet read?": False,
    }
    guardrail_rows = [
        {
            "guardrail": item,
            "status": "BLOCK",
            "passed_in_this_prep": "True",
            "notes": "Strictly prohibited in prep task.",
        }
        for item in PROHIBITIONS
    ]
    guardrail_rows.extend(
        {
            "guardrail": label,
            "status": "PASS" if value is False else "FAIL",
            "passed_in_this_prep": "True" if value is False else "False",
            "notes": "Execution boundary check.",
        }
        for label, value in no_run_flags.items()
    )
    write_csv(OUT_DIR / "single_factor_guardrail_checklist.csv", guardrail_rows, ["guardrail", "status", "passed_in_this_prep", "notes"])
    guardrail_checklist_passed = all(row["passed_in_this_prep"] == "True" for row in guardrail_rows)

    plan_md = "\n".join(
        [
            "# Simple Fundamental Single-factor Evaluation Run Plan v0",
            "",
            f"Final decision: {final_decision}",
            "",
            "## Frozen Inputs",
            "",
            f"- Input panel path: {INPUT_PANEL_PATH}",
            f"- Split config path: {SPLIT_CONFIG_PATH}",
            f"- Target: {TARGET_COLUMN}",
            f"- Target definition: {TARGET_DEFINITION}",
            "",
            "## Candidate Features",
            "",
            *[f"- {feature}: POSITIVE" for feature in CANDIDATE_FEATURES],
            "",
            "## Next Run Metrics",
            "",
            "- monthly Pearson IC",
            "- monthly Spearman Rank IC",
            "- IC summary",
            "- single-factor decile diagnostics",
            "- D10-D1 spread",
            "- validation/test aggregate summary",
            "",
            "## Sample Policy",
            "",
            "- train = diagnostics only",
            "- validation = validation evaluation",
            "- test = out-of-sample evaluation",
            "- final conclusion = validation + test only",
            "",
            "This prep task did not read parquet content or calculate any metric.",
            "",
        ]
    )
    write_text(OUT_DIR / "single_factor_eval_run_plan.md", plan_md)

    summary = {
        "run_timestamp": run_timestamp,
        "task_name": TASK_NAME,
        "final_decision": final_decision,
        "prerequisites_passed": prerequisites_passed,
        "candidate_feature_count": len(CANDIDATE_FEATURES),
        "candidate_features": CANDIDATE_FEATURES,
        "expected_direction_all_positive": expected_direction_all_positive,
        "input_panel_path_recorded": str(INPUT_PANEL_PATH),
        "training_panel_parquet_read": False,
        "split_config_path": str(SPLIT_CONFIG_PATH),
        "sample_policy_generated": True,
        "metric_plan_generated": True,
        "run_config_draft_generated": True,
        "guardrail_checklist_passed": guardrail_checklist_passed,
        "compact_f_rescue_blocked": compact_f_rescue_blocked,
        "sign_flip_production_blocked": sign_flip_production_blocked,
        "lightgbm_first_blocked": lightgbm_first_blocked,
        "training_run": False,
        "backtest_run": False,
        "ic_calculated": False,
        "rank_ic_calculated": False,
        "decile_calculated": False,
        "shap_calculated": False,
        "tuning_run": False,
        "holdings_generated": False,
        "production_modified": False,
        "next_step_recommendation": "Run Simple Fundamental Single-factor Evaluation Run v0 using the generated draft config.",
    }
    write_json(OUT_DIR / "simple_fundamental_single_factor_eval_prep_summary.json", summary)

    report = "\n".join(
        [
            "# Simple Fundamental Single-factor Evaluation Prep v0",
            "",
            f"Final decision: {final_decision}",
            "",
            "The five simple fundamental candidates, positive directions, target, input panel path, split config, sample policy, and next-run metric plan are frozen.",
            "",
            "No parquet content was read. No IC, Rank IC, decile return, training, backtest, SHAP, tuning, holdings, or production write was performed.",
            "",
        ]
    )
    write_text(OUT_DIR / "simple_fundamental_single_factor_eval_prep_report.md", report)

    qa_rows = [
        {"check": "required_outputs_generated", "status": "PASS", "detail": "All requested prep outputs were written."},
        {"check": "training_panel_parquet_not_read", "status": "PASS", "detail": str(INPUT_PANEL_PATH)},
        {"check": "no_training_backtest_ic_rank_ic_decile", "status": "PASS", "detail": "All execution flags false."},
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
            f"Prerequisites passed: {prerequisites_passed}",
            f"Candidate feature count: {len(CANDIDATE_FEATURES)}",
            "Training/backtest/IC/Rank IC/decile/SHAP/tuning/holdings/production: all false",
            "Training panel parquet content read: false",
            "",
        ]
    )
    write_text(OUT_DIR / "task_completion_card.md", completion_card)

    write_json(
        OUT_DIR / "terminal_summary.json",
        {
            "task_name": TASK_NAME,
            "run_timestamp": run_timestamp,
            "script": str(Path(__file__).resolve()),
            "stdout_log": str(RUN_DIR / "run_stdout.txt"),
            "stderr_log": str(RUN_DIR / "run_stderr.txt"),
            "final_decision": final_decision,
            "outputs_dir": str(OUT_DIR),
            "no_heavy_operations": True,
        },
    )

    write_text(
        RUN_DIR / "RUN_STATE.md",
        "\n".join(
            [
                "# RUN_STATE.md",
                "",
                f"Task: {TASK_NAME}",
                "Status: completed",
                f"Final decision: {final_decision}",
                "",
                "Resume note:",
                "- No parquet content was read.",
                "- No training/backtest/IC/Rank IC/decile/SHAP/tuning/holdings/production actions were run.",
                "",
            ]
        ),
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    del seed_rows, direction_rows, eval_plan_rows, baseline_guard_rows, do_not_rows, split_rows, manifest_rows, metric_rows, guardrail_rows
    gc.collect()
    return 0 if final_decision != FINAL_FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
