from __future__ import annotations

import gc
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd


TASK_NAME = "simple_baseline_portfolio_evaluation_prep_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
AGENT_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME
CONSTRUCTION_DIR = ROOT / "output" / "simple_baseline_portfolio_construction_run_v0"
SCORE_EVAL_DIR = ROOT / "output" / "simple_baseline_score_evaluation_run_v0"

CONSTRUCTION_SUMMARY = CONSTRUCTION_DIR / "simple_baseline_portfolio_construction_run_summary.json"
COVERAGE_SUMMARY = CONSTRUCTION_DIR / "portfolio_coverage_summary.csv"
WEIGHTS_QA_BY_MONTH = CONSTRUCTION_DIR / "portfolio_weights_qa_by_month.csv"
LEAKAGE_QA = CONSTRUCTION_DIR / "portfolio_leakage_exclusion_qa.csv"
CONSTRUCTION_GUARDRAIL_QA = CONSTRUCTION_DIR / "portfolio_construction_guardrail_qa.csv"
WEIGHT_PANEL_PATH = CONSTRUCTION_DIR / "simple_baseline_research_weights_v0.parquet"
SCORE_EVAL_SUMMARY = SCORE_EVAL_DIR / "simple_baseline_score_evaluation_run_summary.json"
SCORE_FINAL_RANKING = SCORE_EVAL_DIR / "simple_baseline_score_final_ranking.csv"

PORTFOLIO_NAMES = [
    "BP_SINGLE_TOP_DECILE_EQUAL_WEIGHT",
    "BP_SINGLE_TOP50_EQUAL_WEIGHT",
    "VALUE_QUALITY_TOP_DECILE_EQUAL_WEIGHT",
    "VALUE_QUALITY_TOP50_EQUAL_WEIGHT",
]
TARGET_RETURN_COLUMN = "fwd_ret_1m"

ALLOWED_METRICS = [
    "weighted_monthly_forward_return",
    "mean_monthly_return",
    "median_monthly_return",
    "monthly_return_std",
    "positive_month_ratio",
    "cumulative_simple_return_path",
]
BLOCKED_METRICS = [
    "transaction_cost_adjusted_return",
    "turnover",
    "Sharpe",
    "MaxDD",
    "benchmark_relative_return",
    "alpha_beta_regression",
    "production_backtest_metric",
]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def pass_status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prerequisite_check(construction_summary: dict, coverage: pd.DataFrame, weights_qa: pd.DataFrame, leakage_qa: pd.DataFrame, guardrail_qa: pd.DataFrame, score_eval_summary: dict, score_ranking: pd.DataFrame) -> dict:
    expected_files = {
        "construction_summary": CONSTRUCTION_SUMMARY,
        "coverage_summary": COVERAGE_SUMMARY,
        "weights_qa_by_month": WEIGHTS_QA_BY_MONTH,
        "leakage_qa": LEAKAGE_QA,
        "construction_guardrail_qa": CONSTRUCTION_GUARDRAIL_QA,
        "weight_panel_path": WEIGHT_PANEL_PATH,
        "score_eval_summary": SCORE_EVAL_SUMMARY,
        "score_final_ranking": SCORE_FINAL_RANKING,
    }
    files_exist = {name: path.exists() for name, path in expected_files.items()}
    construction_ready = (
        construction_summary.get("final_decision")
        == "SIMPLE_BASELINE_PORTFOLIO_CONSTRUCTION_RUN_READY_FOR_PORTFOLIO_EVALUATION_PREP"
    )
    portfolio_names_match = construction_summary.get("portfolio_names") == PORTFOLIO_NAMES
    checks = {
        "required_files_exist": all(files_exist.values()),
        "portfolio_construction_ready": construction_ready,
        "weight_panel_path_exists_recorded_only": WEIGHT_PANEL_PATH.exists(),
        "portfolio_names_match_expected": portfolio_names_match,
        "construction_weights_qa_passed": bool(construction_summary.get("weights_qa_passed")) and bool(weights_qa["status"].eq("PASS").all()),
        "construction_coverage_passed": bool(coverage["status"].eq("PASS").all()),
        "construction_leakage_qa_passed": bool(construction_summary.get("leakage_exclusion_qa_passed")) and bool(leakage_qa["status"].eq("PASS").all()),
        "construction_guardrail_qa_passed": bool(construction_summary.get("guardrail_qa_passed")) and bool(guardrail_qa["status"].eq("PASS").all()),
        "score_eval_summary_read": bool(score_eval_summary),
        "score_final_ranking_read": len(score_ranking) > 0,
        "fwd_ret_not_used_for_selection": construction_summary.get("fwd_ret_used_for_selection") is False,
        "no_forbidden_metrics_already_calculated": all(
            construction_summary.get(key) is False
            for key in [
                "portfolio_return_calculated",
                "backtest_run",
                "transaction_cost_calculated",
                "turnover_calculated",
                "sharpe_calculated",
                "maxdd_calculated",
                "training_run",
                "shap_calculated",
                "tuning_run",
                "feature_importance_calculated",
                "production_holdings_generated",
                "live_order_ready_file_generated",
                "production_modified",
            ]
        ),
    }
    return {
        "run_timestamp": now_iso(),
        "task_name": TASK_NAME,
        "expected_files": {name: str(path) for name, path in expected_files.items()},
        "files_exist": files_exist,
        "checks": checks,
        "prerequisites_passed": all(checks.values()),
    }


def build_candidate_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "portfolio_name": name,
                "research_only": True,
                "include_in_eval_run": True,
                "target_return_column": TARGET_RETURN_COLUMN,
                "source_weight_panel_path": str(WEIGHT_PANEL_PATH),
                "notes": "Research-only hypothetical portfolio; evaluation prep records path only and does not read parquet content.",
            }
            for name in PORTFOLIO_NAMES
        ]
    )


def build_metric_plan() -> pd.DataFrame:
    rows = []
    for metric in ALLOWED_METRICS:
        rows.append(
            {
                "metric_group": "research_return_evaluation",
                "metric_name": metric,
                "allowed_in_next_run": True,
                "calculated_in_this_task": False,
                "blocked_reason_if_not_allowed": "",
                "notes": "Allowed only in Simple Baseline Portfolio Evaluation Run v0.",
            }
        )
    blocked_notes = {
        "transaction_cost_adjusted_return": "Transaction cost is outside current research-only evaluation scope.",
        "turnover": "Turnover remains blocked.",
        "Sharpe": "Sharpe remains blocked.",
        "MaxDD": "MaxDD remains blocked.",
        "benchmark_relative_return": "Benchmark-relative performance remains blocked.",
        "alpha_beta_regression": "Benchmark/regression evaluation remains blocked.",
        "production_backtest_metric": "Production backtest metrics remain blocked.",
    }
    for metric in BLOCKED_METRICS:
        rows.append(
            {
                "metric_group": "blocked_metric",
                "metric_name": metric,
                "allowed_in_next_run": False,
                "calculated_in_this_task": False,
                "blocked_reason_if_not_allowed": blocked_notes[metric],
                "notes": "Do not calculate in next run.",
            }
        )
    return pd.DataFrame(rows)


def build_sample_policy() -> dict:
    return {
        "research_only": True,
        "target_return_column": TARGET_RETURN_COLUMN,
        "target_non_null_required": True,
        "missing_target_handling": "For next evaluation run, exclude selected holding rows with missing fwd_ret_1m from weighted return numerator/denominator and report target coverage by portfolio-month.",
        "evaluation_sample_policy": "Final evaluation uses all available portfolio months in the research weight panel.",
        "primary_result_policy": "Raw close-to-close forward return based research-only evaluation.",
        "transaction_cost_policy": "blocked; no transaction cost",
        "turnover_policy": "blocked; no turnover",
        "sharpe_policy": "blocked; no Sharpe",
        "maxdd_policy": "blocked; no MaxDD",
        "benchmark_policy": "blocked; no benchmark-relative performance",
        "production_policy": "blocked; no production use, no live holdings, no live-order-ready output",
    }


def build_run_config() -> dict:
    return {
        "weight_panel_path": str(WEIGHT_PANEL_PATH),
        "portfolio_names": PORTFOLIO_NAMES,
        "target_return_column": TARGET_RETURN_COLUMN,
        "research_only": True,
        "output_directory_for_next_run": str(ROOT / "output" / "simple_baseline_portfolio_evaluation_run_v0"),
        "allowed_metrics": ALLOWED_METRICS,
        "blocked_metrics": BLOCKED_METRICS,
        "target_non_null_required": True,
        "missing_target_handling": "exclude_missing_target_rows_and_report_coverage",
        "transaction_cost_allowed": False,
        "turnover_allowed": False,
        "sharpe_allowed": False,
        "maxdd_allowed": False,
        "benchmark_allowed": False,
        "production_backtest_allowed": False,
        "calculated_now": False,
    }


def build_guardrail_checklist() -> pd.DataFrame:
    rows = [
        ("weight panel parquet not read", True, "Prep records path only."),
        ("portfolio return not calculated", True, "No return computation in prep."),
        ("no backtest", True, "No backtest in prep."),
        ("no transaction cost", True, "Blocked."),
        ("no turnover", True, "Blocked."),
        ("no Sharpe", True, "Blocked."),
        ("no MaxDD", True, "Blocked."),
        ("no benchmark-relative return", True, "Blocked."),
        ("no training", True, "Blocked."),
        ("no tuning", True, "Blocked."),
        ("no SHAP", True, "Blocked."),
        ("no feature importance", True, "Blocked."),
        ("no production holdings", True, "Blocked."),
        ("no live-order-ready file", True, "Blocked."),
        ("no production write", True, "Blocked."),
        ("Compact-F rescue blocked", True, "Blocked."),
        ("sign-flip production blocked", True, "Blocked."),
        ("LightGBM-first blocked", True, "Blocked."),
    ]
    return pd.DataFrame(
        [{"guardrail": name, "passed": ok, "status": pass_status(ok), "notes": notes} for name, ok, notes in rows]
    )


def build_report(summary: dict) -> str:
    return "\n".join(
        [
            "# Simple Baseline Portfolio Evaluation Prep v0",
            "",
            "## 决策",
            "",
            f"- final_decision: {summary['final_decision']}",
            f"- recommended_next_step: {summary['recommended_next_step']}",
            "",
            "## 冻结内容",
            "",
            f"- weight_panel_path: {summary['weight_panel_path_recorded']}",
            f"- portfolio_count: {summary['portfolio_count']}",
            f"- target_return_column: {summary['target_return_column']}",
            "- research_only: true",
            "",
            "## 下一步允许",
            "",
            "- 读取 research weights 并计算 research-only weighted monthly forward return。",
            "- 生成 mean、median、std、positive month ratio、cumulative simple return path。",
            "",
            "## 本任务未做",
            "",
            "- 未读取 weight panel parquet 内容。",
            "- 未计算 portfolio return、交易成本、换手、Sharpe、MaxDD、benchmark-relative return。",
            "- 未训练、未调参、未计算 SHAP、未生成 production holdings 或 live-order-ready 文件。",
        ]
    ) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    (AGENT_DIR / "RUN_STATE.md").write_text(
        f"# RUN_STATE\n\n- task_name: {TASK_NAME}\n- status: running\n- updated_at: {now_iso()}\n- current_step: starting\n",
        encoding="utf-8",
    )
    write_json(
        AGENT_DIR / "RUN_STATE.json",
        {
            "task_name": TASK_NAME,
            "status": "running",
            "updated_at": now_iso(),
            "current_step": "starting",
            "resume_instruction": f"先读取 {AGENT_DIR / 'RUN_STATE.md'} 再继续。",
        },
    )

    required = [
        CONSTRUCTION_SUMMARY,
        COVERAGE_SUMMARY,
        WEIGHTS_QA_BY_MONTH,
        LEAKAGE_QA,
        CONSTRUCTION_GUARDRAIL_QA,
        WEIGHT_PANEL_PATH,
        SCORE_EVAL_SUMMARY,
        SCORE_FINAL_RANKING,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + "; ".join(missing))

    construction_summary = read_json(CONSTRUCTION_SUMMARY)
    coverage = pd.read_csv(COVERAGE_SUMMARY)
    weights_qa = pd.read_csv(WEIGHTS_QA_BY_MONTH)
    leakage_qa = pd.read_csv(LEAKAGE_QA)
    construction_guardrail_qa = pd.read_csv(CONSTRUCTION_GUARDRAIL_QA)
    score_eval_summary = read_json(SCORE_EVAL_SUMMARY)
    score_ranking = pd.read_csv(SCORE_FINAL_RANKING)

    prereq = prerequisite_check(
        construction_summary,
        coverage,
        weights_qa,
        leakage_qa,
        construction_guardrail_qa,
        score_eval_summary,
        score_ranking,
    )
    write_json(OUT_DIR / "portfolio_eval_prep_prerequisite_check.json", prereq)

    candidate_manifest = build_candidate_manifest()
    candidate_manifest.to_csv(OUT_DIR / "portfolio_eval_candidate_manifest.csv", index=False, encoding="utf-8-sig")

    metric_plan = build_metric_plan()
    metric_plan.to_csv(OUT_DIR / "portfolio_eval_metric_plan.csv", index=False, encoding="utf-8-sig")

    sample_policy = build_sample_policy()
    write_json(OUT_DIR / "portfolio_eval_sample_policy.json", sample_policy)

    run_config = build_run_config()
    write_json(OUT_DIR / "portfolio_eval_run_config_draft.json", run_config)

    guardrail = build_guardrail_checklist()
    guardrail.to_csv(OUT_DIR / "portfolio_eval_guardrail_checklist.csv", index=False, encoding="utf-8-sig")

    next_step = (
        "# Next Step: Simple Baseline Portfolio Evaluation Run v0\n\n"
        "下一步只能执行 research-only portfolio return evaluation。\n\n"
        "允许：读取 research weights，按 portfolio-month 计算 weighted average fwd_ret_1m，生成 raw monthly return series、"
        "mean、median、std、positive month ratio、cumulative simple return path，并执行 weight sum / target availability / guardrail QA。\n\n"
        "禁止：production backtest、benchmark-relative performance、transaction cost、turnover、Sharpe、MaxDD、"
        "live holdings、model training、SHAP、tuning。\n"
    )
    (OUT_DIR / "next_step_simple_baseline_portfolio_evaluation_run_plan.md").write_text(next_step, encoding="utf-8")

    guardrail_passed = bool(guardrail["passed"].all())
    prerequisites_passed = bool(prereq["prerequisites_passed"])
    metric_plan_generated = (OUT_DIR / "portfolio_eval_metric_plan.csv").exists()
    sample_policy_generated = (OUT_DIR / "portfolio_eval_sample_policy.json").exists()
    run_config_draft_generated = (OUT_DIR / "portfolio_eval_run_config_draft.json").exists()
    construction_ready = bool(prereq["checks"]["portfolio_construction_ready"])
    blocking_passed = (
        prerequisites_passed
        and construction_ready
        and WEIGHT_PANEL_PATH.exists()
        and metric_plan_generated
        and sample_policy_generated
        and run_config_draft_generated
        and guardrail_passed
    )
    final_decision = (
        "SIMPLE_BASELINE_PORTFOLIO_EVAL_PREP_READY_FOR_EVALUATION_RUN"
        if blocking_passed
        else "SIMPLE_BASELINE_PORTFOLIO_EVAL_PREP_FAIL"
    )

    summary = {
        "run_timestamp": now_iso(),
        "prerequisites_passed": prerequisites_passed,
        "portfolio_construction_ready": construction_ready,
        "weight_panel_path_recorded": str(WEIGHT_PANEL_PATH),
        "weight_panel_parquet_read": False,
        "portfolio_count": len(PORTFOLIO_NAMES),
        "portfolio_names": PORTFOLIO_NAMES,
        "target_return_column": TARGET_RETURN_COLUMN,
        "research_only_policy": True,
        "fwd_ret_used_for_selection": False,
        "allowed_metric_count": len(ALLOWED_METRICS),
        "blocked_metric_count": len(BLOCKED_METRICS),
        "allowed_metrics": ALLOWED_METRICS,
        "blocked_metrics": BLOCKED_METRICS,
        "metric_plan_generated": metric_plan_generated,
        "sample_policy_generated": sample_policy_generated,
        "run_config_draft_generated": run_config_draft_generated,
        "guardrail_checklist_passed": guardrail_passed,
        "transaction_cost_allowed_next_run": False,
        "turnover_allowed_next_run": False,
        "sharpe_allowed_next_run": False,
        "maxdd_allowed_next_run": False,
        "benchmark_allowed_next_run": False,
        "production_backtest_allowed_next_run": False,
        "portfolio_return_calculated": False,
        "transaction_cost_calculated": False,
        "turnover_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "benchmark_relative_return_calculated": False,
        "training_run": False,
        "shap_calculated": False,
        "tuning_run": False,
        "feature_importance_calculated": False,
        "production_holdings_generated": False,
        "live_order_ready_file_generated": False,
        "production_modified": False,
        "compact_f_rescue_blocked": True,
        "sign_flip_production_blocked": True,
        "lightgbm_first_blocked": True,
        "final_decision": final_decision,
        "recommended_next_step": "Simple Baseline Portfolio Evaluation Run v0",
    }
    write_json(OUT_DIR / "simple_baseline_portfolio_evaluation_prep_summary.json", summary)
    (OUT_DIR / "simple_baseline_portfolio_evaluation_prep_report.md").write_text(build_report(summary), encoding="utf-8")

    final_qa = pd.DataFrame(
        [
            ["prerequisites_passed", prerequisites_passed],
            ["portfolio_construction_ready", construction_ready],
            ["weight_panel_parquet_read", False],
            ["metric_plan_generated", metric_plan_generated],
            ["sample_policy_generated", sample_policy_generated],
            ["run_config_draft_generated", run_config_draft_generated],
            ["guardrail_checklist_passed", guardrail_passed],
            ["final_decision", final_decision],
        ],
        columns=["check_name", "observed_value"],
    )
    final_qa.to_csv(AGENT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    write_json(
        AGENT_DIR / "terminal_summary.json",
        {
            "task_name": TASK_NAME,
            "completed_at": now_iso(),
            "final_decision": final_decision,
            "stdout_log": str(AGENT_DIR / "run_stdout.txt"),
            "stderr_log": str(AGENT_DIR / "run_stderr.txt"),
            "summary_path": str(OUT_DIR / "simple_baseline_portfolio_evaluation_prep_summary.json"),
        },
    )
    (AGENT_DIR / "task_completion_card.md").write_text(
        "\n".join(
            [
                "# Task Completion Card",
                "",
                f"- task_name: {TASK_NAME}",
                f"- completed_at: {now_iso()}",
                f"- final_decision: {final_decision}",
                f"- weight_panel_path_recorded: {WEIGHT_PANEL_PATH}",
                "- weight_panel_parquet_read: false",
                "- portfolio_return_calculated: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (AGENT_DIR / "RUN_STATE.md").write_text(
        f"# RUN_STATE\n\n- task_name: {TASK_NAME}\n- status: complete\n- updated_at: {now_iso()}\n- final_decision: {final_decision}\n",
        encoding="utf-8",
    )
    write_json(
        AGENT_DIR / "RUN_STATE.json",
        {"task_name": TASK_NAME, "status": "complete", "updated_at": now_iso(), "final_decision": final_decision},
    )

    print(json.dumps({"final_decision": final_decision, "weight_panel_parquet_read": False}, ensure_ascii=False))

    del construction_summary, coverage, weights_qa, leakage_qa, construction_guardrail_qa
    del score_eval_summary, score_ranking, candidate_manifest, metric_plan, guardrail, final_qa
    gc.collect()


if __name__ == "__main__":
    main()
