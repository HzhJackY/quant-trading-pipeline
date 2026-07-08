from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


TASK_NAME = "Robust Cleaned Score Evaluation Prep v0"
OUT_DIR = Path("output/robust_cleaned_score_evaluation_prep_v0")
RUN_DIR = Path("output/_agent_runs") / TASK_NAME
RUN_STATE = RUN_DIR / "RUN_STATE.md"

ROBUST_BUILD_DIR = Path("output/robust_cleaned_fundamental_factor_variant_build_v0")
ROBUST_PANEL_PATH = ROBUST_BUILD_DIR / "robust_cleaned_factor_score_panel_v0.parquet"
ROBUST_BUILD_SUMMARY_PATH = ROBUST_BUILD_DIR / "robust_cleaned_fundamental_factor_variant_build_summary.json"
ROBUST_FORMULA_PATH = ROBUST_BUILD_DIR / "robust_score_formula_manifest.csv"
ROBUST_COVERAGE_PATH = ROBUST_BUILD_DIR / "robust_score_coverage_qa.csv"
ROBUST_EFFECTIVENESS_PATH = ROBUST_BUILD_DIR / "robust_extreme_control_effectiveness.csv"
ROBUST_LEAKAGE_PATH = ROBUST_BUILD_DIR / "robust_leakage_guardrail_qa.csv"

PRIOR_EVAL_DIR = Path("output/asof_industry_neutral_score_evaluation_run_v0")
PRIOR_SUMMARY_PATH = PRIOR_EVAL_DIR / "asof_industry_neutral_score_evaluation_summary.json"
PRIOR_MONTH_AGG_PATH = PRIOR_EVAL_DIR / "unique_month_score_aggregate.csv"
PRIOR_RAW_VS_NEUTRAL_PATH = PRIOR_EVAL_DIR / "raw_vs_neutral_comparison.csv"
PRIOR_DECISION_PATH = PRIOR_EVAL_DIR / "neutral_score_decision_matrix.csv"

EXTREME_AUDIT_DIR = Path("output/core_fundamental_factor_extreme_treatment_audit_v0")
EXTREME_SUMMARY_PATH = EXTREME_AUDIT_DIR / "core_factor_extreme_treatment_audit_summary.json"
EXTREME_MATRIX_PATH = EXTREME_AUDIT_DIR / "factor_extreme_risk_decision_matrix.csv"


ROBUST_RAW_SCORE_COLUMNS = [
    "ROBUST_VALUE_BP_SINGLE_score",
    "ROBUST_VALUE_QUALITY_EQUAL_WEIGHT_score",
]
ROBUST_NEUTRAL_SCORE_COLUMNS = [
    "ROBUST_ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
    "ROBUST_ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
]
REFERENCE_SCORE_COLUMNS = [
    "VALUE_BP_SINGLE_score",
    "VALUE_QUALITY_EQUAL_WEIGHT_score",
    "ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
    "ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
]


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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("运行中", "开始生成 robust score evaluation prep artifacts。")

    required_paths = [
        ROBUST_BUILD_SUMMARY_PATH,
        ROBUST_FORMULA_PATH,
        ROBUST_COVERAGE_PATH,
        ROBUST_EFFECTIVENESS_PATH,
        ROBUST_LEAKAGE_PATH,
        ROBUST_PANEL_PATH,
        PRIOR_SUMMARY_PATH,
        PRIOR_MONTH_AGG_PATH,
        PRIOR_RAW_VS_NEUTRAL_PATH,
        PRIOR_DECISION_PATH,
        EXTREME_SUMMARY_PATH,
        EXTREME_MATRIX_PATH,
    ]
    missing = [str(p) for p in required_paths if not p.exists()]

    panel_columns: list[str] = []
    panel_rows = 0
    if ROBUST_PANEL_PATH.exists():
        parquet_file = pq.ParquetFile(ROBUST_PANEL_PATH)
        panel_columns = parquet_file.schema_arrow.names
        panel_rows = int(parquet_file.metadata.num_rows)

    build_summary = read_json(ROBUST_BUILD_SUMMARY_PATH) if ROBUST_BUILD_SUMMARY_PATH.exists() else {}
    prior_summary = read_json(PRIOR_SUMMARY_PATH) if PRIOR_SUMMARY_PATH.exists() else {}
    extreme_summary = read_json(EXTREME_SUMMARY_PATH) if EXTREME_SUMMARY_PATH.exists() else {}

    leakage = pd.read_csv(ROBUST_LEAKAGE_PATH) if ROBUST_LEAKAGE_PATH.exists() else pd.DataFrame()
    leakage_passed = bool(leakage.empty is False and leakage["passed"].astype(bool).all())

    required_panel_cols = [
        "symbol",
        "month_end",
        "fwd_ret_1m",
        "primary_industry_code",
        "industry_asof_enddate",
        *ROBUST_RAW_SCORE_COLUMNS,
        *ROBUST_NEUTRAL_SCORE_COLUMNS,
    ]
    missing_panel_cols = [col for col in required_panel_cols if col not in panel_columns]
    robust_panel_ready = bool(
        not missing
        and not missing_panel_cols
        and panel_rows > 0
        and build_summary.get("final_decision") == "ROBUST_CLEANED_FACTOR_VARIANT_READY_FOR_SCORE_EVALUATION_PREP"
        and build_summary.get("future_enddate_violation_count") == 0
    )

    prereq = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": bool(not missing and robust_panel_ready and leakage_passed),
        "missing_required_paths": missing,
        "robust_panel_path": str(ROBUST_PANEL_PATH),
        "robust_panel_rows": panel_rows,
        "missing_required_panel_columns": missing_panel_cols,
        "robust_build_final_decision": build_summary.get("final_decision"),
        "robust_leakage_guardrails_passed": leakage_passed,
        "prior_eval_final_decision": prior_summary.get("final_decision"),
        "extreme_audit_final_decision": extreme_summary.get("final_decision"),
    }
    (OUT_DIR / "robust_score_eval_prep_prerequisite_check.json").write_text(
        json.dumps(prereq, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = pd.DataFrame(
        [
            {
                "score_name": "ROBUST_VALUE_BP_SINGLE_score",
                "score_type": "robust_raw",
                "robust_variant": "ROBUST_V0",
                "source_factor_components": "bp_robust_rank",
                "primary_eval": False,
                "appendix_eval": True,
                "notes": "用于 robust raw vs neutral BP 对照。",
            },
            {
                "score_name": "ROBUST_VALUE_QUALITY_EQUAL_WEIGHT_score",
                "score_type": "robust_raw",
                "robust_variant": "ROBUST_V0",
                "source_factor_components": "bp_robust_rank;ep_ttm_robust_rank;cfo_to_earnings_parent_robust_rank",
                "primary_eval": False,
                "appendix_eval": True,
                "notes": "至少 2/3 robust component 非空。",
            },
            {
                "score_name": "ROBUST_ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
                "score_type": "robust_asof_industry_neutral",
                "robust_variant": "ROBUST_V0",
                "source_factor_components": "industry_within_rank(bp_robust_rank)",
                "primary_eval": True,
                "appendix_eval": False,
                "notes": "primary comparison with non-robust industry-neutral BP。",
            },
            {
                "score_name": "ROBUST_ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
                "score_type": "robust_asof_industry_neutral",
                "robust_variant": "ROBUST_V0",
                "source_factor_components": "industry neutral robust BP;EP;CFO/earnings ranks",
                "primary_eval": True,
                "appendix_eval": False,
                "notes": "primary comparison with non-robust industry-neutral value-quality。",
            },
        ]
    )
    manifest.to_csv(OUT_DIR / "robust_score_eval_manifest.csv", index=False)

    comparison_pairs = [
        {
            "comparison_pair_name": "industry_neutral_value_quality_nonrobust_vs_robust",
            "nonrobust_score": "ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
            "robust_score": "ROBUST_ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
            "comparison_type": "nonrobust_vs_robust_industry_neutral",
            "complete_case_required": True,
            "notes": "primary comparison; same symbol-month complete-case sample。",
        },
        {
            "comparison_pair_name": "industry_neutral_bp_nonrobust_vs_robust",
            "nonrobust_score": "ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
            "robust_score": "ROBUST_ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
            "comparison_type": "nonrobust_vs_robust_industry_neutral",
            "complete_case_required": True,
            "notes": "primary comparison; same symbol-month complete-case sample。",
        },
    ]
    pd.DataFrame(comparison_pairs).to_csv(OUT_DIR / "robust_vs_nonrobust_comparison_pairs.csv", index=False)

    raw_vs_neutral_pairs = [
        {
            "comparison_pair_name": "robust_value_quality_raw_vs_neutral",
            "raw_robust_score": "ROBUST_VALUE_QUALITY_EQUAL_WEIGHT_score",
            "neutral_robust_score": "ROBUST_ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
            "complete_case_required": True,
            "notes": "diagnostic robust raw vs industry-neutral comparison。",
        },
        {
            "comparison_pair_name": "robust_bp_raw_vs_neutral",
            "raw_robust_score": "ROBUST_VALUE_BP_SINGLE_score",
            "neutral_robust_score": "ROBUST_ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
            "complete_case_required": True,
            "notes": "diagnostic robust raw vs industry-neutral comparison。",
        },
    ]
    pd.DataFrame(raw_vs_neutral_pairs).to_csv(OUT_DIR / "robust_raw_vs_neutral_comparison_pairs.csv", index=False)

    complete_case_policy = {
        "complete_case_required": True,
        "target_column": "fwd_ret_1m",
        "symbol_column": "symbol",
        "date_column": "month_end",
        "robust_score_columns": ROBUST_RAW_SCORE_COLUMNS + ROBUST_NEUTRAL_SCORE_COLUMNS,
        "reference_score_columns": REFERENCE_SCORE_COLUMNS,
        "comparison_sample_policy": "SAME_SYMBOL_MONTH_SAMPLE",
        "expected_robust_value_quality_rows": int(build_summary.get("robust_value_quality_non_null_rows", 76679)),
        "no_future_enddate_required": True,
        "reason": "下一阶段所有 robust/non-robust 与 raw/neutral 对照必须在同一 symbol-month complete-case 样本上评估，避免覆盖差异污染比较。",
    }
    (OUT_DIR / "complete_case_policy.json").write_text(
        json.dumps(complete_case_policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metric_plan = pd.DataFrame(
        [
            ["monthly_pearson_ic", True, False, "primary", "下一阶段允许；本 prep 不计算。"],
            ["monthly_spearman_rank_ic", True, False, "primary", "下一阶段允许；本 prep 不计算。"],
            ["unique_month_aggregate", True, False, "primary", "下一阶段按 unique month 聚合。"],
            ["ic_ir", True, False, "primary", "下一阶段由 monthly IC 计算。"],
            ["ic_t_stat", True, False, "primary", "下一阶段由 monthly IC 计算。"],
            ["positive_ic_month_ratio", True, False, "primary", "下一阶段允许。"],
            ["decile_return_table", True, False, "primary", "下一阶段允许 decile return；本 prep 禁止。"],
            ["d10_d1_spread", True, False, "primary", "下一阶段允许；本 prep 禁止。"],
            ["positive_spread_month_ratio", True, False, "primary", "下一阶段允许。"],
            ["monotonicity_label", True, False, "primary", "下一阶段允许。"],
            ["robust_vs_nonrobust_comparison", True, False, "primary", "使用 same symbol-month complete-case。"],
            ["raw_vs_industry_neutral_comparison", True, False, "diagnostic", "评估 industry neutralization 对 robust score 的影响。"],
            ["complete_case_comparison", True, False, "diagnostic", "记录每个 pair 的样本数。"],
            ["portfolio_weights", False, False, "blocked", "下一阶段仍禁止构造组合。"],
            ["portfolio_return", False, False, "blocked", "下一阶段仍禁止组合收益。"],
            ["backtest", False, False, "blocked", "下一阶段仍禁止回测。"],
        ],
        columns=["metric_name", "allowed_next_run", "calculated_in_this_task", "primary_or_diagnostic", "notes"],
    )
    metric_plan.to_csv(OUT_DIR / "robust_score_metric_plan.csv", index=False)

    decision_framework = pd.DataFrame(
        [
            {
                "decision_label": "ROBUST_SCORE_STRONG_PASS",
                "condition": "robust industry-neutral Rank IC > 0; t-stat > 1.5; positive Rank IC month ratio >= 0.55; D10-D1 > 0; no material underperformance vs non-robust; monotonicity not INVERTED; anomaly reduced",
                "interpretation": "清洗后 alpha 仍较稳健。",
                "recommended_action": "进入 robust industry-neutral portfolio construction prep。",
            },
            {
                "decision_label": "ROBUST_SCORE_PARTIAL_PASS",
                "condition": "robust industry-neutral Rank IC > 0; positive Rank IC month ratio >= 0.52; D10-D1 or monotonicity weak",
                "interpretation": "信号可用但边际偏弱。",
                "recommended_action": "进入 prep 前人工复核稳定性和覆盖。",
            },
            {
                "decision_label": "ROBUST_SCORE_WATCH_SIGNAL_LOSS_AFTER_CLEANING",
                "condition": "non-robust score strong; robust Rank IC or D10-D1 materially decays",
                "interpretation": "原始信号可能依赖极端值。",
                "recommended_action": "人工判断是否回到 raw industry-exposed baseline 或重新设计因子。",
            },
            {
                "decision_label": "ROBUST_SCORE_FAIL",
                "condition": "robust Rank IC <= 0 or robust D10-D1 <= 0 with weak stability",
                "interpretation": "清洗后 alpha 不受支持。",
                "recommended_action": "停止 portfolio construction prep。",
            },
            {
                "decision_label": "ROBUST_SCORE_EVAL_RUN_READY_FOR_PORTFOLIO_CONSTRUCTION_PREP",
                "condition": "至少一个 robust industry-neutral score 为 STRONG_PASS 或 PARTIAL_PASS，且无 guardrail violation",
                "interpretation": "run-level 可进入组合准备。",
                "recommended_action": "开启 robust industry-neutral portfolio construction prep。",
            },
            {
                "decision_label": "ROBUST_SCORE_EVAL_RUN_WATCH_SIGNAL_WEAKENED",
                "condition": "robust 后 signal 明显衰减",
                "interpretation": "清洗削弱信号，需要人工取舍。",
                "recommended_action": "人工评审后再决定下一阶段。",
            },
            {
                "decision_label": "ROBUST_SCORE_EVAL_RUN_FAIL_CLEANED_ALPHA_NOT_SUPPORTED",
                "condition": "robust scores 均失败",
                "interpretation": "清洗后 score 不支持 alpha。",
                "recommended_action": "停止组合准备。",
            },
            {
                "decision_label": "ROBUST_SCORE_EVAL_RUN_FAIL_GUARDRAIL",
                "condition": "fwd_ret 泄露、future EndDate、输入异常等",
                "interpretation": "评估无效。",
                "recommended_action": "修复输入或策略后重跑。",
            },
        ]
    )
    decision_framework.to_csv(OUT_DIR / "robust_score_decision_framework.csv", index=False)

    run_config = {
        "input_panel_path": str(ROBUST_PANEL_PATH),
        "target_column": "fwd_ret_1m",
        "date_column": "month_end",
        "symbol_column": "symbol",
        "industry_column": "primary_industry_code",
        "industry_asof_enddate_column": "industry_asof_enddate",
        "robust_raw_score_columns": ROBUST_RAW_SCORE_COLUMNS,
        "robust_industry_neutral_score_columns": ROBUST_NEUTRAL_SCORE_COLUMNS,
        "reference_nonrobust_score_columns": REFERENCE_SCORE_COLUMNS,
        "comparison_pairs": comparison_pairs + raw_vs_neutral_pairs,
        "complete_case_required": True,
        "comparison_sample_policy": "SAME_SYMBOL_MONTH_SAMPLE",
        "unique_month_policy_primary": True,
        "output_directory_for_next_run": "output/robust_cleaned_score_evaluation_run_v0/",
        "calculate_ic_next_run_allowed": True,
        "calculate_decile_next_run_allowed": True,
        "calculate_portfolio_next_run_allowed": False,
        "calculate_portfolio_return_next_run_allowed": False,
        "backtest_allowed_next_run": False,
        "production_allowed_next_run": False,
        "calculated_now": False,
        "notes": "reference non-robust scores are comparison targets; next run must enforce complete-case before comparing robust vs non-robust。",
    }
    (OUT_DIR / "robust_score_eval_run_config_draft.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    guardrail_rows = [
        ("robust_panel_exists", ROBUST_PANEL_PATH.exists(), str(ROBUST_PANEL_PATH)),
        ("robust_panel_ready", robust_panel_ready, "required robust score columns present and build summary ready"),
        ("fwd_ret_not_used_in_prep_metrics", True, "prep only writes config; no metric calculation"),
        ("no_ic_calculated_in_prep", True, "metric plan only"),
        ("no_d10_d1_calculated_in_prep", True, "metric plan only"),
        ("no_decile_return_calculated_in_prep", True, "metric plan only"),
        ("no_portfolio_constructed", True, "config blocks portfolio next run"),
        ("no_portfolio_return_calculated", True, "config blocks portfolio return next run"),
        ("no_backtest_run", True, "config blocks backtest next run"),
        ("no_future_enddate_required", build_summary.get("future_enddate_violation_count") == 0, "robust build future violation count"),
        ("robust_leakage_guardrails_passed", leakage_passed, "robust build leakage QA"),
        ("production_not_modified", True, "outputs restricted to prep output directory"),
    ]
    guardrail = pd.DataFrame(guardrail_rows, columns=["guardrail", "passed", "evidence"])
    guardrail.to_csv(OUT_DIR / "robust_score_eval_guardrail_checklist.csv", index=False)
    guardrail_checklist_passed = bool(guardrail["passed"].astype(bool).all())

    metric_plan_generated = (OUT_DIR / "robust_score_metric_plan.csv").exists()
    run_config_draft_generated = (OUT_DIR / "robust_score_eval_run_config_draft.json").exists()
    comparison_pair_count = len(comparison_pairs) + len(raw_vs_neutral_pairs)

    final_decision = (
        "ROBUST_CLEANED_SCORE_EVAL_PREP_READY_FOR_RUN"
        if prereq["prerequisites_passed"]
        and metric_plan_generated
        and run_config_draft_generated
        and comparison_pair_count == 4
        and guardrail_checklist_passed
        else "ROBUST_CLEANED_SCORE_EVAL_PREP_FAIL"
    )

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "robust_panel_ready": robust_panel_ready,
        "robust_panel_path": str(ROBUST_PANEL_PATH),
        "robust_variant_name": build_summary.get("robust_variant_name", "ROBUST_V0"),
        "robust_raw_score_columns": ROBUST_RAW_SCORE_COLUMNS,
        "robust_industry_neutral_score_columns": ROBUST_NEUTRAL_SCORE_COLUMNS,
        "reference_nonrobust_score_columns": REFERENCE_SCORE_COLUMNS,
        "comparison_pair_count": comparison_pair_count,
        "complete_case_required": True,
        "expected_robust_value_quality_rows": int(build_summary.get("robust_value_quality_non_null_rows", 76679)),
        "unique_month_policy_primary": True,
        "metric_plan_generated": metric_plan_generated,
        "run_config_draft_generated": run_config_draft_generated,
        "guardrail_checklist_passed": guardrail_checklist_passed,
        "ic_calculated": False,
        "d10_d1_calculated": False,
        "decile_return_calculated": False,
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
        "recommended_next_step": "运行 robust_cleaned_score_evaluation_run_v0；该 run 可计算 IC 与 decile/D10-D1，但仍禁止 portfolio、backtest、production。"
        if final_decision == "ROBUST_CLEANED_SCORE_EVAL_PREP_READY_FOR_RUN"
        else "先修复输入或 guardrail，再重新生成 prep。",
    }
    (OUT_DIR / "robust_cleaned_score_eval_prep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# Robust Cleaned Score Evaluation Prep v0",
        "",
        f"- final_decision: {final_decision}",
        f"- robust_variant_name: {summary['robust_variant_name']}",
        f"- robust_panel_ready: {robust_panel_ready}",
        f"- comparison_pair_count: {comparison_pair_count}",
        f"- expected_robust_value_quality_rows: {summary['expected_robust_value_quality_rows']}",
        f"- guardrail_checklist_passed: {guardrail_checklist_passed}",
        "",
        "## 下一步",
        summary["recommended_next_step"],
    ]
    (OUT_DIR / "robust_cleaned_score_eval_prep_report.md").write_text("\n".join(report), encoding="utf-8")

    (OUT_DIR / "next_step_robust_score_evaluation_run_plan.md").write_text(
        "# 下一步：Robust Cleaned Score Evaluation Run v0\n\n"
        "1. 使用 robust_cleaned_factor_score_panel_v0.parquet 作为主输入。\n"
        "2. 对每个 comparison pair 执行 SAME_SYMBOL_MONTH_SAMPLE complete-case 过滤。\n"
        "3. 允许计算 monthly Pearson IC、Spearman Rank IC、unique-month aggregate、decile return、D10-D1、monotonicity。\n"
        "4. 禁止构造 portfolio、计算 portfolio return、回测、交易成本、换手、Sharpe、MaxDD、production 输出。\n",
        encoding="utf-8",
    )

    final_qa = pd.DataFrame(
        [
            {"check": "prerequisites_passed", "value": summary["prerequisites_passed"]},
            {"check": "final_decision", "value": final_decision},
            {"check": "robust_panel_ready", "value": robust_panel_ready},
            {"check": "metric_plan_generated", "value": metric_plan_generated},
            {"check": "run_config_draft_generated", "value": run_config_draft_generated},
            {"check": "guardrail_checklist_passed", "value": guardrail_checklist_passed},
            {"check": "ic_calculated", "value": False},
            {"check": "d10_d1_calculated", "value": False},
            {"check": "decile_return_calculated", "value": False},
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
        f"prep artifacts 已生成。final_decision={final_decision}。关键输出：{OUT_DIR / 'robust_score_eval_run_config_draft.json'}",
    )


if __name__ == "__main__":
    main()
