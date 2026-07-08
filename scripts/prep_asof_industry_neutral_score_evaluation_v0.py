from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


TASK_NAME = "As-Of Industry Neutral Score Evaluation Prep v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "asof_industry_neutral_score_evaluation_prep_v0"
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

NEUTRAL_RUN_DIR = ROOT / "output" / "debt_institutioninfo_annual_industry_neutral_score_run_v0"
RAW_EVAL_DIR = ROOT / "output" / "simple_baseline_score_evaluation_run_v0"

INPUTS = {
    "neutral_summary": NEUTRAL_RUN_DIR / "debt_institutioninfo_annual_industry_neutral_score_run_summary.json",
    "annual_join_qa": NEUTRAL_RUN_DIR / "annual_industry_asof_join_qa.csv",
    "neutral_score_qa": NEUTRAL_RUN_DIR / "asof_industry_neutral_score_qa.csv",
    "formula_manifest": NEUTRAL_RUN_DIR / "asof_industry_neutral_score_formula_manifest.csv",
    "neutral_score_panel": NEUTRAL_RUN_DIR / "simple_baseline_asof_industry_neutral_score_panel_v0.parquet",
    "raw_eval_summary": RAW_EVAL_DIR / "simple_baseline_score_evaluation_run_summary.json",
}

RAW_RANKING_CANDIDATES = [
    RAW_EVAL_DIR / "score_evaluation_final_ranking.csv",
    RAW_EVAL_DIR / "simple_baseline_score_final_ranking.csv",
]
UNIQUE_MONTH_CANDIDATES = [
    RAW_EVAL_DIR / "score_unique_month_aggregate.csv",
    RAW_EVAL_DIR / "unique_month_score_eval_summary.csv",
]
METHODOLOGY_INPUTS = {
    "methodology_qa_summary": ROOT
    / "output"
    / "simple_fundamental_single_factor_methodology_qa_v0"
    / "simple_fundamental_single_factor_methodology_qa_summary.json",
    "unique_month_recompute_summary": ROOT
    / "output"
    / "simple_single_factor_unique_month_recompute_v0"
    / "simple_single_factor_unique_month_recompute_summary.json",
}

RAW_SCORE_COLUMNS = ["VALUE_BP_SINGLE_score", "VALUE_QUALITY_EQUAL_WEIGHT_score"]
NEUTRAL_SCORE_COLUMNS = [
    "ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
    "ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
]
COMPARISON_PAIRS = [
    {
        "comparison_pair_name": "BP_SINGLE_raw_vs_asof_industry_neutral",
        "raw_score": "VALUE_BP_SINGLE_score",
        "neutral_score": "ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
        "comparison_type": "RAW_VS_ASOF_INDUSTRY_NEUTRAL",
        "complete_case_required": True,
        "notes": "同一 symbol-month 样本比较 BP single raw 与 as-of 行业中性版本。",
    },
    {
        "comparison_pair_name": "VALUE_QUALITY_EQUAL_WEIGHT_raw_vs_asof_industry_neutral",
        "raw_score": "VALUE_QUALITY_EQUAL_WEIGHT_score",
        "neutral_score": "ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
        "comparison_type": "RAW_VS_ASOF_INDUSTRY_NEUTRAL",
        "complete_case_required": True,
        "notes": "同一 symbol-month 样本比较 value-quality raw 与 as-of 行业中性版本。",
    },
]


def write_run_state(status: str, details: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "# RUN_STATE",
            "",
            f"- task_name: {TASK_NAME}",
            f"- status: {status}",
            f"- updated_at: {datetime.now().isoformat(timespec='seconds')}",
            f"- output_dir: {OUT_DIR}",
            f"- run_dir: {RUN_DIR}",
            "",
            "## Details",
            "```json",
            json.dumps(details, ensure_ascii=False, indent=2, default=str),
            "```",
            "",
        ]
    )
    (OUT_DIR / "RUN_STATE.md").write_text(text, encoding="utf-8")
    (RUN_DIR / "RUN_STATE.md").write_text(text, encoding="utf-8")


def choose_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_run_state("running", {"step": "start"})

    raw_ranking_path = choose_existing(RAW_RANKING_CANDIDATES)
    unique_month_path = choose_existing(UNIQUE_MONTH_CANDIDATES)
    prereq = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "required_inputs": {k: str(v) for k, v in INPUTS.items()},
        "required_exists": {k: v.exists() for k, v in INPUTS.items()},
        "raw_ranking_candidates": [str(p) for p in RAW_RANKING_CANDIDATES],
        "raw_ranking_path": None if raw_ranking_path is None else str(raw_ranking_path),
        "unique_month_candidates": [str(p) for p in UNIQUE_MONTH_CANDIDATES],
        "unique_month_path": None if unique_month_path is None else str(unique_month_path),
        "methodology_inputs": {k: str(v) for k, v in METHODOLOGY_INPUTS.items()},
        "methodology_exists": {k: v.exists() for k, v in METHODOLOGY_INPUTS.items()},
    }
    prereq["prerequisites_passed"] = all(prereq["required_exists"].values()) and raw_ranking_path is not None and unique_month_path is not None
    write_json(OUT_DIR / "asof_industry_neutral_score_eval_prep_prerequisite_check.json", prereq)

    if not prereq["prerequisites_passed"]:
        summary = {
            "run_timestamp": prereq["run_timestamp"],
            "prerequisites_passed": False,
            "neutral_score_panel_ready": False,
            "neutral_score_panel_path": str(INPUTS["neutral_score_panel"]),
            "raw_score_columns": RAW_SCORE_COLUMNS,
            "neutral_score_columns": NEUTRAL_SCORE_COLUMNS,
            "comparison_pair_count": 0,
            "complete_case_required": True,
            "expected_complete_case_rows": 0,
            "small_group_sensitivity_required": True,
            "unique_month_policy_primary": True,
            "old_split_role_policy_secondary": True,
            "metric_plan_generated": False,
            "run_config_draft_generated": False,
            "guardrail_checklist_passed": False,
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
            "final_decision": "ASOF_INDUSTRY_NEUTRAL_SCORE_EVAL_PREP_FAIL",
            "recommended_next_step": "补齐缺失输入后重跑 prep。",
        }
        write_json(OUT_DIR / "asof_industry_neutral_score_eval_prep_summary.json", summary)
        write_run_state("failed", summary)
        raise SystemExit("Missing required input for eval prep")

    write_run_state("running", {"step": "read_required_summaries_and_panel_schema"})
    neutral_summary = json.loads(INPUTS["neutral_summary"].read_text(encoding="utf-8"))
    join_qa = pd.read_csv(INPUTS["annual_join_qa"])
    neutral_qa = pd.read_csv(INPUTS["neutral_score_qa"])
    formula_manifest = pd.read_csv(INPUTS["formula_manifest"])
    raw_summary = json.loads(INPUTS["raw_eval_summary"].read_text(encoding="utf-8"))
    raw_ranking = pd.read_csv(raw_ranking_path, nrows=20)
    unique_month = pd.read_csv(unique_month_path, nrows=20)
    panel_probe = pd.read_parquet(
        INPUTS["neutral_score_panel"],
        columns=[
            "symbol",
            "month_end",
            "industry_asof_enddate",
            "primary_industry_code",
            "fwd_ret_1m",
            "small_group_flag",
            *RAW_SCORE_COLUMNS,
            *NEUTRAL_SCORE_COLUMNS,
        ],
    )

    panel_columns_present = set(panel_probe.columns)
    required_panel_cols = {"symbol", "month_end", "fwd_ret_1m", "small_group_flag", *RAW_SCORE_COLUMNS, *NEUTRAL_SCORE_COLUMNS}
    neutral_score_panel_ready = (
        neutral_summary.get("neutral_score_generated") is True
        and INPUTS["neutral_score_panel"].exists()
        and required_panel_cols.issubset(panel_columns_present)
        and len(panel_probe) == int(neutral_summary.get("neutral_score_row_count", -1))
    )
    expected_complete_case_rows = int(neutral_summary.get("neutral_score_row_count", len(panel_probe)))
    small_group_flag_available = "small_group_flag" in panel_probe.columns
    small_group_required = bool(neutral_summary.get("small_industry_group_detected", False)) and small_group_flag_available

    manifest = pd.DataFrame(
        [
            {
                "score_name": "VALUE_BP_SINGLE_score",
                "score_type": "raw",
                "raw_pair_score": "",
                "formula_source": "previous simple baseline score panel",
                "primary_eval": True,
                "appendix_eval": False,
                "notes": "Raw BP single score; compare only on complete-case sample.",
            },
            {
                "score_name": "VALUE_QUALITY_EQUAL_WEIGHT_score",
                "score_type": "raw",
                "raw_pair_score": "",
                "formula_source": "previous simple baseline score panel",
                "primary_eval": True,
                "appendix_eval": False,
                "notes": "Raw value-quality equal weight score; compare only on complete-case sample.",
            },
            {
                "score_name": "ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
                "score_type": "asof_industry_neutral",
                "raw_pair_score": "VALUE_BP_SINGLE_score",
                "formula_source": "asof_industry_neutral_score_formula_manifest.csv",
                "primary_eval": True,
                "appendix_eval": False,
                "notes": "industry_within_rank(bp_rank).",
            },
            {
                "score_name": "ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
                "score_type": "asof_industry_neutral",
                "raw_pair_score": "VALUE_QUALITY_EQUAL_WEIGHT_score",
                "formula_source": "asof_industry_neutral_score_formula_manifest.csv",
                "primary_eval": True,
                "appendix_eval": False,
                "notes": "Mean of industry-within ranks for bp, ep_ttm, cfo_to_earnings_parent.",
            },
        ]
    )
    manifest.to_csv(OUT_DIR / "asof_industry_neutral_score_eval_manifest.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(COMPARISON_PAIRS).to_csv(
        OUT_DIR / "raw_vs_neutral_comparison_pairs.csv", index=False, encoding="utf-8-sig"
    )

    complete_case_policy = {
        "complete_case_required": True,
        "raw_score_columns": RAW_SCORE_COLUMNS,
        "neutral_score_columns": NEUTRAL_SCORE_COLUMNS,
        "target_column": "fwd_ret_1m",
        "join_keys": ["symbol", "month_end"],
        "comparison_sample_policy": "SAME_SYMBOL_MONTH_SAMPLE",
        "expected_complete_case_rows_from_previous_step": expected_complete_case_rows,
        "reason": "Raw panel has 76,722 rows while neutral joined panel has 76,606 rows; raw-vs-neutral comparison must use the same symbol-month rows with raw score, neutral score, fwd_ret_1m, and successful as-of industry join.",
    }
    write_json(OUT_DIR / "complete_case_policy.json", complete_case_policy)

    small_group_policy = {
        "small_group_flag_available": bool(small_group_flag_available),
        "base_eval_include_small_groups": True,
        "sensitivity_exclude_small_groups": True,
        "blocking_if_large_difference": "WATCH_MANUAL_REVIEW_REQUIRED",
        "notes": "上一步检测到 small_industry_group_detected=true；下一阶段必须同时输出全样本 complete-case 和剔除 small_group_flag=true 后的 sensitivity。",
    }
    write_json(OUT_DIR / "small_group_sensitivity_policy.json", small_group_policy)

    metric_plan = pd.DataFrame(
        [
            {"metric_name": "monthly Pearson IC", "allowed_next_run": True, "calculated_in_this_task": False, "primary_or_diagnostic": "primary", "notes": "下一阶段按 complete-case symbol-month 计算。"},
            {"metric_name": "monthly Spearman Rank IC", "allowed_next_run": True, "calculated_in_this_task": False, "primary_or_diagnostic": "primary", "notes": "下一阶段主指标。"},
            {"metric_name": "unique-month average IC", "allowed_next_run": True, "calculated_in_this_task": False, "primary_or_diagnostic": "primary", "notes": "unique-calendar-month aggregate is primary。"},
            {"metric_name": "IC IR", "allowed_next_run": True, "calculated_in_this_task": False, "primary_or_diagnostic": "primary", "notes": "基于月度 IC 序列。"},
            {"metric_name": "positive IC month ratio", "allowed_next_run": True, "calculated_in_this_task": False, "primary_or_diagnostic": "primary", "notes": "Strong pass 参考阈值 >= 0.55。"},
            {"metric_name": "decile return table", "allowed_next_run": True, "calculated_in_this_task": False, "primary_or_diagnostic": "primary", "notes": "下一阶段允许，仅用于评价，不构造组合。"},
            {"metric_name": "D10-D1 spread", "allowed_next_run": True, "calculated_in_this_task": False, "primary_or_diagnostic": "primary", "notes": "下一阶段允许。"},
            {"metric_name": "positive spread month ratio", "allowed_next_run": True, "calculated_in_this_task": False, "primary_or_diagnostic": "primary", "notes": "下一阶段允许。"},
            {"metric_name": "monotonicity / U-shape / non-monotonic label", "allowed_next_run": True, "calculated_in_this_task": False, "primary_or_diagnostic": "diagnostic", "notes": "判断 decile shape 稳定性。"},
            {"metric_name": "raw vs industry-neutral comparison", "allowed_next_run": True, "calculated_in_this_task": False, "primary_or_diagnostic": "primary", "notes": "同一 complete-case 样本。"},
            {"metric_name": "complete-case comparison", "allowed_next_run": True, "calculated_in_this_task": False, "primary_or_diagnostic": "primary", "notes": "强制执行。"},
            {"metric_name": "small industry group sensitivity", "allowed_next_run": True, "calculated_in_this_task": False, "primary_or_diagnostic": "diagnostic", "notes": "全样本与剔除 small group 两套结果。"},
            {"metric_name": "portfolio return", "allowed_next_run": False, "calculated_in_this_task": False, "primary_or_diagnostic": "forbidden", "notes": "下一阶段仍不允许。"},
            {"metric_name": "backtest", "allowed_next_run": False, "calculated_in_this_task": False, "primary_or_diagnostic": "forbidden", "notes": "下一阶段仍不允许。"},
        ]
    )
    metric_plan.to_csv(OUT_DIR / "evaluation_metric_plan.csv", index=False, encoding="utf-8-sig")

    decision_framework = pd.DataFrame(
        [
            {
                "decision_label": "Strong pass",
                "condition": "Neutral Rank IC positive, D10-D1 positive, positive IC month ratio >= 0.55, D10-D1 not strongly worse than raw, no severe U-shape / unstable decile shape.",
                "interpretation": "行业中性后仍保留 stock-selection signal。",
                "recommended_action": "进入下一阶段组合构建准备，但仍需独立 portfolio prep guardrail。",
            },
            {
                "decision_label": "Partial pass",
                "condition": "Neutral Rank IC positive, but D10-D1 weak or non-monotonic.",
                "interpretation": "存在一定选股信号，但组合构造需谨慎。",
                "recommended_action": "保留候选，优先做稳定性与小行业组敏感性复核。",
            },
            {
                "decision_label": "Industry exposure dependent",
                "condition": "Raw score strong, but neutral score materially decays or reverses.",
                "interpretation": "Raw baseline 可能主要来自行业配置。",
                "recommended_action": "不要直接宣称 stock-selection alpha；保留为 industry-exposed value strategy candidate。",
            },
            {
                "decision_label": "Fail",
                "condition": "Neutral Rank IC <= 0 or D10-D1 <= 0 with poor stability.",
                "interpretation": "行业中性分支缺乏有效信号。",
                "recommended_action": "停止 industry-neutral score branch，保留 raw baseline only if appropriate。",
            },
        ]
    )
    decision_framework.to_csv(OUT_DIR / "evaluation_decision_framework.csv", index=False, encoding="utf-8-sig")

    run_config = {
        "neutral_score_panel_path": str(INPUTS["neutral_score_panel"]),
        "target_column": "fwd_ret_1m",
        "date_column": "month_end",
        "symbol_column": "symbol",
        "raw_score_columns": RAW_SCORE_COLUMNS,
        "neutral_score_columns": NEUTRAL_SCORE_COLUMNS,
        "comparison_pairs": COMPARISON_PAIRS,
        "complete_case_required": True,
        "unique_month_policy_primary": True,
        "old_split_role_policy_secondary": True,
        "small_group_sensitivity_required": True,
        "output_directory_for_next_run": str(ROOT / "output" / "asof_industry_neutral_score_evaluation_run_v0"),
        "calculate_ic_next_run_allowed": True,
        "calculate_decile_next_run_allowed": True,
        "calculate_portfolio_next_run_allowed": False,
        "calculate_portfolio_return_next_run_allowed": False,
        "backtest_allowed_next_run": False,
        "production_allowed_next_run": False,
        "calculated_now": False,
        "raw_eval_reference_summary": str(INPUTS["raw_eval_summary"]),
        "raw_eval_reference_ranking": str(raw_ranking_path),
        "unique_month_reference": str(unique_month_path),
    }
    write_json(OUT_DIR / "asof_industry_neutral_score_eval_run_config_draft.json", run_config)

    guardrails = pd.DataFrame(
        [
            {"guardrail": "neutral_score_panel_ready", "passed": neutral_score_panel_ready, "notes": str(INPUTS["neutral_score_panel"])},
            {"guardrail": "comparison_pairs_generated", "passed": len(COMPARISON_PAIRS) == 2, "notes": "Only BP single and value-quality equal weight primary comparisons."},
            {"guardrail": "complete_case_policy_generated", "passed": True, "notes": "SAME_SYMBOL_MONTH_SAMPLE required."},
            {"guardrail": "small_group_sensitivity_policy_generated", "passed": True, "notes": "Base + exclude-small-group sensitivity required."},
            {"guardrail": "metric_plan_generated", "passed": True, "notes": "No metric calculated now."},
            {"guardrail": "ic_not_calculated", "passed": True, "notes": ""},
            {"guardrail": "d10_d1_not_calculated", "passed": True, "notes": ""},
            {"guardrail": "decile_return_not_calculated", "passed": True, "notes": ""},
            {"guardrail": "portfolio_not_constructed", "passed": True, "notes": ""},
            {"guardrail": "production_not_modified", "passed": True, "notes": ""},
        ]
    )
    guardrails.to_csv(OUT_DIR / "asof_industry_neutral_score_eval_guardrail_checklist.csv", index=False, encoding="utf-8-sig")
    guardrail_checklist_passed = bool(guardrails["passed"].all())

    final_decision = (
        "ASOF_INDUSTRY_NEUTRAL_SCORE_EVAL_PREP_READY_FOR_RUN"
        if neutral_score_panel_ready and guardrail_checklist_passed and small_group_required
        else "ASOF_INDUSTRY_NEUTRAL_SCORE_EVAL_PREP_WATCH_MANUAL_REVIEW_REQUIRED"
    )
    if not neutral_score_panel_ready or not guardrail_checklist_passed:
        final_decision = "ASOF_INDUSTRY_NEUTRAL_SCORE_EVAL_PREP_FAIL"

    plan = f"""# As-Of Industry Neutral Score Evaluation Run Plan

## 样本

- 使用 complete-case sample：同一 `symbol, month_end` 必须同时有 raw score、neutral score、`fwd_ret_1m`、成功 as-of industry join。
- expected_complete_case_rows: {expected_complete_case_rows}

## 下一阶段允许

- 计算 monthly Pearson / Spearman IC、unique-month average IC、IC IR、positive IC month ratio。
- 计算 decile return table 和 D10-D1 spread。
- 输出 raw vs industry-neutral complete-case comparison。
- 输出 small-group sensitivity：全样本与剔除 `small_group_flag=true`。

## 下一阶段仍禁止

- 不构造 portfolio，不计算 portfolio return，不回测，不写 production。
"""
    (OUT_DIR / "next_step_asof_industry_neutral_score_evaluation_run_plan.md").write_text(plan, encoding="utf-8")

    report = f"""# As-Of Industry Neutral Score Evaluation Prep v0

## 结论

- final_decision: `{final_decision}`
- neutral_score_panel_ready: `{neutral_score_panel_ready}`
- expected_complete_case_rows: `{expected_complete_case_rows}`
- small_group_sensitivity_required: `{small_group_required}`
- metrics_calculated_now: `false`

本任务只生成 evaluation run 配置与 guardrail，不计算 IC、D10-D1、decile return、组合收益或回测。
"""
    (OUT_DIR / "asof_industry_neutral_score_eval_prep_report.md").write_text(report, encoding="utf-8")

    summary = {
        "run_timestamp": prereq["run_timestamp"],
        "prerequisites_passed": True,
        "neutral_score_panel_ready": bool(neutral_score_panel_ready),
        "neutral_score_panel_path": str(INPUTS["neutral_score_panel"]),
        "raw_score_columns": RAW_SCORE_COLUMNS,
        "neutral_score_columns": NEUTRAL_SCORE_COLUMNS,
        "comparison_pair_count": len(COMPARISON_PAIRS),
        "complete_case_required": True,
        "expected_complete_case_rows": expected_complete_case_rows,
        "small_group_sensitivity_required": bool(small_group_required),
        "unique_month_policy_primary": True,
        "old_split_role_policy_secondary": True,
        "metric_plan_generated": True,
        "run_config_draft_generated": True,
        "guardrail_checklist_passed": bool(guardrail_checklist_passed),
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
        "recommended_next_step": "运行 asof_industry_neutral_score_evaluation_run_v0，按 complete-case 和 small-group sensitivity 计算 IC/decile 评价；仍不得构造 portfolio 或回测。",
    }
    write_json(OUT_DIR / "asof_industry_neutral_score_eval_prep_summary.json", summary)

    final_qa = pd.DataFrame(
        [
            {"check": "ic_calculated", "value": False, "passed": True},
            {"check": "d10_d1_calculated", "value": False, "passed": True},
            {"check": "decile_return_calculated", "value": False, "passed": True},
            {"check": "portfolio_constructed", "value": False, "passed": True},
            {"check": "portfolio_return_calculated", "value": False, "passed": True},
            {"check": "backtest_run", "value": False, "passed": True},
            {"check": "production_modified", "value": False, "passed": True},
        ]
    )
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")

    terminal_summary = {
        "task_name": TASK_NAME,
        "status": "completed",
        "stdout_log": str(RUN_DIR / "run_stdout.txt"),
        "stderr_log": str(RUN_DIR / "run_stderr.txt"),
        "outputs": [str(p) for p in sorted(OUT_DIR.glob("*")) if p.is_file()],
    }
    write_json(OUT_DIR / "terminal_summary.json", terminal_summary)

    card = f"""# Task Completion Card

- task_name: {TASK_NAME}
- status: completed
- final_decision: {final_decision}
- output_dir: {OUT_DIR}
- run_dir: {RUN_DIR}
- logs: {RUN_DIR / 'run_stdout.txt'} ; {RUN_DIR / 'run_stderr.txt'}
"""
    (OUT_DIR / "task_completion_card.md").write_text(card, encoding="utf-8")
    write_run_state("completed", {"final_decision": final_decision, "summary_path": str(OUT_DIR / "asof_industry_neutral_score_eval_prep_summary.json")})

    del join_qa, neutral_qa, formula_manifest, raw_summary, raw_ranking, unique_month, panel_probe, manifest, metric_plan
    gc.collect()
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": str(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
