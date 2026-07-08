import json
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


TASK_NAME = "DGTW_Adjusted_Portfolio_Eval_Prep"
OUT_DIR = Path("output/dgtw_adjusted_portfolio_evaluation_prep_v0")
RUN_DIR = Path("output/_agent_runs") / TASK_NAME

DGTW_DIR = Path("output/dgtw_benchmark_source_audit_stock_matching_feasibility_v1")
STOCK_MONTH_CANDIDATE = DGTW_DIR / "dgtw_stock_month_matched_benchmark_candidate.parquet"
ALIGNMENT_POLICY = DGTW_DIR / "dgtw_alignment_policy.json"
MATCH_FEASIBILITY = DGTW_DIR / "dgtw_portfolio_matching_feasibility_by_portfolio.csv"
DGTW_SUMMARY = DGTW_DIR / "dgtw_benchmark_source_audit_stock_matching_feasibility_summary.json"
PORTFOLIO_NET_RETURN = Path("output/unified_robust_portfolio_evaluation_run_v0/unified_portfolio_monthly_net_return_by_cost.csv")
PERFORMANCE_SUMMARY = Path("output/unified_robust_portfolio_evaluation_run_v0/unified_portfolio_performance_summary_by_cost.csv")

NEXT_RUN_DIR = Path("output/dgtw_adjusted_portfolio_evaluation_run_v0")

PORTFOLIOS = [
    "ROBUST_VQ_TOP20_EXCLUDE_SOFT_ANOMALY_EQUAL_WEIGHT",
    "ROBUST_VQ_FLAG_CLEAN_TOP50_EQUAL_WEIGHT",
    "ROBUST_VQ_FLAG_CLEAN_TOP50_BUFFER_EQUAL_WEIGHT",
    "ROBUST_VQ_D7_D9_BAND_EQUAL_WEIGHT",
    "ROBUST_VQ_TOP30_PERCENT_EQUAL_WEIGHT",
]
PRIMARY_PORTFOLIO = "ROBUST_VQ_FLAG_CLEAN_TOP50_BUFFER_EQUAL_WEIGHT"
PRIMARY_COST_BPS = 20
MATCHED_WEIGHT_SHARE_THRESHOLD = 0.95


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def write_state(status, checkpoint, notes=None):
    lines = [
        "# DGTW-Adjusted Portfolio Evaluation Prep v0",
        "",
        f"- task_name: {TASK_NAME}",
        f"- status: {status}",
        f"- last_checkpoint: {checkpoint}",
        f"- updated_at: {datetime.now().isoformat(timespec='seconds')}",
        "- resume_command: `python scripts\\prep_dgtw_adjusted_portfolio_evaluation_v0.py > output\\_agent_runs\\DGTW_Adjusted_Portfolio_Eval_Prep\\run_stdout.txt 2> output\\_agent_runs\\DGTW_Adjusted_Portfolio_Eval_Prep\\run_stderr.txt`",
        "- guardrails: prep only; no DGTW benchmark aggregation; no adjusted return; no alpha/beta; no IR; no TE; no weights edits; no production",
    ]
    if notes:
        lines += ["", "## Notes"] + [f"- {x}" for x in notes]
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_row_count(path):
    if not path.exists():
        return 0
    return int(pq.ParquetFile(path).metadata.num_rows)


def read_portfolio_roles():
    if not PORTFOLIO_NET_RETURN.exists():
        return {}
    usecols = ["portfolio_name", "portfolio_role"]
    df = pd.read_csv(PORTFOLIO_NET_RETURN, usecols=usecols)
    df = df[df["portfolio_name"].isin(PORTFOLIOS)].dropna(subset=["portfolio_name"])
    roles = df.drop_duplicates("portfolio_name").set_index("portfolio_name")["portfolio_role"].to_dict()
    del df
    return roles


def expected_month_count():
    if not PORTFOLIO_NET_RETURN.exists():
        return None
    df = pd.read_csv(PORTFOLIO_NET_RETURN, usecols=["portfolio_name", "month_end"])
    df = df[df["portfolio_name"].isin(PORTFOLIOS)]
    count = int(df["month_end"].nunique())
    del df
    return count


def cost_scenarios():
    if not PORTFOLIO_NET_RETURN.exists():
        return []
    df = pd.read_csv(PORTFOLIO_NET_RETURN, usecols=["cost_bps"])
    vals = sorted(pd.to_numeric(df["cost_bps"], errors="coerce").dropna().astype(int).unique().tolist())
    del df
    return vals


def main():
    ensure_dirs()
    write_state("running", "checking prerequisites and generating prep manifests")

    dgtw_summary = load_json(DGTW_SUMMARY) if DGTW_SUMMARY.exists() else {}
    alignment_policy = load_json(ALIGNMENT_POLICY) if ALIGNMENT_POLICY.exists() else {}
    candidate_rows = candidate_row_count(STOCK_MONTH_CANDIDATE)
    feasibility = pd.read_csv(MATCH_FEASIBILITY) if MATCH_FEASIBILITY.exists() else pd.DataFrame()

    required_paths = {
        "stock_month_candidate": STOCK_MONTH_CANDIDATE.exists(),
        "alignment_policy": ALIGNMENT_POLICY.exists(),
        "match_feasibility": MATCH_FEASIBILITY.exists(),
        "dgtw_summary": DGTW_SUMMARY.exists(),
        "portfolio_net_return": PORTFOLIO_NET_RETURN.exists(),
        "performance_summary": PERFORMANCE_SUMMARY.exists(),
    }
    stock_month_candidate_ready = bool(
        required_paths["stock_month_candidate"]
        and candidate_rows > 0
        and dgtw_summary.get("stock_month_candidate_generated") is True
    )
    avg_match = float(dgtw_summary.get("avg_final_dgtw_match_ratio", 0.0) or 0.0)
    lowest_match = float(dgtw_summary.get("lowest_portfolio_final_match_ratio", 0.0) or 0.0)
    prerequisites_passed = bool(all(required_paths.values()) and stock_month_candidate_ready and avg_match >= 0.95)

    prereq = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "required_paths": {k: str(v) for k, v in {
            "stock_month_candidate": STOCK_MONTH_CANDIDATE,
            "alignment_policy": ALIGNMENT_POLICY,
            "match_feasibility": MATCH_FEASIBILITY,
            "dgtw_summary": DGTW_SUMMARY,
            "portfolio_net_return": PORTFOLIO_NET_RETURN,
            "performance_summary": PERFORMANCE_SUMMARY,
        }.items()},
        "required_files_found": required_paths,
        "stock_month_candidate_row_count": candidate_rows,
        "stock_month_candidate_ready": stock_month_candidate_ready,
        "avg_final_dgtw_match_ratio": avg_match,
        "lowest_portfolio_final_match_ratio": lowest_match,
        "prerequisites_passed": prerequisites_passed,
    }
    (OUT_DIR / "dgtw_adjusted_eval_prep_prerequisite_check.json").write_text(json.dumps(prereq, ensure_ascii=False, indent=2), encoding="utf-8")

    roles = read_portfolio_roles()
    manifest_rows = []
    for p in PORTFOLIOS:
        role = roles.get(p, "Unknown")
        notes = "Primary candidate for 20bps DGTW-adjusted check" if p == PRIMARY_PORTFOLIO else "Included in next DGTW-adjusted evaluation run"
        manifest_rows.append({
            "portfolio_name": p,
            "portfolio_role": role,
            "include_in_dgtw_eval": True,
            "primary_candidate_flag": p == PRIMARY_PORTFOLIO,
            "notes": notes,
        })
    pd.DataFrame(manifest_rows).to_csv(OUT_DIR / "dgtw_adjusted_portfolio_manifest.csv", index=False, encoding="utf-8-sig")

    matching_policy = {
        "stock_month_candidate_path": str(STOCK_MONTH_CANDIDATE),
        "recommended_tradingyear_mapping_rule": "RULE_B_JULY_TO_JUNE",
        "recommended_is_not_bse_policy": 1,
        "benchmark_return_unit": "DECIMAL_RETURN",
        "unit_conversion_needed": False,
        "missing_match_policy": {
            "unmatched_stock_month_dgtw_benchmark_return": "missing",
            "portfolio_month_aggregation_must_output_matched_weight_share": True,
            "low_coverage_flag": "LOW_DGTW_MATCH_COVERAGE",
            "no_imputation": True,
            "no_index_or_industry_fill": True,
        },
        "matched_weight_share_threshold": MATCHED_WEIGHT_SHARE_THRESHOLD,
        "renormalization_policy": {
            "primary": "do_not_renormalize_unmatched_weight",
            "unmatched_exposure": "record_as_missing_exposure",
            "matched_only_normalized_sensitivity": "required_next_run",
        },
        "sensitivity_required": True,
    }
    (OUT_DIR / "dgtw_adjusted_matching_policy.json").write_text(json.dumps(matching_policy, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_rows = [
        ("portfolio_dgtw_benchmark_return", True, False, "primary", "Next run may aggregate matched stock-month DGTW benchmark by portfolio weights."),
        ("portfolio_dgtw_adjusted_return", True, False, "primary", "Next run may calculate portfolio_net_return - portfolio_dgtw_benchmark_return."),
        ("mean_monthly_dgtw_adjusted_return", True, False, "primary", "Performance metric for adjusted return series."),
        ("annualized_dgtw_adjusted_return_approximation", True, False, "primary", "Use monthly mean approximation in next run."),
        ("dgtw_adjusted_t_stat", True, False, "primary", "Primary statistical diagnostic in next run."),
        ("positive_adjusted_month_ratio", True, False, "primary", "Primary candidate check requires > 0.50."),
        ("cumulative_dgtw_adjusted_return", True, False, "primary", "Explicitly not calculated in this prep task."),
        ("dgtw_adjusted_sharpe", True, False, "primary", "Adjusted risk-return diagnostic."),
        ("worst_adjusted_month", True, False, "diagnostic", "Downside diagnostic."),
        ("best_adjusted_month", True, False, "diagnostic", "Upside diagnostic."),
        ("alpha_beta_regression", False, False, "forbidden", "Out of scope for next DGTW characteristic-adjusted run unless separately approved."),
        ("information_ratio", False, False, "forbidden", "Forbidden by current task guardrail."),
        ("tracking_error", False, False, "forbidden", "Forbidden by current task guardrail."),
    ]
    pd.DataFrame(metric_rows, columns=["metric_name", "allowed_next_run", "calculated_in_this_task", "primary_or_diagnostic", "notes"]).to_csv(
        OUT_DIR / "dgtw_adjusted_metric_plan.csv", index=False, encoding="utf-8-sig"
    )

    sample_policy = {
        "portfolio_month_count_expected": expected_month_count(),
        "cost_scenarios": cost_scenarios(),
        "primary_cost_bps": PRIMARY_COST_BPS,
        "low_match_coverage_policy": {
            "threshold": MATCHED_WEIGHT_SHARE_THRESHOLD,
            "flag": "LOW_DGTW_MATCH_COVERAGE",
            "main_analysis_requirement": "matched_weight_share >= 0.95",
        },
        "no_imputation_policy": "Do not fill missing stock-month DGTW benchmark returns with mean, index, or industry returns.",
        "matched_only_sensitivity_required": True,
    }
    (OUT_DIR / "dgtw_adjusted_sample_policy.json").write_text(json.dumps(sample_policy, ensure_ascii=False, indent=2), encoding="utf-8")

    run_config = {
        "stock_month_dgtw_candidate_path": str(STOCK_MONTH_CANDIDATE),
        "portfolio_net_return_path": str(PORTFOLIO_NET_RETURN),
        "portfolio_performance_summary_path": str(PERFORMANCE_SUMMARY),
        "output_directory_for_next_run": str(NEXT_RUN_DIR),
        "portfolios_to_evaluate": PORTFOLIOS,
        "primary_portfolio": PRIMARY_PORTFOLIO,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "recommended_tradingyear_mapping_rule": "RULE_B_JULY_TO_JUNE",
        "recommended_is_not_bse_policy": 1,
        "benchmark_return_unit": "DECIMAL_RETURN",
        "unit_conversion_needed": False,
        "matched_weight_share_threshold": MATCHED_WEIGHT_SHARE_THRESHOLD,
        "calculate_dgtw_benchmark_return_next_run_allowed": True,
        "calculate_dgtw_adjusted_return_next_run_allowed": True,
        "calculate_cumulative_adjusted_return_next_run_allowed": True,
        "calculate_adjusted_tstat_next_run_allowed": True,
        "production_allowed_next_run": False,
    }
    (OUT_DIR / "dgtw_adjusted_run_config_draft.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    guardrail = {
        "portfolio_weights_modified": False,
        "portfolio_weights_reconstructed": False,
        "portfolio_dgtw_benchmark_return_calculated": False,
        "portfolio_dgtw_adjusted_return_calculated": False,
        "cumulative_adjusted_return_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "training_run": False,
        "shap_calculated": False,
        "production_modified": False,
    }
    guardrail["guardrail_pass"] = not any(guardrail.values())
    pd.DataFrame([guardrail]).to_csv(OUT_DIR / "dgtw_adjusted_guardrail_checklist.csv", index=False, encoding="utf-8-sig")

    plan = [
        "# Next Step DGTW-Adjusted Evaluation Run Plan",
        "",
        "## 当前问题",
        "",
        "- DGTW source 已可用。",
        "- stock-month matched DGTW benchmark candidate 已生成。",
        "- portfolio-level DGTW benchmark return 尚未计算。",
        "- DGTW adjusted return 尚未计算。",
        "- 下一步需要将 stock-month DGTW benchmark 按 portfolio weights 聚合到 portfolio-month 层面。",
        "- 本 prep 任务不得计算聚合收益，当前脚本只生成 plan / manifest / config。",
        "",
        "## DGTW 定位",
        "",
        "- DGTW 不是 official market benchmark。",
        "- DGTW 是 characteristic-matched benchmark。",
        "- 它用于检验策略是否在控制 size / book-to-market / momentum 后仍有超额。",
        "",
        "## 下一阶段核心规则",
        "",
        "- 使用 `RULE_B_JULY_TO_JUNE`，不要重新发明 TradingYear mapping rule。",
        "- 使用 `IsNotBSE = 1`。",
        "- `BenchmarkReturns` 为 decimal return，不需要单位转换。",
        "- 缺失 stock-month match 不填充；portfolio-month 必须输出 `matched_weight_share`。",
        "- 主分析要求 `matched_weight_share >= 0.95`，低于阈值标记 `LOW_DGTW_MATCH_COVERAGE`。",
        "- 主口径不重归一化 unmatched weight；另输出 matched-only normalized sensitivity。",
        "",
        "## 下一阶段允许计算",
        "",
        "- portfolio DGTW benchmark return。",
        "- portfolio DGTW-adjusted return。",
        "- adjusted return 的 mean、annualized approximation、t-stat、positive month ratio、cumulative return、Sharpe、best/worst month。",
        "",
        "## 下一阶段仍禁止",
        "",
        "- production。",
        "- 修改或重构 portfolio weights。",
        "- alpha/beta、information ratio、tracking error，除非另开任务明确批准。",
    ]
    (OUT_DIR / "next_step_dgtw_adjusted_evaluation_run_plan.md").write_text("\n".join(plan) + "\n", encoding="utf-8")

    if not guardrail["guardrail_pass"]:
        final_decision = "DGTW_ADJUSTED_EVAL_PREP_FAIL_GUARDRAIL"
    elif not prerequisites_passed:
        final_decision = "DGTW_ADJUSTED_EVAL_PREP_FAIL"
    elif dgtw_summary.get("dgtw_cell_coverage_pass") is False:
        final_decision = "DGTW_ADJUSTED_EVAL_PREP_WATCH_CELL_COVERAGE_CAVEAT"
    else:
        final_decision = "DGTW_ADJUSTED_EVAL_PREP_READY_FOR_RUN"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prerequisites_passed,
        "dgtw_stock_month_candidate_ready": stock_month_candidate_ready,
        "stock_month_candidate_row_count": candidate_rows,
        "recommended_tradingyear_mapping_rule": dgtw_summary.get("recommended_tradingyear_mapping_rule", "RULE_B_JULY_TO_JUNE"),
        "recommended_is_not_bse_policy": dgtw_summary.get("recommended_is_not_bse_policy", 1),
        "benchmark_return_unit_detected": dgtw_summary.get("benchmark_return_unit_detected", "DECIMAL_RETURN"),
        "unit_conversion_needed": bool(dgtw_summary.get("unit_conversion_needed", False)),
        "dgtw_cell_coverage_pass": bool(dgtw_summary.get("dgtw_cell_coverage_pass", False)),
        "avg_final_dgtw_match_ratio": avg_match,
        "lowest_portfolio_final_match_ratio": lowest_match,
        "portfolio_count_planned": len(PORTFOLIOS),
        "portfolios_planned": PORTFOLIOS,
        "primary_portfolio": PRIMARY_PORTFOLIO,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "matched_weight_share_threshold": MATCHED_WEIGHT_SHARE_THRESHOLD,
        "matched_only_sensitivity_required": True,
        "calculate_dgtw_benchmark_return_next_run_allowed": True,
        "calculate_dgtw_adjusted_return_next_run_allowed": True,
        "calculate_cumulative_adjusted_return_next_run_allowed": True,
        "calculate_adjusted_tstat_next_run_allowed": True,
        "portfolio_weights_modified": False,
        "portfolio_weights_reconstructed": False,
        "portfolio_dgtw_benchmark_return_calculated": False,
        "portfolio_dgtw_adjusted_return_calculated": False,
        "cumulative_adjusted_return_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "training_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "final_decision": final_decision,
        "recommended_next_step": "进入 DGTW-adjusted portfolio evaluation run；执行前保留 dgtw_cell_coverage_pass=false 的 source caveat。" if "WATCH_CELL_COVERAGE_CAVEAT" in final_decision else "进入 DGTW-adjusted portfolio evaluation run。",
    }
    (OUT_DIR / "dgtw_adjusted_portfolio_evaluation_prep_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# DGTW-Adjusted Portfolio Evaluation Prep v0",
        "",
        f"- final_decision: {final_decision}",
        f"- prerequisites_passed: {prerequisites_passed}",
        f"- stock_month_candidate_row_count: {candidate_rows}",
        f"- avg_final_dgtw_match_ratio: {avg_match}",
        f"- lowest_portfolio_final_match_ratio: {lowest_match}",
        f"- dgtw_cell_coverage_pass: {summary['dgtw_cell_coverage_pass']}",
        "",
        "## 当前问题",
        "",
        "- DGTW source 已可用。",
        "- stock-month matched DGTW benchmark candidate 已生成。",
        "- portfolio-level DGTW benchmark return 尚未计算。",
        "- DGTW adjusted return 尚未计算。",
        "- 下一步需要将 stock-month DGTW benchmark 按 portfolio weights 聚合到 portfolio-month 层面。",
        "- 本 prep 任务没有计算聚合收益。",
        "",
        "## Guardrail",
        "",
        "- 未修改或重构 portfolio weights。",
        "- 未计算 portfolio DGTW benchmark return、DGTW-adjusted return、cumulative adjusted return。",
        "- 未计算 alpha/beta、information ratio、tracking error。",
        "- 未训练、未 SHAP、未写 production。",
    ]
    (OUT_DIR / "dgtw_adjusted_portfolio_evaluation_prep_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    final_qa = pd.DataFrame([{
        "final_decision": final_decision,
        "prerequisites_passed": prerequisites_passed,
        "stock_month_candidate_ready": stock_month_candidate_ready,
        "run_config_generated": True,
        "guardrail_pass": guardrail["guardrail_pass"],
        "required_outputs_created": True,
    }])
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")

    terminal_summary = {
        "task_name": TASK_NAME,
        "status": "completed",
        "stdout_log": str(RUN_DIR / "run_stdout.txt"),
        "stderr_log": str(RUN_DIR / "run_stderr.txt"),
        "summary_json": str(OUT_DIR / "dgtw_adjusted_portfolio_evaluation_prep_summary.json"),
        "final_decision": final_decision,
    }
    (OUT_DIR / "terminal_summary.json").write_text(json.dumps(terminal_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "task_completion_card.md").write_text(
        "\n".join([
            "# Task Completion Card",
            "",
            f"- task_name: {TASK_NAME}",
            "- status: completed",
            f"- final_decision: {final_decision}",
            f"- summary_json: {OUT_DIR / 'dgtw_adjusted_portfolio_evaluation_prep_summary.json'}",
            f"- final_qa: {OUT_DIR / 'final_qa.csv'}",
        ]) + "\n",
        encoding="utf-8",
    )
    write_state("completed", "all required outputs written", [f"final_decision: {final_decision}"])
    print(json.dumps(terminal_summary, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ensure_dirs()
        err = traceback.format_exc()
        (RUN_DIR / "last_error.txt").write_text(err, encoding="utf-8")
        write_state("failed", "exception captured in last_error.txt")
        raise
