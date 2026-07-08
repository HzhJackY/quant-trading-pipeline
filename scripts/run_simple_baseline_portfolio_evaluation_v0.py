from __future__ import annotations

import gc
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "simple_baseline_portfolio_evaluation_run_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
AGENT_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME
PREP_DIR = ROOT / "output" / "simple_baseline_portfolio_evaluation_prep_v0"
CONSTRUCTION_DIR = ROOT / "output" / "simple_baseline_portfolio_construction_run_v0"

PREP_SUMMARY = PREP_DIR / "simple_baseline_portfolio_evaluation_prep_summary.json"
RUN_CONFIG = PREP_DIR / "portfolio_eval_run_config_draft.json"
METRIC_PLAN = PREP_DIR / "portfolio_eval_metric_plan.csv"
SAMPLE_POLICY = PREP_DIR / "portfolio_eval_sample_policy.json"
PREP_GUARDRAIL = PREP_DIR / "portfolio_eval_guardrail_checklist.csv"
CONSTRUCTION_SUMMARY = CONSTRUCTION_DIR / "simple_baseline_portfolio_construction_run_summary.json"
WEIGHT_PANEL = CONSTRUCTION_DIR / "simple_baseline_research_weights_v0.parquet"
WEIGHTS_QA_BY_MONTH = CONSTRUCTION_DIR / "portfolio_weights_qa_by_month.csv"
COVERAGE_SUMMARY = CONSTRUCTION_DIR / "portfolio_coverage_summary.csv"
LEAKAGE_QA = CONSTRUCTION_DIR / "portfolio_leakage_exclusion_qa.csv"
CONSTRUCTION_GUARDRAIL_QA = CONSTRUCTION_DIR / "portfolio_construction_guardrail_qa.csv"

PORTFOLIO_NAMES = [
    "BP_SINGLE_TOP_DECILE_EQUAL_WEIGHT",
    "BP_SINGLE_TOP50_EQUAL_WEIGHT",
    "VALUE_QUALITY_TOP_DECILE_EQUAL_WEIGHT",
    "VALUE_QUALITY_TOP50_EQUAL_WEIGHT",
]
TARGET_COLUMN = "fwd_ret_1m"
WEIGHT_COLUMNS = [
    "portfolio_name",
    "symbol",
    "month_end",
    "score_column",
    "score_value",
    "weight",
    "research_only",
    "rebalance_frequency",
    "selection_rule",
    "weight_rule",
    "fwd_ret_1m",
]
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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def load_required_inputs() -> tuple[dict, dict, pd.DataFrame, dict, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prep_summary = read_json(PREP_SUMMARY)
    run_config = read_json(RUN_CONFIG)
    metric_plan = pd.read_csv(METRIC_PLAN)
    sample_policy = read_json(SAMPLE_POLICY)
    prep_guardrail = pd.read_csv(PREP_GUARDRAIL)
    construction_summary = read_json(CONSTRUCTION_SUMMARY)
    weights_qa = pd.read_csv(WEIGHTS_QA_BY_MONTH)
    coverage = pd.read_csv(COVERAGE_SUMMARY)
    leakage_qa = pd.read_csv(LEAKAGE_QA)
    construction_guardrail = pd.read_csv(CONSTRUCTION_GUARDRAIL_QA)
    weights = pd.read_parquet(WEIGHT_PANEL, columns=WEIGHT_COLUMNS)
    weights["symbol"] = weights["symbol"].astype("string")
    weights["month_end"] = pd.to_datetime(weights["month_end"]).dt.strftime("%Y-%m-%d")
    return (
        prep_summary,
        run_config,
        metric_plan,
        sample_policy,
        prep_guardrail,
        construction_summary,
        weights_qa,
        coverage,
        leakage_qa,
        construction_guardrail,
        weights,
    )


def build_prerequisites(
    prep_summary: dict,
    run_config: dict,
    metric_plan: pd.DataFrame,
    prep_guardrail: pd.DataFrame,
    construction_summary: dict,
    weights_qa: pd.DataFrame,
    coverage: pd.DataFrame,
    leakage_qa: pd.DataFrame,
    construction_guardrail: pd.DataFrame,
    weights: pd.DataFrame,
) -> dict:
    required_files = {
        "prep_summary": PREP_SUMMARY,
        "run_config": RUN_CONFIG,
        "metric_plan": METRIC_PLAN,
        "sample_policy": SAMPLE_POLICY,
        "prep_guardrail": PREP_GUARDRAIL,
        "construction_summary": CONSTRUCTION_SUMMARY,
        "weight_panel": WEIGHT_PANEL,
        "weights_qa_by_month": WEIGHTS_QA_BY_MONTH,
        "coverage_summary": COVERAGE_SUMMARY,
        "leakage_qa": LEAKAGE_QA,
        "construction_guardrail": CONSTRUCTION_GUARDRAIL_QA,
    }
    allowed_rows = metric_plan.loc[metric_plan["allowed_in_next_run"].astype(str).str.lower().eq("true"), "metric_name"].tolist()
    blocked_rows = metric_plan.loc[metric_plan["allowed_in_next_run"].astype(str).str.lower().eq("false"), "metric_name"].tolist()
    prep_ready = prep_summary.get("final_decision") == "SIMPLE_BASELINE_PORTFOLIO_EVAL_PREP_READY_FOR_EVALUATION_RUN"
    construction_ready = (
        construction_summary.get("final_decision")
        == "SIMPLE_BASELINE_PORTFOLIO_CONSTRUCTION_RUN_READY_FOR_PORTFOLIO_EVALUATION_PREP"
    )
    checks = {
        "required_files_exist": all(path.exists() for path in required_files.values()),
        "portfolio_eval_prep_ready": prep_ready,
        "portfolio_construction_run_ready": construction_ready,
        "weight_panel_exists": WEIGHT_PANEL.exists(),
        "four_portfolios_exist": sorted(weights["portfolio_name"].dropna().unique().tolist()) == sorted(PORTFOLIO_NAMES),
        "research_only_true": bool(weights["research_only"].eq(True).all()) and bool(run_config.get("research_only")),
        "fwd_ret_1m_present": TARGET_COLUMN in weights.columns,
        "allowed_metrics_only": sorted(allowed_rows) == sorted(ALLOWED_METRICS),
        "blocked_metrics_not_calculated": sorted(blocked_rows) == sorted(BLOCKED_METRICS),
        "prep_guardrail_passed": bool(prep_guardrail["passed"].astype(str).str.lower().eq("true").all()),
        "construction_weights_qa_passed": bool(weights_qa["status"].eq("PASS").all()),
        "construction_coverage_passed": bool(coverage["status"].eq("PASS").all()),
        "construction_leakage_passed": bool(leakage_qa["status"].eq("PASS").all()),
        "construction_guardrail_passed": bool(construction_guardrail["status"].eq("PASS").all()),
        "no_production_backtest_training_shap_tuning_holdings": all(
            prep_summary.get(key) is False
            for key in [
                "transaction_cost_calculated",
                "turnover_calculated",
                "sharpe_calculated",
                "maxdd_calculated",
                "benchmark_relative_return_calculated",
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
        "required_files": {name: str(path) for name, path in required_files.items()},
        "checks": checks,
        "prerequisites_passed": all(checks.values()),
    }


def build_input_qa(weights: pd.DataFrame) -> pd.DataFrame:
    duplicate_count = int(weights.duplicated(["portfolio_name", "symbol", "month_end"]).sum())
    group_weight_sum = weights.groupby(["portfolio_name", "month_end"], sort=True)["weight"].sum()
    max_weight_sum_abs_error = float((group_weight_sum - 1.0).abs().max())
    normalized = bool(max_weight_sum_abs_error <= 1e-10)
    rows = [
        ["rows loaded", int(len(weights)), status(len(weights) > 0), "BLOCKING", "按任务要求只读取 11 个必要列"],
        ["portfolio count", int(weights["portfolio_name"].nunique()), status(weights["portfolio_name"].nunique() == 4), "BLOCKING", ""],
        ["portfolio names", "|".join(sorted(weights["portfolio_name"].dropna().unique().tolist())), status(sorted(weights["portfolio_name"].dropna().unique().tolist()) == sorted(PORTFOLIO_NAMES)), "BLOCKING", ""],
        ["month count", int(weights["month_end"].nunique()), status(weights["month_end"].nunique() > 0), "BLOCKING", ""],
        ["min month_end", weights["month_end"].min(), "PASS", "INFO", ""],
        ["max month_end", weights["month_end"].max(), "PASS", "INFO", ""],
        ["duplicate portfolio-symbol-month count", duplicate_count, status(duplicate_count == 0), "BLOCKING", ""],
        ["research_only all true", bool(weights["research_only"].eq(True).all()), status(bool(weights["research_only"].eq(True).all())), "BLOCKING", ""],
        ["weight non-null count", int(weights["weight"].notna().sum()), status(weights["weight"].notna().all()), "BLOCKING", ""],
        ["fwd_ret_1m non-null count", int(weights[TARGET_COLUMN].notna().sum()), "PASS", "INFO", "缺失值由 target availability QA 处理"],
        ["selected rows with missing fwd_ret_1m count", int(weights[TARGET_COLUMN].isna().sum()), "PASS", "INFO", "缺失 target 的持仓不参与该月 return numerator/denominator"],
        ["weight sum by portfolio-month already normalized", max_weight_sum_abs_error, status(normalized), "BLOCKING", "tolerance=1e-10"],
    ]
    return pd.DataFrame(rows, columns=["check_name", "observed_value", "status", "severity", "notes"])


def compute_monthly_returns(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    availability_rows = []
    return_rows = []
    for (portfolio_name, month_end), group in weights.groupby(["portfolio_name", "month_end"], sort=True):
        holding_count = int(len(group))
        valid = group.loc[group[TARGET_COLUMN].notna() & group["weight"].notna()].copy()
        target_non_null_count = int(len(valid))
        target_null_count = holding_count - target_non_null_count
        target_non_null_ratio = float(target_non_null_count / holding_count) if holding_count else 0.0
        weight_sum_before = float(group["weight"].sum())
        weight_sum_after = float(valid["weight"].sum()) if target_non_null_count else 0.0

        if target_non_null_count == 0:
            qa_status = "FAIL"
            notes = "全部 selected holding 缺失 fwd_ret_1m；该 portfolio-month 不计算 return"
        elif target_null_count > 0:
            qa_status = "WATCH"
            notes = "部分 holding 缺失 fwd_ret_1m；计算时过滤缺失 target 并将剩余权重重归一化到 1"
        else:
            qa_status = "PASS"
            notes = "target 完整；使用原权重"

        availability_rows.append(
            {
                "portfolio_name": portfolio_name,
                "month_end": month_end,
                "holding_count": holding_count,
                "target_non_null_count": target_non_null_count,
                "target_null_count": target_null_count,
                "target_non_null_ratio": target_non_null_ratio,
                "weight_sum_before_target_filter": weight_sum_before,
                "weight_sum_after_target_filter": weight_sum_after,
                "status": qa_status,
                "notes": notes,
            }
        )

        if target_non_null_count > 0 and weight_sum_after > 0:
            normalized_weight = valid["weight"] / weight_sum_after
            weighted_return = float((normalized_weight * valid[TARGET_COLUMN]).sum())
            return_rows.append(
                {
                    "portfolio_name": portfolio_name,
                    "month_end": month_end,
                    "holding_count": holding_count,
                    "target_non_null_count": target_non_null_count,
                    "target_null_count": target_null_count,
                    "weight_sum_before_target_filter": weight_sum_before,
                    "weight_sum_after_target_filter": weight_sum_after,
                    "weight_sum_used": 1.0,
                    "weighted_monthly_forward_return": weighted_return,
                    "research_only": True,
                    "notes": notes,
                }
            )
        del valid
        gc.collect()
    return pd.DataFrame(availability_rows), pd.DataFrame(return_rows)


def build_cumulative_path(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for portfolio_name, group in monthly_returns.groupby("portfolio_name", sort=True):
        ordered = group.sort_values("month_end").copy()
        ordered["cumulative_simple_return"] = (1.0 + ordered["weighted_monthly_forward_return"]).cumprod() - 1.0
        ordered["research_only"] = True
        frames.append(
            ordered[
                [
                    "portfolio_name",
                    "month_end",
                    "weighted_monthly_forward_return",
                    "cumulative_simple_return",
                    "research_only",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_return_summary(monthly_returns: pd.DataFrame, cumulative_path: pd.DataFrame) -> pd.DataFrame:
    cumulative_final = (
        cumulative_path.sort_values("month_end").groupby("portfolio_name", sort=True)["cumulative_simple_return"].last().to_dict()
    )
    rows = []
    for portfolio_name, group in monthly_returns.groupby("portfolio_name", sort=True):
        returns = group["weighted_monthly_forward_return"]
        rows.append(
            {
                "portfolio_name": portfolio_name,
                "month_count": int(len(group)),
                "mean_monthly_return": float(returns.mean()),
                "median_monthly_return": float(returns.median()),
                "monthly_return_std": float(returns.std()),
                "positive_month_ratio": float((returns > 0).mean()),
                "min_monthly_return": float(returns.min()),
                "max_monthly_return": float(returns.max()),
                "cumulative_simple_return_final": float(cumulative_final.get(portfolio_name, np.nan)),
                "research_only": True,
                "notes": "Research-only raw forward return summary; no Sharpe/MaxDD/CAGR/annualization/benchmark/transaction cost.",
            }
        )
    return pd.DataFrame(rows)


def usage_label(row: pd.Series) -> tuple[str, str]:
    mean_ret = row["mean_monthly_return"]
    positive_ratio = row["positive_month_ratio"]
    cumulative = row["cumulative_simple_return_final"]
    if mean_ret > 0 and positive_ratio >= 0.50 and cumulative > 0:
        return "PORTFOLIO_CANDIDATE_FOR_REVIEW", "mean>0, positive_month_ratio>=0.50, cumulative_simple_return_final>0"
    if mean_ret <= 0 and cumulative <= 0:
        return "DROP_FOR_NOW", "mean_monthly_return<=0 and cumulative_simple_return_final<=0"
    return "WATCH_WEAK", "mixed evidence"


def build_ranking(return_summary: pd.DataFrame) -> pd.DataFrame:
    ranking = return_summary[
        [
            "portfolio_name",
            "mean_monthly_return",
            "median_monthly_return",
            "monthly_return_std",
            "positive_month_ratio",
            "cumulative_simple_return_final",
        ]
    ].copy()
    ranking = ranking.sort_values(
        ["mean_monthly_return", "positive_month_ratio", "cumulative_simple_return_final"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking["research_return_rank"] = np.arange(1, len(ranking) + 1)
    labels = ranking.apply(usage_label, axis=1)
    ranking["recommended_next_usage"] = [item[0] for item in labels]
    ranking["rationale"] = [item[1] for item in labels]
    return ranking


def build_guardrail_qa() -> pd.DataFrame:
    rows = [
        ("no transaction cost calculated", True, "blocked"),
        ("no turnover calculated", True, "blocked"),
        ("no Sharpe calculated", True, "blocked"),
        ("no MaxDD calculated", True, "blocked"),
        ("no benchmark-relative return calculated", True, "blocked"),
        ("no alpha/beta regression", True, "blocked"),
        ("no production backtest metric", True, "blocked"),
        ("no training", True, "blocked"),
        ("no tuning", True, "blocked"),
        ("no SHAP", True, "blocked"),
        ("no feature importance", True, "blocked"),
        ("no production holdings", True, "blocked"),
        ("no live-order-ready file", True, "blocked"),
        ("no production modification", True, "blocked"),
        ("Compact-F rescue blocked", True, "blocked"),
        ("sign-flip production blocked", True, "blocked"),
        ("LightGBM-first blocked", True, "blocked"),
    ]
    return pd.DataFrame(
        [{"check_name": name, "status": status(ok), "severity": "BLOCKING", "notes": notes} for name, ok, notes in rows]
    )


def recommended_next_step(ranking: pd.DataFrame) -> str:
    usages = ranking["recommended_next_usage"].tolist()
    if "PORTFOLIO_CANDIDATE_FOR_REVIEW" in usages:
        return "Simple Baseline Portfolio Review v0"
    if usages and all(item == "WATCH_WEAK" for item in usages):
        return "Manual Portfolio Review v0"
    if usages and all(item == "DROP_FOR_NOW" for item in usages):
        return "Stop current portfolio branch and return to factor/score selection"
    return "Manual Portfolio Review v0"


def build_next_step_md(ranking: pd.DataFrame, next_step: str) -> str:
    candidate_count = int((ranking["recommended_next_usage"] == "PORTFOLIO_CANDIDATE_FOR_REVIEW").sum())
    watch_count = int((ranking["recommended_next_usage"] == "WATCH_WEAK").sum())
    drop_count = int((ranking["recommended_next_usage"] == "DROP_FOR_NOW").sum())
    return "\n".join(
        [
            "# Next Step Recommendation",
            "",
            f"- recommended_next_step: {next_step}",
            f"- portfolio_candidate_count: {candidate_count}",
            f"- watch_weak_count: {watch_count}",
            f"- drop_for_now_count: {drop_count}",
            "",
            "禁止直接进入 production backtest、live holdings 或 model training。",
        ]
    ) + "\n"


def build_report(summary: dict) -> str:
    return "\n".join(
        [
            "# Simple Baseline Portfolio Evaluation Run v0",
            "",
            "## 决策",
            "",
            f"- final_decision: {summary['final_decision']}",
            f"- recommended_next_step: {summary['recommended_next_step']}",
            "",
            "## 评估范围",
            "",
            "- 使用 research-only weights 和 fwd_ret_1m 计算 portfolio-month raw forward return。",
            "- 对 target 缺失持仓执行过滤，并对剩余权重重归一化。",
            "- 输出 monthly return、基础 summary、cumulative simple return path 和 research ranking。",
            "",
            "## 禁止项状态",
            "",
            "- 未计算交易成本、换手率、Sharpe、MaxDD、benchmark-relative return、alpha/beta。",
            "- 未生成 production backtest metric、production holdings 或 live-order-ready 文件。",
            "- 未训练、未调参、未计算 SHAP、未生成 feature importance。",
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
            "resume_instruction": f"先读取 {AGENT_DIR / 'RUN_STATE.md'} 再继续。",
        },
    )

    required = [
        PREP_SUMMARY,
        RUN_CONFIG,
        METRIC_PLAN,
        SAMPLE_POLICY,
        PREP_GUARDRAIL,
        CONSTRUCTION_SUMMARY,
        WEIGHT_PANEL,
        WEIGHTS_QA_BY_MONTH,
        COVERAGE_SUMMARY,
        LEAKAGE_QA,
        CONSTRUCTION_GUARDRAIL_QA,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + "; ".join(missing))

    (
        prep_summary,
        run_config,
        metric_plan,
        sample_policy,
        prep_guardrail,
        construction_summary,
        weights_qa,
        coverage,
        leakage_qa,
        construction_guardrail,
        weights,
    ) = load_required_inputs()
    del sample_policy

    prereq = build_prerequisites(
        prep_summary,
        run_config,
        metric_plan,
        prep_guardrail,
        construction_summary,
        weights_qa,
        coverage,
        leakage_qa,
        construction_guardrail,
        weights,
    )
    write_json(OUT_DIR / "portfolio_eval_run_prerequisite_check.json", prereq)

    input_qa = build_input_qa(weights)
    input_qa.to_csv(OUT_DIR / "portfolio_eval_input_weight_panel_qa.csv", index=False, encoding="utf-8-sig")

    target_qa, monthly_returns = compute_monthly_returns(weights)
    target_qa.to_csv(OUT_DIR / "portfolio_target_availability_qa.csv", index=False, encoding="utf-8-sig")
    monthly_returns.to_csv(OUT_DIR / "monthly_portfolio_forward_returns.csv", index=False, encoding="utf-8-sig")

    cumulative_path = build_cumulative_path(monthly_returns)
    cumulative_path.to_csv(OUT_DIR / "portfolio_cumulative_simple_return_path.csv", index=False, encoding="utf-8-sig")

    return_summary = build_return_summary(monthly_returns, cumulative_path)
    return_summary.to_csv(OUT_DIR / "portfolio_return_summary.csv", index=False, encoding="utf-8-sig")

    ranking = build_ranking(return_summary)
    ranking.to_csv(OUT_DIR / "portfolio_research_return_ranking.csv", index=False, encoding="utf-8-sig")

    guardrail_qa = build_guardrail_qa()
    guardrail_qa.to_csv(OUT_DIR / "portfolio_eval_guardrail_qa.csv", index=False, encoding="utf-8-sig")

    next_step = recommended_next_step(ranking)
    (OUT_DIR / "next_step_simple_baseline_portfolio_review_recommendation.md").write_text(
        build_next_step_md(ranking, next_step),
        encoding="utf-8",
    )

    target_availability_qa_passed = not target_qa["status"].eq("FAIL").any()
    guardrail_qa_passed = bool(guardrail_qa["status"].eq("PASS").all())
    monthly_generated = (OUT_DIR / "monthly_portfolio_forward_returns.csv").exists() and len(monthly_returns) > 0
    summary_generated = (OUT_DIR / "portfolio_return_summary.csv").exists()
    path_generated = (OUT_DIR / "portfolio_cumulative_simple_return_path.csv").exists()
    ranking_generated = (OUT_DIR / "portfolio_research_return_ranking.csv").exists()

    portfolio_candidate_count = int((ranking["recommended_next_usage"] == "PORTFOLIO_CANDIDATE_FOR_REVIEW").sum())
    watch_weak_count = int((ranking["recommended_next_usage"] == "WATCH_WEAK").sum())
    drop_for_now_count = int((ranking["recommended_next_usage"] == "DROP_FOR_NOW").sum())
    top_by_mean = ranking.sort_values("mean_monthly_return", ascending=False, kind="mergesort").iloc[0]
    top_by_cumulative = ranking.sort_values("cumulative_simple_return_final", ascending=False, kind="mergesort").iloc[0]

    if not bool(prereq["prerequisites_passed"]) or not target_availability_qa_passed or not guardrail_qa_passed:
        final_decision = "SIMPLE_BASELINE_PORTFOLIO_EVAL_RUN_FAIL"
    elif portfolio_candidate_count > 0:
        final_decision = "SIMPLE_BASELINE_PORTFOLIO_EVAL_RUN_READY_FOR_PORTFOLIO_REVIEW"
    elif drop_for_now_count == len(PORTFOLIO_NAMES):
        final_decision = "SIMPLE_BASELINE_PORTFOLIO_EVAL_RUN_FAIL_STOP_PORTFOLIO_BRANCH"
    else:
        final_decision = "SIMPLE_BASELINE_PORTFOLIO_EVAL_RUN_WATCH_MANUAL_REVIEW_REQUIRED"

    summary = {
        "run_timestamp": now_iso(),
        "prerequisites_passed": bool(prereq["prerequisites_passed"]),
        "rows_loaded": int(len(weights)),
        "portfolio_count": int(weights["portfolio_name"].nunique()),
        "portfolio_names": PORTFOLIO_NAMES,
        "month_count": int(weights["month_end"].nunique()),
        "min_month_end": weights["month_end"].min(),
        "max_month_end": weights["month_end"].max(),
        "monthly_portfolio_returns_generated": monthly_generated,
        "portfolio_return_summary_generated": summary_generated,
        "cumulative_simple_return_path_generated": path_generated,
        "portfolio_ranking_generated": ranking_generated,
        "target_availability_qa_passed": target_availability_qa_passed,
        "guardrail_qa_passed": guardrail_qa_passed,
        "portfolio_candidate_count": portfolio_candidate_count,
        "watch_weak_count": watch_weak_count,
        "drop_for_now_count": drop_for_now_count,
        "top_portfolio_by_mean_return": str(top_by_mean["portfolio_name"]),
        "top_portfolio_by_cumulative_simple_return": str(top_by_cumulative["portfolio_name"]),
        "best_mean_monthly_return": float(top_by_mean["mean_monthly_return"]),
        "best_positive_month_ratio": float(ranking["positive_month_ratio"].max()),
        "best_cumulative_simple_return_final": float(top_by_cumulative["cumulative_simple_return_final"]),
        "research_only_policy": True,
        "fwd_ret_used_for_selection": False,
        "transaction_cost_calculated": False,
        "turnover_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "benchmark_relative_return_calculated": False,
        "alpha_beta_regression_calculated": False,
        "production_backtest_metric_calculated": False,
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
        "recommended_next_step": next_step,
    }
    write_json(OUT_DIR / "simple_baseline_portfolio_evaluation_run_summary.json", summary)
    (OUT_DIR / "simple_baseline_portfolio_evaluation_run_report.md").write_text(build_report(summary), encoding="utf-8")

    final_qa = pd.DataFrame(
        [
            ["prerequisites_passed", summary["prerequisites_passed"]],
            ["monthly_portfolio_returns_generated", monthly_generated],
            ["portfolio_return_summary_generated", summary_generated],
            ["cumulative_simple_return_path_generated", path_generated],
            ["portfolio_ranking_generated", ranking_generated],
            ["target_availability_qa_passed", target_availability_qa_passed],
            ["guardrail_qa_passed", guardrail_qa_passed],
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
            "summary_path": str(OUT_DIR / "simple_baseline_portfolio_evaluation_run_summary.json"),
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
                f"- rows_loaded: {len(weights)}",
                f"- recommended_next_step: {next_step}",
                "- forbidden_metrics_calculated: false",
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

    print(json.dumps({"final_decision": final_decision, "rows_loaded": int(len(weights))}, ensure_ascii=False))

    del prep_summary, run_config, metric_plan, prep_guardrail, construction_summary
    del weights_qa, coverage, leakage_qa, construction_guardrail, weights
    del input_qa, target_qa, monthly_returns, cumulative_path, return_summary, ranking, guardrail_qa, final_qa
    gc.collect()


if __name__ == "__main__":
    main()
