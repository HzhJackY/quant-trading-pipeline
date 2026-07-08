from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


TASK_NAME = "Unified Robust Portfolio Evaluation Prep v0"
OUT_DIR = Path("output/unified_robust_portfolio_evaluation_prep_v0")
RUN_DIR = Path("output/_agent_runs") / TASK_NAME

FLAG_DIR = Path("output/flag_based_top50_buffer_portfolio_construction_run_v0")
LOW_HOLDING_DIR = Path("output/flag_based_portfolio_low_holding_count_qa_review_v0")
FALLBACK_DIR = Path("output/robust_formation_portfolio_construction_run_v0")

FLAG_WEIGHTS = FLAG_DIR / "flag_based_top50_buffer_research_weights_v0.parquet"
FALLBACK_WEIGHTS = FALLBACK_DIR / "robust_formation_research_weights_v0.parquet"

PORTFOLIOS = [
    {
        "portfolio_name": "ROBUST_VQ_TOP20_EXCLUDE_SOFT_ANOMALY_EQUAL_WEIGHT",
        "portfolio_role": "Flag-based primary",
        "source_branch": "flag_based",
    },
    {
        "portfolio_name": "ROBUST_VQ_FLAG_CLEAN_TOP50_EQUAL_WEIGHT",
        "portfolio_role": "Flag-based top50-style",
        "source_branch": "flag_based",
    },
    {
        "portfolio_name": "ROBUST_VQ_FLAG_CLEAN_TOP50_BUFFER_EQUAL_WEIGHT",
        "portfolio_role": "Flag-based buffer-style",
        "source_branch": "flag_based",
    },
    {
        "portfolio_name": "ROBUST_VQ_D7_D9_BAND_EQUAL_WEIGHT",
        "portfolio_role": "Fallback diagnostic",
        "source_branch": "fallback",
    },
    {
        "portfolio_name": "ROBUST_VQ_TOP30_PERCENT_EQUAL_WEIGHT",
        "portfolio_role": "Fallback diagnostic",
        "source_branch": "fallback",
    },
]

REQUIRED_INPUTS = [
    FLAG_DIR / "flag_based_top50_buffer_portfolio_construction_summary.json",
    FLAG_WEIGHTS,
    FLAG_DIR / "flag_based_top50_buffer_holding_count_qa.csv",
    FLAG_DIR / "flag_based_top50_buffer_weight_sum_qa.csv",
    FLAG_DIR / "flag_based_top50_buffer_anomaly_exposure_qa.csv",
    FLAG_DIR / "flag_based_top50_buffer_industry_exposure_qa.csv",
    FLAG_DIR / "flag_based_top50_buffer_component_profile_qa.csv",
    FLAG_DIR / "flag_based_top50_buffer_transition_qa.csv",
    FLAG_DIR / "flag_based_top50_buffer_guardrail_qa.csv",
    LOW_HOLDING_DIR / "flag_based_portfolio_low_holding_count_qa_review_summary.json",
    LOW_HOLDING_DIR / "low_holding_portfolio_month_detail.csv",
    LOW_HOLDING_DIR / "low_holding_evaluation_policy_recommendation.csv",
    FALLBACK_DIR / "robust_formation_portfolio_construction_summary.json",
    FALLBACK_WEIGHTS,
    FALLBACK_DIR / "robust_formation_portfolio_weight_sum_qa.csv",
    FALLBACK_DIR / "robust_formation_portfolio_industry_exposure_qa.csv",
    FALLBACK_DIR / "robust_formation_portfolio_component_profile_qa.csv",
    FALLBACK_DIR / "robust_formation_portfolio_guardrail_qa.csv",
]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def truthy_guardrail(summary: dict, guardrail_path: Path) -> bool:
    for key in ("guardrail_pass", "guardrail_qa_pass", "all_guardrails_passed"):
        if key in summary:
            return bool(summary[key])
    if guardrail_path.exists():
        qa = pd.read_csv(guardrail_path)
        bool_cols = [c for c in qa.columns if "pass" in c.lower()]
        if bool_cols:
            out = bool(qa[bool_cols[0]].astype(str).str.lower().isin(["true", "1", "pass"]).all())
            del qa
            gc.collect()
            return out
        del qa
        gc.collect()
    return True


def load_weight_slice(path: Path, wanted: list[str]) -> pd.DataFrame:
    cols = ["portfolio_name", "symbol", "month_end", "weight", "fwd_ret_1m"]
    df = pd.read_parquet(path, columns=cols)
    df = df[df["portfolio_name"].isin(wanted)].copy()
    df["symbol"] = df["symbol"].astype("string")
    return df


def weight_source_manifest(flag_df: pd.DataFrame, fallback_df: pd.DataFrame, flag_guardrail: bool, fallback_guardrail: bool) -> pd.DataFrame:
    rows = []
    for item in PORTFOLIOS:
        source = item["source_branch"]
        df = flag_df if source == "flag_based" else fallback_df
        path = FLAG_WEIGHTS if source == "flag_based" else FALLBACK_WEIGHTS
        sub = df[df["portfolio_name"] == item["portfolio_name"]]
        rows.append(
            {
                "source_branch": source,
                "weights_path": str(path),
                "portfolio_name": item["portfolio_name"],
                "portfolio_role": item["portfolio_role"],
                "row_count": int(len(sub)),
                "month_count": int(sub["month_end"].nunique(dropna=True)),
                "symbol_count": int(sub["symbol"].nunique(dropna=True)),
                "guardrail_pass": bool(flag_guardrail if source == "flag_based" else fallback_guardrail),
                "notes": "research-only weights; next run must not reconstruct or alter weights",
            }
        )
    return pd.DataFrame(rows)


def sample_policy() -> dict:
    return {
        "policy_language": "zh-CN",
        "current_problem": [
            "flag-based weights 已生成",
            "fallback weights 已生成",
            "flag-based top20 组合存在 3 个 low-holding months，但 QA 通过",
            "现在需要统一评估 5 条 research-only portfolios",
            "不能单独只看表现最好的分支",
        ],
        "solution": [
            "建立统一 evaluation policy",
            "下一阶段统一计算 portfolio monthly return",
            "下一阶段统一计算 turnover / cost / Sharpe / MaxDD / benchmark-relative / alpha-beta",
            "对 low-holding months 做 sensitivity",
            "对 flag-based vs fallback 做公平比较",
        ],
        "must_use_existing_weights": True,
        "must_not_reconstruct_weights": True,
        "row_requirements": {
            "portfolio_name_non_null": True,
            "symbol_non_null": True,
            "month_end_non_null": True,
            "weight_non_null": True,
            "fwd_ret_1m_non_null": True,
            "weight_positive": True,
            "no_duplicate_symbol_within_portfolio_month": True,
        },
        "missing_target_policy": "记录 missing target；不得用其他收益替代",
        "target_column": "fwd_ret_1m",
    }


def metric_plan() -> pd.DataFrame:
    allowed_metrics = [
        ("portfolio_monthly_return", True, "primary", "按既有 weights 和 fwd_ret_1m 计算；不得改变持仓"),
        ("cumulative_return", True, "primary", "仅 research evaluation；不得称为 production backtest"),
        ("turnover", True, "primary", "需定义单边或双边；buffer portfolio 单独标记"),
        ("transaction_cost", True, "diagnostic", "0/10/20/30 bps 情景"),
        ("monthly_mean_return", True, "primary", "risk metric"),
        ("volatility", True, "primary", "monthly volatility"),
        ("sharpe", True, "primary", "monthly return based Sharpe"),
        ("maxdd", True, "primary", "research evaluation drawdown"),
        ("positive_month_ratio", True, "diagnostic", "月度胜率"),
        ("worst_month", True, "diagnostic", "最差月份"),
        ("best_month", True, "diagnostic", "最好月份"),
        ("benchmark_excess_return", False, "diagnostic", "benchmark source missing 时阻塞"),
        ("tracking_difference", False, "diagnostic", "benchmark source missing 时阻塞"),
        ("alpha_beta_regression", False, "diagnostic", "benchmark source missing 时阻塞"),
        ("industry_exposure", True, "diagnostic", "monthly industry weights / concentration"),
        ("low_holding_sensitivity", True, "diagnostic", "base vs exclude low_holding_count_flag"),
    ]
    return pd.DataFrame(
        [
            {
                "metric_name": name,
                "allowed_next_run": allowed,
                "calculated_in_this_task": False,
                "primary_or_diagnostic": role,
                "notes": notes,
            }
            for name, allowed, role, notes in allowed_metrics
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().isoformat(timespec="seconds")
    write_json(
        RUN_DIR / "RUN_STATE.md",
        {
            "task_name": TASK_NAME,
            "status": "running",
            "run_timestamp": run_timestamp,
            "mode": "low-resource checkpoint-first resume-safe",
            "note": "prep only; no portfolio return / turnover / cost / Sharpe / MaxDD calculated",
        },
    )

    missing = [str(p) for p in REQUIRED_INPUTS if not p.exists()]
    flag_summary = read_json(REQUIRED_INPUTS[0]) if not missing else {}
    low_summary = read_json(REQUIRED_INPUTS[9]) if not missing else {}
    fallback_summary = read_json(REQUIRED_INPUTS[12]) if not missing else {}

    flag_guardrail = False if missing else truthy_guardrail(flag_summary, FLAG_DIR / "flag_based_top50_buffer_guardrail_qa.csv")
    fallback_guardrail = False if missing else truthy_guardrail(fallback_summary, FALLBACK_DIR / "robust_formation_portfolio_guardrail_qa.csv")

    flag_names = [p["portfolio_name"] for p in PORTFOLIOS if p["source_branch"] == "flag_based"]
    fallback_names = [p["portfolio_name"] for p in PORTFOLIOS if p["source_branch"] == "fallback"]
    flag_df = load_weight_slice(FLAG_WEIGHTS, flag_names) if not missing else pd.DataFrame()
    fallback_df = load_weight_slice(FALLBACK_WEIGHTS, fallback_names) if not missing else pd.DataFrame()

    manifest = weight_source_manifest(flag_df, fallback_df, flag_guardrail, fallback_guardrail) if not missing else pd.DataFrame()
    all_portfolios_present = (not manifest.empty) and bool((manifest["row_count"] > 0).all()) and len(manifest) == 5
    flag_ready = not missing and bool(manifest[manifest["source_branch"] == "flag_based"]["row_count"].gt(0).all()) and flag_guardrail
    fallback_ready = not missing and bool(manifest[manifest["source_branch"] == "fallback"]["row_count"].gt(0).all()) and fallback_guardrail

    benchmark_source_available = False
    prerequisites_passed = bool(not missing and flag_ready and fallback_ready and all_portfolios_present)
    benchmark_relative_eval_planned = False
    alpha_beta_eval_planned = False

    if not prerequisites_passed:
        final_decision = "UNIFIED_PORTFOLIO_EVAL_PREP_FAIL"
    elif not benchmark_source_available:
        final_decision = "UNIFIED_PORTFOLIO_EVAL_PREP_WATCH_BENCHMARK_SOURCE_MISSING"
    else:
        final_decision = "UNIFIED_PORTFOLIO_EVAL_PREP_READY_FOR_RUN"

    prereq = {
        "run_timestamp": run_timestamp,
        "required_inputs_checked": [str(p) for p in REQUIRED_INPUTS],
        "missing_inputs": missing,
        "flag_based_weights_ready": flag_ready,
        "fallback_weights_ready": fallback_ready,
        "flag_based_guardrail_pass": flag_guardrail,
        "fallback_guardrail_pass": fallback_guardrail,
        "all_5_portfolios_present": all_portfolios_present,
        "prerequisites_passed": prerequisites_passed,
    }
    write_json(OUT_DIR / "unified_portfolio_eval_prep_prerequisite_check.json", prereq)

    manifest.to_csv(OUT_DIR / "unified_portfolio_weight_source_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(PORTFOLIOS).to_csv(OUT_DIR / "unified_portfolio_taxonomy.csv", index=False, encoding="utf-8-sig")
    metric_plan().to_csv(OUT_DIR / "unified_portfolio_evaluation_metric_plan.csv", index=False, encoding="utf-8-sig")
    write_json(OUT_DIR / "unified_portfolio_sample_policy.json", sample_policy())

    sensitivity_policy = {
        "low_holding_sensitivity": {
            "required": True,
            "base": "include all portfolio-months",
            "sensitivity": "exclude low_holding_count_flag = true",
            "affected_portfolios": ["ROBUST_VQ_TOP20_EXCLUDE_SOFT_ANOMALY_EQUAL_WEIGHT"],
            "low_holding_months": low_summary.get("low_holding_months", ["2017-06-30", "2017-07-31", "2018-07-31"]),
        },
        "cost_sensitivity_bps": [0, 10, 20, 30],
        "turnover_sensitivity": "buffer vs non-buffer comparison",
        "industry_exposure_sensitivity": {
            "dominant_industry_share_threshold": 0.20,
            "threshold_is_diagnostic_not_deletion_rule": True,
        },
        "flag_anomaly_exposure": {
            "flag_based_composite_anomaly_weight_share_expected": 0.0,
            "fallback_anomaly_exposure_used_as_diagnostic_comparison": True,
        },
    }
    write_json(OUT_DIR / "unified_portfolio_sensitivity_policy.json", sensitivity_policy)
    pd.DataFrame(
        [
            {"cost_scenario_bps": bps, "allowed_next_run": True, "calculated_in_this_task": False, "notes": "apply after turnover definition in evaluation run"}
            for bps in [0, 10, 20, 30]
        ]
    ).to_csv(OUT_DIR / "unified_portfolio_cost_scenario_policy.csv", index=False, encoding="utf-8-sig")

    benchmark_policy = {
        "benchmark_source_required": True,
        "benchmark_source_available": benchmark_source_available,
        "benchmark_relative_eval_allowed_next_run": benchmark_source_available,
        "alpha_beta_eval_allowed_next_run": benchmark_source_available,
        "benchmark_relative_eval_blocked_by_missing_benchmark_source": not benchmark_source_available,
        "action_if_missing": "不得伪造 benchmark；仅执行 non-benchmark portfolio evaluation，并在报告中标记 benchmark-relative / alpha-beta blocked",
    }
    write_json(OUT_DIR / "unified_portfolio_benchmark_policy.json", benchmark_policy)
    write_json(
        OUT_DIR / "unified_portfolio_turnover_policy.json",
        {
            "turnover_allowed_next_run": True,
            "calculated_in_this_task": False,
            "definition_required_next_run": True,
            "recommended_definition": "monthly one-way turnover = 0.5 * sum(abs(w_t_after_rebalance - drifted_w_t_minus_1))) if return data supports drift; otherwise report explicit no-drift approximation",
            "buffer_portfolio_must_be_flagged": True,
            "buffer_portfolio_name": "ROBUST_VQ_FLAG_CLEAN_TOP50_BUFFER_EQUAL_WEIGHT",
        },
    )
    write_json(
        OUT_DIR / "unified_portfolio_industry_exposure_policy.json",
        {
            "industry_exposure_allowed_next_run": True,
            "calculated_in_this_task": False,
            "diagnostics": ["monthly industry weights", "dominant industry share", "average industry count", "industry concentration", "flag-based vs fallback exposure comparison"],
            "dominant_industry_share_threshold": 0.20,
            "threshold_is_diagnostic_not_deletion_rule": True,
        },
    )

    run_config = {
        "flag_based_weights_path": str(FLAG_WEIGHTS),
        "fallback_weights_path": str(FALLBACK_WEIGHTS),
        "output_directory_for_next_run": "output\\unified_robust_portfolio_evaluation_run_v0\\",
        "portfolios_to_evaluate": [p["portfolio_name"] for p in PORTFOLIOS],
        "target_column": "fwd_ret_1m",
        "date_column": "month_end",
        "symbol_column": "symbol",
        "weight_column": "weight",
        "calculate_portfolio_return_next_run_allowed": True,
        "calculate_cumulative_return_next_run_allowed": True,
        "calculate_turnover_next_run_allowed": True,
        "calculate_transaction_cost_next_run_allowed": True,
        "calculate_sharpe_next_run_allowed": True,
        "calculate_maxdd_next_run_allowed": True,
        "calculate_industry_exposure_next_run_allowed": True,
        "calculate_benchmark_relative_next_run_allowed": benchmark_source_available,
        "calculate_alpha_beta_next_run_allowed": benchmark_source_available,
        "backtest_allowed_next_run": False,
        "production_allowed_next_run": False,
    }
    write_json(OUT_DIR / "unified_portfolio_evaluation_run_config_draft.json", run_config)

    guardrails = pd.DataFrame(
        [
            {"guardrail_item": "use_existing_weights_only", "required": True, "status": "planned", "notes": "next run must not reconstruct weights"},
            {"guardrail_item": "no_return_calculated_in_prep", "required": True, "status": "pass", "notes": "prep generated policy/config only"},
            {"guardrail_item": "no_turnover_calculated_in_prep", "required": True, "status": "pass", "notes": "turnover deferred"},
            {"guardrail_item": "no_cost_metric_calculated_in_prep", "required": True, "status": "pass", "notes": "cost scenarios only"},
            {"guardrail_item": "no_sharpe_maxdd_calculated_in_prep", "required": True, "status": "pass", "notes": "risk metrics deferred"},
            {"guardrail_item": "benchmark_not_fabricated", "required": True, "status": "pass", "notes": "benchmark source unavailable; blocked flags set"},
            {"guardrail_item": "production_not_modified", "required": True, "status": "pass", "notes": "research-only prep"},
        ]
    )
    guardrails.to_csv(OUT_DIR / "unified_portfolio_guardrail_checklist.csv", index=False, encoding="utf-8-sig")

    next_plan = """# 下一步：Unified Portfolio Evaluation Run v0

## 目标
基于本 prep 生成的 run config，对 5 条 research-only portfolios 做统一、公平的 evaluation run。

## 允许计算
- portfolio monthly return：使用既有 weight 与 fwd_ret_1m，不得改变持仓。
- cumulative return：仅 research evaluation，不得称为 production backtest。
- turnover：先明确单边或双边定义，并单独标记 buffer portfolio。
- transaction cost：0 / 10 / 20 / 30 bps。
- risk metrics：mean return、volatility、Sharpe、MaxDD、positive month ratio、worst/best month。
- industry exposure diagnostics。
- low-holding sensitivity：base 与 exclude low_holding_count_flag=true。

## 阻塞项
- benchmark source 当前不可得；不得伪造 benchmark。
- benchmark-relative evaluation 与 alpha/beta regression 需等待 benchmark source。

## 禁止
- 不得重构 weights。
- 不得训练、调参、SHAP、写 production 或生成 live-order-ready holdings。
"""
    (OUT_DIR / "next_step_unified_portfolio_evaluation_run_plan.md").write_text(next_plan, encoding="utf-8")

    summary = {
        "run_timestamp": run_timestamp,
        "prerequisites_passed": prerequisites_passed,
        "flag_based_weights_ready": flag_ready,
        "fallback_weights_ready": fallback_ready,
        "portfolio_count_planned_for_evaluation": 5,
        "portfolios_planned_for_evaluation": [p["portfolio_name"] for p in PORTFOLIOS],
        "low_holding_sensitivity_required": True,
        "cost_scenarios_planned": [0, 10, 20, 30],
        "turnover_eval_planned": True,
        "industry_exposure_eval_planned": True,
        "benchmark_source_available": benchmark_source_available,
        "benchmark_relative_eval_planned": benchmark_relative_eval_planned,
        "alpha_beta_eval_planned": alpha_beta_eval_planned,
        "calculate_portfolio_return_next_run_allowed": True,
        "calculate_cumulative_return_next_run_allowed": True,
        "calculate_turnover_next_run_allowed": True,
        "calculate_transaction_cost_next_run_allowed": True,
        "calculate_sharpe_next_run_allowed": True,
        "calculate_maxdd_next_run_allowed": True,
        "backtest_allowed_next_run": False,
        "production_allowed_next_run": False,
        "portfolio_return_calculated": False,
        "cumulative_return_calculated": False,
        "turnover_calculated": False,
        "transaction_cost_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "benchmark_relative_return_calculated": False,
        "alpha_beta_regression_calculated": False,
        "backtest_run": False,
        "training_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "final_decision": final_decision,
        "recommended_next_step": "运行 Unified Portfolio Evaluation Run v0；若需 benchmark-relative / alpha-beta，先提供 benchmark source。",
    }
    write_json(OUT_DIR / "unified_robust_portfolio_evaluation_prep_summary.json", summary)

    report = f"""# Unified Robust Portfolio Evaluation Prep v0

## 当前问题
- flag-based weights 已生成。
- fallback weights 已生成。
- flag-based top20 组合存在 3 个 low-holding months，但 QA 通过。
- 现在需要统一评估 5 条 research-only portfolios。
- 不能单独只看表现最好的分支。

## 解决方案
- 建立统一 evaluation policy。
- 下一阶段统一计算 portfolio monthly return。
- 下一阶段统一计算 turnover / cost / Sharpe / MaxDD / benchmark-relative / alpha-beta。
- 对 low-holding months 做 sensitivity。
- 对 flag-based vs fallback 做公平比较。

## Prep 结论
- final_decision: {final_decision}
- prerequisites_passed: {prerequisites_passed}
- benchmark_source_available: {benchmark_source_available}
- benchmark-relative / alpha-beta: {'allowed' if benchmark_source_available else 'blocked by missing benchmark source'}

## Prep 禁止项确认
本任务未计算 portfolio return、cumulative return、turnover、transaction cost、Sharpe、MaxDD、benchmark-relative return、alpha/beta；未回测、未训练、未计算 SHAP、未写 production。
"""
    (OUT_DIR / "unified_robust_portfolio_evaluation_prep_report.md").write_text(report, encoding="utf-8")

    final_qa = pd.DataFrame(
        [
            {"qa_item": "required_inputs_exist", "pass": not missing, "detail": ";".join(missing) if missing else "all required inputs exist"},
            {"qa_item": "five_portfolios_in_plan", "pass": all_portfolios_present, "detail": "5 portfolios planned"},
            {"qa_item": "flag_based_weights_ready", "pass": flag_ready, "detail": str(flag_ready)},
            {"qa_item": "fallback_weights_ready", "pass": fallback_ready, "detail": str(fallback_ready)},
            {"qa_item": "no_forbidden_metrics_calculated", "pass": True, "detail": "prep only"},
            {"qa_item": "benchmark_source_handled", "pass": True, "detail": "missing source marked as blocked"},
            {"qa_item": "final_decision_valid", "pass": final_decision in {
                "UNIFIED_PORTFOLIO_EVAL_PREP_READY_FOR_RUN",
                "UNIFIED_PORTFOLIO_EVAL_PREP_WATCH_BENCHMARK_SOURCE_MISSING",
                "UNIFIED_PORTFOLIO_EVAL_PREP_WATCH_MANUAL_REVIEW_REQUIRED",
                "UNIFIED_PORTFOLIO_EVAL_PREP_FAIL",
            }, "detail": final_decision},
        ]
    )
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    final_qa.to_csv(RUN_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")

    terminal_summary = {
        "task_name": TASK_NAME,
        "run_timestamp": run_timestamp,
        "script": "scripts/prep_unified_robust_portfolio_evaluation_v0.py",
        "stdout_log": str(RUN_DIR / "run_stdout.txt"),
        "stderr_log": str(RUN_DIR / "run_stderr.txt"),
        "output_directory": str(OUT_DIR),
        "final_decision": final_decision,
        "forbidden_metrics_calculated": False,
    }
    write_json(RUN_DIR / "terminal_summary.json", terminal_summary)
    completion_card = f"""# Task Completion Card

- task_name: {TASK_NAME}
- run_timestamp: {run_timestamp}
- final_decision: {final_decision}
- output_directory: {OUT_DIR}
- logs: {RUN_DIR / 'run_stdout.txt'} / {RUN_DIR / 'run_stderr.txt'}
- forbidden_metrics_calculated: false
- production_modified: false
"""
    (RUN_DIR / "task_completion_card.md").write_text(completion_card, encoding="utf-8")
    write_json(
        RUN_DIR / "RUN_STATE.md",
        {
            "task_name": TASK_NAME,
            "status": "completed",
            "run_timestamp": run_timestamp,
            "output_directory": str(OUT_DIR),
            "final_decision": final_decision,
            "resume_instruction": "如需继续下一阶段，读取本 RUN_STATE.md 与 unified_portfolio_evaluation_run_config_draft.json。",
        },
    )

    del flag_df, fallback_df, manifest, final_qa
    gc.collect()
    print(json.dumps({"final_decision": final_decision, "prerequisites_passed": prerequisites_passed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
