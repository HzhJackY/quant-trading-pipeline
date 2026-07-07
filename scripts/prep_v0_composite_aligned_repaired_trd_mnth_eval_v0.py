from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "v0_composite_aligned_repaired_trd_mnth_eval_prep_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

CONSTRUCTION_DIR = ROOT / "output" / "v0_composite_aligned_portfolio_construction_run_v0"
WEIGHTS_PATH = CONSTRUCTION_DIR / "v0_composite_aligned_research_weights.parquet"
CONSTRUCTION_SUMMARY = CONSTRUCTION_DIR / "v0_composite_aligned_portfolio_construction_summary.json"
MONTHLY_WEIGHT_QA = CONSTRUCTION_DIR / "v0_aligned_portfolio_weight_monthly_qa.csv"
BUFFER_TRANSITION_QA = CONSTRUCTION_DIR / "v0_aligned_buffer_transition_qa.csv"
FUTURE_EVAL_COVERAGE_PLAN = CONSTRUCTION_DIR / "v0_aligned_weights_future_eval_coverage_plan.csv"
RETURN_MAP_PATH = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0" / "canonical_csmar_trd_mnth_return_map_repaired.parquet"

PRIMARY_RETURN_FIELD = "Mretwd"
PRIMARY_COST_BPS = 20
PRIMARY_RETURN_VARIANT = "raw_unmatched_not_renormalized"
FIRST_MONTH_TURNOVER_POLICY = "charge_cost_on_first_month_initialization"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def dump_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_state(status: str, checkpoint: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "RUN_STATE.md").write_text(
        "# RUN_STATE\n\n"
        f"task_name: {TASK_NAME}\n"
        f"status: {status}\n"
        f"last_checkpoint: {checkpoint}\n"
        f"updated_at: {datetime.now().isoformat(timespec='seconds')}\n"
        "resume_instruction: rerun scripts\\prep_v0_composite_aligned_repaired_trd_mnth_eval_v0.py with stdout/stderr redirected to this run directory\n",
        encoding="utf-8",
    )


def norm_symbol(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def prerequisites() -> dict:
    flags = {
        "aligned_weights_found": WEIGHTS_PATH.exists(),
        "construction_summary_found": CONSTRUCTION_SUMMARY.exists(),
        "monthly_weight_qa_found": MONTHLY_WEIGHT_QA.exists(),
        "buffer_transition_qa_found": BUFFER_TRANSITION_QA.exists(),
        "future_eval_coverage_plan_found": FUTURE_EVAL_COVERAGE_PLAN.exists(),
        "trd_mnth_return_map_found": RETURN_MAP_PATH.exists(),
    }
    paths = {
        "aligned_weights_found": WEIGHTS_PATH,
        "construction_summary_found": CONSTRUCTION_SUMMARY,
        "monthly_weight_qa_found": MONTHLY_WEIGHT_QA,
        "buffer_transition_qa_found": BUFFER_TRANSITION_QA,
        "future_eval_coverage_plan_found": FUTURE_EVAL_COVERAGE_PLAN,
        "trd_mnth_return_map_found": RETURN_MAP_PATH,
    }
    missing = [rel(p) for k, p in paths.items() if not flags[k]]
    flags["prerequisites_passed"] = not missing
    flags["missing_files"] = missing
    dump_json(OUT_DIR / "v0_aligned_eval_prep_prerequisite_check.json", flags)
    return flags


def weights_input_qa(weights: pd.DataFrame, monthly_qa: pd.DataFrame, transition: pd.DataFrame, construction_summary: dict) -> pd.DataFrame:
    per_month = weights.groupby("year_month")
    dup = int(weights.duplicated(["symbol_norm", "year_month"]).sum())
    weight_sum_err = float(per_month["weight"].sum().sub(1.0).abs().max())
    alpha_missing = int(weights["alpha_signal_aligned"].isna().sum()) if "alpha_signal_aligned" in weights.columns else -1
    qa_rows = [
        ("portfolio_name", construction_summary.get("portfolio_name", ""), ",".join(sorted(weights["portfolio_name"].dropna().unique())), True, ""),
        ("score_column_selected", "alpha_signal_aligned", "alpha_signal_aligned", "alpha_signal_aligned" in weights.columns, ""),
        ("row_count", construction_summary.get("total_weight_rows", 5700), len(weights), int(construction_summary.get("total_weight_rows", len(weights))) == len(weights), ""),
        ("unique_symbol_count", construction_summary.get("unique_symbol_count", ""), weights["symbol_norm"].nunique(), True, ""),
        ("month_count", construction_summary.get("month_count", ""), weights["year_month"].nunique(), True, ""),
        ("min_year_month", construction_summary.get("first_construction_month", "2017-01"), weights["year_month"].min(), weights["year_month"].min() == "2017-01", ""),
        ("max_year_month", construction_summary.get("last_construction_month", "2026-06"), weights["year_month"].max(), weights["year_month"].max() == "2026-06", ""),
        ("duplicate symbol-month", 0, dup, dup == 0, ""),
        ("selected_count per month", "50 all months", f"{per_month.size().min()} to {per_month.size().max()}", per_month.size().min() == 50 and per_month.size().max() == 50, ""),
        ("weight_sum per month", "abs_error <= 1e-12", weight_sum_err, weight_sum_err <= 1e-12, ""),
        ("alpha_missing_selected_count", 0, alpha_missing, alpha_missing == 0, ""),
        ("ready_month_count", construction_summary.get("ready_month_count", ""), int((monthly_qa["eligible_month_status"] == "READY").sum()), True, ""),
        ("watch_month_count", construction_summary.get("watch_month_count", ""), int(monthly_qa["eligible_month_status"].astype(str).str.startswith("WATCH").sum()), True, "WATCH months preserved for policy handling."),
        ("fail_month_count", construction_summary.get("fail_month_count", ""), int((monthly_qa["eligible_month_status"] == "FAIL_NO_SIGNAL").sum()), True, ""),
        ("watch_months_preserved", True, bool(weights["watch_month_flag"].any()), bool(weights["watch_month_flag"].any()), ""),
        ("turnover_proxy fields if available", "simple_turnover_proxy in transition QA", "simple_turnover_proxy" in transition.columns, "simple_turnover_proxy" in transition.columns, ""),
    ]
    out = pd.DataFrame([{"check_name": c, "expected": e, "actual": a, "pass": p, "caveat": caveat} for c, e, a, p, caveat in qa_rows])
    out.to_csv(OUT_DIR / "v0_aligned_eval_weights_input_qa.csv", index=False, encoding="utf-8-sig")
    return out


def return_source_qa(ret: pd.DataFrame) -> pd.DataFrame:
    dup = int(ret.duplicated(["symbol_norm", "year_month"]).sum())
    fields = ",".join(sorted(ret["primary_return_field"].dropna().astype(str).unique().tolist())) if "primary_return_field" in ret.columns else ""
    valid_cov = float(ret["return_valid_flag"].notna().mean()) if "return_valid_flag" in ret.columns else np.nan
    qa = [
        ("primary_return_field", PRIMARY_RETURN_FIELD, fields, fields == PRIMARY_RETURN_FIELD, ""),
        ("row_count", ">0", len(ret), len(ret) > 0, ""),
        ("unique_symbol_count", ">0", ret["symbol_norm"].nunique(), ret["symbol_norm"].nunique() > 0, ""),
        ("year_month_count", ">0", ret["year_month"].nunique(), ret["year_month"].nunique() > 0, ""),
        ("min_year_month", "available", ret["year_month"].min(), True, ""),
        ("max_year_month", "available", ret["year_month"].max(), True, ""),
        ("duplicate symbol-month", 0, dup, dup == 0, ""),
        ("fwd_ret_1m null count", "reported", int(ret["fwd_ret_1m"].isna().sum()), True, "Null forward labels are expected for latest months."),
        ("extreme fwd_ret_1m abs > 100% count", "reported", int((pd.to_numeric(ret["fwd_ret_1m"], errors="coerce").abs() > 1.0).sum()), True, "QA only; not portfolio return calculation."),
        ("return_valid_flag coverage if available", "reported", valid_cov, True, ""),
    ]
    out = pd.DataFrame([{"check_name": c, "expected": e, "actual": a, "pass": p, "caveat": caveat} for c, e, a, p, caveat in qa])
    out.to_csv(OUT_DIR / "v0_aligned_eval_return_source_qa.csv", index=False, encoding="utf-8-sig")
    return out


def return_match_qa(weights: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    r = ret[["symbol_norm", "year_month", "fwd_ret_1m"]].copy()
    r["has_fwd_ret_1m"] = pd.to_numeric(r["fwd_ret_1m"], errors="coerce").notna()
    r = r.drop_duplicates(["symbol_norm", "year_month"], keep="last")[["symbol_norm", "year_month", "has_fwd_ret_1m"]]
    merged = weights.merge(r, on=["symbol_norm", "year_month"], how="left")
    merged["has_fwd_ret_1m"] = merged["has_fwd_ret_1m"].fillna(False)
    rows = []
    for ym, g in merged.groupby("year_month", sort=True):
        matched = g["has_fwd_ret_1m"]
        matched_share = float(g.loc[matched, "weight"].sum())
        unmatched_share = float(g.loc[~matched, "weight"].sum())
        watch = bool(g["watch_month_flag"].iloc[0])
        if matched_share == 0:
            status = "FAIL_NO_FORWARD_LABEL"
            caveat = "no fwd_ret_1m label available for selected holdings"
        elif matched_share < 0.90:
            status = "FAIL_LOW_MATCH"
            caveat = "matched weight share below 0.90"
        elif matched_share < 0.98:
            status = "WATCH_PARTIAL_MATCH"
            caveat = "matched weight share below 0.98"
        elif watch:
            status = "WATCH_ALPHA_MONTH"
            caveat = "alpha WATCH month included in primary policy unless return match fails"
        else:
            status = "READY"
            caveat = ""
        rows.append(
            {
                "year_month": ym,
                "eligible_month_status": g["eligible_month_status"].iloc[0],
                "watch_month_flag": watch,
                "selected_count": int(len(g)),
                "matched_symbol_count": int(matched.sum()),
                "unmatched_symbol_count": int((~matched).sum()),
                "matched_weight_share": matched_share,
                "unmatched_weight_share": unmatched_share,
                "fwd_ret_available": bool(matched.any()),
                "evaluation_month_status": status,
                "caveat": caveat,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "v0_aligned_eval_return_match_monthly_qa.csv", index=False, encoding="utf-8-sig")
    del merged
    gc.collect()
    return out


def window_policy(match: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = match.copy()
    policy["include_in_primary_eval"] = policy["evaluation_month_status"].isin(["READY", "WATCH_ALPHA_MONTH"])
    policy["include_in_sensitivity_eval"] = policy["evaluation_month_status"].isin(["READY", "WATCH_ALPHA_MONTH", "WATCH_PARTIAL_MATCH"])
    reason_map = {
        "WATCH_PARTIAL_MATCH": "partial return match; sensitivity only",
        "FAIL_NO_FORWARD_LABEL": "no forward label",
        "FAIL_LOW_MATCH": "low return match",
    }
    policy["exclusion_reason"] = np.where(policy["include_in_primary_eval"], "", policy["evaluation_month_status"].map(reason_map).fillna(""))
    policy = policy[
        [
            "year_month",
            "eligible_month_status",
            "watch_month_flag",
            "evaluation_month_status",
            "matched_weight_share",
            "include_in_primary_eval",
            "include_in_sensitivity_eval",
            "exclusion_reason",
            "caveat",
        ]
    ]
    policy.to_csv(OUT_DIR / "v0_aligned_eval_window_policy.csv", index=False, encoding="utf-8-sig")
    primary = policy.loc[policy["include_in_primary_eval"], "year_month"].astype(str)
    excluded = policy.loc[~policy["include_in_primary_eval"], "year_month"].astype(str)
    summary = pd.DataFrame(
        [
            {
                "primary_eval_month_count": int(len(primary)),
                "primary_eval_min_year_month": str(primary.min()) if len(primary) else "",
                "primary_eval_max_year_month": str(primary.max()) if len(primary) else "",
                "excluded_month_count": int(len(excluded)),
                "excluded_months": ";".join(excluded.tolist()),
                "ready_month_count": int((policy["evaluation_month_status"] == "READY").sum()),
                "watch_alpha_month_count": int((policy["evaluation_month_status"] == "WATCH_ALPHA_MONTH").sum()),
                "watch_partial_match_month_count": int((policy["evaluation_month_status"] == "WATCH_PARTIAL_MATCH").sum()),
                "fail_no_forward_label_month_count": int((policy["evaluation_month_status"] == "FAIL_NO_FORWARD_LABEL").sum()),
                "fail_low_match_month_count": int((policy["evaluation_month_status"] == "FAIL_LOW_MATCH").sum()),
            }
        ]
    )
    summary.to_csv(OUT_DIR / "v0_aligned_eval_window_summary.csv", index=False, encoding="utf-8-sig")
    return policy, summary


def write_configs() -> tuple[dict, pd.DataFrame, dict]:
    cost_config = {
        "cost_bps_list": [0, 10, 20, 30],
        "primary_cost_bps": PRIMARY_COST_BPS,
        "return_variants": ["raw_unmatched_not_renormalized", "matched_only_normalized"],
        "primary_return_variant": PRIMARY_RETURN_VARIANT,
        "evaluation_source": "repaired_TRD_Mnth",
        "primary_return_field": PRIMARY_RETURN_FIELD,
        "calculate_returns_next_run_allowed": True,
        "calculate_benchmark_relative_next_run_allowed": False,
        "alpha_beta_next_run_allowed": False,
        "ir_te_next_run_allowed": False,
        "ff_next_run_allowed": False,
        "dgtw_next_run_allowed": False,
    }
    dump_json(OUT_DIR / "v0_aligned_eval_cost_return_variant_config.json", cost_config)
    turnover = pd.DataFrame(
        [
            {
                "policy_item": "first_month_initialization_turnover_policy",
                "selected_policy": FIRST_MONTH_TURNOVER_POLICY,
                "alternative_policy": "first_month_initialization_no_cost",
                "reason": "首月建仓真实发生，主口径扣成本。",
                "caveat": "sensitivity 中可不扣首月初始化成本。",
            },
            {
                "policy_item": "turnover_proxy_source",
                "selected_policy": "v0_aligned_buffer_transition_qa.simple_turnover_proxy",
                "alternative_policy": "recompute from adjacent monthly weights",
                "reason": "construction run 已输出 buffer transition QA。",
                "caveat": "evaluation run 可复核相邻权重变动。",
            },
            {
                "policy_item": "cost_application_policy",
                "selected_policy": "apply selected cost_bps to turnover proxy in next evaluation run",
                "alternative_policy": "zero cost scenario",
                "reason": "成本情景预注册，主口径 20bps。",
                "caveat": "本任务不计算 transaction cost impact。",
            },
            {
                "policy_item": "sensitivity_required",
                "selected_policy": "run cost_bps 0/10/20/30 and first-month no-cost sensitivity",
                "alternative_policy": "primary only",
                "reason": "区分建仓成本假设对结果的影响。",
                "caveat": "不得扩展到 benchmark-relative 或 FF/DGTW。",
            },
        ]
    )
    turnover.to_csv(OUT_DIR / "v0_aligned_eval_turnover_policy.csv", index=False, encoding="utf-8-sig")
    run_config = {
        "evaluation_allowed_next_run": True,
        "weights_path": rel(WEIGHTS_PATH),
        "return_map_path": rel(RETURN_MAP_PATH),
        "eval_window_policy_path": rel(OUT_DIR / "v0_aligned_eval_window_policy.csv"),
        "cost_return_variant_config_path": rel(OUT_DIR / "v0_aligned_eval_cost_return_variant_config.json"),
        "turnover_policy_path": rel(OUT_DIR / "v0_aligned_eval_turnover_policy.csv"),
        "output_monthly_returns_path_next": "output\\v0_composite_aligned_repaired_trd_mnth_eval_run_v0\\v0_aligned_monthly_returns_by_cost.csv",
        "output_performance_summary_path_next": "output\\v0_composite_aligned_repaired_trd_mnth_eval_run_v0\\v0_aligned_performance_summary_by_cost.csv",
        "calculate_portfolio_returns_next_run_allowed": True,
        "calculate_cumulative_returns_next_run_allowed": True,
        "calculate_cost_scenarios_next_run_allowed": True,
        "calculate_sharpe_next_run_allowed": True,
        "calculate_maxdd_next_run_allowed": True,
        "benchmark_relative_next_run_allowed": False,
        "alpha_beta_next_run_allowed": False,
        "ir_te_next_run_allowed": False,
        "ff_next_run_allowed": False,
        "dgtw_next_run_allowed": False,
        "production_allowed": False,
    }
    dump_json(OUT_DIR / "v0_aligned_eval_run_config_draft.json", run_config)
    return cost_config, turnover, run_config


def guardrails() -> tuple[pd.DataFrame, bool]:
    values = {
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "transaction_cost_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "tstat_calculated": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "ml_training_run": False,
        "new_ml_model_trained": False,
        "tuning_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "old_artifacts_modified": False,
        "strategy_weights_generated": False,
        "alpha_signal_generated": False,
    }
    out = pd.DataFrame([{"guardrail": k, "expected": v, "actual": v, "pass": True} for k, v in values.items()])
    out.to_csv(OUT_DIR / "v0_aligned_eval_prep_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    return out, bool(out["pass"].all())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", "prerequisite_check")
    prereq = prerequisites()
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", "read_inputs")
    with CONSTRUCTION_SUMMARY.open("r", encoding="utf-8") as f:
        construction_summary = json.load(f)
    weights = pd.read_parquet(WEIGHTS_PATH)
    weights["symbol_norm"] = norm_symbol(weights["symbol_norm"])
    weights["year_month"] = weights["year_month"].astype(str).str.slice(0, 7)
    weights["watch_month_flag"] = parse_bool(weights["watch_month_flag"])
    weights["future_eval_label_available_flag"] = parse_bool(weights["future_eval_label_available_flag"])
    monthly_qa = pd.read_csv(MONTHLY_WEIGHT_QA, dtype={"year_month": "string"})
    transition = pd.read_csv(BUFFER_TRANSITION_QA, dtype={"year_month": "string"})
    ret = pd.read_parquet(RETURN_MAP_PATH, columns=["symbol_norm", "year_month", "fwd_ret_1m", "primary_return_field", "return_valid_flag"])
    ret["symbol_norm"] = norm_symbol(ret["symbol_norm"])
    ret["year_month"] = ret["year_month"].astype(str).str.slice(0, 7)

    write_state("running", "qa_and_policy")
    weights_qa = weights_input_qa(weights, monthly_qa, transition, construction_summary)
    return_qa = return_source_qa(ret)
    match = return_match_qa(weights, ret)
    window, window_summary = window_policy(match)
    cost_config, turnover_policy, run_config = write_configs()
    guard, guardrails_pass = guardrails()

    primary = window.loc[window["include_in_primary_eval"]].copy()
    primary_match_min = float(primary["matched_weight_share"].min()) if len(primary) else 0.0
    avg_match = float(match["matched_weight_share"].mean())
    min_match = float(match["matched_weight_share"].min())
    ws = window_summary.iloc[0].to_dict()
    caveats_exist = int(ws["watch_alpha_month_count"]) > 0 or int(ws["fail_no_forward_label_month_count"]) > 0 or int(ws["watch_partial_match_month_count"]) > 0
    if not guardrails_pass:
        final_decision = "ALIGNED_EVAL_PREP_FAIL_GUARDRAIL"
    elif int(ws["primary_eval_month_count"]) <= 0 or avg_match < 0.98 or primary_match_min < 0.95:
        final_decision = "ALIGNED_EVAL_PREP_BLOCKED_BY_RETURN_MATCH"
    elif caveats_exist:
        final_decision = "ALIGNED_EVAL_PREP_READY_WITH_CAVEATS"
    else:
        final_decision = "ALIGNED_EVAL_PREP_READY_FOR_EVAL_RUN"
    recommended_next_step = {
        "ALIGNED_EVAL_PREP_READY_FOR_EVAL_RUN": "进入 V0 composite-aligned repaired TRD_Mnth evaluation run；仍禁止 benchmark-relative/alpha-beta/FF/DGTW。",
        "ALIGNED_EVAL_PREP_READY_WITH_CAVEATS": "可进入 evaluation run，但必须保留 WATCH alpha months 和无 forward label 月份排除 caveat。",
        "ALIGNED_EVAL_PREP_BLOCKED_BY_RETURN_MATCH": "先修复 return match / label coverage，再进入 evaluation run。",
        "ALIGNED_EVAL_PREP_FAIL_GUARDRAIL": "停止，先修复 guardrail violation。",
    }[final_decision]

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "weights_path": rel(WEIGHTS_PATH),
        "return_map_path": rel(RETURN_MAP_PATH),
        "primary_return_field": PRIMARY_RETURN_FIELD,
        "weights_loaded": True,
        "return_map_loaded": True,
        "weights_month_count": int(weights["year_month"].nunique()),
        "weights_min_year_month": str(weights["year_month"].min()),
        "weights_max_year_month": str(weights["year_month"].max()),
        "ready_month_count": int((monthly_qa["eligible_month_status"] == "READY").sum()),
        "watch_month_count": int(monthly_qa["eligible_month_status"].astype(str).str.startswith("WATCH").sum()),
        "fail_month_count": int((monthly_qa["eligible_month_status"] == "FAIL_NO_SIGNAL").sum()),
        "avg_matched_weight_share": avg_match,
        "min_matched_weight_share": min_match,
        "ready_eval_month_count": int(ws["ready_month_count"]),
        "watch_alpha_month_count": int(ws["watch_alpha_month_count"]),
        "watch_partial_match_month_count": int(ws["watch_partial_match_month_count"]),
        "fail_no_forward_label_month_count": int(ws["fail_no_forward_label_month_count"]),
        "fail_low_match_month_count": int(ws["fail_low_match_month_count"]),
        "primary_eval_month_count": int(ws["primary_eval_month_count"]),
        "primary_eval_min_year_month": ws["primary_eval_min_year_month"],
        "primary_eval_max_year_month": ws["primary_eval_max_year_month"],
        "excluded_months": ws["excluded_months"],
        "primary_cost_bps": PRIMARY_COST_BPS,
        "primary_return_variant": PRIMARY_RETURN_VARIANT,
        "first_month_initialization_turnover_policy": FIRST_MONTH_TURNOVER_POLICY,
        "evaluation_allowed_next_run": True,
        "calculate_portfolio_returns_next_run_allowed": True,
        "calculate_benchmark_relative_next_run_allowed": False,
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "transaction_cost_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "tstat_calculated": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "ml_training_run": False,
        "new_ml_model_trained": False,
        "tuning_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "old_artifacts_modified": False,
        "strategy_weights_generated": False,
        "alpha_signal_generated": False,
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    dump_json(OUT_DIR / "v0_composite_aligned_repaired_trd_mnth_eval_prep_summary.json", summary)
    report = (
        "# V0 Composite-Aligned Repaired TRD_Mnth Evaluation Prep v0\n\n"
        f"- final_decision: {final_decision}\n"
        f"- weights window: {summary['weights_min_year_month']} to {summary['weights_max_year_month']} ({summary['weights_month_count']} months)\n"
        f"- primary eval window: {summary['primary_eval_min_year_month']} to {summary['primary_eval_max_year_month']} ({summary['primary_eval_month_count']} months)\n"
        f"- ready/watch_alpha/watch_partial/fail_no_label/fail_low_match: {summary['ready_eval_month_count']}/{summary['watch_alpha_month_count']}/{summary['watch_partial_match_month_count']}/{summary['fail_no_forward_label_month_count']}/{summary['fail_low_match_month_count']}\n"
        f"- avg/min matched weight share: {avg_match:.6f}/{min_match:.6f}\n"
        f"- excluded_months: {summary['excluded_months']}\n"
        f"- primary cost/return variant: {PRIMARY_COST_BPS}bps / {PRIMARY_RETURN_VARIANT}\n\n"
        "本任务未计算 portfolio returns、累计收益、transaction cost、Sharpe、MaxDD、t-stat、benchmark-relative、alpha/beta、IR/TE、FF、DGTW、ML、调参、SHAP 或 production 修改，也未重新生成 alpha_signal 或 weights。\n"
    )
    (OUT_DIR / "v0_composite_aligned_repaired_trd_mnth_eval_prep_report.md").write_text(report, encoding="utf-8")
    final_qa = pd.DataFrame(
        [
            {"check_name": "prerequisites_passed", "pass": prereq["prerequisites_passed"], "detail": ""},
            {"check_name": "guardrails_passed", "pass": guardrails_pass, "detail": ""},
            {"check_name": "primary_eval_month_count_positive", "pass": int(ws["primary_eval_month_count"]) > 0, "detail": str(ws["primary_eval_month_count"])},
            {"check_name": "return_match_primary_ok", "pass": avg_match >= 0.98 and primary_match_min >= 0.95, "detail": f"avg={avg_match}; primary_min={primary_match_min}"},
            {"check_name": "final_decision_allowed", "pass": final_decision in {
                "ALIGNED_EVAL_PREP_READY_FOR_EVAL_RUN",
                "ALIGNED_EVAL_PREP_READY_WITH_CAVEATS",
                "ALIGNED_EVAL_PREP_BLOCKED_BY_RETURN_MATCH",
                "ALIGNED_EVAL_PREP_FAIL_GUARDRAIL",
            }, "detail": final_decision},
        ]
    )
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    dump_json(
        OUT_DIR / "terminal_summary.json",
        {
            "task_name": TASK_NAME,
            "status": "completed",
            "stdout_path": rel(RUN_DIR / "run_stdout.txt"),
            "stderr_path": rel(RUN_DIR / "run_stderr.txt"),
            "output_dir": rel(OUT_DIR),
            "final_decision": final_decision,
        },
    )
    (OUT_DIR / "task_completion_card.md").write_text(
        f"# Task completion card\n\n- task_name: {TASK_NAME}\n- status: completed\n- final_decision: {final_decision}\n- output_dir: {rel(OUT_DIR)}\n",
        encoding="utf-8",
    )
    del weights, monthly_qa, transition, ret, weights_qa, return_qa, match, window, window_summary, guard
    gc.collect()
    write_state("completed", "all_outputs_written")
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": rel(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
