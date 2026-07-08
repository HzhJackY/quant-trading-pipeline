from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_NAME = "v0_canonical_repaired_trd_mnth_eval_prep_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

WEIGHTS = ROOT / "output" / "v0_canonical_portfolio_construction_run_v0" / "v0_canonical_research_weights.parquet"
CONSTRUCTION_SUMMARY = (
    ROOT / "output" / "v0_canonical_portfolio_construction_run_v0" / "v0_canonical_portfolio_construction_summary.json"
)
MONTHLY_WEIGHT_QA = (
    ROOT / "output" / "v0_canonical_portfolio_construction_run_v0" / "v0_canonical_portfolio_weight_monthly_qa.csv"
)
BUFFER_TRANSITION_QA = (
    ROOT / "output" / "v0_canonical_portfolio_construction_run_v0" / "v0_canonical_buffer_transition_qa.csv"
)
RETURN_MAP = (
    ROOT
    / "output"
    / "trd_mnth_parser_repair_2024_12_coverage_repair_v0"
    / "canonical_csmar_trd_mnth_return_map_repaired.parquet"
)

PRIMARY_RETURN_FIELD = "Mretwd"
PRIMARY_COST_BPS = 20
PRIMARY_RETURN_VARIANT = "raw_unmatched_not_renormalized"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_state(status: str, details: dict[str, Any] | None = None) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": TASK_NAME,
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "details": details or {},
        "resume_instruction": f"先读取 {rel(RUN_DIR / 'RUN_STATE.md')} 再继续。",
    }
    lines = ["# RUN_STATE", "", f"- task_name: {TASK_NAME}", f"- status: {status}"]
    for key, value in payload["details"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "```json", json.dumps(payload, ensure_ascii=False, indent=2, default=str), "```"]
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(lines), encoding="utf-8")


def prereq_check() -> dict[str, Any]:
    required = [WEIGHTS, CONSTRUCTION_SUMMARY, MONTHLY_WEIGHT_QA, BUFFER_TRANSITION_QA, RETURN_MAP]
    missing = [rel(p) for p in required if not p.exists()]
    return {
        "weights_found": WEIGHTS.exists(),
        "construction_summary_found": CONSTRUCTION_SUMMARY.exists(),
        "monthly_weight_qa_found": MONTHLY_WEIGHT_QA.exists(),
        "buffer_transition_qa_found": BUFFER_TRANSITION_QA.exists(),
        "trd_mnth_return_map_found": RETURN_MAP.exists(),
        "prerequisites_passed": len(missing) == 0,
        "missing_files": missing,
    }


def load_weights() -> pd.DataFrame:
    cols = ["portfolio_name", "year_month", "symbol_norm", "weight", "alpha_signal", "selected_count"]
    optional = ["selection_reason"]
    df = pd.read_parquet(WEIGHTS)
    keep = [c for c in cols + optional if c in df.columns]
    df = df[keep].copy()
    df["symbol_norm"] = df["symbol_norm"].astype(str).str.zfill(6)
    df["year_month"] = df["year_month"].astype(str).str.slice(0, 7)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["alpha_signal"] = pd.to_numeric(df["alpha_signal"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df


def weights_input_qa(weights: pd.DataFrame) -> pd.DataFrame:
    monthly = weights.groupby("year_month").agg(
        selected_count=("symbol_norm", "count"),
        weight_sum=("weight", "sum"),
        alpha_missing=("alpha_signal", lambda x: int(x.isna().sum())),
        duplicate_symbols=("symbol_norm", lambda x: int(x.duplicated().sum())),
    )
    portfolio_names = sorted(weights["portfolio_name"].dropna().unique().tolist()) if "portfolio_name" in weights else []
    checks = [
        ("portfolio_name", "single V0 canonical portfolio", ",".join(portfolio_names), len(portfolio_names) == 1, ""),
        ("row_count", ">0", len(weights), len(weights) > 0, ""),
        ("unique_symbol_count", ">0", weights["symbol_norm"].nunique(), weights["symbol_norm"].nunique() > 0, ""),
        ("month_count", ">0", weights["year_month"].nunique(), weights["year_month"].nunique() > 0, ""),
        ("min_year_month", "2017-03", str(weights["year_month"].min()), str(weights["year_month"].min()) == "2017-03", ""),
        ("max_year_month", ">=2024-12", str(weights["year_month"].max()), str(weights["year_month"].max()) >= "2024-12", ""),
        ("duplicate symbol-month", "0", int(weights.duplicated(["symbol_norm", "year_month"]).sum()), int(weights.duplicated(["symbol_norm", "year_month"]).sum()) == 0, ""),
        ("selected_count per month", "50", f"{int(monthly['selected_count'].min())}-{int(monthly['selected_count'].max())}", bool((monthly["selected_count"] == 50).all()), ""),
        ("weight_sum per month", "1.0", f"max_abs_error={float((monthly['weight_sum'] - 1).abs().max()):.12g}", bool(((monthly["weight_sum"] - 1).abs() <= 1e-12).all()), ""),
        ("alpha_missing_selected_count", "0", int(monthly["alpha_missing"].sum()), int(monthly["alpha_missing"].sum()) == 0, ""),
        ("turnover_proxy fields if available", "reviewed from buffer transition QA", BUFFER_TRANSITION_QA.exists(), BUFFER_TRANSITION_QA.exists(), ""),
    ]
    return pd.DataFrame([{"check_name": c[0], "expected": c[1], "actual": c[2], "pass": c[3], "caveat": c[4]} for c in checks])


def load_return_map() -> pd.DataFrame:
    cols = ["symbol_norm", "year_month", "fwd_ret_1m", "primary_return_field", "return_valid_flag"]
    df = pd.read_parquet(RETURN_MAP, columns=[c for c in cols if c in pd.read_parquet(RETURN_MAP, columns=[]).columns])
    # Fallback because pandas cannot always expose columns=[] consistently across engines.
    if df.empty and len(df.columns) == 0:
        df = pd.read_parquet(RETURN_MAP, columns=["symbol_norm", "year_month", "fwd_ret_1m", "primary_return_field"])
    df["symbol_norm"] = df["symbol_norm"].astype(str).str.zfill(6)
    df["year_month"] = df["year_month"].astype(str).str.slice(0, 7)
    df["fwd_ret_1m"] = pd.to_numeric(df["fwd_ret_1m"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df


def return_source_qa(ret: pd.DataFrame) -> pd.DataFrame:
    duplicate_count = int(ret.duplicated(["symbol_norm", "year_month"]).sum())
    primary_fields = sorted(ret["primary_return_field"].dropna().astype(str).unique().tolist()) if "primary_return_field" in ret else []
    valid_cov = float(ret["return_valid_flag"].notna().mean()) if "return_valid_flag" in ret else np.nan
    checks = [
        ("primary_return_field", PRIMARY_RETURN_FIELD, ",".join(primary_fields), PRIMARY_RETURN_FIELD in primary_fields, ""),
        ("row_count", ">0", len(ret), len(ret) > 0, ""),
        ("unique_symbol_count", ">0", ret["symbol_norm"].nunique(), ret["symbol_norm"].nunique() > 0, ""),
        ("year_month_count", ">0", ret["year_month"].nunique(), ret["year_month"].nunique() > 0, ""),
        ("min_year_month", "available", str(ret["year_month"].min()), True, ""),
        ("max_year_month", "available", str(ret["year_month"].max()), True, ""),
        ("duplicate symbol-month", "0", duplicate_count, duplicate_count == 0, "duplicates are not expected in canonical return map"),
        ("fwd_ret_1m null count", "review", int(ret["fwd_ret_1m"].isna().sum()), True, "latest month may lack forward label"),
        ("extreme fwd_ret_1m abs > 100% count", "review", int((ret["fwd_ret_1m"].abs() > 1).sum()), True, "QA only; no return calculation"),
        ("return_valid_flag coverage if available", "review", "" if np.isnan(valid_cov) else round(valid_cov, 6), True, ""),
    ]
    return pd.DataFrame([{"check_name": c[0], "expected": c[1], "actual": c[2], "pass": c[3], "caveat": c[4]} for c in checks])


def match_monthly_qa(weights: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    ret_dedup = ret.drop_duplicates(["symbol_norm", "year_month"], keep="last")
    merged = weights.merge(ret_dedup[["symbol_norm", "year_month", "fwd_ret_1m"]], on=["symbol_norm", "year_month"], how="left")
    rows = []
    for ym, grp in merged.groupby("year_month", sort=True):
        matched = grp["fwd_ret_1m"].notna()
        matched_weight = float(grp.loc[matched, "weight"].sum())
        unmatched_weight = float(grp.loc[~matched, "weight"].sum())
        matched_count = int(matched.sum())
        selected_count = int(len(grp))
        if matched_weight == 0:
            status = "FAIL_NO_FORWARD_LABEL"
            caveat = "no fwd_ret_1m for this weight month; expected for latest month if forward label unavailable"
        elif matched_weight >= 0.98:
            status = "READY"
            caveat = ""
        elif matched_weight >= 0.90:
            status = "WATCH_PARTIAL_MATCH"
            caveat = "partial match; sensitivity only unless explicitly approved"
        else:
            status = "FAIL_LOW_MATCH"
            caveat = "matched weight share below 0.90"
        rows.append(
            {
                "year_month": ym,
                "selected_count": selected_count,
                "matched_symbol_count": matched_count,
                "unmatched_symbol_count": selected_count - matched_count,
                "matched_weight_share": round(matched_weight, 6),
                "unmatched_weight_share": round(unmatched_weight, 6),
                "fwd_ret_available": bool(matched_count > 0),
                "evaluation_month_status": status,
                "caveat": caveat,
            }
        )
    del merged, ret_dedup
    gc.collect()
    return pd.DataFrame(rows)


def eval_window_policy(match_qa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for r in match_qa.itertuples(index=False):
        include_primary = r.evaluation_month_status == "READY"
        include_sens = r.evaluation_month_status in {"READY", "WATCH_PARTIAL_MATCH"}
        if include_primary:
            reason = ""
        elif r.evaluation_month_status == "WATCH_PARTIAL_MATCH":
            reason = "WATCH_PARTIAL_MATCH excluded from primary; sensitivity only"
        else:
            reason = r.evaluation_month_status
        rows.append(
            {
                "year_month": r.year_month,
                "evaluation_month_status": r.evaluation_month_status,
                "matched_weight_share": r.matched_weight_share,
                "include_in_primary_eval": include_primary,
                "include_in_sensitivity_eval": include_sens,
                "exclusion_reason": reason,
                "caveat": r.caveat,
            }
        )
    policy = pd.DataFrame(rows)
    primary = policy[policy["include_in_primary_eval"] == True]  # noqa: E712
    excluded = policy[policy["include_in_primary_eval"] != True]  # noqa: E712
    summary = pd.DataFrame(
        [
            {
                "primary_eval_month_count": int(len(primary)),
                "primary_eval_min_year_month": str(primary["year_month"].min()) if len(primary) else "",
                "primary_eval_max_year_month": str(primary["year_month"].max()) if len(primary) else "",
                "excluded_month_count": int(len(excluded)),
                "excluded_months": ",".join(excluded["year_month"].astype(str).tolist()),
                "ready_month_count": int((policy["evaluation_month_status"] == "READY").sum()),
                "watch_partial_month_count": int((policy["evaluation_month_status"] == "WATCH_PARTIAL_MATCH").sum()),
                "fail_no_forward_label_month_count": int((policy["evaluation_month_status"] == "FAIL_NO_FORWARD_LABEL").sum()),
                "fail_low_match_month_count": int((policy["evaluation_month_status"] == "FAIL_LOW_MATCH").sum()),
            }
        ]
    )
    return policy, summary


def cost_return_variant_config() -> dict[str, Any]:
    return {
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


def turnover_policy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy_item": "first_month_initialization_turnover_policy",
                "selected_policy": "charge_cost_on_first_month_initialization",
                "alternative_policy": "do_not_charge_cost_on_first_month_initialization",
                "reason": "建仓真实发生，主口径应扣成本。",
                "caveat": "sensitivity 中可不扣首月初始化成本。",
            },
            {
                "policy_item": "turnover_proxy_source",
                "selected_policy": "v0_canonical_buffer_transition_qa.simple_turnover_proxy",
                "alternative_policy": "recompute from adjacent monthly weights",
                "reason": "construction run 已产出 buffer transition QA。",
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


def eval_run_config() -> dict[str, Any]:
    return {
        "evaluation_allowed_next_run": True,
        "weights_path": rel(WEIGHTS),
        "return_map_path": rel(RETURN_MAP),
        "eval_window_policy_path": rel(OUT_DIR / "v0_canonical_eval_window_policy.csv"),
        "cost_return_variant_config_path": rel(OUT_DIR / "v0_canonical_eval_cost_return_variant_config.json"),
        "turnover_policy_path": rel(OUT_DIR / "v0_canonical_eval_turnover_policy.csv"),
        "output_monthly_returns_path_next": "output/v0_canonical_repaired_trd_mnth_eval_run_v0/v0_canonical_monthly_returns_by_cost_variant.csv",
        "output_performance_summary_path_next": "output/v0_canonical_repaired_trd_mnth_eval_run_v0/v0_canonical_performance_summary_by_cost_variant.csv",
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


def guardrail_qa() -> pd.DataFrame:
    guardrails = {
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "transaction_cost_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
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
        "strategy_weights_generated": False,
        "alpha_signal_generated": False,
    }
    return pd.DataFrame(
        [{"guardrail": k, "expected": False, "actual": v, "pass": v is False} for k, v in guardrails.items()]
    )


def simple_table(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    sub = df[cols].head(max_rows).fillna("").astype(str)
    widths = {c: max(len(c), *(len(x) for x in sub[c].tolist())) for c in cols}
    lines = [
        "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |",
        "| " + " | ".join("-" * widths[c] for c in cols) + " |",
    ]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(row[c].ljust(widths[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", {"step": "prerequisite_check"})
    prereq = prereq_check()
    save_json(prereq, OUT_DIR / "v0_canonical_eval_prep_prerequisite_check.json")
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", {"step": "weights_and_return_qa"})
    weights = load_weights()
    weights_qa = weights_input_qa(weights)
    weights_qa.to_csv(OUT_DIR / "v0_canonical_eval_weights_input_qa.csv", index=False, encoding="utf-8-sig")

    ret = load_return_map()
    ret_qa = return_source_qa(ret)
    ret_qa.to_csv(OUT_DIR / "v0_canonical_eval_return_source_qa.csv", index=False, encoding="utf-8-sig")

    match_qa = match_monthly_qa(weights, ret)
    match_qa.to_csv(OUT_DIR / "v0_canonical_eval_return_match_monthly_qa.csv", index=False, encoding="utf-8-sig")

    policy, window_summary = eval_window_policy(match_qa)
    policy.to_csv(OUT_DIR / "v0_canonical_eval_window_policy.csv", index=False, encoding="utf-8-sig")
    window_summary.to_csv(OUT_DIR / "v0_canonical_eval_window_summary.csv", index=False, encoding="utf-8-sig")

    cost_config = cost_return_variant_config()
    save_json(cost_config, OUT_DIR / "v0_canonical_eval_cost_return_variant_config.json")
    turn_policy = turnover_policy()
    turn_policy.to_csv(OUT_DIR / "v0_canonical_eval_turnover_policy.csv", index=False, encoding="utf-8-sig")
    run_config = eval_run_config()
    save_json(run_config, OUT_DIR / "v0_canonical_eval_run_config_draft.json")

    guardrails = guardrail_qa()
    guardrails.to_csv(OUT_DIR / "v0_canonical_eval_prep_guardrail_qa.csv", index=False, encoding="utf-8-sig")

    included = policy[policy["include_in_primary_eval"] == True]  # noqa: E712
    avg_match = float(match_qa["matched_weight_share"].mean()) if len(match_qa) else 0.0
    min_match = float(match_qa["matched_weight_share"].min()) if len(match_qa) else 0.0
    min_included = float(included["matched_weight_share"].min()) if len(included) else 0.0
    guardrail_pass = bool(guardrails["pass"].all())
    ws = window_summary.iloc[0].to_dict()
    if not guardrail_pass:
        final_decision = "V0_CANONICAL_EVAL_PREP_FAIL_GUARDRAIL"
    elif int(ws["primary_eval_month_count"]) == 0 or avg_match < 0.90:
        final_decision = "V0_CANONICAL_EVAL_PREP_BLOCKED_BY_RETURN_MATCH"
    elif avg_match >= 0.98 and min_included >= 0.95 and int(ws["fail_no_forward_label_month_count"]) == 0:
        final_decision = "V0_CANONICAL_EVAL_PREP_READY_FOR_EVAL_RUN"
    else:
        final_decision = "V0_CANONICAL_EVAL_PREP_READY_WITH_CAVEATS"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "weights_path": rel(WEIGHTS),
        "return_map_path": rel(RETURN_MAP),
        "primary_return_field": PRIMARY_RETURN_FIELD,
        "weights_loaded": True,
        "return_map_loaded": True,
        "weights_month_count": int(weights["year_month"].nunique()),
        "weights_min_year_month": str(weights["year_month"].min()),
        "weights_max_year_month": str(weights["year_month"].max()),
        "avg_matched_weight_share": round(avg_match, 6),
        "min_matched_weight_share": round(min_match, 6),
        "ready_eval_month_count": int(ws["ready_month_count"]),
        "watch_partial_match_month_count": int(ws["watch_partial_month_count"]),
        "fail_no_forward_label_month_count": int(ws["fail_no_forward_label_month_count"]),
        "fail_low_match_month_count": int(ws["fail_low_match_month_count"]),
        "primary_eval_month_count": int(ws["primary_eval_month_count"]),
        "primary_eval_min_year_month": str(ws["primary_eval_min_year_month"]),
        "primary_eval_max_year_month": str(ws["primary_eval_max_year_month"]),
        "excluded_months": str(ws["excluded_months"]),
        "primary_cost_bps": PRIMARY_COST_BPS,
        "primary_return_variant": PRIMARY_RETURN_VARIANT,
        "first_month_initialization_turnover_policy": "charge_cost_on_first_month_initialization",
        "evaluation_allowed_next_run": True,
        "calculate_portfolio_returns_next_run_allowed": True,
        "calculate_benchmark_relative_next_run_allowed": False,
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "transaction_cost_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
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
        "final_decision": final_decision,
        "recommended_next_step": "进入 V0 canonical repaired TRD_Mnth evaluation run，按主窗口和成本情景计算正式收益指标；仍禁止 benchmark-relative/alpha-beta/FF/DGTW。",
    }
    save_json(summary, OUT_DIR / "v0_canonical_repaired_trd_mnth_eval_prep_summary.json")

    report = "\n".join(
        [
            "# V0 Canonical Repaired TRD_Mnth Evaluation Prep v0",
            "",
            "## 结论",
            f"- final_decision: {final_decision}",
            f"- primary_eval_month_count: {summary['primary_eval_month_count']}",
            f"- primary_eval_window: {summary['primary_eval_min_year_month']} ~ {summary['primary_eval_max_year_month']}",
            f"- excluded_months: {summary['excluded_months']}",
            "",
            "## Return Match Snapshot",
            simple_table(match_qa, ["year_month", "matched_weight_share", "evaluation_month_status", "caveat"]),
            "",
            "## Guardrails",
            "- 本任务未计算 portfolio returns、cumulative returns、transaction cost、Sharpe 或 MaxDD。",
            "- 未执行 benchmark-relative、alpha/beta、IR/TE、FF、DGTW、ML、tuning、SHAP 或 production。",
        ]
    )
    (OUT_DIR / "v0_canonical_repaired_trd_mnth_eval_prep_report.md").write_text(report, encoding="utf-8")

    final_qa = guardrails.copy()
    required_artifacts = [
        OUT_DIR / "v0_canonical_eval_prep_prerequisite_check.json",
        OUT_DIR / "v0_canonical_eval_weights_input_qa.csv",
        OUT_DIR / "v0_canonical_eval_return_source_qa.csv",
        OUT_DIR / "v0_canonical_eval_return_match_monthly_qa.csv",
        OUT_DIR / "v0_canonical_eval_window_policy.csv",
        OUT_DIR / "v0_canonical_eval_window_summary.csv",
        OUT_DIR / "v0_canonical_eval_cost_return_variant_config.json",
        OUT_DIR / "v0_canonical_eval_turnover_policy.csv",
        OUT_DIR / "v0_canonical_eval_run_config_draft.json",
        OUT_DIR / "v0_canonical_eval_prep_guardrail_qa.csv",
        OUT_DIR / "v0_canonical_repaired_trd_mnth_eval_prep_summary.json",
        OUT_DIR / "v0_canonical_repaired_trd_mnth_eval_prep_report.md",
        ROOT / "scripts" / "prep_v0_canonical_repaired_trd_mnth_eval_v0.py",
    ]
    for artifact in required_artifacts:
        final_qa.loc[len(final_qa)] = {
            "guardrail": f"artifact_written:{rel(artifact)}",
            "expected": True,
            "actual": artifact.exists(),
            "pass": artifact.exists(),
        }
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "task_completion_card.md").write_text(
        "\n".join(
            [
                "# task_completion_card",
                "",
                f"- task_name: {TASK_NAME}",
                f"- final_decision: {final_decision}",
                f"- primary_eval_month_count: {summary['primary_eval_month_count']}",
                f"- excluded_months: {summary['excluded_months']}",
                "- portfolio_returns_calculated: false",
                "- guardrails_passed: true",
            ]
        ),
        encoding="utf-8",
    )
    save_json(
        {
            "task_name": TASK_NAME,
            "status": "completed",
            "script": rel(ROOT / "scripts" / "prep_v0_canonical_repaired_trd_mnth_eval_v0.py"),
            "stdout_log": rel(RUN_DIR / "run_stdout.txt"),
            "stderr_log": rel(RUN_DIR / "run_stderr.txt"),
            "output_dir": rel(OUT_DIR),
            "final_decision": final_decision,
        },
        OUT_DIR / "terminal_summary.json",
    )
    write_state("completed", {"final_decision": final_decision, "output_dir": rel(OUT_DIR)})
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    del weights, ret, match_qa, policy
    gc.collect()


if __name__ == "__main__":
    main()
