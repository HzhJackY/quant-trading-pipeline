from __future__ import annotations

import gc
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


TASK_NAME = "v0_route_b_eval_prep_recheck_with_label_policy_v0"
TASK_TITLE = "V0 Route B Evaluation Prep Recheck with Label Policy v0"
POLICY_NAME = "EXCLUDE_AFFECTED_MONTH_FROM_PRIMARY_EVAL"
PREV_TASK_TITLE = "V0 Route B Raw TRD Evidence Acquisition for Missing Labels v0"
PREV_FINAL_DECISION = "RAW_TRD_EVIDENCE_SUPPORTS_POLICY_RECHECK"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = repo_root()
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

INPUTS = {
    "weights": ROOT / "output" / "v0_legacy_compatible_pit_strict_lag_replay_portfolio_construction_run_v0" / "v0_route_b_research_weights.parquet",
    "portfolio_summary": ROOT / "output" / "v0_legacy_compatible_pit_strict_lag_replay_portfolio_construction_run_v0" / "v0_legacy_compatible_pit_strict_lag_replay_portfolio_construction_summary.json",
    "return_map": ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0" / "canonical_csmar_trd_mnth_return_map_repaired.parquet",
    "raw_summary": ROOT / "output" / "v0_route_b_raw_trd_evidence_acquisition_v0" / "v0_route_b_raw_trd_evidence_acquisition_summary.json",
    "raw_lookup_matrix": ROOT / "output" / "v0_route_b_raw_trd_evidence_acquisition_v0" / "v0_missing_label_raw_trd_lookup_matrix.csv",
    "diagnosis": ROOT / "output" / "v0_route_b_raw_trd_evidence_acquisition_v0" / "v0_missing_label_no_trade_suspension_delisting_diagnosis.csv",
    "policy_design": ROOT / "output" / "v0_route_b_raw_trd_evidence_acquisition_v0" / "v0_missing_label_handling_policy_design.csv",
    "unblock_decision": ROOT / "output" / "v0_route_b_raw_trd_evidence_acquisition_v0" / "v0_raw_trd_evidence_eval_unblock_decision.csv",
    "repair_summary": ROOT / "output" / "v0_route_b_label_edge_case_repair_recheck_v0" / "v0_route_b_label_edge_case_repair_recheck_summary.json",
    "lineage": ROOT / "output" / "v0_route_b_label_edge_case_repair_recheck_v0" / "v0_route_b_label_lineage_drilldown.csv",
    "reconstruction": ROOT / "output" / "v0_route_b_label_edge_case_repair_recheck_v0" / "v0_route_b_fwd_ret_reconstruction_check.csv",
    "repair_decision": ROOT / "output" / "v0_route_b_label_edge_case_repair_recheck_v0" / "v0_route_b_label_edge_case_repair_decision.csv",
}


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def month_add_1(year_month: str) -> str:
    y, m = map(int, year_month.split("-"))
    if m == 12:
        return f"{y + 1:04d}-01"
    return f"{y:04d}-{m + 1:02d}"


def bool_cell(value: bool) -> str:
    return "true" if bool(value) else "false"


def write_run_state(status: str, final_decision: str | None, note: str) -> None:
    content = "\n".join(
        [
            f"# {TASK_TITLE}",
            "",
            f"- task_name: `{TASK_NAME}`",
            f"- status: `{status}`",
            f"- final_decision: `{final_decision or 'PENDING'}`",
            f"- output_dir: `{OUT_DIR}`",
            f"- run_dir: `{RUN_DIR}`",
            f"- note: {note}",
            "",
            "Resume protocol: rerun the script with stdout/stderr redirected to the run_dir logs.",
        ]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "RUN_STATE.md").write_text(content, encoding="utf-8")
    (RUN_DIR / "RUN_STATE.md").write_text(content, encoding="utf-8")


def make_blocked_outputs(reason: str, final_decision: str, prereq_rows: list[dict]) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    pd.DataFrame(prereq_rows).to_csv(OUT_DIR / "v0_route_b_eval_prep_recheck_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    summary = {
        "run_timestamp": now,
        "task_name": TASK_TITLE,
        "prerequisites_passed": False,
        "previous_final_decision": None,
        "policy_name": POLICY_NAME,
        "policy_application_passed": False,
        "weights_month_count": None,
        "weights_min_year_month": None,
        "weights_max_year_month": None,
        "expected_final_no_label_months": [],
        "excluded_policy_month_count": 0,
        "excluded_policy_months": [],
        "primary_eval_month_count_after_policy": 0,
        "primary_eval_min_year_month_after_policy": None,
        "primary_eval_max_year_month_after_policy": None,
        "primary_eval_included_missing_label_count": None,
        "primary_eval_included_missing_label_weight_share": None,
        "remaining_unexpected_missing_label_count": None,
        "remaining_unexpected_missing_label_weight_share": None,
        "avg_matched_weight_share_primary_eval": None,
        "min_matched_weight_share_primary_eval": None,
        "guardrails_passed": False,
        "evaluation_block_removed": False,
        "route_b_formal_eval_allowed_next": False,
        "calculate_returns_next_run_allowed": False,
        "benchmark_relative_allowed": False,
        "ff_allowed": False,
        "dgtw_allowed": False,
        "production_allowed": False,
        "final_decision": final_decision,
        "recommended_next_step": reason,
    }
    write_json(OUT_DIR / "v0_route_b_eval_prep_recheck_with_label_policy_summary.json", summary)
    (OUT_DIR / "report.md").write_text(f"# {TASK_TITLE}\n\nBLOCKED: {reason}\n", encoding="utf-8")
    write_json(OUT_DIR / "terminal_summary.json", {"final_decision": final_decision, "reason": reason})
    pd.DataFrame([summary]).to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "task_completion_card.md").write_text(f"# 任务完成卡\n\n- final_decision: `{final_decision}`\n- reason: {reason}\n", encoding="utf-8")
    write_run_state("blocked", final_decision, reason)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_run_state("running", None, "starting prerequisite checks")

    scripts_dir = OUT_DIR / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), scripts_dir / Path(__file__).name)

    prereq_rows = []
    missing_inputs = []
    for name, path in INPUTS.items():
        exists = path.exists()
        prereq_rows.append({"check": f"input_exists_{name}", "expected": True, "actual": exists, "pass": exists})
        if not exists:
            missing_inputs.append(str(path))
    if missing_inputs:
        make_blocked_outputs("missing required input files", "ROUTE_B_EVAL_PREP_RECHECK_WITH_POLICY_FAIL_GUARDRAIL", prereq_rows)
        return

    raw_summary = read_json(INPUTS["raw_summary"])
    required_summary_checks = {
        "final_decision": PREV_FINAL_DECISION,
        "prerequisites_passed": True,
        "missing_label_policy_designed": True,
        "recommended_policy": POLICY_NAME,
        "eval_prep_recheck_allowed": True,
        "direct_eval_run_allowed": False,
        "calculate_returns_next_run_allowed": False,
    }
    for key, expected in required_summary_checks.items():
        actual = raw_summary.get(key)
        ok = actual == expected
        prereq_rows.append({"check": f"raw_summary_{key}", "expected": expected, "actual": actual, "pass": ok})
    prerequisites_passed = all(row["pass"] for row in prereq_rows)
    if not prerequisites_passed:
        make_blocked_outputs("previous raw TRD evidence summary does not authorize policy recheck", "ROUTE_B_EVAL_PREP_RECHECK_WITH_POLICY_FAIL_GUARDRAIL", prereq_rows)
        return

    diagnosis = pd.read_csv(INPUTS["diagnosis"], dtype=str)
    policy_design = pd.read_csv(INPUTS["policy_design"], dtype=str)
    lineage = pd.read_csv(INPUTS["lineage"], dtype={"case_id": str, "year_month": str, "symbol_norm": str, "weight": float})
    raw_gap = diagnosis.loc[diagnosis["final_diagnosis"].eq("VERIFIED_RAW_TRD_GAP")].copy()
    cases = (
        raw_gap[["case_id", "symbol_norm", "year_month", "expected_label_month", "final_diagnosis"]]
        .merge(policy_design[["case_id", "recommended_policy"]], on="case_id", how="left")
        .merge(lineage[["case_id", "weight", "diagnosis"]], on="case_id", how="left", suffixes=("", "_lineage"))
    )
    policy_ambiguous = (
        len(cases) != 3
        or cases[["case_id", "symbol_norm", "year_month", "expected_label_month"]].isna().any().any()
        or not cases["recommended_policy"].eq(POLICY_NAME).all()
        or cases["case_id"].duplicated().any()
    )
    cases["policy_action"] = "exclude affected portfolio month from primary eval"
    cases["missing_reason"] = cases["final_diagnosis"]
    cases = cases.rename(
        columns={
            "symbol_norm": "symbol",
            "year_month": "portfolio_month",
            "expected_label_month": "forward_label_month",
            "weight": "selected_weight",
        }
    )
    excluded_policy_months = sorted(cases["portfolio_month"].dropna().unique().tolist())
    policy_excluded = cases[
        [
            "case_id",
            "symbol",
            "portfolio_month",
            "forward_label_month",
            "selected_weight",
            "missing_reason",
            "policy_action",
        ]
    ].copy()
    policy_excluded.to_csv(OUT_DIR / "v0_route_b_policy_excluded_months.csv", index=False, encoding="utf-8-sig")

    weights_cols = ["portfolio_name", "year_month", "symbol_norm", "weight"]
    weights = pq.read_table(INPUTS["weights"], columns=weights_cols).to_pandas()
    weights["year_month"] = weights["year_month"].astype(str)
    weights["symbol_norm"] = weights["symbol_norm"].astype(str)
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")
    all_months = sorted(weights["year_month"].unique().tolist())
    final_no_label_months = [max(all_months)]
    primary_months = [m for m in all_months if m not in set(final_no_label_months) | set(excluded_policy_months)]

    eval_window = {
        "all_weight_months": len(all_months),
        "weights_min_year_month": min(all_months),
        "weights_max_year_month": max(all_months),
        "excluded_final_no_label_months": ";".join(final_no_label_months),
        "excluded_policy_months": ";".join(excluded_policy_months),
        "primary_eval_months_after_policy": len(primary_months),
        "primary_eval_min_year_month": min(primary_months) if primary_months else None,
        "primary_eval_max_year_month": max(primary_months) if primary_months else None,
        "primary_eval_month_count": len(primary_months),
    }
    pd.DataFrame([eval_window]).to_csv(OUT_DIR / "v0_route_b_eval_window_recheck.csv", index=False, encoding="utf-8-sig")

    return_map = pq.read_table(INPUTS["return_map"], columns=["symbol_norm", "year_month", "fwd_ret_1m"]).to_pandas()
    return_map["symbol_norm"] = return_map["symbol_norm"].astype(str)
    return_map["year_month"] = return_map["year_month"].astype(str)
    return_map["fwd_ret_1m_available_flag"] = return_map["fwd_ret_1m"].notna()
    return_flags = (
        return_map[["symbol_norm", "year_month", "fwd_ret_1m_available_flag"]]
        .drop_duplicates(["symbol_norm", "year_month"], keep="first")
    )
    del return_map
    gc.collect()

    detail = weights.merge(return_flags, on=["symbol_norm", "year_month"], how="left")
    del return_flags
    gc.collect()
    detail["fwd_ret_1m_available_flag"] = detail["fwd_ret_1m_available_flag"].fillna(False).astype(bool)
    detail["label_exists"] = detail["fwd_ret_1m_available_flag"]
    detail["portfolio_month"] = detail["year_month"]
    detail["symbol"] = detail["symbol_norm"]
    detail["forward_label_month"] = detail["portfolio_month"].map(month_add_1)
    case_key = set(zip(cases["portfolio_month"], cases["symbol"]))
    detail["is_registered_policy_case"] = list(zip(detail["portfolio_month"], detail["symbol"]))
    detail["is_registered_policy_case"] = detail["is_registered_policy_case"].map(lambda x: x in case_key)
    detail["policy_exclusion_applied"] = detail["portfolio_month"].isin(excluded_policy_months)
    detail["excluded_final_no_label"] = detail["portfolio_month"].isin(final_no_label_months)
    detail["primary_eval_included"] = detail["portfolio_month"].isin(primary_months)
    detail["missing_label_reason"] = ""
    detail.loc[detail["excluded_final_no_label"] & ~detail["label_exists"], "missing_label_reason"] = "EXPECTED_FINAL_NO_LABEL"
    detail.loc[detail["is_registered_policy_case"] & ~detail["label_exists"], "missing_label_reason"] = "VERIFIED_RAW_TRD_GAP"
    unexpected_missing_mask = (
        ~detail["label_exists"]
        & ~detail["excluded_final_no_label"]
        & ~detail["is_registered_policy_case"]
    )
    detail.loc[unexpected_missing_mask, "missing_label_reason"] = "UNEXPECTED_MISSING_LABEL"
    detail["detail_status"] = "LABEL_AVAILABLE"
    detail.loc[detail["primary_eval_included"] & detail["label_exists"], "detail_status"] = "PRIMARY_EVAL_LABEL_AVAILABLE"
    detail.loc[detail["primary_eval_included"] & ~detail["label_exists"], "detail_status"] = "BLOCKED_UNEXPECTED_MISSING_LABEL"
    detail.loc[detail["excluded_final_no_label"], "detail_status"] = "EXCLUDED_FINAL_NO_LABEL"
    detail.loc[detail["policy_exclusion_applied"] & detail["label_exists"], "detail_status"] = "EXCLUDED_POLICY_MONTH_LABEL_AVAILABLE"
    detail.loc[detail["is_registered_policy_case"] & ~detail["label_exists"], "detail_status"] = "EXCLUDED_POLICY_RAW_TRD_GAP"
    detail_out = detail[
        [
            "portfolio_month",
            "symbol",
            "weight",
            "forward_label_month",
            "label_exists",
            "fwd_ret_1m_available_flag",
            "missing_label_reason",
            "policy_exclusion_applied",
            "primary_eval_included",
            "detail_status",
        ]
    ].copy()
    detail_out.to_csv(OUT_DIR / "v0_route_b_label_match_detail_after_policy.csv", index=False, encoding="utf-8-sig")

    missing_detail = detail_out.loc[~detail_out["label_exists"]].copy()
    missing_detail.to_csv(OUT_DIR / "v0_route_b_missing_label_detail_after_policy.csv", index=False, encoding="utf-8-sig")

    monthly_rows = []
    for month, g in detail.groupby("portfolio_month", sort=True):
        selected_count = int(len(g))
        total_weight = float(g["weight"].sum())
        missing = g.loc[~g["label_exists"]]
        missing_count = int(len(missing))
        missing_weight = float(missing["weight"].sum()) if missing_count else 0.0
        matched_weight_share = 1.0 - missing_weight / total_weight if total_weight else 0.0
        if month in final_no_label_months:
            status = "EXCLUDED_FINAL_NO_LABEL"
            eligible = False
            reason = "expected final no-label month"
        elif month in excluded_policy_months:
            status = "EXCLUDED_POLICY_RAW_TRD_GAP"
            eligible = False
            reason = "pre-registered raw TRD gap policy exclusion"
        elif missing_count:
            status = "BLOCKED_UNEXPECTED_MISSING_LABEL"
            eligible = False
            reason = "unexpected missing label remains in primary eval candidate month"
        else:
            status = "PRIMARY_EVAL_INCLUDED"
            eligible = True
            reason = "all selected holdings have fwd_ret_1m availability"
        monthly_rows.append(
            {
                "portfolio_month": month,
                "selected_count": selected_count,
                "total_weight": total_weight,
                "matched_weight_share": matched_weight_share,
                "unmatched_weight_share": 1.0 - matched_weight_share,
                "missing_label_count": missing_count,
                "missing_label_weight_share": missing_weight / total_weight if total_weight else 0.0,
                "month_status": status,
                "primary_eval_eligible": eligible,
                "reason": reason,
            }
        )
    monthly_qa = pd.DataFrame(monthly_rows)
    monthly_qa.to_csv(OUT_DIR / "v0_route_b_label_match_monthly_qa_after_policy.csv", index=False, encoding="utf-8-sig")

    affected_weight_share_original = float(cases["selected_weight"].astype(float).sum())
    audit = pd.DataFrame(
        [
            {
                "policy_name": POLICY_NAME,
                "policy_source_task": PREV_TASK_TITLE,
                "policy_source_final_decision": PREV_FINAL_DECISION,
                "affected_case_count": int(len(cases)),
                "affected_month_count": int(len(excluded_policy_months)),
                "affected_weight_share_original": affected_weight_share_original,
                "policy_action": "exclude affected portfolio month from primary eval",
                "zero_fill_used": False,
                "holding_deleted": False,
                "matched_only_renormalization_used": False,
                "original_return_map_modified": False,
                "route_b_weights_modified": False,
                "old_artifacts_overwritten": False,
                "policy_pre_registered": True,
                "policy_application_passed": not policy_ambiguous,
            }
        ]
    )
    audit.to_csv(OUT_DIR / "v0_route_b_policy_application_audit.csv", index=False, encoding="utf-8-sig")

    primary_missing = detail.loc[detail["primary_eval_included"] & ~detail["label_exists"]]
    remaining_unexpected = detail.loc[unexpected_missing_mask]
    primary_monthly = monthly_qa.loc[monthly_qa["month_status"].eq("PRIMARY_EVAL_INCLUDED")]
    primary_eval_window_locked = (not policy_ambiguous) and primary_missing.empty and len(primary_months) > 0

    guard_checks = {
        "no_portfolio_returns_calculated": True,
        "no_cumulative_returns_calculated": True,
        "no_transaction_cost_calculated": True,
        "no_sharpe_calculated": True,
        "no_maxdd_calculated": True,
        "no_tstat_calculated": True,
        "no_benchmark_relative_calculated": True,
        "no_active_return_calculated": True,
        "no_alpha_beta_calculated": True,
        "no_ir_te_calculated": True,
        "no_ff_calculated": True,
        "no_dgtw_calculated": True,
        "no_production": True,
        "no_ml_training": True,
        "no_shap": True,
        "no_zero_fill": True,
        "no_delete_missing_holdings": True,
        "no_matched_only_renormalization_bypass": True,
        "no_original_return_map_modified": True,
        "no_route_b_weights_modified": True,
        "no_old_artifacts_overwritten": True,
        "policy_exclusion_only": True,
        "primary_eval_window_locked": primary_eval_window_locked,
        "remaining_unexpected_missing_label_count_is_zero": int(len(remaining_unexpected)) == 0,
    }
    guardrails_passed = all(guard_checks.values())
    guard_rows = [{"check": k, "expected": True, "actual": v, "pass": bool(v)} for k, v in guard_checks.items()]
    guard_rows.append({"check": "guardrails_passed", "expected": True, "actual": guardrails_passed, "pass": guardrails_passed})
    pd.DataFrame(guard_rows).to_csv(OUT_DIR / "v0_route_b_eval_prep_recheck_guardrail_qa.csv", index=False, encoding="utf-8-sig")

    if policy_ambiguous:
        final_decision = "ROUTE_B_EVAL_PREP_RECHECK_WITH_POLICY_BLOCKED_BY_POLICY_AMBIGUITY"
    elif not guardrails_passed:
        final_decision = "ROUTE_B_EVAL_PREP_RECHECK_WITH_POLICY_FAIL_GUARDRAIL"
    elif len(primary_missing) > 0:
        final_decision = "ROUTE_B_EVAL_PREP_RECHECK_WITH_POLICY_BLOCKED_BY_UNEXPECTED_MISSING_LABELS"
    else:
        final_decision = "ROUTE_B_EVAL_PREP_RECHECK_WITH_POLICY_READY_WITH_CAVEATS"

    allowed_next = final_decision in {
        "ROUTE_B_EVAL_PREP_RECHECK_WITH_POLICY_READY_FOR_FORMAL_EVAL",
        "ROUTE_B_EVAL_PREP_RECHECK_WITH_POLICY_READY_WITH_CAVEATS",
    }
    recommended_next_step = (
        "run Route B formal evaluation with locked primary window and disclose policy exclusion caveat"
        if allowed_next
        else "resolve blocking policy or missing-label issue before formal evaluation"
    )
    config = {
        "portfolio_name": str(weights["portfolio_name"].dropna().iloc[0]) if "portfolio_name" in weights and not weights["portfolio_name"].dropna().empty else "route_b",
        "weights_path": str(INPUTS["weights"]),
        "return_map_path": str(INPUTS["return_map"]),
        "primary_return_field": "Mretwd",
        "label_field": "fwd_ret_1m",
        "policy_name": POLICY_NAME,
        "excluded_final_no_label_months": final_no_label_months,
        "excluded_policy_months": excluded_policy_months,
        "primary_eval_months": primary_months,
        "primary_eval_min_year_month": min(primary_months) if primary_months else None,
        "primary_eval_max_year_month": max(primary_months) if primary_months else None,
        "primary_eval_month_count": len(primary_months),
        "primary_cost_bps": 20,
        "return_variant": "raw_unmatched_not_renormalized",
        "first_month_initialization_turnover_policy": "charge_cost_on_first_month_initialization",
        "benchmark_relative_allowed": False,
        "ff_allowed": False,
        "dgtw_allowed": False,
        "production_allowed": False,
        "route_b_formal_eval_allowed_next": allowed_next,
        "calculate_returns_allowed_next": allowed_next,
    }
    write_json(OUT_DIR / "v0_route_b_formal_eval_next_run_config.json", config)

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "task_name": TASK_TITLE,
        "prerequisites_passed": prerequisites_passed,
        "previous_final_decision": raw_summary.get("final_decision"),
        "policy_name": POLICY_NAME,
        "policy_application_passed": not policy_ambiguous,
        "weights_month_count": len(all_months),
        "weights_min_year_month": min(all_months),
        "weights_max_year_month": max(all_months),
        "expected_final_no_label_months": final_no_label_months,
        "excluded_policy_month_count": len(excluded_policy_months),
        "excluded_policy_months": excluded_policy_months,
        "primary_eval_month_count_after_policy": len(primary_months),
        "primary_eval_min_year_month_after_policy": min(primary_months) if primary_months else None,
        "primary_eval_max_year_month_after_policy": max(primary_months) if primary_months else None,
        "primary_eval_included_missing_label_count": int(len(primary_missing)),
        "primary_eval_included_missing_label_weight_share": float(primary_missing["weight"].sum()) if len(primary_missing) else 0.0,
        "remaining_unexpected_missing_label_count": int(len(remaining_unexpected)),
        "remaining_unexpected_missing_label_weight_share": float(remaining_unexpected["weight"].sum()) if len(remaining_unexpected) else 0.0,
        "avg_matched_weight_share_primary_eval": float(primary_monthly["matched_weight_share"].mean()) if len(primary_monthly) else None,
        "min_matched_weight_share_primary_eval": float(primary_monthly["matched_weight_share"].min()) if len(primary_monthly) else None,
        "guardrails_passed": guardrails_passed,
        "evaluation_block_removed": allowed_next,
        "route_b_formal_eval_allowed_next": allowed_next,
        "calculate_returns_next_run_allowed": allowed_next,
        "benchmark_relative_allowed": False,
        "ff_allowed": False,
        "dgtw_allowed": False,
        "production_allowed": False,
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    write_json(OUT_DIR / "v0_route_b_eval_prep_recheck_with_label_policy_summary.json", summary)

    report = "\n".join(
        [
            f"# {TASK_TITLE}",
            "",
            f"- final_decision: `{final_decision}`",
            f"- policy_name: `{POLICY_NAME}`",
            f"- excluded_policy_months: `{';'.join(excluded_policy_months)}`",
            f"- primary_eval_window_after_policy: `{summary['primary_eval_min_year_month_after_policy']}` to `{summary['primary_eval_max_year_month_after_policy']}`, {len(primary_months)} months",
            f"- primary_eval_included_missing_label_count: `{len(primary_missing)}`",
            f"- remaining_unexpected_missing_label_count: `{len(remaining_unexpected)}`",
            f"- avg_matched_weight_share_primary_eval: `{summary['avg_matched_weight_share_primary_eval']}`",
            f"- min_matched_weight_share_primary_eval: `{summary['min_matched_weight_share_primary_eval']}`",
            f"- guardrails_passed: `{guardrails_passed}`",
            "",
            "说明：本轮只做 label availability 和 policy exclusion recheck，未计算组合收益、累计收益或任何绩效指标。",
        ]
    )
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    (OUT_DIR / "task_completion_card.md").write_text(
        "\n".join(
            [
                "# 任务完成卡",
                "",
                f"- task_name: `{TASK_TITLE}`",
                f"- final_decision: `{final_decision}`",
                f"- prerequisites_passed: `{bool_cell(prerequisites_passed)}`",
                f"- guardrails_passed: `{bool_cell(guardrails_passed)}`",
                f"- output_dir: `{OUT_DIR}`",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        OUT_DIR / "terminal_summary.json",
        {
            "task_name": TASK_TITLE,
            "final_decision": final_decision,
            "prerequisites_passed": prerequisites_passed,
            "policy_application_passed": not policy_ambiguous,
            "excluded_policy_months": excluded_policy_months,
            "primary_eval_month_count_after_policy": len(primary_months),
            "guardrails_passed": guardrails_passed,
            "route_b_formal_eval_allowed_next": allowed_next,
            "calculate_returns_next_run_allowed": allowed_next,
        },
    )
    pd.DataFrame([summary]).to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    write_run_state("completed", final_decision, recommended_next_step)

    del weights, detail, detail_out, missing_detail, monthly_qa, diagnosis, policy_design, lineage, cases
    gc.collect()


if __name__ == "__main__":
    main()
