from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_NAME = "V0 Legacy-Compatible PIT Strict-Lag Replay Evaluation Prep v0"
OUT_NAME = "v0_legacy_compatible_pit_strict_lag_replay_eval_prep_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / OUT_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

CONSTRUCTION_DIR = ROOT / "output" / "v0_legacy_compatible_pit_strict_lag_replay_portfolio_construction_run_v0"
WEIGHTS = CONSTRUCTION_DIR / "v0_route_b_research_weights.parquet"
CONSTRUCTION_SUMMARY = CONSTRUCTION_DIR / "v0_legacy_compatible_pit_strict_lag_replay_portfolio_construction_summary.json"
MONTHLY_WEIGHT_QA = CONSTRUCTION_DIR / "v0_route_b_portfolio_weight_monthly_qa.csv"
BUFFER_TRANSITION_QA = CONSTRUCTION_DIR / "v0_route_b_buffer_transition_qa.csv"
FUTURE_COVERAGE_PLAN = CONSTRUCTION_DIR / "v0_route_b_weights_future_eval_coverage_plan.csv"

RETURN_MAP = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0" / "canonical_csmar_trd_mnth_return_map_repaired.parquet"
ADAPTER = ROOT / "output" / "v0_legacy_compatible_pit_adapter_replay_dry_run_v0" / "v0_pit_legacy_compatible_input.parquet"
ROUTE_B_ALPHA = ROOT / "output" / "v0_legacy_compatible_pit_strict_lag_replay_alpha_build_v0" / "v0_legacy_pit_route_b_strict_lag_alpha_panel.parquet"

PRIMARY_RETURN_FIELD = "Mretwd"
PRIMARY_COST_BPS = 20
PRIMARY_RETURN_VARIANT = "raw_unmatched_not_renormalized"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_state(status: str, checkpoint: str, extra: dict[str, Any] | None = None) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": TASK_NAME,
        "status": status,
        "checkpoint": checkpoint,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "resume_instruction": f"先读取 {rel(RUN_DIR / 'RUN_STATE.md')}；继续时运行 scripts\\prep_v0_legacy_compatible_pit_strict_lag_replay_eval_v0.py，并重定向 stdout/stderr 到本目录。",
    }
    if extra:
        payload.update(extra)
    lines = [
        "# RUN_STATE", "", f"- task_name: {TASK_NAME}", f"- status: {status}",
        f"- checkpoint: {checkpoint}", "", "```json",
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), "```",
    ]
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(lines), encoding="utf-8")


def norm_symbol(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)


def prereq_check() -> dict[str, Any]:
    flags = {
        "route_b_weights_found": WEIGHTS.exists(),
        "construction_summary_found": CONSTRUCTION_SUMMARY.exists(),
        "monthly_weight_qa_found": MONTHLY_WEIGHT_QA.exists(),
        "buffer_transition_qa_found": BUFFER_TRANSITION_QA.exists(),
        "future_eval_coverage_plan_found": FUTURE_COVERAGE_PLAN.exists(),
        "trd_mnth_return_map_found": RETURN_MAP.exists(),
        "adapter_artifact_found": ADAPTER.exists(),
        "route_b_alpha_found": ROUTE_B_ALPHA.exists(),
    }
    paths = {
        "route_b_weights_found": WEIGHTS,
        "construction_summary_found": CONSTRUCTION_SUMMARY,
        "monthly_weight_qa_found": MONTHLY_WEIGHT_QA,
        "buffer_transition_qa_found": BUFFER_TRANSITION_QA,
        "future_eval_coverage_plan_found": FUTURE_COVERAGE_PLAN,
        "trd_mnth_return_map_found": RETURN_MAP,
        "adapter_artifact_found": ADAPTER,
        "route_b_alpha_found": ROUTE_B_ALPHA,
    }
    missing = [rel(p) for k, p in paths.items() if not flags[k]]
    flags["prerequisites_passed"] = len(missing) == 0
    flags["missing_files"] = missing
    flags["caveat"] = "Evaluation prep only; no portfolio returns, costs, or performance metrics are calculated."
    return flags


def load_weights() -> pd.DataFrame:
    cols = ["portfolio_name", "year_month", "symbol_norm", "weight", "rank", "alpha_signal_route_b_strict_lag"]
    weights = pd.read_parquet(WEIGHTS, columns=cols)
    weights["symbol_norm"] = norm_symbol(weights["symbol_norm"])
    weights["year_month"] = weights["year_month"].astype(str).str.slice(0, 7)
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")
    weights["rank"] = pd.to_numeric(weights["rank"], errors="coerce")
    weights["alpha_signal_route_b_strict_lag"] = pd.to_numeric(weights["alpha_signal_route_b_strict_lag"], errors="coerce")
    return weights


def weights_input_qa(weights: pd.DataFrame) -> pd.DataFrame:
    monthly = pd.read_csv(MONTHLY_WEIGHT_QA, dtype={"year_month": str})
    trans = pd.read_csv(BUFFER_TRANSITION_QA, dtype={"year_month": str})
    first_turnover = float(trans.sort_values("year_month")["turnover_proxy"].iloc[0]) if len(trans) else np.nan
    rows = [
        ("portfolio_name", "non-empty", ";".join(sorted(weights["portfolio_name"].dropna().unique().astype(str).tolist())), weights["portfolio_name"].notna().all(), ""),
        ("row_count", ">0", int(len(weights)), len(weights) > 0, ""),
        ("unique_symbol_count", ">0", int(weights["symbol_norm"].nunique()), weights["symbol_norm"].nunique() > 0, ""),
        ("month_count", ">0", int(weights["year_month"].nunique()), weights["year_month"].nunique() > 0, ""),
        ("min_year_month", "2017-02", str(weights["year_month"].min()), str(weights["year_month"].min()) == "2017-02", ""),
        ("max_year_month", "2026-06", str(weights["year_month"].max()), str(weights["year_month"].max()) == "2026-06", ""),
        ("duplicate symbol-month", 0, int(weights.duplicated(["symbol_norm", "year_month"]).sum()), int(weights.duplicated(["symbol_norm", "year_month"]).sum()) == 0, ""),
        ("selected_count per month", 50, int(monthly["selected_count"].min()), bool((monthly["selected_count"] == 50).all()), ""),
        ("weight_sum per month", 1.0, float(monthly["weight_sum"].mean()), bool((monthly["weight_sum_abs_error"] <= 1e-12).all()), ""),
        ("alpha_missing_selected_count", 0, int(monthly["alpha_missing_selected_count"].sum()), int(monthly["alpha_missing_selected_count"].sum()) == 0, ""),
        ("turnover_proxy availability", True, len(trans) > 0 and trans["turnover_proxy"].notna().all(), len(trans) > 0 and trans["turnover_proxy"].notna().all(), ""),
        ("first construction month turnover policy", 1.0, first_turnover, abs(first_turnover - 1.0) <= 1e-12, "first month initialization turnover is a policy input; no cost calculated"),
    ]
    return pd.DataFrame(rows, columns=["check_name", "expected", "actual", "pass", "caveat"])


def load_return_map() -> pd.DataFrame:
    ret = pd.read_parquet(RETURN_MAP, columns=["symbol_norm", "year_month", "fwd_ret_1m", "primary_return_field"])
    ret["symbol_norm"] = norm_symbol(ret["symbol_norm"])
    ret["year_month"] = ret["year_month"].astype(str).str.slice(0, 7)
    ret["fwd_ret_1m"] = pd.to_numeric(ret["fwd_ret_1m"], errors="coerce")
    ret["primary_return_field"] = ret["primary_return_field"].astype(str)
    ret = ret.loc[ret["primary_return_field"].eq(PRIMARY_RETURN_FIELD)].copy()
    ret = ret.drop_duplicates(["symbol_norm", "year_month"], keep="last")
    return ret


def return_source_qa(ret: pd.DataFrame) -> pd.DataFrame:
    missing_by_month = ret.groupby("year_month")["fwd_ret_1m"].apply(lambda s: int(s.isna().sum())).to_dict()
    rows = [
        ("primary_return_field = Mretwd", PRIMARY_RETURN_FIELD, PRIMARY_RETURN_FIELD, True, ""),
        ("row_count", ">0", int(len(ret)), len(ret) > 0, ""),
        ("unique_symbol_count", ">0", int(ret["symbol_norm"].nunique()), ret["symbol_norm"].nunique() > 0, ""),
        ("year_month_count", ">0", int(ret["year_month"].nunique()), ret["year_month"].nunique() > 0, ""),
        ("min_year_month", "non-empty", str(ret["year_month"].min()), pd.notna(ret["year_month"].min()), ""),
        ("max_year_month", "non-empty", str(ret["year_month"].max()), pd.notna(ret["year_month"].max()), ""),
        ("duplicate symbol-month", 0, int(ret.duplicated(["symbol_norm", "year_month"]).sum()), int(ret.duplicated(["symbol_norm", "year_month"]).sum()) == 0, ""),
        ("fwd_ret_1m null count", "tracked", int(ret["fwd_ret_1m"].isna().sum()), True, ""),
        ("fwd_ret_1m extreme count abs>1", "tracked", int((ret["fwd_ret_1m"].abs() > 1).sum()), True, ""),
        ("fwd_ret_1m missing by month", "tracked", json.dumps({str(k): int(v) for k, v in missing_by_month.items()}, ensure_ascii=False), True, ""),
        ("symbol_norm format consistency", "6 digit", float(ret["symbol_norm"].astype(str).str.match(r"^\\d{6}$").mean()), ret["symbol_norm"].astype(str).str.match(r"^\\d{6}$").mean() > 0.99, ""),
    ]
    return pd.DataFrame(rows, columns=["check_name", "expected", "actual", "pass", "caveat"])


def match_weights(weights: pd.DataFrame, ret: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = weights.merge(ret[["symbol_norm", "year_month", "fwd_ret_1m"]], on=["symbol_norm", "year_month"], how="left")
    merged["matched_label"] = merged["fwd_ret_1m"].notna()
    rows = []
    for ym, g in merged.groupby("year_month", sort=True):
        matched_share = float(g.loc[g["matched_label"], "weight"].sum())
        unmatched_share = float(g.loc[~g["matched_label"], "weight"].sum())
        if matched_share >= 0.98:
            status = "READY"
        elif matched_share >= 0.90:
            status = "WATCH_PARTIAL_MATCH"
        elif matched_share > 0:
            status = "FAIL_LOW_MATCH"
        else:
            status = "FAIL_NO_FORWARD_LABEL"
        rows.append({
            "year_month": ym,
            "selected_count": int(len(g)),
            "matched_symbol_count": int(g["matched_label"].sum()),
            "unmatched_symbol_count": int((~g["matched_label"]).sum()),
            "matched_weight_share": matched_share,
            "unmatched_weight_share": unmatched_share,
            "fwd_ret_available": matched_share > 0,
            "evaluation_month_status": status,
            "caveat": "no portfolio returns calculated",
        })
    monthly = pd.DataFrame(rows)
    unmatched = merged.loc[~merged["matched_label"], ["year_month", "symbol_norm", "weight", "rank", "alpha_signal_route_b_strict_lag", "matched_label"]].copy()
    unmatched["suspected_reason"] = ""
    unmatched["edge_case_flag"] = True
    unmatched["caveat"] = "diagnostic unmatched label detail only"
    return monthly, unmatched


def edge_case_qa(weights: pd.DataFrame, ret: pd.DataFrame, unmatched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    alpha = pd.read_parquet(ROUTE_B_ALPHA, columns=["symbol_norm", "year_month"])
    alpha["symbol_norm"] = norm_symbol(alpha["symbol_norm"])
    alpha["year_month"] = alpha["year_month"].astype(str).str.slice(0, 7)
    adapter = pd.read_parquet(ADAPTER, columns=["symbol_norm", "year_month", "split_group"])
    adapter["symbol_norm"] = norm_symbol(adapter["symbol_norm"])
    adapter["year_month"] = adapter["year_month"].astype(str).str.slice(0, 7)
    all_months = sorted(
        set(ret["year_month"].dropna().astype(str))
        .union(set(weights["year_month"].dropna().astype(str)))
        .union(set(alpha["year_month"].dropna().astype(str)))
    )
    max_weight_month = str(weights["year_month"].max())
    ret_key = set(zip(ret["symbol_norm"].astype(str), ret["year_month"].astype(str)))
    alpha_key = set(zip(alpha["symbol_norm"].astype(str), alpha["year_month"].astype(str)))
    adapter_key = set(zip(adapter["symbol_norm"].astype(str), adapter["year_month"].astype(str)))
    ret_months_by_symbol = ret.groupby("symbol_norm")["year_month"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()
    rows = []
    matched = weights.merge(ret[["symbol_norm", "year_month", "fwd_ret_1m"]], on=["symbol_norm", "year_month"], how="left")
    for r in matched.itertuples(index=False):
        ym = str(r.year_month)
        sym = str(r.symbol_norm)
        w = float(r.weight)
        fwd = getattr(r, "fwd_ret_1m")
        if pd.notna(fwd):
            issue = "EXTREME_RETURN_REVIEW" if abs(float(fwd)) > 1 else "NORMAL_MATCHED"
            severity = "WATCH" if issue == "EXTREME_RETURN_REVIEW" else "INFO"
            reason = "abs(fwd_ret_1m)>1" if issue == "EXTREME_RETURN_REVIEW" else "matched Mretwd fwd_ret_1m"
            evidence = f"fwd_ret_1m={fwd}"
            handling = "review but keep as configured source" if issue == "EXTREME_RETURN_REVIEW" else "include if month is READY"
        elif ym == max_weight_month:
            issue = "EXPECTED_NO_LABEL_FINAL_MONTH"
            severity = "INFO"
            reason = "final construction month has no next-month label"
            evidence = f"year_month={ym}; max_weight_month={max_weight_month}"
            handling = "exclude from primary evaluation"
        else:
            later_months = [m for m in ret_months_by_symbol.get(sym, []) if m > ym]
            next_idx = all_months.index(ym) + 1 if ym in all_months and all_months.index(ym) + 1 < len(all_months) else None
            next_month = all_months[next_idx] if next_idx is not None else ""
            has_next_ret = (sym, next_month) in ret_key if next_month else False
            exists_alpha_next = any((sym, m) in alpha_key for m in all_months if m > ym)
            exists_adapter_next = any((sym, m) in adapter_key for m in all_months if m > ym)
            if not later_months and not exists_alpha_next and not exists_adapter_next:
                issue = "POSSIBLE_DELISTING"
                reason = "symbol disappears from return map, alpha, and adapter after selected month"
            elif not has_next_ret and (exists_alpha_next or exists_adapter_next):
                issue = "POSSIBLE_RETURN_MAP_GAP"
                reason = "next-month return row missing while symbol continues in alpha/adapter"
            elif not has_next_ret:
                issue = "POSSIBLE_NO_TRADE_MONTH"
                reason = "next-month TRD_Mnth row missing"
            else:
                issue = "INCONCLUSIVE"
                reason = "missing fwd_ret_1m despite partial evidence"
            severity = "HIGH" if issue in {"POSSIBLE_RETURN_MAP_GAP", "POSSIBLE_SYMBOL_MAPPING_BREAK"} else "WATCH"
            evidence = f"next_month={next_month}; has_next_ret={has_next_ret}; later_ret_month_count={len(later_months)}; alpha_next={exists_alpha_next}; adapter_next={exists_adapter_next}"
            handling = "block if material non-final weight share; otherwise sensitivity/exclusion policy"
        rows.append({
            "year_month": ym,
            "symbol_norm": sym,
            "weight": w,
            "issue_type": issue,
            "suspected_reason": reason,
            "evidence": evidence,
            "severity": severity,
            "recommended_eval_handling": handling,
            "caveat": "no raw TRD_Mnth status fields parsed; inferred from repaired return map and alpha/adapter continuity",
        })
    detail = pd.DataFrame(rows)
    non_final_missing = detail.loc[(detail["issue_type"] != "NORMAL_MATCHED") & (detail["issue_type"] != "EXTREME_RETURN_REVIEW") & (detail["issue_type"] != "EXPECTED_NO_LABEL_FINAL_MONTH")]
    final_missing = detail.loc[detail["issue_type"].eq("EXPECTED_NO_LABEL_FINAL_MONTH")]
    summary_rows = []
    for issue, g in detail.groupby("issue_type", sort=True):
        by_month = g.groupby("year_month")["weight"].sum()
        summary_rows.append({
            "issue_type": issue,
            "row_count": int(len(g)),
            "weight_share_sum": float(g["weight"].sum()),
            "affected_month_count": int(g["year_month"].nunique()),
            "max_month_weight_share": float(by_month.max()) if len(by_month) else 0.0,
            "severity": str(g["severity"].iloc[0]),
            "recommended_action": str(g["recommended_eval_handling"].iloc[0]),
        })
    non_final_weight = float(non_final_missing["weight"].sum())
    final_weight = float(final_missing["weight"].sum())
    possible_delisting_count = int((detail["issue_type"] == "POSSIBLE_DELISTING").sum())
    possible_suspension_count = int((detail["issue_type"] == "POSSIBLE_SUSPENSION").sum())
    possible_mapping_count = int((detail["issue_type"] == "POSSIBLE_SYMBOL_MAPPING_BREAK").sum())
    extreme_count = int((detail["issue_type"] == "EXTREME_RETURN_REVIEW").sum())
    if non_final_weight >= 0.01 or possible_mapping_count > 0:
        status = "BLOCK_EVAL_PENDING_LABEL_FIX"
    elif non_final_weight > 0 or final_weight > 0 or extreme_count > 0:
        status = "PASS_WITH_CAVEATS"
    else:
        status = "PASS"
    summary = pd.DataFrame(summary_rows)
    metrics = {
        "non_final_month_missing_label_count": int(len(non_final_missing)),
        "non_final_month_missing_label_weight_share": non_final_weight,
        "final_month_missing_label_count": int(len(final_missing)),
        "final_month_missing_label_weight_share": final_weight,
        "possible_delisting_count": possible_delisting_count,
        "possible_suspension_count": possible_suspension_count,
        "possible_symbol_mapping_break_count": possible_mapping_count,
        "extreme_return_review_count": extreme_count,
        "edge_case_qa_status": status,
    }
    del alpha, adapter, matched
    gc.collect()
    return detail, summary, metrics


def window_policy(monthly_match: pd.DataFrame, edge_metrics: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for r in monthly_match.itertuples(index=False):
        status = str(r.evaluation_month_status)
        if status == "READY":
            primary = True
            sensitivity = True
            reason = ""
        elif status == "WATCH_PARTIAL_MATCH":
            primary = False
            sensitivity = True
            reason = "partial label match"
        else:
            primary = False
            sensitivity = False
            reason = status
        rows.append({
            "year_month": r.year_month,
            "selected_count": r.selected_count,
            "matched_weight_share": r.matched_weight_share,
            "evaluation_month_status": status,
            "edge_case_status": edge_metrics["edge_case_qa_status"],
            "include_in_primary_eval": primary,
            "include_in_sensitivity_eval": sensitivity,
            "exclusion_reason": reason,
            "caveat": "primary evaluation excludes WATCH/FAIL months",
        })
    policy = pd.DataFrame(rows)
    primary = policy.loc[policy["include_in_primary_eval"]]
    excluded = policy.loc[~policy["include_in_primary_eval"]]
    summary = pd.DataFrame([{
        "primary_eval_month_count": int(len(primary)),
        "primary_eval_min_year_month": str(primary["year_month"].min()) if len(primary) else "",
        "primary_eval_max_year_month": str(primary["year_month"].max()) if len(primary) else "",
        "excluded_month_count": int(len(excluded)),
        "excluded_months": ",".join(excluded["year_month"].astype(str).tolist()),
        "ready_month_count": int((policy["evaluation_month_status"] == "READY").sum()),
        "watch_partial_match_month_count": int((policy["evaluation_month_status"] == "WATCH_PARTIAL_MATCH").sum()),
        "fail_no_forward_label_month_count": int((policy["evaluation_month_status"] == "FAIL_NO_FORWARD_LABEL").sum()),
        "fail_low_match_month_count": int((policy["evaluation_month_status"] == "FAIL_LOW_MATCH").sum()),
        "edge_case_qa_status": edge_metrics["edge_case_qa_status"],
        "non_final_month_missing_label_count": edge_metrics["non_final_month_missing_label_count"],
        "non_final_month_missing_label_weight_share": edge_metrics["non_final_month_missing_label_weight_share"],
    }])
    return policy, summary


def cost_return_config() -> dict[str, Any]:
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
    rows = [
        ("first_construction_month", "2017-02", "", "Route B first construction month", ""),
        ("first_month_turnover_proxy", "1.0", "", "initial holdings are full portfolio initialization", ""),
        ("first_month_cost_policy", "charge_cost_on_first_month_initialization", "no_cost_on_first_month_initialization", "primary is conservative", "cost not calculated in prep"),
    ]
    return pd.DataFrame(rows, columns=["policy_item", "selected_policy", "alternative_policy", "reason", "caveat"])


def next_run_config(eval_allowed: bool) -> dict[str, Any]:
    return {
        "recommended_next_run": "V0 Legacy-Compatible PIT Strict-Lag Replay Evaluation Run v0",
        "recommended_next_run_reason": "Run only after edge-case QA accepts the evaluation window; primary evaluation excludes no-label months.",
        "weights_path": rel(WEIGHTS),
        "return_map_path": rel(RETURN_MAP),
        "eval_window_policy_path": rel(OUT_DIR / "v0_route_b_eval_window_policy.csv"),
        "edge_case_qa_path": rel(OUT_DIR / "v0_route_b_label_edge_case_qa.csv"),
        "cost_return_variant_config_path": rel(OUT_DIR / "v0_route_b_eval_cost_return_variant_config.json"),
        "turnover_policy_path": rel(OUT_DIR / "v0_route_b_eval_turnover_policy.csv"),
        "calculate_portfolio_returns_next_run_allowed": eval_allowed,
        "calculate_cumulative_returns_next_run_allowed": eval_allowed,
        "calculate_cost_scenarios_next_run_allowed": eval_allowed,
        "calculate_sharpe_next_run_allowed": eval_allowed,
        "calculate_maxdd_next_run_allowed": eval_allowed,
        "benchmark_relative_next_run_allowed": False,
        "alpha_beta_next_run_allowed": False,
        "ir_te_next_run_allowed": False,
        "ff_next_run_allowed": False,
        "dgtw_next_run_allowed": False,
        "production_allowed": False,
    }


def guardrails() -> pd.DataFrame:
    actuals = {
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "transaction_cost_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "tstat_calculated": False,
        "benchmark_relative_returns_calculated": False,
        "active_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "ir_te_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "ml_training_run": False,
        "tuning_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "old_artifacts_modified": False,
        "route_b_weights_modified": False,
        "repaired_return_map_modified": False,
    }
    return pd.DataFrame([{"guardrail": k, "expected": False, "actual": v, "pass": v is False} for k, v in actuals.items()])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", "prerequisite_check")
    prereq = prereq_check()
    write_json(OUT_DIR / "v0_route_b_eval_prep_prerequisite_check.json", prereq)
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", "input_and_return_source_qa")
    weights = load_weights()
    wqa = weights_input_qa(weights)
    wqa.to_csv(OUT_DIR / "v0_route_b_eval_weights_input_qa.csv", index=False, encoding="utf-8-sig")
    ret = load_return_map()
    rqa = return_source_qa(ret)
    rqa.to_csv(OUT_DIR / "v0_route_b_eval_return_source_qa.csv", index=False, encoding="utf-8-sig")

    write_state("running", "match_and_edge_case_qa")
    monthly_match, unmatched = match_weights(weights, ret)
    monthly_match.to_csv(OUT_DIR / "v0_route_b_eval_return_match_monthly_qa.csv", index=False, encoding="utf-8-sig")
    edge_detail, edge_summary, edge_metrics = edge_case_qa(weights, ret, unmatched)
    # Fill unmatched detail after edge classification.
    edge_lookup = edge_detail.loc[edge_detail["issue_type"].ne("NORMAL_MATCHED"), ["year_month", "symbol_norm", "issue_type", "suspected_reason"]]
    unmatched = unmatched.merge(edge_lookup, on=["year_month", "symbol_norm"], how="left")
    unmatched["suspected_reason"] = unmatched["suspected_reason_y"].fillna(unmatched["suspected_reason_x"])
    unmatched = unmatched.drop(columns=["suspected_reason_x", "suspected_reason_y"])
    unmatched["edge_case_flag"] = True
    unmatched.to_csv(OUT_DIR / "v0_route_b_unmatched_label_detail.csv", index=False, encoding="utf-8-sig")
    edge_detail.to_csv(OUT_DIR / "v0_route_b_label_edge_case_qa.csv", index=False, encoding="utf-8-sig")
    extra_rows = pd.DataFrame([{"issue_type": k, "row_count": v, "weight_share_sum": "", "affected_month_count": "", "max_month_weight_share": "", "severity": "", "recommended_action": ""} for k, v in edge_metrics.items()])
    pd.concat([edge_summary, extra_rows], ignore_index=True).to_csv(OUT_DIR / "v0_route_b_label_edge_case_summary.csv", index=False, encoding="utf-8-sig")

    write_state("running", "window_and_config")
    window, window_summary = window_policy(monthly_match, edge_metrics)
    window.to_csv(OUT_DIR / "v0_route_b_eval_window_policy.csv", index=False, encoding="utf-8-sig")
    window_summary.to_csv(OUT_DIR / "v0_route_b_eval_window_summary.csv", index=False, encoding="utf-8-sig")
    cfg = cost_return_config()
    write_json(OUT_DIR / "v0_route_b_eval_cost_return_variant_config.json", cfg)
    tpol = turnover_policy()
    tpol.to_csv(OUT_DIR / "v0_route_b_eval_turnover_policy.csv", index=False, encoding="utf-8-sig")
    guard = guardrails()
    guard.to_csv(OUT_DIR / "v0_route_b_eval_prep_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    guard_pass = bool(guard["pass"].all())

    ws = window_summary.iloc[0].to_dict()
    eval_allowed = (
        edge_metrics["edge_case_qa_status"] in {"PASS", "PASS_WITH_CAVEATS"}
        and edge_metrics["non_final_month_missing_label_weight_share"] < 0.01
        and int(ws["primary_eval_month_count"]) > 0
        and guard_pass
    )
    write_json(OUT_DIR / "v0_route_b_eval_run_config_draft.json", next_run_config(eval_allowed))

    if not guard_pass:
        final_decision = "ROUTE_B_EVAL_PREP_FAIL_GUARDRAIL"
    elif float(monthly_match["matched_weight_share"].max()) <= 0:
        final_decision = "ROUTE_B_EVAL_PREP_BLOCKED_BY_RETURN_MATCH"
    elif edge_metrics["edge_case_qa_status"] == "BLOCK_EVAL_PENDING_LABEL_FIX":
        final_decision = "ROUTE_B_EVAL_PREP_BLOCKED_BY_LABEL_EDGE_CASES"
    elif edge_metrics["edge_case_qa_status"] == "PASS":
        final_decision = "ROUTE_B_EVAL_PREP_READY_FOR_EVAL_RUN"
    else:
        final_decision = "ROUTE_B_EVAL_PREP_READY_WITH_LABEL_CAVEATS"

    if final_decision == "ROUTE_B_EVAL_PREP_BLOCKED_BY_LABEL_EDGE_CASES":
        recommended_next_step = "先运行 label edge-case 修复/复核任务，定位非最终月份 missing fwd_ret_1m 的 return-map gap / no-trade case；修复或确认处理规则后再进入 evaluation run。"
    elif final_decision == "ROUTE_B_EVAL_PREP_BLOCKED_BY_RETURN_MATCH":
        recommended_next_step = "先修复 weights × repaired TRD_Mnth 的 label matching，再进入 evaluation run。"
    elif final_decision == "ROUTE_B_EVAL_PREP_FAIL_GUARDRAIL":
        recommended_next_step = "停止，先修复 guardrail violation。"
    else:
        recommended_next_step = "运行 V0 Legacy-Compatible PIT Strict-Lag Replay Evaluation Run v0；primary window 排除 no-label 月份，且保留 label caveat。"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "weights_path": rel(WEIGHTS),
        "return_map_path": rel(RETURN_MAP),
        "primary_return_field": PRIMARY_RETURN_FIELD,
        "weights_loaded": len(weights) > 0,
        "return_map_loaded": len(ret) > 0,
        "weights_month_count": int(weights["year_month"].nunique()),
        "weights_min_year_month": str(weights["year_month"].min()),
        "weights_max_year_month": str(weights["year_month"].max()),
        "avg_matched_weight_share": float(monthly_match["matched_weight_share"].mean()),
        "min_matched_weight_share": float(monthly_match["matched_weight_share"].min()),
        "ready_eval_month_count": int((monthly_match["evaluation_month_status"] == "READY").sum()),
        "watch_partial_match_month_count": int((monthly_match["evaluation_month_status"] == "WATCH_PARTIAL_MATCH").sum()),
        "fail_no_forward_label_month_count": int((monthly_match["evaluation_month_status"] == "FAIL_NO_FORWARD_LABEL").sum()),
        "fail_low_match_month_count": int((monthly_match["evaluation_month_status"] == "FAIL_LOW_MATCH").sum()),
        "expected_no_label_months": monthly_match.loc[monthly_match["evaluation_month_status"].eq("FAIL_NO_FORWARD_LABEL"), "year_month"].astype(str).tolist(),
        "edge_case_qa_status": edge_metrics["edge_case_qa_status"],
        "non_final_month_missing_label_count": edge_metrics["non_final_month_missing_label_count"],
        "non_final_month_missing_label_weight_share": edge_metrics["non_final_month_missing_label_weight_share"],
        "possible_delisting_count": edge_metrics["possible_delisting_count"],
        "possible_suspension_count": edge_metrics["possible_suspension_count"],
        "possible_symbol_mapping_break_count": edge_metrics["possible_symbol_mapping_break_count"],
        "extreme_return_review_count": edge_metrics["extreme_return_review_count"],
        "primary_eval_month_count": int(ws["primary_eval_month_count"]),
        "primary_eval_min_year_month": str(ws["primary_eval_min_year_month"]),
        "primary_eval_max_year_month": str(ws["primary_eval_max_year_month"]),
        "excluded_months": str(ws["excluded_months"]).split(",") if str(ws["excluded_months"]) else [],
        "primary_cost_bps": PRIMARY_COST_BPS,
        "primary_return_variant": PRIMARY_RETURN_VARIANT,
        "first_month_initialization_turnover_policy": "charge_cost_on_first_month_initialization",
        "evaluation_allowed_next_run": eval_allowed,
        "calculate_portfolio_returns_next_run_allowed": eval_allowed,
        "calculate_benchmark_relative_next_run_allowed": False,
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "transaction_cost_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "tstat_calculated": False,
        "benchmark_relative_returns_calculated": False,
        "active_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "ir_te_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "ml_training_run": False,
        "tuning_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "old_artifacts_modified": False,
        "route_b_weights_modified": False,
        "repaired_return_map_modified": False,
        "guardrails_passed": guard_pass,
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    write_json(OUT_DIR / "v0_legacy_compatible_pit_strict_lag_replay_eval_prep_summary.json", summary)
    report = "\n".join([
        "# V0 Legacy-Compatible PIT Strict-Lag Replay Evaluation Prep v0",
        "",
        f"- final_decision: {final_decision}",
        f"- avg_matched_weight_share: {summary['avg_matched_weight_share']:.6f}",
        f"- min_matched_weight_share: {summary['min_matched_weight_share']:.6f}",
        f"- edge_case_qa_status: {edge_metrics['edge_case_qa_status']}",
        f"- primary_eval_window: {summary['primary_eval_min_year_month']} to {summary['primary_eval_max_year_month']} ({summary['primary_eval_month_count']} months)",
        f"- excluded_months: {','.join(summary['excluded_months'])}",
        "",
        "本任务只做 evaluation prep、label matching 和 edge-case QA；未计算 portfolio returns、成本或绩效指标。",
    ])
    (OUT_DIR / "v0_legacy_compatible_pit_strict_lag_replay_eval_prep_report.md").write_text(report, encoding="utf-8")
    final_qa = pd.DataFrame([
        {"check_name": "prerequisites_passed", "expected": True, "actual": prereq["prerequisites_passed"], "pass": prereq["prerequisites_passed"], "caveat": ""},
        {"check_name": "matching_available", "expected": True, "actual": monthly_match["matched_weight_share"].max() > 0, "pass": monthly_match["matched_weight_share"].max() > 0, "caveat": ""},
        {"check_name": "edge_case_status_allowed", "expected": "PASS or PASS_WITH_CAVEATS", "actual": edge_metrics["edge_case_qa_status"], "pass": edge_metrics["edge_case_qa_status"] in {"PASS", "PASS_WITH_CAVEATS"}, "caveat": ""},
        {"check_name": "guardrails_passed", "expected": True, "actual": guard_pass, "pass": guard_pass, "caveat": ""},
    ])
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    write_json(OUT_DIR / "terminal_summary.json", {
        "task_name": TASK_NAME,
        "status": "completed",
        "stdout_log": rel(RUN_DIR / "run_stdout.txt"),
        "stderr_log": rel(RUN_DIR / "run_stderr.txt"),
        "output_dir": rel(OUT_DIR),
        "final_decision": final_decision,
    })
    (OUT_DIR / "task_completion_card.md").write_text(
        "\n".join(["# task_completion_card", "", f"- task_name: {TASK_NAME}", "- status: completed", f"- final_decision: {final_decision}", f"- output_dir: {rel(OUT_DIR)}"]),
        encoding="utf-8",
    )
    write_state("completed", "all_outputs_written", {"final_decision": final_decision, "output_dir": rel(OUT_DIR)})
    del weights, ret, wqa, rqa, monthly_match, unmatched, edge_detail, edge_summary, window, window_summary, guard
    gc.collect()
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": rel(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
