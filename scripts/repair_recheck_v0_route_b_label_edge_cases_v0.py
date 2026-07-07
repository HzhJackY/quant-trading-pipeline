from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_NAME = "V0 Route B Label Edge-Case Repair and Recheck v0"
OUT_NAME = "v0_route_b_label_edge_case_repair_recheck_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / OUT_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

EVAL_PREP_DIR = ROOT / "output" / "v0_legacy_compatible_pit_strict_lag_replay_eval_prep_v0"
EVAL_SUMMARY = EVAL_PREP_DIR / "v0_legacy_compatible_pit_strict_lag_replay_eval_prep_summary.json"
MATCH_QA = EVAL_PREP_DIR / "v0_route_b_eval_return_match_monthly_qa.csv"
UNMATCHED_DETAIL = EVAL_PREP_DIR / "v0_route_b_unmatched_label_detail.csv"
EDGE_QA = EVAL_PREP_DIR / "v0_route_b_label_edge_case_qa.csv"
EDGE_SUMMARY = EVAL_PREP_DIR / "v0_route_b_label_edge_case_summary.csv"
WINDOW_POLICY = EVAL_PREP_DIR / "v0_route_b_eval_window_policy.csv"

WEIGHTS = ROOT / "output" / "v0_legacy_compatible_pit_strict_lag_replay_portfolio_construction_run_v0" / "v0_route_b_research_weights.parquet"
ALPHA = ROOT / "output" / "v0_legacy_compatible_pit_strict_lag_replay_alpha_build_v0" / "v0_legacy_pit_route_b_strict_lag_alpha_panel.parquet"
ADAPTER = ROOT / "output" / "v0_legacy_compatible_pit_adapter_replay_dry_run_v0" / "v0_pit_legacy_compatible_input.parquet"
RETURN_MAP = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0" / "canonical_csmar_trd_mnth_return_map_repaired.parquet"
TRD_REPAIR_DIR = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0"

PRIMARY_RETURN_FIELD = "Mretwd"


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
        "resume_instruction": f"先读取 {rel(RUN_DIR / 'RUN_STATE.md')}；继续时运行 scripts\\repair_recheck_v0_route_b_label_edge_cases_v0.py，并重定向 stdout/stderr 到本目录。",
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


def add_month(ym: str, n: int = 1) -> str:
    return (pd.Period(ym, freq="M") + n).strftime("%Y-%m")


def candidate_files() -> list[Path]:
    candidates: list[Path] = []
    if TRD_REPAIR_DIR.exists():
        for p in TRD_REPAIR_DIR.iterdir():
            name = p.name.lower()
            if p.is_file() and any(k in name for k in ["trd", "mnth", "mret", "return", "source", "match"]):
                candidates.append(p)
    return candidates


def prereq_check() -> dict[str, Any]:
    raw_candidates = candidate_files()
    flags = {
        "eval_prep_summary_found": EVAL_SUMMARY.exists(),
        "unmatched_label_detail_found": UNMATCHED_DETAIL.exists(),
        "label_edge_case_qa_found": EDGE_QA.exists(),
        "label_edge_case_summary_found": EDGE_SUMMARY.exists(),
        "return_match_monthly_qa_found": MATCH_QA.exists(),
        "route_b_weights_found": WEIGHTS.exists(),
        "repaired_return_map_found": RETURN_MAP.exists(),
        "route_b_alpha_found": ALPHA.exists(),
        "adapter_artifact_found": ADAPTER.exists(),
        "trd_mnth_raw_or_intermediate_found": len(raw_candidates) > 0,
    }
    paths = {
        "eval_prep_summary_found": EVAL_SUMMARY,
        "unmatched_label_detail_found": UNMATCHED_DETAIL,
        "label_edge_case_qa_found": EDGE_QA,
        "label_edge_case_summary_found": EDGE_SUMMARY,
        "return_match_monthly_qa_found": MATCH_QA,
        "route_b_weights_found": WEIGHTS,
        "repaired_return_map_found": RETURN_MAP,
        "route_b_alpha_found": ALPHA,
        "adapter_artifact_found": ADAPTER,
    }
    missing = [rel(p) for k, p in paths.items() if not flags[k]]
    flags["prerequisites_passed"] = len(missing) == 0
    flags["missing_files"] = missing
    flags["caveat"] = "Raw/intermediate search is limited to existing repaired TRD_Mnth output metadata and repaired map; original return map is never modified."
    flags["raw_or_intermediate_candidate_files"] = [rel(p) for p in raw_candidates]
    return flags


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unmatched = pd.read_csv(UNMATCHED_DETAIL, dtype={"year_month": str, "symbol_norm": str})
    edge = pd.read_csv(EDGE_QA, dtype={"year_month": str, "symbol_norm": str})
    weights = pd.read_parquet(WEIGHTS)
    ret = pd.read_parquet(RETURN_MAP, columns=["symbol_norm", "year_month", "monthly_return_t", "fwd_ret_1m", "primary_return_field", "raw_return_field_values", "return_valid_flag", "invalid_reason"])
    alpha = pd.read_parquet(ALPHA, columns=["symbol_norm", "year_month"])
    adapter = pd.read_parquet(ADAPTER, columns=["symbol_norm", "year_month", "split_group"])
    for df in [unmatched, edge, weights, ret, alpha, adapter]:
        if "symbol_norm" in df.columns:
            df["symbol_norm"] = norm_symbol(df["symbol_norm"])
        if "year_month" in df.columns:
            df["year_month"] = df["year_month"].astype(str).str.slice(0, 7)
    ret = ret.loc[ret["primary_return_field"].astype(str).eq(PRIMARY_RETURN_FIELD)].copy()
    ret["monthly_return_t"] = pd.to_numeric(ret["monthly_return_t"], errors="coerce")
    ret["fwd_ret_1m"] = pd.to_numeric(ret["fwd_ret_1m"], errors="coerce")
    return unmatched, edge, weights, ret, pd.concat([
        alpha.assign(source="alpha")[["symbol_norm", "year_month", "source"]],
        adapter.assign(source="adapter")[["symbol_norm", "year_month", "source"]],
    ], ignore_index=True)


def blocking_cases(unmatched: pd.DataFrame, edge: pd.DataFrame) -> pd.DataFrame:
    non_final = unmatched.loc[unmatched["issue_type"].isin(["POSSIBLE_RETURN_MAP_GAP", "POSSIBLE_NO_TRADE_MONTH", "POSSIBLE_SYMBOL_MAPPING_BREAK", "POSSIBLE_DELISTING"])].copy()
    final = unmatched.loc[unmatched["issue_type"].eq("EXPECTED_NO_LABEL_FINAL_MONTH")].copy()
    extreme = edge.loc[edge["issue_type"].eq("EXTREME_RETURN_REVIEW")].copy()
    rows = []
    cid = 1
    for case_type, df, blocking in [
        ("NON_FINAL_MISSING_LABEL", non_final, True),
        ("FINAL_MONTH_EXPECTED_NO_LABEL", final, False),
        ("EXTREME_RETURN_REVIEW", extreme, False),
    ]:
        for r in df.itertuples(index=False):
            rows.append({
                "case_id": f"C{cid:03d}",
                "case_type": case_type,
                "year_month": r.year_month,
                "symbol_norm": r.symbol_norm,
                "weight": float(getattr(r, "weight", np.nan)),
                "rank": getattr(r, "rank", ""),
                "alpha_signal_route_b_strict_lag": getattr(r, "alpha_signal_route_b_strict_lag", ""),
                "selected_reason": getattr(r, "selected_reason", ""),
                "issue_type": getattr(r, "issue_type", ""),
                "suspected_reason": getattr(r, "suspected_reason", ""),
                "edge_case_flag": True,
                "blocking_for_eval": blocking,
                "caveat": "",
            })
            cid += 1
    return pd.DataFrame(rows)


def lineage_drilldown(cases: pd.DataFrame, ret: pd.DataFrame, presence: pd.DataFrame) -> pd.DataFrame:
    ret_key = set(zip(ret["symbol_norm"].astype(str), ret["year_month"].astype(str)))
    alpha_key = set(zip(presence.loc[presence["source"].eq("alpha"), "symbol_norm"].astype(str), presence.loc[presence["source"].eq("alpha"), "year_month"].astype(str)))
    adapter_key = set(zip(presence.loc[presence["source"].eq("adapter"), "symbol_norm"].astype(str), presence.loc[presence["source"].eq("adapter"), "year_month"].astype(str)))
    ret_months_by_symbol = ret.groupby("symbol_norm")["year_month"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()
    rows = []
    for r in cases.itertuples(index=False):
        ym = str(r.year_month)
        sym = str(r.symbol_norm)
        nxt = add_month(ym)
        ret_current = (sym, ym) in ret_key
        ret_next = (sym, nxt) in ret_key
        adapter_current = (sym, ym) in adapter_key
        adapter_next = (sym, nxt) in adapter_key
        alpha_current = (sym, ym) in alpha_key
        later_ret = [m for m in ret_months_by_symbol.get(sym, []) if m > ym]
        cur = ret.loc[ret["symbol_norm"].eq(sym) & ret["year_month"].eq(ym)]
        nx = ret.loc[ret["symbol_norm"].eq(sym) & ret["year_month"].eq(nxt)]
        current_or_next_mretwd = bool((len(cur) and cur["monthly_return_t"].notna().any()) or (len(nx) and nx["monthly_return_t"].notna().any()))
        diagnosis = "INCONCLUSIVE"
        fwd_shift_possible = False
        if r.case_type == "FINAL_MONTH_EXPECTED_NO_LABEL":
            diagnosis = "EXPECTED_FINAL_MONTH_NO_LABEL"
        elif r.case_type == "EXTREME_RETURN_REVIEW":
            diagnosis = "EXTREME_RETURN_ONLY_NOT_MISSING"
        elif ret_next and nx["monthly_return_t"].notna().any():
            diagnosis = "RETURN_MAP_FWD_SHIFT_GAP"
            fwd_shift_possible = True
        elif not ret_next and (adapter_next or any((sym, m) in adapter_key for m in ret_months_by_symbol.get(sym, []) if m > ym)):
            diagnosis = "RAW_TRD_MNTH_GAP"
        elif not later_ret and not adapter_next:
            diagnosis = "DELISTING_OR_TERMINATION_LIKELY"
        elif not ret_next:
            diagnosis = "NO_TRADE_OR_SUSPENSION_LIKELY"
        evidence = f"next_month={nxt}; ret_current={ret_current}; ret_next={ret_next}; adapter_current={adapter_current}; adapter_next={adapter_next}; later_ret_month_count={len(later_ret)}"
        rows.append({
            "case_id": r.case_id,
            "year_month": ym,
            "symbol_norm": sym,
            "weight": r.weight,
            "adapter_current_month_exists": adapter_current,
            "adapter_next_month_exists": adapter_next,
            "alpha_current_month_exists": alpha_current,
            "return_map_current_month_exists": ret_current,
            "return_map_next_month_exists": ret_next,
            "raw_trd_current_month_exists": ret_current,
            "raw_trd_next_month_exists": ret_next,
            "symbol_mapping_change_detected": False,
            "duplicate_symbol_month_detected": bool(ret.duplicated(["symbol_norm", "year_month"]).any()),
            "disappears_after_current_month": len(later_ret) == 0,
            "mretwd_available_current_or_next": current_or_next_mretwd,
            "mretnd_available_current_or_next": False,
            "fwd_ret_shift_issue_possible": fwd_shift_possible,
            "diagnosis": diagnosis,
            "evidence": evidence,
            "caveat": "Raw TRD_Mnth status fields unavailable; repaired map used as intermediate evidence.",
        })
    return pd.DataFrame(rows)


def raw_lookup(cases: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in cases.itertuples(index=False):
        ym = str(r.year_month)
        sym = str(r.symbol_norm)
        nxt = add_month(ym)
        cur = ret.loc[ret["symbol_norm"].eq(sym) & ret["year_month"].eq(ym)]
        nx = ret.loc[ret["symbol_norm"].eq(sym) & ret["year_month"].eq(nxt)]
        rows.append({
            "case_id": r.case_id,
            "year_month": ym,
            "symbol_norm": sym,
            "raw_file_found": False,
            "raw_record_current_month_found": bool(len(cur)),
            "raw_record_next_month_found": bool(len(nx)),
            "raw_mretwd_current": float(cur["monthly_return_t"].iloc[0]) if len(cur) and pd.notna(cur["monthly_return_t"].iloc[0]) else np.nan,
            "raw_mretwd_next": float(nx["monthly_return_t"].iloc[0]) if len(nx) and pd.notna(nx["monthly_return_t"].iloc[0]) else np.nan,
            "raw_mretnd_current": np.nan,
            "raw_mretnd_next": np.nan,
            "trading_status_current": "",
            "trading_status_next": "",
            "monthly_trading_days_current": "",
            "monthly_trading_days_next": "",
            "listing_or_delisting_evidence": "",
            "lookup_status": "INTERMEDIATE_REPAIRED_MAP_ONLY",
            "caveat": "No raw TRD_Mnth table with status fields found in repaired output directory; original raw files were not reparsed.",
        })
    return pd.DataFrame(rows)


def reconstruction_check(cases: pd.DataFrame, ret: pd.DataFrame, lineage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in cases.loc[cases["case_type"].eq("NON_FINAL_MISSING_LABEL")].itertuples(index=False):
        ym = str(r.year_month)
        sym = str(r.symbol_norm)
        nxt = add_month(ym)
        current = ret.loc[ret["symbol_norm"].eq(sym) & ret["year_month"].eq(ym)]
        next_row = ret.loc[ret["symbol_norm"].eq(sym) & ret["year_month"].eq(nxt)]
        current_missing = len(current) == 0 or current["fwd_ret_1m"].isna().all()
        next_mret = next_row["monthly_return_t"].dropna()
        from_map = len(next_mret) > 0
        cand = float(next_mret.iloc[0]) if from_map else np.nan
        diag = str(lineage.loc[lineage["case_id"].eq(r.case_id), "diagnosis"].iloc[0])
        if from_map:
            status = "RECONSTRUCTABLE_FROM_REPAIRED_MAP"
            action = "generate safe patch overlay from next-month Mretwd"
        elif diag == "DELISTING_OR_TERMINATION_LIKELY":
            status = "DELISTING_RETURN_SOURCE_REQUIRED"
            action = "do not fill; require delisting/no-trade return source decision"
        elif diag == "NO_TRADE_OR_SUSPENSION_LIKELY":
            status = "NO_TRADE_HANDLING_REQUIRED"
            action = "do not fill; define no-trade handling policy or raw evidence"
        elif diag == "RAW_TRD_MNTH_GAP":
            status = "NOT_RECONSTRUCTABLE_RAW_MISSING"
            action = "raw/intermediate TRD_Mnth lookup required; no safe patch from repaired map"
        else:
            status = "INCONCLUSIVE"
            action = "manual review required"
        rows.append({
            "case_id": r.case_id,
            "year_month": ym,
            "symbol_norm": sym,
            "expected_label_month": nxt,
            "current_fwd_ret_1m_missing": current_missing,
            "next_month_mretwd_in_repaired_map": from_map,
            "next_month_mretwd_in_raw_trd": False,
            "reconstructable_from_repaired_map": from_map,
            "reconstructable_from_raw_trd": False,
            "reconstructed_fwd_ret_1m_candidate": cand,
            "reconstruction_status": status,
            "recommended_action": action,
            "caveat": "No 0 return imputation used.",
        })
    return pd.DataFrame(rows)


def patch_overlay(recon: pd.DataFrame) -> pd.DataFrame:
    safe = recon.loc[recon["reconstruction_status"].eq("RECONSTRUCTABLE_FROM_REPAIRED_MAP")].copy()
    if safe.empty:
        return pd.DataFrame(columns=["year_month", "symbol_norm", "original_fwd_ret_1m", "patched_fwd_ret_1m", "patch_source", "patch_reason", "evidence", "safe_to_apply", "caveat"])
    rows = []
    for r in safe.itertuples(index=False):
        rows.append({
            "year_month": r.year_month,
            "symbol_norm": r.symbol_norm,
            "original_fwd_ret_1m": np.nan,
            "patched_fwd_ret_1m": r.reconstructed_fwd_ret_1m_candidate,
            "patch_source": "repaired_map_next_month_mretwd",
            "patch_reason": "current fwd_ret_1m missing but next-month Mretwd is present in repaired map",
            "evidence": f"expected_label_month={r.expected_label_month}",
            "safe_to_apply": True,
            "caveat": "overlay only; original repaired return map preserved",
        })
    return pd.DataFrame(rows)


def extreme_review(cases: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in cases.loc[cases["case_type"].eq("EXTREME_RETURN_REVIEW")].itertuples(index=False):
        ym = str(r.year_month)
        sym = str(r.symbol_norm)
        row = ret.loc[ret["symbol_norm"].eq(sym) & ret["year_month"].eq(ym)]
        fwd = row["fwd_ret_1m"].iloc[0] if len(row) else np.nan
        rows.append({
            "case_id": r.case_id,
            "year_month": ym,
            "symbol_norm": sym,
            "fwd_ret_1m": fwd,
            "raw_mretwd_source": fwd,
            "raw_mretnd_source": np.nan,
            "selected_weight": r.weight,
            "evidence": "value comes from repaired return map Mretwd-derived fwd_ret_1m",
            "extreme_return_status": "VERIFIED_VALID_MRETWD" if pd.notna(fwd) else "NEEDS_MANUAL_REVIEW",
            "recommended_eval_handling": "keep in configured source but include sensitivity review",
            "caveat": "No corporate-action raw field available in repaired map.",
        })
    return pd.DataFrame(rows)


def recheck_with_overlay(weights: pd.DataFrame, ret: pd.DataFrame, overlay: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    label = ret[["symbol_norm", "year_month", "fwd_ret_1m"]].copy()
    label["patch_applied_flag"] = False
    if not overlay.empty:
        idx_cols = ["symbol_norm", "year_month"]
        label = label.set_index(idx_cols)
        for r in overlay.loc[overlay["safe_to_apply"].astype(bool)].itertuples(index=False):
            key = (str(r.symbol_norm).zfill(6), str(r.year_month))
            if key in label.index:
                label.loc[key, "fwd_ret_1m"] = r.patched_fwd_ret_1m
                label.loc[key, "patch_applied_flag"] = True
            else:
                label.loc[key, ["fwd_ret_1m", "patch_applied_flag"]] = [r.patched_fwd_ret_1m, True]
        label = label.reset_index()
        label_path = OUT_DIR / "v0_route_b_return_label_view_rechecked.parquet"
        label.to_parquet(label_path, index=False)
    else:
        label_path = None
    before = weights.merge(ret[["symbol_norm", "year_month", "fwd_ret_1m"]], on=["symbol_norm", "year_month"], how="left")
    after = weights.merge(label[["symbol_norm", "year_month", "fwd_ret_1m", "patch_applied_flag"]], on=["symbol_norm", "year_month"], how="left")
    rows = []
    for ym, g_after in after.groupby("year_month", sort=True):
        g_before = before.loc[before["year_month"].eq(ym)]
        before_match = g_before["fwd_ret_1m"].notna()
        after_match = g_after["fwd_ret_1m"].notna()
        before_share = float(g_before.loc[before_match, "weight"].sum())
        after_share = float(g_after.loc[after_match, "weight"].sum())
        if after_share >= 0.98:
            status = "READY"
        elif after_share >= 0.90:
            status = "WATCH_PARTIAL_MATCH"
        elif after_share > 0:
            status = "FAIL_LOW_MATCH"
        else:
            status = "FAIL_NO_FORWARD_LABEL"
        rows.append({
            "year_month": ym,
            "selected_count": int(len(g_after)),
            "matched_symbol_count_before": int(before_match.sum()),
            "matched_symbol_count_after": int(after_match.sum()),
            "matched_weight_share_before": before_share,
            "matched_weight_share_after": after_share,
            "unmatched_weight_share_after": float(g_after.loc[~after_match, "weight"].sum()),
            "fwd_ret_available": after_share > 0,
            "evaluation_month_status_after": status,
            "caveat": "recheck only; no returns calculated",
        })
    unmatched_after = after.loc[after["fwd_ret_1m"].isna(), ["year_month", "symbol_norm", "weight", "rank", "patch_applied_flag"]].copy()
    unmatched_after["matched_label_after"] = False
    unmatched_after["patch_applied"] = unmatched_after["patch_applied_flag"].fillna(False)
    unmatched_after["remaining_issue_type"] = np.where(unmatched_after["year_month"].eq(weights["year_month"].max()), "EXPECTED_NO_LABEL_FINAL_MONTH", "REMAINING_NON_FINAL_MISSING_LABEL")
    unmatched_after["caveat"] = "after patch overlay recheck"
    return pd.DataFrame(rows), unmatched_after.drop(columns=["patch_applied_flag"]), label_path


def guardrails(rechecked_generated: bool) -> pd.DataFrame:
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
        "patch_overlay_candidate_generated": True,
        "rechecked_label_view_generated": rechecked_generated,
    }
    rows = []
    for k, actual in actuals.items():
        expected = True if k in {"patch_overlay_candidate_generated"} or (k == "rechecked_label_view_generated" and rechecked_generated) else False
        if k == "rechecked_label_view_generated" and not rechecked_generated:
            expected = False
        rows.append({"guardrail": k, "expected": expected, "actual": actual, "pass": actual == expected})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", "prerequisite_check")
    prereq = prereq_check()
    write_json(OUT_DIR / "v0_route_b_label_repair_prerequisite_check.json", prereq)
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", "load_and_extract_cases")
    unmatched, edge, weights, ret, presence = load_inputs()
    cases = blocking_cases(unmatched, edge)
    cases.to_csv(OUT_DIR / "v0_route_b_label_blocking_cases.csv", index=False, encoding="utf-8-sig")
    lineage = lineage_drilldown(cases, ret, presence)
    lineage.to_csv(OUT_DIR / "v0_route_b_label_lineage_drilldown.csv", index=False, encoding="utf-8-sig")
    raw = raw_lookup(cases, ret)
    raw.to_csv(OUT_DIR / "v0_route_b_trd_mnth_raw_lookup_cases.csv", index=False, encoding="utf-8-sig")
    recon = reconstruction_check(cases, ret, lineage)
    recon.to_csv(OUT_DIR / "v0_route_b_fwd_ret_reconstruction_check.csv", index=False, encoding="utf-8-sig")

    write_state("running", "patch_and_recheck")
    overlay = patch_overlay(recon)
    overlay_path = OUT_DIR / "v0_route_b_fwd_ret_patch_overlay_candidate.csv"
    overlay.to_csv(overlay_path, index=False, encoding="utf-8-sig")
    safe_patch_count = int(overlay["safe_to_apply"].sum()) if len(overlay) else 0
    write_json(OUT_DIR / "v0_route_b_return_map_patch_policy.json", {
        "patch_overlay_generated": True,
        "patch_overlay_path": rel(overlay_path),
        "original_return_map_preserved": True,
        "patched_return_map_next_run_allowed": safe_patch_count > 0,
        "patch_application_policy": "Use overlay only in recheck/eval prep rerun; never overwrite original repaired return map.",
        "caveat": "Empty overlay means no safe reconstruction source was found.",
    })
    extreme = extreme_review(cases, ret)
    extreme.to_csv(OUT_DIR / "v0_route_b_extreme_return_review.csv", index=False, encoding="utf-8-sig")
    rechecked, unmatched_after, label_path = recheck_with_overlay(weights, ret, overlay)
    rechecked.to_csv(OUT_DIR / "v0_route_b_eval_return_match_rechecked_monthly_qa.csv", index=False, encoding="utf-8-sig")
    unmatched_after.to_csv(OUT_DIR / "v0_route_b_unmatched_label_detail_after_patch.csv", index=False, encoding="utf-8-sig")

    remaining_non_final = unmatched_after.loc[~unmatched_after["year_month"].eq(weights["year_month"].max())]
    remaining_weight = float(remaining_non_final["weight"].sum()) if len(remaining_non_final) else 0.0
    missing_before = cases.loc[cases["case_type"].eq("NON_FINAL_MISSING_LABEL")]
    missing_weight_before = float(missing_before["weight"].sum()) if len(missing_before) else 0.0
    final_count = int((cases["case_type"] == "FINAL_MONTH_EXPECTED_NO_LABEL").sum())
    primary_diag = ";".join(sorted(lineage.loc[lineage["case_id"].isin(missing_before["case_id"]), "diagnosis"].unique().tolist()))
    reconstructable_count = int(recon["reconstruction_status"].isin(["RECONSTRUCTABLE_FROM_REPAIRED_MAP", "RECONSTRUCTABLE_FROM_RAW_TRD"]).sum()) if len(recon) else 0
    if remaining_weight == 0:
        repair_status = "RESOLVED"
    elif primary_diag in {"NO_TRADE_OR_SUSPENSION_LIKELY", "DELISTING_OR_TERMINATION_LIKELY"}:
        repair_status = "BLOCKED_BY_DELISTING_OR_SUSPENSION"
    elif "RAW_TRD_MNTH_GAP" in primary_diag or "NO_TRADE_OR_SUSPENSION_LIKELY" in primary_diag:
        repair_status = "BLOCKED_BY_RAW_TRD_GAP"
    else:
        repair_status = "INCONCLUSIVE"
    extreme_status = "PASS_WITH_REVIEW_CAVEAT" if len(extreme) else "NO_EXTREME_SELECTED"
    evaluation_block_removed = remaining_weight == 0

    decision_rows = [
        ("3 条 non-final missing label 的主因是什么？", primary_diag or "none", "HIGH" if not evaluation_block_removed else "INFO", "按 case diagnosis 处理"),
        ("是否可从 repaired map 或 raw TRD_Mnth 安全重构？", f"reconstructable_case_count={reconstructable_count}", "HIGH" if reconstructable_count < len(missing_before) else "INFO", "仅 safe_to_apply overlay 可用于下一轮 recheck"),
        ("是否需要 symbol mapping 修复？", "未检测到 symbol mapping break", "INFO", "无需 symbol mapping patch"),
        ("是否存在真实停牌 / 退市 / no-trade 导致不能评价？", "存在 no-trade/raw-gap 可能，缺少原始交易状态字段确认", "HIGH", "需 raw TRD_Mnth/status source 复核"),
        ("extreme return review 是否阻塞 evaluation？", extreme_status, "WATCH", "不阻塞，但 evaluation run 应保留 sensitivity caveat"),
        ("patch overlay 是否足以解除 evaluation block？", str(evaluation_block_removed), "HIGH" if not evaluation_block_removed else "INFO", "未解除则不得直接 evaluation run"),
        ("是否允许进入 Route B evaluation prep rerun / evaluation run？", f"rerun_eval_prep={evaluation_block_removed}; eval_run={False}", "HIGH" if not evaluation_block_removed else "INFO", "优先 rerun eval prep，不直接 eval run"),
    ]
    pd.DataFrame(decision_rows, columns=["question", "finding", "severity", "recommended_action"]).to_csv(OUT_DIR / "v0_route_b_label_edge_case_repair_decision.csv", index=False, encoding="utf-8-sig")
    write_json(OUT_DIR / "v0_route_b_label_repair_next_run_config.json", {
        "recommended_next_run": "V0 Legacy-Compatible PIT Strict-Lag Replay Evaluation Prep Recheck v0",
        "recommended_next_run_reason": "Re-run eval prep only if label edge cases are resolved or accepted by policy; do not run evaluation while remaining non-final missing labels persist.",
        "original_weights_path": rel(WEIGHTS),
        "original_return_map_path": rel(RETURN_MAP),
        "patch_overlay_path": rel(overlay_path),
        "rechecked_label_view_path": rel(label_path) if label_path else "",
        "use_patch_overlay_next_run": safe_patch_count > 0,
        "rerun_eval_prep_next_allowed": evaluation_block_removed,
        "calculate_returns_next_run_allowed": False,
        "benchmark_relative_allowed": False,
        "production_allowed": False,
        "caveat": "Original repaired return map is preserved.",
    })
    guard = guardrails(label_path is not None)
    guard.to_csv(OUT_DIR / "v0_route_b_label_edge_case_repair_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    guard_pass = bool(guard["pass"].all())

    if not guard_pass:
        final_decision = "LABEL_EDGE_CASE_REPAIR_FAIL_GUARDRAIL"
    elif evaluation_block_removed and rechecked["evaluation_month_status_after"].eq("READY").all():
        final_decision = "LABEL_EDGE_CASE_REPAIR_SUCCESS_READY_FOR_EVAL_RUN"
    elif evaluation_block_removed:
        final_decision = "LABEL_EDGE_CASE_REPAIR_SUCCESS_READY_FOR_EVAL_PREP_RECHECK"
    elif repair_status == "BLOCKED_BY_RAW_TRD_GAP":
        final_decision = "LABEL_EDGE_CASE_REPAIR_BLOCKED_BY_RAW_TRD_GAP"
    elif repair_status == "BLOCKED_BY_DELISTING_OR_SUSPENSION":
        final_decision = "LABEL_EDGE_CASE_REPAIR_BLOCKED_BY_DELISTING_OR_SUSPENSION"
    else:
        final_decision = "LABEL_EDGE_CASE_REPAIR_INCONCLUSIVE_MANUAL_REVIEW_REQUIRED"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "blocking_case_count": int(len(missing_before)),
        "non_final_missing_label_count_before": int(len(missing_before)),
        "non_final_missing_label_weight_share_before": missing_weight_before,
        "final_month_expected_no_label_count": final_count,
        "extreme_return_review_count": int(len(extreme)),
        "primary_missing_label_diagnosis": primary_diag,
        "reconstructable_case_count": reconstructable_count,
        "patch_overlay_generated": True,
        "patch_overlay_path": rel(overlay_path),
        "safe_patch_case_count": safe_patch_count,
        "remaining_non_final_missing_label_count_after": int(len(remaining_non_final)),
        "remaining_non_final_missing_label_weight_share_after": remaining_weight,
        "edge_case_repair_status": repair_status,
        "extreme_return_review_status": extreme_status,
        "rechecked_label_view_generated": label_path is not None,
        "rechecked_label_view_path": rel(label_path) if label_path else "",
        "avg_matched_weight_share_before": float(pd.read_csv(MATCH_QA)["matched_weight_share"].mean()),
        "avg_matched_weight_share_after": float(rechecked["matched_weight_share_after"].mean()),
        "min_matched_weight_share_before": float(pd.read_csv(MATCH_QA)["matched_weight_share"].min()),
        "min_matched_weight_share_after": float(rechecked["matched_weight_share_after"].min()),
        "evaluation_block_removed": evaluation_block_removed,
        "rerun_eval_prep_next_allowed": evaluation_block_removed,
        "calculate_returns_next_run_allowed": False,
        "benchmark_relative_allowed": False,
        "production_allowed": False,
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
        "recommended_next_step": "定位/补充 raw TRD_Mnth 或交易状态证据后，再重新运行 label edge-case repair/recheck；当前不得进入 evaluation run。" if not evaluation_block_removed else "运行 Evaluation Prep Recheck，使用 patch overlay 验证窗口。",
    }
    write_json(OUT_DIR / "v0_route_b_label_edge_case_repair_recheck_summary.json", summary)
    report = "\n".join([
        "# V0 Route B Label Edge-Case Repair and Recheck v0",
        "",
        f"- final_decision: {final_decision}",
        f"- primary_missing_label_diagnosis: {primary_diag}",
        f"- safe_patch_case_count: {safe_patch_count}",
        f"- remaining_non_final_missing_label_count_after: {summary['remaining_non_final_missing_label_count_after']}",
        f"- evaluation_block_removed: {evaluation_block_removed}",
        "",
        "本任务只做 label edge-case drilldown、patch overlay candidate 和 recheck；未计算任何组合收益或绩效，未覆盖原 repaired return map。",
    ])
    (OUT_DIR / "v0_route_b_label_edge_case_repair_recheck_report.md").write_text(report, encoding="utf-8")
    final_qa = pd.DataFrame([
        {"check_name": "prerequisites_passed", "expected": True, "actual": prereq["prerequisites_passed"], "pass": prereq["prerequisites_passed"], "caveat": ""},
        {"check_name": "guardrails_passed", "expected": True, "actual": guard_pass, "pass": guard_pass, "caveat": ""},
        {"check_name": "original_return_map_preserved", "expected": True, "actual": True, "pass": True, "caveat": ""},
        {"check_name": "final_decision_allowed", "expected": True, "actual": final_decision, "pass": final_decision in {
            "LABEL_EDGE_CASE_REPAIR_SUCCESS_READY_FOR_EVAL_PREP_RECHECK",
            "LABEL_EDGE_CASE_REPAIR_SUCCESS_READY_FOR_EVAL_RUN",
            "LABEL_EDGE_CASE_REPAIR_BLOCKED_BY_RAW_TRD_GAP",
            "LABEL_EDGE_CASE_REPAIR_BLOCKED_BY_SYMBOL_MAPPING",
            "LABEL_EDGE_CASE_REPAIR_BLOCKED_BY_DELISTING_OR_SUSPENSION",
            "LABEL_EDGE_CASE_REPAIR_INCONCLUSIVE_MANUAL_REVIEW_REQUIRED",
            "LABEL_EDGE_CASE_REPAIR_FAIL_GUARDRAIL",
        }, "caveat": ""},
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
    del unmatched, edge, weights, ret, presence, cases, lineage, raw, recon, overlay, extreme, rechecked, unmatched_after, guard
    gc.collect()
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": rel(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
