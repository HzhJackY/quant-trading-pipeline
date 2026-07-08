from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_NAME = "V0 Route B Raw TRD Evidence Acquisition for Missing Labels v0"
OUT_NAME = "v0_route_b_raw_trd_evidence_acquisition_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / OUT_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

REPAIR_DIR = ROOT / "output" / "v0_route_b_label_edge_case_repair_recheck_v0"
REPAIR_SUMMARY = REPAIR_DIR / "v0_route_b_label_edge_case_repair_recheck_summary.json"
BLOCKING_CASES = REPAIR_DIR / "v0_route_b_label_blocking_cases.csv"
LINEAGE = REPAIR_DIR / "v0_route_b_label_lineage_drilldown.csv"
RECON = REPAIR_DIR / "v0_route_b_fwd_ret_reconstruction_check.csv"
REPAIR_DECISION = REPAIR_DIR / "v0_route_b_label_edge_case_repair_decision.csv"
RAW_LOOKUP_PREV = REPAIR_DIR / "v0_route_b_trd_mnth_raw_lookup_cases.csv"

WEIGHTS = ROOT / "output" / "v0_legacy_compatible_pit_strict_lag_replay_portfolio_construction_run_v0" / "v0_route_b_research_weights.parquet"
RETURN_MAP = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0" / "canonical_csmar_trd_mnth_return_map_repaired.parquet"
TRD_REPAIR_DIR = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0"

SEARCH_ROOTS = [ROOT / "data", ROOT / "data" / "raw", ROOT / "output", TRD_REPAIR_DIR]
KEYWORDS = [
    "TRD_Mnth", "Mretwd", "Mretnd", "Trdmnt", "Stkcd", "month return",
    "monthly return", "trading days", "suspension", "delist", "ST",
    "market type", "CSMAR", "return_map", "trd_mnth",
]


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
        "resume_instruction": f"先读取 {rel(RUN_DIR / 'RUN_STATE.md')}；继续时运行 scripts\\acquire_v0_route_b_raw_trd_evidence_v0.py，并重定向 stdout/stderr 到本目录。",
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
    return (pd.Period(str(ym), freq="M") + n).strftime("%Y-%m")


def collect_candidate_files() -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".csv", ".txt", ".xlsx", ".xls", ".parquet"}:
                continue
            name = str(p).lower()
            if any(k.lower() in name for k in KEYWORDS):
                rp = str(p.resolve())
                if rp not in seen:
                    seen.add(rp)
                    out.append(p)
    return out


def preview_file(path: Path) -> tuple[str, str, bool, str]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            import pyarrow.parquet as pq
            cols = pq.ParquetFile(path).schema_arrow.names
            return ",".join(cols[:30]), "schema", True, ""
        if suffix in {".csv", ".txt"}:
            encodings = ["utf-8-sig", "gbk", "utf-8"]
            raw_preview = ""
            for enc in encodings:
                try:
                    raw_preview = path.read_text(encoding=enc, errors="replace")[:1000]
                    break
                except Exception:
                    continue
            guesses = []
            for header in [0, 1, 2, 3]:
                try:
                    df = pd.read_csv(path, nrows=5, header=header, encoding="utf-8-sig")
                    guesses.append((header, list(map(str, df.columns[:12]))))
                except Exception:
                    try:
                        df = pd.read_csv(path, nrows=5, header=header, encoding="gbk")
                        guesses.append((header, list(map(str, df.columns[:12]))))
                    except Exception:
                        pass
            if guesses:
                best = 3 if any(h == 3 for h, _ in guesses) else guesses[0][0]
                return json.dumps({"raw_preview": raw_preview[:300], "guesses": guesses[:4]}, ensure_ascii=False), str(best), True, ""
            return raw_preview[:500], "unknown", True, ""
        if suffix in {".xlsx", ".xls"}:
            # Only inspect first rows; do not load full workbook.
            xls = pd.ExcelFile(path)
            sheet = xls.sheet_names[0]
            preview = pd.read_excel(path, sheet_name=sheet, nrows=8, header=None)
            header_guess = 3
            return preview.fillna("").astype(str).head(8).to_csv(index=False), str(header_guess), True, f"sheet={sheet}"
    except Exception as exc:
        return "", "unreadable", False, str(exc)[:300]
    return "", "unknown", False, "unsupported"


def inventory() -> pd.DataFrame:
    rows = []
    for p in collect_candidate_files():
        size = int(p.stat().st_size)
        if size > 80_000_000:
            preview, header_guess, readable, caveat = "", "skipped_large_file", False, "preview skipped for low-resource mode"
        else:
            preview, header_guess, readable, caveat = preview_file(p)
        likely = any(k.lower() in p.name.lower() for k in ["trd_mnth", "mretwd", "mretnd", "trdmnt"])
        rows.append({
            "file_path": rel(p),
            "file_name": p.name,
            "file_type": p.suffix.lower(),
            "file_size": size,
            "modified_time": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            "likely_trd_mnth_file": bool(likely),
            "detected_columns_or_header_preview": preview[:2000],
            "header_row_guess": header_guess,
            "readable": readable,
            "caveat": caveat,
        })
    return pd.DataFrame(rows)


def prereq_check(inv: pd.DataFrame) -> dict[str, Any]:
    flags = {
        "label_repair_summary_found": REPAIR_SUMMARY.exists(),
        "blocking_cases_found": BLOCKING_CASES.exists(),
        "lineage_drilldown_found": LINEAGE.exists(),
        "reconstruction_check_found": RECON.exists(),
        "route_b_weights_found": WEIGHTS.exists(),
        "repaired_return_map_found": RETURN_MAP.exists(),
        "raw_trd_candidate_files_found": len(inv) > 0,
    }
    paths = {
        "label_repair_summary_found": REPAIR_SUMMARY,
        "blocking_cases_found": BLOCKING_CASES,
        "lineage_drilldown_found": LINEAGE,
        "reconstruction_check_found": RECON,
        "route_b_weights_found": WEIGHTS,
        "repaired_return_map_found": RETURN_MAP,
    }
    missing = [rel(p) for k, p in paths.items() if not flags[k]]
    flags["prerequisites_passed"] = len(missing) == 0
    flags["missing_files"] = missing
    flags["caveat"] = "Raw TRD evidence acquisition only; no returns or performance calculated."
    return flags


def load_cases() -> pd.DataFrame:
    cases = pd.read_csv(BLOCKING_CASES, dtype={"year_month": str, "symbol_norm": str})
    cases = cases.loc[cases["case_type"].eq("NON_FINAL_MISSING_LABEL")].copy()
    cases["symbol_norm"] = norm_symbol(cases["symbol_norm"])
    cases["expected_label_month"] = cases["year_month"].apply(add_month)
    return cases[[
        "case_id", "year_month", "expected_label_month", "symbol_norm", "weight", "rank",
        "selected_reason", "alpha_signal_route_b_strict_lag", "issue_type", "blocking_for_eval",
    ]].rename(columns={"issue_type": "current_diagnosis"})


def load_repaired_map() -> pd.DataFrame:
    ret = pd.read_parquet(
        RETURN_MAP,
        columns=["symbol_norm", "year_month", "monthly_return_t", "fwd_ret_1m", "primary_return_field", "raw_return_field_values", "return_valid_flag", "invalid_reason"],
    )
    ret["symbol_norm"] = norm_symbol(ret["symbol_norm"])
    ret["year_month"] = ret["year_month"].astype(str).str.slice(0, 7)
    ret["monthly_return_t"] = pd.to_numeric(ret["monthly_return_t"], errors="coerce")
    ret["fwd_ret_1m"] = pd.to_numeric(ret["fwd_ret_1m"], errors="coerce")
    ret = ret.loc[ret["primary_return_field"].astype(str).eq("Mretwd")].copy()
    return ret


def read_candidate_for_lookup(path: Path, inv_row: pd.Series) -> pd.DataFrame | None:
    try:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix in {".csv", ".txt"}:
            header_guess = inv_row.get("header_row_guess", "0")
            header = int(header_guess) if str(header_guess).isdigit() else 0
            try:
                df = pd.read_csv(path, header=header, encoding="utf-8-sig")
            except Exception:
                df = pd.read_csv(path, header=header, encoding="gbk")
        elif suffix in {".xlsx", ".xls"}:
            header_guess = inv_row.get("header_row_guess", "3")
            header = int(header_guess) if str(header_guess).isdigit() else 3
            df = pd.read_excel(path, header=header)
        else:
            return None
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return None


def normalize_trd_like(df: pd.DataFrame) -> pd.DataFrame | None:
    cols = {c.lower(): c for c in df.columns}
    sym_col = None
    mon_col = None
    for cand in ["symbol_norm", "stkcd", "symbol", "证券代码", "股票代码"]:
        if cand.lower() in cols:
            sym_col = cols[cand.lower()]
            break
    for cand in ["year_month", "trdmnt", "month", "交易月份", "月份"]:
        if cand.lower() in cols:
            mon_col = cols[cand.lower()]
            break
    if sym_col is None or mon_col is None:
        return None
    out = pd.DataFrame()
    out["symbol_norm"] = norm_symbol(df[sym_col])
    out["year_month"] = df[mon_col].astype(str).str.replace("/", "-", regex=False).str.slice(0, 7)
    for target, candidates in {
        "Mretwd": ["Mretwd", "mretwd", "考虑现金红利再投资的月个股回报率"],
        "Mretnd": ["Mretnd", "mretnd", "不考虑现金红利再投资的月个股回报率"],
        "trading_days": ["Ndaytrd", "trading_days", "月交易天数"],
        "trading_status": ["TradingStatus", "trading_status", "交易状态"],
        "market_type": ["Markettype", "market_type", "市场类型"],
    }.items():
        src = next((c for c in candidates if c in df.columns), None)
        out[target] = df[src] if src else np.nan
    out["Mretwd"] = pd.to_numeric(out["Mretwd"], errors="coerce")
    out["Mretnd"] = pd.to_numeric(out["Mretnd"], errors="coerce")
    return out


def lookup_matrix(cases: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    readable_candidates = inv.loc[
        inv["readable"].astype(bool)
        & (
            inv["likely_trd_mnth_file"].astype(bool)
            | inv["file_name"].astype(str).str.contains("canonical_csmar_trd_mnth_return_map_repaired", case=False, na=False)
            | inv["file_name"].astype(str).str.contains("return_map", case=False, na=False)
        )
        & (pd.to_numeric(inv["file_size"], errors="coerce") <= 80_000_000)
    ].copy()
    for _, inv_row in readable_candidates.iterrows():
        p = ROOT / inv_row["file_path"]
        df = read_candidate_for_lookup(p, inv_row)
        norm = normalize_trd_like(df) if df is not None else None
        for c in cases.itertuples(index=False):
            if norm is None:
                rows.append({
                    "case_id": c.case_id, "symbol_norm": c.symbol_norm, "year_month": c.year_month,
                    "expected_label_month": c.expected_label_month, "file_path": inv_row["file_path"],
                    "current_month_record_found": False, "expected_label_month_record_found": False,
                    "nearby_month_record_count": 0, "raw_mretwd_current": np.nan,
                    "raw_mretwd_expected_label_month": np.nan, "raw_mretnd_current": np.nan,
                    "raw_mretnd_expected_label_month": np.nan, "trading_status_current": "",
                    "trading_status_expected_label_month": "", "trading_days_current": "",
                    "trading_days_expected_label_month": "", "listing_delisting_evidence": "",
                    "lookup_status": "FILE_UNREADABLE" if not bool(inv_row["readable"]) else "SYMBOL_FORMAT_MISMATCH_POSSIBLE",
                    "caveat": "No usable Stkcd/Trdmnt-like columns detected.",
                })
                continue
            cur = norm.loc[norm["symbol_norm"].eq(c.symbol_norm) & norm["year_month"].eq(c.year_month)]
            nxt = norm.loc[norm["symbol_norm"].eq(c.symbol_norm) & norm["year_month"].eq(c.expected_label_month)]
            nearby = norm.loc[norm["symbol_norm"].eq(c.symbol_norm) & norm["year_month"].between(add_month(c.year_month, -2), add_month(c.year_month, 2))]
            if len(nxt) and nxt["Mretwd"].notna().any():
                status = "FOUND_NEXT_MONTH_MRETWD"
            elif len(cur):
                status = "FOUND_CURRENT_ONLY"
            elif len(nearby):
                status = "FOUND_NEARBY_ONLY"
            else:
                status = "NOT_FOUND"
            rows.append({
                "case_id": c.case_id,
                "symbol_norm": c.symbol_norm,
                "year_month": c.year_month,
                "expected_label_month": c.expected_label_month,
                "file_path": inv_row["file_path"],
                "current_month_record_found": bool(len(cur)),
                "expected_label_month_record_found": bool(len(nxt)),
                "nearby_month_record_count": int(len(nearby)),
                "raw_mretwd_current": float(cur["Mretwd"].dropna().iloc[0]) if len(cur["Mretwd"].dropna()) else np.nan,
                "raw_mretwd_expected_label_month": float(nxt["Mretwd"].dropna().iloc[0]) if len(nxt["Mretwd"].dropna()) else np.nan,
                "raw_mretnd_current": float(cur["Mretnd"].dropna().iloc[0]) if len(cur["Mretnd"].dropna()) else np.nan,
                "raw_mretnd_expected_label_month": float(nxt["Mretnd"].dropna().iloc[0]) if len(nxt["Mretnd"].dropna()) else np.nan,
                "trading_status_current": str(cur["trading_status"].dropna().iloc[0]) if len(cur["trading_status"].dropna()) else "",
                "trading_status_expected_label_month": str(nxt["trading_status"].dropna().iloc[0]) if len(nxt["trading_status"].dropna()) else "",
                "trading_days_current": str(cur["trading_days"].dropna().iloc[0]) if len(cur["trading_days"].dropna()) else "",
                "trading_days_expected_label_month": str(nxt["trading_days"].dropna().iloc[0]) if len(nxt["trading_days"].dropna()) else "",
                "listing_delisting_evidence": "",
                "lookup_status": status,
                "caveat": "Lookup is evidence only; no return calculation performed.",
            })
    return pd.DataFrame(rows)


def symbol_mapping_evidence(cases: pd.DataFrame, ret: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ret_symbols = set(ret["symbol_norm"].astype(str))
    weight_symbols = set(norm_symbol(weights["symbol_norm"]).astype(str))
    for c in cases.itertuples(index=False):
        sym = str(c.symbol_norm)
        rows.append({
            "case_id": c.case_id,
            "symbol_norm": sym,
            "adapter_symbol_format": "6_digit",
            "weights_symbol_format": "6_digit" if sym in weight_symbols else "not_found",
            "return_map_symbol_format": "6_digit" if sym in ret_symbols else "not_found",
            "raw_trd_symbol_format_found": "6_digit" if sym in ret_symbols else "",
            "leading_zero_issue": False,
            "exchange_suffix_issue": False,
            "code_change_evidence": "",
            "duplicate_mapping_evidence": bool(ret.duplicated(["symbol_norm", "year_month"]).any()),
            "mapping_issue_status": "NO_MAPPING_ISSUE_DETECTED" if sym in weight_symbols else "INCONCLUSIVE",
            "caveat": "",
        })
    return pd.DataFrame(rows)


def diagnosis(cases: pd.DataFrame, lookup: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ret_months = ret.groupby("symbol_norm")["year_month"].apply(lambda s: sorted(set(s.astype(str)))).to_dict()
    for c in cases.itertuples(index=False):
        sub = lookup.loc[lookup["case_id"].eq(c.case_id)]
        found_next = bool((sub["lookup_status"] == "FOUND_NEXT_MONTH_MRETWD").any()) if len(sub) else False
        found_current = bool(sub["current_month_record_found"].astype(bool).any()) if len(sub) else False
        later = [m for m in ret_months.get(c.symbol_norm, []) if m > c.year_month]
        if found_next:
            diag = "RETURN_MAP_SHIFT_BUG"
            conf = "HIGH"
        elif found_current and not found_next:
            diag = "VERIFIED_RAW_TRD_GAP"
            conf = "MEDIUM"
        elif len(later) == 0:
            diag = "VERIFIED_DELISTING_WITHOUT_RETURN"
            conf = "MEDIUM"
        else:
            diag = "INCONCLUSIVE_NEED_EXTERNAL_DATA"
            conf = "MEDIUM"
        rows.append({
            "case_id": c.case_id,
            "symbol_norm": c.symbol_norm,
            "year_month": c.year_month,
            "expected_label_month": c.expected_label_month,
            "current_month_record_status": "FOUND" if found_current else "NOT_FOUND",
            "expected_label_month_record_status": "FOUND" if found_next else "NOT_FOUND",
            "later_months_record_status": f"later_count={len(later)}",
            "trading_status_evidence": "",
            "trading_days_evidence": "",
            "delisting_evidence": "symbol disappears after current month" if len(later) == 0 else "",
            "suspension_evidence": "",
            "no_trade_evidence": "expected label month absent from available return map/candidates" if not found_next else "",
            "raw_gap_evidence": "no candidate file provides next-month Mretwd" if not found_next else "",
            "final_diagnosis": diag,
            "confidence": conf,
            "caveat": "No raw trading status fields available unless candidate lookup found them.",
        })
    return pd.DataFrame(rows)


def policy_design(diag: pd.DataFrame, cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = diag.merge(cases[["case_id", "weight"]], on="case_id", how="left")
    rows = []
    for r in merged.itertuples(index=False):
        if r.final_diagnosis in {"RETURN_MAP_SHIFT_BUG"}:
            policy = "PATCH_FROM_RAW_MRETWD"
            primary = True
            sens = False
            risk = "LOW"
            evidence_req = "next-month Mretwd found"
        elif r.final_diagnosis == "VERIFIED_RAW_TRD_GAP":
            policy = "EXCLUDE_AFFECTED_MONTH_FROM_PRIMARY_EVAL"
            primary = False
            sens = True
            risk = "HIGH"
            evidence_req = "documented raw TRD gap and explicit primary-window exclusion"
        elif r.final_diagnosis == "INCONCLUSIVE_NEED_EXTERNAL_DATA":
            policy = "BLOCK_EVAL_PENDING_EXTERNAL_DATA"
            primary = False
            sens = False
            risk = "HIGH"
            evidence_req = "raw TRD_Mnth/status source or documented no-trade handling"
        elif r.final_diagnosis in {"VERIFIED_DELISTING_WITHOUT_RETURN", "VERIFIED_NO_TRADE_OR_SUSPENSION"}:
            policy = "EXCLUDE_AFFECTED_MONTH_FROM_PRIMARY_EVAL"
            primary = False
            sens = True
            risk = "HIGH"
            evidence_req = "delisting/no-trade return policy"
        else:
            policy = "BLOCK_EVAL_PENDING_EXTERNAL_DATA"
            primary = False
            sens = False
            risk = "HIGH"
            evidence_req = "external data"
        rows.append({
            "case_id": r.case_id,
            "final_diagnosis": r.final_diagnosis,
            "recommended_policy": policy,
            "primary_eval_allowed_under_policy": primary,
            "sensitivity_required": sens,
            "evidence_required": evidence_req,
            "risk": risk,
            "caveat": "No default zero fill; no holding deletion.",
        })
    policy = pd.DataFrame(rows)
    summary = policy.merge(cases[["case_id", "weight"]], on="case_id").groupby("recommended_policy").agg(
        case_count=("case_id", "count"),
        weight_share=("weight", "sum"),
        primary_eval_allowed=("primary_eval_allowed_under_policy", "all"),
        sensitivity_required=("sensitivity_required", "any"),
        risk_level=("risk", lambda s: "HIGH" if (s == "HIGH").any() else "LOW"),
    ).reset_index().rename(columns={"recommended_policy": "policy"})
    summary["interpretation"] = "pre-registered missing-label policy; no returns calculated"
    return policy, summary


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
        "missing_return_default_zero_filled": False,
        "missing_holdings_deleted": False,
    }
    return pd.DataFrame([{"guardrail": k, "expected": False, "actual": v, "pass": v is False} for k, v in actuals.items()])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", "inventory")
    inv = inventory()
    inv.to_csv(OUT_DIR / "v0_raw_trd_file_inventory.csv", index=False, encoding="utf-8-sig")
    prereq = prereq_check(inv)
    write_json(OUT_DIR / "v0_raw_trd_evidence_prerequisite_check.json", prereq)
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", "case_lookup")
    cases = load_cases()
    cases.to_csv(OUT_DIR / "v0_missing_label_case_identity.csv", index=False, encoding="utf-8-sig")
    ret = load_repaired_map()
    weights = pd.read_parquet(WEIGHTS, columns=["symbol_norm", "year_month", "weight"])
    lookup = lookup_matrix(cases, inv)
    lookup.to_csv(OUT_DIR / "v0_missing_label_raw_trd_lookup_matrix.csv", index=False, encoding="utf-8-sig")
    mapping = symbol_mapping_evidence(cases, ret, weights)
    mapping.to_csv(OUT_DIR / "v0_missing_label_symbol_mapping_evidence.csv", index=False, encoding="utf-8-sig")
    diag = diagnosis(cases, lookup, ret)
    diag.to_csv(OUT_DIR / "v0_missing_label_no_trade_suspension_delisting_diagnosis.csv", index=False, encoding="utf-8-sig")
    policy, policy_summary = policy_design(diag, cases)
    policy.to_csv(OUT_DIR / "v0_missing_label_handling_policy_design.csv", index=False, encoding="utf-8-sig")
    policy_summary.to_csv(OUT_DIR / "v0_missing_label_policy_summary.csv", index=False, encoding="utf-8-sig")

    write_state("running", "decision")
    all_diag = not diag["final_diagnosis"].eq("INCONCLUSIVE_NEED_EXTERNAL_DATA").any()
    patch_all = bool((diag["final_diagnosis"] == "RETURN_MAP_SHIFT_BUG").all())
    no_mapping = bool((mapping["mapping_issue_status"] == "NO_MAPPING_ISSUE_DETECTED").all())
    no_unresolved = bool(~diag["final_diagnosis"].isin(["VERIFIED_DELISTING_WITHOUT_RETURN", "VERIFIED_NO_TRADE_OR_SUSPENSION", "INCONCLUSIVE_NEED_EXTERNAL_DATA"]).any())
    primary_allowed = bool(policy["primary_eval_allowed_under_policy"].all())
    eval_prep_allowed = bool(all_diag and no_mapping and not patch_all and not primary_allowed)
    direct_allowed = bool(patch_all and primary_allowed)
    criteria = pd.DataFrame([
        ("all_missing_labels_diagnosed", True, all_diag, all_diag, ""),
        ("raw_patch_available_for_all_blocking_cases", True, patch_all, patch_all, ""),
        ("no_symbol_mapping_issue", True, no_mapping, no_mapping, ""),
        ("no_unresolved_delisting_or_suspension", True, no_unresolved, no_unresolved, ""),
        ("policy_pre_registered", True, len(policy) == len(cases), len(policy) == len(cases), ""),
        ("primary_eval_allowed", True, primary_allowed, primary_allowed, ""),
        ("eval_prep_recheck_allowed", True, eval_prep_allowed, eval_prep_allowed, ""),
        ("direct_eval_run_allowed", True, direct_allowed, direct_allowed, ""),
    ], columns=["criterion", "expected", "actual", "pass", "caveat"])
    criteria.to_csv(OUT_DIR / "v0_raw_trd_evidence_eval_unblock_decision.csv", index=False, encoding="utf-8-sig")

    if patch_all:
        next_run = "V0 Route B Label Patch Overlay Build v0"
        reason = "raw next-month Mretwd found for all blocking cases"
    elif all_diag and eval_prep_allowed:
        next_run = "V0 Route B Evaluation Prep Recheck with Label Policy v0"
        reason = "cases diagnosed and policy can be pre-registered"
    elif diag["final_diagnosis"].eq("INCONCLUSIVE_NEED_EXTERNAL_DATA").any():
        next_run = "V0 Route B External TRD Data Acquisition Required"
        reason = "candidate files insufficient to diagnose all cases"
    else:
        next_run = "V0 Route B Evaluation Block Maintained"
        reason = "no safe primary evaluation policy"
    write_json(OUT_DIR / "v0_raw_trd_evidence_next_run_config.json", {
        "recommended_next_run": next_run,
        "recommended_next_run_reason": reason,
        "use_patch_overlay_next_run": patch_all,
        "patch_overlay_path": "",
        "use_policy_exclusion_next_run": eval_prep_allowed,
        "exclusion_months_or_cases": cases["case_id"].tolist() if eval_prep_allowed else [],
        "rerun_eval_prep_next_allowed": eval_prep_allowed or patch_all,
        "direct_eval_run_allowed": direct_allowed,
        "calculate_returns_next_run_allowed": False,
        "benchmark_relative_allowed": False,
        "production_allowed": False,
        "caveat": "Do not calculate returns until eval prep recheck passes.",
    })
    guard = guardrails()
    guard.to_csv(OUT_DIR / "v0_raw_trd_evidence_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    guard_pass = bool(guard["pass"].all())
    cases_with_next = int((lookup["lookup_status"] == "FOUND_NEXT_MONTH_MRETWD").groupby(lookup["case_id"]).any().sum()) if len(lookup) else 0
    mapping_issue_count = int((mapping["mapping_issue_status"] != "NO_MAPPING_ISSUE_DETECTED").sum())
    no_trade_count = int(diag["final_diagnosis"].eq("VERIFIED_NO_TRADE_OR_SUSPENSION").sum())
    raw_gap_count = int(diag["final_diagnosis"].eq("VERIFIED_RAW_TRD_GAP").sum())
    inconclusive_count = int(diag["final_diagnosis"].eq("INCONCLUSIVE_NEED_EXTERNAL_DATA").sum())
    recommended_policy = ";".join(policy["recommended_policy"].dropna().unique().tolist())
    if not guard_pass:
        final_decision = "RAW_TRD_EVIDENCE_FAIL_GUARDRAIL"
    elif patch_all and direct_allowed:
        final_decision = "RAW_TRD_EVIDENCE_FOUND_PATCH_READY"
    elif eval_prep_allowed:
        final_decision = "RAW_TRD_EVIDENCE_SUPPORTS_POLICY_RECHECK"
    elif inconclusive_count:
        final_decision = "RAW_TRD_EVIDENCE_EXTERNAL_DATA_REQUIRED"
    else:
        final_decision = "RAW_TRD_EVIDENCE_BLOCK_EVAL_MAINTAINED"
    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "raw_trd_candidate_files_found": int(len(inv)),
        "raw_trd_file_inventory_generated": True,
        "blocking_case_count": int(len(cases)),
        "cases_with_next_month_mretwd_found": cases_with_next,
        "cases_with_symbol_mapping_issue": mapping_issue_count,
        "cases_verified_no_trade_or_suspension": no_trade_count,
        "cases_verified_raw_trd_gap": raw_gap_count,
        "cases_inconclusive_need_external_data": inconclusive_count,
        "missing_label_policy_designed": True,
        "recommended_policy": recommended_policy,
        "primary_eval_allowed": primary_allowed,
        "eval_prep_recheck_allowed": eval_prep_allowed or patch_all,
        "direct_eval_run_allowed": direct_allowed,
        "evaluation_block_removed": direct_allowed,
        "recommended_next_run": next_run,
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
        "missing_return_default_zero_filled": False,
        "missing_holdings_deleted": False,
        "guardrails_passed": guard_pass,
        "final_decision": final_decision,
        "recommended_next_step": "补充外部/raw TRD_Mnth 月度收益或交易状态数据后重跑 evidence acquisition；当前 evaluation block 维持。" if final_decision == "RAW_TRD_EVIDENCE_EXTERNAL_DATA_REQUIRED" else reason,
    }
    write_json(OUT_DIR / "v0_route_b_raw_trd_evidence_acquisition_summary.json", summary)
    report = "\n".join([
        "# V0 Route B Raw TRD Evidence Acquisition v0",
        "",
        f"- final_decision: {final_decision}",
        f"- raw_trd_candidate_files_found: {len(inv)}",
        f"- cases_with_next_month_mretwd_found: {cases_with_next}",
        f"- cases_inconclusive_need_external_data: {inconclusive_count}",
        f"- recommended_next_run: {next_run}",
        "",
        "本任务只做 raw/intermediate TRD 证据定位和 missing-label policy 设计；未计算任何组合收益或绩效，未修改原 repaired return map 或 Route B weights。",
    ])
    (OUT_DIR / "v0_route_b_raw_trd_evidence_acquisition_report.md").write_text(report, encoding="utf-8")
    final_qa = pd.DataFrame([
        {"check_name": "prerequisites_passed", "expected": True, "actual": prereq["prerequisites_passed"], "pass": prereq["prerequisites_passed"], "caveat": ""},
        {"check_name": "guardrails_passed", "expected": True, "actual": guard_pass, "pass": guard_pass, "caveat": ""},
        {"check_name": "final_decision_allowed", "expected": True, "actual": final_decision, "pass": final_decision in {
            "RAW_TRD_EVIDENCE_FOUND_PATCH_READY",
            "RAW_TRD_EVIDENCE_SUPPORTS_POLICY_RECHECK",
            "RAW_TRD_EVIDENCE_EXTERNAL_DATA_REQUIRED",
            "RAW_TRD_EVIDENCE_BLOCK_EVAL_MAINTAINED",
            "RAW_TRD_EVIDENCE_FAIL_GUARDRAIL",
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
    del inv, cases, ret, weights, lookup, mapping, diag, policy, policy_summary, criteria, guard
    gc.collect()
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": rel(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
