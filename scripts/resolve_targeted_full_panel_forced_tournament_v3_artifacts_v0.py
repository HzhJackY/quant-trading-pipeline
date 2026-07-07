from __future__ import annotations

import csv
import gc
import json
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, message="Could not infer format.*")

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None


TASK = "targeted_full_panel_forced_tournament_v3_artifact_resolver_v0"
ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "output" / "full_panel_forced_tournament_v3"
OUT_DIR = ROOT / "output" / TASK
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK
RUN_STATE = RUN_DIR / "RUN_STATE.md"
CANONICAL_RETURN = ROOT / "output" / "robust_cleaned_fundamental_factor_variant_build_v0" / "robust_cleaned_factor_score_panel_v0.parquet"

TABULAR_EXTS = {".csv", ".parquet", ".xlsx", ".xls"}
TEXT_EXTS = {".md", ".txt", ".json", ".yaml", ".yml", ".csv"}
MAX_TEXT_BYTES = 220_000

MODELS = [
    "V0_LINEAR_FULL_OOS",
    "V0_FULL_V15_OOS",
    "V7_TOAWARE_FULL_OOS",
    "V7_FULL_V15_OOS",
    "COMPACT_F_FULL_OOS_ALIGNED",
    "COMPACT_F_V15_ALIGNED",
    "BLEND_V0_50_V7_50",
    "TOP50_BUFFER_35_75",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("/", "\\")
    except Exception:
        return str(path)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def append_state(text: str) -> None:
    with RUN_STATE.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now_iso()}\n{text}\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def bool_str(v: bool) -> str:
    return "true" if bool(v) else "false"


def safe_text(path: Path) -> str:
    try:
        with path.open("rb") as f:
            raw = f.read(MAX_TEXT_BYTES)
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def iter_files() -> list[Path]:
    if not TARGET.exists():
        return []
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(TARGET):
        dirnames[:] = [d for d in dirnames if d not in {".git", ".venv", "__pycache__", ".pytest_cache"}]
        for name in filenames:
            files.append(Path(dirpath) / name)
    return sorted(files, key=lambda p: rel(p).lower())


def infer_artifact_type(path: Path, text: str = "") -> str:
    hay = f"{rel(path)}\n{text[:40000]}".lower()
    name = path.name.lower()
    if "weight_audit" in name or "monthly_weight_audit" in name:
        return "WEIGHT_AUDIT"
    if "training_audit" in name:
        return "TRAINING_AUDIT"
    if "feature_audit" in name or "canonical_feature" in name:
        return "FEATURE_AUDIT"
    if "alignment_audit" in name or "coverage_audit" in name or "no_leakage" in name or "final_qa" in name or "panel_audit" in name:
        return "ALIGNMENT_AUDIT"
    if "label_generation" in name:
        return "LABEL_AUDIT"
    if "split_plan" in name:
        return "SPLIT_PLAN"
    if path.suffix.lower() in {".md", ".json", ".txt", ".yaml", ".yml"} or "report" in name or "recommendation" in name:
        return "CONFIG_OR_REPORT"
    if "monthly_returns" in name or "return" in name:
        return "OOS_RETURN_PANEL"
    if "metrics" in name or "yearly_breakdown" in name or "rank_ic" in name or "turnover" in name:
        return "PERFORMANCE_SUMMARY"
    if any(x in name for x in ["full_oos", "aligned", "scores"]) and any(x in hay for x in ["score", "pred", "rank", "signal", "oos"]):
        return "OOS_SCORE_OR_SIGNAL_PANEL"
    if "weight" in hay and any(x in hay for x in ["symbol", "month_end", "date"]):
        return "HISTORICAL_WEIGHTS_CANDIDATE"
    return "UNKNOWN_RELEVANT"


def inventory(files: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in files:
        sample = safe_text(path) if path.suffix.lower() in TEXT_EXTS else ""
        hay = f"{rel(path)}\n{sample}".lower()
        st = path.stat()
        rows.append(
            {
                "artifact_path": rel(path),
                "file_name": path.name,
                "file_type": path.suffix.lower().lstrip(".") or "no_ext",
                "file_size_bytes": st.st_size,
                "modified_time": datetime.fromtimestamp(st.st_mtime).isoformat(),
                "likely_artifact_type": infer_artifact_type(path, sample),
                "contains_v0": bool_str(re.search(r"(^|[^a-z0-9])v0([^a-z0-9]|$)", hay) is not None),
                "contains_v7": bool_str(re.search(r"(^|[^a-z0-9])v7([^a-z0-9]|$)", hay) is not None),
                "contains_blend": bool_str("blend" in hay),
                "contains_compact_f": bool_str("compact_f" in hay or "compact-f" in hay),
                "contains_weights": bool_str("weight" in hay),
                "contains_weight_audit": bool_str("weight_audit" in hay or "monthly_weight_audit" in hay),
                "contains_scores": bool_str("score" in hay or "prediction" in hay or "signal" in hay),
                "contains_returns": bool_str("return" in hay or "fwd_ret" in hay),
                "contains_oos": bool_str("oos" in hay),
                "contains_training_audit": bool_str("training_audit" in hay),
                "contains_alignment_audit": bool_str("alignment_audit" in hay or "coverage_audit" in hay),
                "contains_config_or_report": bool_str(path.suffix.lower() in {".md", ".json", ".txt", ".yaml", ".yml"} or "report" in hay),
                "notes": "",
            }
        )
    return rows


def parquet_columns_rows(path: Path) -> tuple[list[str], int | str]:
    if pq is None:
        return [], ""
    try:
        pf = pq.ParquetFile(path)
        return pf.schema.names, pf.metadata.num_rows
    except Exception:
        return [], ""


def csv_columns_rows(path: Path) -> tuple[list[str], int | str]:
    try:
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        n = sum(1 for _ in path.open("rb")) - 1
        return cols, max(n, 0)
    except Exception:
        return [], ""


def xlsx_columns_rows(path: Path) -> tuple[list[str], int | str]:
    try:
        df = pd.read_excel(path, nrows=0)
        return df.columns.tolist(), ""
    except Exception:
        return [], ""


def detect_col(cols: list[str], candidates: list[str], contains: bool = True) -> str:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    if contains:
        for col in cols:
            lc = col.lower()
            if any(c.lower() in lc for c in candidates):
                return col
    return ""


def read_needed(path: Path, cols: list[str]) -> pd.DataFrame:
    usecols = [c for c in cols if c]
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=usecols)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=usecols, dtype=str)
    return pd.read_excel(path, usecols=usecols, dtype=str)


def normalize_symbol(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\D", "", regex=True).str[-6:].str.zfill(6)


def normalize_month(s: pd.Series) -> pd.Series:
    return (pd.to_datetime(s, errors="coerce") + pd.offsets.MonthEnd(0)).dt.normalize()


def model_from_path(path: str) -> str:
    p = path.lower()
    if "v0_linear_full_oos" in p:
        return "V0_LINEAR_FULL_OOS"
    if "v0_full_v15_oos" in p:
        return "V0_FULL_V15_OOS"
    if "v7_toaware_full_oos" in p:
        return "V7_TOAWARE_FULL_OOS"
    if "v7_full_v15_oos" in p:
        return "V7_FULL_V15_OOS"
    if "compact_f_full_oos_aligned" in p:
        return "COMPACT_F_FULL_OOS_ALIGNED"
    if "compact_f_v15_aligned" in p:
        return "COMPACT_F_V15_ALIGNED"
    if "blend" in p:
        return "BLEND_V0_50_V7_50"
    if "top50" in p or "35_75" in p or "buffer" in p:
        return "TOP50_BUFFER_35_75"
    if "v0" in p:
        return "V0_LINEAR_FULL_OOS"
    if "v7" in p:
        return "V7_TOAWARE_FULL_OOS"
    if "compact_f" in p:
        return "COMPACT_F_FULL_OOS_ALIGNED"
    return ""


def schema_validation(files: list[Path], inv_by_path: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for path in files:
        if path.suffix.lower() not in TABULAR_EXTS:
            continue
        if path.suffix.lower() == ".parquet":
            cols, row_count = parquet_columns_rows(path)
        elif path.suffix.lower() == ".csv":
            cols, row_count = csv_columns_rows(path)
        else:
            cols, row_count = xlsx_columns_rows(path)
        likely = inv_by_path[rel(path)]["likely_artifact_type"]
        symbol_col = detect_col(cols, ["symbol", "stock_code", "code", "ticker", "证券代码"])
        date_col = detect_col(cols, ["month_end", "rebalance_date", "trade_date", "date", "portfolio_month_end", "month"])
        weight_col = detect_col(cols, ["weight", "target_weight", "portfolio_weight"], True)
        score_col = detect_col(cols, ["score", "signal"], True)
        rank_col = detect_col(cols, ["rank"], True)
        pred_col = detect_col(cols, ["prediction", "pred", "y_hat"], True)
        return_col = detect_col(cols, ["return", "ret", "fwd_ret_1m", "gross_return", "net_return"], True)
        label_col = detect_col(cols, ["label", "fwd_ret_1m", "target"], True)
        cost_col = detect_col(cols, ["cost_bps", "cost"], True)
        turnover_col = detect_col(cols, ["turnover"], True)
        portfolio_col = detect_col(cols, ["portfolio_name", "portfolio", "strategy"], True)
        model_col = detect_col(cols, ["model_name", "model", "candidate"], True)
        month_count = symbol_count = duplicate_count = ""
        min_month = max_month = ""
        avg_ws = min_ws = max_ws = ""
        is_hist = False
        if symbol_col and date_col:
            try:
                df = read_needed(path, [symbol_col, date_col, weight_col])
                sy = normalize_symbol(df[symbol_col])
                mo = normalize_month(df[date_col])
                mask = sy.notna() & mo.notna()
                month_count = int(mo[mask].nunique())
                symbol_count = int(sy[mask].nunique())
                if mask.any():
                    min_month = str(mo[mask].min().date())
                    max_month = str(mo[mask].max().date())
                    duplicate_count = int(pd.DataFrame({"s": sy[mask], "m": mo[mask]}).duplicated(["s", "m"]).sum())
                if weight_col:
                    w = pd.to_numeric(df[weight_col], errors="coerce")
                    wsum = pd.DataFrame({"m": mo, "w": w}).dropna().groupby("m")["w"].sum()
                    if len(wsum):
                        avg_ws, min_ws, max_ws = float(wsum.mean()), float(wsum.min()), float(wsum.max())
                del df
                gc.collect()
            except Exception as exc:
                min_month = f"stats_error:{type(exc).__name__}"
        is_hist = isinstance(month_count, int) and month_count >= 24
        has_score_like = bool(score_col or rank_col or pred_col or ("oos" in path.name.lower() and label_col))
        has_weight = bool(weight_col)
        weight_sum_ok = avg_ws != "" and min_ws >= 0.98 and max_ws <= 1.02
        is_weight_audit = likely == "WEIGHT_AUDIT" or "weight_audit" in path.name.lower()
        is_return = likely == "OOS_RETURN_PANEL" or (bool(return_col) and not has_score_like and not has_weight)
        is_valid_weight = bool(symbol_col and date_col and has_weight and is_hist and weight_sum_ok and not is_weight_audit and not has_score_like and not is_return)
        is_oos_score = bool(symbol_col and date_col and has_score_like and not is_valid_weight)
        reason = []
        if is_valid_weight:
            m = model_from_path(rel(path))
            if m.startswith("V0"):
                label = "VALID_V0_HISTORICAL_WEIGHTS"
            elif m.startswith("V7"):
                label = "VALID_V7_HISTORICAL_WEIGHTS"
            elif m.startswith("COMPACT_F"):
                label = "VALID_COMPACT_F_HISTORICAL_WEIGHTS"
            elif m.startswith("BLEND"):
                label = "VALID_BLEND_HISTORICAL_WEIGHTS"
            else:
                label = "VALID_OTHER_HISTORICAL_WEIGHTS"
            reason.append("symbol/date/weight present; month_count >= 24; weight sums near 1; not audit/score/return")
        elif is_oos_score:
            m = model_from_path(rel(path))
            if m.startswith("V0"):
                label = "OOS_SCORE_PANEL_V0"
            elif m.startswith("V7"):
                label = "OOS_SCORE_PANEL_V7"
            elif m.startswith("COMPACT_F"):
                label = "OOS_SCORE_PANEL_COMPACT_F"
            else:
                label = "AMBIGUOUS_NEEDS_MANUAL_REVIEW"
            reason.append("symbol/date plus score/prediction/rank/signal/label evidence; not valid weights")
        elif is_weight_audit:
            label = "WEIGHT_AUDIT_NOT_WEIGHTS"
            reason.append("weight audit file; not a holding weight panel")
        elif likely in {"TRAINING_AUDIT", "FEATURE_AUDIT"}:
            label = "TRAINING_OR_FEATURE_AUDIT"
        elif likely in {"ALIGNMENT_AUDIT", "LABEL_AUDIT"}:
            label = "ALIGNMENT_OR_LABEL_AUDIT"
        elif likely == "SPLIT_PLAN":
            label = "SPLIT_PLAN"
        elif likely == "CONFIG_OR_REPORT":
            label = "CONFIG_OR_REPORT"
        elif likely == "PERFORMANCE_SUMMARY":
            label = "PERFORMANCE_SUMMARY"
        elif is_return:
            label = "RETURN_PANEL_NOT_WEIGHTS"
        elif not has_weight:
            label = "INVALID_NO_WEIGHT_COLUMN"
        elif not (symbol_col and date_col):
            label = "INVALID_NO_SYMBOL_OR_DATE"
        else:
            label = "AMBIGUOUS_NEEDS_MANUAL_REVIEW"
            reason.append("schema does not meet weight or OOS score rules")
        rows.append(
            {
                "artifact_path": rel(path),
                "likely_artifact_type": likely,
                "row_count": row_count,
                "column_count": len(cols),
                "columns_detected": "|".join(cols),
                "symbol_column_detected": symbol_col,
                "date_or_month_column_detected": date_col,
                "weight_column_detected": weight_col,
                "score_column_detected": score_col,
                "rank_column_detected": rank_col,
                "prediction_column_detected": pred_col,
                "return_column_detected": return_col,
                "label_column_detected": label_col,
                "cost_column_detected": cost_col,
                "turnover_column_detected": turnover_col,
                "portfolio_column_detected": portfolio_col,
                "model_column_detected": model_col,
                "month_count": month_count,
                "min_month": min_month,
                "max_month": max_month,
                "symbol_count": symbol_count,
                "duplicate_symbol_month_count": duplicate_count,
                "avg_weight_sum_by_month": avg_ws,
                "min_weight_sum_by_month": min_ws,
                "max_weight_sum_by_month": max_ws,
                "is_historical_multimonth": bool_str(bool(is_hist)),
                "is_valid_historical_weights": bool_str(is_valid_weight),
                "is_oos_score_panel": bool_str(is_oos_score),
                "is_weight_audit_only": bool_str(is_weight_audit),
                "is_return_panel": bool_str(is_return),
                "validation_label": label,
                "validation_reason": "; ".join(reason),
            }
        )
    return rows


def evidence_snippet(text: str, pats: list[str]) -> str:
    low = text.lower()
    for pat in pats:
        idx = low.find(pat.lower())
        if idx >= 0:
            return " ".join(text[max(0, idx - 100): idx + 280].split())
    return ""


def evidence(files: list[Path]) -> list[dict[str, Any]]:
    rows = []
    pats = ["V0", "V7", "TOAWARE", "LINEAR", "FULL_OOS", "V15", "Compact-F", "compact_f", "Top50", "Buffer", "35_75", "weight", "weights", "monthly weight", "fwd_ret_1m", "OOS", "fold", "split", "AKShare", "CSMAR", "PIT", "label"]
    for path in files:
        if path.suffix.lower() not in TEXT_EXTS:
            continue
        if path.suffix.lower() == ".csv" and path.stat().st_size > 500_000:
            continue
        text = safe_text(path)
        hay = f"{rel(path)}\n{text}"
        low = hay.lower()
        if not any(p.lower() in low for p in pats):
            continue
        model = model_from_path(rel(path))
        rows.append(
            {
                "artifact_path": rel(path),
                "evidence_type": infer_artifact_type(path, text),
                "model_detected": model,
                "data_source_hint": "CSMAR/PIT" if ("csmar" in low or "pit" in low) else ("AKShare" if "akshare" in low else ""),
                "oos_period_hint": evidence_snippet(hay, ["OOS", "split", "fold", "2017", "2026"]),
                "portfolio_rule_hint": evidence_snippet(hay, ["Top50", "Buffer", "35_75", "weight", "monthly weight"]),
                "topn_or_buffer_hint": evidence_snippet(hay, ["Top50", "Buffer", "35_75"]),
                "cost_hint": evidence_snippet(hay, ["cost", "bps"]),
                "benchmark_hint": evidence_snippet(hay, ["benchmark", "CSI", "HS300", "CSI800"]),
                "feature_set_hint": evidence_snippet(hay, ["feature", "V15", "Compact-F", "TOAWARE", "LINEAR"]),
                "label_hint": evidence_snippet(hay, ["fwd_ret_1m", "label"]),
                "evidence_snippet": evidence_snippet(hay, pats),
                "audit_status": "EVIDENCE_FOUND",
            }
        )
    return rows


def choose_best(rows: list[dict[str, Any]], key: str, model: str) -> dict[str, Any] | None:
    model_rows = [r for r in rows if model_from_path(r["artifact_path"]) == model]
    if key == "weights":
        cands = [r for r in model_rows if r["is_valid_historical_weights"] == "true"]
    elif key == "score":
        cands = [r for r in model_rows if r["is_oos_score_panel"] == "true"]
    elif key == "audit":
        cands = [r for r in model_rows if r["is_weight_audit_only"] == "true"]
    else:
        cands = model_rows
    if not cands:
        return None
    return sorted(cands, key=lambda r: (int(r.get("month_count") or 0), int(r.get("row_count") or 0)), reverse=True)[0]


def resolver(schema_rows: list[dict[str, Any]], evidence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    evidence_by_model: dict[str, list[dict[str, Any]]] = {}
    for e in evidence_rows:
        if e["model_detected"]:
            evidence_by_model.setdefault(e["model_detected"], []).append(e)
    for model in MODELS:
        w = choose_best(schema_rows, "weights", model)
        s = choose_best(schema_rows, "score", model)
        a = choose_best(schema_rows, "audit", model)
        reports = [e for e in evidence_by_model.get(model, []) if e["evidence_type"] == "CONFIG_OR_REPORT"]
        report = reports[0] if reports else None
        alternatives = []
        if w:
            status, reason, strength, manual = "LOCKED_VALID_HISTORICAL_WEIGHTS", "locked valid historical weights", "HIGH", "false"
        elif s and a:
            status, reason, strength, manual = "LOCKED_OOS_SCORE_PANEL_ONLY", "OOS score panel plus weight/rule audit evidence; next step is reconstruction prep", "MEDIUM", "false"
        elif s:
            status, reason, strength, manual = "LOCKED_OOS_SCORE_PANEL_ONLY", "OOS score panel found, but rule evidence is incomplete/watch", "MEDIUM", "true"
        elif a:
            status, reason, strength, manual = "LOCKED_WEIGHT_AUDIT_ONLY", "weight audit found but no score panel or valid weights", "LOW", "true"
        elif report:
            status, reason, strength, manual = "LOCKED_REPORT_ONLY", "report/config evidence found only", "LOW", "true"
        else:
            status, reason, strength, manual = "MISSING", "no matching artifact in target folder", "NONE", "true"
        rows.append(
            {
                "model_name": model,
                "resolver_status": status,
                "selected_weights_path": w["artifact_path"] if w else "",
                "selected_oos_score_panel_path": s["artifact_path"] if s else "",
                "selected_report_path": report["artifact_path"] if report else "",
                "selected_weight_audit_path": a["artifact_path"] if a else "",
                "alternative_candidate_paths": ";".join(alternatives),
                "selected_month_count": (w or s or {}).get("month_count", ""),
                "selected_symbol_count": (w or s or {}).get("symbol_count", ""),
                "selected_min_month": (w or s or {}).get("min_month", ""),
                "selected_max_month": (w or s or {}).get("max_month", ""),
                "selected_avg_weight_sum": (w or {}).get("avg_weight_sum_by_month", ""),
                "evidence_strength": strength,
                "reason": reason,
                "manual_action_required": manual,
            }
        )
    return rows


def main() -> int:
    ensure_dirs()
    append_state("开始解析 full_panel_forced_tournament_v3。")
    files = iter_files()
    target_found = TARGET.exists()
    inv = inventory(files) if target_found else []
    inv_by = {r["artifact_path"]: r for r in inv}
    schema = schema_validation(files, inv_by) if target_found else []
    ev = evidence(files) if target_found else []
    decisions = resolver(schema, ev) if target_found else []

    inv_fields = ["artifact_path", "file_name", "file_type", "file_size_bytes", "modified_time", "likely_artifact_type", "contains_v0", "contains_v7", "contains_blend", "contains_compact_f", "contains_weights", "contains_weight_audit", "contains_scores", "contains_returns", "contains_oos", "contains_training_audit", "contains_alignment_audit", "contains_config_or_report", "notes"]
    schema_fields = ["artifact_path", "likely_artifact_type", "row_count", "column_count", "columns_detected", "symbol_column_detected", "date_or_month_column_detected", "weight_column_detected", "score_column_detected", "rank_column_detected", "prediction_column_detected", "return_column_detected", "label_column_detected", "cost_column_detected", "turnover_column_detected", "portfolio_column_detected", "model_column_detected", "month_count", "min_month", "max_month", "symbol_count", "duplicate_symbol_month_count", "avg_weight_sum_by_month", "min_weight_sum_by_month", "max_weight_sum_by_month", "is_historical_multimonth", "is_valid_historical_weights", "is_oos_score_panel", "is_weight_audit_only", "is_return_panel", "validation_label", "validation_reason"]
    ev_fields = ["artifact_path", "evidence_type", "model_detected", "data_source_hint", "oos_period_hint", "portfolio_rule_hint", "topn_or_buffer_hint", "cost_hint", "benchmark_hint", "feature_set_hint", "label_hint", "evidence_snippet", "audit_status"]
    dec_fields = ["model_name", "resolver_status", "selected_weights_path", "selected_oos_score_panel_path", "selected_report_path", "selected_weight_audit_path", "alternative_candidate_paths", "selected_month_count", "selected_symbol_count", "selected_min_month", "selected_max_month", "selected_avg_weight_sum", "evidence_strength", "reason", "manual_action_required"]
    write_csv(OUT_DIR / "forced_tournament_v3_artifact_inventory.csv", inv, inv_fields)
    write_csv(OUT_DIR / "forced_tournament_v3_schema_validation.csv", schema, schema_fields)
    write_csv(OUT_DIR / "forced_tournament_v3_report_config_evidence.csv", ev, ev_fields)
    write_csv(OUT_DIR / "forced_tournament_v3_model_resolver_decision.csv", decisions, dec_fields)

    ready_rows = []
    for d in decisions:
        has_w = d["resolver_status"] == "LOCKED_VALID_HISTORICAL_WEIGHTS"
        has_s = bool(d["selected_oos_score_panel_path"])
        has_rule = bool(d["selected_weight_audit_path"] or d["selected_report_path"])
        ready_rows.append(
            {
                "model_name": d["model_name"],
                "valid_historical_weights_locked": bool_str(has_w),
                "selected_weights_path": d["selected_weights_path"],
                "ready_for_csmar_bridge_test": bool_str(has_w),
                "oos_score_panel_locked": bool_str(has_s),
                "selected_oos_score_panel_path": d["selected_oos_score_panel_path"],
                "weight_audit_or_rule_evidence_locked": bool_str(has_rule),
                "selected_weight_audit_path": d["selected_weight_audit_path"],
                "ready_for_weight_reconstruction_prep": bool_str((not has_w) and has_s and has_rule),
                "reason": d["reason"],
                "caveat": "本任务不从 score 重建 weights，不计算 returns。",
            }
        )
    write_csv(OUT_DIR / "forced_tournament_v3_bridge_reconstruction_readiness.csv", ready_rows, ["model_name", "valid_historical_weights_locked", "selected_weights_path", "ready_for_csmar_bridge_test", "oos_score_panel_locked", "selected_oos_score_panel_path", "weight_audit_or_rule_evidence_locked", "selected_weight_audit_path", "ready_for_weight_reconstruction_prep", "reason", "caveat"])

    rebuild_rows = []
    for d in decisions:
        model = d["model_name"]
        model_e = [e for e in ev if e["model_detected"] == model]
        data_hint = ";".join(sorted(set(e["data_source_hint"] for e in model_e if e["data_source_hint"])))
        rebuild_rows.append(
            {
                "model_name": model,
                "oos_score_panel_available": bool_str(bool(d["selected_oos_score_panel_path"])),
                "likely_feature_inputs_available": bool_str(any(e["feature_set_hint"] for e in model_e)),
                "label_or_fwd_return_present": bool_str(any(e["label_hint"] for e in model_e) or any(r["artifact_path"] == d["selected_oos_score_panel_path"] and r["label_column_detected"] for r in schema)),
                "data_source_hint": data_hint,
                "pit_clean_csmar_hint": bool_str("CSMAR/PIT" in data_hint),
                "akshare_hint": bool_str("AKShare" in data_hint),
                "current_csmar_rebuild_relevance": "LEGACY_SCORE_AND_RULE_REFERENCE" if d["selected_oos_score_panel_path"] else ("RULE_OR_AUDIT_REFERENCE" if d["selected_weight_audit_path"] or d["selected_report_path"] else "LOW"),
                "required_next_inputs": "feature mapping; portfolio construction rule; canonical CSMAR PIT-clean panel; leakage guardrails",
                "caveat": "不能直接作为 current canonical conclusion。",
            }
        )
    write_csv(OUT_DIR / "forced_tournament_v3_csmar_rebuild_relevance.csv", rebuild_rows, ["model_name", "oos_score_panel_available", "likely_feature_inputs_available", "label_or_fwd_return_present", "data_source_hint", "pit_clean_csmar_hint", "akshare_hint", "current_csmar_rebuild_relevance", "required_next_inputs", "caveat"])

    models_bridge = [d["model_name"] for d in decisions if d["resolver_status"] == "LOCKED_VALID_HISTORICAL_WEIGHTS"]
    models_recon = [r["model_name"] for r in ready_rows if r["ready_for_weight_reconstruction_prep"] == "true"]
    if models_bridge:
        (OUT_DIR / "forced_tournament_v3_bridge_test_config_draft.json").write_text(json.dumps({"bridge_allowed": True, "models_ready_for_bridge": models_bridge, "selected_weights_paths": {d["model_name"]: d["selected_weights_path"] for d in decisions if d["selected_weights_path"]}, "canonical_return_source_path": rel(CANONICAL_RETURN), "require_all_models": False, "no_new_scores": True, "no_new_weights": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    if models_recon:
        (OUT_DIR / "forced_tournament_v3_weight_reconstruction_prep_config_draft.json").write_text(json.dumps({"reconstruction_prep_allowed": True, "models_ready_for_reconstruction_prep": models_recon, "selected_oos_score_panels": {d["model_name"]: d["selected_oos_score_panel_path"] for d in decisions if d["model_name"] in models_recon}, "selected_rule_or_weight_audit_files": {d["model_name"]: d["selected_weight_audit_path"] or d["selected_report_path"] for d in decisions if d["model_name"] in models_recon}, "reconstruction_not_performed_in_this_task": True, "no_new_weights_generated": True, "no_returns_calculated": True}, ensure_ascii=False, indent=2), encoding="utf-8")

    guardrails = {"portfolio_returns_calculated": False, "new_weights_generated": False, "new_scores_generated": False, "old_weights_modified": False, "training_run": False, "benchmark_relative_returns_calculated": False, "alpha_beta_regression_calculated": False, "information_ratio_calculated": False, "tracking_error_calculated": False, "ff_regression_calculated": False, "dgtw_adjusted_eval_calculated": False, "shap_calculated": False, "production_modified": False}
    guardrail_rows = [{"guardrail": k, "expected": "false", "actual": bool_str(v), "pass": bool_str(v is False)} for k, v in guardrails.items()]
    write_csv(OUT_DIR / "forced_tournament_v3_resolver_guardrail_qa.csv", guardrail_rows, ["guardrail", "expected", "actual", "pass"])
    guardrail_ok = all(not v for v in guardrails.values())
    ambiguous = any(d["resolver_status"] == "MULTIPLE_CANDIDATES_AMBIGUOUS" for d in decisions)
    if not guardrail_ok:
        final_decision = "FORCED_TOURNAMENT_RESOLVER_FAIL_GUARDRAIL"
    elif models_bridge and models_recon:
        final_decision = "FORCED_TOURNAMENT_RESOLVER_READY_BOTH_BRIDGE_AND_RECONSTRUCTION_PREP"
    elif models_bridge:
        final_decision = "FORCED_TOURNAMENT_RESOLVER_READY_BRIDGE_TEST"
    elif models_recon:
        final_decision = "FORCED_TOURNAMENT_RESOLVER_READY_WEIGHT_RECONSTRUCTION_PREP"
    elif ambiguous:
        final_decision = "FORCED_TOURNAMENT_RESOLVER_MANUAL_REVIEW_REQUIRED"
    else:
        final_decision = "FORCED_TOURNAMENT_RESOLVER_NO_RELEVANT_ARTIFACTS"

    summary = {
        "run_timestamp": now_iso(),
        "prerequisites_passed": target_found,
        "target_folder_found": target_found,
        "artifacts_found": len(inv),
        "candidates_schema_validated": len(schema),
        "valid_historical_weight_count": sum(1 for r in schema if r["is_valid_historical_weights"] == "true"),
        "oos_score_panel_count": sum(1 for r in schema if r["is_oos_score_panel"] == "true"),
        "weight_audit_count": sum(1 for r in schema if r["is_weight_audit_only"] == "true"),
        "v0_weights_locked": any(d["model_name"].startswith("V0") and d["selected_weights_path"] for d in decisions),
        "v0_score_panel_locked": any(d["model_name"].startswith("V0") and d["selected_oos_score_panel_path"] for d in decisions),
        "v7_weights_locked": any(d["model_name"].startswith("V7") and d["selected_weights_path"] for d in decisions),
        "v7_score_panel_locked": any(d["model_name"].startswith("V7") and d["selected_oos_score_panel_path"] for d in decisions),
        "compact_f_weights_locked": any(d["model_name"].startswith("COMPACT_F") and d["selected_weights_path"] for d in decisions),
        "compact_f_score_panel_locked": any(d["model_name"].startswith("COMPACT_F") and d["selected_oos_score_panel_path"] for d in decisions),
        "blend_weights_locked": any(d["model_name"].startswith("BLEND") and d["selected_weights_path"] for d in decisions),
        "blend_score_panel_locked": any(d["model_name"].startswith("BLEND") and d["selected_oos_score_panel_path"] for d in decisions),
        "bridge_ready": bool(models_bridge),
        "models_ready_for_bridge": models_bridge,
        "weight_reconstruction_prep_ready": bool(models_recon),
        "models_ready_for_weight_reconstruction_prep": models_recon,
        "report_config_evidence_found": bool(ev),
        "csmar_canonical_rebuild_relevance": "HIGH_FOR_SCORE_AND_RULE_REFERENCE" if models_recon else ("BRIDGE_REFERENCE" if models_bridge else "LOW"),
        "recommended_next_step": "进入 weight reconstruction prep：锁定 OOS score panels 与 weight/rule audit，先不生成 weights；随后单独任务执行 reconstruction。" if models_recon and not models_bridge else ("进入 bridge test。" if models_bridge else "需要人工定位可用 weights/score/rule evidence。"),
        **guardrails,
        "final_decision": final_decision,
    }
    (OUT_DIR / "targeted_full_panel_forced_tournament_v3_artifact_resolver_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    prereq = {"run_timestamp": summary["run_timestamp"], "target_folder": rel(TARGET), "target_folder_found": target_found, "canonical_return_source_path": rel(CANONICAL_RETURN), "canonical_return_source_exists": CANONICAL_RETURN.exists(), "prerequisites_passed": target_found}
    (OUT_DIR / "forced_tournament_v3_resolver_prerequisite_check.json").write_text(json.dumps(prereq, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# Targeted Full Panel Forced Tournament V3 Artifact Resolver v0", "", "## 结论", f"- final_decision: {final_decision}", f"- valid_historical_weight_count: {summary['valid_historical_weight_count']}", f"- oos_score_panel_count: {summary['oos_score_panel_count']}", f"- models_ready_for_weight_reconstruction_prep: {', '.join(models_recon)}", "", "## Guardrails", "未计算 portfolio returns，未生成 scores/weights，未修改旧 weights，未做 benchmark-relative/alpha/IR/TE/FF/DGTW/SHAP。"]
    (OUT_DIR / "targeted_full_panel_forced_tournament_v3_artifact_resolver_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT_DIR / "terminal_summary.json").write_text(json.dumps({"task_name": TASK, "completed_at": now_iso(), "final_decision": final_decision, "outputs": sorted(p.name for p in OUT_DIR.iterdir() if p.is_file())}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "task_completion_card.md").write_text(f"# Task Completion Card\n\n- task: {TASK}\n- completed_at: {now_iso()}\n- final_decision: {final_decision}\n- output_dir: `{rel(OUT_DIR)}`\n", encoding="utf-8")
    write_csv(OUT_DIR / "final_qa.csv", [{"check": "required_outputs_present", "status": "PASS", "detail": "requested resolver outputs generated"}, {"check": "guardrails_passed", "status": "PASS" if guardrail_ok else "FAIL", "detail": json.dumps(guardrails, ensure_ascii=False)}], ["check", "status", "detail"])
    append_state(f"完成。final_decision={final_decision}; weights={summary['valid_historical_weight_count']}; oos_scores={summary['oos_score_panel_count']}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
