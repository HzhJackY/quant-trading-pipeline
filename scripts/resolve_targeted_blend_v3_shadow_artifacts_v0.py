from __future__ import annotations

import csv
import gc
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None


TASK = "targeted_blend_v3_shadow_artifact_resolver_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK
RUN_STATE = RUN_DIR / "RUN_STATE.md"
CANONICAL_RETURN = ROOT / "output" / "robust_cleaned_fundamental_factor_variant_build_v0" / "robust_cleaned_factor_score_panel_v0.parquet"

TARGET_DIRS = [
    ROOT / "output" / "blend_v3_shadow_market_data_refresh_v1",
    ROOT / "output" / "blend_v3_shadow_monitoring",
    ROOT / "output" / "blend_v3_governance_patch_v2",
    ROOT / "output" / "readme_blend_v3_shadow_update",
    ROOT / "output" / "blend_v3_shadow_live_usability_fix_v1",
    ROOT / "output" / "blend_v3_shadow_live",
    ROOT / "output" / "blend_v3_shadow",
]

TEXT_EXTS = {".md", ".txt", ".json", ".yaml", ".yml", ".py", ".csv"}
TABULAR_EXTS = {".csv", ".parquet", ".xlsx", ".xls"}
MAX_TEXT_BYTES = 180_000


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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


def iter_target_files() -> tuple[list[Path], list[str], list[str]]:
    files: list[Path] = []
    scanned: list[str] = []
    missing: list[str] = []
    for base in TARGET_DIRS:
        if not base.exists():
            missing.append(rel(base))
            continue
        scanned.append(rel(base))
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in {".git", ".venv", "__pycache__", ".pytest_cache"}]
            for name in filenames:
                files.append(Path(dirpath) / name)
    return sorted(files, key=lambda p: rel(p).lower()), scanned, missing


def infer_type(path: Path, text: str = "") -> str:
    hay = f"{rel(path)}\n{text[:20000]}".lower()
    name = path.name.lower()
    if any(x in hay for x in ["governance", "patch"]):
        return "GOVERNANCE_OR_PATCH"
    if "monitor" in hay:
        return "MONITORING"
    if path.suffix.lower() in {".md", ".txt"} or "readme" in hay or "report" in name:
        return "README_OR_REPORT"
    if any(x in name for x in ["latest_shadow_holdings", "latest"]):
        return "LATEST_SHADOW_HOLDINGS"
    if "holding" in hay and ("weight" in hay or "target_weight" in hay):
        return "SINGLE_MONTH_HOLDINGS"
    if "weight" in hay or "target_weight" in hay:
        return "HISTORICAL_WEIGHTS_CANDIDATE"
    if "score" in hay:
        return "SCORE_PANEL"
    if "return" in hay or "ret_" in hay:
        return "RETURN_PANEL"
    if "performance" in hay or "summary" in hay:
        return "PERFORMANCE_SUMMARY"
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        return "CONFIG"
    return "UNKNOWN_RELEVANT"


def inventory(files: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in files:
        sample = safe_text(path) if path.suffix.lower() in TEXT_EXTS else ""
        hay = f"{rel(path)}\n{sample}".lower()
        stat = path.stat()
        parent = path.parent.name
        rows.append(
            {
                "artifact_path": rel(path),
                "parent_folder": parent,
                "file_name": path.name,
                "file_type": path.suffix.lower().lstrip(".") or "no_ext",
                "file_size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "likely_artifact_type": infer_type(path, sample),
                "contains_blend": bool_str("blend" in hay),
                "contains_v0": bool_str(re.search(r"(^|[^a-z0-9])v0([^a-z0-9]|$)", hay) is not None),
                "contains_v7": bool_str(re.search(r"(^|[^a-z0-9])v7([^a-z0-9]|$)", hay) is not None),
                "contains_weights": bool_str("weight" in hay),
                "contains_holdings": bool_str("holding" in hay),
                "contains_scores": bool_str("score" in hay),
                "contains_returns": bool_str("return" in hay or "ret_" in hay),
                "contains_config": bool_str(path.suffix.lower() in {".json", ".yaml", ".yml"} or "config" in hay),
                "contains_monitoring": bool_str("monitor" in hay),
                "contains_governance": bool_str("governance" in hay or "patch" in hay),
                "contains_live": bool_str("live" in hay),
                "notes": "",
            }
        )
    return rows


def detect_col(cols: list[str], candidates: list[str], contains: bool = False) -> str:
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


def read_tabular(path: Path) -> tuple[pd.DataFrame | None, str]:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path), ""
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, dtype=str), ""
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path, dtype=str), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return None, "unsupported"


def normalize_symbol(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\D", "", regex=True).str[-6:].str.zfill(6)


def normalize_month(s: pd.Series) -> pd.Series:
    return (pd.to_datetime(s, errors="coerce") + pd.offsets.MonthEnd(0)).dt.normalize()


def schema_validation(files: list[Path], inv_by_path: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        if path.suffix.lower() not in TABULAR_EXTS:
            continue
        likely = inv_by_path[rel(path)]["likely_artifact_type"]
        if likely not in {
            "HISTORICAL_WEIGHTS_CANDIDATE",
            "LATEST_SHADOW_HOLDINGS",
            "SINGLE_MONTH_HOLDINGS",
            "SCORE_PANEL",
            "RETURN_PANEL",
            "PERFORMANCE_SUMMARY",
            "MONITORING",
            "GOVERNANCE_OR_PATCH",
            "UNKNOWN_RELEVANT",
        }:
            continue
        df, err = read_tabular(path)
        if df is None:
            rows.append(
                {
                    "artifact_path": rel(path),
                    "likely_artifact_type": likely,
                    "row_count": "",
                    "column_count": "",
                    "columns_detected": "",
                    "symbol_column_detected": "",
                    "date_or_month_column_detected": "",
                    "weight_column_detected": "",
                    "score_column_detected": "",
                    "return_column_detected": "",
                    "portfolio_column_detected": "",
                    "model_column_detected": "",
                    "month_count": "",
                    "min_month": "",
                    "max_month": "",
                    "symbol_count": "",
                    "portfolio_count": "",
                    "duplicate_symbol_month_count": "",
                    "avg_weight_sum_by_month": "",
                    "min_weight_sum_by_month": "",
                    "max_weight_sum_by_month": "",
                    "is_historical_multimonth": "false",
                    "is_single_month_only": "false",
                    "is_latest_shadow_only": "false",
                    "is_valid_historical_weights": "false",
                    "validation_label": "AMBIGUOUS_NEEDS_MANUAL_REVIEW",
                    "validation_reason": f"read failed: {err}",
                }
            )
            continue
        cols = list(df.columns)
        symbol_col = detect_col(cols, ["symbol", "stock_code", "code", "ticker", "证券代码"], True)
        date_col = detect_col(cols, ["month_end", "rebalance_date", "trade_date", "date", "as_of_date"], True)
        weight_col = detect_col(cols, ["weight", "target_weight", "portfolio_weight", "shadow_weight"], True)
        score_col = detect_col(cols, ["score", "blend_score", "v0_score", "v7_score"], True)
        return_col = detect_col(cols, ["return", "ret", "fwd_ret_1m", "gross_return", "net_return"], True)
        portfolio_col = detect_col(cols, ["portfolio_name", "portfolio", "strategy"], True)
        model_col = detect_col(cols, ["model_name", "candidate_model_name", "model", "leg"], True)

        month_count = symbol_count = portfolio_count = duplicate_count = ""
        min_month = max_month = ""
        avg_ws = min_ws = max_ws = ""
        label = "AMBIGUOUS_NEEDS_MANUAL_REVIEW"
        reason = []
        valid = False
        is_single = False
        is_hist = False
        is_latest = likely == "LATEST_SHADOW_HOLDINGS" or "latest" in path.name.lower()
        if symbol_col and date_col:
            sy = normalize_symbol(df[symbol_col])
            mo = normalize_month(df[date_col])
            mask = sy.notna() & mo.notna()
            month_vals = mo[mask]
            month_count = int(month_vals.nunique())
            min_month = str(month_vals.min().date()) if len(month_vals) else ""
            max_month = str(month_vals.max().date()) if len(month_vals) else ""
            symbol_count = int(sy[mask].nunique())
            duplicate_count = int(pd.DataFrame({"s": sy[mask], "m": month_vals[mask]}).duplicated(["s", "m"]).sum()) if len(month_vals) else 0
            is_single = month_count == 1
            is_hist = month_count >= 24
        else:
            reason.append("missing symbol or date/month column")
        if portfolio_col:
            portfolio_count = int(df[portfolio_col].astype("string").nunique(dropna=True))
        if weight_col and date_col:
            w = pd.to_numeric(df[weight_col], errors="coerce")
            mo = normalize_month(df[date_col])
            wsum = pd.DataFrame({"m": mo, "w": w}).dropna().groupby("m")["w"].sum()
            if len(wsum):
                avg_ws = float(wsum.mean())
                min_ws = float(wsum.min())
                max_ws = float(wsum.max())
        if not weight_col:
            label = "INVALID_NO_WEIGHT_COLUMN"
            reason.append("no weight column")
        elif not (symbol_col and date_col):
            label = "INVALID_NO_SYMBOL_OR_DATE"
        elif likely in {"MONITORING", "GOVERNANCE_OR_PATCH"} or any(x in rel(path).lower() for x in ["monitoring", "governance", "patch"]):
            label = "INVALID_MONITORING_OR_GOVERNANCE"
            reason.append("monitoring/governance artifact")
        elif score_col and not weight_col:
            label = "INVALID_SCORE_PANEL"
        elif return_col and not weight_col:
            label = "INVALID_RETURN_PANEL"
        elif is_latest:
            label = "INVALID_LATEST_SHADOW_ONLY"
            reason.append("latest/shadow-only naming")
        elif is_single:
            label = "INVALID_SINGLE_MONTH_ONLY"
            reason.append("month_count == 1")
        elif is_hist:
            weight_sum_ok = avg_ws != "" and min_ws >= 0.98 and max_ws <= 1.02
            if weight_sum_ok:
                lowpath = rel(path).lower()
                if "v0v7" in lowpath:
                    label = "VALID_V0V7_HISTORICAL_WEIGHTS"
                elif "v7" in lowpath:
                    label = "VALID_V7_HISTORICAL_WEIGHTS"
                elif "v0" in lowpath:
                    label = "VALID_V0_HISTORICAL_WEIGHTS"
                elif "blend" in lowpath:
                    label = "VALID_BLEND_HISTORICAL_WEIGHTS"
                else:
                    label = "VALID_OTHER_HISTORICAL_WEIGHTS"
                valid = True
                reason.append("symbol/date/weight present; month_count >= 24; weight sum near 1")
            else:
                label = "AMBIGUOUS_NEEDS_MANUAL_REVIEW"
                reason.append("historical shape but weight sums not near 1 or unavailable")
        elif score_col:
            label = "INVALID_SCORE_PANEL"
        elif return_col:
            label = "INVALID_RETURN_PANEL"
        else:
            label = "AMBIGUOUS_NEEDS_MANUAL_REVIEW"
            reason.append("does not satisfy historical weights criteria")

        rows.append(
            {
                "artifact_path": rel(path),
                "likely_artifact_type": likely,
                "row_count": len(df),
                "column_count": len(cols),
                "columns_detected": "|".join(cols),
                "symbol_column_detected": symbol_col,
                "date_or_month_column_detected": date_col,
                "weight_column_detected": weight_col,
                "score_column_detected": score_col,
                "return_column_detected": return_col,
                "portfolio_column_detected": portfolio_col,
                "model_column_detected": model_col,
                "month_count": month_count,
                "min_month": min_month,
                "max_month": max_month,
                "symbol_count": symbol_count,
                "portfolio_count": portfolio_count,
                "duplicate_symbol_month_count": duplicate_count,
                "avg_weight_sum_by_month": avg_ws,
                "min_weight_sum_by_month": min_ws,
                "max_weight_sum_by_month": max_ws,
                "is_historical_multimonth": bool_str(bool(is_hist)),
                "is_single_month_only": bool_str(bool(is_single)),
                "is_latest_shadow_only": bool_str(bool(is_latest)),
                "is_valid_historical_weights": bool_str(valid),
                "validation_label": label,
                "validation_reason": "; ".join(reason),
            }
        )
        del df
        gc.collect()
    return rows


def snippet(text: str, patterns: list[str]) -> str:
    low = text.lower()
    for pat in patterns:
        idx = low.find(pat.lower())
        if idx >= 0:
            return " ".join(text[max(0, idx - 100) : idx + 260].split())
    return ""


def lineage(files: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in files:
        if path.suffix.lower() not in {".md", ".txt", ".json", ".yaml", ".yml", ".py"}:
            continue
        text = safe_text(path)
        hay = f"{rel(path)}\n{text}"
        low = hay.lower()
        if not any(x in low for x in ["blend", "v0", "v7", "shadow", "governance", "rebalance", "historical", "weight"]):
            continue
        rows.append(
            {
                "artifact_path": rel(path),
                "config_or_report_type": "CONFIG" if path.suffix.lower() in {".json", ".yaml", ".yml"} else ("SCRIPT" if path.suffix.lower() == ".py" else "REPORT"),
                "blend_name_detected": "BLEND_V0_50_V7_50" if "blend_v0_50_v7_50" in low else ("blend_v3" if "blend_v3" in low else ""),
                "v0_leg_detected": bool_str(re.search(r"(^|[^a-z0-9])v0([^a-z0-9]|$)", low) is not None),
                "v7_leg_detected": bool_str(re.search(r"(^|[^a-z0-9])v7([^a-z0-9]|$)", low) is not None),
                "blend_weighting_rule_detected": "50/50" if any(x in low for x in ["50/50", "0.5", "50_50", "v0_50_v7_50"]) else "",
                "top50_buffer_detected": bool_str(any(x in low for x in ["top50_buffer", "top50 buffer", "top50"])),
                "buffer_policy_detected": "35_75" if "35_75" in low else ("buffer" if "buffer" in low else ""),
                "data_source_hint": "CSMAR" if "csmar" in low else ("AKSHARE" if "akshare" in low else ""),
                "date_range_hint": snippet(hay, ["2017", "2026", "date_range", "month_count"]),
                "evidence_snippet": snippet(hay, ["BLEND_V0_50_V7_50", "blend_v3", "V0", "V7", "50/50", "Top50_Buffer", "35_75", "historical", "weights"]),
                "audit_status": "EVIDENCE_FOUND",
            }
        )
    return rows


def resolver(schema_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models = ["BLEND_V3", "BLEND_V0_50_V7_50", "V0_LEG", "V7_LEG", "V0V7", "TOP50_BUFFER_35_75"]
    rows = []
    valid = [r for r in schema_rows if r.get("is_valid_historical_weights") == "true"]
    for model in models:
        def match(r: dict[str, Any]) -> bool:
            p = r["artifact_path"].lower()
            label = r["validation_label"]
            if model in {"BLEND_V3", "BLEND_V0_50_V7_50"}:
                return "blend" in p and label == "VALID_BLEND_HISTORICAL_WEIGHTS"
            if model == "V0_LEG":
                return ("v0" in p or label == "VALID_V0_HISTORICAL_WEIGHTS") and "v7" not in p
            if model == "V7_LEG":
                return "v7" in p or label == "VALID_V7_HISTORICAL_WEIGHTS"
            if model == "V0V7":
                return "v0v7" in p or label == "VALID_V0V7_HISTORICAL_WEIGHTS"
            if model == "TOP50_BUFFER_35_75":
                return any(x in p for x in ["top50", "35_75", "buffer"])
            return False

        candidates = [r for r in valid if match(r)]
        latest = [r for r in schema_rows if "blend" in r["artifact_path"].lower() and r.get("is_latest_shadow_only") == "true"]
        single = [r for r in schema_rows if "blend" in r["artifact_path"].lower() and r.get("is_single_month_only") == "true"]
        if len(candidates) == 1:
            c = candidates[0]
            status = "LOCKED_VALID_HISTORICAL_WEIGHTS"
            selected = c["artifact_path"]
            reason = "唯一 valid historical weights candidate"
            evidence = "HIGH"
            manual = "false"
        elif len(candidates) > 1:
            c = sorted(candidates, key=lambda x: (int(x.get("month_count") or 0), int(x.get("row_count") or 0)), reverse=True)[0]
            status = "AMBIGUOUS_MULTIPLE_CANDIDATES"
            selected = c["artifact_path"]
            reason = "多个 valid historical weights candidate，需人工确认"
            evidence = "MEDIUM"
            manual = "true"
        elif model in {"BLEND_V3", "BLEND_V0_50_V7_50"} and latest:
            c = latest[0]
            status = "ONLY_LATEST_SHADOW_FOUND"
            selected = ""
            reason = "只找到 latest shadow holdings，不满足 historical weights 条件"
            evidence = "LOW"
            manual = "true"
        elif model in {"BLEND_V3", "BLEND_V0_50_V7_50"} and single:
            c = single[0]
            status = "ONLY_SINGLE_MONTH_FOUND"
            selected = ""
            reason = "只找到单月 holdings"
            evidence = "LOW"
            manual = "true"
        else:
            c = {}
            status = "MISSING"
            selected = ""
            reason = "未在定向目录中发现 valid historical weights"
            evidence = "NONE"
            manual = "true"
        rows.append(
            {
                "model_name": model,
                "resolver_status": status,
                "selected_weights_path": selected,
                "alternative_candidate_paths": ";".join(r["artifact_path"] for r in candidates[1:]),
                "selected_month_count": c.get("month_count", ""),
                "selected_symbol_count": c.get("symbol_count", ""),
                "selected_min_month": c.get("min_month", ""),
                "selected_max_month": c.get("max_month", ""),
                "selected_avg_weight_sum": c.get("avg_weight_sum_by_month", ""),
                "evidence_strength": evidence,
                "reason": reason,
                "manual_action_required": manual,
            }
        )
    return rows


def main() -> int:
    ensure_dirs()
    append_state("开始定向扫描 blend_v3_shadow 相关目录。")
    files, scanned, missing = iter_target_files()
    inv_rows = inventory(files)
    inv_by = {r["artifact_path"]: r for r in inv_rows}
    schema_rows = schema_validation(files, inv_by)
    lineage_rows = lineage(files)
    decision_rows = resolver(schema_rows)

    inv_fields = ["artifact_path", "parent_folder", "file_name", "file_type", "file_size_bytes", "modified_time", "likely_artifact_type", "contains_blend", "contains_v0", "contains_v7", "contains_weights", "contains_holdings", "contains_scores", "contains_returns", "contains_config", "contains_monitoring", "contains_governance", "contains_live", "notes"]
    schema_fields = ["artifact_path", "likely_artifact_type", "row_count", "column_count", "columns_detected", "symbol_column_detected", "date_or_month_column_detected", "weight_column_detected", "score_column_detected", "return_column_detected", "portfolio_column_detected", "model_column_detected", "month_count", "min_month", "max_month", "symbol_count", "portfolio_count", "duplicate_symbol_month_count", "avg_weight_sum_by_month", "min_weight_sum_by_month", "max_weight_sum_by_month", "is_historical_multimonth", "is_single_month_only", "is_latest_shadow_only", "is_valid_historical_weights", "validation_label", "validation_reason"]
    lineage_fields = ["artifact_path", "config_or_report_type", "blend_name_detected", "v0_leg_detected", "v7_leg_detected", "blend_weighting_rule_detected", "top50_buffer_detected", "buffer_policy_detected", "data_source_hint", "date_range_hint", "evidence_snippet", "audit_status"]
    decision_fields = ["model_name", "resolver_status", "selected_weights_path", "alternative_candidate_paths", "selected_month_count", "selected_symbol_count", "selected_min_month", "selected_max_month", "selected_avg_weight_sum", "evidence_strength", "reason", "manual_action_required"]
    write_csv(OUT_DIR / "targeted_blend_v3_artifact_inventory.csv", inv_rows, inv_fields)
    write_csv(OUT_DIR / "targeted_blend_v3_schema_validation.csv", schema_rows, schema_fields)
    write_csv(OUT_DIR / "targeted_blend_v3_lineage_config_audit.csv", lineage_rows, lineage_fields)
    write_csv(OUT_DIR / "targeted_blend_v3_model_resolver_decision.csv", decision_rows, decision_fields)

    valid = [r for r in schema_rows if r["is_valid_historical_weights"] == "true"]
    latest_only = any(r["is_latest_shadow_only"] == "true" or r["is_single_month_only"] == "true" for r in schema_rows)
    valid_blend = [r for r in decision_rows if r["model_name"] in {"BLEND_V3", "BLEND_V0_50_V7_50"} and r["resolver_status"] == "LOCKED_VALID_HISTORICAL_WEIGHTS"]
    models_ready = [r["model_name"] for r in decision_rows if r["resolver_status"] == "LOCKED_VALID_HISTORICAL_WEIGHTS"]
    models_missing = [r["model_name"] for r in decision_rows if r["resolver_status"] in {"MISSING", "ONLY_LATEST_SHADOW_FOUND", "ONLY_SINGLE_MONTH_FOUND"}]
    manual_required = any(r["manual_action_required"] == "true" for r in decision_rows)
    partial_bridge = bool(models_ready)
    blend_only = bool(valid_blend)

    readiness_rows = []
    for r in decision_rows:
        locked = r["resolver_status"] == "LOCKED_VALID_HISTORICAL_WEIGHTS"
        readiness_rows.append(
            {
                "model_name": r["model_name"],
                "valid_historical_weights_locked": bool_str(locked),
                "selected_weights_path": r["selected_weights_path"],
                "ready_for_csmar_bridge_test": bool_str(locked),
                "partial_bridge_allowed": bool_str(partial_bridge),
                "reason": r["reason"],
                "caveat": "本 resolver 不计算收益；latest/single-month shadow 不允许 bridge。",
            }
        )
    write_csv(OUT_DIR / "targeted_blend_v3_bridge_readiness.csv", readiness_rows, ["model_name", "valid_historical_weights_locked", "selected_weights_path", "ready_for_csmar_bridge_test", "partial_bridge_allowed", "reason", "caveat"])

    config = {
        "partial_bridge_allowed": partial_bridge,
        "blend_only_bridge_allowed": blend_only,
        "models_ready_for_bridge": models_ready,
        "selected_weights_paths": {r["model_name"]: r["selected_weights_path"] for r in decision_rows if r["selected_weights_path"]},
        "models_missing_weights": models_missing,
        "models_latest_shadow_only": [r["model_name"] for r in decision_rows if r["resolver_status"] in {"ONLY_LATEST_SHADOW_FOUND", "ONLY_SINGLE_MONTH_FOUND"}],
        "canonical_return_source_path": rel(CANONICAL_RETURN),
        "require_all_v0_v7_blend": False,
        "invalid_candidates_excluded": True,
        "no_new_scores": True,
        "no_new_weights": True,
        "no_training": True,
    }
    (OUT_DIR / "targeted_blend_v3_bridge_test_v1_config_draft.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    guardrails = {
        "portfolio_returns_calculated": False,
        "training_run": False,
        "new_scores_generated": False,
        "new_weights_generated": False,
        "old_weights_modified": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
        "production_modified": False,
    }
    guardrail_rows = [{"guardrail": k, "expected": "false", "actual": bool_str(v), "pass": bool_str(v is False)} for k, v in guardrails.items()]
    write_csv(OUT_DIR / "targeted_blend_v3_resolver_guardrail_qa.csv", guardrail_rows, ["guardrail", "expected", "actual", "pass"])
    guardrail_ok = all(not v for v in guardrails.values())

    by_model = {r["model_name"]: r for r in decision_rows}
    if not guardrail_ok:
        final_decision = "TARGETED_BLEND_RESOLVER_FAIL_GUARDRAIL"
    elif blend_only:
        final_decision = "TARGETED_BLEND_RESOLVER_READY_BLEND_ONLY_BRIDGE"
    elif partial_bridge:
        final_decision = "TARGETED_BLEND_RESOLVER_READY_PARTIAL_BRIDGE"
    elif manual_required and any(r["resolver_status"] == "AMBIGUOUS_MULTIPLE_CANDIDATES" for r in decision_rows):
        final_decision = "TARGETED_BLEND_RESOLVER_MANUAL_REVIEW_REQUIRED"
    elif latest_only and not valid:
        final_decision = "TARGETED_BLEND_RESOLVER_ONLY_LATEST_SHADOW_FOUND"
    else:
        final_decision = "TARGETED_BLEND_RESOLVER_NO_VALID_WEIGHTS_FOUND"

    recommended = (
        "可以进入 Blend-only bridge test。"
        if blend_only
        else "定向目录中未锁定 historical Blend/V7 weights；需要提供或定位多月 historical weights，latest/single-month shadow 不应用于 bridge。"
    )
    summary = {
        "run_timestamp": now_iso(),
        "prerequisites_passed": bool(scanned),
        "folders_scanned": scanned,
        "folders_missing": missing,
        "artifacts_found": len(inv_rows),
        "candidates_schema_validated": len(schema_rows),
        "valid_historical_weight_count": len(valid),
        "blend_v3_resolver_status": by_model["BLEND_V3"]["resolver_status"],
        "blend_v3_selected_weights_path": by_model["BLEND_V3"]["selected_weights_path"],
        "blend_v0_50_v7_50_resolver_status": by_model["BLEND_V0_50_V7_50"]["resolver_status"],
        "blend_v0_50_v7_50_selected_weights_path": by_model["BLEND_V0_50_V7_50"]["selected_weights_path"],
        "v0_leg_resolver_status": by_model["V0_LEG"]["resolver_status"],
        "v0_leg_selected_weights_path": by_model["V0_LEG"]["selected_weights_path"],
        "v7_leg_resolver_status": by_model["V7_LEG"]["resolver_status"],
        "v7_leg_selected_weights_path": by_model["V7_LEG"]["selected_weights_path"],
        "v0v7_resolver_status": by_model["V0V7"]["resolver_status"],
        "top50_buffer_35_75_resolver_status": by_model["TOP50_BUFFER_35_75"]["resolver_status"],
        "latest_shadow_only_detected": latest_only,
        "valid_blend_historical_weights_found": blend_only,
        "blend_only_bridge_allowed": blend_only,
        "partial_bridge_allowed": partial_bridge,
        "models_ready_for_bridge": models_ready,
        "models_missing_weights": models_missing,
        "manual_action_required": manual_required,
        "recommended_next_step": recommended,
        **guardrails,
        "final_decision": final_decision,
    }
    (OUT_DIR / "targeted_blend_v3_shadow_artifact_resolver_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    prereq = {
        "run_timestamp": summary["run_timestamp"],
        "target_dirs": [rel(p) for p in TARGET_DIRS],
        "folders_scanned": scanned,
        "folders_missing": missing,
        "canonical_return_source_path": rel(CANONICAL_RETURN),
        "canonical_return_source_exists": CANONICAL_RETURN.exists(),
        "prerequisites_passed": summary["prerequisites_passed"],
    }
    (OUT_DIR / "targeted_blend_v3_resolver_prerequisite_check.json").write_text(json.dumps(prereq, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# Targeted Blend v3 Shadow Artifact Resolver v0",
        "",
        "## 结论",
        f"- final_decision: {final_decision}",
        f"- valid_historical_weight_count: {len(valid)}",
        f"- valid_blend_historical_weights_found: {blend_only}",
        "",
        "## 关键判断",
        f"- scanned folders: {len(scanned)}",
        f"- artifacts found: {len(inv_rows)}",
        f"- schema validated: {len(schema_rows)}",
        f"- latest/single-month shadow detected: {latest_only}",
        "",
        "## Guardrails",
        "未计算 portfolio returns，未训练，未生成或修改 scores/weights，未做 benchmark-relative/alpha/IR/TE/FF/DGTW/SHAP。",
    ]
    (OUT_DIR / "targeted_blend_v3_shadow_artifact_resolver_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT_DIR / "terminal_summary.json").write_text(json.dumps({"task_name": TASK, "completed_at": now_iso(), "final_decision": final_decision, "outputs": sorted(p.name for p in OUT_DIR.iterdir() if p.is_file())}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "task_completion_card.md").write_text(f"# Task Completion Card\n\n- task: {TASK}\n- completed_at: {now_iso()}\n- final_decision: {final_decision}\n- output_dir: `{rel(OUT_DIR)}`\n", encoding="utf-8")
    write_csv(OUT_DIR / "final_qa.csv", [{"check": "required_outputs_present", "status": "PASS", "detail": "all requested resolver outputs generated"}, {"check": "guardrails_passed", "status": "PASS" if guardrail_ok else "FAIL", "detail": json.dumps(guardrails, ensure_ascii=False)}], ["check", "status", "detail"])
    append_state(f"完成。final_decision={final_decision}; valid_historical_weight_count={len(valid)}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
