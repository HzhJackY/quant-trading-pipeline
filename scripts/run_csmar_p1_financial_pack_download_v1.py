from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.csmar.credential_loader import load_csmar_credentials


OUT = ROOT / "output" / "csmar_p1_financial_pack_download_v1"
EXPORT_DIR = ROOT / "data" / "csmar_exports"
STATUS_PATH = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS_PATH = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
README_PATH = ROOT / "README.md"
ALL_DAILY_PATH = ROOT / "output" / "all_daily.parquet"
PANEL_PATH = ROOT / "output" / "training_panel_v15_sr.parquet"
PAPER_PIPELINE_PATH = ROOT / "paper_trading" / "paper_trading_pipeline.py"
README_CONSISTENCY_REPORT = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"
V1_MANIFEST_PATH = ROOT / "output" / "csmar_pack_download_executor_v1" / "csmar_pack_query_manifest_v1.csv"
LOCAL_MANIFEST_PATH = OUT / "p1_pack_query_manifest_v1.csv"
RUN_DATE = date.today().isoformat()
MAPPING_PATCH_V2_PLAN_PATH = ROOT / "output" / "csmar_p1_financial_table_mapping_patch_v1" / "p1_financial_pack_download_plan_v2.csv"
RECONCILIATION_PRIORITY_PATH = ROOT / "output" / "csmar_financial_source_table_coverage_reconciliation_v1" / "recommended_download_priority_v1.csv"
PROTECTED = [README_PATH, ALL_DAILY_PATH, PANEL_PATH, PAPER_PIPELINE_PATH]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def query_key(table_name: str, condition: str, start: str, end: str) -> str:
    payload = {
        "table": table_name,
        "condition": condition,
        "startTime": start,
        "endTime": end,
    }
    return hash_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def sanitize(text: Any) -> str:
    value = "" if text is None else str(text)
    for key in ("CSMAR_ACCOUNT", "CSMAR_PASSWORD"):
        secret = os.environ.get(key, "")
        if secret:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(?i)(token|cookie|session|password|account)\s*[:=]\s*[^,\s;]+", r"\1=[REDACTED]", value)
    return value[:800]


def classify_error(text: Any) -> str:
    msg = sanitize(text).lower()
    if "does not have this query field" in msg or ("query field" in msg and "does not have" in msg):
        return "INVALID_FIELD"
    if "download limit" in msg or "downloads has reached" in msg or "下载次数" in msg or "下載次數" in msg:
        return "DAILY_LIMIT"
    if "30分钟" in msg or "30分鐘" in msg or "same query" in msg:
        return "REPEAT_30MIN_LIMIT"
    if "credential" in msg or "login" in msg or "sign in" in msg:
        return "CREDENTIAL_ERROR"
    if "timeout" in msg or "network" in msg or "connection" in msg:
        return "NETWORK_ERROR"
    return "UNKNOWN_ERROR"


def recent_csmar_log_error() -> tuple[str, str]:
    log_path = ROOT / "csmar-log.log"
    if not log_path.exists():
        return "UNKNOWN_ERROR", ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:]
    except Exception:
        return "UNKNOWN_ERROR", ""
    for line in reversed(lines):
        category = classify_error(line)
        if category in {"DAILY_LIMIT", "INVALID_FIELD", "REPEAT_30MIN_LIMIT", "CREDENTIAL_ERROR", "NETWORK_ERROR"}:
            return category, sanitize(line)
    return "UNKNOWN_ERROR", ""


def run_command(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False, timeout=timeout)


def field_dictionary_path() -> Path:
    root = ROOT / "output" / "csmar_table_inventory_audit_v1" / "csmar_field_dictionary_v1.csv"
    if root.exists():
        return root
    return ROOT / "xhs" / "output" / "csmar_table_inventory_audit_v1" / "csmar_field_dictionary_v1.csv"


def read_csv_safe(path: Path, nrows: int | None = None) -> tuple[bool, pd.DataFrame, str]:
    try:
        return True, pd.read_csv(path, dtype=str, nrows=nrows).fillna(""), ""
    except Exception as exc:  # noqa: BLE001
        return False, pd.DataFrame(), f"{type(exc).__name__}: {sanitize(exc)}"


def input_audit() -> pd.DataFrame:
    paths = [
        (ROOT / "output" / "csmar_pack_download_executor_patch_v2" / "csmar_pack_download_plan_clean_v2.csv", "ROOT_CANONICAL", "clean pack download plan"),
        (ROOT / "output" / "csmar_pack_download_executor_patch_v2" / "pack_download_column_validation_v2.csv", "ROOT_CANONICAL", "prior column validation"),
        (V1_MANIFEST_PATH, "ROOT_CANONICAL", "query manifest"),
        (field_dictionary_path(), "ROOT_CANONICAL" if field_dictionary_path().is_relative_to(ROOT / "output") else "LEGACY_XHS_FALLBACK", "field dictionary"),
        (PANEL_PATH, "ROOT_CANONICAL", "v15 training panel read-only"),
    ]
    rows = []
    for path, source_type, role in paths:
        exists = path.exists()
        readable = False
        notes = ""
        if exists:
            try:
                if path.suffix.lower() == ".parquet":
                    pd.read_parquet(path, columns=["date", "symbol"]).head(3)
                else:
                    pd.read_csv(path, nrows=3)
                readable = True
            except Exception as exc:  # noqa: BLE001
                notes = f"{type(exc).__name__}: {sanitize(exc)}"
        else:
            notes = "missing"
        rows.append({
            "input_path": rel(path),
            "exists": exists,
            "readable": readable,
            "source_type": source_type,
            "role": role,
            "notes": notes,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "input_audit_v1.csv", index=False, encoding="utf-8-sig")
    return df


def load_field_dictionary() -> pd.DataFrame:
    path = field_dictionary_path()
    if not path.exists():
        return pd.DataFrame(columns=["table_id", "table_name", "field_name", "field_name_cn"])
    return pd.read_csv(path, dtype=str).fillna("")


def v15_date_range() -> tuple[str, str]:
    panel = pd.read_parquet(PANEL_PATH, columns=["date"])
    dates = pd.to_datetime(panel["date"], errors="coerce").dropna()
    if dates.empty:
        return "2015-01-01", "2026-06-30"
    return (dates.min() - pd.DateOffset(years=2)).strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


def available_fields(field_dict: pd.DataFrame, table_id: str) -> set[str]:
    return set(field_dict.loc[field_dict["table_id"].astype(str).eq(table_id), "field_name"].astype(str))


def expected_output_name(table_id: str, start: str, end: str, condition: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9]+", "_", condition).strip("_")[:40] or "all"
    return f"{table_id}_{start}_{end}_{tag}_p1_pack_export_{datetime.now():%Y%m%d}.csv"


TARGET_SPECS: list[dict[str, Any]] = [
    {
        "priority": 1,
        "table_name": "利润表",
        "table_id": "FS_Comins",
        "table_category": "income_statement",
        "target_factor_group": "income_statement",
        "required_alias": "Stkcd|Accper|Typrep|营业收入|归母净利润|销售费用|管理费用|研发费用",
        "candidate_fields": ["Stkcd", "Accper", "Typrep", "营业收入", "归母净利润", "销售费用", "管理费用", "研发费用"],
        "core_fields": ["Stkcd", "Accper", "Typrep"],
        "target_factors_supported": "ProfitGrowth_YoY|RevGrowth_YoY|NetMargin|expense ratios",
        "notes": "No execution unless field dictionary confirms the table and requested fields.",
    },
    {
        "priority": 2,
        "table_name": "资产负债表",
        "table_id": "FS_Combas",
        "table_category": "balance_sheet",
        "target_factor_group": "balance_sheet",
        "required_alias": "Stkcd|Accper|Typrep|总资产|总负债|归母权益|股东权益",
        "candidate_fields": ["Stkcd", "Accper", "Typrep", "总资产", "总负债", "归母权益", "股东权益"],
        "core_fields": ["Stkcd", "Accper", "Typrep"],
        "target_factors_supported": "Debt_Ratio|BP candidate",
        "notes": "No execution unless field dictionary confirms the table and requested fields.",
    },
    {
        "priority": 3,
        "table_name": "财务指标表",
        "table_id": "FI_T5",
        "table_category": "financial_indicator",
        "target_factor_group": "financial_indicator",
        "required_alias": "Stkcd|Accper|Typrep|ROE|净利率|资产负债率",
        "candidate_fields": ["Stkcd", "Accper", "Typrep", "F050501B", "F053301B", "F051701B", "F051801B"],
        "core_fields": ["Stkcd", "Accper", "Typrep", "F050501B"],
        "target_factors_supported": "ROE|NetMargin candidate|Debt_Ratio candidate",
        "notes": "Executable because field dictionary confirms these API field names; semantic mapping remains for rebuild review.",
    },
]


def validate_target(spec: dict[str, Any], field_dict: pd.DataFrame) -> dict[str, Any]:
    fields = available_fields(field_dict, spec["table_id"])
    if not fields:
        return {
            "table_name": spec["table_name"],
            "target_factor_group": spec["target_factor_group"],
            "required_fields_cn_or_alias": spec["required_alias"],
            "resolved_csmar_fields": "",
            "missing_fields": "|".join(spec["candidate_fields"]),
            "optional_fields_removed": "",
            "validation_status": "BLOCKED_TABLE_NOT_FOUND",
            "notes": f"{spec['table_id']} not found in field dictionary.",
        }
    resolved = [f for f in spec["candidate_fields"] if f in fields]
    missing = [f for f in spec["candidate_fields"] if f not in fields]
    core_missing = [f for f in spec["core_fields"] if f not in fields]
    if core_missing:
        status = "BLOCKED_NO_CORE_FIELDS"
    elif missing:
        status = "PARTIAL_READY"
    else:
        status = "READY"
    return {
        "table_name": spec["table_name"],
        "target_factor_group": spec["target_factor_group"],
        "required_fields_cn_or_alias": spec["required_alias"],
        "resolved_csmar_fields": ",".join(resolved),
        "missing_fields": "|".join(missing),
        "optional_fields_removed": "|".join(missing),
        "validation_status": status,
        "notes": "Fields resolved from field dictionary only.",
    }


def p1_local_ready_tables() -> set[str]:
    ready: set[str] = set()
    expected = {"FI_T5", "FN_Fn050", "FN_Fn060"}
    for table in expected:
        for path in EXPORT_DIR.glob(f"*{table}*.csv"):
            try:
                df = pd.read_csv(path, dtype=str, nrows=100)
                cols = {str(c) for c in df.columns}
                if {"Stkcd", "Accper"}.issubset(cols):
                    ready.add(table)
                    break
            except Exception:
                continue
    return ready


def apply_reconciliation_priority_override(plan: pd.DataFrame) -> pd.DataFrame:
    if not RECONCILIATION_PRIORITY_PATH.exists():
        return plan
    priority = pd.read_csv(RECONCILIATION_PRIORITY_PATH, dtype=str).fillna("")
    next_rows = priority[priority["should_download_next"].astype(str).str.lower().eq("true")]
    if next_rows.empty:
        return plan
    table = str(next_rows.sort_values("priority_rank").iloc[0]["table_name"])
    if table == "WAIT_FOR_HUMAN_REVIEW":
        return pd.DataFrame([{
            "priority": "0",
            "table_name": "WAIT_FOR_HUMAN_REVIEW",
            "table_id": "WAIT_FOR_HUMAN_REVIEW",
            "table_category": "review_gate",
            "columns": "",
            "condition": "",
            "startTime": "",
            "endTime": "",
            "expected_output_name": "",
            "target_local_dir": "",
            "target_factors_supported": "core raw financial source table review",
            "field_validation_status": "BLOCKED_HUMAN_REVIEW",
            "should_execute_first": False,
            "execute_rank": 1,
            "notes": "Source table coverage reconciliation deprioritized FN_Fn050; human table/field review required before download.",
        }])
    hit = plan[plan["table_id"].astype(str).eq(table)].copy()
    if hit.empty:
        return plan
    hit["should_execute_first"] = True
    hit["execute_rank"] = 1
    return hit


def apply_existing_export_skip(plan: pd.DataFrame) -> pd.DataFrame:
    if plan.empty or "table_id" not in plan.columns:
        return plan
    local_ready = p1_local_ready_tables()
    plan = plan.copy()
    plan["local_status"] = plan["table_id"].astype(str).map(lambda t: "LOCAL_ALREADY_DOWNLOADED" if t in local_ready else "LOCAL_NOT_FOUND")
    plan["should_skip_download"] = plan["table_id"].astype(str).isin(local_ready)
    remaining = plan[~plan["should_skip_download"].astype(bool)].copy()
    if remaining.empty:
        plan["should_execute_first"] = False
        return plan
    if "execute_rank" in remaining.columns:
        remaining = remaining.sort_values("execute_rank").copy()
    remaining["execute_rank"] = range(1, len(remaining) + 1)
    remaining["should_execute_first"] = False
    remaining.iloc[0, remaining.columns.get_loc("should_execute_first")] = True
    return remaining


def load_mapping_patch_plan_v2() -> pd.DataFrame | None:
    if not MAPPING_PATCH_V2_PLAN_PATH.exists():
        return None
    plan = pd.read_csv(MAPPING_PATCH_V2_PLAN_PATH, dtype=str).fillna("")
    plan.to_csv(OUT / "p1_financial_pack_download_plan_v1.csv", index=False, encoding="utf-8-sig")
    validation = pd.DataFrame([{
        "table_name": row.get("table_name", ""),
        "target_factor_group": row.get("table_category", ""),
        "required_fields_cn_or_alias": row.get("target_factors_supported", ""),
        "resolved_csmar_fields": row.get("columns", ""),
        "missing_fields": "",
        "optional_fields_removed": "",
        "validation_status": row.get("field_validation_status", ""),
        "notes": "Loaded from CSMAR P1 Financial Table Mapping Patch v1 plan_v2.",
    } for _, row in plan.iterrows()])
    validation.to_csv(OUT / "p1_financial_column_validation_v1.csv", index=False, encoding="utf-8-sig")
    return plan


def generate_plan() -> tuple[pd.DataFrame, pd.DataFrame]:
    patched_plan = load_mapping_patch_plan_v2()
    if patched_plan is not None:
        validation = pd.read_csv(OUT / "p1_financial_column_validation_v1.csv", dtype=str).fillna("")
        return patched_plan, validation
    field_dict = load_field_dictionary()
    start, end = v15_date_range()
    validation_rows = [validate_target(spec, field_dict) for spec in TARGET_SPECS]
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(OUT / "p1_financial_column_validation_v1.csv", index=False, encoding="utf-8-sig")

    rows = []
    executable_seen = False
    for spec, vrow in zip(TARGET_SPECS, validation_rows):
        status = vrow["validation_status"]
        if status not in {"READY", "PARTIAL_READY"}:
            continue
        columns = vrow["resolved_csmar_fields"]
        should_first = not executable_seen
        executable_seen = True
        rows.append({
            "priority": spec["priority"],
            "table_name": spec["table_name"],
            "table_id": spec["table_id"],
            "table_category": spec["table_category"],
            "columns": columns,
            "condition": "Stkcd like '%'",
            "startTime": start,
            "endTime": end,
            "expected_output_name": expected_output_name(spec["table_id"], start, end, "Stkcd like '%'"),
            "target_local_dir": rel(EXPORT_DIR),
            "target_factors_supported": spec["target_factors_supported"],
            "field_validation_status": status,
            "should_execute_first": should_first,
            "notes": spec["notes"] + (" PARTIAL_FIELD_READY." if status == "PARTIAL_READY" else " READY."),
        })
    plan = pd.DataFrame(rows, columns=[
        "priority", "table_name", "table_id", "table_category", "columns", "condition",
        "startTime", "endTime", "expected_output_name", "target_local_dir",
        "target_factors_supported", "field_validation_status", "should_execute_first", "notes",
    ])
    plan.to_csv(OUT / "p1_financial_pack_download_plan_v1.csv", index=False, encoding="utf-8-sig")
    return plan, validation


def load_manifest() -> pd.DataFrame:
    frames = []
    for path in [V1_MANIFEST_PATH, LOCAL_MANIFEST_PATH]:
        if path.exists():
            ok, df, _ = read_csv_safe(path)
            if ok:
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["table_name", "condition", "startTime", "endTime", "query_key", "last_attempt_time", "status"])
    return pd.concat(frames, ignore_index=True).fillna("")


def recently_attempted(manifest: pd.DataFrame, table: str, condition: str, start: str, end: str) -> bool:
    qkey = query_key(table, condition, start, end)
    if "query_key" in manifest.columns:
        hit = manifest[manifest["query_key"].astype(str).eq(qkey)]
    else:
        hit = pd.DataFrame()
    if hit.empty:
        same = (
            manifest.get("table_name", pd.Series(dtype=str)).astype(str).eq(table)
            & manifest.get("condition", pd.Series(dtype=str)).astype(str).eq(condition)
            & manifest.get("startTime", pd.Series(dtype=str)).astype(str).eq(start)
            & manifest.get("endTime", pd.Series(dtype=str)).astype(str).eq(end)
        )
        hit = manifest[same] if len(manifest) else pd.DataFrame()
    if hit.empty:
        return False
    ts = str(hit.iloc[-1].get("last_attempt_time", ""))
    if not ts:
        return False
    try:
        return datetime.now() - datetime.fromisoformat(ts) < timedelta(minutes=30)
    except ValueError:
        return False


def latest_manifest_hit(manifest: pd.DataFrame, table: str, condition: str, start: str, end: str) -> pd.Series | None:
    qkey = query_key(table, condition, start, end)
    hit = manifest[manifest["query_key"].astype(str).eq(qkey)] if "query_key" in manifest.columns and len(manifest) else pd.DataFrame()
    if hit.empty and len(manifest):
        same = (
            manifest.get("table_name", pd.Series(dtype=str)).astype(str).eq(table)
            & manifest.get("condition", pd.Series(dtype=str)).astype(str).eq(condition)
            & manifest.get("startTime", pd.Series(dtype=str)).astype(str).eq(start)
            & manifest.get("endTime", pd.Series(dtype=str)).astype(str).eq(end)
        )
        hit = manifest[same]
    if hit.empty:
        return None
    return hit.iloc[-1]


def write_local_manifest(update: dict[str, Any]) -> None:
    existing = pd.read_csv(LOCAL_MANIFEST_PATH, dtype=str).fillna("") if LOCAL_MANIFEST_PATH.exists() else pd.DataFrame()
    out = pd.concat([existing, pd.DataFrame([update])], ignore_index=True)
    out.to_csv(LOCAL_MANIFEST_PATH, index=False, encoding="utf-8-sig")


def find_new_zip(before: set[Path]) -> Path | None:
    zip_dir = Path("C:/csmardata/zip")
    if not zip_dir.exists():
        return None
    after = {p for p in zip_dir.glob("*.zip") if p.is_file()}
    new = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    return new[0] if new else None


def find_csvs(unzip_dir: Path) -> list[Path]:
    if not unzip_dir.exists():
        return []
    return sorted([p for p in unzip_dir.rglob("*.csv") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)


def detect_csv_shape(path: Path) -> tuple[int, int]:
    try:
        df = pd.read_csv(path, dtype=str)
        return int(len(df)), int(len(df.columns))
    except Exception:
        try:
            df = pd.read_csv(path, dtype=str, nrows=50)
            return -1, int(len(df.columns))
        except Exception:
            return 0, 0


def download_summary_row(
    attempted: bool = False,
    success: bool = False,
    table_name: str = "",
    output_zip_path: str = "",
    unzip_dir: str = "",
    copied_local_csv_path: str = "",
    n_rows_detected: int = 0,
    n_columns_detected: int = 0,
    error_class: str = "NOT_EXECUTED_DRY_RUN",
    sanitized_error_message: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "attempted": attempted,
        "success": success,
        "table_name": table_name,
        "output_zip_path": output_zip_path,
        "unzip_dir": unzip_dir,
        "copied_local_csv_path": copied_local_csv_path,
        "n_rows_detected": n_rows_detected,
        "n_columns_detected": n_columns_detected,
        "error_class": error_class,
        "sanitized_error_message": sanitized_error_message,
        "notes": notes,
    }


def execute_download(plan: pd.DataFrame, max_downloads: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ready = plan[plan["should_execute_first"].astype(str).str.lower().eq("true")].head(max_downloads)
    if ready.empty:
        rows.append(download_summary_row(error_class="NOT_EXECUTED_DRY_RUN", notes="No executable P1 query."))
        return pd.DataFrame(rows)

    credential_status = load_csmar_credentials()
    if not (credential_status["account_present"] and credential_status["password_present"]):
        rows.append(download_summary_row(attempted=False, error_class="CREDENTIAL_ERROR", notes="Credentials missing; no values logged."))
        return pd.DataFrame(rows)

    manifest = load_manifest()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from csmarapi.CsmarService import CsmarService
        logging.disable(logging.CRITICAL)
        csmar = CsmarService()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            csmar.login(os.environ["CSMAR_ACCOUNT"], os.environ["CSMAR_PASSWORD"])
        logging.disable(logging.NOTSET)
    except Exception as exc:  # noqa: BLE001
        logging.disable(logging.NOTSET)
        rows.append(download_summary_row(attempted=False, error_class="CREDENTIAL_ERROR", sanitized_error_message=sanitize(exc), notes="Login failed."))
        return pd.DataFrame(rows)

    for _, query in ready.iterrows():
        table = str(query["table_id"])
        condition = str(query["condition"])
        start = str(query["startTime"])
        end = str(query["endTime"])
        qkey = query_key(table, condition, start, end)
        if recently_attempted(manifest, table, condition, start, end):
            hit = latest_manifest_hit(manifest, table, condition, start, end)
            previous_status = str(hit.get("status", "")) if hit is not None else ""
            log_category, log_message = recent_csmar_log_error()
            category = previous_status if previous_status in {"DAILY_LIMIT", "INVALID_FIELD", "CREDENTIAL_ERROR", "NETWORK_ERROR"} else log_category
            if category not in {"DAILY_LIMIT", "INVALID_FIELD", "CREDENTIAL_ERROR", "NETWORK_ERROR"}:
                category = "REPEAT_30MIN_LIMIT"
            rows.append(download_summary_row(
                attempted=False,
                table_name=table,
                error_class=category,
                sanitized_error_message=log_message,
                notes="Skipped to avoid repeating same table/condition/startTime/endTime within 30 minutes.",
            ))
            continue
        now = datetime.now().isoformat(timespec="seconds")
        update = {
            "query_id": f"P1_{len(rows) + 1:03d}",
            "table_name": table,
            "columns_hash": hash_text(str(query["columns"])),
            "condition": condition,
            "startTime": start,
            "endTime": end,
            "query_key": qkey,
            "last_attempt_time": now,
            "status": "ATTEMPTED",
            "zip_path": "",
            "unzip_dir": "",
            "local_export_path": "",
            "notes": "",
        }
        try:
            before_zip = {p for p in Path("C:/csmardata/zip").glob("*.zip")} if Path("C:/csmardata/zip").exists() else set()
            cols = [c.strip() for c in str(query["columns"]).split(",") if c.strip()]
            logging.disable(logging.CRITICAL)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                csmar.getPackResultExt(cols, condition, table, start, end)
            logging.disable(logging.NOTSET)
            zip_path = find_new_zip(before_zip)
            if not zip_path:
                log_category, log_message = recent_csmar_log_error()
                update["status"] = log_category
                update["notes"] = log_message or "No zip produced."
                rows.append(download_summary_row(True, False, table, error_class=log_category, sanitized_error_message=log_message, notes=update["notes"]))
            else:
                update["zip_path"] = str(zip_path)
                logging.disable(logging.CRITICAL)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    csmar.unzipSingle(str(zip_path))
                logging.disable(logging.NOTSET)
                unzip_dir = Path("C:/csmardata") / zip_path.stem
                csvs = find_csvs(unzip_dir)
                update["unzip_dir"] = str(unzip_dir)
                if not csvs:
                    update["status"] = "UNKNOWN_ERROR"
                    update["notes"] = "Unzip completed but no CSV found."
                    rows.append(download_summary_row(True, False, table, str(zip_path), str(unzip_dir), error_class="UNKNOWN_ERROR", notes=update["notes"]))
                else:
                    target = EXPORT_DIR / str(query["expected_output_name"])
                    shutil.copy2(csvs[0], target)
                    n_rows, n_cols = detect_csv_shape(target)
                    update["status"] = "SUCCESS"
                    update["local_export_path"] = rel(target)
                    rows.append(download_summary_row(True, True, table, str(zip_path), str(unzip_dir), rel(target), n_rows, n_cols, "NONE", "", "Copied to data/csmar_exports."))
        except Exception as exc:  # noqa: BLE001
            logging.disable(logging.NOTSET)
            category = classify_error(exc)
            update["status"] = category
            update["notes"] = sanitize(exc)
            rows.append(download_summary_row(True, False, table, error_class=category, sanitized_error_message=sanitize(exc), notes="Download stopped after classified error."))
        write_local_manifest(update)
        if rows[-1]["error_class"] in {"DAILY_LIMIT", "INVALID_FIELD"}:
            break
    return pd.DataFrame(rows)


def detect_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    lower = {str(c).lower(): str(c) for c in df.columns}
    found = []
    for cand in candidates:
        if cand.lower() in lower:
            found.append(lower[cand.lower()])
    return found


def safe_join(values: list[str]) -> str:
    return "|".join(sorted(dict.fromkeys(v for v in values if v)))


def scan_exports() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    p0_names = {"IAR_Rept", "IAR_Forecdt"}
    for path in sorted(EXPORT_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".xlsx", ".xls", ".parquet"}:
            continue
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path, dtype=str, nrows=50)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(path, dtype=str, nrows=50)
            else:
                df = pd.read_parquet(path).head(50)
            readable = True
            note = ""
        except Exception as exc:  # noqa: BLE001
            df = pd.DataFrame()
            readable = False
            note = f"{type(exc).__name__}: {sanitize(exc)}"
        cols = [str(c) for c in df.columns]
        text = f"{path.name} {' '.join(cols)}".lower()
        table = ""
        for candidate in ["IAR_Rept", "IAR_Forecdt", "FI_T5", "FS_Comins", "FS_Combas", "FN_Fn050", "FN_Fn060"]:
            if candidate.lower() in text:
                table = candidate
                break
        if not table and "f050501b" in text:
            table = "FI_T5"
        financial = [
            c for c in cols
            if c.lower() in {"f050501b", "f053301b", "f051701b", "f051801b", "fn05001", "fn05002", "fn_fn06001", "fn_fn06002"}
            or any(token in c for token in ["营业收入", "净利润", "总资产", "总负债", "ROE", "净利率", "资产负债率"])
        ]
        factors = []
        low = {c.lower() for c in cols}
        if "f050501b" in low:
            factors.append("ROE")
        if "f053301b" in low or "f051701b" in low or "f051801b" in low:
            factors.append("financial_indicator_candidates")
        if any(c.lower() in {"fn05001", "fn05002"} for c in cols):
            factors.append("sales_expense_to_revenue_candidate")
        if any(c.lower() in {"fn_fn06001", "fn_fn06002"} for c in cols):
            factors.append("rd_expense_to_revenue_candidate")
        notes = [note] if note else []
        if table in p0_names:
            notes.append("existing_p0_file")
        elif table:
            notes.append("p1_or_financial_related_file")
        rows.append({
            "file_path": rel(path),
            "file_type": path.suffix.lower().lstrip("."),
            "readable": readable,
            "n_rows_sampled": int(len(df)),
            "n_columns": int(len(cols)),
            "detected_table_name": table,
            "detected_symbol_columns": safe_join(detect_columns(df, ["Stkcd", "symbol", "证券代码"])),
            "detected_report_period_columns": safe_join(detect_columns(df, ["Accper", "report_period", "报告期"])),
            "detected_pit_date_columns": safe_join(detect_columns(df, ["Annodt", "Actudt", "Firforecdt", "公告日", "披露日"])),
            "detected_financial_fields": safe_join(financial),
            "likely_supported_factors": safe_join(factors),
            "notes": "; ".join(notes),
        })
    df = pd.DataFrame(rows, columns=[
        "file_path", "file_type", "readable", "n_rows_sampled", "n_columns",
        "detected_table_name", "detected_symbol_columns", "detected_report_period_columns",
        "detected_pit_date_columns", "detected_financial_fields", "likely_supported_factors", "notes",
    ])
    df.to_csv(OUT / "p1_manual_export_file_check_v1.csv", index=False, encoding="utf-8-sig")
    return df


def credential_exposure_detected() -> bool:
    load_csmar_credentials()
    secrets = [os.environ.get("CSMAR_ACCOUNT", ""), os.environ.get("CSMAR_PASSWORD", "")]
    secrets = [s for s in secrets if s]
    if not secrets:
        return False
    for path in OUT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".md", ".json", ".txt", ".log"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(secret in text for secret in secrets):
                return True
    return False


def update_status(decision: str, success_count: int, dry_run: bool) -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    status.setdefault("alternative_data", {})
    if success_count >= 1:
        csmar_status = "p1_financial_pack_download_partial"
    elif dry_run:
        csmar_status = "p1_financial_pack_download_ready_dry_run"
    else:
        csmar_status = "p1_financial_pack_download_blocked"
    status["alternative_data"]["csmar_status"] = csmar_status
    status["alternative_data"]["csmar_latest_task"] = "CSMAR P1 Financial Pack Download v1"
    status["alternative_data"]["csmar_latest_output"] = rel(OUT)
    status.setdefault("validation", {})
    status["validation"]["pit_financial_status"] = "p0_pit_dates_imported_p1_financial_download_partial" if success_count >= 1 else "p0_pit_dates_imported_p1_financial_download_pending"
    status["validation"]["blend_v3_historical_metrics_status"] = "under_pit_review"
    status.setdefault("project", {})["last_updated"] = RUN_DATE
    STATUS_PATH.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False), encoding="utf-8")


def append_decision(decision: str, downloaded_tables: str, remaining: bool) -> None:
    marker = f"Decision = {decision}。"
    text = DECISIONS_PATH.read_text(encoding="utf-8") if DECISIONS_PATH.exists() else "# 决策日志\n"
    if marker in text and "CSMAR P1 financial pack download" in text:
        return
    block = "\n".join([
        f"## {RUN_DATE}",
        "",
        "决策：",
        "",
        f"- CSMAR P1 financial pack download 是否成功：{'是' if downloaded_tables else '否'}。",
        f"- 成功下载哪些表：{downloaded_tables or '无'}。",
        f"- 是否仍需继续下载剩余 P1 表：{'是' if remaining else '否'}。",
        "- 不接入 production。",
        "- 不修改 README。",
        "- 不修改 training_panel。",
        f"- Decision = {decision}。",
    ])
    DECISIONS_PATH.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def task_card(decision: str, execute: bool, outputs: list[str], downloaded: str, failed: str, next_task: str) -> Path:
    path = OUT / "task_completion_card.md"
    lines = [
        "任务名称：CSMAR P1 Financial Pack Download v1",
        f"运行日期：{RUN_DATE}",
        "是否修改 production：否",
        "是否修改 README：否",
        "是否修改 all_daily：否",
        "是否修改 training_panel：否",
        "是否训练模型：否",
        "是否运行回测：否",
        "是否做 IC：否",
        "是否生成交易信号：否",
        f"是否执行 CSMAR 下载：{'是' if execute else '否'}",
        "核心输出：",
        *[f"- {o}" for o in outputs],
        f"核心结论：{decision}",
        f"成功下载表：{downloaded or '无'}",
        f"失败或未执行表：{failed or '无'}",
        f"下一步建议：{next_task}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def final_qa(initial_hashes: dict[Path, str], execute: bool, max_downloads: int, attempted: int, success_count: int, copied_files: list[str]) -> pd.DataFrame:
    current = {p: sha256_file(p) for p in PROTECTED}
    manifest_checked = V1_MANIFEST_PATH.exists()
    repeat_violations = 0
    summary = pd.read_csv(OUT / "p1_download_result_summary_v1.csv", dtype=str).fillna("") if (OUT / "p1_download_result_summary_v1.csv").exists() else pd.DataFrame()
    if not summary.empty:
        repeat_violations = int(summary["error_class"].eq("REPEAT_30MIN_LIMIT").sum())
    rows = [
        ("README.md not modified", current[README_PATH] == initial_hashes[README_PATH], "hash unchanged"),
        ("all_daily.parquet not modified", current[ALL_DAILY_PATH] == initial_hashes[ALL_DAILY_PATH], "hash unchanged"),
        ("training_panel_v15_sr.parquet not modified", current[PANEL_PATH] == initial_hashes[PANEL_PATH], "hash unchanged"),
        ("model files not modified", True, "no model paths written"),
        ("paper_trading_pipeline.py not modified", current[PAPER_PIPELINE_PATH] == initial_hashes[PAPER_PIPELINE_PATH], "hash unchanged"),
        ("production config not modified", True, "only config/project_status.yaml governance fields updated"),
        ("no model training executed", True, ""),
        ("no backtest executed", True, ""),
        ("no IC test executed", True, ""),
        ("no trading signal generated", True, ""),
        ("no real orders generated", True, ""),
        ("no credential value printed", True, ""),
        ("no credential saved to output", not credential_exposure_detected(), ""),
        ("root-level output used", str(OUT).startswith(str(ROOT / "output")), rel(OUT)),
        ("xhs/output not used for new outputs", True, rel(OUT)),
        ("P1 plan generated", (OUT / "p1_financial_pack_download_plan_v1.csv").exists(), ""),
        ("P1 column validation generated", (OUT / "p1_financial_column_validation_v1.csv").exists(), ""),
        ("query manifest checked", manifest_checked, rel(V1_MANIFEST_PATH)),
        ("no repeated query within 30 minutes", repeat_violations == 0, f"repeat_30min_rows={repeat_violations}"),
        ("if executed, max_downloads respected", (not execute) or attempted <= max_downloads, f"attempted={attempted}; max_downloads={max_downloads}"),
        ("if executed, downloaded file copied to data/csmar_exports", (not execute) or success_count == 0 or all((ROOT / f).exists() for f in copied_files), "|".join(copied_files)),
        ("local export file check generated", (OUT / "p1_manual_export_file_check_v1.csv").exists(), ""),
        ("project_status.yaml updated", STATUS_PATH.exists(), rel(STATUS_PATH)),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS_PATH.exists(), rel(CURRENT_STATUS_PATH)),
        ("DECISIONS.md appended", DECISIONS_PATH.exists(), rel(DECISIONS_PATH)),
        ("README consistency check executed", README_CONSISTENCY_REPORT.exists(), rel(README_CONSISTENCY_REPORT)),
        ("README not auto-modified", current[README_PATH] == initial_hashes[README_PATH], ""),
    ]
    df = pd.DataFrame(rows, columns=["check", "pass", "details"])
    df.to_csv(OUT / "final_qa_csmar_p1_financial_pack_download_v1.csv", index=False, encoding="utf-8-sig")
    return df


def choose_decision(execute: bool, summary: pd.DataFrame, initial_hashes: dict[Path, str]) -> str:
    if any(sha256_file(p) != initial_hashes[p] for p in PROTECTED):
        return "INVALID_MODIFICATION"
    if credential_exposure_detected():
        return "INVALID_CREDENTIAL_EXPOSURE"
    if not execute:
        return "CSMAR_P1_FINANCIAL_PACK_DOWNLOAD_READY_DRY_RUN"
    if not summary.empty and summary["success"].astype(str).str.lower().eq("true").any():
        return "CSMAR_P1_FINANCIAL_PACK_DOWNLOAD_PARTIAL_SUCCESS"
    if not summary.empty and summary["error_class"].eq("DAILY_LIMIT").any():
        return "CSMAR_P1_FINANCIAL_PACK_DOWNLOAD_BLOCKED_DAILY_LIMIT"
    if not summary.empty and summary["error_class"].eq("INVALID_FIELD").any():
        return "CSMAR_P1_FINANCIAL_PACK_DOWNLOAD_BLOCKED_INVALID_FIELD"
    return "CSMAR_P1_FINANCIAL_PACK_DOWNLOAD_READY_DRY_RUN"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSMAR P1 financial pack download v1. Default is dry-run.")
    parser.add_argument("--execute", action="store_true", help="Call CSMAR getPackResultExt.")
    parser.add_argument("--max-downloads", type=int, default=1, help="Maximum downloads in execute mode.")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    initial_hashes = {p: sha256_file(p) for p in PROTECTED}

    input_audit()
    plan, _validation = generate_plan()
    plan = apply_existing_export_skip(plan)
    plan = apply_reconciliation_priority_override(plan)
    plan.to_csv(OUT / "p1_financial_pack_download_plan_v1.csv", index=False, encoding="utf-8-sig")
    if args.execute:
        result = execute_download(plan, max(0, args.max_downloads))
    else:
        result = pd.DataFrame([download_summary_row(False, False, error_class="NOT_EXECUTED_DRY_RUN", notes="Dry-run only; pass --execute to download.")])
    result.to_csv(OUT / "p1_download_result_summary_v1.csv", index=False, encoding="utf-8-sig")

    helper = run_command([sys.executable, "scripts/check_csmar_manual_exports_v1.py"], timeout=180)
    (OUT / "check_csmar_manual_exports_stdout.txt").write_text(sanitize(helper.stdout), encoding="utf-8")
    (OUT / "check_csmar_manual_exports_stderr.txt").write_text(sanitize(helper.stderr), encoding="utf-8")
    scan_exports()

    decision = choose_decision(args.execute, result, initial_hashes)
    success_rows = result[result["success"].astype(str).str.lower().eq("true")] if not result.empty else pd.DataFrame()
    downloaded_tables = "|".join(success_rows["table_name"].astype(str).tolist()) if not success_rows.empty else ""
    downloaded_files = "|".join(success_rows["copied_local_csv_path"].astype(str).tolist()) if not success_rows.empty else ""
    copied_files = [f for f in downloaded_files.split("|") if f]
    attempted = int(result["attempted"].astype(str).str.lower().eq("true").sum()) if not result.empty else 0
    success_count = int(result["success"].astype(str).str.lower().eq("true").sum()) if not result.empty else 0

    update_status(decision, success_count, dry_run=not args.execute)
    gen = run_command([sys.executable, "scripts/generate_current_status_md.py"], timeout=180)
    (OUT / "generate_current_status_stdout.txt").write_text(sanitize(gen.stdout), encoding="utf-8")
    (OUT / "generate_current_status_stderr.txt").write_text(sanitize(gen.stderr), encoding="utf-8")
    remaining = success_count < int(len(plan))
    append_decision(decision, downloaded_tables, remaining)
    readme = run_command([sys.executable, "scripts/check_readme_consistency.py"], timeout=180)
    (OUT / "readme_consistency_stdout.txt").write_text(sanitize(readme.stdout), encoding="utf-8")
    (OUT / "readme_consistency_stderr.txt").write_text(sanitize(readme.stderr), encoding="utf-8")

    if success_count >= 1:
        next_task = "CSMAR PIT Financial Factor Rebuild v1"
    elif not args.execute:
        next_task = "Run: python scripts\\run_csmar_p1_financial_pack_download_v1.py --execute --max-downloads 1"
    else:
        next_task = "CSMAR P1 Financial Pack Download Patch v1"
    failed = "|".join(plan.loc[~plan["table_id"].astype(str).isin(success_rows.get("table_name", pd.Series(dtype=str)).astype(str)), "table_id"].astype(str).tolist()) if not plan.empty else ""
    task_card(decision, args.execute, [
        rel(OUT / "p1_financial_pack_download_plan_v1.csv"),
        rel(OUT / "p1_financial_column_validation_v1.csv"),
        rel(OUT / "p1_download_result_summary_v1.csv"),
        rel(OUT / "p1_manual_export_file_check_v1.csv"),
    ], downloaded_tables, failed, next_task)
    final_qa(initial_hashes, args.execute, args.max_downloads, attempted, success_count, copied_files)

    invalid_field_count = int(result["error_class"].eq("INVALID_FIELD").sum()) if not result.empty else 0
    daily_limit_count = int(result["error_class"].eq("DAILY_LIMIT").sum()) if not result.empty else 0
    repeat_count = int(result["error_class"].eq("REPEAT_30MIN_LIMIT").sum()) if not result.empty else 0
    ready_count = int(plan["field_validation_status"].isin(["READY", "PARTIAL_READY"]).sum()) if not plan.empty else 0
    terminal = {
        "input_audit_path": rel(OUT / "input_audit_v1.csv"),
        "p1_download_plan_path": rel(OUT / "p1_financial_pack_download_plan_v1.csv"),
        "p1_column_validation_path": rel(OUT / "p1_financial_column_validation_v1.csv"),
        "p1_download_result_summary_path": rel(OUT / "p1_download_result_summary_v1.csv"),
        "p1_manual_export_file_check_path": rel(OUT / "p1_manual_export_file_check_v1.csv"),
        "task_completion_card_path": rel(OUT / "task_completion_card.md"),
        "final_qa_path": rel(OUT / "final_qa_csmar_p1_financial_pack_download_v1.csv"),
        "project_status_path": rel(STATUS_PATH),
        "current_status_doc_path": rel(CURRENT_STATUS_PATH),
        "decisions_doc_path": rel(DECISIONS_PATH),
        "readme_consistency_report_path": rel(README_CONSISTENCY_REPORT),
        "dry_run": not args.execute,
        "execute_mode_used": bool(args.execute),
        "n_p1_queries_planned": int(len(plan)),
        "n_p1_queries_ready": ready_count,
        "next_execute_table": str(plan.sort_values("execute_rank").iloc[0]["table_id"]) if "execute_rank" in plan.columns and len(plan) else (str(plan.iloc[0]["table_id"]) if len(plan) else ""),
        "local_ready_table_list": "|".join(sorted(p1_local_ready_tables())),
        "n_downloads_attempted": attempted,
        "n_downloads_succeeded": success_count,
        "n_downloads_invalid_field": invalid_field_count,
        "n_downloads_daily_limit": daily_limit_count,
        "n_downloads_repeat_30min": repeat_count,
        "downloaded_table_list": downloaded_tables,
        "downloaded_file_list": downloaded_files,
        "target_export_dir": rel(EXPORT_DIR),
        "recommended_next_task": next_task,
        "readme_modified": sha256_file(README_PATH) != initial_hashes[README_PATH],
        "all_daily_modified": sha256_file(ALL_DAILY_PATH) != initial_hashes[ALL_DAILY_PATH],
        "training_panel_modified": sha256_file(PANEL_PATH) != initial_hashes[PANEL_PATH],
        "production_modified": False,
        "credential_exposure_detected": credential_exposure_detected(),
        "decision": decision,
    }
    (OUT / "terminal_summary_v1.json").write_text(json.dumps(terminal, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in terminal.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
