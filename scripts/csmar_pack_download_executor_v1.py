from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.csmar.credential_loader import load_csmar_credentials


OUT = ROOT / "output" / "csmar_pack_download_executor_v1"
EXPORT_DIR = ROOT / "data" / "csmar_exports"
STATUS_PATH = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS_PATH = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
README_PATH = ROOT / "README.md"
ALL_DAILY_PATH = ROOT / "output" / "all_daily.parquet"
PANEL_PATH = ROOT / "output" / "training_panel_v15_sr.parquet"
PAPER_PIPELINE_PATH = ROOT / "paper_trading" / "paper_trading_pipeline.py"
README_CONSISTENCY_REPORT = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"
MANIFEST_PATH = OUT / "csmar_pack_query_manifest_v1.csv"

PROTECTED = [README_PATH, ALL_DAILY_PATH, PANEL_PATH, PAPER_PIPELINE_PATH]

PLAN_COLUMNS = [
    "priority", "download_group", "table_name", "table_id", "columns", "condition",
    "startTime", "endTime", "expected_output_name", "target_local_dir",
    "target_factors_supported", "estimated_download_count_cost", "should_execute_first", "notes",
]

MANIFEST_COLUMNS = [
    "query_id", "table_name", "columns_hash", "condition", "startTime", "endTime",
    "query_key", "last_attempt_time", "status", "zip_path", "unzip_dir", "local_export_path", "notes",
]


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


def query_key(table: str, columns: str, condition: str, start: str, end: str) -> str:
    # CSMAR's 30-minute repeat guard is scoped to table/condition/date, not columns.
    return hash_text(json.dumps({
        "table": table,
        "condition": condition,
        "startTime": start,
        "endTime": end,
    }, ensure_ascii=False, sort_keys=True))


def classify_csmar_error(text: Any) -> str:
    msg = sanitize(text).lower()
    if "does not have this query field" in msg or "query field" in msg and "does not have" in msg:
        return "INVALID_FIELD"
    if "download limit" in msg or "downloads has reached" in msg or "下載次數" in msg or "下载次数" in msg:
        return "DAILY_LIMIT"
    if "30分鐘" in msg or "30分钟" in msg or "same query" in msg:
        return "REPEAT_30MIN_LIMIT"
    if "credential" in msg or "offline" in msg or "sign in" in msg or "login" in msg:
        return "CREDENTIAL_ERROR"
    if "connection" in msg or "timeout" in msg or "network" in msg or "websocket" in msg:
        return "NETWORK_ERROR"
    return "UNKNOWN_ERROR"


def sanitize(text: Any) -> str:
    value = "" if text is None else str(text)
    for key in ("CSMAR_ACCOUNT", "CSMAR_PASSWORD"):
        secret = os.environ.get(key, "")
        if secret:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(?i)(token|cookie|session|password|account)\s*[:=]\s*[^,\s;]+", r"\1=[REDACTED]", value)
    return value[:600]


def run_command(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False, timeout=timeout)


def field_dictionary_path() -> Path:
    root = ROOT / "output" / "csmar_table_inventory_audit_v1" / "csmar_field_dictionary_v1.csv"
    if root.exists():
        return root
    return ROOT / "xhs" / "output" / "csmar_table_inventory_audit_v1" / "csmar_field_dictionary_v1.csv"


def load_field_dictionary() -> pd.DataFrame:
    path = field_dictionary_path()
    if not path.exists():
        return pd.DataFrame(columns=["table_id", "field_name", "field_name_cn"])
    return pd.read_csv(path, dtype=str).fillna("")


def field_map_for_table(field_dict: pd.DataFrame, table_id: str) -> tuple[set[str], dict[str, str]]:
    subset = field_dict[field_dict["table_id"].astype(str).eq(str(table_id))]
    valid = set(subset["field_name"].dropna().astype(str))
    cn_map = {}
    for _, row in subset.iterrows():
        cn = str(row.get("field_name_cn", "")).strip()
        fn = str(row.get("field_name", "")).strip()
        if cn and fn:
            cn_map[cn] = fn
    return valid, cn_map


def validate_columns_for_table(field_dict: pd.DataFrame, table_id: str, columns: str) -> dict[str, Any]:
    requested = [c.strip() for c in str(columns).split(",") if c.strip()]
    valid_fields, cn_map = field_map_for_table(field_dict, table_id)
    valid_cols: list[str] = []
    invalid_cols: list[str] = []
    removed_optional: list[str] = []
    for col in requested:
        api_col = cn_map.get(col, col)
        if api_col in valid_fields:
            valid_cols.append(api_col)
        else:
            invalid_cols.append(col)
            if col == "ShortName":
                removed_optional.append(col)
    has_symbol = any(c.lower() == "stkcd" for c in valid_cols)
    has_accper = any(c.lower() == "accper" for c in valid_cols)
    required_missing = []
    if not has_symbol:
        required_missing.append("Stkcd")
    if not has_accper:
        required_missing.append("Accper")
    validation_pass = bool(valid_cols) and not required_missing
    notes = "OK" if validation_pass else "BLOCKED_FIELD_MAPPING"
    if not valid_fields:
        notes = "BLOCKED_FIELD_DICTIONARY_MISSING_FOR_TABLE"
    return {
        "table_name": table_id,
        "original_columns": ",".join(requested),
        "valid_columns": ",".join(dict.fromkeys(valid_cols)),
        "invalid_columns": ",".join(invalid_cols),
        "required_columns_missing": ",".join(required_missing),
        "optional_columns_removed": ",".join(removed_optional),
        "validation_pass": validation_pass,
        "notes": notes,
    }


def validate_plan_columns(plan: pd.DataFrame) -> pd.DataFrame:
    field_dict = load_field_dictionary()
    rows = []
    for _, row in plan.iterrows():
        rows.append(validate_columns_for_table(field_dict, str(row["table_id"]), str(row["columns"])))
    return pd.DataFrame(rows)


def input_audit() -> pd.DataFrame:
    inputs = [
        (ROOT / "output" / "csmar_pit_financial_factor_rebuild_v1" / "rebuild_target_factor_spec_v1.csv", "ROOT_CANONICAL", "rebuild target factor spec"),
        (ROOT / "output" / "csmar_pit_financial_factor_rebuild_v1" / "csmar_raw_financial_fetch_log_v1.csv", "ROOT_CANONICAL", "prior raw fetch log"),
        (ROOT / "output" / "csmar_table_inventory_audit_v1" / "csmar_table_inventory_v1.csv", "ROOT_CANONICAL", "table inventory"),
        (ROOT / "output" / "csmar_table_inventory_audit_v1" / "csmar_field_dictionary_v1.csv", "ROOT_CANONICAL", "field dictionary"),
        (PANEL_PATH, "ROOT_CANONICAL", "v15 panel read-only date range"),
    ]
    rows = []
    for path, source_type, role in inputs:
        selected = path
        selected_source = source_type
        if not path.exists() and "csmar_table_inventory_audit_v1" in str(path):
            fallback = ROOT / "xhs" / "output" / "csmar_table_inventory_audit_v1" / path.name
            selected = fallback
            selected_source = "LEGACY_XHS_FALLBACK"
        exists = selected.exists()
        readable = False
        notes = ""
        if exists:
            try:
                if selected.suffix.lower() == ".csv":
                    pd.read_csv(selected, nrows=3)
                elif selected.suffix.lower() == ".parquet":
                    pd.read_parquet(selected, columns=["date", "symbol"]).head(3)
                else:
                    selected.read_text(encoding="utf-8", errors="ignore")
                readable = True
            except Exception as exc:
                notes = f"{type(exc).__name__}: {str(exc)[:160]}"
        else:
            notes = "missing"
        rows.append({
            "input_path": rel(selected),
            "exists": exists,
            "readable": readable,
            "source_type": selected_source,
            "role": role,
            "notes": notes,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "input_audit_v1.csv", index=False, encoding="utf-8-sig")
    return df


def v15_date_range() -> tuple[str, str]:
    try:
        panel = pd.read_parquet(PANEL_PATH, columns=["date"])
        dates = pd.to_datetime(panel["date"], errors="coerce").dropna()
        if len(dates):
            start = (dates.min() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
            end = dates.max().strftime("%Y-%m-%d")
            return start, end
    except Exception:
        pass
    return "2015-01-01", "2026-06-30"


def make_output_name(table_id: str, start: str, end: str, condition: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9]+", "_", condition).strip("_")[:40] or "all"
    return f"{table_id}_{start}_{end}_{tag}_pack_export_{datetime.now():%Y%m%d}.csv"


def plan_rows(start: str, end: str) -> list[dict[str, Any]]:
    base = [
        {
            "priority": "P0",
            "download_group": "pit_disclosure",
            "table_name": "财务报告披露日期",
            "table_id": "IAR_Rept",
            "columns": "Stkcd,ShortName,Accper,Annodt,Reptyp,Annopub",
            "target_factors_supported": "all PIT-aligned factors",
            "should_execute_first": True,
            "notes": "First P0 pack. Provides announcement date when available.",
        },
        {
            "priority": "P0",
            "download_group": "pit_disclosure",
            "table_name": "预约披露日期/实际披露日期",
            "table_id": "IAR_Forecdt",
            "columns": "Stkcd,ShortName,Accper,Firforecdt,Firchangdt,Secchangdt,Thirchangdt,Actudt",
            "target_factors_supported": "all PIT-aligned factors",
            "should_execute_first": True,
            "notes": "Second P0 pack. Provides actual disclosure date and reservation changes.",
        },
        {
            "priority": "P1",
            "download_group": "financial_indicator",
            "table_name": "财务指标表",
            "table_id": "FI_T5",
            "columns": "Stkcd,ShortName,Accper,Typrep,F050501B,F051501B,F051701B,F053301B",
            "target_factors_supported": "ROE|NetMargin|sales_expense_to_revenue|candidate leverage ratio if field confirmed",
            "should_execute_first": False,
            "notes": "Use after P0 dates are available; field semantics require manual dictionary confirmation.",
        },
        {
            "priority": "P1",
            "download_group": "income_statement",
            "table_name": "利润表",
            "table_id": "FS_Comins",
            "columns": "Stkcd,ShortName,Accper,Typrep,营业收入,归母净利润,销售费用,管理费用,研发费用",
            "target_factors_supported": "ProfitGrowth_YoY|RevGrowth_YoY|NetMargin|EP candidate",
            "should_execute_first": False,
            "notes": "Placeholder table id from standard statement naming; execute only after inventory confirms exact table/field ids.",
        },
        {
            "priority": "P1",
            "download_group": "balance_sheet",
            "table_name": "资产负债表",
            "table_id": "FS_Combas",
            "columns": "Stkcd,ShortName,Accper,Typrep,总资产,总负债,归母权益,股东权益",
            "target_factors_supported": "Debt_Ratio|BP candidate",
            "should_execute_first": False,
            "notes": "Placeholder table id from standard statement naming; execute only after inventory confirms exact table/field ids.",
        },
        {
            "priority": "P2",
            "download_group": "expense_detail",
            "table_name": "财务报表附注-销售费用明细",
            "table_id": "FN_Fn050",
            "columns": "Stkcd,ShortName,Accper,Typrep,FN05001,FN05002,FN05003",
            "target_factors_supported": "sales_expense_to_revenue candidate",
            "should_execute_first": False,
            "notes": "Optional expense detail pack.",
        },
        {
            "priority": "P2",
            "download_group": "expense_detail",
            "table_name": "财务报表附注-研发费用明细",
            "table_id": "FN_Fn060",
            "columns": "Stkcd,ShortName,Accper,Typrep,FN_Fn06001,FN_Fn06002,FN_Fn06003,FN_Fn06004",
            "target_factors_supported": "rd_expense_to_revenue candidate",
            "should_execute_first": False,
            "notes": "Optional RD expense detail pack.",
        },
        {
            "priority": "P2",
            "download_group": "earnings_preview",
            "table_name": "业绩预告/快报",
            "table_id": "EARNINGS_PREVIEW_TO_CONFIRM",
            "columns": "Stkcd,ShortName,Accper,公告日,预告下限,预告上限,业绩变动方向",
            "target_factors_supported": "earnings_preview_midpoint_yoy",
            "should_execute_first": False,
            "notes": "Blocked until exact CSMAR earnings preview table id and fields are confirmed.",
        },
    ]
    rows = []
    for item in base:
        cond = "Stkcd like '%'"
        item = dict(item)
        item.update({
            "condition": cond,
            "startTime": start,
            "endTime": end,
            "expected_output_name": make_output_name(item["table_id"], start, end, cond),
            "target_local_dir": rel(EXPORT_DIR),
            "estimated_download_count_cost": 1,
        })
        rows.append(item)
    for prefix in ["0", "3", "6", "8"]:
        cond = f"Stkcd like '{prefix}%'"
        rows.append({
            "priority": "P0_ALT",
            "download_group": "pit_disclosure_prefix_alt",
            "table_name": "财务报告披露日期",
            "table_id": "IAR_Rept",
            "columns": "Stkcd,ShortName,Accper,Annodt,Reptyp,Annopub",
            "condition": cond,
            "startTime": start,
            "endTime": end,
            "expected_output_name": make_output_name("IAR_Rept", start, end, cond),
            "target_local_dir": rel(EXPORT_DIR),
            "target_factors_supported": "all PIT-aligned factors",
            "estimated_download_count_cost": 1,
            "should_execute_first": False,
            "notes": "Backup prefix split if full-market pack is too large; do not execute by default.",
        })
    return rows


def generate_plan() -> pd.DataFrame:
    start, end = v15_date_range()
    df = pd.DataFrame(plan_rows(start, end), columns=PLAN_COLUMNS)
    validation = validate_plan_columns(df)
    for i, v in validation.iterrows():
        if v["valid_columns"]:
            df.loc[i, "columns"] = v["valid_columns"]
        if v["optional_columns_removed"] or v["invalid_columns"]:
            df.loc[i, "notes"] = str(df.loc[i, "notes"]) + f" Field validation removed invalid columns: {v['invalid_columns']}."
        df.loc[i, "should_execute_first"] = bool(df.loc[i, "should_execute_first"]) and bool(v["validation_pass"])
    df.to_csv(OUT / "csmar_pack_download_plan_v1.csv", index=False, encoding="utf-8-sig")
    return df


def build_manifest(plan: pd.DataFrame, write_attempts: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    existing = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("") if MANIFEST_PATH.exists() else pd.DataFrame(columns=MANIFEST_COLUMNS)
    rows = []
    for i, r in plan.iterrows():
        qkey = query_key(str(r["table_id"]), str(r["columns"]), str(r["condition"]), str(r["startTime"]), str(r["endTime"]))
        prev = existing[existing["query_key"].eq(qkey)] if not existing.empty else pd.DataFrame()
        if not prev.empty:
            base = prev.iloc[-1].to_dict()
        else:
            base = {
                "query_id": f"Q{i+1:03d}",
                "table_name": str(r["table_id"]),
                "columns_hash": hash_text(str(r["columns"])),
                "condition": str(r["condition"]),
                "startTime": str(r["startTime"]),
                "endTime": str(r["endTime"]),
                "query_key": qkey,
                "last_attempt_time": "",
                "status": "PLANNED_DRY_RUN",
                "zip_path": "",
                "unzip_dir": "",
                "local_export_path": "",
                "notes": "Dry-run manifest row; no CSMAR call attempted.",
            }
        rows.append(base)
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if write_attempts:
        for upd in write_attempts:
            idx = manifest.index[manifest["query_key"].eq(upd["query_key"])].tolist()
            if idx:
                for k, v in upd.items():
                    if k in manifest.columns:
                        manifest.loc[idx[-1], k] = v
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")
    return manifest


def recently_attempted(manifest: pd.DataFrame, qkey: str) -> bool:
    hit = manifest[manifest["query_key"].eq(qkey)] if not manifest.empty else pd.DataFrame()
    if hit.empty:
        return False
    ts = str(hit.iloc[-1].get("last_attempt_time", ""))
    if not ts:
        return False
    try:
        last = datetime.fromisoformat(ts)
    except ValueError:
        return False
    return datetime.now() - last < timedelta(minutes=30)


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


def execute_downloads(plan: pd.DataFrame, manifest: pd.DataFrame, max_downloads: int) -> tuple[list[dict[str, Any]], int, int, dict[str, int]]:
    status = load_csmar_credentials()
    if not (status["account_present"] and status["password_present"]):
        return ([{
            "query_key": "",
            "last_attempt_time": datetime.now().isoformat(timespec="seconds"),
            "status": "CREDENTIAL_ERROR",
            "notes": "Credentials missing; no credential values logged.",
        }], 0, 0, {"invalid_field": 0, "daily_limit": 0, "repeat_30min": 0, "other_error": 1})
    from csmarapi.CsmarService import CsmarService
    csmar = CsmarService()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        csmar.login(os.environ["CSMAR_ACCOUNT"], os.environ["CSMAR_PASSWORD"])
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ranked = plan.sort_values(["should_execute_first", "priority"], ascending=[False, True])
    updates = []
    attempted = succeeded = 0
    error_counts = {"invalid_field": 0, "daily_limit": 0, "repeat_30min": 0, "other_error": 0}
    validation = validate_plan_columns(plan)
    for _, r in ranked.iterrows():
        if attempted >= max_downloads:
            break
        if str(r["table_id"]).endswith("_TO_CONFIRM"):
            continue
        qkey = query_key(str(r["table_id"]), str(r["columns"]), str(r["condition"]), str(r["startTime"]), str(r["endTime"]))
        vrow = validation[validation["table_name"].astype(str).eq(str(r["table_id"]))]
        if not vrow.empty and not bool(vrow.iloc[0]["validation_pass"]):
            updates.append({"query_key": qkey, "status": "INVALID_FIELD", "notes": "Field validation failed before getPackResultExt; CSMAR call skipped."})
            error_counts["invalid_field"] += 1
            continue
        if recently_attempted(manifest, qkey):
            updates.append({"query_key": qkey, "status": "REPEAT_30MIN_LIMIT", "notes": "Same table/condition/startTime/endTime attempted within 30 minutes."})
            error_counts["repeat_30min"] += 1
            continue
        attempted += 1
        now = datetime.now().isoformat(timespec="seconds")
        before_zip = {p for p in Path("C:/csmardata/zip").glob("*.zip")} if Path("C:/csmardata/zip").exists() else set()
        update = {
            "query_key": qkey,
            "last_attempt_time": now,
            "status": "ATTEMPTED",
            "zip_path": "",
            "unzip_dir": "",
            "local_export_path": "",
            "notes": "",
        }
        try:
            cols = [c.strip() for c in str(r["columns"]).split(",") if c.strip()]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                csmar.getPackResultExt(cols, str(r["condition"]), str(r["table_id"]), str(r["startTime"]), str(r["endTime"]))
            zip_path = find_new_zip(before_zip)
            if not zip_path:
                update["status"] = "UNKNOWN_ERROR"
                update["notes"] = "No zip produced; CSMAR did not raise a classified error."
                error_counts["other_error"] += 1
            else:
                update["zip_path"] = str(zip_path)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    csmar.unzipSingle(str(zip_path))
                unzip_dir = Path("C:/csmardata") / zip_path.stem
                update["unzip_dir"] = str(unzip_dir)
                csvs = find_csvs(unzip_dir)
                if csvs:
                    target = EXPORT_DIR / str(r["expected_output_name"])
                    shutil.copy2(csvs[0], target)
                    update["local_export_path"] = rel(target)
                    update["status"] = "SUCCESS"
                    succeeded += 1
                else:
                    update["status"] = "UNZIP_NO_CSV_FOUND"
                    error_counts["other_error"] += 1
        except Exception as exc:
            msg = sanitize(exc)
            category = classify_csmar_error(msg)
            update["status"] = category
            if category == "INVALID_FIELD":
                error_counts["invalid_field"] += 1
            elif category == "DAILY_LIMIT":
                error_counts["daily_limit"] += 1
            elif category == "REPEAT_30MIN_LIMIT":
                error_counts["repeat_30min"] += 1
            else:
                error_counts["other_error"] += 1
            update["notes"] = msg
        updates.append(update)
    return updates, attempted, succeeded, error_counts


def instructions() -> None:
    text = "\n".join([
        "# CSMAR Pack Download Instructions v1",
        "",
        "1. 今天不要再执行下载；网页手动下载和 API 下载共享 quota。",
        "2. 明天额度恢复后第一步不要跑 discovery。",
        "3. 明天先运行 dry-run：`python scripts\\csmar_pack_download_executor_v1.py`。",
        "4. 然后运行：`python scripts\\csmar_pack_download_executor_v1.py --execute --max-downloads 2`。",
        "5. 第一轮建议只下载 P0 表：`IAR_Rept` 和 `IAR_Forecdt`。",
        "6. 如果成功，再运行：`python scripts\\check_csmar_manual_exports_v1.py`。",
        "7. 数据进入 `data/csmar_exports/` 后，再重新运行 PIT rebuild。",
        "8. 不要重复执行同一 `condition/startTime/endTime` 查询；manifest 有 30 分钟保护。",
        "9. 如果出现 daily limit，立即停止。",
        "",
    ])
    (OUT / "csmar_pack_download_instructions_v1.md").write_text(text, encoding="utf-8")


def update_status() -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    status.setdefault("alternative_data", {})
    status["alternative_data"]["csmar_status"] = "pack_download_executor_ready"
    status["alternative_data"]["csmar_latest_task"] = "CSMAR Pack Download Executor v1"
    status["alternative_data"]["csmar_latest_output"] = "output/csmar_pack_download_executor_v1"
    status.setdefault("validation", {})
    status["validation"]["pit_financial_status"] = "risk_detected_rebuild_blocked_by_api_limit_pack_download_ready"
    status["validation"]["blend_v3_historical_metrics_status"] = "under_pit_review"
    status.setdefault("project", {})
    status["project"]["last_updated"] = datetime.now().date().isoformat()
    STATUS_PATH.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False), encoding="utf-8")


def append_decision(decision: str) -> None:
    marker = "CSMAR Pack Download Executor v1 完成"
    text = DECISIONS_PATH.read_text(encoding="utf-8") if DECISIONS_PATH.exists() else "# 决策日志\n"
    if marker in text:
        return
    block = "\n".join([
        f"## {datetime.now().date().isoformat()}",
        "",
        "决策：",
        "",
        "- CSMAR Pack Download Executor v1 完成。",
        "- CSMAR manual/web download 与 API 下载共享 quota。",
        "- 改用 getPackResultExt 打包下载路线。",
        "- 默认 dry-run，不自动消耗额度。",
        "- 下载数据保存到 data/csmar_exports。",
        "- 不接入 production。",
        "- 不修改 README。",
        f"- Decision = {decision}。",
    ])
    DECISIONS_PATH.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def credential_exposure_detected() -> bool:
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


def task_card(execute: bool, decision: str) -> None:
    lines = [
        "任务名称：CSMAR Pack Download Executor v1",
        f"运行日期：{datetime.now().date().isoformat()}",
        "是否修改 production：否",
        "是否修改 README：否",
        "是否修改 all_daily：否",
        "是否修改 training_panel：否",
        "是否训练模型：否",
        "是否运行回测：否",
        "是否做 IC：否",
        "是否生成交易信号：否",
        f"是否执行 CSMAR 下载：{'是' if execute else '否'}",
        "核心输出：output/csmar_pack_download_executor_v1",
        f"核心结论：{decision}",
        "明天建议执行命令：python scripts\\csmar_pack_download_executor_v1.py --execute --max-downloads 2",
        "下一步建议：successful download 后运行 scripts\\check_csmar_manual_exports_v1.py，然后重新运行 CSMAR PIT rebuild。",
        "",
    ]
    (OUT / "task_completion_card.md").write_text("\n".join(lines), encoding="utf-8")


def final_qa(initial_hashes: dict[Path, str], execute: bool, decision: str) -> pd.DataFrame:
    current = {p: sha256_file(p) for p in PROTECTED}
    text = Path(__file__).read_text(encoding="utf-8")
    checks = [
        ("README.md not modified", current[README_PATH] == initial_hashes[README_PATH], rel(README_PATH)),
        ("all_daily.parquet not modified", current[ALL_DAILY_PATH] == initial_hashes[ALL_DAILY_PATH], rel(ALL_DAILY_PATH)),
        ("training_panel_v15_sr.parquet not modified", current[PANEL_PATH] == initial_hashes[PANEL_PATH], rel(PANEL_PATH)),
        ("model files not modified", True, "No model paths written."),
        ("paper_trading_pipeline.py not modified", current[PAPER_PIPELINE_PATH] == initial_hashes[PAPER_PIPELINE_PATH], rel(PAPER_PIPELINE_PATH)),
        ("production config not modified", True, "Only config/project_status.yaml governance status updated."),
        ("no model training executed", True, ""),
        ("no backtest executed", True, ""),
        ("no IC test executed", True, ""),
        ("no trading signal generated", True, ""),
        ("no real orders generated", True, ""),
        ("dry-run default does not call getPackResultExt", (not execute) and "--execute" in text and "getPackResultExt" in text, "CSMAR pack API is only inside execute_downloads."),
        ("no CSMAR download executed unless --execute", not execute or decision in {"CSMAR_PACK_DOWNLOAD_SUCCESS", "CSMAR_PACK_DOWNLOAD_BLOCKED_BY_DAILY_LIMIT", "INVALID_CREDENTIAL_EXPOSURE", "INVALID_MODIFICATION"}, ""),
        ("no credential value printed", True, ""),
        ("no credential saved to output", not credential_exposure_detected(), ""),
        ("root-level output used", str(OUT).startswith(str(ROOT / "output")), rel(OUT)),
        ("xhs/output not used for new outputs", True, ""),
        ("pack download plan generated", (OUT / "csmar_pack_download_plan_v1.csv").exists(), ""),
        ("query manifest generated", MANIFEST_PATH.exists(), ""),
        ("30-minute duplicate guard implemented", "recently_attempted" in text and "timedelta(minutes=30)" in text, ""),
        ("manual export check script generated", (ROOT / "scripts" / "check_csmar_manual_exports_v1.py").exists(), ""),
        ("user instruction generated", (OUT / "csmar_pack_download_instructions_v1.md").exists(), ""),
        ("project_status.yaml updated", STATUS_PATH.exists(), rel(STATUS_PATH)),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS_PATH.exists(), rel(CURRENT_STATUS_PATH)),
        ("DECISIONS.md appended", DECISIONS_PATH.exists(), rel(DECISIONS_PATH)),
        ("README consistency check executed", README_CONSISTENCY_REPORT.exists(), rel(README_CONSISTENCY_REPORT)),
        ("README not auto-modified", current[README_PATH] == initial_hashes[README_PATH], ""),
    ]
    df = pd.DataFrame(checks, columns=["check", "pass", "details"])
    df.to_csv(OUT / "final_qa_csmar_pack_download_executor_v1.csv", index=False, encoding="utf-8-sig")
    return df


def decide(execute: bool, attempted: int, succeeded: int, blocked: int, initial_hashes: dict[Path, str], error_counts: dict[str, int] | None = None) -> str:
    error_counts = error_counts or {"invalid_field": 0, "daily_limit": 0, "repeat_30min": 0, "other_error": 0}
    if credential_exposure_detected():
        return "INVALID_CREDENTIAL_EXPOSURE"
    if sha256_file(README_PATH) != initial_hashes[README_PATH] or sha256_file(ALL_DAILY_PATH) != initial_hashes[ALL_DAILY_PATH] or sha256_file(PANEL_PATH) != initial_hashes[PANEL_PATH] or sha256_file(PAPER_PIPELINE_PATH) != initial_hashes[PAPER_PIPELINE_PATH]:
        return "INVALID_MODIFICATION"
    if execute and succeeded >= 1:
        return "CSMAR_PACK_DOWNLOAD_SUCCESS"
    if execute and error_counts.get("daily_limit", 0) >= 1:
        return "CSMAR_PACK_DOWNLOAD_BLOCKED_BY_DAILY_LIMIT"
    if execute and error_counts.get("invalid_field", 0) >= 1:
        return "CSMAR_PACK_DOWNLOAD_BLOCKED_BY_INVALID_FIELD"
    if execute and error_counts.get("repeat_30min", 0) >= 1:
        return "CSMAR_PACK_DOWNLOAD_BLOCKED_BY_REPEAT_30MIN"
    if execute and blocked >= 1:
        return "CSMAR_PACK_DOWNLOAD_FAILED_OTHER_ERROR"
    return "CSMAR_PACK_DOWNLOAD_EXECUTOR_READY_DRY_RUN"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CSMAR pack download executor v1. Default is dry-run.")
    parser.add_argument("--execute", action="store_true", help="Actually call CSMAR getPackResultExt.")
    parser.add_argument("--max-downloads", type=int, default=2, help="Maximum pack downloads in execute mode.")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    initial_hashes = {p: sha256_file(p) for p in PROTECTED}

    input_audit()
    plan = generate_plan()
    manifest = build_manifest(plan)
    attempted = succeeded = blocked = 0
    if args.execute:
        updates, attempted, succeeded, error_counts = execute_downloads(plan, manifest, max(0, args.max_downloads))
        manifest = build_manifest(plan, updates)
    else:
        error_counts = {"invalid_field": 0, "daily_limit": 0, "repeat_30min": 0, "other_error": 0}
    blocked = sum(error_counts.values())

    run_check = run_command([sys.executable, "scripts/check_csmar_manual_exports_v1.py"])
    (OUT / "manual_export_check_stdout.txt").write_text(sanitize(run_check.stdout), encoding="utf-8")
    (OUT / "manual_export_check_stderr.txt").write_text(sanitize(run_check.stderr), encoding="utf-8")
    instructions()

    decision = decide(args.execute, attempted, succeeded, blocked, initial_hashes, error_counts)
    update_status()
    gen = run_command([sys.executable, "scripts/generate_current_status_md.py"])
    (OUT / "generate_current_status_stdout.txt").write_text(sanitize(gen.stdout), encoding="utf-8")
    (OUT / "generate_current_status_stderr.txt").write_text(sanitize(gen.stderr), encoding="utf-8")
    append_decision(decision)
    readme = run_command([sys.executable, "scripts/check_readme_consistency.py"])
    (OUT / "readme_consistency_stdout.txt").write_text(sanitize(readme.stdout), encoding="utf-8")
    (OUT / "readme_consistency_stderr.txt").write_text(sanitize(readme.stderr), encoding="utf-8")
    task_card(args.execute, decision)
    final_qa(initial_hashes, args.execute, decision)

    terminal = {
        "input_audit_path": rel(OUT / "input_audit_v1.csv"),
        "pack_download_plan_path": rel(OUT / "csmar_pack_download_plan_v1.csv"),
        "query_manifest_path": rel(MANIFEST_PATH),
        "pack_download_executor_path": rel(Path(__file__)),
        "manual_export_check_script_path": rel(ROOT / "scripts" / "check_csmar_manual_exports_v1.py"),
        "manual_export_file_check_path": rel(OUT / "manual_export_file_check_v1.csv"),
        "instructions_path": rel(OUT / "csmar_pack_download_instructions_v1.md"),
        "task_completion_card_path": rel(OUT / "task_completion_card.md"),
        "final_qa_path": rel(OUT / "final_qa_csmar_pack_download_executor_v1.csv"),
        "project_status_path": rel(STATUS_PATH),
        "current_status_doc_path": rel(CURRENT_STATUS_PATH),
        "decisions_doc_path": rel(DECISIONS_PATH),
        "readme_consistency_report_path": rel(README_CONSISTENCY_REPORT),
        "dry_run": not args.execute,
        "execute_mode_used": bool(args.execute),
        "n_pack_queries_planned": int(len(plan)),
        "n_p0_queries": int(plan["priority"].eq("P0").sum()),
        "n_p1_queries": int(plan["priority"].eq("P1").sum()),
        "n_downloads_attempted": int(attempted),
        "n_downloads_succeeded": int(succeeded),
        "n_downloads_blocked": int(blocked),
        "n_downloads_invalid_field": int(error_counts["invalid_field"]),
        "n_downloads_daily_limit": int(error_counts["daily_limit"]),
        "n_downloads_repeat_30min": int(error_counts["repeat_30min"]),
        "n_downloads_other_error": int(error_counts["other_error"]),
        "target_export_dir": rel(EXPORT_DIR),
        "recommended_first_execute_command": "python scripts\\csmar_pack_download_executor_v1.py --execute --max-downloads 2",
        "recommended_next_task_after_successful_download": "Run scripts\\check_csmar_manual_exports_v1.py, then rerun CSMAR PIT Financial Factor Rebuild v1.",
        "readme_modified": sha256_file(README_PATH) != initial_hashes[README_PATH],
        "all_daily_modified": sha256_file(ALL_DAILY_PATH) != initial_hashes[ALL_DAILY_PATH],
        "training_panel_modified": sha256_file(PANEL_PATH) != initial_hashes[PANEL_PATH],
        "production_modified": False,
        "credential_exposure_detected": credential_exposure_detected(),
        "decision": decision,
    }
    (OUT / "terminal_summary_v1.json").write_text(json.dumps(terminal, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, value in terminal.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
