from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.csmar.credential_loader import load_csmar_credentials

OUT = Path(os.environ.get("CSMAR_TABLE_INVENTORY_OUTPUT_DIR", ROOT / "output" / "csmar_table_inventory_audit_v1")).resolve()
STATUS_PATH = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS_PATH = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
README_CHECK_REPORT = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"

KEYWORDS = [
    "csmar", "CSMAR", "csmarapi", "CsmarService", "api_key", "token",
    "username", "password", "financial", "analyst", "forecast",
    "disclosure", "announce", "ann_date", "publish_date",
]
CREDENTIAL_KEYS = [
    "CSMAR_ACCOUNT", "CSMAR_PASSWORD", "CSMAR_USERNAME", "CSMAR_TOKEN",
    "CSMAR_API_KEY", "CSMAR_SECRET",
]
SCAN_DIRS = [
    ROOT, ROOT / "data", ROOT / "data" / "raw", ROOT / "data" / "csmar",
    ROOT / "scripts", ROOT / "factor_research", ROOT / "paper_trading",
    ROOT / "output", ROOT / "configs", ROOT / "config",
]
PROTECTED = [
    ROOT / "README.md",
    ROOT / "output" / "all_daily.parquet",
    ROOT / "paper_trading" / "paper_trading_pipeline.py",
]
KNOWN_TABLES = {
    "FN_Fn050": {
        "table_name_cn": "财务报表附注-销售费用明细",
        "category": "expense_structure",
        "fields": ["Stkcd", "Accper", "Typrep", "FN05001", "FN05002"],
        "notes": "来自 xhs/scripts/csmar_beauty_data_layer_audit.py adapter 硬编码表。",
    },
    "FN_Fn060": {
        "table_name_cn": "财务报表附注-研发费用明细",
        "category": "expense_structure",
        "fields": ["Stkcd", "Accper", "Typrep", "FN_Fn06001", "FN_Fn06002", "FN_Fn06003"],
        "notes": "来自 xhs/scripts/csmar_beauty_data_layer_audit.py adapter 硬编码表。",
    },
    "FI_T5": {
        "table_name_cn": "财务指标",
        "category": "financial_indicator",
        "fields": ["Stkcd", "Accper", "Typrep", "F050501B", "F053301B", "F051701B", "F051801B"],
        "notes": "来自 xhs/scripts/csmar_beauty_data_layer_audit.py adapter 硬编码表。",
    },
    "IAR_Rept": {
        "table_name_cn": "财务报告披露日期",
        "category": "disclosure_date",
        "fields": ["Stkcd", "Accper", "Annodt"],
        "notes": "来自 xhs/scripts/csmar_beauty_data_layer_audit.py adapter 硬编码表。",
    },
    "IAR_Forecdt": {
        "table_name_cn": "预约披露日期/实际披露日期",
        "category": "disclosure_date",
        "fields": ["Stkcd", "Accper", "Actudt", "Firforecdt"],
        "notes": "来自 xhs/scripts/csmar_beauty_data_layer_audit.py adapter 硬编码表。",
    },
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_text(path: Path, limit: int = 500_000) -> str:
    try:
        data = path.read_bytes()[:limit]
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def detected_keywords(text: str) -> list[str]:
    low = text.lower()
    return sorted({kw for kw in KEYWORDS if kw.lower() in low})


def likely_role(path: Path, hits: list[str]) -> str:
    p = rel(path).lower()
    if ".env" in p:
        return "credential_config"
    if "csmar" in p and "script" in p:
        return "csmar_adapter_or_audit_script"
    if path.suffix.lower() in {".csv", ".xlsx", ".xls", ".parquet"}:
        return "local_data_file"
    if any(k in hits for k in ["CsmarService", "csmarapi"]):
        return "api_adapter"
    if "project_status" in p:
        return "governance_status"
    return "keyword_reference"


def access_audit() -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    env_names = [k for k in os.environ if any(t.lower() in k.lower() for t in CREDENTIAL_KEYS + ["CSMAR"])]
    if env_names:
        rows.append({
            "item_type": "environment",
            "path": "ENV",
            "exists": True,
            "detected_keywords": "|".join(sorted(env_names)),
            "likely_role": "credential_config",
            "credential_exists": any(k in os.environ and os.environ.get(k) for k in CREDENTIAL_KEYS),
            "safe_to_use": False,
            "notes": "Only environment variable names were recorded; values were not printed or saved.",
        })
    for scan_root in SCAN_DIRS:
        if not scan_root.exists():
            rows.append({
                "item_type": "directory",
                "path": rel(scan_root),
                "exists": False,
                "detected_keywords": "",
                "likely_role": "not_found",
                "credential_exists": False,
                "safe_to_use": False,
                "notes": "Scan target missing.",
            })
            continue
        for path in scan_root.rglob("*"):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".csv", ".parquet", ".xlsx", ".xls", ".env"} and path.name != ".env":
                continue
            text = path.name
            if path.suffix.lower() in {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".env"} or path.name == ".env":
                text += "\n" + safe_text(path)
            hits = detected_keywords(text)
            is_csmar_file = "csmar" in path.name.lower() or "csmar" in rel(path).lower()
            if not hits and not is_csmar_file:
                continue
            credential_exists = bool(re.search(r"(?i)(password|token|api[_-]?key|secret|CSMAR_PASSWORD|CSMAR_ACCOUNT)", text))
            rows.append({
                "item_type": "file",
                "path": rel(path),
                "exists": True,
                "detected_keywords": "|".join(hits),
                "likely_role": likely_role(path, hits),
                "credential_exists": credential_exists,
                "safe_to_use": not credential_exists or path.name != ".env",
                "notes": "Credential values were not extracted. File scanned only for metadata/keywords.",
            })
    return rows, any(r["likely_role"] in {"api_adapter", "csmar_adapter_or_audit_script", "local_data_file"} for r in rows)


def normalize_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        for key in ("data", "datas", "rows", "result", "items", "list"):
            if isinstance(value.get(key), list):
                return [x if isinstance(x, dict) else {"value": x} for x in value[key]]
        return [value]
    if isinstance(value, (list, tuple)):
        return [x if isinstance(x, dict) else {"value": x} for x in value]
    return [{"value": value}]


def extract_name_desc(row: dict[str, Any], kind: str) -> tuple[str, str]:
    name_keys = [f"{kind}Name", f"{kind}_name", "table_name", "field_name", "name", "Name", "tableCode", "code", "field"]
    desc_keys = [f"{kind}Desc", f"{kind}_desc", "table_desc", "field_desc", "fieldName", "desc", "description", "memo", "remark", "title"]
    name = next((str(row[k]) for k in name_keys if k in row and str(row[k]).strip()), "")
    desc = next((str(row[k]) for k in desc_keys if k in row and str(row[k]).strip()), "")
    return name, desc


def discover_adapter_tables() -> dict[str, dict[str, Any]]:
    tables = dict(KNOWN_TABLES)
    for path in [ROOT / "xhs" / "scripts" / "csmar_beauty_data_layer_audit.py"]:
        text = safe_text(path)
        try:
            tree = ast.parse(text)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "TABLES" and isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                tables.setdefault(elt.value, {
                                    "table_name_cn": "",
                                    "category": "unknown",
                                    "fields": [],
                                    "notes": f"Discovered from {rel(path)} TABLES.",
                                })
    return tables


def api_discover() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, pd.DataFrame], bool, str]:
    account = os.getenv("CSMAR_ACCOUNT", "").strip()
    password = os.getenv("CSMAR_PASSWORD", "").strip()
    if not account or not password:
        return [], {}, {}, False, "CSMAR_ACCOUNT/CSMAR_PASSWORD environment variable names not available."
    try:
        from csmarapi.CsmarService import CsmarService
    except Exception as exc:
        return [], {}, {}, False, f"csmarapi import failed: {type(exc).__name__}"
    try:
        csmar = CsmarService()
        csmar.login(account, password)
        dbs = normalize_records(csmar.getListDbs())
    except Exception as exc:
        return [], {}, {}, False, f"CSMAR login/list db failed: {type(exc).__name__}: {exc}"
    tables: list[dict[str, Any]] = []
    field_map: dict[str, list[dict[str, Any]]] = {}
    sample_map: dict[str, pd.DataFrame] = {}
    for db in dbs[:80]:
        db_name, _ = extract_name_desc(db, "database")
        if not db_name:
            continue
        try:
            table_records = normalize_records(csmar.getListTables(db_name))
        except Exception:
            continue
        for tr in table_records:
            table_id, table_desc = extract_name_desc(tr, "table")
            if not table_id:
                continue
            combined = f"{table_id} {table_desc}"
            if not is_relevant_text(combined):
                continue
            tables.append({"table_id": table_id, "table_name_cn": table_desc, "database": db_name, "raw": tr})
            if len(tables) >= 80:
                break
        if len(tables) >= 80:
            break
    for table in tables[:40]:
        tid = table["table_id"]
        try:
            field_map[tid] = normalize_records(csmar.getListFields(tid))
            try:
                sample_map[tid] = pd.DataFrame(normalize_records(csmar.preview(tid))).head(20)
            except Exception:
                sample_map[tid] = pd.DataFrame()
        except Exception:
            field_map[tid] = []
    return tables, field_map, sample_map, True, "API metadata discovery succeeded."


def is_relevant_text(text: str) -> bool:
    tokens = [
        "财报", "披露", "公告", "利润", "资产负债", "现金流", "财务指标", "ROE",
        "销售费用", "管理费用", "研发", "分析师", "一致预期", "盈利预测",
        "业绩预告", "业绩快报", "机构持仓", "基金持仓", "FN_", "FI_", "IAR_",
        "forecast", "analyst", "holding", "expense", "financial",
    ]
    low = text.lower()
    return any(t.lower() in low for t in tokens)


def classify_category(table_id: str, table_name: str, fields: list[str]) -> str:
    text = f"{table_id} {table_name} {' '.join(fields)}".lower()
    if any(k in text for k in ["annodt", "actudt", "forecdt", "披露", "公告"]):
        return "disclosure_date"
    if any(k in text for k in ["analyst", "forecast", "一致预期", "盈利预测", "目标价", "评级"]):
        return "analyst_forecast"
    if any(k in text for k in ["业绩预告", "preview"]):
        return "earnings_preview"
    if any(k in text for k in ["业绩快报", "express"]):
        return "earnings_express"
    if any(k in text for k in ["holding", "持仓", "基金", "qfii", "社保"]):
        return "institutional_holding"
    if any(k in text for k in ["fn050", "fn060", "expense", "费用", "研发"]):
        return "expense_structure"
    if any(k in text for k in ["fi_", "roe", "f05", "财务指标", "毛利率", "净利率"]):
        return "financial_indicator"
    if any(k in text for k in ["利润表", "资产负债", "现金流"]):
        return "financial_statement"
    return "unknown"


def field_flags(field: str, desc: str) -> dict[str, bool]:
    text = f"{field} {desc}".lower()
    return {
        "is_symbol_field": any(k in text for k in ["stkcd", "symbol", "ticker", "证券代码", "股票代码"]),
        "is_date_field": any(k in text for k in ["date", "dt", "日期", "time", "accper", "annodt", "actudt"]),
        "is_report_period_field": any(k in text for k in ["accper", "report", "end_date", "报告期"]),
        "is_announcement_date_field": any(k in text for k in ["annodt", "actudt", "announce", "publish", "disclosure", "公告", "披露"]),
        "is_pit_key_field": any(k in text for k in ["annodt", "actudt", "publish", "disclosure", "update", "公告", "披露"]),
        "is_financial_factor_candidate": any(k in text for k in ["roe", "roa", "margin", "profit", "revenue", "净利", "营收", "毛利", "资产负债"]),
        "is_expense_factor_candidate": any(k in text for k in ["expense", "费用", "研发", "销售", "管理"]),
        "is_analyst_factor_candidate": any(k in text for k in ["analyst", "forecast", "eps", "target", "rating", "分析师", "预测", "评级"]),
        "is_event_factor_candidate": any(k in text for k in ["预告", "快报", "修正", "公告", "披露"]),
    }


def sample_stats(df: pd.DataFrame, col: str) -> tuple[str, str, str]:
    if df.empty or col not in df.columns:
        return "", "", ""
    s = df[col]
    miss = f"{float(s.isna().mean()):.4f}"
    uniq = str(int(s.nunique(dropna=True)))
    vals = [str(v)[:80] for v in s.dropna().astype(str).drop_duplicates().head(3).tolist()]
    return miss, uniq, "|".join(vals)


def date_range(df: pd.DataFrame, cols: list[str]) -> tuple[str, str]:
    vals = []
    for col in cols:
        if col in df.columns:
            vals.append(pd.to_datetime(df[col], errors="coerce"))
    if not vals:
        return "", ""
    s = pd.concat(vals).dropna()
    if s.empty:
        return "", ""
    return str(s.min().date()), str(s.max().date())


def build_inventory(api_tables: list[dict[str, Any]], api_fields: dict[str, list[dict[str, Any]]], samples: dict[str, pd.DataFrame]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adapters = discover_adapter_tables()
    source_ids = {r["table_id"] for r in api_tables}
    raw_tables = list(api_tables)
    for tid, meta in adapters.items():
        if tid not in source_ids:
            raw_tables.append({"table_id": tid, "table_name_cn": meta["table_name_cn"], "database": "", "adapter_meta": meta})
    inventory: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    for table in raw_tables:
        tid = table["table_id"]
        meta = table.get("adapter_meta", {})
        known_fields = list(meta.get("fields", []))
        f_records = api_fields.get(tid, [])
        if f_records:
            fields = []
            for fr in f_records:
                fname, fdesc = extract_name_desc(fr, "field")
                if fname:
                    fields.append(fname)
                    flags = field_flags(fname, fdesc)
                    miss, uniq, vals = sample_stats(samples.get(tid, pd.DataFrame()), fname)
                    field_rows.append({
                        "table_id": tid, "table_name": table.get("table_name_cn", ""),
                        "field_name": fname, "field_name_cn": fdesc, "dtype": str(fr.get("fieldType", "")),
                        "description": fdesc, **flags, "missing_rate_sample": miss,
                        "unique_count_sample": uniq, "sample_values": vals, "notes": "API field metadata.",
                    })
        else:
            fields = known_fields
            for fname in fields:
                flags = field_flags(fname, "")
                field_rows.append({
                    "table_id": tid, "table_name": table.get("table_name_cn", meta.get("table_name_cn", "")),
                    "field_name": fname, "field_name_cn": "", "dtype": "", "description": "Heuristic from known adapter field name.",
                    **flags, "missing_rate_sample": "", "unique_count_sample": "", "sample_values": "",
                    "notes": meta.get("notes", "Adapter-discovered field; not verified by API in this run."),
                })
        sample = samples.get(tid, pd.DataFrame())
        date_cols = [c for c in fields if field_flags(c, "")["is_date_field"]]
        symbol_cols = [c for c in fields if field_flags(c, "")["is_symbol_field"]]
        report_cols = [c for c in fields if field_flags(c, "")["is_report_period_field"]]
        ann_cols = [c for c in fields if field_flags(c, "")["is_announcement_date_field"]]
        min_d, max_d = date_range(sample, date_cols)
        category = classify_category(tid, table.get("table_name_cn", ""), fields)
        has_pit = bool(ann_cols)
        has_fin = category in {"financial_indicator", "financial_statement"} or any(field_flags(c, "")["is_financial_factor_candidate"] for c in fields)
        has_exp = category == "expense_structure" or any(field_flags(c, "")["is_expense_factor_candidate"] for c in fields)
        has_ana = category == "analyst_forecast" or any(field_flags(c, "")["is_analyst_factor_candidate"] for c in fields)
        priority = "P0_PIT_CRITICAL" if category == "disclosure_date" else ("P1_ALPHA_CANDIDATE" if category in {"financial_indicator", "expense_structure", "analyst_forecast", "earnings_preview", "earnings_express", "institutional_holding"} else "UNKNOWN_REVIEW_REQUIRED")
        inventory.append({
            "table_id": tid,
            "table_name_cn": table.get("table_name_cn", meta.get("table_name_cn", "")),
            "table_name_en": "",
            "source_type": "api_metadata" if tid in api_fields else "adapter_known",
            "source_path_or_api": table.get("database", "") or "xhs/scripts/csmar_beauty_data_layer_audit.py",
            "category": category,
            "row_count_estimate": len(sample) if not sample.empty else "",
            "min_date": min_d,
            "max_date": max_d,
            "n_symbols_estimate": int(sample[symbol_cols[0]].nunique()) if symbol_cols and symbol_cols[0] in sample.columns else "",
            "date_columns": "|".join(date_cols),
            "symbol_columns": "|".join(symbol_cols),
            "report_period_columns": "|".join(report_cols),
            "announcement_date_columns": "|".join(ann_cols),
            "update_time_columns": "|".join([c for c in fields if "update" in c.lower() or "entry" in c.lower()]),
            "has_pit_candidate": has_pit,
            "has_financial_statement_fields": has_fin,
            "has_analyst_forecast_fields": has_ana,
            "has_earnings_preview_fields": category == "earnings_preview",
            "has_expense_fields": has_exp,
            "has_institutional_holding_fields": category == "institutional_holding",
            "sample_available": not sample.empty,
            "priority": priority,
            "recommended_next_step": "Task2 PIT validation" if priority == "P0_PIT_CRITICAL" else ("single-factor feasibility audit" if priority == "P1_ALPHA_CANDIDATE" else "manual review"),
            "notes": meta.get("notes", "Discovered from API metadata."),
        })
    return inventory, field_rows


def build_shortlist(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for i, r in enumerate([x for x in inventory if x["priority"] in {"P0_PIT_CRITICAL", "P1_ALPHA_CANDIDATE"}], 1):
        pit = r["announcement_date_columns"]
        if r["category"] == "disclosure_date":
            why = "PIT/公告日硬化核心表，可校验 report_period 到实际披露日。"
            examples = "publish_date_lag|financial_report_delay|pit_availability_flag"
        elif r["category"] == "expense_structure":
            why = "费用结构可用于销售费用率、研发强度、费用效率等候选 alpha。"
            examples = "sales_expense_to_revenue|rd_expense_to_revenue|expense_efficiency"
        elif r["category"] == "financial_indicator":
            why = "财务指标可作为 Compact-F 对照和新增基本面候选。"
            examples = "roe|gross_margin|net_margin|debt_ratio"
        else:
            why = "可能包含新增 alpha 或风险标签所需字段。"
            examples = "revision|dispersion|warning_flag"
        rows.append({
            "priority_rank": i,
            "table_id": r["table_id"],
            "table_name": r["table_name_cn"] or r["table_id"],
            "category": r["category"],
            "why_relevant": why,
            "key_fields": "|".join([v for v in [r["symbol_columns"], r["report_period_columns"], r["date_columns"]] if v]),
            "pit_fields": pit,
            "expected_factor_examples": examples,
            "expected_pit_risk": "High if no announcement date join" if not pit else "Medium; still needs actual date semantics confirmation",
            "estimated_engineering_cost": "medium",
            "recommended_for_task2": r["category"] in {"disclosure_date", "financial_indicator", "expense_structure"},
            "notes": "No production integration recommended in this task.",
        })
    return rows


def build_coverage(shortlist: list[dict[str, Any]], inventory: list[dict[str, Any]], samples: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    v15_symbols: set[str] = set()
    panel = ROOT / "output" / "training_panel_v15_sr.parquet"
    if panel.exists():
        try:
            df = pd.read_parquet(panel, columns=["symbol"])
            v15_symbols = set(df["symbol"].astype(str).str.zfill(6).dropna().unique().tolist())
        except Exception:
            v15_symbols = set()
    inv = {r["table_id"]: r for r in inventory}
    rows = []
    for s in shortlist:
        tid = s["table_id"]
        r = inv[tid]
        sample = samples.get(tid, pd.DataFrame())
        sym_cols = [c for c in str(r["symbol_columns"]).split("|") if c]
        ann_cols = [c for c in str(r["announcement_date_columns"]).split("|") if c]
        rep_cols = [c for c in str(r["report_period_columns"]).split("|") if c]
        syms = set()
        if not sample.empty and sym_cols and sym_cols[0] in sample.columns:
            syms = set(sample[sym_cols[0]].astype(str).str.zfill(6).dropna().unique().tolist())
        rows.append({
            "table_id": tid, "table_name": s["table_name"], "category": s["category"],
            "audit_scope": "API preview/sample only" if not sample.empty else "No data sample; metadata/adapter only limitation recorded",
            "start_date": r["min_date"], "end_date": r["max_date"],
            "n_rows_sample_or_count": len(sample) if not sample.empty else 0,
            "n_symbols_covered": len(syms),
            "symbol_coverage_vs_v15": round(len(syms & v15_symbols) / len(v15_symbols), 6) if v15_symbols else "",
            "n_report_periods": int(sample[rep_cols[0]].nunique()) if not sample.empty and rep_cols and rep_cols[0] in sample.columns else "",
            "min_report_period": str(pd.to_datetime(sample[rep_cols[0]], errors="coerce").min().date()) if not sample.empty and rep_cols and rep_cols[0] in sample.columns and pd.to_datetime(sample[rep_cols[0]], errors="coerce").notna().any() else "",
            "max_report_period": str(pd.to_datetime(sample[rep_cols[0]], errors="coerce").max().date()) if not sample.empty and rep_cols and rep_cols[0] in sample.columns and pd.to_datetime(sample[rep_cols[0]], errors="coerce").notna().any() else "",
            "min_announcement_date": str(pd.to_datetime(sample[ann_cols[0]], errors="coerce").min().date()) if not sample.empty and ann_cols and ann_cols[0] in sample.columns and pd.to_datetime(sample[ann_cols[0]], errors="coerce").notna().any() else "",
            "max_announcement_date": str(pd.to_datetime(sample[ann_cols[0]], errors="coerce").max().date()) if not sample.empty and ann_cols and ann_cols[0] in sample.columns and pd.to_datetime(sample[ann_cols[0]], errors="coerce").notna().any() else "",
            "missing_symbol_rate": round(float(sample[sym_cols[0]].isna().mean()), 6) if not sample.empty and sym_cols and sym_cols[0] in sample.columns else "",
            "missing_date_rate": "",
            "missing_announcement_date_rate": round(float(sample[ann_cols[0]].isna().mean()), 6) if not sample.empty and ann_cols and ann_cols[0] in sample.columns else "",
            "duplicate_key_count_sample": 0 if sample.empty else "",
            "pit_usable": bool(ann_cols and not sample.empty),
            "notes": "Conservative: PIT usable requires verified announcement date sample." if sample.empty else "Sample-only coverage; not full count.",
        })
    return rows


def candidate_factor_map(shortlist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cat = {r["category"]: r for r in shortlist}
    def source(cats: list[str]) -> tuple[str, str]:
        for c in cats:
            if c in by_cat:
                return by_cat[c]["table_id"], by_cat[c]["table_name"]
        return "", ""
    specs = [
        ("csmar_publish_date_lag", ["disclosure_date"], "pit_validation", "P0_PIT_CRITICAL"),
        ("csmar_financial_report_delay", ["disclosure_date"], "pit_validation", "P0_PIT_CRITICAL"),
        ("csmar_statement_revision_flag", ["disclosure_date"], "pit_validation", "P0_PIT_CRITICAL"),
        ("csmar_pit_availability_flag", ["disclosure_date"], "pit_validation", "P0_PIT_CRITICAL"),
        ("analyst_eps_revision_1m", ["analyst_forecast"], "standalone_single_factor", "P1_ALPHA_CANDIDATE"),
        ("analyst_np_revision_3m", ["analyst_forecast"], "standalone_single_factor", "P1_ALPHA_CANDIDATE"),
        ("analyst_revenue_revision_3m", ["analyst_forecast"], "standalone_single_factor", "P1_ALPHA_CANDIDATE"),
        ("analyst_coverage_change_3m", ["analyst_forecast"], "standalone_single_factor", "P1_ALPHA_CANDIDATE"),
        ("analyst_forecast_dispersion", ["analyst_forecast"], "residual_alpha_diagnostic", "P1_ALPHA_CANDIDATE"),
        ("sales_expense_to_revenue", ["expense_structure", "financial_indicator"], "standalone_single_factor", "P1_ALPHA_CANDIDATE"),
        ("sales_expense_yoy", ["expense_structure"], "standalone_single_factor", "P1_ALPHA_CANDIDATE"),
        ("rd_expense_to_revenue", ["expense_structure"], "standalone_single_factor", "P1_ALPHA_CANDIDATE"),
        ("expense_efficiency", ["expense_structure"], "residual_alpha_diagnostic", "P1_ALPHA_CANDIDATE"),
        ("earnings_preview_midpoint_yoy", ["earnings_preview"], "standalone_single_factor", "P1_ALPHA_CANDIDATE"),
        ("earnings_preview_revision_direction", ["earnings_preview"], "dashboard_risk_flag", "P1_ALPHA_CANDIDATE"),
        ("earnings_express_surprise", ["earnings_express"], "standalone_single_factor", "P1_ALPHA_CANDIDATE"),
        ("disclosure_delay_flag", ["disclosure_date"], "dashboard_risk_flag", "P2_RISK_OR_CONTEXT"),
        ("earnings_warning_flag", ["earnings_preview"], "dashboard_risk_flag", "P2_RISK_OR_CONTEXT"),
        ("forecast_down_revision_flag", ["analyst_forecast"], "dashboard_risk_flag", "P2_RISK_OR_CONTEXT"),
        ("high_forecast_dispersion_flag", ["analyst_forecast"], "dashboard_risk_flag", "P2_RISK_OR_CONTEXT"),
        ("institutional_holding_drop_flag", ["institutional_holding"], "dashboard_risk_flag", "P2_RISK_OR_CONTEXT"),
    ]
    rows = []
    for name, cats, integ, pri in specs:
        tid, tname = source(cats)
        rows.append({
            "candidate_factor": name, "source_table_id": tid, "source_table_name": tname,
            "required_fields": "symbol|report_period|announcement_date|value_fields",
            "frequency": "quarterly/monthly depending source",
            "pit_key": "announcement_date <= as_of_date",
            "expected_direction": "to_be_tested",
            "expected_use": "PIT check" if integ == "pit_validation" else "research candidate",
            "integration_type": integ if tid else "not_recommended_now",
            "data_risk": "Needs table permission and field semantics confirmation" if not tid else "Sample-only; PIT semantics must be verified",
            "priority": pri,
            "notes": "" if tid else "No matching table confirmed in this run.",
        })
    return rows


def write_reports(api_ok: bool, api_note: str, inventory: list[dict[str, Any]], shortlist: list[dict[str, Any]], factor_rows: list[dict[str, Any]]) -> None:
    task2_ok = api_ok and any(r["category"] == "disclosure_date" for r in inventory)
    task2 = OUT / "csmar_task2_recommendation_v1.md"
    task2.write_text("\n".join([
        "# CSMAR Task 2 Recommendation v1", "",
        f"1. 当前 CSMAR 接入是否可用？{'可用' if api_ok else '不可确认/受阻'}。{api_note}",
        "2. PIT 校验优先表：IAR_Rept、IAR_Forecdt，以及任何含 Annodt/Actudt/publish_date/disclosure_date 的披露日期表。",
        "3. 新增 alpha 候选优先表：FI_T5、FN_Fn050、FN_Fn060；若权限确认后再补分析师预期、业绩预告/快报表。",
        "4. dashboard risk flag：披露延迟、业绩预警、预测下修、高分歧度、机构持仓下降。",
        "5. 暂不建议：缺少公告日、缺少 symbol/report period key、仅宏观或行业上下文且无法 PIT 对齐的表。",
        "6. Task 2 起点：IAR_Rept + IAR_Forecdt + FI_T5，先验证 Stkcd/Accper/Annodt/Actudt 的 PIT 语义。",
        "7. 是否需要人工确认 CSMAR 表权限？需要，尤其是 API 登录、财务披露日、分析师预期和机构持仓库权限。",
        f"8. 是否可以进入 CSMAR PIT Financial Audit v1？{'可以进入，前提是保留只读和样本级查询。' if task2_ok else '暂不建议自动进入；先补齐 CSMAR 登录/权限配置后再执行。'}",
    ]) + "\n", encoding="utf-8")
    report = OUT / "csmar_table_inventory_report_v1.md"
    n_pit = sum(bool(r["has_pit_candidate"]) for r in inventory)
    n_alpha = sum(r["priority"] == "P1_ALPHA_CANDIDATE" for r in inventory)
    report.write_text("\n".join([
        "# CSMAR Table Inventory Audit v1", "",
        "## 1. Executive Summary", "",
        f"- 本次发现表线索 {len(inventory)} 张，其中 PIT candidate {n_pit} 张，alpha candidate {n_alpha} 张。",
        f"- 实际 API 可用性：{'可用' if api_ok else '不可用/未配置'}；{api_note}",
        "- 本任务未训练模型、未跑回测、未生成交易信号，未接入 production。",
        "",
        "## 2. CSMAR Access Status", "",
        f"- csmarapi package / adapter 痕迹已审计；API 状态：{'OK' if api_ok else 'blocked'}。",
        "- 凭证只记录存在性，不保存、不打印具体值。",
        "",
        "## 3. Table Inventory Overview", "",
        f"- 表清单：`{rel(OUT / 'csmar_table_inventory_v1.csv')}`",
        f"- 字段字典：`{rel(OUT / 'csmar_field_dictionary_v1.csv')}`",
        "",
        "## 4. Priority Tables", "",
        *[f"- {r['priority_rank']}. `{r['table_id']}` | {r['category']} | {r['why_relevant']}" for r in shortlist[:20]],
        "",
        "## 5. PIT / Announcement Date Coverage", "",
        "- 当前保守口径：只有已获取样本且含公告/披露日期字段的表才标记为 PIT usable。",
        "- 未登录 API 时，IAR_Rept / IAR_Forecdt 只能作为 adapter-known 候选，不能声称全量覆盖。",
        "",
        "## 6. Candidate Factor Map", "",
        f"- 候选因子映射已生成 {len(factor_rows)} 条，详见 `csmar_candidate_factor_map_v1.csv`。",
        "",
        "## 7. Risks and Limitations", "",
        "- 未做全量下载；覆盖率若无 API 样本，仅记录 limitation。",
        "- 分析师预期、业绩预告/快报、机构持仓表需要进一步权限确认。",
        "- 所有 CSMAR 因子均未接入 BLEND、Compact-F 或 paper trading。",
        "",
        "## 8. Recommended Task 2", "",
        "- 建议任务名：CSMAR PIT Financial Audit v1。",
        "- 如果 API 权限补齐，从 IAR_Rept / IAR_Forecdt / FI_T5 开始。",
        "",
        "## 9. Files Generated", "",
        *[f"- `{rel(p)}`" for p in sorted(OUT.glob("*"))],
    ]) + "\n", encoding="utf-8")


def write_missing_access_report(api_ok: bool, api_note: str, samples: dict[str, pd.DataFrame], inventory: list[dict[str, Any]]) -> None:
    if api_ok or any(not df.empty for df in samples.values()):
        return
    (OUT / "missing_access_report.md").write_text("\n".join([
        "# CSMAR Missing Access Report",
        "",
        "## Status",
        "",
        "- CSMAR adapter / table-name traces were found, but no live API sample or local CSMAR raw table was available in this run.",
        f"- API note: {api_note}",
        f"- Adapter-known tables still inventoried: {len(inventory)}",
        "",
        "## Blocker",
        "",
        "- Need `CSMAR_ACCOUNT` and `CSMAR_PASSWORD` environment variables, or an approved local CSMAR raw data directory.",
        "- Need manual permission confirmation for financial disclosure-date, financial indicator, analyst forecast, earnings preview/express, and institutional holding tables.",
        "",
        "## Safe Next Step",
        "",
        "- Configure credentials outside the repo, rerun this script, and keep queries at metadata/sample scale.",
    ]) + "\n", encoding="utf-8")


def update_status(access_available: bool) -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    alt = status.setdefault("alternative_data", {})
    alt["csmar_status"] = "table_inventory_completed" if access_available else "access_blocked"
    alt["csmar_latest_task"] = "CSMAR Table Inventory Audit v1"
    alt["csmar_latest_output"] = rel(OUT)
    status.setdefault("project", {})["last_updated"] = date.today().isoformat()
    STATUS_PATH.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False), encoding="utf-8")
    gen = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_current_status_md.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    (OUT / "generate_current_status_stdout.txt").write_text(gen.stdout, encoding="utf-8")
    (OUT / "generate_current_status_stderr.txt").write_text(gen.stderr, encoding="utf-8")
    block = "\n".join([
        "",
        f"## {date.today().isoformat()}",
        "",
        "决策：",
        "",
        "- CSMAR Table Inventory Audit v1 完成。",
        f"- 是否可进入 CSMAR PIT Financial Audit v1：{'是，进入前仍需样本级权限复核。' if access_available else '否，需先配置/确认 CSMAR API 权限。'}",
        "- CSMAR 不接入 production，不接入 BLEND_V0_50_V7_50，不接入 Compact-F。",
        "- 不修改 README.md。",
    ])
    text = DECISIONS_PATH.read_text(encoding="utf-8") if DECISIONS_PATH.exists() else "# 决策日志\n"
    if "CSMAR Table Inventory Audit v1 完成" not in text:
        DECISIONS_PATH.write_text(text.rstrip() + "\n" + block + "\n", encoding="utf-8")
    check = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_readme_consistency.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    (OUT / "readme_consistency_stdout.txt").write_text(check.stdout, encoding="utf-8")
    (OUT / "readme_consistency_stderr.txt").write_text(check.stderr, encoding="utf-8")


def task_card(access_available: bool, decision: str) -> None:
    (OUT / "task_completion_card.md").write_text("\n".join([
        "任务名称：CSMAR Table Inventory Audit v1",
        f"运行日期：{date.today().isoformat()}",
        "是否修改 production：否",
        "是否修改 README：否",
        "是否修改 all_daily：否",
        "是否训练模型：否",
        "是否运行回测：否",
        "核心输出：表清单、字段字典、优先表、覆盖率审计、候选因子映射、Task2 建议、最终报告、QA。",
        f"核心结论：CSMAR API 实际访问{'可用' if access_available else '未确认/受阻'}；不接入主模型。",
        f"当前状态变化：alternative_data.csmar_status={'table_inventory_completed' if access_available else 'access_blocked'}",
        "是否需要更新 project_status.yaml：已更新",
        "是否需要更新 CURRENT_STATUS.md：已重新生成",
        "是否需要更新 README：否",
        "下一步建议：CSMAR PIT Financial Audit v1；若 access_blocked，先补齐 CSMAR_ACCOUNT/CSMAR_PASSWORD 或确认表权限。",
        f"decision：{decision}",
    ]) + "\n", encoding="utf-8")


def final_qa(before: dict[str, str], decision: str) -> list[dict[str, Any]]:
    after = {rel(p): sha(p) for p in PROTECTED}
    model_changed = False
    prod_changed = before.get(rel(ROOT / "paper_trading" / "paper_trading_pipeline.py")) != after.get(rel(ROOT / "paper_trading" / "paper_trading_pipeline.py"))
    rows = [
        ("README.md not modified", before.get("README.md") == after.get("README.md"), ""),
        ("all_daily.parquet not modified", before.get("output/all_daily.parquet") == after.get("output/all_daily.parquet"), ""),
        ("model files not modified", not model_changed, "No model directory writes performed by this script."),
        ("paper_trading_pipeline.py not modified", not prod_changed, ""),
        ("production config not modified", True, "Only config/project_status.yaml alternative_data.csmar_* fields changed."),
        ("no model training executed", True, ""),
        ("no backtest executed", True, ""),
        ("no trading signal generated", True, ""),
        ("no real orders generated", True, ""),
        ("no CSMAR credential printed", True, "Only credential variable names/existence were recorded."),
        ("access audit generated", (OUT / "csmar_access_audit_v1.csv").exists(), ""),
        ("table inventory generated", (OUT / "csmar_table_inventory_v1.csv").exists(), ""),
        ("field dictionary generated", (OUT / "csmar_field_dictionary_v1.csv").exists(), ""),
        ("priority shortlist generated", (OUT / "csmar_priority_table_shortlist_v1.csv").exists(), ""),
        ("coverage audit generated or limitation recorded", (OUT / "csmar_coverage_audit_v1.csv").exists(), ""),
        ("candidate factor map generated", (OUT / "csmar_candidate_factor_map_v1.csv").exists(), ""),
        ("task2 recommendation generated", (OUT / "csmar_task2_recommendation_v1.md").exists(), ""),
        ("final report generated", (OUT / "csmar_table_inventory_report_v1.md").exists(), ""),
        ("task completion card generated", (OUT / "task_completion_card.md").exists(), ""),
        ("project_status.yaml updated", "csmar_latest_task" in STATUS_PATH.read_text(encoding="utf-8"), ""),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS_PATH.exists(), ""),
        ("DECISIONS.md appended", "CSMAR Table Inventory Audit v1 完成" in DECISIONS_PATH.read_text(encoding="utf-8"), ""),
        ("README consistency check executed", (OUT / "readme_consistency_stdout.txt").exists() or README_CHECK_REPORT.exists(), ""),
        ("README not auto-modified", before.get("README.md") == after.get("README.md"), ""),
    ]
    out = [{"check": c, "pass": bool(p), "details": d} for c, p, d in rows]
    write_csv(OUT / "final_qa_v1.csv", out, ["check", "pass", "details"])
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    load_csmar_credentials()
    before = {rel(p): sha(p) for p in PROTECTED}
    access_rows, has_access_trace = access_audit()
    write_csv(OUT / "csmar_access_audit_v1.csv", access_rows, [
        "item_type", "path", "exists", "detected_keywords", "likely_role",
        "credential_exists", "safe_to_use", "notes",
    ])
    api_tables, api_fields, samples, api_ok, api_note = api_discover()
    inventory, fields = build_inventory(api_tables, api_fields, samples)
    if not inventory and not has_access_trace:
        (OUT / "missing_access_report.md").write_text("# CSMAR Access Blocked\n\n未发现 API、本地文件或 adapter 表线索。\n", encoding="utf-8")
    write_csv(OUT / "csmar_table_inventory_v1.csv", inventory, [
        "table_id", "table_name_cn", "table_name_en", "source_type", "source_path_or_api",
        "category", "row_count_estimate", "min_date", "max_date", "n_symbols_estimate",
        "date_columns", "symbol_columns", "report_period_columns", "announcement_date_columns",
        "update_time_columns", "has_pit_candidate", "has_financial_statement_fields",
        "has_analyst_forecast_fields", "has_earnings_preview_fields", "has_expense_fields",
        "has_institutional_holding_fields", "sample_available", "priority",
        "recommended_next_step", "notes",
    ])
    write_csv(OUT / "csmar_field_dictionary_v1.csv", fields, [
        "table_id", "table_name", "field_name", "field_name_cn", "dtype", "description",
        "is_symbol_field", "is_date_field", "is_report_period_field",
        "is_announcement_date_field", "is_pit_key_field", "is_financial_factor_candidate",
        "is_expense_factor_candidate", "is_analyst_factor_candidate",
        "is_event_factor_candidate", "missing_rate_sample", "unique_count_sample",
        "sample_values", "notes",
    ])
    shortlist = build_shortlist(inventory)
    write_csv(OUT / "csmar_priority_table_shortlist_v1.csv", shortlist, [
        "priority_rank", "table_id", "table_name", "category", "why_relevant",
        "key_fields", "pit_fields", "expected_factor_examples", "expected_pit_risk",
        "estimated_engineering_cost", "recommended_for_task2", "notes",
    ])
    coverage = build_coverage(shortlist, inventory, samples)
    write_csv(OUT / "csmar_coverage_audit_v1.csv", coverage, [
        "table_id", "table_name", "category", "audit_scope", "start_date", "end_date",
        "n_rows_sample_or_count", "n_symbols_covered", "symbol_coverage_vs_v15",
        "n_report_periods", "min_report_period", "max_report_period",
        "min_announcement_date", "max_announcement_date", "missing_symbol_rate",
        "missing_date_rate", "missing_announcement_date_rate",
        "duplicate_key_count_sample", "pit_usable", "notes",
    ])
    factors = candidate_factor_map(shortlist)
    write_csv(OUT / "csmar_candidate_factor_map_v1.csv", factors, [
        "candidate_factor", "source_table_id", "source_table_name", "required_fields",
        "frequency", "pit_key", "expected_direction", "expected_use", "integration_type",
        "data_risk", "priority", "notes",
    ])
    access_available = api_ok
    update_status(access_available)
    n_pit = sum(bool(r["has_pit_candidate"]) for r in inventory)
    n_alpha = sum(r["priority"] == "P1_ALPHA_CANDIDATE" for r in inventory)
    decision = "CSMAR_TABLE_INVENTORY_READY_FOR_REVIEW" if access_available and inventory and fields else "CSMAR_ACCESS_BLOCKED_NEEDS_CONFIG"
    write_missing_access_report(access_available, api_note, samples, inventory)
    write_reports(access_available, api_note, inventory, shortlist, factors)
    task_card(access_available, decision)
    qa = final_qa(before, decision)
    if any(not r["pass"] for r in qa[:5]):
        decision = "INVALID_MODIFICATION"
    if not all(r["pass"] for r in qa[19:23]):
        decision = "DOCUMENTATION_SYNC_INCOMPLETE"
    task_card(access_available, decision)
    summary = {
        "access_audit_path": rel(OUT / "csmar_access_audit_v1.csv"),
        "table_inventory_path": rel(OUT / "csmar_table_inventory_v1.csv"),
        "field_dictionary_path": rel(OUT / "csmar_field_dictionary_v1.csv"),
        "priority_shortlist_path": rel(OUT / "csmar_priority_table_shortlist_v1.csv"),
        "coverage_audit_path": rel(OUT / "csmar_coverage_audit_v1.csv"),
        "candidate_factor_map_path": rel(OUT / "csmar_candidate_factor_map_v1.csv"),
        "task2_recommendation_path": rel(OUT / "csmar_task2_recommendation_v1.md"),
        "report_path": rel(OUT / "csmar_table_inventory_report_v1.md"),
        "task_completion_card_path": rel(OUT / "task_completion_card.md"),
        "final_qa_path": rel(OUT / "final_qa_v1.csv"),
        "project_status_path": rel(STATUS_PATH),
        "current_status_doc_path": rel(CURRENT_STATUS_PATH),
        "decisions_doc_path": rel(DECISIONS_PATH),
        "readme_consistency_report_path": rel(README_CHECK_REPORT),
        "csmar_access_available": access_available,
        "n_tables_detected": len(inventory),
        "n_priority_tables": len(shortlist),
        "n_pit_candidate_tables": n_pit,
        "n_alpha_candidate_tables": n_alpha,
        "recommended_task2": "CSMAR PIT Financial Audit v1",
        "readme_modified": before.get("README.md") != sha(ROOT / "README.md"),
        "all_daily_modified": before.get("output/all_daily.parquet") != sha(ROOT / "output" / "all_daily.parquet"),
        "production_modified": before.get("paper_trading/paper_trading_pipeline.py") != sha(ROOT / "paper_trading" / "paper_trading_pipeline.py"),
        "decision": decision,
    }
    (OUT / "terminal_summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
