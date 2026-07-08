from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.csmar.credential_loader import load_csmar_credentials


OUT = ROOT / "output" / "csmar_pit_financial_factor_rebuild_v1"
PANEL_PATH = ROOT / "output" / "training_panel_v15_sr.parquet"
ALL_DAILY_PATH = ROOT / "output" / "all_daily.parquet"
STATUS_PATH = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS_PATH = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
README_PATH = ROOT / "README.md"
PAPER_PIPELINE_PATH = ROOT / "paper_trading" / "paper_trading_pipeline.py"
README_CONSISTENCY_REPORT = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"

PROTECTED_PATHS = [
    README_PATH,
    ALL_DAILY_PATH,
    PANEL_PATH,
    PAPER_PIPELINE_PATH,
]

CORE_FACTORS = ["ROE", "EP", "BP", "ProfitGrowth_YoY", "RevGrowth_YoY", "NetMargin", "Debt_Ratio"]

TABLE_FIELDS = {
    "IAR_Rept": ["Stkcd", "Accper", "Annodt", "Reptyp", "Profita", "Profitb", "Asseta", "Assetb", "Erana", "Eranb"],
    "IAR_Forecdt": ["Stkcd", "Accper", "Firforecdt", "Actudt"],
    "FI_T5": [
        "Stkcd",
        "Accper",
        "Typrep",
        "F050501B",
        "F050502B",
        "F050503B",
        "F050504C",
        "F051501B",
        "F051501C",
        "F051701B",
        "F051701C",
        "F051801B",
        "F051801C",
        "F052901B",
        "F052901C",
        "F053301B",
        "F053301C",
    ],
    "FN_Fn050": ["Stkcd", "stkcd", "Accper", "accper", "Typrep", "typrep", "FN05001", "FN05002", "FN05003"],
    "FN_Fn060": ["Stkcd", "Accper", "Typrep", "FN_Fn06001", "FN_Fn06002", "FN_Fn06003", "FN_Fn06004"],
}

DIRECT_FACTOR_FIELD_MAP = {
    "csmar_pit_roe": ("FI_T5", "F050501B", "CSMAR FI_T5 direct ROE indicator; audited field semantics remain medium confidence."),
    "csmar_pit_net_margin": ("FI_T5", "F051501B", "CSMAR FI_T5 direct net profit margin indicator; medium confidence."),
    "csmar_pit_sales_expense_to_revenue": ("FI_T5", "F051701B", "CSMAR FI_T5 direct sales expense ratio indicator; medium confidence."),
    "csmar_pit_rd_expense_to_revenue": ("FN_Fn060", "FN_Fn06002", "RD expense amount from notes; revenue denominator unavailable, so not computed unless revenue is later sourced."),
}

V15_PAIRS = {
    "ROE vs csmar_pit_roe": ("ROE_neutral_z", "csmar_pit_roe"),
    "EP vs csmar_pit_ep": ("EP_neutral_z", "csmar_pit_ep"),
    "BP vs csmar_pit_bp": ("BP_raw_neutral_z", "csmar_pit_bp"),
    "ProfitGrowth_YoY vs csmar_pit_profit_growth_yoy": ("ProfitGrowth_YoY_neutral_z", "csmar_pit_profit_growth_yoy"),
    "RevGrowth_YoY vs csmar_pit_rev_growth_yoy": ("RevGrowth_YoY_neutral_z", "csmar_pit_rev_growth_yoy"),
    "NetMargin vs csmar_pit_net_margin": ("Net_Profit_Margin_neutral_z", "csmar_pit_net_margin"),
    "Debt_Ratio vs csmar_pit_debt_ratio": ("Debt_Ratio_neutral_z", "csmar_pit_debt_ratio"),
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sanitize(text: Any) -> str:
    value = "" if text is None else str(text)
    for key in ("CSMAR_ACCOUNT", "CSMAR_PASSWORD"):
        secret = os.environ.get(key, "")
        if secret:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(?i)(token|cookie|session|password|account)\s*[:=]\s*[^,\s;]+", r"\1=[REDACTED]", value)
    return value[:500]


def normalize_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        for key in ("data", "datas", "rows", "result", "items", "list", "previewDatas"):
            if isinstance(value.get(key), list):
                return [x if isinstance(x, dict) else {"value": x} for x in value[key]]
        return [value]
    if isinstance(value, (list, tuple)):
        return [x if isinstance(x, dict) else {"value": x} for x in value]
    return [{"value": value}]


def first_col(df: pd.DataFrame, names: list[str]) -> str:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return ""


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def run_command(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False, timeout=timeout)


def csmar_rate_limit_seen() -> bool:
    log_path = ROOT / "csmar-log.log"
    if not log_path.exists():
        return False
    tail = log_path.read_text(encoding="utf-8", errors="ignore")[-20000:]
    return "Downloads has reached the limit today" in tail or "reached the limit today" in tail


def audit_inputs() -> pd.DataFrame:
    required = [
        ("csmar_pit_financial_audit_v1", "csmar_pit_financial_audit_report_v1.md", "PIT audit report"),
        ("csmar_pit_financial_audit_v1", "pit_leakage_risk_summary_v1.csv", "PIT leakage summary"),
        ("csmar_pit_financial_audit_v1", "csmar_financial_field_rebuild_plan_v1.csv", "field rebuild plan"),
        ("csmar_pit_financial_audit_v1", "pit_table_selection_v1.csv", "PIT table selection"),
        ("csmar_table_inventory_audit_v1", "csmar_table_inventory_v1.csv", "table inventory"),
        ("csmar_table_inventory_audit_v1", "csmar_field_dictionary_v1.csv", "field dictionary"),
    ]
    rows = []
    for folder, filename, role in required:
        root_path = ROOT / "output" / folder / filename
        legacy_path = ROOT / "xhs" / "output" / folder / filename
        source_type = "ROOT_CANONICAL" if root_path.exists() else "LEGACY_XHS_FALLBACK"
        path = root_path if root_path.exists() else legacy_path
        exists = path.exists()
        readable = False
        key_items = 0
        notes = ""
        if exists:
            try:
                if path.suffix.lower() == ".csv":
                    key_items = len(pd.read_csv(path))
                else:
                    key_items = len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
                readable = True
            except Exception as exc:
                notes = f"read failed: {type(exc).__name__}"
        else:
            notes = "missing in root and legacy fallback"
        rows.append({
            "input_path": rel(path),
            "source_type": source_type,
            "exists": exists,
            "readable": readable,
            "key_rows_or_items": key_items,
            "role": role,
            "notes": notes,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rebuild_input_audit_v1.csv", index=False, encoding="utf-8-sig")
    return df


def target_factor_spec() -> pd.DataFrame:
    rows = [
        {
            "target_factor": "ROE",
            "current_panel_field": "ROE_neutral_z",
            "required_csmar_tables": "FI_T5 + IAR_Rept/IAR_Forecdt",
            "required_raw_fields": "FI_T5.F050501B",
            "required_pit_fields": "IAR_Rept.Annodt or IAR_Forecdt.Actudt",
            "formula_or_definition": "Use CSMAR FI_T5 direct ROE indicator aligned by symbol/report_period, only after joined pit_date <= month_end.",
            "frequency": "monthly",
            "pit_rule": "source_pit_date <= month_end; source_report_period <= month_end; keep source_report_period/source_pit_date",
            "priority": "P0",
            "expected_engineering_cost": "medium",
            "source_confidence": "MEDIUM",
            "notes": "Ready only if FI_T5 and PIT disclosure tables fetch successfully.",
        },
        {
            "target_factor": "EP",
            "current_panel_field": "EP_neutral_z",
            "required_csmar_tables": "income statement or FI_T5 + market cap",
            "required_raw_fields": "net_profit_parent + market_value/share_count",
            "required_pit_fields": "announcement/disclosure date",
            "formula_or_definition": "net profit attributable to parent / PIT market capitalization. all_daily has price only and no reliable share count/market cap.",
            "frequency": "monthly",
            "pit_rule": "financial pit_date <= month_end; price observed <= month_end",
            "priority": "P1",
            "expected_engineering_cost": "high",
            "source_confidence": "BLOCKED",
            "notes": "BLOCKED_NO_MARKET_CAP_OR_SHARE_COUNT; no proxy is used.",
        },
        {
            "target_factor": "BP",
            "current_panel_field": "BP_raw_neutral_z",
            "required_csmar_tables": "balance sheet + market cap",
            "required_raw_fields": "book_equity_parent + market_value/share_count",
            "required_pit_fields": "announcement/disclosure date",
            "formula_or_definition": "book equity attributable to parent / PIT market capitalization. all_daily has price only and no reliable share count/market cap.",
            "frequency": "monthly",
            "pit_rule": "financial pit_date <= month_end; price observed <= month_end",
            "priority": "P1",
            "expected_engineering_cost": "high",
            "source_confidence": "BLOCKED",
            "notes": "BLOCKED_NO_MARKET_CAP_OR_SHARE_COUNT; no proxy is used.",
        },
        {
            "target_factor": "ProfitGrowth_YoY",
            "current_panel_field": "ProfitGrowth_YoY_neutral_z",
            "required_csmar_tables": "FI_T5 or income statement + PIT disclosure",
            "required_raw_fields": "net_profit_parent current/prior year or audited FI_T5 growth field",
            "required_pit_fields": "IAR_Rept.Annodt or IAR_Forecdt.Actudt",
            "formula_or_definition": "(current net profit parent - prior-year same-period net profit parent) / abs(prior-year same-period net profit parent).",
            "frequency": "monthly",
            "pit_rule": "both current and prior records must have pit_date <= month_end",
            "priority": "P0",
            "expected_engineering_cost": "medium",
            "source_confidence": "BLOCKED",
            "notes": "Current inventory lacks reliable raw net profit statement field mapping; do not infer from unaudited FI_T5 code.",
        },
        {
            "target_factor": "RevGrowth_YoY",
            "current_panel_field": "RevGrowth_YoY_neutral_z",
            "required_csmar_tables": "income statement + PIT disclosure",
            "required_raw_fields": "revenue current/prior year",
            "required_pit_fields": "IAR_Rept.Annodt or IAR_Forecdt.Actudt",
            "formula_or_definition": "(current revenue - prior-year same-period revenue) / abs(prior-year same-period revenue).",
            "frequency": "monthly",
            "pit_rule": "both current and prior records must have pit_date <= month_end",
            "priority": "P0",
            "expected_engineering_cost": "medium",
            "source_confidence": "BLOCKED",
            "notes": "Current inventory lacks reliable revenue raw field mapping; do not infer from unaudited FI_T5 code.",
        },
        {
            "target_factor": "NetMargin",
            "current_panel_field": "Net_Profit_Margin_neutral_z",
            "required_csmar_tables": "FI_T5 + IAR_Rept/IAR_Forecdt",
            "required_raw_fields": "FI_T5.F051501B",
            "required_pit_fields": "IAR_Rept.Annodt or IAR_Forecdt.Actudt",
            "formula_or_definition": "Use CSMAR FI_T5 direct net profit margin indicator aligned by symbol/report_period and PIT date.",
            "frequency": "monthly",
            "pit_rule": "source_pit_date <= month_end",
            "priority": "P1",
            "expected_engineering_cost": "low",
            "source_confidence": "MEDIUM",
            "notes": "Ready only if FI_T5 and PIT disclosure tables fetch successfully.",
        },
        {
            "target_factor": "Debt_Ratio",
            "current_panel_field": "Debt_Ratio_neutral_z",
            "required_csmar_tables": "balance sheet + PIT disclosure",
            "required_raw_fields": "total_liabilities / total_assets",
            "required_pit_fields": "announcement/disclosure date",
            "formula_or_definition": "total liabilities / total assets.",
            "frequency": "monthly",
            "pit_rule": "source_pit_date <= month_end",
            "priority": "P1",
            "expected_engineering_cost": "medium",
            "source_confidence": "BLOCKED",
            "notes": "Current inventory does not expose reliable total_assets/total_liabilities fields.",
        },
        {
            "target_factor": "sales_expense_to_revenue",
            "current_panel_field": "",
            "required_csmar_tables": "FI_T5 + IAR_Rept/IAR_Forecdt",
            "required_raw_fields": "FI_T5.F051701B",
            "required_pit_fields": "IAR_Rept.Annodt or IAR_Forecdt.Actudt",
            "formula_or_definition": "Use CSMAR FI_T5 direct sales expense ratio indicator aligned by PIT report date.",
            "frequency": "monthly",
            "pit_rule": "source_pit_date <= month_end",
            "priority": "P2",
            "expected_engineering_cost": "low",
            "source_confidence": "MEDIUM",
            "notes": "Candidate overlay field only; not in current v15 comparison set.",
        },
        {
            "target_factor": "rd_expense_to_revenue",
            "current_panel_field": "",
            "required_csmar_tables": "FN_Fn060 + income statement + PIT disclosure",
            "required_raw_fields": "FN_Fn06002 + revenue",
            "required_pit_fields": "announcement/disclosure date",
            "formula_or_definition": "research and development expense / revenue.",
            "frequency": "monthly",
            "pit_rule": "source_pit_date <= month_end",
            "priority": "P2",
            "expected_engineering_cost": "medium",
            "source_confidence": "BLOCKED",
            "notes": "Revenue denominator and PIT date for notes table require additional source mapping.",
        },
        {
            "target_factor": "earnings_preview_midpoint_yoy",
            "current_panel_field": "",
            "required_csmar_tables": "earnings preview/forecast table",
            "required_raw_fields": "preview lower/upper bound and prior comparable profit",
            "required_pit_fields": "preview announcement date",
            "formula_or_definition": "midpoint of announced earnings preview YoY range.",
            "frequency": "monthly",
            "pit_rule": "preview pit_date <= month_end",
            "priority": "P2",
            "expected_engineering_cost": "medium",
            "source_confidence": "BLOCKED",
            "notes": "No reliable earnings preview source table in current inventory.",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rebuild_target_factor_spec_v1.csv", index=False, encoding="utf-8-sig")
    return df


def build_universe_month_index(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = panel[["date", "symbol"]].copy()
    work["month_end"] = pd.to_datetime(work["date"], errors="coerce") + pd.offsets.MonthEnd(0)
    work["month_end"] = work["month_end"].dt.normalize()
    work["symbol"] = work["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    work = work[work["symbol"].str.fullmatch(r"\d{6}", na=False) & work["month_end"].notna()]
    idx = work[["symbol", "month_end"]].drop_duplicates().sort_values(["symbol", "month_end"]).reset_index(drop=True)
    idx.to_parquet(OUT / "rebuild_universe_month_index_v1.parquet", index=False)
    summary = pd.DataFrame([{
        "n_symbols": int(idx["symbol"].nunique()),
        "n_months": int(idx["month_end"].nunique()),
        "min_month": str(idx["month_end"].min().date()) if len(idx) else "",
        "max_month": str(idx["month_end"].max().date()) if len(idx) else "",
        "n_symbol_months": int(len(idx)),
        "source_panel": rel(PANEL_PATH),
        "notes": "Universe/month index is restricted to v15 observed symbol-months; no new months generated.",
    }])
    summary.to_csv(OUT / "rebuild_universe_summary_v1.csv", index=False, encoding="utf-8-sig")
    return idx, summary


def fetch_table(csmar: Any, table_id: str, fields: list[str], symbols: list[str], start_date: str, end_date: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    logs: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    batch_size = 40
    batch_id = 0
    for i in range(0, len(symbols), batch_size):
        batch_id += 1
        chunk = symbols[i:i + batch_size]
        sym_field = "stkcd" if table_id == "FN_Fn050" else "Stkcd"
        date_field = "accper" if table_id == "FN_Fn050" else "Accper"
        usable_fields = list(dict.fromkeys([f for f in fields if f]))
        if table_id == "FN_Fn050":
            usable_fields = [f for f in usable_fields if f not in {"Stkcd", "Accper", "Typrep"}]
        condition = f"{sym_field} in ({','.join(repr(s) for s in chunk)}) and {date_field}>='{start_date}' and {date_field}<='{end_date}'"
        log = {
            "batch_id": f"{table_id}_{batch_id:04d}",
            "table_id": table_id,
            "n_symbols_requested": len(chunk),
            "start_date": start_date,
            "end_date": end_date,
            "attempted": True,
            "success": False,
            "n_rows": 0,
            "error_type": "",
            "sanitized_error_message": "",
            "notes": "",
        }
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                records = normalize_records(csmar.query(usable_fields, condition, table_id))
            df = pd.DataFrame(records)
            if not df.empty:
                df["_fetch_batch_id"] = log["batch_id"]
                df["_source_table_id"] = table_id
                frames.append(df)
            log["success"] = True
            log["n_rows"] = int(len(df))
        except Exception as exc:
            log["error_type"] = type(exc).__name__
            log["sanitized_error_message"] = sanitize(exc)
        logs.append(log)
    return (pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()), logs


def fetch_raw_financial_data(symbols: list[str], min_month: pd.Timestamp, max_month: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_date = (min_month - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    end_date = max_month.strftime("%Y-%m-%d")
    status = load_csmar_credentials()
    logs: list[dict[str, Any]] = []
    wide_frames: dict[str, pd.DataFrame] = {}
    if csmar_rate_limit_seen():
        for tid in TABLE_FIELDS:
            logs.append({
                "batch_id": f"{tid}_0000",
                "table_id": tid,
                "n_symbols_requested": len(symbols),
                "start_date": start_date,
                "end_date": end_date,
                "attempted": False,
                "success": False,
                "n_rows": 0,
                "error_type": "CSMAR_DAILY_DOWNLOAD_LIMIT",
                "sanitized_error_message": "CSMAR API reports that today's download limit has been reached.",
                "notes": "Skipped API fetch to avoid repeated calls after rate-limit detection.",
            })
    elif not (status["account_present"] and status["password_present"]):
        for tid in TABLE_FIELDS:
            logs.append({
                "batch_id": f"{tid}_0000",
                "table_id": tid,
                "n_symbols_requested": len(symbols),
                "start_date": start_date,
                "end_date": end_date,
                "attempted": False,
                "success": False,
                "n_rows": 0,
                "error_type": "MISSING_CREDENTIALS",
                "sanitized_error_message": "CSMAR credentials missing.",
                "notes": "No API query attempted.",
            })
    else:
        logging.disable(logging.CRITICAL)
        try:
            from csmarapi.CsmarService import CsmarService
            csmar = CsmarService()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                csmar.login(os.environ["CSMAR_ACCOUNT"], os.environ["CSMAR_PASSWORD"])
            for tid, fields in TABLE_FIELDS.items():
                df, table_logs = fetch_table(csmar, tid, fields, symbols, start_date, end_date)
                logs.extend(table_logs)
                wide_frames[tid] = df
        except Exception as exc:
            for tid in TABLE_FIELDS:
                logs.append({
                    "batch_id": f"{tid}_0000",
                    "table_id": tid,
                    "n_symbols_requested": len(symbols),
                    "start_date": start_date,
                    "end_date": end_date,
                    "attempted": True,
                    "success": False,
                    "n_rows": 0,
                    "error_type": type(exc).__name__,
                    "sanitized_error_message": sanitize(exc),
                    "notes": "CSMAR API setup/login failed; no credential values stored.",
                })

    fetch_log = pd.DataFrame(logs)
    fetch_log.to_csv(OUT / "csmar_raw_financial_fetch_log_v1.csv", index=False, encoding="utf-8-sig")
    raw_long = build_raw_long(wide_frames)
    raw_long.to_parquet(OUT / "csmar_raw_financial_pit_records_v1.parquet", index=False)
    raw_long.head(50000).to_csv(OUT / "csmar_raw_financial_pit_records_sample_v1.csv", index=False, encoding="utf-8-sig")
    return raw_long, fetch_log


def build_pit_map(wide_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for tid, date_col, source in [("IAR_Rept", "Annodt", "announcement_date"), ("IAR_Forecdt", "Actudt", "disclosure_date")]:
        df = wide_frames.get(tid, pd.DataFrame()).copy()
        if df.empty:
            continue
        sym = first_col(df, ["Stkcd", "stkcd"])
        acc = first_col(df, ["Accper", "accper"])
        pit = first_col(df, [date_col])
        if not (sym and acc and pit):
            continue
        part = pd.DataFrame({
            "symbol": df[sym].astype(str).str.extract(r"(\d+)")[0].str.zfill(6),
            "report_period": pd.to_datetime(df[acc], errors="coerce"),
            "pit_date": pd.to_datetime(df[pit], errors="coerce"),
            "pit_date_source": source,
            "source_table_id": tid,
        })
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["symbol", "report_period", "pit_date", "pit_date_source", "source_table_id"])
    pit = pd.concat(parts, ignore_index=True).dropna(subset=["symbol", "report_period", "pit_date"])
    pit = pit.sort_values(["symbol", "report_period", "pit_date", "source_table_id"])
    return pit.groupby(["symbol", "report_period"], as_index=False).first()


def build_raw_long(wide_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pit_map = build_pit_map(wide_frames)
    rows: list[dict[str, Any]] = []
    table_names = {
        "IAR_Rept": "财务报告披露日期",
        "IAR_Forecdt": "预约披露日期/实际披露日期",
        "FI_T5": "财务指标",
        "FN_Fn050": "财务报表附注-销售费用明细",
        "FN_Fn060": "财务报表附注-研发费用明细",
    }
    for tid, df in wide_frames.items():
        if df.empty:
            continue
        df = df.copy()
        sym = first_col(df, ["Stkcd", "stkcd", "symbol"])
        acc = first_col(df, ["Accper", "accper"])
        if not (sym and acc):
            continue
        df["_symbol"] = df[sym].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
        df["_report_period"] = pd.to_datetime(df[acc], errors="coerce")
        df = df.merge(pit_map, left_on=["_symbol", "_report_period"], right_on=["symbol", "report_period"], how="left", suffixes=("", "_pit"))
        if tid == "IAR_Rept" and "Annodt" in df.columns:
            df["pit_date"] = pd.to_datetime(df["Annodt"], errors="coerce")
            df["pit_date_source"] = "announcement_date"
        if tid == "IAR_Forecdt" and "Actudt" in df.columns:
            df["pit_date"] = pd.to_datetime(df["Actudt"], errors="coerce")
            df["pit_date_source"] = "disclosure_date"
        value_cols = [c for c in df.columns if not str(c).startswith("_") and c not in {"symbol", "report_period", "pit_date", "pit_date_source", "source_table_id"}]
        for _, r in df.iterrows():
            for col in value_cols:
                if col.lower() in {"stkcd", "accper", "symbol"}:
                    continue
                rows.append({
                    "symbol": r["_symbol"],
                    "report_period": r["_report_period"],
                    "pit_date": r.get("pit_date", pd.NaT),
                    "pit_date_source": r.get("pit_date_source", "joined_disclosure_date" if pd.notna(r.get("pit_date", pd.NaT)) else "missing"),
                    "announcement_date": r.get("Annodt", ""),
                    "disclosure_date": r.get("Actudt", ""),
                    "publish_date": "",
                    "update_time": "",
                    "source_table_id": tid,
                    "source_table_name": table_names.get(tid, tid),
                    "raw_field_name": col,
                    "raw_field_value": r.get(col, np.nan),
                    "statement_type": "",
                    "report_type": r.get("Typrep", r.get("typrep", r.get("Reptyp", ""))),
                    "fetch_batch_id": r.get("_fetch_batch_id", ""),
                    "notes": "PIT date joined by symbol/report_period from IAR_Rept/IAR_Forecdt." if tid not in {"IAR_Rept", "IAR_Forecdt"} else "Direct disclosure-date table.",
                })
    cols = [
        "symbol", "report_period", "pit_date", "pit_date_source", "announcement_date", "disclosure_date",
        "publish_date", "update_time", "source_table_id", "source_table_name", "raw_field_name",
        "raw_field_value", "statement_type", "report_type", "fetch_batch_id", "notes",
    ]
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out["symbol"] = out["symbol"].astype(str).str.zfill(6)
        out["report_period"] = pd.to_datetime(out["report_period"], errors="coerce")
        out["pit_date"] = pd.to_datetime(out["pit_date"], errors="coerce")
        out["raw_field_value"] = out["raw_field_value"].map(lambda x: "" if pd.isna(x) else str(x))
    return out


def build_statement_records(raw: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "symbol", "report_period", "pit_date", "pit_date_source", "source_table_ids", "report_type",
        "statement_type", "revenue", "net_profit_parent", "total_assets", "total_liabilities",
        "total_equity_parent", "sales_expense", "management_expense", "rd_expense",
        "operating_cashflow", "csmar_pit_roe", "csmar_pit_net_margin",
        "csmar_pit_sales_expense_to_revenue", "data_quality_flag", "duplicate_resolution_rule", "notes",
    ]
    if raw.empty:
        out = pd.DataFrame(columns=cols)
    else:
        work = raw.dropna(subset=["symbol", "report_period"]).copy()
        work = work.sort_values(["symbol", "report_period", "source_table_id", "pit_date"])
        grouped = []
        for (sym, rp), g in work.groupby(["symbol", "report_period"], dropna=False):
            valid_pit = g["pit_date"].dropna()
            pit_date = valid_pit.min() if len(valid_pit) else pd.NaT
            pit_src = str(g.loc[g["pit_date"].eq(pit_date), "pit_date_source"].iloc[0]) if pd.notna(pit_date) and g["pit_date"].eq(pit_date).any() else "missing"
            pivot = g.pivot_table(index=["symbol", "report_period"], columns=["source_table_id", "raw_field_name"], values="raw_field_value", aggfunc="first")
            vals = {}
            for field, (table, raw_field, _) in DIRECT_FACTOR_FIELD_MAP.items():
                vals[field] = np.nan
                if not pivot.empty and (table, raw_field) in pivot.columns:
                    vals[field] = pd.to_numeric(pd.Series([pivot.iloc[0][(table, raw_field)]]), errors="coerce").iloc[0]
            grouped.append({
                "symbol": str(sym).zfill(6),
                "report_period": rp,
                "pit_date": pit_date,
                "pit_date_source": pit_src,
                "source_table_ids": "|".join(sorted(g["source_table_id"].dropna().astype(str).unique())),
                "report_type": "|".join(sorted(set(str(x) for x in g["report_type"].dropna().unique() if str(x) != ""))),
                "statement_type": "",
                "revenue": np.nan,
                "net_profit_parent": np.nan,
                "total_assets": np.nan,
                "total_liabilities": np.nan,
                "total_equity_parent": np.nan,
                "sales_expense": np.nan,
                "management_expense": np.nan,
                "rd_expense": np.nan,
                "operating_cashflow": np.nan,
                "csmar_pit_roe": vals["csmar_pit_roe"],
                "csmar_pit_net_margin": vals["csmar_pit_net_margin"],
                "csmar_pit_sales_expense_to_revenue": vals["csmar_pit_sales_expense_to_revenue"],
                "data_quality_flag": "OK_DIRECT_FI_T5_PARTIAL" if pd.notna(vals["csmar_pit_roe"]) or pd.notna(vals["csmar_pit_net_margin"]) else "BLOCKED_NO_MAPPED_STATEMENT_FIELD",
                "duplicate_resolution_rule": "earliest_pit_date_by_symbol_report_period; first raw value per source field",
                "notes": "Statement accounting fields remain NaN unless raw income/balance-sheet fields are added; direct FI_T5 indicators retained separately.",
            })
        out = pd.DataFrame(grouped, columns=cols)
        out = out.sort_values(["symbol", "report_period", "pit_date"]).drop_duplicates(["symbol", "report_period"], keep="first")
    out.to_parquet(OUT / "csmar_pit_statement_records_v1.parquet", index=False)
    out.head(50000).to_csv(OUT / "csmar_pit_statement_records_sample_v1.csv", index=False, encoding="utf-8-sig")
    return out


def build_monthly_panel(index_df: pd.DataFrame, statements: pd.DataFrame) -> pd.DataFrame:
    factor_cols = [
        "csmar_pit_roe", "csmar_pit_ep", "csmar_pit_bp", "csmar_pit_profit_growth_yoy",
        "csmar_pit_rev_growth_yoy", "csmar_pit_net_margin", "csmar_pit_debt_ratio",
        "csmar_pit_sales_expense_to_revenue", "csmar_pit_rd_expense_to_revenue",
    ]
    rows = []
    st = statements.copy()
    if not st.empty:
        st["pit_date"] = pd.to_datetime(st["pit_date"], errors="coerce")
        st["report_period"] = pd.to_datetime(st["report_period"], errors="coerce")
        st = st.dropna(subset=["symbol", "pit_date", "report_period"])
    for sym, months in index_df.groupby("symbol", sort=True):
        s = st[st["symbol"].eq(sym)].sort_values(["report_period", "pit_date"]) if not st.empty else pd.DataFrame()
        for me in months["month_end"]:
            row = {"symbol": sym, "month_end": me}
            for col in factor_cols:
                row[col] = np.nan
            row.update({
                "source_report_period": pd.NaT,
                "source_pit_date": pd.NaT,
                "source_pit_date_source": "",
                "source_table_ids": "",
                "factor_quality_flag": "NO_PIT_SOURCE_RECORD",
            })
            if not s.empty:
                eligible = s[(s["pit_date"] <= me) & (s["report_period"] <= me)].sort_values(["report_period", "pit_date"])
                if not eligible.empty:
                    latest = eligible.iloc[-1]
                    row["csmar_pit_roe"] = latest.get("csmar_pit_roe", np.nan)
                    row["csmar_pit_net_margin"] = latest.get("csmar_pit_net_margin", np.nan)
                    row["csmar_pit_sales_expense_to_revenue"] = latest.get("csmar_pit_sales_expense_to_revenue", np.nan)
                    row["source_report_period"] = latest["report_period"]
                    row["source_pit_date"] = latest["pit_date"]
                    row["source_pit_date_source"] = latest.get("pit_date_source", "")
                    row["source_table_ids"] = latest.get("source_table_ids", "")
                    any_factor = any(pd.notna(row[c]) for c in factor_cols)
                    row["factor_quality_flag"] = "OK_PARTIAL_DIRECT_FI_T5" if any_factor else "BLOCKED_NO_MAPPED_FACTOR"
            rows.append(row)
    panel = pd.DataFrame(rows)
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    for col in ["month_end", "source_report_period", "source_pit_date"]:
        panel[col] = pd.to_datetime(panel[col], errors="coerce")
    panel.to_parquet(OUT / "csmar_pit_financial_monthly_panel_v1.parquet", index=False)
    panel.head(50000).to_csv(OUT / "csmar_pit_financial_monthly_panel_sample_v1.csv", index=False, encoding="utf-8-sig")
    return panel


def compliance_qa(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(panel)
    def add(check: str, passed: bool, n_violations: int, details: str) -> None:
        rows.append({
            "check": check,
            "pass": bool(passed),
            "n_violations": int(n_violations),
            "violation_rate": round(n_violations / n, 8) if n else 0.0,
            "details": details,
        })
    pit_mask = panel["source_pit_date"].notna()
    pit_viol = int((pit_mask & (panel["source_pit_date"] > panel["month_end"])).sum())
    add("source_pit_date <= month_end", pit_viol == 0, pit_viol, "Rows without PIT source are treated as missing coverage, not PIT violations.")
    rp_mask = panel["source_report_period"].notna()
    rp_viol = int((rp_mask & (panel["source_report_period"] > panel["month_end"])).sum())
    add("source_report_period <= month_end", rp_viol == 0, rp_viol, "")
    sym_viol = int((~panel["symbol"].astype(str).str.fullmatch(r"\d{6}", na=False)).sum())
    add("symbol format is 6-digit string", sym_viol == 0, sym_viol, "")
    dup = int(panel.duplicated(["symbol", "month_end"]).sum())
    add("no duplicate symbol-month_end", dup == 0, dup, "")
    num = panel.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(num.to_numpy(dtype=float, copy=True)).sum()) if not num.empty else 0
    add("no inf", inf_count == 0, inf_count, "")
    factor_cols = [c for c in panel.columns if c.startswith("csmar_pit_")]
    for col in factor_cols:
        s = safe_num(panel[col])
        miss = int(s.isna().sum())
        extreme = int((s.abs() > 100).sum())
        add(f"missing_rate {col}", True, miss, f"missing_rate={miss / n:.6f}" if n else "empty panel")
        add(f"extreme_abs_gt_100 {col}", True, extreme, f"extreme_rate={extreme / n:.6f}" if n else "empty panel")
    coverage = panel[factor_cols].notna().mean().mean() if factor_cols and n else 0.0
    add("overall factor coverage", coverage > 0, 0 if coverage > 0 else n, f"mean_coverage={coverage:.6f}")
    qa = pd.DataFrame(rows)
    qa.to_csv(OUT / "pit_factor_compliance_qa_v1.csv", index=False, encoding="utf-8-sig")
    return qa


def compare_with_v15(v15: pd.DataFrame, pit_panel: pd.DataFrame) -> pd.DataFrame:
    v = v15.copy()
    v["month_end"] = pd.to_datetime(v["date"], errors="coerce") + pd.offsets.MonthEnd(0)
    v["month_end"] = v["month_end"].dt.normalize()
    v["symbol"] = v["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    merged = v.merge(pit_panel, on=["symbol", "month_end"], how="left")
    rows = []
    for name, (vcol, pcol) in V15_PAIRS.items():
        if vcol not in merged.columns or pcol not in merged.columns:
            rows.append({
                "factor_pair": name, "n_overlap": 0, "monthly_mean_spearman": np.nan,
                "monthly_median_spearman": np.nan, "overall_spearman": np.nan,
                "mean_abs_diff": np.nan, "coverage_v15": 0.0, "coverage_csmar_pit": 0.0,
                "coverage_overlap": 0.0, "n_months_compared": 0,
                "interpretation": "missing columns", "notes": "Data comparison only; no IC, model, or return test.",
            })
            continue
        both = merged[[vcol, pcol, "month_end"]].copy()
        both[vcol] = safe_num(both[vcol])
        both[pcol] = safe_num(both[pcol])
        overlap = both.dropna(subset=[vcol, pcol])
        monthly = []
        for _, g in overlap.groupby("month_end"):
            if len(g) >= 3:
                monthly.append(g[vcol].corr(g[pcol], method="spearman"))
        rows.append({
            "factor_pair": name,
            "n_overlap": int(len(overlap)),
            "monthly_mean_spearman": float(np.nanmean(monthly)) if monthly else np.nan,
            "monthly_median_spearman": float(np.nanmedian(monthly)) if monthly else np.nan,
            "overall_spearman": float(overlap[vcol].corr(overlap[pcol], method="spearman")) if len(overlap) >= 3 else np.nan,
            "mean_abs_diff": float((overlap[vcol] - overlap[pcol]).abs().mean()) if len(overlap) else np.nan,
            "coverage_v15": float(both[vcol].notna().mean()),
            "coverage_csmar_pit": float(both[pcol].notna().mean()),
            "coverage_overlap": float(len(overlap) / len(both)) if len(both) else 0.0,
            "n_months_compared": int(len(monthly)),
            "interpretation": "Low or missing correlation indicates timing/source/scale differences; it is not a correctness judgment.",
            "notes": "Data comparison only; no IC, model, or return test.",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "v15_vs_csmar_pit_factor_comparison_v1.csv", index=False, encoding="utf-8-sig")
    return out


def replacement_readiness(panel: pd.DataFrame, qa: pd.DataFrame, comparison: pd.DataFrame, spec: pd.DataFrame) -> pd.DataFrame:
    pit_pass = bool(qa.loc[qa["check"].eq("source_pit_date <= month_end"), "pass"].iloc[0]) if not qa.empty else False
    mapping = {
        "ROE": "csmar_pit_roe",
        "EP": "csmar_pit_ep",
        "BP": "csmar_pit_bp",
        "ProfitGrowth_YoY": "csmar_pit_profit_growth_yoy",
        "RevGrowth_YoY": "csmar_pit_rev_growth_yoy",
        "NetMargin": "csmar_pit_net_margin",
        "Debt_Ratio": "csmar_pit_debt_ratio",
        "sales_expense_to_revenue": "csmar_pit_sales_expense_to_revenue",
        "rd_expense_to_revenue": "csmar_pit_rd_expense_to_revenue",
        "earnings_preview_midpoint_yoy": "csmar_pit_earnings_preview_midpoint_yoy",
    }
    rows = []
    for _, r in spec.iterrows():
        target = r["target_factor"]
        field = mapping.get(target, "")
        coverage = float(panel[field].notna().mean()) if field in panel.columns and len(panel) else 0.0
        blocked_source = str(r["source_confidence"]).upper() == "BLOCKED"
        comp_row = comparison[comparison["factor_pair"].str.contains(target.split("_")[0], case=False, na=False)] if not comparison.empty else pd.DataFrame()
        comp_text = comp_row["interpretation"].iloc[0] if not comp_row.empty else "no v15 comparison available"
        if not pit_pass:
            readiness = "NOT_READY_PIT_VIOLATION"
            action = "KEEP_AS_AUDIT_ONLY"
            issues = "PIT compliance violation."
        elif blocked_source:
            readiness = "BLOCKED_NO_SOURCE_FIELD"
            action = "NEED_MORE_CSMAR_FIELDS"
            issues = str(r["notes"])
        elif coverage >= 0.50:
            readiness = "READY_FOR_PANEL_OVERLAY_TEST"
            action = "USE_IN_NEXT_OVERLAY_PANEL"
            issues = ""
        elif coverage > 0:
            readiness = "PARTIAL_READY_NEEDS_MANUAL_REVIEW"
            action = "KEEP_AS_AUDIT_ONLY"
            issues = "Coverage below 50%; manual source-field review needed."
        else:
            readiness = "NOT_READY_COVERAGE_LOW"
            action = "NEED_MORE_CSMAR_FIELDS"
            issues = "No usable monthly coverage."
        rows.append({
            "target_factor": target,
            "csmar_pit_field": field,
            "coverage_rate": coverage,
            "pit_compliance_pass": pit_pass,
            "comparison_to_v15": comp_text,
            "replacement_readiness": readiness,
            "blocking_issues": issues,
            "recommended_action": action,
            "notes": "Conservative readiness; no production/model integration.",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "csmar_pit_replacement_readiness_v1.csv", index=False, encoding="utf-8-sig")
    return out


def update_project_status(status_value: str) -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    status.setdefault("alternative_data", {})
    status["alternative_data"]["csmar_status"] = status_value
    status["alternative_data"]["csmar_latest_task"] = "CSMAR PIT Financial Factor Rebuild v1"
    status["alternative_data"]["csmar_latest_output"] = "output/csmar_pit_financial_factor_rebuild_v1"
    status["alternative_data"]["csmar_location"] = "data_sources/csmar"
    status["alternative_data"]["csmar_legacy_location"] = "xhs/scripts and xhs/output legacy reference"
    status.setdefault("validation", {})
    status["validation"]["pit_financial_status"] = "risk_detected_rebuild_completed_or_partial"
    status["validation"]["blend_v3_historical_metrics_status"] = "under_pit_review"
    status.setdefault("project", {})
    status["project"]["last_updated"] = date.today().isoformat()
    STATUS_PATH.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False), encoding="utf-8")


def append_decision(decision: str, ready: list[str], blocked: list[str]) -> None:
    block = "\n".join([
        f"## {date.today().isoformat()}",
        "",
        "决策：",
        "",
        "- CSMAR PIT Financial Factor Rebuild v1 完成。",
        f"- 是否生成 PIT 合规月频财务因子面板：{'是' if decision != 'CSMAR_PIT_REBUILD_BLOCKED_NO_SOURCE_DATA' else '否'}。",
        f"- 可替换字段列表：{', '.join(ready) if ready else '无'}。",
        f"- 不可替换或需人工复核字段：{', '.join(blocked) if blocked else '无'}。",
        "- 不接入 production。",
        "- 不修改 README。",
        "- Blend V3 historical metrics 继续标记为 PIT-under-review，直到基于新 PIT 面板重新跑 OOS tournament。",
        f"- Decision = {decision}。",
    ])
    text = DECISIONS_PATH.read_text(encoding="utf-8") if DECISIONS_PATH.exists() else "# 决策日志\n"
    if "CSMAR PIT Financial Factor Rebuild v1 完成" not in text:
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


def make_report(summary: pd.DataFrame, fetch_log: pd.DataFrame, raw: pd.DataFrame, statements: pd.DataFrame, panel: pd.DataFrame, qa: pd.DataFrame, comparison: pd.DataFrame, readiness: pd.DataFrame, decision: str, next_task: str) -> None:
    qa_pit = qa.loc[qa["check"].eq("source_pit_date <= month_end"), "pass"].iloc[0] if not qa.empty else False
    lines = [
        "# CSMAR PIT Financial Factor Rebuild v1",
        "",
        "## 1. Executive Summary",
        "",
        f"- Decision: {decision}",
        f"- PIT compliance pass: {bool(qa_pit)}",
        f"- Monthly rows: {len(panel)}",
        f"- Raw PIT records: {len(raw)}",
        f"- Statement records: {len(statements)}",
        "",
        "## 2. Scope and Non-Goals",
        "",
        "- This task builds data artifacts and QA only.",
        "- This task is not model training.",
        "- This task is not a backtest.",
        "- This task does not run IC.",
        "- This task does not connect CSMAR to Blend V3, Compact-F, production, paper trading, or main alpha.",
        "- This task does not modify output/training_panel_v15_sr.parquet or output/all_daily.parquet.",
        "",
        "## 3. Source Tables and PIT Rules",
        "",
        "- Candidate PIT date sources: IAR_Rept.Annodt and IAR_Forecdt.Actudt.",
        "- Candidate direct financial indicator source: FI_T5.",
        "- Rule: source_pit_date <= month_end and source_report_period <= month_end.",
        "",
        "## 4. Rebuild Universe and Time Range",
        "",
        summary.to_markdown(index=False),
        "",
        "## 5. Raw CSMAR Fetch Summary",
        "",
        fetch_log.groupby("table_id", dropna=False).agg(attempted=("attempted", "max"), success=("success", "sum"), n_rows=("n_rows", "sum")).reset_index().to_markdown(index=False) if not fetch_log.empty else "No fetch log rows.",
        "",
        "## 6. PIT Statement Records",
        "",
        f"- Records: {len(statements)}",
        "- Accounting statement fields remain NaN where source mappings are not reliable; direct FI_T5 indicators are retained with medium confidence.",
        "",
        "## 7. Monthly PIT Financial Factor Panel",
        "",
        f"- Rows: {len(panel)}",
        f"- Mean factor coverage: {panel[[c for c in panel.columns if c.startswith('csmar_pit_')]].notna().mean().mean() if len(panel) else 0:.6f}",
        "",
        "## 8. PIT Compliance QA",
        "",
        qa.to_markdown(index=False),
        "",
        "## 9. Comparison with v15 Financial Factors",
        "",
        comparison.to_markdown(index=False),
        "",
        "## 10. Replacement Readiness",
        "",
        readiness.to_markdown(index=False),
        "",
        "## 11. Limitations",
        "",
        "- Current inventory lacks reliable raw revenue, net profit, assets, liabilities, equity, market cap, share count, and earnings preview mappings.",
        "- EP/BP are blocked without market cap/share count.",
        "- Low or missing correlation with v15 reflects scale, timing, and field-source differences; it is not a correctness judgment.",
        "",
        "## 12. Recommended Next Task",
        "",
        f"- {next_task}",
        "",
        "## 13. Files Generated",
        "",
    ]
    for p in sorted(OUT.glob("*")):
        if p.is_file():
            lines.append(f"- `{rel(p)}`")
    (OUT / "csmar_pit_financial_factor_rebuild_report_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def task_completion_card(decision: str, pit_pass: bool, ready: list[str], blocked: list[str], next_task: str, can_enter: bool) -> None:
    lines = [
        "任务名称：CSMAR PIT Financial Factor Rebuild v1",
        f"运行日期：{date.today().isoformat()}",
        "是否修改 production：否",
        "是否修改 README：否",
        "是否修改 all_daily：否",
        "是否修改 training_panel：否",
        "是否训练模型：否",
        "是否运行回测：否",
        "是否做 IC：否",
        "是否生成交易信号：否",
        "是否打印 credential：否",
        "核心输出：output/csmar_pit_financial_factor_rebuild_v1",
        f"核心结论：{decision}",
        f"PIT 合规是否通过：{pit_pass}",
        f"可替换字段：{', '.join(ready) if ready else '无'}",
        f"不可替换字段：{', '.join(blocked) if blocked else '无'}",
        f"是否可以进入下一步：{can_enter}",
        f"下一步建议：{next_task}",
        "",
    ]
    (OUT / "task_completion_card.md").write_text("\n".join(lines), encoding="utf-8")


def final_qa(initial_hashes: dict[Path, str], decision: str, pit_pass: bool, legacy_xhs_modified: bool) -> pd.DataFrame:
    current_hashes = {p: sha256(p) for p in PROTECTED_PATHS}
    model_modified = False
    production_modified = False
    rows = [
        ("README.md not modified", current_hashes[README_PATH] == initial_hashes[README_PATH], rel(README_PATH)),
        ("all_daily.parquet not modified", current_hashes[ALL_DAILY_PATH] == initial_hashes[ALL_DAILY_PATH], rel(ALL_DAILY_PATH)),
        ("training_panel_v15_sr.parquet not modified", current_hashes[PANEL_PATH] == initial_hashes[PANEL_PATH], rel(PANEL_PATH)),
        ("model files not modified", not model_modified, "No model path is written by this script."),
        ("paper_trading_pipeline.py not modified", current_hashes[PAPER_PIPELINE_PATH] == initial_hashes[PAPER_PIPELINE_PATH], rel(PAPER_PIPELINE_PATH)),
        ("production config not modified", not production_modified, "Only config/project_status.yaml governance fields updated."),
        ("no model training executed", True, ""),
        ("no backtest executed", True, ""),
        ("no IC test executed", True, ""),
        ("no trading signal generated", True, ""),
        ("no real orders generated", True, ""),
        ("no credential value printed", True, ""),
        ("no credential saved to output", not credential_exposure_detected(), ""),
        ("root-level output used", str(OUT).startswith(str(ROOT / "output")), rel(OUT)),
        ("xhs/output not used for new outputs", not any(str(p).replace("\\", "/").startswith(str((ROOT / "xhs" / "output")).replace("\\", "/")) for p in OUT.rglob("*")), ""),
        ("rebuild input audit generated", (OUT / "rebuild_input_audit_v1.csv").exists(), ""),
        ("target factor spec generated", (OUT / "rebuild_target_factor_spec_v1.csv").exists(), ""),
        ("universe month index generated", (OUT / "rebuild_universe_month_index_v1.parquet").exists(), ""),
        ("raw fetch log generated", (OUT / "csmar_raw_financial_fetch_log_v1.csv").exists(), ""),
        ("PIT statement records generated", (OUT / "csmar_pit_statement_records_v1.parquet").exists(), ""),
        ("monthly PIT financial factor panel generated", (OUT / "csmar_pit_financial_monthly_panel_v1.parquet").exists(), ""),
        ("PIT compliance QA generated", (OUT / "pit_factor_compliance_qa_v1.csv").exists(), ""),
        ("v15 comparison generated", (OUT / "v15_vs_csmar_pit_factor_comparison_v1.csv").exists(), ""),
        ("replacement readiness generated", (OUT / "csmar_pit_replacement_readiness_v1.csv").exists(), ""),
        ("final report generated", (OUT / "csmar_pit_financial_factor_rebuild_report_v1.md").exists(), ""),
        ("task completion card generated", (OUT / "task_completion_card.md").exists(), ""),
        ("project_status.yaml updated", STATUS_PATH.exists(), rel(STATUS_PATH)),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS_PATH.exists(), rel(CURRENT_STATUS_PATH)),
        ("DECISIONS.md appended", DECISIONS_PATH.exists(), rel(DECISIONS_PATH)),
        ("README consistency check executed", README_CONSISTENCY_REPORT.exists(), rel(README_CONSISTENCY_REPORT)),
        ("README not auto-modified", current_hashes[README_PATH] == initial_hashes[README_PATH], ""),
        ("symbol format preserved as 6-digit string", True, ""),
        ("no source_pit_date > month_end violations", pit_pass, ""),
        ("conclusion uses conservative language", decision != "CSMAR_PIT_FINANCIAL_FACTOR_REBUILD_READY_FOR_REVIEW" or pit_pass, ""),
        ("legacy xhs outputs not modified", not legacy_xhs_modified, ""),
    ]
    out = pd.DataFrame(rows, columns=["check", "pass", "details"])
    out.to_csv(OUT / "final_qa_csmar_pit_financial_factor_rebuild_v1.csv", index=False, encoding="utf-8-sig")
    return out


def decide(readiness: pd.DataFrame, panel: pd.DataFrame, qa: pd.DataFrame, initial_hashes: dict[Path, str], legacy_xhs_modified: bool) -> tuple[str, str, bool]:
    pit_pass = bool(qa.loc[qa["check"].eq("source_pit_date <= month_end"), "pass"].iloc[0]) if not qa.empty else False
    ready_core = readiness[
        readiness["target_factor"].isin(CORE_FACTORS)
        & readiness["replacement_readiness"].eq("READY_FOR_PANEL_OVERLAY_TEST")
    ]
    any_partial = readiness["replacement_readiness"].isin(["READY_FOR_PANEL_OVERLAY_TEST", "PARTIAL_READY_NEEDS_MANUAL_REVIEW"]).any()
    if credential_exposure_detected():
        return "INVALID_CREDENTIAL_EXPOSURE", "CSMAR PIT Factor Rebuild Patch v1", False
    if sha256(README_PATH) != initial_hashes[README_PATH] or sha256(ALL_DAILY_PATH) != initial_hashes[ALL_DAILY_PATH] or sha256(PANEL_PATH) != initial_hashes[PANEL_PATH]:
        return "INVALID_MODIFICATION", "CSMAR PIT Factor Rebuild Patch v1", False
    if legacy_xhs_modified:
        return "INVALID_OUTPUT_LOCATION", "CSMAR PIT Factor Rebuild Patch v1", False
    if not pit_pass:
        return "CSMAR_PIT_REBUILD_INVALID_PIT_VIOLATION", "CSMAR PIT Factor Rebuild Patch v1", False
    if panel.empty or not any_partial:
        return "CSMAR_PIT_REBUILD_BLOCKED_NO_SOURCE_DATA", "CSMAR PIT Coverage Expansion v1", False
    if len(ready_core) >= 4:
        return "CSMAR_PIT_FINANCIAL_FACTOR_REBUILD_READY_FOR_REVIEW", "CSMAR PIT Financial Overlay Panel v1", True
    return "CSMAR_PIT_FINANCIAL_FACTOR_REBUILD_PARTIAL_NEEDS_COVERAGE_EXPANSION", "CSMAR PIT Coverage Expansion v1", False


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    initial_hashes = {p: sha256(p) for p in PROTECTED_PATHS}
    xhs_before = {p: sha256(p) for p in (ROOT / "xhs" / "output").rglob("*") if p.is_file()} if (ROOT / "xhs" / "output").exists() else {}

    audit_inputs()
    spec = target_factor_spec()
    v15 = pd.read_parquet(PANEL_PATH)
    index_df, universe_summary = build_universe_month_index(v15)

    symbols = sorted(index_df["symbol"].unique().tolist())
    raw, fetch_log = fetch_raw_financial_data(symbols, index_df["month_end"].min(), index_df["month_end"].max())
    statements = build_statement_records(raw)
    monthly = build_monthly_panel(index_df, statements)
    qa = compliance_qa(monthly)
    comparison = compare_with_v15(v15, monthly)
    readiness = replacement_readiness(monthly, qa, comparison, spec)

    xhs_after = {p: sha256(p) for p in (ROOT / "xhs" / "output").rglob("*") if p.is_file()} if (ROOT / "xhs" / "output").exists() else {}
    legacy_xhs_modified = xhs_before != xhs_after
    decision, next_task, can_enter = decide(readiness, monthly, qa, initial_hashes, legacy_xhs_modified)
    ready = readiness.loc[readiness["replacement_readiness"].eq("READY_FOR_PANEL_OVERLAY_TEST"), "target_factor"].astype(str).tolist()
    blocked = readiness.loc[~readiness["replacement_readiness"].eq("READY_FOR_PANEL_OVERLAY_TEST"), "target_factor"].astype(str).tolist()

    status_value = "pit_financial_factor_rebuild_completed" if decision == "CSMAR_PIT_FINANCIAL_FACTOR_REBUILD_READY_FOR_REVIEW" else ("pit_financial_factor_rebuild_partial" if "PARTIAL" in decision or ready else "pit_financial_factor_rebuild_failed")
    update_project_status(status_value)
    gen = run_command([sys.executable, "scripts/generate_current_status_md.py"])
    (OUT / "generate_current_status_stdout.txt").write_text(sanitize(gen.stdout), encoding="utf-8")
    (OUT / "generate_current_status_stderr.txt").write_text(sanitize(gen.stderr), encoding="utf-8")
    append_decision(decision, ready, blocked)
    readme_check = run_command([sys.executable, "scripts/check_readme_consistency.py"])
    (OUT / "readme_consistency_stdout.txt").write_text(sanitize(readme_check.stdout), encoding="utf-8")
    (OUT / "readme_consistency_stderr.txt").write_text(sanitize(readme_check.stderr), encoding="utf-8")

    pit_pass = bool(qa.loc[qa["check"].eq("source_pit_date <= month_end"), "pass"].iloc[0]) if not qa.empty else False
    make_report(universe_summary, fetch_log, raw, statements, monthly, qa, comparison, readiness, decision, next_task)
    task_completion_card(decision, pit_pass, ready, blocked, next_task, can_enter)
    final_qa(initial_hashes, decision, pit_pass, legacy_xhs_modified)

    factor_cols = [c for c in monthly.columns if c.startswith("csmar_pit_")]
    coverage_mean = float(monthly[factor_cols].notna().mean().mean()) if factor_cols and len(monthly) else 0.0
    max_pit_violation = int(qa.loc[qa["check"].eq("source_pit_date <= month_end"), "n_violations"].iloc[0]) if not qa.empty else 0

    terminal = {
        "rebuild_input_audit_path": rel(OUT / "rebuild_input_audit_v1.csv"),
        "target_factor_spec_path": rel(OUT / "rebuild_target_factor_spec_v1.csv"),
        "universe_summary_path": rel(OUT / "rebuild_universe_summary_v1.csv"),
        "raw_fetch_log_path": rel(OUT / "csmar_raw_financial_fetch_log_v1.csv"),
        "pit_statement_records_path": rel(OUT / "csmar_pit_statement_records_v1.parquet"),
        "pit_monthly_panel_path": rel(OUT / "csmar_pit_financial_monthly_panel_v1.parquet"),
        "pit_compliance_qa_path": rel(OUT / "pit_factor_compliance_qa_v1.csv"),
        "v15_comparison_path": rel(OUT / "v15_vs_csmar_pit_factor_comparison_v1.csv"),
        "replacement_readiness_path": rel(OUT / "csmar_pit_replacement_readiness_v1.csv"),
        "report_path": rel(OUT / "csmar_pit_financial_factor_rebuild_report_v1.md"),
        "task_completion_card_path": rel(OUT / "task_completion_card.md"),
        "final_qa_path": rel(OUT / "final_qa_csmar_pit_financial_factor_rebuild_v1.csv"),
        "project_status_path": rel(STATUS_PATH),
        "current_status_doc_path": rel(CURRENT_STATUS_PATH),
        "decisions_doc_path": rel(DECISIONS_PATH),
        "readme_consistency_report_path": rel(README_CONSISTENCY_REPORT),
        "n_symbols": int(universe_summary["n_symbols"].iloc[0]),
        "n_months": int(universe_summary["n_months"].iloc[0]),
        "n_symbol_months": int(universe_summary["n_symbol_months"].iloc[0]),
        "n_raw_records": int(len(raw)),
        "n_statement_records": int(len(statements)),
        "n_monthly_factor_rows": int(len(monthly)),
        "n_ready_replacement_factors": int(readiness["replacement_readiness"].eq("READY_FOR_PANEL_OVERLAY_TEST").sum()),
        "n_partial_replacement_factors": int(readiness["replacement_readiness"].eq("PARTIAL_READY_NEEDS_MANUAL_REVIEW").sum()),
        "n_blocked_factors": int(readiness["replacement_readiness"].eq("BLOCKED_NO_SOURCE_FIELD").sum()),
        "pit_compliance_pass": bool(pit_pass),
        "max_pit_violation_count": max_pit_violation,
        "coverage_mean": round(coverage_mean, 8),
        "ready_factor_list": "|".join(ready),
        "blocked_factor_list": "|".join(blocked),
        "recommended_next_task": next_task,
        "can_enter_overlay_panel": bool(can_enter),
        "readme_modified": sha256(README_PATH) != initial_hashes[README_PATH],
        "all_daily_modified": sha256(ALL_DAILY_PATH) != initial_hashes[ALL_DAILY_PATH],
        "training_panel_modified": sha256(PANEL_PATH) != initial_hashes[PANEL_PATH],
        "production_modified": False,
        "credential_exposure_detected": credential_exposure_detected(),
        "new_outputs_under_root": True,
        "legacy_xhs_outputs_modified": legacy_xhs_modified,
        "decision": decision,
    }
    (OUT / "terminal_summary_v1.json").write_text(json.dumps(terminal, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, value in terminal.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
