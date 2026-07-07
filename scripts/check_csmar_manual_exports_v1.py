from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "csmar_pack_download_executor_v1"
EXPORT_DIR = ROOT / "data" / "csmar_exports"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def safe_join(values: list[str]) -> str:
    return "|".join(sorted(dict.fromkeys(v for v in values if v)))


def read_sample(path: Path) -> tuple[bool, pd.DataFrame, str]:
    try:
        if path.suffix.lower() == ".csv":
            return True, pd.read_csv(path, nrows=50, dtype=str), ""
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return True, pd.read_excel(path, nrows=50, dtype=str), ""
        if path.suffix.lower() == ".parquet":
            return True, pd.read_parquet(path).head(50), ""
        return False, pd.DataFrame(), "unsupported file type"
    except Exception as exc:
        return False, pd.DataFrame(), f"{type(exc).__name__}: {str(exc)[:160]}"


def detect_table_name(path: Path, df: pd.DataFrame) -> str:
    text = f"{path.name} {' '.join(map(str, df.columns))}".lower()
    candidates = {
        "IAR_Rept": ["iar_rept", "annodt", "财务报告披露"],
        "IAR_Forecdt": ["iar_forecdt", "actudt", "firforecdt", "预约披露"],
        "FI_T5": ["fi_t5", "f050501b", "f051501b", "财务指标"],
        "FN_Fn050": ["fn_fn050", "fn050", "销售费用"],
        "FN_Fn060": ["fn_fn060", "fn060", "研发费用"],
    }
    for table, tokens in candidates.items():
        if any(token.lower() in text for token in tokens):
            return table
    return ""


def detect_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    lower = {str(c).lower(): str(c) for c in df.columns}
    found = []
    for cand in candidates:
        if cand.lower() in lower:
            found.append(lower[cand.lower()])
    return found


def likely_factors(cols: list[str]) -> list[str]:
    low = {c.lower() for c in cols}
    out = []
    if any(c in low for c in ["f050501b", "roe"]):
        out.append("ROE")
    if any(c in low for c in ["f051501b", "netmargin", "net_profit_margin"]):
        out.append("NetMargin")
    if {"total_assets", "total_liabilities"}.issubset(low) or any("asset" in c and "liab" in " ".join(low) for c in low):
        out.append("Debt_Ratio")
    if any(c in low for c in ["revenue", "operating_revenue", "营业收入"]) and any(c in low for c in ["net_profit_parent", "归母净利润"]):
        out.extend(["ProfitGrowth_YoY", "RevGrowth_YoY"])
    if any(c in low for c in ["fn05002", "f051701b"]):
        out.append("sales_expense_to_revenue")
    if any(c in low for c in ["fn_fn06002", "fn06002"]):
        out.append("rd_expense_to_revenue")
    return out


def scan_exports() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    files = [p for p in EXPORT_DIR.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".parquet"}]
    for path in sorted(files):
        readable, df, note = read_sample(path)
        cols = [str(c) for c in df.columns] if readable else []
        symbol_cols = detect_columns(df, ["Stkcd", "stkcd", "symbol", "stock_code", "证券代码"]) if readable else []
        report_cols = detect_columns(df, ["Accper", "accper", "report_period", "统计截止日期", "报告期"]) if readable else []
        pit_cols = detect_columns(df, ["Annodt", "Actudt", "Firforecdt", "Firchangdt", "Secchangdt", "Thirchangdt", "publish_date", "disclosure_date", "公告日", "披露日"]) if readable else []
        financial = [c for c in cols if c.lower() in {"f050501b", "f051501b", "f051701b", "f053301b", "fn05002", "fn_fn06002"} or any(token in c for token in ["营业收入", "净利润", "总资产", "总负债", "ROE", "净利率", "资产负债率"])]
        rows.append({
            "file_path": rel(path),
            "file_type": path.suffix.lower().lstrip("."),
            "readable": readable,
            "n_rows_sampled": int(len(df)) if readable else 0,
            "n_columns": int(len(cols)) if readable else 0,
            "detected_table_name": detect_table_name(path, df) if readable else "",
            "detected_symbol_columns": safe_join(symbol_cols),
            "detected_report_period_columns": safe_join(report_cols),
            "detected_pit_date_columns": safe_join(pit_cols),
            "detected_financial_fields": safe_join(financial),
            "likely_supported_factors": safe_join(likely_factors(cols)),
            "notes": note,
        })
    if not rows:
        rows.append({
            "file_path": rel(EXPORT_DIR),
            "file_type": "directory",
            "readable": True,
            "n_rows_sampled": 0,
            "n_columns": 0,
            "detected_table_name": "",
            "detected_symbol_columns": "",
            "detected_report_period_columns": "",
            "detected_pit_date_columns": "",
            "detected_financial_fields": "",
            "likely_supported_factors": "",
            "notes": "No local CSMAR export files found.",
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = scan_exports()
    out_path = OUT / "manual_export_file_check_v1.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"manual_export_file_check_path={rel(out_path)}")
    print(f"n_files_checked={len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
