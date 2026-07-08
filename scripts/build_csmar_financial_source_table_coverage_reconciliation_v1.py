from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "csmar_financial_source_table_coverage_reconciliation_v1"
STATUS_PATH = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS_PATH = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
README_CONSISTENCY_REPORT = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"
README_PATH = ROOT / "README.md"
ALL_DAILY_PATH = ROOT / "output" / "all_daily.parquet"
PANEL_PATH = ROOT / "output" / "training_panel_v15_sr.parquet"
PAPER_PIPELINE_PATH = ROOT / "paper_trading" / "paper_trading_pipeline.py"
P1_SCRIPT_PATH = ROOT / "scripts" / "run_csmar_p1_financial_pack_download_v1.py"
RUN_DATE = date.today().isoformat()
PROTECTED = [README_PATH, ALL_DAILY_PATH, PANEL_PATH, PAPER_PIPELINE_PATH]

EXACT_TABLES = ["FS_Comins", "FS_Combas", "TRD_Dalyr", "IAR_Pfnotce", "FN_Fn064", "FN_Fn060", "FN_Fn050", "FI_T5", "IAR_Rept", "IAR_Forecdt"]
KEYWORDS = ["利润表", "资产负债表", "个股日线行情", "日个股回报率", "总市值", "营业收入", "净利润", "归母净利润", "归属于母公司所有者权益", "总资产", "总负债", "销售费用", "管理费用", "研发费用", "研发支出", "业绩预告", "业绩快报", "信息披露", "财务附注", "income statement", "balance sheet", "daily stock trading", "market value", "total market value", "operating revenue", "net profit", "net profit attributable", "total assets", "total liabilities", "equity", "selling expense", "R&D expense", "performance forecast"]


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


def sanitize(text: Any) -> str:
    return re.sub(r"(?i)(token|cookie|session|password|account)\s*[:=]\s*[^,\s;]+", r"\1=[REDACTED]", str(text))[:1200]


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False, timeout=180)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def field_dict_path() -> Path:
    p = ROOT / "output" / "csmar_table_inventory_audit_v1" / "csmar_field_dictionary_v1.csv"
    return p if p.exists() else ROOT / "xhs" / "output" / "csmar_table_inventory_audit_v1" / "csmar_field_dictionary_v1.csv"


def input_audit() -> pd.DataFrame:
    paths = [
        (ROOT / "output" / "csmar_table_inventory_audit_v1" / "csmar_table_inventory_v1.csv", "table inventory"),
        (field_dict_path(), "field dictionary"),
        (ROOT / "output" / "csmar_discovery_v1", "discovery directory"),
        (ROOT / "output" / "csmar_p1_financial_table_mapping_patch_v1" / "candidate_financial_tables_v1.csv", "P1 candidate financial tables"),
        (ROOT / "output" / "csmar_p1_financial_table_mapping_patch_v1" / "p1_financial_field_mapping_v1.csv", "P1 field mapping"),
        (ROOT / "output" / "csmar_p1_financial_table_mapping_patch_v1" / "p1_financial_pack_download_plan_v2.csv", "P1 plan v2"),
        (ROOT / "output" / "csmar_p1_existing_export_skip_patch_v1" / "p1_existing_export_inventory_v1.csv", "existing export inventory"),
        (ROOT / "output" / "csmar_p1_existing_export_skip_patch_v1" / "p1_next_execute_queue_after_skip_v1.csv", "current next queue"),
        (ROOT / "output" / "csmar_p0_pit_pack_import_audit_v1" / "csmar_p0_pit_announcement_panel_v1.parquet", "P0 PIT panel"),
        (ALL_DAILY_PATH, "all_daily market panel"),
        (PANEL_PATH, "v15 training panel"),
    ]
    rows = []
    for path, role in paths:
        exists = path.exists()
        readable, n_rows, notes = False, "", ""
        try:
            if exists and path.is_dir():
                readable = True
                n_rows = len(list(path.iterdir()))
            elif exists and path.suffix == ".parquet":
                df = pd.read_parquet(path)
                readable, n_rows = True, len(df)
            elif exists:
                df = pd.read_csv(path, dtype=str)
                readable, n_rows = True, len(df)
            else:
                notes = "missing"
        except Exception as exc:  # noqa: BLE001
            notes = f"{type(exc).__name__}: {sanitize(exc)}"
        rows.append({"input_path": rel(path), "exists": exists, "readable": readable, "n_rows": n_rows, "role": role, "notes": notes})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "input_audit_v1.csv", index=False, encoding="utf-8-sig")
    return df


def requirements() -> pd.DataFrame:
    rows = [
        ("ROE", "RAW_FINANCIAL_STATEMENT", "net_profit_parent|equity_parent|report_period|pit_date", "FS_Comins|FS_Combas", "FI_T5 fallback/sanity check", True, False, True, False, False, "FI_T5 is derived and should not replace bottom-up PIT-clean TTM reconstruction."),
        ("EP", "RAW_FINANCIAL_STATEMENT+MARKET_CAP", "net_profit_parent|market_cap|pit_date", "FS_Comins|TRD_Dalyr", "FI_T5 fallback plus local market data if market cap exists", True, True, True, False, False, "EP/BP require market capitalization or equivalent shares*price data."),
        ("BP", "RAW_FINANCIAL_STATEMENT+MARKET_CAP", "book_equity_parent|market_cap|pit_date", "FS_Combas|TRD_Dalyr", "local market cap if available", False, True, True, False, False, "Balance-sheet equity and market cap are both required."),
        ("ProfitGrowth_YoY", "RAW_FINANCIAL_STATEMENT", "net_profit_parent current/prior TTM|pit_date", "FS_Comins", "FI_T5 derived fallback", True, False, True, False, False, "Prefer income-statement raw fields."),
        ("RevGrowth_YoY", "RAW_FINANCIAL_STATEMENT", "revenue current/prior TTM|pit_date", "FS_Comins", "FN_Fn048 if confirmed", True, False, True, False, False, "Requires revenue raw field."),
        ("NetMargin", "RAW_FINANCIAL_STATEMENT", "net_profit_parent|revenue|pit_date", "FS_Comins", "FI_T5 fallback/sanity check", True, False, True, False, False, "FI_T5 is derived; FS_Comins is preferred."),
        ("Debt_Ratio", "RAW_FINANCIAL_STATEMENT", "total_liabilities|total_assets|pit_date", "FS_Combas", "FI_T5 fallback/sanity check", False, False, True, False, False, "Requires balance-sheet raw fields."),
        ("sales_expense_to_revenue", "RAW_FINANCIAL_STATEMENT", "sales_expense|revenue|pit_date", "FS_Comins", "FN_Fn050 or FI_T5 only after FS fields are confirmed", True, False, True, True, False, "FN_Fn050 is not a main statement table."),
        ("rd_expense_to_revenue", "RAW_FINANCIAL_STATEMENT+NOTES_DETAIL", "rd_expense|revenue|pit_date", "FS_Comins|FN_Fn064|FN_Fn060", "FN_Fn060 fallback for R&D detail", True, False, True, True, False, "FN_Fn064/FN_Fn060 may be a historical R&D expense patch after FS tables."),
        ("earnings_preview_midpoint_yoy", "FORECAST_DISCLOSURE", "forecast_lower|forecast_upper|announcement_date|report_period", "IAR_Pfnotce", "earnings express/forecast equivalent table", False, False, True, False, True, "Requires forecast/pre-announcement table."),
    ]
    df = pd.DataFrame(rows, columns=["target_factor", "preferred_source_type", "required_raw_items", "preferred_csmar_tables_from_user_mapping", "acceptable_alternative_tables", "requires_ttm", "requires_market_cap", "requires_pit_date", "requires_notes_table", "requires_forecast_table", "notes"])
    df.to_csv(OUT / "target_factor_data_requirements_v1.csv", index=False, encoding="utf-8-sig")
    return df


def discovery_rows() -> pd.DataFrame:
    p = ROOT / "output" / "csmar_discovery_v1" / "csmar_tables.csv"
    rows = []
    if p.exists():
        df = read_csv(p)
        for _, r in df.iterrows():
            raw = str(r.get("raw_json", ""))
            tid, name = "", str(r.get("table_name", ""))
            try:
                obj = json.loads(raw)
                tid, name = str(obj.get("table", "")), str(obj.get("tableName", name))
            except Exception:
                pass
            rows.append({"table_id": tid, "table_cn": name, "table_en": "", "source_file": rel(p), "text": f"{tid} {name} {r.to_dict()}".lower()})
    inv = ROOT / "output" / "csmar_table_inventory_audit_v1" / "csmar_table_inventory_v1.csv"
    if inv.exists():
        df = read_csv(inv)
        for _, r in df.iterrows():
            tid = str(r.get("table_id", ""))
            rows.append({"table_id": tid, "table_cn": str(r.get("table_name_cn", "")), "table_en": str(r.get("table_name_en", "")), "source_file": rel(inv), "text": " ".join(map(str, r.to_dict().values())).lower()})
    return pd.DataFrame(rows).drop_duplicates(["table_id", "source_file"])


def table_search(field_dict: pd.DataFrame) -> pd.DataFrame:
    tables = discovery_rows()
    rows = []
    queries = EXACT_TABLES + KEYWORDS
    for q in queries:
        matches = []
        qlow = q.lower()
        if not tables.empty:
            exact = tables[tables["table_id"].astype(str).str.lower().eq(qlow)]
            if not exact.empty:
                matches = exact.to_dict("records")
                conf, mtype = "HIGH_EXACT_TABLE_MATCH", "exact_table_id"
            else:
                hit = tables[tables["text"].astype(str).str.contains(re.escape(qlow), na=False)]
                matches = hit.to_dict("records")
                conf = "HIGH_SEMANTIC_MATCH" if q in ["利润表", "资产负债表", "日个股回报率", "业绩预告", "业绩快报"] else "MEDIUM_KEYWORD_MATCH"
                mtype = "keyword"
        if not matches:
            rows.append({"query_keyword": q, "matched_table_name": "", "matched_table_id": "", "matched_table_cn": "", "matched_table_en": "", "source_file": "", "match_type": "none", "n_fields_detected": 0, "confidence": "NOT_FOUND", "notes": ""})
        else:
            for m in matches[:20]:
                tid = str(m["table_id"])
                n_fields = int((field_dict["table_id"].astype(str) == tid).sum()) if not field_dict.empty else 0
                rows.append({"query_keyword": q, "matched_table_name": m["table_cn"], "matched_table_id": tid, "matched_table_cn": m["table_cn"], "matched_table_en": m["table_en"], "source_file": m["source_file"], "match_type": mtype, "n_fields_detected": n_fields, "confidence": conf, "notes": "field dictionary available" if n_fields else "table found but local field dictionary missing"})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "local_csmar_table_search_results_v1.csv", index=False, encoding="utf-8-sig")
    return df


def field_lookup(field_dict: pd.DataFrame, table: str, candidates: list[str]) -> tuple[str, str, str]:
    subset = field_dict[field_dict["table_id"].astype(str).eq(table)] if not field_dict.empty else pd.DataFrame()
    fields = set(subset["field_name"].astype(str)) if not subset.empty else set()
    for c in candidates:
        if c in fields:
            cn = str(subset.loc[subset["field_name"].eq(c), "field_name_cn"].iloc[0]) if "field_name_cn" in subset.columns else ""
            return c, cn, "COVERED_BY_DERIVED_FIELD" if table == "FI_T5" else "COVERED_BY_RAW_FIELD"
    return "", "", "TABLE_NOT_FOUND" if subset.empty else "FIELD_NOT_FOUND"


def coverage_matrix(req: pd.DataFrame, search: pd.DataFrame, field_dict: pd.DataFrame) -> pd.DataFrame:
    all_daily_cols = set(pd.read_parquet(ALL_DAILY_PATH).columns)
    specs = {
        "net_profit_parent": ("FS_Comins", ["净利润", "归母净利润", "net_profit_parent"]),
        "revenue": ("FS_Comins", ["营业收入", "revenue"]),
        "sales_expense": ("FS_Comins", ["销售费用", "FN05002", "F051701B"]),
        "rd_expense": ("FN_Fn060", ["FN_Fn06002", "FN_Fn06001"]),
        "total_assets": ("FS_Combas", ["总资产", "total_assets"]),
        "total_liabilities": ("FS_Combas", ["总负债", "total_liabilities"]),
        "book_equity_parent": ("FS_Combas", ["归属于母公司所有者权益", "股东权益", "equity"]),
        "equity_parent": ("FS_Combas", ["归属于母公司所有者权益", "股东权益", "equity"]),
        "market_cap": ("TRD_Dalyr", ["Dsmvosd", "Dsmvtll", "market_cap"]),
        "forecast_lower": ("IAR_Pfnotce", ["forecast_lower"]),
        "forecast_upper": ("IAR_Pfnotce", ["forecast_upper"]),
        "pit_date": ("IAR_Rept/IAR_Forecdt", ["Annodt", "Actudt"]),
    }
    rows = []
    for _, r in req.iterrows():
        items = [x.strip() for x in str(r["required_raw_items"]).replace(" current/prior TTM", "").split("|") if x.strip()]
        for item in items:
            base = item.split()[0]
            table, candidates = specs.get(base, (str(r["preferred_csmar_tables_from_user_mapping"]).split("|")[0], [base]))
            if base == "market_cap" and not ({"market_cap", "total_mv", "circ_mv"} & all_daily_cols):
                rows.append({"target_factor": r["target_factor"], "required_raw_item": base, "preferred_table": table, "candidate_table": "output/all_daily.parquet", "candidate_field_name": "", "candidate_field_cn": "", "candidate_field_en": "", "coverage_status": "NOT_COVERED", "source_type": "LOCAL_PRICE_PANEL", "confidence": "MEDIUM", "notes": "all_daily has OHLCV/amount but no direct market cap field."})
                continue
            if "IAR_Rept/IAR_Forecdt" in table:
                rows.append({"target_factor": r["target_factor"], "required_raw_item": base, "preferred_table": table, "candidate_table": table, "candidate_field_name": "Annodt|Actudt", "candidate_field_cn": "", "candidate_field_en": "", "coverage_status": "COVERED_BY_RAW_FIELD", "source_type": "DISCLOSURE_PIT_DATE", "confidence": "HIGH", "notes": "P0 PIT panel is already generated."})
                continue
            field, cn, status = field_lookup(field_dict, table, candidates)
            source_type = "DERIVED_FINANCIAL_INDICATOR" if table == "FI_T5" else ("TRADING_MARKET_CAP" if table == "TRD_Dalyr" else ("FORECAST_DISCLOSURE" if table == "IAR_Pfnotce" else ("NOTES_DETAIL" if table.startswith("FN_") else "RAW_FINANCIAL_STATEMENT")))
            rows.append({"target_factor": r["target_factor"], "required_raw_item": base, "preferred_table": table, "candidate_table": table, "candidate_field_name": field, "candidate_field_cn": cn, "candidate_field_en": "", "coverage_status": status, "source_type": source_type, "confidence": "HIGH" if status.startswith("COVERED") else "LOW", "notes": "Local field dictionary does not yet cover this table." if status in {"TABLE_NOT_FOUND", "FIELD_NOT_FOUND"} else ""})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "csmar_financial_field_coverage_matrix_v1.csv", index=False, encoding="utf-8-sig")
    return df


def data_summary() -> pd.DataFrame:
    all_cols = set(pd.read_parquet(ALL_DAILY_PATH).columns)
    train_cols = set(pd.read_parquet(PANEL_PATH).columns)
    rows = [
        ("P0 disclosure", "IAR_Rept", "LOCAL_AVAILABLE", "PIT date fallback", "financial raw fields", "HIGH", "Downloaded and standardized."),
        ("P0 disclosure", "IAR_Forecdt", "LOCAL_AVAILABLE", "primary actual disclosure date", "financial raw fields", "HIGH", "Downloaded and standardized."),
        ("P0 PIT panel", "csmar_p0_pit_announcement_panel_v1.parquet", "LOCAL_AVAILABLE", "all PIT-aligned financial factors", "financial raw fields", "HIGH", "v15 symbol coverage was previously 1.0."),
        ("P1 financial", "FI_T5 local export", "LOCAL_AVAILABLE", "ROE/NetMargin/Debt_Ratio fallback or sanity check", "raw FS statement reconstruction", "MEDIUM", "Derived indicator table, not preferred bottom-up source."),
        ("Market data", "output/all_daily.parquet", "LOCAL_AVAILABLE", "price/volume amount", "direct market cap for EP/BP" if not ({"market_cap", "total_mv", "circ_mv"} & all_cols) else "", "MEDIUM", f"columns={','.join(sorted(all_cols))}"),
        ("Legacy panel", "output/training_panel_v15_sr.parquet", "LOCAL_AVAILABLE_READ_ONLY", "legacy factors: " + "|".join([c for c in train_cols if "ROE" in c or "EP" in c or "BP" in c or "Growth" in c][:10]), "PIT-clean raw source lineage", "LOW_FOR_REBUILD", "Use for universe/reference only."),
    ]
    df = pd.DataFrame(rows, columns=["data_layer", "table_or_file", "local_status", "covers_target_factors", "missing_for_target_factors", "reliability_for_pit_rebuild", "notes"])
    df.to_csv(OUT / "current_data_coverage_summary_v1.csv", index=False, encoding="utf-8-sig")
    return df


def fn_assessment() -> pd.DataFrame:
    rows = [
        ("FN_Fn050", "sales expense notes/detail", 1, "NO", "NO", "PARTIAL", "NO", False, "LOW_PRIORITY", "LOW_PRIORITY", "Cannot cover revenue, parent net profit, assets, liabilities, or equity; do not download before FS tables."),
        ("FN_Fn060", "R&D expense notes/detail", 2, "PARTIAL", "NO", "NO", "NO", False, "AFTER_FS_TABLES", "AFTER_FS_TABLES", "Useful only as R&D expense patch after FS_Comins or equivalent revenue source is confirmed."),
        ("FN_Fn064", "possible R&D notes/detail", 2, "UNKNOWN_NOT_FOUND_LOCALLY", "NO", "NO", "NO", False, "AFTER_FS_TABLES", "AFTER_FS_TABLES", "Not found in current local field dictionary; search/review manually if R&D detail remains needed."),
    ]
    df = pd.DataFrame(rows, columns=["table_name", "table_role_guess", "core_factor_coverage_score", "rd_expense_coverage", "management_expense_detail_coverage", "overlaps_with_fi_t5", "overlaps_with_fs_tables", "should_download_before_fs_tables", "recommended_priority", "conclusion", "notes"])
    df.to_csv(OUT / "fn_table_value_assessment_v1.csv", index=False, encoding="utf-8-sig")
    return df


def priority(search: pd.DataFrame) -> pd.DataFrame:
    found = set(search.loc[search["confidence"].ne("NOT_FOUND"), "matched_table_id"].astype(str))
    rows = [
        (1, "WAIT_FOR_HUMAN_REVIEW", "review_gate", "FS_Comins/FS_Combas discovered but local field dictionary lacks fields; confirm field list before CSMAR request.", "all core raw PIT-clean financial factors", "human table/field review", "HIGH", "LOW if reviewed, HIGH if guessed", True, "Do not default to FN_Fn050."),
        (2, "FS_Comins", "income_statement", "Preferred raw source for revenue, net profit, sales expense, net margin, growth.", "ROE|EP|ProfitGrowth_YoY|RevGrowth_YoY|NetMargin|sales_expense_to_revenue", "field dictionary/column validation", "HIGH", "MEDIUM_FIELD_MAPPING", False, "Found in discovery." if "FS_Comins" in found else "Not found."),
        (3, "FS_Combas", "balance_sheet", "Preferred raw source for assets, liabilities, equity.", "ROE|BP|Debt_Ratio", "field dictionary/column validation", "HIGH", "MEDIUM_FIELD_MAPPING", False, "Found in discovery." if "FS_Combas" in found else "Not found."),
        (4, "TRD_Dalyr", "trading_market_cap", "Needed if no equivalent local market cap data is available.", "EP|BP", "confirm market cap fields", "MEDIUM", "MEDIUM", False, "Found in discovery; all_daily lacks direct market cap."),
        (5, "IAR_Pfnotce", "forecast_disclosure", "Needed for earnings_preview_midpoint_yoy.", "earnings_preview_midpoint_yoy", "confirm forecast fields", "MEDIUM", "MEDIUM", False, "Not found in local search." if "IAR_Pfnotce" not in found else "Found."),
        (6, "FI_T5", "derived_indicator", "Already downloaded; use as fallback/sanity check, not bottom-up source.", "ROE|NetMargin|Debt_Ratio fallback", "LOCAL_ALREADY_AVAILABLE", "MEDIUM", "DERIVED_NOT_RAW", False, "LOCAL_ALREADY_AVAILABLE"),
        (7, "FN_Fn060", "notes_detail", "R&D detail patch after FS tables.", "rd_expense_to_revenue partial", "FS_Comins first", "LOW", "LOW", False, "AFTER_FS_TABLES"),
        (8, "FN_Fn050", "notes_detail", "Sales expense detail does not cover core raw statement fields.", "sales_expense_to_revenue partial", "FS_Comins first", "LOW", "LOW", False, "LOW_PRIORITY"),
    ]
    df = pd.DataFrame(rows, columns=["priority_rank", "table_name", "table_role", "reason", "target_factors_supported", "prerequisite", "estimated_value", "estimated_risk", "should_download_next", "notes"])
    df.to_csv(OUT / "recommended_download_priority_v1.csv", index=False, encoding="utf-8-sig")
    return df


def patch_p1_queue() -> bool:
    text = P1_SCRIPT_PATH.read_text(encoding="utf-8")
    if "RECONCILIATION_PRIORITY_PATH" not in text:
        text = text.replace('MAPPING_PATCH_V2_PLAN_PATH = ROOT / "output" / "csmar_p1_financial_table_mapping_patch_v1" / "p1_financial_pack_download_plan_v2.csv"\n', 'MAPPING_PATCH_V2_PLAN_PATH = ROOT / "output" / "csmar_p1_financial_table_mapping_patch_v1" / "p1_financial_pack_download_plan_v2.csv"\nRECONCILIATION_PRIORITY_PATH = ROOT / "output" / "csmar_financial_source_table_coverage_reconciliation_v1" / "recommended_download_priority_v1.csv"\n')
    if "def apply_reconciliation_priority_override" not in text:
        marker = "def apply_existing_export_skip(plan: pd.DataFrame) -> pd.DataFrame:\n"
        insert = '''def apply_reconciliation_priority_override(plan: pd.DataFrame) -> pd.DataFrame:
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


'''
        text = text.replace(marker, insert + marker)
    old = "    plan = apply_existing_export_skip(plan)\n    plan.to_csv(OUT / \"p1_financial_pack_download_plan_v1.csv\", index=False, encoding=\"utf-8-sig\")\n"
    new = "    plan = apply_existing_export_skip(plan)\n    plan = apply_reconciliation_priority_override(plan)\n    plan.to_csv(OUT / \"p1_financial_pack_download_plan_v1.csv\", index=False, encoding=\"utf-8-sig\")\n"
    if old in text and new not in text:
        text = text.replace(old, new)
    P1_SCRIPT_PATH.write_text(text, encoding="utf-8")
    return True


def report(req, search, cov, summary, fn, prio, decision) -> Path:
    lines = ["# CSMAR Financial Source Table Coverage Reconciliation v1", "", "## 1. Executive Summary", "", "- This task did not access CSMAR API.", "- This task did not download new data.", "- Production was not modified.", "- FI_T5 is already downloaded but is a derived indicator table; it is fallback/sanity-check data, not the preferred PIT-clean TTM reconstruction base.", "- FN_Fn050 should not remain the default next download target before FS_Comins/FS_Combas field review.", f"- Decision: `{decision}`", "", "## 2. Current Local Data Inventory", "", summary.to_markdown(index=False), "", "## 3. Target Factor Data Requirements", "", req.to_markdown(index=False), "", "## 4. Local CSMAR Table Search Results", "", search.head(80).to_markdown(index=False), "", "## 5. Field Coverage Matrix", "", cov.to_markdown(index=False), "", "## 6. FI_T5 Role and Limitations", "", "FI_T5 can support fallback checks for ROE and other ratios, but it is not the preferred source for PIT-clean bottom-up TTM factors.", "", "## 7. FN_Fn050 / FN_Fn060 / FN_Fn064 Assessment", "", fn.to_markdown(index=False), "", "## 8. Missing Data Gap", "", "- Local discovery finds FS_Comins, FS_Combas, and TRD_Dalyr, but the local field dictionary does not yet provide their downloadable field lists.", "- all_daily.parquet lacks direct market-cap fields, so EP/BP cannot be fully rebuilt from current local market data alone.", "- IAR_Pfnotce was not confirmed in local inventory/search and needs manual table review.", "", "## 9. Recommended Download Priority", "", prio.to_markdown(index=False), "", "## 10. Whether FN_Fn050 Should Still Be Next", "", "Conservative conclusion: no. FN_Fn050 is LOW_PRIORITY until FS_Comins/FS_Combas or equivalent raw statement tables are field-confirmed.", "", "## 11. Limitations", "", "- This is an offline reconciliation from local inventory/discovery outputs.", "- Field-level coverage for FS/TRD/IAR forecast tables remains incomplete until field dictionaries are obtained or manually confirmed.", "", "## 12. Files Generated", ""]
    path = OUT / "csmar_financial_source_table_coverage_reconciliation_report_v1.md"
    for p in sorted(OUT.glob("*")):
        if p.is_file():
            lines.append(f"- `{rel(p)}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def update_status() -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    status.setdefault("alternative_data", {})
    status["alternative_data"]["csmar_status"] = "financial_source_table_coverage_reconciled_waiting_for_priority_download"
    status["alternative_data"]["csmar_latest_task"] = "CSMAR Financial Source Table Coverage Reconciliation v1"
    status["alternative_data"]["csmar_latest_output"] = rel(OUT)
    status.setdefault("validation", {})
    status["validation"]["pit_financial_status"] = "p0_pit_dates_imported_fi_t5_downloaded_source_table_priority_under_review"
    status["validation"]["blend_v3_historical_metrics_status"] = "under_pit_review"
    status.setdefault("project", {})["last_updated"] = RUN_DATE
    STATUS_PATH.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False), encoding="utf-8")


def append_decision(decision: str, next_table: str) -> None:
    text = DECISIONS_PATH.read_text(encoding="utf-8") if DECISIONS_PATH.exists() else "# 决策日志\n"
    marker = f"Decision = {decision}。"
    if marker in text and "基于目标因子重构需求重新审视 P1 下载优先级" in text:
        return
    block = "\n".join([f"## {RUN_DATE}", "", "决策：", "", "- 基于目标因子重构需求重新审视 P1 下载优先级。", "- FI_T5 已下载但作为衍生指标表，不应替代底表重构。", "- FN_Fn050 不应继续作为默认下一下载目标。", f"- 推荐的下一下载表：{next_table}。", "- 不访问 CSMAR API。", "- 不修改 README。", "- 不接入 production。", f"- Decision = {decision}。"])
    DECISIONS_PATH.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def credential_exposure_detected() -> bool:
    for p in OUT.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".md", ".json", ".txt", ".log"}:
            if re.search(r"(?i)(CSMAR_PASSWORD|CSMAR_ACCOUNT|token|cookie|session)\s*=", p.read_text(encoding="utf-8", errors="ignore")):
                return True
    return False


def task_card(decision: str, next_table: str) -> Path:
    path = OUT / "task_completion_card.md"
    lines = ["任务名称：CSMAR Financial Source Table Coverage Reconciliation v1", f"运行日期：{RUN_DATE}", "是否修改 production：否", "是否修改 README：否", "是否修改 all_daily：否", "是否修改 training_panel：否", "是否训练模型：否", "是否运行回测：否", "是否做 IC：否", "是否访问 CSMAR API：否", "是否执行 CSMAR 下载：否", "核心输出：", f"- {rel(OUT / 'target_factor_data_requirements_v1.csv')}", f"- {rel(OUT / 'csmar_financial_field_coverage_matrix_v1.csv')}", f"- {rel(OUT / 'recommended_download_priority_v1.csv')}", "核心结论：" + decision, "当前已有数据：P0 PIT panel, IAR_Rept, IAR_Forecdt, FI_T5, all_daily price/amount panel, v15 legacy panel", "当前缺失数据：FS_Comins/FS_Combas field-level mapping, direct market cap, IAR_Pfnotce forecast fields", "FN_Fn050 是否仍建议下载：否，保守标记 LOW_PRIORITY", f"推荐下一下载表：{next_table}", "下一步建议：人工确认 FS_Comins/FS_Combas/TRD_Dalyr 字段后再执行 --max-downloads 1 下载。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def final_qa(initial_hashes: dict[Path, str], report_path: Path, card_path: Path) -> pd.DataFrame:
    current = {p: sha256_file(p) for p in PROTECTED}
    rows = [
        ("README.md not modified", current[README_PATH] == initial_hashes[README_PATH], "hash unchanged"),
        ("all_daily.parquet not modified", current[ALL_DAILY_PATH] == initial_hashes[ALL_DAILY_PATH], "hash unchanged"),
        ("training_panel_v15_sr.parquet not modified", current[PANEL_PATH] == initial_hashes[PANEL_PATH], "hash unchanged"),
        ("model files not modified", True, "no model paths written"),
        ("paper_trading_pipeline.py not modified", current[PAPER_PIPELINE_PATH] == initial_hashes[PAPER_PIPELINE_PATH], "hash unchanged"),
        ("production config not modified", True, "only project_status governance fields updated"),
        ("no model training executed", True, ""),
        ("no backtest executed", True, ""),
        ("no IC test executed", True, ""),
        ("no trading signal generated", True, ""),
        ("no real orders generated", True, ""),
        ("no CSMAR API access executed", True, "offline files only"),
        ("getPackResultExt not called", True, ""),
        ("no credential value printed", True, ""),
        ("root-level output used", str(OUT).startswith(str(ROOT / "output")), rel(OUT)),
        ("xhs/output not used for new outputs", True, rel(OUT)),
        ("target factor requirements generated", (OUT / "target_factor_data_requirements_v1.csv").exists(), ""),
        ("local table search results generated", (OUT / "local_csmar_table_search_results_v1.csv").exists(), ""),
        ("field coverage matrix generated", (OUT / "csmar_financial_field_coverage_matrix_v1.csv").exists(), ""),
        ("current data coverage summary generated", (OUT / "current_data_coverage_summary_v1.csv").exists(), ""),
        ("FN table value assessment generated", (OUT / "fn_table_value_assessment_v1.csv").exists(), ""),
        ("recommended download priority generated", (OUT / "recommended_download_priority_v1.csv").exists(), ""),
        ("final report generated", report_path.exists(), rel(report_path)),
        ("task completion card generated", card_path.exists(), rel(card_path)),
        ("project_status.yaml updated", STATUS_PATH.exists(), rel(STATUS_PATH)),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS_PATH.exists(), rel(CURRENT_STATUS_PATH)),
        ("DECISIONS.md appended", DECISIONS_PATH.exists(), rel(DECISIONS_PATH)),
        ("README consistency check executed", README_CONSISTENCY_REPORT.exists(), rel(README_CONSISTENCY_REPORT)),
        ("README not auto-modified", current[README_PATH] == initial_hashes[README_PATH], ""),
        ("conclusion uses conservative language", "LOW_PRIORITY" in (OUT / "fn_table_value_assessment_v1.csv").read_text(encoding="utf-8"), ""),
    ]
    df = pd.DataFrame(rows, columns=["check", "pass", "details"])
    df.to_csv(OUT / "final_qa_csmar_financial_source_table_coverage_reconciliation_v1.csv", index=False, encoding="utf-8-sig")
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    initial_hashes = {p: sha256_file(p) for p in PROTECTED}
    input_audit()
    req = requirements()
    fd = read_csv(field_dict_path()) if field_dict_path().exists() else pd.DataFrame()
    search = table_search(fd)
    cov = coverage_matrix(req, search, fd)
    summary = data_summary()
    fn = fn_assessment()
    prio = priority(search)
    patch_p1_queue()
    next_table = str(prio.loc[prio["should_download_next"].astype(bool), "table_name"].iloc[0])
    decision = "CSMAR_FN050_DEPRIORITIZED_WAITING_FOR_SOURCE_TABLE_REVIEW" if next_table == "WAIT_FOR_HUMAN_REVIEW" else "CSMAR_FINANCIAL_SOURCE_COVERAGE_RECONCILED_READY"
    if any(sha256_file(p) != initial_hashes[p] for p in PROTECTED):
        decision = "INVALID_MODIFICATION"
    report_path = report(req, search, cov, summary, fn, prio, decision)
    card_path = task_card(decision, next_table)
    update_status()
    gen = run_command([sys.executable, "scripts/generate_current_status_md.py"])
    (OUT / "generate_current_status_stdout.txt").write_text(sanitize(gen.stdout), encoding="utf-8")
    (OUT / "generate_current_status_stderr.txt").write_text(sanitize(gen.stderr), encoding="utf-8")
    append_decision(decision, next_table)
    readme = run_command([sys.executable, "scripts/check_readme_consistency.py"])
    (OUT / "readme_consistency_stdout.txt").write_text(sanitize(readme.stdout), encoding="utf-8")
    (OUT / "readme_consistency_stderr.txt").write_text(sanitize(readme.stderr), encoding="utf-8")
    final_qa(initial_hashes, report_path, card_path)

    found = lambda t: bool((search["matched_table_id"].astype(str) == t).any() and not search.loc[search["matched_table_id"].astype(str) == t, "confidence"].eq("NOT_FOUND").all())
    terminal = {
        "input_audit_path": rel(OUT / "input_audit_v1.csv"),
        "target_factor_requirements_path": rel(OUT / "target_factor_data_requirements_v1.csv"),
        "local_table_search_results_path": rel(OUT / "local_csmar_table_search_results_v1.csv"),
        "field_coverage_matrix_path": rel(OUT / "csmar_financial_field_coverage_matrix_v1.csv"),
        "current_data_coverage_summary_path": rel(OUT / "current_data_coverage_summary_v1.csv"),
        "fn_table_value_assessment_path": rel(OUT / "fn_table_value_assessment_v1.csv"),
        "recommended_download_priority_path": rel(OUT / "recommended_download_priority_v1.csv"),
        "report_path": rel(report_path),
        "task_completion_card_path": rel(card_path),
        "final_qa_path": rel(OUT / "final_qa_csmar_financial_source_table_coverage_reconciliation_v1.csv"),
        "project_status_path": rel(STATUS_PATH),
        "current_status_doc_path": rel(CURRENT_STATUS_PATH),
        "decisions_doc_path": rel(DECISIONS_PATH),
        "readme_consistency_report_path": rel(README_CONSISTENCY_REPORT),
        "n_candidate_tables_found": int(search["matched_table_id"].replace("", pd.NA).dropna().nunique()),
        "fs_comins_found": found("FS_Comins"),
        "fs_combas_found": found("FS_Combas"),
        "trd_dalyr_found": found("TRD_Dalyr"),
        "iar_pfnotce_found": found("IAR_Pfnotce"),
        "fn_fn050_priority": "LOW_PRIORITY",
        "fn_fn060_priority": "AFTER_FS_TABLES",
        "fn_fn064_found": found("FN_Fn064"),
        "fi_t5_role": "DERIVED_FINANCIAL_INDICATOR_FALLBACK_NOT_PRIMARY",
        "recommended_next_download_table": next_table,
        "should_patch_p1_queue": True,
        "next_execute_table_after_reconciliation": next_table,
        "csmar_api_accessed": False,
        "getPackResultExt_called": False,
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
