from __future__ import annotations

import gc
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


warnings.filterwarnings("ignore", message="Workbook contains no default style.*")


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "existing_fs_missing_feature_constructibility_patch_v0"

PREV_SUMMARY = ROOT / "output" / "compact_f_missing_feature_label_resolution_v0" / "compact_f_missing_feature_label_resolution_summary.json"
PREV_CONSTRUCT = ROOT / "output" / "compact_f_missing_feature_label_resolution_v0" / "missing_feature_constructibility.csv"
PREV_GAP = ROOT / "output" / "compact_f_missing_feature_label_resolution_v0" / "compact_f_v3_full_replication_gap.csv"
V3_PANEL = ROOT / "output" / "csmar_pit_clean_core_financial_factors_v3" / "pit_clean_core_financial_factors_monthly_v3.parquet"
TRANSFORMED_PANEL = ROOT / "output" / "transformed_training_panel_v0" / "transformed_training_panel_v0.parquet"

REQUIRED_INPUTS = [PREV_SUMMARY, PREV_CONSTRUCT, PREV_GAP, V3_PANEL, TRANSFORMED_PANEL]

USER_CONFIRMED_FIELDS = {
    "B001100000": "营业总收入",
    "B001101000": "营业收入",
    "B001200000": "营业总成本",
    "B001201000": "营业成本",
    "B001209000": "销售费用",
    "B001210000": "管理费用",
    "B001216000": "研发费用",
    "B001211000": "财务费用",
    "B001300000": "营业利润",
    "B001000000": "利润总额",
    "B002000000": "净利润",
    "B002000101": "归属于母公司所有者的净利润",
    "B003000000": "基本每股收益",
    "B004000000": "稀释每股收益",
    "A001100000": "流动资产合计",
    "A001200000": "非流动资产合计",
    "A001000000": "资产总计",
    "A002100000": "流动负债合计",
    "A002200000": "非流动负债合计",
    "A002000000": "负债合计",
    "A003101000": "实收资本或股本",
    "A003102000": "资本公积",
    "A003102101": "库存股",
    "A003103000": "盈余公积",
    "A0f3104000": "一般风险准备",
    "A003105000": "未分配利润",
    "A0F3109000": "专项储备",
    "A003111000": "其他综合收益",
    "A003100000": "归属于母公司所有者权益合计",
    "A003200000": "少数股东权益",
    "A003000000": "所有者权益合计",
}

REVIEW_FIELDS = [
    "B001100000",
    "B001101000",
    "B001300000",
    "B002000000",
    "B002000101",
    "B003000000",
    "B004000000",
    "A001100000",
    "A002100000",
    "A001000000",
    "A003100000",
    "A003000000",
]

SEARCH_ROOTS = [ROOT / "output", ROOT / "data" / "csmar_exports"]
SEARCH_KEYWORDS = [
    "FS_Comins",
    "利润表",
    "FS_Combas",
    "资产负债表",
    "pit",
    "standardized",
    "normalized",
    "FS_Comscfd",
]
INVENTORY_TERMS = ["存货", "inventories", "inventory", "A001123000"]
CFO_TERMS = ["经营活动产生的现金流量净额", "net_cash_flow_operating", "CFO", "C001000000", "operating_cash_flow"]
MAX_CANDIDATES = 250
MAX_TEXT_READ = 500_000


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def missing_input_report(missing: list[Path]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Missing Input Report",
        "",
        "The field availability patch did not run because required whitelisted inputs are missing.",
        "",
        "## Missing files",
        "",
    ]
    lines.extend(f"- {p.as_posix()}" for p in missing)
    (OUT_DIR / "missing_input_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        OUT_DIR / "existing_fs_missing_feature_constructibility_patch_summary.json",
        {"run_timestamp": now_iso(), "final_decision": "EXISTING_FS_FIELD_REVIEW_FAIL", "missing_inputs": [str(p) for p in missing]},
    )


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema.names


def filename_search() -> list[Path]:
    terms = [x.lower() for x in SEARCH_KEYWORDS]
    out: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if any(term.lower() in name for term in terms):
                out.append(path)
                if len(out) >= MAX_CANDIDATES:
                    return out
    return out


def inspect_candidate(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    columns: list[str] = []
    sample_text = ""
    inspect_note = ""
    try:
        if suffix == ".parquet":
            columns = pq.ParquetFile(path).schema.names
        elif suffix == ".csv":
            columns = list(pd.read_csv(path, nrows=0).columns)
        elif suffix in {".txt", ".md", ".json"}:
            sample_text = path.read_text(encoding="utf-8", errors="replace")[:MAX_TEXT_READ]
            columns = extract_known_terms(sample_text)
        elif suffix in {".xlsx", ".xls"}:
            # Read only workbook metadata/header row through pandas nrows=0; do not load full sheets.
            try:
                xl = pd.ExcelFile(path)
                first_sheet = xl.sheet_names[0]
                columns = list(pd.read_excel(path, sheet_name=first_sheet, nrows=0).columns)
                inspect_note = f"header_only_sheet={first_sheet}"
            except Exception as exc:
                inspect_note = f"excel_header_unreadable:{type(exc).__name__}"
        else:
            inspect_note = "unsupported_file_type_for_schema"
    except Exception as exc:
        inspect_note = f"inspect_error:{type(exc).__name__}"
    haystack = " ".join(columns) + " " + sample_text
    return {
        "path": str(path.relative_to(ROOT)),
        "suffix": suffix,
        "size_bytes": path.stat().st_size,
        "columns_or_terms": ";".join(columns[:120]),
        "matched_review_fields": ";".join(sorted(field for field in REVIEW_FIELDS if field.lower() in haystack.lower())),
        "inventory_terms_found": ";".join(sorted(term for term in INVENTORY_TERMS if term.lower() in haystack.lower())),
        "operating_cash_flow_terms_found": ";".join(sorted(term for term in CFO_TERMS if term.lower() in haystack.lower())),
        "inspect_note": inspect_note,
    }


def extract_known_terms(text: str) -> list[str]:
    terms = list(USER_CONFIRMED_FIELDS.keys()) + list(USER_CONFIRMED_FIELDS.values()) + INVENTORY_TERMS + CFO_TERMS
    low = text.lower()
    return sorted({term for term in terms if term.lower() in low})


def field_present(field: str, v3_cols: set[str], transformed_cols: set[str], candidate_rows: list[dict[str, Any]]) -> tuple[bool, str]:
    low = field.lower()
    locations = []
    if any(low == c.lower() or low in c.lower() for c in v3_cols):
        locations.append("v3_schema")
    if any(low == c.lower() or low in c.lower() for c in transformed_cols):
        locations.append("transformed_schema")
    for row in candidate_rows:
        if low in str(row["matched_review_fields"]).lower() or low in str(row["columns_or_terms"]).lower():
            locations.append(row["path"])
    if field in USER_CONFIRMED_FIELDS:
        locations.append("user_confirmed_existing_fs_field")
    return bool(locations), ";".join(dict.fromkeys(locations))


def term_found(terms: list[str], v3_cols: set[str], transformed_cols: set[str], candidate_rows: list[dict[str, Any]]) -> tuple[bool, str]:
    locations = []
    for term in terms:
        low = term.lower()
        if any(low in c.lower() for c in v3_cols):
            locations.append(f"v3_schema:{term}")
        if any(low in c.lower() for c in transformed_cols):
            locations.append(f"transformed_schema:{term}")
        for row in candidate_rows:
            row_text = " ".join(str(row.get(k, "")) for k in ["columns_or_terms", "inventory_terms_found", "operating_cash_flow_terms_found"])
            if low in row_text.lower():
                locations.append(f"{row['path']}:{term}")
    return bool(locations), ";".join(dict.fromkeys(locations))


def make_availability(v3_cols: set[str], transformed_cols: set[str], candidate_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for field in REVIEW_FIELDS:
        present, locations = field_present(field, v3_cols, transformed_cols, candidate_rows)
        rows.append(
            {
                "field_code_or_term": field,
                "field_name": USER_CONFIRMED_FIELDS.get(field, ""),
                "field_group": "income_statement" if field.startswith("B") else "balance_sheet",
                "available": present,
                "locations": locations,
                "note": "User confirmed field exists in FS source; may require cleaning into v3/transformed panel." if "user_confirmed" in locations else "",
            }
        )
    inv_found, inv_locations = term_found(INVENTORY_TERMS, v3_cols, transformed_cols, candidate_rows)
    rows.append(
        {
            "field_code_or_term": "inventory / A001123000",
            "field_name": "存货",
            "field_group": "balance_sheet_detail",
            "available": inv_found,
            "locations": inv_locations,
            "note": "Needed for Quick_Ratio.",
        }
    )
    cfo_found, cfo_locations = term_found(CFO_TERMS, v3_cols, transformed_cols, candidate_rows)
    rows.append(
        {
            "field_code_or_term": "operating_cash_flow / C001000000",
            "field_name": "经营活动产生的现金流量净额",
            "field_group": "cash_flow_statement",
            "available": cfo_found,
            "locations": cfo_locations,
            "note": "Needed for CFO_to_Earnings.",
        }
    )
    return pd.DataFrame(rows)


def availability_map(avail: pd.DataFrame) -> dict[str, bool]:
    return dict(zip(avail["field_code_or_term"].astype(str), avail["available"].astype(bool)))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        missing_input_report(missing)
        print(f"Missing required inputs: {len(missing)}")
        return 0

    prev_summary = load_json(PREV_SUMMARY)
    prev_construct = pd.read_csv(PREV_CONSTRUCT)
    v3_cols = set(parquet_columns(V3_PANEL))
    transformed_cols = set(parquet_columns(TRANSFORMED_PANEL))

    candidate_paths = filename_search()
    candidate_rows = [inspect_candidate(path) for path in candidate_paths]
    # Keep the candidate audit inside the report directory for traceability without making it a required deliverable.
    pd.DataFrame(candidate_rows).to_csv(OUT_DIR / "fs_source_file_candidates_audit.csv", index=False)

    availability = make_availability(v3_cols, transformed_cols, candidate_rows)
    availability.to_csv(OUT_DIR / "fs_field_availability_check.csv", index=False)
    amap = availability_map(availability)

    current_ratio = bool(amap.get("A001100000") and amap.get("A002100000"))
    inventory_found = bool(amap.get("inventory / A001123000"))
    quick_ratio = bool(current_ratio and inventory_found)
    eps_yoy = bool(amap.get("B003000000") or amap.get("B004000000"))
    equity_multiplier = bool(amap.get("A001000000") and (amap.get("A003100000") or amap.get("A003000000")))
    operating_margin = bool(amap.get("B001300000") and (amap.get("B001101000") or amap.get("B001100000")))
    operating_cash_flow_found = bool(amap.get("operating_cash_flow / C001000000"))
    cfo_to_earnings = bool(operating_cash_flow_found and (amap.get("B002000101") or amap.get("B002000000")))

    reclass_rows = [
        {
            "compact_f_feature": "Current_Ratio_neutral_z_rank",
            "base_feature": "Current_Ratio",
            "new_constructibility_status": "CONSTRUCTIBLE_FROM_EXISTING_FS_FIELDS" if current_ratio else "REQUIRES_BALANCE_SHEET_DETAIL",
            "proposed_action": "ADD_DERIVED_FEATURE_IN_FUTURE_PANEL" if current_ratio else "REQUIRE_ADDITIONAL_DATA_BEFORE_FULL_REPLICATION",
            "reason": "A001100000 and A002100000 are available." if current_ratio else "Current assets/current liabilities unavailable.",
            "can_include_in_derived_candidate_panel": current_ratio,
        },
        {
            "compact_f_feature": "Quick_Ratio_neutral_z_rank",
            "base_feature": "Quick_Ratio",
            "new_constructibility_status": "CONSTRUCTIBLE_FROM_EXISTING_FS_FIELDS" if quick_ratio else "NEEDS_INVENTORY_FIELD",
            "proposed_action": "ADD_DERIVED_FEATURE_IN_FUTURE_PANEL" if quick_ratio else "REQUIRE_ADDITIONAL_DATA_BEFORE_FULL_REPLICATION",
            "reason": "Current assets, inventory, and current liabilities are available." if quick_ratio else "Inventory field was not confirmed in available schemas/candidates.",
            "can_include_in_derived_candidate_panel": quick_ratio,
        },
        {
            "compact_f_feature": "EPS_YoY_neutral_z_rank",
            "base_feature": "EPS_YoY",
            "new_constructibility_status": "CONSTRUCTIBLE_FROM_EXISTING_FS_FIELDS_WITH_NOTE" if eps_yoy else "REQUIRES_EPS_OR_SHARE_BASE",
            "proposed_action": "ADD_DERIVED_FEATURE_IN_FUTURE_PANEL" if eps_yoy else "REQUIRE_ADDITIONAL_DATA_BEFORE_FULL_REPLICATION",
            "reason": "B003000000 basic EPS is available; use lag4 with robust handling of negative/small denominators." if eps_yoy else "EPS fields unavailable.",
            "can_include_in_derived_candidate_panel": eps_yoy,
        },
        {
            "compact_f_feature": "Equity_Multiplier_neutral_z_rank",
            "base_feature": "Equity_Multiplier",
            "new_constructibility_status": "CONSTRUCTIBLE_FROM_EXISTING_FS_FIELDS" if equity_multiplier else "NOT_AVAILABLE_CURRENTLY",
            "proposed_action": "ADD_DERIVED_FEATURE_IN_FUTURE_PANEL" if equity_multiplier else "REQUIRE_ADDITIONAL_DATA_BEFORE_FULL_REPLICATION",
            "reason": "A001000000 and parent/total equity are available; prefer parent equity and record scope." if equity_multiplier else "Assets/equity fields unavailable.",
            "can_include_in_derived_candidate_panel": equity_multiplier,
        },
        {
            "compact_f_feature": "Operating_Margin_neutral_z_rank",
            "base_feature": "Operating_Margin",
            "new_constructibility_status": "CONSTRUCTIBLE_FROM_EXISTING_FS_FIELDS" if operating_margin else "REQUIRES_OPERATING_PROFIT_FIELD",
            "proposed_action": "ADD_DERIVED_FEATURE_IN_FUTURE_PANEL" if operating_margin else "REQUIRE_ADDITIONAL_DATA_BEFORE_FULL_REPLICATION",
            "reason": "B001300000 and B001101000/B001100000 are available; prefer operating revenue denominator." if operating_margin else "Operating profit/revenue fields unavailable.",
            "can_include_in_derived_candidate_panel": operating_margin,
        },
        {
            "compact_f_feature": "CFO_to_Earnings_neutral_z_rank",
            "base_feature": "CFO_to_Earnings",
            "new_constructibility_status": "CONSTRUCTIBLE_FROM_EXISTING_FS_FIELDS" if cfo_to_earnings else "NEEDS_CASH_FLOW_DIRECT_TABLE",
            "proposed_action": "ADD_DERIVED_FEATURE_IN_FUTURE_PANEL" if cfo_to_earnings else "REQUIRE_ADDITIONAL_DATA_BEFORE_FULL_REPLICATION",
            "reason": "Operating cash flow and earnings are available." if cfo_to_earnings else "Operating cash flow field was not confirmed; earnings fields alone are insufficient.",
            "can_include_in_derived_candidate_panel": cfo_to_earnings,
        },
    ]
    reclass = pd.DataFrame(reclass_rows)
    reclass.to_csv(OUT_DIR / "compact_f_missing_feature_reclassified.csv", index=False)

    formula_rows = [
        {
            "derived_feature": "Current_Ratio",
            "formula": "A001100000 / A002100000",
            "field_scope_note": "流动资产合计 / 流动负债合计",
            "candidate_transform": "clip + rank + z; robust denominator guard",
        },
        {
            "derived_feature": "Quick_Ratio",
            "formula": "(A001100000 - inventory) / A002100000",
            "field_scope_note": "Requires inventory field such as A001123000; not confirmed in current review.",
            "candidate_transform": "clip + rank + z; robust denominator guard",
        },
        {
            "derived_feature": "EPS_YoY",
            "formula": "B003000000 / lag4(B003000000) - 1; fallback B004000000 diluted EPS",
            "field_scope_note": "Handle negative EPS, small denominator, stock splits/share-base changes; candidate only.",
            "candidate_transform": "robust winsor/clip + rank + z",
        },
        {
            "derived_feature": "Equity_Multiplier_parent",
            "formula": "A001000000 / A003100000",
            "field_scope_note": "Prefer parent equity scope; alternative total equity A001000000 / A003000000.",
            "candidate_transform": "clip + rank + z",
        },
        {
            "derived_feature": "Operating_Margin",
            "formula": "B001300000 / B001101000; fallback B001300000 / B001100000",
            "field_scope_note": "Fallback total operating revenue includes broader financial income and should be flagged.",
            "candidate_transform": "clip + rank + z",
        },
        {
            "derived_feature": "CFO_to_Earnings",
            "formula": "operating_cash_flow / B002000101; fallback / B002000000",
            "field_scope_note": "Requires cash flow direct table field such as C001000000; not confirmed in current review.",
            "candidate_transform": "clip + rank + z; denominator guard",
        },
    ]
    pd.DataFrame(formula_rows).to_csv(OUT_DIR / "derived_feature_formula_candidates.csv", index=False)

    additional = reclass[~reclass["can_include_in_derived_candidate_panel"]].copy()
    additional["needed_data"] = additional["new_constructibility_status"].map(
        {
            "NEEDS_INVENTORY_FIELD": "inventory field, likely balance sheet detail",
            "NEEDS_CASH_FLOW_DIRECT_TABLE": "operating cash flow field from cash flow statement/direct table",
            "REQUIRES_BALANCE_SHEET_DETAIL": "balance sheet detail",
            "REQUIRES_EPS_OR_SHARE_BASE": "EPS/share-base fields",
            "REQUIRES_OPERATING_PROFIT_FIELD": "operating profit field",
            "NOT_AVAILABLE_CURRENTLY": "manual review",
        }
    )
    additional.to_csv(OUT_DIR / "additional_data_needed.csv", index=False)

    reclassified_constructible_count = int(reclass["can_include_in_derived_candidate_panel"].sum())
    still_requires_count = int((~reclass["can_include_in_derived_candidate_panel"]).sum())
    if reclassified_constructible_count == 6:
        final_decision = "EXISTING_FS_FIELDS_RECOVER_ALL_NON_CFO_MISSING_FEATURES"
        recommended_next_step = "Build Derived Compact-F Missing Features Candidate Panel v0, including Quick_Ratio and CFO_to_Earnings after confirming inventory and cash-flow source joins."
    elif reclassified_constructible_count >= 4 and not quick_ratio and not cfo_to_earnings:
        final_decision = "EXISTING_FS_FIELDS_CAN_RECOVER_PARTIAL_COMPACT_F_MISSING_FEATURES"
        recommended_next_step = "Build Derived Compact-F Missing Features Candidate Panel v0; download/add cash flow direct table and inventory field before Compact-F-v3-full replication."
    elif reclassified_constructible_count >= 5 and not cfo_to_earnings:
        final_decision = "EXISTING_FS_FIELDS_RECOVER_ALL_NON_CFO_MISSING_FEATURES"
        recommended_next_step = "Build Derived Compact-F Missing Features Candidate Panel v0, then add cash flow table for CFO_to_Earnings."
    elif reclassified_constructible_count > 0:
        final_decision = "EXISTING_FS_FIELD_REVIEW_WATCH_INPUT_NEEDED"
        recommended_next_step = "Confirm missing inventory/cash-flow fields and then build a derived candidate panel."
    else:
        final_decision = "EXISTING_FS_FIELD_REVIEW_FAIL"
        recommended_next_step = "Provide verified FS intermediate field locations before derived feature work."

    summary = {
        "run_timestamp": now_iso(),
        "previous_missing_feature_count": int(prev_summary.get("missing_feature_count", len(prev_construct[prev_construct["constructibility_status"].ne("ALREADY_MAPPED")]))),
        "reclassified_constructible_count": reclassified_constructible_count,
        "still_requires_additional_data_count": still_requires_count,
        "current_ratio_constructible": current_ratio,
        "quick_ratio_constructible": quick_ratio,
        "eps_yoy_constructible": eps_yoy,
        "equity_multiplier_constructible": equity_multiplier,
        "operating_margin_constructible": operating_margin,
        "cfo_to_earnings_constructible": cfo_to_earnings,
        "inventory_field_found": inventory_found,
        "operating_cash_flow_field_found": operating_cash_flow_found,
        "production_modified": False,
        "v3_modified": False,
        "transformed_panel_modified": False,
        "training_run": False,
        "backtest_run": False,
        "ic_calculated": False,
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    write_json(OUT_DIR / "existing_fs_missing_feature_constructibility_patch_summary.json", summary)

    report = [
        "# Existing FS Field Availability & Missing Compact-F Feature Constructibility Patch v0",
        "",
        "## 1. Scope",
        "",
        "This run only reviews existing FS field availability and reclassifies missing Compact-F feature constructibility. It does not train, backtest, calculate IC, modify production, modify v3, or modify the transformed panel.",
        "",
        "## 2. Field Availability",
        "",
        f"- Review fields checked: {len(REVIEW_FIELDS)} plus inventory and operating cash flow terms.",
        f"- Inventory field found: {inventory_found}",
        f"- Operating cash flow field found: {operating_cash_flow_found}",
        "",
        "## 3. Missing Feature Reclassification",
        "",
        f"- Reclassified constructible: {reclassified_constructible_count}",
        f"- Still requires additional data: {still_requires_count}",
        "",
        "## 4. Derived Formula Candidates",
        "",
        "See derived_feature_formula_candidates.csv.",
        "",
        "## 5. Additional Data Needed",
        "",
        "See additional_data_needed.csv.",
        "",
        "## 6. Decision",
        "",
        final_decision,
        "",
        "## 7. Recommended Next Step",
        "",
        recommended_next_step,
        "",
    ]
    (OUT_DIR / "existing_fs_missing_feature_constructibility_patch_report.md").write_text("\n".join(report), encoding="utf-8")

    task_card = [
        "# Task Completion Card",
        "",
        "- task_name: Existing FS Field Availability & Missing Compact-F Feature Constructibility Patch v0",
        f"- completed_at: {now_iso()}",
        f"- final_decision: {final_decision}",
        f"- output_dir: {OUT_DIR.relative_to(ROOT).as_posix()}",
    ]
    (OUT_DIR / "task_completion_card.md").write_text("\n".join(task_card) + "\n", encoding="utf-8")
    write_json(
        OUT_DIR / "terminal_summary.json",
        {
            "script": "scripts/patch_existing_fs_missing_feature_constructibility_v0.py",
            "status": "completed",
            "stdout_log": "output/_agent_runs/existing_fs_missing_feature_constructibility_patch_v0/run_stdout.txt",
            "stderr_log": "output/_agent_runs/existing_fs_missing_feature_constructibility_patch_v0/run_stderr.txt",
            "final_decision": final_decision,
        },
    )
    pd.DataFrame([summary]).to_csv(OUT_DIR / "final_qa.csv", index=False)

    del prev_construct, availability, reclass, additional
    gc.collect()
    print(f"final_decision={final_decision}")
    print(f"output_dir={OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
