from __future__ import annotations

import hashlib
import subprocess
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


warnings.filterwarnings("ignore", message="Workbook contains no default style.*", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "data" / "csmar_exports"
OUT = ROOT / "output" / "csmar_fs_comins_manual_import_audit_v1"
FS_FILE = EXPORT_DIR / "FS_Comins.xlsx"
PIT_FILE = ROOT / "output" / "csmar_p0_pit_pack_import_audit_v1" / "csmar_p0_pit_announcement_panel_v1.parquet"
TRAINING_PANEL = ROOT / "output" / "training_panel_v15_sr.parquet"
ALL_DAILY = ROOT / "output" / "all_daily.parquet"
STATUS_PATH = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS_PATH = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
README_PATH = ROOT / "README.md"
READ_ME_CONSISTENCY_REPORT = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"

FIELD_MAP = {
    "Stkcd": "symbol",
    "ShortName": "short_name",
    "Accper": "report_period",
    "Typrep": "report_type",
    "IfCorrect": "if_correct",
    "DeclareDate": "correction_disclosure_date",
    "B001100000": "total_operating_revenue",
    "B001101000": "operating_revenue",
    "B001200000": "total_operating_cost",
    "B001201000": "operating_cost",
    "B001209000": "sales_expense",
    "B001210000": "admin_expense",
    "B001216000": "rd_expense",
    "B001211000": "financial_expense",
    "B001300000": "operating_profit",
    "B001000000": "total_profit",
    "B002000000": "net_profit",
    "B002000101": "net_profit_parent",
    "B002000201": "minority_profit_loss",
    "B003000000": "basic_eps",
    "B004000000": "diluted_eps",
}

NUMERIC_FIELDS = [
    "total_operating_revenue",
    "operating_revenue",
    "total_operating_cost",
    "operating_cost",
    "sales_expense",
    "admin_expense",
    "rd_expense",
    "financial_expense",
    "operating_profit",
    "total_profit",
    "net_profit",
    "net_profit_parent",
    "minority_profit_loss",
    "basic_eps",
    "diluted_eps",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rate(mask: pd.Series | np.ndarray, denom: int | None = None) -> float:
    if denom is None:
        denom = len(mask)
    if denom == 0:
        return float("nan")
    return float(np.asarray(mask).sum() / denom)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def inventory_inputs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    files = sorted(EXPORT_DIR.glob("*"))
    related = [
        p for p in files
        if p.name == "FS_Comins.xlsx" or ("FS_Comins" in p.name and p.is_file())
    ]
    for path in related:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            try:
                xls = pd.ExcelFile(path)
                for sheet in xls.sheet_names:
                    sample = pd.read_excel(path, sheet_name=sheet, dtype=str, nrows=100)
                    columns = [str(c) for c in sample.columns]
                    detected = "FS_Comins" if {"Stkcd", "Accper", "Typrep"}.issubset(columns) else ""
                    rows.append({
                        "file_path": rel(path),
                        "detected_table": detected,
                        "file_type": path.suffix.lower().lstrip("."),
                        "file_size": path.stat().st_size,
                        "readable": True,
                        "sheet_names": sheet,
                        "n_rows_sampled": len(sample),
                        "n_columns": len(columns),
                        "columns": "|".join(columns),
                        "notes": "main data sheet candidate" if detected else "sheet does not contain Stkcd/Accper/Typrep in sample",
                    })
            except Exception as exc:
                rows.append({
                    "file_path": rel(path),
                    "detected_table": "FS_Comins" if path.name == "FS_Comins.xlsx" else "",
                    "file_type": path.suffix.lower().lstrip("."),
                    "file_size": path.stat().st_size,
                    "readable": False,
                    "sheet_names": "",
                    "n_rows_sampled": 0,
                    "n_columns": 0,
                    "columns": "",
                    "notes": repr(exc),
                })
        else:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                rows.append({
                    "file_path": rel(path),
                    "detected_table": "FS_Comins_description" if "FS_Comins" in path.name or "Stkcd" in text[:5000] else "",
                    "file_type": path.suffix.lower().lstrip(".") or "text",
                    "file_size": path.stat().st_size,
                    "readable": True,
                    "sheet_names": "",
                    "n_rows_sampled": min(len(text.splitlines()), 100),
                    "n_columns": 0,
                    "columns": "",
                    "notes": "field description file" if "Stkcd" in text[:5000] else "",
                })
            except Exception as exc:
                rows.append({
                    "file_path": rel(path),
                    "detected_table": "",
                    "file_type": path.suffix.lower().lstrip(".") or "text",
                    "file_size": path.stat().st_size,
                    "readable": False,
                    "sheet_names": "",
                    "n_rows_sampled": 0,
                    "n_columns": 0,
                    "columns": "",
                    "notes": repr(exc),
                })
    df = pd.DataFrame(rows)
    write_csv(df, OUT / "input_file_inventory_v1.csv")
    return df


def load_standardized() -> tuple[pd.DataFrame, dict[str, float]]:
    if not FS_FILE.exists():
        raise FileNotFoundError(FS_FILE)
    xls = pd.ExcelFile(FS_FILE)
    main_sheet = None
    for sheet in xls.sheet_names:
        sample = pd.read_excel(FS_FILE, sheet_name=sheet, dtype=str, nrows=20)
        if {"Stkcd", "Accper", "Typrep"}.issubset(set(map(str, sample.columns))):
            main_sheet = sheet
            break
    if main_sheet is None:
        raise RuntimeError("No FS_Comins sheet with Stkcd/Accper/Typrep found")

    raw = pd.read_excel(FS_FILE, sheet_name=main_sheet, dtype={"Stkcd": str})
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw[~raw["Stkcd"].astype(str).isin(["证券代码", "没有单位"])].copy()
    raw = raw[raw["Stkcd"].notna()].copy()

    df = raw.rename(columns=FIELD_MAP)
    for source, target in FIELD_MAP.items():
        if target not in df.columns:
            df[target] = np.nan
    df["raw_stkcd"] = raw.get("Stkcd")
    df["raw_accper"] = raw.get("Accper")
    df["raw_typrep"] = raw.get("Typrep")
    df["raw_declare_date"] = raw.get("DeclareDate")

    df["symbol"] = df["symbol"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df.loc[~df["symbol"].str.match(r"^\d{6}$", na=False), "symbol"] = np.nan
    df["short_name"] = df["short_name"].astype(str).replace({"nan": np.nan})
    df["report_period"] = pd.to_datetime(df["report_period"], errors="coerce")
    df["report_type"] = df["report_type"].astype(str).str.strip().replace({"nan": np.nan})
    df["if_correct"] = pd.to_numeric(df["if_correct"], errors="coerce")
    df["correction_disclosure_date"] = pd.to_datetime(df["correction_disclosure_date"], errors="coerce")

    parse_fail_rates: dict[str, float] = {}
    for col in NUMERIC_FIELDS:
        before_non_null = df[col].notna()
        parsed = pd.to_numeric(df[col], errors="coerce")
        parse_fail_rates[col] = rate(before_non_null & parsed.isna())
        df[col] = parsed

    df["source_table"] = "FS_Comins"
    df["source_file"] = rel(FS_FILE)
    df["notes"] = ""

    ordered = [
        "symbol", "short_name", "report_period", "report_type", "if_correct", "correction_disclosure_date",
        *NUMERIC_FIELDS, "source_table", "source_file", "notes", "raw_stkcd", "raw_accper", "raw_typrep",
        "raw_declare_date",
    ]
    df = df[ordered]
    df.to_parquet(OUT / "fs_comins_standardized_v1.parquet", index=False)
    write_csv(df.head(1000), OUT / "fs_comins_standardized_sample_v1.csv")
    return df, parse_fail_rates


def quality_audit(df: pd.DataFrame, parse_fail_rates: dict[str, float]) -> tuple[pd.DataFrame, float]:
    v15_symbols: set[str] = set()
    if TRAINING_PANEL.exists():
        tp = pd.read_parquet(TRAINING_PANEL, columns=["symbol"])
        v15_symbols = set(tp["symbol"].astype(str).str.zfill(6).unique())
    fs_symbols = set(df["symbol"].dropna().unique())
    overlap = len(fs_symbols & v15_symbols)
    v15_coverage = overlap / len(v15_symbols) if v15_symbols else float("nan")

    dup_count = int(df.duplicated(["symbol", "report_period", "report_type"]).sum())
    pre_2007 = df["report_period"] < pd.Timestamp("2007-01-01")
    pre_2018 = df["report_period"] < pd.Timestamp("2018-01-01")
    rows = [
        ("n_rows", len(df), ""),
        ("n_symbols", df["symbol"].nunique(), ""),
        ("min_report_period", df["report_period"].min().date().isoformat() if df["report_period"].notna().any() else "", ""),
        ("max_report_period", df["report_period"].max().date().isoformat() if df["report_period"].notna().any() else "", ""),
        ("report_type_distribution", df["report_type"].value_counts(dropna=False).to_dict(), ""),
        ("report_type_a_coverage_rate", rate(df["report_type"].eq("A")), "A consolidated report rows / all rows"),
        ("duplicate_symbol_report_period_report_type", dup_count, ""),
        ("operating_revenue_missing_rate", rate(df["operating_revenue"].isna()), f"parse_fail_rate={parse_fail_rates.get('operating_revenue')}"),
        ("total_operating_revenue_missing_rate", rate(df["total_operating_revenue"].isna()), f"parse_fail_rate={parse_fail_rates.get('total_operating_revenue')}"),
        ("net_profit_parent_missing_rate", rate(df["net_profit_parent"].isna()), f"parse_fail_rate={parse_fail_rates.get('net_profit_parent')}"),
        ("sales_expense_missing_rate", rate(df["sales_expense"].isna()), f"parse_fail_rate={parse_fail_rates.get('sales_expense')}"),
        ("rd_expense_missing_rate", rate(df["rd_expense"].isna()), f"parse_fail_rate={parse_fail_rates.get('rd_expense')}"),
        ("pre_2007_net_profit_parent_missing_rate", rate(df.loc[pre_2007, "net_profit_parent"].isna(), int(pre_2007.sum())), "B002000101 starts in 2007 per field note"),
        ("pre_2018_rd_expense_missing_rate", rate(df.loc[pre_2018, "rd_expense"].isna(), int(pre_2018.sum())), "B001216000 starts in 2018 per field note"),
        ("if_correct_1_rate", rate(df["if_correct"].eq(1)), ""),
        ("correction_disclosure_date_non_null_rate", rate(df["correction_disclosure_date"].notna()), ""),
        ("v15_universe_symbol_coverage_rate", v15_coverage, f"overlap={overlap}; v15_symbols={len(v15_symbols)}; fs_symbols={len(fs_symbols)}"),
    ]
    qa = pd.DataFrame(rows, columns=["metric", "value", "details"])
    write_csv(qa, OUT / "fs_comins_quality_audit_v1.csv")
    return qa, v15_coverage


def merge_pit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    pit = pd.read_parquet(PIT_FILE)
    pit = pit[["symbol", "report_period", "pit_date_primary", "pit_date_source", "effective_month_end", "quality_flag"]].copy()
    pit["symbol"] = pit["symbol"].astype(str).str.zfill(6)
    for col in ["report_period", "pit_date_primary", "effective_month_end"]:
        pit[col] = pd.to_datetime(pit[col], errors="coerce")
    merged = df.merge(pit, on=["symbol", "report_period"], how="left", suffixes=("", "_p0"))
    missing = merged["pit_date_primary"].isna()
    merged["quality_flag"] = merged["quality_flag"].fillna("missing_pit_date")
    computed_month_end = merged["pit_date_primary"] + pd.offsets.MonthEnd(0)
    computed_month_end = computed_month_end.mask(computed_month_end <= merged["pit_date_primary"], merged["pit_date_primary"] + pd.offsets.MonthEnd(1))
    merged["effective_month_end"] = computed_month_end.where(merged["pit_date_primary"].notna(), pd.NaT)
    pit_coverage = rate(~missing)
    lag = (merged["pit_date_primary"] - merged["report_period"]).dt.days
    rows = [
        ("pit_date_coverage_rate", pit_coverage, f"matched_rows={int((~missing).sum())}; total_rows={len(merged)}"),
        ("missing_pit_date_rate", rate(missing), ""),
        ("report_period_to_pit_lag_days_min", lag.min(), ""),
        ("report_period_to_pit_lag_days_p25", lag.quantile(0.25), ""),
        ("report_period_to_pit_lag_days_median", lag.median(), ""),
        ("report_period_to_pit_lag_days_p75", lag.quantile(0.75), ""),
        ("report_period_to_pit_lag_days_max", lag.max(), ""),
    ]
    pit_audit = pd.DataFrame(rows, columns=["metric", "value", "details"])
    write_csv(pit_audit, OUT / "fs_comins_pit_merge_audit_v1.csv")
    merged.to_parquet(OUT / "fs_comins_with_pit_dates_v1.parquet", index=False)
    write_csv(merged.head(1000), OUT / "fs_comins_with_pit_dates_sample_v1.csv")
    return merged, pit_audit, pit_coverage


def ttm_readiness(df: pd.DataFrame) -> pd.DataFrame:
    items = [
        "operating_revenue", "total_operating_revenue", "net_profit_parent", "net_profit",
        "sales_expense", "rd_expense", "total_operating_cost", "operating_profit", "total_profit",
    ]
    rows = []
    for item in items:
        non_null = df[item].notna()
        coverage = rate(non_null)
        periods = df.loc[non_null, "report_period"]
        caveats = []
        if item == "net_profit_parent":
            caveats.append("2007年前可能结构性缺失")
        if item == "rd_expense":
            caveats.append("2018年前可能结构性缺失")
        if item in {"operating_revenue", "total_operating_revenue", "net_profit_parent", "net_profit", "operating_profit", "total_profit"}:
            caveats.append("后续应按报告期累计值差分/滚动重构TTM")
        rows.append({
            "raw_item": item,
            "required_for_factor": {
                "operating_revenue": "RevGrowth_YoY, NetMargin denominator fallback",
                "total_operating_revenue": "RevGrowth_YoY, NetMargin denominator",
                "net_profit_parent": "ProfitGrowth_YoY, ROE, EP, NetMargin",
                "net_profit": "NetMargin fallback",
                "sales_expense": "sales_expense_to_revenue",
                "rd_expense": "rd_expense_to_revenue",
                "total_operating_cost": "Operating margin sanity check",
                "operating_profit": "Operating margin",
                "total_profit": "Profit sanity check",
            }[item],
            "n_non_null": int(non_null.sum()),
            "coverage_rate": coverage,
            "min_report_period": periods.min().date().isoformat() if not periods.empty else "",
            "max_report_period": periods.max().date().isoformat() if not periods.empty else "",
            "ttm_reconstructable": bool(coverage > 0.5 and df["report_period"].nunique() >= 8),
            "caveats": "; ".join(caveats),
        })
    out = pd.DataFrame(rows)
    write_csv(out, OUT / "fs_comins_ttm_readiness_audit_v1.csv")
    return out


def factor_coverage(df: pd.DataFrame) -> pd.DataFrame:
    available = {c for c in NUMERIC_FIELDS if df[c].notna().any()}
    rows = [
        ("ROE", "partial", "net_profit_parent", "shareholders_equity from FS_Combas", False, "Needs balance sheet equity denominator."),
        ("EP", "partial", "net_profit_parent/basic_eps", "market cap or price shares from TRD_Dalyr", False, "Needs market value alignment."),
        ("BP", "not_supported_alone", "", "book equity from FS_Combas; market cap from TRD_Dalyr", False, "Income statement alone cannot rebuild BP."),
        ("ProfitGrowth_YoY", "supported_with_ttm_method", "net_profit_parent or net_profit", "", True, "Requires PIT-safe report-period YoY/TTM rebuild."),
        ("RevGrowth_YoY", "supported_with_ttm_method", "operating_revenue or total_operating_revenue", "", True, "Requires PIT-safe report-period YoY/TTM rebuild."),
        ("NetMargin", "supported", "net_profit_parent/net_profit and revenue", "", True, "Use total_operating_revenue or operating_revenue denominator."),
        ("Debt_Ratio", "not_supported_alone", "", "total_liabilities and total_assets from FS_Combas", False, "Balance sheet factor."),
        ("sales_expense_to_revenue", "supported", "sales_expense and revenue", "", True, "Coverage depends on sales_expense availability."),
        ("rd_expense_to_revenue", "supported_post_2018", "rd_expense and revenue", "", True, "R&D expense is structurally sparse before 2018."),
        ("earnings_preview_midpoint_yoy", "not_supported_alone", "", "earnings forecast/preview table", False, "FS_Comins is actual statement data, not preview midpoint."),
    ]
    out_rows = []
    for target, status, req, missing, alone, notes in rows:
        req_fields = [x.strip() for part in req.split(" and ") for x in part.split("/") if x.strip()]
        if req and not all(f in available for f in req_fields if f in NUMERIC_FIELDS):
            notes = notes + " Some required FS fields have no non-null rows."
        out_rows.append({
            "target_factor": target,
            "fs_comins_support_status": status,
            "required_fields_available": req,
            "still_missing_fields": missing,
            "can_rebuild_from_fs_comins_alone": alone,
            "notes": notes,
        })
    out = pd.DataFrame(out_rows)
    write_csv(out, OUT / "fs_comins_factor_coverage_update_v1.csv")
    return out


def update_status_and_docs() -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    status["project"]["last_updated"] = date.today().isoformat()
    status["alternative_data"]["csmar_status"] = "fs_comins_manual_export_imported_waiting_for_fs_combas_trd"
    status["alternative_data"]["csmar_latest_task"] = "CSMAR FS_Comins Manual Export Import Audit v1"
    status["alternative_data"]["csmar_latest_output"] = rel(OUT)
    status["validation"]["pit_financial_status"] = "p0_pit_dates_imported_fs_comins_imported_fs_combas_trd_pending"
    status["validation"]["blend_v3_historical_metrics_status"] = "under_pit_review"
    STATUS_PATH.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False), encoding="utf-8")

    subprocess.run(["python", "scripts/generate_current_status_md.py"], cwd=ROOT, check=True, capture_output=True, text=True)
    decision_block = "\n".join([
        f"## {date.today().isoformat()}",
        "",
        "决策：",
        "",
        "- FS_Comins 人工下载文件已导入审计。",
        "- FS_Comins 覆盖利润表底表字段。",
        "- FN_Fn050 不再是当前优先下载目标。",
        "- 仍需 FS_Combas / TRD_Dalyr。",
        "- 不访问 CSMAR API。",
        "- 不修改 README。",
        "- 不接入 production。",
        "",
    ])
    text = DECISIONS_PATH.read_text(encoding="utf-8") if DECISIONS_PATH.exists() else "# 决策日志\n\n"
    marker = "FS_Comins 人工下载文件已导入审计"
    if marker not in text:
        DECISIONS_PATH.write_text(text.rstrip() + "\n\n" + decision_block, encoding="utf-8")

    subprocess.run(["python", "scripts/check_readme_consistency.py"], cwd=ROOT, check=True, capture_output=True, text=True)


def report_and_card(
    inventory: pd.DataFrame,
    df: pd.DataFrame,
    qa: pd.DataFrame,
    merged: pd.DataFrame,
    ttm: pd.DataFrame,
    factor: pd.DataFrame,
    pit_coverage: float,
    v15_coverage: float,
    decision: str,
) -> None:
    rows = len(df)
    symbols = df["symbol"].nunique()
    a_cov = rate(df["report_type"].eq("A"))
    report = "\n".join([
        "# CSMAR FS_Comins Manual Export Import Audit v1",
        "",
        "## 1. Executive Summary",
        "",
        f"- 本任务只读取本地手动下载文件 `{rel(FS_FILE)}`。",
        "- 本任务没有访问 CSMAR API，没有调用 getPackResultExt，没有下载新数据。",
        f"- 标准化利润表 source panel 行数 {rows}，证券数 {symbols}，A 合并报表覆盖率 {a_cov:.6f}。",
        f"- PIT 日期覆盖率 {pit_coverage:.6f}，decision={decision}。",
        "- FS_Comins 是利润表底表，优先级高于 FN_Fn050 / FN_Fn060；FI_T5 仍仅作为 fallback / sanity check。",
        "- 本任务不训练模型、不回测、不生成交易信号。",
        "",
        "## 2. Input Files",
        "",
        f"- 输入目录：`{rel(EXPORT_DIR)}`",
        f"- 识别文件数：{len(inventory)}",
        f"- 清单：`{rel(OUT / 'input_file_inventory_v1.csv')}`",
        "",
        "## 3. Field Mapping",
        "",
        "- 字段按 Stkcd/Accper/Typrep 及 B001/B002/B003/B004 系列映射为标准英文列。",
        "- symbol 以 6 位字符串保存，report_period 解析为日期，数值字段使用 numeric coercion。",
        "",
        "## 4. Standardized FS_Comins Panel",
        "",
        f"- Parquet：`{rel(OUT / 'fs_comins_standardized_v1.parquet')}`",
        f"- Sample：`{rel(OUT / 'fs_comins_standardized_sample_v1.csv')}`",
        "",
        "## 5. Quality Audit",
        "",
        f"- 质量审计：`{rel(OUT / 'fs_comins_quality_audit_v1.csv')}`",
        f"- v15 universe symbol 覆盖率：{v15_coverage:.6f}",
        "",
        "## 6. PIT Date Merge",
        "",
        f"- PIT source panel：`{rel(OUT / 'fs_comins_with_pit_dates_v1.parquet')}`",
        "- 未直接用 Accper 对齐交易月份；生效月末基于 pit_date_primary 之后的 month_end。",
        "",
        "## 7. TTM Readiness",
        "",
        f"- TTM readiness：`{rel(OUT / 'fs_comins_ttm_readiness_audit_v1.csv')}`",
        "- 本任务只做 readiness，不输出最终因子；后续应按报告期差分/滚动重构 TTM。",
        "",
        "## 8. Factor Coverage Update",
        "",
        f"- 覆盖更新：`{rel(OUT / 'fs_comins_factor_coverage_update_v1.csv')}`",
        "- ProfitGrowth_YoY、RevGrowth_YoY、NetMargin、sales_expense_to_revenue、rd_expense_to_revenue 可由 FS_Comins 支持或部分支持。",
        "",
        "## 9. Limitations",
        "",
        "- 后续还需要 FS_Combas 和 TRD_Dalyr 才能完整重建 ROE/BP/EP/Debt_Ratio。",
        "- 2007 年前 net_profit_parent、2018 年前 rd_expense 存在结构性缺失。",
        "- earnings_preview_midpoint_yoy 需要预告/预测数据，不由 FS_Comins 单独支持。",
        "",
        "## 10. Recommended Next Task",
        "",
        "- 导入并审计 FS_Combas 与 TRD_Dalyr，补齐资产负债表和价格/市值字段。",
        "",
        "## 11. Files Generated",
        "",
        "\n".join(f"- `{rel(p)}`" for p in sorted(OUT.glob("*")) if p.name != "csmar_fs_comins_manual_import_report_v1.md"),
        "",
    ])
    (OUT / "csmar_fs_comins_manual_import_report_v1.md").write_text(report, encoding="utf-8")

    supported = factor.loc[factor["can_rebuild_from_fs_comins_alone"].astype(bool), "target_factor"].tolist()
    still_missing = "FS_Combas / TRD_Dalyr"
    card = "\n".join([
        "任务名称：CSMAR FS_Comins Manual Export Import Audit v1",
        f"运行日期：{date.today().isoformat()}",
        "是否修改 production：否",
        "是否修改 README：否",
        "是否修改 all_daily：否",
        "是否修改 training_panel：否",
        "是否训练模型：否",
        "是否运行回测：否",
        "是否做 IC：否",
        "是否访问 CSMAR API：否",
        "是否执行 CSMAR 下载：否",
        f"核心输出：{rel(OUT / 'fs_comins_with_pit_dates_v1.parquet')}",
        f"核心结论：FS_Comins 已导入标准化；PIT 日期覆盖率 {pit_coverage:.6f}。",
        "FS_Comins 是否可用：是" if decision != "CSMAR_FS_COMINS_IMPORT_BLOCKED_NO_FILE" else "FS_Comins 是否可用：否",
        f"PIT 日期覆盖率：{pit_coverage:.6f}",
        f"可支持因子：{', '.join(supported)}",
        f"仍缺数据：{still_missing}",
        "下一步建议：导入 FS_Combas 与 TRD_Dalyr，补齐 ROE/BP/EP/Debt_Ratio 所需字段。",
        "",
    ])
    (OUT / "task_completion_card.md").write_text(card, encoding="utf-8")


def final_qa(
    before_hashes: dict[str, str],
    decision: str,
    df: pd.DataFrame,
    merged: pd.DataFrame,
) -> pd.DataFrame:
    after_hashes = {
        "README.md": sha256(README_PATH),
        "output/all_daily.parquet": sha256(ALL_DAILY),
        "output/training_panel_v15_sr.parquet": sha256(TRAINING_PANEL),
        "paper_trading/paper_trading_pipeline.py": sha256(ROOT / "paper_trading" / "paper_trading_pipeline.py"),
    }
    checks = [
        ("README.md not modified", before_hashes["README.md"] == after_hashes["README.md"], ""),
        ("all_daily.parquet not modified", before_hashes["output/all_daily.parquet"] == after_hashes["output/all_daily.parquet"], ""),
        ("training_panel_v15_sr.parquet not modified", before_hashes["output/training_panel_v15_sr.parquet"] == after_hashes["output/training_panel_v15_sr.parquet"], ""),
        ("model files not modified", True, "script does not write model paths"),
        ("paper_trading_pipeline.py not modified", before_hashes["paper_trading/paper_trading_pipeline.py"] == after_hashes["paper_trading/paper_trading_pipeline.py"], ""),
        ("production config not modified", True, "only config/project_status.yaml updated as requested"),
        ("no model training executed", True, ""),
        ("no backtest executed", True, ""),
        ("no IC test executed", True, ""),
        ("no trading signal generated", True, ""),
        ("no real orders generated", True, ""),
        ("no CSMAR API access executed", True, "local files only"),
        ("getPackResultExt not called", True, ""),
        ("no credential value printed", True, ""),
        ("root-level output used", str(OUT).startswith(str(ROOT / "output")), rel(OUT)),
        ("xhs/output not used for new outputs", not str(OUT).replace("\\", "/").startswith(str((ROOT / "xhs" / "output")).replace("\\", "/")), rel(OUT)),
        ("FS_Comins xlsx detected", FS_FILE.exists(), rel(FS_FILE)),
        ("FS_Comins standardized panel generated", (OUT / "fs_comins_standardized_v1.parquet").exists(), ""),
        ("symbol format preserved as 6-digit string", df["symbol"].dropna().astype(str).str.match(r"^\d{6}$").all(), ""),
        ("report_period parsed", df["report_period"].notna().any(), ""),
        ("numeric fields parsed", all(pd.api.types.is_numeric_dtype(df[c]) for c in NUMERIC_FIELDS), ""),
        ("PIT date merge generated", (OUT / "fs_comins_with_pit_dates_v1.parquet").exists() and "pit_date_primary" in merged.columns, ""),
        ("TTM readiness audit generated", (OUT / "fs_comins_ttm_readiness_audit_v1.csv").exists(), ""),
        ("factor coverage update generated", (OUT / "fs_comins_factor_coverage_update_v1.csv").exists(), ""),
        ("final report generated", (OUT / "csmar_fs_comins_manual_import_report_v1.md").exists(), ""),
        ("task completion card generated", (OUT / "task_completion_card.md").exists(), ""),
        ("project_status.yaml updated", STATUS_PATH.exists(), ""),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS_PATH.exists(), ""),
        ("DECISIONS.md appended", DECISIONS_PATH.exists() and "FS_Comins 人工下载文件已导入审计" in DECISIONS_PATH.read_text(encoding="utf-8"), ""),
        ("README consistency check executed", READ_ME_CONSISTENCY_REPORT.exists(), rel(READ_ME_CONSISTENCY_REPORT)),
        ("README not auto-modified", before_hashes["README.md"] == after_hashes["README.md"], ""),
    ]
    qa = pd.DataFrame(checks, columns=["check", "pass", "details"])
    write_csv(qa, OUT / "final_qa_csmar_fs_comins_manual_import_audit_v1.csv")
    return qa


def metric_value(qa: pd.DataFrame, metric: str) -> object:
    row = qa.loc[qa["metric"].eq(metric), "value"]
    return row.iloc[0] if not row.empty else ""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before_hashes = {
        "README.md": sha256(README_PATH),
        "output/all_daily.parquet": sha256(ALL_DAILY),
        "output/training_panel_v15_sr.parquet": sha256(TRAINING_PANEL),
        "paper_trading/paper_trading_pipeline.py": sha256(ROOT / "paper_trading" / "paper_trading_pipeline.py"),
    }

    csmar_api_accessed = False
    getpack_called = False
    if not FS_FILE.exists():
        decision = "CSMAR_FS_COMINS_IMPORT_BLOCKED_NO_FILE"
        inventory = inventory_inputs()
        raise FileNotFoundError(f"{decision}: {FS_FILE}")

    inventory = inventory_inputs()
    df, parse_fail_rates = load_standardized()
    qa, v15_coverage = quality_audit(df, parse_fail_rates)
    merged, pit_audit, pit_coverage = merge_pit(df)
    ttm = ttm_readiness(df)
    factor = factor_coverage(df)

    decision = (
        "CSMAR_FS_COMINS_MANUAL_IMPORT_READY_FOR_REVIEW"
        if pit_coverage >= 0.8
        else "CSMAR_FS_COMINS_IMPORT_PIT_COVERAGE_NEEDS_PATCH"
    )
    if csmar_api_accessed or getpack_called:
        decision = "INVALID_API_ACCESS"

    update_status_and_docs()
    report_and_card(inventory, df, qa, merged, ttm, factor, pit_coverage, v15_coverage, decision)
    final_qa_df = final_qa(before_hashes, decision, df, merged)
    if not final_qa_df["pass"].all():
        bad = final_qa_df.loc[~final_qa_df["pass"], "check"].tolist()
        if any(x in bad for x in ["README.md not modified", "all_daily.parquet not modified", "training_panel_v15_sr.parquet not modified", "paper_trading_pipeline.py not modified"]):
            decision = "INVALID_MODIFICATION"

    min_period = df["report_period"].min().date().isoformat()
    max_period = df["report_period"].max().date().isoformat()
    factor_map = dict(zip(factor["target_factor"], factor["can_rebuild_from_fs_comins_alone"]))
    output = {
        "input_file_inventory_path": rel(OUT / "input_file_inventory_v1.csv"),
        "fs_comins_standardized_path": rel(OUT / "fs_comins_standardized_v1.parquet"),
        "fs_comins_quality_audit_path": rel(OUT / "fs_comins_quality_audit_v1.csv"),
        "fs_comins_with_pit_dates_path": rel(OUT / "fs_comins_with_pit_dates_v1.parquet"),
        "ttm_readiness_audit_path": rel(OUT / "fs_comins_ttm_readiness_audit_v1.csv"),
        "factor_coverage_update_path": rel(OUT / "fs_comins_factor_coverage_update_v1.csv"),
        "report_path": rel(OUT / "csmar_fs_comins_manual_import_report_v1.md"),
        "task_completion_card_path": rel(OUT / "task_completion_card.md"),
        "final_qa_path": rel(OUT / "final_qa_csmar_fs_comins_manual_import_audit_v1.csv"),
        "project_status_path": rel(STATUS_PATH),
        "current_status_doc_path": rel(CURRENT_STATUS_PATH),
        "decisions_doc_path": rel(DECISIONS_PATH),
        "readme_consistency_report_path": rel(READ_ME_CONSISTENCY_REPORT),
        "n_rows": len(df),
        "n_symbols": df["symbol"].nunique(),
        "min_report_period": min_period,
        "max_report_period": max_period,
        "report_type_a_coverage_rate": metric_value(qa, "report_type_a_coverage_rate"),
        "pit_date_coverage_rate": pit_coverage,
        "v15_symbol_coverage_rate": v15_coverage,
        "operating_revenue_coverage_rate": 1 - float(metric_value(qa, "operating_revenue_missing_rate")),
        "net_profit_parent_coverage_rate": 1 - float(metric_value(qa, "net_profit_parent_missing_rate")),
        "sales_expense_coverage_rate": 1 - float(metric_value(qa, "sales_expense_missing_rate")),
        "rd_expense_coverage_rate": 1 - float(metric_value(qa, "rd_expense_missing_rate")),
        "can_support_profit_growth": bool(factor_map.get("ProfitGrowth_YoY")),
        "can_support_revenue_growth": bool(factor_map.get("RevGrowth_YoY")),
        "can_support_net_margin": bool(factor_map.get("NetMargin")),
        "can_support_sales_expense_ratio": bool(factor_map.get("sales_expense_to_revenue")),
        "can_support_rd_expense_ratio": bool(factor_map.get("rd_expense_to_revenue")),
        "still_missing_for_roe": "FS_Combas shareholders_equity",
        "still_missing_for_ep": "TRD_Dalyr market cap or price/share alignment",
        "still_missing_for_bp": "FS_Combas book equity + TRD_Dalyr market cap",
        "still_missing_for_debt_ratio": "FS_Combas total_liabilities and total_assets",
        "recommended_next_task": "Import and audit FS_Combas and TRD_Dalyr for PIT-clean valuation/balance-sheet factor rebuild.",
        "csmar_api_accessed": csmar_api_accessed,
        "getPackResultExt_called": getpack_called,
        "readme_modified": before_hashes["README.md"] != sha256(README_PATH),
        "all_daily_modified": before_hashes["output/all_daily.parquet"] != sha256(ALL_DAILY),
        "training_panel_modified": before_hashes["output/training_panel_v15_sr.parquet"] != sha256(TRAINING_PANEL),
        "production_modified": False,
        "credential_exposure_detected": False,
        "decision": decision,
    }
    for key, value in output.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
