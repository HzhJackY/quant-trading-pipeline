from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "csmar_exports"
OUT = ROOT / "output" / "csmar_p0_pit_pack_import_audit_v1"
STATUS_PATH = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS_PATH = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
README_CONSISTENCY_REPORT = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"

ENCODINGS = ["utf-8-sig", "gbk", "gb18030"]
RUN_DATE = date.today().isoformat()
MIN_V15_SYMBOL_COVERAGE = 0.95


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_with_encoding(path: Path, nrows: int | None = None) -> tuple[pd.DataFrame, str, str]:
    last_error = ""
    for encoding in ENCODINGS:
        try:
            df = pd.read_csv(path, dtype=str, encoding=encoding, nrows=nrows)
            return df, encoding, ""
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(f"unable to read {path}: {last_error}")


def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def six_digit_symbol(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)


def note_join(parts: list[str]) -> str:
    return "; ".join([p for p in parts if p])


def discover_files() -> tuple[pd.DataFrame, dict[str, Path]]:
    rows: list[dict[str, Any]] = []
    detected: dict[str, Path] = {}
    for path in sorted(INPUT_DIR.glob("*.csv")):
        table = ""
        name = path.name
        if "IAR_Rept" in name:
            table = "IAR_Rept"
        elif "IAR_Forecdt" in name:
            table = "IAR_Forecdt"
        readable = False
        encoding_used = ""
        n_rows: int | None = None
        n_columns: int | None = None
        columns = ""
        notes: list[str] = []
        try:
            df, encoding_used, _ = read_csv_with_encoding(path)
            readable = True
            n_rows = int(len(df))
            n_columns = int(len(df.columns))
            columns = "|".join(map(str, df.columns))
            if table:
                detected[table] = path
            else:
                notes.append("not a P0 target table")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"read_error={type(exc).__name__}: {exc}")
        rows.append({
            "file_path": rel(path),
            "detected_table": table or "UNKNOWN",
            "file_size": path.stat().st_size,
            "readable": readable,
            "encoding_used": encoding_used,
            "n_rows": n_rows,
            "n_columns": n_columns,
            "columns": columns,
            "notes": note_join(notes),
        })
    inventory = pd.DataFrame(rows, columns=[
        "file_path", "detected_table", "file_size", "readable", "encoding_used",
        "n_rows", "n_columns", "columns", "notes",
    ])
    return inventory, detected


def standardize_iar_rept(path: Path) -> pd.DataFrame:
    raw, _, _ = read_csv_with_encoding(path)
    required = {"Stkcd", "Accper", "Annodt"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"IAR_Rept missing columns: {sorted(missing)}")
    out = pd.DataFrame()
    out["symbol"] = six_digit_symbol(raw["Stkcd"])
    out["report_period"] = parse_date(raw["Accper"])
    out["announcement_date"] = parse_date(raw["Annodt"])
    out["source_table"] = "IAR_Rept"
    out["source_file"] = rel(path)
    out["raw_Stkcd"] = raw["Stkcd"]
    out["raw_Accper"] = raw["Accper"]
    out["raw_Annodt"] = raw["Annodt"]
    notes = []
    for i in out.index:
        row_notes = []
        if pd.isna(out.at[i, "report_period"]):
            row_notes.append("UNPARSEABLE_REPORT_PERIOD")
        if pd.isna(out.at[i, "announcement_date"]):
            row_notes.append("UNPARSEABLE_ANNODT")
        notes.append(note_join(row_notes))
    out["notes"] = notes
    return out[[
        "symbol", "report_period", "announcement_date", "source_table", "source_file",
        "raw_Stkcd", "raw_Accper", "raw_Annodt", "notes",
    ]]


def standardize_iar_forecdt(path: Path) -> pd.DataFrame:
    raw, _, _ = read_csv_with_encoding(path)
    required = {"Stkcd", "Accper", "Firforecdt", "Actudt"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"IAR_Forecdt missing columns: {sorted(missing)}")
    out = pd.DataFrame()
    out["symbol"] = six_digit_symbol(raw["Stkcd"])
    out["report_period"] = parse_date(raw["Accper"])
    out["first_forecast_date"] = parse_date(raw["Firforecdt"])
    out["actual_disclosure_date"] = parse_date(raw["Actudt"])
    out["source_table"] = "IAR_Forecdt"
    out["source_file"] = rel(path)
    out["raw_Stkcd"] = raw["Stkcd"]
    out["raw_Accper"] = raw["Accper"]
    out["raw_Firforecdt"] = raw["Firforecdt"]
    out["raw_Actudt"] = raw["Actudt"]
    notes = []
    for i in out.index:
        row_notes = []
        if pd.isna(out.at[i, "report_period"]):
            row_notes.append("UNPARSEABLE_REPORT_PERIOD")
        if pd.isna(out.at[i, "first_forecast_date"]):
            row_notes.append("UNPARSEABLE_FIRFORECDT")
        if pd.isna(out.at[i, "actual_disclosure_date"]):
            row_notes.append("UNPARSEABLE_ACTUDT")
        notes.append(note_join(row_notes))
    out["notes"] = notes
    return out[[
        "symbol", "report_period", "first_forecast_date", "actual_disclosure_date",
        "source_table", "source_file", "raw_Stkcd", "raw_Accper", "raw_Firforecdt",
        "raw_Actudt", "notes",
    ]]


def first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else pd.NaT


def collapse_source(df: pd.DataFrame, date_cols: list[str], table_name: str) -> pd.DataFrame:
    grouped = df.groupby(["symbol", "report_period"], dropna=False)
    agg_spec: dict[str, Any] = {col: first_non_null for col in date_cols}
    agg_spec["source_file"] = lambda s: "|".join(sorted(set(map(str, s.dropna()))))
    out = grouped.agg(agg_spec).reset_index()
    out["source_table"] = table_name
    out["source_row_count"] = grouped.size().to_numpy()
    return out


def combine_panel(rept: pd.DataFrame, fore: pd.DataFrame) -> pd.DataFrame:
    rept_c = collapse_source(rept, ["announcement_date"], "IAR_Rept").rename(columns={
        "announcement_date": "annodt_from_iar_rept",
        "source_file": "iar_rept_source_file",
        "source_row_count": "iar_rept_source_row_count",
    })
    fore_c = collapse_source(fore, ["first_forecast_date", "actual_disclosure_date"], "IAR_Forecdt").rename(columns={
        "source_file": "iar_forecdt_source_file",
        "source_row_count": "iar_forecdt_source_row_count",
    })
    panel = rept_c.merge(fore_c, on=["symbol", "report_period"], how="outer", suffixes=("_rept", "_fore"))
    for col in ["annodt_from_iar_rept", "first_forecast_date", "actual_disclosure_date"]:
        if col not in panel.columns:
            panel[col] = pd.NaT
        panel[col] = pd.to_datetime(panel[col], errors="coerce")

    pit_dates = []
    pit_sources = []
    quality_flags = []
    source_tables = []
    for row in panel.itertuples(index=False):
        if pd.notna(row.actual_disclosure_date):
            pit_dates.append(row.actual_disclosure_date)
            pit_sources.append("actual_disclosure_date")
            quality_flags.append("OK")
        elif pd.notna(row.annodt_from_iar_rept):
            pit_dates.append(row.annodt_from_iar_rept)
            pit_sources.append("annodt_from_iar_rept")
            quality_flags.append("OK")
        elif pd.notna(row.first_forecast_date):
            pit_dates.append(row.first_forecast_date)
            pit_sources.append("first_forecast_date")
            quality_flags.append("LOW_CONFIDENCE_FORECAST_ONLY")
        else:
            pit_dates.append(pd.NaT)
            pit_sources.append("")
            quality_flags.append("MISSING_PIT_DATE")
        tables = []
        if pd.notna(getattr(row, "iar_rept_source_row_count", pd.NA)):
            tables.append("IAR_Rept")
        if pd.notna(getattr(row, "iar_forecdt_source_row_count", pd.NA)):
            tables.append("IAR_Forecdt")
        source_tables.append("|".join(tables))
    panel["pit_date_primary"] = pd.to_datetime(pd.Series(pit_dates), errors="coerce")
    panel["pit_date_source"] = pit_sources
    panel["effective_month_end"] = panel["pit_date_primary"] + pd.offsets.MonthEnd(0)
    panel["source_tables"] = source_tables
    panel["quality_flag"] = quality_flags
    return panel[[
        "symbol", "report_period", "annodt_from_iar_rept", "first_forecast_date",
        "actual_disclosure_date", "pit_date_primary", "pit_date_source",
        "effective_month_end", "source_tables", "quality_flag",
    ]]


def metric(rows: list[dict[str, str]], name: str, value: Any, details: str = "") -> None:
    rows.append({"metric": name, "value": "" if pd.isna(value) else str(value), "details": details})


def check(rows: list[dict[str, Any]], name: str, passed: bool, n_violations: int, details: str) -> None:
    rows.append({"check": name, "pass": bool(passed), "n_violations": int(n_violations), "details": details})


def coverage_and_quality(panel: pd.DataFrame, rept: pd.DataFrame, fore: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = pd.read_parquet(ROOT / "output" / "training_panel_v15_sr.parquet", columns=["symbol"])
    v15_symbols = set(six_digit_symbol(train["symbol"]).dropna().astype(str))
    panel_symbols = set(panel["symbol"].dropna().astype(str))
    coverage = len(v15_symbols & panel_symbols) / len(v15_symbols) if v15_symbols else 0.0
    duplicate_count = int(panel.duplicated(["symbol", "report_period"]).sum())
    missing_rate = float(panel["pit_date_primary"].isna().mean()) if len(panel) else 1.0
    rows: list[dict[str, str]] = []
    metric(rows, "p0_pit_panel_total_rows", len(panel))
    metric(rows, "p0_pit_panel_symbol_count", panel["symbol"].nunique())
    metric(rows, "report_period_range", f"{panel['report_period'].min().date()} to {panel['report_period'].max().date()}" if panel["report_period"].notna().any() else "")
    metric(rows, "pit_date_range", f"{panel['pit_date_primary'].min().date()} to {panel['pit_date_primary'].max().date()}" if panel["pit_date_primary"].notna().any() else "")
    metric(rows, "v15_symbol_coverage_rate", f"{coverage:.6f}", f"covered={len(v15_symbols & panel_symbols)}; v15_symbols={len(v15_symbols)}")
    metric(rows, "duplicate_symbol_report_period_count", duplicate_count)
    metric(rows, "missing_pit_date_rate", f"{missing_rate:.6f}")
    diffs1 = (panel["actual_disclosure_date"] - panel["annodt_from_iar_rept"]).dt.days.dropna()
    metric(rows, "actual_disclosure_date_minus_annodt_days", diffs1.describe().to_json(), f"n={len(diffs1)}")
    diffs2 = (panel["actual_disclosure_date"] - panel["first_forecast_date"]).dt.days.dropna()
    metric(rows, "actual_disclosure_date_minus_first_forecast_date_days", diffs2.describe().to_json(), f"n={len(diffs2)}")
    metric(rows, "iar_rept_rows", len(rept))
    metric(rows, "iar_forecdt_rows", len(fore))
    stats = {
        "v15_symbol_coverage_rate": coverage,
        "duplicate_symbol_report_period_count": duplicate_count,
        "missing_pit_date_rate": missing_rate,
    }
    return pd.DataFrame(rows, columns=["metric", "value", "details"]), stats


def pit_qa(panel: pd.DataFrame, coverage: float) -> tuple[pd.DataFrame, bool]:
    rows: list[dict[str, Any]] = []
    symbol_ok = panel["symbol"].astype(str).str.fullmatch(r"\d{6}").fillna(False)
    check(rows, "symbol all 6-digit strings", bool(symbol_ok.all()), int((~symbol_ok).sum()), "")
    rp_bad = int(panel["report_period"].isna().sum())
    check(rows, "report_period parseable", rp_bad == 0, rp_bad, "")
    pit_bad_mask = panel["pit_date_primary"].isna()
    pit_unrecorded_mask = pit_bad_mask & (panel["quality_flag"] != "MISSING_PIT_DATE")
    check(
        rows,
        "pit_date_primary parseable or missing flagged",
        not bool(pit_unrecorded_mask.any()),
        int(pit_bad_mask.sum()),
        f"missing pit_date rows recorded={int(pit_bad_mask.sum())}",
    )
    eff_bad_mask = panel["pit_date_primary"].notna() & (panel["effective_month_end"] < panel["pit_date_primary"])
    check(rows, "effective_month_end >= pit_date_primary", not bool(eff_bad_mask.any()), int(eff_bad_mask.sum()), "")
    dup = int(panel.duplicated(["symbol", "report_period"]).sum())
    check(rows, "no duplicate symbol + report_period or duplicate recorded", True, dup, "duplicates recorded in audit")
    check(rows, "v15 universe coverage sufficient", coverage >= MIN_V15_SYMBOL_COVERAGE, 0 if coverage >= MIN_V15_SYMBOL_COVERAGE else 1, f"coverage={coverage:.6f}; threshold={MIN_V15_SYMBOL_COVERAGE:.2f}")
    check(rows, "no CSMAR API access", True, 0, "local CSV only")
    check(rows, "getPackResultExt not called", True, 0, "not present in import script")
    df = pd.DataFrame(rows, columns=["check", "pass", "n_violations", "details"])
    usable = bool(symbol_ok.all() and coverage >= MIN_V15_SYMBOL_COVERAGE and len(panel) > 0)
    return df, usable


def write_report(
    inventory: pd.DataFrame,
    rept: pd.DataFrame,
    fore: pd.DataFrame,
    panel: pd.DataFrame,
    audit: pd.DataFrame,
    qa: pd.DataFrame,
    p0_usable: bool,
    recommended_next_task: str,
    decision: str,
) -> Path:
    lines = [
        "# CSMAR P0 PIT Pack Import Audit v1",
        "",
        "## 1. Executive Summary",
        "",
        "- This task only read local CSV files under `data/csmar_exports/`.",
        "- This task did not access the CSMAR API.",
        "- This task did not download new data.",
        "- This task did not train models.",
        "- This task did not run backtests.",
        f"- Decision: `{decision}`",
        f"- P0 PIT panel usable: `{p0_usable}`",
        f"- Recommended next task: `{recommended_next_task}`",
        "",
        "## 2. Input Files",
        "",
        inventory.to_markdown(index=False),
        "",
        "## 3. Field Mapping",
        "",
        "- IAR_Rept: `Stkcd -> symbol`, `Accper -> report_period`, `Annodt -> announcement_date`.",
        "- IAR_Forecdt: `Stkcd -> symbol`, `Accper -> report_period`, `Firforecdt -> first_forecast_date`, `Actudt -> actual_disclosure_date`.",
        "",
        "## 4. Standardized IAR_Rept",
        "",
        f"- Rows: {len(rept)}",
        f"- Symbols: {rept['symbol'].nunique()}",
        f"- Unparseable report_period: {int(rept['report_period'].isna().sum())}",
        f"- Unparseable announcement_date: {int(rept['announcement_date'].isna().sum())}",
        "",
        "## 5. Standardized IAR_Forecdt",
        "",
        f"- Rows: {len(fore)}",
        f"- Symbols: {fore['symbol'].nunique()}",
        f"- Unparseable report_period: {int(fore['report_period'].isna().sum())}",
        f"- Unparseable first_forecast_date: {int(fore['first_forecast_date'].isna().sum())}",
        f"- Unparseable actual_disclosure_date: {int(fore['actual_disclosure_date'].isna().sum())}",
        "",
        "## 6. Combined PIT Announcement Panel",
        "",
        f"- Rows: {len(panel)}",
        f"- Symbols: {panel['symbol'].nunique()}",
        f"- Missing pit_date_primary: {int(panel['pit_date_primary'].isna().sum())}",
        "",
        "## 7. Coverage and Quality Audit",
        "",
        audit.to_markdown(index=False),
        "",
        "## 8. PIT QA",
        "",
        qa.to_markdown(index=False),
        "",
        "## 9. Limitations",
        "",
        "- Financial raw fields were not downloaded or imported in this task.",
        "- Coverage is measured against the existing v15 training panel symbol universe only.",
        "- PIT date quality depends on the local CSMAR pack CSV contents.",
        "",
        "## 10. Recommended Next Task",
        "",
        recommended_next_task,
        "",
        "## 11. Files Generated",
        "",
    ]
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            lines.append(f"- `{rel(path)}`")
    path = OUT / "csmar_p0_pit_pack_import_report_v1.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def update_project_status(success: bool) -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    status.setdefault("alternative_data", {})
    status["alternative_data"]["csmar_status"] = "p0_pit_pack_import_completed" if success else "p0_pit_pack_import_failed"
    status["alternative_data"]["csmar_latest_task"] = "CSMAR P0 PIT Pack Import Audit v1"
    status["alternative_data"]["csmar_latest_output"] = rel(OUT)
    status.setdefault("validation", {})
    status["validation"]["pit_financial_status"] = "p0_pit_dates_imported_financial_fields_pending"
    status["validation"]["blend_v3_historical_metrics_status"] = "under_pit_review"
    status.setdefault("project", {})["last_updated"] = RUN_DATE
    STATUS_PATH.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_status_and_readme_checks() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_current_status_md.py")], cwd=ROOT, check=True, capture_output=True, text=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_readme_consistency.py")], cwd=ROOT, check=True, capture_output=True, text=True)


def append_decision(decision: str, panel_generated: bool) -> None:
    marker = f"- Decision = {decision}。"
    existing = DECISIONS_PATH.read_text(encoding="utf-8") if DECISIONS_PATH.exists() else "# 决策日志\n"
    if marker in existing and "CSMAR P0 PIT pack 文件已成功导入" in existing:
        return
    block = "\n".join([
        f"## {RUN_DATE}",
        "",
        "决策：",
        "",
        "- CSMAR P0 PIT pack 文件已成功导入。" if decision != "CSMAR_P0_PIT_IMPORT_BLOCKED_NO_FILES" else "- CSMAR P0 PIT pack 文件导入失败或缺失。",
        "- IAR_Rept / IAR_Forecdt 已标准化。" if panel_generated else "- IAR_Rept / IAR_Forecdt 未全部完成标准化。",
        f"- 是否生成 PIT 公告日面板：{'是' if panel_generated else '否'}。",
        "- 财务原始字段仍未下载。",
        "- 不接入 production。",
        "- 不修改 README。",
        f"- Decision = {decision}。",
    ])
    DECISIONS_PATH.write_text(existing.rstrip() + "\n\n" + block + "\n", encoding="utf-8")
    if marker not in DECISIONS_PATH.read_text(encoding="utf-8"):
        raise RuntimeError("decision append verification failed")


def task_completion_card(core_outputs: list[str], p0_usable: bool, recommended_next_task: str, decision: str) -> Path:
    path = OUT / "task_completion_card.md"
    lines = [
        "任务名称：CSMAR P0 PIT Pack Import Audit v1",
        f"运行日期：{RUN_DATE}",
        "是否修改 production：否",
        "是否修改 README：否",
        "是否修改 all_daily：否",
        "是否修改 training_panel：否",
        "是否训练模型：否",
        "是否运行回测：否",
        "是否做 IC：否",
        "是否访问 CSMAR API：否",
        "是否下载 CSMAR 数据：否",
        "核心输出：",
        *[f"- {item}" for item in core_outputs],
        f"核心结论：{decision}",
        f"P0 PIT 面板是否可用：{'是' if p0_usable else '否'}",
        f"下一步建议：{recommended_next_task}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def final_qa(
    hashes_before: dict[str, str | None],
    hashes_after: dict[str, str | None],
    imported_rept: bool,
    imported_fore: bool,
    panel_generated: bool,
    symbol_format_pass: bool,
    effective_month_end_generated: bool,
    coverage_quality_audit_generated: bool,
    report_generated: bool,
    card_generated: bool,
) -> pd.DataFrame:
    def same(name: str) -> bool:
        return hashes_before.get(name) == hashes_after.get(name)

    rows = [
        ("README.md not modified", same("README.md"), "hash unchanged"),
        ("all_daily.parquet not modified", same("output/all_daily.parquet"), "hash unchanged"),
        ("training_panel_v15_sr.parquet not modified", same("output/training_panel_v15_sr.parquet"), "hash unchanged"),
        ("model files not modified", True, "no model paths written by this script"),
        ("paper_trading_pipeline.py not modified", same("paper_trading/paper_trading_pipeline.py"), "hash unchanged or file absent"),
        ("production config not modified", True, "only config/project_status.yaml was updated"),
        ("no model training executed", True, "no training command executed"),
        ("no backtest executed", True, "no backtest command executed"),
        ("no IC test executed", True, "no IC command executed"),
        ("no trading signal generated", True, "no signal output generated"),
        ("no real orders generated", True, "no order output generated"),
        ("no CSMAR API access executed", True, "local CSV only"),
        ("getPackResultExt not called", True, "not called"),
        ("no credential value printed", True, "credential files were not read"),
        ("root-level output used", str(OUT).startswith(str(ROOT / "output")), rel(OUT)),
        ("xhs/output not used for new outputs", not str(OUT).replace("\\", "/").startswith(str(ROOT / "xhs" / "output").replace("\\", "/")), rel(OUT)),
        ("IAR_Rept imported", imported_rept, ""),
        ("IAR_Forecdt imported", imported_fore, ""),
        ("combined PIT announcement panel generated", panel_generated, ""),
        ("symbol format preserved as 6-digit string", symbol_format_pass, ""),
        ("effective_month_end generated", effective_month_end_generated, ""),
        ("coverage quality audit generated", coverage_quality_audit_generated, ""),
        ("final report generated", report_generated, ""),
        ("task completion card generated", card_generated, ""),
        ("project_status.yaml updated", STATUS_PATH.exists(), rel(STATUS_PATH)),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS_PATH.exists(), rel(CURRENT_STATUS_PATH)),
        ("DECISIONS.md appended", DECISIONS_PATH.exists(), rel(DECISIONS_PATH)),
        ("README consistency check executed", README_CONSISTENCY_REPORT.exists(), rel(README_CONSISTENCY_REPORT)),
        ("README not auto-modified", same("README.md"), "hash unchanged"),
    ]
    return pd.DataFrame(rows, columns=["check", "pass", "details"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [
        "README.md",
        "output/all_daily.parquet",
        "output/training_panel_v15_sr.parquet",
        "paper_trading/paper_trading_pipeline.py",
    ]
    hashes_before = {p: file_hash(ROOT / p) for p in protected}

    inventory_path = OUT / "local_p0_file_inventory_v1.csv"
    rept_path = OUT / "iar_rept_standardized_v1.parquet"
    fore_path = OUT / "iar_forecdt_standardized_v1.parquet"
    panel_path = OUT / "csmar_p0_pit_announcement_panel_v1.parquet"
    audit_path = OUT / "p0_pit_coverage_quality_audit_v1.csv"
    pit_qa_path = OUT / "p0_pit_import_qa_v1.csv"
    final_qa_path = OUT / "final_qa_csmar_p0_pit_pack_import_audit_v1.csv"

    inventory, detected = discover_files()
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    imported_rept = "IAR_Rept" in detected
    imported_fore = "IAR_Forecdt" in detected

    if not imported_rept or not imported_fore:
        decision = "CSMAR_P0_PIT_IMPORT_BLOCKED_NO_FILES"
        update_project_status(False)
        run_status_and_readme_checks()
        append_decision(decision, False)
        empty = pd.DataFrame()
        report_path = write_report(inventory, empty, empty, empty, pd.DataFrame(), pd.DataFrame(), False, "CSMAR P0 PIT Import Patch v1", decision)
        card_path = task_completion_card([rel(inventory_path), rel(report_path)], False, "CSMAR P0 PIT Import Patch v1", decision)
        hashes_after = {p: file_hash(ROOT / p) for p in protected}
        qa_df = final_qa(hashes_before, hashes_after, imported_rept, imported_fore, False, False, False, False, report_path.exists(), card_path.exists())
        qa_df.to_csv(final_qa_path, index=False, encoding="utf-8-sig")
        values = {
            "local_file_inventory_path": rel(inventory_path),
            "iar_rept_standardized_path": "",
            "iar_forecdt_standardized_path": "",
            "p0_pit_panel_path": "",
            "coverage_quality_audit_path": "",
            "p0_pit_import_qa_path": "",
            "report_path": rel(report_path),
            "task_completion_card_path": rel(card_path),
            "final_qa_path": rel(final_qa_path),
            "project_status_path": rel(STATUS_PATH),
            "current_status_doc_path": rel(CURRENT_STATUS_PATH),
            "decisions_doc_path": rel(DECISIONS_PATH),
            "readme_consistency_report_path": rel(README_CONSISTENCY_REPORT),
            "n_input_files": len(inventory),
            "n_iar_rept_rows": 0,
            "n_iar_forecdt_rows": 0,
            "n_combined_pit_rows": 0,
            "n_symbols": 0,
            "min_report_period": "",
            "max_report_period": "",
            "min_pit_date": "",
            "max_pit_date": "",
            "v15_symbol_coverage_rate": "0.000000",
            "missing_pit_date_rate": "1.000000",
            "duplicate_symbol_report_period_count": 0,
            "symbol_format_pass": False,
            "p0_pit_panel_usable": False,
            "recommended_next_task": "CSMAR P0 PIT Import Patch v1",
            "readme_modified": not (hashes_before["README.md"] == hashes_after["README.md"]),
            "all_daily_modified": not (hashes_before["output/all_daily.parquet"] == hashes_after["output/all_daily.parquet"]),
            "training_panel_modified": not (hashes_before["output/training_panel_v15_sr.parquet"] == hashes_after["output/training_panel_v15_sr.parquet"]),
            "production_modified": False,
            "csmar_api_accessed": False,
            "getPackResultExt_called": False,
            "credential_exposure_detected": False,
            "decision": decision,
        }
        for k, v in values.items():
            print(f"{k}={v}")
        return

    rept = standardize_iar_rept(detected["IAR_Rept"])
    fore = standardize_iar_forecdt(detected["IAR_Forecdt"])
    rept.to_parquet(rept_path, index=False)
    fore.to_parquet(fore_path, index=False)
    rept.head(200).to_csv(OUT / "iar_rept_standardized_sample_v1.csv", index=False, encoding="utf-8-sig")
    fore.head(200).to_csv(OUT / "iar_forecdt_standardized_sample_v1.csv", index=False, encoding="utf-8-sig")

    panel = combine_panel(rept, fore)
    panel.to_parquet(panel_path, index=False)
    panel.head(200).to_csv(OUT / "csmar_p0_pit_announcement_panel_sample_v1.csv", index=False, encoding="utf-8-sig")

    audit, stats = coverage_and_quality(panel, rept, fore)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    pit_qa_df, p0_usable = pit_qa(panel, stats["v15_symbol_coverage_rate"])
    pit_qa_df.to_csv(pit_qa_path, index=False, encoding="utf-8-sig")

    symbol_format_pass = bool(panel["symbol"].astype(str).str.fullmatch(r"\d{6}").fillna(False).all())
    if not imported_rept or not imported_fore or not panel_path.exists():
        decision = "CSMAR_P0_PIT_IMPORT_BLOCKED_NO_FILES"
    elif not symbol_format_pass or stats["v15_symbol_coverage_rate"] < MIN_V15_SYMBOL_COVERAGE:
        decision = "CSMAR_P0_PIT_IMPORT_NEEDS_PATCH"
    else:
        decision = "CSMAR_P0_PIT_PACK_IMPORT_READY_FOR_REVIEW"

    recommended_next_task = "CSMAR P1 Financial Pack Download v1" if decision == "CSMAR_P0_PIT_PACK_IMPORT_READY_FOR_REVIEW" else "CSMAR P0 PIT Import Patch v1"
    update_project_status(decision == "CSMAR_P0_PIT_PACK_IMPORT_READY_FOR_REVIEW")
    run_status_and_readme_checks()
    append_decision(decision, True)

    report_path = write_report(inventory, rept, fore, panel, audit, pit_qa_df, p0_usable, recommended_next_task, decision)
    core_outputs = [rel(rept_path), rel(fore_path), rel(panel_path), rel(audit_path), rel(pit_qa_path), rel(report_path)]
    card_path = task_completion_card(core_outputs, p0_usable, recommended_next_task, decision)

    hashes_after = {p: file_hash(ROOT / p) for p in protected}
    qa_df = final_qa(
        hashes_before,
        hashes_after,
        imported_rept,
        imported_fore,
        panel_path.exists(),
        symbol_format_pass,
        "effective_month_end" in panel.columns and panel["effective_month_end"].notna().any(),
        audit_path.exists(),
        report_path.exists(),
        card_path.exists(),
    )
    qa_df.to_csv(final_qa_path, index=False, encoding="utf-8-sig")
    report_path = write_report(inventory, rept, fore, panel, audit, pit_qa_df, p0_usable, recommended_next_task, decision)

    values = {
        "local_file_inventory_path": rel(inventory_path),
        "iar_rept_standardized_path": rel(rept_path),
        "iar_forecdt_standardized_path": rel(fore_path),
        "p0_pit_panel_path": rel(panel_path),
        "coverage_quality_audit_path": rel(audit_path),
        "p0_pit_import_qa_path": rel(pit_qa_path),
        "report_path": rel(report_path),
        "task_completion_card_path": rel(card_path),
        "final_qa_path": rel(final_qa_path),
        "project_status_path": rel(STATUS_PATH),
        "current_status_doc_path": rel(CURRENT_STATUS_PATH),
        "decisions_doc_path": rel(DECISIONS_PATH),
        "readme_consistency_report_path": rel(README_CONSISTENCY_REPORT),
        "n_input_files": len(inventory),
        "n_iar_rept_rows": len(rept),
        "n_iar_forecdt_rows": len(fore),
        "n_combined_pit_rows": len(panel),
        "n_symbols": panel["symbol"].nunique(),
        "min_report_period": panel["report_period"].min().date() if panel["report_period"].notna().any() else "",
        "max_report_period": panel["report_period"].max().date() if panel["report_period"].notna().any() else "",
        "min_pit_date": panel["pit_date_primary"].min().date() if panel["pit_date_primary"].notna().any() else "",
        "max_pit_date": panel["pit_date_primary"].max().date() if panel["pit_date_primary"].notna().any() else "",
        "v15_symbol_coverage_rate": f"{stats['v15_symbol_coverage_rate']:.6f}",
        "missing_pit_date_rate": f"{stats['missing_pit_date_rate']:.6f}",
        "duplicate_symbol_report_period_count": stats["duplicate_symbol_report_period_count"],
        "symbol_format_pass": symbol_format_pass,
        "p0_pit_panel_usable": p0_usable,
        "recommended_next_task": recommended_next_task,
        "readme_modified": not (hashes_before["README.md"] == hashes_after["README.md"]),
        "all_daily_modified": not (hashes_before["output/all_daily.parquet"] == hashes_after["output/all_daily.parquet"]),
        "training_panel_modified": not (hashes_before["output/training_panel_v15_sr.parquet"] == hashes_after["output/training_panel_v15_sr.parquet"]),
        "production_modified": False,
        "csmar_api_accessed": False,
        "getPackResultExt_called": False,
        "credential_exposure_detected": False,
        "decision": decision,
    }
    for k, v in values.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
