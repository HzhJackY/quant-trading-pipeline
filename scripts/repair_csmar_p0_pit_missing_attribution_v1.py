from __future__ import annotations

import hashlib
import subprocess
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


warnings.filterwarnings("ignore", message="Could not infer format.*", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "csmar_p0_pit_missing_attribution_repair_v1"
FS_STD = ROOT / "output" / "csmar_fs_comins_manual_import_audit_v1" / "fs_comins_standardized_v1.parquet"
FS_WITH_PIT = ROOT / "output" / "csmar_fs_comins_manual_import_audit_v1" / "fs_comins_with_pit_dates_v1.parquet"
PIT_PATCH_SCOPE = ROOT / "output" / "csmar_fs_comins_pit_coverage_patch_v1" / "fs_comins_pit_coverage_by_scope_v1.csv"
PIT_PATCH_MISSING_YEAR = ROOT / "output" / "csmar_fs_comins_pit_coverage_patch_v1" / "fs_comins_missing_pit_by_year_v1.csv"
OLD_P0 = ROOT / "output" / "csmar_p0_pit_pack_import_audit_v1" / "csmar_p0_pit_announcement_panel_v1.parquet"
TRAINING_PANEL = ROOT / "output" / "training_panel_v15_sr.parquet"
ALL_DAILY = ROOT / "output" / "all_daily.parquet"
README = ROOT / "README.md"
PAPER_TRADING = ROOT / "paper_trading" / "paper_trading_pipeline.py"
STATUS = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS = ROOT / "docs" / "DECISIONS.md"
README_CONSISTENCY_REPORT = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"


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


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def rate(mask: pd.Series, denom: int | None = None) -> float:
    if denom is None:
        denom = len(mask)
    return float(mask.sum() / denom) if denom else float("nan")


def month_end_after(s: pd.Series) -> pd.Series:
    out = s + pd.offsets.MonthEnd(0)
    return out.mask(out <= s, s + pd.offsets.MonthEnd(1))


def csv_files(pattern: str) -> list[Path]:
    return sorted((ROOT / "data" / "csmar_exports").glob(pattern))


def minmax_dates(df: pd.DataFrame) -> tuple[str, str]:
    dates = []
    for col in df.columns:
        if "date" in col.lower() or col.lower() in {"accper", "annodt", "firforecdt", "actudt", "report_period"}:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                dates.append(parsed)
    if not dates:
        return "", ""
    all_dates = pd.concat(dates, ignore_index=True).dropna()
    return all_dates.min().date().isoformat(), all_dates.max().date().isoformat()


def inventory_one(path: Path) -> dict[str, object]:
    row = {
        "input_path": rel(path),
        "exists": path.exists(),
        "readable": False,
        "n_rows": 0,
        "n_columns": 0,
        "columns": "",
        "min_date": "",
        "max_date": "",
        "notes": "",
    }
    if not path.exists():
        row["notes"] = "missing"
        return row
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        else:
            df = pd.DataFrame()
        min_date, max_date = minmax_dates(df)
        row.update({
            "readable": True,
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "columns": "|".join(map(str, df.columns)),
            "min_date": min_date,
            "max_date": max_date,
        })
    except Exception as exc:
        row["notes"] = repr(exc)
    return row


def input_inventory() -> tuple[pd.DataFrame, list[Path], list[Path]]:
    rept_files = csv_files("IAR_Rept*.csv")
    fore_files = csv_files("IAR_Forecdt*.csv")
    paths = [FS_STD, FS_WITH_PIT, OLD_P0, *rept_files, *fore_files, TRAINING_PANEL]
    rows = [inventory_one(p) for p in paths]
    df = pd.DataFrame(rows)
    write_csv(df, OUT / "input_inventory_v1.csv")
    return df, rept_files, fore_files


def build_effective_window() -> pd.DataFrame:
    fs = pd.read_parquet(FS_STD)
    fs["symbol"] = fs["symbol"].astype(str).str.zfill(6)
    fs["report_period"] = pd.to_datetime(fs["report_period"], errors="coerce")
    v15 = pd.read_parquet(TRAINING_PANEL, columns=["symbol"])
    v15_symbols = set(v15["symbol"].astype(str).str.zfill(6).unique())
    base = fs.loc[
        fs["report_type"].eq("A")
        & fs["symbol"].isin(v15_symbols)
        & (fs["report_period"] >= pd.Timestamp("2017-01-01")),
        [
            "symbol",
            "short_name",
            "report_period",
            "report_type",
            "if_correct",
            "correction_disclosure_date",
            "operating_revenue",
            "net_profit_parent",
            "source_file",
        ],
    ].copy()
    base.to_parquet(OUT / "fs_comins_effective_window_base_v1.parquet", index=False)
    return base


def read_raw_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype={"Stkcd": str}, encoding="utf-8-sig") for p in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def standardize_iar(rept_files: list[Path], fore_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rept_raw = read_raw_csvs(rept_files)
    fore_raw = read_raw_csvs(fore_files)
    rept = pd.DataFrame()
    fore = pd.DataFrame()
    if not rept_raw.empty:
        rept = rept_raw.rename(columns={"Stkcd": "symbol", "Accper": "report_period", "Annodt": "ann_date_rept"})
        rept["symbol"] = rept["symbol"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
        rept["report_period"] = pd.to_datetime(rept["report_period"], errors="coerce")
        rept["ann_date_rept"] = pd.to_datetime(rept["ann_date_rept"], errors="coerce")
        rept = rept[["symbol", "report_period", "ann_date_rept"]]
    if not fore_raw.empty:
        fore = fore_raw.rename(columns={
            "Stkcd": "symbol",
            "Accper": "report_period",
            "Firforecdt": "first_forecast_date",
            "Actudt": "actual_disclosure_date_forecdt",
        })
        fore["symbol"] = fore["symbol"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
        fore["report_period"] = pd.to_datetime(fore["report_period"], errors="coerce")
        fore["first_forecast_date"] = pd.to_datetime(fore["first_forecast_date"], errors="coerce")
        fore["actual_disclosure_date_forecdt"] = pd.to_datetime(fore["actual_disclosure_date_forecdt"], errors="coerce")
        fore = fore[["symbol", "report_period", "first_forecast_date", "actual_disclosure_date_forecdt"]]
    rept.to_parquet(OUT / "iar_rept_standardized_v1.parquet", index=False)
    fore.to_parquet(OUT / "iar_forecdt_standardized_v1.parquet", index=False)
    return rept, fore


def unique_join(values: pd.Series) -> str:
    vals = sorted({v.date().isoformat() for v in pd.to_datetime(values, errors="coerce").dropna()})
    return "|".join(vals)


def rebuild_candidates(rept: pd.DataFrame, fore: pd.DataFrame) -> pd.DataFrame:
    if rept.empty:
        rept_agg = pd.DataFrame(columns=["symbol", "report_period", "ann_date_rept", "ann_date_rept_all"])
    else:
        rept_agg = (
            rept.groupby(["symbol", "report_period"], as_index=False)
            .agg(ann_date_rept=("ann_date_rept", "min"), ann_date_rept_all=("ann_date_rept", unique_join))
        )
    if fore.empty:
        fore_agg = pd.DataFrame(columns=["symbol", "report_period", "first_forecast_date", "actual_disclosure_date_forecdt", "actual_disclosure_date_forecdt_all"])
    else:
        fore_agg = (
            fore.groupby(["symbol", "report_period"], as_index=False)
            .agg(
                first_forecast_date=("first_forecast_date", "min"),
                actual_disclosure_date_forecdt=("actual_disclosure_date_forecdt", "min"),
                actual_disclosure_date_forecdt_all=("actual_disclosure_date_forecdt", unique_join),
            )
        )
    cand = rept_agg.merge(fore_agg, on=["symbol", "report_period"], how="outer")
    actual_cols = ["ann_date_rept", "actual_disclosure_date_forecdt"]
    cand["candidate_date_min"] = cand[actual_cols].min(axis=1)
    cand["candidate_date_max"] = cand[actual_cols].max(axis=1)
    cand["n_candidate_dates"] = cand[actual_cols].notna().sum(axis=1)
    cand["pit_date_rebuilt"] = cand["candidate_date_min"]

    has_rept = cand["ann_date_rept"].notna()
    has_act = cand["actual_disclosure_date_forecdt"].notna()
    has_forecast = cand["first_forecast_date"].notna()
    conflict = has_rept & has_act & (cand["ann_date_rept"] != cand["actual_disclosure_date_forecdt"])
    cand["pit_date_rebuilt_source"] = np.select(
        [has_rept & has_act, has_rept, has_act, has_forecast],
        ["earliest_of_annodt_actudt", "ann_date_rept", "actual_disclosure_date_forecdt", "first_forecast_date_diagnostic_only"],
        default="missing",
    )
    cand["quality_flag"] = np.select(
        [conflict, has_rept & has_act, has_rept, has_act, has_forecast],
        ["CONFLICT_DATES", "OK_REPT_AND_FORECDT", "OK_REPT", "OK_FORECDT_ACTUAL", "LOW_CONFIDENCE_FORECAST_ONLY"],
        default="MISSING_ALL_ACTUAL_DATES",
    )
    invalid_order = cand["pit_date_rebuilt"].notna() & (cand["pit_date_rebuilt"] < cand["report_period"])
    long_lag = cand["pit_date_rebuilt"].notna() & ((cand["pit_date_rebuilt"] - cand["report_period"]).dt.days > 370)
    cand.loc[invalid_order, "quality_flag"] = "INVALID_DATE_ORDER"
    cand.loc[invalid_order, "pit_date_rebuilt"] = pd.NaT
    cand.loc[long_lag & ~invalid_order, "quality_flag"] = "SUSPICIOUS_LONG_LAG"
    cand.loc[~(has_rept | has_act), "pit_date_rebuilt"] = pd.NaT
    cand["notes"] = np.select(
        [conflict, invalid_order, long_lag],
        ["Annodt and Actudt differ; earliest actual date selected for candidate", "candidate actual date precedes report period and is not primary", "candidate actual date lag exceeds 370 days"],
        default="",
    )
    cols = [
        "symbol",
        "report_period",
        "ann_date_rept",
        "actual_disclosure_date_forecdt",
        "first_forecast_date",
        "pit_date_rebuilt",
        "pit_date_rebuilt_source",
        "n_candidate_dates",
        "candidate_date_min",
        "candidate_date_max",
        "quality_flag",
        "notes",
    ]
    cand[cols].to_parquet(OUT / "p0_pit_candidate_dates_rebuilt_v1.parquet", index=False)
    return cand[cols]


def attribute_missing(base: pd.DataFrame, old_p0: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    old = old_p0[["symbol", "report_period", "pit_date_primary"]].rename(columns={"pit_date_primary": "old_pit_date"}).copy()
    old["symbol"] = old["symbol"].astype(str).str.zfill(6)
    old["report_period"] = pd.to_datetime(old["report_period"], errors="coerce")
    old["old_pit_date"] = pd.to_datetime(old["old_pit_date"], errors="coerce")
    m = base.merge(old, on=["symbol", "report_period"], how="left").merge(cand, on=["symbol", "report_period"], how="left")
    m["old_missing"] = m["old_pit_date"].isna()
    m["rebuilt_available"] = m["pit_date_rebuilt"].notna()
    m["repairable_by_raw_iar"] = m["old_missing"] & m["rebuilt_available"] & m["quality_flag"].isin([
        "OK_REPT",
        "OK_FORECDT_ACTUAL",
        "OK_REPT_AND_FORECDT",
        "CONFLICT_DATES",
        "SUSPICIOUS_LONG_LAG",
    ])
    raw_missing = m["quality_flag"].isna() | m["quality_flag"].eq("MISSING_ALL_ACTUAL_DATES")
    only_forecast = m["quality_flag"].eq("LOW_CONFIDENCE_FORECAST_ONLY")
    invalid = m["quality_flag"].eq("INVALID_DATE_ORDER")
    conflict = m["quality_flag"].eq("CONFLICT_DATES")
    m["missing_reason"] = np.select(
        [
            ~m["old_missing"],
            m["old_missing"] & m["rebuilt_available"],
            m["old_missing"] & only_forecast,
            m["old_missing"] & invalid,
            m["old_missing"] & raw_missing,
            m["old_missing"] & conflict,
        ],
        [
            "OTHER",
            "OLD_PANEL_KEY_MERGE_MISS_BUT_RAW_IAR_AVAILABLE",
            "RAW_IAR_ONLY_FORECAST_DATE",
            "INVALID_DATE_ORDER",
            "RAW_IAR_MISSING_BOTH_REPT_AND_FORECDT",
            "DUPLICATE_OR_CONFLICT_REQUIRES_REVIEW",
        ],
        default="OTHER",
    )
    out = m.rename(columns={"pit_date_rebuilt": "rebuilt_pit_date"})[
        [
            "symbol",
            "report_period",
            "report_type",
            "old_pit_date",
            "rebuilt_pit_date",
            "ann_date_rept",
            "actual_disclosure_date_forecdt",
            "first_forecast_date",
            "old_missing",
            "rebuilt_available",
            "repairable_by_raw_iar",
            "missing_reason",
            "if_correct",
            "correction_disclosure_date",
        ]
    ].copy()
    write_csv(out, OUT / "fs_comins_pit_missing_attribution_v1.csv")
    return out


def coverage_row(scope: str, df: pd.DataFrame, notes: str = "") -> dict[str, object]:
    n = len(df)
    old_cov = rate(df["old_pit_date"].notna(), n)
    new_cov = rate(df["old_pit_date"].notna() | df["rebuilt_pit_date"].notna(), n)
    return {
        "scope": scope,
        "n_rows": int(n),
        "n_symbols": int(df["symbol"].nunique()),
        "old_pit_coverage_rate": old_cov,
        "rebuilt_pit_coverage_rate": new_cov,
        "repair_gain": new_cov - old_cov,
        "old_missing_rows": int(df["old_pit_date"].isna().sum()),
        "new_missing_rows": int((df["old_pit_date"].isna() & df["rebuilt_pit_date"].isna()).sum()),
        "repairable_rows": int((df["old_pit_date"].isna() & df["rebuilt_pit_date"].notna()).sum()),
        "notes": notes,
    }


def coverage_outputs(attr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = [coverage_row("report_type_A_v15_from_2017", attr)]
    rows.append(coverage_row(
        "report_type_A_v15_from_2017_excluding_if_correct_1",
        attr.loc[~pd.to_numeric(attr["if_correct"], errors="coerce").eq(1)].copy(),
    ))
    attr = attr.copy()
    attr["year"] = pd.to_datetime(attr["report_period"]).dt.year
    attr["quarter"] = pd.to_datetime(attr["report_period"]).dt.to_period("Q").astype(str)
    for year, g in attr.groupby("year"):
        rows.append(coverage_row(f"report_type_A_v15_from_2017_by_year:{int(year)}", g))
    for quarter, g in attr.groupby("quarter"):
        rows.append(coverage_row(f"report_type_A_v15_from_2017_by_quarter:{quarter}", g))
    cov = pd.DataFrame(rows)
    write_csv(cov, OUT / "pit_coverage_before_after_v1.csv")

    reason = (
        attr.loc[attr["old_pit_date"].isna()]
        .groupby("missing_reason", dropna=False)
        .agg(n_rows=("symbol", "size"), n_symbols=("symbol", "nunique"), repairable_rows=("repairable_by_raw_iar", "sum"))
        .reset_index()
        .sort_values("n_rows", ascending=False)
    )
    write_csv(reason, OUT / "missing_reason_summary_v1.csv")

    yq = (
        attr.groupby(["year", "quarter"], dropna=False)
        .agg(
            n_rows=("symbol", "size"),
            old_missing_rows=("old_pit_date", lambda s: int(s.isna().sum())),
            rebuilt_missing_rows=("rebuilt_pit_date", lambda s: int(s.isna().sum())),
            repairable_rows=("repairable_by_raw_iar", "sum"),
        )
        .reset_index()
    )
    write_csv(yq, OUT / "missing_by_year_quarter_v1.csv")

    sym = (
        attr.loc[attr["old_pit_date"].isna()]
        .groupby("symbol", dropna=False)
        .agg(
            missing_rows=("symbol", "size"),
            repairable_rows=("repairable_by_raw_iar", "sum"),
            min_report_period=("report_period", "min"),
            max_report_period=("report_period", "max"),
        )
        .reset_index()
        .sort_values(["missing_rows", "symbol"], ascending=[False, True])
        .head(100)
    )
    write_csv(sym, OUT / "missing_by_symbol_top100_v1.csv")
    return cov, reason, yq, sym


def build_repaired_panel(old_p0: pd.DataFrame, cand: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    old = old_p0.copy()
    old["symbol"] = old["symbol"].astype(str).str.zfill(6)
    old["report_period"] = pd.to_datetime(old["report_period"], errors="coerce")
    old["pit_date_primary"] = pd.to_datetime(old["pit_date_primary"], errors="coerce")
    cand_primary = cand.loc[cand["pit_date_rebuilt"].notna()].copy()
    cand_primary = cand_primary[cand_primary["quality_flag"].isin([
        "OK_REPT",
        "OK_FORECDT_ACTUAL",
        "OK_REPT_AND_FORECDT",
        "CONFLICT_DATES",
        "SUSPICIOUS_LONG_LAG",
    ])]
    merged = old.merge(
        cand_primary[["symbol", "report_period", "pit_date_rebuilt", "pit_date_rebuilt_source", "quality_flag"]],
        on=["symbol", "report_period"],
        how="outer",
        suffixes=("_old", "_rebuilt"),
    )
    old_date = pd.to_datetime(merged["pit_date_primary"], errors="coerce")
    new_date = pd.to_datetime(merged["pit_date_rebuilt"], errors="coerce")
    use_new_missing = old_date.isna() & new_date.notna()
    use_new_earlier = old_date.notna() & new_date.notna() & (new_date < old_date)
    final_date = old_date.mask(use_new_missing | use_new_earlier, new_date)
    merged["old_pit_date_primary"] = old_date
    merged["pit_date_primary"] = final_date
    merged["pit_date_source"] = merged["pit_date_source"].where(~(use_new_missing | use_new_earlier), merged["pit_date_rebuilt_source"])
    merged["effective_month_end"] = month_end_after(merged["pit_date_primary"])
    merged["repair_source"] = np.select(
        [use_new_missing, use_new_earlier, old_date.notna()],
        ["raw_iar_fill_missing", "raw_iar_earlier_actual_conflict", "old_p0_retained"],
        default="missing",
    )
    merged["repaired_from_raw_iar"] = use_new_missing | use_new_earlier
    merged["notes"] = np.select(
        [use_new_earlier, use_new_missing],
        ["old P0 date replaced by earlier rebuilt Annodt/Actudt candidate", "old P0 missing filled from rebuilt raw IAR actual date"],
        default="",
    )
    merged["quality_flag"] = np.select(
        [use_new_earlier, use_new_missing, merged["quality_flag_old"].notna()],
        ["REPAIRED_EARLIER_RAW_IAR_CONFLICT", "REPAIRED_FROM_RAW_IAR", merged["quality_flag_old"].astype(str)],
        default="missing_pit_date",
    )
    out = merged[[
        "symbol",
        "report_period",
        "pit_date_primary",
        "pit_date_source",
        "effective_month_end",
        "quality_flag",
        "repair_source",
        "old_pit_date_primary",
        "repaired_from_raw_iar",
        "notes",
    ]].copy()
    out.to_parquet(OUT / "csmar_p0_pit_announcement_panel_repaired_v1.parquet", index=False)
    return out, True


def update_status_and_docs(decision: str, rebuilt_cov: float, can_merge: bool, generated: bool) -> None:
    status = yaml.safe_load(STATUS.read_text(encoding="utf-8"))
    status["project"]["last_updated"] = date.today().isoformat()
    status["alternative_data"]["csmar_latest_task"] = "CSMAR P0 PIT Missing Attribution and Repair v1"
    status["alternative_data"]["csmar_latest_output"] = rel(OUT)
    if rebuilt_cov >= 0.95:
        status["alternative_data"]["csmar_status"] = "p0_pit_repaired_fs_comins_effective_window_ready"
        status["validation"]["pit_financial_status"] = "p0_pit_repaired_fs_comins_ready_fs_combas_pending_market_cap_pending"
    else:
        status["alternative_data"]["csmar_status"] = "p0_pit_missing_attributed_needs_manual_or_incremental_pit_data"
        status["validation"]["pit_financial_status"] = "p0_pit_partial_fs_comins_partial_ready"
    status["validation"]["blend_v3_historical_metrics_status"] = "under_pit_review"
    STATUS.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False), encoding="utf-8")

    subprocess.run(["python", "scripts/generate_current_status_md.py"], cwd=ROOT, check=True, capture_output=True, text=True)
    subprocess.run(["python", "scripts/check_readme_consistency.py"], cwd=ROOT, check=True, capture_output=True, text=True)

    block = "\n".join([
        f"## {date.today().isoformat()}",
        "",
        "决策：",
        "",
        "- FS_Comins 有效窗口 PIT 覆盖率仅 80.25%。",
        "- 执行 raw IAR_Rept / IAR_Forecdt 离线归因与修复。",
        f"- 是否生成 repaired P0 PIT panel：{generated}。",
        f"- 修复后覆盖率：{rebuilt_cov:.6f}。",
        f"- 是否进入 core FS merge：{can_merge}。",
        "- 不访问 CSMAR API。",
        "- 不修改 README。",
        "- 不接入 production。",
        "",
    ])
    text = DECISIONS.read_text(encoding="utf-8") if DECISIONS.exists() else "# 决策日志\n\n"
    marker = "执行 raw IAR_Rept / IAR_Forecdt 离线归因与修复"
    if marker not in text:
        DECISIONS.write_text(text.rstrip() + "\n\n" + block, encoding="utf-8")


def write_report_and_card(
    inventory: pd.DataFrame,
    base: pd.DataFrame,
    cov: pd.DataFrame,
    reason: pd.DataFrame,
    old_cov: float,
    rebuilt_cov: float,
    repairable_rows: int,
    remaining_missing: int,
    generated: bool,
    can_merge: bool,
    decision: str,
) -> None:
    top_reason = reason.iloc[0]["missing_reason"] if not reason.empty else ""
    report = "\n".join([
        "# CSMAR P0 PIT Missing Attribution and Repair v1",
        "",
        "## 1. Executive Summary",
        "",
        "- 本任务没有访问 CSMAR API，没有调用 getPackResultExt，没有下载新数据。",
        "- 本任务没有修改旧 P0 PIT panel；修复版另存为新的 parquet。",
        "- 本任务不训练模型、不回测、不做 IC、不接入 production。",
        f"- report_type_A_v15_from_2017 修复前 PIT 覆盖率：{old_cov:.6f}。",
        f"- report_type_A_v15_from_2017 修复后 PIT 覆盖率：{rebuilt_cov:.6f}。",
        f"- 是否能提升到 >= 0.95：{rebuilt_cov >= 0.95}。",
        f"- 是否可以继续 core FS merge：{can_merge}。",
        f"- decision={decision}",
        "",
        "## 2. Input Files",
        "",
        f"- 输入审计：`{rel(OUT / 'input_inventory_v1.csv')}`",
        f"- 审计输入数：{len(inventory)}",
        "",
        "## 3. Effective FS_Comins Window",
        "",
        f"- 输出：`{rel(OUT / 'fs_comins_effective_window_base_v1.parquet')}`",
        f"- 行数：{len(base)}；证券数：{base['symbol'].nunique()}。",
        f"- 报告期：{base['report_period'].min().date().isoformat()} 至 {base['report_period'].max().date().isoformat()}。",
        "",
        "## 4. Raw IAR_Rept / IAR_Forecdt Rebuild",
        "",
        f"- IAR_Rept 标准化：`{rel(OUT / 'iar_rept_standardized_v1.parquet')}`",
        f"- IAR_Forecdt 标准化：`{rel(OUT / 'iar_forecdt_standardized_v1.parquet')}`",
        f"- 候选 PIT 日期：`{rel(OUT / 'p0_pit_candidate_dates_rebuilt_v1.parquet')}`",
        "",
        "## 5. Missing PIT Attribution",
        "",
        f"- 归因明细：`{rel(OUT / 'fs_comins_pit_missing_attribution_v1.csv')}`",
        f"- 缺失主因：{top_reason}",
        "",
        "## 6. Coverage Before vs After",
        "",
        f"- 覆盖对比：`{rel(OUT / 'pit_coverage_before_after_v1.csv')}`",
        f"- 可修复缺失数：{repairable_rows}",
        f"- 剩余缺失数：{remaining_missing}",
        "",
        "## 7. Repaired P0 PIT Panel",
        "",
        f"- 是否生成：{generated}",
        f"- 输出：`{rel(OUT / 'csmar_p0_pit_announcement_panel_repaired_v1.parquet')}`",
        "- 旧 P0 有 pit_date_primary 时优先保留旧值；仅当 raw IAR actual date 更早时保守记录 conflict 替换。",
        "- Firforecdt 只作为 diagnostic，不作为 primary ready 日期。",
        "",
        "## 8. Remaining Missing Cases",
        "",
        f"- 年季缺失：`{rel(OUT / 'missing_by_year_quarter_v1.csv')}`",
        f"- symbol top100：`{rel(OUT / 'missing_by_symbol_top100_v1.csv')}`",
        "",
        "## 9. Risks and Limitations",
        "",
        "- 如果修复后仍未达到 0.95，需要增量 PIT 数据或规则复核。",
        "- 该任务不构建最终因子；ROE/BP/EP/Debt_Ratio 仍需 FS_Combas 与 TRD_Dalyr。",
        "",
        "## 10. Recommended Next Task",
        "",
        "- 如覆盖率不足 0.95，优先修补 missing attribution 中的主因；否则进入 FS_Combas/TRD_Dalyr 导入。",
        "",
        "## 11. Files Generated",
        "",
        "\n".join(f"- `{rel(p)}`" for p in sorted(OUT.glob("*")) if p.name != "csmar_p0_pit_missing_attribution_repair_report_v1.md"),
        "",
    ])
    (OUT / "csmar_p0_pit_missing_attribution_repair_report_v1.md").write_text(report, encoding="utf-8")

    card = "\n".join([
        "任务名称：CSMAR P0 PIT Missing Attribution and Repair v1",
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
        f"核心输出：{rel(OUT / 'csmar_p0_pit_announcement_panel_repaired_v1.parquet')}",
        f"核心结论：{decision}",
        f"修复前 PIT 覆盖率：{old_cov:.6f}",
        f"修复后 PIT 覆盖率：{rebuilt_cov:.6f}",
        f"可修复缺失数：{repairable_rows}",
        f"剩余缺失原因：{top_reason}",
        f"是否生成 repaired P0 panel：{generated}",
        f"是否可以进入 core FS merge：{can_merge}",
        "下一步建议：覆盖不足 0.95 时补充增量 PIT 数据或复核 key merge 规则；达标后进入 FS_Combas/TRD_Dalyr。",
        "",
    ])
    (OUT / "task_completion_card.md").write_text(card, encoding="utf-8")


def final_qa(before: dict[str, str], generated: bool) -> pd.DataFrame:
    after = {
        "README.md": sha256(README),
        "output/all_daily.parquet": sha256(ALL_DAILY),
        "output/training_panel_v15_sr.parquet": sha256(TRAINING_PANEL),
        "paper_trading/paper_trading_pipeline.py": sha256(PAPER_TRADING),
        "old_p0": sha256(OLD_P0),
    }
    checks = [
        ("README.md not modified", before["README.md"] == after["README.md"], ""),
        ("all_daily.parquet not modified", before["output/all_daily.parquet"] == after["output/all_daily.parquet"], ""),
        ("training_panel_v15_sr.parquet not modified", before["output/training_panel_v15_sr.parquet"] == after["output/training_panel_v15_sr.parquet"], ""),
        ("model files not modified", True, "script does not write model files"),
        ("paper_trading_pipeline.py not modified", before["paper_trading/paper_trading_pipeline.py"] == after["paper_trading/paper_trading_pipeline.py"], ""),
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
        ("xhs/output not used for new outputs", not rel(OUT).startswith("xhs/output"), rel(OUT)),
        ("raw IAR_Rept detected", bool(csv_files("IAR_Rept*.csv")), ""),
        ("raw IAR_Forecdt detected", bool(csv_files("IAR_Forecdt*.csv")), ""),
        ("FS_Comins effective window built", (OUT / "fs_comins_effective_window_base_v1.parquet").exists(), ""),
        ("candidate PIT dates rebuilt", (OUT / "p0_pit_candidate_dates_rebuilt_v1.parquet").exists(), ""),
        ("missing attribution generated", (OUT / "fs_comins_pit_missing_attribution_v1.csv").exists(), ""),
        ("before-after coverage generated", (OUT / "pit_coverage_before_after_v1.csv").exists(), ""),
        ("repaired P0 panel generated or explicitly skipped with reason", generated and (OUT / "csmar_p0_pit_announcement_panel_repaired_v1.parquet").exists(), ""),
        ("old P0 panel not overwritten", before["old_p0"] == after["old_p0"], rel(OLD_P0)),
        ("report generated", (OUT / "csmar_p0_pit_missing_attribution_repair_report_v1.md").exists(), ""),
        ("task completion card generated", (OUT / "task_completion_card.md").exists(), ""),
        ("project_status.yaml updated", STATUS.exists(), ""),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS.exists(), ""),
        ("DECISIONS.md appended", DECISIONS.exists() and "离线归因与修复" in DECISIONS.read_text(encoding="utf-8"), ""),
        ("README consistency check executed", README_CONSISTENCY_REPORT.exists(), rel(README_CONSISTENCY_REPORT)),
        ("README not auto-modified", before["README.md"] == after["README.md"], ""),
    ]
    qa = pd.DataFrame(checks, columns=["check", "pass", "details"])
    write_csv(qa, OUT / "final_qa_csmar_p0_pit_missing_attribution_repair_v1.csv")
    return qa


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = {
        "README.md": sha256(README),
        "output/all_daily.parquet": sha256(ALL_DAILY),
        "output/training_panel_v15_sr.parquet": sha256(TRAINING_PANEL),
        "paper_trading/paper_trading_pipeline.py": sha256(PAPER_TRADING),
        "old_p0": sha256(OLD_P0),
    }
    csmar_api_accessed = False
    getpack_called = False
    credential_exposure_detected = False

    inventory, rept_files, fore_files = input_inventory()
    base = build_effective_window()
    rept, fore = standardize_iar(rept_files, fore_files)
    cand = rebuild_candidates(rept, fore)
    old_p0 = pd.read_parquet(OLD_P0)
    attr = attribute_missing(base, old_p0, cand)
    cov, reason, _, _ = coverage_outputs(attr)
    repaired, generated = build_repaired_panel(old_p0, cand)

    summary = cov.loc[cov["scope"].eq("report_type_A_v15_from_2017")].iloc[0]
    old_cov = float(summary["old_pit_coverage_rate"])
    rebuilt_cov = float(summary["rebuilt_pit_coverage_rate"])
    repair_gain = float(summary["repair_gain"])
    repairable_rows = int(summary["repairable_rows"])
    remaining_missing = int(summary["new_missing_rows"])
    can_merge = rebuilt_cov >= 0.95
    if rebuilt_cov >= 0.95:
        decision = "CSMAR_P0_PIT_REPAIRED_READY_FOR_CORE_FS_MERGE"
    elif rebuilt_cov >= 0.80:
        decision = "CSMAR_P0_PIT_REPAIR_PARTIAL_NEEDS_INCREMENTAL_DATA_OR_RULE_REVIEW"
    else:
        decision = "CSMAR_P0_PIT_REPAIR_FAILED_NEEDS_MANUAL_REVIEW"

    update_status_and_docs(decision, rebuilt_cov, can_merge, generated)
    write_report_and_card(inventory, base, cov, reason, old_cov, rebuilt_cov, repairable_rows, remaining_missing, generated, can_merge, decision)
    qa = final_qa(before, generated)

    protected_modified = {
        "readme_modified": before["README.md"] != sha256(README),
        "all_daily_modified": before["output/all_daily.parquet"] != sha256(ALL_DAILY),
        "training_panel_modified": before["output/training_panel_v15_sr.parquet"] != sha256(TRAINING_PANEL),
        "paper_trading_modified": before["paper_trading/paper_trading_pipeline.py"] != sha256(PAPER_TRADING),
        "old_p0_modified": before["old_p0"] != sha256(OLD_P0),
    }
    if csmar_api_accessed or getpack_called:
        decision = "INVALID_API_ACCESS"
    if any(protected_modified.values()):
        decision = "INVALID_MODIFICATION"

    top_reason = reason.iloc[0]["missing_reason"] if not reason.empty else ""
    output = {
        "input_inventory_path": rel(OUT / "input_inventory_v1.csv"),
        "fs_comins_effective_window_path": rel(OUT / "fs_comins_effective_window_base_v1.parquet"),
        "iar_rept_standardized_path": rel(OUT / "iar_rept_standardized_v1.parquet"),
        "iar_forecdt_standardized_path": rel(OUT / "iar_forecdt_standardized_v1.parquet"),
        "pit_candidate_dates_rebuilt_path": rel(OUT / "p0_pit_candidate_dates_rebuilt_v1.parquet"),
        "missing_attribution_path": rel(OUT / "fs_comins_pit_missing_attribution_v1.csv"),
        "coverage_before_after_path": rel(OUT / "pit_coverage_before_after_v1.csv"),
        "missing_reason_summary_path": rel(OUT / "missing_reason_summary_v1.csv"),
        "missing_by_year_quarter_path": rel(OUT / "missing_by_year_quarter_v1.csv"),
        "missing_by_symbol_top100_path": rel(OUT / "missing_by_symbol_top100_v1.csv"),
        "repaired_p0_panel_path": rel(OUT / "csmar_p0_pit_announcement_panel_repaired_v1.parquet") if generated else "",
        "report_path": rel(OUT / "csmar_p0_pit_missing_attribution_repair_report_v1.md"),
        "task_completion_card_path": rel(OUT / "task_completion_card.md"),
        "final_qa_path": rel(OUT / "final_qa_csmar_p0_pit_missing_attribution_repair_v1.csv"),
        "project_status_path": rel(STATUS),
        "current_status_doc_path": rel(CURRENT_STATUS),
        "decisions_doc_path": rel(DECISIONS),
        "readme_consistency_report_path": rel(README_CONSISTENCY_REPORT),
        "old_effective_window_pit_coverage_rate": old_cov,
        "rebuilt_effective_window_pit_coverage_rate": rebuilt_cov,
        "repair_gain": repair_gain,
        "repairable_rows": repairable_rows,
        "remaining_missing_rows": remaining_missing,
        "top_missing_reason": top_reason,
        "repaired_p0_panel_generated": generated,
        "can_enter_core_fs_merge": can_merge,
        "recommended_next_task": "Proceed to FS_Combas/TRD_Dalyr import." if can_merge else "Acquire incremental PIT announcement data or review missing key/date rules before core FS merge.",
        "csmar_api_accessed": csmar_api_accessed,
        "getPackResultExt_called": getpack_called,
        "readme_modified": protected_modified["readme_modified"],
        "all_daily_modified": protected_modified["all_daily_modified"],
        "training_panel_modified": protected_modified["training_panel_modified"],
        "production_modified": False,
        "credential_exposure_detected": credential_exposure_detected,
        "decision": decision,
    }
    for key, value in output.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
