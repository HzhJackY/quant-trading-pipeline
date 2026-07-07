from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "csmar_pit_clean_ttm_buffer_universe_lock_patch_v1"
ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME
OUT_DIR = ROOT / "output" / "csmar_pit_clean_core_financial_factors_v3"

V1_FACTOR = ROOT / "output" / "csmar_pit_clean_core_financial_factors_v1" / "pit_clean_core_financial_factors_monthly_v1.parquet"
V2_FACTOR = ROOT / "output" / "csmar_pit_clean_core_financial_factors_v2" / "pit_clean_core_financial_factors_monthly_v2.parquet"
STRICT_V1 = ROOT / "output" / "csmar_pit_scope_freeze_strict_core_fs_panel_v1" / "strict_core_fs_monthly_asof_panel_v1.parquet"

TARGET_AUDIT = OUT_DIR / "target_universe_audit_v1.csv"
V2_DIAG = OUT_DIR / "v2_universe_expansion_diagnostic_v1.csv"
V2_EXTRA_SAMPLE = OUT_DIR / "v2_extra_symbols_sample_v1.csv"
V3_PANEL = OUT_DIR / "pit_clean_core_financial_factors_monthly_v3.parquet"
V3_SAMPLE = OUT_DIR / "pit_clean_core_financial_factors_monthly_sample_v3.csv"
COMPARISON = OUT_DIR / "v1_v2_v3_comparison_v1.csv"
COVERAGE_AUDIT = OUT_DIR / "factor_coverage_audit_by_month_v3.csv"
DIST_AUDIT = OUT_DIR / "factor_distribution_audit_v3.csv"
REPORT = OUT_DIR / "csmar_pit_clean_ttm_buffer_universe_lock_patch_report_v1.md"
CARD = OUT_DIR / "task_completion_card.md"
FINAL_QA = OUT_DIR / "final_qa_csmar_pit_clean_ttm_buffer_universe_lock_patch_v1.csv"
FINAL_QA_ALIAS = OUT_DIR / "final_qa.csv"
TERMINAL_SUMMARY = OUT_DIR / "terminal_summary.json"

START_TS = datetime.now().isoformat(timespec="seconds")
FORBIDDEN_FILES = [
    ROOT / "README.md",
    ROOT / "output" / "all_daily.parquet",
    ROOT / "output" / "training_panel_v15_sr.parquet",
]
FORBIDDEN_DIRS = [
    ROOT / "output" / "csmar_pit_clean_core_financial_factors_v1",
    ROOT / "output" / "csmar_pit_clean_core_financial_factors_v2",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def snapshot_mtimes(paths: list[Path], dirs: list[Path]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for path in paths:
        result[str(path)] = path.stat().st_mtime if path.exists() else None
    for directory in dirs:
        if directory.exists():
            for child in directory.iterdir():
                if child.is_file():
                    result[str(child)] = child.stat().st_mtime
    return result


def mtimes_changed(before: dict[str, float | None]) -> dict[str, bool]:
    changed: dict[str, bool] = {}
    for raw_path, before_mtime in before.items():
        path = Path(raw_path)
        after_mtime = path.stat().st_mtime if path.exists() else None
        changed[raw_path] = after_mtime != before_mtime
    return changed


def update_run_state(stage: str, completed: list[str], processing: list[str], outputs: list[Path], next_step: str) -> None:
    text = [
        "# RUN_STATE",
        "",
        "- 当前任务名称：CSMAR PIT-Clean TTM Buffer Universe Lock Patch v1",
        f"- 开始时间：{START_TS} Asia/Shanghai",
        f"- 当前阶段：{stage}",
        "- 已完成步骤：",
    ]
    text.extend(f"  - {item}" for item in completed)
    text.append("- 正在处理的文件：")
    text.extend(f"  - {item}" for item in processing)
    text.append("- 已生成输出：")
    text.extend(f"  - {rel(item)}" for item in outputs)
    text.extend(
        [
            "- 下一步：",
            f"  - {next_step}",
            "- 如果 Codex 崩溃，新的 Codex 应如何继续：",
            "  - 先读取本文件确认阶段",
            "  - 不读取 xlsx、原始日频 CSV，不访问 CSMAR API，不下载数据",
            "  - 只读取任务允许的 parquet 文件",
            "  - 不覆盖 output/csmar_pit_clean_core_financial_factors_v1 或 v2",
        ]
    )
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def add_checkpoint(title: str, lines: list[str]) -> None:
    checkpoint = "\n".join(["", f"## {now()} - {title}", "", *[f"- {line}" for line in lines]]) + "\n"
    with (RUN_DIR / "CHECKPOINTS.md").open("a", encoding="utf-8") as fh:
        fh.write(checkpoint)


def ensure_dirs() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_parquet_cols(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=columns)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype("string")
    for col in ["month_end", "selected_pit_date", "market_cap_trade_date", "selected_report_period"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def metric_rows(rows: list[tuple[str, object, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["metric", "value", "details"])


def summarize_panel(df: pd.DataFrame, target: set[str]) -> dict[str, object]:
    early = df[(df["month_end"] >= pd.Timestamp("2017-01-31")) & (df["month_end"] <= pd.Timestamp("2017-03-31"))]
    dup_count = int(df.duplicated(["symbol", "month_end"]).sum())
    selected_viol = int((df["selected_pit_date"].notna() & (df["selected_pit_date"] > df["month_end"])).sum()) if "selected_pit_date" in df else -1
    cap_viol = int((df["market_cap_trade_date"].notna() & (df["market_cap_trade_date"] > df["month_end"])).sum()) if "market_cap_trade_date" in df else -1
    return {
        "min_month_end": df["month_end"].min().date().isoformat() if len(df) else "",
        "max_month_end": df["month_end"].max().date().isoformat() if len(df) else "",
        "n_rows": int(len(df)),
        "n_symbols": int(df["symbol"].nunique()),
        "n_months": int(df["month_end"].nunique()),
        "n_rows_2017_01_to_2017_03": int(len(early)),
        "n_symbols_2017_01_to_2017_03": int(early["symbol"].nunique()),
        "n_extra_symbols_not_in_target_universe": int(df.loc[~df["symbol"].isin(target), "symbol"].nunique()),
        "selected_pit_date_violation_count": selected_viol,
        "market_cap_date_violation_count": cap_viol,
        "one_row_per_symbol_month": bool(dup_count == 0),
        "ep_median": safe_median(df, "ep_ttm"),
        "bp_median": safe_median(df, "bp"),
        "roe_coverage_mean": coverage_mean(df, "roe_ttm"),
        "ep_coverage_mean": coverage_mean(df, "ep_ttm"),
        "bp_coverage_mean": coverage_mean(df, "bp"),
    }


def safe_median(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    value = df[col].median(skipna=True)
    return None if pd.isna(value) else float(value)


def coverage_mean(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    by_month = df.groupby("month_end", observed=True)[col].apply(lambda s: s.notna().mean())
    value = by_month.mean()
    return None if pd.isna(value) else float(value)


def fmt_value(value: object) -> object:
    if isinstance(value, float):
        return round(value, 10)
    return value


def distribution(df: pd.DataFrame, col: str) -> dict[str, object]:
    if col not in df.columns:
        return {
            "factor": col,
            "n": 0,
            "missing_rate": 1.0,
            "mean": np.nan,
            "median": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "min": np.nan,
            "max": np.nan,
            "plausibility_flag": "MISSING_COLUMN",
            "notes": "column not present",
        }
    s = pd.to_numeric(df[col], errors="coerce")
    nonnull = s.dropna()
    missing_rate = float(s.isna().mean()) if len(s) else 1.0
    if nonnull.empty:
        return {
            "factor": col,
            "n": 0,
            "missing_rate": missing_rate,
            "mean": np.nan,
            "median": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "min": np.nan,
            "max": np.nan,
            "plausibility_flag": "NO_VALID_VALUES",
            "notes": "all missing",
        }
    q = nonnull.quantile([0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
    plausible = np.isfinite(nonnull.mean()) and np.isfinite(nonnull.median())
    return {
        "factor": col,
        "n": int(nonnull.size),
        "missing_rate": missing_rate,
        "mean": float(nonnull.mean()),
        "median": float(nonnull.median()),
        "p01": float(q.loc[0.01]),
        "p05": float(q.loc[0.05]),
        "p25": float(q.loc[0.25]),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
        "p99": float(q.loc[0.99]),
        "min": float(nonnull.min()),
        "max": float(nonnull.max()),
        "plausibility_flag": "PASS" if plausible else "REVIEW",
        "notes": "",
    }


def qa_row(check: str, passed: bool, details: str) -> dict[str, object]:
    return {"check": check, "pass": bool(passed), "details": details}


def main() -> None:
    ensure_dirs()
    before_mtimes = snapshot_mtimes(FORBIDDEN_FILES, FORBIDDEN_DIRS)
    outputs: list[Path] = [RUN_DIR / "RUN_STATE.md", RUN_DIR / "CHECKPOINTS.md", RUN_DIR / "PROMPT_SNAPSHOT.md"]
    update_run_state("starting script", ["Checkpoint initialized", "Script started"], [rel(Path(__file__))], outputs, "Build target universe audit")

    v1_symbols_df = read_parquet_cols(V1_FACTOR, ["symbol"])
    strict_symbols_df = read_parquet_cols(STRICT_V1, ["symbol"])
    v1_symbols = set(v1_symbols_df["symbol"].dropna().astype(str))
    strict_symbols = set(strict_symbols_df["symbol"].dropna().astype(str))
    target_universe = v1_symbols
    intersection = v1_symbols & strict_symbols
    v1_only = v1_symbols - strict_symbols
    strict_only = strict_symbols - v1_symbols
    target_audit = metric_rows(
        [
            ("n_v1_factor_symbols", len(v1_symbols), ""),
            ("n_strict_panel_symbols", len(strict_symbols), ""),
            ("n_intersection", len(intersection), ""),
            ("n_v1_only", len(v1_only), "sample=" + "|".join(sorted(v1_only)[:20])),
            ("n_strict_only", len(strict_only), "sample=" + "|".join(sorted(strict_only)[:20])),
            ("target_universe_source", "v1_factor_panel", "default per task instruction"),
            ("target_universe_n_symbols", len(target_universe), ""),
        ]
    )
    target_audit.to_csv(TARGET_AUDIT, index=False, encoding="utf-8-sig")
    del v1_symbols_df, strict_symbols_df, target_audit
    gc.collect()
    outputs.append(TARGET_AUDIT)
    update_run_state("target universe audit complete", ["Target universe audit generated"], [rel(V2_FACTOR)], outputs, "Diagnose v2 universe expansion")
    add_checkpoint("target universe audit complete", [f"target_universe_n_symbols={len(target_universe)}", f"n_strict_only={len(strict_only)}"])

    v2 = read_parquet_cols(V2_FACTOR)
    early_mask = (v2["month_end"] >= pd.Timestamp("2017-01-31")) & (v2["month_end"] <= pd.Timestamp("2017-03-31"))
    extra_mask = ~v2["symbol"].isin(target_universe)
    v2_extra_symbols = set(v2.loc[extra_mask, "symbol"].dropna().astype(str))
    v2_extra_early_mask = early_mask & extra_mask
    months_with_extra = v2.loc[extra_mask, "month_end"].dropna().dt.strftime("%Y-%m-%d").nunique()
    v2_diag = metric_rows(
        [
            ("v2_n_symbols", int(v2["symbol"].nunique()), ""),
            ("v2_n_symbols_not_in_target_universe", len(v2_extra_symbols), ""),
            ("v2_extra_symbols_2017_01_to_2017_03", int(v2.loc[v2_extra_early_mask, "symbol"].nunique()), ""),
            ("v2_extra_rows_2017_01_to_2017_03", int(v2_extra_early_mask.sum()), ""),
            ("v2_extra_rows_after_2017_04", int(((v2["month_end"] >= pd.Timestamp("2017-04-01")) & extra_mask).sum()), ""),
            ("months_with_extra_symbols", int(months_with_extra), ""),
            ("conclusion", "V2_HAS_UNIVERSE_EXPANSION" if v2_extra_symbols else "NO_V2_UNIVERSE_EXPANSION", "target universe is v1 factor panel"),
        ]
    )
    v2_diag.to_csv(V2_DIAG, index=False, encoding="utf-8-sig")
    sample = (
        v2.loc[extra_mask, ["symbol", "month_end"]]
        .assign(is_early_2017_01_to_03=lambda x: (x["month_end"] >= pd.Timestamp("2017-01-31")) & (x["month_end"] <= pd.Timestamp("2017-03-31")))
        .groupby("symbol", as_index=False, observed=True)
        .agg(first_month=("month_end", "min"), last_month=("month_end", "max"), n_rows=("month_end", "size"), n_early_rows=("is_early_2017_01_to_03", "sum"))
        .sort_values(["n_early_rows", "n_rows", "symbol"], ascending=[False, False, True])
        .head(200)
    )
    for col in ["first_month", "last_month"]:
        sample[col] = sample[col].dt.strftime("%Y-%m-%d")
    sample.to_csv(V2_EXTRA_SAMPLE, index=False, encoding="utf-8-sig")
    del v2_diag, sample
    gc.collect()
    outputs.extend([V2_DIAG, V2_EXTRA_SAMPLE])
    update_run_state("v2 universe expansion diagnosed", ["V2 expansion diagnostic generated", "Extra symbol sample generated"], [rel(V2_FACTOR)], outputs, "Generate v3 locked universe panel")
    add_checkpoint("v2 universe expansion diagnosed", [f"v2_extra_symbols={len(v2_extra_symbols)}", f"v2_extra_rows_early={int(v2_extra_early_mask.sum())}"])

    v3 = v2.loc[v2["symbol"].isin(target_universe) & (v2["month_end"] >= pd.Timestamp("2017-01-31")) & (v2["month_end"] <= pd.Timestamp("2026-06-30"))].copy()
    v3 = v3.sort_values(["month_end", "symbol"]).reset_index(drop=True)
    v3.to_parquet(V3_PANEL, index=False)
    v3.head(1000).to_csv(V3_SAMPLE, index=False, encoding="utf-8-sig")
    outputs.extend([V3_PANEL, V3_SAMPLE])
    update_run_state("v3 locked panel generated", ["V3 panel generated from V2 by target universe filter only"], [rel(V1_FACTOR), rel(V2_FACTOR)], outputs, "Generate v1/v2/v3 comparison")
    add_checkpoint("v3 locked panel generated", [f"v3_rows={len(v3)}", f"v3_symbols={v3['symbol'].nunique()}"])

    v1 = read_parquet_cols(V1_FACTOR)
    summaries = {"v1": summarize_panel(v1, target_universe), "v2": summarize_panel(v2, target_universe), "v3": summarize_panel(v3, target_universe)}
    comparison_rows = []
    for metric in [
        "min_month_end",
        "max_month_end",
        "n_rows",
        "n_symbols",
        "n_months",
        "n_rows_2017_01_to_2017_03",
        "n_symbols_2017_01_to_2017_03",
        "n_extra_symbols_not_in_target_universe",
        "selected_pit_date_violation_count",
        "market_cap_date_violation_count",
        "one_row_per_symbol_month",
        "ep_median",
        "bp_median",
        "roe_coverage_mean",
        "ep_coverage_mean",
        "bp_coverage_mean",
    ]:
        comparison_rows.append(
            {
                "metric": metric,
                "v1_value": fmt_value(summaries["v1"][metric]),
                "v2_value": fmt_value(summaries["v2"][metric]),
                "v3_value": fmt_value(summaries["v3"][metric]),
                "details": "target universe=v1 factor panel symbols",
            }
        )
    pd.DataFrame(comparison_rows).to_csv(COMPARISON, index=False, encoding="utf-8-sig")
    outputs.append(COMPARISON)
    update_run_state("comparison complete", ["V1/V2/V3 comparison generated"], [rel(V3_PANEL)], outputs, "Generate v3 QA audits")
    add_checkpoint("comparison complete", [f"v1_symbols={summaries['v1']['n_symbols']}", f"v2_symbols={summaries['v2']['n_symbols']}", f"v3_symbols={summaries['v3']['n_symbols']}"])

    coverage_cols = [
        ("roe_coverage", "roe_ttm"),
        ("ep_coverage", "ep_ttm"),
        ("bp_coverage", "bp"),
        ("profit_growth_yoy_coverage", "profit_growth_yoy"),
        ("rev_growth_yoy_coverage", "rev_growth_yoy"),
        ("net_margin_coverage", "net_margin"),
        ("debt_ratio_coverage", "debt_ratio"),
        ("sales_expense_to_revenue_coverage", "sales_expense_to_revenue"),
        ("admin_expense_to_revenue_coverage", "admin_expense_to_revenue"),
        ("rd_expense_to_revenue_coverage", "rd_expense_to_revenue"),
        ("ttm_complete_rate", "ttm_complete_flag"),
        ("market_cap_coverage", "market_cap_total"),
    ]
    coverage_records = []
    for month, group in v3.groupby("month_end", observed=True):
        row: dict[str, object] = {"month_end": month.date().isoformat(), "n_symbols": int(group["symbol"].nunique())}
        notes = []
        for out_col, src_col in coverage_cols:
            if src_col in group.columns:
                row[out_col] = float(group[src_col].notna().mean())
            else:
                row[out_col] = np.nan
                notes.append(f"missing {src_col}")
        row["notes"] = "; ".join(notes)
        coverage_records.append(row)
    pd.DataFrame(coverage_records).to_csv(COVERAGE_AUDIT, index=False, encoding="utf-8-sig")

    factor_cols = [
        "roe_ttm",
        "ep_ttm",
        "bp",
        "profit_growth_yoy",
        "rev_growth_yoy",
        "net_margin",
        "debt_ratio",
        "sales_expense_to_revenue",
        "admin_expense_to_revenue",
        "rd_expense_to_revenue",
        "revenue_ttm",
        "net_profit_parent_ttm",
        "net_profit_ttm",
        "total_assets",
        "total_liabilities",
        "equity_parent",
        "total_equity",
        "market_cap_total",
        "market_cap_float",
        "total_market_cap_raw_thousand",
    ]
    pd.DataFrame([distribution(v3, col) for col in factor_cols]).to_csv(DIST_AUDIT, index=False, encoding="utf-8-sig")
    outputs.extend([COVERAGE_AUDIT, DIST_AUDIT])
    update_run_state("v3 QA audits complete", ["Coverage audit generated", "Distribution audit generated"], [rel(V3_PANEL)], outputs, "Generate final report, card, and QA")
    add_checkpoint("v3 QA audits complete", ["factor coverage audit generated", "factor distribution audit generated"])

    v3_summary = summaries["v3"]
    selected_viol = int(v3_summary["selected_pit_date_violation_count"])
    market_viol = int(v3_summary["market_cap_date_violation_count"])
    extra_outside = int(v3_summary["n_extra_symbols_not_in_target_universe"])
    early_rows = int(v3_summary["n_rows_2017_01_to_2017_03"])
    min_ok = pd.Timestamp(v3["month_end"].min()) <= pd.Timestamp("2017-01-31")
    date_ok = selected_viol == 0 and market_viol == 0
    no_forbidden_actions = True
    if min_ok and extra_outside == 0 and date_ok:
        decision = "CSMAR_TTM_BUFFER_UNIVERSE_LOCK_PATCH_READY_FOR_V3_QA"
    elif extra_outside > 0:
        decision = "INVALID_UNIVERSE_EXPANSION_REMAINS"
    elif not date_ok:
        decision = "INVALID_PIT_OR_MARKET_CAP_DATE_ALIGNMENT"
    elif early_rows <= 0:
        decision = "CSMAR_TTM_BUFFER_UNIVERSE_LOCK_PATCH_PARTIAL_NEEDS_REVIEW"
    elif not no_forbidden_actions:
        decision = "INVALID_FORBIDDEN_ACTION"
    else:
        decision = "CSMAR_TTM_BUFFER_UNIVERSE_LOCK_PATCH_PARTIAL_NEEDS_REVIEW"

    after_mtime_changes = mtimes_changed(before_mtimes)
    readme_modified = after_mtime_changes.get(str(ROOT / "README.md"), False)
    all_daily_modified = after_mtime_changes.get(str(ROOT / "output" / "all_daily.parquet"), False)
    training_panel_modified = after_mtime_changes.get(str(ROOT / "output" / "training_panel_v15_sr.parquet"), False)
    v1_outputs_overwritten = any(changed for path, changed in after_mtime_changes.items() if str(FORBIDDEN_DIRS[0]) in path)
    v2_outputs_overwritten = any(changed for path, changed in after_mtime_changes.items() if str(FORBIDDEN_DIRS[1]) in path)
    xlsx_read = False
    raw_daily_csv_read = False
    csmar_api_accessed = False
    download_executed = False
    model_training = False
    backtest = False
    ic = False
    signal_generation = False
    production_modified = False

    report_text = f"""# CSMAR PIT-Clean TTM Buffer Universe Lock Patch v1

Run timestamp: {now()}

This task did not access CSMAR API, download data, read Excel files, read raw daily CSV files, train models, run backtests, calculate IC, or connect to production.

V2 issue: early months introduced universe expansion beyond the v1 strict factor universe. V3 locks the final panel to symbols observed in the v1 factor panel while preserving V2 factor values and the V2 TTM pre-window buffer output.

Key results:
- target universe symbols: {len(target_universe)}
- v1 symbols: {summaries['v1']['n_symbols']}
- v2 symbols: {summaries['v2']['n_symbols']}
- v3 symbols: {summaries['v3']['n_symbols']}
- v3 min month_end: {v3_summary['min_month_end']}
- v3 max month_end: {v3_summary['max_month_end']}
- v3 rows: {v3_summary['n_rows']}
- v3 rows 2017-01 to 2017-03: {early_rows}
- v3 extra symbols outside target universe: {extra_outside}
- selected_pit_date violations: {selected_viol}
- market_cap_trade_date violations: {market_viol}
- one row per symbol-month: {v3_summary['one_row_per_symbol_month']}

EP/BP check: V3 retains V2 market-cap columns, including total_market_cap_raw_thousand, and does not winsorize, zscore, rank, or recalculate factor values.

Decision: {decision}

V3 is a factor source panel for v3 QA / FI_T5 sanity check. It is not a final training panel.
"""
    REPORT.write_text(report_text, encoding="utf-8")

    card_text = f"""任务名称：
CSMAR PIT-Clean TTM Buffer Universe Lock Patch v1
运行日期：
{now()}
是否读取 xlsx：
False
是否读取原始日频 CSV：
False
是否访问 CSMAR API：
False
是否下载数据：
False
是否训练模型：
False
是否回测：
False
是否做 IC：
False
是否修改 production：
False
是否修改 README：
{readme_modified}
核心输出：
{rel(V3_PANEL)}
核心结论：
{decision}
target universe symbols：
{len(target_universe)}
v1 symbols：
{summaries['v1']['n_symbols']}
v2 symbols：
{summaries['v2']['n_symbols']}
v3 symbols：
{summaries['v3']['n_symbols']}
v3 min_month_end：
{v3_summary['min_month_end']}
v3 max_month_end：
{v3_summary['max_month_end']}
v3 rows：
{v3_summary['n_rows']}
v3 rows 2017-01 to 2017-03：
{early_rows}
v3 extra symbols outside target universe：
{extra_outside}
selected_pit_date_violation_count：
{selected_viol}
market_cap_date_violation_count：
{market_viol}
one_row_per_symbol_month：
{v3_summary['one_row_per_symbol_month']}
下一步建议：
Run v3 QA / FI_T5 sanity check without production integration.
"""
    CARD.write_text(card_text, encoding="utf-8")

    qa_rows = [
        qa_row("no xlsx read", not xlsx_read, "script only reads parquet inputs"),
        qa_row("no raw daily CSV read", not raw_daily_csv_read, "no CSV inputs were read"),
        qa_row("no CSMAR API access", not csmar_api_accessed, "no API code path used"),
        qa_row("no download", not download_executed, "no network/download code path used"),
        qa_row("no model training", not model_training, "not performed"),
        qa_row("no backtest", not backtest, "not performed"),
        qa_row("no IC", not ic, "not performed"),
        qa_row("no signal generation", not signal_generation, "not performed"),
        qa_row("no production modification", not production_modified, "not performed"),
        qa_row("no README modification", not readme_modified, "mtime unchanged or file absent"),
        qa_row("all_daily.parquet not modified", not all_daily_modified, "mtime unchanged or file absent"),
        qa_row("training_panel_v15_sr.parquet not modified", not training_panel_modified, "mtime unchanged or file absent"),
        qa_row("v1 outputs not overwritten", not v1_outputs_overwritten, "v1 output mtimes unchanged"),
        qa_row("v2 outputs not overwritten", not v2_outputs_overwritten, "v2 output mtimes unchanged"),
        qa_row("root output used", str(OUT_DIR).startswith(str(ROOT / "output")), rel(OUT_DIR)),
        qa_row("target universe generated", TARGET_AUDIT.exists(), rel(TARGET_AUDIT)),
        qa_row("v2 universe expansion diagnosed", V2_DIAG.exists(), rel(V2_DIAG)),
        qa_row("v3 factor panel generated", V3_PANEL.exists(), rel(V3_PANEL)),
        qa_row("final output has no month_end before 2017-01-31", bool((v3["month_end"] >= pd.Timestamp("2017-01-31")).all()), ""),
        qa_row("v3 min_month_end <= 2017-01-31", bool(min_ok), str(v3_summary["min_month_end"])),
        qa_row("no symbols outside target universe", extra_outside == 0, str(extra_outside)),
        qa_row("selected_pit_date <= month_end", selected_viol == 0, str(selected_viol)),
        qa_row("market_cap_trade_date <= month_end", market_viol == 0, str(market_viol)),
        qa_row("total_market_cap_x1000 retained for EP/BP", "total_market_cap_raw_thousand" in v3.columns and "ep_ttm" in v3.columns and "bp" in v3.columns, "column retained as total_market_cap_raw_thousand"),
        qa_row("one row per symbol-month", bool(v3_summary["one_row_per_symbol_month"]), ""),
        qa_row("v1/v2/v3 comparison generated", COMPARISON.exists(), rel(COMPARISON)),
        qa_row("factor coverage audit generated", COVERAGE_AUDIT.exists(), rel(COVERAGE_AUDIT)),
        qa_row("factor distribution audit generated", DIST_AUDIT.exists(), rel(DIST_AUDIT)),
        qa_row("final report generated", REPORT.exists(), rel(REPORT)),
        qa_row("task completion card generated", CARD.exists(), rel(CARD)),
        qa_row("no winsor/zscore/rank performed", True, "v3 is filtered from v2 only"),
        qa_row("no model/production files modified", not production_modified and not training_panel_modified, ""),
    ]
    qa_df = pd.DataFrame(qa_rows)
    qa_df.to_csv(FINAL_QA, index=False, encoding="utf-8-sig")
    qa_df.to_csv(FINAL_QA_ALIAS, index=False, encoding="utf-8-sig")

    summary = {
        "target_universe_audit_path": rel(TARGET_AUDIT),
        "v2_universe_expansion_diagnostic_path": rel(V2_DIAG),
        "v2_extra_symbols_sample_path": rel(V2_EXTRA_SAMPLE),
        "pit_clean_factor_panel_v3_path": rel(V3_PANEL),
        "v1_v2_v3_comparison_path": rel(COMPARISON),
        "factor_coverage_audit_v3_path": rel(COVERAGE_AUDIT),
        "factor_distribution_audit_v3_path": rel(DIST_AUDIT),
        "report_path": rel(REPORT),
        "task_completion_card_path": rel(CARD),
        "final_qa_path": rel(FINAL_QA),
        "run_state_path": rel(RUN_DIR / "RUN_STATE.md"),
        "target_universe_n_symbols": len(target_universe),
        "v1_n_symbols": int(summaries["v1"]["n_symbols"]),
        "v2_n_symbols": int(summaries["v2"]["n_symbols"]),
        "v3_n_symbols": int(summaries["v3"]["n_symbols"]),
        "v3_n_rows": int(summaries["v3"]["n_rows"]),
        "v3_min_month_end": summaries["v3"]["min_month_end"],
        "v3_max_month_end": summaries["v3"]["max_month_end"],
        "v3_n_rows_2017_01_to_2017_03": early_rows,
        "v3_extra_symbols_outside_target_universe": extra_outside,
        "selected_pit_date_violation_count": selected_viol,
        "market_cap_date_violation_count": market_viol,
        "one_row_per_symbol_month": bool(v3_summary["one_row_per_symbol_month"]),
        "ep_median": fmt_value(v3_summary["ep_median"]),
        "bp_median": fmt_value(v3_summary["bp_median"]),
        "ep_coverage_mean": fmt_value(v3_summary["ep_coverage_mean"]),
        "bp_coverage_mean": fmt_value(v3_summary["bp_coverage_mean"]),
        "recommended_next_task": "v3 QA / FI_T5 sanity check",
        "xlsx_read": xlsx_read,
        "raw_daily_csv_read": raw_daily_csv_read,
        "csmar_api_accessed": csmar_api_accessed,
        "download_executed": download_executed,
        "readme_modified": readme_modified,
        "all_daily_modified": all_daily_modified,
        "training_panel_modified": training_panel_modified,
        "production_modified": production_modified,
        "v1_outputs_overwritten": v1_outputs_overwritten,
        "v2_outputs_overwritten": v2_outputs_overwritten,
        "decision": decision,
    }
    TERMINAL_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.extend([REPORT, CARD, FINAL_QA, FINAL_QA_ALIAS, TERMINAL_SUMMARY])
    update_run_state("complete", ["Final report generated", "Task completion card generated", "Final QA generated", "Terminal summary generated"], [], outputs, "Task complete")
    add_checkpoint("complete", [f"decision={decision}", f"final_qa={rel(FINAL_QA)}"])

    del v1, v2, v3, qa_df
    gc.collect()
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
