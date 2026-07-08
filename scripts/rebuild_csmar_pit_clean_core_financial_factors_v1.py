from __future__ import annotations

import csv
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "csmar_pit_clean_core_financial_factor_reconstruction_v1"
OUT_NAME = "csmar_pit_clean_core_financial_factors_v1"
TASK_TITLE = "CSMAR PIT-Clean Core Financial Factor Reconstruction v1"
RUN_START_TIME = datetime.now().astimezone().isoformat(timespec="seconds")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / OUT_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME
STRICT_DIR = ROOT / "output" / "csmar_pit_scope_freeze_strict_core_fs_panel_v1"
STATEMENT_PATH = STRICT_DIR / "strict_core_fs_statement_records_v1.parquet"
MONTHLY_PATH = STRICT_DIR / "strict_core_fs_monthly_asof_panel_v1.parquet"
MKT_PATH = ROOT / "output" / "csmar_trd_dalyr_market_cap_import_lite_v1" / "trd_dalyr_monthly_market_cap_panel_v1.parquet"

STATEMENT_COLS = [
    "symbol",
    "report_period",
    "report_type",
    "pit_date_primary",
    "total_operating_revenue",
    "operating_revenue",
    "sales_expense",
    "admin_expense",
    "rd_expense",
    "total_profit",
    "net_profit",
    "net_profit_parent",
    "total_assets",
    "total_liabilities",
    "equity_parent",
    "total_equity",
    "income_available",
    "balance_available",
]
MONTHLY_COLS = [
    "month_end",
    "symbol",
    "selected_report_period",
    "selected_pit_date",
    "report_lag_days",
    "income_available",
    "balance_available",
]
MKT_COLS = [
    "month_end",
    "symbol",
    "trade_date",
    "total_market_cap_x1000",
    "float_market_cap_x1000",
    "total_market_cap_raw_thousand",
    "market_type",
    "trading_status",
]
TTM_ITEMS = [
    "revenue",
    "net_profit_parent",
    "net_profit",
    "sales_expense",
    "admin_expense",
    "rd_expense",
    "total_profit",
]
FACTOR_COLS = [
    "roe_ttm",
    "ep_ttm",
    "bp",
    "profit_growth_yoy",
    "rev_growth_yoy",
    "net_margin",
    "debt_ratio",
    "sales_expense_to_revenue",
    "rd_expense_to_revenue",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.as_posix()


def append_checkpoint(stage: str, notes: list[str]) -> None:
    with (RUN_DIR / "CHECKPOINTS.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## {now_iso()} - {stage}\n\n")
        for note in notes:
            f.write(f"- {note}\n")


def update_run_state(stage: str, done: list[str], outputs: list[Path], next_step: str) -> None:
    done_lines = "\n".join(f"  - {x}" for x in done) if done else "  - none"
    out_lines = "\n".join(f"  - {rel(x)}" for x in outputs) if outputs else "  - none"
    text = f"""# RUN_STATE

- 当前任务名称: {TASK_TITLE}
- 开始时间: {RUN_START_TIME}
- 当前阶段: {stage}
- 已完成步骤:
{done_lines}
- 正在处理的文件:
  - {rel(STATEMENT_PATH)}
  - {rel(MONTHLY_PATH)}
  - {rel(MKT_PATH)}
- 已生成输出:
{out_lines}
- 下一步:
  - {next_step}
- 如果 Codex 崩溃，新的 Codex 应如何继续:
  - 先读取本文件
  - 检查 run_stdout.txt、run_stderr.txt、terminal_summary.json 和 final QA
  - 如果 terminal_summary.json 不存在，重新运行 scripts/rebuild_csmar_pit_clean_core_financial_factors_v1.py，并将 stdout/stderr 写入本 agent run 目录
  - 不要读取 xlsx、原始日频 CSV、访问 API、下载、训练、回测、IC 或修改 production
"""
    (RUN_DIR / "RUN_STATE.md").write_text(text, encoding="utf-8")


def safe_read_parquet(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def inventory_row(path: Path, cols: list[str], df: pd.DataFrame, role: str, notes: str) -> dict[str, object]:
    date_col = "month_end" if "month_end" in df.columns else "report_period" if "report_period" in df.columns else None
    min_date = pd.to_datetime(df[date_col], errors="coerce").min() if date_col else pd.NaT
    max_date = pd.to_datetime(df[date_col], errors="coerce").max() if date_col else pd.NaT
    return {
        "input_path": rel(path),
        "exists": path.exists(),
        "readable": True,
        "columns_read": "|".join(cols),
        "n_rows": int(len(df)),
        "n_symbols": int(df["symbol"].astype(str).nunique()) if "symbol" in df.columns else np.nan,
        "min_date": "" if pd.isna(min_date) else str(min_date.date()),
        "max_date": "" if pd.isna(max_date) else str(max_date.date()),
        "role": role,
        "notes": notes,
    }


def normalize_keys(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str)
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def prepare_statement_records(stmt: pd.DataFrame) -> pd.DataFrame:
    stmt = normalize_keys(stmt, ["report_period", "pit_date_primary"])
    stmt = stmt.sort_values(["symbol", "report_period", "pit_date_primary", "report_type"], kind="mergesort")
    stmt = stmt.drop_duplicates(["symbol", "report_period"], keep="last").copy()
    stmt["fiscal_year"] = stmt["report_period"].dt.year
    stmt["fiscal_quarter"] = stmt["report_period"].dt.quarter
    stmt["revenue"] = stmt["operating_revenue"]
    stmt["revenue_source"] = np.where(stmt["operating_revenue"].notna(), "operating_revenue", "total_operating_revenue")
    stmt.loc[stmt["revenue"].isna(), "revenue"] = stmt.loc[stmt["revenue"].isna(), "total_operating_revenue"]
    return stmt


def add_quarter_and_ttm(stmt: pd.DataFrame) -> pd.DataFrame:
    stmt = stmt.sort_values(["symbol", "report_period"], kind="mergesort").copy()
    stmt["prev_symbol"] = stmt["symbol"].shift(1)
    stmt["prev_year"] = stmt["fiscal_year"].shift(1)
    stmt["prev_quarter"] = stmt["fiscal_quarter"].shift(1)
    has_prev_q = (
        (stmt["prev_symbol"] == stmt["symbol"])
        & (stmt["prev_year"] == stmt["fiscal_year"])
        & (stmt["prev_quarter"] == (stmt["fiscal_quarter"] - 1))
    )
    stmt["missing_prev_quarter_flag"] = (stmt["fiscal_quarter"] > 1) & ~has_prev_q

    for item in TTM_ITEMS:
        prev = stmt.groupby(["symbol", "fiscal_year"], sort=False)[item].shift(1)
        q_col = f"{item}_quarter"
        ttm_col = f"{item}_ttm"
        stmt[q_col] = np.where(stmt["fiscal_quarter"] == 1, stmt[item], stmt[item] - prev)
        stmt.loc[stmt["missing_prev_quarter_flag"], q_col] = np.nan
        stmt[ttm_col] = (
            stmt.groupby("symbol", sort=False)[q_col]
            .rolling(window=4, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    stmt["ttm_quarters_available"] = (
        stmt.groupby("symbol", sort=False)["net_profit_parent_quarter"]
        .rolling(window=4, min_periods=1)
        .count()
        .reset_index(level=0, drop=True)
        .astype("int64")
    )
    stmt["ttm_complete_flag"] = stmt["ttm_quarters_available"] == 4
    stmt["ttm_warning_flag"] = np.where(
        stmt["ttm_complete_flag"],
        "",
        np.where(stmt["missing_prev_quarter_flag"], "missing_previous_quarter;ttm_incomplete", "ttm_warmup_or_missing_quarters"),
    )
    for item in ("net_profit_parent", "revenue"):
        stmt[f"lag_4_quarter_{item}_ttm"] = stmt.groupby("symbol", sort=False)[f"{item}_ttm"].shift(4)

    out_cols = [
        "symbol",
        "report_period",
        "pit_date_primary",
        "report_type",
        "revenue_ttm",
        "revenue_source",
        "net_profit_parent_ttm",
        "net_profit_ttm",
        "sales_expense_ttm",
        "admin_expense_ttm",
        "rd_expense_ttm",
        "total_profit_ttm",
        "lag_4_quarter_net_profit_parent_ttm",
        "lag_4_quarter_revenue_ttm",
        "total_assets",
        "total_liabilities",
        "equity_parent",
        "total_equity",
        "income_available",
        "balance_available",
        "ttm_quarters_available",
        "ttm_complete_flag",
        "ttm_warning_flag",
    ]
    return stmt[out_cols].copy()


def div_valid(numer: pd.Series, denom: pd.Series, denom_positive: bool = True) -> pd.Series:
    n = pd.to_numeric(numer, errors="coerce")
    d = pd.to_numeric(denom, errors="coerce")
    ok = d.notna() & (d > 0 if denom_positive else d.ne(0))
    out = n / d
    out = out.where(ok)
    return out.replace([np.inf, -np.inf], np.nan)


def build_flags(row: pd.Series) -> str:
    flags: list[str] = []
    if pd.isna(row.get("selected_report_period")):
        flags.append("missing_selected_report_period")
    if not row.get("ttm_complete_flag", False):
        flags.append("ttm_incomplete")
    if pd.isna(row.get("total_market_cap_x1000")) or row.get("total_market_cap_x1000", np.nan) <= 0:
        flags.append("market_cap_invalid")
    if pd.isna(row.get("equity_parent")) or row.get("equity_parent", np.nan) <= 0:
        flags.append("equity_invalid")
    if pd.isna(row.get("revenue_ttm")) or row.get("revenue_ttm", np.nan) <= 0:
        flags.append("revenue_invalid")
    if pd.isna(row.get("total_assets")) or row.get("total_assets", np.nan) <= 0:
        flags.append("assets_invalid")
    if pd.notna(row.get("selected_pit_date")) and pd.notna(row.get("month_end")) and row["selected_pit_date"] > row["month_end"]:
        flags.append("selected_pit_date_after_month_end")
    if pd.notna(row.get("market_cap_trade_date")) and pd.notna(row.get("month_end")) and row["market_cap_trade_date"] > row["month_end"]:
        flags.append("market_cap_trade_date_after_month_end")
    return ";".join(flags)


def distribution_row(df: pd.DataFrame, factor: str) -> dict[str, object]:
    s = pd.to_numeric(df[factor], errors="coerce")
    valid = s.dropna()
    q = valid.quantile([0.01, 0.05, 0.25, 0.75, 0.95, 0.99]) if len(valid) else pd.Series(dtype=float)
    missing_rate = float(1 - len(valid) / len(s)) if len(s) else np.nan
    med = float(valid.median()) if len(valid) else np.nan
    p99_abs = float(valid.abs().quantile(0.99)) if len(valid) else np.nan
    plausible = "plausible"
    notes = "raw factor; no winsorization, zscore, or rank"
    if factor in ("ep_ttm", "bp", "roe_ttm", "net_margin", "debt_ratio", "sales_expense_to_revenue", "rd_expense_to_revenue"):
        if pd.notna(p99_abs) and p99_abs > 100:
            plausible = "extreme_values_review"
            notes = "p99 absolute value exceeds 100; requires QA review"
    if factor in ("profit_growth_yoy", "rev_growth_yoy") and pd.notna(p99_abs) and p99_abs > 1000:
        plausible = "extreme_growth_review"
        notes = "growth p99 absolute value exceeds 1000; requires QA review"
    return {
        "factor": factor,
        "n": int(len(valid)),
        "missing_rate": missing_rate,
        "mean": float(valid.mean()) if len(valid) else np.nan,
        "median": med,
        "p01": float(q.get(0.01, np.nan)),
        "p05": float(q.get(0.05, np.nan)),
        "p25": float(q.get(0.25, np.nan)),
        "p75": float(q.get(0.75, np.nan)),
        "p95": float(q.get(0.95, np.nan)),
        "p99": float(q.get(0.99, np.nan)),
        "min": float(valid.min()) if len(valid) else np.nan,
        "max": float(valid.max()) if len(valid) else np.nan,
        "plausibility_flag": plausible,
        "notes": notes,
    }


def coverage_by_month(factors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month_end, g in factors.groupby("month_end", sort=True):
        n = len(g)
        row: dict[str, object] = {
            "month_end": month_end.strftime("%Y-%m-%d") if pd.notna(month_end) else "",
            "n_symbols": int(g["symbol"].nunique()),
        }
        for col, out in [
            ("roe_ttm", "roe_coverage"),
            ("ep_ttm", "ep_coverage"),
            ("bp", "bp_coverage"),
            ("profit_growth_yoy", "profit_growth_yoy_coverage"),
            ("rev_growth_yoy", "rev_growth_yoy_coverage"),
            ("net_margin", "net_margin_coverage"),
            ("debt_ratio", "debt_ratio_coverage"),
            ("sales_expense_to_revenue", "sales_expense_to_revenue_coverage"),
            ("rd_expense_to_revenue", "rd_expense_to_revenue_coverage"),
        ]:
            row[out] = float(g[col].notna().sum() / n) if n else np.nan
        row["ttm_complete_rate"] = float(g["ttm_complete_flag"].fillna(False).mean()) if n else np.nan
        row["market_cap_coverage"] = float((pd.to_numeric(g["market_cap_total"], errors="coerce") > 0).sum() / n) if n else np.nan
        notes: list[str] = []
        if pd.notna(month_end) and month_end.year == 2017 and row["ttm_complete_rate"] < 0.5:
            notes.append("2017 warm-up TTM coverage low")
        if pd.notna(month_end) and month_end.year < 2018 and row["rd_expense_to_revenue_coverage"] < 0.2:
            notes.append("pre-2018 rd expense structurally sparse")
        row["notes"] = ";".join(notes)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    input_inventory_path = OUT_DIR / "input_inventory_v1.csv"
    ttm_path = OUT_DIR / "report_level_ttm_raw_items_v1.parquet"
    ttm_sample_path = OUT_DIR / "report_level_ttm_raw_items_sample_v1.csv"
    raw_panel_path = OUT_DIR / "monthly_factor_raw_input_panel_v1.parquet"
    raw_panel_sample_path = OUT_DIR / "monthly_factor_raw_input_panel_sample_v1.csv"
    factor_path = OUT_DIR / "pit_clean_core_financial_factors_monthly_v1.parquet"
    factor_sample_path = OUT_DIR / "pit_clean_core_financial_factors_monthly_sample_v1.csv"
    coverage_path = OUT_DIR / "factor_coverage_audit_by_month_v1.csv"
    distribution_path = OUT_DIR / "factor_distribution_audit_v1.csv"
    report_path = OUT_DIR / "csmar_pit_clean_core_financial_factor_reconstruction_report_v1.md"
    card_path = OUT_DIR / "task_completion_card.md"
    qa_path = OUT_DIR / "final_qa_csmar_pit_clean_core_financial_factors_v1.csv"
    qa_alias_path = OUT_DIR / "final_qa.csv"
    terminal_summary_path = OUT_DIR / "terminal_summary.json"

    update_run_state("input_read", ["script started"], [], "read only allowed parquet columns")
    stmt = safe_read_parquet(STATEMENT_PATH, STATEMENT_COLS)
    monthly = safe_read_parquet(MONTHLY_PATH, MONTHLY_COLS)
    mkt = safe_read_parquet(MKT_PATH, MKT_COLS)
    inventory = pd.DataFrame(
        [
            inventory_row(STATEMENT_PATH, STATEMENT_COLS, stmt, "strict PIT statement records", "strict actual PIT source; no Excel read"),
            inventory_row(MONTHLY_PATH, MONTHLY_COLS, monthly, "strict PIT monthly as-of selector", "selected_pit_date controls visibility"),
            inventory_row(MKT_PATH, MKT_COLS, mkt, "monthly market cap", "EP/BP denominator uses total_market_cap_x1000; raw-thousand retained only for audit"),
        ]
    )
    inventory.to_csv(input_inventory_path, index=False, encoding="utf-8-sig")
    append_checkpoint("input_inventory_done", [f"generated {rel(input_inventory_path)}"])

    update_run_state("ttm_reconstruction", ["input inventory generated"], [input_inventory_path], "construct report-level quarter values and TTM raw items")
    stmt_prepared = prepare_statement_records(stmt)
    del stmt
    gc.collect()
    ttm = add_quarter_and_ttm(stmt_prepared)
    del stmt_prepared
    gc.collect()
    ttm.to_parquet(ttm_path, index=False)
    ttm.head(2000).to_csv(ttm_sample_path, index=False, encoding="utf-8-sig")
    append_checkpoint("ttm_reconstruction_done", [f"generated {rel(ttm_path)}", f"rows: {len(ttm)}"])

    update_run_state("monthly_merge", ["report-level TTM generated"], [input_inventory_path, ttm_path, ttm_sample_path], "merge monthly as-of panel with TTM and market cap")
    monthly = normalize_keys(monthly, ["month_end", "selected_report_period", "selected_pit_date"])
    mkt = normalize_keys(mkt, ["month_end", "trade_date"])
    mkt = mkt.sort_values(["symbol", "month_end", "trade_date"], kind="mergesort").drop_duplicates(["symbol", "month_end"], keep="last")
    raw_panel = monthly.merge(
        ttm,
        how="left",
        left_on=["symbol", "selected_report_period"],
        right_on=["symbol", "report_period"],
        suffixes=("_monthly", "_statement"),
        validate="many_to_one",
    )
    raw_panel = raw_panel.merge(mkt, how="left", on=["symbol", "month_end"], validate="many_to_one")
    raw_panel = raw_panel.rename(columns={"trade_date": "market_cap_trade_date"})
    raw_panel["selected_pit_date_violation"] = raw_panel["selected_pit_date"].notna() & raw_panel["month_end"].notna() & (raw_panel["selected_pit_date"] > raw_panel["month_end"])
    raw_panel["market_cap_date_violation"] = raw_panel["market_cap_trade_date"].notna() & raw_panel["month_end"].notna() & (raw_panel["market_cap_trade_date"] > raw_panel["month_end"])
    raw_panel.to_parquet(raw_panel_path, index=False)
    raw_panel.head(2000).to_csv(raw_panel_sample_path, index=False, encoding="utf-8-sig")
    del monthly, mkt, ttm
    gc.collect()
    append_checkpoint("monthly_merge_done", [f"generated {rel(raw_panel_path)}", f"rows: {len(raw_panel)}"])

    update_run_state("factor_calculation", ["monthly raw input panel generated"], [input_inventory_path, ttm_path, raw_panel_path], "compute PIT-clean raw factors")
    factors = raw_panel.copy()
    factors["market_cap_total"] = factors["total_market_cap_x1000"]
    factors["market_cap_float"] = factors["float_market_cap_x1000"]
    factors["roe_ttm"] = div_valid(factors["net_profit_parent_ttm"], factors["equity_parent"])
    factors["ep_ttm"] = div_valid(factors["net_profit_parent_ttm"], factors["total_market_cap_x1000"])
    factors["bp"] = div_valid(factors["equity_parent"], factors["total_market_cap_x1000"])
    factors["profit_growth_yoy"] = div_valid(factors["net_profit_parent_ttm"], factors["lag_4_quarter_net_profit_parent_ttm"]) - 1
    factors["rev_growth_yoy"] = div_valid(factors["revenue_ttm"], factors["lag_4_quarter_revenue_ttm"]) - 1
    factors["net_margin"] = div_valid(factors["net_profit_parent_ttm"], factors["revenue_ttm"])
    factors["debt_ratio"] = div_valid(factors["total_liabilities"], factors["total_assets"])
    factors["sales_expense_to_revenue"] = div_valid(factors["sales_expense_ttm"], factors["revenue_ttm"])
    factors["rd_expense_to_revenue"] = div_valid(factors["rd_expense_ttm"], factors["revenue_ttm"])
    factors["factor_validity_flags"] = factors.apply(build_flags, axis=1)

    factor_cols_out = [
        "month_end",
        "symbol",
        "selected_report_period",
        "selected_pit_date",
        "market_cap_trade_date",
        "report_lag_days",
        "market_cap_total",
        "market_cap_float",
        "total_market_cap_raw_thousand",
        "market_type",
        "trading_status",
        "roe_ttm",
        "ep_ttm",
        "bp",
        "profit_growth_yoy",
        "rev_growth_yoy",
        "net_margin",
        "debt_ratio",
        "sales_expense_to_revenue",
        "rd_expense_to_revenue",
        "revenue_ttm",
        "net_profit_parent_ttm",
        "net_profit_ttm",
        "sales_expense_ttm",
        "admin_expense_ttm",
        "rd_expense_ttm",
        "total_profit_ttm",
        "total_assets",
        "total_liabilities",
        "equity_parent",
        "total_equity",
        "ttm_complete_flag",
        "ttm_quarters_available",
        "factor_validity_flags",
    ]
    factors = factors[factor_cols_out].sort_values(["month_end", "symbol"], kind="mergesort")
    factors.to_parquet(factor_path, index=False)
    factors.head(2000).to_csv(factor_sample_path, index=False, encoding="utf-8-sig")
    append_checkpoint("factor_calculation_done", [f"generated {rel(factor_path)}", f"rows: {len(factors)}"])

    update_run_state("qa_audit", ["factor panel generated"], [input_inventory_path, ttm_path, raw_panel_path, factor_path], "generate coverage, distribution, report, QA")
    coverage = coverage_by_month(factors)
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
    distribution = pd.DataFrame([distribution_row(factors, c) for c in FACTOR_COLS])
    distribution.to_csv(distribution_path, index=False, encoding="utf-8-sig")

    selected_pit_violations = int((raw_panel["selected_pit_date_violation"]).sum())
    market_cap_violations = int((raw_panel["market_cap_date_violation"]).sum())
    dup_count = int(factors.duplicated(["symbol", "month_end"]).sum())
    one_row = dup_count == 0
    ep_cov_mean = float(coverage["ep_coverage"].mean()) if len(coverage) else np.nan
    bp_cov_mean = float(coverage["bp_coverage"].mean()) if len(coverage) else np.nan
    market_cap_cov_mean = float(coverage["market_cap_coverage"].mean()) if len(coverage) else np.nan
    ep_bp_unblocked = ep_cov_mean > 0.5 and bp_cov_mean > 0.5 and market_cap_cov_mean > 0.5
    extreme_review = distribution["plausibility_flag"].str.contains("review", na=False).any()
    ttm_cov_mean = float(coverage["ttm_complete_rate"].mean()) if len(coverage) else np.nan

    if selected_pit_violations or market_cap_violations:
        decision = "INVALID_PIT_OR_MARKET_CAP_DATE_ALIGNMENT"
    elif not ep_bp_unblocked or (pd.notna(ttm_cov_mean) and ttm_cov_mean < 0.5) or extreme_review:
        decision = "CSMAR_PIT_CLEAN_CORE_FINANCIAL_FACTORS_NEEDS_QA_REVIEW"
    else:
        decision = "CSMAR_PIT_CLEAN_CORE_FINANCIAL_FACTORS_READY_FOR_QA"

    min_month = factors["month_end"].min()
    max_month = factors["month_end"].max()
    report = f"""# CSMAR PIT-Clean Core Financial Factor Reconstruction v1

## 1. Executive Summary

Generated a PIT-clean monthly core financial factor source panel from strict actual PIT financial statements and TRD_Dalyr monthly market cap. This is not a final training panel.

Decision: {decision}

## 2. Inputs

- strict_core_fs_statement_records_v1.parquet
- strict_core_fs_monthly_asof_panel_v1.parquet
- trd_dalyr_monthly_market_cap_panel_v1.parquet

## 3. PIT Policy

All financial records use strict actual PIT. The monthly selector uses selected_pit_date as the visibility date. No PIT fallback is used and report_period is not used as a visibility date.

## 4. Unit Alignment

FS_Comins / FS_Combas amounts are treated as yuan. TRD_Dalyr Dsmvtll / Dsmvosd raw units are thousand. EP/BP use total_market_cap_x1000.

## 5. TTM Reconstruction Method

Income statement cumulative fields are converted to single-quarter values by fiscal quarter, then rolling four-quarter sums are used for TTM raw items. Missing previous quarters are flagged and warm-up periods are marked incomplete.

## 6. Factor Definitions

- roe_ttm = net_profit_parent_ttm / equity_parent
- ep_ttm = net_profit_parent_ttm / total_market_cap_x1000
- bp = equity_parent / total_market_cap_x1000
- profit_growth_yoy = net_profit_parent_ttm / lag_4_quarter_net_profit_parent_ttm - 1
- rev_growth_yoy = revenue_ttm / lag_4_quarter_revenue_ttm - 1
- net_margin = net_profit_parent_ttm / revenue_ttm
- debt_ratio = total_liabilities / total_assets
- sales_expense_to_revenue = sales_expense_ttm / revenue_ttm
- rd_expense_to_revenue = rd_expense_ttm / revenue_ttm

No winsorization, zscore, cross-sectional rank, IC, backtest, model training, signal generation, or production integration was performed.

## 7. Monthly Factor Panel

- rows: {len(factors)}
- symbols: {factors["symbol"].nunique()}
- date range: {min_month.date() if pd.notna(min_month) else ""} to {max_month.date() if pd.notna(max_month) else ""}
- one row per symbol-month: {one_row}

## 8. Coverage Audit

Mean EP coverage: {ep_cov_mean}
Mean BP coverage: {bp_cov_mean}
Market-cap blocking removed for EP/BP: {ep_bp_unblocked}

## 9. Distribution Audit

Distribution audit is saved separately. Extreme raw values are not clipped in this source panel.

## 10. Known Limitations

2017 early months may have TTM warm-up gaps. RD expense ratios can be structurally sparse before 2018. FI_T5 was not read and is not used as the PIT-clean source.

## 11. Recommended Next Task

Run a separate factor QA / sanity-check task, including optional FI_T5 comparison. Do not directly connect this source panel to production.

## 12. Files Generated

- {rel(input_inventory_path)}
- {rel(ttm_path)}
- {rel(raw_panel_path)}
- {rel(factor_path)}
- {rel(coverage_path)}
- {rel(distribution_path)}
- {rel(report_path)}
- {rel(card_path)}
- {rel(qa_path)}

## Guardrails

- 本任务没有访问 CSMAR API。
- 本任务没有下载数据。
- 本任务没有读取 Excel。
- 本任务没有训练模型、回测或 IC。
- 本任务没有接入 production。

## FI_T5 Sanity Check Note

FI_T5 可在后续单独任务中作为 sanity check。F050501B = ROE; F053301B = 营业毛利率; F051701B = 销售费用率; F051801B = 管理费用率。FI_T5 不作为当前 PIT-clean 主因子来源。
"""
    report_path.write_text(report, encoding="utf-8")

    card = f"""任务名称：
{TASK_TITLE}
运行日期：
{now_iso()}
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
False
核心输出：
{rel(factor_path)}
{rel(coverage_path)}
{rel(distribution_path)}
{rel(report_path)}
核心结论：
PIT-clean core financial factor source panel generated using strict actual PIT and total_market_cap_x1000 for EP/BP.
月度因子 panel 行数：
{len(factors)}
symbol 数：
{factors["symbol"].nunique()}
日期范围：
{min_month.date() if pd.notna(min_month) else ""} to {max_month.date() if pd.notna(max_month) else ""}
EP/BP 是否已解除 market_cap 阻塞：
{ep_bp_unblocked}
TTM 重构是否完成：
{ttm_path.exists()}
主要可用因子：
{", ".join(FACTOR_COLS)}
主要限制：
2017 early TTM warm-up and pre-2018 RD expense sparsity require QA review.
下一步建议：
factor QA / sanity check; do not connect directly to production.
"""
    card_path.write_text(card, encoding="utf-8")

    protected_paths = {
        "readme_modified": ROOT / "README.md",
        "all_daily_modified": ROOT / "output" / "all_daily.parquet",
        "training_panel_modified": ROOT / "output" / "training_panel_v15_sr.parquet",
    }
    qa_rows = [
        ("no xlsx read", True, "Only parquet inputs were read."),
        ("no raw daily CSV read", True, "No TRD_Dalyr raw daily CSV opened."),
        ("no CSMAR API access", True, "No API code path exists."),
        ("no download", True, "No network/download code path exists."),
        ("no model training", True, "No model training code path exists."),
        ("no backtest", True, "No backtest code path exists."),
        ("no IC", True, "No IC code path exists."),
        ("no signal generation", True, "No signal output generated."),
        ("no production modification", True, "No production files modified by this script."),
        ("no README modification", True, "README not touched."),
        ("all_daily.parquet not modified", True, "Script never writes all_daily.parquet."),
        ("training_panel_v15_sr.parquet not modified", True, "Script never writes training_panel_v15_sr.parquet."),
        ("root output used", str(OUT_DIR).startswith(str(ROOT / "output")), rel(OUT_DIR)),
        ("strict PIT source used", True, rel(STATEMENT_PATH)),
        ("no PIT fallback used", True, "Only selected_report_period from strict monthly as-of panel used."),
        ("selected_pit_date <= month_end", selected_pit_violations == 0, str(selected_pit_violations)),
        ("market_cap_trade_date <= month_end", market_cap_violations == 0, str(market_cap_violations)),
        ("total_market_cap_x1000 used for EP/BP", True, "ep_ttm and bp denominators are total_market_cap_x1000."),
        ("report-level TTM generated", ttm_path.exists(), rel(ttm_path)),
        ("monthly raw input panel generated", raw_panel_path.exists(), rel(raw_panel_path)),
        ("monthly factor panel generated", factor_path.exists(), rel(factor_path)),
        ("one row per symbol-month", one_row, f"duplicate_count={dup_count}"),
        ("factor coverage audit generated", coverage_path.exists(), rel(coverage_path)),
        ("factor distribution audit generated", distribution_path.exists(), rel(distribution_path)),
        ("final report generated", report_path.exists(), rel(report_path)),
        ("task completion card generated", card_path.exists(), rel(card_path)),
        ("FI_T5 not used as source", True, "FI_T5 not read."),
        ("no zscore/rank/winsorization performed", True, "Raw factors only."),
        ("no model/production files modified", True, "No model or production paths written."),
    ]
    qa = pd.DataFrame(qa_rows, columns=["check_name", "passed", "notes"])
    qa.to_csv(qa_path, index=False, encoding="utf-8-sig")
    qa.to_csv(qa_alias_path, index=False, encoding="utf-8-sig")

    summary = {
        "report_level_ttm_path": rel(ttm_path),
        "monthly_raw_input_panel_path": rel(raw_panel_path),
        "pit_clean_factor_panel_path": rel(factor_path),
        "factor_coverage_audit_path": rel(coverage_path),
        "factor_distribution_audit_path": rel(distribution_path),
        "report_path": rel(report_path),
        "task_completion_card_path": rel(card_path),
        "final_qa_path": rel(qa_path),
        "run_state_path": rel(RUN_DIR / "RUN_STATE.md"),
        "n_factor_panel_rows": int(len(factors)),
        "n_symbols": int(factors["symbol"].nunique()),
        "min_month_end": "" if pd.isna(min_month) else str(min_month.date()),
        "max_month_end": "" if pd.isna(max_month) else str(max_month.date()),
        "roe_coverage_mean": float(coverage["roe_coverage"].mean()) if len(coverage) else np.nan,
        "ep_coverage_mean": ep_cov_mean,
        "bp_coverage_mean": bp_cov_mean,
        "profit_growth_yoy_coverage_mean": float(coverage["profit_growth_yoy_coverage"].mean()) if len(coverage) else np.nan,
        "rev_growth_yoy_coverage_mean": float(coverage["rev_growth_yoy_coverage"].mean()) if len(coverage) else np.nan,
        "net_margin_coverage_mean": float(coverage["net_margin_coverage"].mean()) if len(coverage) else np.nan,
        "debt_ratio_coverage_mean": float(coverage["debt_ratio_coverage"].mean()) if len(coverage) else np.nan,
        "sales_expense_to_revenue_coverage_mean": float(coverage["sales_expense_to_revenue_coverage"].mean()) if len(coverage) else np.nan,
        "rd_expense_to_revenue_coverage_mean": float(coverage["rd_expense_to_revenue_coverage"].mean()) if len(coverage) else np.nan,
        "ep_bp_market_cap_block_removed": bool(ep_bp_unblocked),
        "selected_pit_date_violation_count": selected_pit_violations,
        "market_cap_date_violation_count": market_cap_violations,
        "one_row_per_symbol_month": bool(one_row),
        "recommended_next_task": "factor QA / sanity check; do not connect directly to production",
        "xlsx_read": False,
        "raw_daily_csv_read": False,
        "csmar_api_accessed": False,
        "download_executed": False,
        "readme_modified": False,
        "all_daily_modified": False,
        "training_panel_modified": False,
        "production_modified": False,
        "decision": decision,
    }
    terminal_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    outputs = [
        input_inventory_path,
        ttm_path,
        ttm_sample_path,
        raw_panel_path,
        raw_panel_sample_path,
        factor_path,
        factor_sample_path,
        coverage_path,
        distribution_path,
        report_path,
        card_path,
        qa_path,
        qa_alias_path,
        terminal_summary_path,
    ]
    update_run_state("completed", ["all requested outputs generated", f"decision: {decision}"], outputs, "task complete")
    append_checkpoint("completed", [f"generated {rel(terminal_summary_path)}", f"decision: {decision}"])

    del raw_panel, factors, coverage, distribution, inventory, qa
    gc.collect()

    for key in [
        "report_level_ttm_path",
        "monthly_raw_input_panel_path",
        "pit_clean_factor_panel_path",
        "factor_coverage_audit_path",
        "factor_distribution_audit_path",
        "report_path",
        "task_completion_card_path",
        "final_qa_path",
        "run_state_path",
        "n_factor_panel_rows",
        "n_symbols",
        "min_month_end",
        "max_month_end",
        "roe_coverage_mean",
        "ep_coverage_mean",
        "bp_coverage_mean",
        "profit_growth_yoy_coverage_mean",
        "rev_growth_yoy_coverage_mean",
        "net_margin_coverage_mean",
        "debt_ratio_coverage_mean",
        "sales_expense_to_revenue_coverage_mean",
        "rd_expense_to_revenue_coverage_mean",
        "ep_bp_market_cap_block_removed",
        "selected_pit_date_violation_count",
        "market_cap_date_violation_count",
        "one_row_per_symbol_month",
        "recommended_next_task",
        "xlsx_read",
        "raw_daily_csv_read",
        "csmar_api_accessed",
        "download_executed",
        "readme_modified",
        "all_daily_modified",
        "training_panel_modified",
        "production_modified",
        "decision",
    ]:
        print(f"{key}={summary[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
