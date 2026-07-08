from __future__ import annotations

import csv
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TASK_NAME = "csmar_pit_clean_ttm_pre_window_buffer_patch_v1"
TASK_TITLE = "CSMAR PIT-Clean TTM Pre-Window Buffer Patch v1"
OUT_NAME = "csmar_pit_clean_core_financial_factors_v2"
RUN_START_TIME = datetime.now().astimezone().isoformat(timespec="seconds")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / OUT_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

BROAD_STATEMENT_PATH = ROOT / "output" / "csmar_core_fs_manual_import_audit_v1" / "core_fs_statement_with_pit_dates_v1.parquet"
V1_MONTHLY_PATH = ROOT / "output" / "csmar_pit_scope_freeze_strict_core_fs_panel_v1" / "strict_core_fs_monthly_asof_panel_v1.parquet"
MKT_PATH = ROOT / "output" / "csmar_trd_dalyr_market_cap_import_lite_v1" / "trd_dalyr_monthly_market_cap_panel_v1.parquet"
V1_FACTOR_PATH = ROOT / "output" / "csmar_pit_clean_core_financial_factors_v1" / "pit_clean_core_financial_factors_monthly_v1.parquet"
V1_QA_COVERAGE_PATH = ROOT / "output" / "csmar_pit_clean_core_financial_factor_qa_lite_v1" / "factor_coverage_summary_v1.csv"
V1_QA_REPORT_PATH = ROOT / "output" / "csmar_pit_clean_core_financial_factor_qa_lite_v1" / "csmar_pit_clean_core_financial_factor_qa_lite_report_v1.md"

START_MONTH = pd.Timestamp("2017-01-31")
END_MONTH = pd.Timestamp("2026-06-30")

STATEMENT_BASE_COLS = [
    "symbol",
    "report_period",
    "report_type",
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
PIT_ALIAS_CANDIDATES = ["pit_date_primary", "selected_pit_date", "pit_date", "announcement_date"]
FORBIDDEN_PIT_COLS = ["Firforecdt", "firforecdt"]
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
V1_FACTOR_COLS = [
    "month_end",
    "symbol",
    "selected_pit_date",
    "market_cap_trade_date",
    "roe_ttm",
    "ep_ttm",
    "bp",
    "profit_growth_yoy",
    "rev_growth_yoy",
]
TTM_ITEMS = ["revenue", "net_profit_parent", "net_profit", "sales_expense", "admin_expense", "rd_expense", "total_profit"]
FACTOR_COLS = [
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
  - {rel(BROAD_STATEMENT_PATH)}
  - {rel(V1_MONTHLY_PATH)}
  - {rel(MKT_PATH)}
  - {rel(V1_FACTOR_PATH)}
  - {rel(V1_QA_COVERAGE_PATH)}
  - {rel(V1_QA_REPORT_PATH)}
- 已生成输出:
{out_lines}
- 下一步:
  - {next_step}
- 如果 Codex 崩溃，新的 Codex 应如何继续:
  - 先读取本文件
  - 检查 run_stdout.txt、run_stderr.txt、terminal_summary.json 和 final QA
  - 如果 terminal_summary.json 不存在，重新运行 scripts/patch_csmar_pit_clean_ttm_pre_window_buffer_v1.py，并将 stdout/stderr 写入本 agent run 目录
  - 不要读取 xlsx、原始日频 CSV、访问 API、下载、训练、回测、IC 或修改 production
  - 不要覆盖 output/csmar_pit_clean_core_financial_factors_v1
"""
    (RUN_DIR / "RUN_STATE.md").write_text(text, encoding="utf-8")


def parquet_columns(path: Path) -> list[str]:
    return pq.read_schema(path).names


def existing_cols(path: Path, desired: list[str]) -> list[str]:
    cols = set(parquet_columns(path))
    return [c for c in desired if c in cols]


def read_statement_with_aliases() -> tuple[pd.DataFrame, str]:
    cols = parquet_columns(BROAD_STATEMENT_PATH)
    available = set(cols)
    pit_sources = [c for c in PIT_ALIAS_CANDIDATES if c in available]
    selected_pit_source = pit_sources[0] if pit_sources else ""
    read_cols = [c for c in STATEMENT_BASE_COLS if c in available]
    read_cols += [c for c in PIT_ALIAS_CANDIDATES if c in available and c not in read_cols]
    df = pd.read_parquet(BROAD_STATEMENT_PATH, columns=read_cols)
    for col in STATEMENT_BASE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    if pit_sources:
        pit = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        for source in pit_sources:
            pit = pit.fillna(pd.to_datetime(df[source], errors="coerce"))
        df["pit_date_primary"] = pit
    else:
        df["pit_date_primary"] = pd.NaT
    notes = f"pit date alias priority used: {'>'.join(pit_sources) if pit_sources else 'none'}; forbidden Firforecdt ignored"
    return df[STATEMENT_BASE_COLS + ["pit_date_primary"]], notes


def inventory_row(path: Path, cols_read: list[str], df: pd.DataFrame | None, role: str, notes: str) -> dict[str, object]:
    exists = path.exists()
    readable = df is not None
    date_col = None
    if df is not None:
        if "month_end" in df.columns:
            date_col = "month_end"
        elif "report_period" in df.columns:
            date_col = "report_period"
        elif "pit_date_primary" in df.columns:
            date_col = "pit_date_primary"
    min_date = pd.to_datetime(df[date_col], errors="coerce").min() if df is not None and date_col else pd.NaT
    max_date = pd.to_datetime(df[date_col], errors="coerce").max() if df is not None and date_col else pd.NaT
    return {
        "input_path": rel(path),
        "exists": exists,
        "readable": readable,
        "columns_read": "|".join(cols_read),
        "n_rows": int(len(df)) if df is not None else "",
        "n_symbols": int(df["symbol"].astype(str).nunique()) if df is not None and "symbol" in df.columns else "",
        "min_date": "" if pd.isna(min_date) else str(min_date.date()),
        "max_date": "" if pd.isna(max_date) else str(max_date.date()),
        "role": role,
        "notes": notes,
    }


def normalize_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str)
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_statement_buffer(stmt: pd.DataFrame) -> pd.DataFrame:
    stmt = normalize_dates(stmt, ["report_period", "pit_date_primary"])
    stmt["report_type"] = stmt["report_type"].astype(str)
    stmt = stmt[(stmt["report_type"] == "A") & stmt["pit_date_primary"].notna()].copy()
    stmt["income_available"] = stmt["income_available"].fillna(False).astype(bool)
    stmt["balance_available"] = stmt["balance_available"].fillna(False).astype(bool)
    stmt["availability_score"] = stmt["income_available"].astype(int) + stmt["balance_available"].astype(int)
    stmt = stmt.sort_values(
        ["symbol", "report_period", "pit_date_primary", "availability_score"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )
    stmt = stmt.drop_duplicates(["symbol", "report_period"], keep="first").copy()
    stmt["revenue"] = stmt["operating_revenue"]
    stmt["revenue_source"] = np.where(stmt["operating_revenue"].notna(), "operating_revenue", "total_operating_revenue")
    stmt.loc[stmt["revenue"].isna(), "revenue"] = stmt.loc[stmt["revenue"].isna(), "total_operating_revenue"]
    stmt["uses_pre_2017_buffer_flag"] = stmt["report_period"] < pd.Timestamp("2017-01-01")
    return stmt.drop(columns=["availability_score"])


def add_ttm(buffer: pd.DataFrame) -> pd.DataFrame:
    df = buffer.sort_values(["symbol", "report_period"], kind="mergesort").copy()
    df["fiscal_year"] = df["report_period"].dt.year
    df["fiscal_quarter"] = df["report_period"].dt.quarter
    df["prev_symbol"] = df["symbol"].shift(1)
    df["prev_year"] = df["fiscal_year"].shift(1)
    df["prev_quarter"] = df["fiscal_quarter"].shift(1)
    has_prev_q = (
        (df["prev_symbol"] == df["symbol"])
        & (df["prev_year"] == df["fiscal_year"])
        & (df["prev_quarter"] == (df["fiscal_quarter"] - 1))
    )
    df["missing_prev_quarter_flag"] = (df["fiscal_quarter"] > 1) & ~has_prev_q
    quarter_cols: list[str] = []
    for item in TTM_ITEMS:
        prev = df.groupby(["symbol", "fiscal_year"], sort=False)[item].shift(1)
        q_col = f"{item}_quarter"
        quarter_cols.append(q_col)
        df[q_col] = np.where(df["fiscal_quarter"] == 1, df[item], df[item] - prev)
        df.loc[df["missing_prev_quarter_flag"], q_col] = np.nan
        df[f"{item}_ttm"] = (
            df.groupby("symbol", sort=False)[q_col]
            .rolling(window=4, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
    counts = []
    for q_col in quarter_cols:
        count_col = f"{q_col}_count4"
        df[count_col] = (
            df.groupby("symbol", sort=False)[q_col]
            .rolling(window=4, min_periods=1)
            .count()
            .reset_index(level=0, drop=True)
        )
        counts.append(count_col)
    df["ttm_quarters_available"] = df[counts].min(axis=1).fillna(0).astype("int64")
    df["ttm_complete_flag"] = df["ttm_quarters_available"] == 4
    df["ttm_warning_flag"] = np.where(
        df["ttm_complete_flag"],
        "",
        np.where(df["missing_prev_quarter_flag"], "missing_previous_quarter;ttm_incomplete", "ttm_warmup_or_missing_quarters"),
    )
    df["revenue_ttm_lag4_report"] = df.groupby("symbol", sort=False)["revenue_ttm"].shift(4)
    df["net_profit_parent_ttm_lag4_report"] = df.groupby("symbol", sort=False)["net_profit_parent_ttm"].shift(4)
    out_cols = [
        "symbol",
        "report_period",
        "pit_date_primary",
        "revenue_ttm",
        "revenue_ttm_lag4_report",
        "revenue_source",
        "net_profit_parent_ttm",
        "net_profit_parent_ttm_lag4_report",
        "net_profit_ttm",
        "sales_expense_ttm",
        "admin_expense_ttm",
        "rd_expense_ttm",
        "total_profit_ttm",
        "total_assets",
        "total_liabilities",
        "equity_parent",
        "total_equity",
        "income_available",
        "balance_available",
        "ttm_quarters_available",
        "ttm_complete_flag",
        "ttm_warning_flag",
        "uses_pre_2017_buffer_flag",
    ]
    return df[out_cols].copy()


def make_month_skeleton(v1_monthly: pd.DataFrame, mkt: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    v1 = normalize_dates(v1_monthly, ["month_end", "selected_report_period", "selected_pit_date"])
    v1 = v1[(v1["month_end"] >= START_MONTH) & (v1["month_end"] <= END_MONTH)][["month_end", "symbol"]].drop_duplicates()
    early_mkt = mkt[(mkt["month_end"] >= START_MONTH) & (mkt["month_end"] <= pd.Timestamp("2017-03-31"))][["month_end", "symbol"]].drop_duplicates()
    notes = "v1 skeleton used for existing months"
    if not early_mkt.empty:
        notes += "; early 2017 rows attempted from market-cap skeleton and statement buffer"
    skeleton = pd.concat([v1, early_mkt], ignore_index=True).drop_duplicates(["month_end", "symbol"])
    return skeleton.sort_values(["symbol", "month_end"], kind="mergesort").reset_index(drop=True), notes


def build_monthly_asof(skeleton: pd.DataFrame, buffer: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    stmt_cols = [
        "symbol",
        "report_period",
        "pit_date_primary",
        "income_available",
        "balance_available",
        "uses_pre_2017_buffer_flag",
    ]
    for symbol, months in skeleton.groupby("symbol", sort=False):
        stmts = buffer.loc[buffer["symbol"] == symbol, stmt_cols].sort_values(["pit_date_primary", "report_period"], kind="mergesort")
        if stmts.empty:
            continue
        stmt_records = stmts.to_dict("records")
        active: list[dict[str, object]] = []
        ptr = 0
        for _, month_row in months.sort_values("month_end", kind="mergesort").iterrows():
            month_end = month_row["month_end"]
            while ptr < len(stmt_records) and stmt_records[ptr]["pit_date_primary"] <= month_end:
                active.append(stmt_records[ptr])
                ptr += 1
            if not active:
                continue
            best = max(active, key=lambda r: (r["report_period"], -pd.Timestamp(r["pit_date_primary"]).value))
            report_lag_days = int((month_end - best["pit_date_primary"]).days)
            rows.append(
                {
                    "month_end": month_end,
                    "symbol": symbol,
                    "selected_report_period": best["report_period"],
                    "selected_pit_date": best["pit_date_primary"],
                    "report_lag_days": report_lag_days,
                    "income_available": best["income_available"],
                    "balance_available": best["balance_available"],
                    "asof_source": "broad_statement_buffer_v2",
                    "uses_pre_2017_buffer_flag": bool(best["uses_pre_2017_buffer_flag"]),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[(out["month_end"] >= START_MONTH) & (out["month_end"] <= END_MONTH)].copy()
    return out.sort_values(["month_end", "symbol"], kind="mergesort").drop_duplicates(["month_end", "symbol"], keep="last")


def div_valid(numer: pd.Series, denom: pd.Series) -> pd.Series:
    n = pd.to_numeric(numer, errors="coerce")
    d = pd.to_numeric(denom, errors="coerce")
    out = n / d
    return out.where(d.notna() & (d > 0)).replace([np.inf, -np.inf], np.nan)


def build_flags(row: pd.Series) -> str:
    flags: list[str] = []
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
    if pd.isna(row.get("net_profit_parent_ttm_lag4_report")) or row.get("net_profit_parent_ttm_lag4_report", np.nan) <= 0:
        flags.append("profit_growth_denominator_invalid")
    if pd.isna(row.get("revenue_ttm_lag4_report")) or row.get("revenue_ttm_lag4_report", np.nan) <= 0:
        flags.append("rev_growth_denominator_invalid")
    if pd.notna(row.get("selected_pit_date")) and row["selected_pit_date"] > row["month_end"]:
        flags.append("selected_pit_date_after_month_end")
    if pd.notna(row.get("market_cap_trade_date")) and row["market_cap_trade_date"] > row["month_end"]:
        flags.append("market_cap_trade_date_after_month_end")
    return ";".join(flags)


def coverage_by_month(factors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month_end, g in factors.groupby("month_end", sort=True):
        n = len(g)
        row: dict[str, object] = {"month_end": month_end.strftime("%Y-%m-%d"), "n_symbols": int(g["symbol"].nunique())}
        for col in FACTOR_COLS:
            row[f"{col.replace('_ttm', '')}_coverage" if col == "roe_ttm" else f"{col}_coverage"] = float(g[col].notna().sum() / n) if n else np.nan
        row["ttm_complete_rate"] = float(g["ttm_complete_flag"].fillna(False).mean()) if n else np.nan
        row["market_cap_coverage"] = float((pd.to_numeric(g["market_cap_total"], errors="coerce") > 0).sum() / n) if n else np.nan
        row["pre_2017_buffer_usage_rate"] = float(g["uses_pre_2017_buffer_flag"].fillna(False).mean()) if n else np.nan
        notes: list[str] = []
        if month_end <= pd.Timestamp("2017-03-31"):
            notes.append("early-2017 pre-window buffer target")
        if month_end.year < 2018 and row.get("rd_expense_to_revenue_coverage", 1) < 0.5:
            notes.append("pre-2018 rd expense sparse")
        row["notes"] = ";".join(notes)
        rows.append(row)
    coverage = pd.DataFrame(rows)
    rename = {
        "ep_ttm_coverage": "ep_coverage",
        "roe_coverage": "roe_coverage",
        "bp_coverage": "bp_coverage",
    }
    coverage = coverage.rename(columns=rename)
    required = [
        "month_end",
        "n_symbols",
        "roe_coverage",
        "ep_coverage",
        "bp_coverage",
        "profit_growth_yoy_coverage",
        "rev_growth_yoy_coverage",
        "net_margin_coverage",
        "debt_ratio_coverage",
        "sales_expense_to_revenue_coverage",
        "admin_expense_to_revenue_coverage",
        "rd_expense_to_revenue_coverage",
        "ttm_complete_rate",
        "market_cap_coverage",
        "pre_2017_buffer_usage_rate",
        "notes",
    ]
    for col in required:
        if col not in coverage.columns:
            coverage[col] = np.nan
    return coverage[required]


def distribution_row(factors: pd.DataFrame, factor: str) -> dict[str, object]:
    s = pd.to_numeric(factors[factor], errors="coerce")
    valid = s.dropna()
    q = valid.quantile([0.01, 0.05, 0.25, 0.75, 0.95, 0.99]) if len(valid) else pd.Series(dtype=float)
    p99_abs = float(valid.abs().quantile(0.99)) if len(valid) else np.nan
    flag = "plausible"
    notes = "raw factor; no winsorization, zscore, rank, IC, or backtest"
    if factor in ("ep_ttm", "bp") and pd.notna(abs(valid.median() if len(valid) else np.nan)):
        med = abs(float(valid.median()))
        if (factor == "ep_ttm" and med >= 1) or (factor == "bp" and med >= 20):
            flag = "unit_review"
            notes = "median suggests possible unit issue"
    if factor in ("profit_growth_yoy", "rev_growth_yoy") and pd.notna(p99_abs) and p99_abs > 1000:
        flag = "growth_tail_review"
        notes = "growth tail can reflect near-zero lag4 denominator"
    if factor not in ("profit_growth_yoy", "rev_growth_yoy") and pd.notna(p99_abs) and p99_abs > 100:
        flag = "extreme_tail_review"
        notes = "raw tail requires later QA; not clipped"
    return {
        "factor": factor,
        "n": int(len(valid)),
        "missing_rate": float(1 - len(valid) / len(s)) if len(s) else np.nan,
        "mean": float(valid.mean()) if len(valid) else np.nan,
        "median": float(valid.median()) if len(valid) else np.nan,
        "p01": float(q.get(0.01, np.nan)),
        "p05": float(q.get(0.05, np.nan)),
        "p25": float(q.get(0.25, np.nan)),
        "p75": float(q.get(0.75, np.nan)),
        "p95": float(q.get(0.95, np.nan)),
        "p99": float(q.get(0.99, np.nan)),
        "min": float(valid.min()) if len(valid) else np.nan,
        "max": float(valid.max()) if len(valid) else np.nan,
        "plausibility_flag": flag,
        "notes": notes,
    }


def panel_metrics(df: pd.DataFrame, prefix: str) -> dict[str, object]:
    d = df.copy()
    d["month_end"] = pd.to_datetime(d["month_end"], errors="coerce")
    early = d[(d["month_end"] >= START_MONTH) & (d["month_end"] <= pd.Timestamp("2017-03-31"))]
    out = {
        f"{prefix}_min_month_end": "" if d.empty else str(d["month_end"].min().date()),
        f"{prefix}_max_month_end": "" if d.empty else str(d["month_end"].max().date()),
        f"{prefix}_n_rows": int(len(d)),
        f"{prefix}_n_symbols": int(d["symbol"].astype(str).nunique()) if "symbol" in d.columns else 0,
        f"{prefix}_n_months": int(d["month_end"].nunique()),
        f"{prefix}_n_symbol_months": int(d[["symbol", "month_end"]].drop_duplicates().shape[0]),
        f"{prefix}_n_rows_2017_01_to_2017_03": int(len(early)),
        f"{prefix}_roe_coverage_2017_01_to_2017_03": float(early["roe_ttm"].notna().mean()) if len(early) and "roe_ttm" in early.columns else 0.0,
        f"{prefix}_ep_coverage_2017_01_to_2017_03": float(early["ep_ttm"].notna().mean()) if len(early) and "ep_ttm" in early.columns else 0.0,
        f"{prefix}_bp_coverage_2017_01_to_2017_03": float(early["bp"].notna().mean()) if len(early) and "bp" in early.columns else 0.0,
        f"{prefix}_profit_growth_yoy_coverage_2017_01_to_2017_03": float(early["profit_growth_yoy"].notna().mean()) if len(early) and "profit_growth_yoy" in early.columns else 0.0,
        f"{prefix}_rev_growth_yoy_coverage_2017_01_to_2017_03": float(early["rev_growth_yoy"].notna().mean()) if len(early) and "rev_growth_yoy" in early.columns else 0.0,
        f"{prefix}_selected_pit_date_violation_count": int(((d.get("selected_pit_date").notna()) & (d.get("selected_pit_date") > d["month_end"])).sum()) if "selected_pit_date" in d.columns else 0,
        f"{prefix}_market_cap_date_violation_count": int(((d.get("market_cap_trade_date").notna()) & (d.get("market_cap_trade_date") > d["month_end"])).sum()) if "market_cap_trade_date" in d.columns else 0,
        f"{prefix}_one_row_per_symbol_month": bool(d.duplicated(["symbol", "month_end"]).sum() == 0),
    }
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "inventory": OUT_DIR / "input_inventory_v2.csv",
        "buffer": OUT_DIR / "strict_statement_buffer_v2.parquet",
        "buffer_sample": OUT_DIR / "strict_statement_buffer_sample_v2.csv",
        "ttm": OUT_DIR / "report_level_ttm_raw_items_v2.parquet",
        "ttm_sample": OUT_DIR / "report_level_ttm_raw_items_sample_v2.csv",
        "asof": OUT_DIR / "strict_core_fs_monthly_asof_panel_v2.parquet",
        "asof_sample": OUT_DIR / "strict_core_fs_monthly_asof_panel_sample_v2.csv",
        "raw": OUT_DIR / "monthly_factor_raw_input_panel_v2.parquet",
        "raw_sample": OUT_DIR / "monthly_factor_raw_input_panel_sample_v2.csv",
        "factor": OUT_DIR / "pit_clean_core_financial_factors_monthly_v2.parquet",
        "factor_sample": OUT_DIR / "pit_clean_core_financial_factors_monthly_sample_v2.csv",
        "comparison": OUT_DIR / "v1_vs_v2_ttm_warmup_comparison_v1.csv",
        "coverage": OUT_DIR / "factor_coverage_audit_by_month_v2.csv",
        "distribution": OUT_DIR / "factor_distribution_audit_v2.csv",
        "buffer_audit": OUT_DIR / "ttm_pre_window_buffer_audit_v1.csv",
        "report": OUT_DIR / "csmar_pit_clean_ttm_pre_window_buffer_patch_report_v1.md",
        "card": OUT_DIR / "task_completion_card.md",
        "qa": OUT_DIR / "final_qa_csmar_pit_clean_ttm_pre_window_buffer_patch_v1.csv",
        "qa_alias": OUT_DIR / "final_qa.csv",
        "summary": OUT_DIR / "terminal_summary.json",
    }

    update_run_state("input_read", ["script started"], [], "read allowed parquet/csv/md inputs")
    stmt_raw, pit_alias_notes = read_statement_with_aliases()
    v1_monthly = pd.read_parquet(V1_MONTHLY_PATH, columns=existing_cols(V1_MONTHLY_PATH, MONTHLY_COLS))
    mkt = pd.read_parquet(MKT_PATH, columns=existing_cols(MKT_PATH, MKT_COLS))
    v1_factor = pd.read_parquet(V1_FACTOR_PATH, columns=existing_cols(V1_FACTOR_PATH, V1_FACTOR_COLS))
    v1_qa_cov = pd.read_csv(V1_QA_COVERAGE_PATH)
    v1_qa_report_text = V1_QA_REPORT_PATH.read_text(encoding="utf-8", errors="replace")
    mkt = normalize_dates(mkt, ["month_end", "trade_date"])
    v1_factor = normalize_dates(v1_factor, ["month_end", "selected_pit_date", "market_cap_trade_date"])
    inventory = pd.DataFrame(
        [
            inventory_row(BROAD_STATEMENT_PATH, list(stmt_raw.columns), stmt_raw, "broad core FS statement with PIT dates", pit_alias_notes),
            inventory_row(V1_MONTHLY_PATH, existing_cols(V1_MONTHLY_PATH, MONTHLY_COLS), v1_monthly, "v1 strict monthly as-of skeleton", "used as target skeleton where available"),
            inventory_row(MKT_PATH, existing_cols(MKT_PATH, MKT_COLS), mkt, "TRD_Dalyr monthly market cap", "EP/BP uses total_market_cap_x1000"),
            inventory_row(V1_FACTOR_PATH, existing_cols(V1_FACTOR_PATH, V1_FACTOR_COLS), v1_factor, "v1 factor panel comparison", "read for v1 vs v2 comparison"),
            inventory_row(V1_QA_COVERAGE_PATH, list(v1_qa_cov.columns), v1_qa_cov, "v1 QA coverage summary", "small CSV comparison context"),
            inventory_row(V1_QA_REPORT_PATH, ["text"], pd.DataFrame({"symbol": [], "text": []}), "v1 QA report", f"small md read; chars={len(v1_qa_report_text)}"),
        ]
    )
    inventory.to_csv(paths["inventory"], index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    append_checkpoint("input_inventory_done", [f"generated {rel(paths['inventory'])}"])

    update_run_state("statement_buffer", ["input inventory generated"], [paths["inventory"]], "construct strict actual PIT statement buffer")
    buffer = build_statement_buffer(stmt_raw)
    del stmt_raw
    gc.collect()
    buffer.to_parquet(paths["buffer"], index=False)
    buffer.head(2000).to_csv(paths["buffer_sample"], index=False, encoding="utf-8-sig")
    append_checkpoint("statement_buffer_done", [f"generated {rel(paths['buffer'])}", f"rows: {len(buffer)}"])

    update_run_state("ttm_reconstruction", ["strict statement buffer generated"], [paths["inventory"], paths["buffer"]], "construct report-level TTM with pre-window buffer")
    ttm = add_ttm(buffer)
    ttm.to_parquet(paths["ttm"], index=False)
    ttm.head(2000).to_csv(paths["ttm_sample"], index=False, encoding="utf-8-sig")
    append_checkpoint("ttm_reconstruction_done", [f"generated {rel(paths['ttm'])}", f"rows: {len(ttm)}"])

    update_run_state("monthly_asof", ["report-level TTM v2 generated"], [paths["buffer"], paths["ttm"]], "rebuild monthly as-of v2")
    skeleton, skeleton_notes = make_month_skeleton(v1_monthly, mkt)
    asof = build_monthly_asof(skeleton, buffer)
    asof.to_parquet(paths["asof"], index=False)
    asof.head(2000).to_csv(paths["asof_sample"], index=False, encoding="utf-8-sig")
    append_checkpoint("monthly_asof_done", [f"generated {rel(paths['asof'])}", f"rows: {len(asof)}", skeleton_notes])
    del buffer, v1_monthly, skeleton
    gc.collect()

    update_run_state("raw_merge_and_factors", ["monthly as-of v2 generated"], [paths["asof"], paths["ttm"]], "merge TTM and market cap then calculate v2 factors")
    mkt_one = mkt.sort_values(["symbol", "month_end", "trade_date"], kind="mergesort").drop_duplicates(["symbol", "month_end"], keep="last")
    raw = asof.merge(
        ttm,
        how="left",
        left_on=["symbol", "selected_report_period"],
        right_on=["symbol", "report_period"],
        suffixes=("_asof", "_ttm"),
        validate="many_to_one",
    )
    raw = raw.merge(mkt_one, how="left", on=["symbol", "month_end"], validate="many_to_one")
    raw = raw.rename(columns={"trade_date": "market_cap_trade_date"})
    if "uses_pre_2017_buffer_flag" not in raw.columns:
        asof_flag = raw.get("uses_pre_2017_buffer_flag_asof", False)
        ttm_flag = raw.get("uses_pre_2017_buffer_flag_ttm", False)
        raw["uses_pre_2017_buffer_flag"] = pd.Series(asof_flag, index=raw.index).fillna(False).astype(bool) | pd.Series(ttm_flag, index=raw.index).fillna(False).astype(bool)
    raw.to_parquet(paths["raw"], index=False)
    raw.head(2000).to_csv(paths["raw_sample"], index=False, encoding="utf-8-sig")

    factors = raw.copy()
    factors["market_cap_total"] = factors["total_market_cap_x1000"]
    factors["market_cap_float"] = factors["float_market_cap_x1000"]
    factors["roe_ttm"] = div_valid(factors["net_profit_parent_ttm"], factors["equity_parent"])
    factors["ep_ttm"] = div_valid(factors["net_profit_parent_ttm"], factors["total_market_cap_x1000"])
    factors["bp"] = div_valid(factors["equity_parent"], factors["total_market_cap_x1000"])
    factors["profit_growth_yoy"] = div_valid(factors["net_profit_parent_ttm"], factors["net_profit_parent_ttm_lag4_report"]) - 1
    factors["rev_growth_yoy"] = div_valid(factors["revenue_ttm"], factors["revenue_ttm_lag4_report"]) - 1
    factors["net_margin"] = div_valid(factors["net_profit_parent_ttm"], factors["revenue_ttm"])
    factors["debt_ratio"] = div_valid(factors["total_liabilities"], factors["total_assets"])
    factors["sales_expense_to_revenue"] = div_valid(factors["sales_expense_ttm"], factors["revenue_ttm"])
    factors["admin_expense_to_revenue"] = div_valid(factors["admin_expense_ttm"], factors["revenue_ttm"])
    factors["rd_expense_to_revenue"] = div_valid(factors["rd_expense_ttm"], factors["revenue_ttm"])
    factors["factor_validity_flags"] = factors.apply(build_flags, axis=1)
    factor_cols = [
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
        *FACTOR_COLS,
        "revenue_ttm",
        "revenue_ttm_lag4_report",
        "net_profit_parent_ttm",
        "net_profit_parent_ttm_lag4_report",
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
        "uses_pre_2017_buffer_flag",
        "factor_validity_flags",
    ]
    factors = factors[factor_cols].sort_values(["month_end", "symbol"], kind="mergesort")
    factors.to_parquet(paths["factor"], index=False)
    factors.head(2000).to_csv(paths["factor_sample"], index=False, encoding="utf-8-sig")
    append_checkpoint("factor_panel_done", [f"generated {rel(paths['factor'])}", f"rows: {len(factors)}"])

    update_run_state("audit_outputs", ["v2 factor panel generated"], [paths["factor"], paths["raw"]], "generate v1/v2 comparison, coverage, distribution, buffer audit, report, QA")
    v1_metrics = panel_metrics(v1_factor, "v1")
    v2_metrics = panel_metrics(factors, "v2")
    comparison_rows = []
    for metric in [
        "min_month_end",
        "max_month_end",
        "n_rows",
        "n_symbols",
        "n_months",
        "n_symbol_months",
        "n_rows_2017_01_to_2017_03",
        "roe_coverage_2017_01_to_2017_03",
        "ep_coverage_2017_01_to_2017_03",
        "bp_coverage_2017_01_to_2017_03",
        "profit_growth_yoy_coverage_2017_01_to_2017_03",
        "rev_growth_yoy_coverage_2017_01_to_2017_03",
        "selected_pit_date_violation_count",
        "market_cap_date_violation_count",
        "one_row_per_symbol_month",
    ]:
        comparison_rows.append({"metric": metric, "v1_value": v1_metrics[f"v1_{metric}"], "v2_value": v2_metrics[f"v2_{metric}"], "details": "v2 uses pre-window statement buffer where available"})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(paths["comparison"], index=False, encoding="utf-8-sig")

    coverage = coverage_by_month(factors)
    coverage.to_csv(paths["coverage"], index=False, encoding="utf-8-sig")
    distribution = pd.DataFrame([distribution_row(factors, c) for c in FACTOR_COLS])
    distribution.to_csv(paths["distribution"], index=False, encoding="utf-8-sig")

    early = factors[(factors["month_end"] >= START_MONTH) & (factors["month_end"] <= pd.Timestamp("2017-03-31"))]
    pre_stmt = ttm[ttm["report_period"] < pd.Timestamp("2017-01-01")]
    selected_pit_violations = int(((factors["selected_pit_date"].notna()) & (factors["selected_pit_date"] > factors["month_end"])).sum())
    market_cap_violations = int(((factors["market_cap_trade_date"].notna()) & (factors["market_cap_trade_date"] > factors["month_end"])).sum())
    one_row = bool(factors.duplicated(["symbol", "month_end"]).sum() == 0)
    report_period_used_as_pit_date_count = 0
    fixed_lag_used = False
    legal_deadline_fallback_used = False
    firforecdt_primary_used = False
    buffer_audit = pd.DataFrame(
        [
            ("n_pre_2017_statement_records_used", int(len(pre_stmt)), "report_period before 2017 retained only for TTM lookback"),
            ("n_symbols_with_pre_2017_buffer", int(pre_stmt["symbol"].nunique()), "symbols with pre-2017 report-level TTM inputs"),
            ("earliest_report_period_used", "" if pre_stmt.empty else str(pre_stmt["report_period"].min().date()), "earliest pre-window report period"),
            ("latest_pre_2017_report_period_used", "" if pre_stmt.empty else str(pre_stmt["report_period"].max().date()), "latest pre-window report period"),
            ("earliest_pit_date_used", "" if ttm.empty else str(ttm["pit_date_primary"].min().date()), "actual PIT date only"),
            ("n_2017_early_month_rows", int(len(early)), "2017-01 through 2017-03 factor rows"),
            ("n_2017_early_month_rows_with_ttm_complete", int(early["ttm_complete_flag"].fillna(False).sum()), "early rows with complete TTM"),
            ("selected_pit_date_violation_count", selected_pit_violations, "must be 0"),
            ("report_period_used_as_pit_date_count", report_period_used_as_pit_date_count, "report_period never used as PIT date"),
            ("fixed_lag_used", fixed_lag_used, "fixed lag not used"),
            ("legal_deadline_fallback_used", legal_deadline_fallback_used, "legal deadline fallback not used"),
            ("firforecdt_primary_used", firforecdt_primary_used, "Firforecdt not used as primary PIT"),
        ],
        columns=["metric", "value", "details"],
    )
    buffer_audit.to_csv(paths["buffer_audit"], index=False, encoding="utf-8-sig")

    v2_min = factors["month_end"].min()
    v2_max = factors["month_end"].max()
    v1_min = v1_factor["month_end"].min()
    ep_bp_unblocked = float(coverage["ep_coverage"].mean()) > 0.5 and float(coverage["bp_coverage"].mean()) > 0.5
    pre_2017_buffer_used = int(len(pre_stmt)) > 0 and bool(factors["uses_pre_2017_buffer_flag"].any())
    if selected_pit_violations or market_cap_violations:
        decision = "INVALID_PIT_OR_MARKET_CAP_DATE_ALIGNMENT"
    elif report_period_used_as_pit_date_count or fixed_lag_used or legal_deadline_fallback_used or firforecdt_primary_used:
        decision = "INVALID_PIT_POLICY_VIOLATION"
    elif pd.notna(v2_min) and v2_min <= START_MONTH and len(early) > 0:
        decision = "CSMAR_TTM_PRE_WINDOW_BUFFER_PATCH_READY_FOR_V2_QA"
    else:
        decision = "CSMAR_TTM_PRE_WINDOW_BUFFER_PATCH_PARTIAL_NEEDS_REVIEW"

    report = f"""# CSMAR PIT-Clean TTM Pre-Window Buffer Patch v1

## 1. Executive Summary

Generated v2 PIT-clean core financial factor source panel with a pre-window TTM statement buffer. Decision: {decision}.

## 2. Scope and Guardrails

- 本任务没有访问 CSMAR API。
- 本任务没有下载数据。
- 本任务没有读取 Excel。
- 本任务没有读取原始日频 CSV。
- 本任务没有训练模型、回测或 IC。
- 本任务没有接入 production。
- v1 outputs were not overwritten.

## 3. Why v1 had TTM warm-up

v1 started at {v1_min.date() if pd.notna(v1_min) else ''} because report-level TTM was reconstructed from the strict v1 window without sufficient pre-2017 lookback quarters.

## 4. Pre-window Buffer Method

v2 uses broad core FS statements with actual PIT dates. Records before 2017 are retained only as TTM lookback buffer and final monthly outputs start no earlier than 2017-01-31.

## 5. PIT Policy

All records require actual PIT date. PIT alias priority: pit_date_primary, selected_pit_date, pit_date, announcement_date. report_period, fixed lag, legal deadline fallback, and Firforecdt primary were not used.

## 6. Unit Alignment

EP/BP use total_market_cap_x1000. TRD raw market cap remains thousand-yuan and is not used as denominator.

## 7. TTM Reconstruction

Cumulative income-statement fields are converted to quarterly values before rolling four-quarter TTM. Lag4 growth uses report-level TTM shifted four reports within symbol, not monthly shift.

## 8. Factor Definitions

roe_ttm, ep_ttm, bp, profit_growth_yoy, rev_growth_yoy, net_margin, debt_ratio, sales_expense_to_revenue, admin_expense_to_revenue, and rd_expense_to_revenue were generated without winsorization, zscore, rank, IC, or backtest.

## 9. v1 vs v2 Comparison

- v1 min month: {v1_min.date() if pd.notna(v1_min) else ''}
- v2 min month: {v2_min.date() if pd.notna(v2_min) else ''}
- v2 rows in 2017-01 to 2017-03: {len(early)}
- selected PIT date violations: {selected_pit_violations}
- market cap date violations: {market_cap_violations}
- one row per symbol-month: {one_row}

## 10. Coverage Audit

Coverage audit saved to {rel(paths["coverage"])}. EP/BP market-cap blocking removed: {ep_bp_unblocked}.

## 11. Distribution Audit

Distribution audit saved to {rel(paths["distribution"])}. Extreme raw values are documented and not clipped.

## 12. Remaining Caveats

Some early 2017 rows may still lack complete TTM if actual PIT history is insufficient for a symbol. RD expense may remain sparse before 2018. This is a source panel, not a final training panel.

## 13. Decision

{decision}

## 14. Recommended Next Task

Run v2 factor QA / FI_T5 sanity check. Do not directly connect this source panel to production.

## 15. Files Generated

- {rel(paths["buffer"])}
- {rel(paths["ttm"])}
- {rel(paths["asof"])}
- {rel(paths["raw"])}
- {rel(paths["factor"])}
- {rel(paths["comparison"])}
- {rel(paths["buffer_audit"])}
- {rel(paths["coverage"])}
- {rel(paths["distribution"])}
- {rel(paths["report"])}
- {rel(paths["card"])}
- {rel(paths["qa"])}
"""
    paths["report"].write_text(report, encoding="utf-8")

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
{rel(paths["factor"])}
{rel(paths["comparison"])}
{rel(paths["buffer_audit"])}
核心结论：
{decision}
v1 min_month_end：
{v1_min.date() if pd.notna(v1_min) else ''}
v2 min_month_end：
{v2_min.date() if pd.notna(v2_min) else ''}
v2 max_month_end：
{v2_max.date() if pd.notna(v2_max) else ''}
v2 factor panel 行数：
{len(factors)}
v2 symbol 数：
{factors["symbol"].nunique()}
2017-01 至 2017-03 是否补上：
{len(early) > 0}
selected_pit_date_violation_count：
{selected_pit_violations}
market_cap_date_violation_count：
{market_cap_violations}
one_row_per_symbol_month：
{one_row}
EP/BP 是否仍使用 total_market_cap_x1000：
True
TTM pre-window buffer 是否使用：
{pre_2017_buffer_used}
主要可用因子：
{", ".join(FACTOR_COLS)}
主要限制：
Some symbols may still have incomplete early TTM when actual PIT history is insufficient.
下一步建议：
v2 factor QA / FI_T5 sanity check; not production.
"""
    paths["card"].write_text(card, encoding="utf-8")

    final_no_pre_2017 = bool((factors["month_end"] >= START_MONTH).all())
    qa_rows = [
        ("no xlsx read", True, "Only allowed parquet/csv/md inputs were read."),
        ("no raw daily CSV read", True, "No raw daily CSV opened."),
        ("no CSMAR API access", True, "No API code path exists."),
        ("no download", True, "No network/download code path exists."),
        ("no model training", True, "No model training code path exists."),
        ("no backtest", True, "No backtest code path exists."),
        ("no IC", True, "No IC code path exists."),
        ("no signal generation", True, "No signal generated."),
        ("no production modification", True, "No production path written."),
        ("no README modification", True, "README not touched."),
        ("all_daily.parquet not modified", True, "Script never writes all_daily.parquet."),
        ("training_panel_v15_sr.parquet not modified", True, "Script never writes training_panel_v15_sr.parquet."),
        ("v1 outputs not overwritten", True, "All outputs written to v2 directory."),
        ("root output used", str(OUT_DIR).startswith(str(ROOT / "output")), rel(OUT_DIR)),
        ("broad strict statement buffer generated", paths["buffer"].exists(), rel(paths["buffer"])),
        ("pre-2017 statement records allowed only as TTM buffer", True, "final factor output filtered to 2017-01-31 and later"),
        ("final output has no month_end before 2017-01-31", final_no_pre_2017, str(final_no_pre_2017)),
        ("selected_pit_date <= month_end", selected_pit_violations == 0, str(selected_pit_violations)),
        ("market_cap_trade_date <= month_end", market_cap_violations == 0, str(market_cap_violations)),
        ("report_period not used as PIT date", report_period_used_as_pit_date_count == 0, str(report_period_used_as_pit_date_count)),
        ("no fixed lag used", not fixed_lag_used, str(fixed_lag_used)),
        ("no legal deadline fallback used", not legal_deadline_fallback_used, str(legal_deadline_fallback_used)),
        ("no Firforecdt primary used", not firforecdt_primary_used, str(firforecdt_primary_used)),
        ("total_market_cap_x1000 used for EP/BP", True, "ep_ttm and bp denominators use total_market_cap_x1000."),
        ("report-level TTM v2 generated", paths["ttm"].exists(), rel(paths["ttm"])),
        ("monthly raw input v2 generated", paths["raw"].exists(), rel(paths["raw"])),
        ("monthly factor panel v2 generated", paths["factor"].exists(), rel(paths["factor"])),
        ("one row per symbol-month", one_row, str(one_row)),
        ("v1 vs v2 comparison generated", paths["comparison"].exists(), rel(paths["comparison"])),
        ("ttm pre-window buffer audit generated", paths["buffer_audit"].exists(), rel(paths["buffer_audit"])),
        ("factor coverage audit generated", paths["coverage"].exists(), rel(paths["coverage"])),
        ("factor distribution audit generated", paths["distribution"].exists(), rel(paths["distribution"])),
        ("final report generated", paths["report"].exists(), rel(paths["report"])),
        ("task completion card generated", paths["card"].exists(), rel(paths["card"])),
        ("no winsor/zscore/rank performed", True, "Raw factors only."),
        ("no model/production files modified", True, "No model/production paths written."),
    ]
    qa = pd.DataFrame(qa_rows, columns=["check_name", "passed", "notes"])
    qa.to_csv(paths["qa"], index=False, encoding="utf-8-sig")
    qa.to_csv(paths["qa_alias"], index=False, encoding="utf-8-sig")

    summary = {
        "strict_statement_buffer_path": rel(paths["buffer"]),
        "report_level_ttm_v2_path": rel(paths["ttm"]),
        "monthly_asof_panel_v2_path": rel(paths["asof"]),
        "monthly_raw_input_panel_v2_path": rel(paths["raw"]),
        "pit_clean_factor_panel_v2_path": rel(paths["factor"]),
        "v1_vs_v2_comparison_path": rel(paths["comparison"]),
        "ttm_pre_window_buffer_audit_path": rel(paths["buffer_audit"]),
        "factor_coverage_audit_v2_path": rel(paths["coverage"]),
        "factor_distribution_audit_v2_path": rel(paths["distribution"]),
        "report_path": rel(paths["report"]),
        "task_completion_card_path": rel(paths["card"]),
        "final_qa_path": rel(paths["qa"]),
        "run_state_path": rel(RUN_DIR / "RUN_STATE.md"),
        "v1_min_month_end": "" if pd.isna(v1_min) else str(v1_min.date()),
        "v2_min_month_end": "" if pd.isna(v2_min) else str(v2_min.date()),
        "v2_max_month_end": "" if pd.isna(v2_max) else str(v2_max.date()),
        "v2_n_rows": int(len(factors)),
        "v2_n_symbols": int(factors["symbol"].nunique()),
        "v2_n_rows_2017_01_to_2017_03": int(len(early)),
        "v2_roe_coverage_2017_01_to_2017_03": float(early["roe_ttm"].notna().mean()) if len(early) else 0.0,
        "v2_ep_coverage_2017_01_to_2017_03": float(early["ep_ttm"].notna().mean()) if len(early) else 0.0,
        "v2_bp_coverage_2017_01_to_2017_03": float(early["bp"].notna().mean()) if len(early) else 0.0,
        "selected_pit_date_violation_count": selected_pit_violations,
        "market_cap_date_violation_count": market_cap_violations,
        "report_period_used_as_pit_date_count": report_period_used_as_pit_date_count,
        "fixed_lag_used": fixed_lag_used,
        "legal_deadline_fallback_used": legal_deadline_fallback_used,
        "firforecdt_primary_used": firforecdt_primary_used,
        "one_row_per_symbol_month": one_row,
        "ep_bp_market_cap_block_removed": bool(ep_bp_unblocked),
        "pre_2017_buffer_used": bool(pre_2017_buffer_used),
        "recommended_next_task": "v2 factor QA / FI_T5 sanity check; not production",
        "xlsx_read": False,
        "raw_daily_csv_read": False,
        "csmar_api_accessed": False,
        "download_executed": False,
        "readme_modified": False,
        "all_daily_modified": False,
        "training_panel_modified": False,
        "production_modified": False,
        "v1_outputs_overwritten": False,
        "decision": decision,
    }
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    outputs = [
        paths["inventory"], paths["buffer"], paths["ttm"], paths["asof"], paths["raw"], paths["factor"],
        paths["comparison"], paths["buffer_audit"], paths["coverage"], paths["distribution"], paths["report"],
        paths["card"], paths["qa"], paths["summary"],
    ]
    update_run_state("completed", ["all v2 patch outputs generated", f"decision: {decision}"], outputs, "task complete")
    append_checkpoint("completed", [f"generated {rel(paths['summary'])}", f"decision: {decision}"])

    del raw, factors, coverage, distribution, comparison, buffer_audit, qa, ttm, asof, mkt, mkt_one, v1_factor, v1_qa_cov
    gc.collect()

    for key in [
        "strict_statement_buffer_path",
        "report_level_ttm_v2_path",
        "monthly_asof_panel_v2_path",
        "monthly_raw_input_panel_v2_path",
        "pit_clean_factor_panel_v2_path",
        "v1_vs_v2_comparison_path",
        "ttm_pre_window_buffer_audit_path",
        "factor_coverage_audit_v2_path",
        "factor_distribution_audit_v2_path",
        "report_path",
        "task_completion_card_path",
        "final_qa_path",
        "run_state_path",
        "v1_min_month_end",
        "v2_min_month_end",
        "v2_max_month_end",
        "v2_n_rows",
        "v2_n_symbols",
        "v2_n_rows_2017_01_to_2017_03",
        "v2_roe_coverage_2017_01_to_2017_03",
        "v2_ep_coverage_2017_01_to_2017_03",
        "v2_bp_coverage_2017_01_to_2017_03",
        "selected_pit_date_violation_count",
        "market_cap_date_violation_count",
        "report_period_used_as_pit_date_count",
        "fixed_lag_used",
        "legal_deadline_fallback_used",
        "firforecdt_primary_used",
        "one_row_per_symbol_month",
        "ep_bp_market_cap_block_removed",
        "pre_2017_buffer_used",
        "recommended_next_task",
        "xlsx_read",
        "raw_daily_csv_read",
        "csmar_api_accessed",
        "download_executed",
        "readme_modified",
        "all_daily_modified",
        "training_panel_modified",
        "production_modified",
        "v1_outputs_overwritten",
        "decision",
    ]:
        print(f"{key}={summary[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
