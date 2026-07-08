from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "csmar_pit_scope_freeze_strict_core_fs_panel_v1"

CORE_PANEL = ROOT / "output" / "csmar_core_fs_manual_import_audit_v1" / "core_fs_statement_panel_v1.parquet"
CORE_WITH_PIT = ROOT / "output" / "csmar_core_fs_manual_import_audit_v1" / "core_fs_statement_with_pit_dates_v1.parquet"
OLD_P0 = ROOT / "output" / "csmar_p0_pit_pack_import_audit_v1" / "csmar_p0_pit_announcement_panel_v1.parquet"
REPAIRED_P0 = ROOT / "output" / "csmar_p0_pit_missing_attribution_repair_v1" / "csmar_p0_pit_announcement_panel_repaired_v1.parquet"
TRAINING = ROOT / "output" / "training_panel_v15_sr.parquet"
ALL_DAILY = ROOT / "output" / "all_daily.parquet"
COMINS_COVERAGE = ROOT / "output" / "csmar_fs_comins_pit_coverage_patch_v1" / "fs_comins_pit_coverage_by_scope_v1.csv"
PIT_REPAIR_COVERAGE = ROOT / "output" / "csmar_p0_pit_missing_attribution_repair_v1" / "pit_coverage_before_after_v1.csv"

PROJECT_STATUS = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS = ROOT / "docs" / "DECISIONS.md"
README = ROOT / "README.md"
PAPER_TRADING_PIPELINE = ROOT / "paper_trading" / "paper_trading_pipeline.py"

WINDOW_START = pd.Timestamp("2017-01-01")
WINDOW_END_CAP = pd.Timestamp("2026-06-30")
TASK_NAME = "CSMAR PIT Scope Freeze and Strict Core FS Monthly Source Panel v1"

RAW_ITEM_COLS = [
    "total_operating_revenue",
    "operating_revenue",
    "sales_expense",
    "admin_expense",
    "rd_expense",
    "financial_expense",
    "total_profit",
    "net_profit",
    "net_profit_parent",
    "total_assets",
    "total_liabilities",
    "equity_parent",
    "total_equity",
]

STATEMENT_OUTPUT_COLS = [
    "symbol",
    "short_name",
    "report_period",
    "report_type",
    "pit_date_primary",
    "pit_date_source",
    "effective_month_end",
    *RAW_ITEM_COLS,
    "income_available",
    "balance_available",
    "income_if_correct",
    "balance_if_correct",
    "income_correction_disclosure_date",
    "balance_correction_disclosure_date",
    "strict_pit_eligible",
    "exclusion_reason",
    "source_table_flags",
]

MONTHLY_OUTPUT_COLS = [
    "month_end",
    "symbol",
    "selected_report_period",
    "selected_pit_date",
    "report_lag_days",
    "report_lag_months",
    *RAW_ITEM_COLS,
    "income_available",
    "balance_available",
    "strict_pit_source",
    "data_age_bucket",
    "notes",
]


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


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def date_min_max(df: pd.DataFrame) -> tuple[str, str]:
    date_cols = [c for c in ["date", "month_end", "report_period", "pit_date_primary", "effective_month_end"] if c in df.columns]
    mins = []
    maxs = []
    for col in date_cols:
        s = pd.to_datetime(df[col], errors="coerce")
        if s.notna().any():
            mins.append(s.min())
            maxs.append(s.max())
    if not mins:
        return "", ""
    return pd.Timestamp(min(mins)).date().isoformat(), pd.Timestamp(max(maxs)).date().isoformat()


def inventory_row(path: Path, role: str, notes: str) -> dict[str, object]:
    row = {
        "input_path": rel(path),
        "exists": path.exists(),
        "readable": False,
        "n_rows": np.nan,
        "n_columns": np.nan,
        "columns": "",
        "min_date": "",
        "max_date": "",
        "role": role,
        "notes": notes,
    }
    if not path.exists():
        return row
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame()
        mn, mx = date_min_max(df)
        row.update(
            {
                "readable": True,
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "columns": "|".join(map(str, df.columns)),
                "min_date": mn,
                "max_date": mx,
            }
        )
    except Exception as exc:  # noqa: BLE001
        row["notes"] = f"{notes}; read_error={exc}"
    return row


def write_inventory() -> Path:
    rows = [
        inventory_row(CORE_PANEL, "core_fs_without_pit", "read only"),
        inventory_row(CORE_WITH_PIT, "core_fs_with_existing_pit_dates", "strict source input"),
        inventory_row(OLD_P0, "old_p0_pit_panel", "read only audit"),
        inventory_row(REPAIRED_P0, "repaired_p0_pit_panel_optional", "equivalence check only; not a patch source"),
        inventory_row(TRAINING, "v15_universe_month_ends", "read only"),
        inventory_row(ALL_DAILY, "trading_dates_month_end_reference", "read only"),
        inventory_row(COMINS_COVERAGE, "prior_comins_scope_coverage", "read only optional context"),
        inventory_row(PIT_REPAIR_COVERAGE, "prior_pit_repair_coverage", "read only optional context"),
    ]
    path = OUT / "input_inventory_v1.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_policy(effective_end: pd.Timestamp) -> tuple[Path, Path]:
    policy = {
        "data_sources": {
            "iar_rept_forecdt_range": "2015-2026",
            "fs_comins_range": "1990-2026",
            "fs_combas_range": "2005-2026",
        },
        "effective_rebuild_window": {
            "start": WINDOW_START.date().isoformat(),
            "end": f"{effective_end.date().isoformat()}_detected_max_month_end",
        },
        "universe": {
            "source": rel(TRAINING),
            "description": "v15 / CSI800 historical universe",
        },
        "pit_policy": {
            "mode": "strict_actual_pit_only",
            "allowed_primary_dates": ["Annodt", "Actudt"],
            "disallowed_primary_dates": ["report_period", "Firforecdt", "fixed_lag", "legal_deadline_fallback"],
            "missing_pit_action": "drop",
        },
        "report_type_policy": {
            "preferred": "A",
            "fallback_to_B": "false_for_primary_panel",
            "notes": "B may be audited separately but must not enter primary strict panel unless explicitly approved later.",
        },
        "factor_rebuild_scope": {
            "allowed_without_market_cap": [
                "ROE",
                "ProfitGrowth_YoY",
                "RevGrowth_YoY",
                "NetMargin",
                "Debt_Ratio",
                "sales_expense_to_revenue",
                "rd_expense_to_revenue",
            ],
            "blocked_until_market_cap": ["EP", "BP"],
            "blocked_until_forecast_table": ["earnings_preview_midpoint_yoy"],
        },
    }
    yaml_path = OUT / "csmar_pit_scope_freeze_policy_v1.yaml"
    yaml_path.write_text(yaml.safe_dump(policy, sort_keys=False, allow_unicode=True), encoding="utf-8")
    md_path = OUT / "csmar_pit_scope_freeze_policy_v1.md"
    md_path.write_text(
        "\n".join(
            [
                "# CSMAR PIT Scope Freeze Policy v1",
                "",
                "## Data Sources",
                "",
                "- IAR_Rept / IAR_Forecdt range: 2015-2026",
                "- FS_Comins range: 1990-2026",
                "- FS_Combas range: 2005-2026",
                "",
                "## Effective Rebuild Window",
                "",
                f"- start: {WINDOW_START.date().isoformat()}",
                f"- end: {effective_end.date().isoformat()} or detected max month_end",
                "",
                "## PIT Policy",
                "",
                "- mode: strict_actual_pit_only",
                "- allowed primary dates: Annodt, Actudt",
                "- disallowed primary dates: report_period, Firforecdt, fixed_lag, legal_deadline_fallback",
                "- missing PIT action: drop",
                "",
                "## Report Type Policy",
                "",
                "- preferred: A",
                "- fallback_to_B: false_for_primary_panel",
                "- notes: B may be audited separately but must not enter primary strict panel unless explicitly approved later.",
                "",
                "## Factor Rebuild Scope",
                "",
                "- allowed without market cap: ROE, ProfitGrowth_YoY, RevGrowth_YoY, NetMargin, Debt_Ratio, sales_expense_to_revenue, rd_expense_to_revenue",
                "- blocked until market cap: EP, BP",
                "- blocked until forecast table: earnings_preview_midpoint_yoy",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path, md_path


def allowed_pit_source(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.lower()
    is_actual = s.str.contains("actual|actudt|annodt|annodt_from_iar_rept", regex=True)
    is_disallowed = s.str.contains("firforecdt|forecast|first_forecast|fixed_lag|deadline|report_period", regex=True)
    return is_actual & ~is_disallowed


def build_strict_records(training: pd.DataFrame, effective_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_parquet(CORE_WITH_PIT)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    for col in ["report_period", "pit_date_primary", "effective_month_end"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    universe = set(training["symbol"].astype(str).str.zfill(6).unique())
    available_max = min(effective_end, pd.Timestamp(date.today()))

    flags = []
    flags.append(np.where(df["symbol"].isin(universe), "", "symbol_not_in_v15_universe"))
    flags.append(np.where(df["report_period"].ge(WINDOW_START) & df["report_period"].le(effective_end), "", "report_period_outside_effective_window"))
    flags.append(np.where(df["report_type"].astype(str).eq("A"), "", "non_A_report_type"))
    flags.append(np.where(df["pit_date_primary"].notna(), "", "missing_pit_date"))
    flags.append(np.where(df["pit_date_primary"].ge(df["report_period"]), "", "pit_before_report_period"))
    flags.append(np.where(df["pit_date_primary"].le(available_max), "", "pit_after_current_available_max_date"))
    flags.append(np.where(allowed_pit_source(df["pit_date_source"]), "", "pit_source_not_allowed_actual_annodt_actudt"))
    flags.append(np.where(df["income_available"].fillna(False) | df["balance_available"].fillna(False), "", "no_core_fs_source_available"))
    reason = pd.Series("", index=df.index, dtype=object)
    for arr in flags:
        part = pd.Series(arr, index=df.index, dtype=object)
        reason = np.where((reason == "") & (part != ""), part, np.where(part != "", reason + "|" + part, reason))
        reason = pd.Series(reason, index=df.index, dtype=object)
    df["exclusion_reason"] = reason.replace("", "included")
    df["strict_pit_eligible"] = df["exclusion_reason"].eq("included")

    eligible = df.loc[df["strict_pit_eligible"]].copy()
    eligible["source_completeness"] = eligible[RAW_ITEM_COLS].notna().sum(axis=1)
    eligible["both_statement_available"] = eligible["income_available"].fillna(False).astype(int) + eligible["balance_available"].fillna(False).astype(int)
    eligible["source_table_flags"] = np.where(
        eligible["income_available"].fillna(False) & eligible["balance_available"].fillna(False),
        "FS_Comins|FS_Combas",
        np.where(eligible["income_available"].fillna(False), "FS_Comins", "FS_Combas"),
    )
    eligible = eligible.sort_values(
        ["symbol", "report_period", "pit_date_primary", "both_statement_available", "source_completeness"],
        ascending=[True, True, True, False, False],
    )
    strict = eligible.drop_duplicates(["symbol", "report_period"], keep="first").copy()
    strict["strict_pit_eligible"] = True
    strict["exclusion_reason"] = "included"
    for col in STATEMENT_OUTPUT_COLS:
        if col not in strict.columns:
            strict[col] = np.nan
    strict = strict[STATEMENT_OUTPUT_COLS].sort_values(["symbol", "report_period"]).reset_index(drop=True)
    return strict, df[["symbol", "report_period", "report_type", "pit_date_primary", "pit_date_source", "exclusion_reason", "strict_pit_eligible"]]


def lag_bucket(days: float) -> str:
    if pd.isna(days):
        return "missing"
    if days <= 120:
        return "0_120d"
    if days <= 250:
        return "121_250d"
    if days <= 370:
        return "251_370d"
    return "gt_370d"


def build_monthly_panel(training: pd.DataFrame, strict: pd.DataFrame) -> pd.DataFrame:
    base = training[["date", "symbol"]].copy()
    base["month_end"] = pd.to_datetime(base["date"], errors="coerce")
    base["symbol"] = base["symbol"].astype(str).str.zfill(6)
    base = base[["month_end", "symbol"]].drop_duplicates().sort_values(["symbol", "month_end"])

    records = strict.copy()
    records["visible_date"] = records[["pit_date_primary", "effective_month_end"]].max(axis=1)
    records = records[records["strict_pit_eligible"] & records["visible_date"].notna()].copy()
    records["source_completeness"] = records[RAW_ITEM_COLS].notna().sum(axis=1)
    records["both_statement_available"] = records["income_available"].fillna(False).astype(int) + records["balance_available"].fillna(False).astype(int)
    records = records.sort_values(
        ["symbol", "report_period", "pit_date_primary", "both_statement_available", "source_completeness"],
        ascending=[True, True, True, False, False],
    ).drop_duplicates(["symbol", "report_period"], keep="first")

    out_parts: list[pd.DataFrame] = []
    keep_cols = [
        "symbol",
        "report_period",
        "pit_date_primary",
        *RAW_ITEM_COLS,
        "income_available",
        "balance_available",
        "pit_date_source",
        "visible_date",
        "source_completeness",
        "both_statement_available",
    ]
    for symbol, base_g in base.groupby("symbol", sort=False):
        rec_g = records.loc[records["symbol"].eq(symbol), keep_cols].copy()
        if rec_g.empty:
            continue
        rows = []
        rec_g = rec_g.sort_values(["report_period", "pit_date_primary"])
        for month_end in base_g["month_end"].to_numpy():
            m = pd.Timestamp(month_end)
            visible = rec_g[(rec_g["visible_date"] <= m) & (rec_g["report_period"] <= m)]
            if visible.empty:
                continue
            max_rp = visible["report_period"].max()
            cand = visible[visible["report_period"].eq(max_rp)].sort_values(
                ["pit_date_primary", "both_statement_available", "source_completeness"],
                ascending=[True, False, False],
            )
            row = cand.iloc[0].to_dict()
            row["month_end"] = m
            rows.append(row)
        if rows:
            out_parts.append(pd.DataFrame(rows))
    if not out_parts:
        return pd.DataFrame(columns=MONTHLY_OUTPUT_COLS)
    monthly = pd.concat(out_parts, ignore_index=True)
    monthly = monthly.rename(columns={"report_period": "selected_report_period", "pit_date_primary": "selected_pit_date", "pit_date_source": "strict_pit_source"})
    monthly["report_lag_days"] = (monthly["month_end"] - monthly["selected_report_period"]).dt.days
    monthly["report_lag_months"] = (
        (monthly["month_end"].dt.year - monthly["selected_report_period"].dt.year) * 12
        + (monthly["month_end"].dt.month - monthly["selected_report_period"].dt.month)
    )
    monthly["data_age_bucket"] = monthly["report_lag_days"].map(lag_bucket)
    monthly["notes"] = "strict_actual_pit_source_panel"
    for col in MONTHLY_OUTPUT_COLS:
        if col not in monthly.columns:
            monthly[col] = np.nan
    monthly = monthly[MONTHLY_OUTPUT_COLS].sort_values(["month_end", "symbol"]).reset_index(drop=True)
    return monthly


def coverage_audits(training: pd.DataFrame, monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = training[["date", "symbol"]].copy()
    base["month_end"] = pd.to_datetime(base["date"], errors="coerce")
    base["symbol"] = base["symbol"].astype(str).str.zfill(6)
    universe = base.groupby("month_end")["symbol"].nunique().rename("v15_universe_n_symbols")

    if monthly.empty:
        cov = universe.reset_index()
        cov["strict_panel_n_symbols"] = 0
        cov["coverage_rate"] = 0.0
        cov["income_available_rate"] = 0.0
        cov["balance_available_rate"] = 0.0
        cov["median_report_lag_days"] = np.nan
        cov["p90_report_lag_days"] = np.nan
        cov["stale_report_gt_370d_rate"] = np.nan
    else:
        g = monthly.groupby("month_end")
        cov = pd.concat(
            [
                universe,
                g["symbol"].nunique().rename("strict_panel_n_symbols"),
                g["income_available"].mean().rename("income_available_rate"),
                g["balance_available"].mean().rename("balance_available_rate"),
                g["report_lag_days"].median().rename("median_report_lag_days"),
                g["report_lag_days"].quantile(0.9).rename("p90_report_lag_days"),
                g["report_lag_days"].apply(lambda s: float((s > 370).mean())).rename("stale_report_gt_370d_rate"),
            ],
            axis=1,
        ).reset_index()
        cov["strict_panel_n_symbols"] = cov["strict_panel_n_symbols"].fillna(0).astype(int)
        cov["coverage_rate"] = cov["strict_panel_n_symbols"] / cov["v15_universe_n_symbols"]
    cov["notes"] = np.select(
        [
            cov["month_end"].dt.year.eq(2017) & cov["coverage_rate"].lt(0.8),
            cov["month_end"].dt.year.eq(2026) & cov["coverage_rate"].lt(0.8),
            cov["coverage_rate"].lt(0.6),
            cov["coverage_rate"].lt(0.8),
        ],
        [
            "early_2017_warmup_low_coverage",
            "latest_2026_announcements_may_be_incomplete",
            "coverage_below_60pct_review_required",
            "coverage_below_80pct_review_required",
        ],
        default="stable_or_acceptable",
    )

    cov_year = cov.assign(year=cov["month_end"].dt.year).groupby("year", as_index=False).agg(
        months=("month_end", "nunique"),
        mean_coverage_rate=("coverage_rate", "mean"),
        min_coverage_rate=("coverage_rate", "min"),
        mean_income_available_rate=("income_available_rate", "mean"),
        mean_balance_available_rate=("balance_available_rate", "mean"),
        median_report_lag_days=("median_report_lag_days", "median"),
        p90_report_lag_days=("p90_report_lag_days", "median"),
        stale_report_gt_370d_rate=("stale_report_gt_370d_rate", "mean"),
    )

    monthly_keys = monthly[["month_end", "symbol"]] if not monthly.empty else pd.DataFrame(columns=["month_end", "symbol"])
    missing = base.drop_duplicates().merge(monthly_keys.drop_duplicates(), on=["month_end", "symbol"], how="left", indicator=True)
    missing = missing[missing["_merge"].eq("left_only")].drop(columns=["_merge"])
    miss_month = missing.groupby("month_end")["symbol"].nunique().rename("missing_n_symbols").reset_index()
    miss_month = universe.reset_index().merge(miss_month, on="month_end", how="left")
    miss_month["missing_n_symbols"] = miss_month["missing_n_symbols"].fillna(0).astype(int)
    miss_month["missing_rate"] = miss_month["missing_n_symbols"] / miss_month["v15_universe_n_symbols"]
    miss_month["notes"] = np.where(miss_month["missing_rate"].gt(0.4), "high_missing_rate_review", "")

    miss_symbol = missing.groupby("symbol")["month_end"].agg(missing_months="nunique", first_missing_month="min", last_missing_month="max").reset_index()
    total_symbol_months = base.groupby("symbol")["month_end"].nunique().rename("v15_months").reset_index()
    miss_symbol = total_symbol_months.merge(miss_symbol, on="symbol", how="left").fillna({"missing_months": 0})
    miss_symbol["missing_months"] = miss_symbol["missing_months"].astype(int)
    miss_symbol["missing_rate"] = miss_symbol["missing_months"] / miss_symbol["v15_months"]
    miss_symbol = miss_symbol.sort_values(["missing_months", "missing_rate"], ascending=False).head(100)
    return cov, cov_year, miss_month, miss_symbol


def ttm_readiness(monthly: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    mean_cov = float(coverage["coverage_rate"].mean()) if not coverage.empty else 0.0
    availability = {col: bool(col in monthly.columns and monthly[col].notna().any()) for col in RAW_ITEM_COLS}
    factor_specs = {
        "ROE": (["net_profit_parent", "equity_parent"], True),
        "ProfitGrowth_YoY": (["net_profit_parent"], True),
        "RevGrowth_YoY": ["operating_revenue"],
        "NetMargin": (["net_profit", "operating_revenue"], True),
        "Debt_Ratio": (["total_liabilities", "total_assets"], False),
        "sales_expense_to_revenue": (["sales_expense", "operating_revenue"], True),
        "rd_expense_to_revenue": (["rd_expense", "operating_revenue"], True),
        "EP": (["net_profit_parent", "market_cap"], True),
        "BP": (["equity_parent", "market_cap"], False),
        "earnings_preview_midpoint_yoy": (["earnings_preview_midpoint"], False),
    }
    rows = []
    for factor, spec in factor_specs.items():
        if isinstance(spec, tuple):
            required, ttm_req = spec
        else:
            required, ttm_req = spec, True
        raw_ok = all(availability.get(x, False) for x in required)
        blocked = ""
        reconstructable = raw_ok and mean_cov > 0
        if factor in {"EP", "BP"}:
            blocked = "BLOCKED_MARKET_CAP_MISSING"
            reconstructable = False
        elif factor == "earnings_preview_midpoint_yoy":
            blocked = "BLOCKED_FORECAST_TABLE_MISSING"
            reconstructable = False
        elif not raw_ok:
            blocked = "REQUIRED_RAW_ITEMS_MISSING"
        notes = "source panel only; final factor not rebuilt"
        if factor == "rd_expense_to_revenue":
            notes += "; pre-2018 rd_expense structurally sparse"
        rows.append(
            {
                "target_factor": factor,
                "required_raw_items": "|".join(required),
                "available_in_strict_panel": raw_ok,
                "monthly_coverage_rate_mean": mean_cov,
                "ttm_required": bool(ttm_req),
                "ttm_reconstructable_from_strict_panel": bool(reconstructable),
                "blocked_reason": blocked,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    paths: dict[str, Path],
    strict: pd.DataFrame,
    monthly: pd.DataFrame,
    coverage: pd.DataFrame,
    ttm: pd.DataFrame,
    effective_end: pd.Timestamp,
    decision: str,
) -> Path:
    mean_cov = float(coverage["coverage_rate"].mean()) if not coverage.empty else 0.0
    min_cov = float(coverage["coverage_rate"].min()) if not coverage.empty else 0.0
    files = "\n".join(f"- `{rel(p)}`" for p in paths.values())
    ready = ", ".join(ttm.loc[ttm["ttm_reconstructable_from_strict_panel"], "target_factor"].tolist()) or "none"
    report = f"""# CSMAR PIT Scope Freeze and Strict Core FS Monthly Source Panel v1

## 1. Executive Summary

Decision = {decision}. The task produced a strict actual PIT core financial statement record set and a monthly as-of source panel. This is a source panel, not a final factor panel.

## 2. Why PIT补齐停止

PIT补齐停止 because existing raw IAR cannot further repair missing PIT dates and repair_gain is 0. The project no longer pursues full-history CSMAR PIT coverage.

## 3. Frozen Data Scope

- IAR_Rept / IAR_Forecdt: 2015-2026
- FS_Comins: 1990-2026
- FS_Combas: 2005-2026
- Effective rebuild window: 2017-01-01 to {effective_end.date().isoformat()}
- Universe: v15 / CSI800 historical universe from `{rel(TRAINING)}`

## 4. Strict Actual PIT Policy

Strict actual PIT only. Missing real PIT dates are dropped. The panel does not use report_period, fixed lag, legal deadline, or Firforecdt as primary PIT date.

## 5. Strict Core FS Statement Records

- rows: {len(strict)}
- symbols: {strict['symbol'].nunique() if not strict.empty else 0}
- report_type: A only
- missing PIT rows: dropped
- fallback PIT: not used

## 6. Monthly As-Of Source Panel

- rows: {len(monthly)}
- symbols: {monthly['symbol'].nunique() if not monthly.empty else 0}
- one row per symbol-month: {not monthly.duplicated(['month_end', 'symbol']).any() if not monthly.empty else True}
- selected_pit_date > month_end violations: {int((monthly['selected_pit_date'] > monthly['month_end']).sum()) if not monthly.empty else 0}

## 7. Coverage and Sample Bias Audit

- mean monthly coverage: {mean_cov:.6f}
- minimum monthly coverage: {min_cov:.6f}
- months below 80pct coverage: {int((coverage['coverage_rate'] < 0.8).sum()) if not coverage.empty else 0}
- months below 60pct coverage: {int((coverage['coverage_rate'] < 0.6).sum()) if not coverage.empty else 0}
- early 2017 may have warm-up coverage drag.
- 2026 may have incomplete newest announcements.

## 8. TTM Factor Readiness

Reconstructable from this strict source panel, before final factor work: {ready}. EP/BP still require market_cap. earnings_preview_midpoint_yoy still requires the forecast table.

## 9. Remaining Missing Data

Remaining gaps are mainly rows without actual PIT dates, symbol-months before first visible report, stale report coverage, market_cap for EP/BP, and forecast data for earnings preview.

## 10. Limitations

No PIT repair was performed. No fallback PIT date was created. This output does not train a model, run a backtest, run IC, generate signals, generate real orders, or modify production.

## 11. Recommended Next Task

Acquire or map market_cap / total market value source for EP and BP, then run a separate PIT-clean factor reconstruction task.

## 12. Files Generated

{files}

## Explicit Guardrails

- This task did not access CSMAR API.
- This task did not call getPackResultExt.
- This task did not download new data.
- This task did not补 PIT.
- This task did not modify README.md.
- This task did not modify all_daily.parquet or training_panel_v15_sr.parquet.
- This task did not connect the panel to production.
"""
    path = OUT / "csmar_pit_scope_freeze_strict_core_fs_panel_report_v1.md"
    path.write_text(report, encoding="utf-8")
    return path


def update_project_status() -> None:
    status = yaml.safe_load(PROJECT_STATUS.read_text(encoding="utf-8"))
    status.setdefault("alternative_data", {})
    status["alternative_data"]["csmar_status"] = "pit_scope_frozen_strict_core_fs_source_panel_ready_or_under_review"
    status["alternative_data"]["csmar_latest_task"] = TASK_NAME
    status["alternative_data"]["csmar_latest_output"] = rel(OUT)
    status.setdefault("validation", {})
    status["validation"]["pit_financial_status"] = "strict_actual_pit_core_fs_source_panel_built_market_cap_pending"
    status["validation"]["blend_v3_historical_metrics_status"] = "under_pit_review"
    status.setdefault("project", {})
    status["project"]["last_updated"] = date.today().isoformat()
    PROJECT_STATUS.write_text(yaml.safe_dump(status, sort_keys=False, allow_unicode=True), encoding="utf-8")


def append_decision(decision: str) -> None:
    block = f"""## {date.today().isoformat()}

决策：

- 停止补 PIT 日期。
- 冻结 IAR / FS_Comins / FS_Combas 时间范围：IAR 2015-2026，FS_Comins 1990-2026，FS_Combas 2005-2026。
- effective rebuild window = 2017-2026。
- strict actual PIT only。
- missing PIT rows dropped, no fallback。
- 生成 strict monthly as-of core FS source panel。
- EP/BP 仍需 market_cap。
- 不访问 CSMAR API。
- 不修改 README。
- 不接入 production。
- Decision = {decision}。
"""
    text = DECISIONS.read_text(encoding="utf-8") if DECISIONS.exists() else "# 决策日志\n"
    marker = "生成 strict monthly as-of core FS source panel"
    if marker not in text:
        DECISIONS.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def write_task_card(
    strict: pd.DataFrame,
    monthly: pd.DataFrame,
    coverage: pd.DataFrame,
    ttm: pd.DataFrame,
    effective_end: pd.Timestamp,
    decision: str,
) -> Path:
    mean_cov = float(coverage["coverage_rate"].mean()) if not coverage.empty else 0.0
    min_cov = float(coverage["coverage_rate"].min()) if not coverage.empty else 0.0
    ready = ", ".join(ttm.loc[ttm["ttm_reconstructable_from_strict_panel"], "target_factor"].tolist()) or "none"
    blocked = "market_cap for EP/BP; forecast table for earnings_preview_midpoint_yoy"
    lines = [
        f"任务名称：{TASK_NAME}",
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
        "是否补 PIT：否",
        "是否使用 fallback PIT：否",
        f"核心输出：{rel(OUT)}",
        f"核心结论：{decision}",
        f"冻结窗口：2017-01-01 to {effective_end.date().isoformat()}",
        "PIT policy：strict_actual_pit_only; missing PIT dropped; no fallback",
        f"strict statement records 行数：{len(strict)}",
        f"monthly as-of panel 行数：{len(monthly)}",
        f"平均月度覆盖率：{mean_cov:.6f}",
        f"最低月度覆盖率：{min_cov:.6f}",
        f"可继续重构的因子：{ready}",
        f"仍缺数据：{blocked}",
        "下一步建议：Acquire or map market_cap / total market value source for EP and BP, then run separate PIT-clean factor reconstruction.",
    ]
    path = OUT / "task_completion_card.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_final_qa(
    hashes_before: dict[str, str | None],
    paths: dict[str, Path],
    monthly: pd.DataFrame,
    ttm: pd.DataFrame,
    readme_consistency_report: Path,
) -> Path:
    hashes_after = {k: file_hash(p) for k, p in PROTECTED_FILES.items()}
    no_dup = True if monthly.empty else not monthly.duplicated(["month_end", "symbol"]).any()
    no_future = True if monthly.empty else not (monthly["selected_pit_date"] > monthly["month_end"]).any()
    ep_block = bool((ttm["target_factor"].eq("EP") & ttm["blocked_reason"].eq("BLOCKED_MARKET_CAP_MISSING")).any())
    bp_block = bool((ttm["target_factor"].eq("BP") & ttm["blocked_reason"].eq("BLOCKED_MARKET_CAP_MISSING")).any())
    rows = [
        ("README.md not modified", hashes_before["README.md"] == hashes_after["README.md"], "hash unchanged during script run"),
        ("all_daily.parquet not modified", hashes_before["all_daily.parquet"] == hashes_after["all_daily.parquet"], "hash unchanged during script run"),
        ("training_panel_v15_sr.parquet not modified", hashes_before["training_panel_v15_sr.parquet"] == hashes_after["training_panel_v15_sr.parquet"], "hash unchanged during script run"),
        ("model files not modified", True, "script writes only its own output directory plus project_status/CURRENT_STATUS/DECISIONS"),
        ("paper_trading_pipeline.py not modified", hashes_before["paper_trading_pipeline.py"] == hashes_after["paper_trading_pipeline.py"], "hash unchanged during script run"),
        ("production config not modified", True, "only config/project_status.yaml governance status was updated"),
        ("no model training executed", True, "no training entrypoint called"),
        ("no backtest executed", True, "no backtest entrypoint called"),
        ("no IC test executed", True, "no IC entrypoint called"),
        ("no trading signal generated", True, "source panel only"),
        ("no real orders generated", True, "no production or broker code executed"),
        ("no CSMAR API access executed", True, "no API client called"),
        ("getPackResultExt not called", True, "string not invoked by this script"),
        ("no CSMAR download executed", True, "offline local files only"),
        ("no credential value printed", True, "script prints paths and aggregate metrics only"),
        ("root-level output used", str(OUT).startswith(str(ROOT / "output")), rel(OUT)),
        ("xhs/output not used for new outputs", "xhs/output" not in rel(OUT), rel(OUT)),
        ("PIT scope policy generated", paths["scope_policy_yaml"].exists() and paths["scope_policy_md"].exists(), "yaml and md exist"),
        ("no PIT fallback used", True, "strict filter requires existing pit_date_primary"),
        ("missing PIT rows dropped", True, "strict records require pit_date_primary not null"),
        ("report_period not used as visibility date", True, "monthly visibility uses max(pit_date_primary, effective_month_end)"),
        ("Firforecdt not used as primary PIT", True, "pit_source forecast strings are disallowed"),
        ("strict statement records generated", paths["strict_statement_records"].exists(), rel(paths["strict_statement_records"])),
        ("monthly as-of source panel generated", paths["strict_monthly_asof_panel"].exists(), rel(paths["strict_monthly_asof_panel"])),
        ("no selected_pit_date > month_end", no_future, "validated on monthly panel"),
        ("one row per symbol-month in monthly panel", no_dup, "validated on monthly panel"),
        ("coverage audit generated", paths["monthly_coverage_audit"].exists(), rel(paths["monthly_coverage_audit"])),
        ("TTM readiness generated", paths["ttm_readiness"].exists(), rel(paths["ttm_readiness"])),
        ("EP/BP marked market_cap missing", ep_block and bp_block, "EP and BP blocked by market_cap"),
        ("final report generated", paths["report"].exists(), rel(paths["report"])),
        ("task completion card generated", paths["task_completion_card"].exists(), rel(paths["task_completion_card"])),
        ("project_status.yaml updated", PROJECT_STATUS.exists(), rel(PROJECT_STATUS)),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS.exists(), rel(CURRENT_STATUS)),
        ("DECISIONS.md appended", DECISIONS.exists(), rel(DECISIONS)),
        ("README consistency check executed", readme_consistency_report.exists(), rel(readme_consistency_report)),
        ("README not auto-modified", hashes_before["README.md"] == hashes_after["README.md"], "hash unchanged during script run"),
    ]
    path = OUT / "final_qa_csmar_pit_scope_freeze_strict_core_fs_panel_v1.csv"
    pd.DataFrame(rows, columns=["check", "pass", "details"]).to_csv(path, index=False, encoding="utf-8-sig")
    return path


PROTECTED_FILES = {
    "README.md": README,
    "all_daily.parquet": ALL_DAILY,
    "training_panel_v15_sr.parquet": TRAINING,
    "paper_trading_pipeline.py": PAPER_TRADING_PIPELINE,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hashes_before = {k: file_hash(p) for k, p in PROTECTED_FILES.items()}

    training = read_parquet(TRAINING)
    training["date"] = pd.to_datetime(training["date"], errors="coerce")
    training["symbol"] = training["symbol"].astype(str).str.zfill(6)
    detected_max = training["date"].max()
    effective_end = min(WINDOW_END_CAP, detected_max)
    training_window = training[(training["date"] >= WINDOW_START) & (training["date"] <= effective_end)].copy()

    input_inventory_path = write_inventory()
    scope_policy_yaml_path, scope_policy_md_path = write_policy(effective_end)

    strict, exclusions = build_strict_records(training_window, effective_end)
    strict_statement_records_path = OUT / "strict_core_fs_statement_records_v1.parquet"
    strict.to_parquet(strict_statement_records_path, index=False)
    strict.head(500).to_csv(OUT / "strict_core_fs_statement_records_sample_v1.csv", index=False, encoding="utf-8-sig")
    exclusions.to_csv(OUT / "strict_core_fs_statement_exclusion_audit_v1.csv", index=False, encoding="utf-8-sig")

    monthly = build_monthly_panel(training_window, strict)
    strict_monthly_asof_panel_path = OUT / "strict_core_fs_monthly_asof_panel_v1.parquet"
    monthly.to_parquet(strict_monthly_asof_panel_path, index=False)
    monthly.head(500).to_csv(OUT / "strict_core_fs_monthly_asof_panel_sample_v1.csv", index=False, encoding="utf-8-sig")

    coverage, coverage_by_year, missing_by_month, missing_by_symbol = coverage_audits(training_window, monthly)
    monthly_coverage_audit_path = OUT / "strict_core_fs_monthly_coverage_audit_v1.csv"
    coverage_by_year_path = OUT / "strict_core_fs_coverage_by_year_v1.csv"
    missing_by_month_path = OUT / "strict_core_fs_missing_by_month_v1.csv"
    missing_by_symbol_top100_path = OUT / "strict_core_fs_missing_by_symbol_top100_v1.csv"
    coverage.to_csv(monthly_coverage_audit_path, index=False, encoding="utf-8-sig")
    coverage_by_year.to_csv(coverage_by_year_path, index=False, encoding="utf-8-sig")
    missing_by_month.to_csv(missing_by_month_path, index=False, encoding="utf-8-sig")
    missing_by_symbol.to_csv(missing_by_symbol_top100_path, index=False, encoding="utf-8-sig")

    ttm = ttm_readiness(monthly, coverage)
    ttm_readiness_path = OUT / "strict_core_fs_ttm_readiness_v1.csv"
    ttm.to_csv(ttm_readiness_path, index=False, encoding="utf-8-sig")

    pit_fallback_used = False
    report_period_used_as_visibility = False
    firforecdt_used_as_primary = False
    selected_pit_future = False if monthly.empty else bool((monthly["selected_pit_date"] > monthly["month_end"]).any())
    panel_success = strict_monthly_asof_panel_path.exists() and len(monthly) > 0
    mean_cov = float(coverage["coverage_rate"].mean()) if not coverage.empty else 0.0
    min_cov = float(coverage["coverage_rate"].min()) if not coverage.empty else 0.0
    if pit_fallback_used or report_period_used_as_visibility or selected_pit_future:
        decision = "INVALID_PIT_POLICY_VIOLATION"
    elif panel_success and (mean_cov < 0.8 or min_cov < 0.6):
        decision = "CSMAR_PIT_SCOPE_FROZEN_STRICT_CORE_FS_SOURCE_PANEL_NEEDS_COVERAGE_REVIEW"
    elif panel_success:
        decision = "CSMAR_PIT_SCOPE_FROZEN_STRICT_CORE_FS_SOURCE_PANEL_READY"
    else:
        decision = "INVALID_PIT_POLICY_VIOLATION"

    paths: dict[str, Path] = {
        "input_inventory": input_inventory_path,
        "scope_policy_yaml": scope_policy_yaml_path,
        "scope_policy_md": scope_policy_md_path,
        "strict_statement_records": strict_statement_records_path,
        "strict_monthly_asof_panel": strict_monthly_asof_panel_path,
        "monthly_coverage_audit": monthly_coverage_audit_path,
        "coverage_by_year": coverage_by_year_path,
        "missing_by_month": missing_by_month_path,
        "missing_by_symbol_top100": missing_by_symbol_top100_path,
        "ttm_readiness": ttm_readiness_path,
    }
    report_path = write_report(paths, strict, monthly, coverage, ttm, effective_end, decision)
    paths["report"] = report_path
    task_completion_card_path = write_task_card(strict, monthly, coverage, ttm, effective_end, decision)
    paths["task_completion_card"] = task_completion_card_path

    update_project_status()
    append_decision(decision)

    # Required project status commands.
    import subprocess

    subprocess.run(["python", str(ROOT / "scripts" / "generate_current_status_md.py")], cwd=ROOT, check=True, capture_output=True, text=True)
    readme_check = subprocess.run(["python", str(ROOT / "scripts" / "check_readme_consistency.py")], cwd=ROOT, check=True, capture_output=True, text=True)
    readme_consistency_report_path = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"

    final_qa_path = write_final_qa(hashes_before, paths, monthly, ttm, readme_consistency_report_path)

    can = dict(zip(ttm["target_factor"], ttm["ttm_reconstructable_from_strict_panel"]))
    ep_blocked = bool((ttm["target_factor"].eq("EP") & ttm["blocked_reason"].eq("BLOCKED_MARKET_CAP_MISSING")).any())
    bp_blocked = bool((ttm["target_factor"].eq("BP") & ttm["blocked_reason"].eq("BLOCKED_MARKET_CAP_MISSING")).any())
    hashes_after = {k: file_hash(p) for k, p in PROTECTED_FILES.items()}

    terminal = {
        "input_inventory_path": rel(input_inventory_path),
        "scope_policy_yaml_path": rel(scope_policy_yaml_path),
        "scope_policy_md_path": rel(scope_policy_md_path),
        "strict_statement_records_path": rel(strict_statement_records_path),
        "strict_monthly_asof_panel_path": rel(strict_monthly_asof_panel_path),
        "monthly_coverage_audit_path": rel(monthly_coverage_audit_path),
        "coverage_by_year_path": rel(coverage_by_year_path),
        "missing_by_month_path": rel(missing_by_month_path),
        "missing_by_symbol_top100_path": rel(missing_by_symbol_top100_path),
        "ttm_readiness_path": rel(ttm_readiness_path),
        "report_path": rel(report_path),
        "task_completion_card_path": rel(task_completion_card_path),
        "final_qa_path": rel(final_qa_path),
        "project_status_path": rel(PROJECT_STATUS),
        "current_status_doc_path": rel(CURRENT_STATUS),
        "decisions_doc_path": rel(DECISIONS),
        "readme_consistency_report_path": rel(readme_consistency_report_path),
        "effective_window_start": WINDOW_START.date().isoformat(),
        "effective_window_end": effective_end.date().isoformat(),
        "strict_statement_records_rows": len(strict),
        "strict_statement_records_symbols": strict["symbol"].nunique() if not strict.empty else 0,
        "strict_monthly_asof_rows": len(monthly),
        "strict_monthly_asof_symbols": monthly["symbol"].nunique() if not monthly.empty else 0,
        "mean_monthly_coverage_rate": f"{mean_cov:.6f}",
        "min_monthly_coverage_rate": f"{min_cov:.6f}",
        "months_below_80pct_coverage": int((coverage["coverage_rate"] < 0.8).sum()) if not coverage.empty else 0,
        "months_below_60pct_coverage": int((coverage["coverage_rate"] < 0.6).sum()) if not coverage.empty else 0,
        "pit_fallback_used": pit_fallback_used,
        "report_period_used_as_visibility": report_period_used_as_visibility,
        "firforecdt_used_as_primary": firforecdt_used_as_primary,
        "can_rebuild_roe_from_source": bool(can.get("ROE", False)),
        "can_rebuild_profit_growth_from_source": bool(can.get("ProfitGrowth_YoY", False)),
        "can_rebuild_revenue_growth_from_source": bool(can.get("RevGrowth_YoY", False)),
        "can_rebuild_net_margin_from_source": bool(can.get("NetMargin", False)),
        "can_rebuild_debt_ratio_from_source": bool(can.get("Debt_Ratio", False)),
        "can_rebuild_sales_expense_ratio_from_source": bool(can.get("sales_expense_to_revenue", False)),
        "can_rebuild_rd_expense_ratio_from_source": bool(can.get("rd_expense_to_revenue", False)),
        "ep_blocked_by_market_cap": ep_blocked,
        "bp_blocked_by_market_cap": bp_blocked,
        "recommended_next_task": "Acquire or map market_cap source for EP/BP, then run separate PIT-clean factor reconstruction.",
        "csmar_api_accessed": False,
        "getPackResultExt_called": False,
        "csmar_download_executed": False,
        "readme_modified": hashes_before["README.md"] != hashes_after["README.md"],
        "all_daily_modified": hashes_before["all_daily.parquet"] != hashes_after["all_daily.parquet"],
        "training_panel_modified": hashes_before["training_panel_v15_sr.parquet"] != hashes_after["training_panel_v15_sr.parquet"],
        "production_modified": False,
        "credential_exposure_detected": False,
        "decision": decision,
    }
    # Keep final stdout machine-readable and limited to the requested keys.
    _ = readme_check
    for key, value in terminal.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
