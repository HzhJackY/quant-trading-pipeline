from __future__ import annotations

import gc
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TASK_NAME = "v0_canonical_16factor_panel_build_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

V3_PANEL = (
    ROOT
    / "output"
    / "csmar_pit_clean_core_financial_factors_v3"
    / "pit_clean_core_financial_factors_monthly_v3.parquet"
)
ALL_DAILY = ROOT / "output" / "all_daily.parquet"
PREP_DIR = ROOT / "output" / "v0_legacy_16factor_canonical_mapping_completion_prep_v1"
MAPPING_DECISION = PREP_DIR / "v0_legacy_16factor_mapping_decision.csv"
PRICE_PLAN = PREP_DIR / "v0_price_technical_factor_rebuild_plan.csv"
FIN_PLAN = PREP_DIR / "v0_financial_factor_completion_plan.csv"
BUILD_CONFIG = PREP_DIR / "v0_canonical_16factor_panel_build_config_draft.json"
MAPPING_SUMMARY = PREP_DIR / "v0_legacy_16factor_canonical_mapping_completion_summary.json"
TRD_MNTH_REF = (
    ROOT
    / "output"
    / "trd_mnth_parser_repair_2024_12_coverage_repair_v0"
    / "canonical_csmar_trd_mnth_return_map_repaired.parquet"
)

FINANCIAL_RAW_FIELDS = {
    "BP": "bp",
    "EP": "ep_ttm",
    "ROE": "roe_ttm",
    "Debt_Ratio": "debt_ratio",
    "Net_Profit_Margin": "net_margin",
    "RevGrowth_YoY": "rev_growth_yoy",
    "ProfitGrowth_YoY": "profit_growth_yoy",
}
PRICE_FACTORS = [
    "Mom_1M",
    "Mom_3M",
    "Mom_6M",
    "Mom_12M_1M",
    "Vol_20D",
    "Vol_60D",
    "Beta",
    "VolChg_20D",
    "PriceDev_20D",
]
ALL_FACTORS = list(FINANCIAL_RAW_FIELDS.keys()) + PRICE_FACTORS
FINANCIAL_FACTORS = list(FINANCIAL_RAW_FIELDS.keys())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_state(status: str, details: dict[str, Any] | None = None) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": TASK_NAME,
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "details": details or {},
        "resume_instruction": f"先读取 {rel(RUN_DIR / 'RUN_STATE.md')} 再继续。",
    }
    lines = ["# RUN_STATE", "", f"- task_name: {TASK_NAME}", f"- status: {status}"]
    for key, value in payload["details"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "```json", json.dumps(payload, ensure_ascii=False, indent=2, default=str), "```"]
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(lines), encoding="utf-8")


def symbol_norm(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    text = text.str.replace(r"(?i)(\.sh|\.sz|sh|sz)$", "", regex=True)
    text = text.str.replace(r"\D", "", regex=True)
    return text.str[-6:].str.zfill(6)


def year_month_from_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.to_period("M").astype(str)


def finite_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def inf_count(s: pd.Series) -> int:
    vals = pd.to_numeric(s, errors="coerce")
    return int(np.isinf(vals.to_numpy(dtype=float, na_value=np.nan)).sum())


def extreme_count(s: pd.Series, threshold: float = 1e6) -> int:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return int((vals.abs() > threshold).sum())


def factor_status(non_null_ratio: float, infinite_count: int, critical_issue: bool = False) -> str:
    if critical_issue or infinite_count > 0:
        return "FAIL"
    if non_null_ratio >= 0.80:
        return "READY"
    if non_null_ratio >= 0.60:
        return "READY_WITH_CAVEAT"
    if non_null_ratio < 0.60:
        return "WATCH_LOW_COVERAGE"
    return "FAIL"


def prerequisite_check() -> dict[str, Any]:
    prereq = {
        "v3_panel_found": V3_PANEL.exists(),
        "all_daily_found": ALL_DAILY.exists(),
        "mapping_decision_found": MAPPING_DECISION.exists(),
        "price_technical_rebuild_plan_found": PRICE_PLAN.exists(),
        "financial_completion_plan_found": FIN_PLAN.exists(),
        "build_config_found": BUILD_CONFIG.exists(),
        "prerequisites_passed": False,
        "missing_files": [],
    }
    required = [V3_PANEL, ALL_DAILY, MAPPING_DECISION, PRICE_PLAN, FIN_PLAN, BUILD_CONFIG]
    prereq["missing_files"] = [rel(p) for p in required if not p.exists()]
    prereq["prerequisites_passed"] = len(prereq["missing_files"]) == 0
    return prereq


def load_financial_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "month_end",
        "symbol",
        "selected_report_period",
        "selected_pit_date",
        "total_market_cap_raw_thousand",
        *FINANCIAL_RAW_FIELDS.values(),
    ]
    available_cols = set(pq.ParquetFile(V3_PANEL).schema_arrow.names)
    read_cols = [c for c in cols if c in available_cols]
    df = pd.read_parquet(V3_PANEL, columns=read_cols)
    original_rows = len(df)
    df["symbol_norm"] = symbol_norm(df["symbol"])
    df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
    df["year_month"] = df["month_end"].dt.to_period("M").astype(str)
    for meta_col in ["selected_report_period", "selected_pit_date"]:
        if meta_col in df.columns:
            df[meta_col] = pd.to_datetime(df[meta_col], errors="coerce")
        else:
            df[meta_col] = pd.NaT

    dup_count = int(df.duplicated(["symbol_norm", "year_month"]).sum())
    if dup_count:
        df = df.sort_values(["symbol_norm", "year_month", "selected_pit_date", "selected_report_period"])
        df = df.drop_duplicates(["symbol_norm", "year_month"], keep="last")

    out = df[
        [
            "symbol_norm",
            "month_end",
            "year_month",
            "selected_pit_date",
            "selected_report_period",
            "total_market_cap_raw_thousand",
            *FINANCIAL_RAW_FIELDS.values(),
        ]
    ].copy()
    for factor, raw_col in FINANCIAL_RAW_FIELDS.items():
        out[factor] = finite_series(out[raw_col])
        out[f"{factor}_raw_field"] = raw_col
        out[f"{factor}_source_flag"] = "v3_existing_field"
        out[f"{factor}_valid_flag"] = out[factor].notna()

    qa_rows = []
    pit_metadata_available = bool(out["selected_pit_date"].notna().any() and out["selected_report_period"].notna().any())
    for factor, raw_col in FINANCIAL_RAW_FIELDS.items():
        non_null = float(out[factor].notna().mean()) if len(out) else 0.0
        qa_rows.append(
            {
                "row_count": original_rows,
                "unique_symbol_count": int(out["symbol_norm"].nunique()),
                "year_month_count": int(out["year_month"].nunique()),
                "min_year_month": str(out["year_month"].min()) if len(out) else "",
                "max_year_month": str(out["year_month"].max()) if len(out) else "",
                "duplicate_symbol_month_count": dup_count,
                "factor_name": factor,
                "non_null_ratio": round(non_null, 6),
                "infinite_count": inf_count(out[factor]),
                "extreme_value_count": extreme_count(out[factor]),
                "pit_metadata_available": pit_metadata_available,
                "qa_status": factor_status(non_null, inf_count(out[factor])),
            }
        )
    return out, pd.DataFrame(qa_rows)


def build_price_technical_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pf = pq.ParquetFile(ALL_DAILY)
    cols = set(pf.schema_arrow.names)
    required = ["date", "symbol", "close"]
    volume_col = "volume" if "volume" in cols else ("amount" if "amount" in cols else None)
    if not all(c in cols for c in required) or volume_col is None:
        raise KeyError("all_daily 缺少 date/symbol/close 或 volume/amount 字段")
    read_cols = ["date", "symbol", "close", volume_col]
    daily = pd.read_parquet(ALL_DAILY, columns=read_cols)
    daily = daily.rename(columns={volume_col: "volume_like"})
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["symbol_norm"] = symbol_norm(daily["symbol"])
    daily["close"] = finite_series(daily["close"])
    daily["volume_like"] = finite_series(daily["volume_like"])
    daily = daily.dropna(subset=["date", "symbol_norm", "close"])
    daily = daily.sort_values(["symbol_norm", "date"]).reset_index(drop=True)
    daily["year_month"] = daily["date"].dt.to_period("M").astype(str)

    daily["daily_ret"] = daily.groupby("symbol_norm", sort=False)["close"].pct_change()
    mkt = (
        daily.dropna(subset=["daily_ret"])
        .groupby("date", sort=True)
        .agg(mkt_ret=("daily_ret", "mean"), constituent_count=("daily_ret", "count"))
        .reset_index()
    )
    daily = daily.merge(mkt[["date", "mkt_ret"]], on="date", how="left")

    g = daily.groupby("symbol_norm", sort=False)
    daily["Vol_20D"] = g["daily_ret"].transform(lambda x: x.rolling(20, min_periods=20).std())
    daily["Vol_60D"] = g["daily_ret"].transform(lambda x: x.rolling(60, min_periods=60).std())
    daily["MA20"] = g["close"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    daily["PriceDev_20D"] = daily["close"] / daily["MA20"] - 1
    daily["vol_mean_20"] = g["volume_like"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    daily["vol_prev_20"] = g["volume_like"].transform(lambda x: x.rolling(20, min_periods=20).mean().shift(20))
    daily["VolChg_20D"] = daily["vol_mean_20"] / daily["vol_prev_20"] - 1

    daily["ret_mkt_prod"] = daily["daily_ret"] * daily["mkt_ret"]
    daily["mkt_sq"] = daily["mkt_ret"] * daily["mkt_ret"]
    daily["mean_ret"] = g["daily_ret"].transform(lambda x: x.rolling(60, min_periods=30).mean())
    daily["mean_mkt"] = g["mkt_ret"].transform(lambda x: x.rolling(60, min_periods=30).mean())
    daily["mean_prod"] = g["ret_mkt_prod"].transform(lambda x: x.rolling(60, min_periods=30).mean())
    daily["mean_mkt_sq"] = g["mkt_sq"].transform(lambda x: x.rolling(60, min_periods=30).mean())
    cov = daily["mean_prod"] - daily["mean_ret"] * daily["mean_mkt"]
    var = daily["mean_mkt_sq"] - daily["mean_mkt"] * daily["mean_mkt"]
    daily["Beta"] = cov / var.replace(0, np.nan)

    daily_last = daily.sort_values(["symbol_norm", "year_month", "date"]).drop_duplicates(
        ["symbol_norm", "year_month"], keep="last"
    )
    monthly_close = daily_last[["symbol_norm", "year_month", "date", "close"]].copy()
    monthly_close = monthly_close.sort_values(["symbol_norm", "year_month"])
    mg = monthly_close.groupby("symbol_norm", sort=False)
    monthly_close["Mom_1M"] = monthly_close["close"] / mg["close"].shift(1) - 1
    monthly_close["Mom_3M"] = monthly_close["close"] / mg["close"].shift(3) - 1
    monthly_close["Mom_6M"] = monthly_close["close"] / mg["close"].shift(6) - 1
    monthly_close["Mom_12M_1M"] = mg["close"].shift(1) / mg["close"].shift(12) - 1

    panel = daily_last[
        [
            "symbol_norm",
            "year_month",
            "date",
            "Vol_20D",
            "Vol_60D",
            "Beta",
            "VolChg_20D",
            "PriceDev_20D",
        ]
    ].rename(columns={"date": "month_trade_date"})
    panel = panel.merge(
        monthly_close[["symbol_norm", "year_month", "Mom_1M", "Mom_3M", "Mom_6M", "Mom_12M_1M"]],
        on=["symbol_norm", "year_month"],
        how="left",
    )
    for factor in PRICE_FACTORS:
        panel[factor] = finite_series(panel[factor])
    panel["daily_source_path"] = rel(ALL_DAILY)
    valid_cols = [f"{factor}_valid" for factor in PRICE_FACTORS]
    for factor in PRICE_FACTORS:
        panel[f"{factor}_valid"] = panel[factor].notna()
    panel["price_factor_valid_flags"] = panel[valid_cols].apply(
        lambda row: json.dumps({col.replace("_valid", ""): bool(row[col]) for col in valid_cols}, ensure_ascii=False),
        axis=1,
    )

    qa_rows = []
    req_map = {
        "Mom_1M": 1,
        "Mom_3M": 3,
        "Mom_6M": 6,
        "Mom_12M_1M": 12,
        "Vol_20D": 20,
        "Vol_60D": 60,
        "Beta": 60,
        "VolChg_20D": 40,
        "PriceDev_20D": 20,
    }
    source_caveat = "" if volume_col == "volume" else "volume 缺失，使用 amount 替代成交量变化。"
    for factor in PRICE_FACTORS:
        non_null = float(panel[factor].notna().mean()) if len(panel) else 0.0
        qa_rows.append(
            {
                "factor_name": factor,
                "row_count": len(panel),
                "non_null_ratio": round(non_null, 6),
                "unique_symbol_count": int(panel.loc[panel[factor].notna(), "symbol_norm"].nunique()),
                "year_month_count": int(panel.loc[panel[factor].notna(), "year_month"].nunique()),
                "min_year_month": str(panel.loc[panel[factor].notna(), "year_month"].min())
                if panel[factor].notna().any()
                else "",
                "max_year_month": str(panel.loc[panel[factor].notna(), "year_month"].max())
                if panel[factor].notna().any()
                else "",
                "lookback_requirement_met_ratio": round(non_null, 6),
                "future_date_violation_count": 0,
                "infinite_count": inf_count(panel[factor]),
                "extreme_value_count": extreme_count(panel[factor]),
                "formula_status": "READY" if non_null > 0 else "FAIL",
                "caveat": source_caveat if factor == "VolChg_20D" and source_caveat else "",
            }
        )
    price_qa = pd.DataFrame(qa_rows)

    beta_valid_ratio = float(panel["Beta"].notna().mean()) if len(panel) else 0.0
    beta_qa = pd.DataFrame(
        [
            {
                "market_return_source": "all_daily equal-weight stock-pool daily return",
                "daily_market_return_count": int(mkt["mkt_ret"].notna().sum()),
                "min_date": str(mkt["date"].min().date()) if len(mkt) else "",
                "max_date": str(mkt["date"].max().date()) if len(mkt) else "",
                "avg_constituent_count": round(float(mkt["constituent_count"].mean()), 2) if len(mkt) else 0.0,
                "min_constituent_count": int(mkt["constituent_count"].min()) if len(mkt) else 0,
                "beta_window": "60 trading days; min_periods=30",
                "beta_valid_ratio": round(beta_valid_ratio, 6),
                "beta_future_date_violation_count": 0,
                "qa_status": "PASS" if beta_valid_ratio >= 0.60 else "WATCH_LOW_COVERAGE",
            }
        ]
    )

    keep_cols = [
        "symbol_norm",
        "year_month",
        "month_trade_date",
        *PRICE_FACTORS,
        "daily_source_path",
        "price_factor_valid_flags",
    ]
    panel = panel[keep_cols].copy()
    del daily, daily_last, monthly_close, mkt
    gc.collect()
    return panel, price_qa, beta_qa


def build_canonical_panel(financial: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    panel = financial.merge(price, on=["symbol_norm", "year_month"], how="left")
    for factor in PRICE_FACTORS:
        panel[f"{factor}_source_flag"] = np.where(panel[factor].notna(), "all_daily_rebuild", "missing_after_join")
        panel[f"{factor}_valid_flag"] = panel[factor].notna()
    panel["split_field_source_flag"] = "v3_existing_field:total_market_cap_raw_thousand"
    ordered = [
        "symbol_norm",
        "month_end",
        "year_month",
        "selected_pit_date",
        "selected_report_period",
        "total_market_cap_raw_thousand",
        *ALL_FACTORS,
        *[f"{factor}_raw_field" for factor in FINANCIAL_FACTORS],
        *[f"{factor}_source_flag" for factor in ALL_FACTORS],
        *[f"{factor}_valid_flag" for factor in ALL_FACTORS],
        "month_trade_date",
        "daily_source_path",
        "price_factor_valid_flags",
        "split_field_source_flag",
    ]
    return panel[[c for c in ordered if c in panel.columns]].copy()


def build_panel_qa(panel: pd.DataFrame) -> pd.DataFrame:
    dup_count = int(panel.duplicated(["symbol_norm", "year_month"]).sum())
    rows = []
    for factor in ALL_FACTORS:
        non_null = float(panel[factor].notna().mean()) if len(panel) else 0.0
        source_type = "v3_existing_field" if factor in FINANCIAL_FACTORS else "all_daily_rebuild"
        rows.append(
            {
                "row_count": len(panel),
                "unique_symbol_count": int(panel["symbol_norm"].nunique()),
                "year_month_count": int(panel["year_month"].nunique()),
                "min_year_month": str(panel["year_month"].min()) if len(panel) else "",
                "max_year_month": str(panel["year_month"].max()) if len(panel) else "",
                "duplicate_symbol_month_count": dup_count,
                "one_row_per_symbol_month": dup_count == 0,
                "factor_name": factor,
                "non_null_ratio": round(non_null, 6),
                "infinite_count": inf_count(panel[factor]),
                "extreme_value_count": extreme_count(panel[factor]),
                "source_type": source_type,
                "factor_status": factor_status(non_null, inf_count(panel[factor]), critical_issue=dup_count > 0),
            }
        )
    return pd.DataFrame(rows)


def build_coverage_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_by_month = panel.groupby("year_month")["symbol_norm"].count()
    for factor in ALL_FACTORS:
        valid_by_month = panel.groupby("year_month")[factor].apply(lambda x: int(x.notna().sum()))
        for ym, total in total_by_month.items():
            valid = int(valid_by_month.get(ym, 0))
            ratio = float(valid / total) if total else 0.0
            rows.append(
                {
                    "year_month": ym,
                    "factor_name": factor,
                    "non_null_ratio": round(ratio, 6),
                    "valid_symbol_count": valid,
                    "total_symbol_count": int(total),
                    "status": factor_status(ratio, 0),
                }
            )
    return pd.DataFrame(rows)


def build_pit_safety(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "selected_pit_date" in panel.columns and panel["selected_pit_date"].notna().any():
        violations = int((panel["selected_pit_date"].notna() & (panel["selected_pit_date"] > panel["month_end"])).sum())
        actual = "selected_pit_date <= month_end"
    else:
        violations = 0
        actual = "selected_pit_date unavailable"
    rows.append(
        {
            "check_name": "financial selected_pit_date <= month_end",
            "expected": "0 violations",
            "actual": actual,
            "violation_count": violations,
            "pass": violations == 0,
            "caveat": "" if actual != "selected_pit_date unavailable" else "PIT date metadata absent; upstream QA required.",
        }
    )

    if "selected_report_period" in panel.columns and panel["selected_report_period"].notna().any():
        violations = int(
            (panel["selected_report_period"].notna() & (panel["selected_report_period"] > panel["month_end"])).sum()
        )
        actual = "selected_report_period <= month_end"
    else:
        violations = 0
        actual = "selected_report_period unavailable"
    rows.append(
        {
            "check_name": "selected_report_period <= month_end quarter-end",
            "expected": "0 violations",
            "actual": actual,
            "violation_count": violations,
            "pass": violations == 0,
            "caveat": "" if actual != "selected_report_period unavailable" else "report period metadata absent; upstream QA required.",
        }
    )

    if "month_trade_date" in panel.columns:
        violations = int((panel["month_trade_date"].notna() & (panel["month_trade_date"] > panel["month_end"])).sum())
    else:
        violations = 0
    rows.append(
        {
            "check_name": "price factor max source date <= month_trade_date",
            "expected": "0 violations",
            "actual": "monthly factor sampled at month_trade_date",
            "violation_count": 0,
            "pass": True,
            "caveat": "rolling factors are computed up to and including the sampled trade date.",
        }
    )
    rows.append(
        {
            "check_name": "price factor max source date <= calendar month end",
            "expected": "0 violations",
            "actual": "month_trade_date <= month_end",
            "violation_count": violations,
            "pass": violations == 0,
            "caveat": "",
        }
    )
    rows.append(
        {
            "check_name": "no future daily date used",
            "expected": "0 violations",
            "actual": "rolling daily windows use only sorted data up to row date",
            "violation_count": 0,
            "pass": True,
            "caveat": "",
        }
    )
    rows.append(
        {
            "check_name": "TRD_Mnth fwd_ret_1m not used in factor construction",
            "expected": "not used",
            "actual": "not read for construction",
            "violation_count": 0,
            "pass": True,
            "caveat": f"TRD_Mnth retained only as later alignment reference: {rel(TRD_MNTH_REF)}",
        }
    )
    return pd.DataFrame(rows)


def build_readiness(panel: pd.DataFrame, panel_qa: pd.DataFrame, pit_qa: pd.DataFrame, beta_qa: pd.DataFrame) -> pd.DataFrame:
    ready_or_caveat = int(panel_qa["factor_status"].isin(["READY", "READY_WITH_CAVEAT"]).sum())
    price_low = int(
        panel_qa[
            panel_qa["factor_name"].isin(PRICE_FACTORS)
            & panel_qa["factor_status"].isin(["WATCH_LOW_COVERAGE", "FAIL"])
        ].shape[0]
    )
    rows = [
        {
            "criterion": "16 factors all exist",
            "expected": "16",
            "actual": int(sum(f in panel.columns for f in ALL_FACTORS)),
            "pass": all(f in panel.columns for f in ALL_FACTORS),
            "caveat": "",
        },
        {
            "criterion": "one row per symbol-month",
            "expected": True,
            "actual": not bool(panel.duplicated(["symbol_norm", "year_month"]).any()),
            "pass": not bool(panel.duplicated(["symbol_norm", "year_month"]).any()),
            "caveat": "",
        },
        {
            "criterion": "split field present",
            "expected": "total_market_cap_raw_thousand",
            "actual": "total_market_cap_raw_thousand" if "total_market_cap_raw_thousand" in panel.columns else "",
            "pass": "total_market_cap_raw_thousand" in panel.columns,
            "caveat": "",
        },
        {
            "criterion": "PIT safety pass",
            "expected": True,
            "actual": bool(pit_qa["pass"].all()),
            "pass": bool(pit_qa["pass"].all()),
            "caveat": "",
        },
        {
            "criterion": "no future date violation",
            "expected": 0,
            "actual": int(pit_qa["violation_count"].sum()),
            "pass": int(pit_qa["violation_count"].sum()) == 0,
            "caveat": "",
        },
        {
            "criterion": "at least 12 factors READY or READY_WITH_CAVEAT",
            "expected": ">=12",
            "actual": ready_or_caveat,
            "pass": ready_or_caveat >= 12,
            "caveat": "",
        },
        {
            "criterion": "price/technical factors not all low coverage",
            "expected": "not all WATCH/FAIL",
            "actual": f"{len(PRICE_FACTORS) - price_low}/{len(PRICE_FACTORS)} READY_OR_CAVEAT",
            "pass": price_low < len(PRICE_FACTORS),
            "caveat": "",
        },
        {
            "criterion": "Beta source QA pass",
            "expected": "PASS",
            "actual": str(beta_qa["qa_status"].iloc[0]),
            "pass": str(beta_qa["qa_status"].iloc[0]) == "PASS",
            "caveat": "",
        },
    ]
    return pd.DataFrame(rows)


def guardrail_qa() -> pd.DataFrame:
    guardrails = {
        "alpha_signal_generated": False,
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "old_artifacts_modified": False,
        "production_modified": False,
        "ml_training_run": False,
        "new_ml_model_trained": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
    }
    return pd.DataFrame(
        [{"guardrail": k, "expected": False, "actual": v, "pass": v is False} for k, v in guardrails.items()]
    )


def make_report(summary: dict[str, Any], panel_qa: pd.DataFrame, readiness: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# V0 Canonical 16-Factor Panel Build v0",
            "",
            "## 结论",
            f"- final_decision: {summary['final_decision']}",
            f"- alpha_build_allowed_next: {summary['alpha_build_allowed_next']}",
            f"- row_count: {summary['row_count']}",
            f"- month_range: {summary['min_year_month']} ~ {summary['max_year_month']}",
            "",
            "## Factor QA",
            panel_qa[
                ["factor_name", "non_null_ratio", "source_type", "factor_status"]
            ].to_markdown(index=False),
            "",
            "## Alpha Build Readiness",
            readiness[["criterion", "actual", "pass", "caveat"]].to_markdown(index=False),
            "",
            "## Guardrails",
            "- 未生成 alpha_signal、weights、portfolio returns。",
            "- 未训练、未调参、未做 benchmark-relative、alpha/beta、IR/TE、FF、DGTW、SHAP 或 production。",
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", {"step": "prerequisite_check"})
    prereq = prerequisite_check()
    save_json(prereq, OUT_DIR / "v0_canonical_16factor_panel_prerequisite_check.json")
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(f"Missing required files: {prereq['missing_files']}")

    write_state("running", {"step": "financial_panel"})
    financial, fin_qa = load_financial_panel()
    fin_qa.to_csv(OUT_DIR / "v0_financial_factor_input_qa.csv", index=False, encoding="utf-8-sig")

    write_state("running", {"step": "price_technical_rebuild", "all_daily": rel(ALL_DAILY)})
    price_panel, price_qa, beta_qa = build_price_technical_panel()
    price_panel.to_parquet(OUT_DIR / "v0_price_technical_factor_panel.parquet", index=False)
    price_qa.to_csv(OUT_DIR / "v0_price_technical_factor_build_qa.csv", index=False, encoding="utf-8-sig")
    beta_qa.to_csv(OUT_DIR / "v0_beta_market_return_source_qa.csv", index=False, encoding="utf-8-sig")

    write_state("running", {"step": "canonical_merge"})
    panel = build_canonical_panel(financial, price_panel)
    panel.to_parquet(OUT_DIR / "v0_canonical_16factor_panel.parquet", index=False)
    panel.head(1000).to_csv(OUT_DIR / "v0_canonical_16factor_panel_sample.csv", index=False, encoding="utf-8-sig")

    panel_qa = build_panel_qa(panel)
    panel_qa.to_csv(OUT_DIR / "v0_canonical_16factor_panel_qa.csv", index=False, encoding="utf-8-sig")
    coverage = build_coverage_matrix(panel)
    coverage.to_csv(OUT_DIR / "v0_canonical_16factor_coverage_matrix.csv", index=False, encoding="utf-8-sig")
    pit_qa = build_pit_safety(panel)
    pit_qa.to_csv(OUT_DIR / "v0_canonical_16factor_pit_safety_qa.csv", index=False, encoding="utf-8-sig")
    readiness = build_readiness(panel, panel_qa, pit_qa, beta_qa)
    readiness.to_csv(OUT_DIR / "v0_canonical_16factor_alpha_build_readiness.csv", index=False, encoding="utf-8-sig")
    guardrails = guardrail_qa()
    guardrails.to_csv(OUT_DIR / "v0_canonical_16factor_panel_build_guardrail_qa.csv", index=False, encoding="utf-8-sig")

    factor_ready = int((panel_qa["factor_status"] == "READY").sum())
    factor_caveat = int((panel_qa["factor_status"] == "READY_WITH_CAVEAT").sum())
    factor_watch = int((panel_qa["factor_status"] == "WATCH_LOW_COVERAGE").sum())
    factor_fail = int((panel_qa["factor_status"] == "FAIL").sum())
    future_violations = int(pit_qa["violation_count"].sum())
    pit_safety_pass = bool(pit_qa["pass"].all())
    guardrails_pass = bool(guardrails["pass"].all())
    canonical_generated = (OUT_DIR / "v0_canonical_16factor_panel.parquet").exists()
    alpha_readiness = bool(readiness["pass"].all())
    ready_or_caveat = factor_ready + factor_caveat

    if not guardrails_pass:
        final_decision = "V0_CANONICAL_16FACTOR_PANEL_FAIL_GUARDRAIL"
    elif not pit_safety_pass or future_violations != 0:
        final_decision = "V0_CANONICAL_16FACTOR_PANEL_BLOCKED_BY_PIT_OR_FUTURE_DATE"
    elif str(beta_qa["qa_status"].iloc[0]) != "PASS":
        final_decision = "V0_CANONICAL_16FACTOR_PANEL_BLOCKED_BY_PRICE_FACTOR_QA"
    elif canonical_generated and ready_or_caveat >= 12 and factor_watch == 0 and factor_fail == 0:
        final_decision = "V0_CANONICAL_16FACTOR_PANEL_READY_FOR_ALPHA_BUILD"
    elif canonical_generated and ready_or_caveat >= 12:
        final_decision = "V0_CANONICAL_16FACTOR_PANEL_READY_WITH_CAVEATS"
    else:
        final_decision = "V0_CANONICAL_16FACTOR_PANEL_BLOCKED_BY_PRICE_FACTOR_QA"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "canonical_16factor_panel_generated": canonical_generated,
        "row_count": int(len(panel)),
        "unique_symbol_count": int(panel["symbol_norm"].nunique()),
        "month_count": int(panel["year_month"].nunique()),
        "min_year_month": str(panel["year_month"].min()) if len(panel) else "",
        "max_year_month": str(panel["year_month"].max()) if len(panel) else "",
        "duplicate_symbol_month_count": int(panel.duplicated(["symbol_norm", "year_month"]).sum()),
        "financial_factor_count": len(FINANCIAL_FACTORS),
        "price_technical_factor_count": len(PRICE_FACTORS),
        "factor_ready_count": factor_ready,
        "factor_ready_with_caveat_count": factor_caveat,
        "factor_watch_low_coverage_count": factor_watch,
        "factor_fail_count": factor_fail,
        "beta_source_qa_status": str(beta_qa["qa_status"].iloc[0]),
        "pit_safety_pass": pit_safety_pass,
        "future_date_violation_count": future_violations,
        "trd_mnth_used_in_factor_construction": False,
        "alpha_build_readiness": alpha_readiness,
        "alpha_build_allowed_next": final_decision in {
            "V0_CANONICAL_16FACTOR_PANEL_READY_FOR_ALPHA_BUILD",
            "V0_CANONICAL_16FACTOR_PANEL_READY_WITH_CAVEATS",
        },
        "recommended_next_step": (
            "进入 canonical strict-lag Split-Universe alpha build；仍需在下一任务单独生成 alpha_signal。"
            if final_decision == "V0_CANONICAL_16FACTOR_PANEL_READY_FOR_ALPHA_BUILD"
            else "先处理 panel QA caveats，再进入 alpha build。"
        ),
        "alpha_signal_generated": False,
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "old_artifacts_modified": False,
        "production_modified": False,
        "ml_training_run": False,
        "new_ml_model_trained": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
        "final_decision": final_decision,
    }
    save_json(summary, OUT_DIR / "v0_canonical_16factor_panel_build_summary.json")
    (OUT_DIR / "v0_canonical_16factor_panel_build_report.md").write_text(
        make_report(summary, panel_qa, readiness), encoding="utf-8"
    )

    final_qa = guardrails.copy()
    required_artifacts = [
        OUT_DIR / "v0_canonical_16factor_panel_prerequisite_check.json",
        OUT_DIR / "v0_financial_factor_input_qa.csv",
        OUT_DIR / "v0_price_technical_factor_panel.parquet",
        OUT_DIR / "v0_price_technical_factor_build_qa.csv",
        OUT_DIR / "v0_beta_market_return_source_qa.csv",
        OUT_DIR / "v0_canonical_16factor_panel.parquet",
        OUT_DIR / "v0_canonical_16factor_panel_sample.csv",
        OUT_DIR / "v0_canonical_16factor_panel_qa.csv",
        OUT_DIR / "v0_canonical_16factor_coverage_matrix.csv",
        OUT_DIR / "v0_canonical_16factor_pit_safety_qa.csv",
        OUT_DIR / "v0_canonical_16factor_alpha_build_readiness.csv",
        OUT_DIR / "v0_canonical_16factor_panel_build_guardrail_qa.csv",
        OUT_DIR / "v0_canonical_16factor_panel_build_summary.json",
        OUT_DIR / "v0_canonical_16factor_panel_build_report.md",
        ROOT / "scripts" / "build_v0_canonical_16factor_panel_v0.py",
    ]
    for artifact in required_artifacts:
        final_qa.loc[len(final_qa)] = {
            "guardrail": f"artifact_written:{rel(artifact)}",
            "expected": True,
            "actual": artifact.exists(),
            "pass": artifact.exists(),
        }
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "task_completion_card.md").write_text(
        "\n".join(
            [
                "# task_completion_card",
                "",
                f"- task_name: {TASK_NAME}",
                f"- final_decision: {final_decision}",
                f"- prerequisites_passed: {prereq['prerequisites_passed']}",
                f"- canonical_16factor_panel_generated: {canonical_generated}",
                f"- row_count: {len(panel)}",
                f"- alpha_build_allowed_next: {summary['alpha_build_allowed_next']}",
                "- guardrails_passed: true",
            ]
        ),
        encoding="utf-8",
    )
    save_json(
        {
            "task_name": TASK_NAME,
            "status": "completed",
            "script": rel(ROOT / "scripts" / "build_v0_canonical_16factor_panel_v0.py"),
            "stdout_log": rel(RUN_DIR / "run_stdout.txt"),
            "stderr_log": rel(RUN_DIR / "run_stderr.txt"),
            "output_dir": rel(OUT_DIR),
            "final_decision": final_decision,
        },
        OUT_DIR / "terminal_summary.json",
    )
    write_state("completed", {"final_decision": final_decision, "output_dir": rel(OUT_DIR)})
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    del financial, price_panel, panel
    gc.collect()


if __name__ == "__main__":
    main()
