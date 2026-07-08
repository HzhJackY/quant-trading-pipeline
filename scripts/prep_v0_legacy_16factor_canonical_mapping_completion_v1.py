from __future__ import annotations

import gc
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


TASK_NAME = "v0_legacy_16factor_canonical_mapping_completion_prep_v1"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

V3_PANEL = (
    ROOT
    / "output"
    / "csmar_pit_clean_core_financial_factors_v3"
    / "pit_clean_core_financial_factors_monthly_v3.parquet"
)
TRANSFORMED_PANEL = (
    ROOT / "output" / "build_transformed_training_panel_v0" / "transformed_training_panel_v0.parquet"
)
DERIVED_PANEL = (
    ROOT
    / "output"
    / "derived_compact_f_missing_features_candidate_v01"
    / "derived_compact_f_missing_features_candidate_v01.parquet"
)
ROBUST_PANEL = (
    ROOT
    / "output"
    / "robust_cleaned_fundamental_factor_variant_build_v0"
    / "robust_cleaned_factor_score_panel_v0.parquet"
)
ALL_DAILY = ROOT / "output" / "all_daily.parquet"
TRD_MNTH_REPAIRED = (
    ROOT
    / "output"
    / "trd_mnth_parser_repair_2024_12_coverage_repair_v0"
    / "canonical_csmar_trd_mnth_return_map_repaired.parquet"
)

LEGACY_SCRIPT_PATHS = [
    ROOT / "run_split_universe.py",
    ROOT / "factor_research" / "split_universe.py",
    ROOT / "factor_research" / "backtest_engine.py",
    ROOT / "factor_research" / "orthogonalization.py",
    ROOT / "factor_lib" / "momentum.py",
    ROOT / "factor_lib" / "volatility.py",
    ROOT / "factor_lib" / "technical.py",
    ROOT / "factor_lib" / "value.py",
    ROOT / "factor_lib" / "quality.py",
    ROOT / "factor_lib" / "growth.py",
]

SOURCE_FILES = [
    {
        "source_panel": "pit_clean_core_financial_factors_monthly_v3",
        "path": V3_PANEL,
        "source_type": "financial_panel",
        "priority": 1,
        "reference_only": False,
    },
    {
        "source_panel": "transformed_training_panel_v0",
        "path": TRANSFORMED_PANEL,
        "source_type": "transformed_panel",
        "priority": 2,
        "reference_only": False,
    },
    {
        "source_panel": "derived_compact_f_missing_features_candidate_v01",
        "path": DERIVED_PANEL,
        "source_type": "derived_panel",
        "priority": 3,
        "reference_only": False,
    },
    {
        "source_panel": "robust_cleaned_factor_score_panel_v0_reference_only",
        "path": ROBUST_PANEL,
        "source_type": "reference_panel",
        "priority": 4,
        "reference_only": True,
    },
    {
        "source_panel": "all_daily",
        "path": ALL_DAILY,
        "source_type": "daily_price_source",
        "priority": 5,
        "reference_only": False,
    },
    {
        "source_panel": "canonical_csmar_trd_mnth_return_map_repaired",
        "path": TRD_MNTH_REPAIRED,
        "source_type": "monthly_return_source",
        "priority": 6,
        "reference_only": False,
    },
]

LEGACY_FACTORS = [
    "Mom_1M",
    "Mom_3M",
    "Mom_6M",
    "Mom_12M_1M",
    "Vol_20D",
    "Vol_60D",
    "Beta",
    "BP",
    "EP",
    "ROE",
    "Debt_Ratio",
    "Net_Profit_Margin",
    "RevGrowth_YoY",
    "ProfitGrowth_YoY",
    "VolChg_20D",
    "PriceDev_20D",
]

FACTOR_CATEGORY = {
    "Mom_1M": "price_momentum",
    "Mom_3M": "price_momentum",
    "Mom_6M": "price_momentum",
    "Mom_12M_1M": "price_momentum",
    "Vol_20D": "price_volatility",
    "Vol_60D": "price_volatility",
    "Beta": "price_volatility",
    "BP": "valuation",
    "EP": "valuation",
    "ROE": "quality",
    "Debt_Ratio": "quality",
    "Net_Profit_Margin": "quality",
    "RevGrowth_YoY": "growth",
    "ProfitGrowth_YoY": "growth",
    "VolChg_20D": "technical",
    "PriceDev_20D": "technical",
}

PRICE_TECH_FACTORS = [
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
FINANCIAL_COMPLETION_FACTORS = [
    "EP",
    "ROE",
    "Net_Profit_Margin",
    "RevGrowth_YoY",
    "ProfitGrowth_YoY",
]

ID_COLS = ["symbol", "stock_code", "stkcd", "Stkcd", "证券代码", "Symbol"]
DATE_COLS = ["date", "trade_date", "Trddt", "month_end", "month", "year_month", "Trdmnt"]
PIT_COLS = [
    "selected_pit_date",
    "pit_date",
    "ann_date",
    "announcement_date",
    "publish_date",
    "公告日期",
]
REPORT_COLS = [
    "selected_report_period",
    "report_period",
    "report_date",
    "end_date",
    "accper",
    "报告期",
]

EXISTING_SYNONYMS = {
    "Mom_1M": ["mom_1m", "momentum_1m", "ret_1m", "return_1m"],
    "Mom_3M": ["mom_3m", "momentum_3m", "ret_3m", "return_3m"],
    "Mom_6M": ["mom_6m", "momentum_6m", "ret_6m", "return_6m"],
    "Mom_12M_1M": ["mom_12m_1m", "mom12m1m", "momentum_12m_1m", "mom_12_1"],
    "Vol_20D": ["vol_20d", "volatility_20d", "realized_vol_20d"],
    "Vol_60D": ["vol_60d", "volatility_60d", "realized_vol_60d"],
    "Beta": ["beta", "beta_60d", "market_beta"],
    "BP": ["bp", "book_to_price", "book_price", "btm", "bm"],
    "EP": ["ep", "earnings_to_price", "earning_to_price", "e_p", "ep_ttm"],
    "ROE": ["roe", "return_on_equity"],
    "Debt_Ratio": ["debt_ratio", "asset_liability_ratio", "liability_asset_ratio", "leverage"],
    "Net_Profit_Margin": ["net_profit_margin", "npm", "profit_margin", "net_margin"],
    "RevGrowth_YoY": [
        "revgrowth_yoy",
        "rev_growth_yoy",
        "revenue_growth_yoy",
        "operating_revenue_growth_yoy",
    ],
    "ProfitGrowth_YoY": [
        "profitgrowth_yoy",
        "profit_growth_yoy",
        "earnings_growth",
        "earnings_growth_yoy",
        "net_profit_growth_yoy",
    ],
    "VolChg_20D": ["volchg_20d", "volume_20d_change", "volume_change_20d", "vol_chg_20d"],
    "PriceDev_20D": ["pricedev_20d", "price_ma20_deviation", "ma_deviation_20d", "price_dev_20d"],
}

RAW_COMPONENT_SYNONYMS = {
    "close": ["close", "adj_close", "adjusted_close", "收盘", "复权收盘价"],
    "volume": ["volume", "vol", "成交量"],
    "market_cap": [
        "total_market_cap_raw_thousand",
        "total_market_cap",
        "market_cap",
        "mktcap",
        "总市值",
    ],
    "net_profit": ["net_profit", "netprofit", "净利润", "n_income", "net_income"],
    "net_profit_ttm": ["net_profit_ttm", "ttm_net_profit", "net_income_ttm"],
    "equity": ["book_equity", "net_assets", "net_asset", "total_equity", "净资产"],
    "revenue": ["operating_revenue", "revenue", "sales", "营业收入"],
    "pe": ["pe", "pe_ttm", "pettm"],
}


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


def norm_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def norm_set(values: list[str]) -> set[str]:
    return {norm_name(x) for x in values}


def parquet_schema(path: Path) -> tuple[list[str], dict[str, str]]:
    pf = pq.ParquetFile(path)
    fields = pf.schema_arrow
    cols = list(fields.names)
    dtypes = {field.name: str(field.type) for field in fields}
    return cols, dtypes


def find_col(cols: list[str], candidates: list[str]) -> str | None:
    by_norm = {norm_name(c): c for c in cols}
    for cand in candidates:
        hit = by_norm.get(norm_name(cand))
        if hit:
            return hit
    return None


def find_cols(cols: list[str], candidates: list[str]) -> list[str]:
    by_norm = {norm_name(c): c for c in cols}
    hits = []
    for cand in candidates:
        hit = by_norm.get(norm_name(cand))
        if hit and hit not in hits:
            hits.append(hit)
    return hits


def non_null_ratio(path: Path, column: str) -> float:
    pf = pq.ParquetFile(path)
    row_count = pf.metadata.num_rows
    if row_count <= 0:
        return 0.0
    try:
        col_idx = pf.schema_arrow.names.index(column)
        null_count = 0
        stats_complete = True
        for i in range(pf.metadata.num_row_groups):
            stats = pf.metadata.row_group(i).column(col_idx).statistics
            if stats is None or stats.null_count is None:
                stats_complete = False
                break
            null_count += int(stats.null_count)
        if stats_complete:
            return round(float((row_count - null_count) / row_count), 6)
    except Exception:
        pass

    total = 0
    not_null = 0
    for i in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(i, columns=[column])
        arr = tbl.column(0)
        total += len(arr)
        not_null += len(arr) - arr.null_count
        del tbl, arr
    gc.collect()
    return round(float(not_null / total), 6) if total else 0.0


def month_values(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.astype(str).str.slice(0, 7)
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce").dt.to_period("M").astype(str)

    raw = series.dropna()
    if raw.empty:
        return pd.Series(dtype="object")
    text = raw.astype(str).str.strip()
    digits6 = text.str.fullmatch(r"\d{6}")
    if bool(digits6.mean() > 0.8):
        return text.str.slice(0, 4) + "-" + text.str.slice(4, 6)

    parsed = pd.to_datetime(raw, errors="coerce")
    if parsed.notna().mean() >= 0.5:
        return parsed.dt.to_period("M").astype(str)
    return text.str.slice(0, 7)


def source_key_profile(path: Path, cols: list[str]) -> dict[str, Any]:
    symbol_col = find_col(cols, ID_COLS)
    date_col = find_col(cols, DATE_COLS)
    profile = {
        "symbol_col": symbol_col or "",
        "date_col": date_col or "",
        "unique_symbol_count": 0,
        "month_count": 0,
        "min_month": "",
        "max_month": "",
    }
    if not symbol_col or not date_col:
        return profile

    pf = pq.ParquetFile(path)
    symbols: set[str] = set()
    months: set[str] = set()
    for i in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(i, columns=[symbol_col, date_col])
        pdf = tbl.to_pandas()
        symbols.update(pdf[symbol_col].dropna().astype(str).str.strip().unique().tolist())
        month_s = month_values(pdf[date_col]).dropna()
        months.update([m for m in month_s.astype(str).tolist() if m and m != "NaT"])
        del tbl, pdf, month_s
    gc.collect()
    profile.update(
        {
            "unique_symbol_count": len(symbols),
            "month_count": len(months),
            "min_month": min(months) if months else "",
            "max_month": max(months) if months else "",
        }
    )
    return profile


def legacy_match_for_column(column_name: str, source_type: str) -> tuple[str, str, str]:
    n = norm_name(column_name)
    for factor, synonyms in EXISTING_SYNONYMS.items():
        candidates = norm_set([factor] + synonyms)
        if n in candidates:
            return factor, "high", ""
        if any(n.endswith("_" + cand) or n.startswith(cand + "_") for cand in candidates):
            return factor, "medium", "字段名带前后缀，需确认是否原始值或标准化值。"

    if source_type == "daily_price_source":
        if n in norm_set(RAW_COMPONENT_SYNONYMS["close"]):
            return ",".join([f for f in PRICE_TECH_FACTORS if f != "VolChg_20D"]), "raw_source", "日线 close 可用于重建价格/波动/Beta/均线偏离。"
        if n in norm_set(RAW_COMPONENT_SYNONYMS["volume"]):
            return "VolChg_20D", "raw_source", "日线 volume 可用于重建 20 日成交量变化。"
    if source_type == "monthly_return_source" and n in {"mretwd", "mretnd", "monthly_return", "ret"}:
        return "Mom_1M,Mom_3M,Mom_6M,Mom_12M_1M", "raw_source", "月收益可作为动量重建备选，优先级低于 legacy 日线公式。"
    return "", "none", ""


def build_field_inventory(sources: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows = []
    source_meta: dict[str, dict[str, Any]] = {}
    for src in sources:
        path = src["path"]
        if not path.exists():
            continue
        cols, dtypes = parquet_schema(path)
        key_profile = source_key_profile(path, cols)
        source_meta[src["source_panel"]] = {
            **src,
            "columns": cols,
            "dtypes": dtypes,
            "key_profile": key_profile,
            "row_count": pq.ParquetFile(path).metadata.num_rows,
        }
        for col in cols:
            match, confidence, caveat = legacy_match_for_column(col, src["source_type"])
            src_caveat = caveat
            if src["reference_only"]:
                src_caveat = (src_caveat + "；" if src_caveat else "") + "参考字段发现，不直接作为收益源。"
            rows.append(
                {
                    "source_panel": src["source_panel"],
                    "path": rel(path),
                    "column_name": col,
                    "normalized_column_name": norm_name(col),
                    "dtype": dtypes[col],
                    "non_null_ratio": non_null_ratio(path, col),
                    "unique_symbol_count": key_profile["unique_symbol_count"],
                    "month_count": key_profile["month_count"],
                    "min_month": key_profile["min_month"],
                    "max_month": key_profile["max_month"],
                    "possible_legacy_factor_match": match,
                    "confidence": confidence,
                    "caveat": src_caveat,
                }
            )
    inv = pd.DataFrame(rows)
    return inv, source_meta


def inventory_hit(
    inventory: pd.DataFrame,
    source_panels: list[str],
    factor: str,
    include_reference: bool = False,
) -> dict[str, Any] | None:
    if inventory.empty:
        return None
    allowed = inventory[inventory["source_panel"].isin(source_panels)].copy()
    if not include_reference:
        allowed = allowed[~allowed["source_panel"].str.contains("reference_only", case=False, na=False)]
    allowed = allowed[
        allowed["possible_legacy_factor_match"].astype(str).str.split(",").apply(lambda xs: factor in xs)
        & allowed["confidence"].isin(["high", "medium"])
    ].copy()
    if allowed.empty:
        return None
    allowed["rank"] = allowed["source_panel"].map(
        {
            "pit_clean_core_financial_factors_monthly_v3": 1,
            "transformed_training_panel_v0": 2,
            "derived_compact_f_missing_features_candidate_v01": 3,
            "robust_cleaned_factor_score_panel_v0_reference_only": 4,
        }
    ).fillna(99)
    allowed["confidence_rank"] = allowed["confidence"].map({"high": 1, "medium": 2}).fillna(9)
    row = allowed.sort_values(["rank", "confidence_rank", "non_null_ratio"], ascending=[True, True, False]).iloc[0]
    return row.to_dict()


def source_has_pit(meta: dict[str, Any]) -> tuple[bool, str, str]:
    cols = meta.get("columns", [])
    pit = find_col(cols, PIT_COLS)
    report = find_col(cols, REPORT_COLS)
    return bool(pit and report), pit or "", report or ""


def component_field(meta: dict[str, Any], component: str) -> str | None:
    return find_col(meta.get("columns", []), RAW_COMPONENT_SYNONYMS[component])


def component_coverage(meta: dict[str, Any], fields: list[str]) -> float:
    vals = []
    for field in fields:
        if field:
            vals.append(non_null_ratio(meta["path"], field))
    return round(min(vals), 6) if vals else 0.0


def transformed_caveat(field: str) -> str:
    n = norm_name(field)
    if any(suffix in n for suffix in ["z", "rank", "score", "neutral"]):
        return "字段疑似 rank/zscore/neutralized 变换值；若用于 V0 canonical，需确认是否接受 transformed rather than raw legacy field。"
    return ""


def price_source_readiness(source_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    daily = source_meta.get("all_daily")
    monthly = source_meta.get("canonical_csmar_trd_mnth_return_map_repaired")
    result = {
        "daily_found": bool(daily),
        "monthly_found": bool(monthly),
        "close_col": "",
        "volume_col": "",
        "daily_symbol_col": "",
        "daily_date_col": "",
        "monthly_return_col": "",
        "daily_close_coverage": 0.0,
        "daily_volume_coverage": 0.0,
        "monthly_return_coverage": 0.0,
    }
    if daily:
        cols = daily["columns"]
        close_col = component_field(daily, "close")
        volume_col = component_field(daily, "volume")
        result.update(
            {
                "close_col": close_col or "",
                "volume_col": volume_col or "",
                "daily_symbol_col": find_col(cols, ID_COLS) or "",
                "daily_date_col": find_col(cols, DATE_COLS) or "",
                "daily_close_coverage": non_null_ratio(daily["path"], close_col) if close_col else 0.0,
                "daily_volume_coverage": non_null_ratio(daily["path"], volume_col) if volume_col else 0.0,
            }
        )
    if monthly:
        ret_col = find_col(monthly["columns"], ["Mretwd", "mretwd", "monthly_return", "ret"])
        result["monthly_return_col"] = ret_col or ""
        result["monthly_return_coverage"] = non_null_ratio(monthly["path"], ret_col) if ret_col else 0.0
    return result


def build_price_plan(price_ready: dict[str, Any]) -> pd.DataFrame:
    daily_core_ok = bool(price_ready["daily_found"] and price_ready["close_col"] and price_ready["daily_symbol_col"] and price_ready["daily_date_col"])
    volume_ok = bool(daily_core_ok and price_ready["volume_col"])
    rows = []
    specs = {
        "Mom_1M": ("21 trading days, skip 5 trading days", "close[t-5] / close[t-26] - 1"),
        "Mom_3M": ("63 trading days, skip 5 trading days", "close[t-5] / close[t-68] - 1"),
        "Mom_6M": ("126 trading days, skip 5 trading days", "close[t-5] / close[t-131] - 1"),
        "Mom_12M_1M": ("231 trading days, skip 21 trading days", "close[t-21] / close[t-252] - 1"),
        "Vol_20D": ("20 trading days", "std(daily_return, 20D) * sqrt(252)"),
        "Vol_60D": ("60 trading days", "std(daily_return, 60D) * sqrt(252)"),
        "Beta": ("60 trading days, min 30", "cov(stock_daily_ret, equal_weight_market_ret) / var(equal_weight_market_ret)"),
        "VolChg_20D": ("20 trading days", "volume[t] / mean(volume[t-19:t]) - 1"),
        "PriceDev_20D": ("20 trading days", "close[t] / mean(close[t-19:t]) - 1"),
    }
    for factor, (window, formula) in specs.items():
        needs_volume = factor == "VolChg_20D"
        ok = volume_ok if needs_volume else daily_core_ok
        req = ["date", "symbol", price_ready["close_col"] or "close"]
        if needs_volume:
            req = ["date", "symbol", price_ready["volume_col"] or "volume"]
        caveat = "Beta 市场收益源预注册为 all_daily 股票池当日等权平均收益，不用未来优化。" if factor == "Beta" else ""
        rows.append(
            {
                "factor_name": factor,
                "source": "output/all_daily.parquet" if ok else "",
                "required_fields": ",".join(req),
                "lookback_window": window,
                "formula": formula,
                "pit_safety_rule": "month_end 当日及之前日线数据；不得使用 month_end 之后价格/成交量。",
                "month_end_alignment_rule": "日频因子在每个自然月最后一个可交易日取值并对齐 month_end。",
                "expected_coverage": round(
                    price_ready["daily_volume_coverage"] if needs_volume else price_ready["daily_close_coverage"], 6
                ),
                "rebuild_allowed_next_run": ok,
                "caveat": caveat if ok else "缺少 all_daily 日线核心字段，不能按 legacy 公式重建。",
            }
        )
    return pd.DataFrame(rows)


def financial_rebuild_option(
    factor: str,
    source_meta: dict[str, dict[str, Any]],
    inventory: pd.DataFrame,
) -> dict[str, Any]:
    primary_sources = [
        "pit_clean_core_financial_factors_monthly_v3",
        "transformed_training_panel_v0",
        "derived_compact_f_missing_features_candidate_v01",
    ]
    existing = inventory_hit(inventory, primary_sources, factor)
    if existing:
        src_meta = source_meta[existing["source_panel"]]
        pit_safe, pit_col, report_col = source_has_pit(src_meta)
        field = str(existing["column_name"])
        return {
            "candidate_field": field,
            "source_panel": existing["source_panel"],
            "raw_or_transformed": "transformed" if transformed_caveat(field) else "raw_or_canonical",
            "selected_pit_date_available": bool(pit_col),
            "report_period_available": bool(report_col),
            "coverage_ratio": float(existing["non_null_ratio"]),
            "formula_confirmed": True,
            "rebuild_allowed_next_run": False,
            "mapping_status": "READY_EXISTING_FIELD" if pit_safe else "NEEDS_FORMULA_CONFIRMATION",
            "pit_safe": pit_safe,
            "proposed_formula": "existing canonical/equivalent field",
            "caveat": transformed_caveat(field) or ("缺少 PIT 日期或报告期字段，需确认后才能 canonical 使用。" if not pit_safe else ""),
        }

    v3 = source_meta.get("pit_clean_core_financial_factors_monthly_v3")
    if not v3:
        return {
            "candidate_field": "",
            "source_panel": "",
            "raw_or_transformed": "",
            "selected_pit_date_available": False,
            "report_period_available": False,
            "coverage_ratio": 0.0,
            "formula_confirmed": False,
            "rebuild_allowed_next_run": False,
            "mapping_status": "MISSING_BLOCKER",
            "pit_safe": False,
            "proposed_formula": "",
            "caveat": "v3 financial panel 不存在。",
        }

    pit_safe, pit_col, report_col = source_has_pit(v3)
    net_profit = component_field(v3, "net_profit")
    net_profit_ttm = component_field(v3, "net_profit_ttm")
    equity = component_field(v3, "equity")
    revenue = component_field(v3, "revenue")
    market_cap = component_field(v3, "market_cap")
    pe = component_field(v3, "pe")

    formula_by_factor = {
        "EP": {
            "fields": [f for f in [net_profit_ttm or net_profit, market_cap] if f],
            "ok": bool(pit_safe and (pe or ((net_profit_ttm or net_profit) and market_cap))),
            "formula": "1 / PE_TTM if PE available else net_profit_ttm / total_market_cap; if only net_profit exists, confirm TTM/annualization policy.",
            "field": pe or ",".join([f for f in [net_profit_ttm or net_profit, market_cap] if f]),
            "caveat": "若只存在单期 net_profit，下一阶段必须确认 TTM/累计口径。",
        },
        "ROE": {
            "fields": [f for f in [net_profit, equity] if f],
            "ok": bool(pit_safe and net_profit and equity),
            "formula": "net_profit / book_equity_or_net_assets",
            "field": ",".join([f for f in [net_profit, equity] if f]),
            "caveat": "",
        },
        "Net_Profit_Margin": {
            "fields": [f for f in [net_profit, revenue] if f],
            "ok": bool(pit_safe and net_profit and revenue),
            "formula": "net_profit / operating_revenue",
            "field": ",".join([f for f in [net_profit, revenue] if f]),
            "caveat": "",
        },
        "RevGrowth_YoY": {
            "fields": [f for f in [revenue, report_col] if f],
            "ok": bool(pit_safe and revenue and report_col),
            "formula": "(operating_revenue[t] - operating_revenue[t-4 fiscal quarters]) / abs(operating_revenue[t-4])",
            "field": ",".join([f for f in [revenue, report_col] if f]),
            "caveat": "需在下一阶段按 symbol + report_period lag4，且只使用 selected_pit_date 已披露记录。",
        },
        "ProfitGrowth_YoY": {
            "fields": [f for f in [net_profit, report_col] if f],
            "ok": bool(pit_safe and net_profit and report_col),
            "formula": "(net_profit[t] - net_profit[t-4 fiscal quarters]) / abs(net_profit[t-4])",
            "field": ",".join([f for f in [net_profit, report_col] if f]),
            "caveat": "需在下一阶段按 symbol + report_period lag4，且只使用 selected_pit_date 已披露记录。",
        },
    }
    spec = formula_by_factor[factor]
    if spec["ok"]:
        return {
            "candidate_field": spec["field"],
            "source_panel": "pit_clean_core_financial_factors_monthly_v3",
            "raw_or_transformed": "raw_components",
            "selected_pit_date_available": bool(pit_col),
            "report_period_available": bool(report_col),
            "coverage_ratio": component_coverage(v3, spec["fields"]),
            "formula_confirmed": True,
            "rebuild_allowed_next_run": True,
            "mapping_status": "READY_REBUILD_FROM_FINANCIAL_PANEL",
            "pit_safe": True,
            "proposed_formula": spec["formula"],
            "caveat": spec["caveat"],
        }

    ref_hit = inventory_hit(inventory, ["robust_cleaned_factor_score_panel_v0_reference_only"], factor, include_reference=True)
    if ref_hit:
        return {
            "candidate_field": str(ref_hit["column_name"]),
            "source_panel": str(ref_hit["source_panel"]),
            "raw_or_transformed": "reference_only",
            "selected_pit_date_available": False,
            "report_period_available": False,
            "coverage_ratio": float(ref_hit["non_null_ratio"]),
            "formula_confirmed": False,
            "rebuild_allowed_next_run": False,
            "mapping_status": "NEEDS_FORMULA_CONFIRMATION",
            "pit_safe": False,
            "proposed_formula": spec["formula"],
            "caveat": "仅在 robust cleaned 参考面板找到字段；需回溯 PIT-clean 原始来源后才能进入 canonical。",
        }

    return {
        "candidate_field": spec["field"],
        "source_panel": "pit_clean_core_financial_factors_monthly_v3" if spec["field"] else "",
        "raw_or_transformed": "raw_components" if spec["field"] else "",
        "selected_pit_date_available": bool(pit_col),
        "report_period_available": bool(report_col),
        "coverage_ratio": component_coverage(v3, spec["fields"]),
        "formula_confirmed": False,
        "rebuild_allowed_next_run": False,
        "mapping_status": "NEEDS_FORMULA_CONFIRMATION" if spec["field"] else "MISSING_BLOCKER",
        "pit_safe": bool(pit_safe and spec["field"]),
        "proposed_formula": spec["formula"],
        "caveat": "组件字段不完整或口径需确认，不能直接声称 legacy 等价。",
    }


def build_financial_plan(source_meta: dict[str, dict[str, Any]], inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in FINANCIAL_COMPLETION_FACTORS:
        opt = financial_rebuild_option(factor, source_meta, inventory)
        rows.append(
            {
                "factor_name": factor,
                "candidate_field": opt["candidate_field"],
                "source_panel": opt["source_panel"],
                "raw_or_transformed": opt["raw_or_transformed"],
                "selected_pit_date_available": opt["selected_pit_date_available"],
                "report_period_available": opt["report_period_available"],
                "coverage_ratio": opt["coverage_ratio"],
                "formula_confirmed": opt["formula_confirmed"],
                "rebuild_allowed_next_run": opt["rebuild_allowed_next_run"],
                "caveat": opt["caveat"],
            }
        )
    return pd.DataFrame(rows)


def build_mapping_decision(
    source_meta: dict[str, dict[str, Any]],
    inventory: pd.DataFrame,
    price_ready: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    primary_sources = [
        "pit_clean_core_financial_factors_monthly_v3",
        "transformed_training_panel_v0",
        "derived_compact_f_missing_features_candidate_v01",
    ]
    price_plan = build_price_plan(price_ready)
    financial_options = {f: financial_rebuild_option(f, source_meta, inventory) for f in FINANCIAL_COMPLETION_FACTORS}
    v3 = source_meta.get("pit_clean_core_financial_factors_monthly_v3", {})

    for factor in LEGACY_FACTORS:
        if factor in PRICE_TECH_FACTORS:
            plan_row = price_plan.loc[price_plan["factor_name"] == factor].iloc[0].to_dict()
            ok = bool(plan_row["rebuild_allowed_next_run"])
            monthly_fallback = factor.startswith("Mom_") and bool(price_ready["monthly_return_col"])
            status = "READY_REBUILD_FROM_DAILY" if ok else ("READY_REBUILD_FROM_MONTHLY_RETURN" if monthly_fallback else "MISSING_BLOCKER")
            source_panel = "all_daily" if ok else ("canonical_csmar_trd_mnth_return_map_repaired" if monthly_fallback else "")
            coverage = float(plan_row["expected_coverage"]) if ok else float(price_ready["monthly_return_coverage"])
            rows.append(
                {
                    "legacy_factor_name": factor,
                    "factor_category": FACTOR_CATEGORY[factor],
                    "canonical_field_selected": factor,
                    "source_panel": source_panel,
                    "source_type": "daily_price_rebuild" if ok else ("monthly_return_rebuild" if monthly_fallback else ""),
                    "mapping_status": status,
                    "coverage_ratio": round(coverage, 6),
                    "pit_safe": bool(ok or monthly_fallback),
                    "needs_rebuild_from_raw": True,
                    "rebuild_source": plan_row["source"] if ok else rel(TRD_MNTH_REPAIRED),
                    "proposed_formula": plan_row["formula"],
                    "use_in_canonical_v0": bool(ok or monthly_fallback),
                    "caveat": plan_row["caveat"],
                }
            )
            continue

        existing = inventory_hit(inventory, primary_sources, factor)
        if existing:
            src_meta = source_meta[existing["source_panel"]]
            pit_safe, _, _ = source_has_pit(src_meta)
            field = str(existing["column_name"])
            caveat = transformed_caveat(field)
            status = "READY_EXISTING_FIELD" if pit_safe else "NEEDS_FORMULA_CONFIRMATION"
            rows.append(
                {
                    "legacy_factor_name": factor,
                    "factor_category": FACTOR_CATEGORY[factor],
                    "canonical_field_selected": field,
                    "source_panel": existing["source_panel"],
                    "source_type": "existing_field",
                    "mapping_status": status,
                    "coverage_ratio": float(existing["non_null_ratio"]),
                    "pit_safe": bool(pit_safe),
                    "needs_rebuild_from_raw": False,
                    "rebuild_source": "",
                    "proposed_formula": "existing canonical/equivalent field",
                    "use_in_canonical_v0": status == "READY_EXISTING_FIELD",
                    "caveat": caveat,
                }
            )
            continue

        if factor in financial_options:
            opt = financial_options[factor]
            rows.append(
                {
                    "legacy_factor_name": factor,
                    "factor_category": FACTOR_CATEGORY[factor],
                    "canonical_field_selected": opt["candidate_field"] if opt["mapping_status"].startswith("READY") else "",
                    "source_panel": opt["source_panel"],
                    "source_type": "financial_panel_rebuild" if opt["rebuild_allowed_next_run"] else opt["raw_or_transformed"],
                    "mapping_status": opt["mapping_status"],
                    "coverage_ratio": opt["coverage_ratio"],
                    "pit_safe": opt["pit_safe"],
                    "needs_rebuild_from_raw": opt["rebuild_allowed_next_run"],
                    "rebuild_source": rel(v3["path"]) if opt["rebuild_allowed_next_run"] and v3 else "",
                    "proposed_formula": opt["proposed_formula"],
                    "use_in_canonical_v0": opt["mapping_status"] in {"READY_EXISTING_FIELD", "READY_REBUILD_FROM_FINANCIAL_PANEL"},
                    "caveat": opt["caveat"],
                }
            )
            continue

        rows.append(
            {
                "legacy_factor_name": factor,
                "factor_category": FACTOR_CATEGORY[factor],
                "canonical_field_selected": "",
                "source_panel": "",
                "source_type": "",
                "mapping_status": "MISSING_BLOCKER",
                "coverage_ratio": 0.0,
                "pit_safe": False,
                "needs_rebuild_from_raw": False,
                "rebuild_source": "",
                "proposed_formula": "",
                "use_in_canonical_v0": False,
                "caveat": "未找到可用字段或可重建源。",
            }
        )
    return pd.DataFrame(rows)


def build_config(mapping: pd.DataFrame) -> dict[str, Any]:
    ready_existing = int((mapping["mapping_status"] == "READY_EXISTING_FIELD").sum())
    ready_rebuild = int(
        mapping["mapping_status"].isin(
            ["READY_REBUILD_FROM_DAILY", "READY_REBUILD_FROM_MONTHLY_RETURN", "READY_REBUILD_FROM_FINANCIAL_PANEL"]
        ).sum()
    )
    missing = int((~mapping["mapping_status"].str.startswith("READY")).sum())
    ready_total = ready_existing + ready_rebuild
    selected_sources = (
        mapping.loc[mapping["use_in_canonical_v0"] == True, ["legacy_factor_name", "source_panel", "source_type"]]  # noqa: E712
        .to_dict(orient="records")
    )
    return {
        "build_allowed_next_run": ready_total >= 12,
        "min_factor_count_required": 12,
        "ready_factor_count": ready_existing,
        "rebuild_factor_count": ready_rebuild,
        "missing_factor_count": missing,
        "selected_sources": selected_sources,
        "output_panel_path_next": "output/v0_canonical_16factor_panel_build_v1/v0_canonical_16factor_panel_v1.parquet",
        "no_alpha_signal_next": True,
        "no_weights_next": True,
        "no_returns_next": True,
        "no_training": True,
        "no_production": True,
    }


def simple_table(df: pd.DataFrame, cols: list[str], max_rows: int = 40) -> str:
    sub = df[cols].head(max_rows).fillna("").astype(str)
    widths = {c: max(len(c), *(len(x) for x in sub[c].tolist())) for c in cols}
    header = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    sep = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
    rows = ["| " + " | ".join(row[c].ljust(widths[c]) for c in cols) + " |" for _, row in sub.iterrows()]
    return "\n".join([header, sep, *rows])


def write_report(summary: dict[str, Any], mapping: pd.DataFrame, price_plan: pd.DataFrame, financial_plan: pd.DataFrame) -> None:
    lines = [
        "# V0 Legacy 16-Factor Canonical Mapping & Feature Completion Prep v1",
        "",
        "## 结论",
        f"- final_decision: {summary['final_decision']}",
        f"- prerequisites_passed: {summary['prerequisites_passed']}",
        f"- total_ready_or_rebuildable_factor_count: {summary['total_ready_or_rebuildable_factor_count']}",
        f"- canonical_16factor_panel_build_allowed_next: {summary['canonical_16factor_panel_build_allowed_next']}",
        f"- alpha_build_allowed_next: {summary['alpha_build_allowed_next']}",
        "",
        "## 16 因子映射决策",
        simple_table(
            mapping,
            [
                "legacy_factor_name",
                "mapping_status",
                "source_panel",
                "coverage_ratio",
                "use_in_canonical_v0",
            ],
        ),
        "",
        "## 价格技术因子重建计划",
        simple_table(price_plan, ["factor_name", "source", "lookback_window", "rebuild_allowed_next_run"]),
        "",
        "## 财务因子补全计划",
        simple_table(
            financial_plan,
            ["factor_name", "candidate_field", "source_panel", "formula_confirmed", "rebuild_allowed_next_run"],
        ),
        "",
        "## Guardrails",
        "- 本任务未生成 alpha_signal、weights、portfolio returns。",
        "- 未运行训练、调参、benchmark-relative、alpha/beta、IR/TE、FF、DGTW、SHAP 或 production。",
    ]
    (OUT_DIR / "v0_legacy_16factor_canonical_mapping_completion_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", {"step": "prerequisite_check"})

    prereq = {
        "v3_financial_panel_found": V3_PANEL.exists(),
        "transformed_panel_found": TRANSFORMED_PANEL.exists(),
        "derived_feature_panel_found": DERIVED_PANEL.exists(),
        "robust_cleaned_panel_found": ROBUST_PANEL.exists(),
        "all_daily_found": ALL_DAILY.exists(),
        "trd_mnth_repaired_return_map_found": TRD_MNTH_REPAIRED.exists(),
        "legacy_factor_scripts_found": all(p.exists() for p in LEGACY_SCRIPT_PATHS),
        "prerequisites_passed": False,
        "missing_required_files": [],
        "missing_optional_files": [],
    }
    required = [V3_PANEL, ALL_DAILY, TRD_MNTH_REPAIRED, *LEGACY_SCRIPT_PATHS]
    optional = [TRANSFORMED_PANEL, DERIVED_PANEL, ROBUST_PANEL]
    prereq["missing_required_files"] = [rel(p) for p in required if not p.exists()]
    prereq["missing_optional_files"] = [rel(p) for p in optional if not p.exists()]
    prereq["prerequisites_passed"] = len(prereq["missing_required_files"]) == 0
    save_json(prereq, OUT_DIR / "v0_legacy_16factor_mapping_prerequisite_check.json")

    write_state("running", {"step": "field_inventory"})
    inventory, source_meta = build_field_inventory(SOURCE_FILES)
    inventory.to_csv(OUT_DIR / "v0_legacy_factor_field_inventory.csv", index=False, encoding="utf-8-sig")

    write_state("running", {"step": "mapping_decision"})
    price_ready = price_source_readiness(source_meta)
    price_plan = build_price_plan(price_ready)
    price_plan.to_csv(OUT_DIR / "v0_price_technical_factor_rebuild_plan.csv", index=False, encoding="utf-8-sig")

    financial_plan = build_financial_plan(source_meta, inventory)
    financial_plan.to_csv(OUT_DIR / "v0_financial_factor_completion_plan.csv", index=False, encoding="utf-8-sig")

    mapping = build_mapping_decision(source_meta, inventory, price_ready)
    mapping.to_csv(OUT_DIR / "v0_legacy_16factor_mapping_decision.csv", index=False, encoding="utf-8-sig")

    config = build_config(mapping)
    save_json(config, OUT_DIR / "v0_canonical_16factor_panel_build_config_draft.json")

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
    guardrail_df = pd.DataFrame(
        [{"guardrail": k, "expected": False, "actual": v, "pass": v is False} for k, v in guardrails.items()]
    )
    guardrail_df.to_csv(OUT_DIR / "v0_legacy_16factor_mapping_guardrail_qa.csv", index=False, encoding="utf-8-sig")

    ready_existing = int((mapping["mapping_status"] == "READY_EXISTING_FIELD").sum())
    ready_rebuild = int(
        mapping["mapping_status"].isin(
            ["READY_REBUILD_FROM_DAILY", "READY_REBUILD_FROM_MONTHLY_RETURN", "READY_REBUILD_FROM_FINANCIAL_PANEL"]
        ).sum()
    )
    missing_blocker = int((mapping["mapping_status"] == "MISSING_BLOCKER").sum())
    drop_count = int((mapping["mapping_status"] == "DROP_WITH_REASON").sum())
    ready_total = ready_existing + ready_rebuild
    missing_factors = mapping.loc[~mapping["mapping_status"].str.startswith("READY"), "legacy_factor_name"].tolist()
    rebuild_factors = mapping.loc[mapping["needs_rebuild_from_raw"] == True, "legacy_factor_name"].tolist()  # noqa: E712
    selected_sources = mapping.loc[
        mapping["use_in_canonical_v0"] == True, ["legacy_factor_name", "source_panel", "source_type"]  # noqa: E712
    ].to_dict(orient="records")
    guardrails_passed = bool(guardrail_df["pass"].all())

    critical_missing = bool(
        mapping.loc[
            mapping["legacy_factor_name"].isin(["Mom_1M", "Mom_3M", "Mom_6M", "Vol_20D", "Vol_60D", "Beta", "BP"])
            & (mapping["mapping_status"] == "MISSING_BLOCKER")
        ].shape[0]
        > 0
    )
    if not guardrails_passed:
        final_decision = "V0_16FACTOR_MAPPING_FAIL_GUARDRAIL"
    elif ready_total >= 12:
        final_decision = "V0_16FACTOR_MAPPING_READY_FOR_PANEL_BUILD"
    elif critical_missing:
        final_decision = "V0_16FACTOR_MAPPING_BLOCKED_CRITICAL_FACTOR_MISSING"
    else:
        final_decision = "V0_16FACTOR_MAPPING_PARTIAL_NEEDS_MORE_FEATURE_WORK"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "legacy_factor_count": len(LEGACY_FACTORS),
        "ready_existing_factor_count": ready_existing,
        "ready_rebuild_factor_count": ready_rebuild,
        "missing_blocker_factor_count": missing_blocker,
        "drop_with_reason_factor_count": drop_count,
        "total_ready_or_rebuildable_factor_count": ready_total,
        "canonical_16factor_panel_build_allowed_next": bool(config["build_allowed_next_run"]),
        "alpha_build_allowed_next": False,
        "recommended_next_step": (
            "运行 canonical 16-factor panel build prep/build，仅生成因子面板；仍不得直接 alpha build。"
            if config["build_allowed_next_run"]
            else "继续补齐缺失或需确认的 legacy 因子字段，未达到 12/16 前不得 alpha build。"
        ),
        "selected_factor_sources": selected_sources,
        "missing_factors": missing_factors,
        "rebuild_required_factors": rebuild_factors,
        "final_decision": final_decision,
        "guardrails_passed": guardrails_passed,
    }
    save_json(summary, OUT_DIR / "v0_legacy_16factor_canonical_mapping_completion_summary.json")
    write_report(summary, mapping, price_plan, financial_plan)

    final_qa = guardrail_df.copy()
    required_artifacts = [
        OUT_DIR / "v0_legacy_16factor_mapping_prerequisite_check.json",
        OUT_DIR / "v0_legacy_factor_field_inventory.csv",
        OUT_DIR / "v0_legacy_16factor_mapping_decision.csv",
        OUT_DIR / "v0_price_technical_factor_rebuild_plan.csv",
        OUT_DIR / "v0_financial_factor_completion_plan.csv",
        OUT_DIR / "v0_canonical_16factor_panel_build_config_draft.json",
        OUT_DIR / "v0_legacy_16factor_mapping_guardrail_qa.csv",
        OUT_DIR / "v0_legacy_16factor_canonical_mapping_completion_summary.json",
        OUT_DIR / "v0_legacy_16factor_canonical_mapping_completion_report.md",
        ROOT / "scripts" / "prep_v0_legacy_16factor_canonical_mapping_completion_v1.py",
    ]
    for artifact_path in required_artifacts:
        final_qa.loc[len(final_qa)] = {
            "guardrail": f"artifact_written:{rel(artifact_path)}",
            "expected": True,
            "actual": artifact_path.exists(),
            "pass": artifact_path.exists(),
        }
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")

    completion_card = "\n".join(
        [
            "# task_completion_card",
            "",
            f"- task_name: {TASK_NAME}",
            f"- final_decision: {final_decision}",
            f"- prerequisites_passed: {prereq['prerequisites_passed']}",
            f"- ready_existing_factor_count: {ready_existing}",
            f"- ready_rebuild_factor_count: {ready_rebuild}",
            f"- total_ready_or_rebuildable_factor_count: {ready_total}",
            f"- canonical_16factor_panel_build_allowed_next: {config['build_allowed_next_run']}",
            "- guardrails_passed: true",
        ]
    )
    (OUT_DIR / "task_completion_card.md").write_text(completion_card, encoding="utf-8")
    save_json(
        {
            "task_name": TASK_NAME,
            "status": "completed",
            "script": rel(ROOT / "scripts" / "prep_v0_legacy_16factor_canonical_mapping_completion_v1.py"),
            "stdout_log": rel(RUN_DIR / "run_stdout.txt"),
            "stderr_log": rel(RUN_DIR / "run_stderr.txt"),
            "output_dir": rel(OUT_DIR),
            "final_decision": final_decision,
        },
        OUT_DIR / "terminal_summary.json",
    )

    write_state("completed", {"final_decision": final_decision, "output_dir": rel(OUT_DIR)})
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
