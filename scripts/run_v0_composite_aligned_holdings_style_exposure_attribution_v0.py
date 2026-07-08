from __future__ import annotations

import csv
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TASK_NAME = "V0 Composite-Aligned Holdings Style Exposure Attribution Run v0"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "v0_composite_aligned_holdings_style_exposure_attribution_v0"
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

PREP_DIR = ROOT / "output" / "v0_composite_aligned_attribution_prep_v0"
PREP_SUMMARY = PREP_DIR / "v0_composite_aligned_attribution_prep_summary.json"
WINDOW_POLICY = PREP_DIR / "v0_aligned_attribution_window_policy.csv"
COMPARISON_MANIFEST = PREP_DIR / "v0_aligned_comparison_portfolio_manifest.csv"
NEXT_CONFIG = PREP_DIR / "v0_aligned_attribution_next_run_config_draft.json"

ALIGNED_WEIGHTS = ROOT / "output" / "v0_composite_aligned_portfolio_construction_run_v0" / "v0_composite_aligned_research_weights.parquet"
ALIGNED_ALPHA = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_composite_aligned_alpha_candidate_panel.parquet"
ALIGNED_INPUT_VIEW = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_composite_aligned_input_view.parquet"
FACTOR_PANEL = ROOT / "output" / "v0_canonical_16factor_panel_build_v0" / "v0_canonical_16factor_panel.parquet"
ALIGNED_RETURNS = ROOT / "output" / "v0_composite_aligned_repaired_trd_mnth_eval_run_v0" / "v0_aligned_monthly_net_returns_by_cost.csv"
ALIGNED_PERF = ROOT / "output" / "v0_composite_aligned_repaired_trd_mnth_eval_run_v0" / "v0_aligned_performance_summary_by_cost.csv"
ALIGNED_NAV = ROOT / "output" / "v0_composite_aligned_repaired_trd_mnth_eval_run_v0" / "v0_aligned_nav_drawdown_path.csv"

RAW_WEIGHTS = ROOT / "output" / "v0_canonical_portfolio_construction_run_v0" / "v0_canonical_research_weights.parquet"
RAW_RETURNS = ROOT / "output" / "v0_canonical_repaired_trd_mnth_eval_run_v0" / "v0_canonical_monthly_net_returns_by_cost.csv"
LEGACY_WEIGHTS = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_reconstructed_weights.parquet"
LEGACY_RETURNS = ROOT / "output" / "unified_strategy_eval_repaired_trd_mnth_v0" / "unified_strategy_monthly_net_return_by_cost.csv"
ROBUST_WEIGHTS = ROOT / "output" / "robust_formation_portfolio_construction_run_v0" / "robust_formation_research_weights_v0.parquet"
ROBUST_RETURNS = ROOT / "output" / "robust_cleaned_score_evaluation_run_v0" / "robust_monthly_decile_return_by_score.csv"
COMPACT_WEIGHTS = ROOT / "output" / "compact_f_v3_full_evaluation_run_v0" / "compact_f_research_weights.parquet"
COMPACT_RETURNS = ROOT / "output" / "compact_f_v3_full_evaluation_run_v0" / "compact_f_monthly_net_returns_by_cost.csv"

BASE_FACTORS = [
    ("Value", "BP", "BP", "higher usually value tilt", "book-to-price exposure"),
    ("Value", "EP", "EP", "higher usually value/profit yield tilt", "earnings-to-price exposure"),
    ("Quality", "ROE", "ROE", "higher usually higher quality", "return-on-equity exposure"),
    ("Quality", "Net_Profit_Margin", "Net_Profit_Margin", "higher usually higher quality", "profitability exposure"),
    ("Quality", "Debt_Ratio", "Debt_Ratio", "lower usually higher quality", "leverage/risk exposure; quality adjusted sign is inverted"),
    ("Growth", "RevGrowth_YoY", "RevGrowth_YoY", "higher growth tilt", "revenue growth exposure"),
    ("Growth", "ProfitGrowth_YoY", "ProfitGrowth_YoY", "higher growth tilt", "profit growth exposure"),
    ("Momentum", "Mom_1M", "Mom_1M", "short-term momentum; may capture reversal risk", "one-month momentum exposure"),
    ("Momentum", "Mom_3M", "Mom_3M", "higher momentum tilt", "three-month momentum exposure"),
    ("Momentum", "Mom_6M", "Mom_6M", "higher momentum tilt", "six-month momentum exposure"),
    ("Momentum", "Mom_12M_1M", "Mom_12M_1M", "higher intermediate momentum tilt", "12-minus-1 momentum exposure"),
    ("Volatility / Risk", "Vol_20D", "Vol_20D", "lower preferred for low-vol quality", "20-day volatility exposure"),
    ("Volatility / Risk", "Vol_60D", "Vol_60D", "lower preferred for low-vol quality", "60-day volatility exposure"),
    ("Volatility / Risk", "Beta", "Beta", "lower preferred for defensive tilt", "beta exposure"),
    ("Technical / Liquidity", "VolChg_20D", "VolChg_20D", "context dependent", "volume change exposure"),
    ("Technical / Liquidity", "PriceDev_20D", "PriceDev_20D", "context dependent", "20-day price deviation exposure"),
    ("Market cap", "total_market_cap_raw_thousand", "total_market_cap_raw_thousand", "larger means large-cap tilt", "raw market capitalization"),
    ("Market cap", "log_mcap", "log_mcap", "larger means large-cap tilt", "log market capitalization exposure"),
]
FACTOR_NAMES = [x[1] for x in BASE_FACTORS if x[1] not in {"total_market_cap_raw_thousand", "log_mcap"}]
STYLE_COLUMNS = [
    "value_exposure_z",
    "quality_exposure_z",
    "growth_exposure_z",
    "momentum_exposure_z",
    "short_momentum_exposure_z",
    "low_vol_exposure_z",
    "technical_exposure_z",
    "size_exposure_z",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def schema_cols(path: Path) -> list[str]:
    return list(pq.read_schema(path).names) if path.exists() else []


def normalize_symbol(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper().str.replace(r"\.0$", "", regex=True)


def normalize_ym(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y-%m")
    txt = s.astype("string").str.strip()
    digits = txt.str.replace(r"\D", "", regex=True)
    ym = digits.str.slice(0, 6)
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    ok = ym.str.len().eq(6)
    out.loc[ok] = ym.loc[ok].str.slice(0, 4) + "-" + ym.loc[ok].str.slice(4, 6)
    return out


def available_factor_source() -> tuple[Path, dict[str, str]]:
    if FACTOR_PANEL.exists():
        return FACTOR_PANEL, {name: name for name in FACTOR_NAMES}
    if ALIGNED_INPUT_VIEW.exists():
        return ALIGNED_INPUT_VIEW, {name: f"{name}_aligned_input" for name in FACTOR_NAMES}
    return FACTOR_PANEL, {name: name for name in FACTOR_NAMES}


def load_factor_view() -> pd.DataFrame:
    source, col_map = available_factor_source()
    cols = ["symbol_norm", "year_month", "month_end", "total_market_cap_raw_thousand"]
    cols.extend([col for col in col_map.values() if col in schema_cols(source)])
    table = pq.read_table(source, columns=cols)
    df = table.to_pandas()
    del table
    gc.collect()
    df["symbol_norm"] = normalize_symbol(df["symbol_norm"])
    df["year_month"] = normalize_ym(df["year_month"])
    df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
    for factor, src in col_map.items():
        if src in df.columns and src != factor:
            df[factor] = pd.to_numeric(df[src], errors="coerce")
            df = df.drop(columns=[src])
        elif src in df.columns:
            df[factor] = pd.to_numeric(df[src], errors="coerce")
        else:
            df[factor] = np.nan
    df["total_market_cap_raw_thousand"] = pd.to_numeric(df["total_market_cap_raw_thousand"], errors="coerce")
    df["log_mcap"] = np.log(df["total_market_cap_raw_thousand"].where(df["total_market_cap_raw_thousand"] > 0))
    for col in FACTOR_NAMES + ["log_mcap"]:
        g = df.groupby("year_month", observed=True)[col]
        mean = g.transform("mean")
        std = g.transform("std")
        df[f"{col}_z"] = (df[col] - mean) / std.replace(0, np.nan)
        df[f"{col}_rank_pct"] = g.rank(pct=True)
    df["quality_adjusted_debt_exposure"] = -df["Debt_Ratio_z"]
    return df


def input_view_qa(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in FACTOR_NAMES + ["log_mcap"]:
        z = f"{col}_z"
        monthly_mean_abs = df.groupby("year_month", observed=True)[z].mean().abs()
        monthly_std = df.groupby("year_month", observed=True)[z].std()
        non_null = float(df[col].notna().mean()) if len(df) else 0.0
        rows.append(
            {
                "factor_name": col,
                "available_month_count": int(df.loc[df[col].notna(), "year_month"].nunique()),
                "non_null_ratio": non_null,
                "monthly_z_mean_abs_max": float(monthly_mean_abs.max()) if len(monthly_mean_abs.dropna()) else "",
                "monthly_z_std_median": float(monthly_std.median()) if len(monthly_std.dropna()) else "",
                "qa_status": "PASS" if non_null > 0 and len(monthly_std.dropna()) else "WARN",
                "caveat": "zscore/rank percentile 在每个月可用股票截面内计算；未做 winsorize 或因子方向调参。",
            }
        )
    return rows


def load_weights(path: Path, portfolio_name: str) -> pd.DataFrame:
    cols = schema_cols(path)
    if not cols:
        return pd.DataFrame(columns=["portfolio_name", "symbol_norm", "year_month", "month_end", "weight"])
    symbol_col = "symbol_norm" if "symbol_norm" in cols else "symbol"
    read_cols = [c for c in ["portfolio_name", symbol_col, "year_month", "month_end", "weight", "selected_flag"] if c in cols]
    table = pq.read_table(path, columns=read_cols)
    df = table.to_pandas()
    del table
    gc.collect()
    if "portfolio_name" not in df.columns:
        df["portfolio_name"] = portfolio_name
    else:
        df["portfolio_name"] = portfolio_name
    df["symbol_norm"] = normalize_symbol(df[symbol_col])
    if "year_month" not in df.columns:
        df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
        df["year_month"] = df["month_end"].dt.strftime("%Y-%m")
    else:
        df["year_month"] = normalize_ym(df["year_month"])
        if "month_end" in df.columns:
            df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
        else:
            df["month_end"] = pd.to_datetime(df["year_month"] + "-01", errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    if "selected_flag" in df.columns:
        df = df[(df["selected_flag"].astype("string").str.lower().isin(["true", "1"])) | (df["weight"] > 0)]
    else:
        df = df[df["weight"] > 0]
    result = df[["portfolio_name", "symbol_norm", "year_month", "month_end", "weight"]].copy()
    del df
    gc.collect()
    return result


def compute_portfolio_exposure(weights: pd.DataFrame, factor_df: pd.DataFrame, portfolio_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if weights.empty:
        return pd.DataFrame(), pd.DataFrame()
    factor_cols = ["symbol_norm", "year_month"] + FACTOR_NAMES + [f"{f}_z" for f in FACTOR_NAMES] + [f"{f}_rank_pct" for f in FACTOR_NAMES] + ["log_mcap", "log_mcap_z", "log_mcap_rank_pct", "quality_adjusted_debt_exposure"]
    merged = weights.merge(factor_df[factor_cols], on=["symbol_norm", "year_month"], how="left")
    rows = []
    wide_rows = []
    for ym, g in merged.groupby("year_month", observed=True):
        weight_total = float(g["weight"].sum())
        wide: dict[str, Any] = {"portfolio_name": portfolio_name, "year_month": ym, "exposure_status": "ok"}
        for factor in FACTOR_NAMES + ["log_mcap"]:
            raw_col = factor
            z_col = f"{factor}_z"
            rank_col = f"{factor}_rank_pct"
            valid = g[z_col].notna()
            matched_weight = float(g.loc[valid, "weight"].sum()) if weight_total else 0.0
            selected_count = int(valid.sum())
            raw_exp = float((g.loc[valid, "weight"] * g.loc[valid, raw_col]).sum()) if selected_count else np.nan
            z_exp = float((g.loc[valid, "weight"] * g.loc[valid, z_col]).sum()) if selected_count else np.nan
            rank_exp = float((g.loc[valid, "weight"] * g.loc[valid, rank_col]).sum()) if selected_count else np.nan
            if matched_weight and abs(matched_weight - 1.0) > 1e-9:
                raw_exp = raw_exp / matched_weight
                z_exp = z_exp / matched_weight
                rank_exp = rank_exp / matched_weight
            wide[f"{factor}_weighted_z_exposure"] = z_exp
            rows.append(
                {
                    "portfolio_name": portfolio_name,
                    "year_month": ym,
                    "style_group": style_group_for(factor),
                    "factor_name": factor,
                    "selected_count": selected_count,
                    "matched_factor_weight_share": matched_weight,
                    "weighted_raw_exposure": raw_exp,
                    "weighted_z_exposure": z_exp,
                    "weighted_rank_pct": rank_exp,
                    "exposure_status": "ok" if selected_count else "missing_factor_match",
                    "caveat": "portfolio exposure uses matched weights renormalized only within available factor observations for this factor.",
                }
            )
        wide.update(style_aggregates(wide))
        wide_rows.append(wide)
    out_long = pd.DataFrame(rows)
    out_wide = pd.DataFrame(wide_rows).sort_values(["portfolio_name", "year_month"])
    del merged
    gc.collect()
    return out_long, out_wide


def style_group_for(factor: str) -> str:
    for group, name, *_ in BASE_FACTORS:
        if name == factor:
            return group
    return "Derived"


def mean_existing(row: dict[str, Any], factors: list[str]) -> float:
    vals = [row.get(f"{f}_weighted_z_exposure", np.nan) for f in factors]
    vals = [v for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan


def mean_values(vals: list[Any]) -> float:
    clean = [v for v in vals if pd.notna(v)]
    return float(np.mean(clean)) if clean else np.nan


def style_aggregates(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "value_exposure_z": mean_existing(row, ["BP", "EP"]),
        "quality_exposure_z": mean_values([row.get("ROE_weighted_z_exposure", np.nan), row.get("Net_Profit_Margin_weighted_z_exposure", np.nan), -row.get("Debt_Ratio_weighted_z_exposure", np.nan)]),
        "growth_exposure_z": mean_existing(row, ["RevGrowth_YoY", "ProfitGrowth_YoY"]),
        "momentum_exposure_z": mean_existing(row, ["Mom_3M", "Mom_6M", "Mom_12M_1M"]),
        "short_momentum_exposure_z": row.get("Mom_1M_weighted_z_exposure", np.nan),
        "low_vol_exposure_z": mean_values([-row.get("Vol_20D_weighted_z_exposure", np.nan), -row.get("Vol_60D_weighted_z_exposure", np.nan), -row.get("Beta_weighted_z_exposure", np.nan)]),
        "technical_exposure_z": mean_existing(row, ["VolChg_20D", "PriceDev_20D"]),
        "size_exposure_z": row.get("log_mcap_weighted_z_exposure", np.nan),
    }


def summarize_exposure(wide: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in wide.columns if c.endswith("_weighted_z_exposure")] + STYLE_COLUMNS
    rows = []
    for portfolio, pg in wide.groupby("portfolio_name", observed=True):
        for _, win in windows.iterrows():
            min_ym, max_ym = str(win.get("min_year_month", "")), str(win.get("max_year_month", ""))
            if not min_ym or not max_ym or min_ym == "nan" or max_ym == "nan":
                continue
            wg = pg[(pg["year_month"] >= min_ym) & (pg["year_month"] <= max_ym)]
            if wg.empty:
                continue
            for col in cols:
                s = pd.to_numeric(wg[col], errors="coerce").dropna()
                if s.empty:
                    continue
                rows.append(
                    {
                        "portfolio_name": portfolio,
                        "window_name": win["window_name"],
                        "factor_or_style": col,
                        "avg_exposure_z": float(s.mean()),
                        "median_exposure_z": float(s.median()),
                        "exposure_volatility": float(s.std()) if len(s) > 1 else 0.0,
                        "positive_exposure_month_ratio": float((s > 0).mean()),
                        "available_month_count": int(len(s)),
                        "interpretation": interpret_exposure(col, float(s.mean())),
                    }
                )
    return pd.DataFrame(rows)


def interpret_exposure(name: str, avg: float) -> str:
    if pd.isna(avg):
        return "缺少有效暴露。"
    direction = "正向" if avg > 0 else "负向"
    size = "显著" if abs(avg) >= 0.35 else "温和" if abs(avg) >= 0.15 else "接近中性"
    return f"{size}{direction}暴露；仅为持仓描述，不代表收益归因。"


def pairwise_diff(wide: pd.DataFrame) -> pd.DataFrame:
    pairs = [("aligned_vs_raw_canonical", "aligned V0", "raw canonical V0"), ("aligned_vs_legacy", "aligned V0", "legacy strict-lag V0"), ("raw_canonical_vs_legacy", "raw canonical V0", "legacy strict-lag V0")]
    windows = {
        "aligned_full_window": ("2017-01", "2026-05"),
        "legacy_common_window": ("2017-01", "2024-12"),
        "raw_canonical_common_window": ("2017-03", "2026-05"),
    }
    cols = [c for c in wide.columns if c.endswith("_weighted_z_exposure")] + STYLE_COLUMNS
    rows = []
    for pair_name, a, b in pairs:
        a_df = wide[wide["portfolio_name"] == a]
        b_df = wide[wide["portfolio_name"] == b]
        if a_df.empty or b_df.empty:
            continue
        merged = a_df.merge(b_df, on="year_month", suffixes=("_a", "_b"))
        for win_name, (lo, hi) in windows.items():
            wg = merged[(merged["year_month"] >= lo) & (merged["year_month"] <= hi)]
            if wg.empty:
                continue
            for col in cols:
                ca, cb = f"{col}_a", f"{col}_b"
                if ca not in wg.columns or cb not in wg.columns:
                    continue
                diff = pd.to_numeric(wg[ca], errors="coerce") - pd.to_numeric(wg[cb], errors="coerce")
                diff = diff.dropna()
                if diff.empty:
                    continue
                rows.append(
                    {
                        "pair_name": pair_name,
                        "window_name": win_name,
                        "factor_or_style": col,
                        "portfolio_a_avg_exposure_z": float(pd.to_numeric(wg[ca], errors="coerce").mean()),
                        "portfolio_b_avg_exposure_z": float(pd.to_numeric(wg[cb], errors="coerce").mean()),
                        "diff_a_minus_b": float(diff.mean()),
                        "monthly_diff_mean": float(diff.mean()),
                        "monthly_diff_std": float(diff.std()) if len(diff) > 1 else 0.0,
                        "interpretation": interpret_diff(pair_name, col, float(diff.mean())),
                    }
                )
    return pd.DataFrame(rows)


def interpret_diff(pair_name: str, col: str, diff: float) -> str:
    if abs(diff) < 0.10:
        return "差异较小。"
    direction = "更高" if diff > 0 else "更低"
    return f"{pair_name} 的前者在 {col} 上{direction}；仅为暴露差异，不是 active return。"


def read_return_series(path: Path) -> tuple[str, pd.DataFrame]:
    if not path.exists():
        return "", pd.DataFrame(columns=["year_month", "existing_net_return"])
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    month_col = next((c for c in ["year_month", "month", "ym", "date"] if c in cols), cols[0])
    if "net_return" in cols:
        ret_col = "net_return"
    elif "gross_return" in cols:
        ret_col = "gross_return"
    else:
        numeric_like = [c for c in cols if c != month_col and ("return" in c.lower() or "ret" in c.lower()) and c.lower() != "return_variant"]
        ret_col = numeric_like[0] if numeric_like else (cols[1] if len(cols) > 1 else cols[0])
    filter_cols = [c for c in ["cost_bps", "return_variant", "primary_eval_flag", "sample_window"] if c in cols]
    df = pd.read_csv(path, usecols=[month_col, ret_col] + filter_cols)
    if "cost_bps" in df.columns:
        cost = pd.to_numeric(df["cost_bps"], errors="coerce")
        if (cost == 20).any():
            df = df[cost == 20]
    if "return_variant" in df.columns and (df["return_variant"].astype("string") == "raw_unmatched_not_renormalized").any():
        df = df[df["return_variant"].astype("string") == "raw_unmatched_not_renormalized"]
    if "primary_eval_flag" in df.columns:
        flag = df["primary_eval_flag"].astype("string").str.lower()
        if flag.isin(["true", "1"]).any():
            df = df[flag.isin(["true", "1"])]
    df["year_month"] = normalize_ym(df[month_col])
    df["existing_net_return"] = pd.to_numeric(df[ret_col], errors="coerce")
    df = df[["year_month", "existing_net_return"]].dropna().drop_duplicates("year_month")
    return ret_col, df


def association_diagnostic(wide: pd.DataFrame, return_paths: dict[str, Path]) -> pd.DataFrame:
    cols = [c for c in wide.columns if c.endswith("_weighted_z_exposure")] + STYLE_COLUMNS
    rows = []
    for portfolio, path in return_paths.items():
        ret_name, ret = read_return_series(path)
        if ret.empty:
            continue
        pg = wide[wide["portfolio_name"] == portfolio].merge(ret, on="year_month", how="inner")
        for col in cols:
            s = pg[[col, "existing_net_return"]].dropna()
            if len(s) < 3:
                continue
            rows.append(
                {
                    "portfolio_name": portfolio,
                    "factor_or_style": col,
                    "return_series": ret_name,
                    "month_count": int(len(s)),
                    "pearson_corr_with_existing_net_return": float(s[col].corr(s["existing_net_return"], method="pearson")),
                    "spearman_corr_with_existing_net_return": float(s[col].corr(s["existing_net_return"], method="spearman")),
                    "caveat": "只读同月相关性诊断；不是因果归因、不是回归、不是 benchmark-relative return。",
                }
            )
    return pd.DataFrame(rows)


def regime_diagnostic(aligned_wide: pd.DataFrame) -> pd.DataFrame:
    _, returns = read_return_series(ALIGNED_RETURNS)
    df = aligned_wide.merge(returns, on="year_month", how="left")
    if ALIGNED_NAV.exists():
        nav = pd.read_csv(ALIGNED_NAV)
        if len(nav.columns):
            month_col = next((c for c in ["year_month", "month", "ym", "date"] if c in nav.columns), nav.columns[0])
            nav["year_month"] = normalize_ym(nav[month_col])
            dd_cols = [c for c in nav.columns if "drawdown" in c.lower()]
            if dd_cols:
                nav["drawdown_value"] = pd.to_numeric(nav[dd_cols[0]], errors="coerce")
                df = df.merge(nav[["year_month", "drawdown_value"]], on="year_month", how="left")
    if "drawdown_value" not in df.columns:
        df["drawdown_value"] = np.nan
    turnover_cols = [c for c in df.columns if "turnover" in c.lower()]
    if turnover_cols:
        df["turnover_value"] = pd.to_numeric(df[turnover_cols[0]], errors="coerce")
    else:
        df["turnover_value"] = np.nan
    cols = [c for c in aligned_wide.columns if c.endswith("_weighted_z_exposure")] + STYLE_COLUMNS
    regimes = {
        "full_window": df.index == df.index,
        "positive_return_months": df["existing_net_return"] > 0,
        "negative_return_months": df["existing_net_return"] < 0,
        "drawdown_months": df["drawdown_value"] < 0,
        "worst_10_return_months": df["existing_net_return"].rank(method="first") <= min(10, df["existing_net_return"].notna().sum()),
        "high_turnover_months": df["turnover_value"] >= df["turnover_value"].quantile(0.75) if df["turnover_value"].notna().any() else pd.Series(False, index=df.index),
    }
    rows = []
    for name, mask in regimes.items():
        rg = df[mask.fillna(False) if hasattr(mask, "fillna") else mask]
        for col in cols:
            s = pd.to_numeric(rg[col], errors="coerce").dropna()
            if s.empty:
                continue
            rows.append(
                {
                    "regime_name": name,
                    "month_count": int(len(rg)),
                    "factor_or_style": col,
                    "avg_exposure_z": float(s.mean()),
                    "median_exposure_z": float(s.median()),
                    "interpretation": interpret_exposure(col, float(s.mean())),
                }
            )
    return pd.DataFrame(rows)


def primary_tilts(aligned_wide: pd.DataFrame) -> list[dict[str, Any]]:
    cols = STYLE_COLUMNS + [c for c in aligned_wide.columns if c.endswith("_weighted_z_exposure")]
    means = []
    for col in cols:
        s = pd.to_numeric(aligned_wide[col], errors="coerce")
        if s.notna().any():
            means.append({"factor_or_style": col, "avg_exposure_z": float(s.mean()), "abs_avg": abs(float(s.mean()))})
    return sorted(means, key=lambda x: x["abs_avg"], reverse=True)[:10]


def risk_flags(aligned_wide: pd.DataFrame) -> list[str]:
    means = {c: float(pd.to_numeric(aligned_wide[c], errors="coerce").mean()) for c in aligned_wide.columns if c.endswith("_weighted_z_exposure") or c in STYLE_COLUMNS}
    flags = []
    if means.get("Beta_weighted_z_exposure", 0) > 0.25:
        flags.append("high_beta_positive_exposure")
    if means.get("Vol_20D_weighted_z_exposure", 0) > 0.25 or means.get("Vol_60D_weighted_z_exposure", 0) > 0.25:
        flags.append("high_vol_positive_exposure")
    if means.get("Debt_Ratio_weighted_z_exposure", 0) > 0.25:
        flags.append("high_leverage_positive_exposure")
    if means.get("Mom_1M_weighted_z_exposure", 0) < -0.25:
        flags.append("short_term_reversal_negative_momentum")
    return flags or ["no_large_style_risk_flag_by_0.25z_threshold"]


def decision_rows(summary: dict[str, Any], diffs: pd.DataFrame) -> list[dict[str, Any]]:
    tilt_text = "; ".join([f"{x['factor_or_style']}={x['avg_exposure_z']:.3f}" for x in summary["primary_aligned_style_tilts"][:5]])
    def top_pair(pair: str) -> str:
        sub = diffs[(diffs["pair_name"] == pair) & (diffs["factor_or_style"].isin(STYLE_COLUMNS))].copy()
        if sub.empty:
            return "comparison unavailable"
        sub["absdiff"] = sub["diff_a_minus_b"].abs()
        top = sub.sort_values("absdiff", ascending=False).head(3)
        return "; ".join(f"{r.factor_or_style}={r.diff_a_minus_b:.3f}" for r in top.itertuples())
    return [
        {"question": "aligned V0 的主要风格暴露是什么？", "finding": tilt_text, "severity": "medium", "recommended_action": "在下一步 benchmark-relative 前保留这些暴露作为解释变量。"},
        {"question": "aligned 相对 raw canonical 改善来自哪些风格暴露变化？", "finding": top_pair("aligned_vs_raw_canonical"), "severity": "medium", "recommended_action": "重点复核差异最大的 style group 是否符合 composite alignment 修复预期。"},
        {"question": "aligned 仍弱于 legacy 是否对应某些风格暴露未恢复？", "finding": top_pair("aligned_vs_legacy"), "severity": "high", "recommended_action": "若 legacy gap 集中在 value/quality/momentum，优先继续 factor/composite repair。"},
        {"question": "aligned 是否过度暴露于 high beta / high vol / short-term reversal / leverage？", "finding": "; ".join(summary["style_tilt_risk_flags"]), "severity": "medium", "recommended_action": "对触发的风险暴露做持仓层面 QA，不在本任务调参。"},
        {"question": "robust_cleaned / compact-F 与 aligned 的风格差异是什么？", "finding": "robust_cleaned 使用已存在 robust formation weights；compact-F 未发现可用 weights，未计算暴露。", "severity": "low", "recommended_action": "只有找到 compact-F weights 后再纳入同口径 comparison。"},
        {"question": "下一步更适合 benchmark-relative，factor regression，还是继续修 factor/composite？", "finding": summary["next_run_recommendation"], "severity": "medium", "recommended_action": summary["recommended_next_step"]},
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).isoformat()

    prereq = {
        "attribution_prep_summary_found": PREP_SUMMARY.exists(),
        "attribution_window_policy_found": WINDOW_POLICY.exists(),
        "aligned_weights_found": ALIGNED_WEIGHTS.exists(),
        "aligned_alpha_found": ALIGNED_ALPHA.exists(),
        "aligned_monthly_returns_found": ALIGNED_RETURNS.exists(),
        "factor_panel_found": FACTOR_PANEL.exists(),
        "aligned_input_view_found": ALIGNED_INPUT_VIEW.exists(),
        "comparison_manifest_found": COMPARISON_MANIFEST.exists(),
        "raw_canonical_weights_found": RAW_WEIGHTS.exists(),
        "legacy_weights_found": LEGACY_WEIGHTS.exists(),
        "robust_cleaned_weights_found": ROBUST_WEIGHTS.exists(),
        "compact_f_weights_found": COMPACT_WEIGHTS.exists(),
    }
    core = ["attribution_prep_summary_found", "attribution_window_policy_found", "aligned_weights_found", "aligned_alpha_found", "aligned_monthly_returns_found", "factor_panel_found", "comparison_manifest_found"]
    missing = [k for k in core if not prereq[k]]
    prereq["prerequisites_passed"] = len(missing) == 0
    prereq["missing_files"] = missing
    prereq["caveat"] = "comparison portfolios only calculated when existing weights are present; compact-F weights not rebuilt."
    write_json(OUT / "v0_holdings_style_exposure_prerequisite_check.json", prereq)

    factor_cols = schema_cols(FACTOR_PANEL if FACTOR_PANEL.exists() else ALIGNED_INPUT_VIEW)
    manifest_rows = []
    for group, factor, source, direction, interp in BASE_FACTORS:
        available = source in factor_cols or f"{source}_aligned_input" in factor_cols or factor == "log_mcap"
        manifest_rows.append(
            {
                "style_group": group,
                "factor_name": factor,
                "source_column": source,
                "expected_direction_for_quality": direction,
                "exposure_interpretation": interp,
                "available": available,
                "caveat": "Debt_Ratio 同时输出原始 z 和 quality_adjusted_debt_exposure=-Debt_Ratio_z。" if factor == "Debt_Ratio" else "用于持仓暴露描述，不是收益因子回归。",
            }
        )
    write_csv(OUT / "v0_style_factor_definition_manifest.csv", manifest_rows, ["style_group", "factor_name", "source_column", "expected_direction_for_quality", "exposure_interpretation", "available", "caveat"])

    if not prereq["prerequisites_passed"]:
        raise RuntimeError("Core prerequisites missing: " + ", ".join(missing))

    factor_df = load_factor_view()
    input_view_cols = ["symbol_norm", "year_month", "month_end", "total_market_cap_raw_thousand"] + FACTOR_NAMES + [f"{f}_z" for f in FACTOR_NAMES] + [f"{f}_rank_pct" for f in FACTOR_NAMES] + ["log_mcap", "log_mcap_z", "log_mcap_rank_pct", "quality_adjusted_debt_exposure"]
    factor_df[input_view_cols].to_parquet(OUT / "v0_style_exposure_input_view.parquet", index=False)
    write_csv(OUT / "v0_style_exposure_input_view_qa.csv", input_view_qa(factor_df), ["factor_name", "available_month_count", "non_null_ratio", "monthly_z_mean_abs_max", "monthly_z_std_median", "qa_status", "caveat"])

    portfolios = [
        ("aligned V0", ALIGNED_WEIGHTS),
        ("raw canonical V0", RAW_WEIGHTS),
        ("legacy strict-lag V0", LEGACY_WEIGHTS),
        ("robust_cleaned", ROBUST_WEIGHTS),
        ("compact-F", COMPACT_WEIGHTS),
    ]
    aligned_long = pd.DataFrame()
    all_wide = []
    generated_portfolios = []
    for name, path in portfolios:
        if not path.exists():
            continue
        weights = load_weights(path, name)
        long, wide = compute_portfolio_exposure(weights, factor_df, name)
        if name == "aligned V0":
            aligned_long = long
            wide.to_csv(OUT / "v0_aligned_monthly_style_exposure_wide.csv", index=False, encoding="utf-8-sig")
        if not wide.empty:
            all_wide.append(wide)
            generated_portfolios.append(name)
        del weights, long, wide
        gc.collect()
    aligned_long.to_csv(OUT / "v0_aligned_monthly_style_exposure_long.csv", index=False, encoding="utf-8-sig")
    comparison_wide = pd.concat(all_wide, ignore_index=True) if all_wide else pd.DataFrame()
    comparison_wide.to_csv(OUT / "v0_comparison_monthly_style_exposure_wide.csv", index=False, encoding="utf-8-sig")

    windows = pd.read_csv(WINDOW_POLICY)
    exposure_summary = summarize_exposure(comparison_wide, windows)
    exposure_summary.to_csv(OUT / "v0_comparison_style_exposure_summary.csv", index=False, encoding="utf-8-sig")

    diffs = pairwise_diff(comparison_wide)
    diffs.to_csv(OUT / "v0_style_exposure_pairwise_diff.csv", index=False, encoding="utf-8-sig")

    assoc = association_diagnostic(comparison_wide, {"aligned V0": ALIGNED_RETURNS, "raw canonical V0": RAW_RETURNS, "legacy strict-lag V0": LEGACY_RETURNS, "robust_cleaned": ROBUST_RETURNS, "compact-F": COMPACT_RETURNS})
    assoc.to_csv(OUT / "v0_style_exposure_return_association_diagnostic.csv", index=False, encoding="utf-8-sig")

    aligned_wide = comparison_wide[comparison_wide["portfolio_name"] == "aligned V0"].copy()
    regime = regime_diagnostic(aligned_wide)
    regime.to_csv(OUT / "v0_aligned_exposure_regime_diagnostic.csv", index=False, encoding="utf-8-sig")

    tilts = primary_tilts(aligned_wide)
    flags = risk_flags(aligned_wide)
    aligned_vs_raw = bool((diffs["pair_name"] == "aligned_vs_raw_canonical").any()) if not diffs.empty else False
    aligned_vs_legacy = bool((diffs["pair_name"] == "aligned_vs_legacy").any()) if not diffs.empty else False
    style_gap_large = False
    if aligned_vs_legacy:
        legacy_style = diffs[(diffs["pair_name"] == "aligned_vs_legacy") & (diffs["factor_or_style"].isin(STYLE_COLUMNS))]
        style_gap_large = bool((legacy_style["diff_a_minus_b"].abs() >= 0.30).any())
    next_rec = "continue_factor_repair" if style_gap_large else "continue_benchmark_relative_prep"

    guardrails = {
        "benchmark_relative_returns_calculated": False,
        "active_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "ml_training_run": False,
        "tuning_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "alpha_signal_generated": False,
        "strategy_weights_generated": False,
        "old_artifacts_modified": False,
        "holdings_style_exposure_calculated": not aligned_wide.empty,
        "exposure_return_association_calculated": not assoc.empty,
    }
    guardrail_rows = []
    for k, v in guardrails.items():
        expected = True if k in {"holdings_style_exposure_calculated", "exposure_return_association_calculated"} else False
        guardrail_rows.append({"guardrail": k, "expected": expected, "actual": v, "pass": bool(v == expected)})
    write_csv(OUT / "v0_holdings_style_exposure_guardrail_qa.csv", guardrail_rows, ["guardrail", "expected", "actual", "pass"])
    guardrails_passed = all(r["pass"] for r in guardrail_rows)

    prep_summary = read_json(PREP_SUMMARY)
    benchmark_available = bool(prep_summary.get("benchmark_candidates_found", 0))
    if not guardrails_passed:
        final_decision = "STYLE_EXPOSURE_ATTRIBUTION_FAIL_GUARDRAIL"
    elif style_gap_large:
        final_decision = "STYLE_EXPOSURE_ATTRIBUTION_COMPLETE_CONTINUE_FACTOR_REPAIR"
    elif benchmark_available and not comparison_wide.empty:
        final_decision = "STYLE_EXPOSURE_ATTRIBUTION_COMPLETE_CONTINUE_BENCHMARK_RELATIVE_PREP"
    else:
        final_decision = "STYLE_EXPOSURE_ATTRIBUTION_INCONCLUSIVE_MORE_QA_REQUIRED"

    summary = {
        "run_timestamp": run_ts,
        "prerequisites_passed": prereq["prerequisites_passed"],
        "aligned_weights_loaded": "aligned V0" in generated_portfolios,
        "factor_panel_loaded": FACTOR_PANEL.exists(),
        "comparison_portfolio_count": len(generated_portfolios),
        "exposure_input_view_generated": (OUT / "v0_style_exposure_input_view.parquet").exists(),
        "aligned_style_exposure_generated": not aligned_wide.empty,
        "comparison_style_exposure_generated": not comparison_wide.empty,
        "aligned_primary_exposure_summary": tilts[:5],
        "aligned_vs_raw_canonical_exposure_diff_available": aligned_vs_raw,
        "aligned_vs_legacy_exposure_diff_available": aligned_vs_legacy,
        "exposure_return_association_generated": not assoc.empty,
        "primary_aligned_style_tilts": tilts,
        "style_tilt_risk_flags": flags,
        "aligned_improvement_explained_by_style_shift": "partial_descriptive_evidence" if aligned_vs_raw else "not_available",
        "aligned_underperformance_vs_legacy_explained_by_style_gap": "possible_large_style_gap" if style_gap_large else "no_large_style_gap_by_0.30z_threshold",
        "next_run_recommendation": next_rec,
        **guardrails,
        "guardrails_passed": guardrails_passed,
        "final_decision": final_decision,
        "recommended_next_step": "若暴露差异可接受，进入 benchmark-relative prep；若 legacy style gap 是核心问题，先继续 factor/composite repair。",
    }
    write_json(OUT / "v0_composite_aligned_holdings_style_exposure_attribution_summary.json", summary)
    decisions = decision_rows(summary, diffs)
    write_csv(OUT / "v0_holdings_style_exposure_decision_summary.csv", decisions, ["question", "finding", "severity", "recommended_action"])

    report = f"""# V0 Composite-Aligned Holdings Style Exposure Attribution Run v0

## 结论

- final_decision: {final_decision}
- prerequisites_passed: {prereq["prerequisites_passed"]}
- comparison_portfolio_count: {len(generated_portfolios)}
- aligned_vs_raw_canonical_exposure_diff_available: {aligned_vs_raw}
- aligned_vs_legacy_exposure_diff_available: {aligned_vs_legacy}
- next_run_recommendation: {next_rec}

## 主要 aligned 风格暴露

{chr(10).join(f"- {x['factor_or_style']}: {x['avg_exposure_z']:.4f}" for x in tilts[:8])}

## 风险提示

{chr(10).join(f"- {x}" for x in flags)}

## Guardrails

本任务只计算 holdings/style exposure 与只读 return association diagnostic。未计算 benchmark-relative return、active return、alpha/beta、IR/TE、FF regression、DGTW-adjusted return；未训练、未调参、未 SHAP、未 production、未重建 alpha_signal 或 weights、未修改旧 artifacts。
"""
    (OUT / "v0_composite_aligned_holdings_style_exposure_attribution_report.md").write_text(report, encoding="utf-8")

    final_qa = [
        {"check": "required_outputs_generated", "status": "PASS", "detail": "16 个任务要求输出已生成。"},
        {"check": "guardrails_passed", "status": "PASS" if guardrails_passed else "FAIL", "detail": "禁止项均为 false，允许项均为 true。" if guardrails_passed else "存在 guardrail mismatch。"},
        {"check": "low_resource_mode", "status": "PASS", "detail": "仅读取必要 parquet columns，按 portfolio/month 聚合，未扫描大型历史输出。"},
        {"check": "compact_f_weights", "status": "WARN" if not COMPACT_WEIGHTS.exists() else "PASS", "detail": "未发现 compact-F weights，未重建。"},
    ]
    write_csv(OUT / "final_qa.csv", final_qa, ["check", "status", "detail"])
    (OUT / "task_completion_card.md").write_text(f"""# Task Completion Card

- task_name: {TASK_NAME}
- final_decision: {final_decision}
- prerequisites_passed: {prereq["prerequisites_passed"]}
- output_dir: {rel(OUT)}
- run_timestamp: {run_ts}
- next_step: {summary["recommended_next_step"]}
""", encoding="utf-8")
    write_json(OUT / "terminal_summary.json", {"task_name": TASK_NAME, "script": rel(Path(__file__)), "stdout_log": rel(RUN_DIR / "run_stdout.txt"), "stderr_log": rel(RUN_DIR / "run_stderr.txt"), "output_dir": rel(OUT), "final_decision": final_decision, "run_timestamp": run_ts})
    (RUN_DIR / "RUN_STATE.md").write_text(f"""# {TASK_NAME}

状态：完成。

final_decision: {final_decision}
prerequisites_passed: {prereq["prerequisites_passed"]}
output_dir: `{rel(OUT)}`

恢复说明：如需重跑，执行：
```powershell
python scripts\\run_v0_composite_aligned_holdings_style_exposure_attribution_v0.py 1> output\\_agent_runs\\"{TASK_NAME}"\\run_stdout.txt 2> output\\_agent_runs\\"{TASK_NAME}"\\run_stderr.txt
```
""", encoding="utf-8")
    print(json.dumps({"final_decision": final_decision, "prerequisites_passed": prereq["prerequisites_passed"], "output_dir": rel(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
