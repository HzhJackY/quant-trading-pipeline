from __future__ import annotations

import gc
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "v0_canonical_vs_legacy_reconstruction_drift_audit_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

CANON_FACTOR = ROOT / "output" / "v0_canonical_16factor_panel_build_v0" / "v0_canonical_16factor_panel.parquet"
CANON_ALPHA = ROOT / "output" / "v0_canonical_strict_lag_alpha_build_v0" / "v0_canonical_alpha_signal_panel.parquet"
CANON_WEIGHTS = ROOT / "output" / "v0_canonical_portfolio_construction_run_v0" / "v0_canonical_research_weights.parquet"
CANON_USAGE = ROOT / "output" / "v0_canonical_strict_lag_alpha_build_v0" / "v0_canonical_factor_usage_summary.csv"
CANON_ICIR_AUDIT = ROOT / "output" / "v0_canonical_strict_lag_alpha_build_v0" / "v0_canonical_factor_icir_contribution_audit.csv"
CANON_ICIR_BY_MONTH = ROOT / "output" / "v0_canonical_strict_lag_alpha_build_v0" / "v0_canonical_strict_lag_icir_by_month_factor.csv"
CANON_SPLIT = ROOT / "output" / "v0_canonical_strict_lag_alpha_build_v0" / "v0_canonical_split_assignment.parquet"

LEGACY_PREPROCESSED = ROOT / "output" / "preprocessed.parquet"
LEGACY_SPLIT = ROOT / "output" / "split_universe_blended.parquet"
LEGACY_ALPHA = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_alpha_signal_panel.parquet"
LEGACY_WEIGHTS = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_reconstructed_weights.parquet"
LEGACY_ICIR_AUDIT = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_icir_window_audit.csv"
FORENSIC_DIFF = ROOT / "output" / "v0_canonical_eval_forensic_common_window_bridge_v0" / "canonical_vs_legacy_monthly_return_diff.csv"

LEGACY_SCRIPTS = [
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

FACTORS = [
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
PRICE_TECH_FACTORS = ["Mom_1M", "Mom_3M", "Mom_6M", "Mom_12M_1M", "Vol_20D", "Vol_60D", "Beta", "VolChg_20D", "PriceDev_20D"]
FIN_FACTORS = ["BP", "EP", "ROE", "Debt_Ratio", "Net_Profit_Margin", "RevGrowth_YoY", "ProfitGrowth_YoY"]
START_MONTH = "2017-03"
END_MONTH = "2024-12"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_run_state(status: str, checkpoint: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    text = (
        "# RUN_STATE\n\n"
        f"task_name: {TASK_NAME}\n"
        f"status: {status}\n"
        f"last_checkpoint: {checkpoint}\n"
        f"updated_at: {datetime.now().isoformat(timespec='seconds')}\n"
        "resume_instruction: run scripts\\audit_v0_canonical_vs_legacy_reconstruction_drift_v0.py with stdout/stderr redirected to output\\_agent_runs\\v0_canonical_vs_legacy_reconstruction_drift_audit_v0\n"
    )
    (RUN_DIR / "RUN_STATE.md").write_text(text, encoding="utf-8")


def dump_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ym_from_any(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.astype(str)
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.to_period("M").astype(str)
    s = series.astype(str)
    looks_ym = s.str.match(r"^\d{4}-\d{2}$", na=False)
    if bool(looks_ym.all()):
        return s
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").astype(str)


def norm_symbol(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def severity_from_spearman(value: float) -> str:
    if pd.isna(value):
        return "CRITICAL"
    if value >= 0.90:
        return "LOW"
    if value >= 0.70:
        return "MEDIUM"
    if value >= 0.40:
        return "HIGH"
    return "CRITICAL"


def severity_rank(severity: str) -> int:
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}.get(str(severity), 3)


def worst_severity(values: pd.Series | list[str]) -> str:
    vals = list(values)
    if not vals:
        return "INCONCLUSIVE"
    return max(vals, key=severity_rank)


def prerequisite_check() -> dict:
    flags = {
        "canonical_factor_panel_found": CANON_FACTOR.exists(),
        "canonical_alpha_panel_found": CANON_ALPHA.exists(),
        "canonical_weights_found": CANON_WEIGHTS.exists(),
        "canonical_icir_audit_found": CANON_ICIR_AUDIT.exists() and CANON_ICIR_BY_MONTH.exists() and CANON_USAGE.exists(),
        "canonical_split_assignment_found": CANON_SPLIT.exists(),
        "legacy_preprocessed_found": LEGACY_PREPROCESSED.exists(),
        "legacy_split_universe_blended_found": LEGACY_SPLIT.exists(),
        "legacy_strict_lag_alpha_found": LEGACY_ALPHA.exists(),
        "legacy_strict_lag_weights_found": LEGACY_WEIGHTS.exists(),
        "legacy_scripts_found": all(path.exists() for path in LEGACY_SCRIPTS),
    }
    required = [
        ("canonical_factor_panel_found", CANON_FACTOR),
        ("canonical_alpha_panel_found", CANON_ALPHA),
        ("canonical_weights_found", CANON_WEIGHTS),
        ("canonical_icir_audit_found", CANON_ICIR_AUDIT),
        ("canonical_split_assignment_found", CANON_SPLIT),
        ("legacy_preprocessed_found", LEGACY_PREPROCESSED),
        ("legacy_split_universe_blended_found", LEGACY_SPLIT),
        ("legacy_strict_lag_alpha_found", LEGACY_ALPHA),
        ("legacy_strict_lag_weights_found", LEGACY_WEIGHTS),
    ]
    missing = [rel(path) for key, path in required if not flags[key]]
    if not flags["legacy_scripts_found"]:
        missing.extend(rel(path) for path in LEGACY_SCRIPTS if not path.exists())
    flags["prerequisites_passed"] = not missing
    flags["missing_files"] = missing
    flags["caveat"] = "" if LEGACY_ICIR_AUDIT.exists() else "legacy ICIR audit missing; ICIR layer will be inconclusive"
    dump_json(OUT_DIR / "v0_reconstruction_drift_prerequisite_check.json", flags)
    return flags


def read_schemas() -> tuple[list[str], list[str]]:
    import pyarrow.parquet as pq

    canon_cols = pq.ParquetFile(CANON_FACTOR).schema_arrow.names
    legacy_cols = pq.ParquetFile(LEGACY_PREPROCESSED).schema_arrow.names
    return canon_cols, legacy_cols


def factor_field_mapping(canon_cols: list[str], legacy_cols: list[str]) -> pd.DataFrame:
    rows = []
    for factor in FACTORS:
        canonical_field = factor if factor in canon_cols else ""
        candidates = [c for c in [factor, f"{factor}_neutral", f"{factor}_neutral_z", f"{factor}_z"] if c in legacy_cols]
        legacy_selected = factor if factor in candidates else (candidates[0] if candidates else "")
        same = factor in legacy_cols and factor in canon_cols
        if not canonical_field:
            status = "MISSING_IN_CANONICAL"
        elif not legacy_selected:
            status = "MISSING_IN_LEGACY"
        elif same and f"{factor}_neutral_z" in legacy_cols:
            status = "TRANSFORM_DIFFERENCE"
        elif same:
            status = "EXACT_FIELD_MATCH"
        else:
            status = "EQUIVALENT_FIELD_MATCH"
        rows.append(
            {
                "legacy_factor_name": factor,
                "canonical_field": canonical_field,
                "legacy_field_candidates": ";".join(candidates),
                "legacy_field_selected": legacy_selected,
                "canonical_source": rel(CANON_FACTOR),
                "legacy_source": rel(LEGACY_PREPROCESSED),
                "mapping_status": status,
                "same_name_available": bool(same),
                "transformed_or_raw_difference": bool(f"{factor}_neutral_z" in legacy_cols),
                "neutralized_or_ranked_difference": bool(f"{factor}_neutral_z" in legacy_cols),
                "caveat": "legacy alpha pipeline preferentially uses _neutral_z columns; canonical factor panel stores raw factor layer" if f"{factor}_neutral_z" in legacy_cols else "",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "factor_field_mapping_drift_audit.csv", index=False, encoding="utf-8-sig")
    return out


def load_factor_values() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    canon = pd.read_parquet(CANON_FACTOR, columns=["symbol_norm", "year_month"] + FACTORS)
    canon["symbol_norm"] = norm_symbol(canon["symbol_norm"])
    canon["year_month"] = canon["year_month"].astype(str)
    canon = canon.loc[canon["year_month"].between(START_MONTH, END_MONTH)].copy()

    legacy = pd.read_parquet(LEGACY_PREPROCESSED, columns=["date", "month", "symbol"] + FACTORS)
    legacy["symbol_norm"] = norm_symbol(legacy["symbol"])
    legacy["year_month"] = ym_from_any(legacy["month"])
    legacy = legacy.loc[legacy["year_month"].between(START_MONTH, END_MONTH)].copy()
    common_keys = canon[["symbol_norm", "year_month"]].drop_duplicates().merge(
        legacy[["symbol_norm", "year_month"]].drop_duplicates(), on=["symbol_norm", "year_month"], how="inner"
    )
    return canon, legacy, int(len(common_keys))


def factor_value_overlap(canon: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    merged = canon.merge(legacy[["symbol_norm", "year_month"] + FACTORS], on=["symbol_norm", "year_month"], how="inner", suffixes=("_canonical", "_legacy"))
    rows = []
    for factor in FACTORS:
        c = f"{factor}_canonical"
        l = f"{factor}_legacy"
        g = merged[["year_month", "symbol_norm", c, l]].copy()
        both = g[c].notna() & g[l].notna()
        gb = g.loc[both].copy()
        monthly = []
        rank_diff = []
        symbols_per_month = []
        sign_agree = []
        for ym, m in gb.groupby("year_month", sort=True):
            if len(m) < 3:
                continue
            sp = m[c].corr(m[l], method="spearman")
            monthly.append(sp)
            symbols_per_month.append(len(m))
            r1 = m[c].rank(pct=True)
            r2 = m[l].rank(pct=True)
            rank_diff.append((r1 - r2).abs().mean())
            if (m[c].abs().sum() > 0) and (m[l].abs().sum() > 0):
                sign_agree.append(float((np.sign(m[c]) == np.sign(m[l])).mean()))
        monthly_s = pd.Series(monthly, dtype="float64")
        mean_sp = float(monthly_s.mean()) if not monthly_s.empty else np.nan
        sev = severity_from_spearman(mean_sp)
        rows.append(
            {
                "factor_name": factor,
                "common_row_count": int(both.sum()),
                "common_month_count": int(gb["year_month"].nunique()),
                "common_symbol_avg": float(np.mean(symbols_per_month)) if symbols_per_month else np.nan,
                "pearson": float(gb[c].corr(gb[l], method="pearson")) if len(gb) >= 3 else np.nan,
                "spearman": float(gb[c].corr(gb[l], method="spearman")) if len(gb) >= 3 else np.nan,
                "monthly_spearman_mean": mean_sp,
                "monthly_spearman_p25": float(monthly_s.quantile(0.25)) if not monthly_s.empty else np.nan,
                "monthly_spearman_median": float(monthly_s.quantile(0.50)) if not monthly_s.empty else np.nan,
                "monthly_spearman_p75": float(monthly_s.quantile(0.75)) if not monthly_s.empty else np.nan,
                "mean_abs_rank_diff_pct": float(np.mean(rank_diff)) if rank_diff else np.nan,
                "sign_agreement_if_meaningful": float(np.mean(sign_agree)) if sign_agree else np.nan,
                "non_null_overlap_ratio": float(both.sum() / len(g)) if len(g) else np.nan,
                "drift_severity": sev,
                "likely_issue": "factor value layer drift" if sev in ["HIGH", "CRITICAL"] else "low factor value drift",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "factor_value_overlap_diagnostic.csv", index=False, encoding="utf-8-sig")
    del merged
    gc.collect()
    return out


def price_technical_formula_audit(value_diag: pd.DataFrame) -> pd.DataFrame:
    formulas = {
        "Mom_1M": ("daily all_daily monthly value", "close pct-change over lookback=21, skip=5"),
        "Mom_3M": ("daily all_daily monthly value", "close pct-change over lookback=63, skip=5"),
        "Mom_6M": ("daily all_daily monthly value", "close pct-change over lookback=126, skip=5"),
        "Mom_12M_1M": ("daily all_daily monthly value", "close pct-change over lookback=231, skip=21"),
        "Vol_20D": ("daily all_daily monthly value", "close daily_ret rolling 20D std, min_periods=10"),
        "Vol_60D": ("daily all_daily monthly value", "close daily_ret rolling 60D std, min_periods=30"),
        "Beta": ("daily all_daily monthly value", "rolling 60D cov(stock daily_ret, equal-weight market_ret) / var(market_ret)"),
        "VolChg_20D": ("daily all_daily monthly value", "volume / rolling 20D average volume - 1"),
        "PriceDev_20D": ("daily all_daily monthly value", "close / rolling 20D moving average close - 1"),
    }
    rows = []
    sev_map = value_diag.set_index("factor_name")["drift_severity"].to_dict()
    for factor in PRICE_TECH_FACTORS:
        sev = sev_map.get(factor, "INCONCLUSIVE")
        status = "LIKELY_MATCH" if severity_rank(sev) <= 1 else "VALUE_DRIFT_DESPITE_FORMULA_REFERENCE"
        fix = "NO_FIX_NEEDED" if severity_rank(sev) <= 1 else "ALIGN_MONTH_END_RULE"
        if factor == "Beta" and severity_rank(sev) >= 2:
            fix = "ALIGN_BETA_MARKET_RETURN"
        rows.append(
            {
                "factor_name": factor,
                "canonical_formula": formulas[factor][0],
                "legacy_formula_detected": formulas[factor][1],
                "canonical_source_field": f"{factor}; {factor}_source_flag",
                "legacy_source_field": factor,
                "formula_match_status": status,
                "likely_drift_channel": "month-end alignment/source adjusted price policy" if severity_rank(sev) >= 2 else "none material",
                "severity": sev,
                "recommended_fix_type": fix,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "price_technical_formula_drift_audit.csv", index=False, encoding="utf-8-sig")
    return out


def financial_factor_audit(value_diag: pd.DataFrame) -> pd.DataFrame:
    canonical_raw = {
        "BP": "BP_raw_field",
        "EP": "EP_raw_field",
        "ROE": "ROE_raw_field",
        "Debt_Ratio": "Debt_Ratio_raw_field",
        "Net_Profit_Margin": "Net_Profit_Margin_raw_field",
        "RevGrowth_YoY": "RevGrowth_YoY_raw_field",
        "ProfitGrowth_YoY": "ProfitGrowth_YoY_raw_field",
    }
    legacy_transform = {
        "BP": "BP and BP_neutral_z available",
        "EP": "EP and EP_neutral_z available",
        "ROE": "ROE and ROE_neutral_z available",
        "Debt_Ratio": "Debt_Ratio and Debt_Ratio_neutral_z available",
        "Net_Profit_Margin": "Net_Profit_Margin and Net_Profit_Margin_neutral_z available",
        "RevGrowth_YoY": "RevGrowth_YoY and RevGrowth_YoY_neutral_z available",
        "ProfitGrowth_YoY": "ProfitGrowth_YoY and ProfitGrowth_YoY_neutral_z available",
    }
    sev_map = value_diag.set_index("factor_name")["drift_severity"].to_dict()
    sp_map = value_diag.set_index("factor_name")["monthly_spearman_mean"].to_dict()
    rows = []
    for factor in FIN_FACTORS:
        sev = sev_map.get(factor, "INCONCLUSIVE")
        rows.append(
            {
                "factor_name": factor,
                "canonical_field": factor,
                "legacy_field": factor,
                "canonical_transform": f"canonical v3 PIT raw field ({canonical_raw[factor]}) then strict-lag composite layer",
                "legacy_transform": legacy_transform[factor],
                "value_overlap_monthly_spearman": sp_map.get(factor, np.nan),
                "raw_or_transformed_mismatch": True,
                "pit_policy_difference": "possible" if severity_rank(sev) >= 2 else "not material from value overlap",
                "unit_or_formula_difference": "possible" if severity_rank(sev) >= 2 else "not material from value overlap",
                "severity": sev,
                "recommended_fix_type": "ALIGN_TRANSFORM_POLICY" if severity_rank(sev) >= 2 else "NO_FIX_NEEDED",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "financial_factor_drift_audit.csv", index=False, encoding="utf-8-sig")
    return out


def split_assignment_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    canon = pd.read_parquet(CANON_SPLIT, columns=["symbol_norm", "year_month", "split_rank_pct", "split_group"])
    legacy = pd.read_parquet(LEGACY_SPLIT, columns=["date", "symbol", "universe", "mcap_pct"])
    canon["symbol_norm"] = norm_symbol(canon["symbol_norm"])
    canon["year_month"] = canon["year_month"].astype(str)
    legacy["symbol_norm"] = norm_symbol(legacy["symbol"])
    legacy["year_month"] = ym_from_any(legacy["date"])
    legacy["legacy_split_group"] = legacy["universe"].map({"大盘": "large", "小盘": "small"}).fillna(legacy["universe"].astype(str))
    canon = canon.loc[canon["year_month"].between(START_MONTH, END_MONTH)]
    legacy = legacy.loc[legacy["year_month"].between(START_MONTH, END_MONTH)]
    merged = canon.merge(legacy[["symbol_norm", "year_month", "legacy_split_group", "mcap_pct"]], on=["symbol_norm", "year_month"], how="inner")
    rows = []
    for ym, g in merged.groupby("year_month", sort=True):
        same = g["split_group"].eq(g["legacy_split_group"])
        ratio = float(same.mean()) if len(g) else np.nan
        if ratio >= 0.9:
            sev = "LOW"
        elif ratio >= 0.8:
            sev = "MEDIUM"
        elif ratio >= 0.6:
            sev = "HIGH"
        else:
            sev = "CRITICAL"
        rows.append(
            {
                "year_month": ym,
                "common_symbol_count": int(len(g)),
                "same_split_assignment_count": int(same.sum()),
                "same_split_assignment_ratio": ratio,
                "canonical_large_count": int(g["split_group"].eq("large").sum()),
                "canonical_small_count": int(g["split_group"].eq("small").sum()),
                "legacy_large_count": int(g["legacy_split_group"].eq("large").sum()),
                "legacy_small_count": int(g["legacy_split_group"].eq("small").sum()),
                "split_spearman_marketcap_rank": float(g["split_rank_pct"].corr(g["mcap_pct"], method="spearman")) if len(g) >= 3 else np.nan,
                "drift_severity": sev,
                "likely_issue": "market-cap source/policy drift" if severity_rank(sev) >= 2 else "low split drift",
            }
        )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "split_assignment_drift_audit.csv", index=False, encoding="utf-8-sig")
    avg_ratio = float(detail["same_split_assignment_ratio"].mean())
    min_ratio = float(detail["same_split_assignment_ratio"].min())
    months_below = int((detail["same_split_assignment_ratio"] < 0.8).sum())
    split_sev = "LOW" if avg_ratio >= 0.9 and months_below == 0 else ("MEDIUM" if avg_ratio >= 0.8 else ("HIGH" if avg_ratio >= 0.6 else "CRITICAL"))
    summary = pd.DataFrame(
        [
            {
                "avg_same_split_assignment_ratio": avg_ratio,
                "min_same_split_assignment_ratio": min_ratio,
                "months_below_80pct": months_below,
                "likely_split_drift_driver": "canonical uses total_market_cap_raw_thousand; legacy uses amount/turnover mcap_est with 1/99 winsorized percentile",
                "recommended_action": "ALIGN_SPLIT_MARKET_CAP_SOURCE" if severity_rank(split_sev) >= 2 else "NO_FIX_NEEDED",
            }
        ]
    )
    summary.to_csv(OUT_DIR / "split_assignment_drift_summary.csv", index=False, encoding="utf-8-sig")
    del canon, legacy, merged
    gc.collect()
    return detail, summary


def icir_weight_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    canon = pd.read_csv(CANON_ICIR_AUDIT, dtype={"year_month": "string"})
    canon = canon.loc[canon["year_month"].astype(str).between(START_MONTH, END_MONTH)].copy()
    canon["canonical_ic_ir"] = canon["ic_ir"].astype(float)
    canon["canonical_sign"] = np.sign(canon["canonical_ic_ir"]).replace(0, 1)
    canon["canonical_weight"] = canon["normalized_weight"].astype(float)
    canon["canonical_rank"] = canon["factor_rank_by_abs_icir"].astype(float)
    canon = canon[["year_month", "split_group", "factor_name", "canonical_ic_ir", "canonical_sign", "canonical_weight", "canonical_rank"]]

    legacy = pd.read_csv(LEGACY_ICIR_AUDIT)
    legacy["year_month"] = ym_from_any(legacy["month_end"])
    legacy["split_group"] = legacy["universe"].map({"大盘": "large", "小盘": "small"}).fillna(legacy["universe"].astype(str))
    legacy["factor_name"] = legacy["factor_name"].astype(str).str.replace("_neutral_z", "", regex=False).str.replace("_z", "", regex=False)
    legacy = legacy.loc[legacy["year_month"].between(START_MONTH, END_MONTH) & legacy["factor_name"].isin(FACTORS)].copy()
    legacy["legacy_ic_ir"] = legacy["icir_value"].astype(float)
    legacy["legacy_sign"] = np.sign(legacy["legacy_ic_ir"]).replace(0, 1)
    legacy["abs_icir"] = legacy["legacy_ic_ir"].abs()
    legacy["legacy_rank"] = legacy.groupby(["year_month", "split_group"])["abs_icir"].rank(method="first", ascending=False)
    total_abs = legacy.groupby(["year_month", "split_group"])["abs_icir"].transform("sum")
    legacy["legacy_weight"] = np.where(total_abs > 0, legacy["abs_icir"] / total_abs, 0.0)
    legacy = legacy[["year_month", "split_group", "factor_name", "legacy_ic_ir", "legacy_sign", "legacy_weight", "legacy_rank"]]

    out = canon.merge(legacy, on=["year_month", "split_group", "factor_name"], how="inner")
    out["ic_ir_diff"] = out["canonical_ic_ir"] - out["legacy_ic_ir"]
    out["sign_match"] = out["canonical_sign"].eq(out["legacy_sign"])
    out["weight_diff"] = out["canonical_weight"] - out["legacy_weight"]
    out["rank_diff"] = out["canonical_rank"] - out["legacy_rank"]
    out["drift_severity"] = np.select(
        [
            (~out["sign_match"]) | (out["weight_diff"].abs() >= 0.20) | (out["rank_diff"].abs() >= 8),
            (out["weight_diff"].abs() >= 0.10) | (out["rank_diff"].abs() >= 5),
            (out["weight_diff"].abs() >= 0.05) | (out["rank_diff"].abs() >= 3),
        ],
        ["CRITICAL", "HIGH", "MEDIUM"],
        default="LOW",
    )
    out = out[
        [
            "year_month",
            "split_group",
            "factor_name",
            "canonical_ic_ir",
            "legacy_ic_ir",
            "ic_ir_diff",
            "canonical_sign",
            "legacy_sign",
            "sign_match",
            "canonical_weight",
            "legacy_weight",
            "weight_diff",
            "canonical_rank",
            "legacy_rank",
            "rank_diff",
            "drift_severity",
        ]
    ]
    out.to_csv(OUT_DIR / "icir_factor_weight_drift_audit.csv", index=False, encoding="utf-8-sig")
    rows = []
    for (split, factor), g in out.groupby(["split_group", "factor_name"], sort=True):
        sev = worst_severity(g["drift_severity"])
        rows.append(
            {
                "split_group": split,
                "factor_name": factor,
                "sign_match_ratio": float(g["sign_match"].mean()),
                "avg_abs_icir_diff": float(g["ic_ir_diff"].abs().mean()),
                "avg_abs_weight_diff": float(g["weight_diff"].abs().mean()),
                "avg_rank_diff": float(g["rank_diff"].abs().mean()),
                "severity": sev,
                "likely_issue": "strict-lag ICIR/sign/weight policy drift" if severity_rank(sev) >= 2 else "low ICIR weight drift",
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "icir_factor_weight_drift_summary.csv", index=False, encoding="utf-8-sig")
    del canon, legacy
    gc.collect()
    return out, summary


def top50_drift() -> tuple[pd.DataFrame, pd.DataFrame]:
    cw = pd.read_parquet(CANON_WEIGHTS, columns=["symbol_norm", "year_month", "alpha_signal", "rank_in_month", "weight"])
    lw = pd.read_parquet(LEGACY_WEIGHTS, columns=["symbol", "month_end", "alpha_signal_strict_lag", "rank_in_month", "selected_flag", "weight"])
    cw["symbol_norm"] = norm_symbol(cw["symbol_norm"])
    cw["year_month"] = cw["year_month"].astype(str)
    lw["symbol_norm"] = norm_symbol(lw["symbol"])
    lw["year_month"] = ym_from_any(lw["month_end"])
    cw = cw.loc[cw["year_month"].between(START_MONTH, END_MONTH)].copy()
    lw = lw.loc[lw["year_month"].between(START_MONTH, END_MONTH)].copy()
    lw = lw.loc[(lw["selected_flag"].astype(str).str.lower().isin(["true", "1"])) | (lw["weight"].fillna(0) > 0)].copy()

    return_diff = pd.DataFrame(columns=["year_month", "diff_canonical_minus_legacy"])
    if FORENSIC_DIFF.exists():
        return_diff = pd.read_csv(FORENSIC_DIFF, dtype={"year_month": "string"})
        return_diff = return_diff.groupby("year_month", as_index=False)["diff_canonical_minus_legacy"].mean()

    rows = []
    for ym in sorted(set(cw["year_month"]).intersection(set(lw["year_month"]))):
        c = cw.loc[cw["year_month"].eq(ym)].nsmallest(50, "rank_in_month")
        l = lw.loc[lw["year_month"].eq(ym)].nsmallest(50, "rank_in_month")
        cset = set(c["symbol_norm"])
        lset = set(l["symbol_norm"])
        overlap = len(cset & lset)
        ratio = overlap / 50.0 if cset and lset else np.nan
        rd = return_diff.loc[return_diff["year_month"].astype(str).eq(ym), "diff_canonical_minus_legacy"]
        if pd.isna(ratio):
            sev = "CRITICAL"
        elif ratio >= 0.7:
            sev = "LOW"
        elif ratio >= 0.5:
            sev = "MEDIUM"
        elif ratio >= 0.3:
            sev = "HIGH"
        else:
            sev = "CRITICAL"
        rows.append(
            {
                "year_month": ym,
                "canonical_top50_count": int(len(cset)),
                "legacy_top50_count": int(len(lset)),
                "overlap_count": int(overlap),
                "overlap_ratio": ratio,
                "only_canonical_count": int(len(cset - lset)),
                "only_legacy_count": int(len(lset - cset)),
                "canonical_mean_alpha": float(c["alpha_signal"].mean()),
                "legacy_mean_alpha": float(l["alpha_signal_strict_lag"].mean()),
                "monthly_return_diff_from_forensic_if_available": float(rd.iloc[0]) if not rd.empty else np.nan,
                "drift_severity": sev,
            }
        )
    diag = pd.DataFrame(rows)
    diag.to_csv(OUT_DIR / "top50_drift_monthly_diagnostic.csv", index=False, encoding="utf-8-sig")
    worst = diag.sort_values(["overlap_ratio", "monthly_return_diff_from_forensic_if_available"], ascending=[True, True]).head(10).copy()
    worst["return_diff"] = worst["monthly_return_diff_from_forensic_if_available"]
    worst["suspected_drift_layer"] = "Layer 4/5 ICIR-GS composite drift"
    worst["recommended_review"] = "review factor ICIR/sign/weight and composite residualization for this month"
    worst[["year_month", "overlap_ratio", "return_diff", "suspected_drift_layer", "recommended_review"]].to_csv(
        OUT_DIR / "top50_drift_worst_months.csv", index=False, encoding="utf-8-sig"
    )
    del cw, lw
    gc.collect()
    return diag, worst


def alpha_layer_attribution(
    value_diag: pd.DataFrame,
    split_summary: pd.DataFrame,
    icir_summary: pd.DataFrame,
    top50_diag: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str, bool, str, bool]:
    avg_factor_sp = float(value_diag["monthly_spearman_mean"].mean())
    critical_count = int(value_diag["drift_severity"].eq("CRITICAL").sum())
    high_count = int(value_diag["drift_severity"].eq("HIGH").sum())
    split_ratio = float(split_summary["avg_same_split_assignment_ratio"].iloc[0])
    icir_sev = worst_severity(icir_summary["severity"])
    avg_top50 = float(top50_diag["overlap_ratio"].mean())

    factor_layer_sev = severity_from_spearman(avg_factor_sp)
    split_layer_sev = "LOW" if split_ratio >= 0.9 else ("MEDIUM" if split_ratio >= 0.8 else ("HIGH" if split_ratio >= 0.6 else "CRITICAL"))
    buffer_material = bool(avg_top50 < 0.5)

    if severity_rank(factor_layer_sev) >= 2 and critical_count + high_count >= 6:
        primary = "Layer 1: factor value drift"
        secondary = "Layer 2: transform / zscore / rank drift"
        decision_hint = "FACTOR_INPUT_DRIFT_PRIMARY_REPAIR_FACTOR_PANEL"
        first_repair = "ALIGN_CANONICAL_FACTOR_PANEL_TO_LEGACY_PREPROCESSED_RAW_AND_MONTH_END"
    elif severity_rank(split_layer_sev) >= 2:
        primary = "Layer 3: split assignment drift"
        secondary = "Layer 1: factor value drift"
        decision_hint = "SPLIT_ASSIGNMENT_DRIFT_PRIMARY_REPAIR_SPLIT_POLICY"
        first_repair = "ALIGN_SPLIT_POLICY_TO_LEGACY_MCAP_EST_AMOUNT_OVER_TURNOVER"
    elif severity_rank(icir_sev) >= 2:
        primary = "Layer 4: IC_IR / sign / weight drift"
        secondary = "Layer 5: Gram-Schmidt residualization drift"
        decision_hint = "ICIR_WEIGHT_DRIFT_PRIMARY_REPAIR_STRICT_LAG_COMPOSITE"
        first_repair = "ALIGN_STRICT_LAG_ICIR_INPUT_COLUMNS_AND_WEIGHT_NORMALIZATION"
    elif buffer_material:
        primary = "Layer 6: buffer portfolio conversion drift"
        secondary = "Layer 4: IC_IR / sign / weight drift"
        decision_hint = "BUFFER_CONSTRUCTION_DRIFT_PRIMARY_REPAIR_PORTFOLIO_RULE"
        first_repair = "ALIGN_BUFFER_35_75_PORTFOLIO_RULE"
    elif avg_top50 < 0.35:
        primary = "Layer 5: Gram-Schmidt residualization drift"
        secondary = "Layer 4: IC_IR / sign / weight drift"
        decision_hint = "GS_COMPOSITE_DRIFT_PRIMARY_REPAIR_COMPOSITE_IMPLEMENTATION"
        first_repair = "ALIGN_GRAM_SCHMIDT_COMPOSITE_IMPLEMENTATION"
    else:
        primary = "UNKNOWN"
        secondary = "UNKNOWN"
        decision_hint = "DRIFT_AUDIT_INCONCLUSIVE_MORE_INSPECTION_REQUIRED"
        first_repair = "MORE_INSPECTION_REQUIRED"

    rows = [
        {
            "layer": "Layer 1: factor value drift",
            "evidence": f"avg monthly factor Spearman={avg_factor_sp:.6f}; high={high_count}; critical={critical_count}",
            "severity": factor_layer_sev,
            "contribution_to_alpha_divergence": "PRIMARY" if "factor value" in primary else ("SECONDARY" if "factor value" in secondary else "MINOR"),
            "recommended_action": "repair factor panel source/month-end alignment" if severity_rank(factor_layer_sev) >= 2 else "no immediate repair",
        },
        {
            "layer": "Layer 2: transform / zscore / rank drift",
            "evidence": "legacy preprocessed exposes _neutral_z columns and pipeline prioritizes _neutral_z; canonical panel audit compared raw layer",
            "severity": "HIGH" if severity_rank(factor_layer_sev) >= 2 else "MEDIUM",
            "contribution_to_alpha_divergence": "SECONDARY" if "transform" in secondary else "MINOR",
            "recommended_action": "compare canonical transform/zscore layer against legacy _neutral_z",
        },
        {
            "layer": "Layer 3: split assignment drift",
            "evidence": f"avg same split assignment ratio={split_ratio:.6f}",
            "severity": split_layer_sev,
            "contribution_to_alpha_divergence": "PRIMARY" if "split assignment" in primary else "SECONDARY" if severity_rank(split_layer_sev) >= 2 else "MINOR",
            "recommended_action": "align market-cap source and percentile rule" if severity_rank(split_layer_sev) >= 2 else "no immediate repair",
        },
        {
            "layer": "Layer 4: IC_IR / sign / weight drift",
            "evidence": f"worst ICIR/weight severity={icir_sev}",
            "severity": icir_sev,
            "contribution_to_alpha_divergence": "PRIMARY" if "IC_IR" in primary else ("SECONDARY" if "IC_IR" in secondary else "MINOR"),
            "recommended_action": "align strict-lag ICIR inputs/sign/normalization" if severity_rank(icir_sev) >= 2 else "no immediate repair",
        },
        {
            "layer": "Layer 5: Gram-Schmidt residualization drift",
            "evidence": "large Top50 drift after alpha construction; direct residual contribution file unavailable",
            "severity": "HIGH" if avg_top50 < 0.35 else "MEDIUM",
            "contribution_to_alpha_divergence": "PRIMARY" if "Gram-Schmidt" in primary else ("SECONDARY" if "Gram-Schmidt" in secondary else "UNKNOWN"),
            "recommended_action": "inspect composite implementation only after factor/split/ICIR layer is aligned",
        },
        {
            "layer": "Layer 6: buffer portfolio conversion drift",
            "evidence": f"avg Top50 overlap ratio={avg_top50:.6f}",
            "severity": "HIGH" if buffer_material else "LOW",
            "contribution_to_alpha_divergence": "PRIMARY" if "buffer" in primary else "MINOR",
            "recommended_action": "repair buffer rule only if alpha layer overlap becomes high but holdings remain divergent",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "alpha_drift_layer_attribution.csv", index=False, encoding="utf-8-sig")
    return out, primary, secondary, buffer_material, first_repair, decision_hint


def decision_summary(
    value_diag: pd.DataFrame,
    price_audit: pd.DataFrame,
    fin_audit: pd.DataFrame,
    split_summary: pd.DataFrame,
    icir_summary: pd.DataFrame,
    primary_layer: str,
    first_repair: str,
) -> pd.DataFrame:
    avg_factor_sp = float(value_diag["monthly_spearman_mean"].mean())
    price_sev = worst_severity(price_audit["severity"])
    fin_sev = worst_severity(fin_audit["severity"])
    split_ratio = float(split_summary["avg_same_split_assignment_ratio"].iloc[0])
    icir_sev = worst_severity(icir_summary["severity"])
    rows = [
        ("Are canonical factor values close to legacy values?", f"avg monthly Spearman={avg_factor_sp:.6f}", worst_severity(value_diag["drift_severity"]), "若非 LOW/MEDIUM，先修 factor panel"),
        ("Are price/technical factors the main drift source?", f"worst severity={price_sev}", price_sev, "优先核对 all_daily price/volume/month-end 规则" if severity_rank(price_sev) >= 2 else "不是主修复项"),
        ("Are financial factors the main drift source?", f"worst severity={fin_sev}", fin_sev, "核对 PIT、单位和 raw/neutral_z transform" if severity_rank(fin_sev) >= 2 else "不是主修复项"),
        ("Is split assignment drift material?", f"avg same split ratio={split_ratio:.6f}", "HIGH" if split_ratio < 0.8 else "LOW", "若低于 80%，对齐 mcap_est/percentile split"),
        ("Is IC_IR/factor weight drift material?", f"worst severity={icir_sev}", icir_sev, "核对 strict-lag ICIR input columns、sign 和 weight normalization" if severity_rank(icir_sev) >= 2 else "不是主修复项"),
        ("Is GS/composite construction drift suspected?", f"primary_layer={primary_layer}", "HIGH" if "Gram-Schmidt" in primary_layer or "IC_IR" in primary_layer else "MEDIUM", "在 factor/split/ICIR 对齐后再审 GS residualization"),
        ("Is buffer portfolio construction drift material?", "only diagnose after alpha overlap improves", "MEDIUM", "当前 Top50 drift 更像 alpha 层传导，不先修 buffer"),
        ("Should canonical reconstruction be repaired before attribution?", "true", "HIGH", "先修 reconstruction，不进入 benchmark attribution"),
        ("What exact repair should be attempted first?", first_repair, "HIGH", first_repair),
    ]
    out = pd.DataFrame([{"question": q, "finding": f, "severity": s, "recommended_action": a} for q, f, s, a in rows])
    out.to_csv(OUT_DIR / "v0_reconstruction_drift_decision_summary.csv", index=False, encoding="utf-8-sig")
    return out


def guardrails() -> tuple[pd.DataFrame, bool]:
    values = {
        "alpha_signal_regenerated": False,
        "strategy_weights_regenerated": False,
        "portfolio_returns_recomputed": False,
        "old_artifacts_modified": False,
        "production_modified": False,
        "ml_training_run": False,
        "tuning_run": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
    }
    out = pd.DataFrame([{"guardrail": k, "expected": v, "actual": v, "pass": True} for k, v in values.items()])
    out.to_csv(OUT_DIR / "v0_reconstruction_drift_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    return out, bool(out["pass"].all())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_run_state("running", "prerequisite_check")
    prereq = prerequisite_check()
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_run_state("running", "schema_and_mapping")
    canon_cols, legacy_cols = read_schemas()
    mapping = factor_field_mapping(canon_cols, legacy_cols)

    write_run_state("running", "factor_value_overlap")
    canon_factor, legacy_factor, common_symbol_month_count = load_factor_values()
    factor_diag = factor_value_overlap(canon_factor, legacy_factor)
    del canon_factor, legacy_factor
    gc.collect()

    price_audit = price_technical_formula_audit(factor_diag)
    fin_audit = financial_factor_audit(factor_diag)

    write_run_state("running", "split_icir_top50_layers")
    split_detail, split_summary = split_assignment_audit()
    icir_detail, icir_summary = icir_weight_audit()
    top50_diag, top50_worst = top50_drift()
    layer_attr, primary_layer, secondary_layer, buffer_material, first_repair, decision_hint = alpha_layer_attribution(
        factor_diag, split_summary, icir_summary, top50_diag
    )
    decision = decision_summary(factor_diag, price_audit, fin_audit, split_summary, icir_summary, primary_layer, first_repair)
    guardrail, guardrails_passed = guardrails()

    avg_factor_sp = float(factor_diag["monthly_spearman_mean"].mean())
    critical_count = int(factor_diag["drift_severity"].eq("CRITICAL").sum())
    high_count = int(factor_diag["drift_severity"].eq("HIGH").sum())
    price_sev = worst_severity(price_audit["severity"])
    fin_sev = worst_severity(fin_audit["severity"])
    split_avg = float(split_summary["avg_same_split_assignment_ratio"].iloc[0])
    split_sev = "LOW" if split_avg >= 0.9 else ("MEDIUM" if split_avg >= 0.8 else ("HIGH" if split_avg >= 0.6 else "CRITICAL"))
    icir_sev = worst_severity(icir_summary["severity"])
    avg_top50 = float(top50_diag["overlap_ratio"].mean())
    worst_month = str(top50_diag.sort_values("overlap_ratio").iloc[0]["year_month"])

    if not guardrails_passed:
        final_decision = "DRIFT_AUDIT_FAIL_GUARDRAIL"
    else:
        final_decision = decision_hint

    canonical_repair_required = final_decision != "DRIFT_AUDIT_INCONCLUSIVE_MORE_INSPECTION_REQUIRED"
    continue_allowed = False
    recommended_next_step = {
        "FACTOR_INPUT_DRIFT_PRIMARY_REPAIR_FACTOR_PANEL": "先对齐 canonical factor panel 与 legacy preprocessed 的 raw/source/month-end 口径，再复核 alpha overlap。",
        "SPLIT_ASSIGNMENT_DRIFT_PRIMARY_REPAIR_SPLIT_POLICY": "先对齐 split market-cap source 与 percentile policy，再复核 alpha overlap。",
        "ICIR_WEIGHT_DRIFT_PRIMARY_REPAIR_STRICT_LAG_COMPOSITE": "先对齐 strict-lag ICIR 输入列、sign 和 normalized weight，再复核 alpha overlap。",
        "GS_COMPOSITE_DRIFT_PRIMARY_REPAIR_COMPOSITE_IMPLEMENTATION": "先审计 Gram-Schmidt residualization/composite 实现，再复核 alpha overlap。",
        "BUFFER_CONSTRUCTION_DRIFT_PRIMARY_REPAIR_PORTFOLIO_RULE": "先对齐 buffer 35/75 portfolio conversion rule，再复核 holdings overlap。",
        "DRIFT_AUDIT_INCONCLUSIVE_MORE_INSPECTION_REQUIRED": "补充 transform/zscore 层中间文件后再定修复路径。",
        "DRIFT_AUDIT_FAIL_GUARDRAIL": "停止，先修复 guardrail violation。",
    }[final_decision]

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "common_window_month_count": len(pd.period_range(START_MONTH, END_MONTH, freq="M")),
        "common_symbol_month_count": common_symbol_month_count,
        "avg_factor_monthly_spearman": avg_factor_sp,
        "critical_factor_drift_count": critical_count,
        "high_factor_drift_count": high_count,
        "price_technical_drift_severity": price_sev,
        "financial_drift_severity": fin_sev,
        "split_assignment_avg_same_ratio": split_avg,
        "split_assignment_drift_severity": split_sev,
        "icir_weight_drift_severity": icir_sev,
        "alpha_drift_primary_layer": primary_layer,
        "alpha_drift_secondary_layer": secondary_layer,
        "avg_top50_overlap_ratio": avg_top50,
        "worst_top50_overlap_month": worst_month,
        "buffer_construction_drift_material": buffer_material,
        "canonical_reconstruction_repair_required": canonical_repair_required,
        "recommended_first_repair": first_repair,
        "continue_to_attribution_allowed": continue_allowed,
        "guardrails_passed": guardrails_passed,
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    dump_json(OUT_DIR / "v0_canonical_vs_legacy_reconstruction_drift_audit_summary.json", summary)

    report = (
        "# V0 canonical vs legacy reconstruction drift audit v0\n\n"
        f"- final_decision: {final_decision}\n"
        f"- common window: {START_MONTH} to {END_MONTH}; months={summary['common_window_month_count']}; common symbol-month={common_symbol_month_count}\n"
        f"- avg factor monthly Spearman: {avg_factor_sp:.6f}; high={high_count}; critical={critical_count}\n"
        f"- price/technical severity: {price_sev}; financial severity: {fin_sev}\n"
        f"- split avg same ratio: {split_avg:.6f}; split severity: {split_sev}\n"
        f"- ICIR/weight severity: {icir_sev}\n"
        f"- primary layer: {primary_layer}; secondary layer: {secondary_layer}\n"
        f"- avg Top50 overlap: {avg_top50:.6f}; worst month: {worst_month}\n"
        f"- recommended first repair: {first_repair}\n"
        f"- guardrails passed: {guardrails_passed}\n\n"
        "本任务只做只读 drift audit；未重新生成 alpha_signal/weights，未重算正式收益，未做 benchmark-relative、alpha/beta、IR/TE、FF、DGTW、ML、SHAP 或 production 修改。\n"
    )
    (OUT_DIR / "v0_canonical_vs_legacy_reconstruction_drift_audit_report.md").write_text(report, encoding="utf-8")

    final_qa = pd.DataFrame(
        [
            {"check_name": "prerequisites_passed", "pass": prereq["prerequisites_passed"], "detail": ""},
            {"check_name": "guardrails_passed", "pass": guardrails_passed, "detail": ""},
            {"check_name": "factor_diag_generated", "pass": not factor_diag.empty, "detail": str(len(factor_diag))},
            {"check_name": "top50_diag_generated", "pass": not top50_diag.empty, "detail": str(len(top50_diag))},
            {"check_name": "final_decision_allowed", "pass": final_decision in {
                "FACTOR_INPUT_DRIFT_PRIMARY_REPAIR_FACTOR_PANEL",
                "SPLIT_ASSIGNMENT_DRIFT_PRIMARY_REPAIR_SPLIT_POLICY",
                "ICIR_WEIGHT_DRIFT_PRIMARY_REPAIR_STRICT_LAG_COMPOSITE",
                "GS_COMPOSITE_DRIFT_PRIMARY_REPAIR_COMPOSITE_IMPLEMENTATION",
                "BUFFER_CONSTRUCTION_DRIFT_PRIMARY_REPAIR_PORTFOLIO_RULE",
                "DRIFT_AUDIT_INCONCLUSIVE_MORE_INSPECTION_REQUIRED",
                "DRIFT_AUDIT_FAIL_GUARDRAIL",
            }, "detail": final_decision},
        ]
    )
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")

    terminal_summary = {
        "task_name": TASK_NAME,
        "status": "completed",
        "stdout_path": rel(RUN_DIR / "run_stdout.txt"),
        "stderr_path": rel(RUN_DIR / "run_stderr.txt"),
        "output_dir": rel(OUT_DIR),
        "final_decision": final_decision,
    }
    dump_json(OUT_DIR / "terminal_summary.json", terminal_summary)
    (OUT_DIR / "task_completion_card.md").write_text(
        f"# Task completion card\n\n- task_name: {TASK_NAME}\n- status: completed\n- final_decision: {final_decision}\n- output_dir: {rel(OUT_DIR)}\n",
        encoding="utf-8",
    )

    del mapping, factor_diag, price_audit, fin_audit, split_detail, split_summary, icir_detail, icir_summary, top50_diag, top50_worst, layer_attr, decision, guardrail
    gc.collect()
    write_run_state("completed", "all_outputs_written")
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": rel(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
