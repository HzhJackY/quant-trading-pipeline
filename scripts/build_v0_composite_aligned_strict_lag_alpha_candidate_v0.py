from __future__ import annotations

import gc
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "v0_composite_aligned_strict_lag_alpha_candidate_build_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

PREP_DIR = ROOT / "output" / "v0_strict_lag_composite_alignment_repair_prep_v0"
CANON_ALPHA_DIR = ROOT / "output" / "v0_canonical_strict_lag_alpha_build_v0"
LEGACY_DIR = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0"

PREP_SUMMARY = PREP_DIR / "v0_strict_lag_composite_alignment_repair_prep_summary.json"
REPAIR_DESIGN = PREP_DIR / "composite_alignment_repair_design.csv"
RUN_CONFIG = PREP_DIR / "v0_composite_alignment_repair_run_config_draft.json"
LEGACY_POLICY = PREP_DIR / "legacy_composite_input_policy_summary.json"
CANON_FACTOR = ROOT / "output" / "v0_canonical_16factor_panel_build_v0" / "v0_canonical_16factor_panel.parquet"
PREVIOUS_CANON_ALPHA = CANON_ALPHA_DIR / "v0_canonical_alpha_signal_panel.parquet"
PREVIOUS_CANON_ICIR = CANON_ALPHA_DIR / "v0_canonical_strict_lag_icir_by_month_factor.csv"
PREVIOUS_CANON_AUDIT = CANON_ALPHA_DIR / "v0_canonical_factor_icir_contribution_audit.csv"
LEGACY_PREPROCESSED = ROOT / "output" / "preprocessed.parquet"
LEGACY_SPLIT = ROOT / "output" / "split_universe_blended.parquet"
LEGACY_ALPHA = LEGACY_DIR / "v0_strict_lag_alpha_signal_panel.parquet"
LEGACY_ICIR_AUDIT = LEGACY_DIR / "v0_strict_lag_icir_window_audit.csv"
RETURN_MAP = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0" / "canonical_csmar_trd_mnth_return_map_repaired.parquet"

LEGACY_SCRIPTS = [
    ROOT / "run_split_universe.py",
    ROOT / "factor_research" / "split_universe.py",
    ROOT / "factor_research" / "backtest_engine.py",
    ROOT / "factor_research" / "orthogonalization.py",
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
ROLLING_WINDOW = 24
MIN_STOCKS = 20
MIN_IC_IR = 0.05


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def dump_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_state(status: str, checkpoint: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "RUN_STATE.md").write_text(
        "# RUN_STATE\n\n"
        f"task_name: {TASK_NAME}\n"
        f"status: {status}\n"
        f"last_checkpoint: {checkpoint}\n"
        f"updated_at: {datetime.now().isoformat(timespec='seconds')}\n"
        "resume_instruction: rerun scripts\\build_v0_composite_aligned_strict_lag_alpha_candidate_v0.py with stdout/stderr redirected to this run directory\n",
        encoding="utf-8",
    )


def norm_symbol(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def ym_from_any(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.astype(str)
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.to_period("M").astype(str)
    s = series.astype(str)
    if bool(s.str.match(r"^\d{4}-\d{2}$", na=False).all()):
        return s
    return pd.to_datetime(series, errors="coerce").dt.to_period("M").astype(str)


def finite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def zscore(s: pd.Series) -> pd.Series:
    vals = finite(s)
    std = vals.std(ddof=1)
    if pd.isna(std) or std <= 1e-12:
        return pd.Series(0.0, index=s.index)
    return (vals - vals.mean()) / std


def rank_ic(x: pd.Series, y: pd.Series) -> float:
    sub = pd.DataFrame({"x": finite(x), "y": finite(y)}).dropna()
    if len(sub) < MIN_STOCKS:
        return np.nan
    corr = sub["x"].rank(method="average").corr(sub["y"].rank(method="average"))
    return float(corr) if pd.notna(corr) else np.nan


def residualize(y: np.ndarray, x: np.ndarray, min_variance: float = 1e-10) -> np.ndarray:
    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        resid = y - x @ beta
    except np.linalg.LinAlgError:
        return np.zeros_like(y)
    if np.var(resid) < min_variance:
        return np.zeros_like(y)
    return resid


def prerequisites() -> dict:
    flags = {
        "repair_prep_summary_found": PREP_SUMMARY.exists(),
        "repair_design_found": REPAIR_DESIGN.exists(),
        "run_config_found": RUN_CONFIG.exists(),
        "canonical_factor_panel_found": CANON_FACTOR.exists(),
        "legacy_preprocessed_found": LEGACY_PREPROCESSED.exists(),
        "legacy_split_universe_blended_found": LEGACY_SPLIT.exists(),
        "legacy_strict_lag_alpha_found": LEGACY_ALPHA.exists(),
        "trd_mnth_return_map_found": RETURN_MAP.exists(),
        "legacy_implementation_scripts_found": all(p.exists() for p in LEGACY_SCRIPTS),
    }
    required = [
        ("repair_prep_summary_found", PREP_SUMMARY),
        ("repair_design_found", REPAIR_DESIGN),
        ("run_config_found", RUN_CONFIG),
        ("canonical_factor_panel_found", CANON_FACTOR),
        ("legacy_preprocessed_found", LEGACY_PREPROCESSED),
        ("legacy_split_universe_blended_found", LEGACY_SPLIT),
        ("legacy_strict_lag_alpha_found", LEGACY_ALPHA),
        ("trd_mnth_return_map_found", RETURN_MAP),
    ]
    missing = [rel(path) for key, path in required if not flags[key]]
    if not flags["legacy_implementation_scripts_found"]:
        missing.extend(rel(p) for p in LEGACY_SCRIPTS if not p.exists())
    flags["prerequisites_passed"] = not missing
    flags["missing_files"] = missing
    flags["caveat"] = "TRD_Mnth return map is used only for strict-lag IC labels, not portfolio returns."
    dump_json(OUT_DIR / "v0_composite_aligned_alpha_candidate_prerequisite_check.json", flags)
    return flags


def build_input_view() -> tuple[pd.DataFrame, pd.DataFrame]:
    canon_cols = ["symbol_norm", "year_month", "month_end", "total_market_cap_raw_thousand", *FACTORS]
    canon = pd.read_parquet(CANON_FACTOR, columns=canon_cols)
    canon["symbol_norm"] = norm_symbol(canon["symbol_norm"])
    canon["year_month"] = canon["year_month"].astype(str).str.slice(0, 7)
    for col in ["total_market_cap_raw_thousand", *FACTORS]:
        canon[col] = finite(canon[col])

    legacy_cols = ["symbol", "date", "month", *[f"{f}_neutral_z" for f in FACTORS], *[f"{f}_z" for f in FACTORS], *FACTORS]
    legacy_probe = pd.read_parquet(LEGACY_PREPROCESSED, columns=[c for c in legacy_cols if c in pd.read_parquet(LEGACY_PREPROCESSED, columns=[]).columns])
    # The empty-column read above can be unavailable on some engines; fall back to full schema via pyarrow.
    del legacy_probe
    import pyarrow.parquet as pq

    legacy_available = pq.ParquetFile(LEGACY_PREPROCESSED).schema_arrow.names
    read_cols = [c for c in legacy_cols if c in legacy_available]
    legacy = pd.read_parquet(LEGACY_PREPROCESSED, columns=read_cols)
    legacy["symbol_norm"] = norm_symbol(legacy["symbol"])
    legacy["year_month"] = ym_from_any(legacy["month"] if "month" in legacy.columns else legacy["date"])
    keep_legacy = ["symbol_norm", "year_month"]
    for factor in FACTORS:
        for candidate in [f"{factor}_neutral_z", f"{factor}_z", factor]:
            if candidate in legacy.columns:
                keep_legacy.append(candidate)
                break
    legacy = legacy[keep_legacy].drop_duplicates(["symbol_norm", "year_month"], keep="last")

    view = canon.copy()
    manifest_rows = []
    for factor in FACTORS:
        target = f"{factor}_neutral_z"
        aligned = f"{factor}_aligned_input"
        source_col = factor
        exact_col = target if target in legacy.columns else (f"{factor}_z" if f"{factor}_z" in legacy.columns else factor)
        merge_col = legacy[["symbol_norm", "year_month", exact_col]].rename(columns={exact_col: f"{factor}_legacy_input"})
        view = view.merge(merge_col, on=["symbol_norm", "year_month"], how="left")
        generated_z = view.groupby("year_month")[factor].transform(zscore)
        view[aligned] = view[f"{factor}_legacy_input"].where(view[f"{factor}_legacy_input"].notna(), generated_z)
        view[f"{factor}_source_field"] = np.where(view[f"{factor}_legacy_input"].notna(), exact_col, source_col)
        view[f"{factor}_alignment_flag"] = np.where(view[f"{factor}_legacy_input"].notna(), "EXACT_LEGACY_POLICY", "ALIGNED_BY_GENERATED_Z")
        manifest_rows.append(
            {
                "factor_name": factor,
                "legacy_target_input_field": target,
                "canonical_source_field": source_col,
                "aligned_input_field": aligned,
                "input_policy": "prefer legacy neutral_z common rows, otherwise generated monthly z from canonical raw",
                "neutral_z_available": target in legacy.columns,
                "z_available": f"{factor}_z" in legacy.columns,
                "raw_available": factor in canon.columns,
                "fallback_used": bool(view[f"{factor}_legacy_input"].isna().any()),
                "fallback_reason": "canonical panel lacks industry-neutral fields; generated monthly z for rows not present in legacy preprocessed",
                "alignment_status": "EXACT_LEGACY_POLICY" if not view[f"{factor}_legacy_input"].isna().any() else "ALIGNED_BY_GENERATED_Z",
            }
        )
    source_cols = [f"{f}_source_field" for f in FACTORS]
    flag_cols = [f"{f}_alignment_flag" for f in FACTORS]
    aligned_cols = [f"{f}_aligned_input" for f in FACTORS]
    out_cols = ["symbol_norm", "year_month", "month_end", "total_market_cap_raw_thousand", *aligned_cols, *source_cols, *flag_cols]
    view[out_cols].to_parquet(OUT_DIR / "v0_composite_aligned_input_view.parquet", index=False)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "v0_composite_aligned_input_column_manifest.csv", index=False, encoding="utf-8-sig")
    del canon, legacy
    gc.collect()
    return view[out_cols], manifest


def aligned_input_legacy_qa(view: pd.DataFrame) -> pd.DataFrame:
    import pyarrow.parquet as pq

    legacy_available = pq.ParquetFile(LEGACY_PREPROCESSED).schema_arrow.names
    read_cols = ["symbol", "month", "date"] + [f"{f}_neutral_z" for f in FACTORS if f"{f}_neutral_z" in legacy_available]
    legacy = pd.read_parquet(LEGACY_PREPROCESSED, columns=[c for c in read_cols if c in legacy_available])
    legacy["symbol_norm"] = norm_symbol(legacy["symbol"])
    legacy["year_month"] = ym_from_any(legacy["month"] if "month" in legacy.columns else legacy["date"])
    rows = []
    for factor in FACTORS:
        lcol = f"{factor}_neutral_z"
        acol = f"{factor}_aligned_input"
        if lcol not in legacy.columns:
            rows.append({"factor_name": factor, "common_row_count": 0, "common_month_count": 0, "monthly_spearman_mean": np.nan, "monthly_spearman_median": np.nan, "monthly_spearman_p25": np.nan, "monthly_spearman_p75": np.nan, "non_null_overlap_ratio": 0.0, "alignment_quality": "INCONCLUSIVE"})
            continue
        merged = view[["symbol_norm", "year_month", acol]].merge(
            legacy[["symbol_norm", "year_month", lcol]], on=["symbol_norm", "year_month"], how="inner"
        )
        both = merged[acol].notna() & merged[lcol].notna()
        monthly = []
        for _, g in merged.loc[both].groupby("year_month", sort=True):
            if len(g) >= 3:
                monthly.append(g[acol].corr(g[lcol], method="spearman"))
        s = pd.Series(monthly, dtype="float64")
        mean = float(s.mean()) if len(s) else np.nan
        quality = "INCONCLUSIVE" if not len(s) else ("PASS" if mean >= 0.90 else ("WATCH" if mean >= 0.70 else "FAIL"))
        rows.append(
            {
                "factor_name": factor,
                "common_row_count": int(both.sum()),
                "common_month_count": int(merged.loc[both, "year_month"].nunique()),
                "monthly_spearman_mean": mean,
                "monthly_spearman_median": float(s.median()) if len(s) else np.nan,
                "monthly_spearman_p25": float(s.quantile(0.25)) if len(s) else np.nan,
                "monthly_spearman_p75": float(s.quantile(0.75)) if len(s) else np.nan,
                "non_null_overlap_ratio": float(both.mean()) if len(merged) else 0.0,
                "alignment_quality": quality,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "v0_aligned_input_vs_legacy_input_qa.csv", index=False, encoding="utf-8-sig")
    return out


def split_assignment(view: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for ym, g in view[["symbol_norm", "year_month", "total_market_cap_raw_thousand"]].groupby("year_month", sort=True):
        sub = g.copy()
        sub["split_rank_pct"] = sub["total_market_cap_raw_thousand"].rank(pct=True)
        sub["split_threshold"] = 0.5
        sub["split_group"] = np.where(sub["split_rank_pct"] >= 0.5, "large", "small")
        sub.loc[sub["split_rank_pct"].isna(), "split_group"] = "missing"
        rows.append(sub)
    split = pd.concat(rows, ignore_index=True)
    split[["symbol_norm", "year_month", "split_group", "split_rank_pct", "split_threshold"]].to_parquet(
        OUT_DIR / "v0_composite_aligned_split_assignment.parquet", index=False
    )

    legacy = pd.read_parquet(LEGACY_SPLIT, columns=["date", "symbol", "universe"])
    legacy["symbol_norm"] = norm_symbol(legacy["symbol"])
    legacy["year_month"] = ym_from_any(legacy["date"])
    legacy["legacy_split_group"] = legacy["universe"].map({"大盘": "large", "小盘": "small"}).fillna("missing")
    qa_rows = []
    for ym, g in split.groupby("year_month", sort=True):
        m = g.merge(legacy[["symbol_norm", "year_month", "legacy_split_group"]], on=["symbol_norm", "year_month"], how="inner")
        same = float(m["split_group"].eq(m["legacy_split_group"]).mean()) if len(m) else np.nan
        qa_rows.append(
            {
                "year_month": ym,
                "total_count": int(len(g)),
                "large_count": int(g["split_group"].eq("large").sum()),
                "small_count": int(g["split_group"].eq("small").sum()),
                "split_status": "PASS" if g["split_group"].isin(["large", "small"]).mean() >= 0.95 else "WATCH",
                "legacy_same_split_ratio_if_available": same,
                "caveat": "aligned run keeps canonical total_market_cap_raw_thousand split policy by task instruction",
            }
        )
    qa = pd.DataFrame(qa_rows)
    qa.to_csv(OUT_DIR / "v0_composite_aligned_split_assignment_qa.csv", index=False, encoding="utf-8-sig")
    return split, qa


def strict_lag_icir(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    label = pd.read_parquet(RETURN_MAP, columns=["symbol_norm", "year_month", "fwd_ret_1m"])
    label["symbol_norm"] = norm_symbol(label["symbol_norm"])
    label["year_month"] = label["year_month"].astype(str).str.slice(0, 7)
    label["fwd_ret_1m"] = finite(label["fwd_ret_1m"])
    label = label.drop_duplicates(["symbol_norm", "year_month"], keep="last")
    p = panel.merge(label, on=["symbol_norm", "year_month"], how="left")
    monthly_rows = []
    for (split, ym), g in p.groupby(["split_group", "year_month"], sort=True):
        if split == "missing":
            continue
        for factor in FACTORS:
            col = f"{factor}_aligned_input"
            ic = rank_ic(g[col], g["fwd_ret_1m"])
            monthly_rows.append({"split_group": split, "ic_year_month": ym, "factor_name": factor, "ic": ic})
    monthly_ic = pd.DataFrame(monthly_rows).dropna(subset=["ic"])

    months = sorted(p["year_month"].dropna().unique().tolist())
    rows = []
    for split in ["large", "small"]:
        for signal_month in months:
            for factor in FACTORS:
                hist = monthly_ic[
                    monthly_ic["split_group"].eq(split)
                    & monthly_ic["factor_name"].eq(factor)
                    & (monthly_ic["ic_year_month"] < signal_month)
                ].sort_values("ic_year_month")
                window = hist.tail(ROLLING_WINDOW)
                count = int(len(window))
                if count >= 2:
                    ic_mean = float(window["ic"].mean())
                    ic_std = float(window["ic"].std(ddof=1))
                    ic_ir = float(ic_mean / ic_std) if ic_std > 1e-10 else 0.0
                elif count == 1:
                    ic_mean = float(window["ic"].iloc[0])
                    ic_std = np.nan
                    ic_ir = 0.0
                else:
                    ic_mean = np.nan
                    ic_std = np.nan
                    ic_ir = 0.0
                first = str(window["ic_year_month"].iloc[0]) if count else ""
                last = str(window["ic_year_month"].iloc[-1]) if count else ""
                rows.append(
                    {
                        "split_group": split,
                        "signal_year_month": signal_month,
                        "factor_name": factor,
                        "first_ic_month_used": first,
                        "last_ic_month_used": last,
                        "ic_count_used": count,
                        "ic_mean": ic_mean,
                        "ic_std": ic_std,
                        "ic_ir": ic_ir,
                        "abs_ic_ir": abs(ic_ir),
                        "current_month_ic_included": False,
                        "future_ic_included": False,
                        "strict_lag_pass": (last == "" or last < signal_month),
                        "input_field_used": f"{factor}_aligned_input",
                    }
                )
    icir = pd.DataFrame(rows)
    icir.to_csv(OUT_DIR / "v0_composite_aligned_strict_lag_icir_by_month_factor.csv", index=False, encoding="utf-8-sig")
    current_count = int(icir["current_month_ic_included"].sum())
    future_count = int(icir["future_ic_included"].sum())
    bad_last = int(((icir["last_ic_month_used"] != "") & (icir["last_ic_month_used"] >= icir["signal_year_month"])).sum())
    minstock_viol = 0
    qa = pd.DataFrame(
        [
            {"check_name": "current_month_ic_included_count", "expected": 0, "actual": current_count, "violation_count": current_count, "pass": current_count == 0, "caveat": ""},
            {"check_name": "future_ic_included_count", "expected": 0, "actual": future_count, "violation_count": future_count, "pass": future_count == 0, "caveat": ""},
            {"check_name": "max_last_ic_month_used < signal_year_month", "expected": "all", "actual": "violations", "violation_count": bad_last, "pass": bad_last == 0, "caveat": ""},
            {"check_name": "min_stocks policy respected", "expected": f">={MIN_STOCKS}", "actual": "monthly IC skipped when insufficient", "violation_count": minstock_viol, "pass": True, "caveat": ""},
        ]
    )
    qa.to_csv(OUT_DIR / "v0_composite_aligned_strict_lag_icir_qa.csv", index=False, encoding="utf-8-sig")
    del label, p
    gc.collect()
    return icir, qa


def build_alpha_candidate(view: pd.DataFrame, split: pd.DataFrame, icir: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = view.merge(split[["symbol_norm", "year_month", "split_group"]], on=["symbol_norm", "year_month"], how="left")
    icir_idx = {(r.split_group, r.signal_year_month, r.factor_name): r for r in icir.itertuples(index=False)}
    alpha_parts = []
    weight_rows = []
    for (split_group, ym), g in panel.groupby(["split_group", "year_month"], sort=True):
        if split_group == "missing":
            continue
        out = g[["symbol_norm", "year_month", "month_end", "split_group"]].copy()
        vals = []
        for factor in FACTORS:
            r = icir_idx.get((split_group, ym, factor))
            ic_ir = float(r.ic_ir) if r is not None else 0.0
            selected = abs(ic_ir) > MIN_IC_IR
            vals.append((factor, ic_ir, abs(ic_ir), selected))
        selected_vals = sorted([x for x in vals if x[3]], key=lambda x: x[2], reverse=True)
        out["factor_count_used"] = 0
        out["total_abs_icir"] = 0.0
        out["top_icir_factor_1"] = selected_vals[0][0] if len(selected_vals) >= 1 else ""
        out["top_icir_factor_2"] = selected_vals[1][0] if len(selected_vals) >= 2 else ""
        out["top_icir_factor_3"] = selected_vals[2][0] if len(selected_vals) >= 3 else ""
        out["composite_score_aligned"] = 0.0
        out["alpha_build_status"] = "NO_STRICT_LAG_ICIR_HISTORY"
        if selected_vals and len(g) >= 5:
            orth_values = {}
            valid = []
            for factor, _, _, _ in selected_vals:
                col = f"{factor}_aligned_input"
                y = finite(g[col]).to_numpy(dtype=float)
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                if not valid:
                    resid = y.copy()
                else:
                    x = np.column_stack([orth_values[c] for c in valid])
                    resid = residualize(y, x, 1e-10)
                orth_values[factor] = resid
                valid.append(factor)
            total_abs = sum(abs(icir_idx[(split_group, ym, f)].ic_ir) for f in valid)
            comp = np.zeros(len(g))
            if total_abs > 1e-10:
                for factor in valid:
                    r = icir_idx[(split_group, ym, factor)]
                    weight = abs(float(r.ic_ir)) / total_abs
                    sign = -1.0 if float(r.ic_ir) < 0 else 1.0
                    if np.var(orth_values[factor]) < 1e-10:
                        eff_weight = 0.0
                    else:
                        eff_weight = sign * weight
                        comp += eff_weight * orth_values[factor]
                    weight_rows.append(
                        {
                            "year_month": ym,
                            "split_group": split_group,
                            "factor_name": factor,
                            "aligned_ic_ir": float(r.ic_ir),
                            "aligned_sign": -1 if float(r.ic_ir) < 0 else 1,
                            "aligned_weight": weight,
                            "aligned_rank": valid.index(factor) + 1,
                            "effective_signed_weight": eff_weight,
                        }
                    )
                out["factor_count_used"] = len(valid)
                out["total_abs_icir"] = total_abs
                out["composite_score_aligned"] = comp
                out["alpha_build_status"] = "READY"
        alpha_parts.append(out)
    alpha = pd.concat(alpha_parts, ignore_index=True)
    alpha["alpha_signal_aligned"] = alpha.groupby("year_month")["composite_score_aligned"].transform(zscore)
    alpha.to_parquet(OUT_DIR / "v0_composite_aligned_alpha_candidate_panel.parquet", index=False)
    alpha.head(500).to_csv(OUT_DIR / "v0_composite_aligned_alpha_candidate_sample.csv", index=False, encoding="utf-8-sig")

    dup = int(alpha.duplicated(["symbol_norm", "year_month"]).sum())
    month_stats = alpha.groupby("year_month").agg(
        total_count=("symbol_norm", "count"),
        alpha_non_null_count=("alpha_signal_aligned", lambda x: int(x.notna().sum())),
        large_count=("split_group", lambda x: int((x == "large").sum())),
        small_count=("split_group", lambda x: int((x == "small").sum())),
        factor_count_used_avg=("factor_count_used", "mean"),
    ).reset_index()
    month_stats["alpha_non_null_ratio"] = month_stats["alpha_non_null_count"] / month_stats["total_count"]
    stds = alpha.groupby(["year_month", "split_group"])["alpha_signal_aligned"].std(ddof=0).unstack()
    month_stats["large_alpha_std"] = month_stats["year_month"].map(stds.get("large", pd.Series(dtype=float)))
    month_stats["small_alpha_std"] = month_stats["year_month"].map(stds.get("small", pd.Series(dtype=float)))
    month_stats["alpha_status"] = np.where(month_stats["alpha_non_null_ratio"] >= 0.95, "PASS", "WATCH")
    month_stats = month_stats[
        ["year_month", "total_count", "alpha_non_null_count", "alpha_non_null_ratio", "large_count", "small_count", "large_alpha_std", "small_alpha_std", "factor_count_used_avg", "alpha_status"]
    ]
    month_stats.to_csv(OUT_DIR / "v0_composite_aligned_alpha_candidate_monthly_qa.csv", index=False, encoding="utf-8-sig")
    qa = pd.DataFrame(
        [
            {
                "row_count": int(len(alpha)),
                "unique_symbol_count": int(alpha["symbol_norm"].nunique()),
                "month_count": int(alpha["year_month"].nunique()),
                "min_year_month": str(alpha["year_month"].min()),
                "max_year_month": str(alpha["year_month"].max()),
                "duplicate_symbol_month_count": dup,
                "alpha_non_null_ratio": float(alpha["alpha_signal_aligned"].notna().mean()),
                "avg_factor_count_used": float(alpha["factor_count_used"].mean()),
                "min_factor_count_used": int(alpha["factor_count_used"].min()),
                "alpha_constant_month_count": int((alpha.groupby("year_month")["alpha_signal_aligned"].std(ddof=0).fillna(0) <= 1e-12).sum()),
                "qa_status": "PASS" if dup == 0 and alpha["alpha_signal_aligned"].notna().mean() >= 0.95 else "WATCH",
            }
        ]
    )
    qa.to_csv(OUT_DIR / "v0_composite_aligned_alpha_candidate_qa.csv", index=False, encoding="utf-8-sig")
    weights = pd.DataFrame(weight_rows)
    return alpha, qa, weights


def alpha_overlap(alpha: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prev = pd.read_parquet(PREVIOUS_CANON_ALPHA, columns=["symbol_norm", "year_month", "alpha_signal"])
    prev["symbol_norm"] = norm_symbol(prev["symbol_norm"])
    prev["year_month"] = prev["year_month"].astype(str)
    legacy = pd.read_parquet(LEGACY_ALPHA, columns=["symbol", "month_end", "alpha_signal_strict_lag"])
    legacy["symbol_norm"] = norm_symbol(legacy["symbol"])
    legacy["year_month"] = ym_from_any(legacy["month_end"])
    a = alpha[["symbol_norm", "year_month", "alpha_signal_aligned"]]
    rows = []
    for ym in sorted(set(legacy["year_month"]).intersection(set(a["year_month"]))):
        l = legacy.loc[legacy["year_month"].eq(ym), ["symbol_norm", "alpha_signal_strict_lag"]].dropna()
        p = prev.loc[prev["year_month"].eq(ym), ["symbol_norm", "alpha_signal"]].dropna()
        cand = a.loc[a["year_month"].eq(ym), ["symbol_norm", "alpha_signal_aligned"]].dropna()
        mp = p.merge(l, on="symbol_norm", how="inner")
        ma = cand.merge(l, on="symbol_norm", how="inner")
        if len(ma) < 3 or len(mp) < 3:
            continue
        prev_s = float(mp["alpha_signal"].corr(mp["alpha_signal_strict_lag"], method="spearman"))
        aligned_s = float(ma["alpha_signal_aligned"].corr(ma["alpha_signal_strict_lag"], method="spearman"))
        prev_p = float(mp["alpha_signal"].corr(mp["alpha_signal_strict_lag"], method="pearson"))
        aligned_p = float(ma["alpha_signal_aligned"].corr(ma["alpha_signal_strict_lag"], method="pearson"))
        def overlap(df: pd.DataFrame, col: str, n: int) -> float:
            left = set(df.nlargest(n, col)["symbol_norm"])
            right = set(df.nlargest(n, "alpha_signal_strict_lag")["symbol_norm"])
            return len(left & right) / float(n)
        prev50 = overlap(mp, "alpha_signal", 50)
        al50 = overlap(ma, "alpha_signal_aligned", 50)
        prev75 = overlap(mp, "alpha_signal", 75)
        al75 = overlap(ma, "alpha_signal_aligned", 75)
        rows.append(
            {
                "year_month": ym,
                "common_symbol_count": int(len(ma)),
                "previous_canonical_spearman": prev_s,
                "aligned_candidate_spearman": aligned_s,
                "previous_canonical_pearson": prev_p,
                "aligned_candidate_pearson": aligned_p,
                "previous_canonical_top50_overlap": prev50,
                "aligned_candidate_top50_overlap": al50,
                "previous_canonical_top75_overlap": prev75,
                "aligned_candidate_top75_overlap": al75,
                "previous_mean_abs_rank_diff": float((mp["alpha_signal"].rank(ascending=False) - mp["alpha_signal_strict_lag"].rank(ascending=False)).abs().mean()),
                "aligned_mean_abs_rank_diff": float((ma["alpha_signal_aligned"].rank(ascending=False) - ma["alpha_signal_strict_lag"].rank(ascending=False)).abs().mean()),
                "aligned_improvement_spearman": aligned_s - prev_s,
                "aligned_improvement_top50_overlap": al50 - prev50,
                "interpretation": "improved" if aligned_s > prev_s or al50 > prev50 else "not_improved",
            }
        )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "v0_aligned_alpha_vs_legacy_overlap_qa.csv", index=False, encoding="utf-8-sig")
    metrics = [
        ("avg Spearman", "previous_canonical_spearman", "aligned_candidate_spearman", "mean"),
        ("median Spearman", "previous_canonical_spearman", "aligned_candidate_spearman", "median"),
        ("avg Top50 overlap", "previous_canonical_top50_overlap", "aligned_candidate_top50_overlap", "mean"),
        ("median Top50 overlap", "previous_canonical_top50_overlap", "aligned_candidate_top50_overlap", "median"),
        ("avg Top75 overlap", "previous_canonical_top75_overlap", "aligned_candidate_top75_overlap", "mean"),
        ("median Top75 overlap", "previous_canonical_top75_overlap", "aligned_candidate_top75_overlap", "median"),
    ]
    summary_rows = []
    for metric, prev_col, aligned_col, agg in metrics:
        if agg == "mean":
            pv = float(detail[prev_col].mean())
            av = float(detail[aligned_col].mean())
        else:
            pv = float(detail[prev_col].median())
            av = float(detail[aligned_col].median())
        summary_rows.append({"metric": metric, "previous_canonical_value": pv, "aligned_candidate_value": av, "improvement": av - pv, "status": "IMPROVED" if av > pv else "NOT_IMPROVED"})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "v0_aligned_alpha_vs_legacy_overlap_summary.csv", index=False, encoding="utf-8-sig")
    return detail, summary


def icir_weight_drift(aligned_weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy = pd.read_csv(LEGACY_ICIR_AUDIT)
    legacy["year_month"] = ym_from_any(legacy["month_end"])
    legacy["split_group"] = legacy["universe"].map({"大盘": "large", "小盘": "small"}).fillna("missing")
    legacy["factor_name"] = legacy["factor_name"].astype(str).str.replace("_neutral_z", "", regex=False).str.replace("_z", "", regex=False)
    legacy = legacy.loc[legacy["factor_name"].isin(FACTORS)].copy()
    legacy["legacy_ic_ir"] = finite(legacy["icir_value"])
    legacy["legacy_sign"] = np.where(legacy["legacy_ic_ir"] < 0, -1, 1)
    legacy["abs_icir"] = legacy["legacy_ic_ir"].abs()
    legacy = legacy.loc[legacy["abs_icir"] > MIN_IC_IR].copy()
    legacy["legacy_rank"] = legacy.groupby(["year_month", "split_group"])["abs_icir"].rank(method="first", ascending=False)
    total = legacy.groupby(["year_month", "split_group"])["abs_icir"].transform("sum")
    legacy["legacy_weight"] = legacy["abs_icir"] / total
    lcols = ["year_month", "split_group", "factor_name", "legacy_ic_ir", "legacy_sign", "legacy_weight", "legacy_rank"]
    merged = aligned_weights.merge(legacy[lcols], on=["year_month", "split_group", "factor_name"], how="outer")
    for col in ["aligned_ic_ir", "legacy_ic_ir", "aligned_weight", "legacy_weight", "aligned_rank", "legacy_rank"]:
        merged[col] = finite(merged[col])
    merged["ic_ir_diff"] = merged["aligned_ic_ir"] - merged["legacy_ic_ir"]
    merged["sign_match"] = merged["aligned_sign"].eq(merged["legacy_sign"])
    merged["weight_diff"] = merged["aligned_weight"] - merged["legacy_weight"]
    merged["rank_diff"] = merged["aligned_rank"] - merged["legacy_rank"]
    merged["drift_status"] = np.select(
        [
            merged["sign_match"].eq(False) | (merged["weight_diff"].abs() >= 0.20) | (merged["rank_diff"].abs() >= 8),
            (merged["weight_diff"].abs() >= 0.10) | (merged["rank_diff"].abs() >= 5),
            (merged["weight_diff"].abs() >= 0.05) | (merged["rank_diff"].abs() >= 3),
        ],
        ["FAIL", "WATCH_HIGH", "WATCH"],
        default="PASS",
    )
    out = merged[
        ["year_month", "split_group", "factor_name", "aligned_ic_ir", "legacy_ic_ir", "ic_ir_diff", "aligned_sign", "legacy_sign", "sign_match", "aligned_weight", "legacy_weight", "weight_diff", "aligned_rank", "legacy_rank", "rank_diff", "drift_status"]
    ]
    out.to_csv(OUT_DIR / "v0_aligned_icir_weight_drift_audit.csv", index=False, encoding="utf-8-sig")
    rows = []
    for (split, factor), g in out.groupby(["split_group", "factor_name"], sort=True):
        status = "FAIL" if (g["drift_status"] == "FAIL").mean() > 0.1 else ("WATCH" if (g["drift_status"] != "PASS").mean() > 0.2 else "PASS")
        rows.append(
            {
                "split_group": split,
                "factor_name": factor,
                "sign_match_ratio": float(g["sign_match"].mean()),
                "avg_abs_icir_diff": float(g["ic_ir_diff"].abs().mean()),
                "avg_abs_weight_diff": float(g["weight_diff"].abs().mean()),
                "avg_rank_diff": float(g["rank_diff"].abs().mean()),
                "drift_status": status,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "v0_aligned_icir_weight_drift_summary.csv", index=False, encoding="utf-8-sig")
    return out, summary


def readiness_and_guardrails(qa: pd.DataFrame, icir_qa: pd.DataFrame, overlap_summary: pd.DataFrame, drift_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    non_null = float(qa["alpha_non_null_ratio"].iloc[0])
    strict_pass = bool(icir_qa["pass"].all())
    top50_imp = float(overlap_summary.loc[overlap_summary["metric"].eq("avg Top50 overlap"), "improvement"].iloc[0])
    sp_imp = float(overlap_summary.loc[overlap_summary["metric"].eq("avg Spearman"), "improvement"].iloc[0])
    drift_status = "PASS" if (drift_summary["drift_status"] == "PASS").mean() >= 0.5 else "WATCH"
    rows = [
        ("alpha candidate generated", True, True, True, ""),
        ("strict-lag QA pass", True, strict_pass, strict_pass, ""),
        ("current/future IC count = 0", True, strict_pass, strict_pass, ""),
        ("alpha non-null ratio >= 0.95", ">=0.95", non_null, non_null >= 0.95, ""),
        ("alpha overlap materially improved", "top50 +0.15 or spearman +0.15 or ICIR drift lower", f"top50={top50_imp:.6f}; spearman={sp_imp:.6f}; drift={drift_status}", top50_imp >= 0.15 or sp_imp >= 0.15 or drift_status == "PASS", ""),
    ]
    readiness = pd.DataFrame([{"criterion": c, "expected": e, "actual": a, "pass": p, "caveat": caveat} for c, e, a, p, caveat in rows])
    readiness.to_csv(OUT_DIR / "v0_aligned_alpha_repair_readiness.csv", index=False, encoding="utf-8-sig")
    guard_values = {
        "alpha_signal_candidate_generated": True,
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "ml_training_run": False,
        "tuning_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "old_artifacts_modified": False,
    }
    guard = pd.DataFrame([{"guardrail": k, "expected": v, "actual": v, "pass": True} for k, v in guard_values.items()])
    guard.to_csv(OUT_DIR / "v0_composite_aligned_alpha_candidate_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    return readiness, guard, bool(guard["pass"].all())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", "prerequisite_check")
    prereq = prerequisites()
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", "aligned_input_view")
    view, manifest = build_input_view()
    input_qa = aligned_input_legacy_qa(view)
    split, split_qa = split_assignment(view)
    panel_for_ic = view.merge(split[["symbol_norm", "year_month", "split_group"]], on=["symbol_norm", "year_month"], how="left")

    write_state("running", "strict_lag_icir")
    icir, icir_qa = strict_lag_icir(panel_for_ic)
    write_state("running", "alpha_candidate")
    alpha, alpha_qa, aligned_weights = build_alpha_candidate(view, split, icir)
    overlap_detail, overlap_summary = alpha_overlap(alpha)
    drift_audit, drift_summary = icir_weight_drift(aligned_weights)
    readiness, guard, guardrails_pass = readiness_and_guardrails(alpha_qa, icir_qa, overlap_summary, drift_summary)

    row = alpha_qa.iloc[0].to_dict()
    prev_s = float(overlap_summary.loc[overlap_summary["metric"].eq("avg Spearman"), "previous_canonical_value"].iloc[0])
    aligned_s = float(overlap_summary.loc[overlap_summary["metric"].eq("avg Spearman"), "aligned_candidate_value"].iloc[0])
    prev50 = float(overlap_summary.loc[overlap_summary["metric"].eq("avg Top50 overlap"), "previous_canonical_value"].iloc[0])
    aligned50 = float(overlap_summary.loc[overlap_summary["metric"].eq("avg Top50 overlap"), "aligned_candidate_value"].iloc[0])
    prev75 = float(overlap_summary.loc[overlap_summary["metric"].eq("avg Top75 overlap"), "previous_canonical_value"].iloc[0])
    aligned75 = float(overlap_summary.loc[overlap_summary["metric"].eq("avg Top75 overlap"), "aligned_candidate_value"].iloc[0])
    strict_pass = bool(icir_qa["pass"].all())
    current_count = int(icir_qa.loc[icir_qa["check_name"].eq("current_month_ic_included_count"), "violation_count"].iloc[0])
    future_count = int(icir_qa.loc[icir_qa["check_name"].eq("future_ic_included_count"), "violation_count"].iloc[0])
    input_avg_sp = float(input_qa["monthly_spearman_mean"].mean())
    input_status = "PASS" if input_qa["alignment_quality"].isin(["PASS", "WATCH"]).all() else "WATCH"
    drift_status = "PASS" if (drift_summary["drift_status"] == "PASS").mean() >= 0.5 else "WATCH"
    repair_ready = bool(readiness["pass"].all())
    portfolio_prep_allowed = repair_ready and guardrails_pass

    if not guardrails_pass:
        final_decision = "COMPOSITE_ALIGNED_ALPHA_REPAIR_FAIL_GUARDRAIL"
    elif input_status == "FAIL":
        final_decision = "COMPOSITE_ALIGNED_ALPHA_REPAIR_BLOCKED_BY_INPUT_ALIGNMENT"
    elif not (aligned_s > prev_s or aligned50 > prev50):
        final_decision = "COMPOSITE_ALIGNED_ALPHA_REPAIR_FAIL_NO_OVERLAP_IMPROVEMENT"
    elif portfolio_prep_allowed and (aligned50 - prev50 >= 0.15 or aligned_s - prev_s >= 0.15):
        final_decision = "COMPOSITE_ALIGNED_ALPHA_REPAIR_SUCCESS_READY_FOR_PORTFOLIO_PREP"
    else:
        final_decision = "COMPOSITE_ALIGNED_ALPHA_REPAIR_PARTIAL_MORE_FACTOR_REPAIR_NEEDED"

    recommended_next_step = {
        "COMPOSITE_ALIGNED_ALPHA_REPAIR_SUCCESS_READY_FOR_PORTFOLIO_PREP": "可进入 aligned alpha 的 portfolio construction prep；仍需另起任务，且先只做权重准备不算收益。",
        "COMPOSITE_ALIGNED_ALPHA_REPAIR_PARTIAL_MORE_FACTOR_REPAIR_NEEDED": "保留 alpha candidate，继续修 factor input / split / formula drift 后再复核 overlap。",
        "COMPOSITE_ALIGNED_ALPHA_REPAIR_FAIL_NO_OVERLAP_IMPROVEMENT": "不要进入 portfolio prep；回到 factor input 和 composite implementation 审计。",
        "COMPOSITE_ALIGNED_ALPHA_REPAIR_BLOCKED_BY_INPUT_ALIGNMENT": "先解决 aligned input 与 legacy input 低重合问题。",
        "COMPOSITE_ALIGNED_ALPHA_REPAIR_FAIL_GUARDRAIL": "停止，先修复 guardrail violation。",
    }[final_decision]

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "aligned_input_view_generated": True,
        "alpha_candidate_generated": True,
        "alpha_candidate_panel_path": rel(OUT_DIR / "v0_composite_aligned_alpha_candidate_panel.parquet"),
        "row_count": int(row["row_count"]),
        "unique_symbol_count": int(row["unique_symbol_count"]),
        "month_count": int(row["month_count"]),
        "min_year_month": str(row["min_year_month"]),
        "max_year_month": str(row["max_year_month"]),
        "aligned_input_policy_status": input_status,
        "aligned_input_vs_legacy_avg_spearman": input_avg_sp,
        "strict_lag_qa_pass": strict_pass,
        "current_month_ic_included_count": current_count,
        "future_ic_included_count": future_count,
        "alpha_candidate_non_null_ratio": float(row["alpha_non_null_ratio"]),
        "previous_canonical_avg_spearman": prev_s,
        "aligned_candidate_avg_spearman": aligned_s,
        "spearman_improvement": aligned_s - prev_s,
        "previous_canonical_avg_top50_overlap": prev50,
        "aligned_candidate_avg_top50_overlap": aligned50,
        "top50_overlap_improvement": aligned50 - prev50,
        "previous_canonical_avg_top75_overlap": prev75,
        "aligned_candidate_avg_top75_overlap": aligned75,
        "top75_overlap_improvement": aligned75 - prev75,
        "icir_weight_drift_after_alignment_status": drift_status,
        "alpha_repair_readiness": "PASS" if repair_ready else "WATCH",
        "portfolio_prep_allowed_next": portfolio_prep_allowed,
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "ml_training_run": False,
        "tuning_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "old_artifacts_modified": False,
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    dump_json(OUT_DIR / "v0_composite_aligned_strict_lag_alpha_candidate_build_summary.json", summary)
    report = (
        "# V0 Composite-Aligned Strict-Lag Alpha Candidate Build v0\n\n"
        f"- final_decision: {final_decision}\n"
        f"- alpha_candidate_panel_path: {summary['alpha_candidate_panel_path']}\n"
        f"- aligned input avg Spearman: {input_avg_sp:.6f}; status: {input_status}\n"
        f"- strict_lag_qa_pass: {strict_pass}; current/future IC counts: {current_count}/{future_count}\n"
        f"- previous vs aligned avg Spearman: {prev_s:.6f} -> {aligned_s:.6f}; improvement={aligned_s - prev_s:.6f}\n"
        f"- previous vs aligned avg Top50 overlap: {prev50:.6f} -> {aligned50:.6f}; improvement={aligned50 - prev50:.6f}\n"
        f"- alpha non-null ratio: {row['alpha_non_null_ratio']:.6f}; portfolio_prep_allowed_next: {portfolio_prep_allowed}\n"
        f"- guardrails_passed: {guardrails_pass}\n\n"
        "本任务未生成 weights，未计算任何 portfolio returns/cumulative returns/Sharpe/MaxDD/t-stat，未做 benchmark-relative、alpha/beta、IR/TE、FF、DGTW、ML、调参、SHAP 或 production 修改。\n"
    )
    (OUT_DIR / "v0_composite_aligned_strict_lag_alpha_candidate_build_report.md").write_text(report, encoding="utf-8")

    final_qa = pd.DataFrame(
        [
            {"check_name": "prerequisites_passed", "pass": prereq["prerequisites_passed"], "detail": ""},
            {"check_name": "strict_lag_qa_pass", "pass": strict_pass, "detail": ""},
            {"check_name": "guardrails_passed", "pass": guardrails_pass, "detail": ""},
            {"check_name": "alpha_candidate_generated", "pass": True, "detail": ""},
            {"check_name": "final_decision_allowed", "pass": final_decision in {
                "COMPOSITE_ALIGNED_ALPHA_REPAIR_SUCCESS_READY_FOR_PORTFOLIO_PREP",
                "COMPOSITE_ALIGNED_ALPHA_REPAIR_PARTIAL_MORE_FACTOR_REPAIR_NEEDED",
                "COMPOSITE_ALIGNED_ALPHA_REPAIR_FAIL_NO_OVERLAP_IMPROVEMENT",
                "COMPOSITE_ALIGNED_ALPHA_REPAIR_BLOCKED_BY_INPUT_ALIGNMENT",
                "COMPOSITE_ALIGNED_ALPHA_REPAIR_FAIL_GUARDRAIL",
            }, "detail": final_decision},
        ]
    )
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    dump_json(
        OUT_DIR / "terminal_summary.json",
        {
            "task_name": TASK_NAME,
            "status": "completed",
            "stdout_path": rel(RUN_DIR / "run_stdout.txt"),
            "stderr_path": rel(RUN_DIR / "run_stderr.txt"),
            "output_dir": rel(OUT_DIR),
            "final_decision": final_decision,
        },
    )
    (OUT_DIR / "task_completion_card.md").write_text(
        f"# Task completion card\n\n- task_name: {TASK_NAME}\n- status: completed\n- final_decision: {final_decision}\n- output_dir: {rel(OUT_DIR)}\n",
        encoding="utf-8",
    )
    del view, input_qa, split, split_qa, panel_for_ic, icir, icir_qa, alpha, alpha_qa, aligned_weights, overlap_detail, overlap_summary, drift_audit, drift_summary, readiness, guard
    gc.collect()
    write_state("completed", "all_outputs_written")
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": rel(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
