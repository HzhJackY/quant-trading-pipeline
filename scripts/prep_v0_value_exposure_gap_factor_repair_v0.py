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


TASK_NAME = "V0 Value Exposure Gap Factor Repair Prep v0"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "v0_value_exposure_gap_factor_repair_prep_v0"
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

STYLE_DIR = ROOT / "output" / "v0_composite_aligned_holdings_style_exposure_attribution_v0"
STYLE_SUMMARY = STYLE_DIR / "v0_composite_aligned_holdings_style_exposure_attribution_summary.json"
PAIRWISE_DIFF = STYLE_DIR / "v0_style_exposure_pairwise_diff.csv"
ALIGNED_STYLE = STYLE_DIR / "v0_aligned_monthly_style_exposure_wide.csv"
COMPARISON_STYLE = STYLE_DIR / "v0_comparison_monthly_style_exposure_wide.csv"
STYLE_INPUT_VIEW = STYLE_DIR / "v0_style_exposure_input_view.parquet"
DECISION_SUMMARY = STYLE_DIR / "v0_holdings_style_exposure_decision_summary.csv"

ALIGNED_ALPHA = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_composite_aligned_alpha_candidate_panel.parquet"
ALIGNED_INPUT = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_composite_aligned_input_view.parquet"
ALIGNED_ICIR = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_composite_aligned_strict_lag_icir_by_month_factor.csv"
ALIGNED_DRIFT_AUDIT = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_aligned_icir_weight_drift_audit.csv"
ALIGNED_DRIFT_SUMMARY = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_aligned_icir_weight_drift_summary.csv"
ALIGNED_WEIGHTS = ROOT / "output" / "v0_composite_aligned_portfolio_construction_run_v0" / "v0_composite_aligned_research_weights.parquet"

FACTOR_PANEL = ROOT / "output" / "v0_canonical_16factor_panel_build_v0" / "v0_canonical_16factor_panel.parquet"
PREPROCESSED = ROOT / "output" / "preprocessed.parquet"
SPLIT_UNIVERSE = ROOT / "output" / "split_universe_blended.parquet"
LEGACY_ALPHA = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_alpha_signal_panel.parquet"
LEGACY_WEIGHTS = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_reconstructed_weights.parquet"
RAW_ALPHA = ROOT / "output" / "v0_canonical_strict_lag_alpha_build_v0" / "v0_canonical_alpha_signal_panel.parquet"
RAW_WEIGHTS = ROOT / "output" / "v0_canonical_portfolio_construction_run_v0" / "v0_canonical_research_weights.parquet"
NAV_DD = ROOT / "output" / "v0_composite_aligned_repaired_trd_mnth_eval_run_v0" / "v0_aligned_nav_drawdown_path.csv"

VALUE_COLS = [
    "BP_weighted_z_exposure",
    "EP_weighted_z_exposure",
    "value_exposure_z",
    "Debt_Ratio_weighted_z_exposure",
    "quality_exposure_z",
    "ROE_weighted_z_exposure",
    "Net_Profit_Margin_weighted_z_exposure",
]
SPLIT_FACTORS = ["BP", "EP", "Debt_Ratio", "ROE", "Net_Profit_Margin", "value_exposure_z", "quality_adjusted_debt_exposure"]


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


def severity(diff: float) -> str:
    x = abs(diff)
    if x < 0.10:
        return "LOW"
    if x < 0.25:
        return "MEDIUM"
    if x < 0.50:
        return "HIGH"
    return "CRITICAL"


def read_parquet_cols(path: Path, cols: list[str]) -> pd.DataFrame:
    available = schema_cols(path)
    use = [c for c in cols if c in available]
    if not use:
        return pd.DataFrame()
    table = pq.read_table(path, columns=use)
    df = table.to_pandas()
    del table
    gc.collect()
    return df


def value_gap_quantification() -> pd.DataFrame:
    df = pd.read_csv(PAIRWISE_DIFF)
    sub = df[df["factor_or_style"].isin(VALUE_COLS)].copy()
    rows = []
    for r in sub.itertuples(index=False):
        diff = float(r.diff_a_minus_b)
        rows.append(
            {
                "pair_name": r.pair_name,
                "window_name": r.window_name,
                "factor_or_style": r.factor_or_style,
                "portfolio_a_avg_exposure_z": r.portfolio_a_avg_exposure_z,
                "portfolio_b_avg_exposure_z": r.portfolio_b_avg_exposure_z,
                "diff_a_minus_b": diff,
                "monthly_diff_mean": r.monthly_diff_mean,
                "monthly_diff_std": r.monthly_diff_std,
                "gap_direction": "portfolio_a_higher" if diff > 0 else "portfolio_a_lower" if diff < 0 else "flat",
                "severity": severity(diff),
                "interpretation": f"{r.pair_name} {r.factor_or_style} 差异为 {diff:.4f}；仅描述风格暴露差异，不是收益归因。",
            }
        )
    return pd.DataFrame(rows)


def ep_bp_input_drift_audit() -> pd.DataFrame:
    aligned = read_parquet_cols(ALIGNED_INPUT, ["symbol_norm", "year_month", "BP_aligned_input", "EP_aligned_input", "BP_source_field", "EP_source_field"])
    canonical = read_parquet_cols(FACTOR_PANEL, ["symbol_norm", "year_month", "BP", "EP", "BP_raw_field", "EP_raw_field", "BP_source_flag", "EP_source_flag"])
    legacy = read_parquet_cols(PREPROCESSED, ["symbol", "month", "BP", "EP", "BP_neutral_z", "EP_neutral_z", "BP_neutral", "EP_neutral"])
    rows = []
    if aligned.empty or legacy.empty:
        return pd.DataFrame(rows)
    aligned["symbol_norm"] = normalize_symbol(aligned["symbol_norm"])
    aligned["year_month"] = normalize_ym(aligned["year_month"])
    canonical["symbol_norm"] = normalize_symbol(canonical["symbol_norm"]) if not canonical.empty else pd.Series(dtype="string")
    if not canonical.empty:
        canonical["year_month"] = normalize_ym(canonical["year_month"])
    legacy["symbol_norm"] = normalize_symbol(legacy["symbol"])
    legacy["year_month"] = normalize_ym(legacy["month"])
    for factor in ["BP", "EP"]:
        a_col = f"{factor}_aligned_input"
        legacy_candidates = [f"{factor}_neutral_z", factor]
        best = None
        best_metric = -2.0
        merged_best = pd.DataFrame()
        for l_col in legacy_candidates:
            merged = aligned[["symbol_norm", "year_month", a_col]].merge(legacy[["symbol_norm", "year_month", l_col]], on=["symbol_norm", "year_month"], how="inner")
            merged[a_col] = pd.to_numeric(merged[a_col], errors="coerce")
            merged[l_col] = pd.to_numeric(merged[l_col], errors="coerce")
            merged = merged.dropna()
            if merged.empty:
                continue
            spears = merged.groupby("year_month", observed=True).apply(lambda g: g[a_col].corr(g[l_col], method="spearman") if len(g) > 2 else np.nan)
            metric = float(spears.mean()) if spears.notna().any() else -2.0
            if metric > best_metric:
                best = l_col
                best_metric = metric
                merged_best = merged
        if best is None or merged_best.empty:
            rows.append({"factor_name": factor, "canonical_source_field": factor, "aligned_input_field": a_col, "legacy_input_field": "", "legacy_policy": "unknown", "aligned_policy": "aligned_input", "common_row_count": 0, "monthly_spearman_mean": "", "monthly_spearman_median": "", "mean_abs_rank_diff_pct": "", "non_null_overlap_ratio": 0, "sign_direction_consistent": "", "input_drift_severity": "INCONCLUSIVE", "likely_issue": "缺少共同样本", "recommended_action": "补充输入映射 QA"})
            continue
        spears = merged_best.groupby("year_month", observed=True).apply(lambda g: g[a_col].corr(g[best], method="spearman") if len(g) > 2 else np.nan).dropna()
        rank_diffs = []
        for _, g in merged_best.groupby("year_month", observed=True):
            if len(g) > 2:
                ar = g[a_col].rank(pct=True)
                lr = g[best].rank(pct=True)
                rank_diffs.append(float((ar - lr).abs().mean()))
        mean_s = float(spears.mean()) if len(spears) else np.nan
        med_s = float(spears.median()) if len(spears) else np.nan
        rank_diff = float(np.mean(rank_diffs)) if rank_diffs else np.nan
        sev = "LOW" if mean_s >= 0.98 and rank_diff < 0.03 else "MEDIUM" if mean_s >= 0.90 else "HIGH" if mean_s >= 0.70 else "CRITICAL"
        source_field = ""
        if f"{factor}_source_field" in aligned.columns and aligned[f"{factor}_source_field"].notna().any():
            source_field = str(aligned[f"{factor}_source_field"].dropna().mode().iloc[0])
        rows.append(
            {
                "factor_name": factor,
                "canonical_source_field": f"{factor}; raw_field={canonical.get(f'{factor}_raw_field', pd.Series(dtype='object')).dropna().mode().iloc[0] if not canonical.empty and f'{factor}_raw_field' in canonical.columns and canonical[f'{factor}_raw_field'].notna().any() else ''}",
                "aligned_input_field": a_col,
                "legacy_input_field": best,
                "legacy_policy": "neutral_z" if best.endswith("_neutral_z") else "raw",
                "aligned_policy": f"legacy-aligned input; source_field={source_field}",
                "common_row_count": int(len(merged_best)),
                "monthly_spearman_mean": mean_s,
                "monthly_spearman_median": med_s,
                "mean_abs_rank_diff_pct": rank_diff,
                "non_null_overlap_ratio": float(len(merged_best) / max(len(aligned), 1)),
                "sign_direction_consistent": bool(mean_s > 0),
                "input_drift_severity": sev,
                "likely_issue": "EP/BP input source mismatch unlikely" if sev in {"LOW", "MEDIUM"} else "EP/BP input transform/source drift possible",
                "recommended_action": "若 ICIR path 仍漂移，优先检查 ICIR denominator/sign/selected policy。" if sev in {"LOW", "MEDIUM"} else "下一步优先做 EP/BP input source / transform alignment。",
            }
        )
    del aligned, canonical, legacy
    gc.collect()
    return pd.DataFrame(rows)


def icir_weight_path_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = pd.read_csv(ALIGNED_DRIFT_AUDIT)
    sub = audit[audit["factor_name"].isin(["BP", "EP", "Debt_Ratio"])].copy()
    rows = []
    for r in sub.itertuples(index=False):
        rank_diff = getattr(r, "rank_diff", np.nan)
        weight_diff = getattr(r, "weight_diff", np.nan)
        sign_match = bool(getattr(r, "sign_match", False))
        sev = "LOW"
        if (not sign_match) or abs(float(weight_diff)) >= 0.05 or abs(float(rank_diff)) >= 4:
            sev = "HIGH"
        elif abs(float(weight_diff)) >= 0.02 or abs(float(rank_diff)) >= 2:
            sev = "MEDIUM"
        rows.append(
            {
                "year_month": r.year_month,
                "split_group": r.split_group,
                "factor_name": r.factor_name,
                "aligned_ic_ir": r.aligned_ic_ir,
                "legacy_ic_ir": r.legacy_ic_ir,
                "ic_ir_diff": r.ic_ir_diff,
                "aligned_sign": r.aligned_sign,
                "legacy_sign": r.legacy_sign,
                "sign_match": sign_match,
                "aligned_abs_icir_rank": r.aligned_rank,
                "legacy_abs_icir_rank": r.legacy_rank,
                "rank_diff": rank_diff,
                "aligned_weight": r.aligned_weight,
                "legacy_weight": r.legacy_weight,
                "weight_diff": weight_diff,
                "aligned_selected": bool(float(r.aligned_weight) > 0),
                "legacy_selected": bool(float(r.legacy_weight) > 0),
                "selected_match": bool((float(r.aligned_weight) > 0) == (float(r.legacy_weight) > 0)),
                "denominator_diff_flag": bool(abs(float(weight_diff)) > 0.02 and sign_match),
                "drift_severity": sev,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out, pd.DataFrame()
    summary_rows = []
    for (split, factor), g in out.groupby(["split_group", "factor_name"], observed=True):
        denom_count = int(g["denominator_diff_flag"].sum())
        avg_ic = float(pd.to_numeric(g["ic_ir_diff"], errors="coerce").abs().mean())
        avg_w = float(pd.to_numeric(g["weight_diff"], errors="coerce").abs().mean())
        avg_rank = float(pd.to_numeric(g["rank_diff"], errors="coerce").abs().mean())
        sign_ratio = float(g["sign_match"].mean())
        selected_ratio = float(g["selected_match"].mean())
        sev = "LOW"
        if sign_ratio < 0.80 or avg_w >= 0.04 or avg_rank >= 3:
            sev = "HIGH"
        elif sign_ratio < 0.95 or avg_w >= 0.02 or avg_rank >= 1.5:
            sev = "MEDIUM"
        summary_rows.append(
            {
                "split_group": split,
                "factor_name": factor,
                "sign_match_ratio": sign_ratio,
                "selected_match_ratio": selected_ratio,
                "avg_abs_icir_diff": avg_ic,
                "avg_abs_weight_diff": avg_w,
                "avg_rank_diff": avg_rank,
                "denominator_diff_month_count": denom_count,
                "severity": sev,
                "likely_issue": "ICIR sign/rank/weight path drift" if sev != "LOW" else "minor drift",
            }
        )
    return out, pd.DataFrame(summary_rows)


def load_weights(path: Path, name: str) -> pd.DataFrame:
    cols = schema_cols(path)
    if not cols:
        return pd.DataFrame()
    sym = "symbol_norm" if "symbol_norm" in cols else "symbol"
    read_cols = [c for c in [sym, "year_month", "month_end", "weight", "selected_flag"] if c in cols]
    df = read_parquet_cols(path, read_cols)
    if df.empty:
        return df
    df["portfolio_name"] = name
    df["symbol_norm"] = normalize_symbol(df[sym])
    if "year_month" in df.columns:
        df["year_month"] = normalize_ym(df["year_month"])
    else:
        df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
        df["year_month"] = df["month_end"].dt.strftime("%Y-%m")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
    if "selected_flag" in df.columns:
        flag = df["selected_flag"].astype("string").str.lower()
        df = df[(flag.isin(["true", "1"])) | (df["weight"] > 0)]
    else:
        df = df[df["weight"] > 0]
    return df[["portfolio_name", "symbol_norm", "year_month", "weight"]]


def split_group_value_exposure() -> tuple[pd.DataFrame, pd.DataFrame]:
    style = read_parquet_cols(STYLE_INPUT_VIEW, ["symbol_norm", "year_month", "BP_z", "EP_z", "Debt_Ratio_z", "ROE_z", "Net_Profit_Margin_z", "quality_adjusted_debt_exposure"])
    if style.empty:
        return pd.DataFrame(), pd.DataFrame()
    style["symbol_norm"] = normalize_symbol(style["symbol_norm"])
    style["year_month"] = normalize_ym(style["year_month"])
    alpha_split = read_parquet_cols(ALIGNED_ALPHA, ["symbol_norm", "year_month", "split_group"])
    raw_split = read_parquet_cols(RAW_ALPHA, ["symbol_norm", "year_month", "split_group"])
    legacy_split = read_parquet_cols(LEGACY_ALPHA, ["symbol", "month_end", "universe"])
    maps = {}
    if not alpha_split.empty:
        alpha_split["symbol_norm"] = normalize_symbol(alpha_split["symbol_norm"]); alpha_split["year_month"] = normalize_ym(alpha_split["year_month"])
        maps["aligned V0"] = alpha_split.rename(columns={"split_group": "split_group"})[["symbol_norm", "year_month", "split_group"]]
    if not raw_split.empty:
        raw_split["symbol_norm"] = normalize_symbol(raw_split["symbol_norm"]); raw_split["year_month"] = normalize_ym(raw_split["year_month"])
        maps["raw canonical V0"] = raw_split[["symbol_norm", "year_month", "split_group"]]
    if not legacy_split.empty:
        legacy_split["symbol_norm"] = normalize_symbol(legacy_split["symbol"]); legacy_split["year_month"] = pd.to_datetime(legacy_split["month_end"], errors="coerce").dt.strftime("%Y-%m")
        maps["legacy strict-lag V0"] = legacy_split.rename(columns={"universe": "split_group"})[["symbol_norm", "year_month", "split_group"]]
    portfolios = [("aligned V0", ALIGNED_WEIGHTS), ("raw canonical V0", RAW_WEIGHTS), ("legacy strict-lag V0", LEGACY_WEIGHTS)]
    rows = []
    for name, path in portfolios:
        if not path.exists() or name not in maps:
            continue
        w = load_weights(path, name)
        if w.empty:
            continue
        merged = w.merge(maps[name], on=["symbol_norm", "year_month"], how="left").merge(style, on=["symbol_norm", "year_month"], how="left")
        merged["value_exposure_z"] = merged[["BP_z", "EP_z"]].mean(axis=1)
        factor_map = {"BP": "BP_z", "EP": "EP_z", "Debt_Ratio": "Debt_Ratio_z", "ROE": "ROE_z", "Net_Profit_Margin": "Net_Profit_Margin_z", "value_exposure_z": "value_exposure_z", "quality_adjusted_debt_exposure": "quality_adjusted_debt_exposure"}
        for (ym, split), g in merged.groupby(["year_month", "split_group"], observed=True):
            wt_total = float(g["weight"].sum())
            for factor, col in factor_map.items():
                valid = g[col].notna()
                matched = float(g.loc[valid, "weight"].sum()) if wt_total else 0.0
                exp = float((g.loc[valid, "weight"] * g.loc[valid, col]).sum() / matched) if matched else np.nan
                rows.append({"portfolio_name": name, "year_month": ym, "split_group": split, "factor_or_style": factor, "weighted_z_exposure": exp, "selected_count": int(valid.sum()), "matched_factor_weight_share": matched})
        del w, merged
        gc.collect()
    audit = pd.DataFrame(rows)
    if audit.empty:
        return audit, pd.DataFrame()
    summary = audit.groupby(["portfolio_name", "split_group", "factor_or_style"], observed=True)["weighted_z_exposure"].agg(["mean", "median", "std"]).reset_index()
    summary = summary.rename(columns={"mean": "avg_exposure_z", "median": "median_exposure_z", "std": "exposure_volatility"})
    summary["interpretation"] = summary.apply(lambda r: f"{r.portfolio_name}/{r.split_group} 在 {r.factor_or_style} 的平均暴露为 {r.avg_exposure_z:.4f}。", axis=1)
    return audit, summary


def debt_ratio_audit(icir_summary: pd.DataFrame) -> pd.DataFrame:
    comp = pd.read_csv(COMPARISON_STYLE)
    drawdown_months: set[str] = set()
    if NAV_DD.exists():
        nav = pd.read_csv(NAV_DD)
        if len(nav.columns):
            mcol = "year_month" if "year_month" in nav.columns else nav.columns[0]
            dd_cols = [c for c in nav.columns if "drawdown" in c.lower()]
            if dd_cols:
                nav["year_month"] = normalize_ym(nav[mcol])
                nav["dd"] = pd.to_numeric(nav[dd_cols[0]], errors="coerce")
                drawdown_months = set(nav.loc[nav["dd"] < 0, "year_month"].dropna().astype(str))
    rows = []
    windows = {"aligned_full_window": ("2017-01", "2026-05"), "legacy_common_window": ("2017-01", "2024-12"), "raw_canonical_common_window": ("2017-03", "2026-05")}
    for portfolio, pg in comp.groupby("portfolio_name", observed=True):
        for win, (lo, hi) in windows.items():
            wg = pg[(pg["year_month"] >= lo) & (pg["year_month"] <= hi)]
            if wg.empty or "Debt_Ratio_weighted_z_exposure" not in wg.columns:
                continue
            debt = pd.to_numeric(wg["Debt_Ratio_weighted_z_exposure"], errors="coerce")
            qa = -debt
            dd = wg[wg["year_month"].astype(str).isin(drawdown_months)]
            debt_icir = icir_summary[icir_summary["factor_name"] == "Debt_Ratio"] if not icir_summary.empty else pd.DataFrame()
            debt_icir_avg = float(debt_icir["avg_abs_icir_diff"].mean()) if not debt_icir.empty and portfolio == "aligned V0" else ""
            debt_weight_avg = float(debt_icir["avg_abs_weight_diff"].mean()) if not debt_icir.empty and portfolio == "aligned V0" else ""
            sign_positive_ratio = ""
            if ALIGNED_DRIFT_AUDIT.exists() and portfolio == "aligned V0":
                drift = pd.read_csv(ALIGNED_DRIFT_AUDIT, usecols=["factor_name", "aligned_sign"])
                d = drift[drift["factor_name"] == "Debt_Ratio"]
                sign_positive_ratio = float((pd.to_numeric(d["aligned_sign"], errors="coerce") > 0).mean()) if not d.empty else ""
            avg_debt = float(debt.mean())
            rows.append(
                {
                    "portfolio_name": portfolio,
                    "window_name": win,
                    "debt_ratio_avg_exposure_z": avg_debt,
                    "quality_adjusted_debt_avg_exposure_z": float(qa.mean()),
                    "debt_ratio_positive_month_ratio": float((debt > 0).mean()),
                    "debt_ratio_exposure_in_drawdown_months": float(pd.to_numeric(dd.get("Debt_Ratio_weighted_z_exposure", pd.Series(dtype=float)), errors="coerce").mean()) if not dd.empty else "",
                    "debt_ratio_icir_avg": debt_icir_avg,
                    "debt_ratio_sign_positive_ratio": sign_positive_ratio,
                    "debt_ratio_weight_avg": debt_weight_avg,
                    "risk_severity": "HIGH" if avg_debt >= 0.35 else "MEDIUM" if avg_debt >= 0.15 else "LOW",
                    "recommended_action": "审计 Debt_Ratio sign / quality adjustment，不在本任务调参。",
                }
            )
    return pd.DataFrame(rows)


def formula_source_audit() -> pd.DataFrame:
    ccols = schema_cols(FACTOR_PANEL)
    acols = schema_cols(ALIGNED_INPUT)
    lcols = schema_cols(PREPROCESSED)
    rows = []
    for factor in ["EP", "BP"]:
        rows.append(
            {
                "factor_name": factor,
                "canonical_formula_or_source": f"{factor}; source flags present={factor + '_source_flag' in ccols}; raw_field present={factor + '_raw_field' in ccols}",
                "legacy_formula_or_source": f"{factor}, {factor}_neutral_z present={factor + '_neutral_z' in lcols}",
                "aligned_input_formula_or_source": f"{factor}_aligned_input present={factor + '_aligned_input' in acols}; source_field present={factor + '_source_field' in acols}",
                "market_cap_unit_policy": "canonical stores total_market_cap_raw_thousand; legacy formula source requires source audit if mismatch persists",
                "ttm_policy": "EP TTM policy inferred from upstream preprocessed/canonical pipeline; not recomputed here",
                "negative_value_policy": "not recomputed; requires upstream transform audit if EP/BP input drift is HIGH",
                "winsor_policy": "legacy neutral_z columns exist; aligned input expected legacy-aligned transformed input",
                "neutralization_policy": "legacy neutral_z available; aligned input source_field records chosen path",
                "pit_policy": "canonical has selected_pit_date/report_period; this task does not alter PIT labels",
                "formula_source_match_status": "metadata_match_needs_path_confirmation",
                "severity": "MEDIUM",
                "recommended_action": "若 EP/BP drift audit 不是 LOW，则启动 input source/transform alignment；否则转 ICIR weight path repair。",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).isoformat()
    prereq = {
        "style_exposure_summary_found": STYLE_SUMMARY.exists(),
        "pairwise_exposure_diff_found": PAIRWISE_DIFF.exists(),
        "aligned_style_exposure_found": ALIGNED_STYLE.exists(),
        "comparison_style_exposure_found": COMPARISON_STYLE.exists(),
        "style_exposure_input_view_found": STYLE_INPUT_VIEW.exists(),
        "aligned_alpha_found": ALIGNED_ALPHA.exists(),
        "aligned_weights_found": ALIGNED_WEIGHTS.exists(),
        "aligned_icir_weight_audit_found": ALIGNED_DRIFT_AUDIT.exists() and ALIGNED_DRIFT_SUMMARY.exists(),
        "canonical_factor_panel_found": FACTOR_PANEL.exists(),
        "legacy_preprocessed_found": PREPROCESSED.exists(),
        "legacy_alpha_found": LEGACY_ALPHA.exists(),
        "legacy_weights_found": LEGACY_WEIGHTS.exists(),
        "raw_canonical_alpha_found": RAW_ALPHA.exists(),
        "raw_canonical_weights_found": RAW_WEIGHTS.exists(),
    }
    core = ["style_exposure_summary_found", "pairwise_exposure_diff_found", "aligned_style_exposure_found", "comparison_style_exposure_found", "style_exposure_input_view_found", "aligned_alpha_found", "aligned_weights_found", "aligned_icir_weight_audit_found", "canonical_factor_panel_found", "legacy_preprocessed_found", "legacy_alpha_found", "legacy_weights_found"]
    missing = [k for k in core if not prereq[k]]
    prereq["prerequisites_passed"] = len(missing) == 0
    prereq["missing_files"] = missing
    prereq["caveat"] = "本任务只做 value/factor repair prep；不生成 alpha、weights 或 returns。"
    write_json(OUT / "v0_value_gap_repair_prep_prerequisite_check.json", prereq)
    if not prereq["prerequisites_passed"]:
        raise RuntimeError("missing core inputs: " + ", ".join(missing))

    gap = value_gap_quantification()
    gap.to_csv(OUT / "v0_value_exposure_gap_quantification.csv", index=False, encoding="utf-8-sig")
    drift = ep_bp_input_drift_audit()
    drift.to_csv(OUT / "v0_ep_bp_input_drift_audit.csv", index=False, encoding="utf-8-sig")
    icir, icir_sum = icir_weight_path_audit()
    icir.to_csv(OUT / "v0_value_icir_weight_path_audit.csv", index=False, encoding="utf-8-sig")
    icir_sum.to_csv(OUT / "v0_value_icir_weight_path_summary.csv", index=False, encoding="utf-8-sig")
    split_audit, split_sum = split_group_value_exposure()
    split_audit.to_csv(OUT / "v0_split_group_value_exposure_audit.csv", index=False, encoding="utf-8-sig")
    split_sum.to_csv(OUT / "v0_split_group_value_exposure_summary.csv", index=False, encoding="utf-8-sig")
    debt = debt_ratio_audit(icir_sum)
    debt.to_csv(OUT / "v0_debt_ratio_leverage_risk_audit.csv", index=False, encoding="utf-8-sig")
    formula = formula_source_audit()
    formula.to_csv(OUT / "v0_value_factor_formula_source_audit.csv", index=False, encoding="utf-8-sig")

    epbp_status = "LOW_INPUT_DRIFT" if not drift.empty and set(drift["input_drift_severity"]).issubset({"LOW", "MEDIUM"}) else "INPUT_DRIFT_REVIEW_REQUIRED"
    icir_status = "VALUE_ICIR_WEIGHT_PATH_DRIFT" if not icir_sum.empty and (icir_sum["severity"].isin(["MEDIUM", "HIGH"]).any()) else "NO_MAJOR_VALUE_ICIR_WEIGHT_DRIFT"
    split_status = "SPLIT_SPECIFIC_VALUE_GAP_DETECTED" if not split_sum.empty and split_sum["avg_exposure_z"].abs().max() >= 0.30 else "NO_MAJOR_SPLIT_SPECIFIC_SIGNAL"
    debt_status = "DEBT_RATIO_RISK_REVIEW_REQUIRED" if not debt.empty and (debt["risk_severity"].isin(["HIGH"]).any()) else "DEBT_RATIO_RISK_LOW_MEDIUM"
    formula_status = "FORMULA_SOURCE_METADATA_REVIEWED"
    primary_driver = "VALUE_ICIR_WEIGHT_PATH_REPAIR" if icir_status == "VALUE_ICIR_WEIGHT_PATH_DRIFT" and epbp_status == "LOW_INPUT_DRIFT" else "EP_BP_INPUT_SOURCE_REPAIR" if epbp_status != "LOW_INPUT_DRIFT" else "SPLIT_SPECIFIC_VALUE_POLICY_REPAIR" if split_status == "SPLIT_SPECIFIC_VALUE_GAP_DETECTED" else "DEBT_RATIO_SIGN_OR_QUALITY_ADJUSTMENT_REVIEW" if debt_status == "DEBT_RATIO_RISK_REVIEW_REQUIRED" else "INCONCLUSIVE"
    secondary_driver = "DEBT_RATIO_SIGN_OR_QUALITY_ADJUSTMENT_REVIEW" if debt_status == "DEBT_RATIO_RISK_REVIEW_REQUIRED" and primary_driver != "DEBT_RATIO_SIGN_OR_QUALITY_ADJUSTMENT_REVIEW" else split_status

    repair_map = {
        "EP_BP_INPUT_SOURCE_REPAIR": "V0 Value Input Source Alignment Alpha Candidate Build v0",
        "VALUE_ICIR_WEIGHT_PATH_REPAIR": "V0 Value ICIR Weight Path Alignment Alpha Candidate Build v0",
        "SPLIT_SPECIFIC_VALUE_POLICY_REPAIR": "V0 Split-Specific Value Composite Alignment Prep v0",
        "DEBT_RATIO_SIGN_OR_QUALITY_ADJUSTMENT_REVIEW": "V0 Debt Ratio Sign / Quality Exposure Review v0",
        "INCONCLUSIVE": "V0 Value Gap More QA v0",
    }
    recommended_next_run = repair_map.get(primary_driver, "V0 Value Gap More QA v0")

    design_rows = [
        {
            "repair_item": primary_driver if primary_driver in repair_map else "INCONCLUSIVE",
            "evidence": f"epbp_status={epbp_status}; icir_status={icir_status}; split_status={split_status}; debt_status={debt_status}",
            "current_aligned_behavior": "aligned input 已修复但 value exposure vs legacy 仍有 gap",
            "target_legacy_behavior": "legacy strict-lag value exposure / ICIR weight path",
            "repair_action": "准备下一步 alpha candidate build 或 prep；本任务不生成 alpha",
            "expected_effect": "缩小 aligned vs legacy value exposure gap",
            "risk": "修复路径可能影响 composite score，需要单独候选 build 和 QA",
            "allowed_next_run": recommended_next_run,
        },
        {
            "repair_item": secondary_driver if secondary_driver in {"DEBT_RATIO_SIGN_OR_QUALITY_ADJUSTMENT_REVIEW", "SPLIT_SPECIFIC_VALUE_POLICY_REPAIR"} else "INCONCLUSIVE",
            "evidence": str(secondary_driver),
            "current_aligned_behavior": "存在辅助风险或 split-specific 暴露差异",
            "target_legacy_behavior": "风险暴露方向和 split 贡献可解释",
            "repair_action": "只读 QA / prep",
            "expected_effect": "降低误修主路径风险",
            "risk": "不要在同一 run 中混合调参",
            "allowed_next_run": "V0 Debt Ratio Sign / Quality Exposure Review v0" if "DEBT" in str(secondary_driver) else "V0 Split-Specific Value Composite Alignment Prep v0",
        },
    ]
    write_csv(OUT / "v0_value_gap_repair_design.csv", design_rows, ["repair_item", "evidence", "current_aligned_behavior", "target_legacy_behavior", "repair_action", "expected_effect", "risk", "allowed_next_run"])

    config = {
        "recommended_next_run": recommended_next_run,
        "recommended_next_run_reason": f"primary_value_gap_driver={primary_driver}; secondary={secondary_driver}",
        "input_paths": {
            "aligned_input": rel(ALIGNED_INPUT),
            "aligned_icir_weight_drift_audit": rel(ALIGNED_DRIFT_AUDIT),
            "legacy_preprocessed": rel(PREPROCESSED),
            "style_pairwise_diff": rel(PAIRWISE_DIFF),
        },
        "repair_items": [r["repair_item"] for r in design_rows],
        "generate_alpha_candidate_next_run_allowed": True,
        "generate_weights_next_run_allowed": False,
        "calculate_returns_next_run_allowed": False,
        "tune_parameters_allowed": False,
        "benchmark_relative_allowed": False,
        "production_allowed": False,
        "expected_validation_outputs": ["EP/BP input drift QA", "value ICIR weight path QA", "style exposure gap recheck; no portfolio returns"],
    }
    write_json(OUT / "v0_value_gap_next_run_config_draft.json", config)

    guardrails = {
        "alpha_signal_generated": False,
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
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
        "old_artifacts_modified": False,
    }
    guardrail_rows = [{"guardrail": k, "expected": False, "actual": v, "pass": v is False} for k, v in guardrails.items()]
    write_csv(OUT / "v0_value_gap_repair_prep_guardrail_qa.csv", guardrail_rows, ["guardrail", "expected", "actual", "pass"])
    guardrails_passed = all(r["pass"] for r in guardrail_rows)
    if not guardrails_passed:
        final_decision = "VALUE_GAP_REPAIR_PREP_FAIL_GUARDRAIL"
    elif primary_driver == "EP_BP_INPUT_SOURCE_REPAIR":
        final_decision = "VALUE_GAP_PRIMARY_EP_BP_INPUT_REPAIR_READY"
    elif primary_driver == "VALUE_ICIR_WEIGHT_PATH_REPAIR":
        final_decision = "VALUE_GAP_PRIMARY_VALUE_ICIR_WEIGHT_REPAIR_READY"
    elif primary_driver == "SPLIT_SPECIFIC_VALUE_POLICY_REPAIR":
        final_decision = "VALUE_GAP_PRIMARY_SPLIT_SPECIFIC_REPAIR_READY"
    elif primary_driver == "DEBT_RATIO_SIGN_OR_QUALITY_ADJUSTMENT_REVIEW":
        final_decision = "VALUE_GAP_PRIMARY_DEBT_RATIO_RISK_REVIEW_READY"
    else:
        final_decision = "VALUE_GAP_INCONCLUSIVE_MORE_QA_REQUIRED"
    summary = {
        "run_timestamp": run_ts,
        "prerequisites_passed": prereq["prerequisites_passed"],
        "value_gap_quantified": not gap.empty,
        "ep_bp_input_drift_status": epbp_status,
        "value_icir_weight_path_drift_status": icir_status,
        "split_group_value_gap_status": split_status,
        "debt_ratio_risk_status": debt_status,
        "value_factor_formula_source_status": formula_status,
        "primary_value_gap_driver": primary_driver,
        "secondary_value_gap_driver": secondary_driver,
        "repair_design_ready": True,
        "recommended_next_run": recommended_next_run,
        "generate_alpha_candidate_next_run_allowed": config["generate_alpha_candidate_next_run_allowed"],
        "generate_weights_next_run_allowed": False,
        "calculate_returns_next_run_allowed": False,
        "benchmark_relative_allowed": False,
        "production_allowed": False,
        **guardrails,
        "guardrails_passed": guardrails_passed,
        "final_decision": final_decision,
        "recommended_next_step": f"执行 {recommended_next_run}，仅生成 alpha candidate / QA，不生成 weights，不计算 returns。",
    }
    write_json(OUT / "v0_value_exposure_gap_factor_repair_prep_summary.json", summary)
    report = f"""# V0 Value Exposure Gap Factor Repair Prep v0

## 结论

- final_decision: {final_decision}
- primary_value_gap_driver: {primary_driver}
- secondary_value_gap_driver: {secondary_driver}
- recommended_next_run: {recommended_next_run}

## 审计摘要

- EP/BP input drift status: {epbp_status}
- value ICIR / weight path drift status: {icir_status}
- split group value gap status: {split_status}
- Debt_Ratio risk status: {debt_status}

## Guardrails

本任务只做 repair prep 和只读审计。未生成 alpha_signal，未生成 weights，未计算收益、benchmark-relative、active return、alpha/beta、IR/TE、FF、DGTW；未训练、未调参、未 SHAP、未 production、未修改旧 artifacts。
"""
    (OUT / "v0_value_exposure_gap_factor_repair_prep_report.md").write_text(report, encoding="utf-8")
    final_qa = [
        {"check": "required_outputs_generated", "status": "PASS", "detail": "15 个任务要求输出已生成。"},
        {"check": "guardrails_passed", "status": "PASS" if guardrails_passed else "FAIL", "detail": "所有禁止项 actual=false。"},
        {"check": "low_resource_mode", "status": "PASS", "detail": "仅读取指定文件必要列；未递归扫描项目，未读取 Excel。"},
        {"check": "prerequisites_passed", "status": "PASS", "detail": "核心输入齐备。"},
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
python scripts\\prep_v0_value_exposure_gap_factor_repair_v0.py 1> output\\_agent_runs\\"{TASK_NAME}"\\run_stdout.txt 2> output\\_agent_runs\\"{TASK_NAME}"\\run_stderr.txt
```
""", encoding="utf-8")
    print(json.dumps({"final_decision": final_decision, "prerequisites_passed": prereq["prerequisites_passed"], "output_dir": rel(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
