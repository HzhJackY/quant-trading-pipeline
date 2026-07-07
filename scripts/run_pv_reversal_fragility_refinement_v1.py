from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / "output" / "price_volume_divergence_reversal_audit_v1"
OUT = ROOT / "output" / "pv_reversal_fragility_refinement_v1"
ALL_DAILY_PATH = ROOT / "output" / "all_daily.parquet"
TRAINING_PANEL_PATH = ROOT / "output" / "training_panel_v15_sr.parquet"
README_PATH = ROOT / "README.md"
STATUS_PATH = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS_PATH = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
PAPER_TRADING_PIPELINE = ROOT / "paper_trading" / "paper_trading_pipeline.py"
README_REPORT_PATH = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"

FACTOR_PANEL_PATH = PREV / "pv_reversal_factor_panel_v1.parquet"
LABEL_PANEL_PATH = PREV / "pv_reversal_label_panel_v1.parquet"
PREV_IC_PATH = PREV / "pv_reversal_single_factor_ic_v1.csv"
PREV_ROBUSTNESS_PATH = PREV / "pv_reversal_robustness_v1.csv"
PREV_SCORECARD_CANDIDATES = [
    PREV / "pv_reversal_scorecard_v1.csv",
    PREV / "pv_reversal_factor_scorecard_v1.csv",
]

BASE_FACTORS = [
    "reversal_20d",
    "pv_divergence_reversal_20d",
    "price_down_volume_up_20d",
    "panic_reversal_10d",
    "volume_spike_without_price_rebound",
    "risk_adjusted_pv_reversal",
]
REFINED_FACTORS = [
    "reversal_20d_raw",
    "reversal_20d_liquid_only",
    "reversal_20d_non_microcap",
    "reversal_20d_size_liquidity_neutral",
    "reversal_20d_rank_neutral",
    "pv_reversal_neutral",
    "tradable_panic_reversal",
]
TARGETS = ["fwd_ret_1m", "fwd_ret_1m_excess_equal_weight"]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def file_hash(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_any(path: Path) -> tuple[pd.DataFrame | None, str]:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path), ""
        return pd.read_csv(path), ""
    except Exception as exc:
        return None, repr(exc)


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def rank_ic(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 50:
        return np.nan
    return x[valid].rank().corr(y[valid].rank())


def tstat(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    std = s.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return np.nan
    return float(s.mean() / (std / math.sqrt(len(s))))


def neutralize_group(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    valid = y.notna()
    x = x.copy()
    for col in x.columns:
        valid &= x[col].notna()
    out = pd.Series(np.nan, index=y.index)
    if valid.sum() < 50:
        return out
    xv = x.loc[valid].astype(float)
    yv = y.loc[valid].astype(float)
    keep = [c for c in xv.columns if xv[c].std(ddof=0) > 0]
    if not keep:
        out.loc[valid] = yv - yv.mean()
        return out
    mat = np.column_stack([np.ones(len(xv)), xv[keep].to_numpy()])
    try:
        beta = np.linalg.lstsq(mat, yv.to_numpy(), rcond=None)[0]
        out.loc[valid] = yv.to_numpy() - mat @ beta
    except np.linalg.LinAlgError:
        out.loc[valid] = yv - yv.mean()
    return out


def summarize_ic(panel: pd.DataFrame, factor: str, target: str) -> dict[str, object]:
    rows = []
    for m, g in panel.groupby("month_end"):
        ic = rank_ic(g[factor], g[target])
        if np.isfinite(ic):
            rows.append((m, ic, int((g[factor].notna() & g[target].notna()).sum())))
    vals = pd.Series([x[1] for x in rows], dtype="float64")
    coverage = float(panel[factor].notna().mean()) if len(panel) else 0.0
    return {
        "factor_name": factor,
        "target": target,
        "n_months": len(rows),
        "n_obs": int(sum(x[2] for x in rows)),
        "mean_rank_ic": float(vals.mean()) if len(vals) else np.nan,
        "median_rank_ic": float(vals.median()) if len(vals) else np.nan,
        "ic_std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
        "ic_ir": float(vals.mean() / vals.std(ddof=1)) if len(vals) > 1 and vals.std(ddof=1) != 0 else np.nan,
        "positive_ic_rate": float((vals > 0).mean()) if len(vals) else np.nan,
        "coverage_rate": coverage,
        "notes": "monthly Rank IC; minimum 50 stocks per month",
    }


def quantile_group(g: pd.DataFrame, factor: str, n: int = 5) -> pd.Series:
    out = pd.Series(np.nan, index=g.index)
    valid = g[factor].notna()
    if valid.sum() < 50 or g.loc[valid, factor].nunique() < 2:
        return out
    try:
        out.loc[valid] = pd.qcut(g.loc[valid, factor].rank(method="first"), n, labels=False) + 1
    except ValueError:
        pass
    return out


def group_spread(panel: pd.DataFrame, factor: str, target: str) -> dict[str, object]:
    p = panel.copy()
    p["group_id"] = p.groupby("month_end", group_keys=False).apply(lambda g: quantile_group(g, factor, 5), include_groups=False).sort_index()
    valid = p.dropna(subset=["group_id", target])
    top = valid[valid["group_id"] == 5].groupby("month_end")[target].mean()
    bottom = valid[valid["group_id"] == 1].groupby("month_end")[target].mean()
    spread = (top - bottom).dropna()
    by_year = spread.groupby(spread.index.year).mean() if len(spread) else pd.Series(dtype="float64")
    means = valid.groupby("group_id")[target].mean().sort_index()
    mono = float(means.rank().corr(pd.Series(means.index, index=means.index))) if len(means) >= 2 else np.nan
    coverage = float(panel[factor].notna().mean()) if len(panel) else 0.0
    return {
        "factor_name": factor,
        "target": target,
        "top_minus_bottom_mean": float(spread.mean()) if len(spread) else np.nan,
        "top_minus_bottom_tstat": tstat(spread),
        "monotonicity_score": mono,
        "avg_coverage": coverage,
        "worst_year": "" if by_year.empty else str(int(by_year.idxmin())),
        "best_year": "" if by_year.empty else str(int(by_year.idxmax())),
        "notes": "group diagnostic only; not a portfolio backtest" if coverage >= 0.2 else "LOW_COVERAGE_UNSTABLE; group diagnostic only",
    }


def exposure_interpretation(name: str, abs_corr: float, n_months: int) -> str:
    if n_months < 12 or not np.isfinite(abs_corr):
        return "INCONCLUSIVE"
    if abs_corr >= 0.35:
        if "liquidity" in name or "amount" in name:
            return "STRONG_LIQUIDITY_EXPOSURE"
        if "vol" in name:
            return "STRONG_VOLATILITY_EXPOSURE"
        if "size" in name or "market_cap" in name:
            return "STRONG_SIZE_EXPOSURE"
        return "MODERATE_EXPOSURE"
    if abs_corr >= 0.15:
        return "MODERATE_EXPOSURE"
    return "LOW_EXPOSURE"


def update_status(status_value: str) -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    csmar_status = status.get("alternative_data", {}).get("csmar_status")
    status.setdefault("research", {})
    status["research"]["price_volume_reversal_status"] = status_value
    status["research"]["price_volume_reversal_latest_task"] = "PV Reversal Fragility Attribution & Refinement v1"
    status["research"]["price_volume_reversal_latest_output"] = rel(OUT)
    status.setdefault("validation", {})
    status["validation"]["blend_v3_historical_metrics_status"] = "under_pit_review"
    if csmar_status is not None:
        status.setdefault("alternative_data", {})["csmar_status"] = csmar_status
    status.setdefault("project", {})["last_updated"] = date.today().isoformat()
    STATUS_PATH.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")


def append_decision(decision: str, fragile_summary: str, can_enter: bool) -> None:
    block = "\n".join([
        f"## {date.today().isoformat()}",
        "",
        "决策：",
        "",
        "- 完成 PV Reversal Fragility Attribution & Refinement v1。",
        f"- 是否仍然依赖小盘 / 低流动性：{fragile_summary}。",
        f"- 是否可以进入 residual alpha test：{can_enter}。",
        "- 不接入 production。",
        "- 不修改 README。",
        f"- Decision = {decision}。",
    ])
    text = DECISIONS_PATH.read_text(encoding="utf-8") if DECISIONS_PATH.exists() else "# 决策日志\n"
    if "PV Reversal Fragility Attribution & Refinement v1" in text and decision in text:
        return
    DECISIONS_PATH.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def run_status_scripts() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_current_status_md.py")], cwd=ROOT, check=True, capture_output=True, text=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_readme_consistency.py")], cwd=ROOT, check=True, capture_output=True, text=True)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    original_status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    original_csmar_status = original_status.get("alternative_data", {}).get("csmar_status")
    protected = {
        "README.md": file_hash(README_PATH),
        "all_daily.parquet": file_hash(ALL_DAILY_PATH),
        "training_panel_v15_sr.parquet": file_hash(TRAINING_PANEL_PATH),
        "paper_trading_pipeline.py": file_hash(PAPER_TRADING_PIPELINE),
    }
    model_files_before = {rel(p): file_hash(p) for p in (ROOT / "output").glob("production_models*/**/*") if p.is_file()}
    config_before = {rel(p): file_hash(p) for p in (ROOT / "config").glob("*") if p.is_file() and p.name != "project_status.yaml"}

    scorecard_input = next((p for p in PREV_SCORECARD_CANDIDATES if p.exists()), PREV_SCORECARD_CANDIDATES[0])
    inputs = [
        (FACTOR_PANEL_PATH, "previous factor panel"),
        (LABEL_PANEL_PATH, "previous label panel"),
        (PREV_IC_PATH, "previous IC"),
        (PREV_ROBUSTNESS_PATH, "previous robustness"),
        (scorecard_input, "previous scorecard"),
        (ALL_DAILY_PATH, "daily market data read-only"),
        (TRAINING_PANEL_PATH, "v15 universe/month index read-only"),
    ]
    loaded: dict[str, pd.DataFrame | None] = {}
    audit_rows = []
    for path, role in inputs:
        df, err = read_any(path)
        loaded[str(path)] = df
        audit_rows.append({
            "input_path": rel(path),
            "exists": path.exists(),
            "readable": df is not None,
            "n_rows": 0 if df is None else len(df),
            "role": role,
            "notes": err or ("scorecard path fallback used" if role == "previous scorecard" and path.name != "pv_reversal_scorecard_v1.csv" else ""),
        })
    input_audit_path = OUT / "input_audit_v1.csv"
    pd.DataFrame(audit_rows).to_csv(input_audit_path, index=False, encoding="utf-8-sig")

    factor_panel = loaded[str(FACTOR_PANEL_PATH)]
    label_panel = loaded[str(LABEL_PANEL_PATH)]
    all_daily = loaded[str(ALL_DAILY_PATH)]
    if factor_panel is None or label_panel is None or all_daily is None:
        decision = "PV_REVERSAL_REFINEMENT_INVALID_DATA_QUALITY"
        update_status("refinement_still_fragile")
        final_qa_path = OUT / "final_qa_pv_reversal_fragility_refinement_v1.csv"
        pd.DataFrame([{"check": "required inputs readable", "pass": False, "details": "factor/label/all_daily input missing"}]).to_csv(final_qa_path, index=False, encoding="utf-8-sig")
        for k, v in {
            "input_audit_path": rel(input_audit_path),
            "exposure_attribution_path": "",
            "refined_factor_panel_path": "",
            "refined_ic_path": "",
            "refined_group_spread_path": "",
            "refined_fragility_audit_path": "",
            "raw_vs_refined_comparison_path": "",
            "scorecard_path": "",
            "report_path": "",
            "task_completion_card_path": "",
            "final_qa_path": rel(final_qa_path),
            "project_status_path": rel(STATUS_PATH),
            "current_status_doc_path": rel(CURRENT_STATUS_PATH),
            "decisions_doc_path": rel(DECISIONS_PATH),
            "readme_consistency_report_path": rel(README_REPORT_PATH),
            "best_refined_factor": "",
            "best_refined_mean_rank_ic": np.nan,
            "best_refined_ic_ir": np.nan,
            "best_refined_spread": np.nan,
            "best_refined_classification": "STOP_FOR_NOW",
            "n_refined_promising_factors": 0,
            "n_still_fragile_factors": 0,
            "recommended_next_task": "Stop for now due to input data quality",
            "can_enter_residual_alpha_test": False,
            "readme_modified": file_hash(README_PATH) != protected["README.md"],
            "all_daily_modified": file_hash(ALL_DAILY_PATH) != protected["all_daily.parquet"],
            "training_panel_modified": file_hash(TRAINING_PANEL_PATH) != protected["training_panel_v15_sr.parquet"],
            "production_modified": False,
            "csmar_api_accessed": False,
            "decision": decision,
        }.items():
            print(f"{k}={v}")
        return

    factor_panel = factor_panel.copy()
    label_panel = label_panel.copy()
    factor_panel["symbol"] = factor_panel["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    label_panel["symbol"] = label_panel["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    factor_panel["month_end"] = pd.to_datetime(factor_panel["month_end"])
    label_panel["month_end"] = pd.to_datetime(label_panel["month_end"])
    panel = factor_panel.merge(label_panel, on=["symbol", "month_end"], how="left")

    daily = all_daily.copy()
    daily["symbol"] = daily["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["symbol", "date"])
    g = daily.groupby("symbol", group_keys=False)
    daily["daily_ret"] = g["close"].pct_change()
    daily["amount_mean_20d"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    daily["amount_mean_60d"] = g["amount"].transform(lambda s: s.rolling(60, min_periods=20).mean())
    daily["amount_mean_120d"] = g["amount"].transform(lambda s: s.rolling(120, min_periods=60).mean())
    daily["vol_20d_exposure"] = g["daily_ret"].transform(lambda s: s.rolling(20, min_periods=10).std())
    daily["vol_60d_exposure"] = g["daily_ret"].transform(lambda s: s.rolling(60, min_periods=20).std())
    daily["ret_5d_exposure"] = g["close"].pct_change(5)
    daily["ret_10d_exposure"] = g["close"].pct_change(10)
    daily["ret_20d_exposure"] = g["close"].pct_change(20)
    daily_month = daily.rename(columns={"date": "month_end"})
    exposures = daily_month[daily_month["month_end"].isin(panel["month_end"].unique())][[
        "symbol", "month_end", "close", "amount_mean_20d", "amount_mean_60d", "amount_mean_120d",
        "vol_20d_exposure", "vol_60d_exposure", "ret_5d_exposure", "ret_10d_exposure", "ret_20d_exposure",
    ]]
    panel = panel.merge(exposures, on=["symbol", "month_end"], how="left")
    panel["liquidity_proxy"] = panel["amount_mean_60d"].fillna(panel["amount_mean_20d"]).fillna(panel["amount_mean_120d"])
    panel["size_proxy"] = np.nan
    panel["volatility_bucket"] = panel.groupby("month_end")["vol_20d"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low_vol", "mid_vol", "high_vol"]) if s.notna().sum() >= 50 else pd.Series(np.nan, index=s.index)
    )
    panel["liquidity_rank_pct"] = panel.groupby("month_end")["liquidity_proxy"].rank(pct=True)
    panel["vol_rank_pct"] = panel.groupby("month_end")["vol_20d"].rank(pct=True)
    panel["non_low_liquidity_40"] = panel["liquidity_rank_pct"] > 0.4
    panel["non_high_vol_10"] = panel["vol_rank_pct"] <= 0.9
    panel["non_microcap"] = panel["size_bucket"].fillna("micro_or_unknown") != "micro_or_unknown"

    exposure_cols = {
        "size_proxy_unavailable": "size_proxy",
        "liquidity_amount_mean_20d": "amount_mean_20d",
        "liquidity_amount_mean_60d": "amount_mean_60d",
        "liquidity_amount_mean_120d": "amount_mean_120d",
        "volatility_vol_20d": "vol_20d",
        "volatility_vol_60d": "vol_60d",
        "price_level_close": "close",
        "short_term_ret_5d": "ret_5d",
        "short_term_ret_10d": "ret_10d",
        "short_term_ret_20d": "ret_20d",
    }
    exposure_rows = []
    for factor in BASE_FACTORS:
        if factor not in panel.columns:
            continue
        for exposure_name, col in exposure_cols.items():
            corrs = []
            for _, month in panel.groupby("month_end"):
                valid = month[factor].notna() & month[col].notna()
                if valid.sum() >= 50 and month.loc[valid, factor].nunique() > 1 and month.loc[valid, col].nunique() > 1:
                    corrs.append(month.loc[valid, factor].rank().corr(month.loc[valid, col].rank()))
            s = pd.Series(corrs, dtype="float64")
            abs_mean = float(s.abs().mean()) if len(s) else np.nan
            exposure_rows.append({
                "factor_name": factor,
                "exposure_name": exposure_name,
                "n_months": int(s.notna().sum()),
                "mean_spearman_corr": float(s.mean()) if len(s) else np.nan,
                "median_spearman_corr": float(s.median()) if len(s) else np.nan,
                "abs_corr_mean": abs_mean,
                "interpretation": exposure_interpretation(exposure_name, abs_mean, int(s.notna().sum())),
                "notes": "size proxy unavailable in all_daily; liquidity uses amount rolling means" if "size" in exposure_name else "",
            })
    exposure_df = pd.DataFrame(exposure_rows)
    exposure_attribution_path = OUT / "pv_reversal_exposure_attribution_v1.csv"
    exposure_df.to_csv(exposure_attribution_path, index=False, encoding="utf-8-sig")

    refined = panel.copy()
    refined["reversal_20d_raw"] = refined["reversal_20d"]
    refined["reversal_20d_liquid_only"] = refined["reversal_20d"].where(refined["non_low_liquidity_40"])
    refined["reversal_20d_non_microcap"] = refined["reversal_20d"].where(refined["non_microcap"])
    control_cols = ["liquidity_proxy", "vol_20d"]
    refined["reversal_20d_size_liquidity_neutral"] = refined.groupby("month_end", group_keys=False).apply(
        lambda m: neutralize_group(m["reversal_20d"], m[control_cols]), include_groups=False
    ).sort_index()
    rank_controls = refined[control_cols].copy()
    for c in control_cols:
        rank_controls[c] = refined.groupby("month_end")[c].rank(pct=True)
    rank_tmp = refined[["month_end", "reversal_20d", "price_down_volume_up_20d"]].join(rank_controls, rsuffix="_rank")
    refined["reversal_20d_rank_neutral"] = rank_tmp.groupby("month_end", group_keys=False).apply(
        lambda m: neutralize_group(m["reversal_20d"], m[control_cols]), include_groups=False
    ).sort_index()
    refined["pv_reversal_neutral"] = refined.groupby("month_end", group_keys=False).apply(
        lambda m: neutralize_group(m["price_down_volume_up_20d"], m[control_cols]), include_groups=False
    ).sort_index()
    tradable_mask = refined["non_low_liquidity_40"] & refined["non_high_vol_10"] & refined["non_microcap"]
    refined["tradable_panic_reversal"] = refined["price_down_volume_up_20d"].where(tradable_mask)
    refined["refinement_notes"] = np.where(refined["non_microcap"], "", "market cap unavailable; non_microcap filters exclude unknown bucket")
    refined_cols = [
        "symbol", "month_end", "ret_5d", "ret_10d", "ret_20d", "vol_20d", "vol_60d", "liquidity_proxy",
        "liquidity_bucket", "size_bucket", "volatility_bucket", "non_low_liquidity_40", "non_high_vol_10", "non_microcap",
    ] + REFINED_FACTORS + ["refinement_notes"]
    refined_panel = refined[refined_cols].copy()
    refined_factor_panel_path = OUT / "pv_reversal_refined_factor_panel_v1.parquet"
    refined_panel.to_parquet(refined_factor_panel_path, index=False)
    refined_sample_path = OUT / "pv_reversal_refined_factor_panel_sample_v1.csv"
    refined_panel.head(1000).to_csv(refined_sample_path, index=False, encoding="utf-8-sig")

    eval_panel = refined_panel.merge(label_panel, on=["symbol", "month_end"], how="left")
    refined_ic = pd.DataFrame([summarize_ic(eval_panel, f, t) for f in REFINED_FACTORS for t in TARGETS])
    refined_ic_path = OUT / "pv_reversal_refined_ic_v1.csv"
    refined_ic.to_csv(refined_ic_path, index=False, encoding="utf-8-sig")

    spread_df = pd.DataFrame([group_spread(eval_panel, f, t) for f in REFINED_FACTORS for t in TARGETS])
    refined_group_spread_path = OUT / "pv_reversal_refined_group_spread_v1.csv"
    spread_df.to_csv(refined_group_spread_path, index=False, encoding="utf-8-sig")

    frag_rows = []
    dimensions = [
        ("size", "size_bucket"),
        ("liquidity", "liquidity_bucket"),
        ("volatility", "volatility_bucket"),
    ]
    for factor in REFINED_FACTORS:
        for dim, col in dimensions:
            for val, sub in eval_panel.groupby(col, dropna=False):
                if len(sub) < 100:
                    continue
                ic = summarize_ic(sub, factor, "fwd_ret_1m_excess_equal_weight")
                sp = group_spread(sub, factor, "fwd_ret_1m_excess_equal_weight")
                flag = "INCONCLUSIVE"
                if ic["n_months"] >= 12 and np.isfinite(ic["mean_rank_ic"]):
                    if float(ic["mean_rank_ic"]) <= 0 or not np.isfinite(sp["top_minus_bottom_mean"]) or float(sp["top_minus_bottom_mean"]) <= 0:
                        flag = "WEAK_AFTER_FILTER"
                    elif dim == "liquidity" and val == "low_liquidity":
                        flag = "STILL_LOW_LIQUIDITY_DEPENDENT"
                    elif dim == "size" and val == "micro_or_unknown":
                        flag = "STILL_SMALL_CAP_DEPENDENT"
                    elif dim == "volatility" and val == "high_vol":
                        flag = "STILL_HIGH_VOL_DEPENDENT"
                    else:
                        flag = "PASSES_TRADABILITY_FILTER"
                frag_rows.append({
                    "factor_name": factor,
                    "fragility_dimension": dim,
                    "group_value": val,
                    "n_months": ic["n_months"],
                    "n_obs": ic["n_obs"],
                    "mean_rank_ic": ic["mean_rank_ic"],
                    "ic_ir": ic["ic_ir"],
                    "top_minus_bottom_mean": sp["top_minus_bottom_mean"],
                    "fragility_flag": flag,
                    "notes": "size bucket is unknown because market cap fields are unavailable" if dim == "size" else "",
                })
    fragility_df = pd.DataFrame(frag_rows)
    refined_fragility_audit_path = OUT / "pv_reversal_refined_fragility_audit_v1.csv"
    fragility_df.to_csv(refined_fragility_audit_path, index=False, encoding="utf-8-sig")

    raw_ic = summarize_ic(eval_panel.assign(reversal_20d=panel["reversal_20d"]), "reversal_20d", "fwd_ret_1m_excess_equal_weight")
    raw_sp = group_spread(eval_panel.assign(reversal_20d=panel["reversal_20d"]), "reversal_20d", "fwd_ret_1m_excess_equal_weight")
    comparison_rows = []
    for factor in REFINED_FACTORS:
        ic = refined_ic[(refined_ic["factor_name"] == factor) & (refined_ic["target"] == "fwd_ret_1m_excess_equal_weight")].iloc[0]
        sp = spread_df[(spread_df["factor_name"] == factor) & (spread_df["target"] == "fwd_ret_1m_excess_equal_weight")].iloc[0]
        ref_frag = "fragile" if factor in {"reversal_20d_non_microcap", "tradable_panic_reversal"} and float(ic["coverage_rate"]) < 0.2 else "see_fragility_audit"
        delta_ic = float(ic["mean_rank_ic"]) - float(raw_ic["mean_rank_ic"]) if np.isfinite(ic["mean_rank_ic"]) else np.nan
        improvement = "IMPROVED" if np.isfinite(delta_ic) and delta_ic > 0.005 else "DETERIORATED_OR_FLAT"
        comparison_rows.append({
            "raw_factor": "reversal_20d",
            "refined_factor": factor,
            "raw_mean_ic": raw_ic["mean_rank_ic"],
            "refined_mean_ic": ic["mean_rank_ic"],
            "raw_ic_ir": raw_ic["ic_ir"],
            "refined_ic_ir": ic["ic_ir"],
            "raw_spread": raw_sp["top_minus_bottom_mean"],
            "refined_spread": sp["top_minus_bottom_mean"],
            "raw_fragility": "FRAGILE_SMALL_CAP_OR_LIQUIDITY_EXPOSURE",
            "refined_fragility": ref_frag,
            "improvement_or_deterioration": improvement,
            "notes": "comparison is diagnostic only; no trading simulation",
        })
    comparison_df = pd.DataFrame(comparison_rows)
    raw_vs_refined_comparison_path = OUT / "pv_reversal_raw_vs_refined_comparison_v1.csv"
    comparison_df.to_csv(raw_vs_refined_comparison_path, index=False, encoding="utf-8-sig")

    score_rows = []
    for factor in REFINED_FACTORS:
        ic = refined_ic[(refined_ic["factor_name"] == factor) & (refined_ic["target"] == "fwd_ret_1m_excess_equal_weight")].iloc[0]
        sp = spread_df[(spread_df["factor_name"] == factor) & (spread_df["target"] == "fwd_ret_1m_excess_equal_weight")].iloc[0]
        ff = fragility_df[fragility_df["factor_name"] == factor]
        size_frag = "STILL_SMALL_CAP_DEPENDENT" if (ff["fragility_flag"] == "STILL_SMALL_CAP_DEPENDENT").any() else "not_detectable_or_unavailable"
        liq_frag = "STILL_LOW_LIQUIDITY_DEPENDENT" if (ff["fragility_flag"] == "STILL_LOW_LIQUIDITY_DEPENDENT").any() else "not_obvious"
        vol_frag = "STILL_HIGH_VOL_DEPENDENT" if (ff["fragility_flag"] == "STILL_HIGH_VOL_DEPENDENT").any() else "not_obvious"
        coverage = float(ic["coverage_rate"])
        strong = np.isfinite(ic["mean_rank_ic"]) and np.isfinite(ic["ic_ir"]) and np.isfinite(sp["top_minus_bottom_mean"]) and float(ic["mean_rank_ic"]) > 0.015 and float(ic["ic_ir"]) > 0.1 and float(sp["top_minus_bottom_mean"]) > 0 and coverage >= 0.4
        fragile = size_frag.startswith("STILL") or liq_frag.startswith("STILL") or vol_frag.startswith("STILL") or coverage < 0.2
        if strong and not fragile:
            cls = "REFINED_PROMISING_FOR_RESIDUAL_TEST"
            action = "CONTINUE_TO_RESIDUAL_ALPHA_TEST"
        elif np.isfinite(ic["mean_rank_ic"]) and float(ic["mean_rank_ic"]) > 0 and (fragile or coverage < 0.4):
            cls = "IMPROVED_BUT_STILL_FRAGILE"
            action = "REFINE_FURTHER"
        elif not np.isfinite(ic["mean_rank_ic"]) or coverage < 0.05:
            cls = "DIAGNOSTIC_ONLY"
            action = "KEEP_AS_RISK_DIAGNOSTIC"
        else:
            cls = "WEAK_AFTER_NEUTRALIZATION"
            action = "STOP_FOR_NOW"
        score_rows.append({
            "factor_name": factor,
            "mean_rank_ic": ic["mean_rank_ic"],
            "ic_ir": ic["ic_ir"],
            "spread_mean": sp["top_minus_bottom_mean"],
            "coverage_rate": coverage,
            "size_fragility": size_frag,
            "liquidity_fragility": liq_frag,
            "volatility_fragility": vol_frag,
            "tradability_survival": "survives" if strong else "weak_or_fragile",
            "overall_classification": cls,
            "recommended_action": action,
            "notes": "Conservative classification; no tradable alpha or production-ready claim.",
        })
    scorecard = pd.DataFrame(score_rows).sort_values(["overall_classification", "mean_rank_ic"], ascending=[True, False])
    scorecard_path = OUT / "pv_reversal_refinement_scorecard_v1.csv"
    scorecard.to_csv(scorecard_path, index=False, encoding="utf-8-sig")
    sort_score = scorecard.assign(_score=scorecard["mean_rank_ic"].fillna(-9) + scorecard["spread_mean"].fillna(-9))
    best = sort_score.sort_values("_score", ascending=False).iloc[0]
    n_promising = int((scorecard["overall_classification"] == "REFINED_PROMISING_FOR_RESIDUAL_TEST").sum())
    n_still_fragile = int((scorecard["overall_classification"] == "IMPROVED_BUT_STILL_FRAGILE").sum())
    if n_promising:
        decision = "PV_REVERSAL_REFINEMENT_PROMISING_READY_FOR_RESIDUAL_TEST"
    elif n_still_fragile:
        decision = "PV_REVERSAL_REFINEMENT_IMPROVED_BUT_STILL_FRAGILE"
    elif (scorecard["overall_classification"] == "WEAK_AFTER_NEUTRALIZATION").any():
        decision = "PV_REVERSAL_REFINEMENT_WEAK_AFTER_NEUTRALIZATION"
    else:
        decision = "PV_REVERSAL_REFINEMENT_INVALID_DATA_QUALITY"
    can_enter = bool(n_promising)
    recommended_next = "Continue to residual alpha test with strict controls" if can_enter else "Keep as risk diagnostic and refine further before residual alpha test"
    status_value = "refinement_completed" if can_enter else "refinement_still_fragile"

    report_path = OUT / "pv_reversal_fragility_refinement_report_v1.md"
    report_path.write_text("\n".join([
        "# PV Reversal Fragility Attribution & Refinement v1",
        "",
        "## 1. Executive Summary",
        "",
        f"- Decision: {decision}",
        f"- Best refined factor: {best['factor_name']}",
        f"- Best refined mean Rank IC: {float(best['mean_rank_ic']) if np.isfinite(best['mean_rank_ic']) else np.nan:.6f}",
        f"- Best refined classification: {best['overall_classification']}",
        "- Conservative conclusion: this does not establish tradable alpha or production readiness.",
        "",
        "## 2. Scope and Non-Goals",
        "",
        "- This task is not model training.",
        "- This task is not a backtest.",
        "- This task does not connect to Blend V3 or Compact-F.",
        "- This task does not modify production.",
        "- This task does not solve the CSMAR PIT financial issue.",
        "",
        "## 3. Previous Audit Recap",
        "",
        "Previous decision was PV_REVERSAL_AUDIT_FRAGILE_NEEDS_REFINEMENT; reversal_20d had positive single-factor diagnostics but fragile liquidity/size exposure.",
        "",
        "## 4. Exposure Attribution",
        "",
        exposure_df.to_markdown(index=False),
        "",
        "## 5. Refined Factor Definitions",
        "",
        "Neutralized variants are monthly cross-sectional residuals against liquidity and volatility proxies. Size proxy is unavailable, so market-cap neutralization cannot be fully performed.",
        "",
        "## 6. Refined IC Results",
        "",
        refined_ic.to_markdown(index=False),
        "",
        "## 7. Group Return Diagnostics",
        "",
        spread_df.to_markdown(index=False),
        "",
        "## 8. Fragility Re-Audit",
        "",
        fragility_df.to_markdown(index=False),
        "",
        "## 9. Raw vs Refined Comparison",
        "",
        comparison_df.to_markdown(index=False),
        "",
        "## 10. Scorecard",
        "",
        scorecard.to_markdown(index=False),
        "",
        "## 11. Limitations",
        "",
        "- all_daily lacks market-cap, ST, paused, trade-status, turnover, and limit fields.",
        "- Non-microcap filters are limited because size_bucket is unknown without market cap.",
        "- Only diagnostics were run; no model, no backtest, no trading signal.",
        "",
        "## 12. Recommended Next Task",
        "",
        recommended_next,
        "",
        "## 13. Files Generated",
        "",
        "\n".join(f"- `{rel(p)}`" for p in [
            input_audit_path, exposure_attribution_path, refined_factor_panel_path, refined_sample_path, refined_ic_path,
            refined_group_spread_path, refined_fragility_audit_path, raw_vs_refined_comparison_path, scorecard_path
        ]),
        "",
    ]), encoding="utf-8")

    task_completion_card_path = OUT / "task_completion_card.md"
    fragile_summary = "yes_or_unresolved" if not can_enter else "not_obvious_after_refinement"
    task_completion_card_path.write_text("\n".join([
        "任务名称：PV Reversal Fragility Attribution & Refinement v1",
        f"运行日期：{date.today().isoformat()}",
        "是否修改 production：否",
        f"是否修改 README：{'否' if file_hash(README_PATH) == protected['README.md'] else '是'}",
        f"是否修改 all_daily：{'否' if file_hash(ALL_DAILY_PATH) == protected['all_daily.parquet'] else '是'}",
        f"是否修改 training_panel：{'否' if file_hash(TRAINING_PANEL_PATH) == protected['training_panel_v15_sr.parquet'] else '是'}",
        "是否训练模型：否",
        "是否运行回测：否",
        "是否做 IC：是",
        "是否生成交易信号：否",
        f"核心输出：{rel(report_path)}",
        f"核心结论：{decision}",
        f"最佳 refined factor：{best['factor_name']}",
        f"是否仍存在小盘/流动性脆弱性：{fragile_summary}",
        f"是否可以进入 residual alpha test：{can_enter}",
        f"下一步建议：{recommended_next}",
    ]), encoding="utf-8")

    update_status(status_value)
    append_decision(decision, fragile_summary, can_enter)
    run_status_scripts()

    model_files_after = {rel(p): file_hash(p) for p in (ROOT / "output").glob("production_models*/**/*") if p.is_file()}
    config_after = {rel(p): file_hash(p) for p in (ROOT / "config").glob("*") if p.is_file() and p.name != "project_status.yaml"}
    readme_modified = file_hash(README_PATH) != protected["README.md"]
    all_daily_modified = file_hash(ALL_DAILY_PATH) != protected["all_daily.parquet"]
    training_modified = file_hash(TRAINING_PANEL_PATH) != protected["training_panel_v15_sr.parquet"]
    production_modified = (file_hash(PAPER_TRADING_PIPELINE) != protected["paper_trading_pipeline.py"]) or model_files_before != model_files_after or config_before != config_after
    final_decision = "INVALID_MODIFICATION" if any([readme_modified, all_daily_modified, training_modified, production_modified]) else decision
    current_csmar_status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8")).get("alternative_data", {}).get("csmar_status")
    qa_rows = [
        ("README.md not modified", not readme_modified, str(readme_modified)),
        ("all_daily.parquet not modified", not all_daily_modified, str(all_daily_modified)),
        ("training_panel_v15_sr.parquet not modified", not training_modified, str(training_modified)),
        ("model files not modified", model_files_before == model_files_after, "production_models* hashes unchanged"),
        ("paper_trading_pipeline.py not modified", file_hash(PAPER_TRADING_PIPELINE) == protected["paper_trading_pipeline.py"], "hash checked"),
        ("production config not modified", config_before == config_after, "config files except project_status.yaml unchanged"),
        ("no model training executed", True, "only factor residualization, no predictive model training"),
        ("no full backtest executed", True, "group diagnostics only"),
        ("no trading signal generated", True, "no signal output written"),
        ("no real orders generated", True, "paper trading untouched"),
        ("no CSMAR API access executed", True, "no CSMAR code path called"),
        ("no MediaCrawler executed", True, "MediaCrawler untouched"),
        ("root-level output used", str(OUT).startswith(str(ROOT / "output")), rel(OUT)),
        ("xhs/output not used for new outputs", not str(OUT).replace("\\", "/").startswith(str(ROOT / "xhs" / "output").replace("\\", "/")), rel(OUT)),
        ("exposure attribution generated", exposure_attribution_path.exists(), rel(exposure_attribution_path)),
        ("refined factor panel generated", refined_factor_panel_path.exists(), rel(refined_factor_panel_path)),
        ("refined IC generated", refined_ic_path.exists(), rel(refined_ic_path)),
        ("refined group spread generated", refined_group_spread_path.exists(), rel(refined_group_spread_path)),
        ("fragility audit generated", refined_fragility_audit_path.exists(), rel(refined_fragility_audit_path)),
        ("raw vs refined comparison generated", raw_vs_refined_comparison_path.exists(), rel(raw_vs_refined_comparison_path)),
        ("scorecard generated", scorecard_path.exists(), rel(scorecard_path)),
        ("final report generated", report_path.exists(), rel(report_path)),
        ("task completion card generated", task_completion_card_path.exists(), rel(task_completion_card_path)),
        ("project_status.yaml updated without overwriting CSMAR status", current_csmar_status == original_csmar_status, rel(STATUS_PATH)),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS_PATH.exists(), rel(CURRENT_STATUS_PATH)),
        ("DECISIONS.md appended", "PV Reversal Fragility Attribution & Refinement v1" in DECISIONS_PATH.read_text(encoding="utf-8"), rel(DECISIONS_PATH)),
        ("README consistency check executed", README_REPORT_PATH.exists(), rel(README_REPORT_PATH)),
        ("README not auto-modified", not readme_modified, str(readme_modified)),
        ("conclusion uses conservative language", "does not establish tradable alpha" in report_path.read_text(encoding="utf-8"), "conservative wording checked"),
    ]
    final_qa_path = OUT / "final_qa_pv_reversal_fragility_refinement_v1.csv"
    pd.DataFrame(qa_rows, columns=["check", "pass", "details"]).to_csv(final_qa_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_audit_path": rel(input_audit_path),
        "exposure_attribution_path": rel(exposure_attribution_path),
        "refined_factor_panel_path": rel(refined_factor_panel_path),
        "refined_ic_path": rel(refined_ic_path),
        "refined_group_spread_path": rel(refined_group_spread_path),
        "refined_fragility_audit_path": rel(refined_fragility_audit_path),
        "raw_vs_refined_comparison_path": rel(raw_vs_refined_comparison_path),
        "scorecard_path": rel(scorecard_path),
        "report_path": rel(report_path),
        "task_completion_card_path": rel(task_completion_card_path),
        "final_qa_path": rel(final_qa_path),
        "project_status_path": rel(STATUS_PATH),
        "current_status_doc_path": rel(CURRENT_STATUS_PATH),
        "decisions_doc_path": rel(DECISIONS_PATH),
        "readme_consistency_report_path": rel(README_REPORT_PATH),
        "best_refined_factor": best["factor_name"],
        "best_refined_mean_rank_ic": best["mean_rank_ic"],
        "best_refined_ic_ir": best["ic_ir"],
        "best_refined_spread": best["spread_mean"],
        "best_refined_classification": best["overall_classification"],
        "n_refined_promising_factors": n_promising,
        "n_still_fragile_factors": n_still_fragile,
        "recommended_next_task": recommended_next,
        "can_enter_residual_alpha_test": can_enter,
        "readme_modified": readme_modified,
        "all_daily_modified": all_daily_modified,
        "training_panel_modified": training_modified,
        "production_modified": production_modified,
        "csmar_api_accessed": False,
        "decision": final_decision,
    }
    (OUT / "terminal_summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for k, v in summary.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
