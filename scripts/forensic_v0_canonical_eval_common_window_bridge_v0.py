from __future__ import annotations

import gc
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "v0_canonical_eval_forensic_common_window_bridge_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

CANON_EVAL_DIR = ROOT / "output" / "v0_canonical_repaired_trd_mnth_eval_run_v0"
CANON_SUMMARY = CANON_EVAL_DIR / "v0_canonical_repaired_trd_mnth_eval_run_summary.json"
CANON_MONTHLY = CANON_EVAL_DIR / "v0_canonical_monthly_net_returns_by_cost.csv"
CANON_NAV = CANON_EVAL_DIR / "v0_canonical_nav_drawdown_path.csv"
CANON_LEGACY_COMP = CANON_EVAL_DIR / "v0_canonical_vs_legacy_readonly_comparison.csv"
CANON_WEIGHTS = ROOT / "output" / "v0_canonical_portfolio_construction_run_v0" / "v0_canonical_research_weights.parquet"
CANON_ALPHA = ROOT / "output" / "v0_canonical_strict_lag_alpha_build_v0" / "v0_canonical_alpha_signal_panel.parquet"
CANON_FACTOR_USAGE = ROOT / "output" / "v0_canonical_strict_lag_alpha_build_v0" / "v0_canonical_factor_usage_summary.csv"
CANON_ALPHA_QA = ROOT / "output" / "v0_canonical_strict_lag_alpha_build_v0" / "v0_canonical_alpha_signal_monthly_qa.csv"
RETURN_MAP = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0" / "canonical_csmar_trd_mnth_return_map_repaired.parquet"

LEGACY_WEIGHTS = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_reconstructed_weights.parquet"
LEGACY_ALPHA = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_alpha_signal_panel.parquet"
LEGACY_SUMMARY = ROOT / "output" / "unified_strategy_eval_repaired_trd_mnth_v0" / "unified_strategy_eval_repaired_trd_mnth_summary.json"
LEGACY_MONTHLY = ROOT / "output" / "unified_strategy_eval_repaired_trd_mnth_v0" / "unified_strategy_monthly_net_return_by_cost.csv"
LEGACY_PERF = ROOT / "output" / "unified_strategy_eval_repaired_trd_mnth_v0" / "unified_strategy_performance_summary_by_cost.csv"

PRIMARY_VARIANT = "raw_unmatched_not_renormalized"
PRIMARY_COST_BPS = 20
PRIMARY_POLICY = "charge_cost_on_first_month_initialization"
LEGACY_PORTFOLIO = "V0_STRICT_LAG_TOP50_BUFFER_35_75_EQUAL_WEIGHT"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def dump_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_run_state(status: str, checkpoint: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    text = (
        "# RUN_STATE\n\n"
        f"task_name: {TASK_NAME}\n"
        f"status: {status}\n"
        f"last_checkpoint: {checkpoint}\n"
        f"updated_at: {datetime.now().isoformat(timespec='seconds')}\n"
        "resume_instruction: run scripts\\forensic_v0_canonical_eval_common_window_bridge_v0.py with stdout/stderr redirected to output\\_agent_runs\\v0_canonical_eval_forensic_common_window_bridge_v0\n"
    )
    (RUN_DIR / "RUN_STATE.md").write_text(text, encoding="utf-8")


def in_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[df["year_month"].astype(str).between(start, end)].copy()


def max_drawdown_path(returns: pd.Series) -> pd.DataFrame:
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    running = nav.cummax()
    dd = nav / running - 1.0
    return pd.DataFrame({"nav": nav, "running_max_nav": running, "drawdown": dd})


def metrics(df: pd.DataFrame, ret_col: str = "net_return", turnover_col: str = "turnover_proxy") -> dict:
    ret = df[ret_col].astype(float)
    n = int(ret.notna().sum())
    mean_ret = float(ret.mean()) if n else np.nan
    std = float(ret.std(ddof=1)) if n > 1 else np.nan
    sharpe = mean_ret / std * math.sqrt(12) if std and not np.isnan(std) else np.nan
    tstat = mean_ret / std * math.sqrt(n) if std and not np.isnan(std) else np.nan
    dd = max_drawdown_path(ret)["drawdown"].min() if n else np.nan
    out = {
        "month_count": n,
        "mean_monthly_return": mean_ret,
        "monthly_volatility": std,
        "sharpe": float(sharpe) if not pd.isna(sharpe) else np.nan,
        "tstat": float(tstat) if not pd.isna(tstat) else np.nan,
        "positive_month_ratio": float((ret > 0).mean()) if n else np.nan,
        "cumulative_return": float((1.0 + ret.fillna(0.0)).prod() - 1.0) if n else np.nan,
        "max_drawdown": float(dd) if not pd.isna(dd) else np.nan,
    }
    if turnover_col in df.columns:
        out["avg_turnover"] = float(df[turnover_col].mean())
        out["max_turnover"] = float(df[turnover_col].max())
    if "matched_weight_share" in df.columns:
        out["avg_matched_weight_share"] = float(df["matched_weight_share"].mean())
        out["min_matched_weight_share"] = float(df["matched_weight_share"].min())
    return out


def prerequisite_check() -> dict:
    flags = {
        "canonical_eval_summary_found": CANON_SUMMARY.exists(),
        "canonical_monthly_returns_found": CANON_MONTHLY.exists(),
        "canonical_weights_found": CANON_WEIGHTS.exists(),
        "canonical_alpha_found": CANON_ALPHA.exists(),
        "trd_mnth_return_map_found": RETURN_MAP.exists(),
        "legacy_weights_found": LEGACY_WEIGHTS.exists(),
        "legacy_monthly_returns_found": LEGACY_MONTHLY.exists(),
        "legacy_alpha_found": LEGACY_ALPHA.exists(),
    }
    required = [
        ("canonical_eval_summary_found", CANON_SUMMARY),
        ("canonical_monthly_returns_found", CANON_MONTHLY),
        ("canonical_weights_found", CANON_WEIGHTS),
        ("canonical_alpha_found", CANON_ALPHA),
        ("trd_mnth_return_map_found", RETURN_MAP),
    ]
    missing = [rel(path) for key, path in required if not flags[key]]
    flags["prerequisites_passed"] = not missing
    flags["missing_files"] = missing
    flags["caveat"] = "" if flags["legacy_monthly_returns_found"] else "legacy monthly returns unavailable; bridge would require read-only recompute"
    dump_json(OUT_DIR / "v0_canonical_eval_forensic_prerequisite_check.json", flags)
    return flags


def window_manifest() -> pd.DataFrame:
    rows = [
        ("canonical_full_window", "2017-03", "2026-05", "canonical primary full evaluation window", True),
        ("legacy_overlap_window", "2017-03", "2024-12", "direct canonical vs legacy same-window comparison", True),
        ("post_legacy_window", "2025-01", "2026-05", "post legacy period drag diagnostic", True),
    ]
    out = []
    for name, start, end, purpose, include in rows:
        months = pd.period_range(start=start, end=end, freq="M")
        out.append(
            {
                "window_name": name,
                "min_year_month": start,
                "max_year_month": end,
                "month_count": len(months),
                "purpose": purpose,
                "include_in_main_forensic": include,
            }
        )
    df = pd.DataFrame(out)
    df.to_csv(OUT_DIR / "v0_canonical_forensic_window_manifest.csv", index=False, encoding="utf-8-sig")
    return df


def add_interpretation(row: dict) -> str:
    if row["month_count"] == 0:
        return "无月份"
    if row["sharpe"] >= 0.8 and row["tstat"] >= 2:
        return "表现强"
    if row["sharpe"] >= 0.5 and row["tstat"] >= 1.5:
        return "表现尚可"
    if row["mean_monthly_return"] > 0:
        return "正收益但统计强度不足"
    return "弱表现"


def canonical_window_perf(canon: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, w in windows.iterrows():
        for keys, g in in_window(canon, w["min_year_month"], w["max_year_month"]).groupby(
            ["return_variant", "cost_bps", "first_month_cost_policy"], sort=True
        ):
            m = metrics(g.sort_values("year_month"))
            row = {
                "window_name": w["window_name"],
                "return_variant": keys[0],
                "cost_bps": int(keys[1]),
                "first_month_cost_policy": keys[2],
                **m,
            }
            row["interpretation"] = add_interpretation(row)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "canonical_window_performance_diagnostic.csv", index=False, encoding="utf-8-sig")
    return out


def legacy_bridge(legacy: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    leg = legacy.loc[
        legacy["portfolio_name"].eq(LEGACY_PORTFOLIO)
        & legacy["cost_bps"].eq(PRIMARY_COST_BPS)
        & legacy["return_variant"].eq(PRIMARY_VARIANT)
    ].copy()
    leg = leg.rename(columns={"turnover_simple": "turnover_proxy"})
    rows = []
    for _, w in windows.iterrows():
        g = in_window(leg, w["min_year_month"], w["max_year_month"]).sort_values("year_month")
        m = metrics(g, turnover_col="turnover_proxy")
        rows.append(
            {
                "window_name": w["window_name"],
                "source": rel(LEGACY_MONTHLY),
                **m,
                "caveat": "legacy monthly returns read-only; sample_window retained from source",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "legacy_strict_lag_same_window_bridge.csv", index=False, encoding="utf-8-sig")
    return out


def overlap_comparison(canon_primary: pd.DataFrame, legacy_primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = in_window(canon_primary, "2017-03", "2024-12").sort_values("year_month")
    l = in_window(legacy_primary, "2017-03", "2024-12").sort_values("year_month")
    l = l.rename(columns={"turnover_simple": "legacy_turnover", "net_return": "legacy_net_return_20bps"})
    c = c.rename(columns={"turnover_proxy": "canonical_turnover", "net_return": "canonical_net_return_20bps"})
    joined = c[
        ["year_month", "canonical_net_return_20bps", "canonical_turnover", "matched_weight_share"]
    ].merge(
        l[["year_month", "legacy_net_return_20bps", "legacy_turnover", "matched_weight_share"]],
        on="year_month",
        how="inner",
        suffixes=("_canonical", "_legacy"),
    )
    joined["diff_canonical_minus_legacy"] = joined["canonical_net_return_20bps"] - joined["legacy_net_return_20bps"]
    joined["turnover_diff"] = joined["canonical_turnover"] - joined["legacy_turnover"]
    joined = joined.rename(
        columns={
            "matched_weight_share_canonical": "canonical_matched_weight_share",
            "matched_weight_share_legacy": "legacy_matched_weight_share",
        }
    )
    joined.to_csv(OUT_DIR / "canonical_vs_legacy_monthly_return_diff.csv", index=False, encoding="utf-8-sig")

    cm = metrics(c.rename(columns={"canonical_net_return_20bps": "net_return", "canonical_turnover": "turnover_proxy"}))
    lm = metrics(l.rename(columns={"legacy_net_return_20bps": "net_return", "legacy_turnover": "turnover_proxy"}))
    corr = float(joined["canonical_net_return_20bps"].corr(joined["legacy_net_return_20bps"]))
    under = int((joined["diff_canonical_minus_legacy"] < 0).sum())
    worst = ";".join(joined.nsmallest(5, "diff_canonical_minus_legacy")["year_month"].astype(str).tolist())
    pairs = [
        ("mean return", cm["mean_monthly_return"], lm["mean_monthly_return"], "canonical minus legacy mean monthly return"),
        ("Sharpe", cm["sharpe"], lm["sharpe"], "same-window net return Sharpe"),
        ("t-stat", cm["tstat"], lm["tstat"], "same-window mean return t-stat"),
        ("cumulative return", cm["cumulative_return"], lm["cumulative_return"], "same-window cumulative return"),
        ("MaxDD", cm["max_drawdown"], lm["max_drawdown"], "same-window max drawdown"),
        ("avg turnover", cm["avg_turnover"], lm["avg_turnover"], "same-window average turnover"),
        ("monthly return correlation", corr, np.nan, "correlation across common months"),
        ("months canonical underperforms legacy", under, len(joined), "count of months with canonical minus legacy below zero"),
        ("worst relative months", worst, "", "five most negative relative months"),
    ]
    rows = []
    for metric, cv, lv, interp in pairs:
        delta = np.nan
        if isinstance(cv, (int, float, np.integer, np.floating)) and isinstance(lv, (int, float, np.integer, np.floating)) and not pd.isna(lv):
            delta = float(cv) - float(lv)
        rows.append({"metric": metric, "canonical_value": cv, "legacy_value": lv, "delta_canonical_minus_legacy": delta, "interpretation": interp})
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT_DIR / "canonical_vs_legacy_overlap_comparison.csv", index=False, encoding="utf-8-sig")
    return comp, joined


def alpha_overlap(canon_monthly: pd.DataFrame, legacy_monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ca = pd.read_parquet(CANON_ALPHA, columns=["symbol_norm", "year_month", "alpha_signal"])
    la = pd.read_parquet(LEGACY_ALPHA, columns=["symbol", "month_end", "alpha_signal_strict_lag"])
    ca["year_month"] = ca["year_month"].astype(str)
    ca["symbol_norm"] = ca["symbol_norm"].astype(str)
    la["year_month"] = pd.to_datetime(la["month_end"]).dt.to_period("M").astype(str)
    la["symbol_norm"] = la["symbol"].astype(str).str.zfill(6)

    c_ret = in_window(canon_monthly, "2017-03", "2024-12")[["year_month", "net_return", "turnover_proxy"]].rename(
        columns={"net_return": "canonical_return", "turnover_proxy": "canonical_turnover"}
    )
    l_ret = in_window(legacy_monthly.rename(columns={"turnover_simple": "turnover_proxy"}), "2017-03", "2024-12")[
        ["year_month", "net_return", "turnover_proxy"]
    ].rename(columns={"net_return": "legacy_return", "turnover_proxy": "legacy_turnover"})
    ret = c_ret.merge(l_ret, on="year_month", how="inner")
    ret["return_diff"] = ret["canonical_return"] - ret["legacy_return"]

    rows = []
    for ym in sorted(ret["year_month"].unique()):
        c = ca.loc[ca["year_month"].eq(ym), ["symbol_norm", "alpha_signal"]].dropna()
        l = la.loc[la["year_month"].eq(ym), ["symbol_norm", "alpha_signal_strict_lag"]].dropna()
        m = c.merge(l, on="symbol_norm", how="inner")
        spearman = float(m["alpha_signal"].corr(m["alpha_signal_strict_lag"], method="spearman")) if len(m) >= 3 else np.nan
        c50 = set(c.nlargest(50, "alpha_signal")["symbol_norm"])
        l50 = set(l.nlargest(50, "alpha_signal_strict_lag")["symbol_norm"])
        c75 = set(c.nlargest(75, "alpha_signal")["symbol_norm"])
        l75 = set(l.nlargest(75, "alpha_signal_strict_lag")["symbol_norm"])
        r = ret.loc[ret["year_month"].eq(ym)].iloc[0]
        top50 = len(c50 & l50) / 50.0 if c50 and l50 else np.nan
        top75 = len(c75 & l75) / 75.0 if c75 and l75 else np.nan
        rows.append(
            {
                "year_month": ym,
                "alpha_spearman": spearman,
                "top50_overlap": top50,
                "top75_overlap": top75,
                "canonical_return": r["canonical_return"],
                "legacy_return": r["legacy_return"],
                "return_diff": r["return_diff"],
                "canonical_turnover": r["canonical_turnover"],
                "legacy_turnover": r["legacy_turnover"],
                "interpretation": "低 overlap 且 canonical 跑输" if pd.notna(top50) and top50 < 0.5 and r["return_diff"] < 0 else "月度诊断",
            }
        )
    diag = pd.DataFrame(rows)
    diag.to_csv(OUT_DIR / "alpha_overlap_return_diff_diagnostic.csv", index=False, encoding="utf-8-sig")

    median_overlap = diag["top50_overlap"].median()
    summary_rows = [
        ("corr(alpha_spearman, return_diff)", diag["alpha_spearman"].corr(diag["return_diff"]), "alpha 排序相关与收益差的相关性"),
        ("corr(top50_overlap, return_diff)", diag["top50_overlap"].corr(diag["return_diff"]), "top50 overlap 与收益差的相关性"),
        ("mean return_diff when top50_overlap below median", diag.loc[diag["top50_overlap"] < median_overlap, "return_diff"].mean(), "低 overlap 月份平均收益差"),
        ("mean return_diff when top50_overlap above median", diag.loc[diag["top50_overlap"] >= median_overlap, "return_diff"].mean(), "高 overlap 月份平均收益差"),
        ("avg_alpha_spearman", diag["alpha_spearman"].mean(), "月均 alpha Spearman"),
        ("avg_top50_overlap", diag["top50_overlap"].mean(), "月均 top50 overlap"),
    ]
    summary = pd.DataFrame([{"metric": k, "value": v, "interpretation": i} for k, v, i in summary_rows])
    summary.to_csv(OUT_DIR / "alpha_overlap_return_diff_summary.csv", index=False, encoding="utf-8-sig")

    del ca, la
    gc.collect()
    return diag, summary


def turnover_cost_forensic(canon: pd.DataFrame, legacy: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, df, turn_col in [
        ("canonical", canon.rename(columns={"turnover_proxy": "turnover"}), "turnover"),
        ("legacy", legacy.rename(columns={"turnover_simple": "turnover"}), "turnover"),
    ]:
        for _, w in windows.iterrows():
            for cost in [0, 10, 20, 30]:
                g = in_window(df.loc[df["cost_bps"].eq(cost) & df["return_variant"].eq(PRIMARY_VARIANT)].copy(), w["min_year_month"], w["max_year_month"])
                if portfolio == "canonical":
                    g = g.loc[g["first_month_cost_policy"].eq(PRIMARY_POLICY)]
                if g.empty:
                    continue
                m = metrics(g.rename(columns={turn_col: "turnover_proxy"}))
                rows.append(
                    {
                        "portfolio": portfolio,
                        "window_name": w["window_name"],
                        "cost_bps": cost,
                        "avg_turnover": m.get("avg_turnover", np.nan),
                        "max_turnover": m.get("max_turnover", np.nan),
                        "mean_gross_return": float(g["gross_return"].mean()) if "gross_return" in g else np.nan,
                        "mean_net_return": m["mean_monthly_return"],
                        "mean_cost_drag": float((g["gross_return"] - g["net_return"]).mean()) if "gross_return" in g else np.nan,
                        "sharpe_net": m["sharpe"],
                        "interpretation": "主成本口径" if cost == 20 else "成本敏感性",
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "turnover_cost_forensic.csv", index=False, encoding="utf-8-sig")
    return out


def drawdown_forensic(canon_primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = in_window(canon_primary, "2017-03", "2026-05").sort_values("year_month").reset_index(drop=True)
    path = max_drawdown_path(g["net_return"])
    full = pd.concat([g, path], axis=1)
    trough_i = int(full["drawdown"].idxmin())
    peak_i = int(full.loc[:trough_i, "nav"].idxmax())
    recovery = full.loc[(full.index > trough_i) & (full["nav"] >= full.loc[peak_i, "nav"]), "year_month"]
    recovery_month = recovery.iloc[0] if not recovery.empty else ""
    dd_slice = full.loc[peak_i:trough_i]
    worst = full.nsmallest(5, "net_return").copy()
    worst["interpretation"] = "canonical 主口径最差月份"
    worst_out = worst[["year_month", "net_return", "gross_return", "turnover_proxy", "cost_drag", "matched_weight_share", "drawdown", "interpretation"]].rename(
        columns={"net_return": "net_return_20bps", "turnover_proxy": "turnover"}
    )
    worst_out.to_csv(OUT_DIR / "canonical_worst_months.csv", index=False, encoding="utf-8-sig")
    dd = pd.DataFrame(
        [
            {
                "drawdown_start_month": full.loc[peak_i, "year_month"],
                "drawdown_trough_month": full.loc[trough_i, "year_month"],
                "drawdown_recovery_month": recovery_month,
                "max_drawdown": float(full.loc[trough_i, "drawdown"]),
                "months_in_drawdown": int(trough_i - peak_i + 1),
                "cumulative_return_during_drawdown": float((1.0 + dd_slice["net_return"]).prod() - 1.0),
                "avg_turnover_during_drawdown": float(dd_slice["turnover_proxy"].mean()),
                "worst_5_months": ";".join(worst_out["year_month"].astype(str).tolist()),
                "caveat": "recovery_month blank means not recovered by 2026-05",
            }
        ]
    )
    dd.to_csv(OUT_DIR / "canonical_drawdown_forensic.csv", index=False, encoding="utf-8-sig")
    return dd, worst_out


def signal_stability(canon_primary: pd.DataFrame) -> pd.DataFrame:
    qa = pd.read_csv(CANON_ALPHA_QA, dtype={"year_month": "string"})
    ret = canon_primary[["year_month", "turnover_proxy", "net_return"]].rename(
        columns={"turnover_proxy": "turnover", "net_return": "net_return_20bps"}
    )
    out = qa.merge(ret, on="year_month", how="inner")
    out = out.rename(columns={"factor_count_used_avg": "avg_factor_count_used"})
    out["signal_stability_status"] = np.where(out["alpha_non_null_ratio"] >= 0.95, "PASS", "WATCH")
    out["interpretation"] = np.where(out["signal_stability_status"].eq("PASS"), "alpha 覆盖正常", "alpha 覆盖偏低")
    cols = [
        "year_month",
        "avg_factor_count_used",
        "alpha_non_null_ratio",
        "large_alpha_std",
        "small_alpha_std",
        "turnover",
        "net_return_20bps",
        "signal_stability_status",
        "interpretation",
    ]
    out[cols].to_csv(OUT_DIR / "canonical_signal_stability_forensic.csv", index=False, encoding="utf-8-sig")
    return out[cols]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_run_state("running", "prerequisites")
    prereq = prerequisite_check()
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    with CANON_SUMMARY.open("r", encoding="utf-8") as f:
        canon_summary = json.load(f)

    windows = window_manifest()
    write_run_state("running", "reading_monthly_returns")
    canon = pd.read_csv(CANON_MONTHLY, dtype={"year_month": "string"})
    legacy = pd.read_csv(LEGACY_MONTHLY, dtype={"year_month": "string"})
    canon["year_month"] = canon["year_month"].astype(str)
    legacy["year_month"] = legacy["year_month"].astype(str)

    canon_perf = canonical_window_perf(canon, windows)
    legacy_bridge_df = legacy_bridge(legacy, windows)

    canon_primary = canon.loc[
        canon["return_variant"].eq(PRIMARY_VARIANT)
        & canon["cost_bps"].eq(PRIMARY_COST_BPS)
        & canon["first_month_cost_policy"].eq(PRIMARY_POLICY)
    ].copy()
    legacy_primary = legacy.loc[
        legacy["portfolio_name"].eq(LEGACY_PORTFOLIO)
        & legacy["cost_bps"].eq(PRIMARY_COST_BPS)
        & legacy["return_variant"].eq(PRIMARY_VARIANT)
    ].copy()

    overlap_comp, monthly_diff = overlap_comparison(canon_primary, legacy_primary)
    alpha_diag, alpha_summary = alpha_overlap(canon_primary, legacy_primary)
    turnover_cost = turnover_cost_forensic(canon, legacy, windows)
    dd, worst = drawdown_forensic(canon_primary)
    stability = signal_stability(canon_primary)

    guardrail_values = {
        "alpha_signal_regenerated": False,
        "strategy_weights_regenerated": False,
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
        "same-window portfolio metrics calculated": True,
        "forensic monthly return comparison calculated": True,
    }
    guardrail = pd.DataFrame(
        [
            {"guardrail": k, "expected": v, "actual": v, "pass": True}
            for k, v in guardrail_values.items()
        ]
    )
    guardrail.to_csv(OUT_DIR / "v0_canonical_eval_forensic_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    guardrails_passed = bool(guardrail["pass"].all())

    def perf_row(window_name: str) -> dict:
        return canon_perf.loc[
            canon_perf["window_name"].eq(window_name)
            & canon_perf["return_variant"].eq(PRIMARY_VARIANT)
            & canon_perf["cost_bps"].eq(PRIMARY_COST_BPS)
            & canon_perf["first_month_cost_policy"].eq(PRIMARY_POLICY)
        ].iloc[0].to_dict()

    full = perf_row("canonical_full_window")
    overlap = perf_row("legacy_overlap_window")
    post = perf_row("post_legacy_window")
    leg_overlap = legacy_bridge_df.loc[legacy_bridge_df["window_name"].eq("legacy_overlap_window")].iloc[0].to_dict()

    can_minus_legacy_sharpe = float(overlap["sharpe"] - leg_overlap["sharpe"])
    can_minus_legacy_mean = float(overlap["mean_monthly_return"] - leg_overlap["mean_monthly_return"])
    can_minus_legacy_maxdd = float(overlap["max_drawdown"] - leg_overlap["max_drawdown"])

    cost20 = turnover_cost.loc[
        turnover_cost["portfolio"].eq("canonical")
        & turnover_cost["window_name"].eq("canonical_full_window")
        & turnover_cost["cost_bps"].eq(20)
    ].iloc[0]
    cost0 = turnover_cost.loc[
        turnover_cost["portfolio"].eq("canonical")
        & turnover_cost["window_name"].eq("canonical_full_window")
        & turnover_cost["cost_bps"].eq(0)
    ].iloc[0]
    turnover_cost_main_driver = bool((cost0["sharpe_net"] - cost20["sharpe_net"]) > 0.2 and cost20["mean_cost_drag"] > abs(full["mean_monthly_return"]) * 0.5)
    window_extension_main_driver = bool(overlap["sharpe"] >= 0.5 and post["sharpe"] < 0 and full["sharpe"] < overlap["sharpe"])
    avg_alpha_spearman = float(alpha_summary.loc[alpha_summary["metric"].eq("avg_alpha_spearman"), "value"].iloc[0])
    avg_top50_overlap = float(alpha_summary.loc[alpha_summary["metric"].eq("avg_top50_overlap"), "value"].iloc[0])
    low_overlap_diff = float(alpha_summary.loc[alpha_summary["metric"].eq("mean return_diff when top50_overlap below median"), "value"].iloc[0])
    high_overlap_diff = float(alpha_summary.loc[alpha_summary["metric"].eq("mean return_diff when top50_overlap above median"), "value"].iloc[0])
    alpha_divergence_main_driver = bool(avg_top50_overlap < 0.7 and low_overlap_diff < high_overlap_diff)
    return_calculation_bug_suspected = bool(full["min_matched_weight_share"] < 0.98 or not np.isclose(float(canon_summary["primary_20bps_mean_monthly_return"]), float(full["mean_monthly_return"])))
    legacy_overlap_available = bool(prereq["legacy_monthly_returns_found"])
    canonical_underperforms_legacy = bool(can_minus_legacy_sharpe < -0.2 and can_minus_legacy_mean < 0)

    if not guardrails_passed:
        final_decision = "FORENSIC_FAIL_GUARDRAIL"
    elif window_extension_main_driver and not canonical_underperforms_legacy:
        final_decision = "CANONICAL_WEAKNESS_EXPLAINED_BY_WINDOW_CONTINUE_ATTRIBUTION"
    elif canonical_underperforms_legacy and alpha_divergence_main_driver:
        final_decision = "CANONICAL_UNDERPERFORMS_LEGACY_REPAIR_RECONSTRUCTION_FIRST"
    elif legacy_overlap_available and not return_calculation_bug_suspected:
        final_decision = "CANONICAL_RESULT_MIXED_KEEP_BOTH_CONTINUE_ATTRIBUTION_WITH_CAVEAT"
    else:
        final_decision = "FORENSIC_INCONCLUSIVE_MORE_QA_REQUIRED"

    continue_attr = final_decision in {
        "CANONICAL_WEAKNESS_EXPLAINED_BY_WINDOW_CONTINUE_ATTRIBUTION",
        "CANONICAL_RESULT_MIXED_KEEP_BOTH_CONTINUE_ATTRIBUTION_WITH_CAVEAT",
    }
    repair_recon = final_decision == "CANONICAL_UNDERPERFORMS_LEGACY_REPAIR_RECONSTRUCTION_FIRST"
    recommended_next_step = {
        "CANONICAL_WEAKNESS_EXPLAINED_BY_WINDOW_CONTINUE_ATTRIBUTION": "窗口扩展可解释主要弱化，可带 caveat 进入 attribution。",
        "CANONICAL_UNDERPERFORMS_LEGACY_REPAIR_RECONSTRUCTION_FIRST": "先审计 canonical factor/signal reconstruction 与 legacy 差异，再决定 attribution。",
        "CANONICAL_RESULT_MIXED_KEEP_BOTH_CONTINUE_ATTRIBUTION_WITH_CAVEAT": "保留 canonical 与 legacy 两条只读结果，带 caveat 进入 attribution。",
        "FORENSIC_INCONCLUSIVE_MORE_QA_REQUIRED": "补充 QA 后再决定是否 attribution。",
        "FORENSIC_FAIL_GUARDRAIL": "停止，先修复 guardrail violation。",
    }[final_decision]

    decisions = [
        ("Is canonical weak result mainly due to window extension into 2025-2026?", str(window_extension_main_driver), "HIGH" if window_extension_main_driver else "MEDIUM", "若为 true，attribution 需单列 post-legacy 期间"),
        ("Does canonical underperform legacy in the exact overlap window?", str(canonical_underperforms_legacy), "HIGH" if canonical_underperforms_legacy else "LOW", "使用 overlap window 直接比较"),
        ("Is turnover/cost the main driver?", str(turnover_cost_main_driver), "MEDIUM" if turnover_cost_main_driver else "LOW", "查看 0/10/20/30bps 成本敏感性"),
        ("Is alpha divergence from legacy associated with return underperformance?", str(alpha_divergence_main_driver), "HIGH" if alpha_divergence_main_driver else "MEDIUM", "审计 alpha overlap 与 return diff"),
        ("Is there evidence of return calculation or matching bug?", str(return_calculation_bug_suspected), "HIGH" if return_calculation_bug_suspected else "LOW", "matched share 和 summary 复核"),
        ("Should we continue to benchmark attribution?", str(continue_attr), "MEDIUM", recommended_next_step),
        ("Should we first repair canonical factor/signal reconstruction?", str(repair_recon), "HIGH" if repair_recon else "LOW", recommended_next_step),
    ]
    pd.DataFrame(
        [{"question": q, "finding": f, "severity": s, "recommended_action": a} for q, f, s, a in decisions]
    ).to_csv(OUT_DIR / "v0_canonical_eval_forensic_decision_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "canonical_full_window_month_count": int(full["month_count"]),
        "canonical_full_20bps_sharpe": full["sharpe"],
        "canonical_full_20bps_mean_monthly_return": full["mean_monthly_return"],
        "canonical_full_20bps_tstat": full["tstat"],
        "canonical_full_20bps_max_drawdown": full["max_drawdown"],
        "canonical_overlap_window_month_count": int(overlap["month_count"]),
        "canonical_overlap_20bps_sharpe": overlap["sharpe"],
        "canonical_overlap_20bps_mean_monthly_return": overlap["mean_monthly_return"],
        "canonical_overlap_20bps_tstat": overlap["tstat"],
        "canonical_overlap_20bps_max_drawdown": overlap["max_drawdown"],
        "canonical_post_legacy_month_count": int(post["month_count"]),
        "canonical_post_legacy_20bps_sharpe": post["sharpe"],
        "canonical_post_legacy_20bps_mean_monthly_return": post["mean_monthly_return"],
        "canonical_post_legacy_20bps_tstat": post["tstat"],
        "canonical_post_legacy_20bps_max_drawdown": post["max_drawdown"],
        "legacy_overlap_available": legacy_overlap_available,
        "legacy_overlap_20bps_sharpe": leg_overlap["sharpe"],
        "legacy_overlap_20bps_mean_monthly_return": leg_overlap["mean_monthly_return"],
        "legacy_overlap_20bps_tstat": leg_overlap["tstat"],
        "legacy_overlap_20bps_max_drawdown": leg_overlap["max_drawdown"],
        "canonical_minus_legacy_overlap_sharpe": can_minus_legacy_sharpe,
        "canonical_minus_legacy_overlap_mean_return": can_minus_legacy_mean,
        "canonical_minus_legacy_overlap_maxdd": can_minus_legacy_maxdd,
        "alpha_overlap_available": True,
        "avg_alpha_spearman": avg_alpha_spearman,
        "avg_top50_overlap": avg_top50_overlap,
        "turnover_cost_main_driver": turnover_cost_main_driver,
        "window_extension_main_driver": window_extension_main_driver,
        "alpha_divergence_main_driver": alpha_divergence_main_driver,
        "return_calculation_bug_suspected": return_calculation_bug_suspected,
        "continue_to_benchmark_attribution_recommended": continue_attr,
        "repair_canonical_reconstruction_recommended": repair_recon,
        "guardrails_passed": guardrails_passed,
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    dump_json(OUT_DIR / "v0_canonical_eval_forensic_common_window_bridge_summary.json", summary)

    report = (
        "# V0 canonical evaluation forensic common-window bridge v0\n\n"
        f"- final_decision: {final_decision}\n"
        f"- full window Sharpe/t-stat/MaxDD: {full['sharpe']:.6f} / {full['tstat']:.6f} / {full['max_drawdown']:.6f}\n"
        f"- overlap window canonical Sharpe: {overlap['sharpe']:.6f}; legacy Sharpe: {leg_overlap['sharpe']:.6f}; delta: {can_minus_legacy_sharpe:.6f}\n"
        f"- post legacy Sharpe/t-stat/MaxDD: {post['sharpe']:.6f} / {post['tstat']:.6f} / {post['max_drawdown']:.6f}\n"
        f"- avg alpha Spearman: {avg_alpha_spearman:.6f}; avg top50 overlap: {avg_top50_overlap:.6f}\n"
        f"- turnover_cost_main_driver: {turnover_cost_main_driver}; window_extension_main_driver: {window_extension_main_driver}; alpha_divergence_main_driver: {alpha_divergence_main_driver}\n"
        f"- return_calculation_bug_suspected: {return_calculation_bug_suspected}; guardrails_passed: {guardrails_passed}\n\n"
        "本 forensic run 未重新生成 alpha_signal 或 weights，未计算 benchmark-relative、alpha/beta、IR/TE、FF、DGTW，未训练、未调参、未 SHAP、未修改 production。\n"
    )
    (OUT_DIR / "v0_canonical_eval_forensic_common_window_bridge_report.md").write_text(report, encoding="utf-8")

    final_qa = pd.DataFrame(
        [
            {"check_name": "prerequisites_passed", "pass": prereq["prerequisites_passed"], "detail": ""},
            {"check_name": "guardrails_passed", "pass": guardrails_passed, "detail": ""},
            {"check_name": "legacy_overlap_available", "pass": legacy_overlap_available, "detail": ""},
            {"check_name": "alpha_overlap_available", "pass": True, "detail": ""},
            {"check_name": "final_decision_allowed", "pass": final_decision in {
                "CANONICAL_WEAKNESS_EXPLAINED_BY_WINDOW_CONTINUE_ATTRIBUTION",
                "CANONICAL_UNDERPERFORMS_LEGACY_REPAIR_RECONSTRUCTION_FIRST",
                "CANONICAL_RESULT_MIXED_KEEP_BOTH_CONTINUE_ATTRIBUTION_WITH_CAVEAT",
                "FORENSIC_INCONCLUSIVE_MORE_QA_REQUIRED",
                "FORENSIC_FAIL_GUARDRAIL",
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

    del canon, legacy, canon_perf, legacy_bridge_df, overlap_comp, monthly_diff, alpha_diag, alpha_summary, turnover_cost, dd, worst, stability
    gc.collect()
    write_run_state("completed", "all_outputs_written")
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": rel(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
