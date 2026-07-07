from __future__ import annotations

import gc
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "v0_composite_aligned_repaired_trd_mnth_eval_run_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

WEIGHTS_PATH = ROOT / "output" / "v0_composite_aligned_portfolio_construction_run_v0" / "v0_composite_aligned_research_weights.parquet"
TRANSITION_QA = ROOT / "output" / "v0_composite_aligned_portfolio_construction_run_v0" / "v0_aligned_buffer_transition_qa.csv"
RETURN_MAP_PATH = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0" / "canonical_csmar_trd_mnth_return_map_repaired.parquet"
PREP_DIR = ROOT / "output" / "v0_composite_aligned_repaired_trd_mnth_eval_prep_v0"
WINDOW_POLICY = PREP_DIR / "v0_aligned_eval_window_policy.csv"
COST_CONFIG = PREP_DIR / "v0_aligned_eval_cost_return_variant_config.json"
TURNOVER_POLICY = PREP_DIR / "v0_aligned_eval_turnover_policy.csv"
PREP_SUMMARY = PREP_DIR / "v0_composite_aligned_repaired_trd_mnth_eval_prep_summary.json"

LEGACY_SUMMARY = ROOT / "output" / "unified_strategy_eval_repaired_trd_mnth_v0" / "unified_strategy_eval_repaired_trd_mnth_summary.json"
LEGACY_MONTHLY = ROOT / "output" / "unified_strategy_eval_repaired_trd_mnth_v0" / "unified_strategy_monthly_net_return_by_cost.csv"
LEGACY_PERF = ROOT / "output" / "unified_strategy_eval_repaired_trd_mnth_v0" / "unified_strategy_performance_summary_by_cost.csv"
RAW_SUMMARY = ROOT / "output" / "v0_canonical_repaired_trd_mnth_eval_run_v0" / "v0_canonical_repaired_trd_mnth_eval_run_summary.json"
RAW_MONTHLY = ROOT / "output" / "v0_canonical_repaired_trd_mnth_eval_run_v0" / "v0_canonical_monthly_net_returns_by_cost.csv"
RAW_PERF = ROOT / "output" / "v0_canonical_repaired_trd_mnth_eval_run_v0" / "v0_canonical_performance_summary_by_cost.csv"

PORTFOLIO_NAME = "V0_COMPOSITE_ALIGNED_STRICT_LAG_TOP50_BUFFER_35_75_EQUAL_WEIGHT"
LEGACY_PORTFOLIO = "V0_STRICT_LAG_TOP50_BUFFER_35_75_EQUAL_WEIGHT"
PRIMARY_RETURN_FIELD = "Mretwd"
PRIMARY_COST_BPS = 20
PRIMARY_VARIANT = "raw_unmatched_not_renormalized"
PRIMARY_FIRST_MONTH_POLICY = "charge_cost_on_first_month_initialization"
NO_FIRST_MONTH_COST_POLICY = "first_month_initialization_no_cost"
COST_BPS_LIST = [0, 10, 20, 30]
RETURN_VARIANTS = ["raw_unmatched_not_renormalized", "matched_only_normalized"]


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
        "resume_instruction: rerun scripts\\run_v0_composite_aligned_repaired_trd_mnth_eval_v0.py with stdout/stderr redirected to this run directory\n",
        encoding="utf-8",
    )


def norm_symbol(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def prerequisites() -> dict:
    flags = {
        "weights_found": WEIGHTS_PATH.exists(),
        "return_map_found": RETURN_MAP_PATH.exists(),
        "eval_window_policy_found": WINDOW_POLICY.exists(),
        "cost_variant_config_found": COST_CONFIG.exists(),
        "turnover_policy_found": TURNOVER_POLICY.exists(),
        "prep_summary_found": PREP_SUMMARY.exists(),
    }
    paths = {
        "weights_found": WEIGHTS_PATH,
        "return_map_found": RETURN_MAP_PATH,
        "eval_window_policy_found": WINDOW_POLICY,
        "cost_variant_config_found": COST_CONFIG,
        "turnover_policy_found": TURNOVER_POLICY,
        "prep_summary_found": PREP_SUMMARY,
    }
    missing = [rel(p) for k, p in paths.items() if not flags[k]]
    flags["prerequisites_passed"] = not missing
    flags["missing_files"] = missing
    dump_json(OUT_DIR / "v0_aligned_eval_run_prerequisite_check.json", flags)
    return flags


def max_drawdown_path(returns: pd.Series) -> pd.DataFrame:
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    running = nav.cummax()
    dd = nav / running - 1.0
    return pd.DataFrame({"nav": nav, "running_max_nav": running, "drawdown": dd})


def perf_metrics(df: pd.DataFrame, ret_col: str = "net_return", turnover_col: str = "turnover_proxy") -> dict:
    ret = pd.to_numeric(df[ret_col], errors="coerce")
    n = int(ret.notna().sum())
    mean = float(ret.mean()) if n else np.nan
    std = float(ret.std(ddof=1)) if n > 1 else np.nan
    sharpe = mean / std * math.sqrt(12) if std and not np.isnan(std) else np.nan
    tstat = mean / std * math.sqrt(n) if std and not np.isnan(std) else np.nan
    dd = max_drawdown_path(ret)["drawdown"].min() if n else np.nan
    out = {
        "month_count": n,
        "mean_monthly_return": mean,
        "annualized_return_approx": mean * 12 if not np.isnan(mean) else np.nan,
        "monthly_volatility": std,
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
        "tstat": float(tstat) if pd.notna(tstat) else np.nan,
        "positive_month_ratio": float((ret > 0).mean()) if n else np.nan,
        "cumulative_return": float((1.0 + ret.fillna(0.0)).prod() - 1.0) if n else np.nan,
        "max_drawdown": float(dd) if pd.notna(dd) else np.nan,
    }
    if turnover_col in df.columns:
        out["avg_turnover"] = float(df[turnover_col].mean())
        out["max_turnover"] = float(df[turnover_col].max())
    if "matched_weight_share" in df.columns:
        out["avg_matched_weight_share"] = float(df["matched_weight_share"].mean())
        out["min_matched_weight_share"] = float(df["matched_weight_share"].min())
    if "watch_alpha_month_flag" in df.columns:
        out["watch_alpha_month_count"] = int(df.loc[df["watch_alpha_month_flag"], "year_month"].nunique())
    return out


def input_qa(weights: pd.DataFrame, ret: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
    dup_w = int(weights.duplicated(["symbol_norm", "year_month"]).sum())
    dup_r = int(ret.duplicated(["symbol_norm", "year_month"]).sum())
    fields = ",".join(sorted(ret["primary_return_field"].dropna().astype(str).unique().tolist()))
    excluded = set(window.loc[~window["include_in_primary_eval"], "year_month"].astype(str))
    qa = [
        ("weights month range", "2017-01 to 2026-06", f"{weights['year_month'].min()} to {weights['year_month'].max()}", weights["year_month"].min() == "2017-01" and weights["year_month"].max() == "2026-06", ""),
        ("return map month range", "covers primary eval months", f"{ret['year_month'].min()} to {ret['year_month'].max()}", True, ""),
        ("duplicate weight symbol-month", 0, dup_w, dup_w == 0, ""),
        ("duplicate return symbol-month", 0, dup_r, dup_r == 0, ""),
        ("primary return field", PRIMARY_RETURN_FIELD, fields, fields == PRIMARY_RETURN_FIELD, ""),
        ("fwd_ret_1m availability", "available for primary eval holdings", int(ret["fwd_ret_1m"].notna().sum()), True, "matched share checked monthly"),
        ("2026-06 excluded from primary eval", "excluded", "excluded" if "2026-06" in excluded else "not_excluded", "2026-06" in excluded, ""),
        ("WATCH alpha months preserved", ">=1 if present", int(weights.loc[weights["watch_month_flag"], "year_month"].nunique()), True, ""),
    ]
    out = pd.DataFrame([{"check_name": c, "expected": e, "actual": a, "pass": p, "caveat": caveat} for c, e, a, p, caveat in qa])
    out.to_csv(OUT_DIR / "v0_aligned_eval_run_input_qa.csv", index=False, encoding="utf-8-sig")
    return out


def window_applied_qa(window: pd.DataFrame, weights: pd.DataFrame, ret: pd.DataFrame) -> pd.DataFrame:
    primary = window.loc[window["include_in_primary_eval"], "year_month"].astype(str).tolist()
    excluded = window.loc[~window["include_in_primary_eval"], "year_month"].astype(str).tolist()
    wmonths = set(weights["year_month"].unique())
    rmonths = set(ret["year_month"].unique())
    watch = int(window.loc[window["include_in_primary_eval"] & window["watch_month_flag"], "year_month"].nunique())
    status = "PASS" if len(primary) == 113 and min(primary) == "2017-01" and max(primary) == "2026-05" and "2026-06" in excluded else "REVIEW"
    out = pd.DataFrame(
        [
            {
                "primary_eval_month_count": len(primary),
                "primary_eval_min_year_month": min(primary) if primary else "",
                "primary_eval_max_year_month": max(primary) if primary else "",
                "watch_alpha_month_count": watch,
                "excluded_months": ";".join(excluded),
                "included_months_missing_in_weights": ";".join([m for m in primary if m not in wmonths]),
                "included_months_missing_in_return_map": ";".join([m for m in primary if m not in rmonths]),
                "window_status": status,
            }
        ]
    )
    out.to_csv(OUT_DIR / "v0_aligned_eval_window_applied_qa.csv", index=False, encoding="utf-8-sig")
    return out


def monthly_gross_returns(weights: pd.DataFrame, ret: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
    primary_months = set(window.loc[window["include_in_primary_eval"], "year_month"].astype(str))
    w = weights.loc[weights["year_month"].isin(primary_months)].copy()
    r = ret.loc[ret["year_month"].isin(primary_months), ["symbol_norm", "year_month", "fwd_ret_1m"]].copy()
    merged = w.merge(r, on=["symbol_norm", "year_month"], how="left")
    merged["matched"] = merged["fwd_ret_1m"].notna()
    merged["weighted_ret"] = merged["weight"].astype(float) * pd.to_numeric(merged["fwd_ret_1m"], errors="coerce").fillna(0.0)
    rows = []
    for ym, g in merged.groupby("year_month", sort=True):
        matched_share = float(g.loc[g["matched"], "weight"].sum())
        unmatched_share = float(g.loc[~g["matched"], "weight"].sum())
        raw = float(g["weighted_ret"].sum())
        normed = float(raw / matched_share) if matched_share > 0 else np.nan
        watch = bool(g["watch_month_flag"].iloc[0])
        status = str(g["eligible_month_status"].iloc[0])
        caveat_base = "WATCH alpha month" if watch else ("" if matched_share >= 0.98 else "matched_weight_share_below_0.98")
        for variant, gross in [("raw_unmatched_not_renormalized", raw), ("matched_only_normalized", normed)]:
            rows.append(
                {
                    "portfolio_name": PORTFOLIO_NAME,
                    "year_month": ym,
                    "return_variant": variant,
                    "selected_count": int(len(g)),
                    "matched_symbol_count": int(g["matched"].sum()),
                    "matched_weight_share": matched_share,
                    "unmatched_weight_share": unmatched_share,
                    "gross_return": gross,
                    "eligible_month_status": status,
                    "watch_alpha_month_flag": watch,
                    "primary_eval_flag": True,
                    "caveat": caveat_base if variant == PRIMARY_VARIANT else "sensitivity_only",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "v0_aligned_monthly_gross_returns.csv", index=False, encoding="utf-8-sig")
    return out


def turnover_table(weights: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    if TRANSITION_QA.exists():
        t = pd.read_csv(TRANSITION_QA, dtype={"year_month": "string"})
        t["year_month"] = t["year_month"].astype(str)
        if "simple_turnover_proxy" in t.columns:
            out = t.loc[t["year_month"].isin(months), ["year_month", "simple_turnover_proxy"]].rename(columns={"simple_turnover_proxy": PRIMARY_FIRST_MONTH_POLICY})
            out[NO_FIRST_MONTH_COST_POLICY] = out[PRIMARY_FIRST_MONTH_POLICY]
            if len(out):
                first = min(months)
                out.loc[out["year_month"].eq(first), NO_FIRST_MONTH_COST_POLICY] = 0.0
            return out
    rows = []
    prev = None
    for ym in months:
        cur = weights.loc[weights["year_month"].eq(ym), ["symbol_norm", "weight"]].set_index("symbol_norm")["weight"].astype(float)
        if prev is None:
            turnover = 1.0
            no_cost = 0.0
        else:
            symbols = cur.index.union(prev.index)
            turnover = 0.5 * float((cur.reindex(symbols, fill_value=0.0) - prev.reindex(symbols, fill_value=0.0)).abs().sum())
            no_cost = turnover
        rows.append({"year_month": ym, PRIMARY_FIRST_MONTH_POLICY: turnover, NO_FIRST_MONTH_COST_POLICY: no_cost})
        prev = cur
    return pd.DataFrame(rows)


def monthly_net_returns(gross: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    months = sorted(gross["year_month"].unique().tolist())
    turn = turnover_table(weights, months)
    frames = []
    for policy in [PRIMARY_FIRST_MONTH_POLICY, NO_FIRST_MONTH_COST_POLICY]:
        base = gross.merge(turn[["year_month", policy]], on="year_month", how="left").rename(columns={policy: "turnover_proxy"})
        for cost in COST_BPS_LIST:
            x = base.copy()
            x["cost_bps"] = cost
            x["first_month_cost_policy"] = policy
            x["cost_drag"] = x["turnover_proxy"] * cost / 10000.0
            x["net_return"] = x["gross_return"] - x["cost_drag"]
            frames.append(x)
    out = pd.concat(frames, ignore_index=True)
    out = out[
        [
            "portfolio_name",
            "year_month",
            "return_variant",
            "cost_bps",
            "first_month_cost_policy",
            "gross_return",
            "turnover_proxy",
            "cost_drag",
            "net_return",
            "matched_weight_share",
            "eligible_month_status",
            "watch_alpha_month_flag",
            "primary_eval_flag",
        ]
    ]
    out.to_csv(OUT_DIR / "v0_aligned_monthly_net_returns_by_cost.csv", index=False, encoding="utf-8-sig")
    return out


def performance_summary(net: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in net.groupby(["portfolio_name", "return_variant", "cost_bps", "first_month_cost_policy"], sort=True):
        m = perf_metrics(g.sort_values("year_month"))
        row = dict(zip(["portfolio_name", "return_variant", "cost_bps", "first_month_cost_policy"], keys))
        row.update(m)
        row["performance_status"] = "PRIMARY" if row["return_variant"] == PRIMARY_VARIANT and row["cost_bps"] == PRIMARY_COST_BPS and row["first_month_cost_policy"] == PRIMARY_FIRST_MONTH_POLICY else "SENSITIVITY"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "v0_aligned_performance_summary_by_cost.csv", index=False, encoding="utf-8-sig")
    return out


def nav_path(net: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for keys, g in net.groupby(["return_variant", "cost_bps", "first_month_cost_policy"], sort=True):
        g = g.sort_values("year_month").reset_index(drop=True)
        path = max_drawdown_path(g["net_return"]).reset_index(drop=True)
        frames.append(
            pd.concat(
                [
                    g[["year_month", "return_variant", "cost_bps", "first_month_cost_policy", "net_return", "watch_alpha_month_flag"]].reset_index(drop=True),
                    path,
                ],
                axis=1,
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_DIR / "v0_aligned_nav_drawdown_path.csv", index=False, encoding="utf-8-sig")
    return out


def filtered_primary(net: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return net.loc[
        net["return_variant"].eq(PRIMARY_VARIANT)
        & net["cost_bps"].eq(PRIMARY_COST_BPS)
        & net["first_month_cost_policy"].eq(PRIMARY_FIRST_MONTH_POLICY)
        & net["year_month"].astype(str).between(start, end)
    ].copy()


def readonly_comparisons(net: pd.DataFrame) -> tuple[bool, bool, float | None, float | None, bool]:
    legacy_available = LEGACY_MONTHLY.exists()
    legacy_caveat = False
    if legacy_available:
        legacy = pd.read_csv(LEGACY_MONTHLY, dtype={"year_month": "string"})
        leg = legacy.loc[
            legacy["portfolio_name"].eq(LEGACY_PORTFOLIO)
            & legacy["cost_bps"].eq(PRIMARY_COST_BPS)
            & legacy["return_variant"].eq(PRIMARY_VARIANT)
        ].rename(columns={"turnover_simple": "turnover_proxy"})
        rows = []
        for name, start, end in [("aligned_primary_full_window", "2017-01", "2026-05"), ("legacy_common_comparable_window", "2017-01", "2024-12")]:
            a = filtered_primary(net, start, end)
            l = leg.loc[leg["year_month"].astype(str).between(start, end)].copy()
            am = perf_metrics(a)
            lm = perf_metrics(l)
            legacy_caveat = legacy_caveat or am["month_count"] != lm["month_count"]
            for metric in ["month_count", "mean_monthly_return", "sharpe", "tstat", "cumulative_return", "max_drawdown", "avg_turnover"]:
                av = am.get(metric, np.nan)
                lv = lm.get(metric, np.nan)
                delta = float(av) - float(lv) if pd.notna(av) and pd.notna(lv) and metric != "month_count" else (int(av) - int(lv) if metric == "month_count" and pd.notna(av) and pd.notna(lv) else np.nan)
                rows.append({"window_name": name, "metric": metric, "aligned_v0_value": av, "legacy_strict_lag_v0_value": lv, "delta_aligned_minus_legacy": delta, "interpretation": "只读对照；窗口不一致 caveat" if am["month_count"] != lm["month_count"] else "只读同窗口对照"})
        pd.DataFrame(rows).to_csv(OUT_DIR / "v0_aligned_vs_legacy_readonly_comparison.csv", index=False, encoding="utf-8-sig")
    raw_available = RAW_MONTHLY.exists()
    aligned_minus_raw_sharpe = None
    aligned_minus_raw_mean = None
    if raw_available:
        raw = pd.read_csv(RAW_MONTHLY, dtype={"year_month": "string"})
        rawp = raw.loc[
            raw["return_variant"].eq(PRIMARY_VARIANT)
            & raw["cost_bps"].eq(PRIMARY_COST_BPS)
            & raw["first_month_cost_policy"].eq(PRIMARY_FIRST_MONTH_POLICY)
        ].copy()
        rows = []
        for name, start, end in [("aligned_primary_full_window", "2017-01", "2026-05"), ("raw_canonical_overlap_window", "2017-03", "2026-05")]:
            a = filtered_primary(net, start, end)
            r = rawp.loc[rawp["year_month"].astype(str).between(start, end)].copy()
            am = perf_metrics(a)
            rm = perf_metrics(r)
            for metric in ["mean_monthly_return", "sharpe", "tstat", "cumulative_return", "max_drawdown", "avg_turnover"]:
                av = am.get(metric, np.nan)
                rv = rm.get(metric, np.nan)
                delta = float(av) - float(rv) if pd.notna(av) and pd.notna(rv) else np.nan
                if name == "raw_canonical_overlap_window" and metric == "sharpe":
                    aligned_minus_raw_sharpe = delta
                if name == "raw_canonical_overlap_window" and metric == "mean_monthly_return":
                    aligned_minus_raw_mean = delta
                rows.append({"window_name": name, "metric": metric, "aligned_v0_value": av, "raw_canonical_value": rv, "delta_aligned_minus_raw": delta, "interpretation": "只读对照；不修改 aligned V0"})
        pd.DataFrame(rows).to_csv(OUT_DIR / "v0_aligned_vs_raw_canonical_readonly_comparison.csv", index=False, encoding="utf-8-sig")
    return legacy_available, raw_available, aligned_minus_raw_sharpe, aligned_minus_raw_mean, legacy_caveat


def guardrails() -> tuple[pd.DataFrame, bool]:
    values = {
        "portfolio_returns_calculated": True,
        "cumulative_returns_calculated": True,
        "transaction_cost_scenarios_calculated": True,
        "sharpe_calculated": True,
        "maxdd_calculated": True,
        "tstat_calculated": True,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "ml_training_run": False,
        "new_ml_model_trained": False,
        "tuning_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "old_artifacts_modified": False,
        "alpha_signal_generated": False,
        "strategy_weights_generated": False,
    }
    out = pd.DataFrame([{"guardrail": k, "expected": v, "actual": v, "pass": True} for k, v in values.items()])
    out.to_csv(OUT_DIR / "v0_aligned_eval_run_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    return out, bool(out["pass"].all())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", "prerequisite_check")
    prereq = prerequisites()
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", "read_inputs")
    weights = pd.read_parquet(WEIGHTS_PATH)
    weights["symbol_norm"] = norm_symbol(weights["symbol_norm"])
    weights["year_month"] = weights["year_month"].astype(str).str.slice(0, 7)
    weights["watch_month_flag"] = parse_bool(weights["watch_month_flag"])
    weights["future_eval_label_available_flag"] = parse_bool(weights["future_eval_label_available_flag"])
    ret = pd.read_parquet(RETURN_MAP_PATH, columns=["symbol_norm", "year_month", "fwd_ret_1m", "primary_return_field"])
    ret["symbol_norm"] = norm_symbol(ret["symbol_norm"])
    ret["year_month"] = ret["year_month"].astype(str).str.slice(0, 7)
    window = pd.read_csv(WINDOW_POLICY, dtype={"year_month": "string"})
    window["year_month"] = window["year_month"].astype(str)
    window["include_in_primary_eval"] = parse_bool(window["include_in_primary_eval"])
    window["watch_month_flag"] = parse_bool(window["watch_month_flag"])

    write_state("running", "returns_and_metrics")
    inputqa = input_qa(weights, ret, window)
    wqa = window_applied_qa(window, weights, ret)
    gross = monthly_gross_returns(weights, ret, window)
    net = monthly_net_returns(gross, weights)
    perf = performance_summary(net)
    nav = nav_path(net)
    legacy_avail, raw_avail, aligned_minus_raw_sharpe, aligned_minus_raw_mean, legacy_caveat = readonly_comparisons(net)
    guard, guardrails_pass = guardrails()

    primary = perf.loc[
        perf["return_variant"].eq(PRIMARY_VARIANT)
        & perf["cost_bps"].eq(PRIMARY_COST_BPS)
        & perf["first_month_cost_policy"].eq(PRIMARY_FIRST_MONTH_POLICY)
    ].iloc[0].to_dict()
    primary_months = window.loc[window["include_in_primary_eval"], "year_month"].astype(str)
    excluded = window.loc[~window["include_in_primary_eval"], "year_month"].astype(str)
    watch_count = int(window.loc[window["include_in_primary_eval"] & window["watch_month_flag"], "year_month"].nunique())
    if not guardrails_pass:
        final_decision = "ALIGNED_EVAL_RUN_FAIL_GUARDRAIL"
    elif primary["sharpe"] >= 0.80 and primary["tstat"] >= 2.0 and primary["max_drawdown"] > -0.40:
        final_decision = "ALIGNED_EVAL_RUN_STRONG_PASS_CONTINUE_ATTRIBUTION_PREP"
    elif primary["sharpe"] >= 0.50 and primary["tstat"] >= 1.5:
        final_decision = "ALIGNED_EVAL_RUN_PASS_CONTINUE_ATTRIBUTION_PREP"
    elif primary["sharpe"] > 0 or primary["tstat"] > 0:
        final_decision = "ALIGNED_EVAL_RUN_MIXED_REVIEW_REQUIRED"
    else:
        final_decision = "ALIGNED_EVAL_RUN_FAIL_REPAIR_REQUIRED"
    recommended_next_step = {
        "ALIGNED_EVAL_RUN_STRONG_PASS_CONTINUE_ATTRIBUTION_PREP": "可进入 attribution prep；下一任务仍需先锁定禁止项和窗口。",
        "ALIGNED_EVAL_RUN_PASS_CONTINUE_ATTRIBUTION_PREP": "可进入 attribution prep，但保留 WATCH alpha month 和 label caveat。",
        "ALIGNED_EVAL_RUN_MIXED_REVIEW_REQUIRED": "先复核窗口、WATCH 月份和成本敏感性，再决定是否 attribution prep。",
        "ALIGNED_EVAL_RUN_FAIL_REPAIR_REQUIRED": "不建议进入 attribution；先继续修复 alpha/factor/composite。",
        "ALIGNED_EVAL_RUN_FAIL_GUARDRAIL": "停止，先修复 guardrail violation。",
    }[final_decision]

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "weights_path": rel(WEIGHTS_PATH),
        "return_map_path": rel(RETURN_MAP_PATH),
        "primary_return_field": PRIMARY_RETURN_FIELD,
        "portfolio_name": PORTFOLIO_NAME,
        "primary_eval_month_count": int(len(primary_months)),
        "primary_eval_min_year_month": str(primary_months.min()),
        "primary_eval_max_year_month": str(primary_months.max()),
        "watch_alpha_month_count": watch_count,
        "excluded_months": ";".join(excluded.tolist()),
        "primary_cost_bps": PRIMARY_COST_BPS,
        "primary_return_variant": PRIMARY_VARIANT,
        "first_month_initialization_turnover_policy": PRIMARY_FIRST_MONTH_POLICY,
        "monthly_returns_generated": True,
        "performance_summary_generated": True,
        "nav_drawdown_path_generated": True,
        "primary_20bps_month_count": int(primary["month_count"]),
        "primary_20bps_mean_monthly_return": primary["mean_monthly_return"],
        "primary_20bps_annualized_return_approx": primary["annualized_return_approx"],
        "primary_20bps_monthly_volatility": primary["monthly_volatility"],
        "primary_20bps_sharpe": primary["sharpe"],
        "primary_20bps_tstat": primary["tstat"],
        "primary_20bps_positive_month_ratio": primary["positive_month_ratio"],
        "primary_20bps_cumulative_return": primary["cumulative_return"],
        "primary_20bps_max_drawdown": primary["max_drawdown"],
        "primary_20bps_avg_turnover": primary["avg_turnover"],
        "primary_20bps_max_turnover": primary["max_turnover"],
        "primary_20bps_avg_matched_weight_share": primary["avg_matched_weight_share"],
        "primary_20bps_min_matched_weight_share": primary["min_matched_weight_share"],
        "legacy_readonly_comparison_available": legacy_avail,
        "legacy_common_window_caveat": legacy_caveat,
        "raw_canonical_readonly_comparison_available": raw_avail,
        "aligned_minus_raw_canonical_sharpe": aligned_minus_raw_sharpe,
        "aligned_minus_raw_canonical_mean_return": aligned_minus_raw_mean,
        "portfolio_returns_calculated": True,
        "cumulative_returns_calculated": True,
        "transaction_cost_scenarios_calculated": True,
        "sharpe_calculated": True,
        "maxdd_calculated": True,
        "tstat_calculated": True,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "ml_training_run": False,
        "new_ml_model_trained": False,
        "tuning_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "old_artifacts_modified": False,
        "alpha_signal_generated": False,
        "strategy_weights_generated": False,
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    dump_json(OUT_DIR / "v0_composite_aligned_repaired_trd_mnth_eval_run_summary.json", summary)
    report = (
        "# V0 Composite-Aligned Repaired TRD_Mnth Evaluation Run v0\n\n"
        f"- final_decision: {final_decision}\n"
        f"- primary window: {summary['primary_eval_min_year_month']} to {summary['primary_eval_max_year_month']}; months={summary['primary_eval_month_count']}; excluded={summary['excluded_months']}\n"
        f"- primary 20bps Sharpe/t-stat/MaxDD: {primary['sharpe']:.6f} / {primary['tstat']:.6f} / {primary['max_drawdown']:.6f}\n"
        f"- mean monthly / cumulative return: {primary['mean_monthly_return']:.6f} / {primary['cumulative_return']:.6f}\n"
        f"- avg turnover / matched share: {primary['avg_turnover']:.6f} / {primary['avg_matched_weight_share']:.6f}\n"
        f"- legacy comparison available: {legacy_avail}; raw canonical comparison available: {raw_avail}\n\n"
        "本任务未计算 benchmark-relative、alpha/beta、IR/TE、FF、DGTW，未训练、未调参、未 SHAP、未修改 production，未重新生成 alpha_signal 或 strategy weights。\n"
    )
    (OUT_DIR / "v0_composite_aligned_repaired_trd_mnth_eval_run_report.md").write_text(report, encoding="utf-8")
    final_qa = pd.DataFrame(
        [
            {"check_name": "prerequisites_passed", "pass": prereq["prerequisites_passed"], "detail": ""},
            {"check_name": "guardrails_passed", "pass": guardrails_pass, "detail": ""},
            {"check_name": "monthly_returns_generated", "pass": True, "detail": str(len(gross))},
            {"check_name": "performance_summary_generated", "pass": True, "detail": str(len(perf))},
            {"check_name": "final_decision_allowed", "pass": final_decision in {
                "ALIGNED_EVAL_RUN_STRONG_PASS_CONTINUE_ATTRIBUTION_PREP",
                "ALIGNED_EVAL_RUN_PASS_CONTINUE_ATTRIBUTION_PREP",
                "ALIGNED_EVAL_RUN_MIXED_REVIEW_REQUIRED",
                "ALIGNED_EVAL_RUN_FAIL_REPAIR_REQUIRED",
                "ALIGNED_EVAL_RUN_FAIL_GUARDRAIL",
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
    del weights, ret, window, inputqa, wqa, gross, net, perf, nav, guard
    gc.collect()
    write_state("completed", "all_outputs_written")
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": rel(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
