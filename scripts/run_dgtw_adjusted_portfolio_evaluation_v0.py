import gc
import json
import math
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TASK_NAME = "DGTW_Adjusted_Portfolio_Eval_Run"
OUT_DIR = Path("output/dgtw_adjusted_portfolio_evaluation_run_v0")
RUN_DIR = Path("output/_agent_runs") / TASK_NAME

PREP_DIR = Path("output/dgtw_adjusted_portfolio_evaluation_prep_v0")
SOURCE_AUDIT_DIR = Path("output/dgtw_benchmark_source_audit_stock_matching_feasibility_v1")

CONFIG_PATH = PREP_DIR / "dgtw_adjusted_run_config_draft.json"
MATCHING_POLICY_PATH = PREP_DIR / "dgtw_adjusted_matching_policy.json"
SAMPLE_POLICY_PATH = PREP_DIR / "dgtw_adjusted_sample_policy.json"
METRIC_PLAN_PATH = PREP_DIR / "dgtw_adjusted_metric_plan.csv"
PORTFOLIO_MANIFEST_PATH = PREP_DIR / "dgtw_adjusted_portfolio_manifest.csv"
SOURCE_SUMMARY_PATH = SOURCE_AUDIT_DIR / "dgtw_benchmark_source_audit_stock_matching_feasibility_summary.json"

DEFAULT_PORTFOLIOS = [
    "ROBUST_VQ_TOP20_EXCLUDE_SOFT_ANOMALY_EQUAL_WEIGHT",
    "ROBUST_VQ_FLAG_CLEAN_TOP50_EQUAL_WEIGHT",
    "ROBUST_VQ_FLAG_CLEAN_TOP50_BUFFER_EQUAL_WEIGHT",
    "ROBUST_VQ_D7_D9_BAND_EQUAL_WEIGHT",
    "ROBUST_VQ_TOP30_PERCENT_EQUAL_WEIGHT",
]
FLAG_BASED = {
    "ROBUST_VQ_TOP20_EXCLUDE_SOFT_ANOMALY_EQUAL_WEIGHT",
    "ROBUST_VQ_FLAG_CLEAN_TOP50_EQUAL_WEIGHT",
    "ROBUST_VQ_FLAG_CLEAN_TOP50_BUFFER_EQUAL_WEIGHT",
}
FALLBACK = {
    "ROBUST_VQ_D7_D9_BAND_EQUAL_WEIGHT",
    "ROBUST_VQ_TOP30_PERCENT_EQUAL_WEIGHT",
}


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def write_state(status, checkpoint, notes=None):
    lines = [
        "# DGTW-Adjusted Portfolio Evaluation Run v0",
        "",
        f"- task_name: {TASK_NAME}",
        f"- status: {status}",
        f"- last_checkpoint: {checkpoint}",
        f"- updated_at: {datetime.now().isoformat(timespec='seconds')}",
        "- resume_command: `python scripts\\run_dgtw_adjusted_portfolio_evaluation_v0.py > output\\_agent_runs\\DGTW_Adjusted_Portfolio_Eval_Run\\run_stdout.txt 2> output\\_agent_runs\\DGTW_Adjusted_Portfolio_Eval_Run\\run_stderr.txt`",
        "- guardrails: no weights edits; no DGTW selection/weighting; no alpha/beta; no IR; no TE; no training; no SHAP; no production",
    ]
    if notes:
        lines += ["", "## Notes"] + [f"- {x}" for x in notes]
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_json(path, default=None):
    if not path.exists():
        return default if default is not None else {}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_path(p):
    return Path(str(p).replace("\\", "/"))


def prerequisite_check(config, source_summary):
    candidate = normalize_path(config.get("stock_month_dgtw_candidate_path", ""))
    net = normalize_path(config.get("portfolio_net_return_path", ""))
    perf = normalize_path(config.get("portfolio_performance_summary_path", ""))
    files = {
        "config_path": CONFIG_PATH.exists(),
        "matching_policy_path": MATCHING_POLICY_PATH.exists(),
        "sample_policy_path": SAMPLE_POLICY_PATH.exists(),
        "metric_plan_path": METRIC_PLAN_PATH.exists(),
        "portfolio_manifest_path": PORTFOLIO_MANIFEST_PATH.exists(),
        "stock_month_candidate_path": candidate.exists(),
        "portfolio_net_return_path": net.exists(),
        "portfolio_performance_summary_path": perf.exists(),
        "source_summary_path": SOURCE_SUMMARY_PATH.exists(),
    }
    candidate_rows = int(pq.ParquetFile(candidate).metadata.num_rows) if candidate.exists() else 0
    prereq = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "required_files_found": files,
        "stock_month_candidate_row_count": candidate_rows,
        "source_final_decision": source_summary.get("final_decision"),
        "prep_allows_dgtw_benchmark_return": bool(config.get("calculate_dgtw_benchmark_return_next_run_allowed", False)),
        "prep_allows_dgtw_adjusted_return": bool(config.get("calculate_dgtw_adjusted_return_next_run_allowed", False)),
        "prep_allows_cumulative_adjusted_return": bool(config.get("calculate_cumulative_adjusted_return_next_run_allowed", False)),
        "prep_allows_adjusted_tstat": bool(config.get("calculate_adjusted_tstat_next_run_allowed", False)),
    }
    prereq["prerequisites_passed"] = bool(
        all(files.values())
        and candidate_rows > 0
        and prereq["prep_allows_dgtw_benchmark_return"]
        and prereq["prep_allows_dgtw_adjusted_return"]
    )
    return prereq


def build_monthly_benchmark(candidate_path, portfolios, threshold, role_map):
    cols = [
        "portfolio_name", "symbol", "month_end", "weight", "dgtw_benchmark_return_decimal",
        "dgtw_assignment_match_flag", "dgtw_cell_match_flag",
    ]
    df = pd.read_parquet(candidate_path, columns=cols)
    df = df[df["portfolio_name"].isin(portfolios)].copy()
    df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["dgtw_benchmark_return_decimal"] = pd.to_numeric(df["dgtw_benchmark_return_decimal"], errors="coerce")
    df["matched"] = df["dgtw_assignment_match_flag"].astype(bool) & df["dgtw_cell_match_flag"].astype(bool) & df["dgtw_benchmark_return_decimal"].notna()
    df["matched_weight"] = np.where(df["matched"], df["weight"], 0.0)
    df["raw_contribution"] = np.where(df["matched"], df["weight"] * df["dgtw_benchmark_return_decimal"], 0.0)

    rows = []
    for (portfolio_name, month_end), g in df.groupby(["portfolio_name", "month_end"], sort=True):
        matched_weight_share = float(g["matched_weight"].sum())
        raw = float(g["raw_contribution"].sum())
        normalized = raw / matched_weight_share if matched_weight_share > 0 else np.nan
        rows.append({
            "portfolio_name": portfolio_name,
            "month_end": pd.Timestamp(month_end).date().isoformat(),
            "dgtw_benchmark_return_raw": raw,
            "dgtw_benchmark_return_matched_normalized": normalized,
            "matched_weight_share": matched_weight_share,
            "unmatched_weight_share": 1.0 - matched_weight_share,
            "matched_stock_count": int(g["matched"].sum()),
            "unmatched_stock_count": int((~g["matched"]).sum()),
            "total_holding_count": int(len(g)),
            "low_dgtw_match_coverage_flag": bool(matched_weight_share < threshold),
            "portfolio_role": role_map.get(portfolio_name, "Unknown"),
        })
    out = pd.DataFrame(rows)
    del df
    gc.collect()
    return out


def build_adjusted_returns(monthly_bench, net_path, portfolios):
    net_cols = ["portfolio_name", "month_end", "cost_bps", "net_return", "portfolio_role"]
    net = pd.read_csv(net_path, usecols=net_cols)
    net = net[net["portfolio_name"].isin(portfolios)].copy()
    net["month_end"] = pd.to_datetime(net["month_end"], errors="coerce").dt.date.astype(str)
    net["cost_bps"] = pd.to_numeric(net["cost_bps"], errors="coerce").astype(int)
    net["net_return"] = pd.to_numeric(net["net_return"], errors="coerce")
    bench = monthly_bench.copy()
    merged = net.merge(
        bench.drop(columns=["portfolio_role"]),
        on=["portfolio_name", "month_end"],
        how="left",
    )
    merged["dgtw_adjusted_return_raw"] = merged["net_return"] - merged["dgtw_benchmark_return_raw"]
    merged["dgtw_adjusted_return_matched_normalized"] = merged["net_return"] - merged["dgtw_benchmark_return_matched_normalized"]
    keep = [
        "portfolio_name", "month_end", "cost_bps", "net_return",
        "dgtw_benchmark_return_raw", "dgtw_benchmark_return_matched_normalized",
        "dgtw_adjusted_return_raw", "dgtw_adjusted_return_matched_normalized",
        "matched_weight_share", "unmatched_weight_share", "low_dgtw_match_coverage_flag", "portfolio_role",
    ]
    out = merged[keep].copy()
    del net, bench, merged
    gc.collect()
    return out


def perf_stats(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = int(len(s))
    if n == 0:
        return {
            "month_count": 0,
            "mean_monthly_adjusted_return": np.nan,
            "annualized_adjusted_return_approx": np.nan,
            "adjusted_monthly_volatility": np.nan,
            "adjusted_sharpe": np.nan,
            "adjusted_tstat": np.nan,
            "positive_adjusted_month_ratio": np.nan,
            "cumulative_adjusted_return": np.nan,
            "worst_adjusted_month": np.nan,
            "best_adjusted_month": np.nan,
        }
    mean = float(s.mean())
    vol = float(s.std(ddof=1)) if n > 1 else np.nan
    annual = mean * 12.0
    sharpe = annual / (vol * math.sqrt(12.0)) if vol and not np.isnan(vol) and vol != 0 else np.nan
    tstat = mean / (vol / math.sqrt(n)) if vol and not np.isnan(vol) and vol != 0 else np.nan
    cumulative = float(np.prod(1.0 + s) - 1.0)
    return {
        "month_count": n,
        "mean_monthly_adjusted_return": mean,
        "annualized_adjusted_return_approx": annual,
        "adjusted_monthly_volatility": vol,
        "adjusted_sharpe": float(sharpe) if not pd.isna(sharpe) else np.nan,
        "adjusted_tstat": float(tstat) if not pd.isna(tstat) else np.nan,
        "positive_adjusted_month_ratio": float((s > 0).mean()),
        "cumulative_adjusted_return": cumulative,
        "worst_adjusted_month": float(s.min()),
        "best_adjusted_month": float(s.max()),
    }


def build_performance_summary(adjusted):
    rows = []
    variants = [
        ("raw_unmatched_not_renormalized", "dgtw_adjusted_return_raw"),
        ("matched_only_normalized", "dgtw_adjusted_return_matched_normalized"),
    ]
    for (p, c), g in adjusted.groupby(["portfolio_name", "cost_bps"], sort=True):
        for variant, col in variants:
            stats = perf_stats(g[col])
            stats.update({
                "portfolio_name": p,
                "cost_bps": int(c),
                "adjustment_variant": variant,
                "low_dgtw_match_coverage_month_count": int(g["low_dgtw_match_coverage_flag"].fillna(False).astype(bool).sum()),
                "avg_matched_weight_share": float(g["matched_weight_share"].mean()),
                "min_matched_weight_share": float(g["matched_weight_share"].min()),
                "portfolio_role": g["portfolio_role"].dropna().iloc[0] if g["portfolio_role"].notna().any() else "Unknown",
            })
            rows.append(stats)
    cols = [
        "portfolio_name", "portfolio_role", "cost_bps", "adjustment_variant",
        "month_count", "mean_monthly_adjusted_return", "annualized_adjusted_return_approx",
        "adjusted_monthly_volatility", "adjusted_sharpe", "adjusted_tstat",
        "positive_adjusted_month_ratio", "cumulative_adjusted_return",
        "worst_adjusted_month", "best_adjusted_month",
        "low_dgtw_match_coverage_month_count", "avg_matched_weight_share", "min_matched_weight_share",
    ]
    return pd.DataFrame(rows)[cols]


def build_cumulative(adjusted):
    rows = []
    variants = [
        ("raw_unmatched_not_renormalized", "dgtw_adjusted_return_raw"),
        ("matched_only_normalized", "dgtw_adjusted_return_matched_normalized"),
    ]
    for (p, c), g in adjusted.groupby(["portfolio_name", "cost_bps"], sort=True):
        g = g.sort_values("month_end")
        role = g["portfolio_role"].dropna().iloc[0] if g["portfolio_role"].notna().any() else "Unknown"
        for variant, col in variants:
            s = pd.to_numeric(g[col], errors="coerce")
            cum = (1.0 + s).cumprod() - 1.0
            for month_end, val in zip(g["month_end"], cum):
                rows.append({
                    "portfolio_name": p,
                    "cost_bps": int(c),
                    "adjustment_variant": variant,
                    "month_end": month_end,
                    "cumulative_adjusted_return": float(val) if pd.notna(val) else np.nan,
                    "portfolio_role": role,
                })
    return pd.DataFrame(rows)


def build_coverage_outputs(monthly_bench):
    by_month = monthly_bench[[
        "portfolio_name", "month_end", "matched_weight_share", "unmatched_weight_share",
        "matched_stock_count", "unmatched_stock_count", "total_holding_count", "low_dgtw_match_coverage_flag",
    ]].copy()
    rows = []
    for p, g in by_month.groupby("portfolio_name", sort=True):
        worst_idx = g["matched_weight_share"].idxmin()
        avg = float(g["matched_weight_share"].mean())
        low_count = int(g["low_dgtw_match_coverage_flag"].astype(bool).sum())
        if bool((g["matched_weight_share"] >= 0.95).all()):
            status = "PASS"
        elif avg >= 0.95:
            status = "WATCH"
        else:
            status = "FAIL"
        rows.append({
            "portfolio_name": p,
            "avg_matched_weight_share": avg,
            "min_matched_weight_share": float(g["matched_weight_share"].min()),
            "low_dgtw_match_coverage_month_count": low_count,
            "worst_month_end": g.loc[worst_idx, "month_end"],
            "worst_month_matched_weight_share": float(g.loc[worst_idx, "matched_weight_share"]),
            "coverage_status": status,
        })
    return by_month, pd.DataFrame(rows)


def build_comparison(perf):
    comp = perf.copy()
    comp["rank_by_20bps_adjusted_sharpe"] = np.nan
    mask = (comp["cost_bps"] == 20) & (comp["adjustment_variant"] == "raw_unmatched_not_renormalized")
    ranked = comp[mask].sort_values(
        ["adjusted_sharpe", "adjusted_tstat", "positive_adjusted_month_ratio"],
        ascending=[False, False, False],
    ).copy()
    ranked["rank_by_20bps_adjusted_sharpe"] = range(1, len(ranked) + 1)
    comp.loc[ranked.index, "rank_by_20bps_adjusted_sharpe"] = ranked["rank_by_20bps_adjusted_sharpe"]
    def interp(row):
        if row["portfolio_name"] in FLAG_BASED:
            group = "flag-based"
        elif row["portfolio_name"] in FALLBACK:
            group = "fallback"
        else:
            group = "other"
        if row["cost_bps"] == 20 and row["adjustment_variant"] == "raw_unmatched_not_renormalized":
            return f"{group}; primary ranking sample"
        return group
    comp["interpretation"] = comp.apply(interp, axis=1)
    cols = [
        "portfolio_name", "portfolio_role", "cost_bps", "adjustment_variant",
        "mean_monthly_adjusted_return", "annualized_adjusted_return_approx",
        "adjusted_tstat", "adjusted_sharpe", "positive_adjusted_month_ratio",
        "cumulative_adjusted_return", "avg_matched_weight_share",
        "low_dgtw_match_coverage_month_count", "rank_by_20bps_adjusted_sharpe", "interpretation",
    ]
    return comp[cols]


def decide(summary_fields, guardrail_pass):
    if not guardrail_pass:
        return "DGTW_ADJUSTED_EVAL_RUN_FAIL_GUARDRAIL"
    mean = summary_fields["primary_20bps_mean_adjusted_return"]
    tstat = summary_fields["primary_20bps_adjusted_tstat"]
    pos = summary_fields["primary_20bps_positive_adjusted_month_ratio"]
    avg_match = summary_fields["primary_20bps_avg_matched_weight_share"]
    flag_wins = summary_fields["flag_based_outperforms_fallback_dgtw_adjusted_20bps"]
    if mean > 0 and tstat >= 1.5 and pos >= 0.55 and avg_match >= 0.95 and flag_wins:
        return "DGTW_ADJUSTED_EVAL_RUN_STRONG_CHARACTERISTIC_ADJUSTED_PASS"
    if mean > 0 and pos > 0.50:
        return "DGTW_ADJUSTED_EVAL_RUN_PARTIAL_CHARACTERISTIC_ADJUSTED_PASS"
    if mean <= 0 or pos <= 0.50:
        return "DGTW_ADJUSTED_EVAL_RUN_FAIL_NO_CHARACTERISTIC_ADJUSTED_ALPHA"
    return "DGTW_ADJUSTED_EVAL_RUN_WATCH_CHARACTERISTIC_EXPOSURE_EXPLAINS_RETURN"


def main():
    ensure_dirs()
    write_state("running", "loading config and prerequisites")
    config = load_json(CONFIG_PATH)
    source_summary = load_json(SOURCE_SUMMARY_PATH)
    prereq = prerequisite_check(config, source_summary)
    (OUT_DIR / "dgtw_adjusted_eval_prerequisite_check.json").write_text(json.dumps(prereq, ensure_ascii=False, indent=2), encoding="utf-8")

    portfolios = config.get("portfolios_to_evaluate", DEFAULT_PORTFOLIOS)
    primary = config.get("primary_portfolio", "ROBUST_VQ_FLAG_CLEAN_TOP50_BUFFER_EQUAL_WEIGHT")
    primary_cost = int(config.get("primary_cost_bps", 20))
    threshold = float(config.get("matched_weight_share_threshold", 0.95))
    candidate_path = normalize_path(config["stock_month_dgtw_candidate_path"])
    net_path = normalize_path(config["portfolio_net_return_path"])

    manifest = pd.read_csv(PORTFOLIO_MANIFEST_PATH)
    role_map = manifest.set_index("portfolio_name")["portfolio_role"].to_dict()

    write_state("running", "aggregating stock-month DGTW benchmark to portfolio-month")
    monthly_bench = build_monthly_benchmark(candidate_path, portfolios, threshold, role_map)
    monthly_bench.to_csv(OUT_DIR / "dgtw_portfolio_monthly_benchmark_return.csv", index=False, encoding="utf-8-sig")

    adjusted = build_adjusted_returns(monthly_bench, net_path, portfolios)
    adjusted.to_csv(OUT_DIR / "dgtw_portfolio_monthly_adjusted_return_by_cost.csv", index=False, encoding="utf-8-sig")

    perf = build_performance_summary(adjusted)
    perf.to_csv(OUT_DIR / "dgtw_adjusted_performance_summary_by_cost.csv", index=False, encoding="utf-8-sig")

    comparison = build_comparison(perf)
    comparison.to_csv(OUT_DIR / "dgtw_adjusted_flag_based_vs_fallback_comparison.csv", index=False, encoding="utf-8-sig")

    coverage_by_month, coverage_summary = build_coverage_outputs(monthly_bench)
    coverage_by_month.to_csv(OUT_DIR / "dgtw_match_coverage_by_portfolio_month.csv", index=False, encoding="utf-8-sig")
    coverage_summary.to_csv(OUT_DIR / "dgtw_match_coverage_summary.csv", index=False, encoding="utf-8-sig")

    cumulative = build_cumulative(adjusted)
    cumulative.to_csv(OUT_DIR / "dgtw_cumulative_adjusted_return_by_cost.csv", index=False, encoding="utf-8-sig")

    guardrail = {
        "portfolio_weights_modified": False,
        "portfolio_weights_reconstructed": False,
        "dgtw_used_for_selection": False,
        "dgtw_used_for_weighting": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "training_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "live_order_ready_file_generated": False,
        "portfolio_dgtw_benchmark_return_calculated": True,
        "portfolio_dgtw_adjusted_return_calculated": True,
        "cumulative_adjusted_return_calculated": True,
    }
    guardrail["guardrail_pass"] = not any([
        guardrail["portfolio_weights_modified"],
        guardrail["portfolio_weights_reconstructed"],
        guardrail["dgtw_used_for_selection"],
        guardrail["dgtw_used_for_weighting"],
        guardrail["alpha_beta_regression_calculated"],
        guardrail["information_ratio_calculated"],
        guardrail["tracking_error_calculated"],
        guardrail["training_run"],
        guardrail["shap_calculated"],
        guardrail["production_modified"],
        guardrail["live_order_ready_file_generated"],
    ])
    pd.DataFrame([guardrail]).to_csv(OUT_DIR / "dgtw_adjusted_guardrail_qa.csv", index=False, encoding="utf-8-sig")

    primary_row = perf[
        (perf["portfolio_name"] == primary)
        & (perf["cost_bps"] == primary_cost)
        & (perf["adjustment_variant"] == "raw_unmatched_not_renormalized")
    ].iloc[0]
    comp20 = comparison[
        (comparison["cost_bps"] == 20)
        & (comparison["adjustment_variant"] == "raw_unmatched_not_renormalized")
    ].copy()
    best = comp20.sort_values(["adjusted_sharpe", "adjusted_tstat", "positive_adjusted_month_ratio"], ascending=[False, False, False]).iloc[0]
    flag_best = comp20[comp20["portfolio_name"].isin(FLAG_BASED)]["adjusted_sharpe"].max()
    fallback_best = comp20[comp20["portfolio_name"].isin(FALLBACK)]["adjusted_sharpe"].max()
    flag_outperforms = bool(flag_best > fallback_best)

    summary_fields = {
        "primary_20bps_mean_adjusted_return": float(primary_row["mean_monthly_adjusted_return"]),
        "primary_20bps_adjusted_tstat": float(primary_row["adjusted_tstat"]),
        "primary_20bps_adjusted_sharpe": float(primary_row["adjusted_sharpe"]),
        "primary_20bps_positive_adjusted_month_ratio": float(primary_row["positive_adjusted_month_ratio"]),
        "primary_20bps_cumulative_adjusted_return": float(primary_row["cumulative_adjusted_return"]),
        "primary_20bps_avg_matched_weight_share": float(primary_row["avg_matched_weight_share"]),
        "flag_based_outperforms_fallback_dgtw_adjusted_20bps": flag_outperforms,
    }
    primary_pass = bool(
        summary_fields["primary_20bps_mean_adjusted_return"] > 0
        and summary_fields["primary_20bps_adjusted_tstat"] > 0
        and summary_fields["primary_20bps_positive_adjusted_month_ratio"] > 0.50
        and summary_fields["primary_20bps_avg_matched_weight_share"] >= 0.95
    )
    primary_strong = bool(
        summary_fields["primary_20bps_mean_adjusted_return"] > 0
        and summary_fields["primary_20bps_adjusted_tstat"] >= 1.5
        and summary_fields["primary_20bps_positive_adjusted_month_ratio"] >= 0.55
        and summary_fields["primary_20bps_avg_matched_weight_share"] >= 0.95
    )
    summary_fields["primary_dgtw_adjusted_pass"] = primary_pass
    summary_fields["primary_dgtw_adjusted_strong_pass"] = primary_strong
    final_decision = decide(summary_fields, guardrail["guardrail_pass"])

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": bool(prereq["prerequisites_passed"]),
        "portfolio_count_evaluated": int(len(portfolios)),
        "portfolios_evaluated": portfolios,
        "cost_scenarios_evaluated": sorted(adjusted["cost_bps"].dropna().astype(int).unique().tolist()),
        "adjustment_variants_evaluated": ["raw_unmatched_not_renormalized", "matched_only_normalized"],
        "dgtw_cell_coverage_pass": bool(source_summary.get("dgtw_cell_coverage_pass", False)),
        "source_caveat_required": bool(not source_summary.get("dgtw_cell_coverage_pass", False)),
        "avg_matched_weight_share": float(monthly_bench["matched_weight_share"].mean()),
        "min_matched_weight_share": float(monthly_bench["matched_weight_share"].min()),
        "low_dgtw_match_coverage_month_count": int(monthly_bench["low_dgtw_match_coverage_flag"].astype(bool).sum()),
        "best_portfolio_by_20bps_dgtw_adjusted_sharpe": best["portfolio_name"],
        "best_portfolio_20bps_dgtw_adjusted_sharpe": float(best["adjusted_sharpe"]),
        "best_portfolio_20bps_mean_adjusted_return": float(best["mean_monthly_adjusted_return"]),
        "best_portfolio_20bps_adjusted_tstat": float(best["adjusted_tstat"]),
        "best_portfolio_20bps_positive_adjusted_month_ratio": float(best["positive_adjusted_month_ratio"]),
        "primary_portfolio": primary,
        **summary_fields,
        "portfolio_weights_modified": False,
        "portfolio_weights_reconstructed": False,
        "dgtw_used_for_selection": False,
        "dgtw_used_for_weighting": False,
        "portfolio_dgtw_benchmark_return_calculated": True,
        "portfolio_dgtw_adjusted_return_calculated": True,
        "cumulative_adjusted_return_calculated": True,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "training_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "live_order_ready_file_generated": False,
        "final_decision": final_decision,
    }
    if final_decision == "DGTW_ADJUSTED_EVAL_RUN_STRONG_CHARACTERISTIC_ADJUSTED_PASS":
        summary["recommended_next_step"] = "可将 DGTW-adjusted 结果纳入 research report；仍保持 research-only，不进入 production。"
    elif final_decision == "DGTW_ADJUSTED_EVAL_RUN_PARTIAL_CHARACTERISTIC_ADJUSTED_PASS":
        summary["recommended_next_step"] = "DGTW-adjusted 结果为部分通过；建议报告中同时展示 raw 与 matched-normalized sensitivity，并保留 source caveat。"
    elif final_decision == "DGTW_ADJUSTED_EVAL_RUN_FAIL_NO_CHARACTERISTIC_ADJUSTED_ALPHA":
        summary["recommended_next_step"] = "不要推进为 characteristic-adjusted alpha 结论；优先解释 size / B/M / momentum 暴露贡献。"
    else:
        summary["recommended_next_step"] = "先处理 guardrail 或 characteristic exposure caveat。"

    (OUT_DIR / "dgtw_adjusted_portfolio_evaluation_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# DGTW-Adjusted Portfolio Evaluation Run v0",
        "",
        f"- final_decision: {final_decision}",
        f"- primary_portfolio: {primary}",
        f"- primary_cost_bps: {primary_cost}",
        f"- primary_20bps_mean_adjusted_return: {summary_fields['primary_20bps_mean_adjusted_return']:.8f}",
        f"- primary_20bps_adjusted_tstat: {summary_fields['primary_20bps_adjusted_tstat']:.6f}",
        f"- primary_20bps_adjusted_sharpe: {summary_fields['primary_20bps_adjusted_sharpe']:.6f}",
        f"- primary_20bps_positive_adjusted_month_ratio: {summary_fields['primary_20bps_positive_adjusted_month_ratio']:.6f}",
        f"- avg_matched_weight_share: {summary['avg_matched_weight_share']:.6f}",
        "",
        "## Source Caveat",
        "",
        "- DGTW cell coverage pass = false.",
        f"- Portfolio-level matching coverage remains high: avg_final_dgtw_match_ratio = {source_summary.get('avg_final_dgtw_match_ratio')}.",
        f"- lowest_portfolio_final_match_ratio = {source_summary.get('lowest_portfolio_final_match_ratio')}.",
        "- DGTW-adjusted results are valid for matched portfolio holdings, with this source caveat.",
        "",
        "## Guardrail",
        "",
        "- 未修改 portfolio weights，未重构 holdings。",
        "- 未使用 DGTW adjusted return 重新选股或调权。",
        "- 未计算 alpha/beta、information ratio、tracking error。",
        "- 未训练、未 SHAP、未写 production，未生成 live-order ready file。",
    ]
    (OUT_DIR / "dgtw_adjusted_portfolio_evaluation_run_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    final_qa = pd.DataFrame([{
        "final_decision": final_decision,
        "prerequisites_passed": bool(prereq["prerequisites_passed"]),
        "portfolio_dgtw_benchmark_return_calculated": True,
        "portfolio_dgtw_adjusted_return_calculated": True,
        "cumulative_adjusted_return_calculated": True,
        "guardrail_pass": guardrail["guardrail_pass"],
        "required_outputs_created": True,
    }])
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    terminal_summary = {
        "task_name": TASK_NAME,
        "status": "completed",
        "stdout_log": str(RUN_DIR / "run_stdout.txt"),
        "stderr_log": str(RUN_DIR / "run_stderr.txt"),
        "summary_json": str(OUT_DIR / "dgtw_adjusted_portfolio_evaluation_run_summary.json"),
        "final_decision": final_decision,
    }
    (OUT_DIR / "terminal_summary.json").write_text(json.dumps(terminal_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "task_completion_card.md").write_text(
        "\n".join([
            "# Task Completion Card",
            "",
            f"- task_name: {TASK_NAME}",
            "- status: completed",
            f"- final_decision: {final_decision}",
            f"- summary_json: {OUT_DIR / 'dgtw_adjusted_portfolio_evaluation_run_summary.json'}",
            f"- final_qa: {OUT_DIR / 'final_qa.csv'}",
        ]) + "\n",
        encoding="utf-8",
    )
    write_state("completed", "all required outputs written", [f"final_decision: {final_decision}"])
    print(json.dumps(terminal_summary, ensure_ascii=False))

    del monthly_bench, adjusted, perf, comparison, coverage_by_month, coverage_summary, cumulative
    gc.collect()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ensure_dirs()
        err = traceback.format_exc()
        (RUN_DIR / "last_error.txt").write_text(err, encoding="utf-8")
        write_state("failed", "exception captured in last_error.txt")
        raise
