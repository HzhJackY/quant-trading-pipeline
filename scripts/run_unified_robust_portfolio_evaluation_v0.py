from __future__ import annotations

import gc
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "Unified Robust Portfolio Evaluation Run v0"
OUT_DIR = Path("output/unified_robust_portfolio_evaluation_run_v0")
RUN_DIR = Path("output/_agent_runs") / TASK_NAME
PREP_DIR = Path("output/unified_robust_portfolio_evaluation_prep_v0")
LOW_HOLDING_DIR = Path("output/flag_based_portfolio_low_holding_count_qa_review_v0")

FLAG_WEIGHTS = Path("output/flag_based_top50_buffer_portfolio_construction_run_v0/flag_based_top50_buffer_research_weights_v0.parquet")
FALLBACK_WEIGHTS = Path("output/robust_formation_portfolio_construction_run_v0/robust_formation_research_weights_v0.parquet")

PORTFOLIOS = [
    "ROBUST_VQ_TOP20_EXCLUDE_SOFT_ANOMALY_EQUAL_WEIGHT",
    "ROBUST_VQ_FLAG_CLEAN_TOP50_EQUAL_WEIGHT",
    "ROBUST_VQ_FLAG_CLEAN_TOP50_BUFFER_EQUAL_WEIGHT",
    "ROBUST_VQ_D7_D9_BAND_EQUAL_WEIGHT",
    "ROBUST_VQ_TOP30_PERCENT_EQUAL_WEIGHT",
]
FLAG_PORTFOLIOS = set(PORTFOLIOS[:3])
FALLBACK_PORTFOLIOS = set(PORTFOLIOS[3:])
TOP20 = "ROBUST_VQ_TOP20_EXCLUDE_SOFT_ANOMALY_EQUAL_WEIGHT"
BUFFER = "ROBUST_VQ_FLAG_CLEAN_TOP50_BUFFER_EQUAL_WEIGHT"
NON_BUFFER_TOP50 = "ROBUST_VQ_FLAG_CLEAN_TOP50_EQUAL_WEIGHT"
COST_BPS = [0, 10, 20, 30]
WEIGHT_SUM_TOL = 1e-10
INDUSTRY_CONCENTRATION_THRESHOLD = 0.20

REQUIRED_INPUTS = [
    PREP_DIR / "unified_robust_portfolio_evaluation_prep_summary.json",
    PREP_DIR / "unified_portfolio_weight_source_manifest.csv",
    PREP_DIR / "unified_portfolio_taxonomy.csv",
    PREP_DIR / "unified_portfolio_evaluation_metric_plan.csv",
    PREP_DIR / "unified_portfolio_sample_policy.json",
    PREP_DIR / "unified_portfolio_sensitivity_policy.json",
    PREP_DIR / "unified_portfolio_cost_scenario_policy.csv",
    PREP_DIR / "unified_portfolio_benchmark_policy.json",
    PREP_DIR / "unified_portfolio_turnover_policy.json",
    PREP_DIR / "unified_portfolio_industry_exposure_policy.json",
    PREP_DIR / "unified_portfolio_evaluation_run_config_draft.json",
    PREP_DIR / "unified_portfolio_guardrail_checklist.csv",
    FLAG_WEIGHTS,
    FALLBACK_WEIGHTS,
    LOW_HOLDING_DIR / "flag_based_portfolio_low_holding_count_qa_review_summary.json",
    LOW_HOLDING_DIR / "low_holding_portfolio_month_detail.csv",
    LOW_HOLDING_DIR / "low_holding_evaluation_policy_recommendation.csv",
]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs(run_timestamp: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        RUN_DIR / "RUN_STATE.md",
        {
            "task_name": TASK_NAME,
            "status": "running",
            "run_timestamp": run_timestamp,
            "mode": "low-resource checkpoint-first resume-safe",
            "note": "research evaluation only; no benchmark-relative, alpha/beta, training, SHAP, or production writes",
        },
    )


def load_weights(path: Path, wanted: set[str]) -> pd.DataFrame:
    cols = [
        "portfolio_name",
        "portfolio_role",
        "symbol",
        "month_end",
        "weight",
        "selected_count_for_month",
        "low_holding_count_flag",
        "primary_industry_code",
        "primary_industry_name",
        "fwd_ret_1m",
    ]
    df = pd.read_parquet(path, columns=cols)
    df["symbol"] = df["symbol"].astype("string")
    df["month_end"] = pd.to_datetime(df["month_end"]).dt.strftime("%Y-%m-%d")
    df = df[df["portfolio_name"].isin(wanted)].copy()
    return df


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna().astype(float)
    if r.empty:
        return np.nan
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def perf_metrics(df: pd.DataFrame, sample_variant: str | None = None) -> dict:
    returns = df["net_return"].dropna().astype(float)
    turnover = df["one_way_turnover"].dropna().astype(float)
    vol = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    annual_vol = vol * math.sqrt(12) if pd.notna(vol) else np.nan
    mean_ret = float(returns.mean()) if len(returns) else np.nan
    ann_ret = mean_ret * 12 if pd.notna(mean_ret) else np.nan
    sharpe = ann_ret / annual_vol if pd.notna(annual_vol) and annual_vol != 0 else np.nan
    out = {
        "month_count": int(len(returns)),
        "mean_monthly_net_return": mean_ret,
        "monthly_volatility": vol,
        "annualized_return_approx": ann_ret,
        "annualized_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(returns),
        "positive_month_ratio": float((returns > 0).mean()) if len(returns) else np.nan,
        "worst_month_return": float(returns.min()) if len(returns) else np.nan,
        "best_month_return": float(returns.max()) if len(returns) else np.nan,
        "avg_turnover": float(turnover.mean()) if len(turnover) else np.nan,
        "median_turnover": float(turnover.median()) if len(turnover) else np.nan,
        "avg_holding_count": float(df["holding_count"].mean()) if "holding_count" in df else np.nan,
        "min_holding_count": int(df["holding_count"].min()) if "holding_count" in df and len(df) else 0,
        "max_holding_count": int(df["holding_count"].max()) if "holding_count" in df and len(df) else 0,
    }
    if sample_variant is not None:
        out["sample_variant"] = sample_variant
    return out


def validate_weights(weights: pd.DataFrame) -> dict:
    weight_sum = weights.groupby(["portfolio_name", "month_end"], as_index=False)["weight"].sum()
    weight_sum["weight_sum_abs_error"] = (weight_sum["weight"] - 1.0).abs()
    max_err = float(weight_sum["weight_sum_abs_error"].max()) if len(weight_sum) else np.nan
    duplicates = weights.duplicated(["portfolio_name", "month_end", "symbol"]).sum()
    missing_target = int(weights["fwd_ret_1m"].isna().sum())
    null_required = {
        col: int(weights[col].isna().sum())
        for col in ["portfolio_name", "symbol", "month_end", "weight", "fwd_ret_1m"]
    }
    non_positive_weight = int((weights["weight"] <= 0).sum())
    return {
        "max_weight_sum_abs_error": max_err,
        "weight_sum_pass": bool(pd.notna(max_err) and max_err <= WEIGHT_SUM_TOL),
        "duplicate_symbol_within_portfolio_month_count": int(duplicates),
        "missing_target_count": missing_target,
        "null_required_counts": null_required,
        "non_positive_weight_count": non_positive_weight,
        "row_requirements_pass": bool(duplicates == 0 and missing_target == 0 and non_positive_weight == 0 and all(v == 0 for v in null_required.values())),
    }


def monthly_industry_exposure(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exposure = (
        weights.groupby(
            ["portfolio_name", "month_end", "primary_industry_code", "primary_industry_name", "portfolio_role"],
            dropna=False,
            as_index=False,
        )
        .agg(industry_weight=("weight", "sum"), industry_holding_count=("symbol", "nunique"))
    )
    dominant = (
        exposure.groupby(["portfolio_name", "month_end"], as_index=False)
        .agg(dominant_industry_share=("industry_weight", "max"), industry_count=("primary_industry_code", "nunique"))
    )
    summary = (
        dominant.groupby("portfolio_name", as_index=False)
        .agg(
            avg_industry_count=("industry_count", "mean"),
            avg_dominant_industry_share=("dominant_industry_share", "mean"),
            max_dominant_industry_share=("dominant_industry_share", "max"),
            high_concentration_month_count=("dominant_industry_share", lambda s: int((s > INDUSTRY_CONCENTRATION_THRESHOLD).sum())),
        )
    )
    summary["dominant_industry_share_threshold"] = INDUSTRY_CONCENTRATION_THRESHOLD
    return exposure, summary


def monthly_gross_return(weights: pd.DataFrame, dominant: pd.DataFrame) -> pd.DataFrame:
    gross = (
        weights.assign(weighted_return=weights["weight"] * weights["fwd_ret_1m"])
        .groupby(["portfolio_name", "month_end", "portfolio_role"], as_index=False)
        .agg(
            gross_return=("weighted_return", "sum"),
            holding_count=("symbol", "nunique"),
            weight_sum=("weight", "sum"),
            low_holding_count_flag=("low_holding_count_flag", "max"),
        )
    )
    gross["low_holding_count_flag"] = gross["low_holding_count_flag"].fillna(False).astype(bool)
    gross = gross.merge(dominant[["portfolio_name", "month_end", "dominant_industry_share"]], on=["portfolio_name", "month_end"], how="left")
    return gross.sort_values(["portfolio_name", "month_end"]).reset_index(drop=True)


def monthly_turnover(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, g in weights.sort_values(["portfolio_name", "month_end", "symbol"]).groupby("portfolio_name"):
        role = g["portfolio_role"].dropna().iloc[0] if g["portfolio_role"].notna().any() else ""
        by_month = {m: sub.set_index("symbol")["weight"].astype(float) for m, sub in g.groupby("month_end")}
        months = sorted(by_month)
        prev_symbols: set[str] | None = None
        prev_w: pd.Series | None = None
        for m in months:
            cur_w = by_month[m]
            cur_symbols = set(cur_w.index)
            if prev_w is None:
                turnover = np.nan
                entry_count = np.nan
                exit_count = np.nan
                retained_count = np.nan
            else:
                symbols = sorted(cur_symbols | set(prev_w.index))
                diff = cur_w.reindex(symbols, fill_value=0.0) - prev_w.reindex(symbols, fill_value=0.0)
                turnover = float(0.5 * diff.abs().sum())
                entry_count = int(len(cur_symbols - (prev_symbols or set())))
                exit_count = int(len((prev_symbols or set()) - cur_symbols))
                retained_count = int(len(cur_symbols & (prev_symbols or set())))
            rows.append(
                {
                    "portfolio_name": portfolio,
                    "month_end": m,
                    "one_way_turnover": turnover,
                    "holding_count": int(len(cur_symbols)),
                    "entry_count": entry_count,
                    "exit_count": exit_count,
                    "retained_count": retained_count,
                    "portfolio_role": role,
                }
            )
            prev_w = cur_w
            prev_symbols = cur_symbols
    return pd.DataFrame(rows).sort_values(["portfolio_name", "month_end"]).reset_index(drop=True)


def build_net_returns(gross: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    base = gross.merge(
        turnover[["portfolio_name", "month_end", "one_way_turnover"]],
        on=["portfolio_name", "month_end"],
        how="left",
    )
    rows = []
    for bps in COST_BPS:
        tmp = base.copy()
        tmp["cost_bps"] = bps
        tmp["first_month_turnover_missing"] = tmp["one_way_turnover"].isna()
        tmp["transaction_cost_drag"] = tmp["one_way_turnover"].fillna(0.0) * bps / 10000.0
        tmp["net_return"] = tmp["gross_return"] - tmp["transaction_cost_drag"]
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)[
        [
            "portfolio_name",
            "month_end",
            "cost_bps",
            "gross_return",
            "one_way_turnover",
            "transaction_cost_drag",
            "net_return",
            "first_month_turnover_missing",
            "low_holding_count_flag",
            "portfolio_role",
            "holding_count",
            "dominant_industry_share",
        ]
    ]


def cumulative_returns(net: pd.DataFrame) -> pd.DataFrame:
    out = net.sort_values(["portfolio_name", "cost_bps", "month_end"]).copy()
    out["cumulative_net_return"] = out.groupby(["portfolio_name", "cost_bps"])["net_return"].transform(lambda s: (1.0 + s).cumprod() - 1.0)
    return out[["portfolio_name", "cost_bps", "month_end", "cumulative_net_return", "portfolio_role"]]


def performance_summary(net: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (portfolio, bps), g in net.groupby(["portfolio_name", "cost_bps"]):
        role = g["portfolio_role"].dropna().iloc[0] if g["portfolio_role"].notna().any() else ""
        row = {"portfolio_name": portfolio, "portfolio_role": role, "cost_bps": int(bps)}
        row.update(perf_metrics(g))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["cost_bps", "portfolio_name"]).reset_index(drop=True)


def low_holding_sensitivity(net: pd.DataFrame) -> pd.DataFrame:
    rows = []
    affected = net[net["portfolio_name"] == TOP20]
    for bps, g in affected.groupby("cost_bps"):
        for variant, sub in [
            ("base", g),
            ("exclude_low_holding", g[~g["low_holding_count_flag"].astype(bool)]),
        ]:
            row = {"portfolio_name": TOP20, "cost_bps": int(bps)}
            row.update(perf_metrics(sub, sample_variant=variant))
            rows.append(row)
    result = pd.DataFrame(rows)
    base20 = result[(result["cost_bps"] == 20) & (result["sample_variant"] == "base")]
    ex20 = result[(result["cost_bps"] == 20) & (result["sample_variant"] == "exclude_low_holding")]
    passed = False
    if len(base20) and len(ex20):
        b = float(base20["sharpe"].iloc[0])
        e = float(ex20["sharpe"].iloc[0])
        passed = bool(pd.notna(b) and pd.notna(e) and np.sign(b) == np.sign(e) and abs(e - b) <= max(0.25, abs(b) * 0.25))
    interpretation = "low-holding sensitivity 不改变 20bps Sharpe 方向且幅度可控" if passed else "low-holding sensitivity 对 20bps 结果有可见影响，需在研究结论中保留 watch"
    result["interpretation"] = interpretation
    return result[
        [
            "portfolio_name",
            "cost_bps",
            "sample_variant",
            "month_count",
            "mean_monthly_net_return",
            "sharpe",
            "max_drawdown",
            "positive_month_ratio",
            "avg_turnover",
            "interpretation",
        ]
    ]


def comparison(summary: pd.DataFrame, industry_summary: pd.DataFrame) -> pd.DataFrame:
    comp = summary.merge(industry_summary[["portfolio_name", "avg_dominant_industry_share"]], on="portfolio_name", how="left")
    base20 = comp[comp["cost_bps"] == 20].copy()
    base20["sort_maxdd"] = base20["max_drawdown"].abs()
    base20["sort_holding_reasonable"] = (base20["avg_holding_count"] - 50.0).abs()
    base20 = base20.sort_values(["sharpe", "sort_maxdd", "avg_turnover", "sort_holding_reasonable"], ascending=[False, True, True, True])
    rank_map = {p: i + 1 for i, p in enumerate(base20["portfolio_name"])}
    comp["overall_rank_by_risk_adjusted_performance"] = comp["portfolio_name"].map(rank_map)
    comp["interpretation"] = comp.apply(
        lambda r: "flag-based research portfolio" if r["portfolio_name"] in FLAG_PORTFOLIOS else "fallback diagnostic portfolio",
        axis=1,
    )
    return comp[
        [
            "portfolio_name",
            "portfolio_role",
            "cost_bps",
            "mean_monthly_net_return",
            "annualized_return_approx",
            "sharpe",
            "max_drawdown",
            "positive_month_ratio",
            "avg_turnover",
            "avg_holding_count",
            "avg_dominant_industry_share",
            "overall_rank_by_risk_adjusted_performance",
            "interpretation",
        ]
    ].sort_values(["cost_bps", "overall_rank_by_risk_adjusted_performance"])


def final_decision(summary: pd.DataFrame, low_sens: pd.DataFrame, guardrail_violation: bool) -> tuple[str, dict]:
    if guardrail_violation:
        return "UNIFIED_PORTFOLIO_EVAL_RUN_FAIL_GUARDRAIL", {}
    s20 = summary[summary["cost_bps"] == 20].copy()
    best = s20.sort_values(["sharpe", "max_drawdown", "avg_turnover"], ascending=[False, False, True]).iloc[0]
    flag_best = s20[s20["portfolio_name"].isin(FLAG_PORTFOLIOS)]["sharpe"].max()
    fallback_best = s20[s20["portfolio_name"].isin(FALLBACK_PORTFOLIOS)]["sharpe"].max()
    flag_out = bool(pd.notna(flag_best) and pd.notna(fallback_best) and flag_best > fallback_best)
    flag_strong = bool(pd.notna(flag_best) and pd.notna(fallback_best) and flag_best >= fallback_best + 0.10)
    top20_base = low_sens[(low_sens["cost_bps"] == 20) & (low_sens["sample_variant"] == "base")]
    top20_ex = low_sens[(low_sens["cost_bps"] == 20) & (low_sens["sample_variant"] == "exclude_low_holding")]
    sens_pass = False
    if len(top20_base) and len(top20_ex):
        b = float(top20_base["sharpe"].iloc[0])
        e = float(top20_ex["sharpe"].iloc[0])
        sens_pass = bool(pd.notna(b) and pd.notna(e) and np.sign(b) == np.sign(e) and abs(e - b) <= max(0.25, abs(b) * 0.25))
    buffer_turn = s20.loc[s20["portfolio_name"] == BUFFER, "avg_turnover"]
    non_buffer_turn = s20.loc[s20["portfolio_name"] == NON_BUFFER_TOP50, "avg_turnover"]
    buffer_reduces = bool(len(buffer_turn) and len(non_buffer_turn) and float(buffer_turn.iloc[0]) < float(non_buffer_turn.iloc[0]))
    maxdd_ok = bool(pd.notna(best["max_drawdown"]) and best["max_drawdown"] > -0.80)
    turnover_ok = bool(pd.notna(best["avg_turnover"]) and best["avg_turnover"] < 1.00)
    if flag_strong and maxdd_ok and turnover_ok and sens_pass:
        decision = "UNIFIED_PORTFOLIO_EVAL_RUN_STRONG_RESEARCH_PASS"
    elif flag_out and maxdd_ok:
        decision = "UNIFIED_PORTFOLIO_EVAL_RUN_PARTIAL_RESEARCH_PASS"
    elif flag_out or abs(float(flag_best) - float(fallback_best)) <= 0.10:
        decision = "UNIFIED_PORTFOLIO_EVAL_RUN_WATCH_MIXED_RESULTS"
    else:
        decision = "UNIFIED_PORTFOLIO_EVAL_RUN_FAIL_NO_ROBUST_PORTFOLIO"
    return decision, {
        "best": best,
        "top20_low_holding_sensitivity_passed": sens_pass,
        "flag_based_outperforms_fallback_at_20bps": flag_out,
        "buffer_reduces_turnover_vs_non_buffer": buffer_reduces,
    }


def main() -> None:
    run_timestamp = datetime.now().isoformat(timespec="seconds")
    ensure_dirs(run_timestamp)

    missing = [str(p) for p in REQUIRED_INPUTS if not p.exists()]
    prep_summary = read_json(PREP_DIR / "unified_robust_portfolio_evaluation_prep_summary.json") if not missing else {}
    benchmark_policy = read_json(PREP_DIR / "unified_portfolio_benchmark_policy.json") if not missing else {}
    taxonomy = pd.read_csv(PREP_DIR / "unified_portfolio_taxonomy.csv") if not missing else pd.DataFrame()

    benchmark_source_available = bool(benchmark_policy.get("benchmark_source_available", False))
    benchmark_blocked = not benchmark_source_available

    flag_weights = load_weights(FLAG_WEIGHTS, FLAG_PORTFOLIOS) if not missing else pd.DataFrame()
    fallback_weights = load_weights(FALLBACK_WEIGHTS, FALLBACK_PORTFOLIOS) if not missing else pd.DataFrame()
    weights = pd.concat([flag_weights, fallback_weights], ignore_index=True)
    registered_set = set(PORTFOLIOS)
    unregistered_portfolio_detected = bool(len(set(weights["portfolio_name"].unique()) - registered_set)) if len(weights) else False
    weights = weights[weights["portfolio_name"].isin(PORTFOLIOS)].copy()
    taxonomy_roles = taxonomy.set_index("portfolio_name")["portfolio_role"].to_dict() if not taxonomy.empty else {}
    weights["portfolio_role"] = weights["portfolio_name"].map(taxonomy_roles).fillna(weights["portfolio_role"])

    validation = validate_weights(weights) if len(weights) else {
        "weight_sum_pass": False,
        "row_requirements_pass": False,
        "max_weight_sum_abs_error": np.nan,
        "duplicate_symbol_within_portfolio_month_count": 0,
        "missing_target_count": 0,
        "null_required_counts": {},
        "non_positive_weight_count": 0,
    }
    portfolios_present = sorted(weights["portfolio_name"].unique().tolist()) if len(weights) else []
    prerequisites_passed = bool(
        not missing
        and prep_summary.get("prerequisites_passed") is True
        and set(portfolios_present) == registered_set
        and validation["weight_sum_pass"]
        and validation["row_requirements_pass"]
        and not benchmark_source_available
    )

    prereq = {
        "run_timestamp": run_timestamp,
        "required_inputs_checked": [str(p) for p in REQUIRED_INPUTS],
        "missing_inputs": missing,
        "prep_prerequisites_passed": prep_summary.get("prerequisites_passed"),
        "portfolios_present": portfolios_present,
        "registered_portfolios": PORTFOLIOS,
        "unregistered_portfolio_detected": unregistered_portfolio_detected,
        "benchmark_source_available": benchmark_source_available,
        "benchmark_relative_eval_blocked_by_missing_benchmark_source": benchmark_blocked,
        **validation,
        "prerequisites_passed": prerequisites_passed,
    }
    write_json(OUT_DIR / "unified_portfolio_eval_prerequisite_check.json", prereq)

    guardrails = {
        "weights_modified": False,
        "weights_reconstructed": False,
        "fwd_ret_used_for_selection": False,
        "fwd_ret_used_for_weighting": False,
        "unregistered_portfolio_detected": unregistered_portfolio_detected,
        "benchmark_relative_return_calculated": False,
        "alpha_beta_regression_calculated": False,
        "training_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "live_order_ready_file_generated": False,
    }
    guardrail_violation = bool(
        guardrails["weights_modified"]
        or guardrails["weights_reconstructed"]
        or guardrails["fwd_ret_used_for_selection"]
        or guardrails["fwd_ret_used_for_weighting"]
        or guardrails["benchmark_relative_return_calculated"]
        or guardrails["alpha_beta_regression_calculated"]
        or guardrails["training_run"]
        or guardrails["shap_calculated"]
        or guardrails["production_modified"]
        or guardrails["live_order_ready_file_generated"]
        or not prerequisites_passed
    )

    if not prerequisites_passed:
        decision = "UNIFIED_PORTFOLIO_EVAL_RUN_FAIL_GUARDRAIL"
        # Write empty structured outputs so downstream checks can fail visibly without stale files.
        for name in [
            "unified_portfolio_monthly_gross_return.csv",
            "unified_portfolio_monthly_turnover.csv",
            "unified_portfolio_monthly_net_return_by_cost.csv",
            "unified_portfolio_cumulative_return_by_cost.csv",
            "unified_portfolio_performance_summary_by_cost.csv",
            "unified_portfolio_low_holding_sensitivity.csv",
            "unified_portfolio_monthly_industry_exposure.csv",
            "unified_portfolio_industry_exposure_summary.csv",
            "unified_portfolio_flag_based_vs_fallback_comparison.csv",
        ]:
            pd.DataFrame().to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")
        best_info = {}
        month_count = 0
    else:
        industry_exposure, industry_summary = monthly_industry_exposure(weights)
        dominant = industry_exposure.groupby(["portfolio_name", "month_end"], as_index=False).agg(dominant_industry_share=("industry_weight", "max"))
        gross = monthly_gross_return(weights, dominant)
        turnover = monthly_turnover(weights)
        net = build_net_returns(gross, turnover)
        cumulative = cumulative_returns(net)
        perf = performance_summary(net)
        low_sens = low_holding_sensitivity(net)
        comp = comparison(perf, industry_summary)

        industry_exposure.to_csv(OUT_DIR / "unified_portfolio_monthly_industry_exposure.csv", index=False, encoding="utf-8-sig")
        industry_summary.to_csv(OUT_DIR / "unified_portfolio_industry_exposure_summary.csv", index=False, encoding="utf-8-sig")
        gross.to_csv(OUT_DIR / "unified_portfolio_monthly_gross_return.csv", index=False, encoding="utf-8-sig")
        turnover.to_csv(OUT_DIR / "unified_portfolio_monthly_turnover.csv", index=False, encoding="utf-8-sig")
        net.drop(columns=["holding_count", "dominant_industry_share"]).to_csv(OUT_DIR / "unified_portfolio_monthly_net_return_by_cost.csv", index=False, encoding="utf-8-sig")
        cumulative.to_csv(OUT_DIR / "unified_portfolio_cumulative_return_by_cost.csv", index=False, encoding="utf-8-sig")
        perf.to_csv(OUT_DIR / "unified_portfolio_performance_summary_by_cost.csv", index=False, encoding="utf-8-sig")
        low_sens.to_csv(OUT_DIR / "unified_portfolio_low_holding_sensitivity.csv", index=False, encoding="utf-8-sig")
        comp.to_csv(OUT_DIR / "unified_portfolio_flag_based_vs_fallback_comparison.csv", index=False, encoding="utf-8-sig")

        decision, best_info = final_decision(perf, low_sens, guardrail_violation)
        month_count = int(gross["month_end"].nunique())

        del industry_exposure, industry_summary, dominant, gross, turnover, net, cumulative, perf, low_sens, comp
        gc.collect()

    guardrail_rows = [{"guardrail_item": k, "value": v, "pass": not bool(v) if k != "unregistered_portfolio_detected" else not bool(v), "notes": "research evaluation guardrail"} for k, v in guardrails.items()]
    pd.DataFrame(guardrail_rows).to_csv(OUT_DIR / "unified_portfolio_guardrail_qa.csv", index=False, encoding="utf-8-sig")

    best = best_info.get("best")
    summary_payload = {
        "run_timestamp": run_timestamp,
        "prerequisites_passed": prerequisites_passed,
        "portfolio_count_evaluated": len(portfolios_present) if prerequisites_passed else 0,
        "portfolios_evaluated": portfolios_present if prerequisites_passed else [],
        "month_count": month_count,
        "cost_scenarios_evaluated": COST_BPS if prerequisites_passed else [],
        "benchmark_source_available": benchmark_source_available,
        "benchmark_relative_eval_blocked_by_missing_benchmark_source": benchmark_blocked,
        "benchmark_relative_return_calculated": False,
        "alpha_beta_regression_calculated": False,
        "best_portfolio_by_20bps_sharpe": None if best is None else best["portfolio_name"],
        "best_portfolio_20bps_sharpe": None if best is None else float(best["sharpe"]),
        "best_portfolio_20bps_maxdd": None if best is None else float(best["max_drawdown"]),
        "best_portfolio_20bps_avg_turnover": None if best is None else float(best["avg_turnover"]),
        "best_portfolio_20bps_avg_holding_count": None if best is None else float(best["avg_holding_count"]),
        "top20_low_holding_sensitivity_passed": bool(best_info.get("top20_low_holding_sensitivity_passed", False)),
        "flag_based_outperforms_fallback_at_20bps": bool(best_info.get("flag_based_outperforms_fallback_at_20bps", False)),
        "buffer_reduces_turnover_vs_non_buffer": bool(best_info.get("buffer_reduces_turnover_vs_non_buffer", False)),
        **guardrails,
        "portfolio_return_calculated": prerequisites_passed,
        "cumulative_return_calculated": prerequisites_passed,
        "turnover_calculated": prerequisites_passed,
        "transaction_cost_calculated": prerequisites_passed,
        "sharpe_calculated": prerequisites_passed,
        "maxdd_calculated": prerequisites_passed,
        "final_decision": decision,
        "recommended_next_step": "在不引入 production 语境的前提下复核 20bps 排名、low-holding sensitivity 与 turnover/cost tradeoff；如需 benchmark-relative/alpha-beta，先补充 benchmark source。",
    }
    write_json(OUT_DIR / "unified_robust_portfolio_evaluation_run_summary.json", summary_payload)

    report = f"""# Unified Robust Portfolio Evaluation Run v0

## 结论
- final_decision: {decision}
- prerequisites_passed: {prerequisites_passed}
- evaluated portfolios: {len(summary_payload['portfolios_evaluated'])}
- benchmark_source_available: {benchmark_source_available}
- benchmark-relative / alpha-beta: blocked by missing benchmark source

## 关键研究结果
- best_portfolio_by_20bps_sharpe: {summary_payload['best_portfolio_by_20bps_sharpe']}
- best_portfolio_20bps_sharpe: {summary_payload['best_portfolio_20bps_sharpe']}
- best_portfolio_20bps_maxdd: {summary_payload['best_portfolio_20bps_maxdd']}
- best_portfolio_20bps_avg_turnover: {summary_payload['best_portfolio_20bps_avg_turnover']}
- best_portfolio_20bps_avg_holding_count: {summary_payload['best_portfolio_20bps_avg_holding_count']}
- top20_low_holding_sensitivity_passed: {summary_payload['top20_low_holding_sensitivity_passed']}
- flag_based_outperforms_fallback_at_20bps: {summary_payload['flag_based_outperforms_fallback_at_20bps']}
- buffer_reduces_turnover_vs_non_buffer: {summary_payload['buffer_reduces_turnover_vs_non_buffer']}

## Guardrail
未修改 weights，未重构 weights，未使用 fwd_ret_1m 选股或调权，未计算 benchmark-relative return，未计算 alpha/beta，未训练，未 SHAP，未写 production，未生成 live-order-ready 文件。
"""
    (OUT_DIR / "unified_robust_portfolio_evaluation_run_report.md").write_text(report, encoding="utf-8")

    final_qa = pd.DataFrame(
        [
            {"qa_item": "prerequisites_passed", "pass": prerequisites_passed, "detail": str(prerequisites_passed)},
            {"qa_item": "five_portfolios_evaluated", "pass": summary_payload["portfolio_count_evaluated"] == 5, "detail": str(summary_payload["portfolios_evaluated"])},
            {"qa_item": "benchmark_metrics_blocked", "pass": benchmark_blocked and not summary_payload["benchmark_relative_return_calculated"] and not summary_payload["alpha_beta_regression_calculated"], "detail": "benchmark source unavailable"},
            {"qa_item": "no_guardrail_violation", "pass": decision != "UNIFIED_PORTFOLIO_EVAL_RUN_FAIL_GUARDRAIL", "detail": decision},
            {"qa_item": "summary_written", "pass": (OUT_DIR / "unified_robust_portfolio_evaluation_run_summary.json").exists(), "detail": str(OUT_DIR)},
        ]
    )
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    final_qa.to_csv(RUN_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")

    terminal_summary = {
        "task_name": TASK_NAME,
        "run_timestamp": run_timestamp,
        "script": "scripts/run_unified_robust_portfolio_evaluation_v0.py",
        "stdout_log": str(RUN_DIR / "run_stdout.txt"),
        "stderr_log": str(RUN_DIR / "run_stderr.txt"),
        "output_directory": str(OUT_DIR),
        "final_decision": decision,
        "benchmark_relative_return_calculated": False,
        "alpha_beta_regression_calculated": False,
        "production_modified": False,
    }
    write_json(RUN_DIR / "terminal_summary.json", terminal_summary)
    completion_card = f"""# Task Completion Card

- task_name: {TASK_NAME}
- run_timestamp: {run_timestamp}
- final_decision: {decision}
- output_directory: {OUT_DIR}
- logs: {RUN_DIR / 'run_stdout.txt'} / {RUN_DIR / 'run_stderr.txt'}
- benchmark_relative_return_calculated: false
- alpha_beta_regression_calculated: false
- production_modified: false
"""
    (RUN_DIR / "task_completion_card.md").write_text(completion_card, encoding="utf-8")
    write_json(
        RUN_DIR / "RUN_STATE.md",
        {
            "task_name": TASK_NAME,
            "status": "completed",
            "run_timestamp": run_timestamp,
            "output_directory": str(OUT_DIR),
            "final_decision": decision,
            "resume_instruction": "如需复核，读取本 RUN_STATE.md、summary json、final_qa.csv 与 stderr/stdout 日志。",
        },
    )

    del flag_weights, fallback_weights, weights, final_qa
    gc.collect()
    print(json.dumps({"final_decision": decision, "prerequisites_passed": prerequisites_passed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
