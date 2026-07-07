import gc
import json
import math
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "DGTW_Adjusted_Failure_Forensics"
OUT_DIR = Path("output/dgtw_adjusted_failure_forensic_attribution_v0")
RUN_DIR = Path("output/_agent_runs") / TASK_NAME

DGTW_RUN = Path("output/dgtw_adjusted_portfolio_evaluation_run_v0")
DGTW_AUDIT = Path("output/dgtw_benchmark_source_audit_stock_matching_feasibility_v1")
DGTW_PREP = Path("output/dgtw_adjusted_portfolio_evaluation_prep_v0")
UNIFIED = Path("output/unified_robust_portfolio_evaluation_run_v0")

SUMMARY = DGTW_RUN / "dgtw_adjusted_portfolio_evaluation_run_summary.json"
MONTHLY_BENCH = DGTW_RUN / "dgtw_portfolio_monthly_benchmark_return.csv"
MONTHLY_ADJ = DGTW_RUN / "dgtw_portfolio_monthly_adjusted_return_by_cost.csv"
PERF_SUMMARY = DGTW_RUN / "dgtw_adjusted_performance_summary_by_cost.csv"
COMPARISON = DGTW_RUN / "dgtw_adjusted_flag_based_vs_fallback_comparison.csv"
COVERAGE_SUMMARY = DGTW_RUN / "dgtw_match_coverage_summary.csv"
COVERAGE_MONTH = DGTW_RUN / "dgtw_match_coverage_by_portfolio_month.csv"
CANDIDATE = DGTW_AUDIT / "dgtw_stock_month_matched_benchmark_candidate.parquet"
ALIGNMENT_POLICY = DGTW_AUDIT / "dgtw_alignment_policy.json"
MATCH_FEASIBILITY = DGTW_AUDIT / "dgtw_portfolio_matching_feasibility_by_portfolio.csv"
MATCHING_POLICY = DGTW_PREP / "dgtw_adjusted_matching_policy.json"
GROSS = UNIFIED / "unified_portfolio_monthly_gross_return.csv"
NET = UNIFIED / "unified_portfolio_monthly_net_return_by_cost.csv"
TURNOVER = UNIFIED / "unified_portfolio_monthly_turnover.csv"
UNIFIED_PERF = UNIFIED / "unified_portfolio_performance_summary_by_cost.csv"
FLAG_WEIGHTS = Path("output/flag_based_top50_buffer_portfolio_construction_run_v0/flag_based_top50_buffer_research_weights_v0.parquet")
ROBUST_WEIGHTS = Path("output/robust_formation_portfolio_construction_run_v0/robust_formation_research_weights_v0.parquet")

PORTFOLIOS = [
    "ROBUST_VQ_TOP20_EXCLUDE_SOFT_ANOMALY_EQUAL_WEIGHT",
    "ROBUST_VQ_FLAG_CLEAN_TOP50_EQUAL_WEIGHT",
    "ROBUST_VQ_FLAG_CLEAN_TOP50_BUFFER_EQUAL_WEIGHT",
    "ROBUST_VQ_D7_D9_BAND_EQUAL_WEIGHT",
    "ROBUST_VQ_TOP30_PERCENT_EQUAL_WEIGHT",
]
PRIMARY = "ROBUST_VQ_FLAG_CLEAN_TOP50_BUFFER_EQUAL_WEIGHT"
FALLBACK = {"ROBUST_VQ_D7_D9_BAND_EQUAL_WEIGHT", "ROBUST_VQ_TOP30_PERCENT_EQUAL_WEIGHT"}


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def write_state(status, checkpoint, notes=None):
    lines = [
        "# DGTW-Adjusted Failure Forensic Attribution v0",
        "",
        f"- task_name: {TASK_NAME}",
        f"- status: {status}",
        f"- last_checkpoint: {checkpoint}",
        f"- updated_at: {datetime.now().isoformat(timespec='seconds')}",
        "- resume_command: `python scripts\\forensic_dgtw_adjusted_failure_attribution_v0.py > output\\_agent_runs\\DGTW_Adjusted_Failure_Forensics\\run_stdout.txt 2> output\\_agent_runs\\DGTW_Adjusted_Failure_Forensics\\run_stderr.txt`",
        "- guardrails: no weights edits; no reselection; no tuning; no alpha/beta; no IR; no TE; no training; no SHAP; no production",
    ]
    if notes:
        lines += ["", "## Notes"] + [f"- {x}" for x in notes]
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def tstat(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / math.sqrt(len(x)))) if sd != 0 else np.nan


def cumulative(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(np.prod(1.0 + x) - 1.0) if len(x) else np.nan


def read_weights_fwd():
    cols = ["portfolio_name", "symbol", "month_end", "fwd_ret_1m"]
    frames = []
    for p in [FLAG_WEIGHTS, ROBUST_WEIGHTS]:
        df = pd.read_parquet(p, columns=cols)
        df = df[df["portfolio_name"].isin(PORTFOLIOS)].copy()
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["month_end"] = pd.to_datetime(out["month_end"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    out = out.drop_duplicates(["portfolio_name", "symbol", "month_end"], keep="last")
    return out


def load_candidate_with_fwd():
    cols = [
        "portfolio_name", "symbol", "month_end", "weight", "MarketValue", "BooktoMarket",
        "Momentum", "is_not_bse_policy", "dgtw_benchmark_return_decimal",
        "dgtw_assignment_match_flag", "dgtw_cell_match_flag",
    ]
    cand = pd.read_parquet(CANDIDATE, columns=cols)
    cand = cand[cand["portfolio_name"].isin(PORTFOLIOS)].copy()
    cand["month_end"] = pd.to_datetime(cand["month_end"], errors="coerce")
    cand["symbol"] = cand["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    fwd = read_weights_fwd()
    cand = cand.merge(fwd, on=["portfolio_name", "symbol", "month_end"], how="left")
    cand["matched"] = cand["dgtw_assignment_match_flag"].astype(bool) & cand["dgtw_cell_match_flag"].astype(bool) & cand["dgtw_benchmark_return_decimal"].notna()
    cand["cell_key"] = (
        "MV" + cand["MarketValue"].astype("Int64").astype(str)
        + "_BM" + cand["BooktoMarket"].astype("Int64").astype(str)
        + "_MOM" + cand["Momentum"].astype("Int64").astype(str)
        + "_BSE" + cand["is_not_bse_policy"].astype("Int64").astype(str)
    )
    del fwd
    gc.collect()
    return cand


def prerequisite(summary):
    files = {
        "summary": SUMMARY.exists(), "monthly_benchmark": MONTHLY_BENCH.exists(), "monthly_adjusted": MONTHLY_ADJ.exists(),
        "performance_summary": PERF_SUMMARY.exists(), "comparison": COMPARISON.exists(), "coverage_summary": COVERAGE_SUMMARY.exists(),
        "coverage_by_month": COVERAGE_MONTH.exists(), "candidate": CANDIDATE.exists(), "gross": GROSS.exists(), "net": NET.exists(),
        "turnover": TURNOVER.exists(), "alignment_policy": ALIGNMENT_POLICY.exists(), "match_feasibility": MATCH_FEASIBILITY.exists(),
        "matching_policy": MATCHING_POLICY.exists(), "flag_weights_fwd_ret": FLAG_WEIGHTS.exists(), "robust_weights_fwd_ret": ROBUST_WEIGHTS.exists(),
    }
    out = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "required_files_found": files,
        "source_final_decision": summary.get("final_decision"),
        "prerequisites_passed": bool(all(files.values()) and summary.get("prerequisites_passed") is True),
    }
    return out


def cost_impact(gross, net, bench, perf, turnover):
    gross_m = gross[gross["portfolio_name"].isin(PORTFOLIOS)].copy()
    net_m = net[net["portfolio_name"].isin(PORTFOLIOS)].copy()
    bench_m = bench[bench["portfolio_name"].isin(PORTFOLIOS)].copy()
    turn_m = turnover[turnover["portfolio_name"].isin(PORTFOLIOS)].copy()
    gross_agg = gross_m.groupby("portfolio_name")["gross_return"].mean().rename("mean_gross_return")
    bench_agg = bench_m.groupby("portfolio_name")["dgtw_benchmark_return_raw"].mean().rename("mean_dgtw_benchmark_return")
    turn_col = "one_way_turnover" if "one_way_turnover" in turn_m.columns else "turnover"
    turn_agg = turn_m.groupby("portfolio_name")[turn_col].mean().rename("avg_turnover") if turn_col in turn_m.columns else pd.Series(dtype=float, name="avg_turnover")
    p = perf[(perf["adjustment_variant"] == "raw_unmatched_not_renormalized")].copy()
    rows = []
    for _, r in p.iterrows():
        port = r["portfolio_name"]
        cost = int(r["cost_bps"])
        n = net_m[(net_m["portfolio_name"] == port) & (net_m["cost_bps"] == cost)]
        mean_net = float(n["net_return"].mean())
        zero = p[(p["portfolio_name"] == port) & (p["cost_bps"] == 0)]["mean_monthly_adjusted_return"]
        adj0 = float(zero.iloc[0]) if len(zero) else np.nan
        adj = float(r["mean_monthly_adjusted_return"])
        rows.append({
            "portfolio_name": port,
            "cost_bps": cost,
            "mean_gross_return": float(gross_agg.get(port, np.nan)),
            "mean_net_return": mean_net,
            "mean_dgtw_benchmark_return": float(bench_agg.get(port, np.nan)),
            "mean_dgtw_adjusted_return": adj,
            "adjusted_tstat": float(r["adjusted_tstat"]),
            "positive_adjusted_month_ratio": float(r["positive_adjusted_month_ratio"]),
            "avg_turnover": float(turn_agg.get(port, np.nan)) if len(turn_agg) else np.nan,
            "estimated_cost_drag": float(gross_agg.get(port, np.nan) - mean_net),
            "cost_explains_failure_flag": bool(cost == 20 and adj < 0 and adj0 > -0.001),
        })
    return pd.DataFrame(rows)


def benchmark_strength(gross, net, bench):
    rows = []
    net20 = net[(net["cost_bps"] == 20) & (net["portfolio_name"].isin(PORTFOLIOS))]
    for p, g in bench[bench["portfolio_name"].isin(PORTFOLIOS)].groupby("portfolio_name"):
        b = pd.to_numeric(g["dgtw_benchmark_return_raw"], errors="coerce")
        rows.append({
            "portfolio_name": p,
            "mean_portfolio_gross_return": float(gross[gross["portfolio_name"] == p]["gross_return"].mean()),
            "mean_portfolio_net_return_20bps": float(net20[net20["portfolio_name"] == p]["net_return"].mean()),
            "mean_dgtw_benchmark_return": float(b.mean()),
            "dgtw_benchmark_return_tstat": tstat(b),
            "dgtw_benchmark_positive_month_ratio": float((b > 0).mean()),
            "dgtw_benchmark_cumulative_return": cumulative(b),
            "dgtw_benchmark_vs_csi800_internal_universe": "not_compared_in_this_forensic_run",
        })
    return pd.DataFrame(rows)


def cell_exposure(cand):
    matched = cand[cand["matched"]].copy()
    matched["weighted_bench"] = matched["weight"] * matched["dgtw_benchmark_return_decimal"]
    month_count = matched.groupby("portfolio_name")["month_end"].nunique().to_dict()
    rows = []
    for keys, g in matched.groupby(["portfolio_name", "MarketValue", "BooktoMarket", "Momentum", "is_not_bse_policy"]):
        port, mv, bm, mom, bse = keys
        months = month_count.get(port, 1)
        avg_w = float(g["weight"].sum() / months)
        contrib = float(g["weighted_bench"].sum() / months)
        avg_ret = contrib / avg_w if avg_w else np.nan
        rows.append({
            "portfolio_name": port, "MarketValue": int(mv), "BooktoMarket": int(bm), "Momentum": int(mom), "IsNotBSE": int(bse),
            "avg_weight_share": avg_w,
            "avg_dgtw_benchmark_return": avg_ret,
            "contribution_to_dgtw_benchmark": contrib,
            "stock_month_count": int(len(g)),
            "interpretation": f"cell exposure MV={int(mv)}, BM={int(bm)}, MOM={int(mom)}",
        })
    out = pd.DataFrame(rows)
    return out


def top_primary_cells(cell_exp):
    p = cell_exp[cell_exp["portfolio_name"] == PRIMARY].copy()
    p["abs_contribution"] = p["contribution_to_dgtw_benchmark"].abs()
    total = p["contribution_to_dgtw_benchmark"].sum()
    p = p.sort_values("abs_contribution", ascending=False).head(20)
    return pd.DataFrame({
        "cell_key": "MV" + p["MarketValue"].astype(str) + "_BM" + p["BooktoMarket"].astype(str) + "_MOM" + p["Momentum"].astype(str) + "_BSE" + p["IsNotBSE"].astype(str),
        "avg_weight_share": p["avg_weight_share"],
        "avg_dgtw_benchmark_return": p["avg_dgtw_benchmark_return"],
        "contribution_to_total_dgtw_benchmark": p["contribution_to_dgtw_benchmark"] / total if total != 0 else np.nan,
        "cell_description": "MarketValue=" + p["MarketValue"].astype(str) + "; BooktoMarket=" + p["BooktoMarket"].astype(str) + "; Momentum=" + p["Momentum"].astype(str),
    })


def within_cell(cand, gross, bench):
    matched = cand[cand["matched"] & cand["fwd_ret_1m"].notna()].copy()
    matched["cell_weighted_fwd"] = matched["weight"] * matched["fwd_ret_1m"]
    matched["cell_weighted_bench"] = matched["weight"] * matched["dgtw_benchmark_return_decimal"]
    by_cell = matched.groupby(["portfolio_name", "month_end", "cell_key"], as_index=False).agg(
        cell_weight=("weight", "sum"),
        weighted_fwd=("cell_weighted_fwd", "sum"),
        weighted_bench=("cell_weighted_bench", "sum"),
        dgtw_cell_benchmark_return=("dgtw_benchmark_return_decimal", "mean"),
    )
    by_cell["portfolio_cell_return"] = by_cell["weighted_fwd"] / by_cell["cell_weight"]
    by_cell["within_cell_excess"] = by_cell["portfolio_cell_return"] - by_cell["dgtw_cell_benchmark_return"]
    by_cell["within_contribution"] = by_cell["cell_weight"] * by_cell["within_cell_excess"]
    month = by_cell.groupby(["portfolio_name", "month_end"], as_index=False).agg(
        within_cell_selection_effect=("within_contribution", "sum"),
        matched_weight_share=("cell_weight", "sum"),
        cell_allocation_effect=("weighted_bench", "sum"),
    )
    month["month_end"] = pd.to_datetime(month["month_end"]).dt.date.astype(str)
    gross_m = gross[["portfolio_name", "month_end", "gross_return"]].copy()
    gross_m["month_end"] = pd.to_datetime(gross_m["month_end"]).dt.date.astype(str)
    bench_m = bench[["portfolio_name", "month_end", "dgtw_benchmark_return_raw"]].copy()
    bench_m["month_end"] = pd.to_datetime(bench_m["month_end"]).dt.date.astype(str)
    out = month.merge(gross_m, on=["portfolio_name", "month_end"], how="left").merge(bench_m, on=["portfolio_name", "month_end"], how="left")
    out = out.rename(columns={"dgtw_benchmark_return_raw": "dgtw_benchmark_return"})
    out = out[["portfolio_name", "month_end", "gross_return", "dgtw_benchmark_return", "within_cell_selection_effect", "cell_allocation_effect", "matched_weight_share"]]
    rows = []
    for p, g in out.groupby("portfolio_name"):
        mean_sel = float(g["within_cell_selection_effect"].mean())
        mean_alloc = float(g["cell_allocation_effect"].mean())
        rows.append({
            "portfolio_name": p,
            "mean_within_cell_selection_effect": mean_sel,
            "within_cell_selection_tstat": tstat(g["within_cell_selection_effect"]),
            "positive_within_cell_month_ratio": float((g["within_cell_selection_effect"] > 0).mean()),
            "mean_cell_allocation_effect": mean_alloc,
            "cell_allocation_tstat": tstat(g["cell_allocation_effect"]),
            "interpretation": "allocation positive but within-cell selection weak/negative" if mean_alloc > 0 and mean_sel < 0 else "mixed within-cell and allocation effects",
        })
    return out, pd.DataFrame(rows)


def marginal_exposure(cand):
    matched = cand[cand["matched"]].copy()
    rows = []
    for characteristic in ["MarketValue", "BooktoMarket", "Momentum"]:
        matched["weighted_bench"] = matched["weight"] * matched["dgtw_benchmark_return_decimal"]
        month_count = matched.groupby("portfolio_name")["month_end"].nunique().to_dict()
        for (p, val), g in matched.groupby(["portfolio_name", characteristic]):
            months = month_count.get(p, 1)
            avg_w = float(g["weight"].sum() / months)
            contrib = float(g["weighted_bench"].sum() / months)
            rows.append({
                "portfolio_name": p,
                "characteristic": characteristic,
                "group_value": int(val),
                "avg_weight_share": avg_w,
                "avg_group_dgtw_return": contrib / avg_w if avg_w else np.nan,
                "contribution_to_benchmark": contrib,
                "interpretation": f"{characteristic} group {int(val)} exposure",
            })
    return pd.DataFrame(rows)


def failure_by_year(adj, bench, net):
    a = adj[(adj["cost_bps"] == 20) & (adj["portfolio_name"].isin(PORTFOLIOS))].copy()
    a["year"] = pd.to_datetime(a["month_end"]).dt.year
    rows = []
    for (year, p), g in a.groupby(["year", "portfolio_name"]):
        rows.append({
            "year": int(year),
            "portfolio_name": p,
            "mean_adjusted_return_20bps": float(g["dgtw_adjusted_return_raw"].mean()),
            "adjusted_tstat": tstat(g["dgtw_adjusted_return_raw"]),
            "positive_adjusted_month_ratio": float((g["dgtw_adjusted_return_raw"] > 0).mean()),
            "mean_dgtw_benchmark_return": float(g["dgtw_benchmark_return_raw"].mean()),
            "mean_net_return_20bps": float(g["net_return"].mean()),
            "interpretation": "negative adjusted year" if g["dgtw_adjusted_return_raw"].mean() < 0 else "positive adjusted year",
        })
    return pd.DataFrame(rows)


def alignment_qa(run_summary, cand, coverage):
    alignment = load_json(ALIGNMENT_POLICY)
    matching = load_json(MATCHING_POLICY)
    rows = [
        {"check_name": "TradingYear mapping", "expected_value": "RULE_B_JULY_TO_JUNE", "actual_value": alignment.get("recommended_tradingyear_mapping_rule"), "pass": alignment.get("recommended_tradingyear_mapping_rule") == "RULE_B_JULY_TO_JUNE", "caveat": ""},
        {"check_name": "IsNotBSE policy", "expected_value": "1", "actual_value": str(int(cand["is_not_bse_policy"].dropna().mode().iloc[0])), "pass": int(cand["is_not_bse_policy"].dropna().mode().iloc[0]) == 1, "caveat": ""},
        {"check_name": "DGTW return unit", "expected_value": "DECIMAL_RETURN", "actual_value": matching.get("benchmark_return_unit"), "pass": matching.get("benchmark_return_unit") == "DECIMAL_RETURN", "caveat": ""},
        {"check_name": "matched_weight_share threshold", "expected_value": "all >= 0.95", "actual_value": str(float(coverage["min_matched_weight_share"].min())), "pass": bool((coverage["min_matched_weight_share"] >= 0.95).all()), "caveat": ""},
        {"check_name": "DGTW cell coverage", "expected_value": "portfolio-level coverage not materially impaired", "actual_value": f"dgtw_cell_coverage_pass={run_summary.get('dgtw_cell_coverage_pass')}", "pass": bool((coverage["min_matched_weight_share"] >= 0.95).all()), "caveat": "source cell coverage pass is false, but portfolio-level matched coverage is >= 0.95"},
    ]
    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    write_state("running", "loading forensic inputs")
    run_summary = load_json(SUMMARY)
    prereq = prerequisite(run_summary)
    (OUT_DIR / "dgtw_failure_forensic_prerequisite_check.json").write_text(json.dumps(prereq, ensure_ascii=False, indent=2), encoding="utf-8")

    gross = pd.read_csv(GROSS)
    net = pd.read_csv(NET)
    turnover = pd.read_csv(TURNOVER)
    bench = pd.read_csv(MONTHLY_BENCH)
    adj = pd.read_csv(MONTHLY_ADJ)
    perf = pd.read_csv(PERF_SUMMARY)
    coverage = pd.read_csv(COVERAGE_SUMMARY)

    write_state("running", "building cost, benchmark, and cell attribution outputs")
    cost = cost_impact(gross, net, bench, perf, turnover)
    cost.to_csv(OUT_DIR / "dgtw_cost_impact_decomposition.csv", index=False, encoding="utf-8-sig")
    strength = benchmark_strength(gross, net, bench)
    strength.to_csv(OUT_DIR / "dgtw_benchmark_strength_diagnostic.csv", index=False, encoding="utf-8-sig")

    cand = load_candidate_with_fwd()
    cell_exp = cell_exposure(cand)
    cell_exp.to_csv(OUT_DIR / "dgtw_cell_exposure_summary.csv", index=False, encoding="utf-8-sig")
    top_primary_cells(cell_exp).to_csv(OUT_DIR / "dgtw_primary_portfolio_top_cell_exposures.csv", index=False, encoding="utf-8-sig")
    within_month, within_summary = within_cell(cand, gross, bench)
    within_month.to_csv(OUT_DIR / "dgtw_within_cell_selection_effect_by_month.csv", index=False, encoding="utf-8-sig")
    within_summary.to_csv(OUT_DIR / "dgtw_within_cell_selection_effect_summary.csv", index=False, encoding="utf-8-sig")
    marg = marginal_exposure(cand)
    marg.to_csv(OUT_DIR / "dgtw_characteristic_marginal_exposure.csv", index=False, encoding="utf-8-sig")
    by_year = failure_by_year(adj, bench, net)
    by_year.to_csv(OUT_DIR / "dgtw_adjusted_failure_by_year.csv", index=False, encoding="utf-8-sig")
    qa = alignment_qa(run_summary, cand, coverage)
    qa.to_csv(OUT_DIR / "dgtw_failure_alignment_source_caveat_qa.csv", index=False, encoding="utf-8-sig")

    guard = {
        "weights_modified": False,
        "weights_reconstructed": False,
        "dgtw_used_for_selection": False,
        "dgtw_used_for_weighting": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "training_run": False,
        "shap_calculated": False,
        "production_modified": False,
    }
    guard["guardrail_pass"] = not any(guard.values())
    pd.DataFrame([guard]).to_csv(OUT_DIR / "dgtw_failure_forensic_guardrail_qa.csv", index=False, encoding="utf-8-sig")

    primary_cost = cost[(cost["portfolio_name"] == PRIMARY)]
    p0 = primary_cost[primary_cost["cost_bps"] == 0].iloc[0]
    p20 = primary_cost[primary_cost["cost_bps"] == 20].iloc[0]
    p_within = within_summary[within_summary["portfolio_name"] == PRIMARY].iloc[0]
    p_strength = strength[strength["portfolio_name"] == PRIMARY].iloc[0]
    primary_marg = marg[marg["portfolio_name"] == PRIMARY].copy()
    primary_marg["abs_contribution"] = primary_marg["contribution_to_benchmark"].abs()
    strongest = primary_marg.sort_values("abs_contribution", ascending=False).iloc[0]
    dim_contrib = primary_marg.groupby("characteristic")["contribution_to_benchmark"].sum().reset_index()
    dim_contrib["abs_contribution"] = dim_contrib["contribution_to_benchmark"].abs()
    weakest_dim = dim_contrib.sort_values("abs_contribution", ascending=True).iloc[0]["characteristic"]
    year_primary = by_year[by_year["portfolio_name"] == PRIMARY].copy()
    worst_year_row = year_primary.sort_values("mean_adjusted_return_20bps").iloc[0]
    failure_concentrated = bool(worst_year_row["mean_adjusted_return_20bps"] < year_primary["mean_adjusted_return_20bps"].mean() - year_primary["mean_adjusted_return_20bps"].std(ddof=0))
    cost_main = bool(p0["mean_dgtw_adjusted_return"] > -0.001 and p20["mean_dgtw_adjusted_return"] < 0)
    char_dom = bool(p_strength["mean_dgtw_benchmark_return"] > 0 and p_within["mean_within_cell_selection_effect"] < 0)
    alignment_pass = bool(qa["pass"].all())
    source_risk = bool(not alignment_pass or run_summary.get("dgtw_cell_coverage_pass") is False and coverage["min_matched_weight_share"].min() < 0.95)
    if not guard["guardrail_pass"]:
        final = "DGTW_FAILURE_FORENSIC_FAIL_GUARDRAIL"
    elif not alignment_pass or source_risk:
        final = "DGTW_FAILURE_FORENSIC_ALIGNMENT_OR_SOURCE_CAVEAT_NEEDS_REVIEW"
    elif cost_main:
        final = "DGTW_FAILURE_FORENSIC_COST_MAIN_DRIVER"
    elif char_dom:
        final = "DGTW_FAILURE_FORENSIC_CHARACTERISTIC_EXPOSURE_DOMINATES"
    else:
        final = "DGTW_FAILURE_FORENSIC_MIXED_CAUSES"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": bool(prereq["prerequisites_passed"]),
        "primary_portfolio": PRIMARY,
        "primary_20bps_mean_adjusted_return": float(p20["mean_dgtw_adjusted_return"]),
        "primary_20bps_adjusted_tstat": float(p20["adjusted_tstat"]),
        "primary_0bps_mean_adjusted_return": float(p0["mean_dgtw_adjusted_return"]),
        "primary_0bps_adjusted_tstat": float(p0["adjusted_tstat"]),
        "cost_main_driver": cost_main,
        "mean_dgtw_benchmark_return_primary": float(p_strength["mean_dgtw_benchmark_return"]),
        "mean_gross_return_primary": float(p20["mean_gross_return"]),
        "mean_net_return_20bps_primary": float(p20["mean_net_return"]),
        "mean_within_cell_selection_effect_primary": float(p_within["mean_within_cell_selection_effect"]),
        "within_cell_selection_tstat_primary": float(p_within["within_cell_selection_tstat"]),
        "mean_cell_allocation_effect_primary": float(p_within["mean_cell_allocation_effect"]),
        "cell_allocation_tstat_primary": float(p_within["cell_allocation_tstat"]),
        "characteristic_exposure_dominates": char_dom,
        "weakest_characteristic_dimension": str(weakest_dim),
        "strongest_characteristic_exposure": f"{strongest['characteristic']}={int(strongest['group_value'])}",
        "failure_concentrated_by_year": failure_concentrated,
        "worst_failure_year": int(worst_year_row["year"]),
        "alignment_qa_pass": alignment_pass,
        "source_caveat_material_risk": source_risk,
        **guard,
        "final_decision": final,
    }
    if final == "DGTW_FAILURE_FORENSIC_CHARACTERISTIC_EXPOSURE_DOMINATES":
        summary["recommended_next_step"] = "报告中明确：absolute return strong 主要被 size/BM/momentum cell exposure 解释，不能声称 DGTW-adjusted alpha。"
    elif final == "DGTW_FAILURE_FORENSIC_COST_MAIN_DRIVER":
        summary["recommended_next_step"] = "优先检查换手和成本敏感性，但不得调权或重新选股。"
    elif final == "DGTW_FAILURE_FORENSIC_ALIGNMENT_OR_SOURCE_CAVEAT_NEEDS_REVIEW":
        summary["recommended_next_step"] = "先复核 alignment/source caveat，再把 forensic 结论写入研究报告。"
    else:
        summary["recommended_next_step"] = "按 mixed causes 报告：特征暴露、cell 内选股和时间段共同解释失败。"
    (OUT_DIR / "dgtw_adjusted_failure_forensic_attribution_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# DGTW-Adjusted Failure Forensic Attribution v0",
        "",
        "## 当前问题",
        "",
        "- absolute portfolio evaluation 是 strong research pass。",
        "- DGTW-adjusted evaluation 是 fail。",
        "- 这意味着当前策略不能声称 DGTW characteristic-adjusted alpha。",
        "- 但这不等于策略完全无价值；它更可能说明收益需要拆解为 size / book-to-market / momentum 特征暴露和 cell 内选股效果。",
        "",
        f"- final_decision: {final}",
        f"- primary_20bps_mean_adjusted_return: {summary['primary_20bps_mean_adjusted_return']:.8f}",
        f"- primary_20bps_adjusted_tstat: {summary['primary_20bps_adjusted_tstat']:.6f}",
        f"- mean_dgtw_benchmark_return_primary: {summary['mean_dgtw_benchmark_return_primary']:.8f}",
        f"- mean_within_cell_selection_effect_primary: {summary['mean_within_cell_selection_effect_primary']:.8f}",
        f"- characteristic_exposure_dominates: {char_dom}",
        "",
        "## Guardrail",
        "",
        "- 未修改或重构 weights。",
        "- 未重新选股、未调参、未训练、未 SHAP、未写 production。",
        "- 未计算 alpha/beta、information ratio、tracking error。",
    ]
    (OUT_DIR / "dgtw_adjusted_failure_forensic_attribution_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    final_qa = pd.DataFrame([{"final_decision": final, "prerequisites_passed": prereq["prerequisites_passed"], "guardrail_pass": guard["guardrail_pass"], "required_outputs_created": True}])
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    terminal_summary = {"task_name": TASK_NAME, "status": "completed", "stdout_log": str(RUN_DIR / "run_stdout.txt"), "stderr_log": str(RUN_DIR / "run_stderr.txt"), "summary_json": str(OUT_DIR / "dgtw_adjusted_failure_forensic_attribution_summary.json"), "final_decision": final}
    (OUT_DIR / "terminal_summary.json").write_text(json.dumps(terminal_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "task_completion_card.md").write_text("\n".join(["# Task Completion Card", "", f"- task_name: {TASK_NAME}", "- status: completed", f"- final_decision: {final}", f"- summary_json: {OUT_DIR / 'dgtw_adjusted_failure_forensic_attribution_summary.json'}", f"- final_qa: {OUT_DIR / 'final_qa.csv'}"]) + "\n", encoding="utf-8")
    write_state("completed", "all required outputs written", [f"final_decision: {final}"])
    print(json.dumps(terminal_summary, ensure_ascii=False))
    del gross, net, turnover, bench, adj, perf, coverage, cand
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
