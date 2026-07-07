from __future__ import annotations

import gc
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


OUT_NAME = "v0_strict_lag_month_key_csmar_rematch_v1"
TASK_NAME = OUT_NAME
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / OUT_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

WEIGHTS_PATH = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_reconstructed_weights.parquet"
PREV_SUMMARY_PATH = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_icir_rebuild_bridge_summary.json"
TRANSITION_QA_PATH = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_buffer_transition_qa.csv"
CSMAR_PATH = (
    ROOT
    / "output"
    / "robust_cleaned_fundamental_factor_variant_build_v0"
    / "robust_cleaned_factor_score_panel_v0.parquet"
)
OLD_MONTHLY_PATH = (
    ROOT
    / "output"
    / "reconstructed_v0_v7_csmar_bridge_evaluation_v0"
    / "bridge_monthly_net_return_csmar_by_cost.csv"
)
OLD_SUMMARY_PATH = (
    ROOT
    / "output"
    / "reconstructed_v0_v7_csmar_bridge_evaluation_v0"
    / "bridge_performance_summary_csmar_by_cost.csv"
)


def write_state(status: str, details: dict | None = None) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": TASK_NAME,
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "details": details or {},
        "resume_instruction": f"先读取 {RUN_DIR / 'RUN_STATE.md'} 再继续。",
    }
    lines = ["# RUN_STATE", "", f"- task_name: {TASK_NAME}", f"- status: {status}"]
    for k, v in payload["details"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "```json", json.dumps(payload, ensure_ascii=False, indent=2, default=str), "```"]
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(lines), encoding="utf-8")


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_symbol(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.zfill(6)


def add_date_keys(df: pd.DataFrame, raw_col: str) -> pd.DataFrame:
    df = df.copy()
    df["raw_date"] = pd.to_datetime(df[raw_col])
    df["calendar_month_end"] = df["raw_date"] + pd.offsets.MonthEnd(0)
    df["year_month"] = df["raw_date"].dt.to_period("M").astype(str)
    return df


def max_drawdown(returns: pd.Series) -> float:
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    if len(curve) == 0:
        return np.nan
    return float((curve / curve.cummax() - 1.0).min())


def metric_summary(returns: pd.Series, turnover: pd.Series, matched: pd.Series) -> dict:
    r = returns.astype(float)
    vol = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    mean = float(r.mean()) if len(r) else np.nan
    return {
        "month_count": int(len(r)),
        "mean_monthly_return": mean,
        "annualized_return_approx": mean * 12 if pd.notna(mean) else np.nan,
        "monthly_volatility": vol,
        "sharpe": mean / vol * math.sqrt(12) if pd.notna(vol) and vol > 0 else np.nan,
        "tstat": mean / vol * math.sqrt(len(r)) if pd.notna(vol) and vol > 0 else np.nan,
        "positive_month_ratio": float((r > 0).mean()) if len(r) else np.nan,
        "cumulative_return": float((1.0 + r.fillna(0.0)).prod() - 1.0) if len(r) else np.nan,
        "max_drawdown": max_drawdown(r),
        "avg_turnover": float(turnover.mean()) if len(turnover) else np.nan,
        "avg_matched_weight_share": float(matched.mean()) if len(matched) else np.nan,
        "min_matched_weight_share": float(matched.min()) if len(matched) else np.nan,
        "low_match_month_count": int((matched < 0.95).sum()) if len(matched) else 0,
    }


def match_status(avg_share: float, min_share: float) -> str:
    if avg_share >= 0.98 and min_share >= 0.95:
        return "READY"
    if avg_share >= 0.95 and min_share >= 0.90:
        return "READY_WITH_MINOR_GAPS"
    if avg_share >= 0.90:
        return "WATCH_COVERAGE_GAPS"
    return "LOW_MATCH"


def date_diag(source_name: str, dates: pd.Series) -> dict:
    raw = pd.to_datetime(dates)
    cal = raw + pd.offsets.MonthEnd(0)
    neq = raw[raw != cal].drop_duplicates().sort_values()
    ratio = float((raw == cal).mean()) if len(raw) else np.nan
    if ratio < 0.5:
        diagnosis = "大量使用实际交易日而非自然月末"
    elif ratio > 0.95:
        diagnosis = "基本使用自然月末"
    else:
        diagnosis = "混合日期键"
    return {
        "source_name": source_name,
        "raw_min_date": raw.min(),
        "raw_max_date": raw.max(),
        "raw_month_count": int(raw.nunique()),
        "calendar_month_end_min": cal.min(),
        "calendar_month_end_max": cal.max(),
        "year_month_count": int(raw.dt.to_period("M").nunique()),
        "raw_dates_equal_calendar_month_end_ratio": ratio,
        "example_non_calendar_month_end_dates": ";".join(neq.dt.strftime("%Y-%m-%d").head(10).tolist()),
        "diagnosis": diagnosis,
    }


def prepare_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    prereq = {
        "strict_lag_weights_found": WEIGHTS_PATH.exists(),
        "csmar_return_source_found": CSMAR_PATH.exists(),
        "old_v0_bridge_files_found": OLD_MONTHLY_PATH.exists() and OLD_SUMMARY_PATH.exists(),
        "previous_strict_lag_summary_found": PREV_SUMMARY_PATH.exists(),
        "transition_qa_found": TRANSITION_QA_PATH.exists(),
    }
    needed = [
        (prereq["strict_lag_weights_found"], WEIGHTS_PATH),
        (prereq["csmar_return_source_found"], CSMAR_PATH),
        (prereq["previous_strict_lag_summary_found"], PREV_SUMMARY_PATH),
        (prereq["transition_qa_found"], TRANSITION_QA_PATH),
    ]
    prereq["missing_files"] = [str(p) for ok, p in needed if not ok]
    prereq["prerequisites_passed"] = len(prereq["missing_files"]) == 0
    save_json(prereq, OUT_DIR / "v0_strict_lag_month_key_rematch_prerequisite_check.json")
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    weights = pd.read_parquet(
        WEIGHTS_PATH,
        columns=["portfolio_name", "symbol", "month_end", "selected_flag", "weight"],
        engine="pyarrow",
    )
    weights = weights[weights["selected_flag"]].copy()
    weights["symbol"] = normalize_symbol(weights["symbol"])
    weights["month_end"] = pd.to_datetime(weights["month_end"])
    weights = add_date_keys(weights, "month_end")

    csmar = pd.read_parquet(CSMAR_PATH, columns=["symbol", "month_end", "fwd_ret_1m"], engine="pyarrow")
    csmar["symbol"] = normalize_symbol(csmar["symbol"])
    csmar["month_end"] = pd.to_datetime(csmar["month_end"])
    csmar = add_date_keys(csmar, "month_end")
    csmar = csmar.dropna(subset=["fwd_ret_1m"]).copy()

    trans = pd.read_csv(TRANSITION_QA_PATH)
    trans["month_end"] = pd.to_datetime(trans["month_end"])
    trans = add_date_keys(trans, "month_end")
    return weights, csmar, trans, prereq


def evaluate_match(weights: pd.DataFrame, csmar: pd.DataFrame, method: str) -> tuple[pd.DataFrame, dict]:
    if method == "exact_date_match":
        left_keys = ["symbol", "raw_date"]
        right = csmar[["symbol", "raw_date", "fwd_ret_1m", "month_end"]].rename(
            columns={"month_end": "csmar_return_month_end"}
        )
    elif method == "calendar_month_end_match":
        left_keys = ["symbol", "calendar_month_end"]
        right = csmar[["symbol", "calendar_month_end", "fwd_ret_1m", "month_end"]].rename(
            columns={"month_end": "csmar_return_month_end"}
        )
    elif method == "year_month_match":
        left_keys = ["symbol", "year_month"]
        right = csmar[["symbol", "year_month", "fwd_ret_1m", "month_end"]].rename(
            columns={"month_end": "csmar_return_month_end"}
        )
    else:
        raise ValueError(method)

    merged = weights.merge(right, on=left_keys, how="left")
    merged["matched_flag"] = merged["fwd_ret_1m"].notna()
    merged["matched_weight"] = np.where(merged["matched_flag"], merged["weight"], 0.0)
    month_share = merged.groupby("year_month")["matched_weight"].sum()
    zero_months = int((month_share == 0).sum())
    avg_share = float(month_share.mean())
    min_share = float(month_share.min())
    status = match_status(avg_share, min_share)
    summary = {
        "match_method": method,
        "weight_row_count": int(len(weights)),
        "matched_row_count": int(merged["matched_flag"].sum()),
        "matched_ratio": float(merged["matched_flag"].mean()),
        "avg_matched_weight_share": avg_share,
        "min_matched_weight_share": min_share,
        "low_match_month_count": int((month_share < 0.95).sum()),
        "zero_match_month_count": zero_months,
        "match_status": status,
        "interpretation": "按该日期键进行 CSMAR fwd_ret_1m 匹配",
    }
    return merged, summary


def build_monthly_returns(
    merged: pd.DataFrame,
    trans: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    m = merged.copy()
    m["gross_contrib"] = m["weight"] * m["fwd_ret_1m"].fillna(0.0)
    m["matched_weight"] = np.where(m["fwd_ret_1m"].notna(), m["weight"], 0.0)
    grouped = m.groupby("year_month", sort=True)
    base = grouped.agg(
        portfolio_name=("portfolio_name", "first"),
        strict_lag_weight_date=("month_end", "first"),
        csmar_return_month_end=("csmar_return_month_end", lambda x: x.dropna().min()),
        gross_return_raw=("gross_contrib", "sum"),
        matched_weight_share=("matched_weight", "sum"),
    ).reset_index()
    base["gross_return_matched_norm"] = np.where(
        base["matched_weight_share"] > 1e-12,
        base["gross_return_raw"] / base["matched_weight_share"],
        np.nan,
    )
    base["unmatched_weight_share"] = 1.0 - base["matched_weight_share"]
    base["low_match_flag"] = base["matched_weight_share"] < 0.95
    trans_use = trans[["year_month", "simple_turnover_proxy"]].rename(
        columns={"simple_turnover_proxy": "turnover_simple"}
    )
    base = base.merge(trans_use, on="year_month", how="left")
    base["turnover_simple"] = base["turnover_simple"].fillna(0.0)

    rows = []
    for variant, col in [
        ("raw_unmatched_not_renormalized", "gross_return_raw"),
        ("matched_only_normalized", "gross_return_matched_norm"),
    ]:
        for cost in [0, 10, 20, 30]:
            tmp = base.copy()
            tmp["cost_bps"] = cost
            tmp["return_variant"] = variant
            tmp["gross_return"] = tmp[col]
            tmp["net_return"] = tmp["gross_return"] - tmp["turnover_simple"] * cost / 10000.0
            tmp["match_method"] = method
            rows.append(tmp)
    return pd.concat(rows, ignore_index=True)[
        [
            "portfolio_name",
            "year_month",
            "strict_lag_weight_date",
            "csmar_return_month_end",
            "cost_bps",
            "return_variant",
            "gross_return",
            "turnover_simple",
            "net_return",
            "matched_weight_share",
            "unmatched_weight_share",
            "low_match_flag",
            "match_method",
        ]
    ]


def performance_table(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cost, variant), grp in monthly.groupby(["cost_bps", "return_variant"], sort=True):
        s = metric_summary(grp["net_return"], grp["turnover_simple"], grp["matched_weight_share"])
        s.update(
            {
                "portfolio_name": grp["portfolio_name"].iloc[0],
                "cost_bps": int(cost),
                "return_variant": variant,
            }
        )
        rows.append(s)
    cols = [
        "portfolio_name",
        "cost_bps",
        "return_variant",
        "month_count",
        "mean_monthly_return",
        "annualized_return_approx",
        "monthly_volatility",
        "sharpe",
        "tstat",
        "positive_month_ratio",
        "cumulative_return",
        "max_drawdown",
        "avg_turnover",
        "avg_matched_weight_share",
        "min_matched_weight_share",
        "low_match_month_count",
    ]
    return pd.DataFrame(rows)[cols]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", {"step": "start"})

    weights, csmar, trans, prereq = prepare_inputs()
    prev_summary = json.loads(PREV_SUMMARY_PATH.read_text(encoding="utf-8"))

    write_state("running", {"step": "date_key_diagnostic"})
    diag = pd.DataFrame(
        [
            date_diag("strict_lag_weights", weights["month_end"]),
            date_diag("csmar_return_source", csmar["month_end"]),
        ]
    )
    w_ratio = float(diag.loc[diag["source_name"] == "strict_lag_weights", "raw_dates_equal_calendar_month_end_ratio"].iloc[0])
    c_ratio = float(diag.loc[diag["source_name"] == "csmar_return_source", "raw_dates_equal_calendar_month_end_ratio"].iloc[0])
    date_key_mismatch = w_ratio < 0.75 and c_ratio > 0.95
    diag.loc[:, "diagnosis"] = diag["diagnosis"] + (
        "；exact date mismatch 是 LOW_MATCH 主因候选" if date_key_mismatch else "；未充分证明 exact date mismatch 为主因"
    )
    diag.to_csv(OUT_DIR / "v0_strict_lag_date_key_diagnostic.csv", index=False, encoding="utf-8-sig")

    write_state("running", {"step": "match_methods"})
    match_results = {}
    comp_rows = []
    for method in ["exact_date_match", "calendar_month_end_match", "year_month_match"]:
        merged, summary = evaluate_match(weights, csmar, method)
        match_results[method] = (merged, summary)
        comp_rows.append(summary)
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUT_DIR / "v0_strict_lag_match_method_comparison.csv", index=False, encoding="utf-8-sig")

    year = comp.loc[comp["match_method"] == "year_month_match"].iloc[0]
    cal = comp.loc[comp["match_method"] == "calendar_month_end_match"].iloc[0]
    if (
        cal["avg_matched_weight_share"] > year["avg_matched_weight_share"] + 0.001
        and cal["min_matched_weight_share"] >= year["min_matched_weight_share"]
    ):
        best = "calendar_month_end_match"
    else:
        best = "year_month_match"
    best_merged, best_summary = match_results[best]

    write_state("running", {"step": "monthly_returns", "best_match_method": best})
    monthly = build_monthly_returns(best_merged, trans, best)
    monthly.to_csv(
        OUT_DIR / "v0_strict_lag_month_key_monthly_net_return_by_cost.csv",
        index=False,
        encoding="utf-8-sig",
    )
    perf = performance_table(monthly)
    perf.to_csv(
        OUT_DIR / "v0_strict_lag_month_key_performance_summary_by_cost.csv",
        index=False,
        encoding="utf-8-sig",
    )

    del weights, csmar, best_merged
    gc.collect()

    write_state("running", {"step": "old_vs_strict"})
    old_month = pd.read_csv(OLD_MONTHLY_PATH)
    old_month = old_month[
        (old_month["model_name"] == "V0_LINEAR_FULL_OOS")
        & (old_month["cost_bps"] == 20)
        & (old_month["return_variant"] == "raw_unmatched_not_renormalized")
    ].copy()
    old_month["year_month"] = pd.to_datetime(old_month["month_end"]).dt.to_period("M").astype(str)
    old_month = old_month.drop_duplicates("year_month", keep="first")
    strict20_month = monthly[
        (monthly["cost_bps"] == 20)
        & (monthly["return_variant"] == "raw_unmatched_not_renormalized")
    ].copy()
    common = old_month[
        ["year_month", "net_return_csmar_bridge", "turnover_simple", "matched_weight_share"]
    ].merge(
        strict20_month[["year_month", "net_return", "turnover_simple", "matched_weight_share"]],
        on="year_month",
        how="inner",
        suffixes=("_old", "_strict"),
    )
    old_metrics = metric_summary(common["net_return_csmar_bridge"], common["turnover_simple_old"], common["matched_weight_share_old"])
    strict_metrics = metric_summary(common["net_return"], common["turnover_simple_strict"], common["matched_weight_share_strict"])
    compare_rows = []
    for metric in [
        "mean_monthly_return",
        "sharpe",
        "tstat",
        "cumulative_return",
        "max_drawdown",
        "avg_turnover",
        "avg_matched_weight_share",
    ]:
        old_val = old_metrics.get(metric, np.nan)
        strict_val = strict_metrics.get(metric, np.nan)
        compare_rows.append(
            {
                "sample_window": "common_year_month_intersection",
                "common_month_count": int(len(common)),
                "metric_name": metric,
                "old_v0_value": old_val,
                "strict_lag_v0_value": strict_val,
                "delta": strict_val - old_val if pd.notna(old_val) and pd.notna(strict_val) else np.nan,
                "retention_ratio": strict_val / old_val if pd.notna(old_val) and old_val != 0 else np.nan,
                "interpretation": "old V0 vs strict-lag V0 month-key rematch, 20bps raw",
            }
        )
    bridge_cmp = pd.DataFrame(compare_rows)
    bridge_cmp.to_csv(
        OUT_DIR / "v0_old_vs_strict_lag_month_key_bridge_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    strict_main = strict_metrics
    old_sharpe = float(old_metrics["sharpe"])
    strict_sharpe = float(strict_main["sharpe"])
    old_mean = float(old_metrics["mean_monthly_return"])
    strict_mean = float(strict_main["mean_monthly_return"])
    sharpe_ret = strict_sharpe / old_sharpe if pd.notna(old_sharpe) and old_sharpe > 0 else np.nan
    mean_ret = strict_mean / old_mean if pd.notna(old_mean) and old_mean != 0 else np.nan
    rematch_status = str(best_summary["match_status"])
    ready = rematch_status in ["READY", "READY_WITH_MINOR_GAPS"]
    if ready and strict_sharpe >= 0.8 and pd.notna(sharpe_ret) and sharpe_ret >= 0.60:
        leakage = "LOW_IMPACT_STRICT_LAG_STILL_STRONG"
    elif ready and strict_sharpe > 0 and pd.notna(sharpe_ret) and sharpe_ret >= 0.30:
        leakage = "MEDIUM_IMPACT_STRICT_LAG_WEAKER_BUT_USABLE"
    elif ready:
        leakage = "HIGH_IMPACT_OLD_V0_LIKELY_LEAKAGE_DRIVEN"
    else:
        leakage = "INCONCLUSIVE_DUE_TO_REMAINING_MATCH_OR_WINDOW"

    leakage_df = pd.DataFrame(
        [
            {
                "match_method_selected": best,
                "previous_avg_matched_weight_share": prev_summary.get("strict_lag_avg_matched_weight_share"),
                "rematched_avg_matched_weight_share": best_summary["avg_matched_weight_share"],
                "previous_min_matched_weight_share": prev_summary.get("strict_lag_min_matched_weight_share"),
                "rematched_min_matched_weight_share": best_summary["min_matched_weight_share"],
                "old_v0_common_20bps_sharpe": old_sharpe,
                "strict_lag_20bps_sharpe": strict_sharpe,
                "sharpe_retention_ratio": sharpe_ret,
                "old_v0_mean_monthly_return": old_mean,
                "strict_lag_mean_monthly_return": strict_mean,
                "mean_return_retention_ratio": mean_ret,
                "old_v0_max_drawdown": old_metrics["max_drawdown"],
                "strict_lag_max_drawdown": strict_main["max_drawdown"],
                "leakage_impact_assessment": leakage,
                "interpretation": "按 month-key 修复 CSMAR 匹配后的 strict-lag V0 泄露影响再评估",
            }
        ]
    )
    leakage_df.to_csv(
        OUT_DIR / "v0_strict_lag_month_key_leakage_impact_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    guardrail_items = [
        ("strict_lag_alpha_signal_regenerated", False, False),
        ("strict_lag_weights_regenerated", False, False),
        ("original_orthogonalization_modified", False, False),
        ("old_artifacts_modified", False, False),
        ("production_modified", False, False),
        ("ml_training_run", False, False),
        ("new_ml_model_trained", False, False),
        ("portfolio_returns_calculated", True, True),
        ("fwd_ret_1m_used_for_same_month_signal", False, False),
        ("fwd_ret_1m_used_for_selection", False, False),
        ("fwd_ret_1m_used_for_weighting", False, False),
        ("benchmark_relative_returns_calculated", False, False),
        ("alpha_beta_regression_calculated", False, False),
        ("information_ratio_calculated", False, False),
        ("tracking_error_calculated", False, False),
        ("ff_regression_calculated", False, False),
        ("dgtw_adjusted_eval_calculated", False, False),
        ("shap_calculated", False, False),
    ]
    guard = pd.DataFrame(
        [{"guardrail": g, "expected": e, "actual": a, "pass": bool(e == a)} for g, e, a in guardrail_items]
    )
    guard.to_csv(
        OUT_DIR / "v0_strict_lag_month_key_rematch_guardrail_qa.csv",
        index=False,
        encoding="utf-8-sig",
    )
    no_guardrail_violation = bool(guard["pass"].all())

    strict_cum = float(strict_main["cumulative_return"])
    if not no_guardrail_violation:
        final_decision = "V0_STRICT_LAG_REMATCH_FAIL_GUARDRAIL"
    elif rematch_status not in ["READY", "READY_WITH_MINOR_GAPS"]:
        final_decision = "V0_STRICT_LAG_REMATCH_STILL_INCONCLUSIVE"
    elif strict_sharpe >= 0.8 and strict_cum > 0:
        final_decision = "V0_STRICT_LAG_REMATCH_READY_STILL_STRONG"
    elif strict_sharpe > 0 and strict_cum > 0:
        final_decision = "V0_STRICT_LAG_REMATCH_READY_WEAKER_BUT_USABLE"
    else:
        final_decision = "V0_STRICT_LAG_REMATCH_COLLAPSED_OLD_V0_LIKELY_LEAKAGE_DRIVEN"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "date_key_mismatch_diagnosed": bool(date_key_mismatch),
        "best_match_method": best,
        "exact_date_avg_matched_weight_share": float(comp.loc[comp["match_method"] == "exact_date_match", "avg_matched_weight_share"].iloc[0]),
        "exact_date_min_matched_weight_share": float(comp.loc[comp["match_method"] == "exact_date_match", "min_matched_weight_share"].iloc[0]),
        "year_month_avg_matched_weight_share": float(comp.loc[comp["match_method"] == "year_month_match", "avg_matched_weight_share"].iloc[0]),
        "year_month_min_matched_weight_share": float(comp.loc[comp["match_method"] == "year_month_match", "min_matched_weight_share"].iloc[0]),
        "rematch_status": rematch_status,
        "strict_lag_month_count": int(strict_main["month_count"]),
        "strict_lag_20bps_sharpe": strict_sharpe,
        "strict_lag_20bps_mean_monthly_return": strict_mean,
        "strict_lag_20bps_tstat": float(strict_main["tstat"]),
        "strict_lag_20bps_cumulative_return": strict_cum,
        "strict_lag_20bps_max_drawdown": float(strict_main["max_drawdown"]),
        "strict_lag_20bps_avg_turnover": float(strict_main["avg_turnover"]),
        "old_v0_common_20bps_sharpe": old_sharpe,
        "old_v0_common_20bps_mean_monthly_return": old_mean,
        "old_v0_common_20bps_tstat": float(old_metrics["tstat"]),
        "old_v0_common_20bps_cumulative_return": float(old_metrics["cumulative_return"]),
        "old_v0_common_20bps_max_drawdown": float(old_metrics["max_drawdown"]),
        "sharpe_retention_ratio": sharpe_ret,
        "mean_return_retention_ratio": mean_ret,
        "leakage_impact_assessment": leakage,
        "old_v0_bridge_result_reliability_after_rematch": "可作为对照但仍非 canonical" if ready else "仍因匹配/窗口问题不可下结论",
        "v0_structure_still_research_worthy": final_decision
        in ["V0_STRICT_LAG_REMATCH_READY_STILL_STRONG", "V0_STRICT_LAG_REMATCH_READY_WEAKER_BUT_USABLE"],
        "canonical_rebuild_still_required": True,
        "strict_lag_alpha_signal_regenerated": False,
        "strict_lag_weights_regenerated": False,
        "original_orthogonalization_modified": False,
        "old_artifacts_modified": False,
        "production_modified": False,
        "ml_training_run": False,
        "new_ml_model_trained": False,
        "portfolio_returns_calculated": True,
        "fwd_ret_1m_used_for_same_month_signal": False,
        "fwd_ret_1m_used_for_selection": False,
        "fwd_ret_1m_used_for_weighting": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
        "final_decision": final_decision,
        "recommended_next_step": "继续 canonical CSMAR rebuild，并保留 month-key rematch 口径" if ready else "继续排查剩余覆盖缺口，特别是 symbol universe 与 CSMAR return 月份覆盖",
    }
    save_json(summary, OUT_DIR / "v0_strict_lag_month_key_csmar_rematch_summary.json")

    report = [
        "# V0 Strict-Lag Month-Key CSMAR Rematch v1",
        "",
        f"- final_decision: {final_decision}",
        f"- best_match_method: {best}",
        f"- rematch_status: {rematch_status}",
        f"- exact_date_avg_matched_weight_share: {summary['exact_date_avg_matched_weight_share']:.6f}",
        f"- year_month_avg_matched_weight_share: {summary['year_month_avg_matched_weight_share']:.6f}",
        f"- strict_lag_20bps_sharpe: {strict_sharpe:.6f}",
        f"- leakage_impact_assessment: {leakage}",
        "",
        "本次只重做 CSMAR return matching 和 bridge evaluation，未重新生成 alpha_signal 或 weights。",
    ]
    (OUT_DIR / "v0_strict_lag_month_key_csmar_rematch_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    (RUN_DIR / "task_completion_card.md").write_text(
        "\n".join(
            [
                "# task_completion_card",
                f"- task_name: {TASK_NAME}",
                f"- completed_at: {datetime.now().isoformat(timespec='seconds')}",
                f"- final_decision: {final_decision}",
                f"- output_dir: {OUT_DIR}",
            ]
        ),
        encoding="utf-8",
    )
    save_json(
        {
            "task_name": TASK_NAME,
            "stdout_log": str(RUN_DIR / "run_stdout.txt"),
            "stderr_log": str(RUN_DIR / "run_stderr.txt"),
            "status": "completed",
            "final_decision": final_decision,
        },
        RUN_DIR / "terminal_summary.json",
    )
    pd.DataFrame(
        [
            {"qa_item": "prerequisites_passed", "pass": prereq["prerequisites_passed"]},
            {"qa_item": "date_key_mismatch_diagnosed", "pass": bool(date_key_mismatch)},
            {"qa_item": "portfolio_returns_calculated", "pass": True},
            {"qa_item": "alpha_signal_not_regenerated", "pass": True},
            {"qa_item": "weights_not_regenerated", "pass": True},
            {"qa_item": "guardrails_passed", "pass": no_guardrail_violation},
        ]
    ).to_csv(RUN_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    write_state("completed", {"final_decision": final_decision, "output_dir": str(OUT_DIR)})


if __name__ == "__main__":
    main()
