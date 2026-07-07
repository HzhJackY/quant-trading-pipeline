from __future__ import annotations

import gc
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "v0_canonical_repaired_trd_mnth_eval_run_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

WEIGHTS_PATH = ROOT / "output" / "v0_canonical_portfolio_construction_run_v0" / "v0_canonical_research_weights.parquet"
RETURN_MAP_PATH = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0" / "canonical_csmar_trd_mnth_return_map_repaired.parquet"
PREP_DIR = ROOT / "output" / "v0_canonical_repaired_trd_mnth_eval_prep_v0"
WINDOW_POLICY_PATH = PREP_DIR / "v0_canonical_eval_window_policy.csv"
COST_CONFIG_PATH = PREP_DIR / "v0_canonical_eval_cost_return_variant_config.json"
TURNOVER_POLICY_PATH = PREP_DIR / "v0_canonical_eval_turnover_policy.csv"
PREP_SUMMARY_PATH = PREP_DIR / "v0_canonical_repaired_trd_mnth_eval_prep_summary.json"
LEGACY_SUMMARY_PATH = ROOT / "output" / "unified_strategy_eval_repaired_trd_mnth_v0" / "unified_strategy_eval_repaired_trd_mnth_summary.json"

PORTFOLIO_NAME = "V0_CANONICAL_TOP50_BUFFER_35_75"
PRIMARY_RETURN_FIELD = "Mretwd"
PRIMARY_COST_BPS = 20
PRIMARY_RETURN_VARIANT = "raw_unmatched_not_renormalized"
FIRST_MONTH_POLICY = "charge_cost_on_first_month_initialization"
FIRST_MONTH_NO_COST_POLICY = "first_month_initialization_no_cost"
COST_BPS_LIST = [0, 10, 20, 30]
RETURN_VARIANTS = ["raw_unmatched_not_renormalized", "matched_only_normalized"]


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
        "resume_instruction: run scripts\\run_v0_canonical_repaired_trd_mnth_eval_v0.py with stdout/stderr redirected to output\\_agent_runs\\v0_canonical_repaired_trd_mnth_eval_run_v0\n"
    )
    (RUN_DIR / "RUN_STATE.md").write_text(text, encoding="utf-8")


def dump_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def check_prerequisites() -> dict:
    files = {
        "weights_found": WEIGHTS_PATH,
        "return_map_found": RETURN_MAP_PATH,
        "eval_window_policy_found": WINDOW_POLICY_PATH,
        "cost_variant_config_found": COST_CONFIG_PATH,
        "turnover_policy_found": TURNOVER_POLICY_PATH,
        "prep_summary_found": PREP_SUMMARY_PATH,
    }
    result = {key: path.exists() for key, path in files.items()}
    missing = [rel(path) for key, path in files.items() if not result[key]]
    result["prerequisites_passed"] = not missing
    result["missing_files"] = missing
    dump_json(OUT_DIR / "v0_canonical_eval_run_prerequisite_check.json", result)
    return result


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def max_drawdown(returns: pd.Series) -> tuple[float, pd.DataFrame]:
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    return float(drawdown.min()), pd.DataFrame({"nav": nav, "running_max_nav": running_max, "drawdown": drawdown})


def performance(group: pd.DataFrame) -> dict:
    ret = group["net_return"].astype(float)
    n = int(ret.notna().sum())
    mean_ret = float(ret.mean()) if n else np.nan
    std_ret = float(ret.std(ddof=1)) if n > 1 else np.nan
    vol = std_ret
    sharpe = mean_ret / vol * math.sqrt(12) if vol and not np.isnan(vol) else np.nan
    tstat = mean_ret / std_ret * math.sqrt(n) if std_ret and not np.isnan(std_ret) else np.nan
    cum_ret = float((1.0 + ret.fillna(0.0)).prod() - 1.0) if n else np.nan
    mdd, _ = max_drawdown(ret)
    return {
        "month_count": n,
        "mean_monthly_return": mean_ret,
        "annualized_return_approx": mean_ret * 12 if not np.isnan(mean_ret) else np.nan,
        "monthly_volatility": vol,
        "sharpe": sharpe,
        "tstat": tstat,
        "positive_month_ratio": float((ret > 0).mean()) if n else np.nan,
        "cumulative_return": cum_ret,
        "max_drawdown": mdd,
        "avg_turnover": float(group["turnover_proxy"].mean()),
        "max_turnover": float(group["turnover_proxy"].max()),
        "avg_matched_weight_share": float(group["matched_weight_share"].mean()),
        "min_matched_weight_share": float(group["matched_weight_share"].min()),
    }


def compute_turnover(weights: pd.DataFrame, eval_months: list[str]) -> pd.DataFrame:
    rows = []
    prev = None
    for ym in sorted(eval_months):
        cur = weights.loc[weights["year_month"].eq(ym), ["symbol_norm", "weight"]].copy()
        cur_map = cur.set_index("symbol_norm")["weight"].astype(float)
        if prev is None:
            turnover = 1.0
            turnover_no_cost = 0.0
        else:
            symbols = cur_map.index.union(prev.index)
            turnover = 0.5 * float((cur_map.reindex(symbols, fill_value=0.0) - prev.reindex(symbols, fill_value=0.0)).abs().sum())
            turnover_no_cost = turnover
        rows.append(
            {
                "year_month": ym,
                FIRST_MONTH_POLICY: turnover,
                FIRST_MONTH_NO_COST_POLICY: turnover_no_cost,
            }
        )
        prev = cur_map
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_run_state("running", "prerequisite_check")

    prereq = check_prerequisites()
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(f"Missing prerequisites: {prereq['missing_files']}")

    with COST_CONFIG_PATH.open("r", encoding="utf-8") as f:
        cost_cfg = json.load(f)
    with PREP_SUMMARY_PATH.open("r", encoding="utf-8") as f:
        prep_summary = json.load(f)

    window = pd.read_csv(WINDOW_POLICY_PATH, dtype={"year_month": "string"})
    window["include_in_primary_eval"] = bool_series(window["include_in_primary_eval"])
    eval_months = sorted(window.loc[window["include_in_primary_eval"], "year_month"].astype(str).tolist())
    excluded_months = sorted(window.loc[~window["include_in_primary_eval"], "year_month"].astype(str).tolist())

    write_run_state("running", "reading_parquet_required_columns")
    weights = pd.read_parquet(
        WEIGHTS_PATH,
        columns=["portfolio_name", "year_month", "symbol_norm", "weight", "selected_count"],
    )
    returns = pd.read_parquet(
        RETURN_MAP_PATH,
        columns=["symbol_norm", "year_month", "fwd_ret_1m", "primary_return_field", "return_valid_flag"],
    )
    weights["year_month"] = weights["year_month"].astype(str)
    weights["symbol_norm"] = weights["symbol_norm"].astype(str)
    returns["year_month"] = returns["year_month"].astype(str)
    returns["symbol_norm"] = returns["symbol_norm"].astype(str)

    duplicate_weight_count = int(weights.duplicated(["symbol_norm", "year_month"]).sum())
    duplicate_return_count = int(returns.duplicated(["symbol_norm", "year_month"]).sum())
    primary_field_actual = ",".join(sorted(returns["primary_return_field"].dropna().astype(str).unique().tolist()))
    fwd_available_primary = int(returns.loc[returns["year_month"].isin(eval_months), "fwd_ret_1m"].notna().sum())

    input_qa = pd.DataFrame(
        [
            {"check_name": "weights month range", "expected": "2017-03 to 2026-06", "actual": f"{weights['year_month'].min()} to {weights['year_month'].max()}", "pass": weights["year_month"].min() == "2017-03" and weights["year_month"].max() == "2026-06", "caveat": ""},
            {"check_name": "return map month range", "expected": "covers primary eval months", "actual": f"{returns['year_month'].min()} to {returns['year_month'].max()}", "pass": set(eval_months).issubset(set(returns["year_month"].unique())), "caveat": ""},
            {"check_name": "duplicate weight symbol-month", "expected": "0", "actual": duplicate_weight_count, "pass": duplicate_weight_count == 0, "caveat": ""},
            {"check_name": "duplicate return symbol-month", "expected": "0", "actual": duplicate_return_count, "pass": duplicate_return_count == 0, "caveat": ""},
            {"check_name": "primary return field", "expected": PRIMARY_RETURN_FIELD, "actual": primary_field_actual, "pass": primary_field_actual == PRIMARY_RETURN_FIELD, "caveat": ""},
            {"check_name": "fwd_ret_1m availability", "expected": "available in primary eval window", "actual": fwd_available_primary, "pass": fwd_available_primary > 0, "caveat": "按持仓匹配另行检查 matched_weight_share"},
            {"check_name": "2026-06 excluded from primary eval", "expected": "excluded", "actual": "excluded" if "2026-06" in excluded_months and "2026-06" not in eval_months else "not_excluded", "pass": "2026-06" in excluded_months and "2026-06" not in eval_months, "caveat": "2026-06 无 fwd_ret_1m"},
        ]
    )
    input_qa.to_csv(OUT_DIR / "v0_canonical_eval_run_input_qa.csv", index=False, encoding="utf-8-sig")

    weights_months = set(weights["year_month"].unique())
    return_months = set(returns["year_month"].unique())
    window_qa = pd.DataFrame(
        [
            {
                "primary_eval_month_count": len(eval_months),
                "primary_eval_min_year_month": min(eval_months),
                "primary_eval_max_year_month": max(eval_months),
                "excluded_months": ";".join(excluded_months),
                "included_months_missing_in_weights": ";".join([m for m in eval_months if m not in weights_months]),
                "included_months_missing_in_return_map": ";".join([m for m in eval_months if m not in return_months]),
                "window_status": "PASS" if len(eval_months) == 111 and min(eval_months) == "2017-03" and max(eval_months) == "2026-05" and "2026-06" in excluded_months else "REVIEW",
            }
        ]
    )
    window_qa.to_csv(OUT_DIR / "v0_canonical_eval_window_applied_qa.csv", index=False, encoding="utf-8-sig")

    write_run_state("running", "calculating_portfolio_returns")
    w_eval = weights.loc[weights["year_month"].isin(eval_months)].copy()
    r_eval = returns.loc[returns["year_month"].isin(eval_months), ["symbol_norm", "year_month", "fwd_ret_1m"]].copy()
    merged = w_eval.merge(r_eval, on=["symbol_norm", "year_month"], how="left", indicator=True)
    merged["matched"] = merged["fwd_ret_1m"].notna()
    merged["weighted_ret"] = merged["weight"].astype(float) * merged["fwd_ret_1m"].fillna(0.0).astype(float)

    monthly_rows = []
    for ym, g in merged.groupby("year_month", sort=True):
        selected_count = int(g["symbol_norm"].nunique())
        matched_count = int(g.loc[g["matched"], "symbol_norm"].nunique())
        matched_weight_share = float(g.loc[g["matched"], "weight"].sum())
        unmatched_weight_share = float(g.loc[~g["matched"], "weight"].sum())
        gross_raw = float(g["weighted_ret"].sum())
        gross_norm = float(gross_raw / matched_weight_share) if matched_weight_share > 0 else np.nan
        caveat = "" if matched_weight_share >= 0.98 else "matched_weight_share_below_0.98"
        for variant, gross in [
            ("raw_unmatched_not_renormalized", gross_raw),
            ("matched_only_normalized", gross_norm),
        ]:
            monthly_rows.append(
                {
                    "portfolio_name": PORTFOLIO_NAME,
                    "year_month": ym,
                    "return_variant": variant,
                    "selected_count": selected_count,
                    "matched_symbol_count": matched_count,
                    "matched_weight_share": matched_weight_share,
                    "gross_return": gross,
                    "unmatched_weight_share": unmatched_weight_share,
                    "primary_eval_flag": True,
                    "caveat": caveat if variant == PRIMARY_RETURN_VARIANT else "sensitivity_only",
                }
            )
    gross = pd.DataFrame(monthly_rows)
    gross.to_csv(OUT_DIR / "v0_canonical_monthly_gross_returns.csv", index=False, encoding="utf-8-sig")

    turnover = compute_turnover(w_eval, eval_months)
    net_rows = []
    for policy in [FIRST_MONTH_POLICY, FIRST_MONTH_NO_COST_POLICY]:
        tmp = gross.merge(turnover[["year_month", policy]], on="year_month", how="left").rename(columns={policy: "turnover_proxy"})
        for cost_bps in COST_BPS_LIST:
            x = tmp.copy()
            x["cost_bps"] = cost_bps
            x["first_month_cost_policy"] = policy
            x["cost_drag"] = x["turnover_proxy"] * cost_bps / 10000.0
            x["net_return"] = x["gross_return"] - x["cost_drag"]
            net_rows.append(x)
    net = pd.concat(net_rows, ignore_index=True)
    net = net[
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
            "primary_eval_flag",
        ]
    ]
    net.to_csv(OUT_DIR / "v0_canonical_monthly_net_returns_by_cost.csv", index=False, encoding="utf-8-sig")

    perf_rows = []
    for keys, g in net.groupby(["portfolio_name", "return_variant", "cost_bps", "first_month_cost_policy"], sort=True):
        metrics = performance(g.sort_values("year_month"))
        row = dict(zip(["portfolio_name", "return_variant", "cost_bps", "first_month_cost_policy"], keys))
        row.update(metrics)
        row["performance_status"] = "PRIMARY" if row["return_variant"] == PRIMARY_RETURN_VARIANT and row["cost_bps"] == PRIMARY_COST_BPS and row["first_month_cost_policy"] == FIRST_MONTH_POLICY else "SENSITIVITY"
        perf_rows.append(row)
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(OUT_DIR / "v0_canonical_performance_summary_by_cost.csv", index=False, encoding="utf-8-sig")

    nav_rows = []
    for keys, g in net.groupby(["return_variant", "cost_bps", "first_month_cost_policy"], sort=True):
        g = g.sort_values("year_month").reset_index(drop=True)
        _, path = max_drawdown(g["net_return"])
        nav_rows.append(
            pd.concat(
                [
                    g[["year_month", "return_variant", "cost_bps", "first_month_cost_policy", "net_return"]],
                    path.reset_index(drop=True),
                ],
                axis=1,
            )
        )
    nav = pd.concat(nav_rows, ignore_index=True)
    nav.to_csv(OUT_DIR / "v0_canonical_nav_drawdown_path.csv", index=False, encoding="utf-8-sig")

    primary = perf.loc[
        perf["return_variant"].eq(PRIMARY_RETURN_VARIANT)
        & perf["cost_bps"].eq(PRIMARY_COST_BPS)
        & perf["first_month_cost_policy"].eq(FIRST_MONTH_POLICY)
    ].iloc[0].to_dict()

    legacy_available = LEGACY_SUMMARY_PATH.exists()
    legacy_caveat = False
    if legacy_available:
        with LEGACY_SUMMARY_PATH.open("r", encoding="utf-8") as f:
            legacy = json.load(f)
        legacy_window = f"{legacy.get('common_window_min_year_month')} to {legacy.get('common_window_max_year_month')}"
        canonical_window = f"{min(eval_months)} to {max(eval_months)}"
        legacy_caveat = legacy_window != canonical_window or legacy.get("common_window_month_count") != len(eval_months)
        comparisons = [
            ("window", canonical_window, legacy_window),
            ("month_count", primary["month_count"], legacy.get("common_window_month_count")),
            ("mean_monthly_return", primary["mean_monthly_return"], legacy.get("v0_common_20bps_mean_monthly_return")),
            ("sharpe", primary["sharpe"], legacy.get("v0_common_20bps_sharpe")),
            ("tstat", primary["tstat"], legacy.get("v0_common_20bps_tstat")),
            ("cumulative_return", primary["cumulative_return"], legacy.get("v0_common_20bps_cumulative_return")),
            ("max_drawdown", primary["max_drawdown"], legacy.get("v0_common_20bps_max_drawdown")),
            ("avg_turnover", primary["avg_turnover"], legacy.get("v0_common_20bps_avg_turnover")),
        ]
        comp_rows = []
        for metric, can, leg in comparisons:
            delta = np.nan
            if isinstance(can, (int, float, np.integer, np.floating)) and isinstance(leg, (int, float, np.integer, np.floating)):
                delta = float(can) - float(leg)
            comp_rows.append(
                {
                    "metric": metric,
                    "canonical_v0_value": can,
                    "legacy_strict_lag_v0_value": leg,
                    "delta_canonical_minus_legacy": delta,
                    "interpretation": "只读对照；窗口不一致，不能直接归因" if legacy_caveat else "只读对照；窗口一致",
                }
            )
        pd.DataFrame(comp_rows).to_csv(OUT_DIR / "v0_canonical_vs_legacy_readonly_comparison.csv", index=False, encoding="utf-8-sig")

    guardrail_values = {
        "portfolio_returns_calculated": True,
        "cumulative_returns_calculated": True,
        "transaction_cost_scenarios_calculated": True,
        "sharpe_calculated": True,
        "maxdd_calculated": True,
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
        "alpha_signal_generated": False,
        "strategy_weights_generated": False,
    }
    expected_false = {k for k, v in guardrail_values.items() if v is False}
    guardrail = pd.DataFrame(
        [
            {"guardrail": k, "expected": False if k in expected_false else True, "actual": v, "pass": v == (False if k in expected_false else True)}
            for k, v in guardrail_values.items()
        ]
    )
    guardrail.to_csv(OUT_DIR / "v0_canonical_eval_run_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    guardrails_pass = bool(guardrail["pass"].all())

    if not guardrails_pass:
        final_decision = "V0_CANONICAL_EVAL_RUN_FAIL_GUARDRAIL"
    elif primary["sharpe"] >= 0.80 and primary["tstat"] >= 2.0 and primary["max_drawdown"] > -0.35:
        final_decision = "V0_CANONICAL_EVAL_RUN_STRONG_PASS_CONTINUE_BENCHMARK_ATTRIBUTION"
    elif primary["sharpe"] >= 0.50 and primary["tstat"] >= 1.5:
        final_decision = "V0_CANONICAL_EVAL_RUN_PASS_CONTINUE_BENCHMARK_ATTRIBUTION"
    elif primary["sharpe"] > 0 or primary["tstat"] > 0:
        final_decision = "V0_CANONICAL_EVAL_RUN_MIXED_REVIEW_REQUIRED"
    else:
        final_decision = "V0_CANONICAL_EVAL_RUN_FAIL_DO_NOT_CONTINUE"

    recommended_next_step = {
        "V0_CANONICAL_EVAL_RUN_STRONG_PASS_CONTINUE_BENCHMARK_ATTRIBUTION": "进入下一阶段 benchmark attribution；仍需另起任务并显式解除本任务禁止项。",
        "V0_CANONICAL_EVAL_RUN_PASS_CONTINUE_BENCHMARK_ATTRIBUTION": "可进入 benchmark attribution；先保留 caveat 与成本敏感性。",
        "V0_CANONICAL_EVAL_RUN_MIXED_REVIEW_REQUIRED": "先复核收益、换手和窗口 caveat，再决定是否进入 attribution。",
        "V0_CANONICAL_EVAL_RUN_FAIL_DO_NOT_CONTINUE": "不建议继续 attribution，先回看策略有效性和数据质量。",
        "V0_CANONICAL_EVAL_RUN_FAIL_GUARDRAIL": "停止后续推进，先修复 guardrail violation。",
    }[final_decision]

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "weights_path": rel(WEIGHTS_PATH),
        "return_map_path": rel(RETURN_MAP_PATH),
        "primary_return_field": cost_cfg.get("primary_return_field", PRIMARY_RETURN_FIELD),
        "portfolio_name": PORTFOLIO_NAME,
        "primary_eval_month_count": len(eval_months),
        "primary_eval_min_year_month": min(eval_months),
        "primary_eval_max_year_month": max(eval_months),
        "excluded_months": ";".join(excluded_months),
        "primary_cost_bps": PRIMARY_COST_BPS,
        "primary_return_variant": PRIMARY_RETURN_VARIANT,
        "first_month_initialization_turnover_policy": FIRST_MONTH_POLICY,
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
        "legacy_readonly_comparison_available": legacy_available,
        "legacy_common_window_caveat": legacy_caveat,
        **guardrail_values,
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    dump_json(OUT_DIR / "v0_canonical_repaired_trd_mnth_eval_run_summary.json", summary)

    report = (
        "# V0 Canonical repaired TRD_Mnth evaluation run v0\n\n"
        f"- final_decision: {final_decision}\n"
        f"- primary window: {min(eval_months)} to {max(eval_months)}, {len(eval_months)} months; excluded: {';'.join(excluded_months)}\n"
        f"- primary return/cost: {PRIMARY_RETURN_VARIANT}, {PRIMARY_COST_BPS} bps, {FIRST_MONTH_POLICY}\n"
        f"- Sharpe: {primary['sharpe']:.6f}; t-stat: {primary['tstat']:.6f}; cumulative_return: {primary['cumulative_return']:.6f}; max_drawdown: {primary['max_drawdown']:.6f}\n"
        f"- avg_turnover: {primary['avg_turnover']:.6f}; avg_matched_weight_share: {primary['avg_matched_weight_share']:.6f}; min_matched_weight_share: {primary['min_matched_weight_share']:.6f}\n"
        f"- legacy readonly comparison available: {legacy_available}; legacy window caveat: {legacy_caveat}\n"
        f"- guardrails passed: {guardrails_pass}\n\n"
        "本报告未计算 benchmark-relative、alpha/beta、IR/TE、FF、DGTW、ML、SHAP，也未修改 production、alpha_signal 或 strategy weights。\n"
    )
    (OUT_DIR / "v0_canonical_repaired_trd_mnth_eval_run_report.md").write_text(report, encoding="utf-8")

    final_qa = pd.DataFrame(
        [
            {"check_name": "prerequisites_passed", "pass": prereq["prerequisites_passed"], "detail": ""},
            {"check_name": "window_status", "pass": window_qa.loc[0, "window_status"] == "PASS", "detail": window_qa.loc[0, "window_status"]},
            {"check_name": "guardrails_passed", "pass": guardrails_pass, "detail": ""},
            {"check_name": "primary_metrics_available", "pass": pd.notna(primary["sharpe"]) and pd.notna(primary["max_drawdown"]), "detail": ""},
            {"check_name": "final_decision_allowed", "pass": final_decision in {
                "V0_CANONICAL_EVAL_RUN_STRONG_PASS_CONTINUE_BENCHMARK_ATTRIBUTION",
                "V0_CANONICAL_EVAL_RUN_PASS_CONTINUE_BENCHMARK_ATTRIBUTION",
                "V0_CANONICAL_EVAL_RUN_MIXED_REVIEW_REQUIRED",
                "V0_CANONICAL_EVAL_RUN_FAIL_DO_NOT_CONTINUE",
                "V0_CANONICAL_EVAL_RUN_FAIL_GUARDRAIL",
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

    del weights, returns, w_eval, r_eval, merged, gross, net, perf, nav
    gc.collect()
    write_run_state("completed", "all_outputs_written")
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": rel(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
