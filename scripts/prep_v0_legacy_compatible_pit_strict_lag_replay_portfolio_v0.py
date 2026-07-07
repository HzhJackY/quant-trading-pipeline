from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


TASK_NAME = "V0 Legacy-Compatible PIT Strict-Lag Replay Portfolio Prep v0"
OUT_NAME = "v0_legacy_compatible_pit_strict_lag_replay_portfolio_prep_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / OUT_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

ALPHA_BUILD_DIR = ROOT / "output" / "v0_legacy_compatible_pit_strict_lag_replay_alpha_build_v0"
ALPHA_SUMMARY = ALPHA_BUILD_DIR / "v0_legacy_compatible_pit_strict_lag_replay_alpha_build_summary.json"
ALPHA_PANEL = ALPHA_BUILD_DIR / "v0_legacy_pit_route_b_strict_lag_alpha_panel.parquet"
LEAKAGE_QA = ALPHA_BUILD_DIR / "v0_route_b_strict_lag_leakage_qa.csv"
COVERAGE_QA = ALPHA_BUILD_DIR / "v0_route_b_alpha_coverage_qa.csv"
OVERLAP_SUMMARY = ALPHA_BUILD_DIR / "v0_route_b_alpha_overlap_summary.csv"
FACTOR_SPLIT_QA = ALPHA_BUILD_DIR / "v0_route_b_factor_split_compatibility_qa.csv"
ICIR_SUMMARY = ALPHA_BUILD_DIR / "v0_route_b_icir_weight_path_summary.csv"

RETURN_MAP = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0" / "canonical_csmar_trd_mnth_return_map_repaired.parquet"
LEGACY_ALPHA = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_alpha_signal_panel.parquet"
LEGACY_WEIGHTS = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_reconstructed_weights.parquet"
COMPOSITE_ALPHA = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_composite_aligned_alpha_candidate_panel.parquet"
COMPOSITE_WEIGHTS = ROOT / "output" / "v0_composite_aligned_portfolio_construction_run_v0" / "v0_composite_aligned_research_weights.parquet"
ADAPTER = ROOT / "output" / "v0_legacy_compatible_pit_adapter_replay_dry_run_v0" / "v0_pit_legacy_compatible_input.parquet"

SCORE_COL = "alpha_signal_route_b_strict_lag"
PORTFOLIO_NAME = "V0_LEGACY_COMPATIBLE_PIT_STRICT_LAG_TOP50_BUFFER_35_75_EQUAL_WEIGHT"
TARGET_HOLDING_COUNT = 50
ENTRY_RANK = 35
EXIT_RANK = 75


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_state(status: str, checkpoint: str, extra: dict[str, Any] | None = None) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": TASK_NAME,
        "status": status,
        "checkpoint": checkpoint,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "resume_instruction": f"先读取 {rel(RUN_DIR / 'RUN_STATE.md')}；继续时运行 scripts\\prep_v0_legacy_compatible_pit_strict_lag_replay_portfolio_v0.py，并重定向 stdout/stderr 到本目录。",
    }
    if extra:
        payload.update(extra)
    lines = [
        "# RUN_STATE", "", f"- task_name: {TASK_NAME}", f"- status: {status}",
        f"- checkpoint: {checkpoint}", "", "```json",
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), "```",
    ]
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(lines), encoding="utf-8")


def prereq_check() -> dict[str, Any]:
    flags = {
        "route_b_alpha_summary_found": ALPHA_SUMMARY.exists(),
        "route_b_alpha_panel_found": ALPHA_PANEL.exists(),
        "route_b_leakage_qa_found": LEAKAGE_QA.exists(),
        "route_b_coverage_qa_found": COVERAGE_QA.exists(),
        "route_b_factor_split_qa_found": FACTOR_SPLIT_QA.exists(),
        "trd_mnth_return_map_found": RETURN_MAP.exists(),
        "legacy_alpha_found": LEGACY_ALPHA.exists(),
        "legacy_weights_found": LEGACY_WEIGHTS.exists(),
        "composite_aligned_alpha_found": COMPOSITE_ALPHA.exists(),
        "composite_aligned_weights_found": COMPOSITE_WEIGHTS.exists(),
    }
    paths = {
        "route_b_alpha_summary_found": ALPHA_SUMMARY,
        "route_b_alpha_panel_found": ALPHA_PANEL,
        "route_b_leakage_qa_found": LEAKAGE_QA,
        "route_b_coverage_qa_found": COVERAGE_QA,
        "route_b_factor_split_qa_found": FACTOR_SPLIT_QA,
        "trd_mnth_return_map_found": RETURN_MAP,
    }
    optional = {
        "legacy_alpha_found": LEGACY_ALPHA,
        "legacy_weights_found": LEGACY_WEIGHTS,
        "composite_aligned_alpha_found": COMPOSITE_ALPHA,
        "composite_aligned_weights_found": COMPOSITE_WEIGHTS,
    }
    missing_required = [rel(p) for k, p in paths.items() if not flags[k]]
    missing_optional = [rel(p) for k, p in optional.items() if not flags[k]]
    flags["prerequisites_passed"] = len(missing_required) == 0
    flags["missing_files"] = missing_required + missing_optional
    flags["caveat"] = "legacy/composite comparison artifacts are optional for planning; no weights or returns are generated in this prep."
    return flags


def load_alpha() -> pd.DataFrame:
    cols = [
        "symbol_norm", "year_month", "month_end", "split_group", SCORE_COL,
        "factor_count_used", "alpha_build_status", "leakage_policy",
    ]
    alpha = pd.read_parquet(ALPHA_PANEL, columns=cols)
    alpha["symbol_norm"] = alpha["symbol_norm"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)
    alpha["year_month"] = alpha["year_month"].astype(str).str.slice(0, 7)
    alpha["month_end"] = pd.to_datetime(alpha["month_end"], errors="coerce")
    alpha[SCORE_COL] = pd.to_numeric(alpha[SCORE_COL], errors="coerce")
    return alpha


def alpha_input_qa(alpha: pd.DataFrame) -> pd.DataFrame:
    leakage = pd.read_csv(LEAKAGE_QA)
    current_future_ok = bool(leakage["pass"].astype(str).str.lower().isin(["true"]).all())
    status_dist = alpha["alpha_build_status"].value_counts(dropna=False).to_dict()
    rows = [
        ("row_count", ">0", int(len(alpha)), len(alpha) > 0, ""),
        ("unique_symbol_count", ">0", int(alpha["symbol_norm"].nunique()), alpha["symbol_norm"].nunique() > 0, ""),
        ("month_count", ">0", int(alpha["year_month"].nunique()), alpha["year_month"].nunique() > 0, ""),
        ("min_year_month", "non-empty", str(alpha["year_month"].min()), pd.notna(alpha["year_month"].min()), ""),
        ("max_year_month", "non-empty", str(alpha["year_month"].max()), pd.notna(alpha["year_month"].max()), ""),
        ("alpha_non_null_ratio", ">=0.80", float(alpha[SCORE_COL].notna().mean()), alpha[SCORE_COL].notna().mean() >= 0.80, ""),
        ("duplicate symbol-month count", 0, int(alpha.duplicated(["symbol_norm", "year_month"]).sum()), int(alpha.duplicated(["symbol_norm", "year_month"]).sum()) == 0, ""),
        ("score column coverage", SCORE_COL, SCORE_COL in alpha.columns, SCORE_COL in alpha.columns, ""),
        ("split_group coverage", "large/small", float(alpha["split_group"].isin(["large", "small"]).mean()), alpha["split_group"].isin(["large", "small"]).mean() > 0.95, ""),
        ("alpha_build_status distribution", "tracked", json.dumps({str(k): int(v) for k, v in status_dist.items()}, ensure_ascii=False), True, ""),
        ("no current/future IC leakage from previous QA", True, current_future_ok, current_future_ok, ""),
        ("route_a_no_label_fallback_used_for_route_b", False, False, True, ""),
    ]
    return pd.DataFrame(rows, columns=["check_name", "expected", "actual", "pass", "caveat"])


def eligible_month_policy(alpha: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for ym, g in alpha.groupby("year_month", sort=True):
        non_null = g[SCORE_COL].notna()
        ratio = float(non_null.mean())
        available = int(non_null.sum())
        selected_possible = available
        warmup = bool(g["alpha_build_status"].astype(str).str.contains("WARMUP", na=False).any())
        if available == 0:
            status = "FAIL_NO_ALPHA"
        elif selected_possible < EXIT_RANK:
            status = "FAIL_LOW_COVERAGE"
        elif ratio >= 0.95 and not warmup:
            status = "READY"
        elif ratio >= 0.80:
            status = "WATCH"
        else:
            status = "FAIL_LOW_COVERAGE"
        rows.append({
            "year_month": ym,
            "alpha_non_null_ratio": ratio,
            "available_symbol_count": available,
            "selected_count_possible": selected_possible,
            "eligible_month_status": status,
            "include_in_construction": status in {"READY", "WATCH"},
            "watch_flag": status == "WATCH",
            "caveat": "early warmup/no alpha caveat" if warmup or status.startswith("FAIL") else "",
        })
    policy = pd.DataFrame(rows)
    included = policy.loc[policy["include_in_construction"]]
    excluded = policy.loc[~policy["include_in_construction"]]
    summary = pd.DataFrame([{
        "total_month_count": int(len(policy)),
        "ready_month_count": int((policy["eligible_month_status"] == "READY").sum()),
        "watch_month_count": int((policy["eligible_month_status"] == "WATCH").sum()),
        "fail_month_count": int(policy["eligible_month_status"].astype(str).str.startswith("FAIL").sum()),
        "first_eligible_month": str(included["year_month"].min()) if len(included) else "",
        "last_eligible_month": str(included["year_month"].max()) if len(included) else "",
        "excluded_months": ",".join(excluded["year_month"].astype(str).tolist()),
        "caveat": "WATCH/fail months retained in policy but next run should include only include_in_construction=true.",
    }])
    return policy, summary


def construction_policy() -> dict[str, Any]:
    return {
        "portfolio_name": PORTFOLIO_NAME,
        "score_column": SCORE_COL,
        "higher_is_better": True,
        "target_holding_count": TARGET_HOLDING_COUNT,
        "entry_rank": ENTRY_RANK,
        "exit_rank": EXIT_RANK,
        "weighting_scheme": "equal_weight",
        "tie_breaker": "symbol_norm ascending",
        "first_month_policy": "select top50 by score among eligible rows",
        "rebalance_frequency": "monthly",
        "buffer_rule": {
            "keep_previous_holdings_if_current_rank_lte": EXIT_RANK,
            "exit_if_rank_gt": EXIT_RANK,
            "add_new_names_from_rank_lte": ENTRY_RANK,
            "fill_to_target_by_next_best_rank": True,
        },
        "long_only": True,
        "shorting_allowed": False,
        "cash_modeled_at_construction_stage": False,
        "cost_calculation_allowed": False,
        "returns_calculation_allowed": False,
        "production_allowed": False,
        "caveat": "This prep locks policy only; it does not generate weights or calculate returns.",
    }


def future_eval_coverage(alpha: pd.DataFrame, month_policy: pd.DataFrame) -> pd.DataFrame:
    ret = pd.read_parquet(RETURN_MAP, columns=["symbol_norm", "year_month", "fwd_ret_1m", "primary_return_field"])
    ret = ret.loc[ret["primary_return_field"].astype(str).eq("Mretwd")].copy()
    ret["symbol_norm"] = ret["symbol_norm"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)
    ret["year_month"] = ret["year_month"].astype(str).str.slice(0, 7)
    ret["fwd_ret_1m"] = pd.to_numeric(ret["fwd_ret_1m"], errors="coerce")
    ret = ret.drop_duplicates(["symbol_norm", "year_month"], keep="last")
    rows = []
    for row in month_policy.itertuples(index=False):
        ym = str(row.year_month)
        include = bool(row.include_in_construction)
        symbols = alpha.loc[alpha["year_month"].eq(ym) & alpha[SCORE_COL].notna(), ["symbol_norm"]].drop_duplicates()
        merged = symbols.merge(ret.loc[ret["year_month"].eq(ym), ["symbol_norm", "fwd_ret_1m"]], on="symbol_norm", how="left")
        matched = int(merged["fwd_ret_1m"].notna().sum())
        expected = include and matched >= min(TARGET_HOLDING_COUNT, len(symbols))
        status = "AVAILABLE" if expected else ("NO_CONSTRUCTION" if not include else "NO_OR_LOW_FUTURE_LABEL")
        rows.append({
            "year_month": ym,
            "include_in_construction": include,
            "expected_future_label_available": expected,
            "matched_label_symbol_count_preview": matched,
            "coverage_status": status,
            "caveat": "evaluation should exclude or wait for label" if include and not expected else "",
        })
    del ret
    gc.collect()
    return pd.DataFrame(rows)


def comparison_plan() -> pd.DataFrame:
    rows = [
        {
            "comparison_target": "legacy_strict_lag",
            "target_alpha_path": rel(LEGACY_ALPHA),
            "target_weights_path": rel(LEGACY_WEIGHTS),
            "available": LEGACY_ALPHA.exists() and LEGACY_WEIGHTS.exists(),
            "planned_metric_next_run": "alpha Spearman; Top50 overlap; weights overlap; turnover proxy; selected count QA",
            "caveat": "read-only comparison target",
        },
        {
            "comparison_target": "composite_aligned",
            "target_alpha_path": rel(COMPOSITE_ALPHA),
            "target_weights_path": rel(COMPOSITE_WEIGHTS),
            "available": COMPOSITE_ALPHA.exists() and COMPOSITE_WEIGHTS.exists(),
            "planned_metric_next_run": "alpha Spearman; Top50 overlap; weights overlap; turnover proxy; selected count QA",
            "caveat": "read-only comparison target",
        },
    ]
    return pd.DataFrame(rows)


def next_run_config() -> dict[str, Any]:
    return {
        "recommended_next_run": "V0 Legacy-Compatible PIT Strict-Lag Replay Portfolio Construction Run v0",
        "recommended_next_run_reason": "Route B alpha QA passed and eligible month policy/policy lock are available; next run may generate weights and weight QA only.",
        "route_b_alpha_path": rel(ALPHA_PANEL),
        "eligible_month_policy_path": rel(OUT_DIR / "v0_route_b_portfolio_eligible_month_policy.csv"),
        "construction_policy_path": rel(OUT_DIR / "v0_route_b_portfolio_construction_policy.json"),
        "future_eval_coverage_plan_path": rel(OUT_DIR / "v0_route_b_future_eval_coverage_plan.csv"),
        "output_weights_path_next": "output/v0_legacy_compatible_pit_strict_lag_replay_portfolio_construction_run_v0/v0_route_b_strict_lag_research_weights.parquet",
        "output_monthly_weight_qa_path_next": "output/v0_legacy_compatible_pit_strict_lag_replay_portfolio_construction_run_v0/v0_route_b_strict_lag_weight_monthly_qa.csv",
        "generate_weights_next_run_allowed": True,
        "calculate_returns_next_run_allowed": False,
        "calculate_transaction_cost_next_run_allowed": False,
        "calculate_sharpe_next_run_allowed": False,
        "benchmark_relative_allowed": False,
        "production_allowed": False,
        "expected_outputs": ["weights parquet", "weights sample", "monthly weight QA", "buffer transition QA", "comparison planning QA", "guardrail QA"],
    }


def guardrails() -> pd.DataFrame:
    actuals = {
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "transaction_cost_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "tstat_calculated": False,
        "benchmark_relative_returns_calculated": False,
        "active_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "ir_te_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "ml_training_run": False,
        "tuning_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "old_artifacts_modified": False,
        "route_b_alpha_modified": False,
    }
    return pd.DataFrame([
        {"guardrail": k, "expected": False, "actual": v, "pass": v is False}
        for k, v in actuals.items()
    ])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", "prerequisite_check")
    prereq = prereq_check()
    write_json(OUT_DIR / "v0_route_b_portfolio_prep_prerequisite_check.json", prereq)
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", "alpha_input_qa")
    alpha = load_alpha()
    input_qa = alpha_input_qa(alpha)
    input_qa.to_csv(OUT_DIR / "v0_route_b_alpha_input_qa.csv", index=False, encoding="utf-8-sig")
    input_qa_pass = bool(input_qa["pass"].all())

    write_state("running", "eligible_policy")
    month_policy, month_summary = eligible_month_policy(alpha)
    month_policy.to_csv(OUT_DIR / "v0_route_b_portfolio_eligible_month_policy.csv", index=False, encoding="utf-8-sig")
    month_summary.to_csv(OUT_DIR / "v0_route_b_portfolio_eligible_month_summary.csv", index=False, encoding="utf-8-sig")

    policy = construction_policy()
    write_json(OUT_DIR / "v0_route_b_portfolio_construction_policy.json", policy)

    write_state("running", "future_coverage_and_configs")
    future = future_eval_coverage(alpha, month_policy)
    future.to_csv(OUT_DIR / "v0_route_b_future_eval_coverage_plan.csv", index=False, encoding="utf-8-sig")
    comp_plan = comparison_plan()
    comp_plan.to_csv(OUT_DIR / "v0_route_b_comparison_plan.csv", index=False, encoding="utf-8-sig")
    cfg = next_run_config()
    write_json(OUT_DIR / "v0_route_b_portfolio_construction_run_config_draft.json", cfg)

    guard = guardrails()
    guard.to_csv(OUT_DIR / "v0_route_b_portfolio_prep_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    guard_pass = bool(guard["pass"].all())

    ms = month_summary.iloc[0].to_dict()
    eligible_exists = int(ms["ready_month_count"]) + int(ms["watch_month_count"]) > 0
    expected_no_label_months = future.loc[
        future["include_in_construction"] & ~future["expected_future_label_available"],
        "year_month",
    ].astype(str).tolist()
    caveats_exist = int(ms["watch_month_count"]) > 0 or bool(expected_no_label_months)
    if not guard_pass:
        final_decision = "ROUTE_B_PORTFOLIO_PREP_FAIL_GUARDRAIL"
    elif not input_qa_pass or not eligible_exists:
        final_decision = "ROUTE_B_PORTFOLIO_PREP_BLOCKED_BY_ALPHA_COVERAGE"
    elif caveats_exist:
        final_decision = "ROUTE_B_PORTFOLIO_PREP_READY_WITH_CAVEATS"
    else:
        final_decision = "ROUTE_B_PORTFOLIO_PREP_READY_FOR_CONSTRUCTION"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "route_b_alpha_path": rel(ALPHA_PANEL),
        "score_column_selected": SCORE_COL,
        "alpha_input_qa_pass": input_qa_pass,
        "total_month_count": int(ms["total_month_count"]),
        "ready_month_count": int(ms["ready_month_count"]),
        "watch_month_count": int(ms["watch_month_count"]),
        "fail_month_count": int(ms["fail_month_count"]),
        "first_eligible_month": str(ms["first_eligible_month"]),
        "last_eligible_month": str(ms["last_eligible_month"]),
        "construction_policy_locked": True,
        "portfolio_name": PORTFOLIO_NAME,
        "target_holding_count": TARGET_HOLDING_COUNT,
        "entry_rank": ENTRY_RANK,
        "exit_rank": EXIT_RANK,
        "weighting_scheme": "equal_weight",
        "future_eval_coverage_planned": True,
        "expected_no_label_months": expected_no_label_months,
        "comparison_plan_generated": True,
        "generate_weights_next_run_allowed": True,
        "calculate_returns_next_run_allowed": False,
        "benchmark_relative_allowed": False,
        "production_allowed": False,
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "transaction_cost_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "tstat_calculated": False,
        "benchmark_relative_returns_calculated": False,
        "active_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "ir_te_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "ml_training_run": False,
        "tuning_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "old_artifacts_modified": False,
        "route_b_alpha_modified": False,
        "guardrails_passed": guard_pass,
        "final_decision": final_decision,
        "recommended_next_step": "运行 V0 Legacy-Compatible PIT Strict-Lag Replay Portfolio Construction Run v0；只生成 weights 与 weight QA，不计算收益或绩效。",
    }
    write_json(OUT_DIR / "v0_legacy_compatible_pit_strict_lag_replay_portfolio_prep_summary.json", summary)

    report = "\n".join([
        "# V0 Legacy-Compatible PIT Strict-Lag Replay Portfolio Prep v0",
        "",
        f"- final_decision: {final_decision}",
        f"- score_column: {SCORE_COL}",
        f"- eligible months ready/watch/fail: {summary['ready_month_count']}/{summary['watch_month_count']}/{summary['fail_month_count']}",
        f"- first/last eligible month: {summary['first_eligible_month']} / {summary['last_eligible_month']}",
        f"- portfolio_name: {PORTFOLIO_NAME}",
        "",
        "本任务只锁定 portfolio prep、eligible month policy、Top50 Buffer 35/75 policy 和未来 evaluation coverage plan；未生成 weights，未计算任何收益或绩效指标。",
    ])
    (OUT_DIR / "v0_legacy_compatible_pit_strict_lag_replay_portfolio_prep_report.md").write_text(report, encoding="utf-8")

    final_qa = pd.DataFrame([
        {"check_name": "prerequisites_passed", "expected": True, "actual": prereq["prerequisites_passed"], "pass": prereq["prerequisites_passed"], "caveat": ""},
        {"check_name": "alpha_input_qa_pass", "expected": True, "actual": input_qa_pass, "pass": input_qa_pass, "caveat": ""},
        {"check_name": "eligible_months_exist", "expected": True, "actual": eligible_exists, "pass": eligible_exists, "caveat": ""},
        {"check_name": "guardrails_passed", "expected": True, "actual": guard_pass, "pass": guard_pass, "caveat": ""},
    ])
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    write_json(OUT_DIR / "terminal_summary.json", {
        "task_name": TASK_NAME,
        "status": "completed",
        "stdout_log": rel(RUN_DIR / "run_stdout.txt"),
        "stderr_log": rel(RUN_DIR / "run_stderr.txt"),
        "output_dir": rel(OUT_DIR),
        "final_decision": final_decision,
    })
    (OUT_DIR / "task_completion_card.md").write_text(
        "\n".join(["# task_completion_card", "", f"- task_name: {TASK_NAME}", "- status: completed", f"- final_decision: {final_decision}", f"- output_dir: {rel(OUT_DIR)}"]),
        encoding="utf-8",
    )
    write_state("completed", "all_outputs_written", {"final_decision": final_decision, "output_dir": rel(OUT_DIR)})
    del alpha, input_qa, month_policy, month_summary, future, comp_plan, guard
    gc.collect()
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": rel(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
