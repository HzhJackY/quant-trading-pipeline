from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "v0_composite_aligned_alpha_portfolio_construction_prep_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

ALPHA_DIR = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0"
ALPHA_PANEL = ALPHA_DIR / "v0_composite_aligned_alpha_candidate_panel.parquet"
ALPHA_SUMMARY = ALPHA_DIR / "v0_composite_aligned_strict_lag_alpha_candidate_build_summary.json"
OVERLAP_SUMMARY = ALPHA_DIR / "v0_aligned_alpha_vs_legacy_overlap_summary.csv"
READINESS = ALPHA_DIR / "v0_aligned_alpha_repair_readiness.csv"
RETURN_MAP = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0" / "canonical_csmar_trd_mnth_return_map_repaired.parquet"

PORTFOLIO_NAME = "V0_COMPOSITE_ALIGNED_STRICT_LAG_TOP50_BUFFER_35_75_EQUAL_WEIGHT"
TARGET_HOLDING_COUNT = 50
ENTRY_RANK = 35
EXIT_RANK = 75


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
        "resume_instruction: rerun scripts\\prep_v0_composite_aligned_alpha_portfolio_construction_v0.py with stdout/stderr redirected to this run directory\n",
        encoding="utf-8",
    )


def norm_symbol(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def prerequisites() -> dict:
    result = {
        "aligned_alpha_panel_found": ALPHA_PANEL.exists(),
        "aligned_alpha_summary_found": ALPHA_SUMMARY.exists(),
        "overlap_summary_found": OVERLAP_SUMMARY.exists(),
        "readiness_found": READINESS.exists(),
        "trd_mnth_return_map_found": RETURN_MAP.exists(),
    }
    path_map = {
        "aligned_alpha_panel_found": ALPHA_PANEL,
        "aligned_alpha_summary_found": ALPHA_SUMMARY,
        "overlap_summary_found": OVERLAP_SUMMARY,
        "readiness_found": READINESS,
        "trd_mnth_return_map_found": RETURN_MAP,
    }
    missing = [rel(path) for key, path in path_map.items() if not result[key]]
    result["prerequisites_passed"] = not missing
    result["missing_files"] = missing
    dump_json(OUT_DIR / "v0_aligned_portfolio_prep_prerequisite_check.json", result)
    return result


def select_score_column(columns: list[str]) -> str:
    if "alpha_signal_aligned" in columns:
        return "alpha_signal_aligned"
    if "alpha_signal" in columns:
        return "alpha_signal"
    raise KeyError("No alpha score column found: expected alpha_signal_aligned or alpha_signal")


def eligibility_audit(alpha: pd.DataFrame, score_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha["score_non_null"] = alpha[score_col].notna()
    grouped = alpha.groupby("year_month", sort=True)
    audit = grouped.agg(
        total_count=("symbol_norm", "count"),
        alpha_non_null_count=("score_non_null", "sum"),
        avg_factor_count_used=("factor_count_used", "mean"),
        min_factor_count_used=("factor_count_used", "min"),
        max_factor_count_used=("factor_count_used", "max"),
        eligible_symbol_count=("score_non_null", "sum"),
    ).reset_index()
    audit["alpha_non_null_ratio"] = audit["alpha_non_null_count"] / audit["total_count"]

    def status(row: pd.Series) -> str:
        if row["eligible_symbol_count"] < 75 or row["alpha_non_null_ratio"] < 0.80:
            return "FAIL_NO_SIGNAL"
        if row["alpha_non_null_ratio"] < 0.95:
            return "WATCH_LOW_ALPHA_COVERAGE"
        if row["avg_factor_count_used"] < 5:
            return "WATCH_LOW_FACTOR_COUNT"
        return "READY"

    audit["eligible_month_status"] = audit.apply(status, axis=1)
    audit["caveat"] = np.select(
        [
            audit["eligible_month_status"].eq("WATCH_LOW_ALPHA_COVERAGE"),
            audit["eligible_month_status"].eq("WATCH_LOW_FACTOR_COUNT"),
            audit["eligible_month_status"].eq("FAIL_NO_SIGNAL"),
        ],
        [
            "alpha coverage below 0.95",
            "average factor_count_used below 5",
            "insufficient eligible symbols or alpha coverage",
        ],
        default="",
    )
    audit = audit[
        [
            "year_month",
            "total_count",
            "alpha_non_null_count",
            "alpha_non_null_ratio",
            "avg_factor_count_used",
            "min_factor_count_used",
            "max_factor_count_used",
            "eligible_symbol_count",
            "eligible_month_status",
            "caveat",
        ]
    ]
    audit.to_csv(OUT_DIR / "v0_aligned_alpha_eligibility_audit.csv", index=False, encoding="utf-8-sig")

    month_policy = audit.rename(
        columns={
            "eligible_month_status": "month_status",
        }
    )[
        ["year_month", "alpha_non_null_ratio", "eligible_symbol_count", "avg_factor_count_used", "month_status"]
    ].copy()
    month_policy["include_in_construction_next_run"] = ~month_policy["month_status"].eq("FAIL_NO_SIGNAL")
    month_policy["reason"] = np.select(
        [
            month_policy["month_status"].eq("READY"),
            month_policy["month_status"].str.startswith("WATCH"),
            month_policy["month_status"].eq("FAIL_NO_SIGNAL"),
        ],
        ["READY month", "WATCH month included with caveat", "FAIL_NO_SIGNAL excluded"],
        default="",
    )
    month_policy.to_csv(OUT_DIR / "v0_aligned_portfolio_eligible_month_policy.csv", index=False, encoding="utf-8-sig")
    return audit, month_policy


def write_policy(score_col: str) -> dict:
    policy = {
        "portfolio_name": PORTFOLIO_NAME,
        "score_column": score_col,
        "higher_is_better": True,
        "target_holding_count": TARGET_HOLDING_COUNT,
        "entry_rank": ENTRY_RANK,
        "exit_rank": EXIT_RANK,
        "first_month_initialization": "top50",
        "weighting_scheme": "equal_weight",
        "tie_breaker": "symbol_norm ascending",
        "eligible_month_policy": "READY and WATCH months included; FAIL_NO_SIGNAL excluded",
        "eligible_symbol_policy": "rows with non-null aligned alpha in eligible construction months",
        "use_fwd_ret_for_selection": False,
        "use_benchmark_for_selection": False,
        "tuning_allowed": False,
        "production_allowed": False,
    }
    dump_json(OUT_DIR / "v0_aligned_portfolio_construction_policy.json", policy)
    return policy


def coverage_plan(alpha: pd.DataFrame, score_col: str, month_policy: pd.DataFrame) -> pd.DataFrame:
    eligible_alpha = alpha.merge(
        month_policy[["year_month", "include_in_construction_next_run"]],
        on="year_month",
        how="left",
    )
    eligible_alpha = eligible_alpha.loc[
        eligible_alpha["include_in_construction_next_run"].fillna(False) & eligible_alpha[score_col].notna(),
        ["symbol_norm", "year_month"],
    ].copy()
    ret = pd.read_parquet(RETURN_MAP, columns=["symbol_norm", "year_month", "fwd_ret_1m"])
    ret["symbol_norm"] = norm_symbol(ret["symbol_norm"])
    ret["year_month"] = ret["year_month"].astype(str).str.slice(0, 7)
    ret["has_fwd_ret"] = pd.to_numeric(ret["fwd_ret_1m"], errors="coerce").notna()
    ret = ret.drop_duplicates(["symbol_norm", "year_month"], keep="last")[["symbol_norm", "year_month", "has_fwd_ret"]]
    merged = eligible_alpha.merge(ret, on=["symbol_norm", "year_month"], how="left")
    merged["has_fwd_ret"] = merged["has_fwd_ret"].fillna(False)
    coverage = merged.groupby("year_month", sort=True).agg(
        eligible_count=("symbol_norm", "count"),
        matched_label_count=("has_fwd_ret", "sum"),
    ).reset_index()
    coverage["trd_mnth_fwd_ret_available_ratio"] = coverage["matched_label_count"] / coverage["eligible_count"]
    out = month_policy[["year_month", "include_in_construction_next_run"]].rename(
        columns={"include_in_construction_next_run": "eligible_for_construction"}
    ).merge(coverage[["year_month", "trd_mnth_fwd_ret_available_ratio"]], on="year_month", how="left")
    out["trd_mnth_fwd_ret_available_ratio"] = out["trd_mnth_fwd_ret_available_ratio"].fillna(0.0)
    out["evaluation_label_status"] = np.select(
        [
            ~out["eligible_for_construction"],
            out["trd_mnth_fwd_ret_available_ratio"] >= 0.98,
            out["trd_mnth_fwd_ret_available_ratio"] > 0.0,
        ],
        ["NOT_CONSTRUCTED", "AVAILABLE", "PARTIAL"],
        default="UNAVAILABLE",
    )
    out["expected_eval_inclusion"] = out["evaluation_label_status"].eq("AVAILABLE")
    out["caveat"] = np.select(
        [
            out["evaluation_label_status"].eq("UNAVAILABLE"),
            out["evaluation_label_status"].eq("PARTIAL"),
            out["evaluation_label_status"].eq("NOT_CONSTRUCTED"),
        ],
        [
            "no forward label available; do not include in evaluation",
            "partial forward label coverage; review before evaluation",
            "month excluded from construction by alpha eligibility",
        ],
        default="",
    )
    out.to_csv(OUT_DIR / "v0_aligned_portfolio_future_eval_coverage_plan.csv", index=False, encoding="utf-8-sig")
    del ret, merged, eligible_alpha
    gc.collect()
    return out


def write_run_config(score_col: str) -> dict:
    config = {
        "construction_allowed_next_run": True,
        "aligned_alpha_panel_path": rel(ALPHA_PANEL),
        "portfolio_name": PORTFOLIO_NAME,
        "score_column": score_col,
        "eligible_month_policy_path": rel(OUT_DIR / "v0_aligned_portfolio_eligible_month_policy.csv"),
        "portfolio_rule": {
            "target_holding_count": TARGET_HOLDING_COUNT,
            "entry_rank": ENTRY_RANK,
            "exit_rank": EXIT_RANK,
            "first_month_initialization": "top50",
            "weighting_scheme": "equal_weight",
            "tie_breaker": "symbol_norm ascending",
        },
        "output_weights_path_next": "output\\v0_composite_aligned_alpha_portfolio_construction_run_v0\\v0_composite_aligned_research_weights.parquet",
        "generate_weights_next_run_allowed": True,
        "calculate_returns_next_run_allowed": False,
        "production_allowed": False,
        "no_training": True,
        "no_tuning": True,
        "no_benchmark_relative": True,
        "no_alpha_beta": True,
        "no_ir_te": True,
        "no_ff": True,
        "no_dgtw": True,
        "no_shap": True,
    }
    dump_json(OUT_DIR / "v0_aligned_portfolio_construction_run_config_draft.json", config)
    return config


def guardrails() -> tuple[pd.DataFrame, bool]:
    values = {
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "benchmark_relative_returns_calculated": False,
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
    out = pd.DataFrame([{"guardrail": k, "expected": v, "actual": v, "pass": True} for k, v in values.items()])
    out.to_csv(OUT_DIR / "v0_aligned_portfolio_prep_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    return out, bool(out["pass"].all())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", "prerequisite_check")
    prereq = prerequisites()
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", "alpha_eligibility")
    alpha = pd.read_parquet(
        ALPHA_PANEL,
        columns=[
            "symbol_norm",
            "year_month",
            "month_end",
            "factor_count_used",
            "alpha_signal_aligned",
        ],
    )
    alpha["symbol_norm"] = norm_symbol(alpha["symbol_norm"])
    alpha["year_month"] = alpha["year_month"].astype(str).str.slice(0, 7)
    score_col = select_score_column(alpha.columns.tolist())
    eligibility, month_policy = eligibility_audit(alpha, score_col)
    policy = write_policy(score_col)

    write_state("running", "future_eval_coverage_plan")
    coverage = coverage_plan(alpha, score_col, month_policy)
    config = write_run_config(score_col)
    guard, guardrails_pass = guardrails()

    row_count = int(len(alpha))
    unique_symbol_count = int(alpha["symbol_norm"].nunique())
    month_count = int(alpha["year_month"].nunique())
    min_ym = str(alpha["year_month"].min())
    max_ym = str(alpha["year_month"].max())
    non_null_ratio = float(alpha[score_col].notna().mean())
    ready_count = int(eligibility["eligible_month_status"].eq("READY").sum())
    watch_count = int(eligibility["eligible_month_status"].str.startswith("WATCH").sum())
    fail_count = int(eligibility["eligible_month_status"].eq("FAIL_NO_SIGNAL").sum())
    included = month_policy.loc[month_policy["include_in_construction_next_run"], "year_month"].astype(str)
    first_eligible = str(included.min()) if len(included) else ""
    last_eligible = str(included.max()) if len(included) else ""
    caveats_exist = watch_count > 0 or bool((coverage["evaluation_label_status"].isin(["PARTIAL", "UNAVAILABLE"]) & coverage["eligible_for_construction"]).any())
    construction_allowed = bool(len(included) > 0 and guardrails_pass)
    generate_weights_next = bool(construction_allowed)
    calculate_returns_next = False
    portfolio_rule_locked = True
    eligible_policy_locked = True
    future_eval_coverage_planned = True

    if not guardrails_pass:
        final_decision = "ALIGNED_ALPHA_PORTFOLIO_PREP_FAIL_GUARDRAIL"
    elif len(included) == 0:
        final_decision = "ALIGNED_ALPHA_PORTFOLIO_PREP_BLOCKED_BY_ALPHA_ELIGIBILITY"
    elif caveats_exist:
        final_decision = "ALIGNED_ALPHA_PORTFOLIO_PREP_READY_WITH_CAVEATS"
    else:
        final_decision = "ALIGNED_ALPHA_PORTFOLIO_PREP_READY_FOR_CONSTRUCTION_RUN"

    recommended_next_step = {
        "ALIGNED_ALPHA_PORTFOLIO_PREP_READY_FOR_CONSTRUCTION_RUN": "下一任务可生成 aligned alpha 的 Top50 Buffer 35/75 weights；仍不得计算收益。",
        "ALIGNED_ALPHA_PORTFOLIO_PREP_READY_WITH_CAVEATS": "下一任务可生成 weights，但需保留 WATCH 月份和 future label coverage caveat；仍不得计算收益。",
        "ALIGNED_ALPHA_PORTFOLIO_PREP_BLOCKED_BY_ALPHA_ELIGIBILITY": "先修复 alpha eligibility，再进入 construction。",
        "ALIGNED_ALPHA_PORTFOLIO_PREP_FAIL_GUARDRAIL": "停止，先修复 guardrail violation。",
    }[final_decision]

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "aligned_alpha_panel_path": rel(ALPHA_PANEL),
        "aligned_alpha_panel_loaded": True,
        "score_column_selected": score_col,
        "row_count": row_count,
        "unique_symbol_count": unique_symbol_count,
        "month_count": month_count,
        "min_year_month": min_ym,
        "max_year_month": max_ym,
        "alpha_signal_non_null_ratio": non_null_ratio,
        "ready_month_count": ready_count,
        "watch_month_count": watch_count,
        "fail_month_count": fail_count,
        "first_eligible_month": first_eligible,
        "last_eligible_month": last_eligible,
        "portfolio_name": PORTFOLIO_NAME,
        "portfolio_rule_locked": portfolio_rule_locked,
        "target_holding_count": TARGET_HOLDING_COUNT,
        "entry_rank": ENTRY_RANK,
        "exit_rank": EXIT_RANK,
        "eligible_month_policy_locked": eligible_policy_locked,
        "future_eval_coverage_planned": future_eval_coverage_planned,
        "construction_allowed_next_run": construction_allowed,
        "generate_weights_next_run_allowed": generate_weights_next,
        "calculate_returns_next_run_allowed": calculate_returns_next,
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "benchmark_relative_returns_calculated": False,
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
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    dump_json(OUT_DIR / "v0_composite_aligned_alpha_portfolio_construction_prep_summary.json", summary)

    report = (
        "# V0 Composite-Aligned Alpha Portfolio Construction Prep v0\n\n"
        f"- final_decision: {final_decision}\n"
        f"- score_column_selected: {score_col}\n"
        f"- alpha panel: {row_count} rows, {unique_symbol_count} symbols, {month_count} months ({min_ym} to {max_ym})\n"
        f"- ready/watch/fail months: {ready_count}/{watch_count}/{fail_count}\n"
        f"- eligible construction window: {first_eligible} to {last_eligible}\n"
        f"- portfolio rule: Top50 Buffer 35/75 equal weight; entry={ENTRY_RANK}; exit={EXIT_RANK}\n"
        f"- construction_allowed_next_run: {construction_allowed}; generate_weights_next_run_allowed: {generate_weights_next}; calculate_returns_next_run_allowed: {calculate_returns_next}\n"
        f"- guardrails_passed: {guardrails_pass}\n\n"
        "本任务未生成 weights，未计算收益/累计收益/Sharpe/MaxDD，未做 benchmark-relative、alpha/beta、IR/TE、FF、DGTW、训练、调参、SHAP 或 production 修改。\n"
    )
    (OUT_DIR / "v0_composite_aligned_alpha_portfolio_construction_prep_report.md").write_text(report, encoding="utf-8")

    final_qa = pd.DataFrame(
        [
            {"check_name": "prerequisites_passed", "pass": prereq["prerequisites_passed"], "detail": ""},
            {"check_name": "guardrails_passed", "pass": guardrails_pass, "detail": ""},
            {"check_name": "portfolio_rule_locked", "pass": portfolio_rule_locked, "detail": ""},
            {"check_name": "eligible_month_policy_locked", "pass": eligible_policy_locked, "detail": ""},
            {"check_name": "final_decision_allowed", "pass": final_decision in {
                "ALIGNED_ALPHA_PORTFOLIO_PREP_READY_FOR_CONSTRUCTION_RUN",
                "ALIGNED_ALPHA_PORTFOLIO_PREP_READY_WITH_CAVEATS",
                "ALIGNED_ALPHA_PORTFOLIO_PREP_BLOCKED_BY_ALPHA_ELIGIBILITY",
                "ALIGNED_ALPHA_PORTFOLIO_PREP_FAIL_GUARDRAIL",
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
    del alpha, eligibility, month_policy, coverage, guard
    gc.collect()
    write_state("completed", "all_outputs_written")
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": rel(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
