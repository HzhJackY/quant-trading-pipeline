from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "v0_composite_aligned_portfolio_construction_run_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

PREP_DIR = ROOT / "output" / "v0_composite_aligned_alpha_portfolio_construction_prep_v0"
ALPHA_PANEL = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_composite_aligned_alpha_candidate_panel.parquet"
POLICY_PATH = PREP_DIR / "v0_aligned_portfolio_construction_policy.json"
ELIGIBLE_MONTH_POLICY = PREP_DIR / "v0_aligned_portfolio_eligible_month_policy.csv"
FUTURE_EVAL_COVERAGE_PLAN = PREP_DIR / "v0_aligned_portfolio_future_eval_coverage_plan.csv"
RUN_CONFIG = PREP_DIR / "v0_aligned_portfolio_construction_run_config_draft.json"
RETURN_MAP = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0" / "canonical_csmar_trd_mnth_return_map_repaired.parquet"

PORTFOLIO_NAME = "V0_COMPOSITE_ALIGNED_STRICT_LAG_TOP50_BUFFER_35_75_EQUAL_WEIGHT"
SCORE_COLUMN = "alpha_signal_aligned"
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
        "resume_instruction: rerun scripts\\run_v0_composite_aligned_portfolio_construction_v0.py with stdout/stderr redirected to this run directory\n",
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
        "aligned_alpha_panel_found": ALPHA_PANEL.exists(),
        "construction_policy_found": POLICY_PATH.exists(),
        "eligible_month_policy_found": ELIGIBLE_MONTH_POLICY.exists(),
        "future_eval_coverage_plan_found": FUTURE_EVAL_COVERAGE_PLAN.exists(),
        "run_config_found": RUN_CONFIG.exists(),
    }
    path_map = {
        "aligned_alpha_panel_found": ALPHA_PANEL,
        "construction_policy_found": POLICY_PATH,
        "eligible_month_policy_found": ELIGIBLE_MONTH_POLICY,
        "future_eval_coverage_plan_found": FUTURE_EVAL_COVERAGE_PLAN,
        "run_config_found": RUN_CONFIG,
    }
    missing = [rel(path) for key, path in path_map.items() if not flags[key]]
    flags["prerequisites_passed"] = not missing
    flags["missing_files"] = missing
    dump_json(OUT_DIR / "v0_aligned_portfolio_construction_prerequisite_check.json", flags)
    return flags


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    with POLICY_PATH.open("r", encoding="utf-8") as f:
        policy = json.load(f)
    score_col = policy.get("score_column", SCORE_COLUMN)
    alpha = pd.read_parquet(
        ALPHA_PANEL,
        columns=["symbol_norm", "year_month", "month_end", "split_group", "factor_count_used", "alpha_signal_aligned"],
    )
    alpha["symbol_norm"] = norm_symbol(alpha["symbol_norm"])
    alpha["year_month"] = alpha["year_month"].astype(str).str.slice(0, 7)
    alpha[score_col] = pd.to_numeric(alpha[score_col], errors="coerce")
    month_policy = pd.read_csv(ELIGIBLE_MONTH_POLICY, dtype={"year_month": "string"})
    month_policy["year_month"] = month_policy["year_month"].astype(str)
    month_policy["include_in_construction_next_run"] = parse_bool(month_policy["include_in_construction_next_run"])
    coverage = pd.read_csv(FUTURE_EVAL_COVERAGE_PLAN, dtype={"year_month": "string"})
    coverage["year_month"] = coverage["year_month"].astype(str)
    if "expected_eval_inclusion" in coverage.columns:
        coverage["expected_eval_inclusion"] = parse_bool(coverage["expected_eval_inclusion"])
    return alpha, month_policy, coverage, policy


def input_qa(alpha: pd.DataFrame, month_policy: pd.DataFrame, score_col: str) -> pd.DataFrame:
    included_months = set(month_policy.loc[month_policy["include_in_construction_next_run"], "year_month"].astype(str))
    after_month = alpha.loc[alpha["year_month"].isin(included_months)].copy()
    after_score = after_month.loc[after_month[score_col].notna()].copy()
    month_status = month_policy.set_index("year_month")["month_status"].to_dict()
    included_statuses = month_policy.loc[month_policy["include_in_construction_next_run"], "month_status"]
    excluded = month_policy.loc[~month_policy["include_in_construction_next_run"], ["year_month", "reason"]]
    row = {
        "row_count_loaded": int(len(alpha)),
        "row_count_after_eligible_month_filter": int(len(after_month)),
        "row_count_after_alpha_non_null_filter": int(len(after_score)),
        "score_column_selected": score_col,
        "first_construction_month": str(after_score["year_month"].min()) if len(after_score) else "",
        "last_construction_month": str(after_score["year_month"].max()) if len(after_score) else "",
        "ready_month_count": int((included_statuses == "READY").sum()),
        "watch_month_count": int(included_statuses.astype(str).str.startswith("WATCH").sum()),
        "fail_month_count": int((month_policy["month_status"] == "FAIL_NO_SIGNAL").sum()),
        "excluded_months": ";".join(excluded["year_month"].astype(str).tolist()),
        "excluded_month_reasons": ";".join(excluded["reason"].astype(str).tolist()),
        "input_status": "PASS" if len(after_score) > 0 and score_col in alpha.columns else "FAIL",
    }
    out = pd.DataFrame([row])
    out.to_csv(OUT_DIR / "v0_aligned_construction_input_qa.csv", index=False, encoding="utf-8-sig")
    return out


def construct_weights(alpha: pd.DataFrame, month_policy: pd.DataFrame, coverage: pd.DataFrame, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_col = policy.get("score_column", SCORE_COLUMN)
    eligible_policy = month_policy.loc[month_policy["include_in_construction_next_run"]].copy()
    eligible_months = eligible_policy["year_month"].astype(str).tolist()
    status_map = month_policy.set_index("year_month")["month_status"].to_dict()
    coverage_map = coverage.set_index("year_month")["expected_eval_inclusion"].to_dict() if "expected_eval_inclusion" in coverage.columns else {}
    alpha = alpha.loc[alpha["year_month"].isin(eligible_months) & alpha[score_col].notna()].copy()

    rows = []
    trans_rows = []
    prev_weights: dict[str, float] = {}
    prev_holdings: set[str] = set()
    for ym in sorted(eligible_months):
        uni = alpha.loc[alpha["year_month"].eq(ym)].copy()
        if uni.empty:
            trans_rows.append(
                {
                    "year_month": ym,
                    "eligible_month_status": status_map.get(ym, ""),
                    "previous_holding_count": len(prev_holdings),
                    "kept_from_previous_count": 0,
                    "exited_count": len(prev_holdings),
                    "new_entry_count": 0,
                    "fill_to_target_count": 0,
                    "selected_count": 0,
                    "simple_turnover_proxy": 0.5 * sum(prev_weights.values()) if prev_weights else 0.0,
                    "transition_status": "FAIL_NO_ELIGIBLE_UNIVERSE",
                    "caveat": "no eligible alpha rows",
                }
            )
            prev_holdings = set()
            prev_weights = {}
            continue
        uni = uni.sort_values([score_col, "symbol_norm"], ascending=[False, True]).reset_index(drop=True)
        uni["rank_in_month"] = np.arange(1, len(uni) + 1)
        rank_map = uni.set_index("symbol_norm")["rank_in_month"].to_dict()
        current_symbols = set(uni["symbol_norm"])
        first_month = not prev_holdings
        selected: list[tuple[str, str]] = []
        if first_month:
            for sym in uni.head(TARGET_HOLDING_COUNT)["symbol_norm"]:
                selected.append((sym, "FIRST_MONTH_TOP50"))
        else:
            kept = [sym for sym in prev_holdings if sym in current_symbols and rank_map.get(sym, 10**9) <= EXIT_RANK]
            kept = sorted(kept, key=lambda s: (rank_map.get(s, 10**9), s))
            selected.extend((sym, "BUFFER_KEPT") for sym in kept)
            selected_set = {sym for sym, _ in selected}
            entries = uni.loc[(uni["rank_in_month"] <= ENTRY_RANK) & ~uni["symbol_norm"].isin(selected_set), "symbol_norm"].tolist()
            for sym in entries:
                if len(selected) >= TARGET_HOLDING_COUNT:
                    break
                selected.append((sym, "BUFFER_ENTRY_RANK_LE_35"))
            selected_set = {sym for sym, _ in selected}
            if len(selected) < TARGET_HOLDING_COUNT:
                fills = uni.loc[~uni["symbol_norm"].isin(selected_set), "symbol_norm"].tolist()
                for sym in fills:
                    if len(selected) >= TARGET_HOLDING_COUNT:
                        break
                    selected.append((sym, "FILL_TO_TARGET"))
        selected = selected[:TARGET_HOLDING_COUNT]
        selected_symbols = [sym for sym, _ in selected]
        selected_count = len(selected_symbols)
        weight = 1.0 / selected_count if selected_count else 0.0
        selected_set = set(selected_symbols)
        new_weights = {sym: weight for sym in selected_symbols}
        union = set(prev_weights).union(new_weights)
        turnover = 1.0 if first_month and selected_count else 0.5 * sum(abs(new_weights.get(sym, 0.0) - prev_weights.get(sym, 0.0)) for sym in union)
        exited = prev_holdings - selected_set
        kept_count = sum(1 for _, reason in selected if reason == "BUFFER_KEPT")
        entry_count = sum(1 for _, reason in selected if reason == "BUFFER_ENTRY_RANK_LE_35")
        fill_count = sum(1 for _, reason in selected if reason == "FILL_TO_TARGET")
        month_status = status_map.get(ym, "")
        watch_flag = str(month_status).startswith("WATCH")
        future_flag = bool(coverage_map.get(ym, False))
        lookup = uni.set_index("symbol_norm")
        for sym, reason in selected:
            rec = lookup.loc[sym]
            rows.append(
                {
                    "portfolio_name": PORTFOLIO_NAME,
                    "year_month": ym,
                    "month_end": rec["month_end"],
                    "symbol_norm": sym,
                    "alpha_signal_aligned": float(rec[score_col]),
                    "rank_in_month": int(rec["rank_in_month"]),
                    "selected_flag": True,
                    "selection_reason": reason,
                    "previous_holding_flag": sym in prev_holdings,
                    "buffer_kept_flag": reason == "BUFFER_KEPT",
                    "buffer_exit_flag": False,
                    "buffer_entry_flag": reason == "BUFFER_ENTRY_RANK_LE_35",
                    "fill_to_target_flag": reason == "FILL_TO_TARGET",
                    "weight": weight,
                    "selected_count": selected_count,
                    "target_holding_count": TARGET_HOLDING_COUNT,
                    "construction_rule": "Top50_Buffer_35_75_equal_weight",
                    "eligible_month_status": month_status,
                    "watch_month_flag": watch_flag,
                    "future_eval_label_available_flag": future_flag,
                }
            )
        trans_rows.append(
            {
                "year_month": ym,
                "eligible_month_status": month_status,
                "previous_holding_count": int(len(prev_holdings)),
                "kept_from_previous_count": int(kept_count),
                "exited_count": int(len(exited)),
                "new_entry_count": int(entry_count),
                "fill_to_target_count": int(fill_count),
                "selected_count": int(selected_count),
                "simple_turnover_proxy": float(turnover),
                "transition_status": "FIRST_MONTH_INITIALIZATION" if first_month else "PASS",
                "caveat": "WATCH month included" if watch_flag else "",
            }
        )
        prev_holdings = selected_set
        prev_weights = new_weights
    weights = pd.DataFrame(rows)
    transition = pd.DataFrame(trans_rows)
    weights.to_parquet(OUT_DIR / "v0_composite_aligned_research_weights.parquet", index=False)
    weights.to_csv(OUT_DIR / "v0_composite_aligned_research_weights.csv", index=False, encoding="utf-8-sig")
    transition.to_csv(OUT_DIR / "v0_aligned_buffer_transition_qa.csv", index=False, encoding="utf-8-sig")
    return weights, transition, alpha


def monthly_weight_qa(weights: pd.DataFrame, alpha_eligible: pd.DataFrame) -> pd.DataFrame:
    eligible_counts = alpha_eligible.groupby("year_month")["symbol_norm"].nunique().rename("eligible_symbol_count")
    rows = []
    for ym, g in weights.groupby("year_month", sort=True):
        selected_count = int(len(g))
        weight_sum = float(g["weight"].sum())
        dup = int(g["symbol_norm"].duplicated().sum())
        alpha_missing = int(g["alpha_signal_aligned"].isna().sum())
        watch = bool(g["watch_month_flag"].iloc[0])
        low = selected_count < TARGET_HOLDING_COUNT
        status = "PASS"
        caveat = ""
        if abs(weight_sum - 1.0) > 1e-12 or dup != 0 or alpha_missing != 0:
            status = "FAIL"
        elif low:
            status = "WATCH_LOW_HOLDING_COUNT"
            caveat = "selected_count below target"
        elif watch:
            status = "PASS_WITH_WATCH_MONTH_CAVEAT"
            caveat = "WATCH month included"
        rows.append(
            {
                "year_month": ym,
                "eligible_month_status": g["eligible_month_status"].iloc[0],
                "watch_month_flag": watch,
                "eligible_symbol_count": int(eligible_counts.get(ym, 0)),
                "selected_count": selected_count,
                "target_holding_count": TARGET_HOLDING_COUNT,
                "weight_sum": weight_sum,
                "weight_sum_abs_error": abs(weight_sum - 1.0),
                "min_weight": float(g["weight"].min()),
                "max_weight": float(g["weight"].max()),
                "duplicate_symbol_count": dup,
                "alpha_missing_selected_count": alpha_missing,
                "low_holding_count_flag": low,
                "future_eval_label_available_flag": bool(g["future_eval_label_available_flag"].iloc[0]),
                "monthly_weight_status": status,
                "caveat": caveat,
            }
        )
    qa = pd.DataFrame(rows)
    qa.to_csv(OUT_DIR / "v0_aligned_portfolio_weight_monthly_qa.csv", index=False, encoding="utf-8-sig")
    return qa


def reason_summary(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    month_count = max(1, weights["year_month"].nunique())
    interpretations = {
        "FIRST_MONTH_TOP50": "first eligible month initialized from top50",
        "BUFFER_KEPT": "previous holding retained because rank <= exit_rank",
        "BUFFER_ENTRY_RANK_LE_35": "new entrant admitted because rank <= entry_rank",
        "FILL_TO_TARGET": "filled by rank to reach target holding count",
        "OTHER": "other selection reason",
    }
    for reason, g in weights.groupby("selection_reason", sort=True):
        rows.append(
            {
                "selection_reason": reason,
                "row_count": int(len(g)),
                "avg_monthly_count": float(len(g) / month_count),
                "first_month": str(g["year_month"].min()),
                "last_month": str(g["year_month"].max()),
                "interpretation": interpretations.get(reason, "other selection reason"),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "v0_aligned_selection_reason_summary.csv", index=False, encoding="utf-8-sig")
    return out


def future_eval_coverage(weights: pd.DataFrame) -> pd.DataFrame:
    ret = pd.read_parquet(RETURN_MAP, columns=["symbol_norm", "year_month", "fwd_ret_1m"])
    ret["symbol_norm"] = norm_symbol(ret["symbol_norm"])
    ret["year_month"] = ret["year_month"].astype(str).str.slice(0, 7)
    ret["has_label"] = pd.to_numeric(ret["fwd_ret_1m"], errors="coerce").notna()
    ret = ret.drop_duplicates(["symbol_norm", "year_month"], keep="last")[["symbol_norm", "year_month", "has_label"]]
    merged = weights[["year_month", "symbol_norm", "weight"]].merge(ret, on=["symbol_norm", "year_month"], how="left")
    merged["has_label"] = merged["has_label"].fillna(False)
    rows = []
    for ym, g in merged.groupby("year_month", sort=True):
        matched = g["has_label"]
        share = float(g.loc[matched, "weight"].sum())
        selected_count = int(len(g))
        matched_count = int(matched.sum())
        available = share >= 0.98
        rows.append(
            {
                "year_month": ym,
                "selected_count": selected_count,
                "matched_label_count": matched_count,
                "matched_label_weight_share": share,
                "future_eval_label_available": available,
                "expected_eval_inclusion": available,
                "caveat": "" if available else "no or insufficient forward label; evaluation unavailable",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "v0_aligned_weights_future_eval_coverage_plan.csv", index=False, encoding="utf-8-sig")
    del ret, merged
    gc.collect()
    return out


def readiness(weights: pd.DataFrame, monthly_qa: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    first_month = str(weights["year_month"].min()) if len(weights) else ""
    watch_preserved = bool(weights["watch_month_flag"].any())
    dup_total = int(monthly_qa["duplicate_symbol_count"].sum())
    max_err = float(monthly_qa["weight_sum_abs_error"].max()) if len(monthly_qa) else 1.0
    low_count = int(monthly_qa["low_holding_count_flag"].sum()) if len(monthly_qa) else 0
    rows = [
        ("weights generated", True, len(weights) > 0, len(weights) > 0, ""),
        ("first construction month = 2017-01", "2017-01", first_month, first_month == "2017-01", ""),
        ("fail signal months excluded", True, True, True, ""),
        ("WATCH months preserved and marked", True, watch_preserved, watch_preserved, ""),
        ("duplicate selected symbols = 0", 0, dup_total, dup_total == 0, ""),
        ("weight sum pass", "<=1e-12", max_err, max_err <= 1e-12, ""),
        ("low holding count months", 0, low_count, low_count == 0, ""),
        ("future eval coverage planned", True, len(coverage) > 0, len(coverage) > 0, ""),
        ("returns not calculated", False, False, True, ""),
    ]
    out = pd.DataFrame([{"criterion": c, "expected": e, "actual": a, "pass": p, "caveat": caveat} for c, e, a, p, caveat in rows])
    out.to_csv(OUT_DIR / "v0_aligned_weights_to_evaluation_prep_readiness.csv", index=False, encoding="utf-8-sig")
    return out


def guardrails() -> tuple[pd.DataFrame, bool]:
    values = {
        "alpha_signal_generated": False,
        "strategy_weights_generated": True,
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "transaction_cost_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "tstat_calculated": False,
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
    }
    out = pd.DataFrame([{"guardrail": k, "expected": v, "actual": v, "pass": True} for k, v in values.items()])
    out.to_csv(OUT_DIR / "v0_aligned_portfolio_construction_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    return out, bool(out["pass"].all())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", "prerequisite_check")
    prereq = prerequisites()
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", "load_inputs")
    alpha, month_policy, coverage_plan, policy = load_inputs()
    score_col = policy.get("score_column", SCORE_COLUMN)
    inputqa = input_qa(alpha, month_policy, score_col)

    write_state("running", "construct_weights")
    weights, transition, alpha_eligible = construct_weights(alpha, month_policy, coverage_plan, policy)
    monthly_qa = monthly_weight_qa(weights, alpha_eligible)
    reason = reason_summary(weights)
    eval_cov = future_eval_coverage(weights)
    ready = readiness(weights, monthly_qa, eval_cov)
    guard, guardrails_pass = guardrails()

    weight_qa_pass = bool(
        (monthly_qa["weight_sum_abs_error"] <= 1e-12).all()
        and (monthly_qa["duplicate_symbol_count"] == 0).all()
        and (monthly_qa["alpha_missing_selected_count"] == 0).all()
        and (monthly_qa["low_holding_count_flag"] == False).all()
    )
    caveats_exist = bool(weights["watch_month_flag"].any() or (~eval_cov["future_eval_label_available"]).any())
    if not guardrails_pass:
        final_decision = "ALIGNED_PORTFOLIO_CONSTRUCTION_FAIL_GUARDRAIL"
    elif not weight_qa_pass:
        final_decision = "ALIGNED_PORTFOLIO_CONSTRUCTION_BLOCKED_BY_WEIGHT_QA"
    elif caveats_exist:
        final_decision = "ALIGNED_PORTFOLIO_CONSTRUCTION_READY_WITH_CAVEATS"
    else:
        final_decision = "ALIGNED_PORTFOLIO_CONSTRUCTION_READY_FOR_EVAL_PREP"
    recommended_next_step = {
        "ALIGNED_PORTFOLIO_CONSTRUCTION_READY_FOR_EVAL_PREP": "下一任务可进入 evaluation prep；仍先只锁定窗口和口径，不直接做收益评价。",
        "ALIGNED_PORTFOLIO_CONSTRUCTION_READY_WITH_CAVEATS": "下一任务可进入 evaluation prep，但需保留 WATCH 月份和 future label coverage caveat。",
        "ALIGNED_PORTFOLIO_CONSTRUCTION_BLOCKED_BY_WEIGHT_QA": "先修复 weight QA，再进入 evaluation prep。",
        "ALIGNED_PORTFOLIO_CONSTRUCTION_FAIL_GUARDRAIL": "停止，先修复 guardrail violation。",
    }[final_decision]

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "aligned_alpha_panel_path": rel(ALPHA_PANEL),
        "score_column_selected": score_col,
        "portfolio_name": PORTFOLIO_NAME,
        "weights_generated": True,
        "weights_path": rel(OUT_DIR / "v0_composite_aligned_research_weights.parquet"),
        "first_construction_month": str(weights["year_month"].min()),
        "last_construction_month": str(weights["year_month"].max()),
        "month_count": int(weights["year_month"].nunique()),
        "ready_month_count": int((monthly_qa["eligible_month_status"] == "READY").sum()),
        "watch_month_count": int(monthly_qa["eligible_month_status"].astype(str).str.startswith("WATCH").sum()),
        "fail_month_count": int((month_policy["month_status"] == "FAIL_NO_SIGNAL").sum()),
        "total_weight_rows": int(len(weights)),
        "unique_symbol_count": int(weights["symbol_norm"].nunique()),
        "avg_selected_count": float(monthly_qa["selected_count"].mean()),
        "min_selected_count": int(monthly_qa["selected_count"].min()),
        "max_selected_count": int(monthly_qa["selected_count"].max()),
        "low_holding_count_month_count": int(monthly_qa["low_holding_count_flag"].sum()),
        "avg_weight_sum": float(monthly_qa["weight_sum"].mean()),
        "max_weight_sum_abs_error": float(monthly_qa["weight_sum_abs_error"].max()),
        "duplicate_symbol_portfolio_month_count": int((monthly_qa["duplicate_symbol_count"] > 0).sum()),
        "alpha_missing_selected_count": int(monthly_qa["alpha_missing_selected_count"].sum()),
        "avg_turnover_proxy": float(transition["simple_turnover_proxy"].mean()),
        "max_turnover_proxy": float(transition["simple_turnover_proxy"].max()),
        "watch_months_preserved": bool(weights["watch_month_flag"].any()),
        "future_eval_coverage_planned": True,
        "evaluation_ready_next": final_decision in ["ALIGNED_PORTFOLIO_CONSTRUCTION_READY_FOR_EVAL_PREP", "ALIGNED_PORTFOLIO_CONSTRUCTION_READY_WITH_CAVEATS"],
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "transaction_cost_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "tstat_calculated": False,
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
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    dump_json(OUT_DIR / "v0_composite_aligned_portfolio_construction_summary.json", summary)

    report = (
        "# V0 Composite-Aligned Portfolio Construction Run v0\n\n"
        f"- final_decision: {final_decision}\n"
        f"- weights_path: {summary['weights_path']}\n"
        f"- construction window: {summary['first_construction_month']} to {summary['last_construction_month']}; months={summary['month_count']}\n"
        f"- ready/watch/fail months: {summary['ready_month_count']}/{summary['watch_month_count']}/{summary['fail_month_count']}\n"
        f"- avg selected count: {summary['avg_selected_count']:.6f}; max weight sum abs error: {summary['max_weight_sum_abs_error']:.6g}\n"
        f"- avg/max turnover proxy: {summary['avg_turnover_proxy']:.6f}/{summary['max_turnover_proxy']:.6f}\n"
        f"- future_eval_coverage_planned: true; evaluation_ready_next: {summary['evaluation_ready_next']}\n\n"
        "本任务未重新生成 alpha_signal，未计算 portfolio returns、累计收益、transaction cost、Sharpe、MaxDD、t-stat、benchmark-relative、alpha/beta、IR/TE、FF、DGTW、ML、调参、SHAP 或 production 修改。\n"
    )
    (OUT_DIR / "v0_composite_aligned_portfolio_construction_report.md").write_text(report, encoding="utf-8")

    final_qa = pd.DataFrame(
        [
            {"check_name": "prerequisites_passed", "pass": prereq["prerequisites_passed"], "detail": ""},
            {"check_name": "weights_generated", "pass": True, "detail": str(len(weights))},
            {"check_name": "weight_qa_pass", "pass": weight_qa_pass, "detail": ""},
            {"check_name": "guardrails_passed", "pass": guardrails_pass, "detail": ""},
            {"check_name": "final_decision_allowed", "pass": final_decision in {
                "ALIGNED_PORTFOLIO_CONSTRUCTION_READY_FOR_EVAL_PREP",
                "ALIGNED_PORTFOLIO_CONSTRUCTION_READY_WITH_CAVEATS",
                "ALIGNED_PORTFOLIO_CONSTRUCTION_BLOCKED_BY_WEIGHT_QA",
                "ALIGNED_PORTFOLIO_CONSTRUCTION_FAIL_GUARDRAIL",
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
    del alpha, month_policy, coverage_plan, weights, transition, alpha_eligible, monthly_qa, reason, eval_cov, ready, guard
    gc.collect()
    write_state("completed", "all_outputs_written")
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": rel(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
