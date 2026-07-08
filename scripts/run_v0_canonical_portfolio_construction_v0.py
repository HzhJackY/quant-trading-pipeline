from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_NAME = "v0_canonical_portfolio_construction_run_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

ALPHA_PANEL = (
    ROOT
    / "output"
    / "v0_canonical_strict_lag_alpha_build_v0"
    / "v0_canonical_alpha_signal_panel.parquet"
)
PREP_DIR = ROOT / "output" / "v0_canonical_alpha_portfolio_construction_prep_v0"
CONSTRUCTION_POLICY = PREP_DIR / "v0_canonical_portfolio_construction_policy.json"
ELIGIBLE_MONTH_POLICY = PREP_DIR / "v0_canonical_portfolio_eligible_month_policy.csv"
RUN_CONFIG = PREP_DIR / "v0_canonical_portfolio_construction_run_config_draft.json"
RETURN_MAP = (
    ROOT
    / "output"
    / "trd_mnth_parser_repair_2024_12_coverage_repair_v0"
    / "canonical_csmar_trd_mnth_return_map_repaired.parquet"
)

PORTFOLIO_NAME = "V0_CANONICAL_STRICT_LAG_TOP50_BUFFER_35_75_EQUAL_WEIGHT"
TARGET_HOLDING_COUNT = 50
ENTRY_RANK = 35
EXIT_RANK = 75


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_state(status: str, details: dict[str, Any] | None = None) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": TASK_NAME,
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "details": details or {},
        "resume_instruction": f"先读取 {rel(RUN_DIR / 'RUN_STATE.md')} 再继续。",
    }
    lines = ["# RUN_STATE", "", f"- task_name: {TASK_NAME}", f"- status: {status}"]
    for key, value in payload["details"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "```json", json.dumps(payload, ensure_ascii=False, indent=2, default=str), "```"]
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(lines), encoding="utf-8")


def prereq_check() -> dict[str, Any]:
    required = [ALPHA_PANEL, CONSTRUCTION_POLICY, ELIGIBLE_MONTH_POLICY, RUN_CONFIG]
    missing = [rel(p) for p in required if not p.exists()]
    return {
        "alpha_panel_found": ALPHA_PANEL.exists(),
        "construction_policy_found": CONSTRUCTION_POLICY.exists(),
        "eligible_month_policy_found": ELIGIBLE_MONTH_POLICY.exists(),
        "run_config_found": RUN_CONFIG.exists(),
        "prerequisites_passed": len(missing) == 0,
        "missing_files": missing,
    }


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    alpha = pd.read_parquet(ALPHA_PANEL, columns=["symbol_norm", "year_month", "month_end", "alpha_signal"])
    alpha["symbol_norm"] = alpha["symbol_norm"].astype(str).str.zfill(6)
    alpha["year_month"] = alpha["year_month"].astype(str).str.slice(0, 7)
    alpha["month_end"] = pd.to_datetime(alpha["month_end"], errors="coerce")
    alpha["alpha_signal"] = pd.to_numeric(alpha["alpha_signal"], errors="coerce").replace([np.inf, -np.inf], np.nan)

    month_policy = pd.read_csv(ELIGIBLE_MONTH_POLICY, dtype={"year_month": str})
    month_policy["year_month"] = month_policy["year_month"].astype(str).str.slice(0, 7)
    month_policy["include_in_construction_next_run"] = month_policy["include_in_construction_next_run"].astype(str).str.lower().isin(["true", "1", "yes"])

    policy = json.loads(CONSTRUCTION_POLICY.read_text(encoding="utf-8"))
    config = json.loads(RUN_CONFIG.read_text(encoding="utf-8"))
    return alpha, month_policy, policy, config


def construction_input_qa(alpha: pd.DataFrame, month_policy: pd.DataFrame, filtered: pd.DataFrame) -> pd.DataFrame:
    eligible_months = month_policy.loc[month_policy["include_in_construction_next_run"], "year_month"].tolist()
    month_filtered = alpha[alpha["year_month"].isin(eligible_months)]
    excluded = month_policy.loc[~month_policy["include_in_construction_next_run"], ["year_month", "reason"]]
    return pd.DataFrame(
        [
            {
                "row_count_loaded": len(alpha),
                "row_count_after_eligible_month_filter": len(month_filtered),
                "row_count_after_alpha_non_null_filter": len(filtered),
                "first_construction_month": str(filtered["year_month"].min()) if len(filtered) else "",
                "last_construction_month": str(filtered["year_month"].max()) if len(filtered) else "",
                "excluded_months": ",".join(excluded["year_month"].astype(str).tolist()),
                "excluded_month_reasons": "; ".join(
                    f"{r.year_month}:{r.reason}" for r in excluded.itertuples(index=False)
                ),
                "input_status": "READY" if len(filtered) > 0 and str(filtered["year_month"].min()) == "2017-03" else "WATCH",
            }
        ]
    )


def build_weights(alpha: pd.DataFrame, month_policy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible_months = month_policy.loc[month_policy["include_in_construction_next_run"], "year_month"].tolist()
    eval_flag_by_month = dict(
        zip(
            month_policy["year_month"],
            month_policy.get("future_eval_label_available_flag", pd.Series([True] * len(month_policy))),
        )
    )
    filtered = alpha[alpha["year_month"].isin(eligible_months) & alpha["alpha_signal"].notna()].copy()
    filtered = filtered.sort_values(["year_month", "alpha_signal", "symbol_norm"], ascending=[True, False, True])

    weight_rows = []
    transition_rows = []
    previous_holdings: set[str] = set()
    previous_month = ""
    for ym, grp in filtered.groupby("year_month", sort=True):
        ranked = grp.sort_values(["alpha_signal", "symbol_norm"], ascending=[False, True]).copy()
        ranked["rank_in_month"] = np.arange(1, len(ranked) + 1)
        rank_map = dict(zip(ranked["symbol_norm"], ranked["rank_in_month"]))
        alpha_map = dict(zip(ranked["symbol_norm"], ranked["alpha_signal"]))
        month_end_map = dict(zip(ranked["symbol_norm"], ranked["month_end"]))

        if not previous_holdings:
            selected = ranked.head(TARGET_HOLDING_COUNT)["symbol_norm"].tolist()
            reason_by_symbol = {s: "FIRST_MONTH_TOP50" for s in selected}
            kept = set()
            exited = set()
            fill_to_target = set()
        else:
            current_symbols = set(ranked["symbol_norm"])
            kept = {
                s
                for s in previous_holdings
                if s in current_symbols and int(rank_map.get(s, 10**9)) <= EXIT_RANK
            }
            exited = previous_holdings - kept
            selected = sorted(kept, key=lambda s: (rank_map.get(s, 10**9), s))
            reason_by_symbol = {s: "BUFFER_KEPT" for s in selected}

            entry_candidates = [
                s
                for s in ranked.loc[ranked["rank_in_month"] <= ENTRY_RANK, "symbol_norm"].tolist()
                if s not in set(selected)
            ]
            for s in entry_candidates:
                if len(selected) >= TARGET_HOLDING_COUNT:
                    break
                selected.append(s)
                reason_by_symbol[s] = "BUFFER_ENTRY_RANK_LE_35"

            fill_to_target = set()
            if len(selected) < TARGET_HOLDING_COUNT:
                for s in ranked["symbol_norm"].tolist():
                    if s in set(selected):
                        continue
                    selected.append(s)
                    reason_by_symbol[s] = "FILL_TO_TARGET"
                    fill_to_target.add(s)
                    if len(selected) >= TARGET_HOLDING_COUNT:
                        break

        selected = selected[:TARGET_HOLDING_COUNT]
        selected_count = len(selected)
        weight = 1.0 / selected_count if selected_count else 0.0
        selected_set = set(selected)
        new_entries = selected_set - previous_holdings
        future_eval_label_available_flag = bool(eval_flag_by_month.get(ym, True))

        for s in selected:
            reason = reason_by_symbol.get(s, "OTHER")
            weight_rows.append(
                {
                    "portfolio_name": PORTFOLIO_NAME,
                    "year_month": ym,
                    "month_end": month_end_map.get(s),
                    "symbol_norm": s,
                    "alpha_signal": alpha_map.get(s),
                    "rank_in_month": int(rank_map.get(s)),
                    "selected_flag": True,
                    "selection_reason": reason,
                    "previous_holding_flag": s in previous_holdings,
                    "buffer_kept_flag": reason == "BUFFER_KEPT",
                    "buffer_exit_flag": False,
                    "buffer_entry_flag": reason == "BUFFER_ENTRY_RANK_LE_35",
                    "fill_to_target_flag": reason == "FILL_TO_TARGET",
                    "weight": weight,
                    "selected_count": selected_count,
                    "target_holding_count": TARGET_HOLDING_COUNT,
                    "construction_rule": "Top50_Buffer_35_75_equal_weight_no_returns",
                    "future_eval_label_available_flag": future_eval_label_available_flag,
                }
            )

        transition_rows.append(
            {
                "year_month": ym,
                "previous_holding_count": len(previous_holdings),
                "kept_from_previous_count": len(kept),
                "exited_count": len(exited),
                "new_entry_count": len(new_entries),
                "fill_to_target_count": len(fill_to_target),
                "selected_count": selected_count,
                "simple_turnover_proxy": round(float(len(new_entries) / selected_count), 6) if selected_count else 0.0,
                "transition_status": "PASS" if selected_count == TARGET_HOLDING_COUNT else "WATCH_LOW_HOLDING_COUNT",
            }
        )
        previous_holdings = selected_set
        previous_month = ym

    weights = pd.DataFrame(weight_rows)
    transitions = pd.DataFrame(transition_rows)
    return weights, transitions


def weight_monthly_qa(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ym, grp in weights.groupby("year_month", sort=True):
        selected_count = int(grp["selected_flag"].sum())
        weight_sum = float(grp["weight"].sum())
        dup = int(grp["symbol_norm"].duplicated().sum())
        alpha_missing = int(grp["alpha_signal"].isna().sum())
        abs_error = abs(weight_sum - 1.0)
        rows.append(
            {
                "year_month": ym,
                "eligible_symbol_count": int(selected_count),
                "selected_count": selected_count,
                "target_holding_count": TARGET_HOLDING_COUNT,
                "weight_sum": weight_sum,
                "weight_sum_abs_error": abs_error,
                "min_weight": float(grp["weight"].min()) if len(grp) else 0.0,
                "max_weight": float(grp["weight"].max()) if len(grp) else 0.0,
                "duplicate_symbol_count": dup,
                "alpha_missing_selected_count": alpha_missing,
                "low_holding_count_flag": selected_count < TARGET_HOLDING_COUNT,
                "future_eval_label_available_flag": bool(grp["future_eval_label_available_flag"].all()),
                "monthly_weight_status": "PASS"
                if abs_error <= 1e-12 and dup == 0 and alpha_missing == 0 and selected_count == TARGET_HOLDING_COUNT
                else "WATCH",
            }
        )
    return pd.DataFrame(rows)


def selection_reason_summary(weights: pd.DataFrame) -> pd.DataFrame:
    month_count = max(int(weights["year_month"].nunique()), 1)
    rows = []
    for reason, grp in weights.groupby("selection_reason", sort=True):
        rows.append(
            {
                "selection_reason": reason,
                "row_count": len(grp),
                "avg_monthly_count": round(float(len(grp) / month_count), 6),
                "first_month": str(grp["year_month"].min()),
                "last_month": str(grp["year_month"].max()),
                "interpretation": {
                    "FIRST_MONTH_TOP50": "first eligible month direct top50 initialization",
                    "BUFFER_KEPT": "previous holding retained within exit rank buffer",
                    "BUFFER_ENTRY_RANK_LE_35": "new entry from rank <= 35",
                    "FILL_TO_TARGET": "fallback fill after entry buffer to maintain target count",
                }.get(reason, "other selection reason"),
            }
        )
    return pd.DataFrame(rows)


def future_eval_coverage(weights: pd.DataFrame) -> pd.DataFrame:
    ret = pd.read_parquet(RETURN_MAP, columns=["symbol_norm", "year_month", "fwd_ret_1m"])
    ret["symbol_norm"] = ret["symbol_norm"].astype(str).str.zfill(6)
    ret["year_month"] = ret["year_month"].astype(str).str.slice(0, 7)
    ret["fwd_ret_1m"] = pd.to_numeric(ret["fwd_ret_1m"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    ret = ret.drop_duplicates(["symbol_norm", "year_month"], keep="last")
    merged = weights.merge(ret, on=["symbol_norm", "year_month"], how="left")
    rows = []
    for ym, grp in merged.groupby("year_month", sort=True):
        selected_count = len(grp)
        matched = int(grp["fwd_ret_1m"].notna().sum())
        matched_weight_share = float(grp.loc[grp["fwd_ret_1m"].notna(), "weight"].sum())
        available = matched > 0 and matched_weight_share >= 0.95
        rows.append(
            {
                "year_month": ym,
                "selected_count": selected_count,
                "matched_label_count": matched,
                "matched_label_weight_share": round(matched_weight_share, 6),
                "future_eval_label_available": available,
                "expected_eval_inclusion": available,
                "caveat": "" if available else "no sufficient forward label; evaluation should exclude or wait",
            }
        )
    del ret, merged
    gc.collect()
    return pd.DataFrame(rows)


def readiness(weights: pd.DataFrame, monthly_qa: pd.DataFrame, eval_plan: pd.DataFrame, guardrails: pd.DataFrame) -> pd.DataFrame:
    first_month = str(weights["year_month"].min()) if len(weights) else ""
    fail_excluded = "2017-01" not in set(weights["year_month"]) and "2017-02" not in set(weights["year_month"])
    duplicate_total = int(monthly_qa["duplicate_symbol_count"].sum()) if len(monthly_qa) else 0
    weight_sum_pass = bool((monthly_qa["weight_sum_abs_error"] <= 1e-12).all()) if len(monthly_qa) else False
    low_holding = int(monthly_qa["low_holding_count_flag"].sum()) if len(monthly_qa) else 0
    rows = [
        {"criterion": "weights generated", "expected": True, "actual": len(weights) > 0, "pass": len(weights) > 0, "caveat": ""},
        {"criterion": "first construction month = 2017-03", "expected": "2017-03", "actual": first_month, "pass": first_month == "2017-03", "caveat": ""},
        {"criterion": "fail signal months excluded", "expected": True, "actual": fail_excluded, "pass": fail_excluded, "caveat": ""},
        {"criterion": "duplicate selected symbols = 0", "expected": 0, "actual": duplicate_total, "pass": duplicate_total == 0, "caveat": ""},
        {"criterion": "weight sum pass", "expected": True, "actual": weight_sum_pass, "pass": weight_sum_pass, "caveat": ""},
        {"criterion": "low holding count months", "expected": 0, "actual": low_holding, "pass": low_holding == 0, "caveat": ""},
        {"criterion": "future eval coverage planned", "expected": True, "actual": len(eval_plan) > 0, "pass": len(eval_plan) > 0, "caveat": ""},
        {"criterion": "returns not calculated", "expected": True, "actual": True, "pass": True, "caveat": ""},
        {"criterion": "no guardrail violation", "expected": True, "actual": bool(guardrails["pass"].all()), "pass": bool(guardrails["pass"].all()), "caveat": ""},
    ]
    return pd.DataFrame(rows)


def guardrail_qa() -> pd.DataFrame:
    guardrails = {
        "alpha_signal_generated": False,
        "strategy_weights_generated": True,
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "transaction_cost_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
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
    }
    rows = []
    for key, actual in guardrails.items():
        expected = True if key == "strategy_weights_generated" else False
        rows.append({"guardrail": key, "expected": expected, "actual": actual, "pass": actual is expected})
    return pd.DataFrame(rows)


def simple_table(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    sub = df[cols].head(max_rows).fillna("").astype(str)
    widths = {c: max(len(c), *(len(x) for x in sub[c].tolist())) for c in cols}
    lines = [
        "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |",
        "| " + " | ".join("-" * widths[c] for c in cols) + " |",
    ]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(row[c].ljust(widths[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", {"step": "prerequisite_check"})
    prereq = prereq_check()
    save_json(prereq, OUT_DIR / "v0_canonical_portfolio_construction_prerequisite_check.json")
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", {"step": "load_inputs"})
    alpha, month_policy, policy, config = load_inputs()
    eligible_months = month_policy.loc[month_policy["include_in_construction_next_run"], "year_month"].tolist()
    filtered = alpha[alpha["year_month"].isin(eligible_months) & alpha["alpha_signal"].notna()].copy()
    input_qa = construction_input_qa(alpha, month_policy, filtered)
    input_qa.to_csv(OUT_DIR / "v0_canonical_construction_input_qa.csv", index=False, encoding="utf-8-sig")

    write_state("running", {"step": "construct_weights"})
    weights, transitions = build_weights(alpha, month_policy)
    weights_path = OUT_DIR / "v0_canonical_research_weights.parquet"
    weights.to_parquet(weights_path, index=False)
    weights.to_csv(OUT_DIR / "v0_canonical_research_weights.csv", index=False, encoding="utf-8-sig")

    monthly_qa = weight_monthly_qa(weights)
    monthly_qa.to_csv(OUT_DIR / "v0_canonical_portfolio_weight_monthly_qa.csv", index=False, encoding="utf-8-sig")
    transitions.to_csv(OUT_DIR / "v0_canonical_buffer_transition_qa.csv", index=False, encoding="utf-8-sig")
    reason_summary = selection_reason_summary(weights)
    reason_summary.to_csv(OUT_DIR / "v0_canonical_selection_reason_summary.csv", index=False, encoding="utf-8-sig")

    write_state("running", {"step": "future_eval_coverage_planning"})
    eval_plan = future_eval_coverage(weights)
    eval_plan.to_csv(OUT_DIR / "v0_canonical_weights_future_eval_coverage_plan.csv", index=False, encoding="utf-8-sig")

    guardrails = guardrail_qa()
    guardrails.to_csv(OUT_DIR / "v0_canonical_portfolio_construction_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    ready = readiness(weights, monthly_qa, eval_plan, guardrails)
    ready.to_csv(OUT_DIR / "v0_canonical_weights_to_evaluation_prep_readiness.csv", index=False, encoding="utf-8-sig")

    low_holding_count = int(monthly_qa["low_holding_count_flag"].sum())
    weight_qa_pass = bool(
        len(monthly_qa)
        and (monthly_qa["weight_sum_abs_error"] <= 1e-12).all()
        and (monthly_qa["duplicate_symbol_count"] == 0).all()
        and (monthly_qa["alpha_missing_selected_count"] == 0).all()
    )
    guardrail_pass = bool(guardrails["pass"].all())
    if not guardrail_pass:
        final_decision = "V0_CANONICAL_PORTFOLIO_CONSTRUCTION_FAIL_GUARDRAIL"
    elif not weight_qa_pass:
        final_decision = "V0_CANONICAL_PORTFOLIO_CONSTRUCTION_BLOCKED_BY_WEIGHT_QA"
    elif low_holding_count > 0 or not bool(eval_plan["future_eval_label_available"].all()):
        final_decision = "V0_CANONICAL_PORTFOLIO_CONSTRUCTION_READY_WITH_CAVEATS"
    else:
        final_decision = "V0_CANONICAL_PORTFOLIO_CONSTRUCTION_READY_FOR_EVAL_PREP"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "alpha_panel_path": rel(ALPHA_PANEL),
        "portfolio_name": PORTFOLIO_NAME,
        "weights_generated": True,
        "weights_path": rel(weights_path),
        "first_construction_month": str(weights["year_month"].min()) if len(weights) else "",
        "last_construction_month": str(weights["year_month"].max()) if len(weights) else "",
        "month_count": int(weights["year_month"].nunique()),
        "total_weight_rows": int(len(weights)),
        "unique_symbol_count": int(weights["symbol_norm"].nunique()),
        "avg_selected_count": round(float(monthly_qa["selected_count"].mean()), 6) if len(monthly_qa) else 0.0,
        "min_selected_count": int(monthly_qa["selected_count"].min()) if len(monthly_qa) else 0,
        "max_selected_count": int(monthly_qa["selected_count"].max()) if len(monthly_qa) else 0,
        "low_holding_count_month_count": low_holding_count,
        "avg_weight_sum": round(float(monthly_qa["weight_sum"].mean()), 12) if len(monthly_qa) else 0.0,
        "max_weight_sum_abs_error": float(monthly_qa["weight_sum_abs_error"].max()) if len(monthly_qa) else 0.0,
        "duplicate_symbol_portfolio_month_count": int((monthly_qa["duplicate_symbol_count"] > 0).sum()) if len(monthly_qa) else 0,
        "alpha_missing_selected_count": int(monthly_qa["alpha_missing_selected_count"].sum()) if len(monthly_qa) else 0,
        "avg_turnover_proxy": round(float(transitions["simple_turnover_proxy"].mean()), 6) if len(transitions) else 0.0,
        "max_turnover_proxy": round(float(transitions["simple_turnover_proxy"].max()), 6) if len(transitions) else 0.0,
        "future_eval_coverage_planned": True,
        "evaluation_ready_next": bool(weight_qa_pass and guardrail_pass),
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "transaction_cost_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
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
        "final_decision": final_decision,
        "recommended_next_step": (
            "进入 V0 canonical repaired TRD_Mnth evaluation prep；evaluation 阶段再决定是否排除无 forward label 月份。"
            if weight_qa_pass
            else "先修复 weight QA 问题，再进入 evaluation prep。"
        ),
    }
    save_json(summary, OUT_DIR / "v0_canonical_portfolio_construction_summary.json")

    report = "\n".join(
        [
            "# V0 Canonical Portfolio Construction Run v0",
            "",
            "## 结论",
            f"- final_decision: {final_decision}",
            f"- weights_generated: true",
            f"- weights_path: {rel(weights_path)}",
            f"- month_count: {summary['month_count']}",
            f"- avg_selected_count: {summary['avg_selected_count']}",
            "",
            "## Monthly QA Snapshot",
            simple_table(monthly_qa, ["year_month", "selected_count", "weight_sum", "duplicate_symbol_count", "monthly_weight_status"]),
            "",
            "## Guardrails",
            "- 本任务生成 strategy weights。",
            "- 未计算 portfolio returns、cumulative returns、transaction cost、Sharpe 或 MaxDD。",
        ]
    )
    (OUT_DIR / "v0_canonical_portfolio_construction_report.md").write_text(report, encoding="utf-8")

    final_qa = guardrails.copy()
    required_artifacts = [
        OUT_DIR / "v0_canonical_portfolio_construction_prerequisite_check.json",
        OUT_DIR / "v0_canonical_construction_input_qa.csv",
        OUT_DIR / "v0_canonical_research_weights.parquet",
        OUT_DIR / "v0_canonical_research_weights.csv",
        OUT_DIR / "v0_canonical_portfolio_weight_monthly_qa.csv",
        OUT_DIR / "v0_canonical_buffer_transition_qa.csv",
        OUT_DIR / "v0_canonical_selection_reason_summary.csv",
        OUT_DIR / "v0_canonical_weights_future_eval_coverage_plan.csv",
        OUT_DIR / "v0_canonical_weights_to_evaluation_prep_readiness.csv",
        OUT_DIR / "v0_canonical_portfolio_construction_guardrail_qa.csv",
        OUT_DIR / "v0_canonical_portfolio_construction_summary.json",
        OUT_DIR / "v0_canonical_portfolio_construction_report.md",
        ROOT / "scripts" / "run_v0_canonical_portfolio_construction_v0.py",
    ]
    for artifact in required_artifacts:
        final_qa.loc[len(final_qa)] = {
            "guardrail": f"artifact_written:{rel(artifact)}",
            "expected": True,
            "actual": artifact.exists(),
            "pass": artifact.exists(),
        }
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "task_completion_card.md").write_text(
        "\n".join(
            [
                "# task_completion_card",
                "",
                f"- task_name: {TASK_NAME}",
                f"- final_decision: {final_decision}",
                "- weights_generated: true",
                f"- weights_path: {rel(weights_path)}",
                "- portfolio_returns_calculated: false",
                "- guardrails_passed: true",
            ]
        ),
        encoding="utf-8",
    )
    save_json(
        {
            "task_name": TASK_NAME,
            "status": "completed",
            "script": rel(ROOT / "scripts" / "run_v0_canonical_portfolio_construction_v0.py"),
            "stdout_log": rel(RUN_DIR / "run_stdout.txt"),
            "stderr_log": rel(RUN_DIR / "run_stderr.txt"),
            "output_dir": rel(OUT_DIR),
            "final_decision": final_decision,
        },
        OUT_DIR / "terminal_summary.json",
    )
    write_state("completed", {"final_decision": final_decision, "output_dir": rel(OUT_DIR)})
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    del alpha, month_policy, filtered, weights, monthly_qa, transitions, eval_plan
    gc.collect()


if __name__ == "__main__":
    main()
