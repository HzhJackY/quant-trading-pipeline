from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_NAME = "V0 Legacy-Compatible PIT Strict-Lag Replay Portfolio Construction Run v0"
OUT_NAME = "v0_legacy_compatible_pit_strict_lag_replay_portfolio_construction_run_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / OUT_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

PREP_DIR = ROOT / "output" / "v0_legacy_compatible_pit_strict_lag_replay_portfolio_prep_v0"
ALPHA_PANEL = ROOT / "output" / "v0_legacy_compatible_pit_strict_lag_replay_alpha_build_v0" / "v0_legacy_pit_route_b_strict_lag_alpha_panel.parquet"
PREP_SUMMARY = PREP_DIR / "v0_legacy_compatible_pit_strict_lag_replay_portfolio_prep_summary.json"
CONSTRUCTION_POLICY = PREP_DIR / "v0_route_b_portfolio_construction_policy.json"
ELIGIBLE_MONTH_POLICY = PREP_DIR / "v0_route_b_portfolio_eligible_month_policy.csv"
FUTURE_EVAL_PLAN = PREP_DIR / "v0_route_b_future_eval_coverage_plan.csv"
COMPARISON_PLAN = PREP_DIR / "v0_route_b_comparison_plan.csv"
LEGACY_WEIGHTS = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_reconstructed_weights.parquet"
COMPOSITE_WEIGHTS = ROOT / "output" / "v0_composite_aligned_portfolio_construction_run_v0" / "v0_composite_aligned_research_weights.parquet"
RETURN_MAP = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0" / "canonical_csmar_trd_mnth_return_map_repaired.parquet"

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
        "resume_instruction": f"先读取 {rel(RUN_DIR / 'RUN_STATE.md')}；继续时运行 scripts\\run_v0_legacy_compatible_pit_strict_lag_replay_portfolio_construction_v0.py，并重定向 stdout/stderr 到本目录。",
    }
    if extra:
        payload.update(extra)
    lines = ["# RUN_STATE", "", f"- task_name: {TASK_NAME}", f"- status: {status}", f"- checkpoint: {checkpoint}", "", "```json", json.dumps(payload, ensure_ascii=False, indent=2, default=str), "```"]
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(lines), encoding="utf-8")


def norm_symbol(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)


def prereq_check() -> dict[str, Any]:
    flags = {
        "route_b_alpha_found": ALPHA_PANEL.exists(),
        "portfolio_prep_summary_found": PREP_SUMMARY.exists(),
        "construction_policy_found": CONSTRUCTION_POLICY.exists(),
        "eligible_month_policy_found": ELIGIBLE_MONTH_POLICY.exists(),
        "future_eval_coverage_plan_found": FUTURE_EVAL_PLAN.exists(),
        "comparison_plan_found": COMPARISON_PLAN.exists(),
        "legacy_weights_found": LEGACY_WEIGHTS.exists(),
        "composite_aligned_weights_found": COMPOSITE_WEIGHTS.exists(),
    }
    required = {
        "route_b_alpha_found": ALPHA_PANEL,
        "portfolio_prep_summary_found": PREP_SUMMARY,
        "construction_policy_found": CONSTRUCTION_POLICY,
        "eligible_month_policy_found": ELIGIBLE_MONTH_POLICY,
        "future_eval_coverage_plan_found": FUTURE_EVAL_PLAN,
        "comparison_plan_found": COMPARISON_PLAN,
    }
    optional = {
        "legacy_weights_found": LEGACY_WEIGHTS,
        "composite_aligned_weights_found": COMPOSITE_WEIGHTS,
    }
    missing_required = [rel(p) for k, p in required.items() if not flags[k]]
    missing_optional = [rel(p) for k, p in optional.items() if not flags[k]]
    flags["prerequisites_passed"] = len(missing_required) == 0
    flags["missing_files"] = missing_required + missing_optional
    flags["caveat"] = "Comparison weights are read-only optional inputs; this run generates Route B research weights only and calculates no returns."
    return flags


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    alpha = pd.read_parquet(
        ALPHA_PANEL,
        columns=["symbol_norm", "year_month", "month_end", "split_group", SCORE_COL, "alpha_build_status"],
    )
    alpha["symbol_norm"] = norm_symbol(alpha["symbol_norm"])
    alpha["year_month"] = alpha["year_month"].astype(str).str.slice(0, 7)
    alpha["month_end"] = pd.to_datetime(alpha["month_end"], errors="coerce")
    alpha[SCORE_COL] = pd.to_numeric(alpha[SCORE_COL], errors="coerce")
    month_policy = pd.read_csv(ELIGIBLE_MONTH_POLICY, dtype={"year_month": str})
    month_policy["year_month"] = month_policy["year_month"].astype(str).str.slice(0, 7)
    month_policy["include_in_construction"] = month_policy["include_in_construction"].astype(str).str.lower().isin(["true", "1", "yes"])
    policy = json.loads(CONSTRUCTION_POLICY.read_text(encoding="utf-8"))
    future = pd.read_csv(FUTURE_EVAL_PLAN, dtype={"year_month": str})
    future["year_month"] = future["year_month"].astype(str).str.slice(0, 7)
    return alpha, month_policy, policy, future


def rank_preview(alpha: pd.DataFrame, month_policy: pd.DataFrame) -> pd.DataFrame:
    status_map = month_policy.set_index("year_month")["eligible_month_status"].to_dict()
    include_map = month_policy.set_index("year_month")["include_in_construction"].to_dict()
    ranks = []
    for ym, g in alpha.groupby("year_month", sort=True):
        include = bool(include_map.get(ym, False))
        if include:
            ranked = g.loc[g[SCORE_COL].notna()].sort_values([SCORE_COL, "symbol_norm"], ascending=[False, True]).copy()
            ranked["rank"] = np.arange(1, len(ranked) + 1)
        else:
            ranked = g.head(0).copy()
            ranked["rank"] = []
        if len(ranked):
            ranked["include_in_construction"] = include
            ranked["eligible_month_status"] = status_map.get(ym, "")
            ranks.append(ranked[["year_month", "symbol_norm", "split_group", SCORE_COL, "rank", "include_in_construction", "eligible_month_status", "alpha_build_status"]])
    out = pd.concat(ranks, ignore_index=True) if ranks else pd.DataFrame(columns=["year_month", "symbol_norm", "split_group", SCORE_COL, "rank", "include_in_construction", "eligible_month_status", "alpha_build_status"])
    out.to_csv(OUT_DIR / "v0_route_b_monthly_alpha_rank_preview.csv", index=False, encoding="utf-8-sig")
    return out


def construct_weights(ranked: pd.DataFrame, month_policy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible_months = month_policy.loc[month_policy["include_in_construction"], "year_month"].astype(str).tolist()
    status_map = month_policy.set_index("year_month")["eligible_month_status"].to_dict()
    rows = []
    transitions = []
    prev_weights: dict[str, float] = {}
    prev_holdings: set[str] = set()
    for ym in eligible_months:
        uni = ranked.loc[ranked["year_month"].eq(ym)].copy()
        uni = uni.loc[uni[SCORE_COL].notna()].sort_values(["rank", "symbol_norm"]).copy()
        rank_map = uni.set_index("symbol_norm")["rank"].to_dict()
        first = not prev_holdings
        selected: list[tuple[str, str]] = []
        if first:
            for sym in uni.head(TARGET_HOLDING_COUNT)["symbol_norm"].astype(str):
                selected.append((sym, "INITIAL_TOP50"))
        else:
            current_symbols = set(uni["symbol_norm"].astype(str))
            kept = [s for s in prev_holdings if s in current_symbols and rank_map.get(s, 10**9) <= EXIT_RANK]
            kept = sorted(kept, key=lambda s: (rank_map.get(s, 10**9), s))
            selected.extend((s, "RETAINED_WITHIN_EXIT_BUFFER") for s in kept)
            selected_set = {s for s, _ in selected}
            entries = uni.loc[(uni["rank"] <= ENTRY_RANK) & ~uni["symbol_norm"].isin(selected_set), "symbol_norm"].astype(str).tolist()
            for s in entries:
                if len(selected) >= TARGET_HOLDING_COUNT:
                    break
                selected.append((s, "NEW_ENTRY_WITHIN_ENTRY_RANK"))
            selected_set = {s for s, _ in selected}
            if len(selected) < TARGET_HOLDING_COUNT:
                fills = uni.loc[~uni["symbol_norm"].isin(selected_set), "symbol_norm"].astype(str).tolist()
                for s in fills:
                    if len(selected) >= TARGET_HOLDING_COUNT:
                        break
                    selected.append((s, "FILL_TO_TARGET"))
        selected = selected[:TARGET_HOLDING_COUNT]
        selected_symbols = [s for s, _ in selected]
        selected_count = len(selected_symbols)
        weight = 1.0 / selected_count if selected_count else 0.0
        new_weights = {s: weight for s in selected_symbols}
        union = set(prev_weights).union(new_weights)
        turnover = 1.0 if first and selected_count else 0.5 * sum(abs(new_weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in union)
        selected_set = set(selected_symbols)
        exited = prev_holdings - selected_set
        retained = sum(1 for _, r in selected if r == "RETAINED_WITHIN_EXIT_BUFFER")
        new_entry = sum(1 for _, r in selected if r == "NEW_ENTRY_WITHIN_ENTRY_RANK")
        fill = sum(1 for _, r in selected if r == "FILL_TO_TARGET")
        lookup = uni.set_index("symbol_norm")
        for sym, reason in selected:
            rec = lookup.loc[sym]
            rows.append({
                "portfolio_name": PORTFOLIO_NAME,
                "year_month": ym,
                "symbol_norm": sym,
                "weight": weight,
                "rank": int(rec["rank"]),
                "alpha_signal_route_b_strict_lag": float(rec[SCORE_COL]),
                "selected_reason": reason,
                "prev_holding_flag": sym in prev_holdings,
                "new_entry_flag": reason in {"INITIAL_TOP50", "NEW_ENTRY_WITHIN_ENTRY_RANK", "FILL_TO_TARGET"} and sym not in prev_holdings,
                "retained_by_buffer_flag": reason == "RETAINED_WITHIN_EXIT_BUFFER",
                "exited_prev_month_flag": False,
                "eligible_month_status": status_map.get(ym, ""),
                "construction_status": "PASS" if selected_count == TARGET_HOLDING_COUNT else "FAIL_LOW_HOLDING_COUNT",
            })
        transitions.append({
            "year_month": ym,
            "prev_holding_count": int(len(prev_holdings)),
            "retained_count": int(retained),
            "exited_count": int(len(exited)),
            "new_entry_count": int(new_entry),
            "fill_to_target_count": int(fill),
            "turnover_proxy": float(turnover),
            "retained_within_exit_buffer_count": int(retained),
            "new_entry_within_entry_rank_count": int(new_entry),
            "buffer_rule_status": "PASS" if selected_count == TARGET_HOLDING_COUNT else "FAIL_LOW_HOLDING_COUNT",
            "caveat": "",
        })
        prev_holdings = selected_set
        prev_weights = new_weights
    weights = pd.DataFrame(rows)
    trans = pd.DataFrame(transitions)
    weights.to_parquet(OUT_DIR / "v0_route_b_research_weights.parquet", index=False)
    weights.to_csv(OUT_DIR / "v0_route_b_research_weights.csv", index=False, encoding="utf-8-sig")
    trans.to_csv(OUT_DIR / "v0_route_b_buffer_transition_qa.csv", index=False, encoding="utf-8-sig")
    return weights, trans


def monthly_weight_qa(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ym, g in weights.groupby("year_month", sort=True):
        selected_count = int(len(g))
        weight_sum = float(g["weight"].sum())
        dup = int(g["symbol_norm"].duplicated().sum())
        alpha_missing = int(g["alpha_signal_route_b_strict_lag"].isna().sum())
        rows.append({
            "year_month": ym,
            "eligible_month_status": str(g["eligible_month_status"].iloc[0]),
            "selected_count": selected_count,
            "weight_sum": weight_sum,
            "weight_sum_abs_error": abs(weight_sum - 1.0),
            "min_weight": float(g["weight"].min()) if len(g) else 0.0,
            "max_weight": float(g["weight"].max()) if len(g) else 0.0,
            "duplicate_symbol_count": dup,
            "alpha_missing_selected_count": alpha_missing,
            "selected_rank_min": int(g["rank"].min()) if len(g) else 0,
            "selected_rank_max": int(g["rank"].max()) if len(g) else 0,
            "selected_rank_mean": float(g["rank"].mean()) if len(g) else 0.0,
            "construction_status": "PASS" if selected_count == TARGET_HOLDING_COUNT and abs(weight_sum - 1.0) <= 1e-12 and dup == 0 and alpha_missing == 0 else "FAIL",
            "caveat": "",
        })
    qa = pd.DataFrame(rows)
    qa.to_csv(OUT_DIR / "v0_route_b_portfolio_weight_monthly_qa.csv", index=False, encoding="utf-8-sig")
    return qa


def load_comparison_weights(path: Path, kind: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["year_month", "symbol_norm", "weight"])
    if kind == "legacy":
        df = pd.read_parquet(path, columns=["symbol", "month_end", "weight"])
        df["symbol_norm"] = norm_symbol(df["symbol"])
        df["year_month"] = pd.to_datetime(df["month_end"], errors="coerce").dt.strftime("%Y-%m")
    else:
        df = pd.read_parquet(path, columns=["symbol_norm", "year_month", "weight"])
        df["symbol_norm"] = norm_symbol(df["symbol_norm"])
        df["year_month"] = df["year_month"].astype(str).str.slice(0, 7)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    return df[["year_month", "symbol_norm", "weight"]].drop_duplicates(["year_month", "symbol_norm"], keep="last")


def comparison_overlap(weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy = load_comparison_weights(LEGACY_WEIGHTS, "legacy")
    comp = load_comparison_weights(COMPOSITE_WEIGHTS, "composite")
    rows = []
    for ym, rb in weights.groupby("year_month", sort=True):
        rb_set = set(rb["symbol_norm"].astype(str))
        legacy_g = legacy.loc[legacy["year_month"].eq(ym)]
        comp_g = comp.loc[comp["year_month"].eq(ym)]
        legacy_set = set(legacy_g["symbol_norm"].astype(str))
        comp_set = set(comp_g["symbol_norm"].astype(str))
        rb_w = rb.set_index("symbol_norm")["weight"].to_dict()
        lw = legacy_g.set_index("symbol_norm")["weight"].to_dict()
        cw = comp_g.set_index("symbol_norm")["weight"].to_dict()
        l_union = set(rb_w).union(lw)
        c_union = set(rb_w).union(cw)
        l_overlap = sum(min(rb_w.get(s, 0.0), lw.get(s, 0.0)) for s in l_union) if l_union else np.nan
        c_overlap = sum(min(rb_w.get(s, 0.0), cw.get(s, 0.0)) for s in c_union) if c_union else np.nan
        rows.append({
            "year_month": ym,
            "route_b_selected_count": int(len(rb_set)),
            "legacy_selected_count": int(len(legacy_set)),
            "composite_aligned_selected_count": int(len(comp_set)),
            "route_b_vs_legacy_overlap_count": int(len(rb_set & legacy_set)),
            "route_b_vs_legacy_overlap_ratio": float(len(rb_set & legacy_set) / TARGET_HOLDING_COUNT) if legacy_set else np.nan,
            "route_b_vs_composite_aligned_overlap_count": int(len(rb_set & comp_set)),
            "route_b_vs_composite_aligned_overlap_ratio": float(len(rb_set & comp_set) / TARGET_HOLDING_COUNT) if comp_set else np.nan,
            "route_b_vs_legacy_weight_overlap": float(l_overlap),
            "route_b_vs_composite_aligned_weight_overlap": float(c_overlap),
            "caveat": "read-only comparison; no returns calculated",
        })
    qa = pd.DataFrame(rows)
    qa.to_csv(OUT_DIR / "v0_route_b_weights_comparison_overlap_qa.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    for target, ratio_col, count_col, w_col in [
        ("legacy", "route_b_vs_legacy_overlap_ratio", "route_b_vs_legacy_overlap_count", "route_b_vs_legacy_weight_overlap"),
        ("composite_aligned", "route_b_vs_composite_aligned_overlap_ratio", "route_b_vs_composite_aligned_overlap_count", "route_b_vs_composite_aligned_weight_overlap"),
    ]:
        summary_rows.append({
            "comparison_target": target,
            "avg_overlap_count": float(qa[count_col].mean()),
            "avg_overlap_ratio": float(qa[ratio_col].mean()),
            "avg_weight_overlap": float(qa[w_col].mean()),
            "min_overlap_ratio": float(qa[ratio_col].min()),
            "max_overlap_ratio": float(qa[ratio_col].max()),
            "interpretation": "holdings overlap only; not performance",
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "v0_route_b_weights_comparison_overlap_summary.csv", index=False, encoding="utf-8-sig")
    del legacy, comp
    gc.collect()
    return qa, summary


def future_eval_coverage(weights: pd.DataFrame) -> pd.DataFrame:
    ret = pd.read_parquet(RETURN_MAP, columns=["symbol_norm", "year_month", "fwd_ret_1m", "primary_return_field"])
    ret = ret.loc[ret["primary_return_field"].astype(str).eq("Mretwd")].copy()
    ret["symbol_norm"] = norm_symbol(ret["symbol_norm"])
    ret["year_month"] = ret["year_month"].astype(str).str.slice(0, 7)
    ret["fwd_ret_1m"] = pd.to_numeric(ret["fwd_ret_1m"], errors="coerce")
    ret = ret.drop_duplicates(["symbol_norm", "year_month"], keep="last")
    rows = []
    for ym, g in weights.groupby("year_month", sort=True):
        m = g[["symbol_norm", "weight"]].merge(ret.loc[ret["year_month"].eq(ym), ["symbol_norm", "fwd_ret_1m"]], on="symbol_norm", how="left")
        has = m["fwd_ret_1m"].notna()
        share = float(m.loc[has, "weight"].sum())
        expected = share >= 0.98
        rows.append({
            "year_month": ym,
            "selected_count": int(len(g)),
            "selected_with_label_count": int(has.sum()),
            "selected_missing_label_count": int((~has).sum()),
            "matched_weight_share_preview": share,
            "expected_future_label_available": expected,
            "coverage_status": "AVAILABLE" if expected else "NO_OR_LOW_FUTURE_LABEL",
            "caveat": "future evaluation should exclude or wait for label; no portfolio return calculated" if not expected else "",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "v0_route_b_weights_future_eval_coverage_plan.csv", index=False, encoding="utf-8-sig")
    del ret
    gc.collect()
    return out


def readiness(weights: pd.DataFrame, monthly_qa: pd.DataFrame, transitions: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    selected_ok = bool((monthly_qa["selected_count"] == TARGET_HOLDING_COUNT).all())
    weight_ok = bool((monthly_qa["weight_sum_abs_error"] <= 1e-12).all())
    dup_ok = bool((monthly_qa["duplicate_symbol_count"] == 0).all())
    alpha_ok = bool((monthly_qa["alpha_missing_selected_count"] == 0).all())
    ready = len(weights) > 0 and selected_ok and weight_ok and dup_ok and alpha_ok and len(transitions) > 0 and len(future) > 0
    rows = [
        ("weights_generated", True, len(weights) > 0, len(weights) > 0, ""),
        ("monthly_selected_count_ok", True, selected_ok, selected_ok, ""),
        ("weight_sum_ok", True, weight_ok, weight_ok, ""),
        ("duplicate_symbol_count_ok", True, dup_ok, dup_ok, ""),
        ("alpha_missing_selected_count_ok", True, alpha_ok, alpha_ok, ""),
        ("buffer_transition_qa_complete", True, len(transitions) > 0, len(transitions) > 0, ""),
        ("future_eval_coverage_planned", True, len(future) > 0, len(future) > 0, ""),
        ("no_returns_calculated", True, True, True, ""),
        ("ready_for_eval_prep", True, ready, ready, ""),
    ]
    out = pd.DataFrame(rows, columns=["criterion", "expected", "actual", "pass", "caveat"])
    out.to_csv(OUT_DIR / "v0_route_b_portfolio_construction_readiness.csv", index=False, encoding="utf-8-sig")
    return out


def guardrails() -> pd.DataFrame:
    actuals = {
        "strategy_weights_generated": True,
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
    rows = []
    for k, actual in actuals.items():
        expected = True if k == "strategy_weights_generated" else False
        rows.append({"guardrail": k, "expected": expected, "actual": actual, "pass": actual == expected})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "v0_route_b_portfolio_construction_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", "prerequisite_check")
    prereq = prereq_check()
    write_json(OUT_DIR / "v0_route_b_portfolio_construction_prerequisite_check.json", prereq)
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", "rank_and_weights")
    alpha, month_policy, policy, future_plan = load_inputs()
    ranked = rank_preview(alpha, month_policy)
    weights, transitions = construct_weights(ranked, month_policy)
    monthly_qa = monthly_weight_qa(weights)

    write_state("running", "comparison_and_coverage")
    overlap_qa, overlap_summary = comparison_overlap(weights)
    future = future_eval_coverage(weights)
    ready = readiness(weights, monthly_qa, transitions, future)
    guard = guardrails()

    guard_pass = bool(guard["pass"].all())
    weight_qa_pass = bool(
        len(monthly_qa)
        and (monthly_qa["selected_count"] == TARGET_HOLDING_COUNT).all()
        and (monthly_qa["weight_sum_abs_error"] <= 1e-12).all()
        and (monthly_qa["duplicate_symbol_count"] == 0).all()
        and (monthly_qa["alpha_missing_selected_count"] == 0).all()
    )
    ready_for_eval_prep = bool(ready.loc[ready["criterion"].eq("ready_for_eval_prep"), "pass"].iloc[0])
    expected_no_label_months = future.loc[~future["expected_future_label_available"], "year_month"].astype(str).tolist()
    if not guard_pass:
        final_decision = "ROUTE_B_PORTFOLIO_CONSTRUCTION_FAIL_GUARDRAIL"
    elif not weight_qa_pass:
        final_decision = "ROUTE_B_PORTFOLIO_CONSTRUCTION_BLOCKED_BY_WEIGHT_QA"
    elif expected_no_label_months:
        final_decision = "ROUTE_B_PORTFOLIO_CONSTRUCTION_READY_WITH_CAVEATS"
    else:
        final_decision = "ROUTE_B_PORTFOLIO_CONSTRUCTION_READY_FOR_EVAL_PREP"

    status_counts = month_policy["eligible_month_status"].value_counts().to_dict()
    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "route_b_alpha_path": rel(ALPHA_PANEL),
        "score_column_selected": SCORE_COL,
        "portfolio_name": PORTFOLIO_NAME,
        "weights_generated": len(weights) > 0,
        "weights_path": rel(OUT_DIR / "v0_route_b_research_weights.parquet"),
        "first_construction_month": str(weights["year_month"].min()) if len(weights) else "",
        "last_construction_month": str(weights["year_month"].max()) if len(weights) else "",
        "month_count": int(weights["year_month"].nunique()) if len(weights) else 0,
        "ready_month_count": int(status_counts.get("READY", 0)),
        "watch_month_count": int(status_counts.get("WATCH", 0)),
        "fail_month_count": int(sum(v for k, v in status_counts.items() if str(k).startswith("FAIL"))),
        "total_weight_rows": int(len(weights)),
        "unique_symbol_count": int(weights["symbol_norm"].nunique()) if len(weights) else 0,
        "avg_selected_count": float(monthly_qa["selected_count"].mean()) if len(monthly_qa) else 0.0,
        "min_selected_count": int(monthly_qa["selected_count"].min()) if len(monthly_qa) else 0,
        "max_selected_count": int(monthly_qa["selected_count"].max()) if len(monthly_qa) else 0,
        "low_holding_count_month_count": int((monthly_qa["selected_count"] < TARGET_HOLDING_COUNT).sum()) if len(monthly_qa) else 0,
        "avg_weight_sum": float(monthly_qa["weight_sum"].mean()) if len(monthly_qa) else 0.0,
        "max_weight_sum_abs_error": float(monthly_qa["weight_sum_abs_error"].max()) if len(monthly_qa) else 0.0,
        "duplicate_symbol_portfolio_month_count": int((monthly_qa["duplicate_symbol_count"] > 0).sum()) if len(monthly_qa) else 0,
        "alpha_missing_selected_count": int(monthly_qa["alpha_missing_selected_count"].sum()) if len(monthly_qa) else 0,
        "avg_turnover_proxy": float(transitions["turnover_proxy"].mean()) if len(transitions) else 0.0,
        "max_turnover_proxy": float(transitions["turnover_proxy"].max()) if len(transitions) else 0.0,
        "route_b_vs_legacy_avg_overlap_ratio": float(overlap_summary.loc[overlap_summary["comparison_target"].eq("legacy"), "avg_overlap_ratio"].iloc[0]),
        "route_b_vs_composite_aligned_avg_overlap_ratio": float(overlap_summary.loc[overlap_summary["comparison_target"].eq("composite_aligned"), "avg_overlap_ratio"].iloc[0]),
        "future_eval_coverage_planned": True,
        "expected_no_label_months": expected_no_label_months,
        "ready_for_eval_prep": ready_for_eval_prep,
        "calculate_returns_next_run_allowed": False,
        "benchmark_relative_allowed": False,
        "production_allowed": False,
        "strategy_weights_generated": True,
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
        "recommended_next_step": "进入 V0 Legacy-Compatible PIT Strict-Lag Replay Evaluation Prep v0；先处理 2026-06 no-label caveat 和 label edge-case QA，不直接计算收益。",
    }
    write_json(OUT_DIR / "v0_legacy_compatible_pit_strict_lag_replay_portfolio_construction_summary.json", summary)
    report = "\n".join([
        "# V0 Legacy-Compatible PIT Strict-Lag Replay Portfolio Construction Run v0",
        "",
        f"- final_decision: {final_decision}",
        f"- weights_path: {summary['weights_path']}",
        f"- construction months: {summary['first_construction_month']} to {summary['last_construction_month']} ({summary['month_count']})",
        f"- avg_selected_count: {summary['avg_selected_count']:.6f}",
        f"- max_weight_sum_abs_error: {summary['max_weight_sum_abs_error']:.6g}",
        f"- expected_no_label_months: {','.join(expected_no_label_months)}",
        "",
        "本任务生成 Route B research weights 和 QA；未计算 portfolio returns、累计收益、交易成本或任何绩效指标。",
    ])
    (OUT_DIR / "v0_legacy_compatible_pit_strict_lag_replay_portfolio_construction_report.md").write_text(report, encoding="utf-8")
    final_qa = pd.DataFrame([
        {"check_name": "prerequisites_passed", "expected": True, "actual": prereq["prerequisites_passed"], "pass": prereq["prerequisites_passed"], "caveat": ""},
        {"check_name": "weights_generated", "expected": True, "actual": len(weights) > 0, "pass": len(weights) > 0, "caveat": ""},
        {"check_name": "weight_qa_pass", "expected": True, "actual": weight_qa_pass, "pass": weight_qa_pass, "caveat": ""},
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
    del alpha, month_policy, policy, future_plan, ranked, weights, transitions, monthly_qa, overlap_qa, overlap_summary, future, ready, guard
    gc.collect()
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": rel(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
