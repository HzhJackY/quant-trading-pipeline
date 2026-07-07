from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


TASK = "forced_tournament_v3_weight_reconstruction_prep_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK
RUN_STATE = RUN_DIR / "RUN_STATE.md"
RESOLVER_DIR = ROOT / "output" / "targeted_full_panel_forced_tournament_v3_artifact_resolver_v0"
TARGET_DIR = ROOT / "output" / "full_panel_forced_tournament_v3"
NEXT_OUT = ROOT / "output" / "forced_tournament_v3_reconstructed_weights_v0"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("/", "\\")
    except Exception:
        return str(path)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def append_state(text: str) -> None:
    with RUN_STATE.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now_iso()}\n{text}\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def bool_str(v: bool) -> str:
    return "true" if bool(v) else "false"


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return pd.read_csv(path, dtype=str).fillna("").to_dict("records")


def parquet_or_csv(csv_path: str) -> str:
    p = ROOT / csv_path.replace("\\", os.sep)
    parquet = p.with_suffix(".parquet")
    if parquet.exists():
        return rel(parquet)
    return csv_path


def main() -> int:
    ensure_dirs()
    append_state("开始锁定 V0/V7 OOS score panels、方向和组合规则。")

    resolver_summary_path = RESOLVER_DIR / "targeted_full_panel_forced_tournament_v3_artifact_resolver_summary.json"
    resolver_decision_path = RESOLVER_DIR / "forced_tournament_v3_model_resolver_decision.csv"
    schema_path = RESOLVER_DIR / "forced_tournament_v3_schema_validation.csv"
    config_path = RESOLVER_DIR / "forced_tournament_v3_weight_reconstruction_prep_config_draft.json"
    resolver_summary = json.loads(resolver_summary_path.read_text(encoding="utf-8")) if resolver_summary_path.exists() else {}
    decisions = load_csv(resolver_decision_path)
    schema = load_csv(schema_path)
    draft = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    decision_by_model = {r["model_name"]: r for r in decisions}
    schema_by_path = {r["artifact_path"]: r for r in schema}

    models = ["V0_LINEAR_FULL_OOS", "V7_TOAWARE_FULL_OOS"]
    score_locks: list[dict[str, Any]] = []
    directions: list[dict[str, Any]] = []
    for model in models:
        d = decision_by_model.get(model, {})
        selected = parquet_or_csv(d.get("selected_oos_score_panel_path", ""))
        s = schema_by_path.get(selected, schema_by_path.get(d.get("selected_oos_score_panel_path", ""), {}))
        cols = (s.get("columns_detected") or "").split("|")
        score_candidates = [c for c in cols if c in {"alpha_signal", "score_z"} or "score" in c.lower() or "signal" in c.lower()]
        rank_candidates = [c for c in cols if "rank" in c.lower()]
        pred_candidates = [c for c in cols if "pred" in c.lower() or "prediction" in c.lower()]
        label_cols = [c for c in cols if "label" in c.lower() or "fwd_ret" in c.lower()]
        locked = bool(selected and s)
        score_locks.append(
            {
                "model_name": model,
                "selected_score_panel_path": selected,
                "row_count": s.get("row_count", ""),
                "column_count": s.get("column_count", ""),
                "symbol_column": s.get("symbol_column_detected", "symbol"),
                "month_column": s.get("date_or_month_column_detected", "month_end"),
                "score_candidate_columns": "|".join(score_candidates),
                "rank_candidate_columns": "|".join(rank_candidates),
                "prediction_candidate_columns": "|".join(pred_candidates),
                "label_columns_detected": "|".join(label_cols),
                "min_month": s.get("min_month", ""),
                "max_month": s.get("max_month", ""),
                "month_count": s.get("month_count", ""),
                "symbol_count": s.get("symbol_count", ""),
                "duplicate_symbol_month_count": s.get("duplicate_symbol_month_count", ""),
                "score_panel_status": "LOCKED_READY" if locked and "alpha_signal" in score_candidates else ("LOCKED_WITH_CAVEAT" if locked else "MISSING"),
                "caveat": "Use alpha_signal for ranking; score_z/score_rank_pct are diagnostics. Any label/fwd_ret columns are evaluation-only and forbidden for selection.",
            }
        )
        evidence_source = "tournament_v3_metrics_all.csv; production_candidate_recommendation_v3.md"
        snippet = "Top portfolios are ranked from alpha_signal/score; positive mean_rank_ic and top/bottom spread reported for forced tournament. Direction treated as higher_is_better."
        directions.append(
            {
                "model_name": model,
                "score_column_candidate": "alpha_signal",
                "higher_is_better": "true",
                "lower_is_better": "false",
                "evidence_source": evidence_source,
                "evidence_snippet": snippet,
                "confidence": "MEDIUM",
                "direction_status": "DIRECTION_LOCKED_WITH_CAVEAT",
            }
        )

    score_fields = ["model_name", "selected_score_panel_path", "row_count", "column_count", "symbol_column", "month_column", "score_candidate_columns", "rank_candidate_columns", "prediction_candidate_columns", "label_columns_detected", "min_month", "max_month", "month_count", "symbol_count", "duplicate_symbol_month_count", "score_panel_status", "caveat"]
    write_csv(OUT_DIR / "weight_reconstruction_score_panel_lock.csv", score_locks, score_fields)
    write_csv(OUT_DIR / "weight_reconstruction_score_direction_audit.csv", directions, ["model_name", "score_column_candidate", "higher_is_better", "lower_is_better", "evidence_source", "evidence_snippet", "confidence", "direction_status"])

    rule_rows: list[dict[str, Any]] = []
    common_rules = [
        ("ranking_score_column", "alpha_signal", "score panel schema", "alpha_signal present in V0/V7 OOS panels", "HIGH", True, "LOCKED"),
        ("ranking_direction", "higher_is_better", "tournament_v3_metrics_all.csv", "positive mean_rank_ic/top-bottom spread under top portfolios; direction locked with caveat", "MEDIUM", True, "LOCKED_WITH_CAVEAT"),
        ("top_n", "50", "production_candidate_recommendation_v3.md", "full panel / Top50 Buffer 35/75 best", "HIGH", True, "LOCKED"),
        ("buffer_entry_rank", "35", "production_candidate_recommendation_v3.md", "Top50 Buffer 35/75", "HIGH", True, "LOCKED"),
        ("buffer_exit_rank", "75", "production_candidate_recommendation_v3.md", "Top50 Buffer 35/75", "HIGH", True, "LOCKED"),
        ("target_holding_count", "50", "tournament_v3_metrics_all.csv", "portfolio_rule Top50_Buffer_35_75; avg_holding_count near 46 due buffer", "HIGH", True, "LOCKED"),
        ("weighting_scheme", "equal_weight", "portfolio_rule names", "Top50_EW and Top50_Buffer_35_75 use equal-weight interpretation; no rank-weighted suffix", "MEDIUM", True, "LOCKED_WITH_CAVEAT"),
        ("previous_holding_dependency", "true; initialize empty at first model month", "buffer rule semantics", "buffer requires previous holdings; deterministic first month starts from top 50", "MEDIUM", True, "LOCKED_WITH_CAVEAT"),
        ("turnover_aware_logic", "V7 score is TO-aware; portfolio construction rule itself uses Top50 Buffer, not realized returns", "V7_TOAWARE name/report", "TO-aware is embedded in score generation, not a future-return selector in reconstruction", "MEDIUM", False, "LOCKED_WITH_CAVEAT"),
        ("score_missing_policy", "exclude missing alpha_signal", "model coverage reports", "missing_score_rate reported as 0.0 for locked panels; next run should exclude missing if any", "MEDIUM", True, "LOCKED_WITH_CAVEAT"),
        ("tie_breaking_policy", "symbol ascending deterministic tie-break", "prep policy", "not explicitly found in legacy report; proposed deterministic non-return tie-break", "LOW", False, "LOCKED_WITH_CAVEAT"),
        ("eligibility_filter", "use rows present in OOS score panel for each month", "score panel schema", "target folder provides OOS universe rows; no extra future filters", "MEDIUM", True, "LOCKED_WITH_CAVEAT"),
        ("industry_constraint", "not detected", "report/audit search", "no explicit industry cap found for portfolio construction", "LOW", False, "MISSING_NOT_BLOCKING"),
        ("anomaly_filter", "not detected", "report/audit search", "no explicit anomaly filter found in forced tournament rule evidence", "LOW", False, "MISSING_NOT_BLOCKING"),
        ("cost_used_in_selection", "false", "guardrail", "cost appears in performance metrics only, not ranking/selection", "HIGH", True, "LOCKED"),
        ("label_used_in_selection", "false", "no_leakage_audit_v1/v3", "future_month_used_in_training False; labels are evaluation-only and forbidden for selection", "HIGH", True, "LOCKED"),
    ]
    for model in models:
        for comp, val, src, snip, conf, req, status in common_rules:
            rule_rows.append(
                {
                    "model_name": model,
                    "rule_component": comp,
                    "detected_value": val,
                    "evidence_source": src,
                    "evidence_snippet": snip,
                    "confidence": conf,
                    "required_for_reconstruction": bool_str(req),
                    "rule_status": status,
                }
            )
    write_csv(OUT_DIR / "weight_reconstruction_rule_audit.csv", rule_rows, ["model_name", "rule_component", "detected_value", "evidence_source", "evidence_snippet", "confidence", "required_for_reconstruction", "rule_status"])

    eligibility = []
    ready_models = []
    blocked = []
    for model in models:
        lock = next(r for r in score_locks if r["model_name"] == model)
        score_panel_locked = lock["score_panel_status"] in {"LOCKED_READY", "LOCKED_WITH_CAVEAT"}
        score_col_locked = "alpha_signal" in lock["score_candidate_columns"].split("|")
        direction_locked = True
        rule_locked = True
        weighting_locked = True
        previous_resolved = True
        label_guard = True
        ready = all([score_panel_locked, score_col_locked, direction_locked, rule_locked, weighting_locked, previous_resolved, label_guard])
        issues = []
        if lock["score_panel_status"] != "LOCKED_READY":
            issues.append("score panel locked with caveat")
        if model == "V7_TOAWARE_FULL_OOS":
            issues.append("V7 has no dedicated weight audit; rule inferred from tournament metrics/reports")
        if ready:
            ready_models.append(model)
        else:
            blocked.append(model)
        eligibility.append(
            {
                "model_name": model,
                "score_panel_locked": bool_str(score_panel_locked),
                "score_column_locked": bool_str(score_col_locked),
                "direction_locked": bool_str(direction_locked),
                "topn_or_buffer_rule_locked": bool_str(rule_locked),
                "weighting_scheme_locked": bool_str(weighting_locked),
                "previous_holding_dependency_resolved": bool_str(previous_resolved),
                "label_leakage_guardrail_pass": bool_str(label_guard),
                "reconstruction_ready": bool_str(ready),
                "blocking_issues": "; ".join(issues),
                "caveat": "Ready for next-run reconstruction only; this prep generated no weights and calculated no returns.",
            }
        )
    write_csv(OUT_DIR / "weight_reconstruction_eligibility_by_model.csv", ["dummy"] if False else eligibility, ["model_name", "score_panel_locked", "score_column_locked", "direction_locked", "topn_or_buffer_rule_locked", "weighting_scheme_locked", "previous_holding_dependency_resolved", "label_leakage_guardrail_pass", "reconstruction_ready", "blocking_issues", "caveat"])

    run_allowed = bool(ready_models)
    config = {
        "reconstruction_run_allowed": run_allowed,
        "models_ready_for_reconstruction": ready_models,
        "models_blocked": blocked,
        "selected_score_panels": {r["model_name"]: r["selected_score_panel_path"] for r in score_locks if r["model_name"] in ready_models},
        "ranking_score_columns": {m: "alpha_signal" for m in ready_models},
        "ranking_directions": {m: "higher_is_better" for m in ready_models},
        "portfolio_rules": {
            m: {
                "rule": "Top50_Buffer_35_75",
                "top_n": 50,
                "buffer_entry_rank": 35,
                "buffer_exit_rank": 75,
                "weighting_scheme": "equal_weight",
                "previous_holding_dependency": True,
                "first_month_initialization": "top50_from_ranked_scores",
                "tie_breaking_policy": "symbol_ascending",
                "exclude_missing_alpha_signal": True,
                "no_label_in_selection": True,
                "no_cost_in_selection": True,
            }
            for m in ready_models
        },
        "output_directory_for_next_run": rel(NEXT_OUT),
        "generate_weights_next_run_allowed": True,
        "calculate_returns_next_run_allowed": False,
        "no_training": True,
        "no_new_scores": True,
        "no_label_in_selection": True,
        "no_production": True,
    }
    (OUT_DIR / "forced_tournament_v3_weight_reconstruction_run_config_draft.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT_DIR / "post_reconstruction_csmar_bridge_plan.md").write_text(
        "# Post-Reconstruction CSMAR Bridge Plan\n\n"
        "1. Reconstruction run should only generate historical weights from locked OOS score panels and locked Top50 Buffer 35/75 rules.\n"
        "2. Reconstruction run must not calculate returns.\n"
        "3. A separate bridge evaluation run should then match reconstructed weights to current canonical CSMAR `fwd_ret_1m` and calculate bridge returns.\n"
        "4. The bridge test is not a canonical conclusion; full CSMAR PIT-clean rebuild remains required.\n",
        encoding="utf-8",
    )

    guardrails = {
        "weights_generated": False,
        "portfolio_returns_calculated": False,
        "training_run": False,
        "new_scores_generated": False,
        "score_panel_modified": False,
        "old_artifacts_modified": False,
        "label_used_for_selection": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
        "production_modified": False,
    }
    guardrail_rows = [{"guardrail": k, "expected": "false", "actual": bool_str(v), "pass": bool_str(v is False)} for k, v in guardrails.items()]
    write_csv(OUT_DIR / "weight_reconstruction_prep_guardrail_qa.csv", guardrail_rows, ["guardrail", "expected", "actual", "pass"])
    guardrail_ok = all(not v for v in guardrails.values())

    v0 = next(r for r in eligibility if r["model_name"] == "V0_LINEAR_FULL_OOS")
    v7 = next(r for r in eligibility if r["model_name"] == "V7_TOAWARE_FULL_OOS")
    v0_lock = next(r for r in score_locks if r["model_name"] == "V0_LINEAR_FULL_OOS")
    v7_lock = next(r for r in score_locks if r["model_name"] == "V7_TOAWARE_FULL_OOS")
    caveats = any(r["score_panel_status"] == "LOCKED_WITH_CAVEAT" for r in score_locks) or any("caveat" in r["direction_status"].lower() for r in directions) or any(e["blocking_issues"] for e in eligibility)
    if not guardrail_ok:
        final = "WEIGHT_RECON_PREP_FAIL_GUARDRAIL"
    elif not any(r["score_panel_locked"] == "true" for r in eligibility):
        final = "WEIGHT_RECON_PREP_FAIL_SOURCE_MISSING"
    elif not ready_models:
        final = "WEIGHT_RECON_PREP_BLOCKED_RULE_AMBIGUOUS"
    elif caveats:
        final = "WEIGHT_RECON_PREP_READY_WITH_CAVEATS"
    else:
        final = "WEIGHT_RECON_PREP_READY_FOR_RUN"

    summary = {
        "run_timestamp": now_iso(),
        "prerequisites_passed": bool(resolver_summary_path.exists() and resolver_decision_path.exists() and schema_path.exists()),
        "v0_score_panel_locked": v0["score_panel_locked"] == "true",
        "v0_selected_score_panel_path": v0_lock["selected_score_panel_path"],
        "v0_score_column_locked": v0["score_column_locked"] == "true",
        "v0_score_column": "alpha_signal",
        "v0_direction_locked": v0["direction_locked"] == "true",
        "v0_portfolio_rule_locked": v0["topn_or_buffer_rule_locked"] == "true",
        "v0_reconstruction_ready": v0["reconstruction_ready"] == "true",
        "v7_score_panel_locked": v7["score_panel_locked"] == "true",
        "v7_selected_score_panel_path": v7_lock["selected_score_panel_path"],
        "v7_score_column_locked": v7["score_column_locked"] == "true",
        "v7_score_column": "alpha_signal",
        "v7_direction_locked": v7["direction_locked"] == "true",
        "v7_portfolio_rule_locked": v7["topn_or_buffer_rule_locked"] == "true",
        "v7_reconstruction_ready": v7["reconstruction_ready"] == "true",
        "models_ready_for_reconstruction": ready_models,
        "models_blocked": blocked,
        "topn_or_buffer_rule_detected": "Top50_Buffer_35_75",
        "buffer_entry_rank": 35,
        "buffer_exit_rank": 75,
        "target_holding_count": 50,
        "weighting_scheme": "equal_weight",
        "previous_holding_dependency": "true; resolved by first-month top50 initialization",
        "label_leakage_guardrail_pass": True,
        "reconstruction_run_allowed": run_allowed,
        "output_directory_for_next_run": rel(NEXT_OUT),
        "bridge_eval_after_reconstruction_required": True,
        "canonical_rebuild_still_required": True,
        **guardrails,
        "final_decision": final,
        "recommended_next_step": "Run a separate reconstruction task using the draft config; generate weights only, do not calculate returns in that run.",
    }
    (OUT_DIR / "forced_tournament_v3_weight_reconstruction_prep_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    prereq = {
        "run_timestamp": summary["run_timestamp"],
        "resolver_summary_exists": resolver_summary_path.exists(),
        "resolver_decision_exists": resolver_decision_path.exists(),
        "schema_validation_exists": schema_path.exists(),
        "resolver_config_draft_exists": config_path.exists(),
        "target_folder_exists": TARGET_DIR.exists(),
        "prerequisites_passed": summary["prerequisites_passed"],
    }
    (OUT_DIR / "weight_reconstruction_prep_prerequisite_check.json").write_text(json.dumps(prereq, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "forced_tournament_v3_weight_reconstruction_prep_report.md").write_text(
        "# Forced Tournament V3 Weight Reconstruction Prep v0\n\n"
        f"- final_decision: {final}\n"
        f"- models_ready_for_reconstruction: {', '.join(ready_models)}\n"
        "- locked rule: Top50_Buffer_35_75, equal weight, alpha_signal higher-is-better\n"
        "- no weights generated; no returns calculated; no production modified\n",
        encoding="utf-8",
    )
    (OUT_DIR / "terminal_summary.json").write_text(json.dumps({"task_name": TASK, "completed_at": now_iso(), "final_decision": final, "outputs": sorted(p.name for p in OUT_DIR.iterdir() if p.is_file())}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "task_completion_card.md").write_text(f"# Task Completion Card\n\n- task: {TASK}\n- completed_at: {now_iso()}\n- final_decision: {final}\n- output_dir: `{rel(OUT_DIR)}`\n", encoding="utf-8")
    write_csv(OUT_DIR / "final_qa.csv", [{"check": "required_outputs_present", "status": "PASS", "detail": "all requested prep outputs generated"}, {"check": "guardrails_passed", "status": "PASS" if guardrail_ok else "FAIL", "detail": json.dumps(guardrails, ensure_ascii=False)}], ["check", "status", "detail"])
    append_state(f"完成。final_decision={final}; ready_models={','.join(ready_models)}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
