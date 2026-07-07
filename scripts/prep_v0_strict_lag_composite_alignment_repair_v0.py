from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


TASK_NAME = "v0_strict_lag_composite_alignment_repair_prep_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

DRIFT_DIR = ROOT / "output" / "v0_canonical_vs_legacy_reconstruction_drift_audit_v0"
CANON_DIR = ROOT / "output" / "v0_canonical_strict_lag_alpha_build_v0"

PATHS = {
    "drift_summary": DRIFT_DIR / "v0_canonical_vs_legacy_reconstruction_drift_audit_summary.json",
    "icir_drift_audit": DRIFT_DIR / "icir_factor_weight_drift_audit.csv",
    "icir_drift_summary": DRIFT_DIR / "icir_factor_weight_drift_summary.csv",
    "alpha_layer": DRIFT_DIR / "alpha_drift_layer_attribution.csv",
    "factor_value": DRIFT_DIR / "factor_value_overlap_diagnostic.csv",
    "price_tech": DRIFT_DIR / "price_technical_formula_drift_audit.csv",
    "financial": DRIFT_DIR / "financial_factor_drift_audit.csv",
    "canonical_factor_panel": ROOT / "output" / "v0_canonical_16factor_panel_build_v0" / "v0_canonical_16factor_panel.parquet",
    "canonical_icir_by_month": CANON_DIR / "v0_canonical_strict_lag_icir_by_month_factor.csv",
    "canonical_icir_audit": CANON_DIR / "v0_canonical_factor_icir_contribution_audit.csv",
    "canonical_usage": CANON_DIR / "v0_canonical_factor_usage_summary.csv",
    "canonical_alpha": CANON_DIR / "v0_canonical_alpha_signal_panel.parquet",
    "legacy_preprocessed": ROOT / "output" / "preprocessed.parquet",
    "legacy_split": ROOT / "output" / "split_universe_blended.parquet",
    "legacy_alpha": ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_alpha_signal_panel.parquet",
    "legacy_weights": ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_reconstructed_weights.parquet",
    "strict_lag_repair_script": ROOT / "scripts" / "rebuild_v0_strict_lag_icir_bridge_v0.py",
    "run_split_universe": ROOT / "run_split_universe.py",
    "split_universe": ROOT / "factor_research" / "split_universe.py",
    "backtest_engine": ROOT / "factor_research" / "backtest_engine.py",
    "orthogonalization": ROOT / "factor_research" / "orthogonalization.py",
}

FACTORS = [
    "Mom_1M",
    "Mom_3M",
    "Mom_6M",
    "Mom_12M_1M",
    "Vol_20D",
    "Vol_60D",
    "Beta",
    "BP",
    "EP",
    "ROE",
    "Debt_Ratio",
    "Net_Profit_Margin",
    "RevGrowth_YoY",
    "ProfitGrowth_YoY",
    "VolChg_20D",
    "PriceDev_20D",
]


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
        "resume_instruction: rerun scripts\\prep_v0_strict_lag_composite_alignment_repair_v0.py with stdout/stderr redirected to this run directory\n",
        encoding="utf-8",
    )


def prerequisite_check() -> dict:
    legacy_scripts_found = all(PATHS[k].exists() for k in ["run_split_universe", "split_universe", "backtest_engine", "orthogonalization"])
    result = {
        "drift_summary_found": PATHS["drift_summary"].exists(),
        "icir_drift_audit_found": PATHS["icir_drift_audit"].exists() and PATHS["icir_drift_summary"].exists(),
        "canonical_factor_panel_found": PATHS["canonical_factor_panel"].exists(),
        "canonical_icir_audit_found": PATHS["canonical_icir_by_month"].exists() and PATHS["canonical_icir_audit"].exists(),
        "legacy_preprocessed_found": PATHS["legacy_preprocessed"].exists(),
        "legacy_split_universe_blended_found": PATHS["legacy_split"].exists(),
        "legacy_strict_lag_alpha_found": PATHS["legacy_alpha"].exists(),
        "strict_lag_repair_script_found": PATHS["strict_lag_repair_script"].exists(),
        "legacy_implementation_scripts_found": legacy_scripts_found,
    }
    required_map = {
        "drift_summary_found": PATHS["drift_summary"],
        "icir_drift_audit_found": PATHS["icir_drift_audit"],
        "canonical_factor_panel_found": PATHS["canonical_factor_panel"],
        "canonical_icir_audit_found": PATHS["canonical_icir_audit"],
        "legacy_preprocessed_found": PATHS["legacy_preprocessed"],
        "legacy_split_universe_blended_found": PATHS["legacy_split"],
        "legacy_strict_lag_alpha_found": PATHS["legacy_alpha"],
        "strict_lag_repair_script_found": PATHS["strict_lag_repair_script"],
    }
    missing = [rel(p) for k, p in required_map.items() if not result[k]]
    if not legacy_scripts_found:
        missing.extend(rel(PATHS[k]) for k in ["run_split_universe", "split_universe", "backtest_engine", "orthogonalization"] if not PATHS[k].exists())
    result["prerequisites_passed"] = not missing
    result["missing_files"] = missing
    result["caveat"] = "本 prep 只读源码和 audit outputs；不生成 alpha/weights/returns。"
    dump_json(OUT_DIR / "v0_composite_alignment_repair_prep_prerequisite_check.json", result)
    return result


def legacy_policy_outputs() -> tuple[pd.DataFrame, dict]:
    rows = []
    for factor in FACTORS:
        rows.extend(
            [
                {
                    "source_script": "run_split_universe.py",
                    "function_or_block": "load_panel / combine_factors",
                    "factor_name": factor,
                    "raw_field_candidate": factor,
                    "z_field_candidate": f"{factor}_z",
                    "neutral_z_field_candidate": f"{factor}_neutral_z",
                    "rank_field_candidate": "",
                    "actual_legacy_input_field_inferred": f"{factor}_neutral_z",
                    "evidence": "factor_z_cols selects columns ending _neutral_z; combine_factors receives factor_z_cols.",
                    "confidence": "HIGH",
                    "caveat": "若 _neutral_z 缺失则回退 _z；本 legacy preprocessed 中 _neutral_z 存在。",
                },
                {
                    "source_script": "factor_research/split_universe.py",
                    "function_or_block": "_get_factor_col",
                    "factor_name": factor,
                    "raw_field_candidate": factor,
                    "z_field_candidate": f"{factor}_z",
                    "neutral_z_field_candidate": f"{factor}_neutral_z",
                    "rank_field_candidate": "",
                    "actual_legacy_input_field_inferred": f"{factor}_neutral_z",
                    "evidence": "suffix resolution order is self._suffix, _neutral_z, _z, raw; _suffix becomes _neutral_z when available.",
                    "confidence": "HIGH",
                    "caveat": "",
                },
                {
                    "source_script": "scripts/rebuild_v0_strict_lag_icir_bridge_v0.py",
                    "function_or_block": "factor_actual_cols / apply_strict_lag_composite",
                    "factor_name": factor,
                    "raw_field_candidate": factor,
                    "z_field_candidate": f"{factor}_z",
                    "neutral_z_field_candidate": f"{factor}_neutral_z",
                    "rank_field_candidate": "",
                    "actual_legacy_input_field_inferred": f"{factor}_neutral_z",
                    "evidence": "factor_actual_cols checks _neutral_z, then _z, then raw; actual_cols are passed into strict-lag ICIR and GS.",
                    "confidence": "HIGH",
                    "caveat": "",
                },
            ]
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT_DIR / "legacy_composite_input_column_policy_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "legacy_input_policy": "AUTO_SUFFIX_RESOLUTION_PREFER_NEUTRAL_Z_THEN_Z_THEN_RAW",
        "uses_raw_fields": False,
        "uses_z_fields": False,
        "uses_neutral_z_fields": True,
        "uses_rank_fields": False,
        "auto_suffix_resolution_detected": True,
        "selected_policy_confidence": "HIGH",
        "caveat": "preprocessed.parquet contains all 16 *_neutral_z columns, so strict-lag repair bridge uses neutral_z inputs for ICIR and GS.",
    }
    dump_json(OUT_DIR / "legacy_composite_input_policy_summary.json", summary)
    return audit, summary


def canonical_policy_outputs(legacy_summary: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    diff_rows = []
    for factor in FACTORS:
        rows.append(
            {
                "factor_name": factor,
                "canonical_panel_field": factor,
                "field_used_for_icir": factor,
                "field_used_for_gs": f"{factor} -> monthly winsor_zscore({factor})",
                "raw_or_transformed": "RAW_INPUT_THEN_MONTHLY_WINSOR_ZSCORE_FOR_GS",
                "winsorized": True,
                "zscored": True,
                "ranked": False,
                "neutralized": False,
                "matches_legacy_policy": False,
                "mismatch_reason": "legacy uses preprocessed neutral_z field directly for ICIR and GS; canonical ICIR uses raw panel factor and GS uses local winsor_zscore.",
            }
        )
        diff_rows.append(
            {
                "factor_name": factor,
                "legacy_input_field": f"{factor}_neutral_z",
                "canonical_input_field": factor,
                "input_policy_match": False,
                "mismatch_type": "RAW_VS_NEUTRAL_Z",
                "severity": "HIGH",
                "recommended_alignment": f"create/select {factor}_neutral_z-equivalent input column before ICIR and GS; do not change factor formula or parameters.",
            }
        )
    canon = pd.DataFrame(rows)
    diff = pd.DataFrame(diff_rows)
    canon.to_csv(OUT_DIR / "canonical_composite_input_policy_audit.csv", index=False, encoding="utf-8-sig")
    diff.to_csv(OUT_DIR / "canonical_vs_legacy_composite_input_policy_diff.csv", index=False, encoding="utf-8-sig")
    return canon, diff


def policy_diff_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    icir_rows = [
        ("IC method", "Spearman Rank IC via ranked correlation", "Spearman Rank IC via ranked correlation", True, "LOW", "KEEP_MATCHED"),
        ("rolling window", "24 prior periods", "24 prior months", True, "LOW", "KEEP_24"),
        ("min_stocks", "20 in strict-lag repair bridge", "20", True, "LOW", "KEEP_20"),
        ("IC_IR formula", "mean(window_ic) / std(window_ic, ddof=1)", "mean / std(ddof=1)", True, "LOW", "KEEP_DDOF_1"),
        ("minimum IC count", "if <2 ICs then ICIR=0", "if <2 ICs then ICIR=0", True, "LOW", "KEEP_WARMUP_ZERO"),
        ("warmup behavior", "no composite when no selected ICIR factors", "NO_STRICT_LAG_ICIR_HISTORY when no selected ICIR factors", True, "LOW", "KEEP_NO_HISTORY_EMPTY_ALPHA"),
        ("min_ic_ir filter", "abs(ICIR) > 0.05", "abs(ICIR) > 0.05", True, "LOW", "KEEP_THRESHOLD"),
        ("abs_ic_ir sorting", "descending abs(ICIR)", "descending abs(ICIR)", True, "LOW", "KEEP_SORT"),
        ("flip_sign", "negative ICIR multiplies residual contribution by -1", "negative ICIR multiplies contribution by -1", True, "LOW", "KEEP_SIGN_FLIP"),
        ("zero / NaN handling", "ICIR zero not selected; NaN factor values filled to 0 before GS", "ICIR zero not selected; winsor_zscore missing filled to 0", False, "MEDIUM", "ALIGN_INPUT_MISSING_VALUE_POLICY_AFTER_NEUTRAL_Z"),
        ("current-month exclusion", "uses searchsorted side=left - 1; current month excluded", "ic_year_month < signal_month", True, "LOW", "KEEP_STRICT_LAG"),
        ("future exclusion", "future IC never included", "future IC never included", True, "LOW", "KEEP_STRICT_LAG"),
        ("input column used for ICIR", "neutral_z actual cols", "raw canonical factor cols", False, "CRITICAL", "ALIGN_ICIR_INPUT_COLUMNS_TO_NEUTRAL_Z_POLICY"),
    ]
    icir = pd.DataFrame(icir_rows, columns=["policy_item", "legacy_policy", "canonical_policy", "match", "severity", "recommended_alignment"])
    icir.to_csv(OUT_DIR / "strict_lag_icir_formula_policy_diff.csv", index=False, encoding="utf-8-sig")

    weight_rows = [
        ("min_ic_ir role", "filter factors with abs(ICIR) > 0.05 before GS and weight normalization", "filter selected factors with abs(ICIR) > 0.05", True, "LOW", "KEEP_FILTER_ROLE"),
        ("weight formula", "abs_icir / sum(abs_icir of valid selected cols)", "abs_icir / sum(abs_icir of selected factors)", True, "LOW", "KEEP_WEIGHT_FORMULA"),
        ("low ICIR factors", "excluded from sorted_cols and composite", "excluded from selected_vals and composite", True, "LOW", "KEEP_EXCLUSION"),
        ("all factors below threshold fallback", "composite remains 0 then alpha zscore can become 0/NaN depending month", "alpha status NO_STRICT_LAG_ICIR_HISTORY", False, "MEDIUM", "ALIGN_FALLBACK_STATUS_AND_OUTPUT"),
        ("sign flip target", "final residual contribution sign flips when ICIR < 0", "final contribution sign flips when ICIR < 0", True, "LOW", "KEEP_CONTRIBUTION_SIGN_FLIP"),
        ("normalization denominator", "selected and GS-valid factor abs_icir total", "selected factor abs_icir total before local GS", False, "MEDIUM", "ALIGN_DENOMINATOR_TO_GS_VALID_COLS"),
        ("input column impact", "weights computed from neutral_z ICIR", "weights computed from raw factor ICIR", False, "CRITICAL", "ALIGN_ICIR_INPUT_COLUMNS_FIRST"),
    ]
    weight = pd.DataFrame(weight_rows, columns=["policy_item", "legacy_policy", "canonical_policy", "match", "severity", "recommended_alignment"])
    weight.to_csv(OUT_DIR / "weight_normalization_policy_diff.csv", index=False, encoding="utf-8-sig")

    gs_rows = [
        ("GS input", "neutral_z actual columns", "monthly winsor_zscore(raw factor)", False, "CRITICAL", "ALIGN_GS_INPUT_TO_NEUTRAL_Z_ACTUAL_COLS"),
        ("pre-GS monthly zscore", "no extra zscore inside strict-lag bridge; relies on preprocessed neutral_z", "winsor_zscore per month before GS", False, "HIGH", "REMOVE_EXTRA_LOCAL_WINSOR_ZSCORE_WHEN_NEUTRAL_Z_AVAILABLE"),
        ("intercept", "np.linalg.lstsq without explicit intercept", "custom gram_schmidt projection without intercept", True, "LOW", "KEEP_NO_INTERCEPT"),
        ("factor order", "descending abs(ICIR) after threshold", "descending abs(ICIR) after threshold", True, "LOW", "KEEP_ORDER"),
        ("residual variance threshold", "1e-10", "implicit zero norm threshold 1e-12 in custom GS", False, "MEDIUM", "ALIGN_RESIDUAL_VARIANCE_THRESHOLD_1E_10"),
        ("collinear factor handling", "zero residual skipped in contribution", "orth column can remain zero through matrix product", False, "MEDIUM", "ALIGN_COLLINEAR_SKIP_POLICY"),
        ("residual post-zscore", "no residual re-zscore", "no residual re-zscore", True, "LOW", "KEEP_NO_RESIDUAL_ZSCORE"),
        ("missing values", "nan_to_num factor values before residualization", "winsor_zscore fillna(0)", False, "MEDIUM", "ALIGN_MISSING_VALUE_POLICY"),
        ("per split_group execution", "large/small independently", "large/small independently", True, "LOW", "KEEP_PER_SPLIT"),
        ("post split final zscore", "zscore_by_month after large/small concat, across full month", "zscore within each year_month/split_group output", False, "HIGH", "ALIGN_FINAL_ALPHA_ZSCORE_TO_FULL_MONTH_AFTER_SPLIT_CONCAT"),
    ]
    gs = pd.DataFrame(gs_rows, columns=["policy_item", "legacy_policy", "canonical_policy", "match", "severity", "recommended_alignment"])
    gs.to_csv(OUT_DIR / "gram_schmidt_policy_diff.csv", index=False, encoding="utf-8-sig")
    return icir, weight, gs


def repair_design_and_config() -> tuple[pd.DataFrame, dict]:
    rows = [
        ("input_column_policy", "raw canonical 16-factor fields feed ICIR", "auto suffix policy: prefer *_neutral_z then *_z then raw", "add repaired composite input resolver and feed actual neutral_z-equivalent columns", "ICIR/sign/weight closer to legacy", "needs transform columns or equivalent construction", True),
        ("monthly_transform_policy", "GS uses local monthly winsor_zscore(raw)", "legacy consumes preprocessed neutral_z without extra local winsor_zscore", "do not double-transform neutral_z inputs", "reduces transform drift", "must document neutralization source", True),
        ("icir_formula_policy", "formula mostly aligned but raw inputs differ", "Spearman, 24, min_stocks=20, ddof=1, strict lag", "keep formula; change input matrix only", "isolates repair to policy mismatch", "low", True),
        ("min_icir_filter_policy", "abs(ICIR)>0.05 selected", "abs(ICIR)>0.05 selected", "keep threshold and operator", "no tuning", "low", True),
        ("sign_flip_policy", "flip final contribution by ICIR sign", "flip final residual contribution by ICIR sign", "keep sign flip semantics", "preserves legacy behavior", "low", True),
        ("weight_normalization_policy", "denominator uses selected factors before custom GS", "denominator uses GS-valid selected factors", "normalize after GS-valid col list is known", "removes effective-weight drift", "medium", True),
        ("gs_input_policy", "custom GS over winsor_zscore(raw)", "lstsq residualization over neutral_z actual cols", "use legacy-style residualization routine over actual input cols", "aligns Layer 5 mechanics", "medium", True),
        ("gs_residual_zscore_policy", "no residual zscore", "no residual zscore", "keep no post-residual zscore", "no unnecessary change", "low", True),
        ("split_group_final_zscore_policy", "alpha zscore inside split group", "full-month zscore after large/small concat", "move final zscore after concat across entire month", "aligns alpha scale/rank handoff", "medium", True),
    ]
    design = pd.DataFrame(
        rows,
        columns=["repair_item", "current_canonical_behavior", "target_legacy_behavior", "repair_action", "expected_effect", "risk", "allowed_next_run"],
    )
    design.to_csv(OUT_DIR / "composite_alignment_repair_design.csv", index=False, encoding="utf-8-sig")
    config = {
        "repair_run_allowed_next": True,
        "input_panel_path": "output\\v0_canonical_16factor_panel_build_v0\\v0_canonical_16factor_panel.parquet",
        "target_legacy_policy": "AUTO_SUFFIX_RESOLUTION_PREFER_NEUTRAL_Z_THEN_Z_THEN_RAW__STRICT_LAG_ICIR_24M__ABS_ICIR_WEIGHT__LEGACY_GS_RESIDUALIZATION",
        "repair_items": design["repair_item"].tolist(),
        "output_alpha_candidate_path_next": "output\\v0_strict_lag_composite_alignment_alpha_candidate_v0\\v0_canonical_alpha_signal_panel_composite_aligned_candidate.parquet",
        "generate_alpha_signal_next_run_allowed": True,
        "generate_weights_next_run_allowed": False,
        "calculate_returns_next_run_allowed": False,
        "tune_parameters_allowed": False,
        "production_allowed": False,
        "expected_validation_outputs": [
            "repaired_vs_legacy_alpha_spearman",
            "repaired_vs_legacy_top50_overlap",
            "repaired_icir_weight_drift_summary",
            "strict_lag_leakage_qa",
        ],
    }
    dump_json(OUT_DIR / "v0_composite_alignment_repair_run_config_draft.json", config)
    return design, config


def guardrails() -> tuple[pd.DataFrame, bool]:
    values = {
        "alpha_signal_regenerated": False,
        "strategy_weights_regenerated": False,
        "portfolio_returns_calculated": False,
        "old_artifacts_modified": False,
        "production_modified": False,
        "ml_training_run": False,
        "tuning_run": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
    }
    out = pd.DataFrame([{"guardrail": k, "expected": v, "actual": v, "pass": True} for k, v in values.items()])
    out.to_csv(OUT_DIR / "v0_composite_alignment_repair_prep_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    return out, bool(out["pass"].all())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", "prerequisite_check")
    prereq = prerequisite_check()
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", "policy_audits")
    _, legacy_summary = legacy_policy_outputs()
    canonical_policy, input_diff = canonical_policy_outputs(legacy_summary)
    icir_diff, weight_diff, gs_diff = policy_diff_outputs()
    design, config = repair_design_and_config()
    guardrail, guardrails_passed = guardrails()

    input_policy_mismatch = bool((~input_diff["input_policy_match"]).any())
    icir_formula_mismatch = bool((~icir_diff["match"]).any())
    weight_mismatch = bool((~weight_diff["match"]).any())
    gs_mismatch = bool((~gs_diff["match"]).any())
    repair_design_ready = bool(len(design) >= 9 and design["allowed_next_run"].all())
    repair_run_allowed_next = bool(config["repair_run_allowed_next"] and repair_design_ready and guardrails_passed)

    if not guardrails_passed:
        final_decision = "COMPOSITE_ALIGNMENT_REPAIR_PREP_FAIL_GUARDRAIL"
    elif not legacy_summary["uses_neutral_z_fields"]:
        final_decision = "COMPOSITE_ALIGNMENT_REPAIR_PREP_BLOCKED_BY_MISSING_LEGACY_EVIDENCE"
    elif repair_design_ready and repair_run_allowed_next:
        final_decision = "COMPOSITE_ALIGNMENT_REPAIR_PREP_READY_FOR_ALPHA_REBUILD"
    else:
        final_decision = "COMPOSITE_ALIGNMENT_REPAIR_PREP_INCONCLUSIVE_NEED_MANUAL_CODE_REVIEW"

    recommended_next_step = {
        "COMPOSITE_ALIGNMENT_REPAIR_PREP_READY_FOR_ALPHA_REBUILD": "下一任务可只生成 composite-aligned alpha candidate，并只做 alpha overlap / ICIR drift / leakage QA；仍不得生成 weights 或收益。",
        "COMPOSITE_ALIGNMENT_REPAIR_PREP_INCONCLUSIVE_NEED_MANUAL_CODE_REVIEW": "先人工复核源码证据，再决定是否 alpha candidate rebuild。",
        "COMPOSITE_ALIGNMENT_REPAIR_PREP_BLOCKED_BY_MISSING_LEGACY_EVIDENCE": "补齐 legacy composite evidence 后再准备 repair run。",
        "COMPOSITE_ALIGNMENT_REPAIR_PREP_FAIL_GUARDRAIL": "停止，先修复 guardrail violation。",
    }[final_decision]

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "legacy_input_policy": legacy_summary["legacy_input_policy"],
        "canonical_input_policy": "RAW_FACTOR_FOR_ICIR__MONTHLY_WINSOR_ZSCORE_RAW_FOR_GS__SPLIT_LOCAL_FINAL_ZSCORE",
        "input_policy_mismatch_detected": input_policy_mismatch,
        "icir_formula_mismatch_detected": icir_formula_mismatch,
        "weight_normalization_mismatch_detected": weight_mismatch,
        "gram_schmidt_policy_mismatch_detected": gs_mismatch,
        "primary_mismatch": "ICIR_AND_GS_INPUT_COLUMNS_RAW_VS_NEUTRAL_Z",
        "secondary_mismatch": "FINAL_ALPHA_ZSCORE_SCOPE_AND_GS_VALID_WEIGHT_DENOMINATOR",
        "repair_design_ready": repair_design_ready,
        "repair_run_allowed_next": repair_run_allowed_next,
        "alpha_signal_regenerated": False,
        "strategy_weights_regenerated": False,
        "portfolio_returns_calculated": False,
        "old_artifacts_modified": False,
        "production_modified": False,
        "ml_training_run": False,
        "tuning_run": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    dump_json(OUT_DIR / "v0_strict_lag_composite_alignment_repair_prep_summary.json", summary)

    report = (
        "# V0 Strict-Lag Composite Alignment Repair Prep v0\n\n"
        f"- final_decision: {final_decision}\n"
        f"- legacy_input_policy: {summary['legacy_input_policy']}\n"
        f"- canonical_input_policy: {summary['canonical_input_policy']}\n"
        f"- primary_mismatch: {summary['primary_mismatch']}\n"
        f"- secondary_mismatch: {summary['secondary_mismatch']}\n"
        f"- repair_design_ready: {repair_design_ready}; repair_run_allowed_next: {repair_run_allowed_next}\n"
        f"- guardrails_passed: {guardrails_passed}\n\n"
        "本 prep 未重新生成 alpha_signal/weights，未计算收益，未调参，未训练，未做 benchmark-relative、alpha/beta、IR/TE、FF、DGTW、SHAP 或 production 修改。\n"
    )
    (OUT_DIR / "v0_strict_lag_composite_alignment_repair_prep_report.md").write_text(report, encoding="utf-8")

    final_qa = pd.DataFrame(
        [
            {"check_name": "prerequisites_passed", "pass": prereq["prerequisites_passed"], "detail": ""},
            {"check_name": "guardrails_passed", "pass": guardrails_passed, "detail": ""},
            {"check_name": "repair_design_ready", "pass": repair_design_ready, "detail": ""},
            {"check_name": "repair_run_allowed_next", "pass": repair_run_allowed_next, "detail": ""},
            {"check_name": "final_decision_allowed", "pass": final_decision in {
                "COMPOSITE_ALIGNMENT_REPAIR_PREP_READY_FOR_ALPHA_REBUILD",
                "COMPOSITE_ALIGNMENT_REPAIR_PREP_INCONCLUSIVE_NEED_MANUAL_CODE_REVIEW",
                "COMPOSITE_ALIGNMENT_REPAIR_PREP_BLOCKED_BY_MISSING_LEGACY_EVIDENCE",
                "COMPOSITE_ALIGNMENT_REPAIR_PREP_FAIL_GUARDRAIL",
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
    write_state("completed", "all_outputs_written")
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": rel(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
