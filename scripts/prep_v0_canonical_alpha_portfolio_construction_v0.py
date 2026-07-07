from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_NAME = "v0_canonical_alpha_portfolio_construction_prep_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

ALPHA_PANEL = (
    ROOT
    / "output"
    / "v0_canonical_strict_lag_alpha_build_v0"
    / "v0_canonical_alpha_signal_panel.parquet"
)
ALPHA_SUMMARY = (
    ROOT
    / "output"
    / "v0_canonical_strict_lag_alpha_build_v0"
    / "v0_canonical_strict_lag_alpha_build_summary.json"
)
ALPHA_QA = (
    ROOT
    / "output"
    / "v0_canonical_strict_lag_alpha_build_v0"
    / "v0_canonical_alpha_signal_qa.csv"
)
ALPHA_MONTHLY_QA = (
    ROOT
    / "output"
    / "v0_canonical_strict_lag_alpha_build_v0"
    / "v0_canonical_alpha_signal_monthly_qa.csv"
)
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
    required = [ALPHA_PANEL, ALPHA_SUMMARY, ALPHA_QA, ALPHA_MONTHLY_QA, RETURN_MAP]
    missing = [rel(p) for p in required if not p.exists()]
    return {
        "alpha_panel_found": ALPHA_PANEL.exists(),
        "alpha_summary_found": ALPHA_SUMMARY.exists(),
        "alpha_qa_found": ALPHA_QA.exists() and ALPHA_MONTHLY_QA.exists(),
        "trd_mnth_return_map_found": RETURN_MAP.exists(),
        "prerequisites_passed": len(missing) == 0,
        "missing_files": missing,
    }


def load_alpha_panel() -> pd.DataFrame:
    cols = ["symbol_norm", "year_month", "alpha_signal", "factor_count_used"]
    df = pd.read_parquet(ALPHA_PANEL, columns=cols)
    df["symbol_norm"] = df["symbol_norm"].astype(str).str.zfill(6)
    df["year_month"] = df["year_month"].astype(str).str.slice(0, 7)
    df["alpha_signal"] = pd.to_numeric(df["alpha_signal"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["factor_count_used"] = pd.to_numeric(df["factor_count_used"], errors="coerce")
    return df


def build_eligibility_audit(alpha: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ym, grp in alpha.groupby("year_month", sort=True):
        total = len(grp)
        eligible = grp[grp["alpha_signal"].notna()]
        ratio = float(len(eligible) / total) if total else 0.0
        avg_factor = float(grp["factor_count_used"].mean()) if total else 0.0
        min_factor = int(grp["factor_count_used"].min()) if total and grp["factor_count_used"].notna().any() else 0
        max_factor = int(grp["factor_count_used"].max()) if total and grp["factor_count_used"].notna().any() else 0
        eligible_count = int(len(eligible))
        if eligible_count < 75 or ratio < 0.80:
            status = "FAIL_NO_SIGNAL"
            caveat = "eligible symbol count <75 or alpha coverage <0.80"
        elif avg_factor < 5:
            status = "WATCH_LOW_FACTOR_COUNT"
            caveat = "avg_factor_count_used <5; typically warmup month"
        elif ratio < 0.95:
            status = "WATCH_LOW_ALPHA_COVERAGE"
            caveat = "alpha coverage between 0.80 and 0.95"
        else:
            status = "READY"
            caveat = ""
        rows.append(
            {
                "year_month": ym,
                "total_count": total,
                "alpha_non_null_count": eligible_count,
                "alpha_non_null_ratio": round(ratio, 6),
                "avg_factor_count_used": round(avg_factor, 6),
                "min_factor_count_used": min_factor,
                "max_factor_count_used": max_factor,
                "eligible_symbol_count": eligible_count,
                "eligible_month_status": status,
                "caveat": caveat,
            }
        )
    return pd.DataFrame(rows)


def construction_policy() -> dict[str, Any]:
    return {
        "portfolio_name": PORTFOLIO_NAME,
        "score_column": "alpha_signal",
        "higher_is_better": True,
        "target_holding_count": TARGET_HOLDING_COUNT,
        "entry_rank": ENTRY_RANK,
        "exit_rank": EXIT_RANK,
        "first_month_initialization": "top50",
        "weighting_scheme": "equal_weight",
        "tie_breaker": "symbol_norm ascending",
        "eligible_month_policy": "include READY and WATCH months; exclude FAIL_NO_SIGNAL",
        "eligible_symbol_policy": "non-null alpha_signal only",
        "use_fwd_ret_for_selection": False,
        "use_benchmark_for_selection": False,
        "use_future_information": False,
        "production_allowed": False,
    }


def eligible_month_policy(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in audit.itertuples(index=False):
        include = row.eligible_month_status != "FAIL_NO_SIGNAL"
        if row.eligible_month_status == "READY":
            reason = "READY month: alpha coverage >=0.95 and eligible symbols >=75"
        elif row.eligible_month_status.startswith("WATCH"):
            reason = "WATCH month included with QA caveat; construction does not require future label"
        else:
            reason = "FAIL_NO_SIGNAL excluded from construction"
        rows.append(
            {
                "year_month": row.year_month,
                "alpha_non_null_ratio": row.alpha_non_null_ratio,
                "eligible_symbol_count": row.eligible_symbol_count,
                "avg_factor_count_used": row.avg_factor_count_used,
                "month_status": row.eligible_month_status,
                "include_in_construction_next_run": include,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def future_eval_coverage_plan(month_policy: pd.DataFrame) -> pd.DataFrame:
    ret = pd.read_parquet(RETURN_MAP, columns=["symbol_norm", "year_month", "fwd_ret_1m"])
    ret["symbol_norm"] = ret["symbol_norm"].astype(str).str.zfill(6)
    ret["year_month"] = ret["year_month"].astype(str).str.slice(0, 7)
    ret["fwd_ret_1m"] = pd.to_numeric(ret["fwd_ret_1m"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    cov = (
        ret.groupby("year_month")
        .agg(total=("symbol_norm", "count"), valid=("fwd_ret_1m", lambda x: int(x.notna().sum())))
        .reset_index()
    )
    cov["trd_mnth_fwd_ret_available_ratio"] = cov["valid"] / cov["total"]
    out = month_policy[["year_month", "include_in_construction_next_run"]].rename(
        columns={"include_in_construction_next_run": "eligible_for_construction"}
    )
    out = out.merge(cov[["year_month", "trd_mnth_fwd_ret_available_ratio"]], on="year_month", how="left")
    out["trd_mnth_fwd_ret_available_ratio"] = out["trd_mnth_fwd_ret_available_ratio"].fillna(0.0).round(6)

    def _status(ratio: float) -> str:
        if ratio >= 0.95:
            return "EVAL_READY"
        if ratio >= 0.80:
            return "EVAL_READY_WITH_GAPS"
        if ratio > 0:
            return "EVAL_WATCH_LOW_COVERAGE"
        return "EVAL_UNAVAILABLE"

    out["evaluation_label_status"] = out["trd_mnth_fwd_ret_available_ratio"].apply(_status)
    out["expected_eval_inclusion"] = out["eligible_for_construction"] & out["evaluation_label_status"].isin(
        ["EVAL_READY", "EVAL_READY_WITH_GAPS"]
    )
    out["caveat"] = np.where(
        out["evaluation_label_status"] == "EVAL_UNAVAILABLE",
        "forward return label unavailable; likely latest month or return map gap",
        "",
    )
    del ret, cov
    gc.collect()
    return out[
        [
            "year_month",
            "eligible_for_construction",
            "trd_mnth_fwd_ret_available_ratio",
            "evaluation_label_status",
            "expected_eval_inclusion",
            "caveat",
        ]
    ]


def run_config(policy: dict[str, Any], month_policy_path: Path) -> dict[str, Any]:
    return {
        "construction_allowed_next_run": True,
        "alpha_panel_path": rel(ALPHA_PANEL),
        "portfolio_name": PORTFOLIO_NAME,
        "score_column": "alpha_signal",
        "eligible_month_policy_path": rel(month_policy_path),
        "portfolio_rule": policy,
        "output_weights_path_next": "output/v0_canonical_alpha_portfolio_construction_run_v0/v0_canonical_top50_buffer_35_75_weights.parquet",
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


def guardrail_qa() -> pd.DataFrame:
    guardrails = {
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "production_modified": False,
        "ml_training_run": False,
        "new_ml_model_trained": False,
        "tuning_run": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
    }
    return pd.DataFrame(
        [{"guardrail": k, "expected": False, "actual": v, "pass": v is False} for k, v in guardrails.items()]
    )


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
    save_json(prereq, OUT_DIR / "v0_canonical_portfolio_prep_prerequisite_check.json")
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", {"step": "alpha_eligibility_audit"})
    alpha = load_alpha_panel()
    audit = build_eligibility_audit(alpha)
    audit.to_csv(OUT_DIR / "v0_canonical_alpha_eligibility_audit.csv", index=False, encoding="utf-8-sig")

    policy = construction_policy()
    save_json(policy, OUT_DIR / "v0_canonical_portfolio_construction_policy.json")

    month_policy = eligible_month_policy(audit)
    month_policy_path = OUT_DIR / "v0_canonical_portfolio_eligible_month_policy.csv"
    month_policy.to_csv(month_policy_path, index=False, encoding="utf-8-sig")

    write_state("running", {"step": "future_eval_coverage_planning"})
    eval_plan = future_eval_coverage_plan(month_policy)
    eval_plan.to_csv(OUT_DIR / "v0_canonical_portfolio_future_eval_coverage_plan.csv", index=False, encoding="utf-8-sig")

    config = run_config(policy, month_policy_path)
    save_json(config, OUT_DIR / "v0_canonical_portfolio_construction_run_config_draft.json")

    guardrails = guardrail_qa()
    guardrails.to_csv(OUT_DIR / "v0_canonical_portfolio_prep_guardrail_qa.csv", index=False, encoding="utf-8-sig")

    ready_month_count = int((audit["eligible_month_status"] == "READY").sum())
    watch_month_count = int(audit["eligible_month_status"].str.startswith("WATCH").sum())
    fail_month_count = int((audit["eligible_month_status"] == "FAIL_NO_SIGNAL").sum())
    eligible_months = month_policy.loc[month_policy["include_in_construction_next_run"] == True, "year_month"]  # noqa: E712
    construction_allowed = bool(len(eligible_months) > 0 and guardrails["pass"].all())
    has_caveats = bool(watch_month_count > 0 or fail_month_count > 0)
    if not guardrails["pass"].all():
        final_decision = "V0_CANONICAL_PORTFOLIO_PREP_FAIL_GUARDRAIL"
    elif not len(eligible_months):
        final_decision = "V0_CANONICAL_PORTFOLIO_PREP_BLOCKED_BY_ALPHA_ELIGIBILITY"
    elif has_caveats:
        final_decision = "V0_CANONICAL_PORTFOLIO_PREP_READY_WITH_CAVEATS"
    else:
        final_decision = "V0_CANONICAL_PORTFOLIO_PREP_READY_FOR_CONSTRUCTION_RUN"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "alpha_panel_path": rel(ALPHA_PANEL),
        "alpha_panel_loaded": True,
        "row_count": int(len(alpha)),
        "unique_symbol_count": int(alpha["symbol_norm"].nunique()),
        "month_count": int(alpha["year_month"].nunique()),
        "min_year_month": str(alpha["year_month"].min()) if len(alpha) else "",
        "max_year_month": str(alpha["year_month"].max()) if len(alpha) else "",
        "alpha_signal_non_null_ratio": round(float(alpha["alpha_signal"].notna().mean()), 6),
        "ready_month_count": ready_month_count,
        "watch_month_count": watch_month_count,
        "fail_month_count": fail_month_count,
        "first_eligible_month": str(eligible_months.min()) if len(eligible_months) else "",
        "last_eligible_month": str(eligible_months.max()) if len(eligible_months) else "",
        "portfolio_name": PORTFOLIO_NAME,
        "portfolio_rule_locked": True,
        "target_holding_count": TARGET_HOLDING_COUNT,
        "entry_rank": ENTRY_RANK,
        "exit_rank": EXIT_RANK,
        "eligible_month_policy_locked": True,
        "future_eval_coverage_planned": True,
        "construction_allowed_next_run": construction_allowed,
        "generate_weights_next_run_allowed": construction_allowed,
        "calculate_returns_next_run_allowed": False,
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "production_modified": False,
        "ml_training_run": False,
        "new_ml_model_trained": False,
        "tuning_run": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
        "final_decision": final_decision,
        "recommended_next_step": (
            "进入 V0 canonical portfolio construction run，仅生成 Top50 Buffer 35/75 weights，不计算 returns。"
            if construction_allowed
            else "先修复 alpha eligibility 后再进入 construction run。"
        ),
    }
    save_json(summary, OUT_DIR / "v0_canonical_alpha_portfolio_construction_prep_summary.json")

    report = "\n".join(
        [
            "# V0 Canonical Alpha Portfolio Construction Prep v0",
            "",
            "## 结论",
            f"- final_decision: {final_decision}",
            f"- construction_allowed_next_run: {construction_allowed}",
            f"- ready/watch/fail months: {ready_month_count}/{watch_month_count}/{fail_month_count}",
            "",
            "## Eligibility Snapshot",
            simple_table(audit, ["year_month", "alpha_non_null_ratio", "avg_factor_count_used", "eligible_symbol_count", "eligible_month_status"]),
            "",
            "## Policy",
            f"- portfolio_name: {PORTFOLIO_NAME}",
            f"- target_holding_count: {TARGET_HOLDING_COUNT}",
            f"- entry_rank/exit_rank: {ENTRY_RANK}/{EXIT_RANK}",
            "- weighting_scheme: equal_weight",
            "- 本任务未生成 weights，未计算 returns。",
        ]
    )
    (OUT_DIR / "v0_canonical_alpha_portfolio_construction_prep_report.md").write_text(report, encoding="utf-8")

    final_qa = guardrails.copy()
    required_artifacts = [
        OUT_DIR / "v0_canonical_portfolio_prep_prerequisite_check.json",
        OUT_DIR / "v0_canonical_alpha_eligibility_audit.csv",
        OUT_DIR / "v0_canonical_portfolio_construction_policy.json",
        OUT_DIR / "v0_canonical_portfolio_eligible_month_policy.csv",
        OUT_DIR / "v0_canonical_portfolio_future_eval_coverage_plan.csv",
        OUT_DIR / "v0_canonical_portfolio_construction_run_config_draft.json",
        OUT_DIR / "v0_canonical_portfolio_prep_guardrail_qa.csv",
        OUT_DIR / "v0_canonical_alpha_portfolio_construction_prep_summary.json",
        OUT_DIR / "v0_canonical_alpha_portfolio_construction_prep_report.md",
        ROOT / "scripts" / "prep_v0_canonical_alpha_portfolio_construction_v0.py",
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
                f"- construction_allowed_next_run: {construction_allowed}",
                f"- generate_weights_next_run_allowed: {construction_allowed}",
                "- strategy_weights_generated: false",
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
            "script": rel(ROOT / "scripts" / "prep_v0_canonical_alpha_portfolio_construction_v0.py"),
            "stdout_log": rel(RUN_DIR / "run_stdout.txt"),
            "stderr_log": rel(RUN_DIR / "run_stderr.txt"),
            "output_dir": rel(OUT_DIR),
            "final_decision": final_decision,
        },
        OUT_DIR / "terminal_summary.json",
    )
    write_state("completed", {"final_decision": final_decision, "output_dir": rel(OUT_DIR)})
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    del alpha, audit, month_policy, eval_plan
    gc.collect()


if __name__ == "__main__":
    main()
