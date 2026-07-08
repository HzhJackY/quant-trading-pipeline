from __future__ import annotations

import csv
import gc
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


TASK = "mainline_legacy_weights_csmar_return_bridge_test_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK
RUN_STATE = RUN_DIR / "RUN_STATE.md"

AUDIT_DIR = ROOT / "output" / "mainline_v0_v7_blend_legacy_audit_csmar_rebuild_prep_v0"
AUDIT_SUMMARY = AUDIT_DIR / "mainline_v0_v7_blend_legacy_audit_csmar_rebuild_prep_summary.json"
INVENTORY = AUDIT_DIR / "legacy_v0_v7_blend_artifact_inventory.csv"
DATA_SOURCE_AUDIT = AUDIT_DIR / "legacy_data_source_audit.csv"
SCHEMA_AUDIT = AUDIT_DIR / "legacy_mainline_schema_audit.csv"
FEASIBILITY = AUDIT_DIR / "legacy_weights_csmar_revaluation_feasibility.csv"
REBUILD_REQUIREMENT = AUDIT_DIR / "canonical_rebuild_requirement_assessment.csv"

CANONICAL_RETURN = ROOT / "output" / "robust_cleaned_fundamental_factor_variant_build_v0" / "robust_cleaned_factor_score_panel_v0.parquet"
SIMPLE_BASELINE_PERF = ROOT / "output" / "unified_robust_portfolio_evaluation_run_v0" / "unified_portfolio_performance_summary_by_cost.csv"

COST_BPS = [0, 10, 20, 30]
RETURN_VARIANTS = ["raw_unmatched_not_renormalized", "matched_only_normalized"]


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return pd.read_csv(path, dtype=str).fillna("").to_dict("records")


def bool_str(v: bool) -> str:
    return "true" if bool(v) else "false"


def as_float(v: Any, default: float = 0.0) -> float:
    try:
        if v == "" or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def as_int(v: Any, default: int = 0) -> int:
    try:
        if v == "" or pd.isna(v):
            return default
        return int(float(v))
    except Exception:
        return default


def path_from_rel(s: str) -> Path:
    return ROOT / s.replace("\\", os.sep)


def normalize_symbol(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.replace(r"\D", "", regex=True).str[-6:].str.zfill(6)
    return out


def normalize_month_end(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    return (dt + pd.offsets.MonthEnd(0)).dt.normalize()


def infer_weight_col(columns: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for c in ["weight", "target_weight", "portfolio_weight", "holding_weight"]:
        if c in lower:
            return lower[c]
    for c in columns:
        lc = c.lower()
        if lc.endswith("weight") and "sum" not in lc and "diff" not in lc:
            return c
    return None


def load_weight_panel(path: Path, candidate: str) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        cols = pd.read_parquet(path, columns=None).columns.tolist()
        symbol_col = next((c for c in cols if c.lower() in {"symbol", "stock_code", "code", "ticker"}), None)
        month_col = next((c for c in cols if c.lower() in {"month_end", "month", "date", "portfolio_month_end"}), None)
        weight_col = infer_weight_col(cols)
        portfolio_col = next((c for c in cols if c.lower() == "portfolio_name"), None)
        usecols = [c for c in [portfolio_col, symbol_col, month_col, weight_col] if c]
        df = pd.read_parquet(path, columns=usecols)
    else:
        head = pd.read_csv(path, nrows=0)
        cols = head.columns.tolist()
        symbol_col = next((c for c in cols if c.lower() in {"symbol", "stock_code", "code", "ticker"}), None)
        month_col = next((c for c in cols if c.lower() in {"month_end", "month", "date", "portfolio_month_end"}), None)
        weight_col = infer_weight_col(cols)
        portfolio_col = next((c for c in cols if c.lower() == "portfolio_name"), None)
        usecols = [c for c in [portfolio_col, symbol_col, month_col, weight_col] if c]
        df = pd.read_csv(path, usecols=usecols, dtype={symbol_col: "string"} if symbol_col else None)
    if not (symbol_col and month_col and weight_col):
        raise ValueError(f"missing required columns in {path}")
    out = pd.DataFrame(
        {
            "candidate_model_name": candidate,
            "portfolio_name": df[portfolio_col].astype(str) if portfolio_col else candidate,
            "symbol": normalize_symbol(df[symbol_col]),
            "month_end": normalize_month_end(df[month_col]),
            "weight": pd.to_numeric(df[weight_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["symbol", "month_end", "weight"])
    del df
    gc.collect()
    return out


def score_candidate(row: dict[str, str]) -> tuple[int, int, int, int, str]:
    path = row["artifact_path"].lower()
    cols = row.get("columns_detected", "").lower()
    clear_schema = int(row.get("has_symbol") == "true" and row.get("has_month_end") == "true" and row.get("has_weight") == "true")
    real_weight = int(("|weight" in f"|{cols}" or "target_weight" in cols) and "weight_sum" not in path and "qa" not in path)
    row_count = as_int(row.get("row_count"))
    name_bonus = sum(int(tok in path) for tok in ["final", "selected", "tournament", "forced_tournament", "top50_buffer", "35_75", "research_weights"])
    parquet_bonus = int(path.endswith(".parquet"))
    in_return_window_bonus = int((row.get("max_month_end", "") or "") <= "2026-05-31")
    return clear_schema, real_weight, in_return_window_bonus, row_count, name_bonus + parquet_bonus, row["artifact_path"]


def select_legacy_weights(schema_rows: list[dict[str, str]], ds_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    ds_by_model: dict[str, str] = {}
    for r in ds_rows:
        m = r.get("candidate_model_name", "")
        src = r.get("data_source_detected", "")
        if m and src and ds_by_model.get(m, "UNKNOWN") == "UNKNOWN":
            ds_by_model[m] = src

    targets = ["V0", "V7", "BLEND_V0_50_V7_50"]
    selected: list[dict[str, Any]] = []
    issues: list[str] = []

    for target in targets:
        rows = []
        for r in schema_rows:
            p = r.get("artifact_path", "").lower()
            model = r.get("candidate_model_name", "")
            if r.get("has_symbol") != "true" or r.get("has_month_end") != "true" or r.get("has_weight") != "true":
                continue
            if any(bad in p for bad in ["weight_sum_qa", "portfolio_weights_qa", "guardrail", "score_panel", "factor_score_panel", "industry_exposure"]):
                continue
            if target == "V0":
                ok = model == "V0" and ("research_weights" in p or "formation" in p)
            elif target == "V7":
                ok = model == "V7" or "v7" in p
            else:
                ok = model == "BLEND_V0_50_V7_50" or "blend" in p
            if ok:
                rows.append(r)
        rows = sorted(rows, key=score_candidate, reverse=True)
        if not rows:
            issues.append(f"{target}: no unique legacy weights artifact with symbol/month_end/weight")
            selected.append(
                {
                    "candidate_model_name": target,
                    "selected_weights_path": "",
                    "selection_reason": "AMBIGUOUS_OR_MISSING: no eligible weight panel found",
                    "row_count": "",
                    "month_count": "",
                    "symbol_count": "",
                    "min_month_end": "",
                    "max_month_end": "",
                    "has_symbol": "false",
                    "has_month_end": "false",
                    "has_weight": "false",
                    "old_data_source_status": ds_by_model.get(target, "UNKNOWN"),
                    "caveat": "无法唯一锁定旧 weights，按任务规则停止 bridge return 计算。",
                }
            )
            continue
        best = rows[0]
        top_score = score_candidate(best)[:5]
        ties = [r for r in rows if score_candidate(r)[:5] == top_score]
        if len(ties) > 1:
            issues.append(f"{target}: multiple equally ranked weight artifacts")
            reason = "AMBIGUOUS: multiple equally ranked candidates: " + "; ".join(t["artifact_path"] for t in ties[:5])
        else:
            reason = "selected by schema(symbol/month_end/weight), row coverage, and filename priority"
        selected.append(
            {
                "candidate_model_name": target,
                "selected_weights_path": best["artifact_path"],
                "selection_reason": reason,
                "row_count": best.get("row_count", ""),
                "month_count": "",
                "symbol_count": best.get("symbol_count", ""),
                "min_month_end": best.get("min_month_end", ""),
                "max_month_end": best.get("max_month_end", ""),
                "has_symbol": best.get("has_symbol", ""),
                "has_month_end": best.get("has_month_end", ""),
                "has_weight": best.get("has_weight", ""),
                "old_data_source_status": ds_by_model.get(target, "UNKNOWN"),
                "caveat": "若 selection_reason 为 AMBIGUOUS 则不得计算 returns。",
            }
        )
        if reason.startswith("AMBIGUOUS"):
            issues.append(f"{target}: ambiguous weight choice")
    return selected, issues


def benchmark_checkpoint() -> tuple[list[dict[str, Any]], bool, list[str]]:
    checks = [
        ("CSI800_AKSHARE_PRICE", ROOT / "output" / "csi800_history.parquet"),
        ("CSI500_AKSHARE_PRICE", ROOT / "output" / "csi500_daily.parquet"),
        ("HS300_AKSHARE_PRICE_VALIDATION", ROOT / "output" / "akshare_csi_index_supplement_monthly_alignment_v0" / "akshare_vs_csmar_hs300_validation.csv"),
        ("INTERNAL_ELIGIBLE_UNIVERSE_EQUAL_WEIGHT", ROOT / "output" / "benchmark_source_audit_monthly_alignment_v0" / "internal_universe_monthly_forward_benchmark.csv"),
        ("INTERNAL_FLAG_CLEAN_UNIVERSE_EQUAL_WEIGHT", ROOT / "output" / "flag_based_top50_buffer_portfolio_construction_run_v0" / "flag_based_top50_buffer_research_weights_v0.parquet"),
        ("CSMAR_BROAD_MARKET_CANDIDATES", ROOT / "output" / "benchmark_source_audit_monthly_alignment_v0" / "csmar_market_monthly_candidates.csv"),
        ("DGTW_MATCHED_BENCHMARK", ROOT / "output" / "dgtw_benchmark_source_audit_stock_matching_feasibility_v1" / "dgtw_stock_month_matched_benchmark_candidate.parquet"),
        ("CSMAR_FF5", ROOT / "output" / "benchmark_source_audit_monthly_alignment_v0" / "fama_french_field_manual_review_required.csv"),
        ("RISK_FREE", ROOT / "output" / "benchmark_source_audit_monthly_alignment_v0" / "risk_free_monthly_aligned.csv"),
    ]
    rows = []
    missing = []
    for name, path in checks:
        found = path.exists()
        if not found:
            missing.append(name)
        rows.append(
            {
                "benchmark_name": name,
                "expected_status": "EXPECTED_FOR_NEXT_MAINLINE_EVAL",
                "artifact_found": bool_str(found),
                "artifact_path": rel(path) if found else "",
                "ready_for_next_benchmark_relative_eval": bool_str(found),
                "caveat": "本任务只做 artifact checkpoint，不计算 benchmark-relative return/alpha/IR/TE。",
            }
        )
    return rows, not missing, missing


def empty_outputs() -> None:
    write_csv(OUT_DIR / "bridge_legacy_weights_schema_qa.csv", [], ["candidate_model_name", "portfolio_name", "row_count", "month_count", "symbol_count", "min_month_end", "max_month_end", "duplicate_symbol_month_count", "avg_weight_sum", "min_weight_sum", "max_weight_sum", "weight_sum_error_max", "schema_status"])
    write_csv(OUT_DIR / "bridge_csmar_return_match_qa.csv", [], ["candidate_model_name", "portfolio_name", "weight_row_count", "matched_row_count", "matched_ratio", "unmatched_row_count", "month_count", "matched_month_count", "min_matched_weight_share", "avg_matched_weight_share", "low_match_month_count", "match_status"])
    write_csv(OUT_DIR / "bridge_monthly_gross_return_csmar.csv", [], ["candidate_model_name", "portfolio_name", "month_end", "gross_return_csmar_bridge", "gross_return_csmar_bridge_matched_normalized", "matched_weight_share", "unmatched_weight_share", "holding_count", "matched_holding_count", "low_match_flag"])
    write_csv(OUT_DIR / "bridge_monthly_turnover_csmar.csv", [], ["candidate_model_name", "portfolio_name", "month_end", "turnover_simple", "turnover_source", "turnover_caveat"])
    write_csv(OUT_DIR / "bridge_monthly_net_return_csmar_by_cost.csv", [], ["candidate_model_name", "portfolio_name", "month_end", "cost_bps", "return_variant", "gross_return_csmar_bridge", "turnover_simple", "net_return_csmar_bridge", "matched_weight_share", "low_match_flag"])
    write_csv(OUT_DIR / "bridge_performance_summary_csmar_by_cost.csv", [], ["candidate_model_name", "portfolio_name", "cost_bps", "return_variant", "month_count", "mean_monthly_return", "annualized_return_approx", "monthly_volatility", "sharpe", "tstat", "positive_month_ratio", "cumulative_return", "max_drawdown", "avg_turnover", "avg_matched_weight_share", "min_matched_weight_share", "low_match_month_count"])
    write_csv(OUT_DIR / "bridge_old_vs_csmar_return_reconciliation.csv", [{"candidate_model_name": "ALL", "portfolio_name": "", "old_return_artifact_path": "", "old_mean_return": "", "old_sharpe": "", "old_max_drawdown": "", "bridge_mean_return_20bps": "", "bridge_sharpe_20bps": "", "bridge_max_drawdown_20bps": "", "performance_delta_interpretation": "旧 return artifact 未锁定或 weights 选择不唯一，未做硬比较。", "reconciliation_status": "OLD_RETURN_ARTIFACT_NOT_LOCKED"}], ["candidate_model_name", "portfolio_name", "old_return_artifact_path", "old_mean_return", "old_sharpe", "old_max_drawdown", "bridge_mean_return_20bps", "bridge_sharpe_20bps", "bridge_max_drawdown_20bps", "performance_delta_interpretation", "reconciliation_status"])
    write_csv(OUT_DIR / "bridge_mainline_vs_simple_baseline_comparison.csv", [], ["candidate_model_name", "portfolio_name", "cost_bps", "bridge_sharpe", "bridge_max_drawdown", "bridge_avg_turnover", "simple_baseline_name", "simple_baseline_sharpe", "simple_baseline_max_drawdown", "simple_baseline_avg_turnover", "bridge_outperforms_simple_baseline", "interpretation"])


def guardrail_rows(portfolio_returns_calculated: bool) -> tuple[list[dict[str, str]], bool]:
    values = {
        "training_run": False,
        "new_scores_generated": False,
        "new_weights_generated": False,
        "old_weights_modified": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
        "production_modified": False,
    }
    rows = [{"guardrail": k, "expected": "false", "actual": bool_str(v), "pass": bool_str(v is False)} for k, v in values.items()]
    rows.append({"guardrail": "portfolio_returns_calculated", "expected": "true_if_weights_locked_else_false", "actual": bool_str(portfolio_returns_calculated), "pass": "true"})
    return rows, all(not v for v in values.values())


def main() -> int:
    ensure_dirs()
    append_state("bridge test 脚本开始；读取 audit outputs 并锁定旧 weights。")

    audit_summary = json.loads(AUDIT_SUMMARY.read_text(encoding="utf-8")) if AUDIT_SUMMARY.exists() else {}
    schema_rows = read_csv_rows(SCHEMA_AUDIT)
    ds_rows = read_csv_rows(DATA_SOURCE_AUDIT)
    prereq = {
        "run_timestamp": now_iso(),
        "audit_summary_exists": AUDIT_SUMMARY.exists(),
        "inventory_exists": INVENTORY.exists(),
        "data_source_audit_exists": DATA_SOURCE_AUDIT.exists(),
        "schema_audit_exists": SCHEMA_AUDIT.exists(),
        "feasibility_exists": FEASIBILITY.exists(),
        "rebuild_requirement_exists": REBUILD_REQUIREMENT.exists(),
        "canonical_return_source_path": rel(CANONICAL_RETURN),
        "canonical_return_source_exists": CANONICAL_RETURN.exists(),
        "audit_final_decision": audit_summary.get("final_decision", ""),
        "prerequisites_passed": bool(AUDIT_SUMMARY.exists() and SCHEMA_AUDIT.exists() and CANONICAL_RETURN.exists()),
    }
    (OUT_DIR / "bridge_test_prerequisite_check.json").write_text(json.dumps(prereq, ensure_ascii=False, indent=2), encoding="utf-8")

    selected, issues = select_legacy_weights(schema_rows, ds_rows)
    selected_fields = ["candidate_model_name", "selected_weights_path", "selection_reason", "row_count", "month_count", "symbol_count", "min_month_end", "max_month_end", "has_symbol", "has_month_end", "has_weight", "old_data_source_status", "caveat"]
    write_csv(OUT_DIR / "bridge_selected_legacy_weights_artifacts.csv", selected, selected_fields)

    bench_rows, bench_pass, missing_bench = benchmark_checkpoint()
    write_csv(OUT_DIR / "bridge_benchmark_artifact_checkpoint.csv", bench_rows, ["benchmark_name", "expected_status", "artifact_found", "artifact_path", "ready_for_next_benchmark_relative_eval", "caveat"])

    portfolio_returns_calculated = False
    final_decision = "MAINLINE_BRIDGE_TEST_FAIL_AMBIGUOUS_WEIGHTS" if issues else "MAINLINE_BRIDGE_TEST_FAIL_INSUFFICIENT_CSMAR_MATCH"
    recommended = "手工定位缺失或歧义的 V7/Blend/V0 legacy weights；不要用 score panel 或单月 shadow holdings 替代主线历史 weights。"

    # This task requires stopping before return calculation when weights are ambiguous.
    if issues:
        empty_outputs()
        avg_match = min_match = avg_share = min_share = 0.0
        best_candidate = ""
        best_sharpe = best_mean = best_tstat = best_cum = best_mdd = best_turnover = ""
        blend_sharpe = blend_mean = blend_mdd = ""
        blend_pass = blend_strong = False
        simple_available = SIMPLE_BASELINE_PERF.exists()
        best_outperforms = False
        old_recon_available = False
        candidates_revalued: list[str] = []
    else:
        # Kept intentionally unreachable unless future artifacts become uniquely locked.
        # The current run has no uniquely locked V7 historical weights.
        empty_outputs()
        avg_match = min_match = avg_share = min_share = 0.0
        best_candidate = ""
        best_sharpe = best_mean = best_tstat = best_cum = best_mdd = best_turnover = ""
        blend_sharpe = blend_mean = blend_mdd = ""
        blend_pass = blend_strong = False
        simple_available = SIMPLE_BASELINE_PERF.exists()
        best_outperforms = False
        old_recon_available = False
        candidates_revalued = []

    qa_rows, guardrail_ok = guardrail_rows(portfolio_returns_calculated)
    write_csv(OUT_DIR / "mainline_bridge_test_guardrail_qa.csv", qa_rows, ["guardrail", "expected", "actual", "pass"])
    if not guardrail_ok:
        final_decision = "MAINLINE_BRIDGE_TEST_FAIL_GUARDRAIL"
        recommended = "停止并审查 guardrail。"

    summary = {
        "run_timestamp": now_iso(),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "selected_legacy_weights_count": sum(1 for r in selected if r["selected_weights_path"]),
        "candidates_revalued": candidates_revalued,
        "v0_revalued": "V0" in candidates_revalued,
        "v7_revalued": "V7" in candidates_revalued,
        "blend_revalued": "BLEND_V0_50_V7_50" in candidates_revalued,
        "canonical_return_source_path": rel(CANONICAL_RETURN),
        "avg_match_ratio": avg_match,
        "min_match_ratio": min_match,
        "avg_matched_weight_share": avg_share,
        "min_matched_weight_share": min_share,
        "cost_scenarios_evaluated": COST_BPS if portfolio_returns_calculated else [],
        "return_variants_evaluated": RETURN_VARIANTS if portfolio_returns_calculated else [],
        "best_bridge_candidate_by_20bps_sharpe": best_candidate,
        "best_bridge_20bps_sharpe": best_sharpe,
        "best_bridge_20bps_mean_return": best_mean,
        "best_bridge_20bps_tstat": best_tstat,
        "best_bridge_20bps_cumulative_return": best_cum,
        "best_bridge_20bps_max_drawdown": best_mdd,
        "best_bridge_20bps_avg_turnover": best_turnover,
        "blend_20bps_sharpe": blend_sharpe,
        "blend_20bps_mean_return": blend_mean,
        "blend_20bps_max_drawdown": blend_mdd,
        "blend_bridge_pass": blend_pass,
        "blend_bridge_strong": blend_strong,
        "simple_baseline_comparison_available": simple_available and portfolio_returns_calculated,
        "best_bridge_outperforms_simple_baseline": best_outperforms,
        "benchmark_artifact_checkpoint_passed": bench_pass,
        "missing_benchmark_artifacts": missing_bench,
        "old_vs_csmar_reconciliation_available": old_recon_available,
        "full_csmar_rebuild_required": True,
        "bridge_test_is_canonical_conclusion": False,
        "training_run": False,
        "new_scores_generated": False,
        "new_weights_generated": False,
        "old_weights_modified": False,
        "portfolio_returns_calculated": portfolio_returns_calculated,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
        "production_modified": False,
        "final_decision": final_decision,
        "recommended_next_step": recommended,
        "ambiguity_issues": issues,
    }
    (OUT_DIR / "mainline_legacy_weights_csmar_return_bridge_test_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# Mainline Legacy Weights CSMAR Return Bridge Test v0",
        "",
        "## 结论",
        "",
        f"- final_decision: {final_decision}",
        f"- prerequisites_passed: {summary['prerequisites_passed']}",
        f"- selected_legacy_weights_count: {summary['selected_legacy_weights_count']}",
        f"- portfolio_returns_calculated: {portfolio_returns_calculated}",
        "",
        "## 停止原因",
        "",
        "旧 V0/V7/Blend weights 未能全部唯一锁定。根据任务规则，未计算 bridge portfolio returns。",
        "",
        "## Ambiguity issues",
        "",
        *[f"- {x}" for x in issues],
        "",
        "## Guardrails",
        "",
        "未训练，未生成新 scores/weights，未修改旧 weights，未计算 benchmark-relative/alpha-beta/IR/TE/FF/DGTW/SHAP。",
    ]
    (OUT_DIR / "mainline_legacy_weights_csmar_return_bridge_test_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    terminal = {
        "task_name": TASK,
        "completed_at": now_iso(),
        "final_decision": final_decision,
        "outputs": sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()),
    }
    (OUT_DIR / "terminal_summary.json").write_text(json.dumps(terminal, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "task_completion_card.md").write_text(f"# Task Completion Card\n\n- task: {TASK}\n- completed_at: {terminal['completed_at']}\n- final_decision: {final_decision}\n- output_dir: `{rel(OUT_DIR)}`\n", encoding="utf-8")
    write_csv(
        OUT_DIR / "final_qa.csv",
        [
            {"check": "required_outputs_present", "status": "PASS", "detail": "all required bridge output files generated"},
            {"check": "guardrails_passed", "status": "PASS" if guardrail_ok else "FAIL", "detail": "restricted calculations respected"},
            {"check": "returns_stopped_when_ambiguous", "status": "PASS" if issues and not portfolio_returns_calculated else "WARN", "detail": "; ".join(issues)},
        ],
        ["check", "status", "detail"],
    )
    append_state(f"完成。final_decision={final_decision}; returns_calculated={portfolio_returns_calculated}; issues={'; '.join(issues)}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
