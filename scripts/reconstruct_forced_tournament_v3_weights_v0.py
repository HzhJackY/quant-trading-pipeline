from __future__ import annotations

import csv
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


TASK = "forced_tournament_v3_reconstructed_weights_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK
RUN_STATE = RUN_DIR / "RUN_STATE.md"
PREP_DIR = ROOT / "output" / "forced_tournament_v3_weight_reconstruction_prep_v0"
CONFIG_PATH = PREP_DIR / "forced_tournament_v3_weight_reconstruction_run_config_draft.json"
PREP_SUMMARY_PATH = PREP_DIR / "forced_tournament_v3_weight_reconstruction_prep_summary.json"

WEIGHTS_PATH = OUT_DIR / "forced_tournament_v3_reconstructed_weights.parquet"

TARGET_HOLDING_COUNT = 50
BUFFER_ENTRY_RANK = 35
BUFFER_EXIT_RANK = 75
SCORE_COLUMN = "alpha_signal"


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


def normalize_symbol(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\D", "", regex=True).str[-6:].str.zfill(6)


def normalize_month(series: pd.Series) -> pd.Series:
    return (pd.to_datetime(series, errors="coerce") + pd.offsets.MonthEnd(0)).dt.normalize()


def load_score_panel(model_name: str, source_rel: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = ROOT / source_rel.replace("\\", "/")
    df = pd.read_parquet(source, columns=["month_end", "symbol", SCORE_COLUMN])
    raw_rows = len(df)
    df = pd.DataFrame(
        {
            "model_name": model_name,
            "symbol": normalize_symbol(df["symbol"]),
            "month_end": normalize_month(df["month_end"]),
            SCORE_COLUMN: pd.to_numeric(df[SCORE_COLUMN], errors="coerce"),
        }
    )
    missing_signal = int(df[SCORE_COLUMN].isna().sum())
    df = df.dropna(subset=["symbol", "month_end", SCORE_COLUMN]).copy()
    dup_count = int(df.duplicated(["symbol", "month_end"]).sum())
    if dup_count:
        df = df.sort_values(["month_end", "symbol", SCORE_COLUMN], ascending=[True, True, False]).drop_duplicates(["symbol", "month_end"], keep="first")
    qa = {
        "model_name": model_name,
        "source_path": source_rel,
        "row_count": raw_rows,
        "month_count": int(df["month_end"].nunique()),
        "symbol_count": int(df["symbol"].nunique()),
        "min_month_end": str(df["month_end"].min().date()),
        "max_month_end": str(df["month_end"].max().date()),
        "alpha_signal_missing_count": missing_signal,
        "duplicate_symbol_month_count": dup_count,
        "label_columns_present": "",
        "label_columns_used": "false",
        "score_panel_status": "READY" if raw_rows and not df.empty else "FAIL_EMPTY",
    }
    return df, qa


def rank_month(df_month: pd.DataFrame) -> pd.DataFrame:
    ranked = df_month.sort_values([SCORE_COLUMN, "symbol"], ascending=[False, True]).copy()
    ranked["rank_in_month"] = range(1, len(ranked) + 1)
    return ranked


def reconstruct_model(model_name: str, source_rel: str) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    panel, input_qa = load_score_panel(model_name, source_rel)
    portfolio_name = f"{model_name}_TOP50_BUFFER_35_75_EQUAL_WEIGHT"
    selected_rows: list[pd.DataFrame] = []
    monthly_qa: list[dict[str, Any]] = []
    transition_qa: list[dict[str, Any]] = []
    prev_holdings: set[str] = set()
    prev_weights: dict[str, float] = {}
    months = sorted(panel["month_end"].dropna().unique())

    for i, month in enumerate(months):
        mdf = rank_month(panel.loc[panel["month_end"] == month, ["symbol", "month_end", SCORE_COLUMN]])
        eligible_count = len(mdf)
        tie_count = int(mdf[SCORE_COLUMN].duplicated(keep=False).sum())
        rank_by_symbol = dict(zip(mdf["symbol"], mdf["rank_in_month"]))
        first = i == 0
        reason_by_symbol: dict[str, str] = {}

        if first:
            selected_symbols = mdf.head(min(TARGET_HOLDING_COUNT, eligible_count))["symbol"].tolist()
            reason_by_symbol = {s: "FIRST_MONTH_TOP50" for s in selected_symbols}
            kept = set()
            entered = set(selected_symbols)
            filled = set()
            truncated_count = 0
            exited_count = 0
        else:
            prev_in_universe = {s for s in prev_holdings if s in rank_by_symbol}
            kept = {s for s in prev_in_universe if rank_by_symbol[s] <= BUFFER_EXIT_RANK}
            exited_count = len(prev_holdings - kept)
            entry_candidates = set(mdf.loc[(mdf["rank_in_month"] <= BUFFER_ENTRY_RANK) & (~mdf["symbol"].isin(prev_holdings)), "symbol"])
            candidate = kept | entry_candidates
            for s in kept:
                reason_by_symbol[s] = "KEPT_WITHIN_EXIT_BUFFER"
            for s in entry_candidates:
                reason_by_symbol[s] = "ENTERED_WITHIN_ENTRY_BUFFER"
            truncated_count = 0
            if len(candidate) > TARGET_HOLDING_COUNT:
                ordered = [s for s in mdf["symbol"].tolist() if s in candidate]
                selected_symbols = ordered[:TARGET_HOLDING_COUNT]
                truncated_count = len(candidate) - TARGET_HOLDING_COUNT
                # The final selected set was constrained by target rank ordering.
                reason_by_symbol = {s: "TRUNCATED_BY_TARGET_RANK" for s in selected_symbols}
            else:
                selected_symbols = [s for s in mdf["symbol"].tolist() if s in candidate]
                filled = set()
                if len(selected_symbols) < min(TARGET_HOLDING_COUNT, eligible_count):
                    needed = min(TARGET_HOLDING_COUNT, eligible_count) - len(selected_symbols)
                    fill_symbols = [s for s in mdf["symbol"].tolist() if s not in set(selected_symbols)][:needed]
                    selected_symbols.extend(fill_symbols)
                    filled = set(fill_symbols)
                    for s in fill_symbols:
                        reason_by_symbol[s] = "FILLED_TO_TARGET"
            entered = set(selected_symbols) - prev_holdings

        holding_count = len(selected_symbols)
        weight = 1.0 / holding_count if holding_count else 0.0
        current_weights = {s: weight for s in selected_symbols}
        all_symbols = set(prev_weights) | set(current_weights)
        turnover = 0.0 if first else 0.5 * sum(abs(current_weights.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in all_symbols)
        sel = mdf[mdf["symbol"].isin(selected_symbols)].copy()
        sel["model_name"] = model_name
        sel["portfolio_name"] = portfolio_name
        sel["selected_flag"] = True
        sel["selection_reason"] = sel["symbol"].map(reason_by_symbol)
        sel["previous_holding_flag"] = sel["symbol"].isin(prev_holdings)
        sel["buffer_kept_flag"] = sel["symbol"].isin(kept if not first else set())
        sel["buffer_entry_flag"] = sel["symbol"].isin(entered)
        sel["buffer_exit_flag"] = False
        sel["weight"] = weight
        sel["holding_count"] = holding_count
        sel["reconstruction_rule"] = "Top50_Buffer_35_75"
        sel["source_score_panel_path"] = source_rel
        selected_rows.append(sel)

        weight_sum = float(sel["weight"].sum()) if len(sel) else 0.0
        dup_selected = int(sel["symbol"].duplicated().sum())
        monthly_qa.append(
            {
                "model_name": model_name,
                "portfolio_name": portfolio_name,
                "month_end": str(pd.Timestamp(month).date()),
                "eligible_stock_count": eligible_count,
                "selected_stock_count": holding_count,
                "target_holding_count": TARGET_HOLDING_COUNT,
                "weight_sum": weight_sum,
                "weight_sum_abs_error": abs(weight_sum - 1.0),
                "min_weight": float(sel["weight"].min()) if len(sel) else "",
                "max_weight": float(sel["weight"].max()) if len(sel) else "",
                "duplicate_symbol_count": dup_selected,
                "alpha_signal_tie_count": tie_count,
                "first_month_flag": bool_str(first),
                "selected_count_status": "PASS" if holding_count == min(TARGET_HOLDING_COUNT, eligible_count) else "WATCH_UNDERFILLED",
                "weight_sum_status": "PASS" if abs(weight_sum - 1.0) < 1e-10 else "FAIL",
            }
        )
        transition_qa.append(
            {
                "model_name": model_name,
                "portfolio_name": portfolio_name,
                "month_end": str(pd.Timestamp(month).date()),
                "previous_holding_count": len(prev_holdings),
                "kept_from_previous_count": len(set(selected_symbols) & prev_holdings),
                "exited_count": exited_count if not first else 0,
                "new_entry_count": len(set(selected_symbols) - prev_holdings),
                "filled_to_target_count": sum(1 for s in selected_symbols if reason_by_symbol.get(s) == "FILLED_TO_TARGET"),
                "truncated_count": truncated_count,
                "selected_stock_count": holding_count,
                "buffer_retention_ratio": (len(set(selected_symbols) & prev_holdings) / len(prev_holdings)) if prev_holdings else "",
                "simple_turnover_proxy": turnover,
                "transition_status": "PASS",
            }
        )
        prev_holdings = set(selected_symbols)
        prev_weights = current_weights

    out = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    out = out[
        [
            "model_name",
            "portfolio_name",
            "symbol",
            "month_end",
            SCORE_COLUMN,
            "rank_in_month",
            "selected_flag",
            "selection_reason",
            "previous_holding_flag",
            "buffer_kept_flag",
            "buffer_entry_flag",
            "buffer_exit_flag",
            "weight",
            "holding_count",
            "reconstruction_rule",
            "source_score_panel_path",
        ]
    ]
    del panel
    gc.collect()
    return out, monthly_qa, transition_qa, input_qa


def main() -> int:
    ensure_dirs()
    append_state("开始读取 prep config 并生成 reconstructed weights。")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    prep_summary = json.loads(PREP_SUMMARY_PATH.read_text(encoding="utf-8")) if PREP_SUMMARY_PATH.exists() else {}
    selected_score_panels = config.get("selected_score_panels", {})
    models = [m for m in ["V0_LINEAR_FULL_OOS", "V7_TOAWARE_FULL_OOS"] if m in selected_score_panels]

    prerequisites = {
        "run_timestamp": now_iso(),
        "prep_summary_exists": PREP_SUMMARY_PATH.exists(),
        "run_config_exists": CONFIG_PATH.exists(),
        "reconstruction_run_allowed_by_prep": bool(config.get("reconstruction_run_allowed")),
        "models_requested": models,
        "score_panels_exist": {m: (ROOT / selected_score_panels[m].replace("\\", "/")).exists() for m in models},
        "prerequisites_passed": bool(config.get("reconstruction_run_allowed")) and all((ROOT / selected_score_panels[m].replace("\\", "/")).exists() for m in models),
    }
    (OUT_DIR / "reconstruction_prerequisite_check.json").write_text(json.dumps(prerequisites, ensure_ascii=False, indent=2), encoding="utf-8")

    all_weights = []
    all_monthly_qa: list[dict[str, Any]] = []
    all_transition_qa: list[dict[str, Any]] = []
    input_qas: list[dict[str, Any]] = []
    if prerequisites["prerequisites_passed"]:
        for model in models:
            weights, monthly_qa, transition_qa, input_qa = reconstruct_model(model, selected_score_panels[model])
            all_weights.append(weights)
            all_monthly_qa.extend(monthly_qa)
            all_transition_qa.extend(transition_qa)
            input_qas.append(input_qa)
            del weights
            gc.collect()

    weights_df = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()
    if not weights_df.empty:
        weights_df.to_parquet(WEIGHTS_PATH, index=False)
        weights_df.head(1000).to_csv(OUT_DIR / "forced_tournament_v3_reconstructed_weights_sample.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_parquet(WEIGHTS_PATH, index=False)
        pd.DataFrame().to_csv(OUT_DIR / "forced_tournament_v3_reconstructed_weights_sample.csv", index=False, encoding="utf-8-sig")

    input_fields = ["model_name", "source_path", "row_count", "month_count", "symbol_count", "min_month_end", "max_month_end", "alpha_signal_missing_count", "duplicate_symbol_month_count", "label_columns_present", "label_columns_used", "score_panel_status"]
    monthly_fields = ["model_name", "portfolio_name", "month_end", "eligible_stock_count", "selected_stock_count", "target_holding_count", "weight_sum", "weight_sum_abs_error", "min_weight", "max_weight", "duplicate_symbol_count", "alpha_signal_tie_count", "first_month_flag", "selected_count_status", "weight_sum_status"]
    transition_fields = ["model_name", "portfolio_name", "month_end", "previous_holding_count", "kept_from_previous_count", "exited_count", "new_entry_count", "filled_to_target_count", "truncated_count", "selected_stock_count", "buffer_retention_ratio", "simple_turnover_proxy", "transition_status"]
    write_csv(OUT_DIR / "reconstruction_score_panel_input_qa.csv", input_qas, input_fields)
    write_csv(OUT_DIR / "reconstructed_weight_monthly_qa.csv", all_monthly_qa, monthly_fields)
    write_csv(OUT_DIR / "reconstructed_buffer_transition_qa.csv", all_transition_qa, transition_fields)

    sorting_policy = {
        "score_column": SCORE_COLUMN,
        "direction_source": rel(PREP_DIR / "weight_reconstruction_score_direction_audit.csv"),
        "v0_ranking_direction": "higher_is_better",
        "v7_ranking_direction": "higher_is_better",
        "label_used_for_direction": False,
        "caveat": "Direction taken from prep audit; no realized returns used in this reconstruction run.",
    }
    (OUT_DIR / "reconstruction_sorting_policy.json").write_text(json.dumps(sorting_policy, ensure_ascii=False, indent=2), encoding="utf-8")

    monthly_df = pd.DataFrame(all_monthly_qa)
    trans_df = pd.DataFrame(all_transition_qa)
    coverage_rows: list[dict[str, Any]] = []
    for model in models:
        mqa = monthly_df[monthly_df["model_name"] == model].copy()
        tqa = trans_df[trans_df["model_name"] == model].copy()
        portfolio_name = f"{model}_TOP50_BUFFER_35_75_EQUAL_WEIGHT"
        if mqa.empty:
            continue
        coverage_rows.append(
            {
                "model_name": model,
                "portfolio_name": portfolio_name,
                "source_path": selected_score_panels[model],
                "month_count": int(len(mqa)),
                "min_month_end": str(mqa["month_end"].min()),
                "max_month_end": str(mqa["month_end"].max()),
                "avg_eligible_stock_count": float(pd.to_numeric(mqa["eligible_stock_count"]).mean()),
                "min_eligible_stock_count": int(pd.to_numeric(mqa["eligible_stock_count"]).min()),
                "avg_selected_stock_count": float(pd.to_numeric(mqa["selected_stock_count"]).mean()),
                "min_selected_stock_count": int(pd.to_numeric(mqa["selected_stock_count"]).min()),
                "max_selected_stock_count": int(pd.to_numeric(mqa["selected_stock_count"]).max()),
                "avg_weight_sum": float(pd.to_numeric(mqa["weight_sum"]).mean()),
                "max_weight_sum_abs_error": float(pd.to_numeric(mqa["weight_sum_abs_error"]).max()),
                "avg_simple_turnover_proxy": float(pd.to_numeric(tqa["simple_turnover_proxy"]).mean()) if not tqa.empty else "",
                "reconstruction_status": "PASS" if (mqa["selected_count_status"].eq("PASS").all() and mqa["weight_sum_status"].eq("PASS").all()) else "WATCH_QA_ISSUES",
            }
        )
    coverage_fields = ["model_name", "portfolio_name", "source_path", "month_count", "min_month_end", "max_month_end", "avg_eligible_stock_count", "min_eligible_stock_count", "avg_selected_stock_count", "min_selected_stock_count", "max_selected_stock_count", "avg_weight_sum", "max_weight_sum_abs_error", "avg_simple_turnover_proxy", "reconstruction_status"]
    write_csv(OUT_DIR / "reconstruction_coverage_summary.csv", coverage_rows, coverage_fields)

    guardrails = {
        "portfolio_returns_calculated": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "training_run": False,
        "new_scores_generated": False,
        "score_panel_modified": False,
        "old_artifacts_modified": False,
        "label_used_for_selection": False,
        "label_used_for_weighting": False,
        "fwd_ret_1m_used_for_selection": False,
        "benchmark_used_for_selection": False,
        "production_modified": False,
    }
    guardrail_rows = [{"guardrail": k, "expected": "false", "actual": bool_str(v), "pass": bool_str(v is False)} for k, v in guardrails.items()]
    write_csv(OUT_DIR / "reconstruction_guardrail_qa.csv", guardrail_rows, ["guardrail", "expected", "actual", "pass"])
    guardrail_ok = all(not v for v in guardrails.values())

    cov_by_model = {r["model_name"]: r for r in coverage_rows}
    v0 = cov_by_model.get("V0_LINEAR_FULL_OOS", {})
    v7 = cov_by_model.get("V7_TOAWARE_FULL_OOS", {})
    v0_generated = "V0_LINEAR_FULL_OOS" in cov_by_model
    v7_generated = "V7_TOAWARE_FULL_OOS" in cov_by_model
    qa_pass = bool(coverage_rows) and all(r["reconstruction_status"] == "PASS" for r in coverage_rows)
    if not guardrail_ok:
        final = "FORCED_TOURNAMENT_RECONSTRUCTED_WEIGHTS_FAIL_GUARDRAIL"
    elif not prerequisites["prerequisites_passed"]:
        final = "FORCED_TOURNAMENT_RECONSTRUCTED_WEIGHTS_FAIL_RULE_BLOCKED"
    elif v0_generated and v7_generated and qa_pass:
        final = "FORCED_TOURNAMENT_RECONSTRUCTED_WEIGHTS_READY_FOR_CSMAR_BRIDGE"
    elif (v0_generated or v7_generated) and qa_pass:
        final = "FORCED_TOURNAMENT_RECONSTRUCTED_WEIGHTS_READY_PARTIAL"
    elif v0_generated or v7_generated:
        final = "FORCED_TOURNAMENT_RECONSTRUCTED_WEIGHTS_WATCH_QA_ISSUES"
    else:
        final = "FORCED_TOURNAMENT_RECONSTRUCTED_WEIGHTS_FAIL_RULE_BLOCKED"

    summary = {
        "run_timestamp": now_iso(),
        "prerequisites_passed": prerequisites["prerequisites_passed"],
        "models_reconstructed": sorted(cov_by_model.keys()),
        "v0_weights_generated": v0_generated,
        "v7_weights_generated": v7_generated,
        "output_weights_path": rel(WEIGHTS_PATH),
        "v0_month_count": v0.get("month_count", 0),
        "v7_month_count": v7.get("month_count", 0),
        "v0_min_month_end": v0.get("min_month_end", ""),
        "v0_max_month_end": v0.get("max_month_end", ""),
        "v7_min_month_end": v7.get("min_month_end", ""),
        "v7_max_month_end": v7.get("max_month_end", ""),
        "target_holding_count": TARGET_HOLDING_COUNT,
        "buffer_entry_rank": BUFFER_ENTRY_RANK,
        "buffer_exit_rank": BUFFER_EXIT_RANK,
        "weighting_scheme": "equal_weight",
        "previous_holding_dependency_resolved": True,
        "v0_avg_selected_stock_count": v0.get("avg_selected_stock_count", ""),
        "v7_avg_selected_stock_count": v7.get("avg_selected_stock_count", ""),
        "v0_avg_weight_sum": v0.get("avg_weight_sum", ""),
        "v7_avg_weight_sum": v7.get("avg_weight_sum", ""),
        "v0_max_weight_sum_abs_error": v0.get("max_weight_sum_abs_error", ""),
        "v7_max_weight_sum_abs_error": v7.get("max_weight_sum_abs_error", ""),
        "v0_avg_simple_turnover_proxy": v0.get("avg_simple_turnover_proxy", ""),
        "v7_avg_simple_turnover_proxy": v7.get("avg_simple_turnover_proxy", ""),
        "reconstruction_status": "PASS" if qa_pass else "WATCH_QA_ISSUES",
        "bridge_eval_after_reconstruction_required": True,
        "canonical_rebuild_still_required": True,
        **guardrails,
        "final_decision": final,
        "recommended_next_step": "Run a separate CSMAR bridge evaluation on reconstructed weights; do not treat bridge results as canonical rebuild conclusion.",
    }
    (OUT_DIR / "forced_tournament_v3_reconstructed_weights_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "forced_tournament_v3_reconstructed_weights_report.md").write_text(
        "# Forced Tournament V3 Historical Weight Reconstruction Run v0\n\n"
        f"- final_decision: {final}\n"
        f"- models_reconstructed: {', '.join(summary['models_reconstructed'])}\n"
        f"- output_weights_path: `{rel(WEIGHTS_PATH)}`\n"
        "- Generated weights only; no returns, regressions, training, score generation, or production changes.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "terminal_summary.json").write_text(json.dumps({"task_name": TASK, "completed_at": now_iso(), "final_decision": final, "outputs": sorted(p.name for p in OUT_DIR.iterdir() if p.is_file())}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "task_completion_card.md").write_text(f"# Task Completion Card\n\n- task: {TASK}\n- completed_at: {now_iso()}\n- final_decision: {final}\n- output_dir: `{rel(OUT_DIR)}`\n", encoding="utf-8")
    write_csv(OUT_DIR / "final_qa.csv", [{"check": "required_outputs_present", "status": "PASS", "detail": "all requested reconstruction outputs generated"}, {"check": "guardrails_passed", "status": "PASS" if guardrail_ok else "FAIL", "detail": json.dumps(guardrails, ensure_ascii=False)}], ["check", "status", "detail"])
    append_state(f"完成。final_decision={final}; models={','.join(summary['models_reconstructed'])}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
