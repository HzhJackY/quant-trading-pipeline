from __future__ import annotations

import gc
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "factor_transform_planning_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "factor_transform_planning_v0"
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

INPUTS = {
    "v3_panel": ROOT / "output" / "csmar_pit_clean_core_financial_factors_v3" / "pit_clean_core_financial_factors_monthly_v3.parquet",
    "fi_t5_sanity_summary": ROOT / "output" / "fi_t5_sanity_check_v0" / "fi_t5_sanity_check_summary.json",
    "fi_t5_sanity_metrics": ROOT / "output" / "fi_t5_sanity_check_v0" / "sanity_metrics_summary.csv",
    "fi_t5_field_mapping": ROOT / "output" / "fi_t5_sanity_check_v0" / "field_mapping.csv",
    "fi_t5_watch_summary": ROOT / "output" / "fi_t5_watch_review_v01" / "fi_t5_watch_review_summary.json",
    "fi_t5_watch_report": ROOT / "output" / "fi_t5_watch_review_v01" / "fi_t5_watch_review_report.md",
    "roe_review_summary": ROOT / "output" / "roe_formula_review_v02" / "roe_formula_review_summary.json",
    "roe_review_report": ROOT / "output" / "roe_formula_review_v02" / "roe_formula_review_report.md",
    "roe_candidate_comparison": ROOT / "output" / "roe_formula_review_v02" / "roe_candidate_formula_comparison.csv",
    "roe_candidate_availability": ROOT / "output" / "roe_formula_review_v02" / "roe_candidate_formula_availability.csv",
}

ID_COLS = {
    "symbol",
    "month_end",
    "selected_report_period",
    "selected_pit_date",
    "market_cap_trade_date",
}


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def update_run_state(stage: str, completed: list[str], generated: list[str], next_step: str) -> None:
    body = f"""# RUN_STATE

- 当前任务名称: Factor Transform Planning v0
- 开始时间: see CHECKPOINTS.md initial checkpoint
- 当前阶段: {stage}
- 已完成步骤:
{chr(10).join(f"  - {x}" for x in completed)}
- 正在处理的文件:
{chr(10).join(f"  - {str(p.relative_to(ROOT)).replace(chr(92), '/')}" for p in INPUTS.values())}
- 已生成输出:
{chr(10).join(f"  - {x}" for x in generated)}
- 下一步:
  - {next_step}
- 如果 Codex 崩溃，新的 Codex 应如何继续:
  - 先读取本文件
  - 不扫描全仓，不读取白名单以外文件
  - 继续运行 python scripts/plan_factor_transform_v0.py，并将 stdout/stderr 重定向到 run_stdout.txt/run_stderr.txt
"""
    write_text(RUN_DIR / "RUN_STATE.md", body)
    with (RUN_DIR / "CHECKPOINTS.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now().isoformat(timespec='seconds')} {stage}\n\n")
        for item in completed:
            f.write(f"- {item}\n")
        f.write(f"- 下一步: {next_step}\n")


def read_json_optional(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_unique_count(s: pd.Series) -> int:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return int(s.astype("string").nunique(dropna=True))


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        inf_flag = False
        stats = {"min": None, "p1": None, "p5": None, "p25": None, "p50": None, "p75": None, "p95": None, "p99": None, "max": None}
        extreme_flag = False
        if is_num:
            arr = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
            inf_flag = bool(np.isinf(pd.to_numeric(s, errors="coerce")).any())
            valid = arr.dropna()
            if len(valid) > 0:
                qs = valid.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
                stats = {
                    "min": float(valid.min()),
                    "p1": float(qs.loc[0.01]),
                    "p5": float(qs.loc[0.05]),
                    "p25": float(qs.loc[0.25]),
                    "p50": float(qs.loc[0.50]),
                    "p75": float(qs.loc[0.75]),
                    "p95": float(qs.loc[0.95]),
                    "p99": float(qs.loc[0.99]),
                    "max": float(valid.max()),
                }
                iqr = stats["p75"] - stats["p25"]
                if iqr > 0:
                    extreme_flag = bool((stats["max"] > stats["p75"] + 10 * iqr) or (stats["min"] < stats["p25"] - 10 * iqr))
                elif stats["max"] != stats["min"]:
                    extreme_flag = True
            del arr
        rows.append(
            {
                "column_name": col,
                "dtype": str(s.dtype),
                "non_null_count": non_null,
                "coverage": non_null / n if n else 0.0,
                **stats,
                "unique_values": safe_unique_count(s),
                "constant_flag": safe_unique_count(s) <= 1,
                "infinite_flag": inf_flag,
                "extreme_outlier_flag": extreme_flag,
            }
        )
    gc.collect()
    return pd.DataFrame(rows)


def classify_column(col: str) -> tuple[str, str]:
    c = col.lower()
    if col in ID_COLS or "date" in c or "period" in c or c in {"symbol", "code", "ticker"}:
        return "identifier_date", "EXCLUDE"
    if any(x in c for x in ["flag", "source", "warm", "audit", "valid", "available"]):
        return "audit_helper", "EXCLUDE"
    if any(x in c for x in ["revenue", "profit", "equity", "asset", "liabilit", "expense", "income", "cash", "cost"]) and not any(
        x in c for x in ["ratio", "margin", "growth", "turnover", "roe", "roa"]
    ):
        return "raw_accounting_component", "RAW_COMPONENT_ONLY"
    if any(x in c for x in ["ep", "bp", "earnings_yield", "book_to_market", "bm"]):
        return "value", "READY"
    if any(x in c for x in ["roe", "roa", "margin", "expense_ratio", "debt_ratio", "asset_turnover"]):
        return "quality", "READY_WITH_NOTE" if "admin" in c else "READY"
    if any(x in c for x in ["growth", "yoy", "qoq"]):
        return "growth", "READY_WITH_NOTE"
    if any(x in c for x in ["debt", "liability", "leverage", "beta", "volatility", "vol_"]):
        return "leverage_risk", "READY_WITH_NOTE"
    if any(x in c for x in ["market_cap", "mktcap", "size", "log_mcap", "liquidity", "turnover"]):
        return "size_liquidity", "READY"
    return "other_candidate", "WATCH"


def direction_for(col: str, group: str) -> str:
    c = col.lower()
    if any(x in c for x in ["ep", "bp", "roe", "roa", "margin", "growth", "asset_turnover"]):
        return "higher_better"
    if any(x in c for x in ["debt_ratio", "leverage", "liability_ratio"]):
        return "lower_generally_better"
    if "admin_expense_ratio" in c:
        return "lower_generally_better_but_business_model_dependent"
    if "sales_expense_ratio" in c or "expense_ratio" in c:
        return "unknown_context_dependent"
    if group in {"raw_accounting_component", "audit_helper", "identifier_date"}:
        return "not_model_feature"
    return "model_learned"


def winsor_for(col: str, group: str, extreme: bool) -> tuple[str, dict]:
    c = col.lower()
    if group in {"identifier_date", "audit_helper", "raw_accounting_component"}:
        return "none", {}
    if "admin_expense_ratio" in c:
        return "month_cross_section_mad_with_quantile_fallback", {"mad_k": 5, "fallback_quantiles": [0.005, 0.995]}
    if any(x in c for x in ["ratio", "margin", "roe", "roa", "growth"]) or extreme:
        return "month_cross_section_mad_with_quantile_fallback", {"mad_k": 5, "fallback_quantiles": [0.01, 0.99]}
    return "month_cross_section_quantile", {"lower": 0.01, "upper": 0.99}


def build_inventory(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in profile.to_dict("records"):
        group, status = classify_column(r["column_name"])
        rows.append(
            {
                "column_name": r["column_name"],
                "factor_group": group,
                "dtype": r["dtype"],
                "coverage": r["coverage"],
                "non_null_count": r["non_null_count"],
                "review_status": status,
                "direction": direction_for(r["column_name"], group),
                "constant_flag": r["constant_flag"],
                "infinite_flag": r["infinite_flag"],
                "extreme_outlier_flag": r["extreme_outlier_flag"],
                "notes": notes_for(r["column_name"], group, status),
            }
        )
    return pd.DataFrame(rows)


def notes_for(col: str, group: str, status: str) -> str:
    c = col.lower()
    if col == "roe_ttm":
        return "Alias recommended: roe_parent_ttm_ending_equity; do not treat as strict FI_T5 F050501B replication."
    if "admin_expense_ratio" in c:
        return "FI_T5 sanity PASS_WITH_OUTLIER_NOTE; raw value must be robust clipped before modeling."
    if "sales_expense_ratio" in c:
        return "FI_T5 sanity PASS."
    if c in {"ep", "bp"} or "market_cap" in c:
        return "EP/BP unit review cleared; transform distribution only, do not recalculate units."
    if status == "RAW_COMPONENT_ONLY":
        return "Retain for audit or alternative formula construction; not direct model feature in v0."
    return ""


def build_spec(inventory: pd.DataFrame, profile: pd.DataFrame, columns: set[str]) -> list[dict]:
    profile_by_col = profile.set_index("column_name").to_dict("index")
    specs = []
    for row in inventory.to_dict("records"):
        col = row["column_name"]
        group = row["factor_group"]
        status = row["review_status"]
        winsor_method, winsor_params = winsor_for(col, group, bool(profile_by_col[col]["extreme_outlier_flag"]))
        enabled = status in {"READY", "READY_WITH_NOTE", "WATCH"}
        specs.append(
            {
                "factor_name": col,
                "source_column": col,
                "factor_group": group,
                "enabled_for_candidate_training": bool(enabled and group not in {"identifier_date", "audit_helper"}),
                "transform_method": "raw_clipped_plus_month_cross_section_rank_and_zscore" if enabled else "metadata_or_raw_component_only",
                "winsor_method": winsor_method,
                "winsor_params": winsor_params,
                "rank_method": "percentile_rank_by_month_within_locked_csi800_v3_universe" if enabled else "none",
                "zscore_method": "standard_zscore_by_month_after_winsor_within_locked_csi800_v3_universe" if enabled else "none",
                "missing_policy": missing_policy_for(col, group),
                "add_missing_indicator": bool(enabled and group not in {"identifier_date", "audit_helper"}),
                "direction": row["direction"],
                "neutralization_candidate": neutralization_candidate_for(col, group),
                "notes": row["notes"],
                "review_status": status,
            }
        )
    if {"net_profit_ttm", "total_equity"}.issubset(columns) or {"net_profit_parent_ttm", "total_equity"}.issubset(columns):
        src = "net_profit_ttm / total_equity" if "net_profit_ttm" in columns else "net_profit_parent_ttm / total_equity"
        specs.append(
            {
                "factor_name": "roe_total_ttm_ending_equity",
                "source_column": src,
                "factor_group": "quality",
                "enabled_for_candidate_training": True,
                "transform_method": "derived_candidate_then_raw_clipped_plus_month_cross_section_rank_and_zscore",
                "winsor_method": "month_cross_section_mad_with_quantile_fallback",
                "winsor_params": {"mad_k": 5, "fallback_quantiles": [0.01, 0.99]},
                "rank_method": "percentile_rank_by_month_within_locked_csi800_v3_universe",
                "zscore_method": "standard_zscore_by_month_after_winsor_within_locked_csi800_v3_universe",
                "missing_policy": "derive only when numerator and denominator are PIT-available and denominator is valid; otherwise missing plus indicator",
                "add_missing_indicator": True,
                "direction": "higher_better",
                "neutralization_candidate": "experiment_only",
                "notes": "Alternative ROE candidate from ROE Formula Review; do not replace production roe_ttm directly.",
                "review_status": "READY_WITH_NOTE",
            }
        )
    return specs


def missing_policy_for(col: str, group: str) -> str:
    c = col.lower()
    if group == "identifier_date":
        return "required_metadata_no_fill"
    if "market_cap" in c:
        return "market_cap_missing_remains_missing; no zero fill"
    if any(x in c for x in ["ttm", "growth", "yoy", "qoq"]):
        return "distinguish true missing, PIT unavailable, warm-up unavailable, and accounting not disclosed; add indicator"
    if any(x in c for x in ["ratio", "margin", "roe", "roa", "ep", "bp"]):
        return "denominator invalid or PIT unavailable remains missing; add indicator; no blind zero fill"
    if group == "raw_accounting_component":
        return "audit/raw component missing preserved"
    return "preserve missing; add model-stage missing indicator when used"


def neutralization_candidate_for(col: str, group: str) -> str:
    if group in {"identifier_date", "audit_helper", "raw_accounting_component"}:
        return "none"
    return "future_experiment_only"


def detect_neutralization(columns: set[str]) -> dict:
    lower = {c.lower(): c for c in columns}
    industry_cols = [c for c in columns if "industry" in c.lower() or c.lower() in {"sw_l1", "sw_industry", "申万行业"}]
    log_size_cols = [c for c in columns if "log" in c.lower() and ("cap" in c.lower() or "mkt" in c.lower() or "size" in c.lower())]
    market_cap_cols = [c for c in columns if "market_cap" in c.lower() or "mktcap" in c.lower() or c.lower() in {"mcap", "size"}]
    beta_cols = [c for c in columns if "beta" in c.lower()]
    vol_cols = [c for c in columns if "volatility" in c.lower() or c.lower().startswith("vol_")]
    if industry_cols and (log_size_cols or market_cap_cols):
        recommendation = "size + industry neutralization feasible as future experiment"
    elif industry_cols:
        recommendation = "industry neutralization feasible as future experiment"
    elif log_size_cols or market_cap_cols:
        recommendation = "size neutralization feasible as future experiment"
    else:
        recommendation = "no neutralization feasible from v3 columns"
    return {
        "industry_columns": industry_cols,
        "log_market_cap_columns": log_size_cols,
        "market_cap_columns": market_cap_cols,
        "beta_columns": beta_cols,
        "volatility_columns": vol_cols,
        "recommendation": recommendation,
        "executed": False,
    }


def write_reports(df: pd.DataFrame, profile: pd.DataFrame, inventory: pd.DataFrame, specs: list[dict], neutral: dict) -> dict:
    rows = int(len(df))
    symbols = int(df["symbol"].nunique()) if "symbol" in df.columns else None
    months = int(df["month_end"].nunique()) if "month_end" in df.columns else None
    counts = pd.Series([x["review_status"] for x in specs]).value_counts().to_dict()
    summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "v3_file_used": str(INPUTS["v3_panel"].relative_to(ROOT)).replace("\\", "/"),
        "rows": rows,
        "symbols": symbols,
        "months": months,
        "factor_count_total": len(specs),
        "factor_count_ready": int(counts.get("READY", 0)),
        "factor_count_ready_with_note": int(counts.get("READY_WITH_NOTE", 0)),
        "factor_count_watch": int(counts.get("WATCH", 0)),
        "factor_count_exclude": int(counts.get("EXCLUDE", 0)),
        "raw_component_only_count": int(counts.get("RAW_COMPONENT_ONLY", 0)),
        "roe_handling": {
            "roe_ttm_current_formula": "net_profit_parent_ttm / equity_parent",
            "recommended_alias": "roe_parent_ttm_ending_equity",
            "do_not_overwrite_original_column": True,
            "alternative_candidate": "roe_total_ttm_ending_equity if components exist",
            "production_replacement": False,
        },
        "expense_ratio_handling": {
            "sales_expense_ratio": "FI_T5 sanity PASS",
            "admin_expense_ratio": "PASS_WITH_OUTLIER_NOTE; robust winsorize/clip before modeling; do not use unclipped raw value",
        },
        "neutralization_feasibility": neutral,
        "production_modified": False,
        "v3_modified": False,
        "training_run": False,
        "backtest_run": False,
        "ic_calculated": False,
        "transformed_panel_built": False,
        "final_decision": "FACTOR_TRANSFORM_PLANNING_READY_FOR_TRANSFORMED_PANEL_BUILD",
        "recommended_next_step": "Build Transformed Training Panel v0",
    }

    profile.to_csv(OUT_DIR / "v3_column_profile.csv", index=False, encoding="utf-8-sig")
    inventory.to_csv(OUT_DIR / "factor_inventory.csv", index=False, encoding="utf-8-sig")
    write_json(OUT_DIR / "factor_transform_spec_v0.json", specs)
    write_json(OUT_DIR / "factor_transform_planning_summary.json", summary)
    write_training_plan(neutral)
    write_main_report(summary, inventory, specs, neutral)
    write_completion_files(summary)
    return summary


def write_training_plan(neutral: dict) -> None:
    text = f"""# Training Panel Integration Plan

1. Use the v3 PIT-clean source panel as the only financial factor source input.
2. Recommended transformed panel path: output/transformed_training_panel_v0/transformed_training_panel_v0.parquet.
3. Metadata columns to preserve: symbol, month_end, selected_report_period, selected_pit_date, market_cap_trade_date.
4. Model feature columns should be generated from READY / READY_WITH_NOTE / reviewed WATCH factors using clipped raw, month cross-sectional rank, month cross-sectional zscore, and missing indicators.
5. Raw accounting components remain audit/helper columns unless explicitly used to derive approved candidate factors.
6. PIT leakage control: never use data beyond selected_pit_date; missing because accounting was not disclosed remains missing.
7. Universe control: all cross-sectional transforms must be computed within the locked CSI800 v3 universe by month.
8. Rank baseline control: never rank against full A-share or a mixed universe.
9. Versioning: store transform_version = factor_transform_v0 and write the exact spec JSON path into panel metadata.
10. Compact-F / BLEND compatibility: transformed panel build should be followed by revalidation only; this task does not replace existing production candidates.

Neutralization feasibility: {neutral['recommendation']}. Neutralization is a later experiment and is not executed here.
"""
    write_text(OUT_DIR / "training_panel_integration_plan.md", text)


def write_main_report(summary: dict, inventory: pd.DataFrame, specs: list[dict], neutral: dict) -> None:
    group_lines = inventory.groupby("factor_group")["column_name"].apply(lambda s: ", ".join(s.astype(str))).to_dict()
    status_counts = pd.Series([x["review_status"] for x in specs]).value_counts().to_dict()
    text = f"""# Factor Transform Planning v0

## 1. Scope

This task only designs transform planning, schema, coverage, and transform specification. It does not train, backtest, calculate IC, modify production, modify v3, or build a transformed panel.

## 2. Inputs

{chr(10).join(f"- {str(p.relative_to(ROOT)).replace(chr(92), '/')}" for p in INPUTS.values())}

## 3. v3 Source Panel Status

- rows: {summary['rows']}
- symbols: {summary['symbols']}
- months: {summary['months']}
- source file: {summary['v3_file_used']}
- v3 modified: false

## 4. Factor Inventory

{chr(10).join(f"- {k}: {v}" for k, v in group_lines.items())}

## 5. Transform Policy

- Winsorization: month cross-sectional winsorization inside the locked CSI800 v3 universe. Default is p1/p99 quantile clipping; ratio, growth, ROE/ROA/margin, and extreme columns use median +/- 5 MAD with p1/p99 fallback. Admin expense ratio uses robust clipping and may use p0.5/p99.5 fallback.
- Missing values: distinguish true missing, PIT unavailable, warm-up unavailable, denominator invalid, market cap missing, and accounting not disclosed yet. Do not blindly fill zero; add missing indicators at model input stage.
- Direction: EP/BP/ROE/margins/growth are generally higher better; leverage/debt ratios are generally lower better; sales expense ratio is context dependent; admin expense ratio is lower generally better but business-model dependent.
- Cross-sectional transforms: keep raw clipped value, percentile rank by month, zscore by month, and missing indicators. All cross-sectional transforms must use the locked CSI800 v3 universe by month; never full A-share and never mixed universes.

## 6. Neutralization Feasibility

Recommendation: {neutral['recommendation']}. Neutralization is a future experiment and is not executed in this task.

## 7. ROE Special Handling

Current roe_ttm is net_profit_parent_ttm / equity_parent. Recommended alias is roe_parent_ttm_ending_equity. Do not overwrite the original column. roe_total_ttm_ending_equity can be an alternative candidate if components exist, but it must not directly replace production.

## 8. Expense Ratio Special Handling

sales_expense_ratio FI_T5 sanity result is PASS. admin_expense_ratio is PASS_WITH_OUTLIER_NOTE; do not use raw unclipped admin expense ratio in modeling.

## 9. Transform Spec Summary

{chr(10).join(f"- {k}: {int(v)}" for k, v in sorted(status_counts.items()))}

## 10. Training Panel Integration Plan

See training_panel_integration_plan.md. The next panel build should apply this spec, preserve PIT metadata, record transform_version, and revalidate Compact-F / BLEND compatibility after panel construction.

## 11. Decision

{summary['final_decision']}

## 12. Recommended Next Step

{summary['recommended_next_step']}
"""
    write_text(OUT_DIR / "factor_transform_planning_report.md", text)


def write_completion_files(summary: dict) -> None:
    card = f"""# Task Completion Card

- task: Factor Transform Planning v0
- final_decision: {summary['final_decision']}
- production_modified: false
- v3_modified: false
- training_run: false
- backtest_run: false
- ic_calculated: false
- transformed_panel_built: false
- recommended_next_step: {summary['recommended_next_step']}
"""
    write_text(OUT_DIR / "task_completion_card.md", card)
    terminal = {
        "task": "Factor Transform Planning v0",
        "completed": True,
        "outputs_dir": str(OUT_DIR.relative_to(ROOT)).replace("\\", "/"),
        "stdout_log": str((RUN_DIR / "run_stdout.txt").relative_to(ROOT)).replace("\\", "/"),
        "stderr_log": str((RUN_DIR / "run_stderr.txt").relative_to(ROOT)).replace("\\", "/"),
        "final_decision": summary["final_decision"],
    }
    write_json(OUT_DIR / "terminal_summary.json", terminal)
    qa = pd.DataFrame(
        [
            ["production_modified", summary["production_modified"]],
            ["v3_modified", summary["v3_modified"]],
            ["training_run", summary["training_run"]],
            ["backtest_run", summary["backtest_run"]],
            ["ic_calculated", summary["ic_calculated"]],
            ["transformed_panel_built", summary["transformed_panel_built"]],
            ["final_decision", summary["final_decision"]],
        ],
        columns=["check", "value"],
    )
    qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log("Checking explicit input allowlist.")
    missing = [str(p.relative_to(ROOT)).replace("\\", "/") for p in INPUTS.values() if not p.exists()]
    if missing:
        report = "# Missing Input Report\n\n" + "\n".join(f"- {x}" for x in missing) + "\n"
        write_text(OUT_DIR / "missing_input_report.md", report)
        update_run_state("missing_inputs", ["Checked input allowlist", "Generated missing_input_report.md"], ["output/factor_transform_planning_v0/missing_input_report.md"], "Provide missing inputs or rerun after they are restored")
        log("Missing required inputs; exiting gracefully.")
        return 2

    update_run_state("inputs_checked", ["Checked all explicit input paths"], ["output/_agent_runs/factor_transform_planning_v0/RUN_STATE.md"], "Read v3 parquet schema and profile columns")

    log("Reading v3 parquet panel.")
    df = pd.read_parquet(INPUTS["v3_panel"])
    log(f"Loaded v3 panel rows={len(df)} columns={len(df.columns)}.")

    sanity_summary = read_json_optional(INPUTS["fi_t5_sanity_summary"])
    watch_summary = read_json_optional(INPUTS["fi_t5_watch_summary"])
    roe_summary = read_json_optional(INPUTS["roe_review_summary"])
    _ = (sanity_summary, watch_summary, roe_summary)

    profile = column_profile(df)
    update_run_state("schema_profile_complete", ["Computed v3 schema and column-level profile"], ["output/factor_transform_planning_v0/v3_column_profile.csv"], "Build factor inventory and transform spec")

    inventory = build_inventory(profile)
    specs = build_spec(inventory, profile, set(df.columns))
    neutral = detect_neutralization(set(df.columns))
    summary = write_reports(df, profile, inventory, specs, neutral)

    del profile, inventory, specs, neutral
    del df
    gc.collect()

    generated = [
        "output/factor_transform_planning_v0/v3_column_profile.csv",
        "output/factor_transform_planning_v0/factor_inventory.csv",
        "output/factor_transform_planning_v0/factor_transform_spec_v0.json",
        "output/factor_transform_planning_v0/training_panel_integration_plan.md",
        "output/factor_transform_planning_v0/factor_transform_planning_report.md",
        "output/factor_transform_planning_v0/factor_transform_planning_summary.json",
        "output/factor_transform_planning_v0/task_completion_card.md",
        "output/factor_transform_planning_v0/terminal_summary.json",
        "output/factor_transform_planning_v0/final_qa.csv",
    ]
    update_run_state("completed", ["Built factor inventory", "Built transform spec", "Built reports and QA outputs"], generated, "Task complete")
    log(json.dumps({"final_decision": summary["final_decision"], "factor_count_total": summary["factor_count_total"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        write_text(OUT_DIR / "error_report.md", traceback.format_exc())
        update_run_state("failed", ["Encountered exception", f"Exception type: {type(exc).__name__}"], ["output/factor_transform_planning_v0/error_report.md"], "Inspect error_report.md and rerun after fixing the issue")
        print(traceback.format_exc(), file=sys.stderr)
        raise
