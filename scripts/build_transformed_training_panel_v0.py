from __future__ import annotations

import gc
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "build_transformed_training_panel_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "transformed_training_panel_v0"
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

SOURCE_PANEL = ROOT / "output" / "csmar_pit_clean_core_financial_factors_v3" / "pit_clean_core_financial_factors_monthly_v3.parquet"
SPEC_PATH = ROOT / "output" / "factor_transform_planning_v0" / "factor_transform_spec_v0.json"
INPUTS = {
    "source_panel": SOURCE_PANEL,
    "planning_summary": ROOT / "output" / "factor_transform_planning_v0" / "factor_transform_planning_summary.json",
    "factor_inventory": ROOT / "output" / "factor_transform_planning_v0" / "factor_inventory.csv",
    "transform_spec": SPEC_PATH,
    "integration_plan": ROOT / "output" / "factor_transform_planning_v0" / "training_panel_integration_plan.md",
    "planning_report": ROOT / "output" / "factor_transform_planning_v0" / "factor_transform_planning_report.md",
}

VALID_STATUSES = {"READY", "READY_WITH_NOTE", "WATCH", "EXCLUDE", "RAW_COMPONENT_ONLY"}
METADATA_COLS = [
    "symbol",
    "month_end",
    "selected_report_period",
    "selected_pit_date",
    "market_cap_trade_date",
    "ttm_complete_flag",
    "ttm_quarters_available",
    "uses_pre_2017_buffer_flag",
    "factor_validity_flags",
]
ROE_COMPONENT_COLS = ["net_profit_ttm", "total_equity", "net_profit_parent_ttm", "equity_parent"]
TRANSFORM_VERSION = "transform_v0"
SOURCE_PANEL_VERSION = "csmar_pit_clean_core_financial_factors_v3"


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def update_run_state(stage: str, completed: list[str], generated: list[str], next_step: str) -> None:
    body = f"""# RUN_STATE

- 当前任务名称: Build Transformed Training Panel v0
- 开始时间: see CHECKPOINTS.md initial checkpoint
- 当前阶段: {stage}
- 已完成步骤:
{chr(10).join(f"  - {x}" for x in completed)}
- 正在处理的文件:
  - scripts/build_transformed_training_panel_v0.py
{chr(10).join(f"  - {rel(p)}" for p in INPUTS.values())}
- 已生成输出:
{chr(10).join(f"  - {x}" for x in generated)}
- 下一步:
  - {next_step}
- 如果 Codex 崩溃，新的 Codex 应如何继续:
  - 先读取本文件
  - 不扫描全仓，不读取白名单以外文件
  - 继续运行 python scripts/build_transformed_training_panel_v0.py，并将 stdout/stderr 重定向到 run_stdout.txt/run_stderr.txt
"""
    write_text(RUN_DIR / "RUN_STATE.md", body)
    with (RUN_DIR / "CHECKPOINTS.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now().isoformat(timespec='seconds')} {stage}\n\n")
        for item in completed:
            f.write(f"- {item}\n")
        f.write(f"- 下一步: {next_step}\n")


def parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema.names)


def load_spec() -> list[dict]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def direct_source_columns(specs: list[dict]) -> set[str]:
    cols: set[str] = set()
    for spec in specs:
        src = str(spec.get("source_column", ""))
        if src and all(op not in src for op in ["/", "*", "+", "-"]) and " " not in src:
            cols.add(src)
    return cols


def read_source_panel(columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(SOURCE_PANEL, columns=columns)


def to_dt(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def base_qa(df: pd.DataFrame) -> dict:
    to_dt(df, "month_end")
    to_dt(df, "selected_pit_date")
    to_dt(df, "market_cap_trade_date")
    dup_count = int(df.duplicated(["symbol", "month_end"]).sum())
    pit_v = 0
    mcap_v = 0
    if {"selected_pit_date", "month_end"}.issubset(df.columns):
        pit_v = int((df["selected_pit_date"].notna() & df["month_end"].notna() & (df["selected_pit_date"] > df["month_end"])).sum())
    if {"market_cap_trade_date", "month_end"}.issubset(df.columns):
        mcap_v = int((df["market_cap_trade_date"].notna() & df["month_end"].notna() & (df["market_cap_trade_date"] > df["month_end"])).sum())
    return {
        "rows": int(len(df)),
        "symbols": int(df["symbol"].nunique()) if "symbol" in df.columns else None,
        "months": int(df["month_end"].nunique()) if "month_end" in df.columns else None,
        "one_row_per_symbol_month": bool(dup_count == 0),
        "duplicate_symbol_month_count": dup_count,
        "month_min": str(df["month_end"].min().date()) if "month_end" in df.columns and df["month_end"].notna().any() else None,
        "month_max": str(df["month_end"].max().date()) if "month_end" in df.columns and df["month_end"].notna().any() else None,
        "selected_pit_date_violation_count": pit_v,
        "market_cap_trade_date_violation_count": mcap_v,
        "row_count_matches_v3_expected": bool(len(df) == 77538),
        "symbol_count_matches_v3_expected": bool(("symbol" in df.columns) and df["symbol"].nunique() == 1352),
        "month_count_matches_v3_expected": bool(("month_end" in df.columns) and df["month_end"].nunique() == 114),
    }


def robust_clip_month(s: pd.Series, months: pd.Series, fallback_quantiles=(0.01, 0.99), mad_k=5.0) -> tuple[pd.Series, int]:
    warnings = 0
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    values = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for _, idx in values.groupby(months, sort=False).groups.items():
        x = values.loc[idx]
        valid = x.dropna()
        if len(valid) < 5:
            warnings += 1
            out.loc[idx] = x
            continue
        med = valid.median()
        mad = (valid - med).abs().median()
        if pd.isna(mad) or mad <= 0:
            lo, hi = valid.quantile(list(fallback_quantiles))
            warnings += 1
        else:
            lo, hi = med - mad_k * mad, med + mad_k * mad
        out.loc[idx] = x.clip(lower=lo, upper=hi)
    return out, warnings


def quantile_clip_month(s: pd.Series, months: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    values = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for _, idx in values.groupby(months, sort=False).groups.items():
        x = values.loc[idx]
        valid = x.dropna()
        if len(valid) < 2:
            out.loc[idx] = x
        else:
            lo, hi = valid.quantile([lower, upper])
            out.loc[idx] = x.clip(lower=lo, upper=hi)
    return out


def monthly_rank(s: pd.Series, months: pd.Series) -> pd.Series:
    return s.groupby(months, sort=False).rank(method="average", pct=True)


def monthly_zscore(s: pd.Series, months: pd.Series) -> tuple[pd.Series, int]:
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    warnings = 0
    for _, idx in s.groupby(months, sort=False).groups.items():
        x = s.loc[idx]
        valid = x.dropna()
        if len(valid) < 2:
            warnings += 1
            continue
        std = valid.std(ddof=0)
        if pd.isna(std) or std == 0:
            warnings += 1
            continue
        out.loc[idx] = (x - valid.mean()) / std
    return out, warnings


def parse_derived_source(src: str, df: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None, str]:
    if "/" not in src:
        return None, None, "not_derived"
    left, right = [x.strip() for x in src.split("/", 1)]
    if left not in df.columns or right not in df.columns:
        return None, None, f"missing_components:{left},{right}"
    num = pd.to_numeric(df[left], errors="coerce")
    den = pd.to_numeric(df[right], errors="coerce")
    invalid = den.isna() | (den == 0) | ~np.isfinite(den)
    value = num / den.where(~invalid)
    value = value.replace([np.inf, -np.inf], np.nan)
    return value, invalid, "derived"


def source_series_for(spec: dict, df: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None, str]:
    factor = spec["factor_name"]
    src = str(spec.get("source_column", ""))
    if factor == "roe_parent_ttm_ending_equity" and "roe_ttm" in df.columns:
        return pd.to_numeric(df["roe_ttm"], errors="coerce"), None, "roe_alias"
    if "/" in src:
        return parse_derived_source(src, df)
    if src in df.columns:
        return pd.to_numeric(df[src], errors="coerce"), None, "source_column"
    if factor == "roe_total_ttm_ending_equity":
        if {"net_profit_ttm", "total_equity"}.issubset(df.columns):
            return parse_derived_source("net_profit_ttm / total_equity", df)
        if {"net_profit_parent_ttm", "total_equity"}.issubset(df.columns):
            return parse_derived_source("net_profit_parent_ttm / total_equity", df)
    return None, None, "missing_source"


def transform_factor(spec: dict, df: pd.DataFrame, panel: pd.DataFrame) -> tuple[list[dict], list[dict], dict]:
    factor = spec["factor_name"]
    review_status = spec["review_status"]
    group = spec.get("factor_group", "")
    source, invalid, source_note = source_series_for(spec, df)
    if source is None:
        return [], [], {"factor": factor, "skipped": True, "reason": source_note}

    raw_col = f"{factor}_raw"
    clip_col = f"{factor}_clip"
    rank_col = f"{factor}_rank"
    z_col = f"{factor}_z"
    miss_col = f"{factor}_missing"
    invalid_col = f"{factor}_invalid"

    raw = pd.to_numeric(source, errors="coerce").replace([np.inf, -np.inf], np.nan)
    invalid_mask = invalid if invalid is not None else pd.to_numeric(source, errors="coerce").isin([np.inf, -np.inf])
    missing = raw.isna() | invalid_mask.fillna(False)

    winsor_method = str(spec.get("winsor_method", "month_cross_section_quantile"))
    winsor_params = spec.get("winsor_params") or {}
    z_warnings = 0
    clip_warnings = 0
    if "mad" in winsor_method:
        fallback = tuple(winsor_params.get("fallback_quantiles", [0.01, 0.99]))
        clip, clip_warnings = robust_clip_month(raw, df["month_end"], fallback_quantiles=fallback, mad_k=float(winsor_params.get("mad_k", 5)))
    elif "quantile" in winsor_method:
        clip = quantile_clip_month(raw, df["month_end"], lower=float(winsor_params.get("lower", 0.01)), upper=float(winsor_params.get("upper", 0.99)))
    else:
        clip = raw.copy()

    rank = monthly_rank(clip, df["month_end"])
    z, z_warnings = monthly_zscore(clip, df["month_end"])

    panel[raw_col] = raw
    panel[clip_col] = clip
    panel[rank_col] = rank
    panel[z_col] = z
    panel[miss_col] = missing.astype("int8")
    feature_rows = []
    for col, ttype in [(raw_col, "raw"), (clip_col, "clip"), (rank_col, "rank"), (z_col, "z"), (miss_col, "missing_indicator")]:
        feature_rows.append(
            {
                "feature_name": col,
                "source_factor": factor,
                "transform_type": ttype,
                "review_status": review_status,
                "factor_group": group,
                "direction": spec.get("direction", ""),
                "notes": spec.get("notes", ""),
            }
        )
    audit_rows = []
    if bool(invalid_mask.any()):
        panel[invalid_col] = invalid_mask.astype("int8")
        audit_rows.append({"column_name": invalid_col, "audit_type": "invalid_indicator", "source_factor": factor, "notes": source_note})
    diag = {
        "factor": factor,
        "source_note": source_note,
        "raw_non_null": int(raw.notna().sum()),
        "raw_coverage": float(raw.notna().mean()),
        "missing_count": int(missing.sum()),
        "clip_warning_count": int(clip_warnings),
        "zscore_warning_count": int(z_warnings),
        "rank_min": float(rank.min()) if rank.notna().any() else None,
        "rank_max": float(rank.max()) if rank.notna().any() else None,
        "z_mean_abs_max_by_month": None,
    }
    del raw, clip, rank, z, missing
    gc.collect()
    return feature_rows, audit_rows, diag


def leakage_flags(features: pd.DataFrame) -> dict:
    bad_status = features["review_status"].isin(["EXCLUDE", "RAW_COMPONENT_ONLY"]) if len(features) else pd.Series(dtype=bool)
    names = features["feature_name"].astype(str).str.lower() if len(features) else pd.Series(dtype=str)
    audit_terms = ["selected_pit_date", "market_cap_trade_date", "report_period", "validity_flags"]
    return {
        "exclude_feature_leakage_detected": bool((features["review_status"] == "EXCLUDE").any()) if len(features) else False,
        "raw_component_feature_leakage_detected": bool((features["review_status"] == "RAW_COMPONENT_ONLY").any()) if len(features) else False,
        "audit_metadata_feature_leakage_detected": bool(names.apply(lambda x: any(t in x for t in audit_terms)).any()) if len(features) else False,
    }


def build_reports(panel: pd.DataFrame, features: pd.DataFrame, audit_cols: pd.DataFrame, diagnostics: pd.DataFrame, base: dict) -> dict:
    feature_names = features["feature_name"].tolist()
    rank_cols = [c for c in feature_names if c.endswith("_rank")]
    z_cols = [c for c in feature_names if c.endswith("_z")]
    rank_viol = 0
    for c in rank_cols:
        s = panel[c].dropna()
        rank_viol += int(((s < 0) | (s > 1)).sum())
    inf_count = int(np.isinf(panel[feature_names].select_dtypes(include=[np.number])).sum().sum()) if feature_names else 0
    all_null = int(sum(panel[c].isna().all() for c in feature_names))
    constant = int(sum(panel[c].nunique(dropna=True) <= 1 for c in feature_names))
    z_warn = int(diagnostics["zscore_warning_count"].sum()) if len(diagnostics) and "zscore_warning_count" in diagnostics else 0
    leaks = leakage_flags(features)
    status_counts = features.drop_duplicates("source_factor")["review_status"].value_counts().to_dict() if len(features) else {}
    feature_status_counts = features["review_status"].value_counts().to_dict() if len(features) else {}

    fatal = (
        base["duplicate_symbol_month_count"] > 0
        or base["selected_pit_date_violation_count"] > 0
        or rank_viol > 0
        or inf_count > 0
        or any(leaks.values())
    )
    if fatal:
        decision = "TRANSFORMED_PANEL_BUILD_FAIL_BLOCK_REVALIDATION"
    elif int(status_counts.get("WATCH", 0)) > 0 or z_warn > 0 or all_null > 0 or constant > 0:
        decision = "TRANSFORMED_PANEL_BUILD_WATCH_REVIEW_REQUIRED"
    else:
        decision = "TRANSFORMED_PANEL_BUILD_READY_FOR_QA_REVIEW"

    summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_panel_used": rel(SOURCE_PANEL),
        "transform_spec_used": rel(SPEC_PATH),
        "transformed_panel_path": rel(OUT_DIR / "transformed_training_panel_v0.parquet"),
        **{k: base[k] for k in ["rows", "symbols", "months", "one_row_per_symbol_month", "duplicate_symbol_month_count", "selected_pit_date_violation_count", "market_cap_trade_date_violation_count"]},
        "transform_version": TRANSFORM_VERSION,
        "source_panel_version": SOURCE_PANEL_VERSION,
        "total_model_features": int(len(features)),
        "ready_feature_count": int(feature_status_counts.get("READY", 0)),
        "ready_with_note_feature_count": int(feature_status_counts.get("READY_WITH_NOTE", 0)),
        "watch_feature_count": int(feature_status_counts.get("WATCH", 0)),
        **leaks,
        "rank_range_violation_count": int(rank_viol),
        "zscore_warning_count": int(z_warn),
        "infinite_value_count": int(inf_count),
        "all_null_feature_count": int(all_null),
        "constant_feature_count": int(constant),
        "production_modified": False,
        "v3_modified": False,
        "training_run": False,
        "backtest_run": False,
        "ic_calculated": False,
        "neutralization_executed": False,
        "transformed_panel_built": True,
        "final_decision": decision,
        "recommended_next_step": "Transformed Panel QA Review / Compact-F Revalidation Prep" if decision != "TRANSFORMED_PANEL_BUILD_FAIL_BLOCK_REVALIDATION" else "Fix transformed panel build issues before revalidation",
    }

    coverage_rows = []
    for c in feature_names:
        s = panel[c]
        coverage_rows.append({"feature_name": c, "non_null_count": int(s.notna().sum()), "coverage": float(s.notna().mean()), "unique_values": int(s.nunique(dropna=True)), "all_null": bool(s.isna().all()), "constant": bool(s.nunique(dropna=True) <= 1)})
    coverage = pd.DataFrame(coverage_rows)
    qa = pd.DataFrame(
        [
            ["rows", summary["rows"]],
            ["symbols", summary["symbols"]],
            ["months", summary["months"]],
            ["one_row_per_symbol_month", summary["one_row_per_symbol_month"]],
            ["duplicate_symbol_month_count", summary["duplicate_symbol_month_count"]],
            ["selected_pit_date_violation_count", summary["selected_pit_date_violation_count"]],
            ["market_cap_trade_date_violation_count", summary["market_cap_trade_date_violation_count"]],
            ["rank_range_violation_count", summary["rank_range_violation_count"]],
            ["zscore_warning_count", summary["zscore_warning_count"]],
            ["infinite_value_count", summary["infinite_value_count"]],
            ["all_null_feature_count", summary["all_null_feature_count"]],
            ["constant_feature_count", summary["constant_feature_count"]],
            ["watch_feature_count", summary["watch_feature_count"]],
            ["exclude_feature_leakage_detected", summary["exclude_feature_leakage_detected"]],
            ["raw_component_feature_leakage_detected", summary["raw_component_feature_leakage_detected"]],
            ["audit_metadata_feature_leakage_detected", summary["audit_metadata_feature_leakage_detected"]],
            ["final_decision", summary["final_decision"]],
        ],
        columns=["check", "value"],
    )

    features.to_csv(OUT_DIR / "model_feature_list_v0.csv", index=False, encoding="utf-8-sig")
    write_json(OUT_DIR / "model_feature_list_v0.json", features.to_dict("records"))
    audit_cols.to_csv(OUT_DIR / "audit_column_list_v0.csv", index=False, encoding="utf-8-sig")
    write_json(OUT_DIR / "audit_column_list_v0.json", audit_cols.to_dict("records"))
    coverage.to_csv(OUT_DIR / "transformed_feature_coverage.csv", index=False, encoding="utf-8-sig")
    qa.to_csv(OUT_DIR / "transformed_training_panel_qa.csv", index=False, encoding="utf-8-sig")
    qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    write_json(OUT_DIR / "transformed_training_panel_summary.json", summary)
    write_execution_log(diagnostics, summary)
    write_report(summary, diagnostics, audit_cols)
    write_completion_files(summary)
    return summary


def write_execution_log(diagnostics: pd.DataFrame, summary: dict) -> None:
    lines = ["# Transform Execution Log", ""]
    lines.append(f"- run_timestamp: {summary['run_timestamp']}")
    lines.append(f"- transform_version: {TRANSFORM_VERSION}")
    lines.append("- scope: build transformed candidate panel only; no training/backtest/IC/neutralization")
    lines.append("")
    lines.append("## Factor Diagnostics")
    for row in diagnostics.to_dict("records"):
        lines.append(f"- {row.get('factor')}: source={row.get('source_note')}, raw_coverage={row.get('raw_coverage')}, zscore_warnings={row.get('zscore_warning_count')}")
    write_text(OUT_DIR / "transform_execution_log.md", "\n".join(lines) + "\n")


def write_report(summary: dict, diagnostics: pd.DataFrame, audit_cols: pd.DataFrame) -> None:
    text = f"""# Build Transformed Training Panel v0

## 1. Scope

This task builds a transformed training candidate panel only. It does not train, backtest, calculate IC, modify production, modify v3, or execute neutralization.

## 2. Inputs

- {summary['source_panel_used']}
- {summary['transform_spec_used']}

## 3. Transform Execution

Generated raw, clipped, monthly percentile rank, monthly zscore, and missing indicator features for READY / READY_WITH_NOTE / WATCH enabled factors. All cross-sectional operations were performed by month_end within the v3 locked universe.

## 4. ROE Handling

roe_ttm was preserved. roe_parent_ttm_ending_equity was created as an alias when roe_ttm was available. roe_total_ttm_ending_equity was treated only as an alternative candidate when components were available.

## 5. Admin Expense Handling

admin_expense_ratio used robust month-level clipping per the planning spec and is not passed as an unclipped raw-only value.

## 6. Model Feature List

- total_model_features: {summary['total_model_features']}
- READY features: {summary['ready_feature_count']}
- READY_WITH_NOTE features: {summary['ready_with_note_feature_count']}
- WATCH features: {summary['watch_feature_count']}

## 7. Audit Columns

Audit columns include metadata, PIT dates, validity flags when available, raw accounting components, source columns retained for audit, and transform diagnostics. Audit column count: {len(audit_cols)}.

## 8. QA Results

- rows: {summary['rows']}
- symbols: {summary['symbols']}
- months: {summary['months']}
- one_row_per_symbol_month: {summary['one_row_per_symbol_month']}
- selected_pit_date_violation_count: {summary['selected_pit_date_violation_count']}
- market_cap_trade_date_violation_count: {summary['market_cap_trade_date_violation_count']}
- rank_range_violation_count: {summary['rank_range_violation_count']}
- infinite_value_count: {summary['infinite_value_count']}
- all_null_feature_count: {summary['all_null_feature_count']}
- constant_feature_count: {summary['constant_feature_count']}

## 9. Decision

{summary['final_decision']}

## 10. Recommended Next Step

{summary['recommended_next_step']}
"""
    write_text(OUT_DIR / "transformed_training_panel_report.md", text)


def write_completion_files(summary: dict) -> None:
    card = f"""# Task Completion Card

- task: Build Transformed Training Panel v0
- final_decision: {summary['final_decision']}
- transformed_panel_path: {summary['transformed_panel_path']}
- production_modified: false
- v3_modified: false
- training_run: false
- backtest_run: false
- ic_calculated: false
- neutralization_executed: false
"""
    write_text(OUT_DIR / "task_completion_card.md", card)
    write_json(
        OUT_DIR / "terminal_summary.json",
        {
            "task": "Build Transformed Training Panel v0",
            "completed": True,
            "outputs_dir": rel(OUT_DIR),
            "stdout_log": rel(RUN_DIR / "run_stdout.txt"),
            "stderr_log": rel(RUN_DIR / "run_stderr.txt"),
            "final_decision": summary["final_decision"],
        },
    )


def fail_summary(base: dict, reason: str) -> dict:
    summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_panel_used": rel(SOURCE_PANEL),
        "transform_spec_used": rel(SPEC_PATH),
        "transformed_panel_path": None,
        **{k: base.get(k) for k in ["rows", "symbols", "months", "one_row_per_symbol_month", "duplicate_symbol_month_count", "selected_pit_date_violation_count", "market_cap_trade_date_violation_count"]},
        "transform_version": TRANSFORM_VERSION,
        "source_panel_version": SOURCE_PANEL_VERSION,
        "total_model_features": 0,
        "ready_feature_count": 0,
        "ready_with_note_feature_count": 0,
        "watch_feature_count": 0,
        "exclude_feature_leakage_detected": False,
        "raw_component_feature_leakage_detected": False,
        "audit_metadata_feature_leakage_detected": False,
        "rank_range_violation_count": 0,
        "zscore_warning_count": 0,
        "infinite_value_count": 0,
        "all_null_feature_count": 0,
        "constant_feature_count": 0,
        "production_modified": False,
        "v3_modified": False,
        "training_run": False,
        "backtest_run": False,
        "ic_calculated": False,
        "neutralization_executed": False,
        "transformed_panel_built": False,
        "final_decision": "TRANSFORMED_PANEL_BUILD_FAIL_BLOCK_REVALIDATION",
        "recommended_next_step": reason,
    }
    write_json(OUT_DIR / "transformed_training_panel_summary.json", summary)
    pd.DataFrame(summary.items(), columns=["check", "value"]).to_csv(OUT_DIR / "transformed_training_panel_qa.csv", index=False, encoding="utf-8-sig")
    write_text(OUT_DIR / "transformed_training_panel_report.md", f"# Build Transformed Training Panel v0\n\nFAIL: {reason}\n")
    return summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log("Checking explicit input allowlist.")
    missing = [rel(p) for p in INPUTS.values() if not p.exists()]
    if missing:
        write_text(OUT_DIR / "missing_input_report.md", "# Missing Input Report\n\n" + "\n".join(f"- {x}" for x in missing) + "\n")
        update_run_state("missing_inputs", ["Checked explicit input allowlist", "Generated missing_input_report.md"], [rel(OUT_DIR / "missing_input_report.md")], "Restore missing inputs and rerun")
        return 2

    specs = load_spec()
    bad_status = [s for s in specs if s.get("review_status") not in VALID_STATUSES]
    if bad_status:
        write_json(OUT_DIR / "invalid_spec_status_report.json", bad_status)
        update_run_state("invalid_spec", ["Loaded transform spec", "Detected invalid review_status"], [rel(OUT_DIR / "invalid_spec_status_report.json")], "Fix transform spec status values")
        return 2
    update_run_state("spec_validated", ["Validated transform spec statuses"], [rel(RUN_DIR / "RUN_STATE.md")], "Read source panel necessary columns")

    available = set(parquet_columns(SOURCE_PANEL))
    if "symbol" not in available or "month_end" not in available:
        summary = fail_summary({}, "source panel missing required symbol/month_end")
        update_run_state("failed_required_columns", ["Checked parquet schema", "Missing required key columns"], [rel(OUT_DIR / "transformed_training_panel_summary.json")], "Fix source panel")
        return 2

    requested = set(METADATA_COLS) | direct_source_columns(specs) | set(ROE_COMPONENT_COLS)
    requested = {c for c in requested if c in available}
    requested.update(["symbol", "month_end"])
    log(f"Reading source panel columns={len(requested)}.")
    df = read_source_panel(sorted(requested))
    base = base_qa(df)
    if base["duplicate_symbol_month_count"] or base["selected_pit_date_violation_count"] or base["market_cap_trade_date_violation_count"]:
        summary = fail_summary(base, "duplicate key or PIT/date violation detected before transform")
        update_run_state("failed_base_qa", ["Read source panel necessary columns", "Base QA failed"], [rel(OUT_DIR / "transformed_training_panel_summary.json")], "Inspect base QA before rebuilding")
        return 1
    update_run_state("base_qa_passed", ["Read source panel necessary columns", "Base integrity QA passed"], [rel(RUN_DIR / "RUN_STATE.md")], "Build transformed features")

    panel = df[[c for c in METADATA_COLS if c in df.columns]].copy()
    panel["transform_version"] = TRANSFORM_VERSION
    panel["source_panel_version"] = SOURCE_PANEL_VERSION
    if "roe_ttm" in df.columns:
        panel["roe_parent_ttm_ending_equity"] = pd.to_numeric(df["roe_ttm"], errors="coerce")

    eligible = [
        s
        for s in specs
        if s.get("review_status") in {"READY", "READY_WITH_NOTE", "WATCH"}
        and bool(s.get("enabled_for_candidate_training"))
        and s.get("factor_group") not in {"identifier_date", "audit_helper", "raw_accounting_component"}
    ]

    feature_rows: list[dict] = []
    audit_rows: list[dict] = []
    diag_rows: list[dict] = []
    audit_set = set(panel.columns)
    raw_component_cols = [s.get("source_column") for s in specs if s.get("review_status") == "RAW_COMPONENT_ONLY" and str(s.get("source_column", "")) in df.columns]
    source_audit_cols = sorted(set(raw_component_cols) | (set(METADATA_COLS) & set(df.columns)))
    for c in source_audit_cols:
        if c not in panel.columns and c in df.columns:
            panel[c] = df[c]
        audit_set.add(c)

    for spec in eligible:
        rows, audits, diag = transform_factor(spec, df, panel)
        feature_rows.extend(rows)
        audit_rows.extend(audits)
        diag_rows.append(diag)

    features = pd.DataFrame(feature_rows)
    diagnostics = pd.DataFrame(diag_rows)
    for c in sorted(audit_set):
        audit_rows.append({"column_name": c, "audit_type": "metadata_or_source_audit", "source_factor": "", "notes": "not in model_feature_list"})
    audit_cols = pd.DataFrame(audit_rows).drop_duplicates("column_name") if audit_rows else pd.DataFrame(columns=["column_name", "audit_type", "source_factor", "notes"])

    panel.to_parquet(OUT_DIR / "transformed_training_panel_v0.parquet", index=False)
    summary = build_reports(panel, features, audit_cols, diagnostics, base)

    del df, panel, features, audit_cols, diagnostics
    gc.collect()
    generated = [
        rel(OUT_DIR / "transformed_training_panel_v0.parquet"),
        rel(OUT_DIR / "transformed_training_panel_summary.json"),
        rel(OUT_DIR / "transformed_training_panel_qa.csv"),
        rel(OUT_DIR / "transformed_feature_coverage.csv"),
        rel(OUT_DIR / "model_feature_list_v0.json"),
        rel(OUT_DIR / "model_feature_list_v0.csv"),
        rel(OUT_DIR / "audit_column_list_v0.json"),
        rel(OUT_DIR / "audit_column_list_v0.csv"),
        rel(OUT_DIR / "transform_execution_log.md"),
        rel(OUT_DIR / "transformed_training_panel_report.md"),
        rel(OUT_DIR / "task_completion_card.md"),
        rel(OUT_DIR / "terminal_summary.json"),
        rel(OUT_DIR / "final_qa.csv"),
    ]
    update_run_state("completed", ["Built transformed panel", "Generated QA reports", "Generated feature and audit lists"], generated, "Task complete")
    log(json.dumps({"final_decision": summary["final_decision"], "rows": summary["rows"], "total_model_features": summary["total_model_features"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        write_text(OUT_DIR / "error_report.md", traceback.format_exc())
        update_run_state("failed_exception", ["Encountered exception", f"Exception type: {type(exc).__name__}"], [rel(OUT_DIR / "error_report.md")], "Inspect error_report.md and rerun after fixing the issue")
        print(traceback.format_exc(), file=sys.stderr)
        raise
