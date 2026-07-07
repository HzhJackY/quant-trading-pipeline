from __future__ import annotations

import gc
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "output" / "transformed_training_panel_v0"
PLAN_DIR = ROOT / "output" / "factor_transform_planning_v0"
OUT_DIR = ROOT / "output" / "transformed_panel_qa_review_v0"

PANEL_PATH = INPUT_DIR / "transformed_training_panel_v0.parquet"
SUMMARY_PATH = INPUT_DIR / "transformed_training_panel_summary.json"
QA_PATH = INPUT_DIR / "transformed_training_panel_qa.csv"
COVERAGE_PATH = INPUT_DIR / "transformed_feature_coverage.csv"
MODEL_CSV_PATH = INPUT_DIR / "model_feature_list_v0.csv"
MODEL_JSON_PATH = INPUT_DIR / "model_feature_list_v0.json"
AUDIT_JSON_PATH = INPUT_DIR / "audit_column_list_v0.json"
AUDIT_CSV_PATH = INPUT_DIR / "audit_column_list_v0.csv"
REPORT_PATH = INPUT_DIR / "transformed_training_panel_report.md"
SPEC_PATH = PLAN_DIR / "factor_transform_spec_v0.json"
INVENTORY_PATH = PLAN_DIR / "factor_inventory.csv"
PLAN_SUMMARY_PATH = PLAN_DIR / "factor_transform_planning_summary.json"

REQUIRED_INPUTS = [
    PANEL_PATH,
    SUMMARY_PATH,
    QA_PATH,
    COVERAGE_PATH,
    MODEL_CSV_PATH,
    MODEL_JSON_PATH,
    AUDIT_JSON_PATH,
    AUDIT_CSV_PATH,
    REPORT_PATH,
    SPEC_PATH,
    INVENTORY_PATH,
    PLAN_SUMMARY_PATH,
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=clean_value), encoding="utf-8")


def missing_input_report(missing: list[Path]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Missing Input Report",
        "",
        "The QA review did not run because required whitelisted inputs are missing.",
        "",
        "## Missing files",
        "",
    ]
    lines.extend(f"- {p.as_posix()}" for p in missing)
    (OUT_DIR / "missing_input_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "run_timestamp": now_iso(),
        "final_decision": "TRANSFORMED_PANEL_QA_FAIL_BLOCK_REVALIDATION",
        "missing_inputs": [str(p) for p in missing],
    }
    write_json(OUT_DIR / "transformed_panel_qa_review_summary.json", summary)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing(names: list[str], columns: set[str]) -> str | None:
    for name in names:
        if name in columns:
            return name
    return None


def read_audit_columns() -> set[str]:
    cols: set[str] = set()
    try:
        payload = load_json(AUDIT_JSON_PATH)
        if isinstance(payload, list):
            cols.update(str(x) for x in payload)
        elif isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list):
                    cols.update(str(x) for x in value)
                elif isinstance(value, str):
                    cols.add(value)
    except Exception:
        pass
    try:
        audit_df = pd.read_csv(AUDIT_CSV_PATH)
        for column in audit_df.columns:
            if "column" in column.lower() or "feature" in column.lower():
                cols.update(audit_df[column].dropna().astype(str).tolist())
    except Exception:
        pass
    return cols


def spec_reason_map() -> dict[str, str]:
    try:
        spec = load_json(SPEC_PATH)
    except Exception:
        return {}
    out: dict[str, str] = {}

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            feature = obj.get("feature_name") or obj.get("output_feature") or obj.get("name")
            reason = obj.get("reason") or obj.get("notes") or obj.get("review_note") or obj.get("watch_reason")
            if feature and reason:
                out[str(feature)] = str(reason)
            for value in obj.values():
                visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(spec)
    return out


def feature_stats(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    non_null_count = int(series.notna().sum())
    missing_count = int(series.isna().sum())
    infinite_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
    finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    quantiles = finite.quantile([0.01, 0.05, 0.5, 0.95, 0.99]) if len(finite) else pd.Series(dtype=float)
    return {
        "non_null_count": non_null_count,
        "missing_count": missing_count,
        "coverage": non_null_count / len(series) if len(series) else 0.0,
        "unique_count": int(series.nunique(dropna=True)),
        "min": clean_value(finite.min()) if len(finite) else None,
        "p1": clean_value(quantiles.get(0.01)) if len(finite) else None,
        "p5": clean_value(quantiles.get(0.05)) if len(finite) else None,
        "p50": clean_value(quantiles.get(0.5)) if len(finite) else None,
        "p95": clean_value(quantiles.get(0.95)) if len(finite) else None,
        "p99": clean_value(quantiles.get(0.99)) if len(finite) else None,
        "max": clean_value(finite.max()) if len(finite) else None,
        "infinite_count": infinite_count,
        "constant_flag": int(series.nunique(dropna=True)) <= 1 and non_null_count > 0,
        "all_null_flag": non_null_count == 0,
    }


def watch_action(stats: dict[str, Any]) -> str:
    if stats["all_null_flag"] or stats["infinite_count"] > 0:
        return "EXCLUDE_FROM_REVALIDATION"
    if stats["constant_flag"]:
        return "KEEP_BUT_EXCLUDE_BY_DEFAULT"
    if stats["coverage"] < 0.5:
        return "NEED_MANUAL_REVIEW"
    if stats["coverage"] < 0.7:
        return "KEEP_BUT_EXCLUDE_BY_DEFAULT"
    return "KEEP_FOR_REVALIDATION_WITH_NOTE"


def constant_action(feature_name: str) -> str:
    low = feature_name.lower()
    if low.endswith("_z") or low.endswith("_rank") or "_z_" in low:
        return "DROP_FROM_MODEL_FEATURE_LIST"
    if "missing" in low or "is_null" in low or "null_flag" in low:
        return "KEEP_AS_AUDIT_ONLY"
    if "valid" in low or "flag" in low:
        return "KEEP_AS_AUDIT_ONLY"
    return "DROP_FROM_MODEL_FEATURE_LIST"


def coverage_action(row: pd.Series) -> str:
    coverage = float(row.get("coverage", 0) or 0)
    if bool(row.get("all_null", False)) or bool(row.get("constant", False)):
        return "EXCLUDE_BY_DEFAULT"
    if coverage < 0.3:
        return "EXCLUDE_BY_DEFAULT"
    if coverage < 0.5:
        return "NEED_MANUAL_REVIEW"
    if coverage < 0.7:
        return "KEEP_WITH_MISSING_INDICATOR"
    return "KEEP"


def leakage_review(model_df: pd.DataFrame, audit_columns: set[str]) -> pd.DataFrame:
    features = model_df["feature_name"].astype(str).tolist()
    lower = {f: f.lower() for f in features}

    checks: list[tuple[str, list[str]]] = [
        ("selected_pit_date", [f for f, low in lower.items() if "selected_pit_date" in low]),
        ("selected_report_period", [f for f, low in lower.items() if "selected_report_period" in low]),
        ("market_cap_trade_date", [f for f, low in lower.items() if "market_cap_trade_date" in low]),
        ("audit_helper_columns", [f for f in features if f in audit_columns]),
        ("EXCLUDE_status_features", model_df.loc[model_df["review_status"].astype(str).str.upper().eq("EXCLUDE"), "feature_name"].astype(str).tolist()),
        ("RAW_COMPONENT_ONLY_features", model_df.loc[model_df["review_status"].astype(str).str.upper().isin(["RAW_COMPONENT_ONLY", "RAW_COMPONENT"]), "feature_name"].astype(str).tolist()),
        ("production_live_holding_prediction_columns", [f for f, low in lower.items() if any(x in low for x in ["production", "live", "holding", "prediction", "pred_"])]),
    ]
    rows = []
    for leakage_type, offenders in checks:
        offenders = sorted(set(offenders))
        severity = "FAIL" if offenders else "NONE"
        rows.append(
            {
                "leakage_type": leakage_type,
                "detected": bool(offenders),
                "offending_features": ";".join(offenders),
                "severity": severity,
                "action": "BLOCK_REVALIDATION_UNTIL_REMOVED" if offenders else "NO_ACTION",
            }
        )
    return pd.DataFrame(rows)


def rank_zscore_review(panel: pd.DataFrame, feature_names: list[str], month_col: str | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in feature_names:
        if feature not in panel.columns or not (feature.endswith("_rank") or feature.endswith("_z")):
            continue
        series = pd.to_numeric(panel[feature], errors="coerce")
        non_null = int(series.notna().sum())
        row: dict[str, Any] = {
            "feature_name": feature,
            "feature_type": "rank" if feature.endswith("_rank") else "zscore",
            "non_null_count": non_null,
            "min": clean_value(series.min(skipna=True)),
            "max": clean_value(series.max(skipna=True)),
            "warning": "",
            "pass": True,
        }
        warnings: list[str] = []
        if feature.endswith("_rank"):
            outside = int(((series < 0) | (series > 1)).fillna(False).sum())
            row["rank_outside_0_1_count"] = outside
            row["by_month_count_violation_count"] = 0
            if month_col:
                by_month_non_null = series.groupby(panel[month_col]).count()
                month_rows = panel.groupby(month_col).size()
                row["by_month_count_violation_count"] = int((by_month_non_null > month_rows).sum())
            if outside:
                warnings.append("rank_outside_0_1")
        else:
            row["rank_outside_0_1_count"] = None
            row["by_month_count_violation_count"] = None
            if month_col:
                grouped = series.groupby(panel[month_col])
                means = grouped.mean()
                stds = grouped.std(ddof=0)
                row["max_abs_month_mean"] = clean_value(means.abs().max())
                row["max_abs_month_std_minus_1"] = clean_value((stds - 1).abs().max())
                zero_std_count = int((stds.fillna(0) == 0).sum())
                row["zero_or_null_month_std_count"] = zero_std_count
                if zero_std_count:
                    warnings.append("zero_or_null_month_std")
                if len(means) and means.abs().max() > 0.25:
                    warnings.append("month_mean_not_near_zero")
                if len(stds.dropna()) and (stds - 1).abs().max() > 0.5:
                    warnings.append("month_std_not_near_one")
            else:
                warnings.append("month_column_not_found")
        row["warning"] = ";".join(warnings)
        row["pass"] = not any(w in warnings for w in ["rank_outside_0_1", "month_column_not_found"])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        missing_input_report(missing)
        print(f"Missing required inputs: {len(missing)}")
        return 0

    summary = load_json(SUMMARY_PATH)
    report_text = REPORT_PATH.read_text(encoding="utf-8", errors="replace")
    model_df = pd.read_csv(MODEL_CSV_PATH)
    coverage_df = pd.read_csv(COVERAGE_PATH)
    audit_columns = read_audit_columns()
    reasons = spec_reason_map()

    required_model_cols = {"feature_name", "source_factor", "transform_type", "review_status", "factor_group", "direction"}
    missing_model_cols = sorted(required_model_cols - set(model_df.columns))
    if missing_model_cols:
        raise ValueError(f"model_feature_list_v0.csv missing columns: {missing_model_cols}")

    previous_check = {
        "final_decision": summary.get("final_decision"),
        "final_decision_ok": summary.get("final_decision") == "TRANSFORMED_PANEL_BUILD_WATCH_REVIEW_REQUIRED",
        "panel_exists": PANEL_PATH.exists(),
        "production_modified": bool(summary.get("production_modified")),
        "v3_modified": bool(summary.get("v3_modified")),
        "training_run": bool(summary.get("training_run")),
        "backtest_run": bool(summary.get("backtest_run")),
        "neutralization_executed": bool(summary.get("neutralization_executed")),
        "report_read": bool(report_text),
    }
    write_json(OUT_DIR / "previous_build_check.json", previous_check)

    feature_names = model_df["feature_name"].astype(str).tolist()
    parquet = pq.ParquetFile(PANEL_PATH)
    panel_columns = set(parquet.schema.names)
    symbol_col = first_existing(["symbol", "Symbol", "stock_code", "stkcd", "ts_code"], panel_columns)
    month_col = first_existing(["month", "month_end", "trade_month", "month_date", "yyyymm"], panel_columns)
    read_columns = [c for c in [symbol_col, month_col] if c] + [f for f in feature_names if f in panel_columns]
    panel = pd.read_parquet(PANEL_PATH, columns=read_columns)

    rows = int(len(panel))
    symbols = int(panel[symbol_col].nunique()) if symbol_col else int(summary.get("symbols", 0) or 0)
    months = int(panel[month_col].nunique()) if month_col else int(summary.get("months", 0) or 0)
    one_row = bool(summary.get("one_row_per_symbol_month"))
    if symbol_col and month_col:
        one_row = not panel.duplicated([symbol_col, month_col]).any()

    stats_by_feature = {f: feature_stats(panel[f]) for f in feature_names if f in panel.columns}

    watch_df = model_df[model_df["review_status"].astype(str).str.upper().eq("WATCH")].copy()
    watch_rows = []
    for _, row in watch_df.iterrows():
        feature = str(row["feature_name"])
        stats = stats_by_feature.get(feature, {})
        out = {
            "feature_name": feature,
            "source_factor": row.get("source_factor"),
            "transform_type": row.get("transform_type"),
            "factor_group": row.get("factor_group"),
            "direction": row.get("direction"),
            **stats,
            "reason_from_spec": reasons.get(feature, row.get("notes", "")),
        }
        out["recommended_action"] = watch_action(stats)
        watch_rows.append(out)
    watch_review = pd.DataFrame(watch_rows)
    watch_review.to_csv(OUT_DIR / "watch_feature_review.csv", index=False)

    constant_features = [f for f, s in stats_by_feature.items() if s["constant_flag"]]
    constant_rows = []
    for feature in constant_features:
        meta = model_df.loc[model_df["feature_name"].astype(str).eq(feature)].iloc[0]
        non_null_values = panel[feature].dropna().unique()
        low = feature.lower()
        action = constant_action(feature)
        constant_rows.append(
            {
                "feature_name": feature,
                "source_factor": meta.get("source_factor"),
                "transform_type": meta.get("transform_type"),
                "unique_count": stats_by_feature[feature]["unique_count"],
                "constant_value": clean_value(non_null_values[0]) if len(non_null_values) else None,
                "non_null_count": stats_by_feature[feature]["non_null_count"],
                "coverage": stats_by_feature[feature]["coverage"],
                "is_missing_indicator": any(x in low for x in ["missing", "is_null", "null_flag"]),
                "is_validity_flag": any(x in low for x in ["valid", "flag"]),
                "is_rank_or_zscore": low.endswith("_rank") or low.endswith("_z"),
                "likely_reason": "single_unique_non_null_value",
                "recommended_action": action,
            }
        )
    constant_review = pd.DataFrame(constant_rows)
    constant_review.to_csv(OUT_DIR / "constant_feature_review.csv", index=False)

    coverage_review = coverage_df[coverage_df["feature_name"].astype(str).isin(feature_names)].copy()
    coverage_review["LOW_COVERAGE"] = coverage_review["coverage"].astype(float) < 0.70
    coverage_review["VERY_LOW_COVERAGE"] = coverage_review["coverage"].astype(float) < 0.50
    coverage_review["SPARSE"] = coverage_review["coverage"].astype(float) < 0.30
    coverage_review["sparse_flag"] = coverage_review["SPARSE"]
    coverage_review["recommended_action"] = coverage_review.apply(coverage_action, axis=1)
    coverage_review.to_csv(OUT_DIR / "feature_coverage_review.csv", index=False)

    rz_review = rank_zscore_review(panel, feature_names, month_col)
    rz_review.to_csv(OUT_DIR / "rank_zscore_sanity_review.csv", index=False)

    leak_df = leakage_review(model_df, audit_columns)
    leak_df.to_csv(OUT_DIR / "feature_leakage_review.csv", index=False)

    constant_set = set(constant_review["feature_name"].astype(str)) if not constant_review.empty else set()
    all_null_set = {f for f, s in stats_by_feature.items() if s["all_null_flag"]}
    watch_action_map = dict(zip(watch_review["feature_name"], watch_review["recommended_action"])) if not watch_review.empty else {}
    leakage_features = set()
    for offenders in leak_df["offending_features"].astype(str):
        if offenders:
            leakage_features.update(x for x in offenders.split(";") if x)

    revalidation_rows = []
    for _, row in model_df.iterrows():
        feature = str(row["feature_name"])
        status = str(row["review_status"]).upper()
        qa_rec = ""
        included = False
        reason = ""
        if feature in leakage_features:
            qa_rec, reason = "EXCLUDE_BY_DEFAULT", "leakage_or_forbidden_feature"
        elif feature in constant_set:
            qa_rec, reason = "EXCLUDE_BY_DEFAULT", "constant_feature"
        elif feature in all_null_set:
            qa_rec, reason = "EXCLUDE_BY_DEFAULT", "all_null_feature"
        elif status in {"READY", "READY_WITH_NOTE"}:
            qa_rec, included, reason = "KEEP", True, status.lower()
        elif status == "WATCH":
            action = watch_action_map.get(feature, "NEED_MANUAL_REVIEW")
            qa_rec = action
            included = action == "KEEP_FOR_REVALIDATION_WITH_NOTE"
            reason = f"watch_{action.lower()}"
        else:
            qa_rec, reason = "EXCLUDE_BY_DEFAULT", f"status_{status.lower()}"
        revalidation_rows.append(
            {
                "feature_name": feature,
                "source_factor": row.get("source_factor"),
                "transform_type": row.get("transform_type"),
                "factor_group": row.get("factor_group"),
                "review_status": row.get("review_status"),
                "qa_recommendation": qa_rec,
                "included_by_default": bool(included),
                "reason": reason,
            }
        )
    revalidation_df = pd.DataFrame(revalidation_rows)
    revalidation_df.to_csv(OUT_DIR / "model_feature_list_for_revalidation_v0.csv", index=False)
    write_json(OUT_DIR / "model_feature_list_for_revalidation_v0.json", revalidation_df.to_dict(orient="records"))

    leakage_detected = bool(leak_df["detected"].any())
    severe_leakage = bool((leak_df["severity"] == "FAIL").any())
    rank_range_violations = int(summary.get("rank_range_violation_count", 0) or 0)
    infinite_values = int(summary.get("infinite_value_count", 0) or 0)
    pit_violations = int(summary.get("selected_pit_date_violation_count", 0) or 0)
    market_cap_violations = int(summary.get("market_cap_trade_date_violation_count", 0) or 0)
    duplicate_count = int(summary.get("duplicate_symbol_month_count", 0) or 0)
    fatal = any([severe_leakage, rank_range_violations, infinite_values, pit_violations, market_cap_violations, duplicate_count, not one_row])
    manual_needed = bool(
        (watch_review["recommended_action"].eq("NEED_MANUAL_REVIEW").any() if not watch_review.empty else False)
        or (constant_review["recommended_action"].eq("NEED_MANUAL_REVIEW").any() if not constant_review.empty else False)
    )
    if fatal:
        final_decision = "TRANSFORMED_PANEL_QA_FAIL_BLOCK_REVALIDATION"
        recommended_next_step = "Fix transformed panel build before revalidation."
    elif manual_needed:
        final_decision = "TRANSFORMED_PANEL_QA_WATCH_MANUAL_FEATURE_REVIEW_REQUIRED"
        recommended_next_step = "Manual review of flagged WATCH/constant features."
    else:
        final_decision = "TRANSFORMED_PANEL_QA_CLEARED_READY_FOR_COMPACT_F_REVALIDATION_PREP"
        recommended_next_step = "Compact-F Revalidation Prep."

    review_summary = {
        "run_timestamp": now_iso(),
        "transformed_panel_used": str(PANEL_PATH.relative_to(ROOT)),
        "model_feature_list_used": str(MODEL_CSV_PATH.relative_to(ROOT)),
        "rows": rows,
        "symbols": symbols,
        "months": months,
        "one_row_per_symbol_month": one_row,
        "pit_violation_count": pit_violations,
        "market_cap_date_violation_count": market_cap_violations,
        "rank_range_violation_count": rank_range_violations,
        "infinite_value_count": infinite_values,
        "watch_feature_count": int(len(watch_review)),
        "watch_keep_count": int(watch_review["recommended_action"].eq("KEEP_FOR_REVALIDATION_WITH_NOTE").sum()) if not watch_review.empty else 0,
        "watch_exclude_count": int(watch_review["recommended_action"].isin(["KEEP_BUT_EXCLUDE_BY_DEFAULT", "EXCLUDE_FROM_REVALIDATION"]).sum()) if not watch_review.empty else 0,
        "watch_manual_review_count": int(watch_review["recommended_action"].eq("NEED_MANUAL_REVIEW").sum()) if not watch_review.empty else 0,
        "constant_feature_count": int(len(constant_review)),
        "constant_drop_count": int(constant_review["recommended_action"].eq("DROP_FROM_MODEL_FEATURE_LIST").sum()) if not constant_review.empty else 0,
        "constant_audit_only_count": int(constant_review["recommended_action"].isin(["KEEP_AS_AUDIT_ONLY", "KEEP_IF_MISSING_INDICATOR"]).sum()) if not constant_review.empty else 0,
        "low_coverage_feature_count": int(coverage_review["LOW_COVERAGE"].sum()),
        "very_low_coverage_feature_count": int(coverage_review["VERY_LOW_COVERAGE"].sum()),
        "leakage_detected": leakage_detected,
        "severe_leakage_detected": severe_leakage,
        "default_revalidation_feature_count": int(revalidation_df["included_by_default"].sum()),
        "production_modified": bool(summary.get("production_modified")),
        "v3_modified": bool(summary.get("v3_modified")),
        "training_run": bool(summary.get("training_run")),
        "backtest_run": bool(summary.get("backtest_run")),
        "ic_calculated": False,
        "neutralization_executed": bool(summary.get("neutralization_executed")),
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    write_json(OUT_DIR / "transformed_panel_qa_review_summary.json", review_summary)

    report_lines = [
        "# Transformed Panel QA Review v0",
        "",
        "## 1. Scope",
        "",
        "This run only reviews the transformed panel QA surface: WATCH items, constant features, coverage, rank/zscore sanity, leakage, and default feature-list safety. It does not train, backtest, calculate IC, modify production, modify v3, rebuild the panel, or execute neutralization.",
        "",
        "## 2. Previous Build Result",
        "",
        f"- Previous final_decision: {summary.get('final_decision')}",
        f"- Rows / symbols / months: {rows} / {symbols} / {months}",
        f"- One row per symbol-month: {one_row}",
        f"- PIT / market-cap date violations: {pit_violations} / {market_cap_violations}",
        "",
        "## 3. WATCH Feature Review",
        "",
        f"- WATCH features: {len(watch_review)}",
        f"- Keep with note: {review_summary['watch_keep_count']}",
        f"- Exclude by default: {review_summary['watch_exclude_count']}",
        f"- Manual review: {review_summary['watch_manual_review_count']}",
        "",
        "## 4. Constant Feature Review",
        "",
        f"- Constant features: {len(constant_review)}",
        f"- Drop from model feature list: {review_summary['constant_drop_count']}",
        f"- Keep as audit only: {review_summary['constant_audit_only_count']}",
        "",
        "## 5. Coverage Review",
        "",
        f"- Low coverage (<0.70): {review_summary['low_coverage_feature_count']}",
        f"- Very low coverage (<0.50): {review_summary['very_low_coverage_feature_count']}",
        "",
        "## 6. Rank / Zscore Sanity",
        "",
        f"- Rank range violations from build summary: {rank_range_violations}",
        f"- Rank/zscore reviewed features: {len(rz_review)}",
        "",
        "## 7. Leakage Review",
        "",
        f"- Leakage detected: {leakage_detected}",
        f"- Severe leakage detected: {severe_leakage}",
        "",
        "## 8. Default Feature List for Revalidation",
        "",
        f"- Included by default: {review_summary['default_revalidation_feature_count']}",
        f"- Excluded or manual: {len(revalidation_df) - review_summary['default_revalidation_feature_count']}",
        "",
        "## 9. Decision",
        "",
        final_decision,
        "",
        "## 10. Recommended Next Step",
        "",
        recommended_next_step,
        "",
    ]
    (OUT_DIR / "transformed_panel_qa_review_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    task_card = [
        "# Task Completion Card",
        "",
        f"- task_name: Transformed Panel QA Review v0",
        f"- completed_at: {now_iso()}",
        f"- final_decision: {final_decision}",
        f"- output_dir: {OUT_DIR.relative_to(ROOT).as_posix()}",
    ]
    (OUT_DIR / "task_completion_card.md").write_text("\n".join(task_card) + "\n", encoding="utf-8")
    terminal_summary = {
        "script": "scripts/review_transformed_panel_qa_v0.py",
        "status": "completed",
        "stdout_log": "output/_agent_runs/transformed_panel_qa_review_v0/run_stdout.txt",
        "stderr_log": "output/_agent_runs/transformed_panel_qa_review_v0/run_stderr.txt",
        "final_decision": final_decision,
    }
    write_json(OUT_DIR / "terminal_summary.json", terminal_summary)
    pd.DataFrame([review_summary]).to_csv(OUT_DIR / "final_qa.csv", index=False)

    del panel, model_df, coverage_df
    gc.collect()
    print(f"final_decision={final_decision}")
    print(f"output_dir={OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
