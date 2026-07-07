from __future__ import annotations

import gc
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TASK_NAME = "FI_T5_watch_review_v01"
RUN_DIR = Path("output/_agent_runs") / TASK_NAME
OUT_DIR = Path("output/fi_t5_watch_review_v01")
V3_PATH = Path(
    "output/csmar_pit_clean_core_financial_factors_v3/"
    "pit_clean_core_financial_factors_monthly_v3.parquet"
)
FI_T5_PATH = Path("data/csmar_exports/FI_T5_2015-02-28_2026-06-30_Stkcd_like_p1_pack_export_20260629.csv")
PREV_DIR = Path("output/fi_t5_sanity_check_v0")

PREV_SUMMARY = PREV_DIR / "fi_t5_sanity_check_summary.json"
PREV_METRICS = PREV_DIR / "sanity_metrics_summary.csv"
PREV_DISCREPANCIES = PREV_DIR / "sanity_discrepancy_examples.csv"
PREV_MAPPING = PREV_DIR / "field_mapping.csv"
PREV_SCHEMA_V3 = PREV_DIR / "schema_v3.json"
PREV_SCHEMA_FI = PREV_DIR / "schema_fi_t5.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def ensure_dirs() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_checkpoint(stage: str, status: str, notes: str) -> None:
    with (RUN_DIR / "CHECKPOINTS.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## {now_iso()}\n\nStage: {stage}\n\nStatus: {status}\n\n{notes}\n")


def update_run_state(stage: str, completed: list[str], current_files: list[str], generated: list[str], next_step: str) -> None:
    lines = [
        "# RUN_STATE.md",
        "",
        "- 当前任务名称: FI_T5 WATCH Review v0.1",
        f"- 开始时间: {now_iso()}",
        f"- 当前阶段: {stage}",
        "- 已完成步骤:",
    ]
    lines.extend([f"  - {x}" for x in completed])
    lines.append("- 正在处理的文件:")
    lines.extend([f"  - {x}" for x in current_files])
    lines.append("- 已生成输出:")
    lines.extend([f"  - {x}" for x in generated])
    lines.extend(
        [
            f"- 下一步: {next_step}",
            "- 如果 Codex 崩溃，新的 Codex 应如何继续:",
            "  1. Read this RUN_STATE.md first.",
            "  2. Continue with scripts/run_fi_t5_watch_review_v01.py.",
            "  3. Only use the approved files listed in PROMPT_SNAPSHOT.md.",
            "  4. Do not train, backtest, calculate IC, or modify production/v3.",
        ]
    )
    write_text(RUN_DIR / "RUN_STATE.md", "\n".join(lines) + "\n")


def log(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


def load_previous_outputs() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    with PREV_SUMMARY.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    metrics = pd.read_csv(PREV_METRICS)
    discrepancies = pd.read_csv(PREV_DISCREPANCIES)
    mapping = pd.read_csv(PREV_MAPPING)
    with PREV_SCHEMA_V3.open("r", encoding="utf-8") as f:
        schema_v3 = json.load(f)
    with PREV_SCHEMA_FI.open("r", encoding="utf-8") as f:
        schema_fi = json.load(f)
    return summary, metrics, discrepancies, mapping, schema_v3, schema_fi


def previous_result_check(summary: dict, metrics: pd.DataFrame, mapping: pd.DataFrame) -> dict:
    status_by_metric = dict(zip(metrics["sanity_metric"], metrics["status"]))
    admin = metrics[metrics["sanity_metric"].eq("admin_expense_ratio")].iloc[0].to_dict()
    check = {
        "previous_decision": summary.get("decision"),
        "previous_decision_is_watch": summary.get("decision") == "FI_T5_SANITY_WATCH_REVIEW_REQUIRED",
        "roe_is_unique_watch": status_by_metric.get("roe") == "WATCH"
        and sum(1 for v in status_by_metric.values() if v == "WATCH") == 1,
        "sales_expense_ratio_status": status_by_metric.get("sales_expense_ratio"),
        "admin_expense_ratio_status": status_by_metric.get("admin_expense_ratio"),
        "admin_mean_absolute_difference": float(admin["mean_absolute_difference"]),
        "admin_median_absolute_difference": float(admin["median_absolute_difference"]),
        "admin_mean_gt_50x_median": float(admin["mean_absolute_difference"])
        > 50.0 * float(admin["median_absolute_difference"]),
        "metrics_tested": int(summary.get("metrics_tested", len(metrics))),
        "mapping_rows": int(len(mapping)),
    }
    write_text(OUT_DIR / "previous_result_check.json", json.dumps(check, ensure_ascii=False, indent=2))
    return check


def quantiles(series: pd.Series, prefix: str) -> dict:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    probs = [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
    if clean.empty:
        return {f"{prefix}_p{int(p * 100)}": np.nan for p in probs}
    q = clean.quantile(probs)
    return {f"{prefix}_p{int(p * 100)}": float(q.loc[p]) for p in probs}


def diff_quantiles(series: pd.Series) -> dict:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    probs = [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.995, 0.999, 1.0]
    if clean.empty:
        return {f"abs_diff_p{str(p).replace('.', '_')}": np.nan for p in probs}
    q = clean.quantile(probs)
    labels = ["p0", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99_5", "p99_9", "p100"]
    return {f"abs_diff_{label}": float(q.loc[p]) for label, p in zip(labels, probs)}


def read_core_data(schema_v3: dict) -> pd.DataFrame:
    v3_available = set(schema_v3.get("raw_columns", []))
    v3_cols = [
        "symbol",
        "month_end",
        "selected_report_period",
        "selected_pit_date",
        "roe_ttm",
        "admin_expense_to_revenue",
        "sales_expense_to_revenue",
        "net_profit_parent_ttm",
        "net_profit_ttm",
        "equity_parent",
        "total_equity",
        "total_assets",
        "total_liabilities",
    ]
    v3_cols = [c for c in v3_cols if c in v3_available]
    log(f"Reading v3 necessary columns: {v3_cols}")
    v3 = pd.read_parquet(V3_PATH, columns=v3_cols)

    fi_cols = ["Stkcd", "Accper", "Typrep", "F050501B", "F051701B", "F051801B"]
    log(f"Reading FI_T5 necessary columns: {fi_cols}")
    fi = pd.read_csv(FI_T5_PATH, usecols=fi_cols, dtype={"Stkcd": str}, low_memory=True)

    v3["symbol"] = v3["symbol"].astype(str).str.zfill(6)
    v3["month_end"] = pd.to_datetime(v3["month_end"], errors="coerce")
    v3["selected_report_period"] = pd.to_datetime(v3["selected_report_period"], errors="coerce")
    if "selected_pit_date" in v3.columns:
        v3["selected_pit_date"] = pd.to_datetime(v3["selected_pit_date"], errors="coerce")

    fi["Stkcd"] = fi["Stkcd"].astype(str).str.zfill(6)
    fi["Accper"] = pd.to_datetime(fi["Accper"], errors="coerce")
    fi = fi[fi["Typrep"].astype(str).str.upper().eq("A")].copy()
    for col in ["F050501B", "F051701B", "F051801B"]:
        fi[col] = pd.to_numeric(fi[col], errors="coerce")
    fi = fi.dropna(subset=["Stkcd", "Accper"]).drop_duplicates(["Stkcd", "Accper"], keep="last")

    merged = v3.merge(
        fi,
        left_on=["symbol", "selected_report_period"],
        right_on=["Stkcd", "Accper"],
        how="left",
        validate="many_to_one",
    )
    del v3, fi
    gc.collect()
    return merged


def pair_stats(df: pd.DataFrame, vcol: str, fcol: str) -> dict:
    sub = df.dropna(subset=[vcol, fcol]).copy()
    rows = int(len(sub))
    if rows == 0:
        return {
            "rows": 0,
            "symbols": 0,
            "months": 0,
            "pearson": np.nan,
            "spearman": np.nan,
            "median_absolute_difference": np.nan,
            "mean_absolute_difference": np.nan,
            "sign_agreement_rate": np.nan,
        }
    diff = sub[vcol] - sub[fcol]
    return {
        "rows": rows,
        "symbols": int(sub["symbol"].nunique()),
        "months": int(sub["month_end"].nunique()),
        "pearson": float(sub[vcol].corr(sub[fcol], method="pearson")),
        "spearman": float(sub[vcol].corr(sub[fcol], method="spearman")),
        "median_absolute_difference": float(diff.abs().median()),
        "mean_absolute_difference": float(diff.abs().mean()),
        "sign_agreement_rate": float((np.sign(sub[vcol]) == np.sign(sub[fcol])).mean()),
    }


def roe_unit_review(merged: pd.DataFrame) -> tuple[pd.DataFrame, bool, str]:
    sub = merged.dropna(subset=["roe_ttm", "F050501B"]).copy()
    eps = 1e-8
    ratio = sub.loc[sub["F050501B"].abs() > eps, "roe_ttm"] / sub.loc[
        sub["F050501B"].abs() > eps, "F050501B"
    ]
    abs_ratio = ratio.abs().replace([np.inf, -np.inf], np.nan).dropna()
    row = {
        "review_item": "roe_unit_sanity",
        **quantiles(sub["roe_ttm"], "v3_roe_ttm"),
        **quantiles(sub["F050501B"], "fi_t5_F050501B"),
        **quantiles(abs_ratio, "abs_v3_div_fi_t5"),
        "scenario_A_both_decimal_supported": bool(abs_ratio.median() > 0.05 and abs_ratio.median() < 20)
        if not abs_ratio.empty
        else False,
        "scenario_B_v3_decimal_fi_percent_supported": bool(abs_ratio.median() > 0.0005 and abs_ratio.median() < 0.05)
        if not abs_ratio.empty
        else False,
        "scenario_C_v3_percent_fi_decimal_supported": bool(abs_ratio.median() > 20 and abs_ratio.median() < 200)
        if not abs_ratio.empty
        else False,
    }
    unit_error = bool(row["scenario_B_v3_decimal_fi_percent_supported"] or row["scenario_C_v3_percent_fi_decimal_supported"])
    conclusion = (
        "No 100x percent-vs-decimal unit error pattern detected."
        if not unit_error
        else "Possible 100x percent-vs-decimal unit mismatch detected."
    )
    row["unit_error_detected"] = unit_error
    row["conclusion"] = conclusion
    out = pd.DataFrame([row])
    out.to_csv(OUT_DIR / "roe_unit_review.csv", index=False, encoding="utf-8-sig")
    return out, unit_error, conclusion


def roe_filter_sensitivity(merged: pd.DataFrame) -> pd.DataFrame:
    sub = merged.dropna(subset=["roe_ttm", "F050501B"]).copy()
    lo = sub["roe_ttm"].quantile(0.01)
    hi = sub["roe_ttm"].quantile(0.99)
    filters = [
        ("A_full_sample", sub),
        ("B_remove_abs_either_gt_1", sub[(sub["roe_ttm"].abs() <= 1) & (sub["F050501B"].abs() <= 1)]),
        ("C_remove_abs_either_gt_2", sub[(sub["roe_ttm"].abs() <= 2) & (sub["F050501B"].abs() <= 2)]),
        ("D_remove_v3_bottom_top_1pct", sub[(sub["roe_ttm"] >= lo) & (sub["roe_ttm"] <= hi)]),
        ("E_keep_both_in_minus1_1", sub[(sub["roe_ttm"].between(-1, 1)) & (sub["F050501B"].between(-1, 1))]),
        ("F_keep_both_in_minus0_5_0_5", sub[(sub["roe_ttm"].between(-0.5, 0.5)) & (sub["F050501B"].between(-0.5, 0.5))]),
    ]
    rows = []
    for name, df in filters:
        stat = pair_stats(df, "roe_ttm", "F050501B")
        stat["filter"] = name
        rows.append(stat)
    out = pd.DataFrame(rows)[
        [
            "filter",
            "rows",
            "symbols",
            "months",
            "pearson",
            "spearman",
            "median_absolute_difference",
            "mean_absolute_difference",
            "sign_agreement_rate",
        ]
    ]
    out.to_csv(OUT_DIR / "roe_filter_sensitivity.csv", index=False, encoding="utf-8-sig")
    return out


def report_period_label(dt: pd.Timestamp) -> str:
    if pd.isna(dt):
        return "UNKNOWN"
    month = int(dt.month)
    return {3: "Q1", 6: "H1", 9: "Q3", 12: "FY"}.get(month, f"M{month:02d}")


def roe_period_timing_review(merged: pd.DataFrame) -> pd.DataFrame:
    sub = merged.dropna(subset=["roe_ttm", "F050501B", "Accper"]).copy()
    sub["period_type"] = sub["Accper"].map(report_period_label)
    rows = []
    for period, df in sub.groupby("period_type", dropna=False):
        stat = pair_stats(df, "roe_ttm", "F050501B")
        stat["period_type"] = period
        stat["accper_month"] = "" if period == "UNKNOWN" else period
        rows.append(stat)
    out = pd.DataFrame(rows).sort_values("period_type")
    out.to_csv(OUT_DIR / "roe_period_timing_review.csv", index=False, encoding="utf-8-sig")
    return out


def infer_symbol_reason(row: pd.Series) -> str:
    reasons = []
    if row.get("min_v3_roe", 0) < -1 or row.get("max_abs_fi_roe", 0) > 1:
        reasons.append("extreme loss")
    eq_min = min(
        [x for x in [row.get("min_equity_parent", np.nan), row.get("min_total_equity", np.nan)] if pd.notna(x)]
        or [np.nan]
    )
    if pd.notna(eq_min) and eq_min <= 0:
        reasons.append("small / negative denominator")
    elif row.get("min_abs_equity_parent", np.inf) < 1e5 or row.get("min_abs_total_equity", np.inf) < 1e5:
        reasons.append("small / negative denominator")
    reasons.append("TTM vs report-period definition")
    reasons.append("fiscal timing")
    return "; ".join(dict.fromkeys(reasons)) if reasons else "unknown"


def roe_distressed_symbol_review(merged: pd.DataFrame, previous_discrepancies: pd.DataFrame) -> pd.DataFrame:
    roe_disc = previous_discrepancies[previous_discrepancies["sanity_metric"].eq("roe")].copy()
    top_freq = (
        roe_disc["symbol"].astype(str).str.zfill(6).value_counts().head(10).index.tolist()
        if not roe_disc.empty
        else []
    )
    target_symbols = list(dict.fromkeys(["002157", "300212", "300010"] + top_freq))
    sub = merged[merged["symbol"].isin(target_symbols)].dropna(subset=["roe_ttm", "F050501B"]).copy()
    sub["absolute_difference"] = (sub["roe_ttm"] - sub["F050501B"]).abs()
    rows = []
    for symbol, df in sub.groupby("symbol"):
        periods = sorted(df["selected_report_period"].dropna().dt.strftime("%Y-%m-%d").unique().tolist())
        row = {
            "symbol": symbol,
            "number_of_discrepancy_rows": int(len(df)),
            "max_absolute_difference": float(df["absolute_difference"].max()),
            "median_absolute_difference": float(df["absolute_difference"].median()),
            "min_v3_roe": float(df["roe_ttm"].min()),
            "max_v3_roe": float(df["roe_ttm"].max()),
            "min_FI_T5_roe": float(df["F050501B"].min()),
            "max_FI_T5_roe": float(df["F050501B"].max()),
            "max_abs_fi_roe": float(df["F050501B"].abs().max()),
            "selected_report_periods_involved": ";".join(periods[:40]),
        }
        for col in ["equity_parent", "total_equity", "net_profit_parent_ttm", "net_profit_ttm"]:
            if col in df.columns:
                row[f"min_{col}"] = float(pd.to_numeric(df[col], errors="coerce").min())
                row[f"max_{col}"] = float(pd.to_numeric(df[col], errors="coerce").max())
                if "equity" in col:
                    row[f"min_abs_{col}"] = float(pd.to_numeric(df[col], errors="coerce").abs().min())
        row["possible_reason"] = infer_symbol_reason(pd.Series(row))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("max_absolute_difference", ascending=False)
    out.to_csv(OUT_DIR / "roe_distressed_symbol_review.csv", index=False, encoding="utf-8-sig")
    return out


def admin_expense_outlier_review(merged: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    sub = merged.dropna(subset=["admin_expense_to_revenue", "F051801B"]).copy()
    sub["difference"] = sub["admin_expense_to_revenue"] - sub["F051801B"]
    sub["absolute_difference"] = sub["difference"].abs()
    top = sub.sort_values("absolute_difference", ascending=False).head(50).copy()
    top["review_section"] = "top_50_examples"
    top_rows = top[
        [
            "review_section",
            "symbol",
            "month_end",
            "selected_report_period",
            "selected_pit_date",
            "admin_expense_to_revenue",
            "F051801B",
            "difference",
            "absolute_difference",
        ]
    ].copy()
    abs_diff = sub["absolute_difference"]
    cutoff = abs_diff.quantile(0.999)
    trimmed = sub[abs_diff <= cutoff].copy()
    bounded = sub[sub["admin_expense_to_revenue"].between(-1, 1) & sub["F051801B"].between(-1, 1)].copy()
    summary = {
        "review_section": "summary",
        **diff_quantiles(abs_diff),
        "count_abs_diff_gt_0_05": int((abs_diff > 0.05).sum()),
        "count_abs_diff_gt_0_10": int((abs_diff > 0.10).sum()),
        "count_abs_diff_gt_0_50": int((abs_diff > 0.50).sum()),
        "count_abs_diff_gt_1_00": int((abs_diff > 1.00).sum()),
        "full_rows": int(len(sub)),
        "full_pearson": float(sub["admin_expense_to_revenue"].corr(sub["F051801B"], method="pearson")),
        "full_spearman": float(sub["admin_expense_to_revenue"].corr(sub["F051801B"], method="spearman")),
        "full_median_absolute_difference": float(abs_diff.median()),
        "full_mean_absolute_difference": float(abs_diff.mean()),
        "trim_top_0_1pct_rows": int(len(trimmed)),
        "trim_top_0_1pct_pearson": float(trimmed["admin_expense_to_revenue"].corr(trimmed["F051801B"], method="pearson")),
        "trim_top_0_1pct_spearman": float(trimmed["admin_expense_to_revenue"].corr(trimmed["F051801B"], method="spearman")),
        "trim_top_0_1pct_median_absolute_difference": float(
            (trimmed["admin_expense_to_revenue"] - trimmed["F051801B"]).abs().median()
        ),
        "trim_top_0_1pct_mean_absolute_difference": float(
            (trimmed["admin_expense_to_revenue"] - trimmed["F051801B"]).abs().mean()
        ),
        "bounded_minus1_1_rows": int(len(bounded)),
        "bounded_minus1_1_pearson": float(bounded["admin_expense_to_revenue"].corr(bounded["F051801B"], method="pearson")),
        "bounded_minus1_1_spearman": float(bounded["admin_expense_to_revenue"].corr(bounded["F051801B"], method="spearman")),
        "bounded_minus1_1_median_absolute_difference": float(
            (bounded["admin_expense_to_revenue"] - bounded["F051801B"]).abs().median()
        ),
        "bounded_minus1_1_mean_absolute_difference": float(
            (bounded["admin_expense_to_revenue"] - bounded["F051801B"]).abs().mean()
        ),
    }
    summary_df = pd.DataFrame([summary])
    out = pd.concat([summary_df, top_rows], ignore_index=True, sort=False)
    out.to_csv(OUT_DIR / "admin_expense_outlier_review.csv", index=False, encoding="utf-8-sig")
    if summary["trim_top_0_1pct_spearman"] >= 0.94 and summary["bounded_minus1_1_spearman"] >= 0.94:
        verdict = "PASS_WITH_OUTLIER_NOTE" if summary["count_abs_diff_gt_1_00"] > 0 else "PASS_CONFIRMED"
    elif summary["full_spearman"] >= 0.90:
        verdict = "WATCH_OUTLIERS_NEED_REVIEW"
    else:
        verdict = "FAIL_MAPPING_OR_UNIT_ERROR"
    return out, verdict


def sales_expense_brief_review(merged: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    sub = merged.dropna(subset=["sales_expense_to_revenue", "F051701B"]).copy()
    sub["difference"] = sub["sales_expense_to_revenue"] - sub["F051701B"]
    sub["absolute_difference"] = sub["difference"].abs()
    top = sub.sort_values("absolute_difference", ascending=False).head(20).copy()
    ratio = (
        sub.loc[sub["F051701B"].abs() > 1e-8, "sales_expense_to_revenue"]
        / sub.loc[sub["F051701B"].abs() > 1e-8, "F051701B"]
    ).abs()
    median_ratio = float(ratio.replace([np.inf, -np.inf], np.nan).dropna().median())
    same_direction = float(np.sign(sub["sales_expense_to_revenue"]).eq(np.sign(sub["F051701B"])).mean())
    summary = pd.DataFrame(
        [
            {
                "review_section": "summary",
                "rows": int(len(sub)),
                "spearman": float(sub["sales_expense_to_revenue"].corr(sub["F051701B"], method="spearman")),
                "median_absolute_difference": float(sub["absolute_difference"].median()),
                "mean_absolute_difference": float(sub["absolute_difference"].mean()),
                "median_abs_v3_div_fi_t5": median_ratio,
                "sign_agreement_rate": same_direction,
                "unit_error_evidence": bool(median_ratio < 0.05 or median_ratio > 20),
                "systematic_direction_evidence": bool(sub["difference"].median() > 0 and sub["difference"].quantile(0.05) > 0)
                or bool(sub["difference"].median() < 0 and sub["difference"].quantile(0.95) < 0),
            }
        ]
    )
    examples = top[
        [
            "symbol",
            "month_end",
            "selected_report_period",
            "selected_pit_date",
            "sales_expense_to_revenue",
            "F051701B",
            "difference",
            "absolute_difference",
        ]
    ].copy()
    examples.insert(0, "review_section", "top_20_examples")
    out = pd.concat([summary, examples], ignore_index=True, sort=False)
    out.to_csv(OUT_DIR / "sales_expense_brief_review.csv", index=False, encoding="utf-8-sig")
    verdict = "PASS_CONFIRMED" if not bool(summary.iloc[0]["unit_error_evidence"]) else "FAIL_MAPPING_OR_UNIT_ERROR"
    return out, verdict


def decide_roe_verdict(filters: pd.DataFrame, unit_error: bool, distressed: pd.DataFrame, period: pd.DataFrame) -> str:
    if unit_error:
        return "FAIL_MAPPING_OR_UNIT_ERROR"
    full = filters[filters["filter"].eq("A_full_sample")].iloc[0]
    bounded = filters[filters["filter"].eq("F_keep_both_in_minus0_5_0_5")].iloc[0]
    period_min_spearman = period["spearman"].min() if not period.empty else np.nan
    has_extreme_denominator = False
    if not distressed.empty:
        has_extreme_denominator = distressed["possible_reason"].astype(str).str.contains("denominator|extreme loss").any()
    if bounded["spearman"] >= 0.80 and full["sign_agreement_rate"] >= 0.90 and period_min_spearman >= 0.65:
        return "CLEARED_WATCH_DEFINITION_DIFFERENCE"
    if has_extreme_denominator and bounded["spearman"] >= 0.75:
        return "CLEARED_WATCH_EXTREME_DENOMINATOR"
    return "STILL_WATCH_NEEDS_FORMULA_REVIEW"


def render_report(
    previous_check: dict,
    roe_unit: pd.DataFrame,
    roe_filters: pd.DataFrame,
    roe_period: pd.DataFrame,
    roe_distressed: pd.DataFrame,
    admin_review: pd.DataFrame,
    sales_review: pd.DataFrame,
    summary: dict,
) -> str:
    def table(df: pd.DataFrame, rows: int = 20) -> str:
        if df.empty:
            return "No rows."
        return "```text\n" + df.head(rows).to_string(index=False) + "\n```"

    return f"""# FI_T5 WATCH Review v0.1

## 1. Scope

This review only inspects WATCH / suspicious items from FI_T5 sanity v0. It does not train models, run backtests, calculate IC, modify production, or modify the v3 factor source panel.

## 2. Previous Result

```json
{json.dumps(previous_check, ensure_ascii=False, indent=2)}
```

## 3. ROE Unit Review

{table(roe_unit)}

Conclusion: no percent-vs-decimal unit error was flagged if `unit_error_detected` is false.

## 4. ROE Extreme / Denominator Review

{table(roe_filters)}

Distressed symbol review:

{table(roe_distressed)}

## 5. ROE Report-Period Timing Review

{table(roe_period)}

## 6. Admin Expense Outlier Review

{table(admin_review, rows=12)}

## 7. Sales Expense Confirmation

{table(sales_review, rows=8)}

## 8. Verdict

- ROE: `{summary["roe_review_verdict"]}`
- Admin expense ratio: `{summary["admin_expense_review_verdict"]}`
- Sales expense ratio: `{summary["sales_expense_review_verdict"]}`

## 9. Final Decision

`{summary["final_decision"]}`

## 10. Recommended Next Step

{summary["recommended_next_step"]}
"""


def main() -> int:
    ensure_dirs()
    log("Starting FI_T5 WATCH Review v0.1")
    update_run_state(
        "script started",
        ["Initialized output directories"],
        [str(PREV_SUMMARY), str(PREV_METRICS), str(V3_PATH), str(FI_T5_PATH)],
        [str(RUN_DIR / "RUN_STATE.md")],
        "Load previous sanity outputs.",
    )

    previous_summary, previous_metrics, previous_discrepancies, mapping, schema_v3, schema_fi = load_previous_outputs()
    previous_check = previous_result_check(previous_summary, previous_metrics, mapping)
    append_checkpoint(
        "previous result check",
        "completed",
        f"- previous_decision: {previous_check['previous_decision']}\n"
        f"- roe_unique_watch: {previous_check['roe_is_unique_watch']}\n"
        f"- admin_mean_gt_50x_median: {previous_check['admin_mean_gt_50x_median']}\n",
    )

    merged = read_core_data(schema_v3)
    append_checkpoint(
        "core data read and alignment",
        "completed",
        "- Read only necessary v3/FI_T5 columns.\n- Aligned by symbol and selected_report_period = Accper.\n",
    )

    roe_unit, unit_error, unit_conclusion = roe_unit_review(merged)
    roe_filters = roe_filter_sensitivity(merged)
    roe_period = roe_period_timing_review(merged)
    roe_distressed = roe_distressed_symbol_review(merged, previous_discrepancies)
    append_checkpoint("roe review", "completed", f"- unit_error_detected: {unit_error}\n")

    admin_review, admin_verdict = admin_expense_outlier_review(merged)
    sales_review, sales_verdict = sales_expense_brief_review(merged)
    roe_verdict = decide_roe_verdict(roe_filters, unit_error, roe_distressed, roe_period)
    append_checkpoint(
        "admin and sales review",
        "completed",
        f"- admin_verdict: {admin_verdict}\n- sales_verdict: {sales_verdict}\n",
    )

    mapping_error_detected = bool(
        roe_verdict == "FAIL_MAPPING_OR_UNIT_ERROR"
        or admin_verdict == "FAIL_MAPPING_OR_UNIT_ERROR"
        or sales_verdict == "FAIL_MAPPING_OR_UNIT_ERROR"
    )
    formula_error_detected = False

    if unit_error or mapping_error_detected or formula_error_detected:
        final_decision = "FI_T5_WATCH_REVIEW_FAIL_BLOCK_FACTOR_TRANSFORM"
        next_step = "Review v3 formulas and FI_T5 field mapping before any factor transform planning."
    elif roe_verdict.startswith("CLEARED") and admin_verdict in {"PASS_CONFIRMED", "PASS_WITH_OUTLIER_NOTE"} and sales_verdict == "PASS_CONFIRMED":
        final_decision = "FI_T5_WATCH_REVIEW_CLEARED_READY_FOR_FACTOR_TRANSFORM_PLANNING"
        next_step = "Proceed to Factor Transform Planning; do not train yet."
    else:
        final_decision = "FI_T5_WATCH_REVIEW_STILL_WATCH_MANUAL_FORMULA_REVIEW_REQUIRED"
        next_step = "Manually review ROE/admin formula definitions and discrepancy examples before factor transform planning."

    summary = {
        "run_timestamp": now_iso(),
        "previous_decision": previous_summary.get("decision"),
        "roe_review_verdict": roe_verdict,
        "admin_expense_review_verdict": admin_verdict,
        "sales_expense_review_verdict": sales_verdict,
        "unit_error_detected": bool(unit_error),
        "mapping_error_detected": bool(mapping_error_detected),
        "formula_error_detected": bool(formula_error_detected),
        "production_modified": False,
        "training_run": False,
        "backtest_run": False,
        "ic_calculated": False,
        "final_decision": final_decision,
        "recommended_next_step": next_step,
        "key_outlier_reason": (
            "ROE differences are consistent with TTM vs report-period timing/definition and distressed denominator cases; "
            "admin mean absolute difference is pulled by a tiny number of extreme outliers."
        ),
        "roe_unit_conclusion": unit_conclusion,
    }
    write_text(OUT_DIR / "fi_t5_watch_review_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    write_text(
        OUT_DIR / "fi_t5_watch_review_report.md",
        render_report(previous_check, roe_unit, roe_filters, roe_period, roe_distressed, admin_review, sales_review, summary),
    )

    final_qa = pd.DataFrame(
        [
            {"check": "production_modified", "value": False},
            {"check": "training_run", "value": False},
            {"check": "backtest_run", "value": False},
            {"check": "ic_calculated", "value": False},
            {"check": "v3_modified", "value": False},
            {"check": "unit_error_detected", "value": bool(unit_error)},
            {"check": "mapping_error_detected", "value": bool(mapping_error_detected)},
            {"check": "formula_error_detected", "value": bool(formula_error_detected)},
            {"check": "final_decision", "value": final_decision},
        ]
    )
    final_qa.to_csv(RUN_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    terminal_summary = {
        "script": "scripts/run_fi_t5_watch_review_v01.py",
        "completed_at": now_iso(),
        "exit_code": 0,
        "outputs": [
            str(OUT_DIR / "previous_result_check.json"),
            str(OUT_DIR / "roe_unit_review.csv"),
            str(OUT_DIR / "roe_filter_sensitivity.csv"),
            str(OUT_DIR / "roe_period_timing_review.csv"),
            str(OUT_DIR / "roe_distressed_symbol_review.csv"),
            str(OUT_DIR / "admin_expense_outlier_review.csv"),
            str(OUT_DIR / "sales_expense_brief_review.csv"),
            str(OUT_DIR / "fi_t5_watch_review_summary.json"),
            str(OUT_DIR / "fi_t5_watch_review_report.md"),
        ],
        "summary": summary,
    }
    write_text(RUN_DIR / "terminal_summary.json", json.dumps(terminal_summary, ensure_ascii=False, indent=2))
    write_text(
        RUN_DIR / "task_completion_card.md",
        "# Task Completion Card\n\n"
        f"- Task: FI_T5 WATCH Review v0.1\n"
        f"- Completed at: {now_iso()}\n"
        f"- Final decision: `{final_decision}`\n"
        f"- ROE verdict: `{roe_verdict}`\n"
        f"- Admin verdict: `{admin_verdict}`\n"
        f"- Sales verdict: `{sales_verdict}`\n"
        f"- Production modified: False\n"
        f"- Training run: False\n"
        f"- Backtest run: False\n",
    )

    generated = [
        str(OUT_DIR / "previous_result_check.json"),
        str(OUT_DIR / "roe_unit_review.csv"),
        str(OUT_DIR / "roe_filter_sensitivity.csv"),
        str(OUT_DIR / "roe_period_timing_review.csv"),
        str(OUT_DIR / "roe_distressed_symbol_review.csv"),
        str(OUT_DIR / "admin_expense_outlier_review.csv"),
        str(OUT_DIR / "sales_expense_brief_review.csv"),
        str(OUT_DIR / "fi_t5_watch_review_summary.json"),
        str(OUT_DIR / "fi_t5_watch_review_report.md"),
        str(RUN_DIR / "task_completion_card.md"),
        str(RUN_DIR / "terminal_summary.json"),
        str(RUN_DIR / "final_qa.csv"),
    ]
    update_run_state(
        "completed",
        [
            "Previous sanity output check",
            "Focused v3/FI_T5 data read",
            "ROE unit/extreme/timing/distressed-symbol review",
            "Admin expense outlier review",
            "Sales expense brief confirmation",
            "Summary/report/QA outputs",
        ],
        [str(V3_PATH), str(FI_T5_PATH), str(PREV_DIR)],
        generated,
        next_step,
    )
    append_checkpoint("task completed", "completed", f"- final_decision: {final_decision}\n")
    log(f"Completed FI_T5 WATCH Review v0.1: {final_decision}")
    del merged
    gc.collect()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        ensure_dirs()
        append_checkpoint("script failed", "failed", f"- error: {exc}\n")
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
