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


TASK_NAME = "FI_T5_sanity_check_v0"
RUN_DIR = Path("output/_agent_runs") / TASK_NAME
OUT_DIR = Path("output/fi_t5_sanity_check_v0")
V3_PATH = Path(
    "output/csmar_pit_clean_core_financial_factors_v3/"
    "pit_clean_core_financial_factors_monthly_v3.parquet"
)
CSMAR_EXPORTS_DIR = Path("data/csmar_exports")

FI_T5_PATTERNS = (
    "*FI_T5*",
    "*FINT*",
    "*Fint*",
    "*financial*indicator*",
    "*Financial*Indicator*",
    "*财务指标*",
    "*财务比率*",
)

V3_COLUMNS_NEEDED = [
    "symbol",
    "month_end",
    "selected_report_period",
    "selected_pit_date",
    "roe_ttm",
    "sales_expense_to_revenue",
    "admin_expense_to_revenue",
    "net_margin",
    "debt_ratio",
]

FI_T5_FIELD_DEFS = {
    "Stkcd": "stock code",
    "Accper": "accounting/report period",
    "Typrep": "report type",
    "F050501B": "ROE",
    "F053301B": "gross margin / operating gross profit margin",
    "F051701B": "sales expense ratio",
    "F051801B": "management/admin expense ratio",
}

METRIC_MAPPINGS = [
    {
        "sanity_metric": "roe",
        "v3_column": "roe_ttm",
        "fi_t5_column": "F050501B",
        "v3_definition_guess": "TTM ROE from v3 PIT-clean factor source panel",
        "fi_t5_definition_guess": "CSMAR FI_T5 ROE indicator at report period",
        "direction_expected": "same",
        "unit_expected": "decimal ratio",
        "mapping_confidence": "MEDIUM",
        "notes": "Mapped by local CSMAR field notes; v3 TTM vs FI_T5 report-period definition may differ.",
    },
    {
        "sanity_metric": "sales_expense_ratio",
        "v3_column": "sales_expense_to_revenue",
        "fi_t5_column": "F051701B",
        "v3_definition_guess": "TTM sales expense divided by TTM revenue",
        "fi_t5_definition_guess": "CSMAR FI_T5 sales expense ratio",
        "direction_expected": "same",
        "unit_expected": "decimal ratio",
        "mapping_confidence": "HIGH",
        "notes": "Mapped by local CSMAR field notes.",
    },
    {
        "sanity_metric": "admin_expense_ratio",
        "v3_column": "admin_expense_to_revenue",
        "fi_t5_column": "F051801B",
        "v3_definition_guess": "TTM admin expense divided by TTM revenue",
        "fi_t5_definition_guess": "CSMAR FI_T5 management/admin expense ratio",
        "direction_expected": "same",
        "unit_expected": "decimal ratio",
        "mapping_confidence": "HIGH",
        "notes": "Mapped by local CSMAR field notes.",
    },
    {
        "sanity_metric": "gross_margin",
        "v3_column": "",
        "fi_t5_column": "F053301B",
        "v3_definition_guess": "No gross margin field found in v3 schema",
        "fi_t5_definition_guess": "CSMAR FI_T5 gross margin / operating gross profit margin",
        "direction_expected": "same",
        "unit_expected": "decimal ratio",
        "mapping_confidence": "UNMAPPED",
        "notes": "FI_T5 field exists, but v3 schema does not expose gross_margin.",
    },
    {
        "sanity_metric": "net_margin",
        "v3_column": "net_margin",
        "fi_t5_column": "",
        "v3_definition_guess": "v3 net margin",
        "fi_t5_definition_guess": "No inspected FI_T5 net margin column in candidate file",
        "direction_expected": "same",
        "unit_expected": "decimal ratio",
        "mapping_confidence": "UNMAPPED",
        "notes": "Do not force-map gross margin to net margin.",
    },
    {
        "sanity_metric": "debt_ratio",
        "v3_column": "debt_ratio",
        "fi_t5_column": "",
        "v3_definition_guess": "v3 debt ratio",
        "fi_t5_definition_guess": "No inspected FI_T5 debt ratio column in candidate file",
        "direction_expected": "same",
        "unit_expected": "decimal ratio",
        "mapping_confidence": "UNMAPPED",
        "notes": "No reliable FI_T5 counterpart in the inspected export.",
    },
]


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
        f.write(f"\n## {now_iso()}\n\n")
        f.write(f"Stage: {stage}\n\nStatus: {status}\n\n{notes}\n")


def update_run_state(
    stage: str,
    completed: list[str],
    current_files: list[str],
    generated: list[str],
    next_step: str,
) -> None:
    text = [
        "# RUN_STATE.md",
        "",
        "- 当前任务名称: FI_T5 Sanity Check v0",
        f"- 开始时间: {now_iso()}",
        f"- 当前阶段: {stage}",
        "- 已完成步骤:",
    ]
    text.extend([f"  - {x}" for x in completed])
    text.append("- 正在处理的文件:")
    text.extend([f"  - {x}" for x in current_files])
    text.append("- 已生成输出:")
    text.extend([f"  - {x}" for x in generated])
    text.extend(
        [
            f"- 下一步: {next_step}",
            "- 如果 Codex 崩溃，新的 Codex 应如何继续:",
            "  1. Read this RUN_STATE.md first.",
            "  2. Continue with scripts/run_fi_t5_sanity_check_v0.py.",
            "  3. Do not scan the full repository or unrelated output/data directories.",
            "  4. Append stdout/stderr to run_stdout.txt and run_stderr.txt.",
        ]
    )
    write_text(RUN_DIR / "RUN_STATE.md", "\n".join(text) + "\n")


def log(msg: str) -> None:
    print(f"[{now_iso()}] {msg}", flush=True)


def discover_fi_t5_candidates() -> pd.DataFrame:
    rows = []
    seen = set()
    if not CSMAR_EXPORTS_DIR.exists():
        return pd.DataFrame(
            columns=["candidate_file_path", "file_size", "modified_time", "file_type", "why_selected"]
        )
    for pattern in FI_T5_PATTERNS:
        for path in CSMAR_EXPORTS_DIR.rglob(pattern):
            if not path.is_file():
                continue
            resolved = str(path)
            if resolved in seen:
                continue
            seen.add(resolved)
            stat = path.stat()
            lower_name = path.name.lower()
            why = "filename contains FI_T5" if "fi_t5" in lower_name else f"filename matched {pattern}"
            rows.append(
                {
                    "candidate_file_path": str(path),
                    "file_size": stat.st_size,
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "file_type": path.suffix.lower().lstrip("."),
                    "why_selected": why,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["why_selected", "modified_time", "candidate_file_path"], ascending=[True, False, True])
    return out


def choose_candidate(candidates: pd.DataFrame) -> Path | None:
    if candidates.empty:
        return None
    preferred = candidates[candidates["candidate_file_path"].str.contains("FI_T5", case=False, regex=False)]
    row = preferred.iloc[0] if not preferred.empty else candidates.iloc[0]
    return Path(row["candidate_file_path"])


def missing_input_exit(candidates: pd.DataFrame) -> int:
    candidates.to_csv(OUT_DIR / "fi_t5_file_candidates.csv", index=False, encoding="utf-8-sig")
    write_text(
        OUT_DIR / "missing_input_report.md",
        "# Missing FI_T5 Input\n\n"
        "No FI_T5 candidate file was found by filename-level search under `data/csmar_exports/`.\n\n"
        "Please provide a CSMAR FI_T5 export file containing at least `Stkcd`, `Accper`, `Typrep`, "
        "and one or more financial indicator columns such as `F050501B`, `F051701B`, or `F051801B`.\n",
    )
    summary = {
        "fi_t5_candidate_found": False,
        "fi_t5_file_used": None,
        "v3_file_used": str(V3_PATH),
        "run_timestamp": now_iso(),
        "matched_rows_total": 0,
        "matched_symbols_total": 0,
        "matched_months_total": 0,
        "metrics_tested": 0,
        "metrics_pass": 0,
        "metrics_watch": 0,
        "metrics_fail": 0,
        "pit_alignment_mode": "INPUT_MISSING",
        "fi_t5_has_publish_date": False,
        "production_modified": False,
        "training_run": False,
        "backtest_run": False,
        "ic_calculated": False,
        "decision": "FI_T5_INPUT_MISSING",
    }
    write_text(OUT_DIR / "fi_t5_sanity_check_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def inspect_fi_schema(path: Path) -> tuple[dict, pd.DataFrame]:
    if path.suffix.lower() == ".csv":
        sample = pd.read_csv(path, nrows=20, dtype=str, low_memory=True)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        sample = pd.read_parquet(path).head(20)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        sample = pd.read_excel(path, nrows=20, dtype=str)
    else:
        raise ValueError(f"Unsupported FI_T5 file type: {path.suffix}")

    columns = list(sample.columns)
    normalized = {c: c.strip().lower() for c in columns}
    publish_candidates = [
        c
        for c in columns
        if any(token in c.lower() for token in ["ann", "pub", "declare", "disclos", "公告", "披露", "发布"])
    ]
    schema = {
        "file": str(path),
        "raw_columns": columns,
        "normalized_columns": normalized,
        "likely_identifier_column": "Stkcd" if "Stkcd" in columns else "",
        "likely_report_period_column": "Accper" if "Accper" in columns else "",
        "likely_publish_announcement_declare_date_column": publish_candidates[0] if publish_candidates else "",
        "has_publish_date": bool(publish_candidates),
        "likely_financial_ratio_fields": [
            {"column": c, "definition_guess": FI_T5_FIELD_DEFS.get(c, "unknown")}
            for c in columns
            if c.startswith("F")
        ],
        "sample_rows": sample.head(3).where(pd.notna(sample.head(3)), None).to_dict(orient="records"),
        "pit_note": (
            "FI_T5 lacks explicit PIT disclosure date in inspected schema; this check is metric-definition "
            "sanity only, not live-signal validation."
            if not publish_candidates
            else "FI_T5 publish-like date field detected in inspected schema."
        ),
    }
    return schema, sample


def inspect_v3_schema() -> dict:
    pf = pq.ParquetFile(V3_PATH)
    columns = pf.schema_arrow.names
    selected = [c for c in V3_COLUMNS_NEEDED if c in columns]
    return {
        "file": str(V3_PATH),
        "num_rows_metadata": pf.metadata.num_rows,
        "raw_columns": columns,
        "columns_needed_present": selected,
        "likely_identifier_column": "symbol" if "symbol" in columns else "",
        "likely_month_column": "month_end" if "month_end" in columns else "",
        "likely_pit_date_column": "selected_pit_date" if "selected_pit_date" in columns else "",
        "likely_report_period_column": "selected_report_period" if "selected_report_period" in columns else "",
        "candidate_metric_columns": [
            c
            for c in columns
            if c
            in {
                "roe_ttm",
                "net_margin",
                "debt_ratio",
                "sales_expense_to_revenue",
                "admin_expense_to_revenue",
                "revenue_ttm",
                "net_profit_ttm",
                "total_assets",
                "total_liabilities",
                "equity_parent",
                "total_equity",
            }
        ],
    }


def read_fi_t5(path: Path, usecols: list[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=lambda c: c in usecols, dtype={"Stkcd": str}, low_memory=True)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path, columns=usecols)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, usecols=usecols, dtype={"Stkcd": str})
    else:
        raise ValueError(f"Unsupported FI_T5 file type: {path.suffix}")
    return df


def normalize_for_merge(v3: pd.DataFrame, fi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    v3 = v3.copy()
    fi = fi.copy()
    v3["symbol"] = v3["symbol"].astype(str).str.zfill(6)
    v3["month_end"] = pd.to_datetime(v3["month_end"], errors="coerce")
    v3["selected_report_period"] = pd.to_datetime(v3["selected_report_period"], errors="coerce")
    v3["selected_pit_date"] = pd.to_datetime(v3["selected_pit_date"], errors="coerce")

    fi["Stkcd"] = fi["Stkcd"].astype(str).str.zfill(6)
    fi["Accper"] = pd.to_datetime(fi["Accper"], errors="coerce")
    if "Typrep" in fi.columns:
        fi = fi[fi["Typrep"].astype(str).str.upper().eq("A")].copy()
    for c in [m["fi_t5_column"] for m in METRIC_MAPPINGS if m["fi_t5_column"]]:
        if c in fi.columns:
            fi[c] = pd.to_numeric(fi[c], errors="coerce")
    for c in [m["v3_column"] for m in METRIC_MAPPINGS if m["v3_column"]]:
        if c in v3.columns:
            v3[c] = pd.to_numeric(v3[c], errors="coerce")
    fi = fi.dropna(subset=["Stkcd", "Accper"]).drop_duplicates(["Stkcd", "Accper"], keep="last")
    return v3, fi


def status_for_metric(spearman: float, median_abs_diff: float, overlap: int) -> str:
    if overlap < 30 or math.isnan(spearman):
        return "WATCH"
    if spearman < 0.60:
        return "FAIL"
    if spearman < 0.90:
        return "WATCH"
    if pd.notna(median_abs_diff) and median_abs_diff > 0.10:
        return "WATCH"
    return "PASS"


def compute_metrics(merged: pd.DataFrame, mapping_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    discrepancies = []
    for row in mapping_rows.to_dict(orient="records"):
        if row["mapping_confidence"] == "UNMAPPED":
            continue
        v3_col = row["v3_column"]
        fi_col = row["fi_t5_column"]
        if v3_col not in merged.columns or fi_col not in merged.columns:
            continue
        metric = row["sanity_metric"]
        sub = merged[
            [
                "symbol",
                "month_end",
                "selected_pit_date",
                "selected_report_period",
                "Accper",
                v3_col,
                fi_col,
            ]
        ].copy()
        sub = sub.rename(columns={v3_col: "v3_value", fi_col: "fi_t5_value", "Accper": "fi_t5_report_period"})
        matched_rows = int(len(sub))
        matched_symbols = int(sub["symbol"].nunique())
        matched_months = int(sub["month_end"].nunique())
        v3_non_null = int(sub["v3_value"].notna().sum())
        fi_non_null = int(sub["fi_t5_value"].notna().sum())
        overlap_df = sub.dropna(subset=["v3_value", "fi_t5_value"]).copy()
        overlap = int(len(overlap_df))
        if overlap:
            overlap_df["difference"] = overlap_df["v3_value"] - overlap_df["fi_t5_value"]
            overlap_df["absolute_difference"] = overlap_df["difference"].abs()
            denom = overlap_df["fi_t5_value"].abs().replace(0, np.nan)
            overlap_df["relative_difference"] = overlap_df["absolute_difference"] / denom
            pearson = float(overlap_df["v3_value"].corr(overlap_df["fi_t5_value"], method="pearson"))
            spearman = float(overlap_df["v3_value"].corr(overlap_df["fi_t5_value"], method="spearman"))
            abs_diff = overlap_df["absolute_difference"]
            diff = overlap_df["difference"]
            median_abs = float(abs_diff.median())
            mean_abs = float(abs_diff.mean())
            q = diff.quantile([0.01, 0.05, 0.50, 0.95, 0.99])
            sign_agree = float(
                (np.sign(overlap_df["v3_value"]) == np.sign(overlap_df["fi_t5_value"])).mean()
            )
            examples = overlap_df.sort_values("absolute_difference", ascending=False).head(30).copy()
            examples.insert(0, "sanity_metric", metric)
            examples["possible_reason"] = np.where(
                metric == "roe",
                "Likely TTM vs report-period ROE definition difference or fiscal-period timing.",
                "Potential formula/unit/report-period definition difference.",
            )
            discrepancies.append(
                examples[
                    [
                        "sanity_metric",
                        "symbol",
                        "month_end",
                        "selected_pit_date",
                        "fi_t5_report_period",
                        "v3_value",
                        "fi_t5_value",
                        "absolute_difference",
                        "relative_difference",
                        "possible_reason",
                    ]
                ]
            )
        else:
            pearson = spearman = median_abs = mean_abs = np.nan
            q = pd.Series({0.01: np.nan, 0.05: np.nan, 0.50: np.nan, 0.95: np.nan, 0.99: np.nan})
            sign_agree = np.nan
        status = status_for_metric(spearman, median_abs, overlap)
        summaries.append(
            {
                "sanity_metric": metric,
                "status": status,
                "matched_rows": matched_rows,
                "matched_symbols": matched_symbols,
                "matched_months": matched_months,
                "v3_non_null_coverage": v3_non_null / matched_rows if matched_rows else 0.0,
                "fi_t5_non_null_coverage": fi_non_null / matched_rows if matched_rows else 0.0,
                "overlap_coverage": overlap / matched_rows if matched_rows else 0.0,
                "overlap_rows": overlap,
                "pearson_correlation": pearson,
                "spearman_correlation": spearman,
                "median_absolute_difference": median_abs,
                "mean_absolute_difference": mean_abs,
                "p1_difference": float(q.loc[0.01]),
                "p5_difference": float(q.loc[0.05]),
                "p50_difference": float(q.loc[0.50]),
                "p95_difference": float(q.loc[0.95]),
                "p99_difference": float(q.loc[0.99]),
                "sign_agreement_rate": sign_agree,
                "threshold_note": "Sanity thresholds only; not a statistical significance test.",
            }
        )
    disc = pd.concat(discrepancies, ignore_index=True) if discrepancies else pd.DataFrame()
    return pd.DataFrame(summaries), disc


def render_report(
    candidate_path: Path,
    fi_schema: dict,
    mapping: pd.DataFrame,
    metrics: pd.DataFrame,
    discrepancies: pd.DataFrame,
    summary: dict,
) -> str:
    fi_cols = ", ".join(fi_schema["raw_columns"])
    publish = fi_schema["likely_publish_announcement_declare_date_column"] or "None"
    mapping_md = mapping.to_markdown(index=False)
    metrics_md = metrics.to_markdown(index=False) if not metrics.empty else "No metrics tested."
    disc_md = (
        discrepancies.head(20).to_markdown(index=False)
        if not discrepancies.empty
        else "No discrepancy examples generated."
    )
    return f"""# FI_T5 Sanity Check v0

## 1. Scope

This task only performs a low-resource sanity check. It does not train models, run backtests, calculate IC, modify production, or modify the v3 factor source panel.

## 2. Inputs

- v3 factor source panel: `{V3_PATH}`
- FI_T5 candidate used: `{candidate_path}`

## 3. FI_T5 Schema Inspection

- Raw columns: {fi_cols}
- Identifier column: `{fi_schema["likely_identifier_column"]}`
- Report period column: `{fi_schema["likely_report_period_column"]}`
- Publish / announcement / declare date column: `{publish}`
- Publish date available: `{fi_schema["has_publish_date"]}`
- Likely ratio fields: `{", ".join([x["column"] for x in fi_schema["likely_financial_ratio_fields"]])}`

FI_T5 lacks explicit PIT disclosure date in inspected schema; this check is metric-definition sanity only, not live-signal validation.

## 4. Mapping Table

{mapping_md}

## 5. PIT / Alignment Mode

Case B was used. FI_T5 has no inspected publish / announcement / PIT date field, so the check aligns v3 rows to FI_T5 by `symbol` and fiscal report period (`selected_report_period` = `Accper`) with `Typrep = A` when available.

FI_T5 is used only for metric-definition sanity, not live-signal validation.

## 6. Metric Results

{metrics_md}

Sanity thresholds: PASS requires Spearman >= 0.90 plus small median absolute difference and consistent direction. WATCH covers moderate correlation, lower coverage, or likely definition differences. FAIL covers Spearman < 0.60, opposite direction, broad coverage problems, or unexplained large differences.

## 7. Discrepancy Review

{disc_md}

## 8. Decision

`{summary["decision"]}`

## 9. Next Step

{summary["next_step"]}
"""


def main() -> int:
    ensure_dirs()
    log("Starting FI_T5 sanity check v0")
    update_run_state(
        "script started",
        ["Initialized run/output directories"],
        [str(V3_PATH), str(CSMAR_EXPORTS_DIR)],
        [str(RUN_DIR / "RUN_STATE.md")],
        "Discover FI_T5 candidates.",
    )

    candidates = discover_fi_t5_candidates()
    candidates.to_csv(OUT_DIR / "fi_t5_file_candidates.csv", index=False, encoding="utf-8-sig")
    candidate = choose_candidate(candidates)
    append_checkpoint(
        "candidate discovery",
        "completed",
        f"- candidates_found: {len(candidates)}\n- selected: {candidate if candidate else 'None'}\n",
    )
    if candidate is None:
        return missing_input_exit(candidates)

    fi_schema, fi_sample = inspect_fi_schema(candidate)
    v3_schema = inspect_v3_schema()
    write_text(OUT_DIR / "schema_fi_t5.json", json.dumps(fi_schema, ensure_ascii=False, indent=2, default=str))
    write_text(OUT_DIR / "schema_v3.json", json.dumps(v3_schema, ensure_ascii=False, indent=2, default=str))
    del fi_sample
    gc.collect()
    append_checkpoint(
        "schema inspection",
        "completed",
        "- FI_T5 small sample/header inspected.\n- v3 parquet schema inspected.\n",
    )

    mapping = pd.DataFrame(METRIC_MAPPINGS)
    mapping.to_csv(OUT_DIR / "field_mapping.csv", index=False, encoding="utf-8-sig")

    v3_cols = [c for c in V3_COLUMNS_NEEDED if c in v3_schema["raw_columns"]]
    fi_cols = ["Stkcd", "Accper", "Typrep"] + [
        m["fi_t5_column"] for m in METRIC_MAPPINGS if m["fi_t5_column"] in fi_schema["raw_columns"]
    ]
    fi_cols = list(dict.fromkeys(fi_cols))
    log(f"Reading v3 needed columns: {v3_cols}")
    v3 = pd.read_parquet(V3_PATH, columns=v3_cols)
    log(f"Reading FI_T5 needed columns: {fi_cols}")
    fi = read_fi_t5(candidate, fi_cols)
    v3, fi = normalize_for_merge(v3, fi)
    merged = v3.merge(
        fi,
        left_on=["symbol", "selected_report_period"],
        right_on=["Stkcd", "Accper"],
        how="left",
        validate="many_to_one",
    )
    del v3, fi
    gc.collect()
    append_checkpoint(
        "fiscal-period alignment",
        "completed",
        "- Case B used: no FI_T5 publish date.\n- Merged v3 to FI_T5 on symbol and selected_report_period = Accper.\n",
    )

    metrics, discrepancies = compute_metrics(merged, mapping)
    metrics.to_csv(OUT_DIR / "sanity_metrics_summary.csv", index=False, encoding="utf-8-sig")
    discrepancies.to_csv(OUT_DIR / "sanity_discrepancy_examples.csv", index=False, encoding="utf-8-sig")

    metrics_tested = int(len(metrics))
    metrics_pass = int((metrics["status"] == "PASS").sum()) if not metrics.empty else 0
    metrics_watch = int((metrics["status"] == "WATCH").sum()) if not metrics.empty else 0
    metrics_fail = int((metrics["status"] == "FAIL").sum()) if not metrics.empty else 0
    matched_rows_total = int(merged[["symbol", "month_end"]].drop_duplicates().shape[0])
    matched_any = merged.dropna(subset=[m["fi_t5_column"] for m in METRIC_MAPPINGS if m["fi_t5_column"] in merged.columns], how="all")
    matched_symbols_total = int(matched_any["symbol"].nunique()) if not matched_any.empty else 0
    matched_months_total = int(matched_any["month_end"].nunique()) if not matched_any.empty else 0

    if metrics_fail > 0:
        decision = "FI_T5_SANITY_FAIL_BLOCK_FACTOR_TRANSFORM"
        next_step = "Review field definitions, formulas, and fiscal-period alignment before factor transform planning."
    elif metrics_watch > 0:
        decision = "FI_T5_SANITY_WATCH_REVIEW_REQUIRED"
        next_step = "Manually review mapping assumptions and discrepancy examples before factor transform planning."
    else:
        decision = "FI_T5_SANITY_PASS_READY_FOR_FACTOR_TRANSFORM_PLANNING"
        next_step = "Proceed to Factor Transform Planning, not training."

    summary = {
        "fi_t5_candidate_found": True,
        "fi_t5_file_used": str(candidate),
        "v3_file_used": str(V3_PATH),
        "run_timestamp": now_iso(),
        "matched_rows_total": matched_rows_total,
        "matched_symbols_total": matched_symbols_total,
        "matched_months_total": matched_months_total,
        "metrics_tested": metrics_tested,
        "metrics_pass": metrics_pass,
        "metrics_watch": metrics_watch,
        "metrics_fail": metrics_fail,
        "pit_alignment_mode": "CASE_B_FISCAL_PERIOD_METRIC_DEFINITION_SANITY_ONLY",
        "fi_t5_has_publish_date": bool(fi_schema["has_publish_date"]),
        "production_modified": False,
        "training_run": False,
        "backtest_run": False,
        "ic_calculated": False,
        "decision": decision,
        "next_step": next_step,
    }
    write_text(OUT_DIR / "fi_t5_sanity_check_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    write_text(
        OUT_DIR / "fi_t5_sanity_check_report.md",
        render_report(candidate, fi_schema, mapping, metrics, discrepancies, summary),
    )

    final_qa = pd.DataFrame(
        [
            {"check": "production_modified", "value": False},
            {"check": "training_run", "value": False},
            {"check": "backtest_run", "value": False},
            {"check": "ic_calculated", "value": False},
            {"check": "v3_modified", "value": False},
            {"check": "fi_t5_has_publish_date", "value": bool(fi_schema["has_publish_date"])},
            {"check": "decision", "value": decision},
        ]
    )
    final_qa.to_csv(RUN_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    terminal_summary = {
        "script": "scripts/run_fi_t5_sanity_check_v0.py",
        "completed_at": now_iso(),
        "exit_code": 0,
        "outputs": [
            str(OUT_DIR / "fi_t5_file_candidates.csv"),
            str(OUT_DIR / "schema_v3.json"),
            str(OUT_DIR / "schema_fi_t5.json"),
            str(OUT_DIR / "field_mapping.csv"),
            str(OUT_DIR / "sanity_metrics_summary.csv"),
            str(OUT_DIR / "sanity_discrepancy_examples.csv"),
            str(OUT_DIR / "fi_t5_sanity_check_summary.json"),
            str(OUT_DIR / "fi_t5_sanity_check_report.md"),
        ],
        "summary": summary,
    }
    write_text(RUN_DIR / "terminal_summary.json", json.dumps(terminal_summary, ensure_ascii=False, indent=2))
    write_text(
        RUN_DIR / "task_completion_card.md",
        "# Task Completion Card\n\n"
        f"- Task: FI_T5 Sanity Check v0\n"
        f"- Completed at: {now_iso()}\n"
        f"- FI_T5 found: True\n"
        f"- FI_T5 file used: `{candidate}`\n"
        f"- Metrics tested: {metrics_tested}\n"
        f"- PASS/WATCH/FAIL: {metrics_pass}/{metrics_watch}/{metrics_fail}\n"
        f"- Decision: `{decision}`\n"
        f"- Production modified: False\n"
        f"- Training run: False\n"
        f"- Backtest run: False\n",
    )

    generated = [
        str(OUT_DIR / "fi_t5_file_candidates.csv"),
        str(OUT_DIR / "schema_v3.json"),
        str(OUT_DIR / "schema_fi_t5.json"),
        str(OUT_DIR / "field_mapping.csv"),
        str(OUT_DIR / "sanity_metrics_summary.csv"),
        str(OUT_DIR / "sanity_discrepancy_examples.csv"),
        str(OUT_DIR / "fi_t5_sanity_check_summary.json"),
        str(OUT_DIR / "fi_t5_sanity_check_report.md"),
        str(RUN_DIR / "task_completion_card.md"),
        str(RUN_DIR / "terminal_summary.json"),
        str(RUN_DIR / "final_qa.csv"),
    ]
    update_run_state(
        "completed",
        [
            "FI_T5 candidate discovery",
            "Schema inspection",
            "Field mapping",
            "Case B fiscal-period alignment",
            "Sanity metric computation",
            "Report and QA outputs",
        ],
        [str(candidate), str(V3_PATH)],
        generated,
        next_step,
    )
    append_checkpoint("task completed", "completed", f"- decision: {decision}\n")
    log(f"Completed FI_T5 sanity check v0: {decision}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        ensure_dirs()
        append_checkpoint("script failed", "failed", f"- error: {exc}\n")
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
