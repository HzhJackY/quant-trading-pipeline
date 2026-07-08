from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "compact_f_missing_feature_label_resolution_v0"

TRANSFORMED_PANEL = ROOT / "output" / "transformed_training_panel_v0" / "transformed_training_panel_v0.parquet"
FINAL_FEATURE_CSV = ROOT / "output" / "finalized_revalidation_feature_list_v0" / "finalized_model_feature_list_v0.csv"
FINAL_FEATURE_JSON = ROOT / "output" / "finalized_revalidation_feature_list_v0" / "finalized_model_feature_list_v0.json"
FINAL_SUMMARY = ROOT / "output" / "finalized_revalidation_feature_list_v0" / "finalized_feature_list_summary.json"
PREP_SUMMARY = ROOT / "output" / "compact_f_revalidation_prep_v0" / "compact_f_revalidation_prep_summary.json"
PREP_MAPPING = ROOT / "output" / "compact_f_revalidation_prep_v0" / "compact_f_feature_mapping.csv"
PREP_REPORT = ROOT / "output" / "compact_f_revalidation_prep_v0" / "compact_f_revalidation_prep_report.md"
PREP_CANDIDATES = ROOT / "output" / "compact_f_revalidation_prep_v0" / "compact_f_config_candidates.csv"
BUILD_SUMMARY = ROOT / "output" / "transformed_training_panel_v0" / "transformed_training_panel_summary.json"
QA_SUMMARY = ROOT / "output" / "transformed_panel_qa_review_v0" / "transformed_panel_qa_review_summary.json"
V3_PANEL = ROOT / "output" / "csmar_pit_clean_core_financial_factors_v3" / "pit_clean_core_financial_factors_monthly_v3.parquet"

REQUIRED_INPUTS = [
    TRANSFORMED_PANEL,
    FINAL_FEATURE_CSV,
    FINAL_FEATURE_JSON,
    FINAL_SUMMARY,
    PREP_SUMMARY,
    PREP_MAPPING,
    PREP_REPORT,
    PREP_CANDIDATES,
    BUILD_SUMMARY,
    QA_SUMMARY,
    V3_PANEL,
]

LABEL_SEARCH_TERMS = ["label", "forward", "fwd", "return", "ret", "training_panel", "panel", "target", "excess"]
LABEL_COLUMN_TERMS = ["label", "fwd", "forward", "next", "return", "ret", "target", "excess"]
SYMBOL_NAMES = {"symbol", "stock_code", "stkcd", "ts_code", "ticker", "secu_code"}
MONTH_NAMES = {"month", "month_end", "trade_month", "month_date", "yyyymm", "date"}
RAW_DAILY_MARKERS = ["daily", "raw_daily", "stock_daily", "trading_day", "price_daily"]

MISSING_FEATURE_RULES = {
    "CFO_to_Earnings_neutral_z_rank": {
        "required": ["operating_cash_flow or CFO", "net_profit or earnings"],
        "component_groups": [["operating_cash_flow", "cash_flow", "cfo"], ["net_profit", "earnings", "profit_parent"]],
        "missing_status": "REQUIRES_CASH_FLOW_TABLE",
        "required_source": "cash flow table",
    },
    "Current_Ratio_neutral_z_rank": {
        "required": ["current_assets", "current_liabilities"],
        "component_groups": [["current_assets"], ["current_liabilities"]],
        "missing_status": "REQUIRES_BALANCE_SHEET_DETAIL",
        "required_source": "balance sheet detail",
    },
    "Quick_Ratio_neutral_z_rank": {
        "required": ["current_assets", "inventories", "current_liabilities"],
        "component_groups": [["current_assets"], ["inventory", "inventories"], ["current_liabilities"]],
        "missing_status": "REQUIRES_BALANCE_SHEET_DETAIL",
        "required_source": "balance sheet detail",
    },
    "EPS_YoY_neutral_z_rank": {
        "required": ["EPS current and lag, or net_profit plus shares"],
        "component_groups": [["eps"], ["shares", "share_base", "total_share"]],
        "missing_status": "REQUIRES_EPS_OR_SHARE_BASE",
        "required_source": "EPS/share-base fields",
    },
    "Equity_Multiplier_neutral_z_rank": {
        "required": ["total_assets", "equity_parent or total_equity"],
        "component_groups": [["total_assets", "assets_total"], ["equity_parent", "total_equity", "shareholders_equity"]],
        "missing_status": "NOT_AVAILABLE_CURRENTLY",
        "required_source": "balance sheet total assets and equity",
    },
    "Operating_Margin_neutral_z_rank": {
        "required": ["operating_profit", "revenue"],
        "component_groups": [["operating_profit"], ["revenue", "operating_revenue"]],
        "missing_status": "REQUIRES_OPERATING_PROFIT_FIELD",
        "required_source": "income statement operating profit field",
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def norm_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def missing_input_report(missing: list[Path]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Missing Input Report",
        "",
        "Compact-F missing feature and label source resolution did not run because required whitelisted inputs are missing.",
        "",
        "## Missing files",
        "",
    ]
    lines.extend(f"- {p.as_posix()}" for p in missing)
    (OUT_DIR / "missing_input_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        OUT_DIR / "compact_f_missing_feature_label_resolution_summary.json",
        {
            "run_timestamp": now_iso(),
            "final_decision": "COMPACT_F_RESOLUTION_FAIL_BLOCK_REVALIDATION",
            "missing_inputs": [str(p) for p in missing],
        },
    )


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema.names


def matching_components(columns: list[str], keywords: list[str]) -> list[str]:
    out = []
    for col in columns:
        low = col.lower()
        if any(key.lower() in low for key in keywords):
            out.append(col)
    return sorted(out)


def assess_constructibility(feature: str, v3_cols: list[str], transformed_cols: list[str]) -> dict[str, Any]:
    rule = MISSING_FEATURE_RULES[feature]
    all_cols = sorted(set(v3_cols) | set(transformed_cols))
    available_by_group = [matching_components(all_cols, group) for group in rule["component_groups"]]
    available_flat = sorted({item for group in available_by_group for item in group})
    all_groups_available = all(bool(group) for group in available_by_group)

    if all_groups_available:
        status = "CONSTRUCTIBLE_FROM_V3_COMPONENTS"
        proposed_action = "ADD_DERIVED_FEATURE_IN_FUTURE_PANEL"
        reason = "Required component groups are present in v3/transformed schema; derivation should be specified in a future panel version."
        can_core = False
        can_full = True
    else:
        status = rule["missing_status"]
        proposed_action = "REQUIRE_ADDITIONAL_DATA_BEFORE_FULL_REPLICATION"
        reason = f"Missing one or more required component groups; requires {rule['required_source']}."
        can_core = False
        can_full = False

    if status == "NOT_AVAILABLE_CURRENTLY":
        proposed_action = "EXCLUDE_FROM_COMPACT_F_V3_CORE"
    return {
        "compact_f_feature": feature,
        "required_components": "; ".join(rule["required"]),
        "available_components": "; ".join(available_flat),
        "constructibility_status": status,
        "proposed_action": proposed_action,
        "reason": reason,
        "can_include_in_core_revalidation": can_core,
        "can_include_in_full_revalidation_later": can_full,
    }


def label_filename_candidates() -> list[Path]:
    base = ROOT / "output"
    terms = [term.lower() for term in LABEL_SEARCH_TERMS]
    candidates: list[Path] = []
    if not base.exists():
        return candidates
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(marker in str(path).lower() for marker in RAW_DAILY_MARKERS):
            continue
        if any(term in name for term in terms):
            candidates.append(path)
    candidates.sort(key=lambda p: (p.suffix.lower() not in {".parquet", ".csv", ".json"}, p.stat().st_size if p.exists() else 0, str(p)))
    return candidates[:200]


def inspect_label_candidate(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    size = path.stat().st_size
    columns: list[str] = []
    date_range = ""
    symbols_sample_count = 0
    rows_sample_count = 0
    risk_note = ""

    try:
        if suffix == ".parquet":
            pf = pq.ParquetFile(path)
            columns = pf.schema.names
            rows_sample_count = min(int(pf.metadata.num_rows), 1000)
            sample_cols = choose_sample_columns(columns)
            if sample_cols:
                table = pf.read(columns=sample_cols).slice(0, min(1000, pf.metadata.num_rows))
                sample = table.to_pandas()
                date_col = first_by_set(sample.columns, MONTH_NAMES)
                symbol_col = first_by_set(sample.columns, SYMBOL_NAMES)
                if date_col:
                    date_range = f"{sample[date_col].min()} to {sample[date_col].max()}"
                if symbol_col:
                    symbols_sample_count = int(sample[symbol_col].nunique())
                del sample, table
        elif suffix == ".csv":
            if size > 50_000_000:
                risk_note = "large_csv_schema_only"
                sample = pd.read_csv(path, nrows=0)
            else:
                sample = pd.read_csv(path, nrows=1000)
                rows_sample_count = len(sample)
                date_col = first_by_set(sample.columns, MONTH_NAMES)
                symbol_col = first_by_set(sample.columns, SYMBOL_NAMES)
                if date_col and rows_sample_count:
                    date_range = f"{sample[date_col].min()} to {sample[date_col].max()}"
                if symbol_col and rows_sample_count:
                    symbols_sample_count = int(sample[symbol_col].nunique())
            columns = list(sample.columns)
            del sample
        elif suffix == ".json":
            text = path.read_text(encoding="utf-8", errors="replace")[:200_000]
            columns = [key for key in LABEL_COLUMN_TERMS if key in text.lower()]
        else:
            if size <= 2_000_000:
                text = path.read_text(encoding="utf-8", errors="replace")[:200_000]
                columns = [key for key in LABEL_COLUMN_TERMS if key in text.lower()]
    except Exception as exc:
        risk_note = f"inspect_error:{type(exc).__name__}"

    label_cols = [c for c in columns if is_label_column(c)]
    has_symbol = any(c.lower() in SYMBOL_NAMES for c in columns)
    has_month = any(c.lower() in MONTH_NAMES for c in columns)
    strict_label_cols = [c for c in label_cols if is_strict_forward_label_column(c)]
    likely = bool(has_symbol and has_month and strict_label_cols)
    if any(marker in str(path).lower() for marker in ["daily", "raw"]):
        likely = False
        risk_note = (risk_note + "; " if risk_note else "") + "raw_or_daily_like_path_skipped_as_primary"
    return {
        "candidate_path": str(path.relative_to(ROOT)),
        "file_type": suffix.lstrip("."),
        "size": size,
        "columns": ";".join(columns[:120]),
        "has_symbol": has_symbol,
        "has_month_end": has_month,
        "label_candidate_columns": ";".join(label_cols),
        "date_range": date_range,
        "symbols_sample_count": symbols_sample_count,
        "rows_sample_count": rows_sample_count,
        "likely_label_source": likely,
        "risk_note": risk_note,
    }


def first_by_set(columns: Any, names: set[str]) -> str | None:
    for col in columns:
        if str(col).lower() in names:
            return str(col)
    return None


def choose_sample_columns(columns: list[str]) -> list[str]:
    chosen: list[str] = []
    for col in columns:
        low = col.lower()
        if low in SYMBOL_NAMES or low in MONTH_NAMES or is_label_column(col):
            chosen.append(col)
    return chosen[:30]


def is_label_column(column: str) -> bool:
    low = str(column).lower()
    return low == "y" or any(term in low for term in LABEL_COLUMN_TERMS)


def is_strict_forward_label_column(column: str) -> bool:
    low = str(column).lower()
    return low == "y" or any(term in low for term in ["label", "fwd", "forward", "next", "target", "excess"])


def pick_label_recommendation(candidates_df: pd.DataFrame) -> tuple[str, str, str, list[str], str]:
    if candidates_df.empty or not candidates_df["likely_label_source"].any():
        return (
            "LABEL_SOURCE_NOT_FOUND_NEED_LABEL_INTEGRATION",
            "",
            "",
            ["symbol", "month"],
            "No reliable existing label source was confirmed by filename-level output search.",
        )
    likely = candidates_df[candidates_df["likely_label_source"]].copy()
    likely["priority"] = likely["candidate_path"].astype(str).str.contains("training_panel|full|target|label", case=False, regex=True).astype(int)
    likely = likely.sort_values(["priority", "size"], ascending=[False, False])
    row = likely.iloc[0]
    label_cols = [c for c in str(row["label_candidate_columns"]).split(";") if c]
    preferred = ""
    for name in ["fwd_ret_1m", "forward_return_1m", "ret_fwd_1m", "next_month_return", "excess_fwd_ret_1m", "label", "y"]:
        if name in label_cols:
            preferred = name
            break
    if not preferred and label_cols:
        preferred = label_cols[0]
    return (
        "LABEL_SOURCE_CANDIDATE_FOUND_NEEDS_USER_CONFIRMATION",
        str(row["candidate_path"]),
        preferred,
        ["symbol", "month_end"],
        "Candidate found from existing output artifact; user should confirm it is the verified label source before integration.",
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        missing_input_report(missing)
        print(f"Missing required inputs: {len(missing)}")
        return 0

    prep_summary = load_json(PREP_SUMMARY)
    final_summary = load_json(FINAL_SUMMARY)
    build_summary = load_json(BUILD_SUMMARY)
    qa_summary = load_json(QA_SUMMARY)
    mapping = pd.read_csv(PREP_MAPPING)
    finalized = pd.read_csv(FINAL_FEATURE_CSV)
    finalized["included_by_default"] = finalized["included_by_default"].apply(norm_bool)
    transformed_cols = parquet_columns(TRANSFORMED_PANEL)
    v3_cols = parquet_columns(V3_PANEL)

    mapping_recheck = mapping.copy()
    mapping_recheck["is_mapped"] = mapping_recheck["mapping_status"].isin(["EXACT_MATCH", "ALIAS_MATCH", "TRANSFORMED_EQUIVALENT"])
    mapping_recheck["is_missing"] = mapping_recheck["mapping_status"].eq("MISSING_IN_TRANSFORMED_PANEL")
    mapping_recheck.to_csv(OUT_DIR / "compact_f_mapping_recheck.csv", index=False)

    compact_f_original_feature_count = int(len(mapping_recheck))
    mapped_feature_count = int(mapping_recheck["is_mapped"].sum())
    missing_rows = mapping_recheck[mapping_recheck["is_missing"]].copy()
    missing_features = missing_rows["compact_f_feature"].astype(str).tolist()

    construct_rows: list[dict[str, Any]] = []
    for _, row in mapping_recheck.iterrows():
        feature = str(row["compact_f_feature"])
        if bool(row["is_mapped"]):
            construct_rows.append(
                {
                    "compact_f_feature": feature,
                    "required_components": "",
                    "available_components": str(row.get("transformed_feature_candidate", "")),
                    "constructibility_status": "ALREADY_MAPPED",
                    "proposed_action": "INCLUDE_CURRENT_MAPPING",
                    "reason": "Already mapped to finalized transformed feature.",
                    "can_include_in_core_revalidation": True,
                    "can_include_in_full_revalidation_later": True,
                }
            )
        elif feature in MISSING_FEATURE_RULES:
            construct_rows.append(assess_constructibility(feature, v3_cols, transformed_cols))
        else:
            construct_rows.append(
                {
                    "compact_f_feature": feature,
                    "required_components": "",
                    "available_components": "",
                    "constructibility_status": "NEED_MANUAL_DEFINITION_REVIEW",
                    "proposed_action": "NEED_MANUAL_REVIEW",
                    "reason": "No rule defined for this missing feature.",
                    "can_include_in_core_revalidation": False,
                    "can_include_in_full_revalidation_later": False,
                }
            )
    construct_df = pd.DataFrame(construct_rows)
    construct_df.to_csv(OUT_DIR / "missing_feature_constructibility.csv", index=False)

    default_features = set(finalized.loc[finalized["included_by_default"], "feature_name"].astype(str))
    core_rows: list[dict[str, Any]] = []
    for _, row in mapping_recheck[mapping_recheck["is_mapped"]].iterrows():
        transformed = str(row["transformed_feature_candidate"])
        included = transformed in default_features and transformed != "trading_status_z"
        core_rows.append(
            {
                "compact_f_original_feature": row["compact_f_feature"],
                "transformed_feature": transformed,
                "transform_type": row.get("transform_type", ""),
                "review_status": row.get("review_status", ""),
                "included": included,
                "reason": "safe mapped feature included in Compact-F-v3-core" if included else "mapped feature not included by finalized default list",
                "limitation_note": "Compact-F-v3-core uses transformed rank feature, not original neutral_z_rank full replication.",
            }
        )
    core_df = pd.DataFrame(core_rows)
    core_df.to_csv(OUT_DIR / "compact_f_v3_core_feature_set.csv", index=False)
    write_json(OUT_DIR / "compact_f_v3_core_feature_set.json", core_df.to_dict(orient="records"))

    gap_rows = []
    for feature in missing_features:
        info = construct_df[construct_df["compact_f_feature"].eq(feature)].iloc[0]
        status = str(info["constructibility_status"])
        severity = "MEDIUM" if status == "CONSTRUCTIBLE_FROM_V3_COMPONENTS" else "HIGH"
        gap_rows.append(
            {
                "missing_feature": feature,
                "reason_missing": info["reason"],
                "required_data_source": MISSING_FEATURE_RULES.get(feature, {}).get("required_source", "manual definition"),
                "severity": severity,
                "recommended_future_action": info["proposed_action"],
            }
        )
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(OUT_DIR / "compact_f_v3_full_replication_gap.csv", index=False)

    label_candidates = [inspect_label_candidate(path) for path in label_filename_candidates()]
    label_candidates_df = pd.DataFrame(label_candidates)
    if label_candidates_df.empty:
        label_candidates_df = pd.DataFrame(
            columns=[
                "candidate_path",
                "file_type",
                "size",
                "columns",
                "has_symbol",
                "has_month_end",
                "label_candidate_columns",
                "date_range",
                "symbols_sample_count",
                "rows_sample_count",
                "likely_label_source",
                "risk_note",
            ]
        )
    label_candidates_df.to_csv(OUT_DIR / "label_source_candidates.csv", index=False)
    label_status, label_source, label_column, join_keys, label_note = pick_label_recommendation(label_candidates_df)

    rec_lines = [
        "# Label Source Recommendation",
        "",
        f"- recommendation_status: {label_status}",
        f"- recommended_label_source: {label_source or 'none'}",
        f"- recommended_label_column: {label_column or 'none'}",
        f"- recommended_join_key: {', '.join(join_keys)}",
        f"- risk_note: {label_note}",
        "",
        "Next step: Label Integration v0 only after the label source is confirmed. Do not recompute labels from raw daily data in this task.",
    ]
    (OUT_DIR / "label_source_recommendation.md").write_text("\n".join(rec_lines) + "\n", encoding="utf-8")

    constructible_missing_count = int(construct_df["constructibility_status"].eq("CONSTRUCTIBLE_FROM_V3_COMPONENTS").sum())
    requires_additional_data_count = int(
        construct_df["constructibility_status"].isin(
            [
                "REQUIRES_CASH_FLOW_TABLE",
                "REQUIRES_BALANCE_SHEET_DETAIL",
                "REQUIRES_EPS_OR_SHARE_BASE",
                "REQUIRES_OPERATING_PROFIT_FIELD",
                "NOT_AVAILABLE_CURRENTLY",
                "NEED_MANUAL_DEFINITION_REVIEW",
            ]
        ).sum()
    )
    compact_f_v3_core_feature_count = int(core_df["included"].sum()) if not core_df.empty else 0
    full_replication_gap_count = int(len(gap_df))

    leakage_or_schema_fail = any(
        [
            bool(final_summary.get("leakage_detected")),
            bool(final_summary.get("severe_leakage_detected")),
            bool(qa_summary.get("leakage_detected")),
            bool(qa_summary.get("severe_leakage_detected")),
            int(build_summary.get("selected_pit_date_violation_count", 0) or 0) > 0,
            int(build_summary.get("duplicate_symbol_month_count", 0) or 0) > 0,
            compact_f_original_feature_count != 15,
            mapped_feature_count != 9,
            len(missing_features) != 6,
        ]
    )
    if leakage_or_schema_fail:
        final_decision = "COMPACT_F_RESOLUTION_FAIL_BLOCK_REVALIDATION"
        recommended_next_step = "Fix feature mapping/schema issue before revalidation."
    elif label_status == "LABEL_SOURCE_CANDIDATE_FOUND_NEEDS_USER_CONFIRMATION":
        final_decision = "COMPACT_F_RESOLUTION_WATCH_USER_CONFIRMATION_REQUIRED"
        recommended_next_step = "User confirms label source, then run Label Integration v0 for Compact-F-v3-core."
    else:
        final_decision = "COMPACT_F_CORE_FEATURE_SET_READY_LABEL_INTEGRATION_REQUIRED"
        recommended_next_step = "Label Integration v0 is required; user may need to provide or confirm an existing verified label source."

    summary = {
        "run_timestamp": now_iso(),
        "compact_f_original_feature_count": compact_f_original_feature_count,
        "mapped_feature_count": mapped_feature_count,
        "missing_feature_count": len(missing_features),
        "compact_f_v3_core_feature_count": compact_f_v3_core_feature_count,
        "full_replication_gap_count": full_replication_gap_count,
        "constructible_missing_features_count": constructible_missing_count,
        "requires_additional_data_count": requires_additional_data_count,
        "label_source_status": label_status,
        "recommended_label_source": label_source,
        "recommended_label_column": label_column,
        "label_join_keys": join_keys,
        "production_modified": False,
        "v3_modified": False,
        "transformed_panel_modified": False,
        "training_run": False,
        "backtest_run": False,
        "ic_calculated": False,
        "neutralization_executed": False,
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    write_json(OUT_DIR / "compact_f_missing_feature_label_resolution_summary.json", summary)

    construct_section = [
        f"- {row.compact_f_feature}: {row.constructibility_status} ({row.proposed_action})"
        for row in construct_df[construct_df["constructibility_status"].ne("ALREADY_MAPPED")].itertuples()
    ]
    label_candidate_count = int(label_candidates_df["likely_label_source"].sum()) if "likely_label_source" in label_candidates_df else 0
    report = [
        "# Compact-F Missing Feature & Label Source Resolution v0",
        "",
        "## 1. Scope",
        "",
        "This run only resolves missing Compact-F features and discovers label source candidates. It does not train, backtest, calculate IC, or modify production, v3, or the transformed panel.",
        "",
        "## 2. Previous Prep Result",
        "",
        f"- Previous decision: {prep_summary.get('final_decision')}",
        f"- Compact-F features: {compact_f_original_feature_count}",
        f"- Mapped / missing: {mapped_feature_count} / {len(missing_features)}",
        "",
        "## 3. Missing Feature Constructibility",
        "",
        *(construct_section or ["- None"]),
        "",
        "## 4. Compact-F-v3-core Feature Set",
        "",
        f"- Core feature count: {compact_f_v3_core_feature_count}",
        "- This is not a full Compact-F replication; it is a Compact-F-v3-core baseline on finalized transformed panel features.",
        "",
        "## 5. Full Replication Gap",
        "",
        f"- Full replication gap count: {full_replication_gap_count}",
        "- Missing features require additional component data or future derived-feature definitions before full replication.",
        "",
        "## 6. Label Source Discovery",
        "",
        f"- Likely label source candidates: {label_candidate_count}",
        "",
        "## 7. Label Source Recommendation",
        "",
        f"- Status: {label_status}",
        f"- Recommended source: {label_source or 'none'}",
        f"- Recommended label column: {label_column or 'none'}",
        "",
        "## 8. Decision",
        "",
        final_decision,
        "",
        "## 9. Recommended Next Step",
        "",
        recommended_next_step,
        "",
    ]
    (OUT_DIR / "compact_f_missing_feature_label_resolution_report.md").write_text("\n".join(report), encoding="utf-8")

    task_card = [
        "# Task Completion Card",
        "",
        "- task_name: Compact-F Missing Feature & Label Source Resolution v0",
        f"- completed_at: {now_iso()}",
        f"- final_decision: {final_decision}",
        f"- output_dir: {OUT_DIR.relative_to(ROOT).as_posix()}",
    ]
    (OUT_DIR / "task_completion_card.md").write_text("\n".join(task_card) + "\n", encoding="utf-8")
    write_json(
        OUT_DIR / "terminal_summary.json",
        {
            "script": "scripts/resolve_compact_f_missing_features_and_label_v0.py",
            "status": "completed",
            "stdout_log": "output/_agent_runs/compact_f_missing_feature_label_resolution_v0/run_stdout.txt",
            "stderr_log": "output/_agent_runs/compact_f_missing_feature_label_resolution_v0/run_stderr.txt",
            "final_decision": final_decision,
        },
    )
    pd.DataFrame([summary]).to_csv(OUT_DIR / "final_qa.csv", index=False)

    del mapping, finalized, construct_df, core_df, gap_df, label_candidates_df
    gc.collect()
    print(f"final_decision={final_decision}")
    print(f"output_dir={OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
