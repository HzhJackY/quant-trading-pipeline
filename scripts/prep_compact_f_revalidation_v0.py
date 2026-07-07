from __future__ import annotations

import csv
import gc
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "compact_f_revalidation_prep_v0"

PANEL_PATH = ROOT / "output" / "transformed_training_panel_v0" / "transformed_training_panel_v0.parquet"
FINAL_FEATURE_CSV = ROOT / "output" / "finalized_revalidation_feature_list_v0" / "finalized_model_feature_list_v0.csv"
FINAL_FEATURE_JSON = ROOT / "output" / "finalized_revalidation_feature_list_v0" / "finalized_model_feature_list_v0.json"
FINAL_SUMMARY = ROOT / "output" / "finalized_revalidation_feature_list_v0" / "finalized_feature_list_summary.json"
FINAL_REPORT = ROOT / "output" / "finalized_revalidation_feature_list_v0" / "finalized_feature_list_report.md"
QA_SUMMARY = ROOT / "output" / "transformed_panel_qa_review_v0" / "transformed_panel_qa_review_summary.json"
QA_FEATURE_CSV = ROOT / "output" / "transformed_panel_qa_review_v0" / "model_feature_list_for_revalidation_v0.csv"
TRANSFORM_SPEC = ROOT / "output" / "factor_transform_planning_v0" / "factor_transform_spec_v0.json"
BUILD_SUMMARY = ROOT / "output" / "transformed_training_panel_v0" / "transformed_training_panel_summary.json"
BUILD_FEATURE_CSV = ROOT / "output" / "transformed_training_panel_v0" / "model_feature_list_v0.csv"

REQUIRED_INPUTS = [
    PANEL_PATH,
    FINAL_FEATURE_CSV,
    FINAL_FEATURE_JSON,
    FINAL_SUMMARY,
    FINAL_REPORT,
    QA_SUMMARY,
    QA_FEATURE_CSV,
    TRANSFORM_SPEC,
    BUILD_SUMMARY,
    BUILD_FEATURE_CSV,
]

SEARCH_TERMS = [
    "Compact-F",
    "Compact_F",
    "compact_f",
    "CompactF",
    "compact",
    "model_config",
    "tournament",
    "full_panel",
]
SEARCH_DIRS = [ROOT, ROOT / "configs", ROOT / "scripts", ROOT / "output"]
READABLE_EXTS = {".json", ".csv", ".txt", ".md", ".yaml", ".yml", ".py", ".toml"}
MAX_CANDIDATE_READ_BYTES = 2_000_000
LABEL_CANDIDATES = [
    "fwd_ret_1m",
    "forward_return_1m",
    "ret_fwd_1m",
    "next_month_return",
    "y",
    "label",
    "excess_return_1m",
    "excess_fwd_ret_1m",
]


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
        "Compact-F revalidation prep did not run because required whitelisted inputs are missing.",
        "",
        "## Missing files",
        "",
    ]
    lines.extend(f"- {p.as_posix()}" for p in missing)
    (OUT_DIR / "missing_input_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        OUT_DIR / "compact_f_revalidation_prep_summary.json",
        {
            "run_timestamp": now_iso(),
            "final_decision": "COMPACT_F_REVALIDATION_PREP_FAIL_BLOCK_RUN",
            "missing_inputs": [str(p) for p in missing],
        },
    )


def first_existing(candidates: list[str], columns: set[str]) -> str | None:
    for item in candidates:
        if item in columns:
            return item
    return None


def limited_filename_search() -> list[dict[str, Any]]:
    terms_lower = [t.lower() for t in SEARCH_TERMS]
    seen: set[Path] = set()
    candidates: list[dict[str, Any]] = []

    def allowed_depth(base: Path, path: Path) -> bool:
        try:
            rel_parts = path.relative_to(base).parts
        except ValueError:
            return False
        if base == ROOT:
            return len(rel_parts) == 1
        if base.name.lower() == "output":
            return len(rel_parts) <= 4
        return len(rel_parts) <= 8

    for base in SEARCH_DIRS:
        if not base.exists():
            continue
        if base == ROOT:
            iterator = [p for p in base.iterdir() if p.is_file()]
        else:
            iterator = base.rglob("*")
        for path in iterator:
            if not path.is_file() or path in seen:
                continue
            if not allowed_depth(base, path):
                continue
            name_lower = path.name.lower()
            if not any(term in name_lower for term in terms_lower):
                continue
            seen.add(path)
            try:
                stat = path.stat()
            except OSError:
                continue
            candidates.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "filename": path.name,
                    "suffix": path.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "readable_small_config": path.suffix.lower() in READABLE_EXTS and stat.st_size <= MAX_CANDIDATE_READ_BYTES,
                    "matched_terms": ";".join(t for t in SEARCH_TERMS if t.lower() in name_lower),
                }
            )
    candidates.sort(key=lambda x: (not x["readable_small_config"], x["size_bytes"], x["path"]))
    return candidates


def read_candidate_text(path: Path) -> str:
    if path.suffix.lower() not in READABLE_EXTS:
        return ""
    try:
        if path.stat().st_size > MAX_CANDIDATE_READ_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def extract_feature_like_list(text: str) -> list[str]:
    features: list[str] = []
    patterns = [
        r'"features"\s*:\s*\[(.*?)\]',
        r'"feature_list"\s*:\s*\[(.*?)\]',
        r"FEATURES\s*=\s*\[(.*?)\]",
        r"feature_cols\s*=\s*\[(.*?)\]",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            chunk = match.group(1)
            features.extend(re.findall(r'["\']([A-Za-z_][A-Za-z0-9_]*)["\']', chunk))
    return sorted(dict.fromkeys(features))


def infer_config_details(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    parsed: list[dict[str, Any]] = []
    for row in candidates:
        if not row["readable_small_config"]:
            continue
        path = ROOT / row["path"]
        text = read_candidate_text(path)
        if not text:
            continue
        features = extract_feature_like_list(text)
        if features or "compact" in text.lower():
            target_match = re.search(r'["\']?(?:target|label|y_col|target_col)["\']?\s*[:=]\s*["\']([^"\']+)["\']', text, re.I)
            model_match = re.search(r'["\']?(?:model_name|name)["\']?\s*[:=]\s*["\']([^"\']*compact[^"\']*)["\']', text, re.I)
            type_match = re.search(r'["\']?(?:model_type|estimator)["\']?\s*[:=]\s*["\']([^"\']+)["\']', text, re.I)
            name_lower = path.name.lower()
            priority = 0
            if features:
                priority += 100
            if "compact_f_config" in name_lower:
                priority += 50
            elif "compact" in name_lower and "config" in name_lower:
                priority += 30
            if path.suffix.lower() == ".json":
                priority += 10
            parsed.append({
                "path": row["path"],
                "model_name": model_match.group(1) if model_match else "Compact-F",
                "feature_list": features,
                "target": target_match.group(1) if target_match else "",
                "model_type": type_match.group(1) if type_match else "",
                "raw_text_checked": True,
                "_priority": priority,
            })
    if parsed:
        parsed.sort(key=lambda x: (-x["_priority"], x["path"]))
        selected = parsed[0]
        selected.pop("_priority", None)
        return selected
    return {"path": "", "model_name": "", "feature_list": [], "target": "", "model_type": "", "raw_text_checked": False}


def make_mapping(compact_features: list[str], finalized_df: pd.DataFrame, panel_columns: set[str]) -> pd.DataFrame:
    default_df = finalized_df[finalized_df["included_by_default"].apply(norm_bool)].copy()
    by_feature = {str(r["feature_name"]): r for _, r in finalized_df.iterrows()}
    default_features = set(default_df["feature_name"].astype(str))
    all_features = set(finalized_df["feature_name"].astype(str))
    rows: list[dict[str, Any]] = []
    transformed_by_source: dict[str, list[str]] = {}
    for _, row in finalized_df.iterrows():
        transformed_by_source.setdefault(str(row.get("source_factor", "")), []).append(str(row["feature_name"]))

    alias_to_source = {
        "ep_neutral_z_rank": "ep_ttm",
        "bp_raw_neutral_z_rank": "bp",
        "sr_roe_neutral_z_rank": "roe_ttm",
        "net_profit_margin_neutral_z_rank": "net_margin",
        "profitgrowth_yoy_neutral_z_rank": "profit_growth_yoy",
        "sr_profitgrowth_yoy_neutral_z_rank": "profit_growth_yoy",
        "revgrowth_yoy_neutral_z_rank": "rev_growth_yoy",
        "sr_revgrowth_yoy_neutral_z_rank": "rev_growth_yoy",
        "debt_ratio_neutral_z_rank": "debt_ratio",
    }

    for feature in compact_features:
        candidate = ""
        status = "NEED_MANUAL_MAPPING"
        note = ""
        feature_key = feature.lower()
        if feature in panel_columns and feature in default_features:
            candidate = feature
            status = "EXACT_MATCH"
        elif feature in all_features:
            candidate = feature
            status = "EXCLUDED_BY_QA"
        elif feature_key == "roe_ttm":
            aliases = [f for f in all_features if f.startswith("roe_parent_ttm_ending_equity")]
            if aliases:
                candidate = aliases[0]
                status = "ALIAS_MATCH" if candidate in default_features else "EXCLUDED_BY_QA"
                note = "ROE alias: roe_ttm maps to roe_parent_ttm_ending_equity transformed columns."
        elif feature_key in alias_to_source:
            source = alias_to_source[feature_key]
            options = transformed_by_source.get(source, [])
            ranked_options = [f for f in options if f.endswith("_rank") and f in default_features]
            z_options = [f for f in options if f.endswith("_z") and f in default_features]
            raw_options = [f for f in options if f in default_features]
            selected = (ranked_options or z_options or raw_options or options)
            if selected:
                candidate = selected[0]
                status = "TRANSFORMED_EQUIVALENT" if candidate in default_features else "EXCLUDED_BY_QA"
                note = "Original Compact-F neutral_z_rank maps to finalized transformed feature; neutralization is not executed in prep."
        elif feature in transformed_by_source:
            options = [f for f in transformed_by_source[feature] if f in default_features]
            if options:
                candidate = options[0]
                status = "TRANSFORMED_EQUIVALENT"
                note = "Original raw factor maps to finalized transformed feature; transform may differ from Compact-F raw input."
            else:
                options = transformed_by_source[feature]
                candidate = options[0] if options else ""
                status = "EXCLUDED_BY_QA" if candidate else "NEED_MANUAL_MAPPING"
        elif feature not in panel_columns:
            status = "MISSING_IN_TRANSFORMED_PANEL"
        meta = by_feature.get(candidate, {})
        rows.append(
            {
                "compact_f_feature": feature,
                "transformed_feature_candidate": candidate,
                "mapping_status": status,
                "transform_type": meta.get("transform_type", ""),
                "review_status": meta.get("review_status", ""),
                "included_by_default": meta.get("included_by_default", False) if candidate else False,
                "notes": note,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        missing_input_report(missing)
        print(f"Missing required inputs: {len(missing)}")
        return 0

    finalized_df = pd.read_csv(FINAL_FEATURE_CSV)
    final_summary = load_json(FINAL_SUMMARY)
    qa_summary = load_json(QA_SUMMARY)
    build_summary = load_json(BUILD_SUMMARY)
    parquet = pq.ParquetFile(PANEL_PATH)
    panel_columns = set(parquet.schema.names)

    finalized_df["included_by_default"] = finalized_df["included_by_default"].apply(norm_bool)
    default_df = finalized_df[finalized_df["included_by_default"]].copy()
    default_features = default_df["feature_name"].astype(str).tolist()
    default_feature_set = set(default_features)

    trading_status_z_in_default = "trading_status_z" in default_feature_set
    constant_in_default = sorted(default_df.loc[default_df["reason"].astype(str).str.contains("constant", case=False, na=False), "feature_name"].astype(str))
    exclude_in_default = sorted(default_df.loc[default_df["review_status"].astype(str).str.upper().isin(["EXCLUDE", "RAW_COMPONENT_ONLY", "RAW_COMPONENT"]), "feature_name"].astype(str))
    audit_terms = ["selected_pit_date", "selected_report_period", "market_cap_trade_date", "audit", "production", "live", "holding", "prediction", "pred_"]
    audit_in_default = sorted(f for f in default_features if any(term in f.lower() for term in audit_terms))

    finalized_check = {
        "finalized_default_feature_count_from_summary": int(final_summary.get("finalized_default_feature_count", -1)),
        "included_by_default_feature_count": len(default_features),
        "trading_status_z_in_default_features": trading_status_z_in_default,
        "constant_features_in_default_features": constant_in_default,
        "audit_metadata_in_default_features": audit_in_default,
        "exclude_or_raw_component_in_default_features": exclude_in_default,
        "feature_list_safe": not any([trading_status_z_in_default, constant_in_default, audit_in_default, exclude_in_default]),
    }
    pd.DataFrame([finalized_check]).to_csv(OUT_DIR / "finalized_feature_list_check.csv", index=False)
    write_json(OUT_DIR / "finalized_feature_list_check.json", finalized_check)

    symbol_col = first_existing(["symbol", "Symbol", "stock_code", "stkcd", "ts_code"], panel_columns)
    month_col = first_existing(["month", "month_end", "trade_month", "month_date", "yyyymm"], panel_columns)
    date_cols = [c for c in [symbol_col, month_col, "selected_pit_date", "market_cap_trade_date"] if c and c in panel_columns]
    label_candidates = [c for c in LABEL_CANDIDATES if c in panel_columns]
    read_cols = sorted(set(date_cols + label_candidates))
    meta_df = pd.read_parquet(PANEL_PATH, columns=read_cols) if read_cols else pd.DataFrame(index=range(parquet.metadata.num_rows))

    rows = int(parquet.metadata.num_rows)
    symbols = int(meta_df[symbol_col].nunique()) if symbol_col and symbol_col in meta_df else int(build_summary.get("symbols", 0) or 0)
    months = int(meta_df[month_col].nunique()) if month_col and month_col in meta_df else int(build_summary.get("months", 0) or 0)
    date_min = str(meta_df[month_col].min()) if month_col and month_col in meta_df else ""
    date_max = str(meta_df[month_col].max()) if month_col and month_col in meta_df else ""
    one_row = bool(build_summary.get("one_row_per_symbol_month"))
    if symbol_col and month_col and symbol_col in meta_df and month_col in meta_df:
        one_row = not meta_df.duplicated([symbol_col, month_col]).any()

    missing_finalized_features = sorted(f for f in default_features if f not in panel_columns)
    extra_panel_feature_columns = sorted(c for c in panel_columns if c.endswith(("_rank", "_z", "_clip", "_raw")) and c not in default_feature_set)
    schema_check = {
        "rows": rows,
        "symbols": symbols,
        "months": months,
        "date_range_min": date_min,
        "date_range_max": date_max,
        "symbol_column": symbol_col,
        "month_column": month_col,
        "one_row_per_symbol_month": one_row,
        "selected_pit_date_violation_count": int(build_summary.get("selected_pit_date_violation_count", 0) or 0),
        "market_cap_trade_date_violation_count": int(build_summary.get("market_cap_trade_date_violation_count", 0) or 0),
        "finalized_features_missing_from_panel": missing_finalized_features,
        "finalized_features_missing_from_panel_count": len(missing_finalized_features),
        "extra_transformed_feature_columns_not_in_default_count": len(extra_panel_feature_columns),
        "extra_transformed_feature_columns_not_in_default_sample": extra_panel_feature_columns[:50],
        "label_candidates_found": label_candidates,
    }
    write_json(OUT_DIR / "transformed_panel_schema_check.json", schema_check)

    candidates = limited_filename_search()
    pd.DataFrame(candidates).to_csv(OUT_DIR / "compact_f_config_candidates.csv", index=False)
    config = infer_config_details(candidates)
    compact_f_config_found = bool(config["path"])
    if not compact_f_config_found:
        (OUT_DIR / "missing_compact_f_config_report.md").write_text(
            "# Missing Compact-F Config Report\n\n"
            "No readable Compact-F config/report with an extractable feature list was found by the allowed lightweight filename search.\n\n"
            "Please provide the Compact-F config/report path before running revalidation.\n",
            encoding="utf-8",
        )

    compact_features = config["feature_list"]
    if compact_f_config_found:
        mapping_df = make_mapping(compact_features, finalized_df, panel_columns)
    else:
        mapping_df = pd.DataFrame(
            columns=[
                "compact_f_feature",
                "transformed_feature_candidate",
                "mapping_status",
                "transform_type",
                "review_status",
                "included_by_default",
                "notes",
            ]
        )
    mapping_df.to_csv(OUT_DIR / "compact_f_feature_mapping.csv", index=False)

    label_rows: list[dict[str, Any]] = []
    for label in label_candidates:
        series = meta_df[label]
        label_rows.append(
            {
                "label_name": label,
                "coverage": float(series.notna().mean()) if len(series) else 0.0,
                "non_null_count": int(series.notna().sum()),
                "date_range_min": date_min,
                "date_range_max": date_max,
                "is_future_return_candidate": any(x in label.lower() for x in ["fwd", "forward", "next", "label", "y"]),
                "leakage_risk": "WATCH" if label in {"y", "label"} else "NONE",
            }
        )
    label_df = pd.DataFrame(label_rows)
    label_df.to_csv(OUT_DIR / "label_readiness_check.csv", index=False)
    label_found = not label_df.empty
    label_selected = str(label_df.iloc[0]["label_name"]) if label_found else ""
    label_coverage = float(label_df.iloc[0]["coverage"]) if label_found else None

    protocol = [
        "# Compact-F Revalidation Protocol Proposal",
        "",
        f"- source_panel_path: {PANEL_PATH.relative_to(ROOT).as_posix()}",
        f"- finalized_feature_list_path: {FINAL_FEATURE_CSV.relative_to(ROOT).as_posix()}",
        f"- model_feature_set: finalized features with included_by_default = true ({len(default_features)} features)",
        f"- target_label: {label_selected or 'TBD via Label Integration v0'}",
        "- train / validation / OOS split proposal: reuse the original Compact-F split once config is supplied; otherwise use chronological monthly split after label integration.",
        "- monthly_frequency: true",
        "- universe: locked CSI800 v3 universe",
        "- GS by default: no",
        "- BP retained: yes",
        "- technical-factor expansion: no unless explicitly requested",
        "- neutralization: no unless separate experiment",
        "- random_seed_policy: fixed seeds recorded in revalidation run config",
        "- output_path_proposal: output/compact_f_revalidation_run_v0/",
        "- safety_checks_before_run: feature list, label coverage, PIT metadata, duplicate keys, leakage columns, config mapping.",
    ]
    (OUT_DIR / "compact_f_revalidation_protocol_proposal.md").write_text("\n".join(protocol) + "\n", encoding="utf-8")

    mapped_count = int(mapping_df["mapping_status"].isin(["EXACT_MATCH", "ALIAS_MATCH", "TRANSFORMED_EQUIVALENT"]).sum()) if not mapping_df.empty else 0
    missing_count = int(mapping_df["mapping_status"].eq("MISSING_IN_TRANSFORMED_PANEL").sum()) if not mapping_df.empty else 0
    manual_count = int(mapping_df["mapping_status"].eq("NEED_MANUAL_MAPPING").sum()) if not mapping_df.empty else 0

    fatal_schema = any(
        [
            not finalized_check["feature_list_safe"],
            len(missing_finalized_features) > 0,
            not one_row,
            schema_check["selected_pit_date_violation_count"] > 0,
            schema_check["market_cap_trade_date_violation_count"] > 0,
            bool(final_summary.get("leakage_detected")),
            bool(final_summary.get("severe_leakage_detected")),
        ]
    )
    if fatal_schema:
        final_decision = "COMPACT_F_REVALIDATION_PREP_FAIL_BLOCK_RUN"
        recommended_next_step = "Fix feature-list or transformed-panel schema issues before revalidation."
    elif not compact_f_config_found or len(compact_features) == 0 or manual_count or missing_count:
        final_decision = "COMPACT_F_REVALIDATION_PREP_WATCH_CONFIG_OR_MAPPING_REVIEW_REQUIRED"
        recommended_next_step = "Provide or confirm Compact-F config and feature mapping before revalidation."
    elif not label_found:
        final_decision = "COMPACT_F_REVALIDATION_PREP_READY_FOR_LABEL_INTEGRATION"
        recommended_next_step = "Label Integration v0."
    else:
        final_decision = "COMPACT_F_REVALIDATION_PREP_READY_FOR_REVALIDATION_RUN"
        recommended_next_step = "Compact-F Revalidation Run v0."

    summary = {
        "run_timestamp": now_iso(),
        "transformed_panel_used": str(PANEL_PATH.relative_to(ROOT)),
        "finalized_feature_list_used": str(FINAL_FEATURE_CSV.relative_to(ROOT)),
        "rows": rows,
        "symbols": symbols,
        "months": months,
        "finalized_default_feature_count": len(default_features),
        "trading_status_z_in_default_features": trading_status_z_in_default,
        "constant_features_in_default_features": constant_in_default,
        "audit_metadata_in_default_features": audit_in_default,
        "finalized_features_missing_from_panel_count": len(missing_finalized_features),
        "compact_f_config_found": compact_f_config_found,
        "compact_f_config_used": config["path"],
        "compact_f_feature_count": len(compact_features),
        "compact_f_features_mapped_count": mapped_count,
        "compact_f_features_missing_count": missing_count,
        "compact_f_features_need_manual_mapping_count": manual_count,
        "label_found": label_found,
        "label_candidates": label_candidates,
        "label_selected": label_selected,
        "label_coverage": label_coverage,
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
    write_json(OUT_DIR / "compact_f_revalidation_prep_summary.json", summary)

    inputs = [
        PANEL_PATH,
        FINAL_FEATURE_CSV,
        FINAL_FEATURE_JSON,
        FINAL_SUMMARY,
        FINAL_REPORT,
        QA_SUMMARY,
        QA_FEATURE_CSV,
        TRANSFORM_SPEC,
        BUILD_SUMMARY,
        BUILD_FEATURE_CSV,
    ]
    report = [
        "# Compact-F Revalidation Prep v0",
        "",
        "## 1. Scope",
        "",
        "This run only prepares and audits configuration for Compact-F revalidation. It does not train, backtest, calculate IC, tune parameters, modify production, modify v3, or modify the transformed panel.",
        "",
        "## 2. Inputs",
        "",
        *[f"- {p.relative_to(ROOT).as_posix()}" for p in inputs],
        "",
        "## 3. Finalized Feature List Check",
        "",
        f"- Default feature count: {len(default_features)}",
        f"- trading_status_z in default: {trading_status_z_in_default}",
        f"- Constant features in default: {len(constant_in_default)}",
        f"- Audit metadata in default: {len(audit_in_default)}",
        "",
        "## 4. Transformed Panel Schema Check",
        "",
        f"- Rows / symbols / months: {rows} / {symbols} / {months}",
        f"- Date range: {date_min} to {date_max}",
        f"- One row per symbol-month: {one_row}",
        f"- Finalized features missing from panel: {len(missing_finalized_features)}",
        "",
        "## 5. Compact-F Config Discovery",
        "",
        f"- Config found: {compact_f_config_found}",
        f"- Config used: {config['path'] or 'none'}",
        "",
        "## 6. Compact-F Feature Mapping",
        "",
        f"- Compact-F feature count: {len(compact_features)}",
        f"- Mapped: {mapped_count}",
        f"- Missing: {missing_count}",
        f"- Need manual mapping: {manual_count}",
        "",
        "## 7. Label Readiness",
        "",
        f"- Label found: {label_found}",
        f"- Label candidates: {', '.join(label_candidates) if label_candidates else 'none'}",
        f"- Selected label: {label_selected or 'none'}",
        "",
        "## 8. Revalidation Protocol Proposal",
        "",
        "See compact_f_revalidation_protocol_proposal.md.",
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
    (OUT_DIR / "compact_f_revalidation_prep_report.md").write_text("\n".join(report), encoding="utf-8")

    task_card = [
        "# Task Completion Card",
        "",
        "- task_name: Compact-F Revalidation Prep v0",
        f"- completed_at: {now_iso()}",
        f"- final_decision: {final_decision}",
        f"- output_dir: {OUT_DIR.relative_to(ROOT).as_posix()}",
    ]
    (OUT_DIR / "task_completion_card.md").write_text("\n".join(task_card) + "\n", encoding="utf-8")
    write_json(
        OUT_DIR / "terminal_summary.json",
        {
            "script": "scripts/prep_compact_f_revalidation_v0.py",
            "status": "completed",
            "stdout_log": "output/_agent_runs/compact_f_revalidation_prep_v0/run_stdout.txt",
            "stderr_log": "output/_agent_runs/compact_f_revalidation_prep_v0/run_stderr.txt",
            "final_decision": final_decision,
        },
    )
    pd.DataFrame([summary]).to_csv(OUT_DIR / "final_qa.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    del finalized_df, default_df, meta_df
    gc.collect()
    print(f"final_decision={final_decision}")
    print(f"output_dir={OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
