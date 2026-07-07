from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "ER_Announcement Supplemental Industry Source Suitability Audit v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "er_announcement_industry_source_audit_v0"
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

ER_XLSX = ROOT / "data" / "csmar_exports" / "ER_Announcement.xlsx"
SCORE_PANEL = ROOT / "output" / "simple_baseline_score_run_v0" / "simple_baseline_score_panel_v0.parquet"
TRD_CO_SOURCE = ROOT / "output" / "trd_co_static_industry_neutral_score_run_v1" / "cleaned_trd_co_static_industry_source.csv"

OPTIONAL_INPUTS = {
    "trd_co_join_forensics_summary": ROOT / "output" / "trd_co_static_industry_join_forensics_v0" / "trd_co_static_industry_join_forensics_summary.json",
    "missing_symbol_profile": ROOT / "output" / "trd_co_static_industry_join_forensics_v0" / "missing_symbol_profile.csv",
    "missing_industry_symbol_master": ROOT / "output" / "historical_industry_source_gap_resolution_v0" / "missing_industry_symbol_master.csv",
    "missing_symbols_for_industry_download": ROOT / "output" / "historical_industry_source_gap_resolution_v0" / "missing_symbols_for_industry_download.csv",
}

ER_USECOLS = [
    "AnnouncementID",
    "InstitutionID",
    "Symbol",
    "SecurityID",
    "ShortName",
    "DeclareDate",
    "ClassID",
    "ClassName",
    "IndustryCode",
    "IndustryName",
    "IndustryCode1",
    "IndustryName1",
]


def write_run_state(status: str, details: dict) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# RUN_STATE",
        f"",
        f"- task_name: {TASK_NAME}",
        f"- status: {status}",
        f"- updated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- output_dir: {OUT_DIR}",
        f"- run_dir: {RUN_DIR}",
        f"",
        "## Details",
        "```json",
        json.dumps(details, ensure_ascii=False, indent=2, default=str),
        "```",
    ]
    text = "\n".join(lines) + "\n"
    (RUN_DIR / "RUN_STATE.md").write_text(text, encoding="utf-8")
    (OUT_DIR / "RUN_STATE.md").write_text(text, encoding="utf-8")


def norm_symbol(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.extract(r"(\d{6})", expand=False)


def non_empty(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.notna() & (s != "") & (s.str.lower() != "nan")


def first_non_null(series: pd.Series):
    s = series[non_empty(series)]
    return None if s.empty else s.iloc[0]


def dominant_value(group: pd.DataFrame, code_col: str, name_col: str) -> pd.Series:
    valid = group[non_empty(group[code_col])].copy()
    if valid.empty:
        return pd.Series(
            {
                "dominant_code": pd.NA,
                "dominant_name": pd.NA,
                "dominant_share": 0.0,
                "unique_code_count": 0,
                "unique_name_count": int(group.loc[non_empty(group[name_col]), name_col].nunique()),
            }
        )
    counts = valid.groupby(code_col, dropna=True).size().sort_values(ascending=False)
    dom_code = counts.index[0]
    dom_rows = valid[valid[code_col] == dom_code]
    return pd.Series(
        {
            "dominant_code": dom_code,
            "dominant_name": first_non_null(dom_rows[name_col]),
            "dominant_share": float(counts.iloc[0] / len(valid)),
            "unique_code_count": int(valid[code_col].nunique()),
            "unique_name_count": int(valid.loc[non_empty(valid[name_col]), name_col].nunique()),
        }
    )


def find_col(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_score_symbols() -> tuple[pd.DataFrame, set[str]]:
    score = pd.read_parquet(SCORE_PANEL, columns=["symbol", "month_end"])
    score["symbol"] = norm_symbol(score["symbol"])
    score = score[score["symbol"].notna()].copy()
    symbols = set(score["symbol"].dropna().unique().tolist())
    return score, symbols


def load_trd_co_symbols(score_symbols: set[str]) -> tuple[pd.DataFrame, set[str]]:
    header = pd.read_csv(TRD_CO_SOURCE, nrows=0)
    cols = header.columns.tolist()
    symbol_col = find_col(cols, ["symbol", "Symbol", "Stkcd", "stkcd", "证券代码"])
    code_col = find_col(cols, ["IndcdZX", "indcdzx", "primary_industry_code", "IndustryCode1", "IndustryCode", "industry_code"])
    name_col = find_col(cols, ["IndnmeZX", "indnmezx", "primary_industry_name", "IndustryName1", "IndustryName", "industry_name"])
    usecols = [c for c in [symbol_col, code_col, name_col] if c is not None]
    if symbol_col is None:
        raise ValueError(f"Cannot identify symbol column in {TRD_CO_SOURCE}")
    trd = pd.read_csv(TRD_CO_SOURCE, dtype=str, usecols=usecols)
    trd["symbol"] = norm_symbol(trd[symbol_col])
    if code_col is not None:
        trd["_industry_code"] = trd[code_col].astype("string").str.strip()
    else:
        trd["_industry_code"] = pd.NA
    if name_col is not None:
        trd["_industry_name"] = trd[name_col].astype("string").str.strip()
    else:
        trd["_industry_name"] = pd.NA
    covered = set(
        trd.loc[
            trd["symbol"].isin(score_symbols) & (non_empty(trd["_industry_code"]) | non_empty(trd["_industry_name"])),
            "symbol",
        ]
        .dropna()
        .unique()
        .tolist()
    )
    return trd, covered


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_run_state("running", {"step": "start"})

    required = {
        "er_xlsx": ER_XLSX,
        "score_panel": SCORE_PANEL,
        "trd_co_source": TRD_CO_SOURCE,
    }
    prereq = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "required_inputs": {k: str(v) for k, v in required.items()},
        "required_exists": {k: v.exists() for k, v in required.items()},
        "optional_inputs": {k: str(v) for k, v in OPTIONAL_INPUTS.items()},
        "optional_exists": {k: v.exists() for k, v in OPTIONAL_INPUTS.items()},
    }
    prereq["prerequisites_passed"] = all(prereq["required_exists"].values())
    (OUT_DIR / "er_announcement_audit_prerequisite_check.json").write_text(
        json.dumps(prereq, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not prereq["prerequisites_passed"]:
        write_run_state("failed", {"reason": "required_input_missing", "prereq": prereq})
        raise SystemExit("Required input missing")

    write_run_state("running", {"step": "read_score_panel_and_trd_co"})
    score, score_symbols = load_score_symbols()
    trd, trd_covered_symbols = load_trd_co_symbols(score_symbols)
    trd_missing_symbols = score_symbols - trd_covered_symbols

    write_run_state("running", {"step": "read_er_xlsx", "file_size_bytes": ER_XLSX.stat().st_size})
    er = pd.read_excel(ER_XLSX, header=0, dtype=str, usecols=ER_USECOLS, engine="openpyxl")
    er["Symbol"] = er["Symbol"].astype("string").str.strip()
    er = er[~er["Symbol"].isin(["股票代码", "没有单位"])].copy()
    er["symbol"] = norm_symbol(er["Symbol"])
    er = er[er["symbol"].notna() & er["symbol"].str.fullmatch(r"\d{6}")].copy()
    er["DeclareDate_dt"] = pd.to_datetime(er["DeclareDate"], errors="coerce")
    er["DeclareYear"] = er["DeclareDate_dt"].dt.year.astype("Int64")
    for c in ["IndustryCode", "IndustryName", "IndustryCode1", "IndustryName1", "ClassID", "ClassName"]:
        er[c] = er[c].astype("string").str.strip()

    row_count = int(len(er))
    er_unique_symbols = int(er["symbol"].nunique())
    schema = pd.DataFrame(
        {
            "column": er.columns,
            "dtype": [str(er[c].dtype) for c in er.columns],
            "non_null_count": [int(non_empty(er[c]).sum()) if c in er.columns else 0 for c in er.columns],
        }
    )
    schema.to_csv(OUT_DIR / "er_announcement_schema_profile.csv", index=False, encoding="utf-8-sig")

    years = sorted([int(y) for y in er["DeclareYear"].dropna().unique().tolist()])
    row_by_year = er.groupby("DeclareYear", dropna=False).size().reset_index(name="row_count")

    fields = ["Symbol", "DeclareDate", "ClassID", "ClassName", "IndustryCode", "IndustryName", "IndustryCode1", "IndustryName1"]
    coverage_rows = []
    for c in fields:
        mask = non_empty(er[c])
        samples = er.loc[mask, c].drop_duplicates().head(10).tolist()
        coverage_rows.append(
            {
                "field": c,
                "non_null_count": int(mask.sum()),
                "non_null_ratio": float(mask.mean()) if row_count else 0.0,
                "unique_value_count": int(er.loc[mask, c].nunique()),
                "sample_values": json.dumps(samples, ensure_ascii=False),
            }
        )
    field_cov = pd.DataFrame(coverage_rows)
    field_cov.to_csv(OUT_DIR / "er_announcement_field_coverage_profile.csv", index=False, encoding="utf-8-sig")

    class_dist = (
        er.groupby(["ClassID", "ClassName"], dropna=False)
        .agg(
            row_count=("symbol", "size"),
            unique_symbol_count=("symbol", "nunique"),
            date_min=("DeclareDate_dt", "min"),
            date_max=("DeclareDate_dt", "max"),
        )
        .reset_index()
        .sort_values(["row_count", "unique_symbol_count"], ascending=False)
    )
    class_dist.to_csv(OUT_DIR / "er_announcement_class_distribution.csv", index=False, encoding="utf-8-sig")

    both_code = non_empty(er["IndustryCode"]) & non_empty(er["IndustryCode1"])
    both_name = non_empty(er["IndustryName"]) & non_empty(er["IndustryName1"])
    code_equal = er.loc[both_code, "IndustryCode"].eq(er.loc[both_code, "IndustryCode1"])
    name_equal = er.loc[both_name, "IndustryName"].eq(er.loc[both_name, "IndustryName1"])
    conflicts = er.loc[both_code & ~er["IndustryCode"].eq(er["IndustryCode1"]), [
        "symbol", "DeclareDate", "IndustryCode", "IndustryName", "IndustryCode1", "IndustryName1"
    ]].head(50)
    consistency = pd.DataFrame(
        [
            {
                "both_available_rows": int(both_code.sum()),
                "code_equal_ratio": float(code_equal.mean()) if len(code_equal) else np.nan,
                "name_equal_ratio": float(name_equal.mean()) if len(name_equal) else np.nan,
                "conflict_count": int((both_code & ~er["IndustryCode"].eq(er["IndustryCode1"])).sum()),
                "conflict_examples": conflicts.to_json(orient="records", force_ascii=False, date_format="iso"),
            }
        ]
    )
    consistency.to_csv(OUT_DIR / "er_industry_field_consistency_check.csv", index=False, encoding="utf-8-sig")

    code1_dom = er.groupby("symbol", sort=False).apply(dominant_value, "IndustryCode1", "IndustryName1", include_groups=False)
    code_dom = er.groupby("symbol", sort=False).apply(dominant_value, "IndustryCode", "IndustryName", include_groups=False)
    dates = er.groupby("symbol").agg(
        announcement_count=("symbol", "size"),
        first_declare_date=("DeclareDate_dt", "min"),
        last_declare_date=("DeclareDate_dt", "max"),
    )
    sym = dates.join(code1_dom.add_prefix("code1_")).join(code_dom.add_prefix("fallback_")).reset_index()
    sym = sym.rename(
        columns={
            "code1_unique_code_count": "unique_industry_code1_count",
            "code1_unique_name_count": "unique_industry_name1_count",
            "code1_dominant_code": "dominant_industry_code1",
            "code1_dominant_name": "dominant_industry_name1",
            "code1_dominant_share": "dominant_industry_share",
            "fallback_dominant_code": "fallback_industry_code",
            "fallback_dominant_name": "fallback_industry_name",
        }
    )
    sym["industry_conflict_flag"] = sym["unique_industry_code1_count"].fillna(0).astype(int) > 1
    sym["supplement_use_status"] = np.select(
        [
            sym["unique_industry_code1_count"].fillna(0).astype(int).eq(1),
            sym["unique_industry_code1_count"].fillna(0).astype(int).gt(1) & sym["dominant_industry_share"].fillna(0).ge(0.90),
            sym["unique_industry_code1_count"].fillna(0).astype(int).gt(1) & sym["dominant_industry_share"].fillna(0).lt(0.90),
            sym["unique_industry_code1_count"].fillna(0).astype(int).eq(0) & sym["fallback_unique_code_count"].fillna(0).astype(int).gt(0),
        ],
        [
            "AUTO_USE_UNIQUE_INDUSTRY",
            "WATCH_USE_DOMINANT_INDUSTRY",
            "CONFLICT_MANUAL_REVIEW",
            "AUTO_USE_UNIQUE_INDUSTRY",
        ],
        default="NOT_USABLE_MISSING_INDUSTRY",
    )
    sym.loc[
        sym["unique_industry_code1_count"].fillna(0).astype(int).eq(0)
        & sym["fallback_unique_code_count"].fillna(0).astype(int).gt(1)
        & sym["fallback_dominant_share"].fillna(0).lt(0.90),
        "supplement_use_status",
    ] = "CONFLICT_MANUAL_REVIEW"
    sym["industry_code_for_supplement"] = sym["dominant_industry_code1"].where(
        non_empty(sym["dominant_industry_code1"]), sym["fallback_industry_code"]
    )
    sym["industry_name_for_supplement"] = sym["dominant_industry_name1"].where(
        non_empty(sym["dominant_industry_name1"]), sym["fallback_industry_name"]
    )
    sym.loc[~non_empty(sym["industry_code_for_supplement"]) & ~non_empty(sym["industry_name_for_supplement"]), "supplement_use_status"] = "NOT_USABLE_MISSING_INDUSTRY"

    required_sym_cols = [
        "symbol",
        "announcement_count",
        "first_declare_date",
        "last_declare_date",
        "unique_industry_code1_count",
        "unique_industry_name1_count",
        "dominant_industry_code1",
        "dominant_industry_name1",
        "dominant_industry_share",
        "fallback_industry_code",
        "fallback_industry_name",
        "industry_conflict_flag",
        "supplement_use_status",
    ]
    sym[required_sym_cols].to_csv(OUT_DIR / "er_symbol_industry_consistency_profile.csv", index=False, encoding="utf-8-sig")

    er_symbols = set(sym["symbol"].dropna().unique().tolist())
    er_overlap_score = er_symbols & score_symbols
    er_overlap_missing = er_symbols & trd_missing_symbols
    usable_status = {"AUTO_USE_UNIQUE_INDUSTRY", "WATCH_USE_DOMINANT_INDUSTRY"}
    supplement_symbols = set(
        sym.loc[
            sym["symbol"].isin(trd_missing_symbols) & sym["supplement_use_status"].isin(usable_status),
            "symbol",
        ].tolist()
    )
    merged_symbols = trd_covered_symbols | supplement_symbols
    score_rows = len(score)
    score_rows_covered = int(score["symbol"].isin(merged_symbols).sum())

    pd.DataFrame(
        [
            {
                "score_panel_unique_symbols": len(score_symbols),
                "er_unique_symbols": len(er_symbols),
                "er_overlap_score_symbols": len(er_overlap_score),
                "er_overlap_score_ratio": len(er_overlap_score) / len(score_symbols) if score_symbols else 0.0,
            }
        ]
    ).to_csv(OUT_DIR / "er_coverage_against_score_panel.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "trd_co_missing_unique_symbols": len(trd_missing_symbols),
                "er_overlap_trd_co_missing_symbols": len(er_overlap_missing),
                "er_overlap_trd_co_missing_ratio": len(er_overlap_missing) / len(trd_missing_symbols) if trd_missing_symbols else 0.0,
                "remaining_missing_symbols_after_er_supplement": len(score_symbols - merged_symbols),
            }
        ]
    ).to_csv(OUT_DIR / "er_coverage_against_trd_co_missing_symbols.csv", index=False, encoding="utf-8-sig")

    merged_cov = {
        "trd_co_existing_coverage_unique_symbols": len(trd_covered_symbols),
        "er_supplement_unique_symbols": len(supplement_symbols),
        "merged_estimated_covered_unique_symbols": len(merged_symbols),
        "merged_estimated_coverage_over_score_symbols": len(merged_symbols) / len(score_symbols) if score_symbols else 0.0,
        "merged_estimated_row_coverage_over_score_panel": score_rows_covered / score_rows if score_rows else 0.0,
    }
    pd.DataFrame([merged_cov]).to_csv(OUT_DIR / "merged_trd_co_er_estimated_coverage.csv", index=False, encoding="utf-8-sig")

    candidate = sym.loc[
        sym["symbol"].isin(trd_missing_symbols) & sym["supplement_use_status"].isin(usable_status),
        [
            "symbol",
            "industry_code_for_supplement",
            "industry_name_for_supplement",
            "supplement_use_status",
            "dominant_industry_share",
            "first_declare_date",
            "last_declare_date",
            "announcement_count",
            "dominant_industry_code1",
        ],
    ].copy()
    candidate["source"] = "ER_Announcement"
    candidate["industry_source_field"] = np.where(non_empty(candidate["dominant_industry_code1"]), "IndustryCode1/IndustryName1", "IndustryCode/IndustryName")
    candidate["pit_quality_status"] = "STATIC_SUPPLEMENT_NOT_PIT"
    candidate = candidate.rename(
        columns={
            "industry_code_for_supplement": "industry_code",
            "industry_name_for_supplement": "industry_name",
        }
    )
    candidate[
        [
            "symbol",
            "source",
            "industry_code",
            "industry_name",
            "industry_source_field",
            "supplement_use_status",
            "dominant_industry_share",
            "first_declare_date",
            "last_declare_date",
            "announcement_count",
            "pit_quality_status",
        ]
    ].to_csv(OUT_DIR / "er_supplement_candidate_symbol_industry.csv", index=False, encoding="utf-8-sig")

    remaining = pd.DataFrame({"symbol": sorted(score_symbols - merged_symbols)})
    remaining.to_csv(OUT_DIR / "remaining_missing_symbols_after_er_supplement.csv", index=False, encoding="utf-8-sig")

    conflict_symbol_count = int(sym["industry_conflict_flag"].sum())
    unique_symbol_count = int(sym["unique_industry_code1_count"].fillna(0).astype(int).eq(1).sum())
    conflict_rate = conflict_symbol_count / er_unique_symbols if er_unique_symbols else 1.0
    merged_ratio = merged_cov["merged_estimated_coverage_over_score_symbols"]
    can_supplement = len(supplement_symbols) > 0 and conflict_rate < 0.50
    if conflict_rate >= 0.50:
        final_decision = "ER_ANNOUNCEMENT_SUPPLEMENT_FAIL_CONFLICT_TOO_HIGH"
    elif merged_ratio >= 0.95 and can_supplement:
        final_decision = "ER_ANNOUNCEMENT_SUPPLEMENT_READY_FOR_MERGED_STATIC_INDUSTRY_SOURCE_BUILD"
    elif 0.80 <= merged_ratio < 0.95 and can_supplement:
        final_decision = "ER_ANNOUNCEMENT_SUPPLEMENT_WATCH_PARTIAL_COVERAGE"
    elif merged_ratio < 0.80:
        final_decision = "ER_ANNOUNCEMENT_SUPPLEMENT_FAIL_NOT_ENOUGH_COVERAGE"
    else:
        final_decision = "ER_ANNOUNCEMENT_SUPPLEMENT_FAIL"

    field_cov_map = {r["field"]: r for r in coverage_rows}
    code1_cov = field_cov_map["IndustryCode1"]["non_null_ratio"]
    name1_cov = field_cov_map["IndustryName1"]["non_null_ratio"]
    code_cov = field_cov_map["IndustryCode"]["non_null_ratio"]
    name_cov = field_cov_map["IndustryName"]["non_null_ratio"]
    er_industry_source_type = "EVENT_ATTACHED_STATIC_INDUSTRY_CANDIDATE" if can_supplement else "NOT_USABLE"
    pit_quality_status = "STATIC_SUPPLEMENT_NOT_PIT" if can_supplement else "NOT_USABLE"

    policy = {
        "er_industry_source_type": er_industry_source_type,
        "pit_quality_status": pit_quality_status,
        "preferred_industry_fields": ["IndustryCode1", "IndustryName1"],
        "fallback_industry_fields": ["IndustryCode", "IndustryName"],
        "class_fields_not_industry": ["ClassID", "ClassName"],
        "declare_date_interpretation": "DeclareDate 是公告日期，不得自动解释为行业分类生效日期。",
        "can_supplement_trd_co": bool(can_supplement),
        "can_use_for_static_neutral_score_after_merge": bool(final_decision == "ER_ANNOUNCEMENT_SUPPLEMENT_READY_FOR_MERGED_STATIC_INDUSTRY_SOURCE_BUILD"),
        "can_claim_pit_industry_neutralization": False,
        "limitations": [
            "ER_Announcement 只能作为 supplemental static industry source 候选。",
            "2016-2019 公告数据不能单独证明 2020-2026 的行业 PIT 状态。",
            "作为 static supplement 时必须标注 STATIC_NOT_PIT。",
            "ClassID/ClassName 是公告事件分类，不得替代行业字段。",
        ],
    }
    (OUT_DIR / "er_supplement_industry_source_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "run_timestamp": prereq["run_timestamp"],
        "prerequisites_passed": True,
        "er_xlsx_read": True,
        "er_xlsx_path": str(ER_XLSX),
        "er_row_count": row_count,
        "er_unique_symbols": er_unique_symbols,
        "er_declare_date_min": None if er["DeclareDate_dt"].dropna().empty else er["DeclareDate_dt"].min().date().isoformat(),
        "er_declare_date_max": None if er["DeclareDate_dt"].dropna().empty else er["DeclareDate_dt"].max().date().isoformat(),
        "er_years_covered": years,
        "row_count_by_year": row_by_year.to_dict(orient="records"),
        "industry_code1_coverage_ratio": code1_cov,
        "industry_name1_coverage_ratio": name1_cov,
        "industry_code_coverage_ratio": code_cov,
        "industry_name_coverage_ratio": name_cov,
        "industry_code1_vs_industry_code_conflict_count": int(consistency.loc[0, "conflict_count"]),
        "symbols_with_unique_industry_count": unique_symbol_count,
        "symbols_with_conflicting_industry_count": conflict_symbol_count,
        "score_panel_unique_symbols": len(score_symbols),
        "trd_co_existing_covered_unique_symbols": len(trd_covered_symbols),
        "trd_co_missing_unique_symbols": len(trd_missing_symbols),
        "er_overlap_score_symbols": len(er_overlap_score),
        "er_overlap_trd_co_missing_symbols": len(er_overlap_missing),
        "er_auto_supplement_symbol_count": int((candidate["supplement_use_status"] == "AUTO_USE_UNIQUE_INDUSTRY").sum()),
        "er_watch_supplement_symbol_count": int((candidate["supplement_use_status"] == "WATCH_USE_DOMINANT_INDUSTRY").sum()),
        "remaining_missing_symbols_after_er_supplement": len(score_symbols - merged_symbols),
        **merged_cov,
        **policy,
        "neutral_score_generated": False,
        "ic_calculated": False,
        "d10_d1_calculated": False,
        "portfolio_constructed": False,
        "portfolio_return_calculated": False,
        "backtest_run": False,
        "transaction_cost_calculated": False,
        "turnover_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "benchmark_relative_return_calculated": False,
        "alpha_beta_regression_calculated": False,
        "training_run": False,
        "shap_calculated": False,
        "tuning_run": False,
        "feature_importance_calculated": False,
        "production_modified": False,
        "final_decision": final_decision,
        "recommended_next_step": "若接受 WATCH_PARTIAL_COVERAGE，则继续下载更完整历史/退市行业源；若目标必须 >=0.95，则进入 merged static source 前还需补更多来源。",
    }
    (OUT_DIR / "er_announcement_industry_source_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    plan = f"""# ER Announcement Supplement Plan

## 结论

- final_decision: {final_decision}
- PIT 标记: {pit_quality_status}
- 不生成 neutral score，不计算 IC，不回测。

## 建议

1. 仅将 ER_Announcement 作为 TRD_Co 缺失 symbol 的 static supplement 候选。
2. 自动使用 `AUTO_USE_UNIQUE_INDUSTRY`，人工复核 `WATCH_USE_DOMINANT_INDUSTRY`。
3. 若需要 PIT 行业历史，必须另找含行业生效日期或历史行业变更的来源。
"""
    (OUT_DIR / "next_step_er_supplement_plan.md").write_text(plan, encoding="utf-8")

    report = f"""# ER_Announcement Supplemental Industry Source Suitability Audit v0

## 审计结论

- final_decision: `{final_decision}`
- er_industry_source_type: `{er_industry_source_type}`
- pit_quality_status: `{pit_quality_status}`
- can_supplement_trd_co: `{can_supplement}`
- can_claim_pit_industry_neutralization: `False`

## 核心覆盖

- score_panel_unique_symbols: {len(score_symbols)}
- trd_co_existing_covered_unique_symbols: {len(trd_covered_symbols)}
- trd_co_missing_unique_symbols: {len(trd_missing_symbols)}
- er_overlap_score_symbols: {len(er_overlap_score)}
- er_overlap_trd_co_missing_symbols: {len(er_overlap_missing)}
- er_supplement_unique_symbols: {len(supplement_symbols)}
- merged_estimated_coverage_over_score_symbols: {merged_ratio:.6f}

## PIT 风险说明

DeclareDate 是公告日期，不得自动解释为行业分类生效日期。ER_Announcement 2016-2019 数据只能作为 supplemental static industry source 候选，不能单独证明 2020-2026 的 PIT 行业状态。若后续作为 static supplement，必须标注 `STATIC_NOT_PIT`。
"""
    (OUT_DIR / "er_announcement_industry_source_audit_report.md").write_text(report, encoding="utf-8")

    final_qa = pd.DataFrame(
        [
            {"check": "neutral_score_generated", "value": False, "passed": True},
            {"check": "ic_calculated", "value": False, "passed": True},
            {"check": "portfolio_return_calculated", "value": False, "passed": True},
            {"check": "backtest_run", "value": False, "passed": True},
            {"check": "production_modified", "value": False, "passed": True},
            {"check": "class_fields_not_used_as_industry", "value": True, "passed": True},
            {"check": "declare_date_not_treated_as_effective_date", "value": True, "passed": True},
        ]
    )
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")

    terminal_summary = {
        "task_name": TASK_NAME,
        "status": "completed",
        "stdout_log": str(RUN_DIR / "run_stdout.txt"),
        "stderr_log": str(RUN_DIR / "run_stderr.txt"),
        "outputs": [str(p) for p in sorted(OUT_DIR.glob("*")) if p.is_file()],
    }
    (OUT_DIR / "terminal_summary.json").write_text(json.dumps(terminal_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    card = f"""# Task Completion Card

- task_name: {TASK_NAME}
- status: completed
- final_decision: {final_decision}
- output_dir: {OUT_DIR}
- run_dir: {RUN_DIR}
- logs: {RUN_DIR / 'run_stdout.txt'} ; {RUN_DIR / 'run_stderr.txt'}
"""
    (OUT_DIR / "task_completion_card.md").write_text(card, encoding="utf-8")

    del er, score, trd, sym, candidate, remaining, class_dist, field_cov, schema
    gc.collect()
    write_run_state("completed", {"final_decision": final_decision, "summary_path": str(OUT_DIR / "er_announcement_industry_source_audit_summary.json")})
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": str(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
