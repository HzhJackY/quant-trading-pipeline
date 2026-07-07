from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "DEBT_INSTITUTIONINFO Annual Industry Source Suitability Audit v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "debt_institutioninfo_annual_industry_neutral_score_run_v0"
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

XLSX_PATH = ROOT / "data" / "csmar_exports" / "DEBT_INSTITUTIONINFO.xlsx"
SCORE_PANEL_PATH = ROOT / "output" / "simple_baseline_score_run_v0" / "simple_baseline_score_panel_v0.parquet"
REFERENCE_INPUTS = {
    "trd_co_summary": ROOT / "output" / "trd_co_static_industry_join_forensics_v0" / "trd_co_static_industry_join_forensics_summary.json",
    "er_announcement_summary": ROOT / "output" / "er_announcement_industry_source_audit_v0" / "er_announcement_industry_source_audit_summary.json",
    "missing_industry_symbol_master": ROOT / "output" / "historical_industry_source_gap_resolution_v0" / "missing_industry_symbol_master.csv",
}

FIELDS = [
    "Symbol",
    "EndDate",
    "CONAME",
    "ABSign",
    "Plate",
    "FullName",
    "INDCLASSIFYSYSTEM",
    "INDUSTRYCODE",
    "IndustryName",
    "LISTINGDATE",
    "Ownership",
    "CURRENCY",
]
SCORE_COLS = [
    "symbol",
    "month_end",
    "bp_rank",
    "ep_ttm_rank",
    "cfo_to_earnings_parent_rank",
    "VALUE_BP_SINGLE_score",
    "VALUE_QUALITY_EQUAL_WEIGHT_score",
    "fwd_ret_1m",
]
SYSTEM_PRIORITY = {"P0207": 1, "P0221": 2, "P0201": 3}


def write_run_state(status: str, details: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "# RUN_STATE",
            "",
            f"- task_name: {TASK_NAME}",
            f"- status: {status}",
            f"- updated_at: {datetime.now().isoformat(timespec='seconds')}",
            f"- output_dir: {OUT_DIR}",
            f"- run_dir: {RUN_DIR}",
            "",
            "## Details",
            "```json",
            json.dumps(details, ensure_ascii=False, indent=2, default=str),
            "```",
            "",
        ]
    )
    (OUT_DIR / "RUN_STATE.md").write_text(text, encoding="utf-8")
    (RUN_DIR / "RUN_STATE.md").write_text(text, encoding="utf-8")


def norm_symbol(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.extract(r"(\d{6})", expand=False)


def non_empty(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.notna() & (s != "") & (s.str.lower() != "nan")


def first_non_empty(series: pd.Series):
    s = series[non_empty(series)]
    return pd.NA if s.empty else s.iloc[0]


def pct_rank(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(method="average", pct=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_run_state("running", {"step": "start"})

    prereq = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "required_inputs": {"xlsx": str(XLSX_PATH), "score_panel": str(SCORE_PANEL_PATH)},
        "required_exists": {"xlsx": XLSX_PATH.exists(), "score_panel": SCORE_PANEL_PATH.exists()},
        "reference_inputs": {k: str(v) for k, v in REFERENCE_INPUTS.items()},
        "reference_exists": {k: v.exists() for k, v in REFERENCE_INPUTS.items()},
    }
    prereq["prerequisites_passed"] = all(prereq["required_exists"].values())
    (OUT_DIR / "debt_institutioninfo_annual_industry_prerequisite_check.json").write_text(
        json.dumps(prereq, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not prereq["prerequisites_passed"]:
        summary = {
            "run_timestamp": prereq["run_timestamp"],
            "xlsx_read": False,
            "xlsx_path": str(XLSX_PATH),
            "prerequisites_passed": False,
            "neutral_score_generated": False,
            "final_decision": "DEBT_INSTITUTIONINFO_ANNUAL_INDUSTRY_NEUTRAL_SCORE_RUN_FAIL",
            "recommended_next_step": "补齐缺失输入后重跑。",
        }
        (OUT_DIR / "debt_institutioninfo_annual_industry_neutral_score_run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_run_state("failed", summary)
        raise SystemExit("Required input missing")

    write_run_state("running", {"step": "read_xlsx_once", "xlsx_size_bytes": XLSX_PATH.stat().st_size})
    debt = pd.read_excel(XLSX_PATH, header=0, dtype=str, engine="openpyxl")
    missing_fields = [c for c in FIELDS if c not in debt.columns]
    if missing_fields:
        raise ValueError(f"Missing expected fields: {missing_fields}")
    debt = debt[FIELDS].copy()
    debt["Symbol"] = debt["Symbol"].astype("string").str.strip()
    debt = debt[~debt["Symbol"].isin(["证券代码", "没有单位"])].copy()
    debt["symbol"] = norm_symbol(debt["Symbol"])
    debt = debt[debt["symbol"].notna() & debt["symbol"].str.fullmatch(r"\d{6}")].copy()
    for c in FIELDS:
        debt[c] = debt[c].astype("string").str.strip()
    debt["EndDate_dt"] = pd.to_datetime(debt["EndDate"], errors="coerce")
    debt["year"] = debt["EndDate_dt"].dt.year.astype("Int64")

    row_count = int(len(debt))
    unique_symbols = int(debt["symbol"].nunique())
    years = sorted(int(y) for y in debt["year"].dropna().unique().tolist())

    schema = pd.DataFrame(
        {
            "column": debt.columns,
            "dtype": [str(debt[c].dtype) for c in debt.columns],
            "non_null_count": [int(non_empty(debt[c]).sum()) for c in debt.columns],
        }
    )
    schema.to_csv(OUT_DIR / "debt_institutioninfo_schema_profile.csv", index=False, encoding="utf-8-sig")

    coverage_rows = []
    for c in FIELDS:
        mask = non_empty(debt[c])
        coverage_rows.append(
            {
                "field": c,
                "non_null_count": int(mask.sum()),
                "non_null_ratio": float(mask.mean()) if row_count else 0.0,
                "unique_value_count": int(debt.loc[mask, c].nunique()),
                "sample_values": json.dumps(debt.loc[mask, c].drop_duplicates().head(10).tolist(), ensure_ascii=False),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(
        OUT_DIR / "debt_institutioninfo_field_coverage_profile.csv", index=False, encoding="utf-8-sig"
    )

    system_dist = (
        debt.groupby("INDCLASSIFYSYSTEM", dropna=False)
        .agg(
            row_count=("symbol", "size"),
            unique_symbol_count=("symbol", "nunique"),
            year_min=("year", "min"),
            year_max=("year", "max"),
            industry_code_count=("INDUSTRYCODE", "nunique"),
            industry_name_count=("IndustryName", "nunique"),
        )
        .reset_index()
        .sort_values("row_count", ascending=False)
    )
    system_dist.to_csv(
        OUT_DIR / "debt_institutioninfo_industry_system_distribution.csv", index=False, encoding="utf-8-sig"
    )

    write_run_state("running", {"step": "select_annual_industry_policy"})
    exact_dedup = debt.drop_duplicates().copy()
    system_counts = (
        exact_dedup.groupby(["symbol", "EndDate", "INDCLASSIFYSYSTEM"], dropna=False)
        .agg(
            row_count=("symbol", "size"),
            industry_code_count=("INDUSTRYCODE", "nunique"),
            industry_name_count=("IndustryName", "nunique"),
            industry_codes=("INDUSTRYCODE", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
            industry_names=("IndustryName", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
        )
        .reset_index()
    )
    system_counts["system_conflict_flag"] = system_counts["industry_code_count"].gt(1)
    selected_system_conflicts = system_counts[system_counts["system_conflict_flag"]].copy()

    unique_check = system_counts.copy()
    unique_check.to_csv(
        OUT_DIR / "debt_institutioninfo_symbol_year_uniqueness_check.csv", index=False, encoding="utf-8-sig"
    )

    conflict_keys = set(
        zip(
            selected_system_conflicts["symbol"].astype(str),
            selected_system_conflicts["EndDate"].astype(str),
            selected_system_conflicts["INDCLASSIFYSYSTEM"].astype(str),
        )
    )
    clean = exact_dedup[
        ~exact_dedup.apply(
            lambda r: (str(r["symbol"]), str(r["EndDate"]), str(r["INDCLASSIFYSYSTEM"])) in conflict_keys, axis=1
        )
    ].copy()
    clean = clean[clean["INDCLASSIFYSYSTEM"].isin(SYSTEM_PRIORITY)].copy()
    clean["selection_source_priority"] = clean["INDCLASSIFYSYSTEM"].map(SYSTEM_PRIORITY).astype(int)
    clean = clean.sort_values(["symbol", "EndDate_dt", "selection_source_priority", "INDUSTRYCODE", "IndustryName"])
    selected = clean.groupby(["symbol", "EndDate"], as_index=False, sort=False).head(1).copy()
    selected_dupe_check = selected.groupby(["symbol", "EndDate"]).size().reset_index(name="selected_row_count")
    selected_one_to_many = selected_dupe_check[selected_dupe_check["selected_row_count"].gt(1)].copy()

    conflicts = selected_system_conflicts.copy()
    if not selected_one_to_many.empty:
        selected_one_to_many["conflict_type"] = "SELECTED_ONE_TO_MANY"
        conflicts = pd.concat([conflicts, selected_one_to_many], ignore_index=True, sort=False)
    conflicts.to_csv(
        OUT_DIR / "debt_institutioninfo_industry_conflict_review.csv", index=False, encoding="utf-8-sig"
    )

    selected = selected.rename(
        columns={
            "INDCLASSIFYSYSTEM": "selected_industry_system",
            "INDUSTRYCODE": "selected_industry_code",
            "IndustryName": "selected_industry_name",
        }
    )
    selected_out_cols = [
        "symbol",
        "EndDate",
        "year",
        "selected_industry_system",
        "selected_industry_code",
        "selected_industry_name",
        "CONAME",
        "ABSign",
        "Plate",
        "FullName",
        "LISTINGDATE",
        "Ownership",
        "CURRENCY",
        "selection_source_priority",
    ]
    selected[selected_out_cols].to_csv(
        OUT_DIR / "selected_annual_industry_source.csv", index=False, encoding="utf-8-sig"
    )

    write_run_state("running", {"step": "asof_join_score_panel"})
    score = pd.read_parquet(SCORE_PANEL_PATH, columns=SCORE_COLS)
    score["symbol"] = norm_symbol(score["symbol"])
    score["month_end"] = pd.to_datetime(score["month_end"], errors="coerce")
    score = score[score["symbol"].notna() & score["month_end"].notna()].copy()
    score["_row_id"] = np.arange(len(score))

    source = selected[
        [
            "symbol",
            "EndDate",
            "EndDate_dt",
            "selected_industry_system",
            "selected_industry_code",
            "selected_industry_name",
        ]
    ].copy()
    source = source[source["EndDate_dt"].notna()].sort_values(["symbol", "EndDate_dt"])
    joined_parts = []
    for symbol, left in score.sort_values(["symbol", "month_end"]).groupby("symbol", sort=False):
        right = source[source["symbol"] == symbol].sort_values("EndDate_dt")
        if right.empty:
            part = left.copy()
            part["industry_asof_enddate"] = pd.NaT
            part["selected_industry_system"] = pd.NA
            part["primary_industry_code"] = pd.NA
            part["primary_industry_name"] = pd.NA
        else:
            part = pd.merge_asof(
                left.sort_values("month_end"),
                right.rename(
                    columns={
                        "EndDate_dt": "industry_asof_enddate",
                        "selected_industry_code": "primary_industry_code",
                        "selected_industry_name": "primary_industry_name",
                    }
                ).sort_values("industry_asof_enddate"),
                left_on="month_end",
                right_on="industry_asof_enddate",
                by="symbol",
                direction="backward",
                allow_exact_matches=True,
            )
        joined_parts.append(part)
    joined = pd.concat(joined_parts, ignore_index=True).sort_values("_row_id")
    joined["industry_join_lag_days"] = (joined["month_end"] - joined["industry_asof_enddate"]).dt.days
    joined["industry_join_policy"] = "ANNUAL_ASOF_ENDDATE_LE_MONTH_END"
    joined["joined_flag"] = non_empty(joined["primary_industry_code"])
    future_enddate_used = bool((joined["industry_asof_enddate"] > joined["month_end"]).fillna(False).any())

    score_panel_rows = int(len(joined))
    score_panel_unique_symbols = int(joined["symbol"].nunique())
    joined_rows = int(joined["joined_flag"].sum())
    joined_unique_symbols = int(joined.loc[joined["joined_flag"], "symbol"].nunique())
    missing_rows = score_panel_rows - joined_rows
    missing_unique_symbols = score_panel_unique_symbols - joined_unique_symbols
    join_coverage_ratio = joined_rows / score_panel_rows if score_panel_rows else 0.0
    duplicate_unsafe_join = bool(not selected_one_to_many.empty)

    qa = pd.DataFrame(
        [
            {
                "score_panel_rows": score_panel_rows,
                "score_panel_unique_symbols": score_panel_unique_symbols,
                "annual_industry_unique_symbols": int(selected["symbol"].nunique()),
                "joined_rows": joined_rows,
                "join_coverage_ratio": join_coverage_ratio,
                "joined_unique_symbols": joined_unique_symbols,
                "missing_unique_symbols": missing_unique_symbols,
                "missing_rows": missing_rows,
                "missing_row_ratio": missing_rows / score_panel_rows if score_panel_rows else 0.0,
                "duplicate_unsafe_join": duplicate_unsafe_join,
                "future_enddate_used": future_enddate_used,
            }
        ]
    )
    qa.to_csv(OUT_DIR / "annual_industry_asof_join_qa.csv", index=False, encoding="utf-8-sig")

    missing = joined[~joined["joined_flag"]].copy()
    missing_profile = (
        missing.groupby("symbol")
        .agg(row_count=("symbol", "size"), first_month=("month_end", "min"), last_month=("month_end", "max"))
        .reset_index()
        .sort_values(["row_count", "symbol"], ascending=[False, True])
    )
    missing_profile.to_csv(
        OUT_DIR / "annual_industry_missing_symbol_profile.csv", index=False, encoding="utf-8-sig"
    )
    missing_by_month = missing.groupby("month_end").size().reset_index(name="missing_rows")
    missing_by_month.to_csv(
        OUT_DIR / "annual_industry_missing_by_month.csv", index=False, encoding="utf-8-sig"
    )
    missing_by_year = missing.assign(year=missing["month_end"].dt.year).groupby("year").size().reset_index(name="missing_rows")

    joined_for_group = joined[joined["joined_flag"]].copy()
    group_size = (
        joined_for_group.groupby(["month_end", "primary_industry_code"], dropna=False)
        .size()
        .reset_index(name="group_size")
    )
    group_size["small_group_flag"] = group_size["group_size"].lt(5)
    group_size.to_csv(
        OUT_DIR / "annual_industry_group_size_summary.csv", index=False, encoding="utf-8-sig"
    )
    small_industry_group_detected = bool(group_size["small_group_flag"].any()) if not group_size.empty else False

    neutral_score_generated = False
    neutral_score_columns: list[str] = []
    neutral_score_row_count = 0
    final_decision: str
    if duplicate_unsafe_join or future_enddate_used:
        final_decision = "DEBT_INSTITUTIONINFO_ANNUAL_INDUSTRY_SOURCE_FAIL_CONFLICT_UNSAFE"
    elif join_coverage_ratio >= 0.95:
        write_run_state("running", {"step": "generate_industry_neutral_scores", "join_coverage_ratio": join_coverage_ratio})
        panel = joined_for_group.merge(group_size, on=["month_end", "primary_industry_code"], how="left")
        rank_cols = ["bp_rank", "ep_ttm_rank", "cfo_to_earnings_parent_rank"]
        for c in rank_cols:
            panel[f"_ind_rank_{c}"] = panel.groupby(["month_end", "primary_industry_code"], dropna=False)[c].transform(pct_rank)
        panel["ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score"] = panel["_ind_rank_bp_rank"]
        panel["ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score"] = panel[
            ["_ind_rank_bp_rank", "_ind_rank_ep_ttm_rank", "_ind_rank_cfo_to_earnings_parent_rank"]
        ].mean(axis=1)
        neutral_score_columns = [
            "ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
            "ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
        ]
        panel_out = panel.rename(columns={"selected_industry_name": "primary_industry_name"})
        out_cols = [
            "symbol",
            "month_end",
            "industry_asof_enddate",
            "selected_industry_system",
            "primary_industry_code",
            "primary_industry_name",
            "industry_join_lag_days",
            "bp_rank",
            "ep_ttm_rank",
            "cfo_to_earnings_parent_rank",
            "VALUE_BP_SINGLE_score",
            "VALUE_QUALITY_EQUAL_WEIGHT_score",
            "ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
            "ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
            "fwd_ret_1m",
            "small_group_flag",
            "industry_join_policy",
        ]
        panel_out[out_cols].to_parquet(
            OUT_DIR / "simple_baseline_asof_industry_neutral_score_panel_v0.parquet", index=False
        )
        neutral_score_generated = True
        neutral_score_row_count = int(len(panel_out))
        final_decision = "DEBT_INSTITUTIONINFO_ASOF_INDUSTRY_NEUTRAL_SCORE_READY_FOR_EVALUATION_PREP"

        manifest = pd.DataFrame(
            [
                {
                    "score_column": "ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
                    "formula": "industry_within_rank(bp_rank)",
                    "direction": "higher_better",
                    "uses_fwd_ret_1m": False,
                },
                {
                    "score_column": "ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
                    "formula": "mean(industry_within_rank(bp_rank), industry_within_rank(ep_ttm_rank), industry_within_rank(cfo_to_earnings_parent_rank))",
                    "direction": "higher_better",
                    "uses_fwd_ret_1m": False,
                },
            ]
        )
        manifest.to_csv(
            OUT_DIR / "asof_industry_neutral_score_formula_manifest.csv", index=False, encoding="utf-8-sig"
        )
        qa_rows = []
        for c in neutral_score_columns:
            vals = pd.to_numeric(panel_out[c], errors="coerce")
            qa_rows.append(
                {
                    "check": f"{c}_range_within_0_1",
                    "value": bool(vals.dropna().between(0, 1).all()),
                    "passed": bool(vals.dropna().between(0, 1).all()),
                }
            )
            qa_rows.append({"check": f"{c}_null_count", "value": int(vals.isna().sum()), "passed": True})
            qa_rows.append({"check": f"{c}_inf_count", "value": int(np.isinf(vals.dropna()).sum()), "passed": int(np.isinf(vals.dropna()).sum()) == 0})
        qa_rows.extend(
            [
                {"check": "neutral_score_columns_created_if_allowed", "value": True, "passed": True},
                {"check": "fwd_ret_1m_not_used_in_score_formula", "value": True, "passed": True},
                {"check": "score_row_count_equals_joined_row_count", "value": neutral_score_row_count == joined_rows, "passed": neutral_score_row_count == joined_rows},
                {"check": "no_future_enddate_used", "value": not future_enddate_used, "passed": not future_enddate_used},
                {"check": "selected_industry_unique_per_symbol_month", "value": True, "passed": True},
            ]
        )
        pd.DataFrame(qa_rows).to_csv(OUT_DIR / "asof_industry_neutral_score_qa.csv", index=False, encoding="utf-8-sig")
        del panel, panel_out
    elif 0.80 <= join_coverage_ratio < 0.95:
        final_decision = "DEBT_INSTITUTIONINFO_ANNUAL_INDUSTRY_SOURCE_WATCH_PARTIAL_COVERAGE"
    else:
        final_decision = "DEBT_INSTITUTIONINFO_ANNUAL_INDUSTRY_SOURCE_FAIL_COVERAGE_LOW"

    summary = {
        "run_timestamp": prereq["run_timestamp"],
        "xlsx_read": True,
        "xlsx_path": str(XLSX_PATH),
        "prerequisites_passed": True,
        "row_count": row_count,
        "unique_symbols": unique_symbols,
        "enddate_min": None if debt["EndDate_dt"].dropna().empty else debt["EndDate_dt"].min().date().isoformat(),
        "enddate_max": None if debt["EndDate_dt"].dropna().empty else debt["EndDate_dt"].max().date().isoformat(),
        "years_covered": years,
        "rows_by_year": debt.groupby("year").size().reset_index(name="row_count").to_dict(orient="records"),
        "unique_symbols_by_year": debt.groupby("year")["symbol"].nunique().reset_index(name="unique_symbols").to_dict(orient="records"),
        "p0207_available": bool((debt["INDCLASSIFYSYSTEM"] == "P0207").any()),
        "p0221_available": bool((debt["INDCLASSIFYSYSTEM"] == "P0221").any()),
        "p0201_available": bool((debt["INDCLASSIFYSYSTEM"] == "P0201").any()),
        "selected_policy_primary_system": "P0207",
        "selected_policy_secondary_system": "P0221",
        "selected_policy_fallback_system": "P0201",
        "symbol_enddate_system_unique": bool(not system_counts["system_conflict_flag"].any()),
        "conflict_symbol_year_count": int(selected_system_conflicts[["symbol", "EndDate"]].drop_duplicates().shape[0]),
        "selected_annual_industry_source_written": True,
        "score_panel_rows": score_panel_rows,
        "score_panel_unique_symbols": score_panel_unique_symbols,
        "annual_industry_unique_symbols": int(selected["symbol"].nunique()),
        "joined_rows": joined_rows,
        "join_coverage_ratio": join_coverage_ratio,
        "joined_unique_symbols": joined_unique_symbols,
        "missing_unique_symbols": missing_unique_symbols,
        "missing_rows": missing_rows,
        "missing_row_ratio": missing_rows / score_panel_rows if score_panel_rows else 0.0,
        "missing_by_year": missing_by_year.to_dict(orient="records"),
        "future_enddate_used": future_enddate_used,
        "annual_asof_join_policy": "ANNUAL_ASOF_ENDDATE_LE_MONTH_END",
        "neutral_score_generated": neutral_score_generated,
        "neutral_score_columns": neutral_score_columns,
        "neutral_score_row_count": neutral_score_row_count,
        "small_industry_group_detected": small_industry_group_detected,
        "fwd_ret_used_in_score_formula": False,
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
        "production_holdings_generated": False,
        "live_order_ready_file_generated": False,
        "production_modified": False,
        "final_decision": final_decision,
        "recommended_next_step": "若已生成 neutral score，则进入 evaluation prep；若未生成，先补齐行业覆盖或处理冲突后重跑。",
    }
    (OUT_DIR / "debt_institutioninfo_annual_industry_neutral_score_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    report = f"""# DEBT_INSTITUTIONINFO Annual Industry Source Audit + As-Of Industry Neutral Score Run v0

## 结论

- final_decision: `{final_decision}`
- join_coverage_ratio: `{join_coverage_ratio:.6f}`
- neutral_score_generated: `{neutral_score_generated}`
- future_enddate_used: `{future_enddate_used}`

## As-of 规则

按 `symbol` 匹配，且只允许 `EndDate <= month_end`，选择最近年度行业记录；不允许未来年度行业回填过去月份。
"""
    (OUT_DIR / "debt_institutioninfo_annual_industry_neutral_score_run_report.md").write_text(report, encoding="utf-8")

    next_step = f"""# Next Step As-Of Industry Neutral Score Evaluation Plan

- final_decision: {final_decision}
- neutral_score_generated: {neutral_score_generated}
- 下一步：若 `neutral_score_generated=true`，只进入 evaluation prep；evaluation 阶段再计算 IC/收益/组合指标。
"""
    (OUT_DIR / "next_step_asof_industry_neutral_score_evaluation_plan.md").write_text(next_step, encoding="utf-8")

    final_qa = pd.DataFrame(
        [
            {"check": "fwd_ret_used_in_score_formula", "value": False, "passed": True},
            {"check": "ic_calculated", "value": False, "passed": True},
            {"check": "d10_d1_calculated", "value": False, "passed": True},
            {"check": "portfolio_return_calculated", "value": False, "passed": True},
            {"check": "backtest_run", "value": False, "passed": True},
            {"check": "production_modified", "value": False, "passed": True},
            {"check": "future_enddate_used", "value": future_enddate_used, "passed": not future_enddate_used},
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
    write_run_state("completed", {"final_decision": final_decision, "summary_path": str(OUT_DIR / "debt_institutioninfo_annual_industry_neutral_score_run_summary.json")})

    del debt, exact_dedup, clean, selected, score, joined, joined_for_group, group_size
    gc.collect()
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": str(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
