from __future__ import annotations

import csv
import gc
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"C:\dev\quant")
TASK_NAME = "trd_co_static_industry_join_forensics_v0"
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

PREV_DIR = ROOT / "output" / "trd_co_static_industry_neutral_score_run_v1"
DEBUG_DIR = ROOT / "output" / "debug_trd_co_excel_ingestion_v0"
SCORE_PANEL = ROOT / "output" / "simple_baseline_score_run_v0" / "simple_baseline_score_panel_v0.parquet"

REQUIRED_INPUTS = [
    PREV_DIR / "trd_co_static_industry_neutral_score_run_summary.json",
    PREV_DIR / "cleaned_trd_co_static_industry_source.csv",
    PREV_DIR / "simple_baseline_static_industry_join_qa.csv",
    PREV_DIR / "trd_co_static_industry_neutral_score_run_report.md",
    DEBUG_DIR / "debug_trd_co_excel_ingestion_summary.json",
    DEBUG_DIR / "trd_co_required_column_check.csv",
    DEBUG_DIR / "trd_co_cleaned_preview.csv",
    SCORE_PANEL,
]

SCORE_COLUMNS = [
    "symbol",
    "month_end",
    "VALUE_BP_SINGLE_score",
    "VALUE_QUALITY_EQUAL_WEIGHT_score",
    "bp_rank",
    "ep_ttm_rank",
    "cfo_to_earnings_parent_rank",
    "fwd_ret_1m",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def exact_norm(value: Any) -> str | None:
    text = as_text(value)
    return text if text else None


def six_digit_strip_suffix(value: Any) -> str | None:
    text = as_text(value).upper()
    if not text:
        return None
    if "." in text:
        text = text.split(".", 1)[0]
    digits = re.findall(r"\d+", text)
    if not digits:
        return None
    code = digits[0]
    if len(code) > 6:
        code = code[-6:]
    return code.zfill(6)


def numeric_code_norm(value: Any) -> str | None:
    text = as_text(value)
    if not text:
        return None
    if "." in text and text.split(".", 1)[0].isdigit():
        text = text.split(".", 1)[0]
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    try:
        return f"{int(digits):06d}"[-6:]
    except ValueError:
        return None


def code_prefix(value: Any) -> str:
    code = six_digit_strip_suffix(value)
    return code[:2] if code else "NA"


def sample_values(series: pd.Series, n: int = 50) -> str:
    return ";".join(series.dropna().astype(str).drop_duplicates().head(n).tolist())


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def join_metrics(score_df: pd.DataFrame, source_keys: pd.DataFrame, key_col: str, method: str) -> dict[str, Any]:
    source = source_keys[[key_col]].dropna().copy()
    source = source.rename(columns={key_col: "_join_key"})
    counts = source["_join_key"].value_counts()
    duplicate_keys = counts[counts > 1]
    dedup_source = source.drop_duplicates("_join_key", keep="first")
    score_keys = score_df[["_score_row_id", "symbol", "_score_raw_symbol", key_col]].rename(columns={key_col: "_join_key"})
    merged = score_keys.merge(dedup_source.assign(_matched=True), on="_join_key", how="left")
    matched_mask = merged["_matched"].fillna(False).astype(bool)
    joined_rows = int(matched_mask.sum())
    score_rows = int(len(score_df))
    matched_symbols = merged.loc[matched_mask, "symbol"].nunique(dropna=True)
    missing_symbols = merged.loc[~matched_mask, "symbol"].nunique(dropna=True)
    score_key_counts = score_keys["_join_key"].value_counts()
    one_to_many_join_count = int(score_keys["_join_key"].isin(duplicate_keys.index).sum()) if len(duplicate_keys) else 0
    many_to_one_join_count = int((score_key_counts > 1).sum())
    return {
        "join_method": method,
        "joined_rows": joined_rows,
        "join_coverage_ratio": joined_rows / score_rows if score_rows else 0.0,
        "joined_unique_symbols": int(matched_symbols),
        "missing_unique_symbols": int(missing_symbols),
        "duplicate_join_detected": bool(len(duplicate_keys) > 0),
        "one_to_many_join_count": one_to_many_join_count,
        "many_to_one_join_count": many_to_one_join_count,
        "notes": f"source duplicate normalized keys={len(duplicate_keys)}",
    }


def former_code_source(cleaned: pd.DataFrame, base_key_col: str, source_key_col: str) -> pd.DataFrame:
    primary = cleaned[[base_key_col]].dropna().rename(columns={base_key_col: source_key_col})
    if "FormerCode" not in cleaned.columns:
        return primary
    former_rows = cleaned[["FormerCode"]].dropna().copy()
    former_rows[source_key_col] = former_rows["FormerCode"].map(six_digit_strip_suffix)
    former_rows = former_rows[[source_key_col]].dropna()
    combined = pd.concat([primary, former_rows], ignore_index=True)
    return combined


def reason_guess(symbol: str, trd_symbols: set[str], former_symbols: set[str]) -> str:
    code = six_digit_strip_suffix(symbol)
    if code in former_symbols:
        return "FORMER_CODE_MAPPING_NEEDED"
    if code and code not in trd_symbols:
        if code.startswith(("0", "3", "6", "8", "4")):
            return "DELISTED_OR_HISTORICAL_BUFFER_MISSING"
        return "SCORE_PANEL_HAS_NON_A_SHARE_OR_SYNTHETIC_SYMBOLS"
    return "UNKNOWN"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{run_timestamp}] start {TASK_NAME}")

    prereq_rows = [
        {"path": rel(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None}
        for path in REQUIRED_INPUTS
    ]
    prerequisites_passed = all(row["exists"] for row in prereq_rows)
    if not prerequisites_passed:
        summary = {
            "run_timestamp": run_timestamp,
            "prerequisites_passed": False,
            "final_decision": "TRD_CO_STATIC_INDUSTRY_JOIN_FORENSICS_FAIL",
            "recommended_next_step": "补齐缺失输入后重跑。",
        }
        write_json(OUT_DIR / "trd_co_static_industry_join_forensics_summary.json", summary)
        return 1

    prev_summary = read_json(PREV_DIR / "trd_co_static_industry_neutral_score_run_summary.json")
    debug_summary = read_json(DEBUG_DIR / "debug_trd_co_excel_ingestion_summary.json")
    previous_join_coverage_ratio = float(prev_summary.get("join_coverage_ratio", 0.0))

    cleaned = pd.read_csv(PREV_DIR / "cleaned_trd_co_static_industry_source.csv", dtype=str)
    score_df = pd.read_parquet(SCORE_PANEL, columns=SCORE_COLUMNS)
    score_df["_score_row_id"] = range(len(score_df))
    score_df["_score_raw_symbol"] = score_df["symbol"].astype(str)

    for frame in [score_df, cleaned]:
        if "symbol" in frame.columns:
            frame["symbol"] = frame["symbol"].astype(str).str.strip()

    score_df["_exact_symbol"] = score_df["symbol"].map(exact_norm)
    score_df["_six_digit_symbol"] = score_df["symbol"].map(six_digit_strip_suffix)
    score_df["_numeric_symbol"] = score_df["symbol"].map(numeric_code_norm)
    cleaned["_exact_symbol"] = cleaned["symbol"].map(exact_norm)
    cleaned["_six_digit_symbol"] = cleaned["symbol"].map(six_digit_strip_suffix)
    cleaned["_numeric_symbol"] = cleaned["symbol"].map(numeric_code_norm)
    cleaned["_former_six_digit_symbol"] = cleaned["FormerCode"].map(six_digit_strip_suffix) if "FormerCode" in cleaned.columns else pd.NA

    score_panel_rows = int(len(score_df))
    score_panel_unique_symbols = int(score_df["symbol"].nunique(dropna=True))
    trd_co_unique_symbols = int(cleaned["symbol"].nunique(dropna=True))

    score_symbols = score_df["symbol"].astype(str)
    score_profile_rows = [
        {"check_name": "dtype", "value": str(score_df["symbol"].dtype), "notes": ""},
        {"check_name": "raw_symbol_sample_50", "value": sample_values(score_df["symbol"], 50), "notes": ""},
        {"check_name": "contains_SH_suffix", "value": bool_text(score_symbols.str.upper().str.contains(r"\.SH", regex=True).any()), "notes": ""},
        {"check_name": "contains_SZ_suffix", "value": bool_text(score_symbols.str.upper().str.contains(r"\.SZ", regex=True).any()), "notes": ""},
        {"check_name": "contains_BJ_suffix", "value": bool_text(score_symbols.str.upper().str.contains(r"\.BJ", regex=True).any()), "notes": ""},
        {"check_name": "six_digit_string_ratio_after_strip_suffix", "value": float(score_df["_six_digit_symbol"].map(lambda x: bool(re.fullmatch(r"\d{6}", x or ""))).mean()), "notes": ""},
        {"check_name": "leading_zero_present", "value": bool_text(score_df["_six_digit_symbol"].dropna().astype(str).str.startswith("0").any()), "notes": ""},
        {"check_name": "has_surrounding_spaces", "value": bool_text((score_df["_score_raw_symbol"] != score_df["_score_raw_symbol"].str.strip()).any()), "notes": ""},
    ]
    write_csv(OUT_DIR / "score_panel_symbol_profile.csv", score_profile_rows, ["check_name", "value", "notes"])

    trd_symbols = cleaned["symbol"].astype(str)
    trd_profile_rows = [
        {"check_name": "raw_Stkcd_sample_50", "value": sample_values(cleaned["Stkcd"], 50) if "Stkcd" in cleaned.columns else "", "notes": ""},
        {"check_name": "cleaned_symbol_sample_50", "value": sample_values(cleaned["symbol"], 50), "notes": ""},
        {"check_name": "symbol_dtype", "value": str(cleaned["symbol"].dtype), "notes": ""},
        {"check_name": "all_six_digit_string", "value": bool_text(cleaned["_six_digit_symbol"].map(lambda x: bool(re.fullmatch(r"\d{6}", x or ""))).all()), "notes": ""},
        {"check_name": "leading_zero_present", "value": bool_text(cleaned["_six_digit_symbol"].dropna().astype(str).str.startswith("0").any()), "notes": ""},
        {"check_name": "has_surrounding_spaces", "value": bool_text((trd_symbols != trd_symbols.str.strip()).any()), "notes": ""},
        {"check_name": "former_code_non_null_count", "value": int(cleaned["FormerCode"].notna().sum()) if "FormerCode" in cleaned.columns else 0, "notes": ""},
    ]
    write_csv(OUT_DIR / "trd_co_symbol_profile.csv", trd_profile_rows, ["check_name", "value", "notes"])

    method_rows = []
    method_rows.append(join_metrics(score_df, cleaned, "_exact_symbol", "exact_symbol_join"))
    method_rows.append(join_metrics(score_df, cleaned, "_six_digit_symbol", "six_digit_strip_suffix_join"))
    method_rows.append(join_metrics(score_df, cleaned, "_numeric_symbol", "numeric_code_join"))
    former_source = former_code_source(cleaned, "_six_digit_symbol", "_former_assisted_symbol")
    score_df["_former_assisted_symbol"] = score_df["_six_digit_symbol"]
    method_rows.append(join_metrics(score_df, former_source, "_former_assisted_symbol", "former_code_assisted_join"))
    write_csv(
        OUT_DIR / "join_method_comparison.csv",
        method_rows,
        [
            "join_method",
            "joined_rows",
            "join_coverage_ratio",
            "joined_unique_symbols",
            "missing_unique_symbols",
            "duplicate_join_detected",
            "one_to_many_join_count",
            "many_to_one_join_count",
            "notes",
        ],
    )

    best = sorted(method_rows, key=lambda row: (float(row["join_coverage_ratio"]), -int(row["one_to_many_join_count"])), reverse=True)[0]
    best_join_method = str(best["join_method"])
    best_key = {
        "exact_symbol_join": "_exact_symbol",
        "six_digit_strip_suffix_join": "_six_digit_symbol",
        "numeric_code_join": "_numeric_symbol",
        "former_code_assisted_join": "_former_assisted_symbol",
    }[best_join_method]

    if best_join_method == "former_code_assisted_join":
        best_source = former_source.rename(columns={"_former_assisted_symbol": "_best_key"}).drop_duplicates("_best_key")
        score_df["_best_key"] = score_df["_former_assisted_symbol"]
    else:
        best_source = cleaned[[best_key]].rename(columns={best_key: "_best_key"}).dropna().drop_duplicates("_best_key")
        score_df["_best_key"] = score_df[best_key]
    matched = score_df[["_score_row_id", "symbol", "_score_raw_symbol", "month_end", "_best_key"]].merge(
        best_source.assign(_matched=True), on="_best_key", how="left"
    )
    matched_mask = matched["_matched"].fillna(False).astype(bool)
    missing = matched.loc[~matched_mask].copy()
    score_df["_month_end_dt"] = pd.to_datetime(score_df["month_end"], errors="coerce")
    missing["_month_end_dt"] = pd.to_datetime(missing["month_end"], errors="coerce")

    trd_symbol_set = set(cleaned["_six_digit_symbol"].dropna().astype(str))
    former_symbol_set = set(cleaned["_former_six_digit_symbol"].dropna().astype(str))
    missing_profile_rows = []
    for sym, group in missing.groupby("_best_key", dropna=False):
        raw_examples = ";".join(group["_score_raw_symbol"].dropna().astype(str).drop_duplicates().head(5).tolist())
        first_month = group["_month_end_dt"].min()
        last_month = group["_month_end_dt"].max()
        missing_profile_rows.append(
            {
                "normalized_symbol": sym if pd.notna(sym) else "",
                "raw_symbol_examples": raw_examples,
                "row_count": int(len(group)),
                "first_month": str(first_month.date()) if pd.notna(first_month) else "",
                "last_month": str(last_month.date()) if pd.notna(last_month) else "",
                "missing_reason_guess": reason_guess(str(sym), trd_symbol_set, former_symbol_set),
                "notes": "",
            }
        )
    missing_profile_rows = sorted(missing_profile_rows, key=lambda row: int(row["row_count"]), reverse=True)
    write_csv(
        OUT_DIR / "missing_symbol_profile.csv",
        missing_profile_rows[:500],
        ["normalized_symbol", "raw_symbol_examples", "row_count", "first_month", "last_month", "missing_reason_guess", "notes"],
    )

    total_by_month = score_df.groupby("month_end").size().rename("total_rows")
    missing_by_month_df = missing.groupby("month_end").size().rename("missing_rows").to_frame().join(total_by_month, how="right").fillna(0)
    missing_by_month_df["missing_rows"] = missing_by_month_df["missing_rows"].astype(int)
    missing_by_month_df["total_rows"] = missing_by_month_df["total_rows"].astype(int)
    missing_by_month_df["missing_ratio"] = missing_by_month_df["missing_rows"] / missing_by_month_df["total_rows"]
    missing_by_month_df.reset_index()[["month_end", "total_rows", "missing_rows", "missing_ratio"]].to_csv(
        OUT_DIR / "missing_by_month.csv", index=False, encoding="utf-8-sig"
    )

    missing["_code_prefix"] = missing["_best_key"].map(code_prefix)
    prefix_df = missing.groupby("_code_prefix").agg(row_count=("_score_row_id", "size"), unique_symbol_count=("_best_key", "nunique")).reset_index()
    prefix_rows = [
        {
            "code_prefix": row["_code_prefix"],
            "row_count": int(row["row_count"]),
            "unique_symbol_count": int(row["unique_symbol_count"]),
            "notes": "A-share-like prefix" if str(row["_code_prefix"]) in {"00", "30", "60", "68", "83", "87", "43", "92"} else "",
        }
        for _, row in prefix_df.sort_values("row_count", ascending=False).iterrows()
    ]
    write_csv(OUT_DIR / "missing_by_code_prefix.csv", prefix_rows, ["code_prefix", "row_count", "unique_symbol_count", "notes"])

    best_cov = float(best["join_coverage_ratio"])
    best_missing_unique = int(best["missing_unique_symbols"])
    duplicate_join_detected = bool(best["duplicate_join_detected"])
    former_code_used = best_join_method == "former_code_assisted_join"
    exact_cov = float(method_rows[0]["join_coverage_ratio"])
    strip_cov = float(method_rows[1]["join_coverage_ratio"])
    symbol_format_issue_detected = strip_cov > exact_cov + 0.01
    current_listed_only_source_suspected = best_cov < 0.95 and bool(missing_by_month_df.sort_index()["missing_ratio"].head(12).mean() > missing_by_month_df["missing_ratio"].tail(12).mean())
    delisted_or_historical_buffer_missing_suspected = best_cov < 0.95 and any(row["missing_reason_guess"] == "DELISTED_OR_HISTORICAL_BUFFER_MISSING" for row in missing_profile_rows[:500])
    score_panel_non_a_share_or_synthetic_symbols_suspected = any(
        row["missing_reason_guess"] == "SCORE_PANEL_HAS_NON_A_SHARE_OR_SYNTHETIC_SYMBOLS" for row in missing_profile_rows[:500]
    )
    whether_former_code_needed = float(method_rows[3]["join_coverage_ratio"]) > strip_cov + 0.01

    cause_guesses = []
    if symbol_format_issue_detected:
        cause_guesses.append("SYMBOL_FORMAT_MISMATCH")
    if current_listed_only_source_suspected:
        cause_guesses.append("CURRENT_LISTED_ONLY_SOURCE")
    if delisted_or_historical_buffer_missing_suspected:
        cause_guesses.append("DELISTED_OR_HISTORICAL_BUFFER_MISSING")
    if whether_former_code_needed:
        cause_guesses.append("FORMER_CODE_MAPPING_NEEDED")
    if score_panel_non_a_share_or_synthetic_symbols_suspected:
        cause_guesses.append("SCORE_PANEL_HAS_NON_A_SHARE_OR_SYNTHETIC_SYMBOLS")
    if not cause_guesses:
        cause_guesses.append("UNKNOWN")
    primary_cause = cause_guesses[0]

    if best_cov >= 0.95 and not duplicate_join_detected:
        final_decision = "TRD_CO_STATIC_INDUSTRY_JOIN_FORENSICS_READY_TO_RERUN_NEUTRAL_SCORE_WITH_FIXED_JOIN"
        recommended_next_step = "重跑 static neutral score run，采用最佳 join key normalization。"
    elif best_cov >= 0.80:
        final_decision = "TRD_CO_STATIC_INDUSTRY_JOIN_FORENSICS_WATCH_PARTIAL_COVERAGE"
        recommended_next_step = "可做 WATCH 版本 neutral score，但需人工确认 universe loss。"
    else:
        final_decision = "TRD_CO_STATIC_INDUSTRY_JOIN_FORENSICS_FAIL_COVERAGE_STILL_LOW"
        recommended_next_step = "不应生成 neutral score；先补历史/退市行业源或修复 symbol coverage。"

    safe_to_generate = final_decision == "TRD_CO_STATIC_INDUSTRY_JOIN_FORENSICS_READY_TO_RERUN_NEUTRAL_SCORE_WITH_FIXED_JOIN"

    policy = {
        "best_join_method": best_join_method,
        "best_join_coverage_ratio": best_cov,
        "recommended_score_symbol_normalization": "strip .SH/.SZ/.BJ suffix and left-pad numeric code to 6 digits" if best_join_method != "exact_symbol_join" else "exact string",
        "recommended_trd_co_symbol_normalization": "strip whitespace/extract numeric code and left-pad to 6 digits",
        "former_code_used": former_code_used,
        "duplicate_join_policy": "reject one-to-many joins; deduplicate source keys only for forensic comparison",
        "safe_to_use_for_neutral_score": safe_to_generate,
        "limitations": [
            "This task did not generate neutral scores.",
            "fwd_ret_1m was not used for any calculation.",
            "Coverage below 0.95 blocks ready neutral score rerun.",
        ],
    }
    write_json(OUT_DIR / "best_join_method_policy.json", policy)

    diagnosis = {
        "primary_cause_guess": primary_cause,
        "secondary_cause_guesses": cause_guesses[1:],
        "coverage_after_best_normalization": best_cov,
        "whether_symbol_format_bug": symbol_format_issue_detected,
        "whether_current_listed_only_suspected": current_listed_only_source_suspected,
        "whether_former_code_needed": whether_former_code_needed,
        "whether_extra_industry_source_needed": best_cov < 0.95,
        "recommended_next_step": recommended_next_step,
    }
    write_json(OUT_DIR / "trd_co_join_coverage_diagnosis.json", diagnosis)

    forbidden_false = {
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
    }
    summary = {
        "run_timestamp": run_timestamp,
        "prerequisites_passed": prerequisites_passed,
        "previous_join_coverage_ratio": previous_join_coverage_ratio,
        "score_panel_rows": score_panel_rows,
        "score_panel_unique_symbols": score_panel_unique_symbols,
        "trd_co_unique_symbols": trd_co_unique_symbols,
        "best_join_method": best_join_method,
        "best_joined_rows": int(best["joined_rows"]),
        "best_join_coverage_ratio": best_cov,
        "best_joined_unique_symbols": int(best["joined_unique_symbols"]),
        "best_missing_unique_symbols": best_missing_unique,
        "duplicate_join_detected": duplicate_join_detected,
        "former_code_used": former_code_used,
        "symbol_format_issue_detected": symbol_format_issue_detected,
        "current_listed_only_source_suspected": current_listed_only_source_suspected,
        "delisted_or_historical_buffer_missing_suspected": delisted_or_historical_buffer_missing_suspected,
        "score_panel_non_a_share_or_synthetic_symbols_suspected": score_panel_non_a_share_or_synthetic_symbols_suspected,
        "safe_to_generate_neutral_score_after_fix": safe_to_generate,
        **forbidden_false,
        "final_decision": final_decision,
        "recommended_next_step": recommended_next_step,
    }
    write_json(OUT_DIR / "trd_co_static_industry_join_forensics_summary.json", summary)

    prereq = {
        "run_timestamp": run_timestamp,
        "prerequisites_passed": prerequisites_passed,
        "required_inputs": prereq_rows,
        "score_panel_columns_read": SCORE_COLUMNS,
        "debug_ingestion_final_decision": debug_summary.get("final_decision"),
    }
    write_json(OUT_DIR / "join_forensics_prerequisite_check.json", prereq)

    report = f"""# TRD_Co Static Industry Join Coverage Forensics v0

## 结论

`{final_decision}`

## 覆盖率

- previous_join_coverage_ratio: `{previous_join_coverage_ratio:.6f}`
- best_join_method: `{best_join_method}`
- best_join_coverage_ratio: `{best_cov:.6f}`
- best_joined_rows: `{int(best["joined_rows"])}`
- best_missing_unique_symbols: `{best_missing_unique}`

## 原因判断

- primary_cause_guess: `{primary_cause}`
- secondary_cause_guesses: `{", ".join(cause_guesses[1:])}`
- symbol_format_issue_detected: `{symbol_format_issue_detected}`
- current_listed_only_source_suspected: `{current_listed_only_source_suspected}`
- delisted_or_historical_buffer_missing_suspected: `{delisted_or_historical_buffer_missing_suspected}`

本任务没有生成 neutral score，没有计算 IC、D10-D1、收益、回测、交易成本、换手、Sharpe、MaxDD、benchmark-relative return、alpha/beta，没有训练、调参、SHAP 或写 production。
"""
    (OUT_DIR / "trd_co_static_industry_join_forensics_report.md").write_text(report, encoding="utf-8")

    terminal_summary = {
        "task_name": TASK_NAME,
        "run_timestamp": run_timestamp,
        "stdout_log": rel(RUN_DIR / "run_stdout.txt"),
        "stderr_log": rel(RUN_DIR / "run_stderr.txt"),
        "output_directory": rel(OUT_DIR),
        "final_decision": final_decision,
        "exit_code": 0,
    }
    write_json(OUT_DIR / "terminal_summary.json", terminal_summary)
    write_csv(
        OUT_DIR / "final_qa.csv",
        [
            {"check": "prerequisites_passed", "passed": prerequisites_passed, "notes": ""},
            {"check": "neutral_score_not_generated", "passed": True, "notes": ""},
            {"check": "no_forbidden_calculations", "passed": True, "notes": "No IC/return/backtest/training/production."},
            {"check": "best_join_policy_written", "passed": True, "notes": str(OUT_DIR / "best_join_method_policy.json")},
        ],
        ["check", "passed", "notes"],
    )
    (OUT_DIR / "task_completion_card.md").write_text(
        f"""# Task Completion Card

- task_name: `{TASK_NAME}`
- final_decision: `{final_decision}`
- output_directory: `{rel(OUT_DIR)}`
- neutral_score_generated: `False`
- production_modified: `False`
""",
        encoding="utf-8",
    )
    (RUN_DIR / "RUN_STATE.md").write_text(
        f"""# RUN_STATE

任务：{TASK_NAME}
状态：完成

输出目录：
- {OUT_DIR}

final_decision: {final_decision}
best_join_method: {best_join_method}
best_join_coverage_ratio: {best_cov}

禁止项确认：
- 未生成 neutral score
- 未计算 IC / D10-D1 / 收益 / 回测
- 未训练 / 调参 / SHAP
- 未写 production
""",
        encoding="utf-8",
    )

    del cleaned, score_df, matched, missing, missing_by_month_df
    gc.collect()
    print(f"final_decision={final_decision}")
    print(f"best_join_method={best_join_method}")
    print(f"best_join_coverage_ratio={best_cov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
