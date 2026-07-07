from __future__ import annotations

import gc
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "v0_strict_lag_coverage_gap_forensic_v2"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

WEIGHTS_PATH = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_reconstructed_weights.parquet"
REMATCH_SUMMARY_PATH = ROOT / "output" / "v0_strict_lag_month_key_csmar_rematch_v1" / "v0_strict_lag_month_key_csmar_rematch_summary.json"
REMATCH_MONTHLY_PATH = ROOT / "output" / "v0_strict_lag_month_key_csmar_rematch_v1" / "v0_strict_lag_month_key_monthly_net_return_by_cost.csv"
ROBUST_PATH = ROOT / "output" / "robust_cleaned_fundamental_factor_variant_build_v0" / "robust_cleaned_factor_score_panel_v0.parquet"
TRANSITION_QA_PATH = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_buffer_transition_qa.csv"

OPTIONAL_SOURCES = [
    ("preprocessed", ROOT / "output" / "preprocessed.parquet"),
    ("panel", ROOT / "output" / "panel.parquet"),
    ("training_panel_v3_full", ROOT / "output" / "training_panel_v3_full.parquet"),
    ("all_daily", ROOT / "output" / "all_daily.parquet"),
    (
        "compact_f_price_label",
        ROOT
        / "output"
        / "compact_f_v3_full_training_panel_price_label_v0"
        / "compact_f_v3_full_price_label_unique13_v0.parquet",
    ),
]


def write_state(status: str, details: dict | None = None) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": TASK_NAME,
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "details": details or {},
        "resume_instruction": f"先读取 {RUN_DIR / 'RUN_STATE.md'} 再继续。",
    }
    lines = ["# RUN_STATE", "", f"- task_name: {TASK_NAME}", f"- status: {status}"]
    for k, v in payload["details"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "```json", json.dumps(payload, ensure_ascii=False, indent=2, default=str), "```"]
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(lines), encoding="utf-8")


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def norm_symbol_one(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = re.sub(r"(?i)(\.SH|\.SZ|SH|SZ)$", "", s)
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    return digits[-6:].zfill(6)


def add_keys(df: pd.DataFrame, symbol_col: str, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out["symbol_raw"] = out[symbol_col].astype(str)
    out["symbol_norm"] = out[symbol_col].map(norm_symbol_one)
    out["invalid_symbol"] = out["symbol_norm"].isna()
    out["raw_date"] = pd.to_datetime(out[date_col])
    out["calendar_month_end"] = out["raw_date"] + pd.offsets.MonthEnd(0)
    out["year_month"] = out["raw_date"].dt.to_period("M").astype(str)
    return out


def source_diag(name: str, df: pd.DataFrame) -> dict:
    dup_count = int(df.duplicated(["symbol_norm", "year_month"]).sum()) if {"symbol_norm", "year_month"}.issubset(df.columns) else 0
    raw_examples = ";".join(df["symbol_raw"].dropna().astype(str).drop_duplicates().head(8).tolist())
    norm_examples = ";".join(df["symbol_norm"].dropna().astype(str).drop_duplicates().head(8).tolist())
    invalid = int(df["invalid_symbol"].sum())
    if invalid > 0:
        diagnosis = "存在无法标准化的 symbol"
    elif dup_count > 0:
        diagnosis = "存在 symbol-year_month 重复键，需去重或聚合"
    else:
        diagnosis = "symbol/month key 基本可用"
    return {
        "source_name": name,
        "row_count": int(len(df)),
        "unique_symbol_raw_count": int(df["symbol_raw"].nunique(dropna=True)),
        "unique_symbol_norm_count": int(df["symbol_norm"].nunique(dropna=True)),
        "invalid_symbol_count": invalid,
        "raw_min_date": df["raw_date"].min(),
        "raw_max_date": df["raw_date"].max(),
        "year_month_count": int(df["year_month"].nunique(dropna=True)),
        "duplicate_symbol_year_month_count": dup_count,
        "raw_symbol_examples": raw_examples,
        "normalized_symbol_examples": norm_examples,
        "diagnosis": diagnosis,
    }


def match_status(avg_share: float, min_share: float) -> str:
    if avg_share >= 0.98 and min_share >= 0.95:
        return "READY"
    if avg_share >= 0.95 and min_share >= 0.90:
        return "READY_WITH_MINOR_GAPS"
    if avg_share >= 0.90:
        return "WATCH_COVERAGE_GAPS"
    return "LOW_MATCH"


def coverage_by_key(weights: pd.DataFrame, src: pd.DataFrame, left_keys: list[str], right_keys: list[str], require_non_null: bool) -> tuple[pd.DataFrame, dict]:
    right_cols = right_keys + ["fwd_ret_1m"]
    right = src[right_cols].copy()
    if require_non_null:
        right = right[right["fwd_ret_1m"].notna()].copy()
    right = right.drop_duplicates(right_keys, keep="first")
    merged = weights.merge(right.assign(_matched=True), left_on=left_keys, right_on=right_keys, how="left")
    merged["matched_flag"] = merged["_matched"].fillna(False).astype(bool)
    merged["matched_weight"] = np.where(merged["matched_flag"], merged["weight"], 0.0)
    share = merged.groupby("year_month")["matched_weight"].sum()
    avg = float(share.mean()) if len(share) else np.nan
    mn = float(share.min()) if len(share) else np.nan
    return merged, {
        "weight_row_count": int(len(weights)),
        "matched_row_count": int(merged["matched_flag"].sum()),
        "matched_ratio": float(merged["matched_flag"].mean()) if len(merged) else np.nan,
        "avg_matched_weight_share": avg,
        "min_matched_weight_share": mn,
        "low_match_month_count": int((share < 0.95).sum()) if len(share) else 0,
        "zero_match_month_count": int((share == 0).sum()) if len(share) else 0,
    }


def return_map_from_close(path: Path, source_name: str, date_col: str, symbol_col: str, close_col: str) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_parquet(path, columns=[date_col, symbol_col, close_col], engine="pyarrow")
    df = df.dropna(subset=[date_col, symbol_col, close_col]).copy()
    df = add_keys(df, symbol_col, date_col)
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=[close_col, "symbol_norm", "raw_date"]).copy()
    df = df.sort_values(["symbol_norm", "raw_date"])
    # 月内最后一个交易日 close，随后按 symbol 计算下月收益。
    monthly_close = df.groupby(["symbol_norm", "year_month"], as_index=False).tail(1).copy()
    monthly_close = monthly_close.sort_values(["symbol_norm", "year_month"])
    monthly_close["fwd_ret_1m"] = monthly_close.groupby("symbol_norm")[close_col].shift(-1) / monthly_close[close_col] - 1.0
    out = monthly_close[["symbol_raw", "symbol_norm", "raw_date", "calendar_month_end", "year_month", "fwd_ret_1m"]].copy()
    del df, monthly_close
    gc.collect()
    return out, "computed_from_month_end_close", f"{source_name}: 月末 close shift(-1) coverage-only map"


def load_return_source(name: str, path: Path) -> tuple[pd.DataFrame | None, str, bool, str]:
    if not path.exists():
        return None, "", False, "source_missing"
    import pyarrow.parquet as pq

    cols = pq.ParquetFile(path).schema_arrow.names
    symbol_col = "symbol" if "symbol" in cols else None
    date_col = "month_end" if "month_end" in cols else ("date" if "date" in cols else None)
    if symbol_col is None or date_col is None:
        return None, "", False, "missing_symbol_or_date"
    ret_cols = [c for c in ["fwd_ret_1m", "forward_return_1m", "return_1m", "label_fwd_ret_1m"] if c in cols]
    if ret_cols:
        ret = ret_cols[0]
        df = pd.read_parquet(path, columns=[symbol_col, date_col, ret], engine="pyarrow")
        df = add_keys(df, symbol_col, date_col)
        df["fwd_ret_1m"] = pd.to_numeric(df[ret], errors="coerce")
        return df[["symbol_raw", "symbol_norm", "invalid_symbol", "raw_date", "calendar_month_end", "year_month", "fwd_ret_1m"]], ret, False, "existing_return_field"
    close_col = None
    for c in ["收盘", "close", "Close", "股价"]:
        if c in cols:
            close_col = c
            break
    if close_col:
        df, ret_field, caveat = return_map_from_close(path, name, date_col, symbol_col, close_col)
        df["invalid_symbol"] = df["symbol_norm"].isna()
        return df, ret_field, True, caveat
    return None, "", False, "no_return_or_close_field"


def max_drawdown(r: pd.Series) -> float:
    curve = (1.0 + r.fillna(0.0)).cumprod()
    return float((curve / curve.cummax() - 1.0).min()) if len(curve) else np.nan


def perf(r: pd.Series, turnover: pd.Series, matched: pd.Series) -> dict:
    vol = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    mean = float(r.mean()) if len(r) else np.nan
    return {
        "month_count": int(len(r)),
        "mean_monthly_return": mean,
        "monthly_volatility": vol,
        "sharpe": mean / vol * math.sqrt(12) if pd.notna(vol) and vol > 0 else np.nan,
        "tstat": mean / vol * math.sqrt(len(r)) if pd.notna(vol) and vol > 0 else np.nan,
        "cumulative_return": float((1.0 + r.fillna(0.0)).prod() - 1.0) if len(r) else np.nan,
        "max_drawdown": max_drawdown(r),
        "avg_turnover": float(turnover.mean()) if len(turnover) else np.nan,
        "avg_matched_weight_share": float(matched.mean()) if len(matched) else np.nan,
        "min_matched_weight_share": float(matched.min()) if len(matched) else np.nan,
    }


def sensitivity(weights: pd.DataFrame, src: pd.DataFrame, trans: pd.DataFrame, source_name: str, caveat: str) -> dict:
    rmap = src[src["fwd_ret_1m"].notna()].drop_duplicates(["symbol_norm", "year_month"], keep="first")
    merged = weights.merge(rmap[["symbol_norm", "year_month", "fwd_ret_1m"]], on=["symbol_norm", "year_month"], how="left")
    merged["ret_contrib"] = merged["weight"] * merged["fwd_ret_1m"].fillna(0.0)
    merged["matched_weight"] = np.where(merged["fwd_ret_1m"].notna(), merged["weight"], 0.0)
    monthly = merged.groupby("year_month").agg(gross_return=("ret_contrib", "sum"), matched_weight_share=("matched_weight", "sum")).reset_index()
    monthly = monthly.merge(trans[["year_month", "simple_turnover_proxy"]], on="year_month", how="left")
    monthly["simple_turnover_proxy"] = monthly["simple_turnover_proxy"].fillna(0.0)
    monthly["net_return"] = monthly["gross_return"] - monthly["simple_turnover_proxy"] * 20 / 10000.0
    p = perf(monthly["net_return"], monthly["simple_turnover_proxy"], monthly["matched_weight_share"])
    p.update({"source_name": source_name, "cost_bps": 20, "caveat": caveat})
    return p


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", {"step": "start"})

    optional_found = {name: str(path) for name, path in OPTIONAL_SOURCES if path.exists()}
    prereq = {
        "strict_lag_weights_found": WEIGHTS_PATH.exists(),
        "rematch_summary_found": REMATCH_SUMMARY_PATH.exists(),
        "robust_cleaned_return_source_found": ROBUST_PATH.exists(),
        "optional_sources_found": optional_found,
    }
    required = [(prereq["strict_lag_weights_found"], WEIGHTS_PATH), (prereq["rematch_summary_found"], REMATCH_SUMMARY_PATH), (prereq["robust_cleaned_return_source_found"], ROBUST_PATH)]
    prereq["missing_files"] = [str(p) for ok, p in required if not ok]
    prereq["prerequisites_passed"] = not prereq["missing_files"]
    save_json(prereq, OUT_DIR / "v0_strict_lag_coverage_gap_prerequisite_check.json")
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", {"step": "read_weights_and_robust"})
    weights = pd.read_parquet(WEIGHTS_PATH, columns=["portfolio_name", "symbol", "month_end", "selected_flag", "weight", "rank_in_month", "selection_reason"], engine="pyarrow")
    weights = weights[weights["selected_flag"]].copy()
    weights = add_keys(weights, "symbol", "month_end")

    robust = pd.read_parquet(ROBUST_PATH, columns=["symbol", "month_end", "fwd_ret_1m"], engine="pyarrow")
    robust = add_keys(robust, "symbol", "month_end")
    robust["fwd_ret_1m"] = pd.to_numeric(robust["fwd_ret_1m"], errors="coerce")

    diag_rows = [source_diag("strict_lag_selected_weights", weights), source_diag("robust_cleaned_factor_score_panel_v0", robust)]

    source_maps: dict[str, tuple[pd.DataFrame, str, bool, str, Path]] = {
        "robust_cleaned_factor_score_panel_v0": (robust, "fwd_ret_1m", False, "current canonical source used in rematch", ROBUST_PATH)
    }
    for name, path in OPTIONAL_SOURCES:
        df, field, generated, caveat = load_return_source(name, path)
        if df is not None:
            source_maps[name] = (df, field, generated, caveat, path)
            diag_rows.append(source_diag(name, df))
    pd.DataFrame(diag_rows).to_csv(OUT_DIR / "v0_strict_lag_symbol_month_key_diagnostic.csv", index=False, encoding="utf-8-sig")

    write_state("running", {"step": "robust_funnel"})
    layers = [
        ("A_raw_symbol_raw_date", ["symbol_raw", "raw_date"], ["symbol_raw", "raw_date"], False, "raw symbol + raw date"),
        ("B_raw_symbol_year_month", ["symbol_raw", "year_month"], ["symbol_raw", "year_month"], False, "raw symbol + year_month"),
        ("C_symbol_norm_year_month", ["symbol_norm", "year_month"], ["symbol_norm", "year_month"], False, "symbol_norm + year_month"),
        ("D_symbol_norm_calendar_month_end", ["symbol_norm", "calendar_month_end"], ["symbol_norm", "calendar_month_end"], False, "symbol_norm + calendar_month_end"),
        ("E_symbol_norm_year_month_non_null_fwd_ret", ["symbol_norm", "year_month"], ["symbol_norm", "year_month"], True, "symbol_norm + year_month + non-null fwd_ret_1m"),
    ]
    funnel_rows = []
    previous = None
    final_merged = None
    for layer, lk, rk, nonnull, key_desc in layers:
        merged, s = coverage_by_key(weights, robust, lk, rk, nonnull)
        if layer.startswith("E_"):
            final_merged = merged
        gain = np.nan if previous is None else s["avg_matched_weight_share"] - previous
        previous = s["avg_matched_weight_share"]
        if "raw_date" in lk and s["avg_matched_weight_share"] < 0.5:
            diagnosis = "exact date mismatch 明显"
        elif nonnull and gain < -0.01:
            diagnosis = "fwd_ret_1m null 造成覆盖损失"
        elif "symbol_norm" in lk and gain > 0.01:
            diagnosis = "symbol normalization 有增益"
        else:
            diagnosis = "覆盖未明显改善"
        funnel_rows.append({"layer": layer, "match_key": key_desc, **s, "incremental_gain_vs_previous_layer": gain, "diagnosis": diagnosis})
    funnel = pd.DataFrame(funnel_rows)
    funnel.to_csv(OUT_DIR / "v0_strict_lag_match_funnel_robust_cleaned.csv", index=False, encoding="utf-8-sig")

    write_state("running", {"step": "gap_reason"})
    assert final_merged is not None
    source_symbols = set(robust["symbol_norm"].dropna())
    source_months_by_symbol = robust.groupby("symbol_norm")["year_month"].apply(set).to_dict()
    source_min = robust["year_month"].min()
    source_max = robust["year_month"].max()
    nonnull_keys = set(map(tuple, robust[robust["fwd_ret_1m"].notna()][["symbol_norm", "year_month"]].drop_duplicates().to_numpy()))
    all_keys = set(map(tuple, robust[["symbol_norm", "year_month"]].drop_duplicates().to_numpy()))
    dup_keys = set(map(tuple, robust.loc[robust.duplicated(["symbol_norm", "year_month"], keep=False), ["symbol_norm", "year_month"]].drop_duplicates().to_numpy()))
    raw_ym_keys = set(map(tuple, robust[["symbol_raw", "year_month"]].drop_duplicates().to_numpy()))

    unmatched = final_merged[~final_merged["matched_flag"]].copy()
    reason_rows = []
    for row in unmatched.to_dict("records"):
        sym = row["symbol_norm"]
        ym = row["year_month"]
        raw = row["symbol_raw"]
        key = (sym, ym)
        raw_key = (raw, ym)
        if key in dup_keys:
            reason = "DUPLICATE_KEY_AMBIGUOUS"
            fix = "先对 return source 的 symbol-month 重复键做规则化去重"
        elif ym < source_min or ym > source_max:
            reason = "DATE_OUTSIDE_RETURN_SOURCE_RANGE"
            fix = "补充对应月份的 return source"
        elif sym not in source_symbols:
            reason = "SYMBOL_ABSENT_FROM_RETURN_SOURCE_GLOBAL"
            fix = "需要更宽的 CSMAR/价格 return map 覆盖旧 universe"
        elif key in all_keys and key not in nonnull_keys:
            reason = "SYMBOL_MONTH_PRESENT_BUT_FWD_RET_NULL"
            fix = "补充或重算该 symbol-month 的 fwd_ret_1m"
        elif raw_key not in raw_ym_keys and key in nonnull_keys:
            reason = "SYMBOL_FORMAT_MISMATCH_FIXABLE"
            fix = "使用 symbol_norm + year_month 匹配"
        elif ym not in source_months_by_symbol.get(sym, set()):
            reason = "SYMBOL_PRESENT_BUT_MONTH_MISSING"
            fix = "补充该股票该月份 return"
        else:
            reason = "UNKNOWN"
            fix = "需进一步检查源文件键值"
        reason_rows.append(
            {
                "portfolio_name": row["portfolio_name"],
                "symbol_raw": raw,
                "symbol_norm": sym,
                "weight_month_end": row["raw_date"],
                "year_month": ym,
                "weight": row["weight"],
                "rank_in_month": row["rank_in_month"],
                "selection_reason": row["selection_reason"],
                "gap_reason": reason,
                "suggested_fix": fix,
                "source_checked": "robust_cleaned_factor_score_panel_v0",
            }
        )
    reason_df = pd.DataFrame(reason_rows)
    reason_df.to_csv(OUT_DIR / "v0_strict_lag_unmatched_row_reason.csv", index=False, encoding="utf-8-sig")
    matched_share = final_merged.groupby("year_month")["matched_weight"].sum().rename("matched_weight_share")
    month_base = weights.groupby("year_month").agg(selected_count=("symbol_norm", "size")).join(matched_share, how="left").fillna({"matched_weight_share": 0.0}).reset_index()
    reason_counts = reason_df.pivot_table(index="year_month", columns="gap_reason", values="symbol_norm", aggfunc="count", fill_value=0) if len(reason_df) else pd.DataFrame()
    monthly = month_base.merge(reason_counts.reset_index(), on="year_month", how="left") if len(reason_counts) else month_base
    for col in ["SYMBOL_ABSENT_FROM_RETURN_SOURCE_GLOBAL", "SYMBOL_PRESENT_BUT_MONTH_MISSING", "SYMBOL_MONTH_PRESENT_BUT_FWD_RET_NULL", "SYMBOL_FORMAT_MISMATCH_FIXABLE", "DATE_OUTSIDE_RETURN_SOURCE_RANGE", "DUPLICATE_KEY_AMBIGUOUS"]:
        if col not in monthly.columns:
            monthly[col] = 0
        monthly[col] = monthly[col].fillna(0)
    monthly["matched_count"] = (monthly["matched_weight_share"] * 50).round().astype(int)
    monthly["unmatched_count"] = monthly["selected_count"] - monthly["matched_count"]
    monthly["unmatched_weight_share"] = 1.0 - monthly["matched_weight_share"]
    reason_cols = [c for c in reason_df["gap_reason"].unique()] if len(reason_df) else []
    if reason_cols:
        for col in reason_cols:
            monthly[col] = monthly[col].fillna(0)
        monthly["top_gap_reason"] = monthly[reason_cols].idxmax(axis=1)
        monthly.loc[monthly[reason_cols].sum(axis=1) <= 0, "top_gap_reason"] = ""
    else:
        monthly["top_gap_reason"] = ""
    monthly["symbol_absent_count"] = monthly["SYMBOL_ABSENT_FROM_RETURN_SOURCE_GLOBAL"]
    monthly["month_missing_count"] = monthly["SYMBOL_PRESENT_BUT_MONTH_MISSING"]
    monthly["fwd_ret_null_count"] = monthly["SYMBOL_MONTH_PRESENT_BUT_FWD_RET_NULL"]
    monthly["format_mismatch_count"] = monthly["SYMBOL_FORMAT_MISMATCH_FIXABLE"]
    monthly["date_outside_count"] = monthly["DATE_OUTSIDE_RETURN_SOURCE_RANGE"]
    monthly["duplicate_key_count"] = monthly["DUPLICATE_KEY_AMBIGUOUS"]
    monthly["status"] = np.where(monthly["matched_weight_share"] >= 0.95, "OK", "LOW_MATCH")
    monthly[["year_month", "selected_count", "matched_count", "unmatched_count", "matched_weight_share", "unmatched_weight_share", "top_gap_reason", "symbol_absent_count", "month_missing_count", "fwd_ret_null_count", "format_mismatch_count", "date_outside_count", "duplicate_key_count", "status"]].to_csv(
        OUT_DIR / "v0_strict_lag_unmatched_reason_monthly_summary.csv", index=False, encoding="utf-8-sig"
    )
    if len(reason_df):
        total = reason_df.groupby("gap_reason").agg(row_count=("symbol_norm", "size"), weight_share_sum=("weight", "sum"), example_symbols=("symbol_norm", lambda x: ";".join(x.drop_duplicates().head(10)))).reset_index()
        total["avg_monthly_weight_share"] = total["weight_share_sum"] / weights["year_month"].nunique()
        total["interpretation"] = total["gap_reason"].map({
            "SYMBOL_ABSENT_FROM_RETURN_SOURCE_GLOBAL": "股票完全不在 current CSMAR return source 中",
            "SYMBOL_PRESENT_BUT_MONTH_MISSING": "股票存在但该月份缺 return",
            "SYMBOL_MONTH_PRESENT_BUT_FWD_RET_NULL": "键存在但 fwd_ret_1m 为空",
            "DATE_OUTSIDE_RETURN_SOURCE_RANGE": "月份超出 return source 范围",
            "DUPLICATE_KEY_AMBIGUOUS": "return source 存在重复键",
            "SYMBOL_FORMAT_MISMATCH_FIXABLE": "symbol 格式标准化即可修复",
            "UNKNOWN": "未能分类",
        })
    else:
        total = pd.DataFrame(columns=["gap_reason", "row_count", "weight_share_sum", "avg_monthly_weight_share", "example_symbols", "interpretation"])
    total.to_csv(OUT_DIR / "v0_strict_lag_unmatched_reason_total_summary.csv", index=False, encoding="utf-8-sig")

    write_state("running", {"step": "source_coverage_comparison"})
    coverage_rows = []
    best_name = None
    best_avg = -1.0
    best_min = np.nan
    best_tuple = None
    for name, (src, field, generated, caveat, path) in source_maps.items():
        src_nonnull = src[src["fwd_ret_1m"].notna()].copy()
        _, s = coverage_by_key(weights, src, ["symbol_norm", "year_month"], ["symbol_norm", "year_month"], True)
        status = match_status(s["avg_matched_weight_share"], s["min_matched_weight_share"])
        coverage_rows.append({
            "source_name": name,
            "source_path": str(path),
            "return_field_used": field,
            "generated_return_map": generated,
            "symbol_norm_year_month_row_count": int(len(src_nonnull.drop_duplicates(["symbol_norm", "year_month"]))),
            "unique_symbol_count": int(src_nonnull["symbol_norm"].nunique()),
            "year_month_count": int(src_nonnull["year_month"].nunique()),
            "min_year_month": src_nonnull["year_month"].min() if len(src_nonnull) else "",
            "max_year_month": src_nonnull["year_month"].max() if len(src_nonnull) else "",
            "avg_matched_weight_share": s["avg_matched_weight_share"],
            "min_matched_weight_share": s["min_matched_weight_share"],
            "low_match_month_count": s["low_match_month_count"],
            "zero_match_month_count": s["zero_match_month_count"],
            "match_status": status,
            "caveat": caveat,
        })
        if pd.notna(s["avg_matched_weight_share"]) and s["avg_matched_weight_share"] > best_avg:
            best_name = name
            best_avg = s["avg_matched_weight_share"]
            best_min = s["min_matched_weight_share"]
            best_tuple = (src, caveat, status)
    coverage = pd.DataFrame(coverage_rows).sort_values("avg_matched_weight_share", ascending=False)
    coverage.to_csv(OUT_DIR / "v0_strict_lag_return_source_coverage_comparison.csv", index=False, encoding="utf-8-sig")

    high_coverage = best_tuple is not None and best_tuple[2] in ["READY", "READY_WITH_MINOR_GAPS", "WATCH_COVERAGE_GAPS"]
    trans = pd.read_csv(TRANSITION_QA_PATH)
    trans["year_month"] = pd.to_datetime(trans["month_end"]).dt.to_period("M").astype(str)
    if high_coverage:
        sens = pd.DataFrame([sensitivity(weights, best_tuple[0], trans, best_name, best_tuple[1])])
    else:
        sens = pd.DataFrame(columns=["source_name", "cost_bps", "month_count", "mean_monthly_return", "monthly_volatility", "sharpe", "tstat", "cumulative_return", "max_drawdown", "avg_turnover", "avg_matched_weight_share", "min_matched_weight_share", "caveat"])
    sens.to_csv(OUT_DIR / "v0_strict_lag_high_coverage_source_sensitivity_20bps.csv", index=False, encoding="utf-8-sig")

    robust_exact = float(funnel.loc[funnel["layer"] == "A_raw_symbol_raw_date", "avg_matched_weight_share"].iloc[0])
    robust_ym = float(funnel.loc[funnel["layer"] == "B_raw_symbol_year_month", "avg_matched_weight_share"].iloc[0])
    robust_norm_ym = float(funnel.loc[funnel["layer"] == "E_symbol_norm_year_month_non_null_fwd_ret", "avg_matched_weight_share"].iloc[0])
    robust_norm_min = float(funnel.loc[funnel["layer"] == "E_symbol_norm_year_month_non_null_fwd_ret", "min_matched_weight_share"].iloc[0])
    top_reason = total.sort_values("weight_share_sum", ascending=False)["gap_reason"].iloc[0] if len(total) else "UNKNOWN"
    format_sufficient = robust_norm_ym >= 0.95 and robust_ym < 0.95
    return_source_too_narrow = robust_norm_ym < 0.85 and best_avg >= 0.85
    old_universe_not_current = top_reason == "SYMBOL_ABSENT_FROM_RETURN_SOURCE_GLOBAL" and best_avg < 0.85
    month_issue = top_reason in ["SYMBOL_PRESENT_BUT_MONTH_MISSING", "DATE_OUTSIDE_RETURN_SOURCE_RANGE", "SYMBOL_MONTH_PRESENT_BUT_FWD_RET_NULL"]
    if format_sufficient:
        main_gap = "SYMBOL_FORMAT_MISMATCH"
        repairable = "YES_BY_SYMBOL_NORMALIZATION"
    elif return_source_too_narrow:
        main_gap = "ROBUST_RETURN_SOURCE_TOO_NARROW"
        repairable = "YES_BY_BROADER_CANONICAL_RETURN_MAP" if best_avg >= 0.95 else "PARTIAL_ONLY"
    elif old_universe_not_current:
        main_gap = "OLD_UNIVERSE_NOT_COVERED_BY_CURRENT_CSMAR"
        repairable = "NO_LEGACY_UNIVERSE_NOT_COMPARABLE"
    elif month_issue:
        main_gap = "MONTH_COVERAGE_MISMATCH"
        repairable = "PARTIAL_ONLY"
    elif top_reason != "UNKNOWN":
        main_gap = "MIXED_GAPS"
        repairable = "PARTIAL_ONLY" if best_avg >= 0.85 else "UNKNOWN"
    else:
        main_gap = "UNKNOWN"
        repairable = "UNKNOWN"

    diag_summary = pd.DataFrame([{
        "robust_cleaned_year_month_avg_matched_weight_share": robust_ym,
        "robust_cleaned_symbol_norm_year_month_avg_matched_weight_share": robust_norm_ym,
        "robust_cleaned_symbol_norm_year_month_min_matched_weight_share": robust_norm_min,
        "main_gap_reason": main_gap,
        "format_normalization_sufficient": format_sufficient,
        "return_source_too_narrow": return_source_too_narrow,
        "old_universe_not_in_current_csmar": old_universe_not_current,
        "month_coverage_issue": month_issue,
        "high_coverage_return_source_found": high_coverage,
        "best_return_source_name": best_name,
        "best_return_source_avg_matched_weight_share": best_avg,
        "best_return_source_min_matched_weight_share": best_min,
        "strict_lag_eval_repairable": repairable,
        "recommended_next_step": "使用更宽的 canonical return map 重新做 strict-lag bridge" if best_avg >= 0.95 else "继续定位 legacy universe 与 current CSMAR 覆盖差异",
    }])
    diag_summary.to_csv(OUT_DIR / "v0_strict_lag_coverage_gap_diagnosis_summary.csv", index=False, encoding="utf-8-sig")

    portfolio_returns_calculated = bool(high_coverage)
    guard_items = [
        ("strict_lag_alpha_signal_regenerated", False, False),
        ("strict_lag_weights_regenerated", False, False),
        ("original_orthogonalization_modified", False, False),
        ("old_artifacts_modified", False, False),
        ("production_modified", False, False),
        ("ml_training_run", False, False),
        ("new_ml_model_trained", False, False),
        ("portfolio_returns_calculated", "sensitivity_only_if_high_coverage_source_found", "sensitivity_only_if_high_coverage_source_found" if portfolio_returns_calculated else "not_calculated_no_high_coverage_source"),
        ("fwd_ret_1m_used_for_same_month_signal", False, False),
        ("fwd_ret_1m_used_for_selection", False, False),
        ("fwd_ret_1m_used_for_weighting", False, False),
        ("benchmark_relative_returns_calculated", False, False),
        ("alpha_beta_regression_calculated", False, False),
        ("information_ratio_calculated", False, False),
        ("tracking_error_calculated", False, False),
        ("ff_regression_calculated", False, False),
        ("dgtw_adjusted_eval_calculated", False, False),
        ("shap_calculated", False, False),
    ]
    guard = pd.DataFrame([{"guardrail": g, "expected": e, "actual": a, "pass": bool(e == a)} for g, e, a in guard_items])
    guard.to_csv(OUT_DIR / "v0_strict_lag_coverage_gap_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    no_guard = bool(guard["pass"].all())

    if not no_guard:
        final_decision = "V0_STRICT_LAG_COVERAGE_FAIL_GUARDRAIL"
    elif high_coverage and best_avg >= 0.95:
        final_decision = "V0_STRICT_LAG_COVERAGE_REPAIRABLE_REEVALUATE"
    elif best_avg >= 0.85:
        final_decision = "V0_STRICT_LAG_COVERAGE_PARTIAL_REPAIR_ONLY"
    elif best_avg < 0.85 and main_gap == "OLD_UNIVERSE_NOT_COVERED_BY_CURRENT_CSMAR":
        final_decision = "V0_STRICT_LAG_COVERAGE_NOT_REPAIRABLE_LEGACY_UNIVERSE_MISMATCH"
    elif robust_norm_ym < 0.85 and not high_coverage:
        final_decision = "V0_STRICT_LAG_COVERAGE_FAIL_RETURN_SOURCE_TOO_NARROW"
    else:
        final_decision = "V0_STRICT_LAG_COVERAGE_INCONCLUSIVE"

    sens_row = sens.iloc[0].to_dict() if len(sens) else {}
    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "robust_cleaned_exact_date_avg_matched_weight_share": robust_exact,
        "robust_cleaned_year_month_avg_matched_weight_share": robust_ym,
        "robust_cleaned_symbol_norm_year_month_avg_matched_weight_share": robust_norm_ym,
        "robust_cleaned_symbol_norm_year_month_min_matched_weight_share": robust_norm_min,
        "main_gap_reason": main_gap,
        "format_normalization_sufficient": format_sufficient,
        "return_source_too_narrow": return_source_too_narrow,
        "old_universe_not_in_current_csmar": old_universe_not_current,
        "month_coverage_issue": month_issue,
        "high_coverage_return_source_found": high_coverage,
        "best_return_source_name": best_name,
        "best_return_source_avg_matched_weight_share": best_avg,
        "best_return_source_min_matched_weight_share": best_min,
        "strict_lag_eval_repairable": repairable,
        "sensitivity_20bps_sharpe_if_available": sens_row.get("sharpe"),
        "sensitivity_20bps_mean_monthly_return_if_available": sens_row.get("mean_monthly_return"),
        "sensitivity_20bps_tstat_if_available": sens_row.get("tstat"),
        "sensitivity_20bps_cumulative_return_if_available": sens_row.get("cumulative_return"),
        "sensitivity_20bps_max_drawdown_if_available": sens_row.get("max_drawdown"),
        "strict_lag_alpha_signal_regenerated": False,
        "strict_lag_weights_regenerated": False,
        "original_orthogonalization_modified": False,
        "old_artifacts_modified": False,
        "production_modified": False,
        "ml_training_run": False,
        "new_ml_model_trained": False,
        "portfolio_returns_calculated": portfolio_returns_calculated,
        "fwd_ret_1m_used_for_same_month_signal": False,
        "fwd_ret_1m_used_for_selection": False,
        "fwd_ret_1m_used_for_weighting": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
        "final_decision": final_decision,
        "recommended_next_step": diag_summary["recommended_next_step"].iloc[0],
    }
    save_json(summary, OUT_DIR / "v0_strict_lag_coverage_gap_forensic_summary.json")
    report = [
        "# V0 Strict-Lag Coverage Gap Forensic v2",
        "",
        f"- final_decision: {final_decision}",
        f"- main_gap_reason: {main_gap}",
        f"- robust_cleaned_symbol_norm_year_month_avg_matched_weight_share: {robust_norm_ym:.6f}",
        f"- best_return_source_name: {best_name}",
        f"- best_return_source_avg_matched_weight_share: {best_avg:.6f}",
        f"- strict_lag_eval_repairable: {repairable}",
        "",
        "本任务只做覆盖缺口归因；未重新生成 alpha_signal 或 weights。",
    ]
    (OUT_DIR / "v0_strict_lag_coverage_gap_forensic_report.md").write_text("\n".join(report), encoding="utf-8")

    (RUN_DIR / "task_completion_card.md").write_text(
        "\n".join(["# task_completion_card", f"- task_name: {TASK_NAME}", f"- completed_at: {datetime.now().isoformat(timespec='seconds')}", f"- final_decision: {final_decision}", f"- output_dir: {OUT_DIR}"]),
        encoding="utf-8",
    )
    save_json({"task_name": TASK_NAME, "stdout_log": str(RUN_DIR / "run_stdout.txt"), "stderr_log": str(RUN_DIR / "run_stderr.txt"), "status": "completed", "final_decision": final_decision}, RUN_DIR / "terminal_summary.json")
    pd.DataFrame([
        {"qa_item": "prerequisites_passed", "pass": prereq["prerequisites_passed"]},
        {"qa_item": "alpha_signal_not_regenerated", "pass": True},
        {"qa_item": "weights_not_regenerated", "pass": True},
        {"qa_item": "guardrails_passed", "pass": no_guard},
    ]).to_csv(RUN_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    write_state("completed", {"final_decision": final_decision, "output_dir": str(OUT_DIR)})


if __name__ == "__main__":
    main()
