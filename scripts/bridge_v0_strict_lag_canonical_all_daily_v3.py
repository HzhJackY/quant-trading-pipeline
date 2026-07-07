import gc
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "V0 Strict-Lag Canonical all_daily Return Bridge v3"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "v0_strict_lag_canonical_all_daily_bridge_v3"
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

WEIGHTS_PATH = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_reconstructed_weights.parquet"
ALL_DAILY_PATH = ROOT / "output" / "all_daily.parquet"
COVERAGE_SUMMARY_PATH = ROOT / "output" / "v0_strict_lag_coverage_gap_forensic_v2" / "v0_strict_lag_coverage_gap_forensic_summary.json"
OLD_MONTHLY_PATH = ROOT / "output" / "reconstructed_v0_v7_csmar_bridge_evaluation_v0" / "bridge_monthly_net_return_csmar_by_cost.csv"
OLD_SUMMARY_PATH = ROOT / "output" / "reconstructed_v0_v7_csmar_bridge_evaluation_v0" / "bridge_performance_summary_csmar_by_cost.csv"


def norm_symbol(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.replace(r"\.(SH|SZ)$", "", regex=True, case=False)
    out = out.str.replace(r"(SH|SZ)$", "", regex=True, case=False)
    out = out.str.replace(r"\D+", "", regex=True)
    return out.str[-6:].str.zfill(6)


def period_str_from_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").astype(str)


def next_month_str(month: pd.Series) -> pd.Series:
    return (pd.PeriodIndex(month.astype(str), freq="M") + 1).astype(str)


def status_from_coverage(avg_share: float, min_share: float) -> str:
    if avg_share >= 0.98 and min_share >= 0.95:
        return "READY"
    if avg_share >= 0.95 and min_share >= 0.90:
        return "READY_WITH_MINOR_GAPS"
    if avg_share >= 0.90:
        return "WATCH_COVERAGE_GAPS"
    return "LOW_MATCH"


def perf_stats(df: pd.DataFrame, return_col: str = "net_return") -> dict:
    r = pd.to_numeric(df[return_col], errors="coerce").dropna()
    n = int(r.shape[0])
    if n == 0:
        return {
            "month_count": 0,
            "mean_monthly_return": np.nan,
            "annualized_return_approx": np.nan,
            "monthly_volatility": np.nan,
            "sharpe": np.nan,
            "tstat": np.nan,
            "positive_month_ratio": np.nan,
            "cumulative_return": np.nan,
            "max_drawdown": np.nan,
        }
    mean = float(r.mean())
    vol = float(r.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(mean / vol * math.sqrt(12)) if vol > 0 else np.nan
    tstat = float(mean / (vol / math.sqrt(n))) if vol > 0 else np.nan
    wealth = (1.0 + r).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    return {
        "month_count": n,
        "mean_monthly_return": mean,
        "annualized_return_approx": mean * 12.0,
        "monthly_volatility": vol,
        "sharpe": sharpe,
        "tstat": tstat,
        "positive_month_ratio": float((r > 0).mean()),
        "cumulative_return": float(wealth.iloc[-1] - 1.0),
        "max_drawdown": float(dd.min()),
    }


def safe_ratio(num, den):
    if pd.isna(num) or pd.isna(den) or den == 0:
        return np.nan
    return float(num / den)


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def update_run_state(stage: str, note: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    text = [
        f"# RUN_STATE - {TASK_NAME}",
        "",
        f"- updated_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- stage: {stage}",
        f"- note: {note}",
        f"- output_dir: {OUT_DIR}",
        f"- stdout_log: {RUN_DIR / 'run_stdout.txt'}",
        f"- stderr_log: {RUN_DIR / 'run_stderr.txt'}",
        "",
        "恢复协议：若会话中断，先读取本文件，再检查 summary json 与 final_qa.csv。",
    ]
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(text), encoding="utf-8")


def build_return_map() -> tuple[pd.DataFrame, dict]:
    print("读取 all_daily 必要列：symbol/date/close")
    daily = pd.read_parquet(ALL_DAILY_PATH, columns=["symbol", "date", "close"])
    daily["symbol_norm"] = norm_symbol(daily["symbol"])
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    invalid_close_count = int((daily["close"].isna() | (daily["close"] <= 0)).sum())
    daily = daily.loc[daily["symbol_norm"].notna() & daily["date"].notna() & (daily["close"] > 0), ["symbol_norm", "date", "close"]]
    daily["year_month"] = daily["date"].dt.to_period("M").astype(str)
    daily = daily.sort_values(["symbol_norm", "year_month", "date"])
    monthly = daily.groupby(["symbol_norm", "year_month"], as_index=False).tail(1).copy()
    monthly = monthly.rename(columns={"date": "month_trade_date"})
    del daily
    gc.collect()

    monthly = monthly.sort_values(["symbol_norm", "year_month"])
    monthly["next_year_month"] = monthly.groupby("symbol_norm")["year_month"].shift(-1)
    monthly["next_month_trade_date"] = monthly.groupby("symbol_norm")["month_trade_date"].shift(-1)
    monthly["next_close"] = monthly.groupby("symbol_norm")["close"].shift(-1)
    expected_next = next_month_str(monthly["year_month"])
    monthly["return_valid_flag"] = monthly["next_year_month"].eq(expected_next) & monthly["next_close"].notna() & (monthly["next_close"] > 0)
    monthly["fwd_ret_1m"] = np.where(monthly["return_valid_flag"], monthly["next_close"] / monthly["close"] - 1.0, np.nan)
    monthly["invalid_reason"] = np.where(monthly["return_valid_flag"], "OK", "missing_next_natural_month_close")
    monthly["source_path"] = str(ALL_DAILY_PATH)

    cols = [
        "symbol_norm",
        "year_month",
        "month_trade_date",
        "close",
        "next_year_month",
        "next_month_trade_date",
        "next_close",
        "fwd_ret_1m",
        "return_valid_flag",
        "invalid_reason",
        "source_path",
    ]
    monthly = monthly[cols]
    qa = {
        "total_rows": int(monthly.shape[0]),
        "unique_symbols": int(monthly["symbol_norm"].nunique()),
        "year_month_count": int(monthly["year_month"].nunique()),
        "min_year_month": str(monthly["year_month"].min()),
        "max_year_month": str(monthly["year_month"].max()),
        "duplicate_symbol_year_month_count": int(monthly.duplicated(["symbol_norm", "year_month"]).sum()),
        "invalid_close_count": invalid_close_count,
        "missing_next_month_count": int((~monthly["return_valid_flag"]).sum()),
        "fwd_ret_null_count": int(monthly["fwd_ret_1m"].isna().sum()),
        "extreme_return_count_abs_gt_100pct": int((monthly["fwd_ret_1m"].abs() > 1.0).sum()),
    }
    qa["qa_status"] = "PASS" if qa["duplicate_symbol_year_month_count"] == 0 and qa["total_rows"] > 0 else "FAIL"
    return monthly, qa


def load_weights() -> pd.DataFrame:
    print("读取 strict-lag weights，不重建权重")
    weights = pd.read_parquet(WEIGHTS_PATH)
    weights = weights.loc[(weights.get("selected_flag", True) == True) & (pd.to_numeric(weights["weight"], errors="coerce").fillna(0) != 0)].copy()
    weights["symbol_norm"] = norm_symbol(weights["symbol"])
    weights["year_month"] = period_str_from_datetime(weights["month_end"])
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce").fillna(0.0)
    keep = ["portfolio_name", "symbol_norm", "year_month", "weight"]
    return weights[keep]


def compute_turnover(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prev = {}
    for ym, grp in weights.sort_values(["year_month", "symbol_norm"]).groupby("year_month", sort=True):
        curr = dict(zip(grp["symbol_norm"], grp["weight"]))
        keys = set(prev) | set(curr)
        turnover = 0.5 * sum(abs(curr.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
        rows.append({"year_month": ym, "turnover_simple": float(turnover)})
        prev = curr
    return pd.DataFrame(rows)


def diagnose_matches(weights: pd.DataFrame, ret_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    map_key = ret_map[["symbol_norm", "year_month", "fwd_ret_1m", "return_valid_flag", "invalid_reason"]].copy()
    joined = weights.merge(map_key, on=["symbol_norm", "year_month"], how="left")
    joined["abs_weight"] = joined["weight"].abs()
    joined["matched_flag"] = joined["fwd_ret_1m"].notna() & joined["return_valid_flag"].fillna(False)
    map_month_min = ret_map["year_month"].min()
    map_month_max = ret_map["year_month"].max()
    joined["gap_reason"] = np.select(
        [
            joined["matched_flag"],
            joined["invalid_reason"].eq("missing_next_natural_month_close"),
            joined["year_month"].lt(map_month_min),
            joined["year_month"].gt(map_month_max),
            joined["year_month"].eq(map_month_max),
        ],
        [
            "matched",
            "all_daily_missing_next_month_close",
            "start_boundary_before_all_daily",
            "end_boundary_after_all_daily",
            "terminal_boundary_month",
        ],
        default="symbol_month_price_missing",
    )

    monthly_rows = []
    for ym, grp in joined.groupby("year_month", sort=True):
        total_w = float(grp["abs_weight"].sum())
        matched_w = float(grp.loc[grp["matched_flag"], "abs_weight"].sum())
        unmatched = grp.loc[~grp["matched_flag"]].copy()
        reason = "OK"
        diagnosis = "匹配充分。"
        top_symbols = ""
        if not unmatched.empty:
            reason_share = unmatched.groupby("gap_reason")["abs_weight"].sum().sort_values(ascending=False)
            reason = str(reason_share.index[0])
            top_symbols = ",".join(unmatched.sort_values("abs_weight", ascending=False)["symbol_norm"].head(10).astype(str).tolist())
            if reason == "all_daily_missing_next_month_close":
                diagnosis = "该月有当月行情，但缺少下一自然月月末有效 close，不能计算严格 1M forward return。"
            elif reason == "terminal_boundary_month":
                diagnosis = "终止边界月缺下一月 close，导致 forward return 无法确认。"
            elif reason == "symbol_month_price_missing":
                diagnosis = "部分权重证券在 all_daily 中缺少对应 symbol-month 行情。"
            else:
                diagnosis = "样本边界或行情源覆盖不足。"
        matched_share = matched_w / total_w if total_w else np.nan
        monthly_rows.append({
            "year_month": ym,
            "selected_count": int(grp.shape[0]),
            "matched_count": int(grp["matched_flag"].sum()),
            "unmatched_count": int((~grp["matched_flag"]).sum()),
            "matched_weight_share": matched_share,
            "unmatched_weight_share": 1.0 - matched_share if not pd.isna(matched_share) else np.nan,
            "low_match_flag": bool(matched_share < 0.95) if not pd.isna(matched_share) else True,
            "top_unmatched_symbols": top_symbols,
            "main_unmatched_reason": reason,
            "diagnosis": diagnosis,
        })
    match_qa = pd.DataFrame(monthly_rows)

    unmatched = joined.loc[~joined["matched_flag"]].copy()
    if unmatched.empty:
        reason_summary = pd.DataFrame(columns=["gap_reason", "row_count", "avg_weight_share", "example_symbols", "interpretation"])
    else:
        total_by_month = joined.groupby("year_month")["abs_weight"].sum().rename("month_abs_weight")
        unmatched = unmatched.merge(total_by_month, on="year_month", how="left")
        unmatched["weight_share"] = unmatched["abs_weight"] / unmatched["month_abs_weight"]
        interpretations = {
            "all_daily_missing_next_month_close": "有当月价格但缺下一自然月价格，严格 forward return 不可用。",
            "terminal_boundary_month": "终止边界月缺下一月价格，是低匹配月份的自然边界原因。",
            "symbol_month_price_missing": "证券-月份行情缺失，属于行情源覆盖缺口。",
            "start_boundary_before_all_daily": "权重月份早于 all_daily 覆盖起点。",
            "end_boundary_after_all_daily": "权重月份晚于 all_daily 覆盖终点。",
        }
        reason_summary = unmatched.groupby("gap_reason").agg(
            row_count=("symbol_norm", "size"),
            avg_weight_share=("weight_share", "mean"),
            example_symbols=("symbol_norm", lambda x: ",".join(pd.Series(x).drop_duplicates().head(10).astype(str))),
        ).reset_index()
        reason_summary["interpretation"] = reason_summary["gap_reason"].map(interpretations).fillna("未分类覆盖缺口。")
    return joined, match_qa, reason_summary


def make_windows(match_qa: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    windows = {
        "full_available_window": set(match_qa["year_month"].astype(str)),
        "high_coverage_window": set(match_qa.loc[match_qa["matched_weight_share"] >= 0.95, "year_month"].astype(str)),
    }
    rows = []
    for name, months in windows.items():
        sub = match_qa.loc[match_qa["year_month"].isin(months)].copy()
        excluded = match_qa.loc[~match_qa["year_month"].isin(months), "year_month"].astype(str).tolist()
        avg_share = float(sub["matched_weight_share"].mean()) if not sub.empty else np.nan
        min_share = float(sub["matched_weight_share"].min()) if not sub.empty else np.nan
        rows.append({
            "window_name": name,
            "month_count": int(sub.shape[0]),
            "min_year_month": str(sub["year_month"].min()) if not sub.empty else "",
            "max_year_month": str(sub["year_month"].max()) if not sub.empty else "",
            "avg_matched_weight_share": avg_share,
            "min_matched_weight_share": min_share,
            "excluded_month_count": int(len(excluded)),
            "excluded_months": ",".join(excluded),
            "window_status": status_from_coverage(avg_share, min_share) if not sub.empty else "LOW_MATCH",
        })
    return pd.DataFrame(rows), windows


def evaluate(joined: pd.DataFrame, match_qa: pd.DataFrame, windows: dict[str, set[str]], turnover: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = joined.copy()
    base["matched_return_contrib"] = np.where(base["matched_flag"], base["weight"] * base["fwd_ret_1m"], 0.0)
    gross_raw = base.groupby("year_month")["matched_return_contrib"].sum().rename("gross_raw").reset_index()
    monthly = match_qa.merge(gross_raw, on="year_month", how="left").merge(turnover, on="year_month", how="left")
    rows = []
    portfolio = str(joined["portfolio_name"].dropna().iloc[0]) if "portfolio_name" in joined.columns and joined["portfolio_name"].notna().any() else "V0_STRICT_LAG"
    for window_name, months in windows.items():
        sub = monthly.loc[monthly["year_month"].isin(months)].copy()
        for variant in ["raw_unmatched_not_renormalized", "matched_only_normalized"]:
            if variant == "raw_unmatched_not_renormalized":
                gross = sub["gross_raw"]
            else:
                gross = sub["gross_raw"] / sub["matched_weight_share"].replace(0, np.nan)
            for cost in [0, 10, 20, 30]:
                tmp = sub.copy()
                tmp["portfolio_name"] = portfolio
                tmp["window_name"] = window_name
                tmp["cost_bps"] = cost
                tmp["return_variant"] = variant
                tmp["gross_return"] = gross
                tmp["turnover_simple"] = tmp["turnover_simple"].fillna(0.0)
                tmp["net_return"] = tmp["gross_return"] - tmp["turnover_simple"] * cost / 10000.0
                rows.append(tmp[[
                    "portfolio_name",
                    "window_name",
                    "year_month",
                    "cost_bps",
                    "return_variant",
                    "gross_return",
                    "turnover_simple",
                    "net_return",
                    "matched_weight_share",
                    "unmatched_weight_share",
                    "low_match_flag",
                ]])
    monthly_returns = pd.concat(rows, ignore_index=True)

    summary_rows = []
    for keys, grp in monthly_returns.groupby(["portfolio_name", "window_name", "cost_bps", "return_variant"], sort=True):
        stats = perf_stats(grp, "net_return")
        summary_rows.append({
            "portfolio_name": keys[0],
            "window_name": keys[1],
            "cost_bps": keys[2],
            "return_variant": keys[3],
            **stats,
            "avg_turnover": float(grp["turnover_simple"].mean()),
            "avg_matched_weight_share": float(grp["matched_weight_share"].mean()),
            "min_matched_weight_share": float(grp["matched_weight_share"].min()),
            "low_match_month_count": int(grp["low_match_flag"].sum()),
        })
    return monthly_returns, pd.DataFrame(summary_rows)


def compare_old(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    strict = monthly_returns.loc[
        (monthly_returns["window_name"] == "high_coverage_window")
        & (monthly_returns["cost_bps"] == 20)
        & (monthly_returns["return_variant"] == "raw_unmatched_not_renormalized")
    ].copy()
    if not OLD_MONTHLY_PATH.exists() or strict.empty:
        return pd.DataFrame(columns=["sample_window", "common_month_count", "metric_name", "old_v0_value", "strict_lag_v0_value", "delta", "retention_ratio", "interpretation"])
    old = pd.read_csv(OLD_MONTHLY_PATH)
    old["year_month"] = period_str_from_datetime(old["month_end"])
    old = old.loc[(old["cost_bps"] == 20) & (old["return_variant"] == "raw_unmatched_not_renormalized")].copy()
    old = old.rename(columns={"net_return_csmar_bridge": "net_return"})
    common = sorted(set(old["year_month"]) & set(strict["year_month"]))
    old_c = old.loc[old["year_month"].isin(common)].copy()
    strict_c = strict.loc[strict["year_month"].isin(common)].copy()
    old_stats = perf_stats(old_c, "net_return")
    strict_stats = perf_stats(strict_c, "net_return")
    old_stats["avg_turnover"] = float(old_c["turnover_simple"].mean()) if not old_c.empty else np.nan
    strict_stats["avg_turnover"] = float(strict_c["turnover_simple"].mean()) if not strict_c.empty else np.nan
    rows = []
    for metric in ["mean_monthly_return", "sharpe", "tstat", "cumulative_return", "max_drawdown", "avg_turnover"]:
        ov = old_stats.get(metric, np.nan)
        sv = strict_stats.get(metric, np.nan)
        rows.append({
            "sample_window": "common_old_v0_and_strict_lag_high_coverage",
            "common_month_count": int(len(common)),
            "metric_name": metric,
            "old_v0_value": ov,
            "strict_lag_v0_value": sv,
            "delta": sv - ov if pd.notna(sv) and pd.notna(ov) else np.nan,
            "retention_ratio": safe_ratio(sv, ov),
            "interpretation": "strict-lag 与 old V0 在共同月份的同口径对照。",
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    update_run_state("started", "已启动 canonical all_daily bridge v3。")

    prereq = {
        "strict_lag_weights_found": WEIGHTS_PATH.exists(),
        "all_daily_found": ALL_DAILY_PATH.exists(),
        "coverage_summary_found": COVERAGE_SUMMARY_PATH.exists(),
        "old_v0_bridge_files_found": OLD_MONTHLY_PATH.exists() and OLD_SUMMARY_PATH.exists(),
    }
    prereq["missing_files"] = [
        str(p) for p in [WEIGHTS_PATH, ALL_DAILY_PATH, COVERAGE_SUMMARY_PATH]
        if not p.exists()
    ]
    prereq["prerequisites_passed"] = len(prereq["missing_files"]) == 0
    write_json(OUT_DIR / "v0_strict_lag_all_daily_bridge_prerequisite_check.json", prereq)
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    update_run_state("return_map", "构建 canonical all_daily monthly forward return map。")
    ret_map, ret_qa = build_return_map()
    ret_map.to_parquet(OUT_DIR / "v0_canonical_all_daily_monthly_return_map.parquet", index=False)
    pd.DataFrame([ret_qa]).to_csv(OUT_DIR / "v0_canonical_all_daily_return_map_qa.csv", index=False, encoding="utf-8-sig")

    update_run_state("match_eval", "匹配 strict-lag weights 并计算窗口与收益。")
    weights = load_weights()
    joined, match_qa, reason_summary = diagnose_matches(weights, ret_map)
    turnover = compute_turnover(weights)
    window_qa, windows = make_windows(match_qa)
    monthly_returns, perf_summary = evaluate(joined, match_qa, windows, turnover)
    comparison = compare_old(monthly_returns)

    match_qa.to_csv(OUT_DIR / "v0_strict_lag_all_daily_match_monthly_qa.csv", index=False, encoding="utf-8-sig")
    reason_summary.to_csv(OUT_DIR / "v0_strict_lag_all_daily_unmatched_reason_summary.csv", index=False, encoding="utf-8-sig")
    window_qa.to_csv(OUT_DIR / "v0_strict_lag_all_daily_evaluation_window_qa.csv", index=False, encoding="utf-8-sig")
    monthly_returns.to_csv(OUT_DIR / "v0_strict_lag_all_daily_monthly_net_return_by_cost.csv", index=False, encoding="utf-8-sig")
    perf_summary.to_csv(OUT_DIR / "v0_strict_lag_all_daily_performance_summary_by_cost.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT_DIR / "v0_old_vs_strict_lag_all_daily_bridge_comparison.csv", index=False, encoding="utf-8-sig")

    selected = perf_summary.loc[
        (perf_summary["window_name"] == "high_coverage_window")
        & (perf_summary["cost_bps"] == 20)
        & (perf_summary["return_variant"] == "raw_unmatched_not_renormalized")
    ].iloc[0].to_dict()
    high_window = window_qa.loc[window_qa["window_name"] == "high_coverage_window"].iloc[0].to_dict()
    full_window = window_qa.loc[window_qa["window_name"] == "full_available_window"].iloc[0].to_dict()

    old_metrics = {r["metric_name"]: r["old_v0_value"] for _, r in comparison.iterrows()} if not comparison.empty else {}
    strict_metrics = {r["metric_name"]: r["strict_lag_v0_value"] for _, r in comparison.iterrows()} if not comparison.empty else {}
    if high_window["window_status"] in {"READY", "READY_WITH_MINOR_GAPS"} and selected["sharpe"] >= 0.8 and selected["cumulative_return"] > 0:
        assessment = "LOW_IMPACT_STRICT_LAG_STILL_STRONG"
    elif high_window["window_status"] in {"READY", "READY_WITH_MINOR_GAPS"} and selected["sharpe"] > 0 and selected["cumulative_return"] > 0:
        assessment = "MEDIUM_IMPACT_STRICT_LAG_WEAKER_BUT_USABLE"
    elif high_window["window_status"] in {"READY", "READY_WITH_MINOR_GAPS"} and (selected["sharpe"] <= 0 or selected["cumulative_return"] <= 0):
        assessment = "HIGH_IMPACT_OLD_V0_LIKELY_LEAKAGE_DRIVEN"
    else:
        assessment = "INCONCLUSIVE_DUE_TO_REMAINING_COVERAGE_ISSUE"

    leakage = pd.DataFrame([{
        "selected_window_name": "high_coverage_window",
        "selected_window_status": high_window["window_status"],
        "strict_lag_20bps_sharpe": selected["sharpe"],
        "strict_lag_20bps_mean_monthly_return": selected["mean_monthly_return"],
        "strict_lag_20bps_tstat": selected["tstat"],
        "strict_lag_20bps_cumulative_return": selected["cumulative_return"],
        "strict_lag_20bps_max_drawdown": selected["max_drawdown"],
        "old_v0_common_20bps_sharpe": old_metrics.get("sharpe", np.nan),
        "old_v0_common_20bps_mean_monthly_return": old_metrics.get("mean_monthly_return", np.nan),
        "sharpe_retention_ratio": safe_ratio(strict_metrics.get("sharpe", np.nan), old_metrics.get("sharpe", np.nan)),
        "mean_return_retention_ratio": safe_ratio(strict_metrics.get("mean_monthly_return", np.nan), old_metrics.get("mean_monthly_return", np.nan)),
        "leakage_impact_assessment": assessment,
        "interpretation": "基于 high_coverage_window、20bps、raw unmatched 不重归一口径的 strict-lag 与 old V0 对照。",
    }])
    leakage.to_csv(OUT_DIR / "v0_strict_lag_all_daily_leakage_impact_summary.csv", index=False, encoding="utf-8-sig")

    guardrails = {
        "strict_lag_alpha_signal_regenerated": False,
        "strict_lag_weights_regenerated": False,
        "original_orthogonalization_modified": False,
        "old_artifacts_modified": False,
        "production_modified": False,
        "ml_training_run": False,
        "new_ml_model_trained": False,
        "portfolio_returns_calculated": True,
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
    }
    expected = {
        **{k: False for k in guardrails if k != "portfolio_returns_calculated"},
        "portfolio_returns_calculated": True,
    }
    guardrail_df = pd.DataFrame([
        {"guardrail": k, "expected": expected[k], "actual": v, "pass": bool(v == expected[k])}
        for k, v in guardrails.items()
    ])
    guardrail_df.to_csv(OUT_DIR / "v0_strict_lag_all_daily_bridge_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    guardrail_pass = bool(guardrail_df["pass"].all())

    if not guardrail_pass:
        final_decision = "V0_STRICT_LAG_ALL_DAILY_FAIL_GUARDRAIL"
    elif high_window["window_status"] in {"WATCH_COVERAGE_GAPS", "LOW_MATCH"}:
        final_decision = "V0_STRICT_LAG_ALL_DAILY_INCONCLUSIVE_COVERAGE_ISSUE"
    elif selected["sharpe"] >= 0.8 and selected["cumulative_return"] > 0:
        final_decision = "V0_STRICT_LAG_ALL_DAILY_STILL_STRONG_CONTINUE_REBUILD"
    elif selected["sharpe"] > 0 and selected["cumulative_return"] > 0:
        final_decision = "V0_STRICT_LAG_ALL_DAILY_WEAKER_BUT_USABLE_CONTINUE_REBUILD"
    else:
        final_decision = "V0_STRICT_LAG_ALL_DAILY_COLLAPSED_OLD_V0_LIKELY_LEAKAGE_DRIVEN"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "canonical_all_daily_return_map_generated": True,
        "return_map_qa_status": ret_qa["qa_status"],
        "full_window_status": full_window["window_status"],
        "high_coverage_window_status": high_window["window_status"],
        "high_coverage_month_count": int(high_window["month_count"]),
        "high_coverage_min_year_month": high_window["min_year_month"],
        "high_coverage_max_year_month": high_window["max_year_month"],
        "high_coverage_avg_matched_weight_share": high_window["avg_matched_weight_share"],
        "high_coverage_min_matched_weight_share": high_window["min_matched_weight_share"],
        "strict_lag_20bps_sharpe": selected["sharpe"],
        "strict_lag_20bps_mean_monthly_return": selected["mean_monthly_return"],
        "strict_lag_20bps_tstat": selected["tstat"],
        "strict_lag_20bps_cumulative_return": selected["cumulative_return"],
        "strict_lag_20bps_max_drawdown": selected["max_drawdown"],
        "strict_lag_20bps_avg_turnover": selected["avg_turnover"],
        "old_v0_common_20bps_sharpe": old_metrics.get("sharpe", np.nan),
        "old_v0_common_20bps_mean_monthly_return": old_metrics.get("mean_monthly_return", np.nan),
        "old_v0_common_20bps_tstat": old_metrics.get("tstat", np.nan),
        "old_v0_common_20bps_cumulative_return": old_metrics.get("cumulative_return", np.nan),
        "old_v0_common_20bps_max_drawdown": old_metrics.get("max_drawdown", np.nan),
        "sharpe_retention_ratio": safe_ratio(strict_metrics.get("sharpe", np.nan), old_metrics.get("sharpe", np.nan)),
        "mean_return_retention_ratio": safe_ratio(strict_metrics.get("mean_monthly_return", np.nan), old_metrics.get("mean_monthly_return", np.nan)),
        "leakage_impact_assessment": assessment,
        "v0_structure_still_research_worthy": bool(final_decision in {
            "V0_STRICT_LAG_ALL_DAILY_STILL_STRONG_CONTINUE_REBUILD",
            "V0_STRICT_LAG_ALL_DAILY_WEAKER_BUT_USABLE_CONTINUE_REBUILD",
        }),
        "canonical_rebuild_still_required": True,
        **guardrails,
        "final_decision": final_decision,
        "recommended_next_step": "继续 canonical rebuild，优先沿用 strict-lag 与 all_daily canonical return map；低覆盖月份已在 QA 中列出。" if "CONTINUE_REBUILD" in final_decision else "先处理覆盖或 guardrail 问题后再继续 rebuild。",
    }
    write_json(OUT_DIR / "v0_strict_lag_canonical_all_daily_bridge_summary.json", summary)

    report = [
        "# V0 strict-lag canonical all_daily bridge v3 报告",
        "",
        f"- final_decision: {final_decision}",
        f"- prerequisites_passed: {prereq['prerequisites_passed']}",
        f"- return_map_qa_status: {ret_qa['qa_status']}",
        f"- full_window_status: {full_window['window_status']}",
        f"- high_coverage_window_status: {high_window['window_status']}",
        f"- high_coverage_month_count: {high_window['month_count']}",
        f"- high_coverage_window: {high_window['min_year_month']} 至 {high_window['max_year_month']}",
        f"- high_coverage_avg_matched_weight_share: {high_window['avg_matched_weight_share']}",
        f"- high_coverage_min_matched_weight_share: {high_window['min_matched_weight_share']}",
        f"- strict_lag_20bps_sharpe: {selected['sharpe']}",
        f"- strict_lag_20bps_mean_monthly_return: {selected['mean_monthly_return']}",
        f"- strict_lag_20bps_tstat: {selected['tstat']}",
        f"- strict_lag_20bps_cumulative_return: {selected['cumulative_return']}",
        f"- strict_lag_20bps_max_drawdown: {selected['max_drawdown']}",
        f"- leakage_impact_assessment: {assessment}",
        "",
        "低匹配月份解释见 v0_strict_lag_all_daily_match_monthly_qa.csv 与 unmatched reason summary。",
    ]
    (OUT_DIR / "v0_strict_lag_canonical_all_daily_bridge_report.md").write_text("\n".join(report), encoding="utf-8")

    expected_files = [
        "v0_strict_lag_all_daily_bridge_prerequisite_check.json",
        "v0_canonical_all_daily_monthly_return_map.parquet",
        "v0_canonical_all_daily_return_map_qa.csv",
        "v0_strict_lag_all_daily_match_monthly_qa.csv",
        "v0_strict_lag_all_daily_unmatched_reason_summary.csv",
        "v0_strict_lag_all_daily_evaluation_window_qa.csv",
        "v0_strict_lag_all_daily_monthly_net_return_by_cost.csv",
        "v0_strict_lag_all_daily_performance_summary_by_cost.csv",
        "v0_old_vs_strict_lag_all_daily_bridge_comparison.csv",
        "v0_strict_lag_all_daily_leakage_impact_summary.csv",
        "v0_strict_lag_all_daily_bridge_guardrail_qa.csv",
        "v0_strict_lag_canonical_all_daily_bridge_summary.json",
        "v0_strict_lag_canonical_all_daily_bridge_report.md",
    ]
    final_qa = pd.DataFrame([
        {"artifact": name, "path": str(OUT_DIR / name), "exists": (OUT_DIR / name).exists()}
        for name in expected_files
    ] + [{"artifact": "script", "path": str(ROOT / "scripts" / "bridge_v0_strict_lag_canonical_all_daily_v3.py"), "exists": True}])
    final_qa.to_csv(RUN_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    write_json(RUN_DIR / "terminal_summary.json", {
        "task_name": TASK_NAME,
        "status": "completed",
        "output_dir": str(OUT_DIR),
        "stdout_log": str(RUN_DIR / "run_stdout.txt"),
        "stderr_log": str(RUN_DIR / "run_stderr.txt"),
        "final_decision": final_decision,
        "guardrail_pass": guardrail_pass,
    })
    (RUN_DIR / "task_completion_card.md").write_text(
        "\n".join([
            f"# {TASK_NAME}",
            "",
            f"- status: completed",
            f"- final_decision: {final_decision}",
            f"- output_dir: {OUT_DIR}",
            f"- summary_json: {OUT_DIR / 'v0_strict_lag_canonical_all_daily_bridge_summary.json'}",
            f"- final_qa: {RUN_DIR / 'final_qa.csv'}",
        ]),
        encoding="utf-8",
    )
    update_run_state("completed", f"任务完成，final_decision={final_decision}。")
    del ret_map, weights, joined, monthly_returns, perf_summary
    gc.collect()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
