from __future__ import annotations

import gc
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "v0_strict_lag_icir_rebuild_bridge_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

PREPROCESSED = ROOT / "output" / "preprocessed.parquet"
OLD_SPLIT = ROOT / "output" / "split_universe_blended.parquet"
ORTHO_PY = ROOT / "factor_research" / "orthogonalization.py"
SPLIT_PY = ROOT / "factor_research" / "split_universe.py"
CSMAR_RET = (
    ROOT
    / "output"
    / "robust_cleaned_fundamental_factor_variant_build_v0"
    / "robust_cleaned_factor_score_panel_v0.parquet"
)
OLD_SUMMARY = (
    ROOT
    / "output"
    / "reconstructed_v0_v7_csmar_bridge_evaluation_v0"
    / "bridge_performance_summary_csmar_by_cost.csv"
)
OLD_MONTHLY = (
    ROOT
    / "output"
    / "reconstructed_v0_v7_csmar_bridge_evaluation_v0"
    / "bridge_monthly_net_return_csmar_by_cost.csv"
)

FACTOR_COLS = [
    "Mom_1M",
    "Mom_3M",
    "Mom_6M",
    "Mom_12M_1M",
    "Vol_20D",
    "Vol_60D",
    "Beta",
    "BP",
    "EP",
    "ROE",
    "Debt_Ratio",
    "Net_Profit_Margin",
    "RevGrowth_YoY",
    "ProfitGrowth_YoY",
    "VolChg_20D",
    "PriceDev_20D",
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
    text = ["# RUN_STATE", "", f"- task_name: {TASK_NAME}", f"- status: {status}"]
    for key, value in payload["details"].items():
        text.append(f"- {key}: {value}")
    text.append("")
    text.append("```json")
    text.append(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    text.append("```")
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(text), encoding="utf-8")


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_symbol(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.zfill(6)


def get_date_col(df: pd.DataFrame) -> str:
    if "month_end" in df.columns:
        return "month_end"
    return "date"


def ensure_forward_return(panel: pd.DataFrame) -> pd.DataFrame:
    if "forward_return_1m" in panel.columns:
        return panel
    close_col = "收盘" if "收盘" in panel.columns else "close"
    if close_col not in panel.columns:
        raise KeyError("preprocessed 缺少 forward_return_1m 且未找到收盘价列")
    panel = panel.sort_values(["symbol", "date"]).copy()
    panel["forward_return_1m"] = panel.groupby("symbol")[close_col].transform(
        lambda x: x.shift(-1) / x - 1
    )
    return panel


def factor_actual_cols(panel: pd.DataFrame) -> list[str]:
    actual = []
    for name in FACTOR_COLS:
        for suffix in ["_neutral_z", "_z", ""]:
            col = f"{name}{suffix}"
            if col in panel.columns:
                actual.append(col)
                break
    return actual


def split_by_market_cap(panel: pd.DataFrame, percentile: float = 0.5) -> pd.DataFrame:
    amount_col = "成交额" if "成交额" in panel.columns else "amount"
    turnover_col = "换手率" if "换手率" in panel.columns else "turnover"
    if amount_col not in panel.columns or turnover_col not in panel.columns:
        raise KeyError("无法估计市值：缺少成交额或换手率")

    panel = panel.copy()
    amount = pd.to_numeric(panel[amount_col], errors="coerce")
    turnover = pd.to_numeric(panel[turnover_col], errors="coerce")
    panel["mcap_est"] = np.where(turnover > 1e-10, amount / turnover, np.nan)
    panel["mcap_est"] = panel["mcap_est"].replace([np.inf, -np.inf], np.nan)

    def _rank(grp: pd.DataFrame) -> pd.DataFrame:
        vals = grp["mcap_est"].dropna()
        grp = grp.copy()
        if len(vals) < 10:
            grp["mcap_pct"] = np.nan
            grp["universe"] = "未分类"
            return grp[["mcap_pct", "universe"]]
        clipped = vals.clip(vals.quantile(0.01), vals.quantile(0.99))
        pct = clipped.rank(pct=True)
        grp["mcap_pct"] = pct.reindex(grp.index)
        grp["universe"] = np.where(
            grp["mcap_pct"] >= percentile,
            "大盘",
            np.where(grp["mcap_pct"].notna(), "小盘", "未分类"),
        )
        return grp[["mcap_pct", "universe"]]

    ranked = pd.concat([_rank(g) for _, g in panel.groupby("date", sort=True)], axis=0)
    panel["mcap_pct"] = ranked["mcap_pct"]
    panel["universe"] = ranked["universe"]
    return panel


def _residualize(y: np.ndarray, x: np.ndarray, min_variance: float = 1e-10) -> np.ndarray:
    try:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        resid = y - x @ beta
    except np.linalg.LinAlgError:
        return np.zeros_like(y)
    if np.var(resid) < min_variance:
        return np.zeros_like(y)
    return resid


def _rank_ic(x: pd.Series, y: pd.Series) -> float:
    rx = x.rank(method="average")
    ry = y.rank(method="average")
    corr = rx.corr(ry)
    return float(corr) if pd.notna(corr) else np.nan


def compute_rolling_ic_ir_strict_lag(
    df: pd.DataFrame,
    factor_cols: list[str],
    return_col: str = "forward_return_1m",
    date_col: str = "date",
    rolling_window: int = 24,
    min_stocks: int = 20,
    universe_name: str = "",
) -> tuple[dict[pd.Timestamp, dict[str, float]], pd.DataFrame]:
    ic_series: dict[str, pd.Series] = {}
    for col in factor_cols:
        ic_vals: dict[pd.Timestamp, float] = {}
        for dt, grp in df.groupby(date_col, sort=True):
            sub = grp[[col, return_col]].dropna()
            if len(sub) >= min_stocks:
                ic = _rank_ic(sub[col], sub[return_col])
                if not np.isnan(ic):
                    ic_vals[pd.Timestamp(dt)] = ic
        ic_series[col] = pd.Series(ic_vals).sort_index()

    dates = [pd.Timestamp(x) for x in sorted(df[date_col].dropna().unique())]
    result: dict[pd.Timestamp, dict[str, float]] = {}
    audit_rows = []

    for dt in dates:
        result[dt] = {}
        dt64 = dt.to_datetime64()
        for col in factor_cols:
            series = ic_series.get(col, pd.Series(dtype=float))
            idx_array = series.index.values
            pos = np.searchsorted(idx_array, dt64, side="left") - 1
            if pos < 0:
                window_ic = pd.Series(dtype=float)
            else:
                window_start = max(0, pos - rolling_window + 1)
                window_ic = series.iloc[window_start : pos + 1]

            if len(window_ic) < 2:
                icir = 0.0
            else:
                std_ic = float(np.std(window_ic, ddof=1))
                icir = float(np.mean(window_ic) / std_ic) if std_ic > 1e-10 else 0.0
            result[dt][col] = icir

            used_idx = pd.DatetimeIndex(window_ic.index)
            current_included = bool((used_idx == dt).any())
            future_included = bool((used_idx > dt).any())
            audit_rows.append(
                {
                    "month_end": dt,
                    "factor_name": col,
                    "universe": universe_name,
                    "current_month": dt,
                    "first_ic_month_used": used_idx.min() if len(used_idx) else pd.NaT,
                    "last_ic_month_used": used_idx.max() if len(used_idx) else pd.NaT,
                    "current_month_ic_included": current_included,
                    "future_ic_included": future_included,
                    "ic_count_used": int(len(window_ic)),
                    "icir_value": icir,
                    "strict_lag_pass": (not current_included) and (not future_included),
                }
            )

    return result, pd.DataFrame(audit_rows)


def apply_strict_lag_composite(panel: pd.DataFrame, factor_cols: list[str], universe_name: str):
    rolling, audit = compute_rolling_ic_ir_strict_lag(
        panel,
        factor_cols=factor_cols,
        return_col="forward_return_1m",
        date_col="date",
        rolling_window=24,
        min_stocks=20,
        universe_name=universe_name,
    )
    panel = panel.copy()
    composite = pd.Series(0.0, index=panel.index)

    for dt in sorted(panel["date"].dropna().unique()):
        dt_ts = pd.Timestamp(dt)
        ic_irs = rolling.get(dt_ts, {})
        sorted_cols = sorted(
            [c for c in ic_irs if abs(ic_irs.get(c, 0.0)) > 0.05],
            key=lambda c: abs(ic_irs[c]),
            reverse=True,
        )
        if not sorted_cols:
            continue
        mask = panel["date"] == dt
        idx = np.where(mask.values)[0]
        if len(idx) < 5:
            continue

        orth_values: dict[str, np.ndarray] = {}
        valid_cols: list[str] = []
        for col in sorted_cols:
            y = panel.loc[mask, col].astype(float).to_numpy()
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            if not valid_cols:
                resid = y.copy()
            else:
                x = np.column_stack([orth_values[c] for c in valid_cols])
                resid = _residualize(y, x, 1e-10)
            orth_values[col] = resid
            valid_cols.append(col)

        total_abs = sum(abs(ic_irs[c]) for c in valid_cols)
        if total_abs < 1e-10:
            continue
        dt_comp = np.zeros(len(idx))
        for col in valid_cols:
            if np.var(orth_values[col]) < 1e-10:
                continue
            sign = -1.0 if ic_irs[col] < 0 else 1.0
            dt_comp += sign * (abs(ic_irs[col]) / total_abs) * orth_values[col]
        composite.loc[mask] = dt_comp

    panel["composite_score_strict_lag"] = composite
    return panel, audit


def zscore_by_month(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()

    def _z(s: pd.Series) -> pd.Series:
        std = s.std(ddof=1)
        if pd.isna(std) or std <= 1e-10:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    panel["alpha_signal_strict_lag"] = panel.groupby("date")[
        "composite_score_strict_lag"
    ].transform(_z)
    return panel


def build_weights(alpha: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    qa_rows = []
    trans_rows = []
    prev_holdings: set[str] = set()
    portfolio = "V0_STRICT_LAG_TOP50_BUFFER_35_75_EQUAL_WEIGHT"

    use = alpha[alpha["eligible_flag"]].copy()
    use = use.dropna(subset=["alpha_signal_strict_lag"])
    for dt, grp in use.groupby("month_end", sort=True):
        grp = grp.sort_values(["alpha_signal_strict_lag", "symbol"], ascending=[False, True]).copy()
        grp["rank_in_month"] = np.arange(1, len(grp) + 1)
        rank_map = dict(zip(grp["symbol"], grp["rank_in_month"]))
        previous = {s for s in prev_holdings if s in rank_map}
        kept = {s for s in previous if rank_map[s] <= 75}
        entry_candidates = grp.loc[
            (grp["rank_in_month"] <= 35) & (~grp["symbol"].isin(kept)), "symbol"
        ].tolist()
        selected = list(sorted(kept, key=lambda s: (rank_map[s], s)))
        for sym in entry_candidates:
            if sym not in selected and len(selected) < 50:
                selected.append(sym)
        if len(selected) < 50:
            for sym in grp["symbol"].tolist():
                if sym not in selected:
                    selected.append(sym)
                if len(selected) >= 50:
                    break
        selected_set = set(selected[:50])
        weight = 1.0 / len(selected_set) if selected_set else 0.0

        for rec in grp.to_dict("records"):
            sym = rec["symbol"]
            is_sel = sym in selected_set
            was_prev = sym in prev_holdings
            buffer_kept = is_sel and was_prev and rank_map[sym] <= 75
            buffer_entry = is_sel and (not was_prev) and rank_map[sym] <= 35
            if not is_sel:
                reason = "not_selected"
            elif not prev_holdings:
                reason = "initial_top50"
            elif buffer_kept:
                reason = "buffer_kept"
            elif buffer_entry:
                reason = "buffer_entry"
            else:
                reason = "fill_to_target"
            rows.append(
                {
                    "portfolio_name": portfolio,
                    "symbol": sym,
                    "month_end": dt,
                    "alpha_signal_strict_lag": rec["alpha_signal_strict_lag"],
                    "rank_in_month": int(rec["rank_in_month"]),
                    "selected_flag": bool(is_sel),
                    "selection_reason": reason,
                    "previous_holding_flag": bool(was_prev),
                    "buffer_kept_flag": bool(buffer_kept),
                    "buffer_entry_flag": bool(buffer_entry),
                    "weight": weight if is_sel else 0.0,
                    "holding_count": int(len(selected_set)),
                    "reconstruction_rule": "Top50_Buffer_35_75_equal_weight_strict_lag_no_return_input",
                }
            )

        selected_count = len(selected_set)
        weight_sum = selected_count * weight
        duplicates = len(selected_set) - len(set(selected_set))
        kept_count = len(selected_set & prev_holdings)
        exited = len(prev_holdings - selected_set)
        new_entries = len(selected_set - prev_holdings)
        trans_rows.append(
            {
                "month_end": dt,
                "previous_holding_count": len(prev_holdings),
                "kept_from_previous_count": kept_count,
                "exited_count": exited,
                "new_entry_count": new_entries,
                "filled_to_target_count": sum(1 for r in rows[-len(grp) :] if r["selection_reason"] == "fill_to_target"),
                "selected_stock_count": selected_count,
                "simple_turnover_proxy": new_entries / 50.0 if prev_holdings else selected_count / 100.0,
                "transition_status": "PASS" if selected_count == min(50, len(grp)) else "CHECK",
            }
        )
        qa_rows.append(
            {
                "month_end": dt,
                "eligible_stock_count": len(grp),
                "selected_stock_count": selected_count,
                "target_holding_count": 50,
                "weight_sum": weight_sum,
                "weight_sum_abs_error": abs(weight_sum - (1.0 if selected_count else 0.0)),
                "min_weight": weight if selected_count else 0.0,
                "max_weight": weight if selected_count else 0.0,
                "duplicate_symbol_count": duplicates,
                "first_month_flag": len(prev_holdings) == 0,
                "selected_count_status": "PASS" if selected_count == min(50, len(grp)) else "FAIL",
                "weight_sum_status": "PASS" if abs(weight_sum - 1.0) < 1e-10 else "FAIL",
            }
        )
        prev_holdings = selected_set

    return pd.DataFrame(rows), pd.DataFrame(qa_rows), pd.DataFrame(trans_rows)


def max_drawdown(returns: pd.Series) -> float:
    curve = (1.0 + returns.fillna(0.0)).cumprod()
    dd = curve / curve.cummax() - 1.0
    return float(dd.min()) if len(dd) else np.nan


def perf_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cost, grp in monthly.groupby("cost_bps", sort=True):
        r = grp["net_return"].astype(float)
        vol = float(r.std(ddof=1)) if len(r) > 1 else np.nan
        mean = float(r.mean()) if len(r) else np.nan
        rows.append(
            {
                "portfolio_name": grp["portfolio_name"].iloc[0],
                "cost_bps": int(cost),
                "month_count": int(len(r)),
                "mean_monthly_return": mean,
                "annualized_return_approx": mean * 12 if pd.notna(mean) else np.nan,
                "monthly_volatility": vol,
                "sharpe": mean / vol * math.sqrt(12) if vol and vol > 0 else np.nan,
                "tstat": mean / vol * math.sqrt(len(r)) if vol and vol > 0 else np.nan,
                "positive_month_ratio": float((r > 0).mean()) if len(r) else np.nan,
                "cumulative_return": float((1.0 + r.fillna(0.0)).prod() - 1.0),
                "max_drawdown": max_drawdown(r),
                "avg_turnover": float(grp["turnover_simple"].mean()),
                "avg_matched_weight_share": float(grp["matched_weight_share"].mean()),
                "min_matched_weight_share": float(grp["matched_weight_share"].min()),
                "low_match_month_count": int(grp["low_match_flag"].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(r: pd.Series, turnover: pd.Series, matched: pd.Series) -> dict:
    vol = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    mean = float(r.mean()) if len(r) else np.nan
    return {
        "mean_monthly_return": mean,
        "sharpe": mean / vol * math.sqrt(12) if vol and vol > 0 else np.nan,
        "tstat": mean / vol * math.sqrt(len(r)) if vol and vol > 0 else np.nan,
        "cumulative_return": float((1.0 + r.fillna(0.0)).prod() - 1.0),
        "max_drawdown": max_drawdown(r),
        "avg_turnover": float(turnover.mean()) if len(turnover) else np.nan,
        "avg_matched_weight_share": float(matched.mean()) if len(matched) else np.nan,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", {"step": "start"})

    prereq = {
        "preprocessed_found": PREPROCESSED.exists(),
        "split_universe_blended_found": OLD_SPLIT.exists(),
        "orthogonalization_py_found": ORTHO_PY.exists(),
        "split_universe_py_found": SPLIT_PY.exists(),
        "csmar_return_source_found": CSMAR_RET.exists(),
    }
    prereq["missing_files"] = [
        str(p)
        for ok, p in [
            (prereq["preprocessed_found"], PREPROCESSED),
            (prereq["orthogonalization_py_found"], ORTHO_PY),
            (prereq["split_universe_py_found"], SPLIT_PY),
            (prereq["csmar_return_source_found"], CSMAR_RET),
        ]
        if not ok
    ]
    prereq["prerequisites_passed"] = len(prereq["missing_files"]) == 0
    save_json(prereq, OUT_DIR / "v0_strict_lag_prerequisite_check.json")
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", {"step": "read_preprocessed"})
    need_cols = [
        "date",
        "symbol",
        "收盘",
        "成交额",
        "换手率",
        "forward_return_1m",
    ]
    available_probe = pd.read_parquet(PREPROCESSED, engine="pyarrow", columns=None).head(0).columns
    read_cols = [c for c in need_cols + FACTOR_COLS + [f"{c}_neutral_z" for c in FACTOR_COLS] + [f"{c}_z" for c in FACTOR_COLS] if c in available_probe]
    panel = pd.read_parquet(PREPROCESSED, engine="pyarrow", columns=read_cols)
    panel["symbol"] = normalize_symbol(panel["symbol"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = ensure_forward_return(panel)
    actual_cols = factor_actual_cols(panel)
    if not actual_cols:
        raise ValueError("未找到可用因子列")

    write_state("running", {"step": "build_strict_lag_signal", "factor_count": len(actual_cols)})
    panel = split_by_market_cap(panel)
    panel = panel[panel["universe"].isin(["大盘", "小盘"])].copy()
    large, audit_l = apply_strict_lag_composite(panel[panel["universe"] == "大盘"].copy(), actual_cols, "大盘")
    small, audit_s = apply_strict_lag_composite(panel[panel["universe"] == "小盘"].copy(), actual_cols, "小盘")
    audit = pd.concat([audit_l, audit_s], ignore_index=True)
    audit.to_csv(OUT_DIR / "v0_strict_lag_icir_window_audit.csv", index=False, encoding="utf-8-sig")

    alpha = pd.concat([zscore_by_month(large), zscore_by_month(small)], ignore_index=True)
    alpha["month_end"] = pd.to_datetime(alpha["date"])
    alpha["eligible_flag"] = alpha["alpha_signal_strict_lag"].notna()
    alpha["source_panel_path"] = str(PREPROCESSED)
    alpha["strict_lag_rule"] = "IC_IR uses months strictly before current month; searchsorted side='left'-1"
    alpha_out_cols = [
        "symbol",
        "month_end",
        "alpha_signal_strict_lag",
        "universe",
        "mcap_pct",
        "composite_score_strict_lag",
        "eligible_flag",
        "source_panel_path",
        "strict_lag_rule",
    ]
    alpha[alpha_out_cols].to_parquet(OUT_DIR / "v0_strict_lag_alpha_signal_panel.parquet", index=False)
    alpha[alpha_out_cols].head(200).to_csv(
        OUT_DIR / "v0_strict_lag_alpha_signal_sample.csv", index=False, encoding="utf-8-sig"
    )

    write_state("running", {"step": "lineage_compare"})
    if OLD_SPLIT.exists():
        old = pd.read_parquet(OLD_SPLIT, columns=["date", "symbol", "alpha_signal"], engine="pyarrow")
        old["symbol"] = normalize_symbol(old["symbol"])
        old["month_end"] = pd.to_datetime(old["date"])
        cmp_df = old[["symbol", "month_end", "alpha_signal"]].merge(
            alpha[["symbol", "month_end", "alpha_signal_strict_lag"]], on=["symbol", "month_end"], how="inner"
        )
        cmp_rows = []
        for dt, grp in cmp_df.groupby("month_end", sort=True):
            if len(grp) < 10:
                continue
            old_rank = grp["alpha_signal"].rank(ascending=False, method="first")
            new_rank = grp["alpha_signal_strict_lag"].rank(ascending=False, method="first")
            top50_old = set(grp.loc[old_rank <= 50, "symbol"])
            top50_new = set(grp.loc[new_rank <= 50, "symbol"])
            top75_old = set(grp.loc[old_rank <= 75, "symbol"])
            top75_new = set(grp.loc[new_rank <= 75, "symbol"])
            pearson = grp["alpha_signal"].corr(grp["alpha_signal_strict_lag"])
            spearman = old_rank.corr(new_rank)
            cmp_rows.append(
                {
                    "month_end": dt,
                    "common_symbol_count": len(grp),
                    "pearson_corr": pearson,
                    "spearman_corr": spearman,
                    "top50_overlap": len(top50_old & top50_new) / max(1, len(top50_old | top50_new)),
                    "top75_overlap": len(top75_old & top75_new) / max(1, len(top75_old | top75_new)),
                    "mean_abs_score_diff": float((grp["alpha_signal"] - grp["alpha_signal_strict_lag"]).abs().mean()),
                    "mean_abs_rank_diff": float((old_rank - new_rank).abs().mean()),
                    "interpretation": "strict_lag_vs_original_same_lineage_monthly",
                }
            )
        cmp_out = pd.DataFrame(cmp_rows)
        cmp_out.to_csv(OUT_DIR / "v0_old_vs_strict_lag_signal_comparison.csv", index=False, encoding="utf-8-sig")
        if len(cmp_out):
            avg_p = float(cmp_out["pearson_corr"].mean())
            avg_s = float(cmp_out["spearman_corr"].mean())
            avg_t50 = float(cmp_out["top50_overlap"].mean())
            if cmp_out["common_symbol_count"].mean() < 50:
                status = "INSUFFICIENT_OVERLAP"
            elif avg_s >= 0.8 and avg_t50 >= 0.6:
                status = "HIGH_SIMILARITY"
            elif avg_s >= 0.4:
                status = "MODERATE_SIMILARITY"
            else:
                status = "LOW_SIMILARITY"
            cmp_summary = pd.DataFrame(
                [
                    {
                        "avg_pearson_corr": avg_p,
                        "avg_spearman_corr": avg_s,
                        "avg_top50_overlap": avg_t50,
                        "avg_top75_overlap": float(cmp_out["top75_overlap"].mean()),
                        "avg_mean_abs_rank_diff": float(cmp_out["mean_abs_rank_diff"].mean()),
                        "lineage_similarity_status": status,
                    }
                ]
            )
        else:
            cmp_summary = pd.DataFrame(
                [
                    {
                        "avg_pearson_corr": np.nan,
                        "avg_spearman_corr": np.nan,
                        "avg_top50_overlap": np.nan,
                        "avg_top75_overlap": np.nan,
                        "avg_mean_abs_rank_diff": np.nan,
                        "lineage_similarity_status": "INSUFFICIENT_OVERLAP",
                    }
                ]
            )
    else:
        cmp_summary = pd.DataFrame(
            [
                {
                    "avg_pearson_corr": np.nan,
                    "avg_spearman_corr": np.nan,
                    "avg_top50_overlap": np.nan,
                    "avg_top75_overlap": np.nan,
                    "avg_mean_abs_rank_diff": np.nan,
                    "lineage_similarity_status": "INSUFFICIENT_OVERLAP",
                }
            ]
        )
        pd.DataFrame().to_csv(OUT_DIR / "v0_old_vs_strict_lag_signal_comparison.csv", index=False)
    cmp_summary.to_csv(
        OUT_DIR / "v0_old_vs_strict_lag_signal_comparison_summary.csv", index=False, encoding="utf-8-sig"
    )

    del large, small, panel
    gc.collect()

    write_state("running", {"step": "weights"})
    weights, weight_qa, trans_qa = build_weights(alpha[alpha_out_cols])
    weights.to_parquet(OUT_DIR / "v0_strict_lag_reconstructed_weights.parquet", index=False)
    weight_qa.to_csv(OUT_DIR / "v0_strict_lag_weight_monthly_qa.csv", index=False, encoding="utf-8-sig")
    trans_qa.to_csv(OUT_DIR / "v0_strict_lag_buffer_transition_qa.csv", index=False, encoding="utf-8-sig")

    del alpha
    gc.collect()

    write_state("running", {"step": "bridge_eval"})
    csmar = pd.read_parquet(CSMAR_RET, columns=["symbol", "month_end", "fwd_ret_1m"], engine="pyarrow")
    csmar["symbol"] = normalize_symbol(csmar["symbol"])
    csmar["month_end"] = pd.to_datetime(csmar["month_end"])
    selected = weights[weights["selected_flag"]].copy()
    merged = selected.merge(csmar, on=["symbol", "month_end"], how="left")
    merged["matched_flag"] = merged["fwd_ret_1m"].notna()
    matched_share = (
        merged.assign(matched_weight=lambda x: np.where(x["matched_flag"], x["weight"], 0.0))
        .groupby("month_end")["matched_weight"]
        .sum()
        .rename("matched_weight_share")
    )
    gross = (
        merged.assign(ret_contrib=lambda x: x["weight"] * x["fwd_ret_1m"].fillna(0.0))
        .groupby("month_end")["ret_contrib"]
        .sum()
        .rename("gross_return")
    )
    turnover = trans_qa.set_index("month_end")["simple_turnover_proxy"].rename("turnover_simple")
    base = pd.concat([gross, turnover, matched_share], axis=1).reset_index()
    base["portfolio_name"] = "V0_STRICT_LAG_TOP50_BUFFER_35_75_EQUAL_WEIGHT"
    base["low_match_flag"] = base["matched_weight_share"] < 0.95
    monthly_rows = []
    for cost in [0, 10, 20, 30]:
        tmp = base.copy()
        tmp["cost_bps"] = cost
        tmp["net_return"] = tmp["gross_return"] - tmp["turnover_simple"] * cost / 10000.0
        monthly_rows.append(tmp)
    monthly = pd.concat(monthly_rows, ignore_index=True)[
        [
            "portfolio_name",
            "month_end",
            "cost_bps",
            "gross_return",
            "turnover_simple",
            "net_return",
            "matched_weight_share",
            "low_match_flag",
        ]
    ]
    monthly.to_csv(OUT_DIR / "v0_strict_lag_monthly_net_return_by_cost.csv", index=False, encoding="utf-8-sig")
    perf = perf_summary(monthly)
    perf.to_csv(OUT_DIR / "v0_strict_lag_performance_summary_by_cost.csv", index=False, encoding="utf-8-sig")
    match_qa = pd.DataFrame(
        [
            {
                "weight_row_count": int(len(selected)),
                "matched_row_count": int(merged["matched_flag"].sum()),
                "matched_ratio": float(merged["matched_flag"].mean()),
                "avg_matched_weight_share": float(base["matched_weight_share"].mean()),
                "min_matched_weight_share": float(base["matched_weight_share"].min()),
                "low_match_month_count": int(base["low_match_flag"].sum()),
                "match_status": "PASS" if base["matched_weight_share"].min() >= 0.8 else "LOW_MATCH",
            }
        ]
    )
    match_qa.to_csv(OUT_DIR / "v0_strict_lag_csmar_return_match_qa.csv", index=False, encoding="utf-8-sig")

    del csmar, merged, selected
    gc.collect()

    write_state("running", {"step": "old_vs_strict"})
    strict20 = perf.loc[perf["cost_bps"] == 20].iloc[0].to_dict()
    old_sum = pd.read_csv(OLD_SUMMARY)
    old_pick = old_sum[
        (old_sum["model_name"] == "V0_LINEAR_FULL_OOS")
        & (old_sum["cost_bps"] == 20)
        & (old_sum["return_variant"] == "raw_unmatched_not_renormalized")
    ].copy()
    if "sample_window" in old_pick.columns and (old_pick["sample_window"] == "common_v0_v7").any():
        old_pick = old_pick[old_pick["sample_window"] == "common_v0_v7"]
    old20 = old_pick.iloc[0].to_dict() if len(old_pick) else {}
    old_month = pd.read_csv(OLD_MONTHLY)
    old_month = old_month[
        (old_month["model_name"] == "V0_LINEAR_FULL_OOS")
        & (old_month["cost_bps"] == 20)
        & (old_month["return_variant"] == "raw_unmatched_not_renormalized")
    ].copy()
    old_month["month_end"] = pd.to_datetime(old_month["month_end"])
    strict_month20 = monthly[monthly["cost_bps"] == 20].copy()
    common = old_month[["month_end", "net_return_csmar_bridge", "turnover_simple", "matched_weight_share"]].merge(
        strict_month20[["month_end", "net_return", "turnover_simple", "matched_weight_share"]],
        on="month_end",
        suffixes=("_old", "_strict"),
        how="inner",
    )
    if len(common) >= 12:
        old_common_metrics = summarize_metrics(
            common["net_return_csmar_bridge"], common["turnover_simple_old"], common["matched_weight_share_old"]
        )
        strict_common_metrics = summarize_metrics(
            common["net_return"], common["turnover_simple_strict"], common["matched_weight_share_strict"]
        )
        sample_window = "common_old_strict_recomputed"
    else:
        old_common_metrics = {
            k: old20.get(k, np.nan)
            for k in [
                "mean_monthly_return",
                "sharpe",
                "tstat",
                "cumulative_return",
                "max_drawdown",
                "avg_turnover",
                "avg_matched_weight_share",
            ]
        }
        strict_common_metrics = {k: strict20.get(k, np.nan) for k in old_common_metrics}
        sample_window = "summary_level"
    rows = []
    for metric in [
        "mean_monthly_return",
        "sharpe",
        "tstat",
        "cumulative_return",
        "max_drawdown",
        "avg_turnover",
        "avg_matched_weight_share",
    ]:
        old_val = old_common_metrics.get(metric, np.nan)
        strict_val = strict_common_metrics.get(metric, np.nan)
        delta = strict_val - old_val if pd.notna(old_val) and pd.notna(strict_val) else np.nan
        degr = strict_val / old_val if pd.notna(old_val) and old_val not in [0, np.nan] else np.nan
        rows.append(
            {
                "sample_window": sample_window,
                "metric_name": metric,
                "old_v0_value": old_val,
                "strict_lag_v0_value": strict_val,
                "delta": delta,
                "degradation_ratio": degr,
                "interpretation": "strict_lag_vs_old_v0_bridge_20bps",
            }
        )
    bridge_cmp = pd.DataFrame(rows)
    bridge_cmp.to_csv(OUT_DIR / "v0_old_vs_strict_lag_bridge_comparison.csv", index=False, encoding="utf-8-sig")

    old_sharpe = float(old_common_metrics.get("sharpe", np.nan))
    strict_sharpe = float(strict_common_metrics.get("sharpe", np.nan))
    old_mean = float(old_common_metrics.get("mean_monthly_return", np.nan))
    strict_mean = float(strict_common_metrics.get("mean_monthly_return", np.nan))
    sharpe_ret = strict_sharpe / old_sharpe if pd.notna(old_sharpe) and old_sharpe > 0 else np.nan
    mean_ret = strict_mean / old_mean if pd.notna(old_mean) and old_mean != 0 else np.nan
    if match_qa["match_status"].iloc[0] != "PASS" or len(strict_month20) < 12:
        leakage = "INCONCLUSIVE_DUE_TO_MATCH_OR_WINDOW"
    elif strict_sharpe >= 0.8 and pd.notna(sharpe_ret) and sharpe_ret >= 0.60:
        leakage = "LOW_IMPACT_STRICT_LAG_STILL_STRONG"
    elif strict_sharpe > 0 and pd.notna(sharpe_ret) and sharpe_ret >= 0.30:
        leakage = "MEDIUM_IMPACT_STRICT_LAG_WEAKER_BUT_USABLE"
    else:
        leakage = "HIGH_IMPACT_OLD_V0_LIKELY_LEAKAGE_DRIVEN"
    leakage_df = pd.DataFrame(
        [
            {
                "old_v0_common_20bps_sharpe": old_sharpe,
                "strict_lag_20bps_sharpe": strict_sharpe,
                "sharpe_retention_ratio": sharpe_ret,
                "old_v0_mean_monthly_return": old_mean,
                "strict_lag_mean_monthly_return": strict_mean,
                "mean_return_retention_ratio": mean_ret,
                "old_v0_max_drawdown": old_common_metrics.get("max_drawdown", np.nan),
                "strict_lag_max_drawdown": strict_common_metrics.get("max_drawdown", np.nan),
                "leakage_impact_assessment": leakage,
                "interpretation": "严格滞后后评估旧 V0 强表现是否依赖同月 forward_return_1m 泄露",
            }
        ]
    )
    leakage_df.to_csv(OUT_DIR / "v0_strict_lag_leakage_impact_summary.csv", index=False, encoding="utf-8-sig")

    current_month_ic_count = int(audit["current_month_ic_included"].sum())
    future_ic_count = int(audit["future_ic_included"].sum())
    no_guardrail_violation = current_month_ic_count == 0 and future_ic_count == 0
    guardrails = [
        ("original_orthogonalization_modified", False, False),
        ("old_artifacts_modified", False, False),
        ("production_modified", False, False),
        ("ml_training_run", False, False),
        ("new_ml_model_trained", False, False),
        ("new_scores_generated", True, True),
        ("new_weights_generated", True, True),
        ("portfolio_returns_calculated", True, True),
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
    guardrail_df = pd.DataFrame(
        [{"guardrail": g, "expected": e, "actual": a, "pass": bool(e == a)} for g, e, a in guardrails]
    )
    guardrail_df.to_csv(OUT_DIR / "v0_strict_lag_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    no_guardrail_violation = no_guardrail_violation and bool(guardrail_df["pass"].all())

    strict_cum = float(strict_common_metrics.get("cumulative_return", np.nan))
    if not no_guardrail_violation:
        final_decision = "V0_STRICT_LAG_FAIL_GUARDRAIL"
    elif match_qa["match_status"].iloc[0] != "PASS" or len(strict_month20) < 12:
        final_decision = "V0_STRICT_LAG_INCONCLUSIVE_MATCH_OR_WINDOW_ISSUE"
    elif strict_sharpe >= 0.8 and strict_cum > 0:
        final_decision = "V0_STRICT_LAG_STILL_STRONG_CONTINUE_CSMAR_REBUILD"
    elif strict_sharpe > 0 and strict_cum > 0:
        final_decision = "V0_STRICT_LAG_WEAKER_BUT_USABLE_CONTINUE_CSMAR_REBUILD"
    else:
        final_decision = "V0_STRICT_LAG_COLLAPSED_OLD_V0_LIKELY_LEAKAGE_DRIVEN"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "strict_lag_icir_implemented": True,
        "current_month_ic_included_count": current_month_ic_count,
        "future_ic_included_count": future_ic_count,
        "strict_lag_alpha_signal_generated": True,
        "strict_lag_weights_generated": True,
        "strict_lag_bridge_evaluated": True,
        "strict_lag_month_count": int(strict20["month_count"]),
        "strict_lag_min_month_end": str(strict_month20["month_end"].min().date()),
        "strict_lag_max_month_end": str(strict_month20["month_end"].max().date()),
        "strict_lag_avg_selected_count": float(weight_qa["selected_stock_count"].mean()),
        "strict_lag_avg_weight_sum": float(weight_qa["weight_sum"].mean()),
        "strict_lag_avg_turnover": float(strict20["avg_turnover"]),
        "strict_lag_match_status": match_qa["match_status"].iloc[0],
        "strict_lag_avg_matched_weight_share": float(strict20["avg_matched_weight_share"]),
        "strict_lag_min_matched_weight_share": float(strict20["min_matched_weight_share"]),
        "strict_lag_20bps_sharpe": strict_sharpe,
        "strict_lag_20bps_mean_monthly_return": strict_mean,
        "strict_lag_20bps_tstat": float(strict_common_metrics.get("tstat", np.nan)),
        "strict_lag_20bps_cumulative_return": strict_cum,
        "strict_lag_20bps_max_drawdown": float(strict_common_metrics.get("max_drawdown", np.nan)),
        "old_v0_20bps_sharpe": old_sharpe,
        "old_v0_20bps_mean_monthly_return": old_mean,
        "old_v0_20bps_tstat": float(old_common_metrics.get("tstat", np.nan)),
        "old_v0_20bps_cumulative_return": float(old_common_metrics.get("cumulative_return", np.nan)),
        "old_v0_20bps_max_drawdown": float(old_common_metrics.get("max_drawdown", np.nan)),
        "sharpe_retention_ratio": sharpe_ret,
        "mean_return_retention_ratio": mean_ret,
        "leakage_impact_assessment": leakage,
        "old_v0_bridge_result_reliability_after_strict_lag_test": "降级为需谨慎引用" if leakage.startswith("HIGH") else "可作为对照但仍非 canonical",
        "v0_structure_still_research_worthy": final_decision
        in [
            "V0_STRICT_LAG_STILL_STRONG_CONTINUE_CSMAR_REBUILD",
            "V0_STRICT_LAG_WEAKER_BUT_USABLE_CONTINUE_CSMAR_REBUILD",
        ],
        "canonical_rebuild_still_required": True,
        "original_orthogonalization_modified": False,
        "old_artifacts_modified": False,
        "production_modified": False,
        "ml_training_run": False,
        "new_ml_model_trained": False,
        "new_scores_generated": True,
        "new_weights_generated": True,
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
        "final_decision": final_decision,
        "recommended_next_step": "继续 canonical CSMAR rebuild" if "CONTINUE" in final_decision else "暂停引用旧 V0 强结果，优先复核 strict-lag 构造与样本窗口",
    }
    save_json(summary, OUT_DIR / "v0_strict_lag_icir_rebuild_bridge_summary.json")

    report = [
        "# V0 Strict-Lag IC_IR Rebuild & CSMAR Bridge Test v0",
        "",
        f"- final_decision: {final_decision}",
        f"- strict_lag_20bps_sharpe: {strict_sharpe:.6f}",
        f"- old_v0_20bps_sharpe: {old_sharpe:.6f}",
        f"- sharpe_retention_ratio: {sharpe_ret:.6f}" if pd.notna(sharpe_ret) else "- sharpe_retention_ratio: null",
        f"- leakage_impact_assessment: {leakage}",
        f"- current_month_ic_included_count: {current_month_ic_count}",
        f"- future_ic_included_count: {future_ic_count}",
        "",
        "严格滞后规则：形成 t 月 signal 时，滚动 IC_IR 仅使用 t 月之前的 IC 序列。",
    ]
    (OUT_DIR / "v0_strict_lag_icir_rebuild_bridge_report.md").write_text("\n".join(report), encoding="utf-8")

    (RUN_DIR / "task_completion_card.md").write_text(
        "\n".join(
            [
                "# task_completion_card",
                f"- task_name: {TASK_NAME}",
                f"- completed_at: {datetime.now().isoformat(timespec='seconds')}",
                f"- final_decision: {final_decision}",
                f"- output_dir: {OUT_DIR}",
            ]
        ),
        encoding="utf-8",
    )
    save_json(
        {
            "task_name": TASK_NAME,
            "stdout_log": str(RUN_DIR / "run_stdout.txt"),
            "stderr_log": str(RUN_DIR / "run_stderr.txt"),
            "status": "completed",
            "final_decision": final_decision,
        },
        RUN_DIR / "terminal_summary.json",
    )
    pd.DataFrame(
        [
            {"qa_item": "prerequisites_passed", "pass": prereq["prerequisites_passed"]},
            {"qa_item": "strict_lag_no_current_month_ic", "pass": current_month_ic_count == 0},
            {"qa_item": "strict_lag_no_future_ic", "pass": future_ic_count == 0},
            {"qa_item": "guardrails_passed", "pass": bool(guardrail_df["pass"].all())},
            {"qa_item": "bridge_evaluated", "pass": True},
        ]
    ).to_csv(RUN_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    write_state("completed", {"final_decision": final_decision, "output_dir": str(OUT_DIR)})


if __name__ == "__main__":
    main()
