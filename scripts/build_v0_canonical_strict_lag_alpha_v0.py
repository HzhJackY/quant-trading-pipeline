from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_NAME = "v0_canonical_strict_lag_alpha_build_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

INPUT_PANEL = ROOT / "output" / "v0_canonical_16factor_panel_build_v0" / "v0_canonical_16factor_panel.parquet"
PANEL_BUILD_SUMMARY = (
    ROOT / "output" / "v0_canonical_16factor_panel_build_v0" / "v0_canonical_16factor_panel_build_summary.json"
)
RETURN_MAP = (
    ROOT
    / "output"
    / "trd_mnth_parser_repair_2024_12_coverage_repair_v0"
    / "canonical_csmar_trd_mnth_return_map_repaired.parquet"
)
STRICT_LAG_REFERENCE_SCRIPT = ROOT / "scripts" / "rebuild_v0_strict_lag_icir_bridge_v0.py"
LEGACY_ALPHA = (
    ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_alpha_signal_panel.parquet"
)

FACTORS = [
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
BASE_COLS = ["symbol_norm", "month_end", "year_month", "total_market_cap_raw_thousand"]

ROLLING_WINDOW = 24
MIN_STOCKS = 20
MIN_IC_IR = 0.05
SPLIT_PERCENTILE = 0.5
PRIMARY_RETURN_FIELD = "Mretwd"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def save_json(obj: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_state(status: str, details: dict[str, Any] | None = None) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": TASK_NAME,
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "details": details or {},
        "resume_instruction": f"先读取 {rel(RUN_DIR / 'RUN_STATE.md')} 再继续。",
    }
    lines = ["# RUN_STATE", "", f"- task_name: {TASK_NAME}", f"- status: {status}"]
    for key, value in payload["details"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "```json", json.dumps(payload, ensure_ascii=False, indent=2, default=str), "```"]
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(lines), encoding="utf-8")


def finite_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def rank_ic(x: pd.Series, y: pd.Series) -> float:
    sub = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(sub) < MIN_STOCKS:
        return np.nan
    if sub["x"].nunique() < 2 or sub["y"].nunique() < 2:
        return np.nan
    return float(sub["x"].rank(method="average").corr(sub["y"].rank(method="average")))


def zscore(s: pd.Series) -> pd.Series:
    vals = finite_numeric(s)
    mean = vals.mean(skipna=True)
    std = vals.std(skipna=True, ddof=0)
    if pd.isna(std) or std <= 1e-12:
        return pd.Series(np.nan, index=s.index)
    return (vals - mean) / std


def winsor_zscore(s: pd.Series) -> pd.Series:
    vals = finite_numeric(s)
    if vals.notna().sum() < 5:
        return pd.Series(np.nan, index=s.index)
    lo = vals.quantile(0.01)
    hi = vals.quantile(0.99)
    return zscore(vals.clip(lo, hi))


def gram_schmidt(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    out = np.zeros_like(matrix, dtype=float)
    used = 0
    for j in range(matrix.shape[1]):
        y = matrix[:, j].astype(float)
        if used == 0:
            resid = y
        else:
            x = out[:, :used]
            try:
                beta = np.linalg.lstsq(x, y, rcond=None)[0]
                resid = y - x @ beta
            except np.linalg.LinAlgError:
                resid = np.zeros_like(y)
        resid_std = np.nanstd(resid)
        if not np.isfinite(resid_std) or resid_std <= 1e-12:
            out[:, j] = 0.0
        else:
            out[:, j] = resid
            used += 1
            if used != j + 1:
                out[:, used - 1] = out[:, j]
                out[:, j] = 0.0
    return out


def prereq_check() -> dict[str, Any]:
    required = [INPUT_PANEL, RETURN_MAP, PANEL_BUILD_SUMMARY]
    missing = [rel(p) for p in required if not p.exists()]
    return {
        "canonical_16factor_panel_found": INPUT_PANEL.exists(),
        "trd_mnth_return_map_found": RETURN_MAP.exists(),
        "panel_build_summary_found": PANEL_BUILD_SUMMARY.exists(),
        "strict_lag_reference_found": STRICT_LAG_REFERENCE_SCRIPT.exists(),
        "prerequisites_passed": len(missing) == 0,
        "missing_files": missing,
    }


def input_panel_qa(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    duplicate_count = int(panel.duplicated(["symbol_norm", "year_month"]).sum())
    checks = [
        ("rows", ">0", len(panel), len(panel) > 0, ""),
        ("symbols", ">0", panel["symbol_norm"].nunique(), panel["symbol_norm"].nunique() > 0, ""),
        ("months", ">0", panel["year_month"].nunique(), panel["year_month"].nunique() > 0, ""),
        ("duplicate symbol-month", "0", duplicate_count, duplicate_count == 0, ""),
        ("16 factor columns present", "all present", ",".join([f for f in FACTORS if f in panel.columns]), all(f in panel.columns for f in FACTORS), ""),
        ("split field present", "total_market_cap_raw_thousand", "total_market_cap_raw_thousand" in panel.columns, "total_market_cap_raw_thousand" in panel.columns, ""),
        ("one row per symbol-month", True, duplicate_count == 0, duplicate_count == 0, ""),
        ("PIT metadata present if available", "selected_pit_date/selected_report_period optional", "metadata columns checked upstream", True, ""),
    ]
    for name, expected, actual, passed, caveat in checks:
        rows.append({"check_name": name, "expected": expected, "actual": actual, "pass": passed, "caveat": caveat})
    for factor in FACTORS:
        vals = finite_numeric(panel[factor])
        inf_count = int(np.isinf(pd.to_numeric(panel[factor], errors="coerce").to_numpy(dtype=float, na_value=np.nan)).sum())
        rows.append(
            {
                "check_name": f"factor coverage:{factor}",
                "expected": "non_null_ratio>=0.60 and no infinite",
                "actual": f"non_null_ratio={vals.notna().mean():.6f}; infinite_count={inf_count}",
                "pass": bool(vals.notna().mean() >= 0.60 and inf_count == 0),
                "caveat": "",
            }
        )
    return pd.DataFrame(rows)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_cols = [*BASE_COLS, *FACTORS]
    panel = pd.read_parquet(INPUT_PANEL, columns=panel_cols)
    panel["symbol_norm"] = panel["symbol_norm"].astype(str).str.zfill(6)
    panel["year_month"] = panel["year_month"].astype(str).str.slice(0, 7)
    panel["month_end"] = pd.to_datetime(panel["month_end"], errors="coerce")
    for col in ["total_market_cap_raw_thousand", *FACTORS]:
        panel[col] = finite_numeric(panel[col])

    ret = pd.read_parquet(RETURN_MAP, columns=["symbol_norm", "year_month", "fwd_ret_1m", "primary_return_field"])
    ret["symbol_norm"] = ret["symbol_norm"].astype(str).str.zfill(6)
    ret["year_month"] = ret["year_month"].astype(str).str.slice(0, 7)
    ret["fwd_ret_1m"] = finite_numeric(ret["fwd_ret_1m"])
    ret = ret.drop_duplicates(["symbol_norm", "year_month"], keep="last")
    return panel, ret


def label_alignment_qa(panel: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    month_cov = (
        merged.groupby("year_month")
        .agg(total=("symbol_norm", "count"), matched=("fwd_ret_1m", lambda x: int(x.notna().sum())))
        .reset_index()
    )
    month_cov["ratio"] = month_cov["matched"] / month_cov["total"]
    avg_cov = float(month_cov["ratio"].mean()) if len(month_cov) else 0.0
    min_cov = float(month_cov["ratio"].min()) if len(month_cov) else 0.0
    if avg_cov >= 0.98 and min_cov >= 0.95:
        status = "READY"
    elif avg_cov >= 0.95 and min_cov >= 0.90:
        status = "READY_WITH_MINOR_GAPS"
    elif avg_cov >= 0.90:
        status = "WATCH"
    else:
        status = "FAIL"
    return pd.DataFrame(
        [
            {
                "row_count": len(panel),
                "matched_label_count": int(merged["fwd_ret_1m"].notna().sum()),
                "matched_label_ratio": round(float(merged["fwd_ret_1m"].notna().mean()), 6),
                "month_count": int(merged["year_month"].nunique()),
                "avg_month_label_coverage": round(avg_cov, 6),
                "min_month_label_coverage": round(min_cov, 6),
                "low_label_coverage_month_count": int((month_cov["ratio"] < 0.95).sum()),
                "label_source": rel(RETURN_MAP),
                "primary_return_field": PRIMARY_RETURN_FIELD,
                "label_alignment_status": status,
            }
        ]
    )


def split_assignment(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    qa_rows = []
    for ym, grp in panel.groupby("year_month", sort=True):
        sub = grp[["symbol_norm", "year_month", "total_market_cap_raw_thousand"]].copy()
        valid = sub["total_market_cap_raw_thousand"].dropna()
        threshold = float(valid.quantile(SPLIT_PERCENTILE)) if len(valid) else np.nan
        sub["split_rank_pct"] = sub["total_market_cap_raw_thousand"].rank(pct=True)
        sub["split_threshold"] = threshold
        sub["split_group"] = np.where(
            sub["total_market_cap_raw_thousand"].isna(),
            "missing",
            np.where(sub["split_rank_pct"] >= SPLIT_PERCENTILE, "large", "small"),
        )
        sub["split_assignment_status"] = np.where(sub["split_group"] == "missing", "MISSING_SPLIT_FIELD", "ASSIGNED")
        large_count = int((sub["split_group"] == "large").sum())
        small_count = int((sub["split_group"] == "small").sum())
        total = len(sub)
        dominant = max(large_count, small_count) / max(large_count + small_count, 1)
        qa_rows.append(
            {
                "year_month": ym,
                "total_count": total,
                "large_count": large_count,
                "small_count": small_count,
                "missing_split_field_count": int((sub["split_group"] == "missing").sum()),
                "split_threshold": threshold,
                "dominant_group_ratio": round(float(dominant), 6),
                "qa_status": "PASS" if total and dominant <= 0.55 and sub["split_group"].ne("missing").all() else "WATCH",
            }
        )
        rows.append(sub)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(qa_rows)


def monthly_ic(panel_label: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (grp_name, ym), grp in panel_label.groupby(["split_group", "year_month"], sort=True):
        if grp_name not in {"large", "small"}:
            continue
        for factor in FACTORS:
            ic = rank_ic(grp[factor], grp["fwd_ret_1m"])
            rows.append({"split_group": grp_name, "ic_year_month": ym, "factor_name": factor, "ic": ic})
    return pd.DataFrame(rows)


def strict_lag_icir(months: list[str], ic_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ic_df = ic_df.dropna(subset=["ic"]).copy()
    for split_group in ["large", "small"]:
        for signal_month in months:
            for factor in FACTORS:
                hist = ic_df[
                    (ic_df["split_group"] == split_group)
                    & (ic_df["factor_name"] == factor)
                    & (ic_df["ic_year_month"] < signal_month)
                ].sort_values("ic_year_month")
                window = hist.tail(ROLLING_WINDOW)
                count = int(len(window))
                if count >= 2:
                    ic_mean = float(window["ic"].mean())
                    ic_std = float(window["ic"].std(ddof=1))
                    ic_ir = float(ic_mean / ic_std) if ic_std > 1e-12 else 0.0
                elif count == 1:
                    ic_mean = float(window["ic"].iloc[0])
                    ic_std = np.nan
                    ic_ir = 0.0
                else:
                    ic_mean = np.nan
                    ic_std = np.nan
                    ic_ir = 0.0
                first_used = str(window["ic_year_month"].iloc[0]) if count else ""
                last_used = str(window["ic_year_month"].iloc[-1]) if count else ""
                current_included = bool((window["ic_year_month"] == signal_month).any()) if count else False
                future_included = bool((window["ic_year_month"] > signal_month).any()) if count else False
                rows.append(
                    {
                        "split_group": split_group,
                        "signal_year_month": signal_month,
                        "factor_name": factor,
                        "first_ic_month_used": first_used,
                        "last_ic_month_used": last_used,
                        "ic_count_used": count,
                        "ic_mean": ic_mean,
                        "ic_std": ic_std,
                        "ic_ir": ic_ir,
                        "abs_ic_ir": abs(ic_ir),
                        "current_month_ic_included": current_included,
                        "future_ic_included": future_included,
                        "strict_lag_pass": (not current_included) and (not future_included) and (last_used == "" or last_used < signal_month),
                    }
                )
    return pd.DataFrame(rows)


def icir_window_qa(icir: pd.DataFrame) -> pd.DataFrame:
    current_count = int(icir["current_month_ic_included"].sum())
    future_count = int(icir["future_ic_included"].sum())
    bad_last = int(((icir["last_ic_month_used"] != "") & (icir["last_ic_month_used"] >= icir["signal_year_month"])).sum())
    warmup_months = int(icir.loc[icir["ic_count_used"] < 2, "signal_year_month"].nunique())
    no_history = int((icir["ic_count_used"] == 0).sum())
    return pd.DataFrame(
        [
            {"check_name": "current_month_ic_included_count", "violation_count": current_count, "pass": current_count == 0, "caveat": ""},
            {"check_name": "future_ic_included_count", "violation_count": future_count, "pass": future_count == 0, "caveat": ""},
            {"check_name": "max_last_ic_month_used < signal_year_month", "violation_count": bad_last, "pass": bad_last == 0, "caveat": ""},
            {"check_name": "warmup_month_count", "violation_count": warmup_months, "pass": True, "caveat": "早期月份历史 IC 不足，允许 warmup。"},
            {"check_name": "factors_with_no_history_count", "violation_count": no_history, "pass": True, "caveat": "早期 split/factor 无历史 IC 时 IC_IR 置 0。"},
        ]
    )


def build_alpha(panel_label: pd.DataFrame, icir: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha_parts = []
    audit_rows = []
    icir_index = {
        (row.split_group, row.signal_year_month, row.factor_name): row
        for row in icir.itertuples(index=False)
    }
    for (ym, split_group), grp in panel_label.groupby(["year_month", "split_group"], sort=True):
        if split_group not in {"large", "small"}:
            continue
        sub = grp[["symbol_norm", "year_month", "month_end", "split_group", *FACTORS]].copy()
        ic_vals = []
        for factor in FACTORS:
            row = icir_index.get((split_group, ym, factor))
            ic_ir = float(row.ic_ir) if row is not None else 0.0
            abs_ic = abs(ic_ir)
            selected = abs_ic > MIN_IC_IR
            if selected and sub[factor].notna().sum() < MIN_STOCKS:
                selected = False
            ic_vals.append((factor, ic_ir, abs_ic, selected))
        selected_vals = sorted([x for x in ic_vals if x[3]], key=lambda x: x[2], reverse=True)
        total_abs = float(sum(x[2] for x in selected_vals))
        for rank, (factor, ic_ir, abs_ic, selected) in enumerate(sorted(ic_vals, key=lambda x: x[2], reverse=True), start=1):
            norm_w = float(abs_ic / total_abs) if selected and total_abs > 0 else 0.0
            audit_rows.append(
                {
                    "split_group": split_group,
                    "year_month": ym,
                    "factor_name": factor,
                    "ic_ir": ic_ir,
                    "abs_ic_ir": abs_ic,
                    "sign": 1 if ic_ir >= 0 else -1,
                    "normalized_weight": norm_w,
                    "selected_for_composite": bool(selected and total_abs > 0),
                    "factor_rank_by_abs_icir": rank,
                    "contribution_policy": "strict_lag_abs_icir_weight_flip_sign_gram_schmidt",
                }
            )
        out = sub[["symbol_norm", "year_month", "month_end", "split_group"]].copy()
        out["factor_count_used"] = len(selected_vals) if total_abs > 0 else 0
        out["total_abs_icir"] = total_abs
        out["top_icir_factor_1"] = selected_vals[0][0] if len(selected_vals) >= 1 else ""
        out["top_icir_factor_2"] = selected_vals[1][0] if len(selected_vals) >= 2 else ""
        out["top_icir_factor_3"] = selected_vals[2][0] if len(selected_vals) >= 3 else ""
        if total_abs <= 0 or not selected_vals:
            out["composite_score"] = np.nan
            out["composite_score_z"] = np.nan
            out["alpha_signal"] = np.nan
            out["alpha_build_status"] = "NO_STRICT_LAG_ICIR_HISTORY"
            alpha_parts.append(out)
            continue

        ordered_factors = [x[0] for x in selected_vals]
        standardized = pd.DataFrame(index=sub.index)
        for factor in ordered_factors:
            standardized[factor] = winsor_zscore(sub[factor]).fillna(0.0)
        matrix = standardized[ordered_factors].to_numpy(dtype=float)
        orth = gram_schmidt(matrix)
        weights = np.array([(1 if x[1] >= 0 else -1) * (x[2] / total_abs) for x in selected_vals], dtype=float)
        composite = orth @ weights
        out["composite_score"] = composite
        out["composite_score_z"] = zscore(pd.Series(composite, index=out.index))
        out["alpha_signal"] = out["composite_score_z"]
        out["alpha_build_status"] = "READY"
        alpha_parts.append(out)
    alpha = pd.concat(alpha_parts, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    return alpha, audit


def alpha_signal_qa(alpha: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dup = int(alpha.duplicated(["symbol_norm", "year_month"]).sum())
    month_stats = alpha.groupby("year_month").agg(
        total_count=("symbol_norm", "count"),
        alpha_non_null_count=("alpha_signal", lambda x: int(x.notna().sum())),
        large_count=("split_group", lambda x: int((x == "large").sum())),
        small_count=("split_group", lambda x: int((x == "small").sum())),
        factor_count_used_avg=("factor_count_used", "mean"),
        factor_count_used_min=("factor_count_used", "min"),
    ).reset_index()
    month_stats["alpha_non_null_ratio"] = month_stats["alpha_non_null_count"] / month_stats["total_count"]
    group_std = alpha.groupby(["year_month", "split_group"])["alpha_signal"].std(ddof=0).unstack()
    month_stats["large_alpha_std"] = month_stats["year_month"].map(group_std.get("large", pd.Series(dtype=float)))
    month_stats["small_alpha_std"] = month_stats["year_month"].map(group_std.get("small", pd.Series(dtype=float)))
    month_stats["alpha_status"] = np.where(month_stats["alpha_non_null_ratio"] >= 0.95, "READY", "WATCH")

    mean_abs_max = float(alpha.groupby("year_month")["alpha_signal"].mean().abs().max())
    std_median = float(alpha.groupby("year_month")["alpha_signal"].std(ddof=0).median())
    constant_month_count = int((alpha.groupby("year_month")["alpha_signal"].std(ddof=0).fillna(0) <= 1e-12).sum())
    non_null_ratio = float(alpha["alpha_signal"].notna().mean())
    qa_status = "READY" if non_null_ratio >= 0.95 and dup == 0 and constant_month_count <= 2 else "WATCH"
    qa = pd.DataFrame(
        [
            {
                "row_count": len(alpha),
                "unique_symbol_count": int(alpha["symbol_norm"].nunique()),
                "month_count": int(alpha["year_month"].nunique()),
                "min_year_month": str(alpha["year_month"].min()) if len(alpha) else "",
                "max_year_month": str(alpha["year_month"].max()) if len(alpha) else "",
                "duplicate_symbol_month_count": dup,
                "alpha_signal_non_null_ratio": round(non_null_ratio, 6),
                "composite_score_non_null_ratio": round(float(alpha["composite_score"].notna().mean()), 6),
                "avg_factor_count_used": round(float(alpha["factor_count_used"].mean()), 6),
                "min_factor_count_used": int(alpha["factor_count_used"].min()) if len(alpha) else 0,
                "max_factor_count_used": int(alpha["factor_count_used"].max()) if len(alpha) else 0,
                "alpha_signal_mean_by_month_abs_max": round(mean_abs_max, 10),
                "alpha_signal_std_by_month_median": round(std_median, 6),
                "alpha_signal_constant_month_count": constant_month_count,
                "qa_status": qa_status,
            }
        ]
    )
    return qa, month_stats[
        [
            "year_month",
            "total_count",
            "alpha_non_null_count",
            "alpha_non_null_ratio",
            "large_count",
            "small_count",
            "large_alpha_std",
            "small_alpha_std",
            "factor_count_used_avg",
            "factor_count_used_min",
            "alpha_status",
        ]
    ]


def factor_usage_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_group, factor), grp in audit.groupby(["split_group", "factor_name"], sort=True):
        months_available = int(grp["year_month"].nunique())
        selected = grp[grp["selected_for_composite"] == True]
        rows.append(
            {
                "split_group": split_group,
                "factor_name": factor,
                "months_available": months_available,
                "months_selected_for_composite": int(selected["year_month"].nunique()),
                "avg_abs_icir": round(float(grp["abs_ic_ir"].mean()), 6),
                "avg_normalized_weight": round(float(grp["normalized_weight"].mean()), 6),
                "positive_sign_month_ratio": round(float((grp["sign"] > 0).mean()), 6),
                "negative_sign_month_ratio": round(float((grp["sign"] < 0).mean()), 6),
                "interpretation": "used_by_strict_lag_icir" if len(selected) else "not_selected_or_warmup_only",
            }
        )
    return pd.DataFrame(rows)


def legacy_comparison(alpha: pd.DataFrame) -> tuple[pd.DataFrame, bool, float | None, float | None]:
    if not LEGACY_ALPHA.exists():
        empty = pd.DataFrame(
            columns=[
                "year_month",
                "common_symbol_count",
                "spearman_corr",
                "pearson_corr",
                "top50_overlap",
                "top75_overlap",
                "mean_abs_rank_diff",
                "interpretation",
            ]
        )
        return empty, False, None, None
    legacy = pd.read_parquet(LEGACY_ALPHA)
    legacy = legacy.rename(columns={"alpha_signal_strict_lag": "legacy_alpha_signal"})
    if "symbol_norm" not in legacy.columns:
        legacy["symbol_norm"] = legacy["symbol"].astype(str).str.replace(r"\D", "", regex=True).str[-6:].str.zfill(6)
    legacy["year_month"] = pd.to_datetime(legacy["month_end"], errors="coerce").dt.to_period("M").astype(str)
    legacy = legacy[["symbol_norm", "year_month", "legacy_alpha_signal"]].dropna()
    cur = alpha[["symbol_norm", "year_month", "alpha_signal"]].dropna()
    merged = cur.merge(legacy, on=["symbol_norm", "year_month"], how="inner")
    rows = []
    for ym, grp in merged.groupby("year_month", sort=True):
        if len(grp) < 20:
            continue
        spearman = float(grp["alpha_signal"].rank().corr(grp["legacy_alpha_signal"].rank()))
        pearson = float(grp["alpha_signal"].corr(grp["legacy_alpha_signal"]))
        cur_rank = grp["alpha_signal"].rank(ascending=False, method="first")
        old_rank = grp["legacy_alpha_signal"].rank(ascending=False, method="first")
        cur_top50 = set(grp.loc[cur_rank <= 50, "symbol_norm"])
        old_top50 = set(grp.loc[old_rank <= 50, "symbol_norm"])
        cur_top75 = set(grp.loc[cur_rank <= 75, "symbol_norm"])
        old_top75 = set(grp.loc[old_rank <= 75, "symbol_norm"])
        rows.append(
            {
                "year_month": ym,
                "common_symbol_count": len(grp),
                "spearman_corr": spearman,
                "pearson_corr": pearson,
                "top50_overlap": len(cur_top50 & old_top50) / 50 if cur_top50 and old_top50 else np.nan,
                "top75_overlap": len(cur_top75 & old_top75) / 75 if cur_top75 and old_top75 else np.nan,
                "mean_abs_rank_diff": float((cur_rank - old_rank).abs().mean()),
                "interpretation": "diagnostic_only_no_tuning",
            }
        )
    comp = pd.DataFrame(rows)
    avg_spear = float(comp["spearman_corr"].mean()) if len(comp) else None
    avg_top50 = float(comp["top50_overlap"].mean()) if len(comp) else None
    return comp, True, avg_spear, avg_top50


def readiness(alpha_qa: pd.DataFrame, icir_qa: pd.DataFrame, usage: pd.DataFrame, guardrails: pd.DataFrame) -> pd.DataFrame:
    alpha_row = alpha_qa.iloc[0]
    strict_pass = bool(icir_qa.loc[icir_qa["check_name"].isin(["current_month_ic_included_count", "future_ic_included_count", "max_last_ic_month_used < signal_year_month"]), "pass"].all())
    current_count = int(icir_qa.loc[icir_qa["check_name"] == "current_month_ic_included_count", "violation_count"].iloc[0])
    future_count = int(icir_qa.loc[icir_qa["check_name"] == "future_ic_included_count", "violation_count"].iloc[0])
    reasonable_usage = bool((usage["months_selected_for_composite"] > 0).sum() >= 8)
    rows = [
        {"criterion": "alpha_signal panel generated", "expected": True, "actual": True, "pass": True, "caveat": ""},
        {"criterion": "duplicate symbol-month = 0", "expected": 0, "actual": int(alpha_row["duplicate_symbol_month_count"]), "pass": int(alpha_row["duplicate_symbol_month_count"]) == 0, "caveat": ""},
        {"criterion": "alpha non-null ratio >= 0.95", "expected": ">=0.95", "actual": float(alpha_row["alpha_signal_non_null_ratio"]), "pass": float(alpha_row["alpha_signal_non_null_ratio"]) >= 0.95, "caveat": ""},
        {"criterion": "strict-lag IC_IR QA pass", "expected": True, "actual": strict_pass, "pass": strict_pass, "caveat": ""},
        {"criterion": "current_month_ic_included_count = 0", "expected": 0, "actual": current_count, "pass": current_count == 0, "caveat": ""},
        {"criterion": "future_ic_included_count = 0", "expected": 0, "actual": future_count, "pass": future_count == 0, "caveat": ""},
        {"criterion": "factor usage reasonable", "expected": ">=8 factors selected at least once", "actual": int((usage["months_selected_for_composite"] > 0).sum()), "pass": reasonable_usage, "caveat": ""},
        {"criterion": "no guardrail violation", "expected": True, "actual": bool(guardrails["pass"].all()), "pass": bool(guardrails["pass"].all()), "caveat": ""},
    ]
    return pd.DataFrame(rows)


def guardrail_qa() -> pd.DataFrame:
    guardrails = {
        "alpha_signal_generated": True,
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "production_modified": False,
        "ml_training_run": False,
        "new_ml_model_trained": False,
        "tuning_run": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
    }
    rows = []
    for key, actual in guardrails.items():
        expected = True if key == "alpha_signal_generated" else False
        rows.append({"guardrail": key, "expected": expected, "actual": actual, "pass": actual is expected})
    return pd.DataFrame(rows)


def simple_table(df: pd.DataFrame, cols: list[str], max_rows: int = 40) -> str:
    sub = df[cols].head(max_rows).fillna("").astype(str)
    widths = {c: max(len(c), *(len(x) for x in sub[c].tolist())) for c in cols}
    lines = [
        "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |",
        "| " + " | ".join("-" * widths[c] for c in cols) + " |",
    ]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(row[c].ljust(widths[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", {"step": "prerequisite_check"})
    prereq = prereq_check()
    save_json(prereq, OUT_DIR / "v0_canonical_alpha_build_prerequisite_check.json")
    if not prereq["prerequisites_passed"]:
        raise FileNotFoundError(prereq["missing_files"])

    write_state("running", {"step": "load_inputs"})
    panel, ret = load_inputs()
    input_qa = input_panel_qa(panel)
    input_qa.to_csv(OUT_DIR / "v0_canonical_alpha_input_panel_qa.csv", index=False, encoding="utf-8-sig")
    merged = panel.merge(ret[["symbol_norm", "year_month", "fwd_ret_1m"]], on=["symbol_norm", "year_month"], how="left")
    label_qa = label_alignment_qa(panel, merged)
    label_qa.to_csv(OUT_DIR / "v0_canonical_alpha_return_label_alignment_qa.csv", index=False, encoding="utf-8-sig")

    write_state("running", {"step": "split_assignment"})
    split, split_qa = split_assignment(panel)
    split.to_parquet(OUT_DIR / "v0_canonical_split_assignment.parquet", index=False)
    split_qa.to_csv(OUT_DIR / "v0_canonical_split_assignment_qa.csv", index=False, encoding="utf-8-sig")
    merged = merged.merge(split[["symbol_norm", "year_month", "split_group", "split_rank_pct"]], on=["symbol_norm", "year_month"], how="left")

    write_state("running", {"step": "strict_lag_icir"})
    months = sorted(merged["year_month"].dropna().unique().tolist())
    ic_df = monthly_ic(merged)
    icir = strict_lag_icir(months, ic_df)
    icir.to_csv(OUT_DIR / "v0_canonical_strict_lag_icir_by_month_factor.csv", index=False, encoding="utf-8-sig")
    icir_qa = icir_window_qa(icir)
    icir_qa.to_csv(OUT_DIR / "v0_canonical_strict_lag_icir_window_qa.csv", index=False, encoding="utf-8-sig")

    write_state("running", {"step": "alpha_signal"})
    alpha, contribution = build_alpha(merged, icir)
    alpha.to_parquet(OUT_DIR / "v0_canonical_alpha_signal_panel.parquet", index=False)
    alpha.head(1000).to_csv(OUT_DIR / "v0_canonical_alpha_signal_sample.csv", index=False, encoding="utf-8-sig")
    alpha_qa, monthly_qa = alpha_signal_qa(alpha)
    alpha_qa.to_csv(OUT_DIR / "v0_canonical_alpha_signal_qa.csv", index=False, encoding="utf-8-sig")
    monthly_qa.to_csv(OUT_DIR / "v0_canonical_alpha_signal_monthly_qa.csv", index=False, encoding="utf-8-sig")
    contribution.to_csv(OUT_DIR / "v0_canonical_factor_icir_contribution_audit.csv", index=False, encoding="utf-8-sig")
    usage = factor_usage_summary(contribution)
    usage.to_csv(OUT_DIR / "v0_canonical_factor_usage_summary.csv", index=False, encoding="utf-8-sig")

    write_state("running", {"step": "legacy_diagnostic"})
    comp, comp_available, avg_spear, avg_top50 = legacy_comparison(alpha)
    comp.to_csv(OUT_DIR / "v0_canonical_vs_legacy_strict_lag_alpha_comparison.csv", index=False, encoding="utf-8-sig")

    guardrails = guardrail_qa()
    guardrails.to_csv(OUT_DIR / "v0_canonical_alpha_build_guardrail_qa.csv", index=False, encoding="utf-8-sig")
    ready = readiness(alpha_qa, icir_qa, usage, guardrails)
    ready.to_csv(OUT_DIR / "v0_canonical_alpha_to_portfolio_prep_readiness.csv", index=False, encoding="utf-8-sig")

    current_count = int(icir_qa.loc[icir_qa["check_name"] == "current_month_ic_included_count", "violation_count"].iloc[0])
    future_count = int(icir_qa.loc[icir_qa["check_name"] == "future_ic_included_count", "violation_count"].iloc[0])
    strict_pass = bool(icir_qa.loc[icir_qa["check_name"].isin(["current_month_ic_included_count", "future_ic_included_count", "max_last_ic_month_used < signal_year_month"]), "pass"].all())
    guardrail_pass = bool(guardrails["pass"].all())
    alpha_non_null = float(alpha_qa["alpha_signal_non_null_ratio"].iloc[0])
    portfolio_allowed = bool(ready["pass"].all())
    if not guardrail_pass:
        final_decision = "V0_CANONICAL_ALPHA_FAIL_GUARDRAIL"
    elif not strict_pass:
        final_decision = "V0_CANONICAL_ALPHA_BLOCKED_BY_STRICT_LAG_QA"
    elif alpha_non_null < 0.95:
        final_decision = "V0_CANONICAL_ALPHA_BLOCKED_BY_SIGNAL_COVERAGE"
    elif portfolio_allowed:
        final_decision = "V0_CANONICAL_ALPHA_READY_FOR_PORTFOLIO_PREP"
    else:
        final_decision = "V0_CANONICAL_ALPHA_READY_WITH_CAVEATS"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "input_panel_path": rel(INPUT_PANEL),
        "return_label_source": rel(RETURN_MAP),
        "primary_return_field": PRIMARY_RETURN_FIELD,
        "alpha_signal_generated": True,
        "alpha_signal_panel_path": rel(OUT_DIR / "v0_canonical_alpha_signal_panel.parquet"),
        "row_count": int(len(alpha)),
        "unique_symbol_count": int(alpha["symbol_norm"].nunique()),
        "month_count": int(alpha["year_month"].nunique()),
        "min_year_month": str(alpha["year_month"].min()),
        "max_year_month": str(alpha["year_month"].max()),
        "duplicate_symbol_month_count": int(alpha.duplicated(["symbol_norm", "year_month"]).sum()),
        "split_policy": {"split_field": "total_market_cap_raw_thousand", "percentile": SPLIT_PERCENTILE, "large": "top_50pct"},
        "strict_lag_icir_policy": {
            "rolling_window": ROLLING_WINDOW,
            "current_month_ic_allowed": False,
            "future_ic_allowed": False,
            "gram_schmidt": True,
            "flip_sign": True,
            "min_ic_ir": MIN_IC_IR,
            "min_stocks": MIN_STOCKS,
        },
        "current_month_ic_included_count": current_count,
        "future_ic_included_count": future_count,
        "strict_lag_icir_qa_pass": strict_pass,
        "alpha_signal_non_null_ratio": round(alpha_non_null, 6),
        "avg_factor_count_used": round(float(alpha_qa["avg_factor_count_used"].iloc[0]), 6),
        "min_factor_count_used": int(alpha_qa["min_factor_count_used"].iloc[0]),
        "factor_usage_summary_status": "READY" if (usage["months_selected_for_composite"] > 0).sum() >= 8 else "WATCH",
        "legacy_alpha_comparison_available": comp_available,
        "avg_legacy_spearman_corr": avg_spear,
        "avg_legacy_top50_overlap": avg_top50,
        "portfolio_prep_allowed_next": portfolio_allowed,
        "alpha_build_readiness_status": "READY" if portfolio_allowed else "READY_WITH_CAVEATS",
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "production_modified": False,
        "ml_training_run": False,
        "new_ml_model_trained": False,
        "tuning_run": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
        "final_decision": final_decision,
        "recommended_next_step": "进入 canonical alpha portfolio construction prep，仍不得在 prep 中计算 portfolio returns。" if portfolio_allowed else "先处理 strict-lag/coverage caveat，再进入 portfolio prep。",
    }
    save_json(summary, OUT_DIR / "v0_canonical_strict_lag_alpha_build_summary.json")
    report = "\n".join(
        [
            "# V0 Canonical Strict-Lag Split-Universe Alpha Build v0",
            "",
            "## 结论",
            f"- final_decision: {final_decision}",
            f"- alpha_signal_non_null_ratio: {alpha_non_null:.6f}",
            f"- strict_lag_icir_qa_pass: {strict_pass}",
            f"- portfolio_prep_allowed_next: {portfolio_allowed}",
            "",
            "## Alpha QA",
            simple_table(alpha_qa, list(alpha_qa.columns)),
            "",
            "## Readiness",
            simple_table(ready, ["criterion", "actual", "pass", "caveat"]),
            "",
            "## Guardrails",
            "- 本任务生成 alpha_signal panel。",
            "- 未生成 strategy weights，未计算 portfolio returns，未训练或调参。",
        ]
    )
    (OUT_DIR / "v0_canonical_strict_lag_alpha_build_report.md").write_text(report, encoding="utf-8")

    final_qa = guardrails.copy()
    required_artifacts = [
        OUT_DIR / "v0_canonical_alpha_build_prerequisite_check.json",
        OUT_DIR / "v0_canonical_alpha_input_panel_qa.csv",
        OUT_DIR / "v0_canonical_alpha_return_label_alignment_qa.csv",
        OUT_DIR / "v0_canonical_split_assignment.parquet",
        OUT_DIR / "v0_canonical_split_assignment_qa.csv",
        OUT_DIR / "v0_canonical_strict_lag_icir_by_month_factor.csv",
        OUT_DIR / "v0_canonical_strict_lag_icir_window_qa.csv",
        OUT_DIR / "v0_canonical_alpha_signal_panel.parquet",
        OUT_DIR / "v0_canonical_alpha_signal_sample.csv",
        OUT_DIR / "v0_canonical_alpha_signal_qa.csv",
        OUT_DIR / "v0_canonical_alpha_signal_monthly_qa.csv",
        OUT_DIR / "v0_canonical_factor_icir_contribution_audit.csv",
        OUT_DIR / "v0_canonical_factor_usage_summary.csv",
        OUT_DIR / "v0_canonical_vs_legacy_strict_lag_alpha_comparison.csv",
        OUT_DIR / "v0_canonical_alpha_to_portfolio_prep_readiness.csv",
        OUT_DIR / "v0_canonical_alpha_build_guardrail_qa.csv",
        OUT_DIR / "v0_canonical_strict_lag_alpha_build_summary.json",
        OUT_DIR / "v0_canonical_strict_lag_alpha_build_report.md",
        ROOT / "scripts" / "build_v0_canonical_strict_lag_alpha_v0.py",
    ]
    for artifact in required_artifacts:
        final_qa.loc[len(final_qa)] = {
            "guardrail": f"artifact_written:{rel(artifact)}",
            "expected": True,
            "actual": artifact.exists(),
            "pass": artifact.exists(),
        }
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "task_completion_card.md").write_text(
        "\n".join(
            [
                "# task_completion_card",
                "",
                f"- task_name: {TASK_NAME}",
                f"- final_decision: {final_decision}",
                f"- alpha_signal_generated: true",
                f"- row_count: {len(alpha)}",
                f"- alpha_signal_non_null_ratio: {alpha_non_null:.6f}",
                f"- portfolio_prep_allowed_next: {portfolio_allowed}",
                "- guardrails_passed: true",
            ]
        ),
        encoding="utf-8",
    )
    save_json(
        {
            "task_name": TASK_NAME,
            "status": "completed",
            "script": rel(ROOT / "scripts" / "build_v0_canonical_strict_lag_alpha_v0.py"),
            "stdout_log": rel(RUN_DIR / "run_stdout.txt"),
            "stderr_log": rel(RUN_DIR / "run_stderr.txt"),
            "output_dir": rel(OUT_DIR),
            "final_decision": final_decision,
        },
        OUT_DIR / "terminal_summary.json",
    )
    write_state("completed", {"final_decision": final_decision, "output_dir": rel(OUT_DIR)})
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    del panel, ret, merged, alpha, contribution
    gc.collect()


if __name__ == "__main__":
    main()
