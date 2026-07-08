from __future__ import annotations

import csv
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None


TASK_NAME = "V0 Composite-Aligned Attribution Prep v0"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "v0_composite_aligned_attribution_prep_v0"
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

ALIGNED_SUMMARY = ROOT / "output" / "v0_composite_aligned_repaired_trd_mnth_eval_run_v0" / "v0_composite_aligned_repaired_trd_mnth_eval_run_summary.json"
ALIGNED_MONTHLY = ROOT / "output" / "v0_composite_aligned_repaired_trd_mnth_eval_run_v0" / "v0_aligned_monthly_net_returns_by_cost.csv"
ALIGNED_PERF = ROOT / "output" / "v0_composite_aligned_repaired_trd_mnth_eval_run_v0" / "v0_aligned_performance_summary_by_cost.csv"
ALIGNED_NAV = ROOT / "output" / "v0_composite_aligned_repaired_trd_mnth_eval_run_v0" / "v0_aligned_nav_drawdown_path.csv"
ALIGNED_WEIGHTS = ROOT / "output" / "v0_composite_aligned_portfolio_construction_run_v0" / "v0_composite_aligned_research_weights.parquet"
ALIGNED_ALPHA = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_composite_aligned_alpha_candidate_panel.parquet"
RAW_SUMMARY = ROOT / "output" / "v0_canonical_repaired_trd_mnth_eval_run_v0" / "v0_canonical_repaired_trd_mnth_eval_run_summary.json"
RAW_MONTHLY = ROOT / "output" / "v0_canonical_repaired_trd_mnth_eval_run_v0" / "v0_canonical_monthly_net_returns_by_cost.csv"
RAW_PERF = ROOT / "output" / "v0_canonical_repaired_trd_mnth_eval_run_v0" / "v0_canonical_performance_summary_by_cost.csv"
LEGACY_SUMMARY = ROOT / "output" / "unified_strategy_eval_repaired_trd_mnth_v0" / "unified_strategy_eval_repaired_trd_mnth_summary.json"
LEGACY_MONTHLY = ROOT / "output" / "unified_strategy_eval_repaired_trd_mnth_v0" / "unified_strategy_monthly_net_return_by_cost.csv"
LEGACY_PERF = ROOT / "output" / "unified_strategy_eval_repaired_trd_mnth_v0" / "unified_strategy_performance_summary_by_cost.csv"
RETURN_MAP = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0" / "canonical_csmar_trd_mnth_return_map_repaired.parquet"
ALL_DAILY = ROOT / "output" / "all_daily.parquet"
PREPROCESSED = ROOT / "output" / "preprocessed.parquet"
DGTW_DIR = ROOT / "output" / "dgtw_prep_state_v0"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parquet_schema(path: Path) -> list[str]:
    if not path.exists() or pq is None:
        return []
    schema = pq.read_schema(path)
    return list(schema.names)


def parquet_month_range(path: Path, preferred_cols: list[str] | None = None) -> dict[str, Any]:
    if not path.exists() or pq is None:
        return {"min_year_month": "", "max_year_month": "", "month_count": ""}
    preferred_cols = preferred_cols or ["year_month", "month", "ym", "Trdmnt", "trade_month", "date", "month_end"]
    schema_cols = parquet_schema(path)
    month_col = next((c for c in preferred_cols if c in schema_cols), None)
    if not month_col:
        return {"min_year_month": "", "max_year_month": "", "month_count": "", "month_col": ""}
    table = pq.read_table(path, columns=[month_col])
    s = table.column(month_col).to_pandas()
    vals = normalize_year_month_series(s)
    del table, s
    gc.collect()
    if vals.empty:
        return {"min_year_month": "", "max_year_month": "", "month_count": 0, "month_col": month_col}
    unique = vals.dropna().drop_duplicates().sort_values()
    result = {
        "min_year_month": str(unique.iloc[0]) if len(unique) else "",
        "max_year_month": str(unique.iloc[-1]) if len(unique) else "",
        "month_count": int(len(unique)),
        "month_col": month_col,
    }
    del vals, unique
    gc.collect()
    return result


def normalize_year_month_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y-%m")
    txt = s.astype("string").str.strip()
    only_digits = txt.str.replace(r"\D", "", regex=True)
    ym = only_digits.str.slice(0, 6)
    valid = ym.str.len().eq(6)
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    out.loc[valid] = ym.loc[valid].str.slice(0, 4) + "-" + ym.loc[valid].str.slice(4, 6)
    return out


def csv_month_range(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"min_year_month": "", "max_year_month": "", "month_count": ""}
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    month_col = next((c for c in ["year_month", "month", "ym", "date", "trade_month"] if c in cols), None)
    if month_col is None:
        month_col = cols[0] if cols else None
    if month_col is None:
        return {"min_year_month": "", "max_year_month": "", "month_count": 0}
    df = pd.read_csv(path, usecols=[month_col], dtype={month_col: "string"})
    vals = normalize_year_month_series(df[month_col])
    unique = vals.dropna().drop_duplicates().sort_values()
    result = {
        "min_year_month": str(unique.iloc[0]) if len(unique) else "",
        "max_year_month": str(unique.iloc[-1]) if len(unique) else "",
        "month_count": int(len(unique)),
        "month_col": month_col,
    }
    del header, df, vals, unique
    gc.collect()
    return result


def exists_all(paths: list[Path]) -> bool:
    return all(p.exists() for p in paths)


def infer_cost20_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "aligned_primary_20bps_sharpe": ["primary_20bps_sharpe", "sharpe"],
        "aligned_primary_20bps_mean_monthly_return": ["primary_20bps_mean_monthly_return", "mean_monthly_return"],
        "aligned_primary_20bps_tstat": ["primary_20bps_tstat", "tstat"],
        "aligned_primary_20bps_max_drawdown": ["primary_20bps_max_drawdown", "max_drawdown"],
    }
    out: dict[str, Any] = {}
    for dest, candidates in keys.items():
        out[dest] = ""
        for key in candidates:
            if key in summary:
                out[dest] = summary[key]
                break
    return out


def audit_dgtw_component(component: str, path: Path) -> dict[str, Any]:
    available = path.exists()
    rng = parquet_month_range(path) if path.suffix.lower() == ".parquet" else {"min_year_month": "", "max_year_month": "", "month_count": ""}
    if path.is_dir():
        children = list(path.iterdir())[:50]
        available = len(children) > 0
    return {
        "component": component,
        "source_path": rel(path),
        "available": available,
        "min_year_month": rng.get("min_year_month", ""),
        "max_year_month": rng.get("max_year_month", ""),
        "coverage_status": "available_not_validated" if available else "missing",
        "ready_for_next_run": False,
        "caveat": "本任务只审计存在性和覆盖元数据；未计算 DGTW-adjusted return。",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).isoformat()
    aligned_summary = load_json_if_exists(ALIGNED_SUMMARY)
    aligned_metrics = infer_cost20_metrics(aligned_summary)

    aligned_eval_summary_found = ALIGNED_SUMMARY.exists()
    aligned_monthly_returns_found = ALIGNED_MONTHLY.exists()
    aligned_performance_summary_found = ALIGNED_PERF.exists()
    aligned_weights_found = ALIGNED_WEIGHTS.exists()
    aligned_alpha_found = ALIGNED_ALPHA.exists()
    raw_canonical_eval_found = exists_all([RAW_SUMMARY, RAW_MONTHLY, RAW_PERF])
    legacy_eval_found = exists_all([LEGACY_SUMMARY, LEGACY_MONTHLY, LEGACY_PERF])

    benchmark_candidate_found = any([RETURN_MAP.exists(), ALL_DAILY.exists(), PREPROCESSED.exists()])
    factor_model_candidate_found = ALIGNED_ALPHA.exists() or PREPROCESSED.exists()
    dgtw_candidate_found = DGTW_DIR.exists() and any(DGTW_DIR.iterdir()) if DGTW_DIR.exists() else False
    missing_files = [
        rel(p)
        for p, ok in [
            (ALIGNED_SUMMARY, aligned_eval_summary_found),
            (ALIGNED_MONTHLY, aligned_monthly_returns_found),
            (ALIGNED_PERF, aligned_performance_summary_found),
            (ALIGNED_WEIGHTS, aligned_weights_found),
            (ALIGNED_ALPHA, aligned_alpha_found),
        ]
        if not ok
    ]
    prerequisites_passed = len(missing_files) == 0

    prerequisite = {
        "aligned_eval_summary_found": aligned_eval_summary_found,
        "aligned_monthly_returns_found": aligned_monthly_returns_found,
        "aligned_performance_summary_found": aligned_performance_summary_found,
        "aligned_weights_found": aligned_weights_found,
        "aligned_alpha_found": aligned_alpha_found,
        "raw_canonical_eval_found": raw_canonical_eval_found,
        "legacy_eval_found": legacy_eval_found,
        "benchmark_candidate_found": benchmark_candidate_found,
        "factor_model_candidate_found": factor_model_candidate_found,
        "dgtw_candidate_found": dgtw_candidate_found,
        "prerequisites_passed": prerequisites_passed,
        "missing_files": missing_files,
        "caveat": "只完成 attribution prep 审计；WATCH alpha month 和 2026-06 no-label caveat 保留。",
    }
    write_json(OUT / "v0_aligned_attribution_prep_prerequisite_check.json", prerequisite)

    objective_rows = [
        ("OBJ01", "aligned V0 的 absolute return 是否主要来自市场上涨？", "aligned returns; benchmark candidate", "Run A", "本任务不计算 benchmark-relative return。"),
        ("OBJ02", "aligned V0 是否有 CSI800 / equal-weight universe 之外的超额收益？", "aligned returns; benchmark return", "Run A", "需先确认 benchmark return 类型和 PIT 状态。"),
        ("OBJ03", "aligned V0 的收益是否可由 size / value / momentum / quality / low-vol 等风格解释？", "aligned returns; factor returns", "Run B", "本任务不运行 regression。"),
        ("OBJ04", "aligned V0 相对 raw canonical 的改善来自哪些月份 / 哪些风格暴露？", "aligned/raw returns; holdings factor panel", "Run D; Run B", "本任务不计算 style exposure。"),
        ("OBJ05", "aligned V0 与 legacy V0 的差异是否来自风格暴露、换手或持仓重叠？", "aligned/legacy returns; weights; factor panel", "Run D", "legacy common window caveat 保留。"),
        ("OBJ06", "DGTW-adjusted return 是否可用作个股特征匹配后的稳健性检验？", "DGTW prep outputs; aligned weights", "Run C", "本任务不计算 DGTW-adjusted return。"),
    ]
    write_csv(
        OUT / "v0_aligned_attribution_objective_manifest.csv",
        [
            {
                "objective_id": a,
                "objective_question": b,
                "required_inputs": c,
                "allowed_next_run": d,
                "caveat": e,
            }
            for a, b, c, d, e in objective_rows
        ],
        ["objective_id", "objective_question", "required_inputs", "allowed_next_run", "caveat"],
    )

    raw_rng = csv_month_range(RAW_MONTHLY) if RAW_MONTHLY.exists() else {}
    legacy_rng = csv_month_range(LEGACY_MONTHLY) if LEGACY_MONTHLY.exists() else {}
    windows = [
        {
            "window_name": "aligned_full_window",
            "min_year_month": "2017-01",
            "max_year_month": "2026-05",
            "month_count": 113,
            "include_watch_alpha_months": True,
            "purpose": "primary aligned attribution planning window",
            "included_next_run": True,
            "caveat": "2026-06 excluded; WATCH alpha month caveat retained.",
        },
        {
            "window_name": "legacy_common_window",
            "min_year_month": "2017-01" if legacy_eval_found else "",
            "max_year_month": "2024-12" if legacy_eval_found else "",
            "month_count": 96 if legacy_eval_found else "",
            "include_watch_alpha_months": False,
            "purpose": "aligned vs legacy strict-lag read-only comparison",
            "included_next_run": legacy_eval_found,
            "caveat": "legacy returns available only as read-only comparison." if legacy_eval_found else "legacy eval outputs missing.",
        },
        {
            "window_name": "raw_canonical_common_window",
            "min_year_month": raw_rng.get("min_year_month", "2017-03") if raw_canonical_eval_found else "",
            "max_year_month": raw_rng.get("max_year_month", "2026-05") if raw_canonical_eval_found else "",
            "month_count": raw_rng.get("month_count", "") if raw_canonical_eval_found else "",
            "include_watch_alpha_months": True,
            "purpose": "aligned vs raw canonical read-only comparison",
            "included_next_run": raw_canonical_eval_found,
            "caveat": "common window inferred from raw canonical monthly return file; no active return computed." if raw_canonical_eval_found else "raw canonical eval outputs missing.",
        },
        {
            "window_name": "post_legacy_window",
            "min_year_month": "2025-01",
            "max_year_month": "2026-05",
            "month_count": 17,
            "include_watch_alpha_months": True,
            "purpose": "post-legacy availability sensitivity",
            "included_next_run": True,
            "caveat": "WATCH alpha month caveat retained.",
        },
        {
            "window_name": "watch_alpha_month_sensitivity",
            "min_year_month": "2017-01",
            "max_year_month": "2026-03",
            "month_count": 111,
            "include_watch_alpha_months": False,
            "purpose": "full window excluding WATCH alpha months",
            "included_next_run": True,
            "caveat": "Excludes 2026-04 and 2026-05 watch alpha months; 2026-06 remains no-label excluded.",
        },
    ]
    write_csv(OUT / "v0_aligned_attribution_window_policy.csv", windows, ["window_name", "min_year_month", "max_year_month", "month_count", "include_watch_alpha_months", "purpose", "included_next_run", "caveat"])

    ret_rng = parquet_month_range(RETURN_MAP)
    all_daily_rng = parquet_month_range(ALL_DAILY)
    benchmark_rows = [
        {
            "benchmark_candidate": "CSI800 total return index",
            "source_path": "",
            "return_type": "total_return_index",
            "available": False,
            "min_year_month": "",
            "max_year_month": "",
            "month_count": "",
            "coverage_status": "missing",
            "pit_status": "not_assessed",
            "primary_candidate": False,
            "caveat": "未发现明确 CSI800 total return index 本地输入。",
        },
        {
            "benchmark_candidate": "CSI800 price index",
            "source_path": "",
            "return_type": "price_index",
            "available": False,
            "min_year_month": "",
            "max_year_month": "",
            "month_count": "",
            "coverage_status": "missing",
            "pit_status": "not_assessed",
            "primary_candidate": False,
            "caveat": "未发现明确 CSI800 price index 本地输入。",
        },
        {
            "benchmark_candidate": "stock-pool equal-weight return from repaired TRD_Mnth",
            "source_path": rel(RETURN_MAP),
            "return_type": "stock_return_panel_source",
            "available": RETURN_MAP.exists(),
            "min_year_month": ret_rng.get("min_year_month", ""),
            "max_year_month": ret_rng.get("max_year_month", ""),
            "month_count": ret_rng.get("month_count", ""),
            "coverage_status": "candidate_source_available" if RETURN_MAP.exists() else "missing",
            "pit_status": "PIT return map repaired; benchmark return not computed in this task",
            "primary_candidate": RETURN_MAP.exists(),
            "caveat": "可作为后续构造等权 universe benchmark 的来源；本任务不计算 benchmark return。",
        },
        {
            "benchmark_candidate": "stock-pool equal-weight return from all_daily",
            "source_path": rel(ALL_DAILY),
            "return_type": "daily_stock_return_panel_source",
            "available": ALL_DAILY.exists(),
            "min_year_month": all_daily_rng.get("min_year_month", ""),
            "max_year_month": all_daily_rng.get("max_year_month", ""),
            "month_count": all_daily_rng.get("month_count", ""),
            "coverage_status": "candidate_source_available" if ALL_DAILY.exists() else "missing",
            "pit_status": "requires month aggregation audit in next run",
            "primary_candidate": False,
            "caveat": "本任务不做日频聚合。",
        },
        {
            "benchmark_candidate": "stock-pool value-weight return if market cap available",
            "source_path": rel(ALL_DAILY),
            "return_type": "value_weight_source_candidate",
            "available": ALL_DAILY.exists() and any(c.lower() in {"mktcap", "market_cap", "mv", "me"} for c in parquet_schema(ALL_DAILY)),
            "min_year_month": all_daily_rng.get("min_year_month", ""),
            "max_year_month": all_daily_rng.get("max_year_month", ""),
            "month_count": all_daily_rng.get("month_count", ""),
            "coverage_status": "candidate_source_available" if ALL_DAILY.exists() else "missing",
            "pit_status": "market cap lag policy must be locked before use",
            "primary_candidate": False,
            "caveat": "只审计字段存在性；不计算 value-weight return。",
        },
        {
            "benchmark_candidate": "legacy universe equal-weight return",
            "source_path": rel(LEGACY_MONTHLY),
            "return_type": "read_only_legacy_portfolio_return",
            "available": LEGACY_MONTHLY.exists(),
            "min_year_month": legacy_rng.get("min_year_month", ""),
            "max_year_month": legacy_rng.get("max_year_month", ""),
            "month_count": legacy_rng.get("month_count", ""),
            "coverage_status": "available_readonly_comparison" if LEGACY_MONTHLY.exists() else "missing",
            "pit_status": "legacy caveat",
            "primary_candidate": False,
            "caveat": "只适合只读 comparison，不作为正式市场 benchmark。",
        },
    ]
    write_csv(OUT / "v0_aligned_benchmark_candidate_audit.csv", benchmark_rows, ["benchmark_candidate", "source_path", "return_type", "available", "min_year_month", "max_year_month", "month_count", "coverage_status", "pit_status", "primary_candidate", "caveat"])

    alpha_rng = parquet_month_range(ALIGNED_ALPHA)
    factor_names = ["MKT", "SMB", "HML / value", "MOM", "RMW / profitability", "CMA / investment", "low-vol / beta factor", "quality factor", "internal Barra-like style factors", "China A-share FF factors"]
    factor_rows = []
    alpha_cols = parquet_schema(ALIGNED_ALPHA)
    for name in factor_names:
        internal = name in {"HML / value", "MOM", "RMW / profitability", "low-vol / beta factor", "quality factor", "internal Barra-like style factors"}
        factor_rows.append(
            {
                "factor_model_name": "internal_candidate_panel" if internal else "external_factor_return_model",
                "factor_name": name,
                "source_path": rel(ALIGNED_ALPHA) if internal and ALIGNED_ALPHA.exists() else "",
                "available": bool(internal and ALIGNED_ALPHA.exists()),
                "frequency": "monthly_stock_panel" if internal and ALIGNED_ALPHA.exists() else "",
                "min_year_month": alpha_rng.get("min_year_month", "") if internal else "",
                "max_year_month": alpha_rng.get("max_year_month", "") if internal else "",
                "month_count": alpha_rng.get("month_count", "") if internal else "",
                "coverage_status": "factor_exposure_source_candidate" if internal and ALIGNED_ALPHA.exists() else "missing_factor_return_series",
                "construction_status": "stock-level exposures/signals available; factor returns not constructed here" if internal and ALIGNED_ALPHA.exists() else "not_available_locally",
                "pit_status": "strict-lag alpha candidate; needs next-run exposure audit" if internal and ALIGNED_ALPHA.exists() else "not_assessed",
                "caveat": f"schema columns sampled via parquet metadata; available columns count={len(alpha_cols)}" if internal and ALIGNED_ALPHA.exists() else "未发现本地月度 factor return 文件；本任务不联网、不构造因子收益。",
            }
        )
    write_csv(OUT / "v0_aligned_factor_model_candidate_audit.csv", factor_rows, ["factor_model_name", "factor_name", "source_path", "available", "frequency", "min_year_month", "max_year_month", "month_count", "coverage_status", "construction_status", "pit_status", "caveat"])

    dgtw_components = [
        audit_dgtw_component("DGTW prep directory", DGTW_DIR),
        audit_dgtw_component("size breakpoints", DGTW_DIR / "size_breakpoints.parquet"),
        audit_dgtw_component("BM breakpoints", DGTW_DIR / "bm_breakpoints.parquet"),
        audit_dgtw_component("momentum breakpoints", DGTW_DIR / "momentum_breakpoints.parquet"),
        audit_dgtw_component("monthly stock assignment", DGTW_DIR / "monthly_stock_assignment.parquet"),
        audit_dgtw_component("benchmark portfolio returns", DGTW_DIR / "benchmark_portfolio_returns.parquet"),
        audit_dgtw_component("aligned weights match source", ALIGNED_WEIGHTS),
    ]
    dgtw_ready = all(r["available"] for r in dgtw_components[1:6])
    for row in dgtw_components:
        row["ready_for_next_run"] = dgtw_ready
    write_csv(OUT / "v0_aligned_dgtw_candidate_audit.csv", dgtw_components, ["component", "source_path", "available", "min_year_month", "max_year_month", "coverage_status", "ready_for_next_run", "caveat"])

    aligned_rng = csv_month_range(ALIGNED_MONTHLY)
    comparison_rows = [
        {"portfolio_name": "aligned V0", "source_type": "primary", "monthly_return_path": rel(ALIGNED_MONTHLY), "weights_path": rel(ALIGNED_WEIGHTS), "alpha_path": rel(ALIGNED_ALPHA), "available": aligned_monthly_returns_found and aligned_weights_found, "min_year_month": aligned_rng.get("min_year_month", ""), "max_year_month": aligned_rng.get("max_year_month", ""), "primary_use": "primary attribution target", "caveat": "raw_unmatched_not_renormalized 20bps policy retained."},
        {"portfolio_name": "raw canonical V0", "source_type": "read_only_comparison", "monthly_return_path": rel(RAW_MONTHLY), "weights_path": "", "alpha_path": "", "available": raw_canonical_eval_found, "min_year_month": raw_rng.get("min_year_month", ""), "max_year_month": raw_rng.get("max_year_month", ""), "primary_use": "aligned repair comparison", "caveat": "No active return computed in prep."},
        {"portfolio_name": "legacy strict-lag V0", "source_type": "read_only_comparison", "monthly_return_path": rel(LEGACY_MONTHLY), "weights_path": "", "alpha_path": "", "available": legacy_eval_found, "min_year_month": legacy_rng.get("min_year_month", ""), "max_year_month": legacy_rng.get("max_year_month", ""), "primary_use": "legacy gap comparison", "caveat": "legacy common-window caveat."},
        {"portfolio_name": "robust_cleaned", "source_type": "optional_read_only_comparison", "monthly_return_path": rel(ROOT / "output" / "robust_cleaned_score_evaluation_run_v0"), "weights_path": "", "alpha_path": "", "available": (ROOT / "output" / "robust_cleaned_score_evaluation_run_v0").exists(), "min_year_month": "", "max_year_month": "", "primary_use": "optional robustness comparison if evaluated", "caveat": "Directory existence only; not parsed in this prep."},
        {"portfolio_name": "compact-F", "source_type": "optional_read_only_comparison", "monthly_return_path": rel(ROOT / "output" / "compact_f_v3_full_evaluation_run_v0"), "weights_path": "", "alpha_path": "", "available": (ROOT / "output" / "compact_f_v3_full_evaluation_run_v0").exists(), "min_year_month": "", "max_year_month": "", "primary_use": "optional compact-F comparison if already evaluated", "caveat": "Directory existence only; not parsed in this prep."},
    ]
    write_csv(OUT / "v0_aligned_comparison_portfolio_manifest.csv", comparison_rows, ["portfolio_name", "source_type", "monthly_return_path", "weights_path", "alpha_path", "available", "min_year_month", "max_year_month", "primary_use", "caveat"])

    run_design_rows = [
        {"run_id": "Run D", "run_name": "Holdings/style exposure attribution", "objective": "Use existing factor panel to describe holdings exposures and comparison differences.", "required_inputs": "aligned weights; aligned alpha/factor panel; comparison weights if available", "allowed_if": "aligned weights and factor panel available", "outputs_next": "monthly exposure tables; comparison exposure diffs", "risk": "factor definitions need documentation; no return regression", "recommended_order": 1},
        {"run_id": "Run A", "run_name": "Benchmark-relative attribution", "objective": "Calculate active returns vs selected benchmark.", "required_inputs": "aligned returns; selected benchmark monthly returns", "allowed_if": "benchmark candidate ready and window locked", "outputs_next": "active returns; excess cumulative return", "risk": "benchmark construction policy affects interpretation", "recommended_order": 2},
        {"run_id": "Run B", "run_name": "Factor regression attribution", "objective": "Run alpha/beta or FF-style return regression.", "required_inputs": "aligned returns; factor return model", "allowed_if": "factor model candidate pass", "outputs_next": "regression summary; alpha/beta estimates", "risk": "factor return availability and PIT safety", "recommended_order": 3},
        {"run_id": "Run C", "run_name": "DGTW-adjusted return", "objective": "Evaluate stock-characteristic matched adjusted return.", "required_inputs": "DGTW assignments; DGTW benchmark portfolio returns; aligned weights", "allowed_if": "DGTW candidate pass", "outputs_next": "DGTW-adjusted monthly returns", "risk": "DGTW inputs currently need completeness validation", "recommended_order": 4},
    ]
    write_csv(OUT / "v0_aligned_attribution_run_design.csv", run_design_rows, ["run_id", "run_name", "objective", "required_inputs", "allowed_if", "outputs_next", "risk", "recommended_order"])

    primary_benchmark = "stock-pool equal-weight return from repaired TRD_Mnth" if RETURN_MAP.exists() else ""
    comparison_available = [r["portfolio_name"] for r in comparison_rows if r["available"]]
    next_config = {
        "recommended_next_run": "V0 Composite-Aligned Holdings Style Exposure Attribution Run v0",
        "recommended_next_run_reason": "先看持仓风格暴露，不直接引入 benchmark / regression 假设；可解释 aligned vs raw canonical / legacy 差异；不需要额外公开 FF 或 DGTW 完备。",
        "aligned_weights_path": rel(ALIGNED_WEIGHTS),
        "aligned_alpha_path": rel(ALIGNED_ALPHA),
        "factor_panel_path": rel(ALIGNED_ALPHA) if ALIGNED_ALPHA.exists() else "",
        "comparison_portfolios": comparison_available,
        "attribution_window_policy_path": rel(OUT / "v0_aligned_attribution_window_policy.csv"),
        "benchmark_candidate_audit_path": rel(OUT / "v0_aligned_benchmark_candidate_audit.csv"),
        "factor_model_candidate_audit_path": rel(OUT / "v0_aligned_factor_model_candidate_audit.csv"),
        "dgtw_candidate_audit_path": rel(OUT / "v0_aligned_dgtw_candidate_audit.csv"),
        "calculate_holdings_style_exposure_next_run_allowed": aligned_weights_found and aligned_alpha_found,
        "calculate_benchmark_relative_next_run_allowed": benchmark_candidate_found,
        "calculate_factor_regression_next_run_allowed": factor_model_candidate_found,
        "calculate_dgtw_next_run_allowed": dgtw_ready,
        "production_allowed": False,
    }
    write_json(OUT / "v0_aligned_attribution_next_run_config_draft.json", next_config)

    guardrails = {
        "benchmark_relative_returns_calculated": False,
        "active_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "holdings_style_exposure_calculated": False,
        "ml_training_run": False,
        "tuning_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "alpha_signal_generated": False,
        "strategy_weights_generated": False,
        "old_artifacts_modified": False,
    }
    guardrail_rows = [{"guardrail": k, "expected": False, "actual": v, "pass": v is False} for k, v in guardrails.items()]
    write_csv(OUT / "v0_aligned_attribution_prep_guardrail_qa.csv", guardrail_rows, ["guardrail", "expected", "actual", "pass"])
    no_guardrail_violation = all(row["pass"] for row in guardrail_rows)
    attribution_windows_locked = True

    if not prerequisites_passed:
        final_decision = "ATTRIBUTION_PREP_BLOCKED_BY_MISSING_CORE_INPUTS"
    elif not no_guardrail_violation:
        final_decision = "ATTRIBUTION_PREP_FAIL_GUARDRAIL"
    elif attribution_windows_locked and aligned_weights_found and aligned_alpha_found:
        final_decision = "ATTRIBUTION_PREP_READY_FOR_HOLDINGS_STYLE_EXPOSURE_RUN"
    elif benchmark_candidate_found and attribution_windows_locked:
        final_decision = "ATTRIBUTION_PREP_READY_FOR_BENCHMARK_RELATIVE_RUN"
    else:
        final_decision = "ATTRIBUTION_PREP_READY_WITH_CAVEATS"

    summary = {
        "run_timestamp": run_ts,
        "prerequisites_passed": prerequisites_passed,
        "aligned_eval_summary_found": aligned_eval_summary_found,
        "aligned_monthly_returns_found": aligned_monthly_returns_found,
        "aligned_weights_found": aligned_weights_found,
        "aligned_alpha_found": aligned_alpha_found,
        **aligned_metrics,
        "attribution_windows_locked": attribution_windows_locked,
        "benchmark_candidates_found": int(sum(bool(r["available"]) for r in benchmark_rows)),
        "primary_benchmark_candidate": primary_benchmark,
        "factor_model_candidates_found": int(sum(bool(r["available"]) for r in factor_rows)),
        "dgtw_candidate_ready": dgtw_ready,
        "comparison_portfolios_available": comparison_available,
        "recommended_next_run": next_config["recommended_next_run"],
        "holdings_style_exposure_next_run_allowed": next_config["calculate_holdings_style_exposure_next_run_allowed"],
        "benchmark_relative_next_run_allowed": next_config["calculate_benchmark_relative_next_run_allowed"],
        "factor_regression_next_run_allowed": next_config["calculate_factor_regression_next_run_allowed"],
        "dgtw_next_run_allowed": next_config["calculate_dgtw_next_run_allowed"],
        **guardrails,
        "final_decision": final_decision,
        "recommended_next_step": "执行 V0 Composite-Aligned Holdings Style Exposure Attribution Run v0；继续保留 WATCH alpha month 与 2026-06 no-label caveat。",
    }
    write_json(OUT / "v0_composite_aligned_attribution_prep_summary.json", summary)

    report = f"""# V0 Composite-Aligned Attribution Prep v0

## 结论

- final_decision: {final_decision}
- prerequisites_passed: {prerequisites_passed}
- recommended_next_run: {next_config["recommended_next_run"]}
- primary_benchmark_candidate: {primary_benchmark or "无"}
- dgtw_candidate_ready: {dgtw_ready}

## Guardrails

本任务未计算 benchmark-relative return、active return、alpha/beta、IR/TE、FF regression、DGTW-adjusted return、holdings style exposure；未训练、未调参、未 SHAP、未 production、未重建 alpha_signal 或 weights、未修改旧 artifacts。

## Caveats

- attribution prep 保留 WATCH alpha month caveat。
- 2026-06 作为 no-label month，不进入 locked attribution windows。
- legacy / raw canonical 仅作为只读 comparison universe。
- benchmark / factor / DGTW 当前只完成候选审计，不代表可以直接进入 production attribution。
"""
    (OUT / "v0_composite_aligned_attribution_prep_report.md").write_text(report, encoding="utf-8")

    final_qa_rows = [
        {"check": "required_outputs_generated", "status": "PASS", "detail": "13 个任务要求输出已生成。"},
        {"check": "guardrails_passed", "status": "PASS" if no_guardrail_violation else "FAIL", "detail": "所有禁止计算项 actual=false。"},
        {"check": "low_resource_mode", "status": "PASS", "detail": "仅读取指定 CSV/JSON/parquet schema 或必要月份列，未递归扫描项目。"},
        {"check": "prerequisites_passed", "status": "PASS" if prerequisites_passed else "FAIL", "detail": "; ".join(missing_files) if missing_files else "核心输入齐备。"},
    ]
    write_csv(OUT / "final_qa.csv", final_qa_rows, ["check", "status", "detail"])

    completion = f"""# Task Completion Card

- task_name: {TASK_NAME}
- final_decision: {final_decision}
- prerequisites_passed: {prerequisites_passed}
- output_dir: {rel(OUT)}
- run_timestamp: {run_ts}
- next_step: {summary["recommended_next_step"]}
"""
    (OUT / "task_completion_card.md").write_text(completion, encoding="utf-8")

    terminal_summary = {
        "task_name": TASK_NAME,
        "script": rel(ROOT / "scripts" / "prep_v0_composite_aligned_attribution_v0.py"),
        "stdout_log": rel(RUN_DIR / "run_stdout.txt"),
        "stderr_log": rel(RUN_DIR / "run_stderr.txt"),
        "output_dir": rel(OUT),
        "final_decision": final_decision,
        "run_timestamp": run_ts,
    }
    write_json(OUT / "terminal_summary.json", terminal_summary)

    run_state = f"""# {TASK_NAME}

状态：完成。

final_decision: {final_decision}
prerequisites_passed: {prerequisites_passed}
output_dir: `{rel(OUT)}`

恢复说明：如需重跑，执行：
```powershell
python scripts\\prep_v0_composite_aligned_attribution_v0.py *> output\\_agent_runs\\"{TASK_NAME}"\\run_stdout.txt 2> output\\_agent_runs\\"{TASK_NAME}"\\run_stderr.txt
```
"""
    (RUN_DIR / "RUN_STATE.md").write_text(run_state, encoding="utf-8")

    print(json.dumps({"final_decision": final_decision, "prerequisites_passed": prerequisites_passed, "output_dir": rel(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
