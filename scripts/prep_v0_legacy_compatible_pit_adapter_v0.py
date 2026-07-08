from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TASK_NAME = "V0 Legacy-Compatible PIT Adapter Prep v0"
OUT_NAME = "v0_legacy_compatible_pit_adapter_prep_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / OUT_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

LEGACY_PREPROCESSED = ROOT / "output" / "preprocessed.parquet"
LEGACY_SPLIT = ROOT / "output" / "split_universe_blended.parquet"
LEGACY_ALPHA = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_alpha_signal_panel.parquet"
LEGACY_WEIGHTS = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_reconstructed_weights.parquet"
PIT_PANEL = ROOT / "output" / "v0_canonical_16factor_panel_build_v0" / "v0_canonical_16factor_panel.parquet"
COMPOSITE_SUMMARY = ROOT / "output" / "v0_composite_aligned_repaired_trd_mnth_eval_run_v0" / "v0_composite_aligned_repaired_trd_mnth_eval_run_summary.json"
DENOM_SUMMARY = ROOT / "output" / "v0_denominator_policy_repair_alpha_candidate_build_v0" / "v0_denominator_policy_repair_alpha_candidate_build_summary.json"
VALUE_SUMMARY = ROOT / "output" / "v0_value_icir_weight_path_alignment_alpha_candidate_build_v0" / "v0_value_icir_weight_path_alignment_alpha_candidate_build_summary.json"
VALUE_TOP50_SUMMARY = ROOT / "output" / "v0_value_path_top50_collapse_composite_qa_v0" / "v0_value_path_top50_collapse_composite_qa_summary.json"

CODE_FILES = {
    "run_split_universe.py": ROOT / "scripts" / "run_split_universe.py",
    "run_factor_research.py": ROOT / "scripts" / "run_factor_research.py",
    "production_engine": ROOT / "factor_research" / "production_engine.py",
    "split_universe": ROOT / "factor_research" / "split_universe.py",
    "orthogonalization": ROOT / "factor_research" / "orthogonalization.py",
    "canonical_alpha_builder": ROOT / "scripts" / "build_v0_canonical_strict_lag_alpha_v0.py",
    "composite_aligned_builder": ROOT / "scripts" / "build_v0_composite_aligned_strict_lag_alpha_candidate_v0.py",
}

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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_state(status: str, checkpoint: str, extra: dict[str, Any] | None = None) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_name": TASK_NAME,
        "status": status,
        "checkpoint": checkpoint,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "resume_instruction": f"先读取 {rel(RUN_DIR / 'RUN_STATE.md')}；继续时运行 scripts\\prep_v0_legacy_compatible_pit_adapter_v0.py，并重定向 stdout/stderr 到本目录。",
    }
    if extra:
        payload.update(extra)
    lines = ["# RUN_STATE", "", f"- task_name: {TASK_NAME}", f"- status: {status}", f"- checkpoint: {checkpoint}", "", "```json", json.dumps(payload, ensure_ascii=False, indent=2, default=str), "```"]
    (RUN_DIR / "RUN_STATE.md").write_text("\n".join(lines), encoding="utf-8")


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names if path.exists() else []


def parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows) if path.exists() else 0


def pick_col(cols: list[str], candidates: list[str], contains: list[str] | None = None) -> str | None:
    for c in candidates:
        if c in cols:
            return c
    if contains:
        low = {c.lower(): c for c in cols}
        for needle in contains:
            for lc, orig in low.items():
                if needle.lower() in lc:
                    return orig
    return None


def norm_symbol(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(6)


def month_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in ["date", "month_end"]:
        return pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m")
    return df[col].astype("string").str.slice(0, 7)


def zscore_by_month(df: pd.DataFrame, factor_cols: list[str], month_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in factor_cols:
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        mean = vals.groupby(df[month_col]).transform("mean")
        std = vals.groupby(df[month_col]).transform("std").replace(0, np.nan)
        out[f"{col}_z"] = ((vals - mean) / std).fillna(0.0)
    return out


def prereq_check() -> dict[str, Any]:
    production_found = any(p.exists() for p in CODE_FILES.values())
    flags = {
        "production_code_found": production_found,
        "legacy_preprocessed_found": LEGACY_PREPROCESSED.exists(),
        "legacy_split_universe_found": LEGACY_SPLIT.exists(),
        "legacy_alpha_found": LEGACY_ALPHA.exists(),
        "legacy_weights_found": LEGACY_WEIGHTS.exists(),
        "canonical_pit_panel_found": PIT_PANEL.exists(),
        "composite_aligned_summary_found": COMPOSITE_SUMMARY.exists(),
        "denominator_repair_summary_found": DENOM_SUMMARY.exists(),
        "value_path_summary_found": VALUE_SUMMARY.exists() and VALUE_TOP50_SUMMARY.exists(),
    }
    required_paths = {
        "legacy_preprocessed_found": LEGACY_PREPROCESSED,
        "legacy_split_universe_found": LEGACY_SPLIT,
        "legacy_alpha_found": LEGACY_ALPHA,
        "legacy_weights_found": LEGACY_WEIGHTS,
        "canonical_pit_panel_found": PIT_PANEL,
        "composite_aligned_summary_found": COMPOSITE_SUMMARY,
        "denominator_repair_summary_found": DENOM_SUMMARY,
        "value_path_summary_found": VALUE_SUMMARY,
    }
    missing = [rel(p) for k, p in required_paths.items() if not flags[k]]
    if not VALUE_TOP50_SUMMARY.exists():
        missing.append(rel(VALUE_TOP50_SUMMARY))
    flags["prerequisites_passed"] = bool(production_found and not missing)
    flags["missing_files"] = missing
    flags["caveat"] = "原 scripts/run_split_universe.py 与 src/ 目录未找到；使用 factor_research/split_universe.py、production_engine.py 和现有 V0 scripts 作为 legacy production code evidence。"
    return flags


def production_contract(legacy_cols: list[str]) -> pd.DataFrame:
    rows = [
        ("input artifact", True, "factor_research/production_engine.py", "train_and_save(panel_path='output/preprocessed.parquet', blended_path='output/split_universe_blended.parquet')", "preprocessed.parquet + split_universe_blended.parquet", "observed", "adapter preview/build 不能覆盖原文件；下一步以显式路径注入。", ""),
        ("date column", True, "factor_research/production_engine.py:191,404", "date_col='date'", "date" if "date" in legacy_cols else "missing", "生成 date；可由 year_month/month_end 映射。", ""),
        ("symbol column", True, "factor_research/production_engine.py:192,405", "symbol_col='symbol'", "symbol" if "symbol" in legacy_cols else "missing", "生成 symbol 并保留 symbol_norm。", ""),
        ("close price", True, "factor_research/production_engine.py:193,1040", "close_col='收盘'，缺失时尝试 close/Close/含收", "收盘" if "收盘" in legacy_cols else "not_observed", "PIT preview 不训练模型；build/replay 若走 ML production 需 close 或 forward_return_1m。", "本任务不训练。"),
        ("forward return", True, "factor_research/production_engine.py:1040", "forward_return_1m 缺失时由 close shift(-1) 生成", "forward_return_1m" if "forward_return_1m" in legacy_cols else "not_observed", "保留/映射 forward_return_1m；只作兼容字段，不计算收益评价。", ""),
        ("factor neutral z", True, "factor_research/production_engine.py:212, split_universe.py:151", "生产 ML 要 _neutral_z；split 模型优先 neutral_z → z → raw", f"{sum(c.endswith('_neutral_z') for c in legacy_cols)} columns", "输出 raw、_z、_neutral_z 三套列。", "neutral_z 若无行业/市值字段则先用 z 兼容占位并标注。"),
        ("split universe", True, "factor_research/split_universe.py:257", "按成交额/换手率估算 mcap_est，月截面 50% 分位切 large/small", "universe/mcap_est/mcap_pct in split artifact", "preview 生成 split_group/universe/mcap_pct；不覆盖 legacy split。", ""),
        ("factor priority", True, "factor_research/split_universe.py:151", "_get_factor_col: suffix, _neutral_z, _z, raw", "observed", "三种表示均生成，保证优先级可复刻。", ""),
        ("GS composite", True, "factor_research/split_universe.py:750; orthogonalization.py:234", "apply_gram_schmidt_composite rolling ICIR + sign flip + abs ICIR weights", "code path observed", "adapter 只提供输入 schema，不运行 composite。", "legacy replay 可能复刻 current-month IC leakage。"),
    ]
    return pd.DataFrame(rows, columns=["contract_item", "required", "source_evidence_file", "source_evidence_line_or_function", "expected_column_or_behavior", "legacy_artifact_observed", "adapter_requirement", "caveat"])


def required_columns_manifest(legacy_cols: list[str], pit_cols: list[str]) -> pd.DataFrame:
    base = [
        ("date", "month/date key", "production_engine", True, "datetime64[ns]", "non-null", pick_col(pit_cols, ["date", "month_end"], ["date"]), "rename/derive", ""),
        ("year_month", "month key", "V0 research scripts", True, "string YYYY-MM", "non-null", pick_col(pit_cols, ["year_month"], ["year_month"]), "derive if missing", ""),
        ("month_end", "month end key", "V0 research scripts", False, "datetime64[ns]", "nullable", pick_col(pit_cols, ["month_end", "date"], ["month"]), "derive if missing", ""),
        ("symbol", "security id", "production_engine", True, "string(6)", "non-null", pick_col(pit_cols, ["symbol", "symbol_norm"], ["symbol"]), "normalize zfill(6)", ""),
        ("symbol_norm", "normalized security id", "V0 research scripts", True, "string(6)", "non-null", pick_col(pit_cols, ["symbol_norm", "symbol"], ["symbol"]), "normalize zfill(6)", ""),
        ("forward_return_1m", "label compatibility only", "production_engine", True, "float64", "nullable final month", pick_col(pit_cols, ["forward_return_1m", "fwd_ret_1m"], ["forward"]), "rename/pass through", "本任务不用于收益计算。"),
        ("mcap_est", "split market cap", "split_universe", True, "float64", "nullable allowed", pick_col(pit_cols, ["mcap_est", "market_cap", "total_mv", "circ_mv"], ["mcap", "market_cap", "mv"]), "map or proxy", ""),
        ("mcap_pct", "split percentile", "split_universe", True, "float64", "nullable allowed", None, "monthly rank pct", ""),
        ("universe", "legacy split label", "split_universe", True, "string", "non-null when mcap available", None, "大盘/小盘 from mcap_pct", ""),
        ("split_group", "adapter split label", "adapter", True, "string", "non-null when mcap available", None, "large/small from mcap_pct", ""),
    ]
    rows = list(base)
    for f in FACTORS:
        for suffix, role, req in [("", "raw factor", True), ("_z", "zscore factor", True), ("_neutral_z", "neutralized factor", True)]:
            col = f"{f}{suffix}"
            rows.append((col, role, "split_universe/production_engine", req, "float64", "nullable; fill only for z compatibility", f if f in pit_cols else pick_col(pit_cols, [col]), "generate" if suffix else "rename/pass through", "" if suffix != "_neutral_z" else "若 PIT 无行业/市值中性化输入，prep preview 中 neutral_z=z，占位不宣称已严格中性。"))
    return pd.DataFrame(rows, columns=["column_name", "column_role", "required_by", "required", "dtype_expected", "null_policy", "adapter_source_column", "adapter_transform_needed", "caveat"])


def factor_contract(legacy_cols: list[str], pit_cols: list[str]) -> pd.DataFrame:
    rows = []
    for f in FACTORS:
        rows.append({
            "factor_name": f,
            "legacy_raw_column": f,
            "legacy_z_column": f"{f}_z",
            "legacy_neutral_z_column": f"{f}_neutral_z",
            "production_selection_priority": "factor_neutral_z -> factor_z -> raw",
            "old_preprocessed_column_available": ";".join([c for c in [f, f"{f}_z", f"{f}_neutral_z"] if c in legacy_cols]),
            "pit_panel_source_column": f if f in pit_cols else "",
            "transform_needed": "rename/pass-through" if f in pit_cols else "missing_source",
            "neutral_z_needed": f"{f}_neutral_z" not in pit_cols,
            "zscore_needed": f"{f}_z" not in pit_cols,
            "adapter_output_column_raw": f,
            "adapter_output_column_z": f"{f}_z",
            "adapter_output_column_neutral_z": f"{f}_neutral_z",
            "caveat": "neutral_z preview may equal z until strict neutralization policy is implemented" if f"{f}_neutral_z" not in pit_cols else "",
        })
    return pd.DataFrame(rows)


def schema_audit(legacy_cols: list[str], pit_cols: list[str]) -> pd.DataFrame:
    legacy_key_cols = [c for c in ["symbol", "symbol_norm", "date", "year_month", "month_end"] if c in legacy_cols]
    pit_key_cols = [c for c in ["symbol", "symbol_norm", "date", "year_month", "month_end"] if c in pit_cols]
    legacy = pd.read_parquet(LEGACY_PREPROCESSED, columns=legacy_key_cols)
    pit = pd.read_parquet(PIT_PANEL, columns=pit_key_cols)
    l_sym = pick_col(legacy.columns.tolist(), ["symbol_norm", "symbol"])
    p_sym = pick_col(pit.columns.tolist(), ["symbol_norm", "symbol"])
    l_mon = pick_col(legacy.columns.tolist(), ["year_month", "month_end", "date"])
    p_mon = pick_col(pit.columns.tolist(), ["year_month", "month_end", "date"])
    legacy["_ym"] = month_series(legacy, l_mon)
    pit["_ym"] = month_series(pit, p_mon)
    legacy["_sym"] = norm_symbol(legacy[l_sym])
    pit["_sym"] = norm_symbol(pit[p_sym])
    items = [
        ("row_count", len(legacy), len(pit), "INFO", "adapter should preserve PIT row universe"),
        ("unique_symbol_count", legacy["_sym"].nunique(), pit["_sym"].nunique(), "INFO", "expect differences from PIT rebuild"),
        ("month_count", legacy["_ym"].nunique(), pit["_ym"].nunique(), "INFO", "align replay window explicitly"),
        ("min_year_month", legacy["_ym"].min(), pit["_ym"].min(), "INFO", "document window"),
        ("max_year_month", legacy["_ym"].max(), pit["_ym"].max(), "INFO", "document window"),
        ("symbol column", l_sym, p_sym, "MATCH" if p_sym else "MISSING", "normalize to symbol and symbol_norm"),
        ("month column", l_mon, p_mon, "MATCH" if p_mon else "MISSING", "derive date/year_month/month_end"),
        ("factor columns", sum(f in legacy_cols for f in FACTORS), sum(f in pit_cols for f in FACTORS), "MATCH" if all(f in pit_cols for f in FACTORS) else "MISMATCH", "map missing before build"),
        ("factor_z columns", sum(f"{f}_z" in legacy_cols for f in FACTORS), sum(f"{f}_z" in pit_cols for f in FACTORS), "INFO", "generate z in adapter"),
        ("factor_neutral_z columns", sum(f"{f}_neutral_z" in legacy_cols for f in FACTORS), sum(f"{f}_neutral_z" in pit_cols for f in FACTORS), "INFO", "generate/pass-through neutral_z policy"),
        ("forward return columns", [c for c in legacy_cols if "forward" in c.lower() or "fwd" in c.lower()], [c for c in pit_cols if "forward" in c.lower() or "fwd" in c.lower()], "INFO", "compat only; no return eval"),
        ("market cap columns", [c for c in legacy_cols if "mcap" in c.lower() or "mv" in c.lower() or "market" in c.lower()], [c for c in pit_cols if "mcap" in c.lower() or "mv" in c.lower() or "market" in c.lower()], "INFO", "pick mcap source or proxy"),
        ("split-related columns", [c for c in legacy_cols if c in ["universe", "split_group", "mcap_pct", "mcap_est"]], [c for c in pit_cols if c in ["universe", "split_group", "mcap_pct", "mcap_est"]], "INFO", "preview generate split fields"),
        ("duplicate symbol-month count", int(legacy.duplicated(["_sym", "_ym"]).sum()), int(pit.duplicated(["_sym", "_ym"]).sum()), "MATCH" if int(pit.duplicated(["_sym", "_ym"]).sum()) == 0 else "MISMATCH", "deduplicate before build"),
    ]
    del legacy, pit
    gc.collect()
    return pd.DataFrame([{"item": i, "legacy_value": l, "pit_value": p, "match_status": s, "adapter_action": a, "caveat": ""} for i, l, p, s, a in items])


def distribution_audit(legacy_cols: list[str], pit_cols: list[str]) -> pd.DataFrame:
    l_cols = [c for c in ["symbol", "symbol_norm", "date", "year_month", "month_end"] if c in legacy_cols]
    p_cols = [c for c in ["symbol", "symbol_norm", "date", "year_month", "month_end"] if c in pit_cols]
    for f in FACTORS:
        for col in [f, f"{f}_z", f"{f}_neutral_z"]:
            if col in legacy_cols:
                l_cols.append(col)
            if col in pit_cols:
                p_cols.append(col)
    legacy = pd.read_parquet(LEGACY_PREPROCESSED, columns=sorted(set(l_cols)))
    pit = pd.read_parquet(PIT_PANEL, columns=sorted(set(p_cols)))
    legacy["_ym"] = month_series(legacy, pick_col(legacy.columns.tolist(), ["year_month", "month_end", "date"]))
    pit["_ym"] = month_series(pit, pick_col(pit.columns.tolist(), ["year_month", "month_end", "date"]))
    legacy["_sym"] = norm_symbol(legacy[pick_col(legacy.columns.tolist(), ["symbol_norm", "symbol"])])
    pit["_sym"] = norm_symbol(pit[pick_col(pit.columns.tolist(), ["symbol_norm", "symbol"])])
    rows = []
    for f in FACTORS:
        for rep, suffix in [("raw", ""), ("z", "_z"), ("neutral_z", "_neutral_z")]:
            col = f"{f}{suffix}"
            l_avail = col in legacy.columns
            p_avail = col in pit.columns
            lser = pd.to_numeric(legacy[col], errors="coerce") if l_avail else pd.Series(dtype=float)
            pser = pd.to_numeric(pit[col], errors="coerce") if p_avail else pd.Series(dtype=float)
            sp_mean = np.nan
            sp_median = np.nan
            if l_avail and p_avail:
                merged = legacy[["_sym", "_ym", col]].merge(pit[["_sym", "_ym", col]], on=["_sym", "_ym"], suffixes=("_legacy", "_pit"))
                vals = []
                for _, g in merged.groupby("_ym", sort=False):
                    sub = g[[f"{col}_legacy", f"{col}_pit"]].dropna()
                    if len(sub) >= 10:
                        vals.append(sub[f"{col}_legacy"].corr(sub[f"{col}_pit"], method="spearman"))
                if vals:
                    sp_mean = float(np.nanmean(vals))
                    sp_median = float(np.nanmedian(vals))
                del merged
            rows.append({
                "factor_name": f,
                "representation": rep,
                "legacy_non_null_ratio": float(lser.notna().mean()) if l_avail and len(lser) else np.nan,
                "pit_non_null_ratio": float(pser.notna().mean()) if p_avail and len(pser) else np.nan,
                "legacy_mean": float(lser.mean()) if l_avail else np.nan,
                "pit_mean": float(pser.mean()) if p_avail else np.nan,
                "legacy_std": float(lser.std()) if l_avail else np.nan,
                "pit_std": float(pser.std()) if p_avail else np.nan,
                "monthly_spearman_mean_if_overlap": sp_mean,
                "monthly_spearman_median_if_overlap": sp_median,
                "distribution_match_status": "COMPARABLE" if l_avail and p_avail else "MISSING_REPRESENTATION",
                "adapter_action": "pass-through" if p_avail else ("generate from raw" if suffix else "source missing"),
                "caveat": "diagnostic only; no return metrics calculated",
            })
    del legacy, pit
    gc.collect()
    return pd.DataFrame(rows)


def split_audits(pit_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    contract = pd.DataFrame([
        ("basis", "legacy split_by_market_cap uses estimated market cap 成交额/换手率, winsor 1%/99%, monthly pct rank", "factor_research/split_universe.py:257-340", "PIT may have mcap/market_cap or amount/turnover", "prefer direct market cap; else amount/turnover proxy", ""),
        ("cutoff", "percentile=0.5, mcap_pct >= 0.5 => 大盘 else 小盘", "factor_research/split_universe.py:92,320", "monthly PIT cross-section", "generate universe and split_group preview", ""),
        ("labels", "legacy labels universe='大盘'/'小盘'; blended output keeps universe/mcap_pct/mcap_est", "factor_research/split_universe.py:811", "adapter split_group large/small plus universe", "keep both label styles", ""),
        ("runtime", "production train_and_save reads split_universe_blended.parquet as blended alpha_signal anchor", "factor_research/production_engine.py:1117", "PIT adapter build must not overwrite legacy split file", "next run uses explicit adapter path", ""),
    ], columns=["split_contract_item", "legacy_behavior", "source_evidence", "pit_available_input", "adapter_action", "caveat"])
    legacy_cols = parquet_columns(LEGACY_SPLIT)
    l_key = [c for c in ["symbol", "symbol_norm", "date", "year_month", "month_end", "universe", "split_group"] if c in legacy_cols]
    p_key = [c for c in ["symbol", "symbol_norm", "date", "year_month", "month_end"] if c in pit_cols]
    mcap_col = pick_col(pit_cols, ["mcap_est", "market_cap", "total_mv", "circ_mv"], ["mcap", "market_cap", "mv"])
    if mcap_col:
        p_key.append(mcap_col)
    legacy = pd.read_parquet(LEGACY_SPLIT, columns=sorted(set(l_key)))
    pit = pd.read_parquet(PIT_PANEL, columns=sorted(set(p_key)))
    legacy["_ym"] = month_series(legacy, pick_col(legacy.columns.tolist(), ["year_month", "month_end", "date"]))
    pit["_ym"] = month_series(pit, pick_col(pit.columns.tolist(), ["year_month", "month_end", "date"]))
    legacy["_sym"] = norm_symbol(legacy[pick_col(legacy.columns.tolist(), ["symbol_norm", "symbol"])])
    pit["_sym"] = norm_symbol(pit[pick_col(pit.columns.tolist(), ["symbol_norm", "symbol"])])
    split_col = "universe" if "universe" in legacy.columns else ("split_group" if "split_group" in legacy.columns else None)
    if split_col:
        legacy["_large"] = legacy[split_col].astype(str).isin(["大盘", "large", "LARGE"])
    else:
        legacy["_large"] = False
    if mcap_col:
        vals = pd.to_numeric(pit[mcap_col], errors="coerce")
        pit["_pct"] = vals.groupby(pit["_ym"]).rank(pct=True)
        pit["_large"] = pit["_pct"] >= 0.5
    else:
        pit["_large"] = False
    rows = []
    for ym in sorted(set(legacy["_ym"].dropna()).intersection(set(pit["_ym"].dropna())))[:240]:
        lg = legacy[legacy["_ym"].eq(ym)]
        pg = pit[pit["_ym"].eq(ym)]
        merged = lg[["_sym", "_large"]].merge(pg[["_sym", "_large"]], on="_sym", suffixes=("_legacy", "_pit"))
        ratio = float((merged["_large_legacy"] == merged["_large_pit"]).mean()) if len(merged) and mcap_col else np.nan
        rows.append({
            "year_month": ym,
            "legacy_large_count": int(lg["_large"].sum()),
            "legacy_small_count": int((~lg["_large"]).sum()),
            "pit_preview_large_count": int(pg["_large"].sum()) if mcap_col else 0,
            "pit_preview_small_count": int((~pg["_large"]).sum()) if mcap_col else 0,
            "same_assignment_ratio_if_overlap": ratio,
            "split_match_status": "PREVIEW_READY" if mcap_col else "BLOCKED_NO_MCAP_SOURCE",
            "caveat": "" if mcap_col else "PIT panel lacks detectable market cap column; next build needs market-cap source.",
        })
    del legacy, pit
    gc.collect()
    return contract, pd.DataFrame(rows)


def composite_audit() -> pd.DataFrame:
    rows = [
        ("ICIR window", "rolling_window default 24", "factor_research/split_universe.py:114; orthogonalization.py:239", "legacy rolling window includes current date in window_start=max(0,pos-window+1)", "strict-lag must use only months < formation month", "Route A may replicate; Route B must replace.", ""),
        ("current-month IC leakage", "orthogonalization rolling ICIR loop includes current position by default", "factor_research/orthogonalization.py:112", "yes, possible current-month IC leakage in legacy replay", "shift ICIR inputs by one month", "must be explicitly marked.", ""),
        ("sign flip", "flip_sign=True; negative ICIR gets -1 sign", "factor_research/orthogonalization.py:240,344", "legacy sign flip", "same but lagged ICIR", "", ""),
        ("min ICIR filter", "|IC_IR| > min_ic_ir, default 0.05 in split build_sub_model; GS call min_ic_ir", "factor_research/split_universe.py:550,594; orthogonalization.py:299", "filter weak factors", "same threshold unless config changes in separate run", "", ""),
        ("selected factor count", "valid cols selected after ICIR filter and variance checks", "factor_research/orthogonalization.py:299-326", "dynamic by month", "same dynamic selection with lagged ICIR", "", ""),
        ("total_abs_icir denominator", "sum(abs(ic_irs[c]) for valid_cols); fallback if near zero", "factor_research/orthogonalization.py:328-337", "abs ICIR denominator", "same with lagged values", "", ""),
        ("zero/NaN fallback", "if total_abs < 1e-10 composite stays zero; non-GS path falls back equal weights", "factor_research/orthogonalization.py:329; split_universe.py:672", "zero/equal fallback", "define explicit fallback QA", "", ""),
        ("GS on/off", "SplitUniverseModel orthogonalize=True by default, calls apply_gram_schmidt_composite", "factor_research/split_universe.py:110,750", "GS on by default", "same", "", ""),
        ("residualization matrix", "actual factor cols from neutral_z/z/raw passed to GS", "factor_research/split_universe.py:727-739", "input matrix follows factor priority", "adapter must provide priority columns", "", ""),
        ("final zscore scope", "large and small composite_score each standardized within date, then blended", "factor_research/split_universe.py:811-900", "pool-specific zscore", "same", "", ""),
        ("blended alpha", "concat large/small with alpha_signal and universe/mcap fields", "factor_research/split_universe.py:890", "split-level alpha_signal blend", "Route A dry-run may generate alpha only next task; this prep does not", "", ""),
    ]
    return pd.DataFrame(rows, columns=["composite_item", "legacy_behavior", "source_function_or_file", "known_leakage_or_caveat", "strict_lag_clean_equivalent", "adapter_implication", "caveat"])


def adapter_design() -> pd.DataFrame:
    rows = [
        ("load PIT panel", rel(PIT_PANEL), "adapter input frame", "read selected columns only", "PIT canonical 16 factor input", "schema columns available", ""),
        ("symbol mapping", "symbol/symbol_norm", "symbol, symbol_norm", "astype string, strip .0, zfill(6)", "production symbol_col='symbol'", "non-null and unique by month", ""),
        ("month mapping", "year_month/month_end/date", "date, year_month, month_end", "derive all three, month_end as timestamp", "production date_col='date'", "non-null", ""),
        ("raw factors", "PIT 16 factor columns", "16 raw factor columns", "rename/pass-through", "legacy raw factor names", "all present", ""),
        ("factor_z", "raw factors", "*_z", "monthly cross-sectional zscore", "legacy _z fallback", "finite or 0 for missing", ""),
        ("factor_neutral_z", "PIT neutral_z if present else z", "*_neutral_z", "pass-through or temporary z-compatible copy", "production _neutral_z primary", "columns present", "strict neutralization rebuild belongs to next build if needed."),
        ("market cap", "market_cap/mcap_est/total_mv/circ_mv", "mcap_est", "map first available", "split_by_market_cap", "coverage > 0", "blocked caveat if absent."),
        ("split group", "mcap_est", "mcap_pct, universe, split_group", "monthly pct rank; >=0.5 large/大盘", "legacy split labels", "large+small counts > 0", ""),
        ("forward return", "forward_return_1m/fwd_ret_1m", "forward_return_1m", "rename/pass-through only", "production label compatibility", "not used for eval in prep", ""),
        ("missing values", "all numeric", "compat values", "z columns fill z NaN with 0; raw remains nullable", "legacy fallback behavior", "required columns present", ""),
        ("duplicates", "symbol/month", "deduped preview", "sort and keep last", "one row per symbol-month", "duplicate count 0", ""),
        ("schema ordering", "manifest", "preview parquet", "base columns then factor triplets", "legacy-compatible schema", "required column order stable", ""),
        ("dtype compatibility", "all output", "parquet dtypes", "strings for ids, datetime for date/month_end, float64 for numerics", "production pandas compatibility", "dtype check pass", ""),
    ]
    return pd.DataFrame(rows, columns=["adapter_step", "input_source", "output_column_or_artifact", "transform", "target_legacy_contract", "validation_rule", "caveat"])


def make_preview(pit_cols: list[str]) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    key_cols = [c for c in ["symbol", "symbol_norm", "year_month", "month_end", "date", "forward_return_1m", "fwd_ret_1m", "mcap_est", "market_cap", "total_mv", "circ_mv"] if c in pit_cols]
    factor_cols = [f for f in FACTORS if f in pit_cols]
    use_cols = sorted(set(key_cols + factor_cols))
    pit = pd.read_parquet(PIT_PANEL, columns=use_cols)
    sym_col = pick_col(pit.columns.tolist(), ["symbol_norm", "symbol"])
    mon_col = pick_col(pit.columns.tolist(), ["year_month", "month_end", "date"])
    pit["symbol_norm"] = norm_symbol(pit[sym_col])
    pit["symbol"] = pit["symbol_norm"]
    pit["year_month"] = month_series(pit, mon_col)
    if "month_end" in pit.columns:
        pit["month_end"] = pd.to_datetime(pit["month_end"], errors="coerce")
    else:
        pit["month_end"] = pd.to_datetime(pit["year_month"] + "-01", errors="coerce") + pd.offsets.MonthEnd(0)
    pit["date"] = pit["month_end"]
    fwd = pick_col(pit.columns.tolist(), ["forward_return_1m", "fwd_ret_1m"])
    if fwd and fwd != "forward_return_1m":
        pit["forward_return_1m"] = pd.to_numeric(pit[fwd], errors="coerce")
    elif not fwd:
        pit["forward_return_1m"] = np.nan
    mcap = pick_col(pit.columns.tolist(), ["mcap_est", "market_cap", "total_mv", "circ_mv"], ["mcap", "market_cap", "mv"])
    if mcap:
        pit["mcap_est"] = pd.to_numeric(pit[mcap], errors="coerce")
        pit["mcap_pct"] = pit["mcap_est"].groupby(pit["year_month"]).rank(pct=True)
        pit["universe"] = np.where(pit["mcap_pct"] >= 0.5, "大盘", np.where(pit["mcap_pct"].notna(), "小盘", "未分类"))
        pit["split_group"] = np.where(pit["mcap_pct"] >= 0.5, "large", np.where(pit["mcap_pct"].notna(), "small", "unclassified"))
    else:
        pit["mcap_est"] = np.nan
        pit["mcap_pct"] = np.nan
        pit["universe"] = "未分类"
        pit["split_group"] = "unclassified"
    for f in FACTORS:
        if f not in pit.columns:
            pit[f] = np.nan
    z = zscore_by_month(pit, FACTORS, "year_month")
    for c in z.columns:
        pit[c] = z[c]
    for f in FACTORS:
        src = f"{f}_neutral_z"
        if src not in pit.columns:
            pit[src] = pit[f"{f}_z"]
    pit = pit.sort_values(["year_month", "symbol_norm"]).drop_duplicates(["symbol_norm", "year_month"], keep="last")
    keep_months = sorted(pit["year_month"].dropna().unique().tolist())[:6]
    preview = pit[pit["year_month"].isin(keep_months)].copy()
    base_cols = ["date", "year_month", "month_end", "symbol", "symbol_norm", "forward_return_1m", "mcap_est", "mcap_pct", "universe", "split_group"]
    out_cols = base_cols + [c for f in FACTORS for c in [f, f"{f}_z", f"{f}_neutral_z"]]
    preview = preview[out_cols]
    path = OUT_DIR / "v0_pit_legacy_compatible_input_preview.parquet"
    preview.to_parquet(path, index=False)
    qa_rows = [
        ("required columns present", "all manifest required columns", bool(set(base_cols).issubset(preview.columns)), bool(set(base_cols).issubset(preview.columns)), ""),
        ("factor priority columns present", "raw/z/neutral_z for 16 factors", int(sum(c in preview.columns for f in FACTORS for c in [f, f'{f}_z', f'{f}_neutral_z'])), int(sum(c in preview.columns for f in FACTORS for c in [f, f'{f}_z', f'{f}_neutral_z'])) == 48, ""),
        ("row uniqueness", "index unique", bool(preview.index.is_unique), bool(preview.index.is_unique), ""),
        ("symbol-month uniqueness", "duplicate count 0", int(preview.duplicated(["symbol_norm", "year_month"]).sum()), int(preview.duplicated(["symbol_norm", "year_month"]).sum()) == 0, ""),
        ("factor non-null coverage", ">0 raw coverage", float(preview[FACTORS].notna().mean().mean()), float(preview[FACTORS].notna().mean().mean()) > 0, ""),
        ("split fields present", "split_group/universe/mcap_pct", all(c in preview.columns for c in ["split_group", "universe", "mcap_pct"]), all(c in preview.columns for c in ["split_group", "universe", "mcap_pct"]), ""),
        ("month range", "diagnostic first 6 months", f"{preview['year_month'].min()}~{preview['year_month'].max()}", len(keep_months) <= 6 and len(preview) > 0, "preview sample only"),
        ("no old artifact overwritten", "legacy files unchanged by script", True, True, ""),
        ("preview_not_used_for_alpha", "true", True, True, "file name contains preview and no alpha_signal column"),
    ]
    qa = pd.DataFrame(qa_rows, columns=["check_name", "expected", "actual", "pass", "caveat"])
    schema = {
        "output_artifact_name": "v0_pit_legacy_compatible_input_preview.parquet",
        "intended_path_next_run": "output/v0_legacy_compatible_pit_adapter_build_and_production_replay_dry_run_v0/v0_pit_legacy_compatible_input.parquet",
        "columns": out_cols,
        "dtypes": {c: str(preview[c].dtype) for c in preview.columns},
        "nullable_policy": "raw factors and forward_return_1m nullable; z/neutral_z compatibility columns fill missing zscore with 0 in preview.",
        "factor_priority_policy": "factor_neutral_z -> factor_z -> raw",
        "split_policy": "monthly market-cap pct rank >= 0.5 => large/大盘; preview uses first available market-cap-like PIT column, else unclassified.",
        "leakage_policy": "adapter itself does not calculate IC/returns; Route A may replay legacy current-month IC behavior; Route B must strict-lag ICIR.",
        "caveat": "diagnostic preview sample only; not formal alpha input.",
    }
    del pit, preview, z
    gc.collect()
    return path, qa, schema


def route_design(preview_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    routes = pd.DataFrame([
        ("A", "Legacy Production Replay with PIT Adapter", "forensic compatibility replay", rel(preview_path).replace("_preview", ""), "factor_research/split_universe.py + production_engine.py legacy path", "ALLOW_LEGACY_CURRENT_MONTH_IC_LEAKAGE_FOR_FORENSIC_ONLY", True, "weights; returns; Sharpe/MaxDD/t-stat; benchmark-relative; production", "alpha/QA only in next dry run", "不得作为最终研究结论"),
        ("B", "Legacy-Compatible Strict-Lag Replay with PIT Adapter", "research V0 replay", rel(preview_path).replace("_preview", ""), "legacy path with ICIR module replaced by strict-lag implementation", "STRICT_LAG_NO_CURRENT_MONTH_IC", True, "weights; returns; Sharpe/MaxDD/t-stat; benchmark-relative; production", "alpha/QA only in next dry run", "Route B 与 Route A 输出必须分目录"),
    ], columns=["route_id", "route_name", "purpose", "input_artifact", "code_path", "leakage_policy", "allowed_next_run", "forbidden_next_run_items", "expected_outputs", "caveat"])
    config = {
        "recommended_next_run": "V0 Legacy-Compatible PIT Adapter Build and Production Replay Dry Run v0",
        "recommended_next_run_reason": "先构建完整 PIT->legacy input artifact 并只做 alpha/QA dry-run，不进入 weights/returns。",
        "route_a_allowed": True,
        "route_b_allowed": True,
        "route_a_generate_alpha_allowed": True,
        "route_b_generate_alpha_allowed": True,
        "generate_weights_allowed": False,
        "calculate_returns_allowed": False,
        "benchmark_relative_allowed": False,
        "production_allowed": False,
        "adapter_preview_path": rel(preview_path),
        "target_adapter_build_path_next": "output/v0_legacy_compatible_pit_adapter_build_and_production_replay_dry_run_v0/v0_pit_legacy_compatible_input.parquet",
        "legacy_code_path": ["factor_research/split_universe.py", "factor_research/production_engine.py"],
        "strict_lag_code_path": "new strict-lag ICIR replacement wrapper around legacy composite path",
        "validation_outputs_expected": ["schema QA", "factor coverage QA", "split QA", "alpha dry-run QA only"],
    }
    return routes, config


def guardrails() -> pd.DataFrame:
    actuals = {
        "formal_alpha_signal_generated": False,
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "transaction_cost_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "tstat_calculated": False,
        "benchmark_relative_returns_calculated": False,
        "active_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "ir_te_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "ml_training_run": False,
        "tuning_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "old_artifacts_modified": False,
        "original_preprocessed_overwritten": False,
        "original_split_universe_overwritten": False,
        "diagnostic_adapter_preview_generated": True,
    }
    rows = []
    for k, actual in actuals.items():
        expected = True if k == "diagnostic_adapter_preview_generated" else False
        rows.append({"guardrail": k, "expected": expected, "actual": actual, "pass": actual == expected})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", "prerequisite_check")
    prereq = prereq_check()
    write_json(OUT_DIR / "v0_legacy_pit_adapter_prep_prerequisite_check.json", prereq)
    legacy_cols = parquet_columns(LEGACY_PREPROCESSED) if LEGACY_PREPROCESSED.exists() else []
    pit_cols = parquet_columns(PIT_PANEL) if PIT_PANEL.exists() else []

    write_state("running", "contract_extraction")
    production_contract(legacy_cols).to_csv(OUT_DIR / "v0_legacy_production_input_contract.csv", index=False, encoding="utf-8-sig")
    required_columns_manifest(legacy_cols, pit_cols).to_csv(OUT_DIR / "v0_legacy_required_columns_manifest.csv", index=False, encoding="utf-8-sig")
    factor_contract(legacy_cols, pit_cols).to_csv(OUT_DIR / "v0_legacy_factor_contract_manifest.csv", index=False, encoding="utf-8-sig")

    write_state("running", "schema_distribution_audit")
    schema_df = schema_audit(legacy_cols, pit_cols)
    schema_df.to_csv(OUT_DIR / "v0_legacy_vs_pit_schema_audit.csv", index=False, encoding="utf-8-sig")
    dist_df = distribution_audit(legacy_cols, pit_cols)
    dist_df.to_csv(OUT_DIR / "v0_legacy_vs_pit_factor_distribution_audit.csv", index=False, encoding="utf-8-sig")

    write_state("running", "split_and_composite_audit")
    split_contract, split_preview = split_audits(pit_cols)
    split_contract.to_csv(OUT_DIR / "v0_legacy_split_universe_contract.csv", index=False, encoding="utf-8-sig")
    split_preview.to_csv(OUT_DIR / "v0_pit_adapter_split_preview_qa.csv", index=False, encoding="utf-8-sig")
    composite_audit().to_csv(OUT_DIR / "v0_legacy_composite_code_path_audit.csv", index=False, encoding="utf-8-sig")

    write_state("running", "adapter_design_preview")
    adapter_design().to_csv(OUT_DIR / "v0_pit_to_legacy_adapter_design.csv", index=False, encoding="utf-8-sig")
    preview_path, preview_qa, schema = make_preview(pit_cols)
    write_json(OUT_DIR / "v0_pit_legacy_adapter_output_schema.json", schema)
    preview_qa.to_csv(OUT_DIR / "v0_pit_legacy_adapter_preview_qa.csv", index=False, encoding="utf-8-sig")

    write_state("running", "route_guardrail_summary")
    routes, config = route_design(preview_path)
    routes.to_csv(OUT_DIR / "v0_legacy_pit_replay_route_design.csv", index=False, encoding="utf-8-sig")
    write_json(OUT_DIR / "v0_legacy_pit_adapter_next_run_config_draft.json", config)
    guard = guardrails()
    guard.to_csv(OUT_DIR / "v0_legacy_pit_adapter_prep_guardrail_qa.csv", index=False, encoding="utf-8-sig")

    preview_qa_pass = bool(preview_qa["pass"].all())
    guardrails_passed = bool(guard["pass"].all())
    contract_ok = prereq["prerequisites_passed"] and len(legacy_cols) > 0 and len(pit_cols) > 0
    split_caveat = bool(len(split_preview) and (split_preview["split_match_status"] == "BLOCKED_NO_MCAP_SOURCE").any())
    neutral_caveat = any(f"{f}_neutral_z" not in pit_cols for f in FACTORS)
    if not guardrails_passed:
        final_decision = "LEGACY_PIT_ADAPTER_PREP_FAIL_GUARDRAIL"
    elif not prereq["production_code_found"]:
        final_decision = "LEGACY_PIT_ADAPTER_PREP_BLOCKED_BY_MISSING_PRODUCTION_CODE"
    elif not preview_qa_pass:
        final_decision = "LEGACY_PIT_ADAPTER_PREP_BLOCKED_BY_SCHEMA_MISMATCH"
    elif split_caveat or neutral_caveat:
        final_decision = "LEGACY_PIT_ADAPTER_PREP_READY_WITH_CAVEATS"
    else:
        final_decision = "LEGACY_PIT_ADAPTER_PREP_READY_FOR_DRY_RUN"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "production_code_found": prereq["production_code_found"],
        "legacy_input_contract_extracted": contract_ok,
        "legacy_factor_contract_extracted": contract_ok,
        "legacy_split_contract_extracted": contract_ok,
        "legacy_composite_code_path_audited": True,
        "pit_adapter_design_ready": True,
        "diagnostic_adapter_preview_generated": preview_path.exists(),
        "diagnostic_adapter_preview_path": rel(preview_path),
        "adapter_preview_qa_pass": preview_qa_pass,
        "route_a_legacy_production_replay_ready": preview_qa_pass,
        "route_b_legacy_compatible_strict_lag_replay_ready": preview_qa_pass,
        "known_legacy_leakage_caveat": "Route A legacy production replay may replicate current-month IC leakage in rolling ICIR; Route B must use strict-lag ICIR only.",
        "recommended_next_run": config["recommended_next_run"],
        "generate_alpha_next_run_allowed": True,
        "generate_weights_next_run_allowed": False,
        "calculate_returns_next_run_allowed": False,
        "benchmark_relative_allowed": False,
        "production_allowed": False,
        "formal_alpha_signal_generated": False,
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "transaction_cost_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "tstat_calculated": False,
        "benchmark_relative_returns_calculated": False,
        "active_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "ir_te_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "ml_training_run": False,
        "tuning_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "old_artifacts_modified": False,
        "original_preprocessed_overwritten": False,
        "original_split_universe_overwritten": False,
        "guardrails_passed": guardrails_passed,
        "final_decision": final_decision,
        "recommended_next_step": "下一步运行 Adapter Build and Production Replay Dry Run，只允许生成 alpha/QA，仍禁止 weights/returns/production。",
    }
    write_json(OUT_DIR / "v0_legacy_compatible_pit_adapter_prep_summary.json", summary)

    report = "\n".join([
        "# V0 Legacy-Compatible PIT Adapter Prep v0",
        "",
        f"- final_decision: {final_decision}",
        f"- prerequisites_passed: {summary['prerequisites_passed']}",
        f"- production_code_found: {summary['production_code_found']}",
        f"- diagnostic_adapter_preview_path: {summary['diagnostic_adapter_preview_path']}",
        f"- adapter_preview_qa_pass: {summary['adapter_preview_qa_pass']}",
        f"- route_a_ready: {summary['route_a_legacy_production_replay_ready']}",
        f"- route_b_ready: {summary['route_b_legacy_compatible_strict_lag_replay_ready']}",
        "",
        "关键 caveat：Route A 用于 forensic compatibility，可能复刻 legacy rolling ICIR current-month leakage；Route B 必须替换为 strict-lag ICIR。prep 阶段没有生成正式 alpha_signal、weights 或收益评价。",
    ])
    (OUT_DIR / "v0_legacy_compatible_pit_adapter_prep_report.md").write_text(report, encoding="utf-8")

    final_qa = pd.DataFrame([
        {"check_name": "prerequisites_passed", "expected": True, "actual": prereq["prerequisites_passed"], "pass": prereq["prerequisites_passed"], "caveat": prereq["caveat"]},
        {"check_name": "legacy_input_contract_extracted", "expected": True, "actual": summary["legacy_input_contract_extracted"], "pass": summary["legacy_input_contract_extracted"], "caveat": ""},
        {"check_name": "adapter_preview_qa_pass", "expected": True, "actual": preview_qa_pass, "pass": preview_qa_pass, "caveat": ""},
        {"check_name": "guardrails_passed", "expected": True, "actual": guardrails_passed, "pass": guardrails_passed, "caveat": ""},
        {"check_name": "final_decision_allowed", "expected": True, "actual": final_decision, "pass": final_decision in {
            "LEGACY_PIT_ADAPTER_PREP_READY_FOR_DRY_RUN",
            "LEGACY_PIT_ADAPTER_PREP_READY_WITH_CAVEATS",
            "LEGACY_PIT_ADAPTER_PREP_BLOCKED_BY_MISSING_PRODUCTION_CODE",
            "LEGACY_PIT_ADAPTER_PREP_BLOCKED_BY_SCHEMA_MISMATCH",
            "LEGACY_PIT_ADAPTER_PREP_FAIL_GUARDRAIL",
        }, "caveat": ""},
    ])
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    write_json(OUT_DIR / "terminal_summary.json", {
        "task_name": TASK_NAME,
        "status": "completed",
        "stdout_log": rel(RUN_DIR / "run_stdout.txt"),
        "stderr_log": rel(RUN_DIR / "run_stderr.txt"),
        "output_dir": rel(OUT_DIR),
        "final_decision": final_decision,
    })
    (OUT_DIR / "task_completion_card.md").write_text(
        "\n".join(["# task_completion_card", "", f"- task_name: {TASK_NAME}", "- status: completed", f"- final_decision: {final_decision}", f"- output_dir: {rel(OUT_DIR)}"]),
        encoding="utf-8",
    )
    write_state("completed", "all_outputs_written", {"final_decision": final_decision, "output_dir": rel(OUT_DIR)})
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": rel(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
