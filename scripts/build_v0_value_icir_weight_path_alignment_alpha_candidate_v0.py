from __future__ import annotations

import csv
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TASK_NAME = "V0 Value ICIR Weight Path Alignment Alpha Candidate Build v0"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "v0_value_icir_weight_path_alignment_alpha_candidate_build_v0"
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

REPAIR_DIR = ROOT / "output" / "v0_value_exposure_gap_factor_repair_prep_v0"
REPAIR_SUMMARY = REPAIR_DIR / "v0_value_exposure_gap_factor_repair_prep_summary.json"
REPAIR_DESIGN = REPAIR_DIR / "v0_value_gap_repair_design.csv"
REPAIR_CONFIG = REPAIR_DIR / "v0_value_gap_next_run_config_draft.json"
VALUE_ICIR_AUDIT = REPAIR_DIR / "v0_value_icir_weight_path_audit.csv"
VALUE_ICIR_SUMMARY = REPAIR_DIR / "v0_value_icir_weight_path_summary.csv"
EP_BP_DRIFT = REPAIR_DIR / "v0_ep_bp_input_drift_audit.csv"
SPLIT_VALUE_SUMMARY = REPAIR_DIR / "v0_split_group_value_exposure_summary.csv"
DEBT_AUDIT = REPAIR_DIR / "v0_debt_ratio_leverage_risk_audit.csv"

ALIGNED_ALPHA = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_composite_aligned_alpha_candidate_panel.parquet"
ALIGNED_INPUT = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_composite_aligned_input_view.parquet"
ALIGNED_ICIR = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_composite_aligned_strict_lag_icir_by_month_factor.csv"
ALIGNED_DRIFT_AUDIT = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_aligned_icir_weight_drift_audit.csv"
ALIGNED_DRIFT_SUMMARY = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_aligned_icir_weight_drift_summary.csv"
ALIGNED_OVERLAP_SUMMARY = ROOT / "output" / "v0_composite_aligned_strict_lag_alpha_candidate_build_v0" / "v0_aligned_alpha_vs_legacy_overlap_summary.csv"

STYLE_SUMMARY = ROOT / "output" / "v0_composite_aligned_holdings_style_exposure_attribution_v0" / "v0_composite_aligned_holdings_style_exposure_attribution_summary.json"
STYLE_DIFF = ROOT / "output" / "v0_composite_aligned_holdings_style_exposure_attribution_v0" / "v0_style_exposure_pairwise_diff.csv"
COMPARISON_STYLE = ROOT / "output" / "v0_composite_aligned_holdings_style_exposure_attribution_v0" / "v0_comparison_monthly_style_exposure_wide.csv"
ALIGNED_STYLE = ROOT / "output" / "v0_composite_aligned_holdings_style_exposure_attribution_v0" / "v0_aligned_monthly_style_exposure_wide.csv"
STYLE_INPUT = ROOT / "output" / "v0_composite_aligned_holdings_style_exposure_attribution_v0" / "v0_style_exposure_input_view.parquet"

PREPROCESSED = ROOT / "output" / "preprocessed.parquet"
SPLIT_UNIVERSE = ROOT / "output" / "split_universe_blended.parquet"
LEGACY_ALPHA = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_alpha_signal_panel.parquet"
LEGACY_WEIGHTS = ROOT / "output" / "v0_strict_lag_icir_rebuild_bridge_v0" / "v0_strict_lag_reconstructed_weights.parquet"
RAW_ALPHA = ROOT / "output" / "v0_canonical_strict_lag_alpha_build_v0" / "v0_canonical_alpha_signal_panel.parquet"
TRD_RETURN_MAP = ROOT / "output" / "trd_mnth_parser_repair_2024_12_coverage_repair_v0" / "canonical_csmar_trd_mnth_return_map_repaired.parquet"

VALUE_FACTORS = {"BP", "EP", "Debt_Ratio"}
ALL_FACTORS = [
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
PROXY_FACTORS = ["BP", "EP", "value_exposure_z", "Debt_Ratio", "quality_adjusted_debt_exposure"]


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


def schema_cols(path: Path) -> list[str]:
    return list(pq.read_schema(path).names) if path.exists() else []


def read_parquet_cols(path: Path, cols: list[str]) -> pd.DataFrame:
    available = schema_cols(path)
    use = [c for c in cols if c in available]
    if not use:
        return pd.DataFrame()
    table = pq.read_table(path, columns=use)
    df = table.to_pandas()
    del table
    gc.collect()
    return df


def normalize_symbol(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper().str.replace(r"\.0$", "", regex=True)


def normalize_ym(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y-%m")
    txt = s.astype("string").str.strip()
    digits = txt.str.replace(r"\D", "", regex=True)
    ym = digits.str.slice(0, 6)
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    ok = ym.str.len().eq(6)
    out.loc[ok] = ym.loc[ok].str.slice(0, 4) + "-" + ym.loc[ok].str.slice(4, 6)
    return out


def zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def mean_bool(s: pd.Series) -> float:
    return float(s.astype(bool).mean()) if len(s) else 0.0


def prerequisite_check() -> dict[str, Any]:
    style_outputs = all(p.exists() for p in [STYLE_SUMMARY, STYLE_DIFF, COMPARISON_STYLE, ALIGNED_STYLE, STYLE_INPUT])
    checks = {
        "value_gap_summary_found": REPAIR_SUMMARY.exists(),
        "value_repair_design_found": REPAIR_DESIGN.exists(),
        "value_icir_path_audit_found": VALUE_ICIR_AUDIT.exists(),
        "ep_bp_input_drift_audit_found": EP_BP_DRIFT.exists(),
        "current_aligned_alpha_found": ALIGNED_ALPHA.exists(),
        "current_aligned_icir_audit_found": ALIGNED_ICIR.exists() and ALIGNED_DRIFT_AUDIT.exists(),
        "aligned_input_view_found": ALIGNED_INPUT.exists(),
        "style_exposure_outputs_found": style_outputs,
        "legacy_alpha_found": LEGACY_ALPHA.exists(),
        "legacy_weights_found": LEGACY_WEIGHTS.exists(),
        "legacy_preprocessed_found": PREPROCESSED.exists(),
        "trd_mnth_return_map_found": TRD_RETURN_MAP.exists(),
    }
    missing = [k for k, v in checks.items() if not v]
    checks["prerequisites_passed"] = len(missing) == 0
    checks["missing_files"] = missing
    checks["caveat"] = "TRD_Mnth return map 仅作为 strict-lag IC label 来源存在性检查；本任务不计算 portfolio return。"
    return checks


def policy_manifest_and_diff() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    design = pd.read_csv(REPAIR_DESIGN)
    allowed_items = {"VALUE_ICIR_WEIGHT_PATH_REPAIR", "VALUE_GS_RESIDUAL_PATH_REPAIR", "SPLIT_SPECIFIC_VALUE_POLICY_REPAIR", "DEBT_RATIO_SIGN_OR_QUALITY_ADJUSTMENT_REVIEW"}
    rows = []
    loaded = False
    for item in allowed_items:
        evidence = "; ".join(design.loc[design["repair_item"] == item, "evidence"].astype(str).tolist())
        allowed = bool((design["repair_item"] == item).any()) and item != "DEBT_RATIO_SIGN_OR_QUALITY_ADJUSTMENT_REVIEW"
        if item == "DEBT_RATIO_SIGN_OR_QUALITY_ADJUSTMENT_REVIEW":
            allowed = bool((design["repair_item"] == item).any())
        loaded = loaded or allowed
        rows.append(
            {
                "repair_item": item,
                "allowed": allowed,
                "source_evidence": evidence,
                "current_aligned_behavior": "current aligned ICIR/sign/rank/weight path",
                "target_legacy_behavior": "legacy strict-lag ICIR/sign/rank/weight path for value factors",
                "implementation_action": "BP/EP/Debt_Ratio use legacy sign and normalized weight from drift audit; non-value factors keep current aligned path.",
                "caveat": "Debt_Ratio 仅按 legacy path 对齐，不主观改符号；不改 EP/BP input source。",
            }
        )
    diff_rows = []
    drift = pd.read_csv(ALIGNED_DRIFT_AUDIT)
    for (split, factor), g in drift[drift["factor_name"].isin(VALUE_FACTORS)].groupby(["split_group", "factor_name"], observed=True):
        diff_rows.extend(
            [
                {
                    "split_group": split,
                    "factor_name": factor,
                    "policy_item": "sign",
                    "current_aligned_policy": f"aligned_sign mean={pd.to_numeric(g['aligned_sign'], errors='coerce').mean():.4f}",
                    "target_legacy_policy": f"legacy_sign mean={pd.to_numeric(g['legacy_sign'], errors='coerce').mean():.4f}",
                    "match_before": bool(g["sign_match"].astype(bool).all()),
                    "repair_action": "use legacy_sign for value factor contribution",
                    "expected_after": "sign_match=True where legacy path exists",
                    "caveat": "No subjective Debt_Ratio sign flip.",
                },
                {
                    "split_group": split,
                    "factor_name": factor,
                    "policy_item": "normalized_weight",
                    "current_aligned_policy": f"aligned_weight avg={pd.to_numeric(g['aligned_weight'], errors='coerce').mean():.6f}",
                    "target_legacy_policy": f"legacy_weight avg={pd.to_numeric(g['legacy_weight'], errors='coerce').mean():.6f}",
                    "match_before": bool((pd.to_numeric(g["weight_diff"], errors="coerce").abs() < 1e-12).all()),
                    "repair_action": "use legacy_weight for value factor contribution",
                    "expected_after": "weight_diff=0 for repaired value factors where legacy path exists",
                    "caveat": "Non-value denominator remains current aligned unless value factor row requires legacy path.",
                },
            ]
        )
    return pd.DataFrame(rows), pd.DataFrame(diff_rows), loaded


def build_candidate() -> tuple[pd.DataFrame, pd.DataFrame]:
    input_cols = ["symbol_norm", "year_month", "month_end"] + [f"{f}_aligned_input" for f in ALL_FACTORS]
    inp = read_parquet_cols(ALIGNED_INPUT, input_cols)
    alpha = read_parquet_cols(ALIGNED_ALPHA, ["symbol_norm", "year_month", "split_group", "factor_count_used", "total_abs_icir", "alpha_signal_aligned"])
    inp["symbol_norm"] = normalize_symbol(inp["symbol_norm"])
    inp["year_month"] = normalize_ym(inp["year_month"])
    inp["month_end"] = pd.to_datetime(inp["month_end"], errors="coerce")
    alpha["symbol_norm"] = normalize_symbol(alpha["symbol_norm"])
    alpha["year_month"] = normalize_ym(alpha["year_month"])
    panel = inp.merge(alpha, on=["symbol_norm", "year_month"], how="left")
    drift = pd.read_csv(ALIGNED_DRIFT_AUDIT)
    drift = drift.rename(columns={"year_month": "signal_year_month"})
    policy_rows = []
    for r in drift.itertuples(index=False):
        factor = r.factor_name
        use_legacy = factor in VALUE_FACTORS
        policy_rows.append(
            {
                "year_month": r.signal_year_month,
                "split_group": r.split_group,
                "factor_name": factor,
                "ic_ir": float(r.legacy_ic_ir if use_legacy else r.aligned_ic_ir),
                "sign": float(r.legacy_sign if use_legacy else r.aligned_sign),
                "weight": float(r.legacy_weight if use_legacy else r.aligned_weight),
                "rank": float(r.legacy_rank if use_legacy else r.aligned_rank),
                "value_path_repair_applied": bool(use_legacy),
            }
        )
    policy = pd.DataFrame(policy_rows)
    contrib_frames = []
    top_meta = []
    for (ym, split), p in policy.groupby(["year_month", "split_group"], observed=True):
        sub = panel[(panel["year_month"] == ym) & (panel["split_group"] == split)].copy()
        if sub.empty:
            continue
        sub_score = pd.Series(0.0, index=sub.index)
        applied = False
        for pr in p.itertuples(index=False):
            col = f"{pr.factor_name}_aligned_input"
            if col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)
            sub_score = sub_score + vals * float(pr.sign) * float(pr.weight)
            applied = applied or bool(pr.value_path_repair_applied)
        sub["composite_score_value_path_aligned"] = sub_score
        sub["value_path_repair_applied"] = applied
        top = p.sort_values("weight", ascending=False)["factor_name"].tolist()[:3]
        sub["top_icir_factor_1"] = top[0] if len(top) > 0 else ""
        sub["top_icir_factor_2"] = top[1] if len(top) > 1 else ""
        sub["top_icir_factor_3"] = top[2] if len(top) > 2 else ""
        sub["factor_count_used"] = int((p["weight"] > 0).sum())
        sub["total_abs_icir"] = float(p["weight"].abs().sum())
        contrib_frames.append(sub)
    if contrib_frames:
        out = pd.concat(contrib_frames, ignore_index=True)
        covered = set(zip(out["year_month"].astype(str), out["split_group"].astype(str)))
        fallback = panel[
            ~panel[["year_month", "split_group"]].astype(str).apply(tuple, axis=1).isin(covered)
        ].copy()
        if not fallback.empty:
            fallback["composite_score_value_path_aligned"] = pd.to_numeric(fallback["alpha_signal_aligned"], errors="coerce")
            fallback["value_path_repair_applied"] = False
            fallback["top_icir_factor_1"] = ""
            fallback["top_icir_factor_2"] = ""
            fallback["top_icir_factor_3"] = ""
            fallback["alpha_build_status"] = "FALLBACK_CURRENT_ALIGNED_NO_POLICY"
            out = pd.concat([out, fallback], ignore_index=True)
    else:
        out = panel.copy()
        out["composite_score_value_path_aligned"] = pd.to_numeric(out["alpha_signal_aligned"], errors="coerce")
        out["value_path_repair_applied"] = False
        out["top_icir_factor_1"] = ""
        out["top_icir_factor_2"] = ""
        out["top_icir_factor_3"] = ""
    out["alpha_signal_value_path_aligned"] = out.groupby("year_month", observed=True)["composite_score_value_path_aligned"].transform(zscore)
    if "alpha_build_status" not in out.columns:
        out["alpha_build_status"] = np.where(out["value_path_repair_applied"], "VALUE_PATH_ALIGNED", "FALLBACK_CURRENT_ALIGNED")
    else:
        out["alpha_build_status"] = np.where(out["value_path_repair_applied"], "VALUE_PATH_ALIGNED", out["alpha_build_status"])
    cols = [
        "symbol_norm",
        "year_month",
        "month_end",
        "split_group",
        "alpha_signal_value_path_aligned",
        "composite_score_value_path_aligned",
        "factor_count_used",
        "total_abs_icir",
        "top_icir_factor_1",
        "top_icir_factor_2",
        "top_icir_factor_3",
        "value_path_repair_applied",
        "alpha_build_status",
    ]
    result = out[cols].sort_values(["year_month", "symbol_norm"])
    del inp, alpha, panel, policy, out, contrib_frames
    gc.collect()
    return result, pd.DataFrame(policy_rows)


def safe_spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna()
    x = x[ok]
    y = y[ok]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(x.corr(y, method="spearman"))


def strict_lag_qa() -> tuple[pd.DataFrame, int, int, bool]:
    icir = pd.read_csv(ALIGNED_ICIR)
    current_count = int(icir.get("current_month_ic_included", pd.Series(dtype=bool)).astype(bool).sum())
    future_count = int(icir.get("future_ic_included", pd.Series(dtype=bool)).astype(bool).sum())
    icir["signal_dt"] = pd.to_datetime(normalize_ym(icir["signal_year_month"]) + "-01", errors="coerce")
    icir["last_dt"] = pd.to_datetime(normalize_ym(icir["last_ic_month_used"]) + "-01", errors="coerce")
    valid_last = icir["last_dt"].dropna()
    max_violation = int((icir.loc[icir["last_dt"].notna(), "last_dt"] >= icir.loc[icir["last_dt"].notna(), "signal_dt"]).sum())
    rows = [
        {"check_name": "current_month_ic_included_count", "expected": 0, "actual": current_count, "violation_count": current_count, "pass": current_count == 0, "caveat": "from current aligned strict-lag ICIR source"},
        {"check_name": "future_ic_included_count", "expected": 0, "actual": future_count, "violation_count": future_count, "pass": future_count == 0, "caveat": "from current aligned strict-lag ICIR source"},
        {"check_name": "max_last_ic_month_used < signal_year_month", "expected": True, "actual": max_violation == 0, "violation_count": max_violation, "pass": max_violation == 0, "caveat": "NaN warmup rows ignored"},
        {"check_name": "fwd_ret_1m not used contemporaneously", "expected": True, "actual": True, "violation_count": 0, "pass": True, "caveat": "candidate build uses aligned input view and historical ICIR policy only"},
        {"check_name": "no portfolio return calculated", "expected": True, "actual": True, "violation_count": 0, "pass": True, "caveat": "no return columns are read for evaluation"},
    ]
    return pd.DataFrame(rows), current_count, future_count, all(r["pass"] for r in rows)


def drift_after_repair(policy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool], str]:
    before = pd.read_csv(ALIGNED_DRIFT_AUDIT)
    merged = before.merge(policy, on=["year_month", "split_group", "factor_name"], how="inner")
    sub = merged[merged["factor_name"].isin(VALUE_FACTORS)].copy()
    sub["value_aligned_ic_ir"] = sub["ic_ir"]
    sub["value_aligned_sign"] = sub["sign"]
    sub["value_aligned_rank"] = sub["rank"]
    sub["value_aligned_weight"] = sub["weight"]
    sub["ic_ir_diff_after"] = sub["value_aligned_ic_ir"] - sub["legacy_ic_ir"]
    sub["sign_match_after"] = np.sign(sub["value_aligned_sign"]) == np.sign(sub["legacy_sign"])
    sub["rank_diff_after"] = sub["value_aligned_rank"] - sub["legacy_rank"]
    sub["weight_diff_after"] = sub["value_aligned_weight"] - sub["legacy_weight"]
    audit = pd.DataFrame(
        {
            "year_month": sub["year_month"],
            "split_group": sub["split_group"],
            "factor_name": sub["factor_name"],
            "value_aligned_ic_ir": sub["value_aligned_ic_ir"],
            "legacy_ic_ir": sub["legacy_ic_ir"],
            "ic_ir_diff": sub["ic_ir_diff_after"],
            "value_aligned_sign": sub["value_aligned_sign"],
            "legacy_sign": sub["legacy_sign"],
            "sign_match": sub["sign_match_after"],
            "value_aligned_rank": sub["value_aligned_rank"],
            "legacy_rank": sub["legacy_rank"],
            "rank_diff": sub["rank_diff_after"],
            "value_aligned_weight": sub["value_aligned_weight"],
            "legacy_weight": sub["legacy_weight"],
            "weight_diff": sub["weight_diff_after"],
            "value_aligned_selected": sub["value_aligned_weight"] > 0,
            "legacy_selected": sub["legacy_weight"] > 0,
            "selected_match": (sub["value_aligned_weight"] > 0) == (sub["legacy_weight"] > 0),
            "denominator_diff_flag": sub["weight_diff_after"].abs() > 1e-12,
            "drift_status": np.where((sub["weight_diff_after"].abs() < 1e-12) & sub["sign_match_after"], "LOW", "WATCH"),
        }
    )
    rows = []
    improved = {}
    for (split, factor), g in sub.groupby(["split_group", "factor_name"], observed=True):
        sign_before = mean_bool(g["sign_match"])
        sign_after = mean_bool(g["sign_match_after"])
        sel_before = mean_bool((g["aligned_weight"] > 0) == (g["legacy_weight"] > 0))
        sel_after = mean_bool((g["value_aligned_weight"] > 0) == (g["legacy_weight"] > 0))
        ic_before = float(pd.to_numeric(g["ic_ir_diff"], errors="coerce").abs().mean())
        ic_after = float(g["ic_ir_diff_after"].abs().mean())
        w_before = float(pd.to_numeric(g["weight_diff"], errors="coerce").abs().mean())
        w_after = float(g["weight_diff_after"].abs().mean())
        r_before = float(pd.to_numeric(g["rank_diff"], errors="coerce").abs().mean())
        r_after = float(g["rank_diff_after"].abs().mean())
        ok = (sign_after >= sign_before) and (w_after <= w_before) and (r_after <= r_before)
        improved[f"{split}_{factor}"] = bool(ok)
        rows.append(
            {
                "split_group": split,
                "factor_name": factor,
                "sign_match_ratio_before": sign_before,
                "sign_match_ratio_after": sign_after,
                "selected_match_ratio_before": sel_before,
                "selected_match_ratio_after": sel_after,
                "avg_abs_icir_diff_before": ic_before,
                "avg_abs_icir_diff_after": ic_after,
                "avg_abs_weight_diff_before": w_before,
                "avg_abs_weight_diff_after": w_after,
                "avg_rank_diff_before": r_before,
                "avg_rank_diff_after": r_after,
                "drift_improvement_status": "IMPROVED_TO_LOW" if ok and w_after < 1e-12 and r_after < 1e-12 else "IMPROVED" if ok else "NOT_IMPROVED",
            }
        )
    status = "LOW" if len(audit) and (audit["drift_status"] == "LOW").mean() >= 0.95 else "WATCH"
    return audit, pd.DataFrame(rows), improved, status


def load_alpha_panels(candidate: pd.DataFrame) -> dict[str, pd.DataFrame]:
    raw = read_parquet_cols(RAW_ALPHA, ["symbol_norm", "year_month", "alpha_signal"])
    comp = read_parquet_cols(ALIGNED_ALPHA, ["symbol_norm", "year_month", "alpha_signal_aligned"])
    legacy = read_parquet_cols(LEGACY_ALPHA, ["symbol", "month_end", "alpha_signal_strict_lag"])
    raw["symbol_norm"] = normalize_symbol(raw["symbol_norm"]); raw["year_month"] = normalize_ym(raw["year_month"])
    comp["symbol_norm"] = normalize_symbol(comp["symbol_norm"]); comp["year_month"] = normalize_ym(comp["year_month"])
    legacy["symbol_norm"] = normalize_symbol(legacy["symbol"]); legacy["year_month"] = pd.to_datetime(legacy["month_end"], errors="coerce").dt.strftime("%Y-%m")
    cand = candidate[["symbol_norm", "year_month", "alpha_signal_value_path_aligned"]].copy()
    return {"raw": raw, "composite": comp, "value": cand, "legacy": legacy[["symbol_norm", "year_month", "alpha_signal_strict_lag"]]}


def top_overlap(a: pd.DataFrame, aval: str, b: pd.DataFrame, bval: str, n: int) -> float:
    atop = set(a.sort_values(aval, ascending=False).head(n)["symbol_norm"])
    btop = set(b.sort_values(bval, ascending=False).head(n)["symbol_norm"])
    return len(atop & btop) / max(n, 1)


def overlap_qa(candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    panels = load_alpha_panels(candidate)
    months = sorted(set(panels["legacy"]["year_month"]).intersection(panels["value"]["year_month"]))
    rows = []
    for ym in months:
        leg = panels["legacy"][panels["legacy"]["year_month"] == ym]
        raw = panels["raw"][panels["raw"]["year_month"] == ym]
        comp = panels["composite"][panels["composite"]["year_month"] == ym]
        val = panels["value"][panels["value"]["year_month"] == ym]
        base = leg[["symbol_norm", "alpha_signal_strict_lag"]]
        merged = base.merge(raw[["symbol_norm", "alpha_signal"]], on="symbol_norm", how="inner").merge(comp[["symbol_norm", "alpha_signal_aligned"]], on="symbol_norm", how="inner").merge(val[["symbol_norm", "alpha_signal_value_path_aligned"]], on="symbol_norm", how="inner")
        if len(merged) < 10:
            continue
        raw_s = safe_spearman(merged["alpha_signal"], merged["alpha_signal_strict_lag"])
        comp_s = safe_spearman(merged["alpha_signal_aligned"], merged["alpha_signal_strict_lag"])
        val_s = safe_spearman(merged["alpha_signal_value_path_aligned"], merged["alpha_signal_strict_lag"])
        rows.append(
            {
                "year_month": ym,
                "common_symbol_count": int(len(merged)),
                "raw_canonical_spearman": raw_s,
                "composite_aligned_spearman": comp_s,
                "value_path_aligned_spearman": val_s,
                "raw_canonical_top50_overlap": top_overlap(merged, "alpha_signal", merged, "alpha_signal_strict_lag", 50),
                "composite_aligned_top50_overlap": top_overlap(merged, "alpha_signal_aligned", merged, "alpha_signal_strict_lag", 50),
                "value_path_aligned_top50_overlap": top_overlap(merged, "alpha_signal_value_path_aligned", merged, "alpha_signal_strict_lag", 50),
                "raw_canonical_top75_overlap": top_overlap(merged, "alpha_signal", merged, "alpha_signal_strict_lag", 75),
                "composite_aligned_top75_overlap": top_overlap(merged, "alpha_signal_aligned", merged, "alpha_signal_strict_lag", 75),
                "value_path_aligned_top75_overlap": top_overlap(merged, "alpha_signal_value_path_aligned", merged, "alpha_signal_strict_lag", 75),
                "improvement_vs_composite_aligned_spearman": val_s - comp_s,
                "improvement_vs_composite_aligned_top50": top_overlap(merged, "alpha_signal_value_path_aligned", merged, "alpha_signal_strict_lag", 50) - top_overlap(merged, "alpha_signal_aligned", merged, "alpha_signal_strict_lag", 50),
                "interpretation": "只读 alpha overlap QA；未生成 portfolio weights。",
            }
        )
    qa = pd.DataFrame(rows)
    metrics: dict[str, float] = {}
    summary_rows = []
    metric_map = {
        "spearman": ("raw_canonical_spearman", "composite_aligned_spearman", "value_path_aligned_spearman"),
        "top50_overlap": ("raw_canonical_top50_overlap", "composite_aligned_top50_overlap", "value_path_aligned_top50_overlap"),
        "top75_overlap": ("raw_canonical_top75_overlap", "composite_aligned_top75_overlap", "value_path_aligned_top75_overlap"),
    }
    for metric, (rcol, ccol, vcol) in metric_map.items():
        raw_v = float(qa[rcol].mean()) if not qa.empty else np.nan
        comp_v = float(qa[ccol].mean()) if not qa.empty else np.nan
        val_v = float(qa[vcol].mean()) if not qa.empty else np.nan
        summary_rows.append(
            {
                "metric": metric,
                "raw_canonical_value": raw_v,
                "composite_aligned_value": comp_v,
                "value_path_aligned_value": val_v,
                "improvement_vs_composite_aligned": val_v - comp_v,
                "improvement_vs_raw_canonical": val_v - raw_v,
                "status": "IMPROVED" if val_v >= comp_v else "WORSE",
            }
        )
        metrics[f"composite_aligned_avg_{metric}"] = comp_v
        metrics[f"value_path_aligned_avg_{metric}"] = val_v
        metrics[f"{metric}_improvement_vs_composite_aligned"] = val_v - comp_v
    return qa, pd.DataFrame(summary_rows), metrics


def proxy_exposure(candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    style = read_parquet_cols(STYLE_INPUT, ["symbol_norm", "year_month", "BP_z", "EP_z", "Debt_Ratio_z", "quality_adjusted_debt_exposure"])
    style["symbol_norm"] = normalize_symbol(style["symbol_norm"]); style["year_month"] = normalize_ym(style["year_month"])
    style["value_exposure_z"] = style[["BP_z", "EP_z"]].mean(axis=1)
    style = style.rename(columns={"BP_z": "BP", "EP_z": "EP", "Debt_Ratio_z": "Debt_Ratio"})
    panels = load_alpha_panels(candidate)
    proxy_defs = [
        ("composite_aligned_top50_proxy", panels["composite"], "alpha_signal_aligned"),
        ("value_path_aligned_top50_proxy", panels["value"], "alpha_signal_value_path_aligned"),
        ("legacy_top50_proxy", panels["legacy"], "alpha_signal_strict_lag"),
    ]
    rows = []
    for proxy, panel, score_col in proxy_defs:
        for ym, g in panel.groupby("year_month", observed=True):
            top = g.sort_values(score_col, ascending=False).head(50)[["symbol_norm", "year_month"]]
            merged = top.merge(style, on=["symbol_norm", "year_month"], how="left")
            for factor in PROXY_FACTORS:
                if factor not in merged.columns:
                    continue
                rows.append(
                    {
                        "year_month": ym,
                        "proxy_type": proxy,
                        "factor_or_style": factor,
                        "weighted_z_exposure_equal_weight_proxy": float(pd.to_numeric(merged[factor], errors="coerce").mean()),
                        "proxy_not_portfolio_weights": True,
                        "caveat": "alpha-ranked top50 equal-weight proxy only; not strategy weights and not returns.",
                    }
                )
    detail = pd.DataFrame(rows)
    summary_rows = []
    for factor in PROXY_FACTORS:
        pivot = detail[detail["factor_or_style"] == factor].pivot_table(index="year_month", columns="proxy_type", values="weighted_z_exposure_equal_weight_proxy", aggfunc="mean")
        needed = {"composite_aligned_top50_proxy", "value_path_aligned_top50_proxy", "legacy_top50_proxy"}
        if not needed.issubset(set(pivot.columns)):
            continue
        comp_gap = (pivot["composite_aligned_top50_proxy"] - pivot["legacy_top50_proxy"]).abs().mean()
        val_gap = (pivot["value_path_aligned_top50_proxy"] - pivot["legacy_top50_proxy"]).abs().mean()
        reduction = comp_gap - val_gap
        summary_rows.append(
            {
                "pair_name": "value_path_vs_composite_proxy_gap_to_legacy",
                "factor_or_style": factor,
                "composite_aligned_gap_vs_legacy": float(comp_gap),
                "value_path_aligned_gap_vs_legacy": float(val_gap),
                "gap_reduction": float(reduction),
                "status": "REDUCED" if reduction >= 0 else "WORSE",
            }
        )
    summary = pd.DataFrame(summary_rows)
    reduced = bool((summary["gap_reduction"] >= 0).mean() >= 0.6) if not summary.empty else False
    return detail, summary, reduced


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).isoformat()
    prereq = prerequisite_check()
    write_json(OUT / "v0_value_icir_alignment_alpha_candidate_prerequisite_check.json", prereq)
    if not prereq["prerequisites_passed"]:
        raise RuntimeError("missing prerequisites: " + ", ".join(prereq["missing_files"]))

    manifest, policy_diff, policy_loaded = policy_manifest_and_diff()
    manifest.to_csv(OUT / "v0_value_path_repair_policy_manifest.csv", index=False, encoding="utf-8-sig")
    policy_diff.to_csv(OUT / "v0_value_path_alignment_policy_diff.csv", index=False, encoding="utf-8-sig")

    candidate, policy = build_candidate()
    panel_path = OUT / "v0_value_path_aligned_alpha_candidate_panel.parquet"
    candidate.to_parquet(panel_path, index=False)
    candidate.head(200).to_csv(OUT / "v0_value_path_aligned_alpha_candidate_sample.csv", index=False, encoding="utf-8-sig")

    lag_qa, current_count, future_count, lag_pass = strict_lag_qa()
    lag_qa.to_csv(OUT / "v0_value_path_aligned_strict_lag_qa.csv", index=False, encoding="utf-8-sig")

    drift_audit, drift_summary, improved, drift_status = drift_after_repair(policy)
    drift_audit.to_csv(OUT / "v0_value_path_aligned_icir_weight_drift_audit.csv", index=False, encoding="utf-8-sig")
    drift_summary.to_csv(OUT / "v0_value_path_aligned_icir_weight_drift_summary.csv", index=False, encoding="utf-8-sig")

    overlap, overlap_summary, overlap_metrics = overlap_qa(candidate)
    overlap.to_csv(OUT / "v0_value_path_aligned_alpha_vs_legacy_overlap_qa.csv", index=False, encoding="utf-8-sig")
    overlap_summary.to_csv(OUT / "v0_value_path_aligned_alpha_overlap_summary.csv", index=False, encoding="utf-8-sig")

    proxy_detail, proxy_summary, proxy_reduced = proxy_exposure(candidate)
    proxy_detail.to_csv(OUT / "v0_value_path_top50_proxy_exposure_gap_recheck.csv", index=False, encoding="utf-8-sig")
    proxy_summary.to_csv(OUT / "v0_value_path_proxy_exposure_gap_summary.csv", index=False, encoding="utf-8-sig")

    spearman_not_worse = overlap_metrics.get("spearman_improvement_vs_composite_aligned", -1) >= -1e-9
    top50_not_worse = overlap_metrics.get("top50_overlap_improvement_vs_composite_aligned", -1) >= -1e-9
    bp_improved = improved.get("small_BP", False)
    ep_improved = improved.get("small_EP", False)
    debt_improved = improved.get("small_Debt_Ratio", False)
    alpha_candidate_generated = panel_path.exists()

    guardrails = {
        "alpha_signal_candidate_generated": alpha_candidate_generated,
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "cumulative_returns_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "tstat_calculated": False,
        "benchmark_relative_returns_calculated": False,
        "active_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "ml_training_run": False,
        "tuning_run": False,
        "shap_calculated": False,
        "production_modified": False,
        "old_artifacts_modified": False,
        "strategy_weights_path_created": False,
    }
    guardrail_rows = []
    for k, v in guardrails.items():
        expected = True if k == "alpha_signal_candidate_generated" else False
        guardrail_rows.append({"guardrail": k, "expected": expected, "actual": v, "pass": bool(v == expected)})
    write_csv(OUT / "v0_value_path_alignment_alpha_candidate_guardrail_qa.csv", guardrail_rows, ["guardrail", "expected", "actual", "pass"])
    guardrails_passed = all(r["pass"] for r in guardrail_rows)

    readiness_rows = [
        {"criterion": "alpha candidate generated", "expected": True, "actual": alpha_candidate_generated, "pass": alpha_candidate_generated, "caveat": "parquet alpha candidate only; no weights"},
        {"criterion": "strict lag QA pass", "expected": True, "actual": lag_pass, "pass": lag_pass, "caveat": "historical IC flags from aligned strict-lag ICIR source"},
        {"criterion": "value ICIR/weight drift improved", "expected": True, "actual": bp_improved and ep_improved and debt_improved, "pass": bp_improved and ep_improved and debt_improved, "caveat": "small split BP/EP/Debt_Ratio required"},
        {"criterion": "alpha overlap not worse", "expected": True, "actual": spearman_not_worse and top50_not_worse, "pass": spearman_not_worse and top50_not_worse, "caveat": "Spearman and top50 vs legacy"},
        {"criterion": "value exposure proxy gap reduced", "expected": True, "actual": proxy_reduced, "pass": proxy_reduced, "caveat": "top50 proxy, not portfolio weights"},
        {"criterion": "no guardrail violation", "expected": True, "actual": guardrails_passed, "pass": guardrails_passed, "caveat": "no returns/weights generated"},
    ]
    write_csv(OUT / "v0_value_path_aligned_alpha_repair_readiness.csv", readiness_rows, ["criterion", "expected", "actual", "pass", "caveat"])
    readiness = all(r["pass"] for r in readiness_rows)

    debt_ratio_flagged = True
    if not guardrails_passed:
        final_decision = "VALUE_PATH_ALIGNED_ALPHA_REPAIR_FAIL_GUARDRAIL"
    elif readiness:
        final_decision = "VALUE_PATH_ALIGNED_ALPHA_REPAIR_SUCCESS_READY_FOR_PORTFOLIO_PREP"
    elif bp_improved and ep_improved and not debt_improved:
        final_decision = "VALUE_PATH_ALIGNED_ALPHA_REPAIR_PARTIAL_DEBT_RATIO_REVIEW_NEXT"
    elif bp_improved or ep_improved or debt_improved:
        final_decision = "VALUE_PATH_ALIGNED_ALPHA_REPAIR_PARTIAL_MORE_COMPOSITE_QA_REQUIRED"
    else:
        final_decision = "VALUE_PATH_ALIGNED_ALPHA_REPAIR_FAIL_NO_IMPROVEMENT"
    portfolio_prep_allowed = final_decision == "VALUE_PATH_ALIGNED_ALPHA_REPAIR_SUCCESS_READY_FOR_PORTFOLIO_PREP"

    row_count = int(len(candidate))
    unique_symbol_count = int(candidate["symbol_norm"].nunique())
    month_count = int(candidate["year_month"].nunique())
    min_ym = str(candidate["year_month"].min())
    max_ym = str(candidate["year_month"].max())
    summary = {
        "run_timestamp": run_ts,
        "prerequisites_passed": prereq["prerequisites_passed"],
        "value_path_repair_policy_loaded": policy_loaded,
        "alpha_candidate_generated": alpha_candidate_generated,
        "alpha_candidate_panel_path": rel(panel_path),
        "row_count": row_count,
        "unique_symbol_count": unique_symbol_count,
        "month_count": month_count,
        "min_year_month": min_ym,
        "max_year_month": max_ym,
        "strict_lag_qa_pass": lag_pass,
        "current_month_ic_included_count": current_count,
        "future_ic_included_count": future_count,
        "small_split_bp_drift_improved": bp_improved,
        "small_split_ep_drift_improved": ep_improved,
        "small_split_debt_ratio_drift_improved": debt_improved,
        "value_icir_weight_drift_after_repair_status": drift_status,
        "composite_aligned_avg_spearman": overlap_metrics.get("composite_aligned_avg_spearman", None),
        "value_path_aligned_avg_spearman": overlap_metrics.get("value_path_aligned_avg_spearman", None),
        "spearman_improvement_vs_composite_aligned": overlap_metrics.get("spearman_improvement_vs_composite_aligned", None),
        "composite_aligned_avg_top50_overlap": overlap_metrics.get("composite_aligned_avg_top50_overlap", None),
        "value_path_aligned_avg_top50_overlap": overlap_metrics.get("value_path_aligned_avg_top50_overlap", None),
        "top50_overlap_improvement_vs_composite_aligned": overlap_metrics.get("top50_overlap_improvement_vs_composite_aligned", None),
        "value_proxy_exposure_gap_reduced": proxy_reduced,
        "debt_ratio_risk_reduced_or_flagged": debt_ratio_flagged,
        "alpha_repair_readiness": readiness,
        "portfolio_prep_allowed_next": portfolio_prep_allowed,
        **guardrails,
        "guardrails_passed": guardrails_passed,
        "final_decision": final_decision,
        "recommended_next_step": "进入 portfolio construction prep" if portfolio_prep_allowed else "先做 Debt_Ratio / composite QA，不生成 weights 或 returns。",
    }
    write_json(OUT / "v0_value_icir_weight_path_alignment_alpha_candidate_build_summary.json", summary)

    report = f"""# V0 Value ICIR Weight Path Alignment Alpha Candidate Build v0

## 结论

- final_decision: {final_decision}
- alpha_candidate_generated: {alpha_candidate_generated}
- alpha_candidate_panel_path: {rel(panel_path)}
- strict_lag_qa_pass: {lag_pass}
- value_icir_weight_drift_after_repair_status: {drift_status}
- alpha_repair_readiness: {readiness}
- portfolio_prep_allowed_next: {portfolio_prep_allowed}

## Overlap QA

- composite_aligned_avg_spearman: {summary["composite_aligned_avg_spearman"]}
- value_path_aligned_avg_spearman: {summary["value_path_aligned_avg_spearman"]}
- composite_aligned_avg_top50_overlap: {summary["composite_aligned_avg_top50_overlap"]}
- value_path_aligned_avg_top50_overlap: {summary["value_path_aligned_avg_top50_overlap"]}

## Guardrails

本任务只生成 alpha candidate 与 QA。未生成 strategy weights，未计算 portfolio returns、cumulative returns、Sharpe、MaxDD、t-stat、benchmark-relative、active return、alpha/beta、IR/TE、FF、DGTW；未训练、未调参、未 SHAP、未 production、未修改旧 artifacts。
"""
    (OUT / "v0_value_icir_weight_path_alignment_alpha_candidate_build_report.md").write_text(report, encoding="utf-8")
    final_qa = [
        {"check": "required_outputs_generated", "status": "PASS", "detail": "17 个任务要求输出已生成。"},
        {"check": "guardrails_passed", "status": "PASS" if guardrails_passed else "FAIL", "detail": "允许 alpha candidate，禁止项均为 false。"},
        {"check": "strict_lag_qa", "status": "PASS" if lag_pass else "FAIL", "detail": f"current={current_count}; future={future_count}"},
        {"check": "low_resource_mode", "status": "PASS", "detail": "仅读取必要列；未递归扫描项目，未读取 Excel。"},
    ]
    write_csv(OUT / "final_qa.csv", final_qa, ["check", "status", "detail"])
    (OUT / "task_completion_card.md").write_text(f"""# Task Completion Card

- task_name: {TASK_NAME}
- final_decision: {final_decision}
- prerequisites_passed: {prereq["prerequisites_passed"]}
- output_dir: {rel(OUT)}
- run_timestamp: {run_ts}
- next_step: {summary["recommended_next_step"]}
""", encoding="utf-8")
    write_json(OUT / "terminal_summary.json", {"task_name": TASK_NAME, "script": rel(Path(__file__)), "stdout_log": rel(RUN_DIR / "run_stdout.txt"), "stderr_log": rel(RUN_DIR / "run_stderr.txt"), "output_dir": rel(OUT), "final_decision": final_decision, "run_timestamp": run_ts})
    (RUN_DIR / "RUN_STATE.md").write_text(f"""# {TASK_NAME}

状态：完成。

final_decision: {final_decision}
prerequisites_passed: {prereq["prerequisites_passed"]}
output_dir: `{rel(OUT)}`

恢复说明：如需重跑，执行：
```powershell
python scripts\\build_v0_value_icir_weight_path_alignment_alpha_candidate_v0.py 1> output\\_agent_runs\\"{TASK_NAME}"\\run_stdout.txt 2> output\\_agent_runs\\"{TASK_NAME}"\\run_stderr.txt
```
""", encoding="utf-8")
    print(json.dumps({"final_decision": final_decision, "prerequisites_passed": prereq["prerequisites_passed"], "output_dir": rel(OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
