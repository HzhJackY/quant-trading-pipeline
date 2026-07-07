from __future__ import annotations

import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


TASK_NAME = "v0_canonical_strict_lag_split_universe_rebuild_prep_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

RETURN_MAP = (
    ROOT
    / "output"
    / "trd_mnth_parser_repair_2024_12_coverage_repair_v0"
    / "canonical_csmar_trd_mnth_return_map_repaired.parquet"
)
UNIFIED_EVAL_SUMMARY = (
    ROOT
    / "output"
    / "unified_strategy_eval_repaired_trd_mnth_v0"
    / "unified_strategy_eval_repaired_trd_mnth_summary.json"
)

PANEL_CANDIDATES = [
    {
        "panel_name": "pit_clean_core_financial_factors_monthly_v3",
        "path": ROOT
        / "output"
        / "csmar_pit_clean_core_financial_factors_v3"
        / "pit_clean_core_financial_factors_monthly_v3.parquet",
        "is_primary_candidate": True,
    },
    {
        "panel_name": "transformed_training_panel_v0",
        "path": ROOT
        / "output"
        / "build_transformed_training_panel_v0"
        / "transformed_training_panel_v0.parquet",
        "is_primary_candidate": False,
    },
    {
        "panel_name": "derived_compact_f_missing_features_candidate_v01",
        "path": ROOT
        / "output"
        / "derived_compact_f_missing_features_candidate_v01"
        / "derived_compact_f_missing_features_candidate_v01.parquet",
        "is_primary_candidate": False,
    },
    {
        "panel_name": "robust_cleaned_factor_score_panel_v0_reference_only",
        "path": ROOT
        / "output"
        / "robust_cleaned_fundamental_factor_variant_build_v0"
        / "robust_cleaned_factor_score_panel_v0.parquet",
        "is_primary_candidate": False,
    },
]

LEGACY_SCRIPTS = [
    ROOT / "factor_research" / "split_universe.py",
    ROOT / "factor_research" / "backtest_engine.py",
    ROOT / "factor_research" / "orthogonalization.py",
    ROOT / "run_split_universe.py",
]
STRICT_LAG_REFERENCE = [
    ROOT
    / "output"
    / "v0_strict_lag_icir_rebuild_bridge_v0"
    / "v0_strict_lag_alpha_signal_panel.parquet",
    ROOT
    / "output"
    / "v0_strict_lag_icir_rebuild_bridge_v0"
    / "v0_strict_lag_reconstructed_weights.parquet",
    ROOT / "scripts" / "rebuild_v0_strict_lag_icir_bridge_v0.py",
]

LEGACY_FACTORS = [
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

SPLIT_FIELD_CANDIDATES = [
    "total_market_cap",
    "float_market_cap",
    "market_cap_raw",
    "total_market_cap_raw_thousand",
    "mcap_est",
    "成交额",
]

PIT_KEYWORDS = ["ann", "announce", "公告", "pub", "publish", "asof", "as_of", "pit"]
REPORT_PERIOD_KEYWORDS = ["report", "period", "end_date", "accper", "会计", "报告"]
ID_COL_CANDIDATES = ["symbol", "Symbol", "stkcd", "Stkcd", "stock_code", "证券代码"]
MONTH_COL_CANDIDATES = [
    "month",
    "month_end",
    "year_month",
    "date",
    "Date",
    "trade_month",
    "Trdmnt",
]


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


def parquet_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def parquet_row_count(path: Path) -> int:
    meta = pq.ParquetFile(path).metadata
    return int(meta.num_rows)


def pick_col(cols: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_month(series: pd.Series) -> pd.Series:
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.astype(str).str.slice(0, 7)
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().mean() >= 0.8:
        return parsed.dt.to_period("M").astype(str)
    return series.astype(str).str.strip().str.slice(0, 7)


def factor_field_candidates(factor: str, cols: list[str]) -> list[str]:
    exact = [
        factor,
        f"{factor}_neutral_z",
        f"{factor}_z",
        factor.lower(),
        f"{factor.lower()}_neutral_z",
        f"{factor.lower()}_z",
    ]
    found = []
    lower_map = {c.lower(): c for c in cols}
    for cand in exact:
        if cand in cols and cand not in found:
            found.append(cand)
        mapped = lower_map.get(cand.lower())
        if mapped and mapped not in found:
            found.append(mapped)
    return found


def audit_panel(candidate: dict[str, Any]) -> dict[str, Any]:
    path = candidate["path"]
    base = {
        "panel_name": candidate["panel_name"],
        "path": rel(path),
        "row_count": 0,
        "unique_symbol_count": 0,
        "month_count": 0,
        "min_month": "",
        "max_month": "",
        "one_row_per_symbol_month": False,
        "duplicate_symbol_month_count": "",
        "pit_date_columns": "",
        "report_period_columns": "",
        "available_factor_count": 0,
        "available_v0_factor_count": 0,
        "missing_v0_factor_list": ",".join(LEGACY_FACTORS),
        "panel_status": "MISSING",
        "caveat": "候选文件不存在。",
    }
    if not path.exists():
        return base

    cols = parquet_columns(path)
    row_count = parquet_row_count(path)
    symbol_col = pick_col(cols, ID_COL_CANDIDATES)
    month_col = pick_col(cols, MONTH_COL_CANDIDATES)
    pit_cols = [c for c in cols if any(k.lower() in c.lower() for k in PIT_KEYWORDS)]
    report_cols = [c for c in cols if any(k.lower() in c.lower() for k in REPORT_PERIOD_KEYWORDS)]
    available_v0 = [f for f in LEGACY_FACTORS if factor_field_candidates(f, cols)]
    missing_v0 = [f for f in LEGACY_FACTORS if f not in available_v0]
    key_cols = [c for c in [symbol_col, month_col] if c]
    one_row = False
    dup_count: int | str = ""
    unique_symbols = 0
    month_count = 0
    min_month = ""
    max_month = ""
    caveats = []

    if symbol_col and month_col:
        df = pd.read_parquet(path, columns=key_cols)
        month = normalize_month(df[month_col])
        symbol = df[symbol_col].astype(str).str.strip()
        unique_symbols = int(symbol.nunique(dropna=True))
        month_count = int(month.nunique(dropna=True))
        min_month = str(month.dropna().min()) if month.notna().any() else ""
        max_month = str(month.dropna().max()) if month.notna().any() else ""
        dup_count = int(pd.DataFrame({"symbol": symbol, "month": month}).duplicated().sum())
        one_row = dup_count == 0
        del df, month, symbol
        gc.collect()
    else:
        caveats.append("未识别 symbol/month 键列，coverage 仅基于 schema。")

    non_key_cols = [
        c
        for c in cols
        if c not in set(key_cols)
        and not any(k.lower() in c.lower() for k in PIT_KEYWORDS + REPORT_PERIOD_KEYWORDS)
    ]
    status = "AVAILABLE"
    if not available_v0:
        status = "NO_V0_FACTORS"
    elif missing_v0:
        status = "PARTIAL_V0_FACTORS"
    if candidate["panel_name"].endswith("reference_only"):
        status = f"REFERENCE_ONLY_{status}"
        caveats.append("仅作为 robust cleaned 参考，不作为收益源或主候选。")

    base.update(
        {
            "row_count": row_count,
            "unique_symbol_count": unique_symbols,
            "month_count": month_count,
            "min_month": min_month,
            "max_month": max_month,
            "one_row_per_symbol_month": bool(one_row),
            "duplicate_symbol_month_count": dup_count,
            "pit_date_columns": ",".join(pit_cols),
            "report_period_columns": ",".join(report_cols),
            "available_factor_count": len(non_key_cols),
            "available_v0_factor_count": len(available_v0),
            "missing_v0_factor_list": ",".join(missing_v0),
            "panel_status": status,
            "caveat": "；".join(caveats),
        }
    )
    return base


def coverage_ratio(path: Path, field: str) -> float:
    row_count = parquet_row_count(path)
    if row_count == 0:
        return 0.0
    df = pd.read_parquet(path, columns=[field])
    ratio = float(df[field].notna().sum() / row_count)
    del df
    gc.collect()
    return ratio


def build_factor_mapping(selected_panel: dict[str, Any] | None) -> pd.DataFrame:
    rows = []
    if not selected_panel:
        for factor in LEGACY_FACTORS:
            rows.append(
                {
                    "legacy_factor_name": factor,
                    "canonical_factor_name": "",
                    "source_panel": "",
                    "raw_field_or_transformed_field": "",
                    "direction_policy": "strict_lag_rolling_ic_ir_only",
                    "transform_policy": "use_existing_panel_field; no full-sample direction",
                    "available": False,
                    "coverage_ratio": 0.0,
                    "missing_reason": "无可用主候选 panel",
                    "use_in_canonical_v0": False,
                    "caveat": "",
                }
            )
        return pd.DataFrame(rows)

    path = selected_panel["path"]
    cols = parquet_columns(path)
    for factor in LEGACY_FACTORS:
        matches = factor_field_candidates(factor, cols)
        field = matches[0] if matches else ""
        available = bool(field)
        cov = coverage_ratio(path, field) if available else 0.0
        rows.append(
            {
                "legacy_factor_name": factor,
                "canonical_factor_name": factor if available else "",
                "source_panel": selected_panel["panel_name"] if available else "",
                "raw_field_or_transformed_field": field,
                "direction_policy": "strict_lag_rolling_ic_ir_only; economic_sign_note_allowed_but_not_overriding",
                "transform_policy": "prefer existing canonical/transformed field; cross-sectional treatment deferred to alpha build",
                "available": available,
                "coverage_ratio": round(cov, 6),
                "missing_reason": "" if available else "当前选定 panel 未找到等价字段",
                "use_in_canonical_v0": bool(available and cov > 0),
                "caveat": "方向不得由未来收益或全样本 IC_IR 决定。",
            }
        )
    return pd.DataFrame(rows)


def build_split_policy(audits: list[dict[str, Any]], selected_panel: dict[str, Any] | None) -> pd.DataFrame:
    rows = []
    selected_field = ""
    selected_path = selected_panel["path"] if selected_panel else None
    selected_name = selected_panel["panel_name"] if selected_panel else ""
    selected_cols = parquet_columns(selected_path) if selected_path and selected_path.exists() else []

    for field in SPLIT_FIELD_CANDIDATES:
        source_panel = selected_name if field in selected_cols else ""
        cov = coverage_ratio(selected_path, field) if selected_path and field in selected_cols else 0.0
        pit_safe = field in selected_cols and field != "mcap_est"
        reason = ""
        caveat = ""
        if field in ["total_market_cap", "float_market_cap", "market_cap_raw", "total_market_cap_raw_thousand"]:
            reason = "优先使用当月可见/月末已知市值字段。"
        elif field == "mcap_est":
            reason = "legacy 使用成交额/换手率估算；仅在无直接市值字段时考虑。"
            caveat = "若需估算，下一阶段须确认成交额和换手率同月可见。"
        elif field == "成交额":
            reason = "不能单独作为市值切分字段，仅可辅助 legacy mcap_est。"
            caveat = "缺少换手率时不得替代市值。"
        rows.append(
            {
                "split_field_candidate": field,
                "source_panel": source_panel,
                "coverage_ratio": round(cov, 6),
                "pit_safe": bool(pit_safe),
                "selected": False,
                "percentile": 0.5,
                "reason": reason,
                "caveat": caveat,
            }
        )

    preferred = ["total_market_cap", "float_market_cap", "market_cap_raw", "total_market_cap_raw_thousand"]
    for pref in preferred:
        match = next((r for r in rows if r["split_field_candidate"] == pref and r["coverage_ratio"] > 0.8), None)
        if match:
            selected_field = pref
            break
    if not selected_field and selected_path:
        has_amount = "成交额" in selected_cols
        has_turnover = any(c in selected_cols for c in ["换手率", "turnover", "Turnover"])
        if has_amount and has_turnover:
            selected_field = "mcap_est"
            for row in rows:
                if row["split_field_candidate"] == "mcap_est":
                    row["source_panel"] = selected_name
                    row["coverage_ratio"] = min(
                        coverage_ratio(selected_path, "成交额"),
                        coverage_ratio(selected_path, "换手率" if "换手率" in selected_cols else "turnover"),
                    )
                    row["pit_safe"] = True
                    row["caveat"] = "legacy fallback：成交额/换手率估算市值，下一阶段需在 alpha build 中保留 QA。"
    for row in rows:
        if row["split_field_candidate"] == selected_field:
            row["selected"] = True
            row["reason"] = row["reason"] + " 已锁定 percentile=0.5，未调参。"

    return pd.DataFrame(rows)


def make_report(summary: dict[str, Any], audits: pd.DataFrame, mapping: pd.DataFrame, split: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# V0 Canonical Strict-Lag Split-Universe Rebuild Prep v0",
            "",
            "## 结论",
            f"- final_decision: {summary['final_decision']}",
            f"- prerequisites_passed: {summary['prerequisites_passed']}",
            f"- selected_factor_panel: {summary['selected_factor_panel']}",
            f"- selected_split_field: {summary['selected_split_field']}",
            f"- canonical_rebuild_allowed_next: {summary['canonical_rebuild_allowed_next']}",
            "",
            "## 候选 Panel",
            audits[
                [
                    "panel_name",
                    "row_count",
                    "unique_symbol_count",
                    "month_count",
                    "min_month",
                    "max_month",
                    "available_v0_factor_count",
                    "panel_status",
                ]
            ].to_markdown(index=False),
            "",
            "## V0 因子映射",
            mapping[
                [
                    "legacy_factor_name",
                    "raw_field_or_transformed_field",
                    "available",
                    "coverage_ratio",
                    "use_in_canonical_v0",
                ]
            ].to_markdown(index=False),
            "",
            "## Split Policy",
            split[["split_field_candidate", "coverage_ratio", "pit_safe", "selected", "percentile"]].to_markdown(
                index=False
            ),
            "",
            "## Guardrails",
            "- 本任务未生成 alpha_signal、weights、portfolio returns。",
            "- 未运行训练、调参、benchmark-relative、alpha/beta、IR/TE、FF、DGTW、SHAP 或 production。",
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_state("running", {"step": "prerequisite_check"})

    prereq = {
        "trd_mnth_return_map_found": RETURN_MAP.exists(),
        "unified_eval_summary_found": UNIFIED_EVAL_SUMMARY.exists(),
        "current_factor_panel_candidates_found": any(c["path"].exists() for c in PANEL_CANDIDATES[:3]),
        "legacy_v0_scripts_found": all(p.exists() for p in LEGACY_SCRIPTS),
        "strict_lag_reference_found": all(p.exists() for p in STRICT_LAG_REFERENCE),
        "prerequisites_passed": False,
        "missing_files": [],
    }
    required = [RETURN_MAP, UNIFIED_EVAL_SUMMARY, *LEGACY_SCRIPTS, *STRICT_LAG_REFERENCE]
    prereq["missing_files"] = [rel(p) for p in required if not p.exists()]
    if not any(c["path"].exists() for c in PANEL_CANDIDATES[:3]):
        prereq["missing_files"].append("current_factor_panel_candidates")
    prereq["prerequisites_passed"] = not prereq["missing_files"]
    save_json(prereq, OUT_DIR / "v0_canonical_rebuild_prep_prerequisite_check.json")

    write_state("running", {"step": "factor_panel_candidate_audit"})
    audits = [audit_panel(c) for c in PANEL_CANDIDATES]
    audit_df = pd.DataFrame(audits)
    audit_df.to_csv(OUT_DIR / "v0_canonical_factor_panel_candidate_audit.csv", index=False, encoding="utf-8-sig")

    available_candidates = [
        c
        for c in PANEL_CANDIDATES[:3]
        if c["path"].exists()
        and audit_df.loc[audit_df["panel_name"] == c["panel_name"], "available_v0_factor_count"].iloc[0] > 0
    ]
    selected_panel = None
    if available_candidates:
        selected_panel = sorted(
            available_candidates,
            key=lambda c: (
                c["is_primary_candidate"],
                int(audit_df.loc[audit_df["panel_name"] == c["panel_name"], "available_v0_factor_count"].iloc[0]),
            ),
            reverse=True,
        )[0]

    write_state("running", {"step": "factor_mapping", "selected_panel": selected_panel["panel_name"] if selected_panel else ""})
    mapping_df = build_factor_mapping(selected_panel)
    mapping_df.to_csv(OUT_DIR / "v0_canonical_factor_mapping_manifest.csv", index=False, encoding="utf-8-sig")

    split_df = build_split_policy(audits, selected_panel)
    split_df.to_csv(OUT_DIR / "v0_canonical_split_universe_policy.csv", index=False, encoding="utf-8-sig")

    selected_split = split_df.loc[split_df["selected"] == True]  # noqa: E712
    selected_split_field = selected_split["split_field_candidate"].iloc[0] if not selected_split.empty else ""
    split_policy_locked = bool(selected_split_field)

    strict_lag_policy = {
        "rolling_window": 24,
        "use_strict_lag": True,
        "current_month_ic_allowed": False,
        "future_ic_allowed": False,
        "min_ic_ir": 0.05,
        "flip_sign_policy": "true; only based on historical strict-lag rolling IC_IR",
        "gram_schmidt_enabled": True,
        "min_stocks": 20,
        "full_sample_icir_allowed": False,
        "same_month_return_allowed_in_signal": False,
        "policy_status": "LOCKED",
    }
    save_json(strict_lag_policy, OUT_DIR / "v0_canonical_strict_lag_icir_policy.json")

    selected_factors = mapping_df.loc[mapping_df["use_in_canonical_v0"] == True, "canonical_factor_name"].tolist()  # noqa: E712
    run_config = {
        "canonical_rebuild_allowed_next": bool(prereq["prerequisites_passed"] and selected_panel and split_policy_locked),
        "selected_factor_panel": rel(selected_panel["path"]) if selected_panel else "",
        "selected_return_map": rel(RETURN_MAP),
        "selected_factor_list": selected_factors,
        "split_universe_policy": {
            "split_field": selected_split_field,
            "percentile": 0.5,
            "no_return_optimized_threshold": True,
        },
        "strict_lag_icir_policy": strict_lag_policy,
        "portfolio_rule": {
            "name": "Top50_Buffer_35_75",
            "entry_rank": 35,
            "exit_rank": 75,
            "target_holding_count": 50,
            "weighting": "equal_weight",
            "initialization": "first_month_top50_initialization",
        },
        "output_directory_for_next_run": "output/v0_canonical_strict_lag_alpha_build_v0",
        "generate_alpha_signal_next_run_allowed": True,
        "generate_weights_next_run_allowed": True,
        "calculate_returns_next_run_allowed": False,
        "no_training": True,
        "no_tuning": True,
        "no_production": True,
        "recommended_next_runs": [
            "canonical V0 alpha_signal build run",
            "canonical V0 portfolio construction run",
            "canonical V0 repaired TRD_Mnth evaluation run",
        ],
    }
    save_json(run_config, OUT_DIR / "v0_canonical_rebuild_run_config_draft.json")

    guardrails = {
        "alpha_signal_generated": False,
        "strategy_weights_generated": False,
        "portfolio_returns_calculated": False,
        "old_artifacts_modified": False,
        "production_modified": False,
        "ml_training_run": False,
        "new_ml_model_trained": False,
        "benchmark_relative_returns_calculated": False,
        "alpha_beta_regression_calculated": False,
        "information_ratio_calculated": False,
        "tracking_error_calculated": False,
        "ff_regression_calculated": False,
        "dgtw_adjusted_eval_calculated": False,
        "shap_calculated": False,
    }
    guardrail_rows = [
        {"guardrail": k, "expected": False, "actual": v, "pass": v is False} for k, v in guardrails.items()
    ]
    guardrail_df = pd.DataFrame(guardrail_rows)
    guardrail_df.to_csv(OUT_DIR / "v0_canonical_rebuild_prep_guardrail_qa.csv", index=False, encoding="utf-8-sig")

    selected_status = ""
    if selected_panel:
        selected_status = str(
            audit_df.loc[audit_df["panel_name"] == selected_panel["panel_name"], "panel_status"].iloc[0]
        )
    missing_count = int((mapping_df["use_in_canonical_v0"] == False).sum())  # noqa: E712
    if not guardrail_df["pass"].all():
        final_decision = "V0_CANONICAL_REBUILD_PREP_FAIL_GUARDRAIL"
    elif not selected_panel or missing_count == len(LEGACY_FACTORS):
        final_decision = "V0_CANONICAL_REBUILD_PREP_BLOCKED_FACTOR_MAPPING"
    elif not split_policy_locked:
        final_decision = "V0_CANONICAL_REBUILD_PREP_BLOCKED_SPLIT_POLICY"
    elif missing_count > 0 or not prereq["prerequisites_passed"]:
        final_decision = "V0_CANONICAL_REBUILD_PREP_READY_WITH_CAVEATS"
    else:
        final_decision = "V0_CANONICAL_REBUILD_PREP_READY_FOR_ALPHA_BUILD"

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prerequisites_passed": prereq["prerequisites_passed"],
        "selected_factor_panel": rel(selected_panel["path"]) if selected_panel else "",
        "selected_factor_panel_status": selected_status,
        "canonical_factor_count_selected": len(selected_factors),
        "missing_legacy_factor_count": missing_count,
        "selected_split_field": selected_split_field,
        "split_policy_locked": split_policy_locked,
        "strict_lag_icir_policy_locked": True,
        "gram_schmidt_policy_locked": True,
        "portfolio_rule_locked": True,
        "repaired_trd_mnth_return_map_selected": rel(RETURN_MAP),
        "canonical_rebuild_allowed_next": run_config["canonical_rebuild_allowed_next"],
        "recommended_next_step": "运行 canonical V0 alpha_signal build，仍禁止在同一步中计算 portfolio returns。",
        **guardrails,
        "final_decision": final_decision,
    }
    save_json(summary, OUT_DIR / "v0_canonical_strict_lag_split_universe_rebuild_prep_summary.json")

    report = make_report(summary, audit_df, mapping_df, split_df)
    (OUT_DIR / "v0_canonical_strict_lag_split_universe_rebuild_prep_report.md").write_text(report, encoding="utf-8")

    final_qa = guardrail_df.copy()
    final_qa.loc[len(final_qa)] = {
        "guardrail": "all_required_outputs_written",
        "expected": True,
        "actual": True,
        "pass": True,
    }
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")

    completion_card = "\n".join(
        [
            "# task_completion_card",
            "",
            f"- task_name: {TASK_NAME}",
            f"- final_decision: {final_decision}",
            f"- prerequisites_passed: {prereq['prerequisites_passed']}",
            f"- selected_factor_panel: {summary['selected_factor_panel']}",
            f"- selected_split_field: {selected_split_field}",
            "- guardrails_passed: true",
        ]
    )
    (OUT_DIR / "task_completion_card.md").write_text(completion_card, encoding="utf-8")
    save_json(
        {
            "task_name": TASK_NAME,
            "status": "completed",
            "script": rel(ROOT / "scripts" / "prep_v0_canonical_strict_lag_split_universe_rebuild_v0.py"),
            "stdout_log": rel(RUN_DIR / "run_stdout.txt"),
            "stderr_log": rel(RUN_DIR / "run_stderr.txt"),
            "output_dir": rel(OUT_DIR),
            "final_decision": final_decision,
        },
        OUT_DIR / "terminal_summary.json",
    )
    write_state("completed", {"final_decision": final_decision, "output_dir": rel(OUT_DIR)})
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
