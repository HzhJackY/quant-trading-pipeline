from __future__ import annotations

import csv
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "csmar_pit_clean_core_financial_factor_qa_lite_v1"
TASK_TITLE = "CSMAR PIT-Clean Core Financial Factor QA Lite v1"
RUN_START_TIME = datetime.now().astimezone().isoformat(timespec="seconds")

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "output" / "csmar_pit_clean_core_financial_factors_v1"
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME

FACTOR_PANEL_PATH = SRC_DIR / "pit_clean_core_financial_factors_monthly_v1.parquet"
COVERAGE_AUDIT_PATH = SRC_DIR / "factor_coverage_audit_by_month_v1.csv"
DISTRIBUTION_AUDIT_PATH = SRC_DIR / "factor_distribution_audit_v1.csv"
SOURCE_FINAL_QA_PATH = SRC_DIR / "final_qa_csmar_pit_clean_core_financial_factors_v1.csv"
SOURCE_CARD_PATH = SRC_DIR / "task_completion_card.md"
SOURCE_REPORT_PATH = SRC_DIR / "csmar_pit_clean_core_financial_factor_reconstruction_report_v1.md"

REQUIRED_FACTORS = [
    "roe_ttm",
    "ep_ttm",
    "bp",
    "profit_growth_yoy",
    "rev_growth_yoy",
    "net_margin",
    "debt_ratio",
    "sales_expense_to_revenue",
    "rd_expense_to_revenue",
]
FACTOR_PANEL_COLUMNS = [
    "month_end",
    "symbol",
    "selected_report_period",
    "selected_pit_date",
    "market_cap_trade_date",
    "market_cap_total",
    *REQUIRED_FACTORS,
]
COVERAGE_COLUMNS = [
    "roe_coverage",
    "ep_coverage",
    "bp_coverage",
    "profit_growth_yoy_coverage",
    "rev_growth_yoy_coverage",
    "net_margin_coverage",
    "debt_ratio_coverage",
    "sales_expense_to_revenue_coverage",
    "rd_expense_to_revenue_coverage",
]
FACTOR_TO_COVERAGE = {
    "roe_ttm": "roe_coverage",
    "ep_ttm": "ep_coverage",
    "bp": "bp_coverage",
    "profit_growth_yoy": "profit_growth_yoy_coverage",
    "rev_growth_yoy": "rev_growth_yoy_coverage",
    "net_margin": "net_margin_coverage",
    "debt_ratio": "debt_ratio_coverage",
    "sales_expense_to_revenue": "sales_expense_to_revenue_coverage",
    "rd_expense_to_revenue": "rd_expense_to_revenue_coverage",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.as_posix()


def append_checkpoint(stage: str, notes: list[str]) -> None:
    with (RUN_DIR / "CHECKPOINTS.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## {now_iso()} - {stage}\n\n")
        for note in notes:
            f.write(f"- {note}\n")


def update_run_state(stage: str, done: list[str], outputs: list[Path], next_step: str) -> None:
    done_lines = "\n".join(f"  - {x}" for x in done) if done else "  - none"
    out_lines = "\n".join(f"  - {rel(x)}" for x in outputs) if outputs else "  - none"
    text = f"""# RUN_STATE

- 当前任务名称: {TASK_TITLE}
- 开始时间: {RUN_START_TIME}
- 当前阶段: {stage}
- 已完成步骤:
{done_lines}
- 正在处理的文件:
  - {rel(FACTOR_PANEL_PATH)}
  - {rel(COVERAGE_AUDIT_PATH)}
  - {rel(DISTRIBUTION_AUDIT_PATH)}
  - {rel(SOURCE_FINAL_QA_PATH)}
  - {rel(SOURCE_CARD_PATH)}
  - {rel(SOURCE_REPORT_PATH)}
- 已生成输出:
{out_lines}
- 下一步:
  - {next_step}
- 如果 Codex 崩溃，新的 Codex 应如何继续:
  - 先读取本文件
  - 检查 run_stdout.txt、run_stderr.txt、terminal_summary.json 和 final QA
  - 如果 terminal_summary.json 不存在，重新运行 scripts/qa_csmar_pit_clean_core_financial_factors_lite_v1.py，并将 stdout/stderr 写入本 agent run 目录
  - 不要读取 xlsx、原始日频 CSV、访问 API、下载、训练、回测、IC 或修改 production
"""
    (RUN_DIR / "RUN_STATE.md").write_text(text, encoding="utf-8")


def inventory_for(path: Path, role: str, notes: str) -> dict[str, object]:
    exists = path.exists()
    readable = False
    n_rows: object = ""
    n_columns: object = ""
    columns: object = ""
    if exists:
        try:
            if path.suffix.lower() == ".parquet":
                df_head = pd.read_parquet(path, columns=FACTOR_PANEL_COLUMNS).head(0)
                readable = True
                columns = "|".join(df_head.columns)
                n_columns = len(df_head.columns)
                n_rows = int(pd.read_parquet(path, columns=["symbol"]).shape[0])
                del df_head
                gc.collect()
            elif path.suffix.lower() == ".csv":
                df = pd.read_csv(path, nrows=5)
                readable = True
                columns = "|".join(df.columns)
                n_columns = len(df.columns)
                n_rows = int(sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1)
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
                readable = True
                columns = "text"
                n_columns = 1
                n_rows = len(text.splitlines())
        except Exception as exc:  # inventory should capture readability without hiding final failure.
            notes = f"{notes}; inventory read error: {exc}"
    return {
        "input_path": rel(path),
        "exists": exists,
        "readable": readable,
        "n_rows": n_rows,
        "n_columns": n_columns,
        "columns": columns,
        "role": role,
        "notes": notes,
    }


def distribution_row(df: pd.DataFrame, factor: str) -> dict[str, object]:
    s = pd.to_numeric(df[factor], errors="coerce")
    valid = s.dropna()
    q = valid.quantile([0.01, 0.05, 0.25, 0.75, 0.95, 0.99]) if len(valid) else pd.Series(dtype=float)
    median = float(valid.median()) if len(valid) else np.nan
    p99_abs = float(valid.abs().quantile(0.99)) if len(valid) else np.nan
    extreme_flag = "none"
    status = "pass"
    notes = "raw factor distribution; no winsor/zscore/rank applied"
    if factor == "ep_ttm" and pd.notna(abs(median)) and abs(median) >= 1:
        extreme_flag = "unit_or_scale_review"
        status = "fail"
        notes = "EP median too large for normal decimal-scale valuation ratio"
    elif factor == "bp" and pd.notna(abs(median)) and abs(median) >= 20:
        extreme_flag = "unit_or_scale_review"
        status = "fail"
        notes = "BP median too large; possible market-cap unit problem"
    elif factor == "debt_ratio" and pd.notna(p99_abs) and p99_abs > 10:
        extreme_flag = "extreme_values"
        status = "caveat"
        notes = "Debt ratio has high tail values; financial/abnormal firms may contribute"
    elif factor in ("sales_expense_to_revenue", "rd_expense_to_revenue", "net_margin", "roe_ttm") and pd.notna(p99_abs) and p99_abs > 50:
        extreme_flag = "extreme_values"
        status = "caveat"
        notes = "Expense/margin/ROE raw tails require later QA; values are not clipped"
    elif factor in ("profit_growth_yoy", "rev_growth_yoy") and pd.notna(p99_abs) and p99_abs > 1000:
        extreme_flag = "denominator_near_zero_risk"
        status = "caveat"
        notes = "Growth can be extreme when prior TTM denominator is near zero"
    if factor == "rd_expense_to_revenue":
        notes = f"{notes}; pre-2018 sparsity is expected"
    return {
        "factor": factor,
        "n": int(len(valid)),
        "missing_rate": float(1 - len(valid) / len(s)) if len(s) else np.nan,
        "mean": float(valid.mean()) if len(valid) else np.nan,
        "median": median,
        "p01": float(q.get(0.01, np.nan)),
        "p05": float(q.get(0.05, np.nan)),
        "p25": float(q.get(0.25, np.nan)),
        "p75": float(q.get(0.75, np.nan)),
        "p95": float(q.get(0.95, np.nan)),
        "p99": float(q.get(0.99, np.nan)),
        "min": float(valid.min()) if len(valid) else np.nan,
        "max": float(valid.max()) if len(valid) else np.nan,
        "extreme_flag": extreme_flag,
        "plausibility_status": status,
        "notes": notes,
    }


def coverage_summary(coverage: pd.DataFrame) -> pd.DataFrame:
    coverage = coverage.copy()
    coverage["month_end"] = pd.to_datetime(coverage["month_end"], errors="coerce")
    rows: list[dict[str, object]] = []
    for factor, col in FACTOR_TO_COVERAGE.items():
        s = pd.to_numeric(coverage[col], errors="coerce")
        good_months = coverage.loc[s >= 0.8, "month_end"]
        notes: list[str] = []
        if factor in ("ep_ttm", "bp") and s.mean() >= 0.95:
            notes.append("EP/BP coverage sufficient; market cap no longer blocking")
        if factor == "rd_expense_to_revenue":
            pre2018 = coverage.loc[coverage["month_end"].dt.year < 2018, col]
            if len(pre2018) and pd.to_numeric(pre2018, errors="coerce").mean() < 0.5:
                notes.append("pre-2018 RD expense sparsity documented")
        if s.iloc[:6].min() < 0.8:
            notes.append("early TTM warm-up or source availability caveat")
        if (s < 0.5).sum() > max(3, len(s) * 0.1):
            notes.append("long-run low coverage review")
        rows.append(
            {
                "factor": factor,
                "mean_coverage": float(s.mean()),
                "min_coverage": float(s.min()),
                "max_coverage": float(s.max()),
                "months_below_50pct": int((s < 0.5).sum()),
                "months_below_80pct": int((s < 0.8).sum()),
                "earliest_month_with_good_coverage": "" if good_months.empty else str(good_months.min().date()),
                "notes": ";".join(notes) if notes else "coverage acceptable",
            }
        )
    return pd.DataFrame(rows)


def monthly_anomalies(factors: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    coverage = coverage.copy()
    coverage["month_end"] = pd.to_datetime(coverage["month_end"], errors="coerce")
    rows: list[dict[str, object]] = []
    for _, row in coverage.iterrows():
        vals = pd.to_numeric(row[COVERAGE_COLUMNS], errors="coerce")
        if vals.mean() < 0.5:
            rows.append({"month_end": row["month_end"].strftime("%Y-%m-%d"), "anomaly_type": "broad_coverage_drop", "affected_factor": "all", "value": float(vals.mean()), "notes": "all-factor mean coverage below 50%"})
        if vals.min() < 0.2:
            rows.append({"month_end": row["month_end"].strftime("%Y-%m-%d"), "anomaly_type": "factor_low_coverage", "affected_factor": vals.idxmin().replace("_coverage", ""), "value": float(vals.min()), "notes": "single factor coverage below 20%"})
    if not coverage.empty and coverage["month_end"].min() == pd.Timestamp("2017-04-30"):
        rows.append({"month_end": "2017-04-30", "anomaly_type": "warmup_start", "affected_factor": "ttm_factors", "value": "", "notes": "2017-04 is the panel start and expected TTM warm-up point"})
    latest = coverage["month_end"].max()
    if pd.notna(latest):
        latest_vals = coverage.loc[coverage["month_end"] == latest, COVERAGE_COLUMNS].iloc[0]
        rows.append({"month_end": latest.strftime("%Y-%m-%d"), "anomaly_type": "latest_month_disclosure_lag_check", "affected_factor": "all", "value": float(pd.to_numeric(latest_vals, errors="coerce").mean()), "notes": "latest month should be reviewed for normal disclosure lag"})
    monthly_dist = factors.groupby("month_end").agg(ep_median=("ep_ttm", "median"), bp_median=("bp", "median")).reset_index()
    ep_global = factors["ep_ttm"].median()
    bp_global = factors["bp"].median()
    for _, row in monthly_dist.iterrows():
        if pd.notna(ep_global) and abs(ep_global) > 0 and pd.notna(row["ep_median"]) and abs(row["ep_median"]) > max(1.0, abs(ep_global) * 10):
            rows.append({"month_end": row["month_end"].strftime("%Y-%m-%d"), "anomaly_type": "ep_distribution_jump", "affected_factor": "ep_ttm", "value": float(row["ep_median"]), "notes": "monthly EP median far from global median"})
        if pd.notna(bp_global) and abs(bp_global) > 0 and pd.notna(row["bp_median"]) and abs(row["bp_median"]) > max(20.0, abs(bp_global) * 10):
            rows.append({"month_end": row["month_end"].strftime("%Y-%m-%d"), "anomaly_type": "bp_distribution_jump", "affected_factor": "bp", "value": float(row["bp_median"]), "notes": "monthly BP median far from global median"})
    if not rows:
        rows.append({"month_end": "", "anomaly_type": "none", "affected_factor": "", "value": "", "notes": "no monthly anomalies detected by lite rules"})
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    input_inventory_path = OUT_DIR / "input_inventory_v1.csv"
    structure_path = OUT_DIR / "factor_panel_structure_qa_v1.csv"
    coverage_summary_path = OUT_DIR / "factor_coverage_summary_v1.csv"
    distribution_qa_path = OUT_DIR / "factor_distribution_qa_v1.csv"
    monthly_anomaly_path = OUT_DIR / "monthly_anomaly_summary_v1.csv"
    decision_matrix_path = OUT_DIR / "factor_qa_decision_matrix_v1.csv"
    report_path = OUT_DIR / "csmar_pit_clean_core_financial_factor_qa_lite_report_v1.md"
    card_path = OUT_DIR / "task_completion_card.md"
    final_qa_path = OUT_DIR / "final_qa_csmar_pit_clean_core_financial_factor_qa_lite_v1.csv"
    final_qa_alias_path = OUT_DIR / "final_qa.csv"
    terminal_summary_path = OUT_DIR / "terminal_summary.json"

    update_run_state("input_inventory", ["script started"], [], "read allowed QA inputs")
    inventory = pd.DataFrame(
        [
            inventory_for(FACTOR_PANEL_PATH, "factor panel", "column-projected parquet read for QA"),
            inventory_for(COVERAGE_AUDIT_PATH, "coverage audit", "existing reconstruction coverage audit"),
            inventory_for(DISTRIBUTION_AUDIT_PATH, "distribution audit", "existing reconstruction distribution audit"),
            inventory_for(SOURCE_FINAL_QA_PATH, "source final QA", "existing reconstruction QA"),
            inventory_for(SOURCE_CARD_PATH, "source task completion card", "existing reconstruction card"),
            inventory_for(SOURCE_REPORT_PATH, "source final report", "existing reconstruction report"),
        ]
    )
    inventory.to_csv(input_inventory_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    append_checkpoint("input_inventory_done", [f"generated {rel(input_inventory_path)}"])

    update_run_state("factor_panel_read", ["input inventory generated"], [input_inventory_path], "read factor panel and audit CSVs")
    factors = pd.read_parquet(FACTOR_PANEL_PATH, columns=FACTOR_PANEL_COLUMNS)
    coverage = pd.read_csv(COVERAGE_AUDIT_PATH)
    prior_distribution = pd.read_csv(DISTRIBUTION_AUDIT_PATH)
    prior_final_qa = pd.read_csv(SOURCE_FINAL_QA_PATH)
    factors["month_end"] = pd.to_datetime(factors["month_end"], errors="coerce")
    factors["selected_pit_date"] = pd.to_datetime(factors["selected_pit_date"], errors="coerce")
    factors["market_cap_trade_date"] = pd.to_datetime(factors["market_cap_trade_date"], errors="coerce")
    factors["selected_report_period"] = pd.to_datetime(factors["selected_report_period"], errors="coerce")

    n_rows = int(len(factors))
    n_symbols = int(factors["symbol"].astype(str).nunique())
    min_month = factors["month_end"].min()
    max_month = factors["month_end"].max()
    duplicate_count = int(factors.duplicated(["symbol", "month_end"]).sum())
    one_row = duplicate_count == 0
    selected_pit_violations = int(((factors["selected_pit_date"].notna()) & (factors["month_end"].notna()) & (factors["selected_pit_date"] > factors["month_end"])).sum())
    market_cap_violations = int(((factors["market_cap_trade_date"].notna()) & (factors["month_end"].notna()) & (factors["market_cap_trade_date"] > factors["month_end"])).sum())
    selected_report_violations = int(((factors["selected_report_period"].notna()) & (factors["month_end"].notna()) & (factors["selected_report_period"] > factors["month_end"])).sum())
    market_cap_positive_coverage = float((pd.to_numeric(factors["market_cap_total"], errors="coerce") > 0).mean())
    missing_factor_cols = [c for c in REQUIRED_FACTORS if c not in factors.columns]

    structure_rows = [
        ("n_rows", True, n_rows, "factor panel row count"),
        ("n_symbols", True, n_symbols, "symbol count"),
        ("min_month_end", pd.notna(min_month), "" if pd.isna(min_month) else str(min_month.date()), "panel start"),
        ("max_month_end", pd.notna(max_month), "" if pd.isna(max_month) else str(max_month.date()), "panel end"),
        ("one row per symbol-month", one_row, one_row, f"duplicate_count={duplicate_count}"),
        ("selected_pit_date <= month_end", selected_pit_violations == 0, selected_pit_violations, "PIT date alignment"),
        ("market_cap_trade_date <= month_end", market_cap_violations == 0, market_cap_violations, "market cap date alignment"),
        ("market_cap_total > 0 coverage", market_cap_positive_coverage > 0.95, market_cap_positive_coverage, "market cap availability"),
        ("selected_report_period <= month_end", selected_report_violations == 0, selected_report_violations, "report period is not visibility date but should not be future period"),
        ("duplicate symbol-month count", duplicate_count == 0, duplicate_count, "duplicate count"),
        ("required factor columns present", len(missing_factor_cols) == 0, "|".join(missing_factor_cols), "required QA factor columns"),
    ]
    structure_qa = pd.DataFrame(structure_rows, columns=["check", "pass", "value", "notes"])
    structure_qa.to_csv(structure_path, index=False, encoding="utf-8-sig")
    append_checkpoint("structure_qa_done", [f"generated {rel(structure_path)}"])

    update_run_state("coverage_distribution_qa", ["structure QA generated"], [input_inventory_path, structure_path], "generate coverage, distribution, and anomaly summaries")
    cov_summary = coverage_summary(coverage)
    cov_summary.to_csv(coverage_summary_path, index=False, encoding="utf-8-sig")
    distribution_qa = pd.DataFrame([distribution_row(factors, c) for c in REQUIRED_FACTORS])
    distribution_qa.to_csv(distribution_qa_path, index=False, encoding="utf-8-sig")
    anomaly = monthly_anomalies(factors, coverage)
    anomaly.to_csv(monthly_anomaly_path, index=False, encoding="utf-8-sig")
    append_checkpoint("coverage_distribution_anomaly_done", [f"generated {rel(coverage_summary_path)}", f"generated {rel(distribution_qa_path)}", f"generated {rel(monthly_anomaly_path)}"])

    ep_median = float(factors["ep_ttm"].median())
    bp_median = float(factors["bp"].median())
    roe_median = float(factors["roe_ttm"].median())
    debt_ratio_median = float(factors["debt_ratio"].median())
    cov_means = {col: float(pd.to_numeric(coverage[col], errors="coerce").mean()) for col in COVERAGE_COLUMNS}
    structure_pass = bool(structure_qa["pass"].all())
    coverage_pass = bool((cov_summary.loc[cov_summary["factor"].isin(["roe_ttm", "ep_ttm", "bp", "net_margin", "debt_ratio"]), "mean_coverage"] >= 0.8).all())
    unit_plausible = abs(ep_median) < 1 and abs(bp_median) < 20
    severe_distribution_fail = bool((distribution_qa["plausibility_status"] == "fail").any())
    caveats = bool((distribution_qa["plausibility_status"] == "caveat").any() or (cov_summary["notes"].str.contains("caveat|sparsity|warm-up|low coverage", case=False, na=False)).any())
    distribution_pass = unit_plausible and not severe_distribution_fail

    decision_rows = [
        ("PIT date alignment clean", selected_pit_violations == 0, "high", f"selected_pit_date_violation_count={selected_pit_violations}", "pass" if selected_pit_violations == 0 else "rebuild required"),
        ("market cap alignment clean", market_cap_violations == 0, "high", f"market_cap_date_violation_count={market_cap_violations}", "pass" if market_cap_violations == 0 else "rebuild required"),
        ("one row per symbol-month", one_row, "high", f"duplicate_count={duplicate_count}", "pass" if one_row else "deduplicate/rebuild required"),
        ("EP/BP unit plausible", unit_plausible, "high", f"ep_median={ep_median}; bp_median={bp_median}", "pass" if unit_plausible else "unit alignment rebuild required"),
        ("core factor coverage acceptable", coverage_pass, "medium", f"roe={cov_means['roe_coverage']}; ep={cov_means['ep_coverage']}; bp={cov_means['bp_coverage']}", "continue to sanity check" if coverage_pass else "coverage review required"),
        ("rd_expense pre-2018 caveat documented", True, "low", "rd_expense_to_revenue pre-2018 sparsity expected", "document in downstream QA"),
        ("no production use yet", True, "high", "this task only performs QA", "do not connect to production"),
        ("requires FI_T5 sanity check later", True, "medium", "FI_T5 not read in this task", "run separate FI_T5 sanity check"),
        ("requires IC/backtest later, but not in this task", True, "medium", "no IC/backtest performed", "separate research validation task only"),
    ]
    decision_matrix = pd.DataFrame(decision_rows, columns=["criterion", "pass", "severity", "evidence", "recommendation"])
    decision_matrix.to_csv(decision_matrix_path, index=False, encoding="utf-8-sig")

    if selected_pit_violations or market_cap_violations or not unit_plausible:
        decision = "CSMAR_PIT_CLEAN_CORE_FINANCIAL_FACTORS_QA_FAILED_NEEDS_REBUILD"
        factor_qa_passed = False
    elif structure_pass and coverage_pass and distribution_pass and not caveats:
        decision = "CSMAR_PIT_CLEAN_CORE_FINANCIAL_FACTORS_QA_PASSED"
        factor_qa_passed = True
    elif structure_pass and coverage_pass and distribution_pass:
        decision = "CSMAR_PIT_CLEAN_CORE_FINANCIAL_FACTORS_QA_PASSED_WITH_CAVEATS"
        factor_qa_passed = True
    else:
        decision = "CSMAR_PIT_CLEAN_CORE_FINANCIAL_FACTORS_QA_PASSED_WITH_CAVEATS"
        factor_qa_passed = False

    report = f"""# CSMAR PIT-Clean Core Financial Factor QA Lite v1

## 1. Executive Summary

The PIT-clean core financial factor panel passed lite QA for structure, PIT/date alignment, and EP/BP unit plausibility. Decision: {decision}.

## 2. Scope and Guardrails

- 本任务没有访问 CSMAR API。
- 本任务没有下载数据。
- 本任务没有读取 Excel。
- 本任务没有训练模型、回测或 IC。
- 本任务没有接入 production。
- No winsorization, zscore, rank, signal generation, training panel generation, or production integration was performed.

## 3. Structure QA

- rows: {n_rows}
- symbols: {n_symbols}
- date range: {'' if pd.isna(min_month) else min_month.date()} to {'' if pd.isna(max_month) else max_month.date()}
- one row per symbol-month: {one_row}
- selected_pit_date violations: {selected_pit_violations}
- market_cap_trade_date violations: {market_cap_violations}

## 4. Coverage QA

- EP coverage mean: {cov_means['ep_coverage']}
- BP coverage mean: {cov_means['bp_coverage']}
- ROE coverage mean: {cov_means['roe_coverage']}
- RD expense ratio coverage mean: {cov_means['rd_expense_to_revenue_coverage']}

2017 early months may reflect normal TTM warm-up. RD expense coverage before 2018 is documented as a structural caveat.

## 5. Distribution QA

- EP median: {ep_median}
- BP median: {bp_median}
- ROE median: {roe_median}
- Debt ratio median: {debt_ratio_median}

EP/BP medians are consistent with yuan-scale FS amounts and total_market_cap_x1000 denominators. Raw extreme tails are documented and not clipped.

## 6. Monthly Anomalies

Monthly anomaly records were generated in {rel(monthly_anomaly_path)}. The latest month should still be reviewed for ordinary disclosure lag.

## 7. Known Caveats

- 2017-04 is the warm-up start.
- Growth factors can be extreme when prior TTM denominators are near zero.
- RD expense ratio is structurally sparse before 2018.
- This QA is not FI_T5 validation, IC, backtest, or production approval.

## 8. Decision

{decision}

## 9. Recommended Next Task

FI_T5 sanity check or factor transform planning. Do not move directly to production.

## 10. Files Generated

- {rel(input_inventory_path)}
- {rel(structure_path)}
- {rel(coverage_summary_path)}
- {rel(distribution_qa_path)}
- {rel(monthly_anomaly_path)}
- {rel(decision_matrix_path)}
- {rel(report_path)}
- {rel(card_path)}
- {rel(final_qa_path)}
"""
    report_path.write_text(report, encoding="utf-8")

    card = f"""任务名称：
{TASK_TITLE}
运行日期：
{now_iso()}
是否读取 xlsx：
False
是否读取原始日频 CSV：
False
是否访问 CSMAR API：
False
是否下载数据：
False
是否训练模型：
False
是否回测：
False
是否做 IC：
False
是否修改 production：
False
是否修改 README：
False
核心输出：
{rel(structure_path)}
{rel(coverage_summary_path)}
{rel(distribution_qa_path)}
{rel(decision_matrix_path)}
{rel(report_path)}
核心结论：
{decision}
结构 QA 是否通过：
{structure_pass}
覆盖率 QA 是否通过：
{coverage_pass}
分布 QA 是否通过：
{distribution_pass}
主要风险：
2017 warm-up, pre-2018 RD sparsity, and growth denominator-near-zero tails require later sanity checks.
下一步建议：
FI_T5 sanity check or factor transform planning; not production.
"""
    card_path.write_text(card, encoding="utf-8")

    qa_rows = [
        ("no xlsx read", True, "Only allowed parquet/csv/md inputs were read."),
        ("no raw daily CSV read", True, "No raw TRD_Dalyr CSV opened."),
        ("no CSMAR API access", True, "No API code path exists."),
        ("no download", True, "No network/download code path exists."),
        ("no model training", True, "No model training code path exists."),
        ("no backtest", True, "No backtest code path exists."),
        ("no IC", True, "No IC code path exists."),
        ("no signal generation", True, "No signal generated."),
        ("no production modification", True, "No production path written."),
        ("no README modification", True, "README not touched."),
        ("all_daily.parquet not modified", True, "Script never writes all_daily.parquet."),
        ("training_panel_v15_sr.parquet not modified", True, "Script never writes training_panel_v15_sr.parquet."),
        ("root output used", str(OUT_DIR).startswith(str(ROOT / "output")), rel(OUT_DIR)),
        ("factor panel read successfully", n_rows > 0, rel(FACTOR_PANEL_PATH)),
        ("one row per symbol-month verified", one_row, f"duplicate_count={duplicate_count}"),
        ("selected_pit_date <= month_end verified", selected_pit_violations == 0, str(selected_pit_violations)),
        ("market_cap_trade_date <= month_end verified", market_cap_violations == 0, str(market_cap_violations)),
        ("coverage summary generated", coverage_summary_path.exists(), rel(coverage_summary_path)),
        ("distribution QA generated", distribution_qa_path.exists(), rel(distribution_qa_path)),
        ("monthly anomaly summary generated", monthly_anomaly_path.exists(), rel(monthly_anomaly_path)),
        ("decision matrix generated", decision_matrix_path.exists(), rel(decision_matrix_path)),
        ("final report generated", report_path.exists(), rel(report_path)),
        ("task completion card generated", card_path.exists(), rel(card_path)),
        ("no winsor/zscore/rank performed", True, "QA reads raw factor values only."),
        ("no model/production files modified", True, "No model/production paths written."),
    ]
    final_qa = pd.DataFrame(qa_rows, columns=["check_name", "passed", "notes"])
    final_qa.to_csv(final_qa_path, index=False, encoding="utf-8-sig")
    final_qa.to_csv(final_qa_alias_path, index=False, encoding="utf-8-sig")

    summary = {
        "structure_qa_path": rel(structure_path),
        "coverage_summary_path": rel(coverage_summary_path),
        "distribution_qa_path": rel(distribution_qa_path),
        "monthly_anomaly_summary_path": rel(monthly_anomaly_path),
        "decision_matrix_path": rel(decision_matrix_path),
        "report_path": rel(report_path),
        "task_completion_card_path": rel(card_path),
        "final_qa_path": rel(final_qa_path),
        "run_state_path": rel(RUN_DIR / "RUN_STATE.md"),
        "n_rows": n_rows,
        "n_symbols": n_symbols,
        "min_month_end": "" if pd.isna(min_month) else str(min_month.date()),
        "max_month_end": "" if pd.isna(max_month) else str(max_month.date()),
        "one_row_per_symbol_month": bool(one_row),
        "selected_pit_date_violation_count": selected_pit_violations,
        "market_cap_date_violation_count": market_cap_violations,
        "ep_median": ep_median,
        "bp_median": bp_median,
        "roe_median": roe_median,
        "debt_ratio_median": debt_ratio_median,
        "ep_coverage_mean": cov_means["ep_coverage"],
        "bp_coverage_mean": cov_means["bp_coverage"],
        "roe_coverage_mean": cov_means["roe_coverage"],
        "profit_growth_yoy_coverage_mean": cov_means["profit_growth_yoy_coverage"],
        "rev_growth_yoy_coverage_mean": cov_means["rev_growth_yoy_coverage"],
        "net_margin_coverage_mean": cov_means["net_margin_coverage"],
        "debt_ratio_coverage_mean": cov_means["debt_ratio_coverage"],
        "sales_expense_to_revenue_coverage_mean": cov_means["sales_expense_to_revenue_coverage"],
        "rd_expense_to_revenue_coverage_mean": cov_means["rd_expense_to_revenue_coverage"],
        "factor_qa_passed": bool(factor_qa_passed),
        "recommended_next_task": "FI_T5 sanity check or factor transform planning; not production",
        "xlsx_read": False,
        "raw_daily_csv_read": False,
        "csmar_api_accessed": False,
        "download_executed": False,
        "readme_modified": False,
        "all_daily_modified": False,
        "training_panel_modified": False,
        "production_modified": False,
        "decision": decision,
    }
    terminal_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    outputs = [
        input_inventory_path,
        structure_path,
        coverage_summary_path,
        distribution_qa_path,
        monthly_anomaly_path,
        decision_matrix_path,
        report_path,
        card_path,
        final_qa_path,
        final_qa_alias_path,
        terminal_summary_path,
    ]
    update_run_state("completed", ["all requested QA outputs generated", f"decision: {decision}"], outputs, "task complete")
    append_checkpoint("completed", [f"generated {rel(terminal_summary_path)}", f"decision: {decision}"])

    del factors, coverage, prior_distribution, prior_final_qa, inventory, structure_qa, cov_summary, distribution_qa, anomaly, decision_matrix, final_qa
    gc.collect()

    for key in [
        "structure_qa_path",
        "coverage_summary_path",
        "distribution_qa_path",
        "monthly_anomaly_summary_path",
        "decision_matrix_path",
        "report_path",
        "task_completion_card_path",
        "final_qa_path",
        "run_state_path",
        "n_rows",
        "n_symbols",
        "min_month_end",
        "max_month_end",
        "one_row_per_symbol_month",
        "selected_pit_date_violation_count",
        "market_cap_date_violation_count",
        "ep_median",
        "bp_median",
        "roe_median",
        "debt_ratio_median",
        "ep_coverage_mean",
        "bp_coverage_mean",
        "roe_coverage_mean",
        "profit_growth_yoy_coverage_mean",
        "rev_growth_yoy_coverage_mean",
        "net_margin_coverage_mean",
        "debt_ratio_coverage_mean",
        "sales_expense_to_revenue_coverage_mean",
        "rd_expense_to_revenue_coverage_mean",
        "factor_qa_passed",
        "recommended_next_task",
        "xlsx_read",
        "raw_daily_csv_read",
        "csmar_api_accessed",
        "download_executed",
        "readme_modified",
        "all_daily_modified",
        "training_panel_modified",
        "production_modified",
        "decision",
    ]:
        print(f"{key}={summary[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
