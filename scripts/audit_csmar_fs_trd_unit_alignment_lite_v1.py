from __future__ import annotations

import csv
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "csmar_fs_trd_unit_alignment_audit_lite_v1"
TASK_TITLE = "CSMAR FS TRD Unit Alignment Audit Lite v1"
RUN_START_TIME = datetime.now().astimezone().isoformat(timespec="seconds")
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME
FS_PATH = ROOT / "output" / "csmar_pit_scope_freeze_strict_core_fs_panel_v1" / "strict_core_fs_monthly_asof_panel_v1.parquet"
TRD_PATH = ROOT / "output" / "csmar_trd_dalyr_market_cap_import_lite_v1" / "trd_dalyr_monthly_market_cap_panel_v1.parquet"
DES_DIR = ROOT / "data" / "csmar_exports"

FS_COLUMNS = [
    "month_end",
    "symbol",
    "selected_report_period",
    "net_profit_parent",
    "equity_parent",
    "total_assets",
    "total_liabilities",
    "operating_revenue",
]
TRD_COLUMNS = [
    "month_end",
    "symbol",
    "total_market_cap_raw_thousand",
    "total_market_cap_x1000",
    "float_market_cap_raw_thousand",
    "float_market_cap_x1000",
]
KEYWORDS = ["单位", "元", "千", "千元", "万元", "人民币", "Dsmvtll", "Dsmvosd", "B002000101", "A003100000", "A001000000"]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_checkpoint(stage: str, notes: list[str]) -> None:
    ckpt = RUN_DIR / "CHECKPOINTS.md"
    with ckpt.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now_iso()} - {stage}\n\n")
        for note in notes:
            f.write(f"- {note}\n")


def update_run_state(stage: str, done: list[str], outputs: list[Path], next_step: str) -> None:
    output_lines = "\n".join(f"  - {p.as_posix()}" for p in outputs) if outputs else "  - none"
    done_lines = "\n".join(f"  - {x}" for x in done) if done else "  - none"
    content = f"""# RUN_STATE

- 当前任务名称: {TASK_TITLE}
- 开始时间: {RUN_START_TIME}
- 当前阶段: {stage}
- 已完成步骤:
{done_lines}
- 正在处理的文件:
  - {FS_PATH.as_posix()}
  - {TRD_PATH.as_posix()}
  - {DES_DIR.as_posix()}/FS_Comins*DES*.txt
  - {DES_DIR.as_posix()}/FS_Combas*DES*.txt
  - {DES_DIR.as_posix()}/TRD_Dalyr*DES*.txt
- 已生成输出:
{output_lines}
- 下一步:
  - {next_step}
- 如果 Codex 崩溃，新的 Codex 应如何继续:
  - 先读取本文件
  - 检查 run_stdout.txt、run_stderr.txt、terminal_summary.json 和 final_qa 文件
  - 若 terminal_summary.json 不存在，重新运行 scripts/audit_csmar_fs_trd_unit_alignment_lite_v1.py，并重定向 stdout/stderr 到 agent run 目录
  - 不要读取 xlsx、原始日频 CSV、访问 API 或下载数据
"""
    (RUN_DIR / "RUN_STATE.md").write_text(content, encoding="utf-8")


def find_des_files() -> list[Path]:
    patterns = ["FS_Comins*DES*.txt", "FS_Combas*DES*.txt", "TRD_Dalyr*DES*.txt"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(DES_DIR.glob(pattern)))
    return files


def read_text_small(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "big5"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def infer_unit_from_line(line: str) -> tuple[str, str, str]:
    low = line.lower()
    if "千元" in line or ("单位" in line and "千" in line):
        return "thousand_yuan", "high", "line contains thousand-yuan unit wording"
    if "万元" in line:
        return "ten_thousand_yuan", "medium", "line contains ten-thousand-yuan unit wording"
    if "人民币元" in line or ("单位" in line and "元" in line and "千" not in line and "万" not in line):
        return "yuan", "high", "line contains yuan unit wording"
    if "dsmvtll" in low or "dsmvosd" in low:
        return "thousand_yuan", "medium", "TRD_Dalyr market-cap field known from import audit as raw thousand"
    if any(code in line for code in ("B002000101", "A003100000", "A001000000")):
        return "unknown_amount_field", "low", "target FS field code found without explicit unit in this line"
    if "人民币" in line or "元" in line:
        return "yuan_related", "low", "currency wording found but exact scale unclear"
    if "千" in line:
        return "thousand_related", "low", "thousand wording found but exact field scope unclear"
    return "unknown", "low", "keyword context only"


def des_keyword_audit() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for path in find_des_files():
        text = read_text_small(path)
        for line in text.splitlines():
            stripped = " ".join(line.strip().split())
            if not stripped:
                continue
            for kw in KEYWORDS:
                if kw in stripped:
                    unit, conf, notes = infer_unit_from_line(stripped)
                    rows.append(
                        {
                            "file_name": path.name,
                            "matched_keyword": kw,
                            "matched_line": stripped[:600],
                            "inferred_unit": unit,
                            "confidence": conf,
                            "notes": notes,
                        }
                    )
    if not rows:
        rows.append(
            {
                "file_name": "",
                "matched_keyword": "",
                "matched_line": "",
                "inferred_unit": "unknown",
                "confidence": "low",
                "notes": "no DES keyword matches found",
            }
        )
    return pd.DataFrame(rows)


def finite_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    ratio = pd.to_numeric(numer, errors="coerce") / pd.to_numeric(denom, errors="coerce")
    return ratio.replace([np.inf, -np.inf], np.nan)


def dist_row(name: str, s: pd.Series, base_n: int, notes: str) -> dict[str, object]:
    valid = s.dropna()
    q = valid.quantile([0.01, 0.05, 0.25, 0.75, 0.95, 0.99]) if len(valid) else pd.Series(dtype=float)
    median = float(valid.median()) if len(valid) else np.nan
    abs_median = float(valid.abs().median()) if len(valid) else np.nan
    if "ep_" in name or "bp_" in name:
        plausible = "plausible_scale" if pd.notna(abs_median) and 0.0001 <= abs_median <= 10 else "implausible_or_review"
    elif name == "debt_ratio_check":
        plausible = "plausible_sanity" if pd.notna(median) and -0.5 <= median <= 2.0 else "review"
    elif name == "roe_check":
        plausible = "plausible_sanity" if pd.notna(median) and -2.0 <= median <= 2.0 else "review"
    else:
        plausible = "review"
    return {
        "ratio_name": name,
        "n": int(len(valid)),
        "missing_rate": float(1 - len(valid) / base_n) if base_n else np.nan,
        "mean": float(valid.mean()) if len(valid) else np.nan,
        "median": median,
        "p01": float(q.get(0.01, np.nan)),
        "p05": float(q.get(0.05, np.nan)),
        "p25": float(q.get(0.25, np.nan)),
        "p75": float(q.get(0.75, np.nan)),
        "p95": float(q.get(0.95, np.nan)),
        "p99": float(q.get(0.99, np.nan)),
        "abs_median": abs_median,
        "plausibility_flag": plausible,
        "notes": notes,
    }


def infer_fs_units(des_df: pd.DataFrame, ratio_df: pd.DataFrame) -> tuple[str, str, str, str, str]:
    fs_des = des_df[des_df["file_name"].str.contains("FS_Com", na=False)]
    explicit_yuan = fs_des["inferred_unit"].isin(["yuan", "yuan_related"]).any()
    explicit_thousand = fs_des["inferred_unit"].eq("thousand_yuan").any()
    med = dict(zip(ratio_df["ratio_name"], ratio_df["median"]))
    ep_raw = abs(float(med.get("ep_using_raw_thousand", np.nan)))
    ep_x1000 = abs(float(med.get("ep_using_x1000", np.nan)))
    bp_raw = abs(float(med.get("bp_using_raw_thousand", np.nan)))
    bp_x1000 = abs(float(med.get("bp_using_x1000", np.nan)))
    raw_to_x_ep = ep_raw / ep_x1000 if ep_x1000 and np.isfinite(ep_x1000) else np.nan
    raw_to_x_bp = bp_raw / bp_x1000 if bp_x1000 and np.isfinite(bp_x1000) else np.nan

    x1000_plausible = np.isfinite(ep_x1000) and np.isfinite(bp_x1000) and ep_x1000 < 10 and bp_x1000 < 10
    raw_big = (np.isfinite(raw_to_x_ep) and 500 <= raw_to_x_ep <= 1500) and (np.isfinite(raw_to_x_bp) and 500 <= raw_to_x_bp <= 1500)
    if explicit_yuan and x1000_plausible and raw_big:
        return "yuan", "yuan", "total_market_cap_x1000", "high", "DES and ratio scale both indicate FS amounts in yuan; TRD raw market cap is thousand yuan"
    if explicit_thousand and not explicit_yuan and ep_raw < 10 and bp_raw < 10:
        return "thousand_yuan", "thousand_yuan", "total_market_cap_raw_thousand", "medium", "DES suggests FS amounts in thousand yuan and raw-thousand ratios are plausible"
    if x1000_plausible and raw_big:
        return "yuan", "yuan", "total_market_cap_x1000", "medium", "ratio scale indicates raw-thousand denominator makes EP/BP about 1000x larger"
    return "unclear", "unclear", "manual_review", "low", "DES and ratio evidence did not identify one clear denominator"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    des_path = OUT_DIR / "des_unit_keyword_audit_v1.csv"
    sample_path = OUT_DIR / "fs_trd_unit_alignment_sample_v1.csv"
    ratio_path = OUT_DIR / "unit_candidate_ratio_distribution_v1.csv"
    reco_path = OUT_DIR / "unit_alignment_recommendation_v1.csv"
    report_path = OUT_DIR / "csmar_fs_trd_unit_alignment_audit_lite_report_v1.md"
    card_path = OUT_DIR / "task_completion_card.md"
    qa_path = OUT_DIR / "final_qa_csmar_fs_trd_unit_alignment_audit_lite_v1.csv"
    qa_alias_path = OUT_DIR / "final_qa.csv"
    terminal_summary_path = OUT_DIR / "terminal_summary.json"

    update_run_state("des_unit_keyword_audit", ["script started"], [], "read DES txt files only")
    des_df = des_keyword_audit()
    des_df.to_csv(des_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    append_checkpoint("des_unit_keyword_audit_done", [f"generated {des_path.as_posix()}"])

    update_run_state("parquet_column_projected_read", ["DES audit generated"], [des_path], "read only allowed parquet columns")
    fs = pd.read_parquet(FS_PATH, columns=FS_COLUMNS)
    trd = pd.read_parquet(TRD_PATH, columns=TRD_COLUMNS)
    fs["month_end"] = pd.to_datetime(fs["month_end"]).dt.strftime("%Y-%m-%d")
    trd["month_end"] = pd.to_datetime(trd["month_end"]).dt.strftime("%Y-%m-%d")
    fs["symbol"] = fs["symbol"].astype(str)
    trd["symbol"] = trd["symbol"].astype(str)
    merged = fs.merge(trd, how="left", on=["month_end", "symbol"], validate="many_to_one")
    del fs, trd
    gc.collect()
    merged = merged[
        (pd.to_numeric(merged["total_market_cap_raw_thousand"], errors="coerce") > 0)
        & (pd.to_numeric(merged["total_market_cap_x1000"], errors="coerce") > 0)
    ].copy()
    base_n = len(merged)
    merged.head(2000).to_csv(sample_path, index=False, encoding="utf-8-sig")
    append_checkpoint("fs_trd_merge_sample_done", [f"merged rows after positive market-cap filters: {base_n}", f"generated {sample_path.as_posix()}"])

    update_run_state("ratio_distribution", ["parquet column-projected read done", "merge sample generated"], [des_path, sample_path], "compute unit candidate ratio distributions")
    ratios = {
        "ep_using_raw_thousand": finite_ratio(merged["net_profit_parent"], merged["total_market_cap_raw_thousand"]),
        "ep_using_x1000": finite_ratio(merged["net_profit_parent"], merged["total_market_cap_x1000"]),
        "bp_using_raw_thousand": finite_ratio(merged["equity_parent"], merged["total_market_cap_raw_thousand"]),
        "bp_using_x1000": finite_ratio(merged["equity_parent"], merged["total_market_cap_x1000"]),
        "debt_ratio_check": finite_ratio(merged["total_liabilities"], merged["total_assets"]),
        "roe_check": finite_ratio(merged["net_profit_parent"], merged["equity_parent"]),
    }
    ratio_notes = {
        "ep_using_raw_thousand": "net_profit_parent divided by TRD raw market cap stored in thousand-yuan units",
        "ep_using_x1000": "net_profit_parent divided by TRD market cap converted to yuan scale",
        "bp_using_raw_thousand": "equity_parent divided by TRD raw market cap stored in thousand-yuan units",
        "bp_using_x1000": "equity_parent divided by TRD market cap converted to yuan scale",
        "debt_ratio_check": "balance-sheet sanity check only",
        "roe_check": "income/balance sanity check only; FI_T5 F050501B can be used externally as sanity check",
    }
    ratio_df = pd.DataFrame([dist_row(k, v, base_n, ratio_notes[k]) for k, v in ratios.items()])
    ratio_df.to_csv(ratio_path, index=False, encoding="utf-8-sig")

    income_unit, balance_unit, market_cap_col, confidence, evidence = infer_fs_units(des_df, ratio_df)
    can_enter = market_cap_col in ("total_market_cap_x1000", "total_market_cap_raw_thousand")
    decision = "CSMAR_FS_TRD_UNIT_ALIGNMENT_READY_FOR_FACTOR_REBUILD" if can_enter else "CSMAR_FS_TRD_UNIT_ALIGNMENT_NEEDS_MANUAL_REVIEW"
    append_checkpoint("ratio_distribution_done", [f"generated {ratio_path.as_posix()}", f"recommended denominator: {market_cap_col}"])

    reco_rows = [
        {
            "item": "total_market_cap_for_ep_bp",
            "recommended_unit": "yuan" if market_cap_col == "total_market_cap_x1000" else "thousand_yuan" if market_cap_col == "total_market_cap_raw_thousand" else "manual_review",
            "recommended_column": market_cap_col,
            "evidence": evidence,
            "confidence": confidence,
            "notes": "Use only for future EP/BP reconstruction; this audit does not create final factors.",
        },
        {
            "item": "float_market_cap_for_diagnostic",
            "recommended_unit": "yuan" if market_cap_col == "total_market_cap_x1000" else "thousand_yuan" if market_cap_col == "total_market_cap_raw_thousand" else "manual_review",
            "recommended_column": "float_market_cap_x1000" if market_cap_col == "total_market_cap_x1000" else "float_market_cap_raw_thousand" if market_cap_col == "total_market_cap_raw_thousand" else "manual_review",
            "evidence": "Float market cap should stay on the same scale as total market cap diagnostics.",
            "confidence": confidence,
            "notes": "Diagnostic only.",
        },
        {
            "item": "fs_income_statement_amount_unit",
            "recommended_unit": income_unit,
            "recommended_column": "net_profit_parent, operating_revenue",
            "evidence": evidence,
            "confidence": confidence,
            "notes": "FS_Comins amount scale inferred from DES keywords and EP scale audit.",
        },
        {
            "item": "fs_balance_sheet_amount_unit",
            "recommended_unit": balance_unit,
            "recommended_column": "equity_parent, total_assets, total_liabilities",
            "evidence": evidence,
            "confidence": confidence,
            "notes": "FS_Combas amount scale inferred from DES keywords and BP/debt-ratio scale audit.",
        },
        {
            "item": "ep_denominator_column",
            "recommended_unit": "same as total_market_cap_for_ep_bp",
            "recommended_column": market_cap_col,
            "evidence": evidence,
            "confidence": confidence,
            "notes": "No final EP factor generated.",
        },
        {
            "item": "bp_denominator_column",
            "recommended_unit": "same as total_market_cap_for_ep_bp",
            "recommended_column": market_cap_col,
            "evidence": evidence,
            "confidence": confidence,
            "notes": "No final BP factor generated.",
        },
    ]
    reco_df = pd.DataFrame(reco_rows)
    reco_df.to_csv(reco_path, index=False, encoding="utf-8-sig")

    med = dict(zip(ratio_df["ratio_name"], ratio_df["median"]))
    report = f"""# CSMAR FS TRD Unit Alignment Audit Lite v1

## Scope Guardrails

- 本任务没有访问 CSMAR API。
- 本任务没有下载数据。
- 本任务没有读取 Excel。
- 本任务没有读取原始 TRD_Dalyr 日频 CSV。
- 本任务没有计算最终因子。
- 本任务没有训练模型、回测或做 IC。

## TRD_Dalyr Market Cap Unit

TRD_Dalyr Dsmvtll / Dsmvosd 原始单位是千。已导入月度源同时包含 raw-thousand 与 x1000 列。

## Ratio Evidence

- ep_using_raw_thousand median: {med.get("ep_using_raw_thousand")}
- ep_using_x1000 median: {med.get("ep_using_x1000")}
- bp_using_raw_thousand median: {med.get("bp_using_raw_thousand")}
- bp_using_x1000 median: {med.get("bp_using_x1000")}

## Recommendation

- 推荐 EP/BP 使用 market cap 列: {market_cap_col}
- FS_Comins 财务金额单位判断: {income_unit}
- FS_Combas 财务金额单位判断: {balance_unit}
- 是否可以进入 PIT-clean factor reconstruction: {str(can_enter)}

## FI_T5 Sanity Check Note

FI_T5 仅作为 sanity check，不在本任务中计算最终因子:

- F050501B = ROE
- F053301B = 营业毛利率
- F051701B = 销售费用率
- F051801B = 管理费用率

## Decision

{decision}
"""
    report_path.write_text(report, encoding="utf-8")

    card = f"""任务名称：
{TASK_TITLE}
运行日期：
{now_iso()}
是否读取 xlsx：
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
{des_path.as_posix()}
{ratio_path.as_posix()}
{reco_path.as_posix()}
{report_path.as_posix()}
核心结论：
{evidence}
推荐 market cap 列：
{market_cap_col}
FS 金额单位判断：
FS_Comins={income_unit}; FS_Combas={balance_unit}
是否可以进入因子重建：
{str(can_enter)}
下一步建议：
PIT-clean factor reconstruction using {market_cap_col} if decision is READY.
"""
    card_path.write_text(card, encoding="utf-8")

    qa_rows = [
        ("no xlsx read", True, "No xlsx file opened; DES txt filenames may contain xlsx token but are text metadata."),
        ("no raw daily CSV read", True, "No TRD_Dalyr raw daily CSV opened."),
        ("no CSMAR API access", True, "No API code path exists."),
        ("no download", True, "No network/download code path exists."),
        ("no model training", True, "No model code path exists."),
        ("no backtest", True, "No backtest code path exists."),
        ("no IC", True, "No IC code path exists."),
        ("no production modification", True, "Only script and output audit files written."),
        ("no README modification", True, "README not modified."),
        ("root output used", str(OUT_DIR).startswith(str(ROOT / "output")), OUT_DIR.as_posix()),
        ("DES unit audit generated", des_path.exists(), des_path.as_posix()),
        ("unit candidate ratio distribution generated", ratio_path.exists(), ratio_path.as_posix()),
        ("unit alignment recommendation generated", reco_path.exists(), reco_path.as_posix()),
        ("report generated", report_path.exists(), report_path.as_posix()),
        ("task completion card generated", card_path.exists(), card_path.as_posix()),
        ("no final factor panel generated", True, "No final factor panel output path was created by this script."),
    ]
    qa_df = pd.DataFrame(qa_rows, columns=["check_name", "passed", "notes"])
    qa_df.to_csv(qa_path, index=False, encoding="utf-8-sig")
    qa_df.to_csv(qa_alias_path, index=False, encoding="utf-8-sig")

    summary = {
        "des_unit_keyword_audit_path": des_path.as_posix(),
        "unit_candidate_ratio_distribution_path": ratio_path.as_posix(),
        "unit_alignment_recommendation_path": reco_path.as_posix(),
        "report_path": report_path.as_posix(),
        "task_completion_card_path": card_path.as_posix(),
        "final_qa_path": qa_path.as_posix(),
        "terminal_summary_path": terminal_summary_path.as_posix(),
        "run_state_path": (RUN_DIR / "RUN_STATE.md").as_posix(),
        "recommended_market_cap_column_for_ep_bp": market_cap_col,
        "fs_income_statement_unit_inferred": income_unit,
        "fs_balance_sheet_unit_inferred": balance_unit,
        "ep_raw_thousand_median": med.get("ep_using_raw_thousand"),
        "ep_x1000_median": med.get("ep_using_x1000"),
        "bp_raw_thousand_median": med.get("bp_using_raw_thousand"),
        "bp_x1000_median": med.get("bp_using_x1000"),
        "can_enter_factor_reconstruction": can_enter,
        "recommended_next_task": f"PIT-clean factor reconstruction using {market_cap_col}" if can_enter else "manual unit review",
        "xlsx_read": False,
        "raw_daily_csv_read": False,
        "csmar_api_accessed": False,
        "download_executed": False,
        "readme_modified": False,
        "production_modified": False,
        "decision": decision,
    }
    terminal_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    generated = [des_path, sample_path, ratio_path, reco_path, report_path, card_path, qa_path, qa_alias_path, terminal_summary_path]
    update_run_state("completed", ["all audit outputs generated", f"decision: {decision}"], generated, "task complete")
    append_checkpoint("completed", [f"generated {terminal_summary_path.as_posix()}", f"decision: {decision}"])

    del merged, des_df, ratio_df, reco_df, qa_df
    gc.collect()

    for key in [
        "des_unit_keyword_audit_path",
        "unit_candidate_ratio_distribution_path",
        "unit_alignment_recommendation_path",
        "report_path",
        "task_completion_card_path",
        "final_qa_path",
        "terminal_summary_path",
        "run_state_path",
        "recommended_market_cap_column_for_ep_bp",
        "fs_income_statement_unit_inferred",
        "fs_balance_sheet_unit_inferred",
        "ep_raw_thousand_median",
        "ep_x1000_median",
        "bp_raw_thousand_median",
        "bp_x1000_median",
        "can_enter_factor_reconstruction",
        "recommended_next_task",
        "xlsx_read",
        "raw_daily_csv_read",
        "csmar_api_accessed",
        "download_executed",
        "readme_modified",
        "production_modified",
        "decision",
    ]:
        print(f"{key}={summary[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
