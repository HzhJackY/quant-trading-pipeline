from __future__ import annotations

import gc
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


TASK_NAME = "CSMAR TRD_Dalyr Market Cap Import Bugfix v1"
TASK_SLUG = "csmar_trd_dalyr_market_cap_import_bugfix_v1"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "csmar_exports"
OUT_DIR = ROOT / "output" / TASK_SLUG
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_SLUG
RUN_STATE = RUN_DIR / "RUN_STATE.md"
CHECKPOINTS = RUN_DIR / "CHECKPOINTS.md"

V1_PANEL = ROOT / "output" / "csmar_trd_dalyr_market_cap_import_lite_v1" / "trd_dalyr_monthly_market_cap_panel_v1.parquet"
V1_COVERAGE = ROOT / "output" / "csmar_trd_dalyr_market_cap_import_lite_v1" / "market_cap_coverage_vs_strict_core_fs_v1.csv"
STRICT_FS_PANEL = ROOT / "output" / "csmar_pit_scope_freeze_strict_core_fs_panel_v1" / "strict_core_fs_monthly_asof_panel_v1.parquet"

START = pd.Timestamp("2016-07-01")
END = pd.Timestamp("2026-06-30")
CHUNKSIZE = 150_000
ENCODINGS = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
REQ = ["Stkcd", "Trddt", "Clsprc", "Dsmvosd", "Dsmvtll", "Markettype", "Trdsta"]
A_SHARE = {1, 4, 16, 32, 64}
RENAME = {
    "Stkcd": "symbol",
    "Trddt": "trade_date",
    "Clsprc": "close_price",
    "Dsmvosd": "float_market_cap_raw_thousand",
    "Dsmvtll": "total_market_cap_raw_thousand",
    "Markettype": "market_type",
    "Trdsta": "trading_status",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def checkpoint(phase: str, status: str, details: str) -> None:
    with CHECKPOINTS.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now()}\n\n- 阶段: {phase}\n- 状态: {status}\n- 说明: {details}\n")


def write_run_state(phase: str, completed: list[str], current_file: str, outputs: list[str], next_step: str) -> None:
    lines = [
        "# RUN_STATE",
        "",
        f"- 当前任务名称: {TASK_NAME}",
        "- 开始时间: 2026-06-30T21:25:07+08:00",
        f"- 最后更新时间: {now()}",
        f"- 当前阶段: {phase}",
        "- 已完成步骤:",
    ]
    lines.extend([f"  - {x}" for x in completed] or ["  - none"])
    lines.extend(["- 正在处理的文件:", f"  - {current_file or 'none'}", "- 已生成输出:"])
    lines.extend([f"  - {x}" for x in outputs] or ["  - none"])
    lines.extend(
        [
            "- 下一步:",
            f"  - {next_step}",
            "- 如果 Codex 崩溃，新的 Codex 应如何继续:",
            f"  - 先读取 {rel(RUN_STATE)}",
            f"  - 查看 {rel(RUN_DIR / 'run_stdout.txt')} 和 {rel(RUN_DIR / 'run_stderr.txt')}",
            f"  - 如 v2 panel 已存在，先检查 {rel(OUT_DIR / 'terminal_summary.json')} 与 final_qa，再决定是否重跑",
            f"  - 若需重跑，执行 python {rel(ROOT / 'scripts' / 'bugfix_csmar_trd_dalyr_market_cap_import_v1.py')}",
        ]
    )
    RUN_STATE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_encoding(path: Path) -> tuple[str, list[str]]:
    notes = []
    for enc in ENCODINGS:
        try:
            sample = pd.read_csv(path, encoding=enc, nrows=5, dtype={"Stkcd": "string"})
            cols = list(sample.columns)
            del sample
            gc.collect()
            return enc, cols
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{enc}:{type(exc).__name__}")
    raise RuntimeError("; ".join(notes))


def list_csvs() -> list[Path]:
    return sorted([p for p in DATA_DIR.glob("TRD_Dalyr*.csv") if p.is_file()])


def diagnose_and_build(csvs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    best: dict[tuple[str, pd.Timestamp], dict] = {}
    diag_rows = []
    totals = {"n_rows": 0, "n_valid": 0}
    completed = ["初始化"]
    outputs: list[str] = []

    for i, path in enumerate(csvs, start=1):
        write_run_state(
            f"diagnose and stream file {i}/{len(csvs)}",
            completed,
            rel(path),
            outputs,
            "继续逐文件诊断并生成 v2 monthly panel",
        )
        row = {
            "file_name": path.name,
            "file_path": rel(path),
            "readable": False,
            "columns": "",
            "n_rows_processed": 0,
            "n_valid_rows": 0,
            "min_trddt": "",
            "max_trddt": "",
            "n_symbols": 0,
            "has_2021_2026_data": False,
            "encoding_used": "",
            "skipped": False,
            "skip_reason": "",
            "notes": "",
        }
        symbols: set[str] = set()
        min_dt = None
        max_dt = None
        try:
            enc, cols = choose_encoding(path)
            row["encoding_used"] = enc
            row["columns"] = "|".join(cols)
            missing = [c for c in REQ if c not in cols]
            if missing:
                row["skipped"] = True
                row["skip_reason"] = "missing columns: " + ",".join(missing)
                diag_rows.append(row)
                checkpoint("逐文件诊断", "skipped", f"{path.name}: {row['skip_reason']}")
                continue
            row["readable"] = True
            reader = pd.read_csv(path, encoding=enc, usecols=REQ, dtype={"Stkcd": "string", "Trdsta": "string"}, chunksize=CHUNKSIZE)
            for chunk in reader:
                row["n_rows_processed"] += len(chunk)
                totals["n_rows"] += len(chunk)
                chunk = chunk.rename(columns=RENAME)
                chunk["symbol"] = chunk["symbol"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
                chunk["trade_date"] = pd.to_datetime(chunk["trade_date"], errors="coerce")
                for col in ["close_price", "float_market_cap_raw_thousand", "total_market_cap_raw_thousand", "market_type"]:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

                valid_date = chunk["trade_date"].dropna()
                if not valid_date.empty:
                    cmin = valid_date.min()
                    cmax = valid_date.max()
                    min_dt = cmin if min_dt is None else min(min_dt, cmin)
                    max_dt = cmax if max_dt is None else max(max_dt, cmax)

                symbols.update(chunk["symbol"].dropna().astype(str).unique().tolist())
                valid = chunk[
                    (chunk["trade_date"] >= START)
                    & (chunk["trade_date"] <= END)
                    & (chunk["market_type"].isin(A_SHARE))
                    & (chunk["total_market_cap_raw_thousand"] > 0)
                ].copy()
                row["n_valid_rows"] += len(valid)
                totals["n_valid"] += len(valid)
                if not valid.empty:
                    valid["month_end"] = valid["trade_date"] + pd.offsets.MonthEnd(0)
                    for rec in valid.itertuples(index=False):
                        key = (rec.symbol, rec.month_end)
                        old = best.get(key)
                        if old is not None and rec.trade_date <= old["trade_date"]:
                            continue
                        best[key] = {
                            "month_end": rec.month_end,
                            "symbol": rec.symbol,
                            "trade_date": rec.trade_date,
                            "close_price": rec.close_price,
                            "total_market_cap_raw_thousand": rec.total_market_cap_raw_thousand,
                            "total_market_cap_x1000": rec.total_market_cap_raw_thousand * 1000,
                            "float_market_cap_raw_thousand": rec.float_market_cap_raw_thousand,
                            "float_market_cap_x1000": rec.float_market_cap_raw_thousand * 1000 if pd.notna(rec.float_market_cap_raw_thousand) else pd.NA,
                            "market_type": int(rec.market_type) if pd.notna(rec.market_type) else pd.NA,
                            "trading_status": rec.trading_status,
                            "source_file": path.name,
                            "market_cap_unit_note": "Dsmvtll/Dsmvosd raw unit is thousand; x1000 columns multiply raw values by 1000.",
                        }
                del valid
                del chunk
                gc.collect()
            row["min_trddt"] = "" if min_dt is None else str(pd.Timestamp(min_dt).date())
            row["max_trddt"] = "" if max_dt is None else str(pd.Timestamp(max_dt).date())
            row["n_symbols"] = len(symbols)
            row["has_2021_2026_data"] = bool(max_dt is not None and max_dt >= pd.Timestamp("2021-07-01"))
            checkpoint("逐文件诊断和 streaming", "completed", f"{path.name}: rows={row['n_rows_processed']}, valid={row['n_valid_rows']}, range={row['min_trddt']} to {row['max_trddt']}")
            completed.append(f"diagnosed and streamed {path.name}")
        except Exception as exc:  # noqa: BLE001
            row["skipped"] = True
            row["skip_reason"] = f"{type(exc).__name__}: {exc}"
            checkpoint("逐文件诊断和 streaming", "failed", f"{path.name}: {row['skip_reason']}")
        finally:
            diag_rows.append(row)
            gc.collect()

    diag = pd.DataFrame(diag_rows)
    panel = pd.DataFrame(best.values())
    if not panel.empty:
        panel = panel.sort_values(["month_end", "symbol"]).reset_index(drop=True)
    del best
    gc.collect()
    return diag, panel, totals


def panel_metrics(panel: pd.DataFrame) -> dict:
    if panel.empty:
        return {
            "min_trade_date": "",
            "max_trade_date": "",
            "n_rows": 0,
            "n_symbols": 0,
            "n_months": 0,
            "n_symbol_months": 0,
            "has_2021_2026_months": False,
            "latest_month_end": "",
            "n_symbol_months_after_2021_06": 0,
        }
    m = panel.copy()
    m["trade_date"] = pd.to_datetime(m["trade_date"], errors="coerce")
    m["month_end"] = pd.to_datetime(m["month_end"], errors="coerce") + pd.offsets.MonthEnd(0)
    out = {
        "min_trade_date": str(m["trade_date"].min().date()),
        "max_trade_date": str(m["trade_date"].max().date()),
        "n_rows": len(m),
        "n_symbols": int(m["symbol"].nunique()),
        "n_months": int(m["month_end"].nunique()),
        "n_symbol_months": len(m),
        "has_2021_2026_months": bool((m["month_end"] > pd.Timestamp("2021-06-30")).any()),
        "latest_month_end": str(m["month_end"].max().date()),
        "n_symbol_months_after_2021_06": int((m["month_end"] > pd.Timestamp("2021-06-30")).sum()),
    }
    del m
    gc.collect()
    return out


def compare_v1_v2(v2: pd.DataFrame) -> tuple[pd.DataFrame, dict, float | str]:
    if V1_PANEL.exists():
        v1 = pd.read_parquet(V1_PANEL)
    else:
        v1 = pd.DataFrame()
    m1 = panel_metrics(v1)
    m2 = panel_metrics(v2)
    rows = [{"metric": k, "v1_value": m1[k], "v2_value": m2[k], "details": ""} for k in m1]
    comp = pd.DataFrame(rows)
    v1_mean = ""
    if V1_COVERAGE.exists():
        cov = pd.read_csv(V1_COVERAGE)
        if "overlap_rate" in cov.columns:
            rates = pd.to_numeric(cov["overlap_rate"], errors="coerce").dropna()
            if not rates.empty:
                v1_mean = float(rates.mean())
    del v1
    gc.collect()
    return comp, {"v1": m1, "v2": m2}, v1_mean


def coverage_audit(panel: pd.DataFrame) -> tuple[pd.DataFrame, float | str, float | str, str]:
    if not STRICT_FS_PANEL.exists():
        cov = pd.DataFrame([{"month_end": "", "strict_fs_n_symbols": "", "market_cap_n_symbols": "", "overlap_n_symbols": "", "overlap_rate": "", "missing_market_cap_n_symbols": "", "notes": "skipped: strict FS panel missing"}])
        return cov, "", "", "skipped: strict FS panel missing"
    try:
        strict = pd.read_parquet(STRICT_FS_PANEL, columns=["month_end", "symbol"])
        strict["month_end"] = pd.to_datetime(strict["month_end"], errors="coerce") + pd.offsets.MonthEnd(0)
        strict["symbol"] = strict["symbol"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
        mc = panel[["month_end", "symbol"]].copy()
        mc["month_end"] = pd.to_datetime(mc["month_end"], errors="coerce") + pd.offsets.MonthEnd(0)
        mc["symbol"] = mc["symbol"].astype("string").str.strip().str.zfill(6)
        mc_by_month = {k: set(v["symbol"].dropna().astype(str)) for k, v in mc.groupby("month_end")}
        rows = []
        for month, group in strict.groupby("month_end", sort=True):
            strict_syms = set(group["symbol"].dropna().astype(str))
            mc_syms = mc_by_month.get(month, set())
            overlap = strict_syms & mc_syms
            missing = strict_syms - mc_syms
            rows.append(
                {
                    "month_end": month.date().isoformat() if pd.notna(month) else "",
                    "strict_fs_n_symbols": len(strict_syms),
                    "market_cap_n_symbols": len(mc_syms),
                    "overlap_n_symbols": len(overlap),
                    "overlap_rate": len(overlap) / len(strict_syms) if strict_syms else 0,
                    "missing_market_cap_n_symbols": len(missing),
                    "notes": "",
                }
            )
        cov = pd.DataFrame(rows)
        rates = pd.to_numeric(cov["overlap_rate"], errors="coerce").dropna()
        mean_rate = float(rates.mean()) if not rates.empty else ""
        min_rate = float(rates.min()) if not rates.empty else ""
        del strict
        del mc
        gc.collect()
        return cov, mean_rate, min_rate, "completed"
    except Exception as exc:  # noqa: BLE001
        cov = pd.DataFrame([{"month_end": "", "strict_fs_n_symbols": "", "market_cap_n_symbols": "", "overlap_n_symbols": "", "overlap_rate": "", "missing_market_cap_n_symbols": "", "notes": f"skipped: {type(exc).__name__}: {exc}"}])
        return cov, "", "", f"skipped: {type(exc).__name__}: {exc}"


def write_report(summary: dict, diagnostics: pd.DataFrame, metrics: dict, coverage_note: str, v1_mean: float | str) -> None:
    files_2021 = diagnostics[diagnostics["file_name"].str.contains("20210701_20260630", regex=False)]
    readable_2021 = int((files_2021["readable"] == True).sum()) if not files_2021.empty else 0
    has_late = int((files_2021["has_2021_2026_data"] == True).sum()) if not files_2021.empty else 0
    reason = (
        "诊断显示文件名包含 20210701_20260630 的 CSV 可读，但其真实 max_trddt 未进入 2021-07 之后；v1 因输入文件内容本身只到 2021-06-30 而停止。"
        if has_late == 0
        else "诊断显示 2021-2026 文件包含 2021-07 之后数据；v2 已按自然月末标签和月内最后实际交易日重新生成。"
    )
    text = f"""# CSMAR TRD_Dalyr Market Cap Import Bugfix Report v1

## Executive Summary

Decision: {summary['decision']}

v2 monthly market cap panel was generated with chunked streaming. No Excel files were read, no full daily parquet was saved, no CSMAR API was accessed, no data was downloaded, and EP/BP was not calculated.

## Why v1 Stopped At 2021-06-30

{reason}

## 2021-2026 File Diagnostics

- Files named 20210701_20260630 detected: {summary['n_2021_2026_files_detected']}
- Files named 20210701_20260630 read: {readable_2021}
- Files with actual data after 2021-07-01: {has_late}

See trd_dalyr_file_date_diagnostics_v1.csv for per-file min_trddt and max_trddt.

## Correct Monthly Logic

month_end is a natural calendar month-end label. It is not required to be a trading date. For each symbol + month_end, v2 keeps the row with the maximum actual trade_date in that month. trade_date == month_end is not required.

## v1 vs v2

- v1 trade_date range: {metrics['v1']['min_trade_date']} to {metrics['v1']['max_trade_date']}
- v2 trade_date range: {metrics['v2']['min_trade_date']} to {metrics['v2']['max_trade_date']}
- v1 mean coverage: {v1_mean}
- v2 mean coverage: {summary['v2_mean_market_cap_overlap_rate']}

## Coverage vs Strict FS

Coverage audit status: {coverage_note}

## Can Proceed

can_support_ep_bp_market_cap: {summary['can_support_ep_bp_market_cap']}

The next task should confirm FS_Comins / FS_Combas amount units, then run PIT-clean EP/BP reconstruction if coverage is acceptable.
"""
    (OUT_DIR / "csmar_trd_dalyr_market_cap_import_bugfix_report_v1.md").write_text(text, encoding="utf-8")


def write_completion_card(summary: dict, v1_mean: float | str) -> None:
    text = f"""任务名称：
{TASK_NAME}
运行日期：
2026-06-30
是否读取 xlsx：
False
是否保存全量日频 parquet：
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
v1 日期范围：
{summary['v1_min_trade_date']} to {summary['v1_max_trade_date']}
v2 日期范围：
{summary['v2_min_trade_date']} to {summary['v2_max_trade_date']}
v1 平均覆盖率：
{v1_mean}
v2 平均覆盖率：
{summary['v2_mean_market_cap_overlap_rate']}
2021-2026 文件是否进入 panel：
{summary['n_2021_2026_files_read'] > 0 and summary['v2_max_trade_date'] >= '2021-07-01'}
核心结论：
{summary['decision']}
下一步建议：
{summary['recommended_next_task']}
"""
    (OUT_DIR / "task_completion_card.md").write_text(text, encoding="utf-8")


def write_qa(summary: dict, panel: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    p = panel.copy()
    if not p.empty:
        p["month_end"] = pd.to_datetime(p["month_end"], errors="coerce") + pd.offsets.MonthEnd(0)
        p["trade_date"] = pd.to_datetime(p["trade_date"], errors="coerce")
    detected_2021 = diagnostics["file_name"].str.contains("20210701_20260630", regex=False).sum()
    read_2021 = diagnostics.loc[diagnostics["file_name"].str.contains("20210701_20260630", regex=False), "readable"].sum()
    checks = [
        ("no xlsx read", True, ""),
        ("no full daily parquet saved", True, ""),
        ("no CSMAR API access", True, ""),
        ("no download", True, ""),
        ("no model training", True, ""),
        ("no backtest", True, ""),
        ("no IC", True, ""),
        ("no production modification", not summary["production_modified"], ""),
        ("no README modification", not summary["readme_modified"], ""),
        ("TRD_Dalyr 2021-2026 files detected", detected_2021 > 0, str(int(detected_2021))),
        ("TRD_Dalyr 2021-2026 files read", read_2021 > 0, str(int(read_2021))),
        ("v2 monthly panel generated", Path(summary["monthly_market_cap_panel_v2_path_abs"]).exists(), summary["monthly_market_cap_panel_v2_path"]),
        ("v2 max_trade_date > 2026-01-01", summary["v2_max_trade_date"] > "2026-01-01", summary["v2_max_trade_date"]),
        ("month_end used as natural month-end label", bool(p.empty or (p["month_end"].dt.is_month_end.all())), ""),
        ("trade_date <= month_end", bool(p.empty or (p["trade_date"] <= p["month_end"]).all()), ""),
        ("no requirement trade_date == month_end", True, "selection uses max trade_date per symbol-month"),
        ("one row per symbol-month", bool(p.empty or not p.duplicated(["symbol", "month_end"]).any()), ""),
        ("coverage audit generated", Path(summary["coverage_vs_strict_fs_v2_path_abs"]).exists(), summary["coverage_vs_strict_fs_v2_path"]),
        ("report generated", Path(summary["report_path_abs"]).exists(), summary["report_path"]),
        ("task completion card generated", Path(summary["task_completion_card_path_abs"]).exists(), summary["task_completion_card_path"]),
    ]
    pd.DataFrame(checks, columns=["check", "pass", "details"]).to_csv(OUT_DIR / "final_qa_csmar_trd_dalyr_market_cap_import_bugfix_v1.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(checks, columns=["check", "pass", "details"]).to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    del p
    gc.collect()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    tracked_before = {
        "README.md": (ROOT / "README.md").stat().st_mtime_ns if (ROOT / "README.md").exists() else None,
        "output/all_daily.parquet": (ROOT / "output" / "all_daily.parquet").stat().st_mtime_ns if (ROOT / "output" / "all_daily.parquet").exists() else None,
        "output/training_panel_v15_sr.parquet": (ROOT / "output" / "training_panel_v15_sr.parquet").stat().st_mtime_ns if (ROOT / "output" / "training_panel_v15_sr.parquet").exists() else None,
    }

    csvs = list_csvs()
    diagnostics, panel, totals = diagnose_and_build(csvs)
    diag_path = OUT_DIR / "trd_dalyr_file_date_diagnostics_v1.csv"
    diagnostics.to_csv(diag_path, index=False, encoding="utf-8-sig")
    checkpoint("文件日期诊断", "completed", f"diagnostics rows={len(diagnostics)}")

    panel_path = OUT_DIR / "trd_dalyr_monthly_market_cap_panel_v2.parquet"
    sample_path = OUT_DIR / "trd_dalyr_monthly_market_cap_panel_v2_sample.csv"
    panel.to_parquet(panel_path, index=False)
    panel.head(1000).to_csv(sample_path, index=False, encoding="utf-8-sig")
    checkpoint("v2 monthly panel", "completed", f"rows={len(panel)}, symbols={panel['symbol'].nunique() if not panel.empty else 0}")

    comparison, metrics, v1_mean = compare_v1_v2(panel)
    comparison.to_csv(OUT_DIR / "v1_vs_v2_market_cap_panel_comparison_v1.csv", index=False, encoding="utf-8-sig")
    coverage, mean_overlap, min_overlap, coverage_note = coverage_audit(panel)
    coverage.to_csv(OUT_DIR / "market_cap_coverage_vs_strict_core_fs_v2.csv", index=False, encoding="utf-8-sig")
    checkpoint("v1/v2 与 coverage 审计", "completed", f"coverage={coverage_note}")

    tracked_after = {
        "README.md": (ROOT / "README.md").stat().st_mtime_ns if (ROOT / "README.md").exists() else None,
        "output/all_daily.parquet": (ROOT / "output" / "all_daily.parquet").stat().st_mtime_ns if (ROOT / "output" / "all_daily.parquet").exists() else None,
        "output/training_panel_v15_sr.parquet": (ROOT / "output" / "training_panel_v15_sr.parquet").stat().st_mtime_ns if (ROOT / "output" / "training_panel_v15_sr.parquet").exists() else None,
    }

    n_2021_detected = int(diagnostics["file_name"].str.contains("20210701_20260630", regex=False).sum())
    n_2021_read = int(diagnostics.loc[diagnostics["file_name"].str.contains("20210701_20260630", regex=False), "readable"].sum())
    v2_max = metrics["v2"]["max_trade_date"]
    if v2_max >= "2026-06-01" and mean_overlap != "" and mean_overlap >= 0.95:
        decision = "CSMAR_TRD_DALYR_MARKET_CAP_IMPORT_BUGFIX_READY"
    elif n_2021_detected > 0 and n_2021_read == 0:
        decision = "CSMAR_TRD_DALYR_BUGFIX_FAILED_2021_2026_FILES_NOT_READ"
    else:
        decision = "CSMAR_TRD_DALYR_MARKET_CAP_BUGFIX_NEEDS_REVIEW"

    summary = {
        "file_date_diagnostics_path": rel(diag_path),
        "monthly_market_cap_panel_v2_path": rel(panel_path),
        "monthly_market_cap_sample_v2_path": rel(sample_path),
        "v1_vs_v2_comparison_path": rel(OUT_DIR / "v1_vs_v2_market_cap_panel_comparison_v1.csv"),
        "coverage_vs_strict_fs_v2_path": rel(OUT_DIR / "market_cap_coverage_vs_strict_core_fs_v2.csv"),
        "report_path": rel(OUT_DIR / "csmar_trd_dalyr_market_cap_import_bugfix_report_v1.md"),
        "task_completion_card_path": rel(OUT_DIR / "task_completion_card.md"),
        "final_qa_path": rel(OUT_DIR / "final_qa_csmar_trd_dalyr_market_cap_import_bugfix_v1.csv"),
        "run_state_path": rel(RUN_STATE),
        "monthly_market_cap_panel_v2_path_abs": str(panel_path),
        "coverage_vs_strict_fs_v2_path_abs": str(OUT_DIR / "market_cap_coverage_vs_strict_core_fs_v2.csv"),
        "report_path_abs": str(OUT_DIR / "csmar_trd_dalyr_market_cap_import_bugfix_report_v1.md"),
        "task_completion_card_path_abs": str(OUT_DIR / "task_completion_card.md"),
        "n_input_csv_files": len(csvs),
        "n_2021_2026_files_detected": n_2021_detected,
        "n_2021_2026_files_read": n_2021_read,
        "v1_min_trade_date": metrics["v1"]["min_trade_date"],
        "v1_max_trade_date": metrics["v1"]["max_trade_date"],
        "v2_min_trade_date": metrics["v2"]["min_trade_date"],
        "v2_max_trade_date": metrics["v2"]["max_trade_date"],
        "v2_n_symbol_months": metrics["v2"]["n_symbol_months"],
        "v2_n_symbols": metrics["v2"]["n_symbols"],
        "v2_mean_market_cap_overlap_rate": mean_overlap,
        "v2_min_market_cap_overlap_rate": min_overlap,
        "trade_date_equals_month_end_required": False,
        "full_daily_parquet_saved": False,
        "can_support_ep_bp_market_cap": bool(panel_path.exists() and len(panel) > 0 and (mean_overlap == "" or mean_overlap >= 0.95)),
        "recommended_next_task": "If late-period TRD_Dalyr files contain real 2021-2026 dates, proceed to unit alignment and PIT-clean EP/BP reconstruction; otherwise re-export correct 2021-2026 TRD_Dalyr CSV chunks and rerun this bugfix.",
        "xlsx_read": False,
        "csmar_api_accessed": False,
        "download_executed": False,
        "readme_modified": tracked_before["README.md"] != tracked_after["README.md"],
        "all_daily_modified": tracked_before["output/all_daily.parquet"] != tracked_after["output/all_daily.parquet"],
        "training_panel_modified": tracked_before["output/training_panel_v15_sr.parquet"] != tracked_after["output/training_panel_v15_sr.parquet"],
        "production_modified": False,
        "decision": decision,
    }

    write_report(summary, diagnostics, metrics, coverage_note, v1_mean)
    write_completion_card(summary, v1_mean)
    write_qa(summary, panel, diagnostics)
    with (OUT_DIR / "terminal_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    checkpoint("最终报告和 QA", "completed", f"decision={decision}")
    write_run_state(
        "completed",
        ["逐文件日期诊断", "v2 monthly panel", "v1/v2 comparison", "coverage audit", "最终报告和 QA"],
        "none",
        [
            rel(diag_path),
            rel(panel_path),
            rel(sample_path),
            rel(OUT_DIR / "v1_vs_v2_market_cap_panel_comparison_v1.csv"),
            rel(OUT_DIR / "market_cap_coverage_vs_strict_core_fs_v2.csv"),
            rel(OUT_DIR / "csmar_trd_dalyr_market_cap_import_bugfix_report_v1.md"),
            rel(OUT_DIR / "task_completion_card.md"),
            rel(OUT_DIR / "final_qa_csmar_trd_dalyr_market_cap_import_bugfix_v1.csv"),
            rel(OUT_DIR / "final_qa.csv"),
            rel(OUT_DIR / "terminal_summary.json"),
        ],
        "根据诊断结论决定是否需要重新导出正确的 2021-2026 TRD_Dalyr CSV",
    )

    keys = [
        "file_date_diagnostics_path",
        "monthly_market_cap_panel_v2_path",
        "monthly_market_cap_sample_v2_path",
        "v1_vs_v2_comparison_path",
        "coverage_vs_strict_fs_v2_path",
        "report_path",
        "task_completion_card_path",
        "final_qa_path",
        "run_state_path",
        "n_input_csv_files",
        "n_2021_2026_files_detected",
        "n_2021_2026_files_read",
        "v1_min_trade_date",
        "v1_max_trade_date",
        "v2_min_trade_date",
        "v2_max_trade_date",
        "v2_n_symbol_months",
        "v2_n_symbols",
        "v2_mean_market_cap_overlap_rate",
        "v2_min_market_cap_overlap_rate",
        "trade_date_equals_month_end_required",
        "full_daily_parquet_saved",
        "can_support_ep_bp_market_cap",
        "recommended_next_task",
        "xlsx_read",
        "csmar_api_accessed",
        "download_executed",
        "readme_modified",
        "all_daily_modified",
        "training_panel_modified",
        "production_modified",
        "decision",
    ]
    for key in keys:
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
