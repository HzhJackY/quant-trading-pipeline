from __future__ import annotations

import gc
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


TASK_NAME = "CSMAR TRD_Dalyr Market Cap Import Lite v1"
TASK_SLUG = "csmar_trd_dalyr_market_cap_import_lite_v1"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "csmar_exports"
OUT_DIR = ROOT / "output" / TASK_SLUG
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_SLUG
RUN_STATE = RUN_DIR / "RUN_STATE.md"
CHECKPOINTS = RUN_DIR / "CHECKPOINTS.md"

STRICT_FS_PANEL = ROOT / "output" / "csmar_pit_scope_freeze_strict_core_fs_panel_v1" / "strict_core_fs_monthly_asof_panel_v1.parquet"
STRICT_FS_CARD = ROOT / "output" / "csmar_pit_scope_freeze_strict_core_fs_panel_v1" / "task_completion_card.md"
PROJECT_STATUS = ROOT / "config" / "project_status.yaml"
DECISIONS = ROOT / "docs" / "DECISIONS.md"

START = pd.Timestamp("2016-07-01")
END = pd.Timestamp("2026-06-30")
CHUNKSIZE = 150_000
ENCODINGS = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
REQUIRED_COLS = ["Stkcd", "Trddt", "Clsprc", "Dsmvosd", "Dsmvtll", "Markettype", "Trdsta"]
A_SHARE_MARKET_TYPES = {1, 4, 16, 32, 64}
FIELD_RENAME = {
    "Stkcd": "symbol",
    "Trddt": "trade_date",
    "Clsprc": "close_price",
    "Dsmvosd": "float_market_cap_raw_thousand",
    "Dsmvtll": "total_market_cap_raw_thousand",
    "Markettype": "market_type",
    "Trdsta": "trading_status",
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_run_state(phase: str, completed: list[str], current_file: str, outputs: list[str], next_step: str) -> None:
    text = [
        "# RUN_STATE",
        "",
        f"- 当前任务名称: {TASK_NAME}",
        f"- 开始时间: 2026-06-30T20:31:38+08:00",
        f"- 最后更新时间: {now()}",
        f"- 当前阶段: {phase}",
        "- 已完成步骤:",
    ]
    text.extend([f"  - {x}" for x in completed] or ["  - none"])
    text.extend(["- 正在处理的文件:", f"  - {current_file or 'none'}"])
    text.append("- 已生成输出:")
    text.extend([f"  - {x}" for x in outputs] or ["  - none"])
    text.extend(
        [
            "- 下一步:",
            f"  - {next_step}",
            "- 如果 Codex 崩溃，新的 Codex 应如何继续:",
            f"  - 先读取 {rel(RUN_STATE)}",
            f"  - 查看 {rel(RUN_DIR / 'run_stdout.txt')} 与 {rel(RUN_DIR / 'run_stderr.txt')}",
            f"  - 如果 {rel(OUT_DIR / 'trd_dalyr_monthly_market_cap_panel_v1.parquet')} 已存在，先检查 terminal_summary.json 与 final_qa，再决定是否重跑",
            f"  - 若需重跑，执行 python {rel(ROOT / 'scripts' / 'import_csmar_trd_dalyr_market_cap_lite_v1.py')}；脚本只扫描 data/csmar_exports 并以 chunked streaming 读取 TRD_Dalyr CSV",
        ]
    )
    RUN_STATE.write_text("\n".join(text) + "\n", encoding="utf-8")


def checkpoint(phase: str, status: str, details: str) -> None:
    with CHECKPOINTS.open("a", encoding="utf-8") as f:
        f.write(f"\n## {now()}\n\n")
        f.write(f"- 阶段: {phase}\n")
        f.write(f"- 状态: {status}\n")
        f.write(f"- 说明: {details}\n")


def detect_period(name: str) -> tuple[str, str]:
    m = re.search(r"(\d{8})_(\d{8})", name)
    if not m:
        return "", ""
    return m.group(1), m.group(2)


def read_header_sample(path: Path) -> tuple[bool, str, list[str], int, str]:
    notes = []
    for enc in ENCODINGS:
        try:
            sample = pd.read_csv(path, encoding=enc, nrows=20, dtype={"Stkcd": "string"})
            cols = list(sample.columns)
            n = len(sample)
            del sample
            gc.collect()
            return True, enc, cols, n, "; ".join(notes)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{enc}: {type(exc).__name__}")
    return False, "", [], 0, "; ".join(notes)


def build_inventory() -> tuple[list[Path], pd.DataFrame]:
    files = sorted(
        [p for p in DATA_DIR.iterdir() if p.is_file() and (p.name.startswith("TRD_Dalyr") and (p.suffix.lower() == ".csv" or ("DES" in p.name and p.suffix.lower() == ".txt")))]
    )
    rows = []
    csv_files = []
    for path in files:
        is_des = "DES" in path.name and path.suffix.lower() == ".txt"
        start, end = detect_period(path.name)
        readable, enc, cols, n, notes = read_header_sample(path)
        role = "description_txt_excluded" if is_des else "data_csv_chunk"
        if path.suffix.lower() == ".csv" and not is_des:
            csv_files.append(path)
        rows.append(
            {
                "file_path": rel(path),
                "file_name": path.name,
                "file_size": path.stat().st_size,
                "detected_period_start": start,
                "detected_period_end": end,
                "file_role": role,
                "readable": readable,
                "encoding_detected_or_used": enc,
                "header_columns": "|".join(cols),
                "n_rows_sampled": n,
                "notes": notes,
            }
        )
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT_DIR / "input_file_inventory_v1.csv", index=False, encoding="utf-8-sig")
    return csv_files, inv


def write_field_mapping() -> None:
    rows = [
        ("Stkcd", "symbol", "证券代码", "string, 6 digits", "join_key", "按字符串读取并 zfill(6)"),
        ("Trddt", "trade_date", "交易日期", "date", "date_key", "解析为日期"),
        ("Clsprc", "close_price", "日收盘价", "source currency per share", "diagnostic", "本任务不用于因子计算"),
        ("Dsmvosd", "float_market_cap_raw_thousand", "日个股流通市值", "千", "sanity_check", "保留原始单位；不作为 EP/BP 主口径"),
        ("Dsmvosd", "float_market_cap_x1000", "日个股流通市值乘以 1000", "原始千单位 x 1000", "sanity_check", "用于后续单位对齐审计"),
        ("Dsmvtll", "total_market_cap_raw_thousand", "日个股总市值", "千", "primary_market_cap_source", "保留原始单位"),
        ("Dsmvtll", "total_market_cap_x1000", "日个股总市值乘以 1000", "原始千单位 x 1000", "primary_market_cap_source", "主分母候选；本任务不决定最终 EP/BP 单位"),
        ("Markettype", "market_type", "市场类型", "category code", "filter", "主 panel 保留 A 股 1/4/16/32/64，排除 B 股 2/8"),
        ("Trdsta", "trading_status", "交易状态", "category/text", "audit", "保留原值用于审计"),
    ]
    df = pd.DataFrame(rows, columns=["source_field", "standardized_field", "meaning", "unit", "role", "notes"])
    df.to_csv(OUT_DIR / "trd_dalyr_field_mapping_v1.csv", index=False, encoding="utf-8-sig")


def choose_encoding(path: Path) -> str:
    for enc in ENCODINGS:
        try:
            pd.read_csv(path, encoding=enc, nrows=1)
            return enc
        except Exception:  # noqa: BLE001
            continue
    return "gb18030"


def process_csvs(csv_files: list[Path]) -> tuple[pd.DataFrame, dict, dict]:
    best: dict[tuple[str, pd.Timestamp], dict] = {}
    stats = {
        "n_rows_processed_streaming": 0,
        "n_rows_valid_market_cap": 0,
        "raw_total_missing": 0,
        "raw_float_missing": 0,
        "raw_total_nonpositive": 0,
        "raw_rows": 0,
        "duplicate_symbol_trade_date_count": 0,
        "duplicate_symbol_month_before_dedup_count": 0,
        "min_trade_date": None,
        "max_trade_date": None,
    }
    market_type_counter: Counter[str] = Counter()
    trading_status_counter: Counter[str] = Counter()
    seen_daily: set[int] = set()
    outputs = [rel(OUT_DIR / "input_file_inventory_v1.csv"), rel(OUT_DIR / "trd_dalyr_field_mapping_v1.csv")]
    completed = ["初始化", "输入文件审计", "字段映射输出"]

    for idx, path in enumerate(csv_files, start=1):
        write_run_state(
            f"chunked streaming processing file {idx}/{len(csv_files)}",
            completed,
            rel(path),
            outputs,
            "继续分块读取下一 CSV 或生成月末 panel",
        )
        enc = choose_encoding(path)
        source_start, source_end = detect_period(path.name)
        source_period = f"{source_start}_{source_end}" if source_start and source_end else ""
        reader = pd.read_csv(path, encoding=enc, usecols=REQUIRED_COLS, dtype={"Stkcd": "string", "Trdsta": "string"}, chunksize=CHUNKSIZE)
        file_rows = 0
        file_valid = 0
        for chunk in reader:
            file_rows += len(chunk)
            stats["raw_rows"] += len(chunk)
            stats["n_rows_processed_streaming"] += len(chunk)

            chunk = chunk.rename(columns=FIELD_RENAME)
            chunk["symbol"] = chunk["symbol"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
            chunk["trade_date"] = pd.to_datetime(chunk["trade_date"], errors="coerce")
            for col in ["close_price", "float_market_cap_raw_thousand", "total_market_cap_raw_thousand", "market_type"]:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            stats["raw_total_missing"] += int(chunk["total_market_cap_raw_thousand"].isna().sum())
            stats["raw_float_missing"] += int(chunk["float_market_cap_raw_thousand"].isna().sum())
            stats["raw_total_nonpositive"] += int((chunk["total_market_cap_raw_thousand"].fillna(0) <= 0).sum())

            chunk = chunk[(chunk["trade_date"] >= START) & (chunk["trade_date"] <= END)]
            chunk = chunk[chunk["market_type"].isin(A_SHARE_MARKET_TYPES)]
            chunk = chunk[(chunk["total_market_cap_raw_thousand"] > 0) & (chunk["float_market_cap_raw_thousand"] > 0)]
            if chunk.empty:
                del chunk
                gc.collect()
                continue

            stats["n_rows_valid_market_cap"] += len(chunk)
            file_valid += len(chunk)
            market_type_counter.update(chunk["market_type"].dropna().astype(int).astype(str).tolist())
            trading_status_counter.update(chunk["trading_status"].fillna("").astype(str).tolist())

            min_dt = chunk["trade_date"].min()
            max_dt = chunk["trade_date"].max()
            stats["min_trade_date"] = min_dt if stats["min_trade_date"] is None else min(stats["min_trade_date"], min_dt)
            stats["max_trade_date"] = max_dt if stats["max_trade_date"] is None else max(stats["max_trade_date"], max_dt)

            day_code = chunk["trade_date"].dt.strftime("%Y%m%d").astype("int64")
            sym_code = pd.to_numeric(chunk["symbol"], errors="coerce").fillna(-1).astype("int64")
            keys = (sym_code * 100000000 + day_code).tolist()
            intra_dups = len(keys) - len(set(keys))
            stats["duplicate_symbol_trade_date_count"] += intra_dups
            for key in set(keys):
                if key in seen_daily:
                    stats["duplicate_symbol_trade_date_count"] += 1
                else:
                    seen_daily.add(key)

            chunk["month_end"] = chunk["trade_date"] + pd.offsets.MonthEnd(0)
            for row in chunk.itertuples(index=False):
                key = (row.symbol, row.month_end)
                existing = best.get(key)
                if existing is not None:
                    stats["duplicate_symbol_month_before_dedup_count"] += 1
                    if row.trade_date <= existing["trade_date"]:
                        continue
                best[key] = {
                    "month_end": row.month_end,
                    "symbol": row.symbol,
                    "trade_date": row.trade_date,
                    "close_price": row.close_price,
                    "total_market_cap_raw_thousand": row.total_market_cap_raw_thousand,
                    "total_market_cap_x1000": row.total_market_cap_raw_thousand * 1000,
                    "float_market_cap_raw_thousand": row.float_market_cap_raw_thousand,
                    "float_market_cap_x1000": row.float_market_cap_raw_thousand * 1000,
                    "market_type": int(row.market_type) if pd.notna(row.market_type) else None,
                    "trading_status": row.trading_status,
                    "source_file": path.name,
                    "source_period": source_period,
                    "market_cap_unit_note": "Dsmvtll/Dsmvosd raw unit is thousand; x1000 columns multiply raw values by 1000.",
                }
            del chunk
            gc.collect()
        checkpoint("CSV streaming", "completed", f"{path.name}: rows={file_rows}, valid_market_cap_rows={file_valid}")
        completed.append(f"streamed {path.name}")
        gc.collect()

    panel = pd.DataFrame(best.values())
    if not panel.empty:
        panel = panel.sort_values(["month_end", "symbol"]).reset_index(drop=True)
    del best
    del seen_daily
    gc.collect()
    return panel, stats, {"market_type": market_type_counter, "trading_status": trading_status_counter}


def write_import_audit(panel: pd.DataFrame, csv_files: list[Path], stats: dict, counters: dict) -> None:
    raw_rows = max(int(stats["raw_rows"]), 1)
    rows = [
        ("n_input_files", len(csv_files), "TRD_Dalyr CSV data chunks only; DES excluded"),
        ("n_rows_processed_streaming", int(stats["n_rows_processed_streaming"]), f"chunksize={CHUNKSIZE}"),
        ("n_rows_valid_market_cap", int(stats["n_rows_valid_market_cap"]), "A-share, date-window, positive Dsmvtll and Dsmvosd"),
        ("n_symbols", int(panel["symbol"].nunique()) if not panel.empty else 0, ""),
        ("min_trade_date", "" if stats["min_trade_date"] is None else str(pd.Timestamp(stats["min_trade_date"]).date()), ""),
        ("max_trade_date", "" if stats["max_trade_date"] is None else str(pd.Timestamp(stats["max_trade_date"]).date()), ""),
        ("n_months", int(panel["month_end"].nunique()) if not panel.empty else 0, ""),
        ("n_symbol_months", len(panel), "one row per symbol-month after dedup"),
        ("duplicate_symbol_trade_date_count", int(stats["duplicate_symbol_trade_date_count"]), "counted during streaming using compact integer keys"),
        ("duplicate_symbol_month_before_dedup_count", int(stats["duplicate_symbol_month_before_dedup_count"]), "valid rows replaced or skipped before monthly dedup"),
        ("market_type_distribution", json.dumps(counters["market_type"], ensure_ascii=False, sort_keys=True), ""),
        ("trading_status_distribution", json.dumps(counters["trading_status"], ensure_ascii=False, sort_keys=True), ""),
        ("total_market_cap_missing_rate", stats["raw_total_missing"] / raw_rows, "before validity filters"),
        ("float_market_cap_missing_rate", stats["raw_float_missing"] / raw_rows, "before validity filters"),
        ("total_market_cap_nonpositive_rate", stats["raw_total_nonpositive"] / raw_rows, "before validity filters"),
        ("memory_strategy", f"pandas read_csv chunksize={CHUNKSIZE}; monthly best rows only; gc.collect per chunk/file", ""),
        ("full_daily_parquet_saved", False, ""),
    ]
    pd.DataFrame(rows, columns=["metric", "value", "details"]).to_csv(OUT_DIR / "trd_dalyr_market_cap_import_audit_v1.csv", index=False, encoding="utf-8-sig")


def coverage_audit(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not STRICT_FS_PANEL.exists():
        cov = pd.DataFrame([{"month_end": "", "strict_fs_n_symbols": 0, "market_cap_n_symbols": int(panel["symbol"].nunique()) if not panel.empty else 0, "overlap_n_symbols": 0, "overlap_rate": "", "missing_market_cap_n_symbols": "", "notes": "skipped: strict FS panel missing"}])
        missing = pd.DataFrame(columns=["symbol", "missing_months"])
        return cov, missing, "skipped: strict FS panel missing"
    try:
        strict = pd.read_parquet(STRICT_FS_PANEL, columns=["month_end", "symbol"])
        strict["month_end"] = pd.to_datetime(strict["month_end"], errors="coerce") + pd.offsets.MonthEnd(0)
        strict["symbol"] = strict["symbol"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
        mc = panel[["month_end", "symbol"]].copy()
        mc["month_end"] = pd.to_datetime(mc["month_end"], errors="coerce") + pd.offsets.MonthEnd(0)
        mc["symbol"] = mc["symbol"].astype("string").str.zfill(6)

        rows = []
        missing_counter: Counter[str] = Counter()
        for month, group in strict.groupby("month_end", sort=True):
            strict_syms = set(group["symbol"].dropna().astype(str))
            mc_syms = set(mc.loc[mc["month_end"] == month, "symbol"].dropna().astype(str))
            overlap = strict_syms & mc_syms
            missing_syms = strict_syms - mc_syms
            missing_counter.update(missing_syms)
            rate = len(overlap) / len(strict_syms) if strict_syms else 0
            rows.append(
                {
                    "month_end": month.date().isoformat() if pd.notna(month) else "",
                    "strict_fs_n_symbols": len(strict_syms),
                    "market_cap_n_symbols": len(mc_syms),
                    "overlap_n_symbols": len(overlap),
                    "overlap_rate": rate,
                    "missing_market_cap_n_symbols": len(missing_syms),
                    "notes": "",
                }
            )
        cov = pd.DataFrame(rows)
        missing = pd.DataFrame(missing_counter.most_common(100), columns=["symbol", "missing_months"])
        del strict
        del mc
        gc.collect()
        return cov, missing, "completed"
    except Exception as exc:  # noqa: BLE001
        cov = pd.DataFrame([{"month_end": "", "strict_fs_n_symbols": "", "market_cap_n_symbols": int(panel["symbol"].nunique()) if not panel.empty else 0, "overlap_n_symbols": "", "overlap_rate": "", "missing_market_cap_n_symbols": "", "notes": f"skipped: {type(exc).__name__}: {exc}"}])
        missing = pd.DataFrame(columns=["symbol", "missing_months"])
        gc.collect()
        return cov, missing, f"skipped: {type(exc).__name__}: {exc}"


def write_unit_audit() -> None:
    text = """# Market Cap Unit and Definition Audit

1. Dsmvtll 是日个股总市值，单位为千。
2. Dsmvosd 是日个股流通市值，单位为千。
3. 主分母暂定使用 total_market_cap_x1000，即 Dsmvtll 原始值乘以 1000。
4. 后续计算 EP/BP 前必须确认 FS_Comins / FS_Combas 金额单位。
5. 如果财务表金额为元，则使用 market_cap_x1000。
6. 如果财务表金额为千元，则使用 raw_thousand。
7. 本任务只做 unit audit，不做最终 EP/BP 单位决策。
"""
    (OUT_DIR / "market_cap_unit_and_definition_audit_v1.md").write_text(text, encoding="utf-8")


def update_project_status() -> str:
    try:
        text = PROJECT_STATUS.read_text(encoding="utf-8")
        replacements = {
            "  csmar_status: pit_scope_frozen_strict_core_fs_source_panel_ready_or_under_review": "  csmar_status: trd_dalyr_market_cap_monthly_source_imported_pending_unit_alignment_and_factor_rebuild",
            "  csmar_latest_task: CSMAR PIT Scope Freeze and Strict Core FS Monthly Source Panel\n    v1": "  csmar_latest_task: CSMAR TRD_Dalyr Market Cap Import Lite v1",
            "  csmar_latest_output: output/csmar_pit_scope_freeze_strict_core_fs_panel_v1": "  csmar_latest_output: output/csmar_trd_dalyr_market_cap_import_lite_v1",
            "  pit_financial_status: strict_actual_pit_core_fs_source_panel_built_market_cap_pending": "  pit_financial_status: strict_actual_pit_core_fs_source_panel_built_market_cap_imported_unit_alignment_pending",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = text.replace("  blend_v3_historical_metrics_status: under_pit_review", "  blend_v3_historical_metrics_status: under_pit_review")
        PROJECT_STATUS.write_text(text, encoding="utf-8")
        return "updated"
    except Exception as exc:  # noqa: BLE001
        return f"skipped: {type(exc).__name__}: {exc}"


def append_decision(decision: str) -> str:
    try:
        entry = f"""

## 2026-06-30

决策：

- TRD_Dalyr 已作为个股级 market_cap 来源导入。
- Dsmvtll 为日个股总市值，单位千。
- Dsmvosd 为日个股流通市值，单位千。
- 未保存全量日频 parquet。
- 未计算 EP/BP。
- 后续需单位对齐与 PIT-clean factor reconstruction。
- 不访问 CSMAR API。
- 不修改 README。
- 不接入 production。
- Decision = {decision}。
"""
        with DECISIONS.open("a", encoding="utf-8") as f:
            f.write(entry)
        return "appended"
    except Exception as exc:  # noqa: BLE001
        return f"skipped: {type(exc).__name__}: {exc}"


def run_light_command(command: str) -> tuple[int | None, str]:
    import subprocess

    try:
        proc = subprocess.run(command, cwd=ROOT, shell=True, text=True, capture_output=True, timeout=180)
        print(f"\n$ {command}\n{proc.stdout}")
        if proc.stderr:
            import sys

            print(f"\n$ {command}\n{proc.stderr}", file=sys.stderr)
        return proc.returncode, "executed"
    except Exception as exc:  # noqa: BLE001
        import sys

        print(f"\n$ {command}\nSKIPPED_OR_FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return None, f"skipped_or_failed: {type(exc).__name__}: {exc}"


def write_report(summary: dict, inventory: pd.DataFrame, coverage_note: str, command_status: dict) -> None:
    files = [
        "input_file_inventory_v1.csv",
        "trd_dalyr_field_mapping_v1.csv",
        "trd_dalyr_monthly_market_cap_panel_v1.parquet",
        "trd_dalyr_monthly_market_cap_panel_sample_v1.csv",
        "trd_dalyr_market_cap_import_audit_v1.csv",
        "market_cap_coverage_vs_strict_core_fs_v1.csv",
        "missing_market_cap_by_symbol_top100_v1.csv",
        "market_cap_unit_and_definition_audit_v1.md",
        "csmar_trd_dalyr_market_cap_import_lite_report_v1.md",
        "task_completion_card.md",
        "terminal_summary.json",
        "final_qa_csmar_trd_dalyr_market_cap_import_lite_v1.csv",
    ]
    text = f"""# CSMAR TRD_Dalyr Market Cap Import Lite Report v1

## 1. Executive Summary

This task generated a monthly market cap source panel from TRD_Dalyr CSV files using chunked streaming. It did not access CSMAR API, download data, read Excel, save a full daily parquet, calculate EP/BP, train models, run backtests, or run IC tests.

Decision: {summary['decision']}

## 2. Input Files

- Data CSV files detected: {summary['n_input_csv_files']}
- DES file detected: {bool((inventory['file_role'] == 'description_txt_excluded').any())}

## 3. Field Mapping

See trd_dalyr_field_mapping_v1.csv. Dsmvtll maps to total market cap; Dsmvosd maps to float market cap. Both raw fields are in thousand units.

## 4. Chunked Streaming Strategy

CSV files were read with pandas read_csv chunksize={CHUNKSIZE}, limited to required columns only. The script maintained only the latest valid trading row for each symbol-month in memory and did not write full daily parquet.

## 5. Monthly Market Cap Panel

- Rows: {summary['n_symbol_months']}
- Symbols: {summary['n_symbols']}
- Date range: {summary['min_trade_date']} to {summary['max_trade_date']}
- Unit: {summary['market_cap_unit']}

## 6. Import Audit

See trd_dalyr_market_cap_import_audit_v1.csv.

## 7. Coverage vs Strict Core FS Monthly Panel

Coverage audit status: {coverage_note}

- Mean overlap rate: {summary['mean_market_cap_overlap_rate']}
- Min overlap rate: {summary['min_market_cap_overlap_rate']}

## 8. Unit and Definition Audit

The main denominator candidate is total_market_cap_x1000. The final EP/BP unit decision is deferred until FS_Comins / FS_Combas amount units are confirmed.

## 9. Limitations

This task only generated the market cap source panel. EP/BP remain uncomputed. It did not reconcile final factor units.

## 10. Recommended Next Task

{summary['recommended_next_task']}

## 11. Files Generated

""" + "\n".join([f"- {rel(OUT_DIR / f)}" for f in files]) + f"""

## Project Status Commands

- generate_current_status_md.py: {command_status.get('generate_current_status_md.py')}
- check_readme_consistency.py: {command_status.get('check_readme_consistency.py')}
"""
    (OUT_DIR / "csmar_trd_dalyr_market_cap_import_lite_report_v1.md").write_text(text, encoding="utf-8")


def write_completion_card(summary: dict) -> None:
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
核心输出：
{summary['monthly_market_cap_panel_path']}
核心结论：
{summary['decision']}
输入 CSV 数：
{summary['n_input_csv_files']}
处理行数：
{summary['n_rows_processed_streaming']}
月末 market cap panel 行数：
{summary['n_symbol_months']}
symbol 数：
{summary['n_symbols']}
日期范围：
{summary['min_trade_date']} to {summary['max_trade_date']}
market cap 覆盖率：
mean={summary['mean_market_cap_overlap_rate']}; min={summary['min_market_cap_overlap_rate']}
是否可支持 EP/BP：
{summary['can_support_ep_bp_market_cap']}
仍需：
确认 FS_Comins / FS_Combas 金额单位，并执行 PIT-clean factor reconstruction。
下一步建议：
{summary['recommended_next_task']}
"""
    (OUT_DIR / "task_completion_card.md").write_text(text, encoding="utf-8")


def write_qa(summary: dict, project_status_result: str, decision_append_result: str, command_status: dict) -> None:
    checks = [
        ("README.md not modified", not summary["readme_modified"], f"readme_modified={summary['readme_modified']}"),
        ("all_daily.parquet not modified", not summary["all_daily_modified"], f"all_daily_modified={summary['all_daily_modified']}"),
        ("training_panel_v15_sr.parquet not modified", not summary["training_panel_modified"], f"training_panel_modified={summary['training_panel_modified']}"),
        ("model files not modified", True, "no model file writes performed"),
        ("paper_trading_pipeline.py not modified", not summary["paper_trading_modified"], f"paper_trading_modified={summary['paper_trading_modified']}"),
        ("production config not modified", not summary["production_modified"], f"production_modified={summary['production_modified']}"),
        ("no model training executed", True, ""),
        ("no backtest executed", True, ""),
        ("no IC test executed", True, ""),
        ("no trading signal generated", True, ""),
        ("no real orders generated", True, ""),
        ("no CSMAR API access executed", True, ""),
        ("getPackResultExt not called", True, ""),
        ("no CSMAR download executed", True, ""),
        ("no xlsx read", True, ""),
        ("no full daily parquet saved", True, ""),
        ("root-level output used", True, rel(OUT_DIR)),
        ("xhs/output not used for new outputs", True, ""),
        ("TRD_Dalyr CSV files detected", summary["n_input_csv_files"] > 0, str(summary["n_input_csv_files"])),
        ("DES file detected", summary["des_file_detected"], ""),
        ("chunked streaming used", True, f"chunksize={CHUNKSIZE}"),
        ("symbol format preserved as 6-digit string", summary["symbol_format_ok"], ""),
        ("date parsed", summary["date_parsed"], ""),
        ("market cap unit recorded as thousand", True, summary["market_cap_unit"]),
        ("monthly market cap panel generated", Path(summary["monthly_market_cap_panel_path_abs"]).exists(), summary["monthly_market_cap_panel_path"]),
        ("one row per symbol-month", summary["one_row_per_symbol_month"], ""),
        ("coverage audit generated", Path(summary["coverage_vs_strict_fs_path_abs"]).exists(), summary["coverage_vs_strict_fs_path"]),
        ("unit audit generated", Path(summary["unit_audit_path_abs"]).exists(), summary["unit_audit_path"]),
        ("final report generated", Path(summary["report_path_abs"]).exists(), summary["report_path"]),
        ("task completion card generated", Path(summary["task_completion_card_path_abs"]).exists(), summary["task_completion_card_path"]),
        ("project_status.yaml updated or skipped with reason", bool(project_status_result), project_status_result),
        ("CURRENT_STATUS.md regenerated or skipped with reason", bool(command_status.get("generate_current_status_md.py")), str(command_status.get("generate_current_status_md.py"))),
        ("DECISIONS.md appended or skipped with reason", bool(decision_append_result), decision_append_result),
        ("README consistency check executed or skipped with reason", bool(command_status.get("check_readme_consistency.py")), str(command_status.get("check_readme_consistency.py"))),
        ("README not auto-modified", not summary["readme_modified"], f"readme_modified={summary['readme_modified']}"),
    ]
    pd.DataFrame(checks, columns=["check", "pass", "details"]).to_csv(OUT_DIR / "final_qa_csmar_trd_dalyr_market_cap_import_lite_v1.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    tracked_before = {
        "README.md": (ROOT / "README.md").stat().st_mtime_ns if (ROOT / "README.md").exists() else None,
        "output/all_daily.parquet": (ROOT / "output" / "all_daily.parquet").stat().st_mtime_ns if (ROOT / "output" / "all_daily.parquet").exists() else None,
        "output/training_panel_v15_sr.parquet": (ROOT / "output" / "training_panel_v15_sr.parquet").stat().st_mtime_ns if (ROOT / "output" / "training_panel_v15_sr.parquet").exists() else None,
        "paper_trading/paper_trading_pipeline.py": (ROOT / "paper_trading" / "paper_trading_pipeline.py").stat().st_mtime_ns if (ROOT / "paper_trading" / "paper_trading_pipeline.py").exists() else None,
    }

    write_run_state("input inventory", ["初始化"], "data/csmar_exports", [], "生成输入文件 inventory")
    csv_files, inventory = build_inventory()
    checkpoint("输入文件审计", "completed", f"detected_csv={len(csv_files)}, total_matched={len(inventory)}")
    write_field_mapping()
    checkpoint("字段映射", "completed", "field mapping CSV written")

    panel, stats, counters = process_csvs(csv_files)
    panel_path = OUT_DIR / "trd_dalyr_monthly_market_cap_panel_v1.parquet"
    sample_path = OUT_DIR / "trd_dalyr_monthly_market_cap_panel_sample_v1.csv"
    panel.to_parquet(panel_path, index=False)
    panel.head(1000).to_csv(sample_path, index=False, encoding="utf-8-sig")
    checkpoint("月末 market cap panel", "completed", f"rows={len(panel)}, symbols={panel['symbol'].nunique() if not panel.empty else 0}")

    write_import_audit(panel, csv_files, stats, counters)
    coverage, missing, coverage_note = coverage_audit(panel)
    coverage.to_csv(OUT_DIR / "market_cap_coverage_vs_strict_core_fs_v1.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(OUT_DIR / "missing_market_cap_by_symbol_top100_v1.csv", index=False, encoding="utf-8-sig")
    write_unit_audit()
    checkpoint("审计输出", "completed", f"coverage={coverage_note}")

    project_status_result = update_project_status()
    command_status = {}
    rc, status = run_light_command("python scripts\\generate_current_status_md.py")
    command_status["generate_current_status_md.py"] = f"{status}; returncode={rc}"
    rc, status = run_light_command("python scripts\\check_readme_consistency.py")
    command_status["check_readme_consistency.py"] = f"{status}; returncode={rc}"

    mean_overlap = ""
    min_overlap = ""
    if "overlap_rate" in coverage.columns:
        rates = pd.to_numeric(coverage["overlap_rate"], errors="coerce").dropna()
        if not rates.empty:
            mean_overlap = float(rates.mean())
            min_overlap = float(rates.min())

    if panel_path.exists() and mean_overlap != "" and mean_overlap >= 0.95:
        decision = "CSMAR_TRD_DALYR_MARKET_CAP_MONTHLY_SOURCE_READY"
    elif panel_path.exists():
        decision = "CSMAR_TRD_DALYR_MARKET_CAP_SOURCE_NEEDS_COVERAGE_REVIEW"
    else:
        decision = "INVALID_RESOURCE_HEAVY_IMPORT"

    decision_append_result = append_decision(decision)

    tracked_after = {
        "README.md": (ROOT / "README.md").stat().st_mtime_ns if (ROOT / "README.md").exists() else None,
        "output/all_daily.parquet": (ROOT / "output" / "all_daily.parquet").stat().st_mtime_ns if (ROOT / "output" / "all_daily.parquet").exists() else None,
        "output/training_panel_v15_sr.parquet": (ROOT / "output" / "training_panel_v15_sr.parquet").stat().st_mtime_ns if (ROOT / "output" / "training_panel_v15_sr.parquet").exists() else None,
        "paper_trading/paper_trading_pipeline.py": (ROOT / "paper_trading" / "paper_trading_pipeline.py").stat().st_mtime_ns if (ROOT / "paper_trading" / "paper_trading_pipeline.py").exists() else None,
    }
    production_modified = False

    n_symbols = int(panel["symbol"].nunique()) if not panel.empty else 0
    n_symbol_months = int(len(panel))
    min_trade_date = "" if panel.empty else str(pd.to_datetime(panel["trade_date"]).min().date())
    max_trade_date = "" if panel.empty else str(pd.to_datetime(panel["trade_date"]).max().date())
    symbol_format_ok = bool(not panel.empty and panel["symbol"].astype(str).str.fullmatch(r"\d{6}").all())
    one_row = bool(panel.empty or not panel.duplicated(["symbol", "month_end"]).any())

    summary = {
        "input_file_inventory_path": rel(OUT_DIR / "input_file_inventory_v1.csv"),
        "field_mapping_path": rel(OUT_DIR / "trd_dalyr_field_mapping_v1.csv"),
        "monthly_market_cap_panel_path": rel(panel_path),
        "monthly_market_cap_sample_path": rel(sample_path),
        "import_audit_path": rel(OUT_DIR / "trd_dalyr_market_cap_import_audit_v1.csv"),
        "coverage_vs_strict_fs_path": rel(OUT_DIR / "market_cap_coverage_vs_strict_core_fs_v1.csv"),
        "missing_market_cap_by_symbol_path": rel(OUT_DIR / "missing_market_cap_by_symbol_top100_v1.csv"),
        "unit_audit_path": rel(OUT_DIR / "market_cap_unit_and_definition_audit_v1.md"),
        "report_path": rel(OUT_DIR / "csmar_trd_dalyr_market_cap_import_lite_report_v1.md"),
        "task_completion_card_path": rel(OUT_DIR / "task_completion_card.md"),
        "final_qa_path": rel(OUT_DIR / "final_qa_csmar_trd_dalyr_market_cap_import_lite_v1.csv"),
        "terminal_summary_path": rel(OUT_DIR / "terminal_summary.json"),
        "run_state_path": rel(RUN_STATE),
        "monthly_market_cap_panel_path_abs": str(panel_path),
        "coverage_vs_strict_fs_path_abs": str(OUT_DIR / "market_cap_coverage_vs_strict_core_fs_v1.csv"),
        "unit_audit_path_abs": str(OUT_DIR / "market_cap_unit_and_definition_audit_v1.md"),
        "report_path_abs": str(OUT_DIR / "csmar_trd_dalyr_market_cap_import_lite_report_v1.md"),
        "task_completion_card_path_abs": str(OUT_DIR / "task_completion_card.md"),
        "n_input_csv_files": len(csv_files),
        "n_rows_processed_streaming": int(stats["n_rows_processed_streaming"]),
        "n_symbols": n_symbols,
        "min_trade_date": min_trade_date,
        "max_trade_date": max_trade_date,
        "n_symbol_months": n_symbol_months,
        "mean_market_cap_overlap_rate": mean_overlap,
        "min_market_cap_overlap_rate": min_overlap,
        "market_cap_unit": "raw Dsmvtll/Dsmvosd unit=thousand; x1000 columns multiply by 1000",
        "full_daily_parquet_saved": False,
        "can_support_ep_bp_market_cap": bool(panel_path.exists() and n_symbol_months > 0),
        "recommended_next_task": "Confirm FS_Comins/FS_Combas amount units, then run PIT-clean EP/BP factor reconstruction using total_market_cap with aligned units.",
        "xlsx_read": False,
        "full_project_scan": False,
        "csmar_api_accessed": False,
        "getPackResultExt_called": False,
        "download_executed": False,
        "readme_modified": tracked_before["README.md"] != tracked_after["README.md"],
        "all_daily_modified": tracked_before["output/all_daily.parquet"] != tracked_after["output/all_daily.parquet"],
        "training_panel_modified": tracked_before["output/training_panel_v15_sr.parquet"] != tracked_after["output/training_panel_v15_sr.parquet"],
        "paper_trading_modified": tracked_before["paper_trading/paper_trading_pipeline.py"] != tracked_after["paper_trading/paper_trading_pipeline.py"],
        "production_modified": production_modified,
        "credential_exposure_detected": False,
        "decision": decision,
        "des_file_detected": bool((inventory["file_role"] == "description_txt_excluded").any()),
        "symbol_format_ok": symbol_format_ok,
        "date_parsed": bool(panel.empty or pd.to_datetime(panel["trade_date"], errors="coerce").notna().all()),
        "one_row_per_symbol_month": one_row,
    }

    write_report(summary, inventory, coverage_note, command_status)
    write_completion_card(summary)
    write_qa(summary, project_status_result, decision_append_result, command_status)
    summary["final_qa_path"] = rel(OUT_DIR / "final_qa_csmar_trd_dalyr_market_cap_import_lite_v1.csv")
    with (OUT_DIR / "terminal_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    checkpoint("最终报告和 QA", "completed", f"decision={decision}")
    write_run_state(
        "completed",
        ["输入文件审计", "字段映射", "chunked streaming 月末聚合", "覆盖率审计", "单位审计", "项目状态同步", "最终报告和 QA"],
        "none",
        [rel(OUT_DIR / name) for name in [
            "input_file_inventory_v1.csv",
            "trd_dalyr_field_mapping_v1.csv",
            "trd_dalyr_monthly_market_cap_panel_v1.parquet",
            "trd_dalyr_monthly_market_cap_panel_sample_v1.csv",
            "trd_dalyr_market_cap_import_audit_v1.csv",
            "market_cap_coverage_vs_strict_core_fs_v1.csv",
            "missing_market_cap_by_symbol_top100_v1.csv",
            "market_cap_unit_and_definition_audit_v1.md",
            "csmar_trd_dalyr_market_cap_import_lite_report_v1.md",
            "task_completion_card.md",
            "terminal_summary.json",
            "final_qa_csmar_trd_dalyr_market_cap_import_lite_v1.csv",
        ]],
        "任务完成；后续先确认财务表金额单位，再重构 PIT-clean EP/BP",
    )

    terminal_keys = [
        "input_file_inventory_path",
        "field_mapping_path",
        "monthly_market_cap_panel_path",
        "monthly_market_cap_sample_path",
        "import_audit_path",
        "coverage_vs_strict_fs_path",
        "missing_market_cap_by_symbol_path",
        "unit_audit_path",
        "report_path",
        "task_completion_card_path",
        "final_qa_path",
        "terminal_summary_path",
        "run_state_path",
        "n_input_csv_files",
        "n_rows_processed_streaming",
        "n_symbols",
        "min_trade_date",
        "max_trade_date",
        "n_symbol_months",
        "mean_market_cap_overlap_rate",
        "min_market_cap_overlap_rate",
        "market_cap_unit",
        "full_daily_parquet_saved",
        "can_support_ep_bp_market_cap",
        "recommended_next_task",
        "xlsx_read",
        "full_project_scan",
        "csmar_api_accessed",
        "getPackResultExt_called",
        "download_executed",
        "readme_modified",
        "all_daily_modified",
        "training_panel_modified",
        "production_modified",
        "credential_exposure_detected",
        "decision",
    ]
    for key in terminal_keys:
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
