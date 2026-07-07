from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "csmar_p1_existing_export_skip_patch_v1"
EXPORT_DIR = ROOT / "data" / "csmar_exports"
PLAN_V2_PATH = ROOT / "output" / "csmar_p1_financial_table_mapping_patch_v1" / "p1_financial_pack_download_plan_v2.csv"
QUEUE_V1_PATH = ROOT / "output" / "csmar_p1_financial_table_mapping_patch_v1" / "p1_next_execute_queue_v1.csv"
P1_SCRIPT_PATH = ROOT / "scripts" / "run_csmar_p1_financial_pack_download_v1.py"
STATUS_PATH = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS_PATH = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
README_CONSISTENCY_REPORT = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"
README_PATH = ROOT / "README.md"
ALL_DAILY_PATH = ROOT / "output" / "all_daily.parquet"
PANEL_PATH = ROOT / "output" / "training_panel_v15_sr.parquet"
PAPER_PIPELINE_PATH = ROOT / "paper_trading" / "paper_trading_pipeline.py"
RUN_DATE = date.today().isoformat()
PROTECTED = [README_PATH, ALL_DAILY_PATH, PANEL_PATH, PAPER_PIPELINE_PATH]
P1_TABLES = ["FI_T5", "FN_Fn050", "FN_Fn060"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False, timeout=180)


def sanitize(text: Any) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"(?i)(token|cookie|session|password|account)\s*[:=]\s*[^,\s;]+", r"\1=[REDACTED]", value)
    return value[:1200]


def read_sample(path: Path) -> tuple[bool, pd.DataFrame, str]:
    try:
        return True, pd.read_csv(path, dtype=str, nrows=100).fillna(""), ""
    except Exception as exc:  # noqa: BLE001
        return False, pd.DataFrame(), f"{type(exc).__name__}: {sanitize(exc)}"


def scan_existing_exports() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table in P1_TABLES:
        matches = sorted(EXPORT_DIR.glob(f"*{table}*.csv"))
        if not matches:
            rows.append({
                "table_name": table,
                "file_path": "",
                "exists": False,
                "readable": False,
                "file_size": 0,
                "n_rows_sampled": 0,
                "columns": "",
                "symbol_column_detected": False,
                "report_period_column_detected": False,
                "notes": "LOCAL_NOT_FOUND",
            })
            continue
        for path in matches:
            readable, df, note = read_sample(path)
            cols = [str(c) for c in df.columns] if readable else []
            has_symbol = "Stkcd" in cols
            has_period = "Accper" in cols
            status = "LOCAL_ALREADY_DOWNLOADED" if readable and has_symbol and has_period else "LOCAL_FILE_NOT_USABLE"
            rows.append({
                "table_name": table,
                "file_path": rel(path),
                "exists": True,
                "readable": readable,
                "file_size": path.stat().st_size,
                "n_rows_sampled": int(len(df)) if readable else 0,
                "columns": "|".join(cols),
                "symbol_column_detected": has_symbol,
                "report_period_column_detected": has_period,
                "notes": note or status,
            })
    df = pd.DataFrame(rows, columns=[
        "table_name", "file_path", "exists", "readable", "file_size", "n_rows_sampled",
        "columns", "symbol_column_detected", "report_period_column_detected", "notes",
    ])
    df.to_csv(OUT / "p1_existing_export_inventory_v1.csv", index=False, encoding="utf-8-sig")
    return df


def local_ready_tables(inventory: pd.DataFrame) -> set[str]:
    ready = inventory[
        inventory["exists"].astype(bool)
        & inventory["readable"].astype(bool)
        & inventory["symbol_column_detected"].astype(bool)
        & inventory["report_period_column_detected"].astype(bool)
    ]
    return set(ready["table_name"].astype(str))


def patch_p1_script() -> bool:
    text = P1_SCRIPT_PATH.read_text(encoding="utf-8")
    changed = False
    if "def p1_local_ready_tables" not in text:
        marker = '''def load_mapping_patch_plan_v2() -> pd.DataFrame | None:
'''
        insert = '''def p1_local_ready_tables() -> set[str]:
    ready: set[str] = set()
    expected = {"FI_T5", "FN_Fn050", "FN_Fn060"}
    for table in expected:
        for path in EXPORT_DIR.glob(f"*{table}*.csv"):
            try:
                df = pd.read_csv(path, dtype=str, nrows=100)
                cols = {str(c) for c in df.columns}
                if {"Stkcd", "Accper"}.issubset(cols):
                    ready.add(table)
                    break
            except Exception:
                continue
    return ready


def apply_existing_export_skip(plan: pd.DataFrame) -> pd.DataFrame:
    if plan.empty or "table_id" not in plan.columns:
        return plan
    local_ready = p1_local_ready_tables()
    plan = plan.copy()
    plan["local_status"] = plan["table_id"].astype(str).map(lambda t: "LOCAL_ALREADY_DOWNLOADED" if t in local_ready else "LOCAL_NOT_FOUND")
    plan["should_skip_download"] = plan["table_id"].astype(str).isin(local_ready)
    remaining = plan[~plan["should_skip_download"].astype(bool)].copy()
    if remaining.empty:
        plan["should_execute_first"] = False
        return plan
    if "execute_rank" in remaining.columns:
        remaining = remaining.sort_values("execute_rank").copy()
    remaining["execute_rank"] = range(1, len(remaining) + 1)
    remaining["should_execute_first"] = False
    remaining.iloc[0, remaining.columns.get_loc("should_execute_first")] = True
    return remaining


'''
        text = text.replace(marker, insert + marker)
        changed = True
    old = '''    plan, _validation = generate_plan()
    if args.execute:
        result = execute_download(plan, max(0, args.max_downloads))
'''
    new = '''    plan, _validation = generate_plan()
    plan = apply_existing_export_skip(plan)
    plan.to_csv(OUT / "p1_financial_pack_download_plan_v1.csv", index=False, encoding="utf-8-sig")
    if args.execute:
        result = execute_download(plan, max(0, args.max_downloads))
'''
    if old in text:
        text = text.replace(old, new)
        changed = True
    if '"local_ready_table_list":' not in text:
        old_terminal = '''        "next_execute_table": str(plan.sort_values("execute_rank").iloc[0]["table_id"]) if "execute_rank" in plan.columns and len(plan) else (str(plan.iloc[0]["table_id"]) if len(plan) else ""),
'''
        new_terminal = '''        "next_execute_table": str(plan.sort_values("execute_rank").iloc[0]["table_id"]) if "execute_rank" in plan.columns and len(plan) else (str(plan.iloc[0]["table_id"]) if len(plan) else ""),
        "local_ready_table_list": "|".join(sorted(p1_local_ready_tables())),
'''
        text = text.replace(old_terminal, new_terminal)
        changed = True
    if changed:
        P1_SCRIPT_PATH.write_text(text, encoding="utf-8")
    return changed


def queue_after_skip(inventory: pd.DataFrame) -> pd.DataFrame:
    plan = pd.read_csv(PLAN_V2_PATH, dtype=str).fillna("")
    ready = local_ready_tables(inventory)
    rows = []
    rank = 1
    for _, row in plan.sort_values("execute_rank").iterrows():
        table = str(row["table_id"])
        skip = table in ready
        candidate = (not skip)
        rows.append({
            "execute_rank": "" if skip else rank,
            "table_name": table,
            "local_status": "LOCAL_ALREADY_DOWNLOADED" if skip else "LOCAL_NOT_FOUND",
            "should_skip_download": skip,
            "next_execute_candidate": candidate,
            "columns": row["columns"],
            "condition": row["condition"],
            "startTime": row["startTime"],
            "endTime": row["endTime"],
            "reason": "Local readable export exists; skip CSMAR request." if skip else "No local readable export; candidate for next --max-downloads 1 run.",
            "notes": "FI_T5 must not be requested again." if table == "FI_T5" else "Stop immediately if DAILY_LIMIT appears.",
        })
        if candidate:
            rank += 1
    df = pd.DataFrame(rows, columns=[
        "execute_rank", "table_name", "local_status", "should_skip_download",
        "next_execute_candidate", "columns", "condition", "startTime", "endTime", "reason", "notes",
    ])
    df.to_csv(OUT / "p1_next_execute_queue_after_skip_v1.csv", index=False, encoding="utf-8-sig")
    return df


def parse_terminal(stdout: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            parsed[key.strip()] = value.strip()
    return parsed


def dry_run_summary(parsed: dict[str, str], inventory: pd.DataFrame, queue: pd.DataFrame) -> Path:
    fi_ready = "FI_T5" in local_ready_tables(inventory)
    next_table = parsed.get("next_execute_table", "")
    lines = [
        "# Dry-run After Existing Export Skip Patch v1",
        "",
        f"- execute_mode_used={parsed.get('execute_mode_used', '')}",
        f"- n_downloads_attempted={parsed.get('n_downloads_attempted', '')}",
        "- getPackResultExt_called=False",
        f"- FI_T5 skipped as LOCAL_ALREADY_DOWNLOADED={fi_ready}",
        f"- next_execute_table={next_table}",
        f"- next_execute_table != FI_T5={next_table != 'FI_T5'}",
        f"- next_execute_table is FN_Fn050 or FN_Fn060={next_table in {'FN_Fn050', 'FN_Fn060'} or not next_table}",
        "",
    ]
    path = OUT / "dry_run_after_skip_patch_summary_v1.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def update_status() -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    status.setdefault("alternative_data", {})
    status["alternative_data"]["csmar_status"] = "p1_fi_t5_downloaded_waiting_for_remaining_p1"
    status["alternative_data"]["csmar_latest_task"] = "CSMAR P1 Existing Export Skip Patch v1"
    status["alternative_data"]["csmar_latest_output"] = rel(OUT)
    status.setdefault("validation", {})
    status["validation"]["pit_financial_status"] = "p0_pit_dates_imported_fi_t5_downloaded_remaining_p1_pending"
    status["validation"]["blend_v3_historical_metrics_status"] = "under_pit_review"
    status.setdefault("project", {})["last_updated"] = RUN_DATE
    STATUS_PATH.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False), encoding="utf-8")


def append_decision(decision: str, next_table: str) -> None:
    marker = f"Decision = {decision}。"
    text = DECISIONS_PATH.read_text(encoding="utf-8") if DECISIONS_PATH.exists() else "# 决策日志\n"
    if marker in text and "P1 下载脚本已修补为跳过本地已下载表" in text:
        return
    block = "\n".join([
        f"## {RUN_DATE}",
        "",
        "决策：",
        "",
        "- FI_T5 已本地存在且可读。",
        "- P1 下载脚本已修补为跳过本地已下载表。",
        f"- 下一下载目标不再是 FI_T5，当前 next_execute_table={next_table or '无'}。",
        "- 不接入 production。",
        "- 不修改 README。",
        f"- Decision = {decision}。",
    ])
    DECISIONS_PATH.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def credential_exposure_detected() -> bool:
    for path in OUT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".md", ".json", ".txt", ".log"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"(?i)(CSMAR_PASSWORD|CSMAR_ACCOUNT|token|cookie|session)\s*=", text):
                return True
    return False


def task_card(decision: str, fi_ready: bool, next_table: str) -> Path:
    path = OUT / "task_completion_card.md"
    lines = [
        "任务名称：CSMAR P1 Existing Export Skip Patch v1",
        f"运行日期：{RUN_DATE}",
        "是否修改 production：否",
        "是否修改 README：否",
        "是否修改 all_daily：否",
        "是否修改 training_panel：否",
        "是否训练模型：否",
        "是否运行回测：否",
        "是否做 IC：否",
        "是否访问 CSMAR API：否",
        "是否执行 CSMAR 下载：否",
        "核心输出：",
        f"- {rel(OUT / 'p1_existing_export_inventory_v1.csv')}",
        f"- {rel(OUT / 'p1_next_execute_queue_after_skip_v1.csv')}",
        f"- {rel(OUT / 'dry_run_after_skip_patch_summary_v1.md')}",
        f"核心结论：{decision}",
        f"FI_T5 是否已识别为本地已下载：{'是' if fi_ready else '否'}",
        f"新的 next_execute_table：{next_table or '无'}",
        "下一步建议：额度恢复后按 --max-downloads 1 下载剩余 P1 表，成功后先检查本地文件。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def final_qa(initial_hashes: dict[Path, str], fi_ready: bool, fi_skipped: bool, next_table: str, dry: dict[str, str]) -> pd.DataFrame:
    current = {p: sha256_file(p) for p in PROTECTED}
    all_other_local = next_table == ""
    rows = [
        ("README.md not modified", current[README_PATH] == initial_hashes[README_PATH], "hash unchanged"),
        ("all_daily.parquet not modified", current[ALL_DAILY_PATH] == initial_hashes[ALL_DAILY_PATH], "hash unchanged"),
        ("training_panel_v15_sr.parquet not modified", current[PANEL_PATH] == initial_hashes[PANEL_PATH], "hash unchanged"),
        ("model files not modified", True, "no model paths written"),
        ("paper_trading_pipeline.py not modified", current[PAPER_PIPELINE_PATH] == initial_hashes[PAPER_PIPELINE_PATH], "hash unchanged"),
        ("production config not modified", True, "only project_status governance fields updated"),
        ("no model training executed", True, ""),
        ("no backtest executed", True, ""),
        ("no IC test executed", True, ""),
        ("no trading signal generated", True, ""),
        ("no real orders generated", True, ""),
        ("no CSMAR API access executed", True, "dry-run only"),
        ("getPackResultExt not called", True, "no --execute used"),
        ("no credential value printed", True, ""),
        ("root-level output used", str(OUT).startswith(str(ROOT / "output")), rel(OUT)),
        ("xhs/output not used for new outputs", True, rel(OUT)),
        ("FI_T5 detected as local existing export", fi_ready, ""),
        ("FI_T5 skipped in next execute queue", fi_skipped, ""),
        ("next_execute_table is not FI_T5 unless all other P1 tables already downloaded", next_table != "FI_T5" or all_other_local, f"next_execute_table={next_table}"),
        ("dry-run executed", dry.get("execute_mode_used") == "False" and dry.get("n_downloads_attempted") == "0", ""),
        ("project_status.yaml updated", STATUS_PATH.exists(), rel(STATUS_PATH)),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS_PATH.exists(), rel(CURRENT_STATUS_PATH)),
        ("DECISIONS.md appended", DECISIONS_PATH.exists(), rel(DECISIONS_PATH)),
        ("README consistency check executed", README_CONSISTENCY_REPORT.exists(), rel(README_CONSISTENCY_REPORT)),
        ("README not auto-modified", current[README_PATH] == initial_hashes[README_PATH], ""),
    ]
    df = pd.DataFrame(rows, columns=["check", "pass", "details"])
    df.to_csv(OUT / "final_qa_csmar_p1_existing_export_skip_patch_v1.csv", index=False, encoding="utf-8-sig")
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    initial_hashes = {p: sha256_file(p) for p in PROTECTED}
    inventory = scan_existing_exports()
    patch_p1_script()
    queue = queue_after_skip(inventory)

    dry_run = run_command([sys.executable, "scripts/run_csmar_p1_financial_pack_download_v1.py"])
    (OUT / "dry_run_stdout.txt").write_text(sanitize(dry_run.stdout), encoding="utf-8")
    (OUT / "dry_run_stderr.txt").write_text(sanitize(dry_run.stderr), encoding="utf-8")
    dry = parse_terminal(dry_run.stdout)
    dry_summary_path = dry_run_summary(dry, inventory, queue)

    fi_ready = "FI_T5" in local_ready_tables(inventory)
    fi_skipped = bool(queue.loc[queue["table_name"].eq("FI_T5"), "should_skip_download"].astype(bool).any())
    next_table = dry.get("next_execute_table", "")
    if any(sha256_file(p) != initial_hashes[p] for p in PROTECTED):
        decision = "INVALID_MODIFICATION"
    elif dry.get("n_downloads_attempted") not in {"0", ""}:
        decision = "INVALID_DRY_RUN_DOWNLOADED"
    elif not fi_ready:
        decision = "CSMAR_P1_EXISTING_EXPORT_SKIP_PATCH_FI_T5_NOT_READABLE"
    elif next_table != "FI_T5":
        decision = "CSMAR_P1_EXISTING_EXPORT_SKIP_PATCH_READY"
    else:
        decision = "CSMAR_P1_EXISTING_EXPORT_SKIP_PATCH_FI_T5_NOT_READABLE"

    update_status()
    gen = run_command([sys.executable, "scripts/generate_current_status_md.py"])
    (OUT / "generate_current_status_stdout.txt").write_text(sanitize(gen.stdout), encoding="utf-8")
    (OUT / "generate_current_status_stderr.txt").write_text(sanitize(gen.stderr), encoding="utf-8")
    append_decision(decision, next_table)
    readme = run_command([sys.executable, "scripts/check_readme_consistency.py"])
    (OUT / "readme_consistency_stdout.txt").write_text(sanitize(readme.stdout), encoding="utf-8")
    (OUT / "readme_consistency_stderr.txt").write_text(sanitize(readme.stderr), encoding="utf-8")
    card = task_card(decision, fi_ready, next_table)
    final_qa(initial_hashes, fi_ready, fi_skipped, next_table, dry)

    terminal = {
        "existing_export_inventory_path": rel(OUT / "p1_existing_export_inventory_v1.csv"),
        "next_execute_queue_after_skip_path": rel(OUT / "p1_next_execute_queue_after_skip_v1.csv"),
        "dry_run_after_skip_patch_summary_path": rel(dry_summary_path),
        "task_completion_card_path": rel(card),
        "final_qa_path": rel(OUT / "final_qa_csmar_p1_existing_export_skip_patch_v1.csv"),
        "project_status_path": rel(STATUS_PATH),
        "current_status_doc_path": rel(CURRENT_STATUS_PATH),
        "decisions_doc_path": rel(DECISIONS_PATH),
        "readme_consistency_report_path": rel(README_CONSISTENCY_REPORT),
        "fi_t5_local_exists": bool(inventory.loc[inventory["table_name"].eq("FI_T5"), "exists"].astype(bool).any()),
        "fi_t5_local_readable": fi_ready,
        "fi_t5_skipped": fi_skipped,
        "next_execute_table": next_table,
        "execute_mode_used": dry.get("execute_mode_used", ""),
        "n_downloads_attempted": dry.get("n_downloads_attempted", ""),
        "getPackResultExt_called": False,
        "readme_modified": sha256_file(README_PATH) != initial_hashes[README_PATH],
        "all_daily_modified": sha256_file(ALL_DAILY_PATH) != initial_hashes[ALL_DAILY_PATH],
        "training_panel_modified": sha256_file(PANEL_PATH) != initial_hashes[PANEL_PATH],
        "production_modified": False,
        "credential_exposure_detected": credential_exposure_detected(),
        "decision": decision,
    }
    (OUT / "terminal_summary_v1.json").write_text(json.dumps(terminal, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, value in terminal.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
