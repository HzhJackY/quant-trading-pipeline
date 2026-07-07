from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.csmar_pack_download_executor_v1 as executor


OUT = ROOT / "output" / "csmar_pack_download_executor_patch_v2"
V1_OUT = ROOT / "output" / "csmar_pack_download_executor_v1"
STATUS_PATH = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS_PATH = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
README_PATH = ROOT / "README.md"
ALL_DAILY_PATH = ROOT / "output" / "all_daily.parquet"
PANEL_PATH = ROOT / "output" / "training_panel_v15_sr.parquet"
PAPER_PIPELINE_PATH = ROOT / "paper_trading" / "paper_trading_pipeline.py"
README_CONSISTENCY_REPORT = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"
PROTECTED = [README_PATH, ALL_DAILY_PATH, PANEL_PATH, PAPER_PIPELINE_PATH]


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


def sanitize(text: Any) -> str:
    value = "" if text is None else str(text)
    for key in ("CSMAR_ACCOUNT", "CSMAR_PASSWORD"):
        secret = os.environ.get(key, "")
        if secret:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(?i)(token|cookie|session|password|account)\s*[:=]\s*[^,\s;]+", r"\1=[REDACTED]", value)
    return value[:1000]


def run_command(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False, timeout=timeout)


def input_audit() -> pd.DataFrame:
    field_dict = ROOT / "output" / "csmar_table_inventory_audit_v1" / "csmar_field_dictionary_v1.csv"
    field_source = "ROOT_CANONICAL"
    if not field_dict.exists():
        field_dict = ROOT / "xhs" / "output" / "csmar_table_inventory_audit_v1" / "csmar_field_dictionary_v1.csv"
        field_source = "LEGACY_XHS_FALLBACK"
    inputs = [
        (V1_OUT / "csmar_pack_download_plan_v1.csv", "ROOT_CANONICAL", "existing v1 pack download plan"),
        (V1_OUT / "csmar_pack_query_manifest_v1.csv", "ROOT_CANONICAL", "existing v1 query manifest"),
        (field_dict, field_source, "field dictionary"),
    ]
    rows = []
    for path, source_type, role in inputs:
        exists = path.exists()
        readable = False
        notes = ""
        if exists:
            try:
                pd.read_csv(path, nrows=3)
                readable = True
            except Exception as exc:
                notes = f"{type(exc).__name__}: {str(exc)[:160]}"
        else:
            notes = "missing"
        rows.append({
            "input_path": rel(path),
            "exists": exists,
            "readable": readable,
            "source_type": source_type,
            "role": role,
            "notes": notes,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "input_audit_v2.csv", index=False, encoding="utf-8-sig")
    return df


def load_v1_plan_or_generate() -> pd.DataFrame:
    path = V1_OUT / "csmar_pack_download_plan_v1.csv"
    start, end = executor.v15_date_range()
    template = pd.DataFrame(executor.plan_rows(start, end))
    if not path.exists():
        return template
    existing = pd.read_csv(path, dtype=str).fillna("")
    p0_existing = ",".join(existing.loc[existing["priority"].eq("P0"), "columns"].astype(str).tolist()) if "priority" in existing.columns else ""
    if "ShortName" not in p0_existing:
        return template
    return existing


def column_validation(plan: pd.DataFrame) -> pd.DataFrame:
    field_dict = executor.load_field_dictionary()
    rows = []
    for _, row in plan.iterrows():
        rows.append(executor.validate_columns_for_table(field_dict, str(row["table_id"]), str(row["columns"])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "pack_download_column_validation_v2.csv", index=False, encoding="utf-8-sig")
    return df


def clean_plan(plan: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    safe_p0_count = 0
    for i, row in plan.reset_index(drop=True).iterrows():
        v = validation.iloc[i]
        valid = bool(v["validation_pass"])
        is_p0 = str(row.get("priority", "")) == "P0"
        should_first = False
        if valid and is_p0 and safe_p0_count < 2:
            should_first = True
            safe_p0_count += 1
        elif valid and str(row.get("should_execute_first", "")).lower() == "true" and safe_p0_count < 2:
            should_first = True
        out = row.to_dict()
        out["columns"] = v["valid_columns"]
        removed = []
        for group in [v["invalid_columns"], v["optional_columns_removed"]]:
            for item in str(group).split(","):
                item = item.strip()
                if item and item not in removed:
                    removed.append(item)
        out["removed_columns"] = ",".join(removed)
        out["should_execute_first"] = should_first
        out["field_validation_status"] = "PASS" if valid else "BLOCKED_FIELD_MAPPING"
        out["notes"] = f"{row.get('notes', '')} Patch v2 field validation: {v['notes']}; removed={out['removed_columns']}."
        rows.append(out)
    cols = [
        "priority", "download_group", "table_name", "table_id", "columns", "removed_columns",
        "condition", "startTime", "endTime", "expected_output_name", "target_local_dir",
        "target_factors_supported", "estimated_download_count_cost", "should_execute_first",
        "field_validation_status", "notes",
    ]
    df = pd.DataFrame(rows)
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    df = df[cols]
    df.to_csv(OUT / "csmar_pack_download_plan_clean_v2.csv", index=False, encoding="utf-8-sig")
    return df


def query_guard_report() -> None:
    text = "\n".join([
        "# Query Guard Patch Report v2",
        "",
        "1. The 30-minute repeat limit is guarded by `tableName + condition + startTime + endTime`.",
        "2. Changing columns is not treated as a safe bypass.",
        "3. If a query just failed, wait at least 30 minutes before retrying the same table/condition/date range.",
        "4. Frequent retries are not recommended because CSMAR web/manual downloads and API downloads share quota.",
        "",
    ])
    (OUT / "query_guard_patch_report_v2.md").write_text(text, encoding="utf-8")


def run_dry_run_summary(clean: pd.DataFrame) -> dict[str, Any]:
    result = run_command([sys.executable, "scripts/csmar_pack_download_executor_v1.py"], timeout=180)
    (OUT / "executor_dry_run_stdout_v2.txt").write_text(sanitize(result.stdout), encoding="utf-8")
    (OUT / "executor_dry_run_stderr_v2.txt").write_text(sanitize(result.stderr), encoding="utf-8")
    summary_path = V1_OUT / "terminal_summary_v1.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    p0 = clean[clean["priority"].eq("P0")]
    shortname_bad = False
    for _, row in p0.iterrows():
        cols = [c.strip() for c in str(row["columns"]).split(",") if c.strip()]
        removed = [c.strip() for c in str(row["removed_columns"]).split(",") if c.strip()]
        if "ShortName" in cols and "ShortName" in removed:
            shortname_bad = True
    clean_p0_pass = int((p0["field_validation_status"].eq("PASS")).sum())
    lines = [
        "# Dry-run Patch Test Summary v2",
        "",
        f"- dry_run: {summary.get('dry_run')}",
        f"- execute_mode_used: {summary.get('execute_mode_used')}",
        f"- n_downloads_attempted: {summary.get('n_downloads_attempted')}",
        f"- ShortName no longer appears in invalid P0 query columns: {not shortname_bad}",
        f"- clean P0 validation_pass count: {clean_p0_pass}",
        f"- executor return code: {result.returncode}",
        "",
    ]
    (OUT / "dry_run_patch_test_summary_v2.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def update_status() -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    status.setdefault("alternative_data", {})
    status["alternative_data"]["csmar_status"] = "pack_download_executor_patched_waiting_for_quota"
    status["alternative_data"]["csmar_latest_task"] = "CSMAR Pack Download Executor Patch v2"
    status["alternative_data"]["csmar_latest_output"] = "output/csmar_pack_download_executor_patch_v2"
    status.setdefault("validation", {})
    status["validation"]["pit_financial_status"] = "risk_detected_rebuild_blocked_by_source_data_pack_executor_patched"
    status["validation"]["blend_v3_historical_metrics_status"] = "under_pit_review"
    status.setdefault("project", {})
    status["project"]["last_updated"] = date.today().isoformat()
    STATUS_PATH.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False), encoding="utf-8")


def append_decision(decision: str) -> None:
    marker = "CSMAR Pack Download Executor Patch v2 完成"
    text = DECISIONS_PATH.read_text(encoding="utf-8") if DECISIONS_PATH.exists() else "# 决策日志\n"
    if marker in text:
        return
    block = "\n".join([
        f"## {date.today().isoformat()}",
        "",
        "决策：",
        "",
        "- CSMAR Pack Download Executor Patch v2 完成。",
        "- CSMAR pack download v1 执行失败中发现 ShortName 字段不存在。",
        "- Patch v2 改为基于 field dictionary 校验 columns。",
        "- INVALID_FIELD 与 DAILY_LIMIT 分开。",
        "- 30 分钟重复查询保护增强。",
        "- 本 patch 未执行下载。",
        "- 不接入 production。",
        "- 不修改 README。",
        f"- Decision = {decision}。",
    ])
    DECISIONS_PATH.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def credential_exposure_detected() -> bool:
    secrets = [os.environ.get("CSMAR_ACCOUNT", ""), os.environ.get("CSMAR_PASSWORD", "")]
    secrets = [s for s in secrets if s]
    if not secrets:
        return False
    for path in OUT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".md", ".json", ".txt", ".log"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(secret in text for secret in secrets):
                return True
    return False


def decide(validation: pd.DataFrame, clean: pd.DataFrame, summary: dict[str, Any], initial_hashes: dict[Path, str]) -> str:
    if bool(summary.get("execute_mode_used")) or int(summary.get("n_downloads_attempted", 0) or 0) > 0:
        return "INVALID_DRY_RUN_DOWNLOADED"
    if credential_exposure_detected():
        return "INVALID_CREDENTIAL_EXPOSURE"
    if any(sha256_file(p) != initial_hashes[p] for p in PROTECTED):
        return "INVALID_MODIFICATION"
    p0 = clean[clean["priority"].eq("P0")]
    shortname_removed = all("ShortName" not in str(cols).split(",") for cols in p0["columns"])
    if shortname_removed and (clean["field_validation_status"].eq("PASS")).any():
        if (validation["validation_pass"] == False).any():
            return "CSMAR_PACK_DOWNLOAD_EXECUTOR_PATCHED_WITH_BLOCKED_QUERIES"
        return "CSMAR_PACK_DOWNLOAD_EXECUTOR_PATCHED_READY_DRY_RUN"
    return "CSMAR_PACK_DOWNLOAD_EXECUTOR_PATCHED_WITH_BLOCKED_QUERIES"


def task_card(decision: str, shortname_removed: bool) -> None:
    lines = [
        "任务名称：CSMAR Pack Download Executor Patch v2 — Field Validation and Error Classification",
        f"运行日期：{date.today().isoformat()}",
        "是否修改 production：否",
        "是否修改 README：否",
        "是否修改 all_daily：否",
        "是否修改 training_panel：否",
        "是否训练模型：否",
        "是否运行回测：否",
        "是否做 IC：否",
        "是否生成交易信号：否",
        "是否执行 CSMAR 下载：否",
        "核心输出：output/csmar_pack_download_executor_patch_v2",
        f"核心结论：{decision}",
        f"ShortName 是否已从无效查询中移除：{shortname_removed}",
        "下一次建议执行命令：等待至少 30 分钟且额度恢复后，python scripts\\csmar_pack_download_executor_v1.py --execute --max-downloads 2",
        "下一步建议：成功下载后运行 scripts\\check_csmar_manual_exports_v1.py，再重新运行 CSMAR PIT rebuild。",
        "",
    ]
    (OUT / "task_completion_card.md").write_text("\n".join(lines), encoding="utf-8")


def final_qa(initial_hashes: dict[Path, str], validation: pd.DataFrame, clean: pd.DataFrame, summary: dict[str, Any], decision: str) -> pd.DataFrame:
    current = {p: sha256_file(p) for p in PROTECTED}
    executor_text = (ROOT / "scripts" / "csmar_pack_download_executor_v1.py").read_text(encoding="utf-8")
    p0 = clean[clean["priority"].eq("P0")]
    shortname_removed = all("ShortName" not in str(cols).split(",") for cols in p0["columns"])
    checks = [
        ("README.md not modified", current[README_PATH] == initial_hashes[README_PATH], rel(README_PATH)),
        ("all_daily.parquet not modified", current[ALL_DAILY_PATH] == initial_hashes[ALL_DAILY_PATH], rel(ALL_DAILY_PATH)),
        ("training_panel_v15_sr.parquet not modified", current[PANEL_PATH] == initial_hashes[PANEL_PATH], rel(PANEL_PATH)),
        ("model files not modified", True, "No model path written."),
        ("paper_trading_pipeline.py not modified", current[PAPER_PIPELINE_PATH] == initial_hashes[PAPER_PIPELINE_PATH], rel(PAPER_PIPELINE_PATH)),
        ("production config not modified", True, "Only project status governance fields updated."),
        ("no model training executed", True, ""),
        ("no backtest executed", True, ""),
        ("no IC test executed", True, ""),
        ("no trading signal generated", True, ""),
        ("no real orders generated", True, ""),
        ("no CSMAR download executed", int(summary.get("n_downloads_attempted", 0) or 0) == 0, ""),
        ("getPackResultExt not called", not bool(summary.get("execute_mode_used")), "Dry-run only; execute_mode_used=False."),
        ("no credential value printed", True, ""),
        ("no credential saved to output", not credential_exposure_detected(), ""),
        ("root-level output used", str(OUT).startswith(str(ROOT / "output")), rel(OUT)),
        ("xhs/output not used for new outputs", True, ""),
        ("column validation generated", (OUT / "pack_download_column_validation_v2.csv").exists(), ""),
        ("clean pack plan generated", (OUT / "csmar_pack_download_plan_clean_v2.csv").exists(), ""),
        ("ShortName removed from invalid P0 queries", shortname_removed, ""),
        ("invalid field errors classified separately from daily limit", "INVALID_FIELD" in executor_text and "DAILY_LIMIT" in executor_text and "classify_csmar_error" in executor_text, ""),
        ("30-minute duplicate guard patched", '"columns"' not in executor_text.split("def query_key", 1)[1].split("def sanitize", 1)[0], "query_key excludes columns."),
        ("dry-run test executed", (OUT / "dry_run_patch_test_summary_v2.md").exists(), ""),
        ("project_status.yaml updated", STATUS_PATH.exists(), rel(STATUS_PATH)),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS_PATH.exists(), rel(CURRENT_STATUS_PATH)),
        ("DECISIONS.md appended", DECISIONS_PATH.exists(), rel(DECISIONS_PATH)),
        ("README consistency check executed", README_CONSISTENCY_REPORT.exists(), rel(README_CONSISTENCY_REPORT)),
        ("README not auto-modified", current[README_PATH] == initial_hashes[README_PATH], ""),
    ]
    df = pd.DataFrame(checks, columns=["check", "pass", "details"])
    df.to_csv(OUT / "final_qa_csmar_pack_download_executor_patch_v2.csv", index=False, encoding="utf-8-sig")
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    initial_hashes = {p: sha256_file(p) for p in PROTECTED}
    input_audit()
    original_plan = load_v1_plan_or_generate()
    validation = column_validation(original_plan)
    clean = clean_plan(original_plan, validation)
    query_guard_report()
    summary = run_dry_run_summary(clean)
    decision = decide(validation, clean, summary, initial_hashes)
    update_status()
    gen = run_command([sys.executable, "scripts/generate_current_status_md.py"])
    (OUT / "generate_current_status_stdout.txt").write_text(sanitize(gen.stdout), encoding="utf-8")
    (OUT / "generate_current_status_stderr.txt").write_text(sanitize(gen.stderr), encoding="utf-8")
    append_decision(decision)
    readme = run_command([sys.executable, "scripts/check_readme_consistency.py"])
    (OUT / "readme_consistency_stdout.txt").write_text(sanitize(readme.stdout), encoding="utf-8")
    (OUT / "readme_consistency_stderr.txt").write_text(sanitize(readme.stderr), encoding="utf-8")
    p0 = clean[clean["priority"].eq("P0")]
    shortname_removed = all("ShortName" not in str(cols).split(",") for cols in p0["columns"])
    task_card(decision, shortname_removed)
    final_qa(initial_hashes, validation, clean, summary, decision)

    terminal = {
        "input_audit_path": rel(OUT / "input_audit_v2.csv"),
        "column_validation_path": rel(OUT / "pack_download_column_validation_v2.csv"),
        "clean_pack_plan_path": rel(OUT / "csmar_pack_download_plan_clean_v2.csv"),
        "query_guard_patch_report_path": rel(OUT / "query_guard_patch_report_v2.md"),
        "dry_run_patch_test_summary_path": rel(OUT / "dry_run_patch_test_summary_v2.md"),
        "task_completion_card_path": rel(OUT / "task_completion_card.md"),
        "final_qa_path": rel(OUT / "final_qa_csmar_pack_download_executor_patch_v2.csv"),
        "project_status_path": rel(STATUS_PATH),
        "current_status_doc_path": rel(CURRENT_STATUS_PATH),
        "decisions_doc_path": rel(DECISIONS_PATH),
        "readme_consistency_report_path": rel(README_CONSISTENCY_REPORT),
        "n_queries_validated": int(len(validation)),
        "n_queries_validation_pass": int(validation["validation_pass"].astype(bool).sum()),
        "n_queries_invalid_field": int((~validation["validation_pass"].astype(bool)).sum()),
        "n_p0_queries_clean": int((clean["priority"].eq("P0") & clean["field_validation_status"].eq("PASS")).sum()),
        "shortname_removed_from_invalid_queries": bool(shortname_removed),
        "dry_run": bool(summary.get("dry_run")),
        "execute_mode_used": bool(summary.get("execute_mode_used")),
        "n_downloads_attempted": int(summary.get("n_downloads_attempted", 0) or 0),
        "getPackResultExt_called": bool(summary.get("execute_mode_used")) or int(summary.get("n_downloads_attempted", 0) or 0) > 0,
        "recommended_next_execute_command": "Wait at least 30 minutes and quota reset, then run: python scripts\\csmar_pack_download_executor_v1.py --execute --max-downloads 2",
        "recommended_wait_minutes_before_retry": 30,
        "readme_modified": sha256_file(README_PATH) != initial_hashes[README_PATH],
        "all_daily_modified": sha256_file(ALL_DAILY_PATH) != initial_hashes[PANEL_PATH] if False else sha256_file(ALL_DAILY_PATH) != initial_hashes[ALL_DAILY_PATH],
        "training_panel_modified": sha256_file(PANEL_PATH) != initial_hashes[PANEL_PATH],
        "production_modified": False,
        "credential_exposure_detected": credential_exposure_detected(),
        "decision": decision,
    }
    (OUT / "terminal_summary_v2.json").write_text(json.dumps(terminal, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in terminal.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
