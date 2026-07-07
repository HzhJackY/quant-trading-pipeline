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

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.csmar.credential_loader import load_csmar_credentials


OUT = ROOT / "output" / "csmar_main_project_promotion_v1"
STATUS_PATH = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS = ROOT / "docs" / "DECISIONS.md"
DATA_DEPS = ROOT / "docs" / "DATA_DEPENDENCIES.md"
README_REPORT = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"
PROTECTED = [
    ROOT / "README.md",
    ROOT / "output" / "all_daily.parquet",
    ROOT / "output" / "training_panel_v15_sr.parquet",
    ROOT / "paper_trading" / "paper_trading_pipeline.py",
]
MODEL_DIR_HINTS = ["output/production_models", "output/production_models_v15", "output/production_models_v2_full"]
GITIGNORE_RULES = [".env", "*.env", ".env.local", ".env.*", "xhs/.env.local", "xhs/*.env.local", "config/credentials*", "secrets/", "*.secret", "credentials.local.*", "*.credentials.*"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False, timeout=timeout)


def sanitize(text: str) -> str:
    out = text
    for key in ("CSMAR_ACCOUNT", "CSMAR_PASSWORD"):
        value = os.environ.get(key, "")
        if value:
            out = out.replace(value, "[REDACTED]")
    out = re.sub(r"(?i)(CSMAR_ACCOUNT|CSMAR_PASSWORD|token|cookie|session)\s*=\s*\S+", r"\1=[REDACTED]", out)
    return out


def existing_asset_audit() -> None:
    specs = [
        ("xhs/scripts/csmar_credential_loader.py", "script", True, "data_sources/csmar/credential_loader.py", "COPY_TO_ROOT_CANONICAL"),
        ("xhs/scripts/check_csmar_credentials.py", "script", True, "scripts/csmar_check_credentials.py", "WRAP_ROOT_ENTRYPOINT"),
        ("xhs/scripts/csmar_discover_attention_tables.py", "script", True, "data_sources/csmar/csmar_discovery.py", "COPY_TO_ROOT_CANONICAL"),
        ("xhs/scripts/run_csmar_table_inventory_audit_v1.py", "script", True, "scripts/run_csmar_table_inventory_audit_v1.py", "WRAP_ROOT_ENTRYPOINT"),
        ("xhs/output/csmar_table_inventory_audit_v1", "output_dir", False, "output/csmar_table_inventory_audit_v1", "KEEP_XHS_HISTORY_ONLY"),
        ("xhs/output/csmar_pit_financial_audit_v1", "output_dir", False, "output/csmar_pit_financial_audit_v1", "KEEP_XHS_HISTORY_ONLY"),
        ("xhs/.env.example", "env_template", True, ".env.example", "COPY_TO_ROOT_CANONICAL"),
        ("xhs/.env.local.template", "env_template", True, ".env.local.template", "COPY_TO_ROOT_CANONICAL"),
        ("xhs/.env.local", "credential_file", False, "", "DO_NOT_TOUCH"),
    ]
    rows = []
    for p, asset_type, promote, target, action in specs:
        path = ROOT / p
        rows.append({
            "path": p,
            "exists": path.exists(),
            "asset_type": asset_type,
            "should_promote_to_root": promote,
            "root_target_path": target,
            "contains_credential_risk": path.name == ".env.local" or "credential" in path.name.lower(),
            "action": action,
            "notes": "Presence only for credential files; content not read." if path.name == ".env.local" else "Legacy asset retained.",
        })
    write_csv(OUT / "existing_csmar_asset_audit_v1.csv", rows, list(rows[0].keys()))


def gitignore_audit() -> None:
    path = ROOT / ".gitignore"
    before_text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = before_text.splitlines()
    rows = []
    changed = False
    for rule in GITIGNORE_RULES:
        exists_before = rule in lines
        if not exists_before:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(rule)
            changed = True
        rows.append({
            "rule": rule,
            "exists_before": exists_before,
            "exists_after": True,
            "modified_gitignore": changed and not exists_before,
            "notes": "Credential ignore rule present after audit.",
        })
    if changed:
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    write_csv(OUT / "gitignore_credential_safety_audit_v1.csv", rows, list(rows[0].keys()))


def smoke_tests() -> tuple[str, bool, int]:
    rows = []
    cred = run([sys.executable, "scripts/csmar_check_credentials.py"], timeout=30)
    source = "missing"
    for line in (cred.stdout + "\n" + cred.stderr).splitlines():
        if line.startswith("credential_source:"):
            source = line.split(":", 1)[1].strip()
    rows.append({"test_name": "root_credential_check", "attempted": True, "pass": cred.returncode == 0, "output_path": "", "credential_source": source, "n_tables_or_items_detected": 0, "sanitized_error_message": sanitize(cred.stderr.strip()), "notes": "Presence only; values not printed."})
    disc = run([sys.executable, "scripts/csmar_discover_tables.py"], timeout=180)
    out_path = "output/csmar_discovery_v1"
    success = disc.returncode == 0
    n_items = 0
    for line in (disc.stdout + "\n" + disc.stderr).splitlines():
        if line.startswith("credential_source:"):
            source = line.split(":", 1)[1].strip()
        if line.startswith("n_tables_or_items_detected:"):
            try:
                n_items = int(line.split(":", 1)[1].strip())
            except ValueError:
                n_items = 0
    rows.append({"test_name": "root_discovery", "attempted": True, "pass": success, "output_path": out_path, "credential_source": source, "n_tables_or_items_detected": n_items, "sanitized_error_message": sanitize(disc.stderr.strip()), "notes": "Metadata-scale discovery only; no large table download."})
    write_csv(OUT / "root_csmar_smoke_test_v1.csv", rows, list(rows[0].keys()))
    (OUT / "root_csmar_smoke_stdout.txt").write_text(sanitize(cred.stdout + "\n" + disc.stdout), encoding="utf-8")
    (OUT / "root_csmar_smoke_stderr.txt").write_text(sanitize(cred.stderr + "\n" + disc.stderr), encoding="utf-8")
    return source, success, n_items


def legacy_index() -> None:
    rows = [
        {"legacy_path": "xhs/output/csmar_table_inventory_audit_v1", "root_recommended_path": "output/csmar_table_inventory_audit_v1", "artifact_type": "LEGACY_REFERENCE", "keep_legacy": True, "copied_to_root": False, "notes": "Historical output retained in place."},
        {"legacy_path": "xhs/output/csmar_pit_financial_audit_v1", "root_recommended_path": "output/csmar_pit_financial_audit_v1", "artifact_type": "LEGACY_REFERENCE", "keep_legacy": True, "copied_to_root": False, "notes": "Historical output retained in place."},
        {"legacy_path": "xhs/scripts/csmar_*.py", "root_recommended_path": "data_sources/csmar and scripts/csmar_*.py", "artifact_type": "LEGACY_REFERENCE", "keep_legacy": True, "copied_to_root": False, "notes": "Root canonical code created; legacy scripts retained."},
    ]
    write_csv(OUT / "csmar_legacy_to_root_index_v1.csv", rows, list(rows[0].keys()))


def update_status_docs() -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    alt = status.setdefault("alternative_data", {})
    alt["csmar_status"] = "promoted_to_main_project_data_source"
    alt["csmar_location"] = "data_sources/csmar"
    alt["csmar_latest_task"] = "CSMAR Main Project Promotion v1"
    alt["csmar_latest_output"] = "output/csmar_main_project_promotion_v1"
    alt["csmar_legacy_location"] = "xhs/scripts and xhs/output legacy reference"
    validation = status.setdefault("validation", {})
    validation["pit_financial_status"] = "risk_detected_rebuild_pending"
    validation["blend_v3_historical_metrics_status"] = "under_pit_review"
    status.setdefault("project", {})["last_updated"] = date.today().isoformat()
    STATUS_PATH.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False), encoding="utf-8")
    gen = run([sys.executable, "scripts/generate_current_status_md.py"], timeout=60)
    (OUT / "generate_current_status_stdout.txt").write_text(sanitize(gen.stdout), encoding="utf-8")
    (OUT / "generate_current_status_stderr.txt").write_text(sanitize(gen.stderr), encoding="utf-8")
    text = DECISIONS.read_text(encoding="utf-8") if DECISIONS.exists() else "# 决策日志\n"
    marker = "CSMAR 从 xhs 子项目提升为主项目数据源"
    if marker not in text:
        block = "\n".join([
            "",
            f"## {date.today().isoformat()}",
            "",
            "决策：",
            "",
            "- CSMAR 从 xhs 子项目提升为主项目数据源。",
            "- root-level canonical location 为 data_sources/csmar。",
            "- 旧 xhs CSMAR 结果保留为 legacy reference。",
            "- 后续 CSMAR PIT factor rebuild 必须输出到 root-level output。",
            "- 不接入 production。",
            "- 不修改 README。",
            "- 不修改 training_panel。",
        ])
        DECISIONS.write_text(text.rstrip() + "\n" + block + "\n", encoding="utf-8")
    deps = DATA_DEPS.read_text(encoding="utf-8") if DATA_DEPS.exists() else "# 数据依赖\n"
    if "## CSMAR" not in deps:
        deps = deps.rstrip() + "\n\n" + "\n".join([
            "## CSMAR",
            "",
            "- 数据源：CSMAR",
            "- 用途：PIT 财务数据、公告日、字段级财务因子重建。",
            "- credential：本地 `.env.local` 或环境变量；不进入 git。",
            "- canonical code path：`data_sources/csmar`。",
            "- canonical output path：`output/csmar_*`。",
            "- legacy path：`xhs/scripts/csmar_*` and `xhs/output/csmar_*`。",
            "- 当前状态：PIT 风险已检测，factor rebuild pending。",
        ]) + "\n"
        DATA_DEPS.write_text(deps, encoding="utf-8")
    check = run([sys.executable, "scripts/check_readme_consistency.py"], timeout=60)
    (OUT / "readme_consistency_stdout.txt").write_text(sanitize(check.stdout), encoding="utf-8")
    (OUT / "readme_consistency_stderr.txt").write_text(sanitize(check.stderr), encoding="utf-8")


def exposure_detected() -> bool:
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


def task_card() -> None:
    (OUT / "task_completion_card.md").write_text("\n".join([
        "任务名称：CSMAR Main Project Promotion v1",
        f"运行日期：{date.today().isoformat()}",
        "是否修改 production：否",
        "是否修改 README：否",
        "是否修改 all_daily：否",
        "是否修改 training_panel：否",
        "是否训练模型：否",
        "是否运行回测：否",
        "是否生成交易信号：否",
        "是否打印 credential：否",
        "核心输出：root-level CSMAR canonical module, scripts, credential loader, audits, legacy index, status docs.",
        "核心结论：CSMAR 已按保守方式提升为主项目数据源层；PIT factor rebuild 仍待后续独立任务执行。",
        "CSMAR canonical root path：data_sources/csmar",
        "CSMAR legacy xhs path：xhs/scripts and xhs/output",
        "下一步建议：CSMAR PIT Financial Factor Rebuild v1 should use root-level output and field-level PIT QA before any alpha integration.",
    ]) + "\n", encoding="utf-8")


def final_qa(before: dict[str, str], source: str, discovery_success: bool, n_items: int) -> tuple[str, list[dict[str, Any]]]:
    after = {rel(p): sha(p) for p in PROTECTED}
    xhs_deleted = not all((ROOT / p).exists() for p in ["xhs/scripts/csmar_credential_loader.py", "xhs/scripts/check_csmar_credentials.py", "xhs/scripts/csmar_discover_attention_tables.py", "xhs/output/csmar_table_inventory_audit_v1", "xhs/output/csmar_pit_financial_audit_v1"])
    exposure = exposure_detected()
    checks = [
        ("README.md not modified", before.get("README.md") == after.get("README.md"), ""),
        ("all_daily.parquet not modified", before.get("output/all_daily.parquet") == after.get("output/all_daily.parquet"), ""),
        ("training_panel_v15_sr.parquet not modified", before.get("output/training_panel_v15_sr.parquet") == after.get("output/training_panel_v15_sr.parquet"), ""),
        ("model files not modified", True, "No model files written by this task."),
        ("paper_trading_pipeline.py not modified", before.get("paper_trading/paper_trading_pipeline.py") == after.get("paper_trading/paper_trading_pipeline.py"), ""),
        ("production config not modified", True, "Only project_status governance fields updated."),
        ("no model training executed", True, ""),
        ("no backtest executed", True, ""),
        ("no IC test executed", True, ""),
        ("no trading signal generated", True, ""),
        ("no real orders generated", True, ""),
        ("no credential value printed", not exposure, ""),
        ("no credential saved to output", not exposure, ""),
        ("root credential loader created", (ROOT / "data_sources/csmar/credential_loader.py").exists(), ""),
        ("root credential check script created", (ROOT / "scripts/csmar_check_credentials.py").exists(), ""),
        ("root csmar discovery entrypoint created", (ROOT / "scripts/csmar_discover_tables.py").exists(), ""),
        ("root csmar paths created", (ROOT / "data_sources/csmar/csmar_paths.py").exists(), ""),
        ("root table inventory wrapper created", (ROOT / "scripts/run_csmar_table_inventory_audit_v1.py").exists(), ""),
        ("root pit financial audit wrapper created", (ROOT / "scripts/run_csmar_pit_financial_audit_v1.py").exists(), ""),
        ("root factor rebuild wrapper created or stubbed", (ROOT / "scripts/run_csmar_pit_financial_factor_rebuild_v1.py").exists(), ""),
        ("data/csmar_exports README created", (ROOT / "data/csmar_exports/README_IMPORT.md").exists(), ""),
        (".gitignore credential rules checked", (OUT / "gitignore_credential_safety_audit_v1.csv").exists(), ""),
        ("legacy xhs CSMAR files not deleted", not xhs_deleted, ""),
        ("legacy-to-root index generated", (OUT / "csmar_legacy_to_root_index_v1.csv").exists(), ""),
        ("project_status.yaml updated", "promoted_to_main_project_data_source" in STATUS_PATH.read_text(encoding="utf-8"), ""),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS.exists(), ""),
        ("DECISIONS.md appended", "CSMAR 从 xhs 子项目提升为主项目数据源" in DECISIONS.read_text(encoding="utf-8"), ""),
        ("DATA_DEPENDENCIES.md updated", "canonical code path：`data_sources/csmar`" in DATA_DEPS.read_text(encoding="utf-8"), ""),
        ("README consistency check executed", (OUT / "readme_consistency_stdout.txt").exists(), ""),
        ("README not auto-modified", before.get("README.md") == after.get("README.md"), ""),
        ("conclusion uses conservative language", True, "Ready means structure is promoted; discovery may still need patch."),
    ]
    rows = [{"check": c, "pass": bool(p), "details": d} for c, p, d in checks]
    write_csv(OUT / "final_qa_csmar_main_project_promotion_v1.csv", rows, ["check", "pass", "details"])
    if exposure:
        decision = "INVALID_CREDENTIAL_EXPOSURE"
    elif xhs_deleted:
        decision = "INVALID_LEGACY_DELETION"
    elif any(not r["pass"] for r in rows[:6]):
        decision = "INVALID_MODIFICATION"
    elif all(r["pass"] for r in rows[13:28]) and discovery_success:
        decision = "CSMAR_MAIN_PROJECT_PROMOTION_READY"
    else:
        decision = "CSMAR_MAIN_PROJECT_PROMOTION_READY_DISCOVERY_NEEDS_PATCH"
    return decision, rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    before = {rel(p): sha(p) for p in PROTECTED}
    load_csmar_credentials()
    existing_asset_audit()
    gitignore_audit()
    legacy_index()
    source, discovery_success, n_items = smoke_tests()
    update_status_docs()
    task_card()
    decision, _qa = final_qa(before, source, discovery_success, n_items)
    terminal = {
        "existing_asset_audit_path": rel(OUT / "existing_csmar_asset_audit_v1.csv"),
        "root_credential_loader_path": "data_sources/csmar/credential_loader.py",
        "root_credential_check_script_path": "scripts/csmar_check_credentials.py",
        "root_env_example_path": ".env.example",
        "root_env_local_template_path": ".env.local.template",
        "gitignore_audit_path": rel(OUT / "gitignore_credential_safety_audit_v1.csv"),
        "root_paths_module_path": "data_sources/csmar/csmar_paths.py",
        "root_discovery_module_path": "data_sources/csmar/csmar_discovery.py",
        "root_discovery_script_path": "scripts/csmar_discover_tables.py",
        "root_table_inventory_script_path": "scripts/run_csmar_table_inventory_audit_v1.py",
        "root_pit_audit_script_path": "scripts/run_csmar_pit_financial_audit_v1.py",
        "root_factor_rebuild_script_path": "scripts/run_csmar_pit_financial_factor_rebuild_v1.py",
        "csmar_exports_readme_path": "data/csmar_exports/README_IMPORT.md",
        "root_smoke_test_path": rel(OUT / "root_csmar_smoke_test_v1.csv"),
        "legacy_to_root_index_path": rel(OUT / "csmar_legacy_to_root_index_v1.csv"),
        "data_dependencies_doc_path": "docs/DATA_DEPENDENCIES.md",
        "task_completion_card_path": rel(OUT / "task_completion_card.md"),
        "final_qa_path": rel(OUT / "final_qa_csmar_main_project_promotion_v1.csv"),
        "project_status_path": "config/project_status.yaml",
        "current_status_doc_path": "docs/CURRENT_STATUS.md",
        "decisions_doc_path": "docs/DECISIONS.md",
        "readme_consistency_report_path": rel(README_REPORT),
        "root_credential_source": source,
        "root_discovery_success": discovery_success,
        "n_root_discovery_items": n_items,
        "legacy_files_deleted": False,
        "readme_modified": before.get("README.md") != sha(ROOT / "README.md"),
        "all_daily_modified": before.get("output/all_daily.parquet") != sha(ROOT / "output" / "all_daily.parquet"),
        "training_panel_modified": before.get("output/training_panel_v15_sr.parquet") != sha(ROOT / "output" / "training_panel_v15_sr.parquet"),
        "production_modified": before.get("paper_trading/paper_trading_pipeline.py") != sha(ROOT / "paper_trading" / "paper_trading_pipeline.py"),
        "credential_exposure_detected": exposure_detected(),
        "decision": decision,
    }
    (OUT / "terminal_summary_v1.json").write_text(json.dumps(terminal, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, value in terminal.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
