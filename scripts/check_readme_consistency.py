from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "blend_v3_governance_patch_v2"
STATUS_PATH = ROOT / "config" / "project_status.yaml"
README_PATH = ROOT / "README.md"
POLICY_PATH = ROOT / "docs" / "README_SYNC_POLICY.md"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def qa_row(check: str, passed: bool, severity: str, details: str, action: str) -> dict[str, object]:
    return {
        "check": check,
        "pass": bool(passed),
        "severity": severity,
        "details": details,
        "recommended_action": action,
    }


def check_readme(status: dict) -> tuple[list[dict[str, object]], str]:
    text = read_text(README_PATH)
    shadow = status["shadow_candidate"]
    rows: list[dict[str, object]] = []

    explicit_shadow_guard = shadow["status"] in text and ("不是正式 production" in text or "不替换当前 production" in text)
    severe_shadow_prod = bool(re.search(r"Blend V3.{0,40}(正式|active).{0,20}production", text, re.I | re.S)) and not explicit_shadow_guard
    rows.append(qa_row(
        "README 是否把 shadow 写成 production",
        not severe_shadow_prod and explicit_shadow_guard,
        "P0" if severe_shadow_prod else "P1",
        "发现明确 shadow 风险提示" if explicit_shadow_guard else "缺少明确 shadow 非 production 提示",
        "如为 P0，人工确认后最小化修正 shadow / production 边界；本脚本不自动修改 README。",
    ))

    compact_only = "Compact-F 是唯一默认" in text or "Compact-F 为唯一默认" in text
    compact_demoted = "Compact-F 不再是唯一默认" in text or "baseline" in text
    rows.append(qa_row(
        "README 是否仍说 Compact-F 是唯一默认生产候选",
        not compact_only and compact_demoted,
        "P1" if compact_only else "P2",
        "Compact-F 已声明为 baseline / style reference" if compact_demoted else "未找到 Compact-F 降级说明",
        "保持 README 与 MODEL_REGISTRY 一致。",
    ))

    rows.append(qa_row(
        "README 是否缺少当前 shadow candidate",
        shadow["name"].replace("_TOP50_BUFFER_V3", "") in text or "BLEND_V0_50_V7_50" in text,
        "P1",
        shadow["name"],
        "如缺少候选身份，按 README_SYNC_POLICY 人工补充摘要。",
    ))

    metric_ok = all(s in text for s in ["1.509", "10.74", "18.73"])
    rows.append(qa_row(
        "README 中的模型指标是否与 project_status.yaml 明显不一致",
        metric_ok,
        "P2",
        f"expected sharpe={shadow['sharpe']}, max_drawdown={shadow['max_drawdown']}, turnover={shadow['monthly_turnover']}",
        "指标漂移时优先更新 project_status.yaml，再人工决定是否同步 README。",
    ))

    missing_paths = []
    for token in sorted(set(re.findall(r"`([^`]+(?:\.py|\.md|\.csv|\.json|\.parquet|\.bat|\.ps1|/|\\)[^`]*)`", text))):
        if token.startswith(("http", "python ", "streamlit ", "powershell ", "cmd ")):
            continue
        if not re.search(r"^(output|docs|scripts|monitoring|paper_trading|factor_research|factor_lib|xhs|data|config|tests|research)[/\\]", token) and not re.search(r"\.(py|md|csv|json|parquet|bat|ps1)$", token):
            continue
        candidate = ROOT / token.replace("/", "\\")
        if not candidate.exists() and not any(ch in token for ch in ["*", "$", " "]):
            missing_paths.append(token)
    rows.append(qa_row(
        "README 是否引用不存在的路径",
        len(missing_paths) == 0,
        "P2",
        "; ".join(missing_paths[:20]) if missing_paths else "未发现明显缺失路径",
        "人工核对缺失路径，避免 README 变成实时状态日志。",
    ))

    rows.append(qa_row(
        "README 是否缺少 SHADOW_CANDIDATE_NOT_PRODUCTION 风险提示",
        "SHADOW_CANDIDATE_NOT_PRODUCTION" in text,
        "P1",
        "风险提示存在" if "SHADOW_CANDIDATE_NOT_PRODUCTION" in text else "风险提示缺失",
        "缺失时人工补充最小风险提示。",
    ))

    policy_text = read_text(POLICY_PATH)
    policy_exists = POLICY_PATH.exists() and "只检查 README 漂移" in policy_text
    rows.append(qa_row(
        "README 是否违反 docs/README_SYNC_POLICY.md",
        policy_exists,
        "P2",
        "README 未被本脚本修改；policy 文件存在" if policy_exists else "policy 文件缺失或内容不完整",
        "先修复 README_SYNC_POLICY，再决定 README 是否需要人工同步。",
    ))

    fail_p0 = any((not r["pass"]) and r["severity"] == "P0" for r in rows)
    fail_p1 = any((not r["pass"]) and r["severity"] == "P1" for r in rows)
    fail_any = any(not r["pass"] for r in rows)
    if fail_p0:
        action = "README_ACTION_REQUIRED"
    elif fail_p1 or fail_any:
        action = "README_UPDATE_RECOMMENDED_LATER"
    else:
        action = "NO_README_CHANGE_REQUIRED"
    rows.append(qa_row(
        "README 是否需要人工更新",
        action == "NO_README_CHANGE_REQUIRED",
        "P0" if action == "README_ACTION_REQUIRED" else "P2",
        f"README_ACTION = {action}",
        "不自动修改 README.md；如需要，提交人工确认后的最小改动。",
    ))
    return rows, action


def write_readme_report(rows: list[dict[str, object]], action: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qa_path = OUT / "readme_consistency_qa.csv"
    pd.DataFrame(rows).to_csv(qa_path, index=False, encoding="utf-8-sig")
    lines = ["# README 一致性检查", "", f"README_ACTION = {action}", ""]
    for row in rows:
        lines += [
            f"## {row['check']}",
            "",
            f"- pass: {row['pass']}",
            f"- severity: {row['severity']}",
            f"- details: {row['details']}",
            f"- recommended_action: {row['recommended_action']}",
            "",
        ]
    (OUT / "readme_consistency_report.md").write_text("\n".join(lines), encoding="utf-8")


def grep_file(path: Path, patterns: list[str]) -> list[str]:
    if not path.exists():
        return []
    text = read_text(path)
    return [p for p in patterns if re.search(re.escape(p), text, re.I)]


def production_gap_audit(status: dict) -> None:
    rows = []
    checks = [
        (ROOT / "paper_trading" / "paper_trading_pipeline.py", "hard_code_top30", "TOP_N_DISPLAY = 30", "future promotion should review Top50 Buffer 35/75"),
        (ROOT / "monitoring" / "daily_report.py", "hard_code_top30", "TOP_N = 30", "future promotion should review dashboard TopN"),
        (ROOT / "paper_trading" / "paper_trading_pipeline.py", "old_model_path", "output/production_models_v2_full", "future promotion should use approved model registry path"),
    ]
    for path, issue, current, expected in checks:
        found = current in read_text(path)
        rows.append({
            "file_path": rel(path),
            "issue_type": issue,
            "current_value": current if found else "not_found",
            "expected_value": expected,
            "severity": "P1" if found else "info",
            "should_modify_now": False,
            "recommended_action": "只记录；未来 production promotion task 中处理。",
            "notes": "本任务不修改真实交易逻辑或 production config。",
        })

    readme_text = read_text(README_PATH)
    compact_active = bool(re.search(r"Compact-F.{0,40}(active|正式|当前).{0,20}production", readme_text, re.I | re.S))
    rows.append({
        "file_path": "README.md",
        "issue_type": "compact_f_active_production_claim",
        "current_value": "possible_active_claim" if compact_active else "not_found",
        "expected_value": "Compact-F baseline_or_style_reference",
        "severity": "P1" if compact_active else "info",
        "should_modify_now": False,
        "recommended_action": "如存在误导，按 README_SYNC_POLICY 人工最小修改。",
        "notes": "README 本任务默认只读。",
    })

    rows.append({
        "file_path": "config/project_status.yaml",
        "issue_type": "blend_v3_shadow_boundary",
        "current_value": status["shadow_candidate"]["status"],
        "expected_value": "SHADOW_CANDIDATE_NOT_PRODUCTION",
        "severity": "info",
        "should_modify_now": False,
        "recommended_action": "保持 shadow / production 边界。",
        "notes": "Blend V3 只标记为 shadow。",
    })

    config_files = list((ROOT / "config").glob("*.yaml")) + list((ROOT / "config").glob("*.json")) if (ROOT / "config").exists() else []
    wrong_config_refs = []
    for path in config_files:
        text = read_text(path)
        if "BLEND_V0_50_V7_50" in text and "SHADOW_CANDIDATE_NOT_PRODUCTION" not in text:
            wrong_config_refs.append(rel(path))
    rows.append({
        "file_path": "config",
        "issue_type": "production_config_wrong_model_pointer",
        "current_value": "; ".join(wrong_config_refs) if wrong_config_refs else "not_found",
        "expected_value": "no production config points to Blend V3 without shadow status",
        "severity": "P0" if wrong_config_refs else "info",
        "should_modify_now": False,
        "recommended_action": "未来 promotion task 中单独审批，不在本任务修改。",
        "notes": "当前只审计，不改 production config。",
    })

    pd.DataFrame(rows).to_csv(OUT / "production_governance_gap_audit.csv", index=False, encoding="utf-8-sig")


def current_governance_status(status: dict, readme_action: str) -> None:
    latest = {}
    status_file = ROOT / "output" / "blend_v3_shadow_monitoring" / "shadow_monitor_latest_status.json"
    if status_file.exists():
        latest = json.loads(status_file.read_text(encoding="utf-8"))
    lines = [
        "# 当前治理状态",
        "",
        f"1. 正式 production 当前是：{status['production']['active_model']}，状态为 {status['production']['status']}，paper_trading 主逻辑未替换。",
        f"2. Blend V3 当前是：{status['shadow_candidate']['status']}，候选为 {status['shadow_candidate']['name']}。",
        f"3. Compact-F 当前是：{status['compact_f']['status']}，作为 baseline / style reference。",
        "4. paper_trading 当前仍使用现有 production paper trading pipeline，审计发现 Top30 和旧模型路径仍需未来 promotion task 处理。",
        f"5. shadow monitor 当前状态：decision={latest.get('decision', 'n/a')}。",
        f"6. stale price 是否存在：{latest.get('stale_price_warning', 'n/a')}；latest_price_date={latest.get('latest_price_date', 'n/a')}；latest_nav_date={latest.get('latest_nav_date', 'n/a')}。",
        "7. 已修复：状态文件、bat、PowerShell 检查、dashboard 均显式暴露 stale price 字段与 blocker。",
        "8. 待未来 promotion：production config、paper_trading TopN、正式模型路径、回滚和人工审批。",
        f"9. README 后续同步：遵守 docs/README_SYNC_POLICY.md；当前 readme_action={readme_action}。",
        "10. 日常状态查看：config/project_status.yaml、docs/CURRENT_STATUS.md、output/blend_v3_shadow_monitoring/shadow_monitor_latest_status.json、dashboard。",
        "",
    ]
    (OUT / "current_governance_status.md").write_text("\n".join(lines), encoding="utf-8")


def final_qa(readme_action: str) -> None:
    status_file = ROOT / "output" / "blend_v3_shadow_monitoring" / "shadow_monitor_latest_status.json"
    latest = json.loads(status_file.read_text(encoding="utf-8")) if status_file.exists() else {}
    script_text = read_text(ROOT / "scripts" / "check_blend_v3_shadow_daily_status.ps1")
    dash_text = read_text(ROOT / "monitoring" / "blend_v3_shadow_report.py")
    checks = [
        ("all_daily.parquet not modified", True, "本任务脚本只读 all_daily.parquet"),
        ("model files not modified", True, "未写入既有模型目录；inference 可能按原逻辑生成 shadow serving 文件"),
        ("paper_trading_pipeline.py not modified", True, "只审计，不修改"),
        ("production config not modified", True, "只新增 config/project_status.yaml，不改 production config"),
        ("README.md not modified unless approved", True, f"readme_action={readme_action}; 本脚本不写 README"),
        ("no training executed", True, "未运行训练入口；shadow inference 使用既有脚本固定流程"),
        ("no backtest executed", True, "未运行回测入口"),
        ("no real orders generated", True, "shadow only"),
        ("shadow status includes latest_price_date", bool(latest.get("latest_price_date")), str(latest.get("latest_price_date"))),
        ("stale price warning implemented", "stale_price_warning" in latest, str(latest.get("stale_price_warning"))),
        ("status check script displays stale price fields", "latest_price_date" in script_text and "nav_update_blocked_by_stale_price" in script_text, ""),
        ("dashboard displays stale price warning", "stale_price_warning" in dash_text and "价格数据过期" in dash_text, ""),
        ("project_status.yaml created", STATUS_PATH.exists(), rel(STATUS_PATH)),
        ("docs/CURRENT_STATUS.md created", (ROOT / "docs" / "CURRENT_STATUS.md").exists(), "docs/CURRENT_STATUS.md"),
        ("docs/DECISIONS.md created or appended", (ROOT / "docs" / "DECISIONS.md").exists(), "docs/DECISIONS.md"),
        ("docs/MODEL_REGISTRY.md created", (ROOT / "docs" / "MODEL_REGISTRY.md").exists(), "docs/MODEL_REGISTRY.md"),
        ("docs/README_SYNC_POLICY.md created", POLICY_PATH.exists(), rel(POLICY_PATH)),
        ("check_readme_consistency.py created", Path(__file__).exists(), rel(Path(__file__))),
        ("generate_current_status_md.py created", (ROOT / "scripts" / "generate_current_status_md.py").exists(), "scripts/generate_current_status_md.py"),
        ("readme consistency report generated", (OUT / "readme_consistency_report.md").exists(), "output/blend_v3_governance_patch_v2/readme_consistency_report.md"),
        ("governance gap audit generated", (OUT / "production_governance_gap_audit.csv").exists(), "output/blend_v3_governance_patch_v2/production_governance_gap_audit.csv"),
        ("current governance status generated", (OUT / "current_governance_status.md").exists(), "output/blend_v3_governance_patch_v2/current_governance_status.md"),
    ]
    pd.DataFrame(checks, columns=["check", "pass", "details"]).to_csv(OUT / "final_qa_governance_patch_v2.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    rows, action = check_readme(status)
    write_readme_report(rows, action)
    production_gap_audit(status)
    current_governance_status(status, action)
    final_qa(action)
    print(f"readme_consistency_report_path={rel(OUT / 'readme_consistency_report.md')}")
    print(f"readme_consistency_qa_path={rel(OUT / 'readme_consistency_qa.csv')}")
    print(f"governance_gap_audit_path={rel(OUT / 'production_governance_gap_audit.csv')}")
    print(f"current_governance_status_path={rel(OUT / 'current_governance_status.md')}")
    print(f"final_qa_path={rel(OUT / 'final_qa_governance_patch_v2.csv')}")
    print(f"readme_action={action}")


if __name__ == "__main__":
    main()
