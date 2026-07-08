from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS_PATH = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"


def load_status() -> dict:
    with STATUS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_current_status(status: dict) -> str:
    project = status["project"]
    production = status["production"]
    shadow = status["shadow_candidate"]
    compact_f = status["compact_f"]
    docs = status["docs"]
    last_updated = project.get("last_updated") or date.today().isoformat()
    return "\n".join([
        "# 当前状态",
        "",
        "## 1. 当前阶段",
        "",
        f"- 项目：{project['name']}",
        f"- 阶段：{project['current_phase']}",
        f"- 状态源：`{STATUS_PATH.relative_to(ROOT)}`",
        "",
        "## 2. 当前 production 状态",
        "",
        f"- active_model：{production['active_model']}",
        f"- 状态：{production['status']}",
        f"- 说明：{production['notes']}",
        "",
        "## 3. 当前 shadow candidate",
        "",
        f"- 名称：{shadow['name']}",
        f"- 状态：{shadow['status']}",
        f"- 面板：`{shadow['panel']}`",
        f"- 组合规则：{shadow['portfolio_rule']}",
        f"- 最新特征月份：{shadow['latest_feature_month']}",
        f"- 最新持仓数量：{shadow['latest_holding_count']}",
        "",
        "## 4. Blend V3 关键指标",
        "",
        f"- Sharpe：{shadow['sharpe']}",
        f"- Max Drawdown：{shadow['max_drawdown']}",
        f"- 月换手：{shadow['monthly_turnover']}",
        "",
        "## 5. Compact-F 当前定位",
        "",
        f"- 状态：{compact_f['status']}",
        f"- 说明：{compact_f['notes']}",
        "",
        "## 6. Shadow monitoring 文件与命令",
        "",
        f"- Dashboard：`{shadow['dashboard']}`",
        f"- 计划任务：{shadow['monitor_task']}",
        "- 手动更新：`cmd /c scripts\\run_blend_v3_shadow_live_update.bat`",
        "- 状态检查：`powershell -ExecutionPolicy Bypass -File scripts\\check_blend_v3_shadow_daily_status.ps1`",
        "- 状态文件：`output/blend_v3_shadow_monitoring/shadow_monitor_latest_status.json`",
        "",
        "## 7. 当前风险提示",
        "",
        "- Blend V3 是 shadow candidate，不是 production，不生成真实订单。",
        "- 当前 paper_trading 主逻辑未替换，production 与 shadow 边界必须保持清晰。",
        "- 如果 `latest_price_date` 落后当前运行日超过 3 个自然日，shadow NAV 更新受 stale price blocker 限制。",
        "",
        "## 8. 下一步",
        "",
        "- 继续监控 stale price 状态与 shadow NAV。",
        "- 在未来独立 promotion task 中审查 TopN、模型路径、production config 和回滚方案。",
        "- README 只按同步规则更新，不作为实时状态源。",
        "",
        "## 9. 最后更新时间",
        "",
        str(last_updated),
        "",
        "## 相关治理文档",
        "",
        f"- 决策日志：`{docs['decisions']}`",
        f"- 模型注册表：`{docs['model_registry']}`",
        f"- README 同步规则：`{docs['readme_sync_policy']}`",
    ])


def append_decision_once() -> None:
    block = "\n".join([
        "## 2026-06-28",
        "",
        "决策：",
        "",
        "- BLEND_V0_50_V7_50 + Top50 Buffer 35/75 进入 shadow live monitoring。",
        "- 不替换 production。",
        "- Compact-F 降级为 baseline / style reference。",
        "- README 不再作为实时状态源，建立 project_status.yaml + docs 状态治理。",
    ])
    if DECISIONS_PATH.exists():
        text = DECISIONS_PATH.read_text(encoding="utf-8")
        if "README 不再作为实时状态源，建立 project_status.yaml + docs 状态治理" in text:
            return
        DECISIONS_PATH.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")
    else:
        DECISIONS_PATH.write_text("# 决策日志\n\n" + block + "\n", encoding="utf-8")


def main() -> None:
    status = load_status()
    CURRENT_STATUS_PATH.write_text(render_current_status(status) + "\n", encoding="utf-8")
    append_decision_once()
    print(f"project_status_path={STATUS_PATH.relative_to(ROOT)}")
    print(f"current_status_doc_path={CURRENT_STATUS_PATH.relative_to(ROOT)}")
    print(f"decisions_doc_path={DECISIONS_PATH.relative_to(ROOT)}")
    print("decision=CURRENT_STATUS_GENERATED")


if __name__ == "__main__":
    main()
