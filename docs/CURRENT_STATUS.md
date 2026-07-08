# 当前状态

## 1. 当前阶段

- 项目：A-share Quant Shadow Trading System
- 阶段：shadow_monitoring
- 状态源：`config\project_status.yaml`

## 2. 当前 production 状态

- active_model：legacy_paper_trading
- 状态：not_replaced
- 说明：Current paper_trading main logic has not been replaced.

## 3. 当前 shadow candidate

- 名称：BLEND_V0_50_V7_50_TOP50_BUFFER_V3
- 状态：SHADOW_CANDIDATE_NOT_PRODUCTION
- 面板：`output/training_panel_v15_sr.parquet`
- 组合规则：Top50 Buffer 35/75
- 最新特征月份：2026-06-30
- 最新持仓数量：50

## 4. Blend V3 关键指标

- Sharpe：1.509353
- Max Drawdown：-0.107414
- 月换手：0.18729

## 5. Compact-F 当前定位

- 状态：baseline_or_style_reference
- 说明：No longer sole default production candidate.

## 6. Shadow monitoring 文件与命令

- Dashboard：`monitoring/blend_v3_shadow_report.py`
- 计划任务：QuantBlendV3ShadowDailyMonitor
- 手动更新：`cmd /c scripts\run_blend_v3_shadow_live_update.bat`
- 状态检查：`powershell -ExecutionPolicy Bypass -File scripts\check_blend_v3_shadow_daily_status.ps1`
- 状态文件：`output/blend_v3_shadow_monitoring/shadow_monitor_latest_status.json`

## 7. 当前风险提示

- Blend V3 是 shadow candidate，不是 production，不生成真实订单。
- 当前 paper_trading 主逻辑未替换，production 与 shadow 边界必须保持清晰。
- 如果 `latest_price_date` 落后当前运行日超过 3 个自然日，shadow NAV 更新受 stale price blocker 限制。

## 8. 下一步

- 继续监控 stale price 状态与 shadow NAV。
- 在未来独立 promotion task 中审查 TopN、模型路径、production config 和回滚方案。
- README 只按同步规则更新，不作为实时状态源。

## 9. 最后更新时间

2026-06-30

## 相关治理文档

- 决策日志：`docs/DECISIONS.md`
- 模型注册表：`docs/MODEL_REGISTRY.md`
- README 同步规则：`docs/README_SYNC_POLICY.md`
