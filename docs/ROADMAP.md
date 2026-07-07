# 项目路线图

## 当前阶段

Blend V3 处于 shadow live monitoring 阶段，不替换当前 production paper trading 主逻辑。

## 下一步

1. 持续监控 shadow NAV、持仓稳定性、stale price blocker。
2. 修复或补齐价格数据更新链路，确保 `latest_price_date` 跟随实际运行日。
3. 在独立 promotion task 中审查 production config、paper trading TopN、模型路径和风控边界。
4. promotion 前生成正式变更说明、回滚方案和人工审批记录。

## 暂不执行

- 不重训模型。
- 不调参。
- 不把 Blend V3 设为 production。
- 不运行 CSMAR、MediaCrawler、Media15 / XHS / 百度接入。
