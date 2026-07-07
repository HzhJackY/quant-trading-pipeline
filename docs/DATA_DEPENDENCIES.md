# 数据依赖

## 当前主链路

- `output/all_daily.parquet`：历史日线价格源，本任务只读。
- `output/training_panel_v15_sr.parquet`：Blend V3 shadow candidate 面板。
- `output/full_panel_forced_tournament_v3/V0_FULL_V15_OOS.parquet`：V0 组件 artifact。
- `output/full_panel_forced_tournament_v3/V7_FULL_V15_OOS.parquet`：V7 组件 artifact。
- `output/paper_trading_db/state.db`：当前 paper trading 状态库和 market cache。

## 替代数据状态

- Media15：封存研究，不接入当前 alpha。
- XHS：封存机制研究，不接入当前 alpha。
- 百度指数：封存探索，不接入当前 alpha。
- CSMAR：表清单待审计，本任务不运行。

## 治理规则

数据依赖变化需要更新 `config/project_status.yaml` 和本文件；如影响入口、风险声明或 production / shadow 边界，再按 `docs/README_SYNC_POLICY.md` 判断是否需要更新 README。

## CSMAR

- 数据源：CSMAR
- 用途：PIT 财务数据、公告日、字段级财务因子重建。
- credential：本地 `.env.local` 或环境变量；不进入 git。
- canonical code path：`data_sources/csmar`。
- canonical output path：`output/csmar_*`。
- legacy path：`xhs/scripts/csmar_*` and `xhs/output/csmar_*`。
- 当前状态：PIT 风险已检测，factor rebuild pending。
