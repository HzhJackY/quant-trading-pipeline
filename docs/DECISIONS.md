# 决策日志

## 2026-06-28

决策：

- BLEND_V0_50_V7_50 + Top50 Buffer 35/75 进入 shadow live monitoring。
- 不替换 production。
- Compact-F 降级为 baseline / style reference。
- README 不再作为实时状态源，建立 project_status.yaml + docs 状态治理。

## 2026-06-28

决策：

- CSMAR Table Inventory Audit v1 完成。
- 是否可进入 CSMAR PIT Financial Audit v1：否，需先配置/确认 CSMAR API 权限。
- CSMAR 不接入 production，不接入 BLEND_V0_50_V7_50，不接入 Compact-F。
- 不修改 README.md。

## 2026-06-29

决策：

- 建立 CSMAR 本地 credential loader。
- 真实 credential 不写入 markdown / README / output。
- 旧 CSMAR discovery 脚本位于 xhs/scripts。
- 是否成功恢复访问：否。
- 是否重跑 Table Inventory：否。
- 不接入 production。
- 不修改 README.md。

## 2026-06-29

决策：

- CSMAR Local Env Reactivation + Table Inventory Rerun v3 完成。

- 本地 .env.local credential loader 是否生效：是。
- 旧 CSMAR discovery 是否成功：是。
- Table Inventory Audit v1 是否重跑成功：是。
- 是否可进入 CSMAR PIT Financial Audit v1：是。
- 不接入 production。
- 不修改 README.md。

## 2026-06-29

决策：

- CSMAR PIT Financial Audit v1 完成。
- PIT 风险等级：INCONCLUSIVE。
- 是否建议重建 CSMAR PIT 财务因子：否，先扩展覆盖或人工复核。
- 不接入 production。
- 不修改 README.md。

## 2026-06-29

决策：

- CSMAR PIT Financial Audit v1 最终复核运行完成。
- PIT 风险等级：HIGH。
- 是否建议重建 CSMAR PIT 财务因子：是，下一步建议 CSMAR PIT Financial Factor Rebuild v1。
- 当前结论是 potential pre-announcement update 风险信号，不宣称已证明真实泄露。
- 不接入 production。
- 不修改 README.md。

## 2026-06-29

决策：

- CSMAR 从 xhs 子项目提升为主项目数据源。
- root-level canonical location 为 data_sources/csmar。
- 旧 xhs CSMAR 结果保留为 legacy reference。
- 后续 CSMAR PIT factor rebuild 必须输出到 root-level output。
- 不接入 production。
- 不修改 README。
- 不修改 training_panel。

## 2026-06-29

决策：

- CSMAR PIT Financial Factor Rebuild v1 完成。
- 是否生成 PIT 合规月频财务因子面板：否。
- 可替换字段列表：无。
- 不可替换或需人工复核字段：ROE, EP, BP, ProfitGrowth_YoY, RevGrowth_YoY, NetMargin, Debt_Ratio, sales_expense_to_revenue, rd_expense_to_revenue, earnings_preview_midpoint_yoy。
- 不接入 production。
- 不修改 README。
- Blend V3 historical metrics 继续标记为 PIT-under-review，直到基于新 PIT 面板重新跑 OOS tournament。
- Decision = CSMAR_PIT_REBUILD_BLOCKED_NO_SOURCE_DATA。

## 2026-06-29

决策：

- CSMAR Pack Download Executor v1 完成。
- CSMAR manual/web download 与 API 下载共享 quota。
- 改用 getPackResultExt 打包下载路线。
- 默认 dry-run，不自动消耗额度。
- 下载数据保存到 data/csmar_exports。
- 不接入 production。
- 不修改 README。
- Decision = CSMAR_PACK_DOWNLOAD_EXECUTOR_READY_DRY_RUN。

## 2026-06-29

决策：

- CSMAR Pack Download Executor Patch v2 完成。
- CSMAR pack download v1 执行失败中发现 ShortName 字段不存在。
- Patch v2 改为基于 field dictionary 校验 columns。
- INVALID_FIELD 与 DAILY_LIMIT 分开。
- 30 分钟重复查询保护增强。
- 本 patch 未执行下载。
- 不接入 production。
- 不修改 README。
- Decision = CSMAR_PACK_DOWNLOAD_EXECUTOR_PATCHED_WITH_BLOCKED_QUERIES。

## 2026-06-29

决策：

- CSMAR P0 PIT pack 文件已成功导入。
- IAR_Rept / IAR_Forecdt 已标准化。
- 是否生成 PIT 公告日面板：是。
- 财务原始字段仍未下载。
- 不接入 production。
- 不修改 README。
- Decision = CSMAR_P0_PIT_PACK_IMPORT_READY_FOR_REVIEW。

## 2026-06-29

决策：

- CSMAR P1 financial pack download 是否成功：否。
- 成功下载哪些表：无。
- 是否仍需继续下载剩余 P1 表：是。
- 不接入 production。
- 不修改 README。
- 不修改 training_panel。
- Decision = CSMAR_P1_FINANCIAL_PACK_DOWNLOAD_READY_DRY_RUN。

## 2026-06-29

决策：

- CSMAR P1 financial pack download 是否成功：否。
- 成功下载哪些表：无。
- 是否仍需继续下载剩余 P1 表：是。
- 不接入 production。
- 不修改 README。
- 不修改 training_panel。
- Decision = CSMAR_P1_FINANCIAL_PACK_DOWNLOAD_BLOCKED_DAILY_LIMIT。

## 2026-06-29

决策：

- FI_T5 字段校验 READY。
- FS_Comins / FS_Combas 被识别为不可用或错误表名。
- P1 财务表映射已离线修正。
- 等待额度恢复后继续 pack download。
- 不接入 production。
- 不修改 README。
- Decision = CSMAR_P1_FINANCIAL_TABLE_MAPPING_PATCH_READY。

## 2026-06-29

决策：

- 启动并完成 Price-Volume Divergence Reversal Factor Audit v1。
- 本任务独立于 CSMAR PIT 财务重建。
- 不接入 production。
- 不修改 README。
- 最佳候选因子：reversal_20d。
- 后续是否进入 residual alpha test 取决于结果；当前 can_enter_residual_alpha_test=False。
- Decision = PV_REVERSAL_AUDIT_FRAGILE_NEEDS_REFINEMENT。

## 2026-06-29

决策：

- 完成 PV Reversal Fragility Attribution & Refinement v1。
- 是否仍然依赖小盘 / 低流动性：yes_or_unresolved。
- 是否可以进入 residual alpha test：False。
- 不接入 production。
- 不修改 README。
- Decision = PV_REVERSAL_REFINEMENT_IMPROVED_BUT_STILL_FRAGILE。

## 2026-06-29

决策：

- 完成 Alpha008 A-Share Reversal Factor Audit v1。
- 是否优于 reversal_20d_liquid_only：False。
- 是否通过流动性 / 市值 / 中性化审计：SURVIVES_NEUTRALIZATION。
- 是否可以进入 residual alpha test：False。
- 不接入 production。
- 不修改 README。
- Decision = ALPHA008_AUDIT_INVALID_DATA_QUALITY。

## 2026-06-30

决策：

- CSMAR P1 financial pack download 是否成功：是。
- 成功下载哪些表：FI_T5。
- 是否仍需继续下载剩余 P1 表：是。
- 不接入 production。
- 不修改 README。
- 不修改 training_panel。
- Decision = CSMAR_P1_FINANCIAL_PACK_DOWNLOAD_PARTIAL_SUCCESS。

## 2026-06-30

决策：

- 完成 Alpha008 A-Share Reversal Factor Audit v1。
- 是否优于 reversal_20d_liquid_only：False。
- 是否通过流动性 / 市值 / 中性化审计：SURVIVES_NEUTRALIZATION。
- 是否可以进入 residual alpha test：False。
- 不接入 production。
- 不修改 README。
- Decision = ALPHA008_AUDIT_WEAK_AFTER_NEUTRALIZATION。

## 2026-06-30

决策：

- FI_T5 已本地存在且可读。
- P1 下载脚本已修补为跳过本地已下载表。
- 下一下载目标不再是 FI_T5，当前 next_execute_table=FN_Fn050。
- 不接入 production。
- 不修改 README。
- Decision = CSMAR_P1_EXISTING_EXPORT_SKIP_PATCH_READY。

## 2026-06-30

决策：

- 基于目标因子重构需求重新审视 P1 下载优先级。
- FI_T5 已下载但作为衍生指标表，不应替代底表重构。
- FN_Fn050 不应继续作为默认下一下载目标。
- 推荐的下一下载表：WAIT_FOR_HUMAN_REVIEW。
- 不访问 CSMAR API。
- 不修改 README。
- 不接入 production。
- Decision = CSMAR_FN050_DEPRIORITIZED_WAITING_FOR_SOURCE_TABLE_REVIEW。

## 2026-06-30

决策：

- FS_Comins 人工下载文件已导入审计。
- FS_Comins 覆盖利润表底表字段。
- FN_Fn050 不再是当前优先下载目标。
- 仍需 FS_Combas / TRD_Dalyr。
- 不访问 CSMAR API。
- 不修改 README。
- 不接入 production。

## 2026-06-30

决策：

- 已完成 FS_Comins PIT 覆盖率分层 patch。
- 有效窗口定义为 report_type=A、v15 universe、2017 年以后。
- 覆盖率决策：CSMAR_FS_COMINS_EFFECTIVE_WINDOW_PARTIAL_PIT_READY。
- 不访问 CSMAR API，不下载数据。
- 不修改 README，不接入 production。

## 2026-06-30

决策：

- FS_Comins / FS_Combas 人工下载文件已导入审计。
- 核心财务底表已落地。
- FN_Fn050 不再是当前优先下载目标。
- 仍需 TRD_Dalyr 或等价市值字段。
- 不访问 CSMAR API。
- 不修改 README。
- 不接入 production。

## 2026-06-30

决策：

- FS_Comins 有效窗口 PIT 覆盖率仅 80.25%。
- 执行 raw IAR_Rept / IAR_Forecdt 离线归因与修复。
- 是否生成 repaired P0 PIT panel：True。
- 修复后覆盖率：0.802468。
- 是否进入 core FS merge：False。
- 不访问 CSMAR API。
- 不修改 README。
- 不接入 production。

## 2026-06-30

决策：

- 停止补 PIT 日期。
- 冻结 IAR / FS_Comins / FS_Combas 时间范围：IAR 2015-2026，FS_Comins 1990-2026，FS_Combas 2005-2026。
- effective rebuild window = 2017-2026。
- strict actual PIT only。
- missing PIT rows dropped, no fallback。
- 生成 strict monthly as-of core FS source panel。
- EP/BP 仍需 market_cap。
- 不访问 CSMAR API。
- 不修改 README。
- 不接入 production。
- Decision = CSMAR_PIT_SCOPE_FROZEN_STRICT_CORE_FS_SOURCE_PANEL_NEEDS_COVERAGE_REVIEW。



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
- Decision = CSMAR_TRD_DALYR_MARKET_CAP_SOURCE_NEEDS_COVERAGE_REVIEW。


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
- Decision = CSMAR_TRD_DALYR_MARKET_CAP_MONTHLY_SOURCE_READY。
