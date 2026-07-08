# Quant Research Project

## Project Overview

本项目是一个量化研究管线，当前主线用于 Clean V0 strict-lag 策略评估，以及 `robust_cleaned` benchmark attribution。`robust_cleaned` 不是新的 portfolio strategy，不带 alpha/weights；它是用于分析 Clean V0 的基准、因子和特征归因框架。

当前暂停点：截至 `Unified_V0_Alpha_Attribution_v0`，Clean V0 已完成重建并确认为官方 strict-lag baseline，robust_cleaned 归因已经完成但带 caveats。当前阶段只做文档整理和报告封存，不直接进入 V7、blend、production、live trading、模型训练或策略优化。

当前推荐下一任务：`Robust_Cleaned_Attribution_Report_Seal_v0`。

## Current Project Status / Robust Cleaned Attribution Status

| 模块 | 当前状态 | 关键结论 | 下一步限制 |
|---|---|---|---|
| Clean V0 official baseline | `V0_REBUILD_CERTIFIED_UNDERPERFORMS_LEGACY` | 2017-03 至 2026-05，109 个月，net 20bps Sharpe 0.4051 | legacy leakage variant 不得作为 clean evidence |
| Data Market Contract | `DATA_MARKET_CONTRACT_IAR_RECHECK_SEALED_WITH_CAVEATS` | IAR_Rept 可作 formal PIT disclosure calendar，IAR_Forecdt 可作 forecast calendar | caveats 保留；不得跳到生产 |
| Monthly EW baseline | `MONTHLY_EW_BASELINE_LADDER_READY_WITH_CAVEATS` | 2010-01 至 2026-05，194 个月；guardrails passed | 194 vs 197 月份差异必须显式处理 |
| FF5 prebuilt alignment | `FF5_PREBUILT_MONTHLY_ALIGNMENT_READY` | CSMAR 预构建日频 FF5 转月频，197 个月，无缺月 | 无独立 RF；不得自建 FF5 替代主线 |
| DGTW prebuilt alignment | `DGTW_PREBUILT_MONTHLY_ALIGNMENT_READY_WITH_UNIVERSE_FILTER_CAVEATS` | primary IsNotBSE=1，197 个月，2026-06 排除 | IsNotBSE caveat 与匹配权重 caveat 保留 |
| Attribution input registry | `ROBUST_CLEANED_ATTRIBUTION_INPUT_REGISTRY_READY_WITH_CAVEATS` | common window 2010-01 至 2026-05，common month count 194 | 允许 unified attribution；不允许 V7/blend/production |
| Unified V0 attribution | `UNIFIED_V0_ALPHA_ATTRIBUTION_READY_WITH_CAVEATS` | 三套框架下 robust positive alpha 未建立 | 推荐先做报告封存或诊断 |
| Report seal | pending | 推荐任务 `Robust_Cleaned_Attribution_Report_Seal_v0` | seal 前不进入新策略推进 |
| V7 / blend / production | not allowed | 当前暂停整理状态，不是上线状态 | 不得直接进入 |

## Critical Route Corrections

| 主题 | 当前路线 |
|---|---|
| `robust_cleaned` 定义 | 不是策略分支，不应描述为有 alpha/weights 的组合策略；它是 Clean V0 的 benchmark / attribution framework。 |
| FF5 来源 | 使用 CSMAR 预构建 `data/csmar_exports/STK_MKT_FIVEFACDAY.xlsx`，当前主线不从 `TRD_Mnth.xlsx`、`FS_Combas.xlsx` 或 `FS_Comins.xlsx` 自建 FF5。 |
| FF5 读取规则 | `header=0, skiprows=[1,2]`；第 1 行为字段名，第 2/3 行为描述/单位，第 4 行起为数据；不要使用 `header=3`。 |
| FF5 语义 | primary selected = 1-series，即 float-market-value weighted；无独立 RF，`RiskPremium1` 用作 `mkt_rf`。 |
| DGTW 来源 | 使用 CSMAR 预构建 `data/csmar_exports/STK_MKT_DGTWASSINGMENTS.xlsx` 和 `data/csmar_exports/STK_MKT_DGTWBENCH.xlsx`；文件名是 `ASSINGMENTS`，不是 `ASSIGNMENTS`。 |
| DGTW 读取规则 | `header=0, skiprows=[1,2]`；当前主线不自建 DGTW。 |
| leakage | legacy leakage variant 不能作为 clean evidence；Clean V0 strict-lag 是官方 baseline。 |
| 月份对齐 | 不允许 silent inner join；月份缺失、policy exclusion 和 common window 必须显式记录。 |

## Main Results

### Clean V0 Official Baseline

| 指标 | 数值 |
|---|---:|
| official window | 2017-03 至 2026-05 |
| official month count | 109 |
| net 20bps Sharpe | 0.4050573915950944 |
| net 20bps mean monthly return | 0.005919115596330279 |
| net 20bps t-stat | 1.2207850091558083 |
| cumulative return | 0.6628369485932033 |
| max drawdown | -0.32474926666288617 |

Policy excluded months: `2017-02`, `2017-04`, `2018-02`。

### Unified Attribution Results

| 框架 | 结果 | 解读 |
|---|---:|---|
| Monthly EW active mean monthly return | -0.0015323185516221286 | Clean V0 未显著跑赢平均股票基准；active return 略负且统计不显著，t-stat = -0.1955400000307503。 |
| FF5 intercept | 0.008045150824189334 | intercept 为正，普通 t-stat = 1.556500596198002，HAC t-stat = 1.972124832012735，但因为无独立 RF，只能视为 raw-return caveated alpha。 |
| FF5 R-squared | 0.06848374996046935 | 因子解释度较低，不能单凭 intercept 宣称强 alpha。 |
| DGTW adjusted mean monthly return | -0.010882826422018347 | DGTW adjusted return 为负，t-stat = -1.597945458324844，HAC t-stat = -2.3892155373000814。 |
| DGTW matched weight share avg | 0.9977981651376148 | 匹配权重占比较高，但 caveat 仍需保留。 |

总体结论：Clean V0 在 robust_cleaned 的 Monthly EW、FF5、DGTW 框架下没有建立稳健正 alpha。FF5 intercept 的正值需要 `no independent RF` caveat；DGTW 调整后收益为显著负值。不得基于单一 FF5 intercept 直接推进 V7、blend 或 production。

## Artifact Index

| 组件 | 关键文件 | 作用 |
|---|---|---|
| Clean V0 registry | `output/clean_v0_baseline_registry_v0/clean_v0_baseline_registry_summary.json` | 官方 clean strict-lag baseline 状态 |
| Data Market Contract | `output/data_market_contract_seal_recheck_with_iar_pit_source_v0/data_market_contract_iar_recheck_summary.json` | PIT calendar 与 forecast calendar 合约 |
| Monthly EW ladder | `output/monthly_equal_weight_baseline_ladder_v0/monthly_ew_ladder_summary.json` | 平均股票 baseline ladder |
| Monthly EW returns | `output/monthly_equal_weight_baseline_ladder_v0/monthly_ew_ladder_combined_returns.parquet` | 月频 baseline returns；优先 parquet，不读大型 xlsx |
| FF5 source QA | `output/ff5_prebuilt_cn_factor_source_qa_v0/ff5_prebuilt_source_qa_summary.json` | FF5 预构建源文件语义确认 |
| FF5 monthly alignment | `output/ff5_prebuilt_cn_factor_monthly_alignment_v0/ff5_monthly_alignment_summary.json` | FF5 月频对齐状态 |
| FF5 aligned factors | `output/ff5_prebuilt_cn_factor_monthly_alignment_v0/ff5_monthly_primary_aligned_factors.parquet` | 月频 FF5 因子输入 |
| DGTW source QA | `output/dgtw_prebuilt_cn_benchmark_source_qa_v0/dgtw_source_qa_summary.json` | DGTW assignment / benchmark 源文件语义确认 |
| DGTW monthly alignment | `output/dgtw_prebuilt_cn_benchmark_monthly_alignment_v0/dgtw_monthly_alignment_summary.json` | DGTW 月频 benchmark 对齐状态 |
| DGTW benchmark panel | `output/dgtw_prebuilt_cn_benchmark_monthly_alignment_v0/dgtw_benchmark_primary_isnotbse1_monthly_panel.parquet` | primary IsNotBSE=1 benchmark panel |
| Robust cleaned registry | `output/robust_cleaned_attribution_input_registry_v0/robust_cleaned_registry_summary.json` | attribution 输入登记与 common months |
| Unified attribution | `output/unified_v0_alpha_attribution_v0/unified_attribution_summary.json` | Clean V0 统一归因总结 |
| Unified attribution report | `output/unified_v0_alpha_attribution_v0/unified_attribution_report.md` | 归因报告草稿 |

完整 artifact 索引见 `output/project_readme_update_robust_cleaned_status_v0/readme_artifact_index.csv`。

## Caveats and Guardrails

- FF5 无独立 RF，`RiskPremium1` 作为 `mkt_rf`，因此 FF5 intercept 是 raw-return caveated alpha。
- DGTW matched weight share avg = 0.9977981651376148，但仍需保留匹配权重 caveat。
- DGTW `IsNotBSE=1` 是 primary universe filter，不是 cell key；其他 `IsNotBSE` 值只能作为 sensitivity。
- Monthly EW 为 194 个月，FF5/DGTW 为 197 个月；差异来自 Monthly EW 缺失 `2017-02`、`2017-04`、`2018-02`，不能 silent inner join。
- Data Market Contract 状态为 `SEALED_WITH_CAVEATS`。
- legacy outperformance 很大程度由 leakage variants 解释；不得使用 leakage variant 证明 clean alpha。
- 不允许 zero-fill。
- 不允许 matched-only renormalization。
- 不允许 silent inner join。
- 本 README 更新任务没有重新计算 returns、读取 Route B weights、修改 upstream artifacts 或进入 V7/blend/production。

## Allowed Next Steps

| 类型 | 任务 | 说明 |
|---|---|---|
| 推荐 | `Robust_Cleaned_Attribution_Report_Seal_v0` | 先封存 robust_cleaned attribution 报告与 caveats。 |
| 可选诊断 | `V0_Improvement_Diagnostic_v0` | 若目标是改进 V0，先做诊断，不直接优化策略。 |
| 可选稳健性 | `Attribution_Robustness_Check_v0` | 若目标是复核归因结论，做独立 robustness check。 |
| 独立审计分支 | `V7_Source_Audit_v0` | 仅作为 separate audit branch；不得直接把当前结论转成 V7。 |

直接禁止：V7、blend、production、live trading、model training、strategy optimization。

## How to Resume

| 目标 | 下一任务 |
|---|---|
| 继续 attribution 封存 | `Robust_Cleaned_Attribution_Report_Seal_v0` |
| 改进 V0 | `V0_Improvement_Diagnostic_v0` |
| 检查归因稳健性 | `Attribution_Robustness_Check_v0` |
| 开新策略分支 | 先运行 `V7_Source_Audit_v0`，并作为 separate audit branch |

如 Codex 会话中断，先读取 `output/project_readme_update_robust_cleaned_status_v0/RUN_STATE.md`，再继续本文档整理任务。


## Deprecated / Historical Notes

以下内容为本次更新前 README 的完整保留版本。若其中内容与上方当前状态冲突，以上方 `Current Project Status / Robust Cleaned Attribution Status` 为准。

<details>
<summary>展开历史 README 内容</summary>

# A 股基本面量化选股与 Shadow Paper Trading 系统

当前项目已经从单一 Compact-F 候选，升级为 V0 Linear / V7 TO-Aware ML / Compact-F / Blend 的统一生产擂台与 shadow 监控系统。

当前最强 shadow candidate 为：

`BLEND_V0_50_V7_50 + Top50 Buffer 35/75`

状态：

`SHADOW_CANDIDATE_NOT_PRODUCTION`

这意味着：

- 已通过 v3 full panel OOS 擂台；
- 已通过 production review gate；
- 已生成 live shadow holdings；
- 已接入 shadow dashboard 和每日自动更新；
- 但尚未替换正式 production；
- 尚未生成真实交易订单；
- 需要 shadow 观察 1-3 个月后再决定是否晋升。

## 当前状态

- `BLEND_V0_50_V7_50 + Top50 Buffer 35/75` 是当前最强 production shadow candidate。
- 当前状态是 `SHADOW_CANDIDATE_NOT_PRODUCTION`，不替换当前 production，不替换当前 paper_trading 主逻辑。
- 当前不做实盘交易指令，不生成真实订单，不覆盖当前 paper trading 持仓。
- Blend V3 已进入 shadow live monitoring，每日自动任务已安装或准备安装。
- Shadow dashboard 已中文化，股票代码前导 0 显示问题已修复。
- Compact-F 不再是唯一默认生产候选，保留为基本面可解释 baseline / 风格对照模型。
- Media15 / XHS / 百度指数仍然是独立另类数据研究线，不进入主 alpha。

## 核心结果

Full Panel Forced Tournament v3：

- `main_panel_path = output\training_panel_v15_sr.parquet`
- `median_symbols_per_month_main_panel = 714`
- `best_full_panel_model = BLEND_V0_50_V7_50`
- `best_full_panel_portfolio_rule = Top50_Buffer_35_75`
- `best_full_panel_net_sharpe = 1.509353`
- `best_full_panel_max_drawdown = -0.107414`
- `best_full_panel_turnover = 0.187290`
- `v0_available = True`
- `v7_available = True`
- `compact_f_available = True`
- `leakage_detected = False`
- `decision = FULL_PANEL_TOURNAMENT_V3_READY_FOR_REVIEW`

Production Candidate v3 Review：

- `review_gate_pass = True`
- `shadow_holdings_generated = True`
- `candidate_status = SHADOW_CANDIDATE_NOT_PRODUCTION`
- `decision = BLEND_V3_SHADOW_MODE_READY`
- 不建议直接替换 production，建议进入 paper trading shadow mode。

当前候选结果表，指标来源为 `output\full_panel_forced_tournament_v3\tournament_v3_full_panel_metrics.csv`：

| 候选 | 数据面板 | 组合规则 | Net Sharpe | MaxDD | 月换手 | 状态 |
|---|---|---|---:|---:|---:|---|
| BLEND_V0_50_V7_50 | training_panel_v15_sr | Top50 Buffer 35/75 | 1.509 | -10.74% | 18.73% | Shadow Candidate |
| V0_FULL_V15_OOS | training_panel_v15_sr | Top50 Buffer 35/75 | 1.147 | -9.87% | 9.47% | 单模型候选 |
| V7_FULL_V15_OOS | training_panel_v15_sr | Top50 Buffer 35/75 | 1.206 | -12.43% | 36.70% | ML 辅助候选 |
| Compact-F | training_panel_v15_sr / aligned | Top50 Buffer 35/75 | 0.273 | -31.97% | 27.00% | 基本面对照 / 风格解释 |

说明：v3 Sharpe 是历史 OOS 回测结果，不保证未来实盘表现，也不代表已上线实盘。

## 模型定位

### BLEND_V0_50_V7_50

- 当前最强 shadow candidate。
- 由 V0 Linear OOS 与 V7 TO-Aware OOS 的标准化分数组合。
- `blend_score = 0.50 * V0_score_z + 0.50 * V7_score_z`
- 使用 `Top50 Buffer 35/75`。
- 当前只用于 shadow monitoring，不用于真实交易指令。

### V0 Linear

- 当前完整 v15 panel 上重新生成严格 OOS。
- 不再使用旧 `split_universe_blended` artifact 作为最终依据。
- 是主 alpha engine 的核心组成部分。

### V7 TO-Aware ML

- 当前完整 v15 panel 上固定规格重训。
- 不做超参数搜索。
- 用于提供 ML 辅助信号和稳定性。
- 与 V0 blend 后表现最佳。

### Compact-F

- 不再作为唯一默认 production candidate。
- 保留为基本面可解释 baseline / 风格对照。
- 历史 Compact-F 结论保留在“历史候选与模型治理”部分。
- 不是失败模型，而是风格干净但收益风险不及 v3 Blend。

## Blend V3 Shadow Live Monitoring

Shadow Live 当前状态：

- `candidate_status = SHADOW_CANDIDATE_NOT_PRODUCTION`
- `latest_feature_month = 2026-06-30`
- `shadow_holding_count = 50`
- `tradability_pass_count = 526`
- `stale_feature_warning = False`
- `decision = BLEND_V3_SHADOW_LIVE_READY`
- QA 全部通过。

关键文件：

- dashboard：`monitoring\blend_v3_shadow_report.py`
- latest holdings：`output\blend_v3_shadow_live\latest_shadow_holdings_live.csv`
- latest report：`output\blend_v3_shadow_live\latest_shadow_report_live.md`
- NAV tracker：`output\blend_v3_shadow_monitoring\shadow_daily_nav.csv`
- latest status：`output\blend_v3_shadow_monitoring\shadow_monitor_latest_status.json`

这是影子组合，不是正式 production，不是交易指令，不生成真实订单。

## Shadow Daily Automation

查看 shadow 状态：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check_blend_v3_shadow_daily_status.ps1
```

手动触发 shadow 任务：

```powershell
schtasks /Run /TN QuantBlendV3ShadowDailyMonitor
```

打开中文 shadow dashboard：

```powershell
streamlit run monitoring\blend_v3_shadow_report.py
```

手动运行 shadow update：

```powershell
cmd /c scripts\run_blend_v3_shadow_live_update.bat
```

安装每日任务：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_blend_v3_shadow_daily_task.ps1
```

任务信息：

- `task_name: QuantBlendV3ShadowDailyMonitor`
- `schedule: 周一到周五 18:25，本地时间`
- `status: shadow only`
- `shadow update bat = scripts\run_blend_v3_shadow_live_update.bat`
- `dashboard = monitoring\blend_v3_shadow_report.py`
- `status check = scripts\check_blend_v3_shadow_daily_status.ps1`

## Project Structure

```text
quant/
├── factor_research/                 # 因子分析、成本回测、择时 baseline、回测引擎
├── paper_trading/                   # paper trading（纸交易）生产管线，当前不被 Blend V3 自动替换
├── monitoring/                      # Streamlit 每日风控看板与 Blend V3 shadow dashboard
├── factor_lib/                      # 因子定义与扩展因子模块
├── data/                            # 数据抓取、缓存与清洗模块
├── output/                          # 报告、模型、回测结果与研究产物
├── scripts/                         # 一次性脚本、审计脚本、自动任务脚本、报告生成脚本
├── tests/                           # 单元测试
├── xhs/                             # 另类数据研究工作区（独立研究或辅助目录）
├── research/                        # 研究草稿与补充材料（独立研究或辅助目录）
├── MediaCrawler/                    # 另类数据采集实验目录（独立研究或辅助目录）
├── run_compact_f_production_validation.py # Compact-F 历史生产候选验证入口
├── run_v15_portfolio_optimization.py # 组合层优化与 Top50 Buffer 评估
├── run_backtest_with_costs.py       # 成本感知回测
├── run_timing_comparison.py         # 择时 baseline 对比
├── requirements.txt                 # 环境依赖
└── README.md
```

## Paper Trading（纸交易）

当前纸交易入口：`paper_trading/paper_trading_pipeline.py`

定位：

- 服务于当前 production / paper trading 主逻辑的模拟运行与执行编排。
- Blend V3 当前不替换 `paper_trading/paper_trading_pipeline.py`。
- 当前不覆盖已有 paper trading 持仓。
- 月末调仓。
- 先做物理风控过滤，再做 universe alignment（股票池对齐）与信号输出。

当前实现中的物理风控过滤包括：

- `ST / *ST`
- 停牌
- 日均成交额低于 5000 万
- 总市值低于 50 亿

当前实现中的核心链路：

- `CSI800 universe alignment`
- `cross_sectional_rank` 只在最终 universe 内计算
- `signal anchor` 持久化上期与当期信号

常用命令：

```bash
pip install -r requirements.txt
python paper_trading/paper_trading_pipeline.py --date 2026-06-30 --force-rebalance -v
python paper_trading/paper_trading_pipeline.py --date 2026-06-30 --force-rebalance --force-refresh
python paper_trading/paper_trading_pipeline.py --date 2026-06-30 --force-rebalance --skip-ingestion
```

## Monitoring（监控看板）

当前主监控入口：`monitoring\daily_report.py`

Blend V3 shadow 监控入口：`monitoring\blend_v3_shadow_report.py`

看板内容包括：

- 风控雷达
- KPI 卡片
- 累计净值
- 因子暴露
- 红黑榜
- 全景持仓
- Blend V3 shadow live holdings / NAV / status

工程栈：

- `Baostock + SQLite + Streamlit`

使用方式：

```bash
streamlit run monitoring/daily_report.py
streamlit run monitoring\blend_v3_shadow_report.py
```

说明：监控看板用于纸交易与风险复核，不等同于 alpha 证明。

## 大盘择时 baseline：仅监控，不作为默认生产仓位

当前规则：

- 中证 500 `MA20/60` 死叉，或 20 日年化波动率超过 252 日 80% 分位时，仓位乘数降为 `0.3`。

历史回测结果：

- 无择时：Net Sharpe `1.13`，年化收益 `21.27%`，MaxDD `-18.01%`
- 有择时：Net Sharpe `0.80`，年化收益 `10.66%`，MaxDD `-12.50%`

结论：

- 该择时方案可以降低回撤，但收益损失过大，Sharpe 恶化。
- 当前参数不具备实盘默认价值。
- 除非显式进行择时实验，否则 production 默认 `multiplier` 应保持 `1.0`。

相关入口：

```bash
python run_timing_comparison.py
python run_backtest_with_costs.py
```

## 另类数据研究：独立研究线

- `Media15 / XHS / 百度指数` 是独立另类数据研究线。
- 它们不进入主 alpha，不修改 Compact-F，不接入 Blend V3。
- `media_neg_share_all` 可以作为 risk diagnostic filter（风险提示变量）或风险复核信号。
- 原始 `XHS attention` 不应直接作为因子。
- 百度指数更适合作为较弱的外部搜索关注对照。
- 当前不声称这些变量已经形成可交易 alpha。
- 可用于报告、履历、风险诊断，不作为当前 production promotion 条件。

## 历史候选与模型治理

以下材料用于研究追溯、面试叙事和方法论展示，不代表当前生产调参路线。

### Compact-F

- 从“当前唯一生产候选”降级为“历史生产候选 / 基本面对照模型”。
- 历史 `Compact-F + Top50 Buffer 35/75` 仍保留其 Top50 Buffer 指标与治理价值。
- 后续 v3 full panel tournament 显示 V0/V7 Blend 更适合作为当前 shadow candidate。
- Compact-F 风格干净、可解释性强，但收益风险不及 v3 Blend。

### V1/V2 Alpha Drift

- V1/V2 Alpha Drift 保留为历史 RCA。
- `V1->V2 Alpha Drift`、GS / colsample / BP / ProfitGrowth drift 等材料用于解释历史模型漂移。
- 不作为当前主线调参路径。

### Media15 / XHS / 百度

- 保留为独立另类数据研究。
- 不进入主 alpha。
- 可用于报告、履历、风险诊断。

代表性归档文件：

- `output\ml_v7_final_report.md`
- `output\V1_to_V2_alpha_drift_investigation_final.md`
- `run_split_universe.py`
- `run_ml_v7.py`
- `run_model_comparison.py`

## Roadmap

### P0 当前已完成

- V0/V7/Compact-F/Blend full panel tournament v3；
- Blend V3 review gate；
- Shadow live holdings；
- 中文 dashboard；
- 股票代码格式修复；
- Shadow daily automation。

### P1 当前进行中

- shadow monitoring 观察 1-3 个月；
- 每日检查 task result、NAV、可交易性；
- 每周检查 shadow vs current paper trading；
- 每月检查 Top50 Buffer 换手和持仓稳定性。

### P2 后续候选

- 将 shadow dashboard 合并进主 monitoring dashboard；
- 执行层模拟：100 股整数手、涨跌停、停牌、滑点；
- production promotion review；
- CSMAR 数据层升级；
- README 和 model registry 持续同步。

### 暂停方向

- 不继续调 Compact-FT / FT3；
- 不继续技术因子堆叠；
- 不把 Media15 / XHS / 百度接入主 alpha；
- 不继续追求更高回测 Sharpe；
- 不在 shadow 观察期频繁改模型。

## 风险提示

- v3 Sharpe 是历史 OOS 回测结果，不保证未来实盘表现。
- 当前 Blend V3 是 shadow candidate，不是正式 production。
- 当前不生成真实交易订单，不发送交易指令。
- 需要 1-3 个月 shadow 观察后再评估是否晋升。
- 需继续监控任务是否每日成功运行。
- 需继续监控 NAV 是否更新。
- 需继续监控缺失价格、ST、停牌、涨跌停。
- 需继续监控流动性。
- 需继续监控与当前 paper trading 差异。
- 需继续监控单行业 / 单风格集中。
- 需继续监控实际换手。

## Quick Start

1. 安装环境

```bash
pip install -r requirements.txt
```

2. 查看 Blend V3 shadow 状态

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check_blend_v3_shadow_daily_status.ps1
```

3. 打开 Blend V3 中文 shadow dashboard

```powershell
streamlit run monitoring\blend_v3_shadow_report.py
```

4. 手动运行 Blend V3 shadow update

```powershell
cmd /c scripts\run_blend_v3_shadow_live_update.bat
```

5. 纸交易 dry run / force rebalance

```bash
python paper_trading/paper_trading_pipeline.py --date 2026-06-30 --force-rebalance -v
```

6. 主监控看板

```bash
streamlit run monitoring/daily_report.py
```

7. 单元测试

```bash
pytest tests/ -v
```

说明：`run_ml_v7.py`、`run_split_universe.py`、`run_model_comparison.py` 不再属于 Quick Start 主流程，应视为历史研究归档入口。

</details>
