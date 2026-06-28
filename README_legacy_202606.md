> Current status as of 2026-06:
> Traditional data mainline has been sealed.
> Default production candidate: Compact-F + Top50 Buffer 35/75.
> Historical Split-Universe / V0–V7 / Alpha Drift materials are retained below for traceability.
> XHS / Baidu Index alternative-data research is a separate track and must not modify the traditional mainline.

# **Quant Factor Research — A 股多因子选股系统**

从因子研究到纸交易的生产级量化选股管线。核心系统是 **Split-Universe 双模型架构**

（大盘/小盘分层因子建模），配合从 Linear → LightGBM → LambdaRank → Turnover-Aware 的

完整 ML 实验链，以及交易成本建模 and GARCH 风险管理。

## **核心结论**

|指标|线性 Alpha (V0)|ML V7 (TO-Aware)|说明|
|-|-|-|-|
|**Net Sharpe**|**1.74** (正交化后)|0.98|★ Gram-Schmidt 正交化释放 IC\_IR 潜力|
|**Max Drawdown**|−6.5% (正交化后)|−27%|16 因子全参与, 分散化更充分|
|**Monthly Turnover**|23.7%|12.6%|TO-Aware loss 有效控换手|
|**Monthly Cost**|5.9 bps|3.3 bps|低换手 → 低成本|

**诚实结论**：在当前 A 股中证 800 因子体系下，线性 IC\_IR 加权合成的稳健性优于 机器学习模型的复杂度收益。ML 实验链完整记录了每一版的设计动机、消融对比和失败分析 — 这种"证伪"过程本身是量化研究的方法论核心。

完整的消融实验链：output/ml\_v7\_final\_report.md

## **项目结构**

quant/  
├── data/  
│   ├── fetcher.py                  # 数据获取 (日线/成分股/财务, PIT 对齐)  
│   └── cleaner.py                  # 预处理 (MAD 去极值/行业中性化/Z-score)  
│  
├── factor\_lib/                     # 16 因子库 (被 run\_factor\_research.py 引用)  
│   ├── value.py                    #   BP, EP (估值)  
│   ├── momentum.py                 #   Mom\_1M, 3M, 6M, 12M-1M (动量)  
│   ├── quality.py                  #   ROE, Debt\_Ratio, Net\_Profit\_Margin (质量)  
│   ├── volatility.py               #   Vol\_20D, 60D, Beta (波动率)  
│   ├── growth.py                   #   RevGrowth\_YoY, ProfitGrowth\_YoY (成长)  
│   └── technical.py                #   VolChg\_20D, PriceDev\_20D (技术面)  
│  
├── factor\_research/  
│   ├── ic\_analysis.py              # Rank IC / IC\_IR / IC 衰减分析  
│   ├── group\_backtest.py           # 5 分组回测 + 多空组合  
│   ├── backtest\_engine.py          # 多因子合成 (IC\_IR 加权/去冗余/符号翻转)  
│   ├── split\_universe.py           # ★ Split-Universe 双模型系统  
│   ├── market\_timing.py            # 大盘择时风控 (MA20/60死叉 + 波动率区间)  
│   ├── report.py                   # 可视化 (IC 图/净值/相关性矩阵)  
│   ├── dynamic\_weight.py           # 动态权重分配 (IC\_IR 衰减 + Vol 调整)  
│   ├── transaction\_cost.py         # 分层交易成本 model (Almgren-Chriss 冲击)  
│   ├── ml\_engine.py                # ML 基础引擎 (Walk-Forward CV)  
│   ├── ml\_engine\_v7.py             # ★ V7 终版: 1M label + 0M gap + TO-Aware  
│   ├── ml\_engine\_v2.py             # \[归档] LambdaRank 实验  
│   ├── ml\_engine\_v5.py             # \[归档] TO-Aware 3M gap 实验  
│   ├── ml\_engine\_v6.py             # \[归档] Label Blending + Time-Decay  
│   └── production\_engine.py        # 生产引擎 (3-seed ensemble, 模型持久化)  
│  
├── paper\_trading/                  # 纸交易生产管线  
│   ├── paper\_trading\_pipeline.py   # ★ 每日 cron 入口 + 月末调仓编排  
│   ├── data\_ingestion.py           # 日线行情 + 基本面并行数据获取 + CSI 800 成分股  
│   ├── factor\_compute.py           # 16 因子实时计算 + Universe 限制 + 风控前置过滤  
│   ├── state\_manager.py            # 信号锚点持久化 + 市场缓存 SQLite  
│   └── baostock\_adapter.py         # Baostock PIT 财务数据适配 (零前看偏差)  
│  
├── monitoring/                      # 实盘监控看板  
│   └── daily\_report.py             # ★ Streamlit 每日风控看板 (组合/基准/Alpha/因子暴露/风控雷达)  
│  
├── output/                         # 回测输出 + ML 预测 + 报告  
│   ├── ml\_v7\_final\_report.md       # ★ V0/V5/V7 最终消融报告  
│   ├── V1\_to\_V2\_alpha\_drift\_investigation\_final.md  # ★ V1→V2 Alpha Drift 项目总结  
│   ├── factor\_meaning\_drift\_audit.md     # 因子含义漂移审计  
│   ├── alpha\_drift\_causal\_decomposition.md # 2×2×2 因果分解  
│   ├── fold\_ensemble\_production\_audit.md  # Fold 集成生产审计  
│   ├── ensemble\_stability\_root\_cause.md   # 集成稳定性根因分析  
│   ├── v1\_v2\_monthly\_attribution.md      # V1/V2 逐月收益归因  
│   ├── factor\_ic\_summary.csv       # 因子 IC 汇总  
│   ├── production\_models/          # V1 生产模型 (54 folds × 3 seeds)  
│   └── production\_models\_v2\_full/  # V2\_Full (71 folds × 3 seeds, CSI 800)  
│  
├── tests/                          # 单元测试  
├── resume/                         # 中英文简历  
├── run\_factor\_research.py          # ★ 主入口: 4 阶段因子研究流水线  
├── run\_split\_universe.py           # Split-Universe 双模型分析  
├── run\_backtest\_with\_costs.py      # 成本感知回测  
├── run\_ml\_v7.py                    # ★ V7 终版 ML 训练 + 回测  
├── run\_ml\_v6.py                    # V6 消融对比  
├── run\_inference\_export.py         # 模型批量预测与结果导出  
├── run\_model\_comparison.py         # ★ 模型擂台对齐对比回测  
│  
├── # ── Alpha Drift 诊断工具链 (2026.06) ──  
├── run\_turnover\_analysis.py        # 换手率根因 (Rank稳定性/Overlap/信号噪声)  
├── run\_ablation.py                 # 单模型消融 (colsample/GS/Universe)  
├── run\_ensemble\_stability\_analysis.py  # 集成稳定性 (Seed/Fold/规模消融)  
├── run\_fold\_ensemble\_audit.py      # 生产推理审计 + Fold方案对比  
├── run\_v1\_v2\_attribution.py        # V1/V2 逐月收益归因 + 风格暴露  
├── run\_alpha\_drift\_forensic.py     # 法证审计 (GS排序/BP删除/增量消融)  
├── run\_causal\_decomposition.py     # 2×2×2 全因子因果分解 (ANOVA)  
├── run\_factor\_meaning\_drift.py     # 因子含义漂移 (分布/排序/IC/Decile)  
├── run\_shap\_diagnosis.py           # SHAP 特征归因 + 交互效应  
│  
├── # ── Phase B 数据管线 (CSI 800 全量重建) ──  
├── run\_phaseb\_fetch\_data.py        # 双轨获取 (akshare日线 + PIT财务)  
├── run\_phaseb\_rebuild\_panel.py     # 训练面板构建 (逐月 + 因子计算)  
├── run\_phaseb\_constituents.py      # 历史 CSI 800 成分股 (20期快照)  
├── run\_phaseb\_pipeline.py          # Phase B 全流程编排  
├── validate\_data\_integrity.py      # 数据完整性验证 (PIT/市值/除权)  
├── run\_retrain\_production.py       # ★ Phase B 生产模型重训  
│  
├── requirements.txt                # 环境依赖 (版本锁定)  
└── .gitignore
├── requirements.txt                # 环境依赖 (版本锁定)  
└── .gitignore

## **Runner 脚本指南**

|脚本|用途|状态|
|-|-|-|
|run\_factor\_research.py|主入口 — 4 阶段因子研究流水线 (数据预取→面板构建→预处理→分析)|★ 活跃|
|run\_split\_universe.py|Split-Universe 大盘/小盘双模型分析 + Baseline 对比|★ 活跃|
|run\_backtest\_with\_costs.py|带交易成本的分层回测 (佣金+印花税+冲击)|★ 活跃|
|run\_ml\_v7.py|V7 终版: 1M label + 0M gap + TO-Aware — 训练 + V0/V5/V7 对比|★ 活跃|
|run\_ml\_v6.py|V6 Label Blending + Time-Decay 消融对比|保留|
|run\_ml\_backtest.py|ML 信号 vs 线性信号回测对比|保留|
|run\_dynamic\_weight.py|动态权重分配 (IC\_IR 衰减 + 波动率调整)|保留|
|run\_ml\_ablation.py|!\[]\[image1]V0-V3 早期消融实验|归档|
|run\_ml\_lambdarank.py|!\[]\[image1]LambdaRank 回测|归档|
|run\_ml\_turnover\_aware.py|!\[]\[image1]V5 λ sweep 实验|归档|
|run\_timing\_comparison.py|择时对比回测 — 有/无择时完整绩效对比 (NAV图+信号日志)|★ 活跃|
|run\_inference\_export.py|模型批量推理与预测结果导出 (供对齐归因使用)|★ 活跃|
|run\_model\_comparison.py|模型擂台 — V1 vs V2\_Full 严格对齐归因对比 (Rank IC/单调性/风格偏离)|★ 活跃|
|run\_turnover\_analysis.py|换手率根因 — Rank稳定性/Overlap/边界敏感性/信号噪声分解|诊断|
|run\_ablation.py|单模型消融 — colsample/GS/Universe 控制变量实验|诊断|
|run\_ensemble\_stability\_analysis.py|集成稳定性 — Seed/Fold/规模消融 + 风格漂移 (213模型 pairwise r)|诊断|
|run\_fold\_ensemble\_audit.py|生产推理审计 — Fold方案对比 (6种) + 严格 OOS 回测|诊断|
|run\_v1\_v2\_attribution.py|V1/V2 逐月收益归因 — 风格暴露/Regime分析/年度Sharpe分解|诊断|
|run\_alpha\_drift\_forensic.py|法证审计 — GS排序变化/BP删除代价/V1精确重建验证|诊断|
|run\_causal\_decomposition.py|2×2×2 全因子因果分解 — Universe×GS×colsample ANOVA|诊断|
|run\_factor\_meaning\_drift.py|因子含义漂移审计 — 分布/排序稳定性/IC迁移/Decile曲线/D1-D10构成|诊断|
|run\_shap\_diagnosis.py|SHAP 特征归因 — 分桶分析/交互效应/叶节点决策路径|工具|
|run\_retrain\_production.py|★ Phase B 生产模型重训 — CSI 800 + GS + colsample 全量训练|★ 活跃|
|run\_phaseb\_fetch\_data.py|Phase B 数据获取 — akshare双轨 (日线+财务) + 断点续传|管线|
|run\_phaseb\_rebuild\_panel.py|Phase B 面板构建 — 逐月因子计算 + Parquet导出|管线|
|run\_phaseb\_constituents.py|CSI 800 历史成分股 — Baostock 20期半年度快照 (1,476只)|管线|
|run\_phaseb\_pipeline.py|Phase B 全流程编排 — 获取→面板→训练 一站执行|管线|
|validate\_data\_integrity.py|数据完整性验证 — PIT/市值/除权/异常值 4模块扫描|管线|
|monitoring/daily\_report.py|Streamlit 每日风控看板 — Top 30 组合 P\&L / 因子暴露 / 风控雷达|★ 活跃|
|diagnose\_stock\_pool.py|股票池诊断 — 采样方法对比 + 行业覆盖分析|工具|

## **快速开始**

\# 1. 环境安装  
pip install -r requirements.txt

\# 2. 因子研究流水线 (首次 \~40min, 缓存后秒过)  
python run\_factor\_research.py

\# 3. Split-Universe 双模型分析 (含 Baseline 对比)  
python run\_split\_universe.py

\# 4. V7 终版 ML 训练 + 回测 (含 V0/V5/V7 三路对比)  
python run\_ml\_v7.py

\# 5. 大盘择时验证  
python -c "from factor\_research.market\_timing import fetch\_csi500, plot\_timing\_history; plot\_timing\_history(fetch\_csi500())"

\# 6. 择时对比回测 (有/无择时完整绩效对比)  
python run\_timing\_comparison.py

\# 7. 纸交易 (每日 cron, 16:00 收盘后运行)

# 首次运行建议加 --force-rebalance 测试调仓:

python paper_trading/paper_trading_pipeline.py --date 2026-06-30 --force-rebalance -v

\# 8. 每日风控看板 (Streamlit, 盘后 18:00 后)  
streamlit run monitoring/daily\_report.py

\# 9. 导出新旧模型预测结果 (模型对比前置步骤)  
python run\_inference\_export.py --model-dir output/production\_models\_v1 --panel output/preprocessed\_v2.parquet --output output/split\_universe\_blended\_v1.parquet  
python run\_inference\_export.py --model-dir output/production\_models\_v2\_full --panel output/preprocessed\_v2.parquet --output output/split\_universe\_blended\_v2\_full.parquet

\# 10. 模型对比回测 (V1 vs V2\_Full 严格对齐归因)  
python run\_model\_comparison.py

\# 单元测试  
pytest tests/ -v

## **流水线架构**

Stage 1 (预取) ────→ Stage 2 (面板) ────→ Stage 3 (预处理) ────→ Stage 4 (分析)  
日线+财务缓存      因子计算+PIT对齐    MAD→中性化→Z-score     IC+回测+因子合成

4 阶段断点续跑 — 中断后重跑自动从断点继续。状态文件 .pipeline\_state.json 记录进度, 中间产物 output/panel.parquet 和 output/preprocessed.parquet 使后续分析无需重新跑全流程。

## **Split-Universe 双模型系统**

### **核心思想**

全市场线性模型忽视了 A 股最核心的结构性差异 — **大市值的机构定价 vs 小市值的散户定价**。 Split-Universe 按流通市值百分位将股票切分为大盘池 (Top 50%) 和小盘池 (Bottom 50%), 各自独立评估因子、独立合成信号, 最后通过池内截面 Z-score 标准化对齐量纲并拼接。

### **因子归属 (数据驱动, 非预设)**

|归属|因子|经济解释|
|-|-|-|
|**大盘型**|Debt\_Ratio, Net\_Profit\_Margin, Mom\_1M|杠杆/利润率是机构定价锚; 短期反转弱|
||PriceDev\_20D, VolChg\_20D|技术信号在低噪音环境中确定性强|
|**小盘型**|ProfitGrowth\_YoY, RevGrowth\_YoY|★ 成长是小盘核心引擎 (IC\_IR 0.47 vs 0.20)|
||BP, EP|深度价值在小盘中同样有效|
||Beta, Mom\_12M\_1M, Vol\_60D|长期趋势+低波策略在小盘中更显著|

## **大盘择时 — Beta 风控系统**

与 Alpha 选股严格解耦的大盘择时模块。在月末生成 target portfolio 权重时, 对总敞口 应用缩放乘数。**不修改选股排名, 仅控制总仓位风险暴露。**

### **触发逻辑**

|条件|触发|乘数|信号来源|
|-|-|-|-|
|正常状态|否|**1.0** (100% 满仓)|—|
|**MA20 死叉**|MA20 < MA60|**0.3** (30% 仓位)|中证 500 日线|
|**波动率飙高**|20日年化波动率 > 252日80% 分位|**0.3** (30% 仓位)|中证 500 日线|
|同时触发|死叉 + 高波|**0.3**|—|

状态无关: 每期独立判断, 不引入记忆效应  
触发时: 30% × \[Alpha Top 30 等权]  +  70% 现金

### **核心模块**

from factor\_research.market\_timing import (  
fetch\_csi500,                    # 获取中证 500 日线 (parquet 缓存)  
compute\_market\_multiplier,       # 单日乘数  
prepare\_timing\_multipliers,      # 批量预计算 (回测场景)  
apply\_position\_sizing,           # 权重缩放  
plot\_timing\_history,             # 择时历史可视化  
timing\_summary,                  # 逐日信号汇总表  
)

### **整合点**

|管线|整合方式|效果|
|-|-|-|
|**纸交易** paper\_trading\_pipeline.py|月末调仓自动计算乘数, \_print\_top\_picks() 显示缩放权重|每只股票权重 = 1/30 × 乘数|
|**成本回测** run\_backtest\_with\_costs()|新增 timing\_multipliers 参数|毛收益/换手率/成本自动按乘数缩放|

### **快速验证**

\# 查看择时触发历史  
python -c "  
from factor\_research.market\_timing import fetch\_csi500, plot\_timing\_history  
index\_df = fetch\_csi500()  
plot\_timing\_history(index\_df)  
"

\# 择时对比回测 (有/无择时完整绩效对比)  
python run\_timing\_comparison.py

### **择时回测基线 (2017–2024, 96 期)**

|指标|无择时|有择时|变化|
|-|-|-|-|
|Net Sharpe|**1.13**|0.80|−0.33|
|年化收益|**21.27%**|10.66%|−10.61%|
|最大回撤|−18.01%|**−12.50%**|+5.51%|
|月均换手|23.7%|14.3%|−9.4%|
|月均成本|5.9 bps|3.4 bps|−2.4 bps|

**诚实结论**: 当前参数 (MA20/60 死叉 + 波动率 80% 分位 → 0.3 乘数) 触发率 59%, 回撤改善约 5.5pp 但收益腰斩。Sharpe 从 1.13 降至 0.80, 风险调整后收益恶化。 该择时方案在当前参数下不具备实盘价值, 保留为基线供后续调参对比。 2018 年触发率 96.2% (全年熊市几乎全空仓), 但也因此错失 2019 年初 V型反弹。

## **实盘监控看板**

每日盘后 (18:00) Streamlit 风控看板, 监控 Top 30 纸交易组合的实时表现:

streamlit run monitoring/daily\_report.py

### **六大模块**

|模块|内容|数据源|
|-|-|-|
|🔴 **风控雷达**|暴跌 (−7%) / ST 警示 / 疑似停牌 扫描|Baostock 实时行情|
|📊 **KPI 卡片**|组合日收益 / 中证 500 基准 / 超额 Alpha|SQLite market\_cache|
|📈 **累计净值**|持有期组合 vs 基准 vs 累计超额三线图|SQLite + Baostock 基准|
|🧬 **因子暴露**|Size / Momentum / Value / Volatility 横截面百分位|基本面 parquet + 行情缓存|
|🏆 **红黑榜**|涨幅/跌幅 Top 3 柱状图|实时行情|
|📋 **全景持仓**|30 只全量明细表 + 日胜率进度条|实时行情|

### **技术亮点**

* **零前看偏差**: 财务数据通过 pubDate 门控做 PIT 对齐
* **纯 Python 栈**: Streamlit + SQLite + Baostock, 无外部数据库依赖
* **自动容错**: 数据缺失时优雅降级（显示 N/A 而非崩溃）
* **缓存策略**: st.cache\_data 5–10 分钟 TTL, 避免重复拉取行情

## **生产推理管线**

纸交易月末调仓的完整执行链路——从原始数据到 Top 30 持仓:

拉取原始数据 (market cache 60d + fundamentals + PIT financials)  
│  
▼  
Phase 2: 物理风控过滤 (apply\_risk\_filters)  
├─ 剔除 ST/\*ST               → name 包含 "ST"  
├─ 剔除停牌                  → 近 5 日无收盘数据  
├─ 剔除低流动性              → 日均成交额 < 5000 万  
└─ 剔除微盘股                → 总市值 < 50 亿  
│  
▼  
Phase 1: Universe 对齐 (CSI 800 截取)  
CSI 800 成分股 ∩ 风控通过 = 最终 Universe (\~688 只)  
│  
▼  
cross\_sectional\_rank (仅限最终 Universe 内部百分位)  
│  
▼  
LightGBM 3-seed × 54-fold ensemble 推理  
│  
▼  
Top 30 输出 → signal\_anchor → 等权持仓

### **常用指令**

\# 日常纸交易 (自动判断月末调仓)  
python paper\_trading/paper\_trading\_pipeline.py

\# 强制调仓 (测试用, 任何日期)  
python paper\_trading/paper\_trading\_pipeline.py --date 2026-06-30 --force-rebalance

\# 强制刷新 PIT 财务数据 (忽略缓存)  
python paper\_trading/paper\_trading\_pipeline.py --date 2026-06-30 --force-rebalance --force-refresh

\# 使用 Baostock PIT 财务 (pubDate 门控, 零前看偏差)  
python paper\_trading/paper\_trading\_pipeline.py --date 2026-06-30 --force-rebalance --use-baostock

\# 跳过数据摄入 (仅使用已有缓存执行调仓)  
python paper\_trading/paper\_trading\_pipeline.py --date 2026-06-30 --force-rebalance --skip-ingestion

\# 调试模式 (详细日志)  
python paper\_trading/paper\_trading\_pipeline.py --date 2026-06-30 --force-rebalance -v

### **后续优化方向**

|优先级|方向|说明|
|-|-|-|
|🔴 **P0**|**V1.5 混合模型设计**|基于 Alpha Drift 调查: 关 GS → 提 colsample→0.70-1.00 → 保留 BP → 1 seed。恢复 V1 风格暴露 (EP+ROE+PG正向)，保留 CSI 800 IC 优势 (0.062 vs 0.058)|
|🔴 **P0**|**因子含义漂移修复**|ProfitGrowth 跨 Universe 排名 r=0.001。测试: sector-relative 因子 / 更严格 winsorization / 替代成长因子|
|🟡 **P1**|信号质量+风格监控|Top30/Bottom30 Spread + 信号自相关 + 因子暴露追踪 (偏离 V1 baseline >0.5σ 告警)|
|🟡 **P1**|GS 软化实验|max\_correlation ∈ {0.85, 0.90, 0.95, 0.99}。当前 GS 无上限正交化导致 55% 特征重要性重分配|
|🟡 **P1**|分级乘数择时|当前 0.3/1.0 二元开关过粗糙。改三级: 0.3/0.6/1.0 或连续乘数 = f(vol\_percentile)|
|🟢 **P2**|因子实时 IC 监控|每期末 Rank IC，检测因子失效 (如 ProfitGrowth IC 0.053→0.022) 并自动降权|
|🟢 **P2**|成本模型实盘校准|用纸交易实际滑点反校准 Almgren-Chriss γ/η 参数|
|🟢 **P2**|线性信号并行|同时输出 IC\_IR 加权线性信号作为对照，在 dashboard 中对比 ML vs Linear Alpha|

## **⚖️ 模型擂台：V1 vs V2\_Full 严格对齐归因对比 (Model Comparison)**

在完成 Phase B 的全量数据抓取与模型重训后，我们需要对比“特征去垄断化”和“消除协变量偏移”带来的真实 Alpha 增量。

run\_model\_comparison.py 是一个专门的模型对比引擎，它会自动取两个模型预测结果的**严格交集（Strict Universe Alignment）**，确保在完全相同的底层股票池中公平竞技。

### **1. 前置准备 (生成预测文件)**

在运行对比脚本前，请确保使用新构建的 run\_inference\_export.py 脚本，对最新的全量面板 (preprocessed\_v2.parquet) 分别执行新旧模型的预测，以确保对比数据的基准完全一致。

**Step 1.1: 生成 V1 (旧模型) 的预测**

python run\_inference\_export.py \\  --model-dir output/production\_models\_v1 \\  --panel output/preprocessed\_v2.parquet \\  --output output/split\_universe\_blended\_v1.parquet

**Step 1.2: 生成 V2\_Full (新模型) 的预测**

python run\_inference\_export.py \\  --model-dir output/production\_models\_v2\_full \\  --panel output/preprocessed\_v2.parquet \\  --output output/split\_universe\_blended\_v2\_full.parquet

### **2. 运行对比脚本**

确认 output 目录下存在上述两个预测文件以及 preprocessed\_v2.parquet 后，在终端运行：

python run\_model\_comparison.py

### **3. 核心输出指标与解读**

脚本运行结束后，会在终端打印矩阵报告，并在 output 目录生成可视化图表 model\_comparison\_report.png。您需要重点检视以下三个维度：

1. **Alpha 质量与稳定性 (Rank IC \& IC\_IR)**：检视 V2\_Full 是否在 IC 绝对值不大幅下降的前提下，显著提升了 IC\_IR（抗风格切换能力）。
2. **多头单调性 (Decile Returns)**：观察柱状图，V2\_Full 的 Top 10% 收益是否比 V1 更高，且 1\~10 组的阶梯递减是否更加平滑完美。
3. **持仓风格漂移 (Style Drift)**：检视终端打印 of Top 30 风格特征暴露，确认 V2\_Full 是否成功降低了对 EP（价值）的绝对依赖，并合理增加了 ProfitGrowth（成长）和 Momentum（动量）的权重。
4. **实盘摩擦考研 (Net Sharpe \& MaxDD)**：在扣除双向印花税与滑点后，检视 V2\_Full 短周期量价特征带来的换手率是否被 Turnover-Aware L2 损失函数成功压制，并实现了最大回撤（MaxDD）的收敛。

## **🔬 V1→V2 Alpha Drift 调查 (2026.06)**

在 Phase B 全量 CSI 800 重训完成后，V2_Full Sharpe=0.51 显著低于 V1 Sharpe=0.70，但 IC 反而更高 (0.062 vs 0.058)。启动了为期 ~11 天的系统性根因调查，经历 4 个阶段、10+ 个主要实验。

### 调查阶段

| 阶段 | 检查内容 | 关键实验 | 核心发现 |
|------|---------|---------|---------|
| A: 参数怀疑 | ProfitGrowth, colsample, GS | SHAP分析, 单模型消融 | colsample=0.50 不是根因 (RankCorr 最高) |
| B: 稳定性调查 | RankCorr, Overlap, Seed, Fold, Ensemble | 集成稳定性分析, Fold审计 | 3 seed 几乎相同 (r≈0.966); fold=-1 已是 Sharpe 最优 |
| C: 风格漂移 | 因子暴露, Regime 分析 | 逐月收益归因 | V2 在上涨市做空 (−0.28%/月 vs V1 +0.30%), anti-growth 暴露 |
| D: 因子含义漂移 | Factor Meaning Drift, DGP Shift | 2×2×2 ANOVA, 分布审计 | Universe 扩张贡献 40% IC 方差; ProfitGrowth 跨Panel r=0.001 |

### 已证伪假设

- ❌ colsample=0.50 导致不稳定
- ❌ Fold/Ensemble 架构导致 Sharpe 下降
- ❌ Seed 方差导致不稳定
- ❌ ProfitGrowth 被模型反向使用
- ❌ BP 被 GS 错误删除 (残余 IC=+0.004)
- ❌ GS 对树模型无效 (55% 特征重要性重分配)

### 当前最可信解释

V1 和 V2 使用名称相同但经济含义不同的因子信号 (Factor Meaning Drift)。Universe 从 297→1,360 stocks 改变了所有因子的截面分布，GS 进一步重排了 12/16 因子的排序。V2 不是 V1 的劣化版——它是风格发生改变的新 Alpha。

完整调查报告: `output/V1_to_V2_alpha_drift_investigation_final.md`

---

## **ML 实验链 (V0 → V7)**

|版本|核心设计|Label|Gap|Sharpe|MaxDD|TO|结论|
|-|-|-|-|-|-|-|-|
|V0|Linear IC\_IR 加权|1M|0M|**1.13**|−18%|23.7%|★ 最优|
|V2|LambdaRank 排序学习|3M|3M|—|—|—|不收敛|
|V5|TO-Aware L2 loss, λ=2.0|3M|3M|0.95|−27%|12.9%|控换手有效, 回撤恶化|
|V6|Label Blending + Time-Decay|混合|3M|0.96|−27%|13.4%|无显著改善|
|V7|TO-Aware + 1M label + 0M gap|1M|0M|0.98|−27%|12.6%|回撤未修复|

**核心发现**: 3M gap 是结构性 MaxDD 根因 — 模型在 3 个月盲区内信号的预测力衰减严重。 移除 gap 后回撤未修复 (V7 -27% vs V0 -18%), 说明 ML 模型的截面排序能力本身弱于 线性 IC\_IR 加权。

## **因子库 (16 因子)**

|类别|因子|全市场 IC\_IR|说明|
|-|-|-|-|
|估值|EP|+0.443|盈利/价格 (Earnings Yield)|
||BP|+0.270|净资产/价格 (Book-to-Price)|
|质量|Net\_Profit\_Margin|+0.333|销售净利率|
||ROE|+0.314|净资产收益率|
||Debt\_Ratio|−0.053|资产负债率 (低负债→高收益)|
|成长|ProfitGrowth\_YoY|+0.343|净利润同比增速|
||RevGrowth\_YoY|+0.243|营业收入同比增速|
|动量|Mom\_1M|−0.174|1 月动量 (A 股短期反转)|
||Mom\_3M|−0.167|3 月动量|
||Mom\_6M|−0.033|6 月动量|
||Mom\_12M\_1M|+0.052|12−1 月动量 (剔除短期反转)|
|波动|Vol\_20D|−0.244|20 日波动率 (低波→高收益)|
||Vol\_60D|−0.234|60 日波动率|
||Beta|−0.082|市场 Beta|
|技术|VolChg\_20D|+0.127|20 日成交量变化率|
||PriceDev\_20D|−0.064|20 日均线偏离 (均值回复)|

## **交易成本模型**

分层成本 = 佣金 (2.5 bps) + 印花税 (5 bps) + 过户费 (0.1 bps) + Almgren-Chriss 市场冲击

|参数|大盘|小盘|说明|
|-|-|-|-|
|Base Slippage|5 bps|15 bps|小盘流动性折价|
|Impact γ (冲击弹性)|0.50|0.65|小盘对交易量更敏感|
|Impact η (量价指数)|1.0|1.5|小盘呈超线性冲击|
|月均成本 (V0)|\~1.6 bps|\~4.3 bps|大盘小盘 \~2.7× 成本差|

## **回测参数**

* **股票池**: 中证 800 成分股 (CSI 300 + CSI 500 并集, 1,476 只历史成分股)
* **区间**: 2017.01 – 2026.06 (V2: 113 个月); V1 区间 2017.01 – 2024.12 (96 个月)
* **频率**: 月度调仓, 月末取日线最后一日
* **预处理**: MAD 3× 去极值 → 板块中性化 → Z-score 标准化 → (V2) GS 正交化
* **行业分类**: 5 大板块 (沪市主板/深市主板/深市中小板/创业板/科创板)
* **因子合成**: V1: 24月滚动 IC\_IR 加权; V2: GS 正交化 + IC\_IR 加权
* **因子正交化**: V2: 截面 GS 正交化 (IC\_IR 降序, max\_correlation=0.85 收缩)
* **大盘择时**: 中证 500 MA20/60 死叉 + 20日年化波动率 80% 分位 → 仓位乘数 1.0/0.3
* **分组回测**: 5 分位法, 做多 Top 20% / 做空 Bottom 20%

## **关键工程决策**

|问题|方案|理由|
|-|-|-|
|重复代码导致行数爆炸|指数去重 + panel 安全去重|CSI 800 同一股票多交易所挂牌|
|财务数据时间对齐|pd.merge\_asof(direction='backward')|PIT (Point-in-Time) 避免前视偏差|
|EastMoney 市值数据不可达|流通市值 ≈ 成交额/换手率|截面排名足够准确|
|申万行业 API 不可达|5 大板块分类 (代码前缀)|行业中性化的代理方案|
|去极值|MAD (中位数绝对偏差)|不受极端值本身影响, 优于均值±3σ|
|Split-Universe 信号量纲|池内 Z-score 后拼接|避免大盘得分天然高于小盘|
|ML 3M gap → 结构性回撤|移除 gap, 改用 1M label|V7 验证 gap 是 MaxDD 根因|
|因子多重共线性|Gram-Schmidt 正交化 (回归残差)|16 因子全保留, 正交后相关性 < 1e-4|
|大盘择时 (Beta 风控)|MA20/60 死叉 + 波动率 80% 分位|与 Alpha 选股严格解耦, 仅缩放敞口|
|**ML 推理 Universe 漂移**|CSI 800 强制对齐 (训练=推理)|全市场 rank → CSI 800 rank, 消除协变量偏移|
|**推理前风控缺失**|ST/停牌/流动性/市值四道过滤|排雷微盘僵尸股, 降低个股爆雷概率|

## **数据来源**

* [akshare](https://github.com/akfamily/akshare) — 开源 A 股数据接口 (Sina/EastMoney 双源日线, 同花顺财务)
* [baostock](http://baostock.com) — PIT 财务数据 (pubDate 门控, 零前看偏差), 用于纸交易管线
* 中证指数 — 成分股列表

## **License**

MIT

