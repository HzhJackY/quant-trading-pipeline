# Quant Factor Research — A 股多因子选股系统

从因子研究到纸交易的生产级量化选股管线。核心系统是 **Split-Universe 双模型架构**
（大盘/小盘分层因子建模），配合从 Linear → LightGBM → LambdaRank → Turnover-Aware 的
完整 ML 实验链，以及交易成本建模和 GARCH 风险管理。

## 核心结论

| 指标 | 线性 Alpha (V0) | ML V7 (TO-Aware) | 说明 |
|------|:---:|:---:|------|
| **Net Sharpe** | **1.74** (正交化后) | 0.98 | ★ Gram-Schmidt 正交化释放 IC_IR 潜力 |
| Max Drawdown | −6.5% (正交化后) | −27% | 16 因子全参与, 分散化更充分 |
| Monthly Turnover | 23.7% | 12.6% | TO-Aware loss 有效控换手 |
| Monthly Cost | 5.9 bps | 3.3 bps | 低换手 → 低成本 |

> **诚实结论**：在当前 A 股中证 800 因子体系下，线性 IC_IR 加权合成的稳健性优于
> 机器学习模型的复杂度收益。ML 实验链完整记录了每一版的设计动机、消融对比和失败分析 —
> 这种"证伪"过程本身是量化研究的方法论核心。

完整的消融实验链：`output/ml_v7_final_report.md`

## 项目结构

```
quant/
├── data/
│   ├── fetcher.py                  # 数据获取 (日线/成分股/财务, PIT 对齐)
│   └── cleaner.py                  # 预处理 (MAD 去极值/行业中性化/Z-score)
│
├── factor_lib/                     # 16 因子库 (被 run_factor_research.py 引用)
│   ├── value.py                    #   BP, EP (估值)
│   ├── momentum.py                 #   Mom_1M, 3M, 6M, 12M-1M (动量)
│   ├── quality.py                  #   ROE, Debt_Ratio, Net_Profit_Margin (质量)
│   ├── volatility.py               #   Vol_20D, 60D, Beta (波动率)
│   ├── growth.py                   #   RevGrowth_YoY, ProfitGrowth_YoY (成长)
│   └── technical.py                #   VolChg_20D, PriceDev_20D (技术面)
│
├── factor_research/
│   ├── ic_analysis.py              # Rank IC / IC_IR / IC 衰减分析
│   ├── group_backtest.py           # 5 分组回测 + 多空组合
│   ├── backtest_engine.py          # 多因子合成 (IC_IR 加权/去冗余/符号翻转)
│   ├── split_universe.py           # ★ Split-Universe 双模型系统
│   ├── market_timing.py            # 大盘择时风控 (MA20/60死叉 + 波动率区间)
│   ├── report.py                   # 可视化 (IC 图/净值/相关性矩阵)
│   ├── dynamic_weight.py           # 动态权重分配 (IC_IR 衰减 + Vol 调整)
│   ├── transaction_cost.py         # 分层交易成本模型 (Almgren-Chriss 冲击)
│   ├── ml_engine.py                # ML 基础引擎 (Walk-Forward CV)
│   ├── ml_engine_v7.py             # ★ V7 终版: 1M label + 0M gap + TO-Aware
│   ├── ml_engine_v2.py             # [归档] LambdaRank 实验
│   ├── ml_engine_v5.py             # [归档] TO-Aware 3M gap 实验
│   ├── ml_engine_v6.py             # [归档] Label Blending + Time-Decay
│   └── production_engine.py        # 生产引擎 (3-seed ensemble, 模型持久化)
│
├── paper_trading/                  # 纸交易生产管线
│   ├── paper_trading_pipeline.py   # ★ 每日 cron 入口 + 月末调仓编排
│   ├── data_ingestion.py           # 日线行情 + 基本面并行数据获取
│   ├── factor_compute.py           # 16 因子实时计算
│   ├── state_manager.py            # 信号锚点持久化 + 市场缓存 SQLite
│   └── baostock_adapter.py         # Baostock PIT 财务数据适配 (零前看偏差)
│
├── output/                         # 回测输出 + ML 预测 + 报告
│   ├── ml_v7_final_report.md       # ★ 最终消融报告 (V0 vs V5 vs V7)
│   ├── factor_ic_summary.csv       # 因子 IC 汇总
│   └── production_models/          # 训练好的生产模型 (54 folds × 3 seeds)
│
├── tests/                          # 单元测试
├── resume/                         # 中英文简历
├── run_factor_research.py          # ★ 主入口: 4 阶段因子研究流水线
├── run_split_universe.py           # Split-Universe 双模型分析
├── run_backtest_with_costs.py      # 成本感知回测
├── run_ml_v7.py                    # ★ V7 终版 ML 训练 + 回测
├── run_ml_v6.py                    # V6 消融对比
├── requirements.txt                # 环境依赖 (版本锁定)
└── .gitignore
```

## Runner 脚本指南

| 脚本 | 用途 | 状态 |
|------|------|:--:|
| `run_factor_research.py` | 主入口 — 4 阶段因子研究流水线 (数据预取→面板构建→预处理→分析) | ★ 活跃 |
| `run_split_universe.py` | Split-Universe 大盘/小盘双模型分析 + Baseline 对比 | ★ 活跃 |
| `run_backtest_with_costs.py` | 带交易成本的分层回测 (佣金+印花税+冲击) | ★ 活跃 |
| `run_ml_v7.py` | V7 终版: 1M label + 0M gap + TO-Aware — 训练 + V0/V5/V7 对比 | ★ 活跃 |
| `run_ml_v6.py` | V6 Label Blending + Time-Decay 消融对比 | 保留 |
| `run_ml_backtest.py` | ML 信号 vs 线性信号回测对比 | 保留 |
| `run_dynamic_weight.py` | 动态权重分配 (IC_IR 衰减 + 波动率调整) | 保留 |
| `run_ml_ablation.py` | [归档] V0-V3 早期消融实验 | 归档 |
| `run_ml_lambdarank.py` | [归档] LambdaRank 回测 | 归档 |
| `run_ml_turnover_aware.py` | [归档] V5 λ sweep 实验 | 归档 |
| `diagnose_stock_pool.py` | 股票池诊断 — 采样方法对比 + 行业覆盖分析 | 工具 |

## 快速开始

```bash
# 1. 环境安装
pip install -r requirements.txt

# 2. 因子研究流水线 (首次 ~40min, 缓存后秒过)
python run_factor_research.py

# 3. Split-Universe 双模型分析 (含 Baseline 对比)
python run_split_universe.py

# 4. V7 终版 ML 训练 + 回测 (含 V0/V5/V7 三路对比)
python run_ml_v7.py

# 5. 大盘择时验证
python -c "from factor_research.market_timing import fetch_csi500, plot_timing_history; plot_timing_history(fetch_csi500())"

# 6. 纸交易 (每日 cron, 16:00 收盘后运行)
python paper_trading/paper_trading_pipeline.py

# 单元测试
pytest tests/ -v
```

## 流水线架构

```
Stage 1 (预取) ────→ Stage 2 (面板) ────→ Stage 3 (预处理) ────→ Stage 4 (分析)
  日线+财务缓存      因子计算+PIT对齐    MAD→中性化→Z-score     IC+回测+因子合成
```

4 阶段断点续跑 — 中断后重跑自动从断点继续。状态文件 `.pipeline_state.json` 记录进度,
中间产物 `output/panel.parquet` 和 `output/preprocessed.parquet` 使后续分析无需重新跑全流程。

## Split-Universe 双模型系统

### 核心思想

全市场线性模型忽视了 A 股最核心的结构性差异 — **大市值的机构定价 vs 小市值的散户定价**。
Split-Universe 按流通市值百分位将股票切分为大盘池 (Top 50%) 和小盘池 (Bottom 50%),
各自独立评估因子、独立合成信号, 最后通过池内截面 Z-score 标准化对齐量纲并拼接。

### 因子归属 (数据驱动, 非预设)

| 归属 | 因子 | 经济解释 |
|------|------|----------|
| **大盘型** | Debt_Ratio, Net_Profit_Margin, Mom_1M | 杠杆/利润率是机构定价锚; 短期反转弱 |
| | PriceDev_20D, VolChg_20D | 技术信号在低噪音环境中确定性强 |
| **小盘型** | ProfitGrowth_YoY, RevGrowth_YoY | ★ 成长是小盘核心引擎 (IC_IR 0.47 vs 0.20) |
| | BP, EP | 深度价值在小盘中同样有效 |
| | Beta, Mom_12M_1M, Vol_60D | 长期趋势+低波策略在小盘中更显著 |

## 大盘择时 — Beta 风控系统

与 Alpha 选股严格解耦的大盘择时模块。在月末生成 target portfolio 权重时, 对总敞口
应用缩放乘数。**不修改选股排名, 仅控制总仓位风险暴露。**

### 触发逻辑

| 条件 | 触发 | 乘数 | 信号来源 |
|------|:----:|:----:|----------|
| 正常状态 | 否 | **1.0** (100% 满仓) | — |
| **MA20 死叉** | MA20 < MA60 | **0.3** (30% 仓位) | 中证 500 日线 |
| **波动率飙高** | 20日年化波动率 > 252日80% 分位 | **0.3** (30% 仓位) | 中证 500 日线 |
| 同时触发 | 死叉 + 高波 | **0.3** | — |

```
状态无关: 每期独立判断, 不引入记忆效应
触发时: 30% × [Alpha Top 30 等权]  +  70% 现金
```

### 核心模块

```python
from factor_research.market_timing import (
    fetch_csi500,                    # 获取中证 500 日线 (parquet 缓存)
    compute_market_multiplier,       # 单日乘数
    prepare_timing_multipliers,      # 批量预计算 (回测场景)
    apply_position_sizing,           # 权重缩放
    plot_timing_history,             # 择时历史可视化
    timing_summary,                  # 逐日信号汇总表
)
```

### 整合点

| 管线 | 整合方式 | 效果 |
|------|----------|------|
| **纸交易** `paper_trading_pipeline.py` | 月末调仓自动计算乘数, `_print_top_picks()` 显示缩放权重 | 每只股票权重 = 1/30 × 乘数 |
| **成本回测** `run_backtest_with_costs()` | 新增 `timing_multipliers` 参数 | 毛收益/换手率/成本自动按乘数缩放 |

### 快速验证

```bash
# 查看择时触发历史
python -c "
from factor_research.market_timing import fetch_csi500, plot_timing_history
index_df = fetch_csi500()
plot_timing_history(index_df)
"
```

## ML 实验链 (V0 → V7)

| 版本 | 核心设计 | Label | Gap | Sharpe | MaxDD | TO | 结论 |
|------|----------|:-----:|:---:|:------:|:-----:|:--:|------|
| V0 | Linear IC_IR 加权 | 1M | 0M | **1.13** | −18% | 23.7% | ★ 最优 |
| V2 | LambdaRank 排序学习 | 3M | 3M | — | — | — | 不收敛 |
| V5 | TO-Aware L2 loss, λ=2.0 | 3M | 3M | 0.95 | −27% | 12.9% | 控换手有效, 回撤恶化 |
| V6 | Label Blending + Time-Decay | 混合 | 3M | 0.96 | −27% | 13.4% | 无显著改善 |
| V7 | TO-Aware + 1M label + 0M gap | 1M | 0M | 0.98 | −27% | 12.6% | 回撤未修复 |

> **核心发现**: 3M gap 是结构性 MaxDD 根因 — 模型在 3 个月盲区内信号的预测力衰减严重。
> 移除 gap 后回撤未修复 (V7 -27% vs V0 -18%), 说明 ML 模型的截面排序能力本身弱于
> 线性 IC_IR 加权。

## 因子库 (16 因子)

| 类别 | 因子 | 全市场 IC_IR | 说明 |
|------|------|:-----------:|------|
| 估值 | EP | +0.443 | 盈利/价格 (Earnings Yield) |
| | BP | +0.270 | 净资产/价格 (Book-to-Price) |
| 质量 | Net_Profit_Margin | +0.333 | 销售净利率 |
| | ROE | +0.314 | 净资产收益率 |
| | Debt_Ratio | −0.053 | 资产负债率 (低负债→高收益) |
| 成长 | ProfitGrowth_YoY | +0.343 | 净利润同比增速 |
| | RevGrowth_YoY | +0.243 | 营业收入同比增速 |
| 动量 | Mom_1M | −0.174 | 1 月动量 (A 股短期反转) |
| | Mom_3M | −0.167 | 3 月动量 |
| | Mom_6M | −0.033 | 6 月动量 |
| | Mom_12M_1M | +0.052 | 12−1 月动量 (剔除短期反转) |
| 波动 | Vol_20D | −0.244 | 20 日波动率 (低波→高收益) |
| | Vol_60D | −0.234 | 60 日波动率 |
| | Beta | −0.082 | 市场 Beta |
| 技术 | VolChg_20D | +0.127 | 20 日成交量变化率 |
| | PriceDev_20D | −0.064 | 20 日均线偏离 (均值回复) |

## 交易成本模型

分层成本 = 佣金 (2.5 bps) + 印花税 (5 bps) + 过户费 (0.1 bps) + Almgren-Chriss 市场冲击

| 参数 | 大盘 | 小盘 | 说明 |
|------|:---:|:---:|------|
| Base Slippage | 5 bps | 15 bps | 小盘流动性折价 |
| Impact γ (冲击弹性) | 0.50 | 0.65 | 小盘对交易量更敏感 |
| Impact η (量价指数) | 1.0 | 1.5 | 小盘呈超线性冲击 |
| 月均成本 (V0) | ~1.6 bps | ~4.3 bps | 大盘小盘 ~2.7× 成本差 |

## 回测参数

- **股票池**: 中证 800 成分股 (随机采样 300 只, seed=42)
- **区间**: 2017.01 – 2024.12 (96 个月)
- **频率**: 月度调仓, 月末取日线最后一日
- **预处理**: MAD 3× 去极值 → 板块中性化 → Z-score 标准化
- **行业分类**: 5 大板块 (沪市主板/深市主板/深市中小板/创业板/科创板)
- **因子合成**: ★ Gram-Schmidt 正交化 + 24月滚动 IC_IR 加权 (替换旧版贪心去冗余)
- **因子正交化**: OLS 回归残差法, 逐截面独立正交, 完全共线因子自动归零权重
- **大盘择时**: 中证 500 MA20/60 死叉 + 20日年化波动率 80% 分位 → 仓位乘数 1.0/0.3
- **分组回测**: 5 分位法, 做多 Top 20% / 做空 Bottom 20%

## 关键工程决策

| 问题 | 方案 | 理由 |
|------|------|------|
| 重复代码导致行数爆炸 | 指数去重 + panel 安全去重 | CSI 800 同一股票多交易所挂牌 |
| 财务数据时间对齐 | `pd.merge_asof(direction='backward')` | PIT (Point-in-Time) 避免前视偏差 |
| EastMoney 市值数据不可达 | `流通市值 ≈ 成交额/换手率` | 截面排名足够准确 |
| 申万行业 API 不可达 | 5 大板块分类 (代码前缀) | 行业中性化的代理方案 |
| 去极值 | MAD (中位数绝对偏差) | 不受极端值本身影响, 优于均值±3σ |
| Split-Universe 信号量纲 | 池内 Z-score 后拼接 | 避免大盘得分天然高于小盘 |
| ML 3M gap → 结构性回撤 | 移除 gap, 改用 1M label | V7 验证 gap 是 MaxDD 根因 |
| 因子多重共线性 | Gram-Schmidt 正交化 (回归残差) | 16 因子全保留, 正交后相关性 < 1e-4 |
| 大盘择时 (Beta 风控) | MA20/60 死叉 + 波动率 80% 分位 | 与 Alpha 选股严格解耦, 仅缩放敞口 |

## 数据来源

- [akshare](https://github.com/akfamily/akshare) — 开源 A 股数据接口 (Sina/EastMoney 双源日线, 同花顺财务)
- [baostock](http://baostock.com) — PIT 财务数据 (pubDate 门控, 零前看偏差), 用于纸交易管线
- 中证指数 — 成分股列表

## License

MIT
