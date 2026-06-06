# Quant Factor Research — A 股多因子选股系统

量化研究员实习申请项目。从数据获取到策略回测的完整因子研究流水线,
核心创新是 **Split-Universe 双模型协同系统** (大盘/小盘分层建模)。

## 核心成果

| 指标 | 统一模型 (Baseline) | Split-Universe | 提升 |
|------|:------------------:|:--------------:|:----:|
| Sharpe Ratio | 0.944 | **1.264** | +33.8% |
| 年化收益 | 18.4% | **21.8%** | +3.4pp |
| 最大回撤 | -19.1% | **-13.1%** | -6.0pp |
| Calmar Ratio | 0.962 | **1.669** | +73.5% |
| 胜率 | 35.8% | **66.3%** | +30.5pp |

> 回测区间: 2017.01 – 2024.12 | 股票池: 中证 800 (随机采样 300 只) | 月度调仓

## 项目结构

```
quant/
├── data/
│   ├── fetcher.py                  # 数据获取 (日线/成分股/财务历史, PIT对齐)
│   └── cleaner.py                  # 预处理 (MAD去极值/行业中性化/Z-score)
│
├── factor_lib/                     # 16 因子库
│   ├── value.py                    #   BP, EP (估值)
│   ├── momentum.py                 #   Mom_1M, 3M, 6M, 12M-1M (动量)
│   ├── quality.py                  #   ROE, Debt_Ratio, Net_Profit_Margin (质量)
│   ├── volatility.py               #   Vol_20D, 60D, Beta (波动率)
│   ├── growth.py                   #   RevGrowth_YoY, ProfitGrowth_YoY (成长)
│   └── technical.py                #   VolChg_20D, PriceDev_20D (技术面)
│
├── factor_research/
│   ├── ic_analysis.py              # Rank IC / IC_IR / IC衰减
│   ├── group_backtest.py           # 5分组回测 + 多空组合
│   ├── backtest_engine.py          # 多因子合成 (IC_IR加权/去冗余/符号翻转)
│   ├── report.py                   # 可视化 (IC图/净值/相关性矩阵)
│   └── split_universe.py           # ★ Split-Universe 双模型系统
│
├── ml_selection/
│   └── training.py                 # 滚动窗口 ElasticNet + XGBoost + LightGBM
│
├── risk_model/
│   ├── volatility.py               # GARCH / EGARCH 波动率建模
│   └── var_backtest.py             # VaR 估计 + Kupiec 回测检验
│
├── run_factor_research.py          # 4阶段断点续跑流水线
├── run_split_universe.py           # Split-Universe 运行脚本 (含 Baseline 对比)
│
├── tests/
├── resume/                         # 中英文简历
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt

# 因子研究流水线 (首次 ~40min 缓存后秒过)
python run_factor_research.py

# Split-Universe 双模型分析
python run_split_universe.py

# 单元测试
pytest tests/ -v
```

## 流水线架构

4 阶段断点续跑 — 中断后重跑自动从断点继续:

```
Stage 1 (预取) ──→ Stage 2 (面板) ──→ Stage 3 (预处理) ──→ Stage 4 (分析)
  日线+财务缓存      因子计算+PIT对齐    MAD→中性化→Z-score     IC+回测+合成
```

状态文件 `.pipeline_state.json` 记录进度, 中间产物 `output/panel.parquet` 和
`output/preprocessed.parquet` 使得后续分析无需重新跑全流程。

## Split-Universe 双模型系统

### 核心思想

全市场线性模型忽视了 A 股最核心的结构性差异——**大市值的机构定价 vs 小市值的散户定价**。
Split-Universe 按流通市值百分位将 300 只股票切分为大盘池 (Top 50%) 和小盘池 (Bottom 50%),
各自独立评估因子、独立合成信号, 最后通过池内截面 Z-score 标准化对齐量纲并拼接。

### 因子归属 (数据驱动, 非预设)

| 归属 | 因子 | 经济解释 |
|------|------|----------|
| **大盘型** | Debt_Ratio, Net_Profit_Margin, Mom_1M | 杠杆/利润率是机构定价锚; 短期反转弱 |
| | PriceDev_20D, VolChg_20D | 技术信号在低噪音环境中确定性强 |
| **小盘型** | ProfitGrowth_YoY, RevGrowth_YoY | ★ 成长是小盘核心引擎 (IC_IR 0.47 vs 0.20) |
| | BP, EP | 深度价值在小盘中同样有效 |
| | Beta, Mom_12M_1M, Vol_60D | 长期趋势+低波策略在小盘中更显著 |

### 方法论要点

1. **市值估计**: `流通市值 ≈ 成交额 / 换手率` (日线数据反推, 无需额外 API)
2. **信号对齐**: 池内 Z-score → 拼接 → 全市场有统一量纲的 Alpha
3. **去冗余**: 贪婪算法按 `|IC_IR|` 降序保留, 移除 `|correlation| > 0.7` 的因子
4. **符号翻转**: 稳定负 IC 因子自动取反 (如短周期反转信号)

## 因子库 (16 因子)

| 类别 | 因子 | 全市场 IC_IR | 说明 |
|------|------|:-----------:|------|
| 估值 | EP | +0.443 | 盈利/价格 (Earnings Yield) |
| | BP | +0.270 | 净资产/价格 (Book-to-Price) |
| 质量 | Net_Profit_Margin | +0.333 | 销售净利率 |
| | ROE | +0.314 | 净资产收益率 |
| | Debt_Ratio | -0.053 | 资产负债率 (低负债→高收益) |
| 成长 | ProfitGrowth_YoY | +0.343 | 净利润同比增速 |
| | RevGrowth_YoY | +0.243 | 营业收入同比增速 |
| 动量 | Mom_1M | -0.174 | 1月动量 (A股短期反转) |
| | Mom_3M | -0.167 | 3月动量 |
| | Mom_6M | -0.033 | 6月动量 |
| | Mom_12M_1M | +0.052 | 12-1 月动量 (剔除短期反转) |
| 波动 | Vol_20D | -0.244 | 20日波动率 (低波→高收益) |
| | Vol_60D | -0.234 | 60日波动率 |
| | Beta | -0.082 | 市场 Beta |
| 技术 | VolChg_20D | +0.127 | 20日成交量变化率 |
| | PriceDev_20D | -0.064 | 20日均线偏离 (均值回复) |

## 回测参数

- **股票池**: 中证 800 成分股 (随机采样 300 只, seed=42)
- **区间**: 2017.01 – 2024.12 (96 个月)
- **频率**: 月度调仓, 月末取日线最后一日
- **预处理**: MAD 3× 去极值 → 板块中性化 → Z-score 标准化
- **行业分类**: 5 大板 (沪市主板/深市主板/深市中小板/创业板/科创板)
- **因子合成**: IC_IR 加权 + 符号翻转 + 去冗余 (|corr| > 0.7)
- **分组回测**: 5 分位法, 做多 Top 20% / 做空 Bottom 20%
- **手续费**: 暂不考虑 (学术回测)

## 关键工程决策

| 问题 | 方案 | 理由 |
|------|------|------|
| 重复代码导致行数爆炸 | 指数去重 + panel 安全去重 | CSI 800 同一股票多交易所挂牌 |
| 财务数据时间对齐 | `pd.merge_asof(direction='backward')` | PIT (Point-in-Time) 避免前视偏差 |
| EastMoney 市值数据被墙 | `流通市值 ≈ 成交额/换手率` | 截面排名足够准确 |
| 申万行业 API 全部不可达 | 5 大板分类 (代码前缀) | 行业中性化的代理方案 |
| 去极值 | MAD (中位数绝对偏差) | 不受极端值本身影响, 优于均值±3σ |
| Split-Universe 信号量纲 | 池内 Z-score 后拼接 | 避免大盘得分天然高于小盘 |

## 数据来源

[akshare](https://github.com/akfamily/akshare) — 免费开源 A 股数据接口
- 日线: Sina / EastMoney 双源
- 财务: 同花顺 (THS) 按报告期
- 成分股: 中证指数

## License

MIT
