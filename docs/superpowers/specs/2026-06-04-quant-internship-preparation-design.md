# 量化研究员实习申请准备 — 设计文档

> 创建日期: 2026-06-04
> 用户: 浙大数学本 + 复旦数学研一, 概率论与随机分析强项, 依赖 AI 编程
> 目标: 2026 暑期实习 / 日常实习, 找量化研究岗位

## 一、概述

### 1.1 核心策略: 分层递进

```
第 0 层 ──── 简历 + 环境搭建    (1-2 天)  →  可投递状态
第 1 层 ──── 因子研究 MVP       (1-2 周)  →  第一个完整项目
第 2 层 ──── ML 选股 MVP        (第 3-4 周) →  第二个完整项目
第 3 层 ──── 差异化风险模型     (第 2-3 月) →  利用随机分析优势
```

每层都有独立可交付物, 完成即可更新简历投递, 不依赖后续层级。

### 1.2 技术栈

| 用途 | 工具 |
|------|------|
| 语言 | Python 3.10+ |
| 数据处理 | pandas, numpy |
| 数据获取 | akshare (主), baostock (补充) |
| 可视化 | matplotlib, seaborn, plotly |
| 统计建模 | statsmodels, scipy |
| 机器学习 | scikit-learn, LightGBM, XGBoost |
| 深度(可选) | PyTorch |
| 环境管理 | pip + venv (或 conda) |

### 1.3 数据源

akshare 免费覆盖 A 股日线、财务、指数成分、行业分类。回测区间 2017.01–2024.12, 约 8 年。

---

## 二、项目目录结构

```
quant/
├── README.md
├── requirements.txt
├── data/
│   ├── fetcher.py              # 数据获取统一接口
│   ├── cleaner.py              # 清洗(去极值/中性化/标准化)
│   ├── raw/                    # 原始数据缓存(CSV)
│   └── processed/              # 因子就绪数据
├── factor_lib/
│   ├── __init__.py
│   ├── value.py                # 估值类因子
│   ├── momentum.py             # 动量类因子
│   ├── quality.py              # 质量类因子
│   ├── volatility.py           # 波动率类因子
│   └── growth.py               # 成长类因子
├── factor_research/
│   ├── __init__.py
│   ├── backtest_engine.py      # 回测引擎(分层+多空)
│   ├── ic_analysis.py          # IC / IC_IR / 衰减
│   ├── group_backtest.py       # 分组收益分析
│   └── report.py               # 报告生成
├── ml_selection/
│   ├── __init__.py
│   ├── features.py             # 特征工程
│   ├── model.py                # 训练框架(XGBoost/LightGBM)
│   ├── backtest.py             # 模型选股回测
│   └── evaluation.py           # 模型评估指标
├── risk_model/
│   ├── __init__.py
│   ├── volatility_model.py     # GARCH/随机波动率
│   ├── covariance.py           # 协方差估计(Shrinkage)
│   └── risk_decomp.py          # 风险分解/归因
├── notebooks/
│   ├── 00_data_explore.ipynb   # 数据概览
│   ├── 01_factor_report.ipynb  # 因子研究报告
│   ├── 02_ml_backtest.ipynb    # ML选股回测报告
│   └── 03_risk_model.ipynb     # 风险模型报告
├── resume/
│   ├── resume_cn.md            # 中文简历源文件
│   ├── resume_cn.pdf
│   ├── resume_en.md
│   └── resume_en.pdf
├── tests/                      # 单元测试
│   ├── test_factor_lib.py
│   ├── test_backtest.py
│   └── test_ml.py
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-06-04-quant-internship-preparation-design.md
```

---

## 三、第 0 层: 简历 + 环境搭建

### 3.1 简历设计

**核心策略**: 数学背景前置, 研究经历学术化包装, 量化项目用 STAR 原则。

**中文简历结构** (按重要性排序):
1. 基本信息(姓名/电话/邮箱/GitHub)
2. 教育背景(ZJU 数学本科 → 复旦数院硕士, 注明核心课程和成绩)
3. 量化项目(因子研究 + ML 选股, 随着项目推进逐步填充)
4. 研究经历(本科论文/科研项目, 重点突出概率论和随机分析)
5. 技能(数学:随机分析/概率论/统计推断 | 编程:Python/pandas/numpy/scikit-learn | 工具:LaTeX/Jupyter)
6. 奖项/课外

### 3.2 环境搭建

```bash
pip install akshare baostock pandas numpy matplotlib seaborn \
            plotly scipy statsmodels scikit-learn lightgbm xgboost \
            jupyter black pytest tqdm
```

`data/fetcher.py` 为核心模块, 提供:
- `get_daily(stock_list, start, end, fields)` → 日线行情
- `get_financial(stock_list, report_dates, fields)` → 财务数据
- `get_index_members(index_code)` → 指数成分股
- `get_industry(stock_list)` → 申万行业分类

---

## 四、第 1 层: 因子研究 MVP

### 4.1 因子池 (~15 个, 5 大类)

| 类别 | 因子 | 公式/说明 |
|------|------|-----------|
| **估值 Value** | BP | 总资产 / 总市值 |
| | EP | 净利润 / 总市值 |
| **动量 Momentum** | Mom_1M | 过去 1 个月收益(跳过最近 5 日) |
| | Mom_3M | 过去 3 个月收益 |
| | Mom_6M | 过去 6 个月收益 |
| | Mom_12M_1M | 过去 12 个月收益(跳过最近 1 个月) |
| **波动率 Vol** | Vol_20D | 过去 20 个交易日收益率标准差 |
| | Vol_60D | 过去 60 个交易日收益率标准差 |
| | Downside_Vol | 上过去 60 个交易日的下行波动率 |
| | Beta | 过去 60 个交易日市场 Beta |
| **质量 Quality** | ROE | 净利润 / 净资产 |
| | Gross_Margin | 毛利 / 营收 |
| | Debt_Ratio | 资产负债率 |
| **成长 Growth** | Rev_Growth_YoY | 营收同比增长率 |
| | Earnings_Growth | 净利润同比增长率 |

### 4.2 数据预处理流水线

```
原始数据(akshare)
    │
    ▼
┌─────────────────┐
│ 股票池过滤       │  中证800, 剔除ST/上市<180天
└────────┬────────┘
         ▼
┌─────────────────┐
│ 缺失值处理       │  因子覆盖率 < 80% 则剔除该因子/该期
└────────┬────────┘
         ▼
┌─────────────────┐
│ 去极值            │  3x MAD 缩尾
└────────┬────────┘
         ▼
┌─────────────────┐
│ 中性化(可选)      │  行业哑变量 + 对数市值回归取残差
└────────┬────────┘
         ▼
┌─────────────────┐
│ 标准化            │  截面 Z-score
└────────┬────────┘
         ▼
    因子就绪数据
```

### 4.3 因子检验指标

| 指标 | 计算方式 | 判断标准 |
|------|----------|----------|
| **Rank IC 均值** | Spearman corr(因子截面排名, 下期收益排名) | \|IC\| > 0.03 有意义 |
| **IC_IR** | mean(IC) / std(IC) | > 0.5 较好, > 0.7 优秀 |
| **IC >0 胜率** | IC > 0 的月份数 / 总月份数 | > 55% |
| **分层超额收益** | 5分组Q5相对Q1的多空收益 | 单调递减为佳 |
| **因子相关性** | 因子间截面 Pearson / Spearman | 寻找低相关因子组合 |
| **因子收益率** | Fama-MacBeth 截面回归系数 | t 值 > 2 |

### 4.4 组合回测参数

| 参数 | 设定 |
|------|------|
| 股票池 | 中证 800 |
| 调仓频率 | 月末调仓 T 日收盘 |
| 回测区间 | 2017.01–2024.12 |
| 分组数 | 5 组(Q1 最弱, Q5 最弱) |
| 多空组合 | Long Q5, Short Q1 |
| 交易成本 | 单边 0.1%(万10) |
| 基准 | 中证 800 等权 |

### 4.5 报告产出

`notebooks/01_factor_report.ipynb`:
1. 数据概览(样本量/时间区间/股票数量)
2. 单因子 IC 分析(IC 时序图 + IC 分布直方图 + 累计 IC)
3. 分层回测净值曲线(5 分组 + 多空)
4. 因子相关性矩阵热力图
5. Top/Bottom 组合风险指标表(年化收益/波动率/Sharpe/最大回撤)
6. 因子合成与多因子组合回测

---

## 五、第 2 层: ML 选股 MVP

### 5.1 核心思路

把因子研究中的 ~15 个因子作为特征, 用 ML 模型预测股票下月收益(或涨跌分类), 根据预测排名构建投资组合。

### 5.2 模型设计

| 项目 | 选择 |
|------|------|
| 目标变量 | 下月收益率(回归) 或 下月涨跌 >0(分类) |
| 训练方式 | 滚动窗口(expanding window): 前 60 个月训练, 下 1 个月预测 |
| 特征集 | ~15 个因子 + 行业虚拟变量 + 对数市值 |
| 模型 1 | **XGBoost**(回归, 评估 MSE/MAE) |
| 模型 2 | **LightGBM**(回归, 评估 MSE/MAE) |
| 模型 3 | **Logistic/ElasticNet**(线性基准, 评估分类准确率) |
| 过拟合防护 | 时间序列交叉验证 + 特征重要性稳定性监控 |

### 5.3 评估指标

| 指标 | 说明 |
|------|------|
| **Rank IC** | 预测值排名 vs 实际收益排名的 Spearman 相关 |
| **多空收益** | 预测 Top 20% Long / Bottom 20% Short |
| **Sharpe Ratio** | 多空组合的年化 Sharpe |
| **换手率** | 逐期持仓变动, 避免过度交易 |
| **特征重要性** | SHAP 值 / 特征重要性, 与因子 IC 交叉验证 |

### 5.4 报告产出

`notebooks/02_ml_backtest.ipynb`:
1. 特征重要性排名(与因子 IC 分析对比)
2. 滚动窗口训练过程的 IC 时序(稳定性)
3. 多空净值曲线(因子基准 vs ML 模型)
4. 分行业选股效果
5. 换手率和交易成本分析

---

## 六、第 3 层: 差异化——随机波动率与风险模型

### 6.1 差异化定位

大多数量化申请者的项目停留在因子 + ML。你的核心优势是**概率论与随机分析**, 这层做的是:
- GARCH 族波动率建模
- 协方差矩阵收缩估计(Shrinkage)
- 组合风险分解/归因

这在面试中能让你脱颖而出——面试官很少见到申请者能把随机微积分应用到风控项目。

### 6.2 核心模块

| 模块 | 内容 | 工具 |
|------|------|------|
| **波动率建模** | GARCH(1,1) / EGARCH 拟合个股波动率, 与历史波动率对比 | `arch` 库 |
| **协方差估计** | Ledoit-Wolf Shrinkage, 与样本协方差对比 | `sklearn.covariance` |
| **风险分解** | 组合波动率分解为因子暴露 + 特质风险 | 自实现 |
| **VaR/CVaR** | 参数法 + 历史模拟法 VaR 回测 | 自实现 |

### 6.3 报告产出

`notebooks/03_risk_model.ipynb`:
1. 个股波动率模型拟合与诊断
2. Shrinkage 协方差 vs 样本协方差的 OOS 误差对比
3. 组合风险分解瀑布图
4. VaR 回测(实际亏损超过 VaR 的天数 / 总天数)

---

## 七、时间线

```
Week 1 (6/4–6/11)
├── Day 1-2: 第 0 层 — 简历初稿 + 环境搭建 + 数据管道打通
├── Day 3-5: 因子定义 + 预处理流水线
├── Day 6-7: IC 分析 + 分层回测 + 可视化
└── 产出: 简历(含因子项目摘要) 可投递 + 因子报告 Notebook

Week 2 (6/12–6/19)
├── 因子研究收尾: 多因子合成 + 组合回测
├── 代码整理 + README
└── 产出: 完整因子研究项目

Week 3-4 (6/20–7/3)
├── ML 特征工程 + 训练框架
├── 滚动窗口训练 + 模型评估
├── ML vs 因子的对比分析报告
└── 产出: 简历更新(追加 ML 项目)

Month 2-3 (7/4–8/30)
├── 第 3 层: 风险模型
├── 简历持续迭代
├── GitHub 仓库整理 + 项目文档完善
└── 产出: 完整 Portfolio
```

---

## 八、成功标准

### 投递策略
- 第 0 层完成: 即刻投递日常实习岗位
- 第 1 层完成: 开始投递正式暑期实习
- 第 2 层完成: 覆盖所有量化研究实习岗位

### 项目质量自检
- [ ] 因子 IC 分析完整, IC_IR > 0.5 的因子至少 3 个
- [ ] 分层回测单调性良好(Q5 > Q4 > ... > Q1)
- [ ] ML 模型 IC 显著优于线性基准
- [ ] 代码有清晰注释, Notebook 有完整解读
- [ ] 每个分析结论能用金融/统计直觉解释

### 面试准备
- 能解释每个因子的经济学逻辑(不是纯数据挖掘)
- 能讨论 IC 衰减的可能原因
- 能回答"为什么选 XGBoost 而不是线性模型"
- 能讨论过拟合在金融 ML 中意味着什么(不像 CV 那么简单)
- 能从随机分析角度解释 GARCH/波动率的适用场景和局限
