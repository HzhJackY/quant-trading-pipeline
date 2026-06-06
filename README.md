# Quant Internship Portfolio

量化研究员实习申请项目集合。

## 项目模块

| 模块 | 说明 | 核心输出 |
|---|---|---|
| **因子研究** | 13因子 A 股回测 + IC + 分组 | `run_factor_research.py` |
| **ML 选股** | ElasticNet/LightGBM/XGBoost 滚动训练 | `ml_selection/training.py` |
| **风险模型** | GARCH/EGARCH + VaR 回测 | `risk_model/` |

## 快速开始

```bash
pip install -r requirements.txt
python run_factor_research.py    # 因子研究完整流水线
pytest tests/ -v                 # 运行测试
```

## 项目结构

```
quant/
├── data/
│   ├── fetcher.py              # akshare 数据获取 (日线/成分股/财务)
│   └── cleaner.py              # 预处理 (去极值/中性化/标准化)
├── factor_lib/
│   ├── value.py                # BP, EP
│   ├── momentum.py             # Mom_1M, 3M, 6M, 12M-1M
│   ├── quality.py              # ROE, GrossMargin, DebtRatio
│   ├── volatility.py           # Vol_20D, 60D, Beta
│   └── growth.py               # RevGrowth, EarningsGrowth
├── factor_research/
│   ├── ic_analysis.py          # Rank IC / IC_IR / 衰减
│   ├── group_backtest.py       # 5分组回测 + 多空
│   ├── backtest_engine.py      # 因子合成 + Sharpe/MaxDD/Calmar
│   └── report.py               # 可视化 (IC图/净值/相关性)
├── ml_selection/
│   └── training.py             # 滚动窗口 ElasticNet+XGBoost+LightGBM
├── risk_model/
│   ├── volatility.py           # GARCH / EGARCH
│   └── var_backtest.py         # VaR + Kupiec 检验
├── run_factor_research.py      # 因子研究一键运行
├── resume/
│   ├── resume_cn.md
│   └── resume_en.md
├── tests/
└── requirements.txt
```

## 回测参数

- **区间**: 2017.01 – 2024.12
- **股票池**: 中证 800
- **频率**: 月度调仓
- **预处理**: MAD 3x 去极值 → 行业+市值中性化 → Z-score 标准化

## 数据来源

[akshare](https://github.com/akfamily/akshare) — 免费 A 股日线/财务/指数成分数据
