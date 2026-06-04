# Quant Internship Portfolio

量化研究员实习申请项目集合，包含:
- **因子研究**: 15 因子 A 股回测, IC 分析 + 5 分组回测 + Sharpe Ratio
- **ML 选股**: XGBoost/LightGBM 滚动训练选股
- **风险模型**: GARCH 波动率 + Shrinkage 协方差 + VaR 回测

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/ -v

# 启动 Jupyter 查看报告
jupyter lab
```

## 数据来源

使用 [akshare](https://github.com/akfamily/akshare) 免费获取 A 股数据。

## 项目结构

```
quant/
├── data/            # 数据获取与清洗
├── factor_lib/      # 因子定义 (估值/动量/质量/波动/成长)
├── factor_research/  # 因子回测与分析
├── ml_selection/    # ML 选股
├── risk_model/      # 风险模型
├── notebooks/       # 研究报告
└── resume/          # 简历
```

## 回测区间

2017.01 – 2024.12, 股票池: 中证 800
