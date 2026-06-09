## Turnover-Aware Alpha Engine V5 — 训练报告

- **目标函数:** Custom L2 + λ·(ŷ−ŷ₋₁)² | λ = 0.5
- **Label:** forward_return_3M → 截面 Rank [0,1]
- **Gap:** 3M (防多期标签泄漏)
- **训练 Folds:** 51
- **特征列数:** 16
- **窗口:** 36M train + 6M val + 1M test
- **超参数:** max_depth=4, num_leaves=24, lr=0.02
- **subsample:** 1.0 (闭包对齐)

### 损失函数

```
L = ½(ŷ − y)²  +  λ·½(ŷ − ŷ_{t−1})²
g = (ŷ − y)     +  λ·(ŷ − ŷ_{t−1})
h = 1 + λ
```

其中 ŷ_{t-1} = alpha_signal lagged by 1 month (线性基准锚点).

### 特征重要性 (平均 Gain)

| Rank | Feature | Gain | Split |
|------|---------|------|-------|
| 1 | EP_neutral_z_rank | 16342.5 | 258.2 |
| 2 | ROE_neutral_z_rank | 3428.3 | 135.8 |
| 3 | Net_Profit_Margin_neutral_z_rank | 3141.0 | 215.6 |
| 4 | ProfitGrowth_YoY_neutral_z_rank | 2345.5 | 193.4 |
| 5 | Vol_60D_neutral_z_rank | 1765.5 | 213.8 |
| 6 | BP_neutral_z_rank | 1025.4 | 97.3 |
| 7 | RevGrowth_YoY_neutral_z_rank | 927.1 | 158.7 |
| 8 | Beta_neutral_z_rank | 250.9 | 48.0 |
| 9 | Debt_Ratio_neutral_z_rank | 233.7 | 60.8 |
| 10 | Mom_3M_neutral_z_rank | 90.7 | 25.0 |
| 11 | Vol_20D_neutral_z_rank | 28.2 | 9.7 |
| 12 | Mom_6M_neutral_z_rank | 21.3 | 8.6 |
| 13 | Mom_1M_neutral_z_rank | 21.0 | 10.0 |
| 14 | Mom_12M_1M_neutral_z_rank | 18.4 | 9.2 |
| 15 | PriceDev_20D_neutral_z_rank | 15.9 | 6.7 |
| 16 | VolChg_20D_neutral_z_rank | 3.1 | 2.4 |