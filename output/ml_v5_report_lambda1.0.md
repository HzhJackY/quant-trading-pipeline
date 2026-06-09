## Turnover-Aware Alpha Engine V5 — 训练报告

- **目标函数:** Custom L2 + λ·(ŷ−ŷ₋₁)² | λ = 1.0
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
| 1 | EP_neutral_z_rank | 40944.7 | 161.0 |
| 2 | ROE_neutral_z_rank | 8621.9 | 80.5 |
| 3 | Net_Profit_Margin_neutral_z_rank | 7199.3 | 132.1 |
| 4 | ProfitGrowth_YoY_neutral_z_rank | 3962.5 | 98.0 |
| 5 | Vol_60D_neutral_z_rank | 2744.1 | 107.7 |
| 6 | BP_neutral_z_rank | 1919.0 | 40.8 |
| 7 | RevGrowth_YoY_neutral_z_rank | 1131.2 | 54.4 |
| 8 | Beta_neutral_z_rank | 325.4 | 19.6 |
| 9 | Debt_Ratio_neutral_z_rank | 317.7 | 23.1 |
| 10 | Vol_20D_neutral_z_rank | 39.8 | 3.7 |
| 11 | Mom_3M_neutral_z_rank | 14.6 | 1.9 |
| 12 | Mom_6M_neutral_z_rank | 10.7 | 2.3 |
| 13 | Mom_12M_1M_neutral_z_rank | 4.1 | 1.2 |
| 14 | PriceDev_20D_neutral_z_rank | 3.5 | 0.5 |
| 15 | Mom_1M_neutral_z_rank | 2.5 | 0.4 |
| 16 | VolChg_20D_neutral_z_rank | 0.4 | 0.1 |