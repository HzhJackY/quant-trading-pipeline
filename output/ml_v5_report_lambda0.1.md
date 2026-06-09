## Turnover-Aware Alpha Engine V5 — 训练报告

- **目标函数:** Custom L2 + λ·(ŷ−ŷ₋₁)² | λ = 0.1
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
| 1 | EP_neutral_z_rank | 1229.0 | 287.0 |
| 2 | ROE_neutral_z_rank | 497.7 | 210.1 |
| 3 | ProfitGrowth_YoY_neutral_z_rank | 479.5 | 284.9 |
| 4 | BP_neutral_z_rank | 351.3 | 261.6 |
| 5 | Net_Profit_Margin_neutral_z_rank | 300.8 | 199.4 |
| 6 | RevGrowth_YoY_neutral_z_rank | 281.2 | 253.7 |
| 7 | Vol_60D_neutral_z_rank | 250.5 | 185.1 |
| 8 | Debt_Ratio_neutral_z_rank | 238.3 | 248.3 |
| 9 | Mom_12M_1M_neutral_z_rank | 127.0 | 147.6 |
| 10 | Beta_neutral_z_rank | 124.2 | 126.4 |
| 11 | Mom_3M_neutral_z_rank | 116.1 | 119.8 |
| 12 | Mom_6M_neutral_z_rank | 80.0 | 98.8 |
| 13 | Mom_1M_neutral_z_rank | 39.7 | 62.8 |
| 14 | Vol_20D_neutral_z_rank | 36.2 | 43.1 |
| 15 | PriceDev_20D_neutral_z_rank | 34.5 | 55.3 |
| 16 | VolChg_20D_neutral_z_rank | 26.6 | 46.5 |