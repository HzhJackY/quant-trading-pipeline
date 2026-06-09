## Turnover-Aware Alpha Engine V5 — 训练报告

- **目标函数:** Custom L2 + λ·(ŷ−ŷ₋₁)² | λ = 0.0
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
| 1 | BP_neutral_z_rank | 205.2 | 176.2 |
| 2 | Debt_Ratio_neutral_z_rank | 192.6 | 195.2 |
| 3 | ProfitGrowth_YoY_neutral_z_rank | 190.3 | 188.6 |
| 4 | RevGrowth_YoY_neutral_z_rank | 174.2 | 190.8 |
| 5 | EP_neutral_z_rank | 173.9 | 177.5 |
| 6 | ROE_neutral_z_rank | 165.8 | 170.0 |
| 7 | Net_Profit_Margin_neutral_z_rank | 145.3 | 148.0 |
| 8 | Mom_12M_1M_neutral_z_rank | 129.1 | 140.1 |
| 9 | Beta_neutral_z_rank | 80.2 | 103.6 |
| 10 | Mom_6M_neutral_z_rank | 79.6 | 96.0 |
| 11 | Vol_60D_neutral_z_rank | 74.4 | 88.1 |
| 12 | Mom_3M_neutral_z_rank | 72.4 | 83.2 |
| 13 | PriceDev_20D_neutral_z_rank | 40.6 | 60.3 |
| 14 | Vol_20D_neutral_z_rank | 35.0 | 49.7 |
| 15 | Mom_1M_neutral_z_rank | 33.3 | 46.6 |
| 16 | VolChg_20D_neutral_z_rank | 33.1 | 55.7 |