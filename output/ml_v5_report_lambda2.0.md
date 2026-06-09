## Turnover-Aware Alpha Engine V5 — 训练报告

- **目标函数:** Custom L2 + λ·(ŷ−ŷ₋₁)² | λ = 2.0
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
| 1 | EP_neutral_z_rank | 67936.8 | 73.0 |
| 2 | ROE_neutral_z_rank | 15713.6 | 38.3 |
| 3 | Net_Profit_Margin_neutral_z_rank | 11004.2 | 61.7 |
| 4 | ProfitGrowth_YoY_neutral_z_rank | 4924.7 | 42.7 |
| 5 | BP_neutral_z_rank | 3565.1 | 17.0 |
| 6 | Vol_60D_neutral_z_rank | 2539.0 | 32.8 |
| 7 | RevGrowth_YoY_neutral_z_rank | 1049.5 | 15.8 |
| 8 | Beta_neutral_z_rank | 375.9 | 7.5 |
| 9 | Debt_Ratio_neutral_z_rank | 355.6 | 7.6 |
| 10 | Vol_20D_neutral_z_rank | 47.3 | 2.0 |
| 11 | Mom_6M_neutral_z_rank | 9.0 | 1.3 |
| 12 | Mom_12M_1M_neutral_z_rank | 5.9 | 0.7 |
| 13 | PriceDev_20D_neutral_z_rank | 0.7 | 0.1 |
| 14 | Mom_3M_neutral_z_rank | 0.6 | 0.1 |
| 15 | Mom_1M_neutral_z_rank | 0.2 | 0.0 |
| 16 | VolChg_20D_neutral_z_rank | 0.0 | 0.0 |