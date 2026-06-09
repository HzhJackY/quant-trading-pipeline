## V6 Alpha Engine — Training Report

- **Label:** 40% * ret_1m + 60% * ret_3m -> rank [0,1]
- **Objective:** Custom L2 + lambda*(pred-prev)^2 | lambda = 2.0
- **Time-Decay:** half_life = 12M | w = exp(-dt * ln(2) / 12)
- **Gap:** 3M (prevents 3M leakage)
- **Trained Folds:** 51
- **Feature Cols:** 16
- **Window:** 36M train + 6M val + 1M test
- **Hyperparams:** max_depth=4, num_leaves=24, lr=0.02
- **subsample:** 1.0 (closure alignment)

### Loss Function

```
L = 0.5*(pred - y)^2  +  lambda*0.5*(pred - prev)^2
g = (pred - y)        +  lambda*(pred - prev)
h = 1 + lambda
```

where prev = alpha_signal lagged by 1 month (linear baseline anchor).

### Label Construction

```
y_target = 0.4 * forward_return_1m + 0.6 * forward_return_3m
y_label = cross_sectional_rank(y_target)  # -> [0, 1]
```

### Time-Decay Weights

```
w_i = exp(-dt * ln(2) / 12)
```

Injected via: lgb.Dataset(X, label=y, weight=sample_weights)

### Feature Importance (Average Gain)

| Rank | Feature | Gain | Split |
|------|---------|------|-------|
| 1 | EP_neutral_z_rank | 68428.1 | 73.4 |
| 2 | ROE_neutral_z_rank | 15827.9 | 38.8 |
| 3 | Net_Profit_Margin_neutral_z_rank | 10967.2 | 62.0 |
| 4 | ProfitGrowth_YoY_neutral_z_rank | 5106.1 | 43.5 |
| 5 | BP_neutral_z_rank | 3557.8 | 17.1 |
| 6 | Vol_60D_neutral_z_rank | 2518.8 | 32.7 |
| 7 | RevGrowth_YoY_neutral_z_rank | 1049.8 | 16.3 |
| 8 | Beta_neutral_z_rank | 365.1 | 7.4 |
| 9 | Debt_Ratio_neutral_z_rank | 349.9 | 7.4 |
| 10 | Vol_20D_neutral_z_rank | 42.3 | 1.8 |
| 11 | Mom_6M_neutral_z_rank | 9.9 | 1.3 |
| 12 | Mom_12M_1M_neutral_z_rank | 5.0 | 0.6 |
| 13 | PriceDev_20D_neutral_z_rank | 1.6 | 0.1 |
| 14 | Mom_3M_neutral_z_rank | 0.7 | 0.1 |
| 15 | Mom_1M_neutral_z_rank | 0.4 | 0.0 |
| 16 | VolChg_20D_neutral_z_rank | 0.0 | 0.0 |