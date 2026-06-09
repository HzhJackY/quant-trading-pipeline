## LightGBM Alpha Engine — 训练报告

- **训练 Folds:** 124
- **特征列数:** 16
- **标签方法:** rank
- **特征方法:** rank
- **窗口:** 36M train + 6M val + 1M test
- **超参数:** max_depth=4, num_leaves=24, lr=0.02

### 特征重要性 (平均 Gain)

| Rank | Feature | Gain | Split |
|------|---------|------|-------|
| 1 | EP_neutral_z_rank | 47.2 | 49.8 |
| 2 | ProfitGrowth_YoY_neutral_z_rank | 38.6 | 48.1 |
| 3 | PriceDev_20D_neutral_z_rank | 32.1 | 41.7 |
| 4 | ROE_neutral_z_rank | 29.2 | 35.3 |
| 5 | BP_neutral_z_rank | 29.0 | 39.5 |
| 6 | VolChg_20D_neutral_z_rank | 27.3 | 40.4 |
| 7 | Beta_neutral_z_rank | 26.7 | 41.9 |
| 8 | Mom_1M_neutral_z_rank | 26.6 | 39.0 |
| 9 | Mom_3M_neutral_z_rank | 26.5 | 38.2 |
| 10 | Mom_12M_1M_neutral_z_rank | 24.1 | 36.1 |
| 11 | RevGrowth_YoY_neutral_z_rank | 23.6 | 34.6 |
| 12 | Vol_20D_neutral_z_rank | 23.0 | 31.6 |
| 13 | Mom_6M_neutral_z_rank | 22.8 | 33.5 |
| 14 | Debt_Ratio_neutral_z_rank | 21.3 | 33.9 |
| 15 | Net_Profit_Margin_neutral_z_rank | 19.2 | 30.0 |
| 16 | Vol_60D_neutral_z_rank | 18.3 | 27.5 |