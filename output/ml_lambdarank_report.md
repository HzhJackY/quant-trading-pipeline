## LambdaRank Alpha Engine V2 — 训练报告

- **目标函数:** lambdarank | **Horizon:** 3M
- **Gap:** 3M (防多期标签泄漏)
- **训练 Folds:** 117
- **特征列数:** 50
- **窗口:** 36M train + 6M val + 1M test
- **eval_at:** [10, 30]
- **超参数:** max_depth=4, num_leaves=24, lr=0.02
- **差分特征:** 启用
- **类别特征:** 启用

### 特征重要性 (平均 Gain)

| Rank | Feature | Gain | Split |
|------|---------|------|-------|
| 1 | EP_neutral_z_d3m | 53.5 | 10.3 |
| 2 | ROE_neutral_z_rank | 42.3 | 11.4 |
| 3 | EP_neutral_z_rank | 35.2 | 10.5 |
| 4 | Mom_6M_neutral_z_rank | 31.5 | 6.3 |
| 5 | ProfitGrowth_YoY_neutral_z_rank | 27.2 | 9.0 |
| 6 | ROE_neutral_z_d3m | 25.3 | 6.5 |
| 7 | BP_neutral_z_rank | 19.1 | 8.3 |
| 8 | mcap_bin | 18.3 | 6.6 |
| 9 | Debt_Ratio_neutral_z_d3m | 16.1 | 7.0 |
| 10 | Beta_neutral_z_rank | 15.5 | 7.2 |
| 11 | Debt_Ratio_neutral_z_rank | 14.6 | 6.1 |
| 12 | ProfitGrowth_YoY_neutral_z_d3m | 11.6 | 5.0 |
| 13 | Net_Profit_Margin_neutral_z_d3m | 11.4 | 4.7 |
| 14 | RevGrowth_YoY_neutral_z_d3m | 10.6 | 5.6 |
| 15 | ProfitGrowth_YoY_neutral_z_d1m | 9.9 | 4.4 |
| 16 | Mom_3M_neutral_z_rank | 9.8 | 3.3 |
| 17 | Net_Profit_Margin_neutral_z_rank | 9.7 | 5.5 |
| 18 | Vol_60D_neutral_z_rank | 9.2 | 3.9 |
| 19 | EP_neutral_z_d1m | 9.0 | 4.2 |
| 20 | RevGrowth_YoY_neutral_z_rank | 7.9 | 4.3 |
| 21 | BP_neutral_z_d3m | 7.8 | 4.2 |
| 22 | ROE_neutral_z_d1m | 7.1 | 2.9 |
| 23 | Mom_12M_1M_neutral_z_rank | 6.9 | 3.7 |
| 24 | Mom_6M_neutral_z_d3m | 6.4 | 3.6 |
| 25 | Debt_Ratio_neutral_z_d1m | 5.6 | 2.6 |
| 26 | PriceDev_20D_neutral_z_rank | 5.5 | 2.3 |
| 27 | Net_Profit_Margin_neutral_z_d1m | 5.1 | 2.4 |
| 28 | Vol_20D_neutral_z_rank | 4.7 | 2.6 |
| 29 | RevGrowth_YoY_neutral_z_d1m | 4.5 | 2.1 |
| 30 | Mom_3M_neutral_z_d3m | 4.3 | 3.2 |
| 31 | Mom_12M_1M_neutral_z_d3m | 4.2 | 3.0 |
| 32 | Vol_60D_neutral_z_d3m | 4.0 | 3.1 |
| 33 | Mom_1M_neutral_z_d3m | 3.6 | 2.6 |
| 34 | PriceDev_20D_neutral_z_d3m | 3.3 | 2.4 |
| 35 | Vol_20D_neutral_z_d3m | 3.3 | 2.3 |
| 36 | Mom_3M_neutral_z_d1m | 3.2 | 2.2 |
| 37 | Beta_neutral_z_d3m | 3.1 | 2.6 |
| 38 | Vol_60D_neutral_z_d1m | 3.0 | 1.9 |
| 39 | VolChg_20D_neutral_z_rank | 2.8 | 1.9 |
| 40 | BP_neutral_z_d1m | 2.5 | 2.0 |
| 41 | Mom_6M_neutral_z_d1m | 2.5 | 2.0 |
| 42 | Mom_1M_neutral_z_rank | 2.3 | 1.6 |
| 43 | Mom_12M_1M_neutral_z_d1m | 2.3 | 2.2 |
| 44 | PriceDev_20D_neutral_z_d1m | 2.1 | 1.8 |
| 45 | Beta_neutral_z_d1m | 2.0 | 2.0 |
| 46 | Mom_1M_neutral_z_d1m | 2.0 | 1.9 |
| 47 | VolChg_20D_neutral_z_d3m | 1.9 | 1.7 |
| 48 | Vol_20D_neutral_z_d1m | 1.8 | 1.7 |
| 49 | VolChg_20D_neutral_z_d1m | 1.2 | 1.3 |
| 50 | board_cat | 0.9 | 1.1 |