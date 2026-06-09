## V6 Label Blending + Time-Decay — Ablation Study

- **AUM:** 5000 万
- **Stock Selection:** Top 30% split-universe equal-weight
- **Cost Model:** Almgren-Chriss impact + tiered fees

### V6 Innovations

| Innovation | Detail |
|------------|--------|
| **Label Blending** | y = 0.4 * ret_1m + 0.6 * ret_3m -> rank [0,1] |
| **Time-Decay Weights** | w = exp(-dt * ln(2) / 12), via lgb.Dataset(weight=...)  |
| **Turnover Penalty** | Custom objective L = 0.5*(p-y)^2 + 2.0*0.5*(p-prev)^2 |
| **Gap**              | 3M (prevents leakage through 3M label component) |

### Performance Comparison (Net of Costs)

| Metric | V0: Linear | V5: TO-Aware (lambda=2.0) | V6: V5 + Blend + Decay |
|--------|:---:|:---:|:---:|
| Annualized Return | 21.27% | 20.01% | 20.07% |
| Annualized Volatility | 16.98% | 18.90% | 18.93% |
| **Sharpe Ratio** | 1.1347 | 0.9527 | 0.9545 |
| **Max Drawdown** | -18.01% | -27.12% | -27.12% |
| Calmar Ratio | 1.1805 | 0.7377 | 0.7400 |
| Monthly Win Rate | 61.46% | 62.50% | 62.50% |

### Trading Characteristics

| Metric | V0: Linear | V5: TO-Aware (lambda=2.0) | V6: V5 + Blend + Decay |
|--------|:---:|:---:|:---:|
| Monthly One-Way Turnover | 23.7% | 12.9% | 13.2% |
| Monthly Avg Cost (bps) | 5.9 | 3.4 | 3.5 |

### V6 vs V5 — Improvement Analysis

| Metric | V5 (lambda=2.0) | V6 | Delta | % Change |
|--------|:---:|:---:|:---:|:---:|
| Sharpe Ratio | 0.9527 | 0.9545 | 0.0018 | +0.2% |
| Annualized Return | 20.01% | 20.07% | 0.06% | +0.3% |
| Max Drawdown | -27.12% | -27.12% | 0.00% | -0.0% |
| Calmar Ratio | 0.7377 | 0.7400 | 0.0023 | +0.3% |
| Monthly Turnover | 12.95% | 13.20% | 0.26% | +2.0% |
| Monthly Cost (bps) | 3.4218 | 3.4732 | +0.0515 | +1.5% |

### Training Time

| Config | Wall Time | Train Time |
|--------|-----------|------------|
| V0: Linear | 1s | 0s |
| V5: TO-Aware (lambda=2.0) | 1s | 0s |
| V6: V5 + Blend + Decay | 4s | 5s |