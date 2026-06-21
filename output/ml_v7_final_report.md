## V7 Final Ablation — 1M + 0M Gap + Turnover-Aware

| Metric | V0: Linear | V5: 3M+gap+TO L2 | V7: 1M+0gap+TO L2 |
|--------|:---:|:---:|:---:|
| Annualized Return | 21.06% | 20.42% | 20.81% |
| Volatility | 17.42% | 19.20% | 19.22% |
| **Sharpe Ratio** | 1.0941 | 0.9594 | 0.9783 |
| **Max Drawdown** | -22.55% | -30.00% | -30.00% |
| Calmar Ratio | 0.9342 | 0.6806 | 0.6935 |
| Win Rate | 62.11% | 63.16% | 64.21% |

### Trading Characteristics

| Metric | V0: Linear | V5: 3M+gap+TO L2 | V7: 1M+0gap+TO L2 |
|--------|:---:|:---:|:---:|
| Monthly Turnover | 27.7% | 18.8% | 18.1% |
| Monthly Cost (bps) | 6.9 | 5.1 | 5.0 |

### V7 vs V5 — Key Improvements

| Metric | V5 | V7 | Delta |
|--------|:---:|:---:|:---:|
| Sharpe | 0.9594 | 0.9783 | +0.0189 |
| MaxDD | -30.00% | -30.00% | 0.00% |
| Turnover | 18.80% | 18.06% | -0.74% |
| Cost | 5.1435 | 4.9561 | -0.1874 |

### Production Readiness Assessment

| Criterion | Met? | Value |
|-----------|------|-------|
| Sharpe > 1.0 | NO | 0.9783 |
| MaxDD < -20% | NO | -30.00% |
| Turnover < 25% | YES | 18.1% |

**1/3 criteria met.**