## V7 Final Ablation — 1M + 0M Gap + Turnover-Aware

| Metric | V0: Linear | V5: 3M+gap+TO L2 | V7: 1M+0gap+TO L2 |
|--------|:---:|:---:|:---:|
| Annualized Return | 21.27% | 20.01% | 20.57% |
| Volatility | 16.98% | 18.90% | 19.01% |
| **Sharpe Ratio** | 1.1347 | 0.9527 | 0.9767 |
| **Max Drawdown** | -18.01% | -27.12% | -27.12% |
| Calmar Ratio | 1.1805 | 0.7377 | 0.7585 |
| Win Rate | 61.46% | 62.50% | 64.58% |

### Trading Characteristics

| Metric | V0: Linear | V5: 3M+gap+TO L2 | V7: 1M+0gap+TO L2 |
|--------|:---:|:---:|:---:|
| Monthly Turnover | 23.7% | 12.9% | 12.6% |
| Monthly Cost (bps) | 5.9 | 3.4 | 3.3 |

### V7 vs V5 — Key Improvements

| Metric | V5 | V7 | Delta |
|--------|:---:|:---:|:---:|
| Sharpe | 0.9527 | 0.9767 | +0.0240 |
| MaxDD | -27.12% | -27.12% | 0.00% |
| Turnover | 12.95% | 12.61% | -0.34% |
| Cost | 3.4218 | 3.3400 | -0.0818 |

### Production Readiness Assessment

| Criterion | Met? | Value |
|-----------|------|-------|
| Sharpe > 1.0 | NO | 0.9767 |
| MaxDD < -20% | NO | -27.12% |
| Turnover < 25% | YES | 12.6% |

**1/3 criteria met.**