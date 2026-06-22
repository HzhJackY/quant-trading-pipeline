# Compact-F Portfolio Construction Optimization Design

Date: 2026-06-22

## Objective

Freeze the Compact-F alpha model and optimize only the monthly portfolio
construction layer. All strategies consume the same existing Compact-F OOS
signal and the same next-month forward returns.

## Frozen Inputs and Timing

- Signal file:
  `output/production_models_v15_compact/Compact_F_oos.parquet`
- Factor/return panel:
  `output/training_panel_v15_sr.parquet`
- Rebalance frequency: month-end.
- At month-end `t`, portfolio weights are constructed only from the Compact-F
  OOS signal available at `t`.
- Portfolio return uses `forward_return_1m` from `t` to `t+1`.
- No model retraining, factor changes, signal smoothing, or future information
  is permitted.

## Common Ranking and Weight Rules

- Rank stocks by `alpha_signal` descending each month.
- Rank 1 is the highest signal.
- Ties are broken deterministically by symbol ascending.
- All long-only strategy weights are normalized to sum to 1.0.
- Buffer portfolios are equal-weighted across their actual holdings. If the
  buy zone cannot refill the target size, the portfolio remains fully invested
  across fewer names; no cash is introduced.
- A stock absent from the current OOS universe cannot remain in the portfolio.

## Strategies

### A. Fixed Top30 Baseline

- Hold exactly the highest-ranked 30 stocks.
- Equal weight: `1/30`.

### B. Top30 Buffer

- Initial month: hold Top 30.
- Target size: 30.
- Existing holdings are retained only when current rank is `<=45`.
- New holdings may be added only when current rank is `<=20`.
- Add eligible candidates by rank until the target is reached.
- If the buy zone is exhausted before reaching 30, do not fill from outside the
  buy zone.
- Record each month:
  - sold stock count;
  - newly bought stock count;
  - final holding count;
  - whether the portfolio remained underfilled because the buy zone was
    insufficient.

### C. Top30 Partial Rebalance

- Monthly target portfolio: equal-weight Top 30.
- Smoothing parameter: `alpha=0.5`.
- Before normalization:

  `raw_weight_t = 0.5 * target_weight_t + 0.5 * actual_weight_(t-1)`

- Stocks outside the target decay gradually toward zero.
- Stocks absent from the current tradable universe are removed immediately.
- Raw weights below `1e-6` are removed.
- Remaining weights are normalized to sum to 1.0, so no cash remains.
- Record each month:
  - pre-normalization total weight;
  - post-normalization total weight;
  - final holding count.

### D. Fixed Top40

- Hold exactly the highest-ranked 40 stocks.
- Equal weight: `1/40`.

### E. Fixed Top50

- Hold exactly the highest-ranked 50 stocks.
- Equal weight: `1/50`.

### F. Top50 Buffer

- Initial month: hold Top 50.
- Target size: 50.
- Existing holdings are retained only when current rank is `<=75`.
- New holdings may be added only when current rank is `<=35`.
- Add eligible candidates by rank until the target is reached.
- If the buy zone is exhausted before reaching 50, do not fill outside the buy
  zone.
- Use the same monthly audit fields as Strategy B.

## Return, Turnover, and Exposure Accounting

For weights selected at month-end `t`:

`portfolio_return_t = sum(weight_i,t * forward_return_1m_i,t)`

Monthly one-way turnover:

`turnover_t = 0.5 * sum(abs(weight_i,t - weight_i,t-1))`

The first month has no turnover observation and is excluded from average
turnover.

Weighted style exposures:

- `SR_ROE_neutral_z`
- `SR_ProfitGrowth_YoY_neutral_z`
- `EP_neutral_z`

Monthly exposure is the weight-weighted mean. Reported exposure is the mean
across evaluated months.

## Evaluation Metrics

For each strategy:

- annualized Sharpe;
- maximum drawdown;
- average monthly one-way turnover;
- mean monthly return;
- monthly hit rate (`return > 0`);
- average ROE exposure;
- average ProfitGrowth exposure;
- average EP exposure;
- annualized Sharpe by calendar year.

Dates without a realized next-month return, including the final OOS month, are
excluded from return metrics. Portfolio state and audit logs are still
retained.

## Acceptance and Selection

A candidate passes only when:

1. Sharpe is greater than or equal to the Fixed Top30 baseline;
2. MaxDD is no worse than baseline MaxDD minus 2 percentage points;
3. turnover is `<=35%`, with `<=30%` preferred;
4. ROE and ProfitGrowth exposures are both positive.

If no candidate satisfies every condition, Fixed Top30 remains the selected
production method.

## Outputs

- `run_v15_portfolio_optimization.py`
- `output/compact_f_portfolio_construction_results.md`
- `output/compact_f_portfolio_construction_monthly.csv`
- `output/compact_f_portfolio_construction_yearly.csv`

The monthly CSV contains strategy returns, turnover, exposures, weight totals,
holding counts, and Buffer buy/sell/underfill audit fields.
