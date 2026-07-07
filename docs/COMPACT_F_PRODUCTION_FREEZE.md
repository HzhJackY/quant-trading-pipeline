# Quant Factor Research

This repository contains the A-share monthly alpha research pipeline and its current production candidate.

## Current Project Status

The traditional data mainline has completed the current validation stage and is now sealed.

The default production candidate is:

- `Compact-F` as the alpha model
- `Top50 Buffer` as the portfolio construction layer
- Buy threshold: `rank <= 35`
- Sell threshold: `rank > 75`
- Rebalance frequency: monthly
- Rebalance timing: month-end
- Weighting: actual holdings equal-weighted and fully invested
- Signal source: frozen `Compact-F` OOS signal

Current guidance:

- Do not continue the technical-factor route for the mainline
- Do not reopen V1/V2 root cause analysis
- Do not re-tune `colsample`, `GS`, `seed`, or `fold`
- Do not mix XHS or other alternative data into the traditional mainline

## Default Production Candidate

| Item | Default |
|---|---|
| Alpha model | `Compact-F` |
| Portfolio construction | `Top50 Buffer` |
| Rebalance frequency | Monthly |
| Rebalance timing | Month-end |
| Buy threshold | `rank <= 35` |
| Sell threshold | `rank > 75` |
| Weighting | Equal-weight actual holdings, fully invested |
| Signal source | Frozen `Compact-F` OOS signal |

## Key Validation Results

| Strategy | Sharpe | MaxDD | Turnover | ROE | ProfitGrowth | EP |
|---|---:|---:|---:|---:|---:|---:|
| Top30 Baseline | 0.4117 | -31.77% | 45.95% | +0.640 | +0.106 | +0.436 |
| Top50 Buffer 35/75 | 0.4132 | -31.29% | 28.04% | +0.648 | +0.108 | +0.385 |

Summary:

- `Top50 Buffer 35/75` is the current default combination layer.
- The main advantage comes from lower turnover and better cost-adjusted behavior, not from a large gross-Sharpe lift.

## Cost Sensitivity

`Top50 Buffer` remains ahead of `Top30 Baseline` on NetSharpe under 10/20/30/50 bps cost assumptions.

## Robustness Check

`Top50 Buffer` passed the following parameter sets:

- `buy <= 30 / sell > 70`
- `buy <= 35 / sell > 75`
- `buy <= 40 / sell > 80`

Observed turnover was roughly `25.77%` to `28.88%`, while ROE, ProfitGrowth, and EP exposures stayed positive.

## Deprecated / Do Not Continue For Now

- Do not continue the technical-factor route
- Do not retune `colsample` / `GS` / `seed` / `fold`
- Do not reopen V1/V2 root cause analysis
- Do not refactor the project structure unless the model research phase is finished and a separate plan is written
- Do not mix XHS factors into the traditional data mainline

## Next Research Track

The next phase is a separate research line:

`XHS Alternative Data Phase 1`

- Entity Alignment
- Monthly XHS Feature Panel
- Sector Relative Z-Score
- Single Factor IC Audit

This track should be executed in a new conversation window and must not modify the `Compact-F` mainline.

## Reproducibility

Key output files:

- `output/compact_f_cost_sensitivity.md`
- `output/compact_f_top50_buffer_robustness.md`
- `output/compact_f_production_candidate_decision.md`
- `output/compact_f_portfolio_construction_results.md`

Key command:

```bash
python run_compact_f_production_validation.py
```

## Project Entry Points

The repository keeps historical research artifacts for traceability, but the current entry point is the production validation path above.

