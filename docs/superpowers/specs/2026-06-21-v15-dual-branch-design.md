# V1.5 Dual-Branch Ensemble Design

Date: 2026-06-21

## Objective

Prevent technical factors from monopolizing LightGBM splits by training
fundamental and technical signals in independent models, then blend their
out-of-sample cross-sectional ranks. Apply stock-level EMA smoothing to reduce
monthly Top-30 turnover.

## Feature Sets

### Fundamental branch (`FEATURES_FUNDA`)

1. `EP_neutral_z`
2. `BP_raw_neutral_z`
3. `SR_ROE_neutral_z`
4. `Net_Profit_Margin_neutral_z`
5. `Operating_Margin_neutral_z`
6. `CFO_to_Earnings_neutral_z`
7. `EPS_YoY_neutral_z`
8. `SR_ProfitGrowth_YoY_neutral_z`
9. `SR_RevGrowth_YoY_neutral_z`
10. `ProfitGrowth_YoY_neutral_z`
11. `RevGrowth_YoY_neutral_z`
12. `Debt_Ratio_neutral_z`
13. `Current_Ratio_neutral_z`
14. `Quick_Ratio_neutral_z`
15. `Equity_Multiplier_neutral_z`

Positive monotonicity constraints apply only to:

- `EP_neutral_z`
- `SR_ROE_neutral_z`
- `SR_ProfitGrowth_YoY_neutral_z`

### Technical branch (`FEATURES_TECH`)

1. `Mom_1M_neutral_z`
2. `Mom_3M_neutral_z`
3. `Mom_6M_neutral_z`
4. `Mom_12M_1M_neutral_z`
5. `RSI_14_neutral_z`
6. `Vol_20D_neutral_z`
7. `Vol_60D_neutral_z`
8. `Vol_120D_neutral_z`
9. `Beta_neutral_z`
10. `Skewness_60D_neutral_z`
11. `MaxDD_60D_neutral_z`
12. `High_Low_Range_20D_neutral_z`
13. `Amihud_Illiquidity_neutral_z`
14. `Dollar_Volume_20D_neutral_z`
15. `Turnover_Volatility_20D_neutral_z`
16. `PriceDev_20D_neutral_z`
17. `VolChg_20D_neutral_z`

The technical branch has no monotonicity constraints.

The two explicit lists contain 32 features. The existing 33-factor single-model
baseline additionally contains `Operating_Cycle_Days_neutral_z`,
`Inventory_Turnover_neutral_z`, and `Receivables_Turnover_neutral_z`, while
some original/reconstructed factors overlap economically. The dual-branch
implementation follows the user-specified lists exactly rather than forcing a
numerical 33-column partition.

## Training Architecture

Both branches reuse `run_v15_experiment.train_single_model` and therefore use
identical:

- panel and labels;
- 36-month train, 6-month validation, 1-month OOS folds;
- seed `42`;
- LightGBM rank-normalized inputs;
- turnover-aware objective with `lambda_turnover=2.0`;
- OOS dates and stock universe.

Branch-specific configs:

| Model | Features | GS | colsample | reg_alpha | Monotonicity |
|---|---:|---|---:|---:|---|
| Model_F | 15 | OFF | 0.75 | 0.10 | EP, SR_ROE, SR_PG = +1 |
| Model_T | 17 | OFF | 0.75 | 0.10 | None |

Outputs:

- `output/production_models_v15_dual/Model_F_oos.parquet`
- `output/production_models_v15_dual/Model_T_oos.parquet`
- branch feature-importance and config files.

## Rank Blend

For each OOS date:

1. Inner-join branch predictions on `date, symbol`.
2. Percentile-rank `pred_f` and `pred_t` independently within the date.
3. Compute:

   `raw_blend_pred = 0.5 * rank_f + 0.5 * rank_t`

Weights are command-line parameters and must sum to one.

## EMA Turnover Control

Sort by `date`, then update each stock independently:

`final_pred_t = 0.6 * raw_blend_pred_t + 0.4 * final_pred_(t-1)`

Rules:

- A new stock with no prior smoothed value uses its current raw blend.
- A stock absent from a month receives no synthetic prediction.
- If it re-enters later, its last observed smoothed signal is retained.
- Non-finite current signals fall back to the last valid smoothed signal; if
  neither exists, the row remains missing and is excluded from portfolio
  selection.

The final prediction file contains:

- `date`
- `symbol`
- `pred_f`
- `pred_t`
- `rank_f`
- `rank_t`
- `raw_blend_pred`
- `final_pred`
- `alpha_signal` (alias of `final_pred` for evaluator compatibility)

## Evaluation

The primary portfolio is long-only, fixed Top 30 stocks each month.

Metrics:

- annualized Sharpe;
- maximum drawdown;
- monthly one-way turnover: fraction of prior Top 30 names that exit;
- mean monthly return;
- number of evaluated OOS months.

Models compared:

1. `Single_033`: existing 33-factor M5, colsample 0.35.
2. `Branch_F`: pure fundamental prediction.
3. `Branch_T`: pure technical prediction.
4. `Dual_Final`: 50/50 rank blend with EMA.

Style exposures for the Top 30:

- `SR_ROE_neutral_z`
- `SR_ProfitGrowth_YoY_neutral_z`
- `EP_neutral_z`
- `BP_raw_neutral_z`

Required verdicts:

- Turnover target: `< 30%`.
- ROE exposure: `> 0`.
- ProfitGrowth exposure: `> 0`.

A Top30/Bottom30 long-short Sharpe is also reported as a diagnostic to separate
stock-selection alpha from broad market exposure.

## Error Handling and Safety

- Hard-fail if either feature list has missing panel columns.
- Hard-fail if branch OOS date sets differ.
- Hard-fail if branch stock intersection on any date is below 30.
- Hard-fail on blend weights that are negative or do not sum to one.
- Run the existing factor pre-flight check before each branch trains.
- Preserve existing single-model artifacts; dual-branch outputs use a separate
  directory.

## Testing

Unit tests cover:

- exact feature membership and branch disjointness;
- monotonicity only on Model_F;
- cross-sectional percentile ranking;
- weighted rank blend;
- EMA initialization, continuation, disappearance, and re-entry;
- fixed Top-30 turnover calculation;
- positive exposure calculation on a controlled fixture;
- invalid weights and mismatched branch predictions.

Integration verification runs:

1. unit tests;
2. syntax compilation;
3. dry-run configuration check;
4. full two-branch training;
5. final evaluation and report generation.
