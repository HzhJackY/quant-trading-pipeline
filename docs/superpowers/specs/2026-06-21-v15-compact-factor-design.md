# V1.5 Compact Factor Model Design

Date: 2026-06-21

## Objective

Test whether a compact, fundamental-led monthly model restores robustness by
removing high-variance technical features that dominated the 33-factor GBDT.

The experiment compares three models under identical labels, walk-forward
folds, seed, LightGBM architecture, and evaluation rules.

## Frequency and Evaluation Contract

- Model frequency: monthly.
- Signals are generated at each month-end.
- Daily OHLCV data may be used to construct a month-end feature, but does not
  imply daily prediction or daily rebalancing.
- Primary portfolio: fixed Top 30 long-only.
- Diagnostic portfolio: fixed Top 30 minus Bottom 30.
- Never compare the fixed Top-30 long-only Sharpe directly with historical
  percentile long-short Sharpe.

## Fundamental Core

All three models use this exact 15-factor fundamental set:

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

Positive monotonicity constraints apply to:

- `EP_neutral_z`
- `SR_ROE_neutral_z`
- `SR_ProfitGrowth_YoY_neutral_z`

`ROE_Stability` is excluded because the current source column has zero usable
coverage. It may only be added after reconstruction from a valid PIT ROE
history.

## Model Matrix

| Model | Feature Set | Total |
|---|---|---:|
| Compact-F | Fundamental core only | 15 |
| Compact-FT | Fundamental core + `Mom_3M_neutral_z` + `Vol_60D_neutral_z` | 17 |
| Compact-FT3 | Compact-FT + `Mom_6M_neutral_z` | 18 |

The following technical features are deliberately excluded:

- `Mom_1M_neutral_z`
- `RSI_14_neutral_z`
- `PriceDev_20D_neutral_z`
- `VolChg_20D_neutral_z`
- `High_Low_Range_20D_neutral_z`
- `MaxDD_60D_neutral_z`
- `Skewness_60D_neutral_z`
- `Amihud_Illiquidity_neutral_z`
- `Dollar_Volume_20D_neutral_z`
- `Turnover_Volatility_20D_neutral_z`
- `Vol_20D_neutral_z`
- `Vol_120D_neutral_z`
- `Beta_neutral_z`
- `Mom_12M_1M_neutral_z`

## Training Configuration

All models reuse `run_v15_experiment.train_single_model`:

- panel: `output/training_panel_v15_sr.parquet`;
- train/validation/test: 36/6/1 months;
- seed: 42;
- GS: OFF;
- `colsample_bytree`: 0.75;
- `learning_rate`: 0.05;
- `reg_alpha`: 0.10;
- turnover-aware objective: `lambda_turnover=2.0`;
- identical monthly labels and OOS dates.

No EMA smoothing is applied in this experiment. The purpose is to isolate the
effect of feature-set reduction before adding portfolio-level persistence.

## Evaluation

For each model report:

- fixed Top-30 long-only Sharpe;
- fixed Top-30 maximum drawdown;
- fixed Top-30 monthly one-way turnover;
- mean monthly Top-30 return;
- fixed Top30/Bottom30 long-short Sharpe;
- Top-30 exposures to ROE, ProfitGrowth, EP, and BP.

Reference model:

- existing `Branch_F` output from the dual-branch experiment;
- Compact-F should reproduce it within numerical equality because it uses the
  same feature list and training configuration.

## Decision Rules

Compact-FT or Compact-FT3 is preferred over Compact-F only when:

1. Top-30 Sharpe improves;
2. maximum drawdown does not worsen by more than 5 percentage points;
3. turnover does not rise by more than 10 percentage points;
4. ROE and ProfitGrowth exposures remain positive;
5. the improvement is not solely caused by one isolated year.

If neither mixed model passes these rules, Compact-F remains the production
candidate.

Cascade residual modeling is explicitly out of scope for this experiment. It
is considered only if one or more retained technical factors show stable
incremental OOS value.

## Outputs

- `run_v15_compact_experiment.py`
- `output/production_models_v15_compact/Compact_F_oos.parquet`
- `output/production_models_v15_compact/Compact_FT_oos.parquet`
- `output/production_models_v15_compact/Compact_FT3_oos.parquet`
- `output/v15_compact_evaluation.csv`
- `output/v15_compact_evaluation.md`

## Tests and Safety

- Assert exact feature membership for all three models.
- Assert no excluded technical feature enters any model.
- Assert all models share fold and target configuration.
- Assert Compact-F feature list equals the approved fundamental branch.
- Use the fixed Top-30 evaluator from `run_v15_dual_branch.py`.
- Hard-fail on missing panel features or incomplete OOS prediction dates.
- Preserve all existing dual-branch and single-model artifacts.
