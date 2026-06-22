# Compact-F Portfolio Construction Results

- Signal: frozen Compact-F OOS predictions
- Rebalance: month-end
- Return: next-month forward return
- Turnover: one-way half-L1 weight turnover

## Comparison

| Strategy        |   Sharpe |    MaxDD |   Turnover |   MeanMonthlyReturn |   HitRate |    ROE |   ProfitGrowth |     EP | PassSharpe   | PassMaxDD   | PassTurnover   | PassStyle   | PassAll   |
|:----------------|---------:|---------:|-----------:|--------------------:|----------:|-------:|---------------:|-------:|:-------------|:------------|:---------------|:------------|:----------|
| A_Fixed_Top30   |   0.4117 | -31.7743 |    45.9524 |              0.6499 |   52.8571 | 0.6398 |         0.1058 | 0.4357 | True         | True        | False          | True        | False     |
| B_Top30_Buffer  |   0.443  | -34.2037 |    34.4015 |              0.7672 |   52.8571 | 0.687  |         0.1096 | 0.4189 | True         | False       | True           | True        | False     |
| C_Top30_Partial |   0.3483 | -35.314  |    27.0172 |              0.5633 |   52.8571 | 0.636  |         0.0974 | 0.4314 | False        | False       | True           | True        | False     |
| D_Fixed_Top40   |   0.3285 | -32.9326 |    42.8929 |              0.5229 |   51.4286 | 0.6239 |         0.1047 | 0.4133 | False        | True        | False          | True        | False     |
| E_Fixed_Top50   |   0.3423 | -31.9615 |    40.4857 |              0.5288 |   51.4286 | 0.6112 |         0.1143 | 0.4026 | False        | True        | False          | True        | False     |
| F_Top50_Buffer  |   0.4132 | -31.2931 |    28.0426 |              0.6452 |   51.4286 | 0.6484 |         0.1078 | 0.3849 | True         | True        | True           | True        | True      |

Percent-formatted columns: MaxDD, Turnover, MeanMonthlyReturn, HitRate.

## Selection

- Selected configuration: **F_Top50_Buffer**
- The selected alternative satisfied Sharpe, drawdown, turnover, and positive-style requirements.

## Buffer and Partial-Rebalance Audit

Detailed monthly sold counts, bought counts, holding counts, underfill flags, and pre/post normalization totals are stored in `output/compact_f_portfolio_construction_monthly.csv`.

## Annual Sharpe

See `output/compact_f_portfolio_construction_yearly.csv` for the full annual decomposition.
