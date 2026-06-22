# Compact-F Cost Sensitivity

| Strategy       |   Cost_bps |   GrossSharpe |   NetSharpe |   NetMaxDD |   AvgTurnover |   NetMeanMonthlyReturn |   HitRate |
|:---------------|-----------:|--------------:|------------:|-----------:|--------------:|-----------------------:|----------:|
| Top30 Baseline |          0 |        0.4117 |      0.4117 |   -31.7743 |       45.9524 |                 0.6499 |   52.8571 |
| Top30 Baseline |         10 |        0.4117 |      0.383  |   -32.5681 |       45.9524 |                 0.6044 |   52.8571 |
| Top30 Baseline |         20 |        0.4117 |      0.3543 |   -33.353  |       45.9524 |                 0.5589 |   51.4286 |
| Top30 Baseline |         30 |        0.4117 |      0.3256 |   -34.1291 |       45.9524 |                 0.5133 |   50      |
| Top30 Baseline |         50 |        0.4117 |      0.2681 |   -35.6553 |       45.9524 |                 0.4223 |   50      |
| Top50 Buffer   |          0 |        0.4132 |      0.4132 |   -31.2931 |       28.0426 |                 0.6452 |   51.4286 |
| Top50 Buffer   |         10 |        0.4132 |      0.3956 |   -31.4847 |       28.0426 |                 0.6175 |   51.4286 |
| Top50 Buffer   |         20 |        0.4132 |      0.378  |   -31.6758 |       28.0426 |                 0.5897 |   51.4286 |
| Top50 Buffer   |         30 |        0.4132 |      0.3604 |   -31.8946 |       28.0426 |                 0.562  |   51.4286 |
| Top50 Buffer   |         50 |        0.4132 |      0.3251 |   -32.9545 |       28.0426 |                 0.5065 |   51.4286 |

Percent-formatted: NetMaxDD, AvgTurnover, NetMeanMonthlyReturn, HitRate.

## Relative Result

- 10 bps: Top50 Buffer NetSharpe 0.3956 vs Top30 0.3830 — better.
- 20 bps: Top50 Buffer NetSharpe 0.3780 vs Top30 0.3543 — better.
- 30 bps: Top50 Buffer NetSharpe 0.3604 vs Top30 0.3256 — better.
- 50 bps: Top50 Buffer NetSharpe 0.3251 vs Top30 0.2681 — better.
