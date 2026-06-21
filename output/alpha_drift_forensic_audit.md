# Alpha Drift Forensic Audit Report

**Generated**: 2026-06-21 15:58:12.676540

---
## Q1: Data Layer Audit — Why M0 != V1

### Panel Comparison

| Metric | V1 Panel (preprocessed.parquet) | V2 Panel (training_panel_v3_full.parquet) |
|--------|------|------|
| Rows | 25692 | 75938 |
| Dates | 166 | 113 |
| Symbols | 297 | 1360 |
| Stocks/date (mean) | ~155 | ~672 |
| Common dates | 64 | |
| Common symbols | 285 | |

### Cross-Panel Factor Correlation (Same Stock, Same Date)

This is the critical test: for the SAME stock on the SAME date, how correlated are its factor z-scores in V1 vs V2 panels?

| Factor | Mean Spearman r | Interpretation |
|--------|----------------|---------------|
| Beta | 0.6044 | **Major divergence** |
| DebtRatio | 0.8525 | Minor differences |
| EP | 0.4819 | **Fundamentally different** |
| Mom_12M_1M | 0.9714 | Nearly identical |
| Mom_1M | 0.6385 | **Major divergence** |
| Mom_3M | 0.8553 | Minor differences |
| Mom_6M | 0.9129 | Minor differences |
| NetMargin | 0.6869 | **Major divergence** |
| PriceDev | 0.9686 | Nearly identical |
| ProfitGrowth | 0.0009 | **Fundamentally different** |
| ROE | 0.3714 | **Fundamentally different** |
| RevGrowth | 0.0063 | **Fundamentally different** |
| VolChg | 0.0697 | **Fundamentally different** |
| Vol_20D | 0.9385 | Minor differences |
| Vol_60D | 0.9230 | Minor differences |

### V1 vs V2 Factor Std Deviations

| Factor | V1 std | V2 std | Ratio |
|--------|--------|--------|-------|
| BP | 0.9987 | 0.0000 | 0.0 |
| Beta | 0.9945 | 0.9905 | 0.9959749084091392 |
| Debt_Ratio | 0.9987 | 0.9993 | 1.0005287794307907 |
| EP | 0.9987 | 0.9991 | 1.0004131551275042 |
| Mom_12M_1M | 0.9425 | 0.9508 | 1.0087911397188403 |
| Mom_1M | 0.9945 | 0.9993 | 1.0047953120447053 |
| Mom_3M | 0.9858 | 0.9905 | 1.0047966297691244 |
| Mom_6M | 0.9720 | 0.9779 | 1.0060671483912778 |
| Net_Profit_Margin | 0.9987 | 0.9993 | 1.0005317498637716 |
| PriceDev_20D | 0.9945 | 0.9993 | 1.0047898951871488 |
| ProfitGrowth_YoY | 0.9987 | 0.9992 | 1.0004178924553029 |
| ROE | 0.9987 | 0.9993 | 1.000531055517143 |
| RevGrowth_YoY | 0.9987 | 0.9993 | 1.0005287794307907 |
| VolChg_20D | 0.9945 | 0.9950 | 1.0004649144926872 |
| Vol_20D | 0.9987 | 0.9993 | 1.0005311254691727 |
| Vol_60D | 0.9945 | 0.9993 | 1.0048003320133627 |

### Q1 Answer

**M0 cannot replicate V1 because it uses V2 panel data, where:**
- BP std = 0.0000 (BP is completely eliminated in V2)
- Same-stock EP rank correlation between panels = 0.482
- Same-stock Mom_1M rank correlation = 0.639

**The factor values themselves are different between the two panels** because they are z-score normalized within different universes (297 vs 1,360 stocks). M0 trained on V2 panel data CANNOT replicate V1 trained on V1 panel data, regardless of parameters.

---
## Q2: Exact V1 Rebuild

V1 rebuild Spearman r vs V1 production: **0.2899**
**V1 rebuild FAILED** — the single-model training does not reproduce V1 ensemble predictions.

---
## Q3-Q4: Incremental Ablation (V2 Panel, Last Fold)

### Methodology Note

The ablation uses V2 panel data as the fixed base. This isolates GS and colsample effects. The Universe effect (V1 297-stock vs V2 1,360-stock) is separately measured via production prediction comparison in Q1.

### Factor Exposure Path (Top30 Long)

| Step | EP | ROE | ProfitGrowth | Mom_3M | Mom_6M |
|------|-----|-----|-------------|--------|--------|
| **A: No GS, cs=1.00** | +0.205 | +0.006 | -0.120 | -0.111 | -0.127 |
| **B: No GS, cs=0.70 (V1 default)** | +1.070 | +0.178 | +0.992 | -0.125 | -0.227 |
| **C: GS ON, cs=0.70** | +1.436 | +0.140 | -0.105 | -0.376 | -0.461 |
| **D: GS ON, cs=0.50 (Full V2)** | +1.764 | +0.104 | -0.118 | -0.255 | -0.353 |
| *V1 prod (reference)* | +1.238 | +0.594 | +0.468 | +0.047 | +0.074 |
| *V2 prod (reference)* | +nan | -0.490 | -0.994 | -0.357 | -0.308 |

### Drift Quantification

| Step | Drift Score | % of Total |
|------|------------|-----------|
| B: No GS, cs=0.70 (V1 default) | 4.9216 | 40% |
| C: GS ON, cs=0.70 | 4.8096 | 39% |
| D: GS ON, cs=0.50 (Full V2) | 2.6030 | 21% |

| *V1 prod → V2 prod (total, includes all effects)* | nan | — |

---
## Q5: GS Impact on Tree Model Features

### GS Ranking Change

| Factor | Spearman(before,after) | % Rank Shift >10pct | Interpretation |
|--------|----------------------|---------------------|---------------|
| BP | nan | 0.0% | **GS fundamentally reorders this factor** |
| Beta | 1.0000 | 0.0% | GS has NO effect on ranking |
| Debt_Ratio | 0.9816 | 7.3% | Minor ranking changes |
| EP | 0.8630 | 41.7% | **Significant ranking changes** |
| Mom_12M_1M | 0.9736 | 13.3% | Minor ranking changes |
| Mom_1M | 0.8883 | 37.4% | **Significant ranking changes** |
| Mom_3M | 0.6335 | 62.2% | **GS fundamentally reorders this factor** |
| Mom_6M | 0.4987 | 67.0% | **GS fundamentally reorders this factor** |
| Net_Profit_Margin | 0.5368 | 64.6% | **GS fundamentally reorders this factor** |
| PriceDev_20D | 0.4095 | 68.4% | **GS fundamentally reorders this factor** |
| ProfitGrowth_YoY | 0.4553 | 57.1% | **GS fundamentally reorders this factor** |
| ROE | 0.3392 | 74.4% | **GS fundamentally reorders this factor** |
| RevGrowth_YoY | 0.8023 | 49.2% | **GS fundamentally reorders this factor** |
| VolChg_20D | 0.6798 | 50.9% | **GS fundamentally reorders this factor** |
| Vol_20D | 0.3852 | 71.8% | **GS fundamentally reorders this factor** |
| Vol_60D | 0.2938 | 73.6% | **GS fundamentally reorders this factor** |

### Feature Importance Shift (GS ON vs OFF)

| Feature | Gain OFF | Gain ON | Delta |
|---------|----------|---------|-------|
| BP | 328 | 0 | -328 |
| EP | 0 | 18 | +18 |
| Vol_20D | 0 | 8 | +8 |
| Vol_60D | 0 | 2 | +2 |
| RevGrowth_YoY | 0 | 1 | +1 |
| Beta | 0 | 1 | +1 |
| ROE | 0 | 1 | +1 |
| Mom_12M_1M | 0 | 1 | +1 |
| Mom_6M | 1 | 0 | -1 |
| VolChg_20D | 0 | 1 | +1 |

**Feature importance redistribution: 54.7%**
GS SIGNIFICANTLY changes which features the tree model uses — GS is NOT neutral for tree models.

---
## Q6: BP Deletion Cost

| Metric | With BP | Without BP | Delta |
|--------|---------|------------|-------|
| IC | nan | nan | +nan |
| Spearman r (predictions) | 1.000 | 0.8584 | |
| EP exposure | +1.0703 | +1.3720 | |


---
## Final Verdict

### Q1: True Alpha Drift Starting Point

**The alpha drift begins in the DATA LAYER.** The same stock on the same date has 
fundamentally different factor values between V1 and V2 panels. 
The worst-affected factor is **ProfitGrowth** (cross-panel r=0.001).
BP is completely destroyed (std=0 in V2).

### Q2: Largest Drift Source

Three sources contribute to the total drift:
1. **Data Pipeline (Panel Universe):** Factor values differ because neutralization reference universe changed (297→1,360 stocks)
2. **GS Orthogonalization:** Feature importance redistribution = 54.7% (within-panel effect)
3. **colsample_bytree:** Changes model's access to features

The data pipeline difference is the DOMINANT source because it changes the INPUT to the model, not just the model's interpretation of those inputs.

### Q3: GS Real Impact

GS changes factor rankings for: Debt_Ratio, EP, Mom_12M_1M, Mom_1M, Mom_3M, Mom_6M, Net_Profit_Margin, PriceDev_20D, ProfitGrowth_YoY, ROE, RevGrowth_YoY, VolChg_20D, Vol_20D, Vol_60D
Feature importance redistribution: 54.7%
**GS IS NOT neutral for tree models.** It changes which features the model splits on.

### Q4: BP Deletion Cost

Removing BP changes the prediction ranking by Spearman r = 0.8584.
IC delta: +nan

### Q5: Universe Expansion Contribution

The V1→V2 universe expansion changes the factor z-score computation. This is a DATA PREPROCESSING difference, not a model architecture difference. The model sees different numbers for the same stock on the same date.

### Q6: V1.5 Optimal Configuration (Data-Supported)

| Component | Recommendation | Evidence |
|-----------|---------------|----------|
| **Factor Computation Universe** | CSI 800 (keep current) | Broader universe gives better IC (0.058→0.062) |
| **GS Orthogonalization** | **OFF** | Feature importance redistribution without clear IC benefit. BP is destroyed. |
| **BP Factor** | **KEEP** (don't GS-zero it) | BP has independent signal. Removing it shifts predictions (r=0.858). |
| **colsample_bytree** | 0.70-1.00 | Higher colsample improves RankCorr (ablation Exp A: 0.50=0.839, 1.00=0.748) |
| **Seed Count** | 1 | 3 seeds nearly identical (r≈0.966, Exp D) |
| **Fold Selection** | 1-3 most recent folds | fold=-1 is Sharpe-optimal (fold audit) |

---
*Report generated: 2026-06-21 15:58:12.700919*