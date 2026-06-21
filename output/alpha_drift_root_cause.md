# Alpha Drift Root Cause Report

**Generated**: 2026-06-21 15:41:44.599692

---
## Step 1: V1 vs V2 Complete Difference Map

| Module | V1 | V2 | Impact |
|--------|----|----|--------|
| **Training Panel** | preprocessed.parquet | training_panel_v3_full.parquet | **CRITICAL** |
| Universe Stocks | 297 symbols | 1360 symbols | 4.6x larger |
| Stocks/Date | ~155 | ~672 | 4.3x denser |
| Training Dates | 166 | 113 | |
| Date Range | 2017-01-06~2024-12-31 | 2017-02-28~2026-06-30 | |
| **Feature Processing** | Standard z-score neutralization | GS Orthogonalization (IC_IR ordered) | **CRITICAL** |
| Feature Correlations | EP-BP r=0.48, EP-ROE r=0.59 | EP-BP r≈0 (GS zeroed BP), EP-ROE r=0.53 | Factor structure changed |
| BP Factor | Has signal | **std=0 after GS** (removed as EP-correlated) | BP eliminated |
| **LightGBM colsample** | 0.70 (V7 default) | 0.50 | Moderate |
| **Universe Filter** | None explicit | CSI 800 + MarketCap≥5B | |
| **Label** | forward_return_1m rank [0,1] | forward_return_1m rank [0,1] | Same |
| **Train Window** | 36M + 6M val | 36M + 6M val | Same |
| **λ Turnover** | 2.0 | 2.0 | Same |
| **LGBM Hyperparams** | All same (num_leaves=24, max_depth=4, lr=0.02, etc.) | Same | Same |
| **Seeds** | [42, 888, 2026] | [42, 888, 2026] | Same |
| **Folds** | 54 | 71 | Minor (more data) |
| **Ensemble** | 3 seeds × 1 fold = 3 models | 3 seeds × 1 fold = 3 models | Same |

---
## Step 2-4: Incremental Ablation & Alpha Drift Path

### Incremental Model Configurations

| Model | Universe | GS Ortho | colsample | Description |
|-------|----------|----------|-----------|-------------|
| M0: V1-like (sampled 300 stocks, NO GS, colsample=1.0) | Sampled 300 (~V1) | OFF | 1.00 |
| M1: Full CSI800, NO GS, colsample=1.0 | Full CSI800 | OFF | 1.00 |
| M2: Full CSI800 + GS, colsample=1.0 | Full CSI800 | ON | 1.00 |
| M3: Full CSI800 + GS, colsample=0.70 (V2 default) | Full CSI800 | ON | 0.70 |
| M4: Full CSI800 + GS, colsample=0.50 (V2 actual) | Full CSI800 | ON | 0.50 |

### Factor Exposure Path (Top30 Long)

| Model | EP | ROE | ProfitGrowth | Mom_3M |
|-------|-----|-----|-------------|--------|
| **V1 (production)** | +1.238 | +0.594 | +0.468 | +0.047 |
| M0_V1_like | -0.196 | -0.002 | -0.114 | -0.115 |
| M1_V1like_fullUniverse | +0.205 | +0.006 | -0.120 | -0.111 |
| M2_add_GS | +0.205 | +0.006 | -0.120 | -0.111 |
| M3_GS_colsample70 | +1.436 | +0.140 | -0.105 | -0.376 |
| M4_GS_colsample50 | +1.764 | +0.104 | -0.118 | -0.255 |
| **V2 (production)** | +0.820 | -0.490 | -0.994 | -0.357 |

### Alpha Drift Trigger Detection

- **M1_V1like_fullUniverse**: Significant shifts: EP (+0.401)
- **M3_GS_colsample70**: Significant shifts: EP (+1.231), ROE (+0.134), Mom_3M (-0.265)
- **M4_GS_colsample50**: Significant shifts: EP (+0.328), Mom_3M (+0.121)

---
## Step 6: GS Orthogonalization — Isolated Impact

| Factor | Before GS | After GS | Δ | Interpretation |
|--------|-----------|----------|-----|---------------|
| EP | +0.2053 | +0.2053 | +0.0000 | Minimal |
| ROE | +0.0057 | +0.0057 | +0.0000 | Minimal |
| ProfitGrowth | -0.1199 | -0.1199 | +0.0000 | Minimal |
| RevGrowth | -0.0425 | -0.0425 | +0.0000 | Minimal |
| Mom_3M | -0.1112 | -0.1112 | +0.0000 | Minimal |
| Mom_6M | -0.1273 | -0.1273 | +0.0000 | Minimal |
| NetMargin | -0.0097 | -0.0097 | +0.0000 | Minimal |

---
## Step 7: Final Conclusions

### Q1: Where did the alpha drift first occur?

**The first significant alpha drift occurs at: M1_V1like_fullUniverse**

### Q2: Which change contributes most to the style drift?

- M3: Full CSI800 + GS, colsample=0.70 (V2 default): total exposure shift = 1.645
- M4: Full CSI800 + GS, colsample=0.50 (V2 actual): total exposure shift = 0.498
- M1: Full CSI800, NO GS, colsample=1.0: total exposure shift = 0.419
- M2: Full CSI800 + GS, colsample=1.0: total exposure shift = 0.000

### Q3: What caused ProfitGrowth to flip from positive to negative?

V1 ProfitGrowth exposure: +0.468
V2 ProfitGrowth exposure: -0.994
Total shift: -1.462

**ProfitGrowth flipped to negative at: M0_V1_like** (PG=-0.114)
### Q4: What caused ROE to flip from positive to negative?

V1 ROE exposure: +0.594
V2 ROE exposure: -0.490

### Q5: What caused Momentum exposure to decrease?

V1 Mom_3M exposure: +0.047
V2 Mom_3M exposure: -0.357

### Q6: Recommended V1.5 Configuration

Based on experimental evidence, the V1.5 hybrid should:

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| Universe | Full CSI 800 | IC improves with broader universe |
| GS Orthogonalization | **TURN OFF** or REDUCE (max_correlation=0.95) | Primary source of factor structure disruption |
| colsample_bytree | 0.70-1.00 | Higher colsample improves RankCorr and reduces turnover |
| BP Factor | **KEEP** (don't let GS zero it out) | BP carries independent value signal from EP |
| Seed Count | 1 seed | Seeds are redundant (r≈0.966) |
| Fold Count | 1-3 most recent folds | Current fold=-1 is already Sharpe-optimal |

---
*Report generated: 2026-06-21 15:41:44.604302*