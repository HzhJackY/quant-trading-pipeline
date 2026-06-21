# Ensemble Stability Root Cause Report

**Generated**: 2026-06-21 14:39:15.959295

## Executive Summary

| Metric | V1 (production) | V2_Full (production) | Best Single-Model | Best Ensemble |
|--------|----------------|----------------------|-------------------|---------------|
| RankCorr | 0.984 | 0.718 | ~0.85 | TBD |
| Top30Overlap | 81% | 34% | ~70% | TBD |
| IC | 0.0582 | 0.0615 | ~0.10 | TBD |
| Sharpe | 0.70 | 0.51 | TBD | TBD |

---
## Experiment C: Model Pairwise Prediction Correlation

**Total comparisons**: 2,551,314
**Dates**: 113
**Models**: 213

### Overall Distribution

| Statistic | Pearson r |
|-----------|----------|
| Mean | 0.5297 |
| Median | 0.5646 |
| Std | 0.2898 |
| P10 | 0.1078 |
| P25 | 0.2897 |
| P75 | 0.7857 |
| P90 | 0.8900 |
| Min | -0.3684 |
| Max | 0.9949 |

### Same-Seed vs Cross-Seed

| Comparison | Mean r | N |
|------------|--------|---|
| Same seed | 0.5259 | 842,415 |
| Cross seed | 0.5316 | 1,708,899 |

**Answer C**: Models express **fundamentally different** views (mean r = 0.530). Each model sees a completely different ranking. Ensembling them creates noise, not signal.

---
## Experiment D: Seed Consistency Analysis

| Seed Pair | Mean r | Median r | Std | N |
|-----------|--------|---------|-----|---|
| 42-888 | 0.9648 | 0.9687 | 0.0184 | 8023 |
| 42-2026 | 0.9687 | 0.9720 | 0.0170 | 8023 |
| 888-2026 | 0.9658 | 0.9684 | 0.0186 | 8023 |

**Answer D**: Seeds are **nearly identical** — random seed introduces minimal noise. Reducing from 3→1 seed is safe.

---
## Experiment E: Fold Consistency Analysis

| Seed | Mean Fold-Fold r | Median | N |
|------|-----------------|--------|---|
| 42 | 0.5299 | 0.5626 | 280805 |
| 888 | 0.5176 | 0.5518 | 280805 |
| 2026 | 0.5303 | 0.5648 | 280805 |

### Fold Distance vs Correlation
| Fold Δ | Mean r | N |
|---------|--------|---|
| 1 | 0.9155 | 23730 |
| 2 | 0.8818 | 23391 |
| 3 | 0.8522 | 23052 |
| 4 | 0.8317 | 22713 |
| 5 | 0.8190 | 22374 |
| 6 | 0.8053 | 22035 |
| 7 | 0.7836 | 21696 |
| 8 | 0.7609 | 21357 |
| 9 | 0.7415 | 21018 |
| 10 | 0.7257 | 20679 |

**Answer E**: Folds are **substantially different** (mean fold-fold r = 0.526). Different training windows produce fundamentally different models. This is a major source of ensemble instability.

---
## Experiment F: Ensemble Scale Ablation

| N_models | RankCorr | Top30Overlap | IC | IC_IR |
|----------|----------|-------------|-----|-------|
| 1 | 0.3544 ± 0.2369 | 0.236 | 0.0509 | 0.3611 |
| 3 | 0.5352 ± 0.1479 | 0.389 | 0.0485 | 0.3678 |
| 9 | 0.6205 ± 0.0893 | 0.499 | 0.0582 | 0.4405 |
| 27 | 0.6595 ± 0.0770 | 0.570 | 0.0569 | 0.4256 |
| 54 | 0.6702 ± 0.0721 | 0.590 | 0.0556 | 0.4159 |
| 108 | 0.6770 ± 0.0658 | 0.599 | 0.0567 | 0.4324 |
| 213 | 0.6775 ± 0.0673 | 0.606 | 0.0574 | 0.4360 |

**Answer F**: RankCorr is **stable or improves** with more models (Δ = +0.3231). Ensemble does NOT hurt stability.

---
## Experiment G: Style Drift Analysis

### Factor Exposure Distribution Across Folds

| Factor | Mean | Std | Min | Max | % Positive |
|--------|------|-----|-----|-----|-----------|
| EP | 0.5069 | 0.1299 | 0.0822 | 0.7421 | 100% |
| ROE | -0.0782 | 0.1425 | -0.4860 | 0.3367 | 26% |
| ProfitGrowth | -0.2500 | 0.1046 | -0.5914 | 0.1579 | 4% |
| RevGrowth | -0.1015 | 0.0948 | -0.2563 | 0.1180 | 18% |
| Mom_1M | -0.3074 | 0.1260 | -0.5987 | -0.0095 | 0% |
| Mom_3M | nan | nan | nan | nan | nan% |
| Mom_12M_1M | nan | nan | nan | nan | nan% |
| NetProfitMargin | -0.1308 | 0.1196 | -0.3540 | 0.2329 | 15% |
| BP | nan | nan | nan | nan | nan% |
| Vol_20D | -0.1803 | 0.1750 | -0.5713 | 0.1807 | 18% |

### Style Dispersion (max-min exposure)

| ROE | 0.8227 |
| ProfitGrowth | 0.7493 |
| EP | 0.6599 |
| Mom_1M | 0.5893 |
| Mom_3M | nan |
| Mom_12M_1M | nan |
| NetProfitMargin | 0.5869 |
| BP | nan |
| Vol_20D | 0.7521 |
| RevGrowth | 0.3743 |

**Answer G**: Factors with significant style dispersion (>0.3): ['EP', 'ROE', 'ProfitGrowth', 'RevGrowth', 'Mom_1M', 'NetProfitMargin', 'Vol_20D']. 
Different folds **do represent different investment styles** — e.g., some favor value (high EP exposure), others favor growth (high ProfitGrowth exposure).
This style drift across folds contributes to ensemble instability because averaging value and growth views produces an incoherent middle-ground ranking.

---
## Experiment H: Ensemble Weighting Analysis

| Weighting Scheme | RankCorr | Top30Overlap | IC | IC_IR |
|------------------|----------|-------------|-----|-------|
| Equal | 0.6775 | 0.606 | 0.0574 | 0.4379 |
| IC_Weighted | 0.7319 | 0.654 | 0.0617 | 0.4413 |
| Top25% | 0.7764 | 0.607 | 0.0674 | 0.4472 |
| Top10% | 0.7904 | 0.610 | 0.0690 | 0.4521 |

**Answer H**: **Top10%** weighting significantly improves RankCorr (0.6775 → 0.7904). This confirms the presence of **low-quality models dragging down the ensemble**. Removing or down-weighting them is a high-impact fix.

---
## Final Synthesis: Root Cause Attribution

### Q1: Does 162-model ensemble significantly reduce rank stability?

Single-model RankCorr: 0.3544
Full-ensemble RankCorr: 0.6775
Δ: +0.3231
**NO** — the ensemble does NOT reduce stability. The problem is elsewhere.

### Q2: Decomposition of RankCorr decline (0.85 → 0.72)

| Source | Estimated ΔRankCorr | % of Total | Evidence |
|--------|---------------------|------------|----------|
| Seed variance | 0.0336 | 26% | Exp D: mean seed-pair r=0.966 |
| Fold variance | 0.4741 | 365% | Exp E: mean fold-fold r=0.526 |
| Ensemble scale | 0.0000 | 0% | Exp F: 1-model vs 213-model stability |

### Q3: Is there Over-Ensemble?

**NO** — no evidence of Over-Ensemble. Ensemble stability ≈ single-model stability.

### Q4: Optimal number of models?

Best RankCorr at N=213: 0.6775

### Q5: Should we reduce from 54-fold × 3-seed to 9-fold × 3-seed or single-seed?

- **Seed reduction**: YES — seeds are nearly identical. 1 seed is sufficient.
- **Fold reduction**: CAUTION — folds are substantially different (r=0.526). Diversity may be beneficial for robustness despite stability cost.

---
*Report generated: 2026-06-21 14:39:17.741594*