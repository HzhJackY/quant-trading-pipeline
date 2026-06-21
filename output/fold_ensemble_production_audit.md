# Fold Ensemble Production Audit Report

**Generated**: 2026-06-21 15:00:31.171381

---
## Part 1: Production Inference Chain Audit

### Q1: What models does production actually load?

```text
seed = [42, 888, 2026]
fold = [70]    <-- fold_idx=-1 resolves to the LAST fold

Total models per inference: 3 seeds x 1 fold = 3 models
```

### Q2: What does fold_idx=-1 mean?

**Answer: A - last fold only**

Code trace:
```python
# paper_trading_pipeline.py:378
signals = engine.predict_cross_section(features=X, prev_signal=prev_array, fold_idx=-1)

# production_engine.py:661-662
if fold_idx < 0:
    fold_idx = n_available + fold_idx   # -1 -> 70 (last fold)

# production_engine.py:670-671
for seed in self.seeds:
    model = self._models[seed][70]  # self._models[seed][70]
    raw_preds.append(model.predict(X))

# production_engine.py:681
ensemble_raw = np.mean(raw_preds, axis=0)  # Equal-weight seed average

# production_engine.py:684
ranked = cs_rank(ensemble_raw)  # Cross-sectional rank -> [0, 1]
```

### Full Call Chain

| Step | Location | Action |
|------|----------|--------|
| 1 | `paper_trading_pipeline.py:547` | `ProductionAlphaEngine.load_models('output/production_models_v2_full')` |
| 2 | `production_engine.py:975-991` | Backtest mode: loads ALL 213 models (71 folds x 3 seeds) |
| 3 | `paper_trading_pipeline.py:378` | `engine.predict_cross_section(X, prev_signal, fold_idx=-1)` |
| 4 | `production_engine.py:662` | fold_idx=-1 -> 70 (last fold) |
| 5 | `production_engine.py:670-675` | For each seed: model[70].predict(X) |
| 6 | `production_engine.py:681` | ensemble_raw = mean(seed_preds) |
| 7 | `production_engine.py:684` | ranked = cs_rank(ensemble_raw) |

### Audit Verdict

| Audit Item | Finding |
|------------|---------|
| Models loaded | 213 (3 seeds x 71 folds) |
| Models USED per inference | 3 (3 seeds x 1 fold) |
| Utilization rate | 1.4% (3/213) |
| Which fold | Last fold (index 70) |
| Inference methodology | Fold 70 used for ALL dates (including 2017) - global fold selection |
| prev_signal usage | Validated but NOT used in model.predict() - turnover penalty only affects training |

**Critical finding**: Production loads 213 models but only uses **3**. 
The last fold (index 70) is applied to ALL dates. This single-fold strategy 
provides zero diversification across training windows and is maximally exposed to 
regime-specific model biases.

---
## Part 3: Fold Ensemble Ablation Results

### Scheme Definitions

| Scheme | Description | Folds | Models (3 seeds) |
|--------|-------------|-------|------------------|
| **A: Production (fold=-1, fold 70 only)** | A: Production (fold=-1, fold 70 only) | [70]... (1 folds) | 3 |
| **B: Last 3 folds (68-70)** | B: Last 3 folds (68-70) | [68, 69, 70]... (3 folds) | 9 |
| **C: Last 10 folds (61-70)** | C: Last 10 folds (61-70) | [61, 62, 63]... (10 folds) | 30 |
| **D: All 71 folds** | D: All 71 folds | [0, 1, 2]... (71 folds) | 213 |
| **E: IC Top 25% (17 folds)** | E: IC Top 25% (17 folds) | [45, 46, 49]... (17 folds) | 51 |
| **F: IC Top 10% (7 folds)** | F: IC Top 10% (7 folds) | [46, 52, 61]... (7 folds) | 21 |

### Unified Metrics Table

| Ensemble | Models | RankCorr | Top30Overlap | IC | IC_IR | Sharpe | MaxDD | Turnover |
|----------|--------|----------|-------------|-----|-------|--------|-------|----------|
| **A: Production (fold=-1, fold 70 only)** | 3 | 0.7177 | 0.503 | 0.0615 | 0.5377 | 0.51 | -0.334 | 0.383 |
| **B: Last 3 folds (68-70)** | 9 | 0.7305 | 0.519 | 0.0643 | 0.5133 | 0.51 | -0.324 | 0.372 |
| **C: Last 10 folds (61-70)** | 30 | 0.7465 | 0.582 | 0.0659 | 0.4655 | 0.47 | -0.326 | 0.354 |
| **D: All 71 folds** | 213 | 0.6775 | 0.606 | 0.0574 | 0.4360 | 0.42 | -0.339 | 0.409 |
| **E: IC Top 25% (17 folds)** | 51 | 0.7754 | 0.607 | 0.0676 | 0.4518 | 0.48 | -0.297 | 0.327 |
| **F: IC Top 10% (7 folds)** | 21 | 0.7859 | 0.602 | 0.0689 | 0.4593 | 0.48 | -0.313 | 0.319 |

### Key Comparisons

| Metric | Production (A) | Best Scheme | Delta | Winner |
|--------|---------------|-------------|-------|--------|
| RankCorr | 0.7177 | 0.7859 | +0.0682 | F: IC Top 10% (7 folds) |
| Sharpe | 0.51 | 0.51 | +0.00 | A: Production (fold=-1, fold 70 only) |
| Turnover | 0.383 | 0.319 | -0.064 | F: IC Top 10% (7 folds) |

---
## Part 5: Root Cause Attribution

### Q1: Is RankCorr=0.718 a fold selection problem or a model problem?

**Fold SELECTION matters, but not decisively.**
Production RankCorr=0.7177, Best RankCorr=0.7859 (Delta=+0.0682).

### Q2: Does using all folds significantly improve metrics?

| Metric | Production (A) | All Folds (D) | Delta | Significant? |
|--------|---------------|--------------|-------|-------------|
| RankCorr | 0.7177 | 0.6775 | -0.0402 | YES |
| Sharpe | 0.5078 | 0.4199 | -0.0879 | YES |
| Turnover | 0.3835 | 0.4085 | +0.0250 | YES |

### Q3: What is the optimal ensemble configuration?

| Rank | Scheme | RankCorr | Sharpe | Turnover | Reason |
|------|--------|----------|--------|----------|--------|
| 1 | **A: Production (fold=-1, fold 70 only)** | 0.7177 | 0.51 | 0.383 | Current production - single fold, regime-concentrated |
| 2 | **B: Last 3 folds (68-70)** | 0.7305 | 0.51 | 0.372 | Recent folds, moderate diversification |
| 3 | **E: IC Top 25% (17 folds)** | 0.7754 | 0.48 | 0.327 | IC-based selection removes low-quality folds |
| 4 | **F: IC Top 10% (7 folds)** | 0.7859 | 0.48 | 0.319 | IC-based selection, highest average IC folds |
| 5 | **C: Last 10 folds (61-70)** | 0.7465 | 0.47 | 0.354 | Broader recent window |
| 6 | **D: All 71 folds** | 0.6775 | 0.42 | 0.409 | Maximum fold diversification |

### Recommended Configuration

**Best scheme**: A: Production (fold=-1, fold 70 only)
- Models: 3
- RankCorr: 0.7177
- Sharpe: 0.51
- Turnover: 0.383

### Seed Recommendation

Based on Experiment D (seed-pair r ~ 0.966), 3 seeds are redundant.
**Recommendation**: Use 1 seed (any of [42, 888, 2026]). Saves 67% inference cost with negligible signal loss.

---
## Final Verdict

### NO - Fold Ensemble Architecture is NOT the primary cause of Sharpe degradation.
All schemes produce similar Sharpe (range: 0.51 to 0.51).
Delta = 0.000 - negligible. The root cause lies elsewhere.

**The Sharpe degradation from 0.70 (V1) to 0.51 (V2_Full) is NOT explained by fold selection.**
The model itself (features, training methodology, architecture) is the primary driver.

---
*Report generated: 2026-06-21 15:00:31.171559*