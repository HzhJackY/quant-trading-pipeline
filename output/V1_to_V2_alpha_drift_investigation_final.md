# V1 → V2 Alpha Drift Investigation — Project Retrospective

**Date**: 2026-06-21
**Status**: Phase complete. Ready for V1.5 design phase.
**Audience**: New researcher joining the project. No prior context assumed.

---

# Executive Summary

Two versions of a quantitative stock selection model were compared:

| Metric | V1 (original) | V2_Full (new) | Delta |
|--------|--------------|---------------|-------|
| Sharpe Ratio | **0.70** | 0.51 | −0.19 |
| Max Drawdown | −29.6% | −33.4% | −3.8pp |
| Turnover (monthly) | **14.5%** | 38.3% | +23.8pp |
| Mean IC | 0.058 | **0.062** | +0.004 |
| IC_IR | 0.539 | 0.538 | stable |
| V1/V2 rank correlation | — | **0.39** | fundamentally different |

**The puzzle**: V2 has better IC but worse Sharpe, higher turnover, and a fundamentally different stock ranking (Spearman r=0.39 with V1).

After a comprehensive, multi-stage investigation spanning ~10 major experiments, the evidence converges on a single root cause:

> **V1 and V2 are not two versions of the same alpha. They are two different alphas, trained on factor data whose economic meaning shifted when the stock universe expanded from 297 to 1,360 stocks.**

The shift has three layers:
1. **Factor Meaning Drift (primary)**: Same factor names (EP, ROE, ProfitGrowth) encode different economic information in different universes. ProfitGrowth rank correlation between universes = 0.001 (essentially random).
2. **GS Orthogonalization (secondary)**: Redistributes 55% of feature importance and fundamentally reorders 12/16 factors. Turns "GS-ROE" into a different feature than "economic ROE."
3. **colsample interaction (tertiary)**: Lower colsample (0.70→0.50) amplifies the effects of (1) and (2) by restricting feature access per tree.

**V2 is not a degraded V1. V2 is a style-shifted alpha that loads anti-growth/anti-momentum in a broader universe.** The Sharpe degradation comes from regime exposure (severe underperformance in 2019-2020 growth bull markets), not from model architecture flaws.

---

# Part 1. Original Problem

## 1.1 Baseline Performance

V1 was trained on `preprocessed.parquet` (297 stocks, ~155 stocks/date, 2017-2024). V2 was a complete rebuild on `training_panel_v3_full.parquet` (1,360 CSI 800 stocks, ~672 stocks/date, 2017-2026) with GS orthogonalization and reduced colsample_bytree.

The full comparison is in `output/model_comparison_report.png` and `output/v1_v2_monthly_attribution.md`.

## 1.2 Why We Investigated

The initial reaction to Sharpe 0.70→0.51 was: "something broke in the new training pipeline." The investigation was designed to identify which specific change caused the degradation, with the goal of fixing it and recovering V1-level performance.

---

# Part 2. Investigation Timeline

## Stage A: Parameter Suspicion (Early Phase)

### A1. ProfitGrowth Reversal Hypothesis
**Question**: Did the model learn to use ProfitGrowth in the wrong direction?

**Experiment**: SHAP analysis on V2 predictions (`output/research_report_shap_diagnosis.md`).

**Finding**: ProfitGrowth SHAP is weakly negative (−0.0004 mean) but the effect is conditional on EP level. In high-EP (value) stocks, ProfitGrowth SHAP turns positive. This is not a simple sign flip — it's a conditional effect.

**Verdict**: ❌ **ProfitGrowth reversal is NOT the primary cause.** Conditional usage is defensible.

### A2. colsample_bytree Hypothesis
**Question**: Is colsample=0.50 (V2) causing feature randomization and unstable rankings?

**Experiment**: Controlled single-model ablation testing colsample ∈ {0.50, 0.70, 0.90, 1.00} (`output/root_cause_ablation.md`).

**Finding**: colsample=0.50 has the BEST RankCorr (0.839). colsample=0.90 has the worst (0.686). Pattern is U-shaped.

**Verdict**: ❌ **colsample=0.50 is NOT the cause of instability.** It actually improves single-model stability.

### A3. GS Orthogonalization Hypothesis
**Question**: Is GS making the model less stable?

**Experiment**: GS ON vs OFF comparison in single-model ablation.

**Finding**: GS OFF has slightly higher RankCorr (0.866 vs 0.839) but MUCH higher turnover (37.5% vs 28.6%). GS actually reduces turnover.

**Verdict**: ❌ **GS is NOT the primary cause of instability** (at the single-model level).

---

## Stage B: Stability Investigation

### B1. Rank Stability & Turnover Root Cause
**Question**: Why did V2 turnover explode from 14.5% to 38.3%?

**Experiment**: Month-over-month rank stability analysis (`output/turnover_root_cause_analysis.md`).

**Key findings**:
- V1 RankCorr (t, t+1) = **0.984** — nearly perfectly stable rankings
- V2 RankCorr (t, t+1) = **0.718** — substantially less stable
- V2 Top30 overlap = **34%** (vs V1 81%) — 66% of positions replaced monthly
- V2 signal Δ = **6.6× larger** than V1

**Verdict**: Rank stability collapse is the MECHANISM of turnover explosion. But what CAUSES the stability collapse?

### B2. Seed Consistency
**Question**: Do random seeds introduce instability?

**Experiment**: Pairwise seed correlation within same fold (`output/ensemble_stability_root_cause.md`).

**Finding**: Seed pairs all correlate at r ≈ 0.965−0.969. Three seeds produce nearly identical predictions.

**Verdict**: ❌ **Seeds are not the cause.** 3→1 seed reduction has negligible signal loss.

### B3. Fold & Ensemble Architecture
**Question**: Does the 213-model walk-forward ensemble cause instability?

**Experiments**:
- Fold consistency: fold-fold prediction r = **0.53** (folds disagree dramatically)
- Ensemble scale ablation: more models → HIGHER RankCorr (0.35→0.68)
- Fold selection audit: 6 different fold schemes compared

**Key finding**: Production (fold=-1, 3 models) has the BEST Sharpe (0.51) among all fold schemes. Using all 213 models produces Sharpe 0.42.

**Verdict**: ❌ **Fold/Ensemble architecture is NOT the cause of Sharpe degradation.** Production fold=-1 is already optimal.

---

## Stage C: Style Drift Investigation

### C1. Factor Exposure Comparison
**Question**: How do V1 and V2 portfolios differ in factor exposures?

**Finding** (`output/v1_v2_monthly_attribution.md`):

| Factor Exposure (Top30 Long) | V1 | V2 | Shift |
|------------------------------|-----|-----|-------|
| EP | +1.24 | +0.82 | −0.42 |
| **ROE** | **+0.59** | **−0.49** | **−1.08 (flipped)** |
| **ProfitGrowth** | **+0.47** | **−0.99** | **−1.46 (flipped)** |
| RevGrowth | +1.45 | +0.21 | −1.24 |
| Mom_3M | +0.05 | **−0.36** | −0.41 |
| Mom_6M | +0.07 | **−0.31** | −0.38 |

V1 is a **deep value + quality** portfolio. V2 is a **moderate value + anti-growth + anti-momentum** portfolio.

### C2. Regime Performance
**Question**: When does V2 underperform V1?

| Market Regime | V1 L/S | V2 L/S | Delta |
|--------------|--------|--------|-------|
| Down (<−3%) | +2.43% | +2.13% | −0.30% |
| Flat | +0.95% | +0.54% | −0.40% |
| **Up (>+3%)** | **+0.30%** | **−0.28%** | **−0.58%** |

**V2 loses money in up markets.** This is consistent with anti-growth/anti-momentum positioning.

### C3. Yearly Sharpe Pattern
| Year | V1 SR | V2 SR | Regime |
|------|-------|-------|--------|
| 2019 | +0.47 | **−1.32** | Growth bull |
| 2020 | +1.56 | **−0.70** | Growth bull |
| 2023 | +2.72 | +3.20 | Value recovery |
| 2024 | +0.74 | +3.22 | Value recovery |

V2's underperformance is concentrated in 2019-2020 growth bull markets. In value-favored years (2023-2024), V2 actually outperforms.

---

## Stage D: Factor Meaning Drift — The Root Cause

### D1. Same Stock, Same Date, Different Factor Values
**Question**: For the same stock on the same date, how correlated are its factor values between V1 and V2 panels?

**Finding** (`output/factor_meaning_drift_audit.md`):

| Factor | Cross-Panel Rank r | Interpretation |
|--------|-------------------|---------------|
| Mom_6M | 0.913 | Minor shift |
| Mom_3M | 0.855 | Minor shift |
| NetMargin | 0.687 | Severe shift |
| EP | 0.482 | Fundamentally different |
| ROE | 0.371 | Fundamentally different |
| **ProfitGrowth** | **0.001** | **Completely random** |
| **RevGrowth** | **0.006** | **Completely random** |

**ProfitGrowth in V1 and V2 are essentially two different factors that happen to share a name.** The same stock's percentile rank changes almost randomly when moving from a 297-stock to a 1,360-stock universe.

### D2. Factor IC Migration
| Factor | IC V1 | IC V2 | Δ |
|--------|-------|-------|-----|
| EP | +0.077 | +0.093 | **improved** |
| NetMargin | +0.036 | +0.049 | improved |
| ROE | +0.053 | +0.059 | improved |
| **ProfitGrowth** | **+0.053** | **+0.022** | **−59% degraded** |

### D3. 2×2×2 Factorial Causal Decomposition
**Design**: Universe (V1/V2) × GS (OFF/ON) × colsample (1.0/0.5) — 8 models, single fold, common test date.

**IC Variance Decomposition** (`output/alpha_drift_causal_decomposition.md`):

| Source | Effect | % of IC Variance |
|--------|--------|-----------------|
| **Universe (data pipeline)** | +0.046 | **40%** |
| U×GS interaction | −0.026 | 22% |
| GS | +0.018 | 16% |
| colsample | +0.016 | 13% |

---

# Part 3. Major Experiments — Summary

| # | Experiment | Question | Key Result | Hypothesis Supported? |
|---|-----------|----------|------------|----------------------|
| 1 | SHAP Analysis | Is ProfitGrowth reversed? | Conditional effect, not simple reversal | ❌ Rejected "simple sign flip" |
| 2 | Turnover RCA | Why 14.5%→38.3% TO? | Rank stability collapse (0.984→0.718) | ✅ Confirmed stability as mechanism |
| 3 | Single-Model Ablation | Is colsample/GS root cause? | colsample=0.50 best stability; GS minor effect | ❌ Rejected parameter blame |
| 4 | Seed Consistency | Do seeds add noise? | Seed-pair r≈0.966 | ❌ Seeds nearly identical |
| 5 | Fold/Ensemble Audit | Is ensemble the cause? | fold=-1 is Sharpe-optimal (0.51) | ❌ Ensemble not the cause |
| 6 | V1 vs V2 Monthly Attribution | Where does Sharpe gap come from? | V2 loses in up markets; style radically different | ✅ Identified style+regime as drivers |
| 7 | 2×2×2 Factorial Causal Decomp | What's the causal ranking? | Universe 40%, U×GS 22%, GS 16%, CS 13% | ✅ Established causal ordering |
| 8 | Factor Meaning Drift Audit | Did factors change meaning? | ProfitGrowth r=0.001 between panels | ✅ Confirmed DGP shift |
| 9 | BP Factor Audit | Was BP incorrectly deleted? | BP residual IC=+0.004 (negligible) | ❌ GS deletion of BP was correct |
| 10 | GS Ranking Change Measurement | Does GS change tree model features? | 55% importance redistribution, ROE r=0.34 before/after | ✅ GS is not neutral for trees |

---

# Part 4. What We Now Believe

## High Confidence (multiple independent experiments confirm)

1. **V1 and V2 use fundamentally different factor signals.** Same factor names, different economic meanings. ProfitGrowth cross-panel rank r = 0.001.

2. **Seed variance is negligible.** 3 seeds → 1 seed with zero signal loss.

3. **Fold/ensemble architecture is not the Sharpe degradation cause.** Production fold=-1 is optimal.

4. **colsample=0.50 is not the root cause of instability.** It has the best single-model RankCorr.

5. **V2's turnover explosion (14.5%→38.3%) is caused by rank stability collapse (0.984→0.718).**

6. **V2 is an anti-growth, reduced-momentum alpha compared to V1's deep value+quality alpha.**

7. **BP's deletion by GS is statistically justified.** BP residual IC after EP = +0.004.

## Medium Confidence (plausible but not definitively proven)

1. **The universe expansion is the single largest causal factor** (40% of IC variance in factorial ANOVA). However, the single-model experiment design limits confidence.

2. **ROE exposure flip is caused by GS reordering**, not by ROE's economic meaning changing. ROE's factor decile curve still works in V2.

3. **The ProfitGrowth IC degradation is a signal-to-noise problem** caused by small-cap noise in the broader universe.

4. **GS is not neutral for tree models.** 55% feature importance redistribution is real, but whether this harms or helps depends on context.

## Low Confidence (data suggests but proof incomplete)

1. **Exact percentage attributions** (40% universe, 22% U×GS, etc.) are specific to the single-fold experimental design and may not generalize to the full walk-forward ensemble.

2. **"100% of ProfitGrowth drift is DGP shift"** — this is the best explanation but rests on correlation, not manipulation-based causal proof.

---

# Part 5. Rejected Hypotheses

| Hypothesis | Evidence Against | Strength of Rejection |
|-----------|-----------------|----------------------|
| "colsample=0.50 causes instability" | colsample=0.50 has best single-model RankCorr (0.839) | **Definitive** |
| "Fold ensemble causes Sharpe degradation" | All fold schemes ≤ production Sharpe (0.51) | **Definitive** |
| "Seeds cause instability" | Seed-pair r ≈ 0.966 | **Definitive** |
| "ProfitGrowth used in wrong direction" | SHAP shows conditional effect, not simple reversal | **Strong** |
| "GS causes worse stability" | GS OFF has similar RankCorr but higher turnover | **Strong** |
| "BP incorrectly deleted by GS" | BP residual IC = +0.004 (negligible independent signal) | **Strong** |
| "Fold diversity is harmful" | More folds → higher RankCorr (wisdom of crowd) | **Strong** |
| "Ensemble is over-ensemble" | RankCorr increases monotonically with model count | **Strong** |

---

# Part 6. Current Best Narrative

```
V1 was trained on preprocessed.parquet:
  297 large-cap stocks, ~155 stocks/date
  Standard z-score neutralization within 297-stock universe
  colsample=0.70, NO GS
  → Learned: high EP + high ROE + high ProfitGrowth = "deep value with quality"
  → Sharpe = 0.70, IC = 0.058

                          ↓ V2 rebuild changes THREE things simultaneously ↓

Change 1: Universe expands 297 → 1,360 CSI 800 stocks
  → Factor z-scores computed in different reference distribution
  → Same stock gets DIFFERENT z-score in V1 vs V2 panels
  → Fundamental factors (EP/ROE/PG) rank-shifted by 50-99%
  → "ProfitGrowth" in V2 ≠ "ProfitGrowth" in V1

Change 2: GS Orthogonalization added
  → Reorders 12/16 factors (some by >70% of stocks)
  → Eliminates BP (correctly — it's redundant with EP)
  → Redistributes 55% of feature importance
  → "GS-ROE" ≠ economic ROE (before/after r=0.34)

Change 3: colsample reduced 0.70 → 0.50
  → Restricts feature access per tree
  → Amplifies effects of changes 1 and 2

Result:
  V2 learns a DIFFERENT alpha:
    Moderate EP, NEGATIVE ROE, NEGATIVE ProfitGrowth, ANTI-Momentum
  → Performs poorly in growth bull markets (2019 SR=-1.32, 2020 SR=-0.70)
  → Performs well in value markets (2023 SR=+3.20, 2024 SR=+3.22)
  → Mean IC actually improves (0.058→0.062) but L/S return drops (1.11%→0.68%/mo)
  → Sharpe = 0.51
```

**This is a DGP (Data Generating Process) shift, not a model architecture failure.**

---

# Part 7. V1.5 Design Principles

Based on the investigation, any V1.5 design should observe these principles. (These are constraints derived from evidence, not a proposed design.)

1. **Retain the broad universe** (CSI 800). IC improves with breadth (0.058→0.062).

2. **But account for factor meaning drift.** Factors computed on 1,360 stocks are not the same as factors computed on 297 stocks. The same name ≠ the same signal.

3. **Turn GS OFF or severely constrain it.** GS reorders factors in ways that break their economic interpretation. 55% feature importance redistribution creates "GS-ROE" that the model learns correctly but that no longer means "ROE."

4. **Keep BP** (or don't let GS destroy it). Even if BP's independent IC is small, its removal changes model behavior.

5. **Raise colsample to 0.70-1.00.** Higher colsample reduces factor selection randomness and improves RankCorr.

6. **Add growth/quality guardrails.** The V2 anti-growth drift can be monitored and constrained through:
   - Factor exposure limits (max negative PG exposure)
   - Positive monotonicity constraints on growth/quality factors
   - Regime-aware ensemble weighting

7. **Use 1 seed.** Three seeds are completely redundant.

8. **Keep fold=-1.** It's already optimal for Sharpe.

---

# Part 8. Open Questions

These remain unanswered and should guide the next phase:

1. **Why does IC improve but Sharpe decline?** V2 IC=0.062 > V1 IC=0.058, but V2 Sharpe=0.51 < V1 Sharpe=0.70. Possible explanations (unverified):
   - IC benefits from small stocks that fail backtest liquidity/cost filters
   - V2's IC is more volatile at tails (large negative IC months hurt more)
   - V2's IC comes from factors with lower capacity

2. **What is the exact mechanism of ROE exposure flipping?** The data shows ROE's raw factor still works (decile spread +1.95% in V2), but the model loads negatively on GS-ROE. Is this purely GS reordering, or is there a model learning interaction?

3. **Is the V1 universe inherently better, or just more concentrated?** V1's 297-stock universe was a curated large-cap set. Were those stocks selected with look-ahead bias? If so, V1's Sharpe is partially in-sample.

4. **Can ProfitGrowth be salvaged through winsorization or alternative construction?** The factor's signal degrades because small-cap noise dominates. Aggressive winsorization or sector-relative computation might restore it.

5. **What is the optimal GS `max_correlation` threshold?** Current GS has no correlation cap — it aggressively orthogonalizes everything. A softer version (max_correlation=0.95 instead of 0.85) might reduce the EP-BP elimination without losing the benefits.

6. **How much of V1's Sharpe advantage is regime luck?** V1 only traded through 2024. V2's data extends to 2026. If 2025-2026 are value-unfavorable, V2's Sharpe would naturally look worse on extended history.

---

# Appendix A: Key Metrics Reference

## A1. Core Performance

| Metric | V1 | V2_Full | Source |
|--------|-----|---------|--------|
| Sharpe (Net) | 0.70 | 0.51 | `run_model_comparison.py` |
| Max Drawdown | −29.6% | −33.4% | `run_model_comparison.py` |
| Turnover (monthly) | 14.5% | 38.3% | `run_model_comparison.py` |
| Mean Rank IC | 0.0582 | 0.0615 | `run_model_comparison.py` |
| IC_IR | 0.539 | 0.538 | `run_model_comparison.py` |
| RankCorr (t, t+1) | 0.984 | 0.718 | `run_turnover_analysis.py` |
| Top30 Overlap | 81% | 34% | `run_turnover_analysis.py` |
| V1/V2 Rank r | — | 0.393 | `run_v1_v2_attribution.py` |

## A2. Style Exposures (Top30 Long)

| Factor | V1 | V2 | Shift |
|--------|-----|-----|-------|
| EP | +1.24 | +0.82 | −0.42 |
| ROE | +0.59 | −0.49 | −1.08 |
| ProfitGrowth | +0.47 | −0.99 | −1.46 |
| RevGrowth | +1.45 | +0.21 | −1.24 |
| Mom_3M | +0.05 | −0.36 | −0.41 |
| NetMargin | +0.28 | −0.23 | −0.51 |

## A3. Cross-Panel Factor Rank Stability

| Factor | Rank r (V1↔V2) |
|--------|----------------|
| Mom_6M | 0.913 |
| Mom_3M | 0.855 |
| NetMargin | 0.687 |
| EP | 0.482 |
| ROE | 0.371 |
| ProfitGrowth | 0.001 |
| RevGrowth | 0.006 |

## A4. Key Experiment Files

| File | Content |
|------|---------|
| `output/model_comparison_report.png` | V1 vs V2 NAV, IC, decile curves |
| `output/turnover_root_cause_analysis.md` | Rank stability, overlap, boundary analysis |
| `output/root_cause_ablation.md` | colsample/GS/universe single-model ablation |
| `output/ensemble_stability_root_cause.md` | Pairwise model correlations, seed/fold analysis |
| `output/fold_ensemble_production_audit.md` | Production inference chain + fold schemes |
| `output/v1_v2_monthly_attribution.md` | Monthly return attribution, regime analysis |
| `output/alpha_drift_forensic_audit.md` | GS ranking change, BP audit, V1 rebuild |
| `output/alpha_drift_causal_decomposition.md` | 2×2×2 factorial ANOVA |
| `output/factor_meaning_drift_audit.md` | Factor distribution, IC migration, decile curves |
| `output/research_report_shap_diagnosis.md` | SHAP analysis |

## A5. Key Scripts

| Script | Purpose |
|--------|---------|
| `run_model_comparison.py` | Head-to-head V1 vs V2 backtest |
| `run_turnover_analysis.py` | Rank stability, overlap, signal noise |
| `run_ablation.py` | Single-model colsample/GS/universe ablation |
| `run_ensemble_stability_analysis.py` | Pairwise model correlation, ensemble scale |
| `run_fold_ensemble_audit.py` | Production audit, fold scheme comparison |
| `run_v1_v2_attribution.py` | Monthly performance attribution |
| `run_alpha_drift_forensic.py` | GS ranking change, BP deletion, V1 rebuild |
| `run_causal_decomposition.py` | 2×2×2 factorial experiment |
| `run_factor_meaning_drift.py` | Factor distribution, IC migration, decile curves |

---

# Appendix B: Next Phase — V1.5 Research Roadmap

This section is a proposed starting point for the next researcher. It is not a plan of action — it is a structured list of open questions that can be converted into experiments.

## Phase I: Verify the Narrative (1-2 experiments)

**Goal**: Confirm that factor meaning drift is the dominant mechanism before designing any solution.

| Experiment | Method | Expected Output |
|-----------|--------|----------------|
| Rebuild V1 on V2 universe without GS | Train on V2 panel with GS=OFF, colsample=0.70 | Does the model recover V1-like exposures? |
| Sector-relative factor computation | Compute PG/ROE/EP as sector-relative z-scores instead of cross-sectional | Does rank stability improve? |

## Phase II: GS Mitigation (1-2 experiments)

**Goal**: Determine whether GS should be removed, softened, or replaced.

| Experiment | Method | Expected Output |
|-----------|--------|----------------|
| GS with max_correlation threshold | Test max_corr ∈ {0.85, 0.90, 0.95, 0.99} | Which threshold preserves factor meaning while controlling EP dominance? |
| Feature importance before/after GS × colsample | 2×2 factorial with full feature importance tracking | Quantify interaction precisely |

## Phase III: Growth/Quality Restoration (2-3 experiments)

**Goal**: Test whether adding growth/quality signals recovers V1-style performance.

| Experiment | Method | Expected Output |
|-----------|--------|----------------|
| Add sector-relative ProfitGrowth as new feature | Compare IC, exposure, Sharpe with/without | Does sector-relative PG restore growth exposure? |
| Monotonicity constraint on growth factors | LightGBM `monotone_constraint` on PG/ROE | Does forced monotonicity improve Sharpe? |
| Alternative growth factors | Test RevGrowth_YoY, EPS_YoY, ROE_YoY as replacements | Find least-degraded growth proxy in broad universe |

## Phase IV: Style Control (1-2 experiments)

**Goal**: Prevent unintended style drift in future retraining.

| Experiment | Method | Expected Output |
|-----------|--------|----------------|
| Factor exposure monitoring dashboard | Compute daily Top30 exposures, alert on drift >0.5σ | Operational guardrail |
| Regime-conditional ensemble | Weight models by recent regime performance | Adaptive style allocation |

---

*Report generated: 2026-06-21*
*Investigation duration: ~11 days across ~10 major experiments*
*Models trained: 0 full retrains; ~30 single-fold diagnostic models*
