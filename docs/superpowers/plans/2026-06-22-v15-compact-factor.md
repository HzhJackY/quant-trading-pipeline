# V1.5 Compact Factor Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and compare three compact monthly LightGBM models that retain the 15-factor fundamental core and add at most three medium-horizon technical risk proxies.

**Architecture:** Reuse `run_v15_experiment.train_single_model` so all models share labels, folds, seed, and training objective. Implement model definitions and orchestration in `run_v15_compact_experiment.py`, then reuse fixed Top-30 evaluation and exposure functions from `run_v15_dual_branch.py`.

**Tech Stack:** Python, pandas, NumPy, LightGBM, pytest, parquet.

## Global Constraints

- Primary evaluation is fixed Top 30 long-only.
- Diagnostic evaluation is fixed Top30/Bottom30 long-short.
- Fundamental core contains exactly the approved 15 features.
- Compact-FT adds only `Mom_3M_neutral_z` and `Vol_60D_neutral_z`.
- Compact-FT3 additionally adds only `Mom_6M_neutral_z`.
- All models use 36/6/1 monthly walk-forward folds, seed 42, GS OFF, `colsample_bytree=0.75`, `learning_rate=0.05`, `reg_alpha=0.10`, and `lambda_turnover=2.0`.
- Existing single-model and dual-branch artifacts are read-only.
- `ROE_Stability` remains excluded because current usable coverage is zero.

---

### Task 1: Define Compact Model Configurations

**Files:**
- Create: `run_v15_compact_experiment.py`
- Create: `tests/test_v15_compact_experiment.py`

**Interfaces:**
- Consumes: `FEATURES_FUNDA` from `run_v15_dual_branch.py`.
- Produces: `COMPACT_F_CONFIG`, `COMPACT_FT_CONFIG`, `COMPACT_FT3_CONFIG`.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_compact_feature_sets_are_exact():
    assert COMPACT_F_CONFIG.feature_neutral_z == FEATURES_FUNDA
    assert COMPACT_FT_CONFIG.feature_neutral_z == FEATURES_FUNDA + [
        "Mom_3M_neutral_z", "Vol_60D_neutral_z"
    ]
    assert COMPACT_FT3_CONFIG.feature_neutral_z == FEATURES_FUNDA + [
        "Mom_3M_neutral_z", "Vol_60D_neutral_z", "Mom_6M_neutral_z"
    ]
```

Also assert identical fold, seed, GS, colsample, learning rate, regularization,
turnover objective, and monotonicity configuration.

- [ ] **Step 2: Run test and observe missing module failure**

Run: `python -m pytest tests/test_v15_compact_experiment.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement exact model configs**

Create three `ExperimentConfig` instances with distinct names and output files.

- [ ] **Step 4: Run configuration tests**

Run: `python -m pytest tests/test_v15_compact_experiment.py -q`

Expected: PASS.

### Task 2: Implement Training and Evaluation Orchestration

**Files:**
- Modify: `run_v15_compact_experiment.py`
- Modify: `tests/test_v15_compact_experiment.py`

**Interfaces:**
- Produces:
  - `run_compact_experiment(skip_training: bool = False) -> pd.DataFrame`
  - `evaluate_compact_predictions(predictions, panel) -> pd.DataFrame`

- [ ] **Step 1: Write failing dry-run and decision-rule tests**

Test CLI defaults, model names, output directory, and a pure function that
marks whether FT/FT3 pass all five approved selection rules.

- [ ] **Step 2: Run tests and observe missing-function failures**

Run: `python -m pytest tests/test_v15_compact_experiment.py -q`

- [ ] **Step 3: Implement orchestration**

For each config:

1. call `prepare_panel_for_config`;
2. call `train_single_model`;
3. save predictions under `output/production_models_v15_compact`;
4. evaluate using `evaluate_top30` and `compute_top30_exposures`;
5. compute annual fixed Top-30 Sharpe for stability diagnosis;
6. save CSV and Markdown reports.

- [ ] **Step 4: Implement selection verdict**

A mixed model passes only when:

- Sharpe is greater than Compact-F;
- MaxDD deterioration is no more than 0.05;
- turnover increase is no more than 0.10;
- ROE and ProfitGrowth are positive;
- improvements are not solely from one year.

- [ ] **Step 5: Run unit tests and dry-run**

```powershell
python -m pytest tests/test_v15_compact_experiment.py -q
python -m py_compile run_v15_compact_experiment.py
python run_v15_compact_experiment.py --dry-run
```

Expected: all commands exit zero.

### Task 3: Run Full Monthly Experiment

**Files:**
- Generated: `output/production_models_v15_compact/*`
- Generated: `output/v15_compact_evaluation.csv`
- Generated: `output/v15_compact_evaluation.md`

- [ ] **Step 1: Train all three models**

Run: `python run_v15_compact_experiment.py`

Expected: 71/71 folds and matching OOS dates for all models.

- [ ] **Step 2: Re-run without training**

Run: `python run_v15_compact_experiment.py --skip-training`

Expected: identical report metrics.

- [ ] **Step 3: Run relevant regression tests**

Run:

```powershell
python -m pytest tests/test_v15_compact_experiment.py tests/test_v15_dual_branch.py tests/test_v15_config.py -q
```

Expected: zero failures.

- [ ] **Step 4: Report evidence**

Report the actual fixed Top-30 Sharpe, MaxDD, turnover, L/S Sharpe, exposures,
and yearly stability. Do not recommend a mixed model unless every approved
decision rule passes.
