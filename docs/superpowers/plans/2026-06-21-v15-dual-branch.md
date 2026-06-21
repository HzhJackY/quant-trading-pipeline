# V1.5 Dual-Branch Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate separate fundamental and technical LightGBM branches, blend their OOS ranks 50/50, and apply gap-aware EMA smoothing for a fixed Top-30 portfolio.

**Architecture:** Reuse `run_v15_experiment.train_single_model` for both branches so labels and walk-forward folds remain identical. Keep blending, gap-aware EMA state, fixed Top-30 evaluation, and style exposure analysis in the focused `run_v15_dual_branch.py` script.

**Tech Stack:** Python 3, pandas, NumPy, LightGBM, pytest, parquet.

## Global Constraints

- Fundamental feature list must match the approved 15-column list exactly.
- Technical feature list must match the approved 17-column list exactly.
- Both branches use the same panel, labels, dates, folds, and seed.
- Blend weights default to 0.5/0.5 and must be non-negative and sum to one.
- EMA defaults to 0.6 current / 0.4 previous.
- EMA state resets after any missed global OOS rebalance date.
- Primary portfolio is fixed Top 30 long-only.
- Existing single-model artifacts must not be overwritten.

---

### Task 1: Define Branch Configurations

**Files:**
- Create: `run_v15_dual_branch.py`
- Create: `tests/test_v15_dual_branch.py`

**Interfaces:**
- Produces: `FEATURES_FUNDA`, `FEATURES_TECH`, `MODEL_F_CONFIG`, `MODEL_T_CONFIG`.
- Consumes: `ExperimentConfig` and `train_single_model` from `run_v15_experiment.py`.

- [ ] **Step 1: Write failing tests for exact feature membership and configs**

```python
def test_feature_sets_match_approved_lists():
    assert FEATURES_FUNDA == [...]
    assert FEATURES_TECH == [...]
    assert set(FEATURES_FUNDA).isdisjoint(FEATURES_TECH)

def test_only_fundamental_branch_has_monotonicity():
    assert MODEL_F_CONFIG.monotone_constraints == {
        "EP_neutral_z": 1,
        "SR_ROE_neutral_z": 1,
        "SR_ProfitGrowth_YoY_neutral_z": 1,
    }
    assert MODEL_T_CONFIG.monotone_constraints == {}
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `python -m pytest tests/test_v15_dual_branch.py -q`

Expected: FAIL because `run_v15_dual_branch.py` does not exist.

- [ ] **Step 3: Implement constants and configs**

Create the exact approved lists and two `ExperimentConfig` objects:

```python
MODEL_F_CONFIG = ExperimentConfig(
    name="Model_F",
    description="Fundamental-only Quality+Value branch",
    panel_path=V15_PANEL_PATH,
    feature_neutral_z=FEATURES_FUNDA,
    gs_enabled=False,
    colsample_bytree=0.75,
    monotone_constraints={...},
)
```

`MODEL_T_CONFIG` uses `FEATURES_TECH` and no constraints.

- [ ] **Step 4: Run configuration tests**

Run: `python -m pytest tests/test_v15_dual_branch.py -q`

Expected: PASS.

### Task 2: Implement Rank Blend and Gap-Aware EMA

**Files:**
- Modify: `run_v15_dual_branch.py`
- Modify: `tests/test_v15_dual_branch.py`

**Interfaces:**
- Produces:
  - `blend_oos_predictions(pred_f, pred_t, weight_f=0.5, weight_t=0.5) -> pd.DataFrame`
  - `apply_gap_aware_ema(predictions, alpha=0.6) -> pd.DataFrame`

- [ ] **Step 1: Write failing blend tests**

Test independent per-date percentile ranks, exact weighted blend, negative
weights, weights not summing to one, date-set mismatch, and fewer than 30
intersecting stocks.

- [ ] **Step 2: Run tests and verify missing-function failures**

Run: `python -m pytest tests/test_v15_dual_branch.py -q`

Expected: FAIL on missing blend functions.

- [ ] **Step 3: Implement rank blend**

Merge branches on `date, symbol`, validate dates and intersection counts,
rank `pred_f` and `pred_t` per date, and calculate `raw_blend_pred`.

- [ ] **Step 4: Write failing EMA tests**

Fixtures cover:

- first observation initializes from raw signal;
- consecutive date applies `0.6 * current + 0.4 * previous`;
- missing one global rebalance date resets memory;
- a current NaN may reuse prior state only on a consecutive date.

- [ ] **Step 5: Implement gap-aware EMA**

Use the sorted global date sequence to assign each date an ordinal. Store
`symbol -> (last_date_ordinal, final_signal)`. Apply prior state only when the
last ordinal equals `current_ordinal - 1`.

- [ ] **Step 6: Run blend and EMA tests**

Run: `python -m pytest tests/test_v15_dual_branch.py -q`

Expected: PASS.

### Task 3: Implement Fixed Top-30 Evaluation and Exposures

**Files:**
- Modify: `run_v15_dual_branch.py`
- Modify: `tests/test_v15_dual_branch.py`

**Interfaces:**
- Produces:
  - `evaluate_top30(predictions, panel, signal_col="alpha_signal") -> dict`
  - `compute_top30_exposures(predictions, panel, signal_col="alpha_signal") -> dict`

- [ ] **Step 1: Write failing portfolio tests**

Use a controlled two-date fixture and assert:

- exactly 30 names selected;
- turnover is exited prior names divided by 30;
- long return is the mean forward return of selected names;
- ROE and PG exposures equal selected-name means.

- [ ] **Step 2: Run tests and verify missing-function failures**

Run: `python -m pytest tests/test_v15_dual_branch.py -q`

- [ ] **Step 3: Implement evaluation**

Compute forward returns from the panel close column, rank by signal, select
exactly 30 names per date, then compute Sharpe, MaxDD, turnover, mean return,
and Top30/Bottom30 L/S diagnostic Sharpe.

- [ ] **Step 4: Implement exposure calculation**

Report mean Top-30 exposures for:

```python
{
    "ROE": "SR_ROE_neutral_z",
    "ProfitGrowth": "SR_ProfitGrowth_YoY_neutral_z",
    "EP": "EP_neutral_z",
    "BP": "BP_raw_neutral_z",
}
```

- [ ] **Step 5: Run portfolio tests**

Run: `python -m pytest tests/test_v15_dual_branch.py -q`

Expected: PASS.

### Task 4: Implement CLI Training and Reporting

**Files:**
- Modify: `run_v15_dual_branch.py`
- Test: `tests/test_v15_dual_branch.py`

**Interfaces:**
- Produces:
  - branch OOS files;
  - `Dual_Final_oos.parquet`;
  - `v15_dual_branch_evaluation.csv`;
  - `v15_dual_branch_evaluation.md`.

- [ ] **Step 1: Add CLI/dry-run test**

Assert parser defaults:

- weights `0.5`, `0.5`;
- EMA alpha `0.6`;
- output directory `output/production_models_v15_dual`.

- [ ] **Step 2: Implement orchestration**

Flow:

1. load V1.5 panel once;
2. validate all feature columns;
3. train Model_F and Model_T with `train_single_model`;
4. blend and smooth;
5. save predictions;
6. load existing M5 baseline;
7. evaluate Single, Branch F, Branch T, Dual Final;
8. save comparison and verdict.

- [ ] **Step 3: Run unit tests and syntax compilation**

Run:

```powershell
python -m pytest tests/test_v15_dual_branch.py -q
python -m py_compile run_v15_dual_branch.py
python run_v15_dual_branch.py --dry-run
```

Expected: all commands exit zero.

### Task 5: Full Training and Evidence-Based Verification

**Files:**
- Generated: `output/production_models_v15_dual/*`
- Generated: `output/v15_dual_branch_evaluation.csv`
- Generated: `output/v15_dual_branch_evaluation.md`

- [ ] **Step 1: Train branches and final model**

Run: `python run_v15_dual_branch.py`

Expected:

- 71/71 folds for each branch;
- equal branch OOS date sets;
- 49,342 predictions before branch intersection, subject to current panel;
- final OOS parquet written.

- [ ] **Step 2: Run complete regression suite**

Run: `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 3: Inspect report requirements**

Confirm the report includes:

- Single, Branch F, Branch T, Dual Final;
- Sharpe, MaxDD, turnover, mean return, L/S Sharpe;
- ROE, ProfitGrowth, EP, BP exposures;
- explicit turnover `<30%` verdict;
- explicit ROE and ProfitGrowth positive verdicts.

- [ ] **Step 4: Report actual outcomes**

Do not claim the turnover or exposure targets were reached unless the generated
metrics demonstrate them.
