# Quant Project Directory Restructure Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the cluttered quant project root into a professional, industry-standard layout (`src/` with subpackages, clean root, preserved data) without breaking imports or deleting raw data.

**Architecture:** Create a `src/` package with four subpackages (`data_pipeline`, `features`, `models`, `backtest`), move all source code from `factor_lib/`, `factor_research/`, `paper_trading/`, and `data/*.py` into them, create `experiments/` for research run scripts, and leave `data/raw/` untouched. All internal imports are rewritten to use the new `src.*` absolute paths.

**Tech Stack:** Python 3.x, shutil, pathlib, re (regex import rewriting)

## Global Constraints

- **DATA/ DIRECTORY IS PROTECTED** — `data/raw/` and `data/error_log.txt` must NOT be moved or deleted. Only `data/*.py` source files may be relocated.
- All `__init__.py` files must exist for every package directory.
- Internal cross-package imports in `src/` use absolute `from src.X.Y import Z` style.
- Root orchestration scripts update their imports to `from src.X.Y`.
- `.claude/`, `.pytest_cache/`, `.git/` are never touched.
- A dry-run mode must exist before any actual moves happen.

---

## File Classification Map

### 🔒 DO NOT TOUCH (stay exactly where they are)

| Path | Reason |
|------|--------|
| `data/raw/**` | Raw CSVs from AkShare — protected data |
| `data/error_log.txt` | Diagnostic log |
| `.claude/` | Claude config |
| `.pytest_cache/` | Pytest cache |
| `.git/` | Git repo |
| `.phaseb_fetch_state.json` | Pipeline runtime state |
| `.pipeline_state.json` | Pipeline runtime state |

### 📦 data/ → Split: source → src/data_pipeline/, raw data stays

| From | To |
|------|-----|
| `data/fetcher.py` | `src/data_pipeline/fetcher.py` |
| `data/cleaner.py` | `src/data_pipeline/cleaner.py` |
| `data/__init__.py` | DELETE (data/ becomes a plain data directory) |

### 📦 factor_lib/ → src/features/ (all factor calculation)

| From | To |
|------|-----|
| `factor_lib/__init__.py` | `src/features/__init__.py` |
| `factor_lib/momentum.py` | `src/features/momentum.py` |
| `factor_lib/value.py` | `src/features/value.py` |
| `factor_lib/growth.py` | `src/features/growth.py` |
| `factor_lib/quality.py` | `src/features/quality.py` |
| `factor_lib/volatility.py` | `src/features/volatility.py` |
| `factor_lib/technical.py` | `src/features/technical.py` |
| `factor_lib/sector_relative.py` | `src/features/sector_relative.py` |

### 📦 factor_research/ → Split across src/features/, src/models/, src/backtest/

| From | To |
|------|-----|
| `factor_research/orthogonalization.py` | `src/features/orthogonalization.py` |
| `factor_research/ml_engine.py` | `src/models/ml_engine.py` |
| `factor_research/ml_engine_v2.py` | `src/models/ml_engine_v2.py` |
| `factor_research/ml_engine_v5.py` | `src/models/ml_engine_v5.py` |
| `factor_research/ml_engine_v6.py` | `src/models/ml_engine_v6.py` |
| `factor_research/ml_engine_v7.py` | `src/models/ml_engine_v7.py` |
| `factor_research/production_engine.py` | `src/models/production_engine.py` |
| `factor_research/split_universe.py` | `src/models/split_universe.py` |
| `factor_research/backtest_engine.py` | `src/backtest/backtest_engine.py` |
| `factor_research/transaction_cost.py` | `src/backtest/transaction_cost.py` |
| `factor_research/dynamic_weight.py` | `src/backtest/dynamic_weight.py` |
| `factor_research/market_timing.py` | `src/backtest/market_timing.py` |
| `factor_research/ic_analysis.py` | `src/backtest/ic_analysis.py` |
| `factor_research/group_backtest.py` | `src/backtest/group_backtest.py` |
| `factor_research/report.py` | `src/backtest/report.py` |
| `factor_research/__init__.py` | DELETE (directory emptied) |

### 📦 paper_trading/ → Split across src/data_pipeline/, src/features/, tests/

| From | To |
|------|-----|
| `paper_trading/data_ingestion.py` | `src/data_pipeline/data_ingestion.py` |
| `paper_trading/baostock_adapter.py` | `src/data_pipeline/baostock_adapter.py` |
| `paper_trading/state_manager.py` | `src/data_pipeline/state_manager.py` |
| `paper_trading/paper_trading_pipeline.py` | `src/data_pipeline/paper_trading_pipeline.py` |
| `paper_trading/factor_compute.py` | `src/features/factor_compute.py` |
| `paper_trading/test_baostock_integration.py` | `tests/test_baostock_integration.py` |
| `paper_trading/__init__.py` | DELETE (directory emptied) |

### 📦 Root Scripts → Organized

**Keep at root (production/ essential):**
| From | To |
|------|-----|
| `run_phaseb_pipeline.py` | `run_pipeline.py` |
| `run_phaseb_fetch_data.py` | `run_fetch_data.py` |
| `run_phaseb_rebuild_panel.py` | `run_rebuild_panel.py` |
| `run_phaseb_constituents.py` | `run_constituents.py` |
| `run_retrain_production.py` | *(keep)* |
| `run_inference_export.py` | *(keep)* |
| `init_local_state.py` | *(keep)* |
| `init_local_state.bat` | *(keep)* |
| `check_factor_ic.py` | *(keep)* |
| `diagnose_stock_pool.py` | *(keep)* |
| `validate_data_integrity.py` | *(keep)* |
| `README.md` | *(keep)* |
| `requirements.txt` | *(keep)* |

**Move to `experiments/` (research/analysis scripts):**
- `run_ablation.py`
- `run_alpha_drift_analysis.py`
- `run_alpha_drift_forensic.py`
- `run_backtest_with_costs.py`
- `run_causal_decomposition.py`
- `run_dynamic_weight.py`
- `run_ensemble_stability_analysis.py`
- `run_factor_meaning_drift.py`
- `run_factor_research.py`
- `run_fold_ensemble_audit.py`
- `run_ml_ablation.py`
- `run_ml_backtest.py`
- `run_ml_lambdarank.py`
- `run_ml_turnover_aware.py`
- `run_ml_v6.py`
- `run_ml_v7.py`
- `run_model_comparison.py`
- `run_shap_diagnosis.py`
- `run_split_universe.py`
- `run_timing_comparison.py`
- `run_turnover_analysis.py`
- `run_v1_v2_attribution.py`
- `run_v15_evaluation.py`
- `run_v15_experiment.py`
- `run_v15_rebuild_panel.py`

**Move to `tests/`:**
| From | To |
|------|-----|
| `test_neutralization.py` | `tests/test_neutralization.py` |

### 📦 Other directories

| From | To |
|------|-----|
| `monitoring/` | *(keep as top-level, Streamlit dashboard)* |
| `notebooks/` | *(keep, already matches target)* |
| `output/` | *(keep, already matches target)* |
| `research/xhsanalysis.md` | `notebooks/xhsanalysis.md` |

### 🗑️ Old directories to remove (after all files moved out)

- `factor_lib/` (empty, all moved)
- `factor_research/` (empty, all moved)
- `paper_trading/` (empty, all moved)
- `research/` (empty, moved to notebooks)

### 🆕 New files to create

- `src/__init__.py` — makes `src` a package
- `src/data_pipeline/__init__.py` — re-exports key symbols
- `src/features/__init__.py` — re-exports key symbols
- `src/models/__init__.py` — re-exports key symbols
- `src/backtest/__init__.py` — re-exports key symbols
- `config.yaml` — global configuration template
- `experiments/__init__.py` — empty, for package compatibility
- `archive/` — empty directory for future archiving

---

## Import Path Rewrite Rules

The `organize_project.py` script applies these regex replacements to ALL `.py` files after moving:

### Cross-package imports (in root scripts AND internal files)

| Old Import Pattern | New Import Pattern |
|-------------------|-------------------|
| `from data.fetcher import X` | `from src.data_pipeline.fetcher import X` |
| `from data.cleaner import X` | `from src.data_pipeline.cleaner import X` |
| `from factor_lib.X import Y` | `from src.features.X import Y` |
| `from factor_research.orthogonalization import X` | `from src.features.orthogonalization import X` |
| `from factor_research.backtest_engine import X` | `from src.backtest.backtest_engine import X` |
| `from factor_research.transaction_cost import X` | `from src.backtest.transaction_cost import X` |
| `from factor_research.dynamic_weight import X` | `from src.backtest.dynamic_weight import X` |
| `from factor_research.market_timing import X` | `from src.backtest.market_timing import X` |
| `from factor_research.ic_analysis import X` | `from src.backtest.ic_analysis import X` |
| `from factor_research.group_backtest import X` | `from src.backtest.group_backtest import X` |
| `from factor_research.report import X` | `from src.backtest.report import X` |
| `from factor_research.ml_engine import X` | `from src.models.ml_engine import X` |
| `from factor_research.ml_engine_v2 import X` | `from src.models.ml_engine_v2 import X` |
| `from factor_research.ml_engine_v5 import X` | `from src.models.ml_engine_v5 import X` |
| `from factor_research.ml_engine_v6 import X` | `from src.models.ml_engine_v6 import X` |
| `from factor_research.ml_engine_v7 import X` | `from src.models.ml_engine_v7 import X` |
| `from factor_research.production_engine import X` | `from src.models.production_engine import X` |
| `from factor_research.split_universe import X` | `from src.models.split_universe import X` |
| `from paper_trading.data_ingestion import X` | `from src.data_pipeline.data_ingestion import X` |
| `from paper_trading.baostock_adapter import X` | `from src.data_pipeline.baostock_adapter import X` |
| `from paper_trading.state_manager import X` | `from src.data_pipeline.state_manager import X` |
| `from paper_trading.factor_compute import X` | `from src.features.factor_compute import X` |
| `from paper_trading.paper_trading_pipeline import X` | `from src.data_pipeline.paper_trading_pipeline import X` |

### Imports from `data.` that aren't `data.fetcher` or `data.cleaner` (should not happen, but guard)

Any `from data.` or `import data.` is checked — if it references `fetcher` or `cleaner`, rewrite; otherwise leave untouched (could reference the data directory path).

### Docstring / usage-string references

The script also scans for `from factor_research.X import` and `from paper_trading.X import` patterns inside docstrings and updates them to keep documentation consistent.

---

## Files That Will Have `ModuleNotFoundError` After Restructure

These files import from directories being moved and MUST have their imports updated:

### Critical (will break immediately):
1. **`paper_trading/paper_trading_pipeline.py:66,79,84,92,95`** — imports from `paper_trading.*` and `factor_research.*`
2. **`init_local_state.py:61,62`** — imports from `paper_trading.*`
3. **`run_alpha_drift_analysis.py:64,116`** — imports from `factor_research.*`
4. **`diagnose_stock_pool.py:22-28`** — imports from `data.*` and `factor_lib.*`
5. **`factor_research/backtest_engine.py:79,545`** — imports from `factor_research.*`
6. **`factor_research/split_universe.py:37-38`** — imports from `data.*` and `factor_research.*`
7. **`paper_trading/data_ingestion.py:1002,1725,2532`** — imports from `paper_trading.*`
8. **`tests/test_fetcher.py:3`** — imports from `data.*`
9. **`run_factor_research.py:32-49`** — imports from `data.*`, `factor_lib.*`, `factor_research.*`
10. **All `run_ml_*.py` scripts** — imports from `factor_research.*`
11. **`run_backtest_with_costs.py:36,42`** — imports from `factor_research.*`
12. **`run_dynamic_weight.py:33`** — imports from `factor_research.*`
13. **`run_fold_ensemble_audit.py:38,307,308`** — imports from `factor_research.*`
14. **`run_inference_export.py:15`** — imports from `factor_research.*`
15. **`run_model_comparison.py:34,35`** — imports from `factor_research.*`
16. **`run_retrain_production.py:99,344`** — imports from `paper_trading.*`, `factor_research.*`
17. **`run_phaseb_fetch_data.py:111`** — imports from `data.*`
18. **`run_split_universe.py:28-37`** — imports from `factor_research.*`
19. **`run_timing_comparison.py:40-45`** — imports from `factor_research.*`
20. **`run_v15_*.py`** — imports from `factor_lib.*`, `factor_research.*`
21. **`test_neutralization.py:18`** — imports from `data.*`
22. **`paper_trading/test_baostock_integration.py:25,88`** — imports from `paper_trading.*`
23. **All `factor_research/ml_engine*.py` docstrings** — reference old import paths

### Total files requiring import updates: ~60 Python files

The `organize_project.py` script handles ALL of these automatically via regex find-and-replace.

---

## organize_project.py Script Design

### Phases:
1. **DRY RUN (default):** Print all planned moves and import rewrites without executing
2. **BACKUP:** Create a git commit before any changes (requires clean working tree)
3. **CREATE DIRS:** Create `src/`, `src/data_pipeline/`, `src/features/`, `src/models/`, `src/backtest/`, `experiments/`, `archive/`
4. **MOVE FILES:** shutil.move each file to its new location
5. **REWRITE IMPORTS:** Open every .py file, apply regex replacements, write back
6. **CREATE __init__.py:** Write package init files with re-exports
7. **CREATE config.yaml:** Write a template config
8. **CLEANUP:** Remove empty old directories
9. **VERIFY:** Run `python -c "import src; print('OK')"` to confirm package loads

### Safety features:
- Git commit as restore point before any changes
- Dry-run mode prints everything, touches nothing
- Every `shutil.move` is wrapped in try/except
- Import rewrites are done AFTER moves, on files at their NEW locations
- Logs every action to `restructure_log.txt`

---
