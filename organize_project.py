#!/usr/bin/env python3
"""
Organize Quant Project Directory — Safe Migration Script
=========================================================
Moves source files into a professional src/ layout, rewrites imports,
and creates necessary __init__.py files — WITHOUT touching data/raw/.

Usage:
    python organize_project.py --dry-run      # Preview all changes
    python organize_project.py --execute       # Actually perform the moves
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

# -- Configuration ----------------------------------------------------------

ROOT = Path(__file__).resolve().parent

# -- File Move Manifest -----------------------------------------------------
# Format: (source_relative, dest_relative)
# Source paths are relative to ROOT.
# Destinations use NEW target paths.

FILE_MOVES: list[tuple[str, str]] = [
    # -- data/*.py -> src/data_pipeline/ --
    ("data/fetcher.py", "src/data_pipeline/fetcher.py"),
    ("data/cleaner.py", "src/data_pipeline/cleaner.py"),

    # -- factor_lib/* -> src/features/ --
    ("factor_lib/__init__.py", "src/features/__init__.py"),
    ("factor_lib/momentum.py", "src/features/momentum.py"),
    ("factor_lib/value.py", "src/features/value.py"),
    ("factor_lib/growth.py", "src/features/growth.py"),
    ("factor_lib/quality.py", "src/features/quality.py"),
    ("factor_lib/volatility.py", "src/features/volatility.py"),
    ("factor_lib/technical.py", "src/features/technical.py"),
    ("factor_lib/sector_relative.py", "src/features/sector_relative.py"),

    # -- factor_research/ -> split across src/features/, src/models/, src/backtest/ --
    ("factor_research/orthogonalization.py", "src/features/orthogonalization.py"),
    ("factor_research/ml_engine.py", "src/models/ml_engine.py"),
    ("factor_research/ml_engine_v2.py", "src/models/ml_engine_v2.py"),
    ("factor_research/ml_engine_v5.py", "src/models/ml_engine_v5.py"),
    ("factor_research/ml_engine_v6.py", "src/models/ml_engine_v6.py"),
    ("factor_research/ml_engine_v7.py", "src/models/ml_engine_v7.py"),
    ("factor_research/production_engine.py", "src/models/production_engine.py"),
    ("factor_research/split_universe.py", "src/models/split_universe.py"),
    ("factor_research/backtest_engine.py", "src/backtest/backtest_engine.py"),
    ("factor_research/transaction_cost.py", "src/backtest/transaction_cost.py"),
    ("factor_research/dynamic_weight.py", "src/backtest/dynamic_weight.py"),
    ("factor_research/market_timing.py", "src/backtest/market_timing.py"),
    ("factor_research/ic_analysis.py", "src/backtest/ic_analysis.py"),
    ("factor_research/group_backtest.py", "src/backtest/group_backtest.py"),
    ("factor_research/report.py", "src/backtest/report.py"),

    # -- paper_trading/ -> split across src/data_pipeline/, src/features/, tests/ --
    ("paper_trading/data_ingestion.py", "src/data_pipeline/data_ingestion.py"),
    ("paper_trading/baostock_adapter.py", "src/data_pipeline/baostock_adapter.py"),
    ("paper_trading/state_manager.py", "src/data_pipeline/state_manager.py"),
    ("paper_trading/paper_trading_pipeline.py", "src/data_pipeline/paper_trading_pipeline.py"),
    ("paper_trading/factor_compute.py", "src/features/factor_compute.py"),
    ("paper_trading/test_baostock_integration.py", "tests/test_baostock_integration.py"),

    # -- test_neutralization.py -> tests/ --
    ("test_neutralization.py", "tests/test_neutralization.py"),

    # -- research/xhsanalysis.md -> notebooks/ --
    ("research/xhsanalysis.md", "notebooks/xhsanalysis.md"),

    # -- Phase-B production scripts -> renamed at root --
    ("run_phaseb_pipeline.py", "run_pipeline.py"),
    ("run_phaseb_fetch_data.py", "run_fetch_data.py"),
    ("run_phaseb_rebuild_panel.py", "run_rebuild_panel.py"),
    ("run_phaseb_constituents.py", "run_constituents.py"),

    # -- Experimental run scripts -> experiments/ --
    ("run_ablation.py", "experiments/run_ablation.py"),
    ("run_alpha_drift_analysis.py", "experiments/run_alpha_drift_analysis.py"),
    ("run_alpha_drift_forensic.py", "experiments/run_alpha_drift_forensic.py"),
    ("run_backtest_with_costs.py", "experiments/run_backtest_with_costs.py"),
    ("run_causal_decomposition.py", "experiments/run_causal_decomposition.py"),
    ("run_dynamic_weight.py", "experiments/run_dynamic_weight.py"),
    ("run_ensemble_stability_analysis.py", "experiments/run_ensemble_stability_analysis.py"),
    ("run_factor_meaning_drift.py", "experiments/run_factor_meaning_drift.py"),
    ("run_factor_research.py", "experiments/run_factor_research.py"),
    ("run_fold_ensemble_audit.py", "experiments/run_fold_ensemble_audit.py"),
    ("run_ml_ablation.py", "experiments/run_ml_ablation.py"),
    ("run_ml_backtest.py", "experiments/run_ml_backtest.py"),
    ("run_ml_lambdarank.py", "experiments/run_ml_lambdarank.py"),
    ("run_ml_turnover_aware.py", "experiments/run_ml_turnover_aware.py"),
    ("run_ml_v6.py", "experiments/run_ml_v6.py"),
    ("run_ml_v7.py", "experiments/run_ml_v7.py"),
    ("run_model_comparison.py", "experiments/run_model_comparison.py"),
    ("run_shap_diagnosis.py", "experiments/run_shap_diagnosis.py"),
    ("run_split_universe.py", "experiments/run_split_universe.py"),
    ("run_timing_comparison.py", "experiments/run_timing_comparison.py"),
    ("run_turnover_analysis.py", "experiments/run_turnover_analysis.py"),
    ("run_v1_v2_attribution.py", "experiments/run_v1_v2_attribution.py"),
    ("run_v15_evaluation.py", "experiments/run_v15_evaluation.py"),
    ("run_v15_experiment.py", "experiments/run_v15_experiment.py"),
    ("run_v15_rebuild_panel.py", "experiments/run_v15_rebuild_panel.py"),
]

# -- Directories to create --------------------------------------------------

DIRS_TO_CREATE: list[str] = [
    "src",
    "src/data_pipeline",
    "src/features",
    "src/models",
    "src/backtest",
    "experiments",
    "archive",
]

# -- Directories to remove (if empty after moves) --------------------------

DIRS_TO_CLEANUP: list[str] = [
    "factor_lib",
    "factor_research",
    "paper_trading",
    "research",
]

# -- Files to delete (old __init__.py that are no longer needed) ------------

FILES_TO_DELETE: list[str] = [
    "data/__init__.py",
]

# -- Import Rewrite Map -----------------------------------------------------
# Format: (old_import_prefix, new_import_prefix)
# Applied as:  from <old> import X  ->  from <new> import X
# Order matters! More specific matches first.

IMPORT_REWRITES: list[tuple[str, str]] = [
    # data.* -> src.data_pipeline.*
    ("data.fetcher", "src.data_pipeline.fetcher"),
    ("data.cleaner", "src.data_pipeline.cleaner"),

    # factor_lib.* -> src.features.*
    ("factor_lib.momentum", "src.features.momentum"),
    ("factor_lib.value", "src.features.value"),
    ("factor_lib.growth", "src.features.growth"),
    ("factor_lib.quality", "src.features.quality"),
    ("factor_lib.volatility", "src.features.volatility"),
    ("factor_lib.technical", "src.features.technical"),
    ("factor_lib.sector_relative", "src.features.sector_relative"),

    # factor_research.orthogonalization -> src.features.orthogonalization
    ("factor_research.orthogonalization", "src.features.orthogonalization"),

    # factor_research.backtest_engine -> src.backtest.*
    ("factor_research.backtest_engine", "src.backtest.backtest_engine"),
    ("factor_research.transaction_cost", "src.backtest.transaction_cost"),
    ("factor_research.dynamic_weight", "src.backtest.dynamic_weight"),
    ("factor_research.market_timing", "src.backtest.market_timing"),
    ("factor_research.ic_analysis", "src.backtest.ic_analysis"),
    ("factor_research.group_backtest", "src.backtest.group_backtest"),
    ("factor_research.report", "src.backtest.report"),

    # factor_research.ml_engine* -> src.models.*
    ("factor_research.ml_engine_v7", "src.models.ml_engine_v7"),
    ("factor_research.ml_engine_v6", "src.models.ml_engine_v6"),
    ("factor_research.ml_engine_v5", "src.models.ml_engine_v5"),
    ("factor_research.ml_engine_v2", "src.models.ml_engine_v2"),
    ("factor_research.ml_engine", "src.models.ml_engine"),
    ("factor_research.production_engine", "src.models.production_engine"),
    ("factor_research.split_universe", "src.models.split_universe"),

    # paper_trading.* -> src.data_pipeline.* or src.features.*
    ("paper_trading.data_ingestion", "src.data_pipeline.data_ingestion"),
    ("paper_trading.baostock_adapter", "src.data_pipeline.baostock_adapter"),
    ("paper_trading.state_manager", "src.data_pipeline.state_manager"),
    ("paper_trading.paper_trading_pipeline", "src.data_pipeline.paper_trading_pipeline"),
    ("paper_trading.factor_compute", "src.features.factor_compute"),
]

# -- Logging Setup ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "restructure_log.txt", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("organize")


# ===========================================================================
# Helper Functions
# ===========================================================================

def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    log.info(f"  [OK] Directory ensured: {path}")


def safe_move(src: Path, dst: Path, dry_run: bool = True) -> bool:
    """
    Move a file from src to dst with sanity checks.
    Returns True on success.
    """
    if not src.exists():
        log.error(f"  [FAIL]  Source missing: {src}")
        return False

    if dst.exists():
        log.warning(f"  [WARN]  Destination already exists, skipping: {dst}")
        return False

    if dry_run:
        log.info(f"  ->  [DRY RUN] Would move: {src}  ->  {dst}")
        return True

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        log.info(f"  [OK]  Moved: {src}  ->  {dst}")
        return True
    except Exception as e:
        log.error(f"  [FAIL]  Move failed: {src}  ->  {dst}  |  {e}")
        return False


def rewrite_imports_in_file(filepath: Path, dry_run: bool = True) -> int:
    """
    Rewrite import statements in a single Python file.
    Returns the number of replacements made.
    """
    if not filepath.exists():
        return 0

    try:
        original = filepath.read_text(encoding="utf-8")
    except Exception as e:
        log.warning(f"  [WARN]  Cannot read {filepath}: {e}")
        return 0

    modified = original
    replacement_count = 0

    for old_prefix, new_prefix in IMPORT_REWRITES:
        # Pattern: from <old_prefix> import <rest>
        # Handles: from X import Y, Z;  from X import (Y, Z)
        # Using a lambda replacement to avoid backreference issues
        pattern = rf"from\s+{re.escape(old_prefix)}\s+import\b"
        replacement = f"from {new_prefix} import"
        new_modified = re.sub(pattern, replacement, modified)
        if new_modified != modified:
            # Count actual replacements (one per line)
            matches = list(re.finditer(pattern, modified))
            replacement_count += len(matches)
            modified = new_modified

    if replacement_count > 0:
        if dry_run:
            log.info(f"  ->  [DRY RUN] Would rewrite {replacement_count} imports in: {filepath}")
        else:
            try:
                filepath.write_text(modified, encoding="utf-8")
                log.info(f"  [OK]  Rewrote {replacement_count} imports in: {filepath}")
            except Exception as e:
                log.error(f"  [FAIL]  Failed to write {filepath}: {e}")
                return 0

    return replacement_count


def collect_all_python_files() -> list[Path]:
    """Find all Python files in the project (excluding .git, .claude, __pycache__, venv)."""
    py_files: list[Path] = []
    exclude_dirs = {".git", ".claude", ".pytest_cache", "__pycache__", "venv", ".venv", "env"}

    for root, dirs, filenames in os.walk(ROOT):
        # Prune excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for fname in filenames:
            if fname.endswith(".py"):
                py_files.append(Path(root) / fname)

    return py_files


def create_init_file(package_path: Path, module_doc: str = "", dry_run: bool = True) -> None:
    """Create a package __init__.py with optional docstring."""
    init_file = package_path / "__init__.py"
    if init_file.exists():
        log.info(f"  -  __init__.py already exists: {init_file}")
        return

    content = f'"""{module_doc}"""\n' if module_doc else ""

    if dry_run:
        log.info(f"  ->  [DRY RUN] Would create: {init_file}")
    else:
        init_file.write_text(content, encoding="utf-8")
        log.info(f"  [OK]  Created: {init_file}")


# ===========================================================================
# Config Template
# ===========================================================================

CONFIG_YAML_TEMPLATE = """\
# ===========================================================================
# Quant Project — Global Configuration
# ===========================================================================

# -- Data Paths ------------------------------------------------------------
data:
  raw_dir: "data/raw"
  panel_cache: "output/panel.parquet"
  preprocessed: "output/preprocessed.parquet"

# -- Universe ---------------------------------------------------------------
universe:
  index: "csi800"               # csi300 | csi500 | csi800 | csi1000
  min_list_days: 60
  exclude_st: true

# -- Features ---------------------------------------------------------------
features:
  momentum:
    periods: [1, 3, 6, 12]
    skip_recent: 1
  volatility:
    windows: [20, 60]
  sector_relative:
    enabled: true
    shenwan_level: 1

# -- Model ------------------------------------------------------------------
model:
  type: "lightgbm"
  objective: "regression"
  num_leaves: 64
  learning_rate: 0.03
  num_boost_round: 500
  early_stopping_rounds: 50
  bagging_fraction: 0.8
  feature_fraction: 0.7
  lambda_l1: 0.5
  lambda_l2: 1.0
  seeds: [42, 888, 2026]
  n_folds: 54

# -- Backtest ---------------------------------------------------------------
backtest:
  initial_capital: 1_000_000_000   # 1B RMB
  commission_bps: 2.5              # 2.5 bps
  slippage_bps: 1.0               # 1.0 bps
  stamp_duty_bps: 10.0            # 10 bps (sell only)
  max_position_pct: 0.01          # 1% max per stock
  rebalance_frequency: "monthly"

# -- Output -----------------------------------------------------------------
output:
  model_dir: "output/production_models"
  report_dir: "output"
  log_level: "INFO"
"""


# ===========================================================================
# Main Orchestration
# ===========================================================================

def phase_create_directories(dry_run: bool) -> None:
    """Create all new directory structures."""
    log.info("-" * 70)
    log.info("PHASE 1: Creating directories")
    log.info("-" * 70)

    for d in DIRS_TO_CREATE:
        dpath = ROOT / d
        if dry_run:
            log.info(f"  ->  [DRY RUN] Would create: {dpath}")
        else:
            ensure_dir(dpath)


def phase_move_files(dry_run: bool) -> tuple[int, int]:
    """Move all files according to FILE_MOVES manifest."""
    log.info("-" * 70)
    log.info("PHASE 2: Moving files")
    log.info("-" * 70)

    success, fail = 0, 0
    for src_rel, dst_rel in FILE_MOVES:
        src_path = ROOT / src_rel
        dst_path = ROOT / dst_rel
        if safe_move(src_path, dst_path, dry_run=dry_run):
            success += 1
        else:
            fail += 1

    log.info(f"  Moves: {success} succeeded, {fail} failed/skipped")
    return success, fail


def phase_delete_old_inits(dry_run: bool) -> None:
    """Delete __init__.py files from directories that are no longer packages."""
    log.info("-" * 70)
    log.info("PHASE 3: Removing old __init__.py from data/")
    log.info("-" * 70)

    for rel_path in FILES_TO_DELETE:
        fpath = ROOT / rel_path
        if not fpath.exists():
            log.info(f"  -  Already gone: {fpath}")
            continue
        if dry_run:
            log.info(f"  ->  [DRY RUN] Would delete: {fpath}")
        else:
            fpath.unlink()
            log.info(f"  [OK]  Deleted: {fpath}")


def phase_rewrite_imports(dry_run: bool) -> int:
    """Scan and rewrite imports in ALL Python files in the project."""
    log.info("-" * 70)
    log.info("PHASE 4: Rewriting import statements")
    log.info("-" * 70)

    py_files = collect_all_python_files()
    total_replacements = 0
    files_touched = 0

    for fpath in sorted(py_files):
        n = rewrite_imports_in_file(fpath, dry_run=dry_run)
        if n > 0:
            total_replacements += n
            files_touched += 1

    log.info(f"  Total: {total_replacements} replacements across {files_touched} files")
    return total_replacements


def phase_create_inits(dry_run: bool) -> None:
    """Create __init__.py for all new packages."""
    log.info("-" * 70)
    log.info("PHASE 5: Creating __init__.py files")
    log.info("-" * 70)

    init_specs = {
        "src": "Quantitative Research — Core Source Package",
        "src/data_pipeline": "Data fetching, cleaning, sector-relative adjustment, and Z-score logic",
        "src/features": "Factor computation, Gram-Schmidt orthogonalization, and alpha definitions",
        "src/models": "LightGBM training, single-fold validation, hyperparameter config, and production ensembling",
        "src/backtest": "Backtest engine, attribution, transaction cost modeling, and style exposure analysis",
        "experiments": "Experimental and one-off research run scripts",
    }

    for rel_path, doc in init_specs.items():
        create_init_file(ROOT / rel_path, doc, dry_run=dry_run)


def phase_create_config(dry_run: bool) -> None:
    """Create config.yaml template if it doesn't exist."""
    log.info("-" * 70)
    log.info("PHASE 6: Creating config.yaml")
    log.info("-" * 70)

    config_path = ROOT / "config.yaml"
    if config_path.exists():
        log.info(f"  -  config.yaml already exists, skipping")
        return

    if dry_run:
        log.info(f"  ->  [DRY RUN] Would create: {config_path}")
    else:
        config_path.write_text(CONFIG_YAML_TEMPLATE, encoding="utf-8")
        log.info(f"  [OK]  Created: {config_path}")


def phase_cleanup_empty_dirs(dry_run: bool) -> None:
    """Remove old directories if they're empty."""
    log.info("-" * 70)
    log.info("PHASE 7: Cleaning up empty old directories")
    log.info("-" * 70)

    for rel_path in DIRS_TO_CLEANUP:
        dpath = ROOT / rel_path
        if not dpath.exists():
            log.info(f"  -  Already gone: {dpath}")
            continue

        # Check if directory is empty (or only contains __pycache__)
        contents = [p for p in dpath.iterdir() if p.name != "__pycache__"]
        if contents:
            log.warning(f"  [WARN]  Directory not empty, skipping: {dpath}")
            log.warning(f"       Remaining: {[p.name for p in contents]}")
            continue

        if dry_run:
            log.info(f"  ->  [DRY RUN] Would remove: {dpath}")
        else:
            # Remove __pycache__ first if present
            pycache = dpath / "__pycache__"
            if pycache.exists():
                shutil.rmtree(str(pycache), ignore_errors=True)
            dpath.rmdir()
            log.info(f"  [OK]  Removed: {dpath}")


def phase_verify(dry_run: bool) -> bool:
    """Verify the new package structure is importable."""
    log.info("-" * 70)
    log.info("PHASE 8: Verification")
    log.info("-" * 70)

    if dry_run:
        log.info("  ->  [DRY RUN] Would verify package imports")
        return True

    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", "import src; print('src package OK')"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        log.info("  [OK]  src package imports successfully")
    else:
        log.error(f"  [FAIL]  Import failed: {result.stderr}")
        return False

    # Check data/raw is intact
    data_raw = ROOT / "data" / "raw"
    if data_raw.exists():
        csv_count = len(list(data_raw.glob("*.csv")))
        log.info(f"  [OK]  data/raw/ intact ({csv_count} CSV files)")
    else:
        log.error("  [FAIL]  data/raw/ MISSING — DO NOT PROCEED")
        return False

    return True


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely restructure quant project into professional src/ layout"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview all actions without executing (default)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually perform the restructuring",
    )
    parser.add_argument(
        "--skip-git-check",
        action="store_true",
        default=False,
        help="Skip the clean git working tree check",
    )
    args = parser.parse_args()

    dry_run = not args.execute

    # -- Safety: Print banner --
    print()
    print("=" * 70)
    if dry_run:
        print("  [DRY RUN MODE] No files will be changed")
    else:
        print("  [EXECUTE MODE] Files WILL be moved and rewritten")
    print("=" * 70)
    print(f"  Project root: {ROOT}")
    print(f"  Moves planned: {len(FILE_MOVES)}")
    print()

    # -- Safety: Check git status before executing --
    if not dry_run and not args.skip_git_check:
        import subprocess
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            log.error("Git working tree is NOT clean. Commit or stash changes first.")
            log.error("Or use --skip-git-check to bypass this safety check.")
            log.error("Uncommitted changes:")
            for line in result.stdout.strip().split("\n")[:20]:
                log.error(f"  {line}")
            sys.exit(1)
        log.info("[OK] Git working tree is clean")

    # -- Execute phases --
    phases = [
        ("Create directories", phase_create_directories),
        ("Move files", phase_move_files),
        ("Delete old __init__.py", phase_delete_old_inits),
        ("Rewrite imports", phase_rewrite_imports),
        ("Create __init__.py", phase_create_inits),
        ("Create config.yaml", phase_create_config),
        ("Cleanup empty dirs", phase_cleanup_empty_dirs),
    ]

    for phase_name, phase_fn in phases:
        try:
            phase_fn(dry_run)
        except Exception as e:
            log.error(f"Phase '{phase_name}' failed: {e}")
            if not dry_run:
                log.error("ABORTING — some changes may have been applied. Check git diff.")
                sys.exit(1)

    # -- Verify --
    ok = phase_verify(dry_run)

    # -- Summary --
    print()
    print("=" * 70)
    if dry_run:
        print("  [PASS]  DRY RUN COMPLETE — Review the log above")
        print("  Run with --execute to apply changes")
    elif ok:
        print("  [PASS]  RESTRUCTURE COMPLETE")
        print()
        print("  Next steps:")
        print("  1. Review restructure_log.txt")
        print("  2. Run:  pytest tests/")
        print("  3. Run:  python run_pipeline.py --help")
        print("  4. Commit: git add -A && git commit -m 'refactor: restructure project'")
    else:
        print("  [FAIL]  Verification failed — check errors above")
        print("  Restore with: git checkout . && git clean -fd")
    print("=" * 70)


if __name__ == "__main__":
    main()
