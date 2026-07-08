"""
Phase B — Full Automated Pipeline: Data → Validation → Panel → Train.

Orchestrates the complete Phase B workflow:
  Step 0: Wait for data fetching to complete (--skip-fetch to bypass)
  Step 1: validate_data_integrity.py       → blacklist_symbols.csv
  Step 2: run_phaseb_rebuild_panel.py       → training_panel_v3_full.parquet
  Step 3: run_retrain_production.py --data  → production_models_v2_full/
  Step 4: Print comparison report card

Usage:
  python run_phaseb_pipeline.py                    # Full pipeline
  python run_phaseb_pipeline.py --skip-validation   # Skip validation
  python run_phaseb_pipeline.py --sample            # Test with 3 dates
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phaseb_pipeline")

OUTPUT_DIR = Path("output")
CHECKPOINT_FILE = Path(".phaseb_fetch_state.json")
BLACKLIST_PATH = OUTPUT_DIR / "blacklist_symbols.csv"
PANEL_PATH = OUTPUT_DIR / "training_panel_v3_full.parquet"
MODEL_DIR_V2 = OUTPUT_DIR / "production_models_v2"
MODEL_DIR_V2_FULL = OUTPUT_DIR / "production_models_v2_full"


def run_cmd(cmd: str, desc: str, timeout_min: int = 30) -> bool:
    """Run a command and log output. Returns True on success."""
    logger.info("=" * 72)
    logger.info("▶ %s", desc)
    logger.info("  CMD: %s", cmd)
    logger.info("=" * 72)

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout_min * 60,
            encoding="utf-8", errors="replace",
        )
        # Print stdout to terminal
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info("  %s", line)
        if result.stderr:
            for line in result.stderr.splitlines():
                if "WARNING" in line or "ERROR" in line or "CRITICAL" in line:
                    logger.warning("  [!] %s", line)

        if result.returncode != 0:
            logger.error("  ✗ COMMAND FAILED (exit=%d)", result.returncode)
            return False
        logger.info("  ✓ %s complete", desc)
        return True
    except subprocess.TimeoutExpired:
        logger.error("  ✗ TIMEOUT after %d min", timeout_min)
        return False
    except Exception as e:
        logger.error("  ✗ ERROR: %s", e)
        return False


def wait_for_data_fetch() -> bool:
    """Check if data fetching has completed. Polls every 30 seconds."""
    if not CHECKPOINT_FILE.exists():
        logger.info("No fetch checkpoint found — assuming data is ready")
        return True

    state = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    daily_done = len(state.get("daily_completed", []))
    daily_fail = len(state.get("daily_failed", {}))
    fin_done = len(state.get("financial_completed", []))
    fin_fail = len(state.get("financial_failed", {}))

    # Count actual CSV files
    daily_files = len(list(Path("data/raw").glob("daily_*.csv")))
    fin_files = len(list(Path("data/raw").glob("financial_*_pit.csv")))

    logger.info("Data fetch status: daily=%d files (%d ok/%d fail), financial=%d files (%d ok/%d fail)",
                 daily_files, daily_done, daily_fail, fin_files, fin_done, fin_fail)

    # Check if financial fetch is still needed
    need_financial = (fin_files < 500)  # rough threshold

    if need_financial:
        logger.info("Financial data still needed — running fetch...")
        return run_cmd(
            "python run_phaseb_fetch_data.py --fin-only",
            "Step 0b: Financial PIT Data Fetch",
            timeout_min=180,  # up to 3 hours for financial data
        )

    # Daily check: are we still fetching?
    if daily_files < 1000:
        logger.info("Daily data still fetching (%d files so far)... polling in 30s", daily_files)
        time.sleep(30)
        return False  # caller should retry

    return True


def print_report_card() -> None:
    """Generate comparison report card between old and new models."""
    logger.info("=" * 72)
    logger.info("  PHASE B PIPELINE — FINAL REPORT CARD")
    logger.info("=" * 72)

    # Panel stats
    if PANEL_PATH.exists():
        panel = pd.read_parquet(PANEL_PATH)
        n_cols = [c for c in panel.columns if c.endswith("_neutral_z")]
        logger.info("  Training Panel: %s", PANEL_PATH)
        logger.info("    Rows:     %d", len(panel))
        logger.info("    Dates:    %d", panel["date"].nunique())
        logger.info("    Symbols:  %d", panel["symbol"].nunique())
        logger.info("    Features: %d _neutral_z columns", len(n_cols))
    else:
        logger.warning("  Training Panel: NOT FOUND")

    # Old model feature importance
    if (MODEL_DIR_V2 / "feature_importance.csv").exists():
        old_imp = pd.read_csv(MODEL_DIR_V2 / "feature_importance.csv")
        # Handle both formats
        if "feature" in old_imp.columns:
            old_gain = old_imp.groupby("feature")["gain"].mean().sort_values(ascending=False)
        else:
            # Format from v2: columns are gain,split without feature names
            logger.warning("  Old model importance CSV missing 'feature' column — skipping comparison")
            old_gain = pd.Series(dtype=float)

        if not old_gain.empty:
            old_total = old_gain.sum()
            logger.info("  Old Model (v2) — Top 5 Features:")
            for i, (feat, val) in enumerate(old_gain.head(5).items()):
                logger.info("    %d. %-35s %8.1f (%5.1f%%)", i+1, feat, val, 100*val/old_total)

    # New model feature importance
    if (MODEL_DIR_V2_FULL / "feature_importance.csv").exists():
        new_imp = pd.read_csv(MODEL_DIR_V2_FULL / "feature_importance.csv")
        if "feature" in new_imp.columns:
            new_gain = new_imp.groupby("feature")["gain"].mean().sort_values(ascending=False)
        else:
            new_gain = pd.Series(dtype=float)

        if not new_gain.empty:
            new_total = new_gain.sum()
            logger.info("  New Model (v2.1 Full) — Top 5 Features:")
            for i, (feat, val) in enumerate(new_gain.head(5).items()):
                logger.info("    %d. %-35s %8.1f (%5.1f%%)", i+1, feat, val, 100*val/new_total)

            # EP dominance check
            if "EP_neutral_z_rank" in new_gain.index:
                ep_pct = 100 * new_gain["EP_neutral_z_rank"] / new_total
                status = "PASS" if ep_pct < 25 else "WARN"
                logger.info("  EP Dominance: %.1f%% [%s] (target: < 25%%)", ep_pct, status)

    # Model counts
    if MODEL_DIR_V2_FULL.exists():
        n_models = len(list(MODEL_DIR_V2_FULL.glob("model_*.pkl")))
        logger.info("  Models saved: %d boosters in %s", n_models, MODEL_DIR_V2_FULL)

    logger.info("=" * 72)
    logger.info("  Pipeline completed at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 72)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase B: Full Automated Pipeline")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetch wait")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation step")
    parser.add_argument("--skip-panel", action="store_true", help="Skip panel rebuild")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--sample", action="store_true", help="Test mode: 3 dates only")
    args = parser.parse_args()

    start_time = time.perf_counter()
    logger.info("╔" + "═" * 70 + "╗")
    logger.info("║  PHASE B: Full Automated Pipeline — Data → Validation → Panel → Train")
    logger.info("║  Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("╚" + "═" * 70 + "╝")

    # ── Step 0: Ensure data is ready ──
    if not args.skip_fetch:
        logger.info("\n[Step 0] Waiting for data fetch to complete...")
        for _ in range(360):  # wait up to 3 hours
            if wait_for_data_fetch():
                break
            time.sleep(30)
        else:
            logger.warning("Data fetch did not complete within 3 hours — proceeding anyway")
    else:
        logger.info("[Step 0] Skipping data fetch wait (--skip-fetch)")

    # ── Step 1: Validation ──
    if not args.skip_validation:
        logger.info("\n[Step 1] Running data integrity validation...")
        if not run_cmd("python validate_data_integrity.py", "Data Validation", timeout_min=120):
            logger.error("Validation FAILED. Check logs. Aborting pipeline.")
            sys.exit(1)
    else:
        logger.info("[Step 1] Skipping validation (--skip-validation)")

    # ── Step 2: Panel Rebuild ──
    if not args.skip_panel:
        logger.info("\n[Step 2] Rebuilding training panel...")
        sample_flag = " --sample" if args.sample else ""
        blacklist_flag = f" --blacklist {BLACKLIST_PATH}" if BLACKLIST_PATH.exists() else ""
        cmd = f"python run_phaseb_rebuild_panel.py{sample_flag}{blacklist_flag}"
        if not run_cmd(cmd, "Panel Rebuild", timeout_min=120):
            logger.error("Panel rebuild FAILED. Aborting pipeline.")
            sys.exit(1)
    else:
        logger.info("[Step 2] Skipping panel rebuild (--skip-panel)")

    # ── Step 3: Model Retraining ──
    if not args.skip_train:
        logger.info("\n[Step 3] Retraining ProductionAlphaEngine (v2.1)...")
        panel_path = PANEL_PATH if PANEL_PATH.exists() else "output/preprocessed.parquet"
        cmd = f"python run_retrain_production.py --data {panel_path} --output {MODEL_DIR_V2_FULL}"
        if not run_cmd(cmd, "Model Retraining", timeout_min=120):
            logger.error("Training FAILED. Aborting pipeline.")
            sys.exit(1)
    else:
        logger.info("[Step 3] Skipping training (--skip-train)")

    # ── Step 4: Report Card ──
    logger.info("\n[Step 4] Generating report card...")
    print_report_card()

    elapsed = time.perf_counter() - start_time
    logger.info("Total pipeline time: %.1f min", elapsed / 60)
    logger.info("Pipeline DONE.")


if __name__ == "__main__":
    main()
