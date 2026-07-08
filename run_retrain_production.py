"""
Production Model Retraining — Universe-Aligned + EP-Corrected V7 LightGBM.

Phase A: Quick retrain using existing preprocessed data with:
  1. CSI 800 membership check (already sampled from CSI 800)
  2. Estimated market cap filter (≥ 5B CNY)
  3. Feature Gram-Schmidt orthogonalization (reduces EP dominance)
  4. Reduced colsample_bytree (0.50 → forces feature diversity)
  5. Same V7 architecture otherwise (1M label, 0M gap, TO λ=2.0)

Usage:
  python run_retrain_production.py                    # Full retrain
  python run_retrain_production.py --quick           # Skip orthogonalization
  python run_retrain_production.py --dry-run          # Just show data stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrain")

OUTPUT_DIR = Path("output")
PREPROCESSED_PATH = OUTPUT_DIR / "preprocessed.parquet"
CSI800_CACHE = OUTPUT_DIR / "csi800_members.parquet"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "production_models_v2_full"

# ── The 16 factor columns (must match metadata.json) ──
FACTOR_COLS = [
    "Mom_1M", "Mom_3M", "Mom_6M", "Mom_12M_1M",
    "Vol_20D", "Vol_60D", "Beta",
    "BP", "EP",
    "ROE", "Debt_Ratio", "Net_Profit_Margin",
    "RevGrowth_YoY", "ProfitGrowth_YoY",
    "VolChg_20D", "PriceDev_20D",
]

# Columns used for label computation and filtering
CLOSE_COL = "收盘"
AMOUNT_COL = "成交额"
NET_PROFIT_COL = "净利润"
EPS_COL = "每股收益"
PRICE_COL = "股价"


# ═══════════════════════════════════════════════════════════
# Step 1: Load and filter data
# ═══════════════════════════════════════════════════════════

def load_data(data_path: str | None = None) -> pd.DataFrame:
    """Load preprocessed panel, keeping only necessary columns."""
    path = Path(data_path) if data_path else PREPROCESSED_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run rebuild panel script first."
        )

    df = pd.read_parquet(path)
    logger.info("Loaded %s: %d rows × %d cols", path, len(df), len(df.columns))

    # Verify required columns
    for col in [CLOSE_COL, "date", "symbol"]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in preprocessed data")

    # Verify all _neutral_z columns exist
    neutral_z_cols = [f"{f}_neutral_z" for f in FACTOR_COLS]
    missing = [c for c in neutral_z_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing _neutral_z columns: {missing}")

    return df


def load_csi800_members() -> set[str]:
    """Load CSI 800 constituent list."""
    if CSI800_CACHE.exists():
        csi = pd.read_parquet(CSI800_CACHE)
        symbols = set(str(s).zfill(6) for s in csi["symbol"].unique())
        logger.info("CSI 800 members: %d (from cache)", len(symbols))
        return symbols

    # Try to fetch live
    try:
        from paper_trading.data_ingestion import fetch_csi800_members
        csi = fetch_csi800_members(force_refresh=False)
        symbols = set(str(s).zfill(6) for s in csi["symbol"].unique())
        logger.info("CSI 800 members: %d (fetched)", len(symbols))
        return symbols
    except Exception as e:
        logger.warning("Cannot fetch CSI 800 members: %s. Skipping CSI filter.", e)
        return set()


def estimate_market_cap(df: pd.DataFrame) -> pd.Series:
    """
    Estimate total market cap from available data.
    total_mcap ≈ (净利润 / 每股收益) × 股价

    Where 净利润 and 每股收益 are trailing-twelve-month.
    Falls back to NaN where data is incomplete.
    """
    has_net_profit = NET_PROFIT_COL in df.columns
    has_eps = EPS_COL in df.columns
    has_price = PRICE_COL in df.columns

    if not (has_net_profit and has_eps and has_price):
        logger.warning(
            "Cannot estimate market cap: need %s, %s, %s",
            NET_PROFIT_COL, EPS_COL, PRICE_COL,
        )
        return pd.Series(np.nan, index=df.index)

    net_profit = pd.to_numeric(df[NET_PROFIT_COL], errors="coerce")
    eps = pd.to_numeric(df[EPS_COL], errors="coerce")
    price = pd.to_numeric(df[PRICE_COL], errors="coerce")

    # shares = net_profit / eps
    # mcap = shares * price
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.where(
            (eps.abs() > 1e-9) & (net_profit.abs() > 0),
            net_profit / eps,
            np.nan,
        )
        mcap = shares * price

    return pd.Series(mcap, index=df.index)


def apply_training_filters(
    df: pd.DataFrame,
    csi800: set[str] | None,
    min_mcap: float = 5_000_000_000,
) -> pd.DataFrame:
    """
    Apply training-time quality filters.

    Returns filtered DataFrame.
    """
    initial = len(df)
    initial_dates = df["date"].nunique()
    initial_symbols = df["symbol"].nunique()

    # 1. CSI 800 membership
    if csi800 and len(csi800) > 0:
        before = len(df)
        df = df[df["symbol"].astype(str).str.zfill(6).isin(csi800)].copy()
        logger.info("  CSI 800 filter: %d → %d rows (removed %d non-CSI-800)",
                     before, len(df), before - len(df))

    # 2. Market cap filter
    mcap = estimate_market_cap(df)
    if mcap.notna().sum() > 0:
        before = len(df)
        valid_mcap = mcap >= min_mcap
        df = df[valid_mcap | mcap.isna()].copy()  # Keep stocks where we can't estimate
        removed = before - len(df)
        logger.info(
            "  Market cap ≥ %.0fB: %d → %d rows (removed %d micro-caps)"
            " | mcap estimated for %d/%d rows",
            min_mcap / 1e9, before, len(df), removed,
            mcap.notna().sum(), before,
        )

    # 3. Remove rows with NaN _neutral_z (can't compute features)
    neutral_z_cols = [f"{f}_neutral_z" for f in FACTOR_COLS]
    before = len(df)
    df = df.dropna(subset=neutral_z_cols, how="any")
    logger.info("  NaN _neutral_z: %d → %d rows", before, len(df))

    # 4. Remove dates with too few stocks (engine also does this, but early filter helps)
    date_counts = df.groupby("date").size()
    good_dates = date_counts[date_counts >= 30].index  # Lower than engine's 50
    before_dates = df["date"].nunique()
    df = df[df["date"].isin(good_dates)].copy()
    logger.info("  Date quality (≥30 stocks): %d → %d dates", before_dates, len(good_dates))

    final = len(df)
    logger.info(
        "Training filters: %d → %d rows (%.1f%% retained) | %d → %d symbols | %d → %d dates",
        initial, final, 100 * final / max(initial, 1),
        initial_symbols, df["symbol"].nunique(),
        initial_dates, df["date"].nunique(),
    )

    return df


# ═══════════════════════════════════════════════════════════
# Step 2: Feature orthogonalization (Gram-Schmidt by IC_IR)
# ═══════════════════════════════════════════════════════════

def _safe_rank_ic(group: pd.DataFrame, factor_col: str, return_col: str) -> float:
    """Compute Spearman rank IC for a single cross-section."""
    valid = group[[factor_col, return_col]].dropna()
    if len(valid) < 30:
        return 0.0
    from scipy.stats import spearmanr
    corr, _ = spearmanr(valid[factor_col], valid[return_col])
    return corr if not np.isnan(corr) else 0.0


def compute_ic_ir_ranking(
    df: pd.DataFrame,
    neutral_z_cols: list[str],
    date_col: str = "date",
) -> list[str]:
    """
    Rank _neutral_z factors by historical IC_IR (descending).
    Higher IC_IR = more predictive = ordered first in Gram-Schmidt.
    """
    # Need forward return for IC computation
    if "forward_return_1m" not in df.columns:
        df = df.sort_values(["symbol", date_col]).copy()
        df["forward_return_1m"] = (
            df.groupby("symbol")[CLOSE_COL]
            .transform(lambda x: x.shift(-1) / x - 1.0)
        )

    ic_irs = {}
    for col in neutral_z_cols:
        ics = df.groupby(date_col).apply(
            lambda g: _safe_rank_ic(g, col, "forward_return_1m"),
            include_groups=False,
        )
        ic_mean = ics.mean()
        ic_std = ics.std()
        ic_ir = abs(ic_mean / ic_std) if ic_std and ic_std > 0 else 0.0
        ic_irs[col] = ic_ir

    ranked = sorted(ic_irs.items(), key=lambda x: -x[1])
    logger.info("Factors ranked by IC_IR:")
    for name, ir_val in ranked:
        logger.info("  %-30s IC_IR = %.4f", name.replace("_neutral_z", ""), ir_val)

    return [name for name, _ in ranked]


def gram_schmidt_orthogonalize(
    df: pd.DataFrame,
    neutral_z_cols: list[str],
    date_col: str = "date",
    max_correlation: float = 0.85,
) -> pd.DataFrame:
    """
    Cross-sectional Gram-Schmidt orthogonalization within each date.

    Factors are orthogonalized in IC_IR-descending order:
      - Factor with highest IC_IR is preserved as-is
      - Each subsequent factor has the projection of all prior factors removed
      - If a factor's correlation with prior factors > max_correlation, it's penalized

    This reduces EP dominance by removing EP's linear overlap with BP, ROE, etc.
    """
    # Rank by IC_IR
    ranked_cols = compute_ic_ir_ranking(df, neutral_z_cols, date_col)
    n_factors = len(ranked_cols)

    result = df.copy()
    new_cols = []

    for date, date_grp in result.groupby(date_col):
        idx = date_grp.index
        n_stocks = len(idx)

        # Extract factor matrix: (n_stocks, n_factors)
        X = np.zeros((n_stocks, n_factors))
        for j, col in enumerate(ranked_cols):
            vals = date_grp[col].values.astype(np.float64)
            # Fill NaN with cross-sectional mean
            col_mean = np.nanmean(vals)
            X[:, j] = np.where(np.isnan(vals), col_mean, vals)

        # Gram-Schmidt
        Q = np.zeros_like(X)
        for j in range(n_factors):
            v = X[:, j].copy()
            # Remove projection onto all previous orthogonalized vectors
            for k in range(j):
                proj = np.dot(v, Q[:, k]) / max(np.dot(Q[:, k], Q[:, k]), 1e-12)
                v = v - proj * Q[:, k]

            # Check residual correlation with prior factors
            # If this factor is too correlated with prior ones, shrink it
            max_abs_corr = 0.0
            for k in range(j):
                if np.std(v) > 1e-12 and np.std(Q[:, k]) > 1e-12:
                    corr = np.corrcoef(v, Q[:, k])[0, 1]
                    max_abs_corr = max(max_abs_corr, abs(corr))

            if max_abs_corr > max_correlation:
                # Shrink toward zero
                shrink = max_correlation / max_abs_corr
                v = v * shrink

            # Normalize to unit variance (preserve z-score scale)
            v_std = np.std(v)
            if v_std > 1e-12:
                v = v / v_std
            Q[:, j] = v

        # Write back orthogonalized values
        for j, col in enumerate(ranked_cols):
            result.loc[idx, col] = Q[:, j]

        new_cols = ranked_cols  # track

    logger.info(
        "Gram-Schmidt orthogonalized %d factors within each of %d dates",
        n_factors, result[date_col].nunique(),
    )
    return result


# ═══════════════════════════════════════════════════════════
# Step 3: Train
# ═══════════════════════════════════════════════════════════

def train_and_save(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    colsample_bytree: float = 0.50,
    lambda_turnover: float = 2.0,
) -> dict:
    """
    Train ProductionAlphaEngine and save models.
    """
    from factor_research.production_engine import (
        ProductionAlphaEngine,
        ProductionConfig,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    config = ProductionConfig(
        seeds=[42, 888, 2026],
        lambda_turnover=lambda_turnover,
        colsample_bytree=colsample_bytree,
        # Keep all other hyperparameters at V7 defaults
    )

    logger.info("=" * 64)
    logger.info("Training ProductionAlphaEngine (V7)")
    logger.info("  colsample_bytree: %.2f (default: 0.70)", colsample_bytree)
    logger.info("  lambda_turnover: %.2f", lambda_turnover)
    logger.info("  seeds: %s", config.seeds)
    logger.info("  Features: %d", len(FACTOR_COLS))
    logger.info("=" * 64)

    engine = ProductionAlphaEngine(config)

    t0 = time.perf_counter()
    engine.fit(
        df,
        blended=None,  # No prev_signal for cold-start training
        date_col="date",
        symbol_col="symbol",
        close_col=CLOSE_COL,
    )
    train_time = time.perf_counter() - t0

    logger.info("Training completed in %.1f min (%d folds × %d seeds)",
                 train_time / 60, engine._n_folds, len(config.seeds))

    # Save
    engine.save_models(output_dir, mode="backtest")

    # Save feature importance summary
    if engine._feature_importance is not None:
        imp = engine._feature_importance
        imp_path = output_dir / "feature_importance.csv"
        imp.to_csv(imp_path, index=False, encoding="utf-8-sig")
        logger.info("Feature importance saved to %s", imp_path)

        # Log top features
        logger.info("Top 5 features by gain:")
        top = (
            imp.groupby("feature")["gain"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        total_gain = top.sum()
        for feat, gain_val in top.items():
            pct = 100 * gain_val / total_gain if total_gain > 0 else 0
            logger.info("  %-35s %.2f (%.1f%%)", feat, gain_val, pct)

    return {
        "train_time_min": train_time / 60,
        "n_folds": engine._n_folds,
        "n_seeds": len(config.seeds),
        "n_features": len(FACTOR_COLS),
        "output_dir": str(output_dir),
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Retrain ProductionAlphaEngine with universe-aligned pipeline"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Skip Gram-Schmidt orthogonalization (faster)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only show data statistics, don't train",
    )
    parser.add_argument(
        "--colsample", type=float, default=0.50,
        help="LightGBM colsample_bytree (default: 0.50, V7: 0.70)",
    )
    parser.add_argument(
        "--min-mcap", type=float, default=5_000_000_000,
        help="Minimum estimated market cap in CNY (default: 5B)",
    )
    parser.add_argument(
        "--output", type=str, default=str(MODEL_OUTPUT_DIR),
        help="Model output directory",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to preprocessed panel parquet (default: output/preprocessed.parquet)",
    )
    args = parser.parse_args()

    # ── Load ──
    logger.info("=" * 64)
    logger.info("Production Model Retraining (V2.1 Frozen Hyperparams)")
    logger.info("=" * 64)

    df = load_data(args.data)
    csi800 = load_csi800_members()

    # ── Filter ──
    logger.info("\n--- Applying Training Filters ---")
    df = apply_training_filters(df, csi800, min_mcap=args.min_mcap)

    # ── Orthogonalize ──
    neutral_z_cols = [f"{f}_neutral_z" for f in FACTOR_COLS]
    if not args.quick:
        logger.info("\n--- Gram-Schmidt Orthogonalization ---")
        df = gram_schmidt_orthogonalize(df, neutral_z_cols, date_col="date")
    else:
        logger.info("\n--- Skipping orthogonalization (--quick) ---")

    if args.dry_run:
        logger.info("\n--- Dry Run Summary ---")
        logger.info("Rows: %d", len(df))
        logger.info("Dates: %d", df["date"].nunique())
        logger.info("Symbols: %d", df["symbol"].nunique())
        logger.info(
            "Stocks/date: min=%d, max=%d, mean=%.0f",
            df.groupby("date").size().min(),
            df.groupby("date").size().max(),
            df.groupby("date").size().mean(),
        )
        logger.info("Date range: %s → %s", df["date"].min(), df["date"].max())
        logger.info(
            "Estimated folds: %d",
            df["date"].nunique() - 36 - 6 - 1,  # train + val + test window
        )
        return

    # ── Train ──
    logger.info("\n--- Training ---")
    result = train_and_save(
        df,
        Path(args.output),
        colsample_bytree=args.colsample,
    )

    # ── Summary ──
    logger.info("\n" + "=" * 64)
    logger.info("Retraining Complete!")
    logger.info("  Training time: %.1f min", result["train_time_min"])
    logger.info("  Folds: %d", result["n_folds"])
    logger.info("  Seeds: %d", result["n_seeds"])
    logger.info("  Features: %d", result["n_features"])
    logger.info("  Models saved to: %s", result["output_dir"])
    logger.info("=" * 64)
    logger.info("Next: Update paper_trading_pipeline.py to use %s", result["output_dir"])


if __name__ == "__main__":
    main()
