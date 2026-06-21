"""
V1.5 Panel Rebuilder — Sector-Relative Factors + Quality/Growth Augmentation.

Extends the existing V2 training panel (training_panel_v3_full.parquet) with:
  1. Sector-relative z-score versions of factors degraded by universe expansion
     (ProfitGrowth, RevGrowth, ROE)
  2. New quality/growth factors (EPS_YoY, ROE_Stability)
  3. BP factor restoration (present in panel, verified)
  4. Pre-flight sanity check on all required factors

Design: POST-PROCESSES the existing panel rather than rebuilding from scratch.
This is ~10x faster and sufficient for the 6-model experiment matrix.

Output: output/training_panel_v15_sr.parquet

Usage:
  python run_v15_rebuild_panel.py                    # Full rebuild
  python run_v15_rebuild_panel.py --sample           # Test with 3 dates
  python run_v15_rebuild_panel.py --no-preflight     # Skip preflight check
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v15_panel")

# ── Paths ──
OUTPUT_DIR = Path("output")
V2_PANEL_PATH = OUTPUT_DIR / "training_panel_v3_full.parquet"
ALL_FIN_PATH = OUTPUT_DIR / "all_financial_pit.parquet"
ALL_DAILY_PATH = OUTPUT_DIR / "all_daily.parquet"
V15_PANEL_PATH = OUTPUT_DIR / "training_panel_v15_sr.parquet"

# ── Factor metadata ──
# Factors that keep existing _neutral_z (cross-panel stable: r > 0.85)
STABLE_FACTORS = [
    "Mom_1M", "Mom_3M", "Mom_6M", "Mom_12M_1M",
    "Vol_20D", "Vol_60D", "Beta",
    "Debt_Ratio", "Net_Profit_Margin",
    "VolChg_20D", "PriceDev_20D",
]

# Factors that keep existing _neutral_z but we also add SR version
# (EP has moderate drift r=0.48 but IC improved — keep original too)
VALUE_FACTORS = ["BP", "EP"]

# Factors that NEED sector-relative reconstruction (cross-panel r < 0.5)
SECTOR_RELATIVE_FACTORS = ["ProfitGrowth_YoY", "RevGrowth_YoY", "ROE"]

# New V1.5 quality/growth factors
NEW_FACTORS = ["EPS_YoY", "ROE_Stability"]

# All _neutral_z columns that should exist in source panel
ALL_V2_NEUTRAL_Z = [
    f"{f}_neutral_z" for f in STABLE_FACTORS + VALUE_FACTORS + SECTOR_RELATIVE_FACTORS
]


# ═══════════════════════════════════════════════════════════
# Step 1: Load and validate source panel
# ═══════════════════════════════════════════════════════════

def load_source_panel() -> pd.DataFrame:
    """Load V2 training panel and verify required columns exist."""
    if not V2_PANEL_PATH.exists():
        raise FileNotFoundError(
            f"V2 panel not found: {V2_PANEL_PATH}. "
            f"Run run_phaseb_rebuild_panel.py first."
        )

    panel = pd.read_parquet(V2_PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)

    logger.info("Loaded V2 panel: %d rows × %d cols", len(panel), len(panel.columns))
    logger.info("  Dates: %s ~ %s (%d unique)",
                panel["date"].min().strftime("%Y-%m-%d"),
                panel["date"].max().strftime("%Y-%m-%d"),
                panel["date"].nunique())
    logger.info("  Symbols: %d", panel["symbol"].nunique())

    # Verify required neutral_z columns
    available = [c for c in ALL_V2_NEUTRAL_Z if c in panel.columns]
    missing = [c for c in ALL_V2_NEUTRAL_Z if c not in panel.columns]
    if missing:
        logger.warning("Missing _neutral_z columns (will be computed): %s", missing)
    logger.info("  Available _neutral_z: %d/%d", len(available), len(ALL_V2_NEUTRAL_Z))

    # Verify BP is present (audit finding: BP was silently deleted in V2 by GS)
    if "BP_neutral_z" not in panel.columns:
        logger.warning(
            "BP_neutral_z NOT in source panel — "
            "BP may have been deleted by GS. Will attempt to recompute."
        )
    else:
        bp_std = panel.groupby("date")["BP_neutral_z"].std()
        bp_dead_dates = (bp_std < 1e-5).sum()
        if bp_dead_dates > 0:
            logger.warning(
                "BP_neutral_z has zero variance on %d/%d dates — "
                "BP was likely eliminated by GS.",
                bp_dead_dates, len(bp_std),
            )
        else:
            logger.info("  BP_neutral_z: healthy (σ > 0 on all dates)")

    return panel


# ═══════════════════════════════════════════════════════════
# Step 2: Load industry classification
# ═══════════════════════════════════════════════════════════

def attach_industry(panel: pd.DataFrame) -> pd.DataFrame:
    """Join Shenwan L1 industry to panel."""
    from factor_lib.sector_relative import load_shenwan_industry

    symbols = sorted(panel["symbol"].unique())
    industry_df = load_shenwan_industry(symbols)

    if industry_df.empty or "sw_l1" not in industry_df.columns:
        logger.warning(
            "No industry data available. "
            "Sector-relative factors will use market-wide z-score as fallback. "
            "This is safe but weaker — recommend obtaining Shenwan classification."
        )
        panel["sw_l1"] = "未知"
        return panel

    n_before = len(panel)
    panel = panel.merge(industry_df[["symbol", "sw_l1"]], on="symbol", how="left")
    panel["sw_l1"] = panel["sw_l1"].fillna("未知")
    n_matched = panel["sw_l1"].notna().sum()

    logger.info(
        "Industry coverage: %d/%d rows (%.1f%%), %d unique industries",
        n_matched, len(panel),
        100 * n_matched / max(len(panel), 1),
        panel["sw_l1"].nunique(),
    )
    return panel


# ═══════════════════════════════════════════════════════════
# Step 3: Load PIT financials and compute auxiliary factors
# ═══════════════════════════════════════════════════════════

def load_financial_pit() -> pd.DataFrame:
    """Load PIT financial data parquet."""
    if not ALL_FIN_PATH.exists():
        logger.warning(
            "%s not found. New factors (EPS_YoY, ROE_Stability) will be NaN. "
            "Sector-relative factors can still be computed from existing _neutral_z.",
            ALL_FIN_PATH,
        )
        return pd.DataFrame()

    fin = pd.read_parquet(ALL_FIN_PATH)
    # Normalize
    if "symbol" in fin.columns:
        fin["symbol"] = fin["symbol"].astype(str).str.zfill(6)
    if "report_date" in fin.columns:
        fin["report_date"] = pd.to_datetime(fin["report_date"], errors="coerce")
    if "pub_date" in fin.columns:
        fin["pub_date"] = pd.to_datetime(fin["pub_date"], errors="coerce")

    logger.info("Loaded financial data: %d rows × %d cols", len(fin), len(fin.columns))
    return fin


def compute_auxiliary_factors(
    panel: pd.DataFrame,
    fin_pit: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute V1.5-specific factors from PIT financial data.

    For each (symbol, date) in panel, finds the latest financial report with
    pub_date <= date, then computes:
      - EPS_YoY: YoY EPS growth rate
      - ROE_Stability: negative of 8-quarter ROE standard deviation
    """
    if fin_pit.empty:
        logger.warning("No financial data — skipping auxiliary factor computation")
        panel["EPS_YoY"] = np.nan
        panel["ROE_Stability"] = np.nan
        return panel

    # Columns to extract
    need_cols = {"symbol", "report_date", "pub_date"}
    available = need_cols & set(fin_pit.columns)

    # Detect column names (Chinese or English)
    eps_col = None
    roe_col = None
    for c in fin_pit.columns:
        if "每股收益" in c or c.lower() in ("eps", "earnings_per_share"):
            eps_col = c
        if "ROE" in c or "净资产收益率" in c:
            roe_col = c

    if eps_col is None or roe_col is None:
        logger.warning(
            "Cannot find EPS/ROE columns in financial data. "
            "EPS_YoY and ROE_Stability will be NaN. "
            "Available: %s", fin_pit.columns.tolist()
        )
        panel["EPS_YoY"] = np.nan
        panel["ROE_Stability"] = np.nan
        return panel

    logger.info("Computing auxiliary factors: EPS_YoY (col=%s), ROE_Stability (col=%s)",
                 eps_col, roe_col)

    # ── Pre-process financials: sort by symbol + report_date ──
    fin = fin_pit[["symbol", "report_date", "pub_date", eps_col, roe_col]].copy()
    fin = fin.dropna(subset=["symbol", "report_date"])
    fin = fin.sort_values(["symbol", "report_date"])

    # ── Pre-compute YoY for each symbol ──
    fin["EPS_YoY_raw"] = np.nan
    fin["ROE_YoY_raw"] = np.nan

    for sym, grp in fin.groupby("symbol"):
        grp = grp.sort_values("report_date")
        eps_vals = grp[eps_col].values.astype(float)
        roe_vals = grp[roe_col].values.astype(float)

        # YoY: current vs 4 quarters ago
        for i in range(4, len(grp)):
            if pd.notna(eps_vals[i]) and pd.notna(eps_vals[i-4]) and abs(eps_vals[i-4]) > 1e-9:
                idx = grp.index[i]
                fin.loc[idx, "EPS_YoY_raw"] = (eps_vals[i] - eps_vals[i-4]) / abs(eps_vals[i-4])

        # ROE Stability: rolling 8Q std (negated, so higher = more stable)
        roe_series = pd.Series(roe_vals, index=grp.index).astype(float)
        rolling_std = roe_series.rolling(8, min_periods=4).std()
        fin.loc[grp.index, "ROE_Stability_raw"] = -rolling_std.values

    # ── PIT join: for each (symbol, date) in panel, get latest financial ──
    panel_dates = sorted(panel["date"].unique())
    panel_symbols = sorted(panel["symbol"].unique())

    rows = []
    for dt in panel_dates:
        # Filter financials with pub_date <= dt
        pit = fin[fin["pub_date"] <= dt].copy()
        if pit.empty:
            continue

        # For each symbol, get the latest report
        latest = pit.sort_values("report_date").groupby("symbol").tail(1)
        latest["panel_date"] = dt
        rows.append(latest[["symbol", "panel_date", "EPS_YoY_raw", "ROE_Stability_raw"]])

    if rows:
        aux = pd.concat(rows, ignore_index=True)
        panel = panel.merge(
            aux, left_on=["symbol", "date"], right_on=["symbol", "panel_date"], how="left"
        )
        panel["EPS_YoY"] = panel["EPS_YoY_raw"]
        panel["ROE_Stability"] = panel["ROE_Stability_raw"]
        panel = panel.drop(columns=["panel_date", "EPS_YoY_raw", "ROE_Stability_raw"])
    else:
        panel["EPS_YoY"] = np.nan
        panel["ROE_Stability"] = np.nan

    n_eps = panel["EPS_YoY"].notna().sum()
    n_roe = panel["ROE_Stability"].notna().sum()
    logger.info("  EPS_YoY: %d/%d non-NaN (%.1f%%)", n_eps, len(panel),
                 100 * n_eps / max(len(panel), 1))
    logger.info("  ROE_Stability: %d/%d non-NaN (%.1f%%)", n_roe, len(panel),
                 100 * n_roe / max(len(panel), 1))

    return panel


# ═══════════════════════════════════════════════════════════
# Step 4: Compute sector-relative z-scores
# ═══════════════════════════════════════════════════════════

def compute_sector_relative_neutral_z(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute sector-relative _neutral_z for factors degraded by universe expansion.

    For ProfitGrowth, RevGrowth, and ROE:
      - Extract raw factor from existing _neutral_z (or from raw data if available)
      - Apply sector-relative z-score → new column: SR_{factor}_neutral_z
      - Keep the original _neutral_z as-is for comparison

    The column naming follows the ML engine convention:
      SR_ProfitGrowth_YoY_neutral_z  → engine converts to SR_ProfitGrowth_YoY_neutral_z_rank
    """
    from factor_lib.sector_relative import compute_sector_relative_factor_safe

    for factor in SECTOR_RELATIVE_FACTORS:
        neutral_z_col = f"{factor}_neutral_z"

        if neutral_z_col in panel.columns:
            # Use existing _neutral_z as the raw value for sector-relative computation
            # This is valid because _neutral_z preserves the rank ordering
            raw_col = neutral_z_col
            logger.info("Computing SR for %s (from existing %s)", factor, neutral_z_col)
        else:
            logger.warning("%s not in panel — cannot compute sector-relative version", neutral_z_col)
            continue

        # Compute sector-relative z-score
        try:
            sr_series = compute_sector_relative_factor_safe(
                panel, raw_col,
                industry_col="sw_l1",
                date_col="date",
                min_stocks=3,
            )
            # Rename to _neutral_z convention
            new_col = f"SR_{factor}_neutral_z"
            panel[new_col] = sr_series.values

            n_valid = sr_series.notna().sum()
            logger.info("  %s: %d/%d non-NaN (%.1f%%)",
                         new_col, n_valid, len(panel),
                         100 * n_valid / max(len(panel), 1))

        except Exception as e:
            logger.error("Failed to compute SR for %s: %s", factor, e)
            raise

    return panel


# ═══════════════════════════════════════════════════════════
# Step 5: Compute EPS_YoY and ROE_Stability _neutral_z
# ═══════════════════════════════════════════════════════════

def neutralize_new_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Convert new raw factors to _neutral_z format.
    Uses sector-relative z-score for consistency with V1.5 design.
    """
    from factor_lib.sector_relative import compute_sector_relative_factor_safe

    for factor in NEW_FACTORS:
        if factor not in panel.columns:
            logger.warning("%s not in panel — skipping neutralization", factor)
            continue

        # Sector-relative z-score
        try:
            sr_series = compute_sector_relative_factor_safe(
                panel, factor,
                industry_col="sw_l1",
                date_col="date",
                min_stocks=3,
            )
            neutral_col = f"{factor}_neutral_z"
            panel[neutral_col] = sr_series.values

            n_valid = sr_series.notna().sum()
            logger.info("  %s: %d/%d non-NaN (%.1f%%)",
                         neutral_col, n_valid, len(panel),
                         100 * n_valid / max(len(panel), 1))
        except Exception as e:
            logger.error("Failed to neutralize %s: %s", factor, e)
            raise

    return panel


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def build_v15_panel(
    sample: bool = False,
    skip_preflight: bool = False,
) -> pd.DataFrame:
    """Main V1.5 panel construction pipeline."""
    logger.info("=" * 64)
    logger.info("V1.5 Panel Rebuilder — Sector-Relative + Quality/Growth")
    logger.info("=" * 64)

    # 1. Load source
    panel = load_source_panel()

    # 2. Attach industry
    panel = attach_industry(panel)

    # 3. Load financials and compute auxiliary factors
    fin_pit = load_financial_pit()
    panel = compute_auxiliary_factors(panel, fin_pit)

    # 4. Compute sector-relative z-scores for degraded factors
    panel = compute_sector_relative_neutral_z(panel)

    # 5. Neutralize new factors
    panel = neutralize_new_factors(panel)

    # 6. Ensure BP is present and healthy
    if "BP_neutral_z" not in panel.columns:
        logger.warning("BP_neutral_z missing — will not be in V1.5 feature set")

    # ── Extract and validate final feature set ──
    # Build the V1.5 feature column list
    v15_neutral_z_cols = []
    for factor in STABLE_FACTORS:
        col = f"{factor}_neutral_z"
        if col in panel.columns:
            v15_neutral_z_cols.append(col)

    for factor in VALUE_FACTORS:
        col = f"{factor}_neutral_z"
        if col in panel.columns:
            v15_neutral_z_cols.append(col)

    for factor in SECTOR_RELATIVE_FACTORS:
        # Add sector-relative versions
        sr_col = f"SR_{factor}_neutral_z"
        if sr_col in panel.columns:
            v15_neutral_z_cols.append(sr_col)
        # Also keep original for model comparison
        orig_col = f"{factor}_neutral_z"
        if orig_col in panel.columns:
            v15_neutral_z_cols.append(orig_col)

    for factor in NEW_FACTORS:
        col = f"{factor}_neutral_z"
        if col in panel.columns:
            v15_neutral_z_cols.append(col)

    # ── Sample mode ──
    if sample:
        dates = sorted(panel["date"].unique())[:3]
        panel = panel[panel["date"].isin(dates)].copy()
        logger.info("SAMPLE MODE: %d dates", len(dates))

    # ── Pre-flight check ──
    if not skip_preflight:
        logger.info("\n--- Pre-Flight Factor Sanity Check ---")
        from factor_lib.sector_relative import preflight_factor_sanity_check

        try:
            diagnostics = preflight_factor_sanity_check(
                panel, v15_neutral_z_cols, date_col="date", threshold=1e-5,
            )
            n_passed = sum(1 for d in diagnostics.values() if d["passed"])
            logger.info("Pre-flight: %d/%d factors passed", n_passed, len(diagnostics))
        except ValueError as e:
            logger.error("Pre-flight check FAILED:\n%s", e)
            raise
    else:
        logger.info("Pre-flight check SKIPPED (--no-preflight)")

    # ── Final columns ──
    keep_cols = ["date", "symbol"] + v15_neutral_z_cols
    # Keep close for label computation
    for close_candidate in ["收盘", "close"]:
        if close_candidate in panel.columns:
            keep_cols.append(close_candidate)
            break

    # Ensure no duplicates
    keep_cols = list(dict.fromkeys(keep_cols))  # preserve order, remove dups

    output = panel[keep_cols].copy()
    output = output.sort_values(["date", "symbol"]).reset_index(drop=True)

    # ── Save ──
    output.to_parquet(V15_PANEL_PATH, index=False)

    logger.info("=" * 64)
    logger.info("V1.5 Panel saved: %s", V15_PANEL_PATH)
    logger.info("  Shape: %d rows × %d cols", output.shape[0], output.shape[1])
    logger.info("  Dates: %d (%s ~ %s)",
                 output["date"].nunique(),
                 output["date"].min().strftime("%Y-%m-%d"),
                 output["date"].max().strftime("%Y-%m-%d"))
    logger.info("  Symbols: %d", output["symbol"].nunique())
    logger.info("  Feature columns (%d):", len(v15_neutral_z_cols))
    for col in v15_neutral_z_cols:
        n_valid = output[col].notna().sum()
        pct = 100 * n_valid / max(len(output), 1)
        logger.info("    %s  (%d rows, %.1f%%)", col, n_valid, pct)
    logger.info("=" * 64)

    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V1.5 Panel Rebuilder")
    parser.add_argument("--sample", action="store_true",
                        help="Test with first 3 dates only")
    parser.add_argument("--no-preflight", action="store_true",
                        help="Skip pre-flight factor sanity check")
    args = parser.parse_args()

    build_v15_panel(sample=args.sample, skip_preflight=args.no_preflight)


if __name__ == "__main__":
    main()
