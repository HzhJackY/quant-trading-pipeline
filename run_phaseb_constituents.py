"""
Phase B Step 2: Historical CSI 800 Constituent Snapshots.

CSI 800 = CSI 300 (沪深300) + CSI 500 (中证500).
Uses Baostock query_hs300_stocks() + query_zz500_stocks() with date parameter
to reconstruct CSI 800 constituents at each semi-annual index review.

Output: output/csi800_history.parquet
  Columns: snapshot_date, symbol, index_source (HS300/ZZ500/CSI800)

Usage: python run_phaseb_constituents.py
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("phaseb_constituents")

OUTPUT_DIR = Path("output")
SNAPSHOT_PATH = OUTPUT_DIR / "csi800_history.parquet"

# CSI index reviews are semi-annual (June and December).
# The new constituents take effect ~mid-month. Using month-end as effective date.
# We include one extra snapshot at the start (2017-01-01) to cover early 2017.
SNAPSHOT_DATES = [
    "2017-01-01",  # initial snapshot for early 2017
    "2017-06-30", "2017-12-31",
    "2018-06-30", "2018-12-31",
    "2019-06-30", "2019-12-31",
    "2020-06-30", "2020-12-31",
    "2021-06-30", "2021-12-31",
    "2022-06-30", "2022-12-31",
    "2023-06-30", "2023-12-31",
    "2024-06-30", "2024-12-31",
    "2025-06-30", "2025-12-31",
    "2026-06-30",  # latest
]


def fetch_snapshot(date_str: str) -> pd.DataFrame:
    """
    Fetch CSI 800 constituents for a single date.
    Source: Baostock HS300 + ZZ500 union.
    """
    import baostock as bs

    bs.login()
    try:
        # CSI 300
        rs300 = bs.query_hs300_stocks(date=date_str)
        hs300_data = rs300.get_data() if rs300.error_code == "0" else pd.DataFrame()

        # CSI 500
        rs500 = bs.query_zz500_stocks(date=date_str)
        zz500_data = rs500.get_data() if rs500.error_code == "0" else pd.DataFrame()

        # Build unified DataFrame
        rows = []
        if not hs300_data.empty:
            for _, row in hs300_data.iterrows():
                code = str(row.get("code", "")).replace("sh.", "").replace("sz.", "")
                if len(code) == 6 and code.isdigit():
                    rows.append({"snapshot_date": date_str, "symbol": code, "index_source": "HS300"})

        if not zz500_data.empty:
            for _, row in zz500_data.iterrows():
                code = str(row.get("code", "")).replace("sh.", "").replace("sz.", "")
                if len(code) == 6 and code.isdigit():
                    rows.append({"snapshot_date": date_str, "symbol": code, "index_source": "ZZ500"})

        result = pd.DataFrame(rows)
        # Dedup: a stock can't be in both, but just in case
        result = result.drop_duplicates(subset=["snapshot_date", "symbol"])
        return result
    finally:
        bs.logout()


def main():
    logger.info("=" * 64)
    logger.info("Phase B Step 2: Historical CSI 800 Constituent Snapshots")
    logger.info("  Method: Baostock HS300 + ZZ500 union")
    logger.info("  Dates: %d semi-annual snapshots", len(SNAPSHOT_DATES))
    logger.info("=" * 64)

    all_snapshots = []
    for i, date_str in enumerate(SNAPSHOT_DATES):
        logger.info("[%2d/%2d] Fetching %s ...", i + 1, len(SNAPSHOT_DATES), date_str)
        try:
            snap = fetch_snapshot(date_str)
            n_csi800 = snap["symbol"].nunique()
            n_hs300 = (snap["index_source"] == "HS300").sum()
            n_zz500 = (snap["index_source"] == "ZZ500").sum()
            logger.info("        HS300=%d  ZZ500=%d  CSI800=%d", n_hs300, n_zz500, n_csi800)
            all_snapshots.append(snap)
        except Exception as e:
            logger.error("        FAILED: %s", e)
        time.sleep(0.3)  # be nice to baostock

    if not all_snapshots:
        logger.error("No snapshots fetched. Aborting.")
        return

    history = pd.concat(all_snapshots, ignore_index=True)
    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"])

    # Summary statistics
    total_unique = history["symbol"].nunique()
    avg_per_snapshot = history.groupby("snapshot_date")["symbol"].nunique().mean()

    logger.info("=" * 64)
    logger.info("CSI 800 Historical Constituents Summary")
    logger.info("  Total snapshots:       %d", len(all_snapshots))
    logger.info("  Unique stocks ever in: %d", total_unique)
    logger.info("  Average per snapshot:  %.0f", avg_per_snapshot)
    logger.info("=" * 64)

    # Count how many stocks joined/left over time
    first_date = history["snapshot_date"].min()
    last_date = history["snapshot_date"].max()
    first_symbols = set(history[history["snapshot_date"] == first_date]["symbol"])
    last_symbols = set(history[history["snapshot_date"] == last_date]["symbol"])
    added = last_symbols - first_symbols
    removed = first_symbols - last_symbols
    logger.info("  Stocks added since %s:   %d", str(first_date)[:10], len(added))
    logger.info("  Stocks removed since %s: %d", str(first_date)[:10], len(removed))
    logger.info("  Survivorship gap (w/o history): %d stocks", len(removed) + len(added))

    history.to_parquet(SNAPSHOT_PATH, index=False)
    logger.info("  Saved to: %s", SNAPSHOT_PATH)


if __name__ == "__main__":
    main()
