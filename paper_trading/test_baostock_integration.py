"""
Integration test for BaostockAdapter — L2 (batch) and L3 (cache round-trip).

Usage:
    python paper_trading/test_baostock_integration.py
"""

import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test")

# ── L2: ProcessPool batch test ──
def test_l2_batch():
    print("=" * 60)
    print("L2: ProcessPool batch — 50 stocks x 4 workers")
    print("=" * 60)

    from paper_trading.baostock_adapter import BaostockAdapter

    # Diverse sectors: banks, liquor, new energy, pharma, brokers, autos, etc.
    symbols = [
        "600000", "000001", "600519", "000858", "300750", "600276", "000002",
        "601012", "600030", "000725", "002415", "300059", "601318", "600809",
        "000568", "600036", "601398", "600900", "000333", "002714", "601166",
        "600887", "000651", "002475", "601899", "000063", "300124", "600585",
        "000538", "002594", "601888", "300015", "600690", "000792", "002230",
        "300274", "601088", "000100", "600104", "002352", "300760", "600048",
        "001979", "300122", "601857", "000625", "002142", "300408",
    ]
    # Deduplicate
    symbols = list(dict.fromkeys(symbols))
    n = len(symbols)
    print(f"Testing {n} stocks...")

    t0 = time.monotonic()
    df = BaostockAdapter.get_pit_financials_batch(symbols, "2026-06-05", workers=4)
    elapsed = time.monotonic() - t0

    if df is None or len(df) == 0:
        print("FAIL: 0 stocks returned")
        return False

    # Quality checks
    n_rev = df["revenue"].notna().sum()
    n_eps = df["eps"].notna().sum()
    n_bps = df["bps"].notna().sum()
    n_debt = df["debt_ratio"].notna().sum()
    n_pub = df["pub_date"].notna().sum()

    print(f"\nResults: {len(df)}/{n} stocks in {elapsed:.0f}s ({n/elapsed:.1f} stocks/s)")
    print(f"  revenue present:   {n_rev}/{len(df)}")
    print(f"  EPS present:       {n_eps}/{len(df)}")
    print(f"  BPS present:       {n_bps}/{len(df)}")
    print(f"  debt_ratio present:{n_debt}/{len(df)}")
    print(f"  pub_date present:  {n_pub}/{len(df)}")

    # PIT safety check: no pub_date should be after 2026-06-05
    from datetime import datetime
    late_pubs = df[df["pub_date"] > pd.Timestamp("2026-06-05")]
    if len(late_pubs) > 0:
        print(f"\n  PIT VIOLATION: {len(late_pubs)} stocks with pub_date > 2026-06-05!")
        print(late_pubs[["symbol", "pub_date"]].to_string())
        return False
    print("  PIT gate: OK (no pub_date after 2026-06-05)")

    # Show sector diversity
    print(f"\nSample (first 10):")
    cols = ["symbol", "eps", "roe", "bps", "debt_ratio", "pub_date"]
    available = [c for c in cols if c in df.columns]
    print(df[available].head(10).to_string())

    return True


# ── L3: Cache round-trip ──
def test_l3_cache():
    print("\n" + "=" * 60)
    print("L3: Cache round-trip (parquet)")
    print("=" * 60)

    from paper_trading.data_ingestion import fetch_and_align_financials
    from pathlib import Path
    import pandas as pd

    cache_path = Path("output/paper_trading_db/pit_financials_20260605.parquet")

    # Remove existing cache to start clean
    if cache_path.exists():
        cache_path.unlink()
        print(f"  Removed existing cache: {cache_path.name}")

    # Step A: Fetch with use_baostock=True (populates cache)
    print("  Step A: Fetching via Baostock (max_stocks=15)...")
    t0 = time.monotonic()
    df_a = fetch_and_align_financials(
        "2026-06-05",
        max_stocks=15,
        force_refresh=True,
        use_baostock=True,
    )
    elapsed_a = time.monotonic() - t0
    print(f"    -> {len(df_a)} stocks in {elapsed_a:.0f}s")

    # Verify cache file exists
    if cache_path.exists():
        size_kb = cache_path.stat().st_size / 1024
        print(f"    Cache written: {cache_path.name} ({size_kb:.0f} KB)")
    else:
        print("    WARNING: Cache file was NOT written!")
        return False

    # Step B: Fetch again WITHOUT force_refresh — should hit cache
    print("  Step B: Fetching again (should hit cache)...")
    t0 = time.monotonic()
    df_b = fetch_and_align_financials(
        "2026-06-05",
        max_stocks=15,
        force_refresh=False,
        use_baostock=True,
    )
    elapsed_b = time.monotonic() - t0
    print(f"    -> {len(df_b)} stocks in {elapsed_b:.3f}s")

    if elapsed_b > 1.0:
        print("    WARNING: Cache hit should be near-instant (< 1s)")
        return False
    print("    Cache hit: OK (near-instant)")

    # Step C: Verify data consistency between A and B
    if len(df_a) != len(df_b):
        print(f"    WARNING: Cache round-trip mismatch ({len(df_a)} vs {len(df_b)})")
        return False

    # Compare key columns (handle NaN-aware comparison)
    import pandas as _pd
    key_cols = ["symbol", "report_period", "eps", "roe", "bps"]
    available_cols = [c for c in key_cols if c in df_a.columns and c in df_b.columns]
    a_sorted = df_a[available_cols].sort_values("symbol").reset_index(drop=True)
    b_sorted = df_b[available_cols].sort_values("symbol").reset_index(drop=True)
    if a_sorted.equals(b_sorted):
        print("    Data consistency: OK (cache matches source)")
    else:
        # Show diff
        diff_mask = (a_sorted != b_sorted) & ~(a_sorted.isna() & b_sorted.isna())
        n_diff = diff_mask.any(axis=1).sum()
        print(f"    WARNING: Cache data differs from source ({n_diff} rows differ)")
        return False

    # Cleanup
    cache_path.unlink()
    print(f"    Cleanup: removed {cache_path.name}")
    return True


if __name__ == "__main__":
    import pandas as pd

    all_ok = True

    try:
        if not test_l2_batch():
            all_ok = False
    except Exception as e:
        print(f"L2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False

    try:
        if not test_l3_cache():
            all_ok = False
    except Exception as e:
        print(f"L3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — see above")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)
