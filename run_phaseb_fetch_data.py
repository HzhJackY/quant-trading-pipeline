"""
Phase B Step 3-4: Incremental Data Fetching for Historical CSI 800 Stocks.

Fetches missing daily OHLCV and PIT financial data for all stocks that have
ever been in CSI 800 (1,476 total). Designed for interrupt-resume with
checkpoint persistence every 5 stocks.

Features:
  - Daily OHLCV: akshare stock_zh_a_hist (qfq adjusted), 2017-01-01 ~ 2026-06-19
  - Financial PIT:  baostock query_profit_data + query_balance_data WITH pubDate
  - Checkpoint:   .phaseb_fetch_state.json — resume after Ctrl+C
  - Rate limiting: 0.3s delay between stocks (daily), 0.5s (financial)
  - Retry:        3 attempts with linear backoff for network errors

Usage:
  python run_phaseb_fetch_data.py              # Fetch both daily + financial
  python run_phaseb_fetch_data.py --daily-only  # Only daily OHLCV
  python run_phaseb_fetch_data.py --fin-only    # Only financial PIT data
  python run_phaseb_fetch_data.py --reset       # Clear checkpoint, start fresh
"""

from __future__ import annotations

import json
import logging
import re
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
logger = logging.getLogger("phaseb_fetch")

# ── Configuration ──
DATA_RAW = Path("data/raw")
CSI800_HISTORY = Path("output/csi800_history.parquet")
CHECKPOINT_FILE = Path(".phaseb_fetch_state.json")

START_DATE = "20170101"
END_DATE = "20260619"  # extended to latest available

DAILY_BATCH_SIZE = 5
FIN_BATCH_SIZE = 5
RETRY_COUNT = 3
RETRY_BASE_DELAY = 5.0  # seconds


# ═══════════════════════════════════════════════════════════
# Inventory
# ═══════════════════════════════════════════════════════════

def get_historical_csi_symbols() -> set[str]:
    """All symbols that ever appeared in CSI 800 (2017-2026)."""
    hist = pd.read_parquet(CSI800_HISTORY)
    return set(str(s).zfill(6) for s in hist["symbol"].unique())


def get_cached_symbols(prefix: str) -> set[str]:
    """Scan data/raw/ for cached files matching prefix pattern."""
    cached = set()
    for f in DATA_RAW.glob(f"{prefix}_*.csv"):
        m = re.match(rf"{prefix}_(\d{{6}})_", f.name)
        if m:
            cached.add(m.group(1))
    return cached


# ═══════════════════════════════════════════════════════════
# Checkpoint management
# ═══════════════════════════════════════════════════════════

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Corrupt checkpoint file, starting fresh")
    return {
        "daily_completed": [],
        "daily_failed": {},
        "financial_completed": [],
        "financial_failed": {},
    }


def save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════
# Daily OHLCV fetching (akshare)
# ═══════════════════════════════════════════════════════════

def fetch_daily_one(symbol: str) -> bool:
    """
    Fetch daily OHLCV for a single stock via akshare.
    Returns True on success, False on failure.
    Caches to data/raw/daily_{symbol}_{start}_{end}_qfq.csv.
    """
    from data.fetcher import Fetcher

    f = Fetcher(cache_dir=str(DATA_RAW))
    for attempt in range(RETRY_COUNT + 1):
        try:
            df = f.get_daily(symbol, START_DATE, END_DATE, adjust="qfq")
            if df is not None and len(df) > 0:
                return True
        except Exception as e:
            if attempt < RETRY_COUNT:
                delay = RETRY_BASE_DELAY * (attempt + 1)
                logger.debug("  %s attempt %d failed: %s — retrying in %.0fs",
                             symbol, attempt + 1, e, delay)
                time.sleep(delay)
            else:
                logger.warning("  %s FAILED after %d attempts: %s", symbol, RETRY_COUNT + 1, e)
    return False


def fetch_daily_batch(symbols: list[str], state: dict) -> None:
    """Fetch daily data for a batch of symbols with checkpointing.
    Skips symbols that already have CSV files on disk OR are in the checkpoint."""

    # Build skip set: checkpoint completed + failed + disk cache
    skip = set(state["daily_completed"])
    # Also check disk for existing files (catches stocks fetched before checkpoint existed)
    disk_cached = get_cached_symbols("daily")
    skip |= disk_cached  # stocks with files already on disk
    skip |= set(state["daily_failed"].keys())

    remaining = [s for s in symbols if s not in skip]

    if not remaining:
        logger.info("Daily: all %d stocks already cached (disk + checkpoint)", len(symbols))
        return

    n_skipped = len(symbols) - len(remaining)
    logger.info("Daily: %d skipped (cached/complete), %d remaining to fetch (of %d total)",
                 n_skipped, len(remaining), len(symbols))

    completed = set(state["daily_completed"])
    failed = dict(state["daily_failed"])

    for i in range(0, len(remaining), DAILY_BATCH_SIZE):
        batch = remaining[i:i + DAILY_BATCH_SIZE]
        for j, sym in enumerate(batch):
            n = i + j + 1
            logger.info("  [%4d/%4d] Daily %s ...", n, len(remaining), sym)
            success = fetch_daily_one(sym)
            if success:
                completed.add(sym)
                state["daily_completed"] = list(completed)
            else:
                failed[sym] = f"Failed after {RETRY_COUNT + 1} attempts"
                state["daily_failed"] = failed

            if j < len(batch) - 1:
                time.sleep(0.2)  # rate limiting

        save_checkpoint(state)

    logger.info("Daily fetch complete: %d ok, %d failed", len(completed), len(failed))


# ═══════════════════════════════════════════════════════════
# Financial PIT data fetching (baostock WITH pubDate)
# ═══════════════════════════════════════════════════════════

def _parse_cn_num(val) -> float | None:
    """Parse akshare financial values: '145.23亿' → 1.4523e10, '3.03%' → 0.0303."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        pass
    s = str(val).strip()
    try:
        if "亿" in s:
            return float(re.sub(r"[^\d.\-]", "", s)) * 1e8
        if "万" in s:
            return float(re.sub(r"[^\d.\-]", "", s)) * 1e4
        if "%" in s:
            return float(re.sub(r"[^\d.\-]", "", s)) / 100.0
    except Exception:
        pass
    return None


# ── Regulatory PIT pub_date ──
def compute_pub_date(report_date: pd.Timestamp) -> pd.Timestamp:
    """Q1/Q2/Q3 +60d, FY +120d."""
    m, d = report_date.month, report_date.day
    if (m == 3 and d == 31) or (m == 6 and d == 30) or (m == 9 and d == 30):
        return report_date + pd.Timedelta(days=60)
    if m == 12 and d == 31:
        return report_date + pd.Timedelta(days=120)
    return report_date + pd.Timedelta(days=90)


def fetch_financial_one_akshare(symbol: str) -> bool:
    """
    ONE akshare call → all quarterly financial history.
    Adds regulatory PIT pub_date. Total share fetched separately.
    Caches to data/raw/financial_{symbol}_akshare.csv.
    """
    import akshare as ak

    sym = str(symbol).zfill(6)
    cache_path = DATA_RAW / f"financial_{sym}_akshare.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path, dtype={"symbol": str})
        if len(df) > 0 and "pub_date" in df.columns:
            return True

    for attempt in range(3):
        try:
            raw = ak.stock_financial_abstract_ths(symbol=sym, indicator="按报告期")
            if raw is None or raw.empty:
                return False
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                return False

    # Map columns
    col_map = {
        "报告期": "report_date", "净利润": "净利润", "营业总收入": "营业收入",
        "净资产收益率": "ROE", "基本每股收益": "每股收益",
        "每股净资产": "每股净资产", "销售净利率": "销售净利率",
        "资产负债率": "Debt_Ratio",
    }
    df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
    df["symbol"] = sym

    # Parse numbers (handle Chinese units like '145.23亿')
    for col in ["净利润", "营业收入", "ROE", "每股收益", "每股净资产", "销售净利率", "Debt_Ratio"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _parse_cn_num(v))

    # report_date already mapped above; use the new column name
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df["pub_date"] = df["report_date"].apply(lambda x: compute_pub_date(x) if pd.notna(x) else pd.NaT)
    df = df.dropna(subset=["report_date", "pub_date"])
    df = df.sort_values("report_date").reset_index(drop=True)
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return len(df) > 0


def fetch_total_share_one(symbol: str) -> tuple[float | None, str | None]:
    """
    ONE baostock call → latest total_share.
    Returns (total_share, report_date) or (None, None).
    """
    import baostock as bs

    sym = str(symbol).zfill(6)
    prefix = "sh." if sym.startswith(("6", "5")) else "sz."

    try:
        bs.login()
        for year in [2026, 2025, 2024]:
            for q in [1, 4, 3, 2]:
                rs = bs.query_profit_data(code=prefix + sym, year=year, quarter=q)
                if rs.error_code != "0":
                    continue
                data = rs.get_data()
                if data.empty:
                    continue
                ts = pd.to_numeric(data.iloc[0].get("totalShare", np.nan), errors="coerce")
                if pd.notna(ts) and ts > 0:
                    bs.logout()
                    return float(ts), str(data.iloc[0].get("statDate", ""))
        bs.logout()
    except Exception:
        try:
            bs.logout()
        except Exception:
            pass
    return None, None


def fetch_financial_batch(symbols: list[str], state: dict) -> None:
    """Fetch financial data (akshare) + total_share (baostock) for symbols."""
    completed = set(state.get("financial_completed", []))
    failed = dict(state.get("financial_failed", {}))
    remaining = [s for s in symbols if s not in completed and s not in failed]

    if not remaining:
        logger.info("Financial + TotalShare: all %d stocks already processed", len(symbols))
        return

    n_skipped = len(symbols) - len(remaining)
    logger.info("Financial+Share: %d skipped, %d remaining (of %d total)",
                 n_skipped, len(remaining), len(symbols))

    # Load total_share cache
    ts_cache = {}
    ts_path = DATA_RAW / "total_share_cache.csv"
    if ts_path.exists():
        ts_df = pd.read_csv(ts_path, dtype={"symbol": str})
        for _, row in ts_df.iterrows():
            ts_cache[str(row["symbol"]).zfill(6)] = {
                "total_share": row["total_share"], "report_date": row.get("report_date", "")
            }

    for i in range(0, len(remaining), FIN_BATCH_SIZE):
        batch = remaining[i:i + FIN_BATCH_SIZE]
        for j, sym in enumerate(batch):
            n = i + j + 1
            logger.info("  [%4d/%4d] Fin+Share %s ...", n, len(remaining), sym)

            # Financial (akshare, fast)
            fin_ok = fetch_financial_one_akshare(sym)
            # Total share (baostock, one query)
            if sym not in ts_cache:
                ts, ts_date = fetch_total_share_one(sym)
                if ts is not None:
                    ts_cache[sym] = {"total_share": ts, "report_date": ts_date or ""}

            if fin_ok:
                completed.add(sym)
                state["financial_completed"] = list(completed)
            else:
                failed[sym] = "akshare financial fetch failed"
                state["financial_failed"] = failed

            if j < len(batch) - 1:
                time.sleep(0.15)

        save_checkpoint(state)
        # Persist total_share cache
        if ts_cache:
            pd.DataFrame([
                {"symbol": k, "total_share": v["total_share"], "report_date": v["report_date"]}
                for k, v in ts_cache.items()
            ]).to_csv(ts_path, index=False, encoding="utf-8-sig")


# ═══════════════════════════════════════════════════════════
# Quality Report (Phase 5)
# ═══════════════════════════════════════════════════════════

def generate_quality_report(target_symbols: set[str]) -> None:
    """Generate data quality report for all target stocks."""
    daily_cached = get_cached_symbols("daily")
    fin_cached = get_cached_symbols("financial")

    report_lines = [
        "=" * 64,
        "Phase B Data Quality Report",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 64,
        "",
        f"Target stocks (historical CSI 800): {len(target_symbols)}",
        f"Daily data cached: {len(daily_cached & target_symbols)} / {len(target_symbols)}",
        f"Financial data cached: {len(fin_cached & target_symbols)} / {len(target_symbols)}",
        "",
    ]

    # Check daily quality for a sample
    suspicious_daily = []
    for sym in sorted(daily_cached & target_symbols):
        csv_files = list(DATA_RAW.glob(f"daily_{sym}_*.csv"))
        if not csv_files:
            suspicious_daily.append((sym, "no file"))
            continue
        try:
            df = pd.read_csv(csv_files[0], parse_dates=["日期"], encoding="utf-8-sig")
            if len(df) < 100:
                suspicious_daily.append((sym, f"only {len(df)} rows"))
            elif df["收盘"].isna().mean() > 0.10:
                suspicious_daily.append((sym, f"{df['收盘'].isna().mean():.0%} NaN close"))
        except Exception as e:
            suspicious_daily.append((sym, str(e)))

    if suspicious_daily:
        report_lines.append(f"--- Flagged Daily Data ({len(suspicious_daily)} stocks) ---")
        for sym, reason in suspicious_daily[:20]:
            report_lines.append(f"  [WARN] {sym}: {reason}")
        report_lines.append("")

    # Check financial quality
    suspicious_fin = []
    for sym in sorted(fin_cached & target_symbols):
        csv_files = list(DATA_RAW.glob(f"financial_{sym}_*.csv"))
        if not csv_files:
            suspicious_fin.append((sym, "no file"))
            continue
        try:
            df = pd.read_csv(csv_files[0], encoding="utf-8-sig")
            if "pub_date" not in df.columns or df["pub_date"].isna().all():
                suspicious_fin.append((sym, "missing pub_date"))
            elif len(df) < 8:
                suspicious_fin.append((sym, f"only {len(df)} reports"))
        except Exception as e:
            suspicious_fin.append((sym, str(e)))

    if suspicious_fin:
        report_lines.append(f"--- Flagged Financial Data ({len(suspicious_fin)} stocks) ---")
        for sym, reason in suspicious_fin[:20]:
            report_lines.append(f"  [WARN] {sym}: {reason}")
        report_lines.append("")

    report = "\n".join(report_lines)
    report_path = Path("output") / "phaseb_quality_report.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Quality report saved to %s", report_path)
    print(report)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase B: Fetch historical CSI 800 data")
    parser.add_argument("--daily-only", action="store_true")
    parser.add_argument("--fin-only", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Clear checkpoint, start fresh")
    parser.add_argument("--report-only", action="store_true", help="Only generate quality report")
    args = parser.parse_args()

    if args.reset and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint reset")

    target_symbols = get_historical_csi_symbols()
    state = load_checkpoint()

    daily_cached = get_cached_symbols("daily")
    fin_cached = get_cached_symbols("financial")

    # Count what's truly missing
    daily_completed_set = set(state["daily_completed"]) | daily_cached
    daily_failed_set = set(state["daily_failed"].keys())
    daily_done = len(daily_completed_set & target_symbols)
    daily_fail_n = len(daily_failed_set & target_symbols)
    daily_missing = target_symbols - daily_completed_set - daily_failed_set

    fin_completed_set = set(state["financial_completed"]) | fin_cached
    fin_failed_set = set(state["financial_failed"].keys())
    fin_done = len(fin_completed_set & target_symbols)
    fin_fail_n = len(fin_failed_set & target_symbols)
    fin_missing = target_symbols - fin_completed_set - fin_failed_set

    logger.info("=" * 64)
    logger.info("Phase B: Data Fetching for Historical CSI 800")
    logger.info("  Target stocks: %d (ever in CSI 800, 2017-2026)", len(target_symbols))
    logger.info("  Date range: %s ~ %s", START_DATE, END_DATE)
    logger.info("  ---")
    logger.info("  Daily:   %d ok | %d fail | %d remaining",
                 daily_done, daily_fail_n, len(daily_missing))
    logger.info("  Financial: %d ok | %d fail | %d remaining",
                 fin_done, fin_fail_n, len(fin_missing))
    logger.info("=" * 64)

    if args.report_only:
        generate_quality_report(target_symbols)
        return

    target_list = sorted(target_symbols)

    if not args.fin_only:
        fetch_daily_batch(target_list, state)

    if not args.daily_only:
        fetch_financial_batch(target_list, state)

    generate_quality_report(target_symbols)


if __name__ == "__main__":
    main()
