"""
init_local_state.py — Cold-start initializer for paper trading state database.

Orchestrates three independent tasks:
  1. Schema initialization — creates all tables and indices via StateSchema DDL
  2. Market cache cold start — fetches past N trading days of OHLCV for all A-shares
     and batch-writes to `market_cache`
  3. Industry classification — fetches SW (申万) industry for all stocks and writes to
     `industry_cache`

Usage:
    python init_local_state.py                          # Full cold start (60-day lookback)
    python init_local_state.py --lookback-days 30       # Shallower history
    python init_local_state.py --throttle 0.1           # Faster (default 0.15s)
    python init_local_state.py --skip-market            # Industry only
    python init_local_state.py --skip-industry          # OHLCV only
    python init_local_state.py --max-stocks 100         # Test with subset
    python init_local_state.py --dry-run                # Validate without side effects

Design notes:
  - Uses fetch_single_stock_history() (one API call per stock covering the full window)
    rather than fetch_daily_market_data() (one call per stock per day) to minimise
    round-trips.  At default throttle and 5 000 stocks this takes roughly 12–15 minutes.
  - Batch-writes every 500 stocks to keep memory bounded and let the auto-pruner
    operate incrementally.
  - All exceptions are caught per-stock — a failing symbol never tears down the run.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════
# Proxy bypass — MUST execute before akshare / requests is
# first imported.  Eastmoney endpoints are directly reachable
# from within China; routing through a corporate proxy breaks
# every connection.
# ═══════════════════════════════════════════════════════════
import os as _os
for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
             "ALL_PROXY", "all_proxy", "REQUESTS_CA_BUNDLE"):
    _os.environ.pop(_var, None)
_os.environ["NO_PROXY"] = "*"
_os.environ["no_proxy"] = "*"
_os.environ["REQUESTS_TRUST_ENV"] = "0"

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Ensure project root is on sys.path ──────────────────────
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from paper_trading.state_manager import StateManager, StateSchema
from paper_trading.data_ingestion import (
    fetch_all_a_share_codes,
    fetch_single_stock_history,
    fetch_industry_classification,
)

# ═══════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════

logger = logging.getLogger("init_local_state")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

DEFAULT_DB_DIR = Path("output/paper_trading_db")
DEFAULT_LOOKBACK_DAYS = 60
DEFAULT_THROTTLE = 0.15   # seconds between per-stock history calls
BATCH_WRITE_SIZE = 500     # stocks per batch-write to control memory
PROGRESS_INTERVAL = 500    # stocks between progress logs

# OHLCV columns expected by StateManager.append_market_data()
OHLCV_COLS = ["open", "high", "low", "close", "volume", "amount",
              "pct_change", "turnover_rate"]


# ═══════════════════════════════════════════════════════════
# Anti-blocking: browser-grade headers for requests/akshare
# ═══════════════════════════════════════════════════════════

_UNIVERSE_CACHE_PATH = Path("output/universe_cache.parquet")

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Referer": "https://data.eastmoney.com/",
    "Origin": "https://data.eastmoney.com",
}


def _inject_browser_headers():
    """
    Inject browser-grade headers at EVERY layer of the requests stack.

    Why three patches are necessary:
      - ``requests.utils.default_user_agent()`` is called by
        ``PreparedRequest.prepare_headers()`` to fill a missing User-Agent
        AFTER Session.request has already been called -> patching
        Session.request alone is NOT enough.
      - ``requests.api.request()`` is the bottleneck for module-level
        calls like ``requests.get()`` / ``requests.post()``.
      - ``requests.Session.request()`` catches the session path.

    Eastmoney drops connections carrying ``python-requests/x.x.x`` UA
    on sight; this triple-patch guarantees it never appears on the wire.
    """
    import requests

    ua = _BROWSER_HEADERS["User-Agent"]

    # ── Level 1: Replace the default UA at the source ──
    requests.utils.default_user_agent = lambda name=None: ua
    # Also blanket-replace the module-level default_headers
    requests.models.default_headers = lambda: {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "*/*",
        "Connection": "keep-alive",
    }

    # ── Level 2: Patch api.request ──
    _orig_api_request = requests.api.request

    def _api_request(method, url, **kwargs):
        _merge_browser_headers(kwargs)
        return _orig_api_request(method, url, **kwargs)

    requests.api.request = _api_request

    # ── Level 3: Patch Session.request ──
    _orig_session_request = requests.Session.request

    def _session_request(self, method, url, **kwargs):
        _merge_browser_headers(kwargs)
        return _orig_session_request(self, method, url, **kwargs)

    requests.Session.request = _session_request

    logger.info("[OK] Browser headers injected (3-layer: default_ua + api + session)")


def _merge_browser_headers(kwargs: dict):
    """Inject _BROWSER_HEADERS into a requests kwargs dict (mutates in-place)."""
    headers = kwargs.get("headers", None)
    if headers is None:
        headers = {}
        kwargs["headers"] = headers
    if isinstance(headers, dict):
        for k, v in _BROWSER_HEADERS.items():
            headers.setdefault(k, v)


# ═══════════════════════════════════════════════════════════
# Direct A-share universe fetch (avoids Eastmoney TLS blocking).
# Strategy: try akshare's ``stock_info_a_code_name()`` first
# (pulls from exchange directly, not Eastmoney), with a
# paginated Eastmoney API as backup.
# ═══════════════════════════════════════════════════════════

def _fetch_universe_exchange():
    """
    Fetch A-share universe via ``ak.stock_info_a_code_name()``.

    This uses the Shenzhen/Shanghai exchange source rather than
    Eastmoney's spot API — far less prone to anti-bot blocking.

    Returns a DataFrame with columns: symbol, name, board
    """
    import akshare as ak

    logger.info("  Fetching via exchange source (akshare stock_info_a_code_name) ...")
    df = ak.stock_info_a_code_name()

    # Columns: code, name
    df = df.rename(columns={"code": "symbol"})
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)

    # Board classification
    def _board(code: str) -> str:
        c = str(code).zfill(6)
        if c.startswith("688"):
            return "科创板"
        elif c.startswith(("300", "301")):
            return "创业板"
        elif c.startswith(("000", "001", "002", "003")):
            return "深市主板"
        elif c.startswith(("600", "601", "603", "605")):
            return "沪市主板"
        return "其他"

    df["board"] = df["symbol"].apply(_board)
    df = df[df["symbol"].str.match(r"^(00|30|60|68)\d{4}$", na=False)]
    logger.info("  -> %d A-share stocks (exchange source)", len(df))
    return df[["symbol", "name", "board"]].reset_index(drop=True)


def _fetch_universe_with_retry(max_attempts: int = 5, cooldown: float = 30.0):
    """
    Fetch A-share universe with manual retry + long cooldown + jitter.

    Eastmoney's anti-scraping drops connections aggressively.  Key design:
      - A pre-delay (5 s) before the FIRST call avoids hitting the server
        right at script start when it looks most like a burst.
      - A long FIXED cooldown (30 s default) between attempts resets the
        rate-limit window far better than exponential backoff from a hot
        start (the built-in ak_retry does 2s->4s->8s, which keeps the
        connection pattern within Eastmoney's short memory).
      - Small random jitter (±20 %) on the cooldown so retry timing
        doesn't look deterministic / bot-like.
    """
    import random

    # Pre-delay: don't rush the first request
    pre_delay = 5.0 + random.uniform(0, 3.0)
    logger.info("  Pre-delay %.1f s before first request ...", pre_delay)
    time.sleep(pre_delay)

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("  Attempt %d/%d ...", attempt, max_attempts)

            # ── Primary: exchange source (most reliable) ──
            df = _fetch_universe_exchange()
            n = len(df) if df is not None else 0
            if n >= 4000:
                return df

            # ── Fallback 1: direct Eastmoney API ──
            logger.warning("  Exchange source returned only %d — trying Eastmoney direct ...", n)
            df = _fetch_universe_direct_via_eastmoney()
            n = len(df) if df is not None else 0
            if n >= 4000:
                return df

            # ── Fallback 2: vanilla akshare ──
            logger.warning("  Direct API returned only %d — trying akshare ...", n)
            df = fetch_all_a_share_codes()
            n = len(df) if df is not None else 0
            logger.info("  -> %d stocks via akshare", n)
            return df

        except Exception as exc:
            last_error = exc
            msg = str(exc)[:120]
            if attempt < max_attempts:
                jitter = cooldown * (0.8 + random.uniform(0, 0.4))
                logger.warning(
                    "  Attempt %d failed (%s) — cooling down %.0f s ...",
                    attempt, msg, jitter,
                )
                time.sleep(jitter)
            else:
                logger.error("  All %d attempts exhausted (%s)", max_attempts, msg)

    raise last_error  # type: ignore[misc]


def _fetch_universe_direct_via_eastmoney():
    """
    Fallback: paginate Eastmoney's JSONP API with browser headers.

    Only used when the exchange source fails — may still be blocked
    if the IP is currently rate-limited.
    """
    import json
    import requests as _req

    logger.info("  Trying direct Eastmoney API (paginated) ...")
    all_items = []
    for page in range(1, 60):
        resp = _req.get(
            "https://push2.eastmoney.com/api/qt/clist/get",
            params={
                "pn": str(page), "pz": "100", "po": "1",
                "np": "1", "fltt": "2", "fid": "f3",
                "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
                "fields": "f12,f14",
            },
            headers=_BROWSER_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        raw = resp.text
        if "(" in raw and raw.endswith(")"):
            raw = raw[raw.index("(") + 1:-1]
        data = json.loads(raw)
        total = data.get("data", {}).get("total", 0)
        items = data.get("data", {}).get("diff", [])
        all_items.extend(items)
        if page == 1:
            logger.info("  Total available: %d -> %d pages", total,
                        (total // 100) + 1)
        if page * 100 >= total:
            break
        if page % 15 == 0:
            time.sleep(2.0)  # generous gap every 15 pages

    if not all_items:
        raise RuntimeError("Eastmoney returned 0 records")

    rows = []
    for item in all_items:
        code = str(item.get("f12", "")).zfill(6)
        if len(code) != 6:
            continue
        if code.startswith("688"):
            board = "科创板"
        elif code.startswith(("300", "301")):
            board = "创业板"
        elif code.startswith(("000", "001", "002", "003")):
            board = "深市主板"
        elif code.startswith(("600", "601", "603", "605")):
            board = "沪市主板"
        else:
            board = "其他"
        rows.append({"symbol": code, "name": str(item.get("f14", "")), "board": board})

    df = pd.DataFrame(rows)
    df = df[df["symbol"].str.match(r"^(00|30|60|68)\d{4}$", na=False)]
    logger.info("  -> %d A-share stocks via Eastmoney direct", len(df))
    return df.reset_index(drop=True)


def _load_or_fetch_universe(
    force_refresh: bool = False,
    max_retries: int = 5,
    cooldown: float = 30.0,
) -> pd.DataFrame:
    """
    Return A-share universe.  On first success the result is cached to
    `output/universe_cache.parquet` — subsequent cold starts read the
    cache directly, avoiding the Eastmoney spot endpoint entirely.

    Pass ``--refresh-universe`` to force a live fetch.
    """
    if not force_refresh and _UNIVERSE_CACHE_PATH.exists():
        try:
            df = pd.read_parquet(_UNIVERSE_CACHE_PATH)
            if len(df) > 4000:
                logger.info(
                    "[OK] Universe loaded from cache (%d stocks) — "
                    "use --refresh-universe to force live fetch",
                    len(df),
                )
                return df
            else:
                logger.warning("Cached universe too small (%d) — re-fetching", len(df))
        except Exception as exc:
            logger.warning("Universe cache corrupted (%s) — re-fetching", exc)

    df = _fetch_universe_with_retry(max_attempts=max_retries, cooldown=cooldown)

    # Persist cache
    try:
        _UNIVERSE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_UNIVERSE_CACHE_PATH, index=False)
        logger.info("[OK] Universe cached -> %s", _UNIVERSE_CACHE_PATH)
    except Exception as exc:
        logger.warning("Could not write universe cache: %s", exc)

    return df


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _compute_lookback_window(lookback_days: int) -> tuple[str, str]:
    """
    Return (start_date, end_date) as 'YYYYMMDD' strings covering a
    conservative calendar window that is guaranteed to contain at least
    `lookback_days` trading days (≈ 1.4× calendar window).
    """
    today = date.today()
    end_date = today.strftime("%Y%m%d")
    # Use ~1.5× the target to safely capture enough trading days
    # (weekends + holidays ≈ 30 % of calendar days)
    calendar_days = max(int(lookback_days * 1.5), lookback_days + 30)
    start_date = (today - timedelta(days=calendar_days)).strftime("%Y%m%d")
    return start_date, end_date


def _row_to_tuple(row, symbol: str) -> tuple:
    """Convert a single OHLCV row dict to the tuple expected by market_cache INSERT."""
    trade_date = row.get("date")
    if isinstance(trade_date, pd.Timestamp):
        trade_date = trade_date.strftime("%Y-%m-%d")
    elif isinstance(trade_date, (datetime, date)):
        trade_date = trade_date.strftime("%Y-%m-%d")
    else:
        trade_date = str(trade_date)[:10]

    def _f(key: str) -> Optional[float]:
        val = row.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    return (
        trade_date,
        str(symbol).zfill(6),
        _f("open"),
        _f("high"),
        _f("low"),
        _f("close"),
        _f("volume"),
        _f("amount"),
        _f("pct_change"),
        _f("turnover_rate"),
    )


# ═══════════════════════════════════════════════════════════
# Step 3: Market Cache Cold Start
# ═══════════════════════════════════════════════════════════

def cold_start_market_cache(
    state: StateManager,
    symbols: list[str],
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    throttle_seconds: float = DEFAULT_THROTTLE,
) -> dict:
    """
    Fetch full OHLCV history for each symbol and batch-write into market_cache.

    Uses `fetch_single_stock_history()` (one API round-trip per stock covering
    the full window) rather than `fetch_daily_market_data()` (one round-trip
    per stock per day).  This keeps total API calls at ~N rather than ~N×D.

    Returns summary dict with keys:
        n_stocks_total, n_failed, n_market_rows, n_market_dates, elapsed_seconds
    """
    start_date, end_date = _compute_lookback_window(lookback_days)

    logger.info("Cold-start OHLCV window: %s -> %s  (target >=%d trading days)",
                start_date, end_date, lookback_days)
    logger.info("Fetching full history for %d stocks (throttle %.2f s, batch-size %d) ...",
                len(symbols), throttle_seconds, BATCH_WRITE_SIZE)
    logger.info("")

    n_total = len(symbols)
    n_failed = 0
    n_rows_total = 0
    pending_rows: list[tuple] = []
    t_start = time.monotonic()

    for i, symbol in enumerate(symbols):
        symbol_ok = False
        try:
            df = fetch_single_stock_history(
                symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            if df is not None and len(df) > 0:
                # Drop rows with NaT dates or missing close
                df = df.dropna(subset=["date", "close"]).copy()
                if len(df) > 0:
                    for _, row in df.iterrows():
                        pending_rows.append(_row_to_tuple(row, symbol))
                    symbol_ok = True
        except Exception as exc:
            n_failed += 1
            if n_failed <= 5:
                logger.warning("  %s -> fetch failed (%s)", symbol, str(exc)[:80])

        if not symbol_ok:
            n_failed += 1

        # ── Batch-write every BATCH_WRITE_SIZE stocks ──
        if len(pending_rows) >= BATCH_WRITE_SIZE * 50 or \
           (i + 1) % BATCH_WRITE_SIZE == 0:
            if pending_rows:
                _batch_insert(state, pending_rows)
                n_rows_total += len(pending_rows)
                pending_rows.clear()

        # ── Progress report ──
        if (i + 1) % PROGRESS_INTERVAL == 0:
            elapsed = time.monotonic() - t_start
            rate_stocks = (i + 1) / elapsed
            rate_calls = n_rows_total / elapsed if n_rows_total > 0 else 0
            eta_sec = (n_total - i - 1) / rate_stocks if rate_stocks > 0 else 0
            logger.info(
                "  %d / %d  (%.0f %%)  |  %d rows written  |  "
                "%.1f stocks/min  |  ETA %.0f s",
                i + 1, n_total,
                100 * (i + 1) / n_total,
                n_rows_total,
                rate_stocks * 60,
                eta_sec,
            )

        # ── Throttle ──
        if throttle_seconds > 0 and i < n_total - 1:
            time.sleep(throttle_seconds)

    # Final flush
    if pending_rows:
        _batch_insert(state, pending_rows)
        n_rows_total += len(pending_rows)
        pending_rows.clear()

    elapsed_total = time.monotonic() - t_start
    stats = state.stats()

    return {
        "n_stocks_total": n_total,
        "n_failed": n_failed,
        "n_market_rows": stats["market_rows"],
        "n_market_dates": stats["market_dates"],
        "elapsed_seconds": round(elapsed_total, 1),
    }


def _batch_insert(state: StateManager, rows: list[tuple]):
    """
    Directly INSERT batches of market_cache rows inside a locked connection.
    Bypasses append_market_data() to avoid repeated distinct-date counts
    and premature pruning during the bulk load.
    """
    sql = (
        "INSERT OR REPLACE INTO market_cache "
        "(trade_date, symbol, open, high, low, close, volume, amount, "
        "pct_change, turnover_rate) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    with state._get_conn() as conn:
        conn.executemany(sql, rows)
        conn.commit()

    # Prune once after the batch
    n_dates = _count_distinct_dates(state)
    if n_dates > state.MAX_CACHE_DAYS_MARGIN:
        _prune_dates(state)


def _count_distinct_dates(state: StateManager) -> int:
    with state._get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(DISTINCT trade_date) FROM market_cache"
        ).fetchone()
    return row[0] if row else 0


def _prune_dates(state: StateManager):
    """Delete rows older than the MAX_CACHE_DAYS-th most recent trading date."""
    with state._get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT trade_date FROM market_cache "
            "ORDER BY trade_date DESC "
            "LIMIT 1 OFFSET ?",
            (state.MAX_CACHE_DAYS - 1,),
        ).fetchall()
        if not rows:
            return
        cutoff = rows[0][0]
        cursor = conn.execute(
            "DELETE FROM market_cache WHERE trade_date < ?", (cutoff,)
        )
        if cursor.rowcount > 0:
            logger.info("  Pruned %d rows (trade_date < %s)", cursor.rowcount, cutoff)
        conn.commit()


# ═══════════════════════════════════════════════════════════
# Step 4: Industry Cache Cold Start
# ═══════════════════════════════════════════════════════════

def cold_start_industry_cache(state: StateManager) -> dict:
    """
    Fetch SW industry classification via AkShare and persist to industry_cache.

    Returns summary dict with keys: n_stocks, n_industries.
    """
    logger.info("Fetching SW (申万) industry classification for all A-shares ...")
    try:
        industry_df = fetch_industry_classification()
    except Exception as exc:
        logger.error("fetch_industry_classification() raised: %s", exc)
        raise

    if industry_df is None or len(industry_df) == 0:
        logger.warning("Industry classification returned empty — "
                       "industry_cache will remain unpopulated.")
        return {"n_stocks": 0, "n_industries": 0}

    state.update_industry_cache(industry_df)

    return {
        "n_stocks": len(industry_df),
        "n_industries": int(industry_df["industry_name"].nunique()),
    }


# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════

def print_summary(
    market_stats: dict,
    industry_stats: dict,
    db_path: Path,
    elapsed_total: float,
):
    """Print a human-readable initialization summary."""
    print()
    print("=" * 66)
    print("  本地状态库初始化 — 完成")
    print("  LOCAL STATE INITIALIZATION COMPLETE")
    print("=" * 66)
    print(f"  Database        : {db_path.resolve()}")
    print(f"  Total elapsed   : {elapsed_total:.0f} s")
    print()

    # ── Market Cache ──
    print(f"  {'─' * 42}")
    print(f"  [Market] Market Cache (market_cache)")
    print(f"  {'-' * 42}")
    print(f"    Stocks attempted : {market_stats.get('n_stocks_total', 0):,}")
    print(f"    Fetch failures   : {market_stats.get('n_failed', 0)}")
    print(f"    Rows written     : {market_stats.get('n_market_rows', 0):,}")
    print(f"    Unique dates     : {market_stats.get('n_market_dates', 0)}")
    if market_stats.get("n_market_dates", 0) >= DEFAULT_LOOKBACK_DAYS:
        print(f"    [OK] 行情数据充足 — Vol_20D / Mom_1M 等时序因子可正常计算")
    elif market_stats.get("n_market_dates", 0) > 0:
        print(f"    [WARN] 日期数不足 {DEFAULT_LOOKBACK_DAYS} — 部分时序因子可能触发 fallback")
    else:
        print(f"    [FAIL] 行情缓存为空 — 请重新运行")
    print()

    # ── Industry Cache ──
    print(f"  {'─' * 42}")
    print(f"  [Industry] Industry Cache (industry_cache)")
    print(f"  {'─' * 42}")
    n_ind_stocks = industry_stats.get("n_stocks", 0)
    n_industries = industry_stats.get("n_industries", 0)
    print(f"    Stocks classified: {n_ind_stocks:,}")
    print(f"    Unique industries : {n_industries}")
    if n_ind_stocks > 0:
        print(f"    [OK] 行业分类就绪 — 因子计算可启用行业中性化")
    else:
        print(f"    ! 行业缓存为空 — 因子将使用全局均值中性化")
    print()

    print(f"  {'─' * 42}")
    print(f"  数据库初始化成功，行情数据填充完毕，准备就绪 [OK]")
    print("=" * 66)
    print()


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cold-start initializer for paper trading state database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python init_local_state.py                              # Full cold start
  python init_local_state.py --lookback-days 30            # 30-day window
  python init_local_state.py --throttle 0.1                # Faster fetch
  python init_local_state.py --skip-market                 # Industry only
  python init_local_state.py --skip-industry               # OHLCV only
  python init_local_state.py --max-stocks 100              # Test run
  python init_local_state.py --dry-run                     # No side effects
        """,
    )
    parser.add_argument(
        "--db-dir", type=str, default=str(DEFAULT_DB_DIR),
        help="Path to SQLite state database directory.",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
        help="Target number of trading days to capture (default: 60).",
    )
    parser.add_argument(
        "--throttle", type=float, default=DEFAULT_THROTTLE,
        help="Seconds between per-stock API calls (default: 0.15).",
    )
    parser.add_argument(
        "--max-stocks", type=int, default=None,
        help="Limit to first N stocks (for testing).",
    )
    parser.add_argument(
        "--skip-market", action="store_true",
        help="Skip OHLCV cold start (schema + industry only).",
    )
    parser.add_argument(
        "--skip-industry", action="store_true",
        help="Skip industry classification fetch.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration without writing to disk.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--refresh-universe", action="store_true",
        help="Force live fetch of stock universe (ignore local cache).",
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("data_ingestion").setLevel(logging.DEBUG)

    db_dir = Path(args.db_dir)

    # ── Banner ──
    print()
    print("=" * 66)
    print("  INIT LOCAL STATE — 本地状态库冷启动")
    print("=" * 66)
    print(f"  DB directory      : {db_dir.resolve()}")
    print(f"  Lookback (target) : >={args.lookback_days} trading days")
    print(f"  Throttle          : {args.throttle:.2f} s / stock")
    print(f"  Skip market       : {args.skip_market}")
    print(f"  Skip industry     : {args.skip_industry}")
    print(f"  Max stocks        : {args.max_stocks or 'ALL'}")
    print(f"  Refresh universe  : {args.refresh_universe}")
    print(f"  Dry run           : {args.dry_run}")
    print("=" * 66)
    print()

    t_total_start = time.monotonic()

    # ── Anti-blocking: inject browser headers ──
    _inject_browser_headers()

    # ═══════════════════════════════════════════════════════
    # Step 1: Schema Initialization
    # ═══════════════════════════════════════════════════════
    if args.dry_run:
        logger.info("[DRY RUN] Would create tables: %s",
                     [d.split()[4] for d in StateSchema.ALL_DDL])
    else:
        state = StateManager(db_dir)
        state.init()
        logger.info("[OK] Schema initialized — %d tables, %d indices",
                     len(StateSchema.ALL_DDL), len(StateSchema.INDICES))

    # ═══════════════════════════════════════════════════════
    # Step 2: Stock Universe (with local cache + retry)
    # ═══════════════════════════════════════════════════════
    logger.info("Loading A-share universe ...")
    try:
        universe = _load_or_fetch_universe(
            force_refresh=args.refresh_universe,
            max_retries=5,
            cooldown=30.0,
        )
    except Exception as exc:
        logger.error("Cannot load stock universe after all retries: %s", exc)
        logger.error(
            "If Eastmoney is actively blocking, wait 10-15 minutes and re-run. "
            "The universe cache at %s only needs to succeed once.",
            _UNIVERSE_CACHE_PATH,
        )
        sys.exit(1)

    symbols_all = universe["symbol"].tolist()
    if args.max_stocks:
        symbols_all = symbols_all[:args.max_stocks]
    logger.info("  -> %d A-share symbols%s",
                 len(symbols_all),
                 f" (capped at {args.max_stocks})" if args.max_stocks else "")

    # ═══════════════════════════════════════════════════════
    # Step 3: Market Cache Cold Start
    # ═══════════════════════════════════════════════════════
    market_stats: dict = {
        "n_stocks_total": 0,
        "n_failed": 0,
        "n_market_rows": 0,
        "n_market_dates": 0,
        "elapsed_seconds": 0,
    }

    if args.skip_market:
        logger.info("[SKIP] Market cache cold start (--skip-market)")
    elif args.dry_run:
        est_calls = len(symbols_all)
        est_minutes = (est_calls * args.throttle) / 60
        logger.info(
            "[DRY RUN] Would call fetch_single_stock_history() %d times "
            "-> ≈%.1f minutes at %.2f s throttle",
            est_calls, est_minutes, args.throttle,
        )
        market_stats["n_stocks_total"] = len(symbols_all)
    else:
        logger.info("-" * 50)
        logger.info("Step 3: Cold-start market cache (this will take a while)")
        logger.info("-" * 50)
        try:
            market_stats = cold_start_market_cache(
                state,
                symbols=symbols_all,
                lookback_days=args.lookback_days,
                throttle_seconds=args.throttle,
            )
            logger.info("[OK] Market cache cold start complete — %d rows, %d dates",
                         market_stats["n_market_rows"],
                         market_stats["n_market_dates"])
        except KeyboardInterrupt:
            logger.warning("Market cache cold start interrupted — "
                           "partial data has been saved.")
            # Don't re-raise — continue to industry step so partial DB is usable
        except Exception as exc:
            logger.error("Market cache cold start failed: %s", exc, exc_info=args.verbose)
            logger.warning("Continuing with industry step — re-run to retry market cache.")

    # ═══════════════════════════════════════════════════════
    # Step 4: Industry Cache Cold Start
    # ═══════════════════════════════════════════════════════
    industry_stats: dict = {"n_stocks": 0, "n_industries": 0}

    if args.skip_industry:
        logger.info("[SKIP] Industry cache cold start (--skip-industry)")
    elif args.dry_run:
        logger.info("[DRY RUN] Would call fetch_industry_classification() "
                    "-> update_industry_cache()")
        industry_stats["n_stocks"] = len(symbols_all)
        industry_stats["n_industries"] = 28  # typical SW level-1 count
    else:
        logger.info("-" * 50)
        logger.info("Step 4: Cold-start industry cache")
        logger.info("-" * 50)
        try:
            industry_stats = cold_start_industry_cache(state)
            logger.info("[OK] Industry cache ready — %d stocks, %d industries",
                         industry_stats["n_stocks"],
                         industry_stats["n_industries"])
        except Exception as exc:
            logger.error("Industry cache cold start failed: %s", exc,
                         exc_info=args.verbose)

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    elapsed_total = time.monotonic() - t_total_start

    if not args.dry_run:
        print_summary(market_stats, industry_stats, db_dir / "state.db", elapsed_total)
    else:
        print()
        print("=" * 66)
        print("  [DRY RUN] 配置验证通过，未执行任何写操作")
        print("  [DRY RUN] Configuration validated — no side effects.")
        print("=" * 66)
        print(f"  Estimated API calls     : {len(symbols_all):,}")
        print(f"  Estimated wall time     : "
              f"{(len(symbols_all) * args.throttle) / 60:.1f} minutes")
        print(f"  Tables to create        : "
              f"{', '.join(d.split()[4] for d in StateSchema.ALL_DDL)}")
        print("=" * 66)


if __name__ == "__main__":
    main()
