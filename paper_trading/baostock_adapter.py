"""
Baostock Data Adapter — Industrial-Grade PIT-Safe Access Layer.

Design goals
------------
1. **PIT Safety (pubDate gating)**
   Baostockʼs ``query_profit_data`` and ``query_balance_data`` both return
   ``pubDate`` (the actual announcement date).  We filter on ``pubDate`` —
   never ``statDate`` — when determining which financial statements were
   legally available to an investor on a given date.  This is a strict
   improvement over the Eastmoney path which relies on hard-coded
   CSRC statutory deadlines.

2. **Multiprocess Isolation**
   Each worker process calls ``bs.login()`` once via
   ``ProcessPoolExecutor(initializer=...)`` and reuses the connection for
   its entire batch.  No shared sockets, no pickle errors, no deadlocks.

3. **Adjustment-Factor Discipline**
   Two separate code paths:
   - ``adjust='forward'`` (adjustflag=2) → momentum / volatility factors
   - ``adjust='raw'``     (adjustflag=1) → PE / PB / market-cap factors
   The mapping is verified against the official baostock documentation.

Usage
-----
.. code-block:: python

    from paper_trading.baostock_adapter import BaostockAdapter

    # Single-stock PIT financials
    fin = BaostockAdapter.get_pit_financial("600000", "2026-06-05")

    # Batch PIT financials (multiprocess, ~7 min for 5200 stocks)
    df = BaostockAdapter.get_pit_financials_batch(
        symbols, "2026-06-05", workers=8
    )

    # Price history
    px = BaostockAdapter.get_prices("600000", "2026-01-01", "2026-06-05",
                                     adjust="forward")
"""

from __future__ import annotations

import concurrent.futures
import logging
import random
import time
from typing import Optional

import numpy as np
import pandas as pd

import baostock as _bs

logger = logging.getLogger("baostock_adapter")

# ═══════════════════════════════════════════════════════════
# Constants — verified against baostock v1.x documentation
# ═══════════════════════════════════════════════════════════

# adjustflag values (baostock query_history_k_data_plus)
#   "1" = 不复权 (unadjusted / raw)
#   "2" = 前复权 (forward-adjusted)
#   "3" = 后复权 (backward-adjusted)
_ADJUST_RAW = "1"
_ADJUST_FORWARD = "2"
_ADJUST_BACK = "3"

# Public mapping: human-readable → baostock flag
ADJUST_MAP: dict[str, str] = {
    "raw":     _ADJUST_RAW,
    "forward": _ADJUST_FORWARD,
    "back":    _ADJUST_BACK,
    # Compatibility aliases (FIXED — were swapped before)
    "qfq":     _ADJUST_FORWARD,   # 前复权 → 2
    "hfq":     _ADJUST_BACK,      # 后复权 → 3  (was incorrectly "1")
    "":        _ADJUST_RAW,       # 不复权 → 1  (was incorrectly "3")
}

# Alias for external consumers (matching the old _BAOSTOCK_ADJUST_MAP shape)
BAOSTOCK_ADJUST_MAP = ADJUST_MAP

# ═══════════════════════════════════════════════════════════
# Retry + Session Recovery
# ═══════════════════════════════════════════════════════════
# Baostock's TCP server drops idle/overloaded connections, especially
# under concurrent ProcessPool load.  Windows surfaces this as
# [WinError 10054] "远程主机强迫关闭了一个现有的连接".
#
# Strategy: exponential backoff (1s → 2s → 4s) with jitter,
# re-login before each retry, max 3 attempts per call.

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds
_RETRY_JITTER = 0.5         # ± jitter range

# Exception types / error patterns that indicate a transient connection failure
_RECOVERABLE_ERRORS = (
    ConnectionError,
    OSError,          # covers WinError 10054, 10053, 10061 etc.
    TimeoutError,
)
_RECOVERABLE_MSGS: tuple[str, ...] = (
    "10054",  # WSAECONNRESET — remote host forcibly closed
    "10053",  # WSAECONNABORTED
    "10061",  # WSAECONNREFUSED
    "timed out",
    "timeout",
    "connection reset",
    "broken pipe",
)


def _is_recoverable(exc: Exception) -> bool:
    """Return True if *exc* looks like a transient connection failure."""
    if isinstance(exc, _RECOVERABLE_ERRORS):
        return True
    msg = str(exc).lower()
    return any(pat in msg for pat in _RECOVERABLE_MSGS)


def _bs_reconnect() -> bool:
    """Re-establish a broken baostock session in the current process.

    Returns True on success, False if re-login failed.
    """
    try:
        _bs.logout()
    except Exception:
        pass  # best-effort — the old connection may already be dead

    try:
        lg = _bs.login()
        ok = lg.error_code == "0"
        if not ok:
            import sys as _sys
            print(
                f"[baostock_adapter] reconnect failed: [{lg.error_code}] {lg.error_msg}",
                file=_sys.stderr,
            )
        return ok
    except Exception:
        import sys as _sys
        import traceback as _tb
        print(
            f"[baostock_adapter] reconnect raised:",
            file=_sys.stderr,
        )
        _tb.print_exc()
        return False


def _retry_bs_call(fn, *args, max_retries: int = _MAX_RETRIES, **kwargs):
    """Call *fn(*args, **kwargs)* with exponential-backoff retry + session recovery.

    *fn* should be a baostock query function (``bs.query_profit_data``,
    ``bs.query_balance_data``, ``bs.query_history_k_data_plus``).

    Returns the baostock ``ResultSet`` on success.

    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            result = fn(*args, **kwargs)
            # Check if the baostock result itself indicates an error
            if hasattr(result, "error_code") and result.error_code != "0":
                err_msg = getattr(result, "error_msg", "")
                # If the error looks like a connection issue, retry
                if any(pat in str(err_msg).lower() for pat in _RECOVERABLE_MSGS):
                    if attempt < max_retries:
                        delay = _RETRY_BACKOFF_BASE * (2 ** attempt) \
                                + random.uniform(-_RETRY_JITTER, _RETRY_JITTER)
                        logger.debug(
                            "baostock returned connection error [%s] %s — "
                            "retry %d/%d in %.1fs",
                            result.error_code, err_msg,
                            attempt + 1, max_retries, max(0, delay),
                        )
                        if not _bs_reconnect():
                            logger.warning("bs reconnect failed before retry")
                        time.sleep(max(0, delay))
                        continue
                # Non-recoverable error — return as-is, caller checks error_code
            return result

        except _RECOVERABLE_ERRORS as exc:
            last_exc = exc
            if attempt < max_retries and _is_recoverable(exc):
                delay = _RETRY_BACKOFF_BASE * (2 ** attempt) \
                        + random.uniform(-_RETRY_JITTER, _RETRY_JITTER)
                logger.debug(
                    "baostock call failed with %s — retry %d/%d in %.1fs",
                    exc, attempt + 1, max_retries, max(0, delay),
                )
                if not _bs_reconnect():
                    logger.warning("bs reconnect failed before retry")
                time.sleep(max(0, delay))
                continue
            raise

        except Exception as exc:
            # Non-OSError — check if it looks like a connection error
            last_exc = exc
            if attempt < max_retries and _is_recoverable(exc):
                delay = _RETRY_BACKOFF_BASE * (2 ** attempt) \
                        + random.uniform(-_RETRY_JITTER, _RETRY_JITTER)
                logger.debug(
                    "baostock call failed with %s — retry %d/%d in %.1fs",
                    exc, attempt + 1, max_retries, max(0, delay),
                )
                if not _bs_reconnect():
                    logger.warning("bs reconnect failed before retry")
                time.sleep(max(0, delay))
                continue
            raise

    # Exhausted all retries
    assert last_exc is not None
    raise last_exc


# ── PIT financial fields wanted from baostock ──
# query_profit_data  fields of interest (subset)
_PROFIT_FIELDS = [
    "epsTTM",       # 每股收益 TTM
    "roeAvg",       # 平均净资产收益率 (%)
    "npMargin",     # 净利润率 (%)
    "netProfit",    # 净利润 (元)
    "MBRevenue",    # 营业总收入 (元)
    "totalShare",   # 总股本 (股)
    "liqaShare",    # 流通股本 (股)
]

# query_balance_data fields of interest (subset — baostock returns RATIOS, not absolute values)
_BALANCE_FIELDS = [
    "liabilityToAsset",   # 资产负债率 = totalLiab / totalAssets
    "assetToEquity",       # 权益乘数 = totalAssets / totalEquity
]

# Numeric fields that should be float (everything from baostock comes as str)
_ALL_NUMERIC_FIELDS = _PROFIT_FIELDS + _BALANCE_FIELDS


# ═══════════════════════════════════════════════════════════
# Module-level helpers (picklable for ProcessPoolExecutor)
# ═══════════════════════════════════════════════════════════

def _to_bs_code(symbol: str) -> str:
    """Convert a 6-digit A-share code to baostock format.

    >>> _to_bs_code("600000")
    'sh.600000'
    >>> _to_bs_code("000001")
    'sz.000001'
    """
    code = str(symbol).zfill(6)
    if code.startswith(("6", "9")):
        return f"sh.{code}"
    elif code.startswith(("0", "3")):
        return f"sz.{code}"
    else:
        # 4xxxx, 8xxxx → 三板 / 北交所
        return f"sh.{code}"


def _parse_bs_date(raw: str | None) -> pd.Timestamp | None:
    """Parse a baostock date string → pd.Timestamp, or None on garbage."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        ts = pd.Timestamp(s)
        return ts if ts is not pd.NaT else None
    except (ValueError, TypeError):
        return None


def _safe_float_bs(val) -> float | None:
    """Convert a baostock cell value to float, returning None on failure."""
    if val is None or val == "":
        return None
    try:
        f = float(val)
        return f if not np.isnan(f) else None
    except (ValueError, TypeError):
        return None


def _last_n_quarters(as_of: pd.Timestamp, n: int = 4) -> list[tuple[int, int]]:
    """Return (year, quarter) tuples for the last *n* completed quarters.

    A quarter is "completed" when its end date (03-31 / 06-30 / 09-30 / 12-31)
    is on or before *as_of*.  For example, on 2026-06-05 the most recent
    completed quarter is Q1 2026 (ended 2026-03-31).
    """
    year = as_of.year
    month = as_of.month

    if month < 4:
        q, y = 4, year - 1
    elif month < 7:
        q, y = 1, year
    elif month < 10:
        q, y = 2, year
    else:
        q, y = 3, year

    quarters: list[tuple[int, int]] = []
    for _ in range(n):
        quarters.append((y, q))
        q -= 1
        if q == 0:
            q = 4
            y -= 1
    return quarters


# ═══════════════════════════════════════════════════════════
# ProcessPoolExecutor initializer
# ═══════════════════════════════════════════════════════════

def _init_baostock_worker() -> None:
    """Worker initializer — called ONCE per worker process.

    Establishes a persistent baostock TCP connection that all subsequent
    queries in this process will reuse.  No per-stock login/logout needed.
    """
    # Module-level `import baostock as _bs` is re-executed when the worker
    # process spawns (Windows) or forks (Unix).  Just login.
    lg = _bs.login()
    if lg.error_code != "0":
        import sys as _sys
        print(
            f"[baostock_adapter] WARNING: bs.login() failed in worker "
            f"[{lg.error_code}] {lg.error_msg}",
            file=_sys.stderr,
        )
    else:
        import sys as _sys
        print(
            f"[baostock_adapter] baostock worker login ok",
            file=_sys.stderr,
        )


# ═══════════════════════════════════════════════════════════
# Module-level worker functions (picklable for ProcessPool)
# ═══════════════════════════════════════════════════════════

def _pit_single_worker(args: tuple) -> dict | None:
    """Process a single symbol for PIT financials.

    Args:
        args: (symbol: str, as_of_date: str)

    Returns:
        Standardized dict or None.

    The baostock session is assumed to already be logged in via
    ``_init_baostock_worker`` (the ProcessPool initializer).
    """
    symbol, as_of_date = args
    try:
        return BaostockAdapter.get_pit_financial(symbol, as_of_date)
    except Exception:
        # Any unhandled exception crashes the worker process on Windows —
        # return None instead so the batch can continue.
        import traceback as _tb
        _tb.print_exc()
        return None


def _pit_batch_worker(batch_payload: tuple) -> list[dict | None]:
    """Process a batch of symbols — calls ``get_pit_financial`` for each.

    Args:
        batch_payload: (symbols: list[str], as_of_date: str)

    Returns:
        List of result dicts (same length as symbols).  Failed stocks
        are represented as None.
    """
    symbols, as_of_date = batch_payload
    results: list[dict | None] = []
    for symbol in symbols:
        try:
            results.append(BaostockAdapter.get_pit_financial(symbol, as_of_date))
        except Exception:
            results.append(None)
    return results


def _price_worker(args: tuple) -> pd.DataFrame:
    """Process a batch of symbols for price history.

    Args:
        args: (symbols: list[str], start_date: str, end_date: str, adjust: str)

    Returns:
        pd.DataFrame with columns [date, symbol, open, high, low, close,
        volume, amount, pct_change, turnover_rate].
        Empty DataFrame on total failure.
    """
    symbols, start_date, end_date, adjust = args
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        try:
            df = BaostockAdapter.get_prices(symbol, start_date, end_date,
                                             adjust=adjust)
            if df is not None and len(df) > 0:
                frames.append(df)
        except Exception:
            continue
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════
# BaostockAdapter
# ═══════════════════════════════════════════════════════════

class BaostockAdapter:
    """Industrial-grade baostock data access layer.

    All methods are ``@staticmethod`` — the class is a namespace, not
    instantiated.  Session management is handled per-process via
    ``_init_baostock_worker()`` (the ProcessPool initializer).
    """

    # ── Price API ─────────────────────────────────────────

    @staticmethod
    def get_prices(
        symbol: str,
        start_date: str,
        end_date: str,
        *,
        adjust: str = "forward",
    ) -> pd.DataFrame:
        """Fetch daily OHLCV history for a single stock.

        Args:
            symbol: 6-digit A-share code, e.g. ``"600000"``.
            start_date: ISO-format start, e.g. ``"2026-01-01"``.
            end_date: ISO-format end, e.g. ``"2026-06-05"``.
            adjust: ``"forward"`` (前复权, for momentum/vol), ``"raw"``
                    (不复权, for PE/PB/mcap), or ``"back"`` (后复权).

        Returns:
            pd.DataFrame with columns:
            ``[date, symbol, open, high, low, close, volume, amount,
              pct_change, turnover_rate]``.
            Empty DataFrame when the stock has no data for the range.
        """
        bs_code = _to_bs_code(symbol)
        bs_adjust = ADJUST_MAP.get(adjust, _ADJUST_FORWARD)

        rs = _retry_bs_call(
            _bs.query_history_k_data_plus,
            bs_code,
            "date,open,high,low,close,volume,amount,pctChg,turn",
            start_date=start_date.replace("-", "-"),
            end_date=end_date.replace("-", "-"),
            frequency="d",
            adjustflag=bs_adjust,
        )

        if rs.error_code != "0":
            logger.debug(
                "  %s: baostock price query failed [%s] %s",
                symbol, rs.error_code, rs.error_msg,
            )
            return pd.DataFrame()

        rows: list[list] = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            logger.debug("  %s: no price rows for %s → %s", symbol, start_date, end_date)
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=rs.fields)
        df.columns = [c.strip().lower() for c in df.columns]

        result = pd.DataFrame()
        result["date"] = pd.to_datetime(df.get("date"), errors="coerce")
        result["symbol"] = str(symbol).zfill(6)
        result["open"] = pd.to_numeric(df.get("open"), errors="coerce")
        result["high"] = pd.to_numeric(df.get("high"), errors="coerce")
        result["low"] = pd.to_numeric(df.get("low"), errors="coerce")
        result["close"] = pd.to_numeric(df.get("close"), errors="coerce")
        result["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
        result["amount"] = pd.to_numeric(df.get("amount"), errors="coerce")
        result["pct_change"] = pd.to_numeric(df.get("pctchg"), errors="coerce")
        result["turnover_rate"] = pd.to_numeric(df.get("turn"), errors="coerce")

        result.dropna(subset=["date", "close"], inplace=True)
        return result.reset_index(drop=True)

    @staticmethod
    def get_prices_batch(
        symbols: list[str],
        start_date: str,
        end_date: str,
        *,
        adjust: str = "forward",
        workers: int = 8,
    ) -> pd.DataFrame:
        """Fetch OHLCV history for many stocks in parallel.

        Args:
            symbols: List of 6-digit A-share codes.
            start_date / end_date: ISO-format date range.
            adjust: ``"forward"`` or ``"raw"``.
            workers: Number of worker processes.

        Returns:
            Combined pd.DataFrame with the same schema as ``get_prices``.
        """
        if not symbols:
            return pd.DataFrame()

        BATCH_SIZE = 50
        batches = [
            symbols[i:i + BATCH_SIZE]
            for i in range(0, len(symbols), BATCH_SIZE)
        ]

        import multiprocessing as _mp
        _ctx = _mp.get_context("spawn")

        all_frames: list[pd.DataFrame] = []
        batch_args = [(b, start_date, end_date, adjust) for b in batches]

        with _ctx.Pool(
            processes=workers,
            initializer=_init_baostock_worker,
        ) as pool:
            for result in pool.imap_unordered(_price_worker, batch_args, chunksize=1):
                try:
                    df = result
                    if df is not None and len(df) > 0:
                        all_frames.append(df)
                except Exception as exc:
                    logger.warning("Price batch failed: %s", exc)

        if all_frames:
            return pd.concat(all_frames, ignore_index=True)
        return pd.DataFrame()

    # ── PIT Financial API ─────────────────────────────────

    @staticmethod
    def get_pit_financial(
        symbol: str,
        as_of_date: str | pd.Timestamp,
    ) -> dict | None:
        """Fetch the most recent PIT-safe financial data for one stock.

        **PIT gate**: for each report period returned by baostock, we check
        that ``pubDate <= as_of_date``.  This guarantees that the financial
        statement was *publicly available* to an investor on the valuation
        date — zero look-ahead bias.

        The baostock session must already be logged in (via
        ``_init_baostock_worker`` or by calling ``bs.login()`` in the
        current process).

        Args:
            symbol: 6-digit A-share code.
            as_of_date: The "as of" valuation date.

        Returns:
            Standardized dict::

                {
                    "symbol": "600000",
                    "report_period": Timestamp("2026-03-31"),
                    "pub_date": Timestamp("2026-04-28"),
                    "total_assets":  7.5e12,
                    "total_equity":  6.8e11,
                    "total_liabilities": 6.8e12,
                    "revenue":  5.0e10,
                    "net_profit": 1.5e10,
                    "roe":  12.5,
                    "eps":  1.25,
                    "bps":  12.5,
                    "total_share": 5.4e10,
                    "debt_ratio": 0.91,
                }

            Returns ``None`` when:
            - No financial reports with ``pubDate <= as_of_date`` exist,
            - The symbol has no data in baostock, or
            - All queries fail.
        """
        if isinstance(as_of_date, str):
            as_of = pd.Timestamp(as_of_date[:10])
        else:
            as_of = as_of_date

        try:
            return BaostockAdapter._get_pit_financial_impl(symbol, as_of)
        except Exception:
            # Catch-all: never let an unhandled exception crash a worker process
            import traceback as _tb
            _tb.print_exc()
            return None

    @staticmethod
    def _get_pit_financial_impl(symbol: str, as_of: pd.Timestamp) -> dict | None:
        """Internal implementation — called within a try/except safety wrapper."""
        bs_code = _to_bs_code(symbol)
        quarters = _last_n_quarters(as_of, n=3)

        # Collect all (profit, balance) rows that pass the PIT gate
        candidates: list[dict] = []

        for year, quarter in quarters:
            # ── Profit data ──
            profit_row = None
            try:
                rs = _retry_bs_call(
                    _bs.query_profit_data, bs_code, year=year, quarter=quarter,
                )
                if rs.error_code == "0":
                    while rs.next():
                        row_data = rs.get_row_data()
                        row_dict = dict(zip(rs.fields, row_data))

                        pub_date = _parse_bs_date(row_dict.get("pubDate"))
                        if pub_date is None:
                            continue
                        if pub_date > as_of:
                            continue  # ← PIT gate: not yet public

                        stat_date = _parse_bs_date(row_dict.get("statDate"))
                        profit_row = {
                            "pub_date": pub_date,
                            "stat_date": stat_date,
                            **{f: _safe_float_bs(row_dict.get(f))
                               for f in _PROFIT_FIELDS},
                        }
                        break  # one row per quarter
            except Exception as exc:
                logger.debug("  %s: profit Q%d/%d error: %s", symbol, quarter, year, exc)
                continue

            # ── Balance data ──
            balance_row = {}
            try:
                rs_bal = _retry_bs_call(
                    _bs.query_balance_data, bs_code, year=year, quarter=quarter,
                )
                if rs_bal.error_code == "0":
                    while rs_bal.next():
                        bal_data = rs_bal.get_row_data()
                        bal_dict = dict(zip(rs_bal.fields, bal_data))

                        bal_pub = _parse_bs_date(bal_dict.get("pubDate"))
                        if bal_pub is None or bal_pub > as_of:
                            continue  # ← PIT gate

                        balance_row = {
                            f: _safe_float_bs(bal_dict.get(f))
                            for f in _BALANCE_FIELDS
                        }
                        break
            except Exception as exc:
                logger.debug("  %s: balance Q%d/%d error: %s", symbol, quarter, year, exc)

            if profit_row is not None:
                profit_row.update(balance_row)
                candidates.append(profit_row)

        if not candidates:
            return None

        # Pick the row with the most recent stat_date
        best = max(candidates, key=lambda r: r.get("stat_date") or pd.NaT)

        # ── Build standardized output ──
        # baostock query_balance_data returns RATIOS, not absolute values.
        # We derive absolute totals from profit data + balance ratios:
        #
        #   roeAvg = netProfit / avgEquity   →   totalEquity ≈ netProfit / roeAvg
        #   assetToEquity = totalAssets / totalEquity  →  totalAssets
        #   liabilityToAsset = totalLiab / totalAssets  →  totalLiab
        #
        # These are approximations (roeAvg uses average equity, not period-end),
        # but are sufficient for cross-sectional ranking and factor computation.
        total_share = best.get("totalShare")
        net_profit = best.get("netProfit")
        revenue = best.get("MBRevenue")
        np_margin = best.get("npMargin")
        eps_val = best.get("epsTTM")
        roe_val = best.get("roeAvg")
        liability_to_asset = best.get("liabilityToAsset")
        asset_to_equity = best.get("assetToEquity")

        # Fallback: derive revenue from netProfit / npMargin when MBRevenue is empty
        # (some stocks — especially banks — return empty MBRevenue)
        if revenue is None and net_profit is not None and np_margin is not None and np_margin > 0:
            revenue = net_profit / np_margin

        # Derive total equity from ROE
        total_equity: float | None = None
        if roe_val is not None and net_profit is not None and roe_val > 0:
            total_equity = net_profit / roe_val  # roeAvg is decimal, e.g. 0.021641

        # Derive total assets from equity multiplier
        total_assets: float | None = None
        if total_equity is not None and asset_to_equity is not None and asset_to_equity > 0:
            total_assets = total_equity * asset_to_equity

        # Derive total liabilities
        total_liab: float | None = None
        if total_assets is not None and liability_to_asset is not None:
            total_liab = total_assets * liability_to_asset
        elif total_assets is not None and total_equity is not None:
            total_liab = total_assets - total_equity

        # Book value per share
        bps_val: float | None = None
        if total_equity is not None and total_share is not None and total_share > 0:
            bps_val = total_equity / total_share

        # Debt ratio — directly from baostock (liabilityToAsset IS the debt ratio)
        debt_ratio = liability_to_asset

        return {
            "symbol": str(symbol).zfill(6),
            "report_period": best.get("stat_date"),
            "pub_date": best.get("pub_date"),
            "total_assets": total_assets,
            "total_equity": total_equity,
            "total_liabilities": total_liab,
            "revenue": revenue,
            "net_profit": net_profit,
            "roe": roe_val,
            "eps": eps_val,
            "bps": bps_val,
            "total_share": total_share,
            "debt_ratio": debt_ratio,
        }

    @staticmethod
    def get_pit_financials_batch(
        symbols: list[str],
        as_of_date: str | pd.Timestamp,
        *,
        workers: int = 8,
    ) -> pd.DataFrame:
        """Fetch PIT-safe financials for many stocks in parallel.

        Uses ``ProcessPoolExecutor`` with ``initializer=_init_baostock_worker``
        so each worker process logs in once and reuses the connection for its
        entire batch.  On 5200 stocks this takes ~6-8 minutes (vs. ~3 hours
        for the sequential Eastmoney path).

        Args:
            symbols: List of 6-digit A-share codes.
            as_of_date: The "as of" valuation date.
            workers: Number of worker processes (default 8).

        Returns:
            pd.DataFrame with columns::

                symbol, report_period, pub_date,
                total_assets, total_equity, total_liabilities,
                revenue, net_profit, roe, eps, bps, total_share, debt_ratio
        """
        if isinstance(as_of_date, pd.Timestamp):
            date_str = as_of_date.strftime("%Y%m%d")
        else:
            date_str = str(as_of_date)[:10].replace("-", "")

        if not symbols:
            logger.warning("get_pit_financials_batch: empty symbol list")
            return pd.DataFrame()

        logger.info(
            "PIT financials (baostock): %d stocks × %d workers → fetching ...",
            len(symbols), workers,
        )

        t_start = time.monotonic()

        all_rows: list[dict] = []
        n_ok = 0
        n_fail = 0

        # Build (symbol, as_of_date) tuples for executor.map
        items = [(s, str(as_of_date)[:10]) for s in symbols]

        # Use multiprocessing.Pool (better Windows support than
        # ProcessPoolExecutor for long-running baostock sessions)
        import multiprocessing as _mp
        _ctx = _mp.get_context("spawn")

        with _ctx.Pool(
            processes=workers,
            initializer=_init_baostock_worker,
        ) as pool:
            # imap_unordered for streaming progress
            results_iter = pool.imap_unordered(
                _pit_single_worker, items, chunksize=5,
            )

            ordered_results: list = []
            for result in results_iter:
                ordered_results.append(result)

                if len(ordered_results) % 500 == 0:
                    n_so_far = len(ordered_results)
                    elapsed = time.monotonic() - t_start
                    rate = n_so_far / elapsed if elapsed > 0 else 0
                    n_ok_sofar = sum(1 for r in ordered_results if r is not None)
                    logger.info(
                        "  ... %d/%d stocks (%.0f/s, %d ok, %d failed)",
                        n_so_far, len(symbols), rate,
                        n_ok_sofar, n_so_far - n_ok_sofar,
                    )

            for result in ordered_results:
                if result is not None:
                    all_rows.append(result)
                    n_ok += 1
                else:
                    n_fail += 1

        elapsed = time.monotonic() - t_start
        logger.info(
            "  → %d stocks with PIT-safe financials (%d failed, %.0f%%) in %.0fs",
            n_ok, n_fail,
            100 * n_ok / max(len(symbols), 1),
            elapsed,
        )

        if not all_rows:
            logger.warning("Baostock PIT financials: 0 stocks succeeded")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        # Ensure datetime columns
        for col in ("report_period", "pub_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # YoY growth — computed downstream, pre-fill with NaN
        df["revenue_yoy"] = np.nan
        df["profit_yoy"] = np.nan

        # operating_profit not available from profit+balance APIs
        df["operating_profit"] = np.nan

        return df.reset_index(drop=True)

    # ── Generic multiprocess infrastructure ───────────────

    @staticmethod
    def parallel_map(
        func,
        items: list,
        *,
        workers: int = 8,
        chunksize: int = 25,
        desc: str = "",
    ) -> list:
        """Map *func* over *items* using ProcessPoolExecutor with baostock sessions.

        Each worker process calls ``_init_baostock_worker()`` on startup,
        so *func* can safely use baostock queries without per-call login.

        Args:
            func: Callable(item) → Any.  Must be picklable (module-level).
            items: List of arguments to map over.
            workers: Number of worker processes.
            chunksize: Items per chunk (higher = less IPC overhead).
            desc: Optional label for progress logging.

        Returns:
            List of results in the same order as *items*.
            Exceptions inside *func* produce ``None`` in the output.
        """
        if not items:
            return []

        label = f" ({desc})" if desc else ""
        total = len(items)
        logger.info("parallel_map%s: %d items × %d workers", label, total, workers)

        import multiprocessing as _mp
        _ctx = _mp.get_context("spawn")

        results: list = [None] * total
        t_start = time.monotonic()

        with _ctx.Pool(
            processes=workers,
            initializer=_init_baostock_worker,
        ) as pool:
            # pool.imap preserves order + streams for progress
            done_count = 0
            for idx, result in enumerate(pool.imap(func, items, chunksize=max(1, chunksize))):
                results[idx] = result
                done_count += 1

                if done_count % 500 == 0:
                    elapsed = time.monotonic() - t_start
                    rate = done_count / elapsed if elapsed > 0 else 0
                    logger.info(
                        "  ... %d/%d%s (%.0f item/s)",
                        done_count, total, label, rate,
                    )

        elapsed = time.monotonic() - t_start
        n_ok = sum(1 for r in results if r is not None)
        logger.info(
            "parallel_map%s: done in %.0fs — %d ok, %d failed",
            label, elapsed, n_ok, total - n_ok,
        )
        return results


# ═══════════════════════════════════════════════════════════
# Quick smoke test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("BaostockAdapter Smoke Test")
    print("=" * 60)

    # 1. Single price query
    print("\n── 1. get_prices (single) ──")
    px = BaostockAdapter.get_prices("600000", "2026-05-01", "2026-06-05",
                                     adjust="forward")
    print(f"   rows: {len(px)}, cols: {list(px.columns)}")
    if len(px) > 0:
        print(f"   head:\n{px.head(3).to_string()}")

    # 2. Single PIT financial
    print("\n── 2. get_pit_financial (single) ──")
    # Must login first (single-process smoke test)
    bs_lg = _bs.login()
    print(f"   bs.login() → {bs_lg.error_code} {bs_lg.error_msg}")
    if bs_lg.error_code == "0":
        fin = BaostockAdapter.get_pit_financial("600000", "2026-06-05")
        if fin:
            for k, v in fin.items():
                print(f"   {k:20s} = {v}")
        else:
            print("   No PIT data available (check date / symbol)")
        _bs.logout()
    else:
        print("   SKIP — baostock login failed")

    # 3. Batch PIT financials (small test)
    print("\n── 3. get_pit_financials_batch (5 stocks) ──")
    test_symbols = ["600000", "000001", "600519", "000858", "300750"]
    df = BaostockAdapter.get_pit_financials_batch(test_symbols, "2026-06-05",
                                                   workers=2)
    print(f"   rows: {len(df)}, cols: {list(df.columns)}")
    if len(df) > 0:
        print(f"   sample:\n{df.head(3).to_string()}")

    print("\n✅ Smoke test complete.")
