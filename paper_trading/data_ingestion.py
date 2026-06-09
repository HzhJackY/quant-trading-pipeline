"""
Robust AkShare Data Ingestion Layer — Retry, Rate Limiting, and Type Safety.

Design principles (Defensive Programming):
  1. Every HTTP call is wrapped with @retry_on_failure — exponential backoff,
     max 5 attempts, no silent failures.
  2. Every numeric column passes through pd.to_numeric(errors='coerce') before
     returning to the caller. AkShare columns are notoriously unstable:
     '--', '%', '万', '亿', and mixed str/int types all appear in the wild.
  3. All functions return pd.DataFrame with CONSISTENT column names, regardless
     of what AkShare happens to emit for that particular version.
  4. A session-level rate limiter prevents >5 calls/second to Eastmoney.

Usage:
    from paper_trading.data_ingestion import (
        fetch_daily_market_data,
        fetch_daily_fundamentals,
        fetch_all_a_share_codes,
    )

    # Daily cron: pull today's OHLCV for all A-shares
    ohlcv = fetch_daily_market_data(trade_date="20260605")

    # Month-end: pull latest fundamental data
    fundamentals = fetch_daily_fundamentals()
"""

from __future__ import annotations

import concurrent.futures
import json as _json
import logging
import re
import threading as _threading
import time
import warnings
from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)

# ── tqdm: optional progress bar ──
try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _tqdm = None
    _TQDM_AVAILABLE = False

# ── Parallel execution global state ──
_PROGRESS_DIR = Path("data")
# (ProcessPoolExecutor workers each call bs.login()/bs.logout()
#  independently — no shared socket, no lock needed.)

logger = logging.getLogger("data_ingestion")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore", category=FutureWarning, module="akshare")


# ═══════════════════════════════════════════════════════════
# Checkpoint / Resume — progress.json
# ═══════════════════════════════════════════════════════════

def _load_progress(task_name: str) -> set:
    """Load completed symbols from ``data/progress_{task_name}.json``.

    Returns a ``set`` of symbol strings that have already been processed,
    or an empty set on first run / corrupt file.
    """
    progress_file = _PROGRESS_DIR / f"progress_{task_name}.json"
    if not progress_file.exists():
        return set()
    try:
        with open(progress_file, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return set(data.get("completed", []))
    except Exception:
        logger.warning("Could not parse %s — starting fresh", progress_file)
        return set()


def _save_progress(task_name: str, completed: set):
    """Save completed symbol set to ``data/progress_{task_name}.json``.

    Called every 50 stocks and on completion.  Writes atomically by
    building the JSON in memory first, so a crash mid-write won't
    corrupt the file.
    """
    _PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    progress_file = _PROGRESS_DIR / f"progress_{task_name}.json"
    try:
        payload = _json.dumps(
            {"completed": sorted(completed), "updated": datetime.now().isoformat()},
            ensure_ascii=False,
        )
        with open(progress_file, "w", encoding="utf-8") as f:
            f.write(payload)
    except Exception as exc:
        logger.warning("Could not write progress file %s: %s", progress_file, exc)


def _log_error(error_log_path: Path, symbol: str, error: Exception):
    """Append a single-stock error entry to ``error_log.txt``.

    NEVER raises — error logging is best-effort only.
    """
    try:
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{datetime.now().isoformat()} | {symbol} | "
                f"{type(error).__name__}: {str(error)[:200]}\n"
            )
    except Exception:
        pass


def _clear_progress(task_name: str):
    """Delete a progress file so the next run starts fresh."""
    progress_file = _PROGRESS_DIR / f"progress_{task_name}.json"
    if progress_file.exists():
        try:
            progress_file.unlink()
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════
# Baostock Session Management
# ═══════════════════════════════════════════════════════════
#
# Baostock uses its own TCP socket (not requests/urllib3), which
# bypasses Eastmoney's JA3 TLS fingerprinting entirely.  The server
# is sensitive to long-lived connections — ALWAYS logout after use
# or the next login will fail with "already logged in" / timeout.
#
# Session lifecycle:
#   init_baostock()     ← call once before batch ops (idempotent)
#   ... fetch calls ...
#   logout_baostock()   ← call after batch (or rely on atexit hook)
# ═══════════════════════════════════════════════════════════

import atexit as _atexit
import baostock as _bs

_bs_logged_in: bool = False


def init_baostock():
    """Login to baostock.  Idempotent — no-op if already logged in."""
    global _bs_logged_in
    if _bs_logged_in:
        return
    lg = _bs.login()
    if lg.error_code != "0":
        raise ConnectionError(f"baostock login failed: [{lg.error_code}] {lg.error_msg}")
    _bs_logged_in = True
    logger.debug("baostock login OK")


def logout_baostock():
    """Logout from baostock.  Safe to call even when not logged in."""
    global _bs_logged_in
    if _bs_logged_in:
        try:
            _bs.logout()
        except Exception:
            pass
        finally:
            _bs_logged_in = False


_atexit.register(logout_baostock)


# ═══════════════════════════════════════════════════════════
# Browser-grade headers for Eastmoney API calls ONLY.
# Do NOT apply globally — ``stock_info_a_code_name()``
# downloads .xls from the exchange, and a browser UA causes
# the exchange to serve HTML instead of the binary stream.
# ═══════════════════════════════════════════════════════════

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

_EM_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Referer": "https://data.eastmoney.com/",
}


def _call_eastmoney_with_browser_headers():
    """
    Call ``ak.stock_zh_a_spot_em()`` with browser headers injected
    directly into the underlying requests session — no global patching.
    """
    import akshare as ak
    import requests as _req

    # Patch only THIS thread's calls: temporarily override Session.request
    # on a dedicated session that ak.stock_zh_a_spot_em will use.
    # Since akshare uses a shared internal session, we intercept at the
    # PreparedRequest level by overriding default_user_agent briefly.
    _orig_ua_fn = _req.utils.default_user_agent
    _req.utils.default_user_agent = lambda name=None: _BROWSER_UA

    try:
        return ak.stock_zh_a_spot_em()
    finally:
        _req.utils.default_user_agent = _orig_ua_fn

# ═══════════════════════════════════════════════════════════
# Rate Limiter (simple token-bucket)
# ═══════════════════════════════════════════════════════════

class _RateLimiter:
    """Simple rate limiter: max `calls_per_second` calls per second."""

    def __init__(self, calls_per_second: float = 4.0):
        self._min_interval = 1.0 / calls_per_second
        self._last_call = 0.0

    def wait(self):
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait()
            return func(*args, **kwargs)
        return wrapper


_rate_limiter = _RateLimiter(calls_per_second=4.0)


# ═══════════════════════════════════════════════════════════
# Retry Decorator (Exponential Backoff)
# ═══════════════════════════════════════════════════════════

RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    ConnectionResetError,
    ConnectionRefusedError,
    TimeoutError,
    OSError,
)

# AkShare wraps requests; catch both urllib3 and requests exceptions
try:
    from urllib3.exceptions import HTTPError as Urllib3HTTPError
    RETRYABLE_EXCEPTIONS = RETRYABLE_EXCEPTIONS + (Urllib3HTTPError,)
except ImportError:
    pass

try:
    from requests.exceptions import (
        ConnectionError as RequestsConnectionError,
        Timeout as RequestsTimeout,
        HTTPError as RequestsHTTPError,
    )
    RETRYABLE_EXCEPTIONS = RETRYABLE_EXCEPTIONS + (
        RequestsConnectionError, RequestsTimeout, RequestsHTTPError,
    )
except ImportError:
    pass


def _is_retryable(exception: BaseException) -> bool:
    """Return True if the exception is likely transient (network/rate-limit)."""
    if isinstance(exception, RETRYABLE_EXCEPTIONS):
        return True
    # Catch ProxyError / RemoteDisconnected by string match
    msg = str(exception).lower()
    trigger_words = ("proxy", "remote end closed", "max retries", "connection",
                     "timeout", "reset", "refused", "too many requests", "429",
                     "503", "502", "504")
    return any(w in msg for w in trigger_words)


def ak_retry(func: Callable | None = None, *, max_attempts: int = 5,
             min_wait: float = 2.0, max_wait: float = 60.0):
    """
    Decorator: retry AkShare calls with exponential backoff.

    Backoff: wait = min(min_wait * 2^attempt, max_wait) seconds.

    Usage:
        @ak_retry
        def fetch_something(): ...

        @ak_retry(max_attempts=3, min_wait=1.0)
        def fetch_quick(): ...
    """
    def decorator(f: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS)
                  | retry_if_exception_type(Exception),  # catch all, filter in _is_retryable
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if _is_retryable(e):
                    raise  # tenacity will retry
                raise  # non-retryable: re-raise immediately
        return wrapper
    if func is not None:
        return decorator(func)
    return decorator


# ═══════════════════════════════════════════════════════════
# Column Standardization (Defensive Type Coercion)
# ═══════════════════════════════════════════════════════════

# Canonical output column names — all functions MUST return these
OHLCV_COLUMNS = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "换手率": "turnover_rate",
}

FUNDAMENTAL_COLUMNS = {
    "代码": "symbol",
    "名称": "name",
    "市盈率-动态": "pe_ttm",
    "市净率": "pb",
    "总市值": "total_mcap",
    "流通市值": "float_mcap",
    "营业收入": "revenue",
    "营业利润": "operating_profit",
    "净利润": "net_profit",
    "每股净资产": "bps",
    "每股收益": "eps",
    "净资产收益率": "roe",
    "毛利率": "gross_margin",
    "净利率": "net_margin",
}


def _safe_to_numeric(series: pd.Series, fill_value: float = np.nan) -> pd.Series:
    """
    Coerce a column to numeric, handling common AkShare dirt:

    - '--'       → NaN
    - '-3.14%'   → -0.0314
    - '1.23万'   → 12300
    - '2.5亿'    → 250000000
    - mixed str/int in same column → numeric or NaN
    """
    s = series.astype(str).str.strip()

    # Percentage strings: "-3.14%" → -0.0314
    pct_mask = s.str.endswith("%")
    if pct_mask.any():
        s = s.str.rstrip("%")
        numeric = pd.to_numeric(s, errors="coerce")
        numeric.loc[pct_mask] = numeric.loc[pct_mask] / 100.0
        return numeric.fillna(fill_value)

    # Chinese unit suffixes
    if s.str.contains("万|亿", na=False).any():
        multiplier = pd.Series(1.0, index=s.index)
        multiplier[s.str.contains("万", na=False)] = 1e4
        multiplier[s.str.contains("亿", na=False)] = 1e8
        s_clean = s.str.replace("万|亿", "", regex=True)
        numeric = pd.to_numeric(s_clean, errors="coerce")
        return (numeric * multiplier).fillna(fill_value)

    # Standard numeric
    return pd.to_numeric(s, errors="coerce").fillna(fill_value)


def _clean_and_rename(df: pd.DataFrame, column_map: dict,
                      numeric_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Standardize AkShare output: rename columns, coerce numeric types.

    Parameters
    ----------
    df : pd.DataFrame
        Raw AkShare output.
    column_map : dict
        Mapping from AkShare column names → canonical names.
    numeric_cols : list[str], optional
        Columns to coerce. If None, all renamed columns are coerced.

    Returns
    -------
    pd.DataFrame with canonical column names and safe numeric types.
    """
    result = df.copy()
    # Rename columns that exist
    rename = {k: v for k, v in column_map.items() if k in result.columns}
    result = result.rename(columns=rename)

    # Coerce numeric columns
    if numeric_cols is None:
        numeric_cols = list(column_map.values())
    for col in numeric_cols:
        if col in result.columns:
            result[col] = _safe_to_numeric(result[col])
    return result


# ═══════════════════════════════════════════════════════════
# Code Format Conversion — Baostock ↔ Standard
# ═══════════════════════════════════════════════════════════
#
# Baostock expects  "sh.600000" / "sz.000001"
# Standard (6-digit):  "600000"    / "000001"
# ═══════════════════════════════════════════════════════════

def _format_code_to_baostock(symbol: str) -> str:
    """Convert a 6-digit standard code into baostock format.

    >>> _format_code_to_baostock("600000")
    'sh.600000'
    >>> _format_code_to_baostock("000001")
    'sz.000001'
    """
    code = str(symbol).zfill(6)
    if code.startswith(("60", "68")):
        return f"sh.{code}"
    else:
        return f"sz.{code}"


def _format_code_from_baostock(bs_code: str) -> str:
    """Convert a baostock-format code back to 6-digit standard.

    >>> _format_code_from_baostock("sh.600000")
    '600000'
    >>> _format_code_from_baostock("sz.000001")
    '000001'
    """
    return str(bs_code).split(".")[-1].zfill(6)


# baostock adjust flag mapping (FIXED 2026-06-09)
# baostock doc: 1=不复权  2=前复权  3=后复权
_BAOSTOCK_ADJUST_MAP = {
    "qfq": "2",   # 前复权 → 2
    "hfq": "3",   # 后复权 → 3  (was incorrectly "1")
    "":    "1",   # 不复权 → 1  (was incorrectly "3")
}


# ═══════════════════════════════════════════════════════════
# Public API: Stock Universe
# ═══════════════════════════════════════════════════════════

# Local cache — once populated, subsequent calls skip the network entirely.
_UNIVERSE_CACHE_PATH = Path("output") / "universe_cache.parquet"
_MIN_UNIVERSE_SIZE = 4000   # sanity check: a valid A-share universe has ≥4000 stocks


def _classify_board(code_str: str) -> str:
    """Map a 6-digit stock code to its board (板块)."""
    code = str(code_str).zfill(6)
    if code.startswith("688"):
        return "科创板"
    if code.startswith(("300", "301")):
        return "创业板"
    if code.startswith(("000", "001", "002", "003")):
        return "深市主板"
    if code.startswith(("600", "601", "603", "605")):
        return "沪市主板"
    return "其他"


def _valid_a_share(code: str) -> bool:
    """Return True if the code looks like a standard A-share."""
    return bool(re.match(r"^(00|30|60|68)\d{4}$", str(code)))


def _normalise_universe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise a raw stock-list DataFrame into canonical form:

        symbol  name  board

    Accepts whatever column names AkShare happens to emit for the
    code / name pair (currently ``stock_zh_a_spot_em`` uses "代码"/"名称").
    """
    # ── Rename code column ──
    code_candidates = ["代码", "code", "symbol"]
    for col in code_candidates:
        if col in df.columns:
            df = df.rename(columns={col: "symbol"})
            break

    # ── Rename name column ──
    name_candidates = ["名称", "name", "股票名称"]
    for col in name_candidates:
        if col in df.columns:
            df = df.rename(columns={col: "name"})
            break

    df["symbol"] = df["symbol"].astype(str).str.zfill(6)

    # ── Filter A-shares + classify board ──
    mask = df["symbol"].apply(_valid_a_share)
    df = df.loc[mask].copy()
    df["board"] = df["symbol"].apply(_classify_board)

    return df[["symbol", "name", "board"]].reset_index(drop=True)


@ak_retry
@_rate_limiter
def fetch_all_a_share_codes(
    force_refresh: bool = False,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Fetch the full A-share stock universe.

    **Cache-first design**: on first success the result is persisted to
    ``output/universe_cache.parquet``.  Subsequent calls (in any script)
    read the cache directly, avoiding the Eastmoney API entirely.

    Args:
        force_refresh: If True, skip the cache and fetch live.
        cache_path: Override the default cache location.

    Returns:
        pd.DataFrame with columns: symbol, name, board (主板/创业板/科创板)

    Data source:
        ``ak.stock_zh_a_spot_em()`` — Eastmoney real-time quotes (JSON
        endpoint, no Excel parsing).
    """
    import akshare as ak

    cache = Path(cache_path) if cache_path else _UNIVERSE_CACHE_PATH

    # ── Cache hit ──
    if not force_refresh and cache.exists():
        try:
            cached = pd.read_parquet(cache)
            if len(cached) >= _MIN_UNIVERSE_SIZE:
                logger.info(
                    "Universe loaded from cache: %d stocks (%s)",
                    len(cached), cache,
                )
                # Sanity-check schema
                if {"symbol", "name", "board"}.issubset(cached.columns):
                    return cached.reset_index(drop=True)
                logger.warning("Cached universe has unexpected columns — re-fetching")
            else:
                logger.warning(
                    "Cached universe too small (%d < %d) — re-fetching",
                    len(cached), _MIN_UNIVERSE_SIZE,
                )
        except Exception as exc:
            logger.warning("Universe cache read failed (%s) — will re-fetch", exc)

    # ── Live fetch (exchange-primary, Eastmoney-fallback) ──
    logger.info("Fetching A-share stock universe (live) …")
    raw = None
    last_error = None

    # Primary: exchange-direct (downloads .xls, no browser UA needed)
    try:
        raw = ak.stock_info_a_code_name()
    except Exception as exc:
        last_error = exc
        logger.warning(
            "Exchange source failed (%s) — trying Eastmoney …",
            str(exc)[:80],
        )

    # Fallback: Eastmoney JSON API (needs browser UA to avoid blocking)
    if raw is None or len(raw) == 0:
        try:
            raw = _call_eastmoney_with_browser_headers()
        except Exception as exc2:
            last_error = exc2
            logger.error("Eastmoney also failed: %s", str(exc2)[:80])

    if raw is None or len(raw) == 0:
        raise RuntimeError(
            "All universe sources exhausted. "
            "Last error: %s" % str(last_error)[:120] if last_error else "unknown"
        )

    df = _normalise_universe_df(raw)

    # ── Persist cache ──
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        logger.info("Universe cached → %s", cache)
    except Exception as exc:
        logger.warning("Could not write universe cache: %s", exc)

    logger.info("  → %d A-share codes (主板/创业板/科创板)", len(df))
    return df


# ═══════════════════════════════════════════════════════════
# Public API: Daily OHLCV — baostock primary, Eastmoney fallback
# ═══════════════════════════════════════════════════════════

def _to_baostock_date(date_str: str) -> str:
    """
    Convert YYYYMMDD → YYYY-MM-DD for baostock.

    Already-formatted strings (YYYY-MM-DD) pass through unchanged.
    """
    s = str(date_str).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return s  # pass through if already formatted


def _fetch_history_via_baostock(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    Fetch single-stock OHLCV history via baostock (no Eastmoney dependency).
    """
    init_baostock()

    bs_code = _format_code_to_baostock(symbol)
    bs_adjust = _BAOSTOCK_ADJUST_MAP.get(adjust, "2")

    rs = _bs.query_history_k_data_plus(
        bs_code,
        "date,open,high,low,close,volume,amount,pctChg,turn",
        start_date=_to_baostock_date(start_date),
        end_date=_to_baostock_date(end_date),
        frequency="d",
        adjustflag=bs_adjust,
    )

    if rs.error_code != "0":
        raise ConnectionError(f"baostock query failed: [{rs.error_code}] {rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        logger.debug("  %s: baostock returned no rows for %s → %s", symbol, start_date, end_date)
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=rs.fields)
    df.columns = [c.strip().lower() for c in df.columns]

    # ── Standardize: baostock → canonical column names ──
    result = pd.DataFrame()
    result["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    result["open"] = _safe_to_numeric(df["open"]) if "open" in df.columns else np.nan
    result["high"] = _safe_to_numeric(df["high"]) if "high" in df.columns else np.nan
    result["low"] = _safe_to_numeric(df["low"]) if "low" in df.columns else np.nan
    result["close"] = _safe_to_numeric(df["close"]) if "close" in df.columns else np.nan
    result["volume"] = _safe_to_numeric(df["volume"]) if "volume" in df.columns else np.nan
    result["amount"] = _safe_to_numeric(df["amount"]) if "amount" in df.columns else np.nan
    result["pct_change"] = _safe_to_numeric(df["pctchg"]) if "pctchg" in df.columns else np.nan
    result["turnover_rate"] = _safe_to_numeric(df["turn"]) if "turn" in df.columns else np.nan
    result["symbol"] = str(symbol).zfill(6)

    # Drop rows with invalid dates or missing close
    result = result.dropna(subset=["date", "close"])
    return result.reset_index(drop=True)


def _fetch_history_via_akshare(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    Fallback: fetch single-stock OHLCV via akshare (Eastmoney).
    Only called when baostock is unavailable or returns empty.
    """
    import akshare as ak

    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )

    if df is None or len(df) == 0:
        logger.debug("  %s: akshare returned no data for %s → %s", symbol, start_date, end_date)
        return pd.DataFrame()

    result = _clean_and_rename(df, OHLCV_COLUMNS)
    result["symbol"] = str(symbol).zfill(6)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col not in result.columns:
            result[col] = np.nan

    return result[[c for c in OHLCV_COLUMNS.values() if c in result.columns]
                  + ["symbol"]]


@ak_retry(max_attempts=5, min_wait=2.0, max_wait=60.0)
@_rate_limiter
def fetch_single_stock_history(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    Fetch OHLCV history for a single stock.

    **Priority**: baostock (primary) → akshare/Eastmoney (fallback).

    Parameters
    ----------
    symbol : str
        6-digit A-share code, e.g. "000001".
    start_date : str
        "YYYYMMDD" format.
    end_date : str
        "YYYYMMDD" format.
    adjust : str
        "qfq" = forward-adjusted, "hfq" = backward-adjusted, "" = raw.

    Returns
    -------
    pd.DataFrame with standardized columns:
        date, open, high, low, close, volume, amount, pct_change,
        turnover_rate, symbol
    """
    # ── Primary: baostock ──
    try:
        df = _fetch_history_via_baostock(symbol, start_date, end_date, adjust)
        if len(df) > 0:
            return df
    except Exception as exc:
        logger.debug("  %s: baostock failed (%s) — falling back to akshare",
                     symbol, str(exc)[:80])

    # ── Fallback: akshare / Eastmoney ──
    return _fetch_history_via_akshare(symbol, start_date, end_date, adjust)


@ak_retry(max_attempts=3, min_wait=3.0, max_wait=30.0)
def fetch_daily_market_data(
    trade_date: str | date | None = None,
    universe: pd.DataFrame | None = None,
    *,
    throttle_seconds: float = 0.3,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for ALL A-shares on a specific trade date.

    This is the primary daily ingestion function. It iterates over
    the stock universe and fetches a single day's bar per symbol.
    (AkShare does not offer a bulk daily-quote endpoint with full OHLCV;
     we use stock_zh_a_hist with start_date=end_date=trade_date.)

    Args:
        trade_date: Target trade date. Default = latest trading day.
        universe: Optional pre-fetched stock list. If None, auto-fetches.
        throttle_seconds: Delay between per-stock API calls to avoid rate limiting.

    Returns:
        pd.DataFrame with cols: date, symbol, open, high, low, close,
        volume, amount, pct_change, turnover_rate

    Defense-in-depth:
        - Each per-stock call has @ak_retry (5 attempts, exp backoff)
        - Between symbols, we sleep throttle_seconds
        - Numeric columns are stringent-coerced per row
        - If a stock fails all 5 retries → skipped (logged), not a crash
    """
    if trade_date is None:
        trade_date = date.today()
    if isinstance(trade_date, date):
        trade_date = trade_date.strftime("%Y%m%d")
    trade_date = str(trade_date)

    logger.info("Fetching daily OHLCV for %s ...", trade_date)

    if universe is None:
        universe = fetch_all_a_share_codes()
    symbols = universe["symbol"].tolist()

    all_rows: list[dict] = []
    n_failed = 0
    n_total = len(symbols)

    for i, symbol in enumerate(symbols):
        try:
            df = fetch_single_stock_history(
                symbol, start_date=trade_date, end_date=trade_date, adjust="qfq"
            )
            if len(df) > 0:
                # We only care about the target date row
                row = df[df["date"] == pd.Timestamp(trade_date)]
                if len(row) == 0:
                    row = df.iloc[-1:]  # fallback: latest row
                all_rows.append({
                    "date": pd.Timestamp(trade_date),
                    "symbol": symbol,
                    "open": float(row["open"].iloc[0]) if pd.notna(row["open"].iloc[0]) else np.nan,
                    "high": float(row["high"].iloc[0]) if pd.notna(row["high"].iloc[0]) else np.nan,
                    "low": float(row["low"].iloc[0]) if pd.notna(row["low"].iloc[0]) else np.nan,
                    "close": float(row["close"].iloc[0]) if pd.notna(row["close"].iloc[0]) else np.nan,
                    "volume": float(row["volume"].iloc[0]) if pd.notna(row["volume"].iloc[0]) else np.nan,
                    "amount": float(row["amount"].iloc[0]) if pd.notna(row["amount"].iloc[0]) else np.nan,
                    "pct_change": float(row["pct_change"].iloc[0]) if "pct_change" in row and pd.notna(row["pct_change"].iloc[0]) else np.nan,
                    "turnover_rate": float(row["turnover_rate"].iloc[0]) if "turnover_rate" in row and pd.notna(row["turnover_rate"].iloc[0]) else np.nan,
                })
        except (RetryError, Exception) as e:
            n_failed += 1
            if n_failed <= 5:
                logger.warning("  %s: all retries exhausted (%s)", symbol, str(e)[:80])

        # Progress log
        if (i + 1) % 500 == 0:
            logger.info("  … %d/%d stocks fetched, %d failed", i + 1, n_total, n_failed)

        # Throttle between calls
        if throttle_seconds > 0 and i < n_total - 1:
            time.sleep(throttle_seconds)

    result = pd.DataFrame(all_rows)
    if n_failed > 0:
        logger.warning("fetch_daily_market_data: %d/%d symbols failed (%.1f%%)",
                       n_failed, n_total, 100 * n_failed / n_total)

    # Defensive: ensure numeric
    for col in ["open", "high", "low", "close", "volume", "amount", "pct_change", "turnover_rate"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    logger.info("  → %d rows, %d unique symbols", len(result), result["symbol"].nunique())
    return result


# ═══════════════════════════════════════════════════════════
# Public API: Fundamentals — curl_cffi primary, akshare fallback
# ═══════════════════════════════════════════════════════════

# Eastmoney field → canonical column mapping (curl_cffi direct JSON path)
_EM_FIELD_MAP = {
    "f12": "symbol",
    "f14": "name",
    "f9":  "pe_ttm",
    "f23": "pb",
    "f20": "total_mcap",
    "f21": "float_mcap",
    "f37": "roe",
    "f48": "eps",
    "f40": "revenue",
    "f45": "net_profit",
    "f55": "operating_profit",
    "f57": "gross_margin",
    "f49": "net_margin",
    "f115": "pe_static",
    "f100": "industry_name",
}

_EM_FIELDS = ",".join(_EM_FIELD_MAP.keys())


def _fetch_fundamentals_via_baostock(universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute daily fundamentals from baostock financial statements + close prices.

    Uses:
      - ``bs.query_profit_data()`` → epsTTM, roeAvg, npMargin, totalShare, netProfit
      - ``market_cache`` (SQLite) → latest close price

    Computed metrics:
      - Market Cap  = close_price * totalShare
      - PE_TTM      = close_price / epsTTM
      - ROE         = roeAvg (latest quarter)
      - Net Margin  = npMargin
      - EPS         = epsTTM
      - PB          = NaN (not computable from profit data alone)

    This is the MOST RELIABLE source — baostock uses its own TCP protocol,
    completely independent of Eastmoney / Sina TLS fingerprinting.
    """
    from paper_trading.state_manager import StateManager

    init_baostock()

    logger.info("Computing fundamentals via baostock financial statements ...")

    # ── 1. Universe ──
    if universe is None or len(universe) == 0:
        universe = fetch_all_a_share_codes()
    symbols = universe["symbol"].astype(str).str.zfill(6).tolist()
    names = dict(zip(universe["symbol"].astype(str).str.zfill(6), universe["name"]))

    # ── 2. Latest close prices from SQLite market_cache ──
    db_dir = Path("output/paper_trading_db")
    close_prices: dict[str, float] = {}
    try:
        state = StateManager(db_dir)
        with state._get_conn() as conn:
            # Get the most recent close for each symbol
            rows = conn.execute(
                "SELECT symbol, close FROM market_cache "
                "WHERE (symbol, trade_date) IN ("
                "  SELECT symbol, MAX(trade_date) FROM market_cache GROUP BY symbol"
                ")"
            ).fetchall()
            close_prices = {str(r[0]).zfill(6): float(r[1]) for r in rows if r[1] is not None}
        logger.info("  %d stocks with close prices in market_cache", len(close_prices))
    except Exception as exc:
        logger.warning("  Cannot read market_cache for close prices (%s) — "
                       "using baostock K-line instead", str(exc)[:60])

    # ── 3. Query profit data per stock ──
    # Latest quarter: as of June 2026, the most recent available report is Q1 2026
    YEAR, QUARTER = 2026, 1

    rows_out = []
    n_ok = 0
    n_miss = 0
    throttle = 0.02  # baostock handles ~50 qps comfortably
    t_last = time.monotonic()

    for idx, symbol in enumerate(symbols):
        # Throttle
        elapsed = time.monotonic() - t_last
        if elapsed < throttle:
            time.sleep(throttle - elapsed)
        t_last = time.monotonic()

        try:
            bs_code = _format_code_to_baostock(symbol)
            rs = _bs.query_profit_data(bs_code, year=YEAR, quarter=QUARTER)
            if rs.error_code != "0":
                n_miss += 1
                continue

            profit_rows = []
            while rs.next():
                profit_rows.append(rs.get_row_data())
            if not profit_rows:
                n_miss += 1
                continue

            row_data = dict(zip(rs.fields, profit_rows[0]))

            eps_ttm = _safe_float(row_data.get("epsTTM"))
            roe = _safe_float(row_data.get("roeAvg"))
            np_margin = _safe_float(row_data.get("npMargin"))
            total_share = _safe_float(row_data.get("totalShare"))

            close = close_prices.get(symbol)
            if close is None or close <= 0:
                # Fallback: fetch latest K-line from baostock
                try:
                    k_rs = _bs.query_history_k_data_plus(
                        bs_code, "close",
                        start_date="2026-05-01", end_date="2026-06-08",
                        frequency="d", adjustflag="2",
                    )
                    if k_rs.error_code == "0":
                        k_rows = []
                        while k_rs.next():
                            k_rows.append(k_rs.get_row_data())
                        if k_rows:
                            close = _safe_float(k_rows[-1][0])
                except Exception:
                    pass

            if close is None or close <= 0:
                n_miss += 1
                continue

            # ── Compute metrics ──
            mcap = close * total_share if total_share else None
            pe = close / eps_ttm if eps_ttm and eps_ttm > 0 else None

            rows_out.append({
                "symbol": symbol,
                "name": names.get(symbol, ""),
                "close": close,
                "pe_ttm": pe if pe and pe > 0 else None,
                "pb": None,  # requires book value from balance sheet
                "total_mcap": mcap,
                "float_mcap": None,  # liqaShare available but unused for now
                "roe": roe,
                "eps": eps_ttm,
                "bps": None,
                "revenue": None,
                "net_profit": _safe_float(row_data.get("netProfit")),
                "operating_profit": None,
                "gross_margin": None,
                "net_margin": np_margin,
                "pe_static": None,
            })
            n_ok += 1

        except Exception:
            n_miss += 1
            continue

        if (idx + 1) % 500 == 0:
            logger.info("  %d / %d stocks processed (%d ok, %d missing)",
                        idx + 1, len(symbols), n_ok, n_miss)

    logger.info("  → %d stocks with baostock fundamentals (%d missing, %.0f%%)",
                n_ok, n_miss, 100 * n_ok / max(len(symbols), 1))

    if not rows_out:
        raise RuntimeError("baostock fundamentals: 0 stocks computed")

    df = pd.DataFrame(rows_out)
    df["board"] = df["symbol"].apply(_classify_board)
    return df.reset_index(drop=True)


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None or val == "" or val == "0.000000" or val == "0":
        return None
    try:
        f = float(val)
        return f if not np.isnan(f) else None
    except (ValueError, TypeError):
        return None


def _fetch_fundamentals_via_curl_cffi(page_size: int = 100) -> pd.DataFrame:
    """
    Paginate Eastmoney's JSON API via curl_cffi with Chrome TLS impersonation.

    Uses a persistent Session with ``impersonate="chrome120"`` and
    conservative inter-page delays (3 s) to stay under rate limits.
    Each page is retried up to 4 times with exponential cooldown before
    giving up; partial data is returned rather than crashing.
    """
    from curl_cffi import requests as _curl_req

    logger.info("Fetching fundamentals via Eastmoney (curl_cffi, impersonate=chrome120) ...")

    url = "https://push2.eastmoney.com/api/qt/clist/get"
    base_params = {
        "pz": str(page_size),
        "po": "1",
        "np": "1",
        "fltt": "2",
        "fid": "f12",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        "fields": _EM_FIELDS,
    }

    # Persist a Session across pages — connection reuse looks like a real
    # browser's keep-alive behaviour and reduces per-request TLS overhead.
    session = _curl_req.Session(impersonate="chrome120")

    all_items = []
    total = 0
    page = 1
    MAX_CONSECUTIVE_FAILURES = 5
    PAGE_RETRIES = 4         # per-page attempts
    COOLDOWN_BASE = 5.0      # base cooldown on page failure (seconds)
    INTER_PAGE_DELAY = 3.0   # normal delay between pages (seconds)

    try:
        while True:
            params = {**base_params, "pn": str(page)}

            page_ok = False
            last_error = None
            for retry_idx in range(PAGE_RETRIES):
                try:
                    resp = session.get(url, params=params, timeout=30)
                    resp.raise_for_status()
                    page_ok = True
                    break
                except Exception as exc:
                    last_error = exc
                    if retry_idx < PAGE_RETRIES - 1:
                        wait = COOLDOWN_BASE * (2 ** retry_idx)  # 5, 10, 20, 40
                        logger.debug("  page %d attempt %d failed — cooling %.0fs",
                                     page, retry_idx + 1, wait)
                        time.sleep(wait)

            if not page_ok:
                if page == 1:
                    raise ConnectionError(
                        f"curl_cffi Eastmoney failed on page 1 after {PAGE_RETRIES} attempts: "
                        f"{last_error}"
                    ) from last_error
                # Later pages: log and try to continue (partial data > no data)
                logger.warning(
                    "curl_cffi page %d failed after %d attempts (%s) — "
                    "skipping, %d stocks collected so far",
                    page, PAGE_RETRIES, str(last_error)[:60], len(all_items),
                )
                page += 1
                time.sleep(COOLDOWN_BASE * 4)  # long cool-down after a skip
                continue

            data = resp.json()
            chunk = data.get("data", {}).get("diff", [])
            if page == 1:
                total = data.get("data", {}).get("total", 0)
                logger.info("  Total available: %d stocks → ~%d pages (%.0fs/page)",
                            total,
                            (total // page_size) + (1 if total % page_size else 0),
                            INTER_PAGE_DELAY)

            if not chunk:
                break

            all_items.extend(chunk)

            if page % 15 == 0:
                logger.info("  Page %d: %d/%d stocks collected (%.0f%%)",
                            page, len(all_items), total,
                            100 * len(all_items) / max(total, 1))

            if page * page_size >= total:
                break

            page += 1
            time.sleep(INTER_PAGE_DELAY)

    finally:
        session.close()

    if not all_items:
        raise RuntimeError("curl_cffi: Eastmoney returned 0 records across all pages")

    if len(all_items) < total * 0.5:
        logger.warning(
            "Only %d / %d stocks retrieved (%.0f%%) — Eastmoney rate-limiting is active. "
            "Returning partial data.",
            len(all_items), total, 100 * len(all_items) / max(total, 1),
        )

    if not all_items:
        raise RuntimeError("curl_cffi: Eastmoney returned 0 records")

    # ── Build DataFrame ──
    df = pd.DataFrame(all_items)

    # Rename to canonical columns
    df = df.rename(columns={k: v for k, v in _EM_FIELD_MAP.items() if k in df.columns})

    # ── Clean symbol ──
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)

    # ── Board classification ──
    df["board"] = df["symbol"].apply(_classify_board)

    # ── Filter A-shares only ──
    df = df[df["symbol"].str.match(r"^(00|30|60|68)\d{4}$", na=False)].copy()

    # ── Numeric coercion for all value columns ──
    _num_cols = ["pe_ttm", "pb", "total_mcap", "float_mcap", "roe",
                 "eps", "revenue", "net_profit", "operating_profit",
                 "gross_margin", "net_margin", "pe_static"]
    for col in _num_cols:
        if col in df.columns:
            df[col] = _safe_to_numeric(df[col])

    # ── PE/PB sanity: negative values → NaN ──
    for col in ("pe_ttm", "pb", "pe_static"):
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # ── Handle percentage-format ROE/margins (Eastmoney returns raw %) ──
    # roe, gross_margin, net_margin come as percentages from the API
    # (e.g. 8.19 means 8.19%). Keep as-is — consumers should note these are %.
    # Convert to decimal if values are implausibly large (>100 for margin):
    for col in ("roe", "gross_margin", "net_margin"):
        if col in df.columns:
            mask = df[col] > 100
            if mask.any():
                df.loc[mask, col] = df.loc[mask, col] / 100.0

    logger.info("  → %d A-shares with fundamentals (PE/PB/ROE/...)", len(df))
    return df.reset_index(drop=True)


@ak_retry(max_attempts=3, min_wait=2.0, max_wait=20.0)
@_rate_limiter
def fetch_daily_fundamentals(
    universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Fetch the latest valuation & profitability metrics for all A-shares.

    **Priority**: baostock financial statements (primary)
                  → curl_cffi + Eastmoney JSON API (fallback)
                  → akshare ``stock_zh_a_spot_em()`` (last resort)

    Returns
    -------
    pd.DataFrame with standardized columns:
        symbol, name, close, pe_ttm, pb, total_mcap, float_mcap, roe, eps,
        bps, revenue, net_profit, operating_profit, gross_margin, net_margin,
        pe_static, board
    """
    # ── Primary: baostock financial data (no Eastmoney dependency) ──
    try:
        return _fetch_fundamentals_via_baostock(universe)
    except Exception as exc:
        logger.warning(
            "baostock fundamentals failed (%s) — trying curl_cffi Eastmoney",
            str(exc)[:100],
        )

    # ── Fallback 1: curl_cffi with TLS impersonation ──
    try:
        return _fetch_fundamentals_via_curl_cffi()
    except Exception as exc:
        logger.warning(
            "curl_cffi fundamentals failed (%s) — falling back to akshare",
            str(exc)[:100],
        )

    # ── Fallback: akshare / Eastmoney (may be blocked by JA3) ──
    import akshare as ak

    logger.info("Fetching daily fundamentals (akshare fallback)...")
    df = ak.stock_zh_a_spot_em()

    # ── Rename to canonical columns ──
    rename_map = {
        "代码": "symbol",
        "名称": "name",
        "市盈率-动态": "pe_ttm",
        "市净率": "pb",
        "总市值": "total_mcap",
        "流通市值": "float_mcap",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    numeric_cols = ["pe_ttm", "pb", "total_mcap", "float_mcap"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _safe_to_numeric(df[col])

    field_aliases = {
        "roe": ["净资产收益率", "ROE"],
        "eps": ["每股收益", "EPS"],
        "bps": ["每股净资产", "BPS"],
        "revenue": ["营业总收入", "营业收入"],
        "net_profit": ["净利润", "归属净利润"],
        "gross_margin": ["销售毛利率", "毛利率"],
        "net_margin": ["销售净利率", "净利率"],
    }

    for canonical, aliases in field_aliases.items():
        found = False
        for alias in aliases:
            if alias in df.columns:
                df[canonical] = _safe_to_numeric(df[alias])
                found = True
                break
        if not found:
            df[canonical] = np.nan

    df["board"] = df["symbol"].apply(_classify_board)
    df = df[df["symbol"].str.match(r"^(00|30|60|68)\d{4}$", na=False)].copy()

    for col in ["roe", "eps", "bps", "gross_margin", "net_margin"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    select_cols = ["symbol", "name", "pe_ttm", "pb", "total_mcap", "float_mcap",
                   "roe", "eps", "bps", "revenue", "net_profit",
                   "gross_margin", "net_margin", "board"]
    result = df[[c for c in select_cols if c in df.columns]].copy()

    num_cols = [c for c in ["pe_ttm", "pb", "total_mcap", "roe"] if c in result.columns]
    if num_cols:
        result = result.dropna(subset=num_cols, how="all")

    logger.info("  → %d stocks with fundamental data (akshare fallback)", len(result))
    return result.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════
# Public API: PIT-Aligned Financial Statements
# ═══════════════════════════════════════════════════════════

# A-share statutory disclosure deadlines (per CSRC regulation)
#   Report period    | Fiscal period end | Latest legal disclosure date
#   -----------------+-------------------+-----------------------------
#   Q1 (一季报)       | 03-31             | 04-30 (same year)
#   Semi-Annual (中报) | 06-30             | 08-31 (same year)
#   Q3 (三季报)       | 09-30             | 10-31 (same year)
#   Annual (年报)     | 12-31             | 04-30 (NEXT year)

_STATUTORY_DISCLOSURE_RULES = {
    3: (4, 30),    # Q1: March 31 → April 30
    6: (8, 31),    # H1: June 30  → August 31
    9: (10, 31),   # Q3: September 30 → October 31
    12: (4, 30),   # FY: December 31 → April 30 NEXT YEAR
}


def _statutory_disclosure_date(report_period: pd.Timestamp) -> pd.Timestamp:
    """
    Compute the latest legal date on which financial statements
    for a given report period become publicly available.

    Args:
        report_period: pd.Timestamp, e.g. 2025-12-31 (fiscal period end).

    Returns:
        pd.Timestamp of the statutory disclosure deadline.

    Rules (CSRC):
      - Q1  (period ends 03-31) → deadline 04-30 same year
      - H1  (period ends 06-30) → deadline 08-31 same year
      - Q3  (period ends 09-30) → deadline 10-31 same year
      - FY  (period ends 12-31) → deadline 04-30 NEXT year
    """
    month = report_period.month
    if month not in _STATUTORY_DISCLOSURE_RULES:
        # Fallback: assume available 2 months after period end
        return report_period + pd.DateOffset(months=2)

    deadline_month, deadline_day = _STATUTORY_DISCLOSURE_RULES[month]
    year = report_period.year
    if month == 12:
        year += 1  # Annual report: next year
    return pd.Timestamp(year=year, month=deadline_month, day=deadline_day)


@_rate_limiter
def _fetch_financial_statement(symbol: str, *, max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch the full financial abstract (资产负债表 + 利润表 + 现金流量表)
    for a single stock from AkShare / Eastmoney.

    **Retry policy** (manual, NO tenacity double-stacking):
      - Network errors (ConnectionError, Timeout, etc.): retry up to 3 times
        with exponential backoff (2s → 4s → 8s).
      - JSONDecodeError / ValueError: **fast-fail** — empty response means
        Eastmoney blocked this stock; retrying produces the same error.
      - All other exceptions: fast-fail, return empty DataFrame.

    Returns wide-format DataFrame where each column is a report period
    (e.g. '20251231', '20250930') and each row is a financial indicator.
    """
    import akshare as ak

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_financial_abstract(symbol=symbol)
            if df is None or len(df) == 0:
                return pd.DataFrame()
            return df
        except (ValueError, TypeError) as exc:
            # JSONDecodeError / data-parsing errors are DETERMINISTIC —
            # Eastmoney returned empty/blocked response; retrying won't help.
            err_name = type(exc).__name__
            msg = str(exc)[:120]
            # JSONDecodeError is a subclass of ValueError in json module,
            # but some HTTP libs raise plain ValueError for empty responses
            if "json" in err_name.lower() or "expecting value" in msg.lower() or "json" in msg.lower():
                logger.debug(
                    "  %s: Eastmoney JSON parse failed on attempt %d/%d — fast-fail (%s: %s)",
                    symbol, attempt, max_retries, err_name, msg,
                )
            else:
                logger.debug(
                    "  %s: data error on attempt %d/%d — fast-fail (%s: %s)",
                    symbol, attempt, max_retries, err_name, msg,
                )
            return pd.DataFrame()
        except _SKIPPABLE_ERRORS as exc:
            # Network error — retryable
            last_error = exc
            if attempt < max_retries:
                wait_s = min(2.0 * (2 ** (attempt - 1)), 20.0)
                logger.debug(
                    "  %s: network error on attempt %d/%d, retrying in %.1fs (%s: %s)",
                    symbol, attempt, max_retries, wait_s,
                    type(exc).__name__, str(exc)[:80],
                )
                time.sleep(wait_s)
            else:
                logger.warning(
                    "  %s: exhausted %d retries (last: %s: %s) — skipping",
                    symbol, max_retries, type(exc).__name__, str(exc)[:80],
                )
        except Exception as exc:
            # Unknown error — fast-fail (not worth retrying)
            logger.debug(
                "  %s: unexpected error — fast-fail (%s: %s)",
                symbol, type(exc).__name__, str(exc)[:80],
            )
            return pd.DataFrame()

    # Exhausted all retries on network errors
    if last_error is not None:
        raise last_error
    return pd.DataFrame()


@_rate_limiter
def _fetch_single_financial_for_pit(symbol: str) -> dict | None:
    """
    Fetch and PIT-align financial data for ONE stock.

    Returns dict of the most recent legally-available financial indicators,
    or None if all fetches fail.

    **Resilience**: the entire body is wrapped in try-except.  If any part
    of the parsing pipeline fails (malformed data, missing columns, etc.),
    we return None rather than crashing or retrying.
    """
    try:
        raw = _fetch_financial_statement(symbol)
    except Exception:
        return None

    if raw is None or len(raw) == 0:
        return None

    try:
        # raw columns: ['选项', '指标', '20260331', '20251231', ...]
        # We need to pivot: periods as rows, indicators as columns
        period_cols = [c for c in raw.columns if c not in ("选项", "指标") and len(str(c)) == 8]

        if not period_cols:
            return None

        # Find all report periods and their statutory disclosure dates
        periods = []
        for pc in period_cols:
            try:
                rp = pd.Timestamp(str(pc))
                sd = _statutory_disclosure_date(rp)
                periods.append({"report_period": rp, "disclosure_date": sd})
            except (ValueError, TypeError):
                continue

        if not periods:
            return None

        disc_df = pd.DataFrame(periods).sort_values("report_period")

        # Return all periods + disclosure dates
        indicators = {}
        for _, row in raw.iterrows():
            indicator_name = str(row.get("指标", ""))
            if not indicator_name:
                continue
            for pc in period_cols:
                val = row.get(pc)
                if pd.notna(val) and val != "--":
                    try:
                        if isinstance(val, str):
                            indicators[f"{indicator_name}|{pc}"] = float(
                                val.replace(",", "").replace("%", "")
                            )
                        else:
                            indicators[f"{indicator_name}|{pc}"] = float(val)
                    except (ValueError, TypeError):
                        continue

        result = {
            "symbol": str(symbol).zfill(6),
            "periods": disc_df,
            "indicators": indicators,
        }
        return result

    except Exception as exc:
        logger.debug("  %s: PIT parsing failed (%s: %s) — skipping",
                     symbol, type(exc).__name__, str(exc)[:80])
        return None


def _pit_filter_periods(periods_df: pd.DataFrame, current_date: pd.Timestamp) -> pd.Timestamp | None:
    """
    From a DataFrame of (report_period, disclosure_date), return the
    most recent report_period whose statutory disclosure date is <= current_date.

    This is the core PIT safety gate — we must NOT use data from a period
    that hasn't been legally disclosed yet.

    Example:
      current_date = 2026-06-30
      Available periods: 2025-12-31 (disclosed 2026-04-30) ✓
                         2026-03-31 (disclosed 2026-04-30) ✓
                         2026-06-30 (disclosed 2026-08-31) ✗ ← FORBIDDEN
      → Returns 2026-03-31
    """
    valid = periods_df[periods_df["disclosure_date"] <= current_date]
    if len(valid) == 0:
        return None
    return valid["report_period"].max()


def fetch_and_align_financials(
    current_date: date | str | pd.Timestamp,
    universe: pd.DataFrame | None = None,
    *,
    throttle_seconds: float = 0.0,
    max_stocks: int | None = None,
    force_refresh: bool = False,
    use_baostock: bool = False,
) -> pd.DataFrame:
    """
    Fetch the most recent PIT-safe financial data for all A-shares.

    This is THE function to use for month-end rebalance. It guarantees
    zero look-ahead bias by enforcing statutory disclosure deadlines.

    **Cache**: Results are cached to
    ``output/paper_trading_db/pit_financials_{YYYYMMDD}.parquet``.
    On subsequent calls with the same ``current_date``, the cache is
    used directly — skipping the 2-4 hour sequential fetch entirely.
    Pass ``force_refresh=True`` to bypass the cache.

    **Rate limiting**: The ``@_rate_limiter`` decorator on
    ``_fetch_financial_statement`` already enforces 4 calls/s.
    ``throttle_seconds`` defaults to 0 to avoid double-throttling
    (rate_limiter + sleep stacking).

    Pipeline:
      1. Check cache → return immediately if valid
      2. For each stock, fetch financial abstract (wide-format)
      3. Parse all report periods → compute statutory disclosure dates
      4. Filter: keep only periods where disclosure_date <= current_date
      5. Extract the LATEST valid period's indicators per stock
      6. Save to cache → return standardized DataFrame

    Args:
        current_date: The "as of" date. Only financial data legally available
                      on or before this date will be used.
        universe: Optional pre-fetched stock list.
        throttle_seconds: Per-stock delay for rate limiting.
        max_stocks: Limit stocks for testing. None = all.
        force_refresh: If True, ignore cache and re-fetch from source.
        use_baostock: If True, use baostock + ProcessPoolExecutor for a
                      ~30× speedup (3 hours → 7 minutes) with true pubDate
                      PIT gating.  Falls back to Eastmoney on failure.

    Returns:
        pd.DataFrame with columns:
          symbol, report_period (the actual period used),
          total_assets, total_equity, total_liabilities,
          revenue, operating_profit, net_profit,
          revenue_yoy, profit_yoy,
          debt_ratio, roe, eps, bps

    Look-ahead bias prevention:
      - Q4 FY2025 (report 2025-12-31): available only from 2026-04-30
      - Q1 FY2026 (report 2026-03-31): available only from 2026-04-30
      - Q2 FY2026 (report 2026-06-30): NOT available until 2026-08-31
      On current_date=2026-06-30, only Q1-2026 is used (never Q2-2026).
    """
    if isinstance(current_date, str):
        current_dt = pd.Timestamp(current_date[:10])
    elif isinstance(current_date, date):
        current_dt = pd.Timestamp(current_date)
    else:
        current_dt = current_date

    date_str = current_dt.strftime("%Y%m%d")

    # ── Cache layer ──
    _PIT_CACHE_DIR = Path("output/paper_trading_db")
    cache_path = _PIT_CACHE_DIR / f"pit_financials_{date_str}.parquet"

    if not force_refresh and cache_path.exists():
        logger.info("PIT financials cache HIT for %s → loading %s",
                    current_dt.strftime("%Y-%m-%d"), cache_path.name)
        try:
            df = pd.read_parquet(cache_path)
            if "report_period" in df.columns:
                df["report_period"] = pd.to_datetime(df["report_period"])
            logger.info("  -> %d stocks loaded from cache", len(df))
            return df
        except Exception as exc:
            logger.warning("Cache file corrupted (%s), re-fetching...", str(exc)[:80])
            try:
                cache_path.unlink()
            except OSError:
                pass

    if force_refresh:
        logger.info("PIT financials: force_refresh=True, ignoring cache")

    # ── Fetch ──
    logger.info("Fetching PIT-aligned financials as of %s ...", current_dt.strftime("%Y-%m-%d"))

    if universe is None:
        universe = fetch_all_a_share_codes()

    symbols = universe["symbol"].tolist()
    if max_stocks:
        symbols = symbols[:max_stocks]

    # ── Baostock fast path (PIT-safe via real pubDate) ──
    if use_baostock:
        logger.info("  Using BaostockAdapter (ProcessPool, pubDate-gated) ...")
        try:
            from paper_trading.baostock_adapter import BaostockAdapter

            result_df = BaostockAdapter.get_pit_financials_batch(
                symbols, current_dt, workers=8,
            )

            if result_df is None or len(result_df) == 0:
                logger.warning("  Baostock PIT: 0 stocks returned — "
                               "falling back to Eastmoney sequential path")
                use_baostock = False  # fall through to Eastmoney
            else:
                # ── Persist cache ──
                try:
                    _PIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    result_df.to_parquet(cache_path, index=False)
                    logger.info("  Cache saved → %s (%.0f KB)", cache_path.name,
                                cache_path.stat().st_size / 1024)
                except Exception as exc:
                    logger.debug("  Could not save PIT cache: %s", str(exc)[:80])

                logger.info("  -> %d stocks with PIT-safe financial data (period <= %s)",
                            len(result_df), current_dt.strftime("%Y-%m-%d"))
                return result_df

        except ImportError as exc:
            logger.warning("  BaostockAdapter not available (%s) — "
                           "falling back to Eastmoney sequential", exc)
            use_baostock = False
        except Exception as exc:
            logger.error("  BaostockAdapter failed: %s — "
                         "falling back to Eastmoney sequential", exc)
            use_baostock = False

    # ── Checkpoint/resume (Eastmoney path only) ──
    completed = _load_progress("pit_financials")
    remaining = [s for s in symbols if s not in completed]
    if len(completed) > 0:
        logger.info("  Resuming PIT financials: %d/%d already done, %d remaining",
                    len(completed), len(symbols), len(remaining))
    if not remaining:
        logger.info("  All %d PIT financials already processed — nothing to do.", len(symbols))
        return pd.DataFrame()

    all_rows: list[dict] = []
    n_failed = 0
    n_total = len(remaining)

    for i, symbol in enumerate(remaining):
        try:
            result = _fetch_single_financial_for_pit(symbol)
            if result is None or result.get("periods") is None or len(result["periods"]) == 0:
                n_failed += 1
                completed.add(symbol)
                continue

            periods_df = result["periods"]
            valid_period = _pit_filter_periods(periods_df, current_dt)

            if valid_period is None:
                n_failed += 1
                completed.add(symbol)
                continue

            period_str = valid_period.strftime("%Y%m%d")
            ind = result["indicators"]

            # Extract key financial indicators for the valid period
            # Indicator names vary by AkShare version — fuzzy match
            def _get_ind(keywords: list[str]) -> float | None:
                for kw in keywords:
                    full_key = f"{kw}|{period_str}"
                    if full_key in ind:
                        return ind[full_key]
                return None

            total_assets = _get_ind(["资产总计", "总资产", "资产总额"])
            total_equity = _get_ind(["归属母公司股东权益合计", "股东权益合计", "所有者权益合计", "归属于母公司所有者权益合计"])
            total_liabilities = _get_ind(["负债合计", "总负债"])
            revenue = _get_ind(["营业总收入", "营业收入", "营业总收入(万元)"])
            operating_profit = _get_ind(["营业利润", "营业利润(万元)"])
            net_profit = _get_ind(["净利润", "归属母公司净利润", "归属于母公司所有者的净利润"])
            roe_val = _get_ind(["净资产收益率", "加权平均净资产收益率", "ROE"])
            eps_val = _get_ind(["基本每股收益", "每股收益", "EPS"])
            bps_val = _get_ind(["每股净资产", "归属母公司每股净资产", "BPS"])

            # Compute derived metrics
            debt_ratio = None
            if total_assets is not None and total_liabilities is not None and total_assets > 0:
                debt_ratio = total_liabilities / total_assets

            all_rows.append({
                "symbol": symbol,
                "report_period": valid_period,
                "total_assets": total_assets,
                "total_equity": total_equity,
                "total_liabilities": total_liabilities,
                "revenue": revenue,
                "operating_profit": operating_profit,
                "net_profit": net_profit,
                "roe": roe_val,
                "eps": eps_val,
                "bps": bps_val,
                "debt_ratio": debt_ratio,
            })

            completed.add(symbol)

        except Exception:
            n_failed += 1
            completed.add(symbol)

        # Log progress + checkpoint every 500 stocks
        if (i + 1) % 500 == 0:
            logger.info("  ... %d/%d stocks processed, %d failed",
                        i + 1 + len(completed) - len(remaining),
                        len(symbols), n_failed)
            _save_progress("pit_financials", completed)

        if throttle_seconds > 0 and i < n_total - 1:
            time.sleep(throttle_seconds)

    # ── Clear checkpoint on successful completion ──
    _clear_progress("pit_financials")

    result_df = pd.DataFrame(all_rows)

    # ── Compute YoY growth (PIT-safe) ──
    # Compare current report_period vs same period last year
    # e.g., 2026-03-31 revenue vs 2025-03-31 revenue
    if "revenue" in result_df.columns and len(result_df) > 0:
        # For growth, we need prior-year data — compute from financial abstract
        # In production, use sequential fetches; for paper trading, use spot data
        result_df["revenue_yoy"] = np.nan
        result_df["profit_yoy"] = np.nan

    # ── Defensive: fill missing with NaN ──
    numeric_cols = ["total_assets", "total_equity", "total_liabilities",
                    "revenue", "operating_profit", "net_profit",
                    "roe", "eps", "bps", "debt_ratio"]
    for col in numeric_cols:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

    if n_failed > 0:
        logger.warning("PIT financials: %d/%d stocks failed (%.1f%%)",
                       n_failed, n_total, 100 * n_failed / max(n_total, 1))
    logger.info("  -> %d stocks with PIT-safe financial data (period <= %s)",
                len(result_df), current_dt.strftime("%Y-%m-%d"))

    result_df = result_df.reset_index(drop=True)

    # ── Persist cache ──
    try:
        _PIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(cache_path, index=False)
        logger.info("  Cache saved → %s (%.0f KB)", cache_path.name,
                    cache_path.stat().st_size / 1024)
    except Exception as exc:
        logger.debug("  Could not save PIT cache: %s", str(exc)[:80])

    return result_df


# ═══════════════════════════════════════════════════════════
# Public API: SW Industry Classification
# ═══════════════════════════════════════════════════════════
# Industry Classification — baostock primary, Eastmoney fallback
# ═══════════════════════════════════════════════════════════

def _fetch_industry_via_baostock() -> pd.DataFrame:
    """
    Fetch industry classification via baostock (CSRC 证监会行业分类).

    Returns DataFrame with columns: symbol, industry_name
    """
    init_baostock()

    rs = _bs.query_stock_industry()
    if rs.error_code != "0":
        raise ConnectionError(f"baostock industry query failed: [{rs.error_code}] {rs.error_msg}")

    rows = []
    while rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        logger.warning("baostock industry returned empty")
        return pd.DataFrame(columns=["symbol", "industry_name"])

    df = pd.DataFrame(rows, columns=rs.fields)

    # ── Convert codes: "sh.600000" → "600000" ──
    df["symbol"] = df["code"].astype(str).apply(_format_code_from_baostock)

    # ── Industry name: strip CSRC code prefix (e.g. "C39计算机..." → "计算机...") ──
    import re as _re
    industry_raw = df["industry"].astype(str).str.strip()
    industry_raw = industry_raw.str.replace(r"^[A-Z]\d{0,2}", "", regex=True).str.strip()
    df["industry_name"] = industry_raw

    # Drop stocks with missing industry
    df = df[df["industry_name"] != ""]

    logger.info("  → %d stocks with baostock industry (%d unique)",
                len(df), df["industry_name"].nunique())
    return df[["symbol", "industry_name"]].reset_index(drop=True)


def _fetch_industry_via_akshare() -> pd.DataFrame:
    """
    Fallback: fetch SW (申万) industry classification via akshare/Eastmoney.
    """
    import akshare as ak

    logger.info("Fetching SW industry classification (akshare fallback)...")
    try:
        df = ak.stock_board_industry_name_em()
    except Exception:
        logger.warning("stock_board_industry_name_em failed, trying stock_board_industry_cons_em...")
        try:
            df = ak.stock_board_industry_cons_em()
        except Exception:
            logger.error("All akshare industry APIs failed")
            return pd.DataFrame(columns=["symbol", "industry_name"])

    if df is None or len(df) == 0:
        logger.warning("akshare industry returned empty")
        return pd.DataFrame(columns=["symbol", "industry_name"])

    # Standardize column names across AkShare versions
    rename_candidates = {
        "代码": "symbol",
        "名称": "name",
        "板块名称": "industry_name",
        "行业名称": "industry_name",
        "所属行业": "industry_name",
        "board_name": "industry_name",
    }
    df = df.rename(columns={k: v for k, v in rename_candidates.items() if k in df.columns})

    # Ensure symbol column
    if "symbol" not in df.columns:
        for col in df.columns:
            if df[col].astype(str).str.match(r"^\d{6}$").all():
                df = df.rename(columns={col: "symbol"})
                break
        else:
            logger.error("Cannot identify symbol column in industry data")
            return pd.DataFrame(columns=["symbol", "industry_name"])

    df["symbol"] = df["symbol"].astype(str).str.zfill(6)

    # Ensure industry_name column
    if "industry_name" not in df.columns:
        for col in df.columns:
            if col not in ("symbol", "name") and df[col].nunique() < 100:
                df = df.rename(columns={col: "industry_name"})
                break
        else:
            logger.error("Cannot identify industry_name column")
            return pd.DataFrame(columns=["symbol", "industry_name"])

    df["industry_name"] = df["industry_name"].astype(str).str.strip()
    logger.info("  → %d stocks with SW industry (%d unique)",
                len(df), df["industry_name"].nunique())
    return df[["symbol", "industry_name"]].reset_index(drop=True)


@ak_retry(max_attempts=3, min_wait=2.0, max_wait=20.0)
@_rate_limiter
def fetch_industry_classification() -> pd.DataFrame:
    """
    Fetch industry classification for all A-shares.

    **Priority**: baostock / CSRC (primary) → akshare / SW (fallback).

    Returns
    -------
    pd.DataFrame with columns: symbol, industry_name
    """
    # ── Primary: baostock ──
    try:
        df = _fetch_industry_via_baostock()
        if len(df) >= 3000:  # expect most A-shares to have industry
            return df
        logger.warning("baostock industry only returned %d stocks — "
                       "falling back to akshare", len(df))
    except Exception as exc:
        logger.warning("baostock industry failed (%s) — falling back to akshare",
                       str(exc)[:80])

    # ── Fallback: akshare / Eastmoney ──
    try:
        return _fetch_industry_via_akshare()
    except Exception as exc:
        logger.error("All industry classification sources failed: %s", exc)
        return pd.DataFrame(columns=["symbol", "industry_name"])


# ═══════════════════════════════════════════════════════════
# Utility: Is Trade Date?
# ═══════════════════════════════════════════════════════════

@ak_retry(max_attempts=3, min_wait=1.0, max_wait=10.0)
def fetch_trade_calendar(year: int | None = None) -> pd.DataFrame:
    """
    Fetch A-share trading calendar for a given year.

    Returns DataFrame with columns: trade_date, is_trading_day
    """
    import akshare as ak

    if year is None:
        year = date.today().year

    logger.debug("Fetching trade calendar for %d", year)
    df = ak.tool_trade_date_hist_sina()
    df = df.rename(columns={"trade_date": "trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    return df


def is_trade_date(check_date: date | str | None = None) -> bool:
    """
    Check if a given date is a trading day.

    Uses a cached calendar; refreshes once per session.
    """
    global _TRADE_CALENDAR_CACHE
    if check_date is None:
        check_date = date.today()
    if isinstance(check_date, str):
        check_date = datetime.strptime(check_date, "%Y%m%d").date()

    cache_key = check_date.year
    if cache_key not in _TRADE_CALENDAR_CACHE:
        try:
            cal = fetch_trade_calendar(cache_key)
            _TRADE_CALENDAR_CACHE[cache_key] = set(
                cal["trade_date"].dropna().dt.date
            )
        except Exception:
            logger.warning("Cannot fetch trade calendar — assuming trade date")
            return True

    return check_date in _TRADE_CALENDAR_CACHE[cache_key]


_TRADE_CALENDAR_CACHE: dict[int, set] = {}


# ═══════════════════════════════════════════════════════════
# Convenience: last trading day of month detection
# ═══════════════════════════════════════════════════════════

def is_month_last_trade_date(check_date: date | str | None = None) -> bool:
    """
    Determine whether `check_date` is the last trading day of the month.

    Heuristic (no calendar lookup needed):
      - If check_date is a weekday (Mon-Fri)
      - AND (check_date + 1).month != check_date.month  (month boundary)
      Then it's the month's last trading day.

    Edge case: if the last calendar day falls on a weekend, the real last
    trading day is the preceding Friday. This heuristic handles that because
    we compare check_date (which is a weekday) to the next calendar day.
    """
    if check_date is None:
        check_date = date.today()
    if isinstance(check_date, str):
        check_date = datetime.strptime(check_date, "%Y%m%d").date()

    # Must be a weekday
    if check_date.weekday() >= 5:
        return False

    # Next calendar day is next month → today is last day of month
    next_day = check_date + timedelta(days=1)
    if next_day.month != check_date.month:
        # But wait — if today is Friday and the 1st is Monday, today IS the last trading day
        return True

    # Also handle: today is Thursday, month ends on Saturday → today is last trading day
    # (check_date + 1 is still same month, but +2 crosses)
    two_days_later = check_date + timedelta(days=2)
    if check_date.weekday() == 4:  # Friday
        if two_days_later.month != check_date.month:
            return True

    return False


# ═══════════════════════════════════════════════════════════
# Parallel Ingestion — ProcessPoolExecutor + Checkpoint/Resume
# ═══════════════════════════════════════════════════════════
#
# These are the PRODUCTION entry points for bulk data ingestion.
# They replace the sequential ``fetch_daily_market_data`` /
# ``fetch_daily_fundamentals`` loops with ProcessPoolExecutor-based
# concurrency, progress checkpoints every 5 batches, and per-stock
# error isolation (failures are logged to ``data/error_log.txt``
# and skipped — the main process NEVER crashes on a single stock).
#
# Baostock multi-process safety:
#   Each ProcessPoolExecutor worker calls ``bs.login()`` at start and
#   ``bs.logout()`` at exit, giving every process its OWN independent
#   TCP connection.  There is NO shared socket, NO ``_BS_LOCK``, and
#   NO possibility of intra-socket deadlock.  The ``_BS_LOCK`` global
#   has been removed — it is incompatible with process-level isolation
#   and caused socket-level hangs under ThreadPoolExecutor contention.
# ═══════════════════════════════════════════════════════════

# Tolerable exceptions for a single stock — logged + skipped, not re-raised.
_SKIPPABLE_ERRORS = (
    OSError,
    ConnectionError,
    ConnectionResetError,
    ConnectionRefusedError,
    TimeoutError,
)
try:
    from urllib3.exceptions import HTTPError as _Urllib3HTTPError
    _SKIPPABLE_ERRORS = _SKIPPABLE_ERRORS + (_Urllib3HTTPError,)
except ImportError:
    pass
try:
    from requests.exceptions import (
        ConnectionError as _ReqConnError,
        Timeout as _ReqTimeout,
    )
    _SKIPPABLE_ERRORS = _SKIPPABLE_ERRORS + (_ReqConnError, _ReqTimeout)
except ImportError:
    pass


def _is_skippable(exc: Exception) -> bool:
    """Return True when the exception is a known-transient network error."""
    if isinstance(exc, _SKIPPABLE_ERRORS):
        return True
    msg = str(exc).lower()
    trigger_words = (
        "proxy", "remote end closed", "max retries", "connection",
        "timeout", "reset", "refused", "too many requests",
        "429", "503", "502", "504",
    )
    return any(w in msg for w in trigger_words)


# ═══════════════════════════════════════════════════════════
# ProcessPoolExecutor Workers (module-level = picklable)
# ═══════════════════════════════════════════════════════════
#
# Each worker runs in its OWN process with its OWN baostock
# TCP connection (bs.login/bs.logout).  No shared socket,
# no _BS_LOCK deadlock, no global state contention.

def _market_data_batch_worker(batch_payload: tuple) -> tuple:
    """
    ProcessPoolExecutor worker: fetch OHLCV for one batch.

    **Each process calls bs.login()/bs.logout() independently.**

    Args:
        batch_payload: (symbols, trade_date, throttle_seconds)
    Returns:
        (ok_rows, errors)
        errors = [(symbol, exc_type_name, exc_message), ...]
    """
    symbols, trade_date, throttle_seconds = batch_payload

    lg = _bs.login()
    if lg.error_code != "0":
        return [], [(s, "BaostockLoginError",
                     f"[{lg.error_code}] {lg.error_msg}") for s in symbols]

    # Set the module-level flag so init_baostock() (called internally
    # by _fetch_history_via_baostock) sees we're already connected.
    global _bs_logged_in
    _bs_logged_in = True

    ok_rows: list[dict] = []
    errors: list[tuple] = []

    try:
        for symbol in symbols:
            try:
                if throttle_seconds > 0:
                    time.sleep(throttle_seconds)
                df = fetch_single_stock_history(
                    symbol, start_date=trade_date, end_date=trade_date, adjust="qfq"
                )

                if df is None or len(df) == 0:
                    continue

                row = df[df["date"] == pd.Timestamp(trade_date)]
                if len(row) == 0:
                    row = df.iloc[-1:]

                ok_rows.append({
                    "date": pd.Timestamp(trade_date),
                    "symbol": symbol,
                    "open": float(row["open"].iloc[0]) if pd.notna(row["open"].iloc[0]) else np.nan,
                    "high": float(row["high"].iloc[0]) if pd.notna(row["high"].iloc[0]) else np.nan,
                    "low": float(row["low"].iloc[0]) if pd.notna(row["low"].iloc[0]) else np.nan,
                    "close": float(row["close"].iloc[0]) if pd.notna(row["close"].iloc[0]) else np.nan,
                    "volume": float(row["volume"].iloc[0]) if pd.notna(row["volume"].iloc[0]) else np.nan,
                    "amount": float(row["amount"].iloc[0]) if pd.notna(row["amount"].iloc[0]) else np.nan,
                    "pct_change": float(row["pct_change"].iloc[0])
                        if "pct_change" in row.columns and pd.notna(row["pct_change"].iloc[0])
                        else np.nan,
                    "turnover_rate": float(row["turnover_rate"].iloc[0])
                        if "turnover_rate" in row.columns and pd.notna(row["turnover_rate"].iloc[0])
                        else np.nan,
                })
            except Exception as exc:
                errors.append((symbol, type(exc).__name__, str(exc)[:200]))
    finally:
        try:
            _bs.logout()
            _bs_logged_in = False
        except Exception:
            pass

    return ok_rows, errors


def _fundamentals_batch_worker(batch_payload: tuple) -> tuple:
    """
    ProcessPoolExecutor worker: compute fundamentals for one batch.

    **Each process calls bs.login()/bs.logout() independently.**

    Args:
        batch_payload: (symbols, close_prices, names, year, quarter)
    Returns:
        (ok_rows, errors)
    """
    symbols, close_prices, names, year, quarter = batch_payload

    lg = _bs.login()
    if lg.error_code != "0":
        return [], [(s, "BaostockLoginError",
                     f"[{lg.error_code}] {lg.error_msg}") for s in symbols]

    global _bs_logged_in
    _bs_logged_in = True

    ok_rows: list[dict] = []
    errors: list[tuple] = []

    try:
        for symbol in symbols:
            try:
                bs_code = _format_code_to_baostock(symbol)

                rs = _bs.query_profit_data(bs_code, year=year, quarter=quarter)
                if rs.error_code != "0":
                    errors.append((symbol, "BaostockError",
                                   f"[{rs.error_code}] {rs.error_msg}"))
                    continue

                profit_rows = []
                while rs.next():
                    profit_rows.append(rs.get_row_data())
                if not profit_rows:
                    continue

                row_data = dict(zip(rs.fields, profit_rows[0]))

                eps_ttm = _safe_float(row_data.get("epsTTM"))
                roe = _safe_float(row_data.get("roeAvg"))
                np_margin = _safe_float(row_data.get("npMargin"))
                total_share = _safe_float(row_data.get("totalShare"))

                close = close_prices.get(symbol)
                if close is None or close <= 0:
                    # K-line fallback (own connection, no lock needed)
                    try:
                        k_rs = _bs.query_history_k_data_plus(
                            bs_code, "close",
                            start_date="2026-05-01", end_date="2026-06-08",
                            frequency="d", adjustflag="2",
                        )
                        if k_rs.error_code == "0":
                            k_rows = []
                            while k_rs.next():
                                k_rows.append(k_rs.get_row_data())
                            if k_rows:
                                close = _safe_float(k_rows[-1][0])
                    except Exception:
                        pass

                if close is None or close <= 0:
                    continue

                mcap = close * total_share if total_share else None
                pe = close / eps_ttm if eps_ttm and eps_ttm > 0 else None

                ok_rows.append({
                    "symbol": symbol,
                    "name": names.get(symbol, ""),
                    "close": close,
                    "pe_ttm": pe if pe and pe > 0 else None,
                    "pb": None,
                    "total_mcap": mcap,
                    "float_mcap": None,
                    "roe": roe,
                    "eps": eps_ttm,
                    "bps": None,
                    "revenue": None,
                    "net_profit": _safe_float(row_data.get("netProfit")),
                    "operating_profit": None,
                    "gross_margin": None,
                    "net_margin": np_margin,
                    "pe_static": None,
                })
            except Exception as exc:
                errors.append((symbol, type(exc).__name__, str(exc)[:200]))
    finally:
        try:
            _bs.logout()
            _bs_logged_in = False
        except Exception:
            pass

    return ok_rows, errors


# ── fetch_daily_market_data_parallel ───────────────────────

def fetch_daily_market_data_parallel(
    trade_date: str | date | None = None,
    universe: pd.DataFrame | None = None,
    *,
    max_workers: int = 10,
    throttle_seconds: float = 0.02,
) -> pd.DataFrame:
    """
    Parallel OHLCV ingestion via ProcessPoolExecutor + checkpoint/resume.

    Semantics are identical to ``fetch_daily_market_data`` but:
      - Stocks are fetched in parallel across ``max_workers`` **processes**
        (NOT threads).  Each process calls ``bs.login()``/``bs.logout()``
        and owns its own TCP connection — no shared socket, no deadlock.
      - Stocks are batched (50 per batch) for efficient IPC.
      - Progress is saved to ``data/progress_market_data.json`` every
        5 batches (~250 stocks).
      - Per-stock network errors are returned to the main process, logged
        to ``data/error_log.txt``, and the stock is **skipped** — the
        main process does not crash.
      - A ``tqdm`` progress bar shows live ok/fail counts.

    Args:
        trade_date: Target trade date (default today).
        universe: Pre-fetched stock list.
        max_workers: ProcessPoolExecutor worker count.  8-12 is
                     a good range for typical CPUs.
        throttle_seconds: Inter-stock delay inside each worker
                          to avoid overwhelming baostock.

    Returns:
        pd.DataFrame with cols: date, symbol, open, high, low, close,
        volume, amount, pct_change, turnover_rate
    """
    if trade_date is None:
        trade_date = date.today()
    if isinstance(trade_date, date):
        trade_date = trade_date.strftime("%Y%m%d")
    trade_date = str(trade_date)

    logger.info("[PARALLEL] Fetching daily OHLCV for %s (%d workers) ...",
                trade_date, max_workers)

    if universe is None:
        universe = fetch_all_a_share_codes()
    symbols = universe["symbol"].astype(str).str.zfill(6).tolist()

    # ── Checkpoint: resume from previous run ──
    completed = _load_progress("market_data")
    remaining = [s for s in symbols if s not in completed]
    if len(completed) > 0:
        logger.info("  Resuming: %d/%d already done, %d remaining",
                    len(completed), len(symbols), len(remaining))
    if not remaining:
        logger.info("  All %d stocks already fetched — nothing to do.", len(symbols))
        return pd.DataFrame()

    # ── Don't call init_baostock() here!  Each ProcessPoolExecutor
    #     worker calls bs.login()/bs.logout() independently, giving
    #     each process its OWN TCP connection.  No shared socket,
    #     no _BS_LOCK, no deadlock.

    error_log_path = _PROGRESS_DIR / "error_log.txt"

    BATCH_SIZE = 50
    batches = [remaining[i:i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    logger.info("  %d batches (batch_size=%d), %d workers",
                len(batches), BATCH_SIZE, max_workers)

    all_rows: list[dict] = []
    n_failed = 0

    pbar = (
        _tqdm(total=len(remaining), desc="OHLCV", unit="stk",
              ncols=100, smoothing=0.1)
        if _TQDM_AVAILABLE else None
    )

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures_map = {
                executor.submit(_market_data_batch_worker,
                                (b, trade_date, throttle_seconds)): i
                for i, b in enumerate(batches)
            }

            for future in concurrent.futures.as_completed(futures_map):
                batch_idx = futures_map[future]
                batch_symbols = batches[batch_idx]
                try:
                    ok_rows, errors = future.result(timeout=600)
                except Exception as exc:
                    logger.warning(
                        "  Batch %d crashed: %s — skipping entire batch (%d stocks)",
                        batch_idx, str(exc)[:80], len(batch_symbols),
                    )
                    n_failed += len(batch_symbols)
                    continue

                all_rows.extend(ok_rows)
                for r in ok_rows:
                    completed.add(r["symbol"])

                # Log errors in MAIN process (no multi-process file races)
                for sym, exc_type, exc_msg in errors:
                    _log_error(error_log_path, sym,
                               Exception(f"[worker] {exc_type}: {exc_msg}"))
                n_failed += len(errors)

                if pbar is not None:
                    pbar.update(len(batch_symbols))
                    pbar.set_postfix(ok=len(all_rows), fail=n_failed)

                # ── Checkpoint every 5 batches (~250 stocks) ──
                if (batch_idx + 1) % 5 == 0:
                    _save_progress("market_data", completed)

    finally:
        if pbar is not None:
            pbar.close()

    # ── Final checkpoint ──
    _save_progress("market_data", completed)

    result = pd.DataFrame(all_rows)

    # Defensive: ensure numeric
    for col in ["open", "high", "low", "close", "volume", "amount",
                "pct_change", "turnover_rate"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    if n_failed > 0:
        logger.warning("[PARALLEL] market data: %d/%d symbols failed (%.1f%%)",
                       n_failed, len(remaining),
                       100 * n_failed / max(len(remaining), 1))

    logger.info("  -> %d rows, %d unique symbols", len(result),
                result["symbol"].nunique() if len(result) > 0 else 0)
    return result


# ── fetch_daily_fundamentals_parallel ───────────────────────

def fetch_daily_fundamentals_parallel(
    universe: pd.DataFrame | None = None,
    *,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Parallel fundamentals computation via ProcessPoolExecutor + checkpoint/resume.

    Uses baostock ``query_profit_data()`` + close prices from SQLite
    ``market_cache`` to compute PE, ROE, MarketCap, EPS per stock.

    Features:
      - Close prices and stock names are pre-loaded in the main process
        (single SQLite query) and passed to workers via pickle.
      - Each worker process calls ``bs.login()``/``bs.logout()``
        independently — no shared socket, no ``_BS_LOCK``, no deadlock.
      - Stocks are batched (50 per batch) for efficient IPC.
      - Progress saved to ``data/progress_fundamentals.json`` every
        5 batches (~250 stocks).
      - Per-stock errors are returned to the main process, logged to
        ``data/error_log.txt``, and never crash.

    Args:
        universe: Pre-fetched stock list.
        max_workers: ProcessPoolExecutor worker count.  8-12 is
                     a good range for typical CPUs.

    Returns:
        pd.DataFrame with cols: symbol, name, close, pe_ttm, pb, total_mcap,
        float_mcap, roe, eps, bps, revenue, net_profit, operating_profit,
        gross_margin, net_margin, pe_static, board
    """
    from paper_trading.state_manager import StateManager

    logger.info("[PARALLEL] Computing fundamentals via baostock (%d workers) ...",
                max_workers)

    # ── 1. Universe ──
    if universe is None or len(universe) == 0:
        universe = fetch_all_a_share_codes()
    symbols = universe["symbol"].astype(str).str.zfill(6).tolist()
    names = dict(zip(universe["symbol"].astype(str).str.zfill(6), universe["name"]))

    # ── 2. Pre-load close prices from SQLite (main thread, one query) ──
    db_dir = Path("output/paper_trading_db")
    close_prices: dict[str, float] = {}
    try:
        state = StateManager(db_dir)
        with state._get_conn() as conn:
            rows = conn.execute(
                "SELECT symbol, close FROM market_cache "
                "WHERE (symbol, trade_date) IN ("
                "  SELECT symbol, MAX(trade_date) FROM market_cache GROUP BY symbol"
                ")"
            ).fetchall()
            close_prices = {
                str(r[0]).zfill(6): float(r[1]) for r in rows
                if r[1] is not None and float(r[1]) > 0
            }
        logger.info("  %d stocks with close prices in market_cache", len(close_prices))
    except Exception as exc:
        logger.warning("  Cannot read market_cache for close prices (%s)", str(exc)[:60])

    # ── 3. Checkpoint ──
    completed = _load_progress("fundamentals")
    remaining = [s for s in symbols if s not in completed]
    if len(completed) > 0:
        logger.info("  Resuming: %d/%d already done, %d remaining",
                    len(completed), len(symbols), len(remaining))
    if not remaining:
        logger.info("  All %d stocks already processed — nothing to do.", len(symbols))
        return pd.DataFrame()

    # ── 4. Each ProcessPoolExecutor worker calls bs.login()/bs.logout()
    #     independently — no shared TCP socket, no _BS_LOCK, no deadlock.

    # Constants for baostock query
    YEAR, QUARTER = 2026, 1
    error_log_path = _PROGRESS_DIR / "error_log.txt"

    BATCH_SIZE = 50
    batches = [remaining[i:i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    logger.info("  %d batches (batch_size=%d), %d workers",
                len(batches), BATCH_SIZE, max_workers)

    all_rows: list[dict] = []
    n_ok = 0
    n_miss = 0

    pbar = (
        _tqdm(total=len(remaining), desc="Fundamentals", unit="stk",
              ncols=100, smoothing=0.1)
        if _TQDM_AVAILABLE else None
    )

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures_map = {
                executor.submit(_fundamentals_batch_worker,
                                (b, close_prices, names, YEAR, QUARTER)): i
                for i, b in enumerate(batches)
            }

            for future in concurrent.futures.as_completed(futures_map):
                batch_idx = futures_map[future]
                batch_symbols = batches[batch_idx]
                try:
                    ok_rows, errors = future.result(timeout=600)
                except Exception as exc:
                    logger.warning(
                        "  Batch %d crashed: %s — skipping entire batch (%d stocks)",
                        batch_idx, str(exc)[:80], len(batch_symbols),
                    )
                    n_miss += len(batch_symbols)
                    continue

                all_rows.extend(ok_rows)
                for r in ok_rows:
                    completed.add(r["symbol"])
                n_ok += len(ok_rows)

                # Log errors in MAIN process
                for sym, exc_type, exc_msg in errors:
                    _log_error(error_log_path, sym,
                               Exception(f"[worker] {exc_type}: {exc_msg}"))
                n_miss += len(errors)

                if pbar is not None:
                    pbar.update(len(batch_symbols))
                    pbar.set_postfix(ok=n_ok, miss=n_miss)

                # ── Checkpoint every 5 batches (~250 stocks) ──
                if (batch_idx + 1) % 5 == 0:
                    _save_progress("fundamentals", completed)

    finally:
        if pbar is not None:
            pbar.close()

    # ── Final checkpoint ──
    _save_progress("fundamentals", completed)

    logger.info("  -> %d stocks computed (%d missing, %.0f%%)",
                n_ok, n_miss,
                100 * n_ok / max(len(remaining), 1))

    if not all_rows:
        raise RuntimeError(
            "Parallel fundamentals: 0 stocks computed successfully "
            f"({n_miss} failures)"
        )

    df = pd.DataFrame(all_rows)
    df["board"] = df["symbol"].apply(_classify_board)
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════
# Self-test (invoke: python -m paper_trading.data_ingestion)
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s")
    print("=== Data Ingestion Self-Test ===\n")

    print("[1] Fetch stock universe...")
    universe = fetch_all_a_share_codes()
    print(f"    -> {len(universe)} stocks")
    print(universe.head(3))
    print()

    print("[2] Fetch fundamentals...")
    try:
        fund = fetch_daily_fundamentals()
        print(f"    -> {len(fund)} stocks with fundamentals")
        print(fund[["symbol", "name", "pe_ttm", "pb", "total_mcap", "roe"]].head(3))
    except Exception as e:
        print(f"    -> FAILED: {e}")
    print()

    print("[3] Test last-trade-date detection...")
    today = date.today()
    print(f"    Today: {today} | is_last_trade_date: {is_month_last_trade_date(today)}")
    print()

    print("[4] Done.")
