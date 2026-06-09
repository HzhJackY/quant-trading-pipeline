"""
Lightweight State Manager — SQLite-backed persistence for paper trading.

Two core state tables:

  1. market_cache       : Rolling 60-trading-day OHLCV buffer.
     ┌──────────┬────────┬───────┬───────┬──────┬──────┬────────┬────────┐
     │ trade_date │ symbol│ open  │ high  │ low  │ close│ volume │ amount │
     └──────────┴────────┴───────┴───────┴──────┴──────┴────────┴────────┘
     - Partitioned by trade_date.
     - Auto-pruned: >60 days deleted on each write.
     - Used to compute Vol_20D, Mom_1M, Mom_3M, etc. at month-end.

  2. signal_anchor     : Month-end alpha_signal persistence (prev_signal source).
     ┌──────────┬────────┬───────────────┬─────────────────┐
     │ ym        │ symbol │ alpha_signal  │ model_timestamp │
     └──────────┴────────┴───────────────┴─────────────────┘
     - ym = "YYYY-MM" (e.g., "2026-05")
     - Each month-end rebalance writes exactly one row per stock held.
     - On next rebalance: read WHERE ym = <previous_month>.

Design invariants:
  - SQLite WAL mode for concurrent reads.
  - write-ahead logging ensures crash safety.
  - No ORM — raw sqlite3 for zero-dependency transparency.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger("state_manager")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════
# StateSchema
# ═══════════════════════════════════════════════════════════

class StateSchema:
    """DDL for SQLite state tables."""

    MARKET_CACHE_DDL = """
    CREATE TABLE IF NOT EXISTS market_cache (
        trade_date  TEXT    NOT NULL,   -- "YYYY-MM-DD"
        symbol      TEXT    NOT NULL,   -- 6-digit code
        open        REAL,
        high        REAL,
        low         REAL,
        close       REAL,
        volume      REAL,
        amount      REAL,
        pct_change  REAL,
        turnover_rate REAL,
        PRIMARY KEY (trade_date, symbol)
    ) WITHOUT ROWID;
    """

    SIGNAL_ANCHOR_DDL = """
    CREATE TABLE IF NOT EXISTS signal_anchor (
        ym          TEXT    NOT NULL,   -- "YYYY-MM"
        symbol      TEXT    NOT NULL,   -- 6-digit code
        alpha_signal REAL  NOT NULL,    -- ∈ [0, 1]
        model_timestamp TEXT,           -- ISO timestamp of inference
        PRIMARY KEY (ym, symbol)
    ) WITHOUT ROWID;
    """

    INDUSTRY_CACHE_DDL = """
    CREATE TABLE IF NOT EXISTS industry_cache (
        symbol          TEXT PRIMARY KEY,   -- 6-digit code
        industry_name   TEXT    NOT NULL,   -- SW first-level industry
        updated         TEXT    NOT NULL    -- ISO timestamp of last fetch
    ) WITHOUT ROWID;
    """

    # Metadata table: schema version, last run, etc.
    META_DDL = """
    CREATE TABLE IF NOT EXISTS meta (
        key     TEXT PRIMARY KEY,
        value   TEXT
    );
    """

    ALL_DDL = [MARKET_CACHE_DDL, SIGNAL_ANCHOR_DDL, INDUSTRY_CACHE_DDL, META_DDL]

    SCHEMA_VERSION = "1.0"

    # Indices for fast lookups
    INDICES = [
        "CREATE INDEX IF NOT EXISTS idx_market_date ON market_cache(trade_date);",
        "CREATE INDEX IF NOT EXISTS idx_market_sym  ON market_cache(symbol);",
        "CREATE INDEX IF NOT EXISTS idx_signal_ym   ON signal_anchor(ym);",
        "CREATE INDEX IF NOT EXISTS idx_industry_name ON industry_cache(industry_name);",
    ]


# ═══════════════════════════════════════════════════════════
# StateManager
# ═══════════════════════════════════════════════════════════

class StateManager:
    """
    Thread-safe SQLite state manager for paper trading.

    Usage:
        sm = StateManager("output/paper_trading_db")
        sm.init()

        # Daily: ingest today's OHLCV
        sm.append_market_data(daily_df)

        # Month-end: read cached data, read prev signal, write new signal
        cache_df = sm.query_market_cache(lookback_days=60)
        prev_signal = sm.get_prev_signal(ym="2026-05")
        sm.write_signal_anchor(ym="2026-06", signals=signal_series)
    """

    MAX_CACHE_DAYS = 60          # Rolling window
    MAX_CACHE_DAYS_MARGIN = 70   # Keep a few extra before pruning

    def __init__(self, db_dir: str | Path = "output/paper_trading_db"):
        self._db_dir = Path(db_dir)
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._db_dir / "state.db"
        self._lock = threading.Lock()
        self._initialized = False

    # ── Lifecycle ──────────────────────────────────────────

    def init(self):
        """Initialize database schema. Idempotent — safe to call repeatedly."""
        with self._get_conn() as conn:
            for ddl in StateSchema.ALL_DDL:
                conn.execute(ddl)
            for idx_ddl in StateSchema.INDICES:
                conn.execute(idx_ddl)
            # Schema version
            conn.execute(
                "INSERT OR REPLACE INTO meta VALUES (?, ?)",
                ("schema_version", StateSchema.SCHEMA_VERSION),
            )
            conn.commit()
        self._initialized = True
        logger.info("StateManager initialized: %s (v%s)", self._db_path, StateSchema.SCHEMA_VERSION)

    def close(self):
        """No-op for SQLite; connection is per-operation."""
        pass

    # ── Market Cache ───────────────────────────────────────

    def append_market_data(self, df: pd.DataFrame):
        """
        Append daily OHLCV rows to the market cache.

        Args:
            df: pd.DataFrame with columns:
                [date, symbol, open, high, low, close, volume, amount,
                 pct_change, turnover_rate]
                'date' can be str, datetime, or Timestamp.
        """
        if df is None or len(df) == 0:
            return

        required = {"date", "symbol", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        rows = []
        for _, row in df.iterrows():
            trade_date = _to_date_str(row["date"])
            symbol = str(row["symbol"]).zfill(6)
            rows.append((
                trade_date,
                symbol,
                float(row.get("open", np.nan)) if pd.notna(row.get("open", np.nan)) else None,
                float(row.get("high", np.nan)) if pd.notna(row.get("high", np.nan)) else None,
                float(row.get("low", np.nan)) if pd.notna(row.get("low", np.nan)) else None,
                float(row["close"]) if pd.notna(row["close"]) else None,
                float(row.get("volume", np.nan)) if pd.notna(row.get("volume", np.nan)) else None,
                float(row.get("amount", np.nan)) if pd.notna(row.get("amount", np.nan)) else None,
                float(row.get("pct_change", np.nan)) if pd.notna(row.get("pct_change", np.nan)) else None,
                float(row.get("turnover_rate", np.nan)) if pd.notna(row.get("turnover_rate", np.nan)) else None,
            ))

        with self._get_conn() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO market_cache
                   (trade_date, symbol, open, high, low, close, volume, amount, pct_change, turnover_rate)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()

        n_dates = self._count_distinct_dates()
        logger.info("Appended %d rows → market_cache (now %d dates)", len(rows), n_dates)

        # Auto-prune if exceeding threshold
        if n_dates > self.MAX_CACHE_DAYS_MARGIN:
            self._prune_old_dates()

    def query_market_cache(
        self,
        lookback_days: int = 60,
        symbols: List[str] | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve up to `lookback_days` trading days of OHLCV data.

        Args:
            lookback_days: Number of recent trading days to retrieve.
            symbols: Optional filter. If None, returns all symbols.

        Returns:
            pd.DataFrame with cols: trade_date, symbol, open, high, low,
            close, volume, amount, pct_change, turnover_rate
        """
        where_clauses = []
        params: list = []

        # Get the N most recent trade dates
        recent_dates = self._get_recent_dates(lookback_days)
        if not recent_dates:
            return pd.DataFrame()

        placeholders = ",".join("?" for _ in recent_dates)
        where_clauses.append(f"trade_date IN ({placeholders})")
        params.extend(recent_dates)

        if symbols:
            sym_placeholders = ",".join("?" for _ in symbols)
            where_clauses.append(f"symbol IN ({sym_placeholders})")
            params.extend(symbols)

        where = " AND ".join(where_clauses)
        query = f"SELECT * FROM market_cache WHERE {where} ORDER BY trade_date, symbol"

        with self._get_conn() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if len(df) == 0:
            return df

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    def get_latest_date(self) -> Optional[pd.Timestamp]:
        """Get the most recent trade_date in the cache."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT MAX(trade_date) FROM market_cache"
            ).fetchone()
        if row and row[0]:
            return pd.Timestamp(row[0])
        return None

    # ── Signal Anchor ──────────────────────────────────────

    def get_prev_signal(
        self,
        ym: str,
        symbols: List[str] | None = None,
    ) -> pd.Series:
        """
        Retrieve alpha_signal for a given month. Returns a pd.Series
        indexed by symbol with values ∈ [0, 1].

        Args:
            ym: "YYYY-MM" string, e.g., "2026-05".
            symbols: Optional symbol list. Used to align with current universe.
                     Missing symbols get fill_value=0.5.

        Returns:
            pd.Series with index=symbol, values=alpha_signal.
        """
        query = "SELECT symbol, alpha_signal FROM signal_anchor WHERE ym = ?"
        params = [ym]

        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            logger.warning("No prev_signal for ym=%s — will default to 0.5", ym)
            if symbols:
                return pd.Series(0.5, index=pd.Index(symbols, name="symbol"))
            return pd.Series(dtype=np.float64)

        result = pd.Series(
            {symbol: float(signal) for symbol, signal in rows},
            name="alpha_signal",
        )

        # Fill missing symbols with 0.5 (new listings, re-listings)
        if symbols:
            missing = set(symbols) - set(result.index)
            if missing:
                logger.info("  %d symbols not in prev anchor → fill 0.5", len(missing))
                fill = pd.Series(0.5, index=list(missing), name="alpha_signal")
                result = pd.concat([result, fill])

        result.index.name = "symbol"
        return result

    def get_latest_ym(self) -> Optional[str]:
        """Get the most recent month (YYYY-MM) in signal_anchor."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT MAX(ym) FROM signal_anchor").fetchone()
        if row and row[0]:
            return row[0]
        return None

    def write_signal_anchor(
        self,
        ym: str,
        signals: pd.Series | Dict[str, float],
        model_timestamp: str | None = None,
    ):
        """
        Write (or overwrite) month-end alpha signals.

        Args:
            ym: "YYYY-MM" string.
            signals: pd.Series (index=symbol) or dict {symbol: alpha}.
            model_timestamp: ISO timestamp of inference. Uses now() if None.
        """
        if model_timestamp is None:
            model_timestamp = datetime.now().isoformat()

        if isinstance(signals, pd.Series):
            items = signals.to_dict()
        else:
            items = signals

        rows = [
            (ym, str(sym).zfill(6), float(v), model_timestamp)
            for sym, v in items.items()
            if not np.isnan(v)
        ]

        with self._get_conn() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO signal_anchor
                   (ym, symbol, alpha_signal, model_timestamp)
                   VALUES (?, ?, ?, ?)""",
                rows,
            )
            conn.commit()

        logger.info("Wrote %d signals → signal_anchor (ym=%s)", len(rows), ym)

    # ── Industry Cache ─────────────────────────────────────

    def update_industry_cache(self, industry_df: pd.DataFrame):
        """
        Store SW industry classification in local SQLite cache.

        Only call once per month — the industry classification rarely changes
        intra-month. Subsequent calls within the same month are no-ops if
        `skip_if_fresh=True` and the cache was updated within 30 days.

        Args:
            industry_df: pd.DataFrame with cols [symbol, industry_name].
        """
        if industry_df is None or len(industry_df) == 0:
            logger.warning("Industry cache update: empty input — skipping")
            return

        required = {"symbol", "industry_name"}
        missing = required - set(industry_df.columns)
        if missing:
            raise ValueError(f"Industry cache: missing columns {missing}")

        now_iso = datetime.now().isoformat()
        rows = [
            (str(row["symbol"]).zfill(6), str(row["industry_name"]).strip(), now_iso)
            for _, row in industry_df.iterrows()
            if pd.notna(row["symbol"]) and pd.notna(row["industry_name"])
        ]

        with self._get_conn() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO industry_cache
                   (symbol, industry_name, updated) VALUES (?, ?, ?)""",
                rows,
            )
            conn.commit()

        n_industries = industry_df["industry_name"].nunique()
        logger.info("Industry cache updated: %d stocks, %d industries", len(rows), n_industries)

    def get_industry_map(self, symbols: list[str] | None = None) -> pd.Series:
        """
        Retrieve SW industry classification from local cache.

        Args:
            symbols: Optional stock list to filter. Missing symbols → "Others".

        Returns:
            pd.Series indexed by symbol, values = industry_name.
        """
        query = "SELECT symbol, industry_name FROM industry_cache"
        params: list = []

        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            query += f" WHERE symbol IN ({placeholders})"
            params = symbols

        with self._get_conn() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if len(df) == 0:
            if symbols:
                return pd.Series("Others", index=pd.Index(symbols, name="symbol"))
            return pd.Series(dtype=str)

        result = df.set_index("symbol")["industry_name"]

        # Fill missing symbols with "Others"
        if symbols:
            missing = set(symbols) - set(result.index)
            if missing:
                fill = pd.Series("Others", index=list(missing), name="industry_name")
                result = pd.concat([result, fill])

        result.index.name = "symbol"
        return result

    def is_industry_cache_fresh(self, max_age_days: int = 30) -> bool:
        """Check whether the industry cache was updated within max_age_days."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT MAX(updated) FROM industry_cache"
            ).fetchone()
        if row and row[0]:
            last_update = datetime.fromisoformat(row[0])
            age = (datetime.now() - last_update).days
            return age <= max_age_days
        return False

    # ── Internal helpers ───────────────────────────────────

    @contextmanager
    def _get_conn(self):
        """Get a thread-safe connection with WAL mode."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path), timeout=15.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA cache_size=-8000;")  # 8MB cache
            try:
                yield conn
            finally:
                conn.close()

    def _count_distinct_dates(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT trade_date) FROM market_cache"
            ).fetchone()
        return row[0] if row else 0

    def _get_recent_dates(self, n: int) -> List[str]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT trade_date FROM market_cache "
                "ORDER BY trade_date DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [r[0] for r in rows]

    def _prune_old_dates(self):
        """Delete rows older than MAX_CACHE_DAYS trading days."""
        cutoff_date = self._get_nth_recent_date_str(self.MAX_CACHE_DAYS)
        if cutoff_date is None:
            return

        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM market_cache WHERE trade_date < ?",
                (cutoff_date,),
            )
            n_deleted = cursor.rowcount
            conn.commit()

        if n_deleted > 0:
            logger.info("Pruned %d rows (trade_date < %s)", n_deleted, cutoff_date)

    def _get_nth_recent_date_str(self, n: int) -> Optional[str]:
        """Return the trade_date string for the Nth most recent date."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT trade_date FROM market_cache "
                "ORDER BY trade_date DESC LIMIT 1 OFFSET ?",
                (n - 1,),
            ).fetchall()
        return rows[0][0] if rows else None

    # ── Stats / Debug ──────────────────────────────────────

    def stats(self) -> dict:
        """Return summary stats for monitoring."""
        with self._get_conn() as conn:
            n_market_rows = conn.execute(
                "SELECT COUNT(*) FROM market_cache"
            ).fetchone()[0]
            n_market_dates = conn.execute(
                "SELECT COUNT(DISTINCT trade_date) FROM market_cache"
            ).fetchone()[0]
            n_signal_months = conn.execute(
                "SELECT COUNT(DISTINCT ym) FROM signal_anchor"
            ).fetchone()[0]
            latest_signal = conn.execute(
                "SELECT MAX(ym) FROM signal_anchor"
            ).fetchone()[0]

        return {
            "market_rows": n_market_rows,
            "market_dates": n_market_dates,
            "signal_months": n_signal_months,
            "latest_signal_ym": latest_signal,
            "db_path": str(self._db_path),
        }


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _to_date_str(val) -> str:
    """Convert any date-like value to 'YYYY-MM-DD'."""
    if isinstance(val, pd.Timestamp):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, (datetime, date)):
        return val.strftime("%Y-%m-%d")
    s = str(val)
    # Try common formats
    for fmt in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(s[:10], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s[:10]  # fallback


def ym_from_date(d: date | str | pd.Timestamp) -> str:
    """Convert date to 'YYYY-MM' format."""
    if isinstance(d, pd.Timestamp):
        return d.strftime("%Y-%m")
    if isinstance(d, datetime):
        return d.strftime("%Y-%m")
    if isinstance(d, date):
        return d.strftime("%Y-%m")
    s = str(d)[:7]  # "2026-06-07" → "2026-06"
    return s


def prev_ym(ym: str) -> str:
    """Return the previous month in 'YYYY-MM' format."""
    year, month = int(ym[:4]), int(ym[5:7])
    if month == 1:
        return f"{year - 1}-12"
    return f"{year}-{month - 1:02d}"


# ═══════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile, os

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s")

    with tempfile.TemporaryDirectory() as tmpdir:
        sm = StateManager(tmpdir)
        sm.init()

        # Simulate daily ingestion
        dates = pd.date_range("2026-05-01", "2026-06-05", freq="B")
        symbols = ["000001", "600519", "300750"]
        for i, d in enumerate(dates):
            df = pd.DataFrame({
                "date": [d] * len(symbols),
                "symbol": symbols,
                "open": 10.0 + i,
                "high": 10.5 + i,
                "low": 9.8 + i,
                "close": 10.2 + i,
                "volume": 1e6,
                "amount": 1e7,
                "pct_change": 0.02,
                "turnover_rate": 0.5,
            })
            sm.append_market_data(df)

        # Query cache
        cache = sm.query_market_cache(lookback_days=30)
        print(f"\nCache: {len(cache)} rows, {cache['trade_date'].nunique()} dates")
        print(cache.head())

        # Write signal anchor
        signals = pd.Series([0.45, 0.72, 0.33], index=symbols)
        sm.write_signal_anchor("2026-05", signals)

        # Read prev signal
        prev = sm.get_prev_signal("2026-05")
        print(f"\nPrev signal for 2026-05:\n{prev}")

        # Stats
        print(f"\nStats: {sm.stats()}")

        print("\n[OK] All state_manager tests passed.")
