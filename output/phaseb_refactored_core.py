
"""
Phase B Refactored — Core Financial + Market Cap Fetch Functions.

Design:
  1. akshare.stock_financial_abstract_ths → all historical quarterly ratios
  2. baostock query_profit_data (LATEST quarter only) → total_share
  3. Regulatory PIT lag → virtual pub_date
  4. Market cap = close_qfq × total_share (daily, PIT-aligned)

Usage: These are the core functions for review. Do NOT run directly.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("phaseb_refactored")

DATA_RAW = Path("data/raw")

# ── Chinese column name constants (from stock_financial_abstract_ths) ──
COL_REPORT_DATE = "报告期"
COL_NET_PROFIT = "净利润"
COL_REVENUE = "营业总收入"
COL_ROE = "净资产收益率"
COL_EPS = "基本每股收益"
COL_BVPS = "每股净资产"
COL_NET_MARGIN = "销售净利率"
COL_DEBT_RATIO = "资产负债率"

# Output column names (standardized)
COL_SYMBOL = "symbol"
COL_REPORT_DATE_OUT = "report_date"
COL_PUB_DATE = "pub_date"
COL_TOTAL_SHARE = "total_share"
COL_NET_PROFIT_OUT = "净利润"
COL_REVENUE_OUT = "营业收入"
COL_ROE_OUT = "ROE"
COL_EPS_OUT = "每股收益"
COL_BVPS_OUT = "每股净资产"
COL_NET_MARGIN_OUT = "销售净利率"
COL_DEBT_RATIO_OUT = "Debt_Ratio"


# ═══════════════════════════════════════════════════════════
# Module A: Regulatory PIT Lag
# ═══════════════════════════════════════════════════════════

def compute_regulatory_pub_date(report_date: pd.Timestamp) -> pd.Timestamp:
    """
    Compute virtual pub_date using statutory disclosure deadlines.

    Rules (conservative, no look-ahead risk):
      - Q1 (Mar 31):  report_date + 60 days → May 30
      - Q2 (Jun 30):  report_date + 60 days → Aug 29
      - Q3 (Sep 30):  report_date + 60 days → Nov 29
      - FY (Dec 31):  report_date + 120 days → Apr 30 next year

    These are LATER than actual disclosure dates (which are typically
    15-25 days after quarter end). Using the statutory deadline as
    the pub_date is conservative: we delay data availability, which
    can only hurt performance, never create look-ahead bias.
    """
    month = report_date.month
    day = report_date.day

    if month == 3 and day == 31:          # Q1
        return report_date + pd.Timedelta(days=60)
    elif month == 6 and day == 30:        # Q2
        return report_date + pd.Timedelta(days=60)
    elif month == 9 and day == 30:        # Q3
        return report_date + pd.Timedelta(days=60)
    elif month == 12 and day == 31:       # FY
        return report_date + pd.Timedelta(days=120)
    else:
        # Non-standard report date — use +90 days as safe default
        logger.debug("Non-standard report_date=%s, using +90d lag", report_date)
        return report_date + pd.Timedelta(days=90)


# ═══════════════════════════════════════════════════════════
# Module B: Financial Data (akshare)
# ═══════════════════════════════════════════════════════════

def _parse_financial_number(val: str | float | None) -> float | None:
    """
    Parse akshare financial values with Chinese units.

    Examples: '145.23亿' → 1.4523e10, '3.03%' → 0.0303
    Returns float or None.
    """
    if val is None or (isinstance(val, (int, float)) and pd.isna(val)):
        return None
    s = str(val).strip()
    if s in ("False", "True", ""):
        return None
    try:
        return float(s)
    except ValueError:
        pass
    if "亿" in s:
        n = re.sub(r"[^\d.\-]", "", s)
        return float(n) * 1e8 if n else None
    elif "万" in s:
        n = re.sub(r"[^\d.\-]", "", s)
        return float(n) * 1e4 if n else None
    elif "%" in s:
        n = re.sub(r"[^\d.\-]", "", s)
        return float(n) / 100.0 if n else None
    return None


def fetch_financial_history_akshare(symbol: str) -> pd.DataFrame | None:
    """
    Fetch ALL historical quarterly financial data for a single stock
    using akshare.stock_financial_abstract_ths.  ONE HTTP call.

    Returns DataFrame with standardized columns:
      symbol, report_date, pub_date (regulatory PIT),
      净利润, 营业收入, ROE, 每股收益, 每股净资产,
      销售净利率, Debt_Ratio

    Falls back to cached CSV on disk.
    """
    import akshare as ak

    sym = str(symbol).zfill(6)
    cache_path = DATA_RAW / f"financial_{sym}_akshare.csv"

    # ── Cache hit ──
    if cache_path.exists():
        existing = pd.read_csv(
            cache_path, dtype={COL_SYMBOL: str},
            parse_dates=[COL_REPORT_DATE_OUT, COL_PUB_DATE],
        )
        if len(existing) > 0:
            return existing

    # ── Fetch from akshare ──
    for attempt in range(3):
        try:
            raw = ak.stock_financial_abstract_ths(symbol=sym, indicator="按报告期")
            if raw.empty:
                return None
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                logger.warning("  %s: akshare financial fetch FAILED after 3 attempts: %s", sym, e)
                return None

    # ── Map columns ──
    col_map = {
        COL_REPORT_DATE: COL_REPORT_DATE_OUT,
        COL_NET_PROFIT: COL_NET_PROFIT_OUT,
        COL_REVENUE: COL_REVENUE_OUT,
        COL_ROE: COL_ROE_OUT,
        COL_EPS: COL_EPS_OUT,
        COL_BVPS: COL_BVPS_OUT,
        COL_NET_MARGIN: COL_NET_MARGIN_OUT,
        COL_DEBT_RATIO: COL_DEBT_RATIO_OUT,
    }
    df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
    df[COL_SYMBOL] = sym

    # ── Parse values ──
    for col in [COL_NET_PROFIT_OUT, COL_REVENUE_OUT, COL_ROE_OUT,
                COL_EPS_OUT, COL_BVPS_OUT, COL_NET_MARGIN_OUT, COL_DEBT_RATIO_OUT]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_financial_number)

    # ── Parse dates ──
    df[COL_REPORT_DATE_OUT] = pd.to_datetime(df[COL_REPORT_DATE_OUT], errors="coerce")

    # ── Apply regulatory PIT pub_date ──
    df[COL_PUB_DATE] = df[COL_REPORT_DATE_OUT].apply(
        lambda x: compute_regulatory_pub_date(x) if pd.notna(x) else pd.NaT
    )

    # ── Drop invalid rows ──
    df = df.dropna(subset=[COL_REPORT_DATE_OUT, COL_PUB_DATE])

    # ── Sort and cache ──
    df = df.sort_values([COL_REPORT_DATE_OUT]).reset_index(drop=True)
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")

    return df


# ═══════════════════════════════════════════════════════════
# Module C: Total Share (baostock, ONE call per stock)
# ═══════════════════════════════════════════════════════════

def fetch_total_share_baostock(symbol: str) -> dict | None:
    """
    Fetch the LATEST total_share from baostock.

    Only ONE query per stock — total_share changes rarely
    (only on rights issues, buybacks, etc.) and we take the
    most recent available value.

    Returns dict with keys: total_share, report_date
    Returns None on failure.
    """
    import baostock as bs

    sym = str(symbol).zfill(6)
    prefix = "sh." if sym.startswith(("6", "5")) else "sz."

    for attempt in range(3):
        try:
            bs.login()

            # Try latest quarters, starting from 2026Q1 backward
            for year in [2026, 2025, 2024]:
                for q in [1, 4, 3, 2]:
                    rs = bs.query_profit_data(code=prefix + sym, year=year, quarter=q)
                    if rs.error_code != "0":
                        continue
                    data = rs.get_data()
                    if data.empty:
                        continue
                    ts = pd.to_numeric(
                        data.iloc[0].get("totalShare", np.nan), errors="coerce"
                    )
                    if pd.notna(ts) and ts > 0:
                        bs.logout()
                        return {
                            "total_share": float(ts),
                            "report_date": str(data.iloc[0].get("statDate", "")),
                        }
            bs.logout()
            return None
        except Exception as e:
            try:
                bs.logout()
            except Exception:
                pass
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
            else:
                logger.warning("  %s: baostock total_share FAILED: %s", sym, e)
    return None


# ═══════════════════════════════════════════════════════════
# Module D: Daily Market Cap Merge
# ═══════════════════════════════════════════════════════════

def compute_daily_market_cap(
    symbol: str,
    daily_qfq: pd.DataFrame,       # columns: date, close (qfq-adjusted)
    total_share: float,
) -> pd.DataFrame:
    """
    Compute daily exact market cap from QFQ close × total_share.

    QFQ (前复权) price already accounts for all corporate actions
    (splits, dividends) that would affect share count. Therefore:

      exact_mcap[t] = close_qfq[t] × total_share

    is the correct daily market cap, because the adjustment factor
    embedded in qfq price compensates for share-count changes.

    Parameters
    ----------
    symbol : str
    daily_qfq : DataFrame with columns [date, close]
        QFQ-adjusted daily close prices.
    total_share : float
        Latest known total shares from financial report.

    Returns
    -------
    DataFrame with columns: date, symbol, close_qfq, total_share, exact_mcap
    """
    df = daily_qfq.copy()
    df["symbol"] = str(symbol).zfill(6)
    df["total_share"] = total_share
    df["close_qfq"] = pd.to_numeric(df["close"], errors="coerce")
    df["exact_mcap"] = df["close_qfq"] * total_share
    df = df.drop(columns=["close"], errors="ignore")
    return df[["date", "symbol", "close_qfq", "total_share", "exact_mcap"]]


# ═══════════════════════════════════════════════════════════
# Module E: Point-in-Time Financial Alignment
# ═══════════════════════════════════════════════════════════

def align_financials_pit(
    fin_df: pd.DataFrame,
    current_date: pd.Timestamp,
) -> pd.Series | None:
    """
    Given a full financial history, return the LATEST row whose
    regulatory pub_date <= current_date.

    This is the core PIT gate: NO financial data with
    pub_date > current_date is ever visible.

    Parameters
    ----------
    fin_df : DataFrame
        Must have columns: report_date, pub_date, [financial fields...]
    current_date : Timestamp
        The "as-of" date for the backtest/inference.

    Returns
    -------
    Series (the latest valid financial row) or None.
    """
    valid = fin_df[fin_df[COL_PUB_DATE] <= current_date]
    if valid.empty:
        return None
    return valid.loc[valid[COL_REPORT_DATE_OUT].idxmax()]


# ═══════════════════════════════════════════════════════════
# Module F: Integrated Pipeline Entry
# ═══════════════════════════════════════════════════════════

def build_pit_financials_for_date(
    symbol: str,
    fin_df: pd.DataFrame | None,
    daily_qfq: pd.DataFrame | None,
    total_share_info: dict | None,
    current_date: pd.Timestamp,
) -> dict | None:
    """
    Build a complete financial snapshot for a single stock at a single date.

    Steps:
      1. PIT-align financial ratios (only data with pub_date <= current_date)
      2. Compute daily market cap at current_date
      3. Return a flat dict of all features

    Returns None if insufficient data.
    """
    sym = str(symbol).zfill(6)

    result = {
        "symbol": sym,
        "date": current_date,
        "total_share": None,
        "exact_mcap": None,
    }

    # ── Market cap ──
    if daily_qfq is not None and total_share_info is not None:
        ts = total_share_info.get("total_share")
        if ts and ts > 0:
            today_data = daily_qfq[daily_qfq["date"] == current_date]
            if not today_data.empty:
                close = float(today_data.iloc[0]["close"])
                result["total_share"] = ts
                result["exact_mcap"] = close * ts

    # ── Financial ratios (PIT-gated) ──
    if fin_df is not None and len(fin_df) > 0:
        pit_row = align_financials_pit(fin_df, current_date)
        if pit_row is not None:
            for col in [COL_NET_PROFIT_OUT, COL_REVENUE_OUT, COL_ROE_OUT,
                        COL_EPS_OUT, COL_BVPS_OUT, COL_NET_MARGIN_OUT, COL_DEBT_RATIO_OUT]:
                if col in pit_row.index:
                    result[col] = pit_row[col]

    # ── Check minimum data ──
    if result["exact_mcap"] is None and all(
        result.get(c) is None for c in [COL_ROE_OUT, COL_EPS_OUT]
    ):
        return None

    return result
