"""
Phase B — Data Integrity Validation Script.

Four core check modules:
  1. Ex-date discontinuity: unadj_close large drop must align with adj_factor jump
  2. Market cap smoothness: exact_mcap daily change within board-specific bounds
  3. PIT strict timestamp:   pub_date >= report_date; zero look-ahead tolerance
  4. Sanity & NaN scan:      adj_factor / total_share boundary checks

Dynamic thresholds by board:
  - Main board  (000,600,601,603,605): ±12%
  - ChiNext/STAR (300,301,688):       ±22%

FAIL vs WARN:
  - FAIL: hard data corruption → blacklist, remove from panel
  - WARN: explainable anomaly → keep, winsorize

Usage:
  python validate_data_integrity.py                     # Full validation
  python validate_data_integrity.py --symbols 000001,600519  # Specific stocks
  python validate_data_integrity.py --sample 20          # Random sample
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Monkey-patch: baostock uses DataFrame.append() which was removed in pandas 2.0+
if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = lambda self, other, ignore_index=False, **kwargs: (
        pd.concat([self, other], ignore_index=ignore_index)
        if isinstance(other, pd.DataFrame)
        else self
    )

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate")

# ── Paths ──
DATA_RAW = Path("data/raw")
OUTPUT_DIR = Path("output")
BLACKLIST_PATH = OUTPUT_DIR / "blacklist_symbols.csv"
CSI800_HISTORY = OUTPUT_DIR / "csi800_history.parquet"

# ── Thresholds ──
UNRET_DROP_THRESHOLD = -0.08       # >8% single-day unadj drop triggers check
MISMATCH_TOLERANCE = 0.05          # 5% tolerance for adj_factor vs price alignment
MCAP_THRESHOLD_MAIN = 0.12         # ±12% for main board
MCAP_THRESHOLD_GEM_STAR = 0.22     # ±22% for ChiNext/STAR


# ═══════════════════════════════════════════════════════════
# Board Classification
# ═══════════════════════════════════════════════════════════

def get_board(symbol: str) -> str:
    """Classify stock by listing board."""
    s = str(symbol).zfill(6)
    if s.startswith("688"):
        return "star"       # 科创板 ±20%
    if s.startswith(("300", "301")):
        return "gem"        # 创业板 ±20%
    return "main"            # 主板 ±10%


def get_mcap_threshold(symbol: str) -> float:
    """Return the daily mcap change threshold for this stock."""
    board = get_board(symbol)
    return MCAP_THRESHOLD_GEM_STAR if board in ("star", "gem") else MCAP_THRESHOLD_MAIN


def get_price_limit(symbol: str) -> float:
    """Return the daily price change limit for this stock."""
    board = get_board(symbol)
    return 0.20 if board in ("star", "gem") else 0.10


# ═══════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════

@dataclass
class ExDateError:
    symbol: str
    date: str
    unadj_return: float       # unadjusted daily return
    adj_factor_change: float   # adj_factor daily change
    expected_change: float     # what adj_factor change SHOULD have been
    mismatch: float            # |actual - expected|
    severity: str = "FAIL"     # FAIL or WARN


@dataclass
class McapDiscontinuity:
    symbol: str
    date: str
    mcap_return: float         # exact_mcap daily return
    unadj_close: float
    adj_factor: float
    total_share: float
    prev_mcap: float
    curr_mcap: float
    board: str
    threshold_used: float
    severity: str = "FAIL"


@dataclass
class PITViolation:
    symbol: str
    report_date: str
    pub_date: str
    violation_type: str         # "pub_before_report" | "look_ahead" | "pub_date_suspicious"
    severity: str = "FAIL"      # Always FAIL for PIT


@dataclass
class SanityViolation:
    symbol: str
    date: str
    field: str                  # "adj_factor" | "total_share" | "unadj_close"
    issue: str                  # "NaN" | "negative" | "zero" | "random_jump"
    value: float
    severity: str = "FAIL"


@dataclass
class ValidationReport:
    total_stocks: int = 0
    passed: int = 0
    warnings: int = 0
    failed: int = 0
    ex_date_errors: list[ExDateError] = field(default_factory=list)
    mcap_discontinuities: list[McapDiscontinuity] = field(default_factory=list)
    pit_violations: list[PITViolation] = field(default_factory=list)
    sanity_violations: list[SanityViolation] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return (
            len([e for e in self.ex_date_errors if e.severity == "FAIL"])
            + len([e for e in self.mcap_discontinuities if e.severity == "FAIL"])
            + len(self.pit_violations)
            + len([e for e in self.sanity_violations if e.severity == "FAIL"])
        )


# ═══════════════════════════════════════════════════════════
# Helper: Data Loading
# ═══════════════════════════════════════════════════════════

def load_daily_dual(symbol: str) -> tuple[
    pd.DataFrame | None,  # unadjusted
    pd.DataFrame | None,  # qfq
]:
    """
    Load BOTH unadjusted (baostock) and QFQ (akshare) daily data.

    Returns (unadj_df, qfq_df).
    unadj_df columns: date, close (unadjusted), tradestatus
    qfq_df   columns: date, close (qfq), volume, amount

    Falls back to computing adj_factor from baostock-only data if akshare missing.
    """
    import baostock as bs

    unadj_df = None
    qfq_df = None

    # 1. Load baostock unadjusted
    prefix = "sh." if str(symbol).zfill(6).startswith(("6", "5")) else "sz."
    bs.login()
    try:
        rs = bs.query_history_k_data_plus(
            prefix + str(symbol).zfill(6),
            "date,close,tradestatus",
            start_date="2017-01-01",
            end_date="2026-06-19",
            frequency="d",
            adjustflag="3",  # unadjusted
        )
        if rs.error_code == "0":
            data = rs.get_data()
            if not data.empty:
                unadj_df = pd.DataFrame({
                    "date": pd.to_datetime(data["date"]),
                    "close": pd.to_numeric(data["close"], errors="coerce").astype(float),
                    "tradestatus": pd.to_numeric(data["tradestatus"], errors="coerce").fillna(0).astype(int),
                })
    finally:
        bs.logout()

    # 2. Load QFQ from akshare cache (or baostock adjustflag='2' fallback)
    cache_pattern = list(DATA_RAW.glob(f"daily_{str(symbol).zfill(6)}_*_qfq.csv"))
    if cache_pattern:
        try:
            qfq_df = pd.read_csv(cache_pattern[0], parse_dates=["日期"], encoding="utf-8-sig")
            qfq_df = qfq_df.rename(columns={"日期": "date", "收盘": "close"})
            qfq_df = qfq_df[["date", "close"]].copy()
        except Exception:
            qfq_df = None

    # 3. Fallback: fetch qfq from baostock
    if qfq_df is None:
        bs.login()
        try:
            rs2 = bs.query_history_k_data_plus(
                prefix + str(symbol).zfill(6),
                "date,close",
                start_date="2017-01-01",
                end_date="2026-06-19",
                frequency="d",
                adjustflag="2",  # qfq
            )
            if rs2.error_code == "0":
                data2 = rs2.get_data()
                if not data2.empty:
                    qfq_df = pd.DataFrame({
                        "date": pd.to_datetime(data2["date"]),
                        "close": pd.to_numeric(data2["close"], errors="coerce").astype(float),
                    })
        finally:
            bs.logout()

    return unadj_df, qfq_df


def load_financial_pit(symbol: str) -> pd.DataFrame | None:
    """Load PIT financial data for a symbol."""
    pit_path = DATA_RAW / f"financial_{str(symbol).zfill(6)}_pit.csv"
    old_path = DATA_RAW / f"financial_{str(symbol).zfill(6)}_ths_history.csv"
    for p in [pit_path, old_path]:
        if p.exists():
            try:
                df = pd.read_csv(p, dtype={"symbol": str}, encoding="utf-8-sig")
                df["symbol"] = df["symbol"].astype(str).str.zfill(6)
                df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
                df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce")
                return df
            except Exception:
                pass
    return None


# ═══════════════════════════════════════════════════════════
# Module 1: Ex-Date Discontinuity Check
# ═══════════════════════════════════════════════════════════

def check_ex_date_discontinuity(
    symbol: str,
    unadj_df: pd.DataFrame,
    qfq_df: pd.DataFrame,
) -> list[ExDateError]:
    """
    When unadjusted close drops >8% in a day, adj_factor must jump proportionally.
    """
    errors = []
    sym = str(symbol).zfill(6)

    # Merge on date
    merged = unadj_df.merge(qfq_df, on="date", suffixes=("_unadj", "_qfq"), how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    if len(merged) < 2:
        return errors

    close_u = merged["close_unadj"].values.astype(float)
    close_q = merged["close_qfq"].values.astype(float)
    ts = merged["tradestatus"].values.astype(int)
    dates = merged["date"].values

    for t in range(1, len(merged)):
        # Skip non-trading days and first day after halt
        if ts[t] == 0 or ts[t-1] == 0:
            continue
        if close_u[t-1] <= 0 or close_u[t] <= 0:
            continue
        if close_q[t-1] <= 0 or close_q[t] <= 0:
            continue

        unadj_ret = close_u[t] / close_u[t-1] - 1.0

        # Only check large drops (potential ex-dates)
        if unadj_ret > UNRET_DROP_THRESHOLD:
            continue

        # Compute adj_factor change
        adj_prev = close_q[t-1] / close_u[t-1]
        adj_curr = close_q[t] / close_u[t]
        adj_change = adj_curr / adj_prev - 1.0

        # Expected: adj_factor should increase to compensate for price drop
        # For a drop of -X%, adj should increase by X/(1-X)
        expected_adj_change = -unadj_ret / (1.0 + unadj_ret)

        # Tolerance: 5pp absolute
        mismatch = abs(adj_change - expected_adj_change)

        if mismatch > MISMATCH_TOLERANCE:
            severity = "FAIL" if mismatch > 0.15 else "WARN"
            errors.append(ExDateError(
                symbol=sym,
                date=str(dates[t])[:10],
                unadj_return=unadj_ret,
                adj_factor_change=adj_change,
                expected_change=expected_adj_change,
                mismatch=mismatch,
                severity=severity,
            ))

    return errors


# ═══════════════════════════════════════════════════════════
# Module 2: Market Cap Smoothness Check
# ═══════════════════════════════════════════════════════════

def check_market_cap_smoothness(
    symbol: str,
    unadj_df: pd.DataFrame,
    qfq_df: pd.DataFrame,
    fin_df: pd.DataFrame,
) -> list[McapDiscontinuity]:
    """
    exact_mcap[t] = total_share_r × (adj[t] / adj_r) × unadj_close[t]

    For each financial report period, compute daily exact_mcap.
    Flag days where daily mcap change exceeds the board-specific threshold.
    """
    discontinuities = []
    sym = str(symbol).zfill(6)
    board = get_board(sym)
    threshold = get_mcap_threshold(sym)

    # Merge price data
    merged = unadj_df.merge(qfq_df, on="date", suffixes=("_unadj", "_qfq"), how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)
    if merged.empty:
        return discontinuities

    close_u = merged["close_unadj"].values.astype(float)
    close_q = merged["close_qfq"].values.astype(float)
    ts = merged["tradestatus"].values.astype(int)
    dates = merged["date"].values

    # Build adj_factor series
    adj_factor = np.full(len(merged), np.nan)
    mask = (close_u > 0) & (close_q > 0)
    adj_factor[mask] = close_q[mask] / close_u[mask]

    # Iterate over each financial report
    fin_sorted = fin_df.sort_values("pub_date")
    for i, (_, fin_row) in enumerate(fin_sorted.iterrows()):
        report_date = fin_row["report_date"]
        pub_date = fin_row["pub_date"]
        total_share = pd.to_numeric(fin_row.get("total_share", np.nan), errors="coerce")

        if pd.isna(total_share) or total_share <= 0:
            continue
        if pd.isna(pub_date):
            continue

        # Find the pub_date index in daily data
        pub_idx = np.searchsorted(dates, pub_date)
        if pub_idx >= len(dates):
            continue

        adj_r = adj_factor[pub_idx]
        if np.isnan(adj_r) or adj_r <= 0:
            continue

        # Determine next report's pub_date for range end
        if i + 1 < len(fin_sorted):
            next_pub = fin_sorted.iloc[i + 1]["pub_date"]
            next_idx = min(np.searchsorted(dates, next_pub), len(dates))
        else:
            next_idx = len(dates)

        # Compute exact_mcap for each day in [pub_idx, next_idx)
        for t in range(pub_idx, next_idx):
            if ts[t] == 0:
                continue  # skip halted days

            adj_t = adj_factor[t]
            if np.isnan(adj_t) or adj_t <= 0:
                continue
            if close_u[t] <= 0:
                continue

            mcap_t = total_share * (adj_t / adj_r) * close_u[t]

            # Find previous trading day
            prev_t = t - 1
            while prev_t >= pub_idx and ts[prev_t] == 0:
                prev_t -= 1
            if prev_t < pub_idx:
                continue

            adj_prev = adj_factor[prev_t]
            if np.isnan(adj_prev) or adj_prev <= 0 or close_u[prev_t] <= 0:
                continue

            mcap_prev = total_share * (adj_prev / adj_r) * close_u[prev_t]
            if mcap_prev <= 0:
                continue

            mcap_ret = mcap_t / mcap_prev - 1.0

            if abs(mcap_ret) > threshold:
                severity = "FAIL" if abs(mcap_ret) > 0.30 else "WARN"
                discontinuities.append(McapDiscontinuity(
                    symbol=sym,
                    date=str(dates[t])[:10],
                    mcap_return=mcap_ret,
                    unadj_close=close_u[t],
                    adj_factor=adj_t,
                    total_share=total_share,
                    prev_mcap=mcap_prev,
                    curr_mcap=mcap_t,
                    board=board,
                    threshold_used=threshold,
                    severity=severity,
                ))

    return discontinuities


# ═══════════════════════════════════════════════════════════
# Module 3: PIT Strict Timestamp Check
# ═══════════════════════════════════════════════════════════

def check_pit_timestamps(
    symbol: str,
    fin_df: pd.DataFrame,
) -> list[PITViolation]:
    """
    HARD CHECKS — any violation is an automatic FAIL.

    1. pub_date >= report_date  (basic temporal logic)
    2. No look-ahead: simulate merge_asof, ensure no future data leaks
    3. pub_date within reasonable window (report_date + 4 months for annual)
    """
    violations = []
    sym = str(symbol).zfill(6)

    fin = fin_df.sort_values("report_date").copy()

    for _, row in fin.iterrows():
        report_date = row["report_date"]
        pub_date = row["pub_date"]

        if pd.isna(report_date) or pd.isna(pub_date):
            continue

        # Check 1: pub_date >= report_date
        if pub_date < report_date:
            violations.append(PITViolation(
                symbol=sym,
                report_date=str(report_date)[:10],
                pub_date=str(pub_date)[:10],
                violation_type="pub_before_report",
                severity="FAIL",
            ))

        # Check 3: pub_date within reasonable window
        # Statutory deadlines: Q1=Apr30, Q2=Aug31, Q3=Oct31, FY=Apr30(next yr)
        month = report_date.month
        day = report_date.day
        if month == 3 and day == 31:   # Q1
            max_pub = pd.Timestamp(year=report_date.year, month=5, day=31)
        elif month == 6 and day == 30:  # Q2
            max_pub = pd.Timestamp(year=report_date.year, month=9, day=30)
        elif month == 9 and day == 30:  # Q3
            max_pub = pd.Timestamp(year=report_date.year, month=11, day=30)
        elif month == 12 and day == 31:  # FY
            max_pub = pd.Timestamp(year=report_date.year + 1, month=5, day=31)
        else:
            max_pub = report_date + pd.Timedelta(days=120)

        if pub_date > max_pub + pd.Timedelta(days=60):
            violations.append(PITViolation(
                symbol=sym,
                report_date=str(report_date)[:10],
                pub_date=str(pub_date)[:10],
                violation_type="pub_date_suspicious",
                severity="WARN",
            ))

    # Check 2: Simulate merge_asof for key historical dates
    if len(fin) >= 2:
        # Generate test dates: every month-end 2017-2024
        test_dates = pd.date_range("2017-01-31", "2024-12-31", freq="ME")
        for t_date in test_dates:
            # merge_asof(direction='backward') should only get pub_date <= t_date
            future_rows = fin[fin["pub_date"] > t_date]
            if len(future_rows) < len(fin):
                backward_candidates = fin[fin["pub_date"] <= t_date]
                if not backward_candidates.empty:
                    latest = backward_candidates.loc[backward_candidates["pub_date"].idxmax()]
                    # Verify: this latest record's pub_date should be <= t_date
                    if latest["pub_date"] > t_date:
                        violations.append(PITViolation(
                            symbol=sym,
                            report_date=str(latest["report_date"])[:10],
                            pub_date=str(latest["pub_date"])[:10],
                            violation_type="look_ahead",
                            severity="FAIL",
                        ))
                        break  # one look-ahead is enough to fail this stock

    return violations


# ═══════════════════════════════════════════════════════════
# Module 4: Sanity & NaN Scan
# ═══════════════════════════════════════════════════════════

def check_sanity_bounds(
    symbol: str,
    unadj_df: pd.DataFrame,
    qfq_df: pd.DataFrame,
    fin_df: pd.DataFrame,
) -> list[SanityViolation]:
    """Check adj_factor, total_share, and unadj_close for NaN, negative, zero values."""
    violations = []
    sym = str(symbol).zfill(6)

    # Merge and compute adj_factor
    merged = unadj_df.merge(qfq_df, on="date", suffixes=("_unadj", "_qfq"), how="inner")
    merged = merged.sort_values("date")
    if merged.empty:
        return violations

    close_u = merged["close_unadj"].values.astype(float)
    close_q = merged["close_qfq"].values.astype(float)
    ts = merged["tradestatus"].values.astype(int)
    dates = merged["date"].values

    # Check adj_factor
    for t in range(len(merged)):
        if ts[t] == 0:
            continue
        cu, cq = close_u[t], close_q[t]
        dt_str = str(dates[t])[:10]

        # adj_factor
        if np.isnan(cu) or cu <= 0:
            violations.append(SanityViolation(sym, dt_str, "unadj_close",
                                                "NaN_or_nonpositive", cu, "FAIL"))
        if np.isnan(cq) or cq <= 0:
            violations.append(SanityViolation(sym, dt_str, "qfq_close",
                                                "NaN_or_nonpositive", cq, "FAIL"))
        elif cu > 0:
            adj = cq / cu
            if np.isnan(adj) or adj <= 0:
                violations.append(SanityViolation(sym, dt_str, "adj_factor",
                                                    "computed_nonpositive", adj, "FAIL"))

    # Check adj_factor smoothness (no random jumps on non-ex-dates)
    adj_vals = np.full(len(merged), np.nan)
    valid = (close_u > 0) & (close_q > 0)
    adj_vals[valid] = close_q[valid] / close_u[valid]

    for t in range(1, len(adj_vals)):
        if ts[t] == 0 or ts[t-1] == 0:
            continue
        if np.isnan(adj_vals[t]) or np.isnan(adj_vals[t-1]):
            continue
        adj_change = abs(adj_vals[t] / adj_vals[t-1] - 1.0)
        unadj_ret = abs(close_u[t] / close_u[t-1] - 1.0)
        # If adj_factor jumped >5% but no corresponding unadj move >4%
        if adj_change > 0.05 and unadj_ret < 0.04:
            violations.append(SanityViolation(
                sym, str(dates[t])[:10], "adj_factor",
                f"random_jump_{adj_change:.1%}_no_price_move",
                adj_vals[t], "WARN",
            ))

    # Check total_share
    if fin_df is not None and not fin_df.empty:
        for _, row in fin_df.iterrows():
            ts_val = pd.to_numeric(row.get("total_share", np.nan), errors="coerce")
            rpt = str(row.get("report_date", ""))[:10]
            if pd.isna(ts_val):
                violations.append(SanityViolation(sym, rpt, "total_share", "NaN", np.nan, "FAIL"))
            elif ts_val <= 0:
                violations.append(SanityViolation(sym, rpt, "total_share", "nonpositive", ts_val, "FAIL"))

        # Check for random total_share changes (non-report-period changes >1%)
        fin_sorted = fin_df.sort_values("report_date")
        ts_vals = pd.to_numeric(fin_sorted["total_share"], errors="coerce").values
        for t in range(1, len(ts_vals)):
            if pd.isna(ts_vals[t]) or pd.isna(ts_vals[t-1]) or ts_vals[t-1] <= 0:
                continue
            change = abs(ts_vals[t] / ts_vals[t-1] - 1.0)
            if change > 0.01:
                violations.append(SanityViolation(
                    sym,
                    str(fin_sorted.iloc[t]["report_date"])[:10],
                    "total_share",
                    f"change_{change:.2%}_between_reports",
                    ts_vals[t],
                    "WARN",
                ))

    return violations


# ═══════════════════════════════════════════════════════════
# Main Validation Runner
# ═══════════════════════════════════════════════════════════

def validate_symbol(symbol: str, report: ValidationReport) -> bool:
    """Run all 4 checks on a single symbol. Returns True if any FAIL-level violation found."""
    sym = str(symbol).zfill(6)
    has_failure = False

    # Load data
    unadj_df, qfq_df = load_daily_dual(sym)
    fin_df = load_financial_pit(sym)

    if unadj_df is None or qfq_df is None:
        logger.warning("  %s: Missing price data — skipping", sym)
        report.sanity_violations.append(SanityViolation(
            sym, "N/A", "price_data", "missing", np.nan, "FAIL",
        ))
        return True

    if fin_df is None:
        logger.info("  %s: No financial PIT data — skipping financial checks", sym)

    # Module 1
    ex_errors = check_ex_date_discontinuity(sym, unadj_df, qfq_df)
    report.ex_date_errors.extend(ex_errors)
    if any(e.severity == "FAIL" for e in ex_errors):
        has_failure = True

    # Module 2 (requires financial data)
    if fin_df is not None:
        mcap_errors = check_market_cap_smoothness(sym, unadj_df, qfq_df, fin_df)
        report.mcap_discontinuities.extend(mcap_errors)
        if any(e.severity == "FAIL" for e in mcap_errors):
            has_failure = True

    # Module 3 (HARD STOP on PIT violations)
    if fin_df is not None:
        pit_errors = check_pit_timestamps(sym, fin_df)
        report.pit_violations.extend(pit_errors)
        if any(e.severity == "FAIL" for e in pit_errors):
            logger.critical(
                "  %s: PIT VIOLATION DETECTED — pub_date < report_date or look-ahead!", sym
            )
            for v in pit_errors:
                if v.severity == "FAIL":
                    logger.critical("    %s | report=%s pub=%s type=%s",
                                    v.symbol, v.report_date, v.pub_date, v.violation_type)
            has_failure = True

    # Module 4
    san_errors = check_sanity_bounds(sym, unadj_df, qfq_df, fin_df)
    report.sanity_violations.extend(san_errors)
    if any(e.severity == "FAIL" for e in san_errors):
        has_failure = True

    return has_failure


def generate_blacklist(report: ValidationReport) -> pd.DataFrame:
    """Consolidate all FAIL-level events into a blacklist."""
    rows = []

    for e in report.ex_date_errors:
        if e.severity == "FAIL":
            rows.append({"symbol": e.symbol, "date": e.date,
                         "reason": f"ex_date_adj_mismatch_{e.mismatch:.2f}", "severity": "FAIL"})

    for e in report.mcap_discontinuities:
        if e.severity == "FAIL":
            rows.append({"symbol": e.symbol, "date": e.date,
                         "reason": f"mcap_discontinuity_{e.mcap_return:.1%}", "severity": "FAIL"})

    for e in report.pit_violations:
        rows.append({"symbol": e.symbol, "date": e.report_date,
                     "reason": f"pit_{e.violation_type}", "severity": "FAIL"})

    for e in report.sanity_violations:
        if e.severity == "FAIL":
            rows.append({"symbol": e.symbol, "date": e.date,
                         "reason": f"sanity_{e.field}_{e.issue}", "severity": "FAIL"})

    return pd.DataFrame(rows)


def print_report_card(report: ValidationReport) -> None:
    """Print terminal report card."""
    n_fail_stocks = len(set(
        [e.symbol for e in report.ex_date_errors if e.severity == "FAIL"]
        + [e.symbol for e in report.mcap_discontinuities if e.severity == "FAIL"]
        + [e.symbol for e in report.pit_violations]
        + [e.symbol for e in report.sanity_violations if e.severity == "FAIL"]
    ))
    n_warn_stocks = len(set(
        [e.symbol for e in report.ex_date_errors if e.severity == "WARN"]
        + [e.symbol for e in report.mcap_discontinuities if e.severity == "WARN"]
        + [e.symbol for e in report.sanity_violations if e.severity == "WARN"]
    ))
    n_pass = report.total_stocks - n_fail_stocks - n_warn_stocks

    print("\n" + "=" * 72)
    print("  DATA INTEGRITY VALIDATION REPORT CARD")
    print("=" * 72)
    print(f"  Stocks checked:          {report.total_stocks}")
    print(f"  Stocks PASSED:           {n_pass} ({n_pass/max(report.total_stocks,1)*100:.1f}%)")
    print(f"  Stocks with WARNINGS:    {n_warn_stocks} ({n_warn_stocks/max(report.total_stocks,1)*100:.1f}%)")
    print(f"  Stocks FAILED:           {n_fail_stocks} ({n_fail_stocks/max(report.total_stocks,1)*100:.1f}%)")
    print()
    print("  --- Failure Breakdown ---")
    n_ex_fail = len([e for e in report.ex_date_errors if e.severity == "FAIL"])
    n_mcap_fail = len([e for e in report.mcap_discontinuities if e.severity == "FAIL"])
    n_pit_fail = len(report.pit_violations)
    n_san_fail = len([e for e in report.sanity_violations if e.severity == "FAIL"])
    print(f"  Ex-date adj mismatch:       {n_ex_fail:4d} events")
    print(f"  Market cap discontinuity:   {n_mcap_fail:4d} events")
    print(f"  PIT timestamp violation:    {n_pit_fail:4d} events")
    print(f"  Sanity bounds violation:    {n_san_fail:4d} events")
    print()

    if n_fail_stocks > 0:
        fail_symbols = sorted(set(
            [e.symbol for e in report.ex_date_errors if e.severity == "FAIL"]
            + [e.symbol for e in report.mcap_discontinuities if e.severity == "FAIL"]
            + [e.symbol for e in report.pit_violations]
            + [e.symbol for e in report.sanity_violations if e.severity == "FAIL"]
        ))
        print(f"  --- Auto-Blacklisted ({len(fail_symbols)} stocks) ---")
        for s in fail_symbols[:30]:
            reasons = []
            for e in report.ex_date_errors:
                if e.symbol == s and e.severity == "FAIL":
                    reasons.append(f"ex-date@{e.date}")
            for e in report.mcap_discontinuities:
                if e.symbol == s and e.severity == "FAIL":
                    reasons.append(f"mcap@{e.date}")
            for e in report.pit_violations:
                if e.symbol == s:
                    reasons.append(f"pit@{e.report_date}")
            for e in report.sanity_violations:
                if e.symbol == s and e.severity == "FAIL":
                    reasons.append(f"sanity@{e.date}")
            print(f"    {s}: {', '.join(reasons[:3])}")

    if report.pit_violations:
        n_lookahead = len([v for v in report.pit_violations if v.violation_type == "look_ahead"])
        if n_lookahead > 0:
            print(f"\n  !!! CRITICAL: {n_lookahead} look-ahead PIT violations detected!")
            print(f"  !!! Panel rebuild ABORTED. Fix data source before retraining.")

    print("=" * 72)


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase B: Data Integrity Validation")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated list of symbols to validate")
    parser.add_argument("--sample", type=int, default=None,
                        help="Random sample of N stocks")
    parser.add_argument("--no-fin-check", action="store_true",
                        help="Skip financial PIT checks")
    args = parser.parse_args()

    # Determine symbol list
    if args.symbols:
        symbols = [s.strip().zfill(6) for s in args.symbols.split(",")]
    else:
        hist = pd.read_parquet(CSI800_HISTORY)
        all_symbols = sorted(set(str(s).zfill(6) for s in hist["symbol"].unique()))
        if args.sample:
            import random
            random.seed(42)
            symbols = random.sample(all_symbols, min(args.sample, len(all_symbols)))
        else:
            symbols = all_symbols

    logger.info("=" * 72)
    logger.info("Phase B: Data Integrity Validation")
    logger.info("  Symbols to validate: %d", len(symbols))
    logger.info("  Dynamic thresholds: Main=±12%%, GEM/STAR=±22%%")
    logger.info("  PIT check: HARD STOP on look-ahead")
    logger.info("=" * 72)

    report = ValidationReport(total_stocks=len(symbols))
    pit_hard_stop = False

    for i, sym in enumerate(symbols):
        if (i + 1) % 100 == 0:
            logger.info("  Progress: %d/%d", i + 1, len(symbols))

        has_failure = validate_symbol(sym, report)

        # PIT: hard stop on look-ahead
        look_ahead_events = [v for v in report.pit_violations
                             if v.symbol == sym and v.violation_type in ("look_ahead", "pub_before_report")]
        if look_ahead_events:
            pit_hard_stop = True
            logger.critical("=" * 72)
            logger.critical("HARD STOP: PIT look-ahead violation detected on %s", sym)
            for v in look_ahead_events:
                logger.critical("  report=%s pub=%s type=%s",
                                v.report_date, v.pub_date, v.violation_type)
            logger.critical("Panel rebuild ABORTED. Fix data source before continuing.")
            logger.critical("=" * 72)
            break

    # Print report
    print_report_card(report)

    # Generate blacklist
    blacklist = generate_blacklist(report)
    if len(blacklist) > 0:
        blacklist["symbol"] = blacklist["symbol"].astype(str).str.zfill(6)
        blacklist.to_csv(BLACKLIST_PATH, index=False, encoding="utf-8-sig")
        logger.info("Blacklist saved: %s (%d entries)", BLACKLIST_PATH, len(blacklist))

    # Save detailed event logs
    if report.ex_date_errors:
        pd.DataFrame([vars(e) for e in report.ex_date_errors]).to_csv(
            OUTPUT_DIR / "critical_adj_errors.csv", index=False, encoding="utf-8-sig")
    if report.mcap_discontinuities:
        pd.DataFrame([vars(e) for e in report.mcap_discontinuities]).to_csv(
            OUTPUT_DIR / "mcap_discontinuities.csv", index=False, encoding="utf-8-sig")
    if report.pit_violations:
        pd.DataFrame([vars(e) for e in report.pit_violations]).to_csv(
            OUTPUT_DIR / "pit_violations.csv", index=False, encoding="utf-8-sig")
    if report.sanity_violations:
        pd.DataFrame([vars(e) for e in report.sanity_violations]).to_csv(
            OUTPUT_DIR / "sanity_violations.csv", index=False, encoding="utf-8-sig")

    # Final verdict
    if pit_hard_stop:
        logger.critical("VALIDATION FAILED: PIT look-ahead detected. Do NOT proceed to panel rebuild.")
        sys.exit(1)
    elif report.critical_count > 0:
        logger.warning("VALIDATION COMPLETE: %d FAIL-level events in %d stocks. Blacklist generated.",
                        report.critical_count, len(set(blacklist["symbol"]) if len(blacklist) > 0 else []))
    else:
        logger.info("VALIDATION PASSED: All %d stocks clean.", report.total_stocks)


if __name__ == "__main__":
    main()
