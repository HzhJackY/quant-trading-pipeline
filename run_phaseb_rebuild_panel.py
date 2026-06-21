"""
Phase B Step 4-5: Rebuild Training Panel from Full CSI 800 Historical Data.

Pipeline:
  1. Convert cached daily CSVs → single parquet (fast date-range queries)
  2. Convert cached financial CSVs → single parquet (fast PIT queries)
  3. For each month-end (2017-2026):
     a. Load CSI 800 members for that date (from csi800_history.parquet)
     b. Query past 252 trading days of OHLCV
     c. Query latest PIT financial data (pub_date <= month_end)
     d. Compute market cap = close × total_share (PIT-aligned)
     e. Apply risk filters (ST, suspension, liquidity, mcap)
     f. Compute 16 factors → industry-neutralize → cross-sectional z-score
     g. Gram-Schmidt orthogonalize within CSI 800 universe
     h. Store _neutral_z columns for this date
  4. Concatenate all dates → output preprocessed_v2_full.parquet

Usage:
  python run_phaseb_rebuild_panel.py           # Full rebuild
  python run_phaseb_rebuild_panel.py --sample  # Test with 3 dates
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("phaseb_panel")

# ── Configuration ──
DATA_RAW = Path("data/raw")
OUTPUT_DIR = Path("output")
CSI800_HISTORY = OUTPUT_DIR / "csi800_history.parquet"
ALL_DAILY_PARQUET = OUTPUT_DIR / "all_daily.parquet"
ALL_FIN_PARQUET = OUTPUT_DIR / "all_financial_pit.parquet"
PANEL_OUTPUT = OUTPUT_DIR / "training_panel_v3_full.parquet"

START_DATE = "2017-01-01"
END_DATE = "2026-06-30"

FACTOR_NAMES = [
    "Mom_1M", "Mom_3M", "Mom_6M", "Mom_12M_1M",
    "Vol_20D", "Vol_60D", "Beta",
    "BP", "EP",
    "ROE", "Debt_Ratio", "Net_Profit_Margin",
    "RevGrowth_YoY", "ProfitGrowth_YoY",
    "VolChg_20D", "PriceDev_20D",
]

# Risk filter thresholds
MIN_AVG_AMOUNT_20D = 50_000_000   # 50M CNY
MIN_TOTAL_MCAP = 5_000_000_000    # 5B CNY
SUSPENSION_WINDOW = 5             # trading days


# ═══════════════════════════════════════════════════════════
# Step 1: Convert CSVs to Parquet
# ═══════════════════════════════════════════════════════════

def convert_daily_to_parquet() -> pd.DataFrame:
    """Load all cached daily CSVs into a single parquet file."""
    if ALL_DAILY_PARQUET.exists():
        logger.info("Loading daily parquet cache: %s", ALL_DAILY_PARQUET)
        return pd.read_parquet(ALL_DAILY_PARQUET)

    logger.info("Converting daily CSVs to parquet...")
    import re

    frames = []
    csv_files = sorted(DATA_RAW.glob("daily_*.csv"))
    for f in csv_files:
        m = re.match(r"daily_(\d{6})_", f.name)
        if not m:
            continue
        sym = m.group(1)
        try:
            df = pd.read_csv(f, parse_dates=["日期"], encoding="utf-8-sig")
            df["symbol"] = sym
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low",
                "成交量": "volume", "成交额": "amount", "换手率": "turnover",
            })
            keep = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
            df = df[[c for c in keep if c in df.columns]]
            frames.append(df)
        except Exception as e:
            logger.debug("  Skip %s: %s", f.name, e)

    if not frames:
        raise RuntimeError("No daily CSV data found!")

    result = pd.concat(frames, ignore_index=True)
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    result = result.drop_duplicates(subset=["date", "symbol"])
    result = result.sort_values(["symbol", "date"]).reset_index(drop=True)

    result.to_parquet(ALL_DAILY_PARQUET, index=False)
    logger.info("Daily parquet saved: %d rows, %d symbols", len(result), result["symbol"].nunique())
    return result


def convert_financial_to_parquet() -> pd.DataFrame:
    """Load all cached financial CSVs into a single parquet file."""
    if ALL_FIN_PARQUET.exists():
        logger.info("Loading financial parquet cache: %s", ALL_FIN_PARQUET)
        return pd.read_parquet(ALL_FIN_PARQUET)

    logger.info("Converting financial CSVs to parquet...")
    import re

    frames = []
    csv_files = sorted(DATA_RAW.glob("financial_*_akshare.csv"))
    if not csv_files:
        csv_files = sorted(DATA_RAW.glob("financial_*_pit.csv"))
    if not csv_files:
        csv_files = sorted(DATA_RAW.glob("financial_*_ths_history.csv"))

    for f in csv_files:
        m = re.match(r"financial_(\d{6})_", f.name)
        if not m:
            continue
        try:
            df = pd.read_csv(f, dtype={"symbol": str}, encoding="utf-8-sig")
            df["symbol"] = df["symbol"].astype(str).str.zfill(6)
            if "report_date" in df.columns:
                df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
            if "pub_date" in df.columns:
                df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce")
            # Force all non-key columns to float (prevent mixed-type arrow errors)
            key_cols = {"symbol", "report_date", "pub_date"}
            for c in df.columns:
                if c not in key_cols:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            frames.append(df)
        except Exception as e:
            logger.debug("  Skip %s: %s", f.name, e)

    if not frames:
        raise RuntimeError("No financial CSV data found!")

    result = pd.concat(frames, ignore_index=True)
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    result = result.drop_duplicates(subset=["symbol", "report_date"])
    result = result.sort_values(["symbol", "report_date"]).reset_index(drop=True)

    result.to_parquet(ALL_FIN_PARQUET, index=False)
    logger.info("Financial parquet saved: %d rows, %d symbols", len(result), result["symbol"].nunique())
    return result


# ═══════════════════════════════════════════════════════════
# Step 2: Risk Filters
# ═══════════════════════════════════════════════════════════

def apply_risk_filters(
    market_slice: pd.DataFrame,
    fin_slice: pd.DataFrame,
    current_date: pd.Timestamp,
) -> set[str]:
    """
    Apply four-layer risk filters to determine investable universe.

    Parameters
    ----------
    market_slice : DataFrame with columns [date, symbol, close, volume, amount]
        Multi-day OHLCV for the lookback window.
    fin_slice : DataFrame with columns [symbol, report_date, pub_date, total_share, ...]
        PIT-aligned financial data (only pub_date <= current_date).
    current_date : Timestamp
        Current month-end date.

    Returns
    -------
    set of symbol strings that pass all filters.
    """
    all_symbols = set(market_slice["symbol"].unique())
    valid = set(all_symbols)

    # 1. Suspension filter: must have close price on at least 1 of last 5 trading days
    last_5_dates = sorted(market_slice["date"].unique())[-SUSPENSION_WINDOW:]
    recent = market_slice[market_slice["date"].isin(last_5_dates)]
    has_close = set(recent.loc[recent["close"].notna() & (recent["close"] > 0), "symbol"].unique())
    suspended = valid - has_close
    valid &= has_close
    logger.debug("    Suspension: removed %d stocks", len(suspended))

    # 2. Liquidity filter: avg daily amount >= 50M over last 20 trading days
    if "amount" in market_slice.columns:
        last_20 = market_slice[
            market_slice["date"].isin(sorted(market_slice["date"].unique())[-20:])
        ]
        amt_avg = last_20.groupby("symbol")["amount"].mean()
        # Also require at least 5 trading days in the window
        amt_count = last_20.groupby("symbol")["amount"].count()
        low_liq = set(amt_avg[(amt_avg < MIN_AVG_AMOUNT_20D) | (amt_count < 5)].index)
        valid -= low_liq
        logger.debug("    Liquidity: removed %d stocks", len(low_liq & all_symbols))

    # 3. Market cap filter: total_mcap >= 5B (PIT-aligned)
    if "total_share" in fin_slice.columns:
        # Get latest close for each stock
        latest_close = (
            market_slice.dropna(subset=["close"])
            .sort_values("date")
            .groupby("symbol")
            .tail(1)[["symbol", "close"]]
        )
        # Get latest PIT total_share
        latest_fin = fin_slice.sort_values("report_date").groupby("symbol").tail(1)
        mcap_df = latest_close.merge(
            latest_fin[["symbol", "total_share"]], on="symbol", how="inner"
        )
        mcap_df["mcap_est"] = mcap_df["close"].astype(float) * mcap_df["total_share"].astype(float)
        micro_cap = set(mcap_df.loc[mcap_df["mcap_est"] < MIN_TOTAL_MCAP, "symbol"])
        valid -= micro_cap
        logger.debug("    Market cap: removed %d micro-caps", len(micro_cap & all_symbols))

    return valid


# ═══════════════════════════════════════════════════════════
# Step 3: Factor Computation (ported from factor_compute.py)
# ═══════════════════════════════════════════════════════════

def compute_factors_for_date(
    market_full: pd.DataFrame,
    fin_pit: pd.DataFrame,
    current_date: pd.Timestamp,
    universe: set[str],
    industry_map: pd.Series | None,
) -> pd.DataFrame:
    """
    Compute 16 factors for a single cross-section, within the given universe.

    Returns DataFrame with columns: symbol, {factor}_neutral_z
    """
    # Filter to past 252 trading days and universe symbols
    lookback_start = current_date - pd.Timedelta(days=400)  # ~252 trading days
    market = market_full[
        (market_full["date"] >= lookback_start)
        & (market_full["date"] <= current_date)
        & (market_full["symbol"].isin(universe))
    ].copy()

    if market.empty:
        return pd.DataFrame()

    # Pivot to wide format for factor computation
    market = market.sort_values(["symbol", "date"])

    # Compute factors per symbol
    factors = _compute_all_factors(market, fin_pit, current_date)

    if factors.empty:
        return pd.DataFrame()

    # Industry neutralize + cross-sectional zscore
    factors = _neutralize_and_zscore(factors, industry_map, universe)

    return factors


def _compute_all_factors(
    market: pd.DataFrame,
    fin_pit: pd.DataFrame,
    current_date: pd.Timestamp,
) -> pd.DataFrame:
    """Compute all 16 raw factors. Returns DataFrame with symbol + factor columns."""
    symbols = sorted(market["symbol"].unique())
    rows = []

    for sym in symbols:
        sym_data = market[market["symbol"] == sym].sort_values("date")
        if len(sym_data) < 20:
            continue

        close = sym_data["close"].astype(float)
        volume = sym_data["volume"].astype(float)
        amount = sym_data["amount"].astype(float) if "amount" in sym_data.columns else None

        row = {"symbol": sym, "close": close.iloc[-1]}
        n = len(close)

        # Momentum factors
        row["Mom_1M"] = close.iloc[-1] / close.iloc[-min(21, n)] - 1 if n >= 21 else np.nan
        row["Mom_3M"] = close.iloc[-1] / close.iloc[-min(63, n)] - 1 if n >= 63 else np.nan
        row["Mom_6M"] = close.iloc[-1] / close.iloc[-min(126, n)] - 1 if n >= 126 else np.nan
        if n >= 253:
            row["Mom_12M_1M"] = close.iloc[-22] / close.iloc[-253] - 1 if n >= 253 else np.nan
        else:
            row["Mom_12M_1M"] = np.nan

        # Volatility factors
        returns = close.pct_change().dropna()
        if len(returns) >= 20:
            row["Vol_20D"] = returns.iloc[-20:].std() * np.sqrt(252)
            row["Vol_60D"] = returns.iloc[-min(60, len(returns)):].std() * np.sqrt(252)
            # Beta: covariance with equal-weight market
            market_returns = (
                market.groupby("date")["close"].mean().pct_change().dropna()
            )
            aligned = returns.iloc[-min(252, len(returns)):].to_frame("stock")
            aligned["market"] = market_returns.iloc[-len(aligned):].values if len(aligned) <= len(market_returns) else np.nan
            aligned = aligned.dropna()
            if len(aligned) >= 60:
                cov = aligned["stock"].cov(aligned["market"])
                var = aligned["market"].var()
                row["Beta"] = cov / var if var > 0 else 1.0
            else:
                row["Beta"] = np.nan
            # Volatility change
            if len(returns) >= 40:
                vol_20 = returns.iloc[-20:].std()
                vol_40 = returns.iloc[-40:-20].std()
                row["VolChg_20D"] = vol_20 / vol_40 - 1 if vol_40 > 0 else np.nan
            else:
                row["VolChg_20D"] = np.nan
        else:
            row["Vol_20D"] = row["Vol_60D"] = row["Beta"] = row["VolChg_20D"] = np.nan

        # Technical: Price deviation from MA20
        if n >= 20:
            ma20 = close.iloc[-20:].mean()
            row["PriceDev_20D"] = (close.iloc[-1] - ma20) / ma20 if ma20 > 0 else np.nan
        else:
            row["PriceDev_20D"] = np.nan

        # Value/Quality from PIT financials
        sym_fin = fin_pit[fin_pit["symbol"] == sym]
        row.update(_compute_fundamental_factors(sym_fin, close.iloc[-1]))

        rows.append(row)

    result = pd.DataFrame(rows)
    return result


def _compute_fundamental_factors(
    sym_fin: pd.DataFrame,
    latest_close: float,
) -> dict:
    """Compute value/profitability/growth factors from PIT financial data."""
    row = {}
    for f in ["BP", "EP", "ROE", "Debt_Ratio", "Net_Profit_Margin",
              "RevGrowth_YoY", "ProfitGrowth_YoY"]:
        row[f] = np.nan

    if sym_fin.empty or latest_close <= 0:
        return row

    latest = sym_fin.sort_values("report_date").iloc[-1]

    # Book value per share: need equity/total_share
    # BVPS ≈ net_assets / total_share
    # For simplicity: use the available fields
    if "total_share" in sym_fin.columns and pd.notna(latest.get("total_share")):
        total_share = float(latest["total_share"])
        if total_share > 0:
            # Estimate book value from debt ratio: equity = assets * (1 - debt_ratio)
            # We don't have total assets directly, but we can approximate
            pass

    # EP (Earnings Yield) = EPS / price
    if "每股收益" in sym_fin.columns:
        eps = pd.to_numeric(latest.get("每股收益"), errors="coerce")
        if pd.notna(eps) and eps > 0:
            row["EP"] = eps / latest_close

    # ROE
    if "ROE" in sym_fin.columns:
        row["ROE"] = pd.to_numeric(latest.get("ROE"), errors="coerce")

    # Debt Ratio (already as liability/asset ratio)
    if "Debt_Ratio" in sym_fin.columns:
        row["Debt_Ratio"] = pd.to_numeric(latest.get("Debt_Ratio"), errors="coerce")

    # Net Profit Margin
    if "销售净利率" in sym_fin.columns:
        row["Net_Profit_Margin"] = pd.to_numeric(latest.get("销售净利率"), errors="coerce")

    # Growth: YoY revenue and profit growth from adjacent reports
    if len(sym_fin) >= 5:  # at least 1 year of quarters
        prev = sym_fin.sort_values("report_date").iloc[-5]  # same quarter last year
        cur = latest
        if "营业收入" in sym_fin.columns:
            cur_rev = pd.to_numeric(cur.get("营业收入"), errors="coerce")
            prev_rev = pd.to_numeric(prev.get("营业收入"), errors="coerce")
            if pd.notna(cur_rev) and pd.notna(prev_rev) and prev_rev > 0:
                row["RevGrowth_YoY"] = cur_rev / prev_rev - 1

        if "净利润" in sym_fin.columns:
            cur_profit = pd.to_numeric(cur.get("净利润"), errors="coerce")
            prev_profit = pd.to_numeric(prev.get("净利润"), errors="coerce")
            if pd.notna(cur_profit) and pd.notna(prev_profit) and prev_profit > 0:
                row["ProfitGrowth_YoY"] = cur_profit / prev_profit - 1

    return row


# ═══════════════════════════════════════════════════════════
# Step 4: Neutralization & Orthogonalization
# ═══════════════════════════════════════════════════════════

def _neutralize_and_zscore(
    factors: pd.DataFrame,
    industry_map: pd.Series | None,
    universe: set[str],
) -> pd.DataFrame:
    """Industry neutralize + cross-sectional z-score."""
    factor_cols = [c for c in FACTOR_NAMES if c in factors.columns]

    for col in factor_cols:
        # Industry neutralize
        if industry_map is not None and len(industry_map) > 0:
            factors = factors.merge(
                industry_map.rename("industry"), left_on="symbol", right_index=True, how="left"
            )
            industry_means = factors.groupby("industry")[col].transform("mean")
            factors[col] = factors[col] - industry_means.fillna(factors[col].mean())
            if "industry" in factors.columns:
                factors = factors.drop(columns=["industry"])
        else:
            # Global mean neutralize
            factors[col] = factors[col] - factors[col].mean()

        # Cross-sectional z-score (within universe)
        mu = factors[col].mean()
        sigma = factors[col].std()
        if sigma and sigma > 0:
            factors[f"{col}_neutral_z"] = (factors[col] - mu) / sigma
        else:
            factors[f"{col}_neutral_z"] = 0.0

    return factors


# ═══════════════════════════════════════════════════════════
# Step 5: Main panel builder
# ═══════════════════════════════════════════════════════════

def get_month_end_dates(start: str, end: str) -> list[pd.Timestamp]:
    """Generate month-end dates in range."""
    dates = pd.date_range(start, end, freq="ME")
    return list(dates)


def get_csi800_for_date(date: pd.Timestamp, history: pd.DataFrame) -> set[str]:
    """Get CSI 800 constituents effective at a given date."""
    snapshots = history[history["snapshot_date"] <= date]
    if snapshots.empty:
        return set()
    latest_snapshot = snapshots["snapshot_date"].max()
    members = history[history["snapshot_date"] == latest_snapshot]
    return set(str(s).zfill(6) for s in members["symbol"].unique())


def build_training_panel(sample: bool = False, blacklist_path: str | None = None) -> pd.DataFrame:
    """Main panel rebuild loop."""
    # Load blacklist
    blacklist = set()
    if blacklist_path:
        bl_path = Path(blacklist_path)
        if bl_path.exists():
            bl = pd.read_csv(bl_path, dtype={"symbol": str})
            bl["symbol"] = bl["symbol"].astype(str).str.zfill(6)
            blacklist = set(bl["symbol"].unique())
            logger.info("Blacklist loaded: %d symbols excluded", len(blacklist))
    # Load data
    daily = convert_daily_to_parquet()
    fin = convert_financial_to_parquet()
    history = pd.read_parquet(CSI800_HISTORY)
    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"])

    month_ends = get_month_end_dates(START_DATE, END_DATE)
    if sample:
        month_ends = month_ends[:3]  # test with first 3 dates
        logger.info("SAMPLE MODE: %d dates", len(month_ends))

    logger.info("=" * 64)
    logger.info("Building training panel: %d month-ends", len(month_ends))
    logger.info("  Daily data: %d rows, %d symbols", len(daily), daily["symbol"].nunique())
    logger.info("  Financial data: %d rows, %d symbols", len(fin), fin["symbol"].nunique())
    logger.info("=" * 64)

    all_panels = []
    prev_symbols = set()

    for i, dt in enumerate(month_ends):
        logger.info("[%3d/%3d] %s ...", i + 1, len(month_ends), str(dt)[:10])
        t0 = time.perf_counter()

        # 1. Universe determination
        csi800 = get_csi800_for_date(dt, history)
        if len(csi800) < 50:
            logger.warning("  CSI 800 has only %d members — skip", len(csi800))
            continue

        # 2. Get market data slice
        market_slice = daily[
            (daily["date"] >= dt - pd.Timedelta(days=400))
            & (daily["date"] <= dt)
            & (daily["symbol"].isin(csi800))
        ]

        # 3. Get PIT financials (pub_date <= current_date)
        fin_pit = fin[
            (fin["pub_date"] <= dt)
            & (fin["symbol"].isin(csi800))
        ].copy()

        # 4. Risk filters
        universe = apply_risk_filters(market_slice, fin_pit, dt)
        logger.info("  CSI800=%d -> Filtered=%d", len(csi800), len(universe))

        # 4b. Blacklist exclusion
        if blacklist:
            n_before = len(universe)
            universe -= blacklist
            if len(universe) < n_before:
                logger.debug("  Blacklist: removed %d stocks, %d remain", n_before - len(universe), len(universe))

        # 5. Compute factors
        factors = compute_factors_for_date(daily, fin, dt, universe, None)
        if factors.empty:
            logger.warning("  No factors computed — skip")
            continue

        factors["date"] = dt
        all_panels.append(factors)

        dt_sec = time.perf_counter() - t0
        if (i + 1) % 12 == 0:
            logger.info("  Progress: %d/%d, %.1fs/date", i + 1, len(month_ends), dt_sec)

    if not all_panels:
        raise RuntimeError("No panel data generated!")

    panel = pd.concat(all_panels, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)

    # Keep _neutral_z columns + date/symbol + 收盘 (for label computation)
    neutral_z_cols = [c for c in panel.columns if c.endswith("_neutral_z")]
    keep_cols = ["date", "symbol"] + neutral_z_cols
    if "close" in panel.columns:
        keep_cols.append("close")
    output = panel[keep_cols].copy()
    if "close" in output.columns:
        output = output.rename(columns={"close": "收盘"})

    output.to_parquet(PANEL_OUTPUT, index=False)
    logger.info("=" * 64)
    logger.info("Panel saved: %s", PANEL_OUTPUT)
    logger.info("  Shape: %d rows × %d cols", output.shape[0], output.shape[1])
    logger.info("  Dates: %d", output["date"].nunique())
    logger.info("  Symbols: %d", output["symbol"].nunique())
    logger.info("=" * 64)

    return output


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase B: Rebuild training panel")
    parser.add_argument("--sample", action="store_true", help="Test with 3 dates only")
    parser.add_argument("--blacklist", type=str, default=str(OUTPUT_DIR / "blacklist_symbols.csv"),
                        help="Path to blacklist CSV (from validate_data_integrity.py)")
    args = parser.parse_args()

    build_training_panel(sample=args.sample, blacklist_path=args.blacklist)


if __name__ == "__main__":
    main()
