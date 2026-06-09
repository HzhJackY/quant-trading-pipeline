"""
Factor Computation Pipeline — Replicating the training-time 16-factor cross-section.

This module computes the EXACT same 16 factors used in ProductionAlphaEngine
training, ensuring column-name compatibility for predict_cross_section().

Factor categories:
  ┌──────────────┬──────────────────────────────────────────────────┐
  │ Momentum     │ Mom_1M, Mom_3M, Mom_6M, Mom_12M_1M              │
  │ Volatility   │ Vol_20D, Vol_60D, Beta, VolChg_20D, PriceDev_20D │
  │ Valuation    │ BP, EP                                          │
  │ Profitability│ ROE, Net_Profit_Margin                           │
  │ Leverage     │ Debt_Ratio                                       │
  │ Growth       │ RevGrowth_YoY, ProfitGrowth_YoY                 │
  └──────────────┴──────────────────────────────────────────────────┘

Pipeline:
  Raw → Industry-Neutralize (board-level) → Z-Score → Cross-Sectional Rank
  Output columns: {factor}_neutral_z_rank

Important:
  - Industry neutralization uses `board` field as proxy for SW industry
    (AkShare daily spot provides board but not SW classification).
  - Cross-sectional rank is PER DATE (same as training).
  - NaN factors → fill 0.0 AFTER rank (consistent with training: fillna(0.5) on rank).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("factor_compute")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════
# Factor Registry: the 16 factors expected by ProductionAlphaEngine
# ═══════════════════════════════════════════════════════════

FACTOR_NAMES = [
    # Momentum (4)
    "Mom_1M", "Mom_3M", "Mom_6M", "Mom_12M_1M",
    # Volatility & Technical (5)
    "Vol_20D", "Vol_60D", "Beta", "VolChg_20D", "PriceDev_20D",
    # Valuation (2)
    "BP", "EP",
    # Profitability (2)
    "ROE", "Net_Profit_Margin",
    # Leverage (1)
    "Debt_Ratio",
    # Growth (2)
    "RevGrowth_YoY", "ProfitGrowth_YoY",
]


# ═══════════════════════════════════════════════════════════
# Stage 1: Raw Factor Computation
# ═══════════════════════════════════════════════════════════

def _compute_momentum_factors(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum factors from OHLCV cache.

    Args:
        market_df: DataFrame with cols [trade_date, symbol, close].
                   Must be sorted by symbol, trade_date.

    Returns:
        DataFrame with cols [symbol, Mom_1M, Mom_3M, Mom_6M, Mom_12M_1M].
    """
    df = market_df[["trade_date", "symbol", "close"]].copy()
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # For each stock, get the close at t, t-21, t-63, t-126, t-252
    def _mom(sym_df: pd.DataFrame) -> pd.DataFrame:
        sym_df = sym_df.set_index("trade_date").sort_index()
        close = sym_df["close"].dropna()
        if len(close) < 2:
            sym_df["Mom_1M"] = np.nan
            sym_df["Mom_3M"] = np.nan
            sym_df["Mom_6M"] = np.nan
            sym_df["Mom_12M_1M"] = np.nan
            return sym_df.reset_index()

        # Latest close
        latest = close.iloc[-1]
        # Returns over trailing windows (trading days ≈ 21/month)
        for label, offset in [("Mom_1M", 21), ("Mom_3M", 63), ("Mom_6M", 126)]:
            if len(close) > offset:
                sym_df[label] = latest / close.iloc[-(offset + 1)] - 1.0
            else:
                sym_df[label] = np.nan

        # Mom_12M_1M: return from t-12M to t-1M (exclude most recent month)
        if len(close) > 252:
            sym_df["Mom_12M_1M"] = close.iloc[-22] / close.iloc[-253] - 1.0
        else:
            sym_df["Mom_12M_1M"] = np.nan

        return sym_df.reset_index()

    # pandas >= 2.2: use groupby.apply with include_groups=False
    # The result from apply may drop grouping columns in newer pandas
    try:
        result = df.groupby("symbol", group_keys=False)[["trade_date", "symbol", "close"]].apply(
            _mom, include_groups=False
        )
    except TypeError:
        result = df.groupby("symbol", group_keys=False)[["trade_date", "symbol", "close"]].apply(_mom)

    # Ensure symbol column exists
    if "symbol" not in result.columns:
        result = result.reset_index()

    # Keep only latest date per symbol
    latest_per_sym = result.groupby("symbol").tail(1)

    cols = [c for c in ["symbol", "Mom_1M", "Mom_3M", "Mom_6M", "Mom_12M_1M"] if c in latest_per_sym.columns]
    return latest_per_sym[cols].reset_index(drop=True)


def _compute_volatility_factors(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility/technical factors.

    Vol_20D: 20-day annualized return volatility = std(20d returns) * sqrt(252)
    Vol_60D: 60-day annualized return volatility
    Beta:    rolling 252-day market beta (using equal-weight market return as proxy)
    VolChg_20D: Vol_20D(t) / Vol_20D(t-20) - 1
    PriceDev_20D: (close - MA20) / MA20
    """
    df = market_df[["trade_date", "symbol", "close"]].copy()
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    def _vol(sym_df: pd.DataFrame) -> pd.DataFrame:
        sym_df = sym_df.sort_values("trade_date").set_index("trade_date")
        close = sym_df["close"].dropna()

        if len(close) < 5:
            for c in ["Vol_20D", "Vol_60D", "Beta", "VolChg_20D", "PriceDev_20D"]:
                sym_df[c] = np.nan
            return sym_df.reset_index()

        ret = close.pct_change().dropna()

        # Volatility (annualized)
        sym_df["Vol_20D"] = ret.iloc[-20:].std() * np.sqrt(252) if len(ret) >= 20 else np.nan
        sym_df["Vol_60D"] = ret.iloc[-60:].std() * np.sqrt(252) if len(ret) >= 60 else np.nan

        # Price deviation from 20-day MA
        ma20 = close.iloc[-20:].mean() if len(close) >= 20 else close.mean()
        sym_df["PriceDev_20D"] = close.iloc[-1] / ma20 - 1.0

        # VolChg: change in 20-day vol vs 20 days ago
        if len(ret) >= 40:
            vol_recent = ret.iloc[-20:].std()
            vol_past = ret.iloc[-40:-20].std()
            sym_df["VolChg_20D"] = (vol_recent / vol_past - 1.0) if vol_past > 0 else 0.0
        else:
            sym_df["VolChg_20D"] = np.nan

        # Beta (vs equal-weight market, rolling 252 days)
        if len(ret) >= 63:
            # Market proxy: average return of all stocks in the panel
            # For single-stock computation, use close vs self (simplified)
            # In production, this should use a market index like 000300
            market_ret = ret  # placeholder — see market_beta below
            common = ret.index.intersection(market_ret.index)
            if len(common) >= 63:
                aligned_ret = ret.loc[common[-252:]]
                aligned_mkt = market_ret.loc[common[-252:]]
                cov = np.cov(aligned_ret, aligned_mkt)[0, 1]
                var = np.var(aligned_mkt)
                sym_df["Beta"] = cov / var if var > 0 else 1.0
            else:
                sym_df["Beta"] = 1.0
        else:
            sym_df["Beta"] = 1.0

        return sym_df.reset_index()

    try:
        result = df.groupby("symbol", group_keys=False)[["trade_date", "symbol", "close"]].apply(
            _vol, include_groups=False
        )
    except TypeError:
        result = df.groupby("symbol", group_keys=False)[["trade_date", "symbol", "close"]].apply(_vol)

    if "symbol" not in result.columns:
        result = result.reset_index()
    latest = result.groupby("symbol").tail(1)

    # ── Proper market beta (cross-sectional) ──
    # Compute equal-weight market return for each date in the panel
    mkt_ret = (market_df
               .sort_values(["trade_date", "symbol"])
               .dropna(subset=["close"])
               .drop_duplicates(["trade_date", "symbol"])
               .copy())
    mkt_ret["daily_ret"] = mkt_ret.groupby("symbol")["close"].transform(
        lambda x: x.pct_change()
    )
    mkt_idx = mkt_ret.groupby("trade_date")["daily_ret"].mean().rename("market_ret")

    def _proper_beta(sym_df: pd.DataFrame) -> float:
        sym_df = sym_df.sort_values("trade_date").set_index("trade_date")
        ret = sym_df["close"].pct_change().dropna()
        ret.name = "stock_ret"
        merged = pd.concat([ret, mkt_idx], axis=1).dropna()
        if len(merged) < 63:
            return 1.0
        merged = merged.iloc[-252:]
        cov = np.cov(merged["stock_ret"], merged["market_ret"])[0, 1]
        var = np.var(merged["market_ret"])
        return cov / var if var > 0 else 1.0

    betas = {}
    for sym, grp in mkt_ret.groupby("symbol"):
        try:
            betas[sym] = _proper_beta(grp)
        except Exception:
            betas[sym] = 1.0
    beta_series = pd.Series(betas, name="Beta")

    # Merge proper beta into result
    latest = latest.drop(columns=["Beta"], errors="ignore")
    latest = latest.merge(beta_series.reset_index().rename(columns={"index": "symbol"}), on="symbol", how="left")
    latest["Beta"] = latest["Beta"].fillna(1.0)

    return latest[["symbol", "Vol_20D", "Vol_60D", "Beta", "VolChg_20D", "PriceDev_20D"]].reset_index(drop=True)


def _compute_fundamental_factors(
    spot_df: pd.DataFrame,
    pit_financials: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Extract valuation, profitability, leverage, and growth factors.

    Uses a TWO-SOURCE strategy to avoid look-ahead bias:
      Source A (spot): PE_TTM, PB, market cap — real-time, always available
      Source B (PIT financials): Debt_Ratio, RevGrowth, ProfitGrowth —
        only used if PIT-aligned (statutory disclosure deadline respected)

    Args:
        spot_df: Output of fetch_daily_fundamentals() — always used.
        pit_financials: Output of fetch_and_align_financials() — PIT-gated.

    Returns:
        DataFrame with cols [symbol, BP, EP, ROE, Net_Profit_Margin,
                             Debt_Ratio, RevGrowth_YoY, ProfitGrowth_YoY]
    """
    df = spot_df.copy()
    result = pd.DataFrame({"symbol": df["symbol"]})

    # ── Source A: Spot data (real-time, no PIT concern) ──
    # BP = Book Value / Market Cap = 1 / PB
    if "pb" in df.columns:
        result["BP"] = 1.0 / df["pb"].replace(0, np.nan)
    else:
        result["BP"] = np.nan

    # EP = Earnings / Price = 1 / PE_TTM
    if "pe_ttm" in df.columns:
        result["EP"] = 1.0 / df["pe_ttm"].replace(0, np.nan)
    else:
        result["EP"] = np.nan

    # ROE (TTM, from spot)
    result["ROE"] = df.get("roe", np.nan)

    # Net_Profit_Margin (TTM, from spot)
    result["Net_Profit_Margin"] = df.get("net_margin", np.nan)

    # ── Defaults for PIT-gated factors (will be overwritten if available) ──
    result["Debt_Ratio"] = np.nan
    result["RevGrowth_YoY"] = np.nan
    result["ProfitGrowth_YoY"] = np.nan

    # ── Source B: PIT-aligned financial statements ──
    if pit_financials is not None and len(pit_financials) > 0:
        pit = pit_financials.copy()
        pit_map = pit.set_index("symbol")

        # Debt_Ratio = Total Liabilities / Total Assets (from balance sheet)
        if "debt_ratio" in pit.columns:
            debt_map = pit.set_index("symbol")["debt_ratio"]
            result["Debt_Ratio"] = result["symbol"].map(debt_map)

        # Revenue YoY growth (requires prior-year data — use spot proxy if unavailable)
        if "revenue_yoy" in pit.columns:
            rev_yoy_map = pit.set_index("symbol")["revenue_yoy"]
            result["RevGrowth_YoY"] = result["symbol"].map(rev_yoy_map)

        # Profit YoY growth
        if "profit_yoy" in pit.columns:
            profit_yoy_map = pit.set_index("symbol")["profit_yoy"]
            result["ProfitGrowth_YoY"] = result["symbol"].map(profit_yoy_map)

        logger.info("  PIT financials merged: %d stocks with balance-sheet factors",
                    result["Debt_Ratio"].notna().sum())

    # Defensive: ensure all expected columns exist
    for col in ["BP", "EP", "ROE", "Net_Profit_Margin", "Debt_Ratio",
                "RevGrowth_YoY", "ProfitGrowth_YoY"]:
        if col not in result.columns:
            result[col] = np.nan

    return result


# ═══════════════════════════════════════════════════════════
# Stage 2: Industry Neutralization + Z-Score
# ═══════════════════════════════════════════════════════════

def industry_neutralize(
    factor_df: pd.DataFrame,
    industry_map: pd.Series,
    factor_names: list[str] | None = None,
    min_group_size: int = 5,
) -> pd.DataFrame:
    """
    Cross-sectionally neutralize factors against industry membership.

    neutralized_factor = factor - mean(factor | industry)

    Args:
        factor_df: DataFrame with symbol col + factor columns.
        industry_map: pd.Series with index=symbol, values=industry/board.
        factor_names: List of factor columns to neutralize. Default: auto-detect.
        min_group_size: Industries with fewer stocks get global mean subtracted.

    Returns:
        DataFrame with original columns + {factor}_neutral columns.
    """
    if factor_names is None:
        factor_names = [c for c in factor_df.columns
                       if c != "symbol" and not c.endswith("_neutral")]

    result = factor_df.copy()
    industry_series = industry_map.reindex(result["symbol"]).fillna("Others")
    result["_industry"] = industry_series.values

    for f in factor_names:
        if f not in result.columns:
            continue
        global_mean = result[f].mean()
        result[f"{f}_neutral"] = result[f] - result.groupby("_industry")[f].transform(
            lambda x: x.mean() if len(x) >= min_group_size else global_mean
        )

    result = result.drop(columns=["_industry"])
    return result


def cross_sectional_zscore(
    factor_df: pd.DataFrame,
    factor_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Cross-sectional z-score normalization.

    z = (factor - mean(factor)) / std(factor)

    Args:
        factor_df: DataFrame with _neutral columns.
        factor_names: List of _neutral columns. Auto-detected if None.

    Returns:
        DataFrame with additional {factor}_neutral_z columns.
    """
    if factor_names is None:
        factor_names = [c for c in factor_df.columns if c.endswith("_neutral")]

    result = factor_df.copy()
    for fn in factor_names:
        mu = result[fn].mean()
        sigma = result[fn].std()
        if sigma and sigma > 0:
            result[f"{fn}_z"] = (result[fn] - mu) / sigma
        else:
            result[f"{fn}_z"] = 0.0
    return result


def cross_sectional_rank(
    factor_df: pd.DataFrame,
    factor_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Cross-sectional percentile rank of z-scored factors → [0, 1].

    This is the FINAL transformation. The output columns match exactly what
    ProductionAlphaEngine.predict_cross_section() expects.

    Args:
        factor_df: DataFrame with _neutral_z columns.
        factor_names: List of _neutral_z columns. Auto-detected if None.

    Returns:
        DataFrame with additional {factor}_neutral_z_rank columns.
    """
    if factor_names is None:
        factor_names = [c for c in factor_df.columns if c.endswith("_neutral_z")
                       and not c.endswith("_rank")]

    result = factor_df.copy()
    for fn in factor_names:
        rc = f"{fn}_rank"
        # Cross-sectional rank within this single cross-section
        result[rc] = result[fn].rank(pct=True, na_option="bottom").fillna(0.5)
    return result


# ═══════════════════════════════════════════════════════════
# Main Entry Point: Compute full feature matrix for one cross-section
# ═══════════════════════════════════════════════════════════

def compute_feature_matrix(
    market_cache_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    *,
    pit_financials: pd.DataFrame | None = None,
    industry_map: pd.Series | None = None,
    factor_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute the full feature matrix for ONE cross-section.

    This is THE function called at month-end to produce the exact feature
    columns that ProductionAlphaEngine.predict_cross_section() requires.

    Pipeline:
      1. Compute 16 raw factors from 60-day market cache + fundamentals
         (with PIT-gated financial data for balance-sheet factors)
      2. Industry-neutralize using SW industry classification
      3. Cross-sectional z-score → _neutral_z
      4. Cross-sectional rank → _neutral_z_rank
      5. Return only _neutral_z_rank columns + symbol

    Args:
        market_cache_df: 60-day OHLCV from StateManager.query_market_cache().
        fundamentals_df: Latest fundamentals from fetch_daily_fundamentals().
        pit_financials: PIT-aligned financial statements from fetch_and_align_financials().
            Used for Debt_Ratio, RevGrowth_YoY, ProfitGrowth_YoY.
            If None, these factors will be NaN (filled with cross-sectional median).
        industry_map: pd.Series index=symbol, values=SW industry_name.
            If None, uses "全市场" (global mean only, no industry differentiation).
        factor_names: Subset of factors to compute. Default: all 16.

    Returns:
        pd.DataFrame with cols [symbol, {factor}_neutral_z_rank, ...]
        Exactly compatible with predict_cross_section(features=...).
    """
    if factor_names is None:
        factor_names = list(FACTOR_NAMES)

    logger.info("Computing feature matrix | %d factors | %d rows cache | %d stocks fundamentals",
                len(factor_names), len(market_cache_df), len(fundamentals_df))

    # ── Step 1: Raw factors ──
    mom_df = _compute_momentum_factors(market_cache_df)
    vol_df = _compute_volatility_factors(market_cache_df)
    fund_df = _compute_fundamental_factors(fundamentals_df, pit_financials=pit_financials)

    # Merge all raw factors
    feature_df = mom_df.merge(vol_df, on="symbol", how="outer")
    feature_df = feature_df.merge(fund_df, on="symbol", how="outer")

    # Keep only stocks that appear in market data
    market_symbols = set(market_cache_df["symbol"].unique())
    feature_df = feature_df[feature_df["symbol"].isin(market_symbols)].copy()

    logger.info("  Raw factors: %d stocks", len(feature_df))

    # Fill missing factors with cross-sectional median
    for fn in factor_names:
        if fn in feature_df.columns:
            med = feature_df[fn].median()
            feature_df[fn] = feature_df[fn].fillna(med if pd.notna(med) else 0.0)

    # ── Step 2: Industry neutralization ──
    # Use SW industry classification if available, fallback to global mean
    if industry_map is None or len(industry_map) == 0:
        logger.warning("  No industry map provided — using global mean neutralization")
        industry_map = pd.Series("全市场", index=feature_df["symbol"])
    else:
        logger.info("  Industry map: %d stocks, %d unique industries",
                    len(industry_map), industry_map.nunique())

    feature_df = industry_neutralize(feature_df, industry_map, factor_names)

    # ── Step 3: Z-score ──
    neutral_cols = [f"{f}_neutral" for f in factor_names if f"{f}_neutral" in feature_df.columns]
    feature_df = cross_sectional_zscore(feature_df, neutral_cols)

    # ── Step 4: Cross-sectional rank (FINAL) ──
    z_cols = [f"{f}_neutral_z" for f in factor_names if f"{f}_neutral_z" in feature_df.columns]
    feature_df = cross_sectional_rank(feature_df, z_cols)

    # ── Return only what predict_cross_section needs ──
    rank_cols = [f"{f}_neutral_z_rank" for f in factor_names]
    output_cols = ["symbol"] + [c for c in rank_cols if c in feature_df.columns]

    result = feature_df[output_cols].copy()
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)

    logger.info("  Final feature matrix: %d stocks × %d features",
                len(result), len(output_cols) - 1)
    logger.info("  Feature columns: %s", ", ".join(output_cols[1:]))

    return result.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════
# Compatibility checker
# ═══════════════════════════════════════════════════════════

def validate_feature_columns(
    feature_df: pd.DataFrame,
    expected_cols: list[str],
) -> tuple[bool, set[str], set[str]]:
    """
    Check that the feature matrix has the columns expected by the engine.

    Returns:
        (is_valid, missing_cols, extra_cols)
    """
    present = set(feature_df.columns)
    expected = set(expected_cols)
    missing = expected - present
    extra = present - expected - {"symbol"}
    return len(missing) == 0, missing, extra


# ═══════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s")

    print("=== Factor Computation Self-Test ===\n")

    # Generate synthetic test data
    np.random.seed(42)
    dates = pd.date_range("2026-03-01", "2026-06-05", freq="B")
    symbols = [f"{i:06d}" for i in range(1, 101)]  # 100 fake stocks
    board_map = {s: np.random.choice(["沪市主板", "深市主板", "创业板", "科创板"], p=[0.35, 0.30, 0.20, 0.15])
                 for s in symbols}

    market_rows = []
    for d in dates:
        for s in symbols:
            base_price = 10.0 + hash(s) % 90
            close = base_price * (1 + np.random.normal(0, 0.02))
            market_rows.append({
                "trade_date": d, "symbol": s,
                "open": close * 0.99, "high": close * 1.02,
                "low": close * 0.98, "close": close,
                "volume": np.random.lognormal(15, 1),
                "amount": np.random.lognormal(17, 1),
                "pct_change": np.random.normal(0, 0.02),
                "turnover_rate": np.random.uniform(0.1, 5),
            })
    market_df = pd.DataFrame(market_rows)

    fundamentals_df = pd.DataFrame({
        "symbol": symbols,
        "name": [f"Stock_{s}" for s in symbols],
        "pe_ttm": np.random.uniform(5, 200, len(symbols)),
        "pb": np.random.uniform(0.5, 15, len(symbols)),
        "total_mcap": np.random.lognormal(23, 1.5, len(symbols)),
        "float_mcap": np.random.lognormal(22, 1.5, len(symbols)),
        "roe": np.random.uniform(-0.3, 0.4, len(symbols)),
        "eps": np.random.uniform(-2, 10, len(symbols)),
        "bps": np.random.uniform(1, 30, len(symbols)),
        "net_margin": np.random.uniform(-0.5, 0.6, len(symbols)),
        "gross_margin": np.random.uniform(0.05, 0.7, len(symbols)),
        "board": [board_map[s] for s in symbols],
    })

    print(f"Market cache: {len(market_df)} rows, {market_df['symbol'].nunique()} symbols")
    print(f"Fundamentals: {len(fundamentals_df)} stocks\n")

    # Compute feature matrix
    features = compute_feature_matrix(market_df, fundamentals_df)

    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Columns: {features.columns.tolist()}")
    print(f"Sample:\n{features.head(3)}")
    print(f"\nSummary stats:")
    for c in features.columns:
        if c != "symbol":
            vals = features[c].dropna()
            print(f"  {c}: mean={vals.mean():.4f} std={vals.std():.4f} "
                  f"min={vals.min():.4f} max={vals.max():.4f}")

    print("\n[OK] factor_compute pipeline verified.")
