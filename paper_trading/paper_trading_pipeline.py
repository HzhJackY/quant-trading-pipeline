"""
Paper Trading Pipeline — Daily Cron Job & Month-End Rebalance Orchestrator.

This is the SINGLE ENTRY POINT for production paper trading.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  daily_job(current_date)                                │
  │    │                                                    │
  │    ├── Is today a trading day? ── No ──> exit           │
  │    │                                                    │
  │    ├── [Always] Fetch daily OHLCV + fundamentals        │
  │    ├── [Always] Append OHLCV -> market_cache (SQLite)    │
  │    │                                                    │
  │    └── Is today the last trading day of month?          │
  │         │                                               │
  │         ├── No ──> exit (daily job complete)            │
  │         │                                               │
  │         └── Yes ──> REBALANCE:                          │
  │           1. Query 60-day market cache                  │
  │           2. Compute 16-factor feature matrix           │
  │           3. Load prev_signal from signal_anchor        │
  │           4. Load ProductionAlphaEngine                 │
  │           5. engine.predict_cross_section(features, prev)│
  │           6. Print Top 30 buy targets                   │
  │           7. Write new signals -> signal_anchor          │
  └─────────────────────────────────────────────────────────┘

Usage:
  # Daily cron (run after market close, e.g., 16:00 CST):
  python paper_trading/paper_trading_pipeline.py

  # Test with a specific date:
  python paper_trading/paper_trading_pipeline.py --date 2026-06-05

  # Force rebalance (even if not month-end):
  python paper_trading/paper_trading_pipeline.py --date 2026-06-30 --force-rebalance

  # Re-fetch PIT financials (ignore parquet cache):
  python paper_trading/paper_trading_pipeline.py --date 2026-06-05 --force-refresh

  # Use baostock + ProcessPoolExecutor for 30x faster PIT fetch
  # (pubDate-gated, zero look-ahead bias):
  python paper_trading/paper_trading_pipeline.py --date 2026-06-05 --use-baostock
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# Ensure project root is on path (needed when invoked as `python paper_trading/...`)
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd

# ── Paper trading modules ──
from paper_trading.data_ingestion import (
    fetch_daily_market_data,
    fetch_daily_market_data_parallel,
    fetch_daily_fundamentals,
    fetch_daily_fundamentals_parallel,
    fetch_all_a_share_codes,
    fetch_csi800_members,
    fetch_and_align_financials,
    fetch_industry_classification,
    is_trade_date,
    is_month_last_trade_date,
    _clear_progress,
)
from paper_trading.state_manager import (
    StateManager,
    ym_from_date,
    prev_ym,
)
from paper_trading.factor_compute import (
    apply_risk_filters,
    compute_feature_matrix,
    validate_feature_columns,
    FACTOR_NAMES,
)

# ── Production engine ──
from factor_research.production_engine import ProductionAlphaEngine, ProductionConfig

# ── Market timing ──
from factor_research.market_timing import fetch_csi500, compute_market_multiplier

logger = logging.getLogger("paper_trading")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

DEFAULT_MODEL_DIR = Path("output/production_models_v2_full")
DEFAULT_DB_DIR = Path("output/paper_trading_db")
TOP_N_DISPLAY = 30


class PipelineConfig:
    """Runtime configuration for the paper trading pipeline."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        db_dir: str | Path = DEFAULT_DB_DIR,
        top_n: int = TOP_N_DISPLAY,
        force_rebalance: bool = False,
        skip_ingestion: bool = False,
        force_refresh: bool = False,
        use_baostock: bool = False,
    ):
        self.model_dir = Path(model_dir)
        self.db_dir = Path(db_dir)
        self.top_n = top_n
        self.force_rebalance = force_rebalance
        self.skip_ingestion = skip_ingestion
        self.force_refresh = force_refresh
        self.use_baostock = use_baostock


# ═══════════════════════════════════════════════════════════
# Core Pipeline
# ═══════════════════════════════════════════════════════════

class PaperTradingPipeline:
    """
    Production paper trading pipeline orchestrator.

    Lifecycle:
      1. pipeline = PaperTradingPipeline(config)
      2. pipeline.run(current_date)  — call daily after market close
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._state = StateManager(self.config.db_dir)
        self._state.init()
        self._engine: Optional[ProductionAlphaEngine] = None
        self._universe: Optional[pd.DataFrame] = None
        self._index_df: Optional[pd.DataFrame] = None  # CSI 500 cache for market timing

    # ── Main Entry Point ──────────────────────────────────

    def run(self, current_date: date | str | None = None):
        """
        Execute the daily pipeline.

        Args:
            current_date: The date to process. Default: today.
        """
        if current_date is None:
            current_date = date.today()
        if isinstance(current_date, str):
            current_date = datetime.strptime(current_date[:10], "%Y-%m-%d").date()

        logger.info("=" * 64)
        logger.info("Paper Trading Pipeline — %s", current_date.isoformat())
        logger.info("=" * 64)

        # ── Gate 1: Is today a trading day? ──
        if not is_trade_date(current_date):
            logger.info("[SKIP]  %s is not a trading day — nothing to do.", current_date)
            return

        # ── Step 1: Daily data ingestion ──
        if not self.config.skip_ingestion:
            self._daily_ingestion(current_date)
        else:
            logger.info("[SKIP]  Skipping ingestion (--skip-ingestion)")

        # ── Gate 2: Is today a rebalance day? ──
        is_rebalance = self.config.force_rebalance or is_month_last_trade_date(current_date)
        if not is_rebalance:
            logger.info("[OK]  Daily job complete (not month-end, no rebalance).")
            self._print_state_stats()
            return

        # ── Step 2-6: Month-end rebalance ──
        logger.info("=" * 64)
        logger.info("[REBAL] MONTH-END REBALANCE TRIGGERED — %s", current_date.isoformat())
        logger.info("=" * 64)
        self._execute_rebalance(current_date)

    # ═══════════════════════════════════════════════════════
    # Step 1: Daily Ingestion
    # ═══════════════════════════════════════════════════════

    def _daily_ingestion(self, current_date: date):
        """Fetch today's data and cache to SQLite (parallel + checkpoint/resume)."""
        trade_date_str = current_date.strftime("%Y%m%d")

        # ── 1a: Fetch OHLCV (parallel, ProcessPoolExecutor, baostock-safe) ──
        logger.info("[FETCH] Fetching daily OHLCV for %s (parallel, 10 workers)...", current_date)
        try:
            ohlcv = fetch_daily_market_data_parallel(
                trade_date=trade_date_str,
                universe=self._universe,
                max_workers=10,
                throttle_seconds=0.02,
            )
            if len(ohlcv) == 0:
                logger.warning("[WARN]  No OHLCV data returned — possible holiday or API issue.")
                return
            logger.info("  -> %d stocks fetched", len(ohlcv))
        except Exception as e:
            logger.error("[ERROR] OHLCV fetch failed: %s", e)
            raise

        # ── 1b: Append to market cache ──
        self._state.append_market_data(ohlcv)

        # ── 1c: Update universe reference ──
        if self._universe is None:
            try:
                self._universe = fetch_all_a_share_codes()
            except Exception:
                logger.warning("Cannot refresh universe — using symbols from OHLCV")
                self._universe = ohlcv[["symbol"]].drop_duplicates()

        # ── 1d: Clear OHLCV progress for next run ──
        # (each day's fetch is independent; old checkpoint is stale)
        _clear_progress("market_data")

    # ═══════════════════════════════════════════════════════
    # Steps 2-6: Month-End Rebalance
    # ═══════════════════════════════════════════════════════

    def _execute_rebalance(self, current_date: date):
        """
        Execute the full month-end rebalance pipeline:

        2. Query 60-day cache -> factor computation
        3. Retrieve prev_signal from anchor
        4. Load ProductionAlphaEngine
        5. predict_cross_section()
        6. Display Top N + persist signals
        """
        ym = ym_from_date(current_date)
        prev = prev_ym(ym)

        # ── Step 2a: Determine universe (CSI 800 ∩ risk-filtered) ──
        logger.info("[REBAL] Determining filtered universe...")
        try:
            csi800 = fetch_csi800_members()
            csi800_set = set(csi800["symbol"].tolist())
            logger.info("  CSI 800: %d constituent stocks", len(csi800_set))
        except Exception as e:
            logger.error("[ERROR] Failed to fetch CSI 800 members: %s", e)
            logger.warning("  Falling back to full A-share universe — results may be unreliable")
            csi800_set = None

        # Apply risk pre-filters (ST/suspension/liquidity/market cap)
        # Need market cache + fundamentals for the filter logic
        raw_market = self._state.query_market_cache(lookback_days=60)
        # Use the LATEST available fundamentals file (not date-specific)
        # — the filter only needs current snapshot data (ST flag, mcap, etc.)
        raw_fund = pd.DataFrame()
        fund_files = sorted(
            self.config.db_dir.glob("fundamentals_*.parquet"), reverse=True
        )
        if fund_files:
            try:
                raw_fund = pd.read_parquet(fund_files[0])
                logger.info("  Using fundamentals: %s (%d stocks)",
                           fund_files[0].name, len(raw_fund))
            except Exception as e:
                logger.warning("  Failed to load fundamentals for risk filters: %s", e)

        risk_valid: set | None = None
        if not raw_market.empty and not raw_fund.empty:
            risk_valid = apply_risk_filters(raw_market, raw_fund)
        else:
            logger.warning("  [RiskFilter] Skipped — missing market or fundamentals data")

        # Combine: CSI 800 ∩ risk-filtered
        if csi800_set is not None and risk_valid is not None:
            final_universe = csi800_set & risk_valid
            logger.info(
                "  Final universe: CSI 800 (%d) ∩ Risk-passed (%d) = %d stocks",
                len(csi800_set), len(risk_valid), len(final_universe),
            )
        elif csi800_set is not None:
            final_universe = csi800_set
            logger.info("  Final universe: CSI 800 only — %d stocks (%d risk-filtered",
                       len(final_universe), len(risk_valid) if risk_valid else 0)
        elif risk_valid is not None:
            final_universe = risk_valid
            logger.info("  Final universe: Risk-filtered only — %d stocks (no CSI 800 restriction)",
                       len(final_universe))
        else:
            final_universe = None

        if final_universe is not None and len(final_universe) < 60:
            logger.warning(
                "  ⚠ Final universe has only %d stocks — may be insufficient for Top 30 selection",
                len(final_universe),
            )

        # ── Step 2b: Compute feature matrix ──
        logger.info("[REBAL] Computing 16-factor feature matrix...")
        features_df = self._compute_features(current_date, universe_symbols=final_universe)
        if features_df is None or len(features_df) == 0:
            logger.error("[ERROR] Feature computation returned empty — aborting rebalance.")
            return

        logger.info("  -> %d stocks × %d features", len(features_df),
                     len(features_df.columns) - 1)

        # ── Step 3: Load prev_signal ──
        logger.info("[ANCHOR] Loading prev_signal from anchor (ym=%s)...", prev)
        symbols = features_df["symbol"].tolist()
        prev_signal_series = self._state.get_prev_signal(prev, symbols=symbols)

        # Handle first-ever rebalance (no previous signal)
        if prev_signal_series is None or len(prev_signal_series) == 0:
            logger.info("  -> No previous signal found — cold start with 0.5")
            prev_signal_series = pd.Series(0.5, index=pd.Index(symbols, name="symbol"))

        # Align prev_signal to feature matrix order
        prev_signal = prev_signal_series.reindex(symbols).fillna(0.5)
        prev_array = prev_signal.values.astype(np.float64)

        logger.info("  -> %d signals loaded (%d filled with 0.5)",
                     len(prev_signal),
                     (prev_signal == 0.5).sum())

        # ── Step 4: Load engine ──
        engine = self._get_engine()

        # Validate feature columns against engine expectations
        is_valid, missing, extra = validate_feature_columns(
            features_df, engine.feature_cols
        )
        if not is_valid:
            logger.error("[ERROR] Feature column mismatch!")
            logger.error("  Missing: %s", missing)
            logger.error("  Extra:   %s", extra)
            logger.error("  Engine expects: %s", engine.feature_cols)
            logger.error("  Features have: %s",
                         [c for c in features_df.columns if c != "symbol"])
            # Attempt recovery: compute only mandatory factors
            logger.warning("Attempting recovery with reduced factor set...")
            # Build features with only the intersection
            common_cols = [c for c in engine.feature_cols if c in features_df.columns]
            if len(common_cols) < 10:
                logger.error("[ERROR] Too few matching columns (%d) — cannot recover.", len(common_cols))
                return
            logger.warning("Proceeding with %d/%d matching columns", len(common_cols), len(engine.feature_cols))
        else:
            logger.info("[OK] Feature columns validated: %d columns match engine expectations.",
                        len(engine.feature_cols))

        # ── Step 5: Predict ──
        logger.info("[INFER] Running multi-seed ensemble inference...")
        # Prepare features in the order expected by the engine
        feature_cols_aligned = [c for c in engine.feature_cols if c in features_df.columns]
        X = features_df[feature_cols_aligned].copy()
        # Fill any NaN (should not happen after factor pipeline, but defense)
        X = X.fillna(0.5)

        try:
            signals = engine.predict_cross_section(
                features=X,
                prev_signal=prev_array,
                fold_idx=-1,
                rank_output=True,
            )
        except Exception as e:
            logger.error("[ERROR] predict_cross_section() failed: %s", e)
            raise

        logger.info("  -> Signal range: [%.4f, %.4f] | mean=%.4f | std=%.4f",
                     signals.min(), signals.max(), signals.mean(), signals.std())

        # ── Step 6: Output & persist ──
        signal_df = pd.DataFrame({
            "symbol": symbols,
            "alpha_signal": signals,
        }).sort_values("alpha_signal", ascending=False).reset_index(drop=True)

        # 择时乘数 (基于中证 500 MA20/MA60 + 波动率区间)
        timing_mult = self._compute_timing_multiplier(current_date)
        self._print_top_picks(signal_df, ym, timing_multiplier=timing_mult)
        self._state.write_signal_anchor(ym, signal_df.set_index("symbol")["alpha_signal"])
        self._print_state_stats()

        mult_tag = f" | 择时乘数={timing_mult:.1f}" if timing_mult < 1.0 else ""
        logger.info("=" * 64)
        logger.info("[OK] Month-end rebalance complete — %s | %d stocks processed%s",
                     ym, len(signal_df), mult_tag)
        logger.info("=" * 64)

    # ═══════════════════════════════════════════════════════
    # Internal Helpers
    # ═══════════════════════════════════════════════════════

    def _compute_features(
        self, current_date: date, universe_symbols: set | None = None,
    ) -> Optional[pd.DataFrame]:
        """
        Query 60-day market cache + fundamentals, compute feature matrix.

        Incorporates:
          - Universe restriction (CSI 800) — enforced BEFORE ranking
          - Risk pre-filters (ST/suspension/liquidity/market cap)
          - PIT-aligned financial statements (no look-ahead bias)
          - SW industry classification (cached, refreshed monthly)

        Args:
            current_date: Rebalance date.
            universe_symbols: If provided, restrict to these symbols only.
                Must be 6-digit string codes.

        Returns:
            pd.DataFrame ready for predict_cross_section(), or None on failure.
        """
        # ── Query market cache ──
        market_df = self._state.query_market_cache(lookback_days=60)
        if len(market_df) == 0:
            logger.error("Market cache is empty — run daily ingestion first.")
            return None

        logger.info("  Market cache: %d rows, %d symbols, %d dates",
                     len(market_df), market_df["symbol"].nunique(),
                     market_df["trade_date"].nunique())

        # ── Fetch fundamentals (parallel, baostock-safe) ──
        # Cache fundamentals to parquet (same pattern as PIT financials)
        fund_cache_path = self.config.db_dir / f"fundamentals_{current_date.strftime('%Y%m%d')}.parquet"

        if not self.config.force_refresh and fund_cache_path.exists():
            fund_df = pd.read_parquet(fund_cache_path)
            logger.info("  Fundamentals: %d stocks (from cache)", len(fund_df))
        else:
            try:
                fund_df = fetch_daily_fundamentals_parallel(
                    universe=self._universe,
                    max_workers=10,
                )
                fund_df.to_parquet(fund_cache_path, index=False)
                logger.info("  Fundamentals: %d stocks (cached → %s)",
                           len(fund_df), fund_cache_path.name)
            except Exception as e:
                logger.error("[ERROR] Parallel fundamentals failed: %s", e)
                # Fall back to sequential if parallel fails
                logger.warning("Retrying fundamentals with sequential fallback...")
                try:
                    fund_df = fetch_daily_fundamentals(universe=self._universe)
                    fund_df.to_parquet(fund_cache_path, index=False)
                    logger.info("  Fundamentals (sequential): %d stocks (cached → %s)",
                               len(fund_df), fund_cache_path.name)
                except Exception as e2:
                    logger.error("[ERROR] Sequential fundamentals also failed: %s", e2)
                    # Last resort: try loading from cache even if force_refresh was set
                    if fund_cache_path.exists():
                        logger.warning("  Falling back to stale fundamentals cache")
                        fund_df = pd.read_parquet(fund_cache_path)
                    else:
                        raise
        # Clear progress checkpoint for next month's run (only on success)
        _clear_progress("fundamentals")

        # ── Fetch PIT-aligned financials (statutory disclosure gating) ──
        pit_financials = None
        try:
            pit_financials = fetch_and_align_financials(
                current_date,
                universe=self._universe,
                force_refresh=self.config.force_refresh,
                use_baostock=self.config.use_baostock,
            )
            if pit_financials is not None and len(pit_financials) > 0:
                n_periods = pit_financials["report_period"].nunique()
                latest_period = pit_financials["report_period"].max()
                logger.info("  PIT financials: %d stocks, %d periods, latest=%s",
                             len(pit_financials), n_periods,
                             str(latest_period)[:10] if pd.notna(latest_period) else "N/A")
            else:
                logger.warning("  [WARN] PIT financials returned empty — "
                               "Debt/Rev/Profit factors will use spot data only")
        except Exception as e:
            logger.warning(
                "[WARN] PIT financials fetch failed: %s — "
                "Debt_Ratio/RevGrowth/ProfitGrowth will fall back to spot data", e)

        # ── Fetch/cache SW industry classification ──
        industry_map = None
        try:
            if not self._state.is_industry_cache_fresh(max_age_days=30):
                logger.info("  Industry cache stale — fetching SW classification...")
                industry_df = fetch_industry_classification()
                self._state.update_industry_cache(industry_df)
                logger.info("  Industry cache updated: %d stocks", len(industry_df))
            else:
                logger.info("  Industry cache fresh — using cached classification")
            industry_map = self._state.get_industry_map()
            if industry_map is not None:
                logger.info("  Industry map: %d stocks, %d unique industries",
                             len(industry_map), industry_map.nunique())
        except Exception as e:
            logger.warning(
                "[WARN] Industry classification failed: %s — "
                "using global mean neutralization", e)

        # ── Compute factors (WITHIN restricted universe) ──
        features = compute_feature_matrix(
            market_df,
            fund_df,
            pit_financials=pit_financials,
            industry_map=industry_map,
            universe_symbols=universe_symbols,
        )
        return features

    def _get_engine(self) -> ProductionAlphaEngine:
        """Lazy-load and cache the ProductionAlphaEngine."""
        if self._engine is not None:
            return self._engine

        model_dir = self.config.model_dir
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. "
                f"Run the following command first:\n"
                f"  python -c \"from factor_research.production_engine import train_and_save; train_and_save()\"\n"
                f"Or in production mode:\n"
                f"  python -c \"from factor_research.production_engine import train_and_save; train_and_save(mode='production')\""
            )

        logger.info("📦 Loading ProductionAlphaEngine from %s ...", model_dir)
        self._engine = ProductionAlphaEngine.load_models(model_dir)
        logger.info("  -> %d models loaded | %d features | %d folds | seeds=%s",
                     self._engine.n_models, len(self._engine.feature_cols),
                     self._engine._n_folds, self._engine.seeds)
        return self._engine

    def _print_top_picks(
        self,
        signal_df: pd.DataFrame,
        ym: str,
        timing_multiplier: float = 1.0,
    ):
        """
        Display top N buy targets with signal strength and position sizing.

        timing_multiplier: Market timing scale factor.
          - 1.0 → full allocation (each stock = 1/N)
          - 0.3 → reduced allocation (each stock = 0.3/N, rest in cash)
        """
        n = min(self.config.top_n, len(signal_df))
        top = signal_df.head(n)
        weight_per_stock = timing_multiplier / n  # e.g., 0.3/30 = 0.01

        print(f"\n{'='*72}")
        print(f"   TOP {n} BUY TARGETS — {ym}")
        if timing_multiplier < 1.0:
            print(f"   {'='*68}")
            print(f"   [!] MARKET TIMING TRIGGERED -- position multiplier={timing_multiplier:.1f}")
            print(f"   Total equity exposure: {timing_multiplier:.0%} | "
                  f"Cash reserve: {1-timing_multiplier:.0%}")
            print(f"   {'='*68}")
        print(f"{'='*72}")
        print(f"  {'Rank':<6} {'Symbol':<10} {'Signal':>8}  {'Weight':>7}  {'Indicative $':>12}")
        print(f"  {'-'*49}")

        for rank, (_, row) in enumerate(top.iterrows(), 1):
            bar_len = int(row["alpha_signal"] * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            # Indicative dollar per stock per $1M AUM
            indicative = weight_per_stock * 1_000_000
            print(f"  {rank:<6} {row['symbol']:<10} {row['alpha_signal']:>8.4f}  "
                  f"{weight_per_stock:>7.4f}  ${indicative:>10,.0f}  {bar}")

        # Distribution stats
        bottom = signal_df.tail(n)
        total_equity_pct = timing_multiplier * 100
        print(f"\n  {'─'*49}")
        print(f"  Top {n} mean signal:     {top['alpha_signal'].mean():.4f}")
        print(f"  Bottom {n} mean signal:  {bottom['alpha_signal'].mean():.4f}")
        print(f"  Signal Spread:           {top['alpha_signal'].mean() - bottom['alpha_signal'].mean():.4f}")
        print(f"  Universe: {len(signal_df)} stocks | "
              f"Median: {signal_df['alpha_signal'].median():.4f} | "
              f"Std: {signal_df['alpha_signal'].std():.4f}")
        print(f"  Portfolio: {n} stocks × {weight_per_stock:.4f} = "
              f"{total_equity_pct:.0f}% equity / {1-timing_multiplier:.0%} cash")
        print(f"{'='*72}\n")

    def _print_state_stats(self):
        """Print database state summary."""
        stats = self._state.stats()
        logger.info(
            "State: %d market rows (%d dates) | %d signal months | latest: %s",
            stats["market_rows"], stats["market_dates"],
            stats["signal_months"], stats["latest_signal_ym"] or "none",
        )

    # ═══════════════════════════════════════════════════════════
    # Market Timing Helpers
    # ═══════════════════════════════════════════════════════════

    def _compute_timing_multiplier(self, current_date: date) -> float:
        """
        Compute the market timing position sizing multiplier.

        Fetches CSI 500 data (cached across calls), computes MA20/MA60
        death cross and volatility regime.

        Returns:
            float: 0.3 (triggered) or 1.0 (normal).
        """
        if self._index_df is None:
            start = (current_date - timedelta(days=700)).isoformat()
            try:
                self._index_df = fetch_csi500(start_date=start, use_cache=True)
            except Exception as e:
                logger.warning("[择时] fetch_csi500 failed: %s — defaulting to 1.0", e)
                return 1.0

        try:
            mult = compute_market_multiplier(self._index_df, current_date)
            return mult
        except Exception as e:
            logger.warning("[择时] compute_market_multiplier failed: %s — defaulting to 1.0", e)
            return 1.0


# ═══════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Paper Trading Pipeline — Daily ingestion + month-end rebalance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paper_trading/paper_trading_pipeline.py
  python paper_trading/paper_trading_pipeline.py --date 2026-06-30
  python paper_trading/paper_trading_pipeline.py --date 2026-06-05 --force-rebalance
  python paper_trading/paper_trading_pipeline.py --date 2026-06-07 --skip-ingestion
  python paper_trading/paper_trading_pipeline.py --date 2026-06-05 --force-refresh
  python paper_trading/paper_trading_pipeline.py --date 2026-06-05 --use-baostock
        """,
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Target date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR),
                        help="Path to trained ProductionAlphaEngine models.")
    parser.add_argument("--db-dir", type=str, default=str(DEFAULT_DB_DIR),
                        help="Path to SQLite state database.")
    parser.add_argument("--top-n", type=int, default=TOP_N_DISPLAY,
                        help="Number of top picks to display.")
    parser.add_argument("--force-rebalance", action="store_true",
                        help="Force month-end rebalance even if not last trading day.")
    parser.add_argument("--skip-ingestion", action="store_true",
                        help="Skip daily data ingestion (use existing cache).")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force re-fetch of PIT financials (ignore parquet cache).")
    parser.add_argument("--use-baostock", action="store_true",
                        help="Use baostock + ProcessPoolExecutor for PIT financials "
                             "(pubDate-gated, ~30× faster than Eastmoney).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for name in ["data_ingestion", "state_manager", "factor_compute", "paper_trading"]:
            logging.getLogger(name).setLevel(logging.DEBUG)

    # Parse date
    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    config = PipelineConfig(
        model_dir=args.model_dir,
        db_dir=args.db_dir,
        top_n=args.top_n,
        force_rebalance=args.force_rebalance,
        skip_ingestion=args.skip_ingestion,
        force_refresh=args.force_refresh,
        use_baostock=args.use_baostock,
    )

    pipeline = PaperTradingPipeline(config)

    # Auto-retry on transient network errors (Windows socket issues, etc.)
    max_retries = 2
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            pipeline.run(target_date)
            break
        except KeyboardInterrupt:
            logger.info("⏹  Pipeline interrupted by user.")
            sys.exit(0)
        except (OSError, ConnectionError, TimeoutError) as e:
            last_error = e
            if attempt < max_retries:
                wait = (attempt + 1) * 30
                logger.warning(
                    "[RETRY] Network error (attempt %d/%d): %s — "
                    "waiting %ds before retry...",
                    attempt + 1, max_retries, e, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "[ERROR] Pipeline failed after %d retries: %s",
                    max_retries, e, exc_info=args.verbose,
                )
                sys.exit(1)
        except Exception as e:
            logger.error("[ERROR] Pipeline failed: %s", e, exc_info=args.verbose)
            sys.exit(1)


# ═══════════════════════════════════════════════════════════
# Standalone test (dry-run with synthetic cache)
# ═══════════════════════════════════════════════════════════

def dry_run_test():
    """
    End-to-end test of the pipeline WITHOUT calling AkShare.

    Populates the state manager with synthetic market data, then
    runs a simulated month-end rebalance.
    """
    import tempfile, os
    np.random.seed(42)

    print("=" * 64)
    print("  PAPER TRADING PIPELINE — DRY RUN TEST")
    print("=" * 64)

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Setup ──
        db_dir = Path(tmpdir) / "test_db"
        config = PipelineConfig(
            model_dir=DEFAULT_MODEL_DIR,
            db_dir=db_dir,
            force_rebalance=True,
            skip_ingestion=True,  # We'll inject data manually
        )

        pipeline = PaperTradingPipeline(config)

        # ── Inject synthetic 60-day market data ──
        symbols = [f"{i:06d}" for i in range(1, 101)]
        boards = np.random.choice(["沪市主板", "深市主板", "创业板", "科创板"], 100)
        dates = pd.date_range("2026-04-01", "2026-06-07", freq="B")

        rows = []
        for d in dates:
            for i, s in enumerate(symbols):
                base = 10.0 + i * 0.5
                close = base * (1 + np.random.normal(0, 0.02))
                rows.append({
                    "date": d,
                    "symbol": s,
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "volume": np.random.lognormal(14, 1),
                    "amount": np.random.lognormal(16, 1),
                    "pct_change": np.random.normal(0, 0.02),
                    "turnover_rate": np.random.uniform(0.1, 5),
                })
        pipeline._state.append_market_data(pd.DataFrame(rows))
        pipeline._state.write_signal_anchor(
            "2026-05",
            pd.Series(np.random.uniform(0.3, 0.7, 100), index=symbols),
        )

        # Inject synthetic fundamentals into the pipeline by mocking
        # We override _compute_features to use synthetic data
        orig_compute = pipeline._compute_features

        def mock_compute_features(current_date=None):
            fund_df = pd.DataFrame({
                "symbol": symbols,
                "name": [f"Stock_{s}" for s in symbols],
                "pe_ttm": np.random.uniform(5, 200, 100),
                "pb": np.random.uniform(0.5, 15, 100),
                "total_mcap": np.random.lognormal(23, 1.5, 100),
                "float_mcap": np.random.lognormal(22, 1.5, 100),
                "roe": np.random.uniform(-0.3, 0.4, 100),
                "eps": np.random.uniform(-2, 10, 100),
                "bps": np.random.uniform(1, 30, 100),
                "net_margin": np.random.uniform(-0.5, 0.6, 100),
                "gross_margin": np.random.uniform(0.05, 0.7, 100),
                "board": boards,
            })
            market_df = pipeline._state.query_market_cache(lookback_days=60)
            return compute_feature_matrix(market_df, fund_df)

        pipeline._compute_features = mock_compute_features

        # ── Run ──
        try:
            pipeline.run(date(2026, 6, 30))
            print("\n[OK] Dry run pipeline executed successfully!")
            print(f"   DB stats: {pipeline._state.stats()}")
        except FileNotFoundError as e:
            print(f"\n[WARN]  Engine models not found at {DEFAULT_MODEL_DIR}")
            print("   (This is expected if you haven't trained the production engine yet.)")
            print(f"   Error: {e}")
        except Exception as e:
            print(f"\n[ERROR] Dry run failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        dry_run_test()
    else:
        main()
