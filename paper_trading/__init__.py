"""
Paper Trading Pipeline — AkShare-based daily data ingestion, state management,
and month-end rebalancing scheduler for ProductionAlphaEngine.

Modules:
  - data_ingestion.py    : Robust AkShare data fetching with retry + type safety
  - state_manager.py     : SQLite-backed 60-day market cache + prev_signal anchor
  - factor_compute.py    : 16-factor cross-sectional computation (replicating training pipeline)
  - paper_trading_pipeline.py : Daily cron job + month-end rebalance orchestrator
"""
