# Compact-F Portfolio Construction Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backtest six monthly portfolio construction methods on frozen Compact-F OOS predictions and select a method only if it passes all turnover, risk, return, and style constraints.

**Architecture:** Convert each strategy into a monthly normalized weight vector keyed by symbol. A common accounting engine computes forward returns, one-way turnover, exposures, annual statistics, and audit fields from those weights.

**Tech Stack:** Python, pandas, NumPy, pytest, parquet.

## Global Constraints

- Read existing Compact-F OOS signals; never retrain or mutate the model.
- Rebalance only at month-end and use next-month forward returns.
- Use weight turnover `0.5 * sum(abs(w_t - w_t-1))`.
- Buffer strategies never buy outside their approved buy zones.
- Partial Rebalance always normalizes to full investment and records raw/final weight totals.
- If no candidate passes all conditions, retain Fixed Top30.

---

### Task 1: Strategy Weight State Machines

**Files:**
- Create: `run_v15_portfolio_optimization.py`
- Create: `tests/test_v15_portfolio_optimization.py`

**Interfaces:**
- Produce:
  - `fixed_top_n_weights(ranked, n) -> dict[str, float]`
  - `buffer_weights(ranked, previous, target_size, buy_rank, sell_rank) -> tuple[dict, dict]`
  - `partial_rebalance_weights(ranked, previous, n=30, alpha=0.5) -> tuple[dict, dict]`

- [ ] Write failing tests for exact fixed Top-N membership and equal weights.
- [ ] Write failing tests proving Buffer sells only beyond sell threshold, buys only inside buy threshold, and records underfill.
- [ ] Write failing tests proving Partial Rebalance decays sold names, ramps new names, removes unavailable names, and normalizes to one.
- [ ] Run `python -m pytest tests/test_v15_portfolio_optimization.py -q` and verify failure from missing module.
- [ ] Implement minimal strategy functions.
- [ ] Re-run tests and verify green.

### Task 2: Common Monthly Accounting

**Files:**
- Modify: `run_v15_portfolio_optimization.py`
- Modify: `tests/test_v15_portfolio_optimization.py`

**Interfaces:**
- Produce:
  - `weight_turnover(previous, current) -> float`
  - `simulate_strategy(predictions, panel, strategy) -> pd.DataFrame`
  - `summarize_monthly(monthly) -> dict`
  - `yearly_sharpe(monthly) -> pd.DataFrame`

- [ ] Write failing tests for one-way weight turnover.
- [ ] Write failing tests for weighted next-month return and weighted ROE/PG/EP exposures.
- [ ] Write failing tests that the last OOS month is excluded from return metrics.
- [ ] Implement the common accounting engine.
- [ ] Verify all strategy and accounting tests pass.

### Task 3: Six-Strategy Orchestration and Reports

**Files:**
- Modify: `run_v15_portfolio_optimization.py`
- Generate: `output/compact_f_portfolio_construction_results.md`
- Generate: `output/compact_f_portfolio_construction_monthly.csv`
- Generate: `output/compact_f_portfolio_construction_yearly.csv`

**Interfaces:**
- Produce:
  - `run_portfolio_optimization() -> pd.DataFrame`
  - `select_winner(results) -> str`

- [ ] Write failing tests for all six strategy names and acceptance selection.
- [ ] Implement frozen input loading and forward-return construction.
- [ ] Simulate all six methods and concatenate monthly/yearly audit outputs.
- [ ] Generate comparison Markdown with explicit acceptance verdicts.
- [ ] Run syntax checks, unit tests, full backtest, and a second reproducibility run.
- [ ] Report actual evidence without selecting a complex method unless it passes every rule.
