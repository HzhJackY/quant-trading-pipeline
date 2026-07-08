"""Cost sensitivity and narrow Top50 Buffer robustness validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_v15_portfolio_optimization import (
    StrategySpec,
    _annualized_sharpe,
    _max_drawdown,
    load_frozen_inputs,
    simulate_strategy,
    summarize_monthly,
)

OUTPUT_DIR = Path("output")
MONTHLY_INPUT = OUTPUT_DIR / "compact_f_portfolio_construction_monthly.csv"
COST_MD = OUTPUT_DIR / "compact_f_cost_sensitivity.md"
COST_CSV = OUTPUT_DIR / "compact_f_cost_sensitivity.csv"
ROBUST_MD = OUTPUT_DIR / "compact_f_top50_buffer_robustness.md"
ROBUST_CSV = OUTPUT_DIR / "compact_f_top50_buffer_robustness.csv"
DECISION_MD = OUTPUT_DIR / "compact_f_production_candidate_decision.md"

COST_LEVELS = [0, 10, 20, 30, 50]
BUFFER_GRID = [(30, 70), (35, 75), (40, 80)]


def apply_transaction_cost(monthly: pd.DataFrame, cost_bps: int) -> pd.Series:
    """Deduct one-way turnover cost; first missing turnover incurs zero cost."""
    cost = monthly["turnover"].fillna(0.0) * cost_bps / 10000.0
    return monthly["portfolio_return"] - cost


def _cost_sensitivity(monthly_all: pd.DataFrame) -> pd.DataFrame:
    strategy_map = {
        "Top30 Baseline": "A_Fixed_Top30",
        "Top50 Buffer": "F_Top50_Buffer",
    }
    rows = []
    for label, strategy in strategy_map.items():
        monthly = monthly_all[monthly_all["strategy"] == strategy].copy()
        realized_gross = monthly["portfolio_return"].dropna()
        gross_sharpe = _annualized_sharpe(realized_gross)
        avg_turnover = float(monthly["turnover"].dropna().mean())
        for cost_bps in COST_LEVELS:
            net = apply_transaction_cost(monthly, cost_bps).dropna()
            rows.append({
                "Strategy": label,
                "Cost_bps": cost_bps,
                "GrossSharpe": gross_sharpe,
                "NetSharpe": _annualized_sharpe(net),
                "NetMaxDD": _max_drawdown(net),
                "AvgTurnover": avg_turnover,
                "NetMeanMonthlyReturn": float(net.mean()),
                "HitRate": float((net > 0).mean()),
            })
    return pd.DataFrame(rows)


def _buffer_robustness() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions, panel = load_frozen_inputs()
    baseline_monthly = simulate_strategy(
        predictions,
        panel,
        StrategySpec("A_Fixed_Top30", "fixed", 30),
    )
    baseline = summarize_monthly(baseline_monthly)
    rows = []
    monthly_frames = []
    for buy, sell in BUFFER_GRID:
        name = f"Top50_Buffer_{buy}_{sell}"
        monthly = simulate_strategy(
            predictions,
            panel,
            StrategySpec(
                name,
                "buffer",
                50,
                buy_rank=buy,
                sell_rank=sell,
            ),
        )
        monthly_frames.append(monthly)
        metrics = summarize_monthly(monthly)
        avg_holding = float(monthly["holding_count"].mean())
        underfill = int(monthly["buy_zone_underfilled"].fillna(False).sum())
        passed = bool(
            metrics["Sharpe"] >= baseline["Sharpe"]
            and metrics["MaxDD"] >= baseline["MaxDD"] - 0.02
            and metrics["Turnover"] <= 0.35
            and metrics["ROE"] > 0
            and metrics["ProfitGrowth"] > 0
        )
        rows.append({
            "Buy": buy,
            "Sell": sell,
            "Sharpe": metrics["Sharpe"],
            "MaxDD": metrics["MaxDD"],
            "Turnover": metrics["Turnover"],
            "AvgHoldingCount": avg_holding,
            "UnderfillMonths": underfill,
            "ROE": metrics["ROE"],
            "ProfitGrowth": metrics["ProfitGrowth"],
            "EP": metrics["EP"],
            "Pass": passed,
        })
    return pd.DataFrame(rows), pd.concat(monthly_frames, ignore_index=True)


def decide_production_candidate(
    cost_advantage: bool,
    passing_parameter_count: int,
    current_candidate_passes: bool,
) -> str:
    """Return A=default, B=research-only, C=fallback."""
    if not current_candidate_passes:
        return "C"
    if cost_advantage and passing_parameter_count >= 2:
        return "A"
    return "B"


def _write_cost_outputs(cost: pd.DataFrame) -> None:
    cost.to_csv(COST_CSV, index=False, encoding="utf-8-sig")
    display = cost.copy()
    for col in ("NetMaxDD", "AvgTurnover", "NetMeanMonthlyReturn", "HitRate"):
        display[col] *= 100
    comparisons = []
    for bps in [10, 20, 30, 50]:
        cross = cost[cost["Cost_bps"] == bps].set_index("Strategy")
        better = (
            cross.loc["Top50 Buffer", "NetSharpe"]
            > cross.loc["Top30 Baseline", "NetSharpe"]
        )
        comparisons.append(
            f"- {bps} bps: Top50 Buffer NetSharpe "
            f"{cross.loc['Top50 Buffer', 'NetSharpe']:.4f} vs Top30 "
            f"{cross.loc['Top30 Baseline', 'NetSharpe']:.4f} — "
            f"{'better' if better else 'not better'}."
        )
    COST_MD.write_text(
        "\n".join([
            "# Compact-F Cost Sensitivity",
            "",
            display.round(4).to_markdown(index=False),
            "",
            "Percent-formatted: NetMaxDD, AvgTurnover, "
            "NetMeanMonthlyReturn, HitRate.",
            "",
            "## Relative Result",
            "",
            *comparisons,
            "",
        ]),
        encoding="utf-8",
    )


def _write_robustness_outputs(robust: pd.DataFrame) -> None:
    robust.to_csv(ROBUST_CSV, index=False, encoding="utf-8-sig")
    display = robust.copy()
    display["MaxDD"] *= 100
    display["Turnover"] *= 100
    passing = int(robust["Pass"].sum())
    if passing == 3:
        verdict = "All three nearby parameter points pass; the buffer is robust."
    elif passing == 1 and bool(
        robust.loc[
            (robust["Buy"] == 35) & (robust["Sell"] == 75), "Pass"
        ].iloc[0]
    ):
        verdict = (
            "Only the current 35/75 point passes; the result is parameter "
            "sensitive and should be treated cautiously."
        )
    else:
        verdict = (
            f"{passing}/3 parameter points pass; robustness is incomplete."
        )
    ROBUST_MD.write_text(
        "\n".join([
            "# Compact-F Top50 Buffer Robustness",
            "",
            display.round(4).to_markdown(index=False),
            "",
            "Percent-formatted: MaxDD, Turnover.",
            "",
            "## Verdict",
            "",
            verdict,
            "",
        ]),
        encoding="utf-8",
    )


def _write_decision(
    decision: str,
    cost_advantage: bool,
    passing_count: int,
    current_passes: bool,
    robust: pd.DataFrame,
) -> None:
    labels = {
        "A": "Recommend Compact-F + Top50 Buffer as the default portfolio layer",
        "B": "Research candidate only; do not make it the default",
        "C": "Revert to Fixed Top30 baseline",
    }
    current = robust[
        (robust["Buy"] == 35) & (robust["Sell"] == 75)
    ].iloc[0]
    reasons = [
        f"- Cost-adjusted NetSharpe advantage at 10/20/30/50 bps: "
        f"**{cost_advantage}**.",
        f"- Passing robustness points: **{passing_count}/3**.",
        f"- Current 35/75 candidate passes hard criteria: "
        f"**{current_passes}**.",
        f"- Current turnover: **{current['Turnover']:.2%}**.",
        f"- Current ROE / ProfitGrowth / EP: "
        f"**{current['ROE']:.4f} / {current['ProfitGrowth']:.4f} / "
        f"{current['EP']:.4f}**.",
    ]
    DECISION_MD.write_text(
        "\n".join([
            "# Compact-F Production Candidate Decision",
            "",
            f"## Decision {decision}",
            "",
            f"**{labels[decision]}**",
            "",
            "## Reasons",
            "",
            *reasons,
            "",
            (
                "The decision emphasizes transaction-cost advantage, turnover "
                "reduction, parameter robustness, and preserved positive "
                "fundamental exposures rather than the small gross Sharpe "
                "difference alone."
            ),
            "",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    monthly = pd.read_csv(MONTHLY_INPUT, parse_dates=["date"])
    cost = _cost_sensitivity(monthly)
    robust, _ = _buffer_robustness()
    _write_cost_outputs(cost)
    _write_robustness_outputs(robust)

    net_pivot = cost[cost["Cost_bps"].isin([10, 20, 30, 50])].pivot(
        index="Cost_bps",
        columns="Strategy",
        values="NetSharpe",
    )
    cost_advantage = bool(
        (
            net_pivot["Top50 Buffer"]
            > net_pivot["Top30 Baseline"]
        ).all()
    )
    passing_count = int(robust["Pass"].sum())
    current_passes = bool(
        robust.loc[
            (robust["Buy"] == 35) & (robust["Sell"] == 75), "Pass"
        ].iloc[0]
    )
    decision = decide_production_candidate(
        cost_advantage,
        passing_count,
        current_passes,
    )
    _write_decision(
        decision,
        cost_advantage,
        passing_count,
        current_passes,
        robust,
    )
    print(cost.round(5).to_string(index=False))
    print()
    print(robust.round(5).to_string(index=False))
    print(f"\nDecision: {decision}")


if __name__ == "__main__":
    main()
