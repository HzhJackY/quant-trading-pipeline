import numpy as np
import pandas as pd
import pytest

from run_v15_portfolio_optimization import (
    STRATEGIES,
    StrategySpec,
    buffer_weights,
    fixed_top_n_weights,
    partial_rebalance_weights,
    simulate_strategy,
    select_winner,
    summarize_monthly,
    weight_turnover,
    yearly_sharpe,
)


def _ranked(n=100):
    return pd.DataFrame({
        "symbol": [f"{i:06d}" for i in range(1, n + 1)],
        "rank": np.arange(1, n + 1),
    })


def test_fixed_top_n_is_exact_and_equal_weighted():
    weights = fixed_top_n_weights(_ranked(), 30)

    assert list(weights) == [f"{i:06d}" for i in range(1, 31)]
    assert len(weights) == 30
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(weight == pytest.approx(1 / 30) for weight in weights.values())


def test_buffer_sells_beyond_threshold_and_buys_only_inside_buy_zone():
    ranked = _ranked()
    previous = {f"{i:06d}": 1 / 30 for i in range(16, 46)}

    weights, audit = buffer_weights(
        ranked,
        previous,
        target_size=30,
        buy_rank=20,
        sell_rank=45,
    )

    assert set(weights) == {f"{i:06d}" for i in range(16, 46)}
    assert audit["sold_count"] == 0
    assert audit["bought_count"] == 0
    assert audit["holding_count"] == 30
    assert audit["buy_zone_underfilled"] is False


def test_buffer_records_underfill_when_buy_zone_is_insufficient():
    ranked = _ranked()
    previous = {f"{i:06d}": 1 / 30 for i in range(40, 70)}

    weights, audit = buffer_weights(
        ranked,
        previous,
        target_size=30,
        buy_rank=20,
        sell_rank=45,
    )

    # Retain ranks 40..45 (6 names), then buy ranks 1..20 (20 names).
    assert len(weights) == 26
    assert all(int(symbol) <= 20 or 40 <= int(symbol) <= 45 for symbol in weights)
    assert audit == {
        "sold_count": 24,
        "bought_count": 20,
        "holding_count": 26,
        "buy_zone_underfilled": True,
    }
    assert sum(weights.values()) == pytest.approx(1.0)


def test_partial_rebalance_decays_sold_names_and_ramps_new_names():
    ranked = _ranked(40)
    previous = {f"{i:06d}": 1 / 30 for i in range(11, 41)}

    weights, audit = partial_rebalance_weights(
        ranked,
        previous,
        n=30,
        alpha=0.5,
    )

    # 1..10 are new, 11..30 overlap, 31..40 decay.
    assert weights["000001"] == pytest.approx(1 / 60)
    assert weights["000011"] == pytest.approx(1 / 30)
    assert weights["000031"] == pytest.approx(1 / 60)
    assert audit["pre_normalization_weight"] == pytest.approx(1.0)
    assert audit["post_normalization_weight"] == pytest.approx(1.0)
    assert audit["holding_count"] == 40


def test_partial_rebalance_removes_names_absent_from_current_universe():
    ranked = _ranked(30)
    previous = {**{f"{i:06d}": 1 / 30 for i in range(1, 30)}, "999999": 1 / 30}

    weights, audit = partial_rebalance_weights(ranked, previous, n=30, alpha=0.5)

    assert "999999" not in weights
    assert sum(weights.values()) == pytest.approx(1.0)
    assert audit["post_normalization_weight"] == pytest.approx(1.0)


def test_weight_turnover_uses_half_l1_distance():
    previous = {"A": 0.5, "B": 0.5}
    current = {"B": 0.5, "C": 0.5}

    assert weight_turnover(previous, current) == pytest.approx(0.5)


def _simulation_fixture():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    signals = []
    panel = []
    for date_index, date in enumerate(dates):
        for i in range(1, 41):
            signals.append({
                "date": date,
                "symbol": f"{i:06d}",
                "alpha_signal": float(i if date_index != 1 else 41 - i),
            })
            panel.append({
                "date": date,
                "symbol": f"{i:06d}",
                "forward_return_1m": (
                    i / 1000 if date != pd.Timestamp("2024-03-31") else np.nan
                ),
                "SR_ROE_neutral_z": i / 10,
                "SR_ProfitGrowth_YoY_neutral_z": i / 20,
                "EP_neutral_z": i / 30,
            })
    return pd.DataFrame(signals), pd.DataFrame(panel)


def test_simulation_uses_weighted_forward_return_and_exposures():
    predictions, panel = _simulation_fixture()
    spec = StrategySpec(name="A_Fixed_Top30", kind="fixed", target_size=30)

    monthly = simulate_strategy(predictions, panel, spec)
    january = monthly[monthly["date"] == pd.Timestamp("2024-01-31")].iloc[0]

    # Signal ascends with i, so Top30 is i=11..40; average i is 25.5.
    assert january["portfolio_return"] == pytest.approx(0.0255)
    assert january["roe_exposure"] == pytest.approx(2.55)
    assert january["profitgrowth_exposure"] == pytest.approx(1.275)
    assert january["ep_exposure"] == pytest.approx(0.85)
    assert january["weight_sum"] == pytest.approx(1.0)


def test_last_oos_month_is_excluded_from_return_summary():
    predictions, panel = _simulation_fixture()
    spec = StrategySpec(name="A_Fixed_Top30", kind="fixed", target_size=30)

    monthly = simulate_strategy(predictions, panel, spec)
    summary = summarize_monthly(monthly)

    assert summary["N_Months"] == 2
    assert np.isnan(monthly.iloc[-1]["portfolio_return"])


def test_yearly_sharpe_counts_only_realized_months():
    predictions, panel = _simulation_fixture()
    spec = StrategySpec(name="A_Fixed_Top30", kind="fixed", target_size=30)

    monthly = simulate_strategy(predictions, panel, spec)
    yearly = yearly_sharpe(monthly)

    assert yearly["n_months"].tolist() == [2]


def test_six_strategy_definitions_are_exact():
    assert list(STRATEGIES) == [
        "A_Fixed_Top30",
        "B_Top30_Buffer",
        "C_Top30_Partial",
        "D_Fixed_Top40",
        "E_Fixed_Top50",
        "F_Top50_Buffer",
    ]
    assert STRATEGIES["B_Top30_Buffer"] == StrategySpec(
        "B_Top30_Buffer", "buffer", 30, buy_rank=20, sell_rank=45
    )
    assert STRATEGIES["C_Top30_Partial"] == StrategySpec(
        "C_Top30_Partial", "partial", 30, alpha=0.5
    )
    assert STRATEGIES["F_Top50_Buffer"] == StrategySpec(
        "F_Top50_Buffer", "buffer", 50, buy_rank=35, sell_rank=75
    )


def test_winner_requires_every_acceptance_condition():
    results = pd.DataFrame([
        {
            "Strategy": "A_Fixed_Top30",
            "Sharpe": 0.40,
            "MaxDD": -0.30,
            "Turnover": 0.46,
            "ROE": 0.60,
            "ProfitGrowth": 0.10,
        },
        {
            "Strategy": "B_Top30_Buffer",
            "Sharpe": 0.42,
            "MaxDD": -0.31,
            "Turnover": 0.29,
            "ROE": 0.50,
            "ProfitGrowth": 0.08,
        },
    ])

    assert select_winner(results) == "B_Top30_Buffer"


def test_winner_falls_back_to_baseline_when_no_candidate_passes():
    results = pd.DataFrame([
        {
            "Strategy": "A_Fixed_Top30",
            "Sharpe": 0.40,
            "MaxDD": -0.30,
            "Turnover": 0.46,
            "ROE": 0.60,
            "ProfitGrowth": 0.10,
        },
        {
            "Strategy": "B_Top30_Buffer",
            "Sharpe": 0.42,
            "MaxDD": -0.31,
            "Turnover": 0.36,
            "ROE": 0.50,
            "ProfitGrowth": 0.08,
        },
    ])

    assert select_winner(results) == "A_Fixed_Top30"
