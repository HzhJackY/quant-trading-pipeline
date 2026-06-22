import numpy as np
import pandas as pd
import pytest

from run_compact_f_production_validation import (
    BUFFER_GRID,
    apply_transaction_cost,
    decide_production_candidate,
)


def test_transaction_cost_uses_one_way_turnover_and_zero_first_cost():
    monthly = pd.DataFrame({
        "portfolio_return": [0.01, 0.02],
        "turnover": [np.nan, 0.5],
    })

    net = apply_transaction_cost(monthly, cost_bps=20)

    assert net.tolist() == pytest.approx([0.01, 0.019])


def test_buffer_grid_is_exactly_the_requested_three_points():
    assert BUFFER_GRID == [(30, 70), (35, 75), (40, 80)]


def test_decision_recommends_default_only_when_cost_and_robustness_pass():
    assert decide_production_candidate(
        cost_advantage=True,
        passing_parameter_count=3,
        current_candidate_passes=True,
    ) == "A"


def test_decision_marks_research_candidate_when_only_current_point_passes():
    assert decide_production_candidate(
        cost_advantage=True,
        passing_parameter_count=1,
        current_candidate_passes=True,
    ) == "B"


def test_decision_falls_back_when_current_candidate_fails():
    assert decide_production_candidate(
        cost_advantage=False,
        passing_parameter_count=0,
        current_candidate_passes=False,
    ) == "C"
