import numpy as np
import pandas as pd
import pytest

from run_v15_dual_branch import (
    FEATURES_FUNDA,
    FEATURES_TECH,
    MODEL_F_CONFIG,
    MODEL_T_CONFIG,
    apply_gap_aware_ema,
    blend_oos_predictions,
    build_arg_parser,
    compute_top30_exposures,
    evaluate_top30,
)


EXPECTED_FUNDA = [
    "EP_neutral_z",
    "BP_raw_neutral_z",
    "SR_ROE_neutral_z",
    "Net_Profit_Margin_neutral_z",
    "Operating_Margin_neutral_z",
    "CFO_to_Earnings_neutral_z",
    "EPS_YoY_neutral_z",
    "SR_ProfitGrowth_YoY_neutral_z",
    "SR_RevGrowth_YoY_neutral_z",
    "ProfitGrowth_YoY_neutral_z",
    "RevGrowth_YoY_neutral_z",
    "Debt_Ratio_neutral_z",
    "Current_Ratio_neutral_z",
    "Quick_Ratio_neutral_z",
    "Equity_Multiplier_neutral_z",
]

EXPECTED_TECH = [
    "Mom_1M_neutral_z",
    "Mom_3M_neutral_z",
    "Mom_6M_neutral_z",
    "Mom_12M_1M_neutral_z",
    "RSI_14_neutral_z",
    "Vol_20D_neutral_z",
    "Vol_60D_neutral_z",
    "Vol_120D_neutral_z",
    "Beta_neutral_z",
    "Skewness_60D_neutral_z",
    "MaxDD_60D_neutral_z",
    "High_Low_Range_20D_neutral_z",
    "Amihud_Illiquidity_neutral_z",
    "Dollar_Volume_20D_neutral_z",
    "Turnover_Volatility_20D_neutral_z",
    "PriceDev_20D_neutral_z",
    "VolChg_20D_neutral_z",
]


def test_feature_sets_match_approved_lists():
    assert FEATURES_FUNDA == EXPECTED_FUNDA
    assert FEATURES_TECH == EXPECTED_TECH
    assert set(FEATURES_FUNDA).isdisjoint(FEATURES_TECH)


def test_only_fundamental_branch_has_monotonicity():
    assert MODEL_F_CONFIG.monotone_constraints == {
        "EP_neutral_z": 1,
        "SR_ROE_neutral_z": 1,
        "SR_ProfitGrowth_YoY_neutral_z": 1,
    }
    assert MODEL_T_CONFIG.monotone_constraints == {}


def _branch_predictions(values, date="2024-01-31"):
    return pd.DataFrame({
        "date": pd.to_datetime([date] * len(values)),
        "symbol": [f"{i:06d}" for i in range(len(values))],
        "alpha_signal": values,
    })


def test_blend_ranks_each_branch_then_weights():
    pred_f = _branch_predictions(np.arange(30, dtype=float))
    pred_t = _branch_predictions(np.arange(29, -1, -1, dtype=float))

    result = blend_oos_predictions(pred_f, pred_t)

    assert result["rank_f"].iloc[-1] == pytest.approx(1.0)
    assert result["rank_t"].iloc[-1] == pytest.approx(1 / 30)
    assert result["raw_blend_pred"].iloc[-1] == pytest.approx(
        0.5 * 1.0 + 0.5 * (1 / 30)
    )


def test_blend_rejects_invalid_weights():
    pred_f = _branch_predictions(np.arange(30, dtype=float))
    pred_t = _branch_predictions(np.arange(30, dtype=float))

    with pytest.raises(ValueError, match="sum to 1"):
        blend_oos_predictions(pred_f, pred_t, weight_f=0.6, weight_t=0.5)


def test_blend_rejects_date_mismatch():
    pred_f = _branch_predictions(np.arange(30, dtype=float), "2024-01-31")
    pred_t = _branch_predictions(np.arange(30, dtype=float), "2024-02-29")

    with pytest.raises(ValueError, match="date sets"):
        blend_oos_predictions(pred_f, pred_t)


def test_ema_uses_previous_signal_only_for_consecutive_global_date():
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-31", "2024-02-29", "2024-02-29", "2024-03-31",
        ]),
        "symbol": ["000001", "000001", "000002", "000001"],
        "raw_blend_pred": [0.2, 0.8, 0.6, 0.5],
    })

    result = apply_gap_aware_ema(df, alpha=0.6)

    stock_a = result[result["symbol"] == "000001"].sort_values("date")
    assert stock_a["final_pred"].tolist() == pytest.approx([
        0.2,
        0.6 * 0.8 + 0.4 * 0.2,
        0.6 * 0.5 + 0.4 * (0.6 * 0.8 + 0.4 * 0.2),
    ])


def test_ema_resets_after_missing_global_rebalance_date():
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-31", "2024-02-29", "2024-03-31",
        ]),
        "symbol": ["000001", "000002", "000001"],
        "raw_blend_pred": [0.2, 0.6, 0.9],
    })

    result = apply_gap_aware_ema(df, alpha=0.6)
    march = result[
        (result["date"] == pd.Timestamp("2024-03-31"))
        & (result["symbol"] == "000001")
    ].iloc[0]

    assert march["final_pred"] == pytest.approx(0.9)


def test_ema_reuses_previous_value_for_consecutive_nan():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
        "symbol": ["000001", "000001"],
        "raw_blend_pred": [0.4, np.nan],
    })

    result = apply_gap_aware_ema(df, alpha=0.6)

    assert result["final_pred"].tolist() == pytest.approx([0.4, 0.4])


def test_ema_does_not_reuse_stale_value_for_nan_after_gap():
    df = pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-31", "2024-02-29", "2024-03-31",
        ]),
        "symbol": ["000001", "000002", "000001"],
        "raw_blend_pred": [0.4, 0.8, np.nan],
    })

    result = apply_gap_aware_ema(df, alpha=0.6)
    march = result[
        (result["date"] == pd.Timestamp("2024-03-31"))
        & (result["symbol"] == "000001")
    ].iloc[0]

    assert np.isnan(march["final_pred"])


def _portfolio_fixture():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    symbols = [f"{i:06d}" for i in range(40)]
    prediction_rows = []
    panel_rows = []
    for date_idx, date in enumerate(dates):
        for i, symbol in enumerate(symbols):
            signal = float(i if date_idx == 0 else (i + 10) % 40)
            prediction_rows.append({
                "date": date,
                "symbol": symbol,
                "alpha_signal": signal,
            })
            panel_rows.append({
                "date": date,
                "symbol": symbol,
                "forward_return_1m": i / 1000,
                "SR_ROE_neutral_z": i / 10,
                "SR_ProfitGrowth_YoY_neutral_z": i / 20,
                "EP_neutral_z": i / 30,
                "BP_raw_neutral_z": i / 40,
            })
    return pd.DataFrame(prediction_rows), pd.DataFrame(panel_rows)


def test_evaluate_top30_uses_exactly_30_names_and_fixed_turnover():
    predictions, panel = _portfolio_fixture()

    metrics = evaluate_top30(predictions, panel)

    assert metrics["n_positions"] == 30
    # First top set is symbols 10..39. Second is 0..29, so 10/30 exited.
    assert metrics["turnover"] == pytest.approx(10 / 30)
    assert metrics["n_months"] == 2


def test_evaluate_top30_counts_only_months_with_realized_returns():
    predictions, panel = _portfolio_fixture()
    panel.loc[panel["date"] == pd.Timestamp("2024-02-29"), "forward_return_1m"] = np.nan

    metrics = evaluate_top30(predictions, panel)

    assert metrics["n_months"] == 1


def test_compute_top30_exposures_uses_selected_name_means():
    predictions, panel = _portfolio_fixture()

    exposures = compute_top30_exposures(predictions, panel)

    # First month selects 10..39, second selects 0..29; average selected i = 19.5.
    assert exposures["ROE"] == pytest.approx(1.95)
    assert exposures["ProfitGrowth"] == pytest.approx(0.975)


def test_cli_defaults_match_approved_design():
    args = build_arg_parser().parse_args([])

    assert args.weight_f == pytest.approx(0.5)
    assert args.weight_t == pytest.approx(0.5)
    assert args.ema_alpha == pytest.approx(0.6)
    assert str(args.output_dir).replace("\\", "/").endswith(
        "output/production_models_v15_dual"
    )
