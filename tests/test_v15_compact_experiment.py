import pandas as pd

from run_v15_dual_branch import FEATURES_FUNDA
from run_v15_compact_experiment import (
    COMPACT_F_CONFIG,
    COMPACT_FT_CONFIG,
    COMPACT_FT3_CONFIG,
    build_arg_parser,
    mixed_model_passes,
    yearly_top30_sharpe,
)


MONOTONE = {
    "EP_neutral_z": 1,
    "SR_ROE_neutral_z": 1,
    "SR_ProfitGrowth_YoY_neutral_z": 1,
}


def test_compact_feature_sets_are_exact():
    assert COMPACT_F_CONFIG.feature_neutral_z == FEATURES_FUNDA
    assert COMPACT_FT_CONFIG.feature_neutral_z == FEATURES_FUNDA + [
        "Mom_3M_neutral_z",
        "Vol_60D_neutral_z",
    ]
    assert COMPACT_FT3_CONFIG.feature_neutral_z == FEATURES_FUNDA + [
        "Mom_3M_neutral_z",
        "Vol_60D_neutral_z",
        "Mom_6M_neutral_z",
    ]


def test_compact_configs_share_training_contract():
    configs = [COMPACT_F_CONFIG, COMPACT_FT_CONFIG, COMPACT_FT3_CONFIG]
    for config in configs:
        assert config.gs_enabled is False
        assert config.colsample_bytree == 0.75
        assert config.learning_rate == 0.05
        assert config.reg_alpha == 0.10
        assert config.lambda_turnover == 2.0
        assert config.seeds == [42]
        assert (config.train_months, config.val_months, config.test_months) == (
            36,
            6,
            1,
        )
        assert config.monotone_constraints == MONOTONE


def test_cli_defaults_are_stable():
    args = build_arg_parser().parse_args([])
    assert str(args.output_dir).replace("\\", "/").endswith(
        "output/production_models_v15_compact"
    )
    assert args.skip_training is False
    assert args.dry_run is False


def test_mixed_model_passes_only_when_all_rules_hold():
    baseline = {
        "Sharpe": 0.40,
        "MaxDD": -0.30,
        "Turnover": 0.45,
    }
    candidate = {
        "Sharpe": 0.45,
        "MaxDD": -0.34,
        "Turnover": 0.53,
        "ROE": 0.20,
        "ProfitGrowth": 0.10,
    }
    yearly = pd.DataFrame({
        "year": [2021, 2022, 2023],
        "baseline_sharpe": [0.1, -0.2, 0.3],
        "candidate_sharpe": [0.2, -0.1, 0.2],
    })

    assert mixed_model_passes(candidate, baseline, yearly) is True


def test_mixed_model_fails_when_improvement_is_only_one_year():
    baseline = {
        "Sharpe": 0.40,
        "MaxDD": -0.30,
        "Turnover": 0.45,
    }
    candidate = {
        "Sharpe": 0.45,
        "MaxDD": -0.34,
        "Turnover": 0.53,
        "ROE": 0.20,
        "ProfitGrowth": 0.10,
    }
    yearly = pd.DataFrame({
        "year": [2021, 2022, 2023],
        "baseline_sharpe": [0.1, -0.2, 0.3],
        "candidate_sharpe": [0.2, -0.3, 0.2],
    })

    assert mixed_model_passes(candidate, baseline, yearly) is False


def test_yearly_top30_sharpe_is_compatible_with_current_pandas():
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    predictions = pd.DataFrame([
        {"date": date, "symbol": f"{i:06d}", "alpha_signal": float(i)}
        for date in dates
        for i in range(40)
    ])
    panel = pd.DataFrame([
        {
            "date": date,
            "symbol": f"{i:06d}",
            "forward_return_1m": i / 1000,
        }
        for date in dates
        for i in range(40)
    ])

    result = yearly_top30_sharpe(predictions, panel)

    assert result["year"].tolist() == [2024]
    assert result["n_months"].tolist() == [2]
