from run_v15_experiment import M5_V15_FULL


def test_m5_extended_factor_configuration():
    assert len(M5_V15_FULL.feature_neutral_z) == 33
    assert M5_V15_FULL.colsample_bytree == 0.35
    assert M5_V15_FULL.reg_alpha == 1.0
    assert "Operating_Margin_neutral_z" in M5_V15_FULL.feature_neutral_z
    assert "Equity_Multiplier_neutral_z" in M5_V15_FULL.feature_neutral_z
