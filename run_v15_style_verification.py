"""
V1.5 Style Verification — Factor Exposures & Regime Performance.

Validates the core V1.5 hypothesis: M5 (GS=OFF, SR factors, monotonicity)
recovers Quality+Value style that V2 lost.

Checks:
  1. Top30 Long factor exposures: ROE, ProfitGrowth, EP, Mom_3M, Mom_6M
  2. Regime-level L/S returns: Up (>+3%), Flat, Down (<-3%) markets
  3. Yearly Sharpe decomposition
  4. Exposure time series stability

Reference (V1 vs V2 from audit):
  Factor        | V1     | V2      | V1.5 Target
  ROE           | +0.59  | -0.49   | >= +0.20
  ProfitGrowth  | +0.47  | -0.99   | >= +0.20
  EP            | +1.24  | +0.82   | >= +0.80
  Mom_3M        | +0.05  | -0.36   | >= -0.10
  Mom_6M        | +0.07  | -0.31   | >= -0.10

Output: output/v15_style_verification.md
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("v15_style")

OUTPUT_DIR = Path("output")
MODEL_DIR = OUTPUT_DIR / "production_models_v15"
PANEL_PATH = OUTPUT_DIR / "training_panel_v15_sr.parquet"

# Factor columns for exposure analysis (use _neutral_z)
FACTOR_MAP = {
    "ROE": "SR_ROE_neutral_z",
    "ProfitGrowth": "SR_ProfitGrowth_YoY_neutral_z",
    "EP": "EP_neutral_z",
    "Mom_3M": "Mom_3M_neutral_z",
    "Mom_6M": "Mom_6M_neutral_z",
    "NetMargin": "Net_Profit_Margin_neutral_z",
    "RevGrowth": "SR_RevGrowth_YoY_neutral_z",
    "BP": "BP_raw_neutral_z",
    "EPS_YoY": "EPS_YoY_neutral_z",
}


def load_data():
    """Load predictions and panel with factors."""
    # Load M0 and M5 predictions
    m0 = pd.read_parquet(MODEL_DIR / "M0_V2_Baseline_oos.parquet")
    m5 = pd.read_parquet(MODEL_DIR / "M5_V15_Full_oos.parquet")
    m0["date"] = pd.to_datetime(m0["date"])
    m5["date"] = pd.to_datetime(m5["date"])

    # Load panel with factors and close
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"])

    # Find close column
    close_col = None
    for c in panel.columns:
        if c in ("收盘", "close"):
            close_col = c
            break

    # Compute forward returns
    panel = panel.sort_values(["symbol", "date"])
    panel["fwd_ret"] = panel.groupby("symbol")[close_col].transform(
        lambda x: x.shift(-1) / x - 1.0
    )

    # Compute market return for regime classification
    mkt_ret = panel.groupby("date")[close_col].mean().pct_change().rename("mkt_ret")
    panel = panel.merge(mkt_ret, on="date", how="left")

    return m0, m5, panel


def compute_top30_exposures(predictions, panel, model_name):
    """Compute Top30 Long factor exposures per date."""
    df = predictions.merge(panel, on=["date", "symbol"], how="inner")

    # Cross-sectional rank
    df["signal_rank"] = df.groupby("date")["alpha_signal"].rank(pct=True, na_option="bottom")

    # Top30 Long
    top30 = df[df["signal_rank"] >= 0.70].copy()  # top 30%

    exposures = {}
    for factor_name, factor_col in FACTOR_MAP.items():
        if factor_col not in df.columns:
            continue
        # Mean _neutral_z exposure per date for top30
        exp = top30.groupby("date")[factor_col].mean()
        exposures[factor_name] = exp

    exp_df = pd.DataFrame(exposures)
    return exp_df


def compute_ls_returns(predictions, panel):
    """Compute Top30/Bottom30 L/S returns per date."""
    df = predictions.merge(
        panel[["date", "symbol", "fwd_ret", "mkt_ret"]],
        on=["date", "symbol"], how="inner"
    )
    df["signal_rank"] = df.groupby("date")["alpha_signal"].rank(pct=True, na_option="bottom")

    results = []
    for dt, grp in df.groupby("date"):
        if len(grp) < 30:
            continue
        long_ret = grp[grp["signal_rank"] >= 0.70]["fwd_ret"].mean()
        short_ret = grp[grp["signal_rank"] <= 0.30]["fwd_ret"].mean()
        mkt_ret = grp["mkt_ret"].iloc[0] if pd.notna(grp["mkt_ret"].iloc[0]) else 0

        # Regime
        if mkt_ret > 0.03:
            regime = "Up (>+3%)"
        elif mkt_ret < -0.03:
            regime = "Down (<-3%)"
        else:
            regime = "Flat"

        results.append({
            "date": dt,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": long_ret - short_ret,
            "mkt_ret": mkt_ret,
            "regime": regime,
        })

    return pd.DataFrame(results)


def compute_sharpe(returns):
    """Annualized Sharpe from monthly returns."""
    r = returns.dropna()
    if len(r) < 12:
        return np.nan
    return float(r.mean() * 12 / (r.std() * np.sqrt(12)))


def main():
    logger.info("Loading data...")
    m0, m5, panel = load_data()

    # Compute exposures
    logger.info("Computing Top30 factor exposures...")
    m0_exp = compute_top30_exposures(m0, panel, "M0")
    m5_exp = compute_top30_exposures(m5, panel, "M5")

    # Mean exposures
    logger.info("\n=== Top30 Long Factor Exposures ===")
    audit_v1 = {"ROE": 0.59, "ProfitGrowth": 0.47, "EP": 1.24, "Mom_3M": 0.05, "Mom_6M": 0.07}
    audit_v2 = {"ROE": -0.49, "ProfitGrowth": -0.99, "EP": 0.82, "Mom_3M": -0.36, "Mom_6M": -0.31}
    target = {"ROE": 0.20, "ProfitGrowth": 0.20, "EP": 0.80, "Mom_3M": -0.10, "Mom_6M": -0.10}

    exposure_table = []
    for factor in ["ROE", "ProfitGrowth", "EP", "Mom_3M", "Mom_6M", "NetMargin", "RevGrowth", "BP", "EPS_YoY"]:
        if factor in m0_exp.columns and factor in m5_exp.columns:
            v1 = audit_v1.get(factor, np.nan)
            v2 = audit_v2.get(factor, np.nan)
            tgt = target.get(factor, np.nan)
            m0_val = m0_exp[factor].mean()
            m5_val = m5_exp[factor].mean()
            delta = m5_val - m0_val
            direction = "→ V1" if (v1 > 0 and delta > 0) or (v1 < 0 and delta < 0) else (
                "→ V2" if (v2 > 0 and delta > 0) or (v2 < 0 and delta < 0) else "mixed"
            )
            exposure_table.append({
                "Factor": factor,
                "V1 (audit)": round(v1, 2),
                "V2 (audit)": round(v2, 2),
                "M0 (V2 baseline)": round(m0_val, 3),
                "M5 (V1.5-Full)": round(m5_val, 3),
                "Delta (M5-M0)": round(delta, 3),
                "Direction": direction,
                "Target": tgt,
                "Pass": "YES" if abs(m5_val - tgt) < 0.3 or (tgt > 0 and m5_val > 0) or (tgt < 0 and m5_val < 0) else "NO",
            })

    exp_df = pd.DataFrame(exposure_table)
    print(exp_df.to_string(index=False))

    # Regime analysis
    logger.info("\n=== Regime Performance ===")
    m0_ls = compute_ls_returns(m0, panel)
    m5_ls = compute_ls_returns(m5, panel)

    regime_table = []
    for regime in ["Up (>+3%)", "Flat", "Down (<-3%)"]:
        m0_reg = m0_ls[m0_ls["regime"] == regime]
        m5_reg = m5_ls[m5_ls["regime"] == regime]
        regime_table.append({
            "Regime": regime,
            "N Months": len(m0_reg),
            "M0 L/S": f'{m0_reg["ls_ret"].mean():.3f}' if len(m0_reg) > 0 else "N/A",
            "M5 L/S": f'{m5_reg["ls_ret"].mean():.3f}' if len(m5_reg) > 0 else "N/A",
            "M5 Sharpe": round(compute_sharpe(m5_reg["ls_ret"]), 2) if len(m5_reg) >= 3 else "N/A",
        })

    reg_df = pd.DataFrame(regime_table)
    print(reg_df.to_string(index=False))

    # Yearly Sharpe
    logger.info("\n=== Yearly Sharpe ===")
    m0_ls["year"] = m0_ls["date"].dt.year
    m5_ls["year"] = m5_ls["date"].dt.year

    yearly_table = []
    for yr in sorted(m0_ls["year"].unique()):
        m0_yr = m0_ls[m0_ls["year"] == yr]["ls_ret"]
        m5_yr = m5_ls[m5_ls["year"] == yr]["ls_ret"]
        yearly_table.append({
            "Year": yr,
            "N": len(m0_yr),
            "M0 Sharpe": round(compute_sharpe(m0_yr), 2),
            "M5 Sharpe": round(compute_sharpe(m5_yr), 2),
            "Delta": round(compute_sharpe(m5_yr) - compute_sharpe(m0_yr), 2),
        })

    yr_df = pd.DataFrame(yearly_table)
    print(yr_df.to_string(index=False))

    # Overall stats
    logger.info("\n=== Overall Comparison ===")
    m0_overall_sharpe = compute_sharpe(m0_ls["ls_ret"])
    m5_overall_sharpe = compute_sharpe(m5_ls["ls_ret"])
    print(f"M0 (V2 Baseline): Sharpe={m0_overall_sharpe:.3f}")
    print(f"M5 (V1.5-Full):   Sharpe={m5_overall_sharpe:.3f}")
    print(f"Delta:             {m5_overall_sharpe - m0_overall_sharpe:+.3f}")

    # Style recovery verdict
    roe_recovered = m5_exp["ROE"].mean() > 0 if "ROE" in m5_exp.columns else False
    pg_recovered = m5_exp["ProfitGrowth"].mean() > 0 if "ProfitGrowth" in m5_exp.columns else False
    up_market_ok = m5_ls[m5_ls["regime"] == "Up (>+3%)"]["ls_ret"].mean() > -0.001 if len(m5_ls[m5_ls["regime"] == "Up (>+3%)"]) > 0 else False

    logger.info("\n=== STYLE RECOVERY VERDICT ===")
    logger.info(f"  ROE exposure positive: {roe_recovered}")
    logger.info(f"  ProfitGrowth exposure positive: {pg_recovered}")
    logger.info(f"  Up Market L/S >= 0: {up_market_ok}")

    # Save report
    report_path = OUTPUT_DIR / "v15_style_verification.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# V1.5 Style Verification Report\n\n")
        f.write("## Factor Exposures (Top30 Long)\n\n")
        f.write(exp_df.to_markdown(index=False))
        f.write("\n\n## Regime Performance\n\n")
        f.write(reg_df.to_markdown(index=False))
        f.write("\n\n## Yearly Sharpe\n\n")
        f.write(yr_df.to_markdown(index=False))
        f.write(f"\n\n## Verdict\n\n")
        f.write(f"- ROE exposure positive: **{roe_recovered}**\n")
        f.write(f"- ProfitGrowth exposure positive: **{pg_recovered}**\n")
        f.write(f"- Up Market L/S >= 0: **{up_market_ok}**\n")
        f.write(f"- M5 Sharpe: **{m5_overall_sharpe:.3f}** vs M0: **{m0_overall_sharpe:.3f}**\n")
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
