"""
V1 vs V2 逐月收益归因分析
========================
目标: 拆解 Sharpe 0.70 -> 0.51 的业绩分化来源
- 哪些月份贡献了差距?
- V2 在什么市场环境下表现更差?
- Long-short 收益差来自 IC 下降还是 spread 变化?
"""
import warnings, logging
from pathlib import Path
import numpy as np, pandas as pd
import scipy.stats as stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("attribution")

OUT = Path("output")
PANEL_PATH = OUT / "training_panel_v3_full.parquet"
V1_PATH = OUT / "predictions_v1.parquet"
V2_PATH = OUT / "predictions_v2_full.parquet"
REPORT_PATH = OUT / "v1_v2_monthly_attribution.md"

# Load
logger.info("Loading data...")
panel = pd.read_parquet(PANEL_PATH); panel["date"] = pd.to_datetime(panel["date"])
v1 = pd.read_parquet(V1_PATH); v1["date"] = pd.to_datetime(v1["date"])
v2 = pd.read_parquet(V2_PATH); v2["date"] = pd.to_datetime(v2["date"])

# Ensure forward returns
if "forward_return_1m" not in panel.columns:
    close_col = "收盘" if "收盘" in panel.columns else "close"
    panel = panel.sort_values(["symbol", "date"])
    panel["forward_return_1m"] = panel.groupby("symbol")[close_col].transform(
        lambda x: x.shift(-1) / x - 1.0)
logger.info(f"Panel={len(panel)}, V1={len(v1)}, V2={len(v2)}")

# Merge predictions
dates = sorted(set(v1["date"]) & set(v2["date"]))
logger.info(f"Common dates: {len(dates)}")

# ======================================================================
# 1. Monthly IC & Long-Short Return Decomposition
# ======================================================================
logger.info("Computing monthly IC and long-short returns...")

monthly = []
for dt in dates:
    p = panel[panel["date"] == dt].copy()
    p1 = v1[v1["date"] == dt][["symbol", "prediction"]].rename(columns={"prediction": "pred_v1"})
    p2 = v2[v2["date"] == dt][["symbol", "prediction"]].rename(columns={"prediction": "pred_v2"})

    m = p.merge(p1, on="symbol").merge(p2, on="symbol")
    m = m.dropna(subset=["forward_return_1m", "pred_v1", "pred_v2"])
    if len(m) < 50:
        continue

    # IC
    ic_v1, _ = stats.spearmanr(m["pred_v1"], m["forward_return_1m"])
    ic_v2, _ = stats.spearmanr(m["pred_v2"], m["forward_return_1m"])

    # Long-short: top 30% vs bottom 30% (matching backtest top_quantile=0.3)
    cutoff = int(len(m) * 0.3)
    m_sorted_v1 = m.sort_values("pred_v1", ascending=False)
    m_sorted_v2 = m.sort_values("pred_v2", ascending=False)

    long_v1 = m_sorted_v1.head(cutoff)["forward_return_1m"].mean()
    short_v1 = m_sorted_v1.tail(cutoff)["forward_return_1m"].mean()
    ls_v1 = long_v1 - short_v1

    long_v2 = m_sorted_v2.head(cutoff)["forward_return_1m"].mean()
    short_v2 = m_sorted_v2.tail(cutoff)["forward_return_1m"].mean()
    ls_v2 = long_v2 - short_v2

    # Factor exposures of long portfolio (for style analysis)
    factor_cols = [c for c in panel.columns if c.endswith("_neutral_z") and not c.endswith("_rank")]
    exposures_v1 = {}
    exposures_v2 = {}
    for fc in factor_cols:
        if fc in m.columns:
            long_idx_v1 = m_sorted_v1.head(cutoff).index
            long_idx_v2 = m_sorted_v2.head(cutoff).index
            exposures_v1[fc] = m.loc[long_idx_v1, fc].mean()
            exposures_v2[fc] = m.loc[long_idx_v2, fc].mean()

    # Market regime indicators
    mkt_return = m["forward_return_1m"].mean()
    mkt_std = m["forward_return_1m"].std()
    n_stocks = len(m)

    monthly.append({
        "date": dt,
        "ic_v1": ic_v1, "ic_v2": ic_v2, "ic_delta": ic_v2 - ic_v1,
        "long_v1": long_v1, "short_v1": short_v1, "ls_v1": ls_v1,
        "long_v2": long_v2, "short_v2": short_v2, "ls_v2": ls_v2,
        "ls_delta": ls_v2 - ls_v1,
        "mkt_return": mkt_return, "mkt_std": mkt_std, "n_stocks": n_stocks,
        **{f"exp_v1_{fc}": exposures_v1.get(fc, np.nan) for fc in factor_cols},
        **{f"exp_v2_{fc}": exposures_v2.get(fc, np.nan) for fc in factor_cols},
    })

mdf = pd.DataFrame(monthly)
mdf["year"] = mdf["date"].dt.year
mdf["month"] = mdf["date"].dt.month
mdf["cum_ls_v1"] = mdf["ls_v1"].cumsum()
mdf["cum_ls_v2"] = mdf["ls_v2"].cumsum()

# ======================================================================
# 2. Period Analysis: Identify when V2 underperforms V1
# ======================================================================
logger.info("Period analysis...")

# Split by market regime
mdf["mkt_regime"] = pd.cut(mdf["mkt_return"], bins=[-99, -0.03, 0.03, 99], labels=["Down", "Flat", "Up"])
mdf["vol_regime"] = pd.cut(mdf["mkt_std"], bins=[0, 0.04, 0.08, 99], labels=["LowVol", "MidVol", "HighVol"])

# Yearly summary
yearly = mdf.groupby("year").agg(
    mean_ic_v1=("ic_v1", "mean"), mean_ic_v2=("ic_v2", "mean"),
    mean_ls_v1=("ls_v1", "mean"), mean_ls_v2=("ls_v2", "mean"),
    sharpe_v1=("ls_v1", lambda x: np.mean(x)/np.std(x)*np.sqrt(12) if np.std(x)>0 else 0),
    sharpe_v2=("ls_v2", lambda x: np.mean(x)/np.std(x)*np.sqrt(12) if np.std(x)>0 else 0),
    n=("ls_v1", "count"),
).reset_index()

# Regime summary
regime_summary = mdf.groupby("mkt_regime", observed=False).agg(
    mean_ls_v1=("ls_v1", "mean"), mean_ls_v2=("ls_v2", "mean"),
    ls_delta=("ls_delta", "mean"), n=("ls_v1", "count"),
).reset_index()

vol_regime_summary = mdf.groupby("vol_regime", observed=False).agg(
    mean_ls_v1=("ls_v1", "mean"), mean_ls_v2=("ls_v2", "mean"),
    ls_delta=("ls_delta", "mean"), n=("ls_v1", "count"),
).reset_index()

# ======================================================================
# 3. Style Attribution: Factor exposure differences
# ======================================================================
logger.info("Style attribution...")
factor_cols = [c for c in panel.columns if c.endswith("_neutral_z") and not c.endswith("_rank")]
exp_cols_v1 = [f"exp_v1_{fc}" for fc in factor_cols if f"exp_v1_{fc}" in mdf.columns]
exp_cols_v2 = [f"exp_v2_{fc}" for fc in factor_cols if f"exp_v2_{fc}" in mdf.columns]

style_diff = {}
for fc in factor_cols:
    col_v1 = f"exp_v1_{fc}"
    col_v2 = f"exp_v2_{fc}"
    if col_v1 in mdf.columns and col_v2 in mdf.columns:
        diff = mdf[col_v2] - mdf[col_v1]
        style_diff[fc.replace("_neutral_z", "")] = {
            "v1_mean": mdf[col_v1].mean(), "v2_mean": mdf[col_v2].mean(),
            "diff_mean": diff.mean(), "diff_std": diff.std(),
        }

# ======================================================================
# 4. Worst months for V2 relative to V1
# ======================================================================
logger.info("Identifying worst months...")
worst_months = mdf.nsmallest(12, "ls_delta")[
    ["date", "ls_v1", "ls_v2", "ls_delta", "ic_v1", "ic_v2", "mkt_return", "mkt_std"]
].copy()
best_months = mdf.nlargest(12, "ls_delta")[
    ["date", "ls_v1", "ls_v2", "ls_delta", "ic_v1", "ic_v2", "mkt_return", "mkt_std"]
].copy()

# ======================================================================
# 5. Generate Report
# ======================================================================
logger.info("Generating report...")
R = []
def w(s=""): R.append(s)

w("# V1 vs V2 逐月收益归因分析")
w()
w(f"**Generated**: {pd.Timestamp.now()}")
w()

# Summary stats
v1_sharpe = mdf["ls_v1"].mean() / mdf["ls_v1"].std() * np.sqrt(12) if mdf["ls_v1"].std() > 0 else 0
v2_sharpe = mdf["ls_v2"].mean() / mdf["ls_v2"].std() * np.sqrt(12) if mdf["ls_v2"].std() > 0 else 0
w("## Summary")
w()
w("| Metric | V1 | V2 | Delta |")
w("|--------|----|----|-------|")
w(f"| Mean IC | {mdf['ic_v1'].mean():.4f} | {mdf['ic_v2'].mean():.4f} | {mdf['ic_v2'].mean()-mdf['ic_v1'].mean():+.4f} |")
w(f"| IC_IR | {mdf['ic_v1'].mean()/mdf['ic_v1'].std():.3f} | {mdf['ic_v2'].mean()/mdf['ic_v2'].std():.3f} | |")
w(f"| Mean L/S Return (monthly) | {mdf['ls_v1'].mean()*100:.2f}% | {mdf['ls_v2'].mean()*100:.2f}% | {mdf['ls_delta'].mean()*100:+.2f}% |")
w(f"| L/S Std | {mdf['ls_v1'].std()*100:.2f}% | {mdf['ls_v2'].std()*100:.2f}% | |")
w(f"| L/S Sharpe (ann) | {v1_sharpe:.2f} | {v2_sharpe:.2f} | {v2_sharpe-v1_sharpe:+.2f} |")
w(f"| Hit Rate (L/S > 0) | {(mdf['ls_v1']>0).mean()*100:.0f}% | {(mdf['ls_v2']>0).mean()*100:.0f}% | |")
w(f"| N months | {len(mdf)} | {len(mdf)} | |")
w()

# Sharpe decomposition
w("### Sharpe Gap Decomposition")
w()
ls_mean_delta = mdf["ls_delta"].mean()
ls_std_v1 = mdf["ls_v1"].std()
ls_std_v2 = mdf["ls_v2"].std()
w(f"- V1 Sharpe: mean={mdf['ls_v1'].mean()*100:.2f}%, std={ls_std_v1*100:.2f}% -> SR={v1_sharpe:.2f}")
w(f"- V2 Sharpe: mean={mdf['ls_v2'].mean()*100:.2f}%, std={ls_std_v2*100:.2f}% -> SR={v2_sharpe:.2f}")
w(f"- Mean return delta: {ls_mean_delta*100:+.2f}% per month")
w(f"- Vol change: {ls_std_v2/ls_std_v1:.2f}x {'wider' if ls_std_v2 > ls_std_v1 else 'tighter'}")
w()

# Yearly breakdown
w("## Yearly Breakdown")
w()
w("| Year | IC V1 | IC V2 | L/S V1 | L/S V2 | SR V1 | SR V2 | N |")
w("|------|-------|-------|--------|--------|-------|-------|---|")
for _, yr in yearly.iterrows():
    w(f"| {int(yr['year'])} | {yr['mean_ic_v1']:.4f} | {yr['mean_ic_v2']:.4f} | "
      f"{yr['mean_ls_v1']*100:.2f}% | {yr['mean_ls_v2']*100:.2f}% | "
      f"{yr['sharpe_v1']:.2f} | {yr['sharpe_v2']:.2f} | {int(yr['n'])} |")
w()

# Market regime analysis
w("## Market Regime Analysis")
w()
w("### By Market Direction")
w()
w("| Regime | L/S V1 | L/S V2 | Delta | N |")
w("|--------|--------|--------|-------|---|")
for _, r in regime_summary.iterrows():
    w(f"| {r['mkt_regime']} | {r['mean_ls_v1']*100:.2f}% | {r['mean_ls_v2']*100:.2f}% | {r['ls_delta']*100:+.2f}% | {int(r['n'])} |")
w()

w("### By Volatility Regime")
w()
w("| Regime | L/S V1 | L/S V2 | Delta | N |")
w("|--------|--------|--------|-------|---|")
for _, r in vol_regime_summary.iterrows():
    w(f"| {r['vol_regime']} | {r['mean_ls_v1']*100:.2f}% | {r['mean_ls_v2']*100:.2f}% | {r['ls_delta']*100:+.2f}% | {int(r['n'])} |")
w()

# Worst months
w("## Worst 12 Months for V2 vs V1")
w()
w("| Date | L/S V1 | L/S V2 | Delta | IC V1 | IC V2 | Mkt Ret | Mkt Std |")
w("|------|--------|--------|-------|-------|-------|---------|---------|")
for _, r in worst_months.iterrows():
    w(f"| {str(r['date'])[:10]} | {r['ls_v1']*100:.2f}% | {r['ls_v2']*100:.2f}% | {r['ls_delta']*100:+.2f}% | "
      f"{r['ic_v1']:.3f} | {r['ic_v2']:.3f} | {r['mkt_return']*100:.2f}% | {r['mkt_std']*100:.2f}% |")
w()

# Best months
w("## Best 12 Months for V2 vs V1")
w()
w("| Date | L/S V1 | L/S V2 | Delta | IC V1 | IC V2 | Mkt Ret | Mkt Std |")
w("|------|--------|--------|-------|-------|-------|---------|---------|")
for _, r in best_months.iterrows():
    w(f"| {str(r['date'])[:10]} | {r['ls_v1']*100:.2f}% | {r['ls_v2']*100:.2f}% | {r['ls_delta']*100:+.2f}% | "
      f"{r['ic_v1']:.3f} | {r['ic_v2']:.3f} | {r['mkt_return']*100:.2f}% | {r['mkt_std']*100:.2f}% |")
w()

# Style drift
w("## Factor Exposure Differences (V2 Long - V1 Long)")
w()
# Factor name mapping
factor_short_names = {
    "Mom_1M_neutral_z": "Mom_1M", "Mom_3M_neutral_z": "Mom_3M", "Mom_6M_neutral_z": "Mom_6M",
    "Mom_12M_1M_neutral_z": "Mom_12_1M", "Vol_20D_neutral_z": "Vol_20D", "Vol_60D_neutral_z": "Vol_60D",
    "Beta_neutral_z": "Beta", "BP_neutral_z": "BP", "EP_neutral_z": "EP", "ROE_neutral_z": "ROE",
    "Debt_Ratio_neutral_z": "DebtRatio", "Net_Profit_Margin_neutral_z": "NetMargin",
    "RevGrowth_YoY_neutral_z": "RevGrowth", "ProfitGrowth_YoY_neutral_z": "ProfitGrowth",
    "VolChg_20D_neutral_z": "VolChg", "PriceDev_20D_neutral_z": "PriceDev",
}
w("| Factor | V1 Mean Exp | V2 Mean Exp | Delta | Std Delta |")
w("|--------|------------|------------|-------|-----------|")
for fc_full, info in sorted(style_diff.items(), key=lambda x: abs(x[1]["diff_mean"]), reverse=True):
    short_name = factor_short_names.get(fc_full + "_neutral_z", fc_full)
    w(f"| {short_name} | {info['v1_mean']:+.4f} | {info['v2_mean']:+.4f} | {info['diff_mean']:+.4f} | {info['diff_std']:.4f} |")
w()

# Monthly IC correlation
w("## IC Consistency")
w()
ic_corr = mdf[["ic_v1", "ic_v2"]].corr().iloc[0, 1]
w(f"Correlation between V1 IC and V2 IC: **{ic_corr:.3f}**")
w()
# Months where V1 and V2 disagree on IC direction
disagree = mdf[(mdf["ic_v1"] > 0) != (mdf["ic_v2"] > 0)]
w(f"Months where IC sign disagrees: **{len(disagree)}/{len(mdf)} ({len(disagree)/len(mdf)*100:.0f}%)**")
if len(disagree) > 0:
    w()
    w("| Date | IC V1 | IC V2 | Mkt Ret |")
    w("|------|-------|-------|---------|")
    for _, r in disagree.iterrows():
        w(f"| {str(r['date'])[:10]} | {r['ic_v1']:+.4f} | {r['ic_v2']:+.4f} | {r['mkt_return']*100:+.2f}% |")
w()

# Cumulative L/S
w("## Cumulative Long-Short Return")
w()
w(f"- V1 Final Cumulative L/S: **{mdf['cum_ls_v1'].iloc[-1]*100:.1f}%**")
w(f"- V2 Final Cumulative L/S: **{mdf['cum_ls_v2'].iloc[-1]*100:.1f}%**")
w(f"- Delta: **{mdf['cum_ls_v2'].iloc[-1]*100 - mdf['cum_ls_v1'].iloc[-1]*100:+.1f}%**")
w()

# Key insight: when does V1 outperform V2 the most?
w("## Key Findings")
w()
# 1. IC contribution
w("### 1. IC Gap Analysis")
w(f"- V1 mean IC: {mdf['ic_v1'].mean():.4f}")
w(f"- V2 mean IC: {mdf['ic_v2'].mean():.4f}")
w(f"- The IC gap IS in V2's favor ({mdf['ic_delta'].mean():+.4f}), which means higher IC does NOT translate to higher L/S return")
w()

# 2. Hit rate by regime
for regime in ["Down", "Flat", "Up"]:
    sub = mdf[mdf["mkt_regime"] == regime]
    if len(sub) > 0:
        v1_hit = (sub["ls_v1"] > 0).mean()
        v2_hit = (sub["ls_v2"] > 0).mean()
        w(f"- **{regime} markets**: V1 hit rate={v1_hit:.0%}, V2 hit rate={v2_hit:.0%} (Delta={v2_hit-v1_hit:+.0%})")
w()

# 3. RankCorr between V1 and V2 predictions over time
w("### 3. Cross-Model Prediction Correlation Over Time")
pred_corrs = []
for dt in dates:
    m = pd.merge(
        v1[v1["date"]==dt][["symbol","prediction"]].rename(columns={"prediction":"p1"}),
        v2[v2["date"]==dt][["symbol","prediction"]].rename(columns={"prediction":"p2"}),
        on="symbol"
    )
    if len(m) >= 30:
        c = stats.spearmanr(m["p1"], m["p2"])[0]
        pred_corrs.append({"date": dt, "spearman_r": c})
pc_df = pd.DataFrame(pred_corrs)
w(f"- Mean Spearman r between V1 and V2 rankings: **{pc_df['spearman_r'].mean():.4f}**")
w(f"- Min r: {pc_df['spearman_r'].min():.4f}, Max r: {pc_df['spearman_r'].max():.4f}")
w(f"- Months where r < 0.5: {(pc_df['spearman_r'] < 0.5).sum()}/{(~pc_df['spearman_r'].isna()).sum()}")
w()

w("---")
w(f"*Report generated: {pd.Timestamp.now()}*")

OUTPUT = "\n".join(R)
REPORT_PATH.write_text(OUTPUT, encoding="utf-8")
logger.info(f"Report saved: {REPORT_PATH}")
print(OUTPUT[:6000])
