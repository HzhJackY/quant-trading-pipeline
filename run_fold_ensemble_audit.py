"""
Production Audit & Fold Ensemble Ablation (v2 - Global Fold Selection)
======================================================================
严格复现生产环境方法论: 使用GLOBAL fold selection (同一组fold用于所有日期).
Production uses fold 70 for ALL dates; schemes B-F follow the same convention.

Part 1: 生产推理链路审计
Part 3: Fold Ensemble Ablation (Schemes A-F, global folds)
Part 4: 统一对比表
Part 5: Root Cause Attribution

约束: 不重训, 不调参, 匹配生产方法论.
"""
import warnings, logging, sys, json, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
from collections import defaultdict
import scipy.stats as stats

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("prod_audit")

OUT = Path("output")
MODEL_DIR = OUT / "production_models_v2_full"
PANEL_PATH = OUT / "training_panel_v3_full.parquet"
REPORT_PATH = OUT / "fold_ensemble_production_audit.md"

# ======================================================================
# PART 1: PRODUCTION INFERENCE CHAIN AUDIT
# ======================================================================
logger.info("=" * 70)
logger.info("PART 1: Production Inference Chain Audit")
logger.info("=" * 70)

from factor_research.production_engine import ProductionAlphaEngine

engine = ProductionAlphaEngine.load_models(MODEL_DIR)
logger.info("load_models() -> %d models | %d features | %d folds | seeds=%s",
            engine.n_models, len(engine.feature_cols), engine._n_folds, engine.seeds)

any_seed = engine.seeds[0]
n_available = len(engine._models[any_seed])
last_fold = n_available - 1
logger.info(f"n_available folds per seed: {n_available}")
logger.info(f"fold_idx=-1 maps to: {n_available} + (-1) = {last_fold}")

last_fold_models = sum(1 for s in engine.seeds if engine._models[s][last_fold] is not None)
logger.info(f"Valid models at fold {last_fold}: {last_fold_models}")

AUDIT = {
    "model_dir": str(MODEL_DIR),
    "load_method": "ProductionAlphaEngine.load_models()",
    "detected_mode": "backtest",
    "seeds": engine.seeds,
    "n_folds_loaded": engine._n_folds,
    "n_models_loaded": engine.n_models,
    "predict_method": "predict_cross_section(fold_idx=-1, rank_output=True)",
    "fold_idx_resolves_to": last_fold,
    "models_per_inference": len(engine.seeds),
    "total_models_used": f"{len(engine.seeds)} seeds x 1 fold = {len(engine.seeds)} models",
    "inference_methodology": f"Fold {last_fold} is used for ALL dates (including 2017 data) - global fold selection",
}

logger.info("AUDIT RESULT:")
for k, v in AUDIT.items():
    logger.info(f"  {k}: {v}")

# ======================================================================
# PART 2: DATA PREPARATION
# ======================================================================
logger.info("=" * 70)
logger.info("PART 2: Data Preparation")
logger.info("=" * 70)

panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])

close_col = "收盘" if "收盘" in panel.columns else "close"
panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
panel["forward_return_1m"] = panel.groupby("symbol")[close_col].transform(
    lambda x: x.shift(-1) / x - 1.0)

nz_cols = [c for c in panel.columns if c.endswith("_neutral_z") and not c.endswith("_rank")]
rank_cols_used = []
for col in nz_cols:
    rc = f"{col}_rank"
    panel[rc] = panel.groupby("date")[col].rank(pct=True, na_option="bottom").fillna(0.5)
    rank_cols_used.append(rc)

dates = sorted(panel["date"].unique())
logger.info(f"Panel: {len(panel)} rows, {len(dates)} dates, {panel['symbol'].nunique()} symbols")

feature_cols = engine.feature_cols
col_map = {}
for fc in feature_cols:
    if fc in panel.columns:
        col_map[fc] = fc
    elif fc.replace("_rank", "") in panel.columns:
        col_map[fc.replace("_rank", "")] = fc
    elif f"{fc}_rank" in panel.columns:
        col_map[f"{fc}_rank"] = fc
logger.info(f"Feature mapping: {len(col_map)}/{len(feature_cols)}")

# Generate all-model raw predictions
logger.info("Generating raw predictions for all models on all dates...")
t0 = time.time()
raw_preds_all = {}
stocks_by_date = {}

for i, dt in enumerate(dates):
    mask = panel["date"] == dt
    df = panel[mask]
    syms = df["symbol"].values
    stocks_by_date[dt] = syms

    X_arr = np.zeros((len(df), len(feature_cols)), dtype=np.float64)
    for j, fc in enumerate(feature_cols):
        panel_col = col_map.get(fc)
        if panel_col and panel_col in df.columns:
            X_arr[:, j] = df[panel_col].values.astype(np.float64)

    date_preds = {}
    for seed in engine.seeds:
        for fold in range(engine._n_folds):
            model = engine._models[seed][fold]
            if model is not None:
                date_preds[(seed, fold)] = model.predict(X_arr)
    raw_preds_all[dt] = date_preds

    if (i + 1) % 20 == 0 or i == 0:
        logger.info("  [%3d/%3d] %s | %d stocks | %.1fs",
                     i + 1, len(dates), str(dt)[:10], len(syms), time.time() - t0)

logger.info(f"All predictions generated in {time.time() - t0:.1f}s")

def cs_rank(arr):
    s = pd.Series(arr)
    return s.rank(pct=True, na_option="bottom").fillna(0.5).values

# ======================================================================
# PART 3: FOLD ENSEMBLE ABLATION (Global Fold Selection)
# ======================================================================
logger.info("=" * 70)
logger.info("PART 3: Fold Ensemble Ablation (Global Fold Selection)")
logger.info("=" * 70)
logger.info("Methodology: Same folds for ALL dates (matching production convention)")
logger.info("Production uses fold 70 globally; all schemes follow the same pattern")

# Pre-compute per-fold global IC
logger.info("Computing per-fold IC...")
fold_global_ic = {}
for fold in range(engine._n_folds):
    fold_ics = []
    for dt in dates:
        seed_preds_list = []
        for seed in engine.seeds:
            key = (seed, fold)
            if key in raw_preds_all.get(dt, {}) and engine._models[seed][fold] is not None:
                seed_preds_list.append(raw_preds_all[dt][key])
        if not seed_preds_list:
            continue
        pred = np.mean(seed_preds_list, axis=0)
        syms = stocks_by_date[dt]
        df_dt = panel[panel["date"] == dt].set_index("symbol").reindex(syms)
        if "forward_return_1m" not in df_dt.columns:
            continue
        valid = df_dt["forward_return_1m"].notna()
        if valid.sum() < 30:
            continue
        ic, _ = stats.spearmanr(pred[valid.values], df_dt["forward_return_1m"].values[valid.values])
        if not np.isnan(ic):
            fold_ics.append(ic)
    fold_global_ic[fold] = np.mean(fold_ics) if fold_ics else 0.0

sorted_folds_ic = sorted(fold_global_ic.items(), key=lambda x: x[1], reverse=True)
n_top25 = max(1, engine._n_folds // 4)
n_top10 = max(1, engine._n_folds // 10)
top25_folds = sorted([f for f, _ in sorted_folds_ic[:n_top25]])
top10_folds = sorted([f for f, _ in sorted_folds_ic[:n_top10]])

logger.info(f"Fold IC range: [{sorted_folds_ic[-1][1]:.4f}, {sorted_folds_ic[0][1]:.4f}]")
logger.info(f"Top25% ({n_top25} folds): {top25_folds[:10]}...")
logger.info(f"Top10% ({n_top10} folds): {top10_folds[:10]}...")

# Scheme definitions (GLOBAL folds, same for all dates)
schemes = {
    "A_production": {
        "desc": "A: Production (fold=-1, fold 70 only)",
        "folds": [last_fold],
    },
    "B_last3": {
        "desc": "B: Last 3 folds (68-70)",
        "folds": list(range(last_fold - 2, last_fold + 1)),
    },
    "C_last10": {
        "desc": "C: Last 10 folds (61-70)",
        "folds": list(range(last_fold - 9, last_fold + 1)),
    },
    "D_all": {
        "desc": "D: All 71 folds",
        "folds": list(range(engine._n_folds)),
    },
    "E_top25pct": {
        "desc": f"E: IC Top 25% ({n_top25} folds)",
        "folds": top25_folds,
    },
    "F_top10pct": {
        "desc": f"F: IC Top 10% ({n_top10} folds)",
        "folds": top10_folds,
    },
}

# Generate predictions
logger.info("Generating predictions for all schemes...")
scheme_predictions = {}

for scheme_name, info in schemes.items():
    selected_folds = info["folds"]
    logger.info("  %s | %d folds", info["desc"], len(selected_folds))
    all_date_preds = []

    for i, dt in enumerate(dates):
        dp = raw_preds_all[dt]
        syms = stocks_by_date[dt]

        raw_list = []
        for seed in engine.seeds:
            for fold in selected_folds:
                key = (seed, fold)
                if key in dp:
                    raw_list.append(dp[key])

        if not raw_list:
            continue

        ensemble_raw = np.mean(raw_list, axis=0)
        ensemble_signal = cs_rank(ensemble_raw)

        pred_df = pd.DataFrame({
            "date": dt, "symbol": syms, "prediction": ensemble_signal,
        })
        all_date_preds.append(pred_df)

        if (i + 1) % 24 == 0 or i == 0:
            logger.info("    [%3d/%3d] %s | %d stocks | %d models",
                         i + 1, len(dates), str(dt)[:10], len(syms), len(raw_list))

    scheme_pred_df = pd.concat(all_date_preds, ignore_index=True)
    scheme_pred_df["date"] = pd.to_datetime(scheme_pred_df["date"])
    scheme_predictions[scheme_name] = scheme_pred_df
    logger.info("    -> %d predictions across %d dates",
                len(scheme_pred_df), scheme_pred_df["date"].nunique())

# ======================================================================
# PART 4: METRICS
# ======================================================================
logger.info("=" * 70)
logger.info("PART 4: Computing Metrics")
logger.info("=" * 70)

def compute_rank_stability(pred_df):
    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    dts = sorted(pred_df["date"].unique())
    corrs = []
    for i in range(1, len(dts)):
        prev = pred_df[pred_df["date"] == dts[i-1]].set_index("symbol")["prediction"]
        curr = pred_df[pred_df["date"] == dts[i]].set_index("symbol")["prediction"]
        common = prev.index.intersection(curr.index)
        if len(common) >= 30:
            c = stats.spearmanr(prev[common], curr[common])[0]
            if not np.isnan(c):
                corrs.append(c)
    return np.mean(corrs) if corrs else np.nan

def compute_overlap(pred_df, top_n=30):
    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    dts = sorted(pred_df["date"].unique())
    overlaps = []
    for i in range(1, len(dts)):
        prev_top = set(pred_df[pred_df["date"] == dts[i-1]].nlargest(top_n, "prediction")["symbol"])
        curr_top = set(pred_df[pred_df["date"] == dts[i]].nlargest(top_n, "prediction")["symbol"])
        if prev_top and curr_top:
            overlaps.append(len(prev_top & curr_top) / top_n)
    return np.mean(overlaps) if overlaps else np.nan

def compute_ic(pred_df):
    merged = pd.merge(
        pred_df, panel[["date", "symbol", "forward_return_1m"]], on=["date", "symbol"]
    ).dropna(subset=["prediction", "forward_return_1m"])
    ic_list = []
    for dt, group in merged.groupby("date"):
        if len(group) < 30:
            continue
        ic, _ = stats.spearmanr(group["prediction"], group["forward_return_1m"])
        if not np.isnan(ic):
            ic_list.append(ic)
    ic_series = pd.Series(ic_list)
    mean_ic = ic_series.mean()
    ic_ir = mean_ic / ic_series.std() if ic_series.std() > 0 else 0
    return mean_ic, ic_ir

from factor_research.backtest_engine import run_backtest_with_costs
from factor_research.transaction_cost import TieredCostModel, UniverseCostConfig

panel_bt = panel.copy()
if "成交额" not in panel_bt.columns:
    daily_path = OUT / "all_daily.parquet"
    if daily_path.exists():
        daily = pd.read_parquet(daily_path)
        daily["date"] = pd.to_datetime(daily["date"])
        panel_bt = panel_bt.merge(
            daily[["date", "symbol", "amount"]].rename(columns={"amount": "成交额"}),
            on=["date", "symbol"], how="left")
    panel_bt["成交额"] = panel_bt["成交额"].fillna(
        panel_bt["成交额"].median() if panel_bt["成交额"].notna().any() else 1e8)
if "总市值" not in panel_bt.columns:
    panel_bt["总市值"] = 100_000_000_000
if "universe" not in panel_bt.columns:
    panel_bt["universe"] = "大盘"
if "Vol_20D" not in panel_bt.columns:
    panel_bt["Vol_20D"] = 0.30

large_config = UniverseCostConfig(commission_bps=2.5, stamp_duty_bps=5.0,
                                   transfer_fee_bps=0.1, base_slippage_bps=5.0)
small_config = UniverseCostConfig(commission_bps=2.5, stamp_duty_bps=5.0,
                                   transfer_fee_bps=0.1, base_slippage_bps=15.0)
cost_model = TieredCostModel(aum=50_000_000, large_cap_config=large_config,
                              small_cap_config=small_config)

results = {}
scheme_order = ["A_production", "B_last3", "C_last10", "D_all", "E_top25pct", "F_top10pct"]

for scheme_name in scheme_order:
    pred_df = scheme_predictions.get(scheme_name)
    if pred_df is None:
        continue
    info = schemes[scheme_name]
    logger.info("Backtesting %s...", info["desc"])

    rc = compute_rank_stability(pred_df)
    ov = compute_overlap(pred_df)
    ic, ic_ir = compute_ic(pred_df)
    n_models = len(info["folds"]) * len(engine.seeds)

    try:
        bt_res = run_backtest_with_costs(
            panel_bt, pred_df, cost_model, top_quantile=0.3,
            min_stocks_per_universe=5, alpha_col="prediction")
        nm = bt_res["net_metrics"]
        sharpe = nm.get("Sharpe_Ratio", np.nan)
        maxdd = nm.get("Max_Drawdown", np.nan)
        turnover = bt_res["avg_turnover"]
    except Exception as e:
        logger.error("Backtest failed: %s", e)
        sharpe, maxdd, turnover = np.nan, np.nan, np.nan

    results[scheme_name] = {
        "desc": info["desc"], "n_models": n_models,
        "rank_corr": rc, "overlap": ov, "ic": ic, "ic_ir": ic_ir,
        "sharpe": sharpe, "maxdd": maxdd, "turnover": turnover,
    }
    logger.info("  RankCorr=%.4f Overlap=%.3f IC=%.4f Sharpe=%.2f MaxDD=%.3f TO=%.3f Models=%d",
                rc, ov, ic, sharpe, maxdd, turnover, n_models)

# ======================================================================
# PART 5: GENERATE REPORT
# ======================================================================
logger.info("Generating report...")

R = []
def w(s=""): R.append(s)

w("# Fold Ensemble Production Audit Report")
w()
w(f"**Generated**: {pd.Timestamp.now()}")
w()
w("---")
w("## Part 1: Production Inference Chain Audit")
w()
w("### Q1: What models does production actually load?")
w()
w("```text")
w(f"seed = {engine.seeds}")
w(f"fold = [{last_fold}]    <-- fold_idx=-1 resolves to the LAST fold")
w(f"")
w(f"Total models per inference: {len(engine.seeds)} seeds x 1 fold = {len(engine.seeds)} models")
w("```")
w()
w("### Q2: What does fold_idx=-1 mean?")
w()
w("**Answer: A - last fold only**")
w()
w("Code trace:")
w("```python")
w("# paper_trading_pipeline.py:378")
w("signals = engine.predict_cross_section(features=X, prev_signal=prev_array, fold_idx=-1)")
w("")
w("# production_engine.py:661-662")
w("if fold_idx < 0:")
w(f"    fold_idx = n_available + fold_idx   # -1 -> {last_fold} (last fold)")
w("")
w("# production_engine.py:670-671")
w(f"for seed in self.seeds:")
w(f"    model = self._models[seed][{last_fold}]  # self._models[seed][{last_fold}]")
w("    raw_preds.append(model.predict(X))")
w("")
w("# production_engine.py:681")
w("ensemble_raw = np.mean(raw_preds, axis=0)  # Equal-weight seed average")
w("")
w("# production_engine.py:684")
w("ranked = cs_rank(ensemble_raw)  # Cross-sectional rank -> [0, 1]")
w("```")
w()
w("### Full Call Chain")
w()
w("| Step | Location | Action |")
w("|------|----------|--------|")
w("| 1 | `paper_trading_pipeline.py:547` | `ProductionAlphaEngine.load_models('output/production_models_v2_full')` |")
w(f"| 2 | `production_engine.py:975-991` | Backtest mode: loads ALL {engine.n_models} models ({engine._n_folds} folds x {len(engine.seeds)} seeds) |")
w("| 3 | `paper_trading_pipeline.py:378` | `engine.predict_cross_section(X, prev_signal, fold_idx=-1)` |")
w(f"| 4 | `production_engine.py:662` | fold_idx=-1 -> {last_fold} (last fold) |")
w(f"| 5 | `production_engine.py:670-675` | For each seed: model[{last_fold}].predict(X) |")
w("| 6 | `production_engine.py:681` | ensemble_raw = mean(seed_preds) |")
w("| 7 | `production_engine.py:684` | ranked = cs_rank(ensemble_raw) |")
w()
w("### Audit Verdict")
w()
w("| Audit Item | Finding |")
w("|------------|---------|")
w(f"| Models loaded | {engine.n_models} ({len(engine.seeds)} seeds x {engine._n_folds} folds) |")
w(f"| Models USED per inference | {len(engine.seeds)} ({len(engine.seeds)} seeds x 1 fold) |")
w(f"| Utilization rate | {len(engine.seeds)/engine.n_models*100:.1f}% ({len(engine.seeds)}/{engine.n_models}) |")
w(f"| Which fold | Last fold (index {last_fold}) |")
w(f"| Inference methodology | Fold {last_fold} used for ALL dates (including 2017) - global fold selection |")
w(f"| prev_signal usage | Validated but NOT used in model.predict() - turnover penalty only affects training |")
w()
w("**Critical finding**: Production loads 213 models but only uses **3**. ")
w(f"The last fold (index {last_fold}) is applied to ALL dates. This single-fold strategy ")
w("provides zero diversification across training windows and is maximally exposed to ")
w("regime-specific model biases.")
w()

w("---")
w("## Part 3: Fold Ensemble Ablation Results")
w()

w("### Scheme Definitions")
w()
w("| Scheme | Description | Folds | Models (3 seeds) |")
w("|--------|-------------|-------|------------------|")
for sn in scheme_order:
    info = schemes[sn]
    w(f"| **{info['desc']}** | {info['desc']} | {info['folds'][:3]}... ({len(info['folds'])} folds) | {len(info['folds']) * len(engine.seeds)} |")
w()

w("### Unified Metrics Table")
w()
w("| Ensemble | Models | RankCorr | Top30Overlap | IC | IC_IR | Sharpe | MaxDD | Turnover |")
w("|----------|--------|----------|-------------|-----|-------|--------|-------|----------|")

for sn in scheme_order:
    r = results.get(sn, {})
    if not r:
        continue
    w(f"| **{r.get('desc', sn)}** | {r['n_models']} | {r['rank_corr']:.4f} | {r['overlap']:.3f} | "
      f"{r['ic']:.4f} | {r['ic_ir']:.4f} | {r['sharpe']:.2f} | {r['maxdd']:.3f} | {r['turnover']:.3f} |")
w()

# Best per metric
valid_results = {k: v for k, v in results.items() if v.get('rank_corr') is not None}
best_rc = max(valid_results.items(), key=lambda x: x[1].get("rank_corr", -999)) if valid_results else (None, {})
best_sharpe = max(valid_results.items(), key=lambda x: x[1].get("sharpe", -999)) if valid_results else (None, {})
best_to = min(valid_results.items(), key=lambda x: x[1].get("turnover", 999)) if valid_results else (None, {})
prod = results.get("A_production", {})

w("### Key Comparisons")
w()
w("| Metric | Production (A) | Best Scheme | Delta | Winner |")
w("|--------|---------------|-------------|-------|--------|")
if prod and best_rc[1]:
    w(f"| RankCorr | {prod.get('rank_corr', 0):.4f} | {best_rc[1]['rank_corr']:.4f} | {best_rc[1]['rank_corr'] - prod.get('rank_corr', 0):+.4f} | {best_rc[1].get('desc', '?')} |")
if prod and best_sharpe[1]:
    w(f"| Sharpe | {prod.get('sharpe', 0):.2f} | {best_sharpe[1]['sharpe']:.2f} | {best_sharpe[1]['sharpe'] - prod.get('sharpe', 0):+.2f} | {best_sharpe[1].get('desc', '?')} |")
if prod and best_to[1]:
    w(f"| Turnover | {prod.get('turnover', 0):.3f} | {best_to[1]['turnover']:.3f} | {best_to[1]['turnover'] - prod.get('turnover', 0):+.3f} | {best_to[1].get('desc', '?')} |")
w()

w("---")
w("## Part 5: Root Cause Attribution")
w()

# Q1
prod_rc = prod.get("rank_corr", np.nan)
all_rc = results.get("D_all", {}).get("rank_corr", np.nan)
w("### Q1: Is RankCorr=0.718 a fold selection problem or a model problem?")
w()
if not np.isnan(prod_rc) and not np.isnan(all_rc):
    delta_rc = all_rc - prod_rc
    best_rc_val = best_rc[1].get("rank_corr", 0)
    delta_best = best_rc_val - prod_rc
    if abs(delta_rc) < 0.03 and abs(delta_best) < 0.03:
        w("**MODEL problem, not fold selection.**")
        w(f"All schemes produce similar RankCorr ({prod_rc:.4f} to {max(r['rank_corr'] for r in results.values() if r.get('rank_corr') is not None):.4f}).")
        w("Changing fold configuration does NOT meaningfully improve stability.")
        w("The root cause lies in model quality, feature set, or training methodology.")
    else:
        w("**Fold SELECTION matters, but not decisively.**")
        w(f"Production RankCorr={prod_rc:.4f}, Best RankCorr={best_rc_val:.4f} (Delta={delta_best:+.4f}).")
w()

# Q2
w("### Q2: Does using all folds significantly improve metrics?")
w()
w("| Metric | Production (A) | All Folds (D) | Delta | Significant? |")
w("|--------|---------------|--------------|-------|-------------|")
for metric, key in [("RankCorr", "rank_corr"), ("Sharpe", "sharpe"), ("Turnover", "turnover")]:
    pv = prod.get(key, np.nan)
    av = results.get("D_all", {}).get(key, np.nan)
    if not np.isnan(pv) and not np.isnan(av):
        d = av - pv
        sig = "YES" if abs(d) > 0.02 else "Minor" if abs(d) > 0.005 else "NO"
        w(f"| {metric} | {pv:.4f} | {av:.4f} | {d:+.4f} | {sig} |")
w()

# Q3
w("### Q3: What is the optimal ensemble configuration?")
w()
ranked_schemes = sorted(results.items(),
    key=lambda x: (x[1].get("sharpe", -999), x[1].get("rank_corr", -999)), reverse=True)
w("| Rank | Scheme | RankCorr | Sharpe | Turnover | Reason |")
w("|------|--------|----------|--------|----------|--------|")
for rank, (sn, r) in enumerate(ranked_schemes, 1):
    reasons = {
        "A_production": "Current production - single fold, regime-concentrated",
        "B_last3": "Recent folds, moderate diversification",
        "C_last10": "Broader recent window",
        "D_all": "Maximum fold diversification",
        "E_top25pct": "IC-based selection removes low-quality folds",
        "F_top10pct": "IC-based selection, highest average IC folds",
    }
    w(f"| {rank} | **{r['desc']}** | {r['rank_corr']:.4f} | {r['sharpe']:.2f} | {r['turnover']:.3f} | {reasons.get(sn, '')} |")
w()

if ranked_schemes:
    best = ranked_schemes[0]
    w("### Recommended Configuration")
    w()
    w(f"**Best scheme**: {best[1]['desc']}")
    w(f"- Models: {best[1]['n_models']}")
    w(f"- RankCorr: {best[1]['rank_corr']:.4f}")
    w(f"- Sharpe: {best[1]['sharpe']:.2f}")
    w(f"- Turnover: {best[1]['turnover']:.3f}")
w()

w("### Seed Recommendation")
w()
w("Based on Experiment D (seed-pair r ~ 0.966), 3 seeds are redundant.")
w(f"**Recommendation**: Use 1 seed (any of {engine.seeds}). Saves 67% inference cost with negligible signal loss.")
w()

w("---")
w("## Final Verdict")
w()
prod_sharpe = prod.get("sharpe", np.nan)
if not np.isnan(prod_sharpe) and best_sharpe[1]:
    best_sharpe_val = best_sharpe[1]["sharpe"]
    sharpe_delta = best_sharpe_val - prod_sharpe
    if sharpe_delta > 0.05:
        w("### YES - Fold Ensemble Architecture is a SIGNIFICANT contributor to Sharpe degradation.")
        w(f"Production (fold=-1): Sharpe={prod_sharpe:.2f}")
        w(f"Best ensemble: Sharpe={best_sharpe_val:.2f} (Delta={sharpe_delta:+.2f})")
    elif sharpe_delta > 0.02:
        w("### PARTIALLY - Fold architecture contributes but is not the dominant factor.")
        w(f"Sharpe improvement: {prod_sharpe:.2f} -> {best_sharpe_val:.2f} (Delta={sharpe_delta:+.2f})")
    else:
        w("### NO - Fold Ensemble Architecture is NOT the primary cause of Sharpe degradation.")
        w(f"All schemes produce similar Sharpe (range: {prod_sharpe:.2f} to {best_sharpe_val:.2f}).")
        w("Delta = {:.3f} - negligible. The root cause lies elsewhere.".format(sharpe_delta))
        w()
        w("**The Sharpe degradation from 0.70 (V1) to 0.51 (V2_Full) is NOT explained by fold selection.**")
        w("The model itself (features, training methodology, architecture) is the primary driver.")
else:
    w("### Insufficient data for final verdict.")

w()
w("---")
w(f"*Report generated: {pd.Timestamp.now()}*")

# Write report
output_text = "\n".join(R)
REPORT_PATH.write_text(output_text, encoding="utf-8")
logger.info(f"Report saved: {REPORT_PATH}")

# Print summary table
print("\n" + "=" * 90)
print("UNIFIED RESULTS TABLE")
print("=" * 90)
print(f"{'Scheme':<30} {'Models':>7} {'RankCorr':>9} {'Overlap':>8} {'IC':>8} {'IC_IR':>8} {'Sharpe':>7} {'MaxDD':>8} {'TO':>7}")
print("-" * 90)
for sn in scheme_order:
    r = results.get(sn, {})
    if not r: continue
    print(f"{r['desc']:<30} {r['n_models']:>7} {r['rank_corr']:>9.4f} {r['overlap']:>8.3f} "
          f"{r['ic']:>8.4f} {r['ic_ir']:>8.4f} {r['sharpe']:>7.2f} {r['maxdd']:>8.3f} {r['turnover']:>7.3f}")
print("=" * 90)
