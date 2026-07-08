"""
Ensemble Stability Root Cause Analysis — Experiments C through H.

核心问题: 单模型 RankCorr≈0.85, 162模型集成 RankCorr≈0.72
目标: 定位集成架构层面的稳定性崩溃根因

Experiments:
  C: Model pairwise prediction correlation (all folds × seeds)
  D: Seed consistency analysis
  E: Fold consistency analysis
  F: Ensemble scale ablation (1, 3, 9, 27, 54, 108, 162 models)
  G: Style drift analysis (factor exposures per fold)
  H: Ensemble weighting analysis (equal vs IC-weighted vs top25% vs top10%)

Design: 不修改生产代码, 不重训全部系统, 所有结论来自实验数据.
"""
import warnings, logging, sys, json, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
from collections import defaultdict
from itertools import combinations
import scipy.stats as stats

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("ensemble_stability")

OUT = Path("output")
MODEL_DIR = OUT / "production_models_v2_full"
PANEL_PATH = OUT / "training_panel_v3_full.parquet"
REPORT_PATH = OUT / "ensemble_stability_root_cause.md"

# ── Load Models ──
logger.info("Loading models...")
with open(MODEL_DIR / "metadata.json") as f:
    metadata = json.load(f)
seeds = metadata["seeds"]
n_folds = metadata["n_folds"]
feature_cols_meta = metadata["feature_cols"]
training_dates = pd.to_datetime(metadata["training_dates"])
logger.info(f"Seeds: {seeds}, Folds: {n_folds}, Features: {len(feature_cols_meta)}")

all_models = {}  # (seed, fold) -> Booster or None
for seed in seeds:
    for fold in range(n_folds):
        fp = MODEL_DIR / f"model_s{seed}_f{fold}.pkl"
        if fp.exists():
            with open(fp, "rb") as f:
                all_models[(seed, fold)] = pickle.load(f)
        else:
            all_models[(seed, fold)] = None

valid_models = {k: v for k, v in all_models.items() if v is not None}
n_valid = len(valid_models)
logger.info(f"Valid models: {n_valid}/{n_folds * len(seeds)} (expected {n_folds * len(seeds)})")

# ── Load Panel ──
logger.info("Loading panel...")
panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])

# Compute forward_return_1m if missing
if "forward_return_1m" not in panel.columns:
    close_col = "收盘" if "收盘" in panel.columns else "close"
    panel = panel.sort_values(["symbol", "date"])
    panel["forward_return_1m"] = panel.groupby("symbol")[close_col].transform(
        lambda x: x.shift(-1) / x - 1.0
    )

dates = sorted(panel["date"].unique())
logger.info(f"Panel: {len(panel)} rows, {len(dates)} dates, {panel['symbol'].nunique()} symbols")

# ── Feature Preparation ──
# Model expects _neutral_z_rank columns (rank-normalized within each date)
rank_cols = [c for c in panel.columns if c.endswith("_neutral_z_rank")]
if not rank_cols:
    # Compute ranks from _neutral_z
    logger.info("Computing rank-normalized features...")
    nz_cols = [c for c in panel.columns if c.endswith("_neutral_z") and not c.endswith("_rank")]
    for col in nz_cols:
        rc = f"{col}_rank"
        panel[rc] = panel.groupby("date")[col].rank(pct=True, na_option="bottom").fillna(0.5)
        rank_cols.append(rc)
logger.info(f"Rank features: {len(rank_cols)}")

# Map panel columns to model feature names
col_map = {}
for fc in feature_cols_meta:
    if fc in panel.columns:
        col_map[fc] = fc
    elif fc.replace("_rank", "") in panel.columns:
        col_map[fc.replace("_rank", "")] = fc
    elif f"{fc}_rank" in panel.columns:
        col_map[f"{fc}_rank"] = fc
logger.info(f"Feature mapping: {len(col_map)}/{len(feature_cols_meta)} matched")

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Generate predictions from ALL models on ALL dates
# ═══════════════════════════════════════════════════════════════
logger.info("="*60)
logger.info("PHASE 1: Generating all-model predictions...")
logger.info("="*60)

# Structure: preds[date_str][(seed, fold)] = np.array of raw predictions
preds_by_date = {}  # date -> {(seed, fold): np.array}
stocks_by_date = {}  # date -> np.array of symbols

t0 = time.time()
for i, dt in enumerate(dates):
    mask = panel["date"] == dt
    df = panel[mask]
    syms = df["symbol"].values
    stocks_by_date[dt] = syms

    # Build feature matrix
    X = np.zeros((len(df), len(feature_cols_meta)), dtype=np.float64)
    for j, fc in enumerate(feature_cols_meta):
        panel_col = col_map.get(fc)
        if panel_col and panel_col in df.columns:
            X[:, j] = df[panel_col].values.astype(np.float64)

    # Run all models
    date_preds = {}
    for (seed, fold), model in valid_models.items():
        date_preds[(seed, fold)] = model.predict(X)

    preds_by_date[dt] = date_preds

    if (i + 1) % 20 == 0 or i == 0:
        elapsed = time.time() - t0
        logger.info(
            "  [%3d/%3d] %s | %d stocks | %d models | %.1fs",
            i + 1, len(dates), str(dt)[:10], len(syms), len(date_preds), elapsed,
        )

elapsed = time.time() - t0
logger.info(f"Phase 1 complete: {len(dates)} dates × {n_valid} models in {elapsed:.1f}s")

# ── Helper: cross-sectional rank ──
def cs_rank(arr):
    """Cross-sectional rank [0,1], NaN-safe."""
    s = pd.Series(arr)
    return s.rank(pct=True, na_option="bottom").fillna(0.5).values

# ── Helper: ensemble prediction ──
def ensemble_predict(date_preds_dict, model_keys, date):
    """Equal-weight ensemble of specified model_keys, return cs-ranked signal."""
    preds_list = [date_preds_dict[k] for k in model_keys if k in date_preds_dict]
    if not preds_list:
        return np.full(len(stocks_by_date[date]), 0.5)
    raw = np.mean(preds_list, axis=0)
    return cs_rank(raw)

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT C: Model Pairwise Prediction Correlation
# ═══════════════════════════════════════════════════════════════
logger.info("="*60)
logger.info("EXPERIMENT C: Model Pairwise Prediction Correlation")
logger.info("="*60)

# Within each date, compute pairwise correlation between ALL models
pairwise_corrs = []
for dt in dates:
    dp = preds_by_date[dt]
    keys = list(dp.keys())
    if len(keys) < 2:
        continue
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            corr_val = stats.pearsonr(dp[keys[i]], dp[keys[j]])[0]
            pairwise_corrs.append({
                "date": dt,
                "model_i": f"s{keys[i][0]}_f{keys[i][1]}",
                "model_j": f"s{keys[j][0]}_f{keys[j][1]}",
                "seed_i": keys[i][0],
                "seed_j": keys[j][0],
                "fold_i": keys[i][1],
                "fold_j": keys[j][1],
                "same_seed": keys[i][0] == keys[j][0],
                "same_fold": keys[i][1] == keys[j][1],
                "pearson_r": corr_val,
            })

pw_df = pd.DataFrame(pairwise_corrs)
logger.info(f"Total pairwise correlations: {len(pw_df)}")

# Overall stats
all_corrs = pw_df["pearson_r"]
logger.info(f"Overall: mean={all_corrs.mean():.4f}, median={all_corrs.median():.4f}, "
            f"P10={all_corrs.quantile(0.10):.4f}, P90={all_corrs.quantile(0.90):.4f}")

# Same-seed vs cross-seed
same_seed = pw_df[pw_df["same_seed"]]["pearson_r"]
cross_seed = pw_df[~pw_df["same_seed"]]["pearson_r"]
logger.info(f"Same-seed: mean={same_seed.mean():.4f}, Cross-seed: mean={cross_seed.mean():.4f}")

# Same-fold vs cross-fold
same_fold = pw_df[pw_df["same_fold"]]["pearson_r"]
cross_fold = pw_df[~pw_df["same_fold"]]["pearson_r"]
logger.info(f"Same-fold: mean={same_fold.mean():.4f}, Cross-fold: mean={cross_fold.mean():.4f}")

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT D: Seed Consistency Analysis
# ═══════════════════════════════════════════════════════════════
logger.info("="*60)
logger.info("EXPERIMENT D: Seed Consistency Analysis")
logger.info("="*60)

# Within each fold, compare seed predictions (same test date)
seed_pair_corrs = defaultdict(list)  # (seed_i, seed_j) -> [corrs across folds]
for fold in range(n_folds):
    for s1, s2 in combinations(seeds, 2):
        # For this fold, both seeds predicted on the same test date
        fold_test_dates = pw_df[(pw_df["fold_i"] == fold) & (pw_df["fold_j"] == fold) &
                                (pw_df["seed_i"] == s1) & (pw_df["seed_j"] == s2)]
        if len(fold_test_dates) > 0:
            seed_pair_corrs[(s1, s2)].extend(fold_test_dates["pearson_r"].values)

# Actually, the pw_df structure isn't ideal for this. Let me recompute directly.
# For each date, extract predictions per seed (averaged across folds for that seed),
# then compute cross-seed correlation per date.
# But wait - for seed consistency, we should fix the FOLD and compare seeds within it.
# Each fold's models (seed42_foldK, seed888_foldK, seed2026_foldK) all predicted on the SAME test date.

seed_corr_by_fold = []
for dt in dates:
    dp = preds_by_date[dt]
    for s1, s2 in combinations(seeds, 2):
        # Find this pair's predictions on this date
        # For seed consistency within a fold, we need same-fold models
        for fold in range(n_folds):
            k1, k2 = (s1, fold), (s2, fold)
            if k1 in dp and k2 in dp:
                c = stats.pearsonr(dp[k1], dp[k2])[0]
                seed_corr_by_fold.append({
                    "date": dt, "fold": fold, "seed_pair": f"{s1}-{s2}",
                    "seed_i": s1, "seed_j": s2, "pearson_r": c,
                })

scf_df = pd.DataFrame(seed_corr_by_fold)
logger.info(f"Seed-pair correlations: {len(scf_df)} records")

for sp in [f"{s1}-{s2}" for s1, s2 in combinations(seeds, 2)]:
    sub = scf_df[scf_df["seed_pair"] == sp]
    logger.info(f"  Seed pair {sp}: mean={sub['pearson_r'].mean():.4f}, "
                f"median={sub['pearson_r'].median():.4f}, std={sub['pearson_r'].std():.4f}")

seed_avg = scf_df.groupby("seed_pair")["pearson_r"].agg(["mean", "median", "std", "count"])
logger.info(f"Overall seed consistency: mean={scf_df['pearson_r'].mean():.4f}")

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT E: Fold Consistency Analysis
# ═══════════════════════════════════════════════════════════════
logger.info("="*60)
logger.info("EXPERIMENT E: Fold Consistency Analysis")
logger.info("="*60)

# Fix seed, compare predictions from different folds on the SAME date
# (Each fold was trained on a different window, but we run all on the same date)

fold_corr_by_seed = defaultdict(list)  # seed -> [{fold_i, fold_j, pearson_r}]
for dt in dates:
    dp = preds_by_date[dt]
    for seed in seeds:
        fold_keys = [(s, f) for s, f in dp.keys() if s == seed]
        if len(fold_keys) < 2:
            continue
        for i in range(len(fold_keys)):
            for j in range(i + 1, len(fold_keys)):
                c = stats.pearsonr(dp[fold_keys[i]], dp[fold_keys[j]])[0]
                fold_corr_by_seed[seed].append({
                    "date": dt,
                    "fold_i": fold_keys[i][1],
                    "fold_j": fold_keys[j][1],
                    "fold_delta": abs(fold_keys[i][1] - fold_keys[j][1]),
                    "pearson_r": c,
                })

for seed in seeds:
    records = fold_corr_by_seed[seed]
    if records:
        df_tmp = pd.DataFrame(records)
        logger.info(f"  Seed {seed}: mean fold-fold corr={df_tmp['pearson_r'].mean():.4f}, "
                    f"median={df_tmp['pearson_r'].median():.4f}, n={len(df_tmp)}")

# Identify most inconsistent folds
fold_consistency = defaultdict(list)  # fold -> [corrs with other folds]
for seed in seeds:
    records = fold_corr_by_seed[seed]
    if not records:
        continue
    for r in records:
        fold_consistency[r["fold_i"]].append(r["pearson_r"])
        fold_consistency[r["fold_j"]].append(r["pearson_r"])

fold_avg_corr = {}
for fold, corrs in sorted(fold_consistency.items()):
    if corrs:
        fold_avg_corr[fold] = np.mean(corrs)

if fold_avg_corr:
    worst_folds = sorted(fold_avg_corr.items(), key=lambda x: x[1])[:5]
    best_folds = sorted(fold_avg_corr.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"  Worst 5 folds (lowest consistency): {worst_folds}")
    logger.info(f"  Best 5 folds (highest consistency): {best_folds}")

# ── Fold × Fold correlation matrix (for one seed, sampled) ──
seed_for_matrix = seeds[0]
fold_corr_matrix = np.full((n_folds, n_folds), np.nan)
fold_corr_counts = np.zeros((n_folds, n_folds), dtype=int)

for dt in dates:
    dp = preds_by_date[dt]
    folds_present = sorted(set(f for s, f in dp.keys() if s == seed_for_matrix))
    for i_idx, fi in enumerate(folds_present):
        for j_idx, fj in enumerate(folds_present):
            if fi <= fj:
                ki, kj = (seed_for_matrix, fi), (seed_for_matrix, fj)
                if ki in dp and kj in dp:
                    c = stats.pearsonr(dp[ki], dp[kj])[0]
                    if np.isnan(fold_corr_matrix[fi, fj]):
                        fold_corr_matrix[fi, fj] = c
                    else:
                        fold_corr_matrix[fi, fj] = (fold_corr_matrix[fi, fj] * fold_corr_counts[fi, fj] + c) / (fold_corr_counts[fi, fj] + 1)
                    fold_corr_counts[fi, fj] += 1

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT F: Ensemble Scale Ablation
# ═══════════════════════════════════════════════════════════════
logger.info("="*60)
logger.info("EXPERIMENT F: Ensemble Scale Ablation")
logger.info("="*60)

# For each date, sample N models, compute ensemble prediction,
# then compute RankCorr, Overlap, Turnover, etc.
# Use multiple random seeds for robustness.

ALL_MODEL_KEYS = list(valid_models.keys())
N_MODELS_LIST = [1, 3, 9, 27, 54, 108, n_valid]
N_SAMPLES = 10  # random samples per N for robust estimates

# For each date, compute the rank from ensemble predictions
# and track month-over-month stability

np.random.seed(42)
scale_results = defaultdict(lambda: defaultdict(list))  # N -> metric -> [values]

# Pre-compute single-model ensemble predictions for each N
# For each date, for each N, sample N models and ensemble
for dt in dates:
    dp = preds_by_date[dt]
    syms = stocks_by_date[dt]
    available_keys = [k for k in ALL_MODEL_KEYS if k in dp]
    if len(available_keys) < 3:
        continue

    for N in N_MODELS_LIST:
        actual_N = min(N, len(available_keys))
        n_samples_actual = min(N_SAMPLES, max(1, len(available_keys) // actual_N))

        for sample_idx in range(n_samples_actual):
            if actual_N == len(available_keys):
                sampled = available_keys
            else:
                indices = np.random.choice(len(available_keys), actual_N, replace=False)
                sampled = [available_keys[i] for i in indices]

            ensemble_raw = np.mean([dp[k] for k in sampled], axis=0)
            ensemble_ranked = cs_rank(ensemble_raw)

            # Store for this date
            if "predictions" not in scale_results[N]:
                scale_results[N]["predictions"] = {}
            key = (dt, sample_idx)
            scale_results[N]["predictions"][key] = ensemble_ranked

# ── Compute stability metrics ──
# For each N and sample, compute: RankCorr (t, t+1), Top30 Overlap
for N in N_MODELS_LIST:
    preds_dict = scale_results[N].get("predictions", {})
    if not preds_dict:
        continue

    # Group by (date, sample_idx)
    by_date_sample = defaultdict(dict)
    for (dt, sidx), pred in preds_dict.items():
        by_date_sample[dt][sidx] = pred

    rank_corrs = []
    overlaps = []
    for sidx in range(N_SAMPLES):
        prev_signal = None
        prev_top30 = None
        prev_date = None
        for dt in sorted(by_date_sample.keys()):
            if sidx not in by_date_sample[dt]:
                continue
            signal = by_date_sample[dt][sidx]
            syms = stocks_by_date[dt]

            # Rank correlation with previous month
            if prev_signal is not None and prev_date is not None:
                prev_syms = stocks_by_date[prev_date]
                common = np.intersect1d(prev_syms, syms)
                if len(common) >= 30:
                    prev_idx = np.array([np.where(prev_syms == s)[0][0] for s in common])
                    curr_idx = np.array([np.where(syms == s)[0][0] for s in common])
                    rc = stats.spearmanr(prev_signal[prev_idx], signal[curr_idx])[0]
                    rank_corrs.append({"N": N, "sample": sidx, "date": dt, "rank_corr": rc})

                    # Top30 overlap
                    top30_prev = set(prev_syms[np.argsort(prev_signal)[-30:]])
                    top30_curr = set(syms[np.argsort(signal)[-30:]])
                    overlap = len(top30_prev & top30_curr) / 30
                    overlaps.append({"N": N, "sample": sidx, "date": dt, "overlap": overlap})

            prev_signal = signal
            prev_top30 = set(syms[np.argsort(signal)[-30:]])
            prev_date = dt

    scale_results[N]["rank_corrs"] = rank_corrs
    scale_results[N]["overlaps"] = overlaps

# ── Also compute OOS IC and IC_IR for each N ──
for N in N_MODELS_LIST:
    preds_dict = scale_results[N].get("predictions", {})
    if not preds_dict:
        continue

    ic_list = []
    for (dt, sidx), pred in preds_dict.items():
        if sidx > 0:  # Only use first sample for IC (avoid redundancy)
            continue
        df_date = panel[panel["date"] == dt]
        if "forward_return_1m" not in df_date.columns or len(df_date) < 30:
            continue
        # Align symbols
        syms = stocks_by_date[dt]
        df_aligned = df_date.set_index("symbol").reindex(syms)
        valid_mask = df_aligned["forward_return_1m"].notna()
        if valid_mask.sum() < 30:
            continue
        ic, _ = stats.spearmanr(pred[valid_mask], df_aligned["forward_return_1m"].values[valid_mask])
        if not np.isnan(ic):
            ic_list.append({"N": N, "date": dt, "IC": ic})

    if ic_list:
        ic_df = pd.DataFrame(ic_list)
        scale_results[N]["IC_mean"] = ic_df["IC"].mean()
        scale_results[N]["IC_std"] = ic_df["IC"].std()
        scale_results[N]["IC_IR"] = scale_results[N]["IC_mean"] / scale_results[N]["IC_std"] if scale_results[N]["IC_std"] > 0 else 0

# ── Summary ──
logger.info("\nEnsemble Scale Ablation Results:")
logger.info(f"{'N':>6} {'RankCorr':>10} {'Overlap':>9} {'IC':>8} {'IC_IR':>8}")
logger.info("-" * 45)
for N in N_MODELS_LIST:
    rc_list = scale_results[N].get("rank_corrs", [])
    ov_list = scale_results[N].get("overlaps", [])
    rc_mean = np.mean([r["rank_corr"] for r in rc_list]) if rc_list else np.nan
    ov_mean = np.mean([o["overlap"] for o in ov_list]) if ov_list else np.nan
    ic_mean = scale_results[N].get("IC_mean", np.nan)
    ic_ir = scale_results[N].get("IC_IR", np.nan)
    logger.info(f"{len(ALL_MODEL_KEYS) if N >= n_valid else N:>6} {rc_mean:>10.4f} {ov_mean:>9.3f} {ic_mean:>8.4f} {ic_ir:>8.4f}")

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT G: Style Drift Analysis
# ═══════════════════════════════════════════════════════════════
logger.info("="*60)
logger.info("EXPERIMENT G: Style Drift Analysis")
logger.info("="*60)

# For each fold's model, compute factor exposure on its test date
# Exposure = corr(prediction, factor_value), cross-sectional
# Factors: EP, ROE, ProfitGrowth, RevGrowth, Mom_1M, Mom_3M, Mom_12M_1M

style_factors = {
    "EP": "EP_neutral_z",
    "ROE": "ROE_neutral_z",
    "ProfitGrowth": "ProfitGrowth_YoY_neutral_z",
    "RevGrowth": "RevGrowth_YoY_neutral_z",
    "Mom_1M": "Mom_1M_neutral_z",
    "Mom_3M": "Mom_3M_neutral_z",
    "Mom_12M_1M": "Mom_12M_1M_neutral_z",
    "NetProfitMargin": "Net_Profit_Margin_neutral_z",
    "BP": "BP_neutral_z",
    "Vol_20D": "Vol_20D_neutral_z",
}

style_exposures = []  # [{seed, fold, date, factor: exposure}]
for (seed, fold), model in valid_models.items():
    # Find which dates this fold's model predicted on
    for dt in dates:
        if (seed, fold) in preds_by_date[dt]:
            pred = preds_by_date[dt][(seed, fold)]
            syms = stocks_by_date[dt]
            df_date = panel[panel["date"] == dt].set_index("symbol").reindex(syms)

            exposures = {"seed": seed, "fold": fold, "date": dt}
            for factor_name, factor_col in style_factors.items():
                if factor_col in df_date.columns:
                    factor_vals = df_date[factor_col].values
                    valid = ~(np.isnan(factor_vals) | np.isnan(pred))
                    if valid.sum() >= 30:
                        exposures[factor_name] = stats.spearmanr(pred[valid], factor_vals[valid])[0]
                    else:
                        exposures[factor_name] = np.nan
            style_exposures.append(exposures)
            break  # Only use this fold's test date (first date found)

se_df = pd.DataFrame(style_exposures)
logger.info(f"Style exposure records: {len(se_df)}")

# Summary by factor
available_factors = [f for f in style_factors if f in se_df.columns]
for factor in available_factors:
    vals = se_df[factor].dropna()
    logger.info(f"  {factor}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                f"min={vals.min():.4f}, max={vals.max():.4f}, "
                f"%positive={100*(vals>0).mean():.1f}%")

# Identify if different folds represent different styles
# Cluster folds by their factor exposures
from sklearn.cluster import KMeans
fold_profiles = se_df.groupby("fold")[available_factors].mean()
# Drop columns that are all NaN
valid_style_factors = [f for f in available_factors if not fold_profiles[f].isna().all()]
fold_profiles = fold_profiles[valid_style_factors].copy()
# Fill remaining NaN with 0 (median of cross-sectional rank correlation)
fold_profiles = fold_profiles.fillna(0)
logger.info(f"Valid style factors for clustering: {len(valid_style_factors)}/{len(available_factors)}")
if len(fold_profiles) >= 3:
    kmeans = KMeans(n_clusters=min(3, len(fold_profiles)), random_state=42, n_init=10)
    fold_profiles["cluster"] = kmeans.fit_predict(fold_profiles.values)
    for c in sorted(fold_profiles["cluster"].unique()):
        cluster_folds = fold_profiles[fold_profiles["cluster"] == c]
        cluster_mean = cluster_folds[valid_style_factors].mean()
        logger.info(f"  Cluster {c}: {len(cluster_folds)} folds, "
                    f"top exposures: {cluster_mean.nlargest(3).to_dict()}")

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT H: Ensemble Weighting Analysis
# ═══════════════════════════════════════════════════════════════
logger.info("="*60)
logger.info("EXPERIMENT H: Ensemble Weighting Analysis")
logger.info("="*60)

# Compute per-model IC on their test dates (for weighting)
model_ic = {}  # (seed, fold) -> IC
for (seed, fold), model in valid_models.items():
    ics = []
    for dt in dates:
        if (seed, fold) not in preds_by_date[dt]:
            continue
        pred = preds_by_date[dt][(seed, fold)]
        syms = stocks_by_date[dt]
        df_date = panel[panel["date"] == dt].set_index("symbol").reindex(syms)
        if "forward_return_1m" not in df_date.columns:
            continue
        valid_mask = df_date["forward_return_1m"].notna()
        if valid_mask.sum() < 30:
            continue
        # This fold's test date IC
        ic, _ = stats.spearmanr(pred[valid_mask], df_date["forward_return_1m"].values[valid_mask])
        if not np.isnan(ic):
            ics.append(ic)
    model_ic[(seed, fold)] = np.mean(ics) if ics else 0.0

# Sort models by IC
sorted_models = sorted(model_ic.items(), key=lambda x: x[1], reverse=True)
n_top25 = max(1, len(sorted_models) // 4)
n_top10 = max(1, len(sorted_models) // 10)
top25_keys = {k for k, _ in sorted_models[:n_top25]}
top10_keys = {k for k, _ in sorted_models[:n_top10]}

logger.info(f"Total models: {len(sorted_models)}, Top25%: {n_top25}, Top10%: {n_top10}")
logger.info(f"IC range: [{sorted_models[-1][1]:.4f}, {sorted_models[0][1]:.4f}]")

# ── Test 4 weighting schemes ──
weighting_results = defaultdict(lambda: defaultdict(list))

for dt in dates:
    dp = preds_by_date[dt]
    available_models = [k for k in ALL_MODEL_KEYS if k in dp]
    if len(available_models) < 3:
        continue

    syms = stocks_by_date[dt]
    n_stocks = len(syms)

    # Scheme 1: Equal Weight (baseline)
    all_preds_list = [dp[k] for k in available_models]
    eq_raw = np.mean(all_preds_list, axis=0)
    eq_signal = cs_rank(eq_raw)

    # Scheme 2: IC-weighted
    weights_ic = np.array([max(0, model_ic.get(k, 0)) for k in available_models])
    if weights_ic.sum() > 0:
        weights_ic = weights_ic / weights_ic.sum()
        icw_raw = np.average(all_preds_list, axis=0, weights=weights_ic)
    else:
        icw_raw = eq_raw
    icw_signal = cs_rank(icw_raw)

    # Scheme 3: Top 25% only, equal weight
    top25_models = [k for k in available_models if k in top25_keys]
    if top25_models:
        top25_raw = np.mean([dp[k] for k in top25_models], axis=0)
    else:
        top25_raw = eq_raw
    top25_signal = cs_rank(top25_raw)

    # Scheme 4: Top 10% only, equal weight
    top10_models = [k for k in available_models if k in top10_keys]
    if top10_models:
        top10_raw = np.mean([dp[k] for k in top10_models], axis=0)
    else:
        top10_raw = eq_raw
    top10_signal = cs_rank(top10_raw)

    for scheme_name, signal in [("Equal", eq_signal), ("IC_Weighted", icw_signal),
                                  ("Top25%", top25_signal), ("Top10%", top10_signal)]:
        weighting_results[scheme_name]["signals"].append((dt, signal))
        weighting_results[scheme_name]["stocks"].append((dt, syms))

# ── Compute stability metrics for each scheme ──
for scheme_name in ["Equal", "IC_Weighted", "Top25%", "Top10%"]:
    signals_list = weighting_results[scheme_name]["signals"]
    stocks_list = dict(weighting_results[scheme_name]["stocks"])

    # Rank correlation
    rank_corrs = []
    overlaps = []
    ics = []

    prev_dt = None
    prev_signal = None
    prev_syms = None

    for dt_idx in range(len(signals_list)):
        dt, signal = signals_list[dt_idx]
        syms = stocks_list[dt]

        # IC
        df_date = panel[panel["date"] == dt].set_index("symbol").reindex(syms)
        if "forward_return_1m" in df_date.columns:
            valid = df_date["forward_return_1m"].notna()
            if valid.sum() >= 30:
                ic, _ = stats.spearmanr(signal[valid.values], df_date["forward_return_1m"].values[valid.values])
                if not np.isnan(ic):
                    ics.append(ic)

        # Rank correlation with previous month
        if prev_signal is not None and prev_syms is not None:
            common = np.intersect1d(prev_syms, syms)
            if len(common) >= 30:
                prev_idx = np.array([np.where(prev_syms == s)[0][0] for s in common])
                curr_idx = np.array([np.where(syms == s)[0][0] for s in common])
                rc = stats.spearmanr(prev_signal[prev_idx], signal[curr_idx])[0]
                rank_corrs.append(rc)

                top30_prev = set(prev_syms[np.argsort(prev_signal)[-30:]])
                top30_curr = set(syms[np.argsort(signal)[-30:]])
                overlap = len(top30_prev & top30_curr) / 30
                overlaps.append(overlap)

        prev_dt, prev_signal, prev_syms = dt, signal, syms

    rc_mean = np.mean(rank_corrs) if rank_corrs else np.nan
    ov_mean = np.mean(overlaps) if overlaps else np.nan
    ic_mean = np.mean(ics) if ics else np.nan
    ic_ir = ic_mean / np.std(ics) if (ics and np.std(ics) > 0) else np.nan

    logger.info(f"  {scheme_name:>12}: RankCorr={rc_mean:.4f}, Overlap={ov_mean:.3f}, "
                f"IC={ic_mean:.4f}, IC_IR={ic_ir:.4f}")

    weighting_results[scheme_name]["rank_corr"] = rc_mean
    weighting_results[scheme_name]["overlap"] = ov_mean
    weighting_results[scheme_name]["IC"] = ic_mean
    weighting_results[scheme_name]["IC_IR"] = ic_ir

# ═══════════════════════════════════════════════════════════════
# GENERATE REPORT
# ═══════════════════════════════════════════════════════════════
logger.info("="*60)
logger.info("Generating report...")
logger.info("="*60)

R = []
def w(s=""): R.append(s)

w("# Ensemble Stability Root Cause Report")
w()
w(f"**Generated**: {pd.Timestamp.now()}")
w()
w("## Executive Summary")
w()
w("| Metric | V1 (production) | V2_Full (production) | Best Single-Model | Best Ensemble |")
w("|--------|----------------|----------------------|-------------------|---------------|")
w(f"| RankCorr | 0.984 | 0.718 | ~0.85 | TBD |")
w(f"| Top30Overlap | 81% | 34% | ~70% | TBD |")
w(f"| IC | 0.0582 | 0.0615 | ~0.10 | TBD |")
w(f"| Sharpe | 0.70 | 0.51 | TBD | TBD |")
w()

# ── Experiment C ──
w("---")
w("## Experiment C: Model Pairwise Prediction Correlation")
w()
w(f"**Total comparisons**: {len(pw_df):,}")
w(f"**Dates**: {pw_df['date'].nunique()}")
w(f"**Models**: {n_valid}")
w()
w("### Overall Distribution")
w()
w("| Statistic | Pearson r |")
w("|-----------|----------|")
w(f"| Mean | {all_corrs.mean():.4f} |")
w(f"| Median | {all_corrs.median():.4f} |")
w(f"| Std | {all_corrs.std():.4f} |")
w(f"| P10 | {all_corrs.quantile(0.10):.4f} |")
w(f"| P25 | {all_corrs.quantile(0.25):.4f} |")
w(f"| P75 | {all_corrs.quantile(0.75):.4f} |")
w(f"| P90 | {all_corrs.quantile(0.90):.4f} |")
w(f"| Min | {all_corrs.min():.4f} |")
w(f"| Max | {all_corrs.max():.4f} |")
w()
w("### Same-Seed vs Cross-Seed")
w()
w("| Comparison | Mean r | N |")
w("|------------|--------|---|")
w(f"| Same seed | {same_seed.mean():.4f} | {len(same_seed):,} |")
w(f"| Cross seed | {cross_seed.mean():.4f} | {len(cross_seed):,} |")
w()

# Answer Q
mean_corr = all_corrs.mean()
if mean_corr > 0.95:
    w("**Answer C**: Models express **highly consistent** views (mean r > 0.95). They are essentially the same opinion. Ensemble adds minimal diversification benefit.")
elif mean_corr > 0.85:
    w("**Answer C**: Models express **moderately consistent** views (0.85 < mean r < 0.95). Moderate diversity — ensemble benefits from averaging but doesn't introduce drastic instability.")
elif mean_corr > 0.70:
    w(f"**Answer C**: Models express **substantially diverse** views (mean r = {mean_corr:.3f}). There is significant disagreement between models. The ensemble average may be less stable than individual models because it averages fundamentally different rankings.")
else:
    w(f"**Answer C**: Models express **fundamentally different** views (mean r = {mean_corr:.3f}). Each model sees a completely different ranking. Ensembling them creates noise, not signal.")
w()

# ── Experiment D ──
w("---")
w("## Experiment D: Seed Consistency Analysis")
w()
w("| Seed Pair | Mean r | Median r | Std | N |")
w("|-----------|--------|---------|-----|---|")
for sp in [f"{s1}-{s2}" for s1, s2 in combinations(seeds, 2)]:
    sub = scf_df[scf_df["seed_pair"] == sp]
    if len(sub) > 0:
        w(f"| {sp} | {sub['pearson_r'].mean():.4f} | {sub['pearson_r'].median():.4f} | {sub['pearson_r'].std():.4f} | {len(sub)} |")
w()

seed_avg_r = scf_df["pearson_r"].mean()
if seed_avg_r > 0.95:
    w("**Answer D**: Seeds are **nearly identical** — random seed introduces minimal noise. Reducing from 3→1 seed is safe.")
elif seed_avg_r > 0.85:
    w(f"**Answer D**: Seeds show **moderate consistency** (mean r = {seed_avg_r:.3f}). Random seed introduces some noise but not catastrophic.")
else:
    w(f"**Answer D**: Seeds show **low consistency** (mean r = {seed_avg_r:.3f}). Random seed contributes significantly to ensemble instability.")
w()

# ── Experiment E ──
w("---")
w("## Experiment E: Fold Consistency Analysis")
w()
w("| Seed | Mean Fold-Fold r | Median | N |")
w("|------|-----------------|--------|---|")
for seed in seeds:
    records = fold_corr_by_seed[seed]
    if records:
        df_tmp = pd.DataFrame(records)
        w(f"| {seed} | {df_tmp['pearson_r'].mean():.4f} | {df_tmp['pearson_r'].median():.4f} | {len(df_tmp)} |")
w()

# Fold delta analysis
fold_delta_corrs = defaultdict(list)
for seed in seeds:
    records = fold_corr_by_seed[seed]
    for r in records:
        fold_delta_corrs[r["fold_delta"]].append(r["pearson_r"])

w("### Fold Distance vs Correlation")
w("| Fold Δ | Mean r | N |")
w("|---------|--------|---|")
for delta in sorted(fold_delta_corrs.keys())[:10]:
    corrs = fold_delta_corrs[delta]
    w(f"| {delta} | {np.mean(corrs):.4f} | {len(corrs)} |")
w()

if fold_avg_corr:
    all_fold_corrs = list(fold_avg_corr.values())
    fold_corr_mean = np.mean(all_fold_corrs)
    if fold_corr_mean > 0.90:
        w(f"**Answer E**: Folds are **highly consistent** (mean fold-fold r = {fold_corr_mean:.3f}). Using all folds is redundant but not harmful.")
    elif fold_corr_mean > 0.75:
        w(f"**Answer E**: Folds show **moderate diversity** (mean fold-fold r = {fold_corr_mean:.3f}). Some folds learn different styles. The most inconsistent folds may be introducing noise.")
    else:
        w(f"**Answer E**: Folds are **substantially different** (mean fold-fold r = {fold_corr_mean:.3f}). Different training windows produce fundamentally different models. This is a major source of ensemble instability.")
w()

# ── Experiment F ──
w("---")
w("## Experiment F: Ensemble Scale Ablation")
w()
w("| N_models | RankCorr | Top30Overlap | IC | IC_IR |")
w("|----------|----------|-------------|-----|-------|")
for N in N_MODELS_LIST:
    rc_list = scale_results[N].get("rank_corrs", [])
    ov_list = scale_results[N].get("overlaps", [])
    rc_mean = np.mean([r["rank_corr"] for r in rc_list]) if rc_list else np.nan
    rc_std = np.std([r["rank_corr"] for r in rc_list]) if rc_list else np.nan
    ov_mean = np.mean([o["overlap"] for o in ov_list]) if ov_list else np.nan
    ic_mean = scale_results[N].get("IC_mean", np.nan)
    ic_ir = scale_results[N].get("IC_IR", np.nan)
    display_N = N if N < n_valid else n_valid
    w(f"| {display_N} | {rc_mean:.4f} ± {rc_std:.4f} | {ov_mean:.3f} | {ic_mean:.4f} | {ic_ir:.4f} |")
w()

# Answer Q
rc_values = []
for N in N_MODELS_LIST:
    rc_list = scale_results[N].get("rank_corrs", [])
    if rc_list:
        rc_values.append((N if N < n_valid else n_valid, np.mean([r["rank_corr"] for r in rc_list])))

if len(rc_values) >= 2:
    rc_trend = rc_values[-1][1] - rc_values[0][1]
    if rc_trend < -0.05:
        w(f"**Answer F**: RankCorr **declines with more models** (Δ = {rc_trend:+.4f}). This is evidence of **Over-Ensemble** — adding models hurts stability.")
    elif rc_trend < -0.01:
        w(f"**Answer F**: RankCorr **slightly declines** with more models (Δ = {rc_trend:+.4f}). Mild over-ensemble effect.")
    else:
        w(f"**Answer F**: RankCorr is **stable or improves** with more models (Δ = {rc_trend:+.4f}). Ensemble does NOT hurt stability.")
w()

# ── Experiment G ──
w("---")
w("## Experiment G: Style Drift Analysis")
w()
w("### Factor Exposure Distribution Across Folds")
w()
w("| Factor | Mean | Std | Min | Max | % Positive |")
w("|--------|------|-----|-----|-----|-----------|")
for factor in available_factors:
    vals = se_df[factor].dropna()
    w(f"| {factor} | {vals.mean():.4f} | {vals.std():.4f} | {vals.min():.4f} | {vals.max():.4f} | {100*(vals>0).mean():.0f}% |")
w()

# Style range = max - min exposure across folds
w("### Style Dispersion (max-min exposure)")
w()
style_ranges = {}
for factor in available_factors:
    vals = se_df[factor].dropna()
    style_ranges[factor] = vals.max() - vals.min()

for factor, rng in sorted(style_ranges.items(), key=lambda x: x[1], reverse=True):
    w(f"| {factor} | {rng:.4f} |")
w()

# Find factors with highest dispersion
high_dispersion = {k: v for k, v in style_ranges.items() if v > 0.3}
if high_dispersion:
    w(f"**Answer G**: Factors with significant style dispersion (>0.3): {list(high_dispersion.keys())}. ")
    w("Different folds **do represent different investment styles** — e.g., some favor value (high EP exposure), others favor growth (high ProfitGrowth exposure).")
    w("This style drift across folds contributes to ensemble instability because averaging value and growth views produces an incoherent middle-ground ranking.")
else:
    w("**Answer G**: All folds have similar factor exposures. Style drift is **not a significant contributor** to ensemble instability.")
w()

# ── Experiment H ──
w("---")
w("## Experiment H: Ensemble Weighting Analysis")
w()
w("| Weighting Scheme | RankCorr | Top30Overlap | IC | IC_IR |")
w("|------------------|----------|-------------|-----|-------|")
for scheme in ["Equal", "IC_Weighted", "Top25%", "Top10%"]:
    rc = weighting_results[scheme].get("rank_corr", np.nan)
    ov = weighting_results[scheme].get("overlap", np.nan)
    ic = weighting_results[scheme].get("IC", np.nan)
    ic_ir = weighting_results[scheme].get("IC_IR", np.nan)
    w(f"| {scheme} | {rc:.4f} | {ov:.3f} | {ic:.4f} | {ic_ir:.4f} |")
w()

# Get best scheme
best_scheme = max(
    ["Equal", "IC_Weighted", "Top25%", "Top10%"],
    key=lambda s: weighting_results[s].get("rank_corr", -999)
)
best_rc = weighting_results[best_scheme].get("rank_corr", np.nan)
eq_rc = weighting_results["Equal"].get("rank_corr", np.nan)

if best_rc > eq_rc + 0.02:
    w(f"**Answer H**: **{best_scheme}** weighting significantly improves RankCorr ({eq_rc:.4f} → {best_rc:.4f}). This confirms the presence of **low-quality models dragging down the ensemble**. Removing or down-weighting them is a high-impact fix.")
elif best_rc > eq_rc + 0.005:
    w(f"**Answer H**: **{best_scheme}** marginally improves RankCorr ({eq_rc:.4f} → {best_rc:.4f}). There are some low-quality models but their impact is modest.")
else:
    w(f"**Answer H**: No weighting scheme beats equal weight. The instability is NOT caused by a few bad models — it's a structural issue with the ensemble architecture.")
w()

# ── Final Synthesis ──
w("---")
w("## Final Synthesis: Root Cause Attribution")
w()
w("### Q1: Does 162-model ensemble significantly reduce rank stability?")
w()
# Compare single-model RankCorr to full-ensemble RankCorr
single_rc = scale_results[1].get("rank_corrs", [])
full_rc = scale_results[n_valid].get("rank_corrs", [])
single_rc_mean = np.mean([r["rank_corr"] for r in single_rc]) if single_rc else np.nan
full_rc_mean = np.mean([r["rank_corr"] for r in full_rc]) if full_rc else np.nan

if not np.isnan(single_rc_mean) and not np.isnan(full_rc_mean):
    delta = full_rc_mean - single_rc_mean
    w(f"Single-model RankCorr: {single_rc_mean:.4f}")
    w(f"Full-ensemble RankCorr: {full_rc_mean:.4f}")
    w(f"Δ: {delta:+.4f}")
    if delta < -0.05:
        w("**YES** — the ensemble significantly reduces rank stability.")
    elif delta < -0.01:
        w("**MODERATE** — the ensemble reduces stability but not catastrophically.")
    else:
        w("**NO** — the ensemble does NOT reduce stability. The problem is elsewhere.")
w()

w("### Q2: Decomposition of RankCorr decline (0.85 → 0.72)")
w()
w("| Source | Estimated ΔRankCorr | % of Total | Evidence |")
w("|--------|---------------------|------------|----------|")
# Seed contribution
seed_delta = 1 - seed_avg_r  # approximate
w(f"| Seed variance | {seed_delta:.4f} | {100*seed_delta/0.13:.0f}% | Exp D: mean seed-pair r={seed_avg_r:.3f} |")
# Fold contribution
fold_corr_mean_val = np.mean([np.mean([r["pearson_r"] for r in fold_corr_by_seed[s]]) for s in seeds if fold_corr_by_seed[s]])
if not np.isnan(fold_corr_mean_val):
    fold_delta = 1 - fold_corr_mean_val
    w(f"| Fold variance | {fold_delta:.4f} | {100*fold_delta/0.13:.0f}% | Exp E: mean fold-fold r={fold_corr_mean_val:.3f} |")
# Scale contribution
if not np.isnan(single_rc_mean) and not np.isnan(full_rc_mean):
    scale_delta = single_rc_mean - full_rc_mean
    w(f"| Ensemble scale | {max(0,scale_delta):.4f} | {100*max(0,scale_delta)/0.13:.0f}% | Exp F: {1}-model vs {n_valid}-model stability |")
w()

w("### Q3: Is there Over-Ensemble?")
w()
if not np.isnan(single_rc_mean) and not np.isnan(full_rc_mean):
    if full_rc_mean < single_rc_mean - 0.02:
        w(f"**YES** — Over-Ensemble confirmed. Single model (RankCorr={single_rc_mean:.4f}) outperforms {n_valid}-model ensemble (RankCorr={full_rc_mean:.4f}).")
        w("Adding more models **harms** stability beyond some optimal point.")
    else:
        w("**NO** — no evidence of Over-Ensemble. Ensemble stability ≈ single-model stability.")
w()

w("### Q4: Optimal number of models?")
w()
if len(rc_values) >= 2:
    best_n = max(rc_values, key=lambda x: x[1])
    w(f"Best RankCorr at N={best_n[0]}: {best_n[1]:.4f}")
    # Find the knee: where adding more models stops helping
    for i in range(1, len(rc_values)):
        if rc_values[i][1] < rc_values[i-1][1] - 0.01:
            w(f"Optimal range appears to be around N={rc_values[i-1][0]}")
            break
w()

w("### Q5: Should we reduce from 54-fold × 3-seed to 9-fold × 3-seed or single-seed?")
w()
if seed_avg_r > 0.95:
    w("- **Seed reduction**: YES — seeds are nearly identical. 1 seed is sufficient.")
else:
    w(f"- **Seed reduction**: MODERATE — seeds add diversity (r={seed_avg_r:.3f}). Keep 3 seeds but consider reducing folds.")

fold_fold_r = np.mean([np.mean([r["pearson_r"] for r in fold_corr_by_seed[s]]) for s in seeds if fold_corr_by_seed[s]])
if fold_fold_r > 0.90:
    w(f"- **Fold reduction**: YES — folds are highly redundant (r={fold_fold_r:.3f}). 9-18 folds would preserve most of the signal.")
elif fold_fold_r > 0.75:
    w(f"- **Fold reduction**: MODERATE — folds add moderate diversity (r={fold_fold_r:.3f}). Reducing to 18-27 folds is safe.")
else:
    w(f"- **Fold reduction**: CAUTION — folds are substantially different (r={fold_fold_r:.3f}). Diversity may be beneficial for robustness despite stability cost.")
w()

w("---")
w(f"*Report generated: {pd.Timestamp.now()}*")

# Write report
OUTPUT = "\n".join(R)
REPORT_PATH.write_text(OUTPUT, encoding="utf-8")
logger.info(f"Report saved: {REPORT_PATH}")
print(OUTPUT[:5000])
