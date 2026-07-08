"""
Root Cause Ablation: Controlled experiments to isolate rank stability collapse.
Experiment A: colsample_bytree ∈ {0.50, 0.70, 0.90, 1.00}
Experiment B: GS Orthogonalization ON vs OFF
Experiment C: Universe ∈ {300 stocks, Full CSI800}
All use seed=42, single model, identical data & hyperparameters otherwise.
"""
import warnings, logging, time
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ablation")

OUT = Path("output")
PANEL = OUT / "training_panel_v3_full.parquet"
REPORT = OUT / "root_cause_ablation.md"

# ── Load panel ──
logger.info("Loading panel...")
panel = pd.read_parquet(PANEL)
panel["date"] = pd.to_datetime(panel["date"])
logger.info(f"Panel: {len(panel)} rows, {panel['date'].nunique()} dates")

# Prepare features
nz = sorted([c for c in panel.columns if c.endswith("_neutral_z")])
fnames = [c.replace("_neutral_z", "") for c in nz]
logger.info(f"Features: {len(fnames)}")

# ── GS orthogonalization ──
def apply_gs_ortho(df, nz_cols):
    """Gram-Schmidt orthogonalization by IC_IR ranking."""
    from scipy.stats import spearmanr
    # Compute forward return
    df = df.copy()
    df = df.sort_values(["symbol", "date"])
    df["fw_ret"] = df.groupby("symbol")["收盘"].transform(lambda x: x.shift(-1) / x - 1.0)

    # IC_IR ranking
    ic_irs = {}
    for c in nz_cols:
        ics = []
        for dt, g in df.groupby("date"):
            valid = g[[c, "fw_ret"]].dropna()
            if len(valid) >= 30:
                corr, _ = spearmanr(valid[c], valid["fw_ret"])
                if not np.isnan(corr): ics.append(abs(corr))
        ic_irs[c] = np.mean(ics) / max(np.std(ics), 1e-9) if len(ics) > 1 else 0

    ranked = sorted(ic_irs.items(), key=lambda x: -x[1])
    ranked_cols = [c for c, _ in ranked]

    # Apply GS per date
    for dt, idx in df.groupby("date").groups.items():
        X = df.loc[idx, ranked_cols].values.astype(float)
        # Fill NaN with column mean
        for j in range(X.shape[1]):
            col_mean = np.nanmean(X[:, j])
            X[:, j] = np.where(np.isnan(X[:, j]), col_mean, X[:, j])

        Q = np.zeros_like(X)
        for j in range(X.shape[1]):
            v = X[:, j].copy()
            for k in range(j):
                proj = np.dot(v, Q[:, k]) / max(np.dot(Q[:, k], Q[:, k]), 1e-12)
                v -= proj * Q[:, k]
            v_std = np.std(v)
            if v_std > 1e-12: v /= v_std
            Q[:, j] = v
        df.loc[idx, ranked_cols] = Q

    return df, ranked_cols

# ── Rank features per date ──
def rank_features(df, cols, date_col="date"):
    ranked = df.copy()
    for c in cols:
        rc = f"{c}_rank"
        ranked[rc] = ranked.groupby(date_col)[c].rank(pct=True, na_option="bottom").fillna(0.5)
    return ranked, [f"{c}_rank" for c in cols]

# ── Metrics computation ──
def compute_metrics(signals_df, panel_df):
    """Compute IC, Rank Corr, Top30 Overlap, Turnover from prediction signals."""
    # IC
    merged = signals_df.merge(
        panel_df[["date", "symbol"]], on=["date", "symbol"], how="inner"
    )
    # Compute forward return
    merged = merged.sort_values(["symbol", "date"])
    merged["fw_ret"] = merged.groupby("symbol")["收盘"].transform(
        lambda x: x.shift(-1) / x - 1.0) if "收盘" in panel_df.columns else np.nan

    dates = sorted(merged["date"].unique())

    # Rank Corr
    rank_corrs = []
    for i in range(1, len(dates)):
        prev = merged[merged["date"] == dates[i-1]].set_index("symbol")["prediction"]
        curr = merged[merged["date"] == dates[i]].set_index("symbol")["prediction"]
        common = prev.index.intersection(curr.index)
        if len(common) >= 30:
            rank_corrs.append(prev[common].corr(curr[common], method="spearman"))

    # Top30 Overlap
    top30_overlaps = []
    dummy_turnovers = []
    for i in range(1, len(dates)):
        prev = merged[merged["date"] == dates[i-1]].nlargest(30, "prediction")
        curr = merged[merged["date"] == dates[i]].nlargest(30, "prediction")
        prev_set = set(prev["symbol"])
        curr_set = set(curr["symbol"])
        overlap = len(prev_set & curr_set) / max(len(prev_set | curr_set), 1)
        top30_overlaps.append(overlap)
        # Dummy turnover: fraction replaced
        dummy_turnovers.append(1 - overlap)

    # IC
    ics = []
    for dt in dates:
        d = merged[merged["date"] == dt].dropna(subset=["prediction", "fw_ret"])
        if len(d) >= 30:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(d["prediction"], d["fw_ret"])
            if not np.isnan(ic): ics.append(ic)

    return {
        "mean_rank_corr": np.mean(rank_corrs) if rank_corrs else np.nan,
        "std_rank_corr": np.std(rank_corrs) if rank_corrs else np.nan,
        "mean_top30_overlap": np.mean(top30_overlaps) if top30_overlaps else np.nan,
        "implied_turnover": np.mean(dummy_turnovers) if dummy_turnovers else np.nan,
        "mean_ic": np.mean(ics) if ics else np.nan,
        "ic_ir": np.mean(ics) / max(np.std(ics), 1e-9) if ics else np.nan,
        "n_dates": len(dates),
    }

# ── Train single model ──
def train_one(panel_ready, rank_cols, colsample, seed=42):
    """Train ONE LightGBM model on latest fold window."""
    import lightgbm as lgb

    df = panel_ready.copy()
    df = df.sort_values(["symbol", "date"])

    # Label: cross-sectional rank of forward_return_1m
    df["fw_ret"] = df.groupby("symbol")["收盘"].transform(lambda x: x.shift(-1) / x - 1.0)
    df["label"] = df.groupby("date")["fw_ret"].rank(pct=True, na_option="bottom").fillna(0.5)
    df = df.dropna(subset=["label"])

    # Ensure rank_cols don't already have _rank suffix added twice
    feature_cols = [c for c in rank_cols if c in df.columns]

    # Train/val/test split: last 3 folds
    dates = sorted(df["date"].unique())
    if len(dates) < 44:
        return None, None

    # Use last window: train=dates[:-2], val=dates[-2], test=dates[-1]
    train_dates = dates[:-2]
    val_date = dates[-2]
    test_date = dates[-1]

    train_df = df[df["date"].isin(train_dates)]
    val_df = df[df["date"] == val_date]
    test_df = df[df["date"] == test_date]

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df["label"].values.astype(float)
    X_val = val_df[feature_cols].values.astype(float)
    y_val = val_df["label"].values.astype(float)
    X_test = test_df[feature_cols].values.astype(float)

    params = {
        "objective": "regression", "metric": "l2", "boosting": "gbdt",
        "num_leaves": 24, "max_depth": 4, "learning_rate": 0.02,
        "n_estimators": 2000, "subsample": 1.0,
        "colsample_bytree": colsample,
        "subsample_freq": 1, "min_child_samples": 100,
        "reg_alpha": 0.10, "reg_lambda": 0.10,
        "early_stopping_rounds": 50, "verbose": -1, "n_jobs": -1,
        "random_state": seed,
    }

    train_ds = lgb.Dataset(X_train, label=y_train)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

    model = lgb.train(params, train_ds, num_boost_round=2000,
                      valid_sets=[val_ds], valid_names=["val"],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)])

    # Predict on ALL data
    X_all = df[feature_cols].values.astype(float)
    preds = model.predict(X_all)
    result = df[["date", "symbol"]].copy()
    result["prediction"] = preds
    result["收盘"] = df["收盘"]

    return result, model

# ═══════════════════════════════════════════════
# EXPERIMENT A: colsample ablation
# ═══════════════════════════════════════════════
logger.info("=" * 60)
logger.info("EXPERIMENT A: colsample_bytree ablation")
logger.info("=" * 60)

# Prepare baseline panel (with GS ortho ON)
panel_gs, _ = apply_gs_ortho(panel.copy(), nz)
panel_gs_ranked, rank_cols = rank_features(panel_gs, nz)

expA_results = []
for cs in [0.50, 0.70, 0.90, 1.00]:
    logger.info(f"Training colsample={cs}...")
    t0 = time.perf_counter()
    signals, model = train_one(panel_gs_ranked, rank_cols, colsample=cs)
    elapsed = time.perf_counter() - t0

    if signals is None:
        logger.warning(f"  colsample={cs}: training failed")
        continue

    metrics = compute_metrics(signals, panel_gs)
    metrics["colsample"] = cs
    metrics["train_time_s"] = elapsed
    expA_results.append(metrics)

    logger.info(f"  colsample={cs}: RankCorr={metrics['mean_rank_corr']:.4f}, "
                f"Overlap={metrics['mean_top30_overlap']:.1%}, "
                f"IC={metrics['mean_ic']:.4f}, IC_IR={metrics['ic_ir']:.4f}, "
                f"TO~={metrics['implied_turnover']:.1%}")

# ═══════════════════════════════════════════════
# EXPERIMENT B: GS Orthogonalization ablation
# ═══════════════════════════════════════════════
logger.info("=" * 60)
logger.info("EXPERIMENT B: GS Orthogonalization ON vs OFF")
logger.info("=" * 60)

# GS OFF: just z-score → rank (no GS)
panel_no_gs = panel.copy()
panel_no_gs_ranked, rank_cols_nogs = rank_features(panel_no_gs, nz)

for label, pdata, rcols in [("GS_ON", panel_gs_ranked, rank_cols),
                              ("GS_OFF", panel_no_gs_ranked, rank_cols_nogs)]:
    logger.info(f"Training {label}...")
    t0 = time.perf_counter()
    signals, model = train_one(pdata, rcols, colsample=0.50)
    elapsed = time.perf_counter() - t0

    if signals is None:
        logger.warning(f"  {label}: training failed")
        continue

    metrics = compute_metrics(signals, panel.copy())
    metrics["gs_mode"] = label
    metrics["colsample"] = 0.50
    metrics["train_time_s"] = elapsed
    # Store separately
    if label == "GS_ON":
        expB_gs_on = metrics
    else:
        expB_gs_off = metrics

    logger.info(f"  {label}: RankCorr={metrics['mean_rank_corr']:.4f}, "
                f"Overlap={metrics['mean_top30_overlap']:.1%}, "
                f"IC={metrics['mean_ic']:.4f}")

# ═══════════════════════════════════════════════
# EXPERIMENT C: Universe size ablation
# ═══════════════════════════════════════════════
logger.info("=" * 60)
logger.info("EXPERIMENT C: Universe size (300 vs Full)")
logger.info("=" * 60)

# Sample 300 stocks per date from full panel
np.random.seed(42)
panel_300 = panel_gs.copy()
dates_all = sorted(panel_300["date"].unique())
sampled_frames = []
for dt in dates_all:
    dt_data = panel_300[panel_300["date"] == dt]
    if len(dt_data) > 300:
        sampled = dt_data.sample(300, random_state=42)
    else:
        sampled = dt_data
    sampled_frames.append(sampled)
panel_300 = pd.concat(sampled_frames, ignore_index=True)
logger.info(f"Sampled panel: {len(panel_300)} rows, ~300 stocks/date")

# Rank the sampled panel
panel_300_ranked, rank_cols_300 = rank_features(panel_300, nz)

for label, pdata, rcols in [("Full_CSI800", panel_gs_ranked, rank_cols),
                              ("Sampled_300", panel_300_ranked, rank_cols_300)]:
    logger.info(f"Training {label}...")
    t0 = time.perf_counter()
    signals, model = train_one(pdata, rcols, colsample=0.50)
    elapsed = time.perf_counter() - t0

    if signals is None:
        logger.warning(f"  {label}: training failed")
        continue

    metrics = compute_metrics(signals, pdata)
    metrics["universe"] = label
    metrics["colsample"] = 0.50
    metrics["train_time_s"] = elapsed
    if label == "Full_CSI800":
        expC_full = metrics
    else:
        expC_300 = metrics

    logger.info(f"  {label}: RankCorr={metrics['mean_rank_corr']:.4f}, "
                f"Overlap={metrics['mean_top30_overlap']:.1%}, "
                f"IC={metrics['mean_ic']:.4f}")

# ═══════════════════════════════════════════
# GENERATE REPORT
# ═══════════════════════════════════════════
logger.info("Generating report...")
R = []
def w(s=""): R.append(s)

w("# Root Cause Ablation Report")
w()
w("## 摘要")
w()
w("| 模型 | RankCorr | Top30Overlap | IC | IC_IR | ImpliedTO |")
w("|------|----------|-------------|-----|-------|-----------|")
w("| V1 (production) | 0.984 | 81% | 0.0582 | 0.539 | 14.5% |")
w("| V2_Full (production) | 0.718 | 34% | 0.0615 | 0.538 | 38.3% |")
w()

# Experiment A
w("---")
w("## Experiment A: colsample_bytree")
w()
w("| colsample | RankCorr | Top30Overlap | IC | IC_IR | ImpliedTO | TrainTime |")
w("|-----------|----------|-------------|-----|-------|-----------|-----------|")
for r in expA_results:
    w(f"| {r['colsample']:.2f} | {r['mean_rank_corr']:.4f} | {r['mean_top30_overlap']:.1%} | "
      f"{r['mean_ic']:.4f} | {r['ic_ir']:.4f} | {r['implied_turnover']:.1%} | {r['train_time_s']:.0f}s |")
w()

# Find sensitivity
if len(expA_results) >= 2:
    cs_low = min(expA_results, key=lambda x: x["colsample"])
    cs_high = max(expA_results, key=lambda x: x["colsample"])
    rc_diff = cs_high["mean_rank_corr"] - cs_low["mean_rank_corr"]
    to_diff = cs_low["implied_turnover"] - cs_high["implied_turnover"]
    w(f"**colsample 0.50→{cs_high['colsample']:.0f} 效果**: RankCorr {rc_diff:+.4f}, ImpliedTO {to_diff:+.1%}")
    w()

# Experiment B
w("---")
w("## Experiment B: GS Orthogonalization")
w()
w("| GS Mode | RankCorr | Top30Overlap | IC | IC_IR | ImpliedTO |")
w("|---------|----------|-------------|-----|-------|-----------|")
for r in [expB_gs_on, expB_gs_off]:
    w(f"| {r['gs_mode']} | {r['mean_rank_corr']:.4f} | {r['mean_top30_overlap']:.1%} | "
      f"{r['mean_ic']:.4f} | {r['ic_ir']:.4f} | {r['implied_turnover']:.1%} |")
w()
gs_diff_rc = expB_gs_on["mean_rank_corr"] - expB_gs_off["mean_rank_corr"] if 'expB_gs_off' in dir() else 0
gs_diff_to = expB_gs_on["implied_turnover"] - expB_gs_off["implied_turnover"] if 'expB_gs_off' in dir() else 0
w(f"**GS Effect**: RankCorr {gs_diff_rc:+.4f}, ImpliedTO {gs_diff_to:+.1%}")
w()

# Experiment C
w("---")
w("## Experiment C: Universe Size")
w()
w("| Universe | RankCorr | Top30Overlap | IC | IC_IR | ImpliedTO |")
w("|----------|----------|-------------|-----|-------|-----------|")
for r in [expC_full, expC_300]:
    w(f"| {r['universe']} | {r['mean_rank_corr']:.4f} | {r['mean_top30_overlap']:.1%} | "
      f"{r['mean_ic']:.4f} | {r['ic_ir']:.4f} | {r['implied_turnover']:.1%} |")
w()
uni_diff_rc = expC_full["mean_rank_corr"] - expC_300["mean_rank_corr"] if 'expC_300' in dir() else 0
uni_diff_to = expC_full["implied_turnover"] - expC_300["implied_turnover"] if 'expC_300' in dir() else 0
w(f"**Universe Effect**: RankCorr {uni_diff_rc:+.4f}, ImpliedTO {uni_diff_to:+.1%}")
w()

# ── Attribution ──
w("---")
w("## Root Cause Attribution")
w()
w("### RankCorr下降 (0.984→0.718, Δ=-0.266) 归因")
w()

contribs = []
if len(expA_results) >= 2:
    cs_contrib = abs(cs_high["mean_rank_corr"] - cs_low["mean_rank_corr"])
    contribs.append(("colsample_bytree", cs_contrib))
if 'expB_gs_off' in dir():
    gs_contrib = abs(expB_gs_on["mean_rank_corr"] - expB_gs_off["mean_rank_corr"])
    contribs.append(("GS正交化", gs_contrib))
if 'expC_300' in dir():
    uni_contrib = abs(expC_full["mean_rank_corr"] - expC_300["mean_rank_corr"])
    contribs.append(("Universe扩大", uni_contrib))

total = sum(c[1] for c in contribs) if contribs else 1
contribs.sort(key=lambda x: -x[1])

w("| Rank | Factor | Absolute ΔRankCorr | % Contribution |")
w("|------|--------|-------------------|----------------|")
for i, (name, val) in enumerate(contribs):
    w(f"| {i+1} | {name} | {val:.4f} | {val/total*100:.1f}% |")
w()

# Unattributed portion
w(f"| — | Unexplained (interaction effects) | {0.266-total:.4f} | {(0.266-total)/0.266*100:.1f}% |")
w()

# ── Answers ──
w("---")
w("## Questions Answered")
w()

# Q1
w("### Q1: RankCorr下降贡献排序")
w()
for i, (name, val) in enumerate(contribs):
    pct = val/total*100
    if pct > 30:
        w(f"{i+1}. **{name}**: {val:.3f} ({pct:.0f}%) — **主要驱动因素**")
    elif pct > 10:
        w(f"{i+1}. **{name}**: {val:.3f} ({pct:.0f}%) — 显著贡献")
    else:
        w(f"{i+1}. {name}: {val:.3f} ({pct:.0f}%) — 次要贡献")
w()

# Q2
w("### Q2: colsample贡献占比")
if len(expA_results) >= 2:
    pct = cs_contrib / 0.266 * 100
    w(f"colsample 从 0.50→{cs_high['colsample']:.0f} 可恢复 **{cs_contrib:.3f}** RankCorr (占Δ的 {pct:.0f}%)")
w()

# Q3
w("### Q3: GS正交化贡献占比")
if 'expB_gs_off' in dir():
    pct = gs_contrib / 0.266 * 100
    w(f"GS正交化贡献 **{gs_contrib:.3f}** RankCorr (占Δ的 {pct:.0f}%)")
w()

# Q4
w("### Q4: Universe扩大贡献占比")
if 'expC_300' in dir():
    pct = uni_contrib / 0.266 * 100
    w(f"Universe扩大贡献 **{uni_contrib:.3f}** RankCorr (占Δ的 {pct:.0f}%)")
w()

# Q5
w("### Q5: 最值得优先修复的变量")
w()
if contribs:
    top_factor = contribs[0]
    w(f"**{top_factor[0]}** — 贡献了 {top_factor[1]/total*100:.0f}% 的 RankCorr 下降, 且修复成本最低(仅需修改一个参数)")
w()

w("---")
w(f"*报告生成: {pd.Timestamp.now()}*")

OUTPUT = "\n".join(R)
REPORT.write_text(OUTPUT, encoding="utf-8")
logger.info(f"Saved: {REPORT}")
print(OUTPUT)
