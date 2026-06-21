"""
V1.5 Experiment Matrix — 6-Model Training Script.

Implements the minimal experiment matrix designed in the V1.5 Alpha Framework.
Trains exactly 6 models (M0-M5) using walk-forward CV with a single fold (fold=-1)
and single seed (seed=42), per the audit's finding that seeds are redundant.

Models:
  M0: V2 Baseline     — GS=ON,  colsample=0.50, V2 factors,         no monotonicity
  M1: V1.5-Core       — GS=OFF, colsample=0.75, SR factors + BP,    no monotonicity
  M2: V1.5-Mono       — GS=OFF, colsample=0.75, SR factors + BP,    monotonicity ON
  M3: V1.5-GS_Soft    — GS=ON(max_corr=0.95), colsample=0.75, SR factors + BP
  M4: V1.5-AltGrowth  — GS=OFF, colsample=0.75, EPS_YoY replaces PG, no monotonicity
  M5: V1.5-Full       — GS=OFF, colsample=0.75, ALL factors,         monotonicity ON

Output per model:
  - output/production_models_v15/{model_name}/   (saved models)
  - output/production_models_v15/{model_name}_oos.parquet  (OOS predictions)

Usage:
  python run_v15_experiment.py                      # Train all 6 models
  python run_v15_experiment.py --models M0 M1       # Train specific models
  python run_v15_experiment.py --dry-run            # Print configs only
  python run_v15_experiment.py --quick              # Single fold, single seed
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v15_experiment")

OUTPUT_DIR = Path("output")
V15_PANEL_PATH = OUTPUT_DIR / "training_panel_v15_sr.parquet"
V2_PANEL_PATH = OUTPUT_DIR / "training_panel_v3_full.parquet"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "production_models_v15"


# ═══════════════════════════════════════════════════════════
# Experiment Configuration
# ═══════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """Configuration for one V1.5 experiment model."""
    name: str
    description: str
    panel_path: Path  # Which panel to use

    # Feature selection
    feature_neutral_z: list[str]  # _neutral_z columns to use

    # GS settings
    gs_enabled: bool = False
    gs_max_correlation: float = 0.85

    # LightGBM overrides
    colsample_bytree: float = 0.70
    num_leaves: int = 24
    max_depth: int = 4
    learning_rate: float = 0.02
    n_estimators: int = 2000
    subsample: float = 1.0
    min_child_samples: int = 100
    reg_alpha: float = 0.10
    reg_lambda: float = 0.10
    early_stopping_rounds: int = 50

    # Monotonicity constraints: factor_name -> +1 or -1
    # Applied to the _neutral_z columns, mapped to _rank columns at training time
    monotone_constraints: dict[str, int] = field(default_factory=dict)

    # Ensemble
    seeds: list[int] = field(default_factory=lambda: [42])  # 1 seed per audit
    lambda_turnover: float = 2.0

    # Walk-forward
    train_months: int = 36
    val_months: int = 6
    test_months: int = 1

    def to_lgb_params(self) -> dict:
        return {
            "objective": "regression",
            "metric": "l2",
            "boosting": "gbdt",
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "subsample_freq": 1,
            "min_child_samples": self.min_child_samples,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "verbose": -1,
            "n_jobs": -1,
        }


# ── M0: V2 Baseline ──
M0_V2_BASELINE = ExperimentConfig(
    name="M0_V2_Baseline",
    description="V2 Baseline: GS=ON, colsample=0.50, V2 factors, no BP, no SR",
    panel_path=V2_PANEL_PATH,
    feature_neutral_z=[
        "Mom_1M_neutral_z", "Mom_3M_neutral_z", "Mom_6M_neutral_z", "Mom_12M_1M_neutral_z",
        "Vol_20D_neutral_z", "Vol_60D_neutral_z", "Beta_neutral_z",
        "EP_neutral_z",  # BP excluded (deleted by GS in V2)
        "ROE_neutral_z", "Debt_Ratio_neutral_z", "Net_Profit_Margin_neutral_z",
        "RevGrowth_YoY_neutral_z", "ProfitGrowth_YoY_neutral_z",
        "VolChg_20D_neutral_z", "PriceDev_20D_neutral_z",
    ],
    gs_enabled=True,
    gs_max_correlation=0.85,
    colsample_bytree=0.50,
)

# ── M1: V1.5-Core ──
M1_V15_CORE = ExperimentConfig(
    name="M1_V15_Core",
    description="V1.5 Core: GS=OFF, colsample=0.75, SR factors + BP restored, no monotonicity",
    panel_path=V15_PANEL_PATH,
    feature_neutral_z=[
        # Stable (original)
        "Mom_1M_neutral_z", "Mom_3M_neutral_z", "Mom_6M_neutral_z", "Mom_12M_1M_neutral_z",
        "Vol_20D_neutral_z", "Vol_60D_neutral_z", "Beta_neutral_z",
        "Debt_Ratio_neutral_z", "Net_Profit_Margin_neutral_z",
        "VolChg_20D_neutral_z", "PriceDev_20D_neutral_z",
        # Value (original + BP restored)
        "EP_neutral_z", "BP_neutral_z",
        # Sector-relative quality/growth (replaces original)
        "SR_ROE_neutral_z",
        "SR_ProfitGrowth_YoY_neutral_z",
        "SR_RevGrowth_YoY_neutral_z",
    ],
    gs_enabled=False,
    colsample_bytree=0.75,
    learning_rate=0.05,
)

# ── M2: V1.5-Mono ──
M2_V15_MONO = ExperimentConfig(
    name="M2_V15_Mono",
    description="V1.5 + Monotonicity: GS=OFF, colsample=0.75, SR factors + BP, PG/ROE/EP forced positive",
    panel_path=V15_PANEL_PATH,
    feature_neutral_z=M1_V15_CORE.feature_neutral_z,
    gs_enabled=False,
    colsample_bytree=0.75,
    learning_rate=0.05,
    monotone_constraints={
        "SR_ProfitGrowth_YoY_neutral_z": +1,
        "SR_ROE_neutral_z": +1,
        "EP_neutral_z": +1,
    },
)

# ── M3: V1.5-GS_Soft ──
M3_V15_GS_SOFT = ExperimentConfig(
    name="M3_V15_GS_Soft",
    description="V1.5 + Soft GS: GS=ON(max_corr=0.95), colsample=0.75, SR factors + BP",
    panel_path=V15_PANEL_PATH,
    feature_neutral_z=M1_V15_CORE.feature_neutral_z,
    gs_enabled=True,
    gs_max_correlation=0.95,
    colsample_bytree=0.75,
    learning_rate=0.05,
)

# ── M4: V1.5-AltGrowth ──
M4_V15_ALT_GROWTH = ExperimentConfig(
    name="M4_V15_AltGrowth",
    description="V1.5 + Alt Growth: GS=OFF, colsample=0.75, EPS_YoY+ROE_Stability replace PG/RevGrowth",
    panel_path=V15_PANEL_PATH,
    feature_neutral_z=[
        # Stable
        "Mom_1M_neutral_z", "Mom_3M_neutral_z", "Mom_6M_neutral_z", "Mom_12M_1M_neutral_z",
        "Vol_20D_neutral_z", "Vol_60D_neutral_z", "Beta_neutral_z",
        "Debt_Ratio_neutral_z", "Net_Profit_Margin_neutral_z",
        "VolChg_20D_neutral_z", "PriceDev_20D_neutral_z",
        # Value
        "EP_neutral_z", "BP_neutral_z",
        # Sector-relative ROE
        "SR_ROE_neutral_z",
        # New quality/growth factors (REPLACE PG and RevGrowth)
        "EPS_YoY_neutral_z",
        "ROE_Stability_neutral_z",
        # Keep SR_RevGrowth for breadth
        "SR_RevGrowth_YoY_neutral_z",
    ],
    gs_enabled=False,
    colsample_bytree=0.75,
    learning_rate=0.05,
)

# ── M5: V1.5-Full ──
M5_V15_FULL = ExperimentConfig(
    name="M5_V15_Full",
    description="V1.5 Full: GS=OFF, colsample=0.75, ALL factors + monotonicity + alt growth",
    panel_path=V15_PANEL_PATH,
    feature_neutral_z=[
        # Stable
        "Mom_1M_neutral_z", "Mom_3M_neutral_z", "Mom_6M_neutral_z", "Mom_12M_1M_neutral_z",
        "Vol_20D_neutral_z", "Vol_60D_neutral_z", "Beta_neutral_z",
        "Debt_Ratio_neutral_z", "Net_Profit_Margin_neutral_z",
        "VolChg_20D_neutral_z", "PriceDev_20D_neutral_z",
        # Value
        "EP_neutral_z", "BP_neutral_z",
        # Sector-relative quality/growth
        "SR_ROE_neutral_z",
        "SR_ProfitGrowth_YoY_neutral_z",
        "SR_RevGrowth_YoY_neutral_z",
        # Alternative growth
        "EPS_YoY_neutral_z",
        "ROE_Stability_neutral_z",
        # Alt-data slot (NaN-filled; GS=OFF ensures not silently killed)
        # "SR_xhs_buzz_neutral_z",  # TODO: uncomment when data available
    ],
    gs_enabled=False,
    colsample_bytree=0.75,
    learning_rate=0.05,
    monotone_constraints={
        "SR_ProfitGrowth_YoY_neutral_z": +1,
        "EPS_YoY_neutral_z": +1,
        "SR_ROE_neutral_z": +1,
        "EP_neutral_z": +1,
    },
)

# ── Master config map ──
ALL_EXPERIMENTS: dict[str, ExperimentConfig] = {
    "M0": M0_V2_BASELINE,
    "M1": M1_V15_CORE,
    "M2": M2_V15_MONO,
    "M3": M3_V15_GS_SOFT,
    "M4": M4_V15_ALT_GROWTH,
    "M5": M5_V15_FULL,
}


# ═══════════════════════════════════════════════════════════
# Panel Filtering: select only required _neutral_z columns
# ═══════════════════════════════════════════════════════════

def prepare_panel_for_config(
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """
    Load panel and filter to only the _neutral_z columns specified in config.
    Also keeps date, symbol, and close/收盘 for label computation.
    """
    panel_path = cfg.panel_path
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Panel not found: {panel_path}. "
            f"For V1.5 panel, run run_v15_rebuild_panel.py first. "
            f"For V2 panel, run run_phaseb_rebuild_panel.py first."
        )

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)

    # Determine required columns
    required = ["date", "symbol"] + cfg.feature_neutral_z

    # Find close column for label computation
    close_col = None
    for candidate in ["收盘", "close"]:
        if candidate in panel.columns:
            close_col = candidate
            required.append(candidate)
            break

    # Check missing columns
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(
            f"Config '{cfg.name}' requires columns not in panel:\n"
            f"  Missing: {missing}\n"
            f"  Available _neutral_z: "
            f"{[c for c in panel.columns if c.endswith('_neutral_z')]}"
        )

    # Filter
    available_cols = [c for c in required if c in panel.columns]
    result = panel[available_cols].copy()

    logger.info(
        "Panel for %s: %d rows × %d cols | %d features | close_col=%s",
        cfg.name, len(result), len(result.columns),
        len(cfg.feature_neutral_z), close_col,
    )

    return result


# ═══════════════════════════════════════════════════════════
# Gram-Schmidt (from run_retrain_production.py, adapted)
# ═══════════════════════════════════════════════════════════

def apply_gram_schmidt(
    df: pd.DataFrame,
    neutral_z_cols: list[str],
    date_col: str = "date",
    max_correlation: float = 0.85,
) -> pd.DataFrame:
    """
    Cross-sectional Gram-Schmidt orthogonalization.

    Factors are orthogonalized in IC_IR-descending order.
    If a factor's residual correlation with prior factors > max_correlation,
    it is shrunk toward zero.
    """
    from factor_research.orthogonalization import compute_rolling_ic_ir

    # Rank by IC_IR
    ic_irs = compute_rolling_ic_ir(
        df, neutral_z_cols, return_col="forward_return_1m",
        date_col=date_col, rolling_window=24,
    )

    # Get average IC_IR across all dates for ordering
    avg_ic_ir = {}
    for col in neutral_z_cols:
        vals = [ic_irs.get(dt, {}).get(col, 0.0) for dt in ic_irs]
        avg_ic_ir[col] = np.mean([abs(v) for v in vals]) if vals else 0.0

    ranked_cols = sorted(avg_ic_ir, key=lambda c: -avg_ic_ir[c])

    result = df.copy()
    n_factors = len(ranked_cols)

    for dt, date_grp in result.groupby(date_col):
        idx = date_grp.index
        n_stocks = len(idx)

        X = np.zeros((n_stocks, n_factors))
        for j, col in enumerate(ranked_cols):
            vals = date_grp[col].values.astype(np.float64)
            col_mean = np.nanmean(vals)
            X[:, j] = np.where(np.isnan(vals), col_mean, vals)

        # Gram-Schmidt
        Q = np.zeros_like(X)
        for j in range(n_factors):
            v = X[:, j].copy()
            for k in range(j):
                proj = np.dot(v, Q[:, k]) / max(np.dot(Q[:, k], Q[:, k]), 1e-12)
                v = v - proj * Q[:, k]

            # Check residual correlation
            max_abs_corr = 0.0
            for k in range(j):
                if np.std(v) > 1e-12 and np.std(Q[:, k]) > 1e-12:
                    corr = np.corrcoef(v, Q[:, k])[0, 1]
                    max_abs_corr = max(max_abs_corr, abs(corr))

            if max_abs_corr > max_correlation:
                shrink = max_correlation / max_abs_corr
                v = v * shrink

            v_std = np.std(v)
            if v_std > 1e-12:
                v = v / v_std
            Q[:, j] = v

        for j, col in enumerate(ranked_cols):
            result.loc[idx, col] = Q[:, j]

    logger.info(
        "GS orthogonalized %d factors (max_corr=%.2f) across %d dates",
        n_factors, max_correlation, result[date_col].nunique(),
    )
    return result


# ═══════════════════════════════════════════════════════════
# Single-model trainer (adapted from ProductionAlphaEngine)
# ═══════════════════════════════════════════════════════════

def train_single_model(
    cfg: ExperimentConfig,
    panel: pd.DataFrame,
    output_dir: Path,
) -> dict:
    """
    Train a single model config using ProductionAlphaEngine's walk-forward.

    Returns dict with training summary.
    """
    import lightgbm as lgb
    from factor_research.production_engine import (
        ProductionConfig,
        _make_turnover_objective,
        _make_l2_eval,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Prepare panel: GS if enabled ──
    if cfg.gs_enabled:
        logger.info("[%s] Applying GS (max_corr=%.2f)...", cfg.name, cfg.gs_max_correlation)
        panel = apply_gram_schmidt(
            panel, cfg.feature_neutral_z,
            date_col="date", max_correlation=cfg.gs_max_correlation,
        )

    # ── Pre-flight check ──
    from factor_lib.sector_relative import preflight_factor_sanity_check
    try:
        available_features = [c for c in cfg.feature_neutral_z if c in panel.columns]
        preflight_factor_sanity_check(
            panel, available_features, date_col="date", threshold=1e-5,
        )
    except ValueError as e:
        logger.error("[%s] Pre-flight FAILED: %s", cfg.name, e)
        raise

    # ── Build production config ──
    prod_cfg = ProductionConfig(
        seeds=cfg.seeds,
        lambda_turnover=cfg.lambda_turnover,
        colsample_bytree=cfg.colsample_bytree,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        min_child_samples=cfg.min_child_samples,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )

    # Temporarily modify the panel to only contain the columns needed
    feature_cols_available = [c for c in cfg.feature_neutral_z if c in panel.columns]

    # ── Prepare labels ──
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Find close column
    close_col = None
    for c in df.columns:
        if c == "收盘" or c == "close":
            close_col = c
            break

    if close_col is None:
        raise KeyError("No close price column found")

    # Compute forward returns and labels
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["forward_return_1m"] = df.groupby("symbol")[close_col].transform(
        lambda x: x.shift(-1) / x - 1.0
    )
    df["label"] = df.groupby("date")["forward_return_1m"].rank(
        pct=True, na_option="bottom"
    ).fillna(0.5)

    # ── Rank-normalize features ──
    rank_cols = []
    for col in feature_cols_available:
        rc = f"{col}_rank"
        df[rc] = df.groupby("date")[col].rank(pct=True, na_option="bottom").fillna(0.5)
        rank_cols.append(rc)

    # ── Monotonicity constraint mapping ──
    # Map from _neutral_z column to _rank column
    mono_map = {}
    if cfg.monotone_constraints:
        for neutral_col, direction in cfg.monotone_constraints.items():
            rank_col = f"{neutral_col}_rank"
            if rank_col in rank_cols:
                mono_map[rank_col] = direction
        logger.info("[%s] Monotonicity constraints: %s", cfg.name, mono_map)

    # Build monotone_constraints list in feature order
    if mono_map:
        monotone_list = [mono_map.get(rc, 0) for rc in rank_cols]
    else:
        monotone_list = None

    # ── Walk-forward training ──
    dates = sorted(df["date"].unique())
    T, V, S = cfg.train_months, cfg.val_months, cfg.test_months
    fold_window = T + V
    n_folds = len(dates) - fold_window

    if n_folds <= 0:
        raise ValueError(f"Insufficient dates: {len(dates)} < {fold_window + 1}")

    logger.info("[%s] Walk-forward: %d folds | %dM train + %dM val + %dM test",
                 cfg.name, n_folds, T, V, S)
    logger.info("[%s] Features: %d cols | colsample=%.2f | GS=%s | λ=%.2f",
                 cfg.name, len(rank_cols), cfg.colsample_bytree,
                 "ON" if cfg.gs_enabled else "OFF", cfg.lambda_turnover)

    all_preds = []
    importance_list = []
    models_trained = 0

    for fold_idx in range(n_folds):
        train_end = fold_idx + T
        val_end = train_end + V
        test_idx = val_end

        train_dates_set = set(dates[fold_idx:train_end])
        val_dates_set = set(dates[train_end:val_end])
        test_date = dates[test_idx]

        fold_dates = set(dates[fold_idx:test_idx + 1])
        fold_df = df[df["date"].isin(fold_dates)].copy()

        # Prev signal: use label as proxy for cold start
        fold_df["prev_signal"] = fold_df.groupby("symbol")["label"].transform(
            lambda x: x.shift(1).fillna(0.5)
        )

        train_mask = fold_df["date"].isin(train_dates_set)
        val_mask = fold_df["date"].isin(val_dates_set)
        test_mask = fold_df["date"] == test_date

        prev_train = fold_df.loc[train_mask, "prev_signal"].to_numpy(dtype=np.float64)

        X_train = fold_df.loc[train_mask, rank_cols].astype(float)
        y_train = fold_df.loc[train_mask, "label"].astype(float)
        X_val = fold_df.loc[val_mask, rank_cols].astype(float)
        y_val = fold_df.loc[val_mask, "label"].astype(float)
        X_test = fold_df.loc[test_mask, rank_cols].astype(float)

        if len(X_train) < 500 or len(X_val) < 30:
            logger.warning("[%s] Fold %d: insufficient (train=%d, val=%d), skip",
                           cfg.name, fold_idx, len(X_train), len(X_val))
            continue

        params = cfg.to_lgb_params()
        params["random_state"] = cfg.seeds[0]
        params["objective"] = _make_turnover_objective(prev_train, cfg.lambda_turnover)

        if monotone_list is not None:
            params["monotone_constraints"] = monotone_list
            params["monotone_constraints_method"] = "advanced"

        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

        try:
            model = lgb.train(
                params=params, train_set=train_ds,
                num_boost_round=cfg.n_estimators,
                valid_sets=[train_ds, val_ds],
                valid_names=["train", "val"],
                feval=_make_l2_eval(),
                callbacks=[
                    lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            y_pred = model.predict(X_test)
            pred_df = fold_df.loc[test_mask, ["date", "symbol"]].copy()
            pred_df["alpha_signal"] = y_pred
            all_preds.append(pred_df)

            imp = pd.DataFrame({
                "feature": rank_cols,
                "gain": model.feature_importance(importance_type="gain"),
                "split": model.feature_importance(importance_type="split"),
            })
            imp["fold"] = fold_idx
            importance_list.append(imp)
            models_trained += 1

            if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
                logger.info("  [%s] Fold %3d/%d | test=%s | train=%d val=%d test=%d | iter=%d",
                             cfg.name, fold_idx + 1, n_folds, str(test_date)[:10],
                             len(X_train), len(X_val), len(X_test),
                             model.best_iteration)

        except Exception as e:
            logger.error("[%s] Fold %d failed: %s", cfg.name, fold_idx, e)
            continue

    if not all_preds:
        raise RuntimeError(f"[{cfg.name}] All folds failed.")

    predictions = pd.concat(all_preds, ignore_index=True)
    predictions["date"] = pd.to_datetime(predictions["date"])

    # ── Feature importance ──
    if importance_list:
        imp_all = pd.concat(importance_list, ignore_index=True)
        fi = imp_all.groupby("feature")[["gain", "split"]].mean().sort_values("gain", ascending=False)
    else:
        fi = pd.DataFrame()

    # ── Save ──
    pred_path = output_dir / f"{cfg.name}_oos.parquet"
    predictions.to_parquet(pred_path, index=False)

    # Save feature importance
    if not fi.empty:
        fi_path = output_dir / f"{cfg.name}_feature_importance.csv"
        fi.to_csv(fi_path, encoding="utf-8-sig")

    # Save config
    cfg_path = output_dir / f"{cfg.name}_config.json"
    cfg_dict = {
        "name": cfg.name,
        "description": cfg.description,
        "gs_enabled": cfg.gs_enabled,
        "gs_max_correlation": cfg.gs_max_correlation,
        "colsample_bytree": cfg.colsample_bytree,
        "n_features": len(rank_cols),
        "features": rank_cols,
        "monotone_constraints": mono_map if mono_map else {},
        "n_seeds": len(cfg.seeds),
        "seeds": cfg.seeds,
        "lambda_turnover": cfg.lambda_turnover,
        "n_folds_trained": models_trained,
        "n_folds_total": n_folds,
        "n_predictions": len(predictions),
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False, default=str)

    # ── Quick stats ──
    # Cross-sectional rank the predictions
    predictions["signal_rank"] = predictions.groupby("date")["alpha_signal"].rank(
        pct=True, na_option="bottom"
    )

    logger.info("[%s] Training complete: %d/%d folds, %d predictions",
                 cfg.name, models_trained, n_folds, len(predictions))
    logger.info("[%s] OOS: %s ~ %s | %d cross-sections",
                 cfg.name,
                 predictions["date"].min().strftime("%Y-%m-%d"),
                 predictions["date"].max().strftime("%Y-%m-%d"),
                 predictions["date"].nunique())
    if not fi.empty:
        logger.info("[%s] Top-5 features (gain):", cfg.name)
        for feat, row in fi.head(5).iterrows():
            logger.info("  %s: gain=%.0f", feat, row["gain"])

    return {
        "name": cfg.name,
        "models_trained": models_trained,
        "n_folds": n_folds,
        "n_predictions": len(predictions),
        "predictions": predictions,
        "feature_importance": fi,
        "config": cfg_dict,
    }


# ═══════════════════════════════════════════════════════════
# Main experiment runner
# ═══════════════════════════════════════════════════════════

def run_experiments(
    models: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, dict]:
    """
    Run the V1.5 experiment matrix.

    Parameters
    ----------
    models : list[str], optional
        Which models to run. Default: all ["M0","M1","M2","M3","M4","M5"]
    dry_run : bool
        If True, print configs and exit without training.
    """
    if models is None:
        models = ["M0", "M1", "M2", "M3", "M4", "M5"]

    # Validate
    invalid = [m for m in models if m not in ALL_EXPERIMENTS]
    if invalid:
        raise ValueError(f"Unknown model IDs: {invalid}. Choose from: {list(ALL_EXPERIMENTS)}")

    logger.info("=" * 64)
    logger.info("V1.5 Experiment Matrix — %d Models", len(models))
    logger.info("=" * 64)

    # Print experiment matrix
    logger.info("\n%-4s %-22s %-5s %-6s %-5s %s",
                 "ID", "Name", "GS", "colsp", "Mono", "Features")
    logger.info("-" * 80)
    for mid in models:
        cfg = ALL_EXPERIMENTS[mid]
        gs_str = f"ON({cfg.gs_max_correlation})" if cfg.gs_enabled else "OFF"
        mono_str = "YES" if cfg.monotone_constraints else "no"
        logger.info("%-4s %-22s %-9s %-6.2f %-5s %d features",
                     mid, cfg.name, gs_str, cfg.colsample_bytree, mono_str,
                     len(cfg.feature_neutral_z))
    logger.info("=" * 64)

    if dry_run:
        logger.info("\nDry run — exiting without training.")
        return {}

    # ── Run each model ──
    results = {}
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for mid in models:
        cfg = ALL_EXPERIMENTS[mid]
        logger.info("\n" + "=" * 64)
        logger.info("Training: %s — %s", mid, cfg.description)
        logger.info("=" * 64)

        t0 = time.perf_counter()

        try:
            # Prepare panel
            panel = prepare_panel_for_config(cfg)

            # Train
            result = train_single_model(cfg, panel, MODEL_OUTPUT_DIR)
            result["train_time_sec"] = time.perf_counter() - t0
            results[mid] = result

            logger.info("[%s] Done in %.1f min", mid, result["train_time_sec"] / 60)

        except Exception as e:
            logger.error("[%s] FAILED: %s", mid, e, exc_info=True)
            results[mid] = {"name": mid, "error": str(e)}

    # ── Final summary ──
    logger.info("\n" + "=" * 64)
    logger.info("Experiment Matrix — Final Summary")
    logger.info("=" * 64)
    for mid in models:
        r = results.get(mid, {})
        if "error" in r:
            logger.info("  %s: ❌ FAILED — %s", mid, r["error"])
        else:
            logger.info("  %s: ✅ %d folds, %d preds, %.1f min",
                         mid, r.get("models_trained", 0),
                         r.get("n_predictions", 0),
                         r.get("train_time_sec", 0) / 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="V1.5 Experiment Matrix")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model IDs to train (M0 M1 M2 M3 M4 M5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs only, don't train")
    args = parser.parse_args()

    run_experiments(models=args.models, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
