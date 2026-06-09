"""
Production Alpha Engine — Multi-Seed Ensemble + Turnover-Aware Inference.

Frozen V7 architecture for live deployment:
  - Label: forward_return_1m → cross-sectional rank [0, 1]
  - Gap: 0M (standard 1-step-forward, no blind zone)
  - Objective: Custom L2 + λ·(ŷ−ŷ_{t−1})²  with λ = 2.0
  - Ensemble: N models (seeds = [42, 888, 2026]) → equal-weight average
  - NO EMA smoothing (preserves crisis-response agility)

Key design decisions for production:
  1. Training / inference strictly separated — fit() trains, predict_cross_section() infers
  2. All models saved to disk; inference loads without retraining
  3. prev_signal is an EXTERNAL input to predict_cross_section()
     → Caller is responsible for maintaining the t−1 signal cache
  4. Cross-sectional rank ensures signal is always in [0, 1] with uniform distribution
     → Robust to distribution shift between training and live data
"""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("production_engine")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

@dataclass
class ProductionConfig:
    """
    Frozen production configuration for the V7 Turnover-Aware Ensemble.

    All hyperparameters are locked — no tuning in production.
    Only seeds, paths, and inference-time parameters may vary.
    """

    # ── Frozen V7 architecture ──
    label_horizon: int = 1                # 1M forward return
    gap_months: int = 0                   # NO gap (standard 1-step-forward)
    lambda_turnover: float = 2.0          # Turnover penalty coefficient
    min_stocks_per_date: int = 50         # Cross-section quality filter

    # ── Multi-seed ensemble ──
    seeds: List[int] = field(default_factory=lambda: [42, 888, 2026])

    # ── LightGBM hyperparameters (FROZEN) ──
    objective: str = "regression"
    metric: str = "l2"
    boosting: str = "gbdt"
    num_leaves: int = 24
    max_depth: int = 4
    learning_rate: float = 0.02
    n_estimators: int = 2000
    subsample: float = 1.0               # Mandatory: closure alignment
    colsample_bytree: float = 0.70
    subsample_freq: int = 1
    min_child_samples: int = 100
    reg_alpha: float = 0.10
    reg_lambda: float = 0.10
    early_stopping_rounds: int = 50
    verbose: int = -1
    n_jobs: int = -1

    # ── Feature pipeline ──
    feature_method: str = "rank"          # Cross-sectional rank normalization

    def to_lgb_params(self) -> dict:
        return {
            "objective": self.objective, "metric": self.metric,
            "boosting": self.boosting, "num_leaves": self.num_leaves,
            "max_depth": self.max_depth, "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "subsample_freq": self.subsample_freq,
            "min_child_samples": self.min_child_samples,
            "reg_alpha": self.reg_alpha, "reg_lambda": self.reg_lambda,
            "verbose": self.verbose, "n_jobs": self.n_jobs,
        }


# ═══════════════════════════════════════════════════════════
# Custom Objective (V7 Turnover-Aware L2 — UNCHANGED)
# ═══════════════════════════════════════════════════════════

def _make_turnover_objective(
    prev_signal: np.ndarray,
    lambda_penalty: float,
):
    """
    L = 0.5*(ŷ−y)² + λ·0.5*(ŷ−ŷₜ₋₁)²
    g = (ŷ−y) + λ·(ŷ−ŷₜ₋₁)
    h = 1 + λ
    """
    _prev = np.asarray(prev_signal, dtype=np.float64)
    _lam = float(lambda_penalty)

    def _objective(preds: np.ndarray, train_data) -> tuple:
        labels = train_data.get_label().astype(np.float64)
        if len(preds) != len(_prev):
            return preds - labels, np.ones_like(preds)
        residual = preds - labels
        turnover = preds - _prev
        grad = residual + _lam * turnover
        hess = np.full_like(preds, 1.0 + _lam, dtype=np.float64)
        return grad, hess
    return _objective


def _make_l2_eval():
    def _fn(preds, train_data) -> tuple:
        return "l2", float(np.mean((preds - train_data.get_label()) ** 2)), False
    return _fn


# ═══════════════════════════════════════════════════════════
# Production Alpha Engine
# ═══════════════════════════════════════════════════════════

class ProductionAlphaEngine:
    """
    Production-grade multi-seed ensemble alpha engine.

    Lifecycle
    ---------
    1. fit(panel, blended)          — Walk-forward ensemble training
       → Trains N seeds × K folds models; saves to disk.

    2. engine = ProductionAlphaEngine.load_models(path)
       → Loads trained models for inference-only deployment.

    3. signals = engine.predict_cross_section(features, prev_signal)
       → Single cross-section inference. Returns ranked signals ∈ [0, 1].

    Architecture invariants
    -----------------------
    - NO EMA smoothing — preserves crisis-response speed (MaxDD -11.14%)
    - Ensemble = equal-weight average of all seeds → cross-sectional rank
    - prev_signal is ALWAYS an external input (caller owns the cache)
    - Cross-sectional rank guarantees [0, 1] output regardless of raw scale

    Raises
    ------
    ValueError
        If feature columns don't match training-time feature set.
        If prev_signal length doesn't match number of stocks.
    RuntimeError
        If predict_cross_section() is called before fit() or load_models().
    """

    def __init__(self, config: Optional[ProductionConfig] = None):
        self.config = config or ProductionConfig()
        self.seeds: List[int] = list(self.config.seeds)

        # Internal state
        self._models: Dict[int, List] = {}      # seed → [Booster per fold]
        self._feature_cols: List[str] = []
        self._fitted: bool = False
        self._n_folds: int = 0
        self._training_dates: List = []          # All dates seen during fit()

        self._feature_importance: Optional[pd.DataFrame] = None

    # ═══════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════

    def fit(
        self,
        panel: pd.DataFrame,
        blended: Optional[pd.DataFrame] = None,
        *,
        date_col: str = "date",
        symbol_col: str = "symbol",
        close_col: str = "收盘",
        train_months: int = 36,
        val_months: int = 6,
        test_months: int = 1,
    ) -> "ProductionAlphaEngine":
        """
        Walk-forward multi-seed ensemble training.

        For each rolling fold:
          1. Extract features (rank-normalized factors + prev_signal anchor)
          2. For each seed ∈ [42, 888, 2026]:
             a. Build lgb.Dataset with training data
             b. Train with Turnover-Aware Custom Objective (closure-injected prev_signal)
             c. Store trained Booster
          3. OOS predict → stored for backtest use

        Parameters
        ----------
        panel : pd.DataFrame
            Must contain _neutral_z factor columns + close price.
        blended : pd.DataFrame, optional
            Must contain alpha_signal column (for prev_signal anchor).
        date_col, symbol_col, close_col : str
        train_months, val_months, test_months : int
            Walk-forward window sizes.

        Returns
        -------
        self : ProductionAlphaEngine
            Fitted engine, ready for predict_cross_section() or save_models().
        """
        import lightgbm as lgb

        cfg = self.config
        T, V, S = train_months, val_months, test_months
        fold_window = T + V

        # ── Preprocessing pipeline ──
        df = self._filter_dates(panel, date_col, cfg.min_stocks_per_date)
        df = self._prepare_labels(df, date_col, symbol_col, close_col)
        df = self._prepare_features(df, blended, date_col, symbol_col)

        dates = sorted(df[date_col].unique())
        n_dates = len(dates)
        n_folds = n_dates - fold_window

        if n_folds <= 0:
            raise ValueError(
                f"Insufficient data: {n_dates} dates, need {fold_window + 1} "
                f"(train={T} + val={V} + test={S})"
            )
        self._n_folds = n_folds
        self._training_dates = list(dates)

        # Initialize model storage: seed → [model_fold_0, model_fold_1, ...]
        self._models = {seed: [] for seed in self.seeds}
        all_preds: List[pd.DataFrame] = []
        importance_acc: Dict[int, List[pd.DataFrame]] = {s: [] for s in self.seeds}

        logger.info("=" * 64)
        logger.info("ProductionAlphaEngine.fit() — Multi-Seed Ensemble Training")
        logger.info("  Seeds: %s | Folds: %d | λ: %.2f | Label: 1M | Gap: 0M",
                     self.seeds, n_folds, cfg.lambda_turnover)
        logger.info("  Window: %dM train + %dM val + %dM test", T, V, S)
        logger.info("=" * 64)

        for fold_idx in range(n_folds):
            # ── Fold time indices (NO gap) ──
            train_end = fold_idx + T
            val_end = train_end + V
            test_idx = val_end

            train_dates = set(dates[fold_idx:train_end])
            val_dates = set(dates[train_end:val_end])
            test_date = dates[test_idx]

            fold_dates = set(dates[fold_idx:test_idx + 1])
            fold_df = df[df[date_col].isin(fold_dates)].copy()

            train_mask = fold_df[date_col].isin(train_dates)
            val_mask = fold_df[date_col].isin(val_dates)
            test_mask = fold_df[date_col] == test_date

            feature_cols = [c for c in self._feature_cols if c in fold_df.columns]

            # Extract prev_signal BEFORE dropping from features
            if "prev_signal" not in fold_df.columns:
                raise KeyError("prev_signal column missing — did prepare_features() run?")
            prev_train = fold_df.loc[train_mask, "prev_signal"].to_numpy(dtype=np.float64)
            prev_val = fold_df.loc[val_mask, "prev_signal"].to_numpy(dtype=np.float64)

            X_train = fold_df.loc[train_mask, feature_cols].astype(float)
            y_train = fold_df.loc[train_mask, "label"].astype(float)
            X_val = fold_df.loc[val_mask, feature_cols].astype(float)
            y_val = fold_df.loc[val_mask, "label"].astype(float)
            X_test = fold_df.loc[test_mask, feature_cols].astype(float)

            if len(X_train) < 500 or len(X_val) < 30:
                logger.warning("Fold %d: insufficient samples (train=%d, val=%d), skip",
                               fold_idx, len(X_train), len(X_val))
                for s in self.seeds:
                    self._models[s].append(None)
                continue

            # ── Train one model per seed ──
            fold_preds_seeds = []
            for seed in self.seeds:
                try:
                    params = cfg.to_lgb_params()
                    params["random_state"] = seed
                    params["objective"] = _make_turnover_objective(prev_train, cfg.lambda_turnover)

                    train_ds = lgb.Dataset(X_train, label=y_train)
                    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

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
                    self._models[seed].append(model)

                    # Per-seed prediction
                    y_pred = model.predict(X_test)
                    fold_preds_seeds.append(y_pred)

                    # Feature importance
                    imp = pd.DataFrame({
                        "feature": feature_cols,
                        "gain": model.feature_importance(importance_type="gain"),
                        "split": model.feature_importance(importance_type="split"),
                    })
                    imp["fold"] = fold_idx
                    imp["seed"] = seed
                    importance_acc[seed].append(imp)

                except Exception as e:
                    logger.error("Fold %d seed %d failed: %s", fold_idx, seed, e)
                    self._models[seed].append(None)

            # ── OOS ensemble prediction (equal-weight average) ──
            if fold_preds_seeds:
                ensemble_pred = np.mean(fold_preds_seeds, axis=0)
                pred_df = fold_df.loc[test_mask, [date_col, symbol_col]].copy()
                pred_df["ensemble_signal"] = ensemble_pred
                all_preds.append(pred_df)

            if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
                train_preds_raw = []
                for seed in self.seeds:
                    m = self._models[seed][-1]
                    if m is not None:
                        train_preds_raw.append(m.predict(X_train))
                if train_preds_raw:
                    avg_train_pred = np.mean(train_preds_raw, axis=0)
                    avg_pen = np.mean((avg_train_pred - prev_train) ** 2)
                    logger.info(
                        "  Fold %3d/%d | test=%s | train=%d val=%d test=%d | "
                        "best_iter=%d | avg_Δpred²=%.4f",
                        fold_idx + 1, n_folds, str(test_date)[:10],
                        len(X_train), len(X_val), len(X_test),
                        self._models[self.seeds[0]][-1].best_iteration
                        if self._models[self.seeds[0]][-1] is not None else 0,
                        avg_pen,
                    )

        # ── Aggregate results ──
        if all_preds:
            self._oos_predictions = pd.concat(all_preds, ignore_index=True)
            self._oos_predictions[date_col] = pd.to_datetime(self._oos_predictions[date_col])
        else:
            self._oos_predictions = pd.DataFrame()

        # Aggregate feature importance
        all_imps = []
        for seed_imps in importance_acc.values():
            all_imps.extend(seed_imps)
        if all_imps:
            imp_all = pd.concat(all_imps, ignore_index=True)
            self._feature_importance = (
                imp_all.groupby("feature")[["gain", "split"]]
                .mean().sort_values("gain", ascending=False)
            )

        self._fitted = True
        n_models = sum(1 for models in self._models.values() for m in models if m is not None)
        logger.info("=" * 64)
        logger.info("Training complete: %d folds × %d seeds = %d models trained",
                     n_folds, len(self.seeds), n_models)
        logger.info("OOS predictions: %d rows, %d cross-sections",
                     len(self._oos_predictions),
                     self._oos_predictions[date_col].nunique() if len(self._oos_predictions) > 0 else 0)
        if self._feature_importance is not None:
            logger.info("Top-3 features: %s",
                         ", ".join(f"{f}({r['gain']:.0f})"
                                   for f, r in self._feature_importance.head(3).iterrows()))
        logger.info("=" * 64)
        return self

    def train_production_models(
        self,
        panel: pd.DataFrame,
        blended: Optional[pd.DataFrame] = None,
        *,
        current_date: str | pd.Timestamp | None = None,
        date_col: str = "date",
        symbol_col: str = "symbol",
        close_col: str = "收盘",
        train_months: int = 36,
        val_months: int = 6,
    ) -> "ProductionAlphaEngine":
        """
        Production retraining: single-window, latest-data-only.

        Unlike fit() which does walk-forward CV across ALL folds,
        this function:
          1. Takes `current_date` (e.g., month-end rebalance date)
          2. Slices a SINGLE training window: the 36 months ending at current_date
          3. Trains one model per seed on that single window
          4. Stores ONLY these models (n_folds=1, 1 model per seed)

        This is what runs on month-end rebalance day to refresh the model
        with the latest 36 months of data. No walk-forward history is
        preserved — only the freshest model.

        Parameters
        ----------
        panel : pd.DataFrame
            Must contain _neutral_z factor columns + close price.
        blended : pd.DataFrame, optional
            Must contain alpha_signal column for prev_signal anchor.
        current_date : str or pd.Timestamp, optional
            The "as of" date. Training window = [current_date - 36M, current_date].
            If None, uses the latest date in panel.
        date_col, symbol_col, close_col : str
        train_months : int
            Training window size in months (default: 36).
        val_months : int
            Validation window size in months (default: 6).

        Returns
        -------
        self : ProductionAlphaEngine
            Fitted with 1 fold per seed. Ready for predict_cross_section()
            and save_models(mode="production").
        """
        import lightgbm as lgb

        cfg = self.config
        T, V = train_months, val_months

        # ── Preprocessing ──
        df = self._filter_dates(panel, date_col, cfg.min_stocks_per_date)
        df = self._prepare_labels(df, date_col, symbol_col, close_col)
        df = self._prepare_features(df, blended, date_col, symbol_col)

        dates = sorted(df[date_col].unique())

        # Determine the training window end
        if current_date is None:
            train_end_date = dates[-1]  # latest available
        else:
            train_end_date = pd.Timestamp(current_date)

        # Find the index of train_end_date in sorted dates
        # Use the latest date <= train_end_date (PIT safety)
        valid_end_indices = [i for i, d in enumerate(dates) if d <= train_end_date]
        if not valid_end_indices:
            raise ValueError(
                f"No dates <= {train_end_date} in panel. "
                f"Panel date range: {dates[0]} ~ {dates[-1]}"
            )
        end_idx = valid_end_indices[-1]

        # Training window: end_idx - V (val) back T+V months
        fold_window = T + V
        start_idx = max(0, end_idx - fold_window + 1)

        train_dates = set(dates[start_idx:start_idx + T])
        val_dates = set(dates[start_idx + T:end_idx + 1])

        fold_dates = set(dates[start_idx:end_idx + 1])
        fold_df = df[df[date_col].isin(fold_dates)].copy()

        train_mask = fold_df[date_col].isin(train_dates)
        val_mask = fold_df[date_col].isin(val_dates)

        feature_cols = [c for c in self._feature_cols if c in fold_df.columns]

        if "prev_signal" not in fold_df.columns:
            raise KeyError("prev_signal column missing")
        prev_train = fold_df.loc[train_mask, "prev_signal"].to_numpy(dtype=np.float64)

        X_train = fold_df.loc[train_mask, feature_cols].astype(float)
        y_train = fold_df.loc[train_mask, "label"].astype(float)
        X_val = fold_df.loc[val_mask, feature_cols].astype(float)
        y_val = fold_df.loc[val_mask, "label"].astype(float)

        if len(X_train) < 500:
            raise ValueError(
                f"Insufficient training samples: {len(X_train)}. "
                f"Need at least 500. Check data range."
            )

        logger.info("=" * 64)
        logger.info("ProductionAlphaEngine.train_production_models()")
        logger.info("  Training window: %s ~ %s (%dM + %dM val)",
                     dates[start_idx].strftime("%Y-%m-%d"),
                     train_end_date.strftime("%Y-%m-%d"), T, V)
        logger.info("  Train: %d rows | Val: %d rows | Features: %d",
                     len(X_train), len(X_val), len(feature_cols))
        logger.info("  Seeds: %s | λ: %.2f", self.seeds, cfg.lambda_turnover)
        logger.info("=" * 64)

        # ── Train one model per seed ──
        self._models = {seed: [] for seed in self.seeds}
        self._n_folds = 1
        importance_acc: dict[int, pd.DataFrame] = {}

        for seed in self.seeds:
            try:
                params = cfg.to_lgb_params()
                params["random_state"] = seed
                params["objective"] = _make_turnover_objective(prev_train, cfg.lambda_turnover)

                train_ds = lgb.Dataset(X_train, label=y_train)
                val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

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
                self._models[seed].append(model)

                imp = pd.DataFrame({
                    "feature": feature_cols,
                    "gain": model.feature_importance(importance_type="gain"),
                    "split": model.feature_importance(importance_type="split"),
                })
                imp["seed"] = seed
                importance_acc[seed] = imp

                train_preds = model.predict(X_train)
                avg_pen = np.mean((train_preds - prev_train) ** 2)
                logger.info("  Seed %d: trained | best_iter=%d | avg_dpred^2=%.4f",
                           seed, model.best_iteration, avg_pen)

            except Exception as e:
                logger.error("Seed %d failed: %s", seed, e)
                self._models[seed].append(None)

        # ── Feature importance ──
        if importance_acc:
            imp_all = pd.concat(importance_acc.values(), ignore_index=True)
            self._feature_importance = (
                imp_all.groupby("feature")[["gain", "split"]]
                .mean().sort_values("gain", ascending=False)
            )

        self._fitted = True
        n_trained = sum(1 for models in self._models.values() for m in models if m is not None)
        logger.info("=" * 64)
        logger.info("Production retraining complete: %d/%d seeds trained", n_trained, len(self.seeds))
        logger.info("=" * 64)
        return self

    def predict_cross_section(
        self,
        features: pd.DataFrame,
        prev_signal: np.ndarray | pd.Series,
        *,
        fold_idx: int = -1,
        rank_output: bool = True,
    ) -> np.ndarray:
        """
        Multi-seed ensemble inference for a SINGLE cross-section.

        This is the production inference entry point. It:
          1. Validates feature alignment against training-time columns
          2. Validates prev_signal length
          3. Runs all models (N seeds × 1 fold) → averages predictions
          4. Optionally applies cross-sectional rank → [0, 1]

        Parameters
        ----------
        features : pd.DataFrame
            Shape (n_stocks, n_features). Must contain exactly the columns
            seen during training (self._feature_cols), in any order.
        prev_signal : np.ndarray or pd.Series
            Shape (n_stocks,). The alpha_signal from t−1 for each stock.
            This is the temporal anchor for the turnover penalty.
            Caller MUST maintain this cache externally.
        fold_idx : int
            Which trained fold to use. Default -1 = most recent fold.
            In production, always use -1 (latest model).
        rank_output : bool
            If True (default), apply cross-sectional rank to [0, 1].
            If False, return raw ensemble predictions.

        Returns
        -------
        np.ndarray, shape (n_stocks,)
            Final alpha signal ∈ [0, 1] if rank_output=True,
            else raw ensemble prediction (unbounded).

        Raises
        ------
        RuntimeError
            If engine not fitted and no models loaded.
        ValueError
            If feature columns mismatch or prev_signal length mismatch.
        """
        if not self._fitted:
            raise RuntimeError(
                "Engine not fitted. Call fit() or load_models() before predict_cross_section()."
            )

        # ── Input validation ──
        if features is None or len(features) == 0:
            raise ValueError("features is empty or None")

        n_stocks = len(features)

        # Validate feature columns
        missing = set(self._feature_cols) - set(features.columns)
        if missing:
            raise ValueError(
                f"Missing feature columns: {missing}. "
                f"Expected {len(self._feature_cols)} columns, got {len(features.columns)}."
            )
        extra = set(features.columns) - set(self._feature_cols)
        if extra:
            logger.warning("Extra columns in features (ignored): %s", extra)

        # Align columns to training order
        X = features[self._feature_cols].astype(float)

        # Validate prev_signal
        prev = np.asarray(prev_signal, dtype=np.float64).ravel()
        if len(prev) != n_stocks:
            raise ValueError(
                f"prev_signal length ({len(prev)}) != n_stocks ({n_stocks}). "
                f"Each stock must have exactly one prev_signal value."
            )
        if np.any(np.isnan(prev)):
            n_nan = np.isnan(prev).sum()
            logger.warning("%d NaN in prev_signal — filling with 0.5", n_nan)
            prev = np.where(np.isnan(prev), 0.5, prev)

        # Validate fold_idx
        if self._models:
            any_seed = next(iter(self._models.values()))
            n_available = len(any_seed)
        else:
            n_available = 0
        if fold_idx < 0:
            fold_idx = n_available + fold_idx
        if fold_idx < 0 or fold_idx >= n_available:
            raise ValueError(
                f"fold_idx={fold_idx} out of range [0, {n_available - 1}]"
            )

        # ── Ensemble inference ──
        raw_preds = []
        for seed in self.seeds:
            model = self._models[seed][fold_idx]
            if model is None:
                logger.warning("Seed %d fold %d model is None — skipping", seed, fold_idx)
                continue
            raw_preds.append(model.predict(X))

        if not raw_preds:
            raise RuntimeError(f"No valid models for fold {fold_idx}")

        # Equal-weight average
        ensemble_raw = np.mean(raw_preds, axis=0)

        # ── Cross-sectional rank (production standard) ──
        if rank_output:
            ranked = pd.Series(ensemble_raw).rank(pct=True, na_option="bottom").fillna(0.5)
            return ranked.to_numpy(dtype=np.float64)
        else:
            return ensemble_raw.astype(np.float64)

    def predict_batch(
        self,
        panel: pd.DataFrame,
        blended: Optional[pd.DataFrame] = None,
        *,
        date_col: str = "date",
        symbol_col: str = "symbol",
    ) -> pd.DataFrame:
        """
        Convenience: run predict_cross_section for ALL dates in a panel.

        Uses the OOS predictions stored during fit() for efficiency.
        If called on a fitted engine, returns stored OOS predictions.
        For new data, iterates through dates and calls predict_cross_section().

        Returns
        -------
        pd.DataFrame with columns [date, symbol, ensemble_signal]
        """
        if hasattr(self, "_oos_predictions") and len(self._oos_predictions) > 0:
            return self._oos_predictions.copy()

        raise NotImplementedError(
            "Batch prediction on new data not yet implemented. "
            "Use predict_cross_section() per cross-section."
        )

    # ═══════════════════════════════════════════════════════
    # Model Persistence
    # ═══════════════════════════════════════════════════════

    def save_models(
        self,
        path: str | Path,
        mode: str = "production",
        keep_versions: int = 3,
    ) -> Path:
        """
        Persist models + metadata to disk.

        Two modes:
          - mode="backtest":  Save ALL folds (walk-forward CV history).
            Naming: model_s{seed}_f{fold}.pkl.  No cleanup.
          - mode="production": Save only the LAST fold per seed,
            with timestamped names.  Cleanup: keep only the most recent
            `keep_versions` retraining runs per seed.

        Production directory structure (after N retraining runs):
          path/
            metadata.json                          — latest config + feature_cols
            v7_seed42_20260630.pkl                 — seed 42, trained on 2026-06-30
            v7_seed888_20260630.pkl
            v7_seed2026_20260630.pkl
            v7_seed42_20260530.pkl                 — seed 42, trained on 2026-05-30 (rollback)
            v7_seed888_20260530.pkl
            v7_seed2026_20260530.pkl
            v7_seed42_20260430.pkl                 — seed 42, trained on 2026-04-30 (rollback)
            ...
            feature_importance.csv

        Parameters
        ----------
        path : str or Path
            Directory to save models. Created if not exists.
        mode : str
            "backtest" or "production".
        keep_versions : int
            Number of recent versions to keep per seed (production mode only).

        Returns
        -------
        Path: the save directory.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # ── Metadata (always writes) ──
        meta = {
            "seeds": self.seeds,
            "n_folds": self._n_folds,
            "feature_cols": self._feature_cols,
            "training_dates": [str(d) for d in self._training_dates],
            "mode": mode,
            "config": {
                "lambda_turnover": self.config.lambda_turnover,
                "label_horizon": self.config.label_horizon,
                "gap_months": self.config.gap_months,
                "num_leaves": self.config.num_leaves,
                "max_depth": self.config.max_depth,
                "learning_rate": self.config.learning_rate,
                "subsample": self.config.subsample,
                "min_child_samples": self.config.min_child_samples,
                "reg_alpha": self.config.reg_alpha,
                "reg_lambda": self.config.reg_lambda,
            },
        }
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

        # ── Models ──
        n_saved = 0

        if mode == "backtest":
            # Legacy behavior: save all folds
            for seed, models in self._models.items():
                for fold_idx, model in enumerate(models):
                    if model is not None:
                        model_path = path / f"model_s{seed}_f{fold_idx}.pkl"
                        with open(model_path, "wb") as f:
                            pickle.dump(model, f)
                        n_saved += 1

        elif mode == "production":
            # Production: only the LAST fold per seed, timestamped
            today_str = pd.Timestamp.now().strftime("%Y%m%d")
            for seed, models in self._models.items():
                # Find the last non-None model
                last_model = None
                for m in reversed(models):
                    if m is not None:
                        last_model = m
                        break

                if last_model is None:
                    logger.warning("Seed %d has no valid model — skipping", seed)
                    continue

                model_path = path / f"v7_seed{seed}_{today_str}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(last_model, f)
                n_saved += 1

            # ── Rolling cleanup: keep only `keep_versions` per seed ──
            self._cleanup_old_models(path, keep_versions)

        else:
            raise ValueError(f"Unknown save mode: {mode}. Use 'backtest' or 'production'.")

        # ── Feature importance ──
        if self._feature_importance is not None:
            self._feature_importance.to_csv(path / "feature_importance.csv")

        logger.info("Saved %d models (mode=%s) → %s", n_saved, mode, path)
        return path

    def _cleanup_old_models(self, path: Path, keep_versions: int):
        """
        Remove old model files, keeping only the latest `keep_versions`
        per seed.  Deletes files matching `v7_seed{seed}_*.pkl`.

        This enables rolling rollback: if a retraining produces worse
        models, the previous 2 versions are still on disk and can be
        loaded manually.
        """
        for seed in self.seeds:
            pattern = f"v7_seed{seed}_*.pkl"
            files = sorted(path.glob(pattern))

            if len(files) <= keep_versions:
                continue

            # Files are sorted alphabetically; timestamp YYYYMMDD sorts correctly
            to_delete = files[:-keep_versions]
            for fp in to_delete:
                try:
                    fp.unlink()
                    logger.debug("  Cleaned up old model: %s", fp.name)
                except OSError as e:
                    logger.warning("  Failed to delete %s: %s", fp.name, e)

            if to_delete:
                logger.info("  Seed %d: cleaned %d old models (kept %d)",
                           seed, len(to_delete), keep_versions)

    @classmethod
    def load_models(
        cls,
        path: str | Path,
        config: Optional[ProductionConfig] = None,
        *,
        model_date: str | None = None,
    ) -> "ProductionAlphaEngine":
        """
        Load trained models from disk for inference-only deployment.

        Auto-detects the save mode:
          - Production mode (v7_seed{seed}_{date}.pkl):
            Loads the LATEST dated models by default.
            Specify model_date="20260630" to load a specific version
            (e.g., for rollback).
          - Backtest mode (model_s{seed}_f{fold}.pkl):
            Loads all folds.

        Parameters
        ----------
        path : str or Path
            Directory containing metadata.json + model files.
        config : ProductionConfig, optional
            If None, reconstructed from metadata.json.
        model_date : str, optional
            "YYYYMMDD" of the specific production models to load.
            If None, auto-loads latest. Ignored in backtest mode.

        Returns
        -------
        ProductionAlphaEngine
            Ready for predict_cross_section(). Training state preserved.

        Raises
        ------
        FileNotFoundError
            If path or metadata.json not found.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")

        meta_path = path / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        save_mode = meta.get("mode", "backtest")

        # ── Reconstruct config ──
        if config is None:
            cfg_dict = meta.get("config", {})
            config = ProductionConfig(
                lambda_turnover=cfg_dict.get("lambda_turnover", 2.0),
                label_horizon=cfg_dict.get("label_horizon", 1),
                gap_months=cfg_dict.get("gap_months", 0),
                num_leaves=cfg_dict.get("num_leaves", 24),
                max_depth=cfg_dict.get("max_depth", 4),
                learning_rate=cfg_dict.get("learning_rate", 0.02),
                subsample=cfg_dict.get("subsample", 1.0),
                min_child_samples=cfg_dict.get("min_child_samples", 100),
                reg_alpha=cfg_dict.get("reg_alpha", 0.10),
                reg_lambda=cfg_dict.get("reg_lambda", 0.10),
            )

        # ── Reconstruct engine ──
        engine = cls(config)
        engine.seeds = meta.get("seeds", [42, 888, 2026])
        engine._feature_cols = meta["feature_cols"]
        engine._training_dates = meta.get("training_dates", [])

        if save_mode == "production":
            # ── Production mode: load latest (or specified) dated models ──
            engine._n_folds = 1  # Production always has 1 fold
            engine._models = {seed: [] for seed in engine.seeds}
            n_loaded = 0

            for seed in engine.seeds:
                if model_date:
                    # Explicit rollback: load specific date
                    fp = path / f"v7_seed{seed}_{model_date}.pkl"
                    if fp.exists():
                        with open(fp, "rb") as f:
                            engine._models[seed].append(pickle.load(f))
                        n_loaded += 1
                    else:
                        logger.warning("Model %s not found for specified date", fp.name)
                        engine._models[seed].append(None)
                else:
                    # Auto-detect latest
                    pattern = f"v7_seed{seed}_*.pkl"
                    files = sorted(path.glob(pattern))
                    if files:
                        latest = files[-1]  # alphabetical YYYYMMDD sort
                        with open(latest, "rb") as f:
                            engine._models[seed].append(pickle.load(f))
                        n_loaded += 1
                        logger.debug("  Loaded %s", latest.name)
                    else:
                        engine._models[seed].append(None)

            if n_loaded == 0:
                raise FileNotFoundError(
                    f"No production models found in {path}. "
                    f"Train with train_production_models() first."
                )
            logger.info("Loaded %d production models from %s (auto-latest)", n_loaded, path)

        else:
            # ── Backtest mode: load all folds ──
            engine._n_folds = meta.get("n_folds", 0)
            engine._models = {seed: [] for seed in engine.seeds}
            n_loaded = 0

            for seed in engine.seeds:
                for fold_idx in range(engine._n_folds):
                    model_path = path / f"model_s{seed}_f{fold_idx}.pkl"
                    if model_path.exists():
                        with open(model_path, "rb") as f:
                            engine._models[seed].append(pickle.load(f))
                        n_loaded += 1
                    else:
                        engine._models[seed].append(None)

            logger.info("Loaded %d models from %s | %d features | %d folds | seeds=%s",
                       n_loaded, path, len(engine._feature_cols),
                       engine._n_folds, engine.seeds)

        # ── Load feature importance if available ──
        imp_path = path / "feature_importance.csv"
        if imp_path.exists():
            engine._feature_importance = pd.read_csv(imp_path, index_col=0)

        engine._fitted = True
        return engine

    # ═══════════════════════════════════════════════════════
    # Accessors
    # ═══════════════════════════════════════════════════════

    @property
    def feature_cols(self) -> List[str]:
        """Feature columns used during training (read-only)."""
        return list(self._feature_cols)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_models(self) -> int:
        """Total number of trained models across all seeds and folds."""
        return sum(1 for models in self._models.values() for m in models if m is not None)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return self._feature_importance

    def get_oos_predictions(self) -> pd.DataFrame:
        """Return OOS ensemble predictions from walk-forward training."""
        if hasattr(self, "_oos_predictions"):
            return self._oos_predictions.copy()
        return pd.DataFrame()

    # ═══════════════════════════════════════════════════════
    # Internal: preprocessing pipeline (same as V7)
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _filter_dates(panel, date_col, min_stocks):
        counts = panel.groupby(date_col).size()
        good = counts[counts >= min_stocks].index
        return panel[panel[date_col].isin(good)].copy()

    def _prepare_labels(self, panel, date_col, symbol_col, close_col):
        df = panel.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)

        if close_col not in df.columns:
            for c in df.columns:
                if "收" in str(c) or "close" in str(c).lower():
                    close_col = c; break
            else:
                raise KeyError("Cannot find close price column")

        ret_col = "forward_return_1m"
        if ret_col not in df.columns:
            df[ret_col] = df.groupby(symbol_col)[close_col].transform(
                lambda x: x.shift(-1) / x - 1.0)

        df["label"] = df.groupby(date_col)[ret_col].rank(
            pct=True, na_option="bottom").fillna(0.5)

        n = df["label"].notna().sum()
        logger.info("Labels: forward_return_1m -> rank [0,1] | valid: %d (%.1f%%)",
                     n, 100 * n / max(len(df), 1))
        return df

    def _prepare_features(self, panel, blended, date_col, symbol_col):
        df = panel.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        factor_cols = [c for c in df.columns if c.endswith("_neutral_z")]
        if not factor_cols:
            raise ValueError("No _neutral_z factor columns found.")

        rank_cols = []
        if self.config.feature_method == "rank":
            for col in factor_cols:
                rc = f"{col}_rank"
                df[rc] = df.groupby(date_col)[col].rank(
                    pct=True, na_option="bottom").fillna(0.5)
                rank_cols.append(rc)
        else:
            rank_cols = factor_cols
            for col in factor_cols:
                df[col] = df[col].fillna(0.0)

        self._feature_cols = list(rank_cols)
        logger.info("Features: %d factors -> %d rank cols", len(factor_cols), len(rank_cols))

        # prev_signal anchor
        df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)
        if blended is not None and "alpha_signal" in blended.columns:
            df = df.merge(
                blended[[date_col, symbol_col, "alpha_signal"]],
                on=[date_col, symbol_col], how="left")
            df["prev_signal"] = df.groupby(symbol_col)["alpha_signal"].transform(
                lambda x: x.shift(1))
        else:
            logger.warning("No alpha_signal in blended — prev_signal filled with 0.5")
            df["prev_signal"] = np.nan

        df["prev_signal"] = df["prev_signal"].fillna(
            df["label"] if "label" in df.columns else 0.5)
        return df


# ═══════════════════════════════════════════════════════════
# Top-level convenience function
# ═══════════════════════════════════════════════════════════

def train_and_save(
    panel_path: str = "output/preprocessed.parquet",
    blended_path: str = "output/split_universe_blended.parquet",
    output_dir: str = "output/production_models",
    seeds: List[int] | None = None,
    *,
    mode: str = "backtest",
    current_date: str | None = None,
) -> ProductionAlphaEngine:
    """
    One-shot: load data, train, save models, return engine.

    Two modes:
      - mode="backtest":      Walk-forward CV across all folds → saves ALL folds.
                              For offline research and historical analysis.
      - mode="production":    Single-window training ending at current_date.
                              Saves ONLY the latest fold with timestamped names.
                              For monthly retraining pipeline.

    Parameters
    ----------
    panel_path, blended_path : str
        Paths to preprocessed data.
    output_dir : str
        Directory for model persistence.
    seeds : List[int], optional
        Ensemble seeds. Default: [42, 888, 2026].
    mode : str
        "backtest" or "production".
    current_date : str, optional
        "YYYY-MM-DD" training window end. Required for mode="production".

    Returns
    -------
    ProductionAlphaEngine
        Fitted engine.
    """
    panel = pd.read_parquet(panel_path)
    blended = pd.read_parquet(blended_path)

    config = ProductionConfig(seeds=seeds or [42, 888, 2026])
    engine = ProductionAlphaEngine(config)

    if mode == "production":
        if current_date is None:
            # Default: use latest date in panel
            current_date = str(panel["date"].max())
            logger.info("Auto current_date = %s", current_date)
        engine.train_production_models(panel, blended, current_date=current_date)
        engine.save_models(output_dir, mode="production")
    elif mode == "backtest":
        engine.fit(panel, blended)
        engine.save_models(output_dir, mode="backtest")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    logger.info("Models saved to %s (mode=%s)", output_dir, mode)
    return engine
