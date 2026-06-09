# ═══════════════════════════════════════════════════════════════
# ARCHIVED: Label Blending + Time-Decay 实验 (Stage 6)
# 已被 V7 (纯 1M labels) 取代 — V6 的混合标签未显著超越 V7
# 保留作为特征工程消融记录 — 不建议用于新工作
# ═══════════════════════════════════════════════════════════════
"""
LightGBM V6 Alpha Engine — Label Blending + Time-Decay Sample Weighting.

V6 builds on V5's Turnover-Aware Custom Objective by adding two innovations:

1. **Label Blending (Target Engineering)**
   y_target = 0.4 * forward_return_1m + 0.6 * forward_return_3m
   → Cross-sectional rank to [0,1]
   Injects short-term "crisis sensitivity" into the label while retaining
   the 3M horizon's endogenous turnover reduction.

2. **Time-Decay Sample Weighting**
   w_i = exp(-Δt * ln(2) / H)   where H = 12 months (half-life)
   → Passed via lgb.Dataset(..., weight=sample_weights)
   Recent samples weighted higher; stale history decays exponentially.

Inherited from V5:
  - Turnover-Aware Custom Objective (λ = 2.0)
  - 3M Gap for walk-forward CV (prevents label leakage through 3M component)
  - prev_signal = alpha_signal_{t-1} (linear baseline anchor via closure)
  - subsample = 1.0 (closure alignment)

Expected improvement over V5:
  - MaxDD: -27.12% → target -20% or better (blended label adds short-term awareness)
  - Sharpe: 0.9527 → target 1.0+ (time-decay improves recency relevance)
  - Turnover: maintained at 12-15% (λ=2.0 penalty preserved)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_engine_v6")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════


@dataclass
class MLConfigV6:
    """
    V6 Alpha Engine 配置 — Label Blending + Time-Decay Weighting.

    Walk-Forward + Gap 参数
    -----------------------
    train_months : int = 36
        训练窗口 (月数)。
    val_months : int = 6
        验证窗口 (月数)。
    test_months : int = 1
        OOS 预测窗口 (月数)。
    label_horizon : int = 3
        Gap 长度。虽然是 Blended Label (含 1M), 但 3M 分量需要 3M gap。

    Label Blending 参数
    -------------------
    blend_alpha : float = 0.4
        1M return 在混合标签中的权重。
        y_target = blend_alpha * ret_1m + (1 - blend_alpha) * ret_3m

    Time-Decay 参数
    ---------------
    half_life : int = 12
        指数衰减半衰期 (月)。H=12 意味着 12 个月前的样本权重为 0.5。

    Turnover 惩罚参数
    -----------------
    lambda_turnover : float = 2.0
        换手惩罚系数 λ (V5 最优值)。

    LightGBM 超参数
    ----------------
    subsample 固定 1.0 (闭包对齐要求)。
    """

    # Walk-Forward + Gap
    train_months: int = 36
    val_months: int = 6
    test_months: int = 1
    label_horizon: int = 3
    min_stocks_per_date: int = 50

    # Label Blending
    blend_alpha: float = 0.4  # weight of 1M return

    # Time-Decay
    half_life: int = 12       # half-life in months

    # Turnover penalty
    lambda_turnover: float = 2.0  # V5 optimal value

    # Label / Feature
    label_method: str = "rank"
    feature_method: str = "rank"

    # LightGBM
    objective: str = "regression"
    metric: str = "l2"
    boosting: str = "gbdt"
    num_leaves: int = 24
    max_depth: int = 4
    learning_rate: float = 0.02
    n_estimators: int = 2000
    subsample: float = 1.0        # ← closure alignment
    colsample_bytree: float = 0.70
    subsample_freq: int = 1
    min_child_samples: int = 100
    reg_alpha: float = 0.10
    reg_lambda: float = 0.10
    early_stopping_rounds: int = 50
    verbose: int = -1
    random_state: int = 42
    n_jobs: int = -1

    def to_lgb_params(self) -> dict:
        return {
            "objective": self.objective,
            "metric": self.metric,
            "boosting": self.boosting,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "subsample_freq": self.subsample_freq,
            "min_child_samples": self.min_child_samples,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "verbose": self.verbose,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


# ═══════════════════════════════════════════════════════════
# Custom Objective: Turnover-Aware L2 Loss (inherited from V5)
# ═══════════════════════════════════════════════════════════


def make_turnover_objective(
    prev_signal: np.ndarray,
    lambda_penalty: float,
) -> callable:
    """
    Factory: create LightGBM-compatible turnover-aware custom loss.

    L  = 0.5*(pred - y)^2  +  lambda*0.5*(pred - prev)^2
    g  = (pred - y)        +  lambda*(pred - prev)
    h  = 1 + lambda

    Closure captures prev_signal array; LightGBM calls fobj(preds, train_data)
    with preds in Dataset sample order, so array alignment is preserved.

    Prerequisites:
      - subsample=1.0 (no row subsampling that would break alignment)
      - prev_signal pre-filled for NaN (first-occurrence stocks)
    """
    _prev = np.asarray(prev_signal, dtype=np.float64)
    _lam = float(lambda_penalty)

    def _objective(preds: np.ndarray, train_data) -> tuple[np.ndarray, np.ndarray]:
        labels = train_data.get_label().astype(np.float64)

        if len(preds) != len(_prev):
            # Safety fallback: pure L2 (should not happen with subsample=1.0)
            grad = preds - labels
            hess = np.ones_like(preds)
            return grad, hess

        residual = preds - labels          # d(0.5*(p-y)^2)/dp = p - y
        turnover = preds - _prev           # d(0.5*(p-prev)^2)/dp = p - prev
        grad = residual + _lam * turnover  # g = (p-y) + lambda*(p-prev)
        hess = np.full_like(preds, 1.0 + _lam, dtype=np.float64)

        return grad, hess

    return _objective


def make_l2_eval_metric() -> callable:
    """Standard L2 eval metric (no turnover penalty in validation)."""
    def _l2_metric(preds: np.ndarray, train_data) -> tuple[str, float, bool]:
        labels = train_data.get_label()
        mse = np.mean((preds - labels) ** 2)
        return "l2", mse, False
    return _l2_metric


# ═══════════════════════════════════════════════════════════
# Time-Decay Weight Computation
# ═══════════════════════════════════════════════════════════


def compute_time_decay_weights(
    train_dates: pd.Series,
    test_date,
    all_dates: list,
    half_life: int = 12,
) -> np.ndarray:
    """
    Compute exponential time-decay sample weights for a training fold.

    w_i = exp(-Δt * ln(2) / H)

    where:
      Δt = number of date-index positions from sample to test date
      H  = half-life in months (default 12)

    Parameters
    ----------
    train_dates : pd.Series
        Date values for each training sample.
    test_date :
        The prediction (test) date.
    all_dates : list
        Complete sorted list of all dates in the dataset.
    half_life : int
        Half-life in months (default 12).

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Sample weights ∈ (0, 1].
    """
    # Build date-to-index mapping
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(all_dates)}
    test_dt = pd.Timestamp(test_date)
    test_idx = date_to_idx.get(test_dt, len(all_dates) - 1)

    # Compute Δt for each sample (in date-index units ≈ months)
    delta_t = train_dates.apply(
        lambda d: max(0, test_idx - date_to_idx.get(pd.Timestamp(d), test_idx))
    )

    # Exponential decay
    decay_factor = np.log(2) / half_life
    weights = np.exp(-delta_t.values.astype(np.float64) * decay_factor)

    return weights


# ═══════════════════════════════════════════════════════════
# 主引擎 V6
# ═══════════════════════════════════════════════════════════


class LightGBMAlphaEngineV6:
    """
    V6 Alpha Engine: Label Blending + Time-Decay + Turnover-Aware Objective.

    Architecture:
      ┌─────────────────────────────────────────────────────────┐
      │  Panel (date x symbol, 16 factors + close price)         │
      │  + Blended (alpha_signal, universe, mcap_est)            │
      │  ↓                                                       │
      │  [Label Blending]                                         │
      │    forward_return_1m = close_{t+1}/close_t - 1            │
      │    forward_return_3m = close_{t+3}/close_t - 1            │
      │    y_target = 0.4*ret_1m + 0.6*ret_3m → rank [0,1]      │
      │  ↓                                                       │
      │  [Feature Engineering] (same as V5)                       │
      │    Rank features + prev_signal = alpha_signal_{t-1}      │
      │  ↓                                                       │
      │  [Walk-Forward + 3M Gap Loop]                             │
      │    Per fold:                                              │
      │      a) Compute time-decay weights for train samples      │
      │      b) Build lgb.Dataset with weight=...                 │
      │      c) Closure-inject prev_signal → custom objective     │
      │      d) Train + OOS predict                               │
      │  ↓                                                       │
      │  [Concatenate OOS predictions] → v6_ml_signal DataFrame   │
      └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: Optional[MLConfigV6] = None):
        self.config = config or MLConfigV6()
        self._feature_cols: list[str] = []
        self._trained_folds: int = 0
        self._feature_importance: pd.DataFrame | None = None

    # ═══════════════════════════════════════════════════════
    # 标签工程: Blended Label (1M + 3M) → rank
    # ═══════════════════════════════════════════════════════

    def prepare_labels(
        self,
        panel: pd.DataFrame,
        date_col: str = "date",
        symbol_col: str = "symbol",
        close_col: str = "收盘",
    ) -> pd.DataFrame:
        """
        Compute BLENDED forward return label.

        y_target = blend_alpha * forward_return_1m
                 + (1 - blend_alpha) * forward_return_3m

        Then cross-sectional rank to [0, 1] for regression.

        Why blended?
        - 1M component injects short-term crisis sensitivity
        - 3M component retains endogenous turnover smoothing
        - A stock that crashes in month 1 but recovers by month 3 will
          have a lower blended score than one that rises steadily —
          penalizing "dead cat bounce" stocks.

        Parameters
        ----------
        panel : pd.DataFrame
            Must contain close_col (close price).
        date_col, symbol_col, close_col : str

        Returns
        -------
        pd.DataFrame: panel + "label" column + intermediate return columns.
        """
        df = panel.copy()
        cfg = self.config
        alpha = cfg.blend_alpha

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)

        # Locate close price column
        if close_col not in df.columns:
            for c in df.columns:
                if "收" in str(c) or "close" in str(c).lower():
                    close_col = c
                    break
            else:
                raise KeyError("Cannot find close price column for return computation")

        # ── Compute forward returns ──
        ret_1m_col = "forward_return_1m"
        ret_3m_col = "forward_return_3m"

        if ret_1m_col not in df.columns:
            logger.info("  Computing forward_return_1m (shift -1) ...")
            df[ret_1m_col] = (
                df.groupby(symbol_col)[close_col]
                .transform(lambda x: x.shift(-1) / x - 1.0)
            )

        if ret_3m_col not in df.columns:
            logger.info("  Computing forward_return_3m (shift -3) ...")
            df[ret_3m_col] = (
                df.groupby(symbol_col)[close_col]
                .transform(lambda x: x.shift(-3) / x - 1.0)
            )

        # ── Blended target ──
        blend_col = "blended_return"
        df[blend_col] = (
            alpha * df[ret_1m_col].fillna(0.0)
            + (1.0 - alpha) * df[ret_3m_col].fillna(0.0)
        )

        # ── Cross-sectional rank → [0, 1] ──
        label_col = "label"
        df[label_col] = (
            df.groupby(date_col)[blend_col]
            .rank(pct=True, na_option="bottom")
            .fillna(0.5)
        )

        # Report
        n_1m = df[ret_1m_col].notna().sum()
        n_3m = df[ret_3m_col].notna().sum()
        n_label = df[label_col].notna().sum()
        logger.info(
            "Label Blending: %.0f%% * ret_1m + %.0f%% * ret_3m -> rank [0,1]",
            alpha * 100, (1 - alpha) * 100,
        )
        logger.info(
            "  ret_1m valid: %d (%.1f%%), ret_3m valid: %d (%.1f%%), "
            "label valid: %d (%.1f%%)",
            n_1m, 100 * n_1m / len(df) if len(df) > 0 else 0,
            n_3m, 100 * n_3m / len(df) if len(df) > 0 else 0,
            n_label, 100 * n_label / len(df) if len(df) > 0 else 0,
        )

        return df

    # ═══════════════════════════════════════════════════════
    # 特征工程: Rank + prev_signal anchor (same as V5)
    # ═══════════════════════════════════════════════════════

    def prepare_features(
        self,
        panel: pd.DataFrame,
        blended: Optional[pd.DataFrame] = None,
        date_col: str = "date",
        symbol_col: str = "symbol",
    ) -> pd.DataFrame:
        """
        Feature engineering (same as V5):
          1. Cross-sectional rank features (16 factors → [0,1])
          2. Anchor: prev_signal = alpha_signal_{t-1} (NOT used as tree feature)

        prev_signal is passed via closure to the custom objective for
        turnover penalty computation — it does NOT participate in tree splits.
        """
        df = panel.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        cfg = self.config

        # Discover factor columns
        factor_cols = [c for c in df.columns if c.endswith("_neutral_z")]
        if not factor_cols:
            raise ValueError("No _neutral_z factor columns found.")

        logger.info("Feature Engineering: %d raw factors", len(factor_cols))

        # ── Step 1: Cross-sectional Rank ──
        rank_cols = []
        if cfg.feature_method == "rank":
            for col in factor_cols:
                rank_col = f"{col}_rank"
                df[rank_col] = (
                    df.groupby(date_col)[col]
                    .rank(pct=True, na_option="bottom")
                    .fillna(0.5)
                )
                rank_cols.append(rank_col)
            logger.info("  Step 1: %d factors -> cross-sectional rank", len(rank_cols))
        else:
            for col in factor_cols:
                df[col] = df[col].fillna(0.0)
            rank_cols = factor_cols
            logger.info("  Step 1: %d factors -> z-score (no rank)", len(rank_cols))

        self._feature_cols = list(rank_cols)

        # ── Step 2: prev_signal anchor ──
        df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)

        if blended is not None and "alpha_signal" in blended.columns:
            df = df.merge(
                blended[[date_col, symbol_col, "alpha_signal"]],
                on=[date_col, symbol_col], how="left",
            )
            df["prev_signal"] = (
                df.groupby(symbol_col)["alpha_signal"]
                .transform(lambda x: x.shift(1))
            )
        else:
            logger.warning(
                "blended data / alpha_signal missing; "
                "prev_signal filled with 0.5 (turnover penalty degrades to L2)"
            )
            df["prev_signal"] = np.nan

        # Fill NaN prev_signal (first occurrence stocks)
        if "label" in df.columns:
            df["prev_signal"] = df["prev_signal"].fillna(df["label"])
        else:
            df["prev_signal"] = df["prev_signal"].fillna(0.5)

        n_valid = df["prev_signal"].notna().sum()
        logger.info(
            "  Step 2: prev_signal = alpha_signal_{t-1} | valid: %d/%d (%.1f%%)",
            n_valid, len(df),
            100 * n_valid / len(df) if len(df) > 0 else 0,
        )

        logger.info("Feature Engineering done: %d feature cols + prev_signal (anchor)",
                     len(self._feature_cols))
        return df

    # ═══════════════════════════════════════════════════════
    # Walk-Forward + Time-Decay + Turnover-Aware Training
    # ═══════════════════════════════════════════════════════

    def walk_forward_train(
        self,
        panel: pd.DataFrame,
        date_col: str = "date",
        symbol_col: str = "symbol",
    ) -> pd.DataFrame:
        """
        Walk-Forward rolling training with Time-Decay weights
        and Turnover-Aware Custom Objective.

        Key enhancements over V5:
        1. Time-decay sample weights: recent samples weighted higher
           w_i = exp(-Δt * ln(2) / half_life)
           Injected via lgb.Dataset(..., weight=sample_weights)
        2. Blended label already baked into panel['label'] by prepare_labels()

        Gap design (H=3, inherited from V5):
          |← Train labels (33M) →| Gap 3M |← Val (3M) →| Gap 3M | Test |

        Returns
        -------
        pd.DataFrame: date, symbol, v6_ml_signal
        """
        import lightgbm as lgb

        df = panel.copy()
        dates = sorted(df[date_col].unique())
        n_dates = len(dates)

        cfg = self.config
        H = cfg.label_horizon
        fold_size = cfg.train_months + cfg.val_months + cfg.test_months
        n_folds = n_dates - fold_size - H + 1

        if n_folds <= 0:
            raise ValueError(
                f"Insufficient data: {n_dates} dates < required "
                f"{fold_size + H} "
                f"(train={cfg.train_months}+val={cfg.val_months}"
                f"+test={cfg.test_months}+horizon={H})"
            )

        logger.info("=" * 60)
        logger.info(
            "V6 Walk-Forward: %d folds | lambda=%.2f | half_life=%dM | "
            "blend=%.0f%%_1M+%.0f%%_3M | Gap=%dM",
            n_folds, cfg.lambda_turnover, cfg.half_life,
            cfg.blend_alpha * 100, (1 - cfg.blend_alpha) * 100, H,
        )
        logger.info(
            "  Window: %dM train + %dM val + %dM test | horizon=%dM",
            cfg.train_months, cfg.val_months, cfg.test_months, H,
        )
        logger.info(
            "  Features: %d cols | objective=regression | "
            "loss=L2+%.2f*Turnover^2 | subsample=%.2f",
            len(self._feature_cols), cfg.lambda_turnover, cfg.subsample,
        )
        logger.info("=" * 60)

        all_predictions: list[pd.DataFrame] = []
        importance_list: list[pd.DataFrame] = []
        self._trained_folds = 0

        for fold_idx in range(n_folds):
            # ── Time indices (with 3M Gap) ──
            train_start = fold_idx
            train_end_raw = train_start + cfg.train_months          # e.g. idx 36
            train_end_label = train_end_raw - H                     # e.g. idx 33
            val_start = train_end_raw                               # idx 36
            val_end_raw = val_start + cfg.val_months                # idx 42
            val_end_label = val_end_raw - H                         # idx 39
            test_idx = val_end_raw                                  # idx 42

            train_label_dates = set(dates[train_start:train_end_label])
            val_label_dates = set(dates[val_start:val_end_label])
            test_date = dates[test_idx]

            # All data in this fold (including Gap period features)
            fold_data_dates = set(dates[train_start:test_idx + 1])
            fold_df = df[df[date_col].isin(fold_data_dates)].copy()

            # ── Split train/val/test ──
            train_mask = fold_df[date_col].isin(train_label_dates)
            val_mask = fold_df[date_col].isin(val_label_dates)
            test_mask = fold_df[date_col] == test_date

            # Feature columns (exclude prev_signal — not a tree feature)
            feature_cols = [c for c in self._feature_cols if c in fold_df.columns]

            # ── Extract prev_signal anchor (MUST extract before dropping!) ──
            if "prev_signal" in fold_df.columns:
                prev_train = fold_df.loc[train_mask, "prev_signal"].values.astype(np.float64)
                prev_val = fold_df.loc[val_mask, "prev_signal"].values.astype(np.float64)
            else:
                logger.warning("Fold %d: prev_signal missing, falling back to pure L2", fold_idx)
                prev_train = fold_df.loc[train_mask, "label"].values.astype(np.float64)
                prev_val = fold_df.loc[val_mask, "label"].values.astype(np.float64)

            # ── Extract data (without prev_signal) ──
            X_train = fold_df.loc[train_mask, feature_cols].astype(float)
            y_train = fold_df.loc[train_mask, "label"].astype(float)
            X_val = fold_df.loc[val_mask, feature_cols].astype(float)
            y_val = fold_df.loc[val_mask, "label"].astype(float)
            X_test = fold_df.loc[test_mask, feature_cols].astype(float)

            # ── Sample size check ──
            if len(X_train) < 500 or len(X_val) < 30:
                logger.warning(
                    "Fold %d: insufficient samples (train=%d, val=%d), skipping",
                    fold_idx, len(X_train), len(X_val),
                )
                continue

            # ── NEW: Time-Decay Sample Weights ──
            train_dates_series = fold_df.loc[train_mask, date_col]
            sample_weights = compute_time_decay_weights(
                train_dates=train_dates_series,
                test_date=test_date,
                all_dates=dates,
                half_life=cfg.half_life,
            )

            # ── Build LightGBM Datasets ──
            # CRITICAL: weight parameter injects time-decay weights
            train_ds = lgb.Dataset(
                X_train, label=y_train,
                weight=sample_weights,
            )
            val_ds = lgb.Dataset(
                X_val, label=y_val,
                reference=train_ds,
            )

            # ── Build Custom Objective (closure captures prev_signal) ──
            fobj = make_turnover_objective(prev_train, cfg.lambda_turnover)
            feval = make_l2_eval_metric()

            # ── Train ──
            params = cfg.to_lgb_params()
            # LightGBM 4.x: custom objective via params['objective'], NOT fobj kwarg
            params["objective"] = fobj

            try:
                model = lgb.train(
                    params=params,
                    train_set=train_ds,
                    num_boost_round=cfg.n_estimators,
                    valid_sets=[train_ds, val_ds],
                    valid_names=["train", "val"],
                    feval=feval,
                    callbacks=[
                        lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )

                # OOS prediction
                y_pred = model.predict(X_test)
                n_iter = model.best_iteration

                # ── Save prediction ──
                pred_df = fold_df.loc[test_mask, [date_col, symbol_col]].copy()
                pred_df["v6_ml_signal"] = y_pred
                all_predictions.append(pred_df)

                # ── Feature importance ──
                imp = pd.DataFrame({
                    "feature": feature_cols,
                    "gain": model.feature_importance(importance_type="gain"),
                    "split": model.feature_importance(importance_type="split"),
                })
                imp["fold"] = fold_idx
                importance_list.append(imp)

                self._trained_folds += 1

                if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
                    train_preds = model.predict(X_train)
                    avg_to_penalty = np.mean((train_preds - prev_train) ** 2)
                    avg_weight = np.mean(sample_weights)
                    min_weight = np.min(sample_weights)
                    logger.info(
                        "  Fold %3d/%d | test=%s | train=%d val=%d test=%d | "
                        "iter=%d | avg_w=%.3f min_w=%.3f | avg_dpred^2=%.4f",
                        fold_idx + 1, n_folds, str(test_date)[:10],
                        len(X_train), len(X_val), len(X_test),
                        n_iter, avg_weight, min_weight, avg_to_penalty,
                    )

            except Exception as e:
                logger.error("Fold %d training failed: %s", fold_idx, e)
                continue

        if not all_predictions:
            raise RuntimeError("All folds failed to train. Check data and parameters.")

        # ── Concatenate OOS predictions ──
        predictions = pd.concat(all_predictions, ignore_index=True)
        predictions[date_col] = pd.to_datetime(predictions[date_col])

        # ── Aggregate feature importance ──
        if importance_list:
            imp_all = pd.concat(importance_list, ignore_index=True)
            self._feature_importance = (
                imp_all.groupby("feature")[["gain", "split"]]
                .mean()
                .sort_values("gain", ascending=False)
            )

        logger.info("=" * 60)
        logger.info(
            "V6 Walk-Forward complete: %d/%d folds succeeded, %d OOS predictions",
            self._trained_folds, n_folds, len(predictions),
        )
        logger.info(
            "OOS dates: %s ~ %s | %d cross-sections | %d unique stocks",
            predictions[date_col].min().strftime("%Y-%m-%d"),
            predictions[date_col].max().strftime("%Y-%m-%d"),
            predictions[date_col].nunique(),
            predictions[symbol_col].nunique(),
        )
        if self._feature_importance is not None:
            logger.info("Top-5 features (by gain):")
            for feat, row in self._feature_importance.head(5).iterrows():
                logger.info("  %s: gain=%.2f split=%.2f", feat, row["gain"], row["split"])
        logger.info("=" * 60)

        return predictions

    # ═══════════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════════

    def run(
        self,
        panel: pd.DataFrame,
        *,
        blended: Optional[pd.DataFrame] = None,
        date_col: str = "date",
        symbol_col: str = "symbol",
        close_col: str = "收盘",
    ) -> pd.DataFrame:
        """
        One-shot V6 ML Pipeline.

        Pipeline:
          1. _filter_populated_dates() — remove sparse mid-month dates
          2. prepare_labels()  — blended 1M+3M label -> rank
          3. prepare_features() — rank + prev_signal anchor
          4. walk_forward_train() — Gap WF + Time-Decay weights + Custom Objective

        Parameters
        ----------
        panel : pd.DataFrame
            Raw factor panel (with _neutral_z columns + close price).
        blended : pd.DataFrame | None
            Panel with alpha_signal (for prev_signal anchor construction).
        date_col, symbol_col, close_col : str

        Returns
        -------
        pd.DataFrame: date, symbol, v6_ml_signal
        """
        logger.info("=" * 64)
        logger.info("LightGBM V6 Engine — Pipeline Start")
        logger.info("  Label: %.0f%%*ret_1m + %.0f%%*ret_3m -> rank",
                     self.config.blend_alpha * 100,
                     (1 - self.config.blend_alpha) * 100)
        logger.info("  Time-Decay: half_life=%dM, weights via lgb.Dataset(weight=...)",
                     self.config.half_life)
        logger.info("  Objective: regression | lambda=%.2f | horizon=%dM | gap=%dM",
                     self.config.lambda_turnover,
                     self.config.label_horizon,
                     self.config.label_horizon)
        logger.info("  Loss = 0.5*(pred-y)^2 + %.2f*0.5*(pred-prev)^2",
                     self.config.lambda_turnover)
        logger.info("=" * 64)

        # Step 0: Filter sparse cross-sections
        df = self._filter_populated_dates(
            panel, date_col=date_col,
            min_stocks=self.config.min_stocks_per_date)

        # Step 1: Blended label (1M + 3M → rank)
        df = self.prepare_labels(
            df,
            date_col=date_col,
            symbol_col=symbol_col,
            close_col=close_col,
        )

        # Step 2: Rank features + prev_signal anchor
        df = self.prepare_features(
            df,
            blended=blended,
            date_col=date_col,
            symbol_col=symbol_col,
        )

        # Step 3: Walk-Forward + Time-Decay + Custom Objective
        predictions = self.walk_forward_train(
            df,
            date_col=date_col,
            symbol_col=symbol_col,
        )

        return predictions

    @staticmethod
    def _filter_populated_dates(
        panel: pd.DataFrame,
        date_col: str = "date",
        min_stocks: int = 50,
    ) -> pd.DataFrame:
        """
        Filter out sparse cross-sections (mid-month dates with 1-2 stocks).

        After filtering to >= min_stocks, shift(-3) ≈ 3 calendar months.
        """
        date_counts = panel.groupby(date_col).size()
        good_dates = date_counts[date_counts >= min_stocks].index
        filtered = panel[panel[date_col].isin(good_dates)].copy()
        n_before = panel[date_col].nunique()
        n_after = filtered[date_col].nunique()
        logger.info(
            "Step 0: Filter dates with >= %d stocks | %d -> %d dates (%.0f%% retained)",
            min_stocks, n_before, n_after,
            100 * n_after / n_before if n_before > 0 else 0,
        )
        return filtered

    # ═══════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════

    def get_feature_importance(self) -> pd.DataFrame | None:
        """Return aggregate feature importance."""
        return self._feature_importance

    def to_markdown_report(self) -> str:
        """Generate training report (Markdown)."""
        cfg = self.config
        lines = [
            "## V6 Alpha Engine — Training Report",
            "",
            f"- **Label:** {cfg.blend_alpha*100:.0f}% * ret_1m + "
            f"{(1-cfg.blend_alpha)*100:.0f}% * ret_3m -> rank [0,1]",
            f"- **Objective:** Custom L2 + lambda*(pred-prev)^2 | lambda = {cfg.lambda_turnover}",
            f"- **Time-Decay:** half_life = {cfg.half_life}M | "
            f"w = exp(-dt * ln(2) / {cfg.half_life})",
            f"- **Gap:** {cfg.label_horizon}M (prevents 3M leakage)",
            f"- **Trained Folds:** {self._trained_folds}",
            f"- **Feature Cols:** {len(self._feature_cols)}",
            f"- **Window:** {cfg.train_months}M train + "
            f"{cfg.val_months}M val + {cfg.test_months}M test",
            f"- **Hyperparams:** max_depth={cfg.max_depth}, "
            f"num_leaves={cfg.num_leaves}, lr={cfg.learning_rate}",
            f"- **subsample:** {cfg.subsample} (closure alignment)",
            "",
            "### Loss Function",
            "",
            "```",
            "L = 0.5*(pred - y)^2  +  lambda*0.5*(pred - prev)^2",
            "g = (pred - y)        +  lambda*(pred - prev)",
            "h = 1 + lambda",
            "```",
            "",
            f"where prev = alpha_signal lagged by 1 month (linear baseline anchor).",
            "",
            "### Label Construction",
            "",
            "```",
            f"y_target = {cfg.blend_alpha} * forward_return_1m "
            f"+ {1-cfg.blend_alpha} * forward_return_3m",
            "y_label = cross_sectional_rank(y_target)  # -> [0, 1]",
            "```",
            "",
            "### Time-Decay Weights",
            "",
            "```",
            f"w_i = exp(-dt * ln(2) / {cfg.half_life})",
            "```",
            "",
            "Injected via: lgb.Dataset(X, label=y, weight=sample_weights)",
            "",
        ]
        if self._feature_importance is not None:
            lines.append("### Feature Importance (Average Gain)")
            lines.append("")
            lines.append("| Rank | Feature | Gain | Split |")
            lines.append("|------|---------|------|-------|")
            for rank, (feat, row) in enumerate(
                self._feature_importance.iterrows(), 1
            ):
                lines.append(
                    f"| {rank} | {feat} | {row['gain']:.1f} | {row['split']:.1f} |"
                )
        return "\n".join(lines)
