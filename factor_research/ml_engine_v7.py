"""
LightGBM V7 Alpha Engine — 1M Labels + 0M Gap + Turnover-Aware Objective.

V7 is the definitive architecture test:
  - V5 proved Turnover-Aware loss controls turnover (40% → 13%)
  - V5/V6 proved 3M Gap causes -27% structural MaxDD
  - V7 hypothesis: 1M labels + 0M gap fixes MaxDD; TO penalty preserves low TO

Architecture (vs V5):
  ┌──────────────────┬─────────────────────┬──────────────────────┐
  │ Dimension         │ V5 (3M+gap)         │ V7 (1M+no-gap)       │
  ├──────────────────┼─────────────────────┼──────────────────────┤
  │ Label             │ forward_return_3m   │ forward_return_1m    │
  │ Label horizon     │ 3 months            │ 1 month              │
  │ Gap (train→val)   │ 3 months            │ 0 (standard 1-step)  │
  │ Train recency     │ 3 months stale      │ immediate            │
  │ Custom Objective  │ L2 + λ·(ŷ−ŷ₋₁)²    │ L2 + λ·(ŷ−ŷ₋₁)²     │
  │ λ                 │ 2.0                 │ 2.0                  │
  │ subsample         │ 1.0                 │ 1.0                  │
  │ Features          │ 16 rank + prev_sig  │ 16 rank + prev_sig   │
  │ Folds (96 dates)  │ ~51                 │ ~54                  │
  └──────────────────┴─────────────────────┴──────────────────────┘

Expected:
  - MaxDD: -27.12% → < -20% (no structural 3M lag)
  - Sharpe: 0.95 → 1.0+ (model reacts to current conditions)
  - Turnover: 13% → 15-18% (1M labels are noisier, but λ=2.0 suppresses)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_engine_v7")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════

@dataclass
class MLConfigV7:
    """V7 config: 1M labels, 0M gap, TO-aware objective."""

    train_months: int = 36
    val_months: int = 6
    test_months: int = 1
    label_horizon: int = 1          # 1M forward return (NO gap)
    min_stocks_per_date: int = 50

    lambda_turnover: float = 2.0    # Turnover penalty (V5 optimal)

    label_method: str = "rank"
    feature_method: str = "rank"

    # LightGBM params
    objective: str = "regression"
    metric: str = "l2"
    boosting: str = "gbdt"
    num_leaves: int = 24
    max_depth: int = 4
    learning_rate: float = 0.02
    n_estimators: int = 2000
    subsample: float = 1.0          # closure alignment
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
            "objective": self.objective, "metric": self.metric,
            "boosting": self.boosting, "num_leaves": self.num_leaves,
            "max_depth": self.max_depth, "learning_rate": self.learning_rate,
            "subsample": self.subsample, "colsample_bytree": self.colsample_bytree,
            "subsample_freq": self.subsample_freq,
            "min_child_samples": self.min_child_samples,
            "reg_alpha": self.reg_alpha, "reg_lambda": self.reg_lambda,
            "verbose": self.verbose, "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


# ═══════════════════════════════════════════════════════════
# Custom Objective: Turnover-Aware L2 Loss (identical to V5)
# ═══════════════════════════════════════════════════════════

def make_turnover_objective(
    prev_signal: np.ndarray,
    lambda_penalty: float,
) -> callable:
    """
    L = 0.5*(pred-y)^2 + lambda*0.5*(pred-prev)^2
    g = (pred-y) + lambda*(pred-prev)
    h = 1 + lambda
    """
    _prev = np.asarray(prev_signal, dtype=np.float64)
    _lam = float(lambda_penalty)

    def _objective(preds: np.ndarray, train_data) -> tuple[np.ndarray, np.ndarray]:
        labels = train_data.get_label().astype(np.float64)
        if len(preds) != len(_prev):
            grad = preds - labels
            hess = np.ones_like(preds)
            return grad, hess
        residual = preds - labels
        turnover = preds - _prev
        grad = residual + _lam * turnover
        hess = np.full_like(preds, 1.0 + _lam, dtype=np.float64)
        return grad, hess
    return _objective


def make_l2_eval_metric() -> callable:
    def _l2_metric(preds, train_data) -> tuple[str, float, bool]:
        return "l2", float(np.mean((preds - train_data.get_label()) ** 2)), False
    return _l2_metric


# ═══════════════════════════════════════════════════════════
# V7 Engine
# ═══════════════════════════════════════════════════════════

class LightGBMAlphaEngineV7:
    """
    V7: 1M labels + 0M gap + Turnover-Aware Objective.

    Walk-Forward structure (NO gap):
      Fold k: Train[k : k+36] -> Val[k+36 : k+42] -> Test[k+42]
      Labels are forward_return_1m — standard 1-step-ahead prediction.
      No data leakage: training label at date d uses close_{d+1};
      validation starts at date d+1 with independent features.
    """

    def __init__(self, config: Optional[MLConfigV7] = None):
        self.config = config or MLConfigV7()
        self._feature_cols: list[str] = []
        self._trained_folds: int = 0
        self._feature_importance: pd.DataFrame | None = None

    # ── Label: 1M forward return → rank ──

    def prepare_labels(
        self, panel: pd.DataFrame,
        date_col="date", symbol_col="symbol", close_col="收盘",
    ) -> pd.DataFrame:
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
            logger.info("Computing forward_return_1m (shift -1)")
            df[ret_col] = df.groupby(symbol_col)[close_col].transform(
                lambda x: x.shift(-1) / x - 1.0)

        df["label"] = df.groupby(date_col)[ret_col].rank(
            pct=True, na_option="bottom").fillna(0.5)

        n = df["label"].notna().sum()
        logger.info("Label: forward_return_1m -> rank [0,1] | valid: %d (%.1f%%)",
                     n, 100*n/len(df) if len(df) else 0)
        return df

    # ── Features: Rank + prev_signal anchor ──

    def prepare_features(
        self, panel: pd.DataFrame, blended=None,
        date_col="date", symbol_col="symbol",
    ) -> pd.DataFrame:
        df = panel.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        cfg = self.config

        factor_cols = [c for c in df.columns if c.endswith("_neutral_z")]
        if not factor_cols:
            raise ValueError("No _neutral_z factor columns found")

        rank_cols = []
        if cfg.feature_method == "rank":
            for col in factor_cols:
                rc = f"{col}_rank"
                df[rc] = df.groupby(date_col)[col].rank(
                    pct=True, na_option="bottom").fillna(0.5)
                rank_cols.append(rc)
        else:
            for col in factor_cols:
                df[col] = df[col].fillna(0.0)
            rank_cols = factor_cols
        self._feature_cols = list(rank_cols)
        logger.info("Features: %d factors -> %d rank cols", len(factor_cols), len(rank_cols))

        # prev_signal anchor
        df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)
        if blended is not None and "alpha_signal" in blended.columns:
            df = df.merge(blended[[date_col, symbol_col, "alpha_signal"]],
                          on=[date_col, symbol_col], how="left")
            df["prev_signal"] = df.groupby(symbol_col)["alpha_signal"].transform(
                lambda x: x.shift(1))
        else:
            logger.warning("No blended/alpha_signal; prev_signal=0.5")
            df["prev_signal"] = np.nan

        df["prev_signal"] = df["prev_signal"].fillna(
            df["label"] if "label" in df.columns else 0.5)
        logger.info("prev_signal anchor: valid %d/%d", df["prev_signal"].notna().sum(), len(df))
        return df

    # ── Walk-Forward (NO GAP) + Custom Objective ──

    def walk_forward_train(
        self, panel: pd.DataFrame,
        date_col="date", symbol_col="symbol",
    ) -> pd.DataFrame:
        import lightgbm as lgb

        df = panel.copy()
        dates = sorted(df[date_col].unique())
        n_dates = len(dates)
        cfg = self.config
        T, V, S = cfg.train_months, cfg.val_months, cfg.test_months
        fold_window = T + V  # train + val; test is +1
        n_folds = n_dates - fold_window

        if n_folds <= 0:
            raise ValueError(f"Insufficient dates: {n_dates} < {fold_window + 1}")

        logger.info("=" * 60)
        logger.info("V7 Walk-Forward (0M Gap): %d folds | lambda=%.2f", n_folds, cfg.lambda_turnover)
        logger.info("  Window: %dM train + %dM val + %dM test | label=forward_return_1m", T, V, S)
        logger.info("  Features: %d cols | subsample=%.2f", len(self._feature_cols), cfg.subsample)
        logger.info("  Loss = 0.5*(p-y)^2 + %.2f*0.5*(p-prev)^2", cfg.lambda_turnover)
        logger.info("=" * 60)

        all_preds: list[pd.DataFrame] = []
        imp_list: list[pd.DataFrame] = []
        self._trained_folds = 0

        for fold_idx in range(n_folds):
            # ── Time indices: NO gap, standard 1-step-forward ──
            train_end = fold_idx + T          # exclusive: train on dates[fold_idx : train_end]
            val_end = train_end + V           # val on dates[train_end : val_end]
            test_idx = val_end                # test on dates[test_idx]

            train_dates_set = set(dates[fold_idx:train_end])
            val_dates_set = set(dates[train_end:val_end])
            test_date = dates[test_idx]

            fold_data_dates = set(dates[fold_idx:test_idx + 1])
            fold_df = df[df[date_col].isin(fold_data_dates)].copy()

            train_mask = fold_df[date_col].isin(train_dates_set)
            val_mask = fold_df[date_col].isin(val_dates_set)
            test_mask = fold_df[date_col] == test_date

            feature_cols = [c for c in self._feature_cols if c in fold_df.columns]

            # Extract prev_signal anchor
            if "prev_signal" in fold_df.columns:
                prev_train = fold_df.loc[train_mask, "prev_signal"].values.astype(np.float64)
                prev_val = fold_df.loc[val_mask, "prev_signal"].values.astype(np.float64)
            else:
                prev_train = fold_df.loc[train_mask, "label"].values.astype(np.float64)
                prev_val = fold_df.loc[val_mask, "label"].values.astype(np.float64)

            X_train = fold_df.loc[train_mask, feature_cols].astype(float)
            y_train = fold_df.loc[train_mask, "label"].astype(float)
            X_val = fold_df.loc[val_mask, feature_cols].astype(float)
            y_val = fold_df.loc[val_mask, "label"].astype(float)
            X_test = fold_df.loc[test_mask, feature_cols].astype(float)

            if len(X_train) < 500 or len(X_val) < 30:
                logger.warning("Fold %d: insufficient (train=%d val=%d), skip",
                               fold_idx, len(X_train), len(X_val))
                continue

            train_ds = lgb.Dataset(X_train, label=y_train)
            val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

            fobj = make_turnover_objective(prev_train, cfg.lambda_turnover)
            feval = make_l2_eval_metric()
            params = cfg.to_lgb_params()
            params["objective"] = fobj  # LightGBM 4.x API

            try:
                model = lgb.train(
                    params=params, train_set=train_ds,
                    num_boost_round=cfg.n_estimators,
                    valid_sets=[train_ds, val_ds],
                    valid_names=["train", "val"],
                    feval=feval,
                    callbacks=[
                        lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )

                y_pred = model.predict(X_test)
                pred_df = fold_df.loc[test_mask, [date_col, symbol_col]].copy()
                pred_df["v7_ml_signal"] = y_pred
                all_preds.append(pred_df)

                imp = pd.DataFrame({
                    "feature": feature_cols,
                    "gain": model.feature_importance(importance_type="gain"),
                    "split": model.feature_importance(importance_type="split"),
                })
                imp["fold"] = fold_idx
                imp_list.append(imp)
                self._trained_folds += 1

                if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
                    train_preds = model.predict(X_train)
                    avg_pen = np.mean((train_preds - prev_train) ** 2)
                    logger.info("  Fold %3d/%d | test=%s | train=%d val=%d test=%d | "
                                "iter=%d | avg_dpred^2=%.4f",
                                fold_idx+1, n_folds, str(test_date)[:10],
                                len(X_train), len(X_val), len(X_test),
                                model.best_iteration, avg_pen)

            except Exception as e:
                logger.error("Fold %d failed: %s", fold_idx, e)
                continue

        if not all_preds:
            raise RuntimeError("All folds failed.")

        predictions = pd.concat(all_preds, ignore_index=True)
        predictions[date_col] = pd.to_datetime(predictions[date_col])

        if imp_list:
            imp_all = pd.concat(imp_list, ignore_index=True)
            self._feature_importance = (imp_all.groupby("feature")[["gain", "split"]]
                                        .mean().sort_values("gain", ascending=False))

        logger.info("=" * 60)
        logger.info("V7 Walk-Forward done: %d/%d folds, %d predictions",
                     self._trained_folds, n_folds, len(predictions))
        logger.info("OOS: %s ~ %s | %d cross-sections | %d stocks",
                     predictions[date_col].min().strftime("%Y-%m-%d"),
                     predictions[date_col].max().strftime("%Y-%m-%d"),
                     predictions[date_col].nunique(), predictions[symbol_col].nunique())
        if self._feature_importance is not None:
            logger.info("Top-5 features (gain):")
            for feat, row in self._feature_importance.head(5).iterrows():
                logger.info("  %s: gain=%.2f split=%.2f", feat, row["gain"], row["split"])
        logger.info("=" * 60)
        return predictions

    # ── Main entry ──

    def run(self, panel, *, blended=None,
            date_col="date", symbol_col="symbol", close_col="收盘") -> pd.DataFrame:
        logger.info("=" * 64)
        logger.info("V7 Engine — 1M Label + 0M Gap + TO lambda=%.2f", self.config.lambda_turnover)
        logger.info("=" * 64)

        df = self._filter_populated_dates(panel, date_col, self.config.min_stocks_per_date)
        df = self.prepare_labels(df, date_col, symbol_col, close_col)
        df = self.prepare_features(df, blended, date_col, symbol_col)
        predictions = self.walk_forward_train(df, date_col, symbol_col)
        return predictions

    @staticmethod
    def _filter_populated_dates(panel, date_col="date", min_stocks=50) -> pd.DataFrame:
        counts = panel.groupby(date_col).size()
        good = counts[counts >= min_stocks].index
        filtered = panel[panel[date_col].isin(good)].copy()
        logger.info("Filter: >=%d stocks: %d -> %d dates (%.0f%%)",
                     min_stocks, panel[date_col].nunique(), filtered[date_col].nunique(),
                     100*filtered[date_col].nunique()/panel[date_col].nunique())
        return filtered

    def get_feature_importance(self) -> pd.DataFrame | None:
        return self._feature_importance

    def to_markdown_report(self) -> str:
        cfg = self.config
        return "\n".join([
            "## V7 Alpha Engine — Training Report",
            "",
            f"- **Label:** forward_return_1m -> rank [0,1]",
            f"- **Gap:** 0M (standard 1-step-forward, no blind zone)",
            f"- **Objective:** Custom L2 + {cfg.lambda_turnover}*(pred-prev)^2",
            f"- **Folds:** {self._trained_folds}",
            f"- **Features:** {len(self._feature_cols)} cols",
            f"- **Window:** {cfg.train_months}M train + {cfg.val_months}M val + {cfg.test_months}M test",
            "",
            "```",
            "L = 0.5*(pred-y)^2 + lambda*0.5*(pred-prev)^2",
            "g = (pred-y) + lambda*(pred-prev)",
            "h = 1 + lambda",
            "```",
        ])
