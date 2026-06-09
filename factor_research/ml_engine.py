"""
LightGBM Alpha 引擎 (Stage 3 — ML 选股模型).

废弃线性等权/IC加权方法, 引入 LightGBM 自动挖掘 16 因子与目标收益
之间的非线性关系及条件交互效应。

核心设计原则:
  1. 截面百分位排名 (Rank) — 特征和标签均转换为 [0,1] 排名, 免疫宏观周期漂移
  2. 严格 Walk-Forward — 36 月训练 + 6 月验证 + 1 月 OOS, 按月滑动
  3. 保守超参数 — 浅树深/低叶子数/高采样率, 对抗金融低信噪比
  4. 早停防过拟合 — 验证集 loss 不再下降时停止训练

与现有系统的集成:
  - 输入: output/preprocessed.parquet (含 16 个 _neutral_z 因子 + 收盘价)
  - 输出: ml_signal DataFrame (date, symbol, ml_signal), 替代 alpha_signal
  - 喂入 run_backtest_with_costs() 无需修改引擎代码

用法:
  from factor_research.ml_engine import LightGBMAlphaEngine
  engine = LightGBMAlphaEngine()
  ml_predictions = engine.run(panel)
  # 将 ml_predictions 合并到 blended panel 后传入 backtest
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_engine")
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

# 抑制 LightGBM 的冗余警告
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════


@dataclass
class MLConfig:
    """
    LightGBM Alpha 引擎配置。

    Walk-Forward 参数
    -----------------
    train_months : int = 36
        训练窗口 (月数)。36 个月 ≈ 3 年, 覆盖一个完整库存周期。
    val_months : int = 6
        验证窗口 (月数)。用于早停, 防止过拟合。
    test_months : int = 1
        测试/OOS 窗口 (月数)。每次预测未来 1 个月。

    标签参数
    --------
    label_method : str = "rank"
        "rank"       — 截面排名 (推荐, 免疫 Beta 波动)
        "neutralize" — 市值中性化残差
        "raw"        — 原始 forward_return_1m (不推荐)

    特征参数
    --------
    feature_method : str = "rank"
        "rank"   — 截面百分位排名 (推荐, 免疫量纲漂移)
        "zscore" — 保留原始 z-score

    LightGBM 超参数
    ---------------
    遵循"金融低信噪比 → 强正则化"原则:
      - max_depth ≤ 5 (防止过深的交互)
      - num_leaves ≤ 31 (每棵树不能太复杂)
      - subsample < 1.0 (行采样增加随机性)
      - colsample_bytree < 1.0 (列采样增加随机性)
      - min_child_samples ≥ 100 (叶节点必须有足够样本)
      - reg_alpha/reg_lambda > 0 (L1/L2 正则化)
    """

    # Walk-Forward
    train_months: int = 36
    val_months: int = 6
    test_months: int = 1

    # Label
    label_method: str = "rank"       # "rank" | "neutralize" | "raw"

    # Feature
    feature_method: str = "rank"     # "rank" | "zscore"

    # LightGBM
    objective: str = "regression"
    metric: str = "l2"
    boosting: str = "gbdt"
    num_leaves: int = 24
    max_depth: int = 4
    learning_rate: float = 0.02
    n_estimators: int = 2000
    subsample: float = 0.70
    colsample_bytree: float = 0.70
    subsample_freq: int = 1
    min_child_samples: int = 100
    reg_alpha: float = 0.10
    reg_lambda: float = 0.10
    early_stopping_rounds: int = 50
    verbose: int = -1
    random_state: int = 42
    n_jobs: int = -1

    # Ensemble & Smoothing
    seeds: list[int] = field(default_factory=lambda: [42])
    ema_alpha: float = 0.0           # 0=不启用; 0.4=40%当期+60%历史

    def to_lgb_params(self, seed: int | None = None) -> dict:
        """转为 LightGBM 参数字典。seed 覆盖 config.random_state。"""
        rs = seed if seed is not None else self.random_state
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
            "random_state": rs,
            "n_jobs": self.n_jobs,
        }


# ═══════════════════════════════════════════════════════════
# 主引擎
# ═══════════════════════════════════════════════════════════


class LightGBMAlphaEngine:
    """
    LightGBM Alpha 引擎 — Walk-Forward 训练 + OOS 预测。

    架构:
      ┌─────────────────────────────────────────────────┐
      │  Panel (date × symbol, 16 因子 + 收盘价)         │
      │  ↓                                              │
      │  [特征工程] 截面 Rank 变换                        │
      │  ↓                                              │
      │  [标签工程] forward_return_1m → 截面 Rank         │
      │  ↓                                              │
      │  [Walk-Forward Loop]                            │
      │    Fold 0: Train[0:36] Val[36:42] Test[42]      │
      │    Fold 1: Train[1:37] Val[37:43] Test[43]      │
      │    ...                                          │
      │    Fold N: Train[N:N+36] Val[N+36:N+42] ...     │
      │  ↓                                              │
      │  [拼接 OOS 预测] → ml_signal DataFrame            │
      └─────────────────────────────────────────────────┘

    Parameters
    ----------
    config : MLConfig
        引擎配置 (含所有超参数)。
    factor_cols : list[str] | None
        因子列名列表。None 时自动发现所有 _neutral_z 列。
    """

    def __init__(
        self,
        config: Optional[MLConfig] = None,
        factor_cols: Optional[list[str]] = None,
    ):
        self.config = config or MLConfig()
        self.factor_cols = factor_cols  # None → 自动发现
        self._feature_cols: list[str] = []     # 实际使用的特征列
        self._trained_folds: int = 0            # 成功训练的 fold 数
        self._feature_importance: pd.DataFrame | None = None  # 累积特征重要性

    # ═══════════════════════════════════════════════════
    # 数据准备
    # ═══════════════════════════════════════════════════

    def prepare_features(
        self,
        panel: pd.DataFrame,
        factor_cols: Optional[list[str]] = None,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        特征工程: 将原始因子转换为月度截面百分位排名 (Rank)。

        为什么用 Rank 而非 Z-Score:
          - Rank ∈ [0, 1] 有界, 不受极端值影响
          - Rank 在牛熊市中分布稳定, Z-Score 在极端行情中会整体漂移
          - 树模型天然适合处理百分位特征

        Parameters
        ----------
        panel : pd.DataFrame
            原始因子面板。
        factor_cols : list[str] | None
            要转换的因子列。None 时使用 self.factor_cols。
        date_col : str
            日期列名。

        Returns
        -------
        pd.DataFrame: 原 panel + 新增 _rank 列。
        """
        df = panel.copy()
        if factor_cols is None:
            factor_cols = self.factor_cols
        if factor_cols is None:
            factor_cols = [c for c in df.columns if c.endswith("_neutral_z")]
            self.factor_cols = factor_cols

        if not factor_cols:
            raise ValueError("未找到因子列。请传入 factor_cols 或确保面板含 _neutral_z 列。")

        method = self.config.feature_method

        if method == "rank":
            # 截面百分位排名: 每个月内, 每个因子独立排名
            for col in factor_cols:
                if col not in df.columns:
                    continue
                rank_col = f"{col}_rank"
                # 按日期分组 → 排名 → 归一化到 [0, 1]
                df[rank_col] = (
                    df.groupby(date_col)[col]
                    .rank(pct=True, na_option="bottom")
                    .fillna(0.5)  # NaN → 中位数
                )
            self._feature_cols = [f"{c}_rank" for c in factor_cols if c in df.columns]
            logger.info(
                "特征工程: %d 因子 → 截面 Rank (百分位), 有效列=%d",
                len(factor_cols), len(self._feature_cols),
            )
        else:
            # z-score: 保留原始, 仅 fillna
            for col in factor_cols:
                if col not in df.columns:
                    continue
                df[col] = df[col].fillna(0.0)
            self._feature_cols = [c for c in factor_cols if c in df.columns]
            logger.info("特征工程: %d 因子 → 保留 Z-Score", len(self._feature_cols))

        return df

    def prepare_labels(
        self,
        panel: pd.DataFrame,
        return_col: str = "forward_return_1m",
        date_col: str = "date",
        symbol_col: str = "symbol",
        close_col: str = "收盘",
    ) -> pd.DataFrame:
        """
        标签工程: 计算并处理 forward_return_1m。

        方法:
          "rank"       — 月度截面排名 [0, 1], 最大化免疫 Beta 波动
          "neutralize" — 市值中性化残差 (需要 mcap 列)
          "raw"        — 原始收益率 (不推荐, Beta 噪音过大)

        Parameters
        ----------
        panel : pd.DataFrame
            面板数据。
        return_col : str
            收益列名。
        date_col : str
            日期列名。
        symbol_col : str
            股票代码列名。
        close_col : str
            收盘价列名 (用于计算 forward_return_1m, 如不存在)。

        Returns
        -------
        pd.DataFrame: panel + label 列。
        """
        df = panel.copy()

        # 确保 forward_return_1m 存在
        if return_col not in df.columns:
            logger.info("计算 forward_return_1m ...")
            if close_col not in df.columns:
                for c in df.columns:
                    if "收" in str(c) or "close" in str(c).lower():
                        close_col = c
                        break
                else:
                    raise KeyError("未找到收盘价列, 无法计算 forward_return_1m")
            df = df.sort_values([symbol_col, date_col])
            df[return_col] = (
                df.groupby(symbol_col)[close_col]
                .transform(lambda x: x.shift(-1) / x - 1.0)
            )

        label_col = "label"

        method = self.config.label_method

        if method == "rank":
            # 截面排名 → [0, 1]
            df[label_col] = (
                df.groupby(date_col)[return_col]
                .rank(pct=True, na_option="bottom")
                .fillna(0.5)
            )
            logger.info("标签工程: forward_return_1m → 截面 Rank [0,1]")

        elif method == "neutralize":
            # 市值中性化残差
            mcap_col = None
            for c in ["mcap_est", "mcap", "市值", "总市值"]:
                if c in df.columns:
                    mcap_col = c
                    break
            if mcap_col is None:
                logger.warning("未找到市值列, 退回到 raw 标签")
                df[label_col] = df[return_col].fillna(0.0)
            else:
                # 每月: label = return - market_cap_weighted_mean
                # 按市值 log 分组做 Loess 太慢, 用简单残差:
                # residual = return - median(return) within mcap tertile
                df["_mcap_tertile"] = df.groupby(date_col)[mcap_col].transform(
                    lambda x: pd.qcut(x, 3, labels=[0, 1, 2], duplicates="drop")
                ).fillna(1)
                df[label_col] = (
                    df[return_col] -
                    df.groupby([date_col, "_mcap_tertile"])[return_col].transform("median")
                )
                df.drop(columns=["_mcap_tertile"], inplace=True)
                logger.info("标签工程: forward_return_1m → 市值中性化残差")

        else:  # "raw"
            df[label_col] = df[return_col].fillna(0.0)
            logger.info("标签工程: forward_return_1m → 原始值 (不推荐)")

        return df

    # ═══════════════════════════════════════════════════
    # Walk-Forward 训练
    # ═══════════════════════════════════════════════════

    def walk_forward_train(
        self,
        panel: pd.DataFrame,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        Walk-Forward 滚动训练 + OOS 预测。

        时间线:
          |←── Train (36M) ──→|← Val (6M) →| Test (1M) |
          |←── Train (36M) ──→|← Val (6M) →| Test (1M) |  ← +1 月滑动
          ...

        严格防范未来函数 (Data Leakage):
          - 每个 fold 的 Train/Val/Test 按时间严格切分
          - 特征 rank 在 fold 内部重新计算 (防止全局信息泄漏)
          - 验证集仅用于早停, 不参与训练

        Parameters
        ----------
        panel : pd.DataFrame
            含特征列 (_rank or _neutral_z), label 列, 和 date 列的面板。
        date_col : str
            日期列名。

        Returns
        -------
        pd.DataFrame: 所有 OOS 预测的拼接 (date, symbol, ml_signal)。
        """
        import lightgbm as lgb  # noqa: F811

        df = panel.copy()
        dates = sorted(df[date_col].unique())
        n_dates = len(dates)

        cfg = self.config
        fold_size = cfg.train_months + cfg.val_months + cfg.test_months
        n_folds = n_dates - fold_size + 1

        if n_folds <= 0:
            raise ValueError(
                f"数据不足: {n_dates} 个截面 < 所需 {fold_size} 个"
                f" (train={cfg.train_months}+val={cfg.val_months}+test={cfg.test_months})"
            )

        # ── 确定 seeds ──
        seeds = cfg.seeds if cfg.seeds else [cfg.random_state]
        n_seeds = len(seeds)

        logger.info("=" * 56)
        logger.info("Walk-Forward 训练: %d folds × %d seeds", n_folds, n_seeds)
        logger.info(
            "  窗口: %dM train + %dM val + %dM test | 总计 %d 截面",
            cfg.train_months, cfg.val_months, cfg.test_months, n_dates,
        )
        logger.info(
            "  特征: %d 列 | 标签: %s | 模型: LightGBM (depth=%d, leaves=%d)",
            len(self._feature_cols), cfg.label_method,
            cfg.max_depth, cfg.num_leaves,
        )
        if n_seeds > 1:
            logger.info("  Ensemble: %d seeds → 等权平均", n_seeds)
        logger.info("=" * 56)

        all_predictions: list[pd.DataFrame] = []
        importance_list: list[pd.DataFrame] = []
        self._trained_folds = 0

        for fold_idx in range(n_folds):
            # ── 时间索引切分 ──
            train_start = fold_idx
            train_end = fold_idx + cfg.train_months
            val_end = train_end + cfg.val_months
            test_idx = val_end  # 1 month OOS

            train_dates = set(dates[train_start:train_end])
            val_dates = set(dates[train_end:val_end])
            test_date = dates[test_idx]

            # 在 fold 内重新 rank (防止信息泄漏: 每个 fold 独立 ranking)
            fold_df = df[df[date_col].isin(train_dates | val_dates | {test_date})].copy()

            # 确保特征列是 rank 格式 (在 fold 内部重新 rank)
            if cfg.feature_method == "rank":
                # 只对训练+验证+测试范围内的数据做 rank
                # 这会稍微降低 ranking 的稳健性, 但防止了全局信息泄漏
                for col in self.factor_cols or []:
                    if col not in fold_df.columns:
                        continue
                    rank_col = f"{col}_rank"
                    fold_df[rank_col] = (
                        fold_df.groupby(date_col)[col]
                        .rank(pct=True, na_option="bottom")
                        .fillna(0.5)
                    )

            # 构造训练/验证/测试集
            train_mask = fold_df[date_col].isin(train_dates)
            val_mask = fold_df[date_col].isin(val_dates)
            test_mask = fold_df[date_col] == test_date

            # 必须的特征列
            feature_cols = self._feature_cols
            if not all(c in fold_df.columns for c in feature_cols):
                missing = [c for c in feature_cols if c not in fold_df.columns]
                logger.warning("Fold %d: 缺失特征列 %s, 跳过", fold_idx, missing)
                continue

            X_train = fold_df.loc[train_mask, feature_cols].astype(float)
            y_train = fold_df.loc[train_mask, "label"].astype(float)
            X_val = fold_df.loc[val_mask, feature_cols].astype(float)
            y_val = fold_df.loc[val_mask, "label"].astype(float)
            X_test = fold_df.loc[test_mask, feature_cols].astype(float)

            # 检查: 必须有足够样本
            if len(X_train) < 500 or len(X_val) < 50:
                logger.warning(
                    "Fold %d: 样本不足 (train=%d, val=%d), 跳过",
                    fold_idx, len(X_train), len(X_val),
                )
                continue

            # ── 多 Seed 集成训练 ──
            seed_predictions: list[np.ndarray] = []
            best_iters: list[int] = []

            for seed_i, seed in enumerate(seeds):
                params = cfg.to_lgb_params(seed=seed)
                params["early_stopping_rounds"] = cfg.early_stopping_rounds

                try:
                    model = lgb.LGBMRegressor(
                        n_estimators=cfg.n_estimators,
                        **{k: v for k, v in params.items()
                           if k not in ("early_stopping_rounds", "verbose")},
                        verbose=-1,
                    )
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric="l2",
                        callbacks=[
                            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                            lgb.log_evaluation(period=0),
                        ],
                    )

                    y_pred = model.predict(X_test)
                    seed_predictions.append(y_pred)
                    best_iters.append(
                        model.best_iteration_ if model.best_iteration_ else 0
                    )

                    # 仅第一个 seed 记录特征重要性 (节省内存)
                    if seed_i == 0:
                        imp = pd.DataFrame({
                            "feature": feature_cols,
                            "gain": model.booster_.feature_importance(importance_type="gain"),
                            "split": model.booster_.feature_importance(importance_type="split"),
                        })
                        imp["fold"] = fold_idx
                        importance_list.append(imp)

                except Exception as e:
                    logger.warning(
                        "Fold %d seed=%d 训练失败: %s", fold_idx, seed, e
                    )
                    continue

            if not seed_predictions:
                logger.warning("Fold %d: 所有 seed 均失败, 跳过", fold_idx)
                continue

            # ── 等权平均所有 seed 的预测 ──
            y_pred_ensemble = np.mean(seed_predictions, axis=0)
            avg_best_iter = int(np.mean(best_iters))

            # ── 保存预测 ──
            pred_df = fold_df.loc[test_mask, ["date", "symbol"]].copy()
            pred_df["ml_signal"] = y_pred_ensemble
            all_predictions.append(pred_df)

            self._trained_folds += 1

            if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
                logger.info(
                    "  Fold %3d/%d | test=%s | n_train=%d | n_val=%d | "
                    "n_test=%d | seeds=%d/%d | avg_iter=%d",
                    fold_idx + 1, n_folds, str(test_date)[:10],
                    len(X_train), len(X_val), len(X_test),
                    len(seed_predictions), n_seeds, avg_best_iter,
                )

        if not all_predictions:
            raise RuntimeError("所有 Fold 均训练失败。请检查数据和参数。")

        # ── 拼接所有 OOS 预测 ──
        predictions = pd.concat(all_predictions, ignore_index=True)
        predictions["date"] = pd.to_datetime(predictions["date"])

        # ── 累积特征重要性 ──
        if importance_list:
            imp_all = pd.concat(importance_list, ignore_index=True)
            self._feature_importance = (
                imp_all.groupby("feature")[["gain", "split"]]
                .mean()
                .sort_values("gain", ascending=False)
            )

        logger.info("=" * 56)
        logger.info(
            "Walk-Forward 完成: %d/%d folds 成功, OOS 预测 %d 条",
            self._trained_folds, n_folds, len(predictions),
        )
        logger.info("OOS 日期范围: %s ~ %s",
                     predictions["date"].min().strftime("%Y-%m-%d"),
                     predictions["date"].max().strftime("%Y-%m-%d"))
        logger.info("OOS 覆盖 %d 个截面, %d 只股票",
                     predictions["date"].nunique(),
                     predictions["symbol"].nunique())
        if self._feature_importance is not None:
            logger.info("Top-5 特征 (按 gain):")
            for feat, row in self._feature_importance.head(5).iterrows():
                logger.info("  %s: gain=%.2f split=%.2f", feat, row["gain"], row["split"])
        logger.info("=" * 56)

        return predictions

    # ═══════════════════════════════════════════════════
    # 预测后处理
    # ═══════════════════════════════════════════════════

    @staticmethod
    def apply_ema_smoothing(
        predictions: pd.DataFrame,
        alpha: float = 0.4,
        date_col: str = "date",
        symbol_col: str = "symbol",
        signal_col: str = "ml_signal",
    ) -> pd.DataFrame:
        """
        对每只股票的预测值做指数移动平均 (EMA), 降低时间序列上的
        预测方差, 从而显著压降换手率。

        公式:
          Final_{i,t} = α × raw_{i,t} + (1−α) × Final_{i,t−1}

        对于首次出现的股票 (无历史), Final_{i,0} = raw_{i,0}。

        Parameters
        ----------
        predictions : pd.DataFrame
            含 date, symbol, ml_signal 的预测表。
        alpha : float
            EMA 衰减因子。α=0.4 表示当期占 40%, 历史占 60%。
        date_col, symbol_col, signal_col : str
            列名。

        Returns
        -------
        pd.DataFrame: 含 date, symbol, {signal_col} (平滑后)。
        """
        if alpha <= 0 or alpha > 1:
            raise ValueError(f"alpha 必须在 (0, 1] 范围内, 实际: {alpha}")

        df = predictions[[date_col, symbol_col, signal_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)

        def _ema_series(series: pd.Series) -> pd.Series:
            """对单个股票的时序预测做 EMA。"""
            result = series.copy().astype(float)
            ema = result.iloc[0]  # 首期 = 原始值
            for i in range(1, len(result)):
                ema = alpha * result.iloc[i] + (1.0 - alpha) * ema
                result.iloc[i] = ema
            return result

        df[signal_col] = (
            df.groupby(symbol_col, group_keys=False)[signal_col]
            .transform(_ema_series)
        )

        logger.info(
            "EMA 平滑: α=%.2f | %d 只股票 | 输出列=%s",
            alpha, df[symbol_col].nunique(), signal_col,
        )
        return df[[date_col, symbol_col, signal_col]]

    # ═══════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════

    def run(
        self,
        panel: pd.DataFrame,
        *,
        date_col: str = "date",
        symbol_col: str = "symbol",
        return_col: str = "forward_return_1m",
        close_col: str = "收盘",
    ) -> pd.DataFrame:
        """
        一键运行完整 ML Pipeline。

        Pipeline:
          1. prepare_labels()  — 计算截面排名标签
          2. prepare_features() — 因子转截面排名
          3. walk_forward_train() — 滚动训练 + OOS 预测
          4. 返回 ml_signal DataFrame

        Parameters
        ----------
        panel : pd.DataFrame
            原始因子面板 (含 _neutral_z 列 + 收盘价)。
        date_col : str
            日期列名。
        symbol_col : str
            股票代码列名。
        return_col : str
            收益列名。
        close_col : str
            收盘价列名。

        Returns
        -------
        pd.DataFrame: date, symbol, ml_signal
        """
        logger.info("=" * 56)
        logger.info("LightGBM Alpha Engine — ML Pipeline 启动")
        logger.info("=" * 56)

        # Step 1: 标签
        df = self.prepare_labels(
            panel,
            return_col=return_col,
            date_col=date_col,
            symbol_col=symbol_col,
            close_col=close_col,
        )

        # Step 2: 特征
        df = self.prepare_features(df, date_col=date_col)

        # Step 3: Walk-Forward (含多 Seed 集成)
        predictions = self.walk_forward_train(df, date_col=date_col)

        # Step 4: EMA 预测平滑 (可选, 由 config.ema_alpha 控制)
        if self.config.ema_alpha > 0:
            logger.info("Step 4: 应用 EMA 预测平滑 (α=%.2f)", self.config.ema_alpha)
            predictions = self.apply_ema_smoothing(
                predictions,
                alpha=self.config.ema_alpha,
                date_col=date_col,
                symbol_col=symbol_col,
                signal_col="ml_signal",
            )

        return predictions

    # ═══════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════

    def get_feature_importance(self) -> pd.DataFrame | None:
        """返回累积特征重要性。"""
        return self._feature_importance

    def to_markdown_report(self) -> str:
        """生成训练报告 (Markdown)。"""
        seeds = self.config.seeds if self.config.seeds else [self.config.random_state]
        lines = [
            "## LightGBM Alpha Engine — 训练报告",
            "",
            f"- **训练 Folds:** {self._trained_folds}",
            f"- **特征列数:** {len(self._feature_cols)}",
            f"- **标签方法:** {self.config.label_method}",
            f"- **特征方法:** {self.config.feature_method}",
            f"- **窗口:** {self.config.train_months}M train + "
            f"{self.config.val_months}M val + {self.config.test_months}M test",
            f"- **超参数:** max_depth={self.config.max_depth}, "
            f"num_leaves={self.config.num_leaves}, "
            f"lr={self.config.learning_rate}",
            f"- **集成:** {len(seeds)} seeds → 等权平均"
            + (f" | seeds={seeds}" if len(seeds) > 1 else ""),
            f"- **EMA 平滑:** α={self.config.ema_alpha}"
            + (" (启用)" if self.config.ema_alpha > 0 else " (关闭)"),
            "",
        ]
        if self._feature_importance is not None:
            lines.append("### 特征重要性 (平均 Gain)")
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
