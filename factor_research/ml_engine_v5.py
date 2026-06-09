# ═══════════════════════════════════════════════════════════════
# ARCHIVED: Turnover-Aware 3M gap 实验 (Stage 5)
# 已被 V7 (1M labels + 0M gap + Turnover-Aware) 取代
# V5 的 Turnover-Aware loss 设计被 V7 继承, 但 3M gap 是 MaxDD 根因
# 保留作为消融实验记录 — 不建议用于新工作
# ═══════════════════════════════════════════════════════════════
"""
LightGBM Turnover-Aware Alpha Engine V5 (Stage 5 — 换手惩罚范式).

核心创新: Custom Objective 内嵌时序换手惩罚
─────────────────────────────────────────
问题: L2/LambdaRank 均导致高频信号翻转 → 50%+ 月换手率
方案: 在损失函数中直接惩罚时序上的预测变化

  L = 1/2(ŷ − y)² + λ/2(ŷ − ŷ_{t−1})²

  Gradient:  g = (ŷ − y) + λ(ŷ − ŷ_{t−1})
  Hessian:   h = 1 + λ

锚点策略: 使用线性 alpha_signal_{t-1} 作为 prev_signal
  - 线性模型换手仅 23.7%, 天然时序稳定
  - ML 模型被正则化, 不远离线性基准, 仅在强证据时偏离
  - 闭包 (closure) 传递 prev_signal 到 LightGBM custom objective

与 V1/V2 的核心区别:
  ┌──────────────────┬────────────┬──────────────┬──────────────────┐
  │ 维度              │ V1 (L2)    │ V2 (Lambda)  │ V5 (T.O.-Aware)  │
  ├──────────────────┼────────────┼──────────────┼──────────────────┤
  │ Objective         │ L2         │ LambdaRank   │ L2 + λ·(ŷ−ŷ₋₁)² │
  │ Label             │ rank(1M)   │ int 0-4, 3M  │ rank(3M)         │
  │ Gap               │ 无          │ 3M           │ 3M               │
  │ Feature dim       │ 16 rank    │ 50+ feat     │ 16 rank          │
  │ Turnover control  │ 无          │ 间接(无效)    │ 直接梯度惩罚       │
  │ subsample         │ 0.70       │ 0.70         │ 1.0 (闭包对齐)    │
  └──────────────────┴────────────┴──────────────┴──────────────────┘

用法:
  from factor_research.ml_engine_v5 import LightGBMAlphaEngineV5, MLConfigV5
  engine = LightGBMAlphaEngineV5(config=MLConfigV5(lambda_turnover=0.5))
  predictions = engine.run(panel, blended=blended)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_engine_v5")
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
class MLConfigV5:
    """
    Turnover-Aware Alpha 引擎配置 V5。

    Walk-Forward + Gap 参数
    -----------------------
    train_months : int = 36
        训练窗口 (月数)。
    val_months : int = 6
        验证窗口 (月数)。
    test_months : int = 1
        OOS 预测窗口 (月数)。
    label_horizon : int = 3
        标签前瞻期 (月)。3 = forward_return_3m。
        同时用作 train↔val 之间的 Gap 长度 (防多期标签泄漏)。

    Turnover 惩罚参数
    ----------------
    lambda_turnover : float = 0.5
        换手惩罚系数 λ。经验范围 0.1 ~ 2.0。
          λ=0.0 → 回退到 V1 (纯 L2, 无惩罚)
          λ=0.5 → 推荐初始值 (温和惩罚)
          λ=1.0 → 强惩罚 (训练慢, 换手极低但可能牺牲 Alpha)

    LightGBM 超参数
    --------------
    注意: subsample 固定为 1.0, 因为 custom objective 的 prev_signal
    通过闭包传递, 必须保证数据顺序与 Dataset 内部一致。
    """

    # Walk-Forward + Gap
    train_months: int = 36
    val_months: int = 6
    test_months: int = 1
    label_horizon: int = 3
    min_stocks_per_date: int = 50   # 过滤稀疏截面 (仅保留月频有效截面)

    # Turnover penalty
    lambda_turnover: float = 0.5

    # Label
    label_method: str = "rank"       # "rank" | "raw"

    # Feature
    feature_method: str = "rank"     # "rank" | "zscore"

    # LightGBM
    objective: str = "regression"    # 回归任务 (非 LambdaRank)
    metric: str = "l2"
    boosting: str = "gbdt"
    num_leaves: int = 24
    max_depth: int = 4
    learning_rate: float = 0.02
    n_estimators: int = 2000
    subsample: float = 1.0           # ← 固定 1.0 (闭包对齐要求)
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
        """转为 LightGBM 参数字典 (回归任务)。"""
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
# Custom Objective: Turnover-Aware L2 Loss
# ═══════════════════════════════════════════════════════════


def make_turnover_objective(
    prev_signal: np.ndarray,
    lambda_penalty: float,
) -> callable:
    """
    工厂函数: 创建 LightGBM 可调用的换手感知自定义损失函数。

    数学公式
    --------
    L  = ½(ŷ − y)²  +  λ·½(ŷ − ŷ_{t−1})²
    g  = (ŷ − y)     +  λ·(ŷ − ŷ_{t−1})
    h  = 1           +  λ                        (恒定 Hession)

    工程技巧 (闭包传递 prev_signal)
    ------------------------------
    LightGBM 的 custom objective 仅接收 (preds, train_data)。
    train_data.get_label() 可获取 labels, 但 prev_signal 不是 label。

    解决方案: 用闭包 (closure) 将 prev_signal 数组"注入"目标函数。
    LightGBM 在每轮 boosting 调用 fobj(preds, train_data) 时,
    preds 的顺序与 Dataset 的样本顺序完全一致, 因此闭包中的
    prev_signal 也与 preds 对齐。

    前置条件:
      - subsample=1.0 (保证每轮用全部样本, 不破坏对齐)
      - prev_signal 已在调用前做好 fillna 处理

    Parameters
    ----------
    prev_signal : np.ndarray, shape (n_samples,)
        每只股票上期的预测信号 (alpha_signal_{t-1})。
        必须与训练数据的样本顺序完全一致。
    lambda_penalty : float
        换手惩罚系数 λ。λ=0 时退化为纯 L2。

    Returns
    -------
    callable
        签名: f(preds: np.ndarray, train_data: lgb.Dataset) -> (grad, hess)
    """
    _prev = np.asarray(prev_signal, dtype=np.float64)
    _lam = float(lambda_penalty)

    def _objective(preds: np.ndarray, train_data) -> tuple[np.ndarray, np.ndarray]:
        """
        Turnover-Aware 目标函数。

        Parameters
        ----------
        preds : np.ndarray
            当前轮的预测值 ŷ, 长度 = 训练样本数。
        train_data : lgb.Dataset
            训练数据集 (用于获取 labels)。

        Returns
        -------
        grad : np.ndarray
            一阶梯度 g = (ŷ − y) + λ(ŷ − prev)
        hess : np.ndarray
            二阶梯度 h = 1 + λ
        """
        labels = train_data.get_label()
        labels = labels.astype(np.float64)

        # 确保长度一致
        if len(preds) != len(_prev):
            # 不应发生: 如果 subsample=1.0, 长度始终一致
            # 万一不一致 (如 validation set 被错误传入), 安全回退到纯 L2
            grad = preds - labels
            hess = np.ones_like(preds)
            return grad, hess

        # ── 核心梯度公式 ──
        residual = preds - labels          # ∂(½(ŷ−y)²)/∂ŷ = ŷ−y
        turnover = preds - _prev           # ∂(½(ŷ−prev)²)/∂ŷ = ŷ−prev

        grad = residual + _lam * turnover  # g = (ŷ−y) + λ(ŷ−prev)
        hess = np.full_like(preds, 1.0 + _lam, dtype=np.float64)  # h = 1 + λ

        return grad, hess

    return _objective


def make_l2_eval_metric() -> callable:
    """
    创建标准 L2 评估指标 (用于验证集, 不含换手惩罚)。

    LightGBM 在验证集上使用 feval 而非 fobj,
    验证指标应反映纯粹的预测精度。

    Returns
    -------
    callable
        签名: f(preds, train_data) -> (name, value, is_higher_better)
    """
    def _l2_metric(preds: np.ndarray, train_data) -> tuple[str, float, bool]:
        labels = train_data.get_label()
        mse = np.mean((preds - labels) ** 2)
        return "l2", mse, False  # lower is better
    return _l2_metric


# ═══════════════════════════════════════════════════════════
# 主引擎 V5
# ═══════════════════════════════════════════════════════════


class LightGBMAlphaEngineV5:
    """
    Turnover-Aware LightGBM Alpha 引擎 V5。

    架构:
      ┌─────────────────────────────────────────────────────────┐
      │  Panel (date × symbol, 16 因子 + 收盘价)                  │
      │  + Blended (alpha_signal, universe, mcap_est)            │
      │  ↓                                                       │
      │  [标签工程] forward_return_3m → 截面 Rank [0,1]           │
      │  ↓                                                       │
      │  [特征工程] Rank + prev_signal(=alpha_signal_{t-1})       │
      │  ↓                                                       │
      │  [Walk-Forward + 3M Gap Loop]                            │
      │    Fold 0: Train[0:33] Gap[33:36] Val[36:39] Test[42]   │
      │    Fold 1: Train[1:34] Gap[34:37] Val[37:40] Test[43]   │
      │    ...                                                   │
      │  ↓                                                       │
      │  [Turnover-Aware Training]                               │
      │    - 闭包传递 prev_signal → custom objective              │
      │    - fobj = make_turnover_objective(prev_train, λ)       │
      │    - subsample=1.0 保证数据对齐                           │
      │  ↓                                                       │
      │  [拼接 OOS 预测] → inertia_ml_signal DataFrame            │
      └─────────────────────────────────────────────────────────┘

    Parameters
    ----------
    config : MLConfigV5
        引擎配置 (含 λ 换手惩罚系数)。
    """

    def __init__(self, config: Optional[MLConfigV5] = None):
        self.config = config or MLConfigV5()
        self._feature_cols: list[str] = []
        self._trained_folds: int = 0
        self._feature_importance: pd.DataFrame | None = None

    # ═══════════════════════════════════════════════════
    # 标签工程: 3M forward return → rank
    # ═══════════════════════════════════════════════════

    def prepare_labels(
        self,
        panel: pd.DataFrame,
        return_col: str = "forward_return_3m",
        date_col: str = "date",
        symbol_col: str = "symbol",
        close_col: str = "收盘",
    ) -> pd.DataFrame:
        """
        计算 3 个月前瞻收益率, 并转为截面排名 [0, 1]。

        forward_return_3m(t) = close_{t+3} / close_t − 1

        使用 groupby(symbol).shift(-3) 实现 (数据按月采样)。

        Parameters
        ----------
        panel : pd.DataFrame
            含收盘价的面板。
        return_col : str
            输出列名。
        date_col, symbol_col, close_col : str

        Returns
        -------
        pd.DataFrame: panel + "label" 列 (截面 rank ∈ [0,1])。
        """
        df = panel.copy()
        cfg = self.config

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)

        # 查找收盘价列
        if close_col not in df.columns:
            for c in df.columns:
                if "收" in str(c) or "close" in str(c).lower():
                    close_col = c
                    break
            else:
                raise KeyError("未找到收盘价列, 无法计算 forward_return_3m")

        # ── 计算 forward_return_3m ──
        if return_col not in df.columns:
            logger.info("计算 forward_return_3m (horizon=%dM) ...", cfg.label_horizon)
            df[return_col] = (
                df.groupby(symbol_col)[close_col]
                .transform(lambda x: x.shift(-cfg.label_horizon) / x - 1.0)
            )

        # ── 截面排名 → [0, 1] 连续值 (回归任务) ──
        label_col = "label"
        df[label_col] = (
            df.groupby(date_col)[return_col]
            .rank(pct=True, na_option="bottom")
            .fillna(0.5)
        )

        n_valid = df[label_col].notna().sum()
        logger.info(
            "标签工程: forward_return_%dM → 截面 Rank [0,1] (回归) | "
            "有效: %d 条 (%.1f%%)",
            cfg.label_horizon,
            n_valid,
            100 * n_valid / len(df) if len(df) > 0 else 0,
        )
        return df

    # ═══════════════════════════════════════════════════
    # 特征工程: Rank + prev_signal 锚点
    # ═══════════════════════════════════════════════════

    def prepare_features(
        self,
        panel: pd.DataFrame,
        blended: Optional[pd.DataFrame] = None,
        date_col: str = "date",
        symbol_col: str = "symbol",
    ) -> pd.DataFrame:
        """
        两步特征工程:
          1. 截面 Rank 特征 (16 因子 → [0,1])
          2. 锚点特征 prev_signal = alpha_signal_{t-1} (不作为建树特征)

        prev_signal 的作用:
          模型特征中不包含 prev_signal (不参与分裂),
          但在训练时通过闭包传递给 custom objective 计算换手惩罚项。

        Parameters
        ----------
        panel : pd.DataFrame
            含 _neutral_z 因子列的面板。
        blended : pd.DataFrame | None
            含 alpha_signal 列的面板。
        date_col, symbol_col : str

        Returns
        -------
        pd.DataFrame: 含 _rank 特征 + prev_signal + label 的面板。
        """
        df = panel.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        cfg = self.config

        # 发现原始因子列
        factor_cols = [c for c in df.columns if c.endswith("_neutral_z")]
        if not factor_cols:
            raise ValueError("未找到 _neutral_z 因子列。")

        logger.info("特征工程: %d 原始因子", len(factor_cols))

        # ── Step 1: 截面 Rank 特征 ──
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
            logger.info("  Step 1: %d 因子 → 截面 Rank", len(rank_cols))
        else:
            for col in factor_cols:
                df[col] = df[col].fillna(0.0)
            rank_cols = factor_cols
            logger.info("  Step 1: %d 因子 → 保留 Z-Score", len(rank_cols))

        self._feature_cols = list(rank_cols)

        # ── Step 2: 锚点特征 prev_signal = alpha_signal_{t-1} ──
        df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)

        if blended is not None and "alpha_signal" in blended.columns:
            # 合并 alpha_signal
            df = df.merge(
                blended[[date_col, symbol_col, "alpha_signal"]],
                on=[date_col, symbol_col], how="left",
            )
            # 滞后 1 期 → prev_signal
            df["prev_signal"] = (
                df.groupby(symbol_col)["alpha_signal"]
                .transform(lambda x: x.shift(1))
            )
        else:
            logger.warning(
                "未提供 blended 数据或缺少 alpha_signal 列, "
                "prev_signal 将用 0.5 填充 (惩罚退化为 L2)"
            )
            df["prev_signal"] = np.nan

        # 首次出现的股票无历史 → 填充为 label (使首期惩罚与 L2 等价)
        if "label" in df.columns:
            df["prev_signal"] = df["prev_signal"].fillna(df["label"])
        else:
            df["prev_signal"] = df["prev_signal"].fillna(0.5)

        n_valid = df["prev_signal"].notna().sum()
        logger.info(
            "  Step 2: prev_signal = alpha_signal_{t-1} | "
            "有效: %d/%d (%.1f%%)",
            n_valid, len(df),
            100 * n_valid / len(df) if len(df) > 0 else 0,
        )

        logger.info("特征工程完成: 共 %d 个特征列 + prev_signal (锚点)",
                     len(self._feature_cols))
        return df

    # ═══════════════════════════════════════════════════
    # Walk-Forward + Turnover-Aware 训练
    # ═══════════════════════════════════════════════════

    def walk_forward_train(
        self,
        panel: pd.DataFrame,
        date_col: str = "date",
        symbol_col: str = "symbol",
    ) -> pd.DataFrame:
        """
        Walk-Forward 滚动训练 + Turnover-Aware OOS 预测。

        核心机制
        ────────
        1. 3M Gap: 同 V2, 防止多期标签的未来数据穿越。
        2. Custom Objective: 闭包注入 prev_signal, 梯度中直接惩罚
           (ŷ − ŷ_{t-1})², 强迫模型在时序上产生一致性预测。
        3. subsample=1.0: 保证闭包数据与 Dataset 内部顺序对齐。

        Gap 设计 (H=3):
          |← Train labels (33M) →| Gap 3M |← Val labels (3M) →| Gap 3M | Test |

        Returns
        -------
        pd.DataFrame: date, symbol, inertia_ml_signal
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
                f"数据不足: {n_dates} 截面 < 所需 {fold_size + H} 个"
                f" (train={cfg.train_months}+val={cfg.val_months}"
                f"+test={cfg.test_months}+horizon={H})"
            )

        logger.info("=" * 56)
        logger.info(
            "Turnover-Aware Walk-Forward: %d folds | λ=%.2f | Gap=%dM",
            n_folds, cfg.lambda_turnover, H,
        )
        logger.info(
            "  窗口: %dM train + %dM val + %dM test | horizon=%dM",
            cfg.train_months, cfg.val_months, cfg.test_months, H,
        )
        logger.info(
            "  特征: %d 列 | objective=regression | "
            "loss=L2+%.2f·Turnover² | subsample=%.2f",
            len(self._feature_cols), cfg.lambda_turnover, cfg.subsample,
        )
        logger.info("=" * 56)

        all_predictions: list[pd.DataFrame] = []
        importance_list: list[pd.DataFrame] = []
        self._trained_folds = 0

        for fold_idx in range(n_folds):
            # ── 时间索引 (含 3M Gap) ──
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

            # fold 内所有数据 (含 Gap 期的特征)
            fold_data_dates = set(dates[train_start:test_idx + 1])
            fold_df = df[df[date_col].isin(fold_data_dates)].copy()

            # ── 切分 train/val/test ──
            train_mask = fold_df[date_col].isin(train_label_dates)
            val_mask = fold_df[date_col].isin(val_label_dates)
            test_mask = fold_df[date_col] == test_date

            # 验证特征列 (排除 prev_signal — 不作为建树特征)
            feature_cols = [c for c in self._feature_cols if c in fold_df.columns]

            # ── 提取 prev_signal 锚点 (在丢弃前取出!) ──
            if "prev_signal" in fold_df.columns:
                prev_train = fold_df.loc[train_mask, "prev_signal"].values.astype(np.float64)
                prev_val = fold_df.loc[val_mask, "prev_signal"].values.astype(np.float64)
            else:
                logger.warning("Fold %d: 缺少 prev_signal 列, 退化为纯 L2", fold_idx)
                prev_train = fold_df.loc[train_mask, "label"].values.astype(np.float64)
                prev_val = fold_df.loc[val_mask, "label"].values.astype(np.float64)

            # ── 提取数据 (不含 prev_signal) ──
            X_train = fold_df.loc[train_mask, feature_cols].astype(float)
            y_train = fold_df.loc[train_mask, "label"].astype(float)
            X_val = fold_df.loc[val_mask, feature_cols].astype(float)
            y_val = fold_df.loc[val_mask, "label"].astype(float)
            X_test = fold_df.loc[test_mask, feature_cols].astype(float)

            # ── 样本检查 ──
            if len(X_train) < 500 or len(X_val) < 30:
                logger.warning(
                    "Fold %d: 样本不足 (train=%d, val=%d), 跳过",
                    fold_idx, len(X_train), len(X_val),
                )
                continue

            # ── 构建 LightGBM Dataset ──
            train_ds = lgb.Dataset(X_train, label=y_train)
            val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

            # ── 构建 Custom Objective (闭包注入 prev_signal) ──
            # LightGBM 4.x: 自定义目标函数通过 params['objective'] 传递 (而非 fobj kwarg)
            fobj = make_turnover_objective(prev_train, cfg.lambda_turnover)
            feval = make_l2_eval_metric()

            # ── 训练 ──
            params = cfg.to_lgb_params()
            # 将自定义目标函数注入 params (LightGBM 4.x 标准做法)
            params["objective"] = fobj

            try:
                model = lgb.train(
                    params=params,
                    train_set=train_ds,
                    num_boost_round=cfg.n_estimators,
                    valid_sets=[train_ds, val_ds],
                    valid_names=["train", "val"],
                    feval=feval,        # ← 标准 L2 验证指标
                    callbacks=[
                        lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )

                # OOS 预测
                y_pred = model.predict(X_test)
                n_iter = model.best_iteration

                # ── 保存预测 ──
                pred_df = fold_df.loc[test_mask, [date_col, symbol_col]].copy()
                pred_df["inertia_ml_signal"] = y_pred
                all_predictions.append(pred_df)

                # ── 特征重要性 ──
                imp = pd.DataFrame({
                    "feature": feature_cols,
                    "gain": model.feature_importance(importance_type="gain"),
                    "split": model.feature_importance(importance_type="split"),
                })
                imp["fold"] = fold_idx
                importance_list.append(imp)

                self._trained_folds += 1

                if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
                    # 计算训练集上的平均 turnover penalty
                    train_preds = model.predict(X_train)
                    avg_to_penalty = np.mean((train_preds - prev_train) ** 2)
                    logger.info(
                        "  Fold %3d/%d | test=%s | train=%d val=%d test=%d | "
                        "iter=%d | avg_Δpred²=%.4f",
                        fold_idx + 1, n_folds, str(test_date)[:10],
                        len(X_train), len(X_val), len(X_test),
                        n_iter, avg_to_penalty,
                    )

            except Exception as e:
                logger.error("Fold %d 训练失败: %s", fold_idx, e)
                continue

        if not all_predictions:
            raise RuntimeError("所有 Fold 均训练失败。请检查数据和参数。")

        # ── 拼接 OOS 预测 ──
        predictions = pd.concat(all_predictions, ignore_index=True)
        predictions[date_col] = pd.to_datetime(predictions[date_col])

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
            "Turnover-Aware Walk-Forward 完成: %d/%d folds 成功, OOS %d 条",
            self._trained_folds, n_folds, len(predictions),
        )
        logger.info(
            "OOS 日期: %s ~ %s | %d 截面 | %d 只股票",
            predictions[date_col].min().strftime("%Y-%m-%d"),
            predictions[date_col].max().strftime("%Y-%m-%d"),
            predictions[date_col].nunique(),
            predictions[symbol_col].nunique(),
        )
        if self._feature_importance is not None:
            logger.info("Top-5 特征 (按 gain):")
            for feat, row in self._feature_importance.head(5).iterrows():
                logger.info("  %s: gain=%.2f split=%.2f", feat, row["gain"], row["split"])
        logger.info("=" * 56)

        return predictions

    # ═══════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════

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
        一键运行 Turnover-Aware ML Pipeline。

        Pipeline:
          1. prepare_labels()   — 3M forward return → 截面 Rank
          2. prepare_features()  — Rank + prev_signal 锚点
          3. walk_forward_train() — Gap WF + Custom Objective

        Parameters
        ----------
        panel : pd.DataFrame
            原始因子面板 (含 _neutral_z 列 + 收盘价)。
        blended : pd.DataFrame | None
            含 alpha_signal 的面板 (用于构造 prev_signal 锚点)。
        date_col, symbol_col, close_col : str

        Returns
        -------
        pd.DataFrame: date, symbol, inertia_ml_signal
        """
        logger.info("=" * 60)
        logger.info("LightGBM Turnover-Aware Engine V5 — Pipeline 启动")
        logger.info("  objective=regression | lambda=%.2f | horizon=%dM | gap=%dM",
                     self.config.lambda_turnover,
                     self.config.label_horizon,
                     self.config.label_horizon)
        logger.info("  loss = 0.5*(pred-y)^2 + %.2f*0.5*(pred-prev)^2",
                     self.config.lambda_turnover)
        logger.info("=" * 60)

        # Step 0: 过滤稀疏截面 (仅保留月频有效日期, 保证 shift(-3)=3个月)
        df = self._filter_populated_dates(
            panel, date_col=date_col,
            min_stocks=self.config.min_stocks_per_date)

        # Step 1: 3M 标签 (截面 Rank)
        df = self.prepare_labels(
            df,
            date_col=date_col,
            symbol_col=symbol_col,
            close_col=close_col,
        )

        # Step 2: Rank 特征 + prev_signal 锚点
        df = self.prepare_features(
            df,
            blended=blended,
            date_col=date_col,
            symbol_col=symbol_col,
        )

        # Step 3: Walk-Forward + Custom Objective
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
        过滤掉截面股票过少的日期 (如月中稀疏日)。

        原始数据混合了月频和半月频日期, 其中月末有 ~220 只股票,
        月中仅有 1-2 只。直接使用会导致 fold 切分落在稀疏日期上。

        过滤后, shift(-3) 恰好跨越 3 个月 (约 90 天)。
        """
        date_counts = panel.groupby(date_col).size()
        good_dates = date_counts[date_counts >= min_stocks].index
        filtered = panel[panel[date_col].isin(good_dates)].copy()
        n_before = panel[date_col].nunique()
        n_after = filtered[date_col].nunique()
        logger.info(
            "Step 0: 过滤截面 >= %d 只股票 | %d → %d 个日期 (%.0f%% 保留)",
            min_stocks, n_before, n_after,
            100 * n_after / n_before if n_before > 0 else 0,
        )
        return filtered

    # ═══════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════

    def get_feature_importance(self) -> pd.DataFrame | None:
        """返回累积特征重要性。"""
        return self._feature_importance

    def to_markdown_report(self) -> str:
        """生成训练报告 (Markdown)。"""
        cfg = self.config
        lines = [
            "## Turnover-Aware Alpha Engine V5 — 训练报告",
            "",
            f"- **目标函数:** Custom L2 + λ·(ŷ−ŷ₋₁)² | λ = {cfg.lambda_turnover}",
            f"- **Label:** forward_return_{cfg.label_horizon}M → 截面 Rank [0,1]",
            f"- **Gap:** {cfg.label_horizon}M (防多期标签泄漏)",
            f"- **训练 Folds:** {self._trained_folds}",
            f"- **特征列数:** {len(self._feature_cols)}",
            f"- **窗口:** {cfg.train_months}M train + "
            f"{cfg.val_months}M val + {cfg.test_months}M test",
            f"- **超参数:** max_depth={cfg.max_depth}, "
            f"num_leaves={cfg.num_leaves}, lr={cfg.learning_rate}",
            f"- **subsample:** {cfg.subsample} (闭包对齐)",
            "",
            "### 损失函数",
            "",
            "```",
            "L = ½(ŷ − y)²  +  λ·½(ŷ − ŷ_{t−1})²",
            "g = (ŷ − y)     +  λ·(ŷ − ŷ_{t−1})",
            "h = 1 + λ",
            "```",
            "",
            f"其中 ŷ_{{t-1}} = alpha_signal lagged by 1 month (线性基准锚点).",
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
