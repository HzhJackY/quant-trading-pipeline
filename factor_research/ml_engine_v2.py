# ═══════════════════════════════════════════════════════════════
# ARCHIVED: LambdaRank 实验 (Stage 4)
# 已被 V7 (1M labels + 0M gap + Turnover-Aware) 取代
# 保留作为消融实验记录 — 不建议用于新工作
# ═══════════════════════════════════════════════════════════════
"""
LightGBM LambdaRank Alpha 引擎 V2 (Stage 4 — 排序学习范式).

范式转移: Regression → Learning-to-Rank

核心改进:
  1. 多期标签 (3M forward return) — 强迫模型学习持续性 Alpha, 内生降换手
  2. 时序差分特征 (Δ1M, Δ3M) — 让树模型感知因子变化方向
  3. 离散类别特征 (Sector, Mcap_Bin) — LightGBM 原生 categorical 处理
  4. LambdaRank 目标函数 — 直接优化截面排序质量, 解决 L2"均值塌陷"
  5. 严格 3 个月 Gap — 杜绝多期标签的未来数据穿越

与 V1 (ml_engine.py) 的关键区别:
  ┌──────────────────────┬─────────────────┬──────────────────────┐
  │ 维度                  │ V1 (L2)         │ V2 (LambdaRank)      │
  ├──────────────────────┼─────────────────┼──────────────────────┤
  │ Label                 │ rank(1M fwd)    │ rank(3M fwd cum)     │
  │ Feature dim           │ 16              │ ~50 (16+32Δ+2 cat)   │
  │ Objective             │ regression (L2) │ lambdarank           │
  │ Group (per date)      │ 无               │ group=[n_stocks/dt]  │
  │ 防泄漏                 │ 时序切分         │ 时序切分 + 3M Gap     │
  │ 特征                  │ rank(%)          │ rank(%) + Δ1M/Δ3M    │
  └──────────────────────┴─────────────────┴──────────────────────┘

用法:
  from factor_research.ml_engine_v2 import LightGBMAlphaEngineV2, MLConfigV2
  engine = LightGBMAlphaEngineV2(config=MLConfigV2())
  predictions = engine.run(panel, blended_info=blended)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ml_engine_v2")
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
class MLConfigV2:
    """
    LambdaRank Alpha 引擎配置 V2。

    Walk-Forward + Gap 参数
    -----------------------
    train_months : int = 36
        训练窗口 (月数)。
    val_months : int = 6
        验证窗口 (月数)。用于早停。
    test_months : int = 1
        OOS 预测窗口 (月数)。
    label_horizon : int = 3
        标签前瞻期 (月)。3 = forward_return_3m。
        同时用作 train↔val 之间的 Gap 长度。

    标签参数
    --------
    label_method : str = "rank"
        "rank" — 截面排名 [0,1] (LambdaRank 天然适配)

    特征参数
    --------
    feature_method : str = "rank"
        "rank" — 截面百分位排名
    use_delta_features : bool = True
        是否生成时序差分特征 (Δ1M, Δ3M)。
    use_categorical_features : bool = True
        是否引入 board (板块) 和 mcap_bin (市值分位) 类别特征。

    LightGBM LambdaRank 超参数
    --------------------------
    保守参数延续 V1 风格, 适配 LambdaRank:
      - objective="lambdarank" (核心变更)
      - max_depth ≤ 5
      - num_leaves ≤ 31
      - 强 L1/L2 正则化
      - eval_at=[10, 30] — 关注 Top-30 的排序质量
    """

    # Walk-Forward + Gap
    train_months: int = 36
    val_months: int = 6
    test_months: int = 1
    label_horizon: int = 3      # 3M forward return + 3M gap

    # Label
    label_method: str = "rank"

    # Feature
    feature_method: str = "rank"
    use_delta_features: bool = True
    use_categorical_features: bool = True

    # LightGBM — LambdaRank
    objective: str = "lambdarank"
    metric: str = "ndcg"
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
    eval_at: tuple = (10, 30)         # NDCG@10, NDCG@30
    verbose: int = -1
    random_state: int = 42
    n_jobs: int = -1

    def to_lgb_params(self, seed: int | None = None) -> dict:
        """转为 LightGBm 参数字典 (LambdaRank)。"""
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
            "eval_at": list(self.eval_at),
            "verbose": self.verbose,
            "random_state": rs,
            "n_jobs": self.n_jobs,
        }


# ═══════════════════════════════════════════════════════════
# 主引擎 V2
# ═══════════════════════════════════════════════════════════


class LightGBMAlphaEngineV2:
    """
    LambdaRank Alpha 引擎 V2 — 排序学习范式。

    架构:
      ┌─────────────────────────────────────────────────────┐
      │  Panel (date × symbol, 16 因子 + 收盘价 + board)     │
      │  ↓                                                  │
      │  [标签工程] forward_return_3m → 截面 Rank             │
      │  ↓                                                  │
      │  [特征工程] Rank + Δ1M/Δ3M + board(cat) + mcap_bin   │
      │  ↓                                                  │
      │  [Walk-Forward + Gap Loop]                          │
      │    Fold 0: Train[0:33] Gap[33:36] Val[36:39] Test[42]│
      │    Fold 1: Train[1:34] Gap[34:37] Val[37:40] Test[43]│
      │    ...                                              │
      │  ↓                                                  │
      │  [LambdaRank 训练] group=每期股票数, objective=lambdarank│
      │  ↓                                                  │
      │  [拼接 OOS 预测] → ml_rank_signal DataFrame           │
      └─────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        config: Optional[MLConfigV2] = None,
    ):
        self.config = config or MLConfigV2()
        self._feature_cols: list[str] = []
        self._trained_folds: int = 0
        self._feature_importance: pd.DataFrame | None = None

    # ═══════════════════════════════════════════════════
    # 标签工程: 3M forward return
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
        计算 3 个月前瞻收益率, 并转为截面排名。

        forward_return_3m(t) = close[t+3M] / close[t] - 1

        使用 groupby(symbol).shift(-3) 实现 (假设数据按月采样)。
        由于不同股票可能在不同月份休市, 实际使用周期性偏移。

        Parameters
        ----------
        panel : pd.DataFrame
        return_col : str
            输出列名。
        date_col, symbol_col, close_col : str

        Returns
        -------
        pd.DataFrame: panel + label 列 (截面 rank)。
        """
        df = panel.copy()
        cfg = self.config

        # 确保有排序
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)

        # ── 计算 forward_return_3m ──
        if return_col not in df.columns:
            logger.info("计算 forward_return_3m (horizon=%dM) ...", cfg.label_horizon)
            # 每月一篇 → shift(-3) 即 3 个月后
            df[return_col] = (
                df.groupby(symbol_col)[close_col]
                .transform(lambda x: x.shift(-cfg.label_horizon) / x - 1.0)
            )

        # ── 截面排名 → 离散整数标签 (LambdaRank 要求 int label) ──
        # 分 5 档: 0=最差, 4=最优
        label_col = "label"
        df["_rank_pct"] = (
            df.groupby(date_col)[return_col]
            .rank(pct=True, na_option="bottom")
            .fillna(0.5)
        )
        df[label_col] = (df["_rank_pct"] * 5).astype(int).clip(0, 4)
        df.drop(columns=["_rank_pct"], inplace=True)

        n_valid = df[label_col].notna().sum()
        logger.info(
            "标签工程: forward_return_%dM → int bins 0-4 (LambdaRank) | "
            "有效标签: %d 条 (%.1f%%)",
            cfg.label_horizon,
            n_valid,
            100 * n_valid / len(df) if len(df) > 0 else 0,
        )
        return df

    # ═══════════════════════════════════════════════════
    # 特征工程: Rank + Delta + Categorical
    # ═══════════════════════════════════════════════════

    def prepare_features(
        self,
        panel: pd.DataFrame,
        blended: Optional[pd.DataFrame] = None,
        date_col: str = "date",
        symbol_col: str = "symbol",
    ) -> pd.DataFrame:
        """
        三步特征工程:
          1. 截面 Rank 特征 (16 因子 → [0,1])
          2. 时序差分特征 (Δ1M, Δ3M) — 捕获变化方向
          3. 离散类别特征 (board, mcap_bin) — LightGBM categorical

        Parameters
        ----------
        panel : pd.DataFrame
            含 _neutral_z 因子列 + board 列。
        blended : pd.DataFrame | None
            含 mcap_est 列的面板 (用于构造市值分位)。
        date_col, symbol_col : str

        Returns
        -------
        pd.DataFrame: 含全部工程特征的面板。
        """
        df = panel.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        cfg = self.config

        # 发现原始因子列
        factor_cols = [c for c in df.columns if c.endswith("_neutral_z")]
        if not factor_cols:
            raise ValueError("未找到 _neutral_z 因子列。")

        logger.info("特征工程: %d 原始因子", len(factor_cols))

        # ── Step 1: 截面 Rank ──
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
            # z-score: 直接使用
            for col in factor_cols:
                df[col] = df[col].fillna(0.0)
            rank_cols = factor_cols
            logger.info("  Step 1: %d 因子 → 保留 Z-Score", len(rank_cols))

        self._feature_cols = list(rank_cols)

        # ── Step 2: 时序差分特征 ──
        if cfg.use_delta_features:
            df = df.sort_values([symbol_col, date_col])
            delta_cols = []
            for col in factor_cols:
                base_col = f"{col}_rank" if cfg.feature_method == "rank" else col
                if base_col not in df.columns:
                    continue
                # Δ1M: t − (t−1)
                d1m_col = f"{col}_d1m"
                df[d1m_col] = (
                    df.groupby(symbol_col)[base_col]
                    .transform(lambda x: x.diff(1))
                    .fillna(0.0)
                )
                delta_cols.append(d1m_col)

                # Δ3M: t − (t−3)
                d3m_col = f"{col}_d3m"
                df[d3m_col] = (
                    df.groupby(symbol_col)[base_col]
                    .transform(lambda x: x.diff(3))
                    .fillna(0.0)
                )
                delta_cols.append(d3m_col)

            self._feature_cols.extend(delta_cols)
            logger.info("  Step 2: +%d 差分特征 (Δ1M, Δ3M)", len(delta_cols))

        # ── Step 3: 离散类别特征 ──
        if cfg.use_categorical_features:
            cat_cols = []

            # 3a. Board (板块) → category code
            if "board" in df.columns:
                df["board_cat"] = df["board"].astype("category").cat.codes
                # -1 表示 NaN, 填为 0
                df["board_cat"] = df["board_cat"].replace(-1, 0).astype(int)
                cat_cols.append("board_cat")
                logger.info("  Step 3a: board → %d 类",
                            df["board"].nunique())

            # 3b. MarketCap_Bin (市值 5 等分位) → category
            mcap_col = None
            if blended is not None and "mcap_est" in blended.columns:
                df = df.merge(
                    blended[[date_col, symbol_col, "mcap_est"]],
                    on=[date_col, symbol_col], how="left",
                )
                mcap_col = "mcap_est"
            else:
                # 尝试在 panel 中查找市值列
                for c in ["mcap_est", "mcap", "总市值"]:
                    if c in df.columns:
                        mcap_col = c
                        break

            if mcap_col:
                df["mcap_bin"] = (
                    df.groupby(date_col)[mcap_col]
                    .transform(
                        lambda x: pd.qcut(
                            x, 5, labels=False, duplicates="drop"
                        )
                    )
                    .fillna(2)  # NaN → 中位
                    .astype(int)
                )
                cat_cols.append("mcap_bin")
                logger.info("  Step 3b: %s → 5 等分位 Mcap_Bin", mcap_col)

            self._feature_cols.extend(cat_cols)

        logger.info("特征工程完成: 共 %d 个特征列", len(self._feature_cols))
        return df

    # ═══════════════════════════════════════════════════
    # Walk-Forward + LambdaRank 训练
    # ═══════════════════════════════════════════════════

    def walk_forward_train(
        self,
        panel: pd.DataFrame,
        date_col: str = "date",
        symbol_col: str = "symbol",
    ) -> pd.DataFrame:
        """
        Walk-Forward 滚动训练 + LambdaRank OOS 预测。

        核心: 3 个月 Gap 防泄漏
        ──────────────────────
        标签 forward_return_3m(t) 需要 t+3 的收盘价。
        如果训练集最后一天是 T_train, 则标签只能算到 T_train − 3,
        否则标签会"偷看"验证集/测试集的未来价格。

        因此每 fold:
          Train labels:  dates[start : start+train_months−horizon]
          Val labels:    dates[start+train_months : start+train_months+val_months−horizon]
          Test (predict): dates[start+train_months+val_months]  (仅特征, 无标签需)

        LambdaRank Group 构建
        ────────────────────
        每个截面日期 = 一个 query group。按日期排序后统计每期股票数,
        传入 lgb.Dataset(group=[n1, n2, ...])。

        Returns
        -------
        pd.DataFrame: date, symbol, ml_rank_signal
        """
        import lightgbm as lgb

        df = panel.copy()
        dates = sorted(df[date_col].unique())
        n_dates = len(dates)

        cfg = self.config
        H = cfg.label_horizon    # 3 months
        fold_size = cfg.train_months + cfg.val_months + cfg.test_months
        # 需要额外 H 个月用于标签 (最后一期可用的标签在 fold 结束前 H 个月)
        n_folds = n_dates - fold_size - H + 1

        if n_folds <= 0:
            raise ValueError(
                f"数据不足: {n_dates} 截面 < 所需 {fold_size + H} 个"
                f" (train={cfg.train_months}+val={cfg.val_months}"
                f"+test={cfg.test_months}+horizon={H})"
            )

        logger.info("=" * 56)
        logger.info("LambdaRank Walk-Forward: %d folds (Gap=%dM)", n_folds, H)
        logger.info(
            "  窗口: %dM train + %dM val + %dM test | horizon=%dM",
            cfg.train_months, cfg.val_months, cfg.test_months, H,
        )
        logger.info(
            "  特征: %d 列 | objective=%s | eval_at=%s",
            len(self._feature_cols), cfg.objective, list(cfg.eval_at),
        )
        logger.info("=" * 56)

        all_predictions: list[pd.DataFrame] = []
        importance_list: list[pd.DataFrame] = []
        self._trained_folds = 0

        for fold_idx in range(n_folds):
            # ── 时间索引 (含 Gap) ──
            #   |←── Train labels (36-3=33M) ──→| Gap 3M |← Val labels (6-3=3M) →| Gap 3M | Test |
            #   |←─────────── Train raw data (36M) ──────────→|←─── Val raw (6M) ──→| 1M |
            train_start = fold_idx
            train_end_raw = train_start + cfg.train_months          # e.g. idx 36
            train_end_label = train_end_raw - H                     # e.g. idx 33
            val_start = train_end_raw                               # idx 36
            val_end_raw = val_start + cfg.val_months                # idx 42
            val_end_label = val_end_raw - H                         # idx 39
            test_idx = val_end_raw                                  # idx 42

            # 日期集
            train_label_dates = set(dates[train_start:train_end_label])   # ~33 dates
            val_label_dates = set(dates[val_start:val_end_label])         # ~3 dates
            test_date = dates[test_idx]

            # fold 内需要的所有数据 (含 Gap 期间的特征)
            fold_data_dates = set(dates[train_start:test_idx + 1])
            fold_df = df[df[date_col].isin(fold_data_dates)].copy()

            # ── 切分 train/val/test ──
            # 注意: 特征 Rank 已在 prepare_features 中按 date 分组计算完毕,
            # 每个截面的 rank 仅依赖当天数据, 无需 fold 内重新 rank。
            train_mask = fold_df[date_col].isin(train_label_dates)
            val_mask = fold_df[date_col].isin(val_label_dates)
            test_mask = fold_df[date_col] == test_date

            # 验证特征列
            feature_cols = [c for c in self._feature_cols if c in fold_df.columns]

            # ── 提取数据 ──
            X_train = fold_df.loc[train_mask, feature_cols].astype(float)
            y_train = fold_df.loc[train_mask, "label"].astype(int)      # LambdaRank 要求 int
            X_val = fold_df.loc[val_mask, feature_cols].astype(float)
            y_val = fold_df.loc[val_mask, "label"].astype(int)
            X_test = fold_df.loc[test_mask, feature_cols].astype(float)

            # ── 样本检查 ──
            # LambdaRank 每组至少需要 2 个样本 (一个 query 内至少 2 个文档)
            train_group_count = train_mask.sum()
            val_group_count = val_mask.sum()
            if train_group_count < 500 or val_group_count < 30:
                logger.warning(
                    "Fold %d: 样本不足 (train=%d, val=%d), 跳过",
                    fold_idx, train_group_count, val_group_count,
                )
                continue

            # ── LambdaRank Group 构建 ──
            # 每个截面 = 一个 query。按日期排序 → 每期股票数 = group sizes。
            def _build_group_sizes(mask, dates_subset):
                """统计 mask 中每期非空股票数, 按日期排序返回 group list。"""
                date_series = fold_df.loc[mask, date_col]
                sizes = (
                    date_series.value_counts(sort=False)
                    .reindex(sorted(dates_subset), fill_value=0)
                )
                return [int(g) for g in sizes.values if g > 0]

            train_group_sizes = _build_group_sizes(train_mask, train_label_dates)
            val_group_sizes = _build_group_sizes(val_mask, val_label_dates)

            if not train_group_sizes or not val_group_sizes:
                logger.warning("Fold %d: group 为空, 跳过", fold_idx)
                continue

            # ── LambdaRank 训练 ──
            params = cfg.to_lgb_params()

            try:
                train_ds = lgb.Dataset(
                    X_train, label=y_train,
                    group=train_group_sizes,
                )
                val_ds = lgb.Dataset(
                    X_val, label=y_val,
                    group=val_group_sizes,
                    reference=train_ds,
                )

                model = lgb.train(
                    params={k: v for k, v in params.items()
                            if k not in ("early_stopping_rounds",)},
                    train_set=train_ds,
                    num_boost_round=cfg.n_estimators,
                    valid_sets=[train_ds, val_ds],
                    valid_names=["train", "val"],
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
                pred_df["ml_rank_signal"] = y_pred
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
                    logger.info(
                        "  Fold %3d/%d | test=%s | train=%d | val=%d | "
                        "test=%d | groups: train=%d val=%d | best_iter=%d",
                        fold_idx + 1, n_folds, str(test_date)[:10],
                        len(X_train), len(X_val), len(X_test),
                        len(train_group_sizes), len(val_group_sizes), n_iter,
                    )

            except Exception as e:
                logger.error("Fold %d 训练失败: %s", fold_idx, e)
                continue

        if not all_predictions:
            raise RuntimeError("所有 Fold 均训练失败。请检查数据和参数。")

        # ── 拼接 ──
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
            "LambdaRank Walk-Forward 完成: %d/%d folds 成功, OOS %d 条",
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
        一键运行 LambdaRank ML Pipeline。

        Pipeline:
          1. prepare_labels()  — 3M forward return → rank
          2. prepare_features() — rank + Δ1M/Δ3M + categorical
          3. walk_forward_train() — LambdaRank + Gap 滚动训练

        Returns
        -------
        pd.DataFrame: date, symbol, ml_rank_signal
        """
        logger.info("=" * 56)
        logger.info("LightGBM LambdaRank Engine V2 — Pipeline 启动")
        logger.info("  objective=lambdarank | horizon=%dM | gap=%dM",
                     self.config.label_horizon, self.config.label_horizon)
        logger.info("=" * 56)

        # Step 1: 3M 标签
        df = self.prepare_labels(
            panel,
            date_col=date_col,
            symbol_col=symbol_col,
            close_col=close_col,
        )

        # Step 2: 扩展特征 (Rank + Delta + Categorical)
        df = self.prepare_features(
            df,
            blended=blended,
            date_col=date_col,
            symbol_col=symbol_col,
        )

        # Step 3: LambdaRank Walk-Forward
        predictions = self.walk_forward_train(
            df,
            date_col=date_col,
            symbol_col=symbol_col,
        )

        return predictions

    # ═══════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════

    def get_feature_importance(self) -> pd.DataFrame | None:
        return self._feature_importance

    def to_markdown_report(self) -> str:
        cfg = self.config
        lines = [
            "## LambdaRank Alpha Engine V2 — 训练报告",
            "",
            f"- **目标函数:** {cfg.objective} | **Horizon:** {cfg.label_horizon}M",
            f"- **Gap:** {cfg.label_horizon}M (防多期标签泄漏)",
            f"- **训练 Folds:** {self._trained_folds}",
            f"- **特征列数:** {len(self._feature_cols)}",
            f"- **窗口:** {cfg.train_months}M train + "
            f"{cfg.val_months}M val + {cfg.test_months}M test",
            f"- **eval_at:** {list(cfg.eval_at)}",
            f"- **超参数:** max_depth={cfg.max_depth}, "
            f"num_leaves={cfg.num_leaves}, lr={cfg.learning_rate}",
            f"- **差分特征:** {'启用' if cfg.use_delta_features else '关闭'}",
            f"- **类别特征:** {'启用' if cfg.use_categorical_features else '关闭'}",
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
