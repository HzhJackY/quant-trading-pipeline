"""
动态权重分配模块 (Dynamic Weight Allocation).

在 Split-Universe 双模型系统产出大盘/小盘独立 Alpha 信号后,
通过滚动均值-方差优化 (Rolling Mean-Variance Optimization)
动态计算两个子策略的资本分配权重, 最大化组合夏普比率。

三种权重方案:
  1. 最优 Sharpe 动态权重 — 滚动 MVO, 边界约束 [0.3, 0.7]
  2. 风险平价 (Risk Parity)  — w_i ∝ 1/σ_i, 同样滚动计算
  3. 固定 50/50             — 朴素等权基准

输出:
  - 动态权重序列 (DataFrame)
  - 权重演变堆叠面积图 (stacked area chart)
  - 三种方案绩效对比表 (年化收益 / 最大回撤 / Sharpe)

用法:
  from factor_research.dynamic_weight import (
      build_sub_universe_returns, DynamicWeightOptimizer,
      compare_all_strategies, plot_weight_evolution,
  )
  large_rets, small_rets = build_sub_universe_returns(blended_panel, panel_with_returns)
  optimizer = DynamicWeightOptimizer(window=60, bounds=(0.3, 0.7))
  weights_df = optimizer.fit(large_rets, small_rets)
  comparison = compare_all_strategies(large_rets, small_rets, weights_df)
  plot_weight_evolution(weights_df, save_path="output/weight_evolution.png")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── Logger 配置 ──────────────────────────────────────────
logger = logging.getLogger("dynamic_weight")
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


# ═══════════════════════════════════════════════════════════
# 1. 提取子域策略收益
# ═══════════════════════════════════════════════════════════

def build_sub_universe_returns(
    blended_panel: pd.DataFrame,
    panel_with_returns: pd.DataFrame,
    alpha_col: str = "alpha_signal",
    universe_col: str = "universe",
    return_col: str = "forward_return_1m",
    date_col: str = "date",
    symbol_col: str = "symbol",
    top_quantile: float = 0.3,
    min_stocks: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """
    从 Split-Universe 拼接面板中提取大盘和小盘的策略收益序列。

    方法:
      在每个截面上, 按 alpha_signal 从高到低排序,
      取前 top_quantile (默认 30%) 的股票,
      计算等权平均收益作为该子域的策略收益。

    为什么用 top_quantile 而不是取所有股票:
      - Alpha 信号的目的是选股, 不是持有全市场
      - 只取排名靠前的股票才能体现 Alpha 的选股能力
      - 30% 是常见的多因子选股比例 (300只选90只)

    参数
    ----
    blended_panel : pd.DataFrame
        Split-Universe 拼接面板, 含 alpha_signal, universe 列。
    panel_with_returns : pd.DataFrame
        包含 forward_return_1m 的原始面板 (用于合并收益)。
    alpha_col : str
        Alpha 信号列名, 默认 "alpha_signal"。
    universe_col : str
        子域标签列名, 默认 "universe"。取值为 "大盘"/"小盘"。
    return_col : str
        下期收益列名, 默认 "forward_return_1m"。
    date_col : str
        日期列名。
    symbol_col : str
        股票代码列名。
    top_quantile : float
        每期选取的股票比例, 默认 0.3 (前 30%)。
    min_stocks : int
        单期最少股票数, 低于此数跳过该日期。

    返回
    ----
    (large_cap_returns, small_cap_returns) : tuple[pd.Series, pd.Series]
        index=date, values=该期该子域 top 组合的等权平均收益。
    """
    # 1. 合并收益到 blended panel
    if return_col not in blended_panel.columns:
        ret_cols = [date_col, symbol_col, return_col]
        available_ret_cols = [c for c in ret_cols if c in panel_with_returns.columns]
        if return_col not in available_ret_cols:
            raise KeyError(
                f"panel_with_returns 中未找到 '{return_col}' 列。"
                f"可用列: {panel_with_returns.columns.tolist()}"
            )
        fwd = panel_with_returns[available_ret_cols].dropna(
            subset=[return_col]
        )
        df = blended_panel.merge(fwd, on=[date_col, symbol_col], how="left")
    else:
        df = blended_panel.copy()

    df = df.dropna(subset=[return_col, alpha_col, universe_col])

    # 2. 按期 + 子域分组, 提取 top quantile 收益
    large_returns: dict[pd.Timestamp, float] = {}
    small_returns: dict[pd.Timestamp, float] = {}

    for dt, date_grp in df.groupby(date_col):
        for uni_name, target_dict in [("大盘", large_returns), ("小盘", small_returns)]:
            uni_data = date_grp[date_grp[universe_col] == uni_name]
            if len(uni_data) < min_stocks:
                continue

            # 按 alpha_signal 降序排列, 取前 top_quantile
            n_select = max(min_stocks, int(len(uni_data) * top_quantile))
            top_stocks = uni_data.nlargest(n_select, alpha_col)

            avg_ret = top_stocks[return_col].mean()
            if not np.isnan(avg_ret):
                target_dict[dt] = float(avg_ret)

    # 3. 构造 Series, 对齐日期
    large_series = pd.Series(large_returns, name="large_cap_return").sort_index()
    small_series = pd.Series(small_returns, name="small_cap_return").sort_index()

    # 对齐: 只保留两个序列都存在的日期
    common_dates = large_series.index.intersection(small_series.index)
    large_series = large_series.reindex(common_dates)
    small_series = small_series.reindex(common_dates)

    logger.info(
        "提取子域策略收益: 大盘=%d 期, 小盘=%d 期, 对齐后=%d 期",
        len(large_returns), len(small_returns), len(common_dates),
    )
    logger.info(
        "  大盘 平均收益=%.4f%%  标准差=%.4f%%",
        large_series.mean() * 100, large_series.std() * 100,
    )
    logger.info(
        "  小盘 平均收益=%.4f%%  标准差=%.4f%%",
        small_series.mean() * 100, small_series.std() * 100,
    )

    return large_series, small_series


# ═══════════════════════════════════════════════════════════
# 2. 滚动均值-方差优化器
# ═══════════════════════════════════════════════════════════

class DynamicWeightOptimizer:
    """
    滚动均值-方差优化器 (Rolling Mean-Variance Optimizer)。

    对于两个资产的收益率序列, 在每个时间点 T 使用前 window 期
    历史数据估计期望收益向量 mu 和协方差矩阵 Sigma, 然后求解使
    夏普比率最大化的最优权重。

    优化问题:
      min  -Sharpe(w) = -(w^T mu) / sqrt(w^T Sigma w)
      s.t. sum(w) = 1
           0.3 <= w_i <= 0.7  (防止过拟合产生极端仓位)

    为什么用 scipy.optimize.minimize 而不是解析解:
      - 带约束的二次规划, SLSQP 算法适合这种小规模问题
      - 2 个资产时优化极快, 但代码保持通用以便未来扩展更多资产

    参数
    ----
    window : int
        滚动窗口大小 (期数)。
        月频数据: 建议 36-60 (3-5年历史)
        日频数据: 建议 60 (约3个月交易数据)
        默认 60。
    bounds : tuple[float, float]
        单边权重下限和上限, 默认 (0.3, 0.7)。
        设为 (0.0, 1.0) 则不做约束 (允许极端仓位)。
    rf : float
        无风险利率 (年化), 默认 0。对于 A 股月度回测,
        无风险利率影响很小, 设为 0 是常见做法。
    freq : str
        数据频率, "M"=月频 (默认), "D"=日频。
        影响年化计算时的转换系数。
    """

    def __init__(
        self,
        window: int = 60,
        bounds: tuple[float, float] = (0.3, 0.7),
        rf: float = 0.0,
        freq: str = "M",
    ):
        if window < 2:
            raise ValueError(f"window 必须 >= 2, 当前值: {window}")
        if bounds[0] < 0 or bounds[1] > 1 or bounds[0] >= bounds[1]:
            raise ValueError(
                f"bounds 必须在 [0, 1] 且 lower < upper, 当前值: {bounds}"
            )

        self.window = window
        self.bounds = bounds
        self.rf = rf
        self.freq = freq
        # 年化系数: 月频×12, 日频×252
        self._periods_per_year = {"M": 12, "D": 252, "W": 52}.get(freq, 12)

        logger.info(
            "DynamicWeightOptimizer: window=%d 期, bounds=[%.1f, %.1f], "
            "rf=%.2f%%, freq=%s",
            window, bounds[0], bounds[1], rf * 100, freq,
        )

    # ── 目标函数 ──────────────────────────────────────────

    def _neg_sharpe_ratio(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> float:
        """
        负夏普比率 (供 scipy.optimize.minimize 最小化)。

        Portfolio_Return    = w^T * mu
        Portfolio_Volatility = sqrt(w^T * Sigma * w)
        Sharpe_Ratio         = (Portfolio_Return - rf_per_period) / Portfolio_Volatility

        由于我们要最大化夏普比率, 而 scipy 做最小化,
        所以返回 -Sharpe_Ratio。

        参数
        ----
        weights : np.ndarray, shape (n_assets,)
            资产权重向量。
        expected_returns : np.ndarray, shape (n_assets,)
            期望收益向量 (样本均值)。
        cov_matrix : np.ndarray, shape (n_assets, n_assets)
            协方差矩阵。

        返回
        ----
        float : -Sharpe_Ratio
        """
        # 将年化无风险利率折算为每期
        rf_per_period = self.rf / self._periods_per_year

        port_return = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        if port_vol < 1e-10:
            # 波动率接近零, 无法计算 Sharpe, 返回大惩罚
            return 1e6

        sharpe = (port_return - rf_per_period) / port_vol
        return float(-sharpe)

    # ── 单期优化 ──────────────────────────────────────────

    def _sharpe_ratio(
        self,
        w_large: float,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> float:
        """
        给定大盘权重, 计算组合夏普比率。

        对于 2 资产问题, w_small = 1 - w_large, 优化退化为 1D 线搜索。
        这比通用的 SLSQP 算法更快、更稳定, 且不会受梯度估计精度影响。

        参数
        ----
        w_large : float
            大盘权重 ∈ [bounds[0], bounds[1]]。
        mu : np.ndarray, shape (2,)
            两个资产的期望收益。
        sigma : np.ndarray, shape (2, 2)
            协方差矩阵。

        返回
        ----
        float : Sharpe Ratio (非负向, 用于最大化)
        """
        w = np.array([w_large, 1.0 - w_large])
        rf_per_period = self.rf / self._periods_per_year
        port_return = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
        if port_vol < 1e-12:
            return -1e6  # 惩罚近零波动率
        return float((port_return - rf_per_period) / port_vol)

    def _optimize_single(
        self,
        returns_window: pd.DataFrame,
    ) -> tuple[np.ndarray, bool]:
        """
        对单个滚动窗口求解最优权重。

        对于 2 资产情况 (大盘/小盘):
          - 将 w_small = 1 - w_large 代入, 问题退化为 1D 线搜索
          - 在 [bounds[0], bounds[1]] 区间内以 0.005 (0.5%) 步长网格搜索
          - 选 Sharpe Ratio 最大的 w_large 作为最优解
          - 网格搜索保证全局最优 (在离散精度内), 不会像梯度法陷于局部

        对于 >2 资产情况 (未来扩展):
          - 回退到 SLSQP 算法 (scipy.optimize.minimize)

        参数
        ----
        returns_window : pd.DataFrame
            滚动窗口内的收益率, 列: ["large_cap_return", "small_cap_return"],
            行: 各期 (window 行)。

        返回
        ----
        (optimal_weights, success_flag) : tuple[np.ndarray, bool]
        """
        n_assets = returns_window.shape[1]

        # 1. 估计 mu 和 Sigma
        mu = returns_window.mean().values
        sigma = returns_window.cov().values

        # 协方差矩阵正则化: 如果接近奇异, 加微小对角扰动
        try:
            eigvals = np.linalg.eigvalsh(sigma)
            if eigvals.min() < 1e-12:
                sigma = sigma + np.eye(n_assets) * 1e-8
        except Exception:
            sigma = sigma + np.eye(n_assets) * 1e-8

        # ── 2 资产: 网格搜索 (推荐路径) ──
        if n_assets == 2:
            lo, hi = self.bounds
            # 以 0.5% 步长搜索 (201 个候选点), 计算量极小但精度足够
            n_steps = 201
            candidates = np.linspace(lo, hi, n_steps)
            best_w = 0.5
            best_sharpe = -np.inf

            for w_large in candidates:
                sr = self._sharpe_ratio(w_large, mu, sigma)
                if sr > best_sharpe:
                    best_sharpe = sr
                    best_w = float(w_large)

            w_opt = np.array([best_w, 1.0 - best_w])
            return w_opt, True

        # ── >2 资产: SLSQP (未来扩展) ──
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        )
        bounds_list = [self.bounds] * n_assets
        initial_guess = np.ones(n_assets) / n_assets

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = minimize(
                    self._neg_sharpe_ratio,
                    x0=initial_guess,
                    args=(mu, sigma),
                    method="SLSQP",
                    bounds=bounds_list,
                    constraints=constraints,
                    options={"ftol": 1e-12, "maxiter": 200, "disp": False},
                )

            if result.success:
                w = result.x
                w = np.clip(w, self.bounds[0], self.bounds[1])
                w = w / w.sum()
                return w, True
            else:
                return initial_guess, False

        except Exception:
            return initial_guess, False

    # ── 滚动优化主循环 ───────────────────────────────────

    def fit(
        self,
        large_returns: pd.Series,
        small_returns: pd.Series,
    ) -> pd.DataFrame:
        """
        滚动均值-方差优化主循环。

        对于每个时间点 T (从 window 期开始到最后一期):
          Step A: 提取 [T-window, T-1] 的历史收益率作为训练样本
          Step B: 计算样本协方差矩阵和均值向量
          Step C: 求解最优权重 (W_large_T, W_small_T)
          Step D: 使用该权重乘以 T 时刻的真实收益, 得到样本外组合收益
            Combined_Return_T = W_large_T × r_large_T + W_small_T × r_small_T

        注意这是严格的样本外 (Out-of-Sample) 测试:
          权重仅使用 T 时刻之前的信息, 不包含 T 时刻的收益。

        参数
        ----
        large_returns : pd.Series
            大盘策略收益序列, index=date。
        small_returns : pd.Series
            小盘策略收益序列, index=date。

        返回
        ----
        pd.DataFrame, 列:
          - date: 调仓日期
          - W_large: 大盘权重
          - W_small: 小盘权重
          - large_ret: 大盘当期收益 (样本外)
          - small_ret: 小盘当期收益 (样本外)
          - combined_ret: 组合收益 (样本外)
          - opt_success: 优化是否成功
          - cum_nav: 累计净值 (从 1.0 开始)
        """
        # 对齐日期
        common_dates = large_returns.index.intersection(small_returns.index)
        large_aligned = large_returns.reindex(common_dates)
        small_aligned = small_returns.reindex(common_dates)

        n_total = len(common_dates)
        if n_total <= self.window:
            raise ValueError(
                f"数据期数 ({n_total}) <= 滚动窗口 ({self.window}), "
                f"无法做滚动优化。请减小 window 或使用更长回测区间。"
            )

        logger.info(
            "开始滚动优化: %d 期数据, window=%d, "
            "产生 %d 个样本外权重",
            n_total, self.window, n_total - self.window,
        )

        # 组装收益率 DataFrame
        returns_df = pd.DataFrame({
            "large_cap_return": large_aligned.values,
            "small_cap_return": small_aligned.values,
        }, index=common_dates)

        records = []
        n_fallback = 0

        for t in range(self.window, n_total):
            # Step A: 训练窗口 [t-window, t-1]
            train_start = t - self.window
            train_end = t - 1
            train_window = returns_df.iloc[train_start:train_end + 1]

            # Step B+C: 优化
            opt_weights, success = self._optimize_single(train_window)
            if not success:
                n_fallback += 1

            w_large = float(opt_weights[0])
            w_small = float(opt_weights[1])

            # Step D: 样本外收益
            r_large_t = float(returns_df.iloc[t]["large_cap_return"])
            r_small_t = float(returns_df.iloc[t]["small_cap_return"])
            combined_ret = w_large * r_large_t + w_small * r_small_t

            records.append({
                "date": common_dates[t],
                "W_large": w_large,
                "W_small": w_small,
                "large_ret": r_large_t,
                "small_ret": r_small_t,
                "combined_ret": combined_ret,
                "opt_success": success,
            })

        result = pd.DataFrame(records)

        # 计算累计净值 (从 1 开始)
        result["cum_nav"] = (1 + result["combined_ret"]).cumprod()

        # ── 日志输出 ──
        avg_w_large = result["W_large"].mean()
        avg_w_small = result["W_small"].mean()
        success_rate = result["opt_success"].mean()

        logger.info("滚动优化完成:")
        logger.info("  样本外期数: %d", len(result))
        logger.info("  平均权重:   大盘=%.2f%%  小盘=%.2f%%",
                     avg_w_large * 100, avg_w_small * 100)
        logger.info("  权重范围:   大盘[%.2f%%, %.2f%%]  小盘[%.2f%%, %.2f%%]",
                     result["W_large"].min() * 100, result["W_large"].max() * 100,
                     result["W_small"].min() * 100, result["W_small"].max() * 100)
        logger.info("  优化成功率: %.1f%% (%d 次回退等权)",
                     success_rate * 100, n_fallback)
        logger.info("  组合平均收益: %.4f%%  标准差: %.4f%%",
                     result["combined_ret"].mean() * 100,
                     result["combined_ret"].std() * 100)

        return result

    # ── 风险平价权重 (静态方法) ────────────────────────────

    @staticmethod
    def risk_parity_weights(returns_window: pd.DataFrame) -> np.ndarray:
        """
        风险平价权重: w_i ∝ 1/σ_i。

        风险平价的核心思想:
          让每个资产对组合风险的贡献相等。
          对于两个资产, 这等价于按波动率的倒数分配权重:
            w_i = (1/σ_i) / Σ(1/σ_j)

        与均值-方差优化的区别:
          - MVO 需要估计期望收益 (极不可靠, estimation error 巨大)
          - 风险平价只需要估计波动率 (相对可靠)
          - 实践中风险平价往往比 MVO 更稳健 (MVO 容易过拟合收益估计)

        参数
        ----
        returns_window : pd.DataFrame
            滚动窗口内的收益率。

        返回
        ----
        np.ndarray : 风险平价权重向量。
        """
        vols = returns_window.std().values
        # 防止除零
        vols = np.where(vols < 1e-10, 1e-10, vols)
        inv_vols = 1.0 / vols
        weights = inv_vols / inv_vols.sum()
        return weights


# ═══════════════════════════════════════════════════════════
# 3. 绩效对比
# ═══════════════════════════════════════════════════════════

def compute_performance_metrics(
    returns: pd.Series,
    freq: str = "M",
    rf: float = 0.0,
) -> dict:
    """
    从收益序列计算绩效指标。

    参数
    ----
    returns : pd.Series
        策略收益序列 (每期)。
    freq : str
        频率: "M"=月, "D"=日, "W"=周。
    rf : float
        无风险利率 (年化)。

    返回
    ----
    dict: {Annualized_Return, Volatility, Sharpe_Ratio, Max_Drawdown, Calmar_Ratio, Win_Rate}
    """
    periods_per_year = {"M": 12, "D": 252, "W": 52}.get(freq, 12)

    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else 0.0

    # 最大回撤
    nav = (1 + returns).cumprod()
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_dd = float(drawdown.min())

    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0.0
    win_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0

    return {
        "Annualized_Return": round(ann_return, 4),
        "Volatility": round(ann_vol, 4),
        "Sharpe_Ratio": round(sharpe, 4),
        "Max_Drawdown": round(max_dd, 4),
        "Calmar_Ratio": round(calmar, 4),
        "Win_Rate": round(win_rate, 4),
        "Periods": len(returns),
    }


def compare_all_strategies(
    large_returns: pd.Series,
    small_returns: pd.Series,
    weights_df: pd.DataFrame,
    window: int = 60,
    freq: str = "M",
    rf: float = 0.0,
) -> pd.DataFrame:
    """
    对比三种权重方案: 最优Sharpe动态权重 / 风险平价 / 50-50。

    对所有方案, 使用相同的样本外期间 (weights_df 的日期范围),
    确保对比公平。

    参数
    ----
    large_returns : pd.Series
        大盘策略收益, index=date。
    small_returns : pd.Series
        小盘策略收益, index=date。
    weights_df : pd.DataFrame
        DynamicWeightOptimizer.fit() 的输出。
    window : int
        滚动窗口大小 (用于风险平价滚动计算)。
    freq : str
        频率。
    rf : float
        无风险利率。

    返回
    ----
    pd.DataFrame: 三种方案的绩效对比表。
    """
    # ── 提取 MVO 的 OOS 收益 ──
    mvo_combined = weights_df.set_index("date")["combined_ret"]

    # ── 对齐到 MVO 日期: 用于等权方案 ──
    oos_dates = mvo_combined.index
    large_oos = large_returns.reindex(oos_dates)
    small_oos = small_returns.reindex(oos_dates)

    # ── 方案 1: 最优 Sharpe 动态权重 (MVO) ──
    perf_mvo = compute_performance_metrics(mvo_combined, freq=freq, rf=rf)

    # ── 方案 2: 风险平价 (滚动) ──
    # 重要: 风险平价同样需要前 window 期历史来估计波动率,
    # 所以使用完整的历史收益序列 (large_returns/small_returns),
    # 而非只截取 OOS 部分。
    all_dates = large_returns.index.intersection(small_returns.index).sort_values()
    full_l = large_returns.reindex(all_dates)
    full_s = small_returns.reindex(all_dates)

    # 找到 OOS 起始位置在完整序列中的索引
    oos_start_date = oos_dates[0]
    try:
        # 找到 >= oos_start_date 的第一个位置
        oos_start_idx = all_dates.get_indexer([oos_start_date])[0]
    except Exception:
        oos_start_idx = window

    rp_combined_list = []
    rp_dates = []
    n_full = len(all_dates)

    for t in range(oos_start_idx, n_full):
        hist_start = max(0, t - window)
        hist_end = t - 1
        if hist_end - hist_start < 2:
            w_l, w_s = 0.5, 0.5
        else:
            hist_data = pd.DataFrame({
                "large": full_l.iloc[hist_start:hist_end + 1].values,
                "small": full_s.iloc[hist_start:hist_end + 1].values,
            })
            rp_w = DynamicWeightOptimizer.risk_parity_weights(hist_data)
            w_l, w_s = float(rp_w[0]), float(rp_w[1])

        combined = w_l * float(full_l.iloc[t]) + w_s * float(full_s.iloc[t])
        rp_combined_list.append(combined)
        rp_dates.append(all_dates[t])

    rp_combined = pd.Series(rp_combined_list, index=rp_dates)
    perf_rp = compute_performance_metrics(rp_combined, freq=freq, rf=rf)

    # ── 方案 3: 固定 50/50 ──
    equal_combined = (large_oos + small_oos) / 2.0
    equal_combined = equal_combined.dropna()
    perf_equal = compute_performance_metrics(equal_combined, freq=freq, rf=rf)

    # ── 组装对比表 ──
    comparison = pd.DataFrame({
        "方案": ["最优Sharpe动态权重", "风险平价", "固定50/50"],
        "年化收益": [
            f"{perf_mvo['Annualized_Return']:.2%}",
            f"{perf_rp['Annualized_Return']:.2%}",
            f"{perf_equal['Annualized_Return']:.2%}",
        ],
        "年化波动": [
            f"{perf_mvo['Volatility']:.2%}",
            f"{perf_rp['Volatility']:.2%}",
            f"{perf_equal['Volatility']:.2%}",
        ],
        "Sharpe": [
            f"{perf_mvo['Sharpe_Ratio']:.4f}",
            f"{perf_rp['Sharpe_Ratio']:.4f}",
            f"{perf_equal['Sharpe_Ratio']:.4f}",
        ],
        "最大回撤": [
            f"{perf_mvo['Max_Drawdown']:.2%}",
            f"{perf_rp['Max_Drawdown']:.2%}",
            f"{perf_equal['Max_Drawdown']:.2%}",
        ],
        "Calmar": [
            f"{perf_mvo['Calmar_Ratio']:.4f}",
            f"{perf_rp['Calmar_Ratio']:.4f}",
            f"{perf_equal['Calmar_Ratio']:.4f}",
        ],
        "胜率": [
            f"{perf_mvo['Win_Rate']:.1%}",
            f"{perf_rp['Win_Rate']:.1%}",
            f"{perf_equal['Win_Rate']:.1%}",
        ],
    })

    # ── 日志输出 ──
    logger.info("\n" + "=" * 72)
    logger.info("三种权重方案绩效对比")
    logger.info("=" * 72)
    header = (
        f"{'方案':<20} {'年化收益':>10} {'年化波动':>10} "
        f"{'Sharpe':>8} {'最大回撤':>10} {'Calmar':>8} {'胜率':>8}"
    )
    logger.info(header)
    logger.info("-" * 72)
    for _, row in comparison.iterrows():
        logger.info(
            f"{row['方案']:<20} {row['年化收益']:>10} {row['年化波动']:>10} "
            f"{row['Sharpe']:>8} {row['最大回撤']:>10} {row['Calmar']:>8} "
            f"{row['胜率']:>8}"
        )
    logger.info("=" * 72)

    return comparison


# ═══════════════════════════════════════════════════════════
# 4. 可视化
# ═══════════════════════════════════════════════════════════

def plot_weight_evolution(
    weights_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
) -> "plt.Figure":
    """
    权重演变堆叠面积图 (Stacked Area Chart)。

    上图: 权重堆叠面积图 — 大盘 (红色) / 小盘 (蓝色)
      直观展示回测期内大小盘权重的动态变化轨迹。
      当大盘权重上升时, 说明 MVO 判断近期大盘 Alpha 更稳定;
      当小盘权重上升时, 说明小盘 Alpha 机会更大。

    下图: 累计净值曲线 — 对比三种方案
      最优Sharpe动态权重 / 风险平价 / 固定50/50

    参数
    ----
    weights_df : pd.DataFrame
        DynamicWeightOptimizer.fit() 的输出。
    save_path : str | None
        保存路径 (PNG), 为 None 则显示。
    figsize : tuple
        画布大小。

    返回
    ----
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import matplotlib
        matplotlib.rcParams["font.sans-serif"] = [
            "SimHei", "Microsoft YaHei", "DejaVu Sans"
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except ImportError:
        logger.warning("matplotlib 不可用, 跳过绘图")
        raise

    dates = pd.to_datetime(weights_df["date"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [1, 1.2]})

    # ── 上图: 权重堆叠面积图 ──
    ax1.stackplot(
        dates,
        weights_df["W_large"],
        weights_df["W_small"],
        labels=["大盘 (Large Cap)", "小盘 (Small Cap)"],
        colors=["#D62728", "#1F77B4"],
        alpha=0.85,
    )

    # 标注平均权重水平线
    avg_large = weights_df["W_large"].mean()
    ax1.axhline(
        y=avg_large, color="#D62728", linestyle="--", alpha=0.6, linewidth=1,
    )
    ax1.text(
        dates.iloc[0], avg_large + 0.02,
        f"大盘均值={avg_large:.1%}",
        color="#D62728", fontsize=9, va="bottom",
    )

    ax1.set_ylabel("权重分配", fontsize=12)
    ax1.set_title("动态权重演变 — 滚动均值-方差优化 (60期窗口)", fontsize=14)
    ax1.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.grid(True, alpha=0.3, axis="y")

    # ── 下图: 净值曲线对比 ──
    # MVO 动态权重净值
    ax2.plot(
        dates, weights_df["cum_nav"],
        color="#2CA02C", linewidth=2, label="最优Sharpe动态权重",
    )

    # 风险平价净值 (滚动计算)
    if "large_ret" in weights_df.columns and "small_ret" in weights_df.columns:
        large_arr = weights_df["large_ret"].values
        small_arr = weights_df["small_ret"].values
        n = len(large_arr)
        window = 60

        # 风险平价
        rp_rets = []
        for t in range(n):
            if t < 2:
                w_l, w_s = 0.5, 0.5
            else:
                hist_start = max(0, t - window)
                hist_data = pd.DataFrame({
                    "large": large_arr[hist_start:t],
                    "small": small_arr[hist_start:t],
                })
                rp_w = DynamicWeightOptimizer.risk_parity_weights(hist_data)
                w_l, w_s = float(rp_w[0]), float(rp_w[1])
            rp_rets.append(w_l * large_arr[t] + w_s * small_arr[t])
        rp_nav = (1 + pd.Series(rp_rets)).cumprod()
        ax2.plot(
            dates, rp_nav,
            color="#FF7F0E", linewidth=1.8, linestyle="--",
            label="风险平价",
        )

    # 50/50 净值
    equal_nav = (1 + (weights_df["large_ret"] + weights_df["small_ret"]) / 2).cumprod()
    ax2.plot(
        dates, equal_nav,
        color="#9467BD", linewidth=1.5, linestyle=":",
        label="固定50/50",
    )

    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax2.set_ylabel("累计净值", fontsize=12)
    ax2.set_xlabel("日期", fontsize=12)
    ax2.set_title("三种权重方案 — 净值曲线对比", fontsize=14)
    ax2.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("权重演变图已保存到: %s", save_path)

    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════
# 5. 便捷函数: 一键运行
# ═══════════════════════════════════════════════════════════

def run_dynamic_weight_analysis(
    blended_panel: pd.DataFrame,
    panel_with_returns: pd.DataFrame,
    window: int = 60,
    bounds: tuple[float, float] = (0.3, 0.7),
    top_quantile: float = 0.3,
    output_dir: str = "output",
    freq: str = "M",
    rf: float = 0.0,
) -> dict:
    """
    一键运行动态权重分析流水线。

    参数
    ----
    blended_panel : pd.DataFrame
        Split-Universe 拼接面板 (含 alpha_signal, universe 列)。
    panel_with_returns : pd.DataFrame
        含 forward_return_1m 的原始面板。
    window : int
        滚动窗口大小。
    bounds : tuple
        权重边界 (lower, upper)。
    top_quantile : float
        每期选取的股票比例 (用于构建子域策略收益)。
    output_dir : str
        输出目录。
    freq : str
        频率。
    rf : float
        无风险利率。

    返回
    ----
    dict: {
        "weights_df": 动态权重 DataFrame,
        "comparison": 三种方案对比表,
        "large_returns": 大盘策略收益,
        "small_returns": 小盘策略收益,
    }
    """
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # Step 1: 提取子域策略收益
    logger.info("Step 1: 提取子域策略收益...")
    large_rets, small_rets = build_sub_universe_returns(
        blended_panel, panel_with_returns,
        top_quantile=top_quantile,
    )

    # Step 2: 滚动均值-方差优化
    logger.info("\nStep 2: 滚动均值-方差优化...")
    optimizer = DynamicWeightOptimizer(
        window=window, bounds=bounds, rf=rf, freq=freq,
    )
    weights_df = optimizer.fit(large_rets, small_rets)

    # Step 3: 三种方案对比
    logger.info("\nStep 3: 绩效对比...")
    comparison = compare_all_strategies(
        large_rets, small_rets, weights_df,
        window=window, freq=freq, rf=rf,
    )

    # Step 4: 可视化
    logger.info("\nStep 4: 生成图表...")
    try:
        plot_weight_evolution(
            weights_df,
            save_path=str(out / "dynamic_weight_evolution.png"),
        )
    except Exception as e:
        logger.warning("绘图失败: %s", e)

    # 保存结果
    weights_df.to_csv(
        out / "dynamic_weights.csv",
        index=False, encoding="utf-8-sig",
    )
    comparison.to_csv(
        out / "dynamic_weight_comparison.csv",
        index=False, encoding="utf-8-sig",
    )

    logger.info("\n所有输出已保存到 %s/", output_dir)
    logger.info("  - dynamic_weights.csv       (动态权重序列)")
    logger.info("  - dynamic_weight_comparison.csv (绩效对比表)")
    logger.info("  - dynamic_weight_evolution.png  (权重演变图)")

    return {
        "weights_df": weights_df,
        "comparison": comparison,
        "large_returns": large_rets,
        "small_returns": small_rets,
    }
