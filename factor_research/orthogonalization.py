"""
Gram-Schmidt 正交化 (基于回归残差法)。

对多重共线性严重的因子矩阵做正交化处理, 确保各因子携带的信息互斥,
解决当前"贪心丢弃高相关因子"导致的信息损失问题。

方法:
  1. 滚动 24 月 IC_IR 动态排序
  2. 按 |IC_IR| 降序排列因子
  3. Gram-Schmidt 正交化: 每个因子对前面所有已正交因子做 OLS 回归, 取残差
  4. IC_IR 加权合成

算法细节:
  - 使用 numpy.linalg.lstsq 求解 OLS, 不含截距项 (因子已截面 Z-score 标准化,
    均值为 0, 无需截距)
  - 当残差方差 ≈ 0 (因子被前面因子完全解释) 时, 权重自动坍塌为 0,
    避免矩阵奇异
  - 逐截面独立正交化, 不跨截面传播误差

References:
  Gram (1883), Schmidt (1907) — 经典 Gram-Schmidt 正交化过程
  Barra Risk Model — 因子正交化是 Barra 多因子模型的标准预处理
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ConstantInputWarning

logger = logging.getLogger(__name__)

# Suppress scipy warning when a cross-section has constant input
# (expected for some factor-date combos with few unique values)
warnings.filterwarnings("ignore", category=ConstantInputWarning)


def compute_rolling_ic_ir(
    df: pd.DataFrame,
    factor_cols: list[str],
    return_col: str = "forward_return_1m",
    date_col: str = "date",
    rolling_window: int = 24,
    min_stocks: int = 20,
) -> dict[pd.Timestamp, dict[str, float]]:
    """
    计算每个因子在滚动窗口上的 IC_IR (时间序列维度)。

    两步法:
      1. 预计算每个因子的逐期 Rank IC 序列 (仅遍历一次数据)
      2. 在每个日期上, 对滚动窗口内的 IC 序列计算 IC_IR

    Parameters
    ----------
    df : pd.DataFrame
        因子面板 (含因子列 + forward_return)。
    factor_cols : list[str]
        因子列名 (存在于 df.columns 的实际列名, 如 BP_neutral_z)。
    return_col : str
        下期收益列名。
    date_col : str
        日期列名。
    rolling_window : int
        滚动窗口期数 (默认 24 个月)。
    min_stocks : int
        计算 Rank IC 时单截面最少股票数。

    Returns
    -------
    dict[date, dict[factor, ic_ir]]
        每个日期上各因子的滚动 IC_IR 值。
        因子缺失或数据不足时为 0.0。
    """
    # ── Step 1: 预计算每个因子的逐期 IC 序列 ──
    ic_series: dict[str, pd.Series] = {}
    for col in factor_cols:
        ic_vals: dict[pd.Timestamp, float] = {}
        for dt, grp in df.groupby(date_col):
            sub = grp[[col, return_col]].dropna()
            if len(sub) >= min_stocks:
                try:
                    ic, _ = spearmanr(sub[col], sub[return_col])
                    if not np.isnan(ic):
                        ic_vals[dt] = ic
                except Exception:
                    pass
        ic_series[col] = pd.Series(ic_vals).sort_index()

    # ── Step 2: 逐日期计算滚动 IC_IR ──
    dates = sorted(df[date_col].unique())
    result: dict[pd.Timestamp, dict[str, float]] = {}

    for dt in dates:
        result[dt] = {}
        for col in factor_cols:
            series = ic_series.get(col)
            if series is None or len(series) == 0:
                result[dt][col] = 0.0
                continue

            # 用 nearest backward 定位 dt 在 IC 序列中的位置
            idx_array = series.index.values
            dt64 = pd.Timestamp(dt).to_datetime64()
            # np.searchsorted: 找到 dt 应插入的位置
            pos = np.searchsorted(idx_array, dt64, side="right") - 1
            if pos < 0:
                result[dt][col] = 0.0
                continue

            window_start = max(0, pos - rolling_window + 1)
            window_ic = series.iloc[window_start:pos + 1]

            if len(window_ic) < 2:
                result[dt][col] = 0.0
                continue

            mean_ic = float(np.mean(window_ic))
            std_ic = float(np.std(window_ic, ddof=1))
            result[dt][col] = mean_ic / std_ic if std_ic > 1e-10 else 0.0

    return result


def _residualize(
    y: np.ndarray,
    X: np.ndarray,
    min_variance: float = 1e-10,
) -> np.ndarray:
    """
    OLS 回归取残差 (无截距项)。

    对 y = Xβ + ε 求解 β, 返回残差 ε。
    处理完全共线情形: 若残差方差 < min_variance, 返回零向量。

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        因变量。
    X : np.ndarray, shape (n, k)
        自变量矩阵 (k >= 1)。
    min_variance : float
        残差方差下限。

    Returns
    -------
    np.ndarray, shape (n,)
        残差向量。
    """
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]  # (k,) or (k, 1)
        predicted = X @ beta
        residual = y - predicted
    except np.linalg.LinAlgError:
        # 矩阵奇异, 返回零向量
        return np.zeros_like(y)

    if np.var(residual) < min_variance:
        return np.zeros_like(y)

    return residual


def gram_schmidt_orthogonalize(
    df: pd.DataFrame,
    factor_cols: list[str],
    date_col: str = "date",
    min_variance: float = 1e-10,
) -> tuple[pd.DataFrame, list[str]]:
    """
    对因子矩阵做 Gram-Schmidt 正交化 (基于回归残差法)。

    对每个日期截面, 按照 factor_cols 的排列顺序逐因子正交化:
      - 第 1 个因子: 保留原值
      - 第 k 个因子: 对前 k-1 个正交化因子做 OLS 回归, 取残差

    Parameters
    ----------
    df : pd.DataFrame
        因子面板。
    factor_cols : list[str]
        按 |IC_IR| 降序排列的因子列名。顺序决定正交化优先级。
    date_col : str
        日期列名。
    min_variance : float
        残差方差下限。低于此值视为完全共线, 残差置零。

    Returns
    -------
    (df, orth_cols)
        df:       添加了 _orth 正交化列的 DataFrame
        orth_cols: 正交化因子列名列表 (与 factor_cols 一一对应)
    """
    df = df.copy()
    n = len(df)

    # 预分配正交化结果
    orth_arrays: dict[str, np.ndarray] = {col: np.zeros(n, dtype=np.float64)
                                           for col in factor_cols}

    # 逐截面处理
    for dt, grp in df.groupby(date_col):
        mask = df[date_col] == dt
        idx_arr = np.where(mask.values)[0]
        n_stocks = len(idx_arr)
        if n_stocks < 5:
            continue

        for i, col in enumerate(factor_cols):
            y = grp[col].values.astype(np.float64)

            if i == 0:
                orth_arrays[col][idx_arr] = y
                continue

            # 用前面 i 个因子的正交化版本作为自变量
            prev_orth = [orth_arrays[c][idx_arr].reshape(-1, 1)
                         for c in factor_cols[:i]]
            X = np.column_stack(prev_orth)

            orth_arrays[col][idx_arr] = _residualize(y, X, min_variance)

    # 写回 DataFrame
    orth_cols: list[str] = []
    for col in factor_cols:
        orth_col = f"{col}_orth"
        df[orth_col] = orth_arrays[col]
        orth_cols.append(orth_col)

    return df, orth_cols


def apply_gram_schmidt_composite(
    df: pd.DataFrame,
    factor_cols: list[str],
    return_col: str = "forward_return_1m",
    date_col: str = "date",
    rolling_window: int = 24,
    flip_sign: bool = True,
    min_factor_variance: float = 1e-10,
    min_stocks: int = 20,
    min_ic_ir: float = 0.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    完整管线: 滚动 IC_IR → 排序 → Gram-Schmidt → IC_IR 加权 → composite_factor

    参数
    ----
    df : pd.DataFrame
        预处理后的因子面板 (含 _neutral_z 列和 forward_return)。
    factor_cols : list[str]
        因子列名 (默认为标准化后的实际列名)。
    return_col : str
        收益列名。
    date_col : str
        日期列名。
    rolling_window : int
        IC_IR 滚动窗口 (期数)。
    flip_sign : bool
        是否翻转负 IC_IR 因子。
    min_factor_variance : float
        正交化残差最小方差阈值。
    min_stocks : int
        计算 IC 时最小股票数。
    min_ic_ir : float
        |IC_IR| 最低阈值, 低于此值的因子不参与合成。
    verbose : bool
        是否打印权重分布。

    Returns
    -------
    pd.DataFrame: 添加了 composite_factor 列的 DataFrame。
    """
    df = df.copy()

    # ── Step 1: 计算滚动 IC_IR ──
    rolling_ic_irs = compute_rolling_ic_ir(
        df, factor_cols, return_col, date_col,
        rolling_window=rolling_window, min_stocks=min_stocks,
    )

    dates = sorted(df[date_col].unique())
    composite = pd.Series(0.0, index=df.index)

    # 日志记录 (最后一段窗口)
    last_weights: list[tuple[str, float, float]] = []
    n_colinear_skipped = 0

    # ── Step 2-5: 逐日期处理 ──
    for dt in dates:
        ic_irs = rolling_ic_irs.get(dt, {})
        if not ic_irs:
            continue

        # Step 2: 按 |IC_IR| 降序, 同时过滤低信噪比因子
        sorted_cols = sorted(
            [c for c in ic_irs if abs(ic_irs.get(c, 0.0)) > min_ic_ir],
            key=lambda c: abs(ic_irs[c]),
            reverse=True,
        )
        if not sorted_cols:
            continue

        # ── Step 3: Gram-Schmidt 正交化 (当前日期截面) ──
        dt_mask = df[date_col] == dt
        idx_arr = np.where(dt_mask.values)[0]
        if len(idx_arr) < 5:
            continue

        orth_values: dict[str, np.ndarray] = {}
        valid_cols: list[str] = []

        for i, col in enumerate(sorted_cols):
            y = df.loc[dt_mask, col].values.astype(np.float64)

            if i == 0:
                resid = y.copy()
            else:
                X = np.column_stack([orth_values[c] for c in valid_cols])
                resid = _residualize(y, X, min_factor_variance)

            orth_values[col] = resid
            valid_cols.append(col)

        # ── Step 4: IC_IR 权重 ──
        total_abs = sum(abs(ic_irs[c]) for c in valid_cols)
        if total_abs < 1e-10:
            continue

        # ── Step 5: 加权合成 ──
        dt_composite = np.zeros(len(idx_arr))
        for col in valid_cols:
            ic_ir = ic_irs[col]
            abs_ic_ir = abs(ic_ir)
            weight = abs_ic_ir / total_abs

            # 完全共线因子 → 有效权重为 0
            if np.var(orth_values[col]) < min_factor_variance:
                n_colinear_skipped += 1
                continue

            sign = -1.0 if (flip_sign and ic_ir < 0) else 1.0
            dt_composite += sign * weight * orth_values[col]

        composite.loc[dt_mask] = dt_composite

        # 保留最后一个日期的权重用于日志
        if verbose and dt == dates[-1]:
            for col in valid_cols:
                ic_ir = ic_irs[col]
                abs_ic_ir = abs(ic_ir)
                weight = abs_ic_ir / total_abs
                last_weights.append((col, weight, ic_ir))

    df["composite_factor"] = composite

    # ── 日志 ──
    if verbose:
        method_desc = f"Gram-Schmidt 正交化 (滚动 {rolling_window} 期 IC_IR)"
        print(f"  合成方法: {method_desc} | 翻转符号: {flip_sign}")
        if n_colinear_skipped > 0:
            print(f"  正交化: {n_colinear_skipped} 个因子-截面因完全共线被跳过")
        if last_weights:
            last_weights.sort(key=lambda x: x[1], reverse=True)
            print(f"  因子权重 (最近截面):")
            for c, w, ic_ir in last_weights:
                flag = "(-)" if ic_ir < 0 else "(+)"
                print(f"    {c:30s} weight={w:.4f}  {flag}  IC_IR={ic_ir:+.4f}")

    return df
