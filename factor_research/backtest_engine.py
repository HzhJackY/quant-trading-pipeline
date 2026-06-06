"""
组合回测引擎。

多因子合成 → 选股 → 净值计算 → 绩效评估。

这是因子研究的最后一环——前面都是"这个因子好不好",
这里问的是"用这些因子做一个组合, 你能赚多少钱?"
"""

import pandas as pd
import numpy as np


def combine_factors(
    factor_df: pd.DataFrame,
    factor_cols: list[str] | None = None,
    method: str = "equal_weight",
) -> pd.DataFrame:
    """
    多因子合成: 将多个标准化后的因子合并成一个复合分数。

    方法:
    - equal_weight: 每个因子贡献一样 (稳健,不overfit)
    - ic_weighted: 用历史 IC_IR 加权 (但需要足够长的历史, 否则 overfit)

    等权是最简单也最常用的——当你不知道哪个因子未来表现更好,
    给每个因子一样的权重就是最诚实的选择。
    """
    df = factor_df.copy()
    if factor_cols is None:
        # 自动检测因子列 (带 _z 或 _neutral 后缀的)
        factor_cols = [
            c
            for c in df.columns
            if c.endswith("_z") or c.endswith("_neutral")
        ]
    if not factor_cols:
        # fallback 到所有数值列
        factor_cols = [
            c
            for c in df.columns
            if c not in ("date", "symbol", "group")
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    available = [c for c in factor_cols if c in df.columns]
    df["composite_factor"] = df[available].mean(axis=1, skipna=True)
    return df


def compute_nav(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """
    从收益序列计算累计净值。

    NAV_t = NAV_0 * Π(1 + r_i)
    """
    return initial_value * (1 + returns).cumprod()


def compute_performance(
    returns: pd.Series,
    freq: str = "M",
    rf: float = 0.02,
) -> dict:
    """
    绩效评估指标。

    这些是量化面试必问的指标——你必须能解释:
    1. 年化收益率: 平均每月赚 X% × 12
    2. 年化波动率: 每月收益的标准差 × √12
    3. Sharpe Ratio: 每承担一单位风险, 获得多少超额收益
    4. 最大回撤: 从最高点到最低点, 最多亏了多少
    5. Calmar Ratio: 年化收益 / |最大回撤|, 回撤调整后收益
    """
    periods_per_year = {"M": 12, "D": 252, "W": 52}.get(freq, 12)

    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else 0.0

    # 最大回撤计算
    nav = compute_nav(returns)
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_dd = float(drawdown.min())

    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0.0
    win_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0

    return {
        "Annualized_Return": round(float(ann_return), 4),
        "Volatility": round(float(ann_vol), 4),
        "Sharpe_Ratio": round(float(sharpe), 4),
        "Max_Drawdown": round(max_dd, 4),
        "Calmar_Ratio": round(float(calmar), 4),
        "Win_Rate": round(win_rate, 4),
        "Periods": int(len(returns)),
    }
