"""
因子 IC (Information Coefficient) 分析。

IC 是量化研究员最核心的指标——衡量因子对下期收益的预测能力。
面试时你要能从三个角度讨论 IC:
1. IC 均值 → 因子有没有预测力
2. IC_IR (IC均值/IC标准差) → 因子预测力的稳定性
3. IC 衰减 → 因子能预测多远

一个因子如果 IC 很高但方差巨大, 就像赌徒偶尔赢一把——不可靠。
IC_IR 把稳定性考虑进来, 是更全面的评价指标。
"""

import pandas as pd
import numpy as np
from scipy import stats


def compute_rank_ic(
    df: pd.DataFrame,
    factor_col: str = "factor",
    return_col: str = "forward_return_1m",
    date_col: str = "date",
) -> pd.Series:
    """
    计算因子在每个截面的 Rank IC。

    Rank IC = Spearman's ρ (因子截面排名, 下期收益截面排名)

    为什么用 Rank IC 而不是 Pearson IC:
    - Pearson 假设线性关系, 实际因子和收益的关系往往是非线性的
    - Spearman 只看排名不看大小, 对异常值不敏感
    - 一颗老鼠屎不会毁了一锅粥 (一个极端值不会影响 Spearman)

    IC ∈ [-1, 1]:
        +0.03 以上 → 有意义的正向预测力
        -0.03 以下 → 反向关系 (可能需要反转因子方向)
        接近 0    → 没有预测力
    """
    ic_vals = {}
    for date, group in df.groupby(date_col):
        valid = group[[factor_col, return_col]].dropna()
        if len(valid) < 30:
            continue
        ic, _ = stats.spearmanr(valid[factor_col], valid[return_col])
        ic_vals[date] = ic
    return pd.Series(ic_vals, name="Rank_IC").sort_index()


def compute_ic_summary(ic_series: pd.Series) -> dict:
    """
    IC 汇总统计。

    核心指标:
    - IC_Mean: 预测力的方向与幅度
    - IC_IR: 预测力的稳定性 (业界标准: > 0.5 可用, > 0.7 优秀)
    - IC_Win_Rate: 预测方向正确的比例 (> 55% 说明有persistent信号)
    - t-stat: 统计显著性 (> 2 说明 IC 显著不为 0)
    """
    ic = ic_series.dropna()
    n = len(ic)
    if n == 0:
        return {}

    mean_ic = ic.mean()
    std_ic = ic.std(ddof=1)
    ir = mean_ic / std_ic if std_ic > 0 else 0.0
    win_rate = (ic > 0).sum() / n
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0.0

    return {
        "IC_Mean": round(float(mean_ic), 4),
        "IC_Std": round(float(std_ic), 4),
        "IC_IR": round(float(ir), 4),
        "IC_Win_Rate": round(float(win_rate), 4),
        "IC_t_stat": round(float(t_stat), 2),
        "Periods": int(n),
    }


def compute_ic_decay(
    df: pd.DataFrame,
    factor_col: str = "factor",
    return_cols: list[str] | None = None,
    date_col: str = "date",
) -> dict[str, float]:
    """
    计算 IC 衰减: 因子对未来不同期限收益的预测力变化。

    比如:
    - forward_return_1m  → IC = 0.05  (预测力强)
    - forward_return_3m  → IC = 0.03  (衰减了)
    - forward_return_6m  → IC = 0.01  (快没了)

    如果 IC 衰减很快 → 你只能做短线 → 换手率高 → 交易成本吃掉 alpha
    如果 IC 衰减很慢 → 你可以低频调仓 → 你的 alpha 更值钱
    """
    if return_cols is None:
        return_cols = [
            "forward_return_1m",
        ]
    decay = {}
    for ret_col in return_cols:
        ic = compute_rank_ic(df, factor_col=factor_col, return_col=ret_col, date_col=date_col)
        summary = compute_ic_summary(ic)
        decay[ret_col] = summary.get("IC_Mean", 0.0)
    return decay
