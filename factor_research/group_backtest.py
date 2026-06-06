"""
分层回测引擎。

IC 分析告诉你因子有没有预测力, 分层回测告诉你这个预测力能不能变成真钱。

分层回测的逻辑:
1. 每月末把所有股票按因子值分为 N 组 (通常是5组)
2. 等权持有每组, 拿到下个月
3. 看各组的累计收益
4. 理想情况: Q1 → Q2 → Q3 → Q4 → Q5 单调递增

单调递增意味着因子值越高 = 收益越好, 关系清晰。
如果 Q3 > Q5 但 Q2 < Q1, 说明因子和收益不是线性关系,
可能需要分段处理或非线性变换。
"""

import pandas as pd
import numpy as np


def assign_quantile_groups(
    df: pd.DataFrame,
    factor_col: str = "factor",
    n_groups: int = 5,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    在每个截面上, 按因子值将股票分为 n_groups 组。

    Q1 = 因子值最低的那 20% 股票
    Q5 = 因子值最高的那 20% 股票

    返回的 DataFrame 新增 "group" 列 (1, 2, 3, 4, 5).
    """
    df = df.copy()
    df["group"] = np.nan

    for date, idx in df.groupby(date_col).groups.items():
        mask = df.loc[idx].dropna(subset=[factor_col]).index
        if len(mask) < n_groups:
            df.loc[mask, "group"] = 1
            continue
        try:
            df.loc[mask, "group"] = pd.qcut(
                df.loc[mask, factor_col],
                q=n_groups,
                labels=range(1, n_groups + 1),
                duplicates="drop",
            )
        except ValueError:
            df.loc[mask, "group"] = 1

    df["group"] = df["group"].astype("Int64")  # nullable int, 容忍 NaN
    return df


def compute_group_returns(
    df: pd.DataFrame,
    return_col: str = "forward_return_1m",
    group_col: str = "group",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    计算每组每期的等权平均收益。

    返回 DataFrame: date, group, return, n_stocks
    """
    rows = []
    for (date, group), gdf in df.groupby([date_col, group_col]):
        valid = gdf.dropna(subset=[return_col])
        if len(valid) == 0:
            continue
        rows.append(
            {
                "date": date,
                "group": int(group),
                "return": float(valid[return_col].mean()),
                "n_stocks": len(valid),
            }
        )
    return pd.DataFrame(rows)


def compute_long_short(
    group_returns: pd.DataFrame,
    long_group: int = 5,
    short_group: int = 1,
) -> pd.DataFrame:
    """
    多空组合收益 = Q5 收益 - Q1 收益。

    含义: 做多因子值最高的股票, 做空因子值最低的股票,
    你拿到的收益是多少? 这是因子 alpha 最纯粹的度量。

    多空组合的好处:
    - 对冲了市场涨跌 (市场涨 Q5涨 Q1也涨, 差价不受市场涨跌影响)
    - 只看因子本身的选股能力
    """
    pivot = group_returns.pivot_table(index="date", columns="group", values="return")
    if long_group not in pivot.columns or short_group not in pivot.columns:
        return pd.DataFrame(columns=["date", "long_short_return"])
    ls = pivot[long_group] - pivot[short_group]
    return ls.reset_index(name="long_short_return")
