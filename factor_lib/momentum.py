"""
动量类因子。

核心思想: 过去涨得好的股票, 未来一段时间内倾向于继续涨。
这是学术上最稳健的异象之一 (Jegadeesh & Titman, 1993)。

但要注意:
- 短期(1个月)有反转效应: 过去1个月涨的股票下个月倾向跌
- 中期(3-12个月)有动量效应: 过去半年涨的股票倾向于继续涨
- 所以我们会算 Mom_1M (看反转) 和 Mom_6M/12M-1M (看动量)
"""

import pandas as pd
import numpy as np


def compute_momentum_1m(daily_data: pd.DataFrame) -> pd.DataFrame:
    """
    过去 1 个月收益 (跳过最近 5 个交易日)。
    这个因子通常显示反转效应: 上个月涨太多 → 下个月回调。
    """
    return _momentum(daily_data, lookback=21, skip=5, name="Mom_1M")


def compute_momentum_3m(daily_data: pd.DataFrame) -> pd.DataFrame:
    """过去 3 个月收益 (跳过最近 5 个交易日)。"""
    return _momentum(daily_data, lookback=63, skip=5, name="Mom_3M")


def compute_momentum_6m(daily_data: pd.DataFrame) -> pd.DataFrame:
    """过去 6 个月收益 (跳过最近 5 个交易日)。"""
    return _momentum(daily_data, lookback=126, skip=5, name="Mom_6M")


def compute_momentum_12m_1m(daily_data: pd.DataFrame) -> pd.DataFrame:
    """
    过去12个月收益, 但跳过最近1个月 (t-12 到 t-1)。
    这是学术界最经典的动量度量——排除最近一个月的反转噪音后,
    纯动量效应最强。
    """
    return _momentum(daily_data, lookback=231, skip=21, name="Mom_12M_1M")


def _momentum(
    daily_data: pd.DataFrame,
    lookback: int,
    skip: int,
    name: str,
) -> pd.DataFrame:
    """
    通用动量计算。

    逻辑: 对每只股票, 在每一天, 计算
        [date - skip - lookback,  date - skip]
    这个窗口内的累计收益。

    为什么跳过最近 skip 天?
    - 短期存在 bid-ask bounce 和流动性噪音
    - 跳过最近 5 天可以得到更干净的信号
    """
    df = daily_data.sort_values(["symbol", "date"]).copy()
    df["close"] = df.groupby("symbol")["close"].transform(
        lambda x: x.ffill()
    )

    rows = []
    for sym, group in df.groupby("symbol"):
        group = group.sort_values("date").reset_index(drop=True)
        closes = group["close"].values
        dates = group["date"].values

        for i in range(lookback + skip, len(closes)):
            # close[i - skip] / close[i - skip - lookback] - 1
            p_end = closes[i - skip]
            p_start = closes[i - skip - lookback]
            if p_start > 0:
                ret = p_end / p_start - 1
                rows.append({"date": dates[i], "symbol": sym, name: ret})

    return pd.DataFrame(rows)
