"""
技术面因子。

从日线数据计算量价技术指标。这些因子与基本面因子 (价值/质量/成长)
通常低相关, 因为它们捕捉的是市场微观结构和投资者行为。

A 股特征:
- 换手率/成交量变化捕捉散户关注度, 在中小盘中更有效
- 均线偏离捕捉均值回复, A 股短期反转效应强
"""

import pandas as pd
import numpy as np


def compute_volume_20d_change(daily_data: pd.DataFrame) -> pd.DataFrame:
    """
    20 日成交量变化率。

    逻辑: 当日成交量 / 过去 20 日均量 - 1。
    放量 (>0): 市场关注度上升, 短期可能伴随动量
    缩量 (<0): 市场关注度下降

    参数
    ----
    daily_data : DataFrame
        必须包含: date, symbol, volume (成交量)

    返回
    ----
    DataFrame: date, symbol, VolChg_20D
    """
    df = daily_data.sort_values(["symbol", "date"]).copy()

    # 确保 volume 列存在
    vol_col = "volume" if "volume" in df.columns else "成交量"
    if vol_col not in df.columns:
        raise KeyError(f"daily_data 需要 'volume' 或 '成交量' 列, 现有: {df.columns.tolist()}")

    rows = []
    for sym, group in df.groupby("symbol"):
        group = group.sort_values("date").reset_index(drop=True)
        vols = group[vol_col].values.astype(float)
        dates = group["date"].values

        if len(vols) < 21:
            continue

        # 滚动 20 日均量
        vol_ma20 = pd.Series(vols).rolling(20).mean().values

        for i in range(20, len(vols)):
            if vol_ma20[i] > 0:
                ratio = vols[i] / vol_ma20[i] - 1
                rows.append({"date": dates[i], "symbol": sym, "VolChg_20D": ratio})

    return pd.DataFrame(rows)


def compute_price_ma_deviation(daily_data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    价格偏离 N 日均线。

    逻辑: (收盘价 - MA_N) / MA_N。
    正值 (>0): 价格在均线上方, 短期超买, 倾向回调
    负值 (<0): 价格在均线下方, 短期超卖, 倾向反弹

    这是 A 股最常见的均值回复技术指标。

    参数
    ----
    daily_data : DataFrame
        必须包含: date, symbol, close (或 收盘)
    window : int
        均线窗口, 默认 20 天

    返回
    ----
    DataFrame: date, symbol, PriceDev_{window}D
    """
    df = daily_data.sort_values(["symbol", "date"]).copy()

    close_col = "close" if "close" in df.columns else "收盘"
    if close_col not in df.columns:
        raise KeyError(f"daily_data 需要 'close' 或 '收盘' 列")

    rows = []
    for sym, group in df.groupby("symbol"):
        group = group.sort_values("date").reset_index(drop=True)
        closes = group[close_col].values.astype(float)
        dates = group["date"].values

        if len(closes) < window:
            continue

        ma = pd.Series(closes).rolling(window).mean().values

        for i in range(window, len(closes)):
            if ma[i] > 0:
                dev = (closes[i] - ma[i]) / ma[i]
                rows.append({
                    "date": dates[i],
                    "symbol": sym,
                    f"PriceDev_{window}D": dev,
                })

    return pd.DataFrame(rows)


def compute_illiquidity(daily_data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Amihud 非流动性指标 (月度)。

    逻辑: 过去 N 天的 |日收益率| / 日成交额 的均值。
    高非流动性 → 交易成本高, 流动性差 → 应该有更高的预期收益 (流动性溢价)。

    这是 Fama-French 五因子之外最常见的学术因子之一 (Amihud, 2002)。

    参数
    ----
    daily_data : DataFrame
        必须包含: date, symbol, close, amount (成交额)
    window : int

    返回
    ----
    DataFrame: date, symbol, Illiquidity
    """
    df = daily_data.sort_values(["symbol", "date"]).copy()

    ret_col = "涨跌幅" if "涨跌幅" in df.columns else None
    close_col = "close" if "close" in df.columns else "收盘"
    amount_col = "amount" if "amount" in df.columns else "成交额"

    if amount_col not in df.columns:
        raise KeyError(f"daily_data 需要 'amount' 或 '成交额' 列")

    rows = []
    for sym, group in df.groupby("symbol"):
        group = group.sort_values("date").reset_index(drop=True)

        # 计算日收益率
        if ret_col and ret_col in group.columns:
            rets = group[ret_col].values.astype(float) / 100  # 涨跌幅是百分比
        else:
            closes = group[close_col].values.astype(float)
            rets = np.diff(closes, prepend=closes[0]) / closes
            rets[0] = 0.0

        amounts = group[amount_col].values.astype(float)
        dates = group["date"].values

        if len(rets) < window:
            continue

        daily_illiq = np.abs(rets) / np.maximum(amounts, 1)  # avoid div by 0

        # 滚动均值 (月度非流动性)
        illiq_series = pd.Series(daily_illiq).rolling(window).mean().values

        for i in range(window, len(illiq_series)):
            if not np.isnan(illiq_series[i]):
                rows.append({
                    "date": dates[i],
                    "symbol": sym,
                    "Illiquidity": illiq_series[i],
                })

    return pd.DataFrame(rows)
