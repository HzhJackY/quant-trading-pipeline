"""
波动率类因子。

核心思想: 低波动率的股票 (所谓的"无聊的股票") 长期收益反而更高。
这是金融学里最反直觉的发现之一——CAPM 说高风险高收益,
但实际上低波动率股票有显著的超额收益 (低波动率异象, Low-Vol Anomaly)。

可能的解释:
- 杠杆约束: 想追求高收益但被限制杠杆的投资者, 只能买高波动股票, 推高了它们价格
- 彩票偏好: 投资者喜欢"中奖"的感觉, 高波动股票有小概率暴涨的彩票属性
"""

import pandas as pd
import numpy as np


def compute_volatility_20d(daily: pd.DataFrame) -> pd.DataFrame:
    """过去 20 个交易日年化波动率。"""
    return _rolling_vol(daily, window=20, name="Vol_20D")


def compute_volatility_60d(daily: pd.DataFrame) -> pd.DataFrame:
    """过去 60 个交易日年化波动率。"""
    return _rolling_vol(daily, window=60, name="Vol_60D")


def _rolling_vol(daily: pd.DataFrame, window: int, name: str) -> pd.DataFrame:
    """滚动窗口年化波动率 = std(日收益) * sqrt(252)"""
    df = daily.sort_values(["symbol", "date"]).copy()
    df["daily_ret"] = df.groupby("symbol")["close"].pct_change()
    df[name] = (
        df.groupby("symbol")["daily_ret"]
        .transform(lambda x: x.rolling(window, min_periods=window // 2).std())
        * np.sqrt(252)
    )
    return df[["date", "symbol", name]].dropna(subset=[name])


def compute_beta(daily: pd.DataFrame) -> pd.DataFrame:
    """
    市场 Beta: 个股收益对市场收益的敏感度。

    Beta = Cov(个股, 市场) / Var(市场)

    Beta = 1: 市场涨1%, 股票也涨1%
    Beta > 1: 进攻型(市场涨你涨更多)
    Beta < 1: 防御型(市场跌你跌更少)
    """
    df = daily.sort_values(["symbol", "date"]).copy()
    df["daily_ret"] = df.groupby("symbol")["close"].pct_change()
    # 用所有股票的等权平均作为市场代理 (如果有指数数据可以替换)
    mkt = df.groupby("date")["daily_ret"].mean().rename("mkt_ret")
    df = df.merge(mkt, on="date", how="left")

    rows = []
    for sym, group in df.groupby("symbol"):
        group = group.sort_values("date").dropna(subset=["daily_ret", "mkt_ret"])
        if len(group) < 30:
            continue
        rolling_cov = (
            group["daily_ret"].rolling(60, min_periods=30).cov(group["mkt_ret"])
        )
        rolling_var = group["mkt_ret"].rolling(60, min_periods=30).var()
        group = group.copy()
        group["Beta"] = (rolling_cov / rolling_var.replace(0, np.nan)).values
        rows.append(group[["date", "symbol", "Beta"]].dropna(subset=["Beta"]))

    return pd.concat(rows) if rows else pd.DataFrame()
