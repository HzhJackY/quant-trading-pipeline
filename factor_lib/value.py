"""
估值类因子。

核心思想: 便宜的股票 (相对于账面价值或盈利) 未来收益更高。
这是价值投资在量化中的体现 (Fama-French HML 因子)。

所有估值因子的逻辑都一样:
    "你每花1块钱买这只股票, 能买到多少账面价值/盈利/现金流?"
    比值越高 → 越便宜 → (理论上) 预期收益越高
"""

import pandas as pd


def compute_bp(
    financial: pd.DataFrame,
    daily: pd.DataFrame,
    market_cap_col: str = "总市值",
) -> pd.DataFrame:
    """
    BP = 净资产 / 总市值 (Book-to-Price)

    含义: 每花1块钱买这只股票, 买到了多少账面净资产。
    BP 越高 → 股票越被低估 → 未来收益倾向于更高。

    BP 的倒数就是 PB (市净率), 是市场上最常见的估值指标。
    """
    merged = daily[["date", "symbol", market_cap_col]].merge(
        financial[["symbol", "净资产"]], on="symbol", how="left"
    )
    merged["BP"] = merged["净资产"] / merged[market_cap_col].replace(0, float("nan"))
    return merged[["date", "symbol", "BP"]].dropna(subset=["BP"])


def compute_ep(
    financial: pd.DataFrame,
    daily: pd.DataFrame,
    market_cap_col: str = "总市值",
) -> pd.DataFrame:
    """
    EP = 净利润(TTM) / 总市值 (Earnings-to-Price)

    含义: 每花1块钱, 买到了多少年化盈利。
    EP 的倒数就是 PE (市盈率), 市场最关注的估值指标。
    TTM = Trailing Twelve Months, 滚动12个月净利润。
    """
    fin = financial.sort_values(["symbol", "report_date"]).copy()
    fin["net_profit_ttm"] = fin.groupby("symbol")["净利润"].transform(
        lambda x: x.rolling(4, min_periods=4).sum()
    )

    merged = daily[["date", "symbol", market_cap_col]].merge(
        fin[["symbol", "net_profit_ttm"]], on="symbol", how="left"
    )
    merged["EP"] = merged["net_profit_ttm"] / merged[market_cap_col].replace(0, float("nan"))
    return merged[["date", "symbol", "EP"]].dropna(subset=["EP"])
