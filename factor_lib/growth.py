"""
成长类因子。

核心思想: 收入和利润增长快的公司, 股价倾向于表现更好。
但要注意——市场已经为高增长付了溢价(体现在高PE上),
所以成长因子真正的有效性取决于"增长是否超预期"。

学术上: 单纯的成长因子 IC 并不高, 因为它与估值因子负相关
(高成长公司往往估值贵)。需要与估值搭配使用。
"""

import pandas as pd


def compute_revenue_growth(financial: pd.DataFrame) -> pd.DataFrame:
    """
    营业收入同比增长率 = (本季度营收 - 去年同期营收) / |去年同期营收|

    同比(YoY)比环比(QoQ)更干净——消除了季节性。
    比如零售行业 Q4(寒假季)天然比 Q3 营收高, 环比没意义。
    """
    fin = financial.sort_values(["symbol", "report_date"]).copy()
    if "营业收入" not in fin.columns:
        return pd.DataFrame(columns=["symbol", "report_date", "Rev_Growth_YoY"])

    fin["rev_lag4"] = fin.groupby("symbol")["营业收入"].shift(4)  # 四个季度前
    fin["Rev_Growth_YoY"] = (
        fin["营业收入"] - fin["rev_lag4"]
    ) / fin["rev_lag4"].abs().replace(0, float("nan"))
    return fin[["symbol", "report_date", "Rev_Growth_YoY"]].dropna(subset=["Rev_Growth_YoY"])


def compute_earnings_growth(financial: pd.DataFrame) -> pd.DataFrame:
    """
    净利润同比增长率。
    更重要的成长指标——营收增长可以被价格战堆出来(负利润),
    但利润增长才是真正的价值创造。
    """
    fin = financial.sort_values(["symbol", "report_date"]).copy()
    if "净利润" not in fin.columns:
        return pd.DataFrame(columns=["symbol", "report_date", "Earnings_Growth"])

    fin["earn_lag4"] = fin.groupby("symbol")["净利润"].shift(4)
    fin["Earnings_Growth"] = (
        fin["净利润"] - fin["earn_lag4"]
    ) / fin["earn_lag4"].abs().replace(0, float("nan"))
    return fin[["symbol", "report_date", "Earnings_Growth"]].dropna(subset=["Earnings_Growth"])
