"""
质量类因子。

核心思想: 财务质量高的公司 (高盈利、低杠杆) 长期表现更好。
这些因子来自于对"好公司"的量化定义。

面试时要知道:
- ROE 是巴菲特最看重的指标, 但单纯买高 ROE 的股票不赚钱 (已经被定价了)
- 真正有效的是 ROE 的稳定性或 ROE 的变化趋势
"""

import pandas as pd


def compute_roe(financial: pd.DataFrame) -> pd.DataFrame:
    """
    ROE = 净利润 / 净资产 (净资产收益率)

    含义: 公司用股东的每1块钱, 一年能赚多少钱。
    是衡量管理层效率最核心的指标。

    注意: ROE 高不一定好——
    如果是通过加杠杆(借钱)推高的 ROE, 那是脆弱的。
    杜邦分析能把 ROE 拆成三部分来看来源。
    """
    fin = financial.copy()
    if "净利润" in fin.columns and "净资产" in fin.columns:
        fin["ROE"] = fin["净利润"] / fin["净资产"].replace(0, float("nan"))
    return fin[["symbol", "report_date", "ROE"]].dropna(subset=["ROE"])


def compute_gross_margin(financial: pd.DataFrame) -> pd.DataFrame:
    """
    毛利率 = (营业收入 - 营业成本) / 营业收入

    含义: 每卖1块钱产品, 扣掉直接成本后还剩多少。
    高毛利率 → 定价能力强 → 有护城河。
    茅台毛利率~90%, 制造业毛利率~15%, 差异巨大。
    """
    fin = financial.copy()
    if "营业收入" in fin.columns and "营业成本" in fin.columns:
        fin["Gross_Margin"] = (
            fin["营业收入"] - fin["营业成本"]
        ) / fin["营业收入"].replace(0, float("nan"))
    return fin[["symbol", "report_date", "Gross_Margin"]].dropna(subset=["Gross_Margin"])


def compute_debt_ratio(financial: pd.DataFrame) -> pd.DataFrame:
    """
    资产负债率 = 总负债 / 总资产

    含义: 公司的资产中有多少是借来的。
    高杠杆在市场下行时是致命风险 (2008年雷曼兄弟)。
    但适度的杠杆(如银行)是商业模式决定的, 不能一刀切。
    """
    fin = financial.copy()
    if "总负债" in fin.columns and "总资产" in fin.columns:
        fin["Debt_Ratio"] = fin["总负债"] / fin["总资产"].replace(0, float("nan"))
    return fin[["symbol", "report_date", "Debt_Ratio"]].dropna(subset=["Debt_Ratio"])
