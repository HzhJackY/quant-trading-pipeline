"""
数据预处理流水线。
三步标准化流程: 去极值 → 中性化 → 标准化
每一层都有独立的金融含义, 不可随意跳过。
"""

import numpy as np
import pandas as pd


def winsorize_mad(series: pd.Series, n_mad: float = 3.0) -> pd.Series:
    """
    MAD (Median Absolute Deviation) 缩尾去极值。

    为什么用 MAD 而不是直接切百分位:
    - MAD 对异常值本身不敏感 (中位数不会因为一个极端值而改变)
    - 均值±3σ 的阈值会被异常值拉偏, MAD 不会
    - 这是量化学界和业界的标准做法 (Barra, MSCI 等)

    参数
    ----
    series : pd.Series
        因子值序列, 可以是单只股票的时间序列或截面上所有股票
    n_mad : float
        MAD 倍数阈值, 默认 3.0 (约等于均值±3σ)

    返回
    ----
    pd.Series
        缩尾后的序列 (极端值被 clip 到边界, 中间值不变)
    """
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return series  # 所有值相同, 无需缩尾
    # 换算系数 1.4826: 正态分布下 MAD × 1.4826 ≈ σ
    upper = median + n_mad * mad * 1.4826
    lower = median - n_mad * mad * 1.4826
    return series.clip(lower=lower, upper=upper)


def standardize_cross_section(
    df: pd.DataFrame,
    factor_col: str = "factor",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    截面标准化: 在每个时间截面上将因子值转为 Z-score。

    为什么必须做:
    - 不同因子的量纲不同 (BP 是 0.01~2, 动量可能是 -0.3~0.5)
    - 你无法比较或合成量纲不同的因子
    - 截面标准化后所有因子都是"今天这只股票在所有股票中排第几"的含义

    Z = (X - mean(X)) / std(X)    在每个时间截面上独立计算

    返回的 DataFrame 新增一列 {factor_col}_z。
    """
    df = df.copy()
    z_scores = df.groupby(date_col)[factor_col].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
        if x.std(ddof=0) > 0
        else 0.0
    )
    df[f"{factor_col}_z"] = z_scores
    return df


def neutralize_industry_market_cap(
    df: pd.DataFrame,
    factor_col: str = "factor",
    industry_col: str = "industry",
    mcap_col: str = "log_market_cap",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    行业 + 市值中性化。
    对每个截面, 将因子对行业虚拟变量和对数市值做 OLS 回归, 取残差。

    为什么要做中性化:
    - 假设你发现了一个"高 ROE 股票收益好"的信号
    - 但高 ROE 的股票可能恰好都是食品饮料行业的, 而食品饮料行业同期表现好
    - 你的因子到底是在选 ROE 还是在选行业?
    - 中性化就是: "在控制了行业和市值之后, 这个因子还有预测力吗?"

    具体做法:
    factor = β_0 + Σβ_i·I(行业=行业_i) + β_m·log(市值) + ε
    取残差 ε 作为新的因子值

    残差的经济含义: "这只股票的因子值, 在扣除了它的行业和市值应有的水平后,
    还剩下多少超额部分?" 这才是因子的纯 alpha。

    返回的 DataFrame 新增一列 {factor_col}_neutral。
    """
    from statsmodels.api import OLS, add_constant

    df = df.copy()
    residual_list = []

    for date, group in df.groupby(date_col):
        if len(group) < 10 or group[factor_col].nunique() < 2:
            # 样本太少无法回归, 保留原值
            group["_resid"] = group[factor_col].values
            residual_list.append(group)
            continue

        # 行业 → 哑变量 (行业A=1,0,0; 行业B=0,1,0; ...)
        industry_dummies = pd.get_dummies(
            group[industry_col], drop_first=True
        ).astype(float)

        # 回归自变量: const + 行业哑变量 + 对数市值
        X = pd.concat([industry_dummies, group[mcap_col].astype(float)], axis=1)
        X = add_constant(X, has_constant="add")
        y = group[factor_col].astype(float)

        try:
            model = OLS(y, X, missing="drop").fit()
            group["_resid"] = model.resid
        except Exception:
            group["_resid"] = y.values

        residual_list.append(group)

    result = pd.concat(residual_list, sort=False)
    result[f"{factor_col}_neutral"] = result["_resid"]
    result = result.drop(columns=["_resid"])
    return result
