"""
VaR (Value at Risk) 回测。

VaR 是风险管理的核心概念。面试官可能问:
- "你怎么验证你的 VaR 模型是对的?"
- "历史模拟法和参数法各有什么优缺点?"

VaR_95% = "在 95% 的置信水平下, 每日最大预期亏损"
意思是: 你有 5% 的概率亏得比 VaR 还多。

回测逻辑:
如果 VaR 模型正确, 实际亏损超过 VaR 的天数应该 ≈ (1-置信水平) × 总天数。
比如 95% VaR, 100天里应该有约5天超过。
"""

import pandas as pd
import numpy as np
from scipy import stats


def var_parametric(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
) -> pd.Series:
    """
    参数法 VaR: VaR_α = -(μ + σ · z_α)

    假设收益率服从正态分布, 用滚动窗口估计 μ 和 σ。
    优点: 计算简单, 可解释性强。
    缺点: 收益不是正态分布——有厚尾, 参数法低估极端风险。
    """
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std()
    z = stats.norm.ppf(1 - confidence)
    var = -(mu + z * sigma)
    return var.dropna()


def var_historical(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
) -> pd.Series:
    """
    历史模拟法 VaR: 取滚动窗口内收益的经验分位数。

    优点: 不需要正态假设, 自然捕获厚尾。
    缺点: 极度依赖历史数据——历史上没发生过的极端事件, 模型不会预警。
          "过去252天没发生金融危机, 所以VaR设为0" → 2008年9月你就完了。
    """
    var = returns.rolling(window).quantile(1 - confidence)
    return -var.dropna()


def backtest_var(
    returns: pd.Series,
    var_series: pd.Series,
    confidence: float = 0.95,
) -> dict:
    """
    VaR 回测: 比较实际亏损超过 VaR 的次数与预期。

    核心指标:
    - exceedance_rate: 实际超出比例
    - expected_rate: 理论超出比例 (1 - confidence)
    - kupiec_pvalue: Kupiec 检验的 p 值
        p > 0.05 → 模型通过检验 (VaR 预测与实际亏损在统计上一致)
        p < 0.05 → VaR 模型有问题 (高估或低估了风险)
    """
    common = returns.dropna().index.intersection(var_series.dropna().index)
    actual_losses = -returns.loc[common]
    var_vals = var_series.loc[common]

    exceed = int((actual_losses > var_vals).sum())
    total = len(common)
    exceed_rate = exceed / total if total > 0 else 0
    expected_rate = 1 - confidence

    # Kupiec 似然比检验
    if total > 0 and 0 < exceed_rate < 1:
        p0 = expected_rate
        p1 = exceed_rate
        try:
            lr = -2 * (
                (total - exceed) * np.log((1 - p0) / (1 - p1))
                + exceed * np.log(p0 / p1)
            )
            p_value = 1 - stats.chi2.cdf(abs(lr), df=1)
        except Exception:
            p_value = None
    else:
        p_value = None

    return {
        "exceedances": exceed,
        "total": total,
        "exceedance_rate": round(float(exceed_rate), 4),
        "expected_rate": round(float(expected_rate), 4),
        "status": "PASS" if p_value is not None and p_value > 0.05 else "FAIL",
        "kupiec_pvalue": round(float(p_value), 4) if p_value is not None else None,
    }
