"""
波动率建模 — GARCH / EGARCH。

这是你的差异化武器。大多数申请者止步于历史波动率(过去60天std)。
你能讨论 GARCH 族模型的适用场景和局限, 说明你真正理解金融时间序列。

GARCH(1,1) 的核心方程:
    r_t = μ + ε_t           (均值方程)
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}   (波动率方程)

直观理解:
- ω: 长期平均波动率水平
- α: 新信息对波动率的影响 (α 大 → 波动率对冲击反应剧烈)
- β: 波动率的持续性 (β 大 → 高波动会持续很久)
- α + β 越接近 1 → 波动率越持久 (金融数据通常 0.95~0.99)

EGARCH 额外捕捉杠杆效应: 坏消息 (负收益) 对波动率的影响 > 同等幅度的好消息。
这是你的随机分析背景可以自然展开的地方——GARCH 本质上就是离散时间的随机波动率 SDE。
"""

import pandas as pd
import numpy as np
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> dict:
    """
    拟合 GARCH(p,q) 模型。

    返回的 conditional_vol 是模型估计的"今天的波动率是多少",
    而不是简单的过去60天std。这在市场剧烈波动时会更快反应。
    """
    scaled = returns.fillna(0) * 100  # 缩放到百分比帮助数值优化

    try:
        model = arch_model(scaled, vol="GARCH", p=p, q=q, mean="constant", dist=dist)
        fitted = model.fit(disp="off", show_warning=False)
        cond_vol = fitted.conditional_volatility / 100  # 缩回小数

        return {
            "conditional_vol": cond_vol,
            "params": fitted.params.to_dict(),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "model": fitted,
        }
    except Exception as e:
        raise RuntimeError(f"GARCH({p},{q}) 拟合失败: {e}")


def fit_egarch(returns: pd.Series, p: int = 1, q: int = 1) -> dict:
    """
    拟合 EGARCH(1,1) — 捕捉杠杆效应。

    杠杆效应: 股价跌20%带来的波动率上升, 比涨20%带来的波动率上升大得多。
    原因: 跌20% → 杠杆率上升 → 公司更脆弱 → 更易受冲击 → 波动率更大。
    EGARCH 通过指数形式的条件方差方程自然捕捉这种不对称。
    """
    scaled = returns.fillna(0) * 100
    try:
        model = arch_model(scaled, vol="EGARCH", p=p, q=q, mean="constant")
        fitted = model.fit(disp="off", show_warning=False)
        cond_vol = fitted.conditional_volatility / 100

        return {
            "conditional_vol": cond_vol,
            "params": fitted.params.to_dict(),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "model": fitted,
        }
    except Exception as e:
        raise RuntimeError(f"EGARCH({p},{q}) 拟合失败: {e}")


def compare_vol_methods(returns: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    对比三种波动率估计: 历史滚动 / GARCH / EGARCH。

    面试时你可以说: "历史波动率用过去20天等权平均, 对新信息反应迟钝;
    GARCH 给近期冲击更高权重; EGARCH 进一步捕捉了涨跌不对称性。"
    """
    hist_vol = returns.rolling(window).std() * np.sqrt(252)
    garch_result = fit_garch(returns)
    egarch_result = fit_egarch(returns)

    result = pd.DataFrame(
        {
            "Historical_Vol": hist_vol,
            "GARCH_Vol": garch_result["conditional_vol"] * np.sqrt(252),
            "EGARCH_Vol": egarch_result["conditional_vol"] * np.sqrt(252),
        },
        index=returns.index,
    )
    return result.dropna()
