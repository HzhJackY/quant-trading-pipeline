"""
组合回测引擎。

多因子合成 → 选股 → 净值计算 → 绩效评估。

这是因子研究的最后一环——前面都是"这个因子好不好",
这里问的是"用这些因子做一个组合, 你能赚多少钱?"
"""

import pandas as pd
import numpy as np


def combine_factors(
    factor_df: pd.DataFrame,
    factor_cols: list[str] | None = None,
    method: str = "equal_weight",
    return_col: str = "forward_return_1m",
    date_col: str = "date",
    max_correlation: float = 0.7,
    flip_sign: bool = True,
) -> pd.DataFrame:
    """
    多因子合成: 将多个标准化后的因子合并成一个复合分数。

    方法:
    - equal_weight: 每个因子贡献一样 (稳健, 不overfit)
    - ic_weighted: 用每个因子的 |IC_IR| 加权

    可选优化:
    - flip_sign=True: 自动翻转 IC 稳定为负的因子 (反转因子)
    - max_correlation: 去冗余阈值 — 当两个因子相关性超过此值时,
      只保留 |IC_IR| 更高的那个, 避免重复计算同一信号
    """
    from scipy.stats import spearmanr

    df = factor_df.copy()
    if factor_cols is None:
        factor_cols = [
            c for c in df.columns
            if c.endswith("_z") or c.endswith("_neutral")
        ]
    if not factor_cols:
        factor_cols = [
            c for c in df.columns
            if c not in ("date", "symbol", "group")
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    available = [c for c in factor_cols if c in df.columns]
    if not available:
        df["composite_factor"] = 0.0
        return df

    # ── Step 1: 计算每个因子在全样本上的 IC_IR ──
    has_returns = return_col in df.columns
    ic_irs = {}
    factor_signs = {}  # +1 或 -1
    factor_corrs = None

    if has_returns:
        for col in available:
            ic_list = []
            for dt, grp in df.groupby(date_col):
                sub = grp[[col, return_col]].dropna()
                if len(sub) >= 20:
                    try:
                        ic, _ = spearmanr(sub[col], sub[return_col])
                        if not np.isnan(ic):
                            ic_list.append(ic)
                    except Exception:
                        pass
            if ic_list:
                mean_ic = np.mean(ic_list)
                std_ic = np.std(ic_list, ddof=1)
                ic_irs[col] = mean_ic / std_ic if std_ic > 0 else 0.0
            else:
                ic_irs[col] = 0.0

        # ── 计算因子截面相关性矩阵 (用于去冗余) ──
        latest_date = df[date_col].max()
        sub_latest = df[df[date_col] == latest_date][available].dropna()
        if len(sub_latest) > 10:
            factor_corrs = sub_latest.corr()

    # 如果没有 forward returns, 所有因子等权
    if not ic_irs:
        df["composite_factor"] = df[available].mean(axis=1, skipna=True)
        return df

    # ── Step 2: 符号翻转 ──
    selected = {}  # col -> (sign, weight)
    for col in available:
        ic_ir = ic_irs.get(col, 0.0)
        if flip_sign and ic_ir < 0:
            selected[col] = (-1.0, abs(ic_ir))  # 翻转为正贡献
        else:
            selected[col] = (1.0, abs(ic_ir))

    # ── Step 3: 去冗余 (相关性 > max_correlation 只保留 |IC_IR| 更高的) ──
    if factor_corrs is not None and max_correlation < 1.0:
        # 按 |IC_IR| 降序排列
        sorted_factors = sorted(selected.keys(), key=lambda c: selected[c][1], reverse=True)
        kept = []
        for col in sorted_factors:
            too_similar = False
            for kept_col in kept:
                if col in factor_corrs.index and kept_col in factor_corrs.columns:
                    corr_val = abs(factor_corrs.loc[col, kept_col])
                    if corr_val > max_correlation:
                        too_similar = True
                        break
            if not too_similar:
                kept.append(col)

        removed_count = len(selected) - len(kept)
        if removed_count > 0:
            print(f"  去冗余: 移除 {removed_count} 个高相关因子, "
                  f"保留 {len(kept)} 个 (阈值={max_correlation})")
        selected = {k: selected[k] for k in kept}
    else:
        kept = list(selected.keys())

    # ── Step 4: 计算权重 ──
    if method == "ic_weighted":
        total_weight = sum(selected[c][1] for c in selected)
        if total_weight > 0:
            weights = {c: selected[c][1] / total_weight for c in selected}
        else:
            weights = {c: 1.0 / len(selected) for c in selected}
    else:
        # equal_weight
        weights = {c: 1.0 / len(selected) for c in selected}

    # ── Step 5: 合成 ──
    df["composite_factor"] = 0.0
    for col in selected:
        sign = selected[col][0]
        w = weights[col]
        col_data = df[col].fillna(0.0)
        df["composite_factor"] += sign * w * col_data

    # 打印合成信息
    print(f"  合成方法: {method} | 翻转符号: {flip_sign} | 去冗余: {max_correlation}")
    print(f"  因子权重:")
    for col in sorted(selected.keys(), key=lambda c: weights[c], reverse=True):
        sign_str = " (-)" if selected[col][0] < 0 else " (+)"
        print(f"    {col}: {weights[col]:.3f}{sign_str}  (|IC_IR|={selected[col][1]:.4f})")

    return df


def compute_nav(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """
    从收益序列计算累计净值。

    NAV_t = NAV_0 * Π(1 + r_i)
    """
    return initial_value * (1 + returns).cumprod()


def compute_performance(
    returns: pd.Series,
    freq: str = "M",
    rf: float = 0.02,
) -> dict:
    """
    绩效评估指标。

    这些是量化面试必问的指标——你必须能解释:
    1. 年化收益率: 平均每月赚 X% × 12
    2. 年化波动率: 每月收益的标准差 × √12
    3. Sharpe Ratio: 每承担一单位风险, 获得多少超额收益
    4. 最大回撤: 从最高点到最低点, 最多亏了多少
    5. Calmar Ratio: 年化收益 / |最大回撤|, 回撤调整后收益
    """
    periods_per_year = {"M": 12, "D": 252, "W": 52}.get(freq, 12)

    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else 0.0

    # 最大回撤计算
    nav = compute_nav(returns)
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_dd = float(drawdown.min())

    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0.0
    win_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0

    return {
        "Annualized_Return": round(float(ann_return), 4),
        "Volatility": round(float(ann_vol), 4),
        "Sharpe_Ratio": round(float(sharpe), 4),
        "Max_Drawdown": round(max_dd, 4),
        "Calmar_Ratio": round(float(calmar), 4),
        "Win_Rate": round(win_rate, 4),
        "Periods": int(len(returns)),
    }
