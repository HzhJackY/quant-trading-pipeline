"""
组合回测引擎。

多因子合成 → 选股 → 净值计算 → 绩效评估。

这是因子研究的最后一环——前面都是"这个因子好不好",
这里问的是"用这些因子做一个组合, 你能赚多少钱?"

v2.0: 新增交易成本感知回测 (Transaction-Cost-Aware Backtest)
  - compute_drifted_weights: 价格漂移修正后的实际权重
  - compute_oneway_turnover:   基于 Target → Drifted 差的真实单边换手率
  - compute_split_universe_trade_cost: 分域成本扣除 (大盘 vs 小盘)
  - run_backtest_with_costs:   持仓级全流程回测 (毛/净收益)
  - generate_comparison_table: 零摩擦 vs 扣费对比表
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def combine_factors(
    factor_df: pd.DataFrame,
    factor_cols: list[str] | None = None,
    method: str = "ic_weighted",
    return_col: str = "forward_return_1m",
    date_col: str = "date",
    max_correlation: float = 0.7,
    flip_sign: bool = True,
    rolling_window: int = 24,
    orthogonalize: bool = True,
    min_factor_variance: float = 1e-10,
) -> pd.DataFrame:
    """
    多因子合成: 将多个标准化后的因子合并成一个复合分数。

    方法:
    - orthogonalize=True (默认): Gram-Schmidt 正交化 + 滚动 IC_IR 加权
    - orthogonalize=False: 旧版贪心去冗余 + IC_IR 加权

    正交化管线 (orthogonalize=True):
      1. 计算 24 月滚动 |IC_IR|
      2. 按 |IC_IR| 降序排列因子
      3. Gram-Schmidt 正交化 (回归残差法)
      4. |IC_IR| 加权合成, 完全共线因子自动权重归零

    旧版管线 (orthogonalize=False):
      1. 全样本 IC_IR 计算
      2. 翻转负 IC_IR 因子
      3. 贪心去冗余 (|corr| > max_correlation 的因子丢弃)
      4. |IC_IR| 加权合成
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

    # ═══════════════════════════════════════════════════════════
    # 新版管线: Gram-Schmidt 正交化 + 滚动 IC_IR
    # ═══════════════════════════════════════════════════════════
    if orthogonalize and return_col in df.columns:
        from factor_research.orthogonalization import apply_gram_schmidt_composite

        return apply_gram_schmidt_composite(
            df,
            factor_cols=available,
            return_col=return_col,
            date_col=date_col,
            rolling_window=rolling_window,
            flip_sign=flip_sign,
            min_factor_variance=min_factor_variance,
            verbose=True,
        )

    # ═══════════════════════════════════════════════════════════
    # 旧版管线: 贪心去冗余 + 全样本 IC_IR (backward compat)
    # ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
# 交易成本感知回测 (v2.0 — 机构级 Net Sharpe Baseline)
# ═══════════════════════════════════════════════════════════


def compute_drifted_weights(
    target_weights: pd.Series,
    period_returns: pd.Series,
) -> pd.Series:
    """
    价格漂移修正：计算持仓经历一期收益后的自然权重。

    核心公式:
      Drifted_W_i = Target_W_i * (1 + R_i) / Σ_j [Target_W_j * (1 + R_j)]

    为什么必须做漂移修正:
      假设上期目标权重为 [A: 50%, B: 50%]。
      持有期间 A 涨了 10%, B 跌了 -5%。
      期末实际持仓权重变为:
        A: 0.5 × 1.10 / (0.5 × 1.10 + 0.5 × 0.95) = 53.66%
        B: 0.5 × 0.95 / 1.025 = 46.34%
      如果直接用 |Target_W_t - Target_W_{t-1}| 算换手,
      会把本不需要交易的 3.66% 也当作交易量 — 高估换手率。

    Parameters
    ----------
    target_weights : pd.Series (index=symbol)
        上期目标权重。
    period_returns : pd.Series (index=symbol)
        持有期间各股票的实际收益 (例如 forward_return_1m)。

    Returns
    -------
    pd.Series (index=symbol): 漂移后的权重, 和为 1。
    """
    # 对齐索引: 只取两者都有的股票
    common_idx = target_weights.index.intersection(period_returns.index)
    if len(common_idx) == 0:
        return pd.Series(dtype=float)

    w = target_weights.loc[common_idx].astype(float)
    r = period_returns.loc[common_idx].astype(float).fillna(0.0)

    # 向量化: drifted = w * (1 + r), 再归一化
    drifted = w * (1.0 + r)
    total = drifted.sum()
    if total <= 0 or np.isnan(total):
        return pd.Series(0.0, index=common_idx)

    return drifted / total


def compute_oneway_turnover(
    target_weights: pd.Series,
    drifted_weights: pd.Series,
) -> float:
    """
    真实单边换手率 (基于价格漂移修正)。

    单边换手率 = 0.5 × Σ_i |Target_W_i − Drifted_W_i|

    - 新股进入: Drifted_W = 0 → |Δw| = Target_W
    - 旧股退出: Target_W = 0 → |Δw| = Drifted_W
    - 权重增减: |Δw| 由 fill_value=0 自动处理

    Parameters
    ----------
    target_weights : pd.Series (index=symbol)
        当期目标权重。
    drifted_weights : pd.Series (index=symbol)
        漂移后的权重。

    Returns
    -------
    float: 单边换手率 (0-1)。
    """
    diff = target_weights.sub(drifted_weights, fill_value=0.0).abs()
    return float(0.5 * diff.sum())


def compute_split_universe_trade_cost(
    target_weights: pd.Series,
    drifted_weights: pd.Series,
    universe_map: pd.Series,
    cost_model: "TieredCostModel",
    daily_amounts: pd.Series,
    volatilities: pd.Series,
) -> dict:
    """
    分域交易成本扣除 (Split-Universe Cost Deduction)。

    算法:
      1. trade_vec = Target_W − Drifted_W  (fill_value=0 处理进出)
      2. 对每只成交股票 i:
           trade_value_i = |trade_vec_i| × AUM
           调用 cost_model.one_way_cost_bps(universe_i, trade_value_i,
               amount_i, vol_i, is_sell=Δw_i < 0)
           成本(占组合%) = |Δw_i| × cost_bps_i / 10000
      3. 按 universe 汇总, 输出大盘/小盘分项。

    Parameters
    ----------
    target_weights : pd.Series (index=symbol)
        当期目标权重。
    drifted_weights : pd.Series (index=symbol)
        漂移后权重。
    universe_map : pd.Series (index=symbol)
        每只股票所属子域 ("大盘" / "小盘")。
    cost_model : TieredCostModel
        AUM 感知的分层成本模型。
    daily_amounts : pd.Series (index=symbol)
        每只股票日均成交额 (元)。
    volatilities : pd.Series (index=symbol)
        每只股票日收益率波动率 (非年化, 如 0.017 = 1.7%)。

    Returns
    -------
    dict:
        total_cost_bps   — 总成本 (占组合比例, bps)
        total_cost_pct   — 总成本 (占组合比例, 小数)
        large_cost_bps   — 大盘分项 (bps)
        small_cost_bps   — 小盘分项 (bps)
        oneway_turnover  — 单边换手率
        n_trades         — 实际成交股票数
    """
    aum = cost_model.aum
    trade_vec = target_weights.sub(drifted_weights, fill_value=0.0)

    # 只处理有实际交易的股票 (|Δw| > 1e-8)
    active = trade_vec.abs() > 1e-8
    if not active.any():
        return {
            "total_cost_bps": 0.0,
            "total_cost_pct": 0.0,
            "large_cost_bps": 0.0,
            "small_cost_bps": 0.0,
            "oneway_turnover": 0.0,
            "n_trades": 0,
        }

    active_trades = trade_vec[active]
    oneway_to = compute_oneway_turnover(target_weights, drifted_weights)

    cost_pct_by_universe: dict[str, float] = {"大盘": 0.0, "小盘": 0.0}
    n_trades = 0

    for sym, delta_w in active_trades.items():
        # 获取该股票参数
        universe = str(universe_map.get(sym, "大盘"))
        amount = float(daily_amounts.get(sym, 0))
        vol = float(volatilities.get(sym, 0.02))

        # 跳过流动性为零的股票 (保护性检查)
        if amount <= 0:
            continue

        trade_value = abs(delta_w) * aum
        is_sell = delta_w < 0  # 减仓 → 卖方需缴印花税
        cost_bps = cost_model.one_way_cost_bps(
            universe=universe,
            trade_value=trade_value,
            daily_amount=amount,
            volatility=vol,
            is_sell=is_sell,
        )
        cost_pct = abs(delta_w) * cost_bps / 10000.0

        key = "大盘" if "小" not in universe else "小盘"
        cost_pct_by_universe.setdefault(key, 0.0)
        cost_pct_by_universe[key] += cost_pct
        n_trades += 1

    total_cost_pct = cost_pct_by_universe.get("大盘", 0.0) + cost_pct_by_universe.get("小盘", 0.0)
    total_cost_bps = total_cost_pct * 10000.0

    return {
        "total_cost_bps": round(total_cost_bps, 2),
        "total_cost_pct": round(total_cost_pct, 6),
        "large_cost_bps": round(cost_pct_by_universe.get("大盘", 0.0) * 10000, 2),
        "small_cost_bps": round(cost_pct_by_universe.get("小盘", 0.0) * 10000, 2),
        "oneway_turnover": round(oneway_to, 6),
        "n_trades": n_trades,
    }


def _compute_first_period_cost(
    target_weights: pd.Series,
    universe_map: pd.Series,
    cost_model: "TieredCostModel",
    daily_amounts: pd.Series,
    volatilities: pd.Series,
) -> dict:
    """
    首期建仓成本: 所有持仓均为买入 (无卖出/无印花税)。
    """
    aum = cost_model.aum
    cost_pct_by_universe: dict[str, float] = {}
    n_trades = 0

    for sym, w in target_weights.items():
        if w <= 0:
            continue
        universe = str(universe_map.get(sym, "大盘"))
        amount = float(daily_amounts.get(sym, 0))
        vol = float(volatilities.get(sym, 0.02))
        if amount <= 0:
            continue

        trade_value = w * aum
        cost_bps = cost_model.one_way_cost_bps(
            universe=universe,
            trade_value=trade_value,
            daily_amount=amount,
            volatility=vol,
            is_sell=False,  # 建仓全部为买入
        )
        cost_pct = w * cost_bps / 10000.0

        key = "大盘" if "小" not in universe else "小盘"
        cost_pct_by_universe[key] = cost_pct_by_universe.get(key, 0.0) + cost_pct
        n_trades += 1

    total_cost_pct = sum(cost_pct_by_universe.values())
    return {
        "total_cost_bps": round(total_cost_pct * 10000, 2),
        "total_cost_pct": round(total_cost_pct, 6),
        "large_cost_bps": round(cost_pct_by_universe.get("大盘", 0.0) * 10000, 2),
        "small_cost_bps": round(cost_pct_by_universe.get("小盘", 0.0) * 10000, 2),
        "oneway_turnover": 1.0,    # 建仓 = 100% 换手
        "n_trades": n_trades,
    }


# ═══════════════════════════════════════════════════════════
# 核心编排: 持仓级成本感知回测
# ═══════════════════════════════════════════════════════════


def run_backtest_with_costs(
    panel: pd.DataFrame,
    blended: pd.DataFrame,
    cost_model: "TieredCostModel",
    *,
    top_quantile: float = 0.3,
    min_stocks_per_universe: int = 5,
    date_col: str = "date",
    symbol_col: str = "symbol",
    return_col: str = "forward_return_1m",
    alpha_col: str = "alpha_signal",
    universe_col: str = "universe",
    close_col: str = "收盘",
    amount_col: str = "成交额",
    vol_col: str = "Vol_20D",
    timing_multipliers: dict[pd.Timestamp, float] | None = None,
) -> dict:
    """
    持仓级交易成本感知回测 (机构级 Net Sharpe Baseline)。

    每期调仓逻辑 (向量化):
      1. 合并 panel + blended, 确保 forward_return_1m 可用
      2. 按 universe 分域, 每期选取 top_quantile 股票, 等权分配
      3. 首期: 建仓成本 (无卖出印花税), turnover = 1.0
      4. 非首期:
         a. compute_drifted_weights(prev_weights, prev_period_returns)
         b. compute_split_universe_trade_cost(...)
         c. 净收益 = 毛收益 − 成本(占组合%)

    择时乘数 (market timing):
      - 如果 timing_multipliers 不为 None, 每期对 target weights 应用乘数
      - 例: multiplier=0.3 → 每只股票权重从 1/N 降为 1/N × 0.3
      - 注意: 上期的 prev_target_weights 也是缩放后的, 换手率自动反映
      - 与 cost 的交互: Trade value 按缩放后权重计算 → 成本自然等比例缩小

    Logger 每期输出:
      [YYYY-MM] TO=XX% | LargeCost=XXbps | SmallCost=XXbps | NetRet=XX%

    Parameters
    ----------
    panel : pd.DataFrame
        含 close_col, amount_col, vol_col 等。forward_return_1m 按需现算。
    blended : pd.DataFrame
        Split-Universe 输出, 含 alpha_col, universe_col。
    cost_model : TieredCostModel
        AUM 感知分层成本模型。
    top_quantile : float
        每期选取比例 (默认 0.3 = 前 30%)。
    min_stocks_per_universe : int
        单子域最少股票数, 低于此跳过。

    Returns
    -------
    dict:
        gross_returns:  pd.Series — 每期毛收益
        net_returns:    pd.Series — 每期净收益
        turnovers:      pd.Series — 每期单边换手率
        cost_breakdown: pd.DataFrame — 每期大盘/小盘成本明细
        gross_nav:      pd.Series — 毛净值曲线
        net_nav:        pd.Series — 净净值曲线
        gross_metrics:  dict — 零摩擦绩效指标
        net_metrics:    dict — 扣费后绩效指标
    """
    from factor_research.transaction_cost import TieredCostModel  # noqa: F811

    # ── 1. 准备数据 ──────────────────────────────────────
    df = panel.copy()

    # 确保 forward_return_1m 存在
    if return_col not in df.columns:
        logger.info("计算 forward_return_1m ...")
        close_col_actual = close_col
        if close_col_actual not in df.columns:
            # 推断收盘价列
            for c in df.columns:
                if "收" in str(c) or "close" in str(c).lower():
                    close_col_actual = c
                    break
            else:
                raise KeyError("未找到收盘价列, 无法计算 forward_return_1m")

        df = df.sort_values([symbol_col, date_col])
        df[return_col] = (
            df.groupby(symbol_col)[close_col_actual]
            .transform(lambda x: x.shift(-1) / x - 1.0)
        )

    # 合并 alpha_signal 和 universe
    merge_cols = [date_col, symbol_col, alpha_col, universe_col]
    available_merge = [c for c in merge_cols if c in blended.columns]
    df = df.merge(blended[available_merge], on=[date_col, symbol_col], how="inner")

    # 推断列名 (兼容中英文)
    def _find_col( candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    close_c = _find_col([close_col]) or close_col
    amount_c = _find_col([amount_col, "成交额", "amount", "Amount"])
    vol_c = _find_col([vol_col, "Vol_20D", "vol_20d"])

    if amount_c is None:
        raise KeyError(f"未找到成交额列。可用列: {df.columns.tolist()}")
    if vol_c is None:
        raise KeyError(f"未找到波动率列。可用列: {df.columns.tolist()}")

    # ── 2. 准备截面循环 ──────────────────────────────────
    dates = sorted(df[date_col].unique())
    n_dates = len(dates)
    if n_dates < 2:
        raise ValueError(f"数据不足: 仅有 {n_dates} 个截面")

    logger.info(
        "持仓级成本感知回测: %d 个截面, top_quantile=%.0f%%, AUM=%.0f万",
        n_dates, top_quantile * 100, cost_model.aum / 1e4,
    )

    # 累积序列
    gross_ret_list: list[float] = []
    net_ret_list: list[float] = []
    turnover_list: list[float] = []
    cost_rows: list[dict] = []

    # 上期状态
    prev_target_weights: pd.Series | None = None
    prev_forward_returns: pd.Series | None = None

    for i, dt in enumerate(dates):
        date_mask = df[date_col] == dt
        date_data = df[date_mask].set_index(symbol_col)

        # ── 选取当期持仓 (分域等权) ──
        # Pre-filter: remove uninvestable stocks (suspension, low liquidity, micro-cap)
        # These filters mirror the paper trading risk pre-filters
        investable_mask = pd.Series(True, index=date_data.index)

        # 1. Suspension / no-return filter: stock must have valid forward return
        if return_col in date_data.columns:
            investable_mask &= date_data[return_col].notna()

        # 2. Liquidity filter: 20-day avg turnover >= 5000万
        # In backtest, we use the single-period amount as a proxy
        if amount_c in date_data.columns:
            investable_mask &= date_data[amount_c].fillna(0) >= 50_000_000

        # 3. Market cap filter: skip micro-caps (< 50亿) if column available
        mcap_candidates = [c for c in date_data.columns if "mcap" in c.lower() or "总市值" in str(c) or "MCap" in str(c)]
        if mcap_candidates:
            mcap_col = mcap_candidates[0]
            investable_mask &= date_data[mcap_col].fillna(0) >= 5_000_000_000

        date_data = date_data[investable_mask]

        holdings = []
        for uni in ["大盘", "小盘"]:
            uni_data = date_data[date_data[universe_col] == uni]
            if len(uni_data) < min_stocks_per_universe:
                continue
            n_select = max(min_stocks_per_universe, int(len(uni_data) * top_quantile))
            top_n = uni_data.nlargest(n_select, alpha_col)
            holdings.append(top_n)

        if not holdings:
            # 跳过: 无合格持仓
            gross_ret_list.append(np.nan)
            net_ret_list.append(np.nan)
            turnover_list.append(np.nan)
            cost_rows.append({
                "date": dt, "total_cost_bps": np.nan,
                "large_cost_bps": np.nan, "small_cost_bps": np.nan,
            })
            prev_target_weights = None
            prev_forward_returns = None
            continue

        curr_holdings = pd.concat(holdings)
        n_holdings = len(curr_holdings)
        curr_target_weights = pd.Series(1.0 / n_holdings, index=curr_holdings.index)

        # ── 择时乘数: 获取当前期仓位缩放因子 ──
        timing_mult = 1.0
        if timing_multipliers is not None:
            timing_mult = timing_multipliers.get(pd.Timestamp(dt), 1.0)
            if timing_mult < 1.0:
                logger.info("  [择时] %s → 仓位乘数=%.1f (30%% in stocks, 70%% cash)",
                             str(dt)[:7], timing_mult)

        # ── 当期毛收益 (择时: 现金部分收益为 0) ──
        curr_forward_rets = curr_holdings[return_col].astype(float)
        gross_ret_100 = float((curr_target_weights * curr_forward_rets).sum())
        gross_ret = timing_mult * gross_ret_100

        # ── 计算成本和净收益 ──
        # 成交额: fillna 用中位数 (个别股票可能缺失)
        raw_amts = curr_holdings[amount_c].astype(float)
        amts = raw_amts.fillna(raw_amts.median() if raw_amts.notna().any() else 1e8)
        # Vol_20D 是年化波动率, 转换为日波动率; fillna 用截面中位数
        raw_vols = curr_holdings[vol_c].astype(float)
        daily_vols = (raw_vols / np.sqrt(252)).fillna(
            raw_vols.median() / np.sqrt(252) if raw_vols.notna().any() else 0.02
        )

        if i == 0 or prev_target_weights is None:
            # 首期: 建仓成本
            uni_map = curr_holdings[universe_col]
            cost_info = _compute_first_period_cost(
                curr_target_weights, uni_map, cost_model, amts, daily_vols,
            )
        else:
            # 非首期: 漂移 + 换仓成本
            uni_map = curr_holdings[universe_col]

            # 价格漂移 (weights 始终用 100% 计算, 择时只影响敞口)
            drifted = compute_drifted_weights(prev_target_weights, prev_forward_returns)

            cost_info = compute_split_universe_trade_cost(
                curr_target_weights, drifted, uni_map, cost_model, amts, daily_vols,
            )

        # 择时成本缩放: 实际交易规模 = timing_mult × 全仓交易规模
        if timing_mult < 1.0:
            for _key in ("total_cost_pct", "total_cost_bps",
                         "large_cost_bps", "small_cost_bps", "oneway_turnover"):
                if _key in cost_info and cost_info[_key] is not None:
                    cost_info[_key] *= timing_mult

        net_ret = gross_ret - cost_info["total_cost_pct"]

        # ── Logger 输出 ──
        mult_tag = f" | Mult={timing_mult:.1f}" if timing_mult < 1.0 else ""
        logger.info(
            "[%s] TO=%.1f%% | LargeCost=%.1fbps | SmallCost=%.1fbps | "
            "Gross=%.3f%% | Net=%.3f%% | N=%d%s",
            str(dt)[:10],
            cost_info["oneway_turnover"] * 100,
            cost_info["large_cost_bps"],
            cost_info["small_cost_bps"],
            gross_ret * 100,
            net_ret * 100,
            n_holdings,
            mult_tag,
        )

        # ── 保存 ──
        gross_ret_list.append(gross_ret)
        net_ret_list.append(net_ret)
        turnover_list.append(cost_info["oneway_turnover"])
        cost_rows.append({
            "date": dt,
            "total_cost_bps": cost_info["total_cost_bps"],
            "large_cost_bps": cost_info["large_cost_bps"],
            "small_cost_bps": cost_info["small_cost_bps"],
            "oneway_turnover": cost_info["oneway_turnover"],
            "n_trades": cost_info.get("n_trades", 0),
            "n_holdings": n_holdings,
        })

        # ── 更新上期状态 ──
        prev_target_weights = curr_target_weights
        prev_forward_returns = curr_forward_rets

    # ── 3. 构造返回 ──
    date_index = pd.DatetimeIndex(dates)
    gross_returns = pd.Series(gross_ret_list, index=date_index, name="gross_return")
    net_returns = pd.Series(net_ret_list, index=date_index, name="net_return")
    turnovers = pd.Series(turnover_list, index=date_index, name="oneway_turnover")
    cost_breakdown = pd.DataFrame(cost_rows).set_index("date")

    # 对齐: 仅保留 gross 和 net 均有效的期数, 确保可比性
    aligned = pd.DataFrame({
        "gross": gross_returns, "net": net_returns,
    }).dropna()
    aligned_gross = aligned["gross"]
    aligned_net = aligned["net"]

    gross_nav = compute_nav(aligned_gross)
    net_nav = compute_nav(aligned_net)

    # ── 4. 绩效评估 ──
    gross_metrics = None
    net_metrics = None
    if len(aligned_gross) > 0:
        gross_metrics = compute_performance(aligned_gross)
    if len(aligned_net) > 0:
        net_metrics = compute_performance(aligned_net)

    avg_to = turnovers.dropna().mean() if len(turnovers.dropna()) > 0 else 0.0
    avg_cost = (
        cost_breakdown["total_cost_bps"].dropna().mean()
        if not cost_breakdown.empty
        else 0.0
    )

    logger.info("=" * 56)
    logger.info("回测完成: %d 期 (gross-net 对齐后)", len(aligned_gross))
    logger.info(
        "零摩擦 | 年化收益=%.2f%%  Sharpe=%.4f  最大回撤=%.2f%%",
        gross_metrics["Annualized_Return"] * 100 if gross_metrics else 0,
        gross_metrics["Sharpe_Ratio"] if gross_metrics else 0,
        gross_metrics["Max_Drawdown"] * 100 if gross_metrics else 0,
    )
    logger.info(
        "扣费后 | 年化收益=%.2f%%  Sharpe=%.4f  最大回撤=%.2f%%",
        net_metrics["Annualized_Return"] * 100 if net_metrics else 0,
        net_metrics["Sharpe_Ratio"] if net_metrics else 0,
        net_metrics["Max_Drawdown"] * 100 if net_metrics else 0,
    )
    logger.info(
        "平均单边换手率=%.1f%%  平均成本=%.1fbps/期",
        avg_to * 100, avg_cost,
    )
    logger.info("=" * 56)

    return {
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "turnovers": turnovers,
        "cost_breakdown": cost_breakdown,
        "gross_nav": gross_nav,
        "net_nav": net_nav,
        "gross_metrics": gross_metrics,
        "net_metrics": net_metrics,
        "avg_turnover": float(avg_to),
        "avg_cost_bps": float(avg_cost),
    }


# ═══════════════════════════════════════════════════════════
# 对比表生成
# ═══════════════════════════════════════════════════════════


def generate_comparison_table(
    result: dict,
    aum: float | None = None,
) -> str:
    """
    生成零摩擦 vs 扣费 绩效对比 Markdown 表格。

    Parameters
    ----------
    result : dict
        run_backtest_with_costs() 的返回字典。
    aum : float | None
        资金规模 (可选, 用于表格标题)。

    Returns
    -------
    str: Markdown 格式对比表。
    """
    gm = result.get("gross_metrics") or {}
    nm = result.get("net_metrics") or {}
    avg_to = result.get("avg_turnover", 0.0)
    avg_cost = result.get("avg_cost_bps", 0.0)

    def fmt_pct(v: float | None) -> str:
        if v is None:
            return "N/A"
        return f"{v * 100:.2f}%"

    def fmt_num(v: float | None, decimals: int = 4) -> str:
        if v is None:
            return "N/A"
        return f"{v:.{decimals}f}"

    aum_str = f" (AUM {aum/1e4:.0f}万)" if aum else ""
    lines = [
        f"## 交易成本对比报告{aum_str}",
        "",
        "| 指标 | Zero-Friction Baseline | Net-Friction Baseline | Δ |",
        "|------|----------------------|----------------------|---|",
        f"| 年化收益 | {fmt_pct(gm.get('Annualized_Return'))} | "
        f"{fmt_pct(nm.get('Annualized_Return'))} | "
        f"{fmt_pct((gm.get('Annualized_Return', 0) or 0) - (nm.get('Annualized_Return', 0) or 0))} |",
        f"| 年化波动率 | {fmt_pct(gm.get('Volatility'))} | "
        f"{fmt_pct(nm.get('Volatility'))} | — |",
        f"| Sharpe Ratio | {fmt_num(gm.get('Sharpe_Ratio'))} | "
        f"{fmt_num(nm.get('Sharpe_Ratio'))} | "
        f"{fmt_num((gm.get('Sharpe_Ratio', 0) or 0) - (nm.get('Sharpe_Ratio', 0) or 0))} |",
        f"| 最大回撤 | {fmt_pct(gm.get('Max_Drawdown'))} | "
        f"{fmt_pct(nm.get('Max_Drawdown'))} | — |",
        f"| Calmar Ratio | {fmt_num(gm.get('Calmar_Ratio'))} | "
        f"{fmt_num(nm.get('Calmar_Ratio'))} | — |",
        f"| 月胜率 | {fmt_pct(gm.get('Win_Rate'))} | "
        f"{fmt_pct(nm.get('Win_Rate'))} | — |",
        f"| 平均单边换手率 | — | {avg_to*100:.1f}% | — |",
        f"| 平均每期成本 | — | {avg_cost:.1f} bps | — |",
        f"| 回测期数 | {gm.get('Periods', 'N/A')} | "
        f"{nm.get('Periods', 'N/A')} | — |",
        "",
        "> **注:** Net-Friction 已扣除佣金(万2.5) + 印花税(万5,卖方) + 过户费 + "
        "滑点 + Almgren-Chriss 市场冲击。",
    ]
    return "\n".join(lines)
