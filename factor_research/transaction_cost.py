"""
交易成本与流动性筛选模块 (Transaction Cost & Liquidity Filter).

机构实盘与学术回测的核心区别在于是否考虑交易摩擦。本模块实现:
  1. 流动性筛选 (LiquidityFilter) — 成交额/停牌/涨跌停三层过滤
  2. AUM 感知分层成本模型 (TieredCostModel) — Almgren-Chriss 冲击估算
  3. 换手率计算与成本扣除

核心公式 (Almgren-Chriss):
  Market_Impact(bps) = eta * sigma * (Q / ADV)^gamma
  其中:
    sigma = 日波动率
    Q     = 交易金额 = AUM * weight
    ADV   = 日均成交额
    gamma = 0.5 (大盘, 平方根法则) / 0.6-0.8 (小盘, 流动性折价更陡)
    eta   = 缩放系数 (一般取 1.0, 校准后可调整)

用法:
  from factor_research.transaction_cost import LiquidityFilter, TieredCostModel

  # 流动性筛选
  lf = LiquidityFilter(min_daily_amount=10_000_000)
  tradeable = lf.filter(panel, date="2024-01-31")

  # 成本估算
  cost_model = TieredCostModel(aum=50_000_000)  # 5000万
  cost_bps = cost_model.one_way_cost(
      universe="小盘", trade_value=500_000,
      daily_amount=15_000_000, volatility=0.028,
  )
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("transaction_cost")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════
# 1. 流动性筛选器 (Liquidity Filter)
# ═══════════════════════════════════════════════════════════

class LiquidityFilter:
    """
    流动性筛选器 — 模拟机构实盘中"能买得到"和"能卖得掉"的约束。

    三层过滤逻辑:
      Layer 1 (成交额): 剔除日均成交额过低 (无法容纳机构资金)
      Layer 2 (停牌):   剔除近期停牌天数过多的股票 (流动性中断)
      Layer 3 (涨跌停): 剔除一字涨跌停天数过多的股票 (无法以公允价格成交)

    为什么涨跌停必须过滤:
      一字涨停 → 买不进 (卖方挂单为零, 排队排不到)
      一字跌停 → 卖不掉 (买方挂单为零, 流动性完全消失)
      对策略的影响:
        - 如果信号推荐买入一只即将一字涨停的票, 实际无法成交 → 信号虚高
        - 如果持仓中有一只一字跌停的票, 无法止损 → 回撤被低估

    参数
    ----
    min_daily_amount : float
        最低日均成交额 (元)。默认 1000 万, 即剔除日均成交额低于 1000 万的股票。
        机构经验值:
          - 1000 万 (小私募, AUM < 1 亿)
          - 3000 万 (中型私募, AUM 1-5 亿)
          - 5000 万 (公募/大私募, AUM 5-20 亿)
    max_suspension_ratio : float
        最大停牌比例 (相对于 lookback 交易日数)。默认 0.10 (10%)。
        超过此比例的股票被视为流动性风险过高, 在当期调仓中排除。
    max_limit_ratio : float
        最大涨跌停比例 (相对于 lookback 交易日数)。默认 0.05 (5%)。
    limit_threshold : float
        涨跌停判定阈值 (涨跌幅绝对值)。主板 ±9.8%, 科创/创业板 ±19.8%。
        实际上不同板块阈值不同, 这里用一个保守值。
        如需精确检测, 使用 detect_limit_events_from_daily() 并传入板块信息。
    lookback_days : int
        回看交易日数, 默认 60 (约 3 个月)。
    amount_col : str
        成交额列名。
    date_col : str
        日期列名。
    symbol_col : str
        股票代码列名。
    """

    def __init__(
        self,
        min_daily_amount: float = 10_000_000,       # 1000 万
        max_suspension_ratio: float = 0.10,          # 10%
        max_limit_ratio: float = 0.05,               # 5%
        limit_threshold: float = 0.098,              # ±9.8% (主板)
        lookback_days: int = 60,
        amount_col: str = "成交额",
        date_col: str = "date",
        symbol_col: str = "symbol",
    ):
        self.min_daily_amount = min_daily_amount
        self.max_suspension_ratio = max_suspension_ratio
        self.max_limit_ratio = max_limit_ratio
        self.limit_threshold = limit_threshold
        self.lookback_days = lookback_days
        self.amount_col = amount_col
        self.date_col = date_col
        self.symbol_col = symbol_col

        # 推断成交额列名 (兼容英文列名)
        if amount_col not in ["成交额"]:
            pass  # 使用传入的列名
        logger.info(
            "LiquidityFilter: min_amount=%.0f万, max_susp=%.0f%%, "
            "max_limit=%.0f%%, lookback=%d天",
            min_daily_amount / 1e4, max_suspension_ratio * 100,
            max_limit_ratio * 100, lookback_days,
        )

    # ── Layer 1: 成交额过滤 ──────────────────────────────

    def filter_by_amount(
        self,
        panel: pd.DataFrame,
    ) -> pd.Series:
        """
        成交额过滤: 剔除日均成交额 < min_daily_amount 的股票。

        注意: 这里的 panel 是月频数据, 所以使用的是该月最后交易日的成交额
        作为该月流动性的代理变量。更精确的做法是取近 20 日均值,
        但单个截面上的差异足够用于排序过滤。

        返回
        ----
        pd.Series[bool]: True = 通过筛选。
        """
        if self.amount_col not in panel.columns:
            # 尝试推断
            for candidate in ["成交额", "amount", "Amount"]:
                if candidate in panel.columns:
                    self.amount_col = candidate
                    break
            else:
                logger.warning("未找到成交额列, 跳过成交额过滤")
                return pd.Series(True, index=panel.index)

        amount = panel[self.amount_col].astype(float)
        passed = amount >= self.min_daily_amount

        n_total = len(panel)
        n_passed = passed.sum()
        logger.info(
            "  [Layer 1 成交额] 通过 %d/%d (%.1f%%), "
            "阈值 >= %.0f 万元",
            n_passed, n_total,
            100 * n_passed / max(n_total, 1),
            self.min_daily_amount / 1e4,
        )
        return passed

    # ── Layer 2: 停牌检测 ─────────────────────────────────

    def detect_suspension(
        self,
        panel: pd.DataFrame,
        daily_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> pd.Series:
        """
        停牌检测: 标记近期停牌天数超过阈值的股票。

        方法 A (有日线数据, 精确):
          对每只股票, 在 lookback 窗口内, 统计以下特征的天数:
            - 成交量为 0 或 NaN
            - 开盘价 == 收盘价 == 最高价 == 最低价 (一字横盘, 极可能是停牌)
          如果满足条件的比例 > max_suspension_ratio, 则排除。

        方法 B (仅有月频数据, 近似):
          对每只股票, 统计在 lookback 期间内:
            - 该股票在面板中出现的次数 vs 该期间总截面数
            - 如果缺失比例 > max_suspension_ratio, 则排除。

        参数
        ----
        panel : pd.DataFrame
            月频因子面板。
        daily_data : dict[str, pd.DataFrame] | None
            {symbol: daily_df} 日线数据字典。如果为 None, 使用方法 B。

        返回
        ----
        pd.Series[bool]: True = 通过筛选 (停牌天数在阈值内)。
        """
        if daily_data is not None:
            return self._detect_suspension_from_daily(panel, daily_data)
        else:
            return self._detect_suspension_from_monthly(panel)

    def _detect_suspension_from_monthly(
        self,
        panel: pd.DataFrame,
    ) -> pd.Series:
        """
        从月频数据近似检测停牌。

        逻辑: 统计每只股票在 lookback 窗口内出现的月份数。
        如果某只股票在 N 个月的窗口中只出现了 M 个月,
        且 M/N < (1 - max_suspension_ratio), 则认为停牌过多。

        局限性:
          - 无法区分"停牌"和"新上市/退市"
          - 月频数据可能低估短期停牌 (停 3 周但在月末复牌 → 月频数据看不出来)
          - 对于机构级回测, 建议传入 daily_data 使用精确方法
        """
        if self.date_col not in panel.columns or self.symbol_col not in panel.columns:
            logger.warning("缺少 date 或 symbol 列, 跳过停牌检测")
            return pd.Series(True, index=panel.index)

        # 转为 datetime
        dates = pd.to_datetime(panel[self.date_col].unique())
        dates = sorted(dates)

        if len(dates) < 2:
            logger.warning("数据不足 (<2 期), 跳过停牌检测")
            return pd.Series(True, index=panel.index)

        # 计算每只股票出现的月份数
        symbol_date_counts = panel.groupby(self.symbol_col)[self.date_col].nunique()

        # 估算 lookback 内的预期月数
        # 月频: lookback_days=60 ≈ 3 个交易月
        expected_months = max(2, int(self.lookback_days / 20))
        max_allowed_missing = int(expected_months * self.max_suspension_ratio)

        # 对每个 symbol, 判定
        results = {}
        for sym in panel[self.symbol_col].unique():
            n_appeared = symbol_date_counts.get(sym, 0)
            n_missing = expected_months - min(n_appeared, expected_months)
            results[sym] = n_missing <= max_allowed_missing

        passed = panel[self.symbol_col].map(results).fillna(True)

        n_total = len(panel)
        n_passed = passed.sum()
        logger.info(
            "  [Layer 2 停牌(月频近似)] 通过 %d/%d (%.1f%%), "
            "允许缺失 <= %d 月",
            n_passed, n_total,
            100 * n_passed / max(n_total, 1),
            max_allowed_missing,
        )
        return pd.Series(passed, index=panel.index)

    def _detect_suspension_from_daily(
        self,
        panel: pd.DataFrame,
        daily_data: dict[str, pd.DataFrame],
    ) -> pd.Series:
        """
        从日线数据精确检测停牌。

        停牌判断标准 (同时满足):
          1. 成交量 = 0 或 NaN
          2. (可选) 当日涨跌幅 = 0 (非交易状态)

        对每只股票, 在 lookback 窗口内统计停牌天数比例。
        """
        results = {}
        for sym in panel[self.symbol_col].unique():
            if sym not in daily_data:
                results[sym] = True  # 无数据, 保守通过
                continue

            ddf = daily_data[sym]
            # 推断日期列和成交量列
            date_col_daily = ddf.columns[0]  # 第一列通常是日期
            vol_col = None
            for c in ddf.columns:
                c_lower = str(c).lower()
                if "volume" in c_lower or "成交" in c_lower or "量" in c_lower:
                    vol_col = c
                    break
            if vol_col is None:
                vol_col = ddf.columns[5] if len(ddf.columns) > 5 else ddf.columns[-1]

            # 取最近 lookback_days 个交易日
            ddf_sorted = ddf.sort_values(date_col_daily).tail(self.lookback_days)
            if len(ddf_sorted) < 5:
                results[sym] = True
                continue

            vol = pd.to_numeric(ddf_sorted[vol_col], errors="coerce")
            suspended = (vol.isna()) | (vol <= 0)
            suspension_ratio = suspended.sum() / max(len(ddf_sorted), 1)
            results[sym] = suspension_ratio <= self.max_suspension_ratio

        passed = panel[self.symbol_col].map(results).fillna(True)
        n_total = len(panel)
        n_passed = passed.sum()
        logger.info(
            "  [Layer 2 停牌(日频精确)] 通过 %d/%d (%.1f%%), "
            "阈值 <= %.0f%%",
            n_passed, n_total,
            100 * n_passed / max(n_total, 1),
            self.max_suspension_ratio * 100,
        )
        return pd.Series(passed, index=panel.index)

    # ── Layer 3: 涨跌停检测 ────────────────────────────────

    def detect_limit_events(
        self,
        panel: pd.DataFrame,
        daily_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> pd.Series:
        """
        涨跌停检测: 标记近期一字涨跌停天数过多的股票。

        一字涨跌停的定义:
          一字涨停: 开盘价 ≈ 最高价 ≈ 最低价 (全天封板, 几乎无成交)
          一字跌停: 开盘价 ≈ 最高价 ≈ 最低价 (全天跌停, 卖不出去)

        检测逻辑 (需要日线 OHLC 数据):
          1. 计算当日涨跌幅 (close / prev_close - 1)
          2. 如果 |涨跌幅| >= limit_threshold → 候选
          3. 如果同时 (high - low) / prev_close < 0.002 (振幅极小) → 确认为一字板

        **重要**: 涨跌停是日内事件, 无法从月频数据可靠检测。
        如果 daily_data=None, 此检测将被跳过 (全部通过),
        并记录警告。对于机构级回测, 强烈建议提供日线数据。

        参数
        ----
        panel : pd.DataFrame
            月频因子面板。
        daily_data : dict[str, pd.DataFrame] | None
            日线数据字典。为 None 时跳过此检测。

        返回
        ----
        pd.Series[bool]: True = 通过筛选。
        """
        if daily_data is not None:
            return self._detect_limit_from_daily(panel, daily_data)
        else:
            logger.warning(
                "  [Layer 3 涨跌停] 未提供日线数据, 跳过一字板检测。"
                "月频数据无法可靠区分'一字涨跌停'和'正常大幅波动'。"
            )
            return pd.Series(True, index=panel.index)

    def _detect_limit_from_monthly(
        self,
        panel: pd.DataFrame,
    ) -> pd.Series:
        """
        从月频数据近似检测涨跌停。

        方法: 按 symbol 分组计算月收益率, 检查 |月收益| >= limit_threshold。
        如果某月涨跌幅接近涨停板, 则标记为潜在涨跌停事件。

        这是非常粗糙的近似 (正常月份也可能涨跌 10%+),
        建议对关键回测使用日线数据精确检测。

        注意: 需要按 symbol 分组后再 pct_change, 避免跨股票边界溢出。
        """
        # 尝试用收盘价计算月收益
        close_col = None
        for c in ["收盘", "close", "Close"]:
            if c in panel.columns:
                close_col = c
                break

        if close_col is None:
            logger.warning("未找到收盘价列, 跳过涨跌停检测")
            return pd.Series(True, index=panel.index)

        # 按 symbol 分组计算月收益 (每组独立计算, 避免跨股票 pct_change 溢出)
        panel_sorted = panel.sort_values([self.symbol_col, self.date_col]).copy()
        monthly_ret = pd.Series(np.nan, index=panel_sorted.index)

        for sym, grp_idx in panel_sorted.groupby(self.symbol_col).groups.items():
            grp = panel_sorted.loc[grp_idx]
            rets = grp[close_col].pct_change()
            monthly_ret.loc[grp_idx] = rets

        # 标记涨跌停
        limit_hit = monthly_ret.abs() >= self.limit_threshold

        # 统计每只股票在 lookback 内的涨跌停比例
        # panel 是月频, lookback_days=60 ≈ 3 月
        expected_months = max(2, int(self.lookback_days / 20))
        max_allowed = int(expected_months * self.max_limit_ratio)

        results = {}
        # 使用 panel_sorted 的索引来获取每只股票的数据
        for sym in panel[self.symbol_col].unique():
            sym_idx = panel_sorted.index[panel_sorted[self.symbol_col] == sym]
            if len(sym_idx) == 0:
                results[sym] = True
                continue

            # 取最近 expected_months 个观测
            n_tail = min(expected_months, len(sym_idx))
            tail_idx = sym_idx[-n_tail:]
            n_limits = limit_hit.loc[tail_idx].sum()
            results[sym] = n_limits <= max_allowed

        passed = panel[self.symbol_col].map(results).fillna(True)
        n_total = len(panel)
        n_passed = passed.sum()
        logger.info(
            "  [Layer 3 涨跌停(月频近似)] 通过 %d/%d (%.1f%%), "
            "阈值 |ret|>=%.0f%%, 允许 <= %d 次/%.0f月",
            n_passed, n_total,
            100 * n_passed / max(n_total, 1),
            self.limit_threshold * 100,
            max_allowed, expected_months,
        )
        return pd.Series(passed, index=panel.index)

    def _detect_limit_from_daily(
        self,
        panel: pd.DataFrame,
        daily_data: dict[str, pd.DataFrame],
    ) -> pd.Series:
        """
        从日线数据精确检测一字涨跌停。

        一字板判定标准 (需同时满足):
          a) |daily_return| >= limit_threshold (接近涨跌停板)
          b) (high - low) / prev_close < 0.002 (日内振幅 < 0.2%, 即一字横盘)

        条件 b 是一字板区别于普通涨跌停的关键:
          - 普通涨停: 开板后可能被撬开, 有成交量, 振幅大
          - 一字涨停: 全天封死, 开盘=收盘≈最高≈最低, 几乎零振幅
        """
        board_type_map = {}  # symbol -> limit threshold (主板10% vs 科创20%)
        for sym in panel[self.symbol_col].unique():
            code = str(sym)
            if code.startswith("300") or code.startswith("301"):
                board_type_map[sym] = 0.198  # 创业板 ±20%
            elif code.startswith("688"):
                board_type_map[sym] = 0.198  # 科创板 ±20%
            elif code.startswith("8") or code.startswith("4"):
                board_type_map[sym] = 0.298  # 北交所 ±30%
            else:
                board_type_map[sym] = 0.098  # 主板 ±10% (ST为5%, 但CSI 800基本无ST)

        results = {}
        for sym in panel[self.symbol_col].unique():
            if sym not in daily_data:
                results[sym] = True
                continue

            ddf = daily_data[sym]
            if len(ddf) < 20:
                results[sym] = True
                continue

            threshold = board_type_map.get(sym, 0.098)

            # 推断 OHLC 列 (daily cache: 日期, 开, 高, 低, 收, 量, 额, 换手率)
            cols = ddf.columns.tolist()
            # 通常: [0]=date, [1]=open, [2]=high, [3]=low, [4]=close
            if len(cols) >= 5:
                date_c = cols[0]
                open_c = cols[1]
                high_c = cols[2]
                low_c = cols[3]
                close_c = cols[4]
            else:
                results[sym] = True
                continue

            ddf_sorted = ddf.sort_values(date_c).tail(self.lookback_days)
            if len(ddf_sorted) < 5:
                results[sym] = True
                continue

            close = pd.to_numeric(ddf_sorted[close_c], errors="coerce")
            high = pd.to_numeric(ddf_sorted[high_c], errors="coerce")
            low = pd.to_numeric(ddf_sorted[low_c], errors="coerce")
            open_p = pd.to_numeric(ddf_sorted[open_c], errors="coerce")

            # 计算日收益率
            daily_ret = close.pct_change()
            prev_close = close.shift(1)

            # 振幅 = (high - low) / prev_close
            amplitude = (high - low) / prev_close

            # 一字板判定
            is_limit = daily_ret.abs() >= threshold
            is_one_word = amplitude < 0.002  # 振幅 < 0.2%
            one_word_limits = is_limit & is_one_word

            limit_ratio = one_word_limits.sum() / max(len(ddf_sorted), 1)
            results[sym] = limit_ratio <= self.max_limit_ratio

        passed = panel[self.symbol_col].map(results).fillna(True)
        n_total = len(panel)
        n_passed = passed.sum()
        logger.info(
            "  [Layer 3 涨跌停(日频精确)] 通过 %d/%d (%.1f%%), "
            "阈值 <= %.0f%% 一字板",
            n_passed, n_total,
            100 * n_passed / max(n_total, 1),
            self.max_limit_ratio * 100,
        )
        return pd.Series(passed, index=panel.index)

    # ── 综合过滤 ──────────────────────────────────────────

    def filter(
        self,
        panel: pd.DataFrame,
        daily_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        执行全部三层流动性筛选, 返回过滤后的面板。

        处理顺序:
          1. 成交额过滤 (必要)
          2. 停牌检测 (可选, 需要日线数据才能精确)
          3. 涨跌停检测 (可选, 需要日线数据才能精确)

        参数
        ----
        panel : pd.DataFrame
            因子面板 (单期或全期)。
        daily_data : dict[str, pd.DataFrame] | None
            日线数据字典。为 None 时使用月频近似。

        返回
        ----
        pd.DataFrame: 过滤后的面板 + 新增列 "liquidity_pass" (True=通过全检)。
        """
        logger.info("=" * 56)
        logger.info("流动性筛选 (Liquidity Filter)")
        logger.info("=" * 56)

        n_before = len(panel)
        panel = panel.copy()

        # Layer 1: 成交额
        pass_amount = self.filter_by_amount(panel)

        # Layer 2: 停牌
        pass_susp = self.detect_suspension(panel, daily_data)

        # Layer 3: 涨跌停
        pass_limit = self.detect_limit_events(panel, daily_data)

        # 综合
        panel["liquidity_pass"] = pass_amount & pass_susp & pass_limit

        n_after = panel["liquidity_pass"].sum()
        n_excluded = n_before - n_after

        logger.info(
            "综合过滤结果: %d/%d 通过 (%.1f%%), 排除 %d 个样本",
            n_after, n_before,
            100 * n_after / max(n_before, 1),
            n_excluded,
        )

        # 统计排除原因
        fail_amount = (~pass_amount).sum()
        fail_susp = (~pass_susp).sum()
        fail_limit = (~pass_limit).sum()
        logger.info(
            "排除原因: 成交额不足=%d | 停牌过多=%d | 涨跌停过多=%d",
            fail_amount, fail_susp, fail_limit,
        )
        logger.info("=" * 56)

        return panel


# ═══════════════════════════════════════════════════════════
# 2. 分层交易成本模型 (Tiered Cost Model)
# ═══════════════════════════════════════════════════════════

@dataclass
class UniverseCostConfig:
    """
    单个子域的交易成本配置。

    参数
    ----
    commission_bps : float
        佣金费率 (bps, 单边)。A股机构约万2.5 = 2.5 bps。
    stamp_duty_bps : float
        印花税 (bps, 仅卖方)。A股 2023年8月起万5 = 5 bps。
    transfer_fee_bps : float
        过户费 (bps, 双边)。A股约十万分之一 ≈ 0.1 bps。
    base_slippage_bps : float
        基础滑点 (bps, 单边)。不含市场冲击, 仅 Bid-Ask spread。
    impact_gamma : float
        Almgren-Chriss 冲击指数 gamma。
        大盘: 0.5 (平方根法则, 流动性充裕)
        小盘: 0.6-0.8 (流动性折价, 冲击更陡)
        微盘: 0.8-1.0 (基本线性, 每多买一点都推高价格)
    impact_eta : float
        冲击缩放系数。默认 1.0, 可通过实盘数据校准。
    min_commission : float
        最低佣金 (元/笔)。A股默认5元。
    """

    commission_bps: float = 2.5        # 万2.5
    stamp_duty_bps: float = 5.0        # 万5 (2023年8月起)
    transfer_fee_bps: float = 0.1      # 十万分之一
    base_slippage_bps: float = 5.0     # 基础滑点 5 bps
    impact_gamma: float = 0.5          # 冲击指数
    impact_eta: float = 1.0            # 冲击缩放
    min_commission: float = 5.0        # 最低5元/笔


class TieredCostModel:
    """
    AUM 感知的分层交易成本模型。

    核心思想:
      交易成本不是固定比例, 而是随资金规模 (AUM) 非线性增长的。
      同样买 500 万的股票:
        - 大盘股 (日均成交额 10 亿): Participation Rate = 0.05% → 冲击几乎为零
        - 小盘股 (日均成交额 2000 万): Participation Rate = 25% → 冲击显著

    Almgren-Chriss 冲击公式:
      Market_Impact(bps) = eta * sigma * (trade_value / daily_amount)^gamma * 10000

      其中:
        sigma       = 日收益率波动率 (如 0.02 = 2%)
        trade_value = 交易金额 (= AUM × 个股权重)
        daily_amount = 股票日均成交额
        gamma       = 冲击指数 (0.5 → 平方根法则)
        eta         = 缩放系数

    总单边成本 = 佣金 + 印花税(仅卖) + 过户费 + 基础滑点 + 市场冲击

    参数
    ----
    aum : float
        管理规模 (元)。默认 1000 万。
        这是核心参数 —— AUM 越大, 同样的交易金额占流动性比例越高,
        冲击成本越大。
    large_cap_config : UniverseCostConfig
        大盘股成本配置。
    small_cap_config : UniverseCostConfig
        小盘股成本配置 (gamma 更高, slippage 更大)。
    """

    def __init__(
        self,
        aum: float = 10_000_000,              # 1000 万
        large_cap_config: Optional[UniverseCostConfig] = None,
        small_cap_config: Optional[UniverseCostConfig] = None,
    ):
        self.aum = aum

        # 大盘默认配置: 标准滑点 + 平方根冲击
        self.large_config = large_cap_config or UniverseCostConfig(
            base_slippage_bps=5.0,
            impact_gamma=0.5,
            impact_eta=1.0,
        )

        # 小盘默认配置: 更高滑点 + 更陡冲击曲线
        self.small_config = small_cap_config or UniverseCostConfig(
            base_slippage_bps=15.0,     # 小盘 bid-ask spread 更大
            impact_gamma=0.65,          # 流动性折价, 冲击更陡
            impact_eta=1.5,             # 冲击更显著
        )

        logger.info(
            "TieredCostModel: AUM=%.0f万 | "
            "大盘 gamma=%.2f slip=%dbps | "
            "小盘 gamma=%.2f slip=%dbps",
            aum / 1e4,
            self.large_config.impact_gamma,
            self.large_config.base_slippage_bps,
            self.small_config.impact_gamma,
            self.small_config.base_slippage_bps,
        )

    # ── 市场冲击估算 ──────────────────────────────────────

    def estimate_market_impact_bps(
        self,
        trade_value: float,
        daily_amount: float,
        volatility: float,
        gamma: float,
        eta: float,
    ) -> float:
        """
        Almgren-Chriss 市场冲击估算 (bps)。

        公式:
          Impact(bps) = eta * sigma * (Q / ADV)^gamma * 10000

        其中 Q = trade_value, ADV = daily_amount。

        参数
        ----
        trade_value : float
            该股票的交易金额 (元)。
        daily_amount : float
            该股票的日均成交额 (元)。
        volatility : float
            该股票的日收益率波动率 (如 0.02 = 2%)。
        gamma : float
            冲击指数。
        eta : float
            缩放系数。

        返回
        ----
        float: 市场冲击 (bps, 单边)。

        示例
        ----
        小盘股: 成交 50 万, 日均 2000 万, vol=2.8%, gamma=0.65
          Participation = 50/2000 = 2.5%
          Impact ≈ 1.5 × 0.028 × (0.025)^0.65 × 10000 ≈ 39 bps
        """
        if daily_amount < 1e-4 or trade_value < 1e-4:
            return 0.0

        participation = trade_value / daily_amount
        # 防止 participation rate 异常大
        participation = min(participation, 1.0)

        impact = eta * volatility * (participation ** gamma) * 10000
        return float(impact)

    # ── 单边/往返成本 ─────────────────────────────────────

    def one_way_cost_bps(
        self,
        universe: str,
        trade_value: float,
        daily_amount: float,
        volatility: float,
        is_sell: bool = False,
    ) -> float:
        """
        计算单边交易总成本 (bps)。

        买入成本 = 佣金 + 过户费 + 滑点 + 市场冲击
        卖出成本 = 佣金 + 过户费 + 滑点 + 市场冲击 + 印花税 ← 仅卖方

        参数
        ----
        universe : str
            子域名称 ("大盘" / "小盘")。
        trade_value : float
            该笔交易金额 (元)。
        daily_amount : float
            该股票日均成交额 (元)。
        volatility : float
            该股票日波动率。
        is_sell : bool
            是否为卖出 (卖出需加印花税)。

        返回
        ----
        float: 单边总成本 (bps)。
        """
        config = self._get_config(universe)

        # 市场冲击 (bps)
        impact = self.estimate_market_impact_bps(
            trade_value, daily_amount, volatility,
            gamma=config.impact_gamma,
            eta=config.impact_eta,
        )

        # 固定费用
        total = config.commission_bps + config.transfer_fee_bps

        # 卖方印花税
        if is_sell:
            total += config.stamp_duty_bps

        # 基础滑点 + 市场冲击
        total += config.base_slippage_bps + impact

        return total

    def round_trip_cost_bps(
        self,
        universe: str,
        buy_value: float,
        sell_value: float,
        daily_amount: float,
        volatility: float,
    ) -> float:
        """
        计算往返交易总成本 (bps)。

        往返 = 买入成本 + 卖出成本
        注意: 买入和卖出的金额可能不同 (因持仓期间股价变化),
        且卖出额外含印花税。

        返回
        ----
        float: 往返总成本 (bps)。
        """
        buy_cost = self.one_way_cost_bps(
            universe, buy_value, daily_amount, volatility, is_sell=False,
        )
        sell_cost = self.one_way_cost_bps(
            universe, sell_value, daily_amount, volatility, is_sell=True,
        )
        return buy_cost + sell_cost

    # ── 预估组合总成本 ────────────────────────────────────

    def estimate_portfolio_cost_bps(
        self,
        holdings: pd.DataFrame,
        volatility_col: str = "Vol_20D",
        amount_col: str = "成交额",
    ) -> dict:
        """
        估算给定持仓组合的总交易成本 (bps)。

        假设等权持仓, 每只股票的交易金额 = AUM / n_stocks。

        参数
        ----
        holdings : pd.DataFrame
            持仓表, 含 universe, volatility, 成交额等列。
        volatility_col : str
            波动率列名。
        amount_col : str
            成交额列名。

        返回
        ----
        dict: {
            avg_one_way_bps: 平均单边成本 (bps),
            avg_impact_bps:  平均市场冲击 (bps),
            total_one_way_bps: 总单边成本,
            n_stocks: 持仓数量,
            cost_breakdown: {universe: {avg_impact, avg_slippage, avg_commission, ...}},
        }
        """
        n_stocks = len(holdings)
        if n_stocks == 0:
            return {"avg_one_way_bps": 0.0, "n_stocks": 0}

        trade_value_per_stock = self.aum / n_stocks

        breakdown = {}
        total_costs = []

        for uni_name in ["大盘", "小盘"]:
            uni_holdings = holdings[holdings.get("universe", "") == uni_name]
            if len(uni_holdings) == 0:
                continue

            uni_costs = []
            uni_impacts = []
            for _, row in uni_holdings.iterrows():
                vol = float(row.get(volatility_col, 0.02))
                amount = float(row.get(amount_col, self.aum))

                impact = self.estimate_market_impact_bps(
                    trade_value_per_stock, amount, vol,
                    gamma=self._get_config(uni_name).impact_gamma,
                    eta=self._get_config(uni_name).impact_eta,
                )
                one_way = self.one_way_cost_bps(
                    uni_name, trade_value_per_stock, amount, vol,
                )

                uni_costs.append(one_way)
                uni_impacts.append(impact)
                total_costs.append(one_way)

            breakdown[uni_name] = {
                "n_stocks": len(uni_holdings),
                "avg_one_way_bps": np.mean(uni_costs) if uni_costs else 0,
                "avg_impact_bps": np.mean(uni_impacts) if uni_impacts else 0,
            }

        result = {
            "avg_one_way_bps": float(np.mean(total_costs)) if total_costs else 0.0,
            "n_stocks": n_stocks,
            "trade_value_per_stock": trade_value_per_stock,
            "breakdown": breakdown,
        }
        return result

    # ── AUM 敏感性分析 ────────────────────────────────────

    def sensitivity_analysis(
        self,
        daily_amount: float = 20_000_000,
        volatility: float = 0.025,
        gamma: float = 0.65,
        eta: float = 1.5,
        aums: Optional[list[float]] = None,
    ) -> pd.DataFrame:
        """
        AUM 敏感性分析: 固定股票特征, 变化 AUM, 看冲击成本如何增长。

        这对理解策略容量 (Capacity) 非常关键。
        一般而言, 当冲击成本超过策略预期 Alpha 的 30% 时,
        策略就达到了容量上限。

        参数
        ----
        daily_amount : float
            示例股票的日均成交额。
        volatility : float
            示例股票的日波动率。
        gamma : float
            冲击指数。
        eta : float
            缩放系数。
        aums : list[float] | None
            要测试的 AUM 列表。默认从 100 万到 10 亿。

        返回
        ----
        pd.DataFrame: AUM vs Impact(bps) 对照表。
        """
        if aums is None:
            aums = [
                1_000_000,    # 100万
                5_000_000,    # 500万
                10_000_000,   # 1000万
                20_000_000,   # 2000万
                50_000_000,   # 5000万
                100_000_000,  # 1亿
                200_000_000,  # 2亿
                500_000_000,  # 5亿
            ]

        rows = []
        for aum in aums:
            trade_value = aum / 60  # 假设 60 只等权持仓
            impact = self.estimate_market_impact_bps(
                trade_value, daily_amount, volatility, gamma, eta,
            )
            participation = trade_value / daily_amount * 100
            rows.append({
                "AUM(万)": f"{aum/1e4:.0f}",
                "单笔交易(万)": f"{trade_value/1e4:.1f}",
                "参与率(%)": f"{participation:.3f}",
                "冲击成本(bps)": f"{impact:.2f}",
                "冲击占1%Alpha": f"{impact/100:.1%}",
            })

        return pd.DataFrame(rows)

    # ── 辅助 ──────────────────────────────────────────────

    def _get_config(self, universe: str) -> UniverseCostConfig:
        """根据子域名称返回对应配置。"""
        if "小" in str(universe):
            return self.small_config
        return self.large_config


# ═══════════════════════════════════════════════════════════
# 3. 换手率计算与净收益
# ═══════════════════════════════════════════════════════════

def compute_turnover(
    prev_positions: set,
    curr_positions: set,
) -> float:
    """
    计算单边换手率。

    单边换手率 = 新进入的股票数 / 持仓总数
    双边换手率 = 单向换手率 × 2 (一只进+一只出 = 两次交易)

    参数
    ----
    prev_positions : set
        上期持仓股票集合 (symbol)。
    curr_positions : set
        当期目标持仓股票集合 (symbol)。

    返回
    ----
    float: 单边换手率 (0-1)。
    """
    if len(curr_positions) == 0:
        return 0.0

    n_positions = len(curr_positions)
    new_entries = len(curr_positions - prev_positions)
    # 单边: 卖出旧股和买入新股的数量应大致相等
    exits = len(prev_positions - curr_positions)
    avg_changes = (new_entries + exits) / 2

    return avg_changes / n_positions


def apply_transaction_costs(
    theoretical_returns: pd.Series,
    turnover: float,
    cost_bps: float,
) -> pd.Series:
    """
    从理论收益中扣除交易成本, 得到净收益。

    净收益 = 理论收益 - turnover × cost_bps / 10000

    参数
    ----
    theoretical_returns : pd.Series
        每期理论收益 (未扣成本)。
    turnover : float
        每期单边换手率 (0-1)。
    cost_bps : float
        单边交易成本 (bps)。

    返回
    ----
    pd.Series: 净收益序列。
    """
    cost_per_period = turnover * cost_bps / 10000
    net_returns = theoretical_returns - cost_per_period
    return net_returns


# ═══════════════════════════════════════════════════════════
# 4. 实用函数: 批量加载日线缓存
# ═══════════════════════════════════════════════════════════

def load_daily_cache(
    symbols: list[str],
    cache_dir: str = "data/raw",
    max_files: int = 300,
) -> dict[str, pd.DataFrame]:
    """
    批量加载日线缓存文件, 返回 {symbol: daily_df} 字典。

    日线数据用于 LiquidityFilter 的 Layer 2 (停牌) 和 Layer 3 (涨跌停)
    精确检测。如果只需要月频近似, 可以跳过此步骤。

    参数
    ----
    symbols : list[str]
        股票代码列表。
    cache_dir : str
        缓存目录。
    max_files : int
        最多加载文件数 (避免内存爆炸)。超过此数则只加载前 N 只。

    返回
    ----
    dict[str, pd.DataFrame]: {symbol: daily_df}
    """
    cache_path = Path(cache_dir)
    daily_data = {}
    loaded = 0

    for sym in symbols:
        if loaded >= max_files:
            break
        # 查找匹配的缓存文件
        pattern = f"daily_{sym}_*_qfq.csv"
        matches = list(cache_path.glob(pattern))
        # 优先取全量文件 (日期范围更宽)
        full_range = [m for m in matches if len(m.stem.split("_")) >= 4]
        if full_range:
            match = sorted(full_range, key=lambda x: len(x.name), reverse=True)[0]
        elif matches:
            match = matches[0]
        else:
            continue

        try:
            ddf = pd.read_csv(match)
            if len(ddf) > 0:
                daily_data[sym] = ddf
                loaded += 1
        except Exception:
            continue

    logger.info("加载日线缓存: %d/%d 只股票", loaded, len(symbols))
    return daily_data
