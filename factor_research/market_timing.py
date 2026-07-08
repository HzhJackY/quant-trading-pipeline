"""
Market Timing — Volatility Regime + Index MA.

择时信号 (敞口缩放, 与 Alpha 严格解耦):
  1. 趋势判断: CSI 500 MA20 跌破 MA60 (死叉) → 减仓
  2. 波动率判断: 20日年化波动率超过 252日 80% 分位数 (高波) → 减仓
  3. 任一条件满足 → 总目标仓位从 100% 压缩至 30%

设计原则:
  - Beta 风控与 Alpha 信号完全解耦: 不修改选股排名, 仅缩放最终敞口
  - 状态无关: 每期独立判断, 不引入记忆效应
  - 零摩擦实现: 回测中直接缩放组合收益; 实盘中缩放 target weights

Integration:
  - run_backtest_with_costs() → 回测中每期对 target weights 应用乘数
  - paper_trading_pipeline.py → 月末调仓时显示缩放后权重

Data source: baostock → sh.000905 (中证 500 指数, 前复权日线)
"""

from __future__ import annotations

import logging
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── 默认参数 ────────────────────────────────────────────────
INDEX_CODE = "000905.SH"       # 中证 500
MA_SHORT = 20                   # 短期均线
MA_LONG = 60                    # 长期均线
VOL_WINDOW = 20                 # 波动率计算窗口 (交易日)
VOL_HIST_WINDOW = 252           # 波动率历史分位窗口
VOL_PERCENTILE = 0.80           # 波动率阈值分位
TRIGGER_MULTIPLIER = 0.3        # 触发风控时的仓位乘数
NORMAL_MULTIPLIER = 1.0         # 正常仓位乘数

# 本地缓存
_CACHE_DIR = Path("output")
_CACHE_FILE = "csi500_daily.parquet"


# ═══════════════════════════════════════════════════════════════
# 数据获取
# ═══════════════════════════════════════════════════════════════

def fetch_csi500(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    获取中证 500 日线 (前复权 close), 带本地 parquet 缓存。

    Parameters
    ----------
    start_date : str, optional
        数据起始日 (YYYY-MM-DD)。默认向前 700 个交易日。
    end_date : str, optional
        数据截止日 (YYYY-MM-DD)。默认今天。
    use_cache : bool
        是否使用/更新本地 parquet 缓存。

    Returns
    -------
    pd.DataFrame
        列: date (datetime64), close (float64)。
        按 date 升序排列。
    """
    cache_path = _CACHE_DIR / _CACHE_FILE

    # ── 尝试从缓存加载 ──
    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached_min = cached["date"].min()
        cached_max = cached["date"].max()
        today = pd.Timestamp(date.today())

        req_start = pd.Timestamp(start_date) if start_date else cached_min
        req_end = pd.Timestamp(end_date) if end_date else today

        # 缓存覆盖需求区间且在 5 天内
        if cached_min <= req_start and cached_max >= req_end:
            logger.info("  CSI500: 从缓存加载 (%d rows)", len(cached))
            return cached.sort_values("date").reset_index(drop=True)

    # ── 从 baostock 获取 ──
    try:
        df = _fetch_from_baostock(start_date, end_date)
        if use_cache:
            _CACHE_DIR.mkdir(exist_ok=True)
            df.to_parquet(cache_path, index=False)
            logger.info("  CSI500: 已缓存到 %s", _CACHE_FILE)
        return df
    except Exception as e:
        if use_cache and cache_path.exists():
            logger.warning("  CSI500: baostock 获取失败 (%s), 回退到缓存", e)
            return (
                pd.read_parquet(cache_path)
                .sort_values("date")
                .reset_index(drop=True)
            )
        raise


def _fetch_from_baostock(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """从 baostock 查询中证 500 日线 (前复权)。"""
    import baostock as bs

    end = (pd.Timestamp(end_date) if end_date else date.today()).strftime("%Y-%m-%d")
    start = (pd.Timestamp(start_date) if start_date else
             date.today() - timedelta(days=700)).strftime("%Y-%m-%d")

    lg = bs.login()
    if lg.error_code != "0":
        raise ConnectionError(f"baostock login failed: [{lg.error_code}] {lg.error_msg}")

    try:
        rs = bs.query_history_k_data_plus(
            "sh.000905",
            "date,close",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="2",  # 前复权
        )
        if rs.error_code != "0":
            raise RuntimeError(
                f"baostock query_history_k_data_plus failed: "
                f"[{rs.error_code}] {rs.error_msg}"
            )

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
    finally:
        bs.logout()

    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    if len(df) == 0:
        logger.warning("  CSI500: baostock 返回空数据 (区间 %s ~ %s)", start, end)
    else:
        logger.info("  CSI500: baostock 返回 %d 行 (%s ~ %s)",
                     len(df), df["date"].min().date(), df["date"].max().date())

    return df


# ═══════════════════════════════════════════════════════════════
# 特征计算
# ═══════════════════════════════════════════════════════════════

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加 MA20, MA60, 20 日年化波动率等衍生列。"""
    d = df.copy().sort_values("date")
    d["MA20"] = d["close"].rolling(MA_SHORT).mean()
    d["MA60"] = d["close"].rolling(MA_LONG).mean()
    d["daily_ret"] = d["close"].pct_change()
    d["vol_20d"] = d["daily_ret"].rolling(VOL_WINDOW).std() * np.sqrt(252)
    return d


# ═══════════════════════════════════════════════════════════════
# 核心 API
# ═══════════════════════════════════════════════════════════════

def compute_market_multiplier(
    index_df: pd.DataFrame,
    current_date: date | str | pd.Timestamp,
) -> float:
    """
    计算给定日期的仓位乘数。

    逻辑:
      1. 找到 <= current_date 的最近交易日
      2. 检查 MA20 < MA60 (死叉)
      3. 检查 20 日年化波动率 > 252 日 80% 分位
      4. 任一成立 → 0.3, 否则 → 1.0

    Parameters
    ----------
    index_df : pd.DataFrame
        中证 500 日线 (含 date, close 两列)。
    current_date : date | str | pd.Timestamp
        评价日期 (月末调仓日)。

    Returns
    -------
    float: 0.3 (触发风控) 或 1.0 (正常满仓)。
    """
    dt = _normalize_date(current_date)
    hist = index_df[index_df["date"].dt.date <= dt].copy()

    if len(hist) < MA_LONG + 5:
        logger.warning("  [择时] 数据不足 (%d 行, 需要 %d+), 返回 multiplier=1.0",
                       len(hist), MA_LONG + 5)
        return NORMAL_MULTIPLIER

    hist = _add_features(hist)
    latest = hist.iloc[-1]

    # ── 条件 1: 死叉 ──
    ma20 = latest["MA20"]
    ma60 = latest["MA60"]
    death_cross = pd.notna(ma20) and pd.notna(ma60) and ma20 < ma60

    # ── 条件 2: 波动率飙高 ──
    vol_series = hist["vol_20d"].dropna()
    current_vol = vol_series.iloc[-1]
    vol_spike = False
    if len(vol_series) >= VOL_HIST_WINDOW:
        hist_vol = vol_series.iloc[-VOL_HIST_WINDOW:]
        vol_threshold = hist_vol.quantile(VOL_PERCENTILE)
        vol_spike = pd.notna(current_vol) and current_vol > vol_threshold

    triggered = death_cross or vol_spike

    # ── 日志 ──
    if triggered:
        flags = []
        if death_cross:
            flags.append(f"死叉 MA20={ma20:.2f} < MA60={ma60:.2f}")
        if vol_spike:
            flags.append(
                f"高波 vol_20d={current_vol:.2%} > pct80={vol_threshold:.2%}"
            )
        logger.info("  [择时] 🚨 TRIGGERED → multiplier=%.1f | %s",
                     TRIGGER_MULTIPLIER, "; ".join(flags))
    else:
        logger.info(
            "  [择时] ✓ Normal → multiplier=%.1f | MA20=%.2f MA60=%.2f | "
            "vol_20d=%.2f%%",
            NORMAL_MULTIPLIER,
            ma20 if pd.notna(ma20) else 0,
            ma60 if pd.notna(ma60) else 0,
            current_vol * 100 if pd.notna(current_vol) else 0,
        )

    return TRIGGER_MULTIPLIER if triggered else NORMAL_MULTIPLIER


def prepare_timing_multipliers(
    index_df: pd.DataFrame,
    dates: list[pd.Timestamp | date],
) -> dict[pd.Timestamp, float]:
    """
    批量预计算择时乘数 (用于回测场景, 避免重复 fetch)。

    一次性计算所有日期的乘数, 返回 {date: multiplier} 字典。

    Parameters
    ----------
    index_df : pd.DataFrame
        中证 500 日线。
    dates : list[date | Timestamp]
        需要计算的日期列表 (如每个月的最后一个交易日)。

    Returns
    -------
    dict[pd.Timestamp, float]
        日期 → 乘数映射。
    """
    df = _add_features(index_df)

    # 预计算波动率阈值序列
    vol_series = df["vol_20d"].dropna()
    vol_threshold_series = vol_series.rolling(
        VOL_HIST_WINDOW, min_periods=VOL_HIST_WINDOW
    ).quantile(VOL_PERCENTILE)

    result: dict[pd.Timestamp, float] = {}

    for dt in sorted(dates):
        dt_ts = pd.Timestamp(dt).normalize()
        hist = df[df["date"] <= dt_ts]
        if len(hist) < MA_LONG + 5:
            result[dt_ts] = NORMAL_MULTIPLIER
            continue

        latest = hist.iloc[-1]

        # 死叉
        ma20 = latest["MA20"]
        ma60 = latest["MA60"]
        death_cross = pd.notna(ma20) and pd.notna(ma60) and ma20 < ma60

        # 高波
        vol_spike = False
        if dt_ts in vol_threshold_series.index:
            current_v = vol_series.loc[:dt_ts].iloc[-1] if dt_ts in vol_series.index else None
            threshold = vol_threshold_series.loc[dt_ts]
            if current_v is not None and pd.notna(threshold):
                vol_spike = current_v > threshold

        result[dt_ts] = TRIGGER_MULTIPLIER if (death_cross or vol_spike) else NORMAL_MULTIPLIER

    # 统计信息
    n_trig = sum(1 for v in result.values() if v < 1.0)
    n_total = len(result)
    if n_total > 0:
        logger.info(
            "[择时] 批量预计算: %d/%d 期触发减仓 (%.1f%%)",
            n_trig, n_total, 100 * n_trig / n_total,
        )

    return result


def apply_position_sizing(
    weights: pd.Series,
    multiplier: float,
) -> pd.Series:
    """
    对目标权重应用仓位缩放乘数。

    示例: weights = [1/30, 1/30, ...], multiplier = 0.3
          → [1/30*0.3, 1/30*0.3, ...]  (总仓位 30%, 剩余为现金)
    """
    return weights * multiplier


# ═══════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════

def _normalize_date(dt: date | str | pd.Timestamp) -> date:
    """统一日期格式为 date。"""
    if isinstance(dt, str):
        return datetime.strptime(str(dt)[:10], "%Y-%m-%d").date()
    if isinstance(dt, pd.Timestamp):
        return dt.date()
    return dt


def plot_timing_history(index_df: pd.DataFrame) -> None:
    """
    绘制择时历史全景图。

    三面板:
      - 上: 指数 + MA20/MA60 (标记死叉区间)
      - 中: 20日年化波动率 + 80% 分位线 (标记高波区间)
      - 下: 仓位乘数步进图

    Parameters
    ----------
    index_df : pd.DataFrame
        中证 500 日线 (date, close 列即可)。
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    df = _add_features(index_df)
    if len(df) < VOL_HIST_WINDOW + 20:
        print(f"[择时] 数据不足: 需要 {VOL_HIST_WINDOW + 20} 行, 实际 {len(df)}")
        return

    # ── 统一对齐: 剔除头部 NaN, 确保所有序列 index 长度一致 ──
    # MA60 需要前 60 行, vol_20d 需要前 20 行, vol_pct80 需要前 272 行
    plot_df = df.dropna(subset=["MA60", "vol_20d"]).copy()
    plot_df = plot_df.iloc[VOL_HIST_WINDOW - 1:]  # 留足 252 行做分位计算
    plot_df = plot_df.reset_index(drop=True)

    if len(plot_df) < 10:
        print(f"[择时] 对齐后数据不足: {len(plot_df)} 行")
        return

    # 逐日乘数 (用于绘图)
    vol_series = plot_df["vol_20d"]
    vol_threshold_series = vol_series.rolling(
        VOL_HIST_WINDOW, min_periods=VOL_HIST_WINDOW
    ).quantile(VOL_PERCENTILE)

    death_cross = plot_df["MA20"] < plot_df["MA60"]
    vol_spike = plot_df["vol_20d"] > vol_threshold_series
    triggered = death_cross.fillna(False) | vol_spike.fillna(False)

    # 乘数序列 (前 VOL_HIST_WINDOW-1 行 vol_threshold 为 NaN → 都用 1.0)
    multipliers = np.where(
        triggered.fillna(False),
        TRIGGER_MULTIPLIER,
        NORMAL_MULTIPLIER,
    )
    plot_dates = plot_df["date"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # ── 上: 指数 + MA ──
    ax1.plot(plot_df["date"], plot_df["close"], label="CSI 500", linewidth=1, alpha=0.5)
    ax1.plot(plot_df["date"], plot_df["MA20"], label="MA20", linewidth=1.2)
    ax1.plot(plot_df["date"], plot_df["MA60"], label="MA60", linewidth=1.2)
    # 用淡红色竖条标记死叉区间
    for i in range(len(plot_df)):
        if death_cross.iloc[i] and (i == 0 or not death_cross.iloc[i - 1]):
            ax1.axvspan(
                plot_df["date"].iloc[i],
                plot_df["date"].iloc[min(i + 30, len(plot_df) - 1)],
                color="red", alpha=0.05,
            )
    ax1.legend(loc="upper left")
    ax1.set_title("中证 500 — 均线死叉 (MA20 < MA60)")
    ax1.grid(True, alpha=0.3)

    # ── 中: 波动率 ──
    ax2.plot(plot_df["date"], plot_df["vol_20d"], label="20日年化波动率",
             linewidth=1, color="purple", alpha=0.6)
    ax2.plot(plot_df["date"], vol_threshold_series,
             label=f"{VOL_PERCENTILE:.0%} 分位线",
             linewidth=1, color="orange", linestyle="--")
    ax2.fill_between(
        plot_df["date"][vol_spike.fillna(False)],
        0, plot_df["vol_20d"][vol_spike.fillna(False)],
        color="orange", alpha=0.1,
    )
    ax2.legend(loc="upper left")
    ax2.set_title("波动率 — 20日年化 vs 252日80%分位")
    ax2.grid(True, alpha=0.3)

    # ── 下: 乘数 ──
    ax3.step(plot_dates, multipliers, where="post",
             color="green", linewidth=2, label="仓位乘数")
    ax3.set_ylim(0, 1.3)
    ax3.axhline(y=0.3, color="red", linestyle="--", alpha=0.5,
                label=f"减仓线 ({TRIGGER_MULTIPLIER})")
    ax3.axhline(y=1.0, color="green", linestyle="--", alpha=0.5,
                label=f"满仓线 ({NORMAL_MULTIPLIER})")
    ax3.legend(loc="upper left")
    ax3.set_title(
        f"仓位乘数 (触发率: {triggered.fillna(False).sum()}/{len(multipliers)}"
        f" = {triggered.fillna(False).mean():.1%})"
    )
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Date")

    plt.tight_layout()
    plt.show()


def timing_summary(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    生成择时信号汇总表。

    对每个交易日输出:
      - date, close, MA20, MA60, 死叉标记, vol_20d, 80% 阈值, 高波标记, 乘数

    Returns
    -------
    pd.DataFrame
    """
    df = _add_features(index_df)
    if len(df) < VOL_HIST_WINDOW + 20:
        print(f"[择时] 数据不足")
        return pd.DataFrame()

    vol_series = df["vol_20d"].dropna()
    vol_threshold_series = vol_series.rolling(
        VOL_HIST_WINDOW, min_periods=VOL_HIST_WINDOW
    ).quantile(VOL_PERCENTILE)

    result = df[["date", "close", "MA20", "MA60"]].copy()
    result["死叉"] = result["MA20"] < result["MA60"]
    result["vol_20d"] = df["vol_20d"]
    result["vol_pct80"] = vol_threshold_series
    result["高波"] = result["vol_20d"] > result["vol_pct80"]
    result["乘数"] = np.where(
        result["死叉"].fillna(False) | result["高波"].fillna(False),
        TRIGGER_MULTIPLIER,
        NORMAL_MULTIPLIER,
    )
    return result.reset_index(drop=True)
