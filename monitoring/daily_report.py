"""
每日持仓监控看板 v2 — 机构级风险与收益归因 Dashboard.

在每日盘后（18:00 之后）运行，监控 Top 30 实盘组合的表现、风格暴露与风险事件。

三个核心增强模块:
  1. 历史超额走势图  — 持有期累计净值 (组合 / 基准 / Alpha)
  2. 风格因子暴露度  — Size, Momentum, Value, Volatility 横截面暴露
  3. 异常风控雷达    — 暴跌 (-7%) / ST / 停牌 风险扫描

数据源:
  - output/paper_trading_db/state.db  — 持仓 (signal_anchor) + 日线缓存 (market_cache)
  - output/paper_trading_db/fundamentals_*.parquet — 基本面 (市值/估值/行业)
  - baostock — 实时行情 + 基准历史

用法:
    streamlit run monitoring/daily_report.py
    streamlit run monitoring/daily_report.py -- --db-path output/paper_trading_db/state.db
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# 确保项目根目录在 path 上
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import baostock as bs

# ═══════════════════════════════════════════════════════════
# 全局配置
# ═══════════════════════════════════════════════════════════

DEFAULT_DB_PATH = _project_root / "output" / "paper_trading_db" / "state.db"
DEFAULT_DB_DIR  = _project_root / "output" / "paper_trading_db"
BENCHMARK_BS_CODE = "sh.000905"
BENCHMARK_SYMBOL = "000905"
TOP_N = 30
MARKET_DATA_READY_TIME = time(17, 0)

# 风控阈值
DRAWDOWN_ALERT_THRESHOLD = -0.07   # 单日跌幅超过 -7% 触发预警
LARGE_CAP_MCAP_THRESHOLD = 2e10    # 200 亿 — 大盘/小盘分界参考

st.set_page_config(
    page_title="Top 30 量化组合 · 风控看板",
    page_icon="📊",
    layout="wide",
)

# ── 自定义 CSS（微调视觉）──
st.markdown("""
<style>
    /* 指标卡片轻微放大 */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
    /* 风险警告区域 */
    .risk-alert {
        border-left: 4px solid #ef4444;
        padding-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def _to_bs_code(symbol: str) -> str:
    """6 位 A 股代码 → Baostock 格式 (sh.600000 / sz.000001)."""
    code = str(symbol).zfill(6)
    if code.startswith(("6", "9")):
        return f"sh.{code}"
    return f"sz.{code}"


def _find_latest_parquet(prefix: str, db_dir: Path = DEFAULT_DB_DIR) -> Optional[Path]:
    """查找 db_dir 下最新的匹配前缀的 parquet 文件。"""
    candidates = sorted(db_dir.glob(f"{prefix}*.parquet"), reverse=True)
    return candidates[0] if candidates else None


# ═══════════════════════════════════════════════════════════
# Section 1 — SQLite 数据层
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def load_positions(db_path: Path, top_n: int = TOP_N) -> Tuple[Optional[str], pd.DataFrame]:
    """
    从 signal_anchor 获取最新调仓期的 Top N 持仓.

    Returns:
        (ym, df) — ym: "YYYY-MM", df: [symbol, alpha_signal] 按 signal 降序.
    """
    if not db_path.exists():
        return None, pd.DataFrame(columns=["symbol", "alpha_signal"])

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT MAX(ym) FROM signal_anchor").fetchone()
        if not row or not row[0]:
            return None, pd.DataFrame(columns=["symbol", "alpha_signal"])
        latest_ym = row[0]
        rows = conn.execute(
            "SELECT symbol, alpha_signal FROM signal_anchor WHERE ym=? "
            "ORDER BY alpha_signal DESC LIMIT ?",
            (latest_ym, top_n),
        ).fetchall()
        df = pd.DataFrame(rows, columns=["symbol", "alpha_signal"]) if rows else pd.DataFrame(columns=["symbol", "alpha_signal"])
        return latest_ym, df
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False)
def load_market_cache_history(
    db_path: Path, symbols: list[str], start_date: str, end_date: str,
) -> pd.DataFrame:
    """
    从 market_cache 中查询指定标的在 [start_date, end_date] 区间的日线收盘价.

    Args:
        symbols: 6 位代码列表。传空列表 = 取全市场。

    Returns:
        pd.DataFrame [trade_date, symbol, close] 已排序.
    """
    if not db_path.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(str(db_path))
    try:
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            query = f"""
                SELECT trade_date, symbol, close
                FROM market_cache
                WHERE trade_date >= ? AND trade_date <= ?
                  AND symbol IN ({placeholders})
                ORDER BY trade_date, symbol
            """
            params = [start_date, end_date] + list(symbols)
        else:
            query = """
                SELECT trade_date, symbol, close
                FROM market_cache
                WHERE trade_date >= ? AND trade_date <= ?
                ORDER BY trade_date, symbol
            """
            params = [start_date, end_date]
        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df
    finally:
        conn.close()


@st.cache_data(ttl=300, show_spinner=False)
def load_fundamentals(db_dir: Path = DEFAULT_DB_DIR) -> pd.DataFrame:
    """
    加载最新的基本面快照（市值 / 估值 / 行业分类）.

    Returns:
        pd.DataFrame 至少包含 [symbol, name, total_mcap, float_mcap, pb, pe_ttm, board].
    """
    fund_path = _find_latest_parquet("fundamentals_", db_dir)
    if fund_path is None:
        return pd.DataFrame()

    df = pd.read_parquet(fund_path)

    # 类型修正: 部分列可能以 object 存储
    for col in ["float_mcap", "pb", "bps", "revenue", "operating_profit", "gross_margin", "pe_static"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ═══════════════════════════════════════════════════════════
# Section 2 — Baostock 数据层
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def get_trade_dates(lookback_days: int = 10) -> Tuple[Optional[str], Optional[str]]:
    """获取最近两个交易日 (T, T-1). Falls back to local cache if baostock fails."""
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    try:
        rs = bs.query_trade_dates(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        if rs.error_code != "0":
            raise RuntimeError(f"baostock error [{rs.error_code}]")
        trade_dates = []
        while rs.next():
            row = rs.get_row_data()
            if row[1].strip() == "1" and row[0].strip() <= end_date.strftime("%Y-%m-%d"):
                trade_dates.append(row[0].strip())
        trade_dates.sort(reverse=True)
        if len(trade_dates) >= 2:
            return trade_dates[0], trade_dates[1]
        elif len(trade_dates) == 1:
            return trade_dates[0], None
        return None, None
    except Exception:
        return _get_trade_dates_from_cache(lookback_days)


def _get_trade_dates_from_cache(lookback_days: int = 10) -> Tuple[Optional[str], Optional[str]]:
    """Fallback: read latest trade dates from local SQLite market_cache."""
    try:
        import sqlite3
        db_path = _find_db_path()
        if not db_path or not db_path.exists():
            return None, None
        conn = sqlite3.connect(str(db_path))
        dates = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM market_cache ORDER BY trade_date DESC LIMIT ?",
            conn, params=(lookback_days,),
        )
        conn.close()
        if len(dates) >= 2:
            return str(dates.iloc[0, 0]), str(dates.iloc[1, 0])
        elif len(dates) == 1:
            return str(dates.iloc[0, 0]), None
        return None, None
    except Exception:
        return None, None


@st.cache_data(ttl=300, show_spinner=False)
def get_cached_trade_dates(db_path: Path, limit: int = 20) -> list[str]:
    """Read recent market_cache trade dates without touching large market data."""
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        dates = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM market_cache ORDER BY trade_date DESC LIMIT ?",
            conn,
            params=(limit,),
        )
        conn.close()
        return [str(v)[:10] for v in dates["trade_date"].tolist()]
    except Exception:
        return []


def resolve_display_trade_dates(
    db_path: Path,
    t_date: Optional[str],
    t1_date: Optional[str],
    now: Optional[datetime] = None,
) -> tuple[Optional[str], Optional[str], str]:
    """
    Choose the date shown by the dashboard.

    Before 17:00 on a trading day, today's close is usually unavailable. In that
    window the dashboard should show the latest cached session instead of trying
    to fetch today's incomplete data and stopping with an empty screen.
    """
    if t_date is None:
        return None, None, "no_trade_date"

    now = now or datetime.now()
    today_str = now.date().strftime("%Y-%m-%d")
    cached_dates = get_cached_trade_dates(db_path)

    if t_date == today_str and now.time() < MARKET_DATA_READY_TIME:
        prior_cached = [d for d in cached_dates if d < today_str]
        if prior_cached:
            display_t = prior_cached[0]
            display_t1 = prior_cached[1] if len(prior_cached) > 1 else None
            return display_t, display_t1, "before_close_use_latest_cached"
        if t1_date:
            prior_to_t1 = [d for d in cached_dates if d < t1_date]
            display_t1 = prior_to_t1[0] if prior_to_t1 else None
            return t1_date, display_t1, "before_close_fallback_to_previous_trade_date"

    return t_date, t1_date, "use_resolved_trade_date"


def _find_db_path() -> Optional[Path]:
    """Find state.db in default or configured locations."""
    for candidate in [
        DEFAULT_DB_PATH,
        _project_root / "output" / "paper_trading_db" / "state.db",
    ]:
        if candidate.exists():
            return candidate
    return None


def fetch_point_close(symbols: list[str], target_date: str) -> pd.DataFrame:
    """
    逐只获取单日收盘价（前复权）.

    Args:
        symbols: 6 位代码列表.  target_date: "YYYY-MM-DD".

    Returns:
        [symbol, close] — 无数据为 NaN.
    """
    if not symbols:
        return pd.DataFrame(columns=["symbol", "close"])
    results = []
    for sym in symbols:
        try:
            bs_code = _to_bs_code(sym)
            rs = bs.query_history_k_data_plus(
                bs_code, "date,close",
                start_date=target_date, end_date=target_date,
                frequency="d", adjustflag="2",
            )
            if rs.error_code != "0":
                results.append({"symbol": sym, "close": np.nan})
                continue
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rows and rows[0][1]:
                results.append({"symbol": sym, "close": float(rows[0][1])})
            else:
                results.append({"symbol": sym, "close": np.nan})
        except Exception:
            results.append({"symbol": sym, "close": np.nan})
    return pd.DataFrame(results)


def _fetch_close_from_cache(symbols: list[str], target_date: str) -> pd.DataFrame:
    """Fallback: read close prices from local SQLite market_cache."""
    try:
        import sqlite3
        db_path = _find_db_path()
        if not db_path or not db_path.exists():
            return pd.DataFrame(columns=["symbol", "close"])
        conn = sqlite3.connect(str(db_path))
        placeholders = ",".join("?" for _ in symbols)
        df = pd.read_sql_query(
            f"SELECT symbol, close FROM market_cache WHERE trade_date = ? AND symbol IN ({placeholders})",
            conn, params=[target_date] + list(symbols),
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame(columns=["symbol", "close"])


@st.cache_data(ttl=600, show_spinner=False)
def fetch_benchmark_history(start_date: str, end_date: str) -> pd.DataFrame:
    """
    拉取中证500 (sh.000905) 在 [start_date, end_date] 区间的每日收盘价（前复权）.

    Returns:
        [trade_date, close].
    """
    try:
        rs = bs.query_history_k_data_plus(
            BENCHMARK_BS_CODE, "date,close",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2",
        )
        if rs.error_code != "0":
            return pd.DataFrame()
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["date", "close"])
        df["trade_date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["symbol"] = BENCHMARK_SYMBOL
        return df[["trade_date", "symbol", "close"]].dropna(subset=["close"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_names(symbols: list[str]) -> dict[str, str]:
    """
    批量获取股票名称（用于 ST 检测）.

    Baostock query_stock_basic 逐只调用; 30 只约 1.5s.
    同时尝试从 fundamentals 缓存补充.

    Returns:
        {symbol: name} 映射; 获取失败的 symbol 值为 "UNKNOWN".
    """
    # 优先从 fundamentals 缓存获取
    fund = load_fundamentals()
    name_map: dict[str, str] = {}
    if not fund.empty and "name" in fund.columns:
        for _, row in fund.iterrows():
            name_map[str(row["symbol"]).zfill(6)] = str(row["name"])

    # 补充 baostock（只获取未命中缓存的）
    missing = [s for s in symbols if s not in name_map]
    if missing:
        for sym in missing:
            try:
                rs = bs.query_stock_basic(code=_to_bs_code(sym))
                if rs.error_code == "0":
                    while rs.next():
                        row_data = rs.get_row_data()
                        if row_data and len(row_data) >= 2:
                            name_map[sym] = row_data[1].strip()
                            break
            except Exception:
                pass
        # 仍未获取到的标记为 UNKNOWN
        for sym in missing:
            if sym not in name_map:
                name_map[sym] = "UNKNOWN"

    return name_map


# ═══════════════════════════════════════════════════════════
# Section 3 — 计算层: 单日收益 + 组合指标
# ═══════════════════════════════════════════════════════════

def compute_daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """daily_return = (close_t - close_t_minus_1) / close_t_minus_1."""
    df = prices_df.copy()
    mask = df["close_t_minus_1"].notna() & (df["close_t_minus_1"] > 0)
    df["daily_return"] = np.where(
        mask,
        (df["close_t"] - df["close_t_minus_1"]) / df["close_t_minus_1"],
        np.nan,
    )
    return df


def compute_metrics(returns_df: pd.DataFrame, bm_symbol: str = BENCHMARK_SYMBOL) -> dict:
    """计算单日组合/基准/Alpha KPI."""
    bm_row = returns_df[returns_df["symbol"] == bm_symbol]
    bm_ret = float(bm_row["daily_return"].values[0]) if len(bm_row) > 0 and pd.notna(bm_row["daily_return"].values[0]) else np.nan

    holdings = returns_df[returns_df["symbol"] != bm_symbol].copy()
    valid = holdings["daily_return"].dropna()

    pf_ret = valid.mean() if len(valid) > 0 else np.nan
    alpha = (pf_ret - bm_ret) if (not np.isnan(pf_ret) and not np.isnan(bm_ret)) else np.nan

    return {
        "portfolio_return": pf_ret,
        "benchmark_return": bm_ret,
        "daily_alpha": alpha,
        "n_stocks": len(valid),
        "n_missing": len(holdings) - len(valid),
        "n_positive": int((valid > 0).sum()),
        "n_negative": int((valid < 0).sum()),
    }


# ═══════════════════════════════════════════════════════════
# Section 4 — 计算层: 累计净值历史 (Historical NAV)
# ═══════════════════════════════════════════════════════════

def compute_cumulative_nav(
    db_path: Path,
    symbols: list[str],
    ym: str,
    t_date: str,
) -> Optional[pd.DataFrame]:
    """
    构建持有期累计净值曲线 (组合 / 基准 / Alpha).

    数据来源:
      - 持仓股票: SQLite market_cache (日线缓存)
      - 基准指数: Baostock 日线 (从月初到 T)

    Args:
        db_path: state.db 路径.
        symbols: Top 30 持仓代码列表.
        ym:      调仓月份 "YYYY-MM".
        t_date:  T 日日期.

    Returns:
        pd.DataFrame [trade_date, portfolio_nav, benchmark_nav, cumulative_alpha]
        或 None（数据不足）.
    """
    # ── 确定回看起始日: ym 月份第一个交易日 ──
    start_date = f"{ym}-01"

    # ── 基准历史 ──
    bench_hist = fetch_benchmark_history(start_date, t_date)
    if bench_hist.empty:
        return None

    # ── 持仓历史 ──
    market_hist = load_market_cache_history(db_path, symbols, start_date, t_date)
    if market_hist.empty:
        return None

    # ── 计算每日等权组合收益 ──
    # 每只股票: daily_return = close / close.shift(1) - 1
    market_hist = market_hist.sort_values(["symbol", "trade_date"])
    market_hist["prev_close"] = market_hist.groupby("symbol")["close"].shift(1)
    market_hist["daily_return"] = np.where(
        market_hist["prev_close"].notna() & (market_hist["prev_close"] > 0),
        market_hist["close"] / market_hist["prev_close"] - 1,
        np.nan,
    )
    # 等权平均
    daily_pf = market_hist.groupby("trade_date")["daily_return"].mean().reset_index()
    daily_pf.columns = ["trade_date", "pf_return"]

    # ── 基准每日收益 ──
    bench_hist = bench_hist.sort_values("trade_date")
    bench_hist["prev_close"] = bench_hist["close"].shift(1)
    bench_hist["bm_return"] = np.where(
        bench_hist["prev_close"].notna() & (bench_hist["prev_close"] > 0),
        bench_hist["close"] / bench_hist["prev_close"] - 1,
        0.0,
    )

    # ── 合并 ──
    merged = daily_pf.merge(
        bench_hist[["trade_date", "bm_return"]],
        on="trade_date", how="inner"
    ).sort_values("trade_date")

    if len(merged) < 2:
        return None

    # ── 累计净值 ──
    merged["portfolio_nav"] = (1 + merged["pf_return"]).cumprod()
    merged["benchmark_nav"]  = (1 + merged["bm_return"]).cumprod()
    merged["cumulative_alpha"] = merged["portfolio_nav"] / merged["benchmark_nav"] - 1

    return merged[["trade_date", "portfolio_nav", "benchmark_nav", "cumulative_alpha"]]


# ═══════════════════════════════════════════════════════════
# Section 5 — 计算层: 风格因子暴露度
# ═══════════════════════════════════════════════════════════

def compute_factor_exposures(
    symbols: list[str],
    db_path: Path,
    t_date: str,
) -> dict:
    """
    计算 Top 30 持仓在核心因子上的横截面暴露.

    因子:
      - Size (市值):       total_mcap 百分位均值 (高 = 大盘)
      - Momentum (动量):   持有期内收益百分位均值 (高 = 动量)
      - Value (价值):      EP (1/PE) 百分位均值 — PB 数据不全时的代理 (高 = 价值)
      - Volatility (波):   可用日收益的波动率百分位均值 (高 = 高波动)

    所有因子均做横截面排序 → 百分位 (0-100), 50 = 中性.

    Args:
        symbols: Top 30 持仓列表.
        db_path: state.db 路径.
        t_date:  T 日日期.

    Returns:
        {
            "Size":        {"value": 65.2, "label": "...", "note": "..."},
            ...
        }
        因子标签含中文风格判定.
    """
    exposures = {}
    t_dt = pd.Timestamp(t_date)
    fund = load_fundamentals()

    # ── Size (市值): 用 total_mcap (float_mcap 在当前数据源中全为空) ──
    if not fund.empty and "total_mcap" in fund.columns:
        fund_clean = fund.dropna(subset=["total_mcap"]).copy()
        if len(fund_clean) >= 100:
            fund_clean["mcap_pct"] = fund_clean["total_mcap"].rank(pct=True) * 100
            fund_clean["sym6"] = fund_clean["symbol"].str.zfill(6)
            held = fund_clean[fund_clean["sym6"].isin(symbols)]
            if not held.empty:
                val = round(held["mcap_pct"].mean(), 1)
                exposures["Size"] = {
                    "value": val,
                    "label": _size_label(val),
                    "note": f"基于 total_mcap ({len(fund_clean)} 只全市场)",
                }

    # ── Value (EP = 1/PE): PB 数据源当前不可用, 用 EP 代理 ──
    if not fund.empty and "pe_ttm" in fund.columns:
        fund_val = fund.dropna(subset=["pe_ttm"]).copy()
        fund_val = fund_val[fund_val["pe_ttm"] > 0]  # PE < 0 无意义
        if len(fund_val) >= 100:
            # EP (Earnings Yield) = 1/PE → 高 EP = 价值股
            fund_val["ep"] = 1.0 / fund_val["pe_ttm"]
            fund_val["value_pct"] = fund_val["ep"].rank(pct=True) * 100
            fund_val["sym6"] = fund_val["symbol"].str.zfill(6)
            held_val = fund_val[fund_val["sym6"].isin(symbols)]
            if not held_val.empty:
                val = round(held_val["value_pct"].mean(), 1)
                exposures["Value"] = {
                    "value": val,
                    "label": _value_label(val),
                    "note": f"EP (1/PE) 代理, {len(fund_val)} 只有效",
                }

    # ── Momentum: 持有期内区间收益（market_cache 可能不足 21 天）─
    lookback_start = (t_dt - pd.DateOffset(days=35)).strftime("%Y-%m-%d")
    market = load_market_cache_history(db_path, [], lookback_start, t_date)
    if not market.empty:
        market = market.sort_values(["symbol", "trade_date"])
        agg = market.groupby("symbol").agg(
            close_start=("close", "first"),
            close_end=("close", "last"),
            n_days=("trade_date", "nunique"),
        ).reset_index()
        # 至少需要 3 个交易日才有意义
        agg = agg[agg["n_days"] >= 3].copy()
        agg["period_return"] = np.where(
            agg["close_start"] > 0,
            agg["close_end"] / agg["close_start"] - 1,
            np.nan,
        )
        agg = agg.dropna(subset=["period_return"])
        if len(agg) >= 100:
            agg["mom_pct"] = agg["period_return"].rank(pct=True) * 100
            held_mom = agg[agg["symbol"].isin(symbols)]
            if not held_mom.empty:
                val = round(held_mom["mom_pct"].mean(), 1)
                max_days = int(agg["n_days"].max())
                exposures["Momentum"] = {
                    "value": val,
                    "label": _mom_label(val),
                    "note": f"区间收益 ({max_days} 交易日, "
                            f"{lookback_start} ~ {t_date})",
                }

    # ── Volatility: 可用交易日内的日收益波动率 (年化) ──
    lookback_vol = (t_dt - pd.DateOffset(days=40)).strftime("%Y-%m-%d")
    market_vol = load_market_cache_history(db_path, [], lookback_vol, t_date)
    if not market_vol.empty:
        market_vol = market_vol.sort_values(["symbol", "trade_date"])
        def _vol(grp):
            if len(grp) < 3:
                return pd.Series({"vol_d": np.nan, "n_days": 0})
            rets = grp.sort_values("trade_date")["close"].pct_change().dropna()
            if len(rets) < 2:
                return pd.Series({"vol_d": np.nan, "n_days": 0})
            return pd.Series({
                "vol_d": rets.std() * np.sqrt(252),  # 年化
                "n_days": len(grp),
            })
        vols = market_vol.groupby("symbol").apply(_vol).reset_index()
        vols = vols.dropna(subset=["vol_d"])
        if len(vols) >= 100:
            vols["vol_pct"] = vols["vol_d"].rank(pct=True) * 100
            held_vol = vols[vols["symbol"].isin(symbols)]
            if not held_vol.empty:
                val = round(held_vol["vol_pct"].mean(), 1)
                max_d = int(vols["n_days"].max())
                exposures["Volatility"] = {
                    "value": val,
                    "label": _vol_label(val),
                    "note": f"基于 {max_d} 交易日收益序列",
                }

    return exposures


def _size_label(pct: float) -> str:
    if pct >= 70:  return "大盘偏重 🔵"
    elif pct >= 55: return "略偏大盘"
    elif pct >= 45: return "均衡 ⚖️"
    elif pct >= 30: return "略偏小盘"
    else:           return "小盘偏重 🟠"


def _mom_label(pct: float) -> str:
    if pct >= 70:  return "强动量 🚀"
    elif pct >= 55: return "轻度动量"
    elif pct >= 45: return "中性"
    elif pct >= 30: return "轻度反转"
    else:           return "强反转 🔄"


def _value_label(pct: float) -> str:
    if pct >= 70:  return "深度价值 💎"
    elif pct >= 55: return "略偏价值"
    elif pct >= 45: return "均衡"
    elif pct >= 30: return "略偏成长"
    else:           return "成长偏重 🌱"


def _vol_label(pct: float) -> str:
    if pct >= 70:  return "高波动 ⚡"
    elif pct >= 55: return "略偏高波动"
    elif pct >= 45: return "中性波动"
    elif pct >= 30: return "略偏低波动"
    else:           return "低波动 🛡️"


# ═══════════════════════════════════════════════════════════
# Section 6 — 计算层: 风控雷达
# ═══════════════════════════════════════════════════════════

def scan_risk_events(
    returns_df: pd.DataFrame,
    name_map: dict[str, str],
    positions_df: pd.DataFrame,
) -> list[dict]:
    """
    扫描 Top 30 组合的风险事件.

    扫描项:
      1. 暴跌: 单日涨跌幅 < -7%
      2. ST 标记: 股票名称含 "ST" 或 "*ST"
      3. 疑似停牌: T-1 有数据但 T 日无收盘价

    Returns:
        [{symbol, name, risk_type, detail, severity}] 列表.
    """
    alerts: list[dict] = []

    holdings = returns_df[returns_df["symbol"] != BENCHMARK_SYMBOL].copy()

    for _, row in holdings.iterrows():
        sym = row["symbol"]
        name = name_map.get(sym, "UNKNOWN")
        ret = row.get("daily_return", np.nan)
        close_t = row.get("close_t", np.nan)
        close_t1 = row.get("close_t_minus_1", np.nan)

        # 1. 暴跌检测
        if pd.notna(ret) and ret < DRAWDOWN_ALERT_THRESHOLD:
            alerts.append({
                "symbol": sym,
                "name": name,
                "risk_type": "📉 单日暴跌",
                "detail": f"跌幅 {ret*100:.1f}%（超过 -7% 阈值）",
                "severity": "error",
            })

        # 2. ST 检测
        if "ST" in name or "*ST" in name:
            alerts.append({
                "symbol": sym,
                "name": name,
                "risk_type": "⚠️ ST 警示",
                "detail": f"股票名称为 {name}，存在退市风险警示",
                "severity": "error",
            })

        # 3. 疑似停牌: T-1 有数据, T 日无
        if pd.notna(close_t1) and pd.isna(close_t) and close_t1 > 0:
            alerts.append({
                "symbol": sym,
                "name": name,
                "risk_type": "🔒 疑似停牌",
                "detail": f"T-1 收盘 {close_t1:.2f}，T 日无交易数据",
                "severity": "warning",
            })

    return alerts


# ═══════════════════════════════════════════════════════════
# Section 7 — UI: 顶部风险雷达
# ═══════════════════════════════════════════════════════════

def render_risk_radar(alerts: list[dict]):
    """在页面最上方渲染醒目的高风险提示."""
    if not alerts:
        st.success("✅ 风控扫描通过 — 无暴跌、无 ST、无停牌异常")
        return

    errors   = [a for a in alerts if a["severity"] == "error"]
    warnings = [a for a in alerts if a["severity"] == "warning"]

    if errors:
        st.error(f"🚨 **高危风险警报 — {len(errors)} 项异常**")
        for a in errors:
            st.markdown(
                f"> 📛 **{a['risk_type']}** | `{a['symbol']}` {a['name']} | {a['detail']}"
            )

    if warnings:
        st.warning(f"⚠️ **风控提示 — {len(warnings)} 项需关注**")
        for a in warnings:
            st.markdown(
                f"> 🔸 **{a['risk_type']}** | `{a['symbol']}` {a['name']} | {a['detail']}"
            )


# ═══════════════════════════════════════════════════════════
# Section 8 — UI: KPI 卡片
# ═══════════════════════════════════════════════════════════

def _arrow(val: float | None) -> str:
    if val is None or np.isnan(val): return "➖"
    return "🟢" if val > 0 else "🔴" if val < 0 else "➖"


def render_kpi_cards(metrics: dict):
    """三列 KPI: 组合收益 / 基准收益 / Alpha."""
    c1, c2, c3 = st.columns(3)

    with c1:
        v = metrics["portfolio_return"]
        st.metric(
            label=f"{_arrow(v)} 组合日收益",
            value=f"{v*100:+.2f}%" if not np.isnan(v) else "N/A",
            delta=f"基准 {metrics['benchmark_return']*100:+.2f}%"
            if not np.isnan(metrics.get("benchmark_return", np.nan)) else None,
            delta_color="off",
        )
    with c2:
        v = metrics["benchmark_return"]
        st.metric(
            label=f"{_arrow(v)} 中证500",
            value=f"{v*100:+.2f}%" if not np.isnan(v) else "N/A",
        )
    with c3:
        v = metrics["daily_alpha"]
        label = ("🌟 超额 Alpha" if v > 0 else "⚡ 负 Alpha" if v < 0 else "Alpha = 0") if not np.isnan(v) else "Alpha N/A"
        st.metric(
            label=label,
            value=f"{v*100:+.2f}%" if not np.isnan(v) else "N/A",
        )

    if metrics.get("n_missing", 0) > 0:
        st.caption(f"⚠️ {metrics['n_missing']} 只无有效数据，已排除")


# ═══════════════════════════════════════════════════════════
# Section 9 — UI: 历史超额走势图
# ═══════════════════════════════════════════════════════════

def render_historical_nav(nav_df: pd.DataFrame, ym: str):
    """累计净值三线图: 组合 / 基准 / 累计超额."""
    st.markdown(f"### 📈 持有期累计净值 — 自 {ym} 调仓以来")

    if nav_df is None or nav_df.empty:
        st.info("历史数据不足（需至少 2 个交易日），累计净值将在后续交易日自动生成。")
        return

    # 准备 chart data
    chart_df = nav_df.set_index("trade_date")[["portfolio_nav", "benchmark_nav"]].copy()
    chart_df.columns = ["Top 30 组合", "中证500"]

    # 摘要指标
    last_row = nav_df.iloc[-1]
    pf_nav   = last_row["portfolio_nav"]
    bm_nav   = last_row["benchmark_nav"]
    alpha_cum = last_row["cumulative_alpha"]

    cols = st.columns(4)
    cols[0].metric("组合累计净值", f"{pf_nav:.4f}")
    cols[1].metric("基准累计净值", f"{bm_nav:.4f}")
    cols[2].metric(
        "累计超额收益",
        f"{alpha_cum*100:+.2f}%",
        delta=f"{alpha_cum*100:+.2f}%" if alpha_cum != 0 else None,
        delta_color="normal" if alpha_cum > 0 else "inverse",
    )
    cols[3].metric("持有交易日", f"{len(nav_df)} 天")

    # 三线图
    st.line_chart(chart_df, height=350, use_container_width=True)

    # 累计 Alpha 独立图
    alpha_chart = nav_df.set_index("trade_date")[["cumulative_alpha"]].copy()
    alpha_chart.columns = ["累计超额 Alpha"]
    st.line_chart(alpha_chart, height=200, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# Section 10 — UI: 风格因子暴露度
# ═══════════════════════════════════════════════════════════

def render_factor_exposure(exposures: dict):
    """横向柱状图展示因子暴露度，50 为中性参考线."""
    st.markdown("### 🧬 组合风格因子暴露度 (横截面百分位)")

    if not exposures:
        st.info("因子暴露数据暂不可用（需基本面 + 行情缓存支持）。数据将随交易日积累自动补充。")
        return

    # 准备表数据
    records = []
    missing_factors = []
    all_factors = ["Size", "Value", "Momentum", "Volatility"]
    for factor in all_factors:
        if factor in exposures:
            info = exposures[factor]
            records.append({
                "因子": factor,
                "暴露度 (百分位)": info["value"],
                "偏离": info["value"] - 50,
                "风格判断": info.get("label", ""),
            })
        else:
            missing_factors.append(factor)

    if not records:
        st.info("当前无任何因子暴露数据可用。")
        return

    exp_df = pd.DataFrame(records).set_index("因子")

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        chart_data = exp_df[["暴露度 (百分位)"]].copy()
        chart_data["中性基准 (50)"] = 50
        st.bar_chart(
            chart_data,
            height=280,
            use_container_width=True,
        )

    with col_table:
        st.dataframe(
            exp_df[["暴露度 (百分位)", "风格判断"]],
            hide_index=False,
            use_container_width=True,
            column_config={
                "暴露度 (百分位)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        # 显示每个因子的计算说明
        for factor in all_factors:
            if factor in exposures and "note" in exposures[factor]:
                st.caption(f"📝 *{factor}*: {exposures[factor]['note']}")

    # 文字解读
    high_factors = [(k, v) for k, v in exposures.items() if v["value"] >= 60]
    low_factors  = [(k, v) for k, v in exposures.items() if v["value"] <= 40]

    if high_factors or low_factors:
        parts = []
        for f, info in high_factors:
            parts.append(f"**{f}** 偏重 ({info['value']:.0f} 分位)")
        for f, info in low_factors:
            parts.append(f"**{f}** 偏低 ({info['value']:.0f} 分位)")
        st.caption("📌 风格漂移提示: " + "；".join(parts) if parts else "风格中性")
    else:
        st.caption("✅ 风格暴露接近中性，无明显漂移")

    if missing_factors:
        st.caption(
            f"⏳ 待补充因子: {', '.join(missing_factors)} "
            f"— 数据积累中（需更多交易日）"
        )


# ═══════════════════════════════════════════════════════════
# Section 11 — UI: 红黑榜 + 全景表
# ═══════════════════════════════════════════════════════════

def render_top_bottom(returns_df: pd.DataFrame):
    """涨幅 Top 3 / 跌幅 Top 3 + 柱状图."""
    holdings = returns_df[returns_df["symbol"] != BENCHMARK_SYMBOL].copy()
    valid = holdings.dropna(subset=["daily_return"]).sort_values("daily_return", ascending=False)

    left, right = st.columns(2)
    with left:
        st.markdown("#### 🟢 涨幅 Top 3")
        top3 = valid.head(3)
        if len(top3) > 0:
            disp = top3[["symbol", "daily_return"]].copy()
            disp["daily_return"] = disp["daily_return"].apply(lambda x: f"{x*100:+.2f}%")
            disp.columns = ["代码", "涨跌幅"]
            st.dataframe(disp, hide_index=True, use_container_width=True)
            chart = top3.set_index("symbol")[["daily_return"]] * 100
            st.bar_chart(chart, height=160, color="#22c55e")
        else:
            st.write("—")

    with right:
        st.markdown("#### 🔴 跌幅 Top 3")
        bottom3 = valid.tail(3).sort_values("daily_return", ascending=True)
        if len(bottom3) > 0:
            disp = bottom3[["symbol", "daily_return"]].copy()
            disp["daily_return"] = disp["daily_return"].apply(lambda x: f"{x*100:+.2f}%")
            disp.columns = ["代码", "涨跌幅"]
            st.dataframe(disp, hide_index=True, use_container_width=True)
            chart = bottom3.set_index("symbol")[["daily_return"]] * 100
            st.bar_chart(chart, height=160, color="#ef4444")
        else:
            st.write("—")


def render_full_table(returns_df: pd.DataFrame, ym: str):
    """全景持仓明细表，按涨跌幅降序排列."""
    st.markdown(f"### 📋 全景持仓明细 — 调仓期 `{ym}`")

    holdings = returns_df[returns_df["symbol"] != BENCHMARK_SYMBOL].copy()
    if holdings.empty:
        st.write("无数据")
        return

    holdings = holdings.sort_values("daily_return", ascending=False, na_position="last")

    disp = holdings[["symbol", "close_t_minus_1", "close_t", "daily_return"]].copy()
    disp["close_t_minus_1"] = disp["close_t_minus_1"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    disp["close_t"]         = disp["close_t"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    disp["daily_return"]    = disp["daily_return"].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "N/A")
    disp.columns = ["代码", "T-1 收盘", "T 收盘", "涨跌幅"]

    st.dataframe(disp, hide_index=True, use_container_width=True, height=850)

    n_total = len(holdings)
    n_valid = holdings["daily_return"].notna().sum()
    n_pos   = int((holdings["daily_return"] > 0).sum())
    n_neg   = int((holdings["daily_return"] < 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总持仓", f"{n_total} 只")
    c2.metric("有效数据", f"{n_valid} 只")
    c3.metric("上涨", f"{n_pos} 只", delta=f"{n_pos/max(n_valid,1)*100:.0f}%" if n_valid else None, delta_color="normal")
    c4.metric("下跌", f"{n_neg} 只", delta=f"{n_neg/max(n_valid,1)*100:.0f}%" if n_valid else None, delta_color="inverse")

    # 胜率条
    if n_valid > 0:
        win_rate = n_pos / n_valid
        st.progress(win_rate, text=f"日胜率: {win_rate*100:.0f}% ({n_pos}/{n_valid})")


# ═══════════════════════════════════════════════════════════
# Section 12 — 市场宽度
# ═══════════════════════════════════════════════════════════

def render_market_breadth(metrics: dict):
    """上涨 vs 下跌 vs 平盘的简洁摘要."""
    n_pos = metrics.get("n_positive", 0)
    n_neg = metrics.get("n_negative", 0)
    n_valid = n_pos + n_neg
    if n_valid == 0:
        return
    cols = st.columns(4)
    cols[0].metric("🟢 上涨", f"{n_pos} 只")
    cols[1].metric("🔴 下跌", f"{n_neg} 只")
    cols[2].metric("📊 上涨占比", f"{n_pos/max(n_valid,1)*100:.0f}%")
    cols[3].metric("📉 下跌占比", f"{n_neg/max(n_valid,1)*100:.0f}%")


# ═══════════════════════════════════════════════════════════
# Section 13 — 主入口
# ═══════════════════════════════════════════════════════════

def main():
    # ── CLI 参数 ──
    db_path = DEFAULT_DB_PATH
    for i, arg in enumerate(sys.argv):
        if arg == "--db-path" and i + 1 < len(sys.argv):
            db_path = Path(sys.argv[i + 1])

    st.title("📊 Top 30 量化组合 · 每日风控看板")

    # ── Baostock 登录 (soft dependency) ──
    login_ph = st.empty()
    use_baostock = False
    try:
        lg = bs.login()
        if lg.error_code == "0":
            use_baostock = True
            login_ph.success("⚡ Baostock 已连接")
        else:
            login_ph.warning(
                f"⚠️ Baostock 不可用 [{lg.error_code}] — "
                f"使用本地缓存数据（行情/基准可能延迟）"
            )
    except Exception:
        login_ph.warning("⚠️ Baostock 连接失败 — 使用本地缓存数据")

    try:
        # ────────────────────────────────────────────────
        # Step 1: 交易日
        # ────────────────────────────────────────────────
        t_date, t1_date = get_trade_dates()
        if t_date is None:
            st.warning("⚠️ 无法获取交易日（Baostock 不可用且本地缓存为空）")
            st.stop()

        today_str = date.today().strftime("%Y-%m-%d")
        raw_t_date, raw_t1_date = t_date, t1_date
        t_date, t1_date, date_source = resolve_display_trade_dates(db_path, raw_t_date, raw_t1_date)
        if t_date is None:
            st.warning("⚠️ 无法确定可展示交易日（本地行情缓存为空）")
            st.stop()

        if date_source.startswith("before_close"):
            st.info(
                f"📅 当前早于 {MARKET_DATA_READY_TIME.strftime('%H:%M')}，"
                f"暂不拉取今日 {today_str} 行情；展示最近已缓存交易日 **{t_date}**"
            )
        elif t_date < today_str:
            st.info(f"📅 最近交易日 **{t_date}**（今日 {today_str} 非交易日或数据未更新）")
        else:
            st.info(f"📅 交易日期 **{t_date}**")

        # ────────────────────────────────────────────────
        # Step 2: 持仓
        # ────────────────────────────────────────────────
        ym, positions = load_positions(db_path, TOP_N)
        if positions.empty:
            st.warning(f"⚠️ 无持仓数据 (`{db_path}`)"); st.stop()

        # ────────────────────────────────────────────────
        # Step 3: 行情数据 (baostock primary, local cache fallback)
        # ────────────────────────────────────────────────
        progress_ph = st.empty()
        use_live_prices = use_baostock and not date_source.startswith("before_close")
        all_symbols = positions["symbol"].tolist() + (
            [BENCHMARK_SYMBOL] if use_live_prices else []
        )

        if use_live_prices:
            progress_ph.text(f"⏳ 正在通过 Baostock 拉取 T={t_date} 收盘价（{len(all_symbols)} 只标的）...")
            t_prices = fetch_point_close(all_symbols, t_date)
            t_prices = t_prices.rename(columns={"close": "close_t"})

            if t1_date:
                progress_ph.text(f"⏳ 正在拉取 T-1={t1_date} 收盘价...")
                t1_prices = fetch_point_close(all_symbols, t1_date)
                t1_prices = t1_prices.rename(columns={"close": "close_t_minus_1"})
                prices_df = t_prices.merge(t1_prices, on="symbol", how="left")
            else:
                prices_df = t_prices.copy()
                prices_df["close_t_minus_1"] = np.nan
        else:
            # ── Local cache fallback ──
            progress_ph.text(f"⏳ 正在从本地缓存读取 T={t_date} 收盘价...")
            t_prices = _fetch_close_from_cache(all_symbols, t_date)
            if "close" in t_prices.columns:
                t_prices = t_prices.rename(columns={"close": "close_t"})
            else:
                t_prices["close_t"] = np.nan

            if t1_date:
                t1_prices = _fetch_close_from_cache(all_symbols, t1_date)
                if "close" in t1_prices.columns:
                    t1_prices = t1_prices.rename(columns={"close": "close_t_minus_1"})
                else:
                    t1_prices["close_t_minus_1"] = np.nan
                prices_df = t_prices.merge(t1_prices, on="symbol", how="left")
            else:
                prices_df = t_prices.copy()
                prices_df["close_t_minus_1"] = np.nan
        progress_ph.empty()

        if prices_df["close_t"].notna().sum() == 0:
            cached_dates = get_cached_trade_dates(db_path)
            fallback_dates = [d for d in cached_dates if d < t_date]
            if fallback_dates:
                fallback_t = fallback_dates[0]
                fallback_t1 = fallback_dates[1] if len(fallback_dates) > 1 else None
                st.info(f"🔁 T={t_date} 暂无行情，自动回退到最近已缓存交易日 **{fallback_t}**")
                t_date, t1_date = fallback_t, fallback_t1
                date_source = "fallback_latest_cached_after_missing_t"
                use_live_prices = False
                progress_ph.text(f"⏳ 正在从本地缓存读取 T={t_date} 收盘价...")
                t_prices = _fetch_close_from_cache(all_symbols, t_date)
                if "close" in t_prices.columns:
                    t_prices = t_prices.rename(columns={"close": "close_t"})
                else:
                    t_prices["close_t"] = np.nan
                if t1_date:
                    t1_prices = _fetch_close_from_cache(all_symbols, t1_date)
                    if "close" in t1_prices.columns:
                        t1_prices = t1_prices.rename(columns={"close": "close_t_minus_1"})
                    else:
                        t1_prices["close_t_minus_1"] = np.nan
                    prices_df = t_prices.merge(t1_prices, on="symbol", how="left")
                else:
                    prices_df = t_prices.copy()
                    prices_df["close_t_minus_1"] = np.nan
                progress_ph.empty()
            if prices_df["close_t"].notna().sum() == 0:
                st.warning(f"🔕 T={t_date} 无交易数据（非交易日或数据未更新）")
                st.stop()

        # ────────────────────────────────────────────────
        # Step 4: 并行计算（在数据就绪后同时进行）
        # ────────────────────────────────────────────────
        # 4a. 单日收益 + KPI
        returns_df = compute_daily_returns(prices_df)
        metrics = compute_metrics(returns_df)

        # 4b. 历史净值
        with st.spinner("正在构建累计净值曲线..."):
            nav_df = compute_cumulative_nav(db_path, positions["symbol"].tolist(), ym, t_date)

        # 4c. 因子暴露
        with st.spinner("正在计算风格因子暴露度..."):
            exposures = compute_factor_exposures(positions["symbol"].tolist(), db_path, t_date)

        # 4d. 风控扫描
        name_map = fetch_stock_names(positions["symbol"].tolist())
        alerts = scan_risk_events(returns_df, name_map, positions)

        # 4e. 合并信号
        returns_df_merged = returns_df.merge(
            positions[["symbol", "alpha_signal"]], on="symbol", how="left",
        )

        # ════════════════════════════════════════════
        # UI 渲染管线
        # ════════════════════════════════════════════

        # ── 汇总行 ──
        st.caption(
            f"T={t_date} ｜ T-1={t1_date or 'N/A'} ｜ "
            f"调仓月 {ym} ｜ 有效持仓 {metrics['n_stocks']}/{TOP_N} ｜ 日期模式 {date_source}"
        )

        # 🔴 风控雷达（置顶）
        render_risk_radar(alerts)
        st.divider()

        # 📊 KPI 卡片
        render_kpi_cards(metrics)

        # 📈 历史净值图
        st.divider()
        render_historical_nav(nav_df, ym)

        # 🧬 风格因子暴露
        st.divider()
        render_factor_exposure(exposures)

        # 📊 市场宽度
        st.divider()
        st.markdown("### 📊 市场宽度")
        render_market_breadth(metrics)

        # 🏆 红黑榜
        st.divider()
        st.markdown("### 🏆 红黑榜 — 极端波动监控")
        render_top_bottom(returns_df_merged)

        # 📋 全景持仓
        st.divider()
        render_full_table(returns_df_merged, ym)

        # 页脚
        st.divider()
        st.caption(
            f"数据来源: {'Baostock' if use_live_prices else '本地缓存'} + SQLite Market Cache · "
            f"更新时间 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
            f"`{db_path}`"
        )

    finally:
        if use_baostock:
            try:
                bs.logout()
            except Exception:
                pass
        login_ph.empty()


if __name__ == "__main__":
    main()
