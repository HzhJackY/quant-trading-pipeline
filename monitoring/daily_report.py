"""
每日持仓监控看板 — Streamlit Dashboard for Top 30 Portfolio Monitoring.

在每日盘后（18:00 之后）运行，监控 Top 30 实盘组合的单日表现。
数据源: output/paper_trading_db/state.db (持仓) + baostock (行情)

用法:
    streamlit run monitoring/daily_report.py
    streamlit run monitoring/daily_report.py -- --db-path output/paper_trading_db/state.db
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# 确保项目根目录在 path 上（供 baostock adapter 等模块引用）
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import baostock as bs

# ═══════════════════════════════════════════════════════════
# 配置常量
# ═══════════════════════════════════════════════════════════

DEFAULT_DB_PATH = _project_root / "output" / "paper_trading_db" / "state.db"
BENCHMARK_BS_CODE = "sh.000905"   # Baostock 格式的中证500
BENCHMARK_SYMBOL = "000905"       # 内部 6 位代码
TOP_N = 30

# Streamlit 页面配置（必须在所有 st 调用之前）
st.set_page_config(
    page_title="Top 30 量化组合每日监控",
    page_icon="📊",
    layout="wide",
)


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def _to_bs_code(symbol: str) -> str:
    """将 6 位 A 股代码转为 Baostock 格式 (sh.600000 / sz.000001)。"""
    code = str(symbol).zfill(6)
    if code.startswith(("6", "9")):
        return f"sh.{code}"
    else:
        return f"sz.{code}"


# ═══════════════════════════════════════════════════════════
# 数据获取层 — SQLite 持仓
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def get_latest_positions(
    db_path: Path, top_n: int = TOP_N
) -> Tuple[Optional[str], pd.DataFrame]:
    """
    从 signal_anchor 表中获取最新调仓期的 Top N 持仓。

    逻辑：
      - 查询 signal_anchor 中最大的 ym（调仓月份）
      - 取出该月 alpha_signal 最高的 top_n 只股票
      - 返回持仓明细 DataFrame

    Args:
        db_path: state.db 文件的路径。
        top_n:  取前 N 只股票（默认 30）。

    Returns:
        (ym, positions_df) —
        - ym:          "YYYY-MM" 格式的调仓月份，无数据时为 None
        - positions_df: columns = [symbol, alpha_signal]，按 signal 降序排列
    """
    if not db_path.exists():
        return None, pd.DataFrame(columns=["symbol", "alpha_signal"])

    conn = sqlite3.connect(str(db_path))
    try:
        # ── 获取最新调仓月份 ──
        row = conn.execute("SELECT MAX(ym) FROM signal_anchor").fetchone()
        if not row or not row[0]:
            return None, pd.DataFrame(columns=["symbol", "alpha_signal"])

        latest_ym = row[0]

        # ── 获取该月的 Top N 持仓（按 alpha_signal 降序）──
        query = """
            SELECT symbol, alpha_signal
            FROM signal_anchor
            WHERE ym = ?
            ORDER BY alpha_signal DESC
            LIMIT ?
        """
        rows = conn.execute(query, (latest_ym, top_n)).fetchall()

        if not rows:
            return latest_ym, pd.DataFrame(columns=["symbol", "alpha_signal"])

        df = pd.DataFrame(rows, columns=["symbol", "alpha_signal"])
        return latest_ym, df
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════
# 数据获取层 — Baostock 交易日
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def get_latest_trade_dates() -> Tuple[Optional[str], Optional[str]]:
    """
    通过 Baostock 的 query_trade_dates() 获取最近两个交易日。

    回溯最近 10 个自然日，筛选出 is_trading_day == "1" 的最近 2 天，
    作为 T 日（最近交易日）和 T-1 日（前一个交易日）。

    Returns:
        (t_date, t_minus_1_date) — "YYYY-MM-DD" 格式；若获取失败则为 (None, None)

    Note:
        非交易日（周末/节假日）：Baostock 仍会返回交易日列表，但最近交易日
        将早于当日日期。UI 层会在 T 日期上显示真实的交易日期。
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=10)

    try:
        rs = bs.query_trade_dates(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        if rs.error_code != "0":
            return None, None

        trade_dates: list[str] = []
        while rs.next():
            row = rs.get_row_data()
            # row[0] = calendar_date, row[1] = is_trading_day ("0"/"1")
            is_trading = row[1].strip() == "1"
            d = row[0].strip()
            if is_trading and d <= end_date.strftime("%Y-%m-%d"):
                trade_dates.append(d)

        # 最新在前
        trade_dates.sort(reverse=True)

        if len(trade_dates) >= 2:
            return trade_dates[0], trade_dates[1]
        elif len(trade_dates) == 1:
            return trade_dates[0], None
        else:
            return None, None
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════════
# 数据获取层 — Baostock 行情
# ═══════════════════════════════════════════════════════════

def fetch_close_prices(symbols: list[str], target_date: str) -> pd.DataFrame:
    """
    逐只获取指定股票列表在 target_date 的收盘价（前复权）。

    Baostock 没有批量日线接口，逐只调用 query_history_k_data_plus。
    31 只标的 × 约 50ms/只 ≈ 1.5s，可接受。

    Args:
        symbols:  6 位代码列表（不含 sh./sz. 前缀）。
        target_date: "YYYY-MM-DD" 格式的交易日。

    Returns:
        pd.DataFrame with columns [symbol, close]。
        无数据的标的 close = NaN。
    """
    if not symbols:
        return pd.DataFrame(columns=["symbol", "close"])

    results: list[dict] = []
    for sym in symbols:
        try:
            bs_code = _to_bs_code(sym)
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,close",
                start_date=target_date,
                end_date=target_date,
                frequency="d",
                adjustflag="2",  # 前复权
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


def fetch_all_prices(
    symbols: list[str],
    t_date: str,
    t1_date: Optional[str],
    progress_placeholder,
) -> pd.DataFrame:
    """
    拉取所有标的两日收盘价的主控函数。

    分两次调用 fetch_close_prices（T 日 + T-1 日），合并为一张表。

    Args:
        symbols:    标的列表（含中证500代码）。
        t_date:     T 日日期 "YYYY-MM-DD"。
        t1_date:    T-1 日日期；None 时仅获取 T 日。
        progress_placeholder: st.empty() 占位符，用于显示进度。

    Returns:
        pd.DataFrame with columns [symbol, close_t, close_t_minus_1]。
    """
    # ── T 日收盘价 ──
    progress_placeholder.text(f"正在拉取 T={t_date} 收盘价（{len(symbols)} 只标的）...")
    t_prices = fetch_close_prices(symbols, t_date)
    t_prices = t_prices.rename(columns={"close": "close_t"})

    # ── T-1 日收盘价 ──
    if t1_date:
        progress_placeholder.text(f"正在拉取 T-1={t1_date} 收盘价（{len(symbols)} 只标的）...")
        t1_prices = fetch_close_prices(symbols, t1_date)
        t1_prices = t1_prices.rename(columns={"close": "close_t_minus_1"})
        result = t_prices.merge(t1_prices, on="symbol", how="left")
    else:
        result = t_prices.copy()
        result["close_t_minus_1"] = np.nan

    progress_placeholder.empty()
    return result


# ═══════════════════════════════════════════════════════════
# 计算层
# ═══════════════════════════════════════════════════════════

def compute_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每只标的的单日涨跌幅。

    daily_return = (close_t - close_t_minus_1) / close_t_minus_1

    Args:
        prices_df: [symbol, close_t, close_t_minus_1]

    Returns:
        同 schema + daily_return 列。无效数据 daily_return = NaN。
    """
    df = prices_df.copy()
    valid_mask = df["close_t_minus_1"].notna() & (df["close_t_minus_1"] > 0)
    df["daily_return"] = np.where(
        valid_mask,
        (df["close_t"] - df["close_t_minus_1"]) / df["close_t_minus_1"],
        np.nan,
    )
    return df


def compute_portfolio_metrics(
    returns_df: pd.DataFrame,
    benchmark_symbol: str = BENCHMARK_SYMBOL,
) -> dict:
    """
    计算组合级别的三个核心 KPI 指标。

    - 组合收益率 (Portfolio Return):  30 只持仓涨跌幅的等权平均
    - 基准收益率 (Benchmark Return):  中证500 的涨跌幅
    - 单日超额收益 (Daily Alpha):     组合收益率 - 基准收益率

    Args:
        returns_df:     [symbol, daily_return] 的 DataFrame
        benchmark_symbol: 基准指数 6 位代码

    Returns:
        {
            "portfolio_return":  float or NaN,
            "benchmark_return":  float or NaN,
            "daily_alpha":       float or NaN,
            "n_stocks":          int,   # 有效（有数据）持仓数
            "n_missing":         int,   # 无数据持仓数
            "n_positive":        int,   # 上涨家数
            "n_negative":        int,   # 下跌家数
        }
    """
    # ── 基准收益率 ──
    bench_mask = returns_df["symbol"] == benchmark_symbol
    bench_row = returns_df[bench_mask]
    benchmark_return = (
        bench_row["daily_return"].values[0]
        if len(bench_row) > 0 and pd.notna(bench_row["daily_return"].values[0])
        else np.nan
    )

    # ── 持仓收益率（排除基准行）──
    holdings = returns_df[~bench_mask].copy()
    valid = holdings["daily_return"].dropna()

    n_stocks = len(valid)
    n_total_holdings = len(holdings)
    n_missing = n_total_holdings - n_stocks
    n_positive = int((valid > 0).sum())
    n_negative = int((valid < 0).sum())

    portfolio_return = valid.mean() if n_stocks > 0 else np.nan

    # ── Alpha ──
    if (not np.isnan(portfolio_return)) and (not np.isnan(benchmark_return)):
        daily_alpha = portfolio_return - benchmark_return
    else:
        daily_alpha = np.nan

    return {
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
        "daily_alpha": daily_alpha,
        "n_stocks": n_stocks,
        "n_missing": n_missing,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


# ═══════════════════════════════════════════════════════════
# UI 渲染层 — KPI 卡片
# ═══════════════════════════════════════════════════════════

def _metric_arrow(val: float | None) -> str:
    """返回涨跌方向箭头。"""
    if val is None or np.isnan(val):
        return "➖"
    return "🟢" if val > 0 else "🔴" if val < 0 else "➖"


def render_kpi_cards(metrics: dict):
    """
    渲染顶部三列 KPI 指标卡片。

    使用 st.metric 组件，正值为绿色 (normal)，负值为红色 (inverse)。
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        val = metrics["portfolio_return"]
        if not np.isnan(val):
            st.metric(
                label=f"{_metric_arrow(val)} 组合收益率",
                value=f"{val * 100:+.2f}%",
                delta=f"基准: {metrics['benchmark_return'] * 100:+.2f}%"
                if not np.isnan(metrics.get("benchmark_return", np.nan))
                else None,
                delta_color="off",
            )
        else:
            st.metric(label="📈 组合收益率", value="N/A")

    with col2:
        val = metrics["benchmark_return"]
        if not np.isnan(val):
            st.metric(
                label=f"{_metric_arrow(val)} 中证500 收益率",
                value=f"{val * 100:+.2f}%",
            )
        else:
            st.metric(label="🏦 中证500 收益率", value="N/A")

    with col3:
        val = metrics["daily_alpha"]
        if not np.isnan(val):
            label = (
                "🌟 正 Alpha (超额)"
                if val > 0
                else "⚡ 负 Alpha"
                if val < 0
                else "➖ Alpha = 0"
            )
            st.metric(
                label=label,
                value=f"{val * 100:+.2f}%",
            )
        else:
            st.metric(label="🚀 今日 Alpha", value="N/A")

    # ── 数据质量提示 ──
    if metrics.get("n_missing", 0) > 0:
        st.info(
            f"⚠️ {metrics['n_missing']} 只股票在 T 日 / T-1 日无有效数据，"
            f"已排除在组合收益率计算之外。"
        )


# ═══════════════════════════════════════════════════════════
# UI 渲染层 — 涨跌排名摘要
# ═══════════════════════════════════════════════════════════

def render_market_breadth(metrics: dict):
    """渲染市场宽度摘要（上涨 vs 下跌家数）。"""
    n_pos = metrics.get("n_positive", 0)
    n_neg = metrics.get("n_negative", 0)
    n_valid = n_pos + n_neg
    if n_valid == 0:
        return

    pos_pct = n_pos / n_valid * 100
    neg_pct = n_neg / n_valid * 100

    cols = st.columns(8)
    cols[0].metric("上涨家数", f"{n_pos} 只")
    cols[1].metric("下跌家数", f"{n_neg} 只")
    cols[2].metric("上涨占比", f"{pos_pct:.0f}%")
    cols[3].metric("下跌占比", f"{neg_pct:.0f}%")


# ═══════════════════════════════════════════════════════════
# UI 渲染层 — 红黑榜（Top / Bottom 3）
# ═══════════════════════════════════════════════════════════

def render_top_bottom(returns_df: pd.DataFrame):
    """
    渲染红黑榜区域 — 涨幅前三 + 跌幅前三，配柱状图。
    """
    # 排除基准指数
    holdings = returns_df[returns_df["symbol"] != BENCHMARK_SYMBOL].copy()
    valid = holdings.dropna(subset=["daily_return"]).sort_values(
        "daily_return", ascending=False
    )

    col_left, col_right = st.columns(2)

    # ── 涨幅前 3 ──
    with col_left:
        st.markdown("### 🟢 涨幅 Top 3")
        top3 = valid.head(3)
        if len(top3) > 0:
            # 表格
            top3_display = top3[["symbol", "daily_return"]].copy()
            top3_display["daily_return"] = top3_display["daily_return"].apply(
                lambda x: f"{x * 100:+.2f}%"
            )
            top3_display.columns = ["股票代码", "涨跌幅"]
            st.dataframe(
                top3_display,
                hide_index=True,
                use_container_width=True,
            )
            # 柱状图
            chart_data = top3.set_index("symbol")[["daily_return"]] * 100
            chart_data.columns = ["涨跌幅 (%)"]
            st.bar_chart(chart_data, height=180, color="#22c55e")
        else:
            st.write("无有效数据")

    # ── 跌幅前 3 ──
    with col_right:
        st.markdown("### 🔴 跌幅 Top 3")
        bottom3 = valid.tail(3).sort_values("daily_return", ascending=True)
        if len(bottom3) > 0:
            bottom3_display = bottom3[["symbol", "daily_return"]].copy()
            bottom3_display["daily_return"] = bottom3_display["daily_return"].apply(
                lambda x: f"{x * 100:+.2f}%"
            )
            bottom3_display.columns = ["股票代码", "涨跌幅"]
            st.dataframe(
                bottom3_display,
                hide_index=True,
                use_container_width=True,
            )
            # 柱状图
            chart_data = bottom3.set_index("symbol")[["daily_return"]] * 100
            chart_data.columns = ["涨跌幅 (%)"]
            st.bar_chart(chart_data, height=180, color="#ef4444")
        else:
            st.write("无有效数据")


# ═══════════════════════════════════════════════════════════
# UI 渲染层 — 全景持仓明细表
# ═══════════════════════════════════════════════════════════

def render_full_table(returns_df: pd.DataFrame, positions_df: pd.DataFrame, ym: str):
    """
    渲染完整持仓明细 DataFrame，按涨跌幅降序排列。

    Args:
        returns_df:   合并了 alpha_signal 的收益率 DataFrame
        positions_df: 原始持仓 DataFrame（用于显示信号强度）
        ym:           调仓月份
    """
    st.markdown(f"### 📋 全景持仓明细 — 调仓期: `{ym}`")

    # 排除基准行
    holdings = returns_df[returns_df["symbol"] != BENCHMARK_SYMBOL].copy()

    if holdings.empty:
        st.write("无持仓数据")
        return

    # 按涨跌幅降序排列（NaN 沉底）
    holdings = holdings.sort_values(
        "daily_return", ascending=False, na_position="last"
    ).reset_index(drop=True)

    # 构造展示用 DataFrame
    display_df = holdings[["symbol", "close_t_minus_1", "close_t", "daily_return"]].copy()

    # 格式化
    def _fmt_price(x):
        return f"{x:.2f}" if pd.notna(x) else "N/A"

    def _fmt_return(x):
        return f"{x * 100:+.2f}%" if pd.notna(x) else "N/A"

    display_df["close_t_minus_1"] = display_df["close_t_minus_1"].apply(_fmt_price)
    display_df["close_t"] = display_df["close_t"].apply(_fmt_price)
    display_df["daily_return"] = display_df["daily_return"].apply(_fmt_return)

    display_df.columns = ["股票代码", "T-1 收盘价", "T 收盘价", "今日涨跌幅"]

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        height=850,
    )

    # ── 快速统计 ──
    n_total = len(holdings)
    n_valid = holdings["daily_return"].notna().sum()
    n_positive = int((holdings["daily_return"] > 0).sum())
    n_negative = int((holdings["daily_return"] < 0).sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总持仓", f"{n_total} 只")
    col2.metric("有效数据", f"{n_valid} 只")
    col3.metric(
        "上涨",
        f"{n_positive} 只",
        delta=f"占比 {n_positive / n_valid * 100:.0f}%" if n_valid > 0 else None,
        delta_color="normal",
    )
    col4.metric(
        "下跌",
        f"{n_negative} 只",
        delta=f"占比 {n_negative / n_valid * 100:.0f}%" if n_valid > 0 else None,
        delta_color="inverse",
    )


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def main():
    """Streamlit 每日监控看板主函数。"""

    # ── 解析命令行 ──
    db_path = DEFAULT_DB_PATH
    for i, arg in enumerate(sys.argv):
        if arg == "--db-path" and i + 1 < len(sys.argv):
            db_path = Path(sys.argv[i + 1])

    # ── 页面标题 ──
    st.title("📊 Top 30 量化组合每日监控看板")

    # ── 登录 Baostock ──
    login_placeholder = st.empty()
    try:
        lg = bs.login()
        if lg.error_code != "0":
            login_placeholder.error(
                f"❌ Baostock 登录失败: [{lg.error_code}] {lg.error_msg}"
            )
            st.stop()
    except Exception as e:
        login_placeholder.error(f"❌ Baostock 连接异常: {e}")
        st.stop()

    login_placeholder.success("⚡ Baostock 已连接")

    try:
        # ── Step 1: 获取最近交易日 ──
        t_date, t1_date = get_latest_trade_dates()
        if t_date is None:
            st.warning(
                "⚠️ 无法获取交易日列表，请检查 Baostock 服务是否正常。"
            )
            st.stop()

        # ── 显示交易日期 ──
        today_str = date.today().strftime("%Y-%m-%d")
        if t_date < today_str:
            st.info(
                f"📅 最近交易日为 **{t_date}**（今日 {today_str} 为非交易日），"
                f"以下数据基于最近交易日收盘价。"
            )
        else:
            st.info(f"📅 交易日期: **{t_date}**")

        # ── Step 2: 获取持仓 ──
        ym, positions = get_latest_positions(db_path, top_n=TOP_N)
        if positions.empty:
            st.warning(
                f"⚠️ 未在 `{db_path}` 中找到有效持仓数据。\n\n"
                "请确认：\n"
                "1. 月末调仓流水线已运行至少一次\n"
                "2. 数据库路径正确（可通过 `--db-path` 参数指定）"
            )
            st.stop()

        # ── Step 3: 拉取行情 ──
        progress_placeholder = st.empty()
        all_symbols = positions["symbol"].tolist() + [BENCHMARK_SYMBOL]

        prices_df = fetch_all_prices(
            symbols=all_symbols,
            t_date=t_date,
            t1_date=t1_date,
            progress_placeholder=progress_placeholder,
        )

        # ── 检查数据有效性 ──
        n_valid_close = prices_df["close_t"].notna().sum()
        if n_valid_close == 0:
            st.warning(
                f"🔕 **今日无交易数据**\n\n"
                f"T = {t_date} 的收盘价全部为空，可能原因：\n"
                f"- 当日为非交易日（周末或节假日）\n"
                f"- Baostock 数据尚未更新（请 18:00 后再试）"
            )
            st.stop()

        # ── Step 4: 计算收益率 ──
        returns_df = compute_returns(prices_df)
        metrics = compute_portfolio_metrics(returns_df)

        # ── 合并信号强度，供全景表使用 ──
        returns_df_merged = returns_df.merge(
            positions[["symbol", "alpha_signal"]],
            on="symbol",
            how="left",
        )

        # ═══════════════════════════════════════════════════
        # UI 渲染
        # ═══════════════════════════════════════════════════

        st.divider()

        # ── 日期 & 调仓信息汇总行 ──
        st.caption(
            f"T = {t_date}　|　T-1 = {t1_date or 'N/A'}　|　"
            f"调仓基准月: {ym}　|　"
            f"有效持仓: {metrics['n_stocks']}/{TOP_N} 只"
        )

        # ── KPI 卡片 ──
        render_kpi_cards(metrics)

        # ── 市场宽度 ──
        st.divider()
        st.markdown("### 📊 市场宽度 (Market Breadth)")
        render_market_breadth(metrics)

        # ── 红黑榜 ──
        st.divider()
        st.markdown("### 🏆 红黑榜 — 极端波动监控")
        render_top_bottom(returns_df_merged)

        # ── 全景持仓表 ──
        st.divider()
        render_full_table(returns_df_merged, positions, ym)

        # ── 页脚 ──
        st.divider()
        st.caption(
            f"数据来源: Baostock · 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
            f"数据库: `{db_path}`"
        )

    finally:
        # ── 确保 Baostock 登出 ──
        bs.logout()
        login_placeholder.empty()


if __name__ == "__main__":
    main()
