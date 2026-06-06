"""
因子研究报告可视化。

输出四张核心图表:
1. IC 时序图 → 因子预测力随时间的稳定性
2. IC 分布图 → 因子预测力的分布特征
3. 分层净值曲线 → 每组能赚多少钱, 是否单调
4. 因子相关性热力图 → 哪些因子在说同一件事
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ─── 中文字体设置 ───────────────────────────────────────
# 尝试多个常见中文字体, 找到第一个可用的
for font in ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]:
    try:
        matplotlib.font_manager.findfont(font, fallback_to_default=False)
        matplotlib.rcParams["font.sans-serif"] = [font]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


def plot_ic_timeseries(ic_series: pd.Series, title: str = "Rank IC", figsize=(14, 5)):
    """
    IC 时序图。

    上半部分: 每期 IC 的柱状图 (红色=负, 绿色=正, 虚线=均值)
    下半部分: 累计 IC (如果因子有效, 应该稳步上升)
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1]},
    )

    vals = ic_series.dropna()
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in vals.values]

    ax1.bar(range(len(vals)), vals.values, color=colors, width=0.8)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.axhline(y=vals.mean(), color="#2980b9", linestyle="--", linewidth=1.5,
                label=f"均值 IC = {vals.mean():.4f}")
    ax1.set_ylabel("Rank IC", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")

    cum_ic = vals.cumsum()
    ax2.fill_between(range(len(cum_ic)), 0, cum_ic.values, color="#2980b9", alpha=0.3)
    ax2.plot(range(len(cum_ic)), cum_ic.values, color="#2980b9", linewidth=1.5)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_ylabel("累计 IC", fontsize=12)
    ax2.set_xlabel("期数", fontsize=12)

    plt.tight_layout()
    return fig


def plot_ic_distribution(ic_series: pd.Series, title: str = "IC 分布", figsize=(10, 5)):
    """IC 分布直方图: 看 IC 是否集中在正区间。"""
    fig, ax = plt.subplots(figsize=figsize)
    vals = ic_series.dropna()
    ax.hist(vals.values, bins=min(20, len(vals) // 2), color="#2980b9",
            edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="#c0392b", linestyle="--", linewidth=1.5, label="IC = 0")
    ax.axvline(x=vals.mean(), color="#27ae60", linestyle="-", linewidth=2,
               label=f"均值 = {vals.mean():.4f}")
    ax.set_xlabel("Rank IC", fontsize=12)
    ax.set_ylabel("频数", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_group_nav(
    group_returns: pd.DataFrame,
    title: str = "分层回测净值",
    long_group: int = 5,
    short_group: int = 1,
    figsize=(14, 5),
):
    """
    分组累计净值曲线。

    期望看到: Q5(深绿,最上面) > Q4 > Q3 > Q2 > Q1(深红,最下面)
    理想情况下多空组合(黑色虚线)稳步上升。
    """
    pivot = group_returns.pivot_table(index="date", columns="group", values="return")
    if pivot.empty:
        return None

    nav = (1 + pivot).cumprod()

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#c0392b", "#e74c3c", "#f39c12", "#2ecc71", "#27ae60"]
    groups = sorted(nav.columns)

    for i, g in enumerate(groups):
        if g in nav.columns:
            color_idx = min(i, len(colors) - 1)
            ax.plot(nav.index, nav[g], label=f"Q{g}", color=colors[color_idx],
                    linewidth=1.5)

    if long_group in nav.columns and short_group in nav.columns:
        ls_ret = pivot[long_group] - pivot[short_group]
        ls_nav = (1 + ls_ret).cumprod()
        ax.plot(ls_nav.index, ls_nav, label="Long-Short (Q5-Q1)",
                color="black", linewidth=2, linestyle="--")

    ax.set_ylabel("累计净值", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    return fig


def plot_factor_correlation(
    df: pd.DataFrame,
    factor_cols: list[str],
    figsize=(12, 10),
):
    """
    因子相关性热力图。

    量化的核心原则之一: 低相关的因子组合在一起才有意义。
    如果三个因子的 pairwise 相关都 >0.7, 它们本质上是同一个东西,
    你给它们 3 倍权重只增加了风险, 没有增加 alpha。
    """
    corr = df[factor_cols].corr()
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True,
        mask=mask, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("因子截面相关性矩阵", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def factor_summary_table(
    ic_results: dict[str, dict],
) -> pd.DataFrame:
    """
    因子汇总对比表: 把所有因子的 IC 指标放到一张表里,
    按 IC_IR 降序排列——一眼看出哪个因子最好。
    """
    rows = []
    for name, summary in ic_results.items():
        rows.append({"因子": name, **summary})
    df = pd.DataFrame(rows)
    if "IC_IR" in df.columns:
        df = df.sort_values("IC_IR", ascending=False)
    return df.reset_index(drop=True)
