"""
P2b: 行业中性化对比测试。

对比: 原版因子 (MAD → Z-score) vs 中性化因子 (MAD → 行业中性化 → Z-score)
行业分类使用股票代码前缀 (5 大板块), 无需额外 API。

用法: python test_neutralization.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

from data.cleaner import winsorize_mad, standardize_cross_section

OUTPUT_DIR = Path("output")
PANEL_FILE = OUTPUT_DIR / "panel.parquet"

FACTOR_COLS = [
    "Mom_1M", "Mom_3M", "Mom_6M", "Mom_12M_1M",
    "Vol_20D", "Vol_60D", "Beta",
    "BP", "EP", "ROE", "Debt_Ratio", "Net_Profit_Margin",
]

# ─── 板块分类 ─────────────────────────────────────────

def classify_board(symbol: str) -> str:
    """按股票代码前缀分类到 5 大板块。"""
    sym = str(symbol)
    if sym.startswith("688"):
        return "科创板"
    if sym.startswith("300") or sym.startswith("301"):
        return "创业板"
    if sym.startswith("002"):
        return "深市中小板"
    if sym.startswith("600") or sym.startswith("601") or sym.startswith("603") or sym.startswith("605"):
        return "沪市主板"
    if sym.startswith("000") or sym.startswith("001"):
        return "深市主板"
    return "其他"


# ─── 行业中性化 (简化版, 无市值) ──────────────────────

def neutralize_by_industry(
    df: pd.DataFrame,
    factor_col: str,
    industry_col: str = "board",
    date_col: str = "date",
) -> pd.Series:
    """
    对因子做行业中性化。

    对每个时间截面, 将因子对行业哑变量做 OLS 回归, 取残差。
    残差 = 因子中不能被行业解释的"纯 alpha"部分。

    参数
    ----
    df : DataFrame, 必须包含 factor_col, industry_col, date_col
    factor_col : 要中性化的因子列名
    industry_col : 行业/板块列名
    date_col : 日期列名

    返回
    ----
    pd.Series : 中性化后的因子残差序列, 与 df 同 index
    """
    from statsmodels.api import OLS, add_constant

    result = pd.Series(np.nan, index=df.index, dtype=float)

    for date, idx in df.groupby(date_col).groups.items():
        group = df.loc[idx]
        y = group[factor_col].astype(float)

        # 排除 NaN
        valid = y.notna()
        if valid.sum() < 10 or group.loc[valid, industry_col].nunique() < 2:
            result.loc[idx] = y.values
            continue

        y_valid = y[valid]
        group_valid = group.loc[valid]

        # 行业 → 哑变量
        industry_dummies = pd.get_dummies(
            group_valid[industry_col], drop_first=True
        ).astype(float)

        X = add_constant(industry_dummies, has_constant="add")

        try:
            model = OLS(y_valid.values, X.values, missing="drop").fit()
            result.loc[idx[valid]] = model.resid
        except Exception:
            result.loc[idx] = y.values

    return result


# ═══════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════

print("=" * 60)
print("P2b: 行业中性化对比测试")
print("=" * 60)

# 1. 加载面板
panel = pd.read_parquet(PANEL_FILE)
print(f"\n加载面板: {panel.shape[0]} 行 × {panel.shape[1]} 列")
print(f"股票数: {panel['symbol'].nunique()}, 日期范围: {panel['date'].min()} ~ {panel['date'].max()}")

# 2. 添加板块列
panel["board"] = panel["symbol"].apply(classify_board)
board_counts = panel.groupby("board")["symbol"].nunique()
print(f"\n板块分布:")
for board, count in board_counts.items():
    print(f"  {board}: {count} 只")

# 3. 去重: 确保每个 (date, symbol) 只有一行 (取第一行)
#    检查是否有重复
dup_count = panel.duplicated(subset=["date", "symbol"]).sum()
if dup_count > 0:
    print(f"\n[WARN] 发现 {dup_count} 行 (date, symbol) 重复, 保留首行")
    panel = panel.drop_duplicates(subset=["date", "symbol"], keep="first")
    print(f"  去重后: {panel.shape[0]} 行")

# 4. 构造下期收益
panel = panel.sort_values(["symbol", "date"]).copy()
panel["forward_return_1m"] = panel.groupby("symbol")["收盘"].transform(
    lambda x: (x.shift(-1) - x) / x
)
panel = panel.dropna(subset=["forward_return_1m"])

# 5. 因子预处理与 IC 计算
available = [c for c in FACTOR_COLS if c in panel.columns]
print(f"\n可用因子: {len(available)}/12")

# ── 方案 A: 原版 (MAD → Z-score) ──
print("\n" + "=" * 60)
print("方案 A: 原版 (MAD → Z-score, 无中性化)")
print("=" * 60)

panel_a = panel.copy()
for col in tqdm(available, desc="方案A 预处理"):
    # MAD 去极值
    panel_a[col] = panel_a.groupby("date")[col].transform(
        lambda x: winsorize_mad(x, n_mad=3.0)
    )
    # Z-score
    panel_a = standardize_cross_section(panel_a, factor_col=col, date_col="date")

# IC 计算
ic_results_a = {}
z_cols_a = [c for c in panel_a.columns if c.endswith("_z")]
for col in tqdm(z_cols_a, desc="方案A IC"):
    ic_list = []
    for dt, grp in panel_a.groupby("date"):
        sub = grp[[col, "forward_return_1m"]].dropna()
        if len(sub) >= 20:
            ic, _ = spearmanr(sub[col], sub["forward_return_1m"])
            ic_list.append(ic)
    if ic_list:
        name = col.replace("_z", "")
        ic_results_a[name] = {
            "IC_Mean": np.mean(ic_list),
            "IC_Std": np.std(ic_list, ddof=1),
            "IC_IR": np.mean(ic_list) / np.std(ic_list, ddof=1) if np.std(ic_list, ddof=1) > 0 else 0,
            "IC_Win_Rate": np.mean(np.array(ic_list) > 0),
            "Periods": len(ic_list),
        }

# ── 方案 B: 行业中性化 (MAD → 中性化 → Z-score) ──
print("\n" + "=" * 60)
print("方案 B: 行业中性化 (MAD → 板块中性化 → Z-score)")
print("=" * 60)

panel_b = panel.copy()
for col in tqdm(available, desc="方案B 预处理"):
    # MAD 去极值
    panel_b[col] = panel_b.groupby("date")[col].transform(
        lambda x: winsorize_mad(x, n_mad=3.0)
    )
    # 行业中性化
    panel_b[f"{col}_neutral"] = neutralize_by_industry(
        panel_b, factor_col=col, industry_col="board", date_col="date"
    )
    # Z-score (在中性化后的值上做)
    panel_b = standardize_cross_section(
        panel_b, factor_col=f"{col}_neutral", date_col="date"
    )

# IC 计算
ic_results_b = {}
z_cols_b = [c for c in panel_b.columns if c.endswith("_neutral_z")]
for col in tqdm(z_cols_b, desc="方案B IC"):
    ic_list = []
    for dt, grp in panel_b.groupby("date"):
        sub = grp[[col, "forward_return_1m"]].dropna()
        if len(sub) >= 20:
            ic, _ = spearmanr(sub[col], sub["forward_return_1m"])
            ic_list.append(ic)
    if ic_list:
        name = col.replace("_neutral_z", "")
        ic_results_b[name] = {
            "IC_Mean": np.mean(ic_list),
            "IC_Std": np.std(ic_list, ddof=1),
            "IC_IR": np.mean(ic_list) / np.std(ic_list, ddof=1) if np.std(ic_list, ddof=1) > 0 else 0,
            "IC_Win_Rate": np.mean(np.array(ic_list) > 0),
            "Periods": len(ic_list),
        }

# ── 对比输出 ──
print("\n" + "=" * 80)
print("对比结果: 原版 vs 行业中性化")
print(f"{'='*80}")
print(f"{'因子':20s}  {'原版 IC_IR':>10s}  {'中性化 IC_IR':>10s}  {'变化':>8s}  {'原版 Win':>8s}  {'中性化 Win':>8s}")
print("-" * 80)

improvements = []
for factor in FACTOR_COLS:
    if factor not in ic_results_a or factor not in ic_results_b:
        continue
    a = ic_results_a[factor]
    b = ic_results_b[factor]
    delta = b["IC_IR"] - a["IC_IR"]
    improvements.append({"factor": factor, "delta_ic_ir": delta, **a, **b})

    direction = "↑" if delta > 0 else "↓"
    print(
        f"{factor:20s}  {a['IC_IR']:+8.4f}  {b['IC_IR']:+8.4f}  "
        f"{delta:+7.4f}{direction}  {a['IC_Win_Rate']:7.1%}  {b['IC_Win_Rate']:7.1%}"
    )

# ── 汇总指标 ──
print(f"\n{'='*60}")
print("汇总指标:")
print(f"{'='*60}")

for label, results in [("原版", ic_results_a), ("行业中性化", ic_results_b)]:
    ic_irs = [v["IC_IR"] for v in results.values()]
    mean_abs = np.mean(np.abs(ic_irs))
    max_abs = np.max(np.abs(ic_irs))
    n_pos = sum(1 for v in ic_irs if v > 0)
    print(f"  {label}: Mean|IC_IR|={mean_abs:.4f}, "
          f"Max|IC_IR|={max_abs:.4f}, "
          f"正向因子={n_pos}/{len(ic_irs)}")

# ── 保存结果 ──
comparison = pd.DataFrame(improvements)
comparison.to_csv(OUTPUT_DIR / "neutralization_comparison.csv", index=False, encoding="utf-8-sig")
print(f"\n对比结果已保存到 output/neutralization_comparison.csv")

# ── 中性化前后的因子相关性变化 ──
print(f"\n{'='*60}")
print("因子相关性变化 (中性化前 → 中性化后):")
print(f"{'='*60}")

# 取最新截面
latest_date = panel_b["date"].max()
sub_before = panel_a[panel_a["date"] == latest_date][z_cols_a].dropna()
sub_after = panel_b[panel_b["date"] == latest_date][z_cols_b].dropna()

# 计算平均绝对相关系数
def mean_abs_corr(df):
    if len(df.columns) < 2:
        return 0
    corr = df.corr()
    # 上三角 (排除对角线)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    return np.abs(corr.values[mask]).mean()

# Rename columns for readable output
sub_before.columns = [c.replace("_z", "") for c in sub_before.columns]
sub_after.columns = [c.replace("_neutral_z", "") for c in sub_after.columns]

mac_before = mean_abs_corr(sub_before)
mac_after = mean_abs_corr(sub_after)
print(f"  平均|相关系数| (中性化前): {mac_before:.4f}")
print(f"  平均|相关系数| (中性化后): {mac_after:.4f}")
print(f"  冗余度变化: {mac_before - mac_after:+.4f} (正值=中性化降低了因子冗余)")

print(f"\n{'='*60}")
print("测试完成。")
print(f"{'='*60}")
