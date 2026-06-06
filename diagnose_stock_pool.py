"""
P2a 诊断: 选股池采样方式对因子 IC 的影响。

对比三种采样方式:
  A. top-200 (当前方式, 按 API 返回顺序取前 200)
  B. random-200 (从 800 只中随机抽 200)
  C. all-800 (全市场)

结果保存到 output/pool_diagnosis.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

# ─── 复用现有模块 ─────────────────────────────────────
from data.fetcher import Fetcher
from data.cleaner import winsorize_mad, standardize_cross_section
from factor_lib.momentum import (
    compute_momentum_1m, compute_momentum_3m,
    compute_momentum_6m, compute_momentum_12m_1m,
)
from factor_lib.volatility import compute_volatility_20d, compute_volatility_60d, compute_beta

OUTPUT_DIR = Path("output")
PANEL_FILE = OUTPUT_DIR / "panel.parquet"
FACTOR_COLS = [
    "Mom_1M", "Mom_3M", "Mom_6M", "Mom_12M_1M",
    "Vol_20D", "Vol_60D", "Beta",
    "BP", "EP", "ROE", "Debt_Ratio", "Net_Profit_Margin",
]

f = Fetcher()


def build_panel_for_symbols(symbols: list[str], label: str) -> pd.DataFrame:
    """为给定股票列表构建因子面板 (复用 Stage 2 逻辑)。"""
    print(f"\n{'='*60}")
    print(f"[{label}] 构建因子面板: {len(symbols)} 只")
    print(f"{'='*60}")

    # ── 月频行情 ──
    all_daily = []
    for sym in tqdm(symbols, desc=f"{label}/月频"):
        try:
            daily = f.get_daily(sym, "20170101", "20241231")
            daily = daily[daily["日期"] >= "20170101"].copy()
            daily["month"] = daily["日期"].dt.to_period("M")
            month_end = daily.groupby("month").tail(1).copy()
            month_end["symbol"] = sym
            all_daily.append(month_end)
        except Exception:
            continue

    daily_panel = pd.concat(all_daily, ignore_index=True)
    daily_panel = daily_panel.rename(columns={"日期": "date"})

    # ── 日度行情 (因子计算) ──
    daily_full = []
    for sym in tqdm(symbols, desc=f"{label}/日度"):
        try:
            d = f.get_daily(sym, "20170101", "20241231")
            d["symbol"] = sym
            d = d.rename(columns={"日期": "date"})
            daily_full.append(d[["date", "symbol", "收盘"]])
        except Exception:
            continue

    if daily_full:
        daily_all = pd.concat(daily_full, ignore_index=True)
        daily_all = daily_all.rename(columns={"收盘": "close"})
    else:
        daily_all = daily_panel[["date", "symbol"]].copy()
        daily_all["close"] = 0.0

    # 动量因子
    mom_1m   = compute_momentum_1m(daily_all)
    mom_3m   = compute_momentum_3m(daily_all)
    mom_6m   = compute_momentum_6m(daily_all)
    mom_12_1 = compute_momentum_12m_1m(daily_all)

    # 波动率因子
    vol_20 = compute_volatility_20d(daily_all)
    vol_60 = compute_volatility_60d(daily_all)
    beta   = compute_beta(daily_all)

    for fdf in [mom_1m, mom_3m, mom_6m, mom_12_1, vol_20, vol_60, beta]:
        if fdf is not None and not fdf.empty:
            daily_panel = daily_panel.merge(fdf, on=["date", "symbol"], how="left")

    # ── 财务因子 (PIT) ──
    fin_frames = []
    for sym in tqdm(symbols, desc=f"{label}/财务"):
        try:
            hist = f.get_financial_history(sym)
            if not hist.empty:
                fin_frames.append(hist)
        except Exception:
            continue

    if fin_frames:
        fin_all = pd.concat(fin_frames, ignore_index=True)
        fin_all = fin_all.dropna(subset=["symbol", "report_date"])

        def _pit_merge(group: pd.DataFrame) -> pd.DataFrame:
            sym = group.name
            fin_sym = fin_all[fin_all["symbol"] == sym]
            if fin_sym.empty:
                return group
            group = group.sort_values("date")
            fin_sym = fin_sym.sort_values("report_date")
            return pd.merge_asof(
                group, fin_sym,
                left_on="date", right_on="report_date",
                direction="backward",
            )

        daily_panel = (
            daily_panel.groupby("symbol", group_keys=False)
            .apply(_pit_merge).reset_index(drop=True)
        )
        daily_panel = daily_panel.rename(columns={"销售净利率": "Net_Profit_Margin"})
        daily_panel["股价"] = daily_panel["收盘"].astype(float)

        if "每股净资产" in daily_panel.columns:
            daily_panel["BP"] = (
                daily_panel["每股净资产"].astype(float)
                / daily_panel["股价"].replace(0, float("nan"))
            )
        if "每股收益" in daily_panel.columns:
            daily_panel["EP"] = (
                daily_panel["每股收益"].astype(float)
                / daily_panel["股价"].replace(0, float("nan"))
            )

    return daily_panel


def compute_ic_for_panel(panel: pd.DataFrame, label: str) -> dict:
    """对面板计算 12 因子的 IC 汇总并返回。"""
    panel = panel.sort_values(["symbol", "date"]).copy()
    panel["next_close"] = panel.groupby("symbol")["收盘"].shift(-1)
    panel["forward_return_1m"] = (
        panel["next_close"] - panel["收盘"].astype(float)
    ) / panel["收盘"].astype(float)
    panel = panel.dropna(subset=["forward_return_1m"])

    # 预处理
    available = [c for c in FACTOR_COLS if c in panel.columns]
    for col in available:
        panel[col] = panel.groupby("date")[col].transform(
            lambda x: winsorize_mad(x, n_mad=3.0)
        )
        panel = standardize_cross_section(panel, factor_col=col, date_col="date")

    # IC 计算
    z_cols = [c for c in panel.columns if c.endswith("_z")]
    results = {}
    for col in z_cols:
        ic_list = []
        for dt, grp in panel.groupby("date"):
            sub = grp[[col, "forward_return_1m"]].dropna()
            if len(sub) >= 20:
                ic, _ = spearmanr(sub[col], sub["forward_return_1m"])
                ic_list.append(ic)
        if ic_list:
            name = col.replace("_z", "")
            results[name] = {
                "IC_Mean": np.mean(ic_list),
                "IC_Std": np.std(ic_list, ddof=1),
                "IC_IR": np.mean(ic_list) / np.std(ic_list, ddof=1) if np.std(ic_list, ddof=1) > 0 else 0,
                "IC_Win_Rate": np.mean(np.array(ic_list) > 0),
                "Periods": len(ic_list),
                "Pool": label,
            }

    return results


# ═══════════════════════════════════════════════════════
# 主诊断流程
# ═══════════════════════════════════════════════════════

print("=" * 60)
print("P2a 诊断: 选股池采样方式对比")
print("=" * 60)

# 获取全量成分股
all_members = f.get_index_members("000906")
print(f"CSI 800 全量成分股: {len(all_members)} 只")

# 方案 A: top-200 (当前方式)
top200 = all_members[:200]

# 方案 B: random-200
rng = np.random.RandomState(42)
random200 = rng.choice(all_members, size=200, replace=False).tolist()

# 方案 C: all-800 (或限制 500 以避免太慢)
all800 = all_members  # 全部

print(f"\n方案 A (top-200):   {len(top200)} 只")
print(f"  样本: {top200[:10]}...")
print(f"\n方案 B (random-200): {len(random200)} 只")
print(f"  样本: {random200[:10]}...")
print(f"\n方案 C (all-800):    {len(all800)} 只")

# 逐方案计算 — 用嵌套字典 structured[pool][factor] = {...}
structured = {}

# 方案 A (如果 panel.parquet 已经存在且股票匹配, 直接复用)
existing_panel = None
if PANEL_FILE.exists():
    existing_panel = pd.read_parquet(PANEL_FILE)
    existing_symbols = sorted(existing_panel["symbol"].unique())
    if set(existing_symbols) == set(top200):
        print("\n[方案 A] 复用已有 panel.parquet, 跳过构建")
        panel_a = existing_panel
    else:
        panel_a = build_panel_for_symbols(top200, "A-top200")
else:
    panel_a = build_panel_for_symbols(top200, "A-top200")

structured["A-top200"] = compute_ic_for_panel(panel_a, "A-top200")

# 方案 B
panel_b = build_panel_for_symbols(random200, "B-random200")
structured["B-random200"] = compute_ic_for_panel(panel_b, "B-random200")

# 方案 C (限制 400 只以避免太慢)
sample_c = all800[:400] if len(all800) > 400 else all800
panel_c = build_panel_for_symbols(sample_c, "C-all400")
structured["C-all400"] = compute_ic_for_panel(panel_c, "C-all400")

comparison_rows = []
for factor in FACTOR_COLS:
    row = {"因子": factor}
    for pool_label in ["A-top200", "B-random200", "C-all400"]:
        if factor in structured[pool_label]:
            r = structured[pool_label][factor]
            row[f"{pool_label}_IC_IR"] = round(r["IC_IR"], 4)
            row[f"{pool_label}_IC_Mean"] = round(r["IC_Mean"], 4)
        else:
            row[f"{pool_label}_IC_IR"] = None
            row[f"{pool_label}_IC_Mean"] = None
    comparison_rows.append(row)

comparison = pd.DataFrame(comparison_rows)

print("\nIC_IR 对比 (越高越好):")
print(comparison[["因子", "A-top200_IC_IR", "B-random200_IC_IR", "C-all400_IC_IR"]].to_string(index=False))

print("\nIC_Mean 对比:")
print(comparison[["因子", "A-top200_IC_Mean", "B-random200_IC_Mean", "C-all400_IC_Mean"]].to_string(index=False))

comparison.to_csv(OUTPUT_DIR / "pool_diagnosis.csv", index=False, encoding="utf-8-sig")
print(f"\n诊断结果已保存到 output/pool_diagnosis.csv")

# ── 关键结论 ──
print(f"\n{'='*60}")
print("关键对比指标:")
print(f"{'='*60}")
for pool_label in ["A-top200", "B-random200", "C-all400"]:
    ic_irs = [structured[pool_label][f]["IC_IR"] for f in FACTOR_COLS if f in structured[pool_label]]
    if ic_irs:
        mean_abs_ic_ir = np.mean(np.abs(ic_irs))
        max_abs_ic_ir = np.max(np.abs(ic_irs))
        n_positive = sum(1 for v in ic_irs if v > 0)
        print(f"  {pool_label}: Mean|IC_IR|={mean_abs_ic_ir:.4f}, "
              f"Max|IC_IR|={max_abs_ic_ir:.4f}, "
              f"正向因子={n_positive}/{len(ic_irs)}")
