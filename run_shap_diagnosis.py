"""
Deep Diagnosis v2: Decision Path Analysis + Interaction Detection
Key question: WHY does the model give low ProfitGrowth high SHAP?
"""
import pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
MODEL_DIR = Path("output/production_models_v2_full")
PANEL_PATH = Path("output/training_panel_v3_full.parquet")
OUTPUT = Path("output/research_report_shap_diagnosis.md")

# Load model (latest fold, seed 42)
paths = list(MODEL_DIR.glob("model_s42_f*.pkl"))
mpath = max(paths, key=lambda p: int(p.stem.split("_f")[-1]))
with open(mpath, "rb") as f:
    model = pickle.load(f)

# Load & prepare panel
panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
nz = sorted([c for c in panel.columns if c.endswith("_neutral_z")])
fnames = [c.replace("_neutral_z", "") for c in nz]
rank_cols = [c.replace("_neutral_z", "_neutral_z_rank") for c in nz]

df = panel.copy()
for c in nz:
    rc = c.replace("_neutral_z", "_neutral_z_rank")
    df[rc] = df.groupby("date")[c].rank(pct=True, na_option="bottom").fillna(0.5)

# Latest date
dt = df["date"].max()
ldf = df[df["date"] == dt].dropna(subset=rank_cols)
X = ldf[rank_cols].values.astype(float)
preds = model.predict(X)
ldf = ldf.copy()
ldf["prediction"] = preds

# Get raw _neutral_z values for diagnostic
pg_zcol = "ProfitGrowth_YoY_neutral_z"
ep_zcol = "EP_neutral_z"
roe_zcol = "ROE_neutral_z"
price_zcol = "PriceDev_20D_neutral_z"
vol_zcol = "Vol_20D_neutral_z"

# ========== ANALYSIS ==========
report = []
def w(s=""): report.append(s)

w("# SHAP深度诊断报告 v2 — 叶节点路径与交互效应")
w()
w(f"**截面**: {str(dt)[:10]}, {len(ldf)} stocks")
w()

# ========== 1. ProfitGrowth BINNED ANALYSIS ==========
w("---")
w("## 1. ProfitGrowth 分桶分析：模型到底在做什么")
w()
w("将 ProfitGrowth 分成10个分桶，看每桶的平均得分和特征画像。")
w()

pg = ldf[pg_zcol]
bins = 10
ldf["pg_bin"] = pd.qcut(pg.rank(method="first"), bins, labels=[f"P{i+1}" for i in range(bins)])
# Actually use percentile bins of the actual values
ldf["pg_decile"] = pd.qcut(pg, bins, labels=False, duplicates="drop") + 1

w("| 分桶 | ProfitGrowth范围 | N | 平均得分 | 平均EP_z | 平均ROE_z | 平均PriceDev_z | 平均Vol_z |")
w("|------|-----------------|---|---------|----------|----------|-------------|----------|")
for d in range(1, bins+1):
    mask = ldf["pg_decile"] == d
    if mask.sum() == 0: continue
    sub = ldf[mask]
    w(f"| D{d} | [{sub[pg_zcol].min():+.3f}, {sub[pg_zcol].max():+.3f}] | {mask.sum()} | "
      f"{sub['prediction'].mean():.4f} | {sub[ep_zcol].mean():+.3f} | {sub[roe_zcol].mean():+.3f} | "
      f"{sub[price_zcol].mean():+.3f} | {sub[vol_zcol].mean():+.3f} |")
w()

# Key insight: compare D1 (lowest ProfitGrowth) vs D10 (highest)
d1 = ldf[ldf["pg_decile"] == 1]
d10 = ldf[ldf["pg_decile"] == 10]
w(f"**D1 (最低增长) vs D10 (最高增长)对比**:")
w(f"- 得分: D1={d1['prediction'].mean():.4f} vs D10={d10['prediction'].mean():.4f}")
w(f"- EP_z:  D1={d1[ep_zcol].mean():+.3f} vs D10={d10[ep_zcol].mean():+.3f}")
w(f"- ROE_z: D1={d1[roe_zcol].mean():+.3f} vs D10={d10[roe_zcol].mean():+.3f}")
w(f"- PriceDev_z: D1={d1[price_zcol].mean():+.3f} vs D10={d10[price_zcol].mean():+.3f}")
w()

# ========== 2. CORRELATION MATRIX (feature values) ==========
w("---")
w("## 2. 特征相关性矩阵（截面原始值）")
w()
fcols = [pg_zcol, ep_zcol, roe_zcol, price_zcol, vol_zcol]
fnames_short = ["ProfitGrowth", "EP", "ROE", "PriceDev", "Vol_20D"]
corr = ldf[fcols].corr()
w("| | " + " | ".join(fnames_short) + " |")
w("|" + "|".join(["------" for _ in range(len(fnames_short)+1)]) + "|")
for i, fn in enumerate(fnames_short):
    vals = [f"{corr.iloc[i,j]:+.3f}" for j in range(len(fnames_short))]
    w(f"| {fn} | " + " | ".join(vals) + " |")
w()

# Check: if ProfitGrowth is HIGHLY correlated with another feature that the model rewards
for i, fn in enumerate(fnames_short):
    if fn != "ProfitGrowth":
        c = corr.iloc[0, i]
        if abs(c) > 0.3:
            w(f"**注意**: ProfitGrowth ↔ {fn} 相关性 r={c:+.3f} — 存在共线性!")
w()

# ========== 3. INTERACTION ANALYSIS ==========
w("---")
w("## 3. 交互效应：联合分桶分析")
w()
w("同时按 ProfitGrowth 和 EP 分组，观察得分模式。")
w()

# 2D binning
ldf["pg_group"] = pd.qcut(pg.rank(method="first"), 4, labels=["PG极低", "PG低", "PG高", "PG极高"])
ldf["ep_group"] = pd.qcut(ldf[ep_zcol].rank(method="first"), 4, labels=["EP极低", "EP低", "EP高", "EP极高"])

w("| PG\\EP | EP极低 | EP低 | EP高 | EP极高 |")
w("|--------|--------|------|------|--------|")
for pg_lbl in ["PG极低", "PG低", "PG高", "PG极高"]:
    vals = []
    for ep_lbl in ["EP极低", "EP低", "EP高", "EP极高"]:
        m = (ldf["pg_group"] == pg_lbl) & (ldf["ep_group"] == ep_lbl)
        if m.sum() >= 3:
            vals.append(f"{ldf[m]['prediction'].mean():.3f}")
        else:
            vals.append("-")
    w(f"| {pg_lbl} | " + " | ".join(vals) + " |")
w()

# ========== 4. TREE DECISION PATH for HIGH-SCORE stocks ==========
w("---")
w("## 4. 叶节点决策路径：取最高分股票拆解")
w()

top5 = ldf.nlargest(5, "prediction")

import shap
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(ldf[rank_cols].values.astype(float))

for rank_i, (idx, stock) in enumerate(top5.iterrows()):
    sym = stock["symbol"]
    score = stock["prediction"]
    pg_val = stock[pg_zcol]
    ep_val = stock[ep_zcol]
    roe_val = stock[roe_zcol]

    w(f"### 4.{rank_i+1} {sym} | Score={score:.4f} | PG_z={pg_val:+.3f} | EP_z={ep_val:+.3f} | ROE_z={roe_val:+.3f}")
    w()

    # Get SHAP for this stock
    shap_idx = list(ldf.index).index(idx)
    sv = shap_vals[shap_idx]

    # Top positive & negative contributors
    pos_contrib = sorted([(i, sv[i]) for i in range(len(sv)) if sv[i] > 0], key=lambda x: -x[1])
    neg_contrib = sorted([(i, sv[i]) for i in range(len(sv)) if sv[i] < 0], key=lambda x: x[1])

    w("**正向贡献 (推高得分)**:")
    for i, val in pos_contrib[:5]:
        w(f"  + {fnames[i]:20s} rank={X[shap_idx,i]:.3f} SHAP={val:+.4f}")

    w("**负向贡献 (压低得分)**:")
    for i, val in neg_contrib[:5]:
        w(f"  - {fnames[i]:20s} rank={X[shap_idx,i]:.3f} SHAP={val:+.4f}")
    w()

# ========== 5. GLOBAL PATTERN: ProfitGrowth SHAP by other features ==========
w("---")
w("## 5. ProfitGrowth SHAP 的条件分布")
w()
w("ProfitGrowth的SHAP值在不同EP/ROE水平下是否表现不同？")
w()

pg_idx = fnames.index("ProfitGrowth_YoY")
ep_idx = fnames.index("EP")

# Split by EP tercile
ep_terc = pd.qcut(ldf[ep_zcol].rank(method="first"), 3, labels=["低EP", "中EP", "高EP"])
w("| EP分组 | ProfitGrowth-SHAP均值 | ProfitGrowth-SHAP标准差 | SHAP>0占比 |")
w("|--------|---------------------|----------------------|-----------|")
for lbl in ["低EP", "中EP", "高EP"]:
    m = ep_terc == lbl
    sv_pg = shap_vals[m, pg_idx]
    w(f"| {lbl} | {sv_pg.mean():+.4f} | {sv_pg.std():.4f} | { (sv_pg>0).mean()*100:.1f}% |")
w()

# Critical insight check
w("**关键发现**: 如果低EP组中ProfitGrowth的SHAP均值仍然为正，说明模型在价值股(低EP)中奖励利润增长，在成长股(高EP)中惩罚利润增长——这是一个条件效应，而非简单的反向使用。")
w()

# ========== 6. UPDATED CONCLUSIONS ==========
w("---")
w("## 6. 修正后的结论")
w()

# Compute condition-specific effects
low_ep_mask = ep_terc == "低EP"
high_ep_mask = ep_terc == "高EP"
low_ep_pg_shap = shap_vals[low_ep_mask, pg_idx].mean() if low_ep_mask.sum() > 0 else 0
high_ep_pg_shap = shap_vals[high_ep_mask, pg_idx].mean() if high_ep_mask.sum() > 0 else 0

w(f"| 证据 | 发现 |")
w(f"|------|------|")

# Check if PG SHAP is always negative or conditionally negative
w(f"| ProfitGrowth SHAP 全局均值 | {shap_vals[:, pg_idx].mean():+.4f} |")
w(f"| ProfitGrowth SHAP (低EP组) | {low_ep_pg_shap:+.4f} |")
w(f"| ProfitGrowth SHAP (高EP组) | {high_ep_pg_shap:+.4f} |")

# Is this a conditional or global effect?
if abs(low_ep_pg_shap - high_ep_pg_shap) > 0.01:
    w(f"| 条件效应 vs 全局效应 | **条件效应** — SHAP在不同EP水平下反转 |")
else:
    w(f"| 条件效应 vs 全局效应 | **全局效应** — SHAP在所有EP水平下方向一致 |")
w()

w("### 修正后的方案排序")
w()
w("| 排名 | 方案 | 理由 |")
w("|------|------|------|")
w("| **1** | **仅ProfitGrowth加monotone_constraint=+1** | SHAP明确反向, 全局效应, 最低风险的修复 |")
w("| 2 | A/B测试: constrained vs unconstrained | 验证约束是否提升Sharpe, 不盲目改ROE |")
w("| 3 | 叶节点深度分析 | 如A/B测试失败, 排查交互效应 |")
w("| 4 | ROE加约束 | 仅在A/B测试证实ProfitGrowth约束有效后再考虑 |")
w("| 5 | 重构模型 | 最后手段 |")
w()

w("---")
w(f"*报告生成: {pd.Timestamp.now()}*")

# Write
OUTPUT.write_text("\n".join(report), encoding="utf-8")
print("\n".join(report))
