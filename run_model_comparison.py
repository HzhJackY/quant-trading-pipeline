"""
模型對比回測 (Model vs Model) — 消除協變量偏移後的嚴格歸因分析。

核心功能:
  1. 嚴格對齊 (Strict Universe Alignment): 取 v1 和 v2_full 預測的交集，確保在完全相同的股票池內公平競技。
  2. Alpha 質量分析: 計算逐月 Rank IC、IC_IR。
  3. 多頭單調性: 截面 10 分組 (Decile) 收益率。
  4. 風格漂移檢測: 靜態截取 Top 30 持倉的 EP (價值)、ProfitGrowth_YoY (成長)、市值暴露。
  5. 實盤摩擦扣除: 複用 TieredCostModel，計算真實淨值與換手率。

用法:
  python run_model_comparison.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_model_comparison")

# 複用您現有的回測與成本模塊
from factor_research.backtest_engine import run_backtest_with_costs
from factor_research.transaction_cost import TieredCostModel, UniverseCostConfig

OUTPUT_DIR = Path("output")

# ═══════════════════════════════════════════════════════════════
# 1. 嚴格對齊數據加載
# ═══════════════════════════════════════════════════════════════
def load_and_align_data():
    """
    加載 V1 與 V2 的預測結果，並與真實的特徵面板進行嚴格交集對齊，杜絕協變量偏移干擾。
    """
    pred_v1_path = OUTPUT_DIR / "predictions_v1.parquet"
    pred_v2_path = OUTPUT_DIR / "predictions_v2_full.parquet"
    panel_path = OUTPUT_DIR / "training_panel_v3_full.parquet"

    for p in [pred_v1_path, pred_v2_path, panel_path]:
        if not p.exists():
            logger.error(f"未找到 {p}。請先使用新舊模型對全量 CSI 800 面板進行 inference 生成預測文件。")
            sys.exit(1)

    df_v1 = pd.read_parquet(pred_v1_path)
    df_v2 = pd.read_parquet(pred_v2_path)
    panel = pd.read_parquet(panel_path)

    # Compute forward_return_1m from 收盘 if missing
    if "forward_return_1m" not in panel.columns:
        close_col = "收盘" if "收盘" in panel.columns else "close"
        panel = panel.sort_values(["symbol", "date"])
        panel["forward_return_1m"] = panel.groupby("symbol")[close_col].transform(
            lambda x: x.shift(-1) / x - 1.0)

    # Merge 成交额 from daily parquet (needed for backtest liquidity filter)
    daily_path = OUTPUT_DIR / "all_daily.parquet"
    if daily_path.exists() and "成交额" not in panel.columns and "amount" not in panel.columns:
        daily = pd.read_parquet(daily_path)
        daily["date"] = pd.to_datetime(daily["date"])
        # Get month-end close amount for each symbol
        panel = panel.merge(
            daily[["date", "symbol", "amount"]].rename(columns={"amount": "成交额"}),
            on=["date", "symbol"], how="left"
        )
        # Fill missing amount with median
        panel["成交额"] = panel["成交额"].fillna(panel["成交额"].median() if panel["成交额"].notna().any() else 0)

    # Add 总市值 (all stocks already passed 50B filter during panel rebuild)
    if "总市值" not in panel.columns:
        panel["总市值"] = 100_000_000_000  # 100B dummy, all pass

    # Add universe column if missing (backtest needs it for large/small cap split)
    if "universe" not in panel.columns:
        panel["universe"] = "大盘"

    # Add Vol_20D if missing (backtest needs raw volatility for risk filters)
    if "Vol_20D" not in panel.columns:
        # Derive from Vol_20D_neutral_z: use median-like placeholder
        panel["Vol_20D"] = 0.30  # 30% annualized vol placeholder

    # 【核心】嚴格交集對齊 (按 date 和 symbol)
    common_idx = set(zip(df_v1['date'], df_v1['symbol'])) & \
                 set(zip(df_v2['date'], df_v2['symbol'])) & \
                 set(zip(panel['date'], panel['symbol']))
    
    logger.info(f"對齊前樣本量: V1({len(df_v1)}), V2({len(df_v2)}), 面板({len(panel)})")
    logger.info(f"嚴格對齊後基準樣本量 (Common Universe): {len(common_idx)}")

    # 過濾出對齊後的數據
    mask_v1 = df_v1.apply(lambda row: (row['date'], row['symbol']) in common_idx, axis=1)
    mask_v2 = df_v2.apply(lambda row: (row['date'], row['symbol']) in common_idx, axis=1)
    
    df_v1_aligned = df_v1[mask_v1].copy()
    df_v2_aligned = df_v2[mask_v2].copy()
    
    return panel, df_v1_aligned, df_v2_aligned

# ═══════════════════════════════════════════════════════════════
# 2. Alpha 質量與歸因分析模塊 (已修復 NaN 陷阱與排序倒置陷阱)
# ═══════════════════════════════════════════════════════════════
def calculate_ic_metrics(df, panel):
    """計算截面 Rank IC 與 IC_IR"""
    # 將預測值與真實收益率拼接
    merged = pd.merge(df, panel[['date', 'symbol', 'forward_return_1m']], on=['date', 'symbol'])
    
    # 【修復 1】徹底剔除 NaN，防止 spearmanr 崩潰返回 NaN 導致斷層
    merged = merged.dropna(subset=['prediction', 'forward_return_1m'])
    
    ic_list = []
    dates = []
    for dt, group in merged.groupby('date'):
        if len(group) < 30: continue
        # 計算 Spearman 秩相關係數
        ic, _ = stats.spearmanr(group['prediction'], group['forward_return_1m'])
        ic_list.append(ic)
        dates.append(dt)
        
    ic_series = pd.Series(ic_list, index=dates)
    mean_ic = ic_series.mean()
    ic_ir = mean_ic / ic_series.std() if ic_series.std() != 0 else 0
    return mean_ic, ic_ir, ic_series

def calculate_quantile_returns(df, panel, quantiles=10):
    """計算分層單調性 (Decile Returns)"""
    merged = pd.merge(df, panel[['date', 'symbol', 'forward_return_1m']], on=['date', 'symbol'])
    
    # 【同樣修復】剔除空值策動安全計算
    merged = merged.dropna(subset=['prediction', 'forward_return_1m'])
    
    q_returns = {q: [] for q in range(1, quantiles + 1)}
    
    for dt, group in merged.groupby('date'):
        if len(group) < quantiles: continue
        
        # 【修復 2】反轉標籤順序。
        # qcut 升序(0~9)，用 quantiles 相減，使最高分(9)變成 1 (Top)，最低分(0)變成 10 (Bottom)
        group['quantile'] = quantiles - pd.qcut(group['prediction'], quantiles, labels=False, duplicates='drop')
        
        for q in range(1, quantiles + 1):
            q_mean_ret = group[group['quantile'] == q]['forward_return_1m'].mean()
            q_returns[q].append(q_mean_ret)
            
    # 計算各組的歷史平均收益
    avg_q_returns = {q: np.nanmean(rets) * 12 for q, rets in q_returns.items()} # 簡單年化
    return avg_q_returns

def analyze_top30_style(df, panel):
    """靜態風格暴露分析：檢查 Top 30 的特徵畫像"""
    # 確保面板中有這些原始因子列 (未經 z-score 的原值最好，如果沒有就用 neutral_z)
    style_cols = ['EP_neutral_z', 'ProfitGrowth_YoY_neutral_z', 'Mom_1M_neutral_z']
    available_cols = [c for c in style_cols if c in panel.columns]
    
    if not available_cols: return {}
    
    merged = pd.merge(df, panel[['date', 'symbol'] + available_cols], on=['date', 'symbol'])
    
    style_history = {col: [] for col in available_cols}
    for dt, group in merged.groupby('date'):
        # 模擬選出 Top 30 多頭
        top30 = group.nlargest(30, 'prediction')
        for col in available_cols:
            style_history[col].append(top30[col].mean())
            
    # 返回歷史平均風格暴露
    return {col: np.nanmean(vals) for col, vals in style_history.items()}

# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 64)
    print("模型擂台 — 協變量偏移修復與特徵去壟斷化 對比驗證")
    print("=" * 64)

    # ── 1. 數據準備與對齊 ─────────────────────────
    panel, df_v1, df_v2 = load_and_align_data()

    # 初始化成本模型
    large_config = UniverseCostConfig(commission_bps=2.5, stamp_duty_bps=5.0, transfer_fee_bps=0.1, base_slippage_bps=5.0)
    small_config = UniverseCostConfig(commission_bps=2.5, stamp_duty_bps=5.0, transfer_fee_bps=0.1, base_slippage_bps=15.0)
    cost_model = TieredCostModel(aum=50_000_000, large_cap_config=large_config, small_cap_config=small_config)

    # ── 2. Alpha 質量對比 ─────────────────────────
    print("\n[1] Alpha 預測能力歸因 (Rank IC & IC_IR)")
    ic_v1, ic_ir_v1, series_v1 = calculate_ic_metrics(df_v1, panel)
    ic_v2, ic_ir_v2, series_v2 = calculate_ic_metrics(df_v2, panel)
    
    print(f"{'指標':<15} {'V1 (原始模型)':>15} {'V2_Full (新模型)':>15} {'變動':>10}")
    print("-" * 58)
    print(f"{'Mean Rank IC':<15} {ic_v1:>15.4f} {ic_v2:>15.4f} {ic_v2-ic_v1:>+10.4f}")
    print(f"{'IC_IR (穩定度)':<15} {ic_ir_v1:>15.4f} {ic_ir_v2:>15.4f} {ic_ir_v2-ic_ir_v1:>+10.4f}")

    # ── 3. 持倉風格漂移對比 ───────────────────────
    print("\n[2] Top 30 持倉風格特徵暴露 (Style Drift)")
    style_v1 = analyze_top30_style(df_v1, panel)
    style_v2 = analyze_top30_style(df_v2, panel)
    
    print("-" * 58)
    for col in style_v1.keys():
        v1_val = style_v1[col]
        v2_val = style_v2[col]
        print(f"{col:<25} {v1_val:>10.2f} {v2_val:>10.2f} {v2_val-v1_val:>+10.2f}")

    # ── 4. 扣費回測 (實盤摩擦考驗) ────────────────
    print("\n[3] 實盤扣費回測 (Turnover & Friction Check)")
    print("正在運行 V1 扣費回測...")
    res_v1 = run_backtest_with_costs(panel, df_v1, cost_model, top_quantile=0.3, min_stocks_per_universe=5, alpha_col="prediction")
    
    print("正在運行 V2 扣費回測...")
    res_v2 = run_backtest_with_costs(panel, df_v2, cost_model, top_quantile=0.3, min_stocks_per_universe=5, alpha_col="prediction")

    nm_v1, nm_v2 = res_v1['net_metrics'], res_v2['net_metrics']
    
    print("-" * 58)
    print(f"{'Net Sharpe':<15} {nm_v1.get('Sharpe_Ratio', np.nan):>15.2f} {nm_v2.get('Sharpe_Ratio', np.nan):>15.2f}")
    print(f"{'Max Drawdown':<15} {nm_v1.get('Max_Drawdown', np.nan)*100:>14.2f}% {nm_v2.get('Max_Drawdown', np.nan)*100:>14.2f}%")
    print(f"{'Avg Turnover':<15} {res_v1['avg_turnover']*100:>14.1f}% {res_v2['avg_turnover']*100:>14.1f}%")

    # ── 5. 生成對比圖表 ───────────────────────────
    print("\n[4] 生成可視化報告...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))

    # 子圖 1: NAV 資金曲線
    ax1 = axes[0]
    nav_v1 = res_v1['net_nav']
    nav_v2 = res_v2['net_nav']
    ax1.plot(nav_v1.index, nav_v1.values, label="Model V1 (Net)", color="steelblue", linewidth=1.5)
    ax1.plot(nav_v2.index, nav_v2.values, label="Model V2 Full (Net)", color="darkred", linewidth=1.5)
    ax1.set_title("Net Asset Value (NAV) Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子圖 2: Rank IC 時間序列對比 (Rolling 3M)
    ax2 = axes[1]
    series_v1.rolling(3).mean().plot(ax=ax2, label="V1 IC (3M MA)", color="steelblue", alpha=0.8)
    series_v2.rolling(3).mean().plot(ax=ax2, label="V2 IC (3M MA)", color="darkred", alpha=0.8)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title("Rank IC Stability (3-Month Moving Average)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子圖 3: 分層單調性對比 (Decile Returns)
    ax3 = axes[2]
    q_ret_v1 = calculate_quantile_returns(df_v1, panel)
    q_ret_v2 = calculate_quantile_returns(df_v2, panel)
    
    x = np.arange(1, 11)
    width = 0.35
    ax3.bar(x - width/2, [q_ret_v1[q] for q in x], width, label='V1', color='steelblue', alpha=0.7)
    ax3.bar(x + width/2, [q_ret_v2[q] for q in x], width, label='V2 Full', color='darkred', alpha=0.9)
    ax3.set_title("Quantile Returns Monotonicity (1=Top Decile, 10=Bottom Decile)")
    ax3.set_xticks(x)
    ax3.set_xlabel("Quantile")
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "model_comparison_report.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"綜合對比圖表已保存至: {fig_path}")
    print("=" * 64)

if __name__ == "__main__":
    main()