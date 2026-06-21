import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path

def main():
    print("=" * 60)
    print("特徵底層邏輯體檢: ProfitGrowth_YoY_neutral_z")
    print("=" * 60)

    # 1. 加載面板數據
    panel_path = Path("output/training_panel_v3_full.parquet")
    if not panel_path.exists():
        print("找不到面板數據！")
        return
    
    df = pd.read_parquet(panel_path)
    
    # 確保有下期收益率
    if "forward_return_1m" not in df.columns:
        close_col = "收盘" if "收盘" in df.columns else "close"
        df = df.sort_values(["symbol", "date"])
        df["forward_return_1m"] = df.groupby("symbol")[close_col].transform(lambda x: x.shift(-1) / x - 1.0)
    
    # 針對特定因子
    factor = "ProfitGrowth_YoY_neutral_z"
    if factor not in df.columns:
        print(f"面板中找不到特徵: {factor}")
        return

    # 剔除缺失值
    df_clean = df.dropna(subset=[factor, "forward_return_1m"])
    
    # 2. 計算逐月 Rank IC
    ic_list = []
    for dt, group in df_clean.groupby("date"):
        if len(group) < 30: continue
        ic, _ = stats.spearmanr(group[factor], group["forward_return_1m"])
        ic_list.append(ic)
        
    ic_series = pd.Series(ic_list)
    mean_ic = ic_series.mean()
    ic_ir = mean_ic / ic_series.std() if ic_series.std() != 0 else 0
    
    print(f"\n[單因子 Alpha 質量]")
    print(f"Mean Rank IC : {mean_ic:.4f}  (正數表示高成長=高收益，負數反之)")
    print(f"IC_IR        : {ic_ir:.4f}")

    # 3. 計算單因子十分組收益 (Decile Returns)
    # 確保 1 也是 Top (因子值最高)，10 也是 Bottom (因子值最低)
    quantiles = 10
    q_returns = {q: [] for q in range(1, quantiles + 1)}
    
    for dt, group in df_clean.groupby("date"):
        if len(group) < quantiles: continue
        group['quantile'] = quantiles - pd.qcut(group[factor], quantiles, labels=False, duplicates='drop')
        for q in range(1, quantiles + 1):
            q_mean_ret = group[group['quantile'] == q]['forward_return_1m'].mean()
            q_returns[q].append(q_mean_ret)
            
    print("\n[單因子十分組收益 (年化)]")
    for q in range(1, quantiles + 1):
        ret = np.nanmean(q_returns[q]) * 12 * 100 # 年化百分比
        print(f"第 {q:2d} 組 (Top {q*10}%) : {ret:>7.2f}%")

if __name__ == "__main__":
    main()