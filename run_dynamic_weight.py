"""
动态权重分配 — 运行脚本。

在 Split-Universe 双模型分析完成后, 本脚本负责:
  1. 从拼接面板提取大盘/小盘子策略收益序列
  2. 执行滚动 60 期均值-方差优化 (Max Sharpe)
  3. 对比三种权重方案: MVO动态 / 风险平价 / 固定50-50
  4. 生成权重演变图和绩效对比表

前置条件:
  - 已运行 run_split_universe.py (生成 output/split_universe_blended.parquet)
  - 已有 output/preprocessed.parquet (含 forward_return_1m)

用法:
  python run_dynamic_weight.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_dynamic_weight")

from factor_research.dynamic_weight import (
    build_sub_universe_returns,
    DynamicWeightOptimizer,
    compare_all_strategies,
    plot_weight_evolution,
    compute_performance_metrics,
)

OUTPUT_DIR = Path("output")


def load_data():
    """加载 Split-Universe 拼接面板和含收益的原始面板。"""
    blended_path = OUTPUT_DIR / "split_universe_blended.parquet"
    preprocessed_path = OUTPUT_DIR / "preprocessed.parquet"

    if not blended_path.exists():
        raise FileNotFoundError(
            f"未找到 {blended_path}。请先运行 run_split_universe.py"
        )
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"未找到 {preprocessed_path}。请先运行 run_factor_research.py 的 Stage 1-3"
        )

    print(f"加载: {blended_path}")
    blended = pd.read_parquet(blended_path)
    print(f"  拼接面板: {blended.shape[0]:,} 行 × {blended.shape[1]} 列")
    print(f"  股票数: {blended['symbol'].nunique()}")
    print(f"  日期范围: {blended['date'].min()} ~ {blended['date'].max()}")
    print(f"  大盘行数: {(blended['universe'] == '大盘').sum()}")
    print(f"  小盘行数: {(blended['universe'] == '小盘').sum()}")

    print(f"\n加载: {preprocessed_path}")
    panel = pd.read_parquet(preprocessed_path)
    print(f"  原始面板: {panel.shape[0]:,} 行 × {panel.shape[1]} 列")

    # 确保有 forward_return_1m
    if "forward_return_1m" not in panel.columns:
        print("  计算 forward_return_1m...")
        close_col = "收盘" if "收盘" in panel.columns else "close"
        panel = panel.sort_values(["symbol", "date"])
        panel["forward_return_1m"] = (
            panel.groupby("symbol")[close_col]
            .transform(lambda x: x.shift(-1) / x - 1)
        )

    return blended, panel


def main():
    print("=" * 64)
    print("动态权重分配 — 滚动均值-方差优化")
    print("=" * 64)

    # ── 参数 ─────────────────────────────────────────
    WINDOW = 60           # 滚动窗口 (月频: 60月=5年)
    BOUNDS = (0.3, 0.7)   # 权重边界: 防止极端仓位
    TOP_QUANTILE = 0.3    # 每期选取前30%股票作为子域策略
    RF = 0.0              # 无风险利率 (月频回测中影响极小)
    FREQ = "M"            # 月频

    # 1. 加载数据
    blended, panel = load_data()

    # 2. 提取子域策略收益
    print("\n" + "-" * 48)
    print("Step 1: 提取大盘/小盘策略收益序列")
    print("-" * 48)
    large_rets, small_rets = build_sub_universe_returns(
        blended_panel=blended,
        panel_with_returns=panel,
        top_quantile=TOP_QUANTILE,
    )

    # 2b. 展示提取结果的统计摘要
    n_months = len(large_rets)
    print(f"\n  策略收益序列: {n_months} 个月")
    print(f"  {'':>12} {'大盘':>12} {'小盘':>12} {'相关性':>10}")
    print(f"  {'平均月收益':>12} {large_rets.mean():>11.4%}  {small_rets.mean():>11.4%}")
    print(f"  {'月收益标准差':>12} {large_rets.std():>11.4%}  {small_rets.std():>11.4%}")
    print(f"  {'年化收益':>12} {large_rets.mean()*12:>11.2%}  {small_rets.mean()*12:>11.2%}")
    print(f"  {'年化波动':>12} {large_rets.std()*np.sqrt(12):>11.2%}  "
          f"{small_rets.std()*np.sqrt(12):>11.2%}")
    corr = large_rets.corr(small_rets)
    print(f"  {'收益相关性':>12} {corr:>11.4f}")

    # 3. 滚动均值-方差优化
    print("\n" + "-" * 48)
    print("Step 2: 滚动均值-方差优化 (Max Sharpe)")
    print("-" * 48)
    optimizer = DynamicWeightOptimizer(
        window=WINDOW,
        bounds=BOUNDS,
        rf=RF,
        freq=FREQ,
    )
    weights_df = optimizer.fit(large_rets, small_rets)

    # 3b. 展示权重统计
    print(f"\n  权重统计 (样本外 {len(weights_df)} 期):")
    print(f"  {'':>18} {'大盘':>10} {'小盘':>10}")
    print(f"  {'平均权重':>18} {weights_df['W_large'].mean():>9.2%}  "
          f"{weights_df['W_small'].mean():>9.2%}")
    print(f"  {'最小权重':>18} {weights_df['W_large'].min():>9.2%}  "
          f"{weights_df['W_small'].min():>9.2%}")
    print(f"  {'最大权重':>18} {weights_df['W_large'].max():>9.2%}  "
          f"{weights_df['W_small'].max():>9.2%}")
    print(f"  {'权重标准差':>18} {weights_df['W_large'].std():>9.4f}  "
          f"{weights_df['W_small'].std():>9.4f}")

    # 4. 三种方案绩效对比
    print("\n" + "-" * 48)
    print("Step 3: 三种权重方案绩效对比")
    print("-" * 48)
    comparison = compare_all_strategies(
        large_rets, small_rets, weights_df,
        window=WINDOW, freq=FREQ, rf=RF,
    )

    # 5. 生成图表
    print("\n" + "-" * 48)
    print("Step 4: 生成权重演变图...")
    print("-" * 48)
    try:
        plot_weight_evolution(
            weights_df,
            save_path=str(OUTPUT_DIR / "dynamic_weight_evolution.png"),
        )
    except Exception as e:
        print(f"  [WARN] 绘图失败: {e}")

    # 6. 保存结果
    print("\n" + "-" * 48)
    print("保存结果...")
    print("-" * 48)

    weights_df.to_csv(
        OUTPUT_DIR / "dynamic_weights.csv",
        index=False, encoding="utf-8-sig",
    )
    print(f"  动态权重: output/dynamic_weights.csv ({len(weights_df)} 行)")

    comparison.to_csv(
        OUTPUT_DIR / "dynamic_weight_comparison.csv",
        index=False, encoding="utf-8-sig",
    )
    print(f"  绩效对比: output/dynamic_weight_comparison.csv")

    # 7. 总结
    print("\n" + "=" * 64)
    print("动态权重分析 完成")
    print("=" * 64)
    print(f"  样本外期数: {len(weights_df)} 个月")
    print(f"  最优Sharpe动态权重  组合Sharpe = {comparison.iloc[0]['Sharpe']}")
    print(f"  风险平价             组合Sharpe = {comparison.iloc[1]['Sharpe']}")
    print(f"  固定50/50            组合Sharpe = {comparison.iloc[2]['Sharpe']}")
    print(f"\n输出文件:")
    print(f"  - output/dynamic_weights.csv         (动态权重序列)")
    print(f"  - output/dynamic_weight_comparison.csv (绩效对比表)")
    print(f"  - output/dynamic_weight_evolution.png  (权重演变图)")
    print("=" * 64)


if __name__ == "__main__":
    main()
