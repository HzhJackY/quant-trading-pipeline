# ═══════════════════════════════════════════════════════════════
# ARCHIVED: 早期消融实验 runner (V0-V3)
# 被 run_ml_v7.py 取代 (V0/V5/V7 三路对比更完整)
# 保留作为实验记录 — 不建议用于新工作
# ═══════════════════════════════════════════════════════════════
"""
ML 换手率优化 — 消融实验 (Ablation Study).

对比四种配置的扣费后绩效, 验证 Multi-Seed Ensemble 和 EMA 平滑
对换手率和 Net Sharpe 的改善效果。

实验矩阵:
  V0: 线性 alpha_signal (Baseline)
  V1: LightGBM, 单 seed=42, 无平滑 (当前 baseline)
  V2: LightGBM, 3-Seed Ensemble [42, 1024, 2026], 无平滑
  V3: LightGBM, 3-Seed Ensemble + EMA(α=0.4) 平滑

预期:
  - V2 vs V1: 换手率下降 15-25%, Sharpe 提升 0.05-0.15
  - V3 vs V2: 换手率再降 20-30%, Sharpe 进一步提升 0.05-0.10
  - V3 vs V0: 目标 Net Sharpe > 1.0, 接近或超越线性 baseline

用法:
  python run_ml_ablation.py                # 运行全部 4 组
  python run_ml_ablation.py --skip-v0-v1   # 仅运行 V2, V3 (新配置)
  python run_ml_ablation.py --v3-only      # 仅运行最优配置 V3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ml_ablation")

from factor_research.ml_engine import LightGBMAlphaEngine, MLConfig
from factor_research.backtest_engine import (
    run_backtest_with_costs,
    generate_comparison_table,
)
from factor_research.transaction_cost import TieredCostModel, UniverseCostConfig

OUTPUT_DIR = Path("output")


# ═══════════════════════════════════════════════════════════
# 数据加载 & 成本模型
# ═══════════════════════════════════════════════════════════


def load_data():
    preprocessed_path = OUTPUT_DIR / "preprocessed.parquet"
    blended_path = OUTPUT_DIR / "split_universe_blended.parquet"

    for p in [preprocessed_path, blended_path]:
        if not p.exists():
            raise FileNotFoundError(f"未找到 {p}。请先运行前置 Pipeline。")

    print(f"加载数据...")
    panel = pd.read_parquet(preprocessed_path)
    blended = pd.read_parquet(blended_path)
    print(f"  panel: {panel.shape[0]:,} 行 | {panel['date'].nunique()} 截面")
    print(f"  blended: {blended.shape[0]:,} 行 | "
          f"大盘={(blended['universe']=='大盘').sum()}, "
          f"小盘={(blended['universe']=='小盘').sum()}")
    return panel, blended


def build_cost_model(aum: float = 50_000_000) -> TieredCostModel:
    large_config = UniverseCostConfig(
        commission_bps=2.5, stamp_duty_bps=5.0, transfer_fee_bps=0.1,
        base_slippage_bps=5.0, impact_gamma=0.5, impact_eta=1.0,
    )
    small_config = UniverseCostConfig(
        commission_bps=2.5, stamp_duty_bps=5.0, transfer_fee_bps=0.1,
        base_slippage_bps=15.0, impact_gamma=0.65, impact_eta=1.5,
    )
    return TieredCostModel(aum=aum, large_cap_config=large_config, small_cap_config=small_config)


# ═══════════════════════════════════════════════════════════
# 实验组定义
# ═══════════════════════════════════════════════════════════

EXPERIMENTS = {
    "V0": {
        "name": "线性 alpha_signal (Baseline)",
        "description": "等权线性 IC 加权, 无 ML",
        "use_ml": False,
        "ml_config": None,
    },
    "V1": {
        "name": "LightGBM 单 Seed",
        "description": "纯 LightGBM, seed=42, 无平滑",
        "use_ml": True,
        "ml_config": MLConfig(
            seeds=[42],
            ema_alpha=0.0,
        ),
    },
    "V2": {
        "name": "LightGBM 3-Seed Ensemble",
        "description": "3 个独立 LightGBM (seed=42,1024,2026) 等权平均",
        "use_ml": True,
        "ml_config": MLConfig(
            seeds=[42, 1024, 2026],
            ema_alpha=0.0,
        ),
    },
    "V3": {
        "name": "LightGBM 3-Seed Ensemble + EMA(0.4)",
        "description": "3-Seed 集成后对每只股票时序 EMA 平滑 (α=0.4)",
        "use_ml": True,
        "ml_config": MLConfig(
            seeds=[42, 1024, 2026],
            ema_alpha=0.4,
        ),
    },
}


# ═══════════════════════════════════════════════════════════
# 单组实验执行
# ═══════════════════════════════════════════════════════════


def run_experiment(
    exp_id: str,
    exp_def: dict,
    panel: pd.DataFrame,
    blended: pd.DataFrame,
    cost_model: TieredCostModel,
    top_quantile: float = 0.3,
    min_stocks: int = 5,
) -> dict:
    """
    执行一组实验: 训练 (如有) + 回测。

    Returns
    -------
    dict with keys:
        nav_returns, nav_nets, turnovers, costs,
        gross_metrics, net_metrics, avg_turnover, avg_cost_bps,
        predictions (ML only), wall_time_sec
    """
    t_start = time.perf_counter()

    if exp_def["use_ml"]:
        # ── ML 训练 ──
        cfg = exp_def["ml_config"]
        print(f"\n  [训练] {exp_def['name']}")
        seeds_str = cfg.seeds if cfg.seeds else [42]
        print(f"    seeds={seeds_str}, ema_alpha={cfg.ema_alpha}")

        engine = LightGBMAlphaEngine(config=cfg)
        predictions = engine.run(panel)

        # 构造 ML blended panel
        ml_blended = blended.merge(
            predictions[["date", "symbol", "ml_signal"]],
            on=["date", "symbol"], how="left",
        )
        ml_blended["ml_signal"] = ml_blended["ml_signal"].fillna(0.5)
        alpha_col = "ml_signal"
        result_blended = ml_blended
    else:
        # ── 线性 baseline ──
        print(f"\n  [回测] {exp_def['name']}")
        alpha_col = "alpha_signal"
        result_blended = blended
        predictions = None

    # ── 回测 ──
    print(f"    回测中 (top={top_quantile:.0%}, min_stocks={min_stocks})...")
    result = run_backtest_with_costs(
        panel=panel,
        blended=result_blended,
        cost_model=cost_model,
        top_quantile=top_quantile,
        min_stocks_per_universe=min_stocks,
        alpha_col=alpha_col,
    )

    wall_time = time.perf_counter() - t_start
    nm = result.get("net_metrics") or {}
    avg_to = result.get("avg_turnover", 0.0)
    avg_cost = result.get("avg_cost_bps", 0.0)

    print(f"    完成 ({wall_time:.0f}s) | "
          f"Sharpe={nm.get('Sharpe_Ratio', 0):.4f} | "
          f"TO={avg_to*100:.1f}% | "
          f"Cost={avg_cost:.1f}bps")

    return {
        "result": result,
        "predictions": predictions,
        "wall_time_sec": wall_time,
    }


# ═══════════════════════════════════════════════════════════
# 消融实验报告
# ═══════════════════════════════════════════════════════════


def generate_ablation_table(
    results: dict[str, dict],
    aum: float,
) -> str:
    """生成消融实验对比表 (Markdown)。"""

    def fmt_pct(v, decimals=2):
        if v is None or np.isnan(v):
            return "N/A"
        return f"{v * 100:.{decimals}f}%"

    def fmt_num(v, decimals=4):
        if v is None or np.isnan(v):
            return "N/A"
        return f"{v:.{decimals}f}"

    def get_metric(exp_id, key):
        r = results.get(exp_id, {}).get("result", {})
        nm = r.get("net_metrics") or {}
        return nm.get(key)

    header = [
        f"## 消融实验报告 — Multi-Seed Ensemble & EMA 平滑",
        f"",
        f"- **AUM:** {aum/1e4:.0f} 万",
        f"- **选股比例:** Top 30% 分域等权",
        f"- **成本模型:** Almgren-Chriss 冲击 + 分域费率",
        f"- **回测期数:** {len(results.get('V0',{}).get('result',{}).get('net_returns',pd.Series()))} 期",
        f"",
        f"### 扣费后绩效对比",
        f"",
        f"| 指标 | V0: 线性 Baseline | V1: 单 Seed | V2: 3-Seed Ensemble | V3: Ensemble + EMA(0.4) |",
        f"|------|:---:|:---:|:---:|:---:|",
    ]

    metrics = [
        ("年化收益", "Annualized_Return", fmt_pct),
        ("年化波动率", "Volatility", fmt_pct),
        ("**Sharpe Ratio**", "Sharpe_Ratio", fmt_num),
        ("最大回撤", "Max_Drawdown", fmt_pct),
        ("Calmar Ratio", "Calmar_Ratio", fmt_num),
        ("月胜率", "Win_Rate", fmt_pct),
    ]

    for label, key, formatter in metrics:
        vals = [formatter(get_metric(eid, key)) for eid in ["V0", "V1", "V2", "V3"]]
        line = f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |"
        header.append(line)

    # 换手率和成本 (不是 net_metrics 的一部分)
    header.append("")
    header.append("### 交易特征")
    header.append("")
    header.append("| 指标 | V0: 线性 Baseline | V1: 单 Seed | V2: 3-Seed Ensemble | V3: Ensemble + EMA(0.4) |")
    header.append("|------|:---:|:---:|:---:|:---:|")

    for label, key in [
        ("月均单边换手率", "avg_turnover"),
        ("月均总成本 (bps)", "avg_cost_bps"),
    ]:
        vals = []
        for eid in ["V0", "V1", "V2", "V3"]:
            r = results.get(eid, {}).get("result", {})
            v = r.get(key, 0)
            if key == "avg_turnover":
                vals.append(f"{v*100:.1f}%" if v else "N/A")
            else:
                vals.append(f"{v:.1f}" if v else "N/A")
        header.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")

    # 改善幅度 (V3 vs V1)
    header.append("")
    header.append("### V3 vs V1: Ensemble + EMA 对纯 ML 的改善")
    header.append("")

    v1_sharpe = get_metric("V1", "Sharpe_Ratio") or 0
    v1_to = results.get("V1", {}).get("result", {}).get("avg_turnover", 0)
    v3_sharpe = get_metric("V3", "Sharpe_Ratio") or 0
    v3_to = results.get("V3", {}).get("result", {}).get("avg_turnover", 0)

    delta_sharpe = v3_sharpe - v1_sharpe
    delta_to_pct = (v3_to - v1_to) / v1_to * 100 if v1_to > 0 else 0

    header.append(f"| 指标 | V1 (单 Seed) | V3 (Ensemble+EMA) | Δ | 改善幅度 |")
    header.append(f"|------|:---:|:---:|:---:|:---:|")
    header.append(f"| Sharpe | {fmt_num(v1_sharpe)} | {fmt_num(v3_sharpe)} | "
                  f"{'+' if delta_sharpe >= 0 else ''}{delta_sharpe:.4f} | "
                  f"{'+' if delta_sharpe >= 0 else ''}{delta_sharpe/v1_sharpe*100:.1f}% |")
    header.append(f"| 月换手率 | {v1_to*100:.1f}% | {v3_to*100:.1f}% | "
                  f"{'+' if delta_to_pct >= 0 else ''}{delta_to_pct:.1f}% | "
                  f"{'↑' if delta_to_pct >= 0 else '↓'} {abs(delta_to_pct):.1f}% |")

    # 训练耗时
    header.append("")
    header.append("### 训练耗时")
    header.append("")
    header.append("| 配置 | 耗时 | Seeds | 模型数 |")
    header.append("|------|------|------|--------|")
    for eid in ["V0", "V1", "V2", "V3"]:
        exp = EXPERIMENTS[eid]
        r = results.get(eid, {})
        wt = r.get("wall_time_sec", 0)
        n_seeds = len(exp.get("ml_config", None).seeds) if exp.get("ml_config") else 0
        n_models = "—" if n_seeds == 0 else f"~{n_seeds}×folds"
        header.append(f"| {eid}: {exp['name']} | {wt:.0f}s | {n_seeds or '—'} | {n_models} |")

    return "\n".join(header)


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="ML 消融实验")
    parser.add_argument("--skip-v0-v1", action="store_true",
                        help="跳过 V0 和 V1 (仅运行新配置 V2, V3)")
    parser.add_argument("--v3-only", action="store_true",
                        help="仅运行 V3 (最快验证)")
    parser.add_argument("--aum", type=float, default=50_000_000,
                        help="AUM (默认 5000 万)")
    parser.add_argument("--top", type=float, default=0.3,
                        help="选股比例 (默认 0.3)")
    parser.add_argument("--min-stocks", type=int, default=5,
                        help="每域最少持仓 (默认 5)")
    args = parser.parse_args()

    print("=" * 64)
    print("ML 换手率优化 — 消融实验")
    print("=" * 64)

    # ── 加载数据 ──
    panel, blended = load_data()
    cost_model = build_cost_model(aum=args.aum)

    # ── 确定实验范围 ──
    if args.v3_only:
        exp_ids = ["V3"]
    elif args.skip_v0_v1:
        exp_ids = ["V2", "V3"]
    else:
        exp_ids = ["V0", "V1", "V2", "V3"]

    # ── 运行实验 ──
    results: dict[str, dict] = {}

    for exp_id in exp_ids:
        exp_def = EXPERIMENTS[exp_id]
        print(f"\n{'─' * 48}")
        print(f"{exp_id}: {exp_def['name']}")
        print(f"{'─' * 48}")

        r = run_experiment(
            exp_id=exp_id,
            exp_def=exp_def,
            panel=panel,
            blended=blended,
            cost_model=cost_model,
            top_quantile=args.top,
            min_stocks=args.min_stocks,
        )
        results[exp_id] = r

    # ── 输出消融实验表 ──
    print(f"\n{'=' * 64}")
    print("消融实验报告")
    print(f"{'=' * 64}")

    table_md = generate_ablation_table(results, aum=args.aum)
    print(table_md)

    # 保存报告
    report_path = OUTPUT_DIR / "ml_ablation_report.md"
    report_path.write_text(table_md, encoding="utf-8")
    print(f"\n报告已保存: {report_path}")

    # ── 保存各配置的 ML 预测 (用于后续分析) ──
    for exp_id in exp_ids:
        r = results.get(exp_id, {})
        preds = r.get("predictions")
        if preds is not None and len(preds) > 0:
            pred_path = OUTPUT_DIR / f"ml_predictions_{exp_id}.parquet"
            preds.to_parquet(pred_path, index=False)
            print(f"  预测保存: {pred_path}")

        # 保存回测结果
        bt_result = r.get("result", {})
        if bt_result:
            ret_path = OUTPUT_DIR / f"ml_backtest_returns_{exp_id}.csv"
            ret_df = pd.DataFrame({
                "date": bt_result["gross_returns"].index,
                "gross_return": bt_result["gross_returns"].values,
                "net_return": bt_result["net_returns"].values,
                "turnover": bt_result["turnovers"].values,
            })
            ret_df.to_csv(ret_path, index=False, encoding="utf-8-sig")

    # ── 总结 ──
    print(f"\n{'=' * 64}")
    print("消融实验完成")
    print(f"{'=' * 64}")

    for exp_id in exp_ids:
        nm = (results.get(exp_id, {}).get("result", {}).get("net_metrics") or {})
        sr = nm.get("Sharpe_Ratio", 0)
        to = results.get(exp_id, {}).get("result", {}).get("avg_turnover", 0)
        print(f"  {exp_id}: Sharpe={sr:.4f} | Turnover={to*100:.1f}% | "
              f"时间={results[exp_id].get('wall_time_sec', 0):.0f}s")

    print(f"\n输出文件:")
    print(f"  - output/ml_ablation_report.md")
    for exp_id in exp_ids:
        print(f"  - output/ml_predictions_{exp_id}.parquet")
        print(f"  - output/ml_backtest_returns_{exp_id}.csv")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
