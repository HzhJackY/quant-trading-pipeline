# ═══════════════════════════════════════════════════════════════
# ARCHIVED: V5 Turnover-Aware 回测 runner
# LightGBMAlphaEngineV5 已被 V7 取代, λ sweep 实验已迁移至 V7
# 保留作为消融实验记录 — 不建议用于新工作
# ═══════════════════════════════════════════════════════════════
"""
Turnover-Aware Custom Objective 回测 — 运行脚本 (Stage 5).

将 LightGBMAlphaEngineV5 (Turnover-Aware L2) 训练的 inertia_ml_signal
接入交易成本感知回测, 对比 λ 惩罚强度对换手率和 Sharpe 的影响。

核心机制:
  - Custom Objective: L = ½(ŷ−y)² + λ·½(ŷ−ŷ₋₁)²
  - 闭包传递 prev_signal (= alpha_signal_{t-1}) 到 LightGBM fobj
  - subsample=1.0 保证数据对齐
  - 3M forward return label + 3M Gap 防泄漏

λ 消融实验:
  - λ=0.0  → 纯 L2 (等价 V1 但用 3M label + Gap)
  - λ=0.1  → 轻微惩罚
  - λ=0.5  → 推荐初始值 (温和)
  - λ=1.0  → 强惩罚
  - λ=2.0  → 极强惩罚

用法:
  python run_ml_turnover_aware.py
  python run_ml_turnover_aware.py --lambda-only 0.5  # 仅运行 λ=0.5
  python run_ml_turnover_aware.py --skip-baselines    # 跳过 V0/V1
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
logger = logging.getLogger("run_ml_turnover_aware")

from factor_research.ml_engine_v5 import LightGBMAlphaEngineV5, MLConfigV5
from factor_research.ml_engine import LightGBMAlphaEngine, MLConfig
from factor_research.backtest_engine import (
    run_backtest_with_costs,
)
from factor_research.transaction_cost import TieredCostModel, UniverseCostConfig

OUTPUT_DIR = Path("output")


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════


def load_data():
    preprocessed_path = OUTPUT_DIR / "preprocessed.parquet"
    blended_path = OUTPUT_DIR / "split_universe_blended.parquet"

    for p in [preprocessed_path, blended_path]:
        if not p.exists():
            raise FileNotFoundError(f"未找到 {p}。请先运行前置 Pipeline。")

    panel = pd.read_parquet(preprocessed_path)
    blended = pd.read_parquet(blended_path)
    print(f"加载: panel {panel.shape[0]:,} 行 × {panel.shape[1]} 列 | "
          f"{panel['date'].nunique()} 截面")
    print(f"加载: blended {blended.shape[0]:,} 行")
    return panel, blended


def build_cost_model(aum: float = 50_000_000) -> TieredCostModel:
    return TieredCostModel(
        aum=aum,
        large_cap_config=UniverseCostConfig(
            commission_bps=2.5, stamp_duty_bps=5.0, transfer_fee_bps=0.1,
            base_slippage_bps=5.0, impact_gamma=0.5, impact_eta=1.0,
        ),
        small_cap_config=UniverseCostConfig(
            commission_bps=2.5, stamp_duty_bps=5.0, transfer_fee_bps=0.1,
            base_slippage_bps=15.0, impact_gamma=0.65, impact_eta=1.5,
        ),
    )


def fmt_pct(v, decimals=2):
    if v is None or np.isnan(v):
        return "N/A"
    return f"{v * 100:.{decimals}f}%"

def fmt_num(v, decimals=4):
    if v is None or np.isnan(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def run_backtest_for_signal(
    panel: pd.DataFrame,
    blended: pd.DataFrame,
    cost_model: TieredCostModel,
    signal_df: pd.DataFrame | None,
    alpha_col: str,
    label: str,
    top_quantile: float = 0.3,
    min_stocks: int = 5,
) -> dict:
    """对单个信号执行回测, 返回结果 dict。"""
    t0 = time.perf_counter()

    if signal_df is not None:
        bt_blended = blended.merge(
            signal_df[["date", "symbol", alpha_col]],
            on=["date", "symbol"], how="left",
        )
        bt_blended[alpha_col] = bt_blended[alpha_col].fillna(0.5)
    else:
        bt_blended = blended

    result = run_backtest_with_costs(
        panel=panel,
        blended=bt_blended,
        cost_model=cost_model,
        top_quantile=top_quantile,
        min_stocks_per_universe=min_stocks,
        alpha_col=alpha_col,
    )

    elapsed = time.perf_counter() - t0
    nm = result.get("net_metrics") or {}
    print(f"  [{label}] {elapsed:.0f}s | "
          f"Sharpe={nm.get('Sharpe_Ratio', 0):.4f} | "
          f"TO={result.get('avg_turnover', 0)*100:.1f}% | "
          f"Cost={result.get('avg_cost_bps', 0):.1f}bps")
    return {"result": result, "wall_time_sec": elapsed}


# ═══════════════════════════════════════════════════════════
# V5 训练 + 回测
# ═══════════════════════════════════════════════════════════


def train_and_backtest_v5(
    panel: pd.DataFrame,
    blended: pd.DataFrame,
    cost_model: TieredCostModel,
    lambda_val: float,
    top_quantile: float = 0.3,
    min_stocks: int = 5,
    label: str | None = None,
) -> dict:
    """
    训练一个 V5 模型 (指定 λ) 并回测。
    """
    exp_label = label or f"V5_λ={lambda_val}"
    print(f"\n{'─' * 56}")
    print(f"[{exp_label}] Turnover-Aware L2 (λ={lambda_val})")
    print(f"{'─' * 56}")

    # ── 配置 ──
    config = MLConfigV5(
        train_months=36,
        val_months=6,
        test_months=1,
        label_horizon=3,
        lambda_turnover=lambda_val,
        max_depth=4,
        num_leaves=24,
        learning_rate=0.02,
        n_estimators=2000,
        subsample=1.0,           # 闭包对齐
        colsample_bytree=0.70,
        min_child_samples=100,
        reg_alpha=0.10,
        reg_lambda=0.10,
        early_stopping_rounds=50,
    )

    # ── 训练 ──
    engine = LightGBMAlphaEngineV5(config=config)
    t0 = time.perf_counter()
    predictions = engine.run(panel, blended=blended)
    train_time = time.perf_counter() - t0

    # ── 保存预测 ──
    pred_path = OUTPUT_DIR / f"ml_v5_predictions_lambda{lambda_val:.1f}.parquet"
    predictions.to_parquet(pred_path, index=False)
    print(f"  预测保存: {pred_path} ({len(predictions)} 条, {predictions['date'].nunique()} 截面)")

    # ── 保存训练报告 ──
    report = engine.to_markdown_report()
    (OUTPUT_DIR / f"ml_v5_report_lambda{lambda_val:.1f}.md").write_text(
        report, encoding="utf-8")

    # ── 回测 ──
    result = run_backtest_for_signal(
        panel, blended, cost_model, predictions,
        alpha_col="inertia_ml_signal",
        label=exp_label,
        top_quantile=top_quantile,
        min_stocks=min_stocks,
    )
    result["train_time_sec"] = train_time
    result["predictions"] = predictions
    result["feature_importance"] = engine.get_feature_importance()

    return result


# ═══════════════════════════════════════════════════════════
# 对比表生成
# ═══════════════════════════════════════════════════════════


def generate_comparison_table(
    results: dict[str, dict],
    lambda_values: list[float],
    aum: float,
) -> str:
    """生成 λ 消融实验对比表 (Markdown)。"""

    def get_metric(exp_id, key):
        r = results.get(exp_id, {}).get("result", {})
        nm = r.get("net_metrics") or {}
        return nm.get(key)

    def get_to(exp_id):
        r = results.get(exp_id, {}).get("result", {})
        return r.get("avg_turnover", 0)

    def get_cost(exp_id):
        r = results.get(exp_id, {}).get("result", {})
        return r.get("avg_cost_bps", 0)

    # ── 表头 ──
    lambda_labels = [f"V5_λ={lv}" for lv in lambda_values]
    all_ids = ["V0_Linear", "V1_L2"] + lambda_labels

    header = [
        f"## Turnover-Aware Custom Objective — λ 消融实验",
        f"",
        f"- **AUM:** {aum/1e4:.0f} 万",
        f"- **选股比例:** Top 30% 分域等权",
        f"- **成本模型:** Almgren-Chriss 冲击 + 分域费率",
        f"- **Label:** forward_return_3M → 截面 Rank [0,1]",
        f"- **Gap:** 3M (防多期标签泄漏)",
        f"- **prev_signal 锚点:** alpha_signal_{{t-1}} (线性基准)",
        f"",
        f"### 损失函数",
        f"",
        f"```",
        f"L = ½(ŷ − y)²  +  λ·½(ŷ − ŷ_{{t−1}})²",
        f"```",
        f"",
        f"### 扣费后绩效对比",
        f"",
        f"| 指标 | V0: 线性 | V1: L2 (1M, no gap) |",
    ]
    for lv in lambda_values:
        header[-1] += f" V5: λ={lv} |"
    header.append("|------|:---:|:---:|" + ":---:|" * len(lambda_values))

    # ── 绩效指标 ──
    metrics = [
        ("年化收益", "Annualized_Return", fmt_pct),
        ("年化波动率", "Volatility", fmt_pct),
        ("**Sharpe Ratio**", "Sharpe_Ratio", fmt_num),
        ("最大回撤", "Max_Drawdown", fmt_pct),
        ("Calmar Ratio", "Calmar_Ratio", fmt_num),
        ("月胜率", "Win_Rate", fmt_pct),
    ]

    for label, key, formatter in metrics:
        vals = [formatter(get_metric(eid, key)) for eid in all_ids]
        header.append(f"| {label} | " + " | ".join(vals) + " |")

    # ── 交易特征 ──
    header.append("")
    header.append("### 交易特征")
    header.append("")
    header.append("| 指标 | V0: 线性 | V1: L2 |" + "".join(f" V5: λ={lv} |" for lv in lambda_values))
    header.append("|------|:---:|:---:|" + ":---:|" * len(lambda_values))

    for label, key in [
        ("月均单边换手率", "avg_turnover"),
        ("月均总成本 (bps)", "avg_cost_bps"),
    ]:
        vals = []
        for eid in all_ids:
            r = results.get(eid, {}).get("result", {})
            v = r.get(key, 0)
            if key == "avg_turnover":
                vals.append(f"{v*100:.1f}%")
            else:
                vals.append(f"{v:.1f}")
        header.append(f"| {label} | " + " | ".join(vals) + " |")

    # ── λ 惩罚效果分析 ──
    header.append("")
    header.append("### λ 惩罚强度 vs 换手率 / Sharpe")
    header.append("")
    header.append("| λ | 换手率 | Δ vs V1 | Sharpe | Δ vs V1 | 年化成本 |")
    header.append("|------|--------|---------|--------|---------|----------|")
    v1_to = get_to("V1_L2") * 100
    v1_sr = get_metric("V1_L2", "Sharpe_Ratio") or 0
    for lv in lambda_values:
        eid = f"V5_λ={lv}"
        to_val = get_to(eid) * 100
        sr_val = get_metric(eid, "Sharpe_Ratio") or 0
        cost_val = get_cost(eid)
        to_delta = to_val - v1_to
        sr_delta = sr_val - v1_sr
        to_pct = f"{to_delta/v1_to*100:+.0f}%" if v1_to > 0 else "N/A"
        header.append(
            f"| {lv:.1f} | {to_val:.1f}% | "
            f"{to_delta:+.1f}% ({to_pct}) | "
            f"{sr_val:.4f} | {sr_delta:+.4f} | {cost_val:.1f} bps |"
        )

    # ── 训练耗时 ──
    header.append("")
    header.append("### 训练耗时")
    header.append("")
    header.append("| 配置 | 耗时 |")
    header.append("|------|------|")
    for eid in all_ids:
        r = results.get(eid, {})
        wt = r.get("wall_time_sec", r.get("train_time_sec", 0))
        header.append(f"| {eid} | {wt:.0f}s |")

    return "\n".join(header)


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Turnover-Aware Custom Objective 回测")
    parser.add_argument("--lambda-only", type=float, default=None,
                        help="仅运行指定 λ 值 (单值)")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="跳过 V0 和 V1")
    parser.add_argument("--aum", type=float, default=50_000_000)
    parser.add_argument("--top", type=float, default=0.3)
    parser.add_argument("--min-stocks", type=int, default=5)
    args = parser.parse_args()

    print("=" * 64)
    print("Turnover-Aware Custom Objective -- lambda Ablation (Stage 5)")
    print("=" * 64)
    print()
    print("Loss: L = 0.5*(pred-y)^2 + lambda*0.5*(pred-prev)^2")
    print("Anchor: prev_signal = alpha_signal_{t-1} (linear baseline)")
    print("Injection: closure -> LightGBM custom objective")
    print("=" * 64)

    # ── 加载数据 ──
    panel, blended = load_data()
    cost_model = build_cost_model(aum=args.aum)

    # ── 确定 λ 搜索范围 ──
    if args.lambda_only is not None:
        lambda_values = [args.lambda_only]
    else:
        lambda_values = [0.0, 0.1, 0.5, 1.0, 2.0]

    results: dict[str, dict] = {}

    # ── V0: Linear Baseline ──
    if not args.skip_baselines:
        print(f"\n{'─' * 56}")
        print("[V0] 线性 alpha_signal (Baseline)")
        print(f"{'─' * 56}")
        results["V0_Linear"] = run_backtest_for_signal(
            panel, blended, cost_model, None, "alpha_signal", "V0",
            top_quantile=args.top,
            min_stocks=args.min_stocks,
        )

    # ── V1: L2 LightGBM (cached) ──
    if not args.skip_baselines:
        cached_pred = OUTPUT_DIR / "ml_signal_predictions.parquet"
        if cached_pred.exists():
            print(f"\n{'─' * 56}")
            print("[V1] L2 LightGBM (cached)")
            print(f"{'─' * 56}")
            preds = pd.read_parquet(cached_pred)
            results["V1_L2"] = run_backtest_for_signal(
                panel, blended, cost_model, preds, "ml_signal", "V1",
                top_quantile=args.top,
                min_stocks=args.min_stocks,
            )
        else:
            print(f"\n{'─' * 56}")
            print("[V1] L2 LightGBM (training...)")
            print(f"{'─' * 56}")
            engine_v1 = LightGBMAlphaEngine(config=MLConfig(seeds=[42]))
            preds = engine_v1.run(panel)
            preds.to_parquet(OUTPUT_DIR / "ml_signal_predictions.parquet", index=False)
            results["V1_L2"] = run_backtest_for_signal(
                panel, blended, cost_model, preds, "ml_signal", "V1",
                top_quantile=args.top,
                min_stocks=args.min_stocks,
            )

    # ── V5: λ 消融 ──
    for lv in lambda_values:
        r = train_and_backtest_v5(
            panel=panel,
            blended=blended,
            cost_model=cost_model,
            lambda_val=lv,
            top_quantile=args.top,
            min_stocks=args.min_stocks,
        )
        results[f"V5_λ={lv}"] = r

    # ── 对比表 ──
    print(f"\n{'=' * 64}")
    print("λ 消融实验报告")
    print(f"{'=' * 64}")

    table_md = generate_comparison_table(results, lambda_values, aum=args.aum)
    print(table_md)

    # 保存报告
    report_path = OUTPUT_DIR / "ml_v5_ablation_report.md"
    report_path.write_text(table_md, encoding="utf-8")
    print(f"\n报告已保存: {report_path}")

    # ── 保存各 λ 的预测 ──
    for lv in lambda_values:
        eid = f"V5_λ={lv}"
        r = results.get(eid, {})
        preds = r.get("predictions")
        if preds is not None and len(preds) > 0:
            pred_path = OUTPUT_DIR / f"ml_v5_predictions_lambda{lv:.1f}.parquet"
            preds.to_parquet(pred_path, index=False)

        bt_result = r.get("result", {})
        if bt_result:
            ret_path = OUTPUT_DIR / f"ml_v5_backtest_returns_lambda{lv:.1f}.csv"
            ret_df = pd.DataFrame({
                "date": bt_result.get("net_returns", pd.Series()).index,
                "net_return": bt_result.get("net_returns", pd.Series()).values,
                "turnover": bt_result.get("turnovers", pd.Series()).values,
            })
            if not ret_df.empty:
                ret_df.to_csv(ret_path, index=False, encoding="utf-8-sig")
                print(f"  回测结果保存: {ret_path}")

    # ── 总结 ──
    print(f"\n{'=' * 64}")
    print("Turnover-Aware 消融实验完成")
    print(f"{'=' * 64}")

    for eid in results:
        nm = (results.get(eid, {}).get("result", {}).get("net_metrics") or {})
        sr = nm.get("Sharpe_Ratio", 0)
        to = results.get(eid, {}).get("result", {}).get("avg_turnover", 0)
        wt = results.get(eid, {}).get("wall_time_sec",
               results.get(eid, {}).get("train_time_sec", 0))
        print(f"  {eid}: Sharpe={sr:.4f} | Turnover={to*100:.1f}% | 耗时={wt:.0f}s")

    print(f"\n输出文件:")
    print(f"  - output/ml_v5_ablation_report.md")
    for lv in lambda_values:
        print(f"  - output/ml_v5_predictions_lambda{lv:.1f}.parquet")
        print(f"  - output/ml_v5_report_lambda{lv:.1f}.md")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
