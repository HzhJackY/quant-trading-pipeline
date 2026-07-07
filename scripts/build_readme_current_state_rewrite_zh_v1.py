from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "readme_current_state_rewrite_zh_v1"


def exists(rel_path: str) -> bool:
    return (ROOT / rel_path).exists()


def build_project_structure_section() -> str:
    entries: list[tuple[str, str]] = [
        ("factor_research/", "因子分析、成本回测、择时 baseline、回测引擎"),
        ("paper_trading/", "paper trading（纸交易）生产管线"),
        ("monitoring/", "Streamlit 每日风控看板"),
        ("factor_lib/", "因子定义与扩展因子模块"),
        ("data/", "数据抓取、缓存与清洗模块"),
        ("output/", "报告、模型、回测结果与研究产物"),
        ("scripts/", "一次性脚本、审计脚本、报告生成脚本"),
        ("tests/", "单元测试"),
    ]
    optional_entries: list[tuple[str, str]] = [
        ("xhs/", "另类数据研究工作区"),
        ("research/", "研究草稿与补充材料"),
        ("MediaCrawler/", "另类数据采集实验目录"),
    ]
    files: list[tuple[str, str]] = [
        ("run_compact_f_production_validation.py", "当前生产候选验证入口"),
        ("run_v15_portfolio_optimization.py", "组合层优化与 Top50 Buffer 评估"),
        ("run_backtest_with_costs.py", "成本感知回测"),
        ("run_timing_comparison.py", "择时 baseline 对比"),
        ("requirements.txt", "环境依赖"),
        ("README.md", "当前英文 README 原文件，保留待人工合并"),
    ]

    lines = ["```text", "quant/"]
    for path, desc in entries:
        if exists(path):
            lines.append(f"├── {path:<32} # {desc}")
    for path, desc in optional_entries:
        if exists(path):
            lines.append(f"├── {path:<32} # {desc}（独立研究或辅助目录）")
    for path, desc in files[:-1]:
        if exists(path):
            lines.append(f"├── {path:<32} # {desc}")
    if exists(files[-1][0]):
        lines.append(f"└── {files[-1][0]:<32} # {files[-1][1]}")
    lines.append("```")
    return "\n".join(lines)


def build_readme_draft() -> str:
    structure_block = build_project_structure_section()
    archive_items = [
        "- Split-Universe 双模型系统与相关回测材料",
        "- V0–V7 ML 实验链，包括 LambdaRank、TO-Aware、Label Blending",
        "- V1→V2 Alpha Drift 调查与 GS / colsample / BP / ProfitGrowth drift RCA",
        "- V1.5 ideas、Phase B rebuild、V1 vs V2_Full model comparison",
    ]
    archive_text = "\n".join(archive_items)

    return f"""# A 股基本面量化选股与纸交易系统

Compact-F 纯基本面模型 + Top50 Buffer 生产候选，配套 paper trading（纸交易）、成本回测、风控看板与历史研究归档。

## 当前状态

- 传统基本面 alpha 主线已封箱。
- 默认生产候选为 `Compact-F + Top50 Buffer 35/75`。
- paper trading（纸交易）与 monitoring（监控）模块服务于当前生产候选与模拟运行。
- 大盘择时当前仅作为 baseline / monitor-only（仅监控），不作为默认生产仓位控制。
- Media15 / XHS / 百度指数属于独立另类数据研究线，不接入 Compact-F。
- 历史 `V0–V7`、`Split-Universe`、`Alpha Drift` 材料保留用于追溯、面试叙事和方法论展示。

## 核心结论

当前生产候选：`Compact-F + Top50 Buffer 35/75`

### 组合层验证

| 方案 | Sharpe | MaxDD | 换手率 | ROE 暴露 | ProfitGrowth 暴露 | EP 暴露 |
|---|---:|---:|---:|---:|---:|---:|
| Top30 baseline | 0.4117 | -31.77% | 45.95% | +0.640 | +0.106 | +0.436 |
| Top50 Buffer 35/75 | 0.4132 | -31.29% | 28.04% | +0.648 | +0.108 | +0.385 |

结论：Top50 Buffer 在 Sharpe 和风格暴露基本不损失的情况下，显著降低换手率，是当前默认生产组合候选。

### Compact-F 与技术因子扩展对比

| 模型 | Sharpe | MaxDD | Turnover | ROE | ProfitGrowth |
|---|---:|---:|---:|---:|---:|
| Compact-F | 0.412 | -31.8% | 46.0% | +0.640 | +0.106 |
| Compact-FT | 0.435 | -42.0% | 64.5% | +0.345 | +0.072 |
| Compact-FT3 | 0.395 | -44.7% | 63.7% | +0.322 | +0.082 |

结论：Compact-FT 虽然略微提高 Sharpe，但显著恶化回撤和换手，并稀释基本面风格。技术因子劫持仍然存在，因此生产推荐保持 Compact-F。

## 项目结构

{structure_block}

说明：

- 上述结构只突出当前活跃模块与重要归档入口。
- 历史脚本与研究报告仍保留在仓库中，但不再作为当前主线说明。

## 当前生产候选：Compact-F + Top50 Buffer

### 模型

- `Compact-F` 是纯基本面模型。
- 信号频率为月频。
- 股票池对齐到 `CSI800`。
- 不继续进行技术因子生产调参。

### 组合

- 组合层采用 `Top50 Buffer`。
- 买入阈值：`rank <= 35`。
- 卖出阈值：`rank > 75`。
- 月频调仓，调仓点为月末。
- 默认等权，除非执行层另有说明。

### 为什么选择 Top50 Buffer

- 与 Top30 的 Sharpe 接近。
- 换手显著更低。
- 成本敏感性更稳。
- 风格暴露保持 `ROE / ProfitGrowth / EP` 正向。

### 明确不是生产主线

- `Compact-FT / Compact-FT3` 未入选生产候选。
- 大盘择时不是默认仓位控制。
- `Media15 / XHS / 百度指数` 不接入 Compact-F。
- `V1/V2 RCA` 不再作为生产调参路径。

## Paper Trading（纸交易）

当前纸交易入口：`paper_trading/paper_trading_pipeline.py`

定位：

- 服务于生产候选的模拟运行与执行编排。
- 月末调仓。
- 先做物理风控过滤，再做 universe alignment（股票池对齐）与信号输出。

当前实现中的物理风控过滤包括：

- `ST / *ST`
- 停牌
- 日均成交额低于 5000 万
- 总市值低于 50 亿

当前实现中的核心链路：

- `CSI800 universe alignment`
- `cross_sectional_rank` 只在最终 universe 内计算
- `signal anchor` 持久化上期与当期信号

实现一致性说明：

- 当前纸交易实现仍可能以 `Top30` 为输出口径；生产候选已更新为 `Top50 Buffer`。该不一致应在 production governance（生产治理）阶段对齐。

常用命令：

```bash
pip install -r requirements.txt
python paper_trading/paper_trading_pipeline.py --date 2026-06-30 --force-rebalance -v
python paper_trading/paper_trading_pipeline.py --date 2026-06-30 --force-rebalance --force-refresh
python paper_trading/paper_trading_pipeline.py --date 2026-06-30 --force-rebalance --skip-ingestion
```

## Monitoring（监控看板）

当前监控入口：`monitoring/daily_report.py`

看板内容包括：

- 风控雷达
- KPI 卡片
- 累计净值
- 因子暴露
- 红黑榜
- 全景持仓

工程栈：

- `Baostock + SQLite + Streamlit`

使用方式：

```bash
streamlit run monitoring/daily_report.py
```

说明：

- 监控看板用于纸交易与风险复核，不等同于 alpha 证明。

## 大盘择时 baseline：仅监控，不作为默认生产仓位

当前规则：

- 中证 500 `MA20/60` 死叉，或 20 日年化波动率超过 252 日 80% 分位时，仓位乘数降为 `0.3`。

回测结果：

- 无择时：Net Sharpe `1.13`，年化收益 `21.27%`，MaxDD `-18.01%`
- 有择时：Net Sharpe `0.80`，年化收益 `10.66%`，MaxDD `-12.50%`

结论：

- 该择时方案可以降低回撤，但收益损失过大，Sharpe 恶化。
- 当前参数不具备实盘默认价值。
- 除非显式进行择时实验，否则生产默认 `multiplier` 应保持 `1.0`。

相关入口：

```bash
python run_timing_comparison.py
python run_backtest_with_costs.py
```

## 另类数据研究：独立研究线

- `Media15 / XHS / 百度指数` 是独立研究线。
- 它们不修改 Compact-F。
- `media_neg_share_all` 可以作为 risk diagnostic filter（风险提示变量）或风险复核信号。
- 原始 `XHS attention` 不应直接作为因子。
- 百度指数更适合作为较弱的外部搜索关注对照。
- 当前不声称这些变量已经形成可交易 alpha。
- 当前不接入 Compact-F。

## 历史研究归档

以下材料用于研究追溯、面试叙事和方法论展示，不代表当前生产调参路线。

{archive_text}

代表性归档文件：

- `output/ml_v7_final_report.md`
- `output/V1_to_V2_alpha_drift_investigation_final.md`
- `run_split_universe.py`
- `run_ml_v7.py`
- `run_model_comparison.py`

## Roadmap

### P0：生产治理

- 冻结生产规格：`Compact-F + Top50 Buffer 35/75`
- 如 paper trading 仍为 `Top30`，需与 `Top50 Buffer` 对齐
- 增加 model registry / model hash
- 增加 production QA gate
- 增加月度调仓报告

### P1：纸交易执行校准

- 记录理论成交价 vs 可成交价
- 处理涨跌停、停牌、100 股整数手约束
- 记录成交失败
- 用纸交易日志校准滑点

### P1：信号质量与风格监控

- `Top/Bottom spread`
- trailing `3/6/12 month Rank IC`
- `alpha_signal` 分布漂移
- style exposure drift vs `Compact-F` baseline

### P2：研究分支

- sector-relative growth factors
- 线性信号并行对照
- 分级择时作为实验，不作为默认生产
- `Media15 risk flag` 接入 dashboard，但不作为 alpha

## Quick Start

1. 安装环境

```bash
pip install -r requirements.txt
```

2. 纸交易 dry run / force rebalance

```bash
python paper_trading/paper_trading_pipeline.py --date 2026-06-30 --force-rebalance -v
```

3. 监控看板

```bash
streamlit run monitoring/daily_report.py
```

4. 择时 baseline 对比

```bash
python run_timing_comparison.py
```

5. 成本感知回测

```bash
python run_backtest_with_costs.py
```

6. 单元测试

```bash
pytest tests/ -v
```

说明：

- `run_ml_v7.py`、`run_split_universe.py`、`run_model_comparison.py` 不再属于 Quick Start 主流程，应视为历史研究归档入口。
"""


def build_change_log() -> str:
    return """# README 中文当前状态重写 v1 变更日志

## 改写目标

- 将 README 主体从旧的 Split-Universe / V0–V7 / ML V7 主线，改写为 2026-06 当前真实状态。
- 明确默认生产候选是 `Compact-F + Top50 Buffer 35/75`。
- 保留历史研究痕迹，但统一降级到“历史研究归档”。

## 主要改动

### 1. 标题与定位

- 旧定位：A 股多因子选股系统，核心系统是 Split-Universe 双模型架构。
- 新定位：A 股基本面量化选股与纸交易系统，主线是 `Compact-F + Top50 Buffer`。

### 2. 核心结论

- 删除旧的 `V0 Linear vs ML V7` 核心对比表。
- 改为当前生产候选、Top30 baseline、Top50 Buffer 35/75、Compact-F vs Compact-FT / FT3 的对比。

### 3. 项目结构

- 从“大量历史脚本逐一列举”改为“当前活跃模块 + 关键入口 + 历史归档说明”。
- 保留 `paper_trading`、`monitoring`、`factor_research`、`output` 等核心目录。

### 4. 生产候选章节

- 新增“当前生产候选：Compact-F + Top50 Buffer”。
- 明确模型、组合、选择原因，以及哪些方向不再属于生产主线。

### 5. Paper Trading 与 Monitoring

- 保留工程模块，但改写为“服务于当前生产候选”。
- 标出当前 paper trading 仍可能是 `Top30` 输出口径，与生产候选 `Top50 Buffer` 需要治理对齐。

### 6. Market Timing 降级

- 保留择时模块和历史结果。
- 明确其状态是 `baseline / monitor-only`，不是默认生产仓位控制。

### 7. 另类数据

- 新增“另类数据研究：独立研究线”。
- 明确 `Media15 / XHS / 百度指数` 不接入 Compact-F，不声称已形成可交易 alpha。

### 8. 历史研究归档

- 将 `Split-Universe`、`V0–V7`、`Alpha Drift`、`V1.5 ideas`、`Phase B rebuild` 等统一收纳到历史归档。

### 9. Roadmap

- 删除旧的 `V1.5`、`GS softening`、`technical tuning` 等 P0 / P1 描述。
- 重写为生产治理、纸交易执行校准、风格监控与独立研究分支。

### 10. Quick Start

- 改成当前实际常用入口：环境安装、paper trading、monitoring、择时 baseline、成本回测、测试。
- 不再把 `run_ml_v7.py`、`run_split_universe.py`、`run_model_comparison.py` 放在主流程。

## 明确保留但降级的内容

- 历史研究材料仍保留在仓库中。
- 原 README.md 不被覆盖。
- 本次输出仅为中文草稿、变更日志和一致性审计。
"""


def build_audit_rows() -> list[dict[str, str]]:
    return [
        {
            "issue": "Split-Universe 主线降级",
            "old_text_or_section": "开头定位 / 项目结构 / Split-Universe 双模型系统",
            "problem": "旧 README 将 Split-Universe 描述为当前核心系统",
            "new_treatment": "移入历史研究归档，不再作为当前生产核心",
            "pass": "True",
        },
        {
            "issue": "V7 主线降级",
            "old_text_or_section": "核心结论 / Runner 脚本指南 / 快速开始",
            "problem": "旧 README 将 ML V7 表述为终版生产模型",
            "new_treatment": "移入历史研究归档，不再作为当前生产模型",
            "pass": "True",
        },
        {
            "issue": "生产候选显式化",
            "old_text_or_section": "README 开头状态说明与主体不一致",
            "problem": "旧主体没有把 Compact-F + Top50 Buffer 写成核心结论",
            "new_treatment": "在当前状态、核心结论、生产候选章节中显式写入",
            "pass": "True",
        },
        {
            "issue": "Top50 Buffer 指标纳入",
            "old_text_or_section": "核心结论",
            "problem": "旧 README 缺少 Top50 Buffer 35/75 指标",
            "new_treatment": "新增 Top30 baseline vs Top50 Buffer 35/75 表格",
            "pass": "True",
        },
        {
            "issue": "Compact-FT / FT3 拒绝入选",
            "old_text_or_section": "旧 README 无该结论",
            "problem": "未明确技术因子扩展不作为生产推荐",
            "new_treatment": "新增 Compact-F vs Compact-FT / FT3 对比与拒绝结论",
            "pass": "True",
        },
        {
            "issue": "Market timing 降级",
            "old_text_or_section": "大盘择时章节",
            "problem": "旧 README 容易让人误解为生产默认仓位控制",
            "new_treatment": "标题改为仅监控，不作为默认生产仓位",
            "pass": "True",
        },
        {
            "issue": "Media15 / XHS / 百度指数独立化",
            "old_text_or_section": "旧 README 无独立研究线边界",
            "problem": "容易被误读为可直接接入主线 alpha",
            "new_treatment": "新增另类数据研究章节，明确不接入 Compact-F",
            "pass": "True",
        },
        {
            "issue": "V1.5 不再是 P0",
            "old_text_or_section": "后续优化方向 / Roadmap",
            "problem": "旧 README 仍将 V1.5、GS softening、technical tuning 列为 P0/P1",
            "new_treatment": "Roadmap 改写为生产治理、执行校准、风格监控、研究分支",
            "pass": "True",
        },
        {
            "issue": "paper_trading Top30 与 production Top50 不一致标注",
            "old_text_or_section": "纸交易章节",
            "problem": "实现口径仍可能是 Top30，若不标注会造成治理歧义",
            "new_treatment": "明确写出当前不一致，并要求后续 production governance 对齐",
            "pass": "True",
        },
        {
            "issue": "历史研究归档保留",
            "old_text_or_section": "Split-Universe / V0-V7 / Alpha Drift / Phase B",
            "problem": "不能删除历史研究痕迹，但必须降级",
            "new_treatment": "集中放入历史研究归档章节，保留追溯用途说明",
            "pass": "True",
        },
    ]


def infer_decision(readme_text: str) -> str:
    if "Compact-F + Top50 Buffer 35/75" not in readme_text:
        return "README_ZH_NEEDS_PRODUCTION_CANDIDATE_FIX"
    if "Split-Universe 双模型系统" in readme_text and "历史研究归档" not in readme_text:
        return "README_ZH_NEEDS_MAINLINE_FIX"
    if "当前纸交易实现仍可能以 `Top30` 为输出口径；生产候选已更新为 `Top50 Buffer`" not in readme_text:
        return "README_ZH_NEEDS_PAPER_TRADING_ALIGNMENT_NOTE"
    return "README_ZH_DRAFT_READY_FOR_REVIEW"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    readme_draft_path = OUTPUT_DIR / "README_current_state_draft_zh.md"
    change_log_path = OUTPUT_DIR / "README_change_log_zh.md"
    consistency_audit_path = OUTPUT_DIR / "README_consistency_audit_zh.csv"

    draft_text = build_readme_draft()
    change_log_text = build_change_log()
    audit_rows = build_audit_rows()
    decision = infer_decision(draft_text)

    readme_draft_path.write_text(draft_text, encoding="utf-8")
    change_log_path.write_text(change_log_text, encoding="utf-8")

    with consistency_audit_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["issue", "old_text_or_section", "problem", "new_treatment", "pass"],
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    print(f"readme_draft_path={readme_draft_path.as_posix()}")
    print(f"change_log_path={change_log_path.as_posix()}")
    print(f"consistency_audit_path={consistency_audit_path.as_posix()}")
    print(f"decision={decision}")


if __name__ == "__main__":
    main()
