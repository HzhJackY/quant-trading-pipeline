# README 中文当前状态重写 v1 变更日志

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
