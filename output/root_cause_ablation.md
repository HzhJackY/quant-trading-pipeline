# Root Cause Ablation Report

## 摘要

| 模型 | RankCorr | Top30Overlap | IC | IC_IR | ImpliedTO |
|------|----------|-------------|-----|-------|-----------|
| V1 (production) | 0.984 | 81% | 0.0582 | 0.539 | 14.5% |
| V2_Full (production) | 0.718 | 34% | 0.0615 | 0.538 | 38.3% |

---
## Experiment A: colsample_bytree

| colsample | RankCorr | Top30Overlap | IC | IC_IR | ImpliedTO | TrainTime |
|-----------|----------|-------------|-----|-------|-----------|-----------|
| 0.50 | 0.8391 | 71.4% | 0.1036 | 0.5435 | 28.6% | 2s |
| 0.70 | 0.7442 | 40.2% | 0.1212 | 0.7070 | 59.8% | 1s |
| 0.90 | 0.6857 | 41.4% | 0.1200 | 0.7282 | 58.6% | 1s |
| 1.00 | 0.7483 | 69.5% | 0.1031 | 0.5825 | 30.5% | 1s |

**colsample 0.50→1 效果**: RankCorr -0.0908, ImpliedTO -1.9%

---
## Experiment B: GS Orthogonalization

| GS Mode | RankCorr | Top30Overlap | IC | IC_IR | ImpliedTO |
|---------|----------|-------------|-----|-------|-----------|
| GS_ON | 0.8391 | 71.4% | 0.1036 | 0.5435 | 28.6% |
| GS_OFF | 0.8655 | 62.5% | 0.1238 | 0.6749 | 37.5% |

**GS Effect**: RankCorr -0.0264, ImpliedTO -8.9%

---
## Experiment C: Universe Size

| Universe | RankCorr | Top30Overlap | IC | IC_IR | ImpliedTO |
|----------|----------|-------------|-----|-------|-----------|
| Full_CSI800 | 0.8391 | 71.4% | 0.1036 | 0.5435 | 28.6% |
| Sampled_300 | 0.8154 | 23.8% | 0.1480 | 1.0420 | 76.2% |

**Universe Effect**: RankCorr +0.0236, ImpliedTO -47.6%

---
## Root Cause Attribution

### RankCorr下降 (0.984→0.718, Δ=-0.266) 归因

| Rank | Factor | Absolute ΔRankCorr | % Contribution |
|------|--------|-------------------|----------------|
| 1 | colsample_bytree | 0.0908 | 64.5% |
| 2 | GS正交化 | 0.0264 | 18.7% |
| 3 | Universe扩大 | 0.0236 | 16.8% |

| — | Unexplained (interaction effects) | 0.1252 | 47.1% |

---
## Questions Answered

### Q1: RankCorr下降贡献排序

1. **colsample_bytree**: 0.091 (64%) — **主要驱动因素**
2. **GS正交化**: 0.026 (19%) — 显著贡献
3. **Universe扩大**: 0.024 (17%) — 显著贡献

### Q2: colsample贡献占比
colsample 从 0.50→1 可恢复 **0.091** RankCorr (占Δ的 34%)

### Q3: GS正交化贡献占比
GS正交化贡献 **0.026** RankCorr (占Δ的 10%)

### Q4: Universe扩大贡献占比
Universe扩大贡献 **0.024** RankCorr (占Δ的 9%)

### Q5: 最值得优先修复的变量

**colsample_bytree** — 贡献了 64% 的 RankCorr 下降, 且修复成本最低(仅需修改一个参数)

---
*报告生成: 2026-06-21 13:15:17.203611*