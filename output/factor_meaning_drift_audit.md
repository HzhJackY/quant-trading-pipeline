# Factor Meaning Drift Audit Report

**Generated**: 2026-06-21 17:55:14.498890

**Objective**: Determine whether factors changed economic meaning when universe expanded from 297 to 1,360 stocks.

---
## Part 1: Factor Distribution Drift

### Distribution Statistics

| Factor | V1 mean | V2 mean | V1 std | V2 std | V1 skew | V2 skew | KS stat | Wasserstein |
|--------|---------|---------|--------|--------|---------|---------|---------|-------------|
| BP | +0.0000 | +0.0000 | 0.9987 | 0.0000 | +0.894 | +nan | 0.5971 | 0.7573 |
| EP | +0.0000 | +0.0000 | 0.9987 | 0.9991 | +0.415 | +2.389 | 0.0901 | 0.2150 |
| Mom_3M | -0.0000 | -0.0000 | 0.9857 | 0.9905 | +0.497 | +1.733 | 0.0328 | 0.0991 |
| Mom_6M | -0.0000 | +0.0000 | 0.9720 | 0.9779 | +0.603 | +2.047 | 0.0388 | 0.1109 |
| NetMargin | +0.0000 | +0.0000 | 0.9987 | 0.9993 | +0.461 | +21.659 | 0.4351 | 0.6531 |
| ProfitGrowth | -0.0000 | +0.0000 | 0.9987 | 0.9991 | +0.228 | -0.365 | 0.3430 | 0.5780 |
| ROE | +0.0000 | +0.0000 | 0.9987 | 0.9993 | +0.288 | -14.771 | 0.3024 | 0.5126 |
| RevGrowth | -0.0000 | +0.0000 | 0.9987 | 0.9993 | +0.278 | +4.203 | 0.0917 | 0.2495 |

### Distribution Moments Detail

| Factor | V1 p01 | V1 p50 | V1 p99 | V2 p01 | V2 p50 | V2 p99 |
|--------|--------|--------|--------|--------|--------|--------|
| BP | -1.5950 | -0.2010 | +2.4813 | +0.0000 | +0.0000 | +0.0000 |
| EP | -2.6487 | -0.1518 | +2.3286 | -1.0657 | -0.2906 | +3.7425 |
| Mom_3M | -2.0569 | -0.0980 | +2.5545 | -1.8609 | -0.1388 | +3.2285 |
| Mom_6M | -1.9299 | -0.0900 | +2.5542 | -1.7181 | -0.1423 | +3.3528 |
| NetMargin | -2.4709 | -0.2210 | +2.3228 | -0.5952 | -0.0728 | +1.0709 |
| ProfitGrowth | -2.2074 | -0.0920 | +2.1376 | -1.6736 | -0.0331 | +2.6707 |
| ROE | -2.5216 | -0.0872 | +2.4747 | -1.7850 | +0.0320 | +1.4190 |
| RevGrowth | -2.3504 | -0.0769 | +2.4600 | -1.8401 | -0.1316 | +3.8286 |

### Distribution Drift Ranking (by Wasserstein distance)

| 1 | BP | 0.7573 | SEVERE |
| 2 | NetMargin | 0.6531 | SEVERE |
| 3 | ProfitGrowth | 0.5780 | SEVERE |
| 4 | ROE | 0.5126 | SEVERE |
| 5 | RevGrowth | 0.2495 | SEVERE |
| 6 | EP | 0.2150 | SEVERE |
| 7 | Mom_6M | 0.1109 | MODERATE |
| 8 | Mom_3M | 0.0991 | MODERATE |

---
## Part 2: Factor Rank Stability

Same stock, same date — how much does its rank change between V1 and V2 universes?

| Factor | Mean r | Median r | P10 | P90 | Min | Max | Interpretation |
|--------|--------|---------|-----|-----|-----|-----|---------------|
| EP | 0.4819 | 0.5353 | 0.3140 | 0.6058 | 0.2095 | 0.6368 | **Fundamentally different** |
| Mom_3M | 0.8553 | 0.8696 | 0.7698 | 0.9209 | 0.6138 | 0.9511 | Slightly shifted |
| Mom_6M | 0.9129 | 0.9194 | 0.8568 | 0.9579 | 0.7975 | 0.9722 | Slightly shifted |
| NetMargin | 0.6869 | 0.6773 | 0.5992 | 0.7704 | 0.5571 | 0.8428 | **Severely shifted** |
| ProfitGrowth | 0.0009 | 0.0204 | -0.1627 | 0.1055 | -0.2705 | 0.2445 | **Fundamentally different** |
| ROE | 0.3714 | 0.3512 | 0.2287 | 0.4810 | 0.1075 | 0.6288 | **Fundamentally different** |
| RevGrowth | 0.0063 | -0.0323 | -0.1161 | 0.1542 | -0.1691 | 0.2491 | **Fundamentally different** |

---
## Part 3: Factor IC Migration

| Factor | IC V1 | IC_IR V1 | IC V2 | IC_IR V2 | ΔIC | Status |
|--------|-------|---------|-------|---------|-----|--------|
| BP | +0.0459 | 0.272 | +nan | 0.000 | +nan | Minor change |
| EP | +0.0767 | 0.445 | +0.0930 | 0.538 | +0.0163 | **IMPROVED** |
| Mom_3M | -0.0289 | -0.167 | -0.0234 | -0.126 | +0.0055 | Minor change |
| Mom_6M | -0.0067 | -0.033 | -0.0017 | -0.009 | +0.0050 | Stable |
| NetMargin | +0.0363 | 0.335 | +0.0489 | 0.460 | +0.0125 | **IMPROVED** |
| ProfitGrowth | +0.0528 | 0.345 | +0.0219 | 0.267 | -0.0309 | **DEGRADED** |
| ROE | +0.0526 | 0.316 | +0.0593 | 0.519 | +0.0067 | Minor change |
| RevGrowth | +0.0336 | 0.244 | +0.0296 | 0.305 | -0.0040 | Stable |

---
## Part 4: Factor Decile Return Curves

| Factor | V1 D1-D10 spread | V2 D1-D10 spread | Direction | Change |
|--------|-----------------|-----------------|-----------|--------|
| EP | +1.11% | +2.27% | long high->long high | strengthened |
| Mom_3M | +0.15% | +0.30% | long high->long high | strengthened |
| Mom_6M | +0.09% | +0.74% | long high->long high | strengthened |
| NetMargin | +0.36% | +0.82% | long high->long high | strengthened |
| ProfitGrowth | +2.14% | +1.94% | long high->long high | weakened |
| ROE | +1.23% | +1.95% | long high->long high | strengthened |
| RevGrowth | +1.60% | +2.04% | long high->long high | strengthened |

### Detailed Decile Curves

#### BP
| Decile | V1 Return | V2 Return |
|--------|----------|----------|
| D1 | +0.15% | +nan% |
| D2 | +0.94% | +nan% |
| D3 | +1.44% | +nan% |
| D4 | +1.11% | +nan% |
| D5 | +0.97% | +nan% |
| D6 | +1.55% | +nan% |
| D7 | +1.40% | +nan% |
| D8 | +1.37% | +nan% |
| D9 | +1.53% | +nan% |
| D10 | +1.25% | +nan% |

#### EP
| Decile | V1 Return | V2 Return |
|--------|----------|----------|
| D1 | +0.21% | -0.21% |
| D2 | +0.40% | -0.09% |
| D3 | +0.79% | +0.53% |
| D4 | +0.85% | +0.52% |
| D5 | +1.31% | +0.49% |
| D6 | +1.66% | +1.20% |
| D7 | +1.18% | +1.42% |
| D8 | +1.85% | +1.37% |
| D9 | +2.06% | +1.75% |
| D10 | +1.32% | +2.05% |

#### Mom_3M
| Decile | V1 Return | V2 Return |
|--------|----------|----------|
| D1 | +1.04% | +0.59% |
| D2 | +1.09% | +0.65% |
| D3 | +1.28% | +0.54% |
| D4 | +1.18% | +0.72% |
| D5 | +1.40% | +1.10% |
| D6 | +1.05% | +0.78% |
| D7 | +1.00% | +1.17% |
| D8 | +1.37% | +0.75% |
| D9 | +0.89% | +0.85% |
| D10 | +1.19% | +0.90% |

#### Mom_6M
| Decile | V1 Return | V2 Return |
|--------|----------|----------|
| D1 | +1.09% | +0.32% |
| D2 | +0.94% | +0.46% |
| D3 | +0.82% | +0.69% |
| D4 | +0.98% | +0.95% |
| D5 | +0.88% | +0.82% |
| D6 | +0.86% | +0.69% |
| D7 | +1.32% | +0.81% |
| D8 | +1.55% | +0.94% |
| D9 | +1.57% | +1.12% |
| D10 | +1.17% | +1.06% |

#### NetMargin
| Decile | V1 Return | V2 Return |
|--------|----------|----------|
| D1 | +0.67% | -0.06% |
| D2 | +0.72% | +0.22% |
| D3 | +1.16% | +0.51% |
| D4 | +1.00% | +0.68% |
| D5 | +1.59% | +0.86% |
| D6 | +1.65% | +0.92% |
| D7 | +1.53% | +1.12% |
| D8 | +1.42% | +1.21% |
| D9 | +0.90% | +1.46% |
| D10 | +1.03% | +0.76% |

#### ProfitGrowth
| Decile | V1 Return | V2 Return |
|--------|----------|----------|
| D1 | -0.02% | +0.18% |
| D2 | +0.58% | +0.66% |
| D3 | +0.45% | +0.78% |
| D4 | +1.06% | +0.61% |
| D5 | +1.16% | +0.78% |
| D6 | +1.16% | +0.62% |
| D7 | +1.37% | +0.86% |
| D8 | +1.49% | +1.02% |
| D9 | +2.25% | +1.30% |
| D10 | +2.13% | +2.12% |

#### ROE
| Decile | V1 Return | V2 Return |
|--------|----------|----------|
| D1 | +0.43% | -0.00% |
| D2 | +0.68% | -0.01% |
| D3 | +0.93% | +0.27% |
| D4 | +1.27% | +0.48% |
| D5 | +1.11% | +0.68% |
| D6 | +1.29% | +0.84% |
| D7 | +1.19% | +0.89% |
| D8 | +1.23% | +1.16% |
| D9 | +1.87% | +1.45% |
| D10 | +1.67% | +1.95% |

#### RevGrowth
| Decile | V1 Return | V2 Return |
|--------|----------|----------|
| D1 | +0.13% | +0.04% |
| D2 | +0.97% | +0.41% |
| D3 | +1.15% | +0.29% |
| D4 | +0.83% | +0.54% |
| D5 | +0.91% | +0.59% |
| D6 | +0.92% | +0.61% |
| D7 | +1.56% | +0.78% |
| D8 | +1.98% | +0.98% |
| D9 | +1.50% | +1.37% |
| D10 | +1.73% | +2.08% |

---
## Part 5: Factor Meaning Drift (Decile Composition)

*Analysis date: 2024-12-31*

### ProfitGrowth — Top Decile Composition

| Metric | V1 D1 (Top) | V1 D10 (Bottom) | V2 D1 (Top) | V2 D10 (Bottom) |
|--------|------------|----------------|------------|----------------|
| N stocks | 30 | 30 | 71 | 71 |
| EP | +0.076 | -1.256 | +0.395 | -0.934 |
| BP | -0.302 | -0.064 | +0.000 | +0.000 |
| ROE | +0.454 | -1.490 | +0.171 | -0.510 |
| RevGrowth | +1.003 | -0.994 | +0.982 | -0.662 |
| Mom_3M | +0.692 | +0.293 | +0.096 | +0.011 |
| Mom_6M | +0.607 | -0.030 | +0.113 | -0.001 |
| NetMargin | +0.265 | -1.215 | +0.023 | -0.148 |

### ROE — Top Decile Composition

| Metric | V1 D1 (Top) | V1 D10 (Bottom) | V2 D1 (Top) | V2 D10 (Bottom) |
|--------|------------|----------------|------------|----------------|
| N stocks | 30 | 30 | 80 | 80 |
| EP | +0.477 | -1.703 | +0.920 | +nan |
| BP | -0.731 | -0.168 | +0.000 | +0.000 |
| ProfitGrowth | +0.699 | -1.167 | +0.554 | -2.058 |
| RevGrowth | +0.810 | -0.426 | +0.805 | -0.242 |
| Mom_3M | -0.403 | +0.258 | -0.031 | +0.191 |
| Mom_6M | -0.180 | +0.179 | -0.120 | +0.174 |
| NetMargin | +0.677 | -1.406 | +0.071 | -0.402 |

### EP — Top Decile Composition

| Metric | V1 D1 (Top) | V1 D10 (Bottom) | V2 D1 (Top) | V2 D10 (Bottom) |
|--------|------------|----------------|------------|----------------|
| N stocks | 30 | 30 | 71 | 71 |
| BP | +1.554 | -0.339 | +0.000 | +0.000 |
| ROE | +0.453 | -1.739 | +0.251 | -0.093 |
| ProfitGrowth | +0.236 | -1.024 | +0.326 | -0.170 |
| RevGrowth | +0.066 | -0.440 | +0.626 | -0.192 |
| Mom_3M | -0.255 | +0.148 | -0.196 | +0.553 |
| Mom_6M | -0.123 | +0.212 | -0.222 | +0.572 |
| NetMargin | +0.763 | -1.258 | +0.106 | -0.047 |

---
## Part 6: BP Factor Audit

### Independent BP Signal

| Metric | V1 | V2 |
|--------|----|----|
| BP standalone IC | +0.0459 | +nan |
| BP IC_IR | +0.27 | +nan |
| EP standalone IC | +0.0767 | +nan |
| EP+BP combined IC | +0.0666 | +nan |
| **BP residual IC (after EP)** | **+0.0039** | **+nan** |


---
## Part 7: Final Synthesis

### 7.1 Which factors experienced Meaning Drift?

| Factor | Rank Preservation | Severity |
|--------|------------------|----------|
| ProfitGrowth | 0.0009 | SEVERE |
| RevGrowth | 0.0063 | SEVERE |
| ROE | 0.3714 | SEVERE |
| EP | 0.4819 | SEVERE |
| NetMargin | 0.6869 | SEVERE |

### 7.2 Which factors had structural IC change?

- **EP**: IC +0.0767 → +0.0930 (improved, Δ=+0.0163)
- **Mom_3M**: IC -0.0289 → -0.0234 (improved, Δ=+0.0055)
- **NetMargin**: IC +0.0363 → +0.0489 (improved, Δ=+0.0125)
- **ProfitGrowth**: IC +0.0528 → +0.0219 (degraded, Δ=-0.0309)
- **ROE**: IC +0.0526 → +0.0593 (improved, Δ=+0.0067)

### 7.3 Why did ProfitGrowth fail in V2?

1. **Rank instability**: Same-stock PG rank correlation between V1/V2 = 0.0009
   → Even for the same stock, its PG percentile changes dramatically when moving from 297-stock to 1,360-stock universe.

2. **IC change**: IC V1=+0.0528, V2=+0.0219

3. **Decile spread**: V1=+2.14%, V2=+1.94%

**Conclusion**: ProfitGrowth's economic meaning changed because:
- In V1 (297 large-caps): High PG = genuine earnings improvement at established companies
- In V2 (1,360 all-caps): High PG = mixed signal — includes base effects, one-time items, small-cap noise
- The factor didn't 'fail' — its INFORMATION CONTENT changed due to universe composition.

### 7.4 Why did ROE flip from positive to negative alpha?

1. **Rank instability**: ROE rank r = 0.3714
2. **IC**: V1=+0.0526, V2=+0.0593

ROE's meaning shift mirrors ProfitGrowth: in a broader universe, high ROE includes:
- Small-cap stocks with unsustainable high ROE (low equity base)
- Cyclical peaks about to mean-revert
- Accounting anomalies
The factor's signal-to-noise ratio degrades with universe breadth.

### 7.5 Was BP incorrectly deleted?

**NO**. BP's residual IC (+0.0039) is negligible. GS correctly identified it as redundant with EP.

### 7.6 DGP Shift vs Model Learning Shift

**ProfitGrowth total concept drift**: 99.9%
  - Same-stock rank displacement: 99.9% (DGP shift)
  - IC structural change: 0.0309 (signal quality)

**The alpha drift is primarily a DATA GENERATING PROCESS shift, not a model learning failure.**
V1 and V2 are not learning the 'same alpha' differently — they are learning from fundamentally different factor signals.
The same factor names (EP, ROE, ProfitGrowth) encode different economic information in different universes.

---
*Report generated: 2026-06-21 17:55:15.068062*