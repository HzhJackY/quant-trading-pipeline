# Phase B Step 1 — 数据抓取脚本设计

## 数据缺口总览

```
CSI 800 当前成分股:      688 只
已缓存日线:              414 只 (其中 335 只 CSI 800, 79 只非 CSI 800)
缺失日线:                353 只 CSI 800 成分股
已缓存财务历史:          418 只 (其中 338 只 CSI 800)
缺失财务历史:            350 只 CSI 800 成分股
```

### 79 只非 CSI 800 缓存股票
这些可能是历史成分股（被调出）或早期 300 样本中的股票。应当保留并纳入训练面板（标记为前 CSI 800 成分股，仅在其实际在指数内的期间参与训练）。

---

## 脚本架构: `run_phaseb_fetch_data.py`

```
Phase 1: 盘点与去重
    ├── 1a. 扫描 data/raw/ 中所有 daily_*.csv + financial_*.csv
    ├── 1b. 去重：同 symbol 多个文件 → 保留日期范围最宽的
    ├── 1c. 统计每个文件的行数、日期范围、缺失交易日数
    └── 1d. 输出缺失清单: missing_daily.txt / missing_financial.txt

Phase 2: 历史成分股快照
    ├── 2a. 尝试 ak.index_stock_cons_csindex("000906") 获取当前成分
    ├── 2b. 尝试 csindex.com.cn 官方 API 获取历史调整记录
    ├── 2c. Fallback: 手动维护 CSI 800 调整记录 (半年一次)
    └── 2d. 生成 csi800_history.parquet: [date, symbol, in_index]

Phase 3: 日线数据补全
    ├── 3a. 遍历 missing_daily.txt (353 只)
    ├── 3b. 调用 Fetcher.get_daily(sym, "20170101", "20241231")
    ├── 3c. 每 5 只存盘一次 checkpoint (断点续跑)
    ├── 3d. 网络错误自动重试 3 次 + 线性退避
    └── 3e. 控制请求频率 (0.5s 间隔, 避免 akshare 限流)

Phase 4: 财务数据补全
    ├── 4a. 遍历 missing_financial.txt (350 只)
    ├── 4b. 调用 Fetcher.get_financial_history(sym)
    ├── 4c. 同 Phase 3 的断点续跑和重试逻辑
    └── 4d. 控制请求频率 (1.0s 间隔)

Phase 5: 质量验证
    ├── 5a. 验证每只 CSI 800 成分股都有 daily CSV
    ├── 5b. 检查日线覆盖率: (实际交易日 / 预期交易日) >= 80%
    ├── 5c. 标记零成交量天数 > 60 的股票 (长期停牌)
    ├── 5d. 标记关键列 (收盘/成交额) 缺失率 > 5% 的股票
    └── 5e. 输出质量报告: data_quality_report.txt
```

---

## 去重与缺失值统计逻辑 (Phase 1 详细)

### 1a. 文件扫描

```python
def scan_cache(data_dir: Path) -> tuple[dict, dict]:
    """
    Returns:
      daily_inventory: {symbol: [FileInfo, ...]}   # 可能有多个文件
      fin_inventory:   {symbol: FileInfo}
    """
```

`FileInfo`:
- `path`: 完整路径
- `date_range`: (start_date, end_date) 从文件名解析
- `row_count`: CSV 行数 (不含表头)
- `columns`: 列名列表
- `adjust`: 复权类型 (qfq/hfq/空)
- `total_days`: end_date - start_date 的自然日数

### 1b. 去重规则

```python
def dedup_daily_files(inventory: dict) -> dict:
    """
    对于同 symbol 的多个 daily CSV:
      1. 优先保留日期范围最宽的
      2. 相同范围时, 优先保留 qfq > hfq > 不复权
      3. 被去重的文件重命名为 .bak 后缀 (不删除)
      4. 记录去重日志
    """
```

去重日志示例：
```
[去重] 000001: 2 files → kept daily_000001_20170101_20241231_qfq.csv
       removed daily_000001_20240101_20240131_qfq.csv (subset)
```

### 1c. 缺失值统计

对每个 daily CSV:
```python
def analyze_daily_quality(csv_path: Path) -> QualityReport:
    df = pd.read_csv(csv_path, parse_dates=["日期"])
    total_rows = len(df)
    expected_trading_days = ... # 根据自然日和工作日计算
    actual_dates = set(df["日期"])

    return QualityReport(
        symbol,
        total_rows,
        date_range=(df["日期"].min(), df["日期"].max()),
        # 关键缺失统计
        close_nan_pct=df["收盘"].isna().mean(),
        volume_zero_pct=(df["成交量"] == 0).mean(),
        amount_zero_pct=(df["成交额"] == 0).mean(),
        # 日期连续性
        max_gap_days=compute_max_gap(df["日期"]),
        date_coverage=len(actual_dates) / expected_trading_days,
        # 标记
        is_suspicious=...,  # coverage < 80% or close_nan_pct > 5%
    )
```

对每个 financial CSV:
```python
def analyze_financial_quality(csv_path: Path) -> QualityReport:
    df = pd.read_csv(csv_path, parse_dates=["report_date"])
    return QualityReport(
        symbol,
        total_reports=len(df),
        report_date_range=(df["report_date"].min(), df["report_date"].max()),
        # 关键列缺失
        roe_nan_pct=df["ROE"].isna().mean(),
        net_profit_nan_pct=df["净利润"].isna().mean(),
        eps_nan_pct=df["每股收益"].isna().mean(),
        # 报告期连续性
        n_quarters=len(df["report_date"].unique()),
        expected_quarters=...,  # 2017Q1 ~ latest
        coverage=n_quarters / expected_quarters,
    )
```

### 1d. 缺失清单输出

`missing_daily.txt`:
```
# CSI 800 stocks missing daily data (353 total)
# Format: symbol  name(optional)
000001  # already cached
000002  # already cached
000017  # MISSING ← 需要抓取
000020  # MISSING ← 需要抓取
...
```

`missing_financial.txt`: 同上格式。

---

## 历史成分股处理 (Phase 2 详细)

### 问题

`akshare` 的 `index_stock_cons()` 和 `index_stock_cons_csindex()` 均不提供历史日期参数，仅返回当前成分。

### 方案 A (优先尝试): 中证指数官网 API

```
https://www.csindex.com.cn/csindex-home/index-component/component-list?indexCode=000906
```

该 URL 可能支持 `date` 查询参数。如果可用，直接获取每个半年节点的快照。

### 方案 B (回退): 手工快照 + 市值近似

1. 收集已有 79 只"前成分股"作为调出池
2. 对于 2017-2024 间的每半年 (6/30, 12/31):
   a. 从 fundamentals parquet 或 akshare 获取该时点全 A 股市值排名
   b. 取市值前 800 + 流动性过滤
   c. 与各期快照交叉验证
3. 最终输出: `csi800_history.parquet`, columns `[snapshot_date, symbol, in_index]`

### 方案 C (保守): 使用当前成分 + 前成分股并集

如果 A/B 都不可行，退而求其次：
- 使用当前 688 只 + 79 只缓存的前成分股 = ~767 只
- 覆盖 CSI 800 历史成分的绝大部分
- 对约 20-30 只因退市/合并而消失的成分股，在回测中接受轻微幸存者偏差
- 在最终报告中明确标注此限制

### 推荐: A → B → C 降级策略。Phase 2a 先尝试 API，2b 兜底。

---

## 质量控制报告示例 (Phase 5 输出)

```
============================================================
Phase B Data Quality Report
Generated: 2026-06-19
============================================================

--- Daily OHLCV Coverage ---
  CSI 800 members:                          688 / 688 (100.0%)
  Date range:                               2017-01-01 ~ 2024-12-31
  Mean trading day coverage:                94.2%
  Stocks with coverage < 80%:               12

--- Financial History Coverage ---
  CSI 800 members with financial data:      685 / 688 (99.6%)
  Mean quarterly report coverage:           91.5%

--- Flagged Anomalies ---
  [WARN] 000017: daily coverage 45% (long suspension suspected)
  [WARN] 000020: close price NaN for 8% of rows
  [WARN] 000511: financial ROE missing for 6 consecutive quarters
  [WARN] 000587: delisted mid-2023, only 1223 trading days

--- Fetch Summary ---
  Daily data fetched:                       347 / 353 (98.3%)
  Daily fetch failed:                       6 (see errors.txt)
  Financial data fetched:                   344 / 350 (98.3%)
  Financial fetch failed:                   6 (see errors.txt)
  Total fetch time:                         ~25 min

============================================================
```

---

## 断点续跑格式

```json
// .phaseb_fetch_state.json
{
  "phase": 3,
  "daily_completed": ["000001", "000002", ...],
  "daily_failed": {"000017": "ConnectionError: timeout"},
  "financial_completed": [...],
  "financial_failed": {...},
  "last_updated": "2026-06-19T21:00:00"
}
```

---

## 待 Review 决策点

1. **历史成分股**: 方案 A/B/C 的降级策略是否合理？是否接受 C 方案的轻微偏差？
2. **日期范围**: 是否仍是 2017-01-01 ~ 2024-12-31？还是扩展到最新（如 2025-12-31）？
3. **复权方式**: qfq 前复权（当前默认）是否 OK？
4. **财务数据**: 是否需要额外获取季报披露日期 (`pub_date`) 做严格的 PIT 对齐？还是用 report_date 即可？
5. **市值数据**: 市值过滤需要 `total_mcap`。当前 CSI 800 是市值加权指数，我们可以从 akshare `stock_zh_a_spot_em()` 获取流通市值，但历史市值需要额外存储。是否一并抓取？
