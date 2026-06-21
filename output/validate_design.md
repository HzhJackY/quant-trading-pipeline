# validate_data_integrity.py — Architecture Design

## Data Dependency

```
Input Data:
  ├── Daily OHLCV (dual source):
  │     ├── QFQ close   — from akshare (already cached) or baostock adjustflag='2'
  │     └── Unadj close — from baostock adjustflag='3' (need to fetch, ~0.1s/stock)
  ├── Financial PIT:
  │     └── financial_*_pit.csv — total_share, report_date, pub_date
  └── CSI 800 History:
        └── csi800_history.parquet

Derived Fields:
  adj_factor[t] = qfq_close[t] / unadjusted_close[t]
  exact_mcap[t]  = total_share[report] × (adj_factor[t] / adj_factor[report]) × unadjusted_close[t]
```

---

## Core Algorithm: 動態市值斷層檢測

這是整個校驗中最關鍵的模塊。

### 公式推導

```
給定:
  report_date:   最新財報截止日 (e.g. 2024-03-31)
  pub_date:      該財報實際披露日 (e.g. 2024-04-28)
  total_share_r: 財報披露的總股本

對任意交易日 t ≥ pub_date:
  adj_r = qfq(pub_date) / unadj(pub_date)    ← 披露日的復權因子
  adj_t = qfq(t) / unadj(t)                  ← t 日的復權因子

  exact_mcap[t] = total_share_r × (adj_t / adj_r) × unadj_close[t]

解釋:
  - total_share_r: 財報截止日的股本基數
  - (adj_t / adj_r): 從財報截止日到 t 日期間因送轉股導致的股本變化倍數
  - unadj_close[t]: t 日未經復權的市場價格
  - 乘積 = t 日真實總市值
```

### 斷層檢測算法

```python
def detect_mcap_discontinuity(
    dates: np.ndarray,          # shape (n_days,)
    unadj_close: np.ndarray,    # shape (n_days,)
    qfq_close: np.ndarray,      # shape (n_days,)
    total_share_report: float,  # scalar
    report_date: pd.Timestamp,
    pub_date: pd.Timestamp,
    tradestatus: np.ndarray,    # 1=trading, 0=halt
) -> list[DiscontinuityEvent]:
    """
    逐日計算 exact_mcap，檢測非停牌日的市值跳變。

    算法步驟:
      1. 計算 adj_factor[t] = qfq_close[t] / unadjusted_close[t]
      2. 在 pub_date 處，鎖定 adj_factor_r 和 unadj_close_r
      3. 對每個 t ≥ pub_date:
         exact_mcap[t] = total_share_r × (adj[t] / adj_r) × unadj[t]
      4. 計算 mcap_return[t] = exact_mcap[t] / exact_mcap[t-1] - 1
      5. 標記: mcap_return[t] > +12% 或 < -12% 且 tradestatus=1 的日期

    為什麼用 ±12% 而非 ±10%?
      - 漲跌停 ±10%
      - 加上 2% 緩衝以容忍 bid-ask bounce 和資料精度誤差
      - 科創板/創業板漲跌停 ±20%，但這些股票同樣受 10% 限制的邏輯檢測
        （若市值跳變 >12% 在非漲跌停日出現，一定是股本/復權錯位）

    返回: DiscontinuityEvent 列表
    """
```

### 關鍵邊界處理

```
Scenario 1: 高送轉 (1:1 拆股)
  - unadj_close: 10.0 → 5.0 (-50%, 觸發單日跌幅 > 8%)
  - adj_factor:  1.0 → 2.0 (+100%, 精確對沖)
  - exact_mcap:  total_share × (2.0/1.0) × 5.0 = total_share × 10.0 (平滑 ✓)
  - 檢測結果: PASS (市值連續)

Scenario 2: 現金分紅 (每股派 1 元)
  - unadj_close: 10.0 → 9.0 (-10%)
  - adj_factor:  1.05 → 1.17 (小幅增加，因分紅調整)
  - exact_mcap: 輕微波動但 < 11%
  - 檢測結果: PASS

Scenario 3: 復權因子缺失 (BUG)
  - unadj_close: 10.0 → 5.0 (-50%)
  - adj_factor:  1.0 → 1.0 (沒有變化!)
  - exact_mcap: total_share × 5.0 = 砍半
  - 檢測結果: FAIL → critical_adj_errors.csv

Scenario 4: 股本數據陳舊 (使用舊財報的 total_share)
  - 公司在 2024-06 完成增發，總股本從 10B → 15B
  - 但我們的 total_share 還是 10B（上一份財報的）
  - 2024-06-15: unadj_close 沒變，但 adj_factor 因增發基準調整而跳升
  - exact_mcap: 10B × (1.5/1.0) × close = 仍正確（因為 adj_factor 同步調整）
  - 檢測結果: PASS (adj_factor 自動吸收了股本變化)
```

---

## Module 1: 除權日價格與復權因子對齊檢查

```python
def check_ex_date_discontinuity(
    symbol: str,
    dates: np.ndarray,
    unadj_close: np.ndarray,
    qfq_close: np.ndarray,
    tradestatus: np.ndarray,
) -> list[ExDateError]:
    """
    對每個交易日:
      1. 計算 unadj_return[t] = unadj_close[t] / unadj_close[t-1] - 1
      2. 計算 adj_return[t]   = qfq_close[t] / qfq_close[t-1] - 1
      3. 若 unadj_return[t] < -8% (單日大跌, 暗示除權):
         a. 計算 adj_factor_change = (qfq[t]/unadj[t]) / (qfq[t-1]/unadj[t-1]) - 1
         b. 預期: adj_factor_change 應與 unadj_return 方向相反、幅度匹配
            - 拆股 (1:2): unadj -50%, adj_factor +100%
            - 送股 (10送10): unadj -50%, adj_factor +100%
            - 分紅 (1元/股, 股價10元): unadj -10%, adj_factor +11%
         c. 若 |adj_factor_change + unadj_return| > 5%:
            → 異常: 復權因子未正確反映除權事件
            → 記錄為 ExDateError

    特殊處理:
      - 停牌恢復首日 (tradestatus[t-1]=0, tradestatus[t]=1):
        unadj_return 可能較大, 跳過檢查
      - IPO 首日: 無前收盤價, 跳過
    """
```

## Module 2: 動態市值平滑度校驗

```python
def check_market_cap_smoothness(
    symbol: str,
    dates: np.ndarray,
    unadj_close: np.ndarray,
    qfq_close: np.ndarray,
    total_share_series: list[tuple[pd.Timestamp, float]],  # [(report_date, total_share), ...]
    pub_dates: list[pd.Timestamp],
    tradestatus: np.ndarray,
) -> list[McapDiscontinuity]:
    """
    使用上述 detect_mcap_discontinuity() 逐財報區間掃描。

    對於每份財報 (report_date_i, pub_date_i, total_share_i):
      1. 在 pub_date_i 處鎖定 adj_factor_r
      2. 從 pub_date_i 到 pub_date_{i+1} 逐日計算 exact_mcap[t]
      3. 檢測 mcap_return[t] 是否超出 [-12%, +12%]
      4. 若超出 → McapDiscontinuity

    輸出包含:
      - 觸發日期
      - 前一日 vs 當日市值
      - 當日 unadj_close / adj_factor / total_share 快照
      - 推測原因 (adj_factor 缺失 / 股本數據過期 / 真實停牌未標記)
    """
```

## Module 3: 財務 PIT 時間戳檢查

```python
def check_pit_timestamps(
    fin_data: pd.DataFrame,  # columns: symbol, report_date, pub_date
) -> list[PITViolation]:
    """
    檢查 1: pub_date >= report_date (基本時序)
      - 遍歷所有財務記錄
      - 若 pub_date < report_date → 嚴重錯誤 (資料源錯誤)

    檢查 2: 模擬 merge_asof 驗證 (反未來函數)
      - 對每個 symbol:
        a. 按 report_date 排序
        b. 對每個日期 T 遍歷歷史:
           - 模擬 pd.merge_asof(direction='backward')
           - 驗證: 任何 pub_date > T 的記錄都不會出現在 merge 結果中
        c. 若發現 look-ahead → 中斷並報錯

    檢查 3: pub_date 合理性
      - report_date 為 2024-03-31 的 Q1 報告
      - pub_date 應在 2024-04-01 ~ 2024-04-30 之間 (法規要求)
      - 若 pub_date 超出合理窗口 > 60 天 → Warning
    """
```

## Module 4: 邊界值與空值掃描

```python
def check_sanity_bounds(
    symbol: str,
    adj_factor: np.ndarray,
    total_share_series: list[tuple[pd.Timestamp, float]],
    dates: np.ndarray,
) -> list[SanityViolation]:
    """
    檢查 adj_factor:
      - NaN 或 None → CRITICAL (無法計算市值)
      - ≤ 0 → CRITICAL (物理不可能)
      - 連續兩個交易日變化 > 5% 但無對應 unadj_close 變化 → WARNING

    檢查 total_share:
      - NaN 或 None → CRITICAL
      - ≤ 0 → CRITICAL
      - 在非報告期切換日發生變化 > 1% → WARNING (可能存在未記錄的股本變動)

    檢查 unadj_close:
      - NaN 在 tradestatus=1 的日期 → ERROR
      - = 0 → ERROR
    """
```

---

## Script Entry Point & Output

```
validate_data_integrity.py

Usage:
  python validate_data_integrity.py                    # Full validation
  python validate_data_integrity.py --sample 10         # Test 10 stocks
  python validate_data_integrity.py --symbols 600519,000001  # Specific stocks

Output Files:
  output/
  ├── validation_report.txt           # Human-readable summary
  ├── critical_adj_errors.csv         # Module 1: 除權因子錯誤
  ├── mcap_discontinuities.csv        # Module 2: 市值斷層
  ├── pit_violations.csv              # Module 3: PIT 時間戳違規
  ├── sanity_violations.csv           # Module 4: 邊界值異常
  └── blacklist_symbols.txt           # 應剔除的股票-月份組合

Terminal Output (Report Card):
  ============================================================
  DATA INTEGRITY VALIDATION REPORT
  ============================================================
  Stocks checked:          1,476
  Stocks PASSED:           1,389 (94.1%)
  Stocks with WARNINGS:       72 (4.9%)
  Stocks FAILED:              15 (1.0%)

  --- Failure Breakdown ---
  Ex-date adj_factor mismatch:      3 stocks, 12 events
  Market cap discontinuity:         7 stocks, 23 events
  PIT look-ahead violation:         0 stocks (CRITICAL: would abort)
  Sanity bounds violation:          5 stocks, 18 events

  --- Auto-Blacklisted ---
  000046: mcap discontinuity on 2021-06-15 (split adj missing)
  002131: adj_factor NaN for 45 days
  ...

  Blacklist saved to: output/blacklist_symbols.txt
  ============================================================
```

---

## 自動剔除機制

```python
def generate_blacklist(
    critical_adj_errors: list[ExDateError],
    mcap_discontinuities: list[McapDiscontinuity],
    sanity_violations: list[SanityViolation],
) -> pd.DataFrame:
    """
    合併所有異常事件，生成 blacklist。

    Blacklist 格式:
      symbol, exclusion_start, exclusion_end, reason

    規則:
      1. adj_factor 缺失 → 剔除該 symbol 在所有日期的記錄
      2. 市值斷層 → 剔除觸發日期前後各 1 個月的記錄
      3. PIT look-ahead → 強制拋出異常，終止面板構建
      4. total_share NaN → 剔除該 symbol 在所有日期的記錄
    """
```

---

## 關鍵決策點

1. **unadj_close 資料源**: 目前只有 akshare qfq。需要額外從 baostock `adjustflag='3'` 批次抓取未復權收盤價。每檔股票 ~0.1s，1,476 檔 ~2.5 分鐘。

2. **adj_factor 計算**: `qfq / unadj` 在同一天內計算。因為兩個價格是同一個交易日的不同表示，比值穩定。

3. **科創板/創業板 ±20% 漲跌停**: 市值斷層檢測使用 ±12% 閾值。對於科創板，可以放寬到 ±22%。但 12% 的保守閾值也能捕捉到科創板的真正異常（股本錯位通常導致 >50% 的市值跳變）。

4. **total_share 變動**: 在非財報切換日的 total_share 變化應記錄為 INFO 級別事件（可能是合法的股本變動如回購、增發），不自動剔除，但需要在報告中列出。
