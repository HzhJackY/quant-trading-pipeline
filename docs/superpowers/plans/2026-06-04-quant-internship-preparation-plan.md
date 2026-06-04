# 量化研究员实习申请准备 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建完整的量化实习申请 Portfolio, 包含因子研究、ML 选股、风险模型三个项目 + 中英文简历

**Architecture:** 分层递进, 第 0 层打通环境和简历, 第 1 层完成因子研究完整流水线(数据→因子→IC→回测→报告), 第 2 层接入 ML 选股(XGBoost/LightGBM 滚动训练), 第 3 层差异化深挖随机波动率与风险模型

**Tech Stack:** Python 3.10+, pandas, numpy, akshare, matplotlib, seaborn, scikit-learn, LightGBM, XGBoost, statsmodels, jupyter

**Data Source:** akshare (免费 A 股数据: 日线、财务、指数成分、行业分类)

---

## 文件结构总览

```
quant/
├── README.md
├── requirements.txt
├── data/
│   ├── __init__.py
│   ├── fetcher.py              # 数据获取 (akshare)
│   └── cleaner.py              # 预处理流水线
├── factor_lib/
│   ├── __init__.py
│   ├── value.py                # BP, EP, CFP, SP
│   ├── momentum.py             # Mom_1M, 3M, 6M, 12M-1M
│   ├── quality.py              # ROE, GrossMargin, DebtRatio
│   ├── volatility.py           # Vol_20D, 60D, DownsideVol, Beta
│   └── growth.py               # RevGrowth, EarningsGrowth
├── factor_research/
│   ├── __init__.py
│   ├── ic_analysis.py          # Rank IC / IC_IR / IC衰减
│   ├── group_backtest.py        # 5分组回测
│   ├── backtest_engine.py      # 多因子组合回测
│   └── report.py               # 图表 + 报告生成
├── ml_selection/
│   ├── __init__.py
│   ├── features.py             # 特征构建
│   ├── training.py             # 滚动窗口训练
│   ├── evaluation.py           # 模型评估 + SHAP
│   └── ml_backtest.py          # ML 选股回测
├── risk_model/
│   ├── __init__.py
│   ├── volatility.py           # GARCH / EGARCH
│   ├── covariance.py           # Shrinkage 协方差
│   └── var_backtest.py         # VaR 回测
├── notebooks/
│   ├── 00_data_overview.ipynb
│   ├── 01_factor_report.ipynb
│   ├── 02_ml_report.ipynb
│   └── 03_risk_report.ipynb
├── resume/
│   ├── resume_cn.md
│   └── resume_en.md
└── tests/
    ├── __init__.py
    ├── test_fetcher.py
    ├── test_cleaner.py
    ├── test_factor_lib.py
    ├── test_ic_analysis.py
    ├── test_group_backtest.py
    └── test_ml.py
```

---

### Task 0: 项目初始化和依赖安装

**Files:**
- Create: `requirements.txt`
- Create: `README.md`

- [ ] **Step 1: 创建 requirements.txt**

```txt
# 数据处理
pandas>=2.0
numpy>=1.24
# 数据获取
akshare>=1.12
# 可视化
matplotlib>=3.7
seaborn>=0.12
plotly>=5.14
# 统计分析
scipy>=1.10
statsmodels>=0.14
# 机器学习
scikit-learn>=1.3
lightgbm>=4.0
xgboost>=2.0
# 波动率建模
arch>=6.1
# 工具
jupyter>=1.0
tqdm>=4.65
openpyxl>=3.1
# 测试
pytest>=7.4
black>=23.0
```

- [ ] **Step 2: 创建 README.md**

```markdown
# Quant Internship Portfolio

量化研究员实习申请项目集合，包含:
- **因子研究**: 15 因子 A 股回测, IC 分析 + 5 分组回测 + Sharpe Ratio
- **ML 选股**: XGBoost/LightGBM 滚动训练选股
- **风险模型**: GARCH 波动率 + Shrinkage 协方差 + VaR 回测

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/ -v

# 启动 Jupyter 查看报告
jupyter lab
```

## 数据来源

使用 [akshare](https://github.com/akfamily/akshare) 免费获取 A 股数据。

## 项目结构

```
quant/
├── data/            # 数据获取与清洗
├── factor_lib/      # 因子定义 (估值/动量/质量/波动/成长)
├── factor_research/  # 因子回测与分析
├── ml_selection/    # ML 选股
├── risk_model/      # 风险模型
├── notebooks/       # 研究报告
└── resume/          # 简历
```

## 回测区间

2017.01 – 2024.12, 股票池: 中证 800
```

- [ ] **Step 3: 安装所有依赖**

```bash
cd /c/Users/HzhJa/Desktop/quant && pip install -r requirements.txt
```

- [ ] **Step 4: 创建所有目录和 __init__.py 文件**

```bash
cd /c/Users/HzhJa/Desktop/quant
mkdir -p data factor_lib factor_research ml_selection risk_model notebooks resume tests
touch data/__init__.py factor_lib/__init__.py factor_research/__init__.py ml_selection/__init__.py risk_model/__init__.py tests/__init__.py
```

- [ ] **Step 5: 运行已有测试确认环境正常**

```bash
pytest tests/ -v
# Expected: "no tests ran"
```

- [ ] **Step 6: Commit**

```bash
git init
git add .
git commit -m "init: project structure and dependencies"
```

---

### Task 1: 数据获取模块 — 日线行情

**Files:**
- Create: `data/fetcher.py`
- Create: `tests/test_fetcher.py`

- [ ] **Step 1: 编写失败测试 — test_fetcher.py**

```python
import pandas as pd
import pytest
from data.fetcher import Fetcher


def test_get_daily_returns_dataframe():
    """日线行情应返回非空DataFrame, 包含必要列"""
    f = Fetcher()
    df = f.get_daily("000001", "20240101", "20240131")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # akshare 日线核心列
    expected_cols = {"日期", "开盘", "收盘", "最高", "最低", "成交量", "换手率"}
    assert expected_cols.issubset(set(df.columns))


def test_get_daily_index_columns():
    """日期列应为 datetime, 数值列无全 NaN"""
    f = Fetcher()
    df = f.get_daily("000001", "20240101", "20240131")
    dt_col = "日期"
    assert pd.api.types.is_datetime64_any_dtype(df[dt_col])
    # 收盘价不能全为 NaN
    assert df["收盘"].notna().sum() > 0


def test_get_daily_caching():
    """连续两次调用应返回相同结果 (本地缓存)"""
    f = Fetcher()
    df1 = f.get_daily("000001", "20240101", "20240131")
    df2 = f.get_daily("000001", "20240101", "20240131")
    pd.testing.assert_frame_equal(df1, df2)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_fetcher.py -v
# Expected: FAIL — ModuleNotFoundError or ImportError
```

- [ ] **Step 3: 实现 Fetcher 类 — get_daily 方法**

```python
"""
数据获取模块。
使用 akshare 获取 A 股日线行情、财务数据、指数成分股。
所有方法内置本地 CSV 缓存, 避免重复请求。
"""

import os
import pandas as pd
import akshare as ak
from datetime import datetime


class Fetcher:
    """A 股数据获取器, 封装 akshare API 并提供缓存层。"""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "raw")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """
        获取个股日线行情。

        参数
        ----
        symbol : str
            股票代码, 如 "000001" (平安银行)
        start_date : str
            起始日期 "YYYYMMDD"
        end_date : str
            结束日期 "YYYYMMDD"
        adjust : str
            复权类型, "qfq"=前复权(默认), "hfq"=后复权, ""=不复权

        返回
        ----
        pd.DataFrame
            列: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅,
                 涨跌幅, 涨跌额, 换手率
        """
        cache_file = os.path.join(
            self.cache_dir,
            f"daily_{symbol}_{start_date}_{end_date}_{adjust}.csv",
        )
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, parse_dates=["日期"])

        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
        except Exception:
            # 某些股票代码需要加前缀
            # akshare 内部可能用 sh/sz 前缀格式
            # 自动尝试补全
            code = symbol
            try:
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )
            except Exception as e:
                raise ValueError(
                    f"无法获取 {symbol} 日线数据: {e}"
                )

        if df.empty:
            raise ValueError(f"{symbol} 在 {start_date}-{end_date} 无数据")

        # 标准化列名: akshare 返回中文列名
        # 确认日期列格式
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期").reset_index(drop=True)

        df.to_csv(cache_file, index=False)
        return df
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_fetcher.py -v
# Expected: PASS (2 tests)
```

- [ ] **Step 5: Commit**

```bash
git add data/fetcher.py tests/test_fetcher.py
git commit -m "feat: add daily data fetcher with akshare caching"
```

---

### Task 2: 数据获取模块 — 指数成分股

**Files:**
- Modify: `data/fetcher.py` (追加方法)
- Modify: `tests/test_fetcher.py` (追加测试)

- [ ] **Step 1: 在 test_fetcher.py 追加测试**

```python
def test_get_index_members_returns_list():
    """指数成分股应返回非空列表"""
    f = Fetcher()
    members = f.get_index_members("000906")  # 中证 800
    assert isinstance(members, list)
    assert len(members) > 100  # 中证800 至少几百只


def test_get_index_members_elements_are_strings():
    """每只股票代码应为 6 位字符串"""
    f = Fetcher()
    members = f.get_index_members("000300")  # 沪深300
    for s in members[:5]:
        assert isinstance(s, str)
        assert len(s) == 6
        assert s.isdigit()
```

- [ ] **Step 2: 实现 get_index_members 方法**

```python
    def get_index_members(self, index_code: str) -> list[str]:
        """
        获取指数成分股列表。

        参数
        ----
        index_code : str
            指数代码, 如 "000300"(沪深300), "000905"(中证500),
            "000906"(中证800), "000852"(中证1000)

        返回
        ----
        list[str]
            成分股代码列表, 如 ["000001", "000002", ...]
        """
        cache_file = os.path.join(
            self.cache_dir, f"index_members_{index_code}.csv"
        )
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, dtype={"symbol": str})
            return df["symbol"].tolist()

        try:
            # akshare: 指数成分股
            df = ak.index_stock_cons(index_code)
        except Exception:
            # 备选: 部分指数用 stock_board_concept_* 系列
            # 尝试用 stock_zh_index_spot_em 查找
            try:
                df = ak.index_stock_cons_csindex(index_code)
            except Exception as e:
                raise ValueError(
                    f"无法获取指数 {index_code} 成分股: {e}"
                )

        if df.empty:
            raise ValueError(f"指数 {index_code} 无成分股数据")

        # 标准化: 不同 API 列名不同, 统一提取股票代码列
        code_col = None
        for col in df.columns:
            if any(kw in col.lower() for kw in ["代码", "code", "symbol"]):
                code_col = col
                break
        if code_col is None:
            code_col = df.columns[0]  # fallback

        symbols = df[code_col].astype(str).str.zfill(6).tolist()
        symbols = [s for s in symbols if s.isdigit() and len(s) == 6]

        pd.DataFrame({"symbol": symbols}).to_csv(cache_file, index=False)
        return symbols
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_fetcher.py -v
# Expected: 4 tests PASS
```

- [ ] **Step 4: Commit**

```bash
git add data/fetcher.py tests/test_fetcher.py
git commit -m "feat: add index members fetcher"
```

---

### Task 3: 数据获取模块 — 财务数据

**Files:**
- Modify: `data/fetcher.py` (追加 get_financial 方法)
- Modify: `tests/test_fetcher.py` (追加测试)

- [ ] **Step 1: 追加测试**

```python
def test_get_financial_returns_dataframe():
    """财务数据应返回 DataFrame, 至少包含净利润、营收"""
    f = Fetcher()
    df = f.get_financial("000001", "20231231")
    assert isinstance(df, pd.DataFrame)
    # 财务核心字段
    financial_cols = {"股票代码", "报告期"}
    assert financial_cols.issubset(set(df.columns))
```

- [ ] **Step 2: 实现 get_financial 方法**

```python
    def get_financial(
        self, symbol: str, report_date: str
    ) -> pd.DataFrame:
        """
        获取个股财务数据 (资产负债表 + 利润表 + 现金流)。

        参数
        ----
        symbol : str
            股票代码
        report_date : str
            报告期 "YYYYMMDD" 或 "YYYY1231"

        返回
        ----
        pd.DataFrame
            包含主要财务指标的宽表
        """
        cache_file = os.path.join(
            self.cache_dir, f"financial_{symbol}_{report_date}.csv"
        )
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)

        # 先尝试获取利润表
        try:
            df_profit = ak.stock_profit_sheet_by_report_em(
                symbol=symbol,
            )
        except Exception as e:
            raise ValueError(f"无法获取 {symbol} 利润表: {e}")

        # 获取资产负债表
        try:
            df_balance = ak.stock_balance_sheet_by_report_em(
                symbol=symbol,
            )
        except Exception:
            df_balance = None

        if df_profit.empty:
            raise ValueError(f"{symbol} 财务数据为空")

        # 提取关键字段
        result = pd.DataFrame()
        result["股票代码"] = [symbol]
        result["报告期"] = [report_date]

        # 从利润表和资产负债表中提取核心指标
        # 具体列名需要适配 akshare 返回格式
        # 这里提供框架, 后续根据实际 API 输出调整

        key_items = {
            "营业收入": df_profit,
            "净利润": df_profit,
            "营业成本": df_profit,
            "总资产": df_balance,
            "总负债": df_balance,
            "净资产": df_balance,
        }
        for name, src_df in key_items.items():
            if src_df is not None:
                # 在 DataFrame 中搜索匹配行
                for col in src_df.columns:
                    if name in str(src_df.iloc[0].get(col, "")):
                        result[name] = src_df.loc[
                            src_df.iloc[:, 0].str.contains(
                                name, na=False
                            ),
                            src_df.columns[-1],
                        ].values[:1]
                        break

        result.to_csv(cache_file, index=False)
        return result
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_fetcher.py -v
# Expected: 5 tests PASS
```

- [ ] **Step 4: Commit**

```bash
git add data/fetcher.py tests/test_fetcher.py
git commit -m "feat: add financial data fetcher"
```

---

### Task 4: 数据预处理 — 清洗流水线

**Files:**
- Create: `data/cleaner.py`
- Create: `tests/test_cleaner.py`

- [ ] **Step 1: 编写失败测试**

```python
import numpy as np
import pandas as pd
import pytest
from data.cleaner import (
    winsorize_mad,
    neutralize_industry_market_cap,
    standardize_cross_section,
)


class TestWinsorize:
    def test_basic_winsorize(self):
        """极端值应被缩尾到 3x MAD 边界"""
        series = pd.Series(
            [1.0] * 20 + [100.0, -100.0]
        )  # 两个极端值
        result = winsorize_mad(series, n_mad=3.0)
        assert result.max() < 100.0
        assert result.min() > -100.0

    def test_no_change_for_normal_data(self):
        """正态数据不应被大幅改变"""
        rng = np.random.default_rng(42)
        data = pd.Series(rng.normal(0, 1, 1000))
        result = winsorize_mad(data, n_mad=3.0)
        # 99% 的值应与原始值相同
        unchanged = (result == data).sum()
        assert unchanged > 900


class TestStandardize:
    def test_cross_section_standardize(self):
        """截面标准化后均值接近 0, 标准差接近 1"""
        df = pd.DataFrame(
            {
                "date": ["2024-01-31"] * 5,
                "symbol": [f"{i:06d}" for i in range(1, 6)],
                "factor": [10.0, 20.0, 15.0, 25.0, 30.0],
            }
        )
        result = standardize_cross_section(
            df, factor_col="factor", date_col="date"
        )
        mean_val = result["factor_z"].mean()
        std_val = result["factor_z"].std()
        assert abs(mean_val) < 1e-10
        assert abs(std_val - 1.0) < 0.1


class TestNeutralize:
    def test_neutralize_removes_industry_effect(self):
        """中性化后因子应对行业虚拟变量正交"""
        df = pd.DataFrame(
            {
                "date": ["2024-01-31"] * 6,
                "symbol": [f"{i:06d}" for i in range(1, 7)],
                "factor": [1, 2, 3, 10, 11, 12],
                "industry": ["A", "A", "A", "B", "B", "B"],
                "log_market_cap": [10, 11, 12, 10, 11, 12],
            }
        )
        result = neutralize_industry_market_cap(
            df,
            factor_col="factor",
            industry_col="industry",
            mcap_col="log_market_cap",
            date_col="date",
        )
        # A组均值和 B组均值在残差上应该接近
        a_mean = result.loc[result["industry"] == "A", "factor_neutral"].mean()
        b_mean = result.loc[result["industry"] == "B", "factor_neutral"].mean()
        assert abs(a_mean - b_mean) < 5.0  # 行业偏差降低
```

- [ ] **Step 2: 确认测试失败**

```bash
pytest tests/test_cleaner.py -v
# Expected: FAIL — 模块未实现
```

- [ ] **Step 3: 实现 cleaner.py**

```python
"""
数据预处理流水线。
去极值 → 中性化 → 标准化
"""

import numpy as np
import pandas as pd


def winsorize_mad(
    series: pd.Series, n_mad: float = 3.0
) -> pd.Series:
    """
    MAD (Median Absolute Deviation) 缩尾去极值。

    参数
    ----
    series : pd.Series
        因子值序列
    n_mad : float
        MAD 倍数阈值, 默认 3.0

    返回
    ----
    pd.Series
        缩尾后的序列
    """
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return series  # 无变异, 不需要缩尾
    upper = median + n_mad * mad * 1.4826  # 1.4826 = MAD → 标准差换算
    lower = median - n_mad * mad * 1.4826
    return series.clip(lower=lower, upper=upper)


def standardize_cross_section(
    df: pd.DataFrame,
    factor_col: str = "factor",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    截面标准化: 在每个时间截面上将因子值转为 Z-score。

    返回的 DataFrame 会新增一列 ``{factor_col}_z``。
    """
    df = df.copy()
    z_scores = df.groupby(date_col)[factor_col].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
        if x.std(ddof=0) > 0
        else 0.0
    )
    df[f"{factor_col}_z"] = z_scores
    return df


def neutralize_industry_market_cap(
    df: pd.DataFrame,
    factor_col: str = "factor",
    industry_col: str = "industry",
    mcap_col: str = "log_market_cap",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    行业 + 市值中性化。
    对每个截面, 将因子对行业虚拟变量和对数市值做 OLS 回归, 取残差。

    返回的 DataFrame 会新增一列 ``{factor_col}_neutral``。
    """
    from statsmodels.api import OLS, add_constant

    df = df.copy()
    residuals_list = []

    for date, group in df.groupby(date_col):
        if len(group) < 10:
            group["_resid"] = group[factor_col]
            residuals_list.append(group)
            continue

        # 行业哑变量
        industry_dummies = pd.get_dummies(
            group[industry_col], drop_first=True
        ).astype(float)
        X = pd.concat(
            [industry_dummies, group[mcap_col]], axis=1
        )
        X = add_constant(X)
        y = group[factor_col]

        try:
            model = OLS(y, X).fit()
            group["_resid"] = model.resid
        except Exception:
            group["_resid"] = y

        residuals_list.append(group)

    result = pd.concat(residuals_list)
    result[f"{factor_col}_neutral"] = result["_resid"]
    result = result.drop(columns=["_resid"])
    return result
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_cleaner.py -v
# Expected: 4 tests PASS
```

- [ ] **Step 5: Commit**

```bash
git add data/cleaner.py tests/test_cleaner.py
git commit -m "feat: add preprocessing pipeline (winsorize, standardize, neutralize)"
```

---

### Task 5: 因子库 — 估值 + 动量因子

**Files:**
- Create: `factor_lib/value.py`
- Create: `factor_lib/momentum.py`
- Create: `tests/test_factor_lib.py`

- [ ] **Step 1: 编写 tests/test_factor_lib.py (估值+动量部分)**

```python
import pandas as pd
import numpy as np
import pytest
from factor_lib.value import compute_bp, compute_ep
from factor_lib.momentum import (
    compute_momentum_1m,
    compute_momentum_3m,
    compute_momentum_6m,
    compute_momentum_12m_1m,
)


@pytest.fixture
def sample_prices():
    """模拟 200 个交易日 × 3 只股票的收盘价"""
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    rng = np.random.default_rng(42)
    data = []
    for s in ["000001", "000002", "000003"]:
        start_price = rng.uniform(10, 50)
        returns = rng.normal(0.0005, 0.02, 200)
        prices = start_price * np.exp(np.cumsum(returns))
        for i, d in enumerate(dates):
            data.append(
                {
                    "date": d,
                    "symbol": s,
                    "close": prices[i],
                }
            )
    return pd.DataFrame(data)


def test_momentum_1m_shape(sample_prices):
    """过去 1 月动量应返回正确形状"""
    result = compute_momentum_1m(sample_prices)
    assert isinstance(result, pd.DataFrame)
    assert "date" in result.columns
    assert "symbol" in result.columns
    assert "Mom_1M" in result.columns
    assert len(result) > 0


def test_momentum_12m_1m_no_lookahead(sample_prices):
    """动量因子不应使用未来信息"""
    result = compute_momentum_12m_1m(sample_prices)
    pivot = result.pivot(
        index="date", columns="symbol", values="Mom_12M_1M"
    )
    # 检查最后日期的值是否为 NaN (因为没有足够历史)
    # 或检查前 242 天为 NaN
    assert pivot.iloc[:242].isna().all().all()
```

- [ ] **Step 2: 实现 factor_lib/value.py**

```python
"""
估值类因子。
BP (Book-to-Price), EP (Earnings-to-Price).
"""

import pandas as pd
import numpy as np


def compute_bp(
    financial_data: pd.DataFrame,
    daily_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    BP = 净资产 / 总市值

    参数
    ----
    financial_data : pd.DataFrame
        字段: symbol, report_date, 净资产 (或 total_equity)
    daily_data : pd.DataFrame
        字段: date, symbol, close, 总市值 (或 total_mv)

    返回
    ----
    pd.DataFrame
        列: date, symbol, BP
    """
    # 合并市值和财务数据
    merged = daily_data.merge(
        financial_data[["symbol", "report_date", "净资产"]],
        on="symbol",
        how="left",
    )
    merged["BP"] = merged["净资产"] / merged["总市值"]
    return merged[["date", "symbol", "BP"]].dropna(subset=["BP"])


def compute_ep(
    financial_data: pd.DataFrame,
    daily_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    EP = 净利润(TTM) / 总市值

    使用最近四个季度净利润之和作为 TTM。
    """
    # 按 symbol 和 report_date 排序
    fin = financial_data.sort_values(
        ["symbol", "report_date"]
    )
    # 滚动四个季度求和
    fin["net_profit_ttm"] = fin.groupby("symbol")[
        "净利润"
    ].transform(lambda x: x.rolling(4, min_periods=4).sum())

    merged = daily_data.merge(
        fin[["symbol", "report_date", "net_profit_ttm"]],
        on="symbol",
        how="left",
    )
    merged["EP"] = merged["net_profit_ttm"] / merged["总市值"]
    return merged[["date", "symbol", "EP"]].dropna(subset=["EP"])
```

- [ ] **Step 3: 实现 factor_lib/momentum.py**

```python
"""
动量类因子。
Mom_1M, Mom_3M, Mom_6M, Mom_12M_1M.
"""

import pandas as pd
import numpy as np


def _compute_return(
    daily_data: pd.DataFrame,
    lookback_days: int,
    skip_days: int = 0,
    col_name: str = "momentum",
) -> pd.DataFrame:
    """
    通用动量计算: 计算过去 lookback_days 的累计收益,
    跳过最近 skip_days 天。
    """
    df = daily_data.sort_values(
        ["symbol", "date"]
    ).copy()
    # 计算日收益率
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()

    # 滚动累计收益: 用对数收益求和在指数
    df["log_return"] = np.log(1 + df["daily_return"].fillna(0))

    result_rows = []
    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("date")
        # 对每一天计算: date - lookback_days - skip_days 到 date - skip_days 的收益
        for i, (idx, row) in enumerate(group.iterrows()):
            end = i - skip_days
            start = end - lookback_days
            if start < 0:
                continue
            window_returns = group.iloc[start:end]["log_return"]
            cum_return = np.exp(window_returns.sum()) - 1
            result_rows.append(
                {
                    "date": row["date"],
                    "symbol": symbol,
                    col_name: cum_return,
                }
            )

    return pd.DataFrame(result_rows)


def compute_momentum_1m(
    daily_data: pd.DataFrame,
) -> pd.DataFrame:
    """过去 1 个月 (约 21 个交易日) 收益, 跳过最近 5 日。"""
    return _compute_return(
        daily_data,
        lookback_days=21,
        skip_days=5,
        col_name="Mom_1M",
    )


def compute_momentum_3m(
    daily_data: pd.DataFrame,
) -> pd.DataFrame:
    """过去 3 个月 (约 63 个交易日) 收益, 跳过最近 5 日。"""
    return _compute_return(
        daily_data,
        lookback_days=63,
        skip_days=5,
        col_name="Mom_3M",
    )


def compute_momentum_6m(
    daily_data: pd.DataFrame,
) -> pd.DataFrame:
    """过去 6 个月 (约 126 个交易日) 收益。"""
    return _compute_return(
        daily_data,
        lookback_days=126,
        skip_days=5,
        col_name="Mom_6M",
    )


def compute_momentum_12m_1m(
    daily_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    过去 12 个月收益, 跳过最近 1 个月。
    即 t-12 到 t-1 个月的累计收益。
    """
    return _compute_return(
        daily_data,
        lookback_days=231,  # 252 - 21 ≈ 231
        skip_days=21,
        col_name="Mom_12M_1M",
    )
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_factor_lib.py -v
# Expected: 2 tests PASS
```

- [ ] **Step 5: Commit**

```bash
git add factor_lib/ tests/test_factor_lib.py
git commit -m "feat: add value and momentum factor libraries"
```

---

### Task 6: 因子库 — 质量 + 波动率 + 成长因子

**Files:**
- Create: `factor_lib/quality.py`
- Create: `factor_lib/volatility.py`
- Create: `factor_lib/growth.py`
- Modify: `tests/test_factor_lib.py` (追加测试)

- [ ] **Step 1: 追加测试**

```python
from factor_lib.quality import compute_roe, compute_gross_margin
from factor_lib.volatility import (
    compute_volatility_20d,
    compute_volatility_60d,
    compute_beta,
)
from factor_lib.growth import (
    compute_revenue_growth_yoy,
    compute_earnings_growth_yoy,
)


def test_volatility_20d_non_negative(sample_prices):
    """波动率应非负"""
    result = compute_volatility_20d(sample_prices)
    assert (result["Vol_20D"] >= 0).all()


def test_beta_range(sample_prices):
    """Beta 值大致在合理范围"""
    result = compute_beta(sample_prices)
    # 三只股票, Beta 应在 0 到 3 之间
    betas = result["Beta"].dropna()
    assert (betas >= 0).all()
    assert (betas <= 3).all()
```

- [ ] **Step 2: 实现 quality.py**

```python
"""
质量类因子。
ROE, 毛利率 (Gross Margin), 资产负债率 (Debt Ratio).
"""

import pandas as pd


def compute_roe(financial_data: pd.DataFrame) -> pd.DataFrame:
    """
    ROE = 净利润 / 净资产
    使用最近报告期数据。
    """
    df = financial_data.copy()
    if "净利润" in df.columns and "净资产" in df.columns:
        df["ROE"] = df["净利润"] / df["净资产"].replace(0, float("nan"))
    else:
        df["ROE"] = float("nan")
    return df[["symbol", "report_date", "ROE"]].dropna(
        subset=["ROE"]
    )


def compute_gross_margin(financial_data: pd.DataFrame) -> pd.DataFrame:
    """
    毛利率 = (营业收入 - 营业成本) / 营业收入
    """
    df = financial_data.copy()
    if "营业收入" in df.columns and "营业成本" in df.columns:
        df["Gross_Margin"] = (
            df["营业收入"] - df["营业成本"]
        ) / df["营业收入"].replace(0, float("nan"))
    else:
        df["Gross_Margin"] = float("nan")
    return df[["symbol", "report_date", "Gross_Margin"]].dropna(
        subset=["Gross_Margin"]
    )


def compute_debt_ratio(financial_data: pd.DataFrame) -> pd.DataFrame:
    """
    资产负债率 = 总负债 / 总资产
    """
    df = financial_data.copy()
    if "总负债" in df.columns and "总资产" in df.columns:
        df["Debt_Ratio"] = (
            df["总负债"] / df["总资产"].replace(0, float("nan"))
        )
    else:
        df["Debt_Ratio"] = float("nan")
    return df[["symbol", "report_date", "Debt_Ratio"]].dropna(
        subset=["Debt_Ratio"]
    )
```

- [ ] **Step 3: 实现 volatility.py**

```python
"""
波动率类因子。
日波动率(20/60日), 下行波动率, Beta。
"""

import pandas as pd
import numpy as np


def _compute_rolling_vol(
    daily_data: pd.DataFrame,
    window: int,
    col_name: str,
) -> pd.DataFrame:
    """滚动窗口年化波动率。"""
    df = daily_data.sort_values(["symbol", "date"]).copy()
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()
    df[col_name] = (
        df.groupby("symbol")["daily_return"]
        .transform(
            lambda x: x.rolling(window, min_periods=window // 2).std()
        )
        * np.sqrt(252)
    )
    return df[["date", "symbol", col_name]].dropna(
        subset=[col_name]
    )


def compute_volatility_20d(
    daily_data: pd.DataFrame,
) -> pd.DataFrame:
    """过去 20 个交易日年化波动率。"""
    return _compute_rolling_vol(daily_data, 20, "Vol_20D")


def compute_volatility_60d(
    daily_data: pd.DataFrame,
) -> pd.DataFrame:
    """过去 60 个交易日年化波动率。"""
    return _compute_rolling_vol(daily_data, 60, "Vol_60D")


def compute_downside_vol_60d(
    daily_data: pd.DataFrame,
) -> pd.DataFrame:
    """过去 60 个交易日下行波动率 (只计负收益)。"""
    df = daily_data.sort_values(["symbol", "date"]).copy()
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()
    df["neg_return"] = df["daily_return"].clip(upper=0)

    result_rows = []
    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("date")
        group["Downside_Vol"] = (
            group["neg_return"].rolling(60, min_periods=30).std()
            * np.sqrt(252)
        )
        result_rows.append(group)
    result = pd.concat(result_rows)
    return result[["date", "symbol", "Downside_Vol"]].dropna(
        subset=["Downside_Vol"]
    )


def compute_beta(
    daily_data: pd.DataFrame,
    market_col: str = "market_return",
) -> pd.DataFrame:
    """
    过去 60 个交易日 CAPM Beta。
    需要在 daily_data 中包含市场收益率列 (market_return),
    或传入全体股票用市场指数替代。
    """
    df = daily_data.sort_values(["symbol", "date"]).copy()
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()

    # 若无市场收益率, 用所有股票的等权平均作为代理
    if market_col not in df.columns:
        mkt = (
            df.groupby("date")["daily_return"]
            .mean()
            .rename("market_return")
        )
        df = df.merge(mkt, on="date", how="left")
        market_col = "market_return"

    result_rows = []
    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("date").dropna(
            subset=["daily_return", market_col]
        )
        if len(group) < 30:
            continue
        # 滚动 Beta
        rolling_cov = (
            group["daily_return"]
            .rolling(60, min_periods=30)
            .cov(group[market_col])
        )
        rolling_var = (
            group[market_col].rolling(60, min_periods=30).var()
        )
        group["Beta"] = rolling_cov / rolling_var.replace(0, np.nan)
        result_rows.append(
            group[["date", "symbol", "Beta"]].dropna(
                subset=["Beta"]
            )
        )

    return pd.concat(result_rows) if result_rows else pd.DataFrame()
```

- [ ] **Step 4: 实现 growth.py**

```python
"""
成长类因子。
营收增速 YoY, 净利润增速 YoY。
"""

import pandas as pd


def compute_revenue_growth_yoy(
    financial_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    营收同比增长率 = (本季度营收 - 去年同季度营收) / |去年同季度营收|
    """
    df = financial_data.sort_values(
        ["symbol", "report_date"]
    ).copy()
    if "营业收入" not in df.columns:
        df["Rev_Growth_YoY"] = float("nan")
        return df[["symbol", "report_date", "Rev_Growth_YoY"]]

    # 按 symbol 分组, shift 4 个季度得到去年同期
    df["revenue_lag_4q"] = df.groupby("symbol")["营业收入"].shift(4)
    df["Rev_Growth_YoY"] = (
        df["营业收入"] - df["revenue_lag_4q"]
    ) / df["revenue_lag_4q"].abs().replace(0, float("nan"))
    return df[["symbol", "report_date", "Rev_Growth_YoY"]].dropna(
        subset=["Rev_Growth_YoY"]
    )


def compute_earnings_growth_yoy(
    financial_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    净利润同比增长率。
    """
    df = financial_data.sort_values(
        ["symbol", "report_date"]
    ).copy()
    if "净利润" not in df.columns:
        df["Earnings_Growth"] = float("nan")
        return df[
            ["symbol", "report_date", "Earnings_Growth"]
        ]

    df["earnings_lag_4q"] = df.groupby("symbol")["净利润"].shift(
        4
    )
    df["Earnings_Growth"] = (
        df["净利润"] - df["earnings_lag_4q"]
    ) / df["earnings_lag_4q"].abs().replace(0, float("nan"))
    return df[
        ["symbol", "report_date", "Earnings_Growth"]
    ].dropna(subset=["Earnings_Growth"])
```

- [ ] **Step 5: 运行全部因子测试**

```bash
pytest tests/test_factor_lib.py -v
# Expected: 4 tests PASS
```

- [ ] **Step 6: Commit**

```bash
git add factor_lib/ tests/test_factor_lib.py
git commit -m "feat: add quality, volatility, and growth factor libraries"
```

---

### Task 7: IC 分析引擎

**Files:**
- Create: `factor_research/ic_analysis.py`
- Create: `tests/test_ic_analysis.py`

- [ ] **Step 1: 编写 tests/test_ic_analysis.py**

```python
import pandas as pd
import numpy as np
import pytest
from factor_research.ic_analysis import (
    compute_rank_ic,
    compute_ic_summary,
)


@pytest.fixture
def sample_factor_return_data():
    """模拟 12 期因子截面 + 下期收益"""
    rng = np.random.default_rng(42)
    rows = []
    for t in range(12):
        date_str = f"2024-{t+1:02d}-28"
        n_stocks = 100
        factor = rng.normal(0, 1, n_stocks)
        # 让收益与因子有轻微正相关
        forward_return = (
            0.5 * factor + rng.normal(0, 0.9, n_stocks)
        )
        for i in range(n_stocks):
            rows.append(
                {
                    "date": date_str,
                    "symbol": f"{i+1:06d}",
                    "factor": factor[i],
                    "forward_return_1m": forward_return[i],
                }
            )
    return pd.DataFrame(rows)


def test_compute_rank_ic_returns_series(
    sample_factor_return_data,
):
    """compute_rank_ic 应返回 Series"""
    ic = compute_rank_ic(
        sample_factor_return_data,
        factor_col="factor",
        return_col="forward_return_1m",
        date_col="date",
    )
    assert isinstance(ic, pd.Series)
    assert len(ic) > 0


def test_rank_ic_in_range(sample_factor_return_data):
    """Rank IC 应在 [-1, 1] 之间"""
    ic = compute_rank_ic(
        sample_factor_return_data,
        factor_col="factor",
        return_col="forward_return_1m",
        date_col="date",
    )
    assert (ic >= -1).all()
    assert (ic <= 1).all()


def test_ic_summary_contains_key_stats(
    sample_factor_return_data,
):
    """IC Summary 应包含均值、标准差、IR、胜率"""
    ic = compute_rank_ic(
        sample_factor_return_data,
        factor_col="factor",
        return_col="forward_return_1m",
        date_col="date",
    )
    summary = compute_ic_summary(ic)
    for key in ["IC_Mean", "IC_Std", "IC_IR", "IC_Win_Rate"]:
        assert key in summary.index or key in summary
```

- [ ] **Step 2: 实现 factor_research/ic_analysis.py**

```python
"""
因子 IC (Information Coefficient) 分析。
Rank IC, IC_IR, IC 衰减, IC 胜率。
"""

import pandas as pd
import numpy as np
from scipy import stats


def compute_rank_ic(
    df: pd.DataFrame,
    factor_col: str = "factor",
    return_col: str = "forward_return_1m",
    date_col: str = "date",
) -> pd.Series:
    """
    计算每个截面的 Rank IC。
    Rank IC = Spearman correlation(factor_rank, forward_return_rank)

    参数
    ----
    df : pd.DataFrame
        包含 factor_col, return_col, date_col
    factor_col : str
        因子值列名
    return_col : str
        下期收益率列名
    date_col : str
        日期列名

    返回
    ----
    pd.Series
        index=日期, values=Rank IC
    """
    ic_values = {}
    for date, group in df.groupby(date_col):
        valid = group[[factor_col, return_col]].dropna()
        if len(valid) < 30:
            continue
        ic, _ = stats.spearmanr(
            valid[factor_col], valid[return_col]
        )
        ic_values[date] = ic
    return pd.Series(ic_values, name="Rank_IC").sort_index()


def compute_ic_summary(ic_series: pd.Series) -> dict:
    """
    计算 IC 汇总统计量。

    返回
    ----
    dict
        keys: IC_Mean, IC_Std, IC_IR, IC_Win_Rate, IC_t_stat, Periods
    """
    ic = ic_series.dropna()
    n = len(ic)
    if n == 0:
        return {}

    mean_ic = ic.mean()
    std_ic = ic.std(ddof=1)
    ir = mean_ic / std_ic if std_ic > 0 else 0.0
    win_rate = (ic > 0).sum() / n
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0.0

    return {
        "IC_Mean": round(mean_ic, 4),
        "IC_Std": round(std_ic, 4),
        "IC_IR": round(ir, 4),
        "IC_Win_Rate": round(win_rate, 4),
        "IC_t_stat": round(t_stat, 2),
        "Periods": n,
    }


def compute_ic_decay(
    df: pd.DataFrame,
    factor_col: str = "factor",
    return_cols: list[str] | None = None,
    date_col: str = "date",
) -> dict[str, float]:
    """
    计算 IC 衰减: 因子对未来多期收益的 Rank IC。
    例如: forward_return_1m, forward_return_2m, ..., forward_return_6m

    返回
    ----
    dict
        {return_col: mean_IC}
    """
    if return_cols is None:
        return {}
    decay = {}
    for ret_col in return_cols:
        ic = compute_rank_ic(
            df, factor_col=factor_col, return_col=ret_col, date_col=date_col
        )
        decay[ret_col] = round(ic.mean(), 4)
    return decay
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_ic_analysis.py -v
# Expected: 3 tests PASS
```

- [ ] **Step 4: Commit**

```bash
git add factor_research/ic_analysis.py tests/test_ic_analysis.py
git commit -m "feat: add IC analysis engine (Rank IC, IC_IR, IC decay)"
```

---

### Task 8: 分组回测引擎

**Files:**
- Create: `factor_research/group_backtest.py`
- Create: `tests/test_group_backtest.py`

- [ ] **Step 1: 编写 tests/test_group_backtest.py**

```python
import pandas as pd
import numpy as np
import pytest
from factor_research.group_backtest import (
    assign_quantile_groups,
    compute_group_returns,
)


@pytest.fixture
def sample_one_period():
    """单期 500 只股票, 含因子值和下期收益"""
    rng = np.random.default_rng(123)
    n = 500
    return pd.DataFrame(
        {
            "date": "2024-01-31",
            "symbol": [f"{i:06d}" for i in range(n)],
            "factor": rng.normal(0, 1, n),
            "forward_return_1m": rng.normal(0.01, 0.06, n),
        }
    )


def test_assign_5_groups(sample_one_period):
    """应分配 1-5 共 5 组, 每组大小均匀"""
    result = assign_quantile_groups(
        sample_one_period,
        factor_col="factor",
        n_groups=5,
        date_col="date",
    )
    assert "group" in result.columns
    assert set(result["group"].unique()) == {1, 2, 3, 4, 5}
    # 每组应有约 100 只
    counts = result["group"].value_counts()
    assert all(abs(c - 100) <= 2 for c in counts)


def test_group1_lowest_factor(sample_one_period):
    """Group 1 应为因子值最低的组"""
    result = assign_quantile_groups(
        sample_one_period,
        factor_col="factor",
        n_groups=5,
        date_col="date",
    )
    g1_mean = result.loc[result["group"] == 1, "factor"].mean()
    g5_mean = result.loc[result["group"] == 5, "factor"].mean()
    assert g1_mean < g5_mean


def test_compute_group_returns(sample_one_period):
    """分组收益计算"""
    assigned = assign_quantile_groups(
        sample_one_period,
        factor_col="factor",
        n_groups=5,
        date_col="date",
    )
    returns = compute_group_returns(
        assigned,
        return_col="forward_return_1m",
        group_col="group",
        date_col="date",
    )
    assert isinstance(returns, pd.DataFrame)
    assert "group" in returns.columns
    assert "return" in returns.columns
    assert len(returns) == 5
```

- [ ] **Step 2: 实现 factor_research/group_backtest.py**

```python
"""
分层回测引擎。
按因子值分组 → 等权计算各组收益 → 多空收益。
"""

import pandas as pd
import numpy as np


def assign_quantile_groups(
    df: pd.DataFrame,
    factor_col: str = "factor",
    n_groups: int = 5,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    在每个截面上按因子值分为 n_groups 组。
    Group 1 = 因子值最低, Group n_groups = 因子值最高。

    返回的 DataFrame 新增 ``group`` 列。
    """
    df = df.copy()
    df["group"] = np.nan

    for date, idx in df.groupby(date_col).groups.items():
        group_df = df.loc[idx].dropna(subset=[factor_col])
        if len(group_df) < n_groups:
            df.loc[idx, "group"] = 1
            continue
        df.loc[idx, "group"] = pd.qcut(
            group_df[factor_col],
            q=n_groups,
            labels=range(1, n_groups + 1),
            duplicates="drop",
        )

    df["group"] = df["group"].astype(int)
    return df


def compute_group_returns(
    df: pd.DataFrame,
    return_col: str = "forward_return_1m",
    group_col: str = "group",
    date_col: str = "date",
    weight_col: str | None = None,
) -> pd.DataFrame:
    """
    计算每组每期的等权(或加权)收益。

    返回
    ----
    pd.DataFrame
        列: date, group, return, [n_stocks]
    """
    results = []
    for (date, group), gdf in df.groupby([date_col, group_col]):
        valid = gdf.dropna(subset=[return_col])
        if len(valid) == 0:
            continue
        if weight_col and weight_col in valid.columns:
            ret = np.average(
                valid[return_col], weights=valid[weight_col]
            )
        else:
            ret = valid[return_col].mean()
        results.append(
            {
                "date": date,
                "group": group,
                "return": ret,
                "n_stocks": len(valid),
            }
        )
    return pd.DataFrame(results)


def compute_long_short_returns(
    group_returns: pd.DataFrame,
    long_group: int = 5,
    short_group: int = 1,
) -> pd.DataFrame:
    """
    计算多空组合收益 = long_group 收益 - short_group 收益。

    返回
    ----
    pd.DataFrame
        列: date, return
    """
    pivot = group_returns.pivot_table(
        index="date", columns="group", values="return"
    )
    if long_group not in pivot.columns or short_group not in pivot.columns:
        raise ValueError(
            f"分组 {long_group} 或 {short_group} 不在收益表中"
        )
    ls_ret = pivot[long_group] - pivot[short_group]
    return ls_ret.reset_index(name="long_short_return")
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_group_backtest.py -v
# Expected: 3 tests PASS
```

- [ ] **Step 4: Commit**

```bash
git add factor_research/group_backtest.py tests/test_group_backtest.py
git commit -m "feat: add group backtest engine (quantile grouping, long-short returns)"
```

---

### Task 9: 组合回测与绩效评估

**Files:**
- Create: `factor_research/backtest_engine.py`

- [ ] **Step 1: 实现 backtest_engine.py**

```python
"""
多因子组合回测引擎。
因子合成 → 选股 → 组合净值计算 → 绩效评估。
"""

import pandas as pd
import numpy as np


def combine_factors(
    factor_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    method: str = "equal_weight",
) -> pd.DataFrame:
    """
    多因子合成。

    参数
    ----
    factor_df : pd.DataFrame
        宽表格式: date, symbol, factor_A, factor_B, ...
    weights : dict | None
        各因子权重, method="weighted" 时使用
    method : str
        "equal_weight": 等权合成
        "ic_weighted": 各因子按历史 IC_IR 加权(需在因子列名中包含 IC_IR)

    返回
    ----
    pd.DataFrame
        新增 ``composite_factor`` 列
    """
    df = factor_df.copy()
    factor_cols = [
        c
        for c in df.columns
        if c not in ["date", "symbol", "composite_factor"]
    ]

    if method == "equal_weight":
        df["composite_factor"] = df[factor_cols].mean(axis=1)

    elif method == "weighted" and weights:
        df["composite_factor"] = sum(
            df[col] * w for col, w in weights.items() if col in df.columns
        )

    else:
        df["composite_factor"] = df[factor_cols].mean(axis=1)

    return df


def compute_nav(
    returns: pd.Series,
    initial_value: float = 1.0,
) -> pd.Series:
    """
    从收益序列计算累计净值。

    参数
    ----
    returns : pd.Series
        每期收益率 (非对数)
    initial_value : float
        初始净值

    返回
    ----
    pd.Series
        累计净值序列
    """
    return initial_value * (1 + returns).cumprod()


def compute_performance_metrics(
    returns: pd.Series,
    freq: str = "M",
    rf: float = 0.02,
) -> dict:
    """
    计算常见绩效指标。

    参数
    ----
    returns : pd.Series
        组合每期收益
    freq : str
        频率: "M" (月度), "D" (日度)
    rf : float
        无风险利率 (年化), 默认 2%

    返回
    ----
    dict
        Annualized_Return, Volatility, Sharpe_Ratio,
        Max_Drawdown, Calmar_Ratio, Win_Rate
    """
    if freq == "M":
        periods_per_year = 12
    elif freq == "D":
        periods_per_year = 252
    else:
        periods_per_year = 12

    ann_return = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)

    # Sharpe: (年化超额收益) / 年化波动率
    excess_return = ann_return - rf
    sharpe = excess_return / ann_vol if ann_vol > 0 else 0

    # 最大回撤
    nav = compute_nav(returns)
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_dd = drawdown.min()

    # Calmar
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0

    # 胜率
    win_rate = (returns > 0).sum() / len(returns)

    return {
        "Annualized_Return": round(ann_return, 4),
        "Volatility": round(ann_vol, 4),
        "Sharpe_Ratio": round(sharpe, 4),
        "Max_Drawdown": round(max_dd, 4),
        "Calmar_Ratio": round(calmar, 4),
        "Win_Rate": round(win_rate, 4),
        "Periods": len(returns),
    }
```

- [ ] **Step 2: Commit**

```bash
git add factor_research/backtest_engine.py
git commit -m "feat: add portfolio backtest engine and performance metrics"
```

---

### Task 10: 可视化与报告生成

**Files:**
- Create: `factor_research/report.py`

- [ ] **Step 1: 实现 report.py**

```python
"""
因子研究报告可视化组件。
IC 时序图、IC 分布直方图、分组净值曲线、相关性热力图。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 中文字体设置
matplotlib.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_ic_timeseries(
    ic_series: pd.Series,
    title: str = "Rank IC 时序图",
    figsize=(12, 5),
):
    """绘制 IC 时序 + 累计 IC"""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    # 上: IC 柱状图
    colors = [
        "#d32f2f" if v < 0 else "#388e3c" for v in ic_series.values
    ]
    ax1.bar(range(len(ic_series)), ic_series.values, color=colors, width=0.8)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.axhline(
        y=ic_series.mean(),
        color="steelblue",
        linestyle="--",
        linewidth=1.5,
        label=f"均值: {ic_series.mean():.4f}",
    )
    ax1.set_ylabel("Rank IC")
    ax1.set_title(title)
    ax1.legend()

    # 下: 累计 IC
    cum_ic = ic_series.cumsum()
    ax2.plot(
        range(len(cum_ic)),
        cum_ic.values,
        color="steelblue",
        linewidth=1.5,
    )
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_ylabel("累计 IC")
    ax2.set_xlabel("期数")

    plt.tight_layout()
    return fig


def plot_group_nav(
    group_returns: pd.DataFrame,
    title: str = "分层回测净值",
    figsize=(12, 5),
):
    """绘制各分组累计净值曲线 + 多空净值"""
    fig, ax = plt.subplots(figsize=figsize)

    pivot = group_returns.pivot_table(
        index="date", columns="group", values="return"
    )
    nav = (1 + pivot).cumprod()

    # 颜色渐变: Q1(红) → Q5(绿)
    colors = ["#d32f2f", "#ff7043", "#ffc107", "#81c784", "#388e3c"]
    for i, group in enumerate(sorted(nav.columns)):
        ax.plot(
            nav.index,
            nav[group],
            label=f"Q{group}",
            color=colors[i],
            linewidth=1.5,
        )

    # 多空
    if 5 in nav.columns and 1 in nav.columns:
        ls_nav = (1 + pivot[5] - pivot[1]).cumprod()
        ax.plot(
            ls_nav.index,
            ls_nav,
            label="Long-Short (Q5-Q1)",
            color="black",
            linewidth=2,
            linestyle="--",
        )

    ax.set_ylabel("累计净值")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_factor_correlation(
    factor_df: pd.DataFrame,
    factor_cols: list[str],
    figsize=(10, 8),
):
    """因子相关性热力图"""
    corr = factor_df[factor_cols].corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title("因子截面相关性矩阵")
    plt.tight_layout()
    return fig


def plot_ic_distribution(
    ic_series: pd.Series,
    title: str = "IC 分布",
    figsize=(10, 5),
):
    """IC 分布直方图"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        ic_series.values,
        bins=20,
        color="steelblue",
        edgecolor="white",
        alpha=0.8,
    )
    ax.axvline(
        x=0, color="red", linestyle="--", linewidth=1, label="IC=0"
    )
    ax.axvline(
        x=ic_series.mean(),
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"均值: {ic_series.mean():.4f}",
    )
    ax.set_xlabel("Rank IC")
    ax.set_ylabel("频数")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig
```

- [ ] **Step 2: Commit**

```bash
git add factor_research/report.py
git commit -m "feat: add visualization and report generation"
```

---

### Task 11: Run Pipeline — 因子研究完整 Notebook

**Files:**
- Create: `notebooks/00_data_overview.ipynb`
- Create: `notebooks/01_factor_report.ipynb`
- Modify: `factor_research/__init__.py`

- [ ] **Step 1: 创建 00_data_overview.ipynb**

This is a Jupyter notebook — we'll create it as a Python script first and convert.

```python
# %% [markdown]
# # 数据概览
# 本 Notebook 获取并展示 A 股数据的基本情况。

# %%
import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data.fetcher import Fetcher

f = Fetcher()

# %% [markdown]
# ## 1. 指数成分股概览

# %%
csi800 = f.get_index_members("000906")
csi300 = f.get_index_members("000300")
csi500 = f.get_index_members("000905")
print(f"沪深300: {len(csi300)} 只")
print(f"中证500: {len(csi500)} 只")
print(f"中证800: {len(csi800)} 只")

# %% [markdown]
# ## 2. 日线行情示例

# %%
sample = f.get_daily("000001", "20170101", "20241231")
print(f"平安银行(000001) 日线数据: {len(sample)} 条")
print(sample.head())
print(f"\n日期范围: {sample['日期'].min()} ~ {sample['日期'].max()}")
sample[["日期", "收盘"]].set_index("日期").plot(
    figsize=(14, 4), title="平安银行 收盘价"
)
plt.show()

# %% [markdown]
# ## 3. 数据质量检查

# %%
# 检查部分股票的停牌天数
import random
symbols = csi300[:10]  # 前 10 只沪深 300
for sym in symbols:
    try:
        df = f.get_daily(sym, "20230101", "20231231")
        # 换手率为 0 的天数
        zero_vol = (df["成交量"].fillna(0) == 0).sum()
        print(f"{sym}: {len(df)} 个交易, 零成交量 {zero_vol} 天")
    except Exception as e:
        print(f"{sym}: 获取失败 - {e}")
```

- [ ] **Step 2: 创建 01_factor_report.ipynb (作为 .py 脚本)**

```python
# %% [markdown]
# # 因子研究报告
# 完整的因子研究流水线: 因子计算 → IC 分析 → 分层回测 → 多因子合成

# %%
import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from factor_research.ic_analysis import compute_rank_ic, compute_ic_summary
from factor_research.group_backtest import (
    assign_quantile_groups,
    compute_group_returns,
    compute_long_short_returns,
)
from factor_research.backtest_engine import (
    combine_factors,
    compute_performance_metrics,
    compute_nav,
)
from factor_research.report import (
    plot_ic_timeseries,
    plot_group_nav,
    plot_factor_correlation,
    plot_ic_distribution,
)

# %% [markdown]
# ## 加载预处理后的因子数据
# (此处假设数据已通过 pipeline 生成)
# 实际使用时替换为真实数据加载路径

# %%
# df = pd.read_csv("../data/processed/panel_data.csv")
# print(f"因子面板数据: {df.shape}")
# print(df.head())

# %% [markdown]
# ## 1. 单因子 IC 分析

# %%
# 示例: 用模拟数据演示
# TODO: 替换为真实因子数据
rng = np.random.default_rng(42)
# ...
# ic = compute_rank_ic(df, factor_col="BP", return_col="forward_return_1m")
# summary = compute_ic_summary(ic)
# print("BP 因子 IC 汇总:")
# for k, v in summary.items():
#     print(f"  {k}: {v}")

# %% [markdown]
# ## 2. IC 可视化

# %%
# fig = plot_ic_timeseries(ic, title="BP 因子 Rank IC 时序")
# plt.show()
# fig2 = plot_ic_distribution(ic)
# plt.show()

# %% [markdown]
# ## 3. 分层回测

# %%
# assigned = assign_quantile_groups(df, factor_col="BP", n_groups=5)
# group_rets = compute_group_returns(assigned, return_col="forward_return_1m")
# fig3 = plot_group_nav(group_rets, title="BP 分层回测净值")
# plt.show()

# %% [markdown]
# ## 4. 绩效指标

# %%
# ls_returns = compute_long_short_returns(group_rets)
# metrics = compute_performance_metrics(ls_returns["long_short_return"])
# print("多空组合绩效:")
# for k, v in metrics.items():
#     print(f"  {k}: {v}")

# %% [markdown]
# ## 5. 多因子相关性

# %%
# factor_cols = [
#     "BP", "EP", "Mom_1M", "Mom_3M", "Mom_6M", "Mom_12M_1M",
#     "Vol_20D", "Vol_60D", "Downside_Vol", "Beta",
#     "ROE", "Gross_Margin", "Debt_Ratio",
#     "Rev_Growth_YoY", "Earnings_Growth",
# ]
# fig4 = plot_factor_correlation(df, factor_cols)
# plt.show()
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/ factor_research/__init__.py
git commit -m "feat: add research report notebooks"
```

---

### Task 12: ML 特征工程与训练框架

**Files:**
- Create: `ml_selection/features.py`
- Create: `ml_selection/training.py`

- [ ] **Step 1: 实现 features.py**

```python
"""
ML 选股特征工程。
将因子面板数据转换为 ML 就绪的 (X, y) 数据集。
"""

import pandas as pd
import numpy as np


def build_feature_matrix(
    panel_df: pd.DataFrame,
    factor_cols: list[str],
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    构建特征矩阵。

    参数
    ----
    panel_df : pd.DataFrame
        因子面板数据: date, symbol, 各因子列
    factor_cols : list[str]
        作为特征使用的因子列
    extra_cols : list[str] | None
        额外的特征列 (如行业 dummies)

    返回
    ----
    pd.DataFrame
        特征矩阵 X, 带有 date 和 symbol 索引
    """
    all_cols = ["date", "symbol"] + factor_cols
    if extra_cols:
        all_cols += extra_cols
    X = panel_df[all_cols].copy()
    return X


def build_target(
    panel_df: pd.DataFrame,
    return_col: str = "forward_return_1m",
    mode: str = "regression",
    threshold: float = 0.0,
) -> pd.Series:
    """
    构建训练目标。

    参数
    ----
    panel_df : pd.DataFrame
        包含 forward_return 列
    return_col : str
        下期收益列名
    mode : str
        "regression": 回归目标 (原始收益率)
        "classification": 分类目标 (收益 > threshold → 1, else 0)
    threshold : float
        分类模式的阈值

    返回
    ----
    pd.Series
        目标变量 y
    """
    if mode == "regression":
        return panel_df[return_col].copy()
    elif mode == "classification":
        return (panel_df[return_col] > threshold).astype(int)
    else:
        raise ValueError(f"不支持的模式: {mode}")


def train_test_split_by_time(
    X: pd.DataFrame,
    y: pd.Series,
    train_periods: int = 60,
    gap: int = 0,
) -> tuple:
    """
    按时间切分训练/测试集 (滚动窗口)。
    每次用最近 train_periods 期训练, 预测下一期。

    这是一个 generator, 每次 yield (X_train, y_train, X_test, y_test, test_date)。

    参数
    ----
    X : pd.DataFrame
        包含 "date" 列
    y : pd.Series
        目标, 与 X 对齐
    train_periods : int
        训练期数 (月)
    gap : int
        训练结束与测试开始的间隔期数 (防止前看偏差)

    Yields
    ------
    tuple
        (X_train, X_test, y_train, y_test, test_date)
    """
    dates = sorted(X["date"].unique())
    if len(dates) <= train_periods + gap:
        raise ValueError(
            f"数据不足以划分训练/测试: {len(dates)} 期, "
            f"需 train_periods={train_periods} + gap={gap}"
        )

    for i in range(train_periods, len(dates) - gap):
        train_dates = dates[i - train_periods : i]
        test_date = dates[i + gap]

        train_mask = X["date"].isin(train_dates)
        test_mask = X["date"] == test_date

        X_train = X.loc[train_mask].drop(
            columns=["date", "symbol"]
        )
        X_test = X.loc[test_mask].drop(
            columns=["date", "symbol"]
        )
        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]

        # 只保留完整无 NA 的样本
        train_valid = X_train.notna().all(axis=1)
        test_valid = X_test.notna().all(axis=1)

        X_train = X_train.loc[train_valid]
        y_train = y_train.loc[train_valid]
        X_test = X_test.loc[test_valid]
        y_test = y_test.loc[test_valid]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        yield X_train, X_test, y_train, y_test, test_date
```

- [ ] **Step 2: 实现 training.py**

```python
"""
滚动窗口训练框架。
在因子面板数据上依次执行 expanding window / rolling window 训练。
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    params: dict | None = None,
) -> tuple[np.ndarray, object]:
    """
    训练 LightGBM 回归模型并预测。

    返回 (predictions, model)
    """
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 200,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    params: dict | None = None,
) -> tuple[np.ndarray, object]:
    """训练 XGBoost 回归模型并预测。"""
    default_params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 0,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, model


def run_rolling_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    model_fn,
    train_periods: int = 60,
    gap: int = 0,
) -> pd.DataFrame:
    """
    执行完整滚动窗口回测, 收集所有预测。

    参数
    ----
    X : pd.DataFrame
        包含 date, symbol 列
    y : pd.Series
        目标变量
    model_fn : callable
        训练函数, 签名: fn(X_train, y_train, X_test) → predictions
    train_periods : int
        滚动训练窗口期数
    gap : int
        训练期与预测期的间隔

    返回
    ----
    pd.DataFrame
        列: date, symbol, prediction
    """
    from ml_selection.features import train_test_split_by_time

    all_preds = []
    dates = sorted(X["date"].unique())

    for X_train, X_test, y_train, y_test, test_date in tqdm(
        train_test_split_by_time(X, y, train_periods, gap),
        total=len(dates) - train_periods - gap,
        desc="Rolling Backtest",
    ):
        try:
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )

            preds, _ = model_fn(
                X_train_scaled, y_train, X_test_scaled
            )

            test_symbols = X.loc[X_test.index, "symbol"]
            all_preds.extend(
                [
                    {
                        "date": test_date,
                        "symbol": sym,
                        "prediction": float(pred),
                    }
                    for sym, pred in zip(test_symbols, preds)
                ]
            )
        except Exception as e:
            print(f"跳过 {test_date}: {e}")
            continue

    return pd.DataFrame(all_preds)


def compute_prediction_ic(
    predictions: pd.DataFrame,
    actual_returns: pd.DataFrame,
    return_col: str = "forward_return_1m",
) -> pd.Series:
    """
    计算模型预测值的 Rank IC (与因子 IC 同口径对比)。
    """
    from factor_research.ic_analysis import compute_rank_ic

    merged = predictions.merge(
        actual_returns[["date", "symbol", return_col]],
        on=["date", "symbol"],
        how="inner",
    )
    return compute_rank_ic(
        merged,
        factor_col="prediction",
        return_col=return_col,
        date_col="date",
    )
```

- [ ] **Step 3: Commit**

```bash
git add ml_selection/
git commit -m "feat: add ML feature engineering and rolling training framework"
```

---

### Task 13: 风险模型 — 波动率建模

**Files:**
- Create: `risk_model/volatility.py`

- [ ] **Step 1: 实现 GARCH 波动率建模**

```python
"""
波动率建模。
GARCH / EGARCH 拟合个股波动率, 与历史波动率对比。
"""

import pandas as pd
import numpy as np
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    mean: str = "constant",
    dist: str = "normal",
) -> dict:
    """
    对收益率序列拟合 GARCH(p,q) 模型。

    参数
    ----
    returns : pd.Series
        日对数收益率序列
    p : int
        GARCH 阶数
    q : int
        ARCH 阶数
    mean : str
        均值模型
    dist : str
        残差分布: "normal", "t", "skewt"

    返回
    ----
    dict
        keys: conditional_vol, params, aic, bic, model (fitted model object)
    """
    # 缩放到百分比以帮助数值优化
    scaled_returns = returns * 100

    try:
        model = arch_model(
            scaled_returns,
            vol="GARCH",
            p=p,
            q=q,
            mean=mean,
            dist=dist,
        )
        fitted = model.fit(disp="off", show_warning=False)

        # conditional vol 缩回小数
        cond_vol = fitted.conditional_volatility / 100

        return {
            "conditional_vol": cond_vol,
            "params": fitted.params.to_dict(),
            "aic": fitted.aic,
            "bic": fitted.bic,
            "model": fitted,
        }
    except Exception as e:
        raise RuntimeError(
            f"GARCH({p},{q}) 拟合失败: {e}"
        )


def fit_egarch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
) -> dict:
    """
    拟合 EGARCH(1,1) 模型 (捕捉杠杆效应).
    """
    scaled_returns = returns * 100

    try:
        model = arch_model(
            scaled_returns,
            vol="EGARCH",
            p=p,
            q=q,
            mean="constant",
        )
        fitted = model.fit(disp="off", show_warning=False)
        cond_vol = fitted.conditional_volatility / 100

        return {
            "conditional_vol": cond_vol,
            "params": fitted.params.to_dict(),
            "aic": fitted.aic,
            "bic": fitted.bic,
            "model": fitted,
        }
    except Exception as e:
        raise RuntimeError(f"EGARCH({p},{q}) 拟合失败: {e}")


def compare_vol_models(
    returns: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """
    对比 GARCH 预测波动率 vs 历史滚动波动率。

    返回
    ----
    pd.DataFrame
        列: date, historical_vol, garch_vol
    """
    # 历史波动率(滚动)
    hist_vol = (
        returns.rolling(window).std() * np.sqrt(252)
    )

    # GARCH(1,1) 拟合
    garch_result = fit_garch(returns, p=1, q=1)
    garch_vol = (
        garch_result["conditional_vol"] * np.sqrt(252)
    )

    result = pd.DataFrame(
        {
            "historical_vol": hist_vol,
            "garch_vol": garch_vol,
        },
        index=returns.index,
    )
    return result.dropna()
```

- [ ] **Step 2: Commit**

```bash
git add risk_model/volatility.py
git commit -m "feat: add GARCH/EGARCH volatility modeling"
```

---

### Task 14: 风险模型 — 协方差估计与 VaR

**Files:**
- Create: `risk_model/covariance.py`
- Create: `risk_model/var_backtest.py`

- [ ] **Step 1: 实现 covariance.py**

```python
"""
协方差矩阵估计。
Ledoit-Wolf Shrinkage vs 样本协方差。
"""

import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf


def estimate_sample_cov(
    returns: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    样本协方差矩阵。

    参数
    ----
    returns : pd.DataFrame
        列=资产, 行=时间, 值为收益率

    返回
    ----
    (cov_matrix, corr_matrix) : ndarray, ndarray
    """
    cov = returns.cov().values
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return cov, corr


def estimate_shrinkage_cov(
    returns: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ledoit-Wolf Shrinkage 协方差估计。
    收缩样本协方差朝向结构化目标(恒等或单因子),
    提升样本外稳定性。

    返回
    ----
    (cov_matrix, corr_matrix) : ndarray, ndarray
    """
    lw = LedoitWolf()
    # 去除全 NaN 列
    clean_returns = returns.dropna(axis=1, how="all").fillna(0)
    cov = lw.fit(clean_returns).covariance_
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std + 1e-8)
    return cov, corr


def compute_min_variance_weights(
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """
    给定协方差矩阵, 计算最小方差组合权重。

    w = inv(Sigma) * 1 / (1^T * inv(Sigma) * 1)
    """
    try:
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_matrix)

    ones = np.ones(len(cov_matrix))
    numerator = inv_cov @ ones
    denominator = ones @ inv_cov @ ones
    if denominator <= 0:
        return ones / len(ones)  # fallback: 等权
    return numerator / denominator
```

- [ ] **Step 2: 实现 var_backtest.py**

```python
"""
VaR (Value at Risk) 回测。
参数法、历史模拟法, 以及 Kupiec 检验。
"""

import pandas as pd
import numpy as np
from scipy import stats


def var_parametric(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
) -> pd.Series:
    """
    参数法 VaR: VaR = μ + σ * z_α
    使用滚动窗口估计 μ 和 σ。
    """
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std()
    z = stats.norm.ppf(1 - confidence)
    var = -(mu + z * sigma)  # VaR 为正数表示损失
    return var.dropna()


def var_historical(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
) -> pd.Series:
    """历史模拟法 VaR: 滚动窗口内收益率的分位数。"""
    var = returns.rolling(window).quantile(1 - confidence)
    return -var.dropna()


def var_backtest(
    returns: pd.Series,
    var_series: pd.Series,
    confidence: float = 0.95,
) -> dict:
    """
    VaR 回测: 比较实际亏损超过 VaR 的天数是否与置信水平一致。

    返回
    ----
    dict
        keys:
        - exceedances: 亏损超过 VaR 的天数
        - total: 总观测天数
        - exceedance_rate: 超出比例
        - expected_rate: 期望超出比例 (1-confidence)
        - kupiec_stat: Kupiec 检验统计量
        - kupiec_pvalue: Kupiec 检验 p 值
    """
    # 对齐
    common_idx = returns.dropna().index.intersection(
        var_series.dropna().index
    )
    actual_losses = -returns.loc[common_idx]
    var_values = var_series.loc[common_idx]

    exceed = (actual_losses > var_values).sum()
    total = len(common_idx)
    exceed_rate = exceed / total if total > 0 else 0
    expected_rate = 1 - confidence

    # Kupiec 似然比检验
    if total > 0 and 0 < exceed_rate < 1:
        p0 = expected_rate
        p1 = exceed_rate
        lr = (
            -2
            * (
                (total - exceed) * np.log((1 - p0) / (1 - p1))
                + exceed * np.log(p0 / p1)
            )
            if p1 > 0
            else 0
        )
        p_value = 1 - stats.chi2.cdf(lr, df=1)
    else:
        kupiec_stat = np.nan
        p_value = np.nan

    return {
        "exceedances": int(exceed),
        "total": int(total),
        "exceedance_rate": round(exceed_rate, 4),
        "expected_rate": round(expected_rate, 4),
        "kupiec_pvalue": (
            round(p_value, 4) if not np.isnan(p_value) else None
        ),
    }
```

- [ ] **Step 3: Commit**

```bash
git add risk_model/
git commit -m "feat: add covariance estimation and VaR backtesting"
```

---

### Task 15: 中英文简历

**Files:**
- Create: `resume/resume_cn.md`
- Create: `resume/resume_en.md`

- [ ] **Step 1: 编写中文简历**

```markdown
# 简历

## 基本信息

- **姓名**: [你的姓名]
- **电话**: [手机号]
- **邮箱**: [邮箱]
- **GitHub**: [GitHub 链接, 指向本项目]

---

## 教育背景

**复旦大学** | 数学科学学院 | 硕士研究生 | 2025.09 – 至今 (预计 2028.06)
- 专业方向: 概率论与数理统计
- 核心课程: 随机分析, 高等概率论, 随机过程, 高等数理统计, 测度论
- GPA: [如果有的话]

**浙江大学** | 数学科学学院 | 本科 | 2021.09 – 2025.06
- 专业: 数学与应用数学
- 核心课程: 实变函数, 泛函分析, 概率论, 数理统计, 偏微分方程, 数值分析
- 荣誉/奖项: [如: 国家奖学金, 数学竞赛奖, 优秀毕业生等]

---

## 量化项目

### 多因子选股研究系统 (Python, pandas, akshare) | 2026.05 – 至今
- 从零构建 A 股量化因子研究流水线, 覆盖估值、动量、质量、波动率、成长五大类 15 个候选因子
- 实现完整因子检验体系: Rank IC 分析, IC_IR, 分层回测 (5 分组), 因子相关性矩阵
- 构建多因子组合回测引擎, 回测区间 2017–2024, 年化 Sharpe Ratio [待补充]
- 数据来源: akshare 免费 A 股 API, 股票池中证 800

### 机器学习选股模型 (LightGBM / XGBoost, scikit-learn) | 2026.06 – 至今
- 将因子研究中的多因子池作为特征, 构建 XGBoost 和 LightGBM 滚动训练选股框架
- 实现 expanding window 训练流水线 (60 个月训练 → 下月预测), 避免前看偏差
- 特征重要性分析 (SHAP) 与因子 IC 交叉验证, 确保模型可解释性
- 模型 Rank IC 显著优于线性因子合成基准 [待补充数据]

### 风险建模与 VaR 回测 | 2026.07 – 至今
- 基于 GARCH/EGARCH 族模型对个股波动率建模, 捕捉波动率聚集和杠杆效应
- 实现 Ledoit-Wolf Shrinkage 协方差估计, 对比样本协方差在样本外的表现
- 参数法/历史模拟法 VaR 回测 + Kupiec 检验, 验证风险模型有效性

---

## 研究经历

[如果有本科论文/科研经历, 写在这里。重点突出数学/统计/概率相关的工作]
- **[论文标题/课题名称]** | 导师: [导师姓名] | [时间]
  - 用到了哪些方法? (随机分析/蒙特卡洛/时间序列等)
  - 得出了什么结论?

---

## 技能

- **数学**: 随机分析 (伊藤积分, SDE, Girsanov 定理), 概率论, 统计推断, 线性模型, 蒙特卡洛方法
- **编程**: Python (pandas, numpy, scikit-learn, matplotlib), Jupyter, LaTeX
- **数据工具**: akshare, baostock, (如会用: Wind 终端操作)
- **版本控制**: Git
- **语言**: 中文 (母语), 英语 (读写流利)

---

## 其他

- [奖项, 证书, 课外活动等]
```

- [ ] **Step 2: 编写英文简历**

```markdown
# Resume

## Contact
- **Name**: [Your Name]
- **Phone**: [Phone]
- **Email**: [Email]
- **GitHub**: [GitHub link]

---

## Education

**Fudan University** | School of Mathematical Sciences | M.Sc. | 2025 – Present (Expected 2028)
- Concentration: Probability Theory and Mathematical Statistics
- Core Courses: Stochastic Analysis, Advanced Probability, Stochastic Processes, Advanced Mathematical Statistics, Measure Theory

**Zhejiang University** | School of Mathematical Sciences | B.Sc. | 2021 – 2025
- Major: Mathematics and Applied Mathematics

---

## Quantitative Projects

### Multi-Factor Equity Research System (Python, pandas) | 2026.05 – Present
- Built end-to-end A-share factor research pipeline covering 15 factors across 5 categories:
  Value, Momentum, Quality, Volatility, and Growth
- Implemented comprehensive factor testing: Rank IC analysis, IC_IR,
  quintile group backtesting, factor correlation matrix
- Constructed multi-factor portfolio backtesting engine, 2017–2024, CSI 800 universe
- Annualized Sharpe Ratio: [TBD]

### Machine Learning Stock Selection (LightGBM / XGBoost) | 2026.06 – Present
- Built rolling-window training framework (60-month train → next-month predict)
  on multi-factor feature set
- Implemented expanding window pipeline preventing look-ahead bias
- SHAP-based feature importance analysis cross-validated with factor IC
- Model Rank IC significantly outperforms linear factor combination baseline

### Risk Modeling & VaR Backtesting | 2026.07 – Present
- GARCH/EGARCH volatility modeling capturing volatility clustering and leverage effects
- Ledoit-Wolf Shrinkage covariance estimation vs. sample covariance out-of-sample
- Parametric & historical VaR backtesting with Kupiec test

---

## Skills
- **Mathematics**: Stochastic Analysis, Probability Theory, Statistical Inference, Monte Carlo Methods
- **Programming**: Python (pandas, numpy, scikit-learn, Jupyter), LaTeX
- **Languages**: Chinese (Native), English (Fluent)
```

- [ ] **Step 3: Commit**

```bash
git add resume/
git commit -m "feat: add Chinese and English resumes"
```

---

### Task 16: 最终集成与 README 更新

- [ ] **Step 1: 更新 README.md 补充运行指南**

```bash
# 确认所有测试通过
pytest tests/ -v
```

- [ ] **Step 2: 运行完整数据流水线测试**

```bash
cd /c/Users/HzhJa/Desktop/quant
python -c "
from data.fetcher import Fetcher
f = Fetcher()
# 测试获取沪深300
members = f.get_index_members('000300')
print(f'沪深300成分股: {len(members)} 只')
print(f'前5只: {members[:5]}')
"
```

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "chore: finalize project, update README, all tests passing"
```

---

## 实施顺序建议

按 Task 编号依次执行: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16

关键里程碑:
- Task 0–4: 数据基础打通
- Task 5–6: 因子库完成
- Task 7–10: 因子研究完整流水线 (此时可投递简历)
- Task 11: 可视化报告
- Task 12: ML 选股
- Task 13–14: 风险模型
- Task 15: 简历完善
- Task 16: 终验
