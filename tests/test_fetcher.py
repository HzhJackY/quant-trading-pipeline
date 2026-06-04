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
