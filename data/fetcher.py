"""
数据获取模块。
使用 akshare 获取 A 股日线行情、财务数据、指数成分股。
所有方法内置本地 CSV 缓存, 避免重复请求。
"""

import os
import pandas as pd
import certifi

# Fix SSL certificate verification on Python installations where the system
# cert bundle is missing (common on Windows with Python >= 3.13).
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# Monkey-patch requests so that every request bypasses system proxy settings
# and uses the certifi CA bundle.  This is required because akshare calls
# requests.get() internally without the ability to pass custom session
# configuration.
import requests as _requests
import requests.api as _api
import requests.sessions as _sessions

_original_request = _api.request


def _patched_request(method, url, **kwargs):
    kwargs.setdefault("verify", certifi.where())
    with _sessions.Session() as session:
        session.trust_env = False
        return session.request(method=method, url=url, **kwargs)


_api.request = _patched_request

import akshare as ak


# Column name mapping from Sina (English) to standard Chinese names used
# throughout the project.
_SINA_TO_CHINESE = {
    "date": "日期",
    "open": "开盘",
    "high": "最高",
    "low": "最低",
    "close": "收盘",
    "volume": "成交量",
    "amount": "成交额",
    "outstanding_share": "流通市值",
    "turnover": "换手率",
}


class Fetcher:
    """A 股数据获取器, 封装 akshare API 并提供缓存层。"""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "raw")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _add_prefix(symbol: str) -> str:
        """根据股票代码自动补全交易所前缀 (sh / sz / bj)."""
        code = symbol.strip()
        if code.startswith(("sh", "sz", "bj")):
            return code
        if code.startswith(("6", "5")):
            return f"sh{code}"
        elif code.startswith(("0", "2", "3")):
            return f"sz{code}"
        elif code.startswith(("8", "4")):
            return f"bj{code}"
        else:
            # 默认当作深圳
            return f"sz{code}"

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
            列: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额,
                 换手率
        """
        cache_file = os.path.join(
            self.cache_dir,
            f"daily_{symbol}_{start_date}_{end_date}_{adjust}.csv",
        )
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, parse_dates=["日期"])

        prefixed = self._add_prefix(symbol)

        df = None

        # Try Sina source first (more reliable in some network environments)
        try:
            df = ak.stock_zh_a_daily(
                symbol=prefixed,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
        except Exception:
            pass

        # Fallback to EastMoney source
        if df is None or df.empty:
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )
            except Exception:
                # Try with prefix
                try:
                    df = ak.stock_zh_a_hist(
                        symbol=prefixed,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust=adjust,
                    )
                except Exception as e:
                    raise ValueError(
                        f"无法获取 {symbol} 日线数据: {e}"
                    )

        if df is None or df.empty:
            raise ValueError(f"{symbol} 在 {start_date}-{end_date} 无数据")

        # Standardize columns: Sina source returns English names, EM returns
        # Chinese names.  Normalise to Chinese for consistency.
        df = df.rename(columns=_SINA_TO_CHINESE)

        # Ensure date column is datetime and sorted
        if "日期" in df.columns:
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.sort_values("日期").reset_index(drop=True)

        # Keep only the columns we care about
        keep_cols = [
            "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "换手率"
        ]
        df = df[[c for c in keep_cols if c in df.columns]]

        df.to_csv(cache_file, index=False)
        return df
