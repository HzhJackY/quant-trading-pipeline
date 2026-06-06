"""
数据获取模块。
使用 akshare 获取 A 股日线行情。
所有方法内置本地 CSV 缓存, 避免重复请求。
"""

import os
import pandas as pd
import certifi

# Fix SSL certificate verification on Python installations where the system
# cert bundle is missing (common on Windows with Python >= 3.13).
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

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
        force_refresh: bool = False,
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
        force_refresh : bool
            当为 True 时跳过缓存, 强制重新请求 (默认 False)

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
        if not force_refresh and os.path.exists(cache_file):
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
        except (ConnectionError, OSError, ValueError) as e:
            import warnings
            warnings.warn(f"Sina source failed for {symbol}, falling back: {e}")

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

    def get_index_members(self, index_code: str) -> list[str]:
        """
        获取指数成分股列表。

        参数
        ----
        index_code : str
            指数代码, 如 "000300"(沪深300), "000905"(中证500),
            "000906"(中证800), "000016"(上证50)

        返回
        ----
        list[str]
            成分股代码列表, 如 ["000001", "000002", ...]
        """
        cache_file = os.path.join(
            self.cache_dir, f"index_members_{index_code}.csv"
        )
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, dtype={"code": str})
            return df["code"].tolist()

        try:
            df = ak.index_stock_cons(index_code)
        except Exception:
            raise ValueError(f"无法获取指数成分股: {index_code}")

        if df.empty:
            raise ValueError(f"指数 {index_code} 成分股数据为空")

        # 中证指数成分股 API 返回英文列名 "品种代码", "品种名称" 等
        # 不同指数源可能不同, 取第一列数字型代码
        code_col = None
        for col in df.columns:
            col_low = col.lower()
            if any(kw in col_low for kw in ["代码", "code", "symbol", "品种"]):
                code_col = col
                break
        if code_col is None:
            code_col = df.columns[0]

        codes = (
            df[code_col]
            .astype(str)
            .str.replace(r"[^0-9]", "", regex=True)
            .str[-6:]  # 取后6位纯数字, 舍弃交易所前缀
        )
        codes = [c for c in codes if c.isdigit() and len(c) == 6]

        pd.DataFrame({"code": codes}).to_csv(cache_file, index=False)
        return codes

    def get_financial(
        self, symbol: str, report_date: str = None
    ) -> dict:
        """
        获取个股最新财务数据摘要。
        返回利润表、资产负债表核心指标。

        参数
        ----
        symbol : str
            股票代码, 如 "000001"
        report_date : str | None
            报告期, 如 "20231231"。为 None 时取最新一期

        返回
        ----
        dict
            keys: 营业收入, 营业成本, 净利润, 总资产, 总负债, 净资产, ...
        """
        cache_file = os.path.join(
            self.cache_dir,
            f"financial_{symbol}_{report_date or 'latest'}.csv",
        )
        if os.path.exists(cache_file):
            cached = pd.read_csv(cache_file)
            return cached.iloc[0].to_dict() if not cached.empty else {}

        # 获取利润表
        try:
            profit_df = ak.stock_profit_sheet_by_report_em(symbol=symbol)
        except Exception as e:
            raise ValueError(f"无法获取 {symbol} 利润表: {e}")

        # 获取资产负债表
        try:
            balance_df = ak.stock_balance_sheet_by_report_em(symbol=symbol)
        except Exception:
            balance_df = None

        # 提取最新一期的核心指标
        result = {"股票代码": symbol, "报告期": report_date or "latest"}

        # 利润表指标: 取第一行(最新报告期)对应数据
        if profit_df is not None and not profit_df.empty:
            # akshare 返回的表结构: 第一列是项目名称, 后续列是报告期
            profit_item_col = profit_df.columns[0]
            # 取最新报告期列(最后一列, 或指定 report_date)
            if report_date and report_date in profit_df.columns:
                val_col = report_date
            else:
                val_col = profit_df.columns[-1]

            profit_map = profit_df.set_index(profit_item_col)[val_col].to_dict()

            # 核心利润表指标
            for item in ["营业收入", "营业成本", "净利润", "营业利润"]:
                result[item] = self._extract_fin_value(profit_map, item)

        # 资产负债表指标
        if balance_df is not None and not balance_df.empty:
            bs_item_col = balance_df.columns[0]
            if report_date and report_date in balance_df.columns:
                val_col = report_date
            else:
                val_col = balance_df.columns[-1]

            bs_map = balance_df.set_index(bs_item_col)[val_col].to_dict()

            for item in ["资产总计", "负债合计", "归属于母公司股东权益合计"]:
                result[item] = self._extract_fin_value(bs_map, item)

        # 派生指标
        total_assets = result.get("资产总计", 0) or 0
        total_liabilities = result.get("负债合计", 0) or 0
        result["净资产"] = total_assets - total_liabilities

        pd.DataFrame([result]).to_csv(cache_file, index=False)
        return result

    @staticmethod
    def _extract_fin_value(fin_map: dict, keyword: str) -> float | None:
        """从财务字典中模糊匹配数值. akshare 的项目名称可能不完全一致."""
        for key, value in fin_map.items():
            if isinstance(key, str) and keyword in key:
                try:
                    return float(value) if value else None
                except (ValueError, TypeError):
                    return None
        return None

    def get_financial_bulk(
        self, symbols: list[str], report_date: str = None
    ) -> "pd.DataFrame":
        """
        批量获取多只股票的财务数据。

        返回
        ----
        pd.DataFrame
            每行一只股票, 列为财务指标
        """
        from tqdm import tqdm

        rows = []
        for sym in tqdm(symbols, desc="获取财务数据"):
            try:
                fin = self.get_financial(sym, report_date)
                rows.append(fin)
            except Exception:
                continue
        return pd.DataFrame(rows)
