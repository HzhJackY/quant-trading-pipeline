"""
Fetch Shenwan Industry Classification for CSI 800 Universe via Baostock.

One-time execution. Caches result to data/sw_industry.parquet.

Usage:
  python fetch_sw_industry.py
"""

import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_sw")

import pandas as pd
import baostock as bs


# CSRC-to-Shenwan L1 mapping
SW_MAP = {
    "银行": "银行", "银行业": "银行", "货币金融": "银行", "货币金融服务": "银行",
    "证券": "非银金融", "保险": "非银金融", "资本市场服务": "非银金融",
    "房地产": "房地产", "房地产业": "房地产",
    "建筑": "建筑装饰", "建筑装饰": "建筑装饰", "土木工程": "建筑装饰", "建筑业": "建筑装饰",
    "有色金属": "有色金属", "有色": "有色金属",
    "钢铁": "钢铁", "黑色金属": "钢铁",
    "化工": "基础化工", "化学": "基础化工", "基础化工": "基础化工",
    "石油": "石油石化", "石化": "石油石化",
    "煤炭": "煤炭", "采掘": "煤炭", "开采辅助": "煤炭",
    "电力设备": "电力设备", "电气机械": "电力设备", "新能源": "电力设备",
    "机械设备": "机械设备", "机械": "机械设备", "专用设备": "机械设备", "通用设备": "机械设备",
    "国防军工": "国防军工", "军工": "国防军工", "航空航天": "国防军工", "铁路船舶": "国防军工",
    "汽车": "汽车", "汽车制造业": "汽车",
    "电子": "电子", "半导体": "电子", "计算机通信": "电子", "元器件": "电子",
    "计算机": "计算机", "软件": "计算机", "信息技术": "计算机", "互联网": "计算机",
    "通信": "通信",
    "传媒": "传媒", "文化": "传媒", "新闻出版": "传媒",
    "食品饮料": "食品饮料", "食品": "食品饮料", "酒饮料": "食品饮料", "白酒": "食品饮料",
    "家用电器": "家用电器", "家电": "家用电器",
    "纺织服装": "纺织服装", "纺织": "纺织服装", "服装": "纺织服装", "服饰": "纺织服装",
    "轻工制造": "轻工制造", "造纸": "轻工制造", "家具": "轻工制造", "文教": "轻工制造",
    "医药": "医药生物", "医药生物": "医药生物", "医药制造": "医药生物",
    "美容": "美容护理", "化妆品": "美容护理",
    "旅游": "社会服务", "酒店": "社会服务", "餐饮": "社会服务",
    "农业": "农林牧渔", "农林牧渔": "农林牧渔", "畜牧业": "农林牧渔", "渔业": "农林牧渔",
    "公用事业": "公用事业", "电力": "公用事业", "水务": "公用事业", "燃气": "公用事业",
    "交通运输": "交通运输", "运输": "交通运输", "铁路": "交通运输", "航空": "交通运输", "水上运输": "交通运输",
    "商贸零售": "商贸零售", "零售": "商贸零售", "批发": "商贸零售", "商业": "商贸零售",
    "环保": "环保", "生态": "环保",
    "建筑材料": "建筑材料", "建材": "建筑材料",
}


def to_bs_code(code: str) -> str:
    code = str(code).zfill(6)
    return f"sh.{code}" if code.startswith(("6", "9")) else f"sz.{code}"


def map_to_sw(name: str) -> str:
    name = str(name).strip()
    for key, val in SW_MAP.items():
        if key in name:
            return val
    return "综合"


def main():
    # Load symbols from V2 panel
    panel = pd.read_parquet("output/training_panel_v3_full.parquet", columns=["symbol"])
    symbols = sorted(panel["symbol"].unique())
    logger.info("Fetching industry for %d symbols...", len(symbols))

    bs.login()
    rows = []
    t0 = time.perf_counter()

    for i, sym in enumerate(symbols):
        try:
            rs = bs.query_stock_industry(to_bs_code(sym))
            if rs.error_code == "0":
                while rs.next():
                    row_data = rs.get_row_data()
                    ind_name = row_data[3] if len(row_data) > 3 else row_data[2]
                    rows.append({"symbol": sym, "sw_l1": map_to_sw(ind_name)})
                    break
        except Exception:
            pass

        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            logger.info("  %d/%d (%.1fs, %.0f sym/s)", i + 1, len(symbols), elapsed, (i + 1) / elapsed)

    bs.logout()

    result = pd.DataFrame(rows)
    result.to_parquet("data/sw_industry.parquet", index=False)
    logger.info("Done: %d/%d industries saved (%.1f%%)", len(result), len(symbols), 100 * len(result) / len(symbols))

    # Show industry distribution
    dist = result["sw_l1"].value_counts().head(15)
    logger.info("Top-15 industries:\n%s", dist.to_string())


if __name__ == "__main__":
    main()
