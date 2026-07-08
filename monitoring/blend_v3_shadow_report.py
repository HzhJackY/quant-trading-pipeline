import json
import re
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "output" / "blend_v3_shadow_live"
mon = ROOT / "output" / "blend_v3_shadow_monitoring"
price_cache = mon / "price_cache"

def normalize_symbol(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return ""
    s = re.sub(r"\.0$", "", s)
    if "." in s:
        head, tail = s.split(".", 1)
        if head.isdigit() and tail.upper() in {"SZ", "SH", "BJ", "SS"}:
            s = head
    return s.zfill(6) if s.isdigit() else s

def read_csv_text(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame({"文件状态": [f"缺失：{path}"]})
    df = pd.read_csv(path, dtype={"symbol": str, "股票代码": str})
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].map(normalize_symbol).astype(str)
    if "股票代码" in df.columns:
        df["股票代码"] = df["股票代码"].map(normalize_symbol).astype(str)
    return df

def zh_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "symbol": "股票代码",
        "name": "股票名称",
        "target_weight": "目标权重",
        "blend_rank": "综合排名",
        "blend_score": "综合分数",
        "v0_score_z": "V0 标准化分数",
        "v7_score_z": "V7 标准化分数",
        "tradability_status": "可交易性状态",
        "selection_reason": "入选原因",
        "daily_return": "日收益",
        "nav": "净值",
    })

st.set_page_config(page_title="Blend V3 Shadow 监控", layout="wide")
st.title("Blend V3 Shadow 监控面板")
st.warning("SHADOW ONLY｜仅用于影子组合观察｜不是正式生产｜不是交易指令")
st.caption("当前为影子组合，不会替代正式纸交易组合。")

st.header("候选状态")
status = mon / "shadow_monitor_latest_status.json"
status_data = {}
if status.exists():
    status_data = json.loads(status.read_text(encoding="utf-8"))
    if status_data.get("stale_price_warning"):
        st.warning("行情数据未更新，当前 NAV 可能停留在旧日期。请检查 shadow price refresh 任务。")
    st.json(status_data)
else:
    st.info(f"状态文件缺失：{status}")

st.header("行情数据状态")
refresh_status_path = price_cache / "shadow_price_refresh_status.json"
refresh_status = json.loads(refresh_status_path.read_text(encoding="utf-8")) if refresh_status_path.exists() else {}
col1, col2, col3 = st.columns(3)
col1.metric("最新特征月份", status_data.get("latest_feature_month", "n/a"))
col2.metric("最新行情日期", status_data.get("latest_price_date", "n/a"))
col3.metric("最新 NAV 日期", status_data.get("latest_nav_date", "n/a"))
col4, col5, col6 = st.columns(3)
col4.metric("行情来源", status_data.get("price_source", "n/a"))
col5.metric("是否行情过期", str(status_data.get("stale_price_warning", "n/a")))
col6.metric("过期天数", status_data.get("stale_price_days", "n/a"))
st.write(f"行情刷新任务状态：{refresh_status.get('decision', 'n/a')}")
st.write(f"失败股票数：{refresh_status.get('failed_count', 'n/a')}")

st.header("最新组合")
hp = base / "latest_shadow_holdings_live.csv"
h = zh_cols(read_csv_text(hp))
st.dataframe(h, use_container_width=True, column_config={"股票代码": st.column_config.TextColumn("股票代码")})

st.header("影子净值")
navp = mon / "shadow_daily_nav.csv"
nav = zh_cols(read_csv_text(navp))
if "date" in nav.columns and "净值" in nav.columns:
    st.line_chart(nav.set_index("date")["净值"])
st.dataframe(nav, use_container_width=True)

st.header("每日收益")
retp = mon / "shadow_daily_return_log.csv"
st.dataframe(zh_cols(read_csv_text(retp)), use_container_width=True)

st.header("可交易性检查")
tp = base / "tradability_audit_v1.csv"
t = zh_cols(read_csv_text(tp))
if "可交易性状态" in t.columns:
    st.dataframe(t[t["可交易性状态"] != "pass"], use_container_width=True, column_config={"股票代码": st.column_config.TextColumn("股票代码")})
else:
    st.dataframe(t, use_container_width=True)

st.header("与当前纸交易组合对比")
dp = mon / "shadow_vs_current_paper_diff.csv"
st.dataframe(zh_cols(read_csv_text(dp)), use_container_width=True, column_config={"股票代码": st.column_config.TextColumn("股票代码")})

st.header("风险提示")
st.write("本页面仅用于 Blend V3 影子组合观察，不生成订单，不替代当前 production 或正式纸交易组合。价格数据过期时，NAV 不代表当前运行日。")

st.header("文件状态")
for p in [hp, navp, retp, tp, dp, status]:
    st.write(f"{'存在' if p.exists() else '缺失'}：{p}")

st.header("最近更新时间")
st.write(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
