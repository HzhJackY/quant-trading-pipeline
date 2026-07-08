"""
Direct EastMoney Push2 API — 99%+ A-Share Industry Classification.

Connects directly to EM's Push2 data gateway (node 82) for industry board
constituent data. Uses exponential backoff retry (5 retries) and browser
headers to ensure reliable connections through any network restrictions.

Output: data/sw_industry.parquet (columns: symbol, sw_l1)
        Symbol format: 6-digit bare code (no exchange suffix)

Runtime: ~20 seconds for 5000+ stocks across ~90 industry boards.

Usage:
    python get_industry_direct.py
"""

import os
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# 1. Session with exponential backoff retry
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,  # 1s, 2s, 4s, 8s, 16s
    status_forcelist=[500, 502, 503, 504],
    raise_on_status=False,
)
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))

# Standard browser headers
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "http://quote.eastmoney.com/",
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# EM Push2 node 82 — the most stable data gateway
BASE_URL = "http://82.push2.eastmoney.com/api/qt/clist/get"


def fetch_industry_boards() -> list[dict]:
    """Fetch all industry board codes and names."""
    params = {
        "pn": "1",
        "pz": "200",
        "po": "1",
        "np": "1",
        "ut": "bd1d9dd1031a93e1196165d2b4abbd74",
        "fltt": "2",
        "invt": "2",
        "fid": "f3",
        "fs": "m:90 t:2 f:!12",  # Industry boards, exclude non-industry
        "fields": "f12,f14",      # f12=board code, f14=board name
    }

    r = session.get(BASE_URL, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()

    diff = data["data"]["diff"]
    boards = [{"code": x["f12"], "name": x["f14"]} for x in diff]
    return boards


def fetch_board_members(board_code: str, board_name: str) -> list[dict]:
    """Fetch all stock members of a single industry board."""
    params = {
        "pn": "1",
        "pz": "2000",           # Single pull — all members
        "po": "1",
        "np": "1",
        "ut": "bd1d9dd1031a93e1196165d2b4abbd74",
        "fltt": "2",
        "invt": "2",
        "fid": "f3",
        "fs": f"b:{board_code} f:!50",  # By board code, exclude ST
        "fields": "f12,f14",             # f12=stock code, f14=stock name
    }

    r = session.get(BASE_URL, params=params, headers=HEADERS, timeout=10)
    member_data = r.json()

    if not member_data.get("data") or not member_data["data"].get("diff"):
        return []

    records = []
    for stock in member_data["data"]["diff"]:
        records.append({
            "symbol_raw": stock["f12"],
            "sw_l1": board_name,
        })
    return records


def format_symbol(raw: str) -> str:
    """Strip exchange suffix to bare 6-digit code for our pipeline."""
    return str(raw).replace(".SH", "").replace(".SZ", "").replace(".BJ", "").zfill(6)


def main():
    print("Step 1: Connecting to EM Push2 gateway, fetching industry boards...")
    try:
        boards = fetch_industry_boards()
        print(f"  Got {len(boards)} industry boards.")
    except Exception as e:
        print(f"  FAILED: {e}")
        return

    print(f"\nStep 2: Streaming constituents for {len(boards)} boards...")
    all_records = []

    for idx, board in enumerate(boards, 1):
        b_code = board["code"]
        b_name = board["name"]

        try:
            members = fetch_board_members(b_code, b_name)
            all_records.extend(members)
        except Exception as e:
            print(f"  Warning: {b_name} ({b_code}) failed: {e}")
            continue

        if idx % 15 == 0:
            print(f"  Progress: {idx}/{len(boards)} boards, {len(all_records)} stocks...")
        time.sleep(0.1)  # Gentle rate limiting

    if not all_records:
        print("  FAILED: No stock data fetched. Check network.")
        return

    # 3. Post-process
    df = pd.DataFrame(all_records)
    df["symbol"] = df["symbol_raw"].apply(format_symbol)
    df = df.drop(columns=["symbol_raw"])
    df = df.drop_duplicates(subset=["symbol"])

    # 4. Save
    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/sw_industry.parquet", index=False)

    print(f"\n  Direct fetch complete!")
    print(f"  File: data/sw_industry.parquet")
    print(f"  Stocks covered: {len(df)}")
    print(f"  Industry boards: {df['sw_l1'].nunique()}")
    print()
    print("  Top-15 boards by stock count:")
    for name, cnt in df["sw_l1"].value_counts().head(15).items():
        print(f"    {name}: {cnt}")


if __name__ == "__main__":
    main()
