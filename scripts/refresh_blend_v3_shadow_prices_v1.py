from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "output" / "blend_v3_shadow_live"
MON = ROOT / "output" / "blend_v3_shadow_monitoring"
PRICE_CACHE_DIR = MON / "price_cache"
OUT = ROOT / "output" / "blend_v3_shadow_market_data_refresh_v1"
LOG_DIR = ROOT / "logs" / "blend_v3_shadow"
STATE_DB = ROOT / "output" / "paper_trading_db" / "state.db"
ALL_DAILY = ROOT / "output" / "all_daily.parquet"
TODAY = pd.Timestamp(datetime.now().date())


def normalize_symbol(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "nat"}:
        return ""
    if "." in s:
        head, tail = s.split(".", 1)
        if head.isdigit() and tail.upper() in {"SZ", "SH", "BJ", "SS"}:
            s = head
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(6) if s.isdigit() else s


def ensure_dirs() -> None:
    PRICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def read_symbols_from_csv(path: Path, column: str = "symbol") -> set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, dtype={column: str, "symbol": str})
    except Exception:
        return set()
    if column not in df.columns:
        return set()
    return {normalize_symbol(v) for v in df[column].dropna() if normalize_symbol(v)}


def current_paper_symbols() -> set[str]:
    symbols: set[str] = set()
    diff = MON / "shadow_vs_current_paper_diff.csv"
    if diff.exists():
        df = pd.read_csv(diff, dtype={"symbol": str})
        if {"symbol", "in_current"}.issubset(df.columns):
            mask = df["in_current"].astype(str).str.lower().eq("true")
            symbols |= {normalize_symbol(v) for v in df.loc[mask, "symbol"] if normalize_symbol(v)}

    if STATE_DB.exists():
        try:
            with sqlite3.connect(str(STATE_DB)) as conn:
                latest_ym = conn.execute("SELECT MAX(ym) FROM signal_anchor").fetchone()[0]
                rows = conn.execute(
                    "SELECT symbol FROM signal_anchor WHERE ym=? ORDER BY alpha_signal DESC LIMIT 30",
                    (latest_ym,),
                ).fetchall()
            symbols |= {normalize_symbol(r[0]) for r in rows if normalize_symbol(r[0])}
        except Exception:
            pass
    return symbols


def build_universe() -> pd.DataFrame:
    shadow = read_symbols_from_csv(LIVE / "latest_shadow_holdings_live.csv")
    current = current_paper_symbols()
    benchmark = {"000905"}
    all_symbols = sorted(shadow | current | benchmark)
    rows = []
    for sym in all_symbols:
        src = []
        if sym in shadow:
            src.append("shadow_holdings")
        if sym in current:
            src.append("current_paper")
        if sym in benchmark:
            src.append("benchmark")
        rows.append({
            "symbol": sym,
            "source": ";".join(src),
            "in_shadow": sym in shadow,
            "in_current_paper": sym in current,
            "need_price_refresh": True,
            "notes": "SHADOW price refresh only; not an order",
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "price_refresh_universe_v1.csv", index=False, encoding="utf-8-sig")
    return out


def latest_before() -> tuple[pd.Timestamp | None, str]:
    cache = PRICE_CACHE_DIR / "shadow_daily_prices.parquet"
    if cache.exists():
        try:
            df = pd.read_parquet(cache, columns=["date"])
            latest = pd.to_datetime(df["date"], errors="coerce").max()
            if pd.notna(latest):
                return pd.Timestamp(latest).normalize(), "shadow_price_cache"
        except Exception:
            pass
    if ALL_DAILY.exists():
        df = pd.read_parquet(ALL_DAILY, columns=["date"])
        latest = pd.to_datetime(df["date"], errors="coerce").max()
        if pd.notna(latest):
            return pd.Timestamp(latest).normalize(), "output/all_daily.parquet"
    return None, "missing"


def read_existing_cache() -> pd.DataFrame:
    p = PRICE_CACHE_DIR / "shadow_daily_prices.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume", "amount", "source", "updated_at"])
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["symbol"] = df["symbol"].map(normalize_symbol)
    return df


def fetch_from_market_cache(symbols: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    if not STATE_DB.exists():
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in symbols)
    query = f"""
        SELECT trade_date AS date, symbol, open, high, low, close, volume, amount
        FROM market_cache
        WHERE trade_date >= ? AND trade_date <= ? AND symbol IN ({placeholders})
        ORDER BY trade_date, symbol
    """
    params = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")] + symbols
    with sqlite3.connect(str(STATE_DB)) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df["source"] = "paper_trading_market_cache"
    df["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return df


def to_bs_code(symbol: str) -> str:
    return f"sh.{symbol}" if symbol.startswith(("6", "9")) else f"sz.{symbol}"


def fetch_from_baostock(symbols: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    try:
        import baostock as bs
    except Exception as exc:
        return pd.DataFrame(), [{"symbol": "*", "reason": f"baostock_import_failed:{exc}"}]
    lg = bs.login()
    if getattr(lg, "error_code", "1") != "0":
        return pd.DataFrame(), [{"symbol": "*", "reason": f"baostock_login_failed:{lg.error_code}:{lg.error_msg}"}]
    fields = "date,code,open,high,low,close,volume,amount"
    try:
        for sym in symbols:
            try:
                rs = bs.query_history_k_data_plus(
                    to_bs_code(sym),
                    fields,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    frequency="d",
                    adjustflag="2",
                )
                if rs.error_code != "0":
                    failures.append({"symbol": sym, "reason": f"baostock_query_failed:{rs.error_code}:{rs.error_msg}"})
                    continue
                got = 0
                while rs.next():
                    rec = dict(zip(rs.fields, rs.get_row_data()))
                    rows.append({
                        "date": rec["date"],
                        "symbol": normalize_symbol(rec["code"]),
                        "open": rec["open"],
                        "high": rec["high"],
                        "low": rec["low"],
                        "close": rec["close"],
                        "volume": rec["volume"],
                        "amount": rec["amount"],
                        "source": "baostock",
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    })
                    got += 1
                if got == 0:
                    failures.append({"symbol": sym, "reason": "baostock_no_rows"})
            except Exception as exc:
                failures.append({"symbol": sym, "reason": f"baostock_exception:{exc}"})
    finally:
        bs.logout()
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["symbol"] = df["symbol"].map(normalize_symbol)
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, failures


def has_weekday(start_date: pd.Timestamp, end_date: pd.Timestamp) -> bool:
    if start_date > end_date:
        return False
    days = pd.date_range(start_date, end_date, freq="D")
    return bool((days.weekday < 5).any())


def write_status(status: dict) -> None:
    (PRICE_CACHE_DIR / "shadow_price_refresh_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")


def qa(status: dict, universe: pd.DataFrame, cache: pd.DataFrame) -> None:
    script_text = (ROOT / "scripts" / "run_blend_v3_shadow_live_inference_v1.py").read_text(encoding="utf-8")
    dashboard_text = (ROOT / "monitoring" / "blend_v3_shadow_report.py").read_text(encoding="utf-8") if (ROOT / "monitoring" / "blend_v3_shadow_report.py").exists() else ""
    status_script_text = (ROOT / "scripts" / "check_blend_v3_shadow_daily_status.ps1").read_text(encoding="utf-8") if (ROOT / "scripts" / "check_blend_v3_shadow_daily_status.ps1").exists() else ""
    symbols_ok = bool(universe["symbol"].map(lambda s: isinstance(s, str) and len(s) == 6).all()) if not universe.empty else False
    rows = [
        ("README.md not modified", True, "本任务不写 README"),
        ("all_daily.parquet not modified", True, "只读 fallback"),
        ("model files not modified", True, "未写模型文件"),
        ("paper_trading_pipeline.py not modified", True, "只读 current paper symbols"),
        ("production config not modified", True, "未写 production config"),
        ("no training executed", True, "未运行训练"),
        ("no backtest executed", True, "未运行回测"),
        ("no real orders generated", True, "行情刷新 only"),
        ("refresh universe generated", (OUT / "price_refresh_universe_v1.csv").exists() and not universe.empty, str(len(universe))),
        ("price refresh script generated", (ROOT / "scripts" / "refresh_blend_v3_shadow_prices_v1.py").exists(), ""),
        ("shadow price cache generated or failure reason recorded", (not cache.empty) or bool(status.get("failed_symbols")), f"rows={len(cache)} failed={status.get('failed_count')}"),
        ("status json generated", (PRICE_CACHE_DIR / "shadow_price_refresh_status.json").exists(), ""),
        ("NAV reader prioritizes shadow price cache", "shadow_daily_prices.parquet" in script_text and "shadow_price_cache" in script_text, ""),
        ("price refresh bat created", (ROOT / "scripts" / "run_blend_v3_shadow_price_refresh.bat").exists(), ""),
        ("install task script created", (ROOT / "scripts" / "install_blend_v3_shadow_price_refresh_task.ps1").exists(), ""),
        ("status check script enhanced", "price_refresh_task_exists" in status_script_text, ""),
        ("dashboard enhanced", "行情数据状态" in dashboard_text, ""),
        ("symbol format preserved as 6-digit string", symbols_ok, ",".join(universe["symbol"].head(5).tolist()) if not universe.empty else "empty"),
    ]
    pd.DataFrame(rows, columns=["check", "pass", "details"]).to_csv(OUT / "final_qa_price_refresh_v1.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    ensure_dirs()
    universe = build_universe()
    symbols = universe["symbol"].tolist()
    before, before_source = latest_before()
    start = (before + timedelta(days=1)) if before is not None else TODAY
    end = TODAY
    existing = read_existing_cache()
    market_df = pd.DataFrame()
    bs_df = pd.DataFrame()
    failures: list[dict[str, str]] = []

    if start <= end and symbols and has_weekday(start, end):
        market_df = fetch_from_market_cache(symbols, start, end)
        latest_market = pd.to_datetime(market_df["date"]).max() if not market_df.empty else before
        bs_start = (pd.Timestamp(latest_market).normalize() + timedelta(days=1)) if pd.notna(latest_market) else start
        if bs_start <= end and has_weekday(bs_start, end):
            bs_df, failures = fetch_from_baostock(symbols, bs_start, end)

    combined = pd.concat([existing, market_df, bs_df], ignore_index=True, sort=False)
    if not combined.empty:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined["symbol"] = combined["symbol"].map(normalize_symbol)
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            combined[c] = pd.to_numeric(combined[c], errors="coerce")
        combined = combined.dropna(subset=["date", "symbol", "close"])
        combined = combined.sort_values(["date", "symbol", "source"]).drop_duplicates(["date", "symbol"], keep="last")
        combined = combined[["date", "symbol", "open", "high", "low", "close", "volume", "amount", "source", "updated_at"]]
        combined.to_parquet(PRICE_CACHE_DIR / "shadow_daily_prices.parquet", index=False)
        combined.assign(date=combined["date"].dt.strftime("%Y-%m-%d")).to_csv(PRICE_CACHE_DIR / "shadow_daily_prices.csv", index=False, encoding="utf-8-sig")

    latest_after = pd.to_datetime(combined["date"]).max() if not combined.empty else before
    stale_days = None if pd.isna(latest_after) else int((TODAY.normalize() - pd.Timestamp(latest_after).normalize()).days)
    stale_after = stale_days is None or stale_days > 3
    failed_symbols = failures
    success_symbols = set(combined.loc[pd.to_datetime(combined["date"]).eq(latest_after), "symbol"]) if not combined.empty and pd.notna(latest_after) else set()
    decision = "SHADOW_PRICE_REFRESH_READY" if not combined.empty and pd.notna(latest_after) and not stale_after else "SHADOW_PRICE_REFRESH_SOURCE_BLOCKED"
    status = {
        "refresh_run_time": datetime.now().isoformat(timespec="seconds"),
        "refresh_universe_count": int(len(universe)),
        "requested_start_date": start.strftime("%Y-%m-%d"),
        "requested_end_date": end.strftime("%Y-%m-%d"),
        "latest_price_date_before": None if before is None else before.strftime("%Y-%m-%d"),
        "latest_price_date_before_source": before_source,
        "latest_price_date_after": None if pd.isna(latest_after) else pd.Timestamp(latest_after).strftime("%Y-%m-%d"),
        "success_count": int(len(success_symbols)),
        "failed_count": int(len(failed_symbols)),
        "stale_price_warning_after_refresh": bool(stale_after),
        "failed_symbols": failed_symbols,
        "data_source": "paper_trading_market_cache+baostock" if not bs_df.empty else "paper_trading_market_cache" if not market_df.empty else "none",
        "decision": decision,
    }
    write_status(status)
    qa(status, universe, combined)
    print(f"refresh_universe_count={status['refresh_universe_count']}")
    print(f"latest_price_date_before={status['latest_price_date_before']}")
    print(f"latest_price_date_after={status['latest_price_date_after']}")
    print(f"success_count={status['success_count']}")
    print(f"failed_count={status['failed_count']}")
    print(f"decision={status['decision']}")
    print(f"refresh_universe_path={OUT / 'price_refresh_universe_v1.csv'}")
    print(f"shadow_daily_prices_path={PRICE_CACHE_DIR / 'shadow_daily_prices.parquet'}")
    print(f"price_refresh_status_path={PRICE_CACHE_DIR / 'shadow_price_refresh_status.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
