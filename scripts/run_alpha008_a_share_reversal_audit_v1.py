from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "alpha008_a_share_reversal_audit_v1"
ALL_DAILY = ROOT / "output" / "all_daily.parquet"
TRAINING = ROOT / "output" / "training_panel_v15_sr.parquet"
REFINED_SCORECARD = ROOT / "output" / "pv_reversal_fragility_refinement_v1" / "pv_reversal_refinement_scorecard_v1.csv"
REFINED_PANEL = ROOT / "output" / "pv_reversal_fragility_refinement_v1" / "pv_reversal_refined_factor_panel_v1.parquet"
STATUS = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS = ROOT / "docs" / "DECISIONS.md"
README = ROOT / "README.md"
PAPER = ROOT / "paper_trading" / "paper_trading_pipeline.py"
README_REPORT = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"

ALPHA_FACTORS = [
    "alpha008_original_direction",
    "alpha008_flipped_direction",
    "alpha008_ranked",
    "alpha008_flipped_ranked",
    "alpha008_original_zscore",
    "alpha008_flipped_zscore",
    "alpha008_liquid_only",
    "alpha008_size_liquidity_neutral",
    "alpha008_vol_neutral",
]
TARGETS = [
    "fwd_ret_5d",
    "fwd_ret_10d",
    "fwd_ret_20d",
    "fwd_ret_1m",
    "fwd_ret_5d_excess_equal_weight",
    "fwd_ret_10d_excess_equal_weight",
    "fwd_ret_20d_excess_equal_weight",
    "fwd_ret_1m_excess_equal_weight",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def file_hash(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: Path) -> tuple[pd.DataFrame | None, str]:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path), ""
        return pd.read_csv(path), ""
    except Exception as exc:
        return None, repr(exc)


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def winsor_by_date(df: pd.DataFrame, cols: list[str], date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = out.groupby(date_col)[col].transform(lambda s: s.clip(s.quantile(0.01), s.quantile(0.99)))
    return out


def neutralize(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    valid = y.notna()
    for col in x.columns:
        valid &= x[col].notna()
    out = pd.Series(np.nan, index=y.index)
    if valid.sum() < 50:
        return out
    xv = x.loc[valid].astype(float)
    keep = [c for c in xv.columns if xv[c].std(ddof=0) > 0]
    if not keep:
        out.loc[valid] = y.loc[valid] - y.loc[valid].mean()
        return out
    mat = np.column_stack([np.ones(len(xv)), xv[keep].to_numpy()])
    try:
        beta = np.linalg.lstsq(mat, y.loc[valid].astype(float).to_numpy(), rcond=None)[0]
        out.loc[valid] = y.loc[valid].astype(float).to_numpy() - mat @ beta
    except np.linalg.LinAlgError:
        out.loc[valid] = y.loc[valid] - y.loc[valid].mean()
    return out


def rank_ic(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 50 or x.loc[valid].nunique() < 2 or y.loc[valid].nunique() < 2:
        return np.nan
    return x.loc[valid].rank().corr(y.loc[valid].rank())


def tstat(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    std = s.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return np.nan
    return float(s.mean() / (std / math.sqrt(len(s))))


def summarize_ic(panel: pd.DataFrame, factor: str, target: str, date_col: str, frequency: str) -> dict[str, object]:
    rows = []
    for dt, g in panel.groupby(date_col):
        ic = rank_ic(g[factor], g[target])
        if np.isfinite(ic):
            rows.append((dt, ic, int((g[factor].notna() & g[target].notna()).sum())))
    vals = pd.Series([x[1] for x in rows], dtype="float64")
    return {
        "factor_name": factor,
        "target": target,
        "frequency": frequency,
        "n_periods": len(rows),
        "n_obs": int(sum(x[2] for x in rows)),
        "mean_rank_ic": float(vals.mean()) if len(vals) else np.nan,
        "median_rank_ic": float(vals.median()) if len(vals) else np.nan,
        "ic_std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
        "ic_ir": float(vals.mean() / vals.std(ddof=1)) if len(vals) > 1 and vals.std(ddof=1) != 0 else np.nan,
        "positive_ic_rate": float((vals > 0).mean()) if len(vals) else np.nan,
        "min_date": "" if not rows else str(min(x[0] for x in rows).date()),
        "max_date": "" if not rows else str(max(x[0] for x in rows).date()),
        "notes": "Rank IC only; no model and no strategy backtest.",
    }


def group_diag(panel: pd.DataFrame, factor: str, target: str, date_col: str, frequency: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    p = panel.copy()
    rank_pct = p.groupby(date_col)[factor].rank(pct=True, method="first")
    valid_counts = p.groupby(date_col)[factor].transform("count")
    p["group_id"] = np.ceil(rank_pct * 5).clip(1, 5)
    p.loc[valid_counts < 50, "group_id"] = np.nan
    valid = p.dropna(subset=["group_id", target])
    dec = []
    for gid, gg in valid.groupby("group_id"):
        by_date = gg.groupby(date_col)[target].mean()
        dec.append({
            "factor_name": factor,
            "target": target,
            "frequency": frequency,
            "group_id": int(gid),
            "n_periods": int(by_date.count()),
            "avg_forward_return": float(by_date.mean()) if len(by_date) else np.nan,
            "hit_rate": float((by_date > 0).mean()) if len(by_date) else np.nan,
            "avg_n_stocks": float(gg.groupby(date_col).size().mean()) if len(gg) else np.nan,
            "notes": "5-group diagnostic only; top group is highest factor value.",
        })
    top = valid[valid["group_id"] == 5].groupby(date_col)[target].mean()
    bottom = valid[valid["group_id"] == 1].groupby(date_col)[target].mean()
    spread = (top - bottom).dropna()
    by_year = spread.groupby(spread.index.year).mean() if len(spread) else pd.Series(dtype="float64")
    means = valid.groupby("group_id")[target].mean().sort_index()
    mono = float(means.rank().corr(pd.Series(means.index, index=means.index))) if len(means) >= 2 else np.nan
    spread_row = {
        "factor_name": factor,
        "target": target,
        "frequency": frequency,
        "top_minus_bottom_mean": float(spread.mean()) if len(spread) else np.nan,
        "top_minus_bottom_tstat": tstat(spread),
        "monotonicity_score": mono,
        "avg_n_stocks": float(valid.groupby(date_col).size().mean()) if len(valid) else np.nan,
        "notes": "Group return diagnostic only; not a portfolio backtest.",
        "worst_year": "" if by_year.empty else str(int(by_year.idxmin())),
        "best_year": "" if by_year.empty else str(int(by_year.idxmax())),
    }
    return dec, spread_row


def update_status(status_value: str) -> None:
    status = yaml.safe_load(STATUS.read_text(encoding="utf-8"))
    csmar = status.get("alternative_data", {}).get("csmar_status")
    status.setdefault("research", {})
    status["research"]["alpha008_status"] = status_value
    status["research"]["alpha008_latest_task"] = "Alpha008 A-Share Reversal Factor Audit v1"
    status["research"]["alpha008_latest_output"] = rel(OUT)
    status.setdefault("validation", {})
    status["validation"]["blend_v3_historical_metrics_status"] = "under_pit_review"
    if csmar is not None:
        status.setdefault("alternative_data", {})["csmar_status"] = csmar
    status.setdefault("project", {})["last_updated"] = date.today().isoformat()
    STATUS.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")


def append_decision(decision: str, beats: bool, neutral_survival: str, can_enter: bool) -> None:
    block = "\n".join([
        f"## {date.today().isoformat()}",
        "",
        "决策：",
        "",
        "- 完成 Alpha008 A-Share Reversal Factor Audit v1。",
        f"- 是否优于 reversal_20d_liquid_only：{beats}。",
        f"- 是否通过流动性 / 市值 / 中性化审计：{neutral_survival}。",
        f"- 是否可以进入 residual alpha test：{can_enter}。",
        "- 不接入 production。",
        "- 不修改 README。",
        f"- Decision = {decision}。",
    ])
    text = DECISIONS.read_text(encoding="utf-8") if DECISIONS.exists() else "# 决策日志\n"
    if "Alpha008 A-Share Reversal Factor Audit v1" in text and decision in text:
        return
    DECISIONS.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def run_status_scripts() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_current_status_md.py")], cwd=ROOT, check=True, capture_output=True, text=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_readme_consistency.py")], cwd=ROOT, check=True, capture_output=True, text=True)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    original_status = yaml.safe_load(STATUS.read_text(encoding="utf-8"))
    original_csmar = original_status.get("alternative_data", {}).get("csmar_status")
    protected = {
        "README": file_hash(README),
        "all_daily": file_hash(ALL_DAILY),
        "training": file_hash(TRAINING),
        "paper": file_hash(PAPER),
    }
    model_before = {rel(p): file_hash(p) for p in (ROOT / "output").glob("production_models*/**/*") if p.is_file()}
    config_before = {rel(p): file_hash(p) for p in (ROOT / "config").glob("*") if p.is_file() and p.name != "project_status.yaml"}

    inputs = [
        (ALL_DAILY, "daily market data"),
        (TRAINING, "v15 universe/month index"),
        (REFINED_SCORECARD, "previous refinement scorecard"),
        (REFINED_PANEL, "previous refined reversal factor panel"),
    ]
    loaded: dict[str, pd.DataFrame | None] = {}
    audit_rows = []
    for path, role in inputs:
        df, err = read_table(path)
        loaded[str(path)] = df
        cols = set(df.columns) if df is not None else set()
        date_col = "date" if "date" in cols else "month_end" if "month_end" in cols else None
        audit_rows.append({
            "input_path": rel(path),
            "exists": path.exists(),
            "readable": df is not None,
            "n_rows": 0 if df is None else len(df),
            "n_symbols": 0 if df is None or "symbol" not in cols else int(df["symbol"].astype(str).nunique()),
            "min_date": "" if df is None or date_col is None else str(pd.to_datetime(df[date_col]).min().date()),
            "max_date": "" if df is None or date_col is None else str(pd.to_datetime(df[date_col]).max().date()),
            "available_fields": "" if df is None else ",".join(map(str, df.columns)),
            "role": role,
            "notes": err or "",
        })
    input_audit_path = OUT / "input_data_audit_v1.csv"
    pd.DataFrame(audit_rows).to_csv(input_audit_path, index=False, encoding="utf-8-sig")

    daily = loaded[str(ALL_DAILY)]
    training = loaded[str(TRAINING)]
    if daily is None or not {"symbol", "date", "open", "close"}.issubset(daily.columns):
        decision = "ALPHA008_AUDIT_INVALID_DATA_QUALITY"
        update_status("audit_inconclusive")
        final_qa = OUT / "final_qa_alpha008_a_share_reversal_audit_v1.csv"
        pd.DataFrame([{"check": "minimum all_daily fields", "pass": False, "details": "symbol/date/open/close required; close substitution is not allowed"}]).to_csv(final_qa, index=False, encoding="utf-8-sig")
        print(f"input_data_audit_path={rel(input_audit_path)}")
        print(f"decision={decision}")
        return

    daily = daily.copy()
    daily["symbol"] = daily["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["symbol", "date"])
    if training is not None and "symbol" in training.columns:
        training = training.copy()
        training["symbol"] = training["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
        training["date"] = pd.to_datetime(training["date"])
        universe_symbols = set(training["symbol"].dropna().unique())
        monthly_dates = training.groupby(training["date"].dt.to_period("M"))["date"].max().sort_values()
        month_source = "training_panel_v15_sr.parquet"
    else:
        universe_symbols = set(daily["symbol"].unique())
        monthly_dates = daily.groupby(daily["date"].dt.to_period("M"))["date"].max().sort_values()
        month_source = "all_daily fallback"
    daily = daily[daily["symbol"].isin(universe_symbols)].copy()
    daily = daily[(daily["date"] >= "2016-01-01") & (daily["date"] <= "2026-06-30")]
    daily_dates = daily.groupby("date")["symbol"].nunique().reset_index(name="n_symbols")
    daily_dates = daily_dates[daily_dates["n_symbols"] >= 50]
    monthly_dates = monthly_dates[(monthly_dates >= "2017-01-01") & (monthly_dates <= "2026-06-30")]

    g = daily.groupby("symbol", group_keys=False)
    daily["daily_return"] = g["close"].pct_change()
    daily["sum_open_5"] = g["open"].transform(lambda s: s.rolling(5, min_periods=5).sum())
    daily["sum_return_5"] = g["daily_return"].transform(lambda s: s.rolling(5, min_periods=5).sum())
    daily["signal_base"] = daily["sum_open_5"] * daily["sum_return_5"]
    daily["alpha008_raw"] = daily["signal_base"] - g["signal_base"].shift(10)
    daily["amount_mean_20d"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=10).mean()) if "amount" in daily.columns else np.nan
    daily["amount_mean_60d"] = g["amount"].transform(lambda s: s.rolling(60, min_periods=20).mean()) if "amount" in daily.columns else np.nan
    daily["amount_mean_120d"] = g["amount"].transform(lambda s: s.rolling(120, min_periods=60).mean()) if "amount" in daily.columns else np.nan
    daily["ret_20d"] = g["close"].pct_change(20)
    daily["vol_20d"] = g["daily_return"].transform(lambda s: s.rolling(20, min_periods=10).std())
    daily["liquidity_proxy"] = daily["amount_mean_60d"].fillna(daily["amount_mean_20d"]).fillna(daily["amount_mean_120d"])
    daily["liquidity_rank_pct"] = daily.groupby("date")["liquidity_proxy"].rank(pct=True)
    daily["vol_rank_pct"] = daily.groupby("date")["vol_20d"].rank(pct=True)
    daily["price_level_bucket"] = daily.groupby("date")["close"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low_price", "mid_price", "high_price"]) if s.notna().sum() >= 50 else pd.Series(np.nan, index=s.index)
    )
    daily["liquidity_bucket"] = daily.groupby("date")["liquidity_proxy"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low_liquidity", "mid_liquidity", "high_liquidity"]) if s.notna().sum() >= 50 else pd.Series(np.nan, index=s.index)
    )
    daily["volatility_bucket"] = daily.groupby("date")["vol_20d"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low_vol", "mid_vol", "high_vol"]) if s.notna().sum() >= 50 else pd.Series(np.nan, index=s.index)
    )
    daily["size_bucket"] = "micro_or_unknown"

    rank_raw = daily.groupby("date")["alpha008_raw"].rank(pct=True)
    daily["alpha008_flipped_direction"] = rank_raw
    daily["alpha008_original_direction"] = -rank_raw
    daily["alpha008_ranked"] = daily.groupby("date")["alpha008_original_direction"].rank(pct=True)
    daily["alpha008_flipped_ranked"] = daily.groupby("date")["alpha008_flipped_direction"].rank(pct=True)
    daily["alpha008_original_zscore"] = daily.groupby("date")["alpha008_original_direction"].transform(zscore)
    daily["alpha008_flipped_zscore"] = daily.groupby("date")["alpha008_flipped_direction"].transform(zscore)
    daily["alpha008_liquid_only"] = daily["alpha008_flipped_direction"].where(daily["liquidity_rank_pct"] > 0.4)
    daily["alpha008_size_liquidity_neutral"] = daily.groupby("date", group_keys=False).apply(
        lambda m: neutralize(m["alpha008_flipped_direction"], m[["liquidity_proxy", "vol_20d", "ret_20d"]]), include_groups=False
    ).sort_index()
    daily["alpha008_vol_neutral"] = daily.groupby("date", group_keys=False).apply(
        lambda m: neutralize(m["alpha008_flipped_direction"], m[["vol_20d"]]), include_groups=False
    ).sort_index()
    daily = winsor_by_date(daily, ALPHA_FACTORS, "date")

    universe = daily.groupby("symbol").agg(n_trading_days=("date", "size"), min_date=("date", "min"), max_date=("date", "max"), avg_amount_60d=("amount_mean_60d", "last")).reset_index()
    universe["in_v15_panel"] = universe["symbol"].isin(universe_symbols)
    universe["notes"] = "market cap unavailable; Alpha008 universe follows v15 symbols when available"
    research_universe_path = OUT / "alpha008_research_universe_v1.csv"
    universe.to_csv(research_universe_path, index=False, encoding="utf-8-sig")

    date_index = pd.concat([
        daily_dates.assign(frequency="daily", source="all_daily"),
        pd.DataFrame({"date": monthly_dates.values, "n_symbols": monthly_dates.map(daily.groupby("date")["symbol"].nunique()).fillna(0).astype(int).values, "frequency": "monthly", "source": month_source}),
    ], ignore_index=True)
    date_index["notes"] = np.where(date_index["frequency"] == "monthly", "used for 1M forward return", "used for 5D/10D/20D forward return")
    date_index_path = OUT / "alpha008_date_index_v1.csv"
    date_index.to_csv(date_index_path, index=False, encoding="utf-8-sig")

    factor_cols = ["symbol", "date", "open", "close", "daily_return", "alpha008_raw", "liquidity_proxy", "liquidity_rank_pct", "vol_20d", "vol_rank_pct", "ret_20d", "liquidity_bucket", "volatility_bucket", "price_level_bucket", "size_bucket"] + ALPHA_FACTORS
    daily_panel = daily[daily["date"].isin(set(daily_dates["date"]))][factor_cols].copy()
    monthly_panel = daily[daily["date"].isin(set(monthly_dates))][factor_cols].copy().rename(columns={"date": "month_end"})
    daily_factor_panel_path = OUT / "alpha008_daily_factor_panel_v1.parquet"
    monthly_factor_panel_path = OUT / "alpha008_monthly_factor_panel_v1.parquet"
    sample_path = OUT / "alpha008_factor_panel_sample_v1.csv"
    daily_panel.to_parquet(daily_factor_panel_path, index=False)
    monthly_panel.to_parquet(monthly_factor_panel_path, index=False)
    daily_panel.head(1000).to_csv(sample_path, index=False, encoding="utf-8-sig")

    label_base = daily[["symbol", "date", "close"]].copy()
    for h in [5, 10, 20]:
        label_base[f"fwd_ret_{h}d"] = g["close"].shift(-h) / daily["close"] - 1
        label_base[f"fwd_ret_{h}d_excess_equal_weight"] = label_base[f"fwd_ret_{h}d"] - label_base.groupby("date")[f"fwd_ret_{h}d"].transform("mean")
    monthly_close = daily[daily["date"].isin(set(monthly_dates))][["symbol", "date", "close"]].sort_values(["symbol", "date"]).copy()
    monthly_close["next_month_close"] = monthly_close.groupby("symbol")["close"].shift(-1)
    monthly_close["fwd_ret_1m"] = monthly_close["next_month_close"] / monthly_close["close"] - 1
    monthly_labels = monthly_close[["symbol", "date", "fwd_ret_1m"]]
    label_base = label_base.merge(monthly_labels, on=["symbol", "date"], how="left")
    label_base["fwd_ret_1m_excess_equal_weight"] = label_base["fwd_ret_1m"] - label_base.groupby("date")["fwd_ret_1m"].transform("mean")
    label_panel_path = OUT / "alpha008_label_panel_v1.parquet"
    label_base[["symbol", "date"] + TARGETS].to_parquet(label_panel_path, index=False)
    label_qa = pd.DataFrame([
        {"check": "daily forward labels generated", "pass": bool(label_base["fwd_ret_5d"].notna().sum() > 0), "details": f"5d_non_null={int(label_base['fwd_ret_5d'].notna().sum())}"},
        {"check": "monthly forward label generated", "pass": bool(label_base["fwd_ret_1m"].notna().sum() > 0), "details": f"1m_non_null={int(label_base['fwd_ret_1m'].notna().sum())}"},
        {"check": "open field used", "pass": True, "details": "Alpha008 uses rolling_sum(open, 5); close was not substituted for open."},
    ])
    label_qa_path = OUT / "alpha008_label_qa_v1.csv"
    label_qa.to_csv(label_qa_path, index=False, encoding="utf-8-sig")

    eval_daily = daily_panel.merge(label_base[["symbol", "date"] + TARGETS], on=["symbol", "date"], how="left")
    eval_monthly = monthly_panel.rename(columns={"month_end": "date"}).merge(label_base[["symbol", "date"] + TARGETS], on=["symbol", "date"], how="left")
    single_factor_ic_path = OUT / "alpha008_single_factor_ic_v1.csv"
    if single_factor_ic_path.exists():
        ic_df = pd.read_csv(single_factor_ic_path)
    else:
        ic_rows = []
        for factor in ALPHA_FACTORS:
            for target in TARGETS:
                if target.endswith("1m") or "1m_" in target:
                    ic_rows.append(summarize_ic(eval_monthly, factor, target, "date", "monthly"))
                else:
                    ic_rows.append(summarize_ic(eval_daily, factor, target, "date", "daily"))
        ic_df = pd.DataFrame(ic_rows)
        ic_df.to_csv(single_factor_ic_path, index=False, encoding="utf-8-sig")

    dec_rows, spread_rows = [], []
    diagnostic_targets = ["fwd_ret_20d_excess_equal_weight", "fwd_ret_1m_excess_equal_weight"]
    for factor in ALPHA_FACTORS:
        for target in diagnostic_targets:
            p = eval_monthly if "1m" in target else eval_daily
            freq = "monthly" if "1m" in target else "daily"
            d, s = group_diag(p, factor, target, "date", freq)
            dec_rows.extend(d)
            spread_rows.append(s)
    decile_return_path = OUT / "alpha008_decile_return_v1.csv"
    group_spread_path = OUT / "alpha008_group_spread_v1.csv"
    pd.DataFrame(dec_rows).to_csv(decile_return_path, index=False, encoding="utf-8-sig")
    spread_df = pd.DataFrame(spread_rows)
    spread_df.to_csv(group_spread_path, index=False, encoding="utf-8-sig")

    refined_panel, _ = read_table(REFINED_PANEL)
    baseline_rows = []
    baseline_ic = np.nan
    baseline_ir = np.nan
    baseline_spread = np.nan
    baseline_corr_by_factor: dict[str, float] = {}
    if refined_panel is not None and {"symbol", "month_end", "reversal_20d_liquid_only"}.issubset(refined_panel.columns):
        base = refined_panel.copy()
        base["symbol"] = base["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
        base["date"] = pd.to_datetime(base["month_end"])
        base = base.merge(label_base[["symbol", "date", "fwd_ret_1m_excess_equal_weight"]], on=["symbol", "date"], how="left")
        b_ic = summarize_ic(base, "reversal_20d_liquid_only", "fwd_ret_1m_excess_equal_weight", "date", "monthly")
        b_sp = group_diag(base, "reversal_20d_liquid_only", "fwd_ret_1m_excess_equal_weight", "date", "monthly")[1]
        baseline_ic, baseline_ir, baseline_spread = b_ic["mean_rank_ic"], b_ic["ic_ir"], b_sp["top_minus_bottom_mean"]
        merged_base = eval_monthly.merge(base[["symbol", "date", "reversal_20d_liquid_only"]], on=["symbol", "date"], how="left")
        for factor in ALPHA_FACTORS:
            corrs = []
            for _, m in merged_base.groupby("date"):
                c = rank_ic(m[factor], m["reversal_20d_liquid_only"])
                if np.isfinite(c):
                    corrs.append(c)
            baseline_corr_by_factor[factor] = float(pd.Series(corrs).mean()) if corrs else np.nan
    for factor in ALPHA_FACTORS:
        a_ic = ic_df[(ic_df["factor_name"] == factor) & (ic_df["target"] == "fwd_ret_1m_excess_equal_weight")].iloc[0]
        a_sp = spread_df[(spread_df["factor_name"] == factor) & (spread_df["target"] == "fwd_ret_1m_excess_equal_weight")].iloc[0]
        corr = baseline_corr_by_factor.get(factor, np.nan)
        if np.isfinite(baseline_ic) and np.isfinite(a_ic["mean_rank_ic"]):
            if float(a_ic["mean_rank_ic"]) > float(baseline_ic) + 0.01 and abs(corr) < 0.7:
                interp = "ALPHA008_BEATS_BASELINE"
            elif abs(corr) >= 0.7:
                interp = "SIMILAR_TO_REVERSAL"
            elif float(a_ic["mean_rank_ic"]) < float(baseline_ic):
                interp = "WEAKER_THAN_BASELINE"
            else:
                interp = "HORIZON_SPECIFIC_ADVANTAGE"
        else:
            interp = "INCONCLUSIVE"
        baseline_rows.append({
            "alpha_factor": factor,
            "baseline_factor": "reversal_20d_liquid_only",
            "target": "fwd_ret_1m_excess_equal_weight",
            "mean_alpha_ic": a_ic["mean_rank_ic"],
            "mean_baseline_ic": baseline_ic,
            "alpha_ic_ir": a_ic["ic_ir"],
            "baseline_ic_ir": baseline_ir,
            "alpha_spread": a_sp["top_minus_bottom_mean"],
            "baseline_spread": baseline_spread,
            "mean_cross_sectional_corr": corr,
            "incremental_interpretation": interp,
            "notes": "read-only baseline comparison; no fusion and no model training",
        })
    baseline_comparison_path = OUT / "alpha008_vs_reversal_baseline_v1.csv"
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(baseline_comparison_path, index=False, encoding="utf-8-sig")

    best_row_for_neut = ic_df[ic_df["target"].str.contains("excess_equal_weight")].sort_values("mean_rank_ic", ascending=False).iloc[0]
    best_factor_raw = str(best_row_for_neut["factor_name"])
    eval_daily["alpha008_neutralized"] = eval_daily.groupby("date", group_keys=False).apply(
        lambda m: neutralize(m[best_factor_raw], m[["liquidity_proxy", "vol_20d", "ret_20d"]]), include_groups=False
    ).sort_index()
    eval_daily["alpha008_neutralized_liquid_only"] = eval_daily["alpha008_neutralized"].where(eval_daily["liquidity_rank_pct"] > 0.4)
    neutral_rows = []
    for nf in ["alpha008_neutralized", "alpha008_neutralized_liquid_only"]:
        target = "fwd_ret_20d_excess_equal_weight" if str(best_row_for_neut["frequency"]) == "daily" else "fwd_ret_1m_excess_equal_weight"
        raw_ic = summarize_ic(eval_daily, best_factor_raw, target, "date", "daily")
        neu_ic = summarize_ic(eval_daily, nf, target, "date", "daily")
        raw_sp = group_diag(eval_daily, best_factor_raw, target, "date", "daily")[1]
        neu_sp = group_diag(eval_daily, nf, target, "date", "daily")[1]
        survival = np.isfinite(neu_ic["mean_rank_ic"]) and float(neu_ic["mean_rank_ic"]) > 0.5 * float(raw_ic["mean_rank_ic"])
        neutral_rows.append({
            "factor_name": nf,
            "target": target,
            "raw_mean_ic": raw_ic["mean_rank_ic"],
            "neutralized_mean_ic": neu_ic["mean_rank_ic"],
            "raw_ic_ir": raw_ic["ic_ir"],
            "neutralized_ic_ir": neu_ic["ic_ir"],
            "raw_spread": raw_sp["top_minus_bottom_mean"],
            "neutralized_spread": neu_sp["top_minus_bottom_mean"],
            "interpretation": "SURVIVES_NEUTRALIZATION" if survival else "WEAK_AFTER_NEUTRALIZATION",
            "notes": "size proxy unavailable; neutralized on liquidity, vol_20d and ret_20d only",
        })
    neutralization_audit_path = OUT / "alpha008_neutralization_audit_v1.csv"
    neutral_df = pd.DataFrame(neutral_rows)
    neutral_df.to_csv(neutralization_audit_path, index=False, encoding="utf-8-sig")

    frag_rows = []
    frag_specs = [
        ("size_bucket", "size_bucket"),
        ("liquidity_bucket", "liquidity_bucket"),
        ("volatility_bucket", "volatility_bucket"),
        ("price_level_bucket", "price_level_bucket"),
        ("calendar_year", "calendar_year"),
        ("exclude_low_liquidity_bottom_20pct", "filter"),
        ("exclude_low_liquidity_bottom_40pct", "filter"),
        ("exclude_high_volatility_top_10pct", "filter"),
        ("exclude_microcap_unknown", "filter"),
    ]
    frag_base = eval_daily.copy()
    frag_base["calendar_year"] = frag_base["date"].dt.year.astype(str)
    fragility_factors = list(dict.fromkeys([
        best_factor_raw,
        "alpha008_flipped_direction",
        "alpha008_original_direction",
        "alpha008_liquid_only",
    ]))
    for factor in fragility_factors:
        for dim, col in frag_specs:
            if col == "filter":
                sub = frag_base.copy()
                if dim.endswith("20pct"):
                    sub = sub[sub["liquidity_rank_pct"] > 0.2]
                elif dim.endswith("40pct"):
                    sub = sub[sub["liquidity_rank_pct"] > 0.4]
                elif "high_volatility" in dim:
                    sub = sub[sub["vol_rank_pct"] <= 0.9]
                elif "microcap" in dim:
                    sub = sub.iloc[0:0]
                groups = [(dim, sub)]
            else:
                groups = list(frag_base.groupby(col, dropna=False))
            for val, sub in groups:
                if len(sub) < 500:
                    ic = {"n_periods": 0, "n_obs": 0, "mean_rank_ic": np.nan, "ic_ir": np.nan}
                    sp = {"top_minus_bottom_mean": np.nan}
                else:
                    ic = summarize_ic(sub, factor, "fwd_ret_20d_excess_equal_weight", "date", "daily")
                    sp = group_diag(sub, factor, "fwd_ret_20d_excess_equal_weight", "date", "daily")[1]
                flag = "INCONCLUSIVE"
                if np.isfinite(ic["mean_rank_ic"]):
                    if float(ic["mean_rank_ic"]) <= 0 or not np.isfinite(sp["top_minus_bottom_mean"]) or float(sp["top_minus_bottom_mean"]) <= 0:
                        flag = "WEAK_AFTER_FILTER"
                    elif val == "low_liquidity":
                        flag = "LOW_LIQUIDITY_DEPENDENT"
                    elif val == "micro_or_unknown":
                        flag = "SMALL_CAP_DEPENDENT"
                    elif val == "high_vol":
                        flag = "HIGH_VOL_DEPENDENT"
                    else:
                        flag = "PASSES_TRADABILITY_FILTER"
                frag_rows.append({
                    "factor_name": factor,
                    "target": "fwd_ret_20d_excess_equal_weight",
                    "fragility_dimension": dim,
                    "group_value": val,
                    "n_periods": ic["n_periods"],
                    "n_obs": ic["n_obs"],
                    "mean_rank_ic": ic["mean_rank_ic"],
                    "ic_ir": ic["ic_ir"],
                    "top_minus_bottom_mean": sp["top_minus_bottom_mean"],
                    "fragility_flag": flag,
                    "notes": "market cap unavailable; microcap exclusion inconclusive" if "microcap" in str(dim) or val == "micro_or_unknown" else "",
                })
    fragility_audit_path = OUT / "alpha008_fragility_audit_v1.csv"
    fragility_df = pd.DataFrame(frag_rows)
    fragility_df.to_csv(fragility_audit_path, index=False, encoding="utf-8-sig")

    score_rows = []
    for factor in ALPHA_FACTORS:
        cand_ics = ic_df[(ic_df["factor_name"] == factor) & (ic_df["target"].str.contains("excess_equal_weight"))].copy()
        cand_ics = cand_ics.sort_values("mean_rank_ic", ascending=False)
        best_ic = cand_ics.iloc[0]
        spread_match = spread_df[(spread_df["factor_name"] == factor) & (spread_df["target"] == best_ic["target"])]
        sp = spread_match.iloc[0] if not spread_match.empty else spread_df[spread_df["factor_name"] == factor].iloc[0]
        base_cmp = baseline_df[baseline_df["alpha_factor"] == factor].iloc[0]
        beats = base_cmp["incremental_interpretation"] == "ALPHA008_BEATS_BASELINE"
        ff = fragility_df[fragility_df["factor_name"] == factor]
        liq_frag = "LOW_LIQUIDITY_DEPENDENT" if (ff["fragility_flag"] == "LOW_LIQUIDITY_DEPENDENT").any() else "not_obvious"
        size_frag = "SMALL_CAP_UNKNOWN" if (ff["fragility_flag"] == "SMALL_CAP_DEPENDENT").any() else "unavailable_not_obvious"
        neutral_row = neutral_df.iloc[0] if factor == best_factor_raw else None
        neutral_survival = neutral_row["interpretation"] if neutral_row is not None else "not_tested_directly"
        strong = bool(beats and neutral_survival == "SURVIVES_NEUTRALIZATION" and liq_frag == "not_obvious" and np.isfinite(best_ic["mean_rank_ic"]) and float(best_ic["mean_rank_ic"]) > 0.015 and np.isfinite(sp["top_minus_bottom_mean"]) and float(sp["top_minus_bottom_mean"]) > 0)
        if strong:
            cls = "PROMISING_FOR_RESIDUAL_ALPHA_TEST"
            action = "CONTINUE_TO_RESIDUAL_ALPHA_TEST"
        elif np.isfinite(best_ic["mean_rank_ic"]) and float(best_ic["mean_rank_ic"]) > 0 and best_ic["frequency"] == "daily":
            cls = "HORIZON_SPECIFIC_BUT_FRAGILE"
            action = "KEEP_AS_SHORT_HORIZON_DIAGNOSTIC"
        elif base_cmp["incremental_interpretation"] == "SIMILAR_TO_REVERSAL":
            cls = "SIMILAR_TO_REVERSAL_BASELINE"
            action = "COMPARE_MORE_ALPHA101_FACTORS"
        elif neutral_survival == "WEAK_AFTER_NEUTRALIZATION":
            cls = "WEAK_AFTER_NEUTRALIZATION"
            action = "STOP_FOR_NOW"
        else:
            cls = "INVALID_DATA_QUALITY" if not np.isfinite(best_ic["mean_rank_ic"]) else "WEAK_AFTER_NEUTRALIZATION"
            action = "STOP_FOR_NOW"
        score_rows.append({
            "factor_name": factor,
            "best_target": best_ic["target"],
            "best_frequency": best_ic["frequency"],
            "mean_rank_ic": best_ic["mean_rank_ic"],
            "ic_ir": best_ic["ic_ir"],
            "positive_ic_rate": best_ic["positive_ic_rate"],
            "spread_mean": sp["top_minus_bottom_mean"],
            "monotonicity_score": sp["monotonicity_score"],
            "beats_reversal_baseline": bool(beats),
            "liquidity_fragility": liq_frag,
            "size_fragility": size_frag,
            "neutralization_survival": neutral_survival,
            "overall_classification": cls,
            "recommended_action": action,
            "notes": "Conservative score; no tradable alpha or production-ready claim.",
        })
    scorecard = pd.DataFrame(score_rows).sort_values("mean_rank_ic", ascending=False)
    scorecard_path = OUT / "alpha008_scorecard_v1.csv"
    scorecard.to_csv(scorecard_path, index=False, encoding="utf-8-sig")
    best = scorecard.iloc[0]
    n_promising = int((scorecard["overall_classification"] == "PROMISING_FOR_RESIDUAL_ALPHA_TEST").sum())
    if n_promising:
        decision = "ALPHA008_AUDIT_PROMISING_READY_FOR_RESIDUAL_TEST"
    elif (scorecard["overall_classification"] == "HORIZON_SPECIFIC_BUT_FRAGILE").any():
        decision = "ALPHA008_AUDIT_HORIZON_SPECIFIC_BUT_FRAGILE"
    elif (baseline_df["incremental_interpretation"] == "SIMILAR_TO_REVERSAL").any():
        decision = "ALPHA008_AUDIT_SIMILAR_TO_REVERSAL_BASELINE"
    elif (scorecard["overall_classification"] == "WEAK_AFTER_NEUTRALIZATION").any() or (scorecard["neutralization_survival"] == "WEAK_AFTER_NEUTRALIZATION").any():
        decision = "ALPHA008_AUDIT_WEAK_AFTER_NEUTRALIZATION"
    else:
        decision = "ALPHA008_AUDIT_INVALID_DATA_QUALITY"
    can_enter = bool(n_promising)
    recommended_next = "Continue to residual alpha test" if can_enter else "Compare more Alpha101 factors; keep Alpha008 as short-horizon diagnostic only"

    report_path = OUT / "alpha008_a_share_reversal_audit_report_v1.md"
    report_path.write_text("\n".join([
        "# Alpha008 A-Share Reversal Factor Audit v1",
        "",
        "## 1. Executive Summary",
        "",
        f"- Decision: {decision}",
        f"- Best Alpha008 variant: {best['factor_name']}",
        f"- Best target/frequency: {best['best_target']} / {best['best_frequency']}",
        f"- Best mean Rank IC: {float(best['mean_rank_ic']):.6f}",
        f"- Classification: {best['overall_classification']}",
        "- Conservative conclusion: this audit does not claim tradable alpha or production readiness.",
        "",
        "## 2. Scope and Non-Goals",
        "",
        "- This task is not model training.",
        "- This task is not a full backtest.",
        "- This task does not connect to Blend V3 or Compact-F.",
        "- This task does not modify production.",
        "- This task does not solve the CSMAR PIT financial issue.",
        "",
        "## 3. Alpha008 Formula and Implementation",
        "",
        "Implemented raw = rolling_sum(open, 5) * rolling_sum(daily_return, 5) minus its 10-trading-day delay, computed per symbol. Cross-sectional rank and z-score variants are computed by trading date.",
        "",
        "## 4. Input Data Audit",
        "",
        pd.DataFrame(audit_rows).to_markdown(index=False),
        "",
        "## 5. Factor Construction",
        "",
        "All rolling windows and delays are by symbol. Cross-sectional transforms are by date. Factor values are oriented so larger values are interpreted as more bullish for diagnostics only.",
        "",
        "## 6. Label Construction",
        "",
        "5D/10D/20D labels use future daily close shifts. 1M labels use next month-end close. Labels are not written back to the main panel.",
        "",
        "## 7. IC Results by Horizon",
        "",
        ic_df.to_markdown(index=False),
        "",
        "## 8. Group Return Diagnostics",
        "",
        spread_df.to_markdown(index=False),
        "",
        "## 9. Comparison with Reversal Baseline",
        "",
        baseline_df.to_markdown(index=False),
        "",
        "## 10. Fragility Audit",
        "",
        fragility_df.to_markdown(index=False),
        "",
        "## 11. Neutralization Results",
        "",
        neutral_df.to_markdown(index=False),
        "",
        "## 12. Scorecard",
        "",
        scorecard.to_markdown(index=False),
        "",
        "## 13. Limitations",
        "",
        "- all_daily lacks market-cap, turnover, ST, paused, trade-status, and limit fields.",
        "- Size audit is limited because market-cap proxy is unavailable.",
        "- Alpha008 can only enter residual alpha testing if it clearly beats reversal_20d_liquid_only and survives filters/neutralization.",
        "",
        "## 14. Recommended Next Task",
        "",
        recommended_next,
        "",
        "## 15. Files Generated",
        "",
        "\n".join(f"- `{rel(p)}`" for p in [
            input_audit_path, research_universe_path, date_index_path, daily_factor_panel_path,
            monthly_factor_panel_path, label_panel_path, label_qa_path, single_factor_ic_path,
            decile_return_path, group_spread_path, baseline_comparison_path, fragility_audit_path,
            neutralization_audit_path, scorecard_path
        ]),
        "",
    ]), encoding="utf-8")

    task_completion_card_path = OUT / "task_completion_card.md"
    task_completion_card_path.write_text("\n".join([
        "任务名称：Alpha008 A-Share Reversal Factor Audit v1",
        f"运行日期：{date.today().isoformat()}",
        "是否修改 production：否",
        f"是否修改 README：{'否' if file_hash(README) == protected['README'] else '是'}",
        f"是否修改 all_daily：{'否' if file_hash(ALL_DAILY) == protected['all_daily'] else '是'}",
        f"是否修改 training_panel：{'否' if file_hash(TRAINING) == protected['training'] else '是'}",
        "是否训练模型：否",
        "是否运行回测：否",
        "是否做 IC：是",
        "是否生成交易信号：否",
        f"核心输出：{rel(report_path)}",
        f"核心结论：{decision}",
        f"最佳 Alpha008 变体：{best['factor_name']}",
        f"是否优于 reversal baseline：{bool(best['beats_reversal_baseline'])}",
        f"是否仍存在小盘/流动性脆弱性：{best['size_fragility']} / {best['liquidity_fragility']}",
        f"是否可以进入 residual alpha test：{can_enter}",
        f"下一步建议：{recommended_next}",
    ]), encoding="utf-8")

    update_status("audit_completed" if decision != "ALPHA008_AUDIT_INVALID_DATA_QUALITY" else "audit_inconclusive")
    append_decision(decision, bool(best["beats_reversal_baseline"]), str(best["neutralization_survival"]), can_enter)
    run_status_scripts()

    model_after = {rel(p): file_hash(p) for p in (ROOT / "output").glob("production_models*/**/*") if p.is_file()}
    config_after = {rel(p): file_hash(p) for p in (ROOT / "config").glob("*") if p.is_file() and p.name != "project_status.yaml"}
    readme_modified = file_hash(README) != protected["README"]
    all_daily_modified = file_hash(ALL_DAILY) != protected["all_daily"]
    training_modified = file_hash(TRAINING) != protected["training"]
    production_modified = file_hash(PAPER) != protected["paper"] or model_before != model_after or config_before != config_after
    current_csmar = yaml.safe_load(STATUS.read_text(encoding="utf-8")).get("alternative_data", {}).get("csmar_status")
    final_decision = "INVALID_MODIFICATION" if any([readme_modified, all_daily_modified, training_modified, production_modified]) else decision
    final_qa_path = OUT / "final_qa_alpha008_a_share_reversal_audit_v1.csv"
    qa_rows = [
        ("README.md not modified", not readme_modified, str(readme_modified)),
        ("all_daily.parquet not modified", not all_daily_modified, str(all_daily_modified)),
        ("training_panel_v15_sr.parquet not modified", not training_modified, str(training_modified)),
        ("model files not modified", model_before == model_after, "production_models* unchanged"),
        ("paper_trading_pipeline.py not modified", file_hash(PAPER) == protected["paper"], "hash checked"),
        ("production config not modified", config_before == config_after, "config files except project_status.yaml unchanged"),
        ("no model training executed", True, "only formula factors and residualization"),
        ("no full backtest executed", True, "group diagnostics only"),
        ("no trading signal generated", True, "no signal output written"),
        ("no real orders generated", True, "paper trading untouched"),
        ("no CSMAR API access executed", True, "no CSMAR code path called"),
        ("no MediaCrawler executed", True, "MediaCrawler untouched"),
        ("root-level output used", str(OUT).startswith(str(ROOT / "output")), rel(OUT)),
        ("xhs/output not used for new outputs", not str(OUT).replace("\\", "/").startswith(str(ROOT / "xhs" / "output").replace("\\", "/")), rel(OUT)),
        ("Alpha008 factor panel generated", daily_factor_panel_path.exists() and monthly_factor_panel_path.exists(), rel(daily_factor_panel_path)),
        ("label panel generated", label_panel_path.exists(), rel(label_panel_path)),
        ("IC by horizon generated", single_factor_ic_path.exists(), rel(single_factor_ic_path)),
        ("group return diagnostics generated", decile_return_path.exists() and group_spread_path.exists(), rel(group_spread_path)),
        ("reversal baseline comparison generated", baseline_comparison_path.exists(), rel(baseline_comparison_path)),
        ("fragility audit generated", fragility_audit_path.exists(), rel(fragility_audit_path)),
        ("neutralization audit generated", neutralization_audit_path.exists(), rel(neutralization_audit_path)),
        ("scorecard generated", scorecard_path.exists(), rel(scorecard_path)),
        ("final report generated", report_path.exists(), rel(report_path)),
        ("task completion card generated", task_completion_card_path.exists(), rel(task_completion_card_path)),
        ("project_status.yaml updated without overwriting CSMAR status", current_csmar == original_csmar, rel(STATUS)),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS.exists(), rel(CURRENT_STATUS)),
        ("DECISIONS.md appended", "Alpha008 A-Share Reversal Factor Audit v1" in DECISIONS.read_text(encoding="utf-8"), rel(DECISIONS)),
        ("README consistency check executed", README_REPORT.exists(), rel(README_REPORT)),
        ("README not auto-modified", not readme_modified, str(readme_modified)),
        ("conclusion uses conservative language", "does not claim tradable alpha" in report_path.read_text(encoding="utf-8"), "conservative wording checked"),
    ]
    pd.DataFrame(qa_rows, columns=["check", "pass", "details"]).to_csv(final_qa_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_data_audit_path": rel(input_audit_path),
        "research_universe_path": rel(research_universe_path),
        "date_index_path": rel(date_index_path),
        "daily_factor_panel_path": rel(daily_factor_panel_path),
        "monthly_factor_panel_path": rel(monthly_factor_panel_path),
        "label_panel_path": rel(label_panel_path),
        "label_qa_path": rel(label_qa_path),
        "single_factor_ic_path": rel(single_factor_ic_path),
        "decile_return_path": rel(decile_return_path),
        "group_spread_path": rel(group_spread_path),
        "baseline_comparison_path": rel(baseline_comparison_path),
        "fragility_audit_path": rel(fragility_audit_path),
        "neutralization_audit_path": rel(neutralization_audit_path),
        "scorecard_path": rel(scorecard_path),
        "report_path": rel(report_path),
        "task_completion_card_path": rel(task_completion_card_path),
        "final_qa_path": rel(final_qa_path),
        "project_status_path": rel(STATUS),
        "current_status_doc_path": rel(CURRENT_STATUS),
        "decisions_doc_path": rel(DECISIONS),
        "readme_consistency_report_path": rel(README_REPORT),
        "n_symbols": int(universe["symbol"].nunique()),
        "n_daily_dates": int(daily_dates["date"].nunique()),
        "n_monthly_dates": int(len(monthly_dates)),
        "n_factor_rows_daily": int(len(daily_panel)),
        "n_factor_rows_monthly": int(len(monthly_panel)),
        "best_alpha008_variant": best["factor_name"],
        "best_target": best["best_target"],
        "best_frequency": best["best_frequency"],
        "best_mean_rank_ic": best["mean_rank_ic"],
        "best_ic_ir": best["ic_ir"],
        "best_spread": best["spread_mean"],
        "best_classification": best["overall_classification"],
        "beats_reversal_baseline": bool(best["beats_reversal_baseline"]),
        "neutralization_survival": best["neutralization_survival"],
        "n_promising_factors": n_promising,
        "recommended_next_task": recommended_next,
        "can_enter_residual_alpha_test": can_enter,
        "readme_modified": readme_modified,
        "all_daily_modified": all_daily_modified,
        "training_panel_modified": training_modified,
        "production_modified": production_modified,
        "csmar_api_accessed": False,
        "decision": final_decision,
    }
    (OUT / "terminal_summary_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for k, v in summary.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
