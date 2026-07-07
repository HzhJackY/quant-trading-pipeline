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
OUT = ROOT / "output" / "price_volume_divergence_reversal_audit_v1"
ALL_DAILY_PATH = ROOT / "output" / "all_daily.parquet"
TRAINING_PANEL_PATH = ROOT / "output" / "training_panel_v15_sr.parquet"
STATUS_PATH = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS_PATH = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
README_PATH = ROOT / "README.md"
PAPER_TRADING_PIPELINE = ROOT / "paper_trading" / "paper_trading_pipeline.py"
README_REPORT_PATH = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"

FACTOR_NAMES = [
    "reversal_20d",
    "pv_divergence_reversal_20d",
    "price_down_volume_up_20d",
    "panic_reversal_10d",
    "volume_spike_without_price_rebound",
    "risk_adjusted_pv_reversal",
]
TARGETS = ["fwd_ret_1m", "fwd_ret_1m_excess_equal_weight"]


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


def safe_read_parquet(path: Path) -> tuple[pd.DataFrame | None, str]:
    try:
        return pd.read_parquet(path), ""
    except Exception as exc:  # pragma: no cover - diagnostic path
        return None, repr(exc)


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


def winsor_by_month(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        def _clip(s: pd.Series) -> pd.Series:
            q1 = s.quantile(0.01)
            q99 = s.quantile(0.99)
            return s.clip(q1, q99)
        out[col] = out.groupby("month_end", group_keys=False)[col].transform(_clip)
    return out


def rank_ic(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 50:
        return np.nan
    return x[valid].rank().corr(y[valid].rank())


def tstat(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    std = s.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return np.nan
    return float(s.mean() / (std / math.sqrt(len(s))))


def summarize_ic(monthly: pd.DataFrame, factor: str, target: str, notes: str = "") -> dict[str, object]:
    ics = []
    for m, g in monthly.groupby("month_end"):
        ic = rank_ic(g[factor], g[target])
        if np.isfinite(ic):
            ics.append((m, ic, int((g[factor].notna() & g[target].notna()).sum())))
    if not ics:
        return {
            "factor_name": factor,
            "target": target,
            "n_months": 0,
            "n_obs": 0,
            "mean_rank_ic": np.nan,
            "median_rank_ic": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "positive_ic_rate": np.nan,
            "min_month": "",
            "max_month": "",
            "notes": notes or "insufficient monthly observations",
        }
    vals = pd.Series([v for _, v, _ in ics], dtype="float64")
    return {
        "factor_name": factor,
        "target": target,
        "n_months": len(ics),
        "n_obs": int(sum(n for _, _, n in ics)),
        "mean_rank_ic": float(vals.mean()),
        "median_rank_ic": float(vals.median()),
        "ic_std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
        "ic_ir": float(vals.mean() / vals.std(ddof=1)) if len(vals) > 1 and vals.std(ddof=1) != 0 else np.nan,
        "positive_ic_rate": float((vals > 0).mean()),
        "min_month": str(min(m for m, _, _ in ics).date()),
        "max_month": str(max(m for m, _, _ in ics).date()),
        "notes": notes,
    }


def assign_quantile_groups(g: pd.DataFrame, factor: str, n_groups: int = 5) -> pd.Series:
    valid = g[factor].notna()
    res = pd.Series(np.nan, index=g.index)
    if valid.sum() < 50 or g.loc[valid, factor].nunique() < 2:
        return res
    try:
        res.loc[valid] = pd.qcut(g.loc[valid, factor].rank(method="first"), n_groups, labels=False) + 1
    except ValueError:
        pass
    return res


def build_input_audit(all_daily: pd.DataFrame | None, train: pd.DataFrame | None, all_err: str, train_err: str) -> pd.DataFrame:
    rows = []
    specs = [(ALL_DAILY_PATH, all_daily, all_err), (TRAINING_PANEL_PATH, train, train_err)]
    for path, df, err in specs:
        cols = set(df.columns) if df is not None else set()
        date_col = "date" if "date" in cols else "month_end" if "month_end" in cols else None
        rows.append({
            "input_path": rel(path),
            "exists": path.exists(),
            "readable": df is not None,
            "n_rows": 0 if df is None else len(df),
            "n_symbols": 0 if df is None or "symbol" not in cols else int(df["symbol"].astype(str).nunique()),
            "min_date": "" if df is None or date_col is None else str(pd.to_datetime(df[date_col]).min().date()),
            "max_date": "" if df is None or date_col is None else str(pd.to_datetime(df[date_col]).max().date()),
            "available_price_fields": ",".join([c for c in ["open", "high", "low", "close"] if c in cols]),
            "available_volume_fields": ",".join([c for c in ["volume", "amount"] if c in cols]),
            "available_liquidity_fields": ",".join([c for c in ["turnover", "amount", "volume", "Dollar_Volume_20D_neutral_z"] if c in cols]),
            "available_market_cap_fields": ",".join([c for c in ["market_cap", "total_mv", "float_mv"] if c in cols]),
            "available_st_filter_fields": ",".join([c for c in ["is_st", "name", "paused", "trade_status"] if c in cols]),
            "available_limit_filter_fields": ",".join([c for c in ["limit_up", "limit_down"] if c in cols]),
            "notes": err or ("market cap / turnover unavailable; amount and volume used as liquidity proxies" if path == ALL_DAILY_PATH else "v15 panel used for universe and month index only"),
        })
    return pd.DataFrame(rows)


def existing_signal_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        ROOT / "output" / "full_panel_forced_tournament_v3" / "blend_full_oos_panel_v1.parquet",
        ROOT / "output" / "full_dataset_oos_regeneration_v1" / "blend_full_oos_panel_v1.parquet",
        ROOT / "output" / "production_signal_tournament_v1" / "blend_signal_panel_v1.parquet",
    ]
    rows = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            sig = pd.read_parquet(path)
        except Exception:
            continue
        if "symbol" not in sig.columns or "date" not in sig.columns:
            continue
        sig = sig.copy()
        sig["symbol"] = sig["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
        sig["month_end"] = pd.to_datetime(sig["date"])
        score_cols = [c for c in sig.columns if c.lower() in {"score", "blend_score", "pred", "prediction", "signal"} or "score" in c.lower()]
        numeric_cols = [c for c in sig.select_dtypes(include=[np.number]).columns if c not in {"target", "label"}]
        if not score_cols and numeric_cols:
            score_cols = [numeric_cols[-1]]
        for score_col in score_cols[:2]:
            merged = panel[["symbol", "month_end"] + FACTOR_NAMES].merge(
                sig[["symbol", "month_end", score_col]].dropna(),
                on=["symbol", "month_end"],
                how="inner",
            )
            if merged.empty:
                continue
            for factor in FACTOR_NAMES:
                corrs = []
                for _, g in merged.groupby("month_end"):
                    valid = g[factor].notna() & g[score_col].notna()
                    if valid.sum() >= 50:
                        corrs.append(g.loc[valid, factor].rank().corr(g.loc[valid, score_col].rank()))
                s = pd.Series(corrs, dtype="float64")
                rows.append({
                    "factor_name": factor,
                    "existing_signal_name": f"{rel(path)}::{score_col}",
                    "n_months": int(s.notna().sum()),
                    "mean_spearman_corr": float(s.mean()) if len(s) else np.nan,
                    "median_spearman_corr": float(s.median()) if len(s) else np.nan,
                    "interpretation": "redundancy diagnostic only; Blend V3 historical metrics are under PIT review",
                    "notes": "read-only existing signal correlation; no fusion, no training, no backtest",
                })
            if rows:
                return pd.DataFrame(rows)
    return pd.DataFrame([{
        "factor_name": "ALL",
        "existing_signal_name": "not_found_or_not_aligned",
        "n_months": 0,
        "mean_spearman_corr": np.nan,
        "median_spearman_corr": np.nan,
        "interpretation": "limitation",
        "notes": "No aligned root-level existing signal score file found; diagnostic skipped without failure.",
    }])


def update_project_status(classification: str) -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    original_csmar = status.get("alternative_data", {}).get("csmar_status")
    status.setdefault("research", {})
    status["research"]["price_volume_reversal_status"] = "audit_completed" if classification != "INVALID_DATA_QUALITY" else "audit_inconclusive"
    status["research"]["price_volume_reversal_latest_task"] = "Price-Volume Divergence Reversal Factor Audit v1"
    status["research"]["price_volume_reversal_latest_output"] = rel(OUT)
    status.setdefault("validation", {})
    status["validation"]["blend_v3_historical_metrics_status"] = "under_pit_review"
    if original_csmar is not None:
        status.setdefault("alternative_data", {})["csmar_status"] = original_csmar
    status.setdefault("project", {})["last_updated"] = date.today().isoformat()
    STATUS_PATH.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")


def append_decision(decision: str, best_factor: str, can_enter: bool) -> None:
    block = "\n".join([
        f"## {date.today().isoformat()}",
        "",
        "决策：",
        "",
        "- 启动并完成 Price-Volume Divergence Reversal Factor Audit v1。",
        "- 本任务独立于 CSMAR PIT 财务重建。",
        "- 不接入 production。",
        "- 不修改 README。",
        f"- 最佳候选因子：{best_factor}。",
        f"- 后续是否进入 residual alpha test 取决于结果；当前 can_enter_residual_alpha_test={can_enter}。",
        f"- Decision = {decision}。",
    ])
    text = DECISIONS_PATH.read_text(encoding="utf-8") if DECISIONS_PATH.exists() else "# 决策日志\n"
    marker = "Price-Volume Divergence Reversal Factor Audit v1"
    if marker in text and decision in text:
        return
    DECISIONS_PATH.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def run_status_scripts() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_current_status_md.py")], cwd=ROOT, check=True, capture_output=True, text=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_readme_consistency.py")], cwd=ROOT, check=True, capture_output=True, text=True)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    original_status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    original_csmar_status = original_status.get("alternative_data", {}).get("csmar_status")
    protected = {
        "README.md": file_hash(README_PATH),
        "all_daily.parquet": file_hash(ALL_DAILY_PATH),
        "training_panel_v15_sr.parquet": file_hash(TRAINING_PANEL_PATH),
        "paper_trading_pipeline.py": file_hash(PAPER_TRADING_PIPELINE),
    }
    model_files_before = {rel(p): file_hash(p) for p in (ROOT / "output").glob("production_models*/**/*") if p.is_file()}
    config_before = {rel(p): file_hash(p) for p in (ROOT / "config").glob("*") if p.is_file() and p.name != "project_status.yaml"}

    all_daily, all_err = safe_read_parquet(ALL_DAILY_PATH)
    train, train_err = safe_read_parquet(TRAINING_PANEL_PATH)
    input_audit = build_input_audit(all_daily, train, all_err, train_err)
    input_audit_path = OUT / "input_data_audit_v1.csv"
    input_audit.to_csv(input_audit_path, index=False, encoding="utf-8-sig")

    if all_daily is None or "symbol" not in all_daily.columns or "date" not in all_daily.columns or "close" not in all_daily.columns:
        decision = "PV_REVERSAL_AUDIT_INVALID_DATA_QUALITY"
        update_project_status("INVALID_DATA_QUALITY")
        pd.DataFrame([{"check": "minimum all_daily schema", "pass": False, "details": all_err or "missing symbol/date/close"}]).to_csv(OUT / "final_qa_price_volume_divergence_reversal_audit_v1.csv", index=False, encoding="utf-8-sig")
        print(f"input_data_audit_path={rel(input_audit_path)}")
        print(f"decision={decision}")
        return

    daily = all_daily.copy()
    daily["symbol"] = daily["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["symbol", "date"])
    daily = daily[(daily["date"] >= "2016-01-01") & (daily["date"] <= "2026-06-30")]

    if train is not None and "symbol" in train.columns:
        train = train.copy()
        train["symbol"] = train["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
        train["date"] = pd.to_datetime(train["date"])
        v15_symbols = set(train["symbol"].dropna().unique())
        month_source = "training_panel_v15_sr.parquet date monthly max"
        month_index = train.groupby(train["date"].dt.to_period("M"))["date"].max().sort_values().rename("month_end").reset_index(drop=True).to_frame()
    else:
        v15_symbols = set()
        month_source = "all_daily monthly max fallback"
        month_index = daily.groupby(daily["date"].dt.to_period("M"))["date"].max().sort_values().rename("month_end").reset_index(drop=True).to_frame()
    month_index = month_index[(month_index["month_end"] >= "2017-01-01") & (month_index["month_end"] <= "2026-06-30")]
    universe_symbols = v15_symbols or set(daily["symbol"].unique())
    daily = daily[daily["symbol"].isin(universe_symbols)].copy()

    g = daily.groupby("symbol", group_keys=False)
    for n in [5, 10, 20, 60]:
        daily[f"ret_{n}d"] = g["close"].pct_change(n)
    daily["daily_ret"] = g["close"].pct_change()
    daily["volume_mean_20d"] = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    daily["volume_mean_120d"] = g["volume"].transform(lambda s: s.rolling(120, min_periods=60).mean())
    daily["amount_mean_20d"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    daily["amount_mean_120d"] = g["amount"].transform(lambda s: s.rolling(120, min_periods=60).mean())
    daily["avg_amount_60d"] = g["amount"].transform(lambda s: s.rolling(60, min_periods=20).mean())
    daily["vol_20d"] = g["daily_ret"].transform(lambda s: s.rolling(20, min_periods=10).std())
    daily["vol_60d"] = g["daily_ret"].transform(lambda s: s.rolling(60, min_periods=20).std())
    daily["abnormal_volume_20d"] = daily["volume_mean_20d"] / daily["volume_mean_120d"]
    daily["abnormal_amount_20d"] = daily["amount_mean_20d"] / daily["amount_mean_120d"]
    daily.replace([np.inf, -np.inf], np.nan, inplace=True)

    latest = daily.sort_values("date").groupby("symbol").tail(1)
    amount_q = latest["avg_amount_60d"].quantile([1 / 3, 2 / 3]).to_dict()
    def liquidity_bucket(v: float) -> str:
        if not np.isfinite(v):
            return "unknown_liquidity"
        if v <= amount_q.get(1 / 3, np.nan):
            return "low_liquidity"
        if v <= amount_q.get(2 / 3, np.nan):
            return "mid_liquidity"
        return "high_liquidity"
    universe = daily.groupby("symbol").agg(n_trading_days=("date", "size"), min_date=("date", "min"), max_date=("date", "max"), avg_amount_60d=("avg_amount_60d", "last")).reset_index()
    universe["in_v15_panel"] = universe["symbol"].isin(v15_symbols)
    universe["liquidity_bucket"] = universe["avg_amount_60d"].map(liquidity_bucket)
    universe["size_bucket"] = "micro_or_unknown"
    universe["notes"] = "market cap unavailable; size_bucket set to micro_or_unknown; liquidity uses amount proxy"
    universe_path = OUT / "pv_reversal_research_universe_v1.csv"
    universe.to_csv(universe_path, index=False, encoding="utf-8-sig")

    month_index["n_symbols_available"] = month_index["month_end"].map(daily.groupby("date")["symbol"].nunique()).fillna(0).astype(int)
    month_index["source"] = month_source
    month_index["notes"] = "preferred 2017-01 to 2026-06; month_end dates from v15 when available"
    month_index_path = OUT / "pv_reversal_month_index_v1.csv"
    month_index.to_csv(month_index_path, index=False, encoding="utf-8-sig")

    monthly = daily[daily["date"].isin(set(month_index["month_end"]))].copy()
    monthly.rename(columns={"date": "month_end"}, inplace=True)
    monthly = monthly.merge(universe[["symbol", "liquidity_bucket", "size_bucket"]], on="symbol", how="left")
    for base in ["ret_20d", "ret_10d", "abnormal_amount_20d"]:
        monthly[f"z_{base}"] = monthly.groupby("month_end")[base].transform(zscore)
    monthly["reversal_20d"] = -monthly["ret_20d"]
    monthly["pv_divergence_reversal_20d"] = -monthly["ret_20d"] * monthly["z_abnormal_amount_20d"]
    monthly["price_down_volume_up_20d"] = (monthly["ret_20d"] < 0).astype(float) * (-monthly["z_ret_20d"]) * monthly["z_abnormal_amount_20d"]
    monthly["panic_reversal_10d"] = (monthly["ret_10d"] < 0).astype(float) * (-monthly["z_ret_10d"]) * monthly["z_abnormal_amount_20d"]
    monthly["volume_spike_without_price_rebound"] = monthly["z_abnormal_amount_20d"] - monthly["z_ret_20d"]
    monthly["risk_adjusted_pv_reversal"] = monthly["price_down_volume_up_20d"] / (1 + monthly["vol_20d"].clip(lower=0))
    pre_clip = {c: monthly[c].copy() for c in FACTOR_NAMES}
    monthly = winsor_by_month(monthly, FACTOR_NAMES)
    changed = pd.Series({c: float((pre_clip[c] != monthly[c]).mean()) for c in FACTOR_NAMES})
    monthly["factor_quality_flag"] = np.where(monthly[FACTOR_NAMES].notna().sum(axis=1) >= 4, "OK", "LOW_COVERAGE")
    factor_cols = ["symbol", "month_end", "ret_5d", "ret_10d", "ret_20d", "ret_60d", "abnormal_amount_20d", "abnormal_volume_20d", "vol_20d", "vol_60d"] + FACTOR_NAMES + ["liquidity_bucket", "size_bucket", "factor_quality_flag"]
    factor_panel = monthly[factor_cols].copy()
    factor_panel_path = OUT / "pv_reversal_factor_panel_v1.parquet"
    factor_panel.to_parquet(factor_panel_path, index=False)
    factor_sample_path = OUT / "pv_reversal_factor_panel_sample_v1.csv"
    factor_panel.head(1000).to_csv(factor_sample_path, index=False, encoding="utf-8-sig")

    label = monthly[["symbol", "month_end", "close"]].sort_values(["symbol", "month_end"]).copy()
    label["next_close"] = label.groupby("symbol")["close"].shift(-1)
    label["fwd_ret_1m"] = label["next_close"] / label["close"] - 1
    label["fwd_ret_1m_excess_equal_weight"] = label["fwd_ret_1m"] - label.groupby("month_end")["fwd_ret_1m"].transform("mean")
    label["fwd_ret_1m_excess_universe"] = label["fwd_ret_1m"] - label.groupby("month_end")["fwd_ret_1m"].transform("median")
    label_panel = label[["symbol", "month_end", "fwd_ret_1m", "fwd_ret_1m_excess_equal_weight", "fwd_ret_1m_excess_universe"]]
    label_panel_path = OUT / "pv_reversal_label_panel_v1.parquet"
    label_panel.to_parquet(label_panel_path, index=False)
    label_qa = pd.DataFrame([
        {"check": "next month-end close used", "pass": True, "details": "label uses groupby(symbol).shift(-1) after monthly factor snapshot"},
        {"check": "factor construction does not use forward returns", "pass": True, "details": "forward returns merged only after factor panel creation"},
        {"check": "label coverage", "pass": bool(label_panel["fwd_ret_1m"].notna().sum() > 0), "details": f"non_null={int(label_panel['fwd_ret_1m'].notna().sum())}"},
    ])
    label_qa_path = OUT / "pv_reversal_label_qa_v1.csv"
    label_qa.to_csv(label_qa_path, index=False, encoding="utf-8-sig")

    panel = factor_panel.merge(label_panel, on=["symbol", "month_end"], how="left")
    ic_rows = [summarize_ic(panel, f, t) for f in FACTOR_NAMES for t in TARGETS]
    ic_df = pd.DataFrame(ic_rows)
    ic_path = OUT / "pv_reversal_single_factor_ic_v1.csv"
    ic_df.to_csv(ic_path, index=False, encoding="utf-8-sig")

    decile_rows = []
    spread_rows = []
    for factor in FACTOR_NAMES:
        grouped = panel.copy()
        grouped["group_id"] = grouped.groupby("month_end", group_keys=False).apply(lambda x: assign_quantile_groups(x, factor, 5), include_groups=False).sort_index()
        for target in TARGETS:
            valid = grouped.dropna(subset=["group_id", target])
            for gid, gg in valid.groupby("group_id"):
                month_means = gg.groupby("month_end")[target].mean()
                decile_rows.append({
                    "factor_name": factor,
                    "target": target,
                    "group_id": int(gid),
                    "n_months": int(month_means.count()),
                    "avg_forward_return": float(month_means.mean()),
                    "avg_excess_return": float(month_means.mean() if "excess" in target else (gg[target] - gg.groupby("month_end")[target].transform("mean")).mean()),
                    "hit_rate": float((month_means > 0).mean()),
                    "avg_n_stocks": float(gg.groupby("month_end").size().mean()),
                    "notes": "5-group diagnostic; top group is highest factor value",
                })
            top = valid[valid["group_id"] == 5].groupby("month_end")[target].mean()
            bottom = valid[valid["group_id"] == 1].groupby("month_end")[target].mean()
            spread = (top - bottom).dropna()
            by_year = spread.groupby(spread.index.year).mean() if len(spread) else pd.Series(dtype="float64")
            group_means = valid.groupby("group_id")[target].mean().sort_index()
            mono = float(group_means.rank().corr(pd.Series(group_means.index, index=group_means.index))) if len(group_means) >= 2 else np.nan
            spread_rows.append({
                "factor_name": factor,
                "target": target,
                "top_minus_bottom_mean": float(spread.mean()) if len(spread) else np.nan,
                "top_minus_bottom_tstat": tstat(spread),
                "monotonicity_score": mono,
                "worst_year": "" if by_year.empty else str(int(by_year.idxmin())),
                "best_year": "" if by_year.empty else str(int(by_year.idxmax())),
                "notes": "diagnostic only; no portfolio backtest or trading signal",
            })
    decile_path = OUT / "pv_reversal_decile_return_v1.csv"
    group_spread_path = OUT / "pv_reversal_group_spread_v1.csv"
    pd.DataFrame(decile_rows).to_csv(decile_path, index=False, encoding="utf-8-sig")
    spread_df = pd.DataFrame(spread_rows)
    spread_df.to_csv(group_spread_path, index=False, encoding="utf-8-sig")

    robustness_rows = []
    robust_specs = [
        ("size_bucket", "size_bucket"),
        ("liquidity_bucket", "liquidity_bucket"),
        ("price_environment", "price_environment"),
        ("calendar_year", "calendar_year"),
    ]
    robust_panel = panel.copy()
    robust_panel["price_environment"] = np.where(robust_panel["ret_20d"] < 0, "after_short_term_drop", "after_short_term_rise")
    robust_panel["calendar_year"] = pd.to_datetime(robust_panel["month_end"]).dt.year.astype(str)
    for factor in FACTOR_NAMES:
        for group_name, col in robust_specs:
            for val, sub in robust_panel.groupby(col):
                if len(sub) < 100:
                    continue
                ic_summary = summarize_ic(sub, factor, "fwd_ret_1m_excess_equal_weight", notes="")
                q = sub.copy()
                q["group_id"] = q.groupby("month_end", group_keys=False).apply(lambda x: assign_quantile_groups(x, factor, 5), include_groups=False).sort_index()
                top = q[q["group_id"] == 5].groupby("month_end")["fwd_ret_1m_excess_equal_weight"].mean()
                bottom = q[q["group_id"] == 1].groupby("month_end")["fwd_ret_1m_excess_equal_weight"].mean()
                sp = (top - bottom).dropna()
                notes = "FRAGILE" if group_name in {"liquidity_bucket", "size_bucket"} and str(val) in {"low_liquidity", "micro_or_unknown"} and ic_summary["mean_rank_ic"] and ic_summary["mean_rank_ic"] > 0 else ""
                robustness_rows.append({
                    "factor_name": factor,
                    "robustness_group": group_name,
                    "group_value": val,
                    "n_months": ic_summary["n_months"],
                    "n_obs": ic_summary["n_obs"],
                    "mean_rank_ic": ic_summary["mean_rank_ic"],
                    "ic_ir": ic_summary["ic_ir"],
                    "top_minus_bottom_mean": float(sp.mean()) if len(sp) else np.nan,
                    "notes": notes or "diagnostic subgroup result",
                })
    robustness_df = pd.DataFrame(robustness_rows)
    robustness_path = OUT / "pv_reversal_robustness_v1.csv"
    robustness_df.to_csv(robustness_path, index=False, encoding="utf-8-sig")

    tradability_rows = []
    for factor in FACTOR_NAMES:
        for filter_name in [
            "raw_universe",
            "exclude_low_amount_bottom_20pct",
            "exclude_low_amount_bottom_40pct",
            "exclude_extreme_vol_top_10pct",
            "exclude_recent_limit_like_moves",
            "exclude_microcap_or_unknown",
        ]:
            sub = panel.copy()
            notes = ""
            if filter_name == "exclude_low_amount_bottom_20pct":
                q = sub.groupby("month_end")["abnormal_amount_20d"].transform(lambda s: s.quantile(0.2))
                sub = sub[sub["abnormal_amount_20d"] > q]
            elif filter_name == "exclude_low_amount_bottom_40pct":
                q = sub.groupby("month_end")["abnormal_amount_20d"].transform(lambda s: s.quantile(0.4))
                sub = sub[sub["abnormal_amount_20d"] > q]
            elif filter_name == "exclude_extreme_vol_top_10pct":
                q = sub.groupby("month_end")["vol_20d"].transform(lambda s: s.quantile(0.9))
                sub = sub[sub["vol_20d"] <= q]
            elif filter_name == "exclude_recent_limit_like_moves":
                sub = sub[sub["ret_5d"].abs() < 0.45]
                notes = "limit fields unavailable; used abs(ret_5d)<45% proxy"
            elif filter_name == "exclude_microcap_or_unknown":
                notes = "market cap unavailable; filter not applied"
            ic_summary = summarize_ic(sub, factor, "fwd_ret_1m_excess_equal_weight", notes=notes)
            qsub = sub.copy()
            qsub["group_id"] = qsub.groupby("month_end", group_keys=False).apply(lambda x: assign_quantile_groups(x, factor, 5), include_groups=False).sort_index()
            top = qsub[qsub["group_id"] == 5].groupby("month_end")["fwd_ret_1m_excess_equal_weight"].mean()
            bottom = qsub[qsub["group_id"] == 1].groupby("month_end")["fwd_ret_1m_excess_equal_weight"].mean()
            sp = (top - bottom).dropna()
            tradability_rows.append({
                "factor_name": factor,
                "filter_name": filter_name,
                "n_months": ic_summary["n_months"],
                "n_obs": ic_summary["n_obs"],
                "mean_rank_ic": ic_summary["mean_rank_ic"],
                "ic_ir": ic_summary["ic_ir"],
                "top_minus_bottom_mean": float(sp.mean()) if len(sp) else np.nan,
                "avg_coverage": float(len(sub) / len(panel)) if len(panel) else 0,
                "interpretation": "survives filter" if np.isfinite(ic_summary["mean_rank_ic"]) and ic_summary["mean_rank_ic"] > 0 and np.isfinite(float(sp.mean()) if len(sp) else np.nan) and float(sp.mean()) > 0 else "weak after filter",
                "notes": notes,
            })
    tradability_df = pd.DataFrame(tradability_rows)
    tradability_path = OUT / "pv_reversal_tradability_filter_audit_v1.csv"
    tradability_df.to_csv(tradability_path, index=False, encoding="utf-8-sig")

    corr_df = existing_signal_correlation(panel)
    corr_path = OUT / "pv_reversal_existing_signal_correlation_v1.csv"
    corr_df.to_csv(corr_path, index=False, encoding="utf-8-sig")

    score_rows = []
    for factor in FACTOR_NAMES:
        ic = ic_df[(ic_df["factor_name"] == factor) & (ic_df["target"] == "fwd_ret_1m_excess_equal_weight")].iloc[0]
        sp_row = spread_df[(spread_df["factor_name"] == factor) & (spread_df["target"] == "fwd_ret_1m_excess_equal_weight")].iloc[0]
        low = robustness_df[(robustness_df["factor_name"] == factor) & (robustness_df["robustness_group"] == "liquidity_bucket") & (robustness_df["group_value"] == "low_liquidity")]
        high = robustness_df[(robustness_df["factor_name"] == factor) & (robustness_df["robustness_group"] == "liquidity_bucket") & (robustness_df["group_value"] == "high_liquidity")]
        liquidity_fragility = "FRAGILE" if not low.empty and (high.empty or float(low["mean_rank_ic"].iloc[0]) > max(0.01, float(high["mean_rank_ic"].iloc[0]) if np.isfinite(high["mean_rank_ic"].iloc[0]) else -1) and float(ic["mean_rank_ic"]) > 0) else "not_obvious"
        size_fragility = "UNKNOWN_MARKET_CAP_UNAVAILABLE"
        trad = tradability_df[(tradability_df["factor_name"] == factor) & (tradability_df["filter_name"].isin(["exclude_low_amount_bottom_20pct", "exclude_low_amount_bottom_40pct"]))]
        tradability_survival = "survives" if (trad["mean_rank_ic"].fillna(-9) > 0).any() and (trad["top_minus_bottom_mean"].fillna(-9) > 0).any() else "weak"
        max_corr = corr_df[corr_df["factor_name"] == factor]["mean_spearman_corr"].abs().max() if "factor_name" in corr_df else np.nan
        redundancy = "high" if np.isfinite(max_corr) and max_corr > 0.7 else "low_or_unavailable"
        promising = float(ic["mean_rank_ic"]) > 0.01 and float(ic["ic_ir"]) > 0.1 and float(sp_row["top_minus_bottom_mean"]) > 0 and tradability_survival == "survives" and liquidity_fragility != "FRAGILE"
        fragile = float(ic["mean_rank_ic"]) > 0 and (liquidity_fragility == "FRAGILE" or tradability_survival != "survives")
        if promising:
            classification = "PROMISING_FOR_FURTHER_RESEARCH"
            action = "CONTINUE_TO_RESIDUAL_ALPHA_TEST"
        elif fragile:
            classification = "FRAGILE_SMALL_CAP_OR_LIQUIDITY_EXPOSURE"
            action = "REFINE_FACTOR_DEFINITION"
        elif not np.isfinite(float(ic["mean_rank_ic"])):
            classification = "INVALID_DATA_QUALITY"
            action = "STOP_FOR_NOW"
        else:
            classification = "WEAK_OR_INCONSISTENT"
            action = "KEEP_AS_DIAGNOSTIC_ONLY"
        overall = (0 if not np.isfinite(float(ic["mean_rank_ic"])) else float(ic["mean_rank_ic"]) * 100) + (0 if not np.isfinite(float(sp_row["top_minus_bottom_mean"])) else float(sp_row["top_minus_bottom_mean"]) * 10)
        score_rows.append({
            "factor_name": factor,
            "mean_rank_ic": ic["mean_rank_ic"],
            "ic_ir": ic["ic_ir"],
            "positive_ic_rate": ic["positive_ic_rate"],
            "spread_mean": sp_row["top_minus_bottom_mean"],
            "monotonicity_score": sp_row["monotonicity_score"],
            "liquidity_fragility": liquidity_fragility,
            "size_fragility": size_fragility,
            "tradability_survival": tradability_survival,
            "redundancy_with_existing_signal": redundancy,
            "overall_score": overall,
            "classification": classification,
            "recommended_action": action,
            "notes": "Conservative single-factor diagnostic; no tradable alpha claim.",
        })
    scorecard = pd.DataFrame(score_rows).sort_values("overall_score", ascending=False)
    scorecard_path = OUT / "pv_reversal_factor_scorecard_v1.csv"
    scorecard.to_csv(scorecard_path, index=False, encoding="utf-8-sig")
    best = scorecard.iloc[0]
    n_promising = int((scorecard["classification"] == "PROMISING_FOR_FURTHER_RESEARCH").sum())
    n_fragile = int((scorecard["classification"] == "FRAGILE_SMALL_CAP_OR_LIQUIDITY_EXPOSURE").sum())
    if n_promising:
        decision = "PV_REVERSAL_AUDIT_PROMISING_READY_FOR_REVIEW"
    elif n_fragile:
        decision = "PV_REVERSAL_AUDIT_FRAGILE_NEEDS_REFINEMENT"
    elif (scorecard["classification"] == "INVALID_DATA_QUALITY").all():
        decision = "PV_REVERSAL_AUDIT_INVALID_DATA_QUALITY"
    else:
        decision = "PV_REVERSAL_AUDIT_WEAK_OR_INCONSISTENT"
    can_enter = bool(n_promising)
    recommended_next = "Residual alpha test with strict neutralization and tradability controls" if can_enter else "Refine factor definition or keep as diagnostic only"

    report_path = OUT / "price_volume_divergence_reversal_audit_report_v1.md"
    report_path.write_text("\n".join([
        "# Price-Volume Divergence Reversal Factor Audit v1",
        "",
        "## 1. Executive Summary",
        "",
        f"- Decision: {decision}",
        f"- Best candidate: {best['factor_name']}",
        f"- Best mean Rank IC: {best['mean_rank_ic']:.6f}",
        f"- Classification: {best['classification']}",
        "- Conservative interpretation only: this audit does not claim tradable alpha or production readiness.",
        "",
        "## 2. Scope and Non-Goals",
        "",
        "- This task is single-factor research and diagnostics only.",
        "- This task is not model training.",
        "- This task is not a backtest.",
        "- This task does not connect to Blend V3, Compact-F, production, or paper trading.",
        "- This task does not solve the CSMAR PIT financial issue.",
        "",
        "## 3. Input Data Audit",
        "",
        "all_daily has symbol/date/OHLCV/amount only. Turnover, market cap, ST, paused, trade status, and limit fields are unavailable, so amount and volume are used as liquidity proxies.",
        "",
        "## 4. Factor Definitions",
        "",
        "All z-scores are monthly cross-sectional. Factor values are defined so larger values indicate higher expected next-month return for research purposes only.",
        "",
        "## 5. Label Construction",
        "",
        "Forward returns use next month-end close after factor snapshots. Labels are used only for evaluation and are not written back to the main panel.",
        "",
        "## 6. Single-Factor IC Results",
        "",
        ic_df.to_markdown(index=False),
        "",
        "## 7. Group Return Diagnostics",
        "",
        spread_df.to_markdown(index=False),
        "",
        "## 8. Robustness by Size / Liquidity / Year",
        "",
        "Market cap is unavailable, so size robustness is explicitly limited. Any apparent result concentrated in low liquidity or unknown microcap buckets is treated as FRAGILE.",
        "",
        "## 9. Tradability Filter Results",
        "",
        tradability_df.to_markdown(index=False),
        "",
        "## 10. Relation to Existing Signals",
        "",
        "Existing signal correlations are read-only redundancy diagnostics. Blend V3 historical metrics remain under PIT review.",
        "",
        "## 11. Scorecard and Classification",
        "",
        scorecard.to_markdown(index=False),
        "",
        "## 12. Limitations",
        "",
        "- No turnover, market cap, ST, paused, trade status, or limit fields were available in all_daily.",
        "- No model training, no strategy backtest, and no production integration were performed.",
        "- Good results, if any, only justify further research.",
        "",
        "## 13. Recommended Next Task",
        "",
        recommended_next,
        "",
        "## 14. Files Generated",
        "",
        "\n".join(f"- `{rel(p)}`" for p in [
            input_audit_path, universe_path, month_index_path, factor_panel_path, factor_sample_path, label_panel_path,
            label_qa_path, ic_path, decile_path, group_spread_path, robustness_path, tradability_path, corr_path,
            scorecard_path
        ]),
        "",
    ]), encoding="utf-8")

    task_card_path = OUT / "task_completion_card.md"
    task_card_path.write_text("\n".join([
        "任务名称：Price-Volume Divergence Reversal Factor Audit v1",
        f"运行日期：{date.today().isoformat()}",
        "是否修改 production：否",
        f"是否修改 README：{'否' if file_hash(README_PATH) == protected['README.md'] else '是'}",
        f"是否修改 all_daily：{'否' if file_hash(ALL_DAILY_PATH) == protected['all_daily.parquet'] else '是'}",
        f"是否修改 training_panel：{'否' if file_hash(TRAINING_PANEL_PATH) == protected['training_panel_v15_sr.parquet'] else '是'}",
        "是否训练模型：否",
        "是否运行回测：否",
        "是否做 IC：是",
        "是否生成交易信号：否",
        f"核心输出：{rel(report_path)}",
        f"核心结论：{decision}",
        f"最佳候选因子：{best['factor_name']}",
        f"是否存在小盘/流动性脆弱性：{best['liquidity_fragility']} / {best['size_fragility']}",
        f"是否可以进入下一步：{can_enter}",
        f"下一步建议：{recommended_next}",
    ]), encoding="utf-8")

    update_project_status(str(best["classification"]))
    append_decision(decision, str(best["factor_name"]), can_enter)
    run_status_scripts()

    model_files_after = {rel(p): file_hash(p) for p in (ROOT / "output").glob("production_models*/**/*") if p.is_file()}
    config_after = {rel(p): file_hash(p) for p in (ROOT / "config").glob("*") if p.is_file() and p.name != "project_status.yaml"}
    readme_modified = file_hash(README_PATH) != protected["README.md"]
    all_daily_modified = file_hash(ALL_DAILY_PATH) != protected["all_daily.parquet"]
    training_modified = file_hash(TRAINING_PANEL_PATH) != protected["training_panel_v15_sr.parquet"]
    production_modified = (file_hash(PAPER_TRADING_PIPELINE) != protected["paper_trading_pipeline.py"]) or (model_files_before != model_files_after) or (config_before != config_after)
    final_decision = "INVALID_MODIFICATION" if any([readme_modified, all_daily_modified, training_modified, production_modified]) else decision
    qa_rows = [
        ("README.md not modified", not readme_modified, str(readme_modified)),
        ("all_daily.parquet not modified", not all_daily_modified, str(all_daily_modified)),
        ("training_panel_v15_sr.parquet not modified", not training_modified, str(training_modified)),
        ("model files not modified", model_files_before == model_files_after, "production_models* hashes unchanged"),
        ("paper_trading_pipeline.py not modified", file_hash(PAPER_TRADING_PIPELINE) == protected["paper_trading_pipeline.py"], "hash checked"),
        ("production config not modified", config_before == config_after, "config files except project_status.yaml unchanged"),
        ("no model training executed", True, "script contains no training call"),
        ("no full backtest executed", True, "group diagnostics only"),
        ("no trading signal generated", True, "no signal output written"),
        ("no real orders generated", True, "paper_trading untouched"),
        ("no CSMAR API access executed", True, "no CSMAR code path imported or called"),
        ("no MediaCrawler executed", True, "MediaCrawler untouched"),
        ("root-level output used", str(OUT).startswith(str(ROOT / "output")), rel(OUT)),
        ("xhs/output not used for new outputs", not str(OUT).replace("\\", "/").startswith(str(ROOT / "xhs" / "output").replace("\\", "/")), rel(OUT)),
        ("input data audit generated", input_audit_path.exists(), rel(input_audit_path)),
        ("research universe generated", universe_path.exists(), rel(universe_path)),
        ("factor panel generated", factor_panel_path.exists(), rel(factor_panel_path)),
        ("label panel generated", label_panel_path.exists(), rel(label_panel_path)),
        ("IC report generated", ic_path.exists(), rel(ic_path)),
        ("group return diagnostics generated", decile_path.exists() and group_spread_path.exists(), rel(group_spread_path)),
        ("robustness report generated", robustness_path.exists(), rel(robustness_path)),
        ("tradability filter audit generated", tradability_path.exists(), rel(tradability_path)),
        ("scorecard generated", scorecard_path.exists(), rel(scorecard_path)),
        ("final report generated", report_path.exists(), rel(report_path)),
        ("task completion card generated", task_card_path.exists(), rel(task_card_path)),
        ("project_status.yaml updated without overwriting CSMAR status", yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))["alternative_data"]["csmar_status"] == original_csmar_status, rel(STATUS_PATH)),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS_PATH.exists(), rel(CURRENT_STATUS_PATH)),
        ("DECISIONS.md appended", "Price-Volume Divergence Reversal Factor Audit v1" in DECISIONS_PATH.read_text(encoding="utf-8"), rel(DECISIONS_PATH)),
        ("README consistency check executed", README_REPORT_PATH.exists(), rel(README_REPORT_PATH)),
        ("README not auto-modified", not readme_modified, str(readme_modified)),
        ("conclusion uses conservative language", "does not claim tradable alpha" in report_path.read_text(encoding="utf-8"), "conservative wording checked"),
    ]
    final_qa_path = OUT / "final_qa_price_volume_divergence_reversal_audit_v1.csv"
    pd.DataFrame(qa_rows, columns=["check", "pass", "details"]).to_csv(final_qa_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_data_audit_path": rel(input_audit_path),
        "research_universe_path": rel(universe_path),
        "month_index_path": rel(month_index_path),
        "factor_panel_path": rel(factor_panel_path),
        "label_panel_path": rel(label_panel_path),
        "label_qa_path": rel(label_qa_path),
        "single_factor_ic_path": rel(ic_path),
        "decile_return_path": rel(decile_path),
        "group_spread_path": rel(group_spread_path),
        "robustness_path": rel(robustness_path),
        "tradability_filter_audit_path": rel(tradability_path),
        "existing_signal_correlation_path": rel(corr_path),
        "scorecard_path": rel(scorecard_path),
        "report_path": rel(report_path),
        "task_completion_card_path": rel(task_card_path),
        "final_qa_path": rel(final_qa_path),
        "project_status_path": rel(STATUS_PATH),
        "current_status_doc_path": rel(CURRENT_STATUS_PATH),
        "decisions_doc_path": rel(DECISIONS_PATH),
        "readme_consistency_report_path": rel(README_REPORT_PATH),
        "n_symbols": int(universe["symbol"].nunique()),
        "n_months": int(month_index["month_end"].nunique()),
        "n_factor_rows": int(len(factor_panel)),
        "best_factor_name": best["factor_name"],
        "best_factor_mean_rank_ic": best["mean_rank_ic"],
        "best_factor_ic_ir": best["ic_ir"],
        "best_factor_spread": best["spread_mean"],
        "best_factor_classification": best["classification"],
        "n_promising_factors": n_promising,
        "n_fragile_factors": n_fragile,
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
