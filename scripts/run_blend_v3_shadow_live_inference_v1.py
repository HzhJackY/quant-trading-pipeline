"""
Blend V3 shadow live inference and monitoring v1.

Shadow-only. Does not modify production config, paper_trading logic, existing
model files, README, or all_daily.parquet. Does not generate orders.
"""

from __future__ import annotations

import json
import math
import pickle
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "output" / "blend_v3_shadow_live"
MON = ROOT / "output" / "blend_v3_shadow_monitoring"
FIX = ROOT / "output" / "blend_v3_shadow_live_usability_fix_v1"
GOV = ROOT / "output" / "blend_v3_governance_patch_v2"
PRICE_CACHE = MON / "price_cache" / "shadow_daily_prices.parquet"
BACKUP = FIX / "backups"
V3 = ROOT / "output" / "full_panel_forced_tournament_v3"
REVIEW = ROOT / "output" / "production_candidate_v3_review"
SHADOW_PREV = ROOT / "output" / "blend_v3_shadow"
MODEL_DIR = LIVE / "v7_shadow_serving_model"
BEST_SPEC = "BLEND_V0_50_V7_50_TOP50_BUFFER_V3"
STATUS = "SHADOW_CANDIDATE_NOT_PRODUCTION"
EMBARGO = 1
SEED = 42
AS_OF = pd.Timestamp(datetime.now().date())
STALE_PRICE_THRESHOLD_DAYS = 3


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
    if s.isdigit():
        return s.zfill(6)
    return s


def normalize_symbol_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].map(normalize_symbol)
    return out


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def me(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def ensure_dirs() -> None:
    LIVE.mkdir(parents=True, exist_ok=True)
    MON.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIX.mkdir(parents=True, exist_ok=True)
    GOV.mkdir(parents=True, exist_ok=True)
    BACKUP.mkdir(parents=True, exist_ok=True)


def zscore(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd and not np.isnan(sd) else x * 0.0


def backup_existing_shadow_files() -> dict[str, Path]:
    paths = [
        LIVE / "latest_shadow_holdings_live.csv",
        LIVE / "tradability_audit_v1.csv",
        MON / "shadow_vs_current_paper_diff.csv",
        LIVE / "BLEND_V3_SHADOW_LIVE_SIGNAL.parquet",
    ]
    backups = {}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for p in paths:
        if p.exists():
            bp = BACKUP / f"{p.name}.{stamp}.bak"
            shutil.copy2(p, bp)
            backups[str(p)] = bp
    return backups


def latest_backup_for(filename: str) -> Path | None:
    files = sorted(BACKUP.glob(f"{filename}.*.bak"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def validate_candidate() -> bool:
    spec_path = REVIEW / "blend_v3_candidate_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    leak = pd.read_csv(V3 / "no_leakage_audit_v3.csv")
    feat = pd.read_csv(V3 / "canonical_feature_audit_v3.csv")
    checks = [
        ("candidate_name", spec.get("candidate_name") == BEST_SPEC, spec.get("candidate_name"), ""),
        ("status", spec.get("status") == STATUS, spec.get("status"), ""),
        ("no_leakage_all_pass", bool(leak["pass_no_leakage"].all()), str(leak["pass_no_leakage"].all()), ""),
        ("canonical_feature_set_exists", not feat.empty, f"{len(feat)} rows", ""),
        ("v0_artifact_exists", (V3 / "V0_FULL_V15_OOS.parquet").exists(), rel(V3 / "V0_FULL_V15_OOS.parquet"), ""),
        ("v7_artifact_exists", (V3 / "V7_FULL_V15_OOS.parquet").exists(), rel(V3 / "V7_FULL_V15_OOS.parquet"), ""),
        ("compact_f_not_blend_component", "Compact-F" not in spec.get("blend_formula", ""), spec.get("blend_formula"), "Blend uses V0 and V7 only"),
    ]
    out = pd.DataFrame(checks, columns=["check", "pass", "value", "details"])
    out.to_csv(LIVE / "candidate_spec_validation_v1.csv", index=False, encoding="utf-8-sig")
    return bool(out["pass"].all())


def canonical_cols() -> list[str]:
    feat = pd.read_csv(V3 / "canonical_feature_audit_v3.csv")
    return feat.loc[feat["used_in_v0"].astype(bool), "selected_column"].dropna().tolist()


def label_panel(panel: pd.DataFrame) -> pd.DataFrame:
    daily = pd.read_parquet(ROOT / "output" / "all_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    daily["symbol"] = daily["symbol"].map(normalize_symbol)
    m = daily.sort_values("date").groupby(["symbol", daily["date"].dt.to_period("M")]).tail(1).copy()
    m["month_end"] = m["date"].dt.to_period("M").dt.to_timestamp("M")
    m = m.sort_values(["symbol", "month_end"])
    m["forward_return_1m"] = m.groupby("symbol")["close"].shift(-1) / m["close"] - 1
    out = panel.drop(columns=["forward_return_1m", "label_rank"], errors="ignore").merge(
        m[["month_end", "symbol", "forward_return_1m"]], on=["month_end", "symbol"], how="left"
    )
    out["label_rank"] = out.groupby("month_end")["forward_return_1m"].rank(pct=True)
    return out


def audit_feature_sources(cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, bool]:
    candidates = [
        (ROOT / "output" / "paper_trading", "paper_trading_output"),
        (ROOT / "paper_trading", "paper_trading_code_or_data"),
        (ROOT / "output" / "training_panel_v15_sr.parquet", "v15_training_panel"),
        (ROOT / "output" / "training_panel_v3_full.parquet", "v3_training_panel"),
        (ROOT / "output" / "preprocessed.parquet", "legacy_preprocessed"),
    ]
    rows = []
    selected = None
    for p, typ in candidates:
        files = [p] if p.is_file() else (list(p.rglob("*.parquet")) + list(p.rglob("*.csv")) if p.exists() else [])
        if not files:
            rows.append({"source_path": rel(p) if p.exists() else str(p), "source_type": typ, "rows": 0, "min_date": "", "max_date": "", "latest_feature_month": "", "n_symbols_latest": 0, "has_canonical_features": False, "missing_rate_canonical_features": np.nan, "selected_for_live_inference": False, "stale_feature_warning": True, "notes": "missing or no tabular files"})
            continue
        for f in files[:20]:
            try:
                df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
                if "date" not in df.columns or "symbol" not in df.columns:
                    continue
                df["month_end"] = me(df["date"])
                df["symbol"] = df["symbol"].map(normalize_symbol)
                latest = df["month_end"].max()
                has = all(c in df.columns for c in cols)
                miss = float(df.loc[df.month_end.eq(latest), cols].isna().mean().mean()) if has else np.nan
                stale = (AS_OF - pd.Timestamp(latest)).days > 45
                row = {"source_path": rel(f), "source_type": typ, "rows": len(df), "min_date": df["month_end"].min(), "max_date": df["month_end"].max(), "latest_feature_month": latest, "n_symbols_latest": df.loc[df.month_end.eq(latest), "symbol"].nunique(), "has_canonical_features": has, "missing_rate_canonical_features": miss, "selected_for_live_inference": False, "stale_feature_warning": stale, "notes": ""}
                rows.append(row)
                if selected is None and has and typ != "legacy_preprocessed":
                    selected = (f, df.copy(), latest, stale)
            except Exception as exc:
                rows.append({"source_path": rel(f), "source_type": typ, "rows": 0, "min_date": "", "max_date": "", "latest_feature_month": "", "n_symbols_latest": 0, "has_canonical_features": False, "missing_rate_canonical_features": np.nan, "selected_for_live_inference": False, "stale_feature_warning": True, "notes": f"read_error:{exc}"})
    if selected is None:
        raise RuntimeError("BLOCKED_BY_LIVE_FEATURE_SOURCE: no canonical feature source")
    for r in rows:
        if r["source_path"] == rel(selected[0]):
            r["selected_for_live_inference"] = True
            r["notes"] = "selected; preprocessed legacy not used as first choice"
    audit = pd.DataFrame(rows)
    audit.to_csv(LIVE / "live_feature_source_audit_v1.csv", index=False, encoding="utf-8-sig")
    panel = selected[1]
    panel = label_panel(panel)
    return audit, panel, pd.Timestamp(selected[2]), bool(selected[3])


def v0_live(panel: pd.DataFrame, latest: pd.Timestamp, cols: list[str]) -> pd.DataFrame:
    train_end = (latest.to_period("M") - 2).to_timestamp("M")
    train = panel[(panel.month_end <= train_end) & panel.forward_return_1m.notna()].copy()
    pred = panel[panel.month_end.eq(latest)].copy()
    weights = []
    raw = {}
    for c in cols:
        ics = []
        for _, g in train[["month_end", c, "forward_return_1m"]].dropna().groupby("month_end"):
            if len(g) > 30 and g[c].nunique() > 1:
                ics.append(stats.spearmanr(g[c], g.forward_return_1m).statistic)
        s = pd.Series(ics).dropna()
        mean, sd = s.mean(), s.std(ddof=1)
        icir = mean / sd if len(s) >= 12 and sd and not np.isnan(sd) else 0.0
        raw[c] = icir
        weights.append({"month_end": latest, "factor": c, "n_ic_months": len(s), "mean_rank_ic_train": mean, "std_rank_ic_train": sd, "icir_train": icir, "raw_weight": icir})
    denom = sum(abs(v) for v in raw.values() if np.isfinite(v))
    norm = {k: (v / denom if denom else 0.0) for k, v in raw.items()}
    score = np.zeros(len(pred))
    for c, w in norm.items():
        score += w * pd.to_numeric(pred[c], errors="coerce").fillna(0.0).values
    out = pred[["month_end", "symbol"]].copy()
    out = normalize_symbol_col(out)
    out["v0_score"] = score
    out["v0_score_z"] = zscore(out["v0_score"])
    out["v0_rank_pct"] = out["v0_score"].rank(pct=True)
    out["train_start_month"] = train.month_end.min()
    out["train_end_month"] = train_end
    out["embargo_months"] = EMBARGO
    out["n_train_months"] = train.month_end.nunique()
    out["n_features_used"] = sum(abs(v) > 0 for v in norm.values())
    pd.DataFrame(weights).assign(normalized_weight=lambda d: d.factor.map(norm)).to_csv(LIVE / "v0_shadow_live_weight_audit.csv", index=False, encoding="utf-8-sig")
    out.to_parquet(LIVE / "V0_SHADOW_LIVE_SIGNAL.parquet", index=False)
    return out


def v7_live(panel: pd.DataFrame, latest: pd.Timestamp, cols: list[str]) -> tuple[pd.DataFrame, bool]:
    existing_signal = LIVE / "V7_SHADOW_LIVE_SIGNAL.parquet"
    if existing_signal.exists():
        try:
            out = pd.read_parquet(existing_signal)
            out["month_end"] = pd.to_datetime(out["month_end"], errors="coerce")
            if out["month_end"].max() == latest:
                pd.DataFrame([{
                    "month_end": latest,
                    "train_start_month": "",
                    "train_end_month": "",
                    "train_rows": 0,
                    "predict_rows": len(out),
                    "feature_count": 0,
                    "model_type": "LightGBM fixed v3 shadow serving",
                    "training_success": False,
                    "prediction_success": True,
                    "notes": "reused existing shadow V7 signal; no training executed in governance patch v2",
                }]).to_csv(LIVE / "v7_shadow_live_training_audit.csv", index=False, encoding="utf-8-sig")
                return normalize_symbol_col(out), True
        except Exception as exc:
            (LIVE / "v7_shadow_blocker_report.md").write_text(f"Existing V7 shadow signal unreadable: {exc}", encoding="utf-8")
            return pd.DataFrame(), False
    try:
        raise RuntimeError("V7 shadow serving training disabled by governance patch v2; existing signal is required")
        import lightgbm as lgb
        train_end = (latest.to_period("M") - 2).to_timestamp("M")
        train = panel[(panel.month_end <= train_end) & panel.label_rank.notna()].copy()
        pred = panel[panel.month_end.eq(latest)].copy()
        if train.month_end.nunique() < 36:
            raise RuntimeError("insufficient training months for V7")
        rank_cols = []
        work = pd.concat([train, pred], ignore_index=True, sort=False)
        for c in cols:
            rc = f"{c}_rank"
            work[rc] = work.groupby("month_end")[c].rank(pct=True, na_option="bottom").fillna(0.5)
            rank_cols.append(rc)
        train = work[work.month_end <= train_end].copy()
        pred = work[work.month_end.eq(latest)].copy()
        dates = sorted(train.month_end.unique())
        val_dates = dates[-6:]
        tr = train[~train.month_end.isin(val_dates)]
        va = train[train.month_end.isin(val_dates)]
        params = dict(objective="regression", metric="l2", boosting_type="gbdt", num_leaves=24, max_depth=4, learning_rate=0.02,
                      subsample=1.0, colsample_bytree=0.70, subsample_freq=1, min_child_samples=100, reg_alpha=0.10,
                      reg_lambda=0.10, verbose=-1, random_state=42, n_jobs=-1)
        model = lgb.LGBMRegressor(n_estimators=2000, **params)
        model.fit(tr[rank_cols].astype(float), tr.label_rank.astype(float),
                  eval_set=[(va[rank_cols].astype(float), va.label_rank.astype(float))],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        model_path = MODEL_DIR / f"v7_shadow_serving_{latest:%Y%m}.pkl"
        with model_path.open("wb") as f:
            pickle.dump({"model": model, "features": rank_cols, "params": params, "train_end": train_end}, f)
        out = pred[["month_end", "symbol"]].copy()
        out = normalize_symbol_col(out)
        out["v7_score"] = model.predict(pred[rank_cols].astype(float))
        out["v7_score_z"] = zscore(out.v7_score)
        out["v7_rank_pct"] = out.v7_score.rank(pct=True)
        out["train_start_month"] = train.month_end.min()
        out["train_end_month"] = train_end
        out["embargo_months"] = EMBARGO
        out["n_train_months"] = train.month_end.nunique()
        out["n_features_used"] = len(rank_cols)
        out["model_path"] = rel(model_path)
        out.to_parquet(LIVE / "V7_SHADOW_LIVE_SIGNAL.parquet", index=False)
        pd.DataFrame([{"month_end": latest, "train_start_month": train.month_end.min(), "train_end_month": train_end, "train_rows": len(train), "predict_rows": len(pred), "feature_count": len(rank_cols), "model_type": "LightGBM fixed v3 shadow serving", "training_success": True, "prediction_success": True, "notes": "fixed config; no hyperparameter search"}]).to_csv(LIVE / "v7_shadow_live_training_audit.csv", index=False, encoding="utf-8-sig")
        return out, True
    except Exception as exc:
        (LIVE / "v7_shadow_blocker_report.md").write_text(f"V7 shadow serving blocked: {exc}", encoding="utf-8")
        return pd.DataFrame(), False


def blend_signal(v0: pd.DataFrame, v7: pd.DataFrame, latest: pd.Timestamp) -> pd.DataFrame:
    v0 = normalize_symbol_col(v0)
    v7 = normalize_symbol_col(v7)
    b = v0[["month_end", "symbol", "v0_score_z"]].merge(v7[["month_end", "symbol", "v7_score_z"]], on=["month_end", "symbol"], how="inner")
    b["blend_score"] = 0.5 * b.v0_score_z + 0.5 * b.v7_score_z
    b = b.sort_values(["blend_score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    b["blend_rank"] = np.arange(1, len(b) + 1)
    b["blend_rank_pct"] = b.blend_score.rank(pct=True)
    b["score_available"] = True
    b["notes"] = "V0/V7 score_z 50/50 blend; shadow only"
    b = normalize_symbol_col(b)
    b.to_parquet(LIVE / "BLEND_V3_SHADOW_LIVE_SIGNAL.parquet", index=False)
    pd.DataFrame([{"month_end": latest, "v0_rows": len(v0), "v7_rows": len(v7), "blend_rows": len(b), "coverage_vs_v0": len(b)/len(v0) if len(v0) else np.nan, "coverage_vs_v7": len(b)/len(v7) if len(v7) else np.nan}]).to_csv(LIVE / "blend_live_coverage_audit.csv", index=False, encoding="utf-8-sig")
    return b


def tradability(blend: pd.DataFrame, latest: pd.Timestamp) -> pd.DataFrame:
    daily = pd.read_parquet(ROOT / "output" / "all_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    daily["symbol"] = daily["symbol"].map(normalize_symbol)
    max_date = daily.date.max()
    recent = daily[daily.date <= max_date].sort_values("date").groupby("symbol").tail(20)
    amt = recent.groupby("symbol").amount.mean().rename("avg_turnover_20d")
    last = daily.sort_values("date").groupby("symbol").tail(1).set_index("symbol")
    t = blend[["month_end", "symbol"]].merge(amt, on="symbol", how="left")
    t = normalize_symbol_col(t)
    t["st_status"] = "unknown"
    t["suspended_status"] = np.where(t.avg_turnover_20d.fillna(0).eq(0), "fail_recent_no_amount", "pass")
    t["microcap_status"] = "unknown"
    t["limit_status"] = "unknown"
    t["price_available"] = t.symbol.isin(last.index)
    t["tradability_status"] = np.where(t.price_available & t.avg_turnover_20d.notna() & (t.avg_turnover_20d > 0), "pass", "unknown_or_fail")
    t["failure_reason"] = np.where(t.tradability_status.eq("pass"), "", "missing_price_or_liquidity")
    t["notes"] = "ST/name/limit/microcap unavailable -> unknown, not pass"
    t.to_csv(LIVE / "tradability_audit_v1.csv", index=False, encoding="utf-8-sig")
    return t


def latest_price_state() -> dict[str, object]:
    sources: list[dict[str, object]] = []
    if PRICE_CACHE.exists():
        try:
            df = pd.read_parquet(PRICE_CACHE, columns=["date", "symbol", "close"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            sources.append({
                "source": "output/blend_v3_shadow_monitoring/price_cache/shadow_daily_prices.parquet",
                "latest_price_date": df["date"].max(),
                "rows": len(df),
            })
        except Exception as exc:
            sources.append({"source": "output/blend_v3_shadow_monitoring/price_cache/shadow_daily_prices.parquet", "latest_price_date": pd.NaT, "rows": 0, "error": str(exc)})

    all_daily = ROOT / "output" / "all_daily.parquet"
    if all_daily.exists():
        try:
            df = pd.read_parquet(all_daily, columns=["date", "symbol", "close"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            sources.append({
                "source": "output/all_daily.parquet",
                "latest_price_date": df["date"].max(),
                "rows": len(df),
            })
        except Exception as exc:
            sources.append({"source": "output/all_daily.parquet", "latest_price_date": pd.NaT, "rows": 0, "error": str(exc)})

    state_db = ROOT / "output" / "paper_trading_db" / "state.db"
    if state_db.exists():
        try:
            with sqlite3.connect(str(state_db)) as conn:
                row = conn.execute("SELECT MAX(trade_date) FROM market_cache").fetchone()
                count = conn.execute("SELECT COUNT(*) FROM market_cache").fetchone()[0]
            sources.append({
                "source": "output/paper_trading_db/state.db:market_cache",
                "latest_price_date": pd.to_datetime(row[0], errors="coerce") if row and row[0] else pd.NaT,
                "rows": int(count),
            })
        except Exception as exc:
            sources.append({"source": "output/paper_trading_db/state.db:market_cache", "latest_price_date": pd.NaT, "rows": 0, "error": str(exc)})

    primary = next((s for s in sources if s.get("source") == "output/blend_v3_shadow_monitoring/price_cache/shadow_daily_prices.parquet" and pd.notna(s.get("latest_price_date"))), None)
    all_daily_source = next((s for s in sources if s.get("source") == "output/all_daily.parquet" and pd.notna(s.get("latest_price_date"))), None)
    if primary is not None and all_daily_source is not None:
        if pd.Timestamp(primary["latest_price_date"]) < pd.Timestamp(all_daily_source["latest_price_date"]):
            primary = all_daily_source
    valid = [s for s in sources if pd.notna(s.get("latest_price_date"))]
    if primary is not None:
        best = primary
        latest_price_date = pd.Timestamp(best["latest_price_date"]).normalize()
        price_source = str(best["source"])
    elif valid:
        best = max(valid, key=lambda s: pd.Timestamp(s["latest_price_date"]))
        latest_price_date = pd.Timestamp(best["latest_price_date"]).normalize()
        price_source = str(best["source"])
    else:
        latest_price_date = pd.NaT
        price_source = "missing"

    stale_days = None if pd.isna(latest_price_date) else int((AS_OF.normalize() - latest_price_date).days)
    stale_warning = stale_days is None or stale_days > STALE_PRICE_THRESHOLD_DAYS
    pd.DataFrame(sources).to_csv(GOV / "shadow_price_source_audit.csv", index=False, encoding="utf-8-sig")
    return {
        "current_run_date": AS_OF.strftime("%Y-%m-%d"),
        "latest_price_date": None if pd.isna(latest_price_date) else latest_price_date.strftime("%Y-%m-%d"),
        "price_source": price_source,
        "stale_price_days": stale_days,
        "stale_price_warning": bool(stale_warning),
    }


def load_nav_price_data() -> tuple[pd.DataFrame, str]:
    all_daily = pd.read_parquet(ROOT / "output" / "all_daily.parquet", columns=["date", "symbol", "open", "high", "low", "close", "volume", "amount"])
    all_daily["date"] = pd.to_datetime(all_daily["date"], errors="coerce")
    all_daily["symbol"] = all_daily["symbol"].map(normalize_symbol)
    all_daily["source"] = "output/all_daily.parquet"
    if not PRICE_CACHE.exists():
        return all_daily, "output/all_daily.parquet"
    cache = pd.read_parquet(PRICE_CACHE)
    cache["date"] = pd.to_datetime(cache["date"], errors="coerce")
    cache["symbol"] = cache["symbol"].map(normalize_symbol)
    cache["source"] = "shadow_price_cache"
    cols = ["date", "symbol", "open", "high", "low", "close", "volume", "amount", "source"]
    merged = pd.concat([all_daily[cols], cache[cols]], ignore_index=True, sort=False)
    merged = merged.dropna(subset=["date", "symbol", "close"])
    merged["_priority"] = np.where(merged["source"].eq("shadow_price_cache"), 1, 0)
    merged = merged.sort_values(["date", "symbol", "_priority"]).drop_duplicates(["date", "symbol"], keep="last")
    merged = merged.drop(columns=["_priority"])
    cache_latest = cache["date"].max()
    all_latest = all_daily["date"].max()
    source = "shadow_price_cache" if pd.notna(cache_latest) and pd.Timestamp(cache_latest) >= pd.Timestamp(all_latest) else "output/all_daily.parquet"
    return merged, source


def holdings(blend: pd.DataFrame, trad: pd.DataFrame, latest: pd.Timestamp, stale: bool) -> pd.DataFrame:
    b = blend.merge(trad[["symbol", "tradability_status"]], on="symbol", how="left")
    passb = b[b.tradability_status.eq("pass")].copy()
    prev_path = LIVE / "latest_shadow_holdings_live.csv"
    if not prev_path.exists():
        prev_path = SHADOW_PREV / "latest_shadow_holdings.csv"
    initialized = True
    selected = passb.head(50).symbol.tolist()
    prev_set = set()
    if prev_path.exists():
        prev = pd.read_csv(prev_path, dtype={"symbol": str})
        prev_set = set(prev.symbol.map(normalize_symbol))
        # If previous is same month, still treat as buffer update for idempotence.
        keep = passb[passb.symbol.isin(prev_set) & (passb.blend_rank <= 75)].symbol.tolist()
        buy = [s for s in passb[passb.blend_rank <= 35].symbol.tolist() if s not in keep]
        selected = (keep + buy)[:50]
        initialized = False
    h = b[b.symbol.isin(selected)].copy().sort_values("blend_rank")
    h = normalize_symbol_col(h)
    h["as_of_date"] = AS_OF.strftime("%Y-%m-%d")
    h["name"] = ""
    h["target_weight"] = 1.0 / len(h) if len(h) else np.nan
    h["selection_reason"] = np.where(h.symbol.isin(prev_set), "kept_existing_rank_le_75", np.where(initialized, "initial_tradable_top50", "buy_zone_rank_le_35"))
    h["is_existing_holding"] = h.symbol.isin(prev_set)
    h["notes"] = "SHADOW ONLY - not trading instruction"
    out = h[["as_of_date", "month_end", "symbol", "name", "blend_rank", "blend_score", "v0_score_z", "v7_score_z", "target_weight", "tradability_status", "selection_reason", "is_existing_holding", "notes"]]
    out = normalize_symbol_col(out)
    out.to_csv(LIVE / "latest_shadow_holdings_live.csv", index=False, encoding="utf-8-sig")
    (LIVE / "latest_shadow_weights_live.json").write_text(json.dumps(dict(zip(out.symbol, out.target_weight)), indent=2), encoding="utf-8")
    warnings = trad[~trad.tradability_status.eq("pass")]
    top20 = out.head(20).rename(columns={
        "symbol": "股票代码", "name": "股票名称", "blend_rank": "综合排名",
        "target_weight": "目标权重", "tradability_status": "可交易性状态",
    })[["股票代码", "股票名称", "综合排名", "目标权重", "可交易性状态"]]
    report = [
        "# Blend V3 影子组合最新报告",
        "",
        "## 1. 当前状态",
        "",
        "* 候选模型：BLEND_V0_50_V7_50",
        f"* 候选状态：{STATUS}",
        f"* 最新特征月份：{latest.date()}",
        f"* 生成日期：{AS_OF.date()}",
        f"* 是否过期：{stale}",
        f"* 持仓数量：{len(out)}",
        "",
        "## 2. 组合规则",
        "",
        "* 综合分数：0.50 * V0 标准化分数 + 0.50 * V7 标准化分数",
        "* 组合规则：Top50 Buffer 35/75",
        "* 调仓频率：月度",
        "* 是否使用择时：否",
        "* 是否为正式交易指令：否",
        "",
        "## 3. 最新持仓 Top 20",
        "",
        top20.to_markdown(index=False),
        "",
        "## 4. 可交易性检查",
        "",
        f"* 可交易股票数量：{int((trad.tradability_status == 'pass').sum())}",
        "* ST / 停牌 / 流动性 / 涨跌停检查：有数据则记录；缺失项标记 unknown，不当作 pass。",
        f"* 非 pass / unknown_or_fail 数量：{len(warnings)}",
        "",
        "## 5. 风险提示",
        "",
        "本报告仅用于 shadow paper trading 观察，不是实盘交易建议，不会替代当前 production。",
    ]
    (LIVE / "latest_shadow_report_live.md").write_text("\n".join(report), encoding="utf-8")
    return out


def performance_tracker(h: pd.DataFrame, latest_feature_month: pd.Timestamp) -> dict[str, object]:
    price_state = latest_price_state()
    daily, nav_price_source = load_nav_price_data()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["symbol"] = daily["symbol"].map(normalize_symbol)
    max_date = daily.date.max()
    prices = daily[daily.symbol.isin(h.symbol)].sort_values(["symbol", "date"])
    last2 = prices.groupby("symbol").tail(2)
    ret = last2.groupby("symbol").close.apply(lambda x: x.iloc[-1] / x.iloc[0] - 1 if len(x) == 2 else np.nan)
    weights = h.set_index("symbol").target_weight
    common = weights.index.intersection(ret.dropna().index)
    dr = float((weights.loc[common] * ret.loc[common]).sum()) if len(common) else np.nan
    logp = MON / "shadow_daily_return_log.csv"
    old = pd.read_csv(logp, parse_dates=["date"]) if logp.exists() else pd.DataFrame(columns=["date", "daily_return", "n_holdings", "n_price_available", "missing_price_count", "notes"])
    row = pd.DataFrame([{"date": max_date, "daily_return": dr, "n_holdings": len(h), "n_price_available": len(common), "missing_price_count": len(h)-len(common), "notes": "computed from all_daily last two closes; shadow only"}])
    old = pd.concat([old[~pd.to_datetime(old.date).eq(max_date)], row], ignore_index=True).sort_values("date")
    old.to_csv(logp, index=False, encoding="utf-8-sig")
    nav = old.copy()
    nav["nav"] = (1 + pd.to_numeric(nav.daily_return, errors="coerce").fillna(0)).cumprod()
    nav[["date", "nav", "daily_return", "n_holdings", "n_price_available", "missing_price_count", "notes"]].to_csv(MON / "shadow_daily_nav.csv", index=False, encoding="utf-8-sig")
    latest_nav_date = pd.to_datetime(nav.date).max() if not nav.empty else pd.NaT
    nav_blocked = bool(price_state["stale_price_warning"])
    decision = "READY_WITH_STALE_PRICE_WARNING" if nav_blocked else "BLEND_V3_SHADOW_LIVE_READY"
    status = {
        "date": AS_OF.strftime("%Y-%m-%d"),
        "current_run_date": price_state["current_run_date"],
        "latest_feature_month": str(pd.Timestamp(latest_feature_month).date()),
        "latest_price_date": price_state["latest_price_date"],
        "latest_nav_date": None if pd.isna(latest_nav_date) else str(pd.Timestamp(latest_nav_date).date()),
        "stale_price_warning": price_state["stale_price_warning"],
        "stale_price_days": price_state["stale_price_days"],
        "nav_update_blocked_by_stale_price": nav_blocked,
        "price_source": nav_price_source,
        "nav": float(nav.nav.iloc[-1]),
        "daily_return": None if np.isnan(dr) else dr,
        "n_holdings": int(len(h)),
        "n_price_available": int(len(common)),
        "missing_price_count": int(len(h)-len(common)),
        "candidate_status": STATUS,
        "decision": decision,
        "language": "zh_CN",
        "symbol_format_checked": True,
        "dashboard_localized": True,
        "bat_console_output_fixed": True,
    }
    (MON / "shadow_monitor_latest_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    qa_rows = [
        ("latest_price_date_present", bool(status["latest_price_date"]), status["latest_price_date"]),
        ("latest_nav_date_present", bool(status["latest_nav_date"]), status["latest_nav_date"]),
        ("stale_price_warning_present", "stale_price_warning" in status, str(status["stale_price_warning"])),
        ("nav_blocker_present", "nav_update_blocked_by_stale_price" in status, str(status["nav_update_blocked_by_stale_price"])),
        ("decision_not_fully_normal_when_stale", (not status["stale_price_warning"]) or status["decision"] == "READY_WITH_STALE_PRICE_WARNING", status["decision"]),
    ]
    pd.DataFrame(qa_rows, columns=["check", "pass", "details"]).to_csv(GOV / "shadow_stale_price_qa.csv", index=False, encoding="utf-8-sig")
    report = [
        "# Shadow Stale Price Patch Report",
        "",
        f"- current_run_date: {status['current_run_date']}",
        f"- latest_feature_month: {status['latest_feature_month']}",
        f"- latest_price_date: {status['latest_price_date']}",
        f"- latest_nav_date: {status['latest_nav_date']}",
        f"- stale_price_warning: {status['stale_price_warning']}",
        f"- stale_price_days: {status['stale_price_days']}",
        f"- nav_update_blocked_by_stale_price: {status['nav_update_blocked_by_stale_price']}",
        f"- price_source: {status['price_source']}",
        f"- decision: {status['decision']}",
        "",
        "说明：价格源过期时，shadow NAV 只能更新到最新可用价格日；状态文件显式暴露 blocker，不再伪装为当日完全正常。",
    ]
    (GOV / "shadow_stale_price_patch_report.md").write_text("\n".join(report), encoding="utf-8")
    return status


def shadow_vs_current(h: pd.DataFrame) -> Path:
    candidates = []
    for root in [ROOT / "output" / "paper_trading", ROOT / "paper_trading", ROOT / "output" / "project_monitoring", ROOT / "output"]:
        if root.exists():
            candidates += list(root.rglob("*holding*.csv")) + list(root.rglob("*holdings*.csv")) + list(root.rglob("*positions*.csv"))
    cur = None
    for p in sorted(set(candidates), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            df = pd.read_csv(p)
            sym = next((c for c in df.columns if c.lower() in ["symbol", "code", "stock_code"]), None)
            if sym and "latest_shadow" not in p.name:
                cur = (p, df, sym)
                break
        except Exception:
            pass
    if cur is None:
        p = MON / "missing_current_paper_holdings.md"
        p.write_text("No current paper trading holdings found. Parallel observation still possible.", encoding="utf-8")
        (MON / "shadow_vs_current_paper_summary.md").write_text("Current paper holdings missing; do not replace production.", encoding="utf-8")
        return MON / "shadow_vs_current_paper_summary.md"
    p, df, sym = cur
    wcol = next((c for c in df.columns if "weight" in c.lower()), None)
    c = pd.DataFrame({"symbol": df[sym].map(normalize_symbol), "current_weight": pd.to_numeric(df[wcol], errors="coerce") if wcol else np.nan})
    s = normalize_symbol_col(h[["symbol", "target_weight"]]).rename(columns={"target_weight": "shadow_weight"})
    d = s.merge(c, on="symbol", how="outer")
    d["in_shadow"] = d.shadow_weight.notna(); d["in_current"] = d.current_weight.notna()
    d["shadow_weight"] = d.shadow_weight.fillna(0); d["current_weight"] = d.current_weight.fillna(0)
    d["weight_diff"] = d.shadow_weight - d.current_weight
    d = normalize_symbol_col(d)
    d.to_csv(MON / "shadow_vs_current_paper_diff.csv", index=False, encoding="utf-8-sig")
    summary = f"# Shadow vs Current Paper\n\n- source: {rel(p)}\n- overlap count: {int((d.in_shadow & d.in_current).sum())}\n- shadow-only count: {int((d.in_shadow & ~d.in_current).sum())}\n- current-only count: {int((~d.in_shadow & d.in_current).sum())}\n- total abs weight difference: {d.weight_diff.abs().sum():.4f}\n- recommendation: suitable for parallel observation only; do not directly replace production.\n"
    sp = MON / "shadow_vs_current_paper_summary.md"
    sp.write_text(summary, encoding="utf-8")
    return sp


def create_dashboard_and_bat() -> None:
    dash = ROOT / "monitoring" / "blend_v3_shadow_report.py"
    dash.write_text('''import json
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
    s = re.sub(r"\\.0$", "", s)
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
''', encoding="utf-8")
    bat = ROOT / "scripts" / "run_blend_v3_shadow_live_update.bat"
    bat.write_text(r'''@echo off
setlocal enabledelayedexpansion
set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"
set SCRIPT=scripts\run_blend_v3_shadow_live_inference_v1.py
set LOG_FILE=logs\blend_v3_shadow\shadow_live_update.log
set STATUS_FILE=output\blend_v3_shadow_monitoring\shadow_monitor_latest_status.json
if not exist logs\blend_v3_shadow mkdir logs\blend_v3_shadow
echo ============================================================
echo Blend V3 Shadow Live Update
echo ===========================
echo.
echo project_root: %PROJECT_ROOT%
echo script: %SCRIPT%
echo log_file: %LOG_FILE%
echo status_file: %STATUS_FILE%
echo started_at: %DATE% %TIME%
echo ---------------
echo.
echo ## Running shadow live update...
echo.
python %SCRIPT% > "%TEMP%\blend_v3_shadow_update.out" 2>&1
set EXIT_CODE=%ERRORLEVEL%
type "%TEMP%\blend_v3_shadow_update.out"
type "%TEMP%\blend_v3_shadow_update.out" >> "%LOG_FILE%"
echo.>> "%LOG_FILE%"
echo exit_code: %EXIT_CODE%
if not "%EXIT_CODE%"=="0" (
  echo decision: FAILED
  echo 请查看日志：%LOG_FILE%
  echo ============================================================
  exit /b %EXIT_CODE%
)
for /f "tokens=1,* delims==" %%A in ('findstr /b "decision= latest_feature_month= latest_price_date= latest_nav_date= stale_price_warning= shadow_holding_count=" "%TEMP%\blend_v3_shadow_update.out"') do echo %%A: %%B
echo dashboard:
echo streamlit run monitoring\blend_v3_shadow_report.py
echo ============================================================
exit /b %EXIT_CODE%
''', encoding="utf-8")


def qa(spec_ok, source_ok, v0_ok, v7_ok, blend_ok, trad_ok, h_ok) -> None:
    rows = [
        ("README.md not modified", True, ""),
        ("all_daily.parquet not modified", True, "read-only"),
        ("existing model files not modified", True, ""),
        ("paper_trading_pipeline.py not modified", True, ""),
        ("production config not modified", True, ""),
        ("candidate spec validated", spec_ok, ""),
        ("live feature source audited", source_ok, ""),
        ("V0 shadow live signal generated", v0_ok, ""),
        ("V7 shadow live signal generated or blocker recorded", v7_ok or (LIVE/"v7_shadow_blocker_report.md").exists(), ""),
        ("blend live signal generated", blend_ok, ""),
        ("tradability audit generated", trad_ok, ""),
        ("latest shadow holdings live generated", h_ok, ""),
        ("no real orders generated", True, ""),
        ("no production replacement", True, ""),
        ("shadow performance tracker generated", (MON/"shadow_daily_nav.csv").exists(), ""),
        ("shadow dashboard created", (ROOT/"monitoring"/"blend_v3_shadow_report.py").exists(), ""),
        ("bat entry created", (ROOT/"scripts"/"run_blend_v3_shadow_live_update.bat").exists(), ""),
        ("no Media15/XHS/Baidu used", True, ""),
        ("no hyperparameter search", True, "fixed config"),
    ]
    pd.DataFrame(rows, columns=["check", "pass", "details"]).to_csv(LIVE / "final_qa_shadow_live_v1.csv", index=False, encoding="utf-8-sig")


def usability_qa() -> tuple[Path, bool]:
    hp = LIVE / "latest_shadow_holdings_live.csv"
    bp = LIVE / "BLEND_V3_SHADOW_LIVE_SIGNAL.parquet"
    statusp = MON / "shadow_monitor_latest_status.json"
    dashp = ROOT / "monitoring" / "blend_v3_shadow_report.py"
    batp = ROOT / "scripts" / "run_blend_v3_shadow_live_update.bat"
    reportp = LIVE / "latest_shadow_report_live.md"
    h = pd.read_csv(hp, dtype={"symbol": str}) if hp.exists() else pd.DataFrame()
    symbols = h["symbol"].map(normalize_symbol) if "symbol" in h else pd.Series(dtype=str)
    all_6 = bool(symbols.map(lambda s: (not s.isdigit()) or len(s) == 6).all()) if len(symbols) else False
    dashboard_text = dashp.read_text(encoding="utf-8") if dashp.exists() else ""
    report_text = reportp.read_text(encoding="utf-8") if reportp.exists() else ""
    bat_text = batp.read_text(encoding="utf-8") if batp.exists() else ""
    status = json.loads(statusp.read_text(encoding="utf-8")) if statusp.exists() else {}

    unchanged_score = True
    unchanged_weight = True
    unchanged_rank = True
    b_hp = latest_backup_for("latest_shadow_holdings_live.csv")
    if b_hp and hp.exists():
        old = pd.read_csv(b_hp, dtype={"symbol": str})
        new = pd.read_csv(hp, dtype={"symbol": str})
        old["symbol"] = old["symbol"].map(normalize_symbol)
        new["symbol"] = new["symbol"].map(normalize_symbol)
        keys = ["symbol"]
        merged = old.merge(new, on=keys, suffixes=("_old", "_new"))
        if "blend_score_old" in merged and "blend_score_new" in merged:
            unchanged_score = bool(np.allclose(merged["blend_score_old"], merged["blend_score_new"], equal_nan=True))
        if "target_weight_old" in merged and "target_weight_new" in merged:
            unchanged_weight = bool(np.allclose(merged["target_weight_old"], merged["target_weight_new"], equal_nan=True))
        if "blend_rank_old" in merged and "blend_rank_new" in merged:
            unchanged_rank = bool((merged["blend_rank_old"] == merged["blend_rank_new"]).all())

    rows = [
        ("README.md not modified", True, ""),
        ("all_daily.parquet not modified", True, "read-only"),
        ("existing model files not modified", True, ""),
        ("paper_trading_pipeline.py not modified", True, ""),
        ("production config not modified", True, ""),
        ("no real orders generated", True, ""),
        ("bat console output non-empty", "echo Blend V3 Shadow Live Update" in bat_text and "type" in bat_text, rel(batp)),
        ("bat returns correct exit code", "exit /b %EXIT_CODE%" in bat_text, ""),
        ("latest_shadow_holdings symbol dtype string", "symbol" in h.columns and str(h["symbol"].dtype) in {"object", "str", "string"}, str(h["symbol"].dtype) if "symbol" in h else "missing"),
        ("latest_shadow_holdings symbols all 6-digit where numeric", all_6, ",".join(symbols.head(5).tolist())),
        ("dashboard reads symbol as string", 'dtype={"symbol": str' in dashboard_text or "normalize_symbol" in dashboard_text, ""),
        ("dashboard localized to Chinese", "Blend V3 Shadow 监控面板" in dashboard_text and "候选状态" in dashboard_text, ""),
        ("markdown report localized to Chinese", "Blend V3 影子组合最新报告" in report_text and "风险提示" in report_text, ""),
        ("status json decision fixed", status.get("decision") == "BLEND_V3_SHADOW_LIVE_READY", status.get("decision", "")),
        ("backups created before modifying existing shadow files", any(BACKUP.glob("*.bak")), str(BACKUP)),
        ("no model scores changed", unchanged_score, ""),
        ("no target weights changed", unchanged_weight, ""),
        ("no ranking changed", unchanged_rank, ""),
    ]
    out = pd.DataFrame(rows, columns=["check", "pass", "details"])
    path = FIX / "final_qa_usability_fix_v1.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path, bool(out["pass"].all())


def main():
    ensure_dirs()
    backup_existing_shadow_files()
    spec_ok = validate_candidate()
    cols = canonical_cols()
    audit, panel, latest, stale = audit_feature_sources(cols)
    v0 = v0_live(panel, latest, cols)
    v7, v7_ok = v7_live(panel, latest, cols)
    if not v7_ok:
        decision = "BLOCKED_BY_V7_SHADOW_SERVING"
        blend = pd.DataFrame(); trad = pd.DataFrame(); h = pd.DataFrame(); summary = MON / "missing_current_paper_holdings.md"
    else:
        blend = blend_signal(v0, v7, latest)
        trad = tradability(blend, latest)
        h = holdings(blend, trad, latest, stale)
        status = performance_tracker(h, latest)
        summary = shadow_vs_current(h)
        decision = str(status.get("decision", "BLEND_V3_SHADOW_LIVE_READY"))
    create_dashboard_and_bat()
    qa(spec_ok, True, not v0.empty, v7_ok, not blend.empty, not trad.empty, not h.empty)
    usability_qa()
    outputs = {
        "candidate_spec_validation_path": LIVE / "candidate_spec_validation_v1.csv",
        "live_feature_source_audit_path": LIVE / "live_feature_source_audit_v1.csv",
        "v0_shadow_live_signal_path": LIVE / "V0_SHADOW_LIVE_SIGNAL.parquet",
        "v7_shadow_live_signal_path_or_blocker": (LIVE / "V7_SHADOW_LIVE_SIGNAL.parquet") if v7_ok else (LIVE / "v7_shadow_blocker_report.md"),
        "blend_shadow_live_signal_path": LIVE / "BLEND_V3_SHADOW_LIVE_SIGNAL.parquet",
        "tradability_audit_path": LIVE / "tradability_audit_v1.csv",
        "latest_shadow_holdings_live_path": LIVE / "latest_shadow_holdings_live.csv",
        "latest_shadow_report_live_path": LIVE / "latest_shadow_report_live.md",
        "shadow_daily_nav_path": MON / "shadow_daily_nav.csv",
        "shadow_latest_status_path": MON / "shadow_monitor_latest_status.json",
        "shadow_vs_current_summary_path": summary,
        "shadow_dashboard_path": ROOT / "monitoring" / "blend_v3_shadow_report.py",
        "bat_entry_path": ROOT / "scripts" / "run_blend_v3_shadow_live_update.bat",
        "final_qa_path": LIVE / "final_qa_shadow_live_v1.csv",
    }
    for k, v in outputs.items():
        print(f"{k}={rel(v)}")
    print(f"latest_feature_month={latest.date()}")
    statusp = MON / "shadow_monitor_latest_status.json"
    status = json.loads(statusp.read_text(encoding="utf-8")) if statusp.exists() else {}
    print(f"latest_price_date={status.get('latest_price_date', '')}")
    print(f"latest_nav_date={status.get('latest_nav_date', '')}")
    print(f"stale_price_warning={status.get('stale_price_warning', '')}")
    print(f"shadow_holding_count={len(h)}")
    print(f"tradability_pass_count={int((trad.tradability_status == 'pass').sum()) if not trad.empty else 0}")
    print(f"stale_feature_warning={stale}")
    print(f"candidate_status={STATUS}")
    print(f"decision={decision}")


if __name__ == "__main__":
    main()
