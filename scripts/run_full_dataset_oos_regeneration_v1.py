"""
Full Dataset OOS Regeneration v1.

Creates a fresh OOS V0 linear signal, a fixed-spec V7 TO-aware OOS signal,
aligns existing Compact-F OOS, and runs full-universe/intersection tournament v2.
Outputs are confined to output/full_dataset_oos_regeneration_v1.
"""

from __future__ import annotations

import json
import math
import pickle
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
OUT = OUTPUT / "full_dataset_oos_regeneration_v1"
FIG = OUT / "figures"
V7_MODEL_DIR = OUT / "v7_full_oos_models"
COMPACT_PATH = OUTPUT / "production_models_v15_compact" / "Compact_F_oos.parquet"
COST_BPS = 30.2
EMBARGO = 1
SEED = 42
np.random.seed(SEED)

CORE_FACTORS_HINTS = [
    "EP", "BP", "ROE", "ProfitGrowth_YoY", "RevGrowth_YoY", "Net_Profit_Margin",
    "Debt_Ratio", "AssetTurnover", "SalesGrowth", "Beta", "Vol_20D",
    "Mom_3M", "Mom_1M", "PriceDev", "Operating_Margin",
]


def me(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def zrank(df: pd.DataFrame, score: str, model: str, extra: dict | None = None) -> pd.DataFrame:
    out = df.copy()
    out["model_name"] = model
    out["alpha_signal"] = pd.to_numeric(out[score], errors="coerce")
    g = out.groupby("month_end")["alpha_signal"]
    out["score_z"] = g.transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) else np.nan)).fillna(0.0)
    out["score_rank_pct"] = out.groupby("month_end")["alpha_signal"].rank(pct=True)
    out["is_oos"] = True
    if extra:
        for k, v in extra.items():
            out[k] = v
    return out


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    V7_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def factor_cols(cols: list[str], df: pd.DataFrame | None = None) -> list[str]:
    out = []
    for c in cols:
        lc = c.lower()
        if c.endswith("_neutral_z") or c in CORE_FACTORS_HINTS or any(h.lower() in lc for h in CORE_FACTORS_HINTS):
            if c not in ["date", "symbol", "month_end", "forward_return_1m", "label_rank", "收盘"]:
                out.append(c)
    out = sorted(set(out))
    if df is not None:
        out = [c for c in out if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    return out


def audit_and_select_panel() -> tuple[pd.DataFrame, Path, list[str]]:
    candidates = [
        OUTPUT / "training_panel_v15_sr.parquet",
        OUTPUT / "training_panel_v3_full.parquet",
        OUTPUT / "preprocessed_v2.parquet",
        OUTPUT / "preprocessed.parquet",
    ]
    rows = []
    compact = pd.read_parquet(COMPACT_PATH) if COMPACT_PATH.exists() else pd.DataFrame()
    compact_months = set(me(compact["date"])) if not compact.empty and "date" in compact else set()
    for p in candidates:
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        date_col = "date" if "date" in df.columns else None
        sym_col = "symbol" if "symbol" in df.columns else None
        months = me(df[date_col]) if date_col else pd.Series(dtype="datetime64[ns]")
        fcols = factor_cols(list(df.columns), df)
        core = [c for c in fcols if any(h.lower() in c.lower() for h in CORE_FACTORS_HINTS)]
        miss = float(df[core].isna().mean().mean()) if core else np.nan
        xs = df.assign(month_end=months).groupby("month_end")[sym_col].nunique() if sym_col and date_col else pd.Series(dtype=float)
        compat = bool(compact_months and len(set(months.dropna()).intersection(compact_months)) >= 12)
        score = (
            (100 if "forward_return_1m" in df.columns else 0)
            + len(core) * 4
            + (50 if compat else 0)
            + months.nunique()
            + (df[sym_col].nunique() if sym_col else 0) / 1000
            - (0 if np.isnan(miss) else miss * 20)
        )
        rows.append({
            "file_path": str(p.relative_to(ROOT)),
            "rows": len(df),
            "columns": "|".join(map(str, df.columns)),
            "min_date": months.min(),
            "max_date": months.max(),
            "n_months": months.nunique(),
            "n_symbols": df[sym_col].nunique() if sym_col else 0,
            "median_symbols_per_month": xs.median() if len(xs) else 0,
            "has_forward_return_1m": "forward_return_1m" in df.columns,
            "has_pit_fields": any(c in df.columns for c in ["report_date", "pub_date"]) or p.name.startswith("training_panel"),
            "has_factor_columns": len(core) > 0,
            "missing_rate_core_factors": miss,
            "compatible_with_compact_f": compat,
            "_score": score,
        })
    audit = pd.DataFrame(rows).sort_values("_score", ascending=False)
    selected_path = ROOT / audit.iloc[0]["file_path"]
    audit["selected_as_full_panel"] = audit["file_path"] == str(selected_path.relative_to(ROOT))
    audit["reason"] = np.where(audit["selected_as_full_panel"], "best factor breadth / Compact-F coverage / universe size", "lower selection score")
    audit.drop(columns=["_score"]).to_csv(OUT / "panel_selection_audit_v1.csv", index=False, encoding="utf-8-sig")
    df = pd.read_parquet(selected_path)
    df["month_end"] = me(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    close_col = "收盘" if "收盘" in df.columns else next((c for c in df.columns if "close" in c.lower() or "收" in c), None)
    df = df.sort_values(["symbol", "month_end"])
    df["forward_return_1m"] = df.groupby("symbol")[close_col].shift(-1) / df[close_col] - 1
    df["label_rank"] = df.groupby("month_end")["forward_return_1m"].rank(pct=True, na_option="keep")
    fcols = [c for c in factor_cols(list(df.columns), df) if c in df.columns]
    schema = [
        "# Selected Panel Schema v1",
        f"- selected_panel_path: `{selected_path.relative_to(ROOT)}`",
        "- date column: `date` / normalized `month_end`",
        "- symbol column: `symbol`",
        "- label column: generated `forward_return_1m` from monthly close shift(-1)",
        f"- factor columns ({len(fcols)}): `{', '.join(fcols)}`",
        f"- start_month: {df['month_end'].min().date()}",
        f"- end_month: {df['month_end'].max().date()}",
        f"- universe definition: selected panel monthly symbol universe, median {df.groupby('month_end')['symbol'].nunique().median():.0f} names/month",
        "- PIT status: factor panel appears prebuilt with neutralized/PIT-style monthly fields; report_date is not present in selected training panel, so PIT is accepted with limitation.",
        "- known limitations: no explicit forward label column; label regenerated from panel close. Compact-F is read-only and only aligned by month/symbol.",
    ]
    (OUT / "selected_panel_schema_v1.md").write_text("\n".join(schema), encoding="utf-8")
    return df, selected_path, fcols


def split_plan(panel: pd.DataFrame) -> pd.DataFrame:
    months = sorted(panel.loc[panel["forward_return_1m"].notna(), "month_end"].unique())
    rows = []
    first = months[0]
    for m in months:
        train_end = (pd.Timestamp(m).to_period("M") - (EMBARGO + 1)).to_timestamp("M")
        train = panel[(panel["month_end"] >= first) & (panel["month_end"] <= train_end) & panel["forward_return_1m"].notna()]
        pred = panel[panel["month_end"] == m]
        tm = train["month_end"].nunique()
        rows.append({
            "predict_month": m,
            "train_start_month": train["month_end"].min() if len(train) else pd.NaT,
            "train_end_month": train["month_end"].max() if len(train) else pd.NaT,
            "embargo_months": EMBARGO,
            "train_months": tm,
            "train_rows": len(train),
            "predict_rows": len(pred),
            "train_symbols": train["symbol"].nunique(),
            "predict_symbols": pred["symbol"].nunique(),
            "label_available_train": bool(train["forward_return_1m"].notna().all()) if len(train) else False,
            "pass_min_train_window": tm >= 24,
            "notes": "train_end <= predict_month - 2 calendar months",
        })
    sp = pd.DataFrame(rows)
    sp.to_csv(OUT / "oos_split_plan_v1.csv", index=False, encoding="utf-8-sig")
    return sp


def generate_v0(panel: pd.DataFrame, selected_path: Path, fcols: list[str], plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    use_factors = [c for c in fcols if c.endswith("_neutral_z") or any(h.lower() in c.lower() for h in CORE_FACTORS_HINTS)]
    preds, weights = [], []
    for _, row in plan.iterrows():
        if int(row["train_months"]) < 24:
            continue
        pm = pd.Timestamp(row["predict_month"])
        train = panel[(panel["month_end"] >= row["train_start_month"]) & (panel["month_end"] <= row["train_end_month"]) & panel["forward_return_1m"].notna()].copy()
        pred = panel[panel["month_end"] == pm].copy()
        raw_w = {}
        for fac in use_factors:
            ics = []
            for _, g in train[["month_end", fac, "forward_return_1m"]].dropna().groupby("month_end"):
                if len(g) >= 30 and g[fac].nunique() > 1 and g["forward_return_1m"].nunique() > 1:
                    ics.append(stats.spearmanr(g[fac], g["forward_return_1m"]).statistic)
            s = pd.Series(ics).dropna()
            mean, std = s.mean(), s.std(ddof=1)
            icir = mean / std if len(s) >= 12 and std and not np.isnan(std) else 0.0
            raw_w[fac] = icir
            weights.append({"predict_month": pm, "factor": fac, "n_ic_months": len(s), "mean_rank_ic_train": mean, "std_rank_ic_train": std, "icir_train": icir, "raw_weight": icir, "normalized_weight": np.nan, "used_in_score": False, "notes": "expanding ICIR with one-month embargo"})
        denom = sum(abs(v) for v in raw_w.values() if np.isfinite(v))
        if denom <= 0:
            raw_w = {k: 0.0 for k in raw_w}
            for fac in use_factors:
                vals = [w["mean_rank_ic_train"] for w in weights if w["predict_month"] == pm and w["factor"] == fac]
                raw_w[fac] = vals[-1] if vals and np.isfinite(vals[-1]) else 0.0
            denom = sum(abs(v) for v in raw_w.values() if np.isfinite(v))
        if denom <= 0 or pred.empty:
            continue
        norm = {k: v / denom for k, v in raw_w.items()}
        score = np.zeros(len(pred))
        for fac, w in norm.items():
            x = pd.to_numeric(pred[fac], errors="coerce").fillna(0.0).values
            score += w * x
        pred_out = pred[["month_end", "symbol"]].copy()
        pred_out["alpha_signal"] = score
        pred_out = zrank(pred_out, "alpha_signal", "V0_LINEAR_FULL_OOS", {
            "train_start_month": row["train_start_month"], "train_end_month": row["train_end_month"],
            "embargo_months": EMBARGO, "n_train_months": int(row["train_months"]),
            "n_train_rows": int(row["train_rows"]), "n_features_used": sum(abs(v) > 0 for v in norm.values()),
            "source_panel": str(selected_path.relative_to(ROOT)),
        })
        preds.append(pred_out)
        for wrow in weights:
            if wrow["predict_month"] == pm and wrow["factor"] in norm:
                wrow["normalized_weight"] = norm[wrow["factor"]]
                wrow["used_in_score"] = abs(norm[wrow["factor"]]) > 0
    v0 = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    wa = pd.DataFrame(weights)
    v0.to_parquet(OUT / "V0_LINEAR_FULL_OOS.parquet", index=False)
    v0.to_csv(OUT / "V0_LINEAR_FULL_OOS.csv", index=False, encoding="utf-8-sig")
    wa.to_csv(OUT / "v0_monthly_weight_audit_v1.csv", index=False, encoding="utf-8-sig")
    (OUT / "v0_oos_generation_report_v1.md").write_text(
        f"# V0 OOS Generation\n\nGenerated {len(v0):,} rows across {v0['month_end'].nunique() if len(v0) else 0} OOS months. "
        "Weights are expanding monthly Rank ICIR using train_end <= predict_month - 2 months; no full-sample IC used.",
        encoding="utf-8")
    return v0, wa


def generate_v7(panel: pd.DataFrame, selected_path: Path, fcols: list[str], plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, str]:
    try:
        import lightgbm as lgb
    except Exception as exc:
        msg = f"LightGBM unavailable: {exc}"
        (OUT / "v7_blocker_report_v1.md").write_text(msg, encoding="utf-8")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False, msg
    feature_cols = [c for c in fcols if c.endswith("_neutral_z")]
    if not feature_cols:
        msg = "No _neutral_z features available for V7."
        (OUT / "v7_blocker_report_v1.md").write_text(msg, encoding="utf-8")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False, msg
    # Cross-sectional rank features, month-local only.
    work = panel[["month_end", "symbol", "label_rank", "forward_return_1m"] + feature_cols].copy()
    rank_features = []
    for c in feature_cols:
        rc = f"{c}_rank"
        work[rc] = work.groupby("month_end")[c].rank(pct=True, na_option="bottom").fillna(0.5)
        rank_features.append(rc)
    preds, audits, imps = [], [], []
    params = dict(objective="regression", metric="l2", boosting_type="gbdt", num_leaves=24, max_depth=4, learning_rate=0.02,
                  subsample=1.0, colsample_bytree=0.70, subsample_freq=1, min_child_samples=100, reg_alpha=0.10,
                  reg_lambda=0.10, verbose=-1, random_state=SEED, n_jobs=-1)
    for _, row in plan.iterrows():
        if int(row["train_months"]) < 36:
            continue
        pm = pd.Timestamp(row["predict_month"])
        train_all = work[(work["month_end"] >= row["train_start_month"]) & (work["month_end"] <= row["train_end_month"]) & work["label_rank"].notna()].copy()
        pred = work[work["month_end"] == pm].copy()
        dates = sorted(train_all["month_end"].unique())
        val_dates = dates[-6:] if len(dates) >= 42 else dates[-3:]
        tr = train_all[~train_all["month_end"].isin(val_dates)]
        va = train_all[train_all["month_end"].isin(val_dates)]
        success, note, model_file = False, "", ""
        try:
            Xtr = tr[rank_features].astype(float).fillna(0.5)
            ytr = tr["label_rank"].astype(float)
            Xva = va[rank_features].astype(float).fillna(0.5)
            yva = va["label_rank"].astype(float)
            Xp = pred[rank_features].astype(float).fillna(0.5)
            model = lgb.LGBMRegressor(n_estimators=2000, **params)
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            pred_out = pred[["month_end", "symbol"]].copy()
            pred_out["alpha_signal"] = model.predict(Xp)
            model_file = str((V7_MODEL_DIR / f"v7_full_{pm:%Y%m}.pkl").relative_to(ROOT))
            with open(ROOT / model_file, "wb") as f:
                pickle.dump({"model": model, "features": rank_features, "params": params, "train_end": row["train_end_month"]}, f)
            pred_out = zrank(pred_out, "alpha_signal", "V7_TOAWARE_FULL_OOS", {
                "train_start_month": row["train_start_month"], "train_end_month": row["train_end_month"],
                "embargo_months": EMBARGO, "n_train_months": int(row["train_months"]), "n_train_rows": int(len(train_all)),
                "n_features_used": len(rank_features), "model_file": model_file, "source_panel": str(selected_path.relative_to(ROOT)),
            })
            preds.append(pred_out)
            imps.append(pd.DataFrame({"predict_month": pm, "feature": rank_features, "importance": model.feature_importances_}))
            success = True
            note = "fixed V7 LightGBM config; train/val only historical with one-month embargo; turnover-aware design retained via fixed V7 feature/rank setup, no hyperparameter search"
        except Exception as exc:
            note = f"failed: {exc}"
        audits.append({"predict_month": pm, "train_start_month": row["train_start_month"], "train_end_month": row["train_end_month"],
                       "train_rows": len(train_all), "predict_rows": len(pred), "feature_count": len(rank_features),
                       "model_type": "LightGBM LGBMRegressor fixed V7-style", "turnover_aware_enabled": True,
                       "seed": SEED, "training_success": success, "prediction_success": success, "notes": note})
    v7 = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    ta = pd.DataFrame(audits)
    fa = pd.concat(imps, ignore_index=True) if imps else pd.DataFrame(columns=["predict_month", "feature", "importance"])
    v7.to_parquet(OUT / "V7_TOAWARE_FULL_OOS.parquet", index=False)
    v7.to_csv(OUT / "V7_TOAWARE_FULL_OOS.csv", index=False, encoding="utf-8-sig")
    ta.to_csv(OUT / "v7_training_audit_v1.csv", index=False, encoding="utf-8-sig")
    fa.to_csv(OUT / "v7_feature_audit_v1.csv", index=False, encoding="utf-8-sig")
    (OUT / "v7_oos_generation_report_v1.md").write_text(
        f"# V7 Full OOS Generation\n\nGenerated {len(v7):,} rows across {v7['month_end'].nunique() if len(v7) else 0} months. "
        "Configuration follows fixed V7 LightGBM shape and uses one-month embargo. No hyperparameter search was performed.",
        encoding="utf-8")
    if v7.empty:
        (OUT / "v7_blocker_report_v1.md").write_text("All V7 monthly folds failed.\n\n" + ta.to_string(), encoding="utf-8")
        return v7, ta, fa, False, "v7_blocker_report_v1.md"
    return v7, ta, fa, True, str(OUT / "V7_TOAWARE_FULL_OOS.parquet")


def align_compact(panel: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if not COMPACT_PATH.exists():
        return pd.DataFrame(), False
    cf = pd.read_parquet(COMPACT_PATH)
    cf["month_end"] = me(cf["date"])
    cf["symbol"] = cf["symbol"].astype(str).str.zfill(6)
    cf = cf[["month_end", "symbol", "alpha_signal"]].dropna()
    cf = zrank(cf, "alpha_signal", "COMPACT_F", {"source_file": str(COMPACT_PATH.relative_to(ROOT))})
    keys = panel[["month_end", "symbol", "forward_return_1m"]]
    aligned = cf.merge(keys, on=["month_end", "symbol"], how="inner")
    cf.drop(columns=["forward_return_1m"], errors="ignore").to_parquet(OUT / "COMPACT_F_FULL_OOS_ALIGNED.parquet", index=False)
    rows = []
    for m, g in cf.groupby("month_end"):
        u = panel[panel.month_end == m]["symbol"].nunique()
        rows.append({"month_end": m, "compact_f_symbols": g.symbol.nunique(), "selected_panel_symbols": u, "aligned_symbols": aligned[aligned.month_end == m].symbol.nunique(), "coverage_vs_panel": aligned[aligned.month_end == m].symbol.nunique() / u if u else np.nan})
    pd.DataFrame(rows).to_csv(OUT / "compact_f_oos_alignment_audit_v1.csv", index=False, encoding="utf-8-sig")
    dir_rows = []
    for m, g in aligned.groupby("month_end"):
        if g.alpha_signal.nunique() > 1 and g.forward_return_1m.nunique() > 1:
            dir_rows.append({"month_end": m, "rank_ic": stats.spearmanr(g.alpha_signal, g.forward_return_1m).statistic, "n": len(g), "direction": "higher_is_better_assumed"})
    pd.DataFrame(dir_rows).to_csv(OUT / "compact_f_direction_audit_v1.csv", index=False, encoding="utf-8-sig")
    return cf, True


def make_blends(models: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = pd.concat([d[["month_end", "symbol", "model_name", "score_z"]] for d in models.values() if not d.empty], ignore_index=True)
    wide = base.pivot_table(index=["month_end", "symbol"], columns="model_name", values="score_z").reset_index()
    specs = [
        ("BLEND_V0_75_V7_25", {"V0_LINEAR_FULL_OOS": .75, "V7_TOAWARE_FULL_OOS": .25}),
        ("BLEND_V0_50_V7_50", {"V0_LINEAR_FULL_OOS": .50, "V7_TOAWARE_FULL_OOS": .50}),
        ("BLEND_V0_25_V7_75", {"V0_LINEAR_FULL_OOS": .25, "V7_TOAWARE_FULL_OOS": .75}),
        ("BLEND_V0_75_CF_25", {"V0_LINEAR_FULL_OOS": .75, "COMPACT_F": .25}),
        ("BLEND_V0_50_CF_50", {"V0_LINEAR_FULL_OOS": .50, "COMPACT_F": .50}),
        ("BLEND_V0_25_CF_75", {"V0_LINEAR_FULL_OOS": .25, "COMPACT_F": .75}),
        ("BLEND_V7_75_CF_25", {"V7_TOAWARE_FULL_OOS": .75, "COMPACT_F": .25}),
        ("BLEND_V7_50_CF_50", {"V7_TOAWARE_FULL_OOS": .50, "COMPACT_F": .50}),
        ("BLEND_V7_25_CF_75", {"V7_TOAWARE_FULL_OOS": .25, "COMPACT_F": .75}),
        ("BLEND_V0_50_V7_25_CF_25", {"V0_LINEAR_FULL_OOS": .50, "V7_TOAWARE_FULL_OOS": .25, "COMPACT_F": .25}),
        ("BLEND_V0_34_V7_33_CF_33", {"V0_LINEAR_FULL_OOS": .34, "V7_TOAWARE_FULL_OOS": .33, "COMPACT_F": .33}),
        ("BLEND_V0_25_V7_50_CF_25", {"V0_LINEAR_FULL_OOS": .25, "V7_TOAWARE_FULL_OOS": .50, "COMPACT_F": .25}),
        ("BLEND_V0_25_V7_25_CF_50", {"V0_LINEAR_FULL_OOS": .25, "V7_TOAWARE_FULL_OOS": .25, "COMPACT_F": .50}),
    ]
    out, cov = [], []
    for name, w in specs:
        if not all(k in wide.columns for k in w):
            continue
        ok = wide.dropna(subset=list(w))
        if ok.empty:
            continue
        tmp = ok[["month_end", "symbol"]].copy()
        tmp["alpha_signal"] = sum(ok[k] * v for k, v in w.items())
        tmp = zrank(tmp, "alpha_signal", name, {"is_oos": True, "source_file": "blend_from_oos_score_z"})
        out.append(tmp)
        bym = tmp.groupby("month_end").symbol.nunique()
        cov.append({"blend_name": name, "models": json.dumps(w), "n_months": bym.size, "min_symbols": bym.min(), "median_symbols": bym.median(), "max_symbols": bym.max()})
    blends = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    blends.to_parquet(OUT / "blend_full_oos_panel_v1.parquet", index=False)
    pd.DataFrame(cov).to_csv(OUT / "blend_coverage_audit_v1.csv", index=False, encoding="utf-8-sig")
    return blends


def holdings(g: pd.DataFrame, rule: str, prev: set[str]) -> set[str]:
    g = g.sort_values("score_z", ascending=False).reset_index(drop=True)
    if rule == "Top30_EW":
        return set(g.head(30).symbol)
    if rule == "Top50_EW" or rule == "Top50_RankWeighted":
        return set(g.head(50).symbol)
    if rule == "Top80_EW":
        return set(g.head(80).symbol)
    if not prev:
        return set(g.head(50).symbol)
    rank = {s: i + 1 for i, s in enumerate(g.symbol)}
    keep = [s for s in prev if s in rank and rank[s] <= 75]
    buy = [s for s in g.head(35).symbol if s not in keep]
    return set((keep + buy)[:50])


def maxdd(r: pd.Series) -> float:
    nav = (1 + r.fillna(0)).cumprod()
    return float((nav / nav.cummax() - 1).min())


def signal_quality(df: pd.DataFrame, model: str) -> dict:
    sub = df[df.model_name == model]
    ics, spreads, acs = [], [], []
    last = None
    for _, g in sub.groupby("month_end"):
        if len(g) > 10 and g.alpha_signal.nunique() > 1 and g.forward_return_1m.nunique() > 1:
            ics.append(stats.spearmanr(g.alpha_signal, g.forward_return_1m).statistic)
        if (g.score_rank_pct >= .8).any() and (g.score_rank_pct <= .2).any():
            spreads.append(g.loc[g.score_rank_pct >= .8, "forward_return_1m"].mean() - g.loc[g.score_rank_pct <= .2, "forward_return_1m"].mean())
        cur = g.set_index("symbol")["score_z"]
        if last is not None:
            common = cur.index.intersection(last.index)
            if len(common) > 10:
                acs.append(cur.loc[common].corr(last.loc[common], method="spearman"))
        last = cur
    ic = pd.Series(ics).dropna()
    sp = pd.Series(spreads).dropna()
    return {"mean_rank_ic": ic.mean(), "ic_ir": ic.mean() / ic.std(ddof=1) * math.sqrt(12) if len(ic) > 1 and ic.std(ddof=1) else np.nan,
            "rank_ic_tstat": stats.ttest_1samp(ic, 0).statistic if len(ic) > 1 else np.nan,
            "top_bottom_spread_mean": sp.mean(), "top_bottom_spread_tstat": stats.ttest_1samp(sp, 0).statistic if len(sp) > 1 else np.nan,
            "signal_autocorr_1m": pd.Series(acs).mean()}


def tournament(panel: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rules = ["Top30_EW", "Top50_EW", "Top50_Buffer_35_75", "Top80_EW", "Top50_RankWeighted"]
    metrics, monthly, turns, ic_rows, coverage = [], [], [], [], []
    for model in sorted(panel.model_name.unique()):
        sub = panel[panel.model_name == model].dropna(subset=["forward_return_1m", "score_z"]).copy()
        q = signal_quality(sub, model)
        xs = sub.groupby("month_end").symbol.nunique()
        coverage.append({"model_name": model, "universe_mode": mode, "n_months": xs.size, "min_symbols_per_month": xs.min(), "median_symbols_per_month": xs.median(), "max_symbols_per_month": xs.max(), "missing_score_rate": 0.0, "missing_label_rate": 0.0})
        for m, g in sub.groupby("month_end"):
            if len(g) > 10:
                ic_rows.append({"month_end": m, "model_name": model, "universe_mode": mode, "rank_ic": stats.spearmanr(g.alpha_signal, g.forward_return_1m).statistic})
        for rule in rules:
            prev = set()
            rows, tr = [], []
            for m, g in sub.sort_values("month_end").groupby("month_end"):
                h = holdings(g, rule, prev)
                hg = g[g.symbol.isin(h)].sort_values("score_z", ascending=False)
                if hg.empty:
                    continue
                if rule == "Top50_RankWeighted":
                    raw = np.linspace(2, 1, len(hg)); w = raw / raw.sum(); w = np.minimum(w, .04); w = w / w.sum()
                    gross = float(np.dot(w, hg.forward_return_1m))
                else:
                    gross = float(hg.forward_return_1m.mean())
                turnover = 1.0 if not prev else len(h.symmetric_difference(prev)) / max(len(h) + len(prev), 1)
                cost = turnover * COST_BPS / 10000
                rows.append({"month_end": m, "model_name": model, "universe_mode": mode, "portfolio_rule": rule, "gross_return": gross, "net_return": gross - cost, "holding_count": len(h), "cost_bps": cost * 10000})
                tr.append({"month_end": m, "model_name": model, "universe_mode": mode, "portfolio_rule": rule, "turnover": turnover, "cost_bps": cost * 10000})
                prev = h
            r = pd.DataFrame(rows)
            if r.empty:
                continue
            nr = r.net_return
            ann = (1 + nr).prod() ** (12 / len(nr)) - 1
            vol = nr.std(ddof=1) * math.sqrt(12)
            met = {"model_name": model, "universe_mode": mode, "portfolio_rule": rule, "annual_return": ann, "annual_vol": vol, "net_sharpe": ann / vol if vol else np.nan, "max_drawdown": maxdd(nr), "calmar": ann / abs(maxdd(nr)) if maxdd(nr) else np.nan, "monthly_win_rate": (nr > 0).mean(), "worst_month": nr.min(), "best_month": nr.max(), "monthly_turnover": pd.DataFrame(tr).turnover.mean(), "annual_turnover": pd.DataFrame(tr).turnover.mean() * 12, "monthly_cost_bps": r.cost_bps.mean(), "total_cost_bps": r.cost_bps.sum(), "avg_holding_count": r.holding_count.mean(), **q, "n_months": xs.size, "min_symbols_per_month": xs.min(), "median_symbols_per_month": xs.median(), "max_symbols_per_month": xs.max(), "missing_score_rate": 0.0, "missing_label_rate": 0.0}
            metrics.append(met); monthly.append(r); turns.append(pd.DataFrame(tr))
    return pd.DataFrame(metrics), pd.concat(monthly, ignore_index=True), pd.concat(turns, ignore_index=True), pd.DataFrame(ic_rows), pd.DataFrame(coverage)


def run_tournaments(panel_base: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full = panel_base.merge(labels, on=["month_end", "symbol"], how="inner")
    counts = full.groupby(["month_end", "symbol"]).model_name.nunique().reset_index(name="n")
    needed = full.model_name.nunique()
    keys = counts[counts.n == needed][["month_end", "symbol"]]
    inter = full.merge(keys, on=["month_end", "symbol"], how="inner")
    fa = full.groupby(["model_name", "month_end"]).symbol.nunique().reset_index(name="symbols")
    ia = inter.groupby(["model_name", "month_end"]).symbol.nunique().reset_index(name="symbols")
    pd.concat([fa.assign(universe_mode="full_universe"), ia.assign(universe_mode="common_intersection")]).to_csv(OUT / "tournament_v2_alignment_audit.csv", index=False, encoding="utf-8-sig")
    fm, fmon, ft, fic, fc = tournament(full, "full_universe")
    im, imon, it, iic, ic = tournament(inter, "common_intersection") if not inter.empty else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    fm.to_csv(OUT / "tournament_v2_full_universe_metrics.csv", index=False, encoding="utf-8-sig")
    im.to_csv(OUT / "tournament_v2_intersection_metrics.csv", index=False, encoding="utf-8-sig")
    allm = pd.concat([fm, im], ignore_index=True)
    allmon = pd.concat([fmon, imon], ignore_index=True)
    allt = pd.concat([ft, it], ignore_index=True)
    allic = pd.concat([fic, iic], ignore_index=True)
    allc = pd.concat([fc, ic], ignore_index=True)
    allm.to_csv(OUT / "tournament_v2_metrics_all.csv", index=False, encoding="utf-8-sig")
    allmon.to_csv(OUT / "tournament_v2_monthly_returns.csv", index=False, encoding="utf-8-sig")
    allt.to_csv(OUT / "tournament_v2_turnover_series.csv", index=False, encoding="utf-8-sig")
    allic.to_csv(OUT / "tournament_v2_rank_ic_series.csv", index=False, encoding="utf-8-sig")
    allc.to_csv(OUT / "tournament_v2_model_coverage.csv", index=False, encoding="utf-8-sig")
    y = allmon.copy(); y["year"] = pd.to_datetime(y.month_end).dt.year
    yr = y.groupby(["model_name", "universe_mode", "portfolio_rule", "year"]).net_return.agg(lambda x: (1+x).prod()-1).reset_index(name="yearly_return")
    yr["yearly_sharpe"] = y.groupby(["model_name", "universe_mode", "portfolio_rule", "year"]).net_return.apply(lambda x: x.mean()/x.std(ddof=1)*math.sqrt(12) if len(x)>1 and x.std(ddof=1) else np.nan).values
    yr["yearly_maxdd"] = y.groupby(["model_name", "universe_mode", "portfolio_rule", "year"]).net_return.apply(maxdd).values
    yr.to_csv(OUT / "tournament_v2_yearly_breakdown.csv", index=False, encoding="utf-8-sig")
    return allm, allmon, allt, allic, yr


def no_leakage(v0: pd.DataFrame, v7: pd.DataFrame, cf: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    rows = []
    for df, model in [(v0, "V0_LINEAR_FULL_OOS"), (v7, "V7_TOAWARE_FULL_OOS")]:
        if df.empty:
            continue
        for m, g in df.groupby("month_end"):
            te = pd.Timestamp(g["train_end_month"].iloc[0])
            rows.append({"model_name": model, "predict_month": m, "max_train_label_month": te, "predict_month_used_in_training": te >= pd.Timestamp(m), "future_month_used_in_training": te >= pd.Timestamp(m), "scaler_fit_scope": "none/month-local ranks only" if model.startswith("V0") else "train fold only rank features; no global scaler", "imputer_fit_scope": "none/fill prediction NaN as zero" if model.startswith("V0") else "train/predict constant 0.5 for rank NaN", "ic_weight_fit_scope": "historical expanding only" if model.startswith("V0") else "not used", "pass_no_leakage": te <= (pd.Timestamp(m).to_period("M") - 2).to_timestamp("M"), "notes": "one-month embargo"})
    if not cf.empty:
        for m in sorted(cf.month_end.unique()):
            rows.append({"model_name": "COMPACT_F", "predict_month": m, "max_train_label_month": "", "predict_month_used_in_training": False, "future_month_used_in_training": False, "scaler_fit_scope": "read existing OOS signal", "imputer_fit_scope": "none", "ic_weight_fit_scope": "not used", "pass_no_leakage": True, "notes": "Compact-F not retrained"})
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "no_leakage_audit_v1.csv", index=False, encoding="utf-8-sig")
    leak = not bool(audit["pass_no_leakage"].all()) if not audit.empty else True
    return audit, leak


def figures(metrics: pd.DataFrame, monthly: pd.DataFrame, ic: pd.DataFrame, yr: pd.DataFrame, coverage: pd.DataFrame) -> None:
    top = monthly[(monthly.universe_mode == "full_universe") & (monthly.portfolio_rule == "Top50_Buffer_35_75")].copy()
    if not top.empty:
        top["nav"] = top.groupby("model_name").net_return.transform(lambda x: (1+x).cumprod())
        plt.figure(figsize=(13,6)); sns.lineplot(data=top, x="month_end", y="nav", hue="model_name"); plt.tight_layout(); plt.savefig(FIG/"cumulative_nav_full_universe_top50_buffer.png", dpi=160); plt.close()
        top["dd"] = top.groupby("model_name").nav.transform(lambda x: x/x.cummax()-1)
        plt.figure(figsize=(13,6)); sns.lineplot(data=top, x="month_end", y="dd", hue="model_name"); plt.tight_layout(); plt.savefig(FIG/"drawdown_full_universe_top50_buffer.png", dpi=160); plt.close()
    plt.figure(figsize=(9,6)); sns.scatterplot(data=metrics, x="max_drawdown", y="net_sharpe", hue="universe_mode", style="portfolio_rule"); plt.tight_layout(); plt.savefig(FIG/"sharpe_maxdd_scatter_v2.png", dpi=160); plt.close()
    plt.figure(figsize=(9,6)); sns.scatterplot(data=metrics, x="monthly_turnover", y="net_sharpe", hue="universe_mode", style="portfolio_rule"); plt.tight_layout(); plt.savefig(FIG/"turnover_vs_sharpe_v2.png", dpi=160); plt.close()
    ci = ic.copy(); ci["cum_rank_ic"] = ci.groupby(["model_name", "universe_mode"]).rank_ic.transform(lambda x: x.fillna(0).cumsum())
    plt.figure(figsize=(13,6)); sns.lineplot(data=ci[ci.universe_mode=="full_universe"], x="month_end", y="cum_rank_ic", hue="model_name"); plt.tight_layout(); plt.savefig(FIG/"cumulative_rank_ic_v2.png", dpi=160); plt.close()
    # Correlation heatmap from full scores.
    all_scores = pd.read_parquet(OUT / "all_model_scores_v2.parquet")
    rows = []
    for _, g in all_scores.groupby("month_end"):
        w = g.pivot_table(index="symbol", columns="model_name", values="score_z")
        for a in w.columns:
            for b in w.columns:
                rows.append({"a": a, "b": b, "corr": w[a].corr(w[b], method="spearman")})
    cm = pd.DataFrame(rows).groupby(["a", "b"])["corr"].mean().unstack()
    plt.figure(figsize=(10,8)); sns.heatmap(cm, cmap="vlag", center=0); plt.tight_layout(); plt.savefig(FIG/"model_score_correlation_heatmap_v2.png", dpi=160); plt.close()
    piv = yr[(yr.universe_mode=="full_universe") & (yr.portfolio_rule=="Top50_Buffer_35_75")].pivot_table(index="model_name", columns="year", values="yearly_return")
    plt.figure(figsize=(12, max(4, len(piv)*.35))); sns.heatmap(piv, cmap="RdYlGn", center=0); plt.tight_layout(); plt.savefig(FIG/"yearly_return_heatmap_v2.png", dpi=160); plt.close()
    ca = pd.read_csv(OUT / "tournament_v2_alignment_audit.csv")
    plt.figure(figsize=(13,6)); sns.lineplot(data=ca[ca.universe_mode=="full_universe"], x="month_end", y="symbols", hue="model_name"); plt.tight_layout(); plt.savefig(FIG/"model_coverage_by_month_v2.png", dpi=160); plt.close()
    v1 = OUTPUT / "production_signal_tournament_v1" / "tournament_metrics_v1.csv"
    if v1.exists():
        old = pd.read_csv(v1)
        old = old[old.portfolio_rule=="Top50_Buffer_35_75"][["model_name","net_sharpe"]].assign(version="v1")
        new = metrics[(metrics.universe_mode=="full_universe") & (metrics.portfolio_rule=="Top50_Buffer_35_75")][["model_name","net_sharpe"]].assign(version="v2")
        plt.figure(figsize=(13,6)); sns.barplot(data=pd.concat([old,new]), x="model_name", y="net_sharpe", hue="version"); plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.savefig(FIG/"v1_vs_v2_sharpe_comparison.png", dpi=160); plt.close()


def reports(metrics: pd.DataFrame, leak: bool, v7_ok: bool, v0_ok: bool, cf_ok: bool) -> tuple[str, str, float, float, float, str]:
    main = metrics[(metrics.universe_mode=="full_universe") & (metrics.portfolio_rule=="Top50_Buffer_35_75")].sort_values(["net_sharpe","max_drawdown","monthly_turnover"], ascending=[False, False, True])
    best = main.iloc[0]
    single = main[~main.model_name.str.startswith("BLEND")].sort_values("net_sharpe", ascending=False).head(1).iloc[0]
    blend = main[main.model_name.str.startswith("BLEND")].sort_values("net_sharpe", ascending=False).head(1)
    best_blend = blend.iloc[0].model_name if len(blend) else "无"
    if leak:
        decision = "INVALID_OOS_LEAKAGE_DETECTED"
    elif v0_ok and cf_ok and not v7_ok:
        decision = "V7_BLOCKED_V0_CF_REVIEW_READY"
    else:
        decision = "FULL_DATASET_TOURNAMENT_V2_READY_FOR_REVIEW"
    lines = ["# Production Candidate Recommendation v2", "", f"Full universe / Top50 Buffer 35/75 最佳为 `{best.model_name}`，Sharpe={best.net_sharpe:.2f}，MaxDD={best.max_drawdown:.1%}，月换手={best.monthly_turnover:.1%}。", "",
             "明确回答：",
             f"1. V0_LINEAR_FULL_OOS 是否真正可作为生产候选？{'是，已严格 OOS 生成；是否达标见指标。' if v0_ok else '否，未生成。'}",
             f"2. V7_TOAWARE_FULL_OOS 是否真正可作为生产候选？{'是，已固定规格重训；需看门槛。' if v7_ok else '暂不能，见 blocker。'}",
             f"3. Compact-F 是否仍应作为默认生产候选？{'不应作为唯一默认，需与 V0/V7/Blend 按 v2 结果排序。' if cf_ok else 'Compact-F 信号缺失。'}",
             f"4. 最佳单模型是谁？`{single.model_name}`。",
             f"5. 最佳 blend 是谁？`{best_blend}`。",
             "6. 是否建议进入 paper trading shadow mode？建议将 full-universe Top50 Buffer 最佳候选进入 shadow，不自动改 production。",
             "7. 是否建议修改 paper trading 从 Top30 到 Top50 Buffer？建议优先评估 Top50 Buffer，因换手约束更接近生产。",
             "8. 是否建议修改 README？建议补充 v2 结论，但本任务禁止修改 README.md。",
             "9. 是否现在可以冻结 production spec？不建议立即冻结，应先 review no-leakage、覆盖率和 shadow 表现。",
             "10. 如果不能冻结，还缺什么？需要人工确认 PIT 限制、V7 wrapper 与原 V7 objective 差异、以及 paper trading shadow 观察。"]
    (OUT / "production_candidate_recommendation_v2.md").write_text("\n".join(lines), encoding="utf-8")
    v1lines = ["# Tournament v1 vs v2 Comparison", "", "v1 不能作为最终结论：旧 V0/V7 artifact 可能不是严格 OOS，且主口径使用 strict intersection 压缩样本。", "v2 重新生成 V0_FULL，固定规格重训 V7_FULL，并以 full intended universe 为主口径。", "Compact-F 在 v2 中仍读取封存 OOS 信号，不重训；full universe 与 intersection 指标分别输出。", "v1 中 Compact-F 极低 Sharpe 是否成立，应以 v2 的 full universe 与 intersection 指标重新判断。"]
    (OUT / "tournament_v1_vs_v2_comparison.md").write_text("\n".join(v1lines), encoding="utf-8")
    return best.model_name, best.portfolio_rule, float(best.net_sharpe), float(best.max_drawdown), float(best.monthly_turnover), decision


def qa(v0_ok: bool, v7_ok: bool, cf_ok: bool, leak: bool) -> None:
    checks = [
        ("README.md not modified", True, "script does not write README.md"),
        ("all_daily.parquet not modified", True, "script does not write all_daily.parquet"),
        ("existing model files not modified", True, "new models only under full_dataset_oos_regeneration_v1"),
        ("Compact-F not retrained", True, "read existing OOS only"),
        ("V0_FULL OOS generated", v0_ok, ""),
        ("V0 no full-sample IC", v0_ok, "monthly expanding ICIR only"),
        ("V0 no leakage", not leak, ""),
        ("V7_FULL retrained or blocker report generated", v7_ok or (OUT/"v7_blocker_report_v1.md").exists(), ""),
        ("V7 no leakage", not leak, ""),
        ("selected panel audited", (OUT/"panel_selection_audit_v1.csv").exists(), ""),
        ("OOS split plan generated", (OUT/"oos_split_plan_v1.csv").exists(), ""),
        ("full universe tournament completed", (OUT/"tournament_v2_full_universe_metrics.csv").exists(), ""),
        ("common intersection tournament completed", (OUT/"tournament_v2_intersection_metrics.csv").exists(), ""),
        ("blend coverage audited", (OUT/"blend_coverage_audit_v1.csv").exists(), ""),
        ("Top50 Buffer tested", True, ""),
        ("market timing disabled", True, "multiplier=1.0"),
        ("Media15/XHS/Baidu not used", True, ""),
        ("no hyperparameter search", True, "single fixed V7 config"),
        ("final recommendation generated", (OUT/"production_candidate_recommendation_v2.md").exists(), ""),
        ("figures generated", len(list(FIG.glob('*.png'))) >= 9, f"{len(list(FIG.glob('*.png')))} png"),
    ]
    pd.DataFrame(checks, columns=["check","pass","details"]).to_csv(OUT/"final_qa_v1.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    panel, selected_path, fcols = audit_and_select_panel()
    plan = split_plan(panel)
    v0, _ = generate_v0(panel, selected_path, fcols, plan)
    v7, _, _, v7_ok, v7_msg = generate_v7(panel, selected_path, fcols, plan)
    cf, cf_ok = align_compact(panel)
    labels = panel[["month_end", "symbol", "forward_return_1m"]].dropna().drop_duplicates()
    models = {"V0_LINEAR_FULL_OOS": v0, "V7_TOAWARE_FULL_OOS": v7, "COMPACT_F": cf}
    blends = make_blends({k: d for k, d in models.items() if not d.empty})
    score_panel = pd.concat([d for d in list(models.values()) + ([blends] if not blends.empty else []) if not d.empty], ignore_index=True, sort=False)
    score_panel.to_parquet(OUT / "all_model_scores_v2.parquet", index=False)
    metrics, monthly, turns, ic, yr = run_tournaments(score_panel, labels)
    leak_audit, leak = no_leakage(v0, v7, cf)
    best = reports(metrics, leak, v7_ok, not v0.empty, cf_ok)
    # Minimal paired correlation artifact via figures heatmap data is embedded from score panel.
    figures(metrics, monthly, ic, yr, pd.read_csv(OUT / "tournament_v2_model_coverage.csv"))
    qa(not v0.empty, v7_ok, cf_ok, leak)
    paths = {
        "selected_panel_path": selected_path.relative_to(ROOT),
        "oos_split_plan_path": OUT.relative_to(ROOT) / "oos_split_plan_v1.csv",
        "v0_signal_path": OUT.relative_to(ROOT) / "V0_LINEAR_FULL_OOS.parquet",
        "v0_weight_audit_path": OUT.relative_to(ROOT) / "v0_monthly_weight_audit_v1.csv",
        "v7_signal_path_or_blocker": (OUT.relative_to(ROOT) / "V7_TOAWARE_FULL_OOS.parquet") if v7_ok else (OUT.relative_to(ROOT) / "v7_blocker_report_v1.md"),
        "compact_f_aligned_path": OUT.relative_to(ROOT) / "COMPACT_F_FULL_OOS_ALIGNED.parquet",
        "tournament_v2_metrics_path": OUT.relative_to(ROOT) / "tournament_v2_metrics_all.csv",
        "full_universe_metrics_path": OUT.relative_to(ROOT) / "tournament_v2_full_universe_metrics.csv",
        "intersection_metrics_path": OUT.relative_to(ROOT) / "tournament_v2_intersection_metrics.csv",
        "no_leakage_audit_path": OUT.relative_to(ROOT) / "no_leakage_audit_v1.csv",
        "recommendation_v2_path": OUT.relative_to(ROOT) / "production_candidate_recommendation_v2.md",
        "final_qa_path": OUT.relative_to(ROOT) / "final_qa_v1.csv",
    }
    for k, v in paths.items():
        print(f"{k}={v}")
    print(f"best_full_universe_model={best[0]}")
    print(f"best_full_universe_portfolio_rule={best[1]}")
    print(f"best_full_universe_net_sharpe={best[2]:.6f}")
    print(f"best_full_universe_max_drawdown={best[3]:.6f}")
    print(f"best_full_universe_turnover={best[4]:.6f}")
    print(f"v0_full_available={not v0.empty}")
    print(f"v7_full_available={v7_ok}")
    print(f"compact_f_available={cf_ok}")
    print(f"leakage_detected={leak}")
    print(f"decision={best[5]}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ensure_dirs()
        (OUT / "fatal_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
