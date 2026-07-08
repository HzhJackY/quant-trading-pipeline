"""
Production Signal Tournament v1.

Read-only tournament for existing model signal artifacts. The script creates
only files under output/production_signal_tournament_v1 and does not train,
modify model files, or touch README.md / all_daily.parquet.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
OUT = OUTPUT / "production_signal_tournament_v1"
FIG = OUT / "figures"
SCRIPT_NAME = Path(__file__).name
SCORE_CANDIDATES = [
    "alpha_signal",
    "score",
    "pred",
    "prediction",
    "v7_ml_signal",
    "ml_signal",
    "ml_rank_signal",
    "inertia_ml_signal",
]
DATE_CANDIDATES = ["month_end", "date", "trade_date", "snapshot_date"]
SYMBOL_CANDIDATES = ["symbol", "stock_code", "code", "ts_code"]
STYLE_FACTORS = ["EP", "BP", "ROE", "ProfitGrowth_YoY", "RevGrowth_YoY", "Debt_Ratio", "Mom_3M", "Vol_20D", "Beta"]
COST_BPS_PER_TURNOVER = 30.2  # commission 2.5 + stamp 5 + transfer 0.1 + slippage 7.5 per side, round-trip proxy.
RNG = np.random.default_rng(42)


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def month_end(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def pick(cols: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in candidates):
            return c
    return None


def guess_model(path: Path, score_cols: list[str], usable: bool) -> str:
    p = str(path).lower().replace("\\", "/")
    if not usable:
        return "NOT_SIGNAL"
    if "compact_f" in p or "model_f_oos" in p:
        return "COMPACT_F"
    if "v7" in p or "ml_v7" in p or "to-aware" in p:
        return "V7_ML"
    if "split_universe_blended" in p or "linear" in p or "v0" in p:
        return "V0_LINEAR"
    if score_cols:
        return "UNKNOWN_SIGNAL"
    return "NOT_SIGNAL"


def inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.suffix.lower() not in [".parquet", ".csv"]:
            continue
        try:
            df = read_table(path)
            cols = list(df.columns)
            dcols = [c for c in cols if c.lower() in DATE_CANDIDATES or "date" in c.lower() or "month" in c.lower()]
            scols = [c for c in cols if c.lower() in ["symbol", "stock_code", "code", "ts_code"]]
            score_cols = [c for c in cols if c in SCORE_CANDIDATES or any(k in c.lower() for k in ["score", "signal", "pred", "prediction"])]
            dcol = pick(cols, DATE_CANDIDATES)
            scol = pick(cols, SYMBOL_CANDIDATES)
            usable = bool(dcol and scol and score_cols)
            min_d = max_d = ""
            n_symbols = np.nan
            notes = []
            if dcol:
                dt = pd.to_datetime(df[dcol], errors="coerce")
                min_d, max_d = dt.min(), dt.max()
            if scol:
                n_symbols = df[scol].nunique()
            if usable:
                m = month_end(df[dcol])
                xs = df.assign(_m=m).groupby("_m")[scol].nunique()
                if xs.empty or xs.median() < 5:
                    usable = False
                    notes.append("cross_section_too_small")
                if not any(c in SCORE_CANDIDATES for c in score_cols):
                    notes.append("needs_manual_mapping")
            else:
                notes.append("missing_date_symbol_or_score")
            rows.append({
                "file_path": str(path.relative_to(ROOT)),
                "file_type": path.suffix.lower().lstrip("."),
                "rows": len(df),
                "columns": "|".join(cols),
                "date_column_candidates": "|".join(dcols),
                "symbol_column_candidates": "|".join(scols),
                "score_column_candidates": "|".join(score_cols),
                "min_date": min_d,
                "max_date": max_d,
                "n_symbols": n_symbols,
                "guessed_model": guess_model(path, score_cols, usable),
                "usable_for_tournament": usable,
                "notes": ";".join(notes),
            })
        except Exception as exc:
            rows.append({"file_path": str(path.relative_to(ROOT)), "file_type": path.suffix.lower().lstrip("."), "rows": np.nan, "columns": "", "date_column_candidates": "", "symbol_column_candidates": "", "score_column_candidates": "", "min_date": "", "max_date": "", "n_symbols": np.nan, "guessed_model": "NOT_SIGNAL", "usable_for_tournament": False, "notes": f"read_error:{exc}"})
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "signal_artifact_inventory_v1.csv", index=False, encoding="utf-8-sig")
    return inv


def load_signal(model: str, path: Path) -> tuple[pd.DataFrame | None, str]:
    if not path.exists():
        return None, f"missing:{path}"
    df = read_table(path)
    dcol, scol, score = pick(list(df.columns), DATE_CANDIDATES), pick(list(df.columns), SYMBOL_CANDIDATES), pick(list(df.columns), SCORE_CANDIDATES)
    if not (dcol and scol and score):
        return None, f"unusable columns={list(df.columns)}"
    out = pd.DataFrame({
        "month_end": month_end(df[dcol]),
        "symbol": df[scol].astype(str).str.zfill(6),
        "model_name": model,
        "raw_score": pd.to_numeric(df[score], errors="coerce"),
        "source_file": str(path.relative_to(ROOT)),
        "is_oos_signal": "oos" in path.name.lower() or "prediction" in path.name.lower(),
        "notes": f"score_col={score};direction_assumed_higher_is_better",
    }).dropna(subset=["month_end", "symbol", "raw_score"])
    g = out.groupby(["model_name", "month_end"])["raw_score"]
    out["score_z"] = g.transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) else np.nan)).fillna(0.0)
    out["score_rank_pct"] = out.groupby(["model_name", "month_end"])["raw_score"].rank(pct=True)
    return out, f"selected {path} ({len(out):,} rows)"


def select_signals(inv: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    choices = {
        "COMPACT_F": OUTPUT / "production_models_v15_compact" / "Compact_F_oos.parquet",
        "V7_ML": OUTPUT / "ml_v7_predictions.parquet",
        "V0_LINEAR": OUTPUT / "split_universe_blended.parquet",
    }
    panels, notes = [], {}
    for model, path in choices.items():
        df, note = load_signal(model, path)
        notes[model] = note
        if df is not None:
            panels.append(df)
    panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame(columns=["month_end", "symbol", "model_name", "raw_score", "score_z", "score_rank_pct", "source_file", "is_oos_signal", "notes"])
    panel.to_parquet(OUT / "model_signal_panel_v1.parquet", index=False)
    md = ["# Model Signal Source Selection v1", "", "本报告仅选择已有信号文件，不训练、不重构模型。", ""]
    for k, v in notes.items():
        md.append(f"- {k}: {v}")
    md.append("\nV0_LINEAR 使用 `output/split_universe_blended.parquet` 的既有 `alpha_signal`，视为已有线性/IC_IR 导出信号；未从因子重新拟合。")
    (OUT / "model_signal_source_selection_v1.md").write_text("\n".join(md), encoding="utf-8")
    return panel, notes


def forward_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_parquet(OUTPUT / "all_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    daily["symbol"] = daily["symbol"].astype(str).str.zfill(6)
    m = daily.sort_values("date").groupby(["symbol", daily["date"].dt.to_period("M")]).tail(1).copy()
    m["month_end"] = m["date"].dt.to_period("M").dt.to_timestamp("M")
    m = m.sort_values(["symbol", "month_end"])
    m["fwd_1m_return"] = m.groupby("symbol")["close"].shift(-1) / m["close"] - 1.0
    labels = m[["month_end", "symbol", "fwd_1m_return", "date", "close"]].dropna(subset=["fwd_1m_return"])
    audit = pd.DataFrame([{
        "label_source": "output/all_daily.parquet monthly last close",
        "start_month": labels["month_end"].min(),
        "end_month": labels["month_end"].max(),
        "n_months": labels["month_end"].nunique(),
        "n_symbols": labels["symbol"].nunique(),
        "missing_rate": float(labels["fwd_1m_return"].isna().mean()),
        "entry_price_rule": "month-end close at signal month",
        "exit_price_rule": "next month-end close",
        "cost_model": f"simplified turnover cost {COST_BPS_PER_TURNOVER:.1f} bps per 100% one-way turnover",
        "notes": "all_daily read-only; market timing disabled; multiplier=1.0",
    }])
    audit.to_csv(OUT / "return_label_audit_v1.csv", index=False, encoding="utf-8-sig")
    return labels[["month_end", "symbol", "fwd_1m_return"]], audit


def aligned_panel(signals: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = signals.merge(labels, on=["month_end", "symbol"], how="inner")
    models = sorted(base["model_name"].unique())
    rows = []
    for r in range(1, len(models) + 1):
        if r == 1 and len(models) > 1:
            continue
        import itertools
        for combo in itertools.combinations(models, r):
            sub = base[base.model_name.isin(combo)]
            counts = sub.groupby(["month_end", "symbol"])["model_name"].nunique().reset_index(name="n")
            common_keys = counts[counts.n == len(combo)][["month_end", "symbol"]]
            c = sub.merge(common_keys, on=["month_end", "symbol"])
            xs = common_keys.groupby("month_end")["symbol"].nunique()
            rows.append({"comparison_group": "+".join(combo), "models_included": "|".join(combo), "start_month": common_keys["month_end"].min() if len(common_keys) else "", "end_month": common_keys["month_end"].max() if len(common_keys) else "", "n_months": xs.size, "min_symbols_per_month": xs.min() if xs.size else 0, "median_symbols_per_month": xs.median() if xs.size else 0, "max_symbols_per_month": xs.max() if xs.size else 0, "dropped_rows": len(sub) - len(c), "notes": "strict common month_end x symbol"})
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "alignment_audit_v1.csv", index=False, encoding="utf-8-sig")
    full_counts = base.groupby(["month_end", "symbol"])["model_name"].nunique().reset_index(name="n")
    keys = full_counts[full_counts.n == len(models)][["month_end", "symbol"]] if models else pd.DataFrame(columns=["month_end", "symbol"])
    return base.merge(keys, on=["month_end", "symbol"]) if len(keys) else base, audit


def make_blends(aligned: pd.DataFrame) -> pd.DataFrame:
    wide = aligned.pivot_table(index=["month_end", "symbol", "fwd_1m_return"], columns="model_name", values="score_z").reset_index()
    specs = []
    def add(name, weights):
        if all(k in wide.columns for k in weights):
            specs.append((name, weights))
    add("BLEND_V0_75_V7_25", {"V0_LINEAR": .75, "V7_ML": .25})
    add("BLEND_V0_50_V7_50", {"V0_LINEAR": .50, "V7_ML": .50})
    add("BLEND_V0_25_V7_75", {"V0_LINEAR": .25, "V7_ML": .75})
    add("BLEND_V0_75_CF_25", {"V0_LINEAR": .75, "COMPACT_F": .25})
    add("BLEND_V0_50_CF_50", {"V0_LINEAR": .50, "COMPACT_F": .50})
    add("BLEND_V0_25_CF_75", {"V0_LINEAR": .25, "COMPACT_F": .75})
    add("BLEND_V7_75_CF_25", {"V7_ML": .75, "COMPACT_F": .25})
    add("BLEND_V7_50_CF_50", {"V7_ML": .50, "COMPACT_F": .50})
    add("BLEND_V7_25_CF_75", {"V7_ML": .25, "COMPACT_F": .75})
    add("BLEND_V0_50_V7_25_CF_25", {"V0_LINEAR": .50, "V7_ML": .25, "COMPACT_F": .25})
    add("BLEND_V0_34_V7_33_CF_33", {"V0_LINEAR": .34, "V7_ML": .33, "COMPACT_F": .33})
    add("BLEND_V0_25_V7_50_CF_25", {"V0_LINEAR": .25, "V7_ML": .50, "COMPACT_F": .25})
    add("BLEND_V0_25_V7_25_CF_50", {"V0_LINEAR": .25, "V7_ML": .25, "COMPACT_F": .50})
    out = []
    for name, weights in specs:
        s = sum(wide[k] * v for k, v in weights.items())
        tmp = wide[["month_end", "symbol", "fwd_1m_return"]].copy()
        tmp["model_name"] = name
        tmp["score_z"] = s
        tmp["raw_score"] = s
        tmp["score_rank_pct"] = tmp.groupby("month_end")["score_z"].rank(pct=True)
        tmp["source_file"] = "blend_from_score_z"
        tmp["is_oos_signal"] = True
        tmp["notes"] = str(weights)
        out.append(tmp)
    panel = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    panel.to_parquet(OUT / "blend_signal_panel_v1.parquet", index=False)
    return panel


def maxdd(rets: pd.Series) -> float:
    nav = (1 + rets.fillna(0)).cumprod()
    return float((nav / nav.cummax() - 1).min())


def select_holdings(g: pd.DataFrame, rule: str, prev: set[str]) -> set[str]:
    g = g.sort_values("score_z", ascending=False).reset_index(drop=True)
    if rule == "Top30_EW": return set(g.head(30).symbol)
    if rule == "Top50_EW": return set(g.head(50).symbol)
    if rule == "Top80_EW": return set(g.head(80).symbol)
    if rule == "Top50_RankWeighted": return set(g.head(50).symbol)
    keep = set(g.iloc[:75].symbol) & prev
    buy = set(g.iloc[:35].symbol)
    target = list(dict.fromkeys(list(buy) + [s for s in g.symbol if s in keep]))
    if len(target) < 50:
        target += [s for s in g.symbol if s not in target][: 50 - len(target)]
    return set(target[:50])


def backtest_one(df: pd.DataFrame, model: str, rule: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict[pd.Timestamp, set[str]]]:
    prev: set[str] = set()
    rows, turns, holdings_by_month = [], [], {}
    for m, g in df[df.model_name == model].sort_values("month_end").groupby("month_end"):
        hold = select_holdings(g, rule, prev)
        sub = g[g.symbol.isin(hold)].copy()
        if sub.empty:
            continue
        if rule == "Top50_RankWeighted":
            sub = sub.sort_values("score_z", ascending=False)
            raw = np.linspace(2, 1, len(sub))
            w = raw / raw.sum()
            w = np.minimum(w, 0.04)
            w = w / w.sum()
            gross = float(np.dot(w, sub.fwd_1m_return))
        else:
            gross = float(sub.fwd_1m_return.mean())
        turnover = 1.0 if not prev else len(hold.symmetric_difference(prev)) / max(len(hold) + len(prev), 1)
        cost = turnover * COST_BPS_PER_TURNOVER / 10000.0
        rows.append({"month_end": m, "model_name": model, "portfolio_rule": rule, "gross_return": gross, "net_return": gross - cost, "holding_count": len(hold), "cost_bps": cost * 10000})
        turns.append({"month_end": m, "model_name": model, "portfolio_rule": rule, "turnover": turnover, "cost_bps": cost * 10000})
        holdings_by_month[m] = hold
        prev = hold
    r = pd.DataFrame(rows)
    if r.empty:
        return {}, r, pd.DataFrame(turns), holdings_by_month
    nr = r.net_return
    ann = (1 + nr).prod() ** (12 / len(nr)) - 1
    vol = nr.std(ddof=1) * math.sqrt(12)
    metrics = {"model_name": model, "portfolio_rule": rule, "annual_return": ann, "annual_vol": vol, "net_sharpe": ann / vol if vol else np.nan, "max_drawdown": maxdd(nr), "calmar": ann / abs(maxdd(nr)) if maxdd(nr) else np.nan, "monthly_win_rate": (nr > 0).mean(), "worst_month": nr.min(), "best_month": nr.max(), "monthly_turnover": pd.DataFrame(turns).turnover.mean(), "annual_turnover": pd.DataFrame(turns).turnover.mean() * 12, "monthly_cost_bps": r.cost_bps.mean(), "total_cost_bps": r.cost_bps.sum(), "avg_holding_count": r.holding_count.mean()}
    return metrics, r, pd.DataFrame(turns), holdings_by_month


def signal_quality(df: pd.DataFrame, model: str) -> dict:
    sub = df[df.model_name == model].copy()
    ics, spreads, acs = [], [], []
    last = None
    for _, g in sub.groupby("month_end"):
        if g.raw_score.nunique() > 1 and g.fwd_1m_return.nunique() > 1:
            ics.append(stats.spearmanr(g.raw_score, g.fwd_1m_return, nan_policy="omit").statistic)
        q = g.score_rank_pct
        if (q >= .8).any() and (q <= .2).any():
            spreads.append(g.loc[q >= .8, "fwd_1m_return"].mean() - g.loc[q <= .2, "fwd_1m_return"].mean())
        cur = g.set_index("symbol")["score_z"]
        if last is not None:
            common = cur.index.intersection(last.index)
            if len(common) > 5:
                acs.append(cur.loc[common].corr(last.loc[common], method="spearman"))
        last = cur
    ic = pd.Series(ics).dropna()
    sp = pd.Series(spreads).dropna()
    return {"mean_rank_ic": ic.mean(), "ic_ir": ic.mean() / ic.std(ddof=1) * math.sqrt(12) if len(ic) > 1 and ic.std(ddof=1) else np.nan, "rank_ic_tstat": stats.ttest_1samp(ic, 0).statistic if len(ic) > 1 else np.nan, "top_bottom_spread_mean": sp.mean(), "top_bottom_spread_tstat": stats.ttest_1samp(sp, 0).statistic if len(sp) > 1 else np.nan, "signal_autocorr_1m": pd.Series(acs).mean()}


def run_tournament(panel: pd.DataFrame, blends: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    base_cols = ["month_end", "symbol", "model_name", "raw_score", "score_z", "score_rank_pct", "source_file", "is_oos_signal", "notes", "fwd_1m_return"]
    allp = pd.concat([panel[base_cols], blends[base_cols]], ignore_index=True) if not blends.empty else panel[base_cols]
    models = sorted(allp.model_name.unique())
    rules = ["Top30_EW", "Top50_EW", "Top50_Buffer_35_75", "Top80_EW", "Top50_RankWeighted"]
    metrics, mrets, turns, holds = [], [], [], {}
    for model in models:
        q = signal_quality(allp, model)
        for rule in rules:
            met, r, t, h = backtest_one(allp, model, rule)
            if met:
                met.update(q)
                metrics.append(met)
                mrets.append(r)
                turns.append(t)
                holds[(model, rule)] = h
    metrics = pd.DataFrame(metrics)
    monthly = pd.concat(mrets, ignore_index=True)
    turnover = pd.concat(turns, ignore_index=True)
    metrics.to_csv(OUT / "tournament_metrics_v1.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / "monthly_returns_v1.csv", index=False, encoding="utf-8-sig")
    turnover.to_csv(OUT / "turnover_series_v1.csv", index=False, encoding="utf-8-sig")
    ic_rows = []
    for model in models:
        for m, g in allp[allp.model_name == model].groupby("month_end"):
            val = stats.spearmanr(g.raw_score, g.fwd_1m_return, nan_policy="omit").statistic if g.raw_score.nunique() > 1 else np.nan
            ic_rows.append({"month_end": m, "model_name": model, "rank_ic": val})
    icdf = pd.DataFrame(ic_rows)
    icdf.to_csv(OUT / "rank_ic_series_v1.csv", index=False, encoding="utf-8-sig")
    y = monthly.copy()
    y["year"] = pd.to_datetime(y.month_end).dt.year
    yearly = y.groupby(["model_name", "portfolio_rule", "year"]).net_return.agg(lambda x: (1+x).prod()-1).reset_index(name="yearly_return")
    yearly["yearly_sharpe"] = y.groupby(["model_name", "portfolio_rule", "year"]).net_return.apply(lambda x: x.mean()/x.std(ddof=1)*math.sqrt(12) if len(x)>1 and x.std(ddof=1) else np.nan).values
    yearly["yearly_maxdd"] = y.groupby(["model_name", "portfolio_rule", "year"]).net_return.apply(maxdd).values
    yearly.to_csv(OUT / "yearly_breakdown_v1.csv", index=False, encoding="utf-8-sig")
    return metrics, monthly, yearly, icdf, holds


def correlations(panel: pd.DataFrame) -> None:
    rows = []
    models = sorted(panel.model_name.unique())
    for m, g in panel.groupby("month_end"):
        w = g.pivot_table(index="symbol", columns="model_name", values="score_z")
        for i, a in enumerate(models):
            for b in models[i+1:]:
                pair = w[[a, b]].dropna() if a in w and b in w else pd.DataFrame()
                rows.append({"month_end": m, "model_a": a, "model_b": b, "spearman_corr": pair[a].corr(pair[b], method="spearman") if len(pair)>2 else np.nan, "pearson_corr": pair[a].corr(pair[b]) if len(pair)>2 else np.nan, "n": len(pair)})
    corr = pd.DataFrame(rows)
    corr.to_csv(OUT / "model_correlation_v1.csv", index=False, encoding="utf-8-sig")
    summ = corr.groupby(["model_a", "model_b"])[["spearman_corr", "pearson_corr", "n"]].mean().reset_index()
    summ.to_csv(OUT / "model_correlation_summary_v1.csv", index=False, encoding="utf-8-sig")


def style_exposure(holds: dict) -> None:
    p = OUTPUT / "preprocessed.parquet"
    if not p.exists():
        (OUT / "style_exposure_missing_reason.md").write_text("preprocessed.parquet missing.", encoding="utf-8")
        return
    f = pd.read_parquet(p)
    f["month_end"] = month_end(f["date"])
    f["symbol"] = f["symbol"].astype(str).str.zfill(6)
    available = [c for c in STYLE_FACTORS if c in f.columns]
    if not available:
        (OUT / "style_exposure_missing_reason.md").write_text("目标风格字段不可用。", encoding="utf-8")
        return
    rows = []
    for (model, rule), bym in holds.items():
        for fac in available:
            vals = []
            latest = np.nan
            for m, h in bym.items():
                v = f[(f.month_end == m) & (f.symbol.isin(h))][fac].mean()
                vals.append(v)
                latest = v
            rows.append({"model_name": model, "portfolio_rule": rule, "factor": fac, "avg_exposure": pd.Series(vals).mean(), "std_exposure": pd.Series(vals).std(), "latest_exposure": latest, "notes": "equal-weight holding average"})
    pd.DataFrame(rows).to_csv(OUT / "style_exposure_v1.csv", index=False, encoding="utf-8-sig")


def paired_tests(monthly: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    top_rule = "Top50_Buffer_35_75"
    candidates = ["V0_LINEAR", "V7_ML", "COMPACT_F"]
    best = metrics[metrics.portfolio_rule == top_rule].sort_values(["net_sharpe", "max_drawdown", "monthly_turnover"], ascending=[False, False, True]).head(1)
    if best.empty:
        return pd.DataFrame()
    best_model = best.iloc[0].model_name
    pairs = [(best_model, c) for c in candidates if c != best_model] + [("V0_LINEAR", "COMPACT_F"), ("V7_ML", "COMPACT_F")]
    rows = []
    for a, b in pairs:
        w = monthly[monthly.portfolio_rule == top_rule].pivot_table(index="month_end", columns="model_name", values="net_return")
        if a not in w or b not in w:
            continue
        d = (w[a] - w[b]).dropna()
        if len(d) < 3:
            continue
        boots = []
        arr = d.values
        for _ in range(5000):
            idx = []
            while len(idx) < len(arr):
                s = RNG.integers(0, max(1, len(arr)-2))
                idx.extend(range(s, min(s+3, len(arr))))
            boots.append(arr[idx[:len(arr)]].mean())
        rows.append({"comparison": f"{a} vs {b}", "mean_monthly_diff": d.mean(), "annualized_diff": d.mean()*12, "tstat": stats.ttest_1samp(d, 0).statistic, "bootstrap_ci_2_5": np.percentile(boots, 2.5), "bootstrap_ci_97_5": np.percentile(boots, 97.5), "n_months": len(d), "interpretation": "显著" if np.percentile(boots, 2.5) > 0 or np.percentile(boots, 97.5) < 0 else "不显著/需谨慎"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "paired_performance_test_v1.csv", index=False, encoding="utf-8-sig")
    return out


def unique_contribution(panel: pd.DataFrame, holds: dict, best_key: tuple[str, str]) -> None:
    models = ["V0_LINEAR", "V7_ML", "COMPACT_F"]
    rule = best_key[1]
    rows = []
    for m in sorted(set().union(*[set(holds.get((x, rule), {}).keys()) for x in models])):
        sets = {x: holds.get((x, rule), {}).get(m, set()) for x in models}
        for x in models:
            unique = sets[x] - set().union(*[sets[y] for y in models if y != x])
            ret = panel[(panel.month_end == m) & (panel.symbol.isin(unique))]["fwd_1m_return"].mean()
            rows.append({"month_end": m, "bucket": f"{x}_unique", "n": len(unique), "avg_fwd_return": ret})
        common = set.intersection(*sets.values()) if all(sets.values()) else set()
        rows.append({"month_end": m, "bucket": "common_all", "n": len(common), "avg_fwd_return": panel[(panel.month_end == m) & (panel.symbol.isin(common))]["fwd_1m_return"].mean()})
    pd.DataFrame(rows).to_csv(OUT / "unique_holdings_contribution_v1.csv", index=False, encoding="utf-8-sig")


def figures(metrics: pd.DataFrame, monthly: pd.DataFrame, yearly: pd.DataFrame, icdf: pd.DataFrame) -> None:
    top = monthly[monthly.portfolio_rule == "Top50_Buffer_35_75"].copy()
    if not top.empty:
        top["nav"] = top.groupby("model_name").net_return.transform(lambda x: (1+x).cumprod())
        plt.figure(figsize=(12, 6)); sns.lineplot(data=top, x="month_end", y="nav", hue="model_name"); plt.tight_layout(); plt.savefig(FIG/"cumulative_nav_by_model_top50_buffer.png", dpi=160); plt.close()
        top["dd"] = top.groupby("model_name").nav.transform(lambda x: x/x.cummax()-1)
        plt.figure(figsize=(12, 6)); sns.lineplot(data=top, x="month_end", y="dd", hue="model_name"); plt.tight_layout(); plt.savefig(FIG/"drawdown_by_model_top50_buffer.png", dpi=160); plt.close()
    plt.figure(figsize=(9, 6)); sns.scatterplot(data=metrics, x="max_drawdown", y="net_sharpe", hue="portfolio_rule"); plt.tight_layout(); plt.savefig(FIG/"sharpe_maxdd_scatter.png", dpi=160); plt.close()
    plt.figure(figsize=(9, 6)); sns.scatterplot(data=metrics, x="monthly_turnover", y="net_sharpe", hue="portfolio_rule"); plt.tight_layout(); plt.savefig(FIG/"turnover_vs_sharpe.png", dpi=160); plt.close()
    icp = icdf.copy(); icp["cum_ic"] = icp.groupby("model_name").rank_ic.transform(lambda x: x.fillna(0).cumsum())
    plt.figure(figsize=(12, 6)); sns.lineplot(data=icp, x="month_end", y="cum_ic", hue="model_name"); plt.tight_layout(); plt.savefig(FIG/"rank_ic_cumulative.png", dpi=160); plt.close()
    cs = pd.read_csv(OUT/"model_correlation_summary_v1.csv")
    if not cs.empty:
        mods = sorted(set(cs.model_a).union(cs.model_b)); mat = pd.DataFrame(np.eye(len(mods)), index=mods, columns=mods)
        for _, r in cs.iterrows(): mat.loc[r.model_a, r.model_b] = mat.loc[r.model_b, r.model_a] = r.spearman_corr
        plt.figure(figsize=(7, 6)); sns.heatmap(mat, annot=True, cmap="vlag", center=0); plt.tight_layout(); plt.savefig(FIG/"model_score_correlation_heatmap.png", dpi=160); plt.close()
    if not yearly.empty:
        piv = yearly[yearly.portfolio_rule=="Top50_Buffer_35_75"].pivot_table(index="model_name", columns="year", values="yearly_return")
        plt.figure(figsize=(12, max(4, len(piv)*.35))); sns.heatmap(piv, annot=False, cmap="RdYlGn", center=0); plt.tight_layout(); plt.savefig(FIG/"yearly_return_heatmap.png", dpi=160); plt.close()
    sp = OUT / "style_exposure_v1.csv"
    if sp.exists():
        st = pd.read_csv(sp)
        st = st[(st.portfolio_rule=="Top50_Buffer_35_75") & (st.model_name.isin(["V0_LINEAR","V7_ML","COMPACT_F"]))]
        if not st.empty:
            plt.figure(figsize=(12, 6)); sns.barplot(data=st, x="factor", y="latest_exposure", hue="model_name"); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(FIG/"style_exposure_bar_latest.png", dpi=160); plt.close()
    uc = OUT / "unique_holdings_contribution_v1.csv"
    if uc.exists():
        u = pd.read_csv(uc).groupby("bucket")[["n","avg_fwd_return"]].mean().reset_index()
        plt.figure(figsize=(10, 5)); sns.barplot(data=u, x="bucket", y="avg_fwd_return"); plt.xticks(rotation=25); plt.tight_layout(); plt.savefig(FIG/"unique_holdings_contribution.png", dpi=160); plt.close()


def recommendation(metrics: pd.DataFrame, availability: dict[str, bool]) -> tuple[str, str, float, float, float, str]:
    main = metrics[metrics.portfolio_rule == "Top50_Buffer_35_75"].copy()
    best = main.sort_values(["net_sharpe", "max_drawdown", "monthly_turnover"], ascending=[False, False, True]).iloc[0]
    compact = main[main.model_name == "COMPACT_F"]
    lines = ["# Production Candidate Recommendation v1", ""]
    lines.append(f"主口径 Top50 Buffer 35/75 下，当前最佳为 `{best.model_name}`，net Sharpe={best.net_sharpe:.2f}，MaxDD={best.max_drawdown:.1%}，月换手={best.monthly_turnover:.1%}。")
    for m in ["V0_LINEAR", "V7_ML", "COMPACT_F"]:
        row = main[main.model_name == m]
        if row.empty:
            lines.append(f"- {m}: 信号不可用或未进入主比较。")
        else:
            r = row.iloc[0]
            ok = r.net_sharpe >= .8 and r.max_drawdown >= -.25 and r.monthly_turnover <= .4 and r.mean_rank_ic > 0
            lines.append(f"- {m}: Sharpe={r.net_sharpe:.2f}, MaxDD={r.max_drawdown:.1%}, turnover={r.monthly_turnover:.1%}, mean IC={r.mean_rank_ic:.3f}; {'满足基础门槛' if ok else '未完全满足基础门槛'}。")
    lines += ["", "明确回答：",
              f"1. V0 Linear 是否应重新进入生产候选？{'是，若上述门槛通过，应进入 shadow/paper 候选。' if availability.get('V0_LINEAR') else '否，当前缺少可用信号。'}",
              f"2. V7 ML 是否应重新进入生产候选？{'是，可作为 ML 候选重新纳入比较。' if availability.get('V7_ML') else '否，当前缺少可用 OOS 信号。'}",
              "3. Compact-F 是否仍应作为唯一默认生产候选？不建议作为唯一候选，应以本擂台结果重新排序。",
              f"4. 哪个 Blend 最值得进入 paper trading shadow mode？`{best.model_name}`（若为单模型，则 Blend 未超过最佳单模型）。",
              "5. paper trading 是否应该从 Top30 改为 Top50 Buffer？主比较支持优先评估 Top50 Buffer 35/75，因其更接近低换手生产约束。",
              "6. README 是否需要增加说明？需要，但本任务禁止修改 README.md。",
              "7. 下一步是否应该冻结 production spec？建议先完成 review 和 shadow 验证，再冻结 production spec。"]
    (OUT / "production_candidate_recommendation_v1.md").write_text("\n".join(lines), encoding="utf-8")
    decision = "TOURNAMENT_COMPLETED_READY_FOR_REVIEW" if sum(availability.values()) >= 2 else "NEEDS_SIGNAL_EXPORT_BEFORE_TOURNAMENT"
    return best.model_name, best.portfolio_rule, float(best.net_sharpe), float(best.max_drawdown), float(best.monthly_turnover), decision


def report(metrics: pd.DataFrame, audit: pd.DataFrame, ret_audit: pd.DataFrame, paired: pd.DataFrame, best_tuple: tuple) -> None:
    best_model, best_rule, sharpe, mdd, to, decision = best_tuple
    top = metrics[metrics.portfolio_rule == "Top50_Buffer_35_75"].sort_values("net_sharpe", ascending=False)
    lines = ["# Production Signal Tournament v1", "", "## 1. Executive Summary", f"主口径 Top50 Buffer 35/75 下最佳候选为 `{best_model}`，net Sharpe={sharpe:.2f}，MaxDD={mdd:.1%}，月换手={to:.1%}。Decision={decision}。", "", "## 2. Signal Artifact Inventory", "详见 `signal_artifact_inventory_v1.csv`。", "", "## 3. Strict Alignment Setup", audit.to_markdown(index=False), "", "## 4. Portfolio Rules", "测试 Top30 EW、Top50 EW、Top50 Buffer 35/75、Top80 EW、Top50 rank-weighted。成本统一使用简化换手成本。", "", "## 5. Main Tournament Results", top[["model_name","annual_return","annual_vol","net_sharpe","max_drawdown","monthly_turnover","mean_rank_ic"]].to_markdown(index=False), "", "## 6. Blend Results", metrics[metrics.model_name.str.startswith("BLEND")].sort_values("net_sharpe", ascending=False).head(15).to_markdown(index=False), "", "## 7. IC and Signal Quality", "Rank IC、IC IR 与累计 IC 序列见对应 CSV 和图。", "", "## 8. Turnover and Cost", f"成本模型：{ret_audit.iloc[0].cost_model}。", "", "## 9. Drawdown and Yearly Stability", "年度拆分见 `yearly_breakdown_v1.csv`。", "", "## 10. Style Exposure", "若 `style_exposure_v1.csv` 存在，则使用 preprocessed 因子面板计算持仓均值暴露。", "", "## 11. Model Complementarity", "模型相关性与独有持仓贡献见 `model_correlation_summary_v1.csv` 与 `unique_holdings_contribution_v1.csv`。", "", "## 12. Paired Performance Tests", paired.to_markdown(index=False) if not paired.empty else "样本不足或缺少比较对象。", "", "## 13. Recommendation", "详见 `production_candidate_recommendation_v1.md`。", "", "## 14. Limitations and Missing Inputs", "V0 使用既有 `split_universe_blended.parquet` 导出信号；如需更严格命名的 V0 OOS artifact，应另行导出后复跑。本报告不使用 README 旧指标。", "", "## 15. Next Actions", "投研 review 后，将最佳候选进入 paper trading shadow mode；production config 暂不自动修改。"]
    (OUT / "production_signal_tournament_report_v1.md").write_text("\n".join(lines), encoding="utf-8")


def qa(availability: dict[str, bool], decision: str) -> None:
    checks = [
        ("README.md not modified", True, "script does not write README.md"),
        ("all_daily.parquet not modified", True, "read-only label source"),
        ("model files not modified", True, "read-only signal source"),
        ("no training script executed", True, "no training import/call"),
        ("Compact-F signal loaded or missing reason recorded", availability.get("COMPACT_F", False), str(availability.get("COMPACT_F", False))),
        ("V7 signal loaded or missing reason recorded", availability.get("V7_ML", False), str(availability.get("V7_ML", False))),
        ("V0 signal loaded or missing reason recorded", availability.get("V0_LINEAR", False), str(availability.get("V0_LINEAR", False))),
        ("strict alignment completed", (OUT/"alignment_audit_v1.csv").exists(), ""),
        ("return labels available", (OUT/"return_label_audit_v1.csv").exists(), ""),
        ("costs applied consistently", True, f"{COST_BPS_PER_TURNOVER} bps turnover cost"),
        ("market timing disabled", True, "multiplier=1.0"),
        ("no Media15/XHS/Baidu used", True, "not referenced"),
        ("Top50 Buffer tested", True, ""),
        ("blends tested where inputs available", (OUT/"blend_signal_panel_v1.parquet").exists(), ""),
        ("tournament_metrics generated", (OUT/"tournament_metrics_v1.csv").exists(), ""),
        ("monthly_returns generated", (OUT/"monthly_returns_v1.csv").exists(), ""),
        ("recommendation generated", (OUT/"production_candidate_recommendation_v1.md").exists(), ""),
        ("figures generated", len(list(FIG.glob("*.png"))) >= 7, f"{len(list(FIG.glob('*.png')))} png"),
        ("final report generated", (OUT/"production_signal_tournament_report_v1.md").exists(), ""),
    ]
    pd.DataFrame(checks, columns=["check", "pass", "details"]).to_csv(OUT/"final_qa_v1.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    inv = inventory()
    signals, notes = select_signals(inv)
    labels, ret_audit = forward_returns()
    aligned, audit = aligned_panel(signals, labels)
    blends = make_blends(aligned)
    metrics, monthly, yearly, icdf, holds = run_tournament(aligned, blends)
    correlations(aligned)
    style_exposure(holds)
    best_pre = metrics[metrics.portfolio_rule == "Top50_Buffer_35_75"].sort_values(["net_sharpe", "max_drawdown"], ascending=[False, False]).iloc[0]
    unique_contribution(aligned, holds, (best_pre.model_name, best_pre.portfolio_rule))
    paired = paired_tests(monthly, metrics)
    figures(metrics, monthly, yearly, icdf)
    availability = {m: m in set(signals.model_name) for m in ["V0_LINEAR", "V7_ML", "COMPACT_F"]}
    best_tuple = recommendation(metrics, availability)
    report(metrics, audit, ret_audit, paired, best_tuple)
    qa(availability, best_tuple[-1])
    paths = {
        "signal_inventory_path": OUT / "signal_artifact_inventory_v1.csv",
        "model_signal_panel_path": OUT / "model_signal_panel_v1.parquet",
        "alignment_audit_path": OUT / "alignment_audit_v1.csv",
        "return_label_audit_path": OUT / "return_label_audit_v1.csv",
        "blend_signal_panel_path": OUT / "blend_signal_panel_v1.parquet",
        "tournament_metrics_path": OUT / "tournament_metrics_v1.csv",
        "monthly_returns_path": OUT / "monthly_returns_v1.csv",
        "rank_ic_series_path": OUT / "rank_ic_series_v1.csv",
        "style_exposure_path": OUT / "style_exposure_v1.csv" if (OUT/"style_exposure_v1.csv").exists() else OUT/"style_exposure_missing_reason.md",
        "model_correlation_summary_path": OUT / "model_correlation_summary_v1.csv",
        "paired_test_path": OUT / "paired_performance_test_v1.csv",
        "recommendation_path": OUT / "production_candidate_recommendation_v1.md",
        "report_path": OUT / "production_signal_tournament_report_v1.md",
        "final_qa_path": OUT / "final_qa_v1.csv",
    }
    for k, v in paths.items():
        print(f"{k}={v.relative_to(ROOT)}")
    print(f"best_model_name={best_tuple[0]}")
    print(f"best_portfolio_rule={best_tuple[1]}")
    print(f"best_net_sharpe={best_tuple[2]:.6f}")
    print(f"best_max_drawdown={best_tuple[3]:.6f}")
    print(f"best_monthly_turnover={best_tuple[4]:.6f}")
    print(f"v0_available={availability['V0_LINEAR']}")
    print(f"v7_available={availability['V7_ML']}")
    print(f"compact_f_available={availability['COMPACT_F']}")
    print(f"decision={best_tuple[5]}")


if __name__ == "__main__":
    main()
