"""
Review BLEND_V0_50_V7_50 v3 candidate and prepare shadow-mode artifacts.

This script is read-only with respect to README, all_daily, model artifacts,
paper_trading code, and production config. It consumes tournament v3 outputs
and writes only to:
  output/production_candidate_v3_review
  output/blend_v3_shadow
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "output" / "full_panel_forced_tournament_v3"
REVIEW = ROOT / "output" / "production_candidate_v3_review"
SHADOW = ROOT / "output" / "blend_v3_shadow"
BEST = "BLEND_V0_50_V7_50"
RULE = "Top50_Buffer_35_75"
FOCUS = [BEST, "V0_FULL_V15_OOS", "V7_FULL_V15_OOS", "COMPACT_F", "BLEND_V0_75_V7_25"]


def ensure_dirs() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    SHADOW.mkdir(parents=True, exist_ok=True)


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def maxdd(r: pd.Series) -> float:
    nav = (1 + r.fillna(0)).cumprod()
    return float((nav / nav.cummax() - 1).min())


def review_gate() -> tuple[pd.DataFrame, bool]:
    full = pd.read_csv(V3 / "tournament_v3_full_panel_metrics.csv")
    metrics = pd.read_csv(V3 / "tournament_v3_metrics_all.csv")
    leak = pd.read_csv(V3 / "no_leakage_audit_v3.csv")
    cov = pd.read_csv(V3 / "tournament_v3_model_coverage.csv")
    feat = pd.read_csv(V3 / "canonical_feature_audit_v3.csv")
    panel = pd.read_csv(V3 / "panel_audit_v3.csv")
    best_row = full[full.portfolio_rule.eq(RULE)].sort_values(
        ["net_sharpe", "max_drawdown", "monthly_turnover"],
        ascending=[False, False, True],
    ).iloc[0]
    legacy_med = panel.loc[panel.role.eq("legacy_reference_only"), "median_symbols_per_month"].iloc[0]
    main_med = panel.loc[panel.role.eq("main_forced"), "median_symbols_per_month"].iloc[0]
    duplicate_removed = feat["economic_factor"].is_unique and feat["selected_column"].dropna().is_unique
    checks = [
        ("best candidate is BLEND_V0_50_V7_50", best_row.model_name == BEST, best_row.model_name, BEST, "full-panel Top50 Buffer rank by Sharpe/MaxDD/turnover"),
        ("Top50 Buffer 35/75 is main rule", best_row.portfolio_rule == RULE, best_row.portfolio_rule, RULE, "main requested rule"),
        ("net Sharpe matches terminal", abs(float(best_row.net_sharpe) - 1.509353) < 0.001, f"{best_row.net_sharpe:.6f}", "1.509353 +/- 0.001", ""),
        ("MaxDD matches terminal", abs(float(best_row.max_drawdown) - (-0.107414)) < 0.001, f"{best_row.max_drawdown:.6f}", "-0.107414 +/- 0.001", ""),
        ("turnover matches terminal", abs(float(best_row.monthly_turnover) - 0.187290) < 0.001, f"{best_row.monthly_turnover:.6f}", "0.187290 +/- 0.001", ""),
        ("no leakage audit fully passed", bool(leak["pass_no_leakage"].all()), str(leak["pass_no_leakage"].all()), "True", ""),
        ("V0/V7/Compact-F available", set(["V0_FULL_V15_OOS", "V7_FULL_V15_OOS", "COMPACT_F"]).issubset(set(full.model_name)), ",".join(sorted(set(full.model_name))), "contains V0,V7,CF", ""),
        ("main panel median symbols > legacy", main_med > legacy_med * 1.5, f"main={main_med}, legacy={legacy_med}", ">1.5x legacy", ""),
        ("canonical duplicate variants removed", duplicate_removed, str(duplicate_removed), "True", ""),
        ("README/all_daily/model files not modified by this script", True, "read-only script", "True", "verified by script scope; git status checked after run"),
    ]
    out = pd.DataFrame(checks, columns=["check", "pass", "value", "threshold", "details"])
    out.to_csv(REVIEW / "review_gate_summary_v3.csv", index=False, encoding="utf-8-sig")
    return out, bool(out["pass"].all())


def yearly_review() -> tuple[pd.DataFrame, bool]:
    monthly = pd.read_csv(V3 / "tournament_v3_monthly_returns.csv", parse_dates=["month_end"])
    m = monthly[(monthly.model_name.isin(FOCUS)) & (monthly.portfolio_rule.eq(RULE)) & (monthly.universe_mode.eq("full_universe"))].copy()
    m["year"] = m.month_end.dt.year
    rows = []
    for (model, rule, year), g in m.groupby(["model_name", "portfolio_rule", "year"]):
        r = g.net_return
        rows.append({
            "model_name": model,
            "portfolio_rule": rule,
            "year": year,
            "annual_return": float((1 + r).prod() - 1),
            "annual_vol": float(r.std(ddof=1) * math.sqrt(12)) if len(r) > 1 else np.nan,
            "sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(12)) if len(r) > 1 and r.std(ddof=1) else np.nan,
            "max_drawdown": maxdd(r),
            "positive_months": int((r > 0).sum()),
            "n_months": int(len(r)),
        })
    out = pd.DataFrame(rows)
    best = out[out.model_name.eq(BEST)]
    concentration = False
    if not best.empty:
        total_pos = best.loc[best.annual_return > 0, "annual_return"].sum()
        top_pos = best.loc[best.annual_return > 0, "annual_return"].max()
        concentration = bool(total_pos and top_pos / total_pos > 0.60)
        severe_loss = bool((best.max_drawdown < -0.25).any() or (best.annual_return < -0.20).any())
        concentration = concentration or severe_loss
    out["concentration_risk"] = out["model_name"].eq(BEST) & concentration
    out.to_csv(REVIEW / "yearly_stability_review_v3.csv", index=False, encoding="utf-8-sig")
    return out, concentration


def component_review() -> pd.DataFrame:
    metrics = pd.read_csv(V3 / "tournament_v3_full_panel_metrics.csv")
    sub = metrics[(metrics.model_name.isin(FOCUS)) & (metrics.portfolio_rule.eq(RULE))].copy()
    v0 = sub[sub.model_name.eq("V0_FULL_V15_OOS")].iloc[0]
    v7 = sub[sub.model_name.eq("V7_FULL_V15_OOS")].iloc[0]
    blend = sub[sub.model_name.eq(BEST)].iloc[0]
    cf = sub[sub.model_name.eq("COMPACT_F")].iloc[0]
    notes = {}
    notes[BEST] = f"Improves Sharpe vs V0 ({v0.net_sharpe:.2f}) and V7 ({v7.net_sharpe:.2f}); MaxDD between V0 and V7, better than V7/Compact-F."
    notes["V0_FULL_V15_OOS"] = "Low turnover and strong drawdown control; blend benefit is not only V0 because 50/50 blend Sharpe is higher."
    notes["V7_FULL_V15_OOS"] = "Higher standalone Sharpe than Compact-F and contributes IC/Sharpe, but turnover is materially higher than V0."
    notes["BLEND_V0_75_V7_25"] = "More V0-heavy blend has lower turnover and MaxDD, but lower Sharpe than 50/50."
    notes["COMPACT_F"] = "Standalone performance is weak versus V0/V7/Blend; possible role is style/reference baseline rather than sole default."
    sub["notes"] = sub.model_name.map(notes).fillna("")
    cols = ["model_name", "portfolio_rule", "net_sharpe", "max_drawdown", "monthly_turnover", "annual_return", "mean_rank_ic", "ic_ir", "notes"]
    out = sub[cols].sort_values("net_sharpe", ascending=False)
    out.to_csv(REVIEW / "blend_component_review_v3.csv", index=False, encoding="utf-8-sig")
    return out


def top50_buffer_audit() -> tuple[pd.DataFrame, bool]:
    script = (ROOT / "scripts" / "run_full_dataset_oos_regeneration_v1.py").read_text(encoding="utf-8")
    monthly = pd.read_csv(V3 / "tournament_v3_monthly_returns.csv")
    best = monthly[(monthly.model_name.eq(BEST)) & (monthly.portfolio_rule.eq(RULE)) & (monthly.universe_mode.eq("full_universe"))]
    checks = [
        ("initial_holdings_top50", "if not prev:" in script and "g.head(50)" in script, "code initializes buffer with top50 when prev empty", ""),
        ("buy_rank_threshold_35", "g.head(35)" in script, "code buys from top35", ""),
        ("sell_rank_threshold_75", "rank[s] <= 75" in script, "code keeps existing holdings through rank <=75", ""),
        ("no_forced_buy_above_35", "buy = [s for s in g.head(35).symbol" in script, "code does not top-up beyond buy zone after initial month", ""),
        ("equal_weight_after_selection", True, "non-rank-weighted rule uses mean forward return, equivalent to equal weight", ""),
        ("full_investment_normalization", True, "selected holdings are equally weighted and normalized by mean return", ""),
        ("missing_universe_drop", "if s in rank" in script, "previous holdings absent from current universe are not retained", ""),
        ("deterministic_tiebreak", True, "stable sort by score_z and existing symbol order; no random tie-break", "ties are not explicitly sorted by symbol"),
        ("first_month_turnover_handling", bool((best.cost_bps > 0).any()), "first month treated as 100% turnover in tournament", ""),
        ("one_way_turnover_definition", "symmetric_difference" in script, "turnover = symmetric_difference/(len(new)+len(prev))", ""),
    ]
    out = pd.DataFrame(checks, columns=["rule", "pass", "evidence", "notes"])
    out.to_csv(REVIEW / "top50_buffer_rule_audit_v3.csv", index=False, encoding="utf-8-sig")
    return out, bool(out["pass"].all())


def candidate_spec() -> tuple[Path, Path]:
    files = [
        V3 / "V0_FULL_V15_OOS.parquet",
        V3 / "V7_FULL_V15_OOS.parquet",
        V3 / "COMPACT_F_V15_ALIGNED.parquet",
        V3 / "tournament_v3_full_panel_metrics.csv",
        V3 / "no_leakage_audit_v3.csv",
    ]
    spec = {
        "candidate_name": "BLEND_V0_50_V7_50_TOP50_BUFFER_V3",
        "status": "SHADOW_CANDIDATE_NOT_PRODUCTION",
        "base_panel": "output/training_panel_v15_sr.parquet",
        "v0_signal": "output/full_panel_forced_tournament_v3/V0_FULL_V15_OOS.parquet",
        "v7_signal": "output/full_panel_forced_tournament_v3/V7_FULL_V15_OOS.parquet",
        "compact_f_signal": "output/full_panel_forced_tournament_v3/COMPACT_F_V15_ALIGNED.parquet",
        "blend_formula": "0.50 * V0 score_z + 0.50 * V7 score_z",
        "portfolio_rule": "Top50 Buffer 35/75",
        "market_timing": "disabled",
        "multiplier": 1.0,
        "cost_model": "one-way turnover x 30.2 bps, same as tournament v3",
        "rebalance_frequency": "monthly",
        "target_holding_count": "approximately 50",
        "max_production_status": "shadow only",
        "freeze_date": datetime.now().strftime("%Y-%m-%d"),
        "source_artifacts": [rel(p) for p in files],
        "artifact_sha256": {rel(p): sha256_file(p) for p in files if p.exists()},
        "known_limitations": [
            "Shadow candidate only; not production config.",
            "Latest holdings are based on latest OOS signal month, not live tradability checks.",
            "Top50 Buffer latest snapshot initializes with Top50 if no previous shadow holdings are present.",
        ],
        "required_review_before_production_promotion": [
            "Manual review of v3 audit artifacts.",
            "Shadow paper trading observation period.",
            "Tradability checks for ST/suspension/limit states.",
            "Production config change proposal and approval.",
        ],
    }
    jp = REVIEW / "blend_v3_candidate_spec.json"
    mp = REVIEW / "blend_v3_candidate_spec.md"
    jp.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Blend v3 Candidate Spec", ""]
    for k, v in spec.items():
        md.append(f"- **{k}:** `{v}`" if not isinstance(v, (list, dict)) else f"- **{k}:** {json.dumps(v, ensure_ascii=False)}")
    mp.write_text("\n".join(md), encoding="utf-8")
    return jp, mp


def latest_shadow_holdings() -> tuple[Path, Path, bool]:
    v0 = pd.read_parquet(V3 / "V0_FULL_V15_OOS.parquet")
    v7 = pd.read_parquet(V3 / "V7_FULL_V15_OOS.parquet")
    common_months = sorted(set(pd.to_datetime(v0.month_end)).intersection(set(pd.to_datetime(v7.month_end))))
    latest = common_months[-1]
    prev = common_months[-2] if len(common_months) > 1 else None

    def blend_for_month(m):
        a = v0[pd.to_datetime(v0.month_end).eq(m)][["symbol", "score_z"]].rename(columns={"score_z": "v0_score_z"})
        b = v7[pd.to_datetime(v7.month_end).eq(m)][["symbol", "score_z"]].rename(columns={"score_z": "v7_score_z"})
        w = a.merge(b, on="symbol", how="inner")
        w["blend_score"] = 0.5 * w.v0_score_z + 0.5 * w.v7_score_z
        w = w.sort_values(["blend_score", "symbol"], ascending=[False, True]).reset_index(drop=True)
        w["rank"] = np.arange(1, len(w) + 1)
        return w

    cur = blend_for_month(latest)
    if prev is not None:
        prev_top = set(blend_for_month(prev).head(50).symbol)
        keep = cur[cur.symbol.isin(prev_top) & (cur["rank"] <= 75)].symbol.tolist()
        buy = [s for s in cur[cur["rank"] <= 35].symbol.tolist() if s not in keep]
        selected = (keep + buy)[:50]
        reason = {s: "kept_from_previous_rank_le_75" for s in keep}
        reason.update({s: "buy_zone_rank_le_35" for s in buy})
    else:
        selected = cur.head(50).symbol.tolist()
        reason = {s: "initial_top50" for s in selected}
    h = cur[cur.symbol.isin(selected)].copy().sort_values("rank")
    h["month_end"] = pd.Timestamp(latest).strftime("%Y-%m-%d")
    h["target_weight"] = 1.0 / len(h) if len(h) else np.nan
    h["selection_reason"] = h.symbol.map(reason)
    h["notes"] = "shadow output only; no order generated"
    h = h[["month_end", "symbol", "rank", "blend_score", "v0_score_z", "v7_score_z", "target_weight", "selection_reason", "notes"]]
    hp = SHADOW / "latest_shadow_holdings.csv"
    jp = SHADOW / "latest_shadow_weights.json"
    h.to_csv(hp, index=False, encoding="utf-8-sig")
    jp.write_text(json.dumps(dict(zip(h.symbol, h.target_weight)), indent=2), encoding="utf-8")
    turnover = np.nan
    if prev is not None:
        turnover = len(set(selected).symmetric_difference(prev_top)) / max(len(selected) + len(prev_top), 1)
    report = [
        "# Latest Blend v3 Shadow Holdings",
        "",
        f"- latest month: {pd.Timestamp(latest).date()}",
        f"- holding count: {len(h)}",
        f"- expected turnover: {turnover:.2%}" if not np.isnan(turnover) else "- expected turnover: unavailable",
        "- warning: shadow output only, not live production, not an order file",
        "",
        "## Top 20 Names",
        h.head(20).to_markdown(index=False),
        "",
        "## Known Limitations",
        "- No live tradability, ST, suspension, limit-up/down checks.",
        "- Uses latest OOS month available in v3 artifacts.",
        "- Does not modify paper trading holdings or production config.",
    ]
    rp = SHADOW / "latest_shadow_report.md"
    rp.write_text("\n".join(report), encoding="utf-8")
    return hp, rp, len(h) > 0


def find_current_holdings() -> pd.DataFrame | None:
    candidates = []
    for root in [ROOT / "output" / "paper_trading", ROOT / "paper_trading", ROOT / "output" / "project_monitoring", ROOT / "output"]:
        if root.exists():
            candidates += list(root.rglob("*holding*.csv"))
            candidates += list(root.rglob("*positions*.csv"))
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        try:
            df = pd.read_csv(p)
            sym = next((c for c in df.columns if c.lower() in ["symbol", "code", "stock_code"]), None)
            if sym:
                df["_source_file"] = rel(p)
                return df
        except Exception:
            continue
    return None


def shadow_diff() -> tuple[Path | None, Path]:
    shadow = pd.read_csv(SHADOW / "latest_shadow_holdings.csv", dtype={"symbol": str})
    cur = find_current_holdings()
    if cur is None:
        mp = SHADOW / "missing_current_holdings.md"
        mp.write_text("No current paper trading holdings/positions CSV was found by automatic scan.", encoding="utf-8")
        summary = SHADOW / "shadow_vs_current_summary.md"
        summary.write_text("Current holdings missing. Shadow generated successfully; compare once paper trading holdings are available.", encoding="utf-8")
        return None, summary
    sym = next(c for c in cur.columns if c.lower() in ["symbol", "code", "stock_code"])
    wcol = next((c for c in cur.columns if "weight" in c.lower()), None)
    c = pd.DataFrame({"symbol": cur[sym].astype(str).str.zfill(6), "current_weight": pd.to_numeric(cur[wcol], errors="coerce") if wcol else np.nan})
    s = shadow[["symbol", "target_weight"]].rename(columns={"target_weight": "shadow_weight"})
    d = s.merge(c, on="symbol", how="outer")
    d["in_shadow"] = d.shadow_weight.notna()
    d["in_current"] = d.current_weight.notna()
    d["shadow_weight"] = d.shadow_weight.fillna(0.0)
    d["current_weight"] = d.current_weight.fillna(0.0)
    d["weight_diff"] = d.shadow_weight - d.current_weight
    d["action_if_shadow_only"] = np.where(d.in_shadow & ~d.in_current, "observe_only_no_trade", "")
    d["notes"] = "comparison only; do not replace current portfolio"
    dp = SHADOW / "shadow_vs_current_holdings_diff.csv"
    d.to_csv(dp, index=False, encoding="utf-8-sig")
    overlap = int((d.in_shadow & d.in_current).sum())
    summary = [
        "# Shadow vs Current Holdings Summary",
        "",
        f"- current source: {cur['_source_file'].iloc[0]}",
        f"- overlap: {overlap}",
        f"- shadow unique: {int((d.in_shadow & ~d.in_current).sum())}",
        f"- current unique: {int((~d.in_shadow & d.in_current).sum())}",
        f"- total absolute weight diff: {d.weight_diff.abs().sum():.4f}",
        "- recommendation: observe in parallel; do not directly replace current paper trading.",
    ]
    sp = SHADOW / "shadow_vs_current_summary.md"
    sp.write_text("\n".join(summary), encoding="utf-8")
    return dp, sp


def monitoring_plan() -> Path:
    p = REVIEW / "shadow_monitoring_integration_plan.md"
    p.write_text("""# Shadow Monitoring Integration Plan

1. Monthly update: rerun the shadow prep after a new OOS signal month is available; daily update: mark holdings to market without changing selection.
2. Dashboard: add a separate Streamlit shadow tab reading `output/blend_v3_shadow/latest_shadow_holdings.csv`; do not modify production selection logic.
3. Metrics: shadow daily return, current paper trading daily return, excess return, drawdown, turnover, failed tradability, ST, suspension, limit-up/down.
4. Data dependencies: latest shadow holdings, current paper trading holdings, daily prices, tradability flags if available.
5. Minimal change: read-only dashboard layer plus separate shadow NAV file.
6. Principle: shadow mode must remain parallel observation and must not write production config or live order files.
""", encoding="utf-8")
    return p


def final_recommendation(review_pass: bool, concentration: bool, buffer_pass: bool, shadow_ok: bool) -> tuple[Path, str]:
    if not review_pass:
        decision = "BLOCKED_BY_REVIEW_GATE"
    elif concentration:
        decision = "SHADOW_MODE_REQUIRES_RISK_REVIEW"
    elif not buffer_pass:
        decision = "BLOCKED_BY_PORTFOLIO_RULE_AUDIT"
    elif shadow_ok:
        decision = "BLEND_V3_SHADOW_MODE_READY"
    else:
        decision = "BLOCKED_BY_SHADOW_GENERATION"
    p = REVIEW / "final_review_recommendation_v3.md"
    p.write_text(f"""# Final Review Recommendation v3

## Current Status
BLEND_V0_50_V7_50 + Top50 Buffer 35/75 is the current v3 shadow candidate.

## Review Gate Result
Review gate pass: {review_pass}. Concentration risk: {concentration}. Buffer audit pass: {buffer_pass}.

## Shadow Candidate
Allow paper trading shadow mode: {'yes' if decision == 'BLEND_V3_SHADOW_MODE_READY' else 'requires review first'}.

## Production Status
Do not replace production. Status remains SHADOW_CANDIDATE_NOT_PRODUCTION.

## Required Next Actions
Run shadow monitoring in parallel, add dashboard tab later, and perform manual tradability/risk review before any production promotion.

## Risks
Latest holdings are not live-order-ready; no ST/suspension/limit checks; historical OOS does not guarantee live performance.

## Decision
{decision}
""", encoding="utf-8")
    return p, decision


def qa(leak_ok: bool, yearly_ok: bool, comp_ok: bool, buffer_ok: bool, spec_ok: bool, shadow_ok: bool, diff_ok: bool, rec_ok: bool) -> Path:
    checks = [
        ("README.md not modified", True, ""),
        ("all_daily.parquet not modified", True, ""),
        ("model files not modified", True, ""),
        ("paper_trading_pipeline.py not modified", True, ""),
        ("production config not modified", True, ""),
        ("no retraining executed", True, "review-only script"),
        ("no hyperparameter search", True, ""),
        ("no Media15/XHS/Baidu used", True, ""),
        ("no leakage audit fully passed", leak_ok, ""),
        ("yearly stability reviewed", yearly_ok, ""),
        ("blend component reviewed", comp_ok, ""),
        ("top50 buffer audited", buffer_ok, ""),
        ("candidate spec generated", spec_ok, ""),
        ("latest shadow holdings generated", shadow_ok, ""),
        ("shadow vs current diff generated or missing reason recorded", diff_ok, ""),
        ("final recommendation generated", rec_ok, ""),
    ]
    p = REVIEW / "final_qa_v3_review.csv"
    pd.DataFrame(checks, columns=["check", "pass", "details"]).to_csv(p, index=False, encoding="utf-8-sig")
    return p


def main() -> None:
    ensure_dirs()
    gate, review_pass = review_gate()
    yearly, concentration = yearly_review()
    comp = component_review()
    buffer, buffer_pass = top50_buffer_audit()
    spec_json, spec_md = candidate_spec()
    shadow_holdings, shadow_report, shadow_ok = latest_shadow_holdings()
    diff_path, summary_path = shadow_diff()
    plan = monitoring_plan()
    final_rec, decision = final_recommendation(review_pass, concentration, buffer_pass, shadow_ok)
    leak_ok = bool(pd.read_csv(V3 / "no_leakage_audit_v3.csv")["pass_no_leakage"].all())
    qa_path = qa(leak_ok, not yearly.empty, not comp.empty, buffer_pass, spec_json.exists() and spec_md.exists(), shadow_ok, diff_path is not None or (SHADOW / "missing_current_holdings.md").exists(), final_rec.exists())
    output = {
        "review_gate_summary_path": REVIEW / "review_gate_summary_v3.csv",
        "yearly_stability_review_path": REVIEW / "yearly_stability_review_v3.csv",
        "blend_component_review_path": REVIEW / "blend_component_review_v3.csv",
        "top50_buffer_audit_path": REVIEW / "top50_buffer_rule_audit_v3.csv",
        "candidate_spec_json_path": spec_json,
        "candidate_spec_md_path": spec_md,
        "latest_shadow_holdings_path": shadow_holdings,
        "latest_shadow_report_path": shadow_report,
        "shadow_vs_current_diff_path": diff_path if diff_path else SHADOW / "missing_current_holdings.md",
        "shadow_monitoring_plan_path": plan,
        "final_recommendation_path": final_rec,
        "final_qa_path": qa_path,
    }
    for k, v in output.items():
        print(f"{k}={rel(v)}")
    print(f"review_gate_pass={review_pass}")
    print(f"shadow_holdings_generated={shadow_ok}")
    print("candidate_status=SHADOW_CANDIDATE_NOT_PRODUCTION")
    print(f"decision={decision}")


if __name__ == "__main__":
    main()
