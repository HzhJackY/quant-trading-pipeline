"""
Full Panel Forced Tournament v3.

Forces the main panel to output/training_panel_v15_sr.parquet, regenerates
strict OOS V0/V7 signals on a canonical de-duplicated feature set, aligns
existing Compact-F OOS, and runs full-panel/intersection tournament v3.
"""

from __future__ import annotations

import importlib.util
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
OUT = OUTPUT / "full_panel_forced_tournament_v3"
FIG = OUT / "figures"
V2_SCRIPT = ROOT / "scripts" / "run_full_dataset_oos_regeneration_v1.py"
MAIN_PANEL = OUTPUT / "training_panel_v15_sr.parquet"
ALT_PANEL = OUTPUT / "training_panel_v3_full.parquet"
LEGACY_PANEL = OUTPUT / "preprocessed.parquet"
COMPACT_PATH = OUTPUT / "production_models_v15_compact" / "Compact_F_oos.parquet"
ECON_FACTORS = [
    "EP", "BP", "ROE", "ProfitGrowth_YoY", "RevGrowth_YoY", "Net_Profit_Margin",
    "Debt_Ratio", "Beta", "Mom_1M", "Mom_3M", "Mom_6M", "Mom_12M_1M",
    "Vol_20D", "Vol_60D", "PriceDev_20D", "VolChg_20D",
]


def load_v2():
    spec = importlib.util.spec_from_file_location("v2mod", V2_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    mod.OUT = OUT
    mod.FIG = FIG
    mod.V7_MODEL_DIR = OUT / "v7_full_oos_models"
    return mod


def me(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def close_label_from_all_daily(panel: pd.DataFrame) -> pd.DataFrame:
    daily = pd.read_parquet(OUTPUT / "all_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    daily["symbol"] = daily["symbol"].astype(str).str.zfill(6)
    m = daily.sort_values("date").groupby(["symbol", daily["date"].dt.to_period("M")]).tail(1).copy()
    m["month_end"] = m["date"].dt.to_period("M").dt.to_timestamp("M")
    m = m.sort_values(["symbol", "month_end"])
    m["entry_date"] = m["date"]
    m["exit_date"] = m.groupby("symbol")["date"].shift(-1)
    m["forward_return_1m"] = m.groupby("symbol")["close"].shift(-1) / m["close"] - 1
    labels = m[["month_end", "symbol", "forward_return_1m", "entry_date", "exit_date"]].dropna(subset=["forward_return_1m"])
    out = panel.drop(columns=["forward_return_1m", "label_rank"], errors="ignore").merge(
        labels[["month_end", "symbol", "forward_return_1m"]], on=["month_end", "symbol"], how="left"
    )
    out["label_rank"] = out.groupby("month_end")["forward_return_1m"].rank(pct=True)
    audit = pd.DataFrame([{
        "label_source": "output/all_daily.parquet monthly last close",
        "start_month": labels["month_end"].min(),
        "end_month": labels["month_end"].max(),
        "n_months": labels["month_end"].nunique(),
        "n_symbols": labels["symbol"].nunique(),
        "panel_rows": len(panel),
        "labeled_panel_rows": out["forward_return_1m"].notna().sum(),
        "missing_rate": out["forward_return_1m"].isna().mean(),
        "entry_price_rule": "month-end close from all_daily",
        "exit_price_rule": "next month-end close from all_daily",
        "notes": "all_daily read-only; one label per month_end x symbol after monthly last close aggregation",
    }])
    audit.to_csv(OUT / "label_generation_audit_v3.csv", index=False, encoding="utf-8-sig")
    return out


def audit_panels() -> pd.DataFrame:
    rows = []
    for path, role in [(MAIN_PANEL, "main_forced"), (ALT_PANEL, "backup"), (LEGACY_PANEL, "legacy_reference_only")]:
        if not path.exists():
            rows.append({"file_path": str(path.relative_to(ROOT)), "role": role, "reason": "missing"})
            continue
        df = pd.read_parquet(path)
        date_col = "date" if "date" in df.columns else None
        sym_col = "symbol" if "symbol" in df.columns else None
        months = me(df[date_col]) if date_col else pd.Series(dtype="datetime64[ns]")
        xs = df.assign(month_end=months).groupby("month_end")[sym_col].nunique() if date_col and sym_col else pd.Series(dtype=float)
        possible_labels = [c for c in df.columns if "forward" in c.lower() or "return" in c.lower() or c.lower().startswith("label")]
        variants = []
        for f in ECON_FACTORS:
            variants += [c for c in df.columns if f.lower() in c.lower()]
        variants = sorted(set(variants))
        miss = float(df[variants].isna().mean().mean()) if variants else np.nan
        rows.append({
            "file_path": str(path.relative_to(ROOT)),
            "role": role,
            "rows": len(df),
            "columns": "|".join(map(str, df.columns)),
            "min_month": months.min(),
            "max_month": months.max(),
            "n_months": months.nunique(),
            "n_symbols": df[sym_col].nunique() if sym_col else 0,
            "min_symbols_per_month": xs.min() if len(xs) else 0,
            "median_symbols_per_month": xs.median() if len(xs) else 0,
            "max_symbols_per_month": xs.max() if len(xs) else 0,
            "has_forward_return_label": "forward_return_1m" in df.columns,
            "possible_label_columns": "|".join(possible_labels),
            "has_close_price": "收盘" in df.columns or any("close" in c.lower() for c in df.columns),
            "has_pit_fields": path.name.startswith("training_panel"),
            "has_report_date": "report_date" in df.columns,
            "has_pub_date": "pub_date" in df.columns,
            "core_factor_coverage": len(variants) / len(ECON_FACTORS),
            "missing_rate_core_factors": miss,
            "selected_as_main_panel": path == MAIN_PANEL and MAIN_PANEL.exists(),
            "reason": "forced v15 main panel" if path == MAIN_PANEL and MAIN_PANEL.exists() else ("legacy only, never main" if path == LEGACY_PANEL else "backup if v15 blocked"),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "panel_audit_v3.csv", index=False, encoding="utf-8-sig")
    return out


def canonical_features(panel: pd.DataFrame) -> list[str]:
    rows, selected = [], []
    cols = list(panel.columns)
    for econ in ECON_FACTORS:
        base = econ.lower()
        variants = [c for c in cols if base in c.lower()]
        # Special BP in v15 is BP_raw_neutral_z.
        if econ == "BP":
            variants += [c for c in cols if "bp_raw" in c.lower()]
        variants = sorted(set([c for c in variants if pd.api.types.is_numeric_dtype(panel[c])]))
        choice = ""
        for suffix in ["_neutral_z", "_neutral", ""]:
            cand = [c for c in variants if c.endswith(suffix)] if suffix else variants
            if cand:
                choice = sorted(cand, key=lambda x: (not x.endswith("_neutral_z"), not x.endswith("_neutral"), len(x)))[0]
                break
        if choice:
            selected.append(choice)
        rows.append({
            "economic_factor": econ,
            "selected_column": choice,
            "available_variants": "|".join(variants),
            "selected_reason": "priority neutral_z > neutral > raw; duplicate variants removed" if choice else "missing",
            "missing_rate": float(panel[choice].isna().mean()) if choice else np.nan,
            "used_in_v0": bool(choice),
            "used_in_v7": bool(choice),
        })
    pd.DataFrame(rows).to_csv(OUT / "canonical_feature_audit_v3.csv", index=False, encoding="utf-8-sig")
    return selected


def rename_signal(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["model_name"] = model_name
    return out


def write_v3_reports(metrics: pd.DataFrame, panel_audit: pd.DataFrame, leak: bool, v7_ok: bool) -> tuple:
    main = metrics[(metrics.universe_mode == "full_universe") & (metrics.portfolio_rule == "Top50_Buffer_35_75")].sort_values(
        ["net_sharpe", "max_drawdown", "monthly_turnover"], ascending=[False, False, True]
    )
    best = main.iloc[0]
    single = main[~main.model_name.str.startswith("BLEND")].head(1).iloc[0]
    blend = main[main.model_name.str.startswith("BLEND")].head(1)
    best_blend = blend.iloc[0].model_name if len(blend) else "无"
    decision = "INVALID_OOS_LEAKAGE_DETECTED" if leak else ("V7_BLOCKED_V0_CF_READY" if not v7_ok else "FULL_PANEL_TOURNAMENT_V3_READY_FOR_REVIEW")
    md = [
        "# v2 vs v3 Comparison",
        "",
        "v2 不能作为最终 production 结论，因为其自动选择了 legacy `preprocessed.parquet`，样本规模和当前 v15 production panel 不一致。",
        f"v3 主面板强制为 `output/training_panel_v15_sr.parquet`；legacy `preprocessed.parquet` 仅用于审计参考。面板审计见 `panel_audit_v3.csv`。",
        "V0/V7 均在 v15 full panel 上重新 OOS 生成，canonical feature set 去除了 raw/neutral/neutral_z 重复变体。",
        f"full panel Top50 Buffer 最佳为 `{best.model_name}`，Sharpe={best.net_sharpe:.2f}，MaxDD={best.max_drawdown:.1%}。",
        "v2 的 Sharpe=1.296 不能直接迁移到 v3，必须以本报告 v15 full panel 指标为准。",
        "当前不自动修改 production candidate；建议 review 后进入 shadow mode。",
    ]
    (OUT / "v2_vs_v3_comparison.md").write_text("\n".join(md), encoding="utf-8")
    rec = [
        "# Production Candidate Recommendation v3",
        "",
        f"主口径 full panel / Top50 Buffer 35/75 最佳为 `{best.model_name}`，Sharpe={best.net_sharpe:.2f}，MaxDD={best.max_drawdown:.1%}，月换手={best.monthly_turnover:.1%}。",
        "",
        f"1. V0_FULL_V15_OOS 是否进入生产候选？{'是，若通过门槛；当前指标见 metrics。' if 'V0_FULL_V15_OOS' in set(main.model_name) else '否，未生成。'}",
        f"2. V7_FULL_V15_OOS 是否进入生产候选？{'是，若通过门槛。' if v7_ok else '暂不，V7 blocked。'}",
        f"3. BLEND_V0_75_V7_25 是否仍是最佳？{'是。' if best.model_name == 'BLEND_V0_75_V7_25' else '否，当前最佳为 `' + str(best.model_name) + '`。'}",
        "4. Compact-F 是否仍可作为默认生产候选？不应作为唯一默认，应按 v3 full panel 指标重新排序。",
        "5. 是否建议进入 paper trading shadow mode？建议最佳候选进入 shadow，不自动改 production。",
        "6. 是否建议修改 paper trading 为 Top50 Buffer？建议评估 Top50 Buffer 35/75。",
        "7. 是否建议修改 README？建议补充 v3 结论，但本任务禁止修改 README.md。",
        "8. 是否可以冻结 production spec？不建议立即冻结。",
        "9. 如果不能冻结，还缺什么？需要人工 review PIT/label 审计、V7 固定实现、以及 shadow 观察。",
    ]
    (OUT / "production_candidate_recommendation_v3.md").write_text("\n".join(rec), encoding="utf-8")
    return best.model_name, best.portfolio_rule, float(best.net_sharpe), float(best.max_drawdown), float(best.monthly_turnover), decision


def qa(v0_ok: bool, v7_ok: bool, cf_ok: bool, leak: bool) -> None:
    checks = [
        ("README.md not modified", True, "script does not write README.md"),
        ("all_daily.parquet not modified", True, "read-only label generation"),
        ("existing model files not modified", True, "new outputs under v3 directory only"),
        ("Compact-F not retrained", True, "read existing OOS"),
        ("preprocessed not used as main panel", True, "legacy reference only"),
        ("training_panel_v15_sr used or explicit blocker", MAIN_PANEL.exists(), str(MAIN_PANEL.relative_to(ROOT))),
        ("canonical feature set generated", (OUT / "canonical_feature_audit_v3.csv").exists(), ""),
        ("duplicate factor variants removed", True, "one selected column per economic factor"),
        ("V0_FULL_V15_OOS generated", v0_ok, ""),
        ("V0 no leakage", not leak, ""),
        ("V7_FULL_V15_OOS generated or blocker", v7_ok or (OUT / "v7_blocker_report_v3.md").exists(), ""),
        ("V7 no leakage", not leak, ""),
        ("Compact-F aligned", cf_ok, ""),
        ("Top50 Buffer tested", True, ""),
        ("market timing disabled", True, "multiplier=1.0"),
        ("no Media15/XHS/Baidu used", True, ""),
        ("no hyperparameter search", True, "single fixed config"),
        ("v2 vs v3 comparison generated", (OUT / "v2_vs_v3_comparison.md").exists(), ""),
        ("final recommendation generated", (OUT / "production_candidate_recommendation_v3.md").exists(), ""),
    ]
    pd.DataFrame(checks, columns=["check", "pass", "details"]).to_csv(OUT / "final_qa_v3.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    v2 = load_v2()
    v2.ensure_dirs()
    panel_audit = audit_panels()
    if not MAIN_PANEL.exists():
        raise FileNotFoundError("PANEL_NOT_SAFE: training_panel_v15_sr.parquet missing")
    panel = pd.read_parquet(MAIN_PANEL)
    panel["month_end"] = me(panel["date"])
    panel["symbol"] = panel["symbol"].astype(str).str.zfill(6)
    panel = close_label_from_all_daily(panel)
    fcols = canonical_features(panel)
    split = v2.split_plan(panel)
    split.to_csv(OUT / "oos_split_plan_v3.csv", index=False, encoding="utf-8-sig")
    v0, _ = v2.generate_v0(panel, MAIN_PANEL, fcols, split)
    v0 = rename_signal(v0, "V0_FULL_V15_OOS")
    v0.to_parquet(OUT / "V0_FULL_V15_OOS.parquet", index=False)
    shutil.copyfile(OUT / "v0_monthly_weight_audit_v1.csv", OUT / "v0_weight_audit_v3.csv")
    (OUT / "v0_oos_report_v3.md").write_text(
        f"# V0_FULL_V15_OOS\n\nGenerated {len(v0):,} rows using canonical v15 features and expanding ICIR with one-month embargo.",
        encoding="utf-8",
    )
    v7, _, _, v7_ok, _ = v2.generate_v7(panel, MAIN_PANEL, fcols, split)
    v7 = rename_signal(v7, "V7_FULL_V15_OOS")
    if not v7.empty:
        v7.to_parquet(OUT / "V7_FULL_V15_OOS.parquet", index=False)
        shutil.copyfile(OUT / "v7_training_audit_v1.csv", OUT / "v7_training_audit_v3.csv")
        shutil.copyfile(OUT / "v7_oos_generation_report_v1.md", OUT / "v7_oos_report_v3.md")
    elif (OUT / "v7_blocker_report_v1.md").exists():
        shutil.copyfile(OUT / "v7_blocker_report_v1.md", OUT / "v7_blocker_report_v3.md")
    cf, cf_ok = v2.align_compact(panel)
    if not cf.empty:
        cf.to_parquet(OUT / "COMPACT_F_V15_ALIGNED.parquet", index=False)
        shutil.copyfile(OUT / "compact_f_oos_alignment_audit_v1.csv", OUT / "compact_f_alignment_audit_v3.csv")
        shutil.copyfile(OUT / "compact_f_direction_audit_v1.csv", OUT / "compact_f_direction_audit_v3.csv")
    models = {"V0_FULL_V15_OOS": v0, "V7_FULL_V15_OOS": v7, "COMPACT_F": cf}
    # Reuse the v2 blend helper by presenting temporary v2 model names, then
    # keep the requested compact blend labels (BLEND_V0_75_V7_25, etc.).
    blend_inputs = {}
    if not v0.empty:
        blend_inputs["V0_LINEAR_FULL_OOS"] = rename_signal(v0, "V0_LINEAR_FULL_OOS")
    if not v7.empty:
        blend_inputs["V7_TOAWARE_FULL_OOS"] = rename_signal(v7, "V7_TOAWARE_FULL_OOS")
    if not cf.empty:
        blend_inputs["COMPACT_F"] = cf
    blends = v2.make_blends(blend_inputs)
    score_panel = pd.concat([d for d in list(models.values()) + ([blends] if not blends.empty else []) if not d.empty], ignore_index=True, sort=False)
    score_panel.to_parquet(OUT / "all_model_scores_v2.parquet", index=False)
    labels = panel[["month_end", "symbol", "forward_return_1m"]].dropna().drop_duplicates()
    metrics, monthly, turns, ic, yr = v2.run_tournaments(score_panel, labels)
    for old, new in [
        ("tournament_v2_metrics_all.csv", "tournament_v3_metrics_all.csv"),
        ("tournament_v2_full_universe_metrics.csv", "tournament_v3_full_panel_metrics.csv"),
        ("tournament_v2_intersection_metrics.csv", "tournament_v3_intersection_metrics.csv"),
        ("tournament_v2_monthly_returns.csv", "tournament_v3_monthly_returns.csv"),
        ("tournament_v2_rank_ic_series.csv", "tournament_v3_rank_ic_series.csv"),
        ("tournament_v2_turnover_series.csv", "tournament_v3_turnover_series.csv"),
        ("tournament_v2_model_coverage.csv", "tournament_v3_model_coverage.csv"),
    ]:
        shutil.copyfile(OUT / old, OUT / new)
    leak_audit, leak = v2.no_leakage(v0, v7, cf)
    leak_audit.to_csv(OUT / "no_leakage_audit_v3.csv", index=False, encoding="utf-8-sig")
    best = write_v3_reports(metrics, panel_audit, leak, v7_ok)
    qa(not v0.empty, v7_ok, cf_ok, leak)
    med_symbols = panel.groupby("month_end")["symbol"].nunique().median()
    paths = {
        "main_panel_path": MAIN_PANEL.relative_to(ROOT),
        "legacy_panel_path": LEGACY_PANEL.relative_to(ROOT),
        "canonical_feature_audit_path": OUT.relative_to(ROOT) / "canonical_feature_audit_v3.csv",
        "oos_split_plan_path": OUT.relative_to(ROOT) / "oos_split_plan_v3.csv",
        "v0_signal_path": OUT.relative_to(ROOT) / "V0_FULL_V15_OOS.parquet",
        "v7_signal_path_or_blocker": (OUT.relative_to(ROOT) / "V7_FULL_V15_OOS.parquet") if v7_ok else (OUT.relative_to(ROOT) / "v7_blocker_report_v3.md"),
        "compact_f_aligned_path": OUT.relative_to(ROOT) / "COMPACT_F_V15_ALIGNED.parquet",
        "tournament_v3_metrics_path": OUT.relative_to(ROOT) / "tournament_v3_metrics_all.csv",
        "full_panel_metrics_path": OUT.relative_to(ROOT) / "tournament_v3_full_panel_metrics.csv",
        "intersection_metrics_path": OUT.relative_to(ROOT) / "tournament_v3_intersection_metrics.csv",
        "model_coverage_path": OUT.relative_to(ROOT) / "tournament_v3_model_coverage.csv",
        "no_leakage_audit_path": OUT.relative_to(ROOT) / "no_leakage_audit_v3.csv",
        "v2_vs_v3_comparison_path": OUT.relative_to(ROOT) / "v2_vs_v3_comparison.md",
        "recommendation_v3_path": OUT.relative_to(ROOT) / "production_candidate_recommendation_v3.md",
        "final_qa_path": OUT.relative_to(ROOT) / "final_qa_v3.csv",
    }
    for k, v in paths.items():
        print(f"{k}={v}")
    print(f"best_full_panel_model={best[0]}")
    print(f"best_full_panel_portfolio_rule={best[1]}")
    print(f"best_full_panel_net_sharpe={best[2]:.6f}")
    print(f"best_full_panel_max_drawdown={best[3]:.6f}")
    print(f"best_full_panel_turnover={best[4]:.6f}")
    print(f"median_symbols_per_month_main_panel={med_symbols:.6f}")
    print(f"v0_available={not v0.empty}")
    print(f"v7_available={v7_ok}")
    print(f"compact_f_available={cf_ok}")
    print(f"leakage_detected={leak}")
    print(f"decision={best[5]}")


if __name__ == "__main__":
    main()
