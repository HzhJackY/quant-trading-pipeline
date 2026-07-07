from __future__ import annotations

import gc
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "As-Of Industry Neutral Score Evaluation Run v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "asof_industry_neutral_score_evaluation_run_v0"
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME
PREP_DIR = ROOT / "output" / "asof_industry_neutral_score_evaluation_prep_v0"
RAW_EVAL_DIR = ROOT / "output" / "simple_baseline_score_evaluation_run_v0"

PREP_INPUTS = {
    "prep_summary": PREP_DIR / "asof_industry_neutral_score_eval_prep_summary.json",
    "run_config": PREP_DIR / "asof_industry_neutral_score_eval_run_config_draft.json",
    "complete_case_policy": PREP_DIR / "complete_case_policy.json",
    "small_group_policy": PREP_DIR / "small_group_sensitivity_policy.json",
    "comparison_pairs": PREP_DIR / "raw_vs_neutral_comparison_pairs.csv",
    "metric_plan": PREP_DIR / "evaluation_metric_plan.csv",
}
RAW_REFERENCE_CANDIDATES = {
    "raw_summary": [RAW_EVAL_DIR / "simple_baseline_score_evaluation_run_summary.json"],
    "raw_unique_month": [
        RAW_EVAL_DIR / "score_unique_month_aggregate.csv",
        RAW_EVAL_DIR / "unique_month_score_eval_summary.csv",
    ],
    "raw_ranking": [
        RAW_EVAL_DIR / "score_evaluation_final_ranking.csv",
        RAW_EVAL_DIR / "simple_baseline_score_final_ranking.csv",
    ],
}

PANEL_COLUMNS = [
    "symbol",
    "month_end",
    "industry_asof_enddate",
    "selected_industry_system",
    "primary_industry_code",
    "primary_industry_name",
    "industry_join_lag_days",
    "bp_rank",
    "ep_ttm_rank",
    "cfo_to_earnings_parent_rank",
    "VALUE_BP_SINGLE_score",
    "VALUE_QUALITY_EQUAL_WEIGHT_score",
    "ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
    "ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
    "fwd_ret_1m",
    "small_group_flag",
    "industry_join_policy",
]
RAW_SCORES = ["VALUE_BP_SINGLE_score", "VALUE_QUALITY_EQUAL_WEIGHT_score"]
NEUTRAL_SCORES = [
    "ASOF_IND_NEUTRAL_VALUE_BP_SINGLE_score",
    "ASOF_IND_NEUTRAL_VALUE_QUALITY_EQUAL_WEIGHT_score",
]
SCORES = RAW_SCORES + NEUTRAL_SCORES
TARGET = "fwd_ret_1m"


def write_run_state(status: str, details: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "# RUN_STATE",
            "",
            f"- task_name: {TASK_NAME}",
            f"- status: {status}",
            f"- updated_at: {datetime.now().isoformat(timespec='seconds')}",
            f"- output_dir: {OUT_DIR}",
            f"- run_dir: {RUN_DIR}",
            "",
            "## Details",
            "```json",
            json.dumps(details, ensure_ascii=False, indent=2, default=str),
            "```",
            "",
        ]
    )
    (OUT_DIR / "RUN_STATE.md").write_text(text, encoding="utf-8")
    (RUN_DIR / "RUN_STATE.md").write_text(text, encoding="utf-8")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def choose_existing(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def safe_tstat(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) < 2:
        return np.nan
    sd = v.std(ddof=1)
    if sd == 0 or pd.isna(sd):
        return np.nan
    return float(v.mean() / sd * math.sqrt(len(v)))


def safe_ir(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) < 2:
        return np.nan
    sd = v.std(ddof=1)
    if sd == 0 or pd.isna(sd):
        return np.nan
    return float(v.mean() / sd)


def corr_or_nan(x: pd.Series, y: pd.Series, method: str) -> float:
    if len(x) < 2 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return np.nan
    return float(x.corr(y, method=method))


def assign_decile(score: pd.Series) -> pd.Series:
    valid = score.dropna()
    if valid.nunique() < 10 or len(valid) < 10:
        return pd.Series(pd.NA, index=score.index, dtype="Int64")
    try:
        labels = pd.qcut(valid.rank(method="first"), 10, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.Series(pd.NA, index=score.index, dtype="Int64")
    out = pd.Series(pd.NA, index=score.index, dtype="Int64")
    out.loc[labels.index] = labels.astype("Int64")
    if out.dropna().nunique() < 10:
        return pd.Series(pd.NA, index=score.index, dtype="Int64")
    return out


def monotonicity_label(avg_deciles: pd.Series) -> str:
    d = avg_deciles.dropna()
    if len(d) < 10:
        return "INSUFFICIENT_DECILE_DATA"
    vals = d.sort_index().astype(float).values
    diffs = np.diff(vals)
    positive_steps = int((diffs >= 0).sum())
    if vals[-1] <= vals[0] and positive_steps <= 4:
        return "INVERTED"
    if vals[0] > vals[4] and vals[-1] > vals[4]:
        return "U_SHAPED"
    if positive_steps == 9 and vals[-1] > vals[0]:
        return "MONOTONIC_POSITIVE"
    if positive_steps >= 6 and vals[-1] > vals[0]:
        return "MOSTLY_MONOTONIC_POSITIVE"
    return "NON_MONOTONIC"


def evaluate_sample(df: pd.DataFrame, sample_type: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ic_rows = []
    decile_rows = []
    d10_rows = []
    aggregate_rows = []

    for score in SCORES:
        for month, g in df.groupby("month_end", sort=True):
            x = pd.to_numeric(g[score], errors="coerce")
            y = pd.to_numeric(g[TARGET], errors="coerce")
            valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]
            n_obs = int(len(x))
            n_unique_score = int(x.nunique(dropna=True))
            n_unique_target = int(y.nunique(dropna=True))
            sample_flag = "PRIMARY_INCLUDED" if n_obs >= 30 and n_unique_score >= 5 else "LOW_SAMPLE"
            ic_rows.append(
                {
                    "evaluation_sample_type": sample_type,
                    "score": score,
                    "month_end": month,
                    "pearson_ic": corr_or_nan(x, y, "pearson"),
                    "spearman_rank_ic": corr_or_nan(x, y, "spearman"),
                    "n_obs": n_obs,
                    "n_unique_score_values": n_unique_score,
                    "n_unique_target_values": n_unique_target,
                    "sample_flag": sample_flag,
                }
            )

            tmp = g.loc[valid, [score, TARGET]].copy()
            tmp["decile"] = assign_decile(pd.to_numeric(tmp[score], errors="coerce"))
            if tmp["decile"].notna().any() and tmp["decile"].nunique(dropna=True) == 10:
                dec = (
                    tmp.dropna(subset=["decile"])
                    .groupby("decile", dropna=False)[TARGET]
                    .agg(mean_fwd_ret_1m="mean", n_obs="size")
                    .reset_index()
                )
                dec["evaluation_sample_type"] = sample_type
                dec["score"] = score
                dec["month_end"] = month
                dec["decile_status"] = "DECILE_OK"
                decile_rows.extend(dec[["evaluation_sample_type", "score", "month_end", "decile", "mean_fwd_ret_1m", "n_obs", "decile_status"]].to_dict("records"))
                dec_map = dec.set_index("decile")["mean_fwd_ret_1m"]
                d10_d1 = float(dec_map.loc[10] - dec_map.loc[1]) if 10 in dec_map.index and 1 in dec_map.index else np.nan
                d10_rows.append(
                    {
                        "evaluation_sample_type": sample_type,
                        "score": score,
                        "month_end": month,
                        "d10_return": float(dec_map.loc[10]) if 10 in dec_map.index else np.nan,
                        "d1_return": float(dec_map.loc[1]) if 1 in dec_map.index else np.nan,
                        "d10_d1": d10_d1,
                        "decile_status": "DECILE_OK",
                    }
                )
            else:
                for decile in range(1, 11):
                    decile_rows.append(
                        {
                            "evaluation_sample_type": sample_type,
                            "score": score,
                            "month_end": month,
                            "decile": decile,
                            "mean_fwd_ret_1m": np.nan,
                            "n_obs": 0,
                            "decile_status": "DECILE_FAIL_LOW_UNIQUE_VALUES",
                        }
                    )
                d10_rows.append(
                    {
                        "evaluation_sample_type": sample_type,
                        "score": score,
                        "month_end": month,
                        "d10_return": np.nan,
                        "d1_return": np.nan,
                        "d10_d1": np.nan,
                        "decile_status": "DECILE_FAIL_LOW_UNIQUE_VALUES",
                    }
                )

    ic_df = pd.DataFrame(ic_rows)
    decile_df = pd.DataFrame(decile_rows)
    d10_df = pd.DataFrame(d10_rows)

    for score in SCORES:
        score_ic = ic_df[(ic_df["score"] == score) & (ic_df["evaluation_sample_type"] == sample_type)]
        primary_ic = score_ic[score_ic["sample_flag"] == "PRIMARY_INCLUDED"]
        all_ic = score_ic
        score_d10 = d10_df[(d10_df["score"] == score) & (d10_df["evaluation_sample_type"] == sample_type)]
        primary_d10 = score_d10[score_d10["decile_status"] == "DECILE_OK"]
        dec_avg = (
            decile_df[
                (decile_df["score"] == score)
                & (decile_df["evaluation_sample_type"] == sample_type)
                & (decile_df["decile_status"] == "DECILE_OK")
            ]
            .groupby("decile")["mean_fwd_ret_1m"]
            .mean()
        )
        aggregate_rows.append(
            {
                "evaluation_sample_type": sample_type,
                "score": score,
                "score_type": "neutral" if score in NEUTRAL_SCORES else "raw",
                "month_count_primary": int(primary_ic["month_end"].nunique()),
                "month_count_all": int(all_ic["month_end"].nunique()),
                "mean_pearson_ic": float(primary_ic["pearson_ic"].mean()) if not primary_ic.empty else np.nan,
                "mean_spearman_rank_ic": float(primary_ic["spearman_rank_ic"].mean()) if not primary_ic.empty else np.nan,
                "diagnostic_all_months_mean_spearman_rank_ic": float(all_ic["spearman_rank_ic"].mean()) if not all_ic.empty else np.nan,
                "rank_ic_std": float(primary_ic["spearman_rank_ic"].std(ddof=1)) if len(primary_ic) > 1 else np.nan,
                "rank_ic_ir": safe_ir(primary_ic["spearman_rank_ic"]),
                "rank_ic_tstat": safe_tstat(primary_ic["spearman_rank_ic"]),
                "positive_rank_ic_month_ratio": float((primary_ic["spearman_rank_ic"] > 0).mean()) if not primary_ic.empty else np.nan,
                "mean_d10_d1": float(primary_d10["d10_d1"].mean()) if not primary_d10.empty else np.nan,
                "d10_d1_std": float(primary_d10["d10_d1"].std(ddof=1)) if len(primary_d10) > 1 else np.nan,
                "d10_d1_tstat": safe_tstat(primary_d10["d10_d1"]),
                "positive_d10_d1_month_ratio": float((primary_d10["d10_d1"] > 0).mean()) if not primary_d10.empty else np.nan,
                "monotonicity_label": monotonicity_label(dec_avg),
                "old_split_role_diagnostic_available": False,
            }
        )

    return ic_df, decile_df, d10_df, pd.DataFrame(aggregate_rows)


def retention(neutral: float, raw: float) -> float:
    if pd.isna(neutral) or pd.isna(raw) or raw == 0:
        return np.nan
    return float(neutral / raw)


def score_decision(neutral_row: pd.Series, pair_row: pd.Series, sensitivity_flip: bool) -> str:
    rank_ic = neutral_row["mean_spearman_rank_ic"]
    tstat = neutral_row["rank_ic_tstat"]
    pos_ic = neutral_row["positive_rank_ic_month_ratio"]
    d10 = neutral_row["mean_d10_d1"]
    mono = neutral_row["monotonicity_label"]
    raw_rank_ic = pair_row["raw_rank_ic"]
    ret_ratio = pair_row["rank_ic_retention_ratio"]
    neutral_d10 = pair_row["neutral_d10_d1"]
    if pd.notna(rank_ic) and rank_ic > 0 and pd.notna(tstat) and tstat > 1.5 and pd.notna(pos_ic) and pos_ic >= 0.55 and pd.notna(d10) and d10 > 0 and mono != "INVERTED" and not sensitivity_flip:
        return "NEUTRAL_SCORE_STRONG_PASS"
    if pd.notna(rank_ic) and rank_ic > 0 and pd.notna(pos_ic) and pos_ic >= 0.52:
        return "NEUTRAL_SCORE_PARTIAL_PASS"
    if pd.notna(raw_rank_ic) and raw_rank_ic > 0 and (pd.isna(ret_ratio) or ret_ratio < 0.50 or pd.isna(neutral_d10) or neutral_d10 <= 0):
        return "NEUTRAL_SCORE_WATCH_INDUSTRY_DEPENDENT"
    return "NEUTRAL_SCORE_FAIL"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    write_run_state("running", {"step": "start"})

    raw_refs = {k: choose_existing(v) for k, v in RAW_REFERENCE_CANDIDATES.items()}
    prereq = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "prep_inputs": {k: str(v) for k, v in PREP_INPUTS.items()},
        "prep_exists": {k: v.exists() for k, v in PREP_INPUTS.items()},
        "raw_reference_paths": {k: None if v is None else str(v) for k, v in raw_refs.items()},
    }
    prereq["prerequisites_passed"] = all(prereq["prep_exists"].values()) and all(v is not None for v in raw_refs.values())
    write_json(OUT_DIR / "asof_industry_neutral_eval_prerequisite_check.json", prereq)
    if not prereq["prerequisites_passed"]:
        write_run_state("failed", prereq)
        raise SystemExit("Missing required inputs")

    prep_summary = json.loads(PREP_INPUTS["prep_summary"].read_text(encoding="utf-8"))
    run_config = json.loads(PREP_INPUTS["run_config"].read_text(encoding="utf-8"))
    complete_case_policy = json.loads(PREP_INPUTS["complete_case_policy"].read_text(encoding="utf-8"))
    small_group_policy = json.loads(PREP_INPUTS["small_group_policy"].read_text(encoding="utf-8"))
    pairs = pd.read_csv(PREP_INPUTS["comparison_pairs"])
    metric_plan = pd.read_csv(PREP_INPUTS["metric_plan"])
    input_panel_path = Path(run_config["neutral_score_panel_path"])

    write_run_state("running", {"step": "read_panel_necessary_columns", "panel": str(input_panel_path)})
    panel = pd.read_parquet(input_panel_path, columns=PANEL_COLUMNS)
    input_rows = int(len(panel))
    panel["month_end"] = pd.to_datetime(panel["month_end"], errors="coerce")
    panel["industry_asof_enddate"] = pd.to_datetime(panel["industry_asof_enddate"], errors="coerce")
    panel["symbol"] = panel["symbol"].astype("string").str.strip()
    panel["primary_industry_code"] = panel["primary_industry_code"].astype("string").str.strip()
    for col in SCORES + [TARGET]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel["small_group_flag"] = panel["small_group_flag"].astype(bool)

    reason_masks = {
        "symbol_missing": panel["symbol"].isna() | (panel["symbol"] == ""),
        "month_end_missing": panel["month_end"].isna(),
        "raw_score_missing_or_inf": panel[RAW_SCORES].isna().any(axis=1) | np.isinf(panel[RAW_SCORES]).any(axis=1),
        "neutral_score_missing_or_inf": panel[NEUTRAL_SCORES].isna().any(axis=1) | np.isinf(panel[NEUTRAL_SCORES]).any(axis=1),
        "fwd_ret_1m_missing_or_inf": panel[TARGET].isna() | np.isinf(panel[TARGET]),
        "primary_industry_code_missing": panel["primary_industry_code"].isna() | (panel["primary_industry_code"] == ""),
        "industry_asof_enddate_missing": panel["industry_asof_enddate"].isna(),
        "future_enddate_violation": panel["industry_asof_enddate"] > panel["month_end"],
    }
    keep = pd.Series(True, index=panel.index)
    for mask in reason_masks.values():
        keep &= ~mask.fillna(False)
    cc = panel[keep].copy()
    future_violations = int(reason_masks["future_enddate_violation"].fillna(False).sum())
    reason_counts = {k: int(v.fillna(False).sum()) for k, v in reason_masks.items()}
    expected_rows = int(complete_case_policy.get("expected_complete_case_rows_from_previous_step", prep_summary.get("expected_complete_case_rows", 0)))
    complete_case_qa = pd.DataFrame(
        [
            {
                "input_rows": input_rows,
                "expected_complete_case_rows_from_prep": expected_rows,
                "actual_complete_case_rows": int(len(cc)),
                "dropped_rows": input_rows - int(len(cc)),
                "dropped_row_reason_counts": json.dumps(reason_counts, ensure_ascii=False),
                "unique_month_count": int(cc["month_end"].nunique()),
                "unique_symbol_count": int(cc["symbol"].nunique()),
                "future_enddate_violation_count": future_violations,
            }
        ]
    )
    complete_case_qa.to_csv(OUT_DIR / "complete_case_sample_qa.csv", index=False, encoding="utf-8-sig")

    if future_violations > 0:
        final_decision = "ASOF_INDUSTRY_NEUTRAL_SCORE_EVAL_RUN_FAIL_GUARDRAIL"
        ic_df = pd.DataFrame()
        decile_df = pd.DataFrame()
        d10_df = pd.DataFrame()
        aggregate = pd.DataFrame()
        sensitivity = pd.DataFrame()
        comparison = pd.DataFrame()
        decisions = pd.DataFrame()
    else:
        write_run_state("running", {"step": "evaluate_base_complete_case", "rows": len(cc)})
        base_ic, base_decile, base_d10, base_agg = evaluate_sample(cc, "BASE_ALL_COMPLETE_CASE")
        no_small = cc[~cc["small_group_flag"]].copy()
        write_run_state("running", {"step": "evaluate_exclude_small_group", "rows": len(no_small)})
        sens_ic, sens_decile, sens_d10, sens_agg = evaluate_sample(no_small, "SENSITIVITY_EXCLUDE_SMALL_GROUP")
        ic_df = pd.concat([base_ic, sens_ic], ignore_index=True)
        decile_df = pd.concat([base_decile, sens_decile], ignore_index=True)
        d10_df = pd.concat([base_d10, sens_d10], ignore_index=True)
        aggregate = pd.concat([base_agg, sens_agg], ignore_index=True)

        base_map = base_agg.set_index("score")
        sens_map = sens_agg.set_index("score")
        sens_rows = []
        for score in SCORES:
            b = base_map.loc[score]
            s = sens_map.loc[score]
            rank_flip = pd.notna(b["mean_spearman_rank_ic"]) and pd.notna(s["mean_spearman_rank_ic"]) and np.sign(b["mean_spearman_rank_ic"]) != np.sign(s["mean_spearman_rank_ic"])
            d10_flip = pd.notna(b["mean_d10_d1"]) and pd.notna(s["mean_d10_d1"]) and np.sign(b["mean_d10_d1"]) != np.sign(s["mean_d10_d1"])
            sens_rows.append(
                {
                    "score": score,
                    "base_rank_ic": b["mean_spearman_rank_ic"],
                    "exclude_small_rank_ic": s["mean_spearman_rank_ic"],
                    "base_d10_d1": b["mean_d10_d1"],
                    "exclude_small_d10_d1": s["mean_d10_d1"],
                    "rank_ic_direction_flip": bool(rank_flip),
                    "d10_d1_direction_flip": bool(d10_flip),
                    "small_group_sensitivity_label": "SMALL_GROUP_SENSITIVITY_RISK" if rank_flip or d10_flip else "NO_DIRECTION_FLIP",
                }
            )
        sensitivity = pd.DataFrame(sens_rows)

        comp_rows = []
        for _, pair in pairs.iterrows():
            raw = pair["raw_score"]
            neutral = pair["neutral_score"]
            raw_row = base_map.loc[raw]
            neutral_row = base_map.loc[neutral]
            rank_ret = retention(neutral_row["mean_spearman_rank_ic"], raw_row["mean_spearman_rank_ic"])
            d10_ret = retention(neutral_row["mean_d10_d1"], raw_row["mean_d10_d1"])
            if pd.notna(rank_ret) and neutral_row["mean_spearman_rank_ic"] > raw_row["mean_spearman_rank_ic"]:
                interpretation = "NEUTRAL_STRONGER_THAN_RAW"
            elif pd.notna(neutral_row["mean_spearman_rank_ic"]) and neutral_row["mean_spearman_rank_ic"] > 0 and pd.notna(rank_ret) and rank_ret >= 0.50:
                interpretation = "STOCK_SELECTION_SIGNAL_RETAINED_AFTER_NEUTRALIZATION"
            elif pd.notna(raw_row["mean_spearman_rank_ic"]) and raw_row["mean_spearman_rank_ic"] > 0 and (pd.isna(neutral_row["mean_spearman_rank_ic"]) or neutral_row["mean_spearman_rank_ic"] <= 0 or pd.isna(rank_ret) or rank_ret < 0.50):
                interpretation = "RAW_SIGNAL_LIKELY_INDUSTRY_EXPOSURE_DEPENDENT"
            else:
                interpretation = "RAW_AND_NEUTRAL_WEAK"
            comp_rows.append(
                {
                    "comparison_pair": pair["comparison_pair_name"],
                    "raw_score": raw,
                    "neutral_score": neutral,
                    "raw_rank_ic": raw_row["mean_spearman_rank_ic"],
                    "neutral_rank_ic": neutral_row["mean_spearman_rank_ic"],
                    "rank_ic_retention_ratio": rank_ret,
                    "raw_d10_d1": raw_row["mean_d10_d1"],
                    "neutral_d10_d1": neutral_row["mean_d10_d1"],
                    "d10_d1_retention_ratio": d10_ret,
                    "raw_positive_ic_ratio": raw_row["positive_rank_ic_month_ratio"],
                    "neutral_positive_ic_ratio": neutral_row["positive_rank_ic_month_ratio"],
                    "raw_monotonicity": raw_row["monotonicity_label"],
                    "neutral_monotonicity": neutral_row["monotonicity_label"],
                    "interpretation": interpretation,
                }
            )
        comparison = pd.DataFrame(comp_rows)

        decision_rows = []
        for _, comp in comparison.iterrows():
            neutral = comp["neutral_score"]
            neutral_row = base_map.loc[neutral]
            sens_flip = bool(
                sensitivity.loc[
                    sensitivity["score"].eq(neutral),
                    ["rank_ic_direction_flip", "d10_d1_direction_flip"],
                ]
                .any(axis=None)
            )
            decision = score_decision(neutral_row, comp, sens_flip)
            decision_rows.append(
                {
                    "neutral_score": neutral,
                    "raw_pair_score": comp["raw_score"],
                    "mean_spearman_rank_ic": neutral_row["mean_spearman_rank_ic"],
                    "rank_ic_tstat": neutral_row["rank_ic_tstat"],
                    "positive_rank_ic_month_ratio": neutral_row["positive_rank_ic_month_ratio"],
                    "mean_d10_d1": neutral_row["mean_d10_d1"],
                    "positive_d10_d1_month_ratio": neutral_row["positive_d10_d1_month_ratio"],
                    "monotonicity_label": neutral_row["monotonicity_label"],
                    "small_group_sensitivity_risk": sens_flip,
                    "rank_ic_retention_ratio": comp["rank_ic_retention_ratio"],
                    "score_decision": decision,
                }
            )
        decisions = pd.DataFrame(decision_rows)
        pass_count = int(decisions["score_decision"].isin(["NEUTRAL_SCORE_STRONG_PASS", "NEUTRAL_SCORE_PARTIAL_PASS"]).sum())
        watch_count = int(decisions["score_decision"].eq("NEUTRAL_SCORE_WATCH_INDUSTRY_DEPENDENT").sum())
        if pass_count > 0:
            final_decision = "ASOF_INDUSTRY_NEUTRAL_SCORE_EVAL_RUN_ALPHA_SURVIVES_READY_FOR_PORTFOLIO_PREP"
        elif watch_count > 0:
            final_decision = "ASOF_INDUSTRY_NEUTRAL_SCORE_EVAL_RUN_WATCH_INDUSTRY_DEPENDENT"
        else:
            final_decision = "ASOF_INDUSTRY_NEUTRAL_SCORE_EVAL_RUN_FAIL_NEUTRAL_ALPHA_NOT_SUPPORTED"

    ic_df.to_csv(OUT_DIR / "monthly_ic_by_score.csv", index=False, encoding="utf-8-sig")
    decile_df.to_csv(OUT_DIR / "monthly_decile_return_by_score.csv", index=False, encoding="utf-8-sig")
    d10_df.to_csv(OUT_DIR / "monthly_d10_d1_by_score.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(OUT_DIR / "unique_month_score_aggregate.csv", index=False, encoding="utf-8-sig")
    shape = aggregate[["evaluation_sample_type", "score", "monotonicity_label"]].copy() if not aggregate.empty else pd.DataFrame(columns=["evaluation_sample_type", "score", "monotonicity_label"])
    shape.to_csv(OUT_DIR / "score_decile_shape_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT_DIR / "raw_vs_neutral_comparison.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUT_DIR / "small_group_sensitivity_comparison.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(OUT_DIR / "neutral_score_decision_matrix.csv", index=False, encoding="utf-8-sig")

    small_risk = bool((sensitivity.get("small_group_sensitivity_label", pd.Series(dtype=str)) == "SMALL_GROUP_SENSITIVITY_RISK").any()) if not sensitivity.empty else False
    guardrail = pd.DataFrame(
        [
            {"guardrail": "prerequisites_passed", "passed": bool(prereq["prerequisites_passed"]), "notes": ""},
            {"guardrail": "future_enddate_violation_count_zero", "passed": future_violations == 0, "notes": str(future_violations)},
            {"guardrail": "fwd_ret_only_used_as_target", "passed": True, "notes": ""},
            {"guardrail": "portfolio_not_constructed", "passed": True, "notes": ""},
            {"guardrail": "portfolio_return_not_calculated", "passed": True, "notes": ""},
            {"guardrail": "backtest_not_run", "passed": True, "notes": ""},
            {"guardrail": "production_not_modified", "passed": True, "notes": ""},
        ]
    )
    guardrail.to_csv(OUT_DIR / "guardrail_qa.csv", index=False, encoding="utf-8-sig")

    base_agg = aggregate[aggregate["evaluation_sample_type"].eq("BASE_ALL_COMPLETE_CASE")].copy() if not aggregate.empty else pd.DataFrame()
    raw_agg = base_agg[base_agg["score_type"].eq("raw")].copy() if not base_agg.empty else pd.DataFrame()
    neutral_agg = base_agg[base_agg["score_type"].eq("neutral")].copy() if not base_agg.empty else pd.DataFrame()
    best_raw = raw_agg.sort_values("mean_spearman_rank_ic", ascending=False).iloc[0] if not raw_agg.empty else pd.Series(dtype=object)
    best_neutral = neutral_agg.sort_values("mean_spearman_rank_ic", ascending=False).iloc[0] if not neutral_agg.empty else pd.Series(dtype=object)
    best_neutral_decision = (
        decisions.loc[decisions["neutral_score"].eq(best_neutral.get("score")), "score_decision"].iloc[0]
        if not decisions.empty and "score" in best_neutral.index and decisions["neutral_score"].eq(best_neutral.get("score")).any()
        else None
    )
    bp_ret = (
        comparison.loc[comparison["comparison_pair"].str.contains("BP_SINGLE", na=False), "rank_ic_retention_ratio"].iloc[0]
        if not comparison.empty and comparison["comparison_pair"].str.contains("BP_SINGLE", na=False).any()
        else np.nan
    )
    vq_ret = (
        comparison.loc[comparison["comparison_pair"].str.contains("VALUE_QUALITY", na=False), "rank_ic_retention_ratio"].iloc[0]
        if not comparison.empty and comparison["comparison_pair"].str.contains("VALUE_QUALITY", na=False).any()
        else np.nan
    )
    strong_count = int(decisions["score_decision"].eq("NEUTRAL_SCORE_STRONG_PASS").sum()) if not decisions.empty else 0
    partial_count = int(decisions["score_decision"].eq("NEUTRAL_SCORE_PARTIAL_PASS").sum()) if not decisions.empty else 0
    watch_count = int(decisions["score_decision"].eq("NEUTRAL_SCORE_WATCH_INDUSTRY_DEPENDENT").sum()) if not decisions.empty else 0
    fail_count = int(decisions["score_decision"].eq("NEUTRAL_SCORE_FAIL").sum()) if not decisions.empty else 0
    industry_supported = strong_count + partial_count > 0 and future_violations == 0
    industry_dependent = bool(watch_count > 0 or (
        not comparison.empty and comparison["interpretation"].eq("RAW_SIGNAL_LIKELY_INDUSTRY_EXPOSURE_DEPENDENT").any()
    ))

    summary = {
        "run_timestamp": prereq["run_timestamp"],
        "prerequisites_passed": bool(prereq["prerequisites_passed"]),
        "input_panel_path": str(input_panel_path),
        "input_rows": input_rows,
        "expected_complete_case_rows": expected_rows,
        "actual_complete_case_rows": int(len(cc)),
        "unique_month_count": int(cc["month_end"].nunique()),
        "unique_symbol_count": int(cc["symbol"].nunique()),
        "future_enddate_violation_count": future_violations,
        "score_count_evaluated": len(SCORES),
        "raw_score_count": len(RAW_SCORES),
        "neutral_score_count": len(NEUTRAL_SCORES),
        "comparison_pair_count": int(len(pairs)),
        "best_raw_score_by_rank_ic": best_raw.get("score"),
        "best_neutral_score_by_rank_ic": best_neutral.get("score"),
        "best_neutral_rank_ic": best_neutral.get("mean_spearman_rank_ic"),
        "best_neutral_rank_ic_tstat": best_neutral.get("rank_ic_tstat"),
        "best_neutral_positive_rank_ic_month_ratio": best_neutral.get("positive_rank_ic_month_ratio"),
        "best_neutral_d10_d1": best_neutral.get("mean_d10_d1"),
        "best_neutral_positive_d10_d1_month_ratio": best_neutral.get("positive_d10_d1_month_ratio"),
        "best_neutral_score_decision": best_neutral_decision,
        "bp_pair_retention_ratio_rank_ic": bp_ret,
        "value_quality_pair_retention_ratio_rank_ic": vq_ret,
        "small_group_sensitivity_risk_detected": small_risk,
        "neutral_score_strong_pass_count": strong_count,
        "neutral_score_partial_pass_count": partial_count,
        "neutral_score_watch_count": watch_count,
        "neutral_score_fail_count": fail_count,
        "industry_neutral_alpha_supported": industry_supported,
        "industry_exposure_dependency_detected": industry_dependent,
        "ic_calculated": True,
        "d10_d1_calculated": True,
        "decile_return_calculated": True,
        "portfolio_constructed": False,
        "portfolio_return_calculated": False,
        "backtest_run": False,
        "transaction_cost_calculated": False,
        "turnover_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "benchmark_relative_return_calculated": False,
        "alpha_beta_regression_calculated": False,
        "training_run": False,
        "shap_calculated": False,
        "tuning_run": False,
        "feature_importance_calculated": False,
        "production_holdings_generated": False,
        "live_order_ready_file_generated": False,
        "production_modified": False,
        "final_decision": final_decision,
        "recommended_next_step": "若 final_decision 为 alpha survives，可进入 industry-neutral portfolio construction prep；仍需单独 guardrail，不能直接回测或写 production。",
    }
    write_json(OUT_DIR / "asof_industry_neutral_score_evaluation_summary.json", summary)

    report = f"""# As-Of Industry Neutral Score Evaluation Run v0

## 结论

- final_decision: `{final_decision}`
- best_neutral_score_by_rank_ic: `{summary['best_neutral_score_by_rank_ic']}`
- best_neutral_rank_ic: `{summary['best_neutral_rank_ic']}`
- best_neutral_d10_d1: `{summary['best_neutral_d10_d1']}`
- small_group_sensitivity_risk_detected: `{small_risk}`

本任务只完成 score evaluation：IC、decile return、D10-D1、raw vs neutral comparison 与 small-group sensitivity。未构造 portfolio，未计算 portfolio return，未回测，未写 production。
"""
    (OUT_DIR / "asof_industry_neutral_score_evaluation_report.md").write_text(report, encoding="utf-8")

    final_qa = pd.DataFrame(
        [
            {"check": "portfolio_constructed", "value": False, "passed": True},
            {"check": "portfolio_return_calculated", "value": False, "passed": True},
            {"check": "backtest_run", "value": False, "passed": True},
            {"check": "transaction_cost_calculated", "value": False, "passed": True},
            {"check": "turnover_calculated", "value": False, "passed": True},
            {"check": "production_modified", "value": False, "passed": True},
            {"check": "future_enddate_violation_count", "value": future_violations, "passed": future_violations == 0},
        ]
    )
    final_qa.to_csv(OUT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    terminal_summary = {
        "task_name": TASK_NAME,
        "status": "completed",
        "stdout_log": str(RUN_DIR / "run_stdout.txt"),
        "stderr_log": str(RUN_DIR / "run_stderr.txt"),
        "outputs": [str(p) for p in sorted(OUT_DIR.glob("*")) if p.is_file()],
    }
    write_json(OUT_DIR / "terminal_summary.json", terminal_summary)
    card = f"""# Task Completion Card

- task_name: {TASK_NAME}
- status: completed
- final_decision: {final_decision}
- output_dir: {OUT_DIR}
- run_dir: {RUN_DIR}
- logs: {RUN_DIR / 'run_stdout.txt'} ; {RUN_DIR / 'run_stderr.txt'}
"""
    (OUT_DIR / "task_completion_card.md").write_text(card, encoding="utf-8")
    write_run_state("completed", {"final_decision": final_decision, "summary_path": str(OUT_DIR / "asof_industry_neutral_score_evaluation_summary.json")})

    del panel, cc, pairs, metric_plan, aggregate, comparison, sensitivity, decisions
    gc.collect()
    print(json.dumps({"status": "completed", "final_decision": final_decision, "output_dir": str(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
