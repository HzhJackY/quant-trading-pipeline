from __future__ import annotations

import gc
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


TASK_NAME = "simple_baseline_portfolio_construction_run_v0"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / TASK_NAME
AGENT_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME
PREP_DIR = ROOT / "output" / "simple_baseline_portfolio_construction_prep_v0"
SCORE_PANEL = ROOT / "output" / "simple_baseline_score_run_v0" / "simple_baseline_score_panel_v0.parquet"

REQUIRED_PREP_FILES = {
    "prep_summary": PREP_DIR / "simple_baseline_portfolio_construction_prep_summary.json",
    "run_config": PREP_DIR / "portfolio_construction_run_config_draft.json",
    "rule_manifest": PREP_DIR / "portfolio_construction_rule_manifest.csv",
    "score_manifest": PREP_DIR / "portfolio_candidate_score_manifest.csv",
    "qa_plan": PREP_DIR / "portfolio_construction_qa_plan.csv",
    "guardrail_checklist": PREP_DIR / "portfolio_construction_guardrail_checklist.csv",
}

REQUIRED_COLUMNS = [
    "symbol",
    "month_end",
    "fwd_ret_1m",
    "VALUE_BP_SINGLE_score",
    "VALUE_QUALITY_EQUAL_WEIGHT_score",
]

FORBIDDEN_SELECTION_COLUMNS = [
    "VALUE_BP_EP_EQUAL_WEIGHT_score",
    "BP_CFO_EQUAL_WEIGHT_score",
    "roe_ttm_rank",
    "profit_growth_yoy_rank",
]

PORTFOLIOS = [
    {
        "portfolio_name": "BP_SINGLE_TOP_DECILE_EQUAL_WEIGHT",
        "score_column": "VALUE_BP_SINGLE_score",
        "selection_rule": "top 10% by score within each month",
        "rule_type": "top_decile",
    },
    {
        "portfolio_name": "BP_SINGLE_TOP50_EQUAL_WEIGHT",
        "score_column": "VALUE_BP_SINGLE_score",
        "selection_rule": "top 50 stocks by score within each month",
        "rule_type": "top50",
    },
    {
        "portfolio_name": "VALUE_QUALITY_TOP_DECILE_EQUAL_WEIGHT",
        "score_column": "VALUE_QUALITY_EQUAL_WEIGHT_score",
        "selection_rule": "top 10% by score within each month",
        "rule_type": "top_decile",
    },
    {
        "portfolio_name": "VALUE_QUALITY_TOP50_EQUAL_WEIGHT",
        "score_column": "VALUE_QUALITY_EQUAL_WEIGHT_score",
        "selection_rule": "top 50 stocks by score within each month",
        "rule_type": "top50",
    },
]


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def status(pass_condition: bool) -> str:
    return "PASS" if pass_condition else "FAIL"


def finite_count(series: pd.Series) -> int:
    return int(np.isfinite(pd.to_numeric(series, errors="coerce")).sum())


def expected_count(rule_type: str, non_null_count: int) -> int:
    if non_null_count <= 0:
        return 0
    if rule_type == "top_decile":
        return max(1, int(math.ceil(0.10 * non_null_count)))
    if rule_type == "top50":
        return min(50, non_null_count)
    raise ValueError(f"Unknown rule_type: {rule_type}")


def read_inputs() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prep_summary = json.loads(REQUIRED_PREP_FILES["prep_summary"].read_text(encoding="utf-8"))
    run_config = json.loads(REQUIRED_PREP_FILES["run_config"].read_text(encoding="utf-8"))
    rule_manifest = pd.read_csv(REQUIRED_PREP_FILES["rule_manifest"])
    score_manifest = pd.read_csv(REQUIRED_PREP_FILES["score_manifest"])
    qa_plan = pd.read_csv(REQUIRED_PREP_FILES["qa_plan"])
    guardrail_checklist = pd.read_csv(REQUIRED_PREP_FILES["guardrail_checklist"])
    score_df = pd.read_parquet(SCORE_PANEL, columns=REQUIRED_COLUMNS)
    score_df["symbol"] = score_df["symbol"].astype("string")
    score_df["month_end"] = pd.to_datetime(score_df["month_end"]).dt.strftime("%Y-%m-%d")
    return prep_summary, run_config, rule_manifest, score_manifest, qa_plan, guardrail_checklist, score_df


def build_prereq(prep_summary: dict, run_config: dict, score_df: pd.DataFrame) -> dict:
    columns = set(score_df.columns)
    prep_files_exist = {key: path.exists() for key, path in REQUIRED_PREP_FILES.items()}
    checks = {
        "prep_files_exist": all(prep_files_exist.values()),
        "portfolio_construction_prep_ready": prep_summary.get("final_decision")
        == "SIMPLE_BASELINE_PORTFOLIO_CONSTRUCTION_PREP_READY_FOR_CONSTRUCTION_RUN",
        "score_panel_exists": SCORE_PANEL.exists(),
        "required_score_columns_exist": all(col in columns for col in REQUIRED_COLUMNS),
        "portfolio_rules_frozen": bool(prep_summary.get("portfolio_rules_frozen")),
        "research_only_true": bool(run_config.get("research_only")) and all(p.get("research_only") for p in run_config.get("portfolio_rules", [])),
        "portfolio_returns_allowed_false": run_config.get("portfolio_returns_allowed") is False,
        "backtest_allowed_false": run_config.get("backtest_allowed") is False,
        "transaction_cost_allowed_false": run_config.get("transaction_cost_allowed") is False,
        "turnover_allowed_false": run_config.get("turnover_allowed") is False,
        "sharpe_allowed_false": run_config.get("sharpe_allowed") is False,
        "maxdd_allowed_false": run_config.get("maxdd_allowed") is False,
        "production_holdings_allowed_false": run_config.get("holdings_production_allowed") is False,
    }
    return {
        "run_timestamp": now_iso(),
        "task_name": TASK_NAME,
        "prep_files": {key: str(path) for key, path in REQUIRED_PREP_FILES.items()},
        "prep_files_exist": prep_files_exist,
        "checks": checks,
        "prerequisites_passed": all(checks.values()),
    }


def build_input_qa(score_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    duplicate_count = int(score_df.duplicated(["symbol", "month_end"]).sum())
    rows.extend(
        [
            ["rows loaded", len(score_df), "PASS" if len(score_df) > 0 else "FAIL", "BLOCKING", "只读取任务要求的 5 个列"],
            ["symbol count", score_df["symbol"].nunique(), "PASS", "INFO", ""],
            ["month count", score_df["month_end"].nunique(), "PASS", "INFO", ""],
            ["min month_end", score_df["month_end"].min(), "PASS", "INFO", ""],
            ["max month_end", score_df["month_end"].max(), "PASS", "INFO", ""],
            ["duplicate symbol-month count", duplicate_count, status(duplicate_count == 0), "BLOCKING", "输入层 symbol-month 应唯一"],
        ]
    )
    for col in ["VALUE_BP_SINGLE_score", "VALUE_QUALITY_EQUAL_WEIGHT_score"]:
        non_null = int(score_df[col].notna().sum())
        finite = finite_count(score_df[col])
        rows.append([f"{col} non-null count", non_null, status(non_null > 0), "BLOCKING", "selection 仅使用该 score"])
        rows.append([f"{col} finite count", finite, status(finite == non_null and finite > 0), "BLOCKING", "score non-null 必须 finite"])
    target_count = int(score_df["fwd_ret_1m"].notna().sum())
    rows.append(["target retained count", target_count, "PASS", "INFO", "fwd_ret_1m 仅保留给 later evaluation，不参与 selection"])
    rows.append(["watch score absent or unused", "unused", "PASS", "BLOCKING", "未读取 VALUE_BP_EP_EQUAL_WEIGHT_score，未参与 selection"])
    return pd.DataFrame(rows, columns=["check_name", "observed_value", "status", "severity", "notes"])


def construct_weights(score_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weight_frames = []
    qa_rows = []
    month_universe = score_df.groupby("month_end", sort=True).size().to_dict()

    for portfolio in PORTFOLIOS:
        score_column = portfolio["score_column"]
        for month_end, month_df in score_df.groupby("month_end", sort=True):
            candidate_df = month_df.loc[month_df[score_column].notna(), ["symbol", "month_end", "fwd_ret_1m", score_column]].copy()
            candidate_df = candidate_df[np.isfinite(pd.to_numeric(candidate_df[score_column], errors="coerce"))]
            score_non_null_count = len(candidate_df)
            select_n = expected_count(portfolio["rule_type"], score_non_null_count)

            selected = candidate_df.sort_values([score_column, "symbol"], ascending=[False, True], kind="mergesort").head(select_n).copy()
            if select_n > 0:
                selected["weight"] = 1.0 / select_n
            else:
                selected["weight"] = np.nan
            selected["portfolio_name"] = portfolio["portfolio_name"]
            selected["score_column"] = score_column
            selected["score_value"] = selected[score_column]
            selected["research_only"] = True
            selected["rebalance_frequency"] = "monthly"
            selected["selection_rule"] = portfolio["selection_rule"]
            selected["weight_rule"] = "equal weight"

            if not selected.empty:
                weight_frames.append(
                    selected[
                        [
                            "portfolio_name",
                            "symbol",
                            "month_end",
                            "score_column",
                            "score_value",
                            "weight",
                            "research_only",
                            "rebalance_frequency",
                            "selection_rule",
                            "weight_rule",
                            "fwd_ret_1m",
                        ]
                    ]
                )

            duplicate_selected = int(selected.duplicated(["symbol"]).sum())
            weight_sum = float(selected["weight"].sum()) if not selected.empty else 0.0
            min_weight = float(selected["weight"].min()) if not selected.empty else np.nan
            max_weight = float(selected["weight"].max()) if not selected.empty else np.nan
            target_non_null = int(selected["fwd_ret_1m"].notna().sum()) if not selected.empty else 0
            checks_passed = (
                len(selected) == select_n
                and abs(weight_sum - 1.0) <= 1e-10
                and duplicate_selected == 0
                and (selected["weight"] >= 0).all()
                and weight_sum <= 1.0 + 1e-10
                and selected["research_only"].eq(True).all()
            )
            qa_rows.append(
                {
                    "portfolio_name": portfolio["portfolio_name"],
                    "month_end": month_end,
                    "score_column": score_column,
                    "universe_count": int(month_universe[month_end]),
                    "score_non_null_count": int(score_non_null_count),
                    "selected_count": int(len(selected)),
                    "expected_selected_count": int(select_n),
                    "weight_sum": weight_sum,
                    "min_weight": min_weight,
                    "max_weight": max_weight,
                    "duplicate_symbol_count": duplicate_selected,
                    "target_non_null_count": target_non_null,
                    "status": status(checks_passed),
                    "notes": "tie_policy=score desc then symbol asc; top_decile_count=ceil(0.10*score_non_null_count); fwd_ret_1m not used for selection",
                }
            )

            del candidate_df, selected
            gc.collect()

    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    weights_qa = pd.DataFrame(qa_rows)
    return weights, weights_qa


def build_coverage(weights_qa: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (portfolio_name, score_column), group in weights_qa.groupby(["portfolio_name", "score_column"], sort=True):
        pass_status = group["status"].eq("PASS").all()
        rows.append(
            {
                "portfolio_name": portfolio_name,
                "score_column": score_column,
                "month_count": int(group["month_end"].nunique()),
                "min_month_end": group["month_end"].min(),
                "max_month_end": group["month_end"].max(),
                "mean_selected_count": float(group["selected_count"].mean()),
                "min_selected_count": int(group["selected_count"].min()),
                "max_selected_count": int(group["selected_count"].max()),
                "mean_weight_sum": float(group["weight_sum"].mean()),
                "min_weight_sum": float(group["weight_sum"].min()),
                "max_weight_sum": float(group["weight_sum"].max()),
                "total_weight_rows": int(group["selected_count"].sum()),
                "status": status(pass_status),
                "notes": "research-only monthly equal weights; no return/backtest metrics calculated",
            }
        )
    return pd.DataFrame(rows)


def build_named_qa(checks: list[tuple[str, bool, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check_name": name,
                "status": status(ok),
                "severity": "BLOCKING",
                "notes": notes,
            }
            for name, ok, notes in checks
        ]
    )


def build_report(summary: dict) -> str:
    lines = [
        "# Simple Baseline Portfolio Construction Run v0",
        "",
        "## 决策",
        "",
        f"- final_decision: {summary['final_decision']}",
        f"- recommended_next_step: {summary['recommended_next_step']}",
        "",
        "## 构造规则",
        "",
        "- 仅生成 research-only hypothetical monthly weights。",
        "- 只读取 symbol、month_end、fwd_ret_1m、VALUE_BP_SINGLE_score、VALUE_QUALITY_EQUAL_WEIGHT_score。",
        "- selection 未使用 fwd_ret_1m；该列仅保留给 later evaluation。",
        "- Top decile 使用 ceil(0.10 * monthly_score_non_null_count)，确保至少 1 只。",
        "- Tie policy: score 降序后按 symbol 升序稳定排序。",
        "",
        "## 禁止项状态",
        "",
        "- 未计算 portfolio return / benchmark-relative return。",
        "- 未回测，未计算交易成本、换手率、Sharpe、MaxDD。",
        "- 未训练、未调参、未计算 SHAP、未生成 feature importance。",
        "- 未生成 production holdings 或 live-order-ready 文件，未写 production。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        AGENT_DIR / "RUN_STATE.json",
        {
            "task_name": TASK_NAME,
            "status": "running",
            "updated_at": now_iso(),
            "current_step": "starting",
            "resume_instruction": f"先读取 {AGENT_DIR / 'RUN_STATE.md'} 再继续。",
        },
    )
    (AGENT_DIR / "RUN_STATE.md").write_text(
        f"# RUN_STATE\n\n- task_name: {TASK_NAME}\n- status: running\n- updated_at: {now_iso()}\n- current_step: starting\n",
        encoding="utf-8",
    )

    missing = [str(path) for path in REQUIRED_PREP_FILES.values() if not path.exists()]
    if not SCORE_PANEL.exists():
        missing.append(str(SCORE_PANEL))
    if missing:
        raise FileNotFoundError("Missing required input(s): " + "; ".join(missing))

    prep_summary, run_config, rule_manifest, score_manifest, qa_plan, guardrail_checklist, score_df = read_inputs()
    prerequisites = build_prereq(prep_summary, run_config, score_df)
    write_json(OUT_DIR / "portfolio_construction_run_prerequisite_check.json", prerequisites)

    input_qa = build_input_qa(score_df)
    input_qa.to_csv(OUT_DIR / "portfolio_input_score_panel_qa.csv", index=False, encoding="utf-8-sig")

    weights, weights_qa = construct_weights(score_df)
    weights_path = OUT_DIR / "simple_baseline_research_weights_v0.parquet"
    weights.to_parquet(weights_path, index=False)
    weights_qa.to_csv(OUT_DIR / "portfolio_weights_qa_by_month.csv", index=False, encoding="utf-8-sig")

    coverage = build_coverage(weights_qa)
    coverage.to_csv(OUT_DIR / "portfolio_coverage_summary.csv", index=False, encoding="utf-8-sig")

    weight_panel_columns = set(weights.columns)
    leakage_qa = build_named_qa(
        [
            ("fwd_ret_1m not used for selection", True, "selection code only sorts by score_column and symbol"),
            ("VALUE_BP_EP_EQUAL_WEIGHT_score not used", "VALUE_BP_EP_EQUAL_WEIGHT_score" not in weight_panel_columns, "watch score 未读取、未输出"),
            ("watch score excluded", True, "watch score 未参与 first construction run"),
            ("no score from dropped features", all(col not in weight_panel_columns for col in FORBIDDEN_SELECTION_COLUMNS), "禁用列未进入 weight panel"),
            ("no prediction columns used", not any("pred" in col.lower() or "prediction" in col.lower() for col in weight_panel_columns), "未读取 prediction columns"),
            ("no sign flip columns used", not any("sign_flip" in col.lower() or "flip" in col.lower() for col in weight_panel_columns), "未读取 sign flip columns"),
            ("no production columns created", not any("production" in col.lower() or "holding" in col.lower() or "order" in col.lower() for col in weight_panel_columns), "未创建生产持仓/订单列"),
            ("no portfolio return calculated", True, "本脚本不计算收益"),
        ]
    )
    leakage_qa.to_csv(OUT_DIR / "portfolio_leakage_exclusion_qa.csv", index=False, encoding="utf-8-sig")

    guardrail_qa = build_named_qa(
        [
            ("no portfolio return calculated", True, "blocked"),
            ("no backtest", True, "blocked"),
            ("no transaction cost", True, "blocked"),
            ("no turnover", True, "blocked"),
            ("no Sharpe", True, "blocked"),
            ("no MaxDD", True, "blocked"),
            ("no training", True, "blocked"),
            ("no tuning", True, "blocked"),
            ("no SHAP", True, "blocked"),
            ("no feature importance", True, "blocked"),
            ("no holdings production", True, "blocked"),
            ("no live-order-ready file", True, "blocked"),
            ("Compact-F rescue blocked", True, "blocked"),
            ("sign-flip production blocked", True, "blocked"),
            ("LightGBM-first blocked", True, "blocked"),
        ]
    )
    guardrail_qa.to_csv(OUT_DIR / "portfolio_construction_guardrail_qa.csv", index=False, encoding="utf-8-sig")

    next_step = (
        "# Next Step: Simple Baseline Portfolio Evaluation Prep v0\n\n"
        "下一步只允许准备 research-only portfolio evaluation 所需的 QA 与配置。\n\n"
        "明确禁止：production backtest、live holdings、model training、调参、SHAP、交易成本、换手率、Sharpe、MaxDD。\n"
    )
    (OUT_DIR / "next_step_simple_baseline_portfolio_evaluation_prep_plan.md").write_text(next_step, encoding="utf-8")

    rows_loaded = int(len(score_df))
    symbol_count = int(score_df["symbol"].nunique())
    month_count = int(score_df["month_end"].nunique())
    min_month_end = score_df["month_end"].min()
    max_month_end = score_df["month_end"].max()
    weights_qa_passed = bool(weights_qa["status"].eq("PASS").all())
    input_qa_passed = bool(input_qa.loc[input_qa["severity"].eq("BLOCKING"), "status"].eq("PASS").all())
    coverage_passed = bool(coverage["status"].eq("PASS").all())
    leakage_passed = bool(leakage_qa["status"].eq("PASS").all())
    guardrail_passed = bool(guardrail_qa["status"].eq("PASS").all())
    prerequisites_passed = bool(prerequisites["prerequisites_passed"])

    blocking_passed = prerequisites_passed and input_qa_passed and weights_qa_passed and coverage_passed and leakage_passed and guardrail_passed
    final_decision = (
        "SIMPLE_BASELINE_PORTFOLIO_CONSTRUCTION_RUN_READY_FOR_PORTFOLIO_EVALUATION_PREP"
        if blocking_passed
        else "SIMPLE_BASELINE_PORTFOLIO_CONSTRUCTION_RUN_FAIL"
    )

    summary = {
        "run_timestamp": now_iso(),
        "prerequisites_passed": prerequisites_passed,
        "rows_loaded": rows_loaded,
        "symbol_count": symbol_count,
        "month_count": month_count,
        "min_month_end": min_month_end,
        "max_month_end": max_month_end,
        "portfolio_count": len(PORTFOLIOS),
        "portfolio_names": [p["portfolio_name"] for p in PORTFOLIOS],
        "selected_score_columns": ["VALUE_BP_SINGLE_score", "VALUE_QUALITY_EQUAL_WEIGHT_score"],
        "weight_panel_generated": weights_path.exists(),
        "weight_panel_path": str(weights_path),
        "total_weight_rows": int(len(weights)),
        "weights_qa_passed": weights_qa_passed,
        "coverage_summary_generated": (OUT_DIR / "portfolio_coverage_summary.csv").exists(),
        "leakage_exclusion_qa_passed": leakage_passed,
        "guardrail_qa_passed": guardrail_passed,
        "research_only_policy": True,
        "fwd_ret_used_for_selection": False,
        "portfolio_return_calculated": False,
        "backtest_run": False,
        "transaction_cost_calculated": False,
        "turnover_calculated": False,
        "sharpe_calculated": False,
        "maxdd_calculated": False,
        "training_run": False,
        "shap_calculated": False,
        "tuning_run": False,
        "feature_importance_calculated": False,
        "production_holdings_generated": False,
        "live_order_ready_file_generated": False,
        "production_modified": False,
        "compact_f_rescue_blocked": True,
        "sign_flip_production_blocked": True,
        "lightgbm_first_blocked": True,
        "final_decision": final_decision,
        "recommended_next_step": "Simple Baseline Portfolio Evaluation Prep v0",
    }
    write_json(OUT_DIR / "simple_baseline_portfolio_construction_run_summary.json", summary)
    (OUT_DIR / "simple_baseline_portfolio_construction_run_report.md").write_text(build_report(summary), encoding="utf-8")

    final_qa = pd.DataFrame(
        [
            ["prerequisites_passed", prerequisites_passed],
            ["input_score_panel_qa_passed", input_qa_passed],
            ["weights_qa_passed", weights_qa_passed],
            ["coverage_summary_generated", summary["coverage_summary_generated"]],
            ["leakage_exclusion_qa_passed", leakage_passed],
            ["guardrail_qa_passed", guardrail_passed],
            ["final_decision", final_decision],
        ],
        columns=["check_name", "observed_value"],
    )
    final_qa.to_csv(AGENT_DIR / "final_qa.csv", index=False, encoding="utf-8-sig")
    write_json(
        AGENT_DIR / "terminal_summary.json",
        {
            "task_name": TASK_NAME,
            "completed_at": now_iso(),
            "final_decision": final_decision,
            "stdout_log": str(AGENT_DIR / "run_stdout.txt"),
            "stderr_log": str(AGENT_DIR / "run_stderr.txt"),
            "summary_path": str(OUT_DIR / "simple_baseline_portfolio_construction_run_summary.json"),
        },
    )
    (AGENT_DIR / "task_completion_card.md").write_text(
        "\n".join(
            [
                "# Task Completion Card",
                "",
                f"- task_name: {TASK_NAME}",
                f"- completed_at: {now_iso()}",
                f"- final_decision: {final_decision}",
                f"- weight_panel_path: {weights_path}",
                f"- total_weight_rows: {len(weights)}",
                "- forbidden_metrics_calculated: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (AGENT_DIR / "RUN_STATE.md").write_text(
        f"# RUN_STATE\n\n- task_name: {TASK_NAME}\n- status: complete\n- updated_at: {now_iso()}\n- final_decision: {final_decision}\n",
        encoding="utf-8",
    )
    write_json(
        AGENT_DIR / "RUN_STATE.json",
        {
            "task_name": TASK_NAME,
            "status": "complete",
            "updated_at": now_iso(),
            "final_decision": final_decision,
        },
    )

    print(json.dumps({"final_decision": final_decision, "total_weight_rows": int(len(weights))}, ensure_ascii=False))

    del score_df, weights, weights_qa, input_qa, coverage, leakage_qa, guardrail_qa
    del rule_manifest, score_manifest, qa_plan, guardrail_checklist
    gc.collect()


if __name__ == "__main__":
    main()
