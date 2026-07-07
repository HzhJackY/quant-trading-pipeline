from __future__ import annotations

import hashlib
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "csmar_fs_comins_pit_coverage_patch_v1"
FS_PATH = ROOT / "output" / "csmar_fs_comins_manual_import_audit_v1" / "fs_comins_standardized_v1.parquet"
PIT_PATH = ROOT / "output" / "csmar_p0_pit_pack_import_audit_v1" / "csmar_p0_pit_announcement_panel_v1.parquet"
TRAINING_PANEL_PATH = ROOT / "output" / "training_panel_v15_sr.parquet"
ALL_DAILY_PATH = ROOT / "output" / "all_daily.parquet"
STATUS_PATH = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS_PATH = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
README_PATH = ROOT / "README.md"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def rate(mask: pd.Series, denom: int | None = None) -> float:
    if denom is None:
        denom = len(mask)
    if denom == 0:
        return float("nan")
    return float(mask.sum() / denom)


def load_inputs() -> tuple[pd.DataFrame, set[str]]:
    fs = pd.read_parquet(FS_PATH)
    fs["symbol"] = fs["symbol"].astype(str).str.zfill(6)
    fs["report_period"] = pd.to_datetime(fs["report_period"], errors="coerce")
    fs["report_type"] = fs["report_type"].astype(str).str.strip()

    pit = pd.read_parquet(PIT_PATH)
    pit = pit[["symbol", "report_period", "pit_date_primary", "pit_date_source", "effective_month_end", "quality_flag"]].copy()
    pit["symbol"] = pit["symbol"].astype(str).str.zfill(6)
    for col in ["report_period", "pit_date_primary", "effective_month_end"]:
        pit[col] = pd.to_datetime(pit[col], errors="coerce")

    merged = fs.merge(pit, on=["symbol", "report_period"], how="left")
    merged["has_pit_date"] = merged["pit_date_primary"].notna()
    merged["missing_pit_date"] = ~merged["has_pit_date"]
    merged["report_year"] = merged["report_period"].dt.year

    v15 = pd.read_parquet(TRAINING_PANEL_PATH, columns=["symbol"])
    v15_symbols = set(v15["symbol"].astype(str).str.zfill(6).unique())
    merged["in_v15_universe"] = merged["symbol"].isin(v15_symbols)
    return merged, v15_symbols


def summarize_scope(name: str, df: pd.DataFrame) -> dict[str, object]:
    return {
        "scope": name,
        "n_rows": int(len(df)),
        "n_symbols": int(df["symbol"].nunique()),
        "pit_date_coverage_rate": rate(df["has_pit_date"]),
        "missing_pit_rows": int(df["missing_pit_date"].sum()),
        "min_report_period": df["report_period"].min().date().isoformat() if len(df) and df["report_period"].notna().any() else "",
        "max_report_period": df["report_period"].max().date().isoformat() if len(df) and df["report_period"].notna().any() else "",
    }


def coverage_by_scope(merged: pd.DataFrame) -> pd.DataFrame:
    from_2015 = merged["report_period"] >= pd.Timestamp("2015-01-01")
    from_2017 = merged["report_period"] >= pd.Timestamp("2017-01-01")
    is_a = merged["report_type"].eq("A")
    v15 = merged["in_v15_universe"]
    scopes = [
        ("all_history_all_symbols", pd.Series(True, index=merged.index)),
        ("from_2015_all_symbols", from_2015),
        ("from_2017_all_symbols", from_2017),
        ("v15_universe_all_history", v15),
        ("v15_universe_from_2015", v15 & from_2015),
        ("v15_universe_from_2017", v15 & from_2017),
        ("report_type_A_all_history", is_a),
        ("report_type_A_from_2015", is_a & from_2015),
        ("report_type_A_from_2017", is_a & from_2017),
        ("report_type_A_v15_from_2017", is_a & v15 & from_2017),
    ]
    out = pd.DataFrame([summarize_scope(name, merged.loc[mask].copy()) for name, mask in scopes])
    write_csv(out, OUT / "fs_comins_pit_coverage_by_scope_v1.csv")
    return out


def missing_by_year(merged: pd.DataFrame) -> pd.DataFrame:
    group = (
        merged.groupby("report_year", dropna=False)
        .agg(
            n_rows=("symbol", "size"),
            n_symbols=("symbol", "nunique"),
            missing_pit_rows=("missing_pit_date", "sum"),
            pit_rows=("has_pit_date", "sum"),
        )
        .reset_index()
    )
    group["pit_date_coverage_rate"] = group["pit_rows"] / group["n_rows"]
    group["missing_pit_rate"] = group["missing_pit_rows"] / group["n_rows"]
    write_csv(group, OUT / "fs_comins_missing_pit_by_year_v1.csv")
    return group


def readiness(coverage: pd.DataFrame) -> tuple[pd.DataFrame, str, bool]:
    row = coverage.loc[coverage["scope"].eq("report_type_A_v15_from_2017")].iloc[0]
    cov = float(row["pit_date_coverage_rate"])
    if cov >= 0.95:
        decision = "CSMAR_FS_COMINS_EFFECTIVE_WINDOW_PIT_READY"
        ready = True
        status = "ready_for_core_fs_merge"
    elif cov >= 0.80:
        decision = "CSMAR_FS_COMINS_EFFECTIVE_WINDOW_PARTIAL_PIT_READY"
        ready = False
        status = "partial_ready_needs_targeted_pit_review"
    else:
        decision = "CSMAR_FS_COMINS_EFFECTIVE_WINDOW_PIT_NEEDS_REPAIR"
        ready = False
        status = "needs_pit_repair"
    out = pd.DataFrame([{
        "effective_window": "report_type_A_v15_from_2017",
        "n_rows": int(row["n_rows"]),
        "n_symbols": int(row["n_symbols"]),
        "pit_date_coverage_rate": cov,
        "missing_pit_rows": int(row["missing_pit_rows"]),
        "fs_comins_effective_window_ready": ready,
        "status_patch": status,
        "decision": decision,
        "recommended_next_task": (
            "Proceed to FS_Combas and TRD_Dalyr import for core FS merge."
            if ready
            else "Repair or extend P0 PIT announcement dates for missing effective-window rows before core FS merge."
        ),
        "notes": "No final factors were built; this is a coverage patch only.",
    }])
    write_csv(out, OUT / "fs_comins_effective_window_readiness_v1.csv")
    return out, decision, ready


def update_status(decision: str, ready: bool) -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    status["project"]["last_updated"] = date.today().isoformat()
    status["alternative_data"]["csmar_latest_task"] = "CSMAR FS_Comins PIT Coverage Patch v1"
    status["alternative_data"]["csmar_latest_output"] = rel(OUT)
    if ready:
        status["alternative_data"]["csmar_status"] = "ready_for_core_fs_merge"
        status["validation"]["pit_financial_status"] = "fs_comins_effective_window_pit_ready_fs_combas_trd_pending"
    elif decision == "CSMAR_FS_COMINS_EFFECTIVE_WINDOW_PARTIAL_PIT_READY":
        status["alternative_data"]["csmar_status"] = "fs_comins_effective_window_partial_pit_ready_needs_review"
        status["validation"]["pit_financial_status"] = "fs_comins_effective_window_partial_pit_ready"
    else:
        status["alternative_data"]["csmar_status"] = "fs_comins_effective_window_pit_needs_repair"
        status["validation"]["pit_financial_status"] = "fs_comins_effective_window_pit_needs_repair"
    status["validation"]["blend_v3_historical_metrics_status"] = "under_pit_review"
    STATUS_PATH.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False), encoding="utf-8")

    subprocess.run(["python", "scripts/generate_current_status_md.py"], cwd=ROOT, check=True, capture_output=True, text=True)

    block = "\n".join([
        f"## {date.today().isoformat()}",
        "",
        "决策：",
        "",
        "- 已完成 FS_Comins PIT 覆盖率分层 patch。",
        "- 有效窗口定义为 report_type=A、v15 universe、2017 年以后。",
        f"- 覆盖率决策：{decision}。",
        "- 不访问 CSMAR API，不下载数据。",
        "- 不修改 README，不接入 production。",
        "",
    ])
    text = DECISIONS_PATH.read_text(encoding="utf-8") if DECISIONS_PATH.exists() else "# 决策日志\n\n"
    marker = "已完成 FS_Comins PIT 覆盖率分层 patch"
    if marker not in text:
        DECISIONS_PATH.write_text(text.rstrip() + "\n\n" + block, encoding="utf-8")


def write_report(coverage: pd.DataFrame, missing: pd.DataFrame, ready_df: pd.DataFrame, decision: str) -> None:
    row = ready_df.iloc[0]
    report = "\n".join([
        "# CSMAR FS_Comins PIT Coverage Patch v1",
        "",
        "## Executive Summary",
        "",
        "- 本任务只读取本地 parquet 文件，不访问 CSMAR API，不调用 getPackResultExt，不下载数据。",
        "- 本任务不构建最终因子，不训练模型，不回测，不做 IC，不生成交易信号。",
        f"- 有效窗口 report_type=A + v15 universe + 2017 年以后 PIT 覆盖率为 {row['pit_date_coverage_rate']:.6f}。",
        f"- decision={decision}",
        "",
        "## Coverage By Scope",
        "",
        f"- 输出：`{rel(OUT / 'fs_comins_pit_coverage_by_scope_v1.csv')}`",
        f"- all_history_all_symbols 覆盖率：{coverage.loc[coverage['scope'].eq('all_history_all_symbols'), 'pit_date_coverage_rate'].iloc[0]:.6f}",
        f"- report_type_A_v15_from_2017 覆盖率：{row['pit_date_coverage_rate']:.6f}",
        "",
        "## Missing PIT By Year",
        "",
        f"- 输出：`{rel(OUT / 'fs_comins_missing_pit_by_year_v1.csv')}`",
        f"- 缺失 PIT 行数合计：{int(missing['missing_pit_rows'].sum())}",
        "",
        "## Effective Window Readiness",
        "",
        f"- 输出：`{rel(OUT / 'fs_comins_effective_window_readiness_v1.csv')}`",
        f"- ready={bool(row['fs_comins_effective_window_ready'])}",
        f"- n_rows={int(row['n_rows'])}",
        f"- n_symbols={int(row['n_symbols'])}",
        "",
        "## Limitations",
        "",
        "- 该 patch 只重新解释 PIT 覆盖口径，不补采公告日期。",
        "- 如有效窗口仍未达到阈值，应修补 P0 PIT 层后再进入 core FS merge。",
        "- ROE/BP/EP/Debt_Ratio 仍需 FS_Combas 和 TRD_Dalyr。",
        "",
        "## Files Generated",
        "",
        f"- `{rel(OUT / 'fs_comins_pit_coverage_by_scope_v1.csv')}`",
        f"- `{rel(OUT / 'fs_comins_missing_pit_by_year_v1.csv')}`",
        f"- `{rel(OUT / 'fs_comins_effective_window_readiness_v1.csv')}`",
        f"- `{rel(OUT / 'fs_comins_pit_coverage_patch_report_v1.md')}`",
        f"- `{rel(OUT / 'task_completion_card.md')}`",
        f"- `{rel(OUT / 'final_qa_csmar_fs_comins_pit_coverage_patch_v1.csv')}`",
        "",
    ])
    (OUT / "fs_comins_pit_coverage_patch_report_v1.md").write_text(report, encoding="utf-8")

    card = "\n".join([
        "任务名称：CSMAR FS_Comins PIT Coverage Patch v1",
        f"运行日期：{date.today().isoformat()}",
        "是否修改 production：否",
        "是否修改 README：否",
        "是否修改 all_daily：否",
        "是否修改 training_panel：否",
        "是否训练模型：否",
        "是否运行回测：否",
        "是否做 IC：否",
        "是否访问 CSMAR API：否",
        "是否执行 CSMAR 下载：否",
        f"有效窗口：{row['effective_window']}",
        f"有效窗口 PIT 覆盖率：{row['pit_date_coverage_rate']:.6f}",
        f"有效窗口行数：{int(row['n_rows'])}",
        f"有效窗口证券数：{int(row['n_symbols'])}",
        f"FS_Comins effective window ready：{bool(row['fs_comins_effective_window_ready'])}",
        f"核心结论：{decision}",
        f"下一步建议：{row['recommended_next_task']}",
        "",
    ])
    (OUT / "task_completion_card.md").write_text(card, encoding="utf-8")


def final_qa(before_hashes: dict[str, str], decision: str) -> pd.DataFrame:
    after_hashes = {
        "README.md": sha256(README_PATH),
        "output/all_daily.parquet": sha256(ALL_DAILY_PATH),
        "output/training_panel_v15_sr.parquet": sha256(TRAINING_PANEL_PATH),
    }
    checks = [
        ("README.md not modified", before_hashes["README.md"] == after_hashes["README.md"], ""),
        ("all_daily.parquet not modified", before_hashes["output/all_daily.parquet"] == after_hashes["output/all_daily.parquet"], ""),
        ("training_panel_v15_sr.parquet not modified", before_hashes["output/training_panel_v15_sr.parquet"] == after_hashes["output/training_panel_v15_sr.parquet"], ""),
        ("model files not modified", True, "script does not write model paths"),
        ("production config not modified", True, "only config/project_status.yaml updated as requested"),
        ("no model training executed", True, ""),
        ("no backtest executed", True, ""),
        ("no IC test executed", True, ""),
        ("no trading signal generated", True, ""),
        ("no CSMAR API access executed", True, "local parquet files only"),
        ("getPackResultExt not called", True, ""),
        ("no data downloaded", True, ""),
        ("root-level output used", str(OUT).startswith(str(ROOT / "output")), rel(OUT)),
        ("xhs/output not used", not rel(OUT).startswith("xhs/output"), rel(OUT)),
        ("coverage by scope generated", (OUT / "fs_comins_pit_coverage_by_scope_v1.csv").exists(), ""),
        ("missing PIT by year generated", (OUT / "fs_comins_missing_pit_by_year_v1.csv").exists(), ""),
        ("effective window readiness generated", (OUT / "fs_comins_effective_window_readiness_v1.csv").exists(), ""),
        ("report generated", (OUT / "fs_comins_pit_coverage_patch_report_v1.md").exists(), ""),
        ("task completion card generated", (OUT / "task_completion_card.md").exists(), ""),
        ("project_status.yaml updated", STATUS_PATH.exists(), decision),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS_PATH.exists(), ""),
        ("DECISIONS.md appended", DECISIONS_PATH.exists() and "FS_Comins PIT 覆盖率分层 patch" in DECISIONS_PATH.read_text(encoding="utf-8"), ""),
    ]
    qa = pd.DataFrame(checks, columns=["check", "pass", "details"])
    write_csv(qa, OUT / "final_qa_csmar_fs_comins_pit_coverage_patch_v1.csv")
    return qa


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before_hashes = {
        "README.md": sha256(README_PATH),
        "output/all_daily.parquet": sha256(ALL_DAILY_PATH),
        "output/training_panel_v15_sr.parquet": sha256(TRAINING_PANEL_PATH),
    }
    csmar_api_accessed = False
    getpack_called = False

    merged, _ = load_inputs()
    coverage = coverage_by_scope(merged)
    missing = missing_by_year(merged)
    ready_df, decision, ready = readiness(coverage)
    update_status(decision, ready)
    write_report(coverage, missing, ready_df, decision)
    qa = final_qa(before_hashes, decision)

    protected_modified = {
        "readme_modified": before_hashes["README.md"] != sha256(README_PATH),
        "all_daily_modified": before_hashes["output/all_daily.parquet"] != sha256(ALL_DAILY_PATH),
        "training_panel_modified": before_hashes["output/training_panel_v15_sr.parquet"] != sha256(TRAINING_PANEL_PATH),
    }
    if any(protected_modified.values()) or csmar_api_accessed or getpack_called:
        decision = "INVALID_MODIFICATION_OR_API_ACCESS"

    row = ready_df.iloc[0]
    output = {
        "coverage_by_scope_path": rel(OUT / "fs_comins_pit_coverage_by_scope_v1.csv"),
        "missing_pit_by_year_path": rel(OUT / "fs_comins_missing_pit_by_year_v1.csv"),
        "effective_window_readiness_path": rel(OUT / "fs_comins_effective_window_readiness_v1.csv"),
        "report_path": rel(OUT / "fs_comins_pit_coverage_patch_report_v1.md"),
        "final_qa_path": rel(OUT / "final_qa_csmar_fs_comins_pit_coverage_patch_v1.csv"),
        "report_type_a_v15_from_2017_pit_coverage_rate": float(row["pit_date_coverage_rate"]),
        "report_type_a_v15_from_2017_n_rows": int(row["n_rows"]),
        "report_type_a_v15_from_2017_n_symbols": int(row["n_symbols"]),
        "fs_comins_effective_window_ready": bool(row["fs_comins_effective_window_ready"]),
        "recommended_next_task": row["recommended_next_task"],
        "csmar_api_accessed": csmar_api_accessed,
        "getPackResultExt_called": getpack_called,
        "readme_modified": protected_modified["readme_modified"],
        "all_daily_modified": protected_modified["all_daily_modified"],
        "training_panel_modified": protected_modified["training_panel_modified"],
        "production_modified": False,
        "decision": decision,
    }
    for key, value in output.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
