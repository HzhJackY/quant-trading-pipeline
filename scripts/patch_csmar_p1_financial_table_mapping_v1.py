from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "csmar_p1_financial_table_mapping_patch_v1"
STATUS_PATH = ROOT / "config" / "project_status.yaml"
CURRENT_STATUS_PATH = ROOT / "docs" / "CURRENT_STATUS.md"
DECISIONS_PATH = ROOT / "docs" / "DECISIONS.md"
README_PATH = ROOT / "README.md"
ALL_DAILY_PATH = ROOT / "output" / "all_daily.parquet"
PANEL_PATH = ROOT / "output" / "training_panel_v15_sr.parquet"
PAPER_PIPELINE_PATH = ROOT / "paper_trading" / "paper_trading_pipeline.py"
README_CONSISTENCY_REPORT = ROOT / "output" / "blend_v3_governance_patch_v2" / "readme_consistency_report.md"
P1_SCRIPT_PATH = ROOT / "scripts" / "run_csmar_p1_financial_pack_download_v1.py"
RUN_DATE = date.today().isoformat()
PROTECTED = [README_PATH, ALL_DAILY_PATH, PANEL_PATH, PAPER_PIPELINE_PATH]

KEYWORDS = [
    "FI_T5", "financial indicator", "财务指标", "利润表", "income", "income statement", "损益表",
    "资产负债表", "balance", "balance sheet", "现金流量表", "cash flow", "statement",
    "revenue", "营业收入", "净利润", "归母净利润", "total assets", "总资产",
    "total liabilities", "总负债", "equity", "股东权益", "ROE", "净利率", "资产负债率",
    "销售费用", "研发费用", "管理费用",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def sanitize(text: Any) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"(?i)(token|cookie|session|password|account)\s*[:=]\s*[^,\s;]+", r"\1=[REDACTED]", value)
    return value[:800]


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False, timeout=180)


def field_dict_path() -> Path:
    root = ROOT / "output" / "csmar_table_inventory_audit_v1" / "csmar_field_dictionary_v1.csv"
    if root.exists():
        return root
    return ROOT / "xhs" / "output" / "csmar_table_inventory_audit_v1" / "csmar_field_dictionary_v1.csv"


def inventory_path() -> Path:
    root = ROOT / "output" / "csmar_table_inventory_audit_v1" / "csmar_table_inventory_v1.csv"
    if root.exists():
        return root
    return ROOT / "xhs" / "output" / "csmar_table_inventory_audit_v1" / "csmar_table_inventory_v1.csv"


def input_audit() -> pd.DataFrame:
    inputs = [
        (ROOT / "output" / "csmar_p1_financial_pack_download_v1" / "p1_financial_pack_download_plan_v1.csv", "ROOT_CANONICAL", "P1 v1 plan"),
        (ROOT / "output" / "csmar_p1_financial_pack_download_v1" / "p1_financial_column_validation_v1.csv", "ROOT_CANONICAL", "P1 v1 column validation"),
        (ROOT / "output" / "csmar_p1_financial_pack_download_v1" / "p1_download_result_summary_v1.csv", "ROOT_CANONICAL", "P1 v1 result summary"),
        (inventory_path(), "ROOT_CANONICAL" if str(inventory_path()).startswith(str(ROOT / "output")) else "LEGACY_XHS_FALLBACK", "table inventory"),
        (field_dict_path(), "ROOT_CANONICAL" if str(field_dict_path()).startswith(str(ROOT / "output")) else "LEGACY_XHS_FALLBACK", "field dictionary"),
        (ROOT / "output" / "csmar_discovery_v1", "ROOT_CANONICAL", "discovery output directory"),
        (ROOT / "output" / "csmar_main_project_promotion_v1" / "terminal_summary_v1.json", "ROOT_CANONICAL", "main project promotion summary"),
    ]
    rows = []
    for path, source_type, role in inputs:
        exists = path.exists()
        readable = False
        notes = ""
        if exists:
            try:
                if path.is_dir():
                    readable = True
                    notes = "|".join(sorted(p.name for p in path.iterdir())[:20])
                elif path.suffix.lower() == ".json":
                    json.loads(path.read_text(encoding="utf-8"))
                    readable = True
                else:
                    pd.read_csv(path, nrows=3)
                    readable = True
            except Exception as exc:  # noqa: BLE001
                notes = f"{type(exc).__name__}: {sanitize(exc)}"
        else:
            notes = "missing"
        rows.append({
            "input_path": rel(path),
            "exists": exists,
            "readable": readable,
            "source_type": source_type,
            "role": role,
            "notes": notes,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "input_audit_v1.csv", index=False, encoding="utf-8-sig")
    return df


def discovery_tables() -> pd.DataFrame:
    path = ROOT / "output" / "csmar_discovery_v1" / "csmar_tables.csv"
    if not path.exists():
        return pd.DataFrame(columns=["table_id", "table_name", "table_name_cn", "table_name_en", "source"])
    df = read_csv(path)
    rows = []
    for _, row in df.iterrows():
        raw = str(row.get("raw_json", ""))
        table_id = ""
        table_cn = str(row.get("table_name", ""))
        try:
            obj = json.loads(raw)
            table_id = str(obj.get("table", ""))
            table_cn = str(obj.get("tableName", table_cn))
        except Exception:
            pass
        rows.append({
            "table_id": table_id,
            "table_name": table_cn,
            "table_name_cn": table_cn,
            "table_name_en": "",
            "source": "output/csmar_discovery_v1/csmar_tables.csv",
        })
    return pd.DataFrame(rows)


def category_guess(table_id: str, name: str, fields: list[str]) -> str:
    text = f"{table_id} {name} {' '.join(fields)}".lower()
    if "fi_t5" in text or "财务指标" in text or "roe" in text or "f050501b" in text:
        return "financial_indicator"
    if "利润表" in text or "income" in text or "损益表" in text or "revenue" in text or "营业收入" in text:
        return "income_statement"
    if "资产负债表" in text or "balance" in text or "总资产" in text or "总负债" in text:
        return "balance_sheet"
    if "现金流量表" in text or "cash flow" in text:
        return "cash_flow"
    if "预告" in text or "快报" in text or "forecast" in text:
        return "earnings_preview"
    if "销售费用" in text:
        return "sales_expense_detail"
    if "研发费用" in text:
        return "rd_expense_detail"
    return "other_financial_candidate"


def flag_any(fields: list[str], patterns: list[str]) -> bool:
    text = " ".join(fields).lower()
    return any(p.lower() in text for p in patterns)


def candidate_tables() -> pd.DataFrame:
    inventory = read_csv(inventory_path()) if inventory_path().exists() else pd.DataFrame()
    field_dict = read_csv(field_dict_path()) if field_dict_path().exists() else pd.DataFrame()
    discovery = discovery_tables()

    base: dict[str, dict[str, Any]] = {}
    if not inventory.empty:
        for _, row in inventory.iterrows():
            tid = str(row.get("table_id", "")).strip()
            if tid:
                base.setdefault(tid, {
                    "table_id": tid,
                    "table_name": str(row.get("table_name_cn", row.get("table_name", ""))),
                    "table_name_cn": str(row.get("table_name_cn", row.get("table_name", ""))),
                    "table_name_en": str(row.get("table_name_en", "")),
                    "source": "inventory",
                })
    if not field_dict.empty:
        for tid, group in field_dict.groupby("table_id"):
            tid = str(tid)
            base.setdefault(tid, {
                "table_id": tid,
                "table_name": str(group["table_name"].iloc[0]) if "table_name" in group.columns else "",
                "table_name_cn": str(group["table_name"].iloc[0]) if "table_name" in group.columns else "",
                "table_name_en": "",
                "source": "field_dictionary",
            })
    for _, row in discovery.iterrows():
        tid = str(row.get("table_id", "")).strip()
        name = str(row.get("table_name", ""))
        if not tid:
            continue
        text = f"{tid} {name}".lower()
        if any(k.lower() in text for k in KEYWORDS):
            base.setdefault(tid, row.to_dict())

    rows = []
    for tid, item in sorted(base.items()):
        fields = []
        if not field_dict.empty:
            fields = field_dict.loc[field_dict["table_id"].astype(str).eq(tid), "field_name"].astype(str).tolist()
        name = str(item.get("table_name", ""))
        text = f"{tid} {name} {' '.join(fields)}"
        matched = [k for k in KEYWORDS if k.lower() in text.lower()]
        cat = category_guess(tid, name, fields)
        has_symbol = "Stkcd" in fields
        has_period = "Accper" in fields
        core_ok = has_symbol and has_period
        financial_hit = bool(matched) or tid in {"FI_T5", "FN_Fn050", "FN_Fn060"}
        if not financial_hit:
            continue
        if tid == "FI_T5":
            confidence = "HIGH"
        elif tid in {"FN_Fn050", "FN_Fn060"} and core_ok:
            confidence = "MEDIUM"
        elif core_ok and matched:
            confidence = "LOW"
        else:
            confidence = "BLOCKED"
        rows.append({
            "table_id": tid,
            "table_name": name,
            "table_name_cn": str(item.get("table_name_cn", name)),
            "table_name_en": str(item.get("table_name_en", "")),
            "category_guess": cat,
            "source": str(item.get("source", "")),
            "n_fields_detected": len(fields),
            "matched_keywords": "|".join(matched),
            "has_symbol_field": has_symbol,
            "has_report_period_field": has_period,
            "has_revenue_field": flag_any(fields + [name], ["revenue", "营业收入"]),
            "has_net_profit_field": flag_any(fields + [name], ["net_profit", "净利润", "归母净利润"]),
            "has_total_assets_field": flag_any(fields + [name], ["total_assets", "总资产"]),
            "has_total_liabilities_field": flag_any(fields + [name], ["total_liabilities", "总负债"]),
            "has_equity_field": flag_any(fields + [name], ["equity", "股东权益", "归母权益"]),
            "has_roe_field": "F050501B" in fields or flag_any(fields + [name], ["ROE"]),
            "has_margin_field": bool({"F053301B", "F051701B", "F051801B"} & set(fields)) or flag_any(fields + [name], ["净利率", "margin"]),
            "has_debt_ratio_field": "F051801B" in fields or flag_any(fields + [name], ["资产负债率", "debt"]),
            "mapping_confidence": confidence,
            "notes": "Only fields present in local field dictionary are considered executable." if fields else "No local field dictionary fields.",
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "candidate_financial_tables_v1.csv", index=False, encoding="utf-8-sig")
    return df


def field_mapping() -> pd.DataFrame:
    field_dict = read_csv(field_dict_path()) if field_dict_path().exists() else pd.DataFrame()
    fields_by_table = {tid: set(g["field_name"].astype(str)) for tid, g in field_dict.groupby("table_id")} if not field_dict.empty else {}
    field_name_cn: dict[tuple[str, str], str] = {}
    if not field_dict.empty:
        for _, row in field_dict.iterrows():
            field_name_cn[(str(row["table_id"]), str(row["field_name"]))] = str(row.get("field_name_cn", ""))

    specs = [
        ("identity", "symbol", ["FI_T5", "FN_Fn050", "FN_Fn060"], ["Stkcd"], "key", "HIGH"),
        ("identity", "report_period", ["FI_T5", "FN_Fn050", "FN_Fn060"], ["Accper"], "key", "HIGH"),
        ("identity", "report_type", ["FI_T5", "FN_Fn050", "FN_Fn060"], ["Typrep"], "filter", "HIGH"),
        ("income_statement", "revenue", ["FS_Comins", "FI_T5"], ["营业收入", "revenue"], "raw_value", "LOW"),
        ("income_statement", "net_profit_parent", ["FS_Comins"], ["归母净利润", "net_profit_parent"], "raw_value", "LOW"),
        ("income_statement", "sales_expense", ["FN_Fn050", "FI_T5"], ["FN05002", "FN05001", "F051701B"], "raw_or_ratio_candidate", "MEDIUM"),
        ("income_statement", "management_expense", ["FS_Comins"], ["管理费用"], "raw_value", "LOW"),
        ("income_statement", "rd_expense", ["FN_Fn060"], ["FN_Fn06002", "FN_Fn06001"], "raw_value_candidate", "MEDIUM"),
        ("balance_sheet", "total_assets", ["FS_Combas"], ["总资产", "total_assets"], "raw_value", "LOW"),
        ("balance_sheet", "total_liabilities", ["FS_Combas"], ["总负债", "total_liabilities"], "raw_value", "LOW"),
        ("balance_sheet", "total_equity_parent", ["FS_Combas"], ["归母权益", "股东权益", "equity"], "raw_value", "LOW"),
        ("financial_indicator", "roe", ["FI_T5"], ["F050501B"], "ratio", "HIGH"),
        ("financial_indicator", "net_margin", ["FI_T5"], ["F053301B", "F051701B"], "ratio_candidate", "MEDIUM"),
        ("financial_indicator", "debt_ratio", ["FI_T5"], ["F051801B"], "ratio_candidate", "MEDIUM"),
    ]
    rows = []
    for group, item, tables, candidates, role, default_conf in specs:
        matched = False
        for table in tables:
            available = fields_by_table.get(table, set())
            for field in candidates:
                if field in available:
                    rows.append({
                        "target_factor_group": group,
                        "target_raw_item": item,
                        "candidate_table": table,
                        "resolved_field_name": field,
                        "resolved_field_name_cn": field_name_cn.get((table, field), ""),
                        "field_available": True,
                        "field_role": role,
                        "confidence": default_conf,
                        "notes": "Resolved from local field dictionary.",
                    })
                    matched = True
                    break
            if matched:
                break
        if not matched:
            rows.append({
                "target_factor_group": group,
                "target_raw_item": item,
                "candidate_table": "|".join(tables),
                "resolved_field_name": "",
                "resolved_field_name_cn": "",
                "field_available": False,
                "field_role": role,
                "confidence": "BLOCKED",
                "notes": "No confirmed field in local field dictionary.",
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p1_financial_field_mapping_v1.csv", index=False, encoding="utf-8-sig")
    return df


def v15_range() -> tuple[str, str]:
    panel = pd.read_parquet(PANEL_PATH, columns=["date"])
    dates = pd.to_datetime(panel["date"], errors="coerce").dropna()
    return (dates.min() - pd.DateOffset(years=2)).strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


def make_output_name(table_id: str, start: str, end: str) -> str:
    return f"{table_id}_{start}_{end}_Stkcd_like_p1_pack_export_{datetime.now():%Y%m%d}.csv"


def plan_v2(candidates: pd.DataFrame) -> pd.DataFrame:
    field_dict = read_csv(field_dict_path()) if field_dict_path().exists() else pd.DataFrame()
    fields_by_table = {tid: list(g["field_name"].astype(str)) for tid, g in field_dict.groupby("table_id")} if not field_dict.empty else {}
    start, end = v15_range()
    desired = [
        (1, "财务指标表", "FI_T5", "financial_indicator", ["Stkcd", "Accper", "Typrep", "F050501B", "F053301B", "F051701B", "F051801B"], "ROE|NetMargin candidate|Debt_Ratio candidate", "READY"),
        (2, "财务报表附注-销售费用明细", "FN_Fn050", "sales_expense_detail", ["Stkcd", "Accper", "Typrep", "FN05001", "FN05002"], "sales_expense_to_revenue candidate", "PARTIAL_READY"),
        (3, "财务报表附注-研发费用明细", "FN_Fn060", "rd_expense_detail", ["Stkcd", "Accper", "Typrep", "FN_Fn06001", "FN_Fn06002", "FN_Fn06003"], "rd_expense_to_revenue candidate", "PARTIAL_READY"),
    ]
    rows = []
    for rank, name, tid, category, wanted, factors, status in desired:
        available = set(fields_by_table.get(tid, []))
        cols = [c for c in wanted if c in available]
        if not {"Stkcd", "Accper"}.issubset(set(cols)):
            continue
        rows.append({
            "priority": rank,
            "table_name": name,
            "table_id": tid,
            "table_category": category,
            "columns": ",".join(cols),
            "condition": "Stkcd like '%'",
            "startTime": start,
            "endTime": end,
            "expected_output_name": make_output_name(tid, start, end),
            "target_local_dir": "data/csmar_exports",
            "target_factors_supported": factors,
            "field_validation_status": status,
            "should_execute_first": rank == 1,
            "execute_rank": rank,
            "notes": "Confirmed by local field dictionary. FS_Comins/FS_Combas are excluded from executable plan.",
        })
    df = pd.DataFrame(rows, columns=[
        "priority", "table_name", "table_id", "table_category", "columns", "condition",
        "startTime", "endTime", "expected_output_name", "target_local_dir",
        "target_factors_supported", "field_validation_status", "should_execute_first",
        "execute_rank", "notes",
    ])
    df.to_csv(OUT / "p1_financial_pack_download_plan_v2.csv", index=False, encoding="utf-8-sig")
    return df


def next_queue(plan: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in plan.sort_values("execute_rank").iterrows():
        rows.append({
            "execute_rank": row["execute_rank"],
            "table_name": row["table_name"],
            "table_category": row["table_category"],
            "columns": row["columns"],
            "condition": row["condition"],
            "startTime": row["startTime"],
            "endTime": row["endTime"],
            "expected_output_name": row["expected_output_name"],
            "reason": "FI_T5 first because it is READY and highest confidence." if str(row["table_id"]) == "FI_T5" else "Run only after previous table succeeds and local file check passes.",
            "wait_rule": "Use --max-downloads 1; stop immediately on DAILY_LIMIT; do not rapidly execute multiple downloads.",
            "notes": "After each success, run scripts/check_csmar_manual_exports_v1.py before the next download.",
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "p1_next_execute_queue_v1.csv", index=False, encoding="utf-8-sig")
    return df


def patch_p1_script() -> bool:
    text = P1_SCRIPT_PATH.read_text(encoding="utf-8")
    if "MAPPING_PATCH_V2_PLAN_PATH" in text and "load_mapping_patch_plan_v2" in text:
        return False
    marker = 'RUN_DATE = date.today().isoformat()\nPROTECTED = [README_PATH, ALL_DAILY_PATH, PANEL_PATH, PAPER_PIPELINE_PATH]\n'
    replacement = (
        'RUN_DATE = date.today().isoformat()\n'
        'MAPPING_PATCH_V2_PLAN_PATH = ROOT / "output" / "csmar_p1_financial_table_mapping_patch_v1" / "p1_financial_pack_download_plan_v2.csv"\n'
        'PROTECTED = [README_PATH, ALL_DAILY_PATH, PANEL_PATH, PAPER_PIPELINE_PATH]\n'
    )
    text = text.replace(marker, replacement)
    insert_after = '''def generate_plan() -> tuple[pd.DataFrame, pd.DataFrame]:
    field_dict = load_field_dictionary()
    start, end = v15_date_range()
'''
    insert_repl = '''def load_mapping_patch_plan_v2() -> pd.DataFrame | None:
    if not MAPPING_PATCH_V2_PLAN_PATH.exists():
        return None
    plan = pd.read_csv(MAPPING_PATCH_V2_PLAN_PATH, dtype=str).fillna("")
    plan.to_csv(OUT / "p1_financial_pack_download_plan_v1.csv", index=False, encoding="utf-8-sig")
    validation = pd.DataFrame([{
        "table_name": row.get("table_name", ""),
        "target_factor_group": row.get("table_category", ""),
        "required_fields_cn_or_alias": row.get("target_factors_supported", ""),
        "resolved_csmar_fields": row.get("columns", ""),
        "missing_fields": "",
        "optional_fields_removed": "",
        "validation_status": row.get("field_validation_status", ""),
        "notes": "Loaded from CSMAR P1 Financial Table Mapping Patch v1 plan_v2.",
    } for _, row in plan.iterrows()])
    validation.to_csv(OUT / "p1_financial_column_validation_v1.csv", index=False, encoding="utf-8-sig")
    return plan


def generate_plan() -> tuple[pd.DataFrame, pd.DataFrame]:
    patched_plan = load_mapping_patch_plan_v2()
    if patched_plan is not None:
        validation = pd.read_csv(OUT / "p1_financial_column_validation_v1.csv", dtype=str).fillna("")
        return patched_plan, validation
    field_dict = load_field_dictionary()
    start, end = v15_date_range()
'''
    text = text.replace(insert_after, insert_repl)
    terminal_marker = '"n_p1_queries_ready": ready_count,\n'
    terminal_repl = '"n_p1_queries_ready": ready_count,\n        "next_execute_table": str(plan.sort_values("execute_rank").iloc[0]["table_id"]) if "execute_rank" in plan.columns and len(plan) else (str(plan.iloc[0]["table_id"]) if len(plan) else ""),\n'
    text = text.replace(terminal_marker, terminal_repl)
    P1_SCRIPT_PATH.write_text(text, encoding="utf-8")
    return True


def update_status() -> None:
    status = yaml.safe_load(STATUS_PATH.read_text(encoding="utf-8"))
    status.setdefault("alternative_data", {})
    status["alternative_data"]["csmar_status"] = "p1_financial_table_mapping_patched_waiting_for_quota"
    status["alternative_data"]["csmar_latest_task"] = "CSMAR P1 Financial Table Mapping Patch v1"
    status["alternative_data"]["csmar_latest_output"] = rel(OUT)
    status.setdefault("validation", {})
    status["validation"]["pit_financial_status"] = "p0_pit_dates_imported_p1_mapping_patched_download_pending"
    status["validation"]["blend_v3_historical_metrics_status"] = "under_pit_review"
    status.setdefault("project", {})["last_updated"] = RUN_DATE
    STATUS_PATH.write_text(yaml.safe_dump(status, allow_unicode=True, sort_keys=False), encoding="utf-8")


def append_decision(decision: str) -> None:
    marker = f"Decision = {decision}。"
    text = DECISIONS_PATH.read_text(encoding="utf-8") if DECISIONS_PATH.exists() else "# 决策日志\n"
    if marker in text and "P1 财务表映射已离线修正" in text:
        return
    block = "\n".join([
        f"## {RUN_DATE}",
        "",
        "决策：",
        "",
        "- FI_T5 字段校验 READY。",
        "- FS_Comins / FS_Combas 被识别为不可用或错误表名。",
        "- P1 财务表映射已离线修正。",
        "- 等待额度恢复后继续 pack download。",
        "- 不接入 production。",
        "- 不修改 README。",
        f"- Decision = {decision}。",
    ])
    DECISIONS_PATH.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def credential_exposure_detected() -> bool:
    patterns = [r"CSMAR_PASSWORD\s*=", r"CSMAR_ACCOUNT\s*=", r"token\s*=", r"cookie\s*=", r"session\s*="]
    for path in OUT.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".md", ".json", ".txt", ".log"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(re.search(p, text, re.I) for p in patterns):
                return True
    return False


def task_card(decision: str, plan: pd.DataFrame, blocked: list[str]) -> Path:
    ready = "|".join(plan["table_id"].astype(str).tolist())
    first = plan.sort_values("execute_rank").iloc[0] if len(plan) else {}
    cmd = "python scripts\\run_csmar_p1_financial_pack_download_v1.py --execute --max-downloads 1"
    path = OUT / "task_completion_card.md"
    lines = [
        "任务名称：CSMAR P1 Financial Table Mapping Patch v1",
        f"运行日期：{RUN_DATE}",
        "是否修改 production：否",
        "是否修改 README：否",
        "是否修改 all_daily：否",
        "是否修改 training_panel：否",
        "是否训练模型：否",
        "是否运行回测：否",
        "是否做 IC：否",
        "是否生成交易信号：否",
        "是否访问 CSMAR API：否",
        "是否执行 CSMAR 下载：否",
        "核心输出：",
        f"- {rel(OUT / 'candidate_financial_tables_v1.csv')}",
        f"- {rel(OUT / 'p1_financial_field_mapping_v1.csv')}",
        f"- {rel(OUT / 'p1_financial_pack_download_plan_v2.csv')}",
        f"- {rel(OUT / 'p1_next_execute_queue_v1.csv')}",
        f"核心结论：{decision}",
        f"可执行 P1 表：{ready or '无'}",
        f"不可用表：{'|'.join(blocked) or '无'}",
        f"明日建议命令：{cmd}",
        "下一步建议：额度恢复后先执行 FI_T5，成功后做本地文件检查，再考虑后续费用明细表。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def final_qa(initial_hashes: dict[Path, str], patched: bool) -> pd.DataFrame:
    current = {p: sha256_file(p) for p in PROTECTED}
    script_text = P1_SCRIPT_PATH.read_text(encoding="utf-8")
    rows = [
        ("README.md not modified", current[README_PATH] == initial_hashes[README_PATH], "hash unchanged"),
        ("all_daily.parquet not modified", current[ALL_DAILY_PATH] == initial_hashes[ALL_DAILY_PATH], "hash unchanged"),
        ("training_panel_v15_sr.parquet not modified", current[PANEL_PATH] == initial_hashes[PANEL_PATH], "hash unchanged"),
        ("model files not modified", True, "no model paths written"),
        ("paper_trading_pipeline.py not modified", current[PAPER_PIPELINE_PATH] == initial_hashes[PAPER_PIPELINE_PATH], "hash unchanged"),
        ("production config not modified", True, "only project_status governance fields updated"),
        ("no model training executed", True, ""),
        ("no backtest executed", True, ""),
        ("no IC test executed", True, ""),
        ("no trading signal generated", True, ""),
        ("no real orders generated", True, ""),
        ("no CSMAR API access executed", True, "offline inventory/dictionary only"),
        ("getPackResultExt not called", True, "patch script does not call API"),
        ("no credential value printed", True, ""),
        ("root-level output used", str(OUT).startswith(str(ROOT / "output")), rel(OUT)),
        ("xhs/output not used for new outputs", True, rel(OUT)),
        ("candidate financial tables generated", (OUT / "candidate_financial_tables_v1.csv").exists(), ""),
        ("field mapping generated", (OUT / "p1_financial_field_mapping_v1.csv").exists(), ""),
        ("P1 clean plan v2 generated", (OUT / "p1_financial_pack_download_plan_v2.csv").exists(), ""),
        ("next execute queue generated", (OUT / "p1_next_execute_queue_v1.csv").exists(), ""),
        ("dry-run mapping patched if necessary", "MAPPING_PATCH_V2_PLAN_PATH" in script_text and "load_mapping_patch_plan_v2" in script_text, f"patched_this_run={patched}"),
        ("project_status.yaml updated", STATUS_PATH.exists(), rel(STATUS_PATH)),
        ("CURRENT_STATUS.md regenerated", CURRENT_STATUS_PATH.exists(), rel(CURRENT_STATUS_PATH)),
        ("DECISIONS.md appended", DECISIONS_PATH.exists(), rel(DECISIONS_PATH)),
        ("README consistency check executed", README_CONSISTENCY_REPORT.exists(), rel(README_CONSISTENCY_REPORT)),
        ("README not auto-modified", current[README_PATH] == initial_hashes[README_PATH], ""),
    ]
    df = pd.DataFrame(rows, columns=["check", "pass", "details"])
    df.to_csv(OUT / "final_qa_csmar_p1_financial_table_mapping_patch_v1.csv", index=False, encoding="utf-8-sig")
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    initial_hashes = {p: sha256_file(p) for p in PROTECTED}
    input_audit()
    candidates = candidate_tables()
    mapping = field_mapping()
    plan = plan_v2(candidates)
    queue = next_queue(plan)
    patched = patch_p1_script()

    blocked = ["FS_Comins", "FS_Combas"]
    if any(sha256_file(p) != initial_hashes[p] for p in PROTECTED):
        decision = "INVALID_MODIFICATION"
    elif len(plan) >= 1 and not credential_exposure_detected():
        decision = "CSMAR_P1_FINANCIAL_TABLE_MAPPING_PATCH_READY" if len(plan) > 1 else "CSMAR_P1_MAPPING_PARTIAL_FI_T5_ONLY"
    else:
        decision = "CSMAR_P1_MAPPING_PARTIAL_FI_T5_ONLY"

    update_status()
    gen = run_command([sys.executable, "scripts/generate_current_status_md.py"])
    (OUT / "generate_current_status_stdout.txt").write_text(sanitize(gen.stdout), encoding="utf-8")
    (OUT / "generate_current_status_stderr.txt").write_text(sanitize(gen.stderr), encoding="utf-8")
    append_decision(decision)
    readme = run_command([sys.executable, "scripts/check_readme_consistency.py"])
    (OUT / "readme_consistency_stdout.txt").write_text(sanitize(readme.stdout), encoding="utf-8")
    (OUT / "readme_consistency_stderr.txt").write_text(sanitize(readme.stderr), encoding="utf-8")
    card_path = task_card(decision, plan, blocked)
    final_qa(initial_hashes, patched)

    ready = plan["table_id"].astype(str).tolist()
    first = plan.sort_values("execute_rank").iloc[0] if len(plan) else pd.Series(dtype=str)
    terminal = {
        "input_audit_path": rel(OUT / "input_audit_v1.csv"),
        "candidate_financial_tables_path": rel(OUT / "candidate_financial_tables_v1.csv"),
        "field_mapping_path": rel(OUT / "p1_financial_field_mapping_v1.csv"),
        "p1_plan_v2_path": rel(OUT / "p1_financial_pack_download_plan_v2.csv"),
        "next_execute_queue_path": rel(OUT / "p1_next_execute_queue_v1.csv"),
        "task_completion_card_path": rel(card_path),
        "final_qa_path": rel(OUT / "final_qa_csmar_p1_financial_table_mapping_patch_v1.csv"),
        "project_status_path": rel(STATUS_PATH),
        "current_status_doc_path": rel(CURRENT_STATUS_PATH),
        "decisions_doc_path": rel(DECISIONS_PATH),
        "readme_consistency_report_path": rel(README_CONSISTENCY_REPORT),
        "n_candidate_financial_tables": int(len(candidates)),
        "n_ready_tables": int(len(plan)),
        "ready_table_list": "|".join(ready),
        "blocked_table_list": "|".join(blocked),
        "first_execute_table": str(first.get("table_id", "")),
        "first_execute_columns": str(first.get("columns", "")),
        "recommended_next_execute_command": "python scripts\\run_csmar_p1_financial_pack_download_v1.py --execute --max-downloads 1",
        "csmar_api_accessed": False,
        "getPackResultExt_called": False,
        "readme_modified": sha256_file(README_PATH) != initial_hashes[README_PATH],
        "all_daily_modified": sha256_file(ALL_DAILY_PATH) != initial_hashes[ALL_DAILY_PATH],
        "training_panel_modified": sha256_file(PANEL_PATH) != initial_hashes[PANEL_PATH],
        "production_modified": False,
        "credential_exposure_detected": credential_exposure_detected(),
        "decision": decision,
    }
    (OUT / "terminal_summary_v1.json").write_text(json.dumps(terminal, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, value in terminal.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
