from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(r"C:\dev\quant")
TASK_NAME = "update_csmar_field_dictionary_trd_co_v0"
TABLE_NAME = "TRD_Co"
DICT_DIR = ROOT / "output" / "csmar_field_dictionary_v0"
RUN_DIR = ROOT / "output" / "_agent_runs" / TASK_NAME
DES_FILE = ROOT / "data" / "csmar_exports" / "TRD_Co[DES][xlsx].txt"
XLSX_RECORDED_ONLY = str(ROOT / "data" / "csmar_exports" / "TRD_Co.xlsx")

EXPECTED_FIELDS = [
    "Stkcd",
    "Stknme",
    "Listdt",
    "Cuntrycd",
    "Conme",
    "Conme_en",
    "Indcd",
    "Indnme",
    "Nindcd",
    "Nindnme",
    "Nnindcd",
    "Nnindnme",
    "IndcdZX",
    "IndnmeZX",
    "Estbdt",
    "PROVINCE",
    "PROVINCECODE",
    "CITY",
    "CITYCODE",
    "OWNERSHIPTYPE",
    "OWNERSHIPTYPECODE",
    "Favaldt",
    "Curtrd",
    "Ipoprm",
    "Ipoprc",
    "Ipocur",
    "Nshripo",
    "Parvcur",
    "Ipodt",
    "Parval",
    "Sctcd",
    "Statco",
    "Crcd",
    "Statdt",
    "Commnt",
    "Markettype",
    "FormerCode",
]

TRD_CO_FIELDS = [
    "table_name",
    "field_name",
    "dtype",
    "chinese_name",
    "unit",
    "description",
    "value_mapping",
    "usage_note",
    "source_des_file",
    "xlsx_file_path_recorded_only",
    "xlsx_read",
    "parse_status",
]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def parse_value_mapping(description: str) -> str:
    if "=" not in description and "＝" not in description:
        return ""
    pairs = re.findall(r"([A-Za-z0-9]+)\s*[=＝]\s*([^,，、]+)", description)
    if not pairs:
        return ""
    return "; ".join(f"{key}={value.strip()}" for key, value in pairs)


def infer_dtype(field_name: str, chinese_name: str, description: str) -> str:
    text = f"{field_name} {chinese_name} {description}"
    lower = field_name.lower()
    if "日期" in text or lower.endswith("dt") or lower in {"listdt", "estbdt", "favaldt", "ipodt", "statdt"}:
        return "date/string"
    if "代码" in text or lower.endswith("cd") or "code" in lower or lower == "stkcd":
        return "string"
    if any(token in lower for token in ["prc", "prm", "shr", "val"]) or "价格" in text or "股数" in text:
        return "numeric"
    return "string"


def usage_note(field_name: str) -> str:
    notes = {
        "Nnindcd": "2012版证监会行业分类代码/名称。可作为公司行业分类字段候选。是否可用于 PIT 行业中性化取决于 TRD_Co 是否包含历史行业变更记录；不能仅凭字段存在直接认定为 PIT monthly industry source。",
        "Nnindnme": "2012版证监会行业分类代码/名称。可作为公司行业分类字段候选。是否可用于 PIT 行业中性化取决于 TRD_Co 是否包含历史行业变更记录；不能仅凭字段存在直接认定为 PIT monthly industry source。",
        "IndcdZX": "中国上市公司协会行业分类代码/名称。可作为 secondary industry classification。是否 PIT 可用需另行审计时间结构。",
        "IndnmeZX": "中国上市公司协会行业分类代码/名称。可作为 secondary industry classification。是否 PIT 可用需另行审计时间结构。",
        "Markettype": "市场类型 / 上市板块字段。可用于 A股市场过滤和 market segment exposure audit，但不是行业分类。可与 Curtrd / Cuntrycd 共同用于 A股人民币样本过滤策略，例如 Cuntrycd=10, Curtrd=CNY, Markettype in {1,4,16,32,64}。具体过滤规则需在后续数据审计中确认。",
        "OWNERSHIPTYPE": "上市公司经营性质 / 所有制字段。可用于 ownership exposure audit，不是行业分类。",
        "OWNERSHIPTYPECODE": "上市公司经营性质 / 所有制字段。可用于 ownership exposure audit，不是行业分类。",
        "Statco": "公司活动情况及情况变动日。可用于上市状态 / 停牌 / 退市状态审计。Statdt 不得自动解释为行业分类生效日期，除非后续数据结构审计证明行业字段随 Statdt 变化。",
        "Statdt": "公司活动情况及情况变动日。可用于上市状态 / 停牌 / 退市状态审计。Statdt 不得自动解释为行业分类生效日期，除非后续数据结构审计证明行业字段随 Statdt 变化。",
        "FormerCode": "股票曾用代码。可用于历史代码映射和 join miss 诊断。",
        "Curtrd": "可与 Cuntrycd / Markettype 共同用于 A股人民币样本过滤策略，例如 Cuntrycd=10, Curtrd=CNY, Markettype in {1,4,16,32,64}。具体过滤规则需在后续数据审计中确认。",
        "Cuntrycd": "可与 Curtrd / Markettype 共同用于 A股人民币样本过滤策略，例如 Cuntrycd=10, Curtrd=CNY, Markettype in {1,4,16,32,64}。具体过滤规则需在后续数据审计中确认。",
    }
    return notes.get(field_name, "")


def parse_des(path: Path) -> dict[str, dict[str, str]]:
    parsed: dict[str, dict[str, str]] = {}
    text = path.read_text(encoding="utf-8-sig")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+\[([^\]]*)\]\s*-\s*(.*)$", line)
        if not match:
            continue
        field_name, chinese_name, description = match.groups()
        parsed[field_name] = {
            "field_name": field_name,
            "chinese_name": chinese_name.strip(),
            "description": description.strip(),
            "value_mapping": parse_value_mapping(description),
            "dtype": infer_dtype(field_name, chinese_name, description),
        }
    return parsed


def build_trd_co_rows(parsed: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field_name in EXPECTED_FIELDS:
        record = parsed.get(field_name, {})
        parse_status = "DES_PARSED" if record else "DES_NOT_PROVIDED"
        rows.append(
            {
                "table_name": TABLE_NAME,
                "field_name": field_name,
                "dtype": record.get("dtype", "unknown"),
                "chinese_name": record.get("chinese_name", ""),
                "unit": "",
                "description": record.get("description", "DES txt 未提供该字段定义；未读取 xlsx，不做推断。"),
                "value_mapping": record.get("value_mapping", ""),
                "usage_note": usage_note(field_name),
                "source_des_file": str(DES_FILE),
                "xlsx_file_path_recorded_only": XLSX_RECORDED_ONLY,
                "xlsx_read": False,
                "parse_status": parse_status,
            }
        )
    for field_name, record in parsed.items():
        if field_name in EXPECTED_FIELDS:
            continue
        rows.append(
            {
                "table_name": TABLE_NAME,
                "field_name": field_name,
                "dtype": record.get("dtype", "unknown"),
                "chinese_name": record.get("chinese_name", ""),
                "unit": "",
                "description": record.get("description", ""),
                "value_mapping": record.get("value_mapping", ""),
                "usage_note": usage_note(field_name),
                "source_des_file": str(DES_FILE),
                "xlsx_file_path_recorded_only": XLSX_RECORDED_ONLY,
                "xlsx_read": False,
                "parse_status": "DES_PARSED_EXTRA_FIELD",
            }
        )
    return rows


def master_row_from_trd(row: dict[str, Any]) -> dict[str, Any]:
    is_key = row["field_name"] == "Stkcd"
    is_report_period = False
    is_publish_date = row["field_name"] in {"Listdt", "Estbdt", "Favaldt", "Ipodt", "Statdt"}
    pit_relevant = row["field_name"] in {"Listdt", "Favaldt", "Statdt"}
    notes = row["usage_note"]
    if row["parse_status"] == "DES_NOT_PROVIDED":
        notes = f"{notes} DES txt 未提供该字段定义；未读取 xlsx。".strip()
    return {
        "field_code": row["field_name"],
        "chinese_name": row["chinese_name"],
        "english_name": "",
        "normalized_name": row["field_name"].lower(),
        "description": row["description"],
        "source_description_file": str(DES_FILE),
        "source_table": TABLE_NAME,
        "statement_type": "company_basic_info",
        "is_key_field": str(is_key).lower(),
        "is_report_period_field": str(is_report_period).lower(),
        "is_publish_date_field": str(is_publish_date).lower(),
        "is_typrep_field": "false",
        "pit_relevant_flag": str(pit_relevant).lower(),
        "compact_f_usage_flag": "false",
        "v3_core_usage_flag": "false",
        "candidate_usage_flag": "false",
        "notes": notes,
    }


def update_master(trd_rows: list[dict[str, Any]]) -> bool:
    master_path = DICT_DIR / "csmar_field_dictionary_master.csv"
    if master_path.exists():
        fieldnames, rows = read_csv(master_path)
    else:
        fieldnames = [
            "field_code",
            "chinese_name",
            "english_name",
            "normalized_name",
            "description",
            "source_description_file",
            "source_table",
            "statement_type",
            "is_key_field",
            "is_report_period_field",
            "is_publish_date_field",
            "is_typrep_field",
            "pit_relevant_flag",
            "compact_f_usage_flag",
            "v3_core_usage_flag",
            "candidate_usage_flag",
            "notes",
        ]
        rows = []

    if not fieldnames:
        return False

    retained = [row for row in rows if row.get("source_table") != TABLE_NAME]
    new_rows = []
    for trd_row in trd_rows:
        master_row = master_row_from_trd(trd_row)
        new_rows.append({field: master_row.get(field, "") for field in fieldnames})
    write_csv(master_path, retained + new_rows, fieldnames)
    return True


def main() -> int:
    DICT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{run_timestamp}] start {TASK_NAME}")

    des_file_found = DES_FILE.exists()
    checked_candidate_paths = [str(DES_FILE)]
    xlsx_read = False

    if not des_file_found:
        final_decision = "CSMAR_FIELD_DICTIONARY_UPDATE_FAIL_DES_NOT_FOUND"
        summary = {
            "run_timestamp": run_timestamp,
            "table_name": TABLE_NAME,
            "des_file_found": False,
            "des_file_path": None,
            "checked_candidate_paths": checked_candidate_paths,
            "xlsx_file_path_recorded_only": XLSX_RECORDED_ONLY,
            "xlsx_read": xlsx_read,
            "field_count_parsed": 0,
            "expected_key_fields_covered": False,
            "master_dictionary_updated": False,
            "trd_co_dictionary_written": False,
            "industry_fields_detected": [],
            "markettype_field_detected": False,
            "ownership_fields_detected": [],
            "status_date_fields_detected": [],
            "final_decision": final_decision,
        }
        write_json(DICT_DIR / "csmar_field_dictionary_update_TRD_Co_summary.json", summary)
        return 1

    parsed = parse_des(DES_FILE)
    trd_rows = build_trd_co_rows(parsed)
    write_csv(DICT_DIR / "csmar_field_dictionary_TRD_Co.csv", trd_rows, TRD_CO_FIELDS)
    master_updated = update_master(trd_rows)

    parsed_expected = [field for field in EXPECTED_FIELDS if field in parsed]
    expected_key_fields_covered = all(field in parsed for field in EXPECTED_FIELDS)
    industry_fields = [
        field
        for field in ["Indcd", "Indnme", "Nindcd", "Nindnme", "Nnindcd", "Nnindnme", "IndcdZX", "IndnmeZX"]
        if field in [row["field_name"] for row in trd_rows]
    ]
    ownership_fields = [
        field for field in ["OWNERSHIPTYPE", "OWNERSHIPTYPECODE"] if field in [row["field_name"] for row in trd_rows]
    ]
    status_date_fields = [field for field in ["Statco", "Statdt"] if field in [row["field_name"] for row in trd_rows]]

    final_decision = (
        "CSMAR_FIELD_DICTIONARY_UPDATE_TRD_CO_COMPLETE"
        if expected_key_fields_covered and master_updated
        else "CSMAR_FIELD_DICTIONARY_UPDATE_WATCH_PARTIAL_FIELDS"
    )

    summary = {
        "run_timestamp": run_timestamp,
        "table_name": TABLE_NAME,
        "des_file_found": des_file_found,
        "des_file_path": str(DES_FILE),
        "xlsx_file_path_recorded_only": XLSX_RECORDED_ONLY,
        "xlsx_read": xlsx_read,
        "field_count_parsed": len(parsed),
        "expected_key_fields_covered": expected_key_fields_covered,
        "expected_fields_total": len(EXPECTED_FIELDS),
        "expected_fields_parsed_from_des": len(parsed_expected),
        "expected_fields_missing_from_des": [field for field in EXPECTED_FIELDS if field not in parsed],
        "master_dictionary_updated": master_updated,
        "trd_co_dictionary_written": True,
        "industry_fields_detected": industry_fields,
        "markettype_field_detected": "Markettype" in [row["field_name"] for row in trd_rows],
        "ownership_fields_detected": ownership_fields,
        "status_date_fields_detected": status_date_fields,
        "final_decision": final_decision,
    }
    write_json(DICT_DIR / "csmar_field_dictionary_update_TRD_Co_summary.json", summary)

    report = f"""# CSMAR Field Dictionary Update: TRD_Co

## 读取来源

- DES 文件：`{DES_FILE}`
- xlsx 文件：仅记录路径 `{XLSX_RECORDED_ONLY}`，未读取。

## 更新结果

- DES 解析字段数：`{len(parsed)}`
- 期望字段数：`{len(EXPECTED_FIELDS)}`
- 期望字段中 DES 覆盖数：`{len(parsed_expected)}`
- final_decision: `{final_decision}`

## 行业相关字段

检测并记录的行业字段：`{", ".join(industry_fields)}`

`Nnindcd / Nnindnme` 可作为 2012 版证监会行业分类候选；`IndcdZX / IndnmeZX` 可作为中国上市公司协会 secondary industry classification。是否可用于 PIT 行业中性化取决于 TRD_Co 是否包含历史行业变更记录，不能仅凭字段存在直接认定为 PIT monthly industry source。

## 非行业但可用于审计的字段

- `Markettype`: 市场类型 / 上市板块字段，可用于 A股市场过滤和 market segment exposure audit，但不是行业分类。
- `OWNERSHIPTYPE / OWNERSHIPTYPECODE`: 所有制字段，可用于 ownership exposure audit，不是行业分类。
- `Statco / Statdt`: 可用于上市状态 / 停牌 / 退市状态审计；`Statdt` 不得自动解释为行业分类生效日期，除非后续数据结构审计证明行业字段随 `Statdt` 变化。
- `FormerCode`: 可用于历史代码映射和 join miss 诊断。
"""
    (DICT_DIR / "csmar_field_dictionary_update_TRD_Co_report.md").write_text(report, encoding="utf-8")

    final_qa_rows = [
        {"check": "des_file_found", "passed": des_file_found, "notes": str(DES_FILE)},
        {"check": "xlsx_not_read", "passed": not xlsx_read, "notes": "TRD_Co.xlsx 未被读取。"},
        {"check": "trd_co_dictionary_written", "passed": True, "notes": "csmar_field_dictionary_TRD_Co.csv 已写入。"},
        {"check": "master_dictionary_updated", "passed": master_updated, "notes": "master 中替换/追加 TRD_Co 记录，不删除其他表。"},
        {"check": "expected_key_fields_covered", "passed": expected_key_fields_covered, "notes": "DES 文件未覆盖全部用户列出的字段。" if not expected_key_fields_covered else "全部覆盖。"},
    ]
    write_csv(DICT_DIR / "final_qa_TRD_Co_update.csv", final_qa_rows, ["check", "passed", "notes"])
    write_csv(DICT_DIR / "final_qa.csv", final_qa_rows, ["check", "passed", "notes"])

    terminal_summary = {
        "task_name": TASK_NAME,
        "run_timestamp": run_timestamp,
        "stdout_log": str(RUN_DIR / "run_stdout.txt"),
        "stderr_log": str(RUN_DIR / "run_stderr.txt"),
        "output_directory": str(DICT_DIR),
        "final_decision": final_decision,
        "xlsx_read": xlsx_read,
    }
    write_json(DICT_DIR / "terminal_summary_TRD_Co_update.json", terminal_summary)
    write_json(DICT_DIR / "terminal_summary.json", terminal_summary)

    completion_card = f"""# Task Completion Card

- task_name: `{TASK_NAME}`
- status: completed
- final_decision: `{final_decision}`
- des_file_path: `{DES_FILE}`
- xlsx_read: `False`
- output_directory: `{DICT_DIR}`
"""
    (DICT_DIR / "task_completion_card_TRD_Co_update.md").write_text(completion_card, encoding="utf-8")
    (DICT_DIR / "task_completion_card.md").write_text(completion_card, encoding="utf-8")

    run_state = f"""# RUN_STATE

任务：Update CSMAR Field Dictionary for TRD_Co v0
状态：完成

已读取：
- {DES_FILE}

未读取：
- {XLSX_RECORDED_ONLY}

输出：
- {DICT_DIR / "csmar_field_dictionary_master.csv"}
- {DICT_DIR / "csmar_field_dictionary_TRD_Co.csv"}
- {DICT_DIR / "csmar_field_dictionary_update_TRD_Co_summary.json"}
- {DICT_DIR / "csmar_field_dictionary_update_TRD_Co_report.md"}

final_decision: {final_decision}
"""
    (RUN_DIR / "RUN_STATE.md").write_text(run_state, encoding="utf-8")

    print(f"final_decision={final_decision}")
    print(f"field_count_parsed={len(parsed)}")
    print(f"xlsx_read={xlsx_read}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
