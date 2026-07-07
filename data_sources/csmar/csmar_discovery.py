from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from data_sources.csmar.credential_loader import load_csmar_credentials
from data_sources.csmar.csmar_paths import CSMAR_OUTPUT_ROOT


KEYWORDS = ["百度", "百度指数", "搜索", "网络关注", "投资者关注", "媒体关注", "舆情", "新闻", "Baidu", "Search", "SVI", "Attention"]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def serialize_raw(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=_json_default)
    except Exception:
        return json.dumps(_safe_text(value), ensure_ascii=False)


def normalize_records(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    if isinstance(result, pd.DataFrame):
        return result.to_dict(orient="records")
    if isinstance(result, dict):
        for key in ("data", "datas", "rows", "result", "items", "list"):
            value = result.get(key)
            if isinstance(value, list):
                return [x if isinstance(x, dict) else {"value": x} for x in value]
        return [result]
    if isinstance(result, (list, tuple)):
        return [x if isinstance(x, dict) else {"value": x} for x in result]
    return [{"value": result}]


def extract_name_desc(row: dict[str, Any], kind: str) -> tuple[str, str]:
    name_candidates = [f"{kind}Name", f"{kind}_name", f"{kind}_id", "database_name", "table_name", "field_name", "name", "Name", "tableCode", "code"]
    desc_candidates = [f"{kind}Desc", f"{kind}_desc", "database_desc", "table_desc", "field_desc", "desc", "description", "memo", "remark", "title"]
    name = next((_safe_text(row[k]) for k in name_candidates if k in row and row[k] not in (None, "")), "")
    desc = next((_safe_text(row[k]) for k in desc_candidates if k in row and row[k] not in (None, "")), "")
    if not name:
        name = next((_safe_text(v) for k, v in row.items() if "name" in k.lower() or "code" in k.lower()), "")
    if not desc:
        desc = next((_safe_text(v) for k, v in row.items() if any(t in k.lower() for t in ["desc", "title", "remark", "memo"])), "")
    return name, desc


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def discover_csmar_tables(output_dir: Path | None = None, refresh: bool = False) -> dict[str, Any]:
    status = load_csmar_credentials()
    out = output_dir or CSMAR_OUTPUT_ROOT / "csmar_discovery_v1"
    out.mkdir(parents=True, exist_ok=True)
    if not (status["account_present"] and status["password_present"]):
        write_csv(out / "csmar_discovery_status.csv", [{
            "status": "failed",
            "credential_source": status["source"],
            "n_databases": 0,
            "n_tables": 0,
            "n_candidates": 0,
            "sanitized_error_message": "CSMAR credentials missing.",
        }], ["status", "credential_source", "n_databases", "n_tables", "n_candidates", "sanitized_error_message"])
        return {"success": False, "credential_source": status["source"], "n_items": 0, "output_dir": str(out), "error": "CSMAR credentials missing."}

    try:
        from csmarapi.CsmarService import CsmarService
    except Exception as exc:
        msg = f"csmarapi import failed: {type(exc).__name__}"
        write_csv(out / "csmar_discovery_status.csv", [{"status": "failed", "credential_source": status["source"], "n_databases": 0, "n_tables": 0, "n_candidates": 0, "sanitized_error_message": msg}], ["status", "credential_source", "n_databases", "n_tables", "n_candidates", "sanitized_error_message"])
        return {"success": False, "credential_source": status["source"], "n_items": 0, "output_dir": str(out), "error": msg}

    db_csv = out / "csmar_databases.csv"
    tables_csv = out / "csmar_tables.csv"
    candidates_csv = out / "csmar_attention_candidate_tables.csv"
    logging.disable(logging.CRITICAL)
    try:
        csmar = CsmarService()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            csmar.login(os.environ["CSMAR_ACCOUNT"], os.environ["CSMAR_PASSWORD"])
            db_records = normalize_records(csmar.getListDbs())
            table_rows: list[dict[str, Any]] = []
            for db in db_records:
                db_name, _ = extract_name_desc(db, "database")
                if not db_name:
                    continue
                for tr in normalize_records(csmar.getListTables(db_name)):
                    table_name, table_desc = extract_name_desc(tr, "table")
                    table_rows.append({"database_name": db_name, "table_name": table_name, "table_desc": table_desc, "raw_json": serialize_raw(tr)})
        db_rows = [{"database_name": extract_name_desc(r, "database")[0] or extract_name_desc(r, "database")[1], "raw_json": serialize_raw(r)} for r in db_records]
        candidates = [r for r in table_rows if any(k.lower() in f"{r['table_name']} {r['table_desc']}".lower() for k in KEYWORDS)]
        write_csv(db_csv, db_rows, ["database_name", "raw_json"])
        write_csv(tables_csv, table_rows, ["database_name", "table_name", "table_desc", "raw_json"])
        write_csv(candidates_csv, candidates, ["database_name", "table_name", "table_desc", "raw_json"])
        write_csv(out / "csmar_discovery_status.csv", [{"status": "success", "credential_source": status["source"], "n_databases": len(db_rows), "n_tables": len(table_rows), "n_candidates": len(candidates), "sanitized_error_message": ""}], ["status", "credential_source", "n_databases", "n_tables", "n_candidates", "sanitized_error_message"])
        return {"success": True, "credential_source": status["source"], "n_items": len(table_rows), "output_dir": str(out), "error": ""}
    except Exception as exc:
        msg = f"{type(exc).__name__}: {str(exc)[:180]}"
        write_csv(out / "csmar_discovery_status.csv", [{"status": "failed", "credential_source": status["source"], "n_databases": 0, "n_tables": 0, "n_candidates": 0, "sanitized_error_message": msg}], ["status", "credential_source", "n_databases", "n_tables", "n_candidates", "sanitized_error_message"])
        return {"success": False, "credential_source": status["source"], "n_items": 0, "output_dir": str(out), "error": msg}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover CSMAR tables using root-level credential loading.")
    parser.add_argument("--output-dir", type=Path, default=CSMAR_OUTPUT_ROOT / "csmar_discovery_v1")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    result = discover_csmar_tables(args.output_dir, args.refresh)
    print(f"output_dir: {result['output_dir']}")
    print(f"credential_source: {result['credential_source']}")
    print(f"success: {result['success']}")
    print(f"n_tables_or_items_detected: {result['n_items']}")
    if result["error"]:
        print(f"sanitized_error_message: {result['error']}")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

