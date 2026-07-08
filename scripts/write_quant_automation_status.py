from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "output" / "automation_reliability_v1"
LATEST_PATH = OUT_DIR / "latest_automation_status.json"
HISTORY_PATH = OUT_DIR / "automation_status_history.jsonl"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "success"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write unified quant automation status.")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--stage", default="")
    parser.add_argument("--success", required=True)
    parser.add_argument("--failure-reason", default="")
    parser.add_argument("--latest-price-date", default="")
    parser.add_argument("--latest-nav-date", default="")
    parser.add_argument("--stale-price-warning", default="")
    parser.add_argument("--log-path", default="")
    parser.add_argument("--detail-path", default="")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "local_run_time": datetime.now().isoformat(timespec="seconds"),
        "task_name": args.task_name,
        "category": args.category,
        "status": args.status,
        "exit_code": args.exit_code,
        "stage": args.stage,
        "success": parse_bool(args.success),
        "failure_reason": args.failure_reason,
        "latest_price_date": args.latest_price_date,
        "latest_nav_date": args.latest_nav_date,
        "stale_price_warning": args.stale_price_warning,
        "log_path": args.log_path,
        "detail_path": args.detail_path,
        "notes": args.notes,
    }

    latest = {"last_updated": record["local_run_time"], "tasks": {}, "latest_record": record}
    if LATEST_PATH.exists():
        try:
            latest = json.loads(LATEST_PATH.read_text(encoding="utf-8"))
            if not isinstance(latest, dict):
                latest = {"tasks": {}}
        except json.JSONDecodeError:
            latest = {"tasks": {}}
    latest.setdefault("tasks", {})
    latest["last_updated"] = record["local_run_time"]
    latest["latest_record"] = record
    latest["tasks"][args.task_name] = record

    with HISTORY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    LATEST_PATH.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
