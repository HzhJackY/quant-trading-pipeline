from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shlex
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).resolve().with_name("project_monitor_config.json")
KEYWORDS = ("top picks", "target weights", "rebalance", "multiplier")


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "paper_trading" / "paper_trading_pipeline.py").exists():
            return parent
    return current.parents[1]


def load_config(root: Path) -> dict[str, Any]:
    default = {
        "project_name": "quant",
        "default_command": "python paper_trading/paper_trading_pipeline.py",
        "default_timeout_minutes": 45,
        "log_dir": "logs/daily_monitor",
        "status_dir": "output/project_monitoring",
        "notify_on_success": True,
        "notify_on_failure": True,
        "known_notes": [],
    }
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            default.update(json.load(fh))
    default["log_dir_path"] = root / default["log_dir"]
    default["status_dir_path"] = root / default["status_dir"]
    return default


def ensure_dirs(config: dict[str, Any], now: datetime) -> tuple[Path, Path]:
    monthly_log_dir = Path(config["log_dir_path"]) / now.strftime("%Y-%m")
    status_dir = Path(config["status_dir_path"])
    monthly_log_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    return monthly_log_dir, status_dir


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tail(text: str, lines: int = 80) -> str:
    parts = text.splitlines()
    return "\n".join(parts[-lines:]) if parts else ""


def relevant_lines(stdout: str, stderr: str) -> list[str]:
    matches: list[str] = []
    for line in (stdout + "\n" + stderr).splitlines():
        low = line.lower()
        if any(keyword in low for keyword in KEYWORDS):
            matches.append(line)
    return matches[:120]


def notify(title: str, message: str, success: bool) -> list[str]:
    notes: list[str] = []
    try:
        import winsound

        if success:
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        else:
            winsound.MessageBeep(winsound.MB_ICONHAND)
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except Exception as exc:
        notes.append(f"winsound notification failed: {exc}")

    ps = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null; "
        "$template=[Windows.UI.Notifications.ToastTemplateType]::ToastText02; "
        "$xml=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($template); "
        "$texts=$xml.GetElementsByTagName('text'); "
        f"$texts.Item(0).AppendChild($xml.CreateTextNode({json.dumps(title)})) | Out-Null; "
        f"$texts.Item(1).AppendChild($xml.CreateTextNode({json.dumps(message)})) | Out-Null; "
        "$toast=[Windows.UI.Notifications.ToastNotification]::new($xml); "
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Quant Daily Monitor').Show($toast)"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if completed.returncode != 0:
            notes.append(f"PowerShell toast failed: {tail(completed.stderr, 5) or completed.returncode}")
    except Exception as exc:
        notes.append(f"PowerShell toast failed: {exc}")
    return notes


def build_command(config: dict[str, Any], args: argparse.Namespace) -> list[str]:
    command = shlex.split(config["default_command"], posix=False)
    if command and command[0].lower() == "python":
        command[0] = sys.executable
    if args.skip_ingestion:
        command.append("--skip-ingestion")
    if args.force_rebalance:
        command.append("--force-rebalance")
    if args.date:
        command.extend(["--date", args.date])
    return command


def dry_run_checks(root: Path, monthly_log_dir: Path, status_dir: Path) -> tuple[bool, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, details: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "details": details})

    add("current working directory", Path.cwd().resolve() == root.resolve(), str(Path.cwd()))
    add("python path", bool(sys.executable), sys.executable)
    add("paper_trading pipeline exists", (root / "paper_trading" / "paper_trading_pipeline.py").exists(), "paper_trading/paper_trading_pipeline.py")
    add("daily report exists", (root / "monitoring" / "daily_report.py").exists(), "monitoring/daily_report.py")
    add("output exists", (root / "output").exists(), "output/")

    try:
        probe = root / "output" / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        add("output writable", True, str(root / "output"))
    except Exception as exc:
        add("output writable", False, str(exc))

    try:
        probe = monthly_log_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        add("logs writable", True, str(monthly_log_dir))
    except Exception as exc:
        add("logs writable", False, str(exc))

    try:
        probe = status_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        add("status writable", True, str(status_dir))
    except Exception as exc:
        add("status writable", False, str(exc))

    for module in ("factor_research", "paper_trading"):
        try:
            importlib.import_module(module)
            add(f"import {module}", True, module)
        except Exception as exc:
            add(f"import {module}", False, repr(exc))

    return all(item["pass"] for item in checks), checks


def rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def check_status(checks: list[dict[str, Any]], name: str) -> str:
    for item in checks:
        if item["check"] == name:
            return "PASS" if item["pass"] else "FAIL"
    return "FAIL"


def render_dry_run_summary(
    root: Path,
    config: dict[str, Any],
    status: dict[str, Any],
    checks: list[dict[str, Any]],
    latest_status_path: Path,
    report_path: Path,
    log_path: Path,
    verbose: bool,
) -> str:
    sep = "=" * 60
    decision = "DRY_RUN_PASS" if status["success"] else "DRY_RUN_FAIL"
    lines = [
        sep,
        "Quant Daily Paper Trading Monitor - DRY RUN",
        sep,
        f"project_root: {root}",
        f"cwd: {Path.cwd()}",
        f"python_executable: {sys.executable}",
        f"paper_trading_pipeline_exists: {check_status(checks, 'paper_trading pipeline exists')}",
        f"daily_report_exists: {check_status(checks, 'daily report exists')}",
        f"output_dir_writable: {check_status(checks, 'output writable')}",
        f"log_dir_writable: {check_status(checks, 'logs writable')}",
        f"import factor_research: {check_status(checks, 'import factor_research')}",
        f"import paper_trading: {check_status(checks, 'import paper_trading')}",
        f"latest_status: {rel_path(latest_status_path, root)}",
        f"report_file: {rel_path(report_path, root)}",
        f"log_file: {rel_path(log_path, root)}",
        f"decision: {decision}",
    ]
    if not status["success"]:
        lines.append(f"failure_reason: {status.get('failure_reason') or 'unknown'}")
    if status.get("exception_type"):
        lines.extend([
            f"exception_type: {status['exception_type']}",
            f"exception_message: {status.get('exception_message', '')}",
        ])
    if verbose:
        lines.extend([
            "-" * 60,
            "verbose:",
            f"sys_path_first_5: {json.dumps(sys.path[:5], ensure_ascii=False)}",
            f"config_path: {CONFIG_PATH}",
            f"status_dir: {config['status_dir_path']}",
            f"log_dir: {config['log_dir_path']}",
            "checks:",
        ])
        for item in checks:
            result = "PASS" if item["pass"] else "FAIL"
            lines.append(f"  - {item['check']}: {result} ({item['details']})")
    lines.append(sep)
    return "\n".join(lines)


def append_history(path: Path, status: dict[str, Any]) -> None:
    fields = [
        "run_id", "start_time", "end_time", "duration_seconds", "success",
        "return_code", "failure_reason", "command", "working_directory", "log_path",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: status.get(field, "") for field in fields})


def markdown_report(status: dict[str, Any], config: dict[str, Any], excerpts: list[str]) -> str:
    stdout_tail = tail(status.get("stdout", ""), 80)
    stderr_tail = tail(status.get("stderr", ""), 80)
    excerpt_text = "\n".join(excerpts) if excerpts else "未检测到 Top picks 摘录"
    notes = "\n".join(f"- {note}" for note in status.get("notes", []) + config.get("known_notes", []))
    return f"""# Quant Daily Paper Trading Monitor

- run_id: {status["run_id"]}
- success: {status["success"]}
- return_code: {status["return_code"]}
- start_time: {status["start_time"]}
- end_time: {status["end_time"]}
- duration_seconds: {status["duration_seconds"]}
- command: `{status["command"]}`
- working_directory: `{status["working_directory"]}`
- log_path: `{status["log_path"]}`
- failure_reason: {status.get("failure_reason") or ""}

## Key Excerpts

```text
{excerpt_text}
```

## Stdout Tail

```text
{stdout_tail}
```

## Stderr Tail

```text
{stderr_tail}
```

## Notes

{notes or "- None"}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily monitor for quant paper trading.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-ingestion", action="store_true")
    parser.add_argument("--force-rebalance", action="store_true")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--timeout-minutes", type=int, default=None)
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = find_project_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    now = datetime.now()
    config = load_config(root)
    monthly_log_dir, status_dir = ensure_dirs(config, now)
    run_id = now.strftime("%Y-%m-%d_%H%M%S")
    log_path = monthly_log_dir / f"daily_monitor_{run_id}.log"
    latest_status_path = status_dir / "latest_status.json"
    history_path = status_dir / "daily_status_history.csv"
    report_path = status_dir / f"daily_monitor_report_{now.strftime('%Y-%m-%d')}.md"

    start = datetime.now()
    status: dict[str, Any] = {
        "run_id": run_id,
        "start_time": start.isoformat(timespec="seconds"),
        "end_time": None,
        "duration_seconds": None,
        "success": False,
        "return_code": None,
        "failure_reason": "",
        "command": "",
        "working_directory": str(root),
        "log_path": str(log_path),
        "report_path": str(report_path),
        "dry_run": bool(args.dry_run),
        "stdout": "",
        "stderr": "",
        "notes": [],
        "exception_type": "",
        "exception_message": "",
    }
    checks: list[dict[str, Any]] = []

    try:
        if args.dry_run:
            ok, checks = dry_run_checks(root, monthly_log_dir, status_dir)
            status["success"] = ok
            status["return_code"] = 0 if ok else 2
            status["failure_reason"] = "" if ok else "dry-run checks failed"
            status["stdout"] = json.dumps(checks, ensure_ascii=False, indent=2)
            status["command"] = "dry-run checks only"
        else:
            command = build_command(config, args)
            status["command"] = subprocess.list2cmdline(command)
            completed = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=(args.timeout_minutes or config["default_timeout_minutes"]) * 60,
            )
            status["return_code"] = completed.returncode
            status["stdout"] = completed.stdout
            status["stderr"] = completed.stderr
            status["success"] = completed.returncode == 0
            if completed.returncode != 0:
                status["failure_reason"] = f"paper trading exited with return code {completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        status["return_code"] = 124
        status["stdout"] = exc.stdout or ""
        status["stderr"] = exc.stderr or ""
        status["failure_reason"] = f"timeout after {args.timeout_minutes or config['default_timeout_minutes']} minutes"
    except Exception as exc:
        status["return_code"] = 1
        status["stderr"] = traceback.format_exc()
        status["failure_reason"] = repr(exc)
        status["exception_type"] = type(exc).__name__
        status["exception_message"] = str(exc)
    finally:
        end = datetime.now()
        status["end_time"] = end.isoformat(timespec="seconds")
        status["duration_seconds"] = round((end - start).total_seconds(), 3)

    if not args.no_notify:
        should_notify = (
            (status["success"] and config.get("notify_on_success", True))
            or ((not status["success"]) and config.get("notify_on_failure", True))
        )
        if should_notify:
            status["notes"].extend(
                notify(
                    "Quant daily monitor OK" if status["success"] else "Quant daily monitor FAILED",
                    f"{status['run_id']} | return_code={status['return_code']}",
                    bool(status["success"]),
                )
            )

    log_body = (
        f"run_id={status['run_id']}\n"
        f"success={status['success']}\n"
        f"return_code={status['return_code']}\n"
        f"command={status['command']}\n"
        f"working_directory={status['working_directory']}\n\n"
        "=== STDOUT ===\n"
        f"{status['stdout']}\n\n"
        "=== STDERR ===\n"
        f"{status['stderr']}\n"
    )
    write_text(log_path, log_body)
    write_text(latest_status_path, json.dumps(status, ensure_ascii=False, indent=2))
    append_history(history_path, status)
    write_text(report_path, markdown_report(status, config, relevant_lines(status["stdout"], status["stderr"])))
    if args.dry_run:
        if not checks and status.get("stdout"):
            try:
                loaded = json.loads(status["stdout"])
                if isinstance(loaded, list):
                    checks = loaded
            except json.JSONDecodeError:
                checks = []
        print(
            render_dry_run_summary(
                root=root,
                config=config,
                status=status,
                checks=checks,
                latest_status_path=latest_status_path,
                report_path=report_path,
                log_path=log_path,
                verbose=args.verbose,
            ),
            flush=True,
        )
    return 0 if status["success"] else int(status["return_code"] or 1)


if __name__ == "__main__":
    raise SystemExit(main())
