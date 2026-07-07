from __future__ import annotations

import os
from pathlib import Path

from data_sources.csmar.csmar_paths import LEGACY_XHS_ENV_LOCAL, ROOT_ENV_LOCAL


CredentialStatus = dict[str, bool | str]


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"CSMAR_ACCOUNT", "CSMAR_PASSWORD"} and value:
            values[key] = value
    return values


def _present() -> tuple[bool, bool]:
    return bool(os.getenv("CSMAR_ACCOUNT", "").strip()), bool(os.getenv("CSMAR_PASSWORD", "").strip())


def load_csmar_credentials() -> CredentialStatus:
    account_present, password_present = _present()
    if account_present and password_present:
        return {"account_present": True, "password_present": True, "source": "environment"}

    for source, path in (("root_env_local", ROOT_ENV_LOCAL), ("xhs_env_local", LEGACY_XHS_ENV_LOCAL)):
        file_values = _parse_env_file(path)
        for key in ("CSMAR_ACCOUNT", "CSMAR_PASSWORD"):
            if not os.getenv(key, "").strip() and file_values.get(key):
                os.environ[key] = file_values[key]
        account_present, password_present = _present()
        if account_present and password_present:
            return {"account_present": True, "password_present": True, "source": source}

    return {
        "account_present": account_present,
        "password_present": password_present,
        "source": "missing",
    }

