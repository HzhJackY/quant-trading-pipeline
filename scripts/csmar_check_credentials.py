from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.csmar.credential_loader import load_csmar_credentials


def main() -> int:
    status = load_csmar_credentials()
    print(f"CSMAR_ACCOUNT present: {status['account_present']}")
    print(f"CSMAR_PASSWORD present: {status['password_present']}")
    print(f"credential_source: {status['source']}")
    return 0 if status["account_present"] and status["password_present"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

