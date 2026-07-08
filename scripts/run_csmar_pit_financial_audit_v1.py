from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.csmar.credential_loader import load_csmar_credentials


def main() -> int:
    status = load_csmar_credentials()
    out = ROOT / "output" / "csmar_pit_financial_audit_v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "README_WRAPPER.md").write_text(
        "\n".join([
            "# CSMAR PIT Financial Audit v1 Root Wrapper",
            "",
            "This root-level wrapper is prepared for canonical output under `output/csmar_pit_financial_audit_v1`.",
            "Legacy xhs outputs remain historical references and are not deleted.",
            "The legacy script still contains xhs-bound paths, so this wrapper does not execute it automatically during project promotion.",
            f"credential_source: {status['source']}",
            f"account_present: {status['account_present']}",
            f"password_present: {status['password_present']}",
            "",
        ]),
        encoding="utf-8",
    )
    print("CSMAR PIT financial audit root wrapper prepared; no audit executed.")
    print(f"credential_source: {status['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
