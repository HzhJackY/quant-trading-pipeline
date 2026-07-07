from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSMAR_ROOT = PROJECT_ROOT / "data_sources" / "csmar"
CSMAR_EXPORT_DIR = PROJECT_ROOT / "data" / "csmar_exports"
CSMAR_OUTPUT_ROOT = PROJECT_ROOT / "output"
LEGACY_XHS_CSMAR_ROOT = PROJECT_ROOT / "xhs"
ROOT_ENV_LOCAL = PROJECT_ROOT / ".env.local"
LEGACY_XHS_ENV_LOCAL = PROJECT_ROOT / "xhs" / ".env.local"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")

