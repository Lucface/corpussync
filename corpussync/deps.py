"""Check runtime dependencies before importing version-specific interfaces."""

from __future__ import annotations

from importlib import metadata
import re


def require_qdrant_client(minimum=(1, 10)) -> None:
    required = ".".join(str(part) for part in minimum)
    try:
        version = metadata.version("qdrant-client")
    except metadata.PackageNotFoundError:
        found = "not installed"
    else:
        parts = tuple(int(part) for part in re.findall(r"\d+", version)[:2])
        if parts >= minimum:
            return
        found = f"found {version}"
    raise SystemExit(
        f"corpussync needs qdrant-client {required} or newer ({found}). "
        "Run: pip install -e . in the corpussync folder"
    )
