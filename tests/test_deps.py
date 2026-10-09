"""source: old or missing qdrant-client installations fail before version-specific imports."""

from __future__ import annotations

from importlib import metadata
import subprocess
import sys

import pytest

from corpussync.deps import require_qdrant_client
from tests.conftest import REPO, isolated_env


@pytest.mark.parametrize("version", ["1.7.4", "1.10.0", "1.12.1"])
def test_qdrant_client_minimum_version(monkeypatch, version):
    """source: qdrant-client 1.7.4 lacks required interfaces while 1.10.0 and 1.12.1 are supported."""
    calls = []

    def installed(package):
        calls.append(package)
        return version

    monkeypatch.setattr(metadata, "version", installed)
    if version == "1.7.4":
        with pytest.raises(SystemExit) as exc:
            require_qdrant_client()
        assert str(exc.value) == (
            "corpussync needs qdrant-client 1.10 or newer (found 1.7.4). "
            "Run: pip install -e . in the corpussync folder"
        )
    else:
        require_qdrant_client()
    assert calls == ["qdrant-client"]


def test_missing_qdrant_client_names_missing_package(monkeypatch):
    """source: missing qdrant-client must report not installed with an installation command."""
    def missing(package):
        raise metadata.PackageNotFoundError(package)

    monkeypatch.setattr(metadata, "version", missing)
    with pytest.raises(SystemExit) as exc:
        require_qdrant_client()
    assert str(exc.value) == (
        "corpussync needs qdrant-client 1.10 or newer (not installed). "
        "Run: pip install -e . in the corpussync folder"
    )


@pytest.mark.parametrize("entry", ["command", "compat"])
def test_old_client_entry_points_exit_without_traceback(tmp_path, entry):
    """source: both CLI entry points must reject an old client before importing unavailable Qdrant models."""
    code = (
        "import importlib.metadata, runpy, sys\n"
        "importlib.metadata.version = lambda package: '1.7.4'\n"
        "sys.argv = ['corpussync', '--help']\n"
    )
    if entry == "command":
        code += "runpy.run_module('corpussync', run_name='__main__')\n"
    else:
        code += f"runpy.run_path({str(REPO / 'corpussync.py')!r}, run_name='__main__')\n"
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                          env=isolated_env(tmp_path / "home"), capture_output=True, text=True)
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "Traceback" not in proc.stderr
    assert [line for line in proc.stderr.splitlines() if line.strip()][-1] == (
        "corpussync needs qdrant-client 1.10 or newer (found 1.7.4). "
        "Run: pip install -e . in the corpussync folder"
    )
