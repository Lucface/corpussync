"""source: the 0.1 shim and sync.sh keep their contracts."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from tests.conftest import REPO, isolated_env


def test_shim_reexports_clean_vtt_and_chunk():
    """source: corpussync.py re-exports clean_vtt and chunk."""
    spec = importlib.util.spec_from_file_location("corpussync_shim", REPO / "corpussync.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from corpussync import __version__
    from corpussync.vtt import clean_vtt

    raw = (REPO / "tests" / "fixtures" / "rolling.vtt").read_text(encoding="utf-8")
    assert module.clean_vtt(raw) == clean_vtt(raw)
    assert module.chunk("one two three four five six seven eight nine ten") == []
    assert __version__ == "0.2.0"


def test_compat_flags_parse_and_stats_runs(tmp_path):
    """source: corpussync.py 0.1 flags parse and --stats runs."""
    env = isolated_env(tmp_path / "home")
    help_run = subprocess.run(
        [sys.executable, str(REPO / "corpussync.py"), "--help"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert help_run.returncode == 0, help_run.stderr
    for flag in (
        "--captions", "--titles", "--source", "--channel",
        "--collection", "--quality", "--stats", "--force",
    ):
        assert flag in help_run.stdout
    stats = subprocess.run(
        [sys.executable, str(REPO / "corpussync.py"), "--source", "demo", "--stats"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert stats.returncode == 0, stats.stderr
    assert "demo-corpus" in stats.stdout
    assert "Traceback" not in stats.stderr
    assert "Traceback" not in stats.stdout


def test_sync_sh_parses_on_bash_without_mapfile():
    """source: bash -n sync.sh passes and sync.sh contains no mapfile."""
    text = (REPO / "sync.sh").read_text(encoding="utf-8")
    assert "mapfile" not in text
    assert "declare -A" not in text
    assert ",," not in text
    assert "corpussync.py" in text
    assert "--captions" in text
    assert "--titles" in text
    assert "--source" in text
    assert "--channel" in text
    assert "COOKIE_JAR" in text
    assert "exported from your own account" in text
    proc = subprocess.run(["bash", "-n", str(REPO / "sync.sh")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert Path(REPO / "sync.sh").exists()


def test_compat_remote_notice_precedes_legacy_probe(home, monkeypatch, capsys):
    """source: defects 6 and 27, a remote compat destination is disclosed before probing it."""
    from types import SimpleNamespace
    from corpussync.cli import main_compat
    from tests.test_state import make_legacy

    make_legacy(home / "ingestion-state.db")
    monkeypatch.setenv("CORPUSSYNC_LEGACY_QDRANT", "https://legacy.example")
    calls = []

    def probe(url, timeout):
        calls.append((url, timeout))
        assert capsys.readouterr().err == "note: document text goes to https://legacy.example (Qdrant)\n"
        return SimpleNamespace(status_code=503)

    assert main_compat(["--source", "notes", "--captions", str(home / "captions")], probe=probe) == 2
    assert calls == [("https://legacy.example/collections", 1)]
    assert "no Qdrant server answers" in capsys.readouterr().err
