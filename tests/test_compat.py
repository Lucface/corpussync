"""source: the 0.1 shim and sync.sh keep their contracts."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

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
    """source: sync.sh must parse on system bash and avoid features missing from bash 3.2."""
    text = (REPO / "sync.sh").read_text(encoding="utf-8")
    for feature in ("mapfile", "declare -A", ",,", "readarray", "declare -n", "local -n", "coproc", "^^", "|&", "&>>"):
        assert feature not in text
    assert "corpussync.py" in text
    assert "--captions" in text
    assert "--titles" in text
    assert "--source" in text
    assert "--channel" in text
    assert "COOKIE_JAR" in text
    assert "exported from your own account" in text
    proc = subprocess.run(["/bin/bash", "-n", str(REPO / "sync.sh")], capture_output=True, text=True)
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


@pytest.mark.parametrize("override", [None, "custom"])
def test_compat_ingest_keeps_bare_collection_untouched(ctx, tmp_path, capsys, override):
    """source: the 0.1 ingest must use source-corpus or its explicit override even when a bare source collection exists."""
    from corpussync.cli import main_compat
    from corpussync.store import ensure_collection, point_count

    ensure_collection(ctx.client, "demo", ctx.settings.embed_dim)
    ctx.close()
    captions = tmp_path / "captions"
    captions.mkdir()
    (captions / "demo.en.vtt").write_text("WEBVTT\n\n" + " ".join(f"word{i}" for i in range(40)))
    args = ["--source", "demo", "--captions", str(captions)]
    if override:
        args.extend(["--collection", override])
    assert main_compat(args) == 0
    target = override or "demo-corpus"
    assert point_count(ctx.client, "demo") == 0
    assert point_count(ctx.client, target) == 1
    ctx.close()
    capsys.readouterr()
    stats_args = ["--source", "demo", "--stats"]
    if override:
        stats_args.extend(["--collection", override])
    assert main_compat(stats_args) == 0
    assert f"{target}: 1 points" in capsys.readouterr().out


def test_compat_port_only_probe(home, monkeypatch):
    """source: the compat privacy probe must pass settings through when only QDRANT_PORT was exported."""
    from types import SimpleNamespace
    from corpussync.cli import main_compat
    from tests.test_state import make_legacy

    make_legacy(home / "ingestion-state.db")
    monkeypatch.setenv("QDRANT_PORT", "16333")
    calls = []

    def probe(url, timeout):
        calls.append((url, timeout))
        return SimpleNamespace(status_code=503)

    assert main_compat(["--source", "demo", "--captions", str(home / "captions")], probe=probe) == 2
    assert calls == [("http://127.0.0.1:16333/collections", 1)]
