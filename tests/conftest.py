"""Shared fixtures. Every test stays offline and inside tmp paths."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_CLEAR = (
    "QDRANT_URL",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "CORPUSSYNC_STATE",
    "CORPUSSYNC_EMBED_MODEL",
    "CORPUSSYNC_EMBED_DIM",
    "CORPUSSYNC_STORE",
    "CORPUSSYNC_LEGACY_QDRANT",
    "OLLAMA_HOST",
    "OLLAMA_PORT",
)


def isolated_env(home: Path) -> dict:
    env = os.environ.copy()
    env["CORPUSSYNC_HOME"] = str(home)
    env["CORPUSSYNC_EMBEDDER"] = "fake"
    env["CORPUSSYNC_CHAT"] = "fake"
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    # Apple's Python 3.9 links LibreSSL, so importing requests makes urllib3 warn on stderr.
    env["PYTHONWARNINGS"] = "ignore:urllib3 v2 only supports OpenSSL"
    for key in _CLEAR:
        env.pop(key, None)
    return env


def run_cli(args, home: Path, cwd: Path | None = None):
    return subprocess.run(
        [sys.executable, "-m", "corpussync", *args],
        cwd=str(cwd or home),
        env=isolated_env(home),
        capture_output=True,
        text=True,
    )


@pytest.fixture
def home(tmp_path, monkeypatch):
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CORPUSSYNC_HOME", str(folder))
    monkeypatch.setenv("CORPUSSYNC_EMBEDDER", "fake")
    monkeypatch.setenv("CORPUSSYNC_CHAT", "fake")
    for key in _CLEAR:
        monkeypatch.delenv(key, raising=False)
    return folder


@pytest.fixture
def ctx(home):
    from corpussync.config import load_context

    context = load_context()
    try:
        yield context
    finally:
        context.close()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """source: round 2 tests never send HTTP requests, including a compat probe."""
    import requests

    def blocked(*args, **kwargs):
        raise AssertionError("test attempted an HTTP request")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
