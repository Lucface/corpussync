"""source: defects 3, 5, 26 and 27, state belongs to one store and collection."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from corpussync.config import load_settings
from corpussync.state import (
    connect, delete_collection_rows, import_legacy_state, legacy_state_path,
    save_hash, saved_hash, store_identity,
)
from corpussync.store import legacy_store_choice, make_client


def make_legacy(path):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE file_hashes (path TEXT PRIMARY KEY, sha256 TEXT, ingested_at TEXT, collection TEXT, chunk_count INTEGER)")
    conn.execute("INSERT INTO file_hashes VALUES ('caption.vtt', 'digest', '', 'notes-corpus', 1)")
    conn.commit()
    conn.close()


def test_state_keys_and_scoped_removal(home):
    """source: defect 3, hash rows and deletion are scoped to store plus collection plus path."""
    db = connect(home / "state.db")
    try:
        for store, collection, digest in [("a", "notes", "one"), ("a", "other", "two"), ("b", "notes", "three")]:
            save_hash(db, store, "file.txt", digest, "", collection, 1)
        assert saved_hash(db, "a", "notes", "file.txt") == ("one",)
        delete_collection_rows(db, "a", "notes")
        assert saved_hash(db, "a", "notes", "file.txt") is None
        assert saved_hash(db, "a", "other", "file.txt") == ("two",)
        assert saved_hash(db, "b", "notes", "file.txt") == ("three",)
        assert db.execute("SELECT name FROM sqlite_master WHERE name='file_hashes'").fetchone() is None
    finally:
        db.close()


def test_store_identity(home):
    """source: defect 3, embedded paths resolve and server URLs have a stable trailing slash rule."""
    settings = load_settings()
    assert store_identity(settings) == "embedded:" + str((home / "qdrant").resolve())
    settings.qdrant_url = "http://server.example:6333/"
    assert store_identity(settings) == "server:http://server.example:6333"
    settings.qdrant_url = None
    settings.qdrant_host = "server.example"
    assert store_identity(settings) == "server:http://server.example:6333"


@pytest.mark.parametrize("in_open_db", [False, True])
def test_server_imports_legacy_once(home, capsys, in_open_db):
    """source: defect 26, server state imports old hashes once without rewriting file_hashes."""
    path = home / "state.db"
    make_legacy(path if in_open_db else home / "ingestion-state.db")
    settings = load_settings()
    settings.qdrant_url = "http://server.example:6333"
    db = connect(path)
    try:
        import_legacy_state(db, settings)
        identity = store_identity(settings)
        assert saved_hash(db, identity, "notes-corpus", "caption.vtt") == ("digest",)
        assert "imported 1 hash rows from 0.1 state" in capsys.readouterr().err
        delete_collection_rows(db, identity, "notes-corpus")
        import_legacy_state(db, settings)
        assert saved_hash(db, identity, "notes-corpus", "caption.vtt") is None
        assert capsys.readouterr().err == ""
    finally:
        db.close()


def test_embedded_never_imports_legacy(home):
    """source: defects 3 and 27, a fresh embedded store cannot inherit server skip state."""
    make_legacy(home / "ingestion-state.db")
    db = connect(home / "state.db")
    try:
        import_legacy_state(db, load_settings())
        assert db.execute("SELECT * FROM ingest_state").fetchall() == []
    finally:
        db.close()


def test_legacy_lookup_read_only_and_fallback(home, monkeypatch):
    """source: defect 27, invalid or empty state is ignored without creating a database."""
    settings = load_settings()
    missing = home / "missing.db"
    monkeypatch.setenv("CORPUSSYNC_STATE", str(missing))
    assert legacy_state_path(settings) is None
    assert not missing.exists()
    missing.write_text("not sqlite")
    assert legacy_state_path(settings) is None
    fallback = home / "ingestion-state.db"
    make_legacy(fallback)
    assert legacy_state_path(settings) == fallback
    missing.unlink()
    make_legacy(missing)
    assert legacy_state_path(settings) == missing


def test_legacy_choice_no_state_does_not_probe(home, capsys):
    """source: defect 27, no 0.1 state means embedded without a probe or server notice."""
    calls = []
    assert legacy_store_choice(load_settings(), lambda *a, **k: calls.append(a)) == "embedded"
    assert calls == []
    assert capsys.readouterr().err == ""


def test_legacy_choice_answering_server(home, monkeypatch, capsys):
    """source: defects 16 and 27, old state and an answering server preserve the 0.1 destination."""
    path = home / "ingestion-state.db"
    make_legacy(path)
    monkeypatch.setenv("CORPUSSYNC_LEGACY_QDRANT", "http://legacy.example:6333/")
    calls = []

    def probe(url, timeout):
        calls.append((url, timeout))
        return SimpleNamespace(status_code=200)

    settings = load_settings()
    assert legacy_store_choice(settings, probe) == "server"
    assert settings.qdrant_url == "http://legacy.example:6333"
    assert calls == [("http://legacy.example:6333/collections", 1)]
    assert str(path) in capsys.readouterr().err


def test_legacy_choice_failure_stops_compat(home, capsys):
    """source: defect 27, missing legacy server exits 2 instead of forking private corpora."""
    from corpussync.cli import main_compat

    make_legacy(home / "ingestion-state.db")

    def probe(*args, **kwargs):
        raise OSError("offline")

    assert main_compat(["--source", "notes", "--stats"], probe=probe) == 2
    assert "set CORPUSSYNC_STORE=embedded" in capsys.readouterr().err
    assert not (home / "qdrant").exists()
    assert not (home / "state.db").exists()


def test_embedded_override_does_not_probe(home, monkeypatch):
    """source: defect 27, the explicit embedded override starts fresh without probing."""
    make_legacy(home / "ingestion-state.db")
    monkeypatch.setenv("CORPUSSYNC_STORE", "embedded")
    calls = []
    assert legacy_store_choice(load_settings(), lambda *a, **k: calls.append(a)) == "embedded"
    assert calls == []


def test_explicit_server_wins_without_probe(home, monkeypatch):
    """source: defect 27, an explicit server wins even when embedded is requested in the environment."""
    monkeypatch.setenv("CORPUSSYNC_STORE", "embedded")
    settings = replace(load_settings(), qdrant_host="server.example")
    calls = []
    assert legacy_store_choice(settings, lambda *a, **k: calls.append(a)) == "server"
    assert calls == []


def test_private_permissions_are_tightened(ctx, tmp_path):
    """source: defect 5, existing corpussync directories tighten and state files are private."""
    home = ctx.settings.home
    home.chmod(0o755)
    store = home / "qdrant"
    store.mkdir(mode=0o755)
    store.chmod(0o755)
    parent = tmp_path / "separate" / "state"
    ctx.settings.state_path = parent / "state.db"
    ctx.client
    ctx.db
    assert home.stat().st_mode & 0o777 == 0o700
    assert store.stat().st_mode & 0o777 == 0o700
    assert parent.stat().st_mode & 0o777 == 0o700
    assert ctx.settings.state_path.stat().st_mode & 0o777 == 0o600


def test_busy_embedded_store_has_actionable_error(ctx, capsys):
    """source: defect 19, a second local client exits 2 with instructions and no raw traceback."""
    ctx.client
    with pytest.raises(SystemExit) as exc:
        make_client(ctx.settings)
    assert exc.value.code == 2
    assert "open in another corpussync process" in capsys.readouterr().err
