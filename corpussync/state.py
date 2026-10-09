"""SQLite hash-skip state scoped to a store, collection, and source path."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
from pathlib import Path

from corpussync.config import private_directory


def connect(path: Path) -> sqlite3.Connection:
    private_directory(path.parent)
    conn = sqlite3.connect(str(path))
    path.chmod(0o600)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS ingest_state (
            store TEXT NOT NULL, collection TEXT NOT NULL, path TEXT NOT NULL,
            sha256 TEXT NOT NULL, ingested_at TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0, PRIMARY KEY (store, collection, path))"""
    )
    conn.execute("CREATE TABLE IF NOT EXISTS state_imports (store TEXT PRIMARY KEY)")
    conn.commit()
    return conn


def store_identity(settings) -> str:
    if settings.qdrant_url:
        return "server:" + settings.qdrant_url.rstrip("/")
    if settings.qdrant_host:
        return f"server:http://{settings.qdrant_host}:{settings.qdrant_port}"
    return "embedded:" + str((settings.home / "qdrant").resolve())


def _read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def legacy_state_path(settings) -> Path | None:
    candidates = []
    if os.environ.get("CORPUSSYNC_STATE"):
        candidates.append(Path(os.environ["CORPUSSYNC_STATE"]).expanduser())
    candidates.append(settings.home / "ingestion-state.db")
    for path in candidates:
        conn = None
        try:
            conn = _read_only(path)
            if conn.execute("SELECT 1 FROM file_hashes LIMIT 1").fetchone():
                return path
        except sqlite3.Error:
            pass
        finally:
            if conn is not None:
                conn.close()
    return None


def import_legacy_state(conn: sqlite3.Connection, settings) -> None:
    if not settings.using_server():
        return
    identity = store_identity(settings)
    if conn.execute("SELECT 1 FROM state_imports WHERE store=?", (identity,)).fetchone():
        return
    rows = []
    if not conn.execute("SELECT 1 FROM ingest_state WHERE store=? LIMIT 1", (identity,)).fetchone():
        legacy = None
        try:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='file_hashes'").fetchone():
                source = conn
            else:
                path = settings.home / "ingestion-state.db"
                source = legacy = _read_only(path)
            rows = source.execute(
                "SELECT collection, path, sha256, ingested_at, chunk_count FROM file_hashes"
            ).fetchall()
        except sqlite3.Error:
            rows = []
        finally:
            if legacy is not None:
                legacy.close()
    conn.executemany(
        "INSERT OR IGNORE INTO ingest_state "
        "(store, collection, path, sha256, ingested_at, chunk_count) VALUES (?,?,?,?,?,?)",
        [(identity, *row) for row in rows],
    )
    conn.execute("INSERT OR IGNORE INTO state_imports (store) VALUES (?)", (identity,))
    conn.commit()
    if rows:
        print(f"imported {len(rows)} hash rows from 0.1 state", file=sys.stderr)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def saved_hash(conn: sqlite3.Connection, store: str, collection: str, source_file: str):
    return conn.execute(
        "SELECT sha256 FROM ingest_state WHERE store=? AND collection=? AND path=?",
        (store, collection, source_file),
    ).fetchone()


def save_hash(conn, store, source_file, digest, ingested_at, collection, chunk_count) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO ingest_state "
        "(store, collection, path, sha256, ingested_at, chunk_count) VALUES (?,?,?,?,?,?)",
        (store, collection, source_file, digest, ingested_at, chunk_count),
    )
    conn.commit()


def delete_collection_rows(conn: sqlite3.Connection, store: str, collection: str) -> None:
    conn.execute("DELETE FROM ingest_state WHERE store=? AND collection=?", (store, collection))
    conn.commit()
