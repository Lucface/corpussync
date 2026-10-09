"""SQLite hash-skip state. Table shape matches 0.1."""

import hashlib
import sqlite3
from pathlib import Path


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS file_hashes (
            path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, ingested_at TEXT NOT NULL,
            collection TEXT NOT NULL, chunk_count INTEGER DEFAULT 0)"""
    )
    conn.commit()
    return conn


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def saved_hash(conn: sqlite3.Connection, source_file: str):
    return conn.execute(
        "SELECT sha256 FROM file_hashes WHERE path=?",
        (source_file,),
    ).fetchone()


def save_hash(
    conn: sqlite3.Connection,
    source_file: str,
    digest: str,
    ingested_at: str,
    collection: str,
    chunk_count: int,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO file_hashes (path, sha256, ingested_at, collection, chunk_count) VALUES (?,?,?,?,?)",
        (source_file, digest, ingested_at, collection, chunk_count),
    )
    conn.commit()


def delete_collection_rows(conn: sqlite3.Connection, collection: str) -> None:
    conn.execute("DELETE FROM file_hashes WHERE collection=?", (collection,))
    conn.commit()
