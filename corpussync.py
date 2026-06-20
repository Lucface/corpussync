#!/usr/bin/env python3
"""
YouTube Channel Corpus Ingestion → Qdrant `<source>-corpus`

Ingests cleaned YouTube caption (.vtt) files for a whole channel into a dedicated
Qdrant collection, so a coding agent can query the channel as an intelligence
corpus. Uses a fixed embedder (nomic-embed-text, 768d), a hash-skip state DB so re-runs
only embed new/changed files, and a stable payload shape any Qdrant client can read.

The captions are pulled FREE with yt-dlp (no API, no transcription cost):
  yt-dlp --flat-playlist --print id "<channel>/videos" > all-ids.txt
  yt-dlp --write-auto-subs --sub-langs en --skip-download \
         -o "captions/%(id)s.%(ext)s" --batch-file all-ids.txt

The one new piece vs the Every ingest is clean_vtt(): YouTube auto-subs roll up
(each cue repeats the prior partial line) and embed inline <timing> tags. We strip
tags, decode entities, and collapse the rolling duplication into clean prose.

Usage:
  qdrant-youtube-ingest.py --captions DIR --titles video-list.tsv \
      --source aiengineer --channel "@aiDotEngineer"
  qdrant-youtube-ingest.py --source aiengineer --stats
  qdrant-youtube-ingest.py --captions DIR --source aiengineer --force
"""

import argparse
import os
import hashlib
import html
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range,
)

# ── CONFIG (mirrors the qdrant-*-ingest.py family) ──
QDRANT_HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", "11434"))
EMBED_MODEL = os.environ.get("CORPUSSYNC_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.environ.get("CORPUSSYNC_EMBED_DIM", "768"))
STATE_DB = Path(os.environ.get("CORPUSSYNC_STATE", str(Path.home() / ".corpussync" / "ingestion-state.db")))


def log(msg: str):
    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {msg}", flush=True)


# ── shared helpers (mirrored from qdrant-every-ingest.py) ──
def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)


def ensure_collection(client: QdrantClient, collection: str):
    existing = {c.name for c in _retry(lambda: client.get_collections(), "get_collections").collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        log(f"created collection {collection} ({EMBED_DIM}d cosine)")


def init_state_db():
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(STATE_DB))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS file_hashes (
            path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, ingested_at TEXT NOT NULL,
            collection TEXT NOT NULL, chunk_count INTEGER DEFAULT 0)"""
    )
    conn.commit()
    return conn


def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def make_point_id(source_file: str, i: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_file}::{i}"))


def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embed",
        json={"model": EMBED_MODEL, "input": texts}, timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _retry(fn, what: str, tries: int = 5, delay: float = 2.0):
    """Retry a Qdrant call through transient tunnel blips (a remote Qdrant (e.g. reached
    over an SSH tunnel) can intermittently refuse connections under load. Without a retry,
    one blip drops a whole file's ingest and a startup blip kills the run."""
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            if attempt < tries - 1:
                log(f"  retry {what} ({attempt + 1}/{tries}): {str(e)[:80]}")
                time.sleep(delay * (attempt + 1))
    raise last


# ── VTT cleaning (the new piece) ──
_TAG = re.compile(r"<[^>]+>")           # <00:00:15.480>, <c>, </c>
_CUE_ATTR = re.compile(r"(align|position):\S+")


def clean_vtt(raw: str) -> str:
    """YouTube auto-sub .vtt → clean prose.

    Auto-subs animate: each cue re-shows the previous line plus a few new words,
    so the same sentence appears 2-3 times. We strip headers, cue-timing lines,
    inline <timing> tags, decode HTML entities, then collapse the roll-up by
    keeping a line only when it adds new text (prefix-grow replace + containment
    skip + consecutive de-dupe).
    """
    kept: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("WEBVTT") or s.startswith("Kind:") or s.startswith("Language:"):
            continue
        if "-->" in s:
            continue
        s = _TAG.sub("", s)
        s = _CUE_ATTR.sub("", s)
        s = html.unescape(s).strip()
        # drop the speaker-change carets YouTube injects; keep the words
        s = s.lstrip("> ").strip()
        if not s:
            continue
        if kept:
            if s == kept[-1]:
                continue
            if s in kept[-1]:          # current is a substring of last → already have it
                continue
            if kept[-1] in s:          # last grew into current → replace with fuller line
                kept[-1] = s
                continue
        kept.append(s)
    text = " ".join(kept).replace("\x00", "")  # null bytes silently truncate SQLite TEXT
    return re.sub(r"\s+", " ", text).strip()


# ── chunking (flat-blob word-window variant of the Every chunker) ──
def chunk(text: str, max_tokens: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    if len(words) <= 10:
        return []
    words_per_chunk = int(max_tokens / 1.3)  # token ≈ words * 1.3
    chunks: list[str] = []
    i = 0
    while i < len(words):
        window = words[i:i + words_per_chunk]
        chunks.append(" ".join(window))
        if i + words_per_chunk >= len(words):
            break
        i += words_per_chunk - overlap
    return [c for c in chunks if len(c.split()) > 10]


def load_titles(titles_path: Path) -> dict[str, str]:
    """video-list.tsv = '<id>\\t<title>' per line."""
    titles: dict[str, str] = {}
    if not titles_path or not titles_path.exists():
        return titles
    for line in titles_path.read_text(encoding="utf-8", errors="replace").splitlines():
        vid, _, title = line.partition("\t")
        if vid.strip():
            titles[vid.strip()] = title.strip()
    return titles


def speaker_from_title(title: str, fallback: str) -> str:
    """AI Engineer titles are 'Talk Name — Speaker, Company'; pull the speaker.
    Other channels rarely use that format, so fall back to the channel handle
    instead of a hard-coded name (otherwise every YC/Hormozi author = 'AI Engineer')."""
    for dash in ("—", " - ", "–"):
        if dash in title:
            tail = title.split(dash)[-1].strip()
            if tail and len(tail) < 80:
                return tail
    return fallback


# ── ingest ──
def ingest_file(client, db, path: Path, collection: str, source: str,
                channel: str, quality: str, titles: dict[str, str], force: bool) -> int:
    raw = path.read_text(encoding="utf-8", errors="replace")
    source_file = str(path)
    h = file_hash(raw)
    row = db.execute("SELECT sha256 FROM file_hashes WHERE path=?", (source_file,)).fetchone()
    if row and row[0] == h and not force:
        return 0

    video_id = path.name.split(".")[0]
    title = titles.get(video_id, video_id)
    text = clean_vtt(raw)
    if not text or len(text.split()) < 30:
        log(f"skip (too little prose): {path.name}")
        return 0

    chunks = chunk(text)
    if not chunks:
        return 0

    # Pull a speaker from the title when it uses the "Talk — Speaker, Company"
    # convention; speaker_from_title() falls back to the channel handle otherwise.
    author = speaker_from_title(title, channel)
    url = f"https://www.youtube.com/watch?v={video_id}"
    now = datetime.now(timezone.utc).isoformat()

    prefixed = [f"[talk] [{channel}] [{author}]: {c}" for c in chunks]
    vectors: list[list[float]] = []
    for i in range(0, len(prefixed), 10):
        vectors.extend(embed_texts(prefixed[i:i + 10]))
    if len(vectors) != len(chunks):
        # Ollama returned the wrong number of embeddings — abort rather than let
        # zip() silently drop trailing chunks and record a wrong chunk_count.
        raise RuntimeError(
            f"embed count mismatch for {video_id}: {len(vectors)} vectors != {len(chunks)} chunks"
        )

    points = []
    for i, (c, v) in enumerate(zip(chunks, vectors)):
        points.append(PointStruct(
            id=make_point_id(source_file, i), vector=v,
            payload={
                "doc_type": "talk",
                "source": "youtube",
                "channel": channel,
                "source_name": source,
                "source_file": source_file,
                "video_id": video_id,
                "url": url,
                "title": title,
                "author": author,
                "publish_date": "",          # not pulled in the cheap flat-playlist path
                "quality_tier": quality,
                "chunk_index": i,
                "chunk_total": len(chunks),
                "text": c,
                "ingested_at": now,
                "expires_at": None,          # learning corpus — no TTL
            },
        ))
    for i in range(0, len(points), 100):
        batch = points[i:i + 100]
        _retry(lambda b=batch: client.upsert(collection_name=collection, points=b), f"upsert {video_id}")

    # Re-ingest only: now that the new points are upserted (deterministic IDs overwrite
    # chunks 0..N-1), drop any STALE extra chunks left over from a longer prior version.
    # Done AFTER upsert so the video is never momentarily absent from the collection.
    if row:
        _retry(lambda: client.delete(
            collection_name=collection,
            points_selector=Filter(must=[
                FieldCondition(key="source_file", match=MatchValue(value=source_file)),
                FieldCondition(key="chunk_index", range=Range(gte=len(chunks))),
            ]),
        ), f"delete-stale {video_id}")

    db.execute(
        "INSERT OR REPLACE INTO file_hashes (path, sha256, ingested_at, collection, chunk_count) VALUES (?,?,?,?,?)",
        (source_file, h, now, collection, len(chunks)),
    )
    db.commit()
    log(f"ingested {video_id} → {len(chunks)} chunks [{author}]")
    return len(chunks)


def _dedupe_by_video(files: list[Path]) -> list[Path]:
    """Keep exactly ONE caption file per video_id. yt-dlp can emit several English
    variants for one video (id.en.vtt, id.en-orig.vtt, id.en-US.vtt); each ingests under
    a different source_file point-ID and would DUPLICATE the video in the corpus. Prefer
    the bare 'en' tag, then the shortest tag, then lexicographic — deterministic."""
    def lang_tag(p: Path) -> str:
        parts = p.name.split(".")          # id.en-orig.vtt -> ["id","en-orig","vtt"]
        return parts[1] if len(parts) >= 3 else ""
    def score(p: Path) -> tuple:
        t = lang_tag(p)
        return (t != "en", len(t), t)      # en first, then shortest tag, then lexicographic
    best: dict[str, Path] = {}
    for p in files:
        vid = p.name.split(".")[0]
        if vid not in best or score(p) < score(best[vid]):
            best[vid] = p
    return sorted(best.values())


def run(captions_dir: Path, titles_path: Path, collection: str, source: str,
        channel: str, quality: str, force: bool):
    if not captions_dir.exists():
        log(f"ERROR: captions dir not found: {captions_dir}")
        return
    client = get_client()
    ensure_collection(client, collection)
    db = init_state_db()
    titles = load_titles(titles_path)
    files = _dedupe_by_video(sorted(captions_dir.glob("*.vtt")))
    log(f"found {len(files)} caption files; {len(titles)} titles loaded → {collection}")
    total_files = total_chunks = 0
    for f in files:
        try:
            n = ingest_file(client, db, f, collection, source, channel, quality, titles, force)
            if n:
                total_files += 1
                total_chunks += n
        except Exception as e:
            log(f"FAILED {f.name}: {e}")
    db.close()
    log(f"DONE: {total_files} files, {total_chunks} chunks → {collection}")


def stats(collection: str):
    client = get_client()
    try:
        info = client.get_collection(collection)
        print(f"{collection}: {info.points_count} points")
    except Exception as e:
        print(f"{collection}: not found ({e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", help="directory of .vtt caption files")
    ap.add_argument("--titles", help="video-list.tsv (id<TAB>title)")
    ap.add_argument("--source", required=True, help="short source name, e.g. aiengineer")
    ap.add_argument("--channel", default="", help="channel handle, e.g. @aiDotEngineer")
    ap.add_argument("--collection", help="override collection name (default <source>-corpus)")
    ap.add_argument("--quality", default="silver", help="quality tier (gold/silver/bronze)")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    collection = args.collection or f"{args.source}-corpus"
    if args.stats:
        stats(collection)
    else:
        if not args.captions:
            ap.error("--captions is required unless --stats")
        run(
            Path(args.captions).expanduser(),
            Path(args.titles).expanduser() if args.titles else None,
            collection, args.source, args.channel or args.source, args.quality, args.force,
        )
