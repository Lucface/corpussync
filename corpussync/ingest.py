"""Ingest files and YouTube captions. One core writes every point."""

import os
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, Range

from corpussync.chunking import chunk
from corpussync.extract import SUPPORTED_EXTENSIONS, extract_file
from corpussync.sparse import sparse_vector
from corpussync.state import content_hash, save_hash, saved_hash
from corpussync.store import ensure_collection, log, make_point_id, retry
from corpussync.vtt import clean_vtt

# 0.1 skipped caption files with fewer than 30 words of cleaned prose.
_MIN_YOUTUBE_WORDS = 30
_UPSERT_BATCH = 100


def speaker_from_title(title: str, fallback: str) -> str:
    """Pull a speaker from a title shaped like 'Talk - Speaker, Company'.

    When the title has no short tail after a dash, return the channel handle.
    """
    dashes = (chr(0x2014), " - ", chr(0x2013))
    for dash in dashes:
        if dash in title:
            tail = title.split(dash)[-1].strip()
            if tail and len(tail) < 80:
                return tail
    return fallback


def load_titles(titles_path: Path | None) -> dict[str, str]:
    """video-list.tsv = '<id>\\t<title>' per line."""
    titles: dict[str, str] = {}
    if not titles_path or not titles_path.exists():
        return titles
    for line in titles_path.read_text(encoding="utf-8", errors="replace").splitlines():
        vid, _, title = line.partition("\t")
        if vid.strip():
            titles[vid.strip()] = title.strip()
    return titles


def _dedupe_by_video(files: list[Path]) -> list[Path]:
    """Keep exactly ONE caption file per video_id. yt-dlp can emit several English
    variants for one video (id.en.vtt, id.en-orig.vtt, id.en-US.vtt); each ingests under
    a different source_file point-ID and would DUPLICATE the video in the corpus. Prefer
    the bare 'en' tag, then the shortest tag, then lexicographic -- deterministic."""
    def lang_tag(p: Path) -> str:
        parts = p.name.split(".")          # id.en-orig.vtt -> ["id","en-orig","vtt"]
        return parts[1] if len(parts) >= 3 else ""

    def score(p: Path) -> tuple:
        tag = lang_tag(p)
        return (tag != "en", len(tag), tag)      # en first, then shortest tag, then lexicographic

    best: dict[str, Path] = {}
    for path in files:
        vid = path.name.split(".")[0]
        if vid not in best or score(path) < score(best[vid]):
            best[vid] = path
    return sorted(best.values())


def _delete_stale(ctx, collection: str, source_file: str, keep: int) -> None:
    # After the upsert: drop chunks that belonged to a longer previous version.
    selector = Filter(must=[
        FieldCondition(key="source_file", match=MatchValue(value=source_file)),
        FieldCondition(key="chunk_index", range=Range(gte=keep)),
    ])
    retry(
        lambda: ctx.client.delete(collection_name=collection, points_selector=selector),
        f"delete-stale {source_file}",
    )


def _vector_for(layout: str, dense: list[float], text: str):
    if layout == "hybrid":
        return {"dense": dense, "bm25": sparse_vector(text)}
    if layout == "dense-named":
        return {"dense": dense}
    return dense


def ingest_document(
    ctx,
    *,
    text,
    source_file,
    corpus,
    title,
    locator,
    extra_payload=None,
    force=False,
) -> int:
    """Write one document. Returns chunks written, or 0 when the content hash is unchanged."""
    text = text or ""
    digest = content_hash(text)
    row = saved_hash(ctx.db, source_file)
    if row and row[0] == digest and not force:
        return 0

    chunks = chunk(text)
    collection = ctx.collection_name(corpus)
    layout = ensure_collection(ctx.client, collection, ctx.settings.embed_dim)
    now = datetime.now(timezone.utc).isoformat()

    if not chunks:
        if row:
            _delete_stale(ctx, collection, source_file, 0)
            save_hash(ctx.db, source_file, digest, now, collection, 0)
        return 0

    vectors = ctx.embedder.embed_documents(chunks)
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"embed count mismatch for {source_file}: {len(vectors)} vectors != {len(chunks)} chunks"
        )

    points = []
    total = len(chunks)
    for index, (piece, vector) in enumerate(zip(chunks, vectors)):
        payload = {
            "source_name": corpus,
            "source_file": source_file,
            "title": title,
        }
        if extra_payload:
            payload.update(extra_payload)
        else:
            payload["doc_type"] = "file"
            payload["source"] = "files"
            payload["locator"] = locator
            file_path = Path(source_file)
            if file_path.exists():
                modified = datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc)
                payload["modified_at"] = modified.isoformat()
            else:
                payload["modified_at"] = now
        payload["chunk_index"] = index
        payload["chunk_total"] = total
        payload["text"] = piece
        payload["ingested_at"] = now
        points.append(PointStruct(
            id=make_point_id(source_file, index),
            vector=_vector_for(layout, vector, piece),
            payload=payload,
        ))

    for start in range(0, len(points), _UPSERT_BATCH):
        batch = points[start:start + _UPSERT_BATCH]
        retry(
            lambda b=batch: ctx.client.upsert(collection_name=collection, points=b),
            f"upsert {source_file}",
        )
    _delete_stale(ctx, collection, source_file, total)
    save_hash(ctx.db, source_file, digest, now, collection, total)
    return total


def ingest_paths(ctx, paths: list[Path], corpus: str, force: bool = False) -> int:
    files = []
    missing = False
    for raw in paths:
        path = raw.expanduser()
        if not path.exists():
            print(f"not found: {path}")
            missing = True
            continue
        if path.is_file():
            if path.name.startswith("."):
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                print(f"skip {path}: unsupported file type")
                continue
            files.append((path, path.parent))
            continue
        if path.name.startswith("."):
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                file_path = Path(dirpath) / name
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append((file_path, path))
    total_files = 0
    total_chunks = 0
    for file_path, root in files:
        try:
            extracted = extract_file(file_path)
        except Exception as exc:
            log(f"FAILED {file_path.name}: {exc}")
            continue
        if extracted is None:
            continue
        text, title = extracted
        if not title:
            title = file_path.stem
        file_abs = file_path.resolve()
        root_abs = root.resolve()
        try:
            locator = file_abs.relative_to(root_abs).as_posix()
        except ValueError:
            locator = file_path.name
        try:
            written = ingest_document(
                ctx,
                text=text,
                source_file=str(file_abs),
                corpus=corpus,
                title=title,
                locator=locator,
                force=force,
            )
        except Exception as exc:
            log(f"FAILED {file_path.name}: {exc}")
            continue
        if written:
            total_files += 1
            total_chunks += written
            log(f"ingested {locator} -> {written} chunks")
    collection = ctx.collection_name(corpus)
    log(f"DONE: {total_files} files, {total_chunks} chunks -> {collection}")
    return 1 if missing else 0


def ingest_youtube(
    ctx,
    captions_dir: Path,
    titles_path: Path | None,
    source: str,
    channel: str,
    quality: str,
    force: bool,
) -> int:
    if not captions_dir.exists():
        log(f"ERROR: captions dir not found: {captions_dir}")
        return 0
    collection = ctx.collection_name(source)
    ensure_collection(ctx.client, collection, ctx.settings.embed_dim)
    titles = load_titles(titles_path)
    files = _dedupe_by_video(sorted(captions_dir.glob("*.vtt")))
    log(f"found {len(files)} caption files; {len(titles)} titles loaded -> {collection}")
    total_files = 0
    total_chunks = 0
    for path in files:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            text = clean_vtt(raw)
            if not text or len(text.split()) < _MIN_YOUTUBE_WORDS:
                log(f"skip (too little prose): {path.name}")
                continue
            video_id = path.name.split(".")[0]
            title = titles.get(video_id, video_id)
            author = speaker_from_title(title, channel)
            source_file = str(path)
            url = f"https://www.youtube.com/watch?v={video_id}"
            extra = {
                "doc_type": "talk",
                "source": "youtube",
                "channel": channel,
                "source_name": source,
                "source_file": source_file,
                "video_id": video_id,
                "url": url,
                "title": title,
                "author": author,
                "publish_date": "",
                "quality_tier": quality,
                "expires_at": None,
            }
            written = ingest_document(
                ctx,
                text=text,
                source_file=source_file,
                corpus=source,
                title=title,
                locator="",
                extra_payload=extra,
                force=force,
            )
        except Exception as exc:
            log(f"FAILED {path.name}: {exc}")
            continue
        if written:
            total_files += 1
            total_chunks += written
            log(f"ingested {video_id} -> {written} chunks [{author}]")
    log(f"DONE: {total_files} files, {total_chunks} chunks -> {collection}")
    return 0
