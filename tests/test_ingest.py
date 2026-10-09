"""source: ingest hash-skips unchanged files, drops stale chunks after upsert, and uses stable ids."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from corpussync.ingest import ingest_document, ingest_paths, speaker_from_title


def _words(count, token="word"):
    return " ".join(f"{token}{i}" for i in range(count))


def _scroll(ctx, corpus):
    records, _offset = ctx.client.scroll(
        collection_name=f"{corpus}-corpus",
        limit=50,
        with_payload=True,
        with_vectors=False,
    )
    return records


class _CountingEmbedder:
    def __init__(self, inner):
        self.inner = inner
        self.docs = 0

    def embed_documents(self, texts, prefixed=True):
        self.docs += 1
        return self.inner.embed_documents(texts, prefixed=prefixed)

    def embed_query(self, text, prefixed=True):
        return self.inner.embed_query(text, prefixed=prefixed)


class _RecordingClient:
    def __init__(self, client):
        self._client = client
        self.ops = []

    def upsert(self, *args, **kwargs):
        self.ops.append(("upsert", None))
        return self._client.upsert(*args, **kwargs)

    def delete(self, *args, **kwargs):
        self.ops.append(("delete", kwargs.get("points_selector")))
        return self._client.delete(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._client, name)


def test_second_ingest_of_unchanged_text_writes_nothing(ctx, tmp_path):
    """source: a second ingest of unchanged content writes nothing."""
    counter = _CountingEmbedder(ctx.embedder)
    ctx.embedder = counter
    source = str(tmp_path / "note.txt")
    text = _words(20)
    first = ingest_document(
        ctx, text=text, source_file=source, corpus="notes", title="Note", locator="note.txt",
    )
    assert first > 0
    assert counter.docs == 1
    second = ingest_document(
        ctx, text=text, source_file=source, corpus="notes", title="Note", locator="note.txt",
    )
    assert second == 0
    assert counter.docs == 1
    assert len(_scroll(ctx, "notes")) == first


def test_changing_a_file_removes_stale_chunks_after_upsert(ctx, tmp_path):
    """source: changing a file removes stale chunks after the upsert, never before it."""
    ctx.client
    recorder = _RecordingClient(ctx.client)
    ctx.client = recorder
    source = str(tmp_path / "doc.txt")
    first = ingest_document(
        ctx, text=_words(800), source_file=source, corpus="notes", title="Doc", locator="doc.txt",
    )
    assert first == 3
    mark = len(recorder.ops)
    second = ingest_document(
        ctx, text=_words(50), source_file=source, corpus="notes", title="Doc", locator="doc.txt",
    )
    assert second == 1
    kinds = [item[0] for item in recorder.ops[mark:]]
    assert "upsert" in kinds and "delete" in kinds
    assert kinds.index("upsert") < kinds.index("delete")
    selector = [item[1] for item in recorder.ops[mark:] if item[0] == "delete"][-1]
    ranges = [
        cond.range.gte
        for cond in selector.must
        if isinstance(cond, FieldCondition) and cond.range is not None
    ]
    assert 1 in ranges
    ids = {str(record.id) for record in _scroll(ctx, "notes")}
    assert str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::0")) in ids
    assert str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::1")) not in ids
    assert str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::2")) not in ids


def test_point_ids_are_deterministic(ctx, tmp_path):
    """source: point ids are uuid5(NAMESPACE_URL, source_file::index)."""
    source = str(tmp_path / "stable.txt")
    ingest_document(
        ctx, text=_words(12, "stable"), source_file=source, corpus="notes", title="Stable", locator="stable.txt",
    )
    expected = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::0"))
    ids = [str(record.id) for record in _scroll(ctx, "notes")]
    assert ids == [expected]


def test_hidden_files_are_skipped_and_title_falls_back_to_stem(ctx, tmp_path):
    """source: hidden files and folders are skipped; a missing heading uses the file stem."""
    root = tmp_path / "notes"
    root.mkdir()
    visible = root / "visible.txt"
    visible.write_text(_words(16, "shown") + "\n", encoding="utf-8")
    (root / ".secret.md").write_text("# Secret\n\n" + _words(16, "hidden") + "\n", encoding="utf-8")
    private = root / ".private"
    private.mkdir()
    (private / "nope.md").write_text("# Nope\n\n" + _words(16, "nope") + "\n", encoding="utf-8")
    assert ingest_paths(ctx, [root], "notes") == 0
    records = _scroll(ctx, "notes")
    files = [record.payload["source_file"] for record in records]
    assert any(path.endswith("visible.txt") for path in files)
    assert all(".secret.md" not in path for path in files)
    assert all("nope.md" not in path for path in files)
    payload = records[0].payload
    assert payload["title"] == "visible"
    assert payload["locator"] == "visible.txt"
    assert payload["doc_type"] == "file"
    assert payload["source"] == "files"
    assert Path(payload["source_file"]).is_absolute()
    assert "modified_at" in payload


def test_explicit_dot_folder_is_scanned(ctx, tmp_path):
    """source: round 4 item 3, an explicitly supplied dot folder must be scanned."""
    root = tmp_path / ".notes"
    root.mkdir()
    path = root / "idea.md"
    path.write_text("# Idea\n\nfive words make this note\n")
    assert ingest_paths(ctx, [root], "notes") == 0
    points = _scroll(ctx, "notes")
    assert len(points) == 1
    assert points[0].payload["source_file"] == str(path.resolve())


def test_explicit_parent_component_is_scanned(ctx, tmp_path):
    """source: round 4 item 3, a command line path ending in .. must scan its parent."""
    root = tmp_path / "a"
    sub = root / "sub"
    sub.mkdir(parents=True)
    path = root / "idea.md"
    path.write_text("# Idea\n\nfive words make this note\n")
    assert ingest_paths(ctx, [sub / ".."], "notes") == 0
    points = _scroll(ctx, "notes")
    assert len(points) == 1
    assert points[0].payload["source_file"] == str(path.resolve())


def test_explicit_dot_file_is_ingested(ctx, tmp_path):
    """source: round 4 item 3, an explicitly supplied dot file must be ingested."""
    path = tmp_path / ".idea.md"
    path.write_text("# Idea\n\nfive words make this note\n")
    assert ingest_paths(ctx, [path], "notes") == 0
    points = _scroll(ctx, "notes")
    assert len(points) == 1
    assert points[0].payload["source_file"] == str(path.resolve())


def test_speaker_from_title_falls_back_to_the_channel_handle():
    """source: speaker_from_title reads a short tail after a dash, else the channel handle."""
    assert speaker_from_title("A plain title", "@YourChannel") == "@YourChannel"
    titled = "A talk " + chr(0x2014) + " Alex Example"
    assert speaker_from_title(titled, "@YourChannel") == "Alex Example"
    assert speaker_from_title("A talk - Alex Example", "@YourChannel") == "Alex Example"
    ended = "A talk " + chr(0x2013) + " Alex Example"
    assert speaker_from_title(ended, "@YourChannel") == "Alex Example"


def test_same_file_enters_two_corpora(ctx, tmp_path):
    """source: defect 3, a matching file hash in one corpus cannot skip another corpus."""
    path = tmp_path / "note.txt"
    path.write_text("five words make this note")
    assert ingest_paths(ctx, [path], "first") == 0
    assert ingest_paths(ctx, [path], "second") == 0
    assert len(_scroll(ctx, "first")) == len(_scroll(ctx, "second")) == 1


def test_wiped_store_invalidates_every_row_before_partial_rebuild(ctx, tmp_path):
    """source: defects 3 and 26, recreating a collection clears all old hashes before one file is rebuilt."""
    root = tmp_path / "notes"
    root.mkdir()
    first = root / "first.txt"
    second = root / "second.txt"
    first.write_text("five words make this note")
    second.write_text("five words make another note")
    assert ingest_paths(ctx, [root], "notes") == 0
    counter = _CountingEmbedder(ctx.embedder)
    ctx.embedder = counter
    ctx.client.delete_collection("notes-corpus")
    assert ingest_paths(ctx, [first], "notes") == 0
    assert counter.docs == 1
    assert ingest_paths(ctx, [root], "notes") == 0
    assert counter.docs == 2
    assert len(_scroll(ctx, "notes")) == 2


def test_switching_store_does_not_reuse_hashes(ctx, tmp_path):
    """source: defect 3, a second embedded store receives files even with the same state database."""
    path = tmp_path / "note.txt"
    path.write_text("five words make this note")
    assert ingest_paths(ctx, [path], "notes") == 0
    ctx.close()
    ctx.settings.home = tmp_path / "other-home"
    assert ingest_paths(ctx, [path], "notes") == 0
    assert len(_scroll(ctx, "notes")) == 1


def test_prune_deleted_renamed_and_keep_missing(ctx, tmp_path, capsys):
    """source: defect 4, a complete folder scan prunes absent files unless keep-missing is set."""
    root = tmp_path / "notes"
    root.mkdir()
    path = root / "old.txt"
    path.write_text("five words make this note")
    assert ingest_paths(ctx, [root], "notes") == 0
    path.rename(root / "new.txt")
    assert ingest_paths(ctx, [root], "notes", keep_missing=True) == 0
    assert len(_scroll(ctx, "notes")) == 2
    assert ingest_paths(ctx, [root], "notes") == 0
    assert len(_scroll(ctx, "notes")) == 1
    assert "removed old.txt (no longer on disk)" in capsys.readouterr().out
    (root / "new.txt").unlink()
    assert ingest_paths(ctx, [root], "notes") == 0
    assert _scroll(ctx, "notes") == []


def test_prune_keeps_existing_file_inside_hidden_folder(ctx, tmp_path, capsys):
    """source: round 4 item 3, a file excluded from scanning keeps points while it still exists."""
    root = tmp_path / "notes"
    path = root / ".drafts" / "idea.md"
    path.parent.mkdir(parents=True)
    path.write_text("# Idea\n\nfive words make this note\n")
    selector = Filter(must=[
        FieldCondition(key="source_file", match=MatchValue(value=str(path.resolve()))),
    ])
    assert ingest_paths(ctx, [path], "notes") == 0
    before = ctx.client.count("notes-corpus", count_filter=selector, exact=True).count
    assert before == 1
    capsys.readouterr()
    assert ingest_paths(ctx, [root], "notes") == 0
    assert ctx.client.count("notes-corpus", count_filter=selector, exact=True).count == before
    assert ctx.db.execute(
        "SELECT path FROM ingest_state WHERE collection=? AND path=?",
        ("notes-corpus", str(path.resolve())),
    ).fetchall() == [(str(path.resolve()),)]
    output = capsys.readouterr().out
    assert not any("removed " in line for line in output.splitlines())
    assert "DONE: 0 files, 0 chunks, 0 removed, 0 skipped, 0 failed -> notes-corpus" in output


def test_file_and_missing_root_never_prune(ctx, tmp_path):
    """source: defect 4, individual files and missing roots cannot remove a folder's state."""
    root = tmp_path / "notes"
    root.mkdir()
    first, second = root / "first.txt", root / "second.txt"
    first.write_text("five words make this note")
    second.write_text("five words make another note")
    assert ingest_paths(ctx, [root], "notes") == 0
    second.unlink()
    assert ingest_paths(ctx, [first], "notes") == 0
    assert len(_scroll(ctx, "notes")) == 2
    root.rename(tmp_path / "renamed")
    assert ingest_paths(ctx, [root], "notes") == 1
    assert len(_scroll(ctx, "notes")) == 2


def test_failed_scan_never_prunes(ctx, tmp_path, monkeypatch):
    """source: defect 4, incomplete scans retain existing points and report failure."""
    root = tmp_path / "notes"
    root.mkdir()
    path = root / "note.txt"
    path.write_text("five words make this note")
    ingest_paths(ctx, [root], "notes")

    def failed_walk(path, onerror):
        onerror(OSError("scan interrupted"))
        return iter([])

    monkeypatch.setattr("corpussync.ingest.os.walk", failed_walk)
    assert ingest_paths(ctx, [root], "notes") == 1
    assert len(_scroll(ctx, "notes")) == 1


def test_five_word_note_becomes_one_point(ctx, tmp_path):
    """source: defect 12, notes below chunk's minimum still produce one normalized point."""
    path = tmp_path / "short.txt"
    path.write_text("five  words\nmake this\t note")
    assert ingest_paths(ctx, [path], "notes") == 0
    points = _scroll(ctx, "notes")
    assert len(points) == 1
    assert points[0].payload["text"] == "five words make this note"


def test_extraction_failure_and_write_failure_return_one(ctx, tmp_path, monkeypatch, capsys):
    """source: defect 8 and round 3 item 4, extraction and write errors count as failures rather than skips."""
    path = tmp_path / "note.txt"
    path.write_text("five words make this note")

    def fail(*args, **kwargs):
        raise ValueError("synthetic failure")

    with monkeypatch.context() as patch:
        patch.setattr("corpussync.ingest.extract_file", fail)
        assert ingest_paths(ctx, [path], "notes") == 1
    with monkeypatch.context() as patch:
        patch.setattr(ctx.embedder, "embed_documents", fail)
        assert ingest_paths(ctx, [path], "notes") == 1
    assert "DONE: 0 files, 0 chunks, 0 removed, 0 skipped, 1 failed -> notes-corpus" in capsys.readouterr().out


@pytest.mark.parametrize("extra,module", [("pdf", "pypdf"), ("docx", "docx")])
def test_missing_extra_skips_without_pruning(ctx, tmp_path, monkeypatch, capsys, extra, module):
    """source: round 3 item 4, missing extras exit zero, count as skipped and preserve existing indexed files."""
    from corpussync.state import saved_hash, store_identity
    from tests.test_extract import _block_import

    root = tmp_path / "notes"
    root.mkdir()
    path = root / ("document." + extra)
    path.write_bytes(b"placeholder")
    ingest_document(ctx, text="previously extracted document text", source_file=str(path.resolve()),
                    corpus="notes", title="Document", locator=path.name)
    (root / "note.txt").write_text("new text note")
    _block_import(monkeypatch, module)
    capsys.readouterr()
    assert ingest_paths(ctx, [root], "notes") == 0
    assert "DONE: 1 files, 1 chunks, 0 removed, 1 skipped, 0 failed -> notes-corpus" in capsys.readouterr().out
    assert {point.payload["source_file"] for point in _scroll(ctx, "notes")} == {
        str(path.resolve()), str((root / "note.txt").resolve()),
    }
    assert saved_hash(ctx.db, store_identity(ctx.settings), "notes-corpus", str(path.resolve())) is not None


def test_caption_digest_matches_raw_legacy_hash_and_rebuild(ctx, tmp_path, capsys):
    """source: defect 26, raw VTT hashes skip old captions and collection recreation re-ingests every file."""
    from corpussync.ingest import ingest_youtube
    from corpussync.state import content_hash, save_hash, store_identity
    from corpussync.store import ensure_collection

    root = tmp_path / "captions"
    root.mkdir()
    raw = "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n" + _words(40) + "\n"
    first, second = root / "first.en.vtt", root / "second.en.vtt"
    first.write_text(raw)
    second.write_text(raw)
    ensure_collection(ctx.client, "notes-corpus", ctx.settings.embed_dim)
    save_hash(ctx.db, store_identity(ctx.settings), str(first), content_hash(raw), "", "notes-corpus", 1)
    counter = _CountingEmbedder(ctx.embedder)
    ctx.embedder = counter
    assert ingest_youtube(ctx, root, None, "notes", "@YourChannel", "silver", False) == 0
    assert counter.docs == 1
    assert len(_scroll(ctx, "notes")) == 1
    ctx.client.delete_collection("notes-corpus")
    assert ingest_youtube(ctx, root, None, "notes", "@YourChannel", "silver", False) == 0
    assert counter.docs == 3
    assert len(_scroll(ctx, "notes")) == 2
    assert "DONE: 2 files, 2 chunks, 0 failed -> notes-corpus" in capsys.readouterr().out


def test_legacy_talk_prefix_is_embedded_without_changing_payload(ctx, tmp_path, monkeypatch):
    """source: round 4 item 8, unnamed vectors retain the talk prefix and plain payload; hybrid stays plain."""
    from corpussync.ingest import ingest_youtube

    ctx.client.create_collection(
        "legacy-corpus",
        vectors_config=VectorParams(size=ctx.settings.embed_dim, distance=Distance.COSINE),
    )
    root = tmp_path / "captions"
    root.mkdir()
    (root / "demo.en.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n" + _words(800) + "\n",
    )
    titles = tmp_path / "titles.tsv"
    titles.write_text("demo\tA plain talk\n")
    calls = []
    original = ctx.embedder.embed_documents

    def record(texts, prefixed=True):
        calls.append((list(texts), prefixed))
        return original(texts, prefixed=prefixed)

    monkeypatch.setattr(ctx.embedder, "embed_documents", record)
    assert ingest_youtube(ctx, root, titles, "legacy", "@YourChannel", "silver", False) == 0
    points = sorted(_scroll(ctx, "legacy"), key=lambda point: point.payload["chunk_index"])
    texts = [text for batch, _prefixed in calls for text in batch]
    assert len(texts) == len(points) > 1
    assert all(prefixed is False for _batch, prefixed in calls)
    prefix = "[talk] [@YourChannel] [@YourChannel]: "
    assert all(text.startswith(prefix) for text in texts)
    assert texts == [prefix + point.payload["text"] for point in points]
    assert all(not point.payload["text"].startswith("[talk]") for point in points)

    calls.clear()
    assert ingest_youtube(ctx, root, titles, "hybrid", "@YourChannel", "silver", False) == 0
    hybrid = sorted(_scroll(ctx, "hybrid"), key=lambda point: point.payload["chunk_index"])
    texts = [text for batch, _prefixed in calls for text in batch]
    assert len(texts) == len(hybrid) == len(points)
    assert all(prefixed is True for _batch, prefixed in calls)
    assert all(not text.startswith("[talk]") for text in texts)
    assert texts == [point.payload["text"] for point in hybrid]
    assert texts == [point.payload["text"] for point in points]


@pytest.mark.parametrize("fields", [{}, {"channel": "@YourChannel"}, {"author": "Alex Example"}])
def test_legacy_talk_allows_missing_prefix_fields(ctx, tmp_path, monkeypatch, fields):
    """source: item E, a talk supplied through the Python API must tolerate missing channel or author."""
    ctx.client.create_collection(
        "legacy-corpus",
        vectors_config=VectorParams(size=ctx.settings.embed_dim, distance=Distance.COSINE),
    )
    calls = []
    original = ctx.embedder.embed_documents

    def record(texts, prefixed=True):
        calls.append((list(texts), prefixed))
        return original(texts, prefixed=prefixed)

    monkeypatch.setattr(ctx.embedder, "embed_documents", record)
    text = _words(40)
    assert ingest_document(
        ctx, text=text, source_file=str(tmp_path / "talk.txt"), corpus="legacy",
        title="A plain talk", locator="talk.txt", extra_payload={"doc_type": "talk", **fields},
    ) == 1
    channel, author = fields.get("channel", ""), fields.get("author", "")
    assert calls == [([f"[talk] [{channel}] [{author}]: {text}"], False)]
    points = _scroll(ctx, "legacy")
    assert len(points) == 1
    assert points[0].payload["text"] == text


def test_missing_and_failed_captions_return_one(ctx, tmp_path, monkeypatch, capsys):
    """source: defect 8, a missing caption directory or failed caption returns 1 and a final count."""
    from corpussync.ingest import ingest_youtube

    root = tmp_path / "captions"
    assert ingest_youtube(ctx, root, None, "notes", "@YourChannel", "silver", False) == 1
    root.mkdir()
    (root / "demo.en.vtt").write_text("WEBVTT\n\n" + _words(40))

    def fail(*args, **kwargs):
        raise ValueError("synthetic caption failure")

    monkeypatch.setattr(ctx.embedder, "embed_documents", fail)
    assert ingest_youtube(ctx, root, None, "notes", "@YourChannel", "silver", False) == 1
    assert "DONE: 0 files, 0 chunks, 1 failed -> notes-corpus" in capsys.readouterr().out


def test_one_collection_listing_per_ingest(ctx, tmp_path, monkeypatch):
    """source: defect 3, collection existence is fetched once and cached throughout an ingest run."""
    root = tmp_path / "notes"
    root.mkdir()
    for name in ("first", "second"):
        (root / (name + ".txt")).write_text("five words make this note")
    original = ctx.client.get_collections
    calls = []

    def listed():
        calls.append(True)
        return original()

    monkeypatch.setattr(ctx.client, "get_collections", listed)
    assert ingest_paths(ctx, [root], "notes") == 0
    assert calls == [True]
