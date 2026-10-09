"""source: ingest hash-skips unchanged files, drops stale chunks after upsert, and uses stable ids."""

import uuid
from pathlib import Path

from qdrant_client.models import FieldCondition

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

    def embed_documents(self, texts):
        self.docs += 1
        return self.inner.embed_documents(texts)

    def embed_query(self, text):
        return self.inner.embed_query(text)


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


def test_speaker_from_title_falls_back_to_the_channel_handle():
    """source: speaker_from_title reads a short tail after a dash, else the channel handle."""
    assert speaker_from_title("A plain title", "@YourChannel") == "@YourChannel"
    titled = "A talk " + chr(0x2014) + " Alex Example"
    assert speaker_from_title(titled, "@YourChannel") == "Alex Example"
    assert speaker_from_title("A talk - Alex Example", "@YourChannel") == "Alex Example"
    ended = "A talk " + chr(0x2013) + " Alex Example"
    assert speaker_from_title(ended, "@YourChannel") == "Alex Example"
