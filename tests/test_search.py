"""source: hybrid ranking, multi-corpus coverage, the relevance floor, and 0.1 collections."""

from __future__ import annotations

import subprocess
import sys

import pytest

from qdrant_client.models import Distance, VectorParams

from corpussync.answer import ask
from corpussync.ingest import ingest_document
from corpussync.search import search
from tests.conftest import REPO, isolated_env, run_cli


def _words(count, token):
    return " ".join(f"{token}{i}" for i in range(count))


class _BoomChat:
    def __init__(self):
        self.calls = 0

    def complete(self, model, messages):
        self.calls += 1
        raise AssertionError("model was called")


def test_hybrid_ranks_exact_keyword_ahead_of_dense(ctx, tmp_path):
    """source: defect 13, hybrid admits full keywords below the dense relevance floor."""
    stop = "the the the the the the the the the the apple banana"
    keyword = "xylophone maintenance protocol for field repair of brass fittings and linen wraps"
    ingest_document(
        ctx, text=stop, source_file=str(tmp_path / "stop.txt"),
        corpus="notes", title="Stopwords", locator="stop.txt",
    )
    ingest_document(
        ctx, text=keyword, source_file=str(tmp_path / "key.txt"),
        corpus="notes", title="Keyword", locator="key.txt",
    )
    query = "the xylophone"
    dense = search(ctx, query, ["notes"], k=4, mode="dense")
    hybrid = search(ctx, query, ["notes"], k=4, mode="hybrid")
    assert dense.results[0].title == "Stopwords", [hit.title for hit in dense.results]
    assert hybrid.results[0].title == "Keyword", [(hit.title, round(hit.score, 4)) for hit in hybrid.results]
    assert "Keyword" not in [hit.title for hit in dense.results]
    assert hybrid.results[0].relevance < 0.65
    assert hybrid.results[0].keyword_coverage == 1.0
    assert "Stopwords" in [hit.title for hit in hybrid.results]


def test_multi_corpus_coverage_counts(ctx, tmp_path):
    """source: searching several corpora reports a hit count for each one."""
    sentence = "alpha zephyr calibration linen"
    ingest_document(
        ctx, text=" ".join([sentence] * 3), source_file=str(tmp_path / "a.txt"),
        corpus="alpha", title="Alpha", locator="a.txt",
    )
    sentence_b = "beta zephyr calibration linen"
    ingest_document(
        ctx, text=" ".join([sentence_b] * 3), source_file=str(tmp_path / "b.txt"),
        corpus="beta", title="Beta", locator="b.txt",
    )
    result = search(ctx, "zephyr calibration", ["alpha", "beta"], k=8, mode="hybrid")
    assert result.coverage["alpha"] == 1
    assert result.coverage["beta"] == 1
    assert {hit.corpus for hit in result.results} == {"alpha", "beta"}


def test_exit_code_2_when_scores_are_below_the_floor(tmp_path):
    """source: search exits 2 when every hit is below the relevance floor."""
    work = tmp_path / "work"
    work.mkdir()
    home = tmp_path / "home"
    notes = work / "notes"
    notes.mkdir()
    body = " ".join(["widget", "calibration", "procedure"] * 4)
    (notes / "widgets.md").write_text("# Widget notes\n\n" + body + "\n", encoding="utf-8")
    ingested = run_cli(["ingest", str(notes), "--corpus", "notes"], home, work)
    assert ingested.returncode == 0, ingested.stderr
    good = run_cli(["search", "widget calibration procedure", "--corpus", "notes"], home, work)
    assert good.returncode == 0, good.stdout + good.stderr
    cfg = work / "high.toml"
    cfg.write_text(
        "[search.min_score]\ndense = 1000000\nkeyword = 1000000\nhybrid = 1000000\n",
        encoding="utf-8",
    )
    bad = run_cli(
        ["--config", str(cfg), "search", "widget calibration procedure", "--corpus", "notes"],
        home,
        work,
    )
    assert bad.returncode == 2
    assert "no good match in: notes" in bad.stdout


def test_ask_does_not_call_the_model_below_the_floor(ctx, tmp_path, capsys):
    """source: ask exits 2 without calling the model when nothing clears the floor."""
    ingest_document(
        ctx, text=" ".join(["widget", "calibration", "procedure"] * 4),
        source_file=str(tmp_path / "widgets.txt"), corpus="notes", title="Widget notes", locator="widgets.txt",
    )
    ctx.chat = _BoomChat()
    ctx.settings.min_score = {"dense": 1000000.0, "keyword": 1000000.0, "hybrid": 1000000.0}
    code = ask(ctx, "widget calibration procedure", ["notes"], k=6)
    assert code == 2
    assert ctx.chat.calls == 0
    assert "no good match in: notes" in capsys.readouterr().out


def test_legacy_collection_searches_dense_with_notice(ctx, tmp_path):
    """source: a 0.1-layout collection searches in dense mode and prints the hybrid notice."""
    text = "legacy dense search keeps this caption text available for the corpus"
    ctx.client.create_collection(
        collection_name="old-corpus",
        vectors_config=VectorParams(size=ctx.settings.embed_dim, distance=Distance.COSINE),
    )
    written = ingest_document(
        ctx, text=text, source_file=str(tmp_path / "old.vtt"),
        corpus="old", title="Legacy Talk", locator="",
    )
    assert written == 1
    records, _offset = ctx.client.scroll(
        collection_name="old-corpus", limit=5, with_vectors=True, with_payload=True,
    )
    assert records
    assert isinstance(records[0].vector, list)
    result = search(ctx, text, ["old"], k=4, mode="hybrid")
    assert result.results
    assert result.results[0].title == "Legacy Talk"
    assert result.notices == ["old: --mode hybrid needs a fresh corpus (remove then ingest again)"]
    home = ctx.settings.home
    ctx.close()
    proc = subprocess.run(
        [sys.executable, "-m", "corpussync", "search", text, "--corpus", "old", "--mode", "hybrid"],
        cwd=str(REPO),
        env=isolated_env(home),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--mode hybrid needs a fresh corpus (remove then ingest again)" in proc.stderr
    assert "Legacy Talk" in proc.stdout


class _ControlledEmbedder:
    def __init__(self, dim, values):
        self.dim = dim
        self.values = values
        self.queries = []
        self.documents = []

    def embed_documents(self, texts, prefixed=True):
        import math

        self.documents.append(prefixed)
        return [[self.values[text], math.sqrt(1 - self.values[text] ** 2)] + [0.0] * (self.dim - 2) for text in texts]

    def embed_query(self, text, prefixed=True):
        self.queries.append(prefixed)
        return [1.0] + [0.0] * (self.dim - 1)


def test_global_ranking_prevents_one_point_corpus_from_winning(ctx, tmp_path):
    """source: defect 14, a one-point compost corpus cannot outrank a matching tire note."""
    values = {"compost garden soil": 0.7, "fix flat tire": 0.9, "wheel service": 0.8}
    ctx.embedder = _ControlledEmbedder(ctx.settings.embed_dim, values)
    for index, (text, corpus) in enumerate([("compost garden soil", "compost"), ("fix flat tire", "notes"), ("wheel service", "notes")]):
        ingest_document(ctx, text=text, source_file=str(tmp_path / str(index)), corpus=corpus, title=text, locator=str(index))
    for mode in ("dense", "hybrid"):
        result = search(ctx, "how do I fix a flat tire", ["compost", "notes"], mode=mode)
        assert result.results[0].title == "fix flat tire"
        assert result.results[0].relevance > 0.89
        assert result.results[-1].title == "compost garden soil"


def test_unrelated_query_exits_two_with_default_floors(tmp_path):
    """source: defect 14, an unrelated query cannot pass on a hybrid RRF score alone."""
    home = tmp_path / "home"
    path = tmp_path / "note.txt"
    path.write_text("fix flat tire repair wheel")
    assert run_cli(["ingest", str(path), "--corpus", "notes"], home, tmp_path).returncode == 0
    for command in ("search", "ask"):
        proc = run_cli([command, "quantum chromodynamics lattice gauge", "--corpus", "notes"], home, tmp_path)
        assert proc.returncode == 2, proc.stdout + proc.stderr
        assert "no good match in: notes" in proc.stdout


def test_new_sparse_vectors_use_idf(ctx, tmp_path):
    """source: defect 13, new sparse collections must enable Qdrant's inverse document frequency modifier."""
    from qdrant_client.models import Modifier

    ingest_document(ctx, text="five words make this note", source_file=str(tmp_path / "note"),
                    corpus="notes", title="Note", locator="note")
    info = ctx.client.get_collection("notes-corpus")
    assert info.config.params.sparse_vectors["bm25"].modifier == Modifier.IDF


def test_keyword_mode_never_calls_closed_embedder(ctx, tmp_path, monkeypatch):
    """source: defect 7, keyword search works with Ollama pointed at a closed port and no HTTP call."""
    from corpussync.embed import OllamaEmbedder

    ingest_document(ctx, text="five words make this note", source_file=str(tmp_path / "note"),
                    corpus="notes", title="Note", locator="note")
    ctx.embedder = OllamaEmbedder("127.0.0.1:1", 11434, "nomic-embed-text")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("keyword mode called Ollama")

    monkeypatch.setattr("requests.post", forbidden)
    result = search(ctx, "five words", ["notes"], mode="keyword")
    assert result.results and result.results[0].keyword_coverage == 1.0
    assert result.results[0].relevance is None
    assert calls == []


def test_partial_keyword_match_gets_no_keyword_rank(ctx, tmp_path):
    """source: defect 24, only full keyword coverage adds an RRF keyword term at the default floor."""
    values = {"should store clutter": 0.7, "should store tomatoes": 0.9}
    ctx.embedder = _ControlledEmbedder(ctx.settings.embed_dim, values)
    for index, text in enumerate(values):
        ingest_document(ctx, text=text, source_file=str(tmp_path / str(index)), corpus="notes", title=text, locator=str(index))
    result = search(ctx, "how should I store tomatoes", ["notes"])
    full, partial = result.results
    assert full.title == "should store tomatoes"
    assert full.keyword_coverage == 1.0
    assert partial.keyword_coverage == 2 / 3
    assert abs(full.score - 2 / 61) < 1e-9
    assert abs(partial.score - 1 / 62) < 1e-9


def test_all_skips_foreign_sparse_and_wrong_dimension(ctx, tmp_path):
    """source: defect 25, --all skips sparse-only and incompatible vectors while returning owned results."""
    from qdrant_client.models import SparseVectorParams

    ingest_document(ctx, text="five words make this note", source_file=str(tmp_path / "note"),
                    corpus="notes", title="Note", locator="note")
    ctx.client.create_collection("foreign-sparse", vectors_config={}, sparse_vectors_config={"bm25": SparseVectorParams()})
    ctx.client.create_collection("foreign-size", vectors_config={"dense": VectorParams(size=3, distance=Distance.COSINE)})
    home = ctx.settings.home
    ctx.close()
    proc = run_cli(["search", "five words", "--all"], home, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Note" in proc.stdout
    assert "foreign-sparse: skipped (not a corpussync corpus)" in proc.stderr
    assert "foreign-size: skipped (not a corpussync corpus)" in proc.stderr
    listed = run_cli(["list"], home, tmp_path)
    assert "foreign  (foreign-sparse)" in listed.stdout
    answer = run_cli(["ask", "five words", "--all"], home, tmp_path)
    assert answer.returncode == 0
    assert "foreign-sparse: skipped" in answer.stderr


def test_failing_collection_does_not_abort_search(ctx, tmp_path, monkeypatch):
    """source: defect 25, one per-collection query failure leaves other corpus results available."""
    for corpus in ("bad", "good"):
        ingest_document(ctx, text="five words make this note", source_file=str(tmp_path / corpus),
                        corpus=corpus, title=corpus, locator=corpus)
    original = ctx.client.query_points

    def query(**kwargs):
        if kwargs["collection_name"] == "bad-corpus":
            raise ValueError("synthetic failure " + "x" * 100)
        return original(**kwargs)

    monkeypatch.setattr(ctx.client, "query_points", query)
    result = search(ctx, "five words", ["bad", "good"])
    assert [hit.corpus for hit in result.results] == ["good"]
    assert result.notices == ["bad: skipped (" + ("synthetic failure " + "x" * 100)[:80] + ")"]


def test_prefix_style_is_shared_once_and_legacy_is_unprefixed(ctx, tmp_path):
    """source: defect 17, old vectors retain unprefixed ingest and queries; each style embeds once."""
    text = "five words make this note"
    ctx.embedder = _ControlledEmbedder(ctx.settings.embed_dim, {text: 0.9})
    ctx.client.create_collection("old-corpus", vectors_config=VectorParams(size=ctx.settings.embed_dim, distance=Distance.COSINE))
    for corpus in ("old", "first", "second"):
        ingest_document(ctx, text=text, source_file=str(tmp_path / corpus), corpus=corpus, title=corpus, locator=corpus)
    assert ctx.embedder.documents == [False, True, True]
    assert len(search(ctx, text, ["old", "first", "second"]).results) == 3
    assert ctx.embedder.queries == [False, True]
    result = search(ctx, text, ["old"], mode="keyword")
    assert result.results == []
    assert result.notices == ["old: keyword mode needs a fresh corpus (remove then ingest again)"]
    assert ctx.embedder.queries == [False, True]


def test_unicode_keyword_tokens_and_weights():
    """source: defect 21, accents and CJK tokenize while term ids and logarithmic tf stay stable."""
    import math
    from corpussync.sparse import keyword_tokens, sparse_vector, split_tokens, term_id

    assert split_tokens("CAFÉ déjà 中文 字 _ A1") == ["café", "déjà", "中文", "字", "a1"]
    assert keyword_tokens("CAFÉ déjà 中文 字 the") == ["café", "déjà", "中文"]
    vector = sparse_vector("café café 中文")
    assert vector.indices == sorted({term_id("café"), term_id("中文")})
    assert dict(zip(vector.indices, vector.values))[term_id("café")] == 1 + math.log(2)


def test_collection_alias_collision_searches_each_once(ctx, tmp_path, monkeypatch, capsys):
    """source: defect 10, exact collection names distinguish notes from notes-corpus under --all."""
    from corpussync.cli import cmd_list, main

    for collection in ("notes", "notes-corpus"):
        ctx.collection_override = collection
        ingest_document(ctx, text="five words make this note", source_file=str(tmp_path / collection),
                        corpus="notes", title=collection, locator=collection)
    ctx.collection_override = None
    cmd_list(ctx)
    assert capsys.readouterr().out.count("notes  1 points  hybrid  (") == 2
    queried = []
    original = ctx.client.query_points

    def query(**kwargs):
        queried.append(kwargs["collection_name"])
        return original(**kwargs)

    monkeypatch.setattr(ctx.client, "query_points", query)
    monkeypatch.setattr("corpussync.cli.load_context", lambda *a: ctx)
    with monkeypatch.context() as patch:
        patch.setattr(ctx, "close", lambda: None)
        assert main(["search", "five words", "--all"]) == 0
    assert queried == ["notes", "notes-corpus"]
    preferred = search(ctx, "five words", ["notes"])
    assert [hit.title for hit in preferred.results] == ["notes-corpus"]
    assert "notes: using collection notes-corpus; the collection notes is also present (select it with --collection notes)" in capsys.readouterr().err
    both = search(ctx, "five words", ["notes", "notes"], collections=["notes", "notes-corpus"])
    assert len(both.results) == 2
    assert search(ctx, "five words", [], collections=["missing"]).missing == ["missing"]


def test_min_score_override_and_display_relevance(ctx, tmp_path, capsys):
    """source: defects 14 and 24, dense overrides gate by cosine and text shows cosine instead of RRF."""
    from corpussync.search import print_hits

    text = "tomato storage"
    ctx.embedder = _ControlledEmbedder(ctx.settings.embed_dim, {text: 0.61})
    ingest_document(ctx, text=text, source_file=str(tmp_path / "note"), corpus="notes", title="Note", locator="note")
    assert search(ctx, "unrelated query", ["notes"]).results == []
    result = search(ctx, "unrelated query", ["notes"], min_score=0.60)
    assert len(result.results) == 1
    print_hits(result)
    assert "[1] 0.6100  notes | Note | note" in capsys.readouterr().out
    assert search(ctx, "tomato storage", ["notes"], mode="dense").results == []


def test_failed_embedding_is_attempted_once_per_style(ctx, tmp_path, monkeypatch):
    """source: defects 7 and 25, a failed query embedding is cached rather than retried per collection."""
    for corpus in ("first", "second"):
        ingest_document(ctx, text="five words make this note", source_file=str(tmp_path / corpus),
                        corpus=corpus, title=corpus, locator=corpus)
    calls = []

    def fail(text, prefixed=True):
        calls.append(prefixed)
        raise ValueError("embedder unavailable")

    monkeypatch.setattr(ctx.embedder, "embed_query", fail)
    result = search(ctx, "five words", ["first", "second"])
    assert result.results == []
    assert len(result.notices) == 2
    assert calls == [True]


@pytest.mark.parametrize("command,mode", [("search", "hybrid"), ("search", "dense"), ("ask", "hybrid")])
@pytest.mark.parametrize("failure", ["embedder", "query"])
def test_all_search_failures_exit_one(ctx, tmp_path, monkeypatch, capsys, command, mode, failure):
    """source: round 3 item 3, failed embedding or queries are could-not outcomes and never call chat."""
    from corpussync.cli import _parser, cmd_ask, cmd_search

    for collection in ("notes", "notes-corpus"):
        ctx.collection_override = collection
        ingest_document(ctx, text="five words make this note", source_file=str(tmp_path / collection),
                        corpus="notes", title=collection, locator=collection)
    ctx.collection_override = None
    error = "service unavailable " + "x" * 160

    def fail(*args, **kwargs):
        if kwargs.get("collection_name") == "notes-corpus":
            raise ValueError("second error")
        raise ValueError(error)

    if failure == "embedder":
        monkeypatch.setattr(ctx.embedder, "embed_query", fail)
    else:
        monkeypatch.setattr(ctx.client, "query_points", fail)
    result = search(ctx, "five words", [], mode=mode, collections=["notes", "notes-corpus"])
    assert result.results == []
    assert result.failed == ["notes", "notes-corpus"]
    args = [command, "five words", "--collection", "notes", "--collection", "notes-corpus"]
    if command == "search":
        args.extend(["--mode", mode])
    ctx.chat = _BoomChat()
    capsys.readouterr()
    handler = cmd_search if command == "search" else cmd_ask
    assert handler(ctx, _parser().parse_args(args)) == 1
    output = capsys.readouterr()
    assert output.err.splitlines()[-1] == "could not search: " + error[:120]
    assert "no good match" not in output.out + output.err
    assert ctx.chat.calls == 0


@pytest.mark.parametrize("command", ["search", "ask"])
def test_successful_empty_collection_preserves_no_match(ctx, tmp_path, monkeypatch, capsys, command):
    """source: round 3 item 3, one successful empty query keeps exit 2 when another collection fails."""
    from types import SimpleNamespace
    from corpussync.cli import _parser, cmd_ask, cmd_search

    for corpus in ("bad", "empty"):
        ingest_document(ctx, text="five words make this note", source_file=str(tmp_path / corpus),
                        corpus=corpus, title=corpus, locator=corpus)

    def query(**kwargs):
        if kwargs["collection_name"] == "bad-corpus":
            raise ValueError("query failed")
        return SimpleNamespace(points=[])

    monkeypatch.setattr(ctx.client, "query_points", query)
    result = search(ctx, "five words", ["bad", "empty"])
    assert result.results == []
    assert result.failed == ["bad-corpus"]
    ctx.chat = _BoomChat()
    capsys.readouterr()
    handler = cmd_search if command == "search" else cmd_ask
    args = _parser().parse_args([command, "five words", "--corpus", "bad", "--corpus", "empty"])
    assert handler(ctx, args) == 2
    output = capsys.readouterr()
    assert output.out == "no good match in: bad, empty\n"
    assert "could not search" not in output.err
    assert ctx.chat.calls == 0


@pytest.mark.parametrize("foreign", [False, True])
def test_incompatible_collections_are_not_failed(ctx, foreign):
    """source: round 3 item 3, foreign layouts and keyword-incompatible collections are ordinary skips."""
    size = 3 if foreign else ctx.settings.embed_dim
    ctx.client.create_collection("old-corpus", vectors_config=VectorParams(size=size, distance=Distance.COSINE))
    result = search(ctx, "five words", ["old"], mode="keyword")
    assert result.results == []
    assert result.failed == []
    assert len(result.notices) == 1


def test_foreign_name_from_all_is_not_treated_as_cli_input(ctx, monkeypatch, capsys):
    """source: defect 25, --all can skip a foreign server collection with a name outside our CLI grammar."""
    from types import SimpleNamespace
    from corpussync.cli import main

    class Store:
        def get_collections(self):
            return SimpleNamespace(collections=[SimpleNamespace(name="foreign..data")])

        def get_collection(self, name):
            return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors={}, sparse_vectors={})))

    ctx.client = Store()
    monkeypatch.setattr("corpussync.cli.load_context", lambda *args: ctx)
    assert main(["search", "question", "--all"]) == 2
    assert "foreign..data: skipped (not a corpussync corpus)" in capsys.readouterr().err
