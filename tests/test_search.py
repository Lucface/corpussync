"""source: hybrid ranking, multi-corpus coverage, the relevance floor, and 0.1 collections."""

import subprocess
import sys

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
    """source: hybrid ranks an exact keyword match first where dense alone does not."""
    ctx.settings.min_score = {"dense": 0.0, "keyword": 0.0, "hybrid": 0.0}
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
