"""source: the offline CLI covers init, ingest, search, ask, stats, list, and remove."""

from __future__ import annotations

import json
import sys

import pytest

from corpussync.config import load_settings, loads_toml_minimal
from tests.conftest import REPO, run_cli


@pytest.mark.parametrize("answer_model,pulled,missing", [
    ("llama3.2", [], True),
    ("llama3.2", ["llama3.2"], False),
    ("llama3.2", ["llama3.2:latest"], False),
    ("llama3.2:small", ["llama3.2:small"], False),
    ("llama3.2:small", ["llama3.2:latest"], True),
])
def test_doctor_answer_model_is_warning_only(ctx, monkeypatch, capsys, answer_model, pulled, missing):
    """source: round 3 item 7, doctor checks answer model names and tags without failing on a missing model."""
    from types import SimpleNamespace
    from corpussync.cli import cmd_doctor

    ctx.settings.answer_model = answer_model
    names = [ctx.settings.embed_model + ":latest"] + pulled
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"models": [{"name": name} for name in names]})

    monkeypatch.setattr("corpussync.cli.requests.get", get)
    monkeypatch.setattr("corpussync.cli.shutil.which", lambda name: str(ctx.settings.home / name))
    assert cmd_doctor(ctx) == 0
    output = capsys.readouterr().out
    assert f"embed model {ctx.settings.embed_model}: ok" in output
    warning = f"warning: answer model {answer_model} is not pulled (only ask needs it). Fix: ollama pull {answer_model}"
    if missing:
        assert warning in output
    else:
        assert "warning: answer model" not in output
    assert calls == ["http://127.0.0.1:11434/api/tags"]


def test_help_lists_commands():
    """source: python -m corpussync --help lists every command."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "corpussync", "--help"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    for name in ("init", "ingest", "youtube", "search", "ask", "stats", "list", "remove", "doctor"):
        assert name in proc.stdout


def test_cli_offline_flow(tmp_path):
    """source: defects 20 and 14, init uses the trusted home and JSON exposes relevance."""
    work = tmp_path / "work"
    work.mkdir()
    home = tmp_path / "home"
    notes = work / "notes"
    notes.mkdir()
    body = " ".join(["widget", "calibration", "procedure"] * 4)
    (notes / "widgets.md").write_text("# Widget notes\n\n" + body + "\n", encoding="utf-8")
    (notes / ".secret.md").write_text("hidden-secret-token " * 12, encoding="utf-8")

    created = run_cli(["init"], home, work)
    assert created.returncode == 0, created.stderr
    toml_path = home / "corpussync.toml"
    text = toml_path.read_text(encoding="utf-8")
    assert "# home = " in text
    assert "# answer_model = " in text
    assert "[search.min_score]" in text
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("["):
            continue
        raise AssertionError(stripped)
    toml_path.write_text(text + "# marker\n", encoding="utf-8")
    again = run_cli(["init"], home, work)
    assert again.returncode == 0
    assert "# marker" in toml_path.read_text(encoding="utf-8")

    ingested = run_cli(["ingest", str(notes), "--corpus", "notes"], home, work)
    assert ingested.returncode == 0, ingested.stderr

    found = run_cli(
        ["search", "widget calibration procedure", "--corpus", "notes", "--json"],
        home,
        work,
    )
    assert found.returncode == 0, found.stdout + found.stderr
    payload = json.loads(found.stdout)
    assert payload["query"] == "widget calibration procedure"
    assert payload["mode"] == "hybrid"
    assert payload["results"]
    hit = payload["results"][0]
    assert set(hit) == {
        "n", "corpus", "score", "title", "locator", "url", "source_file", "chunk_index", "text",
        "relevance", "keyword_coverage",
    }
    assert hit["corpus"] == "notes"
    assert hit["title"] == "Widget notes"
    assert hit["locator"] == "widgets.md"
    assert hit["n"] == 1
    assert "hidden-secret-token" not in hit["text"]
    assert payload["coverage"]["notes"] >= 1

    answer = run_cli(["ask", "widget calibration procedure", "--corpus", "notes"], home, work)
    assert answer.returncode == 0, answer.stdout + answer.stderr
    assert "See [1]." in answer.stdout
    assert "Widget notes" in answer.stdout

    stats = run_cli(["stats", "--corpus", "notes"], home, work)
    assert stats.returncode == 0, stats.stderr
    assert "notes-corpus:" in stats.stdout
    assert "points" in stats.stdout

    listed = run_cli(["list"], home, work)
    assert listed.returncode == 0, listed.stderr
    assert "notes" in listed.stdout
    assert "hybrid" in listed.stdout
    assert "points" in listed.stdout

    blocked = run_cli(["remove", "--corpus", "notes"], home, work)
    assert blocked.returncode == 1
    assert "--yes" in blocked.stdout

    removed = run_cli(["remove", "--corpus", "notes", "--yes"], home, work)
    assert removed.returncode == 0, removed.stderr
    after = run_cli(["list"], home, work)
    assert after.returncode == 0
    assert "notes" not in after.stdout


def test_env_overrides_toml(tmp_path, monkeypatch):
    """source: defect 20, explicit config is read with environment taking precedence."""
    wanted = tmp_path / "from-toml"
    cfg = tmp_path / "corpussync.toml"
    cfg.write_text(
        'home = "' + str(wanted) + '"\n[embed]\nmodel = "from-toml"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CORPUSSYNC_HOME", raising=False)
    monkeypatch.setenv("CORPUSSYNC_EMBED_MODEL", "from-env")
    settings = load_settings(str(cfg))
    assert settings.embed_model == "from-env"
    assert settings.home == wanted
    assert settings.answer_model == "llama3.2"


def test_minimal_toml_reader_parses_tables():
    """source: the Python 3.10 fallback reader parses the config tables."""
    sample = '[search.min_score]\ndense = 0.5\nkeyword = 1\nhybrid = 0.0\n\n[qdrant]\nurl = "http://127.0.0.1:6333"\n'
    got = loads_toml_minimal(sample)
    assert got["search"]["min_score"]["dense"] == 0.5
    assert got["search"]["min_score"]["keyword"] == 1
    assert got["search"]["min_score"]["hybrid"] == 0.0
    assert got["qdrant"]["url"] == "http://127.0.0.1:6333"
    if sys.version_info >= (3, 11):
        import tomllib

        assert tomllib.loads(sample)["search"]["min_score"]["dense"] == 0.5


def test_remove_missing_never_deletes(ctx, monkeypatch, capsys):
    """source: defect 1, removing an absent collection returns 1 without calling delete."""
    from corpussync.cli import cmd_remove

    called = []
    monkeypatch.setattr(ctx.client, "delete_collection", lambda **kwargs: called.append(kwargs))
    assert cmd_remove(ctx, None, True, "missing") == 1
    assert capsys.readouterr().out == "missing: not found\n"
    assert called == []


def test_exact_stats_and_remove_scope(ctx, tmp_path, capsys):
    """source: defects 3 and 10, exact selection counts or removes only the chosen collection and store rows."""
    from corpussync.cli import cmd_remove, cmd_stats
    from corpussync.ingest import ingest_document
    from corpussync.state import save_hash, saved_hash, store_identity

    path = str(tmp_path / "note")
    for collection in ("notes", "notes-corpus"):
        ctx.collection_override = collection
        ingest_document(ctx, text="five words make this note", source_file=path,
                        corpus="notes", title=collection, locator="note")
    ctx.collection_override = None
    identity = store_identity(ctx.settings)
    save_hash(ctx.db, "other-store", path, "other-digest", "", "notes", 1)
    capsys.readouterr()
    assert cmd_stats(ctx, None, ["notes", "notes-corpus"]) == 0
    assert capsys.readouterr().out.splitlines() == ["notes: 1 points", "notes-corpus: 1 points"]
    assert cmd_remove(ctx, None, True, "notes") == 0
    assert "notes-corpus" in ctx.collections
    assert saved_hash(ctx.db, identity, "notes", path) is None
    assert saved_hash(ctx.db, identity, "notes-corpus", path) is not None
    assert saved_hash(ctx.db, "other-store", "notes", path) == ("other-digest",)


def test_remote_notices_precede_first_store_call(ctx, monkeypatch, capsys):
    """source: defect 6, CLI emits destination notices before --all contacts a remote store."""
    from types import SimpleNamespace
    from corpussync.cli import main

    class Store:
        def get_collections(self):
            assert capsys.readouterr().err.splitlines() == [
                "note: document text goes to https://ollama.example (Ollama)",
                "note: document text goes to https://qdrant.example (Qdrant)",
            ]
            return SimpleNamespace(collections=[])

    ctx.client = Store()
    ctx.settings.ollama_host = "https://ollama.example"
    ctx.settings.qdrant_url = "https://qdrant.example"
    monkeypatch.setattr("corpussync.cli.load_context", lambda *args: ctx)
    assert main(["search", "question", "--all"]) == 2
