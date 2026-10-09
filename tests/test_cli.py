"""source: the offline CLI covers init, ingest, search, ask, stats, list, and remove."""

import json
import sys

from corpussync.config import load_settings, loads_toml_minimal
from tests.conftest import REPO, run_cli


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
    """source: init, ingest, search --json, ask, stats, list, and remove --yes run offline."""
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
    toml_path = work / "corpussync.toml"
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
    """source: environment overrides corpussync.toml, which overrides defaults."""
    wanted = tmp_path / "from-toml"
    cfg = tmp_path / "corpussync.toml"
    cfg.write_text(
        'home = "' + str(wanted) + '"\n[embed]\nmodel = "from-toml"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CORPUSSYNC_HOME", raising=False)
    monkeypatch.setenv("CORPUSSYNC_EMBED_MODEL", "from-env")
    settings = load_settings()
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
