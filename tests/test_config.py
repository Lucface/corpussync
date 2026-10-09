"""source: defects 18 and 20, normalize endpoints and trust only selected config files."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from corpussync.config import is_local_host, load_settings, ollama_base_url, privacy_notices
from tests.conftest import run_cli


def test_untrusted_cwd_is_ignored(home, tmp_path):
    """source: defect 20, cwd config cannot redirect the embedder or store."""
    (tmp_path / "corpussync.toml").write_text('[ollama]\nhost="untrusted.example"\n[qdrant]\nhost="untrusted.example"\n')
    settings = load_settings()
    assert settings.ollama_host == "127.0.0.1"
    assert not settings.using_server()
    (home / "corpussync.toml").write_text('[ollama]\nhost="trusted.example"\n[search.min_score]\nhybrid=123\n')
    assert load_settings().ollama_host == "trusted.example"
    assert load_settings().min_score == {"dense": 0.65, "keyword": 1.0}
    assert load_settings(str(tmp_path / "corpussync.toml")).ollama_host == "untrusted.example"


def test_init_explicit_directory_reminds_about_config(home, tmp_path):
    """source: defect 20, init outside the trusted home explains explicit loading."""
    target = tmp_path / "custom"
    proc = run_cli(["init", "--path", str(target)], home, tmp_path)
    assert proc.returncode == 0
    assert "outside CORPUSSYNC_HOME is read only through --config" in proc.stdout
    assert (target / "corpussync.toml").is_file()


def test_init_preserves_existing_custom_home(home, capsys):
    """source: round 3 item 1, init must preserve the permissions of an existing custom home too."""
    from corpussync.cli import cmd_init

    home.chmod(0o755)
    assert cmd_init(None) == 0
    assert home.stat().st_mode & 0o777 == 0o755
    assert (home / "corpussync.toml").is_file()


@pytest.mark.parametrize("host,expected", [
    ("localhost", "http://localhost:1234"),
    ("localhost:5678", "http://localhost:5678"),
    ("https://example.test:5678/ollama/", "https://example.test:5678/ollama"),
    ("[::1]:5678", "http://[::1]:5678"),
    ("[::1]", "http://[::1]:1234"),
])
def test_ollama_base_url(host, expected):
    """source: defect 18, host ports, base URL paths and bracketed IPv6 survive normalization."""
    assert ollama_base_url(SimpleNamespace(ollama_host=host, ollama_port=1234)) == expected


@pytest.mark.parametrize("host,local", [
    ("localhost", True), ("http://localhost:11434/path", True),
    ("127.9.8.7", True), ("http://[::1]:11434", True), ("::1", True),
    ("/tmp/store", True), ("./store", True), ("https://remote.example", False),
    ("192.168.1.2", False), ("localhost.example", False),
])
def test_local_host_classification(host, local):
    """source: defect 6, only loopback endpoints and filesystem paths count as local."""
    assert is_local_host(host) is local


@pytest.mark.parametrize("host", [
    "0.0.0.0", "0.0.0.0:11434", "http://0.0.0.0", "http://0.0.0.0:11434",
    "::", "[::]", "[::]:11434", "http://[::]", "http://[::]:11434",
])
def test_unspecified_hosts_are_local(host):
    """source: round 3 item 2, unspecified IPv4 and IPv6 addresses refer to this machine."""
    assert is_local_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0:11434", "http://[::]:11434"])
def test_unspecified_ollama_host_has_no_remote_notice(home, monkeypatch, capsys, host):
    """source: round 3 item 2, a local Ollama bind address must not warn that text leaves the machine."""
    monkeypatch.setenv("OLLAMA_HOST", host)
    privacy_notices(load_settings())
    assert capsys.readouterr().err == ""


def test_remote_privacy_notices(home, capsys):
    """source: defect 6, remote services are named in one stderr line per destination."""
    settings = load_settings()
    settings.ollama_host = "https://ollama.example/base"
    settings.qdrant_url = "https://qdrant.example"
    privacy_notices(settings)
    assert capsys.readouterr().err.splitlines() == [
        "note: document text goes to https://ollama.example/base (Ollama)",
        "note: document text goes to https://qdrant.example (Qdrant)",
    ]


def test_embed_prefix_flags_and_chat_timeout(monkeypatch):
    """source: defects 17, 18 and 23, prefix style is optional and chat waits 300 seconds."""
    from corpussync.embed import OllamaEmbedder
    from corpussync.answer import OllamaChat

    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "embeddings": [[1.0]], "message": {"content": "answer"},
        })

    monkeypatch.setattr("requests.post", post)
    embedder = OllamaEmbedder("https://ollama.example/base/", 11434, "nomic-embed-text")
    embedder.embed_documents(["note"])
    embedder.embed_query("question")
    embedder.embed_documents(["note"], prefixed=False)
    embedder.embed_query("question", prefixed=False)
    assert [call[1]["json"]["input"] for call in calls] == [
        ["search_document: note"], ["search_query: question"], ["note"], ["question"],
    ]
    chat = OllamaChat("https://ollama.example/base/", 11434)
    assert chat.complete("model", []) == "answer"
    assert calls[-1][0] == "https://ollama.example/base/api/chat"
    assert calls[-1][1]["timeout"] == 300
    assert calls[0][0] == "https://ollama.example/base/api/embed"
