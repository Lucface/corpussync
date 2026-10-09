"""source: defects 18 and 20, normalize endpoints and trust only selected config files."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from corpussync.config import is_local_host, load_settings, ollama_base_url, privacy_notices
from tests.conftest import run_cli


def test_settings_read_toml_numbers_dotted_keys_and_comments(home):
    """source: valid TOML underscores, exponent floats, dotted keys and inline comments must load on every supported Python."""
    config = home / "corpussync.toml"
    config.write_text(
        'search.min_score.dense = 6.5e-1\n'
        'per_run_cap = 1_000 # caption limit\n'
        '[qdrant]\nport = 6_333\n'
    )
    settings = load_settings(str(config))
    assert settings.qdrant_port == 6333
    assert settings.min_score["dense"] == 0.65
    assert settings.per_run_cap == 1000


def test_missing_tomli_on_older_python_has_actionable_error(home, monkeypatch):
    """source: Python 3.9 and 3.10 need an actionable error only when reading a config without tomli."""
    from corpussync import config
    from tests.test_extract import _block_import

    monkeypatch.setattr(config, "sys", SimpleNamespace(version_info=(3, 9)))
    _block_import(monkeypatch, "tomli")
    assert load_settings().qdrant_port == 6333
    path = home / "corpussync.toml"
    path.write_text("per_run_cap = 10\n")
    with pytest.raises(SystemExit) as exc:
        load_settings(str(path))
    assert str(exc.value) == (
        f"corpussync needs tomli on Python 3.9 and 3.10 to read {path}. Run: pip install tomli"
    )


def test_malformed_toml_exits_cleanly(home):
    """source: round 5 item 7, malformed TOML reports the config path without a decode traceback."""
    path = home / "corpussync.toml"
    path.write_text("[qdrant\nport = 6333\n")
    with pytest.raises(SystemExit) as exc:
        load_settings()
    assert str(exc.value).startswith(f"could not read {path}: ")
    proc = run_cli(["stats"], home)
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert proc.stderr.startswith(f"could not read {path}: ")
    assert "Traceback" not in proc.stderr


def test_config_that_is_not_utf8_exits_cleanly(home):
    """source: round 5 gate, a config file that is not UTF-8 raised a decode traceback instead of naming the file."""
    path = home / "corpussync.toml"
    path.write_bytes(b"answer_model = \"caf\xe9\"\n")
    with pytest.raises(SystemExit) as exc:
        load_settings()
    assert str(exc.value).startswith(f"could not read {path}: ")


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


@pytest.mark.parametrize("env_name,toml_key", [
    ("QDRANT_PORT", "qdrant.port"), ("OLLAMA_PORT", "ollama.port"),
    ("CORPUSSYNC_EMBED_DIM", "embed.dim"), ("PER_RUN_CAP", "per_run_cap"),
])
@pytest.mark.parametrize("origin", ["env", "toml"])
@pytest.mark.parametrize("value", ["invalid", "1.5"])
def test_invalid_integer_settings_exit_cleanly(home, monkeypatch, env_name, toml_key, origin, value):
    """source: round 5 item 7, integer settings name their environment or TOML source without truncation or traceback."""
    import subprocess
    import sys
    from tests.conftest import isolated_env

    monkeypatch.delenv("PER_RUN_CAP", raising=False)
    env = isolated_env(home)
    env.pop("PER_RUN_CAP", None)
    if origin == "env":
        monkeypatch.setenv(env_name, value)
        env[env_name] = value
        name = env_name
    else:
        literal = '"invalid"' if value == "invalid" else value
        (home / "corpussync.toml").write_text(f"{toml_key} = {literal}\n")
        name = toml_key
    expected = f"{name} must be a whole number (got: {value})"
    with pytest.raises(SystemExit) as exc:
        load_settings()
    assert str(exc.value) == expected
    proc = subprocess.run([sys.executable, "-m", "corpussync", "stats"],
                          cwd=home, env=env, capture_output=True, text=True)
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert proc.stderr == expected + "\n"


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


@pytest.mark.parametrize("system_words,user_words,expected", [
    (0, 0, 4096), (20, 10, 4096), (1000, 1048, 4096),
    (1000, 1049, 8192), (30, 6 * 390, 8192), (2000, 8000, 16384),
])
def test_chat_context_fits_all_message_words(monkeypatch, system_words, user_words, expected):
    """source: round 5 item 3, chat reserves response space and rounds the complete prompt up to a power of two."""
    from corpussync.answer import OllamaChat

    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"message": {"content": "answer"}})

    monkeypatch.setattr("requests.post", post)
    messages = [
        {"role": "system", "content": "word\t" * system_words},
        {"role": "user", "content": "word\n" * user_words},
    ]
    assert OllamaChat("localhost", 11434).complete("model", messages) == "answer"
    assert len(calls) == 1
    assert calls[0][1]["json"] == {
        "model": "model", "messages": messages, "stream": False,
        "options": {"num_ctx": expected},
    }


@pytest.mark.parametrize("text,expected", [
    (chr(0x6587) * 3000, 8192),
    (chr(0x6587) * 1000, 4096),
    ("x" * 40000, 16384),
])
def test_chat_context_counts_text_without_spaces(text, expected):
    """source: round 5 gate, a word count alone undercounts Chinese, Japanese or code, and Ollama then truncates the prompt."""
    from corpussync.answer import context_size

    assert context_size([{"role": "user", "content": text}]) == expected


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
