"""Settings. Precedence: command-line flags, environment, corpussync.toml, defaults."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_HOME = "~/.corpussync"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_OLLAMA_HOST = "127.0.0.1"
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_EMBED_DIM = 768
DEFAULT_EMBEDDER = "ollama"
DEFAULT_CHAT = "ollama"
DEFAULT_ANSWER_MODEL = "llama3.2"
DEFAULT_MIN_SCORE = {"dense": 0.2, "keyword": 0.05, "hybrid": 0.0}
DEFAULT_DATA_DIR = "~/.corpussync/channels"
DEFAULT_CHANNELS_FILE = "channels.txt"
DEFAULT_PER_RUN_CAP = 250
DEFAULT_PYBIN = "python3"


def config_template() -> str:
    scores = DEFAULT_MIN_SCORE
    return f"""# corpussync.toml
# Uncomment a line to override its default.
# Precedence: command-line flags, then environment variables, then this file, then defaults.
# Root keys stay above the tables. Uncomment a key where it sits.
# On Python 3.10 this file is read by a minimal parser (comments, tables, strings, numbers, booleans).
# Python 3.11 and newer use tomllib.

# home = "{DEFAULT_HOME}"
# answer_model = "{DEFAULT_ANSWER_MODEL}"
# data_dir = "{DEFAULT_DATA_DIR}"
# channels_file = "{DEFAULT_CHANNELS_FILE}"
# per_run_cap = {DEFAULT_PER_RUN_CAP}
# cookie_jar = ""
# pybin = "{DEFAULT_PYBIN}"

[qdrant]
# url = ""
# host = ""
# port = {DEFAULT_QDRANT_PORT}

[ollama]
# host = "{DEFAULT_OLLAMA_HOST}"
# port = {DEFAULT_OLLAMA_PORT}

[embed]
# model = "{DEFAULT_EMBED_MODEL}"
# dim = {DEFAULT_EMBED_DIM}
# backend = "{DEFAULT_EMBEDDER}"

[chat]
# backend = "{DEFAULT_CHAT}"

[state]
# path = ""

[search.min_score]
# dense = {scores["dense"]}
# keyword = {scores["keyword"]}
# hybrid = {scores["hybrid"]}
"""


def loads_toml_minimal(text: str) -> dict:
    """Small TOML reader for the keys this project writes. Used on Python 3.10."""
    root: dict = {}
    current = root

    def strip_comment(line: str) -> str:
        out: list[str] = []
        quote = None
        for ch in line:
            if quote:
                out.append(ch)
                if ch == quote:
                    quote = None
                continue
            if ch in ('"', "'"):
                quote = ch
                out.append(ch)
                continue
            if ch == "#":
                break
            out.append(ch)
        return "".join(out).strip()

    def parse_value(raw: str):
        if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
            return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        if len(raw) >= 2 and raw[0] == "'" and raw[-1] == "'":
            return raw[1:-1]
        if raw == "true":
            return True
        if raw == "false":
            return False
        if _is_int(raw):
            return int(raw)
        if _is_float(raw):
            return float(raw)
        raise ValueError(f"unsupported toml value: {raw}")

    for raw_line in text.splitlines():
        line = strip_comment(raw_line)
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            node = root
            for part in section.split("."):
                if part not in node or not isinstance(node[part], dict):
                    node[part] = {}
                node = node[part]
            current = node
            continue
        if "=" not in line:
            raise ValueError(f"unsupported toml line: {line}")
        key, val = line.split("=", 1)
        current[key.strip()] = parse_value(val.strip())
    return root


def _is_int(raw: str) -> bool:
    if not raw:
        return False
    body = raw[1:] if raw[0] in "+-" else raw
    return bool(body) and body.isdigit()


def _is_float(raw: str) -> bool:
    if raw.count(".") != 1:
        return False
    left, right = raw.split(".")
    return _is_int(left) and bool(right) and right.isdigit()


def load_toml_text(text: str) -> dict:
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib.loads(text)
    return loads_toml_minimal(text)


def load_toml_file(path: Path) -> dict:
    return load_toml_text(path.read_text(encoding="utf-8"))


def _present(value) -> bool:
    return value not in (None, "")


def _pick(key: str, *maps: dict, default=None):
    for item in maps:
        if key in item and _present(item[key]):
            return item[key]
    return default


def read_env() -> dict:
    out: dict = {}
    env = os.environ
    if env.get("CORPUSSYNC_HOME"):
        out["home"] = env["CORPUSSYNC_HOME"]
    if env.get("QDRANT_URL"):
        out["qdrant_url"] = env["QDRANT_URL"]
    if env.get("QDRANT_HOST"):
        out["qdrant_host"] = env["QDRANT_HOST"]
    if env.get("QDRANT_PORT"):
        out["qdrant_port"] = int(env["QDRANT_PORT"])
    if env.get("OLLAMA_HOST"):
        out["ollama_host"] = env["OLLAMA_HOST"]
    if env.get("OLLAMA_PORT"):
        out["ollama_port"] = int(env["OLLAMA_PORT"])
    if env.get("CORPUSSYNC_EMBED_MODEL"):
        out["embed_model"] = env["CORPUSSYNC_EMBED_MODEL"]
    if env.get("CORPUSSYNC_EMBED_DIM"):
        out["embed_dim"] = int(env["CORPUSSYNC_EMBED_DIM"])
    if env.get("CORPUSSYNC_EMBEDDER"):
        out["embedder"] = env["CORPUSSYNC_EMBEDDER"]
    if env.get("CORPUSSYNC_CHAT"):
        out["chat"] = env["CORPUSSYNC_CHAT"]
    if env.get("CORPUSSYNC_STATE"):
        out["state_path"] = env["CORPUSSYNC_STATE"]
    if env.get("CORPUSSYNC_DATA"):
        out["data_dir"] = env["CORPUSSYNC_DATA"]
    if env.get("CHANNELS_FILE"):
        out["channels_file"] = env["CHANNELS_FILE"]
    if env.get("PER_RUN_CAP"):
        out["per_run_cap"] = int(env["PER_RUN_CAP"])
    if env.get("PYBIN"):
        out["pybin"] = env["PYBIN"]
    if env.get("COOKIE_JAR"):
        out["cookie_jar"] = env["COOKIE_JAR"]
    return out


def toml_to_map(data: dict) -> dict:
    out: dict = {}
    if not isinstance(data, dict):
        return out
    if _present(data.get("home")):
        out["home"] = data["home"]
    if _present(data.get("answer_model")):
        out["answer_model"] = data["answer_model"]
    if _present(data.get("data_dir")):
        out["data_dir"] = data["data_dir"]
    if _present(data.get("channels_file")):
        out["channels_file"] = data["channels_file"]
    if _present(data.get("per_run_cap")):
        out["per_run_cap"] = int(data["per_run_cap"])
    if _present(data.get("cookie_jar")):
        out["cookie_jar"] = data["cookie_jar"]
    if _present(data.get("pybin")):
        out["pybin"] = data["pybin"]

    qdrant = data.get("qdrant") if isinstance(data.get("qdrant"), dict) else {}
    if _present(qdrant.get("url")):
        out["qdrant_url"] = qdrant["url"]
    if _present(qdrant.get("host")):
        out["qdrant_host"] = qdrant["host"]
    if "port" in qdrant and _present(qdrant.get("port")):
        out["qdrant_port"] = int(qdrant["port"])

    ollama = data.get("ollama") if isinstance(data.get("ollama"), dict) else {}
    if _present(ollama.get("host")):
        out["ollama_host"] = ollama["host"]
    if "port" in ollama and _present(ollama.get("port")):
        out["ollama_port"] = int(ollama["port"])

    embed = data.get("embed") if isinstance(data.get("embed"), dict) else {}
    if _present(embed.get("model")):
        out["embed_model"] = embed["model"]
    if "dim" in embed and _present(embed.get("dim")):
        out["embed_dim"] = int(embed["dim"])
    if _present(embed.get("backend")):
        out["embedder"] = embed["backend"]

    chat = data.get("chat") if isinstance(data.get("chat"), dict) else {}
    if _present(chat.get("backend")):
        out["chat"] = chat["backend"]

    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    if _present(state.get("path")):
        out["state_path"] = state["path"]

    search = data.get("search") if isinstance(data.get("search"), dict) else {}
    scores = search.get("min_score") if isinstance(search.get("min_score"), dict) else {}
    if scores:
        out["min_score"] = scores
    return out


def find_config(explicit: str | None, home: Path) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SystemExit(f"config not found: {path}")
        return path
    cwd = Path.cwd() / "corpussync.toml"
    if cwd.is_file():
        return cwd
    home_cfg = home / "corpussync.toml"
    if home_cfg.is_file():
        return home_cfg
    return None


def _resolve_qdrant(env_map: dict, toml_map: dict) -> tuple[str | None, str | None, int]:
    """Highest source that sets a URL or host wins. Port follows that source when set."""
    for source in (env_map, toml_map):
        if _present(source.get("qdrant_url")):
            return str(source["qdrant_url"]), None, int(
                _pick("qdrant_port", source, env_map, toml_map, default=DEFAULT_QDRANT_PORT)
            )
        if _present(source.get("qdrant_host")):
            port = source.get("qdrant_port")
            if not _present(port):
                port = _pick("qdrant_port", env_map, toml_map, default=DEFAULT_QDRANT_PORT)
            return None, str(source["qdrant_host"]), int(port)
    return None, None, DEFAULT_QDRANT_PORT


@dataclass
class Settings:
    home: Path
    qdrant_url: str | None
    qdrant_host: str | None
    qdrant_port: int
    ollama_host: str
    ollama_port: int
    embed_model: str
    embed_dim: int
    embedder: str
    chat: str
    answer_model: str
    state_path: Path
    min_score: dict = field(default_factory=lambda: dict(DEFAULT_MIN_SCORE))
    data_dir: Path = field(default_factory=lambda: Path(DEFAULT_DATA_DIR).expanduser())
    channels_file: str = DEFAULT_CHANNELS_FILE
    per_run_cap: int = DEFAULT_PER_RUN_CAP
    cookie_jar: str = ""
    pybin: str = DEFAULT_PYBIN

    def using_server(self) -> bool:
        return bool(self.qdrant_url or self.qdrant_host)


def load_settings(config_path: str | None = None) -> Settings:
    env_map = read_env()
    home_for_lookup = Path(str(env_map.get("home") or DEFAULT_HOME)).expanduser()
    found = find_config(config_path, home_for_lookup)
    toml_map = toml_to_map(load_toml_file(found)) if found else {}

    home = Path(str(_pick("home", env_map, toml_map, default=DEFAULT_HOME))).expanduser()
    url, host, port = _resolve_qdrant(env_map, toml_map)
    state_raw = _pick("state_path", env_map, toml_map)
    if _present(state_raw):
        state_path = Path(str(state_raw)).expanduser()
    else:
        state_path = home / "state.db"
    data_raw = _pick("data_dir", env_map, toml_map, default=DEFAULT_DATA_DIR)
    data_dir = Path(str(data_raw)).expanduser()
    scores = dict(DEFAULT_MIN_SCORE)
    for key, value in (toml_map.get("min_score") or {}).items():
        if key in scores and _present(value):
            scores[key] = float(value)
    dim = int(_pick("embed_dim", env_map, toml_map, default=DEFAULT_EMBED_DIM))
    if dim < 1:
        raise SystemExit(f"embed dim must be positive, got {dim}")
    return Settings(
        home=home,
        qdrant_url=url,
        qdrant_host=host,
        qdrant_port=port,
        ollama_host=str(_pick("ollama_host", env_map, toml_map, default=DEFAULT_OLLAMA_HOST)),
        ollama_port=int(_pick("ollama_port", env_map, toml_map, default=DEFAULT_OLLAMA_PORT)),
        embed_model=str(_pick("embed_model", env_map, toml_map, default=DEFAULT_EMBED_MODEL)),
        embed_dim=dim,
        embedder=str(_pick("embedder", env_map, toml_map, default=DEFAULT_EMBEDDER)).lower(),
        chat=str(_pick("chat", env_map, toml_map, default=DEFAULT_CHAT)).lower(),
        answer_model=str(_pick("answer_model", env_map, toml_map, default=DEFAULT_ANSWER_MODEL)),
        state_path=state_path,
        min_score=scores,
        data_dir=data_dir,
        channels_file=str(_pick("channels_file", env_map, toml_map, default=DEFAULT_CHANNELS_FILE)),
        per_run_cap=int(_pick("per_run_cap", env_map, toml_map, default=DEFAULT_PER_RUN_CAP)),
        cookie_jar=str(_pick("cookie_jar", env_map, toml_map, default="") or ""),
        pybin=str(_pick("pybin", env_map, toml_map, default=DEFAULT_PYBIN)),
    )


class Context:
    """Runtime bundle: settings, embedder, chat, and lazy store handles."""

    def __init__(self, settings: Settings, embedder, chat):
        self.settings = settings
        self.embedder = embedder
        self.chat = chat
        self.collection_override: str | None = None
        self._client = None
        self._db = None

    def collection_name(self, corpus: str) -> str:
        if self.collection_override:
            return self.collection_override
        return f"{corpus}-corpus"

    @property
    def client(self):
        if self._client is None:
            from corpussync.store import make_client

            self._client = make_client(self.settings)
        return self._client

    @client.setter
    def client(self, value) -> None:
        self._client = value

    @property
    def db(self):
        if self._db is None:
            from corpussync.state import connect

            self._db = connect(self.settings.state_path)
        return self._db

    def close(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if close:
                close()
            self._client = None
        if self._db is not None:
            self._db.close()
            self._db = None


def load_context(config_path: str | None = None) -> Context:
    from corpussync.answer import build_chat
    from corpussync.embed import build_embedder

    settings = load_settings(config_path)
    return Context(settings, build_embedder(settings), build_chat(settings))
