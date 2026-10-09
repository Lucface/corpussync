"""Qdrant access. Embedded local mode unless a server URL or host is configured."""

from __future__ import annotations

import time
import uuid
import os
import sys
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    SparseVectorParams,
    VectorParams,
)

from corpussync.config import Settings, private_directory, private_home
from corpussync.names import check_name
from corpussync.state import legacy_state_path


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {msg}", flush=True)


def retry(fn, what: str, tries: int = 5, delay: float = 2.0):
    """Retry a Qdrant call. A remote server can refuse connections under load."""
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if attempt < tries - 1:
                log(f"  retry {what} ({attempt + 1}/{tries}): {str(exc)[:80]}")
                time.sleep(delay * (attempt + 1))
    raise last


def make_client(settings: Settings) -> QdrantClient:
    private_home(settings.home, warn=True)
    path = settings.home / "qdrant"
    private_directory(path, tighten=True)
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, timeout=30)
    if settings.qdrant_host:
        return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=30)
    try:
        return QdrantClient(path=str(path))
    except Exception as exc:
        if "already accessed" not in str(exc):
            raise
        print(
            f"the store at {path} is open in another corpussync process; wait for it to finish, "
            "or run a Qdrant server and set QDRANT_URL", file=sys.stderr,
        )
        raise SystemExit(2) from None


def make_point_id(source_file: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_file}::{index}"))


def collection_names(client: QdrantClient) -> set[str]:
    listed = retry(lambda: client.get_collections(), "get_collections")
    return {item.name for item in listed.collections}


def layout_of(client: QdrantClient, name: str, dim: int | None = None) -> str:
    """'hybrid' (named dense + bm25), 'unnamed' (0.1 single vector), or 'dense-named'."""
    info = client.get_collection(name)
    vectors = info.config.params.vectors
    sparse = info.config.params.sparse_vectors or {}
    if isinstance(vectors, VectorParams):
        return "unnamed" if dim is None or vectors.size == dim else "foreign"
    dense = vectors.get("dense") if isinstance(vectors, dict) else None
    if dense is None or (dim is not None and dense.size != dim):
        return "foreign"
    if "bm25" in sparse:
        return "hybrid"
    return "dense-named"


def layout_label(client: QdrantClient, name: str, dim: int | None = None) -> str:
    kind = layout_of(client, name, dim)
    if kind in ("hybrid", "foreign"):
        return kind
    return "dense"


def ensure_collection(client: QdrantClient, name: str, dim: int, existing=None) -> tuple[str, bool]:
    if existing is None:
        existing = collection_names(client)
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
        )
        log(f"created collection {name} ({dim}d cosine, sparse bm25)")
        existing.add(name)
        return "hybrid", True
    return layout_of(client, name, dim), False


def point_count(client: QdrantClient, name: str) -> int:
    counted = client.count(collection_name=name, exact=True)
    return int(counted.count)


def list_corpora(client: QdrantClient) -> list[str]:
    names = []
    for name in sorted(collection_names(client)):
        if name.endswith("-corpus"):
            names.append(name[: -len("-corpus")])
        else:
            names.append(name)
    return names


def resolve_collection(ctx, corpus: str) -> str | None:
    check_name(corpus, "corpus")
    existing = ctx.collections
    preferred = f"{corpus}-corpus"
    if preferred in existing:
        if corpus in existing and corpus not in ctx._selection_notices:
            print(
                f"{corpus}: using collection {preferred}; the collection {corpus} is also present "
                f"(select it with --collection {corpus})", file=sys.stderr,
            )
            ctx._selection_notices.add(corpus)
        return preferred
    if corpus in existing:
        return corpus
    return None


def legacy_address() -> str:
    return os.environ.get("CORPUSSYNC_LEGACY_QDRANT", "http://127.0.0.1:6333").rstrip("/")


def legacy_server_answers(probe) -> bool:
    try:
        return probe(f"{legacy_address()}/collections", timeout=1).status_code == 200
    except Exception:
        return False


def legacy_store_choice(settings, probe) -> str:
    if settings.using_server():
        return "server"
    if os.environ.get("CORPUSSYNC_STORE") == "embedded":
        return "embedded"
    path = legacy_state_path(settings)
    if path is None:
        return "embedded"
    legacy = legacy_address()
    if legacy_server_answers(probe):
        settings.qdrant_url = legacy
        print(
            f"using the Qdrant server at {legacy}, where 0.1 kept its corpora (0.1 state: {path}); "
            "set QDRANT_URL to use it from corpussync too", file=sys.stderr,
        )
        return "server"
    print(
        f"0.1 state is at {path} but no Qdrant server answers at {legacy}. "
        "Start it, set QDRANT_URL, or set CORPUSSYNC_STORE=embedded to begin a new embedded corpus.",
        file=sys.stderr,
    )
    return "stop"
