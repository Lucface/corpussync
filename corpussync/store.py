"""Qdrant access. Embedded local mode unless a server URL or host is configured."""

import time
import uuid
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    SparseVectorParams,
    VectorParams,
)

from corpussync.config import Settings


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
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, timeout=30)
    if settings.qdrant_host:
        return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=30)
    path = settings.home / "qdrant"
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def make_point_id(source_file: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_file}::{index}"))


def collection_names(client: QdrantClient) -> set[str]:
    listed = retry(lambda: client.get_collections(), "get_collections")
    return {item.name for item in listed.collections}


def layout_of(client: QdrantClient, name: str) -> str:
    """'hybrid' (named dense + bm25), 'unnamed' (0.1 single vector), or 'dense-named'."""
    info = retry(lambda: client.get_collection(name), "get_collection")
    vectors = info.config.params.vectors
    sparse = info.config.params.sparse_vectors or {}
    if isinstance(vectors, VectorParams):
        return "unnamed"
    if isinstance(sparse, dict) and "bm25" in sparse:
        return "hybrid"
    return "dense-named"


def layout_label(client: QdrantClient, name: str) -> str:
    kind = layout_of(client, name)
    if kind == "hybrid":
        return "hybrid"
    return "dense"


def ensure_collection(client: QdrantClient, name: str, dim: int) -> str:
    existing = collection_names(client)
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
        )
        log(f"created collection {name} ({dim}d cosine, sparse bm25)")
        return "hybrid"
    return layout_of(client, name)


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


def resolve_collection(client: QdrantClient, corpus: str) -> str | None:
    existing = collection_names(client)
    preferred = f"{corpus}-corpus"
    if preferred in existing:
        return preferred
    if corpus in existing:
        return corpus
    return None
