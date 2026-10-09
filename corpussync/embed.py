"""Dense embedders. Ollama batches at most 16 texts. Fake is deterministic and offline."""

from __future__ import annotations

import math
from types import SimpleNamespace

import requests

from corpussync.sparse import split_tokens, term_id
from corpussync.config import ollama_base_url

_BATCH = 16
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


class OllamaEmbedder:
    def __init__(self, host: str, port: int, model: str):
        self.model = model
        self.base = ollama_base_url(SimpleNamespace(ollama_host=host, ollama_port=port))
        self._nomic = "nomic-embed" in model.lower()

    def embed_documents(self, texts: list[str], prefixed: bool = True) -> list[list[float]]:
        return self._embed([self._document(text) if prefixed else text for text in texts])

    def embed_query(self, text: str, prefixed: bool = True) -> list[float]:
        return self._embed([self._query(text) if prefixed else text])[0]

    def _document(self, text: str) -> str:
        if self._nomic:
            return _DOC_PREFIX + text
        return text

    def _query(self, text: str) -> str:
        if self._nomic:
            return _QUERY_PREFIX + text
        return text

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            batch = texts[start:start + _BATCH]
            resp = requests.post(
                f"{self.base}/api/embed",
                json={"model": self.model, "input": batch},
                timeout=120,
            )
            resp.raise_for_status()
            embeddings = resp.json().get("embeddings")
            got = len(embeddings) if isinstance(embeddings, list) else None
            if got != len(batch):
                raise RuntimeError(
                    f"embed count mismatch: {got} vectors != {len(batch)} texts"
                )
            out.extend(embeddings)
        return out


class FakeEmbedder:
    """Unit vectors from hashed tokens. Texts that share words score closer.

    Stopwords stay in the vector so keyword search (which drops them) can rank
    a different document first.
    """

    def __init__(self, dim: int):
        self.dim = dim

    def embed_documents(self, texts: list[str], prefixed: bool = True) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def embed_query(self, text: str, prefixed: bool = True) -> list[float]:
        return self._one(text)

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = split_tokens(text)
        if not tokens:
            vec[0] = 1.0
            return vec
        for tok in tokens:
            vec[term_id(tok) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def build_embedder(settings):
    kind = settings.embedder
    if kind == "fake":
        return FakeEmbedder(settings.embed_dim)
    if kind == "ollama":
        return OllamaEmbedder(settings.ollama_host, settings.ollama_port, settings.embed_model)
    raise SystemExit(f"unknown embedder: {kind}")
