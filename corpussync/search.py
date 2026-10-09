"""Dense, keyword, and hybrid search with a shared relevance gate and ranking."""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field

from qdrant_client.models import Fusion, FusionQuery, Prefetch

from corpussync.names import check_name
from corpussync.sparse import keyword_tokens, sparse_vector
from corpussync.store import layout_of, resolve_collection

_RRF_K = 60
_MODES = ("hybrid", "dense", "keyword")


@dataclass
class Hit:
    n: int
    corpus: str
    score: float
    title: str
    locator: str
    url: str
    source_file: str
    chunk_index: int
    text: str
    relevance: float | None
    keyword_coverage: float


@dataclass
class SearchResult:
    results: list[Hit] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def no_match_message(corpora: list[str]) -> str:
    return "no good match in:" + (" " + ", ".join(corpora) if corpora else "")


def snippet(text: str, limit: int = 200) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "..."


def print_notices(notices: list[str]) -> None:
    for notice in notices:
        print(notice, file=sys.stderr)


def print_hits(result: SearchResult) -> None:
    for hit in result.results:
        locator = hit.locator or hit.url
        score = hit.relevance if hit.relevance is not None else hit.keyword_coverage
        print(f"[{hit.n}] {score:.4f}  {hit.corpus} | {hit.title} | {locator}")
        text = snippet(hit.text)
        if text:
            print(text)
        print()


def print_coverage(corpora: list[str], coverage: dict) -> None:
    parts = [f"{name}={coverage.get(name, 0)}" for name in corpora]
    print("coverage: " + ", ".join(parts))


def corpus_label(collection: str) -> str:
    return collection[:-len("-corpus")] if collection.endswith("-corpus") else collection


def select_collections(ctx, corpora, collections=None):
    for name in corpora:
        check_name(name, "corpus")
    selected = []
    missing = []
    for corpus in corpora:
        name = resolve_collection(ctx, corpus)
        if name is None:
            missing.append(corpus)
        elif name not in selected:
            selected.append(name)
    for name in collections or []:
        if name not in ctx.collections:
            if name not in missing:
                missing.append(name)
        elif name not in selected:
            selected.append(name)
    return selected, missing


def _cosine(query, stored) -> float:
    if not stored or len(query) != len(stored):
        return 0.0
    norm = math.sqrt(sum(x * x for x in query) * sum(x * x for x in stored))
    return sum(x * y for x, y in zip(query, stored)) / norm if norm else 0.0


def _tie(hit):
    return hit.corpus, hit.source_file, hit.chunk_index


def search(ctx, query: str, corpora: list[str], k: int = 8, mode: str = "hybrid",
           min_score: float | None = None, collections: list[str] | None = None) -> SearchResult:
    if mode not in _MODES:
        raise ValueError(f"unknown search mode: {mode}")
    selected, missing = select_collections(ctx, corpora, collections)
    result = SearchResult(missing=missing)
    result.coverage = {corpus_label(name): 0 for name in selected}
    result.coverage.update({name: 0 for name in missing})
    dense_floor = min_score if min_score is not None else ctx.settings.min_score.get("dense", 0.65)
    keyword_floor = ctx.settings.min_score.get("keyword", 1.0)
    query_tokens = set(keyword_tokens(query))
    sparse = sparse_vector(query)
    vectors = {}
    vector_errors = {}
    candidates = []
    limit = max(4 * k, 20)
    for collection in selected:
        corpus = corpus_label(collection)
        try:
            layout = layout_of(ctx.client, collection, ctx.settings.embed_dim)
            if layout == "foreign":
                result.notices.append(f"{corpus}: skipped (not a corpussync corpus)")
                continue
            if mode == "keyword" and layout != "hybrid":
                result.notices.append(f"{corpus}: keyword mode needs a fresh corpus (remove then ingest again)")
                continue
            prefixed = layout != "unnamed"
            dense = None
            kwargs = dict(collection_name=collection, limit=limit, with_payload=True)
            if mode != "keyword":
                if prefixed in vector_errors:
                    raise vector_errors[prefixed]
                if prefixed not in vectors:
                    try:
                        vectors[prefixed] = ctx.embedder.embed_query(query, prefixed=prefixed)
                    except Exception as exc:
                        vector_errors[prefixed] = exc
                        raise
                dense = vectors[prefixed]
                kwargs["with_vectors"] = ["dense"] if prefixed else True
            if mode == "keyword":
                if not sparse.indices:
                    continue
                kwargs.update(query=sparse, using="bm25")
            elif mode == "hybrid" and layout == "hybrid":
                kwargs.update(
                    prefetch=[
                        Prefetch(query=dense, using="dense", limit=limit),
                        Prefetch(query=sparse, using="bm25", limit=limit),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                )
            else:
                if mode == "hybrid":
                    result.notices.append(f"{corpus}: --mode hybrid needs a fresh corpus (remove then ingest again)")
                kwargs["query"] = dense
                if prefixed:
                    kwargs["using"] = "dense"
            points = ctx.client.query_points(**kwargs).points
            group = []
            for rank, point in enumerate(points, 1):
                payload = point.payload or {}
                text = str(payload.get("text") or "")
                coverage = len(query_tokens & set(keyword_tokens(text))) / len(query_tokens) if query_tokens else 0.0
                relevance = None
                if dense is not None:
                    stored = point.vector
                    if isinstance(stored, dict):
                        stored = stored.get("dense")
                    relevance = _cosine(dense, stored)
                keep = (
                    coverage > 0 if mode == "keyword" else
                    relevance >= dense_floor or (mode == "hybrid" and bool(query_tokens) and coverage >= keyword_floor)
                )
                if not keep:
                    continue
                hit = Hit(
                    n=0, corpus=corpus, score=0.0,
                    title=str(payload.get("title") or ""),
                    locator=str(payload.get("locator") or ""),
                    url=str(payload.get("url") or ""),
                    source_file=str(payload.get("source_file") or ""),
                    chunk_index=int(payload.get("chunk_index") or 0), text=text,
                    relevance=relevance, keyword_coverage=coverage,
                )
                group.append((hit, rank))
            candidates.extend(group)
        except Exception as exc:
            result.notices.append(f"{corpus}: skipped ({str(exc)[:80]})")
    if mode == "keyword":
        ordered = sorted(candidates, key=lambda item: (-item[0].keyword_coverage, item[1], _tie(item[0])))
        hits = [hit for hit, _rank in ordered]
        for hit in hits:
            hit.score = hit.keyword_coverage
    else:
        hits = sorted((hit for hit, _rank in candidates), key=lambda hit: (-hit.relevance, _tie(hit)))
        if mode == "hybrid":
            keyword_hits = sorted(
                (hit for hit in hits if hit.keyword_coverage >= keyword_floor),
                key=lambda hit: (-hit.keyword_coverage, -hit.relevance, _tie(hit)),
            )
            keyword_ranks = {id(hit): rank for rank, hit in enumerate(keyword_hits, 1)}
            for rank, hit in enumerate(hits, 1):
                hit.score = 1.0 / (_RRF_K + rank)
                if id(hit) in keyword_ranks:
                    hit.score += 1.0 / (_RRF_K + keyword_ranks[id(hit)])
            hits.sort(key=lambda hit: (-hit.score, _tie(hit)))
        else:
            for hit in hits:
                hit.score = hit.relevance
    result.results = hits[:k]
    for n, hit in enumerate(result.results, 1):
        hit.n = n
        result.coverage[hit.corpus] += 1
    return result
