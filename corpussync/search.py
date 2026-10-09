"""Dense, keyword, and hybrid search. Several corpora are fused with RRF."""

import re
import sys
from dataclasses import dataclass, field

from qdrant_client.models import Fusion, FusionQuery, Prefetch

from corpussync.sparse import sparse_vector
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


@dataclass
class SearchResult:
    results: list[Hit] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def no_match_message(corpora: list[str]) -> str:
    if not corpora:
        return "no good match in:"
    return "no good match in: " + ", ".join(corpora)


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
        print(f"[{hit.n}] {hit.score:.4f}  {hit.corpus} | {hit.title} | {locator}")
        text = snippet(hit.text)
        if text:
            print(text)
        print()


def print_coverage(corpora: list[str], coverage: dict) -> None:
    parts = [f"{name}={coverage.get(name, 0)}" for name in corpora]
    print("coverage: " + ", ".join(parts))


def _floor(settings, mode: str) -> float:
    try:
        return float(settings.min_score.get(mode, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _threshold(value: float) -> float | None:
    if value > 0:
        return value
    return None


def _hits_from_points(points, corpus: str) -> list[Hit]:
    hits = []
    for point in points:
        payload = point.payload or {}
        hits.append(Hit(
            n=0,
            corpus=corpus,
            score=float(point.score),
            title=str(payload.get("title") or ""),
            locator=str(payload.get("locator") or ""),
            url=str(payload.get("url") or ""),
            source_file=str(payload.get("source_file") or ""),
            chunk_index=int(payload.get("chunk_index") or 0),
            text=str(payload.get("text") or ""),
        ))
    return hits


def _query_dense(client, collection: str, vector: list[float], using: str | None, limit: int, floor: float):
    kwargs = {
        "collection_name": collection,
        "query": vector,
        "limit": limit,
        "with_payload": True,
    }
    if using:
        kwargs["using"] = using
    threshold = _threshold(floor)
    if threshold is not None:
        kwargs["score_threshold"] = threshold
    response = client.query_points(**kwargs)
    return response.points


def _query_keyword(client, collection: str, query: str, limit: int, floor: float):
    vector = sparse_vector(query)
    if not vector.indices:
        return []
    kwargs = {
        "collection_name": collection,
        "query": vector,
        "using": "bm25",
        "limit": limit,
        "with_payload": True,
    }
    threshold = _threshold(floor)
    if threshold is not None:
        kwargs["score_threshold"] = threshold
    response = client.query_points(**kwargs)
    return response.points


def _query_hybrid(client, collection: str, dense: list[float], query: str, limit: int, settings):
    prefetch_limit = max(limit * 4, 16)
    prefetches = []
    dense_kwargs = {"query": dense, "using": "dense", "limit": prefetch_limit}
    dense_floor = _threshold(_floor(settings, "dense"))
    if dense_floor is not None:
        dense_kwargs["score_threshold"] = dense_floor
    prefetches.append(Prefetch(**dense_kwargs))
    sparse = sparse_vector(query)
    if sparse.indices:
        sparse_kwargs = {"query": sparse, "using": "bm25", "limit": prefetch_limit}
        keyword_floor = _threshold(_floor(settings, "keyword"))
        if keyword_floor is not None:
            sparse_kwargs["score_threshold"] = keyword_floor
        prefetches.append(Prefetch(**sparse_kwargs))
    kwargs = {
        "collection_name": collection,
        "prefetch": prefetches,
        "query": FusionQuery(fusion=Fusion.RRF),
        "limit": limit,
        "with_payload": True,
    }
    hybrid_floor = _threshold(_floor(settings, "hybrid"))
    if hybrid_floor is not None:
        kwargs["score_threshold"] = hybrid_floor
    response = client.query_points(**kwargs)
    return response.points


def _search_one(ctx, query: str, corpus: str, k: int, mode: str) -> tuple[list[Hit], str | None, bool]:
    collection = resolve_collection(ctx.client, corpus)
    if collection is None:
        return [], None, False
    layout = layout_of(ctx.client, collection)
    notice = None
    effective = mode
    using = None
    if layout == "unnamed":
        using = None
        if mode != "dense":
            notice = f"{corpus}: --mode hybrid needs a fresh corpus (remove then ingest again)"
            effective = "dense"
    elif layout == "dense-named":
        using = "dense"
        if mode != "dense":
            notice = f"{corpus}: --mode hybrid needs a fresh corpus (remove then ingest again)"
            effective = "dense"
    dense = ctx.embedder.embed_query(query)
    if effective == "keyword" and layout == "hybrid":
        points = _query_keyword(ctx.client, collection, query, k, _floor(ctx.settings, "keyword"))
    elif effective == "hybrid" and layout == "hybrid":
        points = _query_hybrid(ctx.client, collection, dense, query, k, ctx.settings)
    else:
        points = _query_dense(
            ctx.client,
            collection,
            dense,
            using if layout != "hybrid" else "dense",
            k,
            _floor(ctx.settings, "dense"),
        )
    hits = _hits_from_points(points, corpus)
    floor = _floor(ctx.settings, effective if layout == "hybrid" else "dense")
    if floor > 0:
        hits = [hit for hit in hits if hit.score >= floor]
    return hits, notice, True


def _fuse(groups: list[tuple[str, list[Hit]]], k: int) -> list[Hit]:
    if len(groups) <= 1:
        hits = groups[0][1] if groups else []
        return hits[:k]
    scored: dict[tuple, float] = {}
    chosen: dict[tuple, Hit] = {}
    for _corpus, hits in groups:
        for rank, hit in enumerate(hits, start=1):
            key = (hit.corpus, hit.source_file, hit.chunk_index)
            scored[key] = scored.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            if key not in chosen:
                chosen[key] = hit
    ordered = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    merged = []
    for key, score in ordered[:k]:
        hit = chosen[key]
        hit.score = score
        merged.append(hit)
    return merged


def search(ctx, query: str, corpora: list[str], k: int = 8, mode: str = "hybrid") -> SearchResult:
    if mode not in _MODES:
        raise ValueError(f"unknown search mode: {mode}")
    groups = []
    notices = []
    missing = []
    for corpus in corpora:
        hits, notice, found = _search_one(ctx, query, corpus, k, mode)
        if notice:
            notices.append(notice)
        if not found:
            missing.append(corpus)
        groups.append((corpus, hits))
    merged = _fuse(groups, k)
    for index, hit in enumerate(merged, start=1):
        hit.n = index
    coverage = {corpus: 0 for corpus in corpora}
    for hit in merged:
        coverage[hit.corpus] = coverage.get(hit.corpus, 0) + 1
    return SearchResult(results=merged, coverage=coverage, notices=notices, missing=missing)
