"""Command line: init, ingest, youtube, search, ask, stats, list, remove, doctor."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import requests

from corpussync.answer import ask
from corpussync.config import DEFAULT_HOME, config_template, load_context, ollama_base_url, private_home, privacy_notices
from corpussync.names import InvalidName, check_name
from corpussync.ingest import ingest_paths, ingest_youtube
from corpussync.search import no_match_message, print_coverage, print_hits, print_notices, search, select_collections
from corpussync.state import delete_collection_rows, store_identity, legacy_state_path
from corpussync.store import collection_names, layout_label, point_count, resolve_collection, legacy_store_choice, legacy_server_answers, legacy_address


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="corpussync")
    parser.add_argument("--config", default=None, help="path to corpussync.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="write corpussync.toml with every key commented at its default")
    init.add_argument("--path", help="directory to write corpussync.toml in")

    ingest = sub.add_parser("ingest", help="ingest files and folders into a corpus")
    ingest.add_argument("paths", nargs="+")
    ingest.add_argument("--corpus", required=True)
    ingest.add_argument("--force", action="store_true")
    ingest.add_argument("--keep-missing", action="store_true")

    youtube = sub.add_parser("youtube", help="ingest a directory of YouTube captions")
    youtube.add_argument("--captions", required=True)
    youtube.add_argument("--titles")
    youtube.add_argument("--source", required=True)
    youtube.add_argument("--channel", default="")
    youtube.add_argument("--collection")
    youtube.add_argument("--quality", default="silver")
    youtube.add_argument("--force", action="store_true")

    find = sub.add_parser("search", help="search one or more corpora")
    find.add_argument("query")
    find.add_argument("--corpus", action="append", default=[])
    find.add_argument("--collection", action="append", default=[])
    find.add_argument("--min-score", type=float)
    find.add_argument("--all", action="store_true", dest="search_all")
    find.add_argument("-k", type=int, default=8)
    find.add_argument("--mode", choices=["hybrid", "dense", "keyword"], default="hybrid")
    find.add_argument("--json", action="store_true")

    question = sub.add_parser("ask", help="answer a question from the corpus with a local model")
    question.add_argument("question")
    question.add_argument("--corpus", action="append", default=[])
    question.add_argument("--collection", action="append", default=[])
    question.add_argument("--min-score", type=float)
    question.add_argument("--all", action="store_true", dest="search_all")
    question.add_argument("-k", type=int, default=6)
    question.add_argument("--model")

    stats = sub.add_parser("stats", help="point count for one corpus, or every corpus")
    stats.add_argument("--corpus")
    stats.add_argument("--collection", action="append", default=[])

    listed = sub.add_parser("list", help="each corpus with its point count and layout")

    remove = sub.add_parser("remove", help="drop a corpus and its state rows")
    selection = remove.add_mutually_exclusive_group(required=True)
    selection.add_argument("--corpus")
    selection.add_argument("--collection")
    remove.add_argument("--yes", action="store_true")

    doctor = sub.add_parser("doctor", help="check Ollama, the embed and answer models, the store, and yt-dlp")
    for child in (init, ingest, youtube, find, question, stats, listed, remove, doctor):
        child.add_argument("--config", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    return parser


def _corpora(ctx, args):
    if not args.search_all and not args.corpus and not args.collection:
        print("pass --corpus NAME, --collection NAME or --all", file=sys.stderr)
        return None
    names = list(args.collection)
    if args.search_all:
        names.extend(sorted(ctx.collections))
    exact = []
    for name in names:
        if name not in exact:
            exact.append(name)
    return list(args.corpus), exact


def _result_payload(query: str, mode: str, result) -> dict:
    return {
        "query": query,
        "mode": mode,
        "results": [
            {
                "n": hit.n,
                "corpus": hit.corpus,
                "score": hit.score,
                "relevance": hit.relevance,
                "keyword_coverage": hit.keyword_coverage,
                "title": hit.title,
                "locator": hit.locator,
                "url": hit.url,
                "source_file": hit.source_file,
                "chunk_index": hit.chunk_index,
                "text": hit.text,
            }
            for hit in result.results
        ],
        "coverage": result.coverage,
    }


def cmd_init(directory: str | None) -> int:
    home = Path(os.environ.get("CORPUSSYNC_HOME", DEFAULT_HOME)).expanduser()
    folder = Path(directory).expanduser() if directory else home
    private_home(folder)
    if folder.resolve() != home.resolve():
        print("a file outside CORPUSSYNC_HOME is read only through --config")
    target = folder / "corpussync.toml"
    if target.exists():
        print(f"left existing corpussync.toml at {target}")
        return 0
    target.write_text(config_template(), encoding="utf-8")
    print(f"wrote {target}")
    return 0


def stats_collection(ctx, collection: str) -> int:
    try:
        ctx.client.get_collection(collection)
        count = point_count(ctx.client, collection)
    except Exception as exc:
        print(f"{collection}: not found ({exc})")
        return 0
    print(f"{collection}: {count} points")
    return 0


def cmd_stats(ctx, corpus: str | None, collections=None) -> int:
    if corpus or collections:
        names, missing = select_collections(ctx, [corpus] if corpus else [], collections)
        for name in missing:
            print(f"{name}: not found")
    else:
        names = sorted(ctx.collections)
    if not names:
        print("no corpora")
        return 0
    for name in names:
        stats_collection(ctx, name)
    return 0


def cmd_list(ctx) -> int:
    names = sorted(ctx.collections)
    if not names:
        print("no corpora")
        return 0
    for name in names:
        label = layout_label(ctx.client, name, ctx.settings.embed_dim)
        count = point_count(ctx.client, name)
        corpus = name[: -len("-corpus")] if name.endswith("-corpus") else name
        print(f"{corpus}  {count} points  {label}  ({name})")
    return 0


def cmd_remove(ctx, corpus: str | None, yes: bool, collection: str | None = None) -> int:
    if collection:
        check_name(collection, "collection")
        target = collection if collection in ctx.collections else None
    else:
        target = resolve_collection(ctx, corpus)
    if target is None:
        print(f"{collection or corpus}: not found")
        return 1
    if not yes:
        print(
            f"This drops collection {target} and its state rows. "
            "Re-run with --yes to remove it."
        )
        return 1
    try:
        ctx.client.delete_collection(collection_name=target)
        print(f"removed {target}")
    except Exception as exc:
        print(f"{target}: could not remove ({exc})")
        return 1
    delete_collection_rows(ctx.db, store_identity(ctx.settings), target)
    ctx.collections.discard(target)
    return 0


def cmd_doctor(ctx) -> int:
    failed = False
    settings = ctx.settings
    base = ollama_base_url(settings)
    if not settings.using_server() and legacy_state_path(settings) is not None:
        if legacy_server_answers(requests.get):
            print(f"note: a Qdrant server answers at {legacy_address()}, where 0.1 kept its corpora; set QDRANT_URL to search them")
    try:
        resp = requests.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
        print(f"ollama: ok ({base})")
        names = [item.get("name", "") for item in resp.json().get("models", [])]
        want = settings.embed_model
        pulled = any(name == want or name.split(":")[0] == want for name in names)
        if pulled:
            print(f"embed model {want}: ok")
        else:
            failed = True
            print(f"embed model {want} is not pulled. Fix: ollama pull {want}")
        want = settings.answer_model
        pulled = any(name == want or name.split(":")[0] == want for name in names)
        if pulled:
            print(f"answer model {want}: ok")
        else:
            print(f"warning: answer model {want} is not pulled (only ask needs it). Fix: ollama pull {want}")
    except Exception as exc:
        failed = True
        print(f"Ollama did not answer at {base}. Fix: start Ollama (ollama serve). ({exc})")
        print(
            f"embed model {settings.embed_model} could not be checked. "
            f"Fix: ollama pull {settings.embed_model}"
        )
    try:
        collection_names(ctx.client)
        if settings.qdrant_url:
            where = settings.qdrant_url
        elif settings.qdrant_host:
            where = f"{settings.qdrant_host}:{settings.qdrant_port}"
        else:
            where = str(settings.home / "qdrant")
        print(f"store: ok ({where})")
    except Exception as exc:
        failed = True
        embedded = settings.home / "qdrant"
        print(
            "store did not open. Fix: unset QDRANT_URL and QDRANT_HOST to use the "
            f"embedded store at {embedded}, or start the Qdrant server. ({exc})"
        )
    if shutil.which("yt-dlp"):
        print("yt-dlp: ok")
    else:
        print("warning: yt-dlp is not on PATH. Fix: install yt-dlp to sync channel captions.")
    return 1 if failed else 0


def cmd_search(ctx, args) -> int:
    selection = _corpora(ctx, args)
    if selection is None:
        return 1
    corpora, collections = selection
    if not corpora and not collections:
        print(no_match_message([]))
        return 2
    result = search(ctx, args.query, corpora, k=args.k, mode=args.mode,
                    min_score=args.min_score, collections=collections)
    print_notices(result.notices)
    for corpus in result.missing:
        print(f"{corpus}: not found", file=sys.stderr)
    if not result.results:
        if result.error is not None:
            print(f"could not search: {result.error}", file=sys.stderr)
            return 1
        print(no_match_message(corpora + collections))
        return 2
    if args.json:
        print(json.dumps(_result_payload(args.query, args.mode, result)))
        return 0
    print_hits(result)
    print_coverage(list(result.coverage), result.coverage)
    return 0


def cmd_ask(ctx, args) -> int:
    selection = _corpora(ctx, args)
    if selection is None:
        return 1
    corpora, collections = selection
    if not corpora and not collections:
        print(no_match_message([]))
        return 2
    return ask(ctx, args.question, corpora, k=args.k, model=args.model,
               min_score=args.min_score, collections=collections)


def _validate_names(args) -> None:
    for field, what in (("corpus", "corpus"), ("source", "source"), ("collection", "collection")):
        value = getattr(args, field, None)
        for name in value if isinstance(value, list) else ([value] if value is not None else []):
            check_name(name, what)


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _validate_names(args)
    except InvalidName as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.command == "init":
        return cmd_init(args.path)
    ctx = load_context(args.config)
    try:
        if args.command in ("ingest", "youtube", "search", "ask"):
            privacy_notices(ctx.settings)
        if args.command == "ingest":
            return ingest_paths(ctx, [Path(item) for item in args.paths], args.corpus, args.force, args.keep_missing)
        if args.command == "youtube":
            if args.collection:
                ctx.collection_override = args.collection
            return ingest_youtube(
                ctx,
                Path(args.captions).expanduser(),
                Path(args.titles).expanduser() if args.titles else None,
                args.source,
                args.channel or args.source,
                args.quality,
                args.force,
            )
        if args.command == "search":
            return cmd_search(ctx, args)
        if args.command == "ask":
            return cmd_ask(ctx, args)
        if args.command == "stats":
            return cmd_stats(ctx, args.corpus, args.collection)
        if args.command == "list":
            return cmd_list(ctx)
        if args.command == "remove":
            return cmd_remove(ctx, args.corpus, args.yes, args.collection)
        if args.command == "doctor":
            return cmd_doctor(ctx)
    finally:
        ctx.close()
    return 1


def main_compat(argv=None, probe=None) -> int:
    """0.1 flags: --captions, --titles, --source, --channel, --collection, --quality, --stats, --force."""
    ap = argparse.ArgumentParser(prog="corpussync.py")
    ap.add_argument("--captions", help="directory of .vtt caption files")
    ap.add_argument("--titles", help="video-list.tsv (id<TAB>title)")
    ap.add_argument("--source", required=True, help="short source name, e.g. mychannel")
    ap.add_argument("--channel", default="", help="channel handle, e.g. @YourChannel")
    ap.add_argument("--collection", help="override collection name (default <source>-corpus)")
    ap.add_argument("--quality", default="silver", help="quality tier (gold/silver/bronze)")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    try:
        _validate_names(args)
    except InvalidName as exc:
        print(str(exc), file=sys.stderr)
        return 2
    ctx = load_context()
    try:
        noted = False

        def compat_probe(url, timeout):
            nonlocal noted
            if not args.stats:
                privacy_notices(replace(ctx.settings, qdrant_url=legacy_address()))
                noted = True
            return (probe or requests.get)(url, timeout=timeout)

        if legacy_store_choice(ctx.settings, compat_probe) == "stop":
            return 2
        if not args.stats and not noted:
            privacy_notices(ctx.settings)
        collection = args.collection or f"{args.source}-corpus"
        if args.stats:
            return stats_collection(ctx, collection)
        if not args.captions:
            ap.error("--captions is required unless --stats")
        if args.collection:
            ctx.collection_override = args.collection
        return ingest_youtube(
            ctx,
            Path(args.captions).expanduser(),
            Path(args.titles).expanduser() if args.titles else None,
            args.source,
            args.channel or args.source,
            args.quality,
            args.force,
        )
    finally:
        ctx.close()
