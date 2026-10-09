[![tests](https://github.com/Lucface/corpussync/actions/workflows/test.yml/badge.svg)](https://github.com/Lucface/corpussync/actions/workflows/test.yml)

# CorpusSync

Build a searchable corpus from your own notes, documents, transcripts and your channel's captions, on your own machine, for free.

## Respect what you index

Index what you own or are allowed to use, and follow each platform's terms.

## Quick start

Python 3.9 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e ".[pdf,docx]"
ollama pull nomic-embed-text
corpussync init
corpussync ingest ./notes --corpus notes
corpussync search "your question" --corpus notes
```

`pip install -e ".[dev,pdf,docx]"` adds pytest. The `pdf` and `docx` extras are optional.

## YouTube captions

```bash
cp channels.example.txt channels.txt
# edit channels.txt: source|@handle|URL
./sync.sh
```

`./sync.sh` pulls captions with yt-dlp and no API key. `clean_vtt` collapses the rolling repeats of auto-captions.

A caption directory you already have:

```bash
python3 corpussync.py --captions ./captions --titles video-list.tsv --source mychannel --channel "@YourChannel"
python3 corpussync.py --source mychannel --stats
```

`COOKIE_JAR` is a yt-dlp cookies file exported from your own account, for when YouTube limits requests without one.

Videos without captions are recorded in `$CORPUSSYNC_DATA/<source>/no-captions.txt` only after a successful yt-dlp call and skipped before the per-run cap; delete no-captions.txt to retry those videos. Failed downloads are retried on the next run.

## Ingesting folders

`corpussync ingest ./notes --corpus notes` scans supported, non-hidden files. Short notes become one chunk. After a complete directory scan, deleted and renamed files are removed from that corpus. Pass `--keep-missing` to keep their old points. A file argument or a missing directory never prunes. A missing path or any extraction or write failure exits 1. Files missing the optional PDF or DOCX extra count as skipped, preserve their existing points and do not cause a failure exit. The final line reports files, chunks, removals, skips and failures.

State is scoped to the store, collection and file path. The same file can be ingested into several corpora or stores. Recreating a deleted collection clears its old skip state before rebuilding it.

## Qdrant server

With `QDRANT_URL`, `QDRANT_HOST`, `qdrant.url` and `qdrant.host` unset, the `corpussync` command stores vectors in an embedded Qdrant database at `$CORPUSSYNC_HOME/qdrant` (default `~/.corpussync/qdrant`). Set `QDRANT_URL`, or set `QDRANT_HOST` and `QDRANT_PORT`, or configure the server in the trusted `corpussync.toml`, to use a Qdrant server. Only one process can open the embedded store at a time; wait for it to finish or use a server for concurrent access.

## Hybrid search

Hybrid search asks each collection for dense and keyword candidates, then computes cosine relevance from their stored vectors. Candidates pass when relevance reaches `search.min_score.dense` (default `0.65`) or they cover every query keyword (`search.min_score.keyword`, default `1.0`). All kept candidates share one ranking across collections. Reciprocal rank fusion adds a dense rank term and, only for candidates meeting the keyword coverage floor, a keyword rank term. Partial keyword matches get only the dense term.

Use `--mode hybrid` (the default), `--mode dense`, or `--mode keyword`. Dense mode gates and ranks by cosine relevance. Keyword mode runs without Ollama, keeps any keyword overlap and ranks by keyword coverage. Hybrid and dense output shows cosine relevance; keyword output shows coverage. JSON includes `score`, `relevance` and `keyword_coverage`.

0.65 suits nomic-embed-text. Unrelated questions reached about 0.61 on a corpus of twenty thousand chunks, and relevant ones scored 0.75 and up. Raise it if ask answers from unrelated material, lower it if it refuses questions your notes cover, and set a new value for another model.

Pass `--min-score FLOAT` to search or ask to override the dense floor for one call. Collections with incompatible vectors are skipped with a notice, and a failed collection does not prevent results from other collections.

## Corpus and collection names

Names have 1 to 63 letters, digits, dots, dashes or underscores, start with a letter or digit, and cannot contain `..`. `--corpus notes` selects `notes-corpus` when it exists, otherwise `notes`. If both exist, a notice explains the selection. Use repeatable `--collection NAME` on search, ask and stats to select exact names. Repeat `--corpus` on search or ask for several corpora, or use `--all` to visit each collection once. `list` includes the exact collection name in parentheses.

## Ask

`corpussync ask "your question" --corpus notes` runs a search, then asks a local Ollama chat model to answer only from the numbered sources and to cite them as `[n]`. The model is `answer_model` (default `llama3.2`). Pass `--model NAME` to override it for one question. If nothing clears the relevance floor, ask prints the refusal and exits 2 without calling the model. If every selected collection fails to query or embed, ask prints `could not search: <error>` to stderr and exits 1 without calling the model.

## Coding agents

`corpussync search "your question" --corpus notes --json` prints one JSON object. The process exits 0 when at least one result clears the relevance floor. It exits 2 when none does, and prints `no good match in: <corpora>`. If every selected collection fails to query or embed, it prints `could not search: <error>` to stderr and exits 1.

`corpussync stats --corpus notes` prints a point count. `corpussync list` prints each corpus with its point count and layout. `corpussync doctor` checks that Ollama answers, that the embed model is pulled, that the store opens, whether the answer model is pulled and whether `yt-dlp` is on `PATH`. A missing answer model or `yt-dlp` produces a warning without changing the exit code; only ask needs the answer model.

## Configuration

Command-line flags win, then environment variables, then `corpussync.toml`, then defaults. The file is read from `--config PATH` when given, otherwise `$CORPUSSYNC_HOME/corpussync.toml`. The default home is `~/.corpussync`; a config in the current working directory is never read automatically. `corpussync init` writes the home config with every key commented at its default. `init --path DIR` writes `DIR/corpussync.toml` and reminds you to use `--config` for a file outside the home directory. On Python 3.9 and 3.10 the file is read by a minimal parser (comments, tables, strings, numbers, and booleans). Python 3.11 and newer use `tomllib`. `sync.sh` reads only the home config.

| Environment variable | TOML key | Default |
| --- | --- | --- |
| `QDRANT_URL` | `qdrant.url` | unset |
| `QDRANT_HOST` | `qdrant.host` | unset |
| `QDRANT_PORT` | `qdrant.port` | `6333` |
| `OLLAMA_HOST` | `ollama.host` | `127.0.0.1` |
| `OLLAMA_PORT` | `ollama.port` | `11434` |
| `CORPUSSYNC_EMBED_MODEL` | `embed.model` | `nomic-embed-text` |
| `CORPUSSYNC_EMBED_DIM` | `embed.dim` | `768` |
| `CORPUSSYNC_EMBEDDER` | `embed.backend` | `ollama` (`fake` for tests) |
| `CORPUSSYNC_CHAT` | `chat.backend` | `ollama` (`fake` for tests) |
|  | `answer_model` | `llama3.2` |
| `CORPUSSYNC_STATE` | `state.path` | `$CORPUSSYNC_HOME/state.db` |
| `CORPUSSYNC_HOME` | `home` | `~/.corpussync` |
| `CORPUSSYNC_DATA` | `data_dir` | `~/.corpussync/channels` |
| `CHANNELS_FILE` | `channels_file` | `channels.txt` beside `sync.sh` |
| `PER_RUN_CAP` | `per_run_cap` | `250` |
| `PYBIN` | `pybin` | `python3` |
| `COOKIE_JAR` | `cookie_jar` | unset |
|  | `search.min_score.dense` | `0.65` |
|  | `search.min_score.keyword` | `1.0` |

`CORPUSSYNC_EMBEDDER` and `CORPUSSYNC_CHAT` accept `ollama` or `fake`. `OLLAMA_HOST` accepts a bare host, `host:port`, a bracketed IPv6 address, or a base URL whose port and path are preserved. A bare host uses `OLLAMA_PORT`. When the model name contains `nomic-embed`, documents are prefixed with `search_document: ` and queries with `search_query: `, except in unnamed 0.1 collections, which keep their original unprefixed style. Old hybrid floor config entries are ignored.

## Stop and remove

Nothing runs in the background. Ctrl-C stops a sync. Remove one corpus with `corpussync remove --corpus NAME --yes`, or select an exact collection with `remove --collection NAME --yes`. Removal requires an existing collection and deletes only its state rows in the selected store. Remove the local data by deleting `~/.corpussync`. Uninstall with `pip uninstall corpussync`.

## What leaves your machine

Only yt-dlp's requests to YouTube when you sync a channel, unless you point Ollama or Qdrant at another machine, which corpussync then names on every run. Loopback and unspecified addresses (`0.0.0.0` and `::`) count as local.

CorpusSync creates its home and embedded store with owner-only directory permissions (`0700`). It tightens an existing default home (`~/.corpussync`) and always tightens its `qdrant` directory. An existing custom `CORPUSSYNC_HOME` keeps its permissions; a note warns if it has group or other permission bits. A new state directory is also private and the database file uses `0600`. `sync.sh` creates private files and directories with `umask 077`.

## Upgrading from 0.1

Collections created by 0.1 keep working in dense mode. Search prints a one-line notice that `--mode hybrid` needs a fresh corpus (`corpussync remove --corpus NAME --yes`, then ingest again). `./sync.sh` and `python3 corpussync.py --captions ... --source ... --channel ...` still work, and `python3 corpussync.py --source NAME --stats` still prints a count.

The compatibility path uses an explicitly configured Qdrant server first. Otherwise, `CORPUSSYNC_STORE=embedded` starts fresh without probing. With no 0.1 hash state it also uses embedded storage without probing. With 0.1 state in `CORPUSSYNC_STATE` or `$CORPUSSYNC_HOME/ingestion-state.db`, it checks the 0.1 server at `http://127.0.0.1:6333` (override with `CORPUSSYNC_LEGACY_QDRANT`). If that server answers, it uses it and prints the state path; otherwise it exits 2 and asks you to start the server, set `QDRANT_URL`, or explicitly begin a fresh embedded corpus. It never silently forks an existing 0.1 corpus.

The `corpussync` command keeps its embedded default and never makes that automatic probe. Set `QDRANT_URL` to search the old server through the new command. `doctor` can point out an answering legacy server when old state exists. On first use of a server store, old hash rows are imported once into scoped state; embedded stores never import them. Caption hashes continue to use raw VTT text so unchanged 0.1 files can skip embedding.

## Payload

A YouTube point keeps `doc_type` `talk`, `source` `youtube`, `channel`, `source_name`, `source_file`, `video_id`, `url`, `title`, `author`, `publish_date`, `quality_tier`, `chunk_index`, `chunk_total`, `text`, `ingested_at`, and `expires_at`.

A file point has `doc_type` `file`, `source` `files`, `source_name` (the corpus), `source_file` (absolute path), `locator` (path relative to the ingested root), `title` (first markdown, HTML or DOCX heading, otherwise the file stem), `chunk_index`, `chunk_total`, `text`, `ingested_at`, and `modified_at`. DOCX tables are read in document order, including nested tables. Keyword tokens include accented and non-Latin letters and digits.

Point ids are `uuid5(NAMESPACE_URL, "{source_file}::{index}")`. Re-ingest overwrites those ids, then deletes chunks with `chunk_index` greater than or equal to the new count.

## Python API

`corpussync.vtt.clean_vtt(raw)` and `corpussync.vtt.clean_srt(raw)` return prose. `corpussync.chunking.chunk(text, max_tokens=512, overlap=64)` returns windows. `corpussync.ingest.ingest_document(ctx, *, text, source_file, corpus, title, locator, extra_payload=None, force=False, digest=None)` returns the number of chunks written, or 0 when the content hash is unchanged. `corpussync.search.search(ctx, query, corpora, k=8, mode="hybrid", min_score=None, collections=None)` returns a `SearchResult` with `.results`, `.coverage`, `.notices`, `.missing` and `.failed` (collection names with query or embedder errors). `.error` holds the first error, limited to 120 characters, when every selected collection failed; otherwise it is `None`.

## License

MIT, see LICENSE.
