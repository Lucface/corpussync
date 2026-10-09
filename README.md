[![tests](https://github.com/Lucface/corpussync/actions/workflows/test.yml/badge.svg)](https://github.com/Lucface/corpussync/actions/workflows/test.yml)

# CorpusSync

Build a searchable corpus from your own notes, documents, transcripts and your channel's captions, on your own machine, for free.

## Respect what you index

Index what you own or are allowed to use, and follow each platform's terms.

## Quick start

Python 3.10 or newer.

```bash
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

## Qdrant server

With `QDRANT_URL`, `QDRANT_HOST`, and `qdrant.url` unset, CorpusSync stores vectors in an embedded Qdrant database at `~/.corpussync/qdrant`. Set `QDRANT_URL`, or set `QDRANT_HOST` and `QDRANT_PORT`, or set `qdrant.url` in `corpussync.toml`, to use a Qdrant server.

## Hybrid search

Hybrid search asks the store for a dense vector match and a keyword match, then fuses the two lists with reciprocal rank fusion. Use `--mode hybrid` (the default), `--mode dense`, or `--mode keyword`.

## Ask

`corpussync ask "your question" --corpus notes` runs a search, then asks a local Ollama chat model to answer only from the numbered sources and to cite them as `[n]`. The model is `answer_model` (default `llama3.2`). Pass `--model NAME` to override it for one question. If nothing clears the relevance floor, ask prints the refusal and exits 2 without calling the model.

## Coding agents

`corpussync search "your question" --corpus notes --json` prints one JSON object. The process exits 0 when at least one result clears the relevance floor. It exits 2 when none does, and prints `no good match in: <corpora>`.

`corpussync stats --corpus notes` prints a point count. `corpussync list` prints each corpus with its point count and layout. `corpussync doctor` checks that Ollama answers, that the embed model is pulled, that the store opens, and whether `yt-dlp` is on `PATH` (a warning only).

## Configuration

Command-line flags win, then environment variables, then `corpussync.toml`, then defaults. The file is read from `--config PATH`, then `./corpussync.toml`, then `$CORPUSSYNC_HOME/corpussync.toml`. `corpussync init` writes the file with every key commented at its default. On Python 3.10 the file is read by a minimal parser (comments, tables, strings, numbers, and booleans). Python 3.11 and newer use `tomllib`.

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
|  | `search.min_score.dense` | `0.2` |
|  | `search.min_score.keyword` | `0.05` |
|  | `search.min_score.hybrid` | `0.0` |

`CORPUSSYNC_EMBEDDER` and `CORPUSSYNC_CHAT` accept `ollama` or `fake`. When the model name contains `nomic-embed`, documents are prefixed with `search_document: ` and queries with `search_query: `.

## Stop and remove

Nothing runs in the background. Ctrl-C stops a sync. Remove one corpus with `corpussync remove --corpus NAME --yes`. Remove the local data by deleting `~/.corpussync`. Uninstall with `pip uninstall corpussync`.

## What leaves your machine

What leaves your machine: only the requests yt-dlp makes to YouTube when you sync a channel; embeddings, storage and answers stay local.

## Upgrading from 0.1

Collections created by 0.1 keep working in dense mode. Search prints a one-line notice that `--mode hybrid` needs a fresh corpus (`corpussync remove --corpus NAME --yes`, then ingest again). `./sync.sh` and `python3 corpussync.py --captions ... --source ... --channel ...` still work, and `python3 corpussync.py --source NAME --stats` still prints a count.

## Payload

A YouTube point keeps `doc_type` `talk`, `source` `youtube`, `channel`, `source_name`, `source_file`, `video_id`, `url`, `title`, `author`, `publish_date`, `quality_tier`, `chunk_index`, `chunk_total`, `text`, `ingested_at`, and `expires_at`.

A file point has `doc_type` `file`, `source` `files`, `source_name` (the corpus), `source_file` (absolute path), `locator` (path relative to the ingested root), `title` (first markdown or html heading, otherwise the file stem), `chunk_index`, `chunk_total`, `text`, `ingested_at`, and `modified_at`.

Point ids are `uuid5(NAMESPACE_URL, "{source_file}::{index}")`. Re-ingest overwrites those ids, then deletes chunks with `chunk_index` greater than or equal to the new count.

## Python API

`corpussync.vtt.clean_vtt(raw)` and `corpussync.vtt.clean_srt(raw)` return prose. `corpussync.chunking.chunk(text, max_tokens=512, overlap=64)` returns windows. `corpussync.ingest.ingest_document(ctx, *, text, source_file, corpus, title, locator, extra_payload=None, force=False)` returns the number of chunks written, or 0 when the content hash is unchanged. `corpussync.search.search(ctx, query, corpora, k=8, mode="hybrid")` returns a `SearchResult` with `.results` and `.coverage`.

## License

MIT, see LICENSE.
