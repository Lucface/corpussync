# Agent notes

CorpusSync builds a searchable local corpus from notes, documents, transcripts, and YouTube captions. The default path needs no paid API key and no account.

## Setup

Python 3.9 or newer. Runtime dependencies are `qdrant-client` and `requests`. Optional extras: `pdf` (`pypdf`), `docx` (`python-docx`), and `dev` (`pytest`).

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e ".[pdf,docx]"
pip install -e ".[dev,pdf,docx]"
```

A checkout may already contain a virtualenv at `.venv`. Use it for tests. Do not add dependencies beyond the ones in `setup.cfg`. Do not create or edit pyproject.toml or linter or formatter configuration.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Tests are offline. They set `CORPUSSYNC_EMBEDDER=fake` and `CORPUSSYNC_CHAT=fake`, and they open Qdrant in local mode under a temporary directory. Do not point tests at a live store or a network service.

`bash -n sync.sh` checks the shell script.

## Layout

- `corpussync/` holds config, caption cleaning, chunking, extract, embed, sparse vectors, the Qdrant store, sqlite state, ingest, search, answers, and the CLI.
- `corpussync.py` is the 0.1 compatibility shim. It re-exports `clean_vtt` and `chunk`, and it still accepts `--captions`, `--titles`, `--source`, `--channel`, `--collection`, `--quality`, `--stats`, and `--force`.
- `sync.sh` pulls captions with `yt-dlp` and calls `corpussync.py`. It must run on macOS bash 3.2: no `mapfile`, no associative arrays, and no `${var,,}`.
- `channels.example.txt` uses placeholder handles only (`@YourChannel`, `mychannel`).
- `setup.cfg` defines the package, the `corpussync` console script, and the extras. `setup.py` supports older editable installers. The root shim is not packaged.

On Python 3.9 and 3.10, `corpussync.toml` is read by the minimal parser in `corpussync/config.py`. Python 3.11 and newer use `tomllib`. Every Python module starts after its docstring with `from __future__ import annotations` and stays compatible with Python 3.9 syntax.

## Text rules

Repository text, including code comments and tests:

- No em dash (U+2014) and no en dash (U+2013). Build those characters at runtime with `chr` when a title split needs them.
- In `README.md`, do not use the words "actually" or "real".
- No third-party channel names or handles. Use placeholders such as `@YourChannel` and `mychannel`.
- No event names or dates.
- No links to other repositories by the same author. The only badge allowed is this repository's own CI badge.
- No absolute home paths. Tests build paths at runtime.
- No secrets.

## Behavior to keep

- `./sync.sh` with `channels.txt` pulls captions with `yt-dlp` and ingests them.
- `clean_vtt` collapses the rolling repeats of YouTube auto-captions. Caption timing stays stripped. Do not add a feature that searches captions for a phrase and maps a hit back to cue timestamps.
- `python3 corpussync.py --captions DIR --titles TSV --source NAME --channel HANDLE` and `python3 corpussync.py --source NAME --stats` keep working.
- New collections use a named dense vector `dense` and a sparse vector `bm25`. A 0.1 collection (one unnamed vector) stays searchable in dense mode.

## Config, state and search contracts

- Read config only from `--config PATH` or `$CORPUSSYNC_HOME/corpussync.toml`, never automatically from the working directory. `init` writes the home config by default; `init --path DIR` reminds the user that an outside config needs `--config`. `sync.sh` also uses only the home config.
- Names have 1 to 63 letters, digits, dots, dashes or underscores, start with a letter or digit, and exclude `..`. Validate every incoming corpus, source and collection before store access.
- `--corpus NAME` prefers `NAME-corpus` and falls back to `NAME`, with a notice when both exist. Repeatable `--collection` on search, ask and stats selects exact names. `--all` searches each collection once. Remove requires exactly one corpus or collection and an existing target.
- Hash state is scoped to store, collection and path. Only server stores import 0.1 rows, once. Raw VTT hashes preserve 0.1 skips. Recreating a collection invalidates all its old state rows.
- A completed directory ingest prunes missing supported files from that root. `--keep-missing`, file arguments, missing roots and failed scans prevent pruning. Extraction or write failures return 1 and appear in final counts. Short file notes become one point; YouTube retains its 30-word minimum.
- Hybrid search gates on cosine relevance (default dense floor 0.65) or full keyword coverage (default keyword floor 1.0), then ranks all corpora together. Partial matches earn no keyword rank term. `--min-score` on search and ask overrides the dense floor. No good match exits 2, and ask never calls chat in that case. Keyword mode never embeds. Ignore old hybrid floor config entries.
- 0.65 suits nomic-embed-text; unrelated queries reached about 0.61 on twenty thousand chunks, and relevant ones scored 0.75 and up. Raise it for unrelated answers, lower it for refused covered questions, and tune it for another model.
- Skip foreign vector layouts and isolate collection query errors. Preserve unprefixed embeddings for unnamed 0.1 vectors. New hybrid collections retain `Modifier.IDF`.
- Compat uses explicit server configuration first. Otherwise `CORPUSSYNC_STORE=embedded` starts fresh; without old state, embedded is also the default with no probe. Nonempty 0.1 state means probe the 0.1 server and use it, or stop with exit 2 if it is unavailable. The normal command never makes that automatic probe; doctor may point out a legacy server.
- Create private directories (`0700`), tighten existing home and embedded store directories, and chmod state files to `0600`. `sync.sh` uses `umask 077`.
- Only yt-dlp's requests to YouTube when you sync a channel, unless you point Ollama or Qdrant at another machine, which corpussync then names on every run.
- `OLLAMA_HOST` accepts a bare host, host:port, base URL or bracketed IPv6 address. Embedding, chat and doctor share URL normalization. Chat waits up to 300 seconds.
- `sync.sh` records ids with no captions in `no-captions.txt` and skips them before counting the cap. Delete no-captions.txt to retry those videos. Keep quoted tilde expansion and BSD-compatible sed expressions.
- New or changed tests include a `source:` docstring naming the guarded defect. Tests use injected probes, fake models and temporary stores, never live services.

## Commands

```bash
corpussync init
corpussync ingest ./notes --corpus notes
corpussync search "your question" --corpus notes
corpussync ask "your question" --corpus notes
.venv/bin/python -m pytest -q
bash -n sync.sh
```
