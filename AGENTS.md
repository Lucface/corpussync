# Agent notes

CorpusSync builds a searchable local corpus from notes, documents, transcripts, and YouTube captions. The default path needs no paid API key and no account.

## Setup

Python 3.10 or newer. Runtime dependencies are `qdrant-client` and `requests`. Optional extras: `pdf` (`pypdf`), `docx` (`python-docx`), and `dev` (`pytest`).

```bash
pip install -e ".[dev,pdf,docx]"
```

A checkout may already contain a virtualenv at `.venv`. Use it for tests. Do not add dependencies beyond the ones in `pyproject.toml`.

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
- `pyproject.toml` defines the package, the `corpussync` console script, and the extras.

On Python 3.10, `corpussync.toml` is read by the minimal parser in `corpussync/config.py`. Python 3.11 and newer use `tomllib`.

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

## Commands

```bash
corpussync init
corpussync ingest ./notes --corpus notes
corpussync search "your question" --corpus notes
corpussync ask "your question" --corpus notes
.venv/bin/python -m pytest -q
bash -n sync.sh
```
