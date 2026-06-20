# CorpusSync

Turn any **YouTube channel into a queryable Qdrant corpus — for free.** CorpusSync pulls a channel's auto-captions with `yt-dlp` (no API key, no transcription cost), cleans the rolling-duplication mess YouTube auto-subs produce into real prose, chunks + embeds it with a local Ollama embedder (`nomic-embed-text`, 768d), and upserts into a per-channel Qdrant collection. Re-runs are hash-skipped, so a daily sync only embeds new videos.

Point it at a creator you learn from and your coding agent can query their whole back-catalog as an intelligence corpus.

## How it works

```
yt-dlp (free captions) ──▶ clean_vtt() ──▶ chunk (512-tok windows) ──▶ Ollama embed ──▶ Qdrant <source>-corpus
        no API cost            de-dupes         64-tok overlap          nomic 768d        deterministic IDs, hash-skip
```

The one non-obvious piece is `clean_vtt()`: YouTube auto-subs animate — each cue re-shows the previous partial line plus a few new words, so every sentence appears 2–3× with inline `<timing>` tags. CorpusSync strips the tags and collapses the roll-up back into clean prose before embedding.

## Quick start

```bash
pip install -r requirements.txt        # qdrant-client + requests
# you need a running Qdrant (localhost:6333) and Ollama with nomic-embed-text:
#   ollama pull nomic-embed-text

cp channels.example.txt channels.txt   # edit to your channels
./sync.sh                              # pulls captions + ingests all channels
./sync.sh ycombinator                  # just one
```

Or ingest a directory of `.vtt` files you already have:

```bash
python3 corpussync.py --captions ./captions --titles video-list.tsv --source aiengineer --channel "@aiDotEngineer"
python3 corpussync.py --source aiengineer --stats     # point count
```

## Config (env vars, all optional)

| Var | Default | |
|---|---|---|
| `QDRANT_HOST` / `QDRANT_PORT` | `127.0.0.1` / `6333` | where Qdrant lives (remote over an SSH tunnel works — calls auto-retry through blips) |
| `OLLAMA_HOST` / `OLLAMA_PORT` | `127.0.0.1` / `11434` | the embedder |
| `CORPUSSYNC_EMBED_MODEL` / `CORPUSSYNC_EMBED_DIM` | `nomic-embed-text` / `768` | swap the embedder (match the dim) |
| `CORPUSSYNC_STATE` | `~/.corpussync/ingestion-state.db` | hash-skip state DB |
| `COOKIE_JAR` | — | path to a `yt-dlp` cookie jar to pull authenticated (avoids YouTube's unauthenticated throttle) |
| `PER_RUN_CAP` | `250` | max new caption pulls per channel per run |

## Payload shape

Each point carries `doc_type, source, channel, source_name, video_id, url, title, author, quality_tier, chunk_index, chunk_total, text`. Stable across versions — any Qdrant query client reads it unchanged. Re-ingesting a longer edit of a video overwrites chunks `0..N-1` and deletes stale trailing chunks *after* the upsert, so the video is never momentarily missing.

## License

MIT — see [LICENSE](./LICENSE). By [@Lucface](https://github.com/Lucface).
