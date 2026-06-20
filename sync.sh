#!/usr/bin/env bash
# sync.sh — incremental ingest of tracked YouTube channels into per-channel Qdrant corpora.
# For each channel in a channels file: refresh the video-id list, pull FREE auto-captions
# for ids not yet on disk (yt-dlp, no API/transcription cost), then embed+upsert via
# corpussync.py. Hash-skip means only new videos embed, so it's idempotent + re-runnable.
#
# Usage:
#   ./sync.sh                       # all channels in ./channels.txt
#   ./sync.sh ycombinator           # only matching source(s)
#   CHANNELS_FILE=mine.txt ./sync.sh
#
# channels file format (one per line):   source|@handle|https://www.youtube.com/@handle/videos
# Optional: set COOKIE_JAR=/path/to/cookies.txt to pull authenticated (dodges YouTube's
# unauthenticated bot-throttle). Export PER_RUN_CAP to cap caption pulls per channel/run.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
INGEST="$HERE/corpussync.py"
CHANNELS_FILE="${CHANNELS_FILE:-$HERE/channels.txt}"
DATA_DIR="${CORPUSSYNC_DATA:-$HOME/.corpussync/channels}"
PER_RUN_CAP="${PER_RUN_CAP:-250}"
PYBIN="${PYBIN:-python3}"

command -v yt-dlp >/dev/null || { echo "yt-dlp not found — install it first (brew install yt-dlp)"; exit 1; }
[ -f "$CHANNELS_FILE" ] || { echo "no channels file at $CHANNELS_FILE (copy channels.example.txt)"; exit 1; }

YT_COOKIES=()
[ -n "${COOKIE_JAR:-}" ] && [ -s "${COOKIE_JAR:-}" ] && YT_COOKIES=(--cookies "$COOKIE_JAR")

# optional positional filter
mapfile -t CHANNELS < <(grep -vE '^\s*#|^\s*$' "$CHANNELS_FILE")
if [ "$#" -gt 0 ]; then
  _f=(); for want in "$@"; do for e in "${CHANNELS[@]}"; do [ "${e%%|*}" = "$want" ] && _f+=("$e"); done; done
  CHANNELS=("${_f[@]}")
fi

for entry in "${CHANNELS[@]}"; do
  SOURCE="${entry%%|*}"; rest="${entry#*|}"; HANDLE="${rest%%|*}"; URL="${rest##*|}"
  CAP_DIR="$DATA_DIR/$SOURCE/captions"; mkdir -p "$CAP_DIR"
  echo "── $SOURCE ($HANDLE) ──"

  # 1. refresh id + title list (cheap flat-playlist; no downloads)
  yt-dlp "${YT_COOKIES[@]}" --flat-playlist --print "%(id)s\t%(title)s" "$URL" \
    > "$DATA_DIR/$SOURCE/video-list.tsv" 2>/dev/null || { echo "  list refresh failed"; continue; }
  cut -f1 "$DATA_DIR/$SOURCE/video-list.tsv" > "$DATA_DIR/$SOURCE/all-ids.txt"

  # 2. pull captions only for ids not already on disk (capped per run)
  pulled=0
  while IFS= read -r vid; do
    [ -z "$vid" ] && continue
    ls "$CAP_DIR/$vid".*.vtt >/dev/null 2>&1 && continue
    [ "$pulled" -ge "$PER_RUN_CAP" ] && break
    yt-dlp "${YT_COOKIES[@]}" --write-auto-subs --sub-langs en --skip-download \
      -o "$CAP_DIR/%(id)s.%(ext)s" "https://youtu.be/$vid" >/dev/null 2>&1 || true
    pulled=$((pulled+1))
  done < "$DATA_DIR/$SOURCE/all-ids.txt"
  echo "  pulled up to $pulled new caption files"

  # 3. embed + upsert (hash-skip => only new videos cost anything)
  "$PYBIN" "$INGEST" --captions "$CAP_DIR" --titles "$DATA_DIR/$SOURCE/video-list.tsv" \
    --source "$SOURCE" --channel "$HANDLE"
done
