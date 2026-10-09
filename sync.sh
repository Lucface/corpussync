#!/usr/bin/env bash
# sync.sh: incremental ingest of tracked YouTube channels into per-channel Qdrant corpora.
# For each channel in a channels file: refresh the video-id list, pull auto-captions
# for ids not yet on disk (yt-dlp, no API key, no transcription cost), then embed and
# upsert via corpussync.py. Hash-skip means only new videos embed.
#
# Usage:
#   ./sync.sh                       # all channels in ./channels.txt
#   ./sync.sh mychannel             # only matching source(s)
#   CHANNELS_FILE=mine.txt ./sync.sh
#
# channels file format (one per line):   source|@handle|URL
# COOKIE_JAR: a yt-dlp cookies file exported from your own account, for when
# YouTube limits requests without one.
# PER_RUN_CAP caps new caption pulls per channel per run (default 250).
set -uo pipefail
umask 077

HERE="$(cd "$(dirname "$0")" && pwd)"
INGEST="$HERE/corpussync.py"

toml_val() {
  local key file line val
  key="$1"
  file="$CONFIG_FILE"
  [ -f "$file" ] || return 0
  line=$(grep -E "^[[:space:]]*${key}[[:space:]]*=" "$file" 2>/dev/null | head -n 1 || true)
  [ -z "$line" ] && return 0
  val=${line#*=}
  val=$(printf '%s' "$val" | sed -E 's/^[[:space:]]*//; s/[[:space:]]*$//')
  case "$val" in
    \"*) val=${val#\"}; val=${val%%\"*} ;;
    \'*) val=${val#\'}; val=${val%%\'*} ;;
    *) val=${val%%#*}; val=$(printf '%s' "$val" | sed -E 's/[[:space:]]*$//') ;;
  esac
  printf '%s' "$val"
}

expand_tilde() {
  case "$1" in
    "~")
      printf '%s' "$HOME"
      ;;
    "~/"*)
      printf '%s/%s' "$HOME" "${1#"~/"}"
      ;;
    *)
      printf '%s' "$1"
      ;;
  esac
}

CONFIG_HOME=$(expand_tilde "${CORPUSSYNC_HOME:-$HOME/.corpussync}")
CONFIG_FILE="$CONFIG_HOME/corpussync.toml"

if [ -z "${CHANNELS_FILE:-}" ]; then
  _cf=$(toml_val channels_file || true)
  if [ -n "$_cf" ]; then
    case "$_cf" in
      "~"|"~/"*) CHANNELS_FILE=$(expand_tilde "$_cf") ;;
      /*) CHANNELS_FILE="$_cf" ;;
      *) CHANNELS_FILE="$HERE/$_cf" ;;
    esac
  else
    CHANNELS_FILE="$HERE/channels.txt"
  fi
fi

if [ -z "${CORPUSSYNC_DATA:-}" ]; then
  _dd=$(toml_val data_dir || true)
  if [ -n "$_dd" ]; then
    CORPUSSYNC_DATA=$(expand_tilde "$_dd")
  else
    CORPUSSYNC_DATA="$HOME/.corpussync/channels"
  fi
fi
DATA_DIR="$CORPUSSYNC_DATA"

if [ -z "${PER_RUN_CAP:-}" ]; then
  _cap=$(toml_val per_run_cap || true)
  if [ -n "$_cap" ]; then
    PER_RUN_CAP="$_cap"
  else
    PER_RUN_CAP=250
  fi
fi

CAP_RE='^[0-9]+(_[0-9]+)*$'
if [[ ! $PER_RUN_CAP =~ $CAP_RE ]]; then
  echo "per_run_cap must be a whole number (got: $PER_RUN_CAP)"
  exit 1
fi
PER_RUN_CAP=${PER_RUN_CAP//_/}

if [ -z "${PYBIN:-}" ]; then
  _py=$(toml_val pybin || true)
  if [ -n "$_py" ]; then
    PYBIN=$(expand_tilde "$_py")
  else
    PYBIN="python3"
  fi
fi

if [ -z "${COOKIE_JAR:-}" ]; then
  _cj=$(toml_val cookie_jar || true)
  if [ -n "$_cj" ]; then
    COOKIE_JAR=$(expand_tilde "$_cj")
  fi
fi

command -v yt-dlp >/dev/null || { echo "yt-dlp not found. Install it first (brew install yt-dlp)"; exit 1; }
[ -f "$CHANNELS_FILE" ] || { echo "no channels file at $CHANNELS_FILE (copy channels.example.txt)"; exit 1; }

YT_COOKIES=()
if [ -n "${COOKIE_JAR:-}" ] && [ -s "${COOKIE_JAR}" ]; then
  YT_COOKIES=(--cookies "$COOKIE_JAR")
fi

CHANNELS=()
while IFS= read -r entry; do
  [ -z "$entry" ] && continue
  CHANNELS+=("$entry")
done << CS_END
$(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$CHANNELS_FILE" || true)
CS_END

if [ "$#" -gt 0 ]; then
  _f=()
  if [ "${#CHANNELS[@]}" -gt 0 ]; then
    for want in "$@"; do
      for e in "${CHANNELS[@]}"; do
        if [ "${e%%|*}" = "$want" ]; then
          _f+=("$e")
        fi
      done
    done
  fi
  CHANNELS=()
  if [ "${#_f[@]}" -gt 0 ]; then
    for e in "${_f[@]}"; do
      CHANNELS+=("$e")
    done
  fi
fi

if [ "${#CHANNELS[@]}" -eq 0 ]; then
  echo "no matching channels in $CHANNELS_FILE"
  [ "$#" -gt 0 ] && exit 1
  echo "sync: 0 channels, 0 failed"
  exit 0
fi

NAME_RE='^[A-Za-z0-9][A-Za-z0-9_.-]{0,55}$'
failed=0
for entry in "${CHANNELS[@]}"; do
  SOURCE="${entry%%|*}"
  if [[ ! $SOURCE =~ $NAME_RE ]] || [[ $SOURCE == *..* ]]; then
    echo "  skip $SOURCE: invalid source name (letters, digits, dot, dash, underscore; no ..)"
    failed=$((failed + 1))
    continue
  fi
  rest="${entry#*|}"
  HANDLE="${rest%%|*}"
  URL="${rest##*|}"
  CAP_DIR="$DATA_DIR/$SOURCE/captions"
  NO_CAPTIONS="$DATA_DIR/$SOURCE/no-captions.txt"
  mkdir -p "$CAP_DIR"
  echo "-- $SOURCE ($HANDLE) --"

  yt-dlp ${YT_COOKIES[@]+"${YT_COOKIES[@]}"} --flat-playlist --print "%(id)s\t%(title)s" -- "$URL" \
    > "$DATA_DIR/$SOURCE/video-list.tsv" 2>/dev/null || { echo "  list refresh failed"; failed=$((failed + 1)); continue; }
  cut -f1 "$DATA_DIR/$SOURCE/video-list.tsv" > "$DATA_DIR/$SOURCE/all-ids.txt"

  SKIP_SET=$(mktemp "$DATA_DIR/$SOURCE/no-captions-skip.XXXXXX") || { failed=$((failed + 1)); continue; }
  MISS_INPUT="$NO_CAPTIONS"
  [ -f "$MISS_INPUT" ] || MISS_INPUT=/dev/null
  if ! awk -F '\t' -v now="$(date +%s)" '
    NF && $1 != "" {
      time = ($2 ~ /^[0-9]+$/) ? $2 + 0 : 0
      if (!($1 in lowest) || time < lowest[$1]) lowest[$1] = time
      if (!($1 in highest) || time > highest[$1]) highest[$1] = time
    }
    END {
      for (id in highest)
        if (highest[id] - lowest[id] >= 172800 || now - highest[id] < 172800)
          print id
    }
  ' "$MISS_INPUT" > "$SKIP_SET"; then
    rm -f "$SKIP_SET"
    failed=$((failed + 1))
    continue
  fi

  pulled=0
  while IFS= read -r vid; do
    [ -z "$vid" ] && continue
    ls "$CAP_DIR/$vid".*.vtt >/dev/null 2>&1 && continue
    if grep -Fqx -- "$vid" "$SKIP_SET"; then
      continue
    fi
    [ "$pulled" -ge "$PER_RUN_CAP" ] && break
    if yt-dlp ${YT_COOKIES[@]+"${YT_COOKIES[@]}"} --write-auto-subs --sub-langs en --skip-download \
      -o "$CAP_DIR/%(id)s.%(ext)s" -- "https://youtu.be/$vid" >/dev/null 2>&1; then
      ok=1
    else
      ok=0
    fi
    if [ "$ok" -eq 1 ] && ! ls "$CAP_DIR/$vid".*.vtt >/dev/null 2>&1; then
      printf '%s\t%s\n' "$vid" "$(date +%s)" >> "$NO_CAPTIONS"
    fi
    pulled=$((pulled + 1))
  done < "$DATA_DIR/$SOURCE/all-ids.txt"
  rm -f "$SKIP_SET"
  echo "  pulled up to $pulled new caption files"

  if ! "$PYBIN" "$INGEST" --captions="$CAP_DIR" --titles="$DATA_DIR/$SOURCE/video-list.tsv" \
    --source="$SOURCE" --channel="$HANDLE"; then
    failed=$((failed + 1))
  fi
done
echo "sync: ${#CHANNELS[@]} channels, $failed failed"
[ "$failed" -eq 0 ]
