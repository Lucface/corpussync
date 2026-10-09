"""source: defects 9, 20 and 22, sync runs on system bash using trusted config and caption skip state."""

from __future__ import annotations

import os
import subprocess

import pytest

from tests.conftest import REPO


def _sync_env(tmp_path, channels, ids=1):
    home = tmp_path / "home"
    home.mkdir()
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    (tmp_path / "channels.txt").write_text(channels)
    (tmp_path / "ids.tsv").write_text("".join(f"video{i}\tTitle {i}\n" for i in range(ids)))
    yt = stub_dir / "yt-dlp"
    yt.write_text('''#!/bin/bash
set -eu
printf '%s\\t' "$@" >> "$YT_LOG"
printf '\\n' >> "$YT_LOG"
if [ "$1" = --flat-playlist ]; then
  for arg in "$@"; do url="$arg"; done
  [ "$url" != "$YT_FAIL_URL" ] || exit 1
  cat "$YT_IDS"
fi
''')
    yt.chmod(0o700)
    pybin = stub_dir / "stub-python"
    pybin.write_text('''#!/bin/bash
set -eu
printf '%s\\t' "$@" >> "$PY_LOG"
printf '\\n' >> "$PY_LOG"
source=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --source) shift; source="$1" ;;
  esac
  shift
done
[ "$source" != "$PY_FAIL_SOURCE" ] || exit 3
''')
    pybin.chmod(0o700)
    env = os.environ.copy()
    env.update(
        HOME=str(home), CORPUSSYNC_HOME=str(home / ".corpussync"),
        PATH=str(stub_dir) + ":/usr/bin:/bin", COOKIE_JAR="", PER_RUN_CAP="1",
        CHANNELS_FILE=str(tmp_path / "channels.txt"), CORPUSSYNC_DATA=str(tmp_path / "data"),
        PYBIN=str(pybin), YT_LOG=str(tmp_path / "yt.log"), PY_LOG=str(tmp_path / "py.log"),
        YT_IDS=str(tmp_path / "ids.tsv"), YT_FAIL_URL="", PY_FAIL_SOURCE="",
    )
    return env


@pytest.mark.parametrize("source", ["../x", "bad..name", ".hidden", "-option", "bad name", "x" * 64])
def test_sync_invalid_source_creates_nothing_and_continues(tmp_path, source):
    """source: round 4 item 6, invalid sources cannot create paths and count as failed without stopping sync."""
    env = _sync_env(
        tmp_path,
        f"{source}|@x|https://example.invalid/x\n"
        "mychannel|@YourChannel|https://example.invalid/channel\n",
    )
    proc = subprocess.run(["/bin/bash", str(REPO / "sync.sh")], cwd=tmp_path,
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert not (tmp_path / "data" / ".." / "x").exists()
    assert sorted(path.name for path in (tmp_path / "data").iterdir()) == ["mychannel"]
    assert (
        f"  skip {source}: invalid source name (letters, digits, dot, dash, underscore; no ..)"
    ) in proc.stdout.splitlines()
    calls = [line.split("\t")[:-1] for line in (tmp_path / "py.log").read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0][calls[0].index("--source") + 1] == "mychannel"
    assert proc.stdout.splitlines()[-1] == "sync: 2 channels, 1 failed"


def test_sync_both_yt_calls_end_options_before_url(tmp_path):
    """source: round 4 item 6, both yt-dlp calls must separate options from their URL arguments."""
    env = _sync_env(tmp_path, "mychannel|@YourChannel|--version\n")
    proc = subprocess.run(["/bin/bash", str(REPO / "sync.sh")], cwd=tmp_path,
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = [line.split("\t")[:-1] for line in (tmp_path / "yt.log").read_text().splitlines()]
    listed = [args for args in calls if "--flat-playlist" in args]
    pulled = [args for args in calls if "--write-auto-subs" in args]
    assert len(listed) == len(pulled) == 1
    assert listed[0][-2:] == ["--", "--version"]
    assert pulled[0][-2:] == ["--", "https://youtu.be/video0"]


def test_sync_cap_accepts_toml_underscores(tmp_path):
    """source: round 4 item 6, a digit cap with TOML underscores limits the number of caption pulls."""
    env = _sync_env(tmp_path, "mychannel|@YourChannel|https://example.invalid/channel\n", ids=12)
    env["PER_RUN_CAP"] = "1_0"
    proc = subprocess.run(["/bin/bash", str(REPO / "sync.sh")], cwd=tmp_path,
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = [line.split("\t")[:-1] for line in (tmp_path / "yt.log").read_text().splitlines()]
    pulls = [args for args in calls if "--write-auto-subs" in args]
    assert len(pulls) == 10
    assert [args[-1] for args in pulls] == [f"https://youtu.be/video{i}" for i in range(10)]
    assert proc.stdout.splitlines()[-1] == "sync: 1 channels, 0 failed"


def test_sync_invalid_cap_exits_before_yt_dlp(tmp_path):
    """source: round 4 item 6, an invalid caption cap fails before any yt-dlp call."""
    env = _sync_env(tmp_path, "mychannel|@YourChannel|https://example.invalid/channel\n")
    env["PER_RUN_CAP"] = "abc"
    proc = subprocess.run(["/bin/bash", str(REPO / "sync.sh")], cwd=tmp_path,
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "per_run_cap must be a whole number (got: abc)" in proc.stdout.splitlines()
    assert not (tmp_path / "yt.log").exists()
    assert not (tmp_path / "py.log").exists()


@pytest.mark.parametrize("failure", ["list", "ingest"])
def test_sync_channel_failure_counts_and_continues(tmp_path, failure):
    """source: round 4 item 6, a failed list refresh or ingest counts once and later channels still ingest."""
    env = _sync_env(
        tmp_path,
        "first|@YourChannel|https://example.invalid/first\n"
        "second|@YourChannel|https://example.invalid/second\n",
    )
    if failure == "list":
        env["YT_FAIL_URL"] = "https://example.invalid/first"
    else:
        env["PY_FAIL_SOURCE"] = "first"
    proc = subprocess.run(["/bin/bash", str(REPO / "sync.sh")], cwd=tmp_path,
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout.splitlines()[-1] == "sync: 2 channels, 1 failed"
    calls = [line.split("\t")[:-1] for line in (tmp_path / "py.log").read_text().splitlines()]
    sources = [args[args.index("--source") + 1] for args in calls]
    assert sources == (["second"] if failure == "list" else ["first", "second"])
    if failure == "list":
        assert "  list refresh failed" in proc.stdout.splitlines()


def test_sync_unmatched_requested_channel_exits_one(tmp_path):
    """source: round 4 item 6, explicitly requested names that match no channels must fail."""
    env = _sync_env(tmp_path, "mychannel|@YourChannel|https://example.invalid/channel\n")
    proc = subprocess.run(["/bin/bash", str(REPO / "sync.sh"), "nosuch"], cwd=tmp_path,
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "no matching channels in " + env["CHANNELS_FILE"] in proc.stdout.splitlines()
    assert not (tmp_path / "yt.log").exists()
    assert not (tmp_path / "py.log").exists()


def test_sync_comments_only_channels_reports_no_work(tmp_path):
    """source: round 4 item 6, no channel entries and no requested names report zero work and succeed."""
    env = _sync_env(tmp_path, "# Placeholder channels\n  # Add a channel to sync\n")
    proc = subprocess.run(["/bin/bash", str(REPO / "sync.sh")], cwd=tmp_path,
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.splitlines()[-1] == "sync: 0 channels, 0 failed"
    assert not (tmp_path / "yt.log").exists()
    assert not (tmp_path / "py.log").exists()


def test_sync_tilde_comments_cap_and_no_caption_skip(tmp_path):
    """source: defects 9, 20, 22 and round 3 item 5, caption-less videos skip while failed downloads retry."""
    home = tmp_path / "user"
    home.mkdir()
    config_home = home / ".corpussync"
    config_home.mkdir()
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    yt_log = tmp_path / "yt.log"
    py_log = tmp_path / "py.log"
    (home / "channels.txt").write_text("mychannel|@YourChannel|https://example.test/channel\n")
    (home / "cookies.txt").write_text("synthetic cookies")
    (config_home / "corpussync.toml").write_text(
        'channels_file = "~/channels.txt" # ignored comment\n'
        "data_dir = '~/data # kept' # ignored comment\n"
        'per_run_cap = 1 # trailing comment\n'
        'cookie_jar = "~/cookies.txt" # trailing comment\n'
        'pybin = "~/stub-python" # trailing comment\n'
    )
    (tmp_path / "corpussync.toml").write_text('pybin = "/does/not/exist"\n')
    yt = stub_dir / "yt-dlp"
    yt.write_text('''#!/bin/bash
set -eu
printf '%s\\n' "$*" >> "$YT_LOG"
output=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --flat-playlist)
      printf 'empty\\tNo captions\\nfirst\\tFirst\\nflaky\\tRetry\\nsecond\\tSecond\\n'
      exit 0 ;;
    -o) shift; output="$1" ;;
  esac
  last="$1"
  shift
done
vid=${last##*/}
if [ "$vid" = flaky ] && [ ! -f "$YT_LOG.failed" ]; then
  touch "$YT_LOG.failed"
  exit 1
fi
if [ "$vid" != empty ]; then
  output=${output//%(id)s/$vid}
  output=${output//%(ext)s/en.vtt}
  printf 'WEBVTT\\n' > "$output"
fi
''')
    yt.chmod(0o700)
    pybin = home / "stub-python"
    pybin.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$PY_LOG"\n')
    pybin.chmod(0o700)
    env = os.environ.copy()
    for key in ("CHANNELS_FILE", "CORPUSSYNC_DATA", "PER_RUN_CAP", "PYBIN", "COOKIE_JAR"):
        env.pop(key, None)
    env.update(HOME=str(home), CORPUSSYNC_HOME="~/.corpussync", PATH=str(stub_dir) + ":/usr/bin:/bin",
               YT_LOG=str(yt_log), PY_LOG=str(py_log))
    outputs = []
    for _ in range(6):
        proc = subprocess.run(["/bin/bash", str(REPO / "sync.sh")], cwd=tmp_path,
                              env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        outputs.append(proc.stdout)
        skipped = home / "data # kept" / "mychannel" / "no-captions.txt"
        assert "flaky" not in skipped.read_text().splitlines()
    data = home / "data # kept" / "mychannel"
    assert (data / "no-captions.txt").read_text().splitlines() == ["empty"]
    assert sorted(path.name for path in (data / "captions").glob("*.vtt")) == ["first.en.vtt", "flaky.en.vtt", "second.en.vtt"]
    pulls = [line for line in yt_log.read_text().splitlines() if "--write-auto-subs" in line]
    assert len(pulls) == 5
    assert [line.rsplit("/", 1)[-1] for line in pulls] == ["empty", "first", "flaky", "flaky", "second"]
    assert all("--cookies " + str(home / "cookies.txt") in line for line in pulls)
    assert all("pulled up to 1" in output for output in outputs[:5])
    assert "pulled up to 0" in outputs[5]
    assert str(data / "captions") in py_log.read_text()
    assert "~/" not in py_log.read_text()
    assert (data / "no-captions.txt").stat().st_mode & 0o777 == 0o600
    assert data.stat().st_mode & 0o777 == 0o700
    env.pop("COOKIE_JAR", None)
    (config_home / "corpussync.toml").write_text(
        'channels_file="~/channels.txt"\ndata_dir="~/data # kept"\npybin="~/stub-python"\n'
    )
    proc = subprocess.run(["/bin/bash", str(REPO / "sync.sh")], cwd=tmp_path,
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "pulled up to 0" in proc.stdout
