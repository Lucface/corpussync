"""source: defects 9, 20 and 22, sync runs on system bash using trusted config and caption skip state."""

from __future__ import annotations

import os
import subprocess

from tests.conftest import REPO


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
