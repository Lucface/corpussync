"""source: defect 1, user names never become unchecked filesystem paths."""

from __future__ import annotations

import subprocess
import sys

import pytest

from corpussync.names import InvalidName, check_name
from tests.conftest import REPO, isolated_env, run_cli


@pytest.mark.parametrize("name", ["../x", "../../x", "a..b", "/tmp/x", "-bad", "", "a" * 64, "notes\n"])
def test_invalid_names(name):
    """source: defect 1, traversal and names outside the specified grammar are rejected."""
    with pytest.raises(InvalidName, match="invalid corpus name"):
        check_name(name, "corpus")


@pytest.mark.parametrize("name", ["notes", "A_1.2-x", "a" * 63])
def test_valid_names(name):
    """source: defect 1, safe names within the 63 character limit are accepted."""
    assert check_name(name, "corpus") == name


@pytest.mark.parametrize("args", [
    ["ingest", ".", "--corpus", "../x"],
    ["remove", "--corpus", "../x", "--yes"],
    ["remove", "--collection", "../x", "--yes"],
    ["search", "question", "--corpus", "../x"],
    ["search", "question", "--collection", "../x"],
    ["ask", "question", "--corpus", "../x"],
    ["ask", "question", "--collection", "../x"],
    ["stats", "--corpus", "../x"],
    ["stats", "--collection", "../x"],
    ["youtube", "--captions", ".", "--source", "../x"],
    ["youtube", "--captions", ".", "--source", "notes", "--collection", "../x"],
])
def test_cli_rejects_names_before_store_access(tmp_path, args):
    """source: defect 1, every CLI name is checked before creating or deleting any store path."""
    outside = tmp_path / "x"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("private text")
    home = tmp_path / "home"
    before = sorted(str(path) for path in tmp_path.rglob("*"))
    proc = run_cli(args, home, tmp_path)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "invalid" in proc.stderr and "name: '../x'" in proc.stderr
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before
    assert sentinel.read_text() == "private text"


@pytest.mark.parametrize("flag", ["--source", "--collection"])
def test_compat_rejects_names_before_probe(tmp_path, flag):
    """source: defect 1, compat validates source and collection before probing or opening a store."""
    home = tmp_path / "home"
    args = ["--source", "notes", "--stats", flag, "../x"]
    before = list(tmp_path.rglob("*"))
    proc = subprocess.run([sys.executable, str(REPO / "corpussync.py"), *args],
                          env=isolated_env(home), cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 2
    assert "name: '../x'" in proc.stderr
    assert list(tmp_path.rglob("*")) == before
