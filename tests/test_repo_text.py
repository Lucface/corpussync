"""source: repository text follows the public-repo wording rules."""

import re
import subprocess

from tests.conftest import REPO


def _tracked_files():
    out = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=str(REPO),
        text=True,
    )
    for line in out.splitlines():
        path = REPO / line
        if path.is_file():
            yield line, path


def test_no_em_dash_en_dash_or_home_prefix():
    """source: tracked and new text files have no em dash, en dash, or absolute home path."""
    em = chr(0x2014)
    en = chr(0x2013)
    home = "/" + "Users" + "/"
    bad = []
    for rel, path in _tracked_files():
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            bad.append(rel + " (not utf-8)")
            continue
        if em in text or en in text or home in text:
            bad.append(rel)
    assert bad == []


def test_readme_avoids_banned_words():
    """source: README.md does not use the words actually or real."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for word in ("actu" + "ally", "re" + "al"):
        assert re.search(r"\b" + word + r"\b", readme, flags=re.IGNORECASE) is None


def test_channels_example_has_only_placeholder_handles():
    """source: channels.example.txt has only placeholder handles."""
    text = (REPO / "channels.example.txt").read_text(encoding="utf-8")
    handles = re.findall(r"@[A-Za-z0-9_]+", text)
    assert handles
    assert set(handles) <= {"@YourChannel", "@handle"}
    assert "@YourChannel" in handles
