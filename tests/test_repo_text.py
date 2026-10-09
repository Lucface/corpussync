"""source: repository text follows the public-repo wording rules."""

from __future__ import annotations

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


def test_modules_defer_annotations_and_parse_on_python39():
    """source: every nonempty module defers annotations and parses on Python 3.9; tests has an empty package marker."""
    import ast

    paths = list((REPO / "corpussync").glob("*.py")) + list((REPO / "tests").glob("*.py")) + [REPO / "corpussync.py"]
    for path in paths:
        if path == REPO / "tests" / "__init__.py":
            assert path.read_bytes() == b""
            continue
        tree = ast.parse(path.read_text(), feature_version=(3, 9))
        statements = tree.body
        if statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant) and isinstance(statements[0].value.value, str):
            statements = statements[1:]
        assert statements, str(path)
        first = statements[0]
        assert isinstance(first, ast.ImportFrom) and first.module == "__future__", str(path)
        assert any(alias.name == "annotations" for alias in first.names), str(path)


def test_setup_metadata_and_legacy_editable_entry():
    """source: metadata resolves the package version and declares TOML and dev dependencies for Python 3.9."""
    import configparser

    config = configparser.ConfigParser()
    config.read(REPO / "setup.cfg")
    assert config["metadata"]["name"] == "corpussync"
    assert config["metadata"]["version"] == "attr: corpussync.__init__.__version__"
    assert config["options"]["packages"] == "corpussync"
    assert config["options"]["python_requires"] == ">=3.9"
    assert config["options"]["install_requires"].strip().splitlines() == [
        "qdrant-client>=1.10", "requests>=2.31", 'tomli>=1.1; python_version < "3.11"',
    ]
    assert config["options.entry_points"]["console_scripts"].strip() == "corpussync = corpussync.cli:main"
    assert {key: value.split() for key, value in config["options.extras_require"].items()} == {
        "pdf": ["pypdf>=4"], "docx": ["python-docx>=1.1"], "dev": ["pytest>=8", "setuptools>=61"],
    }
    assert (REPO / "setup.py").read_text() == "from setuptools import setup\nsetup()\n"


def test_requirements_match_packaged_runtime_dependencies():
    """source: round 5 item 15, requirements installs need the same TOML dependency as the package."""
    import configparser

    config = configparser.ConfigParser()
    config.read(REPO / "setup.cfg")
    assert (REPO / "requirements.txt").read_text().splitlines() == [
        line.strip() for line in config["options"]["install_requires"].strip().splitlines()
    ]
