"""source: package metadata must resolve without importing the root compatibility shim."""

from __future__ import annotations

import pytest

import corpussync
from tests.conftest import REPO


def test_setuptools_reads_package_metadata():
    """source: setuptools read the version from the root compatibility shim, and pip install -e . failed"""
    read_configuration = pytest.importorskip("setuptools.config.setupcfg").read_configuration

    config = read_configuration(str(REPO / "setup.cfg"))
    assert config["metadata"]["version"] == corpussync.__version__
    options = config["options"]
    assert "corpussync = corpussync.cli:main" in options["entry_points"]["console_scripts"]
    assert options["python_requires"] == ">=3.9"
    assert "corpussync.py" not in options["packages"]
