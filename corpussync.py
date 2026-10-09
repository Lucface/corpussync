#!/usr/bin/env python3
"""Compatibility entry point for the 0.1 caption ingest flags.

Prefer `corpussync` or `python -m corpussync` for notes, search, and answers.
This file keeps the original flags and re-exports clean_vtt and chunk:

  python3 corpussync.py --captions DIR --titles video-list.tsv \\
      --source mychannel --channel "@YourChannel"
  python3 corpussync.py --source mychannel --stats
"""

from __future__ import annotations

from corpussync.chunking import chunk
from corpussync.cli import main_compat
from corpussync.vtt import clean_vtt

__all__ = ["clean_vtt", "chunk", "main_compat"]

if __name__ == "__main__":
    raise SystemExit(main_compat())
