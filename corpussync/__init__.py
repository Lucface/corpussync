"""Build a searchable corpus from notes, documents, transcripts, and captions."""

from __future__ import annotations

__version__ = "0.2.0"

from corpussync.chunking import chunk
from corpussync.vtt import clean_vtt

__all__ = ["__version__", "chunk", "clean_vtt"]
