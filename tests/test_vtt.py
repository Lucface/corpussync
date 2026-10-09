"""source: clean_vtt and clean_srt strip timing and collapse caption text."""

from __future__ import annotations

from pathlib import Path

from corpussync.vtt import clean_srt, clean_vtt

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_rolling_vtt_collapses_and_strips_markup():
    """source: a rolling auto-caption fixture collapses to clean sentences; tags, entities, and carets are stripped."""
    raw = (FIXTURES / "rolling.vtt").read_text(encoding="utf-8")
    assert clean_vtt(raw) == "Hello there friend Welcome & thanks speaker continues here"


def test_srt_cleaning_drops_cues_tags_and_duplicate_lines():
    """source: SRT cleaning removes cue numbers, timestamps, and inline tags."""
    raw = (FIXTURES / "sample.srt").read_text(encoding="utf-8")
    assert clean_srt(raw) == "Hello there. This is the second line."
