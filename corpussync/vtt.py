"""Caption cleaning. Timing cues are stripped and never searched."""

import html
import re

_TAG = re.compile(r"<[^>]+>")           # <00:00:15.480>, <c>, </c>
_CUE_ATTR = re.compile(r"(align|position):\S+")


def clean_vtt(raw: str) -> str:
    """YouTube auto-sub .vtt -> clean prose.

    Auto-subs animate: each cue re-shows the previous line plus a few new words,
    so the same sentence appears 2-3 times. We strip headers, cue-timing lines,
    inline <timing> tags, decode HTML entities, then collapse the roll-up by
    keeping a line only when it adds new text (prefix-grow replace + containment
    skip + consecutive de-dupe).
    """
    kept: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("WEBVTT") or s.startswith("Kind:") or s.startswith("Language:"):
            continue
        if "-->" in s:
            continue
        s = _TAG.sub("", s)
        s = _CUE_ATTR.sub("", s)
        s = html.unescape(s).strip()
        # drop the speaker-change carets YouTube injects; keep the words
        s = s.lstrip("> ").strip()
        if not s:
            continue
        if kept:
            if s == kept[-1]:
                continue
            if s in kept[-1]:          # current is a substring of last -> already have it
                continue
            if kept[-1] in s:          # last grew into current -> replace with fuller line
                kept[-1] = s
                continue
        kept.append(s)
    text = " ".join(kept).replace("\x00", "")  # null bytes silently truncate SQLite TEXT
    return re.sub(r"\s+", " ", text).strip()


def clean_srt(raw: str) -> str:
    """SRT captions -> prose. Cue numbers, timestamps, tags, and entities are stripped."""
    kept: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.isdigit():
            continue
        if "-->" in s:
            continue
        s = _TAG.sub("", s)
        s = _CUE_ATTR.sub("", s)
        s = html.unescape(s).strip()
        s = s.lstrip("> ").strip()
        if not s:
            continue
        if kept and s == kept[-1]:
            continue
        kept.append(s)
    text = " ".join(kept).replace("\x00", "")
    return re.sub(r"\s+", " ", text).strip()
