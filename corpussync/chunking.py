"""Word-window chunking. Behavior matches the 0.1 chunk() function."""

from __future__ import annotations


def chunk(text: str, max_tokens: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    if len(words) <= 10:
        return []
    words_per_chunk = int(max_tokens / 1.3)  # token approx words * 1.3
    chunks: list[str] = []
    i = 0
    while i < len(words):
        window = words[i:i + words_per_chunk]
        chunks.append(" ".join(window))
        if i + words_per_chunk >= len(words):
            break
        i += words_per_chunk - overlap
    return [c for c in chunks if len(c.split()) > 10]
