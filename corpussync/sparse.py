"""Sparse keyword vectors. Term ids are the first 4 bytes of blake2b."""

from __future__ import annotations

import hashlib
import math
import re

from qdrant_client.models import SparseVector

# Short English stopword list. One-character tokens are dropped as well.
STOPWORDS = frozenset(
    """
    a an the and or but if in on of to for with as at by from is it this that
    be are was were been being not no nor so than then too very can will just
    do does did doing have has had having you your yours we our ours they them
    their he she his her its i me my mine what which who whom when where why how
    about into over after before up down out off again further once here there
    all any each few more most other some such only own same
    """.split()
)

def split_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, including stopwords and single characters."""
    return re.findall(r"[^\W_]+", text.lower())


def keyword_tokens(text: str) -> list[str]:
    """Tokens kept in the sparse vector: drop one-character tokens and stopwords."""
    return [tok for tok in split_tokens(text) if len(tok) > 1 and tok not in STOPWORDS]


def term_id(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def sparse_vector(text: str) -> SparseVector:
    """value = 1 + ln(term frequency). Indices are sorted and unique."""
    counts: dict[str, int] = {}
    for tok in keyword_tokens(text):
        counts[tok] = counts.get(tok, 0) + 1
    acc: dict[int, float] = {}
    for tok, tf in counts.items():
        tid = term_id(tok)
        acc[tid] = acc.get(tid, 0.0) + (1.0 + math.log(tf))
    if not acc:
        return SparseVector(indices=[], values=[])
    items = sorted(acc.items())
    return SparseVector(indices=[i for i, _ in items], values=[v for _, v in items])
