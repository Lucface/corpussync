"""Answer a question from search hits with a local chat model."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import requests

from corpussync.search import no_match_message, print_hits, print_notices, search
from corpussync.config import ollama_base_url

_SYSTEM = (
    "Answer the question using only the numbered sources. "
    "Cite sources as [n]. If the sources do not contain the answer, say so."
)


# Scripts written without spaces between words: Thai, Lao, Myanmar, Khmer, kana, CJK ideographs,
# Hangul syllables, compatibility ideographs, full-width forms, and the supplementary ideographs.
_UNSPACED = (
    (0x0E00, 0x0EFF), (0x1000, 0x109F), (0x1780, 0x17FF), (0x3040, 0x30FF), (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF), (0xAC00, 0xD7AF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF), (0x20000, 0x2FFFF),
)


def _char_tokens(ch: str) -> float:
    code = ord(ch)
    if code < 128:
        return 0.25
    if any(low <= code <= high for low, high in _UNSPACED):
        return 1.5
    return 1 / 3


def context_size(messages: list[dict]) -> int:
    """Context tokens to request: the prompt estimate plus 1024 to answer, at least 4096, as a power of two.

    The estimate is the larger of 1.5 tokens per word and a character count. The character count
    covers scripts written without spaces, such as Chinese, Japanese or Thai, and long runs of code,
    without inflating spaced scripts such as Russian or Greek.
    """
    text = "".join(message["content"] for message in messages)
    words = sum(len(message["content"].split()) for message in messages)
    estimate = max(words * 1.5, sum(_char_tokens(ch) for ch in text))
    needed = max(4096, math.ceil(estimate) + 1024)
    return 1 << (needed - 1).bit_length()


class OllamaChat:
    def __init__(self, host: str, port: int):
        self.base = ollama_base_url(SimpleNamespace(ollama_host=host, ollama_port=port))

    def complete(self, model: str, messages: list[dict]) -> str:
        num_ctx = context_size(messages)
        resp = requests.post(
            f"{self.base}/api/chat",
            json={"model": model, "messages": messages, "stream": False,
                  "options": {"num_ctx": num_ctx}},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class FakeChat:
    """Offline chat for tests. Cites [1] and does not touch the network."""

    def __init__(self):
        self.calls = 0
        self.model = None

    def complete(self, model: str, messages: list[dict]) -> str:
        self.calls += 1
        self.model = model
        return "See [1]."


def build_chat(settings):
    kind = settings.chat
    if kind == "fake":
        return FakeChat()
    if kind == "ollama":
        return OllamaChat(settings.ollama_host, settings.ollama_port)
    raise SystemExit(f"unknown chat backend: {kind}")


def _prompt(question: str, result) -> str:
    lines = [question, "", "Sources:"]
    for hit in result.results:
        lines.append(f"[{hit.n}] {hit.corpus} | {hit.title} | {hit.locator}")
        lines.append(hit.text)
    return "\n".join(lines)


def ask(ctx, question: str, corpora: list[str], k: int = 6, model: str | None = None,
        min_score: float | None = None, collections: list[str] | None = None) -> int:
    """Search, then ask the model. Search failures return 1; no match returns 2 without chat."""
    result = search(ctx, question, corpora, k=k, mode="hybrid", min_score=min_score, collections=collections)
    print_notices(result.notices)
    for corpus in result.missing:
        print(f"{corpus}: not found", file=sys.stderr)
    if not result.results:
        if result.error is not None:
            print(f"could not search: {result.error}", file=sys.stderr)
            return 1
        print(no_match_message(list(corpora) + list(collections or [])))
        return 2
    chosen = model or ctx.settings.answer_model
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _prompt(question, result)},
    ]
    try:
        text = ctx.chat.complete(chosen, messages)
    except Exception as exc:
        print(f"chat failed: {exc}", file=sys.stderr)
        return 1
    print(text)
    print()
    print_hits(result)
    return 0
