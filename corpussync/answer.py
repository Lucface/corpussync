"""Answer a question from search hits with a local chat model."""

import sys

import requests

from corpussync.search import no_match_message, print_hits, print_notices, search

_SYSTEM = (
    "Answer the question using only the numbered sources. "
    "Cite sources as [n]. If the sources do not contain the answer, say so."
)


class OllamaChat:
    def __init__(self, host: str, port: int):
        self.base = f"http://{host}:{port}"

    def complete(self, model: str, messages: list[dict]) -> str:
        resp = requests.post(
            f"{self.base}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=120,
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


def ask(ctx, question: str, corpora: list[str], k: int = 6, model: str | None = None) -> int:
    """Search, then ask the chat model. Returns 2 without calling the model when nothing clears the floor."""
    result = search(ctx, question, corpora, k=k, mode="hybrid")
    print_notices(result.notices)
    for corpus in result.missing:
        print(f"{corpus}-corpus: not found", file=sys.stderr)
    if not result.results:
        print(no_match_message(list(corpora)))
        return 2
    chosen = model or ctx.settings.answer_model
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _prompt(question, result)},
    ]
    try:
        text = ctx.chat.complete(chosen, messages)
    except Exception as exc:
        print(f"chat failed: {exc}")
        return 1
    print(text)
    print()
    print_hits(result)
    return 0
