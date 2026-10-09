"""Validate user names before they can reach Qdrant's filesystem paths."""

from __future__ import annotations

import re

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")


class InvalidName(ValueError):
    pass


def check_name(value: str, what: str) -> str:
    if not NAME_RE.fullmatch(value) or ".." in value:
        raise InvalidName(
            f"invalid {what} name: {value!r} "
            "(letters, digits, dot, dash and underscore; must start with a letter or digit)"
        )
    return value
