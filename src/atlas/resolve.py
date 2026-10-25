"""Map the many names a model appears under (leaderboard labels, API ids, HF repos) to atlas model ids."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from .catalog import Catalog

_SEP = re.compile(r"[\s_.:()\[\],]+")


def norm(name: str) -> str:
    """Matching key: case/separator-insensitive. Deliberately conservative: it never drops tokens."""
    s = name.strip().lower()
    s = _SEP.sub("-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-")


def strip_provider(name: str) -> str:
    """'Qwen/Qwen2.5-Coder-32B-Instruct' and 'accounts/fireworks/models/x' -> last path segment."""
    return name.strip().rstrip("/").split("/")[-1]


def keys_for(name: str) -> set[str]:
    out = {norm(name)}
    tail = strip_provider(name)
    if tail != name:
        out.add(norm(tail))
    return {k for k in out if k}


def model_keys(data: dict, include_name: bool = False) -> set[str]:
    """Strong keys (id, aliases, API ids, HF repo) must be unique across models; the display name is a weak key."""
    names: list[str] = [data["id"]] + ([data.get("name", "")] if include_name else [])
    names += data.get("aliases") or []
    names += data.get("api_ids") or []
    if data.get("hf_repo"):
        names.append(data["hf_repo"])
    keys: set[str] = set()
    for n in names:
        if n:
            keys |= keys_for(str(n))
    return keys


class Resolver:
    def __init__(self, models: Iterable[dict]):
        index: dict[str, set[str]] = defaultdict(set)
        weak: dict[str, set[str]] = defaultdict(set)
        for m in models:
            for k in model_keys(m):
                index[k].add(m["id"])
            for k in keys_for(m.get("name") or ""):
                weak[k].add(m["id"])
        self.index = index
        self.weak = weak

    @classmethod
    def from_catalog(cls, cat: Catalog) -> "Resolver":
        return cls(r.data for r in cat.entities.get("models", {}).values())

    def ambiguous(self) -> dict[str, set[str]]:
        return {k: v for k, v in self.index.items() if len(v) > 1}

    def resolve(self, name: str) -> str | None:
        # Full name first (keeps 'org/x' distinct when both exist), then the provider-stripped tail.
        keys = (norm(name), norm(strip_provider(name)))
        for table in (self.index, self.weak):
            for key in keys:
                ids = table.get(key)
                if ids and len(ids) == 1:
                    return next(iter(ids))
            if any(table.get(k) for k in keys):
                return None  # ambiguous at this strength; do not fall through to weaker matches
        return None
