"""Shared helpers for importers.

An importer:
  1. downloads (cached) published data,
  2. declares the citation(s) it uses (`write_source`) and any harness/benchmark entities it needs (`ensure_*`),
  3. emits score rows keyed by the *raw* model name used in the source (`Importer.score`), plus
     what the source says about that model (`Importer.claim`: org, release date, open/closed, HF repo, ...),
  4. `finish()` writes data/observations/scores/<name>.csv for rows whose model resolves, and stages the
     claims so `atlas register` can create records for models that are not in the atlas yet.
Importers are re-runnable; their output files are fully regenerated each run.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from ..catalog import REPO_ROOT, load, load_schema
from ..resolve import Resolver

DATA = REPO_ROOT / "data"
CACHE = REPO_ROOT / ".cache"
HTTP_CACHE = CACHE / "http"
STAGING = CACHE / "staging"
REPORTS = CACHE / "reports"
TODAY = dt.date.today().isoformat()

SCORE_FIELDS = list(load_schema("score.schema.json")["properties"])
CLAIM_FIELDS = [
    "raw_name", "name", "org", "country", "release_date", "open_weights", "hf_repo",
    "params_total_b", "license", "context_length", "source_id", "importer",
]

USER_AGENT = "local-llm-atlas/0.1 (+https://github.com/seba2390/local-llm-atlas)"


# ---------------------------------------------------------------- HTTP

def fetch(url: str, *, headers: dict[str, str] | None = None, refresh: bool | None = None, retries: int = 4) -> bytes:
    """GET with an on-disk cache (.cache/http). Set ATLAS_REFRESH=1 to bypass the cache."""
    if refresh is None:
        refresh = os.environ.get("ATLAS_REFRESH") == "1"
    HTTP_CACHE.mkdir(parents=True, exist_ok=True)
    path = HTTP_CACHE / hashlib.sha256(url.encode()).hexdigest()
    if path.exists() and not refresh:
        return path.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()
            path.write_bytes(body)
            return body
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(5 * 2**attempt)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(5 * 2**attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def fetch_json(url: str, **kw: Any) -> Any:
    return json.loads(fetch(url, **kw))


def fetch_text(url: str, **kw: Any) -> str:
    return fetch(url, **kw).decode("utf-8")


# ---------------------------------------------------------------- entities

def slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9.]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-.")
    return s


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120))


def write_source(sid: str, **fields: Any) -> str:
    """Create/refresh a source owned by an importer. `accessed` only changes when the content changes."""
    path = DATA / "sources" / f"{sid}.yaml"
    data: dict[str, Any] = {"id": sid, **{k: v for k, v in fields.items() if v not in (None, "", [])}}
    data.setdefault("accessed", TODAY)
    if path.exists():
        old = yaml.safe_load(path.read_text()) or {}
        old_cmp = {k: str(v) for k, v in old.items() if k != "accessed"}
        new_cmp = {k: str(v) for k, v in data.items() if k != "accessed"}
        if old_cmp == new_cmp:
            return sid
    dump_yaml(path, data)
    return sid


def ensure_entity(kind: str, eid: str, data: dict[str, Any]) -> str:
    """Write data/<kind>/<eid>.yaml only if it does not exist (hand edits always win)."""
    path = DATA / kind / f"{eid}.yaml"
    if not path.exists():
        dump_yaml(path, {"id": eid, **{k: v for k, v in data.items() if v not in (None, "", [])}})
    return eid


def ensure_harness(hid: str, name: str, kind: str, url: str | None = None, sources: list[str] | None = None,
                   notes: str | None = None) -> str:
    return ensure_entity("harnesses", hid, {"name": name, "kind": kind, "url": url, "sources": sources, "notes": notes})


def ensure_benchmark(bid: str, **fields: Any) -> str:
    return ensure_entity("benchmarks", bid, fields)


# ---------------------------------------------------------------- importer

def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6g}" if abs(v) < 1e15 else repr(v)
    return str(v).strip()


class Importer:
    def __init__(self, name: str):
        self.name = name
        self.resolver = Resolver.from_catalog(load(DATA))
        self.rows: list[dict[str, Any]] = []
        self.claims: dict[str, dict[str, Any]] = {}
        self.unresolved: Counter[str] = Counter()
        self.skipped: Counter[str] = Counter()

    def claim(self, raw_name: str, **meta: Any) -> None:
        raw_name = raw_name.strip()
        cur = self.claims.setdefault(raw_name, {"raw_name": raw_name, "importer": self.name})
        for k, v in meta.items():
            if v not in (None, "") and not cur.get(k):
                cur[k] = v

    def score(self, raw_name: str, **row: Any) -> bool:
        raw_name = raw_name.strip()
        if not raw_name:
            self.skipped["empty model name"] += 1
            return False
        value = row.get("value")
        if value is None or (isinstance(value, float) and value != value):
            self.skipped["missing value"] += 1
            return False
        mid = self.resolver.resolve(raw_name)
        if mid is None:
            self.unresolved[raw_name] += 1
            return False
        unknown = set(row) - set(SCORE_FIELDS)
        if unknown:
            raise ValueError(f"unknown score fields {unknown}")
        self.rows.append({"model_id": mid, **row})
        return True

    def finish(self) -> dict[str, int]:
        out = DATA / "observations" / "scores" / f"{self.name}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        seen: set[tuple] = set()
        rows = []
        for r in self.rows:
            key = tuple(_fmt(r.get(f)) for f in SCORE_FIELDS if f != "notes")
            if key in seen:
                self.skipped["exact duplicate"] += 1
                continue
            seen.add(key)
            rows.append(r)
        sort_fields = ("model_id", "benchmark_id", "variant", "metric", "harness_id", "reasoning_effort", "eval_date", "locator")
        rows.sort(key=lambda r: tuple(_fmt(r.get(f)) for f in sort_fields))
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SCORE_FIELDS, lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({f: _fmt(r.get(f)) for f in SCORE_FIELDS})

        STAGING.mkdir(parents=True, exist_ok=True)
        with (STAGING / f"{self.name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CLAIM_FIELDS, lineterminator="\n")
            w.writeheader()
            for c in self.claims.values():
                w.writerow({f: _fmt(c.get(f)) for f in CLAIM_FIELDS})

        REPORTS.mkdir(parents=True, exist_ok=True)
        with (REPORTS / f"{self.name}-unresolved.csv").open("w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["raw_name", "rows"])
            w.writerows(self.unresolved.most_common())

        stats = {
            "rows_written": len(rows),
            "models": len({r["model_id"] for r in rows}),
            "unresolved_names": len(self.unresolved),
            "unresolved_rows": sum(self.unresolved.values()),
            **{f"skipped: {k}": v for k, v in self.skipped.items()},
        }
        print(f"[{self.name}] " + ", ".join(f"{k}={v}" for k, v in stats.items()))
        return stats
