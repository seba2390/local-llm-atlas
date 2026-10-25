"""Load the atlas data directory into memory."""

from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "schema"

# kind -> (subdirectory of data/, schema file)
ENTITIES: dict[str, tuple[str, str]] = {
    "sources": ("sources", "source.schema.json"),
    "models": ("models", "model.schema.json"),
    "benchmarks": ("benchmarks", "benchmark.schema.json"),
    "hardware": ("hardware", "hardware.schema.json"),
    "runtimes": ("runtimes", "runtime.schema.json"),
    "harnesses": ("harnesses", "harness.schema.json"),
}

# kind -> (file in data/observations/, schema file)
OBSERVATIONS: dict[str, tuple[str, str]] = {
    "scores": ("scores.csv", "score.schema.json"),
    "artifacts": ("artifacts.csv", "artifact.schema.json"),
    "throughput": ("throughput.csv", "throughput.schema.json"),
    "memory": ("memory.csv", "memory.schema.json"),
}


@dataclass
class Record:
    data: dict[str, Any]
    path: Path
    line: int | None = None

    @property
    def where(self) -> str:
        return f"{self.path}:{self.line}" if self.line else str(self.path)


@dataclass
class Catalog:
    root: Path
    entities: dict[str, dict[str, Record]] = field(default_factory=dict)
    observations: dict[str, list[Record]] = field(default_factory=dict)
    load_errors: list[tuple[str, str]] = field(default_factory=list)
    duplicate_ids: list[tuple[str, str, str]] = field(default_factory=list)


def load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text())


def _stringify_dates(obj: Any) -> Any:
    # YAML turns 2024-11-12 into a date object; the schemas expect ISO strings.
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()[:10]
    if isinstance(obj, dict):
        return {k: _stringify_dates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_dates(v) for v in obj]
    return obj


def _coerce_row(row: dict[str, str], schema: dict[str, Any]) -> dict[str, Any]:
    """Drop empty cells and convert numeric columns according to the schema."""
    props = schema.get("properties", {})
    out: dict[str, Any] = {}
    for key, raw in row.items():
        if key is None:
            out.setdefault("__extra__", raw)
            continue
        value = (raw or "").strip()
        if value == "":
            continue
        typ = props.get(key, {}).get("type")
        try:
            if typ == "integer":
                out[key] = int(value)
            elif typ == "number":
                out[key] = float(value)
            else:
                out[key] = value
        except ValueError:
            out[key] = value  # leave as string; schema validation will report it
    return out


def load(root: Path | str) -> Catalog:
    root = Path(root)
    cat = Catalog(root=root)

    for kind, (subdir, _) in ENTITIES.items():
        records: dict[str, Record] = {}
        for path in sorted((root / subdir).rglob("*.yaml")):
            try:
                data = _stringify_dates(yaml.safe_load(path.read_text()))
            except yaml.YAMLError as exc:
                cat.load_errors.append((str(path), f"YAML parse error: {exc}"))
                continue
            if not isinstance(data, dict) or "id" not in data:
                cat.load_errors.append((str(path), "expected a mapping with an 'id' key"))
                continue
            rid = str(data["id"])
            if rid in records:
                cat.duplicate_ids.append((kind, rid, str(path)))
                continue
            records[rid] = Record(data=data, path=path)
        cat.entities[kind] = records

    for kind, (filename, schema_name) in OBSERVATIONS.items():
        schema = load_schema(schema_name)
        rows: list[Record] = []
        path = root / "observations" / filename
        if path.exists():
            with path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                for i, row in enumerate(reader, start=2):
                    rows.append(Record(data=_coerce_row(row, schema), path=path, line=i))
        cat.observations[kind] = rows

    return cat
