"""Compile the validated atlas into a single JSON file with derived per-model summaries."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import __version__
from .catalog import Catalog


def _minmax(values: list[float]) -> dict[str, float] | None:
    return {"min": min(values), "max": max(values)} if values else None


def model_summaries(cat: Catalog) -> dict[str, dict[str, Any]]:
    """Derived values only; every underlying number stays traceable via the observation rows."""
    disk: dict[str, list[float]] = defaultdict(list)
    gen: dict[str, list[float]] = defaultdict(list)
    gen_consumer: dict[str, list[float]] = defaultdict(list)
    hardware = cat.entities.get("hardware", {})

    for rec in cat.observations.get("artifacts", []):
        disk[rec.data["model_id"]].append(rec.data["disk_bytes"] / 1e9)
    for rec in cat.observations.get("throughput", []):
        d = rec.data
        if "gen_tps" not in d:
            continue
        gen[d["model_id"]].append(d["gen_tps"])
        hw = hardware.get(d.get("hardware_id", ""))
        if hw and hw.data.get("consumer"):
            gen_consumer[d["model_id"]].append(d["gen_tps"])

    out = {}
    for mid in cat.entities.get("models", {}):
        out[mid] = {
            "weights_disk_gb": _minmax(disk[mid]),
            "gen_tps_all_hardware": _minmax(gen[mid]),
            "gen_tps_consumer_hardware": _minmax(gen_consumer[mid]),
        }
    return out


def build(cat: Catalog, out: Path) -> Path:
    payload = {
        "atlas_version": __version__,
        **{kind: [r.data for r in recs.values()] for kind, recs in cat.entities.items()},
        **{kind: [r.data for r in rows] for kind, rows in cat.observations.items()},
        "derived": {"models": model_summaries(cat)},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return out
