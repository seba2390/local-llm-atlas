"""Validate the atlas: schemas, referential integrity, citations and sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from .catalog import ENTITIES, OBSERVATIONS, Catalog, Record, load_schema

PERCENT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.level.upper():7} {self.where}: {self.message}"


def _get_path(data: dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def validate(cat: Catalog) -> list[Issue]:
    issues: list[Issue] = []
    err = lambda where, msg: issues.append(Issue("error", where, msg))  # noqa: E731
    warn = lambda where, msg: issues.append(Issue("warning", where, msg))  # noqa: E731

    for where, msg in cat.load_errors:
        err(where, msg)
    for kind, rid, where in cat.duplicate_ids:
        err(where, f"duplicate {kind} id '{rid}'")

    # 1. JSON Schema
    for kind, (_, schema_name) in ENTITIES.items():
        validator = Draft202012Validator(load_schema(schema_name))
        for rec in cat.entities.get(kind, {}).values():
            for e in validator.iter_errors(rec.data):
                err(rec.where, f"schema: {'/'.join(map(str, e.path)) or '<root>'}: {e.message}")
    for kind, (_, schema_name) in OBSERVATIONS.items():
        validator = Draft202012Validator(load_schema(schema_name))
        for rec in cat.observations.get(kind, []):
            if "__extra__" in rec.data:
                err(rec.where, "row has more cells than the header")
                continue
            for e in validator.iter_errors(rec.data):
                err(rec.where, f"schema: {'/'.join(map(str, e.path)) or '<root>'}: {e.message}")

    sources = cat.entities.get("sources", {})
    models = cat.entities.get("models", {})
    benchmarks = cat.entities.get("benchmarks", {})
    hardware = cat.entities.get("hardware", {})
    runtimes = cat.entities.get("runtimes", {})
    harnesses = cat.entities.get("harnesses", {})
    cited: set[str] = set()

    def check_source(rec: Record, sid: Any) -> None:
        if not isinstance(sid, str):
            return
        cited.add(sid)
        if sid not in sources:
            err(rec.where, f"unknown source '{sid}'")

    def check_ref(rec: Record, table: dict[str, Record], kind: str, rid: Any) -> Record | None:
        if not isinstance(rid, str):
            return None
        if rid not in table:
            err(rec.where, f"unknown {kind} '{rid}'")
            return None
        return table[rid]

    # 2. File layout and entity cross-references
    for kind, records in cat.entities.items():
        for rid, rec in records.items():
            if rec.path.stem != rid:
                err(rec.where, f"file name must be '{rid}.yaml'")
            for sid in rec.data.get("sources", []) or []:
                check_source(rec, sid)

    for rid, rec in models.items():
        d = rec.data
        if rec.path.parent.name != d.get("org"):
            err(rec.where, f"model files live in data/models/<org>/; expected folder '{d.get('org')}'")
        for dotted, sid in (d.get("field_sources") or {}).items():
            if _get_path(d, dotted) is None:
                err(rec.where, f"field_sources refers to missing field '{dotted}'")
            check_source(rec, sid)
        if d.get("open_weights") is False and d.get("weights_url"):
            err(rec.where, "closed-weight model must not have weights_url")
        if d.get("base_model"):
            check_ref(rec, models, "model (base_model)", d["base_model"])
        arch = d.get("architecture") or {}
        total, active = arch.get("params_total_b"), arch.get("params_active_b")
        if isinstance(total, (int, float)) and isinstance(active, (int, float)) and active > total:
            err(rec.where, "params_active_b exceeds params_total_b")
        if "moe" in str(arch.get("type", "")):
            for key in ("params_active_b", "experts_total", "experts_active"):
                if key not in arch:
                    warn(rec.where, f"MoE model is missing architecture.{key}")

    # 3. Observations
    seen_scores: set[tuple] = set()
    for rec in cat.observations.get("scores", []):
        d = rec.data
        check_source(rec, d.get("source_id"))
        check_ref(rec, models, "model", d.get("model_id"))
        harness = check_ref(rec, harnesses, "harness", d.get("harness_id"))
        bench = check_ref(rec, benchmarks, "benchmark", d.get("benchmark_id"))
        if bench is None:
            continue
        b = bench.data
        if b.get("harness_sensitive") and harness and harness.data.get("kind") == "undisclosed":
            warn(rec.where, f"benchmark '{b['id']}' is harness-sensitive but the harness is undisclosed")
        variants = {v["id"]: v for v in b.get("variants", []) or []}
        variant = d.get("variant")
        if variants and variant not in variants:
            err(rec.where, f"variant '{variant}' not in {sorted(variants)} for benchmark '{b['id']}'")
        elif not variants and variant:
            err(rec.where, f"benchmark '{b['id']}' declares no variants but row has variant '{variant}'")
        value = d.get("value")
        if b.get("unit") == "percent" and isinstance(value, (int, float)) and not (
            -PERCENT_TOLERANCE <= value <= 100 + PERCENT_TOLERANCE
        ):
            err(rec.where, f"percent value {value} outside [0, 100]")
        # A score dated before the benchmark existed almost always means the wrong benchmark was cited.
        introduced = (variants.get(variant) or {}).get("introduced") or b.get("introduced")
        src = sources.get(d.get("source_id") or "")
        dated = d.get("eval_date") or (src.data.get("published") if src else None)
        if introduced and dated and str(dated) < str(introduced)[: len(str(dated))]:
            err(rec.where, f"score dated {dated} predates benchmark '{b['id']}' ({introduced})")
        key_fields = (
            "model_id", "benchmark_id", "variant", "metric", "harness_id", "harness_version",
            "setting", "reasoning", "reasoning_effort", "quant", "source_id",
        )
        key = tuple(d.get(k) for k in key_fields)
        if key in seen_scores:
            err(rec.where, f"duplicate score row (same {'/'.join(key_fields)})")
        seen_scores.add(key)

    for kind in ("artifacts", "throughput", "memory"):
        for rec in cat.observations.get(kind, []):
            d = rec.data
            check_source(rec, d.get("source_id"))
            check_ref(rec, models, "model", d.get("model_id"))
            if "runtime_id" in d:
                check_ref(rec, runtimes, "runtime", d["runtime_id"])
            if "hardware_id" in d:
                check_ref(rec, hardware, "hardware", d["hardware_id"])

    for sid, rec in sources.items():
        if sid not in cited:
            warn(rec.where, "source is never cited")

    return issues


def summarize(issues: Iterable[Issue]) -> tuple[int, int]:
    issues = list(issues)
    return (
        sum(i.level == "error" for i in issues),
        sum(i.level == "warning" for i in issues),
    )
