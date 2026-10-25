# Writing an importer

Importers live in `src/atlas/ingest/<name>.py`, expose `run() -> dict`, and are listed in
`src/atlas/ingest/__init__.py` (`IMPORTERS`). Run one with `uv run atlas ingest <name>`;
`uv run atlas ingest all` runs every importer, registers new models/orgs, and runs them again.

## Contract

1. **Download** with `common.fetch` / `fetch_json` / `fetch_text` (cached in `.cache/http`;
   `ATLAS_REFRESH=1` bypasses the cache). Prefer machine-readable data (JSON/CSV/YAML in a repo or
   dataset) over scraping HTML. Pin a commit / dataset revision in the URL when the host supports it.
2. **Cite** with `common.write_source(id, kind=..., provenance=..., title=..., url=..., revision=..., published=...)`.
   `provenance`: `first_party` if the model's own developer published the number (model card, tech report,
   launch blog), `third_party` for independent leaderboards/evaluators, `community` for forums.
   This project never runs its own measurements.
3. **Entities** the importer needs:
   - `common.ensure_harness(id, name, kind, url=..., sources=[...])`: every score names the harness that
     produced it (agent scaffold, eval framework, `aider`, ...). Use `undisclosed` only if the source
     really doesn't say. Put harness options (edit format, max turns, ...) and eval settings (n-shot,
     temperature, date window, ...) in `setting` as `key=value; key=value`.
   - Benchmarks: shared benchmarks are hand-curated in `data/benchmarks/` (e.g. `humaneval`, `mbpp`,
     `swe-bench`, `terminal-bench`, `aider-polyglot`, `livecodebench`, `bigcodebench`, `gpqa`, `mmlu`,
     `math`, `ifeval`, ...). Reuse them; add a new `variants` entry there if needed. Create new benchmarks
     with `common.ensure_benchmark(...)` (or own them like `epoch.py` does with `managed_by: <importer>`).
     Versions/splits whose scores are not comparable are **variants** (e.g. `swe-bench: verified`,
     `terminal-bench: 2.0`); sub-scores of the same run are **metrics**.
4. **Emit** with an `common.Importer(name)` instance:
   - `imp.claim(raw_name, name=..., org=..., country=..., release_date=..., open_weights=..., hf_repo=...,
     params_total_b=..., license=..., context_length=..., source_id=...)` for whatever the source states
     about a model. `atlas register` turns claims of unknown models into `data/models/<org>/<id>.yaml`.
   - `imp.score(raw_name, benchmark_id=..., variant=..., metric=..., value=..., stderr=..., harness_id=...,
     harness_version=..., setting=..., reasoning=on|off, reasoning_effort=..., quant=..., eval_date=...,
     source_id=..., locator=..., original_source=..., original_url=..., notes=...)`.
     `raw_name` is the model name exactly as the source writes it. Percent values are 0-100.
     `locator` must identify the row inside the source (entry name, row id, table).
   - `imp.finish()` writes `data/observations/scores/<name>.csv` (regenerated each run), stages claims in
     `.cache/staging/<name>.csv`, and lists unresolved model names in `.cache/reports/<name>-unresolved.csv`.
5. **Keep every reported number.** The same model on the same benchmark from two sources, or with two
   harnesses/settings, are separate rows. Never average, pick a "best", or drop conflicting rows.
6. `uv run atlas validate` must report 0 errors.
