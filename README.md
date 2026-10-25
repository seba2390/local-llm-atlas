# local-llm-atlas

A **cited** data catalogue of LLMs, open-weight **and** closed (API-only, for comparison), and what it takes
to run the open ones on consumer hardware: benchmark scores (with the harness that produced them), architecture (dense / MoE / hybrid), parameter counts, disk / RAM / VRAM footprint,
measured tokens/s, and the runtimes and tricks (MoE expert offload, disk streaming, non-uniform low-bit
quantisation, ...) that let large models fit on small machines.

**Rule #1: every number has a source.** Rows without a resolvable citation fail CI.

**Rule #2: values are looked up, never measured by this project.** The atlas compiles published numbers only.

## What's in it (2026-10-25)

| | |
|---|---|
| Score rows | **33,675** (every row: model, benchmark/variant, harness + version, settings, source, locator) |
| Models | **968** (529 open-weight, 226 closed/API, 213 not stated by any imported source), each with company + country of its developer where a source states it |
| Benchmarks | 95 (coding, agentic/SWE, tool use, math, science, knowledge, long context, games, ...) |
| Harnesses / agent scaffolds | 209 |
| Organisations | 111 |
| Sources | 178 citation records (papers, model cards pinned to a revision, leaderboard data pinned to a commit) |

Score rows per importer: BFCL 19,952 · Epoch AI 8,886 · Aider 1,116 · vendor model cards / tech reports 1,014 ·
Big Code Models Leaderboard 778 · BigCodeBench 652 · EvalPlus 428 · τ/τ²-bench 316 · SWE-bench 270 · CRUXEval 160 ·
EvalPerf 102 · hand-entered 1.

Not yet imported (importer stubs/plans only): Terminal-Bench direct, LiveCodeBench, LiveBench direct, Open LLM
Leaderboard, LMArena, tokens/s + hardware + offload runtimes. Hugging Face enrichment (architecture from
`config.json`, exact weight/GGUF sizes) ran only partially: 89 models have an architecture type so far.

## Layout

```
data/
  sources/<id>.yaml          one file per citation (paper, model card, leaderboard data, docs, forum post)
  models/<org>/<id>.yaml     one file per model (open checkpoint or closed API snapshot): company, country, architecture, params, license, aliases, ...
  orgs/<id>.yaml             organisations behind models: name, country (copied into every model as company/country)
  benchmarks/<id>.yaml       benchmark definitions + allowed variants (e.g. humaneval: base|plus)
  hardware/<id>.yaml         GPUs / systems (VRAM, bandwidth, consumer or not)
  runtimes/<id>.yaml         inference engines and the memory-saving techniques they implement
  harnesses/<id>.yaml        eval harnesses / agent scaffolds (aider, OpenHands, Terminus, EvalPlus, ..., or 'undisclosed')
  curated/                   hand-extracted tables (e.g. vendor_scores.csv) read by importers
  observations/
    scores/<importer>.csv    model x benchmark x variant x harness(+version) x setting -> value, source_id, locator
    artifacts[/...].csv      downloadable weights per format/quant -> exact disk bytes
    throughput[/...].csv     model x quant x runtime x hardware x offload x context -> prompt/gen tok/s (published)
    memory[/...].csv         published peak VRAM/RAM for a given config
schema/                      JSON Schemas for every entity and observation row
src/atlas/                   loader, validator, builder, resolver (`atlas` CLI)
src/atlas/ingest/            one importer per source (see docs/IMPORTERS.md)
```

Entities (things that exist) are YAML; observations (things someone measured or reported) are CSV rows,
each with a `source_id` and a `locator` (table / line / leaderboard entry inside the source).

### Design choices

- **Keep every reported number.** Self-reported (model vendor) and independent (leaderboard) scores
  often disagree; both are stored and distinguished via the source's `provenance`
  (`first_party` / `third_party` / `community`). Which one a plot uses is a query-time choice.
- **The harness is part of the score.** Every score row names its harness (`harness_id`, `harness_version`) plus the
  options that change the number (`setting`, e.g. edit format, max turns) and reasoning effort. If the source doesn't
  say, the harness is `undisclosed`, and benchmarks marked `harness_sensitive` emit a warning for such rows.
- **One model, many names.** Sources name the same checkpoint differently (`Qwen/Qwen2.5-Coder-32B-Instruct`,
  `qwen-2.5-coder-32b-instruct`, `openrouter/qwen/...`). Each model lists its `aliases`; the resolver matches
  case/separator-insensitively but never drops tokens, so `-Instruct` vs base or dated API snapshots stay distinct.
  Two models sharing an alias is a validation error. Provider and reasoning-effort suffixes (`_high`, `(16k thinking)`,
  `(Fireworks)`) go into the score row (`reasoning_effort`, notes), not into the model.
- **Company and country** come from `data/orgs/<org>.yaml` (cited) and are copied into each model; the validator
  checks they match. Organisations that no imported source describes have no country rather than a guessed one.
- **Aggregators keep the primary source.** When a row comes from an aggregator (e.g. Epoch AI), `original_source` /
  `original_url` hold the paper/leaderboard/model card it cites.
- **Closed models are first-class.** `open_weights: false`, one record per dated API snapshot where the provider
  publishes one (`api_ids`). Architecture fields are filled only from official disclosures, never from rumours.
- **Benchmarks are versioned.** "Aider" and "Aider Polyglot" are different benchmarks; Terminal-Bench 1.0–4.0
  are different task sets. The validator rejects scores dated before the benchmark (variant) existed,
  which catches the most common mislabelling.
- **Other settings that change the number are recorded**: reasoning on/off and effort, quantisation,
  LiveCodeBench date window, etc.
- **Reported vs derived.** Disk sizes come from the hosting API (exact). Published VRAM/tok/s figures live in
  `observations/`. Estimates (e.g. weights + KV-cache at a given context) are computed in `atlas build`
  and never stored as if they were measurements.

## Usage

```bash
uv sync
uv run atlas validate          # schemas, references, citations, sanity checks
uv run atlas validate --strict # warnings are errors too
uv run atlas build             # -> build/atlas.json (+ derived per-model summaries)
uv run pytest

uv run atlas ingest epoch      # re-run one importer (downloads are cached in .cache/)
uv run atlas ingest all        # all importers -> register new models/orgs -> merge duplicates -> re-run
uv run atlas register          # create/refresh model + org records from what importers staged
```

## Scripts

Scripts in `scripts/` have no command-line arguments: edit the capitalised settings at the top and run them.

- `scripts/check_for_model.py`: set `QUERY` and run `uv run python scripts/check_for_model.py` to list every model
  whose name, id or alias contains it (case-insensitive; spaces/hyphens/underscores ignored), with company,
  open/closed, parameters, release date and number of score rows.

### Plotting

`scripts/plot_frontier.py` plots, for each benchmark, the best open-weight model with at most `MAX_PARAMS`
billion parameters on every day from `START` to `END` (step line; each model that took the lead is labelled), with
`REFERENCE_MODELS` (open or closed) as horizontal lines. Edit the settings at the top of the script and run it:

```python
MAX_PARAMS = 35
REFERENCE_MODELS = ["gemini-2.5-flash"]
BENCHMARKS = ["aider-polyglot", "swe-bench:verified", "livecodebench"]  # "id" or "id:variant"
START = "2024-01-01"
END = "2026-10-25"
```

```bash
uv run python scripts/plot_frontier.py   # -> plots/<benchmark>[_<variant>]_le<N>B.png + .csv
```

- A reference model with no score on one of the benchmarks raises `MissingReferenceScoreError` before anything is drawn.
- A model's release date decides when it joins the frontier; models without a release date or parameter count are
  listed as skipped. When a model has several reported scores (sources/harnesses/settings), the highest is used.
- The CSV next to each plot lists the source, locator, harness and settings of every plotted number.
- Reusable building blocks (score selection, frontier computation, drawing, colours) live in
  `scripts/plot_utils.py`.

## Example

Qwen2.5-Coder-32B-Instruct (open) and Claude 3.5 Sonnet 2024-10-22 (closed; 51.6 on Aider Polyglot, aider
0.71.2.dev, diff format) are hand-curated examples. Qwen's launch blog reports
"Aider 73.7". That was published 2024-11-12, before Aider Polyglot existed (2024-12-21), so it is filed under
`aider-code-editing`, next to Aider's own independent 71.4 on that benchmark. On Aider Polyglot the
independent results are 8.0 (diff) and 16.4 (whole), both with aider 0.69.2.dev, which shows how much
the harness configuration matters.

## License

Code: MIT. Data: each record cites its source; check the source's terms before redistributing bulk extracts.
