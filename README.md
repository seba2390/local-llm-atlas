# local-llm-atlas

A **cited** data catalogue of open-weight LLMs and what it takes to run them on consumer hardware:
benchmark scores, architecture (dense / MoE / hybrid), parameter counts, disk / RAM / VRAM footprint,
measured tokens/s, and the runtimes and tricks (MoE expert offload, disk streaming, non-uniform low-bit
quantisation, ...) that let large models fit on small machines.

**Rule #1: every number has a source.** Rows without a resolvable citation fail CI.

## Layout

```
data/
  sources/<id>.yaml          one file per citation (paper, model card, leaderboard data, forum post, own measurement)
  models/<org>/<id>.yaml     one file per released checkpoint: architecture, params, license, context, ...
  benchmarks/<id>.yaml       benchmark definitions + allowed variants (e.g. humaneval: base|plus)
  hardware/<id>.yaml         GPUs / systems (VRAM, bandwidth, consumer or not)
  runtimes/<id>.yaml         inference engines and the memory-saving techniques they implement
  observations/
    scores.csv               model x benchmark x variant x setting -> value, source_id, locator
    artifacts.csv            downloadable weights per format/quant -> exact disk bytes
    throughput.csv           model x quant x runtime x hardware x offload x context -> prompt/gen tok/s
    memory.csv               measured peak VRAM/RAM for a given config
schema/                      JSON Schemas for every entity and observation row
src/atlas/                   loader, validator, builder (`atlas` CLI)
```

Entities (things that exist) are YAML; observations (things someone measured or reported) are CSV rows,
each with a `source_id` and a `locator` (table / line / leaderboard entry inside the source).

### Design choices

- **Keep every reported number.** Self-reported (model vendor) and independent (leaderboard) scores
  often disagree; both are stored and distinguished via the source's `provenance`
  (`first_party` / `third_party` / `community` / `own_measurement`). Which one a plot uses is a query-time choice.
- **Benchmarks are versioned.** "Aider" and "Aider Polyglot" are different benchmarks; Terminal-Bench 1.0–4.0
  are different task sets. The validator rejects scores dated before the benchmark (variant) existed,
  which catches the most common mislabelling.
- **Settings that change the number are recorded**: agent scaffold, edit format, reasoning on/off, quantisation,
  LiveCodeBench date window, etc.
- **Measured vs derived.** Disk sizes come from the hosting API (exact). Measured VRAM/tok/s live in
  `observations/`. Estimates (e.g. weights + KV-cache at a given context) are computed in `atlas build`
  and never stored as if they were measurements.

## Usage

```bash
uv sync
uv run atlas validate          # schemas, references, citations, sanity checks
uv run atlas validate --strict # warnings are errors too
uv run atlas build             # -> build/atlas.json (+ derived per-model summaries)
uv run pytest
```

## Example

`data/` currently contains a fully cited example: Qwen2.5-Coder-32B-Instruct. Qwen's launch blog reports
"Aider 73.7". That was published 2024-11-12, before Aider Polyglot existed (2024-12-21), so it is filed under
`aider-code-editing`, next to Aider's own independent 71.4 on that benchmark. On Aider Polyglot the
independent results are 8.0 (diff) and 16.4 (whole).

## License

Code: MIT. Data: each record cites its source; check the source's terms before redistributing bulk extracts.
