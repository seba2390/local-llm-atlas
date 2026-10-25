# local-llm-atlas

A **cited** data catalogue of LLMs, open-weight **and** closed (API-only, for comparison), and what it takes
to run the open ones on consumer hardware: benchmark scores (with the harness that produced them), architecture (dense / MoE / hybrid), parameter counts, disk / RAM / VRAM footprint,
measured tokens/s, and the runtimes and tricks (MoE expert offload, disk streaming, non-uniform low-bit
quantisation, ...) that let large models fit on small machines.

**Rule #1: every number has a source.** Rows without a resolvable citation fail CI.

**Rule #2: values are looked up, never measured by this project.** The atlas compiles published numbers only.

## Layout

```
data/
  sources/<id>.yaml          one file per citation (paper, model card, leaderboard data, docs, forum post)
  models/<org>/<id>.yaml     one file per model (open checkpoint or closed API snapshot): architecture, params, license, ...
  benchmarks/<id>.yaml       benchmark definitions + allowed variants (e.g. humaneval: base|plus)
  hardware/<id>.yaml         GPUs / systems (VRAM, bandwidth, consumer or not)
  runtimes/<id>.yaml         inference engines and the memory-saving techniques they implement
  harnesses/<id>.yaml        eval harnesses / agent scaffolds (aider, OpenHands, Terminus, EvalPlus, ..., or 'undisclosed')
  observations/
    scores.csv               model x benchmark x variant x harness(+version) x setting -> value, source_id, locator
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
  (`first_party` / `third_party` / `community`). Which one a plot uses is a query-time choice.
- **The harness is part of the score.** Every score row names its harness (`harness_id`, `harness_version`) plus the
  options that change the number (`setting`, e.g. edit format, max turns) and reasoning effort. If the source doesn't
  say, the harness is `undisclosed`, and benchmarks marked `harness_sensitive` emit a warning for such rows.
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
```

## Example

`data/` currently contains two fully cited examples: Qwen2.5-Coder-32B-Instruct (open) and Claude 3.5 Sonnet
2024-10-22 (closed; 51.6 on Aider Polyglot, aider 0.71.2.dev, diff format). Qwen's launch blog reports
"Aider 73.7". That was published 2024-11-12, before Aider Polyglot existed (2024-12-21), so it is filed under
`aider-code-editing`, next to Aider's own independent 71.4 on that benchmark. On Aider Polyglot the
independent results are 8.0 (diff) and 16.4 (whole), both with aider 0.69.2.dev, which shows how much
the harness configuration matters.

## License

Code: MIT. Data: each record cites its source; check the source's terms before redistributing bulk extracts.
