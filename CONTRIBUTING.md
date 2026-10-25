# Contributing data

1. **Add the source first** (`data/sources/<id>.yaml`). Use stable URLs; for GitHub/Hugging Face pin a
   commit/revision in `revision` (and in the URL where possible). Set `provenance` honestly:
   - `first_party`: published by whoever made the model/hardware/runtime (model cards, launch blogs, tech reports)
   - `third_party`: independent evaluators (leaderboards, papers evaluating others' models)
   - `community`: forum posts, Reddit, GitHub discussions

   This project does not run its own measurements; only published values are recorded.
2. **Add or update entities** (`models/`, `orgs/`, `benchmarks/`, `hardware/`, `runtimes/`, `harnesses/`).
   Use `sources` for the default citation and `field_sources` for fields that came from somewhere else.
   After changing an org, run `uv run atlas sync-orgs` to copy company/country into its models.
3. **Append observation rows** (`observations/scores/manual.csv` by hand, or write an importer; see
   docs/IMPORTERS.md). Always fill `source_id` and `locator`.
   For scores, always fill `harness_id` (use `undisclosed` if the source doesn't say), `harness_version` if stated,
   and put harness options / eval settings that affect the number in `setting`.
   Never edit another source's row to "correct" it; add a new row with the new source.
4. Run `uv run atlas validate` and `uv run pytest`.

## IDs

Lower-case, `[a-z0-9._-]`. Models: the upstream repo name lower-cased (`qwen2.5-coder-32b-instruct`).
Model files live under their org folder (`data/models/qwen/`).
Closed models: one record per dated API snapshot when the provider has one (`claude-3-5-sonnet-20241022`),
with `open_weights: false` and `api_ids`.

## What not to add

- Numbers you can't link to a source.
- Numbers read off a chart image unless the source has no table (then say so in `notes`).
- Estimated values as observations. Put estimates in the build step instead.
- Our own measurements.
- Architecture details for closed models that the provider has not officially disclosed.
