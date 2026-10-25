# Contributing data

1. **Add the source first** (`data/sources/<id>.yaml`). Use stable URLs; for GitHub/Hugging Face pin a
   commit/revision in `revision` (and in the URL where possible). Set `provenance` honestly:
   - `first_party`: published by whoever made the model/hardware/runtime (model cards, launch blogs, tech reports)
   - `third_party`: independent evaluators (leaderboards, papers evaluating others' models)
   - `community`: forum posts, Reddit, GitHub discussions
   - `own_measurement`: produced by this project (commit the script/log alongside)
2. **Add or update entities** (`models/`, `benchmarks/`, `hardware/`, `runtimes/`). Use `sources` for the default
   citation and `field_sources` for fields that came from somewhere else.
3. **Append observation rows** (`observations/*.csv`). Always fill `source_id` and `locator`.
   Never edit another source's row to "correct" it; add a new row with the new source.
4. Run `uv run atlas validate` and `uv run pytest`.

## IDs

Lower-case, `[a-z0-9._-]`. Models: the upstream repo name lower-cased (`qwen2.5-coder-32b-instruct`).
Model files live under their org folder (`data/models/qwen/`).

## What not to add

- Numbers you can't link to a source.
- Numbers read off a chart image unless the source has no table (then say so in `notes`).
- Estimated values as observations. Put estimates in the build step instead.
