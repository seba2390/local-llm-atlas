"""BigCodeBench leaderboard, from the Hugging Face datasets behind https://bigcode-bench.github.io/:
bigcode/bigcodebench-results (full set, 1140 tasks) and bigcode/bigcodebench-hard-results (hard subset),
each pinned to a dataset revision.

Row fields: model, link, moe, size (total params, B), act_param (activated params, B), type (🟢 base model /
🔶 chat/instruction-tuned), complete / instruct (calibrated pass@1, greedy decoding), date (model release date),
prefill (False = evaluated without response prefilling, 🧠 on the leaderboard).
"""

from __future__ import annotations

import io
import re
from typing import Any

import pyarrow.parquet as pq

from .common import Importer, ensure_harness, fetch, write_source

DATASETS = {  # split -> (dataset, revision, source id)
    "full": ("bigcode/bigcodebench-results", "08ceb6b15ecd9c0f7c82932e4e248208ffeeb9b4", "bigcodebench-results-hf"),
    "hard": ("bigcode/bigcodebench-hard-results", "522a45334d838a87b421ac96db27b2ef7fc165bb", "bigcodebench-hard-results-hf"),
}
PARQUET = "https://huggingface.co/datasets/{}/resolve/{}/data/train-00000-of-00001.parquet"
TYPES = {"🟢": "base", "🔶": "chat"}


def _parse_name(raw: str) -> tuple[str, list[str], str | None, str | None]:
    """'o1-2024-12-17 (temperature=1, reasoning=high)' -> ('o1-2024-12-17', ['temperature=1'], 'high', 'on')."""
    m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", raw)
    if not m:
        return raw, [], None, None
    inner = [p.strip() for p in m.group(2).split(",") if p.strip()]
    known = all("=" in p or p.lower() in ("w/ reasoning", "reproduced") or re.fullmatch(r"\d+k tokens", p) for p in inner)
    if not known:
        return raw, [], None, None  # e.g. 'DeepSeek-V2-Chat (2024-06-28)': part of the name
    setting: list[str] = []
    effort = reasoning = None
    for p in inner:
        low = p.lower()
        if low == "w/ reasoning":
            reasoning = "on"
        elif low == "reproduced":
            setting.append("run=reproduced")
        elif low.endswith("k tokens"):
            setting.append(f"max_tokens={low.split()[0]}")
        else:
            k, v = (s.strip() for s in p.split("=", 1))
            if k == "reasoning":
                effort = f"budget={v}" if v.isdigit() else v
                reasoning = "on"
            else:
                setting.append(f"{k}={v}")
    return m.group(1).strip(), setting, effort, reasoning


def _rows(split: str) -> list[dict[str, Any]]:
    ds, rev, _ = DATASETS[split]
    return pq.read_table(io.BytesIO(fetch(PARQUET.format(ds, rev)))).to_pylist()


def run() -> dict:
    for split, (ds, rev, sid) in DATASETS.items():
        write_source(sid, kind="leaderboard_data", provenance="third_party",
                     title=f"BigCodeBench leaderboard results, {split} set (Hugging Face dataset {ds})",
                     publisher="BigCode", url=f"https://huggingface.co/datasets/{ds}/tree/{rev}", revision=rev,
                     notes="Data behind https://bigcode-bench.github.io/ (calibrated pass@1, greedy decoding).")
    ensure_harness("bigcodebench", "BigCodeBench official evaluation", "eval-framework",
                   url="https://github.com/bigcode-project/bigcodebench",
                   sources=["arxiv-2406.15877", DATASETS["full"][2]],
                   notes="The BigCodeBench maintainers' own evaluation harness (bigcodebench package).")

    imp = Importer("bigcodebench")
    for split, (_, _, sid) in DATASETS.items():
        for r in _rows(split):
            raw = (r.get("model") or "").strip()
            name, extra_setting, effort, reasoning = _parse_name(raw)
            link = r.get("link") or ""
            hf = re.match(r"https://huggingface\.co/([\w.-]+/[\w.-]+)/?$", link)
            size = r.get("size")
            imp.claim(name, hf_repo=hf.group(1) if hf else None, open_weights=True if hf else None,
                      params_total_b=float(size) if size else None, release_date=r.get("date") or None,
                      source_id=sid)

            setting = ["decoding=greedy", "calibrated=true", f"model_type={TYPES.get(r.get('type'), r.get('type'))}"]
            if r.get("prefill") is False:
                setting.append("prefill=false")
            setting += extra_setting
            notes = []
            if name != raw:
                notes.append(f"listed as '{raw}'")
            if r.get("moe"):
                notes.append(f"MoE, {r.get('act_param')}B activated of {size}B")
            if link:
                notes.append(f"link: {link}")
            for task in ("complete", "instruct"):
                v = r.get(task)
                if v is None:
                    continue
                imp.score(name, benchmark_id="bigcodebench", variant=f"{task}-{split}", value=float(v),
                          harness_id="bigcodebench", setting="; ".join(setting), reasoning=reasoning,
                          reasoning_effort=effort, source_id=sid, locator=f"{DATASETS[split][0]} model={raw} column={task}",
                          notes="; ".join(notes) or None)
    return imp.finish()
