"""EvalPlus leaderboard (HumanEval/HumanEval+ and MBPP/MBPP+), from the results.json behind
https://evalplus.github.io/leaderboard.html, pinned to a commit of github.com/evalplus/evalplus.github.io.

Leaderboard notes: pass@1 with greedy decoding; HumanEval+ v0.1.10, MBPP+ v0.2.0; MBPP and MBPP+ both use the
399-task hand-verified subset of MBPP-sanitized; `prompted` (✨) = evaluated in a chat setting, else direct code
completion; `open-data` Full (💚) = open weights + open data, Partial (💙) = open weights + open SFT data;
`size` = *activated* parameters (so it is only claimed as total size for dense models).
"""

from __future__ import annotations

import re
from typing import Any

from .common import Importer, ensure_harness, fetch_json, write_source

COMMIT = "4ebfa0d4b62edb942232c723b9d1641fe1d24780"  # last commit touching results.json
URL = f"https://raw.githubusercontent.com/evalplus/evalplus.github.io/{COMMIT}/results.json"
SRC = "evalplus-leaderboard-data"

# results.json key -> (benchmark, variant, dataset setting)
COLUMNS = {
    "humaneval": ("humaneval", "base", "dataset=HumanEval+ v0.1.10 (base tests)"),
    "humaneval+": ("humaneval", "plus", "dataset=HumanEval+ v0.1.10"),
    "mbpp": ("mbpp", "base", "subset=399 hand-verified MBPP-sanitized tasks; dataset=MBPP+ v0.2.0 (base tests)"),
    "mbpp+": ("mbpp", "plus", "subset=399 hand-verified MBPP-sanitized tasks; dataset=MBPP+ v0.2.0"),
}
MOE_RE = re.compile(r"mixtral|dbrx|deepseek-v[23]|deepseek-coder-v2|moe\b", re.I)
NAME_SIZE_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)b\b", re.I)


def _params(name: str, size: Any) -> float | None:
    """`size` is activated params: usable as total only for dense models whose name doesn't contradict it."""
    if size is None or MOE_RE.search(name):
        return None
    size = float(size)
    m = NAME_SIZE_RE.search(name)
    if m and abs(float(m.group(1)) - size) > 0.15 * size:
        return None  # e.g. 'gemma-1.1-2b-it' listed with size 7.0
    return size


def run() -> dict:
    write_source(SRC, kind="leaderboard_data", provenance="third_party",
                 title="EvalPlus leaderboard results (results.json)", publisher="EvalPlus",
                 url=f"https://github.com/evalplus/evalplus.github.io/blob/{COMMIT}/results.json", revision=COMMIT,
                 notes="Served at https://evalplus.github.io/results.json and rendered on "
                       "https://evalplus.github.io/leaderboard.html (pass@1, greedy decoding; HumanEval+ v0.1.10, "
                       "MBPP+ v0.2.0).")
    ensure_harness("evalplus", "EvalPlus", "eval-framework", url="https://github.com/evalplus/evalplus",
                   sources=["arxiv-2305.01210", SRC],
                   notes="EvalPlus evaluation framework (HumanEval+/MBPP+ test harness and codegen setup).")

    imp = Importer("evalplus")
    data = fetch_json(URL)
    for model, d in data.items():
        link = d.get("link") or ""
        hf = re.match(r"https://huggingface\.co/([\w.-]+/[\w.-]+)/?$", link)
        if hf and "python" in hf.group(1).lower() and "python" not in model.lower():
            hf = None  # e.g. 'CodeLlama-70B' linked to CodeLlama-70b-Python-hf: link doesn't identify the named model
        open_data = str(d.get("open-data") or "").upper()
        open_w = True if link.startswith("https://huggingface.co/") or open_data in ("FULL", "PARTIAL") else None
        imp.claim(model, hf_repo=hf.group(1) if hf else None, open_weights=open_w,
                  params_total_b=_params(model, d.get("size")), source_id=SRC)

        mode = "chat" if d.get("prompted") else "completion"
        notes = [f"open-data={d.get('open-data')}"]
        if d.get("size") is not None:
            notes.append(f"activated params={d['size']}B")
        if link:
            notes.append(f"link: {link}")
        for col, (bench, variant, ds) in COLUMNS.items():
            v = (d.get("pass@1") or {}).get(col)
            if v is None:
                continue
            imp.score(model, benchmark_id=bench, variant=variant, value=float(v), harness_id="evalplus",
                      setting=f"decoding=greedy; mode={mode}; {ds}", source_id=SRC,
                      locator=f"results.json model={model} pass@1.{col}", notes="; ".join(notes))
    return imp.finish()
