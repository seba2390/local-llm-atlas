"""Big Code Models Leaderboard (Hugging Face space bigcode/bigcode-models-leaderboard).

HumanEval (Python) and MultiPL-E pass@1 for open code models, all run with the bigcode-evaluation-harness
(top-p 0.95, temperature 0.2, n_samples 50, max_length_generation 512). Rows marked 'EXT' were evaluated by the
submitter with the same harness and added via a community PR (linked in the notes).

Instruction-tuned models (🔶) are scored on HumanEval-Python with the HumanEvalSynthesize instruction prompt
(max length 2048); base models (🟢) and every MultiPL-E language use the original completion prompts.
Win Rate (a rank average over the models in the table) and throughput/memory columns are not imported.
"""

from __future__ import annotations

import csv
import io
import re

import yaml

from .common import DATA, Importer, dump_yaml, ensure_harness, fetch_text, write_source

NAME = "bigcode-leaderboard"
SPACE = "bigcode/bigcode-models-leaderboard"
REV = "985e474a4a1d32f8738299951f4041781886fc1b"
CSV_URL = f"https://huggingface.co/spaces/{SPACE}/resolve/{REV}/data/code_eval_board.csv"
SRC = "bigcode-models-leaderboard"
SRC_MULTIPLE = "arxiv-2208.08227"
LANGS = ["java", "javascript", "cpp", "php", "julia", "d", "lua", "r", "racket", "rust", "swift"]

MULTIPLE_BENCH = {
    "id": "multipl-e", "name": "MultiPL-E", "category": "code-generation", "metric": "pass@1", "unit": "percent",
    "higher_is_better": True, "introduced": "2022-08-17", "url": "https://github.com/nuprl/MultiPL-E",
    "variants": [{"id": lang} for lang in LANGS] + [
        {"id": "bigcode-average", "notes": "Big Code Models Leaderboard 'Average score': mean pass@1 over HumanEval-Python "
         "and the 11 MultiPL-E languages above (only comparable within that leaderboard)."},
        {"id": "vendor-average", "notes": "Single 'MultiPL-E' number in a vendor table; language set and averaging not stated "
         "(only comparable within the same vendor table)."}],
    "sources": [SRC_MULTIPLE],
    "description": "HumanEval (and MBPP) translated into many programming languages; variant = target language "
                   "(HumanEval-derived problems).",
    "managed_by": NAME,
}


def _num(s: str | None) -> float | None:
    try:
        return float((s or "").strip())
    except ValueError:
        return None


def _write_benchmark() -> None:
    path = DATA / "benchmarks" / "multipl-e.yaml"
    if path.exists() and (yaml.safe_load(path.read_text()) or {}).get("managed_by") != NAME:
        return
    dump_yaml(path, MULTIPLE_BENCH)


def run() -> dict[str, int]:
    imp = Importer(NAME)
    write_source(SRC, kind="leaderboard_data", provenance="third_party", title="Big Code Models Leaderboard",
                 publisher="BigCode (Hugging Face)", url=f"https://huggingface.co/spaces/{SPACE}", revision=REV,
                 notes="data/code_eval_board.csv in the space repository. Scores from the bigcode-evaluation-harness "
                 "(top-p=0.95, temperature=0.2, n_samples=50); 'EXT' rows are community submissions run by the submitter.")
    write_source(SRC_MULTIPLE, kind="paper", provenance="first_party",
                 title="MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation",
                 authors=["Federico Cassano", "John Gouwar", "et al."], publisher="arXiv",
                 url="https://arxiv.org/abs/2208.08227", published="2022-08-17", notes="Introduces MultiPL-E.")
    ensure_harness("bigcode-evaluation-harness", "bigcode-evaluation-harness", "eval-framework",
                   url="https://github.com/bigcode-project/bigcode-evaluation-harness", sources=[SRC],
                   notes="BigCode's framework for evaluating code generation models (HumanEval, MultiPL-E, MBPP, DS-1000, ...).")
    _write_benchmark()

    rows = list(csv.DictReader(io.StringIO(fetch_text(CSV_URL))))
    for i, row in enumerate(rows, start=2):
        name = (row.get("Model") or "").strip()
        tag = (row.get("T") or "").strip()
        instruct = "🔶" in tag
        ext = "EXT" in tag
        link = (row.get("Links") or "").strip()
        hf = m.group(1) if (m := re.match(r"https?://huggingface\.co/([^/\s]+/[^/\s?#]+)", link)) else None
        size = _num(row.get("Size (B)"))
        ctx = _num(row.get("Seq_length"))
        imp.claim(name, hf_repo=hf, open_weights=True if hf else None, params_total_b=size,
                  context_length=int(ctx) if ctx else None, source_id=SRC)
        base = ["temperature=0.2", "top_p=0.95", "n_samples=50"]
        pr = (row.get("Submission PR") or "").strip()
        notes = ("community submission (EXT), evaluated by the submitter" + (f": {pr}" if pr else "")) if ext else None
        common = dict(harness_id="bigcode-evaluation-harness", source_id=SRC, notes=notes,
                      locator=f"data/code_eval_board.csv row {i} Model={name}")
        kind = "model_type=" + ("instruction-tuned" if instruct else "base")
        he = _num(row.get("humaneval-python"))
        if he is not None:
            prompt = ["prompt=humanevalsynthesize-instruct", "max_length_generation=2048"] if instruct else \
                     ["prompt=original-completion", "max_length_generation=512"]
            imp.score(name, benchmark_id="humaneval", variant="base", value=he,
                      setting="; ".join([kind, *base, *prompt]), **common)
        for lang in LANGS:
            v = _num(row.get(lang))
            if v is not None:
                imp.score(name, benchmark_id="multipl-e", variant=lang, value=v,
                          setting="; ".join([kind, *base, "max_length_generation=512"]), **common)
        avg = _num(row.get("Average score"))
        if avg is not None:
            imp.score(name, benchmark_id="multipl-e", variant="bigcode-average", value=avg,
                      setting="; ".join([kind, *base]), **common)
    return imp.finish()
