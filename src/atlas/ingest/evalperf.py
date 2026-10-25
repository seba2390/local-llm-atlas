"""EvalPerf leaderboard (evalplus.github.io/evalperf.html): efficiency of LLM-generated code.

Data: results/evalperf/COMBINED-RESULTS.json in github.com/evalplus/evalplus.github.io (Apache-2.0), the file the
page renders. Keys are 'org--Model' (HF repo) or API ids; '(⏩)' marks performance-encouraging prompts.
Metrics: DPS (Differential Performance Score, default), normalised DPS and pass@1 on the EvalPerf tasks.
The model/task win rates depend on the set of compared models and are not imported.
"""

from __future__ import annotations

import yaml

from .common import DATA, Importer, dump_yaml, ensure_harness, fetch_json, write_source

NAME = "evalperf"
REV = "87418456d3a8c2bc6265f24ff91d2dd0e7993f4b"
URL = f"https://raw.githubusercontent.com/evalplus/evalplus.github.io/{REV}/results/evalperf/COMBINED-RESULTS.json"
SRC = "evalperf-leaderboard"
SRC_PAPER = "arxiv-2408.06450"

BENCH = {
    "id": "evalperf", "name": "EvalPerf", "category": "code-generation", "metric": "dps", "unit": "percent",
    "higher_is_better": True, "introduced": "2024-08-12", "url": "https://evalplus.github.io/evalperf.html",
    "sources": [SRC_PAPER],
    "description": "Code efficiency: Differential Performance Score (DPS) = percentile of reference solutions the model's "
                   "correct solutions match or outperform on performance-exercising tasks.",
    "notes": "Other metrics: dps_norm, pass@1 (correctness on the same tasks).",
    "managed_by": NAME,
}
METRICS = {"dps": None, "dps_norm": "dps_norm", "pass@1": "pass@1"}


def run() -> dict[str, int]:
    imp = Importer(NAME)
    write_source(SRC, kind="leaderboard_data", provenance="third_party", title="EvalPerf leaderboard",
                 publisher="EvalPlus team (UIUC)", url="https://evalplus.github.io/evalperf.html", revision=REV,
                 notes="Apache-2.0. results/evalperf/COMBINED-RESULTS.json in github.com/evalplus/evalplus.github.io. "
                 "o1-series models use 50 samples (half the usual number).")
    write_source(SRC_PAPER, kind="paper", provenance="first_party", title="Evaluating Language Models for Efficient Code Generation",
                 authors=["Jiawei Liu", "Songrun Xie", "Junhao Wang", "Yuxiang Wei", "Yifeng Ding", "Lingming Zhang"],
                 publisher="arXiv", url="https://arxiv.org/abs/2408.06450", published="2024-08-12",
                 notes="Introduces Differential Performance Evaluation (DPE) and EvalPerf.")
    ensure_harness("evalplus", "EvalPlus", "eval-framework", url="https://github.com/evalplus/evalplus", sources=[SRC_PAPER],
                   notes="EvalPlus evaluation framework (HumanEval+/MBPP+/EvalPerf).")
    path = DATA / "benchmarks" / "evalperf.yaml"
    if not path.exists() or (yaml.safe_load(path.read_text()) or {}).get("managed_by") == NAME:
        dump_yaml(path, BENCH)

    data = fetch_json(URL)
    for key, rec in data.items():
        if key == "heatmap_data" or not isinstance(rec, dict):
            continue
        label = key.strip()
        perf_prompt = "(⏩)" in label
        base = label.replace("(⏩)", "").strip()
        hf = base.replace("--", "/") if "--" in base else None
        raw = hf or base
        imp.claim(raw, hf_repo=hf, open_weights=True if hf else None, source_id=SRC)
        setting = ["temperature=1.0", f"prompt={'performance-encouraging' if perf_prompt else 'default'}"]
        setting.append("n_samples=50" if base.startswith("o1-") else "n_samples=100")
        for field, metric in METRICS.items():
            v = rec.get(field)
            if v is None:
                continue
            imp.score(raw, benchmark_id="evalperf", metric=metric, value=float(v), harness_id="evalplus",
                      setting="; ".join(setting), source_id=SRC, locator=f"COMBINED-RESULTS.json key={label}")
    return imp.finish()
