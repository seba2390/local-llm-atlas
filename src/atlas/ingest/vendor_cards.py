"""Vendor (first-party) benchmark tables: Hugging Face model cards, technical reports and launch blogs.

Rows are hand-extracted into data/curated/vendor_scores.csv (one row per printed number, copied verbatim, with a
locator naming the table/row/column). When a vendor table also lists other developers' models (comparison columns),
those numbers are kept too: they are the vendor's own measurement of a competitor and are cited to the vendor document
(the `notes` column says "comparison model in <vendor> table").

Harness: `authors-internal` when the vendor does not say how a number was produced; otherwise the stated scaffold
(OpenHands, mini-SWE-agent, Terminus-2, Claude Code, ...), with its options in `setting`.
"""

from __future__ import annotations

import csv
from typing import Any

from .common import DATA, Importer, ensure_benchmark, ensure_harness, write_source

NAME = "vendor-cards"
CSV_PATH = DATA / "curated" / "vendor_scores.csv"

# ---------------------------------------------------------------- sources

# sid -> (hf repo, pinned revision sha, published (repo creation date), org display name)
HF_CARDS: dict[str, tuple[str, str, str, str]] = {
    # @@HF_CARDS@@
    "hf-qwen3-235b-a22b-instruct-2507": ("Qwen/Qwen3-235B-A22B-Instruct-2507", "ac9c66cc9b46af7306746a9250f23d47083d689e", "2025-07-21", "Qwen Team, Alibaba"),
    "hf-qwen3-235b-a22b-thinking-2507": ("Qwen/Qwen3-235B-A22B-Thinking-2507", "6cbffae6d8e28b986a6b17bd36f42f9fa0f1f0a5", "2025-07-25", "Qwen Team, Alibaba"),
    "hf-qwen3-30b-a3b-instruct-2507": ("Qwen/Qwen3-30B-A3B-Instruct-2507", "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe", "2025-07-28", "Qwen Team, Alibaba"),
    "hf-qwen3-30b-a3b-thinking-2507": ("Qwen/Qwen3-30B-A3B-Thinking-2507", "144afc2f379b542fdd4e85a1fcd5e1f79112d95d", "2025-07-29", "Qwen Team, Alibaba"),
    "hf-qwen3-4b-instruct-2507": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", "2025-08-05", "Qwen Team, Alibaba"),
    "hf-qwen3-4b-thinking-2507": ("Qwen/Qwen3-4B-Thinking-2507", "768f209d9ea81521153ed38c47d515654e938aea", "2025-08-05", "Qwen Team, Alibaba"),
    "hf-qwen3-next-80b-a3b-instruct": ("Qwen/Qwen3-Next-80B-A3B-Instruct", "9c7f2fbe84465e40164a94cc16cd30b6999b0cc7", "2025-09-09", "Qwen Team, Alibaba"),
    "hf-qwen3-next-80b-a3b-thinking": ("Qwen/Qwen3-Next-80B-A3B-Thinking", "e502dd4100cc68c0de57643fd4317ec93a128670", "2025-09-09", "Qwen Team, Alibaba"),
    "hf-qwen3.5-27b": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654", "2026-02-24", "Qwen Team, Alibaba"),
    "hf-qwen3.5-397b-a17b": ("Qwen/Qwen3.5-397B-A17B", "8472618112abcbd45acbcdc58436aff4233c23f7", "2026-02-16", "Qwen Team, Alibaba"),
    "hf-qwen3.5-9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "2026-02-27", "Qwen Team, Alibaba"),
    "hf-qwen3.6-27b": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", "2026-04-21", "Qwen Team, Alibaba"),
    "hf-qwen3.6-35b-a3b": ("Qwen/Qwen3.6-35B-A3B", "995ad96eacd98c81ed38be0c5b274b04031597b0", "2026-04-15", "Qwen Team, Alibaba"),
    "hf-qwen3.8-27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "2026-08-05", "Qwen Team, Alibaba"),
    "hf-qwen3.8-flash-next": ("Qwen/Qwen3.8-Flash-Next", "de4b8e4d43b917e7706784d8bb445c9af86a3540", "2026-08-24", "Qwen Team, Alibaba"),
    # @@/HF_CARDS@@
}

# Model-card sources that already exist in data/sources/ (hand-curated); cited but never rewritten here.
SHARED_SOURCES = {"hf-qwen2.5-coder-32b-instruct"}

# Non-HF first-party documents (blogs, technical reports) and benchmark references.
OTHER_SOURCES: dict[str, dict[str, Any]] = {
    # @@OTHER_SOURCES@@
    # @@/OTHER_SOURCES@@
}

# Benchmark references (cited by the benchmark records created below).
BENCH_SOURCES: dict[str, dict[str, Any]] = {
    "maa-aime": dict(kind="docs", provenance="first_party", title="American Invitational Mathematics Examination (AIME)",
                     publisher="Mathematical Association of America",
                     url="https://maa.org/maa-invitational-competitions/"),
    "hmmt-site": dict(kind="docs", provenance="first_party", title="Harvard-MIT Mathematics Tournament (HMMT)",
                      publisher="HMMT", url="https://www.hmmt.org/"),
    "matharena": dict(kind="leaderboard", provenance="third_party", title="MathArena: Evaluating LLMs on uncontaminated math competitions",
                      publisher="ETH Zurich SRI Lab", url="https://matharena.ai/",
                      notes="Hosts the final-answer versions of AIME/HMMT competitions that model vendors usually report."),
    "arxiv-2501.01257": dict(kind="paper", provenance="first_party",
                             title="CodeElo: Benchmarking Competition-level Code Generation of LLMs with Human-comparable Elo Ratings",
                             authors=["Shanghaoran Quan", "et al."], publisher="arXiv", url="https://arxiv.org/abs/2501.01257",
                             published="2025-01-02"),
    "arxiv-2401.03065": dict(kind="paper", provenance="first_party",
                             title="CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution",
                             authors=["Alex Gu", "et al."], publisher="arXiv", url="https://arxiv.org/abs/2401.03065",
                             published="2024-01-05"),
    "arxiv-2509.16941": dict(kind="paper", provenance="first_party",
                             title="SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?",
                             authors=["Xiang Deng", "et al."], publisher="arXiv", url="https://arxiv.org/abs/2509.16941",
                             published="2025-09-21"),
    "arxiv-2504.21798": dict(kind="paper", provenance="first_party",
                             title="SWE-smith: Scaling Data for Software Engineering Agents",
                             authors=["John Yang", "et al."], publisher="arXiv", url="https://arxiv.org/abs/2504.21798",
                             published="2025-04-30", notes="Introduces SWE-bench Multilingual (300 tasks, 9 languages)."),
    "swe-bench-multilingual-site": dict(kind="docs", provenance="first_party", title="SWE-bench Multilingual",
                                        publisher="SWE-bench team", url="https://www.swebench.com/multilingual.html"),
}

# ---------------------------------------------------------------- entities

B = dict
BENCHMARKS: dict[str, dict[str, Any]] = {
    "aime": B(name="AIME (American Invitational Mathematics Examination)", category="math", metric="accuracy",
              unit="percent", higher_is_better=True, url="https://maa.org/maa-invitational-competitions/",
              variants=[{"id": "2024", "introduced": "2024-02-01", "notes": "AIME 2024 I + II (30 problems)."},
                        {"id": "2025", "introduced": "2025-02-06", "notes": "AIME 2025 I + II (30 problems)."},
                        {"id": "2026", "notes": "AIME 2026 I + II (30 problems)."}],
              sources=["maa-aime", "matharena"],
              notes="Vendors usually report avg@k / pass@1 over several samples; record k and sampling in `setting` when stated."),
    "hmmt": B(name="HMMT (Harvard-MIT Mathematics Tournament), final-answer problems", category="math",
              metric="accuracy", unit="percent", higher_is_better=True, url="https://www.hmmt.org/",
              variants=[{"id": "feb-2025", "notes": "HMMT February 2025."}, {"id": "nov-2025", "notes": "HMMT November 2025."},
                        {"id": "feb-2026", "notes": "HMMT February 2026."}],
              sources=["hmmt-site", "matharena"]),
    "codeforces": B(name="Codeforces rating (vendor-simulated contests)", category="competitive-programming",
                    metric="rating", unit="elo", higher_is_better=True, url="https://codeforces.com/",
                    sources=["arxiv-2501.01257"],
                    notes="Vendors estimate an Elo rating from their own Codeforces problem sets and simulation protocol; "
                          "ratings from different vendors are not directly comparable. Percentile-only reports are not stored here."),
    "cruxeval": B(name="CRUXEval", category="code-generation", metric="pass@1", unit="percent", higher_is_better=True,
                  introduced="2024-01-05", url="https://crux-eval.github.io/",
                  variants=[{"id": "input", "notes": "CRUXEval-I: predict the input from the output."},
                            {"id": "output", "notes": "CRUXEval-O: predict the output from the input."}],
                  sources=["arxiv-2401.03065"], notes="Record CoT vs direct prompting in `setting`."),
    "swe-bench-pro": B(name="SWE-Bench Pro", category="software-engineering", metric="resolved_rate", unit="percent",
                       higher_is_better=True, harness_sensitive=True, introduced="2025-09-21",
                       url="https://scale.com/leaderboard/swe_bench_pro_public",
                       variants=[{"id": "public", "notes": "Public set (731 tasks)."}, {"id": "commercial"}],
                       sources=["arxiv-2509.16941"]),
    "swe-bench-multilingual": B(name="SWE-bench Multilingual", category="software-engineering", metric="resolved_rate",
                                unit="percent", higher_is_better=True, harness_sensitive=True, introduced="2025-04-30",
                                url="https://www.swebench.com/multilingual.html",
                                sources=["swe-bench-multilingual-site", "arxiv-2504.21798"],
                                notes="300 tasks from 42 repositories in 9 programming languages."),
}

# id -> (name, kind, url, notes); `sources` = the vendor documents that name the harness.
HARNESSES: dict[str, tuple[str, str, str | None, str]] = {
    # @@HARNESSES@@
    "qwen-internal-swe-scaffold": ('Qwen internal SWE agent scaffold (bash + file-edit tools)', 'vendor-internal', None, "Qwen's in-house agent scaffold for SWE-bench-style tasks, as described in Qwen3.6 model cards."),
    # @@/HARNESSES@@
}

# ---------------------------------------------------------------- model claims (subject models of each card)

MODELS: dict[str, dict[str, Any]] = {
    # @@MODELS@@
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {"org": "alibaba", "release_date": "2025-07-21", "open_weights": True, "hf_repo": "Qwen/Qwen3-235B-A22B-Instruct-2507", "params_total_b": 235.1, "license": "apache-2.0", "source_id": "hf-qwen3-235b-a22b-instruct-2507"},
    "Qwen/Qwen3-235B-A22B-Thinking-2507": {"org": "alibaba", "release_date": "2025-07-25", "open_weights": True, "hf_repo": "Qwen/Qwen3-235B-A22B-Thinking-2507", "params_total_b": 235.1, "license": "apache-2.0", "source_id": "hf-qwen3-235b-a22b-thinking-2507"},
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {"org": "alibaba", "release_date": "2025-07-28", "open_weights": True, "hf_repo": "Qwen/Qwen3-30B-A3B-Instruct-2507", "params_total_b": 30.5, "license": "apache-2.0", "source_id": "hf-qwen3-30b-a3b-instruct-2507"},
    "Qwen/Qwen3-30B-A3B-Thinking-2507": {"org": "alibaba", "release_date": "2025-07-29", "open_weights": True, "hf_repo": "Qwen/Qwen3-30B-A3B-Thinking-2507", "params_total_b": 30.5, "license": "apache-2.0", "source_id": "hf-qwen3-30b-a3b-thinking-2507"},
    "Qwen/Qwen3-4B-Instruct-2507": {"org": "alibaba", "release_date": "2025-08-05", "open_weights": True, "hf_repo": "Qwen/Qwen3-4B-Instruct-2507", "params_total_b": 4.0, "license": "apache-2.0", "source_id": "hf-qwen3-4b-instruct-2507"},
    "Qwen/Qwen3-4B-Thinking-2507": {"org": "alibaba", "release_date": "2025-08-05", "open_weights": True, "hf_repo": "Qwen/Qwen3-4B-Thinking-2507", "params_total_b": 4.0, "license": "apache-2.0", "source_id": "hf-qwen3-4b-thinking-2507"},
    "Qwen/Qwen3-Next-80B-A3B-Instruct": {"org": "alibaba", "release_date": "2025-09-09", "open_weights": True, "hf_repo": "Qwen/Qwen3-Next-80B-A3B-Instruct", "params_total_b": 81.3, "license": "apache-2.0", "source_id": "hf-qwen3-next-80b-a3b-instruct"},
    "Qwen/Qwen3-Next-80B-A3B-Thinking": {"org": "alibaba", "release_date": "2025-09-09", "open_weights": True, "hf_repo": "Qwen/Qwen3-Next-80B-A3B-Thinking", "params_total_b": 81.3, "license": "apache-2.0", "source_id": "hf-qwen3-next-80b-a3b-thinking"},
    "Qwen/Qwen3.5-27B": {"org": "alibaba", "release_date": "2026-02-24", "open_weights": True, "hf_repo": "Qwen/Qwen3.5-27B", "params_total_b": 27.8, "license": "apache-2.0", "source_id": "hf-qwen3.5-27b"},
    "Qwen/Qwen3.5-397B-A17B": {"org": "alibaba", "release_date": "2026-02-16", "open_weights": True, "hf_repo": "Qwen/Qwen3.5-397B-A17B", "params_total_b": 403.4, "license": "apache-2.0", "source_id": "hf-qwen3.5-397b-a17b"},
    "Qwen/Qwen3.5-9B": {"org": "alibaba", "release_date": "2026-02-27", "open_weights": True, "hf_repo": "Qwen/Qwen3.5-9B", "params_total_b": 9.7, "license": "apache-2.0", "source_id": "hf-qwen3.5-9b"},
    "Qwen/Qwen3.6-27B": {"org": "alibaba", "release_date": "2026-04-21", "open_weights": True, "hf_repo": "Qwen/Qwen3.6-27B", "params_total_b": 27.8, "license": "apache-2.0", "source_id": "hf-qwen3.6-27b"},
    "Qwen/Qwen3.6-35B-A3B": {"org": "alibaba", "release_date": "2026-04-15", "open_weights": True, "hf_repo": "Qwen/Qwen3.6-35B-A3B", "params_total_b": 36.0, "license": "apache-2.0", "source_id": "hf-qwen3.6-35b-a3b"},
    "Qwen/Qwen3.8-27B": {"org": "alibaba", "release_date": "2026-08-05", "open_weights": True, "hf_repo": "Qwen/Qwen3.8-27B", "params_total_b": 27.8, "license": "apache-2.0", "source_id": "hf-qwen3.8-27b"},
    "Qwen/Qwen3.8-Flash-Next": {"org": "alibaba", "release_date": "2026-08-24", "open_weights": True, "hf_repo": "Qwen/Qwen3.8-Flash-Next", "params_total_b": 180.0, "license": "other", "source_id": "hf-qwen3.8-flash-next"},
    # @@/MODELS@@
}


def _write_sources() -> None:
    for sid, (repo, sha, published, publisher) in HF_CARDS.items():
        if sid in SHARED_SOURCES:
            continue
        write_source(sid, kind="model_card", provenance="first_party", title=f"{repo} model card (README.md)",
                     publisher=publisher, url=f"https://huggingface.co/{repo}", published=published, revision=sha,
                     notes=f"Benchmark tables read from https://huggingface.co/{repo}/raw/{sha}/README.md; "
                           "published = repository creation date.")
    for sid, fields in OTHER_SOURCES.items():
        write_source(sid, **fields)
    for sid, fields in BENCH_SOURCES.items():  # shared references: never overwrite another importer's copy
        if not (DATA / "sources" / f"{sid}.yaml").exists():
            write_source(sid, **fields)


def _harness_sources() -> dict[str, list[str]]:
    used: dict[str, set[str]] = {}
    with CSV_PATH.open() as fh:
        for r in csv.DictReader(fh):
            used.setdefault(r["harness_id"], set()).add(r["source_id"])
    return {h: sorted(s)[:5] for h, s in used.items()}


def run() -> dict:
    _write_sources()
    for bid, fields in BENCHMARKS.items():
        ensure_benchmark(bid, **fields)
    hs = _harness_sources()
    for hid, (name, kind, url, notes) in HARNESSES.items():
        ensure_harness(hid, name, kind, url=url, sources=hs.get(hid) or None, notes=notes)

    imp = Importer(NAME)
    for raw, meta in MODELS.items():
        imp.claim(raw, **meta)
    with CSV_PATH.open() as fh:
        for r in csv.DictReader(fh):
            row = {k: (v.strip() or None) for k, v in r.items() if k != "raw_name"}
            row["value"] = float(row["value"])
            if r["raw_name"] not in MODELS:
                # comparison model in a vendor table: the table is the only source we have for it
                imp.claim(r["raw_name"], source_id=row["source_id"])
            imp.score(r["raw_name"], **row)
    return imp.finish()
