"""Epoch AI 'Capabilities & benchmarking' hub + 'Data on AI models' (both CC-BY 4.0).

Two kinds of rows:
  * Epoch's own evaluations (gpqa_diamond.csv, swe_bench_verified.csv, ...): harness `epoch-ai-evals`,
    linked to the public Inspect log of the run.
  * Aggregated external results (*_external.csv): Epoch cites the primary source per row; we keep it in
    original_source / original_url and pick the harness from (in order) an explicit agent/scaffold column,
    the benchmark's own leaderboard (`<benchmark>-official`), HELM, or the reporting paper's authors.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections import defaultdict
from typing import Any
from urllib.parse import urlparse

from .common import Importer, ensure_harness, fetch, write_source, DATA, dump_yaml

BENCH_ZIP = "https://epoch.ai/data/benchmark_data.zip"
MODELS_ZIP = "https://epoch.ai/data/ai_models.zip"
SRC_RUNS = "epoch-benchmarks-runs"
SRC_EXT = "epoch-benchmarks-external"
SRC_MODELS = "epoch-ai-models"

EFFORT_RE = re.compile(r"_(low|medium|high|xhigh|max|minimal|none|unknown|thinking|nothinking|auto|default|\d+[kK])$", re.I)
HOSTING_PREFIXES = ("accounts/fireworks/models/", "fireworks/", "/HFEndpoint/", "deepinfra/")

# Benchmarks this importer owns (regenerated each run). Shared benchmarks (gpqa, mmlu, swe-bench, ...) are
# hand-curated in data/benchmarks/ and only referenced here.
B = dict  # brevity
BENCHMARKS: dict[str, dict[str, Any]] = {
    "otis-mock-aime": B(name="OTIS Mock AIME 2024-2025", category="math", url="https://epoch.ai/benchmarks/otis-mock-aime"),
    "frontiermath": B(name="FrontierMath", category="math", url="https://epoch.ai/frontiermath"),
    "epoch-chess-puzzles": B(name="Chess Puzzles (Epoch AI)", category="games", url="https://epoch.ai/benchmarks"),
    "ebr-bench": B(name="EBR-bench", category="other", url="https://epoch.ai/benchmarks"),
    "mystery-game-puzzles": B(name="Mystery Game Puzzles", category="games", url="https://epoch.ai/benchmarks"),
    "mirrorcode": B(name="MirrorCode", category="software-engineering", url="https://epoch.ai/benchmarks", harness_sensitive=True),
    "furniture-assembly": B(name="Furniture Assembly", category="reasoning", url="https://epoch.ai/benchmarks"),
    "spireclimb": B(name="SpireClimb (Slay the Spire)", category="games", url="https://epoch.ai/benchmarks", harness_sensitive=True),
    "anli": B(name="ANLI (Adversarial NLI)", category="reasoning", url="https://github.com/facebookresearch/anli"),
    "ale-bench": B(name="ALE-Bench", category="competitive-programming", unit="score", url="https://sakanaai.github.io/ALE-Bench/", harness_sensitive=True),
    "algotune": B(name="AlgoTune", category="software-engineering", unit="ratio", url="https://algotune.io/", harness_sensitive=True),
    "apex-agents": B(name="APEX-Agents", category="agentic", url="https://www.mercor.com/apex/", harness_sensitive=True),
    "arc-agi": B(name="ARC-AGI", category="reasoning", url="https://arcprize.org/leaderboard", variants=[{"id": "1"}, {"id": "2", "introduced": "2025-03-24"}]),
    "balrog": B(name="BALROG", category="games", url="https://balrogai.com/", harness_sensitive=True),
    "blueprint-bench-2": B(name="Blueprint-Bench 2", category="multimodal", url="https://andonlabs.com/evals/blueprint-bench"),
    "boolq": B(name="BoolQ", category="reading-comprehension", url="https://github.com/google-research-datasets/boolean-questions"),
    "btf-3": B(name="Bench to the Future 3 (BTF-3)", category="forecasting", unit="score", url="https://evals.futuresearch.ai/", harness_sensitive=True),
    "cadeval": B(name="CadEval", category="code-generation", url="https://willpatrick.xyz/cadevalresults_20250422_095709/"),
    "cl-bench": B(name="CL-bench", category="long-context", url="https://epoch.ai/benchmarks", variants=[{"id": "main"}, {"id": "life"}]),
    "commonsenseqa-2": B(name="CommonsenseQA 2.0", category="commonsense", url="https://allenai.github.io/csqa2/"),
    "critpt": B(name="CritPt", category="science", url="https://critpt.com/"),
    "cursorbench": B(name="CursorBench", category="software-engineering", url="https://cursor.com/cursorbench", harness_sensitive=True),
    "cybench": B(name="Cybench", category="cybersecurity", url="https://cybench.github.io/", harness_sensitive=True),
    "deepresearch-bench": B(name="Deep Research Bench (FutureSearch)", category="research", url="https://evals.futuresearch.ai/", harness_sensitive=True),
    "deepswe": B(name="DeepSWE (Datacurve)", category="software-engineering", url="https://deepswe.datacurve.ai/", harness_sensitive=True),
    "dtbench": B(name="DTBench", category="reasoning", url="https://conceptualreasoning.ai/dtbench"),
    "enigmaeval": B(name="EnigmaEval", category="reasoning", url="https://labs.scale.com/leaderboard/enigma_eval"),
    "exploitbench": B(name="ExploitBench", category="cybersecurity", unit="score", url="https://epoch.ai/benchmarks", harness_sensitive=True),
    "fiction-livebench": B(name="Fiction.LiveBench", category="long-context", url="https://fiction.live/stories/Fiction-liveBench"),
    "forecastbench": B(name="ForecastBench", category="forecasting", unit="score", url="https://www.forecastbench.org/"),
    "frontiercode": B(name="FrontierCode (Cognition)", category="software-engineering", url="https://cognition.com/frontiercode", harness_sensitive=True),
    "frontierswe": B(name="FrontierSWE", category="software-engineering", url="https://www.frontierswe.com/", harness_sensitive=True),
    "gbaeval": B(name="GBAEval", category="software-engineering", url="https://gbaeval.com/", harness_sensitive=True),
    "gdp-pdf": B(name="GDP.pdf (Surge AI)", category="agentic", url="https://surgehq.ai/benchmarks/gdp-pdf"),
    "gdpval": B(name="GDPval", category="agentic", url="https://openai.com/index/gdpval/"),
    "geobench": B(name="GeoBench", category="multimodal", url="https://geobench.org/"),
    "gso": B(name="GSO (Global Software Optimization)", category="software-engineering", url="https://gso-bench.github.io/", harness_sensitive=True),
    "lambada": B(name="LAMBADA", category="language-modeling", url="https://zenodo.org/record/2630551"),
    "lech-mazur-writing": B(name="Lech Mazur Creative Writing", category="writing", unit="score", url="https://github.com/lechmazur/writing"),
    "lmca": B(name="LMCA", category="reasoning", url="https://conceptualreasoning.ai/lmca"),
    "metr-time-horizon": B(name="METR Time Horizons", category="agentic", unit="minutes", url="https://metr.org/time-horizons/", harness_sensitive=True),
    "mindcube": B(name="MindCube", category="multimodal", url="https://mind-cube.github.io/"),
    "openbookqa": B(name="OpenBookQA", category="knowledge", url="https://allenai.org/data/open-book-qa"),
    "osworld": B(name="OSWorld", category="computer-use", url="https://os-world.github.io/", harness_sensitive=True, variants=[{"id": "1"}, {"id": "2", "introduced": "2026-06-26"}]),
    "piqa": B(name="PIQA", category="commonsense", url="https://yonatanbisk.com/piqa/"),
    "posttrainbench": B(name="PostTrainBench", category="research", url="https://epoch.ai/benchmarks", harness_sensitive=True),
    "proofbench": B(name="ProofBench (Vals AI)", category="math", url="https://www.vals.ai/benchmarks/proof_bench"),
    "remote-labor-index": B(name="Remote Labor Index", category="agentic", url="https://www.remotelabor.ai/", harness_sensitive=True),
    "scicode": B(name="SciCode", category="science", url="https://scicode-bench.github.io/"),
    "scienceqa": B(name="ScienceQA", category="science", url="https://scienceqa.github.io/"),
    "simplebench": B(name="SimpleBench", category="reasoning", url="https://simple-bench.com/"),
    "spatialviz-bench": B(name="SpatialViz-Bench", category="multimodal", url="https://arxiv.org/abs/2507.07610"),
    "superglue": B(name="SuperGLUE", category="reading-comprehension", url="https://super.gluebenchmark.com/"),
    "surface-evolver-bench": B(name="Surface Evolver Bench", category="science", url="https://yhenon.github.io/surface-evolver-bench/"),
    "the-agent-company": B(name="TheAgentCompany", category="agentic", url="https://the-agent-company.com/", harness_sensitive=True),
    "triviaqa": B(name="TriviaQA", category="knowledge", url="https://nlp.cs.washington.edu/triviaqa/"),
    "vending-bench-2": B(name="Vending-Bench 2", category="agentic", unit="usd", url="https://andonlabs.com/evals/vending-bench-2", harness_sensitive=True),
    "video-mme": B(name="Video-MME", category="multimodal", url="https://video-mme.github.io/"),
    "vpct": B(name="VPCT", category="multimodal", url="https://cbrower.dev/vpct"),
    "webdev-arena": B(name="WebDev Arena", category="chat-arena", unit="elo", url="https://web.lmarena.ai/leaderboard"),
    "weirdml": B(name="WeirdML", category="code-generation", url="https://htihle.github.io/weirdml.html", variants=[{"id": "v1-v2"}, {"id": "v3", "introduced": "2026-09-18"}]),
}

# file stem -> how to read it. Keys: bench, variant, col, harness_col, effort_col, shots_col, date_col, stderr_col,
# variant_fn, extra ({column: (variant or None, metric)}), setting_cols, default (harness when no column/leaderboard).
OWN = {
    "gpqa_diamond": B(bench="gpqa", variant="diamond"),
    "math_level_5": B(bench="math", variant="level-5"),
    "otis_mock_aime_2024_2025": B(bench="otis-mock-aime"),
    "swe_bench_verified": B(bench="swe-bench", variant="verified"),
    "frontiermath": B(bench="frontiermath", variant="tiers-1-3-2025-02-28-private"),
    "frontiermath_tier_4": B(bench="frontiermath", variant="tier-4-2025-07-01-private"),
    "frontiermath_tiers_1_3_v2": B(bench="frontiermath", variant="tiers-1-3-v2-private"),
    "frontiermath_tier_4_v2": B(bench="frontiermath", variant="tier-4-v2-private"),
    "frontiermath_erdos": B(bench="frontiermath", variant="erdos"),
    "simpleqa_verified": B(bench="simpleqa", variant="verified"),
    "chess_puzzles": B(bench="epoch-chess-puzzles"),
    "ebr_bench": B(bench="ebr-bench"),
    "mystery_game_puzzles": B(bench="mystery-game-puzzles"),
    "mirrorcode": B(bench="mirrorcode"),
    "furniture_assembly": B(bench="furniture-assembly"),
    "spireclimb": B(bench="spireclimb", variant="combined"),
    "spireclimb_sts1_a20_heart": B(bench="spireclimb", variant="sts1-a20-heart"),
    "spireclimb_sts2_a10": B(bench="spireclimb", variant="sts2-a10"),
}

CLASSIC = "authors"  # default harness for classic academic benchmarks: whoever wrote the cited paper/report
EXTERNAL = {
    "adversarial_nli": B(bench="anli", col="Score", shots_col="Shots", default=CLASSIC,
                         extra={"R1": (None, "accuracy_r1"), "R2": (None, "accuracy_r2"), "R3": (None, "accuracy_r3")}),
    "aider_polyglot": B(bench="aider-polyglot", col="Percent correct", harness="aider", setting_cols={"Edit format": "edit_format"},
                        date_col="Date of evaluation", extra={"Percent using correct edit format": (None, "percent_well_formed")}),
    "ale_bench": B(bench="ale-bench", col="Performance"),
    "algotune": B(bench="algotune", col="Score"),
    "apex_agents": B(bench="apex-agents", col="Pass@1 score", stderr_col="Pass@1 Standard Error", extra={"Mean score": (None, "mean_score")}),
    "arc_agi": B(bench="arc-agi", variant="1", col="Score", extra={"Cost per task": (None, "cost_per_task_usd")}),
    "arc_agi_2": B(bench="arc-agi", variant="2", col="Score", extra={"Cost per task": (None, "cost_per_task_usd")}),
    "arc_ai2": B(bench="arc", variant="challenge", col="Challenge score", shots_col="Shots", default=CLASSIC,
                 extra={"Easy score": ("easy", None)}),
    "balrog": B(bench="balrog", col="Average progress", stderr_col="Average Standard error",
                extra={f"{e} progress": (None, f"{e.lower()}_progress") for e in ["BabyAI", "Crafter", "TextWorld", "BabaIsAI", "MiniHack", "NetHack"]}),
    "bbh": B(bench="bbh", col="Average", shots_col="Shots", default=CLASSIC),
    "blueprint_bench_2": B(bench="blueprint-bench-2", col="Score"),
    "bool_q": B(bench="boolq", col="Score", shots_col="Shots", default=CLASSIC),
    "btf3": B(bench="btf-3", col="Pooled score", harness_col="Harness"),
    "cad_eval": B(bench="cadeval", col="Overall pass (%)", setting_cols={"Prompt": "prompt"}),
    "cl_bench": B(bench="cl-bench", variant="main", col="Overall"),
    "cl_bench_life": B(bench="cl-bench", variant="life", col="Overall"),
    "common_sense_qa_2": B(bench="commonsenseqa-2", col="Score", shots_col="Shots", default=CLASSIC),
    "critpt": B(bench="critpt", col="Accuracy"),
    "cursorbench": B(bench="cursorbench", col="Score", effort_col="Reasoning level"),
    "cybench": B(bench="cybench", col="Unguided % Solved", extra={"Subtask-Guided % Solved": (None, "subtask_guided_solved"), "Subtasks % Solved": (None, "subtasks_solved")}),
    "deepresearchbench": B(bench="deepresearch-bench", col="Average score", harness_col="Agent"),
    "deepswe": B(bench="deepswe", col="Pass@1", harness_col="Harness", effort_col="Reasoning effort", extra={"Pass@4": (None, "pass@4")}),
    "dtbench": B(bench="dtbench", col="Accuracy"),
    "enigma_eval": B(bench="enigmaeval", col="Accuracy", stderr_col="Accuracy Standard Error"),
    "exploitbench": B(bench="exploitbench", col="Mean capability", harness_col="Harness"),
    "fictionlivebench": B(bench="fiction-livebench", col="16k token score", metric="score_16k",
                          extra={f"{k} token score": (None, f"score_{k}") for k in ["0", "400", "1k", "2k", "4k", "8k", "32k", "60k", "120k", "192k"]}),
    "forecastbench": B(bench="forecastbench", col="Overall score", extra={"Dataset score": (None, "dataset_score"), "Market score": (None, "market_score")}),
    "frontiercode": B(bench="frontiercode", col="Main score", harness_col="Harness", effort_col="Reasoning effort", setting_cols={"Aggregation": "aggregation"}),
    "frontierswe": B(bench="frontierswe", col="Score", harness_col="Harness", setting_cols={"Aggregation": "aggregation"}),
    "gbaeval": B(bench="gbaeval", col="Overall score", date_col="Graded at"),
    "gdp_pdf": B(bench="gdp-pdf", col="GDP.pdf score", date_col="Last updated"),
    "gdpval": B(bench="gdpval", col="Win Rate (%)", extra={"Win + tie rate (%)": (None, "win_or_tie_rate")}),
    "geobench": B(bench="geobench", col="ACW Country %", setting_cols={"Tools": "tools"}),
    "gsm8k": B(bench="gsm8k", col="EM", shots_col="Shots", default=CLASSIC),
    "gso": B(bench="gso", col="Score OPT@1", harness_col="Scaffold", date_col="Evaluation date", extra={"Score OPT@10": (None, "opt@10")}),
    "hella_swag": B(bench="hellaswag", col="Overall accuracy", shots_col="Shots", default=CLASSIC),
    "hle": B(bench="hle", col="Accuracy", stderr_col="Accuracy Standard Error", extra={"Calibration Error": (None, "calibration_error")}),
    "lambada": B(bench="lambada", col="Score", shots_col="Shots", default=CLASSIC),
    "lech_mazur_writing": B(bench="lech-mazur-writing", col="Mean score"),
    "live_bench": B(bench="livebench", col="Global average", variant_fn="livebench",
                    extra={f"{c} average": (None, c.lower().replace(" ", "_")) for c in ["Reasoning", "Coding", "Mathematics", "Data analysis", "Language"]} | {"IF Average": (None, "instruction_following")}),
    "lmca": B(bench="lmca", col="Score"),
    "metr_time_horizons": B(bench="metr-time-horizon", col="Time horizon", metric="p50_horizon_minutes", variant_fn="metr",
                            extra={"Time Horizon (80%)": (None, "p80_horizon_minutes")}),
    "mindcube": B(bench="mindcube", col="Overall score"),
    "mmlu": B(bench="mmlu", variant="base", col="EM", shots_col="Shots", default=CLASSIC),
    "open_book_qa": B(bench="openbookqa", col="Accuracy", shots_col="Shots", default=CLASSIC),
    "os_world": B(bench="osworld", variant="1", col="Score", harness_col="Agent", date_col="Date added"),
    "osworld_2": B(bench="osworld", variant="2", col="Binary accuracy", effort_col="Reasoning", setting_cols={"Tool setting": "tools", "Step budget": "step_budget"},
                   extra={"Partial score": (None, "partial_score")}),
    "piqa": B(bench="piqa", col="Score", shots_col="Shots", default=CLASSIC),
    "posttrainbench": B(bench="posttrainbench", col="Average (%)", harness_col="Scaffold"),
    "proofbench": B(bench="proofbench", col="Accuracy", stderr_col="Accuracy Standard Error", effort_col="Reasoning effort"),
    "rli": B(bench="remote-labor-index", col="Score"),
    "scicode": B(bench="scicode", col="Score"),
    "science_qa": B(bench="scienceqa", col="Score", shots_col="Shots", default=CLASSIC),
    "simplebench": B(bench="simplebench", col="Score (AVG@5)"),
    "spatialviz_bench": B(bench="spatialviz-bench", col="Overall score", setting_cols={"Prompting": "prompting"}),
    "superglue": B(bench="superglue", col="Score", default=CLASSIC),
    "surface_evolver_bench": B(bench="surface-evolver-bench", col="Mean score", extra={"Pass rate": (None, "pass_rate")}),
    "terminalbench": B(bench="terminal-bench", col="Accuracy mean", harness_col="Agent", stderr_col="Accuracy SE", date_col="Run date", variant_fn="tbench"),
    "the_agent_company": B(bench="the-agent-company", col="% Resolved", date_col="Date", extra={"% Score": (None, "partial_score")}),
    "trivia_qa": B(bench="triviaqa", col="EM", shots_col="Shots", default=CLASSIC),
    "vending_bench_2": B(bench="vending-bench-2", col="Score", metric="final_balance_usd"),
    "video_mme": B(bench="video-mme", col="Overall (no subtitles)", metric="accuracy_no_subtitles", setting_cols={"Frames": "frames"},
                   extra={"Overall (with subtitles)": (None, "accuracy_with_subtitles")}),
    "vpct": B(bench="vpct", col="Correct"),
    "webdev_arena": B(bench="webdev-arena", col="Arena Score", date_col="Last updated"),
    "weirdml": B(bench="weirdml", variant="v1-v2", col="Accuracy", stderr_col="Accuracy SE"),
    "weirdml_v3": B(bench="weirdml", variant="v3", col="Score", harness_col="Agent", effort_col="Reasoning effort", version_col="Harness version",
                    extra={"Max performance": (None, "max_performance")}),
    "wino_grande": B(bench="winogrande", col="Accuracy", shots_col="Shots", default=CLASSIC),
}

# Sources that mean "the benchmark maintainers' own leaderboard"
LEADERBOARD_HINT = re.compile(r"leaderboard|dashboard|github\.io|\.ai/|\.com/|\.org/|\.dev/|\.xyz|websit|repository|experiment results", re.I)
HELM_HINT = re.compile(r"helm|crfm\.stanford", re.I)
THIRD_PARTY_EVALUATORS = {
    "artificialanalysis.ai": ("artificial-analysis", "Artificial Analysis", "https://artificialanalysis.ai/"),
    "lmcouncil.ai": ("lm-council", "LM Council", "https://lmcouncil.ai/"),
}
NO_DIRECT_SCALE = {"ale-bench", "algotune", "btf-3", "forecastbench", "lech-mazur-writing", "metr-time-horizon", "vending-bench-2", "webdev-arena"}


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip().rstrip("%").replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _is_url(s: str) -> bool:
    return bool(re.match(r"^https?://\S+$", s.strip()))


def split_effort(version: str) -> tuple[str, str | None]:
    # Epoch writes the Pro tier and its effort as one suffix, e.g. gpt-5.6-sol_promax -> gpt-5.6-sol-pro, effort max.
    m = re.search(r"_pro(low|medium|high|xhigh|max|unknown)$", version)
    if m:
        return version[: m.start()] + "-pro", m.group(1)
    m = EFFORT_RE.search(version)
    return (version[: m.start()], m.group(1)) if m else (version, None)


def reasoning_of(effort: str | None) -> str | None:
    if not effort:
        return None
    e = effort.lower()
    if e in ("none", "nothinking"):
        return "off"
    if e in ("unknown", "auto", "default"):
        return None
    return "on"


def _date(s: str | None) -> str | None:
    if not s:
        return None
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s.strip())
    return m.group(1) if m else None


def _read_zip(url: str) -> dict[str, list[dict[str, str]]]:
    zf = zipfile.ZipFile(io.BytesIO(fetch(url)))
    out = {}
    for info in zf.infolist():
        if info.filename.endswith(".csv"):
            out[info.filename] = list(csv.DictReader(io.TextIOWrapper(zf.open(info), encoding="utf-8")))
    return out


def _harness_for(cfg: dict, row: dict, bench_id: str, bench_name: str, orig_src: str, orig_url: str) -> str:
    if cfg.get("harness"):
        return cfg["harness"]
    if cfg.get("harness_col"):
        name = (row.get(cfg["harness_col"]) or "").strip()
        if name:
            hid = re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-.")
            return ensure_harness(hid, name, "agent-scaffold", sources=[SRC_EXT],
                                  notes=f"Agent/scaffold name as listed by Epoch AI for {bench_name}.")
    text = f"{orig_src} {orig_url}"
    if HELM_HINT.search(text):
        return "helm"
    for domain, (hid, name, url) in THIRD_PARTY_EVALUATORS.items():
        if domain in text:
            return ensure_harness(hid, name, "eval-framework", url=url, sources=[SRC_EXT],
                                  notes="Independent evaluator that runs benchmarks with its own harness.")
    if cfg.get("default") == CLASSIC and not re.search(r"leaderboard", text, re.I):
        return "authors-internal"
    return ensure_harness(f"{bench_id}-official", f"{bench_name} maintainers' evaluation", "benchmark-reference",
                          sources=[SRC_EXT], notes="Score from the benchmark maintainers' own leaderboard/evaluation setup.")


def _variant(cfg: dict, row: dict) -> str | None:
    fn = cfg.get("variant_fn")
    if fn == "livebench":
        v = (row.get("LiveBench Version") or "").strip()
        return v.replace("LiveBench-", "") or "unspecified"
    if fn == "metr":
        v = (row.get("METR version") or "").strip()
        return v.replace("METR-Horizon-", "").lower() if v else "unspecified"
    if fn == "tbench":
        text = " ".join(row.get(k) or "" for k in ("Source", "Source Link", "Source link"))
        m = re.search(r"terminal-bench/(\d\.\d)", text)
        if m:
            return m.group(1)
        if re.search(r"\bv2\b", text):
            return "2.0"
        return None
    return cfg.get("variant")


def run() -> dict[str, int]:
    imp = Importer("epoch")
    bench = _read_zip(BENCH_ZIP)
    models_db = _read_zip(MODELS_ZIP)

    write_source(SRC_RUNS, kind="leaderboard_data", provenance="third_party",
                 title="Epoch AI, 'Capabilities & benchmarking' - evaluations run by Epoch AI", publisher="Epoch AI",
                 url=BENCH_ZIP, notes="CC-BY 4.0. Files without the _external suffix; each row links to Epoch's public Inspect log (original_url).")
    write_source(SRC_EXT, kind="leaderboard_data", provenance="third_party",
                 title="Epoch AI, 'Capabilities & benchmarking' - external results compiled by Epoch AI", publisher="Epoch AI",
                 url=BENCH_ZIP, notes="CC-BY 4.0. *_external.csv files. Epoch compiles results from leaderboards, papers and model cards; "
                 "the primary source of each row is kept in original_source/original_url and may be first-party.")
    write_source(SRC_MODELS, kind="leaderboard_data", provenance="third_party", title="Epoch AI, 'Data on AI models'",
                 publisher="Epoch AI", url=MODELS_ZIP, notes="CC-BY 4.0. Model metadata (organisation, country, publication date, parameters, accessibility).")
    ensure_harness("epoch-ai-evals", "Epoch AI evaluation harness (Inspect)", "eval-framework", url="https://epoch.ai/benchmarks",
                   sources=[SRC_RUNS], notes="Epoch AI's own runs; per-run logs are linked in original_url.")
    ensure_harness("helm", "Stanford HELM", "eval-framework", url="https://crfm.stanford.edu/helm/", sources=[SRC_EXT])
    ensure_harness("authors-internal", "Authors' own evaluation setup", "vendor-internal", sources=[SRC_EXT],
                   notes="Number reported in a paper/technical report, produced by the report authors' own (often lightly documented) setup.")

    meta_rows = bench["model_metadata.csv"]
    meta = {r["model_version"]: r for r in meta_rows if r["model_version"]}
    bmeta = {r["benchmark"]: r for r in bench["benchmark_metadata.csv"]}
    all_models = {r["Model"]: r for r in models_db.get("all_ai_models.csv", [])}
    params_by_name = {k: r["Parameters"] for k, r in all_models.items() if r.get("Parameters")}

    def model_key(version: str, row: dict) -> tuple[str, str | None]:
        base, effort = split_effort(version.strip())
        m = meta.get(version) or meta.get(base) or {}
        if base.startswith(HOSTING_PREFIXES) and m.get("model_group"):
            base = m["model_group"]
        return base, effort

    def claim(version: str, base: str, effort: str | None, row: dict) -> None:
        m = meta.get(version) or meta.get(split_effort(version)[0]) or {}
        acc = m.get("accessibility") or ""
        open_w = True if acc.startswith("Open weights") else (False if acc else None)
        name = (m.get("display_name") or "").strip()
        if effort and name.lower().endswith(f"({effort.lower()})"):
            name = name[: -len(effort) - 2].strip()
        group = m.get("model_group") or ""
        params = params_by_name.get(group) if open_w else None
        db = all_models.get(group) or all_models.get(base) or {}
        imp.claim(base, name=name or None, org=m.get("organization") or row.get("Organization") or db.get("Organization"),
                  country=m.get("country") or row.get("Country") or db.get("Country (of organization)"), release_date=_date(m.get("date") or row.get("Release date")),
                  open_weights=open_w, params_total_b=(float(params) / 1e9 if params else None), source_id=SRC_MODELS)

    used_variants: dict[str, set[str]] = defaultdict(set)

    def emit(cfg: dict, stem: str, row: dict, own: bool, rownum: int) -> None:
        version = (row.get("Model version") or "").strip()
        if not version:
            imp.skipped["row without model version"] += 1
            return
        base, effort = model_key(version, row)
        claim(version, base, effort, row)
        bench_id = cfg["bench"]
        bench_name = BENCHMARKS.get(bench_id, {}).get("name", bench_id)
        variant = _variant(cfg, row)
        if cfg.get("variant_fn") == "tbench" and variant is None:
            imp.skipped["terminal-bench row without version"] += 1
            return
        src = (row.get("Source") or "").strip()
        link = (row.get("Source link") or row.get("Source Link") or "").strip().split()[0] if (row.get("Source link") or row.get("Source Link") or "").strip() else ""
        if not link and _is_url(src):
            link, src = src, ""
        if own:
            link = (row.get("Log viewer") or "").strip()
            src = "Epoch AI run log"
        harness = "epoch-ai-evals" if own else _harness_for(cfg, row, bench_id, bench_name, src, link)
        eff = effort or ((row.get(cfg["effort_col"]) or "").strip() if cfg.get("effort_col") else None) or None
        settings = []
        if cfg.get("shots_col") and (row.get(cfg["shots_col"]) or "").strip():
            settings.append(f"n_shots={row[cfg['shots_col']].strip()}")
        for col, key in (cfg.get("setting_cols") or {}).items():
            if (row.get(col) or "").strip():
                settings.append(f"{key}={row[col].strip()}")
        date = _date(row.get(cfg.get("date_col") or ("Started at" if own else "")))
        notes = []
        listed = (row.get("Name") or "").strip()
        if listed and listed != version:
            notes.append(f"listed as '{listed}'")
        if (row.get("Notes") or "").strip():
            notes.append(row["Notes"].strip().replace("\n", " "))
        base_row = dict(harness_id=harness, harness_version=(row.get(cfg["version_col"]) or "").strip() or None if cfg.get("version_col") else None,
                        setting="; ".join(settings) or None, reasoning=reasoning_of(eff), reasoning_effort=eff, eval_date=date,
                        source_id=SRC_RUNS if own else SRC_EXT, locator=f"{stem}.csv id={row['id'].strip()}" if (row.get("id") or "").strip() else f"{stem}.csv row={rownum}",
                        original_source=src or None, original_url=link if link.startswith("http") else None,
                        notes="; ".join(notes) or None)
        unit = BENCHMARKS.get(bench_id, {}).get("unit", "percent")

        def scaled(v: float | None, percent_like: bool) -> float | None:
            if v is None:
                return None
            return v * 100 if percent_like and cfg["_fraction"] else v

        col = "Best score (across scorers)" if own else cfg["col"]
        value = scaled(_num(row.get(col)), unit == "percent")
        stderr = _num(row.get("stderr" if own else cfg.get("stderr_col", "")))
        if value is not None:
            used_variants[bench_id].add(variant or "")
            imp.score(base, benchmark_id=bench_id, variant=variant, metric=cfg.get("metric"), value=value,
                      stderr=scaled(stderr, unit == "percent"), **base_row)
        for xcol, (xvariant, xmetric) in (cfg.get("extra") or {}).items():
            xv = _num(row.get(xcol))
            if xv is None:
                continue
            percent_like = unit == "percent" and not (xmetric or "").endswith(("_usd", "_minutes", "error")) and "cost" not in (xmetric or "")
            if xmetric and xmetric.startswith("p80_horizon"):
                percent_like = False
            if percent_like and cfg["_fraction"]:
                xv *= 100
            v = xvariant or variant
            used_variants[bench_id].add(v or "")
            imp.score(base, benchmark_id=bench_id, variant=v, metric=xmetric, value=xv, **base_row)

    for stem, cfg in {**{k: {**v, "_own": True} for k, v in OWN.items()}, **{f"{k}_external": v for k, v in EXTERNAL.items()}}.items():
        rows = bench.get(f"{stem}.csv")
        if rows is None:
            print(f"[epoch] missing file {stem}.csv")
            continue
        own = bool(cfg.get("_own"))
        col = "Best score (across scorers)" if own else cfg["col"]
        vals = [v for v in (_num(r.get(col)) for r in rows) if v is not None]
        unit = BENCHMARKS.get(cfg["bench"], {}).get("unit", "percent")
        cfg = {**cfg, "_fraction": unit == "percent" and bool(vals) and max(vals) <= 1.0 and cfg["bench"] not in NO_DIRECT_SCALE}
        for rownum, row in enumerate(rows, start=2):
            emit(cfg, stem, row, own, rownum)

    # Benchmarks owned by this importer: (re)write with the variants actually seen + Epoch's release date.
    release = {}
    for name, r in bmeta.items():
        release[re.sub(r"[^a-z0-9]+", "", name.lower())] = r.get("release_date")
    for bid, spec in BENCHMARKS.items():
        path = DATA / "benchmarks" / f"{bid}.yaml"
        if path.exists():
            import yaml as _y
            if (_y.safe_load(path.read_text()) or {}).get("managed_by") != "epoch":
                continue
        intro = release.get(re.sub(r"[^a-z0-9]+", "", spec["name"].lower()))
        variants = spec.get("variants")
        seen = sorted(v for v in used_variants.get(bid, set()) if v)
        if not variants and seen:
            variants = [{"id": v} for v in seen]
        elif variants:
            declared = {v["id"] for v in variants}
            variants = variants + [{"id": v} for v in seen if v not in declared]
        rec = {"id": bid, "name": spec["name"], "category": spec["category"],
               "metric": spec.get("metric", "score" if spec.get("unit", "percent") != "percent" else "accuracy"),
               "unit": spec.get("unit", "percent"), "higher_is_better": spec.get("higher_is_better", True)}
        if spec.get("harness_sensitive"):
            rec["harness_sensitive"] = True
        if intro:
            rec["introduced"] = intro
        rec["url"] = spec["url"]
        if variants:
            rec["variants"] = variants
        rec["sources"] = [SRC_EXT]
        rec["managed_by"] = "epoch"
        dump_yaml(path, rec)
    return imp.finish()
