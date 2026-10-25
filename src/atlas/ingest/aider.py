"""Aider leaderboards: the raw benchmark YAMLs behind aider.chat (repo Aider-AI/aider, aider/website/_data/).

Each YAML entry is one benchmark run of the aider tool. Mapping:
  * refactor_leaderboard.yml -> `aider-refactoring` (89 large-method refactorings, single attempt -> pass_rate_1)
  * runs with ~225 test cases -> `aider-polyglot`
  * runs with ~133 test cases -> `aider-code-editing` (legacy Exercism Python benchmark)
Besides the leaderboards, the folder holds the data of individual blog posts (architect, quantization, Qwen3, ...);
they are imported too, each with its own source, even when an entry duplicates a leaderboard entry.

The `model` field is a display label. It is cleaned into a model name (settings such as "(high)", "(32k thinking)",
"(diff)", Ollama quant tags, architect "+ editor" suffixes move into reasoning_effort / setting / quant), and for the
blog-post files whose labels name a provider/config instead of a model (deepseek-down, quant, qwen3, r1_architect, qwq)
the model is taken from the post subject / the `--model` in the command. The original label is kept in `notes`.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

from .common import Importer, ensure_benchmark, ensure_harness, fetch_text, write_source

COMMIT = "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"
RAW = f"https://raw.githubusercontent.com/Aider-AI/aider/{COMMIT}/aider/website/_data/{{}}"
BLOB = f"https://github.com/Aider-AI/aider/blob/{COMMIT}/aider/website/_data/{{}}"

# file -> (source id, title, page/post the data backs)
FILES: dict[str, tuple[str, str, str]] = {
    "polyglot_leaderboard.yml": ("aider-polyglot-leaderboard-data",
                                 "Aider Polyglot leaderboard raw results (polyglot_leaderboard.yml)", ""),
    "edit_leaderboard.yml": ("aider-edit-leaderboard-data",
                             "Aider legacy code-editing leaderboard raw results (edit_leaderboard.yml)", ""),
    "refactor_leaderboard.yml": ("aider-refactor-leaderboard-data",
                                 "Aider refactoring leaderboard raw results (refactor_leaderboard.yml)",
                                 "https://aider.chat/docs/leaderboards/refactor.html"),
    "o1_polyglot_leaderboard.yml": ("aider-o1-polyglot-leaderboard-data",
                                    "Aider blog 'o1 tops aider's new polyglot leaderboard' raw results (o1_polyglot_leaderboard.yml)",
                                    "https://aider.chat/2024/12/21/polyglot.html"),
    "o1_results.yml": ("aider-o1-results-data",
                       "Aider blog 'o1-preview is SOTA on the aider leaderboard' raw results (o1_results.yml)",
                       "https://aider.chat/2024/09/12/o1.html"),
    "architect.yml": ("aider-architect-data",
                      "Aider blog 'Separating code reasoning and editing' raw results (architect.yml)",
                      "https://aider.chat/2024/09/26/architect.html"),
    "code-in-json.yml": ("aider-code-in-json-data",
                         "Aider blog 'LLMs are bad at returning code in JSON' raw results (code-in-json.yml)",
                         "https://aider.chat/2024/08/14/code-in-json.html"),
    "sonnet-fine.yml": ("aider-sonnet-fine-data",
                        "Aider blog 'Sonnet seems as good as ever' raw results (sonnet-fine.yml)",
                        "https://aider.chat/2024/08/26/sonnet-seems-fine.html"),
    "quant.yml": ("aider-quant-data",
                  "Aider blog 'Details matter with open source models' raw results (quant.yml)",
                  "https://aider.chat/2024/11/21/quantization.html"),
    "qwq.yml": ("aider-qwq-data",
                "Aider blog 'QwQ is a code architect, not an editor' raw results (qwq.yml)",
                "https://aider.chat/2024/12/03/qwq.html"),
    "r1_architect.yml": ("aider-r1-architect-data",
                         "Aider blog 'R1+Sonnet set SOTA on aider's polyglot benchmark' raw results (r1_architect.yml)",
                         "https://aider.chat/2025/01/24/r1-sonnet.html"),
    "deepseek-down.yml": ("aider-deepseek-down-data",
                          "Aider blog 'Alternative DeepSeek V3 providers' raw results (deepseek-down.yml)",
                          "https://aider.chat/2025/01/28/deepseek-down.html"),
    "qwen3_leaderboard.yml": ("aider-qwen3-leaderboard-data",
                              "Aider blog 'Qwen3 benchmark results' raw results (qwen3_leaderboard.yml)",
                              "https://aider.chat/2025/05/08/qwen3.html"),
}
# Not imported: blame.yml (aider's own contribution stats, no benchmark results).

FULL_CASES = {"aider-polyglot": 225, "aider-code-editing": 133, "aider-refactoring": 89}
# Default metric per benchmark (scores in that metric get an empty `metric`).
DEFAULT_METRIC = {"aider-polyglot": "pass_rate_2", "aider-code-editing": "pass_rate_2", "aider-refactoring": "pass_rate_1"}

SRC_REFACTOR_BLOG = "aider-unified-diffs-blog"

# Labels that are not model names (provider/config labels or aliases) -> model, per file.
LABEL_MODEL = {
    "r1_architect.yml": {"R1": "DeepSeek R1", "Sonnet": "claude-3-5-sonnet-20241022"},
    "qwq.yml": {"QwQ": "QwQ-32B-Preview", "Qwen2.5 Coder 32B-I": "Qwen2.5-Coder-32B-Instruct"},
}
FILE_MODEL = {  # every entry of these files benchmarks one model through different providers / quantizations
    "deepseek-down.yml": "DeepSeek V3",
    "quant.yml": "Qwen2.5-Coder-32B-Instruct",
}

EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}
EDIT_FORMATS = {"diff", "whole", "udiff", "diff-fenced"}
THINK_RE = re.compile(r"^(\d+k)\s+think(?:ing)?(?:\s+tokens)?$", re.I)
NO_THINK_RE = re.compile(r"^no\s+think(?:ing)?$", re.I)
QUANT_RE = re.compile(r"[-:](q\d+(?:_k)?(?:_[a-z0-9]+)?|fp16|bf16)(?:-(\d+k))?$", re.I)


def _effort_from_command(cmd: str) -> str | None:
    m = re.search(r"--reasoning-effort\s+(\S+)", cmd)
    if m:
        return m.group(1)
    m = re.search(r"--thinking-tokens\s+(\S+)", cmd)
    if m:
        return f"thinking_tokens={m.group(1)}"
    return None


def _parse_label(fname: str, label: str, e: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Display label -> (model name, extra fields: reasoning_effort, reasoning, quant, setting parts)."""
    extra: dict[str, Any] = {"setting": []}
    name = label.strip()
    if fname in FILE_MODEL:
        extra["setting"].append(f"provider={name}")
        if fname == "quant.yml":
            q = name.split(":", 1)[1].split(",")[0].strip() if ":" in name else ""
            if q and q.lower() not in ("bf16", "unknown", "multiple"):
                extra["quant"] = q
        return FILE_MODEL[fname], extra
    if name.startswith("Qwen3"):
        # e.g. 'Qwen3 235B A22B diff on OpenRouter, all providers, default settings (thinking)'
        m = re.match(r"^Qwen3[ -](\d+B)(?:[ -](A\d+B))?(?:\s+(?:diff|whole))?[\s,]*(.*)$", name)
        if m:
            model = "Qwen3-" + m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
            cfg = re.sub(r"^(on|with|via)\s+", "", m.group(3).strip())
            if cfg:
                extra["setting"].append(f"config={cfg}")
            if "no_think" in cfg or "no think" in cfg:
                extra["reasoning"] = "off"
            elif "(thinking)" in cfg:
                extra["reasoning"] = "on"
            q = re.search(r"\b(Q\d_K_[A-Z]|Q\d_\d)\b", cfg)
            if q:
                extra["quant"] = q.group(1)
            return model, extra
    if e.get("edit_format") == "architect" and "+" in name:
        name = name.split("+")[0].strip()
    name = LABEL_MODEL.get(fname, {}).get(name, name)
    while True:
        m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", name)
        if not m:
            break
        inner, low = m.group(2).strip(), m.group(2).strip().lower()
        if low in EFFORTS:
            extra["reasoning_effort"] = low
        elif THINK_RE.match(low):
            extra["reasoning_effort"] = f"thinking_tokens={THINK_RE.match(low).group(1)}"
            extra["reasoning"] = "on"
        elif NO_THINK_RE.match(low):
            extra["reasoning"] = "off"
        elif low in ("default", "default think"):
            extra["reasoning_effort"] = "default"
        elif low in ("reasoner", "chat"):  # DeepSeek-V3.2-Exp thinking / non-thinking mode
            extra["reasoning"] = "on" if low == "reasoner" else "off"
            extra["setting"].append(f"mode={low}")
        elif low in EDIT_FORMATS:
            pass  # edit_format is recorded from its own field
        elif low == "bf16":
            extra["setting"].append("precision=bf16")
        elif low == "deprecated":
            pass
        else:
            break  # part of the model name (dates, versions, ...)
        name = m.group(1).strip()
    q = QUANT_RE.search(name)
    if q:
        extra["quant"] = q.group(1)
        if q.group(2):
            extra["setting"].append(f"num_ctx={q.group(2)}")
        name = name[: q.start()]
        name = re.sub(r"-GGUF$", "", name, flags=re.I)
    return name, extra


def _benchmark(fname: str, e: dict[str, Any]) -> str:
    if fname == "refactor_leaderboard.yml":
        return "aider-refactoring"
    return "aider-polyglot" if int(e.get("test_cases") or 0) > 150 else "aider-code-editing"


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _sources() -> None:
    for fname, (sid, title, page) in FILES.items():
        notes = f"Data behind {page}" if page else None
        write_source(sid, kind="leaderboard_data", provenance="third_party", title=title, publisher="Aider-AI/aider",
                     url=BLOB.format(fname), revision=COMMIT, notes=notes)
    write_source(SRC_REFACTOR_BLOG, kind="blog", provenance="first_party",
                 title="Unified diffs make GPT-4 Turbo 3X less lazy", publisher="Aider",
                 url="https://aider.chat/2023/12/21/unified-diffs.html", published="2023-12-21",
                 notes="Introduces aider's refactoring benchmark: 89 large methods from large Python classes "
                       "(https://github.com/Aider-AI/refactor-benchmark).")
    ensure_benchmark(
        "aider-refactoring", name="Aider refactoring benchmark", category="code-editing", metric="pass_rate_1",
        unit="percent", higher_is_better=True, harness_sensitive=True, introduced="2023-12-21",
        url="https://aider.chat/docs/leaderboards/refactor.html",
        sources=[SRC_REFACTOR_BLOG, "aider-refactor-leaderboard-data"],
        description="Refactor 89 large methods from large Python classes in 9 open source repos "
                    "(https://github.com/Aider-AI/refactor-benchmark); built to measure 'lazy coding'. Single attempt.",
        notes="Record edit format in `setting`. Not the same benchmark as aider-code-editing or aider-polyglot.",
    )
    ensure_harness("aider", "Aider", "coding-assistant", url="https://aider.chat/",
                   sources=["aider-code-editing-benchmark", "aider-polyglot-announcement"])


def run() -> dict:
    _sources()
    imp = Importer("aider")
    for fname, (sid, _, _) in FILES.items():
        entries = yaml.safe_load(fetch_text(RAW.format(fname))) or []
        for e in entries:
            label = str(e.get("model") or "").strip()
            name, extra = _parse_label(fname, label, e)
            bench = _benchmark(fname, e)
            cmd = str(e.get("command") or "").strip()

            effort = None
            if e.get("reasoning_effort") not in (None, ""):
                effort = str(e["reasoning_effort"])
            elif e.get("thinking_tokens") not in (None, ""):
                effort = f"thinking_tokens={e['thinking_tokens']}"
            else:
                effort = _effort_from_command(cmd) or extra.get("reasoning_effort")
            reasoning = extra.get("reasoning")
            if e.get("thinking_tokens") is not None and not reasoning:
                reasoning = "on" if int(e["thinking_tokens"]) > 0 else "off"

            setting = [f"edit_format={e.get('edit_format')}"]
            if e.get("editor_model"):
                setting.append(f"editor_model={e['editor_model']}")
            if e.get("editor_edit_format"):
                setting.append(f"editor_edit_format={e['editor_edit_format']}")
            setting += extra["setting"]
            n, full = int(e.get("test_cases") or 0), FULL_CASES[bench]
            if n and n < 0.9 * full:
                setting.append(f"partial_run={n}/{full}")
            if fname == "code-in-json.yml":
                setting.append("attempts=1")

            notes = [f"aider label: {label}"] if label != name else []
            if n and n != full:
                notes.append(f"test_cases={n} of {full}")
            notes.append(f"command: {cmd}")
            if fname == "code-in-json.yml":
                notes.append("one of 5 repeated runs per model/format in the post")

            released = e.get("released")
            open_w = True if re.search(r"(ollama/|hf:|mlx-community/)", cmd) or re.search(r"VLLM|llama\.cpp", label) else None
            hf = re.search(r"hf:([\w.-]+/[\w.-]+)", cmd)
            imp.claim(name, release_date=str(released) if released else None, open_weights=open_w,
                      hf_repo=hf.group(1) if hf and fname not in FILE_MODEL else None, source_id=sid)

            date = e.get("date")
            base = dict(benchmark_id=bench, harness_id="aider", harness_version=str(e.get("versions") or "") or None,
                        setting="; ".join(setting), reasoning=reasoning, reasoning_effort=effort,
                        quant=extra.get("quant"), eval_date=str(date) if date else None, source_id=sid,
                        locator=f"{fname} dirname={e.get('dirname')}", notes="; ".join(notes))
            main = DEFAULT_METRIC[bench]
            if _num(e.get(main)) is None and _num(e.get("pass_rate_1")) is not None:
                main = "pass_rate_1"  # single-attempt runs (code-in-json)
            emitted = set()
            for metric in (main, "pass_rate_1", "percent_cases_well_formed"):
                v = _num(e.get(metric))
                if metric in emitted or v is None:
                    continue
                emitted.add(metric)
                imp.score(name, metric=None if metric == DEFAULT_METRIC[bench] else metric, value=v, **base)
            cost = _num(e.get("total_cost"))
            if cost is None or cost == 0:
                # 0 means "not tracked" (free/local endpoints, older runs), not a measured cost.
                imp.skipped["total_cost missing or 0"] += 1
            else:
                imp.score(name, metric="total_cost_usd", value=cost, **base)
    return imp.finish()
