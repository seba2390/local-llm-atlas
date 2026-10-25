"""Berkeley Function Calling Leaderboard (BFCL), UC Berkeley Gorilla team.

Data: the leaderboard team's public result archive (github.com/HuanzhiMao/BFCL-Result, Apache-2.0), one folder per
leaderboard update holding the CSVs the site renders (data_overall.csv, data_non_live.csv, data_agentic.csv, ...),
plus the live https://gorilla.cs.berkeley.edu/data_overall.csv when it differs from every archived snapshot.

* BFCL version is inferred from the columns: Web Search/Memory -> v4, Multi Turn -> v3, otherwise v1
  (snapshots from the v2 period only hold per-model JSON, no aggregate tables, so there is no v2 data).
* The model label carries the calling mode, e.g. 'GPT-4o-2024-08-06 (FC)' / '(Prompt)' / '(FC thinking)';
  it is stripped from the raw name and recorded in `setting` (and reasoning=on for thinking modes).
* Every number of the same (label, version, metric) is kept once per distinct value: a value repeated unchanged in
  later snapshots is emitted only for the first snapshot that lists it (notes say until when it stayed listed).
* Cost/latency columns are not imported (not benchmark scores).
"""

from __future__ import annotations

import csv
import io
import re
from collections import OrderedDict

import yaml

from .common import DATA, Importer, dump_yaml, ensure_harness, fetch_text, write_source

NAME = "bfcl"
REV = "cadd76a8dc62984e7bf15ecaf358f070bae745cc"
RAW = f"https://raw.githubusercontent.com/HuanzhiMao/BFCL-Result/{REV}"
LIVE_CSV = "https://gorilla.cs.berkeley.edu/data_overall.csv"
LIVE_PAGE = "https://gorilla.cs.berkeley.edu/leaderboard.html"
SNAPSHOTS = [
    ("2024-07-06", "score-BFCL/data.csv"),
    *[(d, "score/data_overall.csv") for d in (
        "2024-10-04", "2024-10-21", "2024-11-13", "2024-12-06", "2024-12-29", "2025-01-18", "2025-02-09",
        "2025-02-11", "2025-02-27", "2025-03-17", "2025-03-22", "2025-04-13", "2025-04-25", "2025-06-14",
        "2025-08-26", "2025-11-03", "2025-12-16")],
]
# Extra per-category tables: only columns not already in data_overall.csv.
EXTRA = {
    "data_non_live.csv": {
        "Non_Live Overall Acc": "non_live_overall_acc", "Python Simple AST": "non_live_python_simple_ast",
        "Java Simple AST": "non_live_java_simple_ast", "JavaScript Simple AST": "non_live_javascript_simple_ast",
        "Python Simple Exec": "non_live_python_simple_exec", "REST Simple Exec": "non_live_rest_simple_exec",
    },
    "data_agentic.csv": {"Agentic Overall Acc": "agentic_overall_acc"},
}
SKIP_COLS = {"Rank", "Model", "Model Link", "Organization", "License"}
SKIP_PREFIX = ("Cost", "Total Cost", "Latency")
PP_COLS = {"Format Sensitivity Max Delta", "Format Sensitivity Standard Deviation"}

SRC_ARCHIVE = "bfcl-result-archive"
SRC_LIVE = "bfcl-leaderboard"
SRC_PAPER = "bfcl-icml-2025"

BENCH = {
    "id": "bfcl", "name": "Berkeley Function Calling Leaderboard (BFCL)", "category": "tool-use",
    "metric": "overall_acc", "unit": "percent", "higher_is_better": True, "introduced": "2024-02-26",
    "url": "https://gorilla.cs.berkeley.edu/leaderboard.html",
    "variants": [
        {"id": "v1", "introduced": "2024-02-26", "notes": "Single-turn AST + executable categories (Python/Java/JavaScript/REST)."},
        {"id": "v2", "introduced": "2024-08-19", "notes": "Adds user-contributed 'live' prompts."},
        {"id": "v3", "introduced": "2024-09-19", "notes": "Adds multi-turn/multi-step categories. Executable categories were "
         "retired from the overall score on 2025-04-09."},
        {"id": "v4", "introduced": "2025-07-17", "notes": "Adds agentic web-search and memory categories (overall re-weighted "
         "towards agentic tasks) and format-sensitivity analysis."},
    ],
    "sources": [SRC_PAPER, SRC_ARCHIVE],
    "description": "Function/tool-calling accuracy: AST and execution checks of single calls, relevance detection, "
                   "multi-turn tool use and (v4) agentic web search and memory.",
    "notes": "Metric ids are the leaderboard column names in snake_case (e.g. multi_turn_acc, live_simple_ast). "
             "format_sensitivity_* are spreads in percentage points across prompt formats (lower is better).",
    "managed_by": NAME,
}

MODE_RE = re.compile(r"\s*\(([^()]*)\)\s*$")


def _metric(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")


def _num(s: str | None) -> float | None:
    s = (s or "").strip().rstrip("%").strip()
    try:
        return float(s)
    except ValueError:
        return None


def split_label(label: str) -> tuple[str, list[str], str | None]:
    """'Command-R-Plus (FC) (Optimized)' -> ('Command-R-Plus', ['mode=FC', 'prompt=optimized'], None)."""
    groups: list[str] = []
    base = label.strip()
    while (m := MODE_RE.search(base)):
        groups.insert(0, m.group(1).strip())
        base = base[: m.start()].strip()
    setting: list[str] = []
    reasoning = None
    for g in groups:
        low = g.lower()
        if low.startswith(("fc", "prompt")):
            mode = "FC" if low.startswith("fc") else "Prompt"
            setting.append(f"mode={mode}")
            rest = g[len(mode):].strip().lstrip("+").strip()
            if rest.lower() == "thinking":
                setting.append("thinking=on")
                reasoning = "on"
            elif rest.lower() in ("any", "auto"):
                setting.append(f"tool_choice={rest.lower()}")
            elif rest.lower().startswith("tools-"):
                setting.append(f"api={rest}")
            elif rest:
                setting.append(f"variant={rest}")
        elif low in ("optimized", "original"):
            setting.append(f"prompt={low}")
        else:
            setting.append(f"provider={g}")
    return base, setting, reasoning


def _version(header: list[str]) -> str:
    if "Web Search Acc" in header or "Memory Acc" in header:
        return "v4"
    if "Multi Turn Acc" in header:
        return "v3"
    return "v1"


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def _write_benchmark() -> None:
    path = DATA / "benchmarks" / "bfcl.yaml"
    if path.exists() and (yaml.safe_load(path.read_text()) or {}).get("managed_by") != NAME:
        return
    dump_yaml(path, BENCH)


def run() -> dict[str, int]:
    imp = Importer(NAME)
    write_source(SRC_ARCHIVE, kind="leaderboard_data", provenance="third_party",
                 title="BFCL-Result: public evaluation result archive for the Berkeley Function Calling Leaderboard",
                 publisher="UC Berkeley Gorilla team (Huanzhi Mao)", url="https://github.com/HuanzhiMao/BFCL-Result",
                 revision=REV, notes="Apache-2.0. One folder per leaderboard update (folder name = update date) with the "
                 "score CSVs rendered on gorilla.cs.berkeley.edu/leaderboard.html. The live data_overall.csv was "
                 "byte-identical to the 2025-12-16 folder when checked on 2026-10-25.")
    write_source(SRC_PAPER, kind="paper", provenance="first_party",
                 title="The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models",
                 authors=["Shishir G. Patil", "Huanzhi Mao", "Fanjia Yan", "Charlie Cheng-Jie Ji", "Vishnu Suresh",
                          "Ion Stoica", "Joseph E. Gonzalez"],
                 publisher="PMLR (ICML 2025)", url="https://proceedings.mlr.press/v267/patil25a.html", published="2025-07",
                 notes="BFCL v1 blog: https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html "
                 "(2024-02-26); v2 live: blogs/12_bfcl_v2_live.html; v3 multi-turn: blogs/13_bfcl_v3_multi_turn.html "
                 "(2024-09-19); v4: blogs/15_bfcl_v4_web_search.html (2025-07-17).")
    ensure_harness("bfcl", "BFCL evaluation pipeline (bfcl-eval)", "eval-framework",
                   url="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard",
                   sources=[SRC_PAPER, SRC_ARCHIVE],
                   notes="Official Berkeley Function Calling Leaderboard generation + AST/executable checker. 'mode=FC' "
                         "uses the model's native function-calling API, 'mode=Prompt' describes tools in the prompt.")
    _write_benchmark()

    snapshots: list[tuple[str, str, str, dict[str, dict[str, str]], dict[str, dict[str, dict[str, str]]]]] = []
    archive_texts = set()
    for date, path in SNAPSHOTS:
        text = fetch_text(f"{RAW}/{date}/{path}")
        archive_texts.add(text)
        extras: dict[str, dict[str, dict[str, str]]] = {}
        if path.endswith("data_overall.csv"):
            for fname in EXTRA:
                try:
                    extras[fname] = {r["Model"]: r for r in _rows(fetch_text(f"{RAW}/{date}/score/{fname}"))}
                except Exception:  # file absent in older snapshots
                    pass
        snapshots.append((date, f"BFCL-Result/{date}/{path}", SRC_ARCHIVE, {r["Model"]: r for r in _rows(text)}, extras))

    live = fetch_text(LIVE_CSV)
    if live not in archive_texts:
        page = fetch_text(LIVE_PAGE)
        m = re.search(r"Last Updated:\s*(\d{4}-\d{2}-\d{2})", page)
        date = m.group(1) if m else None
        write_source(SRC_LIVE, kind="leaderboard", provenance="third_party", title="Berkeley Function Calling Leaderboard",
                     publisher="UC Berkeley Gorilla team", url=LIVE_PAGE, revision=f"Last Updated {date}" if date else None,
                     notes=f"Data file {LIVE_CSV}.")
        snapshots.append((date or "", "gorilla.cs.berkeley.edu/data_overall.csv", SRC_LIVE, {r["Model"]: r for r in _rows(live)}, {}))

    # (label, version, metric) -> OrderedDict(value -> [first snapshot info, last date])
    seen: dict[tuple[str, str, str], OrderedDict] = {}
    for date, loc, sid, rows, extras in snapshots:
        header = list(next(iter(rows.values())).keys()) if rows else []
        version = _version(header)
        has_exec = any("Exec" in h for h in header)
        for label, row in rows.items():
            base, setting, reasoning = split_label(label)
            lic = (row.get("License") or "").strip()
            link = (row.get("Model Link") or "").strip()
            hf = None
            if (hm := re.match(r"https?://huggingface\.co/([^/\s]+/[^/\s?#]+)", link)):
                hf = hm.group(1)
            imp.claim(base, org=(row.get("Organization") or "").strip() or None, license=lic or None,
                      open_weights=(lic.lower() != "proprietary") if lic else None, hf_repo=hf, source_id=sid)
            values: list[tuple[str | None, float, str | None]] = []
            for col, raw in row.items():
                if col in SKIP_COLS or col is None or col.startswith(SKIP_PREFIX):
                    continue
                v = _num(raw)
                if v is None:
                    continue
                metric = None if col == "Overall Acc" else _metric(col)
                note = None
                if col in PP_COLS:
                    note = "percentage points across prompt-format configurations; lower is better"
                elif col == "Overall Acc" and version == "v3":
                    note = ("overall includes the executable categories (retired 2025-04-09)" if has_exec
                            else "overall without executable categories")
                values.append((metric, v, note))
            for fname, cols in EXTRA.items():
                xrow = extras.get(fname, {}).get(label)
                for col, metric in cols.items():
                    if xrow and (v := _num(xrow.get(col))) is not None:
                        values.append((metric, v, None))
            for metric, v, note in values:
                key = (label, version, metric or "")
                hist = seen.setdefault(key, OrderedDict())
                if v in hist:
                    hist[v][1] = date
                    continue
                hist[v] = [dict(base=base, setting=setting, reasoning=reasoning, date=date, loc=loc, sid=sid,
                                note=note, version=version, metric=metric, label=label), date]

    for key, hist in seen.items():
        for v, (info, last) in hist.items():
            notes = [info["note"]] if info["note"] else []
            if last != info["date"]:
                notes.append(f"listed unchanged through the {last} snapshot")
            imp.score(info["base"], benchmark_id="bfcl", variant=info["version"], metric=info["metric"], value=v,
                      harness_id="bfcl", setting="; ".join(info["setting"]) or None, reasoning=info["reasoning"],
                      eval_date=info["date"] or None, source_id=info["sid"],
                      locator=f"{info['loc']} Model={info['label']}", notes="; ".join(notes) or None)
    return imp.finish()
