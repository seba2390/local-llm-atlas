"""τ-bench leaderboard (taubench.com, Sierra), text submissions.

Data: web/leaderboard/public/submissions/<dir>/submission.json in github.com/sierra-research/tau2-bench (MIT),
listed in manifest.json under `submissions` (tau2-bench 1.x text) and `legacy_submissions` (original τ-bench and
tau2-bench 0.x). Voice submissions (τ-voice) are not imported.

* Original τ-bench (`tau2_bench_version: original tau-bench`) -> benchmark `tau-bench` (airline / retail).
* tau2-bench 0.x -> `tau2-bench` variants `<domain>-v0`; 1.x -> `<domain>-v1` (1.0.0 fixed 27 airline and 26 retail
  tasks); banking_knowledge -> `banking-knowledge` (>= 1.0.1 grading) or `banking-knowledge-pre-1.0.1`.
* Metric: pass^k (pass_1 is the default metric pass^1). Per-task cost is not imported.
* Self-reported results (submitting organisation == model organisation) carry the cited reference in original_url.
"""

from __future__ import annotations

import re

import yaml

from .common import DATA, Importer, dump_yaml, ensure_harness, fetch_json, write_source

NAME = "taubench"
REV = "0c69df456a96aa76a1162591574dad9c427a34d7"
RAW = f"https://raw.githubusercontent.com/sierra-research/tau2-bench/{REV}/web/leaderboard/public/submissions"
SRC = "taubench-leaderboard"
SRC_TAU1 = "arxiv-2406.12045"
SRC_TAU2 = "arxiv-2506.07982"
SRC_KNOW = "arxiv-2603.04370"
SAME_ORG = {("alibaba", "qwen"), ("alibaba cloud", "qwen")}

TAU1 = {
    "id": "tau-bench", "name": "τ-bench", "category": "tool-use", "metric": "pass^1", "unit": "percent",
    "higher_is_better": True, "harness_sensitive": True, "introduced": "2024-06-17",
    "url": "https://github.com/sierra-research/tau-bench",
    "variants": [{"id": "airline"}, {"id": "retail"}],
    "sources": [SRC_TAU1],
    "description": "Tool-agent-user interaction: an agent with domain APIs and a policy serves an LLM-simulated user. "
                   "pass^k = probability that all k i.i.d. trials of a task succeed.",
    "managed_by": NAME,
}
TAU2 = {
    "id": "tau2-bench", "name": "τ²-bench", "category": "tool-use", "metric": "pass^1", "unit": "percent",
    "higher_is_better": True, "harness_sensitive": True, "introduced": "2025-06-09",
    "url": "https://taubench.com/",
    "variants": [
        {"id": "airline-v0", "notes": "tau2-bench 0.x airline tasks."},
        {"id": "retail-v0", "notes": "tau2-bench 0.x retail tasks."},
        {"id": "telecom-v0", "notes": "tau2-bench 0.x telecom (dual-control) tasks."},
        {"id": "airline-v1", "notes": "tau2-bench 1.x; 27 airline tasks fixed in 1.0.0."},
        {"id": "retail-v1", "notes": "tau2-bench 1.x; 26 retail tasks fixed in 1.0.0."},
        {"id": "telecom-v1", "notes": "tau2-bench 1.x telecom."},
        {"id": "airline", "notes": "Airline domain; tau2-bench version not stated by the source."},
        {"id": "retail", "notes": "Retail domain; tau2-bench version not stated by the source."},
        {"id": "telecom", "notes": "Telecom domain; tau2-bench version not stated by the source."},
        {"id": "overall", "notes": "Single tau2-bench number (domain average as reported by the source; version not stated)."},
        {"id": "banking-knowledge", "introduced": "2026-03-04",
         "notes": "τ-Knowledge banking domain, tau2-bench >= 1.0.1 grading (not comparable with earlier grading)."},
        {"id": "banking-knowledge-pre-1.0.1", "introduced": "2026-03-04",
         "notes": "τ-Knowledge banking domain graded with tau2-bench < 1.0.1."},
    ],
    "sources": [SRC_TAU2, SRC_KNOW, SRC],
    "description": "τ-bench successor with a telecom dual-control domain (user also acts on the environment) and, from "
                   "2026, the τ-Knowledge banking domain. 2.0 (2026-10) changed scoring again (not comparable with 1.x).",
    "notes": "setting records the user simulator; the user-simulator model materially changes scores.",
    "managed_by": NAME,
}


def _own_benchmark(spec: dict) -> None:
    path = DATA / "benchmarks" / f"{spec['id']}.yaml"
    if path.exists() and (yaml.safe_load(path.read_text()) or {}).get("managed_by") != NAME:
        return
    dump_yaml(path, spec)


def _version_major(v: str | None) -> str | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", v or "")
    return m.group(1) if m else None


def run() -> dict[str, int]:
    imp = Importer(NAME)
    write_source(SRC, kind="leaderboard_data", provenance="third_party", title="τ-bench leaderboard submissions",
                 publisher="Sierra", url="https://github.com/sierra-research/tau2-bench/tree/main/web/leaderboard/public/submissions",
                 revision=REV, notes="MIT. Data behind https://taubench.com/ (manifest.json + one submission.json per entry). "
                 "Entries are run by Sierra or submitted by model developers/others; self-reported entries cite their reference.")
    write_source(SRC_TAU1, kind="paper", provenance="first_party",
                 title="τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains",
                 authors=["Shunyu Yao", "Noah Shinn", "Pedram Razavi", "Karthik Narasimhan"], publisher="arXiv",
                 url="https://arxiv.org/abs/2406.12045", published="2024-06-17", notes="Introduces τ-bench.")
    write_source(SRC_TAU2, kind="paper", provenance="first_party",
                 title="τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment",
                 authors=["Victor Barres", "Honghua Dong", "Soham Ray", "Xujie Si", "Karthik Narasimhan"], publisher="arXiv",
                 url="https://arxiv.org/abs/2506.07982", published="2025-06-09", notes="Introduces τ²-bench (telecom domain).")
    write_source(SRC_KNOW, kind="paper", provenance="first_party",
                 title="τ-Knowledge: Evaluating Conversational Agents over Unstructured Knowledge", publisher="arXiv",
                 url="https://arxiv.org/abs/2603.04370", published="2026-03-04", notes="Introduces the banking_knowledge domain.")
    ensure_harness("tau-bench", "τ-bench reference agent/user-simulator harness", "benchmark-reference",
                   url="https://github.com/sierra-research/tau-bench", sources=[SRC_TAU1],
                   notes="Original τ-bench code (tool-calling agent + LLM user simulator).")
    ensure_harness("tau2-bench", "τ²-bench reference harness (tau2 CLI)", "benchmark-reference",
                   url="https://github.com/sierra-research/tau2-bench", sources=[SRC_TAU2, SRC],
                   notes="Default τ²-bench agent scaffold + user simulator; harness_version = tau2-bench release. "
                         "'submission_type=custom' entries changed the benchmark-side setup (see notes).")
    _own_benchmark(TAU1)
    _own_benchmark(TAU2)

    manifest = fetch_json(f"{RAW}/manifest.json")
    for group in ("submissions", "legacy_submissions"):
        for sub in manifest.get(group, []):
            d = fetch_json(f"{RAW}/{sub}/submission.json")
            if (d.get("modality") or "text") != "text":
                imp.skipped["non-text submission"] += 1
                continue
            name = d["model_name"].strip()
            meth = d.get("methodology") or {}
            rel = d.get("model_release") or {}
            imp.claim(name, org=d.get("model_organization"), release_date=rel.get("release_date"), source_id=SRC)
            version = str(meth.get("tau2_bench_version") or "").strip()
            original = "original" in version.lower()
            major = _version_major(version)
            sub_org = (d.get("submitting_organization") or "").strip().lower()
            model_org = (d.get("model_organization") or "").strip().lower()
            self_reported = sub_org == model_org or (sub_org, model_org) in SAME_ORG
            refs = d.get("references") or []
            ref_url = next((r.get("url") for r in refs if (r.get("url") or "").startswith("http")), None)
            effort = d.get("reasoning_effort")
            reasoning = None if not effort else ("off" if str(effort).lower() == "none" else "on")
            setting = []
            if d.get("submission_type"):
                setting.append(f"submission_type={d['submission_type']}")
            if meth.get("user_simulator"):
                setting.append(f"user_simulator={meth['user_simulator']}")
            setting.append(f"submitted_by={d.get('submitting_organization')}")
            notes = []
            verif = meth.get("verification") or {}
            if verif.get("modified_prompts"):
                notes.append("modified prompts")
            if verif.get("details"):
                notes.append(verif["details"])
            if meth.get("notes"):
                notes.append(meth["notes"])
            date = meth.get("evaluation_date") or d.get("submission_date")
            bench_intro = TAU1["introduced"] if original else TAU2["introduced"]
            if date and str(date) < bench_intro:
                notes.insert(0, f"source gives evaluation_date {date}, before the benchmark existed; date omitted")
                date = None
            for domain, res in (d.get("results") or {}).items():
                if not res or all(res.get(f"pass_{k}") is None for k in (1, 2, 3, 4)):
                    continue
                if original:
                    bench, variant = "tau-bench", domain
                    if domain not in ("airline", "retail"):
                        imp.skipped[f"original tau-bench domain {domain}"] += 1
                        continue
                elif domain == "banking_knowledge":
                    bench = "tau2-bench"
                    vt = tuple(int(x) for x in re.findall(r"\d+", version)[:3]) if version else ()
                    variant = "banking-knowledge" if vt >= (1, 0, 1) else "banking-knowledge-pre-1.0.1"
                else:
                    bench = "tau2-bench"
                    variant = f"{domain}-v{'1' if major and int(major) >= 1 else '0'}"
                dsetting = list(setting)
                if res.get("retrieval_config"):
                    dsetting.append(f"retrieval_config={res['retrieval_config']}")
                for k in (1, 2, 3, 4):
                    v = res.get(f"pass_{k}")
                    if v is None:
                        continue
                    imp.score(name, benchmark_id=bench, variant=variant, metric=None if k == 1 else f"pass^{k}",
                              value=float(v), harness_id="tau-bench" if original else "tau2-bench",
                              harness_version=None if original else (version.lstrip("v") or None),
                              setting="; ".join(dsetting), reasoning=reasoning, reasoning_effort=effort,
                              eval_date=date, source_id=SRC,
                              locator=f"submissions/{sub}/submission.json results.{domain}.pass_{k}",
                              original_source=(f"self-reported by {d.get('submitting_organization')}" if self_reported and ref_url else None),
                              original_url=ref_url if self_reported else None,
                              notes=" | ".join(notes) or None)
    return imp.finish()
