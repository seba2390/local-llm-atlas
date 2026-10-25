"""Official SWE-bench leaderboards (swebench.com), from `data/leaderboards.json` in SWE-bench/swe-bench.github.io.

Every board in the file becomes a variant of benchmark `swe-bench` (Test -> full, Lite -> lite, Verified -> verified,
Multimodal -> multimodal, Multilingual -> multilingual). The former 'bash-only' board was merged into Verified upstream
(its mini-SWE-agent entries are Verified rows with harness `mini-swe-agent`).

Per submission:
  * harness  = the `agent` field (agent/system name); a trailing version ('-v1.5', ' 1.0') or a version label in the
               submission name ('(v20250405-dev)', '(2025-06-05)') goes to harness_version; mini-SWE-agent rows use
               `mini-swe-agent_version`.
  * model    = the API id in the 'Model: ...' tag when it agrees with `model_display`, otherwise `model_display`.
               Multi-model systems use the tag whose developer matches `model_org` as primary (all models in notes).
               Rows whose model cannot be determined (Undisclosed / no tags) are skipped and counted.
  * original_url = the submission folder (logs, trajectories, metadata) in SWE-bench/experiments.
"""

from __future__ import annotations

import re
from typing import Any

from .common import DATA, TODAY, Importer, dump_yaml, ensure_harness, fetch_json, slugify, write_source

SITE_REPO = "SWE-bench/swe-bench.github.io"
SITE_SHA = "193160a463a435d05cf44a1fa9dc5eac832113c3"  # master @ 2026-09-01 ("Drop bash-only board ...")
EXPERIMENTS_SHA = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"  # SWE-bench/experiments main @ 2026-09-03
DATA_URL = f"https://raw.githubusercontent.com/{SITE_REPO}/{SITE_SHA}/data/leaderboards.json"
SRC = "swe-bench-leaderboards"

BOARDS = {
    "Test": ("full", "test"),
    "Lite": ("lite", "lite"),
    "Verified": ("verified", "verified"),
    "Multimodal": ("multimodal", "multimodal"),
    "Multilingual": ("multilingual", "multilingual"),
}

# Agent names that refer to the same system under different spellings.
AGENT_ALIASES = {"Composio SWEkit": "Composio SWE-Kit", "RAG": "RAG baseline"}
KNOWN_HARNESSES: dict[str, tuple[str, str, str | None]] = {
    # family name -> (id, display name, url)
    "mini-SWE-agent": ("mini-swe-agent", "mini-SWE-agent", "https://github.com/SWE-agent/mini-swe-agent"),
    "SWE-agent": ("swe-agent", "SWE-agent", "https://github.com/SWE-agent/SWE-agent"),
    "OpenHands": ("openhands", "OpenHands", "https://github.com/All-Hands-AI/OpenHands"),
    "RAG baseline": ("swe-bench-rag-baseline", "SWE-bench RAG baseline (BM25 retrieval + single generation)",
                     "https://github.com/SWE-bench/SWE-bench"),
    "Tools": ("anthropic-tools", "Anthropic 'Tools' scaffold (bash + edit tools)", "https://www.anthropic.com/"),
    "Aider": ("aider", "Aider", "https://aider.chat/"),
}

NOT_A_MODEL = re.compile(r"^(undisclosed|multiple|mixed models|tts\(|\d+x scaled|swe-search|codeact v)", re.I)
MULTI_HINT = re.compile(r"&|\+")
VERSION_SUFFIX = re.compile(r"[\s_-]+(v\d+(?:\.\d+)*|\d+\.\d+(?:\.\d+)*)$", re.I)
HF_ORGS = {"qwen", "mistralai", "meta-llama", "zai-org", "agentica-org", "deepseek-ai", "moonshotai", "microsoft"}

def _override(board: str, r: dict) -> tuple[str, str] | None:
    """(raw model name, note) for submissions whose model fields are family-level or contradict the name."""
    name = r.get("name") or ""
    if r.get("agent") == "SWE-Fixer":
        return "Qwen2.5-72b", "editor model per submission name (Qwen2.5-7b retriever + Qwen2.5-72b editor)"
    if name.startswith("Nebius AI Qwen 2.5 72B Generator"):
        return "Qwen 2.5 72B", "generator model per submission name (critic: LLama 3.1 70B)"
    if r.get("agent") == "Aegis" and (r.get("model_display") or "") == "o3-mini_1.0":
        return "o3-mini", "model_display 'o3-mini_1.0'; model tag 'o3-mini'"
    return None


# ---------------------------------------------------------------- model helpers

ORG_PATTERNS = [
    (re.compile(r"swe-agent-lm", re.I), "swe-agent"),
    (re.compile(r"swe-llama", re.I), "princeton-nlp"),
    (re.compile(r"lingma", re.I), "alibaba"),
    (re.compile(r"skywork", re.I), "skywork"),
    (re.compile(r"deepswe", re.I), "agentica"),
    (re.compile(r"claude", re.I), "anthropic"),
    (re.compile(r"^(openai/)?(gpt|o\d|chatgpt|text-embedding|codex)", re.I), "openai"),
    (re.compile(r"gemini|gemma", re.I), "google"),
    (re.compile(r"deepseek", re.I), "deepseek"),
    (re.compile(r"qwen", re.I), "qwen"),
    (re.compile(r"llama", re.I), "meta"),
    (re.compile(r"kimi|moonshot", re.I), "moonshot"),
    (re.compile(r"glm", re.I), "z.ai"),
    (re.compile(r"devstral|mistral", re.I), "mistral"),
    (re.compile(r"doubao", re.I), "bytedance"),
    (re.compile(r"minimax", re.I), "minimax"),
]


def org_of(model: str) -> str | None:
    for rx, org in ORG_PATTERNS:
        if rx.search(model):
            return org
    return None


def same_org(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    a, b = a.lower(), b.lower()
    alias = {"google deepmind": "google", "z-ai": "z.ai", "zhipu": "z.ai", "moonshot ai": "moonshot", "alibaba": "qwen"}
    a, b = alias.get(a, a), alias.get(b, b)
    return a in b or b in a


def version_tokens(s: str) -> set[str]:
    """Version/size numbers in a model name, ignoring snapshot dates: 'claude-3-5-sonnet-20241022' -> {'3.5'}."""
    s = s.lower().replace("https://huggingface.co/", "")
    s = re.sub(r"(?<!\d)(?:19|20)\d{2}-?\d{2}-?\d{2}(?!\d)", " ", s)
    s = re.sub(r"(?<!\d)\d{4}(?!\d)", " ", s)
    s = re.sub(r"(?<=\d)-(?=\d(?!\d))", ".", s)
    return set(re.findall(r"\d+(?:\.\d+)?b?", s))


def tag_date(model: str) -> str | None:
    m = re.search(r"(?<!\d)((?:19|20)\d{2})-?(\d{2})-?(\d{2})(?!\d)", model)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def clean_tag(tag: str) -> tuple[str, str | None]:
    """'https://huggingface.co/Qwen/X' -> ('Qwen/X', 'Qwen/X'); returns (raw name, hf repo or None)."""
    t = tag.strip()
    if t.startswith("https://huggingface.co/"):
        repo = t[len("https://huggingface.co/"):].strip("/")
        return repo, repo
    if t.count("/") == 1 and t.split("/")[0].lower() in HF_ORGS:
        return t, t
    return t, None


def is_open(v: Any) -> bool:
    return v is True or str(v).lower() == "true"


def iso(yyyymmdd: Any) -> str | None:
    s = str(yyyymmdd or "")
    if re.fullmatch(r"\d{8}", s):
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    return None


def pick_model(board: str, r: dict) -> tuple[str | None, dict[str, Any], list[str], str | None]:
    """-> (raw name, claim fields, all models listed, note). raw name None = undeterminable."""
    tags: list[str] = []
    seen: set[str] = set()
    for t in r.get("tags") or []:
        if isinstance(t, str) and t.startswith("Model: "):
            v = t[len("Model: "):].strip()
            key = re.sub(r"[\s._:-]+", "-", v.lower())
            if v and key not in seen:
                seen.add(key)
                tags.append(v)
    display = (r.get("model_display") or "").strip()
    morg = r.get("model_org")
    date = r.get("date") or ""
    multi = display.lower() in ("multiple", "mixed models") or bool(MULTI_HINT.search(display)) or len(tags) > 1
    display_is_model = bool(display) and not NOT_A_MODEL.search(display) and not MULTI_HINT.search(display)
    note = None

    ov = _override(board, r)
    if ov:
        raw, note = ov
        return raw, {"name": None, "org": None}, tags or [display], note

    def tag_ok(t: str) -> bool:
        d = tag_date(t)
        return not (d and date and d > date)

    chosen: str | None = None
    if tags:
        cands = tags
        if display_is_model:
            # Prefer the tag that agrees with the human-curated display name.
            match = [t for t in tags if version_tokens(t) == version_tokens(display) and
                     (org_of(t) is None or org_of(display) is None or org_of(t) == org_of(display))]
            cands = match or ([] if len(tags) == 1 else tags)
        if cands:
            if len(cands) > 1 and morg:
                by_org = [t for t in cands if same_org(org_of(t), morg)]
                cands = by_org or cands
            cands = [t for t in cands if not re.search(r"embed", t, re.I)] or cands
            chosen = cands[0]
    if chosen and not tag_ok(chosen):
        # Snapshot date in the tag postdates the submission: keep the family, drop the impossible snapshot suffix.
        stripped = re.sub(r"[-_@]?(?:19|20)\d{2}-?\d{2}-?\d{2}$", "", chosen)
        note = f"model tag '{chosen}' postdates the submission; snapshot suffix dropped"
        chosen = stripped if not display_is_model else display
    if chosen is None:
        if display_is_model:
            chosen = display
        else:
            m = re.match(r"^CodeAct v[\d.]+ \((.+)\)$", display)
            chosen = m.group(1) if m else None
    if chosen is None:
        return None, {}, tags, None

    raw, hf = clean_tag(chosen)
    single = not multi
    fam = org_of(raw)
    org_ok = same_org(fam, morg) or (fam is None and single and display_is_model)
    claim = {
        "name": display if (single and display_is_model) else None,
        "org": morg if (morg and org_ok and str(morg).lower() != "multiple") else None,
        "release_date": iso(r.get("model_release_date")) if single else None,
        "open_weights": True if (single and is_open(r.get("os_model"))) else None,
        "hf_repo": hf,
    }
    if hf is None and single and display_is_model and re.match(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$", display):
        claim["hf_repo"] = display
    return raw, claim, tags or ([display] if display else []), note


# ---------------------------------------------------------------- harness helpers

def split_agent(agent: str) -> tuple[str, str | None]:
    agent = AGENT_ALIASES.get(agent.strip(), agent.strip())
    m = VERSION_SUFFIX.search(agent)
    if m and agent[: m.start()].strip():
        return agent[: m.start()].strip(), m.group(1).lstrip("vV")
    return agent, None


def name_version(name: str, agent: str) -> str | None:
    """Version label right after the agent in the submission name: 'Gru(2024-12-08)', 'X (v20250405-dev)', 'X v20250719'."""
    if not name.startswith(agent):
        return None
    rest = name[len(agent):]
    m = re.match(r"^\s*\((v?\d[\w.:-]*)\)", rest) or re.match(r"^\s+(v\d{6,}[\w.-]*)", rest)
    return m.group(1) if m else None


def first_url(site: Any) -> str | None:
    vals = site if isinstance(site, list) else [site]
    for v in vals:
        if isinstance(v, str) and v.strip():
            v = v.strip()
            if not v.startswith("http"):
                v = "https://" + v.lstrip("/")
            return v.split()[0]
    return None


def harness_for(agent: str, site_url: str | None, src: str, note: str) -> str:
    if agent.strip().lower() == "undisclosed":
        return "undisclosed"
    if agent in KNOWN_HARNESSES:
        hid, name, url = KNOWN_HARNESSES[agent]
    else:
        hid, name, url = slugify(agent), agent, site_url
    return ensure_harness(hid, name, "agent-scaffold", url=url, sources=[src], notes=note)


# ---------------------------------------------------------------- variants

def ensure_variants(bid: str, wanted: dict[str, dict[str, str]]) -> None:
    """Append missing variants to a hand-curated benchmark file without touching anything else."""
    import yaml

    path = DATA / "benchmarks" / f"{bid}.yaml"
    data = yaml.safe_load(path.read_text())
    have = {v["id"] for v in data.get("variants") or []}
    missing = [dict(id=k, **v) for k, v in wanted.items() if k not in have]
    if missing:
        data["variants"] = (data.get("variants") or []) + missing
        dump_yaml(path, data)


def ensure_source(sid: str, **fields: Any) -> str:
    """Create a citation once (shared ids such as arxiv-* may also be written by other importers or by hand)."""
    path = DATA / "sources" / f"{sid}.yaml"
    if not path.exists():
        dump_yaml(path, {"id": sid, **{k: v for k, v in fields.items() if v not in (None, "", [])}, "accessed": TODAY})
    return sid


# ---------------------------------------------------------------- run

def run() -> dict[str, int]:
    imp = Importer("swe-bench")
    data = fetch_json(DATA_URL)
    write_source(SRC, kind="leaderboard_data", provenance="third_party",
                 title="SWE-bench leaderboards (swebench.com) - data/leaderboards.json", publisher="SWE-bench team",
                 url=DATA_URL, revision=SITE_SHA,
                 notes="Boards Test (full), Lite, Verified, Multimodal, Multilingual. Submissions are made by agent developers; "
                 "'checked' = run performed or directly checked by the SWE-bench team. Per-submission logs/trajectories in "
                 f"SWE-bench/experiments@{EXPERIMENTS_SHA[:12]} (original_url).")
    ensure_source("arxiv-2410.03859", kind="paper", provenance="third_party",
                  title="SWE-bench Multimodal: Do AI Systems Generalize to Visual Software Domains?",
                  authors=["John Yang", "Carlos E. Jimenez", "et al."], publisher="arXiv",
                  url="https://arxiv.org/abs/2410.03859", published="2024-10-04")

    boards_seen = set()
    for lb in data["leaderboards"]:
        bname = lb["name"]
        if bname not in BOARDS:
            raise ValueError(f"unknown SWE-bench board {bname!r}: add it to BOARDS")
        variant, split = BOARDS[bname]
        boards_seen.add(bname)
        for r in lb["results"]:
            folder = r["folder"]
            agent_raw = (r.get("agent") or "").strip()
            raw, claim, models, mnote = pick_model(bname, r)
            if raw is None:
                imp.skipped["model undisclosed/undeterminable"] += 1
                continue
            family, ver = split_agent(agent_raw)
            site = first_url(r.get("site"))
            hid = harness_for(family, None if family == "mini-SWE-agent" else site, SRC,
                              "Agent/system name as listed on the official SWE-bench leaderboard (swebench.com).")
            if r.get("mini-swe-agent_version"):
                ver = str(r["mini-swe-agent_version"])
            ver = ver or name_version(r.get("name") or "", agent_raw)
            disp = (r.get("model_display") or "").strip()
            if disp.startswith("CodeAct v") and family == "OpenHands":
                ver = ver or disp.split(" (")[0]
            settings = []
            for t in r.get("tags") or []:
                m = re.match(r"System: Attempts - (\S+)", str(t))
                if m:
                    settings.append(f"attempts={m.group(1)}")
            effort = r.get("reasoning_effort")
            effort = str(effort) if effort not in (None, "", "None") else None
            notes = [f"listed as '{r.get('name')}'"]
            if len(models) > 1:
                notes.append("models: " + ", ".join(models) + f" (primary: {raw})")
            if disp and disp != raw and not (len(models) > 1):
                notes.append(f"model_display '{disp}'")
            if mnote:
                notes.append(mnote)
            if r.get("checked") is True:
                notes.append("checked by the SWE-bench team")
            if is_open(r.get("os_system")):
                notes.append("open-source system")
            if site:
                notes.append(f"site: {site}")
            imp.claim(raw, source_id=SRC, **{k: v for k, v in claim.items() if v is not None})
            imp.score(raw, benchmark_id="swe-bench", variant=variant, value=float(r["resolved"]), harness_id=hid,
                      harness_version=ver, setting="; ".join(settings) or None,
                      reasoning=("off" if effort and effort.lower() in ("none", "minimal-off") else "on") if effort else None,
                      reasoning_effort=effort, eval_date=r.get("date") or None, source_id=SRC,
                      locator=f"leaderboards.json {bname} folder={folder}",
                      original_source=f"SWE-bench/experiments evaluation/{split}/{folder}",
                      original_url=f"https://github.com/SWE-bench/experiments/tree/{EXPERIMENTS_SHA}/evaluation/{split}/{folder}",
                      notes="; ".join(notes))

    ensure_variants("swe-bench", {
        "multimodal": {"introduced": "2024-10-04",
                       "notes": "SWE-bench Multimodal (arXiv:2410.03859): JavaScript issues with visual elements; 'Multimodal' board on swebench.com."},
        "multilingual": {"notes": "SWE-bench Multilingual: 300 tasks from 42 repositories in 9 languages (https://www.swebench.com/multilingual.html); "
                                  "'Multilingual' board on swebench.com."},
    })
    return imp.finish()
