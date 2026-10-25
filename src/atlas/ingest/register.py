"""Create/refresh model records from claims staged by importers (.cache/staging/*.csv)."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from typing import Any


import yaml

from ..catalog import load
from ..resolve import keys_for, norm, strip_provider
from .common import DATA, STAGING, dump_yaml, slugify

# Organisation names as written by sources -> atlas org slug (folder under data/models/).
ORG_SLUGS = {
    "openai": "openai", "anthropic": "anthropic", "alibaba": "alibaba", "qwen": "alibaba", "alibaba cloud": "alibaba",
    "google deepmind": "google", "google": "google", "deepmind": "google", "google research": "google", "google brain": "google",
    "deepseek": "deepseek", "deepseek-ai": "deepseek", "meta ai": "meta", "meta": "meta", "facebook": "meta", "meta-llama": "meta",
    "mistral ai": "mistral", "mistralai": "mistral", "mistral": "mistral", "xai": "xai", "x.ai": "xai",
    "z.ai (zhipu ai)": "zhipu", "zhipu ai": "zhipu", "z.ai": "zhipu", "zai-org": "zhipu", "thudm": "zhipu",
    "moonshot": "moonshot", "moonshot ai": "moonshot", "moonshotai": "moonshot",
    "microsoft": "microsoft", "microsoft research": "microsoft", "nvidia": "nvidia", "01.ai": "01-ai", "01-ai": "01-ai",
    "amazon": "amazon", "minimax": "minimax", "minimaxai": "minimax", "xiaomi corp": "xiaomi", "xiaomi": "xiaomi",
    "technology innovation institute": "tii", "tiiuae": "tii", "baichuan": "baichuan", "hugging face": "huggingface",
    "huggingfacetb": "huggingface", "mosaicml": "mosaicml", "stepfun": "stepfun", "cohere": "cohere", "cohereforai": "cohere",
    "coherelabs": "cohere", "databricks": "databricks", "eleutherai": "eleutherai", "inception labs": "inception",
    "bytedance": "bytedance", "bytedance seed": "bytedance", "tsinghua university": "tsinghua", "shanghai ai lab": "shanghai-ai-lab",
    "shanghai ai laboratory": "shanghai-ai-lab", "internlm": "shanghai-ai-lab", "allen institute for ai": "allenai",
    "allenai": "allenai", "ai2": "allenai", "salesforce research": "salesforce", "salesforce": "salesforce", "ibm": "ibm",
    "ibm-granite": "ibm", "tencent": "tencent", "lg ai research": "lg", "lgai-exaone": "lg", "upstage": "upstage",
    "stability ai": "stability", "stabilityai": "stability", "bigcode": "bigcode", "huggingface": "huggingface",
    "nous research": "nous", "nousresearch": "nous", "ant group": "ant-group", "inclusionai": "ant-group",
    "large model systems organization": "lmsys", "lmsys": "lmsys", "baidu": "baidu", "ai21 labs": "ai21", "ai21labs": "ai21",
    "reka ai": "reka", "perplexity": "perplexity", "arcee ai": "arcee", "deep cogito": "deep-cogito", "cerebras systems": "cerebras",
    "prime intellect": "prime-intellect", "thinking machines": "thinking-machines", "cognition": "cognition", "cursor": "cursor",
    "poolside": "poolside", "inflection ai": "inflection", "snowflake": "snowflake", "servicenow": "servicenow",
    "together": "together", "togethercomputer": "together", "codellama": "meta", "wizardlmteam": "microsoft",
    "baichuan-inc": "baichuan", "xai-org": "xai", "bytedance-seed": "bytedance", "xiaomimimo": "xiaomi",
    "stepfun-ai": "stepfun", "opengvlab": "shanghai-ai-lab", "tencent-hunyuan": "tencent", "ai21": "ai21",
    "servicenow-ai": "servicenow", "snowflake ai research": "snowflake", "arcee-ai": "arcee", "deepcogito": "deep-cogito",
    "primeintellect": "prime-intellect", "rhymes-ai": "rhymes", "mistral-community": "mistral", "openbmb": "openbmb",
    "huggingfaceh4": "huggingface", "zhipu": "zhipu", "meta platforms": "meta", "apple": "apple", "amazon nova": "amazon", "ibm research": "ibm",
}

# Brand prefixes (start of model id) that unambiguously identify the developer. Used only when no source states it;
# such records get a note saying the organisation was inferred from the name.
BRAND_PREFIXES = [
    ("claude", "anthropic"), ("chatgpt", "openai"), ("gpt-", "openai"), ("o1-", "openai"), ("o3-", "openai"), ("o4-", "openai"),
    ("codex", "openai"), ("computer-use-preview", "openai"), ("davinci", "openai"), ("text-davinci", "openai"),
    ("gemini", "google"), ("gemma", "google"), ("palm", "google"), ("learnlm", "google"), ("codegemma", "google"),
    ("llama-", "meta"), ("llama3", "meta"), ("codellama", "meta"), ("code-llama", "meta"), ("opt-", "meta"),
    ("qwen", "alibaba"), ("qwq", "alibaba"), ("codeqwen", "alibaba"), ("mistral", "mistral"), ("mixtral", "mistral"),
    ("codestral", "mistral"), ("devstral", "mistral"), ("magistral", "mistral"), ("ministral", "mistral"), ("pixtral", "mistral"),
    ("deepseek", "deepseek"), ("grok", "xai"), ("kimi", "moonshot"), ("glm-", "zhipu"), ("chatglm", "zhipu"), ("codegeex", "zhipu"),
    ("phi-", "microsoft"), ("granite", "ibm"), ("nemotron", "nvidia"), ("nvidia-", "nvidia"), ("internlm", "shanghai-ai-lab"),
    ("internvl", "shanghai-ai-lab"), ("baichuan", "baichuan"), ("minimax", "minimax"), ("c4ai-command", "cohere"),
    ("command-", "cohere"), ("aya-", "cohere"), ("sonar", "perplexity"), ("falcon", "tii"), ("yi-", "01-ai"),
    ("seed-", "bytedance"), ("doubao", "bytedance"), ("ernie", "baidu"), ("hunyuan", "tencent"), ("step-", "stepfun"),
    ("mimo", "xiaomi"), ("olmo", "allenai"), ("starcoder", "bigcode"), ("amazon-nova", "amazon"), ("nova-", "amazon"),
    ("jamba", "ai21"), ("exaone", "lg"), ("solar", "upstage"), ("ling-", "ant-group"), ("ring-", "ant-group"),
]


def brand_org(model_id: str) -> str | None:
    if "llava" in model_id:  # LLaVA-style finetunes are academic projects built on others' LLMs
        return None
    return next((org for prefix, org in BRAND_PREFIXES if model_id.startswith(prefix)), None)


# Importers whose claims win when several sources describe the same model.
PRIORITY = ["manual", "hf", "epoch", "aider", "swe-bench", "evalplus", "livecodebench"]


INFERRED_NOTE = "Organisation inferred from the model's brand name; no imported source states the developer."


def org_slug(org: str) -> str:
    first = org.split(",")[0].strip()
    return ORG_SLUGS.get(first.lower(), slugify(first) or "unknown")


def _read_claims() -> list[dict[str, str]]:
    claims = []
    for path in sorted(STAGING.glob("*.csv")):
        with path.open(newline="") as fh:
            claims += list(csv.DictReader(fh))
    rank = {name: i for i, name in enumerate(PRIORITY)}
    claims.sort(key=lambda c: rank.get(c["importer"], len(PRIORITY)))
    return claims


_NAME_TAG = re.compile(
    r"\s*\((?:[^)]*\bthinking\b[^)]*|no thinking|together|novita|fireworks|deepinfra|chutes|web|web app)\)\s*$", re.I)


def clean_name(name: str) -> str:
    """Drop serving-provider / reasoning-effort tags from display names; those belong in score rows, not the model."""
    name = re.sub(r"\s*\(pro, [^)]*\)\s*$", " Pro", name)
    prev = None
    while prev != name:
        prev, name = name, _NAME_TAG.sub("", name).strip()
    return name


HF_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def _derive(claims: list[dict[str, str]]) -> dict[str, tuple[Any, str]]:
    """field -> (value, source_id), taking the highest-priority claim that states it."""
    out: dict[str, tuple[Any, str]] = {}
    for c in claims:
        for f in ("name", "org", "release_date", "open_weights", "hf_repo", "params_total_b", "license", "context_length"):
            if c.get(f) and f not in out:
                out[f] = (clean_name(c[f]) if f == "name" else c[f], c.get("source_id", ""))
    if "hf_repo" in out and not HF_REPO_RE.match(out["hf_repo"][0]):
        del out["hf_repo"]  # e.g. a leaderboard display name that merely contains a '/'
    if "open_weights" in out:
        out["open_weights"] = ({"true": True, "false": False}.get(out["open_weights"][0]), out["open_weights"][1])
        if out["open_weights"][0] is False:
            out.pop("hf_repo", None)
    if "context_length" in out:
        out["context_length"] = (int(float(out["context_length"][0])), out["context_length"][1])
    if "params_total_b" in out:
        out["params_total_b"] = (round(float(out["params_total_b"][0]), 3), out["params_total_b"][1])
    if "org" in out:
        out["org"] = (org_slug(out["org"][0]), out["org"][1])
    elif "hf_repo" in out:
        out["org"] = (org_slug(out["hf_repo"][0].split("/")[0]), out["hf_repo"][1])
    return out


def _cite(rec: dict[str, Any], field: str, src: str) -> None:
    if not src:
        return
    if src not in rec["sources"]:
        rec["sources"].append(src)
    if src != rec["sources"][0]:
        rec.setdefault("field_sources", {})[field] = src


def _apply(rec: dict[str, Any], derived: dict[str, tuple[Any, str]]) -> dict[str, Any]:
    """Fill fields that are missing in rec from derived claims (existing values always win)."""
    rec = {**rec, "sources": list(rec.get("sources", [])), "field_sources": dict(rec.get("field_sources", {}))}
    for field in ("release_date", "license", "context_length"):
        if field not in rec and field in derived:
            rec[field] = derived[field][0]
            _cite(rec, field, derived[field][1])
    if rec.get("open_weights") is None and derived.get("open_weights", (None,))[0] is not None:
        rec["open_weights"] = derived["open_weights"][0]
        _cite(rec, "open_weights", derived["open_weights"][1])
    if "hf_repo" not in rec and "hf_repo" in derived:
        rec["hf_repo"] = derived["hf_repo"][0]
        rec["weights_url"] = f"https://huggingface.co/{rec['hf_repo']}"
        _cite(rec, "hf_repo", derived["hf_repo"][1])
    # Parameter counts only for open-weight models (closed-model figures are third-party estimates).
    if rec.get("open_weights") and "params_total_b" in derived and "params_total_b" not in (rec.get("architecture") or {}):
        rec["architecture"] = {**(rec.get("architecture") or {}), "params_total_b": derived["params_total_b"][0]}
        _cite(rec, "architecture.params_total_b", derived["params_total_b"][1])
    if not rec["field_sources"]:
        rec.pop("field_sources")
    return rec


ORDER = ["id", "name", "aliases", "org", "company", "country", "other_orgs", "family", "variant", "base_model", "release_date", "open_weights", "license",
         "hf_repo", "weights_url", "api_ids", "context_length", "modalities", "architecture", "sources", "field_sources",
         "notes", "managed_by"]


def _ordered(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: rec[k] for k in ORDER if k in rec} | {k: v for k, v in rec.items() if k not in ORDER}


def register(dry_run: bool = False) -> dict[str, int]:
    from ..catalog import Record

    from ..resolve import Resolver

    cat = load(DATA)
    models = dict(cat.entities.get("models", {}))
    resolver = Resolver(r.data for r in models.values())

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for c in _read_claims():
        groups[norm(strip_provider(c["raw_name"]))].append(c)

    created = updated = ambiguous = clashes = 0
    for key, claims in groups.items():
        hits = {h for h in (resolver.resolve(c["raw_name"]) for c in claims) if h}
        if len(hits) > 1:
            ambiguous += 1
            continue
        derived = _derive(claims)
        if hits:
            mid = next(iter(hits))
            rec = models[mid]
            if rec.data.get("managed_by") != "register":
                continue
            new = dict(rec.data)
            new["aliases"] = sorted(set(new.get("aliases", [])) | ({c["raw_name"] for c in claims} - {mid}))
            if not new["aliases"]:
                new.pop("aliases")
            path = rec.path
            if (new.get("org") == "unknown" or new.get("notes", "").startswith(INFERRED_NOTE)) and derived.get("org", ("unknown",))[0] != "unknown":
                new["org"] = derived["org"][0]
                if new.get("notes", "").startswith(INFERRED_NOTE):
                    new.pop("notes")
                for k in ("company", "country"):
                    new.pop(k, None)
                _cite(new, "org", derived["org"][1])
                path = DATA / "models" / new["org"] / f"{mid}.yaml"
            elif new.get("org") == "unknown" and brand_org(mid):
                new["org"] = brand_org(mid)
                new["notes"] = INFERRED_NOTE
                for k in ("company", "country"):
                    new.pop(k, None)
                path = DATA / "models" / new["org"] / f"{mid}.yaml"
            new = _ordered(_apply(new, derived))
            if new != rec.data or path != rec.path:
                if not dry_run:
                    dump_yaml(path, new)
                    if path != rec.path:
                        rec.path.unlink()
                models[mid] = Record(data=new, path=path)
                resolver = Resolver(r.data for r in models.values())
                updated += 1
            continue
        mid = slugify(strip_provider(claims[0]["raw_name"]))
        if not re.match(r"^[a-z0-9]", mid or "") or mid in models:
            clashes += 1
            continue
        first_src = next((c["source_id"] for c in claims if c.get("source_id")), None)
        if not first_src:
            continue
        name = derived.get("name", (strip_provider(claims[0]["raw_name"]),))[0]
        rec = {
            "id": mid,
            "name": name,
            "aliases": sorted({c["raw_name"] for c in claims} - {mid}),
            "org": derived.get("org", ("unknown",))[0],
            "open_weights": None,
            "sources": [first_src],
            "managed_by": "register",
        }
        rec = _ordered(_apply(rec, derived))
        if not rec["aliases"]:
            rec.pop("aliases")
        if rec["org"] == "unknown" and brand_org(mid):
            rec["org"] = brand_org(mid)
            rec["notes"] = INFERRED_NOTE
        path = DATA / "models" / rec["org"] / f"{mid}.yaml"
        if not dry_run:
            dump_yaml(path, rec)
        created += 1
        models[mid] = Record(data=rec, path=path)
        resolver = Resolver(r.data for r in models.values())
    stats = {"created": created, "updated": updated, "ambiguous_groups": ambiguous, "id_clashes": clashes}
    print("[register] " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    return stats


def register_orgs(dry_run: bool = False) -> dict[str, int]:
    """Create org records for every org slug that claims mention (name + country, cited to the claiming source)."""
    by_slug: dict[str, list[dict[str, str]]] = defaultdict(list)
    for c in _read_claims():
        if c.get("org"):
            by_slug[org_slug(c["org"])].append(c)
    created = updated = 0
    for slug, claims in by_slug.items():
        path = DATA / "orgs" / f"{slug}.yaml"
        names = [c["org"].split(",")[0].strip() for c in claims]
        country = next(((c["country"].split(",")[0].strip(), c["source_id"]) for c in claims if c.get("country")), None)
        if path.exists():
            rec = yaml.safe_load(path.read_text())
            if rec.get("managed_by") != "register":
                continue
            new = dict(rec)
            new["aliases"] = sorted(set(rec.get("aliases", [])) | (set(names) - {rec["name"]}))
            if not new["aliases"]:
                new.pop("aliases")
            if "country" not in new and country:
                new["country"] = country[0]
                if country[1] not in new["sources"]:
                    new["sources"].append(country[1])
                if country[1] != new["sources"][0]:
                    new.setdefault("field_sources", {})["country"] = country[1]
            if new != rec:
                updated += 1
                if not dry_run:
                    dump_yaml(path, new)
            continue
        name = max(set(names), key=names.count)
        src = country[1] if country else claims[0]["source_id"]
        rec = {"id": slug, "name": name}
        aliases = sorted(set(names) - {name})
        if aliases:
            rec["aliases"] = aliases
        if country:
            rec["country"] = country[0]
        rec["sources"] = [src]
        rec["managed_by"] = "register"
        created += 1
        if not dry_run:
            dump_yaml(path, rec)
    # Orgs referenced by models (e.g. derived from a Hugging Face account) that no source describes yet.
    cat = load(DATA)
    known = set(cat.entities.get("orgs", {}))
    pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in cat.entities.get("models", {}).values():
        slug = rec.data.get("org")
        if slug and slug not in known and not (DATA / "orgs" / f"{slug}.yaml").exists():
            pending[slug].append(rec.data)
    for slug, models in pending.items():
        authors = [m["hf_repo"].split("/")[0] for m in models if m.get("hf_repo")]
        name = max(set(authors), key=authors.count) if authors else slug
        src = next((m["sources"][0] for m in models if m.get("hf_repo") and m.get("sources")), models[0]["sources"][0])
        rec = {"id": slug, "name": name, "sources": [src], "managed_by": "register",
               "notes": "Created from the Hugging Face account that publishes the weights; country not stated by any imported source."}
        created += 1
        if not dry_run:
            dump_yaml(DATA / "orgs" / f"{slug}.yaml", rec)
    print(f"[register-orgs] created={created} updated={updated}")
    return {"created": created, "updated": updated}


def sync_orgs() -> int:
    """Copy company name and country from data/orgs/<org>.yaml into every model record."""
    cat = load(DATA)
    orgs = cat.entities.get("orgs", {})
    changed = 0
    for rec in cat.entities.get("models", {}).values():
        org = orgs.get(rec.data.get("org", ""))
        if org is None:
            continue
        new = {k: v for k, v in rec.data.items() if k not in ("company", "country")}
        new["company"] = org.data["name"]
        if org.data.get("country"):
            new["country"] = org.data["country"]
        new = _ordered(new)
        if new != rec.data:
            # Hand-curated files are rewritten only to add/refresh these two derived fields.
            text = rec.path.read_text()
            if rec.data.get("managed_by") == "register":
                dump_yaml(rec.path, new)
            else:
                lines = [l for l in text.splitlines() if not l.startswith(("company:", "country:"))]
                i = next(i for i, l in enumerate(lines) if l.startswith("org:")) + 1
                extra = [f"company: {yaml.safe_dump(new['company']).splitlines()[0]}"]
                if "country" in new:
                    extra.append(f"country: {yaml.safe_dump(new['country']).splitlines()[0]}")
                rec.path.write_text("\n".join(lines[:i] + extra + lines[i:]) + "\n")
            changed += 1
    print(f"[sync-orgs] updated {changed} model records")
    return changed


def merge_duplicates(dry_run: bool = False) -> int:
    """Merge managed model records that share a strong key (id/alias/HF repo): same checkpoint registered twice."""
    from ..resolve import Resolver, strip_provider as _tail

    cat = load(DATA)
    models = cat.entities.get("models", {})
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for ids in Resolver.from_catalog(cat).ambiguous().values():
        ids = sorted(ids)
        for i in ids[1:]:
            parent[find(i)] = find(ids[0])
    groups: dict[str, set[str]] = defaultdict(set)
    for x in list(parent):
        groups[find(x)].add(x)

    merged = 0
    for group in groups.values():
        recs = [models[i] for i in sorted(group)]
        if any(r.data.get("managed_by") != "register" for r in recs):
            hand = [r for r in recs if r.data.get("managed_by") != "register"]
            if len(hand) > 1:
                continue  # two hand-curated records collide: needs a human
            canon = hand[0]
        else:
            hf_ids = {slugify(_tail(r.data["hf_repo"])) for r in recs if r.data.get("hf_repo")}
            canon = next((r for r in recs if r.data["id"] in hf_ids), None) or max(
                recs, key=lambda r: (bool(r.data.get("release_date")), len(r.data.get("sources", [])), -len(r.data["id"])))
        new = dict(canon.data)
        new["sources"] = list(new.get("sources", []))
        aliases = set(new.get("aliases", []))
        for r in recs:
            if r is canon:
                continue
            d = r.data
            aliases |= {d["id"], *d.get("aliases", [])}
            if d.get("name") and d["name"] != new.get("name"):
                aliases.add(d["name"])
            src = (d.get("sources") or [None])[0]
            fsrc = d.get("field_sources") or {}
            for field in ("release_date", "license", "context_length", "hf_repo", "weights_url", "family", "variant"):
                if field not in new and field in d:
                    new[field] = d[field]
                    _cite(new, field, fsrc.get(field, src))
            if new.get("open_weights") is None and d.get("open_weights") is not None:
                new["open_weights"] = d["open_weights"]
                _cite(new, "open_weights", fsrc.get("open_weights", src))
            arch = dict(new.get("architecture") or {})
            for k, v in (d.get("architecture") or {}).items():
                if k not in arch:
                    arch[k] = v
                    _cite(new, f"architecture.{k}", fsrc.get(f"architecture.{k}", src))
            if arch:
                new["architecture"] = arch
            for s in d.get("sources", []):
                if s not in new["sources"]:
                    new["sources"].append(s)
        aliases.discard(new["id"])
        if aliases:
            new["aliases"] = sorted(aliases)
        if new.get("managed_by") == "register":
            new = _ordered(new)
            if not dry_run:
                dump_yaml(canon.path, new)
        elif not dry_run:
            # hand-curated canonical record: only extend its aliases line
            text = canon.path.read_text()
            lines = [l for l in text.splitlines() if not l.startswith("aliases:")]
            i = next(i for i, l in enumerate(lines) if l.startswith("name:")) + 1
            lines.insert(i, "aliases: " + json.dumps(new["aliases"]))
            canon.path.write_text("\n".join(lines) + "\n")
        for r in recs:
            if r is not canon and not dry_run:
                r.path.unlink()
        merged += len(recs) - 1
    print(f"[merge-duplicates] merged {merged} duplicate records into {len(groups)} models")
    return merged
