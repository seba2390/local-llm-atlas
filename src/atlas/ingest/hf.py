"""Hugging Face Hub: find official repos for open-weight models and record what the repo itself states.

For every atlas model that is (or may be) open-weight:
  * locate the official repo (exact name match under an official developer org; never re-uploads),
  * cite it (`hf-<org>-<name>`, pinned to the repo sha),
  * claim hf_repo / org / license / context length / total params (safetensors metadata),
  * fill architecture fields from config.json (dense vs MoE vs hybrid, layers, heads, experts, dtype),
  * record weight artifacts with exact on-disk bytes: the original safetensors and GGUF quants from
    well-known quantizers (bartowski, unsloth, lmstudio-community, ggml-org).
"""

from __future__ import annotations

import json
import re
import urllib.error
from collections import defaultdict
from typing import Any
from urllib.parse import quote

import yaml

from ..catalog import load
from ..resolve import norm, strip_provider
from .common import DATA, Importer, dump_yaml, fetch_json, slugify, write_source
from .register import ORG_SLUGS

API = "https://huggingface.co/api"
OFFICIAL_HF_ORGS = {
    "qwen", "meta-llama", "google", "mistralai", "deepseek-ai", "microsoft", "nvidia", "01-ai", "internlm", "tiiuae",
    "bigcode", "eleutherai", "allenai", "ibm-granite", "ibm", "cohereforai", "coherelabs", "thudm", "zai-org", "stabilityai",
    "salesforce", "lgai-exaone", "upstage", "huggingfacetb", "databricks", "mosaicml", "togethercomputer", "moonshotai",
    "minimaxai", "openai", "xai-org", "baichuan-inc", "bytedance-seed", "tencent", "baidu", "xiaomimimo", "stepfun-ai",
    "inclusionai", "facebook", "codellama", "ai21labs", "snowflake", "servicenow-ai", "nousresearch", "openbmb",
    "opengvlab", "rhymes-ai", "apple", "amazon", "arcee-ai", "deepcogito", "swiss-ai", "nexusflow", "wizardlmteam",
    "mistral-community", "openchat", "lmsys", "hkunlp", "agentica-org", "all-hands", "prime-intellect", "primeintellect",
    "poolside", "cerebras", "ByteDance".lower(), "meituan-longcat", "kwaipilot", "skywork", "inceptionai", "llm360",
}
GGUF_PUBLISHERS = ["bartowski", "unsloth", "lmstudio-community", "ggml-org"]
GGUF_QUANT_RE = re.compile(r"(?i)(?<![A-Za-z0-9])((?:UD-)?(?:IQ\d_(?:XXS|XS|S|M|L|NL)|IQ\d_K(?:_[SML])?|Q\d_K_(?:XL|S|M|L)|Q\d_K|Q\d_\d|TQ\d_\d|BF16|F16|F32|MXFP4(?:_MOE)?))(?![A-Za-z0-9])")
DTYPES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "float8_e4m3fn": "fp8"}


def _get(url: str) -> Any | None:
    try:
        return fetch_json(url)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404):
            return None
        raise


def _find_repo(model: dict[str, Any]) -> str | None:
    if model.get("hf_repo"):
        return model["hf_repo"]
    url = model.get("weights_url") or ""
    m = re.match(r"https://huggingface\.co/([^/]+/[^/]+)", url)
    if m:
        return m.group(1)
    names = [model["id"], *(model.get("aliases") or [])]
    wanted = {norm(strip_provider(n)) for n in names}
    seen: set[str] = set()
    for n in names:
        term = strip_provider(n)
        if term in seen or len(term) < 3:
            continue
        seen.add(term)
        hits = _get(f"{API}/models?search={quote(term)}&limit=100&sort=downloads") or []
        exact = [h["id"] for h in hits if norm(h["id"].split("/")[-1]) in wanted and h["id"].split("/")[0].lower() in OFFICIAL_HF_ORGS]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            # Prefer the org the atlas already attributes the model to.
            org = model.get("org", "")
            same = [r for r in exact if ORG_SLUGS.get(r.split("/")[0].lower()) == org]
            if len(same) == 1:
                return same[0]
    return None


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    for key in ("text_config", "llm_config", "language_config"):
        if isinstance(cfg.get(key), dict) and cfg[key].get("num_hidden_layers"):
            return {**cfg[key], "_parent": cfg}
    return cfg


def architecture_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    c = _text_config(cfg)
    arch: dict[str, Any] = {}
    experts = next((c[k] for k in ("num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts") if isinstance(c.get(k), int)), None)
    active = next((c[k] for k in ("num_experts_per_tok", "num_experts_per_token", "moe_top_k", "moe_k") if isinstance(c.get(k), int)), None)
    shared = next((c[k] for k in ("n_shared_experts", "num_shared_experts", "shared_expert_count") if isinstance(c.get(k), int)), None)
    if shared is None and isinstance(c.get("shared_expert_intermediate_size"), int) and c["shared_expert_intermediate_size"] > 0:
        shared = 1
    layer_kinds = json.dumps({k: c.get(k) for k in ("layer_types", "layers_block_type", "hybrid_override_pattern", "attn_type_list") if k in c}).lower()
    hybrid = any(w in layer_kinds for w in ("mamba", "linear_attention", "ssm", "\"m\"", "gated_deltanet")) or "mamba" in str(c.get("model_type", "")).lower()
    is_moe = bool(experts and experts > 1)
    arch["type"] = ("hybrid-ssm-moe" if is_moe else "hybrid-ssm") if hybrid else ("moe" if is_moe else "dense")
    if isinstance(c.get("num_hidden_layers"), int):
        arch["layers"] = c["num_hidden_layers"]
    if isinstance(c.get("hidden_size"), int):
        arch["hidden_size"] = c["hidden_size"]
    q, kv = c.get("num_attention_heads"), c.get("num_key_value_heads")
    if isinstance(q, int):
        arch["q_heads"] = q
    if isinstance(kv, int):
        arch["kv_heads"] = kv
    if isinstance(c.get("head_dim"), int):
        arch["head_dim"] = c["head_dim"]
    if c.get("kv_lora_rank"):
        arch["attention"] = "mla"
    elif hybrid:
        arch["attention"] = "hybrid"
    elif isinstance(q, int) and isinstance(kv, int):
        arch["attention"] = "mqa" if kv == 1 else ("mha" if kv == q else "gqa")
    elif isinstance(q, int) and kv is None:
        arch["attention"] = "mha"
    if is_moe:
        arch["experts_total"] = experts
        if active:
            arch["experts_active"] = active
        if shared is not None:
            arch["experts_shared"] = shared
    dtype = c.get("torch_dtype") or c.get("dtype") or (c.get("_parent") or {}).get("torch_dtype")
    qcfg = cfg.get("quantization_config") or {}
    if qcfg.get("quant_method") == "fp8" or "fp8" in str(qcfg.get("fmt", "")):
        arch["native_dtype"] = "fp8"
    elif qcfg.get("quant_method") == "mxfp4":
        arch["native_dtype"] = "fp4"
    elif dtype in DTYPES:
        arch["native_dtype"] = DTYPES[dtype]
    return arch


def _context_length(cfg: dict[str, Any]) -> int | None:
    c = _text_config(cfg)
    v = c.get("max_position_embeddings") or c.get("seq_length") or c.get("max_sequence_length")
    return v if isinstance(v, int) and v > 0 else None


def _gguf_artifacts(name: str, base_repo: str) -> list[dict[str, Any]]:
    out = []
    for pub in GGUF_PUBLISHERS:
        repo = f"{pub}/{name}-GGUF"
        info = _get(f"{API}/models/{repo}")
        if not info or not info.get("sha"):
            continue
        tree = _get(f"{API}/models/{repo}/tree/{info['sha']}?recursive=true") or []
        by_quant: dict[str, int] = defaultdict(int)
        for f in tree:
            path = f.get("path", "")
            if not path.endswith(".gguf") or "mmproj" in path.lower() or "imatrix" in path.lower():
                continue
            matches = GGUF_QUANT_RE.findall(path[: -len(".gguf")])
            if not matches:
                continue
            by_quant[matches[-1].upper()] += int(f.get("lfs", {}).get("size") or f.get("size") or 0)
        if not by_quant:
            continue
        sid = write_source(f"hf-{slugify(pub)}-{slugify(name)}-gguf", kind="model_card", provenance="community",
                           title=f"{repo} (GGUF quantisations of {base_repo}) file listing", publisher="Hugging Face",
                           url=f"https://huggingface.co/{repo}", revision=info["sha"],
                           notes="Exact file sizes from the Hugging Face API tree listing at the given revision.")
        for quant, size in sorted(by_quant.items()):
            if size > 0:
                out.append(dict(format="gguf", quant=quant, disk_bytes=size, repo_url=f"https://huggingface.co/{repo}",
                                revision=info["sha"], publisher=pub, source_id=sid))
    return out


def run() -> dict[str, int]:
    imp = Importer("hf")
    cat = load(DATA)
    models = cat.entities.get("models", {})
    artifacts: list[dict[str, Any]] = []
    arch_updates = 0
    errors = 0
    for mid, rec in sorted(models.items()):
        try:
            arch_updates += _one(imp, mid, rec, artifacts)
        except _Skip:
            pass
        except Exception as exc:  # network trouble on one repo must not kill the run
            errors += 1
            print(f"[hf] {mid}: {type(exc).__name__}: {exc}")
    _write_artifacts(artifacts)
    stats = imp.finish()
    print(f"[hf] architecture updates={arch_updates}, artifacts={len(artifacts)}, errors={errors}")
    return {**stats, "artifacts": len(artifacts), "errors": errors}


class _Skip(Exception):
    pass


def _one(imp: Importer, mid: str, rec: Any, artifacts: list[dict[str, Any]]) -> int:
    arch_updates = 0
    if True:
        m = rec.data
        if m.get("open_weights") is False:
            raise _Skip
        repo = _find_repo(m)
        if not repo:
            raise _Skip
        info = _get(f"{API}/models/{repo}")
        if not info or not info.get("sha"):
            raise _Skip
        repo = info.get("id", repo)
        sha = info["sha"]
        author = repo.split("/")[0]
        sid = write_source(f"hf-{slugify(author)}-{slugify(repo.split('/')[1])}", kind="model_config", provenance="first_party",
                           title=f"{repo} on Hugging Face (config.json, safetensors metadata, file listing)", publisher="Hugging Face",
                           url=f"https://huggingface.co/{repo}", revision=sha,
                           notes="Architecture from config.json, parameter count from safetensors metadata, sizes from the file listing, at the given revision.")
        card = info.get("cardData") or {}
        lic = card.get("license") or next((t.split(":", 1)[1] for t in info.get("tags", []) if t.startswith("license:")), None)
        params = (info.get("safetensors") or {}).get("total")
        cfg = None
        try:
            cfg = fetch_json(f"https://huggingface.co/{repo}/raw/{sha}/config.json")
        except (urllib.error.HTTPError, json.JSONDecodeError):
            cfg = None
        imp.claim(mid, hf_repo=repo, org=author, open_weights=True, license=lic if isinstance(lic, str) else None,
                  params_total_b=(params / 1e9 if params else None), context_length=_context_length(cfg) if cfg else None,
                  source_id=sid)

        # Architecture: fill only missing keys, cite the repo for each filled key.
        arch_new = architecture_from_config(cfg) if cfg else {}
        if params:
            arch_new["params_total_b"] = round(params / 1e9, 3)
        old_arch = m.get("architecture") or {}
        add = {k: v for k, v in arch_new.items() if k not in old_arch}
        if add:
            new = dict(m)
            new["architecture"] = {**old_arch, **add}
            new["sources"] = list(m.get("sources", []))
            if sid not in new["sources"]:
                new["sources"].append(sid)
            fs = dict(m.get("field_sources") or {})
            for k in add:
                if sid != new["sources"][0]:
                    fs[f"architecture.{k}"] = sid
            if fs:
                new["field_sources"] = fs
            if "hf_repo" not in new:
                new["hf_repo"] = repo
                new.setdefault("weights_url", f"https://huggingface.co/{repo}")
                if sid != new["sources"][0]:
                    new.setdefault("field_sources", {})["hf_repo"] = sid
            from .register import _ordered
            dump_yaml(rec.path, _ordered(new))
            arch_updates = 1

        tree = _get(f"{API}/models/{repo}/tree/{sha}?recursive=true") or []
        st = sum(int(f.get("lfs", {}).get("size") or f.get("size") or 0) for f in tree if f.get("path", "").endswith(".safetensors"))
        if st:
            q = (arch_new.get("native_dtype") or "native")
            artifacts.append(dict(model_id=mid, format="safetensors", quant=q, bits_per_weight={"bf16": 16, "fp16": 16, "fp32": 32, "fp8": 8}.get(q),
                                  disk_bytes=st, repo_url=f"https://huggingface.co/{repo}", revision=sha, publisher=author, source_id=sid,
                                  notes="Sum of *.safetensors file sizes."))
        for a in _gguf_artifacts(repo.split("/")[1], repo):
            artifacts.append({"model_id": mid, **a})
    return arch_updates


def _write_artifacts(artifacts: list[dict[str, Any]]) -> None:
    import csv
    from ..catalog import load_schema
    fields = list(load_schema("artifact.schema.json")["properties"])
    out = DATA / "observations" / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    artifacts.sort(key=lambda a: (a["model_id"], a["format"], a["publisher"], a["quant"]))
    with (out / "hf.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for a in artifacts:
            w.writerow({k: ("" if a.get(k) is None else a[k]) for k in fields})
