import csv
import json
import shutil
from pathlib import Path

import pytest

from atlas.build import build
from atlas.catalog import REPO_ROOT, load, load_schema
from atlas.resolve import Resolver
from atlas.validate import validate

DATA = REPO_ROOT / "data"
FIXTURE = Path(__file__).parent / "fixtures" / "data"
SCORE_FIELDS = list(load_schema("score.schema.json")["properties"])
QWEN = "models/alibaba/qwen2.5-coder-32b-instruct.yaml"
CLAUDE = "models/anthropic/claude-3-5-sonnet-20241022.yaml"


@pytest.fixture
def data(tmp_path: Path) -> Path:
    dst = tmp_path / "data"
    shutil.copytree(FIXTURE, dst)
    return dst


def issues(root: Path, level: str) -> list[str]:
    return [i.message for i in validate(load(root)) if i.level == level]


def errors(root: Path) -> list[str]:
    return issues(root, "error")


def warnings(root: Path) -> list[str]:
    return issues(root, "warning")


def score(**kw) -> dict:
    row = {"model_id": "qwen2.5-coder-32b-instruct", "benchmark_id": "aider-polyglot", "value": 8.0,
           "harness_id": "aider", "reasoning": "n/a", "eval_date": "2024-12-22", "source_id": "aider-polyglot-leaderboard-data"}
    row.update(kw)
    return {k: v for k, v in row.items() if v is not None}


def set_scores(root: Path, *rows: dict) -> None:
    with (root / "observations" / "scores" / "manual.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SCORE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_fixture_is_valid():
    assert errors(FIXTURE) == []
    assert warnings(FIXTURE) == []


def test_repository_data_is_valid():
    assert errors(DATA) == []


def test_score_predating_benchmark_is_rejected(data):
    # The classic mistake: filing a vendor's pre-Polyglot "Aider" number under Aider Polyglot.
    set_scores(data, score(value=73.7, eval_date=None, source_id="qwen2.5-coder-family-blog"))
    assert any("predates benchmark 'aider-polyglot'" in e for e in errors(data))


def test_variant_introduction_date_is_checked(data):
    set_scores(data, score(benchmark_id="humaneval", variant="plus", harness_id="undisclosed", eval_date="2022-01-01", source_id="arxiv-2305.01210"))
    assert any("predates benchmark 'humaneval'" in e for e in errors(data))


def test_unknown_references_are_rejected(data):
    set_scores(data, score(model_id="no-such-model", benchmark_id="no-such-bench", harness_id="no-such-harness", source_id="no-such-source"))
    errs = errors(data)
    for what in ("model 'no-such-model'", "benchmark 'no-such-bench'", "source 'no-such-source'", "harness 'no-such-harness'"):
        assert any(f"unknown {what}" in e for e in errs), what


def test_missing_source_id_fails_schema(data):
    set_scores(data, score(source_id=""))
    assert any("'source_id' is a required property" in e for e in errors(data))


def test_missing_harness_fails_schema(data):
    set_scores(data, score(harness_id=""))
    assert any("'harness_id' is a required property" in e for e in errors(data))


def test_undisclosed_harness_on_sensitive_benchmark_warns(data):
    set_scores(data, score(harness_id="undisclosed"))
    assert errors(data) == []
    assert any("harness-sensitive but the harness is undisclosed" in w for w in warnings(data))


def test_variant_must_be_declared(data):
    set_scores(data, score(benchmark_id="humaneval", harness_id="undisclosed", eval_date="2024-11-12", source_id="qwen2.5-coder-family-blog"))
    assert any("variant 'None' not in" in e for e in errors(data))


def test_percent_out_of_range(data):
    set_scores(data, score(value=140))
    assert any("outside [0, 100]" in e for e in errors(data))


def test_duplicate_score_rows(data):
    set_scores(data, score(), score())
    assert any("duplicate score row" in e for e in errors(data))


def test_conflicting_reports_from_different_harness_settings_are_kept(data):
    set_scores(data, score(setting="edit_format=diff"), score(value=16.4, setting="edit_format=whole"))
    assert errors(data) == []


def test_model_must_live_in_org_folder(data):
    (data / "models" / "other").mkdir()
    (data / QWEN).rename(data / "models" / "other" / "qwen2.5-coder-32b-instruct.yaml")
    assert any("expected folder 'alibaba'" in e for e in errors(data))


def test_model_company_and_country_must_match_org(data):
    p = data / QWEN
    p.write_text(p.read_text().replace("country: China", "country: Denmark"))
    assert any("country must be 'China'" in e for e in errors(data))
    p.write_text(p.read_text().replace("company: Alibaba\n", ""))
    assert any("company must be 'Alibaba'" in e for e in errors(data))


def test_model_org_must_exist(data):
    (data / "orgs" / "alibaba.yaml").unlink()
    assert any("unknown org 'alibaba'" in e for e in errors(data))


def test_field_sources_must_point_to_existing_fields(data):
    p = data / QWEN
    p.write_text(p.read_text() + "  architecture.experts_total: hf-qwen2.5-coder-32b-instruct\n")
    assert any("missing field 'architecture.experts_total'" in e for e in errors(data))


def test_moe_without_active_params_warns(data):
    p = data / QWEN
    p.write_text(p.read_text().replace("type: dense", "type: moe"))
    assert any("missing architecture.params_active_b" in w for w in warnings(data))


def test_closed_model_needs_no_architecture(data):
    assert "architecture" not in (data / CLAUDE).read_text()
    assert errors(data) == []


def test_closed_model_cannot_have_weights_url(data):
    p = data / CLAUDE
    p.write_text(p.read_text() + "weights_url: https://example.org\n")
    assert "closed-weight model must not have weights_url" in errors(data)


def test_own_measurements_are_not_a_provenance(data):
    p = data / "sources" / "arxiv-2107.03374.yaml"
    p.write_text(p.read_text().replace("provenance: first_party", "provenance: own_measurement"))
    assert any("provenance" in e for e in errors(data))


def test_alias_collision_is_rejected(data):
    p = data / CLAUDE
    p.write_text(p.read_text() + "aliases: [Qwen2.5-Coder-32B-Instruct]\n")
    assert any("share the name/alias key" in e for e in errors(data))


def test_resolver_matches_aliases_hf_ids_and_names():
    r = Resolver([
        {"id": "qwen2.5-coder-32b-instruct", "name": "Qwen2.5-Coder 32B Instruct", "hf_repo": "Qwen/Qwen2.5-Coder-32B-Instruct"},
        {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet (Oct 2024)", "aliases": ["claude-3.5-sonnet-oct"]},
        {"id": "gpt-5.5", "name": "GPT-5.5"},
        {"id": "gpt-5.5-pre-release", "name": "GPT-5.5"},
    ])
    assert r.resolve("Qwen/Qwen2.5-Coder-32B-Instruct") == "qwen2.5-coder-32b-instruct"
    assert r.resolve("openrouter/Qwen2.5_Coder_32B_Instruct") == "qwen2.5-coder-32b-instruct"
    assert r.resolve("Claude 3.5 Sonnet (Oct 2024)") == "claude-3-5-sonnet-20241022"
    assert r.resolve("claude-3.5-sonnet-oct") == "claude-3-5-sonnet-20241022"
    assert r.resolve("gpt-5.5") == "gpt-5.5"  # strong id match beats the shared display name
    assert r.resolve("GPT 5.5 pre release") == "gpt-5.5-pre-release"
    assert r.resolve("Qwen2.5-Coder-32B") is None  # never drops tokens (base != instruct)


def test_hf_architecture_parser():
    from atlas.ingest.hf import architecture_from_config

    moe = architecture_from_config({"num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4,
                                    "num_experts": 128, "num_experts_per_tok": 8, "torch_dtype": "bfloat16"})
    assert moe == {"type": "moe", "layers": 48, "q_heads": 32, "kv_heads": 4, "attention": "gqa",
                   "experts_total": 128, "experts_active": 8, "native_dtype": "bf16"}
    mla = architecture_from_config({"num_hidden_layers": 61, "num_attention_heads": 128, "num_key_value_heads": 128,
                                    "kv_lora_rank": 512, "n_routed_experts": 256, "num_experts_per_tok": 8,
                                    "n_shared_experts": 1, "quantization_config": {"quant_method": "fp8"}})
    assert mla["attention"] == "mla" and mla["experts_shared"] == 1 and mla["native_dtype"] == "fp8"
    dense = architecture_from_config({"text_config": {"num_hidden_layers": 62, "num_attention_heads": 32,
                                                      "num_key_value_heads": 16}, "torch_dtype": "bfloat16"})
    assert dense["type"] == "dense" and dense["layers"] == 62


def test_build_derives_minmax(data, tmp_path):
    (data / "hardware" / "rtx-test.yaml").write_text(
        "id: rtx-test\nname: Test GPU\nvendor: test\nkind: discrete_gpu\nconsumer: true\nvram_gb: 24\n"
        "sources: [hf-qwen2.5-coder-32b-instruct]\n"
    )
    (data / "runtimes" / "rt-test.yaml").write_text(
        "id: rt-test\nname: Test RT\nurl: https://example.org\ntechniques: [gpu-full]\nsources: [hf-qwen2.5-coder-32b-instruct]\n"
    )
    fields = list(load_schema("throughput.schema.json")["properties"])
    with (data / "observations" / "throughput.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for quant, tps in (("Q4_K_M", 30.5), ("Q8_0", 20.0)):
            w.writerow({"model_id": "qwen2.5-coder-32b-instruct", "quant": quant, "runtime_id": "rt-test", "hardware_id": "rtx-test",
                        "offload": "none", "gen_tps": tps, "source_id": "hf-qwen2.5-coder-32b-instruct"})
    assert errors(data) == []
    out = json.loads(build(load(data), tmp_path / "atlas.json").read_text())
    summary = out["derived"]["models"]["qwen2.5-coder-32b-instruct"]
    assert summary["gen_tps_consumer_hardware"] == {"min": 20.0, "max": 30.5}
    assert summary["weights_disk_gb"]["min"] == pytest.approx(65.527841688)
