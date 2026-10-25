import json
import shutil
from pathlib import Path

import pytest

from atlas.build import build
from atlas.catalog import REPO_ROOT, load
from atlas.validate import validate

DATA = REPO_ROOT / "data"
SCORE_HEADER = "model_id,benchmark_id,variant,metric,value,setting,reasoning,quant,eval_date,source_id,locator,notes\n"


@pytest.fixture
def data(tmp_path: Path) -> Path:
    dst = tmp_path / "data"
    shutil.copytree(DATA, dst)
    return dst


def errors(root: Path) -> list[str]:
    return [i.message for i in validate(load(root)) if i.level == "error"]


def warnings(root: Path) -> list[str]:
    return [i.message for i in validate(load(root)) if i.level == "warning"]


def set_scores(root: Path, *rows: str) -> None:
    (root / "observations" / "scores.csv").write_text(SCORE_HEADER + "\n".join(rows) + "\n")


def test_repository_data_is_valid():
    assert errors(DATA) == []


def test_score_predating_benchmark_is_rejected(data):
    # The classic mistake: filing a vendor's pre-Polyglot "Aider" number under Aider Polyglot.
    set_scores(data, "qwen2.5-coder-32b-instruct,aider-polyglot,,,73.7,,n/a,,,qwen2.5-coder-family-blog,,")
    assert any("predates benchmark 'aider-polyglot'" in e for e in errors(data))


def test_variant_introduction_date_is_checked(data):
    set_scores(data, "qwen2.5-coder-32b-instruct,humaneval,plus,,90,,n/a,,2022-01-01,arxiv-2305.01210,,")
    assert any("predates benchmark 'humaneval'" in e for e in errors(data))


def test_unknown_references_are_rejected(data):
    set_scores(data, "no-such-model,no-such-bench,,,50,,,,,no-such-source,,")
    errs = errors(data)
    assert any("unknown model 'no-such-model'" in e for e in errs)
    assert any("unknown benchmark 'no-such-bench'" in e for e in errs)
    assert any("unknown source 'no-such-source'" in e for e in errs)


def test_missing_source_id_fails_schema(data):
    set_scores(data, "qwen2.5-coder-32b-instruct,aider-polyglot,,,8.0,,,,2024-12-22,,,")
    assert any("'source_id' is a required property" in e for e in errors(data))


def test_variant_must_be_declared(data):
    set_scores(data, "qwen2.5-coder-32b-instruct,humaneval,,,90,,n/a,,2024-11-12,qwen2.5-coder-family-blog,,")
    assert any("variant 'None' not in" in e for e in errors(data))


def test_percent_out_of_range(data):
    set_scores(data, "qwen2.5-coder-32b-instruct,aider-polyglot,,,140,,n/a,,2025-01-01,aider-polyglot-leaderboard-data,,")
    assert any("outside [0, 100]" in e for e in errors(data))


def test_duplicate_score_rows(data):
    row = "qwen2.5-coder-32b-instruct,aider-polyglot,,,8.0,edit_format=diff,n/a,,2024-12-22,aider-polyglot-leaderboard-data,,"
    set_scores(data, row, row)
    assert any("duplicate score row" in e for e in errors(data))


def test_model_must_live_in_org_folder(data):
    src = data / "models" / "qwen" / "qwen2.5-coder-32b-instruct.yaml"
    (data / "models" / "other").mkdir()
    src.rename(data / "models" / "other" / src.name)
    assert any("expected folder 'qwen'" in e for e in errors(data))


def test_field_sources_must_point_to_existing_fields(data):
    path = data / "models" / "qwen" / "qwen2.5-coder-32b-instruct.yaml"
    path.write_text(path.read_text() + "  architecture.experts_total: hf-qwen2.5-coder-32b-instruct\n")
    assert any("missing field 'architecture.experts_total'" in e for e in errors(data))


def test_moe_without_active_params_warns(data):
    path = data / "models" / "qwen" / "qwen2.5-coder-32b-instruct.yaml"
    path.write_text(path.read_text().replace("type: dense", "type: moe"))
    warns = warnings(data)
    assert any("missing architecture.params_active_b" in w for w in warns)


def test_build_derives_minmax(data, tmp_path):
    (data / "hardware" / "rtx-test.yaml").write_text(
        "id: rtx-test\nname: Test GPU\nvendor: test\nkind: discrete_gpu\nconsumer: true\nvram_gb: 24\n"
        "sources: [hf-qwen2.5-coder-32b-instruct]\n"
    )
    (data / "runtimes" / "rt-test.yaml").write_text(
        "id: rt-test\nname: Test RT\nurl: https://example.org\ntechniques: [gpu-full]\n"
        "sources: [hf-qwen2.5-coder-32b-instruct]\n"
    )
    tp = data / "observations" / "throughput.csv"
    tp.write_text(
        tp.read_text()
        + "qwen2.5-coder-32b-instruct,Q4_K_M,rt-test,,rtx-test,none,0,1,,30.5,,hf-qwen2.5-coder-32b-instruct,,\n"
        + "qwen2.5-coder-32b-instruct,Q8_0,rt-test,,rtx-test,none,0,1,,20.0,,hf-qwen2.5-coder-32b-instruct,,\n"
    )
    cat = load(data)
    assert errors(data) == []
    out = json.loads(build(cat, tmp_path / "atlas.json").read_text())
    summary = out["derived"]["models"]["qwen2.5-coder-32b-instruct"]
    assert summary["gen_tps_consumer_hardware"] == {"min": 20.0, "max": 30.5}
    assert summary["weights_disk_gb"]["min"] == pytest.approx(65.527841688)
