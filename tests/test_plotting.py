import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "data"

spec = importlib.util.spec_from_file_location("plot_frontier", ROOT / "plotting_scripts" / "plot_frontier.py")
pf = importlib.util.module_from_spec(spec)
sys.modules["plot_frontier"] = pf  # dataclasses look the module up while the class is created
spec.loader.exec_module(pf)


def ms(mid: str, release: str, value: float, params: float = 7.0) -> "pf.ModelScore":
    return pf.ModelScore(model_id=mid, name=mid, company=None, release=dt.date.fromisoformat(release), params_b=params,
                         open_weights=True, value=value)


def test_frontier_walks_every_day_and_records_leaders():
    cands = [ms("old", "2022-06-01", 10), ms("old-better", "2022-12-01", 20), ms("a", "2023-02-01", 30),
             ms("worse", "2023-03-01", 25), ms("b", "2023-04-01", 50), ms("future", "2024-01-01", 99)]
    f = pf.compute_frontier(cands, dt.date(2023, 1, 1), dt.date(2023, 6, 30), higher_is_better=True)
    assert len(f.days) == (dt.date(2023, 6, 30) - dt.date(2023, 1, 1)).days + 1
    # only the best model released before the interval is marked, at the interval start
    assert [(x.isoformat(), m.model_id) for x, m in f.leaders] == [
        ("2023-01-01", "old-better"), ("2023-02-01", "a"), ("2023-04-01", "b")]
    assert f.values[0] == 20 and f.values[f.days.index(dt.date(2023, 3, 15))] == 30 and f.values[-1] == 50


def test_frontier_lower_is_better():
    f = pf.compute_frontier([ms("a", "2023-01-05", 5.0), ms("b", "2023-01-10", 3.0), ms("c", "2023-01-20", 4.0)],
                            dt.date(2023, 1, 1), dt.date(2023, 1, 31), higher_is_better=False)
    assert [m.model_id for _, m in f.leaders] == ["a", "b"]
    assert f.values[0] != f.values[0]  # NaN before the first model


def test_plot_and_sidecar(tmp_path):
    out = pf.plot_benchmarks(benchmarks=["aider-polyglot"], max_params=35, references=["claude-3-5-sonnet-20241022"],
                             start=dt.date(2024, 6, 1), end=dt.date(2025, 6, 1), out_dir=tmp_path, data=FIXTURE,
                             verbose=False)
    assert out == [tmp_path / "aider-polyglot_le35B.png"] and out[0].stat().st_size > 10_000
    sidecar = (tmp_path / "aider-polyglot_le35B.csv").read_text()
    assert "frontier,2024-11-12,qwen2.5-coder-32b-instruct" in sidecar and "reference,,claude-3-5-sonnet-20241022" in sidecar


def test_reference_without_score_raises(tmp_path):
    with pytest.raises(pf.MissingReferenceScoreError, match="claude-3-5-sonnet-20241022"):
        pf.plot_benchmarks(benchmarks=["aider-code-editing"], max_params=35, references=["claude-3-5-sonnet-20241022"],
                           start=dt.date(2024, 1, 1), end=dt.date(2025, 1, 1), out_dir=tmp_path, data=FIXTURE, verbose=False)
    assert not list(tmp_path.iterdir())  # validated before drawing anything


def test_bad_requests_raise(tmp_path):
    kw = dict(max_params=35, start=dt.date(2024, 1, 1), end=dt.date(2025, 1, 1), out_dir=tmp_path, data=FIXTURE, verbose=False)
    with pytest.raises(pf.PlotError, match="has variants"):
        pf.plot_benchmarks(benchmarks=["humaneval"], references=[], **kw)
    with pytest.raises(pf.PlotError, match="unknown benchmark"):
        pf.plot_benchmarks(benchmarks=["no-such-bench"], references=[], **kw)
    with pytest.raises(pf.PlotError, match="unknown model"):
        pf.plot_benchmarks(benchmarks=["aider-polyglot"], references=["no-such-model"], **kw)
    with pytest.raises(pf.PlotError, match="no open-weight model"):
        pf.plot_benchmarks(benchmarks=["aider-polyglot"], references=[], **{**kw, "max_params": 1})
