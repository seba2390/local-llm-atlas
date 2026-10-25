"""Best open-weight model over time on a benchmark, with reference models as horizontal lines.

For every day in [start, end] the script finds the best score among open-weight models with at most
`--max-params` billion parameters that had been released on or before that day, draws that frontier as a
step line, labels each model that took the lead, and overlays the reference models (any model, open or
closed) as horizontal lines. One figure per benchmark, plus a CSV next to it listing the source of every
plotted number.

Examples
--------
    uv run python plotting_scripts/plot_frontier.py \\
        --benchmarks aider-polyglot swe-bench:verified terminal-bench:2.0 \\
        --max-params 35 \\
        --references claude-3-5-sonnet-20241022 gpt-4o-2024-08-06 \\
        --start 2023-01-01 --end 2026-10-25

    # only independent (third-party) results, median over a model's reported scores
    uv run python plotting_scripts/plot_frontier.py --benchmarks humaneval:plus --max-params 15 \\
        --provenance third_party --aggregate median --start 2023-01-01

A benchmark is given as `id` or `id:variant` (see data/benchmarks/<id>.yaml for variants). Reference models
are atlas model ids or any of their names/aliases. A reference model without a score on a requested
benchmark is an error.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from atlas.catalog import REPO_ROOT, Catalog, load  # noqa: E402
from atlas.resolve import Resolver  # noqa: E402


class PlotError(ValueError):
    """Invalid plot request (unknown benchmark/variant/model, reference model without a score, ...)."""


class MissingReferenceScoreError(PlotError):
    pass


# ---------------------------------------------------------------- data selection


@dataclass(frozen=True)
class BenchmarkSpec:
    id: str
    variant: str | None
    name: str
    metric: str
    unit: str
    higher_is_better: bool

    @property
    def label(self) -> str:
        return f"{self.name} ({self.variant})" if self.variant else self.name

    @property
    def slug(self) -> str:
        return f"{self.id}_{self.variant}" if self.variant else self.id


@dataclass
class ModelScore:
    model_id: str
    name: str
    company: str | None
    release: dt.date | None
    params_b: float | None
    open_weights: bool | None
    value: float
    rows: list[dict[str, Any]] = field(default_factory=list)


def parse_benchmark(cat: Catalog, spec: str, metric: str | None) -> BenchmarkSpec:
    bid, _, variant = spec.partition(":")
    benchmarks = cat.entities.get("benchmarks", {})
    if bid not in benchmarks:
        close = difflib.get_close_matches(bid, list(benchmarks), n=5)
        raise PlotError(f"unknown benchmark '{bid}'" + (f"; did you mean {close}?" if close else ""))
    b = benchmarks[bid].data
    declared = [v["id"] for v in b.get("variants") or []]
    if declared and not variant:
        raise PlotError(f"benchmark '{bid}' has variants {declared}; specify one as '{bid}:<variant>'")
    if variant and variant not in declared:
        raise PlotError(f"benchmark '{bid}' has no variant '{variant}' (variants: {declared or 'none'})")
    return BenchmarkSpec(id=bid, variant=variant or None, name=b["name"], metric=metric or b["metric"],
                         unit=b["unit"], higher_is_better=b.get("higher_is_better", True))


def _date(value: Any) -> dt.date | None:
    if not value:
        return None
    text = str(value)
    try:
        return dt.date.fromisoformat(text[:10]) if len(text) >= 10 else dt.date.fromisoformat(f"{text[:7]}-01")
    except ValueError:
        return None


def collect_scores(cat: Catalog, bench: BenchmarkSpec, *, provenance: str = "any",
                   harnesses: Iterable[str] | None = None, aggregate: str = "max",
                   exclude_setting: str | None = None) -> dict[str, ModelScore]:
    """One value per model: all matching score rows aggregated with `aggregate`."""
    benchmark_default = cat.entities["benchmarks"][bench.id].data["metric"]
    sources = cat.entities.get("sources", {})
    models = cat.entities.get("models", {})
    harness_set = set(harnesses or [])
    by_model: dict[str, list[dict[str, Any]]] = {}
    for rec in cat.observations.get("scores", []):
        row = rec.data
        if row.get("benchmark_id") != bench.id or (row.get("variant") or None) != bench.variant:
            continue
        if (row.get("metric") or benchmark_default) != bench.metric:
            continue
        if harness_set and row.get("harness_id") not in harness_set:
            continue
        if exclude_setting and re.search(exclude_setting, row.get("setting") or ""):
            continue
        if provenance != "any":
            src = sources.get(row.get("source_id", ""))
            if not src or src.data.get("provenance") != provenance:
                continue
        if isinstance(row.get("value"), (int, float)):
            by_model.setdefault(row["model_id"], []).append(row)

    out: dict[str, ModelScore] = {}
    for mid, rows in by_model.items():
        if mid not in models:
            continue
        m = models[mid].data
        values = [r["value"] for r in rows]
        if aggregate == "median":
            value, kept = statistics.median(values), rows
        else:
            pick = max if bench.higher_is_better else min
            value = pick(values)
            kept = [r for r in rows if r["value"] == value]
        out[mid] = ModelScore(
            model_id=mid, name=m.get("name") or mid, company=m.get("company"), release=_date(m.get("release_date")),
            params_b=(m.get("architecture") or {}).get("params_total_b"), open_weights=m.get("open_weights"),
            value=float(value), rows=kept,
        )
    return out


def params_of(cat: Catalog, model_id: str, basis: str) -> float | None:
    arch = cat.entities["models"][model_id].data.get("architecture") or {}
    return arch.get("params_active_b" if basis == "active" else "params_total_b")


def resolve_models(cat: Catalog, names: Iterable[str]) -> list[str]:
    resolver = Resolver.from_catalog(cat)
    models = cat.entities.get("models", {})
    ids = []
    for name in names:
        mid = name if name in models else resolver.resolve(name)
        if mid is None:
            close = difflib.get_close_matches(name.lower(), list(models), n=5, cutoff=0.6)
            raise PlotError(f"unknown model '{name}'" + (f"; did you mean {close}?" if close else ""))
        ids.append(mid)
    return ids


# ---------------------------------------------------------------- frontier


@dataclass
class Frontier:
    days: list[dt.date]
    values: list[float]
    leaders: list[tuple[dt.date, ModelScore]]  # (x position, model) each time the lead changed


def compute_frontier(candidates: Iterable[ModelScore], start: dt.date, end: dt.date, higher_is_better: bool) -> Frontier:
    """Walk every day in [start, end]; the frontier on a day is the best model released on or before it."""
    pending = sorted((c for c in candidates if c.release and c.release <= end), key=lambda c: (c.release, c.model_id))
    better = (lambda a, b: a > b) if higher_is_better else (lambda a, b: a < b)
    days, values, leaders = [], [], []
    best: ModelScore | None = None
    i = 0
    day = start
    while day <= end:
        while i < len(pending) and pending[i].release <= day:
            c = pending[i]
            i += 1
            if best is None or better(c.value, best.value):
                best = c
                x = max(c.release, start)
                if leaders and leaders[-1][0] == x:
                    leaders[-1] = (x, c)  # several leads on the same day (or before start): keep the last
                else:
                    leaders.append((x, c))
        days.append(day)
        values.append(best.value if best else math.nan)
        day += dt.timedelta(days=1)
    return Frontier(days=days, values=values, leaders=leaders)


# ---------------------------------------------------------------- drawing

COMPANY_COLORS = {
    "Alibaba": "#7c3aed", "Meta AI": "#2563eb", "Google": "#16a34a", "Google DeepMind": "#16a34a",
    "Mistral AI": "#f97316", "DeepSeek": "#0891b2", "Microsoft": "#0ea5e9", "NVIDIA": "#65a30d", "Nvidia": "#65a30d",
    "OpenAI": "#111827", "Anthropic": "#c2410c", "Z.ai (Zhipu AI)": "#4f46e5", "Moonshot": "#be185d",
    "BigCode": "#eab308", "IBM": "#1d4ed8", "01.AI": "#059669", "MiniMax": "#e11d48", "xAI": "#525252",
}
FALLBACK_COLORS = ["#b91c1c", "#a16207", "#c026d3", "#57534e", "#0f766e", "#1e3a8a", "#831843", "#3f6212"]
REFERENCE_COLORS = ["#db2777", "#2563eb", "#475569", "#059669", "#ea580c", "#7c3aed"]
REFERENCE_STYLES = ["--", "-.", ":", (0, (5, 2, 1, 2)), (0, (1, 1)), (0, (8, 3))]
FRONTIER_COLOR = "#334155"


def fmt_value(v: float, unit: str) -> str:
    if unit == "percent":
        return f"{v:.1f}%"
    if unit == "elo":
        return f"{v:.0f}"
    if unit == "usd":
        return f"${v:,.0f}"
    if unit == "minutes":
        return f"{v:.0f} min"
    return f"{v:.3g}"


def fmt_params(p: float | None) -> str:
    if p is None:
        return ""
    return f"{p:.0f}B" if p >= 10 else f"{p:.1f}B".replace(".0B", "B")


def _company_colors(companies: Iterable[str | None]) -> dict[str | None, str]:
    colors: dict[str | None, str] = {}
    spare = iter(FALLBACK_COLORS * 10)
    for c in sorted({c for c in companies}, key=lambda c: (c is None, c or "")):
        colors[c] = COMPANY_COLORS.get(c or "", next(spare)) if c else "#9ca3af"
    return colors


def _overlap(a, b) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def _place_labels(fig, ax, items: list[tuple[dt.date, float, str, str]], obstacles: list) -> None:
    """Greedy label placement: try offsets around each point, keep the one with least overlap."""
    renderer = fig.canvas.get_renderer()
    placed = list(obstacles)
    axes_box = ax.get_window_extent(renderer)
    offsets = [(-14, 16), (14, 16), (-14, -26), (14, -26), (-40, 30), (40, 30), (-40, -40), (40, -40),
               (-70, 8), (70, 8), (-70, 44), (70, 44), (-70, -50), (70, -50), (0, 58), (0, -64)]
    for x, y, text, color in items:
        best = None
        for dx, dy in offsets:
            ha = "right" if dx < 0 else ("left" if dx > 0 else "center")
            va = "bottom" if dy > 0 else "top"
            ann = ax.annotate(
                text, (mdates.date2num(x), y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontsize=8.5, color="#111827", zorder=6, multialignment="left",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, lw=0.9, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.8, shrinkA=0, shrinkB=4),
            )
            box = ann.get_window_extent(renderer)
            outside = (max(0, axes_box.x0 - box.x0) + max(0, box.x1 - axes_box.x1)
                       + max(0, axes_box.y0 - box.y0) + max(0, box.y1 - axes_box.y1))
            cost = sum(_overlap(box, p) for p in placed) + outside * 1e4 + (abs(dx) + abs(dy)) * 0.5
            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()
                best = (cost, ann, box)
            else:
                ann.remove()
            if cost < 1e-9 + (abs(dx) + abs(dy)) * 0.5:
                break
        placed.append(best[2])


def draw(bench: BenchmarkSpec, frontier: Frontier, references: list[ModelScore], *, max_params: float,
         param_basis: str, start: dt.date, end: dt.date, footnote: str, out: Path,
         candidates: list[ModelScore] | None = None) -> Path:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=200)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.12)
    leaders = [m for _, m in frontier.leaders]
    colors = _company_colors([m.company for m in leaders])

    values = [v for v in frontier.values if not math.isnan(v)] + [r.value for r in references]
    if candidates:
        values += [c.value for c in candidates]
    if not values:
        raise PlotError(f"no data to plot for {bench.label}")
    lo, hi = min(values), max(values)
    if bench.unit == "percent":
        bottom = 0.0
        top = hi + max(8.0, 0.14 * (hi - bottom))
    else:
        span = (hi - lo) or abs(hi) or 1.0
        bottom, top = lo - 0.12 * span, hi + 0.2 * span
    ax.set_ylim(bottom, top)
    ax.set_xlim(mdates.date2num(start) - 10, mdates.date2num(end) + 10)

    ax.set_facecolor("#fbfbfd")
    ax.grid(True, color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9ca3af")
    ax.tick_params(colors="#374151", labelsize=9)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    if bench.unit == "percent":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "" if v > 100.0001 else f"{v:.0f}%"))
    ylabel = bench.metric.replace("_", " ")
    ax.set_ylabel(ylabel[:1].upper() + ylabel[1:], color="#374151")

    if candidates:
        ax.scatter([mdates.date2num(c.release) for c in candidates], [c.value for c in candidates],
                   s=18, color="#cbd5e1", edgecolor="none", zorder=2)

    ax.plot(frontier.days, frontier.values, drawstyle="steps-post", color=FRONTIER_COLOR, lw=2.2, alpha=0.85, zorder=3)

    obstacles = []
    renderer = fig.canvas.get_renderer()
    for i, ref in enumerate(references):
        color, style = REFERENCE_COLORS[i % len(REFERENCE_COLORS)], REFERENCE_STYLES[i % len(REFERENCE_STYLES)]
        ax.axhline(ref.value, color=color, ls=style, lw=1.5, alpha=0.9, zorder=1)
        t = ax.annotate(f"{ref.name}  {fmt_value(ref.value, bench.unit)}", (0.005, ref.value), xycoords=("axes fraction", "data"),
                        xytext=(4, 3), textcoords="offset points", ha="left", va="bottom", fontsize=8.5,
                        color=color, fontweight="bold", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="#fbfbfd", ec="none", alpha=0.85))
        obstacles.append(t.get_window_extent(renderer))

    xs = [mdates.date2num(x) for x, _ in frontier.leaders]
    ys = [m.value for _, m in frontier.leaders]
    ax.scatter(xs, ys, s=95, c=[colors[m.company] for m in leaders], edgecolor="white", linewidth=1.6, zorder=7)
    for x, y in zip(xs, ys):
        px, py = ax.transData.transform((x, y))
        obstacles.append(Bbox.from_bounds(px - 14, py - 14, 28, 28))

    items = []
    for x, m in frontier.leaders:
        size = fmt_params(m.params_b)
        text = f"{m.name}\n{fmt_value(m.value, bench.unit)}" + (f"  ·  {size}" if size else "")
        if m.release and m.release < start:
            text += f"\n(released {m.release:%b %Y})"
        items.append((x, m.value, text, colors[m.company]))
    _place_labels(fig, ax, items, obstacles)

    handles = [Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor=colors[c], markeredgecolor="white",
                      label=c or "Unknown") for c in sorted(colors, key=lambda c: (c is None, c or ""))]
    handles.append(Line2D([], [], color=FRONTIER_COLOR, lw=2.2, label="Best open-weight model"))
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=True, framealpha=0.92, edgecolor="#e5e7eb",
              ncol=2 if len(handles) > 6 else 1)

    basis = "active" if param_basis == "active" else ""
    fig.suptitle(f"{bench.label}  ·  open-weight models ≤ {max_params:g}B {basis} params".replace("  params", " params"),
                 x=0.07, y=0.965, ha="left", fontsize=15, fontweight="bold", color="#111827")
    fig.text(0.98, 0.015, footnote, ha="right", va="bottom", fontsize=7.5, color="#6b7280")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- orchestration


def write_sidecar(path: Path, bench: BenchmarkSpec, frontier: Frontier, references: list[ModelScore]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["role", "date", "model_id", "model_name", "company", "params_total_b", "value", "n_rows",
                    "source_id", "locator", "harness_id", "harness_version", "setting", "reasoning_effort"])
        for role, date, m in [("frontier", x, m) for x, m in frontier.leaders] + [("reference", None, r) for r in references]:
            for r in m.rows:
                w.writerow([role, date or "", m.model_id, m.name, m.company or "", m.params_b or "", m.value, len(m.rows),
                            r.get("source_id", ""), r.get("locator", ""), r.get("harness_id", ""),
                            r.get("harness_version", ""), r.get("setting", ""), r.get("reasoning_effort", "")])


def plot_benchmarks(*, benchmarks: list[str], max_params: float, references: list[str], start: dt.date, end: dt.date,
                    out_dir: Path, metric: str | None = None, provenance: str = "any", harnesses: list[str] | None = None,
                    aggregate: str = "max", exclude_setting: str | None = None, param_basis: str = "total",
                    show_candidates: bool = False,
                    fmt: str = "png", data: Path = REPO_ROOT / "data", verbose: bool = True) -> list[Path]:
    if start > end:
        raise PlotError(f"start {start} is after end {end}")
    cat = load(data)
    specs = [parse_benchmark(cat, b, metric) for b in benchmarks]
    ref_ids = resolve_models(cat, references)

    # Validate everything before drawing anything.
    selected = []
    for bench in specs:
        scores = collect_scores(cat, bench, provenance=provenance, harnesses=harnesses, aggregate=aggregate,
                                exclude_setting=exclude_setting)
        missing = [rid for rid in ref_ids if rid not in scores]
        if missing:
            raise MissingReferenceScoreError(
                f"reference model(s) {missing} have no '{bench.metric}' score on {bench.label} "
                f"(provenance={provenance}, harnesses={harnesses or 'any'})")
        selected.append((bench, scores))

    written = []
    for bench, scores in selected:
        eligible, skipped = [], {"no release date": [], "no parameter count": []}
        for s in scores.values():
            if not s.open_weights:
                continue
            p = params_of(cat, s.model_id, param_basis)
            if p is None:
                skipped["no parameter count"].append(s.model_id)
            elif p <= max_params:
                if s.release is None:
                    skipped["no release date"].append(s.model_id)
                else:
                    eligible.append(s)
        if not eligible:
            raise PlotError(f"no open-weight model ≤ {max_params:g}B with a release date has a score on {bench.label}")
        frontier = compute_frontier(eligible, start, end, bench.higher_is_better)
        if not frontier.leaders:
            raise PlotError(f"no eligible model on {bench.label} was released before {end}")
        refs = [scores[rid] for rid in ref_ids]
        n_sources = len({r["source_id"] for m in eligible + refs for r in m.rows})
        footnote = (f"local-llm-atlas · {aggregate} over reported scores per model · provenance: {provenance}"
                    f" · {n_sources} source{'s' if n_sources != 1 else ''} · {start:%Y-%m-%d} – {end:%Y-%m-%d}")
        stem = f"{bench.slug}_le{max_params:g}B{'_active' if param_basis == 'active' else ''}"
        path = draw(bench, frontier, refs, max_params=max_params, param_basis=param_basis, start=start, end=end,
                    footnote=footnote, out=out_dir / f"{stem}.{fmt}",
                    candidates=[c for c in eligible if start <= c.release <= end] if show_candidates else None)
        write_sidecar(out_dir / f"{stem}.csv", bench, frontier, refs)
        written.append(path)
        if verbose:
            print(f"{bench.label}: {len(eligible)} eligible models, {len(frontier.leaders)} leaders -> {path}")
            for reason, ids in skipped.items():
                if ids:
                    print(f"  skipped (open-weight, {reason}): {', '.join(sorted(ids))}")
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmarks", nargs="+", required=True, metavar="ID[:VARIANT]")
    p.add_argument("--max-params", type=float, required=True, help="maximum parameter count in billions")
    p.add_argument("--references", nargs="*", default=[], metavar="MODEL", help="reference models (horizontal lines)")
    p.add_argument("--start", type=dt.date.fromisoformat, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=dt.date.fromisoformat, default=dt.date.today(), help="YYYY-MM-DD (default: today)")
    p.add_argument("--metric", help="metric to plot (default: the benchmark's headline metric)")
    p.add_argument("--provenance", choices=["any", "first_party", "third_party", "community"], default="any",
                   help="only use scores whose source has this provenance")
    p.add_argument("--harness", nargs="*", dest="harnesses", metavar="HARNESS_ID", help="only use scores from these harnesses")
    p.add_argument("--aggregate", choices=["max", "median"], default="max",
                   help="how to combine several reported scores for the same model (default: max)")
    p.add_argument("--exclude-setting", metavar="REGEX",
                   help="drop score rows whose `setting` matches, e.g. 'attempts=2' to drop best-of-N runs")
    p.add_argument("--param-basis", choices=["total", "active"], default="total",
                   help="compare --max-params with total (default) or active (MoE) parameters")
    p.add_argument("--show-candidates", action="store_true", help="also show non-leading eligible models as grey dots")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "plots", help="output directory (default: plots/)")
    p.add_argument("--format", dest="fmt", choices=["png", "svg", "pdf"], default="png")
    p.add_argument("--data", type=Path, default=REPO_ROOT / "data")
    a = p.parse_args(argv)
    plot_benchmarks(benchmarks=a.benchmarks, max_params=a.max_params, references=a.references, start=a.start, end=a.end,
                    out_dir=a.out, metric=a.metric, provenance=a.provenance, harnesses=a.harnesses,
                    aggregate=a.aggregate, exclude_setting=a.exclude_setting, param_basis=a.param_basis, show_candidates=a.show_candidates, fmt=a.fmt,
                    data=a.data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
