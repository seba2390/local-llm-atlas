"""Best open-weight model over time on each benchmark, with reference models as horizontal lines.

Edit the settings below and run:  uv run python scripts/plot_frontier.py
Plots (and a CSV of the sources behind every plotted number) are written to plots/.
"""

from plot_utils import plot_frontier

MAX_PARAMS = 35  # billions of (total) parameters
REFERENCE_MODELS = ["gemini-2.5-flash"]  # atlas model ids or aliases; each must have a score on every benchmark
BENCHMARKS = ["aider-polyglot", "swe-bench:verified", "livecodebench"]  # "id" or "id:variant"
START = "2024-01-01"
END = "2026-10-25"

if __name__ == "__main__":
    plot_frontier(benchmarks=BENCHMARKS, max_params=MAX_PARAMS, references=REFERENCE_MODELS, start=START, end=END)
