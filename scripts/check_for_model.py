"""Print every model in the catalogue whose name, id or alias contains QUERY.

Matching is case-insensitive and ignores spaces, hyphens and underscores ("qwen3.8 27b" finds "Qwen3.8-27B").

Edit QUERY below and run:  uv run python scripts/check_for_model.py
"""

import re
from collections import Counter

from atlas.catalog import REPO_ROOT, load

QUERY = "qwen3.8"


def _key(text: str) -> str:
    return re.sub(r"[\s_-]+", "", text.lower())


def find_models(query: str, data=REPO_ROOT / "data") -> list[tuple[dict, list[str], int]]:
    """(model record, names that matched, number of score rows) for every matching model, sorted by name."""
    cat = load(data)
    n_scores = Counter(r.data["model_id"] for r in cat.observations.get("scores", []))
    q = _key(query)
    hits = []
    for rec in cat.entities.get("models", {}).values():
        m = rec.data
        names = [m.get("name", ""), m["id"], *(m.get("aliases") or []), *(m.get("api_ids") or [])]
        matched = list(dict.fromkeys(n for n in names if n and q in _key(str(n))))
        if matched:
            hits.append((m, matched, n_scores[m["id"]]))
    return sorted(hits, key=lambda h: (h[0].get("name") or h[0]["id"]).lower())


def main() -> None:
    hits = find_models(QUERY)
    if not hits:
        print(f"No models match '{QUERY}'.")
        return
    weights = {True: "open", False: "closed", None: "?"}
    rows = [("name", "id", "company", "weights", "params", "released", "scores", "matched alias")]
    for m, matched, n in hits:
        params = (m.get("architecture") or {}).get("params_total_b")
        other = [n_ for n_ in matched if n_ not in (m.get("name"), m["id"])]
        rows.append((m.get("name") or "", m["id"], m.get("company") or "", weights[m.get("open_weights")],
                     f"{params:g}B" if params else "", str(m.get("release_date") or ""), str(n), ", ".join(other)))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    print(f"{len(hits)} model(s) matching '{QUERY}':\n")
    for i, r in enumerate(rows):
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
        if i == 0:
            print("  ".join("-" * w for w in widths))


if __name__ == "__main__":
    main()
