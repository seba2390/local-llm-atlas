"""Command-line entry point: `atlas validate` and `atlas build`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .build import build
from .catalog import REPO_ROOT, load
from .validate import summarize, validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atlas")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data", help="data directory")
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="check schemas, references and citations")
    v.add_argument("--strict", action="store_true", help="treat warnings as errors")
    b = sub.add_parser("build", help="validate, then compile build/atlas.json")
    b.add_argument("--out", type=Path, default=REPO_ROOT / "build" / "atlas.json")
    args = parser.parse_args(argv)

    cat = load(args.data)
    issues = validate(cat)
    for issue in issues:
        print(issue)
    errors, warnings = summarize(issues)
    n_ent = {k: len(v) for k, v in cat.entities.items()}
    n_obs = {k: len(v) for k, v in cat.observations.items()}
    print(f"\n{errors} error(s), {warnings} warning(s) | entities {n_ent} | observations {n_obs}")

    if errors or (args.cmd == "validate" and args.strict and warnings):
        return 1
    if args.cmd == "build":
        print(f"wrote {build(cat, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
