#!/usr/bin/env python3
"""Resolve configs/paths.yaml variables for the bash orchestrator.

Prints one shell-safe `name=value` assignment per requested variable, using
EXACTLY the same resolution as every pipeline stage (configs/paths.yaml +
TACTIC_<VAR> environment overrides, as implemented in tactic.common.config). This is what
keeps run_pipeline.sh's existence checks and the Python stages' actual
read/write paths in agreement: both sides resolve through one code path.

Usage: python3 scripts/00_resolve_paths.py data_root models_root ...
       eval "$(uv run python scripts/00_resolve_paths.py data_root ...)"
"""
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.config import find_project_root, load_variables


def main() -> None:
    names = sys.argv[1:]
    if not names:
        print("usage: 00_resolve_paths.py <var_name> [<var_name> ...]", file=sys.stderr)
        raise SystemExit(2)
    project_root = find_project_root(Path(__file__).resolve())
    variables = load_variables(project_root / "configs" / "paths.yaml", project_root)
    missing = [n for n in names if n not in variables]
    if missing:
        print(f"unknown paths.yaml variable(s): {missing} (defined: {sorted(variables)})",
              file=sys.stderr)
        raise SystemExit(2)
    for n in names:
        print(f"{n}={shlex.quote(variables[n])}")


if __name__ == "__main__":
    main()
