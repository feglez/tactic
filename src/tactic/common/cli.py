"""Shared CLI plumbing so every ``scripts/NN_*.py`` entry point behaves identically.

Standard usage inside an entry point::

    from tactic.common.cli import build_parser, load_config_from_args

    parser = build_parser(description="...")
    # (optionally add script-specific arguments here)
    args = parser.parse_args()
    cfg = load_config_from_args(args)

Every entry point therefore supports:

- ``--config <stage yaml>``   (required) which run configuration to use
- ``--paths <paths yaml>``    optional override of configs/paths.yaml
- ``--override a.b.c=value``  repeatable dotted-path config overrides, e.g.
  ``--override params.chunk_size=50000``
"""

from __future__ import annotations

import argparse

from tactic.common.config import load_config


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the stage YAML config (see configs/).",
    )
    parser.add_argument(
        "--paths",
        default=None,
        help="Path to the variables file (default: <project_root>/configs/paths.yaml).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Dotted-path config override, repeatable. Values are parsed as YAML. "
            "Example: --override params.chunk_size=50000"
        ),
    )
    return parser


def load_config_from_args(args: argparse.Namespace) -> dict:
    return load_config(args.config, paths_path=args.paths, overrides=args.override)
