#!/usr/bin/env python3
"""Consolidate one channel-level row per channel for the LLM stage.

Select the dataset via --config (configs/llm/consolidate_{tgdataset,darkgram,tliveugr}.yaml).

Example:
    python scripts/31_consolidate_channels.py \
        --config configs/llm/consolidate_tgdataset.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args  # noqa: E402


def main() -> None:
    parser = build_parser(description="Consolidate channel-level data for the LLM stage.")
    args = parser.parse_args()
    cfg = load_config_from_args(args)
    # Heavy imports deferred until after argparse so --help works without ML deps.
    from tactic.llm import consolidation
    consolidation.run(cfg)


if __name__ == "__main__":
    main()
