#!/usr/bin/env python3
"""Generate per-channel BERTopic topic assignments.

Select the dataset via --config (configs/llm/assignments_{tgdataset,darkgram,tliveugr}.yaml).

Example:
    python scripts/30_generate_topic_assignments.py \
        --config configs/llm/assignments_tgdataset.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args  # noqa: E402


def main() -> None:
    parser = build_parser(description="Generate per-channel BERTopic topic assignments.")
    args = parser.parse_args()
    cfg = load_config_from_args(args)
    # Heavy imports deferred until after argparse so --help works without ML deps.
    from tactic.llm import assignments
    assignments.run(cfg)


if __name__ == "__main__":
    main()
