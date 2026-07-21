#!/usr/bin/env python3
"""Re-fit the tuned BERTopic configuration on a held-out validation split.

Run once per split. The config defaults target validation3. Override the
split-dependent paths and model name for validation1/2 (see
configs/bertopic/validation.yaml).

Usage:
    python scripts/23_validate_bertopic.py --config configs/bertopic/validation.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args


def main():
    parser = build_parser(description=__doc__)
    args = parser.parse_args()
    cfg = load_config_from_args(args)

    from tactic.topic_modeling.trainer import run_training_stage
    run_training_stage(cfg)


if __name__ == "__main__":
    main()
