#!/usr/bin/env python3
"""Composite-score a metrics CSV and print the winning model.

Usage:
    python scripts/22_score_metrics.py --config configs/bertopic/score_final.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args


def main():
    parser = build_parser(description=__doc__)
    args = parser.parse_args()
    cfg = load_config_from_args(args)

    from tactic.topic_modeling.scoring import run_score_stage
    run_score_stage(cfg)


if __name__ == "__main__":
    main()
