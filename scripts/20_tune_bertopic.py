#!/usr/bin/env python3
"""BERTopic hyperparameter tuning (phases 1/2/3, selected by config).

Usage:
    python scripts/20_tune_bertopic.py --config configs/bertopic/phase1.yaml
    python scripts/20_tune_bertopic.py --config configs/bertopic/phase2.yaml
    python scripts/20_tune_bertopic.py --config configs/bertopic/phase3.yaml

Run 21_compute_metrics.py (pointed at the stage's run log) after each phase:
phases 2/3 select their base configs from the previous stage's metrics.
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
