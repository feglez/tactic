#!/usr/bin/env python3
"""Final global BERTopic fit on the 2M dataset (min_cluster_size sweep).

Usage:
    python scripts/24_train_final_bertopic.py --config configs/bertopic/final.yaml

Afterwards run 21_compute_metrics.py (metrics_tgdataset.yaml) and
22_score_metrics.py (score_final.yaml) to select the winner.
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
