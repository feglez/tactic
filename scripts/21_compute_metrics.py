#!/usr/bin/env python3
"""Compute coherence/silhouette/outlier-rate/topic-count metrics for BERTopic models.

Usage:
    python scripts/21_compute_metrics.py --config configs/bertopic/metrics_tgdataset.yaml
    python scripts/21_compute_metrics.py --config configs/bertopic/metrics_darkgram.yaml

The metrics_tgdataset defaults evaluate the final sweep. Use --override to
point the stage-specific paths at another stage (see comments in
metrics_tgdataset.yaml).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args


def main():
    parser = build_parser(description=__doc__)
    args = parser.parse_args()
    cfg = load_config_from_args(args)

    from tactic.topic_modeling.metrics import run_metrics
    run_metrics(cfg)


if __name__ == "__main__":
    main()
