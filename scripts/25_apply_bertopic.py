#!/usr/bin/env python3
"""Apply the frozen final BERTopic model to a dataset (transform only).

Usage:
    python scripts/25_apply_bertopic.py --config configs/bertopic/apply_tgdataset.yaml
    python scripts/25_apply_bertopic.py --config configs/bertopic/apply_darkgram.yaml
    python scripts/25_apply_bertopic.py --config configs/bertopic/apply_tliveugr.yaml

Writes topics.pkl to ${output_root}/bertopic_output/inference/<dataset>/,
which the metrics and topic-assignment stages read directly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args


def main():
    parser = build_parser(description=__doc__)
    args = parser.parse_args()
    cfg = load_config_from_args(args)

    from tactic.topic_modeling.apply import run_apply
    run_apply(cfg)


if __name__ == "__main__":
    main()
