#!/usr/bin/env python3
"""Phase 1, TGDataset step 3: stratified sampling of messages for translation.

Usage: python3 scripts/03_stratified_sample.py --config configs/preprocessing/sampling.yaml \
         [--override params.dataset_name=validation1]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args


def main() -> None:
    parser = build_parser(description=__doc__)
    args = parser.parse_args()
    cfg = load_config_from_args(args)

    from tactic.preprocessing.sampling import run_sampler
    run_sampler(cfg)


if __name__ == "__main__":
    main()
