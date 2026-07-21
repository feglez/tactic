#!/usr/bin/env python3
"""Phase 1, TLiveUGR step 1: extract → clean → detect language → export metrics.

Usage: python3 scripts/09_preprocess_tliveugr.py --config configs/preprocessing/tliveugr.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args


def main() -> None:
    parser = build_parser(description=__doc__)
    args = parser.parse_args()
    cfg = load_config_from_args(args)

    from tactic.preprocessing.pipeline import run_tliveugr
    run_tliveugr(cfg)


if __name__ == "__main__":
    main()
