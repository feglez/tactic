#!/usr/bin/env python3
"""Phase 1, TGDataset step 6: recover channel titles from raw JSONs.

Usage: python3 scripts/06_parse_channel_titles.py --config configs/preprocessing/channel_titles.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args


def main() -> None:
    parser = build_parser(description=__doc__)
    args = parser.parse_args()
    cfg = load_config_from_args(args)

    from tactic.preprocessing.channel_metadata import run_title_parser
    run_title_parser(
        raw_dir=Path(cfg["paths"]["raw_dir"]),
        out_csv=Path(cfg["paths"]["out_csv"]),
    )


if __name__ == "__main__":
    main()
