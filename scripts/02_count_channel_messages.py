#!/usr/bin/env python3
"""Phase 1, TGDataset step 2: per-channel message counts + channel→file mapping.

Usage: python3 scripts/02_count_channel_messages.py --config configs/preprocessing/channel_counts.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args


def main() -> None:
    parser = build_parser(description=__doc__)
    args = parser.parse_args()
    cfg = load_config_from_args(args)

    from tactic.preprocessing.channel_metadata import build_channel_counts_with_sources
    out_csv = Path(cfg["paths"]["out_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    build_channel_counts_with_sources(
        csv_dir=Path(cfg["paths"]["csv_dir"]),
        out_csv=out_csv,
        chunk_size=int(cfg["params"].get("chunk_size", 100_000)),
    )


if __name__ == "__main__":
    main()
