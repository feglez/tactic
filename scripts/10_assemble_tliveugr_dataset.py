#!/usr/bin/env python3
"""Phase 1, TLiveUGR step 2: EN-only filter + rotating channel-grouped CSVs.

Usage: python3 scripts/10_assemble_tliveugr_dataset.py --config configs/preprocessing/tliveugr_assembly.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args


def main() -> None:
    parser = build_parser(description=__doc__)
    args = parser.parse_args()
    cfg = load_config_from_args(args)

    from tactic.preprocessing.tliveugr_assembly import (
        filter_and_group,
        validate_single_csv_header,
    )

    in_path = Path(cfg["paths"]["in_path"])
    if not in_path.exists():
        raise SystemExit(f"Input path not found: {in_path}")

    # Quick header check for a single CSV. For a folder we rely on per-file reads
    validate_single_csv_header(in_path)

    filter_and_group(
        in_path=in_path,
        out_dir=Path(cfg["paths"]["out_dir"]),
        out_prefix=cfg["params"]["out_prefix"],
        metrics_json=Path(cfg["paths"]["metrics_json"]),
        chunk_size=int(cfg["params"].get("chunk_size", 100_000)),
        max_rows_per_file=int(cfg["params"].get("max_rows_per_file", 1_000_000)),
    )


if __name__ == "__main__":
    main()
