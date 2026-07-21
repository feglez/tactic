#!/usr/bin/env python3
"""Phase 1, DarkGram step 2: translate all non-English rows (MarianMT, row-by-row).

Usage: python3 scripts/08_translate_darkgram.py --config configs/preprocessing/translation_darkgram.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args


def main() -> None:
    parser = build_parser(description=__doc__)
    args = parser.parse_args()
    cfg = load_config_from_args(args)

    from tactic.preprocessing.translation import translate_csv, validate_darkgram_header

    in_csv = Path(cfg["paths"]["in_csv"])
    out_csv = Path(cfg["paths"]["out_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Header sanity check
    validate_darkgram_header(in_csv)

    translate_csv(
        in_csv=in_csv,
        out_csv=out_csv,
        metrics_json=Path(cfg["paths"]["metrics_json"]),
        chunk_size=int(cfg["params"].get("chunk_size", 50_000)),
        max_length=int(cfg["params"].get("max_length", 512)),
        save_every=int(cfg["params"].get("save_every", 2_000)),
    )


if __name__ == "__main__":
    main()
