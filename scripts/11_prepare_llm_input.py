#!/usr/bin/env python3
"""Phase 3 preparation: build the LLM input table for a dataset.

Usage: python3 scripts/11_prepare_llm_input.py --config configs/preprocessing/prepare_llm_input_tgdataset.yaml
       python3 scripts/11_prepare_llm_input.py --config configs/preprocessing/prepare_llm_input_darkgram.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args


def main() -> None:
    parser = build_parser(description=__doc__)
    args = parser.parse_args()
    cfg = load_config_from_args(args)

    from tactic.preprocessing.prepare_llm_input import prepare_llm_input

    chid = cfg["paths"].get("chid_to_title_csv")
    prepare_llm_input(
        dataset_csv=Path(cfg["paths"]["dataset_csv"]),
        output_csv=Path(cfg["paths"]["output_csv"]),
        variant=cfg["params"]["variant"],
        chid_to_title_csv=Path(chid) if chid else None,
    )


if __name__ == "__main__":
    main()
