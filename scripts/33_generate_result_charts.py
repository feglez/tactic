#!/usr/bin/env python3
"""Generate category count tables and result charts from the classifier outputs.

Select the dataset via --config (configs/llm/charts_{tgdataset,darkgram,tliveugr}.yaml).

Example:
    python scripts/33_generate_result_charts.py --config configs/llm/charts_darkgram.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args  # noqa: E402


def main() -> None:
    parser = build_parser(description="Generate category counts and result charts.")
    args = parser.parse_args()
    cfg = load_config_from_args(args)
    # Heavy imports deferred until after argparse so --help works without ML deps.
    from tactic.llm import result_charts
    result_charts.run(cfg)


if __name__ == "__main__":
    main()
