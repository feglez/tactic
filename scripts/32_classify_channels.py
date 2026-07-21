#!/usr/bin/env python3
"""Classify Telegram channels into the cybercrime taxonomy with an LLM.

The default transport POSTs to a local Ollama /api/generate endpoint
(GPT-OSS 20B). Set llm.transport=litellm to route through LiteLLM instead
(see src/tactic/llm/llm_client.py).
Select the dataset via --config (configs/llm/classify_{tgdataset,darkgram,tliveugr}.yaml).

Example:
    python scripts/32_classify_channels.py --config configs/llm/classify_tgdataset.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args  # noqa: E402


def main() -> None:
    parser = build_parser(description="Classify Telegram channels with the LLM stage.")
    args = parser.parse_args()
    cfg = load_config_from_args(args)
    # Heavy imports deferred until after argparse so --help works without ML deps.
    from tactic.llm import classifier
    classifier.run(cfg)


if __name__ == "__main__":
    main()
