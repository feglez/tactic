#!/usr/bin/env python3
"""Phase 1, TGDataset step 4: translate the sampled messages (MarianMT).

Usage: python3 scripts/04_translate_selected.py --config configs/preprocessing/translation_tgdataset.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactic.common.cli import build_parser, load_config_from_args


def main() -> None:
    parser = build_parser(description=__doc__)
    args = parser.parse_args()
    cfg = load_config_from_args(args)

    from tactic.preprocessing.translation import translate_selection

    dataset_name = cfg["params"]["dataset_name"]
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output filenames: <name>_dataset.csv / translation_metrics_<name>.json
    translate_selection(
        data_dir=Path(cfg["paths"]["data_dir"]),
        selection_csv=Path(cfg["paths"]["selection_csv"]),
        counts_csv=Path(cfg["paths"]["counts_csv"]),
        out_csv=out_dir / f"{dataset_name}_dataset.csv",
        log_json=out_dir / f"translation_metrics_{dataset_name}.json",
        save_every=int(cfg["params"].get("save_every", 180)),
        map_cache_json=out_dir / "channel_file_map.json",
    )


if __name__ == "__main__":
    main()
