"""JSON-metrics export helpers shared by the Phase 1 preprocessors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Confidence-histogram bucket keys, in the order the metrics JSON lists them.
CONFIDENCE_BUCKET_KEYS = [
    "confidence_between_0.9_and_1_count",
    "confidence_between_0.8_and_0.89_count",
    "confidence_between_0.7_and_0.79_count",
    "confidence_between_0.6_and_0.69_count",
    "confidence_between_0.5_and_0.59_count",
    "confidence_between_0.4_and_0.49_count",
    "confidence_between_0.3_and_0.39_count",
    "confidence_between_0.2_and_0.29_count",
    "confidence_between_0.1_and_0.19_count",
    "confidence_between_0_and_0.09_count",
]


def confidence_bucket_key(c: float) -> str:
    """Map a fastText confidence value to its histogram-bucket key.

    Buckets are deciles, and each key names the decile's bounds, so e.g. the
    "0.8_and_0.89" bucket covers [0.8, 0.9). Plain `int(c * 10)` truncation
    would mis-bucket exact decimal boundaries because of binary float
    representation (0.3 * 10 == 2.9999999999999996 -> int() -> 2), so values
    within 1e-9 of a decile boundary are snapped onto it before truncation.
    """
    if c >= 0.9:
        return "confidence_between_0.9_and_1_count"
    if c < 0.1:
        return "confidence_between_0_and_0.09_count"
    bucket = int(round(c * 10, 9))  # snap 2.9999999996 -> 3, leaving a real 0.2999 alone
    if bucket > int(c * 10) and abs(c * 10 - bucket) > 1e-9:
        bucket = int(c * 10)
    return f"confidence_between_{bucket/10:.1f}_and_{(bucket+1)/10 - 0.01:.2f}_count"


def to_native(obj: Any) -> Any:
    """Recursively convert common NumPy/Pandas objects into JSON-serializable Python types.

    Handles nested dicts/sequences and the most common scalar/container types produced
    by this pipeline, so you can `json.dump(...)` metrics without surprises.

    Rules:
      - dict → dict (values converted via recursion)
      - list/tuple/set → list (elements converted, and sets become lists for JSON)
      - numpy scalars → builtins (np.integer→int, np.floating→float, np.bool_→bool)
      - numpy arrays → list (with elements converted)
      - pandas.Timestamp → ISO 8601 string (or None if NaT)
      - pandas.Series → list

    Args:
        obj: Any object (typically metrics, counters, arrays) to convert.

    Returns:
        A JSON-serializable Python object.
    """
    # Mappings
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}

    # Sequences (coerce tuple/set to list for JSON)
    if isinstance(obj, (list, tuple, set)):
        return [to_native(x) for x in obj]

    # NumPy: arrays and scalars
    if isinstance(obj, np.ndarray):
        return [to_native(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # Pandas types
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, pd.Series):
        return [to_native(x) for x in obj.tolist()]

    return obj


def export_metrics_json(metrics: dict, path: Path, indent: int = 2) -> None:
    """Write a metrics dict as UTF-8 JSON, converting NumPy/Pandas scalars first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_native(metrics), fh, indent=indent, ensure_ascii=False)
