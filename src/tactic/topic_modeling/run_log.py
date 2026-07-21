"""FileLock-guarded run-log CSVs and config recovery for the BERTopic trainers."""

from __future__ import annotations

import ast
import json
import os
from typing import Any

import pandas as pd
from filelock import FileLock


def log_run(model_name: str, params: dict, log_path: str) -> None:
    """Append one trained model's config to the stage run-log CSV.

    Column layout: model_name, umap, hdbscan, vectorizer, representations.
    Dict/list params fields are stored as JSON strings. json.dumps writes
    tuples as lists, so `(1, 1)` round-trips as `[1, 1]`, and recovery
    normalizes ngram_range back to a tuple (see `normalize_vectorizer_args`).
    The model-name hash also uses json.dumps, so tuples and lists hash
    identically there too.
    """
    log_entry = {'model_name': model_name}
    for key, value in params.items():
        log_entry[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
    log_df = pd.DataFrame([log_entry])

    with FileLock(f"{log_path}.lock"):
        if not os.path.exists(log_path):
            log_df.to_csv(log_path, index=False)
        else:
            log_df.to_csv(log_path, mode='a', header=False, index=False)


def recover_config_field(value: Any) -> Any:
    """Recover a params field from a run-log cell.

    JSON is tried first, and ast.literal_eval covers cells holding Python
    literal reprs (dicts containing tuples).
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return ast.literal_eval(value)


def normalize_vectorizer_args(vectorizer_args: dict) -> dict:
    """Coerce ngram_range back to a tuple after recovery.

    JSON round-trips tuples as lists, but CountVectorizer expects a tuple. The
    value is identical either way, and hashing is unaffected (json.dumps
    writes both identically).
    """
    if "ngram_range" in vectorizer_args and isinstance(vectorizer_args["ngram_range"], list):
        vectorizer_args["ngram_range"] = tuple(vectorizer_args["ngram_range"])
    return vectorizer_args
