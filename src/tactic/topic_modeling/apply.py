"""Apply the frozen final BERTopic model to a dataset (transform only)."""

from __future__ import annotations

import glob
import os

import joblib
import numpy as np
import pandas as pd

from tactic.topic_modeling import aggregation

REQUIRED_COLS = ["Channel ID", "message"]


def load_existing_model(model_dir: str):
    """
    Load a previously trained BERTopic model from model_dir.
    Tries model.pkl first, then model.safetensors.
    """
    from bertopic import BERTopic

    candidates = ["model.pkl", "model.safetensors"]
    last_err = None
    topic_model = None
    for fname in candidates:
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            topic_model = BERTopic.load(path)
            break
        except Exception as e:
            last_err = e
            topic_model = None
    if topic_model is None:
        raise FileNotFoundError(
            f"Could not load BERTopic model from {model_dir}. "
            f"Tried: {', '.join(candidates)}. Last error: {last_err}"
        )
    return topic_model


def _load_docs_single_csv(dataset_path: str) -> list:
    """Single translated CSV → coalesce processed_message → micro-batch."""
    # Read Channel ID as string so the sorted micro-batch order matches the
    # topic-assignment stage (which also reads it as string). Otherwise the
    # positional topics.pkl zip would silently misalign.
    df = pd.read_csv(dataset_path, dtype={"Channel ID": "string"})
    print(f"Loaded {len(df)} messages before filtering.")

    df = aggregation.coalesce_processed_message(df)
    print(f"Retained {len(df)} messages after filtering null processed messages.")

    grouped_docs = aggregation.build_grouped_docs(df)
    print(f"✅ Created {len(grouped_docs)} aggregated documents (10 messages each, grouped by channel).")
    return grouped_docs


def _load_docs_csv_folder(folder: str) -> list:
    """Sharded input: merge all shard CSVs, micro-batch the `message` column."""
    csv_paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    parts = []
    for p in csv_paths:
        df = pd.read_csv(p, encoding="utf-8", dtype={"Channel ID": "string"})
        missing = set(REQUIRED_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"{os.path.basename(p)} is missing required columns: {sorted(missing)}")
        # Keep only the columns we need (preserve order)
        df = df[REQUIRED_COLS].copy()
        parts.append(df)

    combined = pd.concat(parts, axis=0, ignore_index=True)
    print(f"Loaded {len(csv_paths)} CSV(s). Combined rows (in memory): {len(combined):,}")

    combined = combined.copy()
    combined["processed_message"] = combined["message"].astype(str)

    grouped_docs = aggregation.build_grouped_docs(combined)
    print(f"✅ Created {len(grouped_docs):,} aggregated documents (10 msgs each, grouped by channel).")
    return grouped_docs


def run_apply(cfg: dict) -> None:
    """Transform the frozen model onto a dataset and save the inference topics."""
    paths = cfg["paths"]
    params = cfg.get("params", {})

    output_dir = paths["output_dir"]
    os.makedirs(output_dir, exist_ok=True)  # ensure output dir exists

    input_mode = params.get("input_mode", "single_csv")
    if input_mode == "single_csv":
        docs = _load_docs_single_csv(paths["dataset_csv"])
    elif input_mode == "csv_folder":
        docs = _load_docs_csv_folder(paths["dataset_dir"])
    else:
        raise ValueError(f"Unknown params.input_mode: {input_mode!r}")

    if not docs:
        raise ValueError("No documents after preprocessing, so there is nothing to transform.")

    embeddings_path = paths["doc_embeddings"]
    if os.path.exists(embeddings_path):
        print("Loading existing document embeddings...")
        doc_embeddings = np.array(joblib.load(embeddings_path))
        # A stale cache of the wrong length would silently mis-align the transform.
        if len(doc_embeddings) != len(docs):
            raise ValueError(
                f"Embeddings count ({len(doc_embeddings)}) != docs count ({len(docs)}). "
                "Delete the embeddings file to recompute."
            )
    else:
        from sentence_transformers import SentenceTransformer
        from tactic.topic_modeling.embeddings import get_device_str

        device = get_device_str()
        print(f"Computing document embeddings on {device}...")
        embedding_model = SentenceTransformer(
            params.get("embedding_model", "all-MiniLM-L6-v2"), device=device
        )
        doc_embeddings = np.array(
            embedding_model.encode(docs, show_progress_bar=True, device=device)
        )
        joblib.dump(doc_embeddings, embeddings_path)
        print(f"✅ Saved embeddings → {embeddings_path}")

    topic_model = load_existing_model(paths["model_dir"])

    topics, _ = topic_model.transform(docs, embeddings=doc_embeddings)

    joblib.dump(topics, os.path.join(output_dir, "topics.pkl"))
    print(f"✅ Inference outputs saved in → {output_dir}")
