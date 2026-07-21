"""Shared BERTopic training driver: three-phase tuning, validation re-fits and
the final fit. Each stage is driven by its YAML config (params.phase selects
the grid builder)."""

from __future__ import annotations

import hashlib
import json
import os
import traceback

import joblib
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from tactic.topic_modeling import aggregation
from tactic.topic_modeling import grids
from tactic.topic_modeling.embeddings import get_or_compute_embeddings
from tactic.topic_modeling.run_log import log_run
from tactic.topic_modeling.stopwords import STOPWORDS

# This is done to disable a warning that appeared because of using multithreading when training the model
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- Load Dataset ---
def load_dataset(dataset_path: str) -> list:
    """Read the dataset CSV (pandas default dtypes, so "Channel ID" parses as
    int64, which fixes the sorted channel iteration order), coalesce
    processed_message, and micro-batch every 10 consecutive messages per
    channel."""
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} messages before filtering.")

    df = aggregation.coalesce_processed_message(df)
    print(f"Retained {len(df)} messages after filtering null processed messages.")

    # --- Group every 10 messages per channel ---
    grouped_docs = aggregation.build_grouped_docs(df, agg_size=10)
    print(f"✅ Created {len(grouped_docs)} aggregated documents (10 messages each, grouped by channel).")
    return grouped_docs


# --- Create BERTopic Model ---
def create_topic_model(docs, doc_embeddings, umap_args, hdbscan_args, vectorizer_args,
                       embedding_model_name: str):
    """Build a BERTopic model from the given UMAP/HDBSCAN/vectorizer arguments
    and fit it on the pre-computed document embeddings."""
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from sentence_transformers import SentenceTransformer

    umap_model = UMAP(**umap_args)
    hdbscan_model = HDBSCAN(**hdbscan_args)
    vectorizer_model = CountVectorizer(**vectorizer_args)
    embedding_model = SentenceTransformer(embedding_model_name)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=KeyBERTInspired(),
        embedding_model=embedding_model,
        calculate_probabilities=True,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=doc_embeddings)

    return topic_model, topics, probs


# --- Generate Short Hash ID ---
def short_hash(params):
    """Short config hash embedded in model directory names: the first 6 hex
    chars of md5(json.dumps(params, sort_keys=True))."""
    json_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()[:6]


# --- Run Individual Model Configuration ---
def run_model_wrapper(args, docs, doc_embeddings, log_path, output_dir,
                      model_name_pattern, embedding_model_name):
    """
    Runs one BERTopic model configuration, saves outputs, and logs evaluation.

    `model_name_pattern` sets the per-stage naming:
      phase 1/2/3:  "bertopic_{uid}"
      final:        "bertopic_final_{uid}"
      validation:   "bertopic_validation<N>" (fixed name, no hash, so re-runs
                    overwrite the model directory).
    """
    umap_args, hdbscan_args, vectorizer_args = args

    # Remove stop_words, we don't want to log them
    vectorizer_args_no_stopwords = dict(vectorizer_args)
    vectorizer_args_no_stopwords.pop("stop_words", None)

    params = {
        'umap': umap_args,
        'hdbscan': hdbscan_args,
        'vectorizer': vectorizer_args_no_stopwords,
        'representations': ['keybert'],
    }
    uid = short_hash(params)
    model_name = model_name_pattern.format(uid=uid)

    try:
        model, topics, probs = create_topic_model(
            docs, doc_embeddings, umap_args, hdbscan_args, vectorizer_args,
            embedding_model_name
        )

        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Save model and artifacts
        model.save(os.path.join(model_dir, "model.pkl"), serialization="pickle")
        model.save(
            os.path.join(model_dir, "model.safetensors"),
            serialization = "safetensors",
            save_ctfidf = True,
            save_embedding_model = "sentence-transformers/" + embedding_model_name
        )

        joblib.dump(topics, os.path.join(model_dir, "topics.pkl"))

        # Logging
        log_run(model_name, params, log_path)

        print(f"[SUCCESS] {model_name}")
    except Exception as e:
        print(f"[ERROR] {model_name}: {e}")
        print(traceback.format_exc())


def _build_configs(phase: str, docs: list, paths: dict, target_topics: int,
                   min_cluster_sizes: list | None = None) -> list:
    """Dispatch to the per-stage grid builder (see grids.py).

    `min_cluster_sizes` (params.min_cluster_sizes, default null = the built-in
    sweep) restricts the phase-1/final min_cluster_size sweep. For example,
    [500] trains only the published winning configuration.
    """
    if phase == "phase1":
        return grids.build_phase1_configs(docs, STOPWORDS,
                                          min_cluster_sizes=min_cluster_sizes)
    if phase == "phase2":
        return grids.build_phase2_configs(
            paths["prev_metrics"], paths["prev_scored"], paths["prev_log"],
            STOPWORDS, target_topics,
        )
    if phase == "phase3":
        return grids.build_phase3_configs(
            paths["prev_metrics"], paths["prev_scored"], paths["prev_log"],
            STOPWORDS, target_topics,
        )
    if phase == "validation":
        return grids.build_validation_config(
            paths["prev_metrics"], paths["prev_scored"], paths["prev_log"],
            STOPWORDS, target_topics,
        )
    if phase == "final":
        return grids.build_final_configs(
            docs, paths["prev_metrics"], paths["prev_scored"], paths["prev_log"],
            STOPWORDS, target_topics,
            min_cluster_sizes=min_cluster_sizes,
        )
    raise ValueError(f"Unknown params.phase: {phase!r}")


def run_training_stage(cfg: dict) -> None:
    """Run one training stage end-to-end from its YAML config."""
    paths = cfg["paths"]
    params = cfg.get("params", {})

    phase = params["phase"]
    output_dir = paths["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    docs = load_dataset(paths["dataset_csv"])

    doc_embeddings = get_or_compute_embeddings(
        docs,
        paths["doc_embeddings"],
        mode=params.get("embeddings_mode", "compute_if_missing"),
        embedding_model_name=params.get("embedding_model", "all-MiniLM-L6-v2"),
    )

    config_list = _build_configs(
        phase, docs, paths,
        target_topics=params.get("target_topics", 45),
        min_cluster_sizes=params.get("min_cluster_sizes"),
    )

    n_jobs = params.get("n_jobs", 4)
    desc = params.get("progress_desc", f"{phase} Runs")
    print(f"Running {phase} with {len(config_list)} configurations and {n_jobs} parallel jobs")

    wrapper_kwargs = dict(
        docs=docs,
        doc_embeddings=doc_embeddings,
        log_path=paths["run_log"],
        output_dir=output_dir,
        model_name_pattern=params.get("model_name_pattern", "bertopic_{uid}"),
        embedding_model_name=params.get("embedding_model", "all-MiniLM-L6-v2"),
    )

    if params.get("parallel", True):
        Parallel(n_jobs=n_jobs)(
            delayed(run_model_wrapper)(args, **wrapper_kwargs)
            for args in tqdm(config_list, desc=desc)
        )
    else:
        # The validation stage runs its single configuration serially.
        for args in tqdm(config_list, desc=desc):
            run_model_wrapper(args, **wrapper_kwargs)
