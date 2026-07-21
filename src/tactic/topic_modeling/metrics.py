"""Metrics engine: coherence / silhouette / outlier rate / topic count."""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from tactic.topic_modeling import aggregation

COHERENCE_MEASURE = "c_v"


def safe_silhouette_score(embeddings, labels, metric="cosine"):
    """Returns NaN if silhouette cannot be computed (e.g., clusters with <2 samples)."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    mask = labels != -1  # remove outliers
    filtered_labels = labels[mask]
    filtered_embeddings = np.asarray(embeddings)[mask]

    if filtered_embeddings.shape[0] < 2:
        return float('nan')

    unique, counts = np.unique(filtered_labels, return_counts=True)
    # need >= 2 clusters
    if len(unique) < 2:
        return float('nan')
    # each cluster needs >= 2 samples
    if np.any(counts < 2):
        return float('nan')

    try:
        return silhouette_score(filtered_embeddings, filtered_labels, metric=metric)
    except ValueError:
        return float('nan')


def topics_as_wordlists(topic_model, top_n=10):
    """Extracts List[List[str]] for Gensim from BERTopic.get_topics(), skipping -1 and empties."""
    topics_dict = topic_model.get_topics()
    out = []
    for topic_id, pairs in topics_dict.items():
        if topic_id == -1 or not pairs:
            continue
        words = [w for (w, _) in pairs[:top_n] if isinstance(w, str) and w]
        if words:
            out.append(words)
    return out


def _hashable(v):
    """
    Convert a possibly unhashable object (like list, set, function)
    into a hashable, stable representation so it can be used in a dict key.

    Why? In Python, dict keys must be hashable (immutable + __hash__ defined).
    Lists/sets/dicts are NOT hashable → TypeError if we try to use them directly.
    """
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool, tuple)):
        return v # already hashable
    if isinstance(v, (list, set)):
        # Convert to sorted tuple so order doesn't affect the key
        try:
            return tuple(sorted(v))
        except Exception:
            return tuple(v)
    # For functions or other objects, use their name if available
    try:
        return getattr(v, "__name__", repr(v))
    except Exception:
        return repr(v)


def vectorizer_cache_key(vec):
    """
    Create a tuple that uniquely identifies the tokenization settings
    of a CountVectorizer (the parts that affect how text is split into tokens).

    We intentionally ignore parameters like min_df/max_df because
    they affect vocabulary filtering, NOT tokenization.
    """
    params = vec.get_params()
    key_fields = [
        "analyzer", "lowercase", "stop_words", "token_pattern",
        "ngram_range", "preprocessor", "tokenizer", "strip_accents"
    ]
    # Build tuple of (param_name, param_value) where value is hashable
    key = tuple((f, _hashable(params.get(f, None))) for f in key_fields)
    return key


def _rebuild_docs(dataset_path: str) -> list:
    """Rebuild the aggregated docs exactly as the training/apply stages built them."""
    df = pd.read_csv(dataset_path)
    df = aggregation.coalesce_processed_message(df)
    grouped_docs = aggregation.build_grouped_docs(df)
    return grouped_docs


def _compute_coherence(topic_model, tokenized_docs, dictionary):
    from gensim.models.coherencemodel import CoherenceModel

    top_words = topics_as_wordlists(topic_model, top_n=10)
    if len(top_words) == 0:
        print("    ⚠️ No valid topics found for coherence (after skipping -1).")
        return float('nan')
    cm = CoherenceModel(
        topics=top_words,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence=COHERENCE_MEASURE
    )
    return cm.get_coherence()


def run_metrics_multi(cfg: dict) -> None:
    """Multi-model metrics over a run log."""
    from bertopic import BERTopic
    from gensim.corpora import Dictionary

    paths = cfg["paths"]

    print("🔹 Loading dataset and document embeddings...")
    docs = _rebuild_docs(paths["dataset_csv"])

    # Embeddings must correspond to 'docs' ordering/content
    doc_embeddings = np.array(joblib.load(paths["doc_embeddings"]))
    if len(doc_embeddings) != len(docs):
        raise ValueError(
            f"Embeddings count ({len(doc_embeddings)}) != docs count ({len(docs)}). "
            "Ensure they were computed over the same grouped docs."
        )
    print(f"✅ Loaded {len(docs)} documents.")

    print("🔹 Loading model log...")
    log_df = pd.read_csv(paths["run_log"])
    print(f"✅ Found {len(log_df)} models to evaluate.")

    # Cache: vec_config_key -> (tokenized_docs, dictionary)
    vectorizer_cache = {}
    results = []

    # --- Evaluation loop ---
    for idx, row in tqdm(log_df.iterrows(), total=len(log_df), desc="Evaluating Models"):
        model_name = row["model_name"]
        model_dir = os.path.join(paths["models_dir"], model_name)

        try:
            print(f"\n🔍 Evaluating model: {model_name}")

            # Load model and artifacts
            topic_model = BERTopic.load(os.path.join(model_dir, "model.pkl"))
            topics = joblib.load(os.path.join(model_dir, "topics.pkl"))

            # --- Build/reuse tokens & dictionary per vectorizer config ---
            vec = topic_model.vectorizer_model
            key = vectorizer_cache_key(vec)

            if key not in vectorizer_cache:
                print("  • Tokenizing docs for NEW vectorizer config (cached afterwards)...")
                analyzer = vec.build_analyzer()
                tokenized_docs = [analyzer(d) for d in docs]
                dictionary = Dictionary(tokenized_docs)
                vectorizer_cache[key] = (tokenized_docs, dictionary)
            else:
                tokenized_docs, dictionary = vectorizer_cache[key]

            # --- Metric 1: Coherence ---
            print(f"  • Calculating coherence score ({COHERENCE_MEASURE})...")
            coherence = _compute_coherence(topic_model, tokenized_docs, dictionary)

            # --- Metric 2: Silhouette Score (raw 384-d document embeddings) ---
            print("  • Calculating silhouette score (cosine)...")
            silhouette = safe_silhouette_score(doc_embeddings, topics, metric="cosine")

            # --- Metric 3: Outlier Rate ---
            outlier_rate = float(np.mean(np.array(topics) == -1))

            # --- Metric 4: Number of Topics ---
            num_topics = len(set(topics) - {-1})

            # Store result
            results.append({
                "model_name": model_name,
                "coherence_score": coherence,
                "silhouette_score": silhouette,
                "outlier_rate": outlier_rate,
                "num_topics": num_topics
            })

        except Exception as e:
            print(f"[❌ ERROR] Failed for {model_name}: {e}")

    # --- Save Results ---
    print("\n💾 Saving all model evaluation results...")
    os.makedirs(os.path.dirname(paths["metrics_csv"]) or ".", exist_ok=True)
    pd.DataFrame(results).to_csv(paths["metrics_csv"], index=False)
    print(f"✅ All metrics saved to {paths['metrics_csv']}")


def run_metrics_single(cfg: dict) -> None:
    """Single frozen-model metrics on inference outputs. Coherence here
    evaluates the training-corpus topic words against the target-dataset docs
    (transfer coherence)."""
    from bertopic import BERTopic
    from gensim.corpora import Dictionary

    paths = cfg["paths"]
    params = cfg.get("params", {})

    print("🔹 Loading dataset and document embeddings...")
    docs = _rebuild_docs(paths["dataset_csv"])

    doc_embeddings = np.array(joblib.load(paths["doc_embeddings"]))
    if len(doc_embeddings) != len(docs):
        raise ValueError(
            f"Embeddings count ({len(doc_embeddings)}) != docs count ({len(docs)}). "
            "Ensure embeddings were computed over the same grouped docs."
        )
    print(f"✅ Loaded {len(docs)} documents.")

    results = []
    model_path = paths["model_dir"]
    model_name = os.path.basename(os.path.normpath(model_path)) + params.get("model_name_suffix", "")

    try:
        print(f"\n🔍 Evaluating model: {model_name}")

        # Load the frozen model and the apply stage's inference topics
        topic_model = BERTopic.load(os.path.join(model_path, "model.pkl"))
        topics = joblib.load(paths["topics_pkl"])

        vec = topic_model.vectorizer_model
        analyzer = vec.build_analyzer()
        tokenized_docs = [analyzer(d) for d in docs]
        dictionary = Dictionary(tokenized_docs)

        # --- Metric 1: Coherence ---
        print(f"  • Calculating coherence score ({COHERENCE_MEASURE})...")
        coherence = _compute_coherence(topic_model, tokenized_docs, dictionary)

        # --- Metric 2: Silhouette Score (raw 384-d document embeddings) ---
        print("  • Calculating silhouette score (cosine)...")
        silhouette = safe_silhouette_score(doc_embeddings, topics, metric="cosine")

        # --- Metric 3: Outlier Rate ---
        outlier_rate = float(np.mean(np.array(topics) == -1))

        # --- Metric 4: Number of Topics ---
        num_topics = len(set(topics) - {-1})

        # Store result
        results.append({
            "model_name": model_name,
            "coherence_score": coherence,
            "silhouette_score": silhouette,
            "outlier_rate": outlier_rate,
            "num_topics": num_topics
        })

    except Exception as e:
        print(f"[❌ ERROR] Failed for {model_name}: {e}")

    print("\n💾 Saving model evaluation results...")
    os.makedirs(os.path.dirname(paths["metrics_csv"]) or ".", exist_ok=True)
    pd.DataFrame(results).to_csv(paths["metrics_csv"], index=False)
    print(f"✅ All metrics saved to {paths['metrics_csv']}")


def run_metrics(cfg: dict) -> None:
    """Dispatch on params.mode: 'multi' (run-log loop) or 'single' (frozen model)."""
    mode = cfg.get("params", {}).get("mode", "multi")
    if mode == "multi":
        run_metrics_multi(cfg)
    elif mode == "single":
        run_metrics_single(cfg)
    else:
        raise ValueError(f"Unknown params.mode: {mode!r}")
