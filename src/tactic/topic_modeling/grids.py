"""Hyperparameter-grid builders for each tuning stage (phase 1/2/3, validation, final).

Each builder returns a list of (umap_args, hdbscan_args, vectorizer_args)
tuples in deterministic construction order."""

from __future__ import annotations

import pandas as pd

from tactic.topic_modeling.run_log import normalize_vectorizer_args, recover_config_field
from tactic.topic_modeling.scoring import score_and_save


def _dedupe_configs(config_list: list) -> list:
    """Remove duplicate configurations, preserving first-occurrence order.

    The document-ratio-derived min_cluster_size values can collide with the
    fixed grid. Duplicate configs would hash to the same model name, so
    parallel workers could train and write the same model directory
    concurrently.
    """
    seen = set()
    out = []
    for args in config_list:
        key = repr(args)
        if key not in seen:
            seen.add(key)
            out.append(args)
    return out


def build_phase1_configs(docs: list, stopwords: list,
                         min_cluster_sizes: list | None = None) -> list:
    """Phase-1 coarse UMAP/HDBSCAN grid on the tuning sample."""
    umap_neighbors = [20, 30, 50, 75, 100, 200]
    if min_cluster_sizes is not None:
        hdbscan_cluster_sizes = list(min_cluster_sizes)
    else:
        # Fixed sweep plus three document-ratio-derived values and 1000.
        hdbscan_cluster_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500] + [int(len(docs) / r) for r in [60, 80, 100]] + [1000]
    hdbscan_min_samples = [5, 15, 20]

    umap_configs = [
        {'n_neighbors': n, 'min_dist': 0.1, 'n_components': 5, 'metric': 'cosine', 'random_state': 42}
        for n in umap_neighbors
    ]

    hdbscan_configs = [
        {'min_cluster_size': mcs, 'min_samples': ms, 'cluster_selection_method': 'eom', 'cluster_selection_epsilon': 0.0, 'prediction_data': True}
        for mcs in hdbscan_cluster_sizes
        for ms in hdbscan_min_samples
    ]

    vectorizer_configs = [
        {'ngram_range': (1, 1), 'min_df': 0.001, 'max_df': 0.9, 'stop_words': stopwords}
    ]

    config_list = [
        (umap, hdb, vec)
        for umap in umap_configs
        for hdb in hdbscan_configs
        for vec in vectorizer_configs
    ]
    return _dedupe_configs(config_list)


def build_phase2_configs(
    stage1_metrics_path: str,
    stage1_scored_path: str,
    stage1_log_path: str,
    stopwords: list,
    target_topics: int,
) -> list:
    """Phase-2 refinement grid (n_components × epsilon) around the stage-1 top-3."""
    metrics_df = score_and_save(stage1_metrics_path, stage1_scored_path, target_topics)

    # Sort by final score
    top_models = metrics_df.sort_values(by="final_score", ascending=False).head(3)["model_name"].tolist()

    # Get their configs
    # NOTE: .isin() keeps log-file row order, not ranking order.
    log_df = pd.read_csv(stage1_log_path)
    top_configs = log_df[log_df["model_name"].isin(top_models)].to_dict(orient="records")

    # Build new config combinations
    stage2_configs = []
    for config in top_configs:
        umap_args = recover_config_field(config['umap'])
        hdbscan_base = recover_config_field(config['hdbscan'])
        vectorizer_args = normalize_vectorizer_args(recover_config_field(config['vectorizer']))

        vectorizer_args['stop_words'] = stopwords

        for nc in [5, 7, 10, 15, 30]:
            for eps in [0.0, 0.1]:
                hdbscan_args = hdbscan_base.copy()
                hdbscan_args['cluster_selection_epsilon'] = eps

                umap_new = umap_args.copy()
                umap_new['n_components'] = nc

                stage2_configs.append((umap_new, hdbscan_args, vectorizer_args))

    return stage2_configs


def build_phase3_configs(
    stage2_metrics_path: str,
    stage2_scored_path: str,
    stage2_log_path: str,
    stopwords: list,
    target_topics: int,
) -> list:
    """Phase-3 unigram vs unigram+bigram sweep on the stage-2 winner."""
    metrics_df = score_and_save(stage2_metrics_path, stage2_scored_path, target_topics)

    top_model_name = metrics_df.sort_values(by="final_score", ascending=False).iloc[0]["model_name"]
    print(f"⭐ Top Stage 2 model: {top_model_name}")

    # Retrieve config for top model
    stage2_log = pd.read_csv(stage2_log_path)
    top_config = stage2_log[stage2_log["model_name"] == top_model_name].iloc[0]
    umap_args = recover_config_field(top_config['umap'])
    hdbscan_args = recover_config_field(top_config['hdbscan'])

    # Two vectorizer configurations
    vectorizer_configs = [
        {'ngram_range': (1, 1), 'min_df': 0.001, 'max_df': 0.9, 'stop_words': stopwords},
        {'ngram_range': (1, 2), 'min_df': 0.001, 'max_df': 0.9, 'stop_words': stopwords},
    ]

    stage3_configs = [(umap_args, hdbscan_args, vec) for vec in vectorizer_configs]
    return stage3_configs


def build_validation_config(
    stage3_metrics_path: str,
    stage3_scored_path: str,
    stage3_log_path: str,
    stopwords: list,
    target_topics: int,
) -> list:
    """Single validation config = the stage-3 winning configuration."""
    metrics_df = score_and_save(stage3_metrics_path, stage3_scored_path, target_topics)

    top_model_name = metrics_df.sort_values(by="final_score", ascending=False).iloc[0]["model_name"]
    print(f"⭐ Top Stage 3 model: {top_model_name}")

    # Retrieve config for top model
    stage3_log = pd.read_csv(stage3_log_path)
    top_config = stage3_log[stage3_log["model_name"] == top_model_name].iloc[0]
    umap_args = recover_config_field(top_config['umap'])
    hdbscan_args = recover_config_field(top_config['hdbscan'])
    vectorizer_args = normalize_vectorizer_args(recover_config_field(top_config['vectorizer']))
    vectorizer_args['stop_words'] = stopwords

    return [(umap_args, hdbscan_args, vectorizer_args)]


def build_final_configs(
    docs: list,
    stage3_metrics_path: str,
    stage3_scored_path: str,
    stage3_log_path: str,
    stopwords: list,
    target_topics: int,
    min_cluster_sizes: list | None = None,
) -> list:
    """Final-fit min_cluster_size sweep on the stage-3 winner.

    On the 2M sample only min_cluster_size is re-tuned. Every other
    hyperparameter is inherited from the stage-3 winning configuration. The
    min_cluster_size grid is deduped and sorted. An explicit
    `min_cluster_sizes` list restricts the sweep ([500] trains only the
    published winning configuration).
    """
    metrics_df = score_and_save(stage3_metrics_path, stage3_scored_path, target_topics)

    top_model_name = metrics_df.sort_values(by="final_score", ascending=False).iloc[0]["model_name"]
    print(f"⭐ Top Stage 3 model: {top_model_name}")

    # Retrieve config for top model
    stage3_log = pd.read_csv(stage3_log_path)
    top_config = stage3_log[stage3_log["model_name"] == top_model_name].iloc[0]

    if min_cluster_sizes is not None:
        mcs_list = list(min_cluster_sizes)
    else:
        # Fixed sweep plus three document-ratio-derived values.
        mcs_list = (
            [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000] +
            [int(len(docs) / r) for r in [60, 80, 100]]
        )
    hdbscan_cluster_sizes = sorted({m for m in mcs_list if m >= 2})

    # Build new config combinations
    stage3_configs = []
    for mcs in hdbscan_cluster_sizes:
        umap_args = recover_config_field(top_config['umap'])
        hdbscan_base = recover_config_field(top_config['hdbscan'])
        vectorizer_args = normalize_vectorizer_args(recover_config_field(top_config['vectorizer']))

        vectorizer_args['stop_words'] = stopwords

        hdbscan_args = hdbscan_base.copy()
        hdbscan_args['min_cluster_size'] = mcs

        stage3_configs.append((umap_args, hdbscan_args, vectorizer_args))

    return stage3_configs
