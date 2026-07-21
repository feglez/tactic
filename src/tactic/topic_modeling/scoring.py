"""Composite-score computation for ranking candidate BERTopic configurations."""

from __future__ import annotations

import pandas as pd

# Target topic count T* used by the topic-count penalty.
TARGET_TOPICS = 45


def score_metrics_df(
    metrics_df: pd.DataFrame,
    target_topics: int = TARGET_TOPICS,
) -> pd.DataFrame:
    """Add sil_norm / topic_penalty / final_score columns.

    composite = 0.55*C + 0.20*S_norm - 0.35*O - 0.25*P
    """
    TARGET_TOPICS_local = target_topics

    # 1) Normalize silhouette score and prepare features
    metrics_df = metrics_df.copy()  # Avoid modifying the original DataFrame

    # Silhouette score ranges from -1 to 1. We shift and scale it to 0–1.
    # This makes it directly comparable to coherence and other metrics in the weighted sum
    metrics_df["sil_norm"] = (metrics_df["silhouette_score"] + 1) / 2
    metrics_df["sil_norm"] = metrics_df["sil_norm"].clip(0, 1) # Ensure no values fall outside [0,1] due to rounding

    # Fill NaNs conservatively
    metrics_df["coherence_score"] = metrics_df["coherence_score"].fillna(0.0)
    metrics_df["sil_norm"]        = metrics_df["sil_norm"].fillna(0.0)
    metrics_df["outlier_rate"]    = metrics_df["outlier_rate"].fillna(1.0)  # worst case
    metrics_df["num_topics"]      = metrics_df["num_topics"].fillna(0)

    # 2) Topic penalty: measures how far the number of topics is from the target
    # Relative deviation = |actual - target| / target
    # Example: If TARGET_TOPICS = 45 and model has 60 topics → deviation = |60-45|/45 = 0.333
    # Clip to [0,1] so extremely off-topic counts don't overly dominate
    rel_dev = (metrics_df["num_topics"] - TARGET_TOPICS_local).abs() / max(TARGET_TOPICS_local, 1)
    metrics_df["topic_penalty"] = rel_dev.clip(0, 1)

    # 3) Weighted final score
    #    Weight rationale for Telegram channel classification:
    #    - Coherence (0.55): top priority. Telegram messages are short and noisy, so
    #      the clarity and interpretability of topics matter most.
    #    - Silhouette (0.20): useful for measuring separation in embedding space, but
    #      less stable for short, informal texts.
    #    - Outlier rate (-0.35): strong penalty. We want the most messages assigned to a topic
    #      to ensure coverage and avoid fragmented clusters.
    #    - Topic penalty (-0.25): penalizes deviation from the target topic count, but less
    #      than coherence/outlier impact, so flexibility is still allowed if quality is high.
    metrics_df["final_score"] = (
        0.55 * metrics_df["coherence_score"] +
        0.20 * metrics_df["sil_norm"] -
        0.35 * metrics_df["outlier_rate"] -
        0.25 * metrics_df["topic_penalty"]
    )

    return metrics_df


def score_and_save(
    metrics_path: str,
    scored_path: str,
    target_topics: int = TARGET_TOPICS,
) -> pd.DataFrame:
    """Read a metrics CSV, score it, save the scored CSV."""
    metrics_df = pd.read_csv(metrics_path)
    metrics_df = score_metrics_df(metrics_df, target_topics)
    metrics_df.to_csv(scored_path, index=False)
    print(f"✅ Saved updated metrics with final scores to {scored_path}")
    return metrics_df


def run_score_stage(cfg: dict) -> str:
    """Entry point for the standalone scoring stage: reads the final-sweep
    metrics, saves the scored CSV, prints the winner."""
    paths = cfg["paths"]
    params = cfg.get("params", {})
    target = params.get("target_topics", TARGET_TOPICS)

    metrics_df = score_and_save(paths["metrics_csv"], paths["scored_csv"], target)

    top_model_name = metrics_df.sort_values(by="final_score", ascending=False).iloc[0]["model_name"]
    print(f"⭐ Top model: {top_model_name}")
    return top_model_name
