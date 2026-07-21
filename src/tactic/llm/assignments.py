"""Per-channel topic assignments and salience filtering.

Rebuilds each channel's aggregated documents, attaches the frozen model's
precomputed topic IDs, and writes per-channel topic distributions (unfiltered
and salience-filtered) plus the topic descriptor tables consumed by
consolidation.

Salience rule: keep topics covering at least ``salience_threshold`` percent of
a channel's micro-batches. If fewer than two non-outlier topics qualify, the
threshold is relaxed by one point until at least two do (or it reaches 0). A
channel whose messages are all outliers keeps its ``-1`` row so it still
reaches the classifier.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tactic.common.config import cfg_get
from tactic.topic_modeling.aggregation import build_aggregated_docs_df

# Base output filenames. The per-dataset `output_prefix` is prepended
# ("" -> TGDataset, "darkgram_" -> DarkGram, "tliveugr_" -> TLiveUGR).
_UNFILTERED_NAME = "channel_topic_assignments_unfiltered.csv"
_FILTERED_NAME = "channel_topic_assignments_filtered.csv"
_EXTRACTED_ALL_NAME = "extracted_topics.csv"
_FILTERED_EXTRACTED_NAME = "filtered_extracted_topics.csv"


def _load_input(cfg: dict) -> pd.DataFrame:
    """Load the LLM-input dataset (single CSV) or the sharded message directory."""
    input_mode = cfg_get(cfg, "params.input_mode", "single_csv")
    msg_col = cfg_get(cfg, "params.message_column", "processed_message")

    if input_mode == "shard_dir":
        folder = Path(cfg_get(cfg, "paths.input_dir"))
        files = sorted(p for p in folder.glob("*.csv") if p.is_file())
        if not files:
            raise FileNotFoundError(f"No CSV files found in: {folder}")
        parts = [
            pd.read_csv(
                p,
                usecols=["Channel ID", msg_col],
                dtype={"Channel ID": "string", msg_col: "string"},
                encoding="utf-8",
            )
            for p in files
        ]
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(cfg_get(cfg, "paths.input_csv"), dtype={"Channel ID": "string"})

    if msg_col != "processed_message":
        df = df.rename(columns={msg_col: "processed_message"})
    # Mirror the apply stage, which coerces messages to str: a null message
    # must not crash the micro-batch join, and every row must be kept so the
    # rebuilt doc sequence aligns positionally with topics.pkl.
    df["processed_message"] = df["processed_message"].astype(str)
    return df


def _filter_assignments(channel_topic_counts: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Salience-filter per channel (see the module docstring for the rule)."""
    filtered_rows = []
    for _, sub in channel_topic_counts.groupby("Channel ID", sort=True):
        temp = threshold
        cur = sub[sub["topic_percentage"] >= temp]
        non_out = cur[cur["topic"] != -1]

        while len(non_out) < 2 and temp > 0:
            temp -= 1
            cur = sub[sub["topic_percentage"] >= temp]
            non_out = cur[cur["topic"] != -1]

        out_only = cur[cur["topic"] == -1]
        chosen = non_out if len(non_out) >= 1 else out_only
        filtered_rows.append(chosen)

    return pd.concat(filtered_rows, ignore_index=True)


def run(cfg: dict) -> None:
    """Execute the topic-assignment stage for one dataset from a resolved config."""
    import joblib

    from tactic.topic_modeling.apply import load_existing_model

    agg_size = int(cfg_get(cfg, "params.agg_size", 10))
    threshold = int(cfg_get(cfg, "params.salience_threshold", 25))
    prefix = cfg_get(cfg, "params.output_prefix", "")

    out_dir = Path(cfg_get(cfg, "paths.output_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🔹 Loading dataset...")
    df = _load_input(cfg)

    print("🔹 Aggregating messages per Channel ID...")
    agg_df = build_aggregated_docs_df(df, agg_size)
    print(f"   → Built {len(agg_df)} aggregated documents across {agg_df['Channel ID'].nunique()} channels.")

    print("🔹 Loading BERTopic model + topics for aggregated docs...")
    topic_model = load_existing_model(cfg_get(cfg, "paths.model_dir"))
    topics = joblib.load(str(Path(cfg_get(cfg, "paths.topics_pkl"))))

    if len(topics) != len(agg_df):
        raise SystemExit(
            f"Length mismatch: topics ({len(topics)}) vs aggregated docs ({len(agg_df)}). "
            "The aggregated-docs construction must match the one used to produce topics.pkl."
        )

    agg_df["topic"] = topics

    print("📋 Extracting topic info (all topics)...")
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(out_dir / f"{prefix}{_EXTRACTED_ALL_NAME}", index=False)

    print("🔄 Calculating topic distributions per Channel ID (by chunk count)...")
    channel_topic_counts = (
        agg_df.groupby(["Channel ID", "topic"])
        .size()
        .reset_index(name="chunk_count")
    )
    total_chunks_per_channel = (
        agg_df.groupby("Channel ID")
        .size()
        .reset_index(name="total_chunks")
    )
    channel_topic_counts = channel_topic_counts.merge(
        total_chunks_per_channel, on="Channel ID", how="left"
    )
    channel_topic_counts["topic_percentage"] = (
        channel_topic_counts["chunk_count"] / channel_topic_counts["total_chunks"] * 100.0
    )
    channel_topic_counts.to_csv(out_dir / f"{prefix}{_UNFILTERED_NAME}", index=False)

    print("🔎 Filtering topic-channel assignments by chunk share...")
    filtered_assignments = _filter_assignments(channel_topic_counts, threshold=threshold)
    filtered_assignments.to_csv(out_dir / f"{prefix}{_FILTERED_NAME}", index=False)

    print("📁 Filtering topic representations to assigned topics...")
    filtered_topic_ids = filtered_assignments["topic"].unique().tolist()
    topic_info_renamed = topic_info.rename(columns={"Topic": "topic"})
    filtered_topics = topic_info_renamed[topic_info_renamed["topic"].isin(filtered_topic_ids)]
    filtered_topics.to_csv(out_dir / f"{prefix}{_FILTERED_EXTRACTED_NAME}", index=False)

    print("✅ Done. Saved four CSVs to", out_dir)
