"""Consolidate one channel-level row per channel for the LLM classification stage.

Joins channel metadata, filtered topic assignments and topic descriptors
(plus the ground-truth category for TGDataset) into one row per channel,
with the topic fields stored as stringified Python lists.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tactic.common.config import cfg_get


def _load_channel_metadata(cfg: dict) -> pd.DataFrame:
    """One row per channel with the first-observed metadata values.

    ``params.metadata_source`` selects the input: ``llm_input_csv`` reads a
    single messages CSV (``paths.messages_csv``), while ``shard_dir`` reads every
    ``*.csv`` shard in ``paths.metadata_dir``.
    """
    metadata_columns = list(cfg_get(cfg, "params.metadata_columns", ["Username"]))
    source = cfg_get(cfg, "params.metadata_source", "llm_input_csv")

    if source == "shard_dir":
        folder = Path(cfg_get(cfg, "paths.metadata_dir"))
        files = sorted(p for p in folder.glob("*.csv") if p.is_file())
        if not files:
            raise FileNotFoundError(f"No CSV files found in: {folder}")
        dfs = []
        for p in files:
            dfs.append(
                pd.read_csv(
                    p,
                    usecols=["Channel ID"] + metadata_columns,
                    dtype={c: "string" for c in ["Channel ID"] + metadata_columns},
                    encoding="utf-8",
                )
            )
        messages_df = pd.concat(dfs, ignore_index=True)
        messages_df["Channel ID"] = messages_df["Channel ID"].astype("string")
    else:
        # Channel ID must stay a string on every read: a numeric round-trip
        # can change the key text (e.g. "123" -> "123.0") and break joins
        # against the string-dtype reads used by the other stages.
        messages_df = pd.read_csv(
            cfg_get(cfg, "paths.messages_csv"), dtype={"Channel ID": "string"}
        )

    return messages_df


def run(cfg: dict) -> None:
    """Execute the consolidation stage for one dataset from a resolved config."""
    metadata_columns = list(cfg_get(cfg, "params.metadata_columns", ["Username"]))
    keep_only_assigned = bool(cfg_get(cfg, "params.keep_only_assigned", False))
    category_map_path = cfg_get(cfg, "paths.category_map", None)
    out_path = Path(cfg_get(cfg, "paths.output_csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("🔹 Loading input files...")
    messages_df = _load_channel_metadata(cfg)
    assignments_df = pd.read_csv(cfg_get(cfg, "paths.assignments_csv"))
    topics_df = pd.read_csv(cfg_get(cfg, "paths.topics_csv"))

    assignments_df["Channel ID"] = assignments_df["Channel ID"].astype(str)

    # --- Optionally keep only channels present in the assignments table ---
    if keep_only_assigned:
        keep_ids = set(assignments_df["Channel ID"].unique())
        messages_df = messages_df[messages_df["Channel ID"].isin(keep_ids)].reset_index(drop=True)

    # --- Channel metadata: first observed value per channel ---
    print("🔄 Extracting channel metadata...")
    channel_metadata = (
        messages_df
        .groupby("Channel ID", as_index=False)
        .agg({c: "first" for c in metadata_columns})
    )

    # --- Optional ground-truth category join (TGDataset) ---
    if category_map_path:
        # ch_ID -> Channel ID, topic -> Category (the curated TGDataset label).
        print("📚 Merging category labels...")
        category_map_df = pd.read_csv(category_map_path)
        category_map_df = category_map_df.rename(columns={"ch_ID": "Channel ID", "topic": "Category"})
        category_map_df["Channel ID"] = category_map_df["Channel ID"].astype(str)
        channel_metadata = channel_metadata.merge(
            category_map_df[["Channel ID", "Category"]],
            on="Channel ID",
            how="left",
        )

    # --- Normalize topic id column name ---
    if "Topic" in topics_df.columns and "topic" not in topics_df.columns:
        topics_df = topics_df.rename(columns={"Topic": "topic"})

    # --- Enrich assignments with topic names/representations ---
    print("📦 Enriching topic assignments with names/representations...")
    assignments_with_meta = assignments_df.merge(
        topics_df[["topic", "Name", "Representation"]],
        on="topic",
        how="left",
    )

    # --- Aggregate topic-level data per Channel ID (parallel lists) ---
    print("📊 Aggregating topic data by Channel ID...")
    agg_topics = (
        assignments_with_meta
        .groupby("Channel ID", as_index=False)
        .agg({
            "topic": lambda x: list(x),
            "Name": lambda x: list(x),
            "Representation": lambda x: list(x),
            "topic_percentage": lambda x: list(x),
        })
        .rename(columns={
            "topic": "topics",
            "Name": "topic_names",
            "Representation": "representations",
            "topic_percentage": "topic_percentages",
        })
    )

    # --- Total chunks per channel (dedupe) ---
    print("🧮 Merging total_chunks per Channel ID...")
    total_chunks_df = assignments_df[["Channel ID", "total_chunks"]].drop_duplicates()

    # --- Final merge (left joins keep channels with no topics) ---
    print("🔗 Merging everything into one dataframe...")
    final_df = (
        channel_metadata
        .merge(total_chunks_df, on="Channel ID", how="left")
        .merge(agg_topics, on="Channel ID", how="left")
    )

    # --- Reorder & rename total_chunks -> number_of_chunks ---
    select_cols = (
        ["Channel ID"]
        + metadata_columns
        + ["total_chunks", "topics", "topic_names", "representations", "topic_percentages"]
        + (["Category"] if category_map_path else [])
    )
    final_df = final_df[select_cols].rename(columns={"total_chunks": "number_of_chunks"})

    print(f"💾 Saving consolidated data to {out_path}")
    final_df.to_csv(out_path, index=False)
    print("✅ Done!")
