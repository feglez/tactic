"""Stratified sampling of messages per topic for translation (index-based).

Selections are stored as CHANNEL-LOCAL POSITIONAL indices (0..message_count-1)
over the cleaned CSVs, not Message IDs. A re-run of preprocessing that changes
row order or counts therefore invalidates previously written selection files.
"""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_used_indices_map(history_dir: Path) -> Dict[str, Set[int]]:
    """Load previously selected (ch_ID, message_index) pairs from older runs.

    Scans files matching ``selected_message_indices_*.csv`` in ``history_dir``
    and returns a mapping from channel ID to the set of message indices that
    have already been used for that channel. The tuning sample is drawn
    independently while the validation and final samples exclude one another,
    so ``*_tuning.csv`` ledgers are skipped.
    """
    used_indices_map: Dict[str, Set[int]] = {}
    for file_path in sorted(glob.glob(str(history_dir / "selected_message_indices_*.csv"))):
        if Path(file_path).name.endswith("_tuning.csv"):
            continue
        df_prev = pd.read_csv(file_path)
        # Defensive: ensure required columns exist
        if {"ch_ID", "message_index"}.issubset(df_prev.columns):
            for ch, idx in zip(df_prev["ch_ID"].astype(str), df_prev["message_index"].astype(int)):
                used_indices_map.setdefault(ch, set()).add(idx)
    return used_indices_map


def relaxed_sampling(
    group_df: pd.DataFrame,
    target_msgs: int,
    used_indices_map: Dict[str, Set[int]],
    max_per_channel: int,
    random_seed: int,
) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
    """Two-phase sampling inside a topic split (English or Non-English).

    Phase 1: iterate by channel-size decile buckets (shuffled), selecting up to the
    per-channel cap and target remaining.

    Phase 2: if the target is not met, relax
    to sampling from the full group (again shuffled). Indices are drawn from
    [0, message_count) excluding indices seen in `used_indices_map`.

    RNG-CRITICAL: `random.sample` draws from the GLOBAL `random` module state
    (seeded once per run), and every `df.sample` uses the same fixed
    `random_state`. Iteration order therefore affects results, so do not reorder.
    """
    selection: List[Dict[str, int]] = []
    index_records: List[Dict[str, int]] = []
    selected_channels: Set[str] = set()  # track channels picked in Phase 1

    # Upper bound on available messages when respecting cap per channel
    available = int(group_df["message_count"].apply(lambda x: min(int(x), max_per_channel)).sum())
    target = min(int(target_msgs), available)

    # --- Phase 1: by decile (stratified by channel size) ---
    for decile in sorted(group_df["decile"].unique()):
        df_decile = group_df[group_df["decile"] == decile].sample(frac=1, random_state=random_seed)
        for _, row in df_decile.iterrows():
            if target <= 0:
                break

            ch_id = str(row["ch_ID"])
            total = int(row["message_count"])
            take_n = min(total, max_per_channel, target)

            # Compute available indices not previously used
            ch_previous_used_indices = used_indices_map.get(ch_id, set())

            # Note (possible future optimization): For very large channels, building
            # `set(range(total))` is memory-heavy. If this ever shows up as a bottleneck,
            # we will switch from materializing all candidate indices to a rejection-
            # sampling approach (draw with random.randrange(total) until k unseen indices
            # are collected), or a helper that chooses between explicit candidate listing
            # vs. rejection sampling based on a simple heuristic, e.g.:
            #   if total > 100_000 and len(ch_previous_used_indices) / total < 0.5:
            #       prefer rejection sampling
            # Otherwise the current approach is simpler and perfectly fine.
            candidates = list(set(range(total)) - ch_previous_used_indices)
            actual_n = min(len(candidates), take_n)
            if actual_n == 0:
                continue

            sampled_indices = random.sample(candidates, actual_n)
            for idx in sampled_indices:
                used_indices_map.setdefault(ch_id, set()).add(idx)
                index_records.append({"ch_ID": ch_id, "message_index": idx})

            selection.append({"ch_ID": ch_id, "num_messages": actual_n})
            selected_channels.add(ch_id)
            target -= actual_n

        if target <= 0:
            break

    # --- Phase 2: relaxed sampling across the entire group ---
    if target > 0:
        for _, row in group_df.sample(frac=1, random_state=random_seed).iterrows():
            if target <= 0:
                break

            # do not re-add a channel already added in phase 1 to `selection`
            ch_id = str(row["ch_ID"])
            if ch_id in selected_channels:
                continue

            total = int(row["message_count"])
            take_n = min(total, max_per_channel, target)

            ch_previous_used_indices = used_indices_map.get(ch_id, set())
            candidates = list(set(range(total)) - ch_previous_used_indices)
            actual_n = min(len(candidates), take_n)
            if actual_n == 0:
                continue

            sampled_indices = random.sample(candidates, actual_n)
            for idx in sampled_indices:
                used_indices_map.setdefault(ch_id, set()).add(idx)
                index_records.append({"ch_ID": ch_id, "message_index": idx})

            selection.append({"ch_ID": ch_id, "num_messages": actual_n})
            target -= actual_n

    return selection, index_records


def run_sampler(cfg_yaml: dict) -> None:
    """Run the stratified sampler.

    Config keys (see configs/preprocessing/sampling.yaml):
      paths.topic_csv / paths.msg_counts_csv / paths.language_tsv: the three inputs
      paths.history_dir: where prior selected_message_indices_*.csv live
      paths.out_dir: where the two outputs are written
      params.dataset_name / total_messages / max_per_channel / english_fraction / random_seed
    """
    paths = cfg_yaml.get("paths", {})
    params = cfg_yaml.get("params", {})

    TOTAL_MESSAGES = int(params.get("total_messages", 2_000_000))
    MAX_PER_CHANNEL = int(params.get("max_per_channel", 2_000))
    ENGLISH_FRACTION = float(params.get("english_fraction", 0.8))
    RANDOM_SEED = int(params.get("random_seed", 45))
    dataset_name = params["dataset_name"]

    TOPIC_PATH = paths["topic_csv"]
    MSG_COUNTS_PATH = paths["msg_counts_csv"]
    LANGUAGE_PATH = paths["language_tsv"]
    history_dir = Path(paths["history_dir"])
    out_dir = Path(paths["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Reproducibility: seed both generators before any RNG use.
    # -------------------------
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Derive output filenames from dataset name
    output_selection_path = str(out_dir / f"selected_channel_messages_{dataset_name}.csv")
    output_message_indices_path = str(out_dir / f"selected_message_indices_{dataset_name}.csv")

    # Remove this dataset's stale indices file before loading history (so it doesn't count as prior)
    try:
        os.remove(output_message_indices_path)
        print(f"🧹 Removed stale {output_message_indices_path} before loading history.")
    except FileNotFoundError:
        pass

    # --- Load inputs ---
    df_topic = pd.read_csv(  # ch_ID, topic
        TOPIC_PATH,
        usecols=["ch_ID", "topic"],
        dtype={"ch_ID": "string", "topic": "string"},
    )
    df_topic.columns = [c.strip() for c in df_topic.columns]
    df_topic = df_topic.dropna(subset=["ch_ID", "topic"])
    df_topic = df_topic[
        (df_topic["ch_ID"].str.strip() != "") &
        (df_topic["topic"].str.strip() != "")
    ]

    df_msg_counts = pd.read_csv(
        MSG_COUNTS_PATH,
        usecols=["ch_ID", "message_count"],
        dtype={"ch_ID": "string", "message_count": "int64"},
    )

    # language file is a TSV, even if named *.csv
    df_lang = pd.read_csv(  # ch_id, language
        LANGUAGE_PATH,
        usecols=["ch_id", "language"],
        dtype={"ch_id": "string", "language": "string"},
        sep="\t",
    )
    df_lang.columns = [c.strip() for c in df_lang.columns]
    df_lang = df_lang.rename(columns={"ch_id": "ch_ID"})
    df_lang = df_lang.dropna(subset=["ch_ID", "language"])
    df_lang = df_lang[
        (df_lang["ch_ID"].str.strip() != "") &
        (df_lang["language"].str.strip() != "")
    ]

    # --- Load previously used indices (from prior runs) ---
    used_indices_map = load_used_indices_map(history_dir)
    total_pairs = sum(len(s) for s in used_indices_map.values())
    print(f"🔒 Loaded {total_pairs:,} previously used (ch_ID, message_index) pairs from disk.")

    # --- Merge (inner joins: channel must exist in all three) ---
    df = (
        df_topic.merge(df_msg_counts[["ch_ID", "message_count"]], on="ch_ID", how="inner")
                .merge(df_lang[["ch_ID", "language"]], on="ch_ID", how="inner")
    )

    # Keep only the expected columns in a stable order
    df = df[["ch_ID", "topic", "message_count", "language"]]

    # --- Compute deciles by message_count (10 buckets: 0..9) ---
    # duplicates="drop" guards against non-unique bin edges when many channels
    # share identical message counts (fewer than 10 strata may then result).
    df["decile"] = pd.qcut(df["message_count"], 10, labels=False, duplicates="drop")

    # The language split below uses the CHANNEL-LEVEL language label
    # (channel_to_language_mapping.csv): a channel labeled "en" is treated as
    # English in full.

    # --- Per-topic quotas ---
    # Topic iteration order is RNG-CRITICAL: the global `random` state is
    # consumed per topic, so a different order changes every downstream draw.
    # Topics are iterated in the merged frame's first-occurrence order.
    topics = df["topic"].unique().tolist()
    if not topics:
        raise ValueError("No topics found after merges, check your inputs.")

    # Allocates messages_per_topic * len(topics) messages in total; any remainder from
    # the integer division is dropped, which is why the output can hold slightly fewer
    # than TOTAL_MESSAGES rows.
    messages_per_topic = TOTAL_MESSAGES // len(topics)

    selected_rows: List[Dict[str, int]] = []          # per-channel selection counts
    message_index_records: List[Dict[str, int]] = []  # (ch_ID, message_index) pairs

    # Iterate over each topic bucket independently so every topic contributes
    # approximately the same number of messages (messages_per_topic).
    for topic in tqdm(topics, desc="Processing topics"):
        # Narrow to just current topic
        df_topic_group = df[df["topic"] == topic]

        # Split by channel language
        df_eng = df_topic_group[df_topic_group["language"] == "en"]
        df_non_eng = df_topic_group[df_topic_group["language"] != "en"]

        # Compute message budgets for this topic
        eng_needed = int(messages_per_topic * ENGLISH_FRACTION)
        non_eng_needed = messages_per_topic - eng_needed  # to avoid rounding gaps

        # 1) Non-English first
        sel_non_eng, idx_non_eng = relaxed_sampling(
            df_non_eng, non_eng_needed, used_indices_map, MAX_PER_CHANNEL, RANDOM_SEED)
        remaining_non_eng = non_eng_needed - sum(x["num_messages"] for x in sel_non_eng)

        # 2) English + any leftover from non-English. The rollover is
        # one-directional: a non-English shortfall boosts the English quota,
        # while an English shortfall is not rolled back. Together with the
        # dropped division remainder, the final total can be below TOTAL_MESSAGES.
        sel_eng, idx_eng = relaxed_sampling(
            df_eng, eng_needed + max(0, remaining_non_eng), used_indices_map, MAX_PER_CHANNEL, RANDOM_SEED)

        selected_rows.extend(sel_eng)
        selected_rows.extend(sel_non_eng)

        message_index_records.extend(idx_eng)
        message_index_records.extend(idx_non_eng)

    # --- Save outputs (dataset-named) ---
    pd.DataFrame(selected_rows).to_csv(output_selection_path, index=False)
    pd.DataFrame(message_index_records).to_csv(output_message_indices_path, index=False)

    total_selected = sum(x["num_messages"] for x in selected_rows)
    print(f"✅ Selection complete. Saved {total_selected:,} messages to {output_selection_path}")
    print(f"🧾 Message-level indices saved to {output_message_indices_path}")
