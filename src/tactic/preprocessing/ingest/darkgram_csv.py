"""DarkGram ingestion: consolidate raw per-category CSVs into one sorted CSV."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm


def consolidate_raw_csvs(
    csv_files: List[Path],
    out_csv: Path,
    metrics: Dict[str, Any],
    per_channel_metrics: Dict[str, Dict[str, Any]],
    chunk_size: int,
) -> None:
    """Read/stream Darkgram CSVs and write a single consolidated CSV.

    Required columns in each raw CSV:
      - "Channel ID"   (username/handle, may be missing on some rows)
      - "peer_channel" (numeric channel id, canonical)
      - "Post ID"      (message id)
      - "message"      (text content)

    Behavior:
      - Rows with empty/NaN `message` are removed (counted in metrics).
      - Final IDs are kept exactly as present, only cast to string:
          * output "Channel ID"  = str(peer_channel)
          * output "Message ID"  = str(Post ID)
          * output "Username"    = str(input "Channel ID")
      - Adds `category` from the source filename stem (basename without `.csv`).
      - Appends to a single CSV with columns:
          ["Channel ID", "Message ID", "Username", "message", "category"].
    """
    # Prepare destination (overwrite if present)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()

    # Track unique peer_channel ids pre/post initial message-emptiness filter
    unique_peers_before_msg_filter: set = set()
    unique_peers_after_msg_filter: set = set()

    stage_start = time.time()
    required_cols = ["Channel ID", "peer_channel", "Post ID", "message"]

    for src in tqdm(csv_files, desc="Consolidating raw CSVs", dynamic_ncols=True):
        # Verify required columns exist in this file
        try:
            header_cols = pd.read_csv(src, nrows=0, encoding="utf-8").columns.tolist()
        except Exception as e:
            tqdm.write(f"[warn] Skipping {src.name} (failed to read header): {e}")
            continue

        missing_cols = [c for c in required_cols if c not in header_cols]
        if missing_cols:
            tqdm.write(f"[warn] Skipping {src.name}: missing columns {missing_cols}")
            continue

        category = src.stem

        reader = pd.read_csv(
            src,
            usecols=required_cols,
            chunksize=chunk_size,
            encoding="utf-8",
            dtype={
                "Channel ID": "string",
                "peer_channel": "string",
                "Post ID": "string",
                "message": "string",
            },
        )

        for chunk in reader:
            # Volume before any filtering
            raw_rows_in_chunk = len(chunk)
            metrics["total_original_messages"] += raw_rows_in_chunk

            if chunk.empty:
                continue

            # Unique peers before message-emptiness filter (for metrics)
            peers_current_chunk = chunk["peer_channel"].astype(str).unique().tolist()
            unique_peers_before_msg_filter.update(peers_current_chunk)

            # Per-channel pre-clean counts (messages present regardless of message emptiness)
            messages_per_peer_initial = (
                chunk.groupby(chunk["peer_channel"].astype(str)).size()
            )
            for peer_id, cnt in messages_per_peer_initial.items():
                peer_id = str(peer_id)
                stats = per_channel_metrics.setdefault(
                    peer_id,
                    {
                        "username": "",
                        "source_csv": "",
                        "total_original_messages": 0,
                        "initial_empty_or_nan_messages": 0,
                    },
                )
                stats["total_original_messages"] += int(cnt)
                if not stats["source_csv"]:
                    stats["source_csv"] = src.name

            # First non-empty username observed per peer_channel in this chunk
            username_by_peer = (
                chunk.assign(
                    __peer=chunk["peer_channel"].astype(str),
                    __uname=chunk["Channel ID"].fillna("").astype(str).str.strip(),
                )
                .groupby("__peer")["__uname"]
                .apply(lambda s: next((v for v in s if v), ""))
            )

            for peer_id, uname in username_by_peer.items():
                peer_id = str(peer_id)
                stats = per_channel_metrics.setdefault(
                    peer_id,
                    {
                        "username": "",
                        "source_csv": "",
                        "total_original_messages": 0,
                        "initial_empty_or_nan_messages": 0,
                    },
                )
                if not stats["username"] and uname:
                    stats["username"] = uname
                if not stats["source_csv"]:
                    stats["source_csv"] = src.name

            # --- Remove empty/NaN messages (count them globally and per-channel) ---
            is_message_empty = chunk["message"].isna() | (
                chunk["message"].astype(str).str.strip() == ""
            )
            empty_msg_count = int(is_message_empty.sum())
            if empty_msg_count:
                metrics["initial_empty_or_nan_messages"] += empty_msg_count

                empty_msgs_per_peer = (
                    chunk.loc[is_message_empty]
                    .groupby(chunk.loc[is_message_empty, "peer_channel"].astype(str))
                    .size()
                )
                for peer_id, cnt in empty_msgs_per_peer.items():
                    peer_id = str(peer_id)
                    stats = per_channel_metrics.setdefault(
                        peer_id,
                        {
                            "username": "",
                            "source_csv": "",
                            "total_original_messages": 0,
                            "initial_empty_or_nan_messages": 0,
                        },
                    )
                    stats["initial_empty_or_nan_messages"] += int(cnt)

            # Apply the emptiness filter for the rows we will actually emit
            chunk = chunk[~is_message_empty]
            if chunk.empty:
                continue

            # Unique peers after initial (message emptiness) filter
            unique_peers_after_msg_filter.update(
                chunk["peer_channel"].astype(str).unique().tolist()
            )

            # --- Build output rows (IDs kept as-is, cast to string only) ---
            output_rows = pd.DataFrame(
                {
                    "Channel ID": chunk["peer_channel"].astype(str),
                    "Message ID": chunk["Post ID"].fillna("").astype(str),
                    "Username": chunk["Channel ID"].fillna("").astype(str),
                    "message": chunk["message"].astype(str),
                    "category": category,
                }
            )

            # Append to the consolidated CSV
            output_rows.to_csv(
                out_csv,
                mode="a",
                index=False,
                header=not out_csv.exists(),
                encoding="utf-8",
            )

    metrics["initial_deletion_runtime_sec"] += round(time.time() - stage_start, 2)
    metrics["initial_unique_channels"] += len(unique_peers_before_msg_filter)
    metrics["unique_channels_after_initial_filter"] += len(unique_peers_after_msg_filter)


def sort_consolidated_csv(csv_path: Path) -> None:
    """Sort the consolidated CSV so rows from the same Channel ID are contiguous.

    Sort order (both as strings, lexicographic):
      1) Channel ID ascending
      2) Message ID ascending

    The file is read into memory, sorted, and written back in place.
    Suitable for typical Darkgram sizes (tens/hundreds of thousands of rows).

    Message ID compares lexicographically as a string ("10" < "2"), and downstream
    micro-batching consumes rows in exactly this order.
    """

    if not csv_path.exists():
        return  # nothing to do

    # Read entire file (we need a global sort)
    df = pd.read_csv(
        csv_path,
        dtype={"Channel ID": "string", "Message ID": "string"},
        encoding="utf-8",
    )
    if df.empty:
        return

    # Ensure string dtype has no NA sentinels that could affect ordering
    df["Channel ID"] = df["Channel ID"].fillna("").astype("string")
    df["Message ID"] = df["Message ID"].fillna("").astype("string")

    df = df.sort_values(
        by=["Channel ID", "Message ID"],
        kind="mergesort"  # stable
    )

    df.to_csv(csv_path, index=False, encoding="utf-8")
