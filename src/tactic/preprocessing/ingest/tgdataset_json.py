"""TGDataset ingestion: stream raw JSON dumps into per-file message CSVs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import ijson                   # streaming JSON parser
import pandas as pd
from tqdm import tqdm


def count_total_messages(json_path: Path) -> int:
    """Count all message records across channels in a TGDataset JSON file.

    Streams the file using `ijson.kvitems` so the whole JSON is never loaded
    into memory. For each top-level channel object, this counts the number of
    entries in its `text_messages` mapping. Non-dict, missing, or null
    `text_messages` are treated as zero (ignored).

    Expected JSON shape (top level is a dict):
        {
            "<channel_id>": {
                "username": "...",
                "text_messages": { "<msg_id>": {"message": "..."} , ... }
            },
            ...
        }
    """
    total_messages = 0

    # Stream top-level (channel_id, channel_data) pairs to avoid loading the entire file.
    with json_path.open("r", encoding="utf-8") as f:
        for _, channel_data in ijson.kvitems(f, ""):
            # Be defensive: only count when text_messages is a dict
            text_messages_raw = channel_data.get("text_messages")
            if isinstance(text_messages_raw, dict):
                total_messages += len(text_messages_raw)

    return total_messages


def extract_json(
    json_path: Path,
    out_csv: Path,
    global_metrics: Dict[str, Any],
    channel_stats: Dict[str, Dict[str, Any]],
    flush_size: int = 100_000,
) -> None:
    """Stream a TGDataset JSON file and write non-empty messages to a CSV.

    For each channel:
      - Reads `text_messages` (must be a dict), skips if missing/non-dict.
      - Emits rows with columns: ["Channel ID", "Message ID", "Username", "message"].
      - Accumulates global and per-channel metrics (pre-cleaning).

    Notes:
        - Only messages with a non-empty `message` field (after `strip()`) are written.
        - `initial_empty_or_nan_messages` counts messages that were present but empty/None.
        - `unique_channels_after_initial_filter` counts channels that had >= 1 written row.
        - Channel-level counters are accumulated per file: TGDataset holds each
          channel in exactly one file, and the counts stage checks for cross-file
          collisions.
    """
    if flush_size <= 0:
        flush_size = 100_000  # sane fallback

    batch_rows: List[Dict[str, str]] = []
    channels_with_rows: set = set()
    total_messages = count_total_messages(json_path)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():  # avoid duplicates on re-runs
        out_csv.unlink()

    stage_start = time.time()

    with json_path.open("r", encoding="utf-8") as f, tqdm(
        total=total_messages,
        desc=f"Processing {json_path.name}",
        dynamic_ncols=True,
        mininterval=5,
        leave=False
    ) as pbar:
        for channel_id, channel_data in ijson.kvitems(f, ""):
            channel_id_str = str(channel_id)
            username = channel_data.get("username", "")

            # Initialize/augment per-channel stats
            ch = channel_stats.setdefault(channel_id_str, {
                "username": "",
                "source_json": "",
                "total_original_messages": 0,
                "initial_empty_or_nan_messages": 0,
            })

            if not ch["username"] and username is not None:
                ch["username"] = str(username)
            if not ch["source_json"]:
                ch["source_json"] = json_path.name

            text_messages_raw = channel_data.get("text_messages")
            if not text_messages_raw:
                # missing/None/empty
                global_metrics["initial_channels_without_text_messages"] += 1
                text_messages = {}
            elif not isinstance(text_messages_raw, dict):
                # malformed (e.g., list/str)
                global_metrics["initial_channels_with_invalid_text_messages"] += 1
                text_messages = {}
            else:
                text_messages = text_messages_raw

            # Global & per-channel counts (pre-cleaning)
            msg_count = len(text_messages)
            global_metrics["total_original_messages"] += msg_count
            global_metrics["initial_unique_channels"] += 1
            ch["total_original_messages"] += msg_count # Accumulates in case the channel appeared in more than 1 file

            # Emit rows for non-empty messages
            for msg_id, msg_data in text_messages.items():
                raw_msg = msg_data.get("message", None)
                if raw_msg is not None and str(raw_msg).strip():
                    batch_rows.append({
                        "Channel ID": channel_id_str,
                        "Message ID": str(msg_id),
                        "Username": username if username is not None else "",
                        "message": str(raw_msg).strip(),
                    })
                    channels_with_rows.add(channel_id_str)
                else:
                    global_metrics["initial_empty_or_nan_messages"] += 1
                    ch["initial_empty_or_nan_messages"] += 1

                pbar.update(1) # Update tqdm for each message

                # Flush in batches
                if len(batch_rows) >= flush_size:
                    pd.DataFrame(batch_rows).to_csv(
                        out_csv,
                        mode="a",
                        index=False,
                        header=not out_csv.exists(),
                        encoding="utf-8",
                    )
                    batch_rows.clear()

    # Final flush
    if batch_rows:
        pd.DataFrame(batch_rows).to_csv(
            out_csv,
            mode="a",
            index=False,
            header=not out_csv.exists(),
            encoding="utf-8",
        )

    global_metrics["unique_channels_after_initial_filter"] += len(channels_with_rows)
    global_metrics["initial_deletion_runtime_sec"] += round(time.time() - stage_start, 2)
