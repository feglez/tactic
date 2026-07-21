"""Channel-level side tables: per-channel message counts and channel titles."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import ijson
import pandas as pd
from tqdm import tqdm


# ===========================================================================
# Channel message counts
# ===========================================================================
def build_channel_counts_with_sources(
    csv_dir: Path,
    out_csv: Path,
    chunk_size: int = 100_000,
) -> None:
    """Aggregate per-channel message counts and record each channel's source CSV.

    Streams all CSVs in ``csv_dir`` reading only the ``"Channel ID"`` column, counts
    messages per channel, and writes a single CSV with columns:
      - ``ch_ID`` (str): channel identifier
      - ``message_count`` (int): total messages observed for that channel
      - ``source_csv`` (str): the first CSV filename where the channel appeared

    Rows are written in first-seen scan order (unsorted). Downstream sampling
    shuffles this file with a fixed seed, so the row order matters.

    Cross-file collisions (a channel appearing in more than one CSV) are printed
    for visibility (first 10) and not persisted. ``source_csv`` keeps the first
    file seen.
    """
    if not csv_dir.exists() or not csv_dir.is_dir():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    csv_files = sorted(p for p in csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")

    # Used for logging at the end
    total_rows = 0

    # Aggregators
    counts: Dict[str, int] = defaultdict(int)              # ch_ID -> total messages
    first_file_for_channel: Dict[str, str] = {}            # ch_ID -> first-seen csv filename
    collisions: Dict[str, Set[str]] = defaultdict(set)     # ch_ID -> {filenames}

    # Iterate files with a progress bar
    for file_path in tqdm(csv_files, desc="Scanning CSVs", unit="file"):
        # Stream only the needed column and enforce string dtype for stable keys
        reader = pd.read_csv(
            file_path,
            usecols=["Channel ID"],
            chunksize=chunk_size,
            dtype={"Channel ID": "string"},
            encoding="utf-8",
        )

        for chunk in reader:
            col = chunk["Channel ID"].dropna()
            total_rows += len(chunk)

            # 1) Vectorized per-chunk counts
            vc = col.value_counts()
            for ch_id, n in vc.items():
                ch = str(ch_id)
                counts[ch] += int(n)

            # 2) Record first file and detect collisions
            for ch in col.unique():
                ch = str(ch)
                prev = first_file_for_channel.get(ch)
                if prev is None:
                    first_file_for_channel[ch] = file_path.name
                elif prev != file_path.name:
                    # Channel appears in more than one CSV, so record the collision
                    collisions[ch].update({prev, file_path.name})

    # Materialize counts CSV (include source file)
    rows = [
        {"ch_ID": ch, "message_count": int(cnt), "source_csv": first_file_for_channel.get(ch, "")}
        for ch, cnt in counts.items()
    ]

    out_df = pd.DataFrame(rows).astype(
        {"ch_ID": "string", "message_count": "int64", "source_csv": "string"}
    )

    out_df.to_csv(out_csv, index=False)
    print(f"✅ Saved {out_csv} with {len(out_df)} channels (scanned ~{total_rows:,} rows).")

    # Report collisions for visibility (no separate JSON output)
    if collisions:
        print(f"⚠️ Detected {len(collisions)} channels present in multiple CSV files.")
        # Show a small sample to avoid spamming the console
        for ch, files in list(collisions.items())[:10]:
            print(f"   - {ch}: {sorted(files)}")
        print("   (The 'source_csv' column shows the first file encountered.)")
    else:
        print("ℹ️ No channel/file collisions detected.")


# ===========================================================================
# Channel titles
# ===========================================================================
def _coerce_str(x) -> str:
    """Return x as a trimmed string. None/falsey → ''."""
    try:
        s = "" if x is None else str(x)
    except Exception:
        return ""
    return s.strip()


def extract_channel_metadata(json_path: Path, out_map: Dict[str, Dict[str, str]]) -> None:
    """Stream one JSON file and merge (channel_id, username, title) into out_map.

    For each channel_id:
      - keep the first non-empty username/title seen,
      - never overwrite a non-empty value with an empty one.
    """
    with json_path.open("r", encoding="utf-8") as f:
        # Iterate top-level items: channel_id → dict
        for channel_id, channel_data in ijson.kvitems(f, ""):
            ch_id = _coerce_str(channel_id)
            if not ch_id:
                continue

            if not isinstance(channel_data, dict):
                # Skip malformed entries
                continue

            username = _coerce_str(channel_data.get("username", ""))
            title = _coerce_str(channel_data.get("title", ""))

            entry = out_map.setdefault(ch_id, {"username": "", "title": ""})

            # Prefer the first non-empty value observed
            if not entry["username"] and username:
                entry["username"] = username
            if not entry["title"] and title:
                entry["title"] = title


def run_title_parser(raw_dir: Path, out_csv: Path) -> None:
    """Recover channel titles from raw TGDataset JSONs.

    Output CSV columns: channel_id, username, title. Rows are sorted by
    channel_id lexicographically (IDs are strings).
    """
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"--raw-dir not found or not a directory: {raw_dir}")

    json_files = sorted([p for p in raw_dir.glob("*.json") if p.is_file()])
    print(f"🔍 Found {len(json_files)} JSON files in {raw_dir}")

    channels: Dict[str, Dict[str, str]] = {}
    errors = 0

    for jp in tqdm(json_files, desc="Processing JSON files", dynamic_ncols=True, mininterval=1.0):
        try:
            extract_channel_metadata(jp, channels)
        except Exception as e:
            errors += 1
            tqdm.write(f"[❌ ERROR] Failed to process {jp.name}: {e}")

    # Save CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"channel_id": ch_id, "username": data["username"], "title": data["title"]}
        for ch_id, data in channels.items()
    ]
    # Sort by channel_id for stable output
    rows.sort(key=lambda r: r["channel_id"])

    print(f"💾 Saving {len(rows)} rows to {out_csv} (errors on {errors} files).")
    with out_csv.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["channel_id", "username", "title"])
        writer.writeheader()
        writer.writerows(rows)

    print("✅ Done.")
