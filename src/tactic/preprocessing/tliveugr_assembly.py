"""TLiveUGR dataset assembly: English-only filter + channel-grouped rotating CSVs.

Non-English rows are dropped (the TLiveUGR lane is not translated).
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# -------------------------
# Schema
# -------------------------
REQUIRED_COLS = {"Channel ID", "Message ID", "Username", "message", "language", "lang_confidence"}
OUT_COLUMNS = ["Channel ID", "Message ID", "Username", "message", "language", "lang_confidence"]


# -------------------------
# Utilities
# -------------------------
def list_input_csvs(in_path: Path) -> List[Path]:
    """Return a sorted list of input CSV files from a file or directory."""
    if in_path.is_file():
        return [in_path]
    return sorted([p for p in in_path.glob("*.csv") if p.is_file()])


def _read_csv_chunks(path: Path, chunk_size: int) -> Iterable[pd.DataFrame]:
    """Read a CSV yielding chunks with only the required columns."""
    return pd.read_csv(
        path,
        chunksize=chunk_size,
        usecols=lambda c: c in REQUIRED_COLS,
        dtype={
            "Channel ID": "string",
            "Message ID": "string",
            "Username": "string",
            "message": "string",
            "language": "string",
            # lang_confidence: let pandas infer
        },
        encoding="utf-8",
    )


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns exist and return them in the correct order."""
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = "" if col != "lang_confidence" else np.nan
    return df[OUT_COLUMNS]


def _normalize_lang_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.split("-").str[0]


def _count_en_rows_by_channel(path: Path, chunk_size: int) -> Counter:
    """Fast pass: count only English rows per channel in a CSV."""
    counts: Counter = Counter()
    for chunk in pd.read_csv(
        path,
        chunksize=chunk_size,
        usecols=["Channel ID", "language"],
        dtype={"Channel ID": "string", "language": "string"},
        encoding="utf-8",
    ):
        chunk["Channel ID"] = chunk["Channel ID"].fillna("").astype("string")
        langs = _normalize_lang_series(chunk["language"])
        mask_en = langs.eq("en")
        if mask_en.any():
            sub = chunk.loc[mask_en, "Channel ID"].fillna("").astype("string")
            counts.update(sub)
    if "" in counts:
        del counts[""]
    return counts


def _spill_to_temp_by_channel_en_only(path: Path, temp_dir: Path, chunk_size: int) -> Counter:
    """
    Split a (possibly multi-channel) CSV into per-channel temp CSVs containing ONLY English rows.
    Return counts of EN rows per channel.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()

    for chunk in _read_csv_chunks(path, chunk_size=chunk_size):
        chunk = _ensure_cols(chunk)
        chunk["Channel ID"] = chunk["Channel ID"].fillna("").astype("string")
        langs = _normalize_lang_series(chunk["language"])
        chunk = chunk.loc[langs.eq("en")]
        if chunk.empty:
            continue

        for ch_id, sub in chunk.groupby("Channel ID"):
            ch = str(ch_id)
            if not ch:
                continue
            counts[ch] += len(sub)
            temp_path = temp_dir / f"channel_{ch}.tmp.csv"
            sub.to_csv(
                temp_path,
                mode="a",
                header=not temp_path.exists(),
                index=False,
                encoding="utf-8",
            )
    return counts


# -------------------------
# Rotating writer (no channel splitting)
# -------------------------
class RotatingCSVWriter:
    def __init__(self, out_dir: Path, prefix: str, max_rows_per_file: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.max_rows = max_rows_per_file
        self.part_idx = 0
        self.rows_in_current = 0
        self.path_current: Path | None = None

    def _next_path(self) -> Path:
        self.part_idx += 1
        return self.out_dir / f"{self.prefix}_{self.part_idx:04d}.csv"

    def _open_new(self):
        self.path_current = self._next_path()
        # ensure a clean file
        if self.path_current.exists():
            self.path_current.unlink()
        self.rows_in_current = 0

    def ensure_capacity_for_channel(self, channel_rows: int):
        """
        Rotate if adding an entire channel would exceed capacity.
        If current file is empty, allow overflow (single huge channel may exceed the limit).
        """
        if self.path_current is None:
            self._open_new()
            return
        if self.rows_in_current == 0:
            return
        if self.rows_in_current + channel_rows > self.max_rows:
            self._open_new()

    def write_df(self, df: pd.DataFrame):
        if df.empty:
            return
        if self.path_current is None:
            self._open_new()
        # enforce output schema + order
        df = _ensure_cols(df)
        df.to_csv(
            self.path_current,
            mode="a",
            header=not self.path_current.exists(),
            index=False,
            encoding="utf-8",
        )
        self.rows_in_current += len(df)


# -------------------------
# Per-file processing
# -------------------------
def process_single_input_csv(
    in_csv: Path,
    writer: RotatingCSVWriter,
    metrics: Dict,
    chunk_size: int,
    tmp_root: Path,
):
    """Process one input CSV: filter to EN and rotate by channel boundaries.

    `channels_without_en` is only tracked on the single-channel path. In the
    multi-channel path, channels with no EN rows have no temp file and are
    skipped.
    """
    # Count EN rows per channel
    counts = _count_en_rows_by_channel(in_csv, chunk_size=chunk_size)
    if not counts:
        metrics["files_processed"] += 1
        return  # nothing to write from this file

    # Single-channel fast path (still streaming)
    if len(counts) == 1:
        (channel_id, en_rows), = counts.items()
        writer.ensure_capacity_for_channel(en_rows)
        written_any = False
        for chunk in _read_csv_chunks(in_csv, chunk_size=chunk_size):
            metrics["total_rows_seen"] += len(chunk)
            langs = _normalize_lang_series(chunk["language"])
            en_chunk = _ensure_cols(chunk.loc[langs.eq("en")])
            metrics["rows_kept_en"] += len(en_chunk)
            metrics["rows_dropped_non_en"] += (len(chunk) - len(en_chunk))
            writer.write_df(en_chunk)
            written_any |= not en_chunk.empty
        if written_any:
            metrics["channels_written"] += 1
        else:
            metrics["channels_without_en"] += 1
        metrics["files_processed"] += 1
        return

    # Multi-channel: spill EN-only rows to per-channel temp
    temp_dir = tmp_root / f".tmp_{in_csv.stem}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # While spilling, accumulate metrics seen/dropped
        for chunk in _read_csv_chunks(in_csv, chunk_size=chunk_size):
            metrics["total_rows_seen"] += len(chunk)
            langs = _normalize_lang_series(chunk["language"])
            en_mask = langs.eq("en")
            metrics["rows_kept_en"] += int(en_mask.sum())
            metrics["rows_dropped_non_en"] += int((~en_mask).sum())

        # Now actually split EN-only rows to temp files (and count en rows per channel)
        counts = _spill_to_temp_by_channel_en_only(in_csv, temp_dir, chunk_size=chunk_size)

        for channel_id, en_rows in sorted(counts.items(), key=lambda kv: kv[0]):
            writer.ensure_capacity_for_channel(en_rows)
            ch_path = temp_dir / f"channel_{channel_id}.tmp.csv"
            if not ch_path.exists():
                continue
            wrote = False
            for chunk in pd.read_csv(
                ch_path,
                chunksize=chunk_size,
                dtype={
                    "Channel ID": "string",
                    "Message ID": "string",
                    "Username": "string",
                    "message": "string",
                    "language": "string",
                },
                encoding="utf-8",
            ):
                writer.write_df(chunk)
                wrote = wrote or not chunk.empty
            if wrote:
                metrics["channels_written"] += 1
            else:
                metrics["channels_without_en"] += 1

        metrics["files_processed"] += 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# -------------------------
# Orchestration
# -------------------------
def filter_and_group(
    in_path: Path,
    out_dir: Path,
    out_prefix: str,
    metrics_json: Path,
    chunk_size: int,
    max_rows_per_file: int,
):
    """Filter every input CSV to English rows and write channel-grouped rotating shards.

    A failure while processing one input file is reported as a warning and the
    file is skipped. Processing continues with the remaining inputs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = out_dir / ".tmp_channels"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    writer = RotatingCSVWriter(out_dir=out_dir, prefix=out_prefix, max_rows_per_file=max_rows_per_file)

    # Metrics
    m = defaultdict(int)
    m["files_processed"] = 0
    m["channels_written"] = 0
    m["channels_without_en"] = 0
    m["total_rows_seen"] = 0
    m["rows_kept_en"] = 0
    m["rows_dropped_non_en"] = 0

    input_files = list_input_csvs(in_path)
    if not input_files:
        raise SystemExit(f"No CSVs found under: {in_path}")

    for f in tqdm(input_files, desc="Filtering & grouping CSVs"):
        try:
            process_single_input_csv(
                in_csv=f,
                writer=writer,
                metrics=m,
                chunk_size=chunk_size,
                tmp_root=tmp_root,
            )
        except Exception as e:
            print(f"[warn] Skipping {f.name} due to error: {e}")

    # Metrics → JSON
    metrics = {
        "files_processed": int(m["files_processed"]),
        "channels_written": int(m["channels_written"]),
        "channels_without_en": int(m["channels_without_en"]),
        "total_rows_seen": int(m["total_rows_seen"]),
        "rows_kept_en": int(m["rows_kept_en"]),
        "rows_dropped_non_en": int(m["rows_dropped_non_en"]),
        "output_directory": str(out_dir),
        "output_prefix": out_prefix,
        "max_rows_per_file": int(max_rows_per_file),
    }
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"✅ Done.\nGrouped CSVs → {out_dir}\nMetrics → {metrics_json}")


def validate_single_csv_header(in_path: Path) -> None:
    """Header sanity check when the input path is a single CSV file."""
    if in_path.is_file():
        hdr = pd.read_csv(in_path, nrows=0, encoding="utf-8").columns.tolist()
        missing = sorted(list(REQUIRED_COLS - set(hdr)))
        if missing:
            raise SystemExit(f"Input CSV is missing required columns: {missing}")
