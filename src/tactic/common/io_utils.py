"""Chunked-CSV IO helpers shared across pipeline stages."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def count_chunks(csv_path: Path, chunk_size: int, encoding: str = "utf-8") -> int:
    """Count how many pandas chunks `read_csv(..., chunksize=chunk_size)` will yield.

    This uses pandas' actual CSV parser instead of line counting, so it's robust to
    quoted newlines and other CSV quirks.

    Args:
        csv_path: Path to the CSV to be streamed.
        chunk_size: Number of rows per chunk. Must be > 0.
        encoding: File encoding used by pandas when reading.

    Returns:
        The number of chunks (0 if the file is empty).

    Raises:
        ValueError: If `chunk_size <= 0`.
        FileNotFoundError: If the CSV file does not exist.
        UnicodeDecodeError: If the file can't be decoded with `encoding`.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0 (got {chunk_size})")

    try:
        it = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            usecols=[0],        # read only first column, enough to drive chunking
            encoding=encoding,
        )
        return sum(1 for _ in it)
    except pd.errors.EmptyDataError:
        # Empty file (no rows, maybe just a header-less empty file)
        return 0


def append_df_to_csv(df: pd.DataFrame, path: Path, **to_csv_kwargs) -> None:
    """Append a DataFrame to a CSV, writing the header only when the file is first created."""
    df.to_csv(path, mode="a", header=not path.exists(), index=False, **to_csv_kwargs)


def atomic_replace(tmp_path: Path, final_path: Path) -> None:
    """Atomically move a fully-written temp file over its final destination.

    Used by the write-to-`<name>.tmp` + `os.replace` pattern with which each
    stage rewrites its output CSV in place.
    """
    os.replace(tmp_path, final_path)
