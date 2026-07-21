"""fastText language detection: per-row detection stage with global/per-channel stats."""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import fasttext                # language ID (lid.176.bin)
import numpy as np
import pandas as pd
from tqdm import tqdm

from tactic.common.io_utils import count_chunks
from tactic.common.metrics_export import confidence_bucket_key

# fastText input normalization (params.fasttext_input_norm, set per dataset config):
#   "replace_newline" → text.replace('\n', ' ').strip(): only newlines become
#                       spaces, other Unicode whitespace reaches the model
#   "split_join"      → " ".join(text.split()): ALL whitespace (incl. newlines)
#                       collapsed to single spaces
INPUT_NORM_SPLIT_JOIN = "split_join"
INPUT_NORM_REPLACE_NEWLINE = "replace_newline"


class LanguageDetector:
    """Thin wrapper around fastText lid.176 for per-message language detection.

    Loads the fastText model once and exposes a `detect_one` method that returns
    `(language_code, confidence)` for a single message. Errors are logged with
    Channel ID and Message ID for traceability.
    """

    def __init__(self, model_path: Path, lang_logger: logging.Logger,
                 input_norm: str = INPUT_NORM_SPLIT_JOIN) -> None:
        """Load the fastText language ID model and configure logging.

        Args:
            model_path: Path to `lid.176.bin`.
            lang_logger: Logger to receive detection errors (should be configured
                by the caller, since this class does not attach handlers).
            input_norm: `params.fasttext_input_norm`, controlling how the message
                text is normalized before prediction (see the module-level note).

        Raises:
            RuntimeError: If the fastText model cannot be loaded.
        """
        if input_norm not in (INPUT_NORM_SPLIT_JOIN, INPUT_NORM_REPLACE_NEWLINE):
            raise ValueError(f"Unknown fasttext_input_norm: {input_norm!r}")
        try:
            self.model = fasttext.load_model(str(model_path))
        except Exception as e:  # pragma: no cover (I/O / external dependency)
            raise RuntimeError(f"Failed to load fastText model at '{model_path}': {e}") from e

        self.logger = lang_logger
        self.input_norm = input_norm

    def detect_one(
        self,
        text: str,
        msg_id: str,
        channel_id: str,
        row_offset: int,
        filename: str,
    ) -> Tuple[str, float]:
        """Detect language for a single message.

        Normalizes the input (per `params.fasttext_input_norm`) and calls
        `fastText.predict(k=1)`. On error, logs a compact record containing
        channel/message IDs, the file-wide row offset, a short
        snippet, and the source filename.

        Returns:
            A tuple `(language_code, confidence)` where:
              - `language_code` is a fastText two-letter/ISO-ish code (e.g., "en"),
                or "unknown" if detection failed.
              - `confidence` is a float in [0, 1], or 0.0 on failure.
        """

        # Guard: empty/whitespace-only text → treat as unknown
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        if self.input_norm == INPUT_NORM_REPLACE_NEWLINE:
            # Only newlines become spaces. Other whitespace reaches the model.
            norm = text.replace("\n", " ").strip()
        else:
            norm = " ".join(text.split())  # collapse all whitespace (incl. newlines) to single spaces

        if not norm:
            return "unknown", 0.0

        try:
            # fastText returns (labels, probabilities), each as lists
            labels, probs = self.model.predict(norm, k=1)
            lang = labels[0].replace("__label__", "") if labels else "unknown"
            conf = float(probs[0]) if probs else 0.0
            return lang, conf
        except Exception as e:  # pragma: no cover (external dependency)
            snippet = norm[:60]
            # printf-style args is used to avoid formatting overhead
            # when the message isn't emitted
            self.logger.warning(
                "[Language Detection Error] Channel: %s | MsgID: %s | RowOffset: %s | "
                "Snippet: %s | File: %s | Error: %s",
                channel_id,
                msg_id,
                row_offset,
                snippet,
                filename,
                repr(e),
            )
            return "unknown", 0.0


def detect_languages(
    csv_path: Path,
    global_metrics: Dict[str, Any],
    det: LanguageDetector,
    cfg,
    channel_stats: Dict[str, Dict[str, Any]],
) -> None:
    """Append language code & confidence to a cleaned CSV, chunked, and rewrite in place.

    Pipeline per chunk:
      1) Ensure `message` is a string (no NaNs).
      2) Detect language via `LanguageDetector.detect_one` (timed).
      3) Materialize `language` and `lang_confidence` columns efficiently.
      4) Update global aggregations (language distribution + confidence stats/buckets).
      5) Update per-channel aggregations (language distribution, non-English, unknown).
      6) Append augmented rows to a temporary CSV.

    At the end, the temp file atomically replaces the original CSV.

    Args:
        csv_path: Path to the cleaned CSV (output of `clean_csv`).
        global_metrics: Global counters/timers updated in place.
        det: Initialized `LanguageDetector`.
        cfg: Config object with `chunk_size`.
        channel_stats: Per-channel counters updated in place.
    """

    # Temp destination (same folder)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".lang.tmp")
    if tmp_path.exists():
        tmp_path.unlink()  # remove stale temp file from previous runs

    # Global aggregators
    lang_counter: Counter = Counter()
    min_conf, max_conf = 1.0, 0.0
    sum_conf, cnt_conf = 0.0, 0
    conf_buckets: Counter = Counter()

    # Progress bar uses an actual chunk count
    num_chunks = count_chunks(csv_path, chunk_size=cfg.chunk_size)

    reader = pd.read_csv(
        csv_path,
        chunksize=cfg.chunk_size,
        encoding="utf-8",
        dtype={"Channel ID": "string", "Message ID": "string"},
    )

    for i, chunk in enumerate(tqdm(reader, total=num_chunks, desc=f"Detecting language: {csv_path.name}")):
        # --- Step 1: coerce to string ---
        # Ensure strings and no NaNs so detector never receives NaN/"nan".
        chunk["message"] = chunk["message"].fillna("").astype(str)

        # --- Step 2: detection (timed) ---
        start = time.time()
        tqdm.pandas(desc=f"Detecting lang in chunk {i}")
        pairs = chunk.apply(
            lambda row: det.detect_one(
                row["message"],
                msg_id=str(row["Message ID"]),
                channel_id=str(row["Channel ID"]),
                row_offset=row.name,
                filename=csv_path.name,
            ),
            axis=1
        )
        end = time.time()
        global_metrics["language_detection_runtime_sec"] += round(end - start, 2)

        # --- Step 3: materialize columns ---
        if len(pairs):
            langs, confs = zip(*pairs)
            chunk["language"] = list(langs)
            chunk["lang_confidence"] = list(confs)
        else:
            chunk["language"] = []
            chunk["lang_confidence"] = []

        # --- Step 4: global aggregations ---
        # Language distribution
        lang_counter.update(chunk["language"].value_counts().to_dict())

        # Confidence stats and buckets
        vals = chunk["lang_confidence"].to_numpy(dtype=float, copy=False)
        if vals.size:
            # min/max/sum/count
            min_conf = min(min_conf, float(np.min(vals)))
            max_conf = max(max_conf, float(np.max(vals)))
            sum_conf += float(np.sum(vals))
            cnt_conf += int(vals.size)

            # Histogram buckets
            for conf in vals:
                c = float(conf)
                key = confidence_bucket_key(c)
                conf_buckets[key] += 1

        # --- Step 5: per-channel aggregations ---
        # Language distribution per channel
        grp = chunk.groupby(["Channel ID", "language"], dropna=False).size()
        for (ch_id, lang), cnt in grp.items():
            ch = channel_stats.setdefault(str(ch_id), {})
            dist = ch.setdefault("language_distribution", {})
            dist[lang] = int(dist.get(lang, 0) + int(cnt))

        # Non-English & unknown per channel
        for ch_id, sub in chunk.groupby("Channel ID", dropna=False):
            ch = channel_stats.setdefault(str(ch_id), {})
            ne = int((sub["language"] != "en").sum())
            un = int((sub["language"] == "unknown").sum())
            ch["non_english_messages"] = int(ch.get("non_english_messages", 0) + ne)
            ch["unknown_language_messages"] = int(ch.get("unknown_language_messages", 0) + un)

        # --- Step 6: append augmented rows to temp file ---
        chunk.to_csv(
            tmp_path,
            mode="a",
            index=False,
            header=not tmp_path.exists(),
            encoding="utf-8",
        )

    # If no rows were written, keep schema by writing an empty CSV with new columns
    if not tmp_path.exists():
        base_cols = pd.read_csv(csv_path, nrows=0, encoding="utf-8").columns.tolist()
        cols = list(base_cols)
        if "language" not in cols:
            cols.append("language")
        if "lang_confidence" not in cols:
            cols.append("lang_confidence")
        pd.DataFrame(columns=cols).to_csv(tmp_path, index=False, encoding="utf-8")

    # Atomically replace the original file with the augmented version
    os.replace(tmp_path, csv_path)

    # --- Finalize global metrics ---
    non_en = int(sum(lang_counter.values()) - lang_counter.get("en", 0))
    global_metrics["non_english_messages"] += non_en
    global_metrics["unknown_language_messages"] += int(lang_counter.get("unknown", 0))

    prev_counter = Counter(global_metrics.get("language_distribution", {}))
    prev_counter.update(lang_counter)
    global_metrics["language_distribution"] = dict(prev_counter)

    if cnt_conf:
        global_metrics["min_lang_confidence"] = min(
            float(global_metrics.get("min_lang_confidence", 1.0)),
            round(min_conf, 4),
        )
        global_metrics["max_lang_confidence"] = max(
            float(global_metrics.get("max_lang_confidence", 0.0)),
            round(max_conf, 4),
        )
        global_metrics["temp_lang_conf_total_sum"] += float(sum_conf)
        global_metrics["temp_lang_conf_total_count"] += int(cnt_conf)

        for k, v in conf_buckets.items():
            global_metrics[k] = int(global_metrics.get(k, 0)) + int(v)
