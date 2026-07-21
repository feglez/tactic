"""Per-dataset preprocessing orchestration (extract → clean → detect → export metrics)."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from tqdm import tqdm

from tactic.common.logging_utils import setup_logger
from tactic.common.metrics_export import to_native
from tactic.preprocessing.cleaning import clean_csv
from tactic.preprocessing.language import LanguageDetector, detect_languages
from tactic.preprocessing.ingest import tgdataset_json, darkgram_csv, tliveugr_json


@dataclass
class PreprocessConfig:
    """Runtime configuration shared by the three dataset preprocessors.

    TGDataset and TLiveUGR write per-file CSVs (`out_dir`), while DarkGram
    writes a single consolidated CSV (`out_csv`) and supports raw-file exclusion.
    """
    raw_dir: Path
    fasttext_model: Path
    out_dir: Path | None = None            # TGDataset / TLiveUGR
    out_csv: Path | None = None            # DarkGram
    chunk_size: int = 100_000
    min_words: int = 4
    exclude: Tuple[str, ...] = ()          # DarkGram only
    artifacts_dir: Path = Path("artifacts")
    logs_dir: Path = Path("logs")

    def __post_init__(self) -> None:
        """Validate basic invariants (no I/O side effects)."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0 (got {self.chunk_size})")
        if self.min_words < 0:
            raise ValueError(f"min_words must be >= 0 (got {self.min_words})")

        self.raw_dir = Path(self.raw_dir)
        self.fasttext_model = Path(self.fasttext_model)
        if self.out_dir is not None:
            self.out_dir = Path(self.out_dir)
        if self.out_csv is not None:
            self.out_csv = Path(self.out_csv)
        self.artifacts_dir = Path(self.artifacts_dir)
        self.logs_dir = Path(self.logs_dir)
        if not isinstance(self.exclude, tuple):
            self.exclude = tuple(self.exclude)


def config_from_yaml(cfg: dict) -> PreprocessConfig:
    """Build a PreprocessConfig from a loaded stage YAML."""
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})
    return PreprocessConfig(
        raw_dir=Path(paths["raw_dir"]),
        fasttext_model=Path(paths["fasttext_model"]),
        out_dir=Path(paths["out_dir"]) if "out_dir" in paths else None,
        out_csv=Path(paths["out_csv"]) if "out_csv" in paths else None,
        chunk_size=int(params.get("chunk_size", 100_000)),
        min_words=int(params.get("min_words", 4)),
        exclude=tuple(params.get("exclude", ())),
        artifacts_dir=Path(paths["artifacts_dir"]),
        logs_dir=Path(paths["logs_dir"]),
    )


# Metric key lists shared by all three datasets (tail sections).
_CLEANING_PERF_KEYS = [
    ####################################
    ####### Cleaning Performance #######
    ####################################

    # Total time spent to clean the messages (i.e., in the clean_message function)
    # NOTE: this time doesn't include the time spent filtering short messages
    "cleaning_runtime_sec",
    # Average time spent in the initial message cleaning per message (not including the short message filter)
    "avg_msg_cleaning_time_ms",
]

_LANGUAGE_KEYS = [
    ##################################
    ####### Language Detection #######
    ##################################

    # Total time spent detecting the language of the messages (i.e., in the detect_one function)
    "language_detection_runtime_sec",
    # Average time spent in language detection per message
    "avg_msg_language_detection_time_ms",
    # Quantity of messages whose language code isn't 'en' (includes unknown code)
    "non_english_messages",
    # Quantity of messages whose language code is 'unknown'
    "unknown_language_messages",
    # Quantity of messages of every language detected
    "language_distribution",
    # Top 5 most common languages
    "top_languages",

    #############################################
    ####### Language Detection Confidence #######
    #############################################

    # Several metrics for visualizing the language confidence scores obtained
    "min_lang_confidence",
    "max_lang_confidence",
    "avg_lang_confidence",
    "confidence_between_0.9_and_1_count",
    "confidence_between_0.8_and_0.89_count",
    "confidence_between_0.7_and_0.79_count",
    "confidence_between_0.6_and_0.69_count",
    "confidence_between_0.5_and_0.59_count",
    "confidence_between_0.4_and_0.49_count",
    "confidence_between_0.3_and_0.39_count",
    "confidence_between_0.2_and_0.29_count",
    "confidence_between_0.1_and_0.19_count",
    "confidence_between_0_and_0.09_count",
]

# Per-channel metric key order, identical for all three datasets except the
# source key (source_json for TGDataset/TLiveUGR, source_csv for DarkGram).
def _per_channel_keys(source_key: str) -> list:
    return [
        # Username of the channel
        "username",
        # File from which the data comes
        source_key,
        # Initial quantity of messages before any filter
        "total_original_messages",
        # Quantity of initial messages that were empty or NaN
        "initial_empty_or_nan_messages",
        # Quantity of remaining messages after cleaning them (after applying the clean_message function)
        # NOTE: this metric is calculated BEFORE filtering short messages (i.e. before applying the has_enough_words function)
        "messages_after_cleaning",
        # Quantity of messages removed shorter than cfg.min_words that were removed by the has_enough_words function
        "short_messages_removed",
        # Quantity of messages whose language code isn't 'en' (includes unknown code)
        "non_english_messages",
        # Quantity of messages whose language code is 'unknown'
        "unknown_language_messages",
        # Quantity of messages of every language detected
        "language_distribution",
    ]


def _export_metrics(
    metrics: Dict[str, Any],
    per_channel_metrics: Dict[str, Dict[str, Any]],
    metrics_ordered_keys: list,
    run_cfg: dict,
    artifacts_dir: Path,
    global_name: str,
    run_config_name: str,
    per_channel_name: str,
    per_channel_source_key: str,
) -> None:
    """Write ordered global metrics, the run-config snapshot, and per-channel metrics."""
    metrics_native = to_native(metrics)
    ordered_metrics = {k: metrics_native[k] for k in metrics_ordered_keys if k in metrics_native}

    with (artifacts_dir / global_name).open("w", encoding="utf-8") as f:
        json.dump(ordered_metrics, f, indent=4, ensure_ascii=False)

    with (artifacts_dir / run_config_name).open("w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=4, ensure_ascii=False)

    per_channel_metrics_ordered_keys = _per_channel_keys(per_channel_source_key)

    per_native = to_native(per_channel_metrics)
    per_channel_ordered: Dict[str, Dict[str, Any]] = {}

    for ch_id, stats in per_native.items():
        row = {k: stats[k] for k in per_channel_metrics_ordered_keys if k in stats}
        # Sort language distribution desc by count, then by lang
        dist = row.get("language_distribution")
        if isinstance(dist, dict):
            row["language_distribution"] = dict(sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))
        per_channel_ordered[ch_id] = row

    with (artifacts_dir / per_channel_name).open("w", encoding="utf-8") as f:
        json.dump(per_channel_ordered, f, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# TGDataset
# ---------------------------------------------------------------------------
def run_tgdataset(cfg_yaml: dict) -> None:
    """Run the full TGDataset preprocessing pipeline."""
    cfg = config_from_yaml(cfg_yaml)
    params = cfg_yaml.get("params", {})

    pipeline_start_time = time.time()

    # --- Setup output folders & logging ---
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = cfg.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = cfg.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    lang_logger = setup_logger(
        "language_detection_errors",
        logs_dir / "language_detection_errors.log",
    )

    # --- Metric accumulators ---
    metrics: Dict[str, Any] = defaultdict(int)
    per_channel_metrics: Dict[str, Dict[str, Any]] = {}

    # --- Stage 1: Extract JSON → CSV ---
    json_files = sorted(cfg.raw_dir.glob("*.json"))
    if not json_files:
        tqdm.write(f"[warn] No JSON files found in: {cfg.raw_dir}")

    for j in tqdm(
        json_files,
        desc="Extracting JSON and removing initial empty/NaN messages",
        dynamic_ncols=True,
        mininterval=2,
        leave=True,
    ):
        out_csv = cfg.out_dir / (j.stem + ".csv")
        tgdataset_json.extract_json(j, out_csv, metrics, per_channel_metrics, flush_size=cfg.chunk_size)

    metrics["avg_msg_initial_deletion_time_ms"] = (
        round(1000 * metrics["initial_deletion_runtime_sec"] / metrics["total_original_messages"], 2)
        if metrics["total_original_messages"] > 0 else None
    )

    # --- Stage 2: Clean CSVs ---
    ws_regex = params.get("whitespace_regex", "ascii_space_tab")
    regex_engine = params.get("regex_engine", "re")
    na_coercion = params.get("message_na_coercion", "astype_str")
    csv_files = sorted(cfg.out_dir.glob("*.csv"))
    for c in tqdm(csv_files, desc="Cleaning CSVs"):
        clean_csv(c, metrics, cfg, per_channel_metrics, whitespace_regex=ws_regex,
                  regex_engine=regex_engine, message_na_coercion=na_coercion)

    denom = (metrics["total_original_messages"] - metrics["initial_empty_or_nan_messages"])
    metrics["avg_msg_cleaning_time_ms"] = (
        round(1000 * metrics["cleaning_runtime_sec"] / denom, 2) if denom > 0 else None
    )

    # --- Stage 3: Language detection ---
    det = LanguageDetector(cfg.fasttext_model, lang_logger,
                           input_norm=params.get("fasttext_input_norm", "replace_newline"))

    # Recompute file list in case files were added/removed by earlier stages
    csv_files = sorted(cfg.out_dir.glob("*.csv"))
    for c in tqdm(csv_files, desc="Language detection"):
        tqdm.write(f"Detecting language on {c.name}...")
        detect_languages(c, metrics, det, cfg, per_channel_metrics)

    denom = (metrics["messages_after_cleaning"] - metrics["short_messages_removed"])
    metrics["avg_msg_language_detection_time_ms"] = (
        round(1000 * metrics["language_detection_runtime_sec"] / denom, 2) if denom > 0 else None
    )

    if metrics.get("temp_lang_conf_total_count", 0):
        metrics["avg_lang_confidence"] = round(
            metrics["temp_lang_conf_total_sum"] / metrics["temp_lang_conf_total_count"], 4
        )
    else:
        metrics["avg_lang_confidence"] = None

    metrics["top_languages"] = dict(Counter(metrics.get("language_distribution", {})).most_common(5))
    metrics["total_execution_time"] = round(time.time() - pipeline_start_time, 2)

    # --- Stages 4-5: metrics export (ordered keys) ---
    metrics_ordered_keys = [
        ######################
        ####### Volume #######
        ######################

        # Initial quantity of messages before any filter
        "total_original_messages",
        # Initial quantity of channels before any filter
        "initial_unique_channels",
        # Initial quantity of channels that didn't even have a text_messages object (i.e., it was missing/None/empty)
        "initial_channels_without_text_messages",
        # Should always be 0, initial quantity of channels whose text_messages object wasn't a dict
        "initial_channels_with_invalid_text_messages",
        # Quantity of channels after deleting those with all collected messages were empty or NaN
        "unique_channels_after_initial_filter",
        # Quantity of initial messages that were empty or NaN
        "initial_empty_or_nan_messages",
        # Time spent to delete all initial empty or NaN messages and store the rest in CSVs
        "initial_deletion_runtime_sec",
        # Average time spent in this initial filter per message
        "avg_msg_initial_deletion_time_ms",
        # Quantity of remaining channels after cleaning the messages (after applying the clean_message function)
        # NOTE: this metric is calculated BEFORE filtering short messages (i.e. before applying the has_enough_words function)
        "unique_channels_after_cleaning",
        # Quantity of remaining messages after cleaning them (after applying the clean_message function)
        # NOTE: this metric is calculated BEFORE filtering short messages, like the previous one
        "messages_after_cleaning",
        # Quantity of remaining channels after cleaning the messages and filtering short ones
        "unique_channels_after_whole_cleaning",
        # Quantity of messages removed shorter than cfg.min_words that were removed by the has_enough_words function
        "short_messages_removed",
        # Total execution time of all the preprocessing pipeline
        "total_execution_time",
    ] + _CLEANING_PERF_KEYS + _LANGUAGE_KEYS

    run_cfg = {
        "raw_dir": str(cfg.raw_dir),
        "out_dir": str(cfg.out_dir),
        "fasttext_model": str(cfg.fasttext_model),
        "chunk_size": cfg.chunk_size,
        "min_words": cfg.min_words,
        "generated_at_epoch": time.time(),
    }

    _export_metrics(
        metrics, per_channel_metrics, metrics_ordered_keys, run_cfg, artifacts_dir,
        global_name="tgdataset_preprocessing_metrics.json",
        run_config_name="tgdataset_run_config.json",
        per_channel_name="tgdataset_per_channel_metrics.json",
        per_channel_source_key="source_json",
    )


# ---------------------------------------------------------------------------
# DarkGram
# ---------------------------------------------------------------------------
def run_darkgram(cfg_yaml: dict) -> None:
    """Run the full DarkGram preprocessing pipeline."""
    cfg = config_from_yaml(cfg_yaml)
    params = cfg_yaml.get("params", {})

    pipeline_start_time = time.time()

    # --- Setup output folders & logging ---
    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
    logs_dir = cfg.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = cfg.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    lang_logger = setup_logger(
        "language_detection_errors",
        logs_dir / "language_detection_errors.log",
    )

    # --- Metric accumulators ---
    metrics: Dict[str, Any] = defaultdict(int)
    per_channel_metrics: Dict[str, Dict[str, Any]] = {}

    # -------------------------
    # Stage 1: Read & consolidate raw CSVs
    # -------------------------
    csv_files = sorted(
        p for p in cfg.raw_dir.glob("*.csv")
        if p.name not in cfg.exclude
    )

    if not csv_files:
        tqdm.write(f"[warn] No CSV files found in: {cfg.raw_dir} (after excluding: {list(cfg.exclude)}).")

    tqdm.write(f"Consolidating raw CSVs")
    darkgram_csv.consolidate_raw_csvs(
        csv_files=csv_files,
        out_csv=cfg.out_csv,
        metrics=metrics,
        per_channel_metrics=per_channel_metrics,
        chunk_size=cfg.chunk_size,
    )

    if not cfg.out_csv.exists():
        tqdm.write("[warn] Consolidation produced no rows. Writing empty CSV and metrics.")
        empty_cols = ["Channel ID", "Message ID", "Username", "message", "category"]
        pd.DataFrame(columns=empty_cols).to_csv(cfg.out_csv, index=False, encoding="utf-8")
    else:
        tqdm.write("Sorting consolidated CSV by Channel ID, then Message ID")
        darkgram_csv.sort_consolidated_csv(cfg.out_csv)

    metrics["avg_msg_initial_deletion_time_ms"] = (
        round(1000 * metrics["initial_deletion_runtime_sec"] / metrics["total_original_messages"], 2)
        if metrics["total_original_messages"] > 0 else None
    )

    # --- Stage 2: Clean CSVs ---
    tqdm.write(f"Cleaning consolidated CSV")
    clean_csv(cfg.out_csv, metrics, cfg, per_channel_metrics,
              strip_presidio_entities=True,
              whitespace_regex=params.get("whitespace_regex", "unicode_horizontal"),
              regex_engine=params.get("regex_engine", "regex"),
              message_na_coercion=params.get("message_na_coercion", "fillna_empty"))

    denom = (metrics["total_original_messages"] - metrics["initial_empty_or_nan_messages"])
    metrics["avg_msg_cleaning_time_ms"] = (
        round(1000 * metrics["cleaning_runtime_sec"] / denom, 2) if denom > 0 else None
    )

    # --- Stage 3: Language detection ---
    det = LanguageDetector(cfg.fasttext_model, lang_logger,
                           input_norm=params.get("fasttext_input_norm", "split_join"))

    tqdm.write(f"Detecting languages on consolidated CSV")
    detect_languages(cfg.out_csv, metrics, det, cfg, per_channel_metrics)

    denom = (metrics["messages_after_cleaning"] - metrics["short_messages_removed"])
    metrics["avg_msg_language_detection_time_ms"] = (
        round(1000 * metrics["language_detection_runtime_sec"] / denom, 2) if denom > 0 else None
    )

    if metrics.get("temp_lang_conf_total_count", 0):
        metrics["avg_lang_confidence"] = round(
            metrics["temp_lang_conf_total_sum"] / metrics["temp_lang_conf_total_count"], 4
        )
    else:
        metrics["avg_lang_confidence"] = None

    metrics["top_languages"] = dict(Counter(metrics.get("language_distribution", {})).most_common(5))
    metrics["total_execution_time"] = round(time.time() - pipeline_start_time, 2)

    # --- Stages 4-5: metrics export (ordered keys) ---
    metrics_ordered_keys = [
        ######################
        ####### Volume #######
        ######################

        # Initial quantity of messages before any filter
        "total_original_messages",
        # Initial quantity of channels before any filter
        "initial_unique_channels",
        # Quantity of channels after deleting those with all collected messages were empty or NaN
        "unique_channels_after_initial_filter",
        # Quantity of initial messages that were empty or NaN
        "initial_empty_or_nan_messages",
        # Time spent to delete all initial empty or NaN messages and store the rest in CSVs
        "initial_deletion_runtime_sec",
        # Average time spent in this initial filter per message
        "avg_msg_initial_deletion_time_ms",
        # Quantity of remaining channels after cleaning the messages (after applying the clean_message function)
        # NOTE: this metric is calculated BEFORE filtering short messages (i.e. before applying the has_enough_words function)
        "unique_channels_after_cleaning",
        # Quantity of remaining messages after cleaning them (after applying the clean_message function)
        # NOTE: this metric is calculated BEFORE filtering short messages, like the previous one
        "messages_after_cleaning",
        # Quantity of remaining channels after cleaning the messages and filtering short ones
        "unique_channels_after_whole_cleaning",
        # Quantity of messages removed shorter than cfg.min_words that were removed by the has_enough_words function
        "short_messages_removed",
        # Total execution time of all the preprocessing pipeline
        "total_execution_time",
    ] + _CLEANING_PERF_KEYS + _LANGUAGE_KEYS

    run_cfg = {
        "raw_dir": str(cfg.raw_dir),
        "out_csv": str(cfg.out_csv),
        "fasttext_model": str(cfg.fasttext_model),
        "chunk_size": cfg.chunk_size,
        "min_words": cfg.min_words,
        "exclude": list(cfg.exclude),
        "generated_at_epoch": time.time(),
    }

    _export_metrics(
        metrics, per_channel_metrics, metrics_ordered_keys, run_cfg, artifacts_dir,
        global_name="darkgram_preprocessing_metrics.json",
        run_config_name="darkgram_run_config.json",
        per_channel_name="darkgram_per_channel_metrics.json",
        per_channel_source_key="source_csv",
    )


# ---------------------------------------------------------------------------
# TLiveUGR
# ---------------------------------------------------------------------------
def run_tliveugr(cfg_yaml: dict) -> None:
    """Run the full TLiveUGR bot-JSON preprocessing pipeline."""
    cfg = config_from_yaml(cfg_yaml)
    params = cfg_yaml.get("params", {})
    DEBUG_MODE = bool(params.get("debug_mode", False))

    pipeline_start_time = time.time()

    # --- Setup output folders & logging ---
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = cfg.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = cfg.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    lang_logger = setup_logger(
        "language_detection_errors",
        logs_dir / "language_detection_errors.log",
    )

    # --- Metric accumulators ---
    metrics: Dict[str, Any] = defaultdict(int)
    per_channel_metrics: Dict[str, Dict[str, Any]] = {}

    # --- Stage 1: Extract JSON → CSV ---
    json_files = sorted(cfg.raw_dir.glob("*.json"))
    if not json_files:
        tqdm.write(f"[warn] No JSON files found in: {cfg.raw_dir}")

    DEBUG_COUNTER = 0
    produced_csvs: set = set()  # CSVs written by this run's extraction stage

    for j in tqdm(
        json_files,
        desc="Extracting JSON and removing initial empty/NaN messages",
        dynamic_ncols=True,
        mininterval=2,
        leave=True,
    ):
        out_csv = cfg.out_dir / (j.stem + ".csv")
        tliveugr_json.extract_json(j, out_csv, metrics, per_channel_metrics,
                                   flush_size=cfg.chunk_size)
        produced_csvs.add(out_csv.name)

        if DEBUG_MODE:
            DEBUG_COUNTER += 1
            if DEBUG_COUNTER == 2:
                break

    metrics["avg_msg_initial_deletion_time_ms"] = (
        round(1000 * metrics["initial_deletion_runtime_sec"] / metrics["total_original_messages"], 2)
        if metrics["total_original_messages"] > 0 else None
    )

    # Stages 2-3 process only the CSVs produced by this run's extraction stage,
    # so files left in out_dir by earlier runs are never folded into this run.
    def _stage_files() -> list:
        return [f for f in sorted(cfg.out_dir.glob("*.csv")) if f.name in produced_csvs]

    # --- Stage 2: Clean CSVs ---
    csv_files = _stage_files()
    for c in tqdm(csv_files, desc="Cleaning CSVs"):
        clean_csv(c, metrics, cfg, per_channel_metrics, skip_empty_chunks=True,
                  whitespace_regex=params.get("whitespace_regex", "unicode_horizontal"),
                  regex_engine=params.get("regex_engine", "regex"),
                  message_na_coercion=params.get("message_na_coercion", "fillna_empty"))

    denom = (metrics["total_original_messages"] - metrics["initial_empty_or_nan_messages"])
    metrics["avg_msg_cleaning_time_ms"] = (
        round(1000 * metrics["cleaning_runtime_sec"] / denom, 2) if denom > 0 else None
    )

    # --- Stage 3: Language detection ---
    det = LanguageDetector(cfg.fasttext_model, lang_logger,
                           input_norm=params.get("fasttext_input_norm", "split_join"))

    # Recompute file list in case files were added/removed by earlier stages
    csv_files = _stage_files()
    for c in tqdm(csv_files, desc="Language detection"):
        tqdm.write(f"Detecting language on {c.name}...")
        detect_languages(c, metrics, det, cfg, per_channel_metrics)

    denom = (metrics["messages_after_cleaning"] - metrics["short_messages_removed"])
    metrics["avg_msg_language_detection_time_ms"] = (
        round(1000 * metrics["language_detection_runtime_sec"] / denom, 2) if denom > 0 else None
    )

    if metrics.get("temp_lang_conf_total_count", 0):
        metrics["avg_lang_confidence"] = round(
            metrics["temp_lang_conf_total_sum"] / metrics["temp_lang_conf_total_count"], 4
        )
    else:
        metrics["avg_lang_confidence"] = None

    metrics["top_languages"] = dict(Counter(metrics.get("language_distribution", {})).most_common(5))
    metrics["total_execution_time"] = round(time.time() - pipeline_start_time, 2)

    # --- Stages 4-5: metrics export (ordered keys) ---
    metrics_ordered_keys = [
        ######################
        ####### Volume #######
        ######################

        # Initial quantity of messages before any filter
        "total_original_messages",
        # Initial quantity of channels before any filter
        "initial_unique_channels",
        # Initial quantity of channels that didn't even have a text_messages object (i.e., it was missing/None/empty)
        "initial_channels_without_text_messages",
        # Quantity of channels whose JSON lacks a channel id (should be 0)
        "channels_without_id",
        # Quantity of channels after deleting those with all collected messages were empty or NaN
        "unique_channels_after_initial_filter",
        # Quantity of initial messages that were empty or NaN
        "initial_empty_or_nan_messages",
        # Time spent to delete all initial empty or NaN messages and store the rest in CSVs
        "initial_deletion_runtime_sec",
        # Average time spent in this initial filter per message
        "avg_msg_initial_deletion_time_ms",
        # Quantity of remaining channels after cleaning the messages (after applying the clean_message function)
        # NOTE: this metric is calculated BEFORE filtering short messages (i.e. before applying the has_enough_words function)
        "unique_channels_after_cleaning",
        # Quantity of remaining messages after cleaning them (after applying the clean_message function)
        # NOTE: this metric is calculated BEFORE filtering short messages, like the previous one
        "messages_after_cleaning",
        # Quantity of remaining channels after cleaning the messages and filtering short ones
        "unique_channels_after_whole_cleaning",
        # Quantity of messages removed shorter than cfg.min_words that were removed by the has_enough_words function
        "short_messages_removed",
        # Total execution time of all the preprocessing pipeline
        "total_execution_time",
    ] + _CLEANING_PERF_KEYS + _LANGUAGE_KEYS

    run_cfg = {
        "raw_dir": str(cfg.raw_dir),
        "out_dir": str(cfg.out_dir),
        "fasttext_model": str(cfg.fasttext_model),
        "chunk_size": cfg.chunk_size,
        "min_words": cfg.min_words,
        "generated_at_epoch": time.time(),
    }

    _export_metrics(
        metrics, per_channel_metrics, metrics_ordered_keys, run_cfg, artifacts_dir,
        global_name="tliveugr_preprocessing_metrics.json",
        run_config_name="tliveugr_run_config.json",
        per_channel_name="tliveugr_per_channel_metrics.json",
        per_channel_source_key="source_json",
    )
