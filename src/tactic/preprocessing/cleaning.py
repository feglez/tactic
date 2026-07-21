"""Message cleaning: regex/emoji normalization and the chunked clean-CSV stage."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Set

import emoji                   # emoji stripping
import pandas as pd
import regex                   # Unicode-aware engine that supports \h for horizontal whitespace
from tqdm import tqdm

from tactic.common.io_utils import count_chunks

# -------------------------
# Regexes (precompiled for speed, once per engine)
# -------------------------
# Two engines are supported (params.regex_engine, set per dataset config):
#   "re"    → stdlib re
#   "regex" → regex module (its `\w` also matches Unicode combining marks, and
#             it provides `\h` for horizontal whitespace)
ENGINE_RE = "re"
ENGINE_REGEX = "regex"


def _compile_patterns(mod) -> dict:
    """Compile the shared scrub patterns under one engine (`re` or `regex`).

    Patterns:
      url:      URLs, both http/https and bare www. ("https://x.y", "www.example.com")
      email:    permissive email-like tokens, intentionally not RFC-complete
                to avoid backtracking ("name.surname+tag@sub.domain.tld")
      username: Telegram-style @usernames. `(?<!\\S)` anchors at start /
                after whitespace so we don't match inside words
      invite:   Telegram invites (t.me / telegram.me / telegram.dog /
                tg://join?invite= ...)
      ws_ascii: ASCII space/tab collapse (`[ \\t]+`)
      entity:   Presidio placeholder tags like <PERSON> (DarkGram only)
    """
    return {
        "url": mod.compile(r"https?://\S+|www\.\S+", mod.I),
        "email": mod.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", mod.I),
        "username": mod.compile(r"(?<!\S)@\w+"),
        "invite": mod.compile(
            r"(?:https?://)?(?:t\.me|telegram\.me|telegram\.dog|tg://join\?invite=)"
            r"(?:/joinchat/|/\+|/)?[\w-]+",
            mod.I,
        ),
        "ws_ascii": mod.compile(r"[ \t]+"),
    }


# Whitespace collapse pattern (params.whitespace_regex, set per dataset config).
# Consecutive *horizontal* whitespace collapses to a single space. Newlines are
# preserved in both variants.
#   "ascii_space_tab"    → [ \t]+  (ASCII space/tab only)
#   "unicode_horizontal" → \h+     (all Unicode horizontal whitespace.
#                                   Requires params.regex_engine=regex, since
#                                   \h does not exist in stdlib re)
WS_UNICODE_HORIZONTAL = "unicode_horizontal"
WS_ASCII_SPACE_TAB = "ascii_space_tab"

# =========================
# Presidio entities, used for tag stripping like <PERSON> (DarkGram only).
# The DarkGram dataset ships pre-anonymized with Microsoft Presidio
# placeholders, and cleaning removes them.
# https://microsoft.github.io/presidio/supported_entities/
# =========================
PRESIDIO_ENTITIES = [
    "CREDIT_CARD", "CRYPTO", "DATE_TIME", "EMAIL_ADDRESS", "IBAN_CODE",
    "IP_ADDRESS", "NRP", "LOCATION", "PERSON", "PHONE_NUMBER",
    "MEDICAL_LICENSE", "URL",
    "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_ITIN",
    "US_PASSPORT", "US_SSN",
    "UK_NHS", "UK_NINO",
    "ES_NIF", "ES_NIE",
    "IT_FISCAL_CODE", "IT_DRIVER_LICENSE", "IT_VAT_CODE",
    "IT_PASSPORT", "IT_IDENTITY_CARD",
    "PL_PESEL",
    "SG_NRIC_FIN", "SG_UEN",
    "AU_ABN", "AU_ACN", "AU_TFN", "AU_MEDICARE",
    "IN_PAN", "IN_AADHAAR", "IN_VEHICLE_REGISTRATION",
    "IN_VOTER", "IN_PASSPORT",
    "FI_PERSONAL_IDENTITY_CODE"
]

# Per-engine compiled pattern tables. The `regex` table additionally carries
# the \h+ Unicode-horizontal-whitespace pattern (regex-module-only syntax).
_ENGINE_PATTERNS: Dict[str, dict] = {
    ENGINE_REGEX: _compile_patterns(regex),
    ENGINE_RE: _compile_patterns(re),
}
_ENGINE_PATTERNS[ENGINE_REGEX]["ws_unicode"] = regex.compile(r"\h+")
for _engine, _mod in ((ENGINE_REGEX, regex), (ENGINE_RE, re)):
    _ENGINE_PATTERNS[_engine]["entity"] = _mod.compile(
        r"<(?:" + "|".join(_mod.escape(e) for e in PRESIDIO_ENTITIES) + r")>"
    )


def clean_message(text: str, strip_presidio_entities: bool = False,
                  whitespace_regex: str = WS_UNICODE_HORIZONTAL,
                  regex_engine: str = ENGINE_REGEX) -> str:
    """Normalize a raw Telegram message for downstream processing.

    This function:
      - removes URLs, email-like tokens, @usernames, and Telegram invite links
        (and, for DarkGram, Presidio placeholder tags like <PERSON>)
      - strips all emoji characters
      - collapses consecutive *horizontal* whitespace to a single space
      - preserves internal newlines but trims leading/trailing whitespace

    Args:
        text: Raw message text. Non-strings are coerced to string, and `None` becomes "".
        strip_presidio_entities: When True, also strips Presidio placeholder tags
            (DarkGram only).
        whitespace_regex: `params.whitespace_regex`, selecting which
            whitespace-collapse pattern to apply (see the constants above).
        regex_engine: `params.regex_engine`, selecting which engine the scrub
            patterns were compiled with (see the engine note at the top of this
            module).

    Returns:
        A cleaned message string with sensitive/noisy tokens removed and
        normalized spacing. May return "" if nothing remains.

    Notes:
        - Email matching uses a permissive pattern (1+ char "TLD").
        - Newlines are preserved *inside* the message. Only edge whitespace is stripped.
    """
    # Normalize input to a string early
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if not text:
        return ""

    try:
        pats = _ENGINE_PATTERNS[regex_engine]
    except KeyError:
        raise ValueError(f"Unknown regex_engine: {regex_engine!r}")

    # Remove noisy/sensitive tokens
    text = pats["url"].sub("", text)
    text = pats["email"].sub("", text)
    text = pats["username"].sub("", text)
    text = pats["invite"].sub("", text)
    if strip_presidio_entities:
        text = pats["entity"].sub("", text)

    # Strip all emoji
    text = emoji.replace_emoji(text, replace="")

    # Collapse horizontal whitespace (keep newlines intact)
    if whitespace_regex == WS_ASCII_SPACE_TAB:
        ws_pattern = pats["ws_ascii"]
    elif whitespace_regex == WS_UNICODE_HORIZONTAL:
        ws_pattern = pats.get("ws_unicode")
        if ws_pattern is None:
            raise ValueError(
                "whitespace_regex=unicode_horizontal requires "
                "regex_engine=regex (stdlib re has no \\h)."
            )
    else:
        raise ValueError(f"Unknown whitespace_regex: {whitespace_regex!r}")
    text = ws_pattern.sub(" ", text)

    # Trim edges
    return text.strip()


def has_enough_words(text: str, min_words: int) -> bool:
    """Check if a message has at least `min_words` whitespace-delimited tokens.

    This is a lightweight, language-agnostic heuristic that counts tokens via
    `str.split()` (i.e., any whitespace, including spaces, tabs, and newlines,
    acts as a separator).
    Use after `clean_message`, which already strips URLs/emails/emoji and collapses
    horizontal whitespace.

    Args:
        text: The message to evaluate. Non-strings are coerced to string, and None becomes "".
        min_words: Minimum number of tokens required (must be >= 0).

    Returns:
        True if the text contains at least `min_words` tokens, otherwise False.

    Notes:
        - Punctuation remains attached to tokens (e.g., "hello!" counts as one).
        - Newlines count as separators (multi-line messages can meet the threshold).
        - If `min_words` <= 0, the function returns True.
    """
    if min_words <= 0:
        return True

    if not isinstance(text, str):
        text = "" if text is None else str(text)

    if not text:
        return False

    return len(text.split()) >= min_words


def clean_csv(
    csv_path: Path,
    global_metrics: Dict[str, Any],
    cfg,
    channel_stats: Dict[str, Dict[str, Any]],
    strip_presidio_entities: bool = False,
    skip_empty_chunks: bool = False,
    whitespace_regex: str = WS_UNICODE_HORIZONTAL,
    regex_engine: str = ENGINE_REGEX,
    message_na_coercion: str = "fillna_empty",
) -> None:
    """Clean a CSV in chunks and rewrite it in place.

    Pipeline per chunk:
      1) Coerce `message` to string and run `clean_message`.
      2) Drop rows with empty messages (after cleaning).
      3) Record per-channel counts *after cleaning* (pre short-filter) and update global metrics.
      4) Filter out short messages (`has_enough_words(message, cfg.min_words)`).
      5) Record per-channel short removals and update global metrics.
      6) Append the cleaned rows to a temporary file.

    At the end, the temp file atomically replaces the original CSV.

    Args:
        csv_path: Path to the CSV produced by the ingestion stage.
        global_metrics: Global counters/timers to update in place.
        cfg: Runtime configuration object with `chunk_size` and `min_words` attributes.
        channel_stats: Per-channel counters to update in place.
        strip_presidio_entities: Passed through to `clean_message` (DarkGram: True).
        skip_empty_chunks: When True, a chunk with no rows left after cleaning is
            skipped without invoking the short-message filter or writing (TLiveUGR).
        whitespace_regex: `params.whitespace_regex`, passed through to `clean_message`.
        regex_engine: `params.regex_engine`, passed through to `clean_message`.
        message_na_coercion: `params.message_na_coercion`, controlling how NaN
            messages are coerced before cleaning (see step 1 below).

    Notes:
        - `cleaning_runtime_sec` measures only the time spent in `clean_message`,
          not the short-message filtering phase.
    """
    if message_na_coercion not in ("fillna_empty", "astype_str"):
        raise ValueError(f"Unknown message_na_coercion: {message_na_coercion!r}")

    # Temp destination (same folder)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()  # remove stale temp file

    # Progress bar uses an actual chunk count (newline-safe vs. counting file lines)
    num_chunks = count_chunks(csv_path, chunk_size=cfg.chunk_size)

    reader = pd.read_csv(
        csv_path,
        chunksize=cfg.chunk_size,
        encoding="utf-8",
        dtype={"Channel ID": "string", "Message ID": "string"},
    )

    cleaning_time_sec = 0.0
    rows_after_clean_total = 0
    channels_after_clean: Set[str] = set()         # pre short-filter (global agg)
    channels_after_whole_clean: Set[str] = set()   # post short-filter (global agg)

    for i, chunk in enumerate(tqdm(reader, total=num_chunks, desc=f"Cleaning {csv_path.name}")):
        # Ensure strings before cleaning (params.message_na_coercion):
        #   "fillna_empty" → fillna("").astype(str): NaN becomes "" and is
        #                    removed by the empty-message filter
        #   "astype_str"   → astype(str): NaN becomes the string "nan" and is
        #                    removed by the short-message filter
        if message_na_coercion == "astype_str":
            chunk["message"] = chunk["message"].astype(str)
        else:
            chunk["message"] = chunk["message"].fillna("").astype(str)

        # --- Step 1: text normalization (timed) ---
        t0 = time.time()
        tqdm.pandas(desc=f"Chunk {i}")
        chunk["message"] = chunk["message"].progress_apply(
            lambda s: clean_message(s, strip_presidio_entities, whitespace_regex, regex_engine)
        )
        t1 = time.time()
        cleaning_time_sec += (t1 - t0)

        # --- Step 2: drop empty messages after cleaning ---
        chunk = chunk[chunk["message"] != ""]
        pre_count = len(chunk)

        # --- Step 3: per-channel counts after cleaning (pre short-filter) ---
        pre_by_channel = chunk.groupby("Channel ID", dropna=False).size()
        for ch_id, cnt in pre_by_channel.items():
            stats = channel_stats.setdefault(str(ch_id), {})
            stats["messages_after_cleaning"] = stats.get("messages_after_cleaning", 0) + int(cnt)

        # Global accumulators (pre short-filter)
        # (skip NA channel ids to avoid a literal "<NA>" entry)
        channels_after_clean.update([str(x) for x in pre_by_channel.index if pd.notna(x)])
        rows_after_clean_total += pre_count

        if skip_empty_chunks and pre_count == 0:
            # Nothing to short-filter or write, so skip to the next chunk.
            continue

        # --- Step 4: filter short messages ---
        tqdm.pandas(desc="Filtering short messages")
        chunk = chunk[chunk["message"].progress_apply(lambda s: has_enough_words(s, cfg.min_words))]

        # --- Step 5: per-channel short removals & global metric ---
        post_by_channel = chunk.groupby("Channel ID", dropna=False).size()
        for ch_id, pre_cnt in pre_by_channel.items():
            post_cnt = int(post_by_channel.get(ch_id, 0))
            removed_cnt = int(pre_cnt) - post_cnt
            stats = channel_stats.setdefault(str(ch_id), {})
            stats["short_messages_removed"] = stats.get("short_messages_removed", 0) + removed_cnt

        # Global accumulators (post short-filter)
        channels_after_whole_clean.update([str(x) for x in post_by_channel.index if pd.notna(x)])
        global_metrics["short_messages_removed"] += pre_count - len(chunk)

        # --- Step 6: append cleaned rows to temp file ---
        chunk.to_csv(
            tmp_path,
            mode="a",
            index=False,
            header=not tmp_path.exists(),
            encoding="utf-8",
        )

    # --- Finalize global metrics ---
    global_metrics["cleaning_runtime_sec"] += round(cleaning_time_sec, 2)
    global_metrics["messages_after_cleaning"] += rows_after_clean_total
    global_metrics["unique_channels_after_cleaning"] += len(channels_after_clean)
    global_metrics["unique_channels_after_whole_cleaning"] += len(channels_after_whole_clean)

    # If no rows were written (all removed), create an empty CSV with the original header
    if not tmp_path.exists():
        header_cols = pd.read_csv(csv_path, nrows=0, encoding="utf-8").columns.tolist()
        pd.DataFrame(columns=header_cols).to_csv(tmp_path, index=False, encoding="utf-8")

    # Atomically replace the original CSV with the cleaned version
    os.replace(tmp_path, csv_path)
