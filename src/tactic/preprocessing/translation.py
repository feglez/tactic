"""MarianMT translation to English for the TGDataset selection and the DarkGram CSV.

The two translators share the MarianMT load/translate approach but differ
subtly (the DarkGram variant lowercases the language code and wraps text in a
list for the tokenizer, while the TGDataset variant does neither), so they are kept
separate with independent model caches.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

# -------------------------
# Translation runtime (device & caches)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TGDataset translator cache
_translation_cache: Dict[str, Optional[Tuple[MarianTokenizer, MarianMTModel]]] = {}

# DarkGram translator cache
_model_cache: Dict[str, Optional[Tuple[MarianTokenizer, MarianMTModel]]] = {}


# ===========================================================================
# TGDataset selected-messages translator
# ===========================================================================

def load_translation_model(lang_code: str) -> Optional[Tuple[MarianTokenizer, MarianMTModel]]:
    """Load (and cache) a MarianMT model for `<lang_code>-en` if available.

    NOTE: unlike the DarkGram variant below, the language code is used as-is
    (not normalized/lowercased).
    """
    if lang_code in _translation_cache:
        return _translation_cache[lang_code]

    model_name = f"Helsinki-NLP/opus-mt-{lang_code}-en"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        model.eval()
        _translation_cache[lang_code] = (tokenizer, model)
        return _translation_cache[lang_code]
    except Exception:
        _translation_cache[lang_code] = None
        return None


def translate_one(text: str, lang_code: str) -> Optional[str]:
    """Translate a single string to English using the model for `lang_code`.

    Returns:
      The translated string, or None on failure/unavailable model.
    """
    pair = load_translation_model(lang_code)
    if not pair:
        return None
    tokenizer, model = pair
    try:
        # Simple single-example generation. (Batching can be added later if needed.)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_length=512)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception:
        return None


def load_selected_indices(path: Path) -> Dict[str, Set[int]]:
    """Load (ch_ID, message_index) selection into a dict: ch_ID -> set(indices)."""
    df = pd.read_csv(path)
    if not {"ch_ID", "message_index"}.issubset(df.columns):
        raise ValueError(f"{path} must have columns: ch_ID, message_index")

    df["ch_ID"] = df["ch_ID"].astype(str)
    df["message_index"] = df["message_index"].astype(int)

    index_map: Dict[str, Set[int]] = defaultdict(set)
    for ch, idx in zip(df["ch_ID"], df["message_index"]):
        index_map[ch].add(int(idx))
    return index_map


def build_channel_file_map(
    counts_csv: Path,
    data_dir: Optional[Path] = None,
    cache_json: Optional[Path] = None,
) -> Dict[str, str]:
    """Build ch_ID → source_csv mapping.

    Preferred source: the `source_csv` column of channel_message_counts.csv.
    When that column is absent (a counts file with only [ch_ID, message_count]),
    the mapping is rebuilt by scanning the processed CSVs once (Channel ID
    column only) and cached as `channel_file_map.json` for later runs. Each
    channel lives in exactly one processed CSV, so the mapping content is
    identical either way.
    """
    header = pd.read_csv(counts_csv, nrows=0)
    if "source_csv" not in header.columns:
        if cache_json is not None and Path(cache_json).exists():
            print("🔹 Loading channel to file mapping (cached channel_file_map.json)...")
            with Path(cache_json).open("r", encoding="utf-8") as f:
                return {str(k): str(v) for k, v in json.load(f).items()}
        if data_dir is None:
            raise ValueError(
                f"{counts_csv} has no source_csv column and no data_dir/cache was "
                "given to rebuild the channel→file map."
            )
        print("🔹 Building channel to file mapping by scanning processed CSVs "
              "(one-time cost, cached as channel_file_map.json)...")
        mapping: Dict[str, str] = {}
        for csv_path in sorted(Path(data_dir).glob("*.csv")):
            for chunk in pd.read_csv(csv_path, usecols=["Channel ID"],
                                     dtype={"Channel ID": "string"},
                                     chunksize=1_000_000):
                for ch in chunk["Channel ID"].dropna().unique():
                    mapping.setdefault(str(ch), csv_path.name)
        if cache_json is not None:
            with Path(cache_json).open("w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False)
        return mapping

    df = pd.read_csv(counts_csv, usecols=["ch_ID", "source_csv"])
    if not {"ch_ID", "source_csv"}.issubset(df.columns):
        raise ValueError(f"{counts_csv} must contain columns: ch_ID, source_csv")

    df["ch_ID"] = df["ch_ID"].astype(str)
    df["source_csv"] = df["source_csv"].astype(str)
    return dict(zip(df["ch_ID"], df["source_csv"]))


def translate_selection(
    data_dir: Path,
    selection_csv: Path,
    counts_csv: Path,
    out_csv: Path,
    log_json: Path,
    save_every: int = 180,
    map_cache_json: Optional[Path] = None,
) -> None:
    """Translate the selected messages and write the dataset and metrics.

    Output CSV columns: Channel ID, original_message, translated_message, language.

    Metrics: `messages_translated` counts every selected row, including English
    pass-throughs, while `translation_failures` counts non-English rows whose
    translation returned None.
    """
    print(f"🔧 Using device: {device}")

    # Prepare output file
    if out_csv.exists():
        out_csv.unlink()

    # Load inputs
    print("🔹 Loading selected message indices...")
    index_map = load_selected_indices(selection_csv)

    print("🔹 Loading channel → source CSV mapping...")
    ch_to_file = build_channel_file_map(counts_csv, data_dir=data_dir,
                                        cache_json=map_cache_json)

    # Metrics
    metrics = defaultdict(int)  # messages_translated, translation_failures, channels_skipped, channels_processed
    translated_rows: List[dict] = []
    total_saved = 0

    # Channels are iterated in the selection file's first-seen (insertion)
    # order, which fixes the output dataset's channel order. Downstream
    # positional alignment depends on it, do not reorder.
    channel_iter = list(index_map.keys())  # dict preserves first-seen order
    for ch_id in tqdm(channel_iter, desc="Translating selected channels"):
        selected_indices = index_map[ch_id]
        file_name = ch_to_file.get(ch_id)

        if not file_name:
            print(f"⚠️  Skipping channel {ch_id}: no associated file in counts CSV.")
            metrics["channels_skipped"] += 1
            continue

        csv_path = data_dir / file_name
        if not csv_path.exists():
            print(f"⚠️  Skipping channel {ch_id}: source CSV not found at {csv_path}.")
            metrics["channels_skipped"] += 1
            continue

        print(f"🔍 Processing channel {ch_id} from file {file_name}...")
        # Read only the needed columns and coerce Channel ID to string
        chunks = pd.read_csv(
            csv_path,
            chunksize=10_000,
            usecols=["Channel ID", "message", "language"],
            dtype={"Channel ID": "string", "message": "string", "language": "string"},
            encoding="utf-8",
        )

        msg_idx = 0  # channel-local index over *this channel's* rows only
        for chunk in chunks:
            # Keep rows for the current channel only
            chunk = chunk[chunk["Channel ID"] == ch_id]
            if chunk.empty:
                continue

            if chunk["message"].isna().any() or chunk["language"].isna().any():
                raise ValueError(f"Found NaNs in {csv_path}, upstream pipeline invariant broken.")

            # Walk rows in order, translate when the channel-local index is selected
            for _, row in chunk.iterrows():
                if msg_idx in selected_indices:
                    text = row["message"]
                    lang = row["language"] or "en"

                    if lang == "en":
                        translated = text  # pass-through
                    else:
                        translated = translate_one(text, lang)

                    translated_rows.append(
                        {
                            "Channel ID": ch_id,
                            "original_message": text,
                            "translated_message": translated if translated is not None else np.nan,
                            "language": lang,
                        }
                    )

                    metrics["messages_translated"] += 1
                    if translated is None and lang != "en":
                        metrics["translation_failures"] += 1

                    # Periodic flush
                    if len(translated_rows) >= save_every:
                        pd.DataFrame(translated_rows).to_csv(
                            out_csv,
                            mode="a",
                            header=not out_csv.exists(),
                            index=False,
                            encoding="utf-8",
                        )
                        total_saved += len(translated_rows)
                        translated_rows.clear()
                # Increment channel-local index for *every* row in this channel
                msg_idx += 1

        metrics["channels_processed"] += 1

    # Final flush
    if translated_rows:
        pd.DataFrame(translated_rows).to_csv(
            out_csv,
            mode="a",
            header=not out_csv.exists(),
            index=False,
            encoding="utf-8",
        )
        total_saved += len(translated_rows)
        translated_rows.clear()

    # Write metrics
    metrics["total_saved"] = int(total_saved)
    with log_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ Done. Translated messages saved to {out_csv}. Metrics written to {log_json}.")


# ===========================================================================
# DarkGram whole-CSV translator
# ===========================================================================

def load_translation_model_darkgram(lang_code: str) -> Optional[Tuple[MarianTokenizer, MarianMTModel]]:
    """Load/cache a MarianMT model for '<lang_code>-en'. Returns None if unavailable."""
    lang_code = (lang_code or "").strip().lower()
    if lang_code in _model_cache:
        return _model_cache[lang_code]

    model_name = f"Helsinki-NLP/opus-mt-{lang_code}-en"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        model.eval()
        _model_cache[lang_code] = (tokenizer, model)
    except Exception:
        _model_cache[lang_code] = None
    return _model_cache[lang_code]


def translate_one_darkgram(text: str, lang_code: str, max_length: int = 512) -> Optional[str]:
    """Translate a single message to English, returning None if the model is missing or translation fails."""
    pair = load_translation_model_darkgram(lang_code)
    if not pair:
        return None
    tokenizer, model = pair
    try:
        enc = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_length=max_length)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception:
        return None


def translate_csv(
    in_csv: Path,
    out_csv: Path,
    metrics_json: Path,
    chunk_size: int = 50_000,
    max_length: int = 512,
    save_every: int = 2_000,
) -> None:
    """Stream the input CSV and write an output CSV with English translations where needed (row-by-row).

    Input columns  (required):
      Channel ID, Message ID, Username, message, category, language, lang_confidence
    Output columns (exact order):
      Channel ID, Message ID, Username, original_message, translated_message, category, language, lang_confidence

    Rows whose language is ""/"unknown" get translated_message = NaN with no
    translation attempt.
    """
    print(f"🔧 Device: {device}")
    if out_csv.exists():
        out_csv.unlink()

    required_cols = [
        "Channel ID",
        "Message ID",
        "Username",
        "message",
        "category",
        "language",
        "lang_confidence",
    ]

    # Metrics
    m = defaultdict(int)

    reader = pd.read_csv(
        in_csv,
        chunksize=chunk_size,
        usecols=required_cols,
        dtype={
            "Channel ID": "string",
            "Message ID": "string",
            "Username": "string",
            "message": "string",
            "category": "string",
            "language": "string",
            # lang_confidence comes through as numeric or string, and we just pass it along
        },
        encoding="utf-8",
    )

    buffer_rows = []
    for chunk in tqdm(reader, desc=f"Translating (row-by-row): {in_csv.name}"):
        # Normalize dtypes / fill
        chunk["Channel ID"] = chunk["Channel ID"].fillna("").astype("string")
        chunk["Message ID"] = chunk["Message ID"].fillna("").astype("string")
        chunk["Username"] = chunk["Username"].fillna("").astype("string")
        chunk["message"] = chunk["message"].fillna("").astype("string")
        chunk["category"] = chunk["category"].fillna("").astype("string")
        chunk["language"] = chunk["language"].fillna("").astype("string")

        # Walk rows one by one (no batching)
        for row in chunk.itertuples(index=False):
            ch_id = str(row[0])
            msg_id = str(row[1])
            uname = str(row[2])
            text = str(row[3])
            category = str(row[4])
            lang = (str(row[5]) or "").lower()
            conf = row[6]

            if lang == "en":
                translated = text  # pass-through
            elif lang in ("", "unknown"):
                translated = np.nan  # do not attempt translation
            else:
                # Ensure the model for this language is loaded (cached)
                if lang not in _model_cache:
                    _ = load_translation_model_darkgram(lang)

                m["non_english_rows"] += 1
                translated = translate_one_darkgram(text, lang, max_length=max_length)
                if translated is None:
                    m["translation_failures"] += 1
                else:
                    m["translated_rows"] += 1

            buffer_rows.append(
                {
                    "Channel ID": ch_id,
                    "Message ID": msg_id,
                    "Username": uname,
                    "original_message": text,
                    "translated_message": translated if translated is not None else np.nan,
                    "category": category,
                    "language": lang,
                    "lang_confidence": conf,
                }
            )

            # Periodic flush
            if len(buffer_rows) >= save_every:
                pd.DataFrame(buffer_rows, columns=[
                    "Channel ID",
                    "Message ID",
                    "Username",
                    "original_message",
                    "translated_message",
                    "category",
                    "language",
                    "lang_confidence",
                ]).to_csv(
                    out_csv,
                    mode="a",
                    header=not out_csv.exists(),
                    index=False,
                    encoding="utf-8",
                )
                buffer_rows.clear()

    # Final flush
    if buffer_rows:
        pd.DataFrame(buffer_rows, columns=[
            "Channel ID",
            "Message ID",
            "Username",
            "original_message",
            "translated_message",
            "category",
            "language",
            "lang_confidence",
        ]).to_csv(
            out_csv,
            mode="a",
            header=not out_csv.exists(),
            index=False,
            encoding="utf-8",
        )
        buffer_rows.clear()

    # Metrics → JSON
    metrics = {
        "messages_translated": int(m["translated_rows"]),
        "translation_failures": int(m["translation_failures"]),
    }
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"✅ Done.\nCSV → {out_csv}\nMetrics → {metrics_json}")


def validate_darkgram_header(in_csv: Path) -> None:
    """Check that the input CSV header contains all required columns."""
    hdr = pd.read_csv(in_csv, nrows=0, encoding="utf-8").columns.tolist()
    needed = {
        "Channel ID",
        "Message ID",
        "Username",
        "message",
        "category",
        "language",
        "lang_confidence",
    }
    missing = sorted(list(needed - set(hdr)))
    if missing:
        raise SystemExit(f"Input CSV is missing required columns: {missing}")
