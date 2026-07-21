"""TLiveUGR ingestion: stream per-channel bot JSONs into per-file message CSVs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import ijson                   # streaming JSON parser
import pandas as pd
from tqdm import tqdm


def derive_username(canal: dict) -> str:
    """Return <username> from 'https://t.me/<username>' or '' if missing."""
    url = (canal or {}).get("url") or ""
    if not url:
        return ""
    # Drop a trailing slash if present, then take the last path segment
    return url.rstrip("/").rsplit("/", 1)[-1]


def count_total_messages(json_path: Path) -> int:
    """Count message records in a bot JSON file: top-level {canal, mensajes[]}."""
    total_messages = 0

    with json_path.open("r", encoding="utf-8") as f:
        try:
            # Stream list items for scalability
            for _ in ijson.items(f, "mensajes.item"):
                total_messages += 1
        except Exception:
            # Fallback: load once (e.g., if ijson path differs)
            f.seek(0)
            data = json.load(f)
            mensajes = data.get("mensajes") or []
            total_messages = len(mensajes) if isinstance(mensajes, list) else 0

    return total_messages


def extract_json(
    json_path: Path,
    out_csv: Path,
    global_metrics: Dict[str, Any],
    channel_stats: Dict[str, Dict[str, Any]],
    flush_size: int = 100_000,
) -> None:
    """Stream a bot JSON file ({canal, mensajes[]}) and write non-empty texts to CSV.

    Emits rows: ["Channel ID", "Message ID", "Username", "message"].

    If the ijson streaming pass fails mid-iteration, iteration falls back to the
    fully-loaded `mensajes` list, resuming after the already-processed prefix so
    no message is double-processed.
    """
    if flush_size <= 0:
        flush_size = 100_000  # sane fallback

    stage_start = time.time()

    batch_rows: List[Dict[str, str]] = []
    channels_with_rows: set = set()  # here: either 0 or 1 channel per file
    total_messages = count_total_messages(json_path)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():  # avoid duplicates on re-runs
        out_csv.unlink()

    # Read the channel header (`canal`). This loads the file once and also
    # provides the fallback `mensajes` list for the streaming pass below.
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    canal = data.get("canal") or {}
    id_raw = canal.get("id", None)
    channel_id = "" if (id_raw is None or (isinstance(id_raw, str) and id_raw.strip() == "")) else str(id_raw).strip()
    if channel_id == "":
        global_metrics["initial_unique_channels"] += 1
        global_metrics["channels_without_id"] += 1
        if not total_messages:
            global_metrics["initial_channels_without_text_messages"] += 1
        global_metrics["initial_deletion_runtime_sec"] += round(time.time() - stage_start, 2)
        return

    username = derive_username(canal)

    # Init per-channel metrics (keyed by Channel ID string)
    ch = channel_stats.setdefault(channel_id, {
        "username": "",
        "source_json": "",
        "total_original_messages": 0,
        "initial_empty_or_nan_messages": 0,
    })

    global_metrics["initial_unique_channels"] += 1

    if not ch["username"] and username is not None:
        ch["username"] = str(username)
    if not ch["source_json"]:
        ch["source_json"] = json_path.name

    # Number of messages the streaming pass completed. The fallback resumes here.
    processed_count = 0

    # Stream mensajes to avoid loading huge arrays
    with json_path.open("r", encoding="utf-8") as f, tqdm(
        total=total_messages,
        desc=f"Processing {json_path.name}",
        dynamic_ncols=True,
        mininterval=5,
        leave=False
    ) as pbar:
        try:
            mensajes_iter = ijson.items(f, "mensajes.item")
            for msg_obj in mensajes_iter:
                msg_id = msg_obj.get("mensaje_id")
                texto = (((msg_obj.get("mensaje") or {}).get("texto")) or "").strip()

                # Global/per-channel counts (pre-cleaning)
                global_metrics["total_original_messages"] += 1
                ch["total_original_messages"] = ch.get("total_original_messages", 0) + 1
                if not texto:
                    global_metrics["initial_empty_or_nan_messages"] += 1
                    ch["initial_empty_or_nan_messages"] += 1
                else:
                    batch_rows.append({
                        "Channel ID": channel_id,
                        "Message ID": str(msg_id) if msg_id is not None else "",
                        "Username": username,
                        "message": texto,
                    })
                    channels_with_rows.add(channel_id)

                processed_count += 1
                pbar.update(1)
                if len(batch_rows) >= flush_size:
                    pd.DataFrame(batch_rows).to_csv(
                        out_csv, mode="a", index=False,
                        header=not out_csv.exists(), encoding="utf-8",
                    )
                    batch_rows.clear()

        except Exception:
            # Fallback if the ijson path fails: iterate the loaded list, resuming
            # after the messages the streaming pass already handled.
            mensajes = (data.get("mensajes") or [])[processed_count:]
            for msg_obj in mensajes:
                msg_id = msg_obj.get("mensaje_id")
                texto = (((msg_obj.get("mensaje") or {}).get("texto")) or "").strip()

                global_metrics["total_original_messages"] += 1
                ch["total_original_messages"] = ch.get("total_original_messages", 0) + 1
                if not texto:
                    global_metrics["initial_empty_or_nan_messages"] += 1
                    ch["initial_empty_or_nan_messages"] += 1
                else:
                    batch_rows.append({
                        "Channel ID": channel_id,
                        "Message ID": str(msg_id) if msg_id is not None else "",
                        "Username": username,
                        "message": texto,
                    })
                    channels_with_rows.add(channel_id)

                pbar.update(1)
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

    # One JSON = one channel → increment these once (if any non-empty message was written)
    if channels_with_rows:
        global_metrics["unique_channels_after_initial_filter"] += 1
    else:
        # If the mensajes array is missing/empty, mark the channel as having no messages
        if not total_messages:
            global_metrics["initial_channels_without_text_messages"] += 1

    global_metrics["initial_deletion_runtime_sec"] += round(time.time() - stage_start, 2)
