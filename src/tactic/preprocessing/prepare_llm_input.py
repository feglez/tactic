"""Build the LLM input tables: coalesce original/translated text into a single
`processed_message` column and attach per-dataset metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def prepare_llm_input(
    dataset_csv: Path,
    output_csv: Path,
    variant: str,
    chid_to_title_csv: Path | None = None,
) -> None:
    """Coalesce original/translated text into `processed_message` and save.

    Args:
        dataset_csv: For "tgdataset": final_dataset.csv (translation output).
                     For "darkgram": darkgram_translated_final.csv.
        output_csv: Destination CSV (tgdataset_llm_input.csv /
                    darkgram_llm_input.csv).
        variant: "tgdataset" (joins username/title metadata) or
                 "darkgram" (keeps the ground-truth `category` column).
        chid_to_title_csv: chID_to_title.csv, required for "tgdataset".
    """
    if variant not in ("tgdataset", "darkgram"):
        raise ValueError(f"variant must be 'tgdataset' or 'darkgram', got {variant!r}")

    # --- Load dataset ---
    print("🔹 Loading dataset...")
    # The whole CSV is loaded into memory at once and coalesced with a
    # row-wise apply. Channel ID is read as a string so the written file
    # preserves the exact key text that the downstream string-dtype stages
    # (apply / assignments / classify) group and sort on.
    df = pd.read_csv(dataset_csv, dtype={"Channel ID": "string"})

    # --- Compute processed_message ---
    df["processed_message"] = df.apply(
        lambda row: row["original_message"] if row["language"] == "en"
        else row["translated_message"] if pd.notnull(row["translated_message"])
        else None,
        axis=1
    )

    # --- Remove null processed messages ---
    df = df[df["processed_message"].notnull()].reset_index(drop=True)

    if variant == "tgdataset":
        if chid_to_title_csv is None:
            raise ValueError("variant 'tgdataset' requires chid_to_title_csv")

        # --- Load channel metadata ---
        print("🔹 Loading channel metadata...")
        channel_meta = pd.read_csv(chid_to_title_csv)

        # Ensure column names are consistent
        channel_meta.columns = [col.strip().lower() for col in channel_meta.columns]
        channel_meta = channel_meta.rename(columns={
            "channel_id": "Channel ID",
            "username": "Username",
            "title": "Title"
        })
        channel_meta["Channel ID"] = channel_meta["Channel ID"].astype("string")

        # --- Merge channel metadata ---
        # LEFT merge: channels absent from chID_to_title.csv keep NaN
        # Username/Title and stay in the output. The downstream classifier
        # handles those NaNs.
        df = df.merge(channel_meta[["Channel ID", "Username", "Title"]], on="Channel ID", how="left")

        # --- Reorder columns ---
        cols = ["Channel ID", "Username", "Title", "original_message", "translated_message", "language", "processed_message"]
        df = df[cols]
    else:
        # --- Reorder columns and drop some ---
        cols = ["Channel ID", "Username", "original_message", "translated_message",
                "language", "processed_message", "category"]
        df = df[cols]

    # --- Save final dataset ---
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved to {output_csv} with {len(df)} messages.")
