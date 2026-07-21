"""Per-channel micro-batching: the document unit for topic modeling.

Messages are grouped per channel into consecutive micro-batches of
``AGG_SIZE`` messages. A trailing remainder batch is kept if it contains at
least ``REMAINDER_MIN_SIZE`` messages, or if it is the channel's only batch.
Channels are iterated in sorted order so that every stage that rebuilds the
documents produces them in the same sequence.

The iteration order follows the dtype of the ``Channel ID`` column in the
input frame (numeric vs. string sorting). Stages that share cached artifacts
(document embeddings, ``topics.pkl``) must load the column with the same dtype.
"""

from __future__ import annotations

import pandas as pd

AGG_SIZE = 10           # messages per micro-batch
REMAINDER_MIN_SIZE = 2  # keep a trailing remainder batch only at this size or larger


def coalesce_processed_message(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``processed_message`` and drop rows where it is null.

    English messages use the original text. Non-English messages use the
    translation when available and are dropped otherwise.
    """
    df = df.copy()
    df['processed_message'] = df.apply(
        lambda row: row['original_message'] if row['language'] == 'en'
        else row['translated_message'] if pd.notnull(row['translated_message'])
        else None,
        axis=1
    )
    df = df[df['processed_message'].notnull()].reset_index(drop=True)
    return df


def build_grouped_docs(df: pd.DataFrame, agg_size: int = AGG_SIZE) -> list:
    """Build the flat list of aggregated documents used for fit/transform/metrics.

    Args:
        df: DataFrame with "Channel ID" and "processed_message" columns.
        agg_size: messages per micro-batch.

    Returns:
        List of space-joined document strings, one per micro-batch, in sorted
        channel order.
    """
    grouped_docs = []
    for _, group in df.groupby("Channel ID", sort=True):
        messages = group["processed_message"].tolist()
        agg_idx = 0
        for i in range(0, len(messages), agg_size):
            chunk = messages[i:i + agg_size]
            if len(chunk) >= REMAINDER_MIN_SIZE or agg_idx == 0:
                grouped_docs.append(" ".join(chunk))
                agg_idx += 1
    return grouped_docs


def build_aggregated_docs_df(df: pd.DataFrame, agg_size: int = AGG_SIZE) -> pd.DataFrame:
    """Build the per-chunk DataFrame used by the topic-assignment stage.

    Returns a DataFrame with one row per micro-batch:
      columns: Channel ID, agg_index (per-channel batch index), doc_text

    The batching rule and iteration order are identical to
    :func:`build_grouped_docs`, so the rows align positionally with a
    ``topics.pkl`` produced from that document list.
    """
    agg_rows = []
    for ch_id, g in df.groupby("Channel ID", sort=True):
        msgs = g["processed_message"].tolist()
        agg_idx = 0
        for i in range(0, len(msgs), agg_size):
            chunk = msgs[i:i + agg_size]
            if len(chunk) >= REMAINDER_MIN_SIZE or agg_idx == 0:
                agg_rows.append({
                    "Channel ID": str(ch_id),
                    "agg_index": agg_idx,
                    "doc_text": " ".join(chunk),
                })
                agg_idx += 1

    return pd.DataFrame.from_records(agg_rows, columns=["Channel ID", "agg_index", "doc_text"])
