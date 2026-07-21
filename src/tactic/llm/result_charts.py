"""Result reporting: category counts and summary charts.

Reads the bucketed classifier outputs (``results_<bucket>.csv``) and produces:
- ``overall_summary.csv``: channel counts per bucket, the total, and the
  detection rate (cybercrime alerts / total channels).
- ``category_counts.csv``: the number of channels assigned to each cybercrime
  category.
- ``ground_truth_breakdown.csv`` (only when a ground-truth column is present):
  bucket counts per ground-truth category.
- bar charts (PNG) for the category counts and the bucket summary.

Every channel is counted in exactly one bucket. Parse errors, API errors and
off-taxonomy outputs are reported as their own buckets, not folded into the
detected categories.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, must precede pyplot import

import matplotlib.pyplot as plt
import pandas as pd

from tactic.common.config import cfg_get

_BUCKETS = ["cybercrime", "safe", "unrecognized", "api_error", "error_parsing_response"]
_BUCKET_LABELS = {
    "cybercrime": "CYBERCRIME ALERT",
    "safe": "SAFE CHANNEL",
    "unrecognized": "UNRECOGNIZED",
    "api_error": "API ERROR",
    "error_parsing_response": "ERROR PARSING RESPONSE",
}
BAR_BLUE = "#9fd3fb"
BAR_RED = "#f8b4b4"


def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-") or "unknown"


def _read_bucket(results_dir: Path, bucket: str) -> pd.DataFrame:
    path = results_dir / f"results_{bucket}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def run(cfg: dict) -> None:
    results_dir = Path(cfg_get(cfg, "paths.results_dir"))
    out_dir = Path(cfg_get(cfg, "paths.output_dir", str(results_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)
    label = cfg_get(cfg, "params.dataset_label", results_dir.name)
    gt_column = cfg_get(cfg, "params.ground_truth_column", None)

    frames = {b: _read_bucket(results_dir, b) for b in _BUCKETS}
    counts = {b: len(frames[b]) for b in _BUCKETS}
    total = sum(counts.values())
    if total == 0:
        raise SystemExit(f"No result rows found under {results_dir}")
    detection_rate = counts["cybercrime"] / total

    # --- overall summary ---
    summary_rows = [{"bucket": _BUCKET_LABELS[b], "channels": counts[b]} for b in _BUCKETS]
    summary_rows.append({"bucket": "TOTAL", "channels": total})
    summary_rows.append({"bucket": "DETECTION RATE", "channels": round(detection_rate, 4)})
    pd.DataFrame(summary_rows).to_csv(out_dir / "overall_summary.csv", index=False)

    # --- per-category counts among cybercrime alerts ---
    cyber = frames["cybercrime"]
    if not cyber.empty and "main_category" in cyber.columns:
        cat_counts = (
            cyber["main_category"].value_counts().rename_axis("category").reset_index(name="channels")
        )
    else:
        cat_counts = pd.DataFrame(columns=["category", "channels"])
    cat_counts.to_csv(out_dir / "category_counts.csv", index=False)

    # --- ground-truth breakdown (optional) ---
    if gt_column:
        rows = []
        for b in _BUCKETS:
            df = frames[b]
            if df.empty or gt_column not in df.columns:
                continue
            for gt_val, n in df[gt_column].fillna("<none>").value_counts().items():
                rows.append({"ground_truth": gt_val, "bucket": _BUCKET_LABELS[b], "channels": int(n)})
        if rows:
            (pd.DataFrame(rows)
             .pivot_table(index="ground_truth", columns="bucket", values="channels", fill_value=0)
             .to_csv(out_dir / "ground_truth_breakdown.csv"))

    _plot_summary(counts, total, detection_rate, label, out_dir / "summary.png")
    _plot_categories(cat_counts, label, out_dir / "category_counts.png")
    print(f"Reporting complete: {total} channels, detection rate {detection_rate:.4f}. Written to {out_dir}")


def _plot_summary(counts, total, detection_rate, label, path):
    labels = [_BUCKET_LABELS[b] for b in _BUCKETS]
    values = [counts[b] for b in _BUCKETS]
    colors = [BAR_RED if b == "cybercrime" else BAR_BLUE for b in _BUCKETS]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Channels")
    ax.set_title(f"{label}: classification outcomes "
                 f"(n={total}, detection rate {detection_rate:.1%})")
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_categories(cat_counts, label, path):
    if cat_counts.empty:
        return
    df = cat_counts.sort_values("channels", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(df))))
    ax.barh(df["category"], df["channels"], color=BAR_RED)
    ax.set_xlabel("Channels")
    ax.set_title(f"{label}: detected cybercrime categories")
    for i, v in enumerate(df["channels"]):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
