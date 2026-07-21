"""Topic-distribution drift between two assignment runs (paper Section 5.5).

Compares the corpus-level topic distributions (chunk-count shares, outlier -1
included) of two channel_topic_assignments_unfiltered.csv files: Jensen-Shannon
divergence, outlier-share delta, and the top-10 topics by absolute share delta,
with a horizontal bar chart of those deltas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TACTIC_SRC = str(Path(__file__).resolve().parents[1] / "src")
if TACTIC_SRC not in sys.path:
    sys.path.insert(0, TACTIC_SRC)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

# Diverging pair for signed deltas + chart chrome (light surface).
COLOR_POSITIVE = "#2a78d6"
COLOR_NEGATIVE = "#e34948"
COLOR_SURFACE = "#fcfcfb"
COLOR_GRID = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_INK = "#0b0b0b"
COLOR_INK_MUTED = "#52514e"


def corpus_distribution(path: str) -> pd.Series:
    """Topic -> share of total chunks (sum of chunk_count per topic / grand total)."""
    df = pd.read_csv(path)
    for col in ("topic", "chunk_count"):
        if col not in df.columns:
            sys.exit(f"ERROR: {path} has no '{col}' column (columns: {list(df.columns)}).")
    counts = df.groupby("topic")["chunk_count"].sum()
    total = counts.sum()
    if total <= 0:
        sys.exit(f"ERROR: {path} has zero total chunk_count.")
    return counts / total


def load_topic_names(path: str) -> dict:
    info = pd.read_csv(path)
    if "Topic" not in info.columns or "Name" not in info.columns:
        sys.exit(f"ERROR: {path} needs 'Topic' and 'Name' columns "
                 f"(columns: {list(info.columns)}).")
    return dict(zip(info["Topic"].astype(int), info["Name"].astype(str)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jensen-Shannon drift between two unfiltered topic-assignment runs."
    )
    parser.add_argument("--assignments-a", required=True,
                        help="channel_topic_assignments_unfiltered.csv (side A).")
    parser.add_argument("--assignments-b", required=True,
                        help="channel_topic_assignments_unfiltered.csv (side B).")
    parser.add_argument("--label-a", required=True, help="Display label for side A.")
    parser.add_argument("--label-b", required=True, help="Display label for side B.")
    parser.add_argument("--topic-info", required=True,
                        help="extracted_topics.csv with Topic/Name columns.")
    parser.add_argument("--out-prefix", required=True,
                        help="Output prefix; writes <prefix>.json, <prefix>_topic_shares.csv, "
                             "<prefix>_top10_deltas.png.")
    args = parser.parse_args()

    dist_a = corpus_distribution(args.assignments_a)
    dist_b = corpus_distribution(args.assignments_b)
    names = load_topic_names(args.topic_info)

    all_topics = sorted(set(dist_a.index).union(dist_b.index))
    p = dist_a.reindex(all_topics, fill_value=0.0).to_numpy(dtype=float)
    q = dist_b.reindex(all_topics, fill_value=0.0).to_numpy(dtype=float)

    js_distance = float(jensenshannon(p, q, base=2))
    js_divergence = js_distance ** 2

    share_a = dict(zip(all_topics, p))
    share_b = dict(zip(all_topics, q))
    outlier_a = float(share_a.get(-1, 0.0))
    outlier_b = float(share_b.get(-1, 0.0))

    table = pd.DataFrame({
        "topic": all_topics,
        "name": [names.get(t, f"Topic {t}") for t in all_topics],
        f"share_{args.label_a}": p,
        f"share_{args.label_b}": q,
    })
    table["delta"] = table[f"share_{args.label_b}"] - table[f"share_{args.label_a}"]
    table["abs_delta"] = table["delta"].abs()
    table = table.sort_values("abs_delta", ascending=False).reset_index(drop=True)

    top10 = table.head(10).copy()

    # --- outputs ---
    csv_path = f"{args.out_prefix}_topic_shares.csv"
    table.to_csv(csv_path, index=False)

    result = {
        "assignments_a": args.assignments_a,
        "assignments_b": args.assignments_b,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "n_topics_union": len(all_topics),
        "jensen_shannon_distance_base2": js_distance,
        "jensen_shannon_divergence_base2": js_divergence,
        "outlier_share_a": outlier_a,
        "outlier_share_b": outlier_b,
        "outlier_share_delta_b_minus_a": outlier_b - outlier_a,
        "top10_by_abs_share_delta": [
            {
                "topic": int(r["topic"]),
                "name": r["name"],
                "share_a": float(r[f"share_{args.label_a}"]),
                "share_b": float(r[f"share_{args.label_b}"]),
                "delta_b_minus_a": float(r["delta"]),
            }
            for _, r in top10.iterrows()
        ],
    }
    json_path = f"{args.out_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    # --- chart: top-10 |delta| topics, horizontal bars, diverging color by sign ---
    plot_df = top10.sort_values("delta")  # most negative at bottom, most positive on top
    labels = [
        f"{name[:42]}…" if len(name) > 43 else name
        for name in plot_df["name"]
    ]
    deltas_pp = plot_df["delta"].to_numpy() * 100.0  # percentage points
    colors = [COLOR_POSITIVE if d >= 0 else COLOR_NEGATIVE for d in deltas_pp]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor(COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    y = np.arange(len(plot_df))
    ax.barh(y, deltas_pp, height=0.62, color=colors, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=COLOR_INK)
    ax.axvline(0, color=COLOR_BASELINE, linewidth=1, zorder=2)
    ax.xaxis.grid(True, color=COLOR_GRID, linewidth=0.75, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_BASELINE)
    ax.tick_params(axis="x", colors=COLOR_INK_MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"Corpus chunk-share delta ({args.label_b} − {args.label_a}), "
                  "percentage points", fontsize=9, color=COLOR_INK_MUTED)
    ax.set_title(f"Top-10 topic share deltas: {args.label_a} → {args.label_b}",
                 fontsize=11, color=COLOR_INK, loc="left")

    # Direct value labels at bar ends.
    span = max(abs(deltas_pp).max(), 1e-9)
    for yi, d in zip(y, deltas_pp):
        offset = span * 0.02
        ax.text(d + (offset if d >= 0 else -offset), yi, f"{d:+.2f}",
                va="center", ha="left" if d >= 0 else "right",
                fontsize=8, color=COLOR_INK_MUTED)
    ax.set_xlim(-span * 1.18, span * 1.18)

    fig.tight_layout()
    png_path = f"{args.out_prefix}_top10_deltas.png"
    fig.savefig(png_path, facecolor=COLOR_SURFACE)
    plt.close(fig)

    print("\n--- Topic drift summary ---")
    print(f"Jensen-Shannon distance (base 2): {js_distance:.4f}")
    print(f"Jensen-Shannon divergence (base 2): {js_divergence:.4f}")
    print(f"Outlier share {args.label_a}: {outlier_a:.4f} | {args.label_b}: {outlier_b:.4f} "
          f"| delta: {outlier_b - outlier_a:+.4f}")
    print("Top-10 topics by |share delta| (b − a, percentage points):")
    for rec in result["top10_by_abs_share_delta"]:
        print(f"  {rec['topic']:>4}  {rec['delta_b_minus_a'] * 100:+7.3f}  {rec['name']}")
    print(f"Saved: {json_path}, {csv_path}, {png_path}")


if __name__ == "__main__":
    main()
