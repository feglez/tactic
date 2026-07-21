"""Composite-score robustness analysis (paper Section 5.1).

Given the final scored tuning-metrics CSV, reports (a) the Pareto front over
(coherence, silhouette, outlier rate, |T - T*|) and whether the composite winner
lies on it, (b) the config each single metric would select on its own, and
(c) one-at-a-time +/-25% perturbations of the composite weights and whether the
winner changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TACTIC_SRC = str(Path(__file__).resolve().parents[1] / "src")
if TACTIC_SRC not in sys.path:
    sys.path.insert(0, TACTIC_SRC)

import numpy as np
import pandas as pd

# Composite weights used by the pipeline scoring stage:
#   composite = 0.55*C + 0.20*S_norm - 0.35*O - 0.25*P
#   S_norm = clip((s + 1) / 2, 0, 1);  P = min(|T - T*| / T*, 1)
BASE_WEIGHTS = {
    "coherence": 0.55,
    "silhouette": 0.20,
    "outlier": 0.35,
    "topic_penalty": 0.25,
}


def _pick_column(columns, want: str):
    """Tolerant column detection; returns the matched column name or None."""
    low = {c: c.lower() for c in columns}
    def first(pred):
        for c in columns:
            if pred(low[c]):
                return c
        return None

    if want == "model":
        return (first(lambda s: s == "model_name")
                or first(lambda s: "model" in s)
                or first(lambda s: s in ("name", "config", "run")))
    if want == "coherence":
        return first(lambda s: "coheren" in s)
    if want == "silhouette":
        return (first(lambda s: "silhouette" in s and "norm" not in s)
                or first(lambda s: "silhouette" in s))
    if want == "outlier":
        return first(lambda s: "outlier" in s)
    if want == "num_topics":
        return (first(lambda s: s in ("num_topics", "n_topics", "nr_topics", "topics"))
                or first(lambda s: "topic" in s
                         and "penalty" not in s and "percentage" not in s
                         and ("num" in s or "count" in s or s.startswith("n_"))))
    if want == "composite":
        return (first(lambda s: s == "final_score")
                or first(lambda s: "composite" in s)
                or first(lambda s: s.endswith("score")
                         and "coheren" not in s and "silhouette" not in s))
    return None


def compute_composite(df, cols, weights, target_topics):
    """Recompute the composite from the raw metric columns (pipeline formula,
    pipeline NaN handling: C->0, S_norm->0, O->1, T->0)."""
    c = df[cols["coherence"]].astype(float).fillna(0.0)
    s = df[cols["silhouette"]].astype(float)
    s_norm = ((s + 1.0) / 2.0).clip(0.0, 1.0).fillna(0.0)
    o = df["_outlier_rate"].fillna(1.0)
    t = df[cols["num_topics"]].astype(float).fillna(0.0)
    penalty = ((t - target_topics).abs() / max(target_topics, 1)).clip(0.0, 1.0)
    return (weights["coherence"] * c
            + weights["silhouette"] * s_norm
            - weights["outlier"] * o
            - weights["topic_penalty"] * penalty)


def pareto_front_mask(objectives: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows; all objectives oriented as MAXIMIZE."""
    n = objectives.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominates_i = (
            (objectives >= objectives[i]).all(axis=1)
            & (objectives > objectives[i]).any(axis=1)
        )
        if dominates_i.any():
            mask[i] = False
    return mask


def row_summary(df, idx, cols):
    row = df.loc[idx]
    return {
        "model": str(row[cols["model"]]),
        "coherence": float(row[cols["coherence"]]) if pd.notna(row[cols["coherence"]]) else None,
        "silhouette": float(row[cols["silhouette"]]) if pd.notna(row[cols["silhouette"]]) else None,
        "outlier_rate": float(row["_outlier_rate"]) if pd.notna(row["_outlier_rate"]) else None,
        "num_topics": int(row[cols["num_topics"]]) if pd.notna(row[cols["num_topics"]]) else None,
        "composite_recomputed": float(row["_composite_base"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robustness checks for the composite model-selection score."
    )
    parser.add_argument("--scored-csv", required=True,
                        help="Final scored tuning-metrics CSV.")
    parser.add_argument("--target-topics", type=int, default=45,
                        help="Target topic count T* (default: 45).")
    parser.add_argument("--out-prefix", required=True,
                        help="Output prefix; writes <prefix>.json and <prefix>_summary.md.")
    args = parser.parse_args()

    df = pd.read_csv(args.scored_csv)
    cols = {want: _pick_column(df.columns, want)
            for want in ("model", "coherence", "silhouette", "outlier", "num_topics")}
    composite_col = _pick_column(df.columns, "composite")

    print("Detected column mapping:")
    for want, col in {**cols, "composite (existing)": composite_col}.items():
        print(f"  {want:22s} -> {col}")
    missing = [w for w, c in cols.items() if c is None]
    if missing:
        sys.exit(f"ERROR: could not detect column(s) {missing} in {list(df.columns)}.")

    # Outlier column may be a rate in [0,1] or a percentage in [0,100].
    outlier_raw = df[cols["outlier"]].astype(float)
    if outlier_raw.max(skipna=True) > 1.5:
        print(f"  note: '{cols['outlier']}' looks like a percentage; dividing by 100.")
        df["_outlier_rate"] = outlier_raw / 100.0
    else:
        df["_outlier_rate"] = outlier_raw

    target = args.target_topics
    df["_composite_base"] = compute_composite(df, cols, BASE_WEIGHTS, target)
    winner_idx = df["_composite_base"].idxmax()
    winner = row_summary(df, winner_idx, cols)
    print(f"\nComposite winner (recomputed, T*={target}): {winner['model']} "
          f"(score {winner['composite_recomputed']:.4f})")

    existing_winner = None
    if composite_col is not None:
        existing_idx = df[composite_col].astype(float).idxmax()
        existing_winner = str(df.loc[existing_idx, cols["model"]])
        agree = existing_idx == winner_idx
        print(f"Winner by existing '{composite_col}' column: {existing_winner} "
              f"({'matches' if agree else 'DIFFERS from'} recomputed winner)")

    # --- (a) Pareto front: max C, max S, min O, min |T - T*| ---
    c_f = df[cols["coherence"]].astype(float).fillna(0.0).to_numpy()
    s_f = df[cols["silhouette"]].astype(float).fillna(-1.0).to_numpy()
    o_f = df["_outlier_rate"].fillna(1.0).to_numpy()
    t_f = df[cols["num_topics"]].astype(float).fillna(0.0).to_numpy()
    t_dist = np.abs(t_f - target)
    objectives = np.column_stack([c_f, s_f, -o_f, -t_dist])  # all maximize
    front_mask = pareto_front_mask(objectives)
    front_indices = df.index[front_mask].tolist()
    winner_on_front = bool(winner_idx in front_indices)
    front_rows = [row_summary(df, i, cols) for i in front_indices]
    print(f"\nPareto front: {len(front_indices)} of {len(df)} configs; "
          f"composite winner on front: {winner_on_front}")

    # --- (b) single-metric selections ---
    single_metric = {}
    picks = {
        "max_coherence": pd.Series(c_f, index=df.index).idxmax(),
        "max_silhouette": pd.Series(s_f, index=df.index).idxmax(),
        "min_outlier_rate": pd.Series(o_f, index=df.index).idxmin(),
        "min_topic_distance": pd.Series(t_dist, index=df.index).idxmin(),
    }
    print("\nSingle-metric selections:")
    for criterion, idx in picks.items():
        single_metric[criterion] = {
            "selected": row_summary(df, idx, cols),
            "same_as_composite_winner": bool(idx == winner_idx),
        }
        sel = single_metric[criterion]["selected"]
        print(f"  {criterion:20s} -> {sel['model']}"
              f"{'  (= composite winner)' if idx == winner_idx else ''}")

    # --- (c) one-at-a-time +/-25% weight perturbations ---
    perturbations = []
    print("\nWeight perturbations (one at a time, +/-25%):")
    for weight_name in BASE_WEIGHTS:
        for factor, tag in ((1.25, "+25%"), (0.75, "-25%")):
            weights = dict(BASE_WEIGHTS)
            weights[weight_name] = BASE_WEIGHTS[weight_name] * factor
            scores = compute_composite(df, cols, weights, target)
            p_idx = scores.idxmax()
            changed = bool(p_idx != winner_idx)
            rec = {
                "weight": weight_name,
                "direction": tag,
                "perturbed_value": round(weights[weight_name], 6),
                "winner": str(df.loc[p_idx, cols["model"]]),
                "winner_changed": changed,
                "winner_score": float(scores.loc[p_idx]),
            }
            perturbations.append(rec)
            print(f"  {weight_name:14s} {tag}: winner {rec['winner']}"
                  f"{'  <-- CHANGED' if changed else ''}")
    n_changed = sum(1 for r in perturbations if r["winner_changed"])
    print(f"Winner changed in {n_changed} of {len(perturbations)} perturbations.")

    result = {
        "scored_csv": args.scored_csv,
        "target_topics": target,
        "detected_columns": {**cols, "composite_existing": composite_col},
        "n_configs": int(len(df)),
        "base_weights": BASE_WEIGHTS,
        "composite_winner": winner,
        "existing_score_column_winner": existing_winner,
        "pareto_front": {
            "size": len(front_indices),
            "winner_on_front": winner_on_front,
            "members": front_rows,
        },
        "single_metric_selections": single_metric,
        "weight_perturbations": perturbations,
        "n_perturbations_changing_winner": n_changed,
    }
    json_path = f"{args.out_prefix}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    # --- markdown summary ---
    def _fmt(v, spec=".4f"):
        return format(v, spec) if v is not None else "n/a"

    md = []
    md.append("# Composite-score robustness summary\n")
    md.append(f"- Scored CSV: `{args.scored_csv}`")
    md.append(f"- Configs: {len(df)} | T* = {target}")
    md.append(f"- Composite winner (recomputed): **{winner['model']}** "
              f"(score {winner['composite_recomputed']:.4f})")
    md.append(f"- Pareto front size: {len(front_indices)}; winner on front: "
              f"**{'yes' if winner_on_front else 'no'}**\n")
    md.append("## Single-metric selections\n")
    md.append("| Criterion | Selected config | Coherence | Silhouette | Outlier | Topics | = winner? |")
    md.append("|---|---|---|---|---|---|---|")
    for criterion, entry in single_metric.items():
        s = entry["selected"]
        md.append(f"| {criterion} | {s['model']} | {_fmt(s['coherence'])} | "
                  f"{_fmt(s['silhouette'])} | {_fmt(s['outlier_rate'])} | "
                  f"{_fmt(s['num_topics'], 'd')} | "
                  f"{'yes' if entry['same_as_composite_winner'] else 'no'} |")
    md.append("\n## Weight perturbations (one at a time)\n")
    md.append("| Weight | Direction | Value | Winner | Changed |")
    md.append("|---|---|---|---|---|")
    for r in perturbations:
        md.append(f"| {r['weight']} | {r['direction']} | {r['perturbed_value']} | "
                  f"{r['winner']} | {'YES' if r['winner_changed'] else 'no'} |")
    md.append(f"\nWinner changed in {n_changed}/{len(perturbations)} perturbations.\n")
    md_path = f"{args.out_prefix}_summary.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    print(f"\nSaved: {json_path}, {md_path}")


if __name__ == "__main__":
    main()
