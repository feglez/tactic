# Analysis scripts

This folder holds the companion scripts behind the analyses reported in the
paper's evaluation; they run under the repository's virtualenv, import the
`tactic` package directly from `../src` without requiring installation, and
render their plots headless.

| Script | Paper reference | What it does |
|---|---|---|
| `composite_score_robustness.py` | Section 5.1 | Pareto front over (coherence, silhouette, outlier rate, topic-count deviation) for the Phase-1 configurations, the configuration each raw metric would select on its own, and perturbations of the composite weights. |
| `lda_baseline.py` | Section 5.2 | LDA baseline on the same micro-batched tuning corpus, scored with the pipeline's own c_v protocol; sweeps topic counts and descriptor vocabularies and re-scores the frozen BERTopic winner in the same accumulation pass as a protocol check. |
| `topic_drift.py` | Section 5.5 | Jensen-Shannon drift between the topic distributions of two assignment runs (e.g., TGDataset 2022 vs TLiveUGR 2025), outlier-share delta, and the top-10 topics by absolute share delta. |

## Usage

```bash
uv run python analysis/composite_score_robustness.py \
    --scored-csv <model_metrics_final_scored.csv> --target-topics 45 --out-prefix e5

uv run python analysis/lda_baseline.py \
    --dataset-csv <tuning_dataset.csv> --model-dir <bertopic_final_dir> \
    --out-prefix lda --topics 10,20,45,80 --vocabs bertopic,50

uv run python analysis/topic_drift.py \
    --assignments-a <a_unfiltered.csv> --assignments-b <b_unfiltered.csv> \
    --label-a 2022 --label-b 2025 --topic-info <extracted_topics.csv> --out-prefix drift
```

Inputs are pipeline outputs: the scored tuning-metrics CSV comes from
`scripts/22_score_metrics.py`, the unfiltered assignment CSVs from
`scripts/30_generate_topic_assignments.py`, and the tuning dataset from the
sampling stage; the corresponding result tables are in
[`../Supplementary Material/tables/`](../Supplementary%20Material/tables/).
