# BERTopic hyperparameter grids and tuning phases

This table lists the values of the BERTopic hyperparameters and the phase in
which each one is tuned (paper Section 3.2.2); the tuning driver is
`scripts/20_tune_bertopic.py`, configured by `configs/bertopic/phase1.yaml`
through `phase3.yaml`.

| BERTopic step | Hyperparameter | Values | Tuning phase |
|---|---|---|---|
| Dimensionality reduction: UMAP | `n_neighbors` | {20, 30, 50, 75, 100, 200} | P1 |
| | `min_dist` | 0.1 | Fixed |
| | `n_components` | {5, 7, 10, 15, 30} | P2 |
| | `metric` | cosine | Fixed |
| Clustering: HDBSCAN | `min_cluster_size` | {50, 100, 150, 200, 250, 300, 350, 400, 450, 500} | P1&dagger; |
| | `min_samples` | {5, 15, 20} | P1 |
| | `cluster_selection_method` | eom | Fixed |
| | `cluster_selection_epsilon` | {0.0, 0.1} | P2 |
| Topic representation: KeyBERTInspired | `ngram_range` | (1,1) and (1,2) | P3 |
| | `min_df` | 0.001 | Fixed |
| | `max_df` | 0.9 | Fixed |
| | `stop_words` | English default + Telegram-specific additions | Fixed |

**Note.** &dagger; In addition to the ten fixed values listed, the
`min_cluster_size` grid includes three corpus-scaled candidates (⌊|D|/60⌋,
⌊|D|/80⌋, ⌊|D|/100⌋, where |D| is the number of micro-batch documents) and the
value 1000, for 14 candidates in total. Phase 1 therefore evaluates
6 × 14 × 3 = 252 configurations. The same `min_cluster_size` grid is
re-evaluated on the final 2M-message fit, with all other hyperparameters fixed
at the top-scoring configuration.
