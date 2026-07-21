# Single-metric selections over the Phase-1 coarse search

This table shows what each raw metric selects when used on its own over the
252 configurations of the Phase-1 coarse search (paper Section 5.1): each row
reports the configuration that optimizes the metric named in its first column,
together with that configuration's other properties, and bold marks the values
that make each single-metric choice unusable in practice; the analysis is
reproducible with `analysis/composite_score_robustness.py`.

| Selection rule | Coherence | Silhouette | Outlier rate | Topics |
|---|---:|---:|---:|---:|
| Maximize coherence | 0.765 | 0.106 | **0.486** | 51 |
| Maximize silhouette | 0.709 | 0.112 | **0.536** | **12** |
| Minimize outlier rate | 0.547 | 0.012 | 0.000 | **2** |
| Composite (this work) | 0.738 | 0.087 | 0.410 | 45 |
