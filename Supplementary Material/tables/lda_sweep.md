# LDA sweep against the selected BERTopic configuration

This table reports the topic coherence (c_v) of the selected BERTopic
configuration and of LDA over a sweep of topic counts and descriptor
vocabularies (paper Section 5.2): all models are scored on the same 48,700
aggregated documents under the same protocol, with only the topic model
differing; the sweep is reproducible with `analysis/lda_baseline.py`.

| Model | Descriptor vocabulary | Topics (K) | Coherence (c_v) |
|---|---|---:|---:|
| **BERTopic** (selected) | 365,253 terms | 45 (emergent) | **0.738** |
| LDA | 13,699 (min. doc. freq. 50) | 10 | 0.424 |
| LDA | 13,699 | 20 | 0.416 |
| LDA | 13,699 | 45 | 0.418 |
| LDA | 13,699 | 80 | 0.416 |
| LDA | 365,253 (as BERTopic) | 10 | 0.355 |
| LDA | 365,253 | 20 | 0.346 |
| LDA | 365,253 | 45 | 0.386 |
| LDA | 365,253 | 80 | 0.381 |
