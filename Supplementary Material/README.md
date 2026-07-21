# Supplementary Material

This folder gathers the assets that the paper releases with the public
implementation.

| Asset | Paper reference |
|---|---|
| [`algorithm.md`](algorithm.md) — pseudocode of the TACTIC per-channel classification procedure, end to end | Section 3.2 |
| [`tables/bertopic_hyperparameter_grids.md`](tables/bertopic_hyperparameter_grids.md) — full BERTopic hyperparameter grids with the tuning phase of each value | Section 3.2.2 |
| [`tables/single_metric_selections.md`](tables/single_metric_selections.md) — what each raw metric selects on its own over the 252 Phase-1 configurations | Section 5.1 |
| [`tables/lda_sweep.md`](tables/lda_sweep.md) — the full LDA sweep (8 models) against the selected BERTopic configuration | Section 5.2 |
| [`tables/topic_drift_movers.md`](tables/topic_drift_movers.md) — the ten topics whose corpus share moves most between TGDataset (2022) and TLiveUGR (2025) | Section 5.5 |
| [`figures/`](figures/) — the complete set of per-category label-distribution figures (19 PNGs) | Section 5.3 |

## Figures

The figures show the distribution of taxonomy labels assigned by TACTIC for
each dataset category; the manuscript keeps one representative per dataset
(`darkgram_pirated-media.png`, `tgdataset_us-news.png`,
`tliveugr_distribution.png`), and the complete set is:

- **DarkGram** (5): `darkgram_blackhat-resources.png`, `darkgram_credential-compromise.png`,
  `darkgram_pirated-media.png`, `darkgram_pirated-software.png`,
  `darkgram_social-media-manipulation.png`
- **TGDataset** (13): `tgdataset_carding.png`, `tgdataset_covid.png`, `tgdataset_crypto.png`,
  `tgdataset_entertainment.png`, `tgdataset_extremists-and-radicals.png`,
  `tgdataset_indian-edu.png`, `tgdataset_porn.png`, `tgdataset_religion.png`,
  `tgdataset_social.png`, `tgdataset_software.png`, `tgdataset_us-news.png`,
  `tgdataset_videogame-modding.png`, `tgdataset_world-news.png`
- **TLiveUGR** (1): `tliveugr_distribution.png` (pooled distribution)
