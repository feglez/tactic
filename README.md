# TACTIC: Topic-guided AI for Cybercriminal Telegram Identification and Classification

TACTIC identifies cybercriminal activity in Telegram channels by coupling a
topic model, used as a semantic featurizer, with an LLM classifier that takes
the final decision:

1. **Topic modeling**: a global **BERTopic** model
   (all-MiniLM-L6-v2 → UMAP → HDBSCAN → KeyBERTInspired) is fitted over
   per-channel *micro-batches* of 10 consecutive messages, then frozen and
   applied to each evaluation dataset.
2. **LLM classification**: a large language model (**GPT-OSS 20B** served by
   Ollama) receives each channel's salient topics, a token-budgeted random
   sample of its posts, and light metadata, and emits a one-line rationale
   followed by a verdict: `CYBERCRIME ALERT: ("<category>")` against a
   55-category taxonomy, or `SAFE CHANNEL`.

The pipeline is evaluated on three corpora:

| Dataset | Role |
|---|---|
| TGDataset | Large public corpus: BERTopic tuning, validation, 2M final fit, and a category-labeled classification audit |
| DarkGram | 324 curated cybercriminal channels, used as a held-out positive benchmark |
| TLiveUGR | 459 channels / 1.15M messages collected live (June–Sept 2025), used as a held-out recent snapshot |

## Repository layout

```
.
├── run_pipeline.sh          # end-to-end driver (dependency-ordered)
├── run_retry_recovery.sh    # drives the retry stage over every results dir, then re-reports
├── pyproject.toml           # pinned dependencies (deterministic environment)
├── configs/                 # one YAML per stage + global paths.yaml
│   ├── paths.yaml           #   global path variables (TACTIC_<VAR> env overrides)
│   ├── preprocessing/       #   Phase 1 configs
│   ├── bertopic/            #   Phase 2 configs (tuning, validation, final fit, apply)
│   └── llm/                 #   Phase 3 configs (assignments, consolidation, classify, charts)
├── scripts/                 # numbered CLI entry points (00 resolver, 0x/1x preprocess,
│                            # 2x BERTopic, 3x LLM incl. 34 retry-unrecognized)
├── src/tactic/
│   ├── common/              # config loader, logging, IO helpers
│   ├── preprocessing/       # ingestion, cleaning, language ID, translation, sampling
│   ├── topic_modeling/      # micro-batching, embeddings, tuning, metrics, scoring, apply
│   └── llm/                 # assignments, consolidation, taxonomy, prompts, budgeting,
│                            # classifier, reporting; prompt_assets/ holds the verbatim
│                            # system prompts, user-message templates, and few-shot exemplars
├── modelfiles/              # Ollama Modelfiles for the two custom GPT-OSS builds
├── collector/               # the live Telegram monitoring client used to gather TLiveUGR
├── data/
│   ├── tgdataset_labels/    # TGDataset channel→category and channel→language maps
│   └── tliveugr_channels.csv  # identifiers of the 459 channels collected for TLiveUGR
├── analysis/                # companion analysis scripts (composite-score robustness,
│                            # LDA baseline, topic drift) — see analysis/README.md
└── Supplementary Material/  # paper supplementary assets: per-channel pseudocode,
                             # hyperparameter grids, result tables, 19 per-category figures
```

## Installation

The project targets Python 3.12, and the environment is managed with
[uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Additional runtime assets:

- **fastText language-ID model** `lid.176.bin` at `${data_root}/models/lid.176.bin`
  (download: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin).
- A **GPU** is strongly recommended for the embedding, translation, and final-fit stages.
- **Ollama** serving the two custom GPT-OSS models built from `modelfiles/`
  (`ollama create ... -f modelfiles/custom_modelfile`, as described in
  `modelfiles/README.txt`).

## Data

Datasets are not distributed with the code: place raw inputs under `./data`,
or point the `TACTIC_<VAR>` environment variables at existing copies, as
documented in `configs/paths.yaml`:

- `data/TGDataset_raw/`: the 121 raw TGDataset JSON files, available from the
  dataset's public release (Zenodo record 7640712).
- `data/tgdataset_labels/`: TGDataset channel→category and channel→language
  maps (included).
- `data/darkgram_raw/`: the DarkGram category CSVs, as released by the
  DarkGram authors (see the paper's references).
- `data/tliveugr/final_logs/`: TLiveUGR collection logs (gathered with
  `collector/`).

TLiveUGR message content is not redistributed, in line with the Telegram Terms
of Service and applicable data-protection regulations. What is released is the
monitoring client (`collector/`) and the list of collected channel identifiers
(`data/tliveugr_channels.csv`), so the collection can be repeated over the same
channels under the same terms; since channels are banned, renamed, or pruned
over time, a re-collection approximates the studied snapshot instead of
replicating it exactly.

## Running

```bash
uv sync
./run_pipeline.sh            # full pipeline in dependency order
DRY_RUN=1 ./run_pipeline.sh  # print the command sequence without executing
```

Any stage can also be run directly, e.g.:

```bash
uv run python scripts/32_classify_channels.py --config configs/llm/classify_darkgram.yaml
```

Every stage reads a YAML config, and every config path can be overridden on the
CLI with `--override key.path=value`. Outputs are written under `${output_root}`
(default `./outputs`). External data/model directories are treated as read-only.

### No-topics ablation

The classification stage can be re-run with the topic section removed from
every prompt while pinning each channel's message budget to a reference run,
which isolates the contribution of the topic cues:

```bash
uv run python scripts/32_classify_channels.py --config configs/llm/classify_darkgram.yaml \
    --override params.omit_topics=true \
    --override params.budget_pin_log=outputs/llm/results/darkgram/classification_log.jsonl \
    --override paths.output_dir=outputs/llm/results/darkgram_notopics
```

### Publishing the winning model

The tuning and final-fit stages write candidate models under
`${output_root}/bertopic_output/`, while the apply and assignment stages read
the frozen model from `${models_root}/bertopic_final_<uid>/` (`model_dir` in
`configs/bertopic/apply_*.yaml` and `configs/llm/assignments_*.yaml`). The
shipped configs expect `bertopic_final_65ea4f`, the published run's winner.
After the final fit, pick the top scorer from the composite scoring output
(`scripts/22_score_metrics.py`), copy its model directory to
`${models_root}/`, and update `model_dir` in those configs if your winner's
uid differs. `run_pipeline.sh` stops before the apply stage if no published
model is found.

### Classification results and the retry stage

The published detection results come from Phase 3: `scripts/32_classify_channels.py`
writes bucketed results and an incremental `classification_log.jsonl`, and
`scripts/33_generate_result_charts.py` produces the per-category counts, the
summary, and the charts (parse/API/off-taxonomy outputs are reported explicitly
instead of being reallocated into the other buckets).

A small proportion of channels do not yield a response satisfying the output
contract on a first pass, and land in the `unrecognized` bucket.
`scripts/34_retry_unrecognized.py` re-issues those requests: the saved system
prompt and the channel's `final_prompt.txt` are replayed verbatim, so the prompt
and `num_ctx` never change and only the decoding varies. It retries until the
response parses to a valid verdict or a bounded attempt limit is reached;
channels that never yield one stay `unrecognized` and are counted as such. No
verdict is ever written by hand. `run_retry_recovery.sh` drives this across every
results directory and then regenerates the stage-33 reports.

**The published detection numbers include channels recovered by this retry pass.**
Because a local LLM is not bit-reproducible even at temperature 0 (see below), a
single classification pass over a corpus will leave a different set of channels
unrecognized than ours did; running the retry stage is what makes a re-run
comparable. Running with a larger `num_ctx` also removes most of these cases, at
the cost of no longer matching the published configuration.

## Reproducibility notes

- The dependency stack is pinned in `pyproject.toml` for deterministic runs,
  and `uv sync` reproduces the locked resolution.
- Sampling and message selection are seeded. A live GPT-OSS generation is not
  bit-deterministic across hardware/engine versions even at temperature 0, so
  classification counts can vary slightly between serving environments. The
  frozen classification log defines the reference for a given run.
- BERTopic training metrics can vary slightly across library versions. The
  frozen final model, once produced, fixes all downstream results.

## Supplementary material and analyses

- [`Supplementary Material/`](Supplementary%20Material/) holds the assets the
  paper releases with the implementation: the end-to-end per-channel
  classification pseudocode, the full BERTopic hyperparameter grids, the
  single-metric selection and LDA-sweep tables, the most-drifted-topics table,
  and the complete set of 19 per-category result figures.
- [`analysis/`](analysis/) holds the companion scripts behind the
  composite-score robustness, LDA baseline, and topic-drift analyses.

## Citation

The reference to the paper will be added upon publication.

## License

The code is released under the MIT license (see [LICENSE](LICENSE)).
