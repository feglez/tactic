#!/usr/bin/env bash
# ============================================================================
# TACTIC end-to-end pipeline driver.
#
# Runs the full pipeline in dependency order: preprocessing -> BERTopic tuning
# and final fit -> frozen-model apply -> LLM classification -> reporting.
#
# Usage:
#   ./run_pipeline.sh              # run everything
#   DRY_RUN=1 ./run_pipeline.sh    # print every command, execute nothing
#   RUN="python" ./run_pipeline.sh # use plain python instead of `uv run python`
#
# Requirements before running (see README.md):
#   - `uv sync` (or an equivalent environment with pyproject's pinned deps)
#   - raw data placed under ./data (or TACTIC_* env overrides pointing at it)
#   - the fastText lid.176.bin model
#   - a running Ollama server with the two custom models built (modelfiles/)
#
# Heavy stages (tuning, the 2M fit, translation, LLM inference) take hours. To
# reuse an artifact you already have, place it at the path the stage writes and
# it will be read instead of recomputed where a stage supports it, or point the
# TACTIC_* variables (paths.yaml) at existing data/models.
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")"

RUN="${RUN:-uv run python}"
DRY_RUN="${DRY_RUN:-0}"

run() {
    echo "+ $*"
    if [[ "$DRY_RUN" != "1" ]]; then
        $RUN "$@"
    fi
}

SPLITS_TUNEVAL=(tuning validation1 validation2 validation3)
DATASETS=(tgdataset darkgram tliveugr)

# -- per-split sampling parameters (total, max_per_channel, seed) -------------
declare -A SPLIT_TOTAL=( [tuning]=500000 [validation1]=500000 [validation2]=500000 [validation3]=500000 [final]=2000000 )
declare -A SPLIT_MAXCH=( [tuning]=1000   [validation1]=1000   [validation2]=1000   [validation3]=1000   [final]=2000 )
declare -A SPLIT_SEED=(  [tuning]=42     [validation1]=42     [validation2]=43     [validation3]=44     [final]=45 )

OUT='${output_root}'

echo "########## Phase 1: Preprocessing ##########"

# --- TGDataset: clean -> count -> sample/translate each split -> titles ---
run scripts/01_preprocess_tgdataset.py  --config configs/preprocessing/tgdataset.yaml
run scripts/02_count_channel_messages.py --config configs/preprocessing/channel_counts.yaml

for split in "${SPLITS_TUNEVAL[@]}" final; do
    run scripts/03_stratified_sample.py --config configs/preprocessing/sampling.yaml \
        --override params.dataset_name="$split" \
        --override params.total_messages="${SPLIT_TOTAL[$split]}" \
        --override params.max_per_channel="${SPLIT_MAXCH[$split]}" \
        --override params.random_seed="${SPLIT_SEED[$split]}"
    run scripts/04_translate_selected.py --config configs/preprocessing/translation_tgdataset.yaml \
        --override params.dataset_name="$split" \
        --override paths.selection_csv="${OUT}/preprocessing/sampling/selected_message_indices_${split}.csv"
done

run scripts/06_parse_channel_titles.py --config configs/preprocessing/channel_titles.yaml

# --- DarkGram: clean -> translate ---
run scripts/07_preprocess_darkgram.py --config configs/preprocessing/darkgram.yaml
run scripts/08_translate_darkgram.py  --config configs/preprocessing/translation_darkgram.yaml

# --- TLiveUGR: clean -> assemble shards ---
run scripts/09_preprocess_tliveugr.py      --config configs/preprocessing/tliveugr.yaml
run scripts/10_assemble_tliveugr_dataset.py --config configs/preprocessing/tliveugr_assembly.yaml

# --- LLM-input preparation (TGDataset + DarkGram) ---
run scripts/11_prepare_llm_input.py --config configs/preprocessing/prepare_llm_input_tgdataset.yaml
run scripts/11_prepare_llm_input.py --config configs/preprocessing/prepare_llm_input_darkgram.yaml

echo "########## Phase 2: BERTopic tuning and final fit ##########"

# --- Three-phase tuning on the tuning sample, with per-phase metrics/scoring ---
run scripts/20_tune_bertopic.py --config configs/bertopic/phase1.yaml
run scripts/21_compute_metrics.py --config configs/bertopic/metrics_tgdataset.yaml \
    --override paths.dataset_csv="${OUT}/preprocessing/translation/tuning_dataset.csv" \
    --override paths.doc_embeddings="${OUT}/bertopic_output/tuning_document_embeddings.pkl" \
    --override paths.run_log="${OUT}/bertopic_output/results_log_stage1.csv" \
    --override paths.metrics_csv="${OUT}/bertopic_output/model_metrics_stage1.csv" \
    --override paths.models_dir="${OUT}/bertopic_output"

run scripts/20_tune_bertopic.py --config configs/bertopic/phase2.yaml
run scripts/21_compute_metrics.py --config configs/bertopic/metrics_tgdataset.yaml \
    --override paths.dataset_csv="${OUT}/preprocessing/translation/tuning_dataset.csv" \
    --override paths.doc_embeddings="${OUT}/bertopic_output/tuning_document_embeddings.pkl" \
    --override paths.run_log="${OUT}/bertopic_output/results_log_stage2.csv" \
    --override paths.metrics_csv="${OUT}/bertopic_output/model_metrics_stage2.csv" \
    --override paths.models_dir="${OUT}/bertopic_output"

run scripts/20_tune_bertopic.py --config configs/bertopic/phase3.yaml
run scripts/21_compute_metrics.py --config configs/bertopic/metrics_tgdataset.yaml \
    --override paths.dataset_csv="${OUT}/preprocessing/translation/tuning_dataset.csv" \
    --override paths.doc_embeddings="${OUT}/bertopic_output/tuning_document_embeddings.pkl" \
    --override paths.run_log="${OUT}/bertopic_output/results_log_stage3.csv" \
    --override paths.metrics_csv="${OUT}/bertopic_output/model_metrics_stage3.csv" \
    --override paths.models_dir="${OUT}/bertopic_output"

# --- Validation: re-fit the winning configuration on three held-out splits ---
for split in validation1 validation2 validation3; do
    run scripts/23_validate_bertopic.py --config configs/bertopic/validation.yaml \
        --override paths.dataset_csv="${OUT}/preprocessing/translation/${split}_dataset.csv" \
        --override paths.doc_embeddings="${OUT}/bertopic_output/${split}_document_embeddings.pkl" \
        --override paths.run_log="${OUT}/bertopic_output/results_log_${split}.csv" \
        --override params.model_name_pattern="bertopic_${split}"
done

# --- Final 2M fit, metrics, and composite scoring to pick the winner ---
run scripts/24_train_final_bertopic.py --config configs/bertopic/final.yaml
run scripts/21_compute_metrics.py --config configs/bertopic/metrics_tgdataset.yaml \
    --override paths.models_dir="${OUT}/bertopic_output"
run scripts/22_score_metrics.py --config configs/bertopic/score_final.yaml

echo ">>> Publish the winning model directory from ${OUT}/bertopic_output to"
echo ">>> \${models_root}/bertopic_final_<uid>/ before the apply stage (see README,"
echo ">>> section \"Publishing the winning model\")."
# The apply/assignments stages read the frozen model from ${models_root}.
# Publishing the winner there is a manual step. Fail fast if it is missing.
if [[ "$DRY_RUN" != "1" ]]; then
    eval "$($RUN scripts/00_resolve_paths.py models_root)"
    if ! compgen -G "${models_root}/bertopic_final_*" > /dev/null; then
        echo ">>> No published model found under ${models_root}/. Stopping before the"
        echo ">>> apply stage. Publish the winner, then re-run."
        exit 1
    fi
fi

echo "########## Phase 2b: Apply the frozen model to each dataset ##########"
run scripts/25_apply_bertopic.py --config configs/bertopic/apply_tgdataset.yaml
run scripts/25_apply_bertopic.py --config configs/bertopic/apply_darkgram.yaml
run scripts/25_apply_bertopic.py --config configs/bertopic/apply_tliveugr.yaml

echo "########## Phase 3: LLM classification and reporting ##########"
for ds in "${DATASETS[@]}"; do
    run scripts/30_generate_topic_assignments.py --config "configs/llm/assignments_${ds}.yaml"
    run scripts/31_consolidate_channels.py       --config "configs/llm/consolidate_${ds}.yaml"
    run scripts/32_classify_channels.py          --config "configs/llm/classify_${ds}.yaml"
    run scripts/33_generate_result_charts.py     --config "configs/llm/charts_${ds}.yaml"
done

echo "########## Done. Results under \${output_root}/llm/results/<dataset>/ ##########"
