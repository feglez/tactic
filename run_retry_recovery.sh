#!/usr/bin/env bash
# TACTIC — one-shot unrecognized-recovery wrapper.
#
# Drives scripts/34_retry_unrecognized.py across every results dir, so a single
# launch takes a completed classification run to the point where no channel is
# left in an error bucket except genuine, honestly-counted refusals.
#
# What it does (so you can launch it and keep working):
#   For all SIX result dirs (full pipeline + topics-removed, three datasets
#   each), it runs the retry stage and escalates automatically:
#     - round 1: the default budget (--max-retries 12, temps 0->0.5);
#     - any dir that STILL has channels in an error bucket -> further rounds at
#       --max-retries 24 --max-temperature 0.7.
#   The retry stage is resumable and REHEATS on every re-run (it remembers each
#   stalled row's hottest temperature and starts above it, +0.1/run, cap 1.0), so
#   each extra round is strictly hotter than the last. Stubborn rows keep getting
#   fresh decodes until they parse to CYBERCRIME ALERT / SAFE CHANNEL or reveal
#   themselves as genuine refusals (which stay honestly counted). It stops a dir
#   early after two hot rounds with no recovery, and never exceeds MAX_ROUNDS.
#   Then it regenerates the stage-33 reports/PNGs from the updated logs.
#
# Launch it and walk away:
#   nohup ./run_retry_recovery.sh >> retry_recovery.log 2>&1 &
#   tail -f retry_recovery.log
#
#   --dry-run     preview every dir (no model calls, no charts) and exit
#   --no-charts   run the retries but skip the stage-33 rebuild
#   MAX_ROUNDS=6  raise/lower the per-dir round cap (default 4)
#
# Run from the repo root.
set -uo pipefail
cd "$(dirname "$0")"

RESULTS=outputs/llm/results
# Override from the environment to run a subset, e.g.
#   DIRS=tliveugr_notopics ./run_retry_recovery.sh
read -r -a DIRS <<<"${DIRS:-tgdataset darkgram tliveugr tgdataset_notopics darkgram_notopics tliveugr_notopics}"
MAX_ROUNDS=${MAX_ROUNDS:-4}
MAX_ABORTS=${MAX_ABORTS:-5}   # consecutive Ollama outages tolerated per dir
ABORT_WAIT=${ABORT_WAIT:-120} # seconds to wait for Ollama after an outage abort
DRY_RUN=0
DO_CHARTS=1
for a in "$@"; do
  case "$a" in
    --dry-run)   DRY_RUN=1 ;;
    --no-charts) DO_CHARTS=0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

say() { echo "[$(date '+%F %T')] $*"; }

# Count rows still in an error bucket (the three retry buckets) for one dir.
remaining() {
  local log="$RESULTS/$1/classification_log.jsonl" n
  [[ -s "$log" ]] || { echo 0; return 0; }
  n=$(grep -cE '"bucket": "(unrecognized|api_error|error_parsing_response)"' "$log" 2>/dev/null || true)
  echo "${n:-0}"
}

# One retry-stage invocation for <dir> at <round> (round>=2 escalates).
retry_once() {
  local dir=$1 round=$2 cfg=${1%_notopics}
  local flags=(--config "configs/llm/classify_${cfg}.yaml" --results-dir "$RESULTS/$dir")
  (( round >= 2 )) && flags+=(--max-retries 24 --max-temperature 0.7)
  uv run python scripts/34_retry_unrecognized.py "${flags[@]}"
}

# ---- dry-run: preview every dir, then exit ----
if (( DRY_RUN )); then
  for dir in "${DIRS[@]}"; do
    cfg=${dir%_notopics}
    say "dry-run preview: $dir"
    uv run python scripts/34_retry_unrecognized.py --config "configs/llm/classify_${cfg}.yaml" \
        --results-dir "$RESULTS/$dir" --dry-run || say "WARN: preview failed for $dir"
  done
  exit 0
fi

say "=== retry recovery starting (max $MAX_ROUNDS rounds per dir) ==="
for dir in "${DIRS[@]}"; do
  if [[ ! -s "$RESULTS/$dir/classification_log.jsonl" ]]; then
    say "SKIP $dir (no classification_log.jsonl)"; continue
  fi
  stall=0; aborts=0; round=1
  while (( round <= MAX_ROUNDS )); do
    before=$(remaining "$dir"); before=${before:-0}
    if (( before == 0 )); then
      say "$dir: clean (0 in error buckets) — nothing to retry"; break
    fi
    say ">>> $dir round $round/$MAX_ROUNDS: $before in error buckets — retrying"
    retry_once "$dir" "$round"; rc=$?
    after=$(remaining "$dir"); after=${after:-0}
    say "<<< $dir round $round: $before -> $after remaining (rc=$rc)"
    (( after == 0 )) && break

    # rc=75 (EX_TEMPFAIL) = Ollama went away. That is INFRASTRUCTURE, not a model
    # refusal: it must not burn a round and must not feed the stall counter. Without
    # this, a dir whose rounds all died on a dead server gets reported as "genuine
    # refusals" — a fabricated claim about channels the model never even saw.
    if (( rc == 75 )); then
      (( ++aborts ))
      if (( aborts > MAX_ABORTS )); then
        say "$dir: Ollama unreachable ${aborts}x — giving up (INFRA, not refusals)"; break
      fi
      say "$dir: infra abort ${aborts}/${MAX_ABORTS} — waiting ${ABORT_WAIT}s; round $round does NOT count"
      sleep "$ABORT_WAIT"; continue
    fi
    (( rc != 0 )) && say "WARN: retry stage exited nonzero ($dir round $round rc=$rc)"

    if (( after >= before )); then
      (( ++stall ))
      if (( round >= 2 && stall >= 2 )); then
        say "$dir: no recovery across two hot rounds ($after remain) — genuine refusals, moving on"
        break
      fi
    else
      stall=0
    fi
    (( ++round ))
  done
  say "=== $dir FINAL: $(remaining "$dir") still unresolved ==="
done

if (( DO_CHARTS )); then
  say "--- regenerating stage-33 reports/PNGs from the updated logs ---"
  for cfg in tgdataset darkgram tliveugr; do
    uv run python scripts/33_generate_result_charts.py --config "configs/llm/charts_${cfg}.yaml" \
      || say "WARN: charts ${cfg} failed"
  done
  for cfg in tgdataset darkgram tliveugr; do
    uv run python scripts/33_generate_result_charts.py --config "configs/llm/charts_${cfg}.yaml" \
      --override "paths.results_dir=$RESULTS/${cfg}_notopics" \
      --override "paths.output_dir=$RESULTS/${cfg}_notopics" \
      || say "WARN: charts ${cfg}_notopics failed"
  done
fi

say "=== retry recovery DONE ==="
say "Next: check each dir's summary.csv — UNRECOGNIZED should have dropped, API ERROR and"
say "ERROR PARSING RESPONSE should still be 0, and TOTAL must be unchanged for every dataset."
