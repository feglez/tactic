#!/usr/bin/env python3
"""Retry-until-parseable recovery for unrecognized LLM classifications.

Why this exists
---------------
The classifier must emit a strict output contract: a ``RATIONALE`` line followed
by either ``CYBERCRIME ALERT: ("<category>")`` or ``SAFE CHANNEL``. A small
proportion of responses do not satisfy it on a first pass -- an empty completion,
a refusal to engage with the request, a reply that paraphrases the channel
without reaching a verdict, or a rationale whose verdict line was cut off because
the model's reasoning filled the configured context window before it got there.
Those channels land in the ``unrecognized`` bucket.

Two properties of the setup make retrying the right remedy:

* Local LLM inference is not bit-reproducible even at temperature 0, so a channel
  that failed to parse once may parse on an identical re-issue. Retrying
  therefore resamples a noisy decoder instead of shopping for a verdict: the
  prompt never changes, only the decoding does.
* Running with a larger ``num_ctx`` also removes most of these cases. This tool
  lets the output contract be satisfied *without* enlarging the context window,
  keeping a run comparable with runs made at the original setting.

Channels that never yield a verdict stay ``unrecognized`` and are reported as
such: they are never assigned to a category, and no verdict is ever written by
hand.

What it does
------------
For every channel whose logged bucket is ``unrecognized`` / ``api_error`` /
``error_parsing_response``, it re-issues the SAME request — the system prompt and
the per-channel ``final_prompt.txt`` are read verbatim from the results folder,
so the prompt and ``num_ctx`` are byte-identical to the original run — to the
SAME model, retrying until the response parses to a valid ``CYBERCRIME ALERT``
(exact taxonomy) or ``SAFE CHANNEL`` verdict. Recovered verdicts are merged back
into ``classification_log.jsonl`` and the bucketed result CSVs are rebuilt via
the pipeline's own ``_write_result_csvs``.

Only the *decoding* varies across retries; the prompt never does. The first
``--temp0-attempts`` retries use the configured temperature (0.0) to honour the
run's determinism first; greedy decoding is effectively deterministic per host,
so if those do not resolve, the temperature ramps by ``--temp-step`` up to
``--max-temperature`` — a shorter reasoning path then leaves room for the verdict.
``num_ctx`` is never changed, so the retried request stays directly comparable to
the original one.

Bounded and resumable: each channel is retried at most ``--max-retries`` times
(default 12) per run — NOT an infinite loop, because a single run at a fixed
temperature could otherwise hang forever on an output that never parses. Channels
unresolved after the cap stay ``unrecognized`` and are counted honestly. The log
is rewritten after every attempt outcome, so you can Ctrl-C at any point and
re-run: rows already fixed are skipped, and everything still in an error bucket is
retried again. A row that failed a previous run remembers how hot it got
(``retry_max_temp``); the NEXT run skips the wasteful temp-0 attempts and starts
above there, raising the ceiling by ``--reheat`` each run (hard cap 1.0), so
successive runs get progressively hotter and stubborn rows eventually resolve (or
reveal themselves as genuine content refusals).

After it finishes, regenerate the summary/breakdown/PNGs from the updated log by
re-running the reporting stage for each affected results dir::

    uv run python scripts/33_generate_result_charts.py --config configs/llm/charts_tgdataset.yaml

Usage (run once per results dir; mirror the classify invocation)::

    # Full pipeline
    uv run python scripts/34_retry_unrecognized.py --config configs/llm/classify_tgdataset.yaml
    uv run python scripts/34_retry_unrecognized.py --config configs/llm/classify_darkgram.yaml
    uv run python scripts/34_retry_unrecognized.py --config configs/llm/classify_tliveugr.yaml
    # Topics-removed ablation: same client config, point at the *_notopics dir
    uv run python scripts/34_retry_unrecognized.py --config configs/llm/classify_tgdataset.yaml \
        --results-dir outputs/llm/results/tgdataset_notopics
    uv run python scripts/34_retry_unrecognized.py --config configs/llm/classify_darkgram.yaml \
        --results-dir outputs/llm/results/darkgram_notopics
    uv run python scripts/34_retry_unrecognized.py --config configs/llm/classify_tliveugr.yaml \
        --results-dir outputs/llm/results/tliveugr_notopics

To retry every results dir in one go, use ``run_retry_recovery.sh`` in the repo
root, which drives this script and then regenerates the stage-33 reports.

Preview only (no model calls): add ``--dry-run``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Make the tactic package importable whether run via `uv run` (installed) or
# plain python from the repo root.
try:
    from tactic.common.cli import build_parser, load_config_from_args
except ModuleNotFoundError:
    sys.path.insert(0, str(Path.cwd() / "src"))
    from tactic.common.cli import build_parser, load_config_from_args

from tactic.common.config import cfg_get
from tactic.llm import prompts as prompts_mod
from tactic.llm.llm_client import LLMClient
from tactic.llm.classifier import (
    BUCKET_API_ERROR,
    BUCKET_CYBERCRIME,
    BUCKET_PARSE_ERROR,
    BUCKET_SAFE,
    BUCKET_UNRECOGNIZED,
    _parse_response,
    _sanitize_channel_identification,
    _write_result_csvs,
)

RETRY_BUCKETS = {BUCKET_UNRECOGNIZED, BUCKET_API_ERROR, BUCKET_PARSE_ERROR}


def _locate_folder(channel_messages_folder: Path, channel_id, channel_username) -> Path | None:
    """Find a channel's saved-prompt folder (sanitized name, with a glob fallback)."""
    guess = channel_messages_folder / _sanitize_channel_identification(channel_id, channel_username)
    if (guess / "final_prompt.txt").exists():
        return guess
    cid = str(channel_id)
    for cand in sorted(channel_messages_folder.glob(f"{cid}(*")) + [channel_messages_folder / cid]:
        if (cand / "final_prompt.txt").exists():
            return cand
    return None


def _write_log(records, log_path: Path) -> None:
    """Atomically rewrite the JSONL log (order preserved). Called after every attempt
    outcome so a stop/restart resumes cleanly: only rows still in an error bucket retry.

    Temp file + fsync + os.replace: a kill landing mid-write leaves either the old
    bytes or the new ones, never a truncated mix. Writing in place would instead
    truncate the real log and stream it back hundreds of times per round, so a
    power loss inside that window could destroy every verdict in the dir."""
    tmp = log_path.with_suffix(log_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, log_path)


def _classify_resilient(client, system_prompt, final_prompt, args):
    """client.classify() with bounded backoff across transient transport failures.

    A dropped connection is NOT a model verdict: it must not consume the channel's
    decode budget, must not touch retry_max_temp/retry_attempts_cum, and must not
    abort the run. We wait for Ollama and re-issue the SAME request -- same prompt,
    same num_ctx, same temperature, same seed. No output was ever observed, so this
    is not seed reuse. Bounded: returns None once the outage budget is spent, which
    the caller turns into a clean SystemExit(75) so the round can be retried without
    being scored as a model refusal."""
    delay = args.transport_backoff
    for t in range(args.transport_retries + 1):
        try:
            return client.classify(system_prompt, final_prompt)
        except Exception as exc:  # ConnectionError / ReadTimeout / ProtocolError
            if t == args.transport_retries:
                print(f"    transport dead after {t + 1} tries ({type(exc).__name__})", flush=True)
                return None
            print(f"    transport error ({type(exc).__name__}); waiting {delay:.0f}s "
                  f"[{t + 1}/{args.transport_retries}]", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, args.transport_backoff_max)
    return None


def _temp_schedule(prior_max_temp: float, args, n: int) -> list:
    """Per-attempt decode temperatures for one channel in one run.

    Fresh row: temp 0 for the first --temp0-attempts (honour determinism), then
    ramp by --temp-step up to --max-temperature. A row that already failed a
    previous run (carrying retry_max_temp) skips the temp-0 attempts, starts where
    it left off, and lifts the ceiling by --reheat (hard cap 1.0), so each re-run
    is hotter than the last."""
    if prior_max_temp and prior_max_temp > 0:
        ceiling = min(1.0, max(prior_max_temp + args.reheat, args.max_temperature))
        temps, t = [], min(prior_max_temp, ceiling)
        for _ in range(n):
            temps.append(round(t, 3))
            t = min(ceiling, t + args.temp_step)
        return temps
    out = []
    for i in range(n):
        out.append(0.0 if i < args.temp0_attempts
                   else round(min(args.max_temperature, args.temp_step * (i - args.temp0_attempts + 1)), 3))
    return out


def main() -> None:
    parser = build_parser(description="Retry unrecognized LLM classifications until parseable.")
    parser.add_argument("--results-dir", default=None,
                        help="Results dir to fix (default: paths.output_dir from the config).")
    parser.add_argument("--max-retries", type=int, default=12,
                        help="Max attempts per channel before giving up.")
    parser.add_argument("--temp0-attempts", type=int, default=3,
                        help="Leading attempts at the configured temperature before escalating.")
    parser.add_argument("--temp-step", type=float, default=0.1,
                        help="Temperature increment per attempt once escalation starts.")
    parser.add_argument("--max-temperature", type=float, default=0.5,
                        help="Ceiling for the escalated retry temperature (first run).")
    parser.add_argument("--reheat", type=float, default=0.1,
                        help="On a re-run, how much hotter than the previous run's highest "
                             "temperature to allow for rows that failed before (hard cap 1.0).")
    parser.add_argument("--transport-retries", type=int, default=20,
                        help="How many times to re-issue a request after a transport failure "
                             "(Ollama restart / connection reset). Not a decode attempt.")
    parser.add_argument("--transport-backoff", type=float, default=10.0,
                        help="Initial wait after a transport failure (s); doubles up to the max.")
    parser.add_argument("--transport-backoff-max", type=float, default=120.0,
                        help="Cap for the transport backoff (s). 20 retries doubling to 120s "
                             "tolerates roughly 35 min of Ollama downtime.")
    parser.add_argument("--request-timeout", type=float, default=600.0,
                        help="Per-request HTTP timeout (s). Without it a silent peer blocks "
                             "recv() forever. Generous: gpt-oss:20b at num_ctx=6000 is slow.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be retried; make no model calls, write nothing.")
    parser.add_argument("--rebuild-only", action="store_true",
                        help="Rebuild results_*.csv + summary.csv from classification_log.jsonl "
                             "and exit. No model calls. Use when a crash left the derived CSVs "
                             "lagging the log (the log is always the source of truth).")
    args = parser.parse_args()

    cfg = load_config_from_args(args)
    result_path = Path(args.results_dir or cfg_get(cfg, "paths.output_dir"))
    log_path = result_path / "classification_log.jsonl"
    sys_prompt_path = result_path / "system_prompt.txt"
    channel_messages_folder = result_path / "channel_messages_evaluated"

    if not log_path.exists():
        raise SystemExit(f"No classification_log.jsonl under {result_path}")
    if not sys_prompt_path.exists():
        raise SystemExit(f"No system_prompt.txt under {result_path} (needed to re-issue the request)")

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    targets = [r for r in records if r.get("bucket") in RETRY_BUCKETS]
    before = Counter(r.get("bucket") for r in records)

    print(f"Results dir : {result_path}")
    print(f"Channels    : {len(records)} total | to retry: {len(targets)} "
          f"(unrecognized={before.get(BUCKET_UNRECOGNIZED,0)}, "
          f"api_error={before.get(BUCKET_API_ERROR,0)}, "
          f"parse_error={before.get(BUCKET_PARSE_ERROR,0)})")

    gt = cfg_get(cfg, "params.ground_truth", None)
    gt_result_column = gt.get("result_column") if gt else None

    if args.rebuild_only:
        _write_result_csvs(cfg, result_path, log_path, gt_result_column)
        n_cyber = before.get(BUCKET_CYBERCRIME, 0)
        print(f"[rebuild-only] CSVs rebuilt from the log. buckets={dict(before)}")
        print(f"[rebuild-only] detection rate : {n_cyber}/{len(records)} = {n_cyber/len(records):.4f}")
        return

    if args.dry_run:
        missing = sum(1 for r in targets
                      if _locate_folder(channel_messages_folder, r["channel_id"], r.get("channel_username", "")) is None)
        print(f"[dry-run] {len(targets)} channels would be retried; {missing} are missing final_prompt.txt.")
        return
    if not targets:
        print("Nothing to retry. Done.")
        return

    system_prompt = sys_prompt_path.read_text(encoding="utf-8")
    evidence_mode = cfg_get(cfg, "params.evidence_mode", "title")
    few_shot = prompts_mod.get_fewshot(evidence_mode)
    client = LLMClient.from_config(cfg, few_shot_messages=few_shot)
    # Opt this tool into an HTTP timeout. The classify stage keeps llm_client's
    # default (None), so enabling it here changes nothing about how channels are
    # classified -- it only stops a silent peer from blocking recv() forever
    # during a long unattended retry run.
    client.request_timeout = args.request_timeout
    base_seed, base_temp = client.seed, client.temperature

    recovered, unresolved, needed_escalation, no_prompt = 0, [], 0, []
    transport_dead = False
    try:
        for i, rec in enumerate(targets, 1):
            cid = rec["channel_id"]
            folder = _locate_folder(channel_messages_folder, cid, rec.get("channel_username", ""))
            if folder is None:
                no_prompt.append(str(cid))
                continue
            final_prompt = (folder / "final_prompt.txt").read_text(encoding="utf-8")

            prior_max_temp = float(rec.get("retry_max_temp", 0.0) or 0.0)
            prior_attempts = int(rec.get("retry_attempts_cum", 0) or 0)
            schedule = _temp_schedule(prior_max_temp, args, args.max_retries)
            run_max_temp = prior_max_temp
            ok = False
            attempts_done = 0
            for attempt, temp in enumerate(schedule):
                client.temperature = temp
                client.seed = base_seed + prior_attempts + attempt  # never repeat a seed across runs
                t0 = time.time()
                res = _classify_resilient(client, system_prompt, final_prompt, args)
                dt = time.time() - t0
                if res is None:  # Ollama never came back within the outage budget
                    transport_dead = True
                    break
                # Only credit heat the model actually sampled at: a temperature charged
                # for an unanswered request would make the next run skip decodes that
                # never happened, narrowing the search for exactly the hardest rows.
                run_max_temp = max(run_max_temp, temp)
                attempts_done += 1
                if res["status_ok"] and not res["parse_error"]:
                    answer = (res["text"] or "").strip()
                    bucket, main_category, rationale = _parse_response(answer)
                    if bucket in (BUCKET_CYBERCRIME, BUCKET_SAFE):
                        rec.update({
                            "bucket": bucket,
                            "main_category": main_category,
                            "rationale": rationale,
                            "final_prompt_token_count": int(res["prompt_eval_count"]),
                            "answer_token_count": int(res["eval_count"]),
                            "request_seconds": round(dt, 4),
                            "retry_attempts_cum": prior_attempts + attempts_done,
                            "retry_temperature": round(temp, 3),
                            "recovered_by_retry": True,
                        })
                        rec.pop("retry_max_temp", None)  # resolved; clear the stall marker
                        (folder / "full_answer.txt").write_text(answer, encoding="utf-8")
                        recovered += 1
                        if temp > 0:
                            needed_escalation += 1
                        ok = True
                        _write_log(records, log_path)  # persist now -> Ctrl-C / restart safe
                        break
            client.seed, client.temperature = base_seed, base_temp  # restore for next channel
            if not ok:
                # remember how hot we got so the NEXT run starts above here (no temp-0
                # waste). Credit REAL attempts only, never the unspent budget.
                rec["retry_max_temp"] = round(run_max_temp, 3)
                rec["retry_attempts_cum"] = prior_attempts + attempts_done
                _write_log(records, log_path)
                unresolved.append(f"{cid}@T{round(run_max_temp, 2)}")
            print(f"  [{i}/{len(targets)}] {cid}: "
                  + (f"recovered -> {rec['bucket']} (attempts_cum {rec.get('retry_attempts_cum')}, "
                     f"T={rec.get('retry_temperature')})" if ok
                     else f"still unresolved (max T tried {rec.get('retry_max_temp')})"))
            if transport_dead:
                print(f"\nABORTING: Ollama unreachable at channel {cid} "
                      f"({i}/{len(targets)}). This is infrastructure, NOT a model refusal.", flush=True)
                break
    finally:
        # The log is the source of truth; the CSVs are derived and must NEVER lag it.
        # Rebuilding here instead of after the loop means an interrupted run cannot
        # leave results_*.csv / summary.csv a round behind the log -- stage 33 reads
        # the CSVs, so a stale one would otherwise be charted as a real number.
        _write_log(records, log_path)
        _write_result_csvs(cfg, result_path, log_path, gt_result_column)

    if transport_dead:
        # EX_TEMPFAIL: the wrapper reads this to retry the round WITHOUT charging it
        # against MAX_ROUNDS and without scoring it as "no recovery -> genuine refusals".
        raise SystemExit(75)

    after = Counter(r.get("bucket") for r in records)
    total = len(records)
    n_cyber = after.get(BUCKET_CYBERCRIME, 0)
    print("\n=== retry summary ===")
    print(f"recovered      : {recovered} (of these {needed_escalation} needed temperature escalation)")
    print(f"still unrecognized/unresolved: {len(unresolved)}" + (f" -> {unresolved}" if unresolved else ""))
    if no_prompt:
        print(f"missing final_prompt.txt (skipped): {len(no_prompt)} -> {no_prompt}")
    print(f"buckets before : {dict(before)}")
    print(f"buckets after  : {dict(after)}")
    print(f"detection rate : {n_cyber}/{total} = {n_cyber/total:.4f}")
    print("\nNext: regenerate summary/breakdown/PNGs from the updated log, e.g.")
    print(f"  uv run python scripts/33_generate_result_charts.py --config configs/llm/charts_<dataset>.yaml")


if __name__ == "__main__":
    main()
