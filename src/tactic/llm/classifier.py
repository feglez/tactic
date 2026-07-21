"""Per-channel LLM cybercrime classification (the final classification stage).

For each channel the model receives its salient topics, a token-budgeted random
sample of its posts, and light metadata, and returns a one-line rationale plus a
single-category verdict. Results are appended to ``classification_log.jsonl`` one
channel at a time, so a run can resume after an interruption. The final bucketed
result CSVs are rebuilt from that log.

Determinism: message sampling uses a per-channel RNG seeded from
``(seed, channel_id)``, so a channel's sampled posts do not depend on the order
in which channels are processed.

Ablation and reuse hooks:
- ``params.omit_topics``: drop the topic section from every prompt (used to
  measure the contribution of the topic guidance).
- ``params.budget_pin_log``: reuse a previous run's per-channel message budget
  (and skip the dry-run request), so an ablation packs the same posts as the
  full-pipeline run instead of gaining room freed by the removed section.
"""

from __future__ import annotations

import ast
import json
import random
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

from tactic.common.config import cfg_get
from tactic.llm import budget as budget_mod
from tactic.llm import prompts as prompts_mod
from tactic.llm.llm_client import LLMClient
from tactic.llm.taxonomy import is_known_category

AVERAGE_CHARS_PER_TOKEN = 4

# Bucket names (also the result CSV suffixes: results_<bucket>.csv).
BUCKET_CYBERCRIME = "cybercrime"
BUCKET_SAFE = "safe"
BUCKET_UNRECOGNIZED = "unrecognized"
BUCKET_API_ERROR = "api_error"
BUCKET_PARSE_ERROR = "error_parsing_response"


def sort_topics(topic_names, representations, topic_percentages):
    """Return the three lists sorted by share descending, kept aligned."""
    triples = sorted(
        zip(topic_names, representations, topic_percentages),
        key=lambda t: t[2],
        reverse=True,
    )
    if not triples:
        return [], [], []
    names, reps, pcts = map(list, zip(*triples))
    return names, reps, pcts


def _sanitize_channel_identification(channel_id, channel_username) -> str:
    """Per-channel folder name: id + sanitized username (<=10 chars)."""
    cid = str(channel_id)
    uname = unicodedata.normalize("NFC", str(channel_username))
    uname = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', uname)
    uname = re.sub(r'[()]+', '', uname)
    uname = re.sub(r'_+', '_', uname)
    uname = uname[:10].rstrip(' ._')
    return f"{cid}({uname})" if uname else cid


def _parse_list_column(col):
    return col.apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])


def _parse_response(text):
    """Parse the model output into (bucket, main_category, rationale).

    Expects a ``RATIONALE:`` line followed by a verdict line. Off-taxonomy
    categories are routed to the unrecognized bucket.
    """
    rationale = ""
    classification = re.sub(r'<.*?>', '', text)  # strip tags like <think>
    lines = [line.strip() for line in classification.split("\n") if line.strip()]

    for line in lines:
        if line.startswith("RATIONALE:"):
            rationale = line[len("RATIONALE:"):].strip()
            break

    verdict_lines = [l for l in lines if l.startswith("CYBERCRIME ALERT:") or l.startswith("SAFE CHANNEL")]
    verdict = verdict_lines[-1] if verdict_lines else (lines[-1] if lines else "")

    if verdict.startswith("CYBERCRIME ALERT:"):
        category_match = re.findall(r'"(.+?)"', verdict)
        if category_match:
            main_category = category_match[0].strip()
            if not is_known_category(main_category):
                return BUCKET_UNRECOGNIZED, "Unknown classification type", rationale
            return BUCKET_CYBERCRIME, main_category, rationale
        return BUCKET_UNRECOGNIZED, "Error in classification format", rationale

    if verdict.startswith("SAFE CHANNEL"):
        return BUCKET_SAFE, "Safe channel", rationale

    return BUCKET_UNRECOGNIZED, "Unknown classification type", rationale


def _load_messages_df(cfg: dict) -> pd.DataFrame:
    """Load the message source for post sampling (single CSV or shards)."""
    source = cfg_get(cfg, "params.messages_source", "llm_input_csv")
    if source == "shard_dir":
        folder = Path(cfg_get(cfg, "paths.messages_dir"))
        files = sorted(p for p in folder.glob("*.csv") if p.is_file())
        if not files:
            raise SystemExit(f"No CSV files found in: {folder}")
        dfs = [pd.read_csv(p, dtype={"Channel ID": "string"}, encoding="utf-8") for p in files]
        return pd.concat(dfs, ignore_index=True, sort=False)
    return pd.read_csv(cfg_get(cfg, "paths.messages_csv"), dtype={"Channel ID": "string"})


def _load_budget_pin(path: Path) -> dict:
    """Read {channel_id -> message_char_budget} from a previous run's log."""
    pin = {}
    if not path.exists():
        raise SystemExit(f"budget_pin_log not found: {path}")
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pin[str(rec["channel_id"])] = int(rec["message_char_budget"])
    return pin


def _load_done_channels(path: Path) -> set:
    """Channel IDs already present in the log (for resume)."""
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    done.add(str(json.loads(line)["channel_id"]))
    return done


def run(cfg: dict) -> None:
    """Execute the LLM classification stage for one dataset from a resolved config."""
    seed = int(cfg_get(cfg, "llm.seed", 42))
    num_ctx = int(cfg_get(cfg, "llm.num_ctx", 6000))
    variability_threshold = int(cfg_get(cfg, "params.variability_threshold", 1800))
    last_chars_budget = int(cfg_get(cfg, "params.last_chars_budget", 50))
    message_column = cfg_get(cfg, "params.message_column", "processed_message")
    evidence_mode = cfg_get(cfg, "params.evidence_mode", "title")
    debug_limit = cfg_get(cfg, "params.debug_limit", None)
    omit_topics = bool(cfg_get(cfg, "params.omit_topics", False))
    resume = bool(cfg_get(cfg, "params.resume", False))

    gt = cfg_get(cfg, "params.ground_truth", None)
    gt_result_column = gt.get("result_column") if gt else None

    result_path = Path(cfg_get(cfg, "paths.output_dir"))
    channel_messages_folder = result_path / "channel_messages_evaluated"
    channel_messages_folder.mkdir(parents=True, exist_ok=True)
    log_path = result_path / "classification_log.jsonl"

    budget_pin = {}
    pin_cfg = cfg_get(cfg, "params.budget_pin_log", None)
    if pin_cfg:
        budget_pin = _load_budget_pin(Path(pin_cfg))

    done = _load_done_channels(log_path) if resume else set()
    if not resume and log_path.exists():
        log_path.unlink()  # fresh run: start a clean log

    # --- Load inputs ---
    consolidated_string_id = bool(cfg_get(cfg, "params.consolidated_channel_id_as_string", False))
    read_kwargs = {"encoding": "utf-8"}
    if consolidated_string_id:
        read_kwargs["dtype"] = {"Channel ID": "string"}
    channel_data_df = pd.read_csv(cfg_get(cfg, "paths.consolidated_csv"), **read_kwargs)
    messages_df = _load_messages_df(cfg)

    for col in ("topics", "topic_names", "representations", "topic_percentages"):
        channel_data_df[col] = _parse_list_column(channel_data_df[col])

    if gt and gt.get("source") == "messages_merge":
        src_col = gt["source_column"]
        channel_data_df = channel_data_df.merge(
            messages_df[["Channel ID", src_col]].dropna().drop_duplicates("Channel ID"),
            on="Channel ID",
            how="left",
        )

    few_shot = prompts_mod.get_fewshot(evidence_mode)
    client = LLMClient.from_config(cfg, few_shot_messages=few_shot)
    final_system_prompt = prompts_mod.build_system_prompt(cfg)

    try:
        from tqdm import tqdm
        iterator = tqdm(channel_data_df.iterrows(), total=len(channel_data_df),
                        desc="Classifying channels", unit="channel")
    except Exception:
        iterator = channel_data_df.iterrows()

    processed = 0
    with log_path.open("a", encoding="utf-8") as log_fh:
        for _, row in iterator:
            channel_id = row["Channel ID"]
            if str(channel_id) in done:
                continue

            channel_username = row["Username"]
            if pd.isna(channel_username):
                channel_username = ""

            topic_names, representations, topic_percentages = sort_topics(
                row["topic_names"], row["representations"], row["topic_percentages"]
            )

            original_category = row[gt["source_column"]] if gt else None

            # metadata section: title (raw) or @username, omitted when absent
            metadata_missing = False
            if evidence_mode == "title":
                metadata_value = row["Title"]
                metadata_missing = pd.isna(metadata_value)
            else:
                if channel_username == "":
                    metadata_value = ""
                    metadata_missing = True
                else:
                    metadata_value = f"@{channel_username}"

            channel_identification = _sanitize_channel_identification(channel_id, channel_username)

            if topic_names:
                bertopic_results = prompts_mod.format_bertopic_results(
                    topic_names, representations, topic_percentages
                )
                topics_present = not omit_topics
            else:
                bertopic_results = ""
                topics_present = False

            # --- base prompt (no posts) for the dry-run token count ---
            base_prompt = prompts_mod.build_user_prompt(
                cfg, bertopic_results, "", metadata_value,
                topics_present=topics_present, posts_present=True, metadata_missing=metadata_missing,
            )

            dry_run_seconds = 0.0
            if str(channel_id) in budget_pin:
                approx_len = budget_pin[str(channel_id)]
                base_prompt_tokens = 0
            else:
                t0 = time.time()
                base_prompt_tokens = client.dry_run_prompt_tokens(final_system_prompt, base_prompt)
                dry_run_seconds = time.time() - t0
                _, approx_len = budget_mod.compute_message_budget(
                    num_ctx, base_prompt_tokens, variability_threshold, AVERAGE_CHARS_PER_TOKEN
                )

            channel_msgs_all = messages_df.loc[messages_df["Channel ID"].astype(str) == str(channel_id)]
            channel_msgs_list = channel_msgs_all[message_column].astype(str).tolist()
            random.Random(f"{seed}:{channel_id}").shuffle(channel_msgs_list)

            selected_messages, longest_message_length, _ = budget_mod.select_messages(
                channel_msgs_list, approx_len, last_chars_budget
            )

            sent_messages = len(selected_messages)
            total_messages = len(channel_msgs_list)
            perc_msgs = (sent_messages / total_messages * 100) if total_messages else 0.0

            posts_present = sent_messages > 0
            if posts_present:
                sampled_messages_str = f"- {selected_messages[0]}\n\n"
                sampled_messages_str += "\n\n".join(f"   - {m}" for m in selected_messages[1:])
            else:
                sampled_messages_str = ""

            formatted_messages = f"***********{channel_identification}**************\n"
            for msg in selected_messages:
                formatted_messages += (
                    "******* START OF MESSAGE *******\n"
                    f"{msg}\n"
                    "******* END OF MESSAGE *******\n\n\n\n\n\n"
                )

            final_prompt = prompts_mod.build_user_prompt(
                cfg, bertopic_results, sampled_messages_str, metadata_value,
                topics_present=topics_present, posts_present=posts_present, metadata_missing=metadata_missing,
            )

            channel_folder = channel_messages_folder / channel_identification
            channel_folder.mkdir(parents=True, exist_ok=True)
            (channel_folder / "final_prompt.txt").write_text(final_prompt, encoding="utf-8")

            t0 = time.time()
            result = client.classify(final_system_prompt, final_prompt)
            request_seconds = time.time() - t0

            final_prompt_token_count = 0
            answer_token_count = 0
            rationale = ""
            full_answer = "No answer available"

            if result["status_ok"] and not result["parse_error"]:
                full_answer = result["text"].strip()
                final_prompt_token_count = int(result["prompt_eval_count"])
                answer_token_count = int(result["eval_count"])
                bucket, main_category, rationale = _parse_response(full_answer)
            elif not result["status_ok"]:
                bucket, main_category = BUCKET_API_ERROR, "API error"
            else:
                bucket, main_category = BUCKET_PARSE_ERROR, "Error parsing response"

            (channel_folder / "messages.txt").write_text(formatted_messages, encoding="utf-8")
            (channel_folder / "full_answer.txt").write_text(full_answer, encoding="utf-8")

            record = {
                "channel_id": channel_id,
                "channel_username": channel_username,
                "bucket": bucket,
                "main_category": main_category,
                "rationale": rationale,
                "percentage_of_messages": perc_msgs,
                "messages_parsed": sent_messages,
                "chars_in_longest_message": longest_message_length,
                "base_prompt_tokens": base_prompt_tokens,
                "message_char_budget": int(approx_len),
                "final_prompt_token_count": final_prompt_token_count,
                "answer_token_count": answer_token_count,
                "request_seconds": round(request_seconds, 4),
                "dry_run_seconds": round(dry_run_seconds, 4),
            }
            if gt_result_column:
                record[gt_result_column] = None if original_category is None or pd.isna(original_category) else original_category
            log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_fh.flush()

            processed += 1
            if debug_limit is not None and processed >= int(debug_limit):
                break

    (result_path / "system_prompt.txt").write_text(final_system_prompt, encoding="utf-8")
    _write_result_csvs(cfg, result_path, log_path, gt_result_column)
    print(f"Classification completed, results saved to {result_path}")


def _write_result_csvs(cfg, result_path, log_path, gt_result_column):
    """Rebuild the bucketed result CSVs and the summary from the JSONL log."""
    records = []
    with log_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise SystemExit("No channels classified. Exiting.")

    columns = ["channel_id", "channel_username", "main_category", "rationale"]
    if gt_result_column:
        columns.append(gt_result_column)
    columns += [
        "percentage_of_messages", "messages_parsed", "chars_in_longest_message",
        "message_char_budget", "final_prompt_token_count", "answer_token_count",
    ]

    buckets = {
        BUCKET_CYBERCRIME: [], BUCKET_SAFE: [], BUCKET_UNRECOGNIZED: [],
        BUCKET_API_ERROR: [], BUCKET_PARSE_ERROR: [],
    }
    for rec in records:
        buckets.get(rec["bucket"], buckets[BUCKET_UNRECOGNIZED]).append(rec)

    for name, rows in buckets.items():
        pd.DataFrame(rows, columns=columns).to_csv(result_path / f"results_{name}.csv", index=False)

    total = len(records)
    n_cyber = len(buckets[BUCKET_CYBERCRIME])
    summary = pd.DataFrame([
        {"bucket": "CYBERCRIME ALERT", "channels": n_cyber},
        {"bucket": "SAFE CHANNEL", "channels": len(buckets[BUCKET_SAFE])},
        {"bucket": "UNRECOGNIZED", "channels": len(buckets[BUCKET_UNRECOGNIZED])},
        {"bucket": "API ERROR", "channels": len(buckets[BUCKET_API_ERROR])},
        {"bucket": "ERROR PARSING RESPONSE", "channels": len(buckets[BUCKET_PARSE_ERROR])},
        {"bucket": "TOTAL", "channels": total},
        {"bucket": "DETECTION RATE", "channels": round(n_cyber / total, 4) if total else 0.0},
    ])
    summary.to_csv(result_path / "summary.csv", index=False)
