#!/usr/bin/env python
"""LDA baseline vs the frozen BERTopic winner, under the pipeline's own c_v protocol (paper Section 5.2).

What this measures
------------------
The published tuning coherence for ``bertopic_e287af`` (0.7378808006907407, 45
topics) was produced by ``tactic.topic_modeling.metrics.run_metrics_multi``:

    docs           = aggregation.build_grouped_docs(coalesce_processed_message(df))
    analyzer       = topic_model.vectorizer_model.build_analyzer()
    tokenized_docs = [analyzer(d) for d in docs]
    dictionary     = Dictionary(tokenized_docs)            # NOT filtered
    CoherenceModel(topics=<top-10 words, -1 skipped>, texts=tokenized_docs,
                   dictionary=dictionary, coherence="c_v")  # topn left at default

This script reuses that exact path and scores BERTopic's own saved topic words
alongside every LDA grid member IN THE SAME accumulation pass. BERTopic's score
therefore serves two purposes: it is the comparison baseline, and, since the
same 0.7378808 score is obtained, we can be sure that the protocol was
replicated.

Why the protocol matters
------------------------
A naive implementation deviates on two axes that both move c_v: tokenizing with
gensim ``simple_preprocess`` instead of the CountVectorizer analyzer, and calling
``filter_extremes(no_below=10, no_above=0.5)`` on the dictionary used to SCORE
coherence, which the pipeline never does (gensim's keep_n=100000 default also
silently caps the vocabulary). Numbers produced that way are not comparable to
0.7379.

Vocabulary fairness
-------------------
BERTopic fits its CountVectorizer inside c-TF-IDF on ONE concatenated document
per topic (bertopic 0.17.4, BERTopic._c_tf_idf -> fit_transform over
documents_per_topic.Document.values). With 46 topic-documents, min_df=0.001
resolves to a threshold of ~0.046 documents (no restriction whatsoever) and
max_df=0.9 only drops words present in >41 of the 46 topic-documents. The saved
model confirms this empirically: its fitted vocabulary holds 365,253 terms.
Imposing the micro-batch-level reading of min_df=0.001 on LDA (df >= ~543 of
543k documents) would have handicapped the baseline by orders of magnitude.

So LDA is offered several vocabularies, including BERTopic's own 365,253-term
pool, and its BEST c_v is reported. Any BERTopic win is then conservative: it
holds against the most favourable LDA configuration found, not one arbitrary
setting.

Cost model
----------
~5.1M retained messages -> ~543k documents -> ~92M tokens. A c_v accumulation
pass over that corpus dominates runtime, so every model is scored in ONE
CoherenceModel over the union of their topic words. That is numerically identical
to scoring each separately: the accumulator's per-word and per-pair counts do not
depend on which other words are relevant, and c_v aggregates per-topic scores
with an arithmetic mean (verified on this run's own data by --verify-batching).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

TACTIC_SRC = Path(__file__).resolve().parents[1] / "src"
if str(TACTIC_SRC) not in sys.path:
    sys.path.insert(0, str(TACTIC_SRC))

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer

from tactic.topic_modeling import aggregation
from tactic.topic_modeling.metrics import COHERENCE_MEASURE
from tactic.topic_modeling.stopwords import STOPWORDS

TOP_N = 10  # metrics.topics_as_wordlists(top_n=10)

BERTOPIC_REFERENCE = {
    "model_name": "bertopic_e287af",
    "published_coherence": 0.7378808006907407,
    "published_num_topics": 45,
    "source_csv": "reference_artifacts/bertopic_output/"
                  "model_metrics_stage1_aggregated_scored_FIXED.csv",
}


def load_bertopic_side(model_safetensors_dir: str):
    """Read the winner's vectorizer params, fitted vocabulary and topic words.

    The safetensors save is used instead of model.pkl because the pickle carries
    CUDA-resident tensors (it was written on a GPU host) and is 448 MB, while the
    only things needed here are the vectorizer config and the topic
    representations. BERTopic's ``topic_representations`` is exactly what
    ``get_topics()`` returns, which is what metrics.topics_as_wordlists() reads.
    """
    d = Path(model_safetensors_dir)
    ctfidf = json.loads((d / "ctfidf_config.json").read_text())
    topics = json.loads((d / "topics.json").read_text())

    params = dict(ctfidf["vectorizer_model"]["params"])
    if isinstance(params.get("ngram_range"), list):
        params["ngram_range"] = tuple(params["ngram_range"])
    fitted_vocab = set(ctfidf["vectorizer_model"]["vocab"])

    reps = topics["topic_representations"]
    # metrics.topics_as_wordlists: skip topic -1 and empty topics, take top-10.
    word_lists, ids = [], []
    for tid in sorted(reps, key=lambda x: int(x)):
        if int(tid) == -1 or not reps[tid]:
            continue
        words = [w for (w, _) in reps[tid][:TOP_N] if isinstance(w, str) and w]
        if words:
            word_lists.append(words)
            ids.append(int(tid))
    return params, fitted_vocab, word_lists, ids


def build_grouped_docs_legacy(df: pd.DataFrame, agg_size: int = 10) -> list:
    """The remainder rule the PUBLISHED tuning corpus was actually built with.

    This deliberately does NOT call aggregation.build_grouped_docs. The release
    module (tactic/.../aggregation.py:56) exempts each CHANNEL's first chunk
    (`agg_idx == 0`), while every historical training/metrics script exempts only
    the first chunk of the WHOLE corpus (`len(grouped_docs) == 0`):

        BERTopic_tuning_phase_1_message_aggregation.py:68
        get_metrics_aggregated.py:37
            if len(chunk) >= 2 or len(grouped_docs) == 0:

    The difference is the 253 single-message channels: the release rule yields
    48,953 documents, the historical rule 48,700. The historical logs record
    48,700 ("Created 48700 aggregated documents", "Loaded 48700 documents") and
    no artifact anywhere states 48,953. Because c_v windows never cross document
    boundaries, the document set IS the reference corpus, so rebuilding it with
    the release rule would score LDA against a different corpus than the 0.7379
    it is compared to. See docs/historical_execution_trace.md R8; the divergence
    is known, and project_refactor keeps a `chunk_rule` flag for it that the
    tactic spin-off dropped.
    """
    grouped_docs = []
    for _, group in df.groupby("Channel ID", sort=True):
        messages = group["processed_message"].tolist()
        for i in range(0, len(messages), agg_size):
            chunk = messages[i:i + agg_size]
            if len(chunk) >= 2 or len(grouped_docs) == 0:
                grouped_docs.append(" ".join(chunk))
    return grouped_docs


def load_documents(dataset_csv: str, chunk_rows: int, max_rows: int | None,
                   channel_id_dtype: str = "string") -> list:
    """Rebuild the published tuning corpus with a bounded memory peak.

    coalesce_processed_message applies a strictly per-row rule then drops nulls,
    so applying the release function chunk-wise and concatenating yields the
    identical row set in the identical order (verified against the historical log:
    499,993 read -> 463,145 retained).

    Channel ID dtype is pinned to string so it cannot drift between chunks. The
    historical scripts let pandas infer it (int64), which visits channels in
    numeric instead of string order. That is immaterial here: the global-first
    exemption fires only if the very first chunk of the corpus holds a single
    message, and it does not (48,700 == 48,953 - 253 exactly, with no +1), so the
    document SET is "every chunk with >= 2 messages" regardless of order, and c_v
    is computed per document. The document-count assertion is the guard.
    """
    reader = pd.read_csv(
        dataset_csv,
        usecols=["Channel ID", "original_message", "translated_message", "language"],
        dtype=({"Channel ID": "string"} if channel_id_dtype == "string" else None),
        chunksize=chunk_rows,
        nrows=max_rows,
    )
    parts, seen = [], 0
    for chunk in reader:
        seen += len(chunk)
        parts.append(aggregation.coalesce_processed_message(chunk)[["Channel ID", "processed_message"]])
        print(f"    read {seen:,} rows", flush=True)
    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    print(f"  retained {len(df):,} of {seen:,} messages ({100*len(df)/seen:.1f}%)", flush=True)
    docs = build_grouped_docs_legacy(df)
    del df
    gc.collect()
    return docs


def tokenize(docs: list, params: dict) -> list:
    """Apply the winner's own CountVectorizer analyzer, freeing docs as we go.

    Tokens are interned so a ~92M-token corpus costs one pointer per token rather
    than one string object per token.
    """
    analyzer = CountVectorizer(**params).build_analyzer()
    intern = sys.intern
    out, n = [], len(docs)
    for i in range(n):
        out.append([intern(t) for t in analyzer(docs[i])])
        docs[i] = None
        if (i + 1) % 50000 == 0:
            print(f"    tokenized {i+1:,}/{n:,}", flush=True)
    return out


class BowCorpus:
    """Streaming bag-of-words view; never materializes the full BoW corpus."""

    def __init__(self, tokenized, dictionary):
        self.tokenized, self.dictionary = tokenized, dictionary

    def __iter__(self):
        doc2bow = self.dictionary.doc2bow
        for toks in self.tokenized:
            yield doc2bow(toks)

    def __len__(self):
        return len(self.tokenized)


def make_train_dict(spec: str, tokenized, dict_score: Dictionary, fitted_vocab: set) -> tuple:
    """Build LDA's training vocabulary. Returns (Dictionary, description)."""
    d = Dictionary(tokenized)
    if spec == "bertopic":
        keep = [d.token2id[t] for t in fitted_vocab if t in d.token2id]
        d.filter_tokens(good_ids=keep)
        desc = f"BERTopic's own fitted vocabulary ({len(fitted_vocab):,} terms; {len(keep):,} present)"
    else:
        nb = int(spec)
        d.filter_extremes(no_below=nb, no_above=0.9, keep_n=None)
        desc = f"filter_extremes(no_below={nb}, no_above=0.9, keep_n=None)"
    d.compactify()
    return d, desc


def verify_topics_pkl_alignment(tokenized, topics, params, fitted_vocab) -> dict:
    """Prove that topics.pkl lines up positionally with the rebuilt documents.

    c_v is order-invariant, so reproducing 0.7378808 proves the document SET is
    right but says nothing about ORDER -- and topics.pkl is positional. This
    refits the winner's CountVectorizer exactly as BERTopic does inside c-TF-IDF
    (one concatenated pseudo-document per topic, including -1) and compares the
    resulting vocabulary against the winner's saved one. A misaligned mapping
    redistributes words across the pseudo-documents, which moves the max_df
    cut and changes the vocabulary, so an exact match is strong evidence the
    ordering matches the historical run.
    """
    per_topic = {}
    for toks, t in zip(tokenized, topics):
        per_topic.setdefault(int(t), []).append(" ".join(toks))
    docs_per_topic = [" ".join(per_topic[t]) for t in sorted(per_topic)]
    vec = CountVectorizer(**params)
    vec.fit(docs_per_topic)
    rebuilt = set(vec.get_feature_names_out())
    return {
        "n_pseudo_documents": len(docs_per_topic),
        "rebuilt_vocab_size": len(rebuilt),
        "saved_vocab_size": len(fitted_vocab),
        "identical": bool(rebuilt == fitted_vocab),
        "only_in_rebuilt": len(rebuilt - fitted_vocab),
        "only_in_saved": len(fitted_vocab - rebuilt),
    }


def train_lda(tokenized, dict_train, k, workers, passes, seed, chunksize):
    t0 = time.perf_counter()
    lda = LdaMulticore(
        corpus=BowCorpus(tokenized, dict_train), id2word=dict_train, num_topics=k,
        workers=workers, passes=passes, random_state=seed, chunksize=chunksize,
    )
    rt = time.perf_counter() - t0
    shown = sorted(lda.show_topics(num_topics=-1, num_words=TOP_N, formatted=False))
    words = [[w for w, _ in pairs] for _, pairs in shown]
    del lda
    gc.collect()
    return words, rt


def main() -> None:
    ap = argparse.ArgumentParser(description="E6 LDA-vs-BERTopic c_v comparison.")
    ap.add_argument("--dataset-csv", required=True)
    ap.add_argument("--model-dir", required=True,
                    help="bertopic_e287af/model.safetensors directory.")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--topics", default="10,20,30,45,60,80,100")
    ap.add_argument("--vocabs", default="bertopic,50",
                    help="Comma-separated: 'bertopic' or an integer no_below.")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default=None, help="Extra seeds, run at --stochasticity-k only.")
    ap.add_argument("--stochasticity-k", type=int, default=45)
    ap.add_argument("--chunksize", type=int, default=10000)
    ap.add_argument("--coherence-processes", type=int, default=3)
    ap.add_argument("--chunk-rows", type=int, default=500000)
    ap.add_argument("--max-rows", type=int, default=None, help="Smoke runs only.")
    ap.add_argument("--expect-docs", type=int, default=48700,
                    help="Abort unless the rebuilt corpus has exactly this many documents "
                         "(48,700 per the historical logs). Set 0 to disable for smoke runs.")
    ap.add_argument("--verify-batching", action="store_true")
    ap.add_argument("--channel-id-dtype", choices=["string", "infer"], default="string",
                    help="'infer' reproduces the historical numeric Channel ID sort, which is "
                         "required for positional alignment with topics.pkl.")
    ap.add_argument("--restrict-topics-pkl", default=None,
                    help="Outlier-asymmetry control: path to the winner's topics.pkl. LDA then "
                         "trains ONLY on the documents BERTopic did not send to topic -1, but is "
                         "still scored against the full reference corpus. Requires "
                         "--channel-id-dtype infer.")
    args = ap.parse_args()

    if args.restrict_topics_pkl and args.channel_id_dtype != "infer":
        sys.exit("--restrict-topics-pkl requires --channel-id-dtype infer (topics.pkl is "
                 "positional and aligns to the historical numeric Channel ID sort).")

    ks = [int(x) for x in args.topics.split(",") if x.strip()]
    vocabs = [v.strip() for v in args.vocabs.split(",") if v.strip()]
    extra_seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else []
    t_start = time.perf_counter()

    print(f"[1/5] Reading the frozen winner from {args.model_dir}", flush=True)
    params, fitted_vocab, bt_words, bt_ids = load_bertopic_side(args.model_dir)
    print(f"  vectorizer: ngram_range={params['ngram_range']} min_df={params['min_df']} "
          f"max_df={params['max_df']} stop_words={len(params['stop_words'])}", flush=True)
    print(f"  stop_words match tactic STOPWORDS: {set(params['stop_words']) == set(STOPWORDS)}", flush=True)
    print(f"  fitted vocabulary: {len(fitted_vocab):,} terms", flush=True)
    print(f"  topic words: {len(bt_words)} topics (-1 skipped), "
          f"{sorted({len(w) for w in bt_words})} words each", flush=True)

    print(f"[2/5] Rebuilding pipeline documents from {args.dataset_csv}", flush=True)
    docs = load_documents(args.dataset_csv, args.chunk_rows, args.max_rows, args.channel_id_dtype)
    n_docs = len(docs)
    print(f"  {n_docs:,} aggregated documents.", flush=True)
    if args.expect_docs and n_docs != args.expect_docs:
        sys.exit(
            f"ABORT: rebuilt {n_docs:,} documents but the published tuning corpus has "
            f"{args.expect_docs:,} (historical logs: 'Created 48700 aggregated documents', "
            f"'Loaded 48700 documents'). c_v is computed against this document set, so a "
            f"mismatched corpus makes the comparison to 0.7378808 invalid. Refusing to run."
        )

    print("[3/5] Tokenizing with the winner's own analyzer", flush=True)
    tokenized = tokenize(docs, params)
    del docs
    gc.collect()
    n_tokens = sum(len(t) for t in tokenized)
    print(f"  {n_tokens:,} tokens ({n_tokens/max(n_docs,1):.1f} per document).", flush=True)

    print("[4/5] Building the UNFILTERED scoring dictionary (as metrics.py does)", flush=True)
    dict_score = Dictionary(tokenized)
    print(f"  scoring vocabulary: {len(dict_score):,} types.", flush=True)

    # --- outlier-asymmetry control -----------------------------------------
    # BERTopic's 45 scored topics describe only the documents HDBSCAN did not
    # send to -1 (28,710 / 48,700 = 59.0%); LDA has no outlier bucket and must
    # model all of them. This trains LDA on BERTopic's kept documents only,
    # while still SCORING against the full corpus exactly as metrics.py does,
    # so the two sides then summarise the same documents.
    train_tokenized, alignment, restrict = tokenized, None, None
    if args.restrict_topics_pkl:
        import joblib
        restrict_topics = np.asarray(joblib.load(args.restrict_topics_pkl))
        if len(restrict_topics) != n_docs:
            sys.exit(f"ABORT: topics.pkl has {len(restrict_topics):,} entries but the rebuilt "
                     f"corpus has {n_docs:,}. They cannot be aligned.")
        print("  verifying topics.pkl alignment by refitting the c-TF-IDF vectorizer ...", flush=True)
        alignment = verify_topics_pkl_alignment(tokenized, restrict_topics, params, fitted_vocab)
        print(f"    {alignment['n_pseudo_documents']} pseudo-documents -> vocab "
              f"{alignment['rebuilt_vocab_size']:,} vs saved {alignment['saved_vocab_size']:,} "
              f"-> {'ALIGNED' if alignment['identical'] else 'MISALIGNED'}", flush=True)
        if not alignment["identical"]:
            sys.exit(f"ABORT: rebuilt c-TF-IDF vocabulary differs from the winner's saved one "
                     f"(+{alignment['only_in_rebuilt']} / -{alignment['only_in_saved']}). The "
                     f"document order does not match the historical run, so topics.pkl cannot be "
                     f"used to select documents.")
        keep = [i for i, t in enumerate(restrict_topics) if int(t) != -1]
        train_tokenized = [tokenized[i] for i in keep]
        restrict = {"topics_pkl": args.restrict_topics_pkl, "n_train_docs": len(keep),
                    "n_all_docs": n_docs, "fraction": len(keep) / n_docs,
                    "alignment": alignment}
        print(f"  LDA will train on {len(keep):,} / {n_docs:,} ({100*len(keep)/n_docs:.1f}%) "
              f"non-outlier documents; scoring still uses all {n_docs:,}.", flush=True)

    # Trained topic words are checkpointed as soon as each model finishes: this
    # run is hours long on a RAM-tight box and an OOM at high K would otherwise
    # discard every model trained before it.
    ckpt_path = Path(f"{args.out_prefix}_checkpoint.jsonl")
    done = {}
    if ckpt_path.exists():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)
                done[m["label"]] = m
        print(f"  checkpoint: {len(done)} model(s) already trained -> {sorted(done)}", flush=True)

    def checkpoint(m: dict) -> None:
        with open(ckpt_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(m) + "\n")

    bt_member = {
        "label": "bertopic_e287af", "model": "BERTopic", "k": len(bt_words),
        "vocab": "fitted", "vocab_size": len(fitted_vocab),
        "seed": None, "topic_words": bt_words, "train_seconds": None,
    }
    members = [bt_member]
    if bt_member["label"] not in done:
        checkpoint(bt_member)

    print("[5/5] LDA sweep", flush=True)
    for spec in vocabs:
        wanted = [(k, sd) for k in ks
                  for sd in ([args.seed] + (extra_seeds if (extra_seeds and k == args.stochasticity_k) else []))]
        pending = [(k, sd) for (k, sd) in wanted if f"lda_{spec}_k{k}_seed{sd}" not in done]
        if not pending:
            print(f"  vocabulary '{spec}': all models present in checkpoint; skipping build.", flush=True)
            members.extend(done[f"lda_{spec}_k{k}_seed{sd}"] for (k, sd) in wanted)
            continue

        dict_train, desc = make_train_dict(spec, tokenized, dict_score, fitted_vocab)
        v = len(dict_train)
        print(f"  vocabulary '{spec}': {desc} -> {v:,} types", flush=True)
        if v == 0:
            print("    empty; skipping.", flush=True)
            del dict_train
            gc.collect()
            continue
        for k, sd in wanted:
            label = f"lda_{spec}_k{k}_seed{sd}"
            if label in done:
                print(f"    {label}: from checkpoint.", flush=True)
                members.append(done[label])
                continue
            print(f"    training {label} ...", flush=True)
            words, rt = train_lda(train_tokenized, dict_train, k, args.workers,
                                  args.passes, sd, args.chunksize)
            print(f"      {len(words)} topics in {rt:.1f}s", flush=True)
            m = {"label": label, "model": "LDA", "k": k, "vocab": spec,
                 "vocab_size": v, "seed": sd, "topic_words": words,
                 "train_seconds": rt}
            checkpoint(m)
            members.append(m)
        del dict_train
        gc.collect()

    n_topics_total = sum(len(m["topic_words"]) for m in members)
    print(f"\nScoring {n_topics_total} topics from {len(members)} models in ONE "
          f"{COHERENCE_MEASURE} pass over {n_docs:,} documents", flush=True)
    all_topics, spans = [], []
    for m in members:
        spans.append((len(all_topics), len(all_topics) + len(m["topic_words"])))
        all_topics.extend(m["topic_words"])

    t0 = time.perf_counter()
    cm = CoherenceModel(topics=all_topics, texts=tokenized, dictionary=dict_score,
                        coherence=COHERENCE_MEASURE, processes=args.coherence_processes)
    per_topic = cm.get_coherence_per_topic()
    coh_seconds = time.perf_counter() - t0
    print(f"  accumulated + scored in {coh_seconds:.1f}s", flush=True)

    for m, (a, b) in zip(members, spans):
        vals = per_topic[a:b]
        m["c_v"] = float(np.mean(vals))
        m["c_v_per_topic"] = [float(x) for x in vals]

    bt = members[0]
    delta = bt["c_v"] - BERTOPIC_REFERENCE["published_coherence"]
    reproduction = {
        "recomputed_c_v": bt["c_v"],
        "published_c_v": BERTOPIC_REFERENCE["published_coherence"],
        "abs_diff": abs(delta),
        "reproduced": bool(abs(delta) < 1e-6),
    }
    print(f"\n  PROTOCOL CHECK: BERTopic recomputed={bt['c_v']:.10f} "
          f"published={BERTOPIC_REFERENCE['published_coherence']:.10f} "
          f"diff={abs(delta):.3e} -> {'REPRODUCED' if reproduction['reproduced'] else 'MISMATCH'}",
          flush=True)

    verification = None
    if args.verify_batching:
        print("  verifying batched == standalone on the BERTopic member ...", flush=True)
        cm2 = CoherenceModel(topics=bt["topic_words"], texts=tokenized, dictionary=dict_score,
                             coherence=COHERENCE_MEASURE, processes=args.coherence_processes)
        standalone = float(np.mean(cm2.get_coherence_per_topic()))
        verification = {
            "label": bt["label"], "batched_c_v": bt["c_v"], "standalone_c_v": standalone,
            "abs_diff": abs(standalone - bt["c_v"]),
            "identical": bool(math.isclose(standalone, bt["c_v"], rel_tol=0, abs_tol=1e-9)),
        }
        print(f"    batched={bt['c_v']:.10f} standalone={standalone:.10f} "
              f"diff={verification['abs_diff']:.2e}", flush=True)

    ldas = [m for m in members if m["model"] == "LDA"]
    best = max(ldas, key=lambda m: m["c_v"]) if ldas else None
    matched = [m for m in ldas if m["k"] == BERTOPIC_REFERENCE["published_num_topics"]]
    best_matched = max(matched, key=lambda m: m["c_v"]) if matched else None

    result = {
        "dataset_csv": args.dataset_csv, "model_dir": args.model_dir,
        "max_rows": args.max_rows, "n_documents": n_docs, "n_tokens": n_tokens,
        "scoring_vocab_size": len(dict_score),
        "vectorizer_params": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in params.items() if k != "stop_words"},
        "stop_words_count": len(params["stop_words"]),
        "stop_words_match_tactic": bool(set(params["stop_words"]) == set(STOPWORDS)),
        "coherence_measure": COHERENCE_MEASURE, "top_n": TOP_N,
        "passes": args.passes, "seed": args.seed,
        "bertopic_reference": BERTOPIC_REFERENCE,
        "protocol_reproduction": reproduction,
        "batching_verification": verification,
        "outlier_control": restrict,
        "channel_id_dtype": args.channel_id_dtype,
        "grid": [{k: v for k, v in m.items() if k not in ("c_v_per_topic", "topic_words")}
                 for m in members],
        "lda_best": {k: v for k, v in best.items() if k not in ("c_v_per_topic", "topic_words")} if best else None,
        "lda_best_at_k45": {k: v for k, v in best_matched.items()
                            if k not in ("c_v_per_topic", "topic_words")} if best_matched else None,
        "coherence_seconds": coh_seconds, "total_seconds": time.perf_counter() - t_start,
    }

    out_json = f"{args.out_prefix}.json"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(f"{args.out_prefix}_topics.txt", "w", encoding="utf-8") as fh:
        for m in members:
            fh.write(f"### {m['label']}  c_v={m['c_v']:.6f}  vocab={m['vocab_size']:,}\n")
            for i, (w, cv) in enumerate(zip(m["topic_words"], m["c_v_per_topic"])):
                fh.write(f"  [{i:3d}] c_v={cv:.4f}  {' '.join(w)}\n")
            fh.write("\n")

    print("\n=== E6 summary ===")
    print(f"documents={n_docs:,}  tokens={n_tokens:,}  scoring_vocab={len(dict_score):,}")
    for m in sorted(members, key=lambda x: -x["c_v"]):
        print(f"  {m['label']:<28} K={m['k']:<4} vocab={m['vocab_size']:>7,}  c_v={m['c_v']:.4f}")
    if best:
        print(f"\nBERTopic e287af (K=45)   c_v={bt['c_v']:.4f}")
        print(f"LDA best  {best['label']:<18} c_v={best['c_v']:.4f}  "
              f"-> BERTopic - LDA_best = {bt['c_v'] - best['c_v']:+.4f}")
        if best_matched:
            print(f"LDA best at matched K=45  c_v={best_matched['c_v']:.4f}  "
                  f"-> BERTopic - LDA_K45 = {bt['c_v'] - best_matched['c_v']:+.4f}")
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
