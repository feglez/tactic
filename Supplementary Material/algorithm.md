# TACTIC per-channel classification (Algorithm 1)

This file states in pseudocode the per-channel classification procedure
implemented by this repository, end to end (paper Section 3.2); its steps map
to the numbered pipeline scripts, with preprocessing in `scripts/01`–`11`,
frozen-model application and summarization in `scripts/25`, `30` and `31`, and
prompt construction and the LLM decision in `scripts/32` and `src/tactic/llm/`.

```text
Require: channel messages M; channel metadata d (title or username);
         frozen BERTopic model B; taxonomy T; salience threshold τ;
         context budget C; attempt limit A
Ensure:  verdict v ∈ {SAFE, CYBERCRIME, UNRESOLVED};
         category c ∈ T (only when v = CYBERCRIME); rationale r

Step 1: summarize the channel into topic cues
 1: clean   ← Preprocess(M)            ▷ clean, language-normalize, translate to English
 2: batches ← MicroBatch(clean, 10)    ▷ 10 consecutive messages per document
 3: topics  ← B.Transform(batches)     ▷ frozen model: one topic label per micro-batch
 4: shares  ← TopicProportions(topics) ▷ per-topic share; outlier label -1 excluded
 5: cues    ← Salient(shares, τ)       ▷ topics with share ≥ τ; relax τ until ≥ 2 qualify

Step 2: assemble the evidence prompt
 6: posts  ← SamplePosts(clean, C)         ▷ random posts, length-bounded by the budget C
 7: prompt ← BuildPrompt(cues, posts, d, T) ▷ missing sections omitted and renumbered

Step 3: LLM decision (two-part output in a single response)
 8: for a = 1, ..., A do
 9:     y ← GPTOSS(prompt)                  ▷ temperature = 0 when a = 1
10:     if y matches the output contract then   ▷ RATIONALE line + verdict line
11:         (v, c, r) ← Parse(y)            ▷ c is set only when v = CYBERCRIME
12:         return (v, c, r)
13: return (UNRESOLVED, ⊥, ⊥)   ▷ no parseable answer: reported separately, no category
```
