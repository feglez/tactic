Ollama Modelfiles for the TACTIC LLM classification stage
=========================================================

These two files define the custom Ollama models used by the classification
stage; both are built from the gpt-oss:20b base model.

  custom_modelfile           -> builds  gpt-oss:20b_telegramResearch
                                (few-shot exemplars use a channel TITLE, used for
                                 the TGDataset run)
  custom_modelfile_darkgram  -> builds  gpt-oss:20b_telegramResearch_darkgram
                                (few-shot exemplars use a channel USERNAME, used
                                 for the DarkGram and TLiveUGR runs)

Each Modelfile contains:
  - FROM gpt-oss:20b
  - two few-shot MESSAGE user/assistant pairs (one cybercrime example ->
    RATIONALE line + CYBERCRIME ALERT with an exact taxonomy string, and one
    benign example -> RATIONALE line + SAFE CHANNEL)
  - the stock gpt-oss harmony TEMPLATE (reasoning=medium)
  - PARAMETER temperature 1  (the classifier overrides this to 0.0 per request)
  - the base model's LICENSE text

There is NO SYSTEM directive: the system prompt is sent per request by the
classifier (see src/tactic/llm/prompts.py). The baked exemplars match that
system prompt's output contract (a RATIONALE line followed by a single-category
verdict).

Build them (requires Ollama + the gpt-oss:20b base model) on the machine that
serves the model: the local machine for the default transport
(llm.transport: ollama_generate), or the remote Ollama host that llm.api_base
points to when using the litellm transport:

  ollama create gpt-oss:20b_telegramResearch          -f custom_modelfile
  ollama create gpt-oss:20b_telegramResearch_darkgram -f custom_modelfile_darkgram

The commented FROM line in each Modelfile records the base-weights blob digest
the models were built from:
  sha256-b112e727c6f18875636c56a779790a590d705aec9e1c0eb5a97d51fc2a778583
After `ollama pull gpt-oss:20b`, compare with
`ollama show gpt-oss:20b --modelfile | grep FROM`: a different digest means the
gpt-oss:20b tag has moved, in which case some classification drift is to be
expected and the digest actually obtained should be recorded, together with the
`ollama --version` in use at the first run.

Note: both Modelfiles declare `PARAMETER temperature 1`, but the classifier sends
per-request options (temperature 0.0, seed 42, num_ctx 6000) on both transports,
so invoking these builds through the pipeline samples deterministically. A live
gpt-oss:20b generation is not bit-deterministic across hardware/engine versions
even so.
