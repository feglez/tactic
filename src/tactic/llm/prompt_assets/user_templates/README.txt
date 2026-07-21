User-message templates for the TACTIC LLM classification stage
===============================================================

The user prompt is assembled dynamically by ``tactic.llm.prompts.build_user_prompt``:
a section is included only when its evidence is available, and sections are
renumbered so the prompt remains well-formed. This yields four prompt families,
each in a metadata variant matching the dataset (Title for TGDataset, Username
for DarkGram and TLiveUGR).

Every file in this directory is the literal output of ``build_user_prompt`` with
the slot names left in place, so the templates are verbatim: rendering a template
with its slot values produces byte-identical prompts to the pipeline's own.

Family 1 — Topics present + metadata
  topics_posts_title.txt
  topics_posts_username.txt
  Includes the salient BERTopic topics with shares, the sampled posts, and the
  channel title or username.

Family 2 — All-outlier (no topics) + metadata
  This family arises in two situations:
  a) The channel has topic assignments but every micro-batch carries the outlier
     label: the topic section consists solely of the explicit
     ALL_MESSAGES_ARE_OUTLIERS marker.
       all_outliers_posts_title.txt
       all_outliers_posts_username.txt
  b) The channel contributed no micro-batch documents to the topic model at all:
     the topic section is omitted altogether and the remaining sections are
     renumbered.
       no_topics_posts_title.txt
       no_topics_posts_username.txt
  In both sub-cases the system prompt instructs the model to treat the BERTopic
  evidence as uninformative, so the decision is based on the sampled posts, with
  the metadata as supporting evidence.

Family 3 — Topics present, no metadata
  topics_posts_no_metadata.txt
  Used when the channel's metadata value (title or username, per the dataset's
  variant) is missing.

Family 4 — No topics and no metadata
  posts_only.txt
  Only sampled posts are provided.

Slots
-----
  {bertopic_results}      rendered by ``tactic.llm.prompts.format_bertopic_results``
                          (main topic + secondary topics, names, keywords, shares)
  {sampled_messages_str}  the token-budgeted random sample of preprocessed posts
                          (see ``tactic.llm.budget``)
  {channel_title}         the channel title (Title variant)
  {channel_username}      the channel username (Username variant)

The families differ only in the input sections included: the instructions come
from a single system prompt per metadata variant (``system_prompt_title.txt`` /
``system_prompt_username.txt`` in the parent directory), which defines how to
weigh the evidence when the topic section is uninformative or absent.
