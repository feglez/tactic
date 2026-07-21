"""System/user prompt construction for the LLM classifier.

The system prompt asks the model to emit a one-line RATIONALE followed by a
single-category verdict. The user prompt is assembled dynamically: a section is
included only when its evidence is available (topics, sampled publications,
channel title/username), so missing inputs simply drop out of the prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

from tactic.common.config import cfg_get
from tactic.llm.taxonomy import category_block

_ASSET_DIR = Path(__file__).resolve().parent / "prompt_assets"

ALL_OUTLIERS_SENTINEL = "ALL_MESSAGES_ARE_OUTLIERS"


def _load_asset(name: str) -> str:
    return (_ASSET_DIR / name).read_text(encoding="utf-8")


def get_system_prompt_template(evidence_mode: str) -> str:
    """System prompt with the {cybercrime_categories} slot still present."""
    return _load_asset(
        "system_prompt_title.txt" if evidence_mode == "title" else "system_prompt_username.txt"
    )


def get_fewshot(evidence_mode: str) -> list:
    """Few-shot exemplars as chat messages.

    In the default Ollama flow the exemplars are baked into the custom model, so
    the classifier does not resend them. They are used only when
    ``llm.few_shot_in_messages`` is true (routing to a proxy without the custom
    model).
    """
    name = "fewshot_title.json" if evidence_mode == "title" else "fewshot_username.json"
    return json.loads((_ASSET_DIR / name).read_text(encoding="utf-8"))


def build_system_prompt(cfg: dict) -> str:
    """Compose the system prompt (category list substituted)."""
    evidence_mode = cfg_get(cfg, "params.evidence_mode", "title")
    return get_system_prompt_template(evidence_mode).format(cybercrime_categories=category_block())


def format_bertopic_results(
    topic_names: list,
    representations: list,
    topic_percentages: list,
) -> str:
    """Render the topic block: the main topic plus any secondary topics.

    Returns the ``ALL_MESSAGES_ARE_OUTLIERS`` sentinel when the highest-share
    topic is the outlier topic (-1).
    """
    main_name = topic_names[0]
    main_kw = representations[0]
    main_pct = float(topic_percentages[0])

    if str(main_name).startswith("-1"):
        return ALL_OUTLIERS_SENTINEL

    bertopic_results = (
        f"Main topic:\n"
        f"     topic name: {main_name}\n"
        f"     keywords: {main_kw}\n"
        f"     share: {main_pct:.2f}%"
    )
    if len(topic_names) > 1:
        sec_lines = [
            (
                f"     - topic name: {t}\n"
                f"       keywords: {r}\n"
                f"       share: {float(p):.2f}%"
            )
            for t, r, p in zip(topic_names[1:], representations[1:], topic_percentages[1:])
        ]
        bertopic_results += "\n   Secondary topics:\n" + "\n".join(sec_lines)
    return bertopic_results


def build_user_prompt(
    cfg: dict,
    bertopic_results: str,
    sampled_messages_str: str,
    metadata_value,
    topics_present: bool = True,
    posts_present: bool = True,
    metadata_missing: bool = False,
) -> str:
    """Compose the user prompt, omitting any section without evidence.

    Sections are renumbered so there are never gaps: a channel with no title
    and no fitting posts still yields a well-formed prompt built from whatever
    remains.
    """
    evidence_mode = cfg_get(cfg, "params.evidence_mode", "title")

    sections = []
    if topics_present:
        sections.append(f"BERTopic analysis results\n   {bertopic_results}")
    if posts_present:
        sections.append(f"Sampled publications\n   {sampled_messages_str}")
    if not metadata_missing:
        label = "Channel title" if evidence_mode == "title" else "Channel username"
        sections.append(f"{label}\n   {metadata_value}")

    body = "\n".join(f"\n{i}) {sec}\n" for i, sec in enumerate(sections, start=1))
    return f"\n{body}\n"
