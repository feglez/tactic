"""TACTIC: Topic-guided AI for Cybercriminal Telegram Identification and Classification.

Configuration-driven research pipeline, organized as:

- ``tactic.common``         shared config / logging / IO / metrics helpers
- ``tactic.preprocessing``  Phase 1: dataset ingestion, cleaning, language
                            detection, sampling, translation, LLM-input prep
- ``tactic.topic_modeling`` Phase 2: BERTopic tuning / validation / final fit /
                            metrics / frozen-model application
- ``tactic.llm``            Phase 3: topic assignments, channel consolidation,
                            LLM classification, result charts

See README.md for the full stage list and how to run it.
"""

__version__ = "1.0.0"
