"""Topic assignments, channel consolidation, and LLM classification.

Channels are classified with GPT-OSS 20B served by a local Ollama instance
(``/api/generate``). A LiteLLM transport is available as a configurable
alternative (see ``llm_client.py``).
"""
