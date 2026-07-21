"""LLM backend client for the classification stage.

Two transports, selected by ``llm.transport``:

- ``ollama_generate`` (default): a direct POST to ``{api_base}/api/generate``
  with the system prompt in the request's separate ``system`` field. Ollama
  renders the model's own template around system+user, so the reported
  ``prompt_eval_count`` matches the local model exactly. Few-shot exemplars are
  expected to be baked into the custom Ollama model (see ``modelfiles/``).
- ``litellm``: route through the ``litellm`` library to a proxy or any other
  backend. The ``ollama/`` prefix flattens system+user into a single prompt
  string, so token accounting differs from the direct transport. Use this for
  convenience, not for token-identical reproduction. Few-shot exemplars are sent
  as chat messages when ``llm.few_shot_in_messages`` is true (only valid here).

Field mapping (Ollama /api/generate  <->  litellm.completion):
  payload["system"]                 -> messages[0] {role: "system"}
  payload["prompt"] (user)          -> messages[-1] {role: "user"}
  options.num_predict = 1 (dry-run) -> max_tokens = 1
  options.temperature/seed/num_ctx  -> temperature/seed + llm.extra_params
  result["response"]                <- choices[0].message.content
  result["prompt_eval_count"]       <- usage.prompt_tokens
  result["eval_count"]              <- usage.completion_tokens

A live generation is not bit-deterministic across hardware/engine versions even
with a fixed seed and temperature 0.
"""

from __future__ import annotations

import os
from typing import Optional

from tactic.common.config import cfg_get

TRANSPORT_OLLAMA_GENERATE = "ollama_generate"
TRANSPORT_LITELLM = "litellm"


class LLMClient:
    """LLM transport client exposing a uniform request/response interface."""

    def __init__(
        self,
        model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        seed: int = 42,
        few_shot_messages: Optional[list] = None,
        extra_params: Optional[dict] = None,
        transport: str = TRANSPORT_OLLAMA_GENERATE,
    ):
        if transport not in (TRANSPORT_OLLAMA_GENERATE, TRANSPORT_LITELLM):
            raise ValueError(f"Unknown llm.transport: {transport!r}")
        if transport == TRANSPORT_OLLAMA_GENERATE and few_shot_messages:
            raise ValueError(
                "llm.transport=ollama_generate cannot carry "
                "llm.few_shot_in_messages=true because /api/generate has no message "
                "array (the custom Ollama model bakes the exemplars in). Use the "
                "litellm transport for proxy few-shot routing."
            )
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.temperature = temperature
        self.seed = seed
        self.few_shot_messages = few_shot_messages or []
        self.extra_params = dict(extra_params or {})
        self.transport = transport
        # Per-request HTTP timeout. None = block until the peer answers, which is the
        # behaviour that produced the reported results, so the classification stage is
        # byte-identical by default. Long unattended jobs (see the retry stage) opt in
        # to a bound so a silent peer cannot block recv() forever.
        self.request_timeout = None

    @classmethod
    def from_config(cls, cfg: dict, few_shot_messages: Optional[list] = None) -> "LLMClient":
        api_key_env = cfg_get(cfg, "llm.api_key_env", None)
        api_key = os.environ.get(api_key_env) if api_key_env else None  # read at runtime, never stored in config
        return cls(
            model=cfg_get(cfg, "llm.model"),
            api_base=cfg_get(cfg, "llm.api_base", None),
            api_key=api_key,
            temperature=float(cfg_get(cfg, "llm.temperature", 0.0)),
            seed=int(cfg_get(cfg, "llm.seed", 42)),
            few_shot_messages=(few_shot_messages if bool(cfg_get(cfg, "llm.few_shot_in_messages", False)) else None),
            extra_params=cfg_get(cfg, "llm.extra_params", {}) or {},
            transport=cfg_get(cfg, "llm.transport", TRANSPORT_OLLAMA_GENERATE),
        )

    # ---- direct Ollama transport (default) ----

    def _generate_url(self) -> str:
        base = (self.api_base or "http://localhost:11434").rstrip("/")
        return f"{base}/api/generate"

    def _ollama_payload(self, system_prompt: str, user_prompt: str) -> dict:
        model = self.model
        if model.startswith("ollama/"):
            model = model[len("ollama/"):]
        options = {"temperature": self.temperature, "seed": self.seed}
        options.update(self.extra_params)
        return {
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": options,
        }

    def _messages(self, system_prompt: str, user_prompt: str) -> list:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.few_shot_messages)  # empty unless llm.few_shot_in_messages
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _kwargs(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int]) -> dict:
        kwargs = {
            "model": self.model,
            "messages": self._messages(system_prompt, user_prompt),
            "temperature": self.temperature,
            "seed": self.seed,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        kwargs.update(self.extra_params)  # Ollama-specific options (e.g. num_ctx) and pass-throughs
        return kwargs

    def dry_run_prompt_tokens(self, system_prompt: str, user_prompt: str) -> int:
        """Base-prompt token count via a 1-token generation."""
        if self.transport == TRANSPORT_OLLAMA_GENERATE:
            import requests

            payload = self._ollama_payload(system_prompt, user_prompt)
            payload["options"]["num_predict"] = 1
            resp = requests.post(self._generate_url(), json=payload, timeout=self.request_timeout)
            return int(resp.json().get("prompt_eval_count", 0) or 0)

        import litellm

        resp = litellm.completion(**self._kwargs(system_prompt, user_prompt, max_tokens=1))
        usage = getattr(resp, "usage", None)
        return int(getattr(usage, "prompt_tokens", 0) or 0)

    def classify(self, system_prompt: str, user_prompt: str) -> dict:
        """Final classification request.

        Returns a dict with fields:
          {status_ok: bool, text: str, prompt_eval_count: int, eval_count: int,
           parse_error: bool}
        ``status_ok=False`` maps to the "API error" bucket, and ``parse_error=True``
        maps to the "Error parsing response" bucket.
        """
        if self.transport == TRANSPORT_OLLAMA_GENERATE:
            import requests

            resp = requests.post(self._generate_url(), json=self._ollama_payload(system_prompt, user_prompt),
                                 timeout=self.request_timeout)

            if resp.status_code != 200:
                return {"status_ok": False, "text": "", "prompt_eval_count": 0, "eval_count": 0, "parse_error": False}

            try:
                data = resp.json()
                text = data.get("response", "") or ""
                prompt_eval_count = int(data.get("prompt_eval_count", 0) or 0)
                eval_count = int(data.get("eval_count", 0) or 0)
            except Exception:
                return {"status_ok": True, "text": "", "prompt_eval_count": 0, "eval_count": 0, "parse_error": True}

            return {
                "status_ok": True,
                "text": text,
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
                "parse_error": False,
            }

        import litellm

        try:
            resp = litellm.completion(**self._kwargs(system_prompt, user_prompt, max_tokens=None))
        except Exception:
            return {"status_ok": False, "text": "", "prompt_eval_count": 0, "eval_count": 0, "parse_error": False}

        try:
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            prompt_eval_count = int(getattr(usage, "prompt_tokens", 0) or 0)
            eval_count = int(getattr(usage, "completion_tokens", 0) or 0)
        except Exception:
            return {"status_ok": True, "text": "", "prompt_eval_count": 0, "eval_count": 0, "parse_error": True}

        return {
            "status_ok": True,
            "text": text,
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "parse_error": False,
        }
