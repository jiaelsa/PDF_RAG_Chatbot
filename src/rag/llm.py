"""LLM access through any OpenAI-compatible chat endpoint.

One small client works with OpenAI, Groq, Google Gemini (OpenAI-compat endpoint),
OpenRouter, Together, or a local Ollama / LM Studio server -- only the three
environment variables below change. See ``.env.example``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .config import ROOT_DIR  # noqa: F401  (also loads the .env file)


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 700

    @property
    def configured(self) -> bool:
        local = any(h in self.base_url for h in ("localhost", "127.0.0.1"))
        return bool(self.model) and (bool(self.api_key) or local)


def get_llm_settings() -> LLMSettings:
    return LLMSettings(
        api_key=os.getenv("LLM_API_KEY", "").strip(),
        base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip(),
        model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile").strip(),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "700")),
    )


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, settings: LLMSettings | None = None):
        self.settings = settings or get_llm_settings()
        self._client = None

    @property
    def configured(self) -> bool:
        return self.settings.configured

    def generate(self, messages: list[dict]) -> str:
        if not self.configured:
            raise LLMError("LLM is not configured. Set LLM_API_KEY / LLM_MODEL in .env.")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.api_key or "not-needed",
                base_url=self.settings.base_url,
                timeout=60,
            )
        try:
            resp = self._client.chat.completions.create(
                model=self.settings.model,
                messages=messages,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
        except Exception as exc:  # network, auth, rate limit, bad model name ...
            raise LLMError(f"LLM request failed: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()
