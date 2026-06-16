"""OpenAI-compatible API provider with exponential backoff on rate limits."""

from __future__ import annotations

from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.llm.providers.base import LLMProvider, chat_with_backoff

_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._model = model or OPENAI_MODEL
        self._client = OpenAI(api_key=api_key or OPENAI_API_KEY)

    def chat(self, system: str, user: str) -> str:
        return chat_with_backoff(
            client=self._client,
            model=self._model,
            system=system,
            user=user,
            max_retries=5,
            base_delay=2.0,
            max_delay=60.0,
            retryable_exceptions=_RETRYABLE,
            retryable_status_codes=(429, 503),
            json_mode=True,
        )

    @property
    def model_name(self) -> str:
        return self._model
