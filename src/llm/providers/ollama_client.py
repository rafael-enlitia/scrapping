"""Ollama (local) provider using the OpenAI-compatible /v1 endpoint with backoff."""

from __future__ import annotations

from openai import OpenAI, APITimeoutError, APIConnectionError

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from src.llm.providers.base import LLMProvider, chat_with_backoff

_RETRYABLE = (APITimeoutError, APIConnectionError)


def _normalize_base_url(base_url: str) -> str:
    """Ensure the base URL ends with /v1 exactly once."""
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self._model = model or OLLAMA_MODEL
        self._client = OpenAI(
            base_url=_normalize_base_url(base_url or OLLAMA_BASE_URL),
            api_key="ollama",
        )

    def chat(self, system: str, user: str) -> str:
        return chat_with_backoff(
            client=self._client,
            model=self._model,
            system=system,
            user=user,
            max_retries=4,
            base_delay=1.0,
            max_delay=30.0,
            retryable_exceptions=_RETRYABLE,
            retryable_status_codes=(429, 503),
            json_mode=False,
        )

    @property
    def model_name(self) -> str:
        return self._model
