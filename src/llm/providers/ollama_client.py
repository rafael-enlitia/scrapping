"""Ollama (local) provider using the OpenAI-compatible /v1 endpoint with backoff."""

from __future__ import annotations

import logging
import time

from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL
from src.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_RETRYABLE = (APITimeoutError, APIConnectionError)
_MAX_RETRIES = 4
_BASE_DELAY = 1.0


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self._model = model or OLLAMA_MODEL
        self._client = OpenAI(
            base_url=f"{base_url or OLLAMA_BASE_URL}/v1",
            api_key="ollama",
        )

    def chat(self, system: str, user: str) -> str:
        delay = _BASE_DELAY
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.2,
                )
                return response.choices[0].message.content or ""
            except _RETRYABLE as exc:
                if attempt == _MAX_RETRIES:
                    raise
                logger.warning(
                    "Ollama transient error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, _MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
            except APIStatusError as exc:
                if exc.status_code in (429, 503):
                    if attempt == _MAX_RETRIES:
                        raise
                    logger.warning(
                        "Ollama %d (attempt %d/%d) — retrying in %.1fs",
                        exc.status_code, attempt, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                else:
                    raise

    @property
    def model_name(self) -> str:
        return self._model
