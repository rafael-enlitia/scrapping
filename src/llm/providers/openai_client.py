"""OpenAI-compatible API provider with exponential backoff on rate limits."""

from __future__ import annotations

import logging
import time

from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)
_MAX_RETRIES = 5
_BASE_DELAY = 2.0


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._model = model or OPENAI_MODEL
        self._client = OpenAI(api_key=api_key or OPENAI_API_KEY)

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
                    "OpenAI transient error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, _MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
            except APIStatusError as exc:
                if exc.status_code == 429:
                    if attempt == _MAX_RETRIES:
                        raise
                    logger.warning(
                        "OpenAI rate limit 429 (attempt %d/%d) — retrying in %.1fs",
                        attempt, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                else:
                    raise

    @property
    def model_name(self) -> str:
        return self._model
