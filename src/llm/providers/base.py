"""Abstract base class and shared utilities for LLM providers."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


def chat_with_backoff(
    client,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple = (),
    retryable_status_codes: tuple = (429,),
    json_mode: bool = False,
) -> str:
    """Send a chat completion request with exponential backoff on transient errors.

    Retries on exceptions in ``retryable_exceptions`` and on APIStatusError with
    status codes in ``retryable_status_codes``. All other exceptions propagate immediately.
    Returns the response content string (empty string if model returned nothing).
    """
    from openai import APIStatusError  # noqa: PLC0415

    delay = base_delay
    for attempt in range(1, max_retries + 1):
        try:
            kwargs: dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            choices = response.choices
            if not choices:
                return ""
            return choices[0].message.content or ""

        except retryable_exceptions as exc:
            if attempt == max_retries:
                raise
            logger.warning(
                "Transient error (attempt %d/%d): %s — retrying in %.1fs",
                attempt, max_retries, exc, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, max_delay)

        except APIStatusError as exc:
            if exc.status_code in retryable_status_codes:
                if attempt == max_retries:
                    raise
                logger.warning(
                    "API status %d (attempt %d/%d) — retrying in %.1fs",
                    exc.status_code, attempt, max_retries, delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
            else:
                raise

    return ""  # unreachable but satisfies type checker


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, system: str, user: str) -> str:
        """Send a system + user message pair and return the raw text response."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier being used."""
