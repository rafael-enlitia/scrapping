"""IAEDU agent-chat provider (multipart POST + streaming response)."""

from __future__ import annotations

import json
import logging
import secrets
import time

import httpx

from src.config import IAEDU_API_KEY, IAEDU_CHANNEL_ID, IAEDU_ENDPOINT
from src.llm.json_utils import extract_json_object, strip_markdown_fences
from src.llm.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 502, 503, 504}


def _normalize_endpoint(url: str) -> str:
    """Fix common double-slash in IAEDU URLs (e.g. agent-chat//api)."""
    return url.replace("agent-chat//", "agent-chat/").rstrip("/")


def _parse_stream_body(raw: str) -> str:
    """Parse IAEDU NDJSON stream (type: token | message | done)."""
    token_parts: list[str] = []
    message_json: str | None = None

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue

        event_type = obj.get("type")
        content = obj.get("content")

        if event_type == "message":
            inner = _message_content_text(content)
            if inner:
                stripped = strip_markdown_fences(inner)
                if stripped.startswith("{") or "{" in stripped:
                    message_json = stripped
        elif event_type == "token" and isinstance(content, str):
            token_parts.append(content)

    if message_json:
        if message_json.lstrip().startswith("{"):
            return message_json
        if "{" in message_json:
            return extract_json_object(message_json)
        return message_json

    joined = "".join(token_parts).strip()
    # Drop leading status text before JSON (e.g. "Processing")
    brace = joined.find("{")
    if brace > 0:
        joined = joined[brace:]
    return joined


def _message_content_text(content: object) -> str | None:
    """Extract assistant text from IAEDU message event content."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        inner = content.get("content")
        if isinstance(inner, str):
            return inner
    return None


class IaeduProvider(LLMProvider):
    """POST multipart form to IAEDU agent-chat stream API."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        channel_id: str | None = None,
    ):
        self._api_key = IAEDU_API_KEY if api_key is None else api_key
        self._endpoint = _normalize_endpoint(IAEDU_ENDPOINT if endpoint is None else endpoint)
        self._channel_id = IAEDU_CHANNEL_ID if channel_id is None else channel_id
        if not self._api_key:
            raise ValueError("IAEDU_API_KEY is required when LLM_PROVIDER=iaedu")
        if not self._channel_id:
            raise ValueError("IAEDU_CHANNEL_ID is required when LLM_PROVIDER=iaedu")
        if not self._endpoint:
            raise ValueError("IAEDU_ENDPOINT is required when LLM_PROVIDER=iaedu")

    def chat(self, system: str, user: str) -> str:
        message = f"{system.strip()}\n\n---\n\n{user.strip()}"
        form = {
            "channel_id": (None, self._channel_id),
            "thread_id": (None, secrets.token_urlsafe(16)),
            "user_info": (None, "{}"),
            "message": (None, message),
        }
        headers = {"x-api-key": self._api_key}

        max_retries = 4
        delay = 2.0
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                with httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
                    with client.stream(
                        "POST",
                        self._endpoint,
                        files=form,
                        headers=headers,
                    ) as response:
                        if response.status_code in _RETRYABLE_STATUS:
                            response.read()
                            if attempt == max_retries:
                                response.raise_for_status()
                            logger.warning(
                                "IAEDU HTTP %s (attempt %d/%d) — retrying in %.1fs",
                                response.status_code,
                                attempt,
                                max_retries,
                                delay,
                            )
                            time.sleep(delay)
                            delay = min(delay * 2, 60.0)
                            continue
                        response.raise_for_status()
                        raw = response.read().decode("utf-8", errors="replace")
                text = _parse_stream_body(raw).strip()
                if not text:
                    preview = raw[:800].replace("\n", "\\n")
                    logger.warning(
                        "IAEDU empty parse (HTTP %s, %d bytes). Stream preview: %s",
                        response.status_code,
                        len(raw),
                        preview,
                    )
                    raise ValueError("IAEDU returned an empty response")
                return text
            except httpx.HTTPStatusError:
                raise
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_error = exc
                if attempt == max_retries:
                    raise
                logger.warning(
                    "IAEDU connection error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)

        if last_error:
            raise last_error
        return ""

    @property
    def model_name(self) -> str:
        return "iaedu-agent"
