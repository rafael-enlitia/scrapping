"""Tests for IAEDU LLM provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.llm.json_utils import extract_json_object
from src.llm.providers.iaedu_client import (
    IaeduProvider,
    _normalize_endpoint,
    _parse_stream_body,
)

SAMPLE_STREAM = """
{"run_id": "x", "type": "start", "content": "Processing"}
{"run_id": "x", "type": "token", "content": "{\\""}
{"run_id": "x", "type": "token", "content": "sentiment\\""}
{"run_id": "x", "type": "message", "content": {"type": "ai", "content": "{\\"sentiment\\":\\"negative\\",\\"topics\\":[\\"bugs\\"],\\"justification\\":\\"Crashes often\\",\\"confidence\\":0.9}"}}
{"run_id": "x", "type": "done", "content": "x", "messageId": "m1"}
""".strip()


class TestParseStreamBody:
    def test_prefers_message_event_json(self):
        raw = SAMPLE_STREAM
        result = _parse_stream_body(raw)
        assert result.startswith("{")
        assert "negative" in result
        assert "Processing" not in result

    def test_token_only_fallback(self):
        import json as _json

        lines = [
            {"type": "token", "content": '{"'},
            {"type": "token", "content": '"sentiment": "positive"'},
            {"type": "token", "content": "}"},
        ]
        raw = "\n".join(_json.dumps(line) for line in lines)
        result = _parse_stream_body(raw)
        assert "positive" in result


class TestExtractJsonObject:
    def test_duplicate_json_objects(self):
        raw = (
            '{"sentiment":"negative","topics":["bugs"],'
            '"justification":"x","confidence":0.9}'
            '{"sentiment":"negative","topics":["bugs"],'
            '"justification":"x","confidence":0.9}'
        )
        cleaned = extract_json_object(raw)
        parsed = __import__("json").loads(cleaned)
        assert parsed["sentiment"] == "negative"


class TestNormalizeEndpoint:
    def test_fixes_double_slash(self):
        url = "https://api.iaedu.pt/agent-chat//api/v1/agent/abc/stream"
        assert "agent-chat//" not in _normalize_endpoint(url)


class TestIaeduProvider:
    def _provider(self):
        return IaeduProvider(
            api_key="sk-test",
            endpoint="https://api.iaedu.pt/agent-chat/api/v1/agent/test/stream",
            channel_id="channel-123",
        )

    def test_model_name(self):
        assert self._provider().model_name == "iaedu-agent"

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="IAEDU_API_KEY"):
            IaeduProvider(api_key="", channel_id="c", endpoint="https://example.com/stream")

    def test_chat_parses_message_event(self):
        p = self._provider()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.read.return_value = (
            '{"type": "message", "content": {"type": "ai", "content": '
            '"{\\"sentiment\\": \\"positive\\", \\"topics\\": [\\"bugs\\"], '
            '\\"justification\\": \\"ok\\", \\"confidence\\": 0.9}"}}'
        ).encode()

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_response)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream.return_value = mock_stream_ctx

        mock_client_ctx = MagicMock()
        mock_client_ctx.__enter__ = MagicMock(return_value=mock_client)
        mock_client_ctx.__exit__ = MagicMock(return_value=False)

        with patch("src.llm.providers.iaedu_client.httpx.Client", return_value=mock_client_ctx):
            result = p.chat("system", "user")

        assert "positive" in result
        assert result.count('"sentiment"') == 1

    def test_get_provider_factory(self):
        with patch("src.llm.classifier.IaeduProvider") as mock_cls:
            mock_cls.return_value = MagicMock()
            from src.llm.classifier import get_provider

            get_provider("iaedu")
            mock_cls.assert_called_once()
