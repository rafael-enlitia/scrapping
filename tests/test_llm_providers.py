"""Tests for LLM provider clients — backoff, retries and basic behaviour."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest


# --------------------------------------------------------------------------
# OpenAI provider
# --------------------------------------------------------------------------

class TestOpenAIProvider:
    def _make_provider(self, api_key="test-key", model="gpt-4o-mini"):
        from src.llm.providers.openai_client import OpenAIProvider
        with patch("src.llm.providers.openai_client.OpenAI"):
            p = OpenAIProvider(api_key=api_key, model=model)
        return p

    def test_model_name(self):
        p = self._make_provider(model="gpt-4o")
        assert p.model_name == "gpt-4o"

    def test_chat_returns_content(self):
        from src.llm.providers.openai_client import OpenAIProvider
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "hello"

        with patch("src.llm.providers.openai_client.OpenAI", return_value=mock_client):
            p = OpenAIProvider(api_key="k", model="m")

        result = p.chat("system", "user")
        assert result == "hello"

    def test_retries_on_rate_limit(self):
        from openai import RateLimitError
        from src.llm.providers.openai_client import OpenAIProvider

        mock_client = MagicMock()
        success_response = MagicMock()
        success_response.choices[0].message.content = "ok"

        # Fail twice, then succeed
        rate_err = RateLimitError.__new__(RateLimitError)
        mock_client.chat.completions.create.side_effect = [
            rate_err,
            rate_err,
            success_response,
        ]

        with patch("src.llm.providers.openai_client.OpenAI", return_value=mock_client):
            with patch("src.llm.providers.openai_client.time.sleep"):
                p = OpenAIProvider(api_key="k", model="m")
                result = p.chat("sys", "usr")

        assert result == "ok"
        assert mock_client.chat.completions.create.call_count == 3

    def test_raises_after_max_retries(self):
        from openai import RateLimitError
        from src.llm.providers.openai_client import _MAX_RETRIES, OpenAIProvider

        mock_client = MagicMock()
        rate_err = RateLimitError.__new__(RateLimitError)
        mock_client.chat.completions.create.side_effect = rate_err

        with patch("src.llm.providers.openai_client.OpenAI", return_value=mock_client):
            with patch("src.llm.providers.openai_client.time.sleep"):
                p = OpenAIProvider(api_key="k", model="m")
                with pytest.raises(RateLimitError):
                    p.chat("sys", "usr")

        assert mock_client.chat.completions.create.call_count == _MAX_RETRIES

    def test_delay_doubles_on_each_retry(self):
        from openai import APITimeoutError
        from src.llm.providers.openai_client import OpenAIProvider

        mock_client = MagicMock()
        timeout_err = APITimeoutError.__new__(APITimeoutError)
        success = MagicMock()
        success.choices[0].message.content = "done"
        mock_client.chat.completions.create.side_effect = [timeout_err, timeout_err, success]

        sleep_calls = []
        with patch("src.llm.providers.openai_client.OpenAI", return_value=mock_client):
            with patch("src.llm.providers.openai_client.time.sleep", side_effect=lambda d: sleep_calls.append(d)):
                p = OpenAIProvider(api_key="k", model="m")
                p.chat("s", "u")

        assert len(sleep_calls) == 2
        assert sleep_calls[1] > sleep_calls[0], "Delay should increase (backoff)"


# --------------------------------------------------------------------------
# Ollama provider
# --------------------------------------------------------------------------

class TestOllamaProvider:
    def test_model_name(self):
        from src.llm.providers.ollama_client import OllamaProvider
        with patch("src.llm.providers.ollama_client.OpenAI"):
            p = OllamaProvider(model="llama3")
        assert p.model_name == "llama3"

    def test_chat_success(self):
        from src.llm.providers.ollama_client import OllamaProvider

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices[0].message.content = "response"

        with patch("src.llm.providers.ollama_client.OpenAI", return_value=mock_client):
            p = OllamaProvider(model="llama3")

        assert p.chat("sys", "usr") == "response"

    def test_retries_on_connection_error(self):
        from openai import APIConnectionError
        from src.llm.providers.ollama_client import OllamaProvider

        mock_client = MagicMock()
        conn_err = APIConnectionError.__new__(APIConnectionError)
        success = MagicMock()
        success.choices[0].message.content = "ok"
        mock_client.chat.completions.create.side_effect = [conn_err, success]

        with patch("src.llm.providers.ollama_client.OpenAI", return_value=mock_client):
            with patch("src.llm.providers.ollama_client.time.sleep"):
                p = OllamaProvider(model="llama3")
                result = p.chat("s", "u")

        assert result == "ok"
        assert mock_client.chat.completions.create.call_count == 2
