"""Tests for src.nlp.sentiment — predict_sentiment with mocked BERT."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestPredictSentiment:
    def _mock_manager(self, star_idx: int = 3, confidence: float = 0.85):
        """Return a mock model_manager that simulates BERT predictions."""
        import torch

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        mock_tokenizer.return_value.to = MagicMock(return_value=mock_tokenizer.return_value)

        # Build a fake logits tensor: star_idx has the highest probability
        logits = torch.full((1, 5), -10.0)
        logits[0, star_idx] = 10.0  # force argmax to star_idx

        mock_model = MagicMock()
        mock_model.return_value.logits = logits

        mock_device = torch.device("cpu")

        mock_manager = MagicMock()
        mock_manager.get_tokenizer.return_value = mock_tokenizer
        mock_manager.get_sentiment_model.return_value = (mock_model, mock_device)
        return mock_manager

    def test_returns_one_result_per_input(self):
        from src.nlp.sentiment import predict_sentiment

        with patch("src.nlp.sentiment.model_manager", self._mock_manager(star_idx=4)):
            results = predict_sentiment(["Good app", "Bad app"])

        assert len(results) == 2

    def test_positive_for_5_star(self):
        from src.nlp.sentiment import predict_sentiment

        with patch("src.nlp.sentiment.model_manager", self._mock_manager(star_idx=4)):
            results = predict_sentiment(["Absolutely fantastic!"])

        assert results[0].sentiment == "positive"
        assert results[0].star_prediction == 5

    def test_negative_for_1_star(self):
        from src.nlp.sentiment import predict_sentiment

        with patch("src.nlp.sentiment.model_manager", self._mock_manager(star_idx=0)):
            results = predict_sentiment(["Terrible, crashes all the time."])

        assert results[0].sentiment == "negative"
        assert results[0].star_prediction == 1

    def test_neutral_for_3_star(self):
        from src.nlp.sentiment import predict_sentiment

        with patch("src.nlp.sentiment.model_manager", self._mock_manager(star_idx=2)):
            results = predict_sentiment(["It is ok, nothing special."])

        assert results[0].sentiment == "neutral"

    def test_confidence_in_0_1_range(self):
        from src.nlp.sentiment import predict_sentiment

        with patch("src.nlp.sentiment.model_manager", self._mock_manager(star_idx=4)):
            results = predict_sentiment(["Amazing!"])

        assert 0.0 <= results[0].confidence <= 1.0

    def test_empty_list_returns_empty(self):
        from src.nlp.sentiment import predict_sentiment

        results = predict_sentiment([])
        assert results == []
