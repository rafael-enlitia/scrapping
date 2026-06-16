"""Tests for src.llm.classifier — batch classification logic."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError


class TestClassifyReview:
    """Unit tests for the single-review classification helper."""

    def test_returns_classification_schema(self, mock_llm_provider):
        from src.llm.classifier import classify_review
        from src.llm.schemas import ReviewClassification

        result = classify_review(mock_llm_provider, "This app is great!")
        assert isinstance(result, ReviewClassification)
        assert result.sentiment in ("positive", "negative", "neutral", "mixed")

    def test_positive_review(self, mock_llm_provider):
        from src.llm.classifier import classify_review

        result = classify_review(mock_llm_provider, "I love this app, it is fantastic!")
        assert result.sentiment == "positive"

    def test_topics_are_list(self, mock_llm_provider):
        from src.llm.classifier import classify_review

        result = classify_review(mock_llm_provider, "Great app!")
        assert isinstance(result.topics, list)

    def test_confidence_returned(self, mock_llm_provider):
        from src.llm.classifier import classify_review

        result = classify_review(mock_llm_provider, "Great app!")
        assert result.confidence is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_retries_on_invalid_json(self):
        from src.llm.classifier import classify_review
        from src.llm.schemas import ReviewClassification

        provider = MagicMock()
        provider.model_name = "test-model"
        # First response is bad JSON, second is valid
        provider.chat.side_effect = [
            "this is not json {{{",
            json.dumps({
                "sentiment": "negative",
                "topics": ["bugs"],
                "confidence": 0.8,
                "justification": "Bugs everywhere.",
            }),
        ]

        result = classify_review(provider, "Buggy app", max_retries=2)
        assert isinstance(result, ReviewClassification)
        assert result.sentiment == "negative"

    def test_raises_after_max_retries(self):
        from src.llm.classifier import classify_review

        provider = MagicMock()
        provider.model_name = "bad-model"
        provider.chat.return_value = "NOT JSON AT ALL"

        with pytest.raises((json.JSONDecodeError, ValidationError, ValueError)):
            classify_review(provider, "Any review", max_retries=2)

    def test_max_retries_must_be_at_least_one(self, mock_llm_provider):
        from src.llm.classifier import classify_review

        with pytest.raises(ValueError, match="max_retries"):
            classify_review(mock_llm_provider, "text", max_retries=0)

    def test_empty_text_raises(self, mock_llm_provider):
        from src.llm.classifier import classify_review

        with pytest.raises(ValueError):
            classify_review(mock_llm_provider, "   ", max_retries=1)


class TestClassifyBatch:
    """Integration-style tests for classify_batch using the test DB."""

    def test_classifies_unclassified_reviews(self, db_with_reviews, monkeypatch):
        engine, Session = db_with_reviews

        import src.db.models as models_mod
        monkeypatch.setattr(models_mod, "engine", engine)
        monkeypatch.setattr(models_mod, "get_session", lambda: Session())

        mock_provider = MagicMock()
        mock_provider.model_name = "mock-model"
        mock_provider.chat.return_value = json.dumps({
            "sentiment": "positive",
            "topics": ["ui_ux"],
            "confidence": 0.9,
            "justification": "Good app.",
        })

        with patch("src.llm.classifier.get_provider", return_value=mock_provider):
            with patch("src.llm.classifier.init_db"):
                with patch("src.llm.classifier.get_session", return_value=Session()):
                    from src.llm.classifier import classify_batch
                    count = classify_batch(app_id="com.example.app", limit=5)

        assert count == 5

    def test_skips_already_classified(self, db_with_classifications, monkeypatch):
        engine, Session = db_with_classifications

        import src.db.models as models_mod
        monkeypatch.setattr(models_mod, "engine", engine)
        monkeypatch.setattr(models_mod, "get_session", lambda: Session())

        mock_provider = MagicMock()
        mock_provider.model_name = "mock-model"
        mock_provider.chat.return_value = json.dumps({
            "sentiment": "neutral",
            "topics": [],
            "confidence": 0.5,
            "justification": "Meh.",
        })

        with patch("src.llm.classifier.get_provider", return_value=mock_provider):
            with patch("src.llm.classifier.init_db"):
                with patch("src.llm.classifier.get_session", return_value=Session()):
                    from src.llm.classifier import classify_batch
                    count = classify_batch(app_id="com.example.app")

        # All 20 reviews are already classified — should classify 0
        assert count == 0
