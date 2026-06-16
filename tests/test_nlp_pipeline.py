"""Tests for src.nlp.pipeline — classify_batch_nlp with mocked BERT and LDA."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


class TestClassifyBatchNLP:
    def _make_sentiment_result(self, sentiment="positive", confidence=0.85):
        from src.nlp.sentiment import SentimentResult
        return SentimentResult(sentiment=sentiment, confidence=confidence, star_prediction=4)

    def _make_topic_result(self, topic_id=0, words="app fast", labels=None):
        from src.nlp.topics import TopicResult
        return TopicResult(topic_id=topic_id, topic_words=words, mapped_labels=labels or ["performance"])

    def test_classifies_unclassified_reviews(self, db_with_reviews, monkeypatch):
        engine, Session = db_with_reviews

        import src.db.models as models_mod
        monkeypatch.setattr(models_mod, "engine", engine)
        monkeypatch.setattr(models_mod, "get_session", lambda: Session())

        mock_sent_results = [self._make_sentiment_result() for _ in range(20)]
        mock_topic_results = [self._make_topic_result() for _ in range(20)]

        mock_lda = MagicMock()
        mock_lda.load.return_value = True
        mock_lda.predict.return_value = mock_topic_results

        with patch("src.nlp.pipeline.predict_sentiment", return_value=mock_sent_results):
            with patch("src.nlp.pipeline.LDAModel", return_value=mock_lda):
                with patch("src.nlp.pipeline.preprocess_batch", return_value=(["clean"] * 20, ["lda_doc"] * 20)):
                    with patch("src.nlp.pipeline.init_db"):
                        with patch("src.nlp.pipeline.get_session", return_value=Session()):
                            from src.nlp.pipeline import classify_batch_nlp
                            count = classify_batch_nlp(app_id="com.example.app")

        assert count == 20

    def test_no_reviews_returns_zero(self, db_with_classifications, monkeypatch):
        """All reviews already classified by NLP — should return 0."""
        engine, Session = db_with_classifications

        import src.db.models as models_mod
        monkeypatch.setattr(models_mod, "engine", engine)
        monkeypatch.setattr(models_mod, "get_session", lambda: Session())

        with patch("src.nlp.pipeline.init_db"):
            with patch("src.nlp.pipeline.get_session", return_value=Session()):
                from src.nlp.pipeline import classify_batch_nlp
                count = classify_batch_nlp(app_id="com.example.app")

        assert count == 0

    def test_lda_mismatch_triggers_retrain(self, db_with_reviews, monkeypatch):
        engine, Session = db_with_reviews

        import src.db.models as models_mod
        monkeypatch.setattr(models_mod, "engine", engine)
        monkeypatch.setattr(models_mod, "get_session", lambda: Session())

        mock_sent_results = [self._make_sentiment_result() for _ in range(20)]
        mock_topic_results = [self._make_topic_result() for _ in range(20)]

        mock_lda = MagicMock()
        # Simulate mismatch: load returns False (triggers retrain)
        mock_lda.load.return_value = False
        mock_lda.predict.return_value = mock_topic_results

        with patch("src.nlp.pipeline.predict_sentiment", return_value=mock_sent_results):
            with patch("src.nlp.pipeline.LDAModel", return_value=mock_lda):
                with patch("src.nlp.pipeline.preprocess_batch", return_value=(["clean"] * 20, ["lda_doc"] * 20)):
                    with patch("src.nlp.pipeline.init_db"):
                        with patch("src.nlp.pipeline.get_session", return_value=Session()):
                            from src.nlp.pipeline import classify_batch_nlp
                            count = classify_batch_nlp(app_id="com.example.app")

        assert mock_lda.fit.called, "LDA.fit should be called when load() returns False"
        assert mock_lda.save.called, "LDA.save should be called after retraining"
        assert count == 20
