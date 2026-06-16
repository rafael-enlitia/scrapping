"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# --------------------------------------------------------------------------
# In-memory SQLite database
# --------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Spin up a fresh in-memory SQLite database for each test."""
    db_path = tmp_path / "test.db"
    db_url = f"sqlite:///{db_path}"

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import src.db.models as models_mod

    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    models_mod.Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)

    monkeypatch.setattr(models_mod, "engine", engine)
    monkeypatch.setattr(models_mod, "get_session", lambda: Session())

    return engine, Session


@pytest.fixture()
def db_with_reviews(tmp_db):
    """Populate the test DB with a handful of fixture reviews."""
    engine, Session = tmp_db
    from src.db.models import Review

    session = Session()
    reviews = [
        Review(
            review_id=f"rev-{i:03d}",
            app_id="com.example.app",
            username=f"user{i}",
            content=f"Review number {i}. Great app!" if i % 2 == 0 else f"Review {i}. Terrible experience.",
            score=(i % 5) + 1,
            thumbs_up=i,
            app_version=f"1.{i // 10}.0",
            review_date=datetime(2024, 1, i + 1, tzinfo=timezone.utc),
            language="pt",
        )
        for i in range(20)
    ]
    session.add_all(reviews)
    session.commit()
    session.close()
    return engine, Session


@pytest.fixture()
def db_with_classifications(db_with_reviews):
    """Add LLM + NLP classifications on top of reviews."""
    engine, Session = db_with_reviews
    from src.db.models import Classification

    session = Session()
    for i in range(20):
        rid = f"rev-{i:03d}"
        sentiment = "positive" if i % 2 == 0 else "negative"
        # LLM classification — topics use valid taxonomy values
        session.add(Classification(
            review_id=rid,
            method="llm",
            sentiment=sentiment,
            confidence=0.9,
            topics=json.dumps(["performance", "ui_ux"] if i % 3 == 0 else ["customer_support"]),
            justification="Test justification",
            model_name="gpt-4o-mini",
            classified_at=datetime.now(timezone.utc),
        ))
        # NLP classification
        session.add(Classification(
            review_id=rid,
            method="nlp",
            sentiment=sentiment,
            confidence=0.75,
            topics=json.dumps(["performance"] if i % 3 == 0 else ["bugs"]),
            model_name="bert-base-multilingual",
            lda_topic_id=i % 5,
            lda_topic_words=f"word{i} word{i+1}",
            classified_at=datetime.now(timezone.utc),
        ))
    session.commit()
    session.close()
    return engine, Session


@pytest.fixture()
def mock_llm_provider():
    """A mock LLMProvider that returns a well-formed JSON response."""
    provider = MagicMock()
    provider.model_name = "mock-model"
    provider.chat.return_value = json.dumps({
        "sentiment": "positive",
        "topics": ["performance", "ui_ux"],
        "confidence": 0.95,
        "justification": "The review is very positive.",
    })
    return provider


@pytest.fixture()
def gold_jsonl_path(tmp_path):
    """Write a small gold.jsonl file and return its path."""
    path = tmp_path / "gold.jsonl"
    entries = [
        {"review_id": "rev-001", "sentiment": "positive", "topics": ["ui_ux"]},
        {"review_id": "rev-002", "sentiment": "negative", "topics": ["bugs"]},
        {"review_id": "rev-003", "sentiment": "neutral", "topics": []},
    ]
    path.write_text("\n".join(json.dumps(e) for e in entries))
    return path
