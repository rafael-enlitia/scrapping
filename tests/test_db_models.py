"""Tests for src.db.models — ORM model creation and constraints."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError


class TestReviewModel:
    def test_create_review(self, db_with_reviews):
        _, Session = db_with_reviews
        from src.db.models import Review

        session = Session()
        reviews = session.query(Review).all()
        session.close()
        assert len(reviews) == 20

    def test_review_fields(self, db_with_reviews):
        _, Session = db_with_reviews
        from src.db.models import Review

        session = Session()
        review = session.query(Review).filter_by(review_id="rev-000").first()
        session.close()

        assert review is not None
        assert review.app_id == "com.example.app"
        assert review.score in range(1, 6)
        assert review.language == "pt"

    def test_duplicate_review_id_raises(self, db_with_reviews):
        _, Session = db_with_reviews
        from src.db.models import Review

        session = Session()
        dupe = Review(
            review_id="rev-000",
            app_id="com.example.app",
            username="dupe_user",
            content="Duplicate review",
            score=3,
            review_date=datetime.now(timezone.utc),
        )
        session.add(dupe)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        session.close()

    def test_review_requires_content(self, tmp_db):
        _, Session = tmp_db
        from src.db.models import Review

        session = Session()
        bad = Review(review_id="no-content-rev", app_id="com.x", score=1)
        session.add(bad)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        session.close()


class TestClassificationModel:
    def test_create_classifications(self, db_with_classifications):
        _, Session = db_with_classifications
        from src.db.models import Classification

        session = Session()
        count = session.query(Classification).count()
        session.close()
        assert count == 40  # 20 reviews × 2 methods

    def test_composite_unique_constraint(self, db_with_classifications):
        """Inserting a second LLM classification for the same review should fail."""
        _, Session = db_with_classifications
        from src.db.models import Classification

        session = Session()
        dupe = Classification(
            review_id="rev-000",
            method="llm",
            sentiment="neutral",
            confidence=0.5,
            topics=json.dumps([]),
        )
        session.add(dupe)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        session.close()

    def test_llm_and_nlp_coexist(self, db_with_classifications):
        _, Session = db_with_classifications
        from src.db.models import Classification

        session = Session()
        for rid in ["rev-000", "rev-001"]:
            methods = (
                session.query(Classification.method)
                .filter_by(review_id=rid)
                .all()
            )
            method_set = {m[0] for m in methods}
            assert "llm" in method_set
            assert "nlp" in method_set
        session.close()

    def test_topics_stored_as_json(self, db_with_classifications):
        _, Session = db_with_classifications
        from src.db.models import Classification

        session = Session()
        clf = session.query(Classification).filter_by(review_id="rev-000", method="llm").first()
        session.close()

        assert clf is not None
        parsed = json.loads(clf.topics)
        assert isinstance(parsed, list)

    def test_error_msg_and_failed_at_are_nullable(self, tmp_db):
        _, Session = tmp_db
        from src.db.models import Classification, Review

        session = Session()
        session.add(Review(
            review_id="r-test",
            app_id="com.x",
            content="Test review",
            score=3,
        ))
        session.flush()
        clf = Classification(
            review_id="r-test",
            method="llm",
            sentiment=None,
            topics=None,
            error_msg="Something went wrong",
            failed_at=datetime.now(timezone.utc),
        )
        session.add(clf)
        session.commit()

        loaded = session.query(Classification).filter_by(review_id="r-test").first()
        assert loaded.error_msg == "Something went wrong"
        assert loaded.failed_at is not None
        session.close()
