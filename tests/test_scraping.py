"""Tests for src.scraping.play_store — persistence and filtering."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


def _make_raw_review(i: int, content: str = "Great app!") -> dict:
    return {
        "reviewId": f"gplay-{i:04d}",
        "userName": f"user{i}",
        "content": content,
        "score": (i % 5) + 1,
        "thumbsUpCount": i * 2,
        "reviewCreatedVersion": "1.0.0",
        "at": datetime(2024, 3, 1, tzinfo=timezone.utc),
        "repliedAt": None,
        "replyContent": None,
        "lang": "pt",
    }


class TestPersist:
    def _run_persist(self, Session, raw, app_id="com.example.app"):
        import src.scraping.play_store as ps_mod
        with patch.object(ps_mod, "get_session", return_value=Session()):
            return ps_mod._persist(raw, app_id)

    def test_saves_new_reviews(self, tmp_db):
        _, Session = tmp_db
        from src.db.models import Review

        raw = [_make_raw_review(i) for i in range(5)]
        stats = self._run_persist(Session, raw)

        assert stats["saved"] == 5
        assert stats["duplicates"] == 0
        session = Session()
        assert session.query(Review).count() == 5
        session.close()

    def test_skips_duplicate_reviews(self, tmp_db):
        _, Session = tmp_db

        raw = [_make_raw_review(i) for i in range(3)]
        first = self._run_persist(Session, raw)
        second = self._run_persist(Session, raw)

        assert first["saved"] == 3
        assert second["saved"] == 0
        assert second["duplicates"] == 3

    def test_skips_reviews_without_id(self, tmp_db):
        _, Session = tmp_db

        raw = [{"reviewId": None, "content": "no id"}]
        stats = self._run_persist(Session, raw)
        assert stats["saved"] == 0
        assert stats["no_id"] == 1


class TestScrapeReviews:
    def test_filters_short_reviews(self, tmp_db):
        _, Session = tmp_db

        short = _make_raw_review(99, content="ok")
        valid = _make_raw_review(100, content="This is a decent review")

        import src.scraping.play_store as ps_mod
        from src.db.models import Review

        with patch.object(ps_mod, "get_session", return_value=Session()):
            with patch.object(ps_mod, "init_db"):
                with patch("src.scraping.play_store.reviews", return_value=([short, valid], None)):
                    result = ps_mod.scrape_reviews("com.example.app", count=10)

        session = Session()
        saved = session.query(Review).count()
        session.close()
        assert saved == 1

    def test_respects_count_limit(self, tmp_db):
        _, Session = tmp_db
        many = [_make_raw_review(i, content="Long enough content here") for i in range(50)]

        import src.scraping.play_store as ps_mod

        with patch.object(ps_mod, "get_session", return_value=Session()):
            with patch.object(ps_mod, "init_db"):
                with patch("src.scraping.play_store.reviews", return_value=(many, None)):
                    result = ps_mod.scrape_reviews("com.example.app", count=10)

        assert len(result.reviews) == 10
