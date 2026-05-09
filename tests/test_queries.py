"""Tests for src.db.queries — data retrieval helpers."""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest


def _patch_engine(engine, monkeypatch):
    """Redirect queries module to use the test engine."""
    import src.db.queries as q

    monkeypatch.setattr(q, "engine", engine)
    monkeypatch.setattr(q, "_ensure_db", lambda: None)
    # Clear Streamlit cache wrappers
    for fn_name in dir(q):
        fn = getattr(q, fn_name)
        if callable(fn) and hasattr(fn, "clear"):
            fn.clear()


class TestGetReviewsDf:
    def test_returns_dataframe(self, db_with_reviews, monkeypatch):
        engine, _ = db_with_reviews
        _patch_engine(engine, monkeypatch)

        from src.db.queries import get_reviews_df

        df = get_reviews_df.__wrapped__("com.example.app", method="llm")
        assert isinstance(df, pd.DataFrame)

    def test_all_reviews_returned(self, db_with_reviews, monkeypatch):
        engine, _ = db_with_reviews
        _patch_engine(engine, monkeypatch)

        from src.db.queries import get_reviews_df

        df = get_reviews_df.__wrapped__("com.example.app", method="llm")
        assert len(df) == 20

    def test_no_duplicate_reviews_in_both_mode(self, db_with_classifications, monkeypatch):
        engine, _ = db_with_classifications
        _patch_engine(engine, monkeypatch)

        from src.db.queries import get_reviews_df

        df = get_reviews_df.__wrapped__("com.example.app", method=None)
        assert df["review_id"].nunique() == len(df), "Duplicate rows detected in 'both' mode"

    def test_both_mode_has_topics_column(self, db_with_classifications, monkeypatch):
        engine, _ = db_with_classifications
        _patch_engine(engine, monkeypatch)

        from src.db.queries import get_reviews_df

        df = get_reviews_df.__wrapped__("com.example.app", method=None)
        assert "topics" in df.columns
        assert "topics_llm" not in df.columns
        assert "topics_nlp" not in df.columns

    def test_app_id_filter(self, db_with_reviews, monkeypatch):
        engine, _ = db_with_reviews
        _patch_engine(engine, monkeypatch)

        from src.db.queries import get_reviews_df

        df = get_reviews_df.__wrapped__("com.other.app", method="llm")
        assert len(df) == 0

    def test_returns_all_when_no_app_id(self, db_with_reviews, monkeypatch):
        engine, _ = db_with_reviews
        _patch_engine(engine, monkeypatch)

        from src.db.queries import get_reviews_df

        df = get_reviews_df.__wrapped__(None, method="llm")
        assert len(df) == 20


class TestGetVersions:
    def test_semver_sorted(self, db_with_reviews, monkeypatch):
        engine, _ = db_with_reviews
        _patch_engine(engine, monkeypatch)

        from src.db.queries import get_versions

        versions = get_versions.__wrapped__("com.example.app")
        assert isinstance(versions, list)
        # Should be sorted newest first
        if len(versions) >= 2:
            def to_parts(v):
                return [int(x) for x in v.split(".") if x.isdigit()]
            assert to_parts(versions[0]) >= to_parts(versions[-1])


class TestSentimentByVersion:
    def test_returns_dataframe(self, db_with_classifications, monkeypatch):
        engine, _ = db_with_classifications
        _patch_engine(engine, monkeypatch)

        from src.db.queries import sentiment_by_version

        df = sentiment_by_version.__wrapped__("com.example.app", method="llm")
        assert isinstance(df, pd.DataFrame)
        assert "sentiment" in df.columns
        assert "count" in df.columns

    def test_both_mode_no_duplicates(self, db_with_classifications, monkeypatch):
        engine, _ = db_with_classifications
        _patch_engine(engine, monkeypatch)

        from src.db.queries import sentiment_by_version

        df = sentiment_by_version.__wrapped__("com.example.app", method=None)
        assert isinstance(df, pd.DataFrame)


class TestAgreementRate:
    def test_returns_dict_keys(self, db_with_classifications, monkeypatch):
        engine, _ = db_with_classifications
        _patch_engine(engine, monkeypatch)

        from src.db.queries import agreement_rate

        result = agreement_rate.__wrapped__("com.example.app")
        assert "total" in result
        assert "agreed" in result
        assert "rate" in result

    def test_rate_between_0_and_100(self, db_with_classifications, monkeypatch):
        engine, _ = db_with_classifications
        _patch_engine(engine, monkeypatch)

        from src.db.queries import agreement_rate

        result = agreement_rate.__wrapped__("com.example.app")
        assert 0.0 <= result["rate"] <= 100.0

    def test_empty_app_returns_zero(self, db_with_reviews, monkeypatch):
        engine, _ = db_with_reviews
        _patch_engine(engine, monkeypatch)

        from src.db.queries import agreement_rate

        result = agreement_rate.__wrapped__("com.nonexistent")
        assert result["total"] == 0
        assert result["rate"] == 0.0


class TestParseTopicsCell:
    def test_parses_json_list(self):
        from src.db.queries import _parse_topics_cell

        assert _parse_topics_cell('["performance", "ui"]') == ["performance", "ui"]

    def test_handles_none(self):
        from src.db.queries import _parse_topics_cell

        assert _parse_topics_cell(None) == []

    def test_handles_nan(self):
        import math
        from src.db.queries import _parse_topics_cell

        assert _parse_topics_cell(float("nan")) == []

    def test_handles_list_directly(self):
        from src.db.queries import _parse_topics_cell

        assert _parse_topics_cell(["a", "b"]) == ["a", "b"]

    def test_handles_invalid_json(self):
        from src.db.queries import _parse_topics_cell

        assert _parse_topics_cell("not valid json{") == []


class TestMergeTopicsJson:
    def test_merges_without_duplicates(self):
        from src.db.queries import _merge_topics_json

        result = json.loads(_merge_topics_json('["a", "b"]', '["b", "c"]'))
        assert result == ["a", "b", "c"]

    def test_handles_nones(self):
        from src.db.queries import _merge_topics_json

        result = json.loads(_merge_topics_json(None, None))
        assert result == []

    def test_preserves_order(self):
        from src.db.queries import _merge_topics_json

        result = json.loads(_merge_topics_json('["z", "a"]', '["m"]'))
        assert result[0] == "z"
