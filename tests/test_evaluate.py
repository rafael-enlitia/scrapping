"""Tests for scripts.evaluate — load_gold and _evaluate_method with synthetic data."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


class TestLoadGold:
    def test_loads_valid_jsonl(self, gold_jsonl_path):
        from scripts.evaluate import load_gold

        items = load_gold(str(gold_jsonl_path))
        assert len(items) == 3
        assert items[0]["review_id"] == "rev-001"
        assert items[0]["sentiment"] == "positive"
        assert items[0]["topics"] == ["ui_ux"]

    def test_skips_empty_lines(self, tmp_path):
        from scripts.evaluate import load_gold

        path = tmp_path / "gold.jsonl"
        path.write_text('\n{"review_id": "r1", "sentiment": "positive", "topics": []}\n\n')

        items = load_gold(str(path))
        assert len(items) == 1

    def test_empty_file_returns_empty_list(self, tmp_path):
        from scripts.evaluate import load_gold

        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert load_gold(str(path)) == []


class TestEvaluateMethod:
    def _make_df(self, review_ids, sentiments):
        """Build a minimal classified DataFrame."""
        return pd.DataFrame({
            "review_id": review_ids,
            "sentiment": sentiments,
            "topics": ['["performance"]'] * len(review_ids),
        })

    def test_perfect_accuracy(self, capsys):
        from scripts.evaluate import _evaluate_method

        gold = [
            {"review_id": "r1", "sentiment": "positive", "topics": ["ui_ux"]},
            {"review_id": "r2", "sentiment": "negative", "topics": ["bugs"]},
        ]
        df = self._make_df(["r1", "r2"], ["positive", "negative"])

        _evaluate_method(gold, df, method_label="llm")

        captured = capsys.readouterr()
        assert "Accuracy: 1.000" in captured.out

    def test_partial_match_reports_missing(self, capsys):
        from scripts.evaluate import _evaluate_method

        gold = [
            {"review_id": "r1", "sentiment": "positive", "topics": []},
            {"review_id": "r-missing", "sentiment": "negative", "topics": []},
        ]
        df = self._make_df(["r1"], ["positive"])

        _evaluate_method(gold, df, method_label="llm")

        captured = capsys.readouterr()
        # Should mention 1 of 2 matched
        assert "1 / 2" in captured.out

    def test_no_match_skips_gracefully(self, capsys):
        from scripts.evaluate import _evaluate_method

        gold = [{"review_id": "nonexistent", "sentiment": "positive", "topics": []}]
        df = self._make_df(["r1"], ["positive"])

        _evaluate_method(gold, df, method_label="llm")

        captured = capsys.readouterr()
        assert "No matching" in captured.out

    def test_nlp_skips_topic_evaluation(self, capsys):
        from scripts.evaluate import _evaluate_method

        gold = [{"review_id": "r1", "sentiment": "positive", "topics": ["ui_ux"]}]
        df = self._make_df(["r1"], ["positive"])

        _evaluate_method(gold, df, method_label="nlp")

        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower()
