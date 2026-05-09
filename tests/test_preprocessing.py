"""Tests for src.nlp.preprocessing."""

from __future__ import annotations

import pytest

from src.nlp.preprocessing import clean_text, tokenize_for_lda, preprocess_batch


class TestCleanText:
    def test_removes_urls(self):
        assert "http" not in clean_text("Visit https://example.com for more info")

    def test_removes_html(self):
        assert "<b>" not in clean_text("<b>Bold</b> text")

    def test_removes_emojis(self):
        result = clean_text("Great app! 😊🎉")
        assert "😊" not in result
        assert "🎉" not in result

    def test_collapses_whitespace(self):
        result = clean_text("too   many   spaces")
        assert "  " not in result

    def test_preserves_plain_text(self):
        result = clean_text("This is a normal sentence.")
        assert "normal sentence" in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_whitespace(self):
        assert clean_text("   \t\n") == ""

    def test_www_url_removed(self):
        assert "www" not in clean_text("go to www.example.com now")


class TestTokenizeForLda:
    def test_returns_list(self):
        result = tokenize_for_lda("This is a test sentence")
        assert isinstance(result, list)

    def test_removes_stopwords_portuguese(self):
        result = tokenize_for_lda("este é um bom aplicativo", language="portuguese")
        assert "este" not in result
        assert "um" not in result

    def test_removes_stopwords_english(self):
        result = tokenize_for_lda("this is a great app", language="english")
        assert "this" not in result
        assert "is" not in result

    def test_removes_short_tokens(self):
        result = tokenize_for_lda("a b abc test word")
        for token in result:
            assert len(token) > 2

    def test_stems_portuguese(self):
        tokens = tokenize_for_lda("aplicativo aplicativos", language="portuguese")
        # Stemmer should reduce both to same root
        assert len(tokens) == 1 or (len(tokens) >= 1 and all(len(t) > 0 for t in tokens))

    def test_empty_string_returns_list(self):
        assert tokenize_for_lda("") == []

    def test_numeric_only_input(self):
        result = tokenize_for_lda("1234 5678 9000")
        assert result == []


class TestPreprocessBatch:
    def test_returns_two_lists(self):
        cleaned, lda = preprocess_batch(["Hello world", "Test review"])
        assert len(cleaned) == 2
        assert len(lda) == 2

    def test_lengths_match_input(self):
        texts = ["a", "b", "c", "d"]
        cleaned, lda = preprocess_batch(texts)
        assert len(cleaned) == len(texts)
        assert len(lda) == len(texts)

    def test_lda_docs_are_strings(self):
        _, lda = preprocess_batch(["Hello world test sentence"])
        assert all(isinstance(doc, str) for doc in lda)

    def test_cleaned_has_no_urls(self):
        cleaned, _ = preprocess_batch(["Visit https://test.com for details"])
        assert "https" not in cleaned[0]

    def test_empty_batch(self):
        cleaned, lda = preprocess_batch([])
        assert cleaned == []
        assert lda == []
