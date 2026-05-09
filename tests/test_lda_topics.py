"""Tests for src.nlp.topics — LDA topic modelling."""

from __future__ import annotations

import pytest


SAMPLE_DOCS = [
    "aplicativo travando muito lento performance problema",
    "interface bonita design visual gráfico",
    "suporte atendimento cliente resposta ajuda",
    "crash erro falha problema bug aplicativo",
    "rápido fluente ótimo excelente performance",
    "lento demais travando frequentemente problema",
    "design interface visual bonita layout",
    "atendimento rápido suporte excelente cliente",
    "bug crash travamento erro frequente",
    "ótima performance rápido excelente aplicativo",
    "interface confusa layout difícil navegar",
    "suporte demorado atendimento ruim cliente",
]


class TestLDAModel:
    def test_fit_and_predict(self):
        from src.nlp.topics import LDAModel

        lda = LDAModel(n_topics=3)
        lda.fit(SAMPLE_DOCS)
        results = lda.predict(SAMPLE_DOCS[:3])
        assert len(results) == 3

    def test_predict_returns_named_tuple_fields(self):
        from src.nlp.topics import LDAModel

        lda = LDAModel(n_topics=3)
        lda.fit(SAMPLE_DOCS)
        result = lda.predict(SAMPLE_DOCS[:1])[0]

        assert hasattr(result, "topic_id")
        assert hasattr(result, "topic_words")
        assert hasattr(result, "mapped_labels")

    def test_topic_id_in_range(self):
        from src.nlp.topics import LDAModel

        lda = LDAModel(n_topics=4)
        lda.fit(SAMPLE_DOCS)
        for result in lda.predict(SAMPLE_DOCS):
            assert 0 <= result.topic_id < 4

    def test_small_corpus_guard(self):
        from src.nlp.topics import LDAModel

        lda = LDAModel(n_topics=3)
        with pytest.raises(ValueError, match="at least"):
            lda.fit(["short doc"] * 5)  # less than 10 docs

    def test_save_and_load(self, tmp_path):
        from src.nlp.topics import LDAModel

        lda = LDAModel(n_topics=3, model_path=str(tmp_path / "lda.pkl"))
        lda.fit(SAMPLE_DOCS)
        lda.save()

        lda2 = LDAModel(n_topics=3, model_path=str(tmp_path / "lda.pkl"))
        loaded = lda2.load()
        assert loaded is True

        results = lda2.predict(SAMPLE_DOCS[:2])
        assert len(results) == 2

    def test_mapped_labels_are_strings(self):
        from src.nlp.topics import LDAModel

        lda = LDAModel(n_topics=3)
        lda.fit(SAMPLE_DOCS)
        for result in lda.predict(SAMPLE_DOCS[:5]):
            assert isinstance(result.mapped_labels, list)
            for label in result.mapped_labels:
                assert isinstance(label, str)
