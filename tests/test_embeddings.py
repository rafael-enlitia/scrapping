"""Tests for src.nlp.embeddings — embedding computation, clustering and persistence."""

from __future__ import annotations

import numpy as np
import pytest


class TestClusterAndReduce:
    def _make_embeddings(self, n: int = 30, dim: int = 16) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal((n, dim)).astype(np.float32)

    def test_returns_correct_shapes(self):
        from src.nlp.embeddings import cluster_and_reduce

        emb = self._make_embeddings(30)
        umap_2d, labels = cluster_and_reduce(emb, n_clusters=4)

        assert umap_2d.shape == (30, 2)
        assert labels.shape == (30,)

    def test_label_count_matches_n_clusters(self):
        from src.nlp.embeddings import cluster_and_reduce

        emb = self._make_embeddings(40)
        _, labels = cluster_and_reduce(emb, n_clusters=5)
        assert len(set(labels.tolist())) == 5

    def test_umap_is_float32(self):
        from src.nlp.embeddings import cluster_and_reduce

        emb = self._make_embeddings()
        umap_2d, _ = cluster_and_reduce(emb)
        assert umap_2d.dtype == np.float32

    def test_labels_are_int32(self):
        from src.nlp.embeddings import cluster_and_reduce

        emb = self._make_embeddings()
        _, labels = cluster_and_reduce(emb)
        assert labels.dtype == np.int32


class TestSaveAndLoadEmbeddings:
    def _make_result(self, n: int = 10):
        rng = np.random.default_rng(0)
        return (
            [f"rev-{i:03d}" for i in range(n)],
            rng.standard_normal((n, 8)).astype(np.float32),
            rng.standard_normal((n, 2)).astype(np.float32),
            np.arange(n, dtype=np.int32) % 3,
        )

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        from src.nlp import embeddings as emb_mod

        monkeypatch.setattr(emb_mod, "_EMBEDDINGS_PATH", tmp_path / "emb.npz")

        ids, vecs, umap_2d, labels = self._make_result()

        emb_mod.save_embeddings(ids, vecs, umap_2d, labels)
        result = emb_mod.load_embeddings()

        assert result is not None
        assert result.review_ids == ids
        assert np.allclose(result.embeddings, vecs, atol=1e-5)
        assert np.allclose(result.umap_2d, umap_2d, atol=1e-5)
        assert (result.cluster_labels == labels).all()

    def test_load_returns_none_when_no_file(self, tmp_path, monkeypatch):
        from src.nlp import embeddings as emb_mod

        monkeypatch.setattr(emb_mod, "_EMBEDDINGS_PATH", tmp_path / "nonexistent.npz")
        result = emb_mod.load_embeddings()
        assert result is None

    def test_save_creates_file(self, tmp_path, monkeypatch):
        from src.nlp import embeddings as emb_mod
        from src.config import ensure_data_dir

        path = tmp_path / "emb.npz"
        monkeypatch.setattr(emb_mod, "_EMBEDDINGS_PATH", path)
        monkeypatch.setattr("src.nlp.embeddings.DATA_DIR", tmp_path)

        ids, vecs, umap_2d, labels = self._make_result()
        emb_mod.save_embeddings(ids, vecs, umap_2d, labels)

        assert path.exists()
