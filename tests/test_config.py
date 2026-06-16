"""Tests for src.config — environment variable loading and helpers."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestIntEnv:
    def test_valid_integer(self):
        with patch.dict(os.environ, {"SOME_INT": "42"}):
            import src.config as cfg
            assert cfg._int_env("SOME_INT", 10) == 42

    def test_uses_default_when_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "SOME_MISSING"}
        with patch.dict(os.environ, env, clear=True):
            import src.config as cfg
            assert cfg._int_env("SOME_MISSING", 99) == 99

    def test_raises_on_non_integer(self):
        with patch.dict(os.environ, {"BAD_INT": "not_a_number"}):
            import src.config as cfg
            with pytest.raises(ValueError, match="BAD_INT"):
                cfg._int_env("BAD_INT", 5)

    def test_raises_on_float_string(self):
        with patch.dict(os.environ, {"FLOAT_VAL": "3.14"}):
            import src.config as cfg
            with pytest.raises(ValueError, match="FLOAT_VAL"):
                cfg._int_env("FLOAT_VAL", 5)


class TestEnsureDataDir:
    def test_creates_directory(self, tmp_path, monkeypatch):
        import src.config as cfg
        new_dir = tmp_path / "new_data_dir"
        monkeypatch.setattr(cfg, "DATA_DIR", new_dir)

        cfg.ensure_data_dir()
        assert new_dir.exists()

    def test_idempotent(self, tmp_path, monkeypatch):
        import src.config as cfg
        monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)

        # Should not raise even if dir already exists
        cfg.ensure_data_dir()
        cfg.ensure_data_dir()


class TestConfigValues:
    def test_default_app_id_exists(self):
        import src.config as cfg
        assert hasattr(cfg, "DEFAULT_APP_ID")

    def test_scrape_lang_is_string(self):
        import src.config as cfg
        assert isinstance(cfg.SCRAPE_LANG, str)

    def test_lda_num_topics_is_int(self):
        import src.config as cfg
        assert isinstance(cfg.LDA_NUM_TOPICS, int)
        assert cfg.LDA_NUM_TOPICS > 0

    def test_nlp_batch_size_is_int(self):
        import src.config as cfg
        assert isinstance(cfg.NLP_BATCH_SIZE, int)
        assert cfg.NLP_BATCH_SIZE > 0
