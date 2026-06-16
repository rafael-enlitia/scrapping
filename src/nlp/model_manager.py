"""Shared BERT model manager — prevents loading duplicate models in the same process."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class _BertModelManager:
    """Singleton that lazily loads and caches BERT models and the shared tokenizer."""

    def __init__(self):
        self._tokenizer = None
        self._sentiment_model = None
        self._embedding_model = None
        self._device = None

    def _get_device(self):
        if self._device is None:
            import torch  # noqa: PLC0415
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def get_tokenizer(self):
        """Return the shared BERT tokenizer, loading it on first call."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer  # noqa: PLC0415
            from src.config import BERT_MODEL  # noqa: PLC0415
            logger.info("Loading BERT tokenizer: %s", BERT_MODEL)
            self._tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        return self._tokenizer

    def get_sentiment_model(self):
        """Return the BERT classification model for sentiment, loading on first call."""
        if self._sentiment_model is None:
            from transformers import AutoModelForSequenceClassification  # noqa: PLC0415
            from src.config import BERT_MODEL  # noqa: PLC0415
            logger.info("Loading BERT sentiment model: %s", BERT_MODEL)
            self._sentiment_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL)
            device = self._get_device()
            self._sentiment_model.to(device)
            self._sentiment_model.eval()
        return self._sentiment_model, self._get_device()

    def get_embedding_model(self):
        """Return the base BERT model for mean-pool embeddings, loading on first call."""
        if self._embedding_model is None:
            from transformers import AutoModel  # noqa: PLC0415
            from src.config import BERT_MODEL  # noqa: PLC0415
            logger.info("Loading base BERT for embeddings: %s", BERT_MODEL)
            self._embedding_model = AutoModel.from_pretrained(BERT_MODEL)
            device = self._get_device()
            self._embedding_model.to(device)
            self._embedding_model.eval()
        return self._embedding_model, self._get_device()


model_manager = _BertModelManager()
