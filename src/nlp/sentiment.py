"""BERT-based multilingual sentiment classifier."""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import BERT_MODEL, NLP_BATCH_SIZE

logger = logging.getLogger(__name__)

STAR_TO_SENTIMENT = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}


class SentimentResult(NamedTuple):
    sentiment: str
    confidence: float
    star_prediction: int


_tokenizer = None
_model = None


_device: torch.device | None = None


def _load_model():
    global _tokenizer, _model, _device
    if _tokenizer is None:
        logger.info("Loading BERT model: %s", BERT_MODEL)
        _tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL)
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        _model.eval()
    return _tokenizer, _model, _device


def predict_sentiment(texts: list[str], batch_size: int = NLP_BATCH_SIZE) -> list[SentimentResult]:
    """Run BERT sentiment on a list of cleaned texts. Returns one result per input."""
    tokenizer, model, device = _load_model()

    results: list[SentimentResult] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Truncate to model max length (512 tokens)
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)
        stars = probs.argmax(dim=-1) + 1  # model outputs 0-4 → 1-5 stars
        confidences = probs.max(dim=-1).values

        for star, conf in zip(stars.cpu().tolist(), confidences.cpu().tolist()):
            results.append(
                SentimentResult(
                    sentiment=STAR_TO_SENTIMENT[star],
                    confidence=round(conf, 4),
                    star_prediction=star,
                )
            )

        logger.info("BERT batch %d–%d / %d done", i + 1, min(i + batch_size, len(texts)), len(texts))

    return results
