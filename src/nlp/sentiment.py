"""BERT-based multilingual sentiment classifier."""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch

from src.config import NLP_BATCH_SIZE
from src.nlp.model_manager import model_manager

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


def predict_sentiment(texts: list[str], batch_size: int = NLP_BATCH_SIZE) -> list[SentimentResult]:
    """Run BERT sentiment on a list of cleaned texts. Returns one result per input."""
    if not texts:
        return []

    tokenizer = model_manager.get_tokenizer()
    model, device = model_manager.get_sentiment_model()

    results: list[SentimentResult] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
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
            star = int(star)
            sentiment = STAR_TO_SENTIMENT.get(star, "neutral")
            results.append(
                SentimentResult(
                    sentiment=sentiment,
                    confidence=round(conf, 4),
                    star_prediction=star,
                )
            )

        logger.info("BERT batch %d–%d / %d done", i + 1, min(i + batch_size, len(texts)), len(texts))

    return results
