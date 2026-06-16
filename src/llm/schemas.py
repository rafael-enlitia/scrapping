"""Pydantic models for LLM classification output."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, field_validator

from src.llm.taxonomy import SENTIMENT_VALUES, TOPIC_VALUES

TOPIC_ALIASES: dict[str, str] = {
    "security": "privacy_security",
    "privacy": "privacy_security",
    "data_privacy": "privacy_security",
    "ads": "pricing",
    "ad": "pricing",
    "advertisements": "pricing",
    "cost": "pricing",
    "subscription": "pricing",
    "design": "ui_ux",
    "ui": "ui_ux",
    "ux": "ui_ux",
    "interface": "ui_ux",
    "navigation": "ui_ux",
    "layout": "ui_ux",
    "crash": "bugs",
    "crashes": "bugs",
    "glitch": "bugs",
    "glitches": "bugs",
    "error": "bugs",
    "errors": "bugs",
    "bug": "bugs",
    "speed": "performance",
    "lag": "performance",
    "battery": "performance",
    "slow": "performance",
    "support": "customer_support",
    "help": "customer_support",
    "update": "updates",
    "version": "updates",
    "ease_of_use": "usability",
    "accessibility": "usability",
    "feature": "features",
    "feature_request": "features",
}

SENTIMENT_ALIASES: dict[str, str] = {
    "pos": "positive",
    "neg": "negative",
    "neut": "neutral",
    "mix": "mixed",
}

_VALID_TOPICS = frozenset(TOPIC_VALUES)
_VALID_SENTIMENTS = frozenset(SENTIMENT_VALUES)


class ReviewClassification(BaseModel):
    sentiment: str
    topics: list[str]
    justification: str
    confidence: Optional[float] = None

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v: str) -> str:
        v = v.lower().strip()
        v = SENTIMENT_ALIASES.get(v, v)
        if v not in _VALID_SENTIMENTS:
            raise ValueError(f"Invalid sentiment '{v}'. Must be one of {SENTIMENT_VALUES}")
        return v

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, v: list[str]) -> list[str]:
        cleaned = []
        for t in v:
            t = t.lower().strip()
            t = TOPIC_ALIASES.get(t, t)
            if t not in _VALID_TOPICS:
                raise ValueError(f"Invalid topic '{t}'. Must be one of {TOPIC_VALUES}")
            if t not in cleaned:
                cleaned.append(t)
        return cleaned

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float | None) -> float | None:
        if v is None:
            return None
        return max(0.0, min(1.0, float(v)))
