"""Batch classification of reviews using the configured LLM provider."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from pydantic import ValidationError

from src.config import LLM_PROVIDER
from src.db.models import Classification, Review, get_session, init_db
from src.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from src.llm.providers.base import LLMProvider
from src.llm.providers.ollama_client import OllamaProvider
from src.llm.providers.openai_client import OpenAIProvider
from src.llm.schemas import ReviewClassification

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def get_provider(provider_name: str | None = None) -> LLMProvider:
    name = (provider_name or LLM_PROVIDER).lower()
    if name == "openai":
        return OpenAIProvider()
    if name == "ollama":
        return OllamaProvider()
    raise ValueError(f"Unknown LLM provider: {name}")


def _clean_json(raw: str) -> str:
    """Extract and sanitize JSON from LLM output."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # Try to extract JSON object if surrounded by extra text
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    return cleaned


def classify_review(
    provider: LLMProvider,
    review_text: str,
    app_version: str | None = None,
    max_retries: int = MAX_RETRIES,
) -> ReviewClassification:
    """Classify a single review with retries on parse/validation errors."""
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    user_prompt = build_user_prompt(review_text, app_version)
    last_error: Exception = RuntimeError("No attempts made")

    for attempt in range(max_retries):
        raw = provider.chat(SYSTEM_PROMPT, user_prompt)
        try:
            cleaned = _clean_json(raw)
            parsed = json.loads(cleaned)
            return ReviewClassification(**parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                logger.debug("Retry %d for review (error: %s)", attempt + 1, exc)
    raise last_error


def classify_batch(
    limit: int | None = None,
    app_id: str | None = None,
    provider_name: str | None = None,
    retry_failed: bool = False,
) -> int:
    """Classify unprocessed reviews and store results. Returns count of new classifications.

    When ``retry_failed`` is True, previously-failed reviews (those with an existing
    LLM classification row) are **deleted** so they can be re-classified fresh.
    """
    init_db()
    session = get_session()
    provider = get_provider(provider_name)

    if retry_failed:
        failed_subq = (
            session.query(Classification.review_id)
            .filter(Classification.method == "llm")
        )
        if app_id:
            failed_subq = failed_subq.join(Review, Review.review_id == Classification.review_id).filter(
                Review.app_id == app_id
            )
        existing_ids = [r[0] for r in failed_subq.all()]
        if existing_ids:
            session.query(Classification).filter(
                Classification.review_id.in_(existing_ids),
                Classification.method == "llm",
            ).delete(synchronize_session=False)
            session.commit()
            logger.info("--retry-failed: cleared %d existing LLM classifications for retry", len(existing_ids))

    query = (
        session.query(Review)
        .outerjoin(
            Classification,
            (Review.review_id == Classification.review_id)
            & (Classification.method == "llm"),
        )
        .filter(Classification.id.is_(None))
    )
    if app_id:
        query = query.filter(Review.app_id == app_id)
    query = query.order_by(Review.review_date.desc())
    if limit:
        query = query.limit(limit)

    reviews = query.all()
    total = len(reviews)
    logger.info("Found %d unclassified reviews", total)

    classified = 0
    failed = 0
    for i, review in enumerate(reviews, 1):
        try:
            result = classify_review(provider, review.content, review.app_version)
            session.add(
                Classification(
                    review_id=review.review_id,
                    method="llm",
                    sentiment=result.sentiment,
                    topics=result.topics,
                    justification=result.justification,
                    model_name=provider.model_name,
                    raw_response=result.model_dump_json(),
                    classified_at=datetime.now(timezone.utc),
                )
            )
            session.commit()
            classified += 1
            logger.info("[%d/%d] Classified %s", i, total, review.review_id)
        except Exception as exc:
            session.rollback()
            failed += 1
            logger.warning("[%d/%d] Failed %s: %s", i, total, review.review_id, exc)

    session.close()
    logger.info("Done: %d classified, %d failed out of %d", classified, failed, total)
    return classified
