"""Batch classification of reviews using the configured LLM provider."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from pydantic import ValidationError

from src.config import LLM_PROVIDER
from src.db.models import Classification, Review, get_session, init_db
from src.llm.json_utils import extract_json_object
from src.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from src.llm.providers.base import LLMProvider
from src.llm.providers.iaedu_client import IaeduProvider
from src.llm.providers.ollama_client import OllamaProvider
from src.llm.providers.openai_client import OpenAIProvider
from src.llm.schemas import ReviewClassification

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
_BATCH_COMMIT_SIZE = 50


def get_provider(provider_name: str | None = None) -> LLMProvider:
    name = (provider_name or LLM_PROVIDER).lower()
    if name == "openai":
        return OpenAIProvider()
    if name == "ollama":
        return OllamaProvider()
    if name == "iaedu":
        return IaeduProvider()
    raise ValueError(f"Unknown LLM provider: {name}. Use openai, ollama, or iaedu.")


def _clean_json(raw: str) -> str:
    """Extract and sanitize JSON from LLM output (fallback for providers without JSON mode)."""
    return extract_json_object(raw)


def classify_review(
    provider: LLMProvider,
    review_text: str,
    app_version: str | None = None,
    max_retries: int = MAX_RETRIES,
) -> ReviewClassification:
    """Classify a single review with retries on parse/validation errors."""
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    if not review_text or not review_text.strip():
        raise ValueError("review_text must not be empty")

    user_prompt = build_user_prompt(review_text, app_version)
    last_error: Exception = RuntimeError("No attempts made")

    for attempt in range(max_retries):
        raw = provider.chat(SYSTEM_PROMPT, user_prompt)
        if not raw:
            last_error = ValueError("Empty response from LLM")
            continue
        try:
            cleaned = _clean_json(raw)
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
            return ReviewClassification(**parsed)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
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
    """Classify unprocessed (or previously failed) reviews and store results.

    Returns count of new successful classifications.

    When ``retry_failed`` is True, retries reviews that previously failed (i.e.
    have a classification row with ``error_msg`` set). New unclassified reviews
    (no row at all) are always included in the normal run.
    """
    init_db()
    session = get_session()
    provider = get_provider(provider_name)

    try:
        if retry_failed:
            # Reset failed rows so they are picked up as unclassified in the next query
            failed_ids_q = (
                session.query(Classification.review_id)
                .filter(
                    Classification.method == "llm",
                    Classification.error_msg.isnot(None),
                )
            )
            if app_id:
                failed_ids_q = failed_ids_q.join(
                    Review, Review.review_id == Classification.review_id
                ).filter(Review.app_id == app_id)

            failed_ids = [r[0] for r in failed_ids_q.all()]
            if failed_ids:
                session.query(Classification).filter(
                    Classification.review_id.in_(failed_ids),
                    Classification.method == "llm",
                    Classification.error_msg.isnot(None),
                ).delete(synchronize_session=False)
                session.commit()
                logger.info("--retry-failed: cleared %d failed LLM rows for retry", len(failed_ids))
            else:
                logger.info("--retry-failed: no previously failed reviews found.")

        # Find reviews with no LLM classification row (or cleared ones)
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
        batch: list[Classification] = []

        if total:
            logger.info("Starting LLM classification…")
            logger.info("PROGRESS 0/%d", total)

        for i, review in enumerate(reviews, 1):
            now = datetime.now(timezone.utc)
            try:
                result = classify_review(provider, review.content, review.app_version)
                batch.append(
                    Classification(
                        review_id=review.review_id,
                        method="llm",
                        sentiment=result.sentiment,
                        confidence=result.confidence,
                        topics=result.topics,
                        justification=result.justification,
                        model_name=provider.model_name,
                        raw_response=result.model_dump_json(),
                        classified_at=now,
                    )
                )
                classified += 1
                logger.info("PROGRESS %d/%d", i, total)
                logger.info("[%d/%d] Classified %s", i, total, review.review_id)
            except Exception as exc:
                failed += 1
                logger.info("PROGRESS %d/%d", i, total)
                logger.warning("[%d/%d] Failed %s: %s", i, total, review.review_id, exc)
                batch.append(
                    Classification(
                        review_id=review.review_id,
                        method="llm",
                        sentiment=None,
                        topics=[],
                        error_msg=str(exc)[:500],
                        failed_at=now,
                        classified_at=now,
                    )
                )

            # Commit in batches to reduce round-trips
            if len(batch) >= _BATCH_COMMIT_SIZE:
                try:
                    session.add_all(batch)
                    session.commit()
                    batch.clear()
                except Exception as exc:
                    session.rollback()
                    logger.error("Batch commit failed: %s", exc)
                    batch.clear()

        # Commit any remaining rows
        if batch:
            try:
                session.add_all(batch)
                session.commit()
            except Exception as exc:
                session.rollback()
                logger.error("Final batch commit failed: %s", exc)

    finally:
        session.close()

    logger.info("Done: %d classified, %d failed out of %d", classified, failed, total)
    return classified
