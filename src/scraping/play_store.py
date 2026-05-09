from __future__ import annotations

import logging
import time
from datetime import datetime

from google_play_scraper import Sort, reviews

from src.db.models import Review, get_session, init_db

logger = logging.getLogger(__name__)

MIN_CONTENT_LENGTH = 5


def scrape_reviews(
    app_id: str,
    lang: str = "pt",
    country: str = "pt",
    count: int = 500,
    sort: Sort = Sort.NEWEST,
) -> list[dict]:
    """Fetch reviews from Google Play and persist new ones to the database."""
    init_db()

    all_reviews: list[dict] = []
    token = None
    batch_size = min(count, 200)

    while len(all_reviews) < count:
        result, token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=sort,
            count=batch_size,
            continuation_token=token,
        )
        if not result:
            break
        all_reviews.extend(result)
        logger.info("Fetched %d reviews so far", len(all_reviews))
        if token is None:
            break
        time.sleep(1)

    all_reviews = all_reviews[:count]

    # Filter out very short / spam reviews
    before = len(all_reviews)
    all_reviews = [r for r in all_reviews if len((r.get("content") or "").strip()) >= MIN_CONTENT_LENGTH]
    skipped = before - len(all_reviews)
    if skipped:
        logger.info("Skipped %d reviews shorter than %d chars", skipped, MIN_CONTENT_LENGTH)

    saved = _persist(all_reviews, app_id)
    logger.info("Saved %d new reviews (out of %d fetched)", saved, len(all_reviews))
    return all_reviews


def _persist(raw_reviews: list[dict], app_id: str) -> int:
    session = get_session()
    saved = 0
    try:
        for r in raw_reviews:
            rid = r.get("reviewId")
            if not rid:
                continue
            exists = session.query(Review.id).filter_by(review_id=rid).first()
            if exists:
                continue

            review_date = r.get("at")
            if isinstance(review_date, str):
                review_date = datetime.fromisoformat(review_date)

            reply_date = r.get("repliedAt")
            if isinstance(reply_date, str):
                reply_date = datetime.fromisoformat(reply_date)

            session.add(
                Review(
                    review_id=rid,
                    app_id=app_id,
                    username=r.get("userName"),
                    content=r.get("content", ""),
                    score=r.get("score"),
                    thumbs_up=r.get("thumbsUpCount", 0),
                    app_version=r.get("reviewCreatedVersion"),
                    review_date=review_date,
                    language=r.get("lang"),
                    reply_content=r.get("replyContent"),
                    reply_date=reply_date,
                )
            )
            saved += 1
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    return saved
