from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime

from google_play_scraper import Sort, reviews

from src.db.models import Review, get_session, init_db

logger = logging.getLogger(__name__)

MIN_CONTENT_LENGTH = 5


@dataclass
class ScrapeResult:
    reviews: list[dict]
    store_fetched: int
    after_filter: int
    saved: int
    skipped_short: int
    skipped_duplicate: int
    skipped_no_id: int

    def summary_line(self) -> str:
        """Machine-readable one-liner for the UI to parse."""
        return (
            f"SCRAPE_SUMMARY saved={self.saved} fetched={self.after_filter} "
            f"store={self.store_fetched} duplicates={self.skipped_duplicate} "
            f"skipped_short={self.skipped_short} no_id={self.skipped_no_id}"
        )


def scrape_reviews(
    app_id: str,
    lang: str = "pt",
    country: str = "pt",
    count: int = 500,
    sort: Sort = Sort.NEWEST,
) -> ScrapeResult:
    """Fetch reviews from Google Play and persist new ones to the database."""
    init_db()

    all_reviews: list[dict] = []
    token = None
    batch_size = min(count, 200)

    while len(all_reviews) < count:
        try:
            result, token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=sort,
                count=batch_size,
                continuation_token=token,
            )
        except Exception as exc:
            logger.error("Scraping failed: %s", exc)
            raise

        if not result:
            break
        all_reviews.extend(result)
        logger.info("PROGRESS %d/%d", len(all_reviews), count)
        if token is None:
            break
        time.sleep(1)

    store_fetched = len(all_reviews)
    all_reviews = all_reviews[:count]

    before = len(all_reviews)
    all_reviews = [r for r in all_reviews if len((r.get("content") or "").strip()) >= MIN_CONTENT_LENGTH]
    skipped_short = before - len(all_reviews)
    if skipped_short:
        logger.info("Skipped %d reviews shorter than %d chars", skipped_short, MIN_CONTENT_LENGTH)

    logger.info("Saving %d reviews to database…", len(all_reviews))
    persist_stats = _persist(all_reviews, app_id)
    saved = persist_stats["saved"]
    skipped_duplicate = persist_stats["duplicates"]
    skipped_no_id = persist_stats["no_id"]

    if skipped_duplicate:
        logger.info("Skipped %d duplicate reviews already in database", skipped_duplicate)
    if skipped_no_id:
        logger.info("Skipped %d reviews without a review ID", skipped_no_id)
    logger.info("Saved %d new reviews (out of %d eligible)", saved, len(all_reviews))

    return ScrapeResult(
        reviews=all_reviews,
        store_fetched=store_fetched,
        after_filter=len(all_reviews),
        saved=saved,
        skipped_short=skipped_short,
        skipped_duplicate=skipped_duplicate,
        skipped_no_id=skipped_no_id,
    )


def _persist(raw_reviews: list[dict], app_id: str) -> dict[str, int]:
    """Bulk-check existing review IDs to avoid N+1 queries, then insert new ones."""
    if not raw_reviews:
        return {"saved": 0, "duplicates": 0, "no_id": 0}

    session = get_session()
    saved = 0
    duplicates = 0
    no_id = 0
    try:
        candidate_ids = [r.get("reviewId") for r in raw_reviews if r.get("reviewId")]
        if not candidate_ids:
            no_id = len(raw_reviews)
            return {"saved": 0, "duplicates": 0, "no_id": no_id}

        existing_ids = set(
            row[0]
            for row in session.query(Review.review_id)
            .filter(Review.review_id.in_(candidate_ids))
            .all()
        )

        for r in raw_reviews:
            rid = r.get("reviewId")
            if not rid:
                no_id += 1
                continue
            if rid in existing_ids:
                duplicates += 1
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
    return {"saved": saved, "duplicates": duplicates, "no_id": no_id}
