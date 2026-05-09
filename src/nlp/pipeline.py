"""Orchestrate the NLP pipeline: preprocessing -> BERT sentiment -> LDA topics -> DB."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.config import BERT_MODEL, LDA_NUM_TOPICS
from src.db.models import Classification, Review, get_session, init_db
from src.nlp.preprocessing import preprocess_batch
from src.nlp.sentiment import predict_sentiment
from src.nlp.topics import LDAModel

logger = logging.getLogger(__name__)


def classify_batch_nlp(
    limit: int | None = None,
    app_id: str | None = None,
    num_topics: int = LDA_NUM_TOPICS,
    retrain_lda: bool = False,
    language: str = "portuguese",
) -> int:
    """Classify reviews with BERT + LDA and store results with method='nlp'."""
    init_db()
    session = get_session()

    query = (
        session.query(Review)
        .outerjoin(
            Classification,
            (Review.review_id == Classification.review_id)
            & (Classification.method == "nlp"),
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
    if total == 0:
        logger.info("No unclassified-by-NLP reviews found.")
        session.close()
        return 0

    logger.info("Found %d reviews to classify with NLP pipeline", total)

    texts = [r.content for r in reviews]
    cleaned_texts, lda_docs = preprocess_batch(texts, language=language)

    # --- BERT sentiment ---
    logger.info("Running BERT sentiment analysis...")
    sentiment_results = predict_sentiment(cleaned_texts)

    # --- LDA topics ---
    lda = LDAModel(n_topics=num_topics)
    loaded = lda.load()

    # Use ALL reviews for LDA corpus (not just the unclassified batch)
    if not loaded or retrain_lda:
        all_reviews = session.query(Review.content).all()
        all_texts = [r[0] for r in all_reviews]
        _, all_lda_docs = preprocess_batch(all_texts, language=language)
        lda.fit(all_lda_docs)
        lda.save()
    else:
        logger.info("Using existing LDA model.")

    logger.info("Predicting LDA topics...")
    topic_results = lda.predict(lda_docs)

    # --- Store results ---
    classified = 0
    for review, sent, topic in zip(reviews, sentiment_results, topic_results):
        try:
            session.add(
                Classification(
                    review_id=review.review_id,
                    method="nlp",
                    sentiment=sent.sentiment,
                    confidence=sent.confidence,
                    topics=topic.mapped_labels,
                    justification=None,
                    model_name=BERT_MODEL,
                    raw_response=None,
                    lda_topic_id=topic.topic_id,
                    lda_topic_words=topic.topic_words,
                    classified_at=datetime.now(timezone.utc),
                )
            )
            session.commit()
            classified += 1
        except Exception as exc:
            session.rollback()
            logger.warning("Failed to store NLP classification for %s: %s", review.review_id, exc)

    session.close()
    logger.info("NLP pipeline done: %d / %d classified", classified, total)
    return classified
