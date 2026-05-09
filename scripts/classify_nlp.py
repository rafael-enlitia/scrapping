"""CLI to classify reviews using the traditional NLP pipeline (BERT + LDA)."""

import argparse
import logging

from src.nlp.pipeline import classify_batch_nlp


def main():
    parser = argparse.ArgumentParser(description="Classify reviews with BERT sentiment + LDA topics")
    parser.add_argument("--limit", type=int, default=None, help="Max reviews to classify")
    parser.add_argument("--app-id", default=None, help="Filter by app package name")
    parser.add_argument("--num-topics", type=int, default=None, help="Number of LDA topics (default from config)")
    parser.add_argument("--retrain-lda", action="store_true", help="Force retrain the LDA model even if saved")
    parser.add_argument("--language", default="portuguese", help="Language for stopwords/stemming (default: portuguese)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    kwargs: dict = {
        "limit": args.limit,
        "app_id": args.app_id,
        "retrain_lda": args.retrain_lda,
        "language": args.language,
    }
    if args.num_topics is not None:
        kwargs["num_topics"] = args.num_topics

    count = classify_batch_nlp(**kwargs)
    print(f"Done — {count} reviews classified with NLP pipeline.")


if __name__ == "__main__":
    main()
