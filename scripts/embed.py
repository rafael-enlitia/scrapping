"""CLI to compute BERT embeddings + UMAP/KMeans clustering for all reviews."""

from __future__ import annotations

import argparse
import logging

from src.db.models import Review, get_session, init_db
from src.nlp.embeddings import run_embedding_pipeline


def main():
    parser = argparse.ArgumentParser(description="Compute review embeddings and cluster them")
    parser.add_argument("--app-id", default=None, help="Filter by app package name (embeddings stored per-app)")
    parser.add_argument("--n-clusters", type=int, default=8, help="Number of KMeans clusters (default: 8)")
    parser.add_argument("--limit", type=int, default=None, help="Max reviews to embed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    init_db()
    session = get_session()
    try:
        query = session.query(Review.review_id, Review.content).order_by(Review.review_date.desc())
        if args.app_id:
            query = query.filter(Review.app_id == args.app_id)
        if args.limit:
            query = query.limit(args.limit)
        rows = query.all()
    finally:
        session.close()

    if not rows:
        print("No reviews found.")
        return

    total = len(rows)
    print(f"EMBED_SUMMARY target={total} clusters={args.n_clusters}")
    logger.info("Loading BERT model (first run may download ~440 MB)…")
    logger.info("PROGRESS 0/%d", total)

    result = run_embedding_pipeline(list(rows), n_clusters=args.n_clusters, app_id=args.app_id)
    print(f"EMBED_SUMMARY embedded={len(result.review_ids)} clusters={args.n_clusters}")
    for label in sorted(set(result.cluster_labels.tolist())):
        count = (result.cluster_labels == label).sum()
        print(f"  Cluster {label}: {count} reviews")


if __name__ == "__main__":
    main()
