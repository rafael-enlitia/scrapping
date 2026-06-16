"""CLI to classify reviews using the LLM pipeline."""
import argparse
import logging

from src.llm.classifier import classify_batch


def main():
    parser = argparse.ArgumentParser(description="Classify reviews with LLM")
    parser.add_argument("--limit", type=int, default=None, help="Max reviews to classify")
    parser.add_argument("--app-id", default=None, help="Filter by app package name")
    parser.add_argument(
        "--provider",
        default=None,
        choices=["openai", "ollama", "iaedu"],
        help="Override LLM provider",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry reviews that previously failed classification (have an error_msg in the DB)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.retry_failed and args.limit is None:
        args.limit = 50
        logging.info("--retry-failed: defaulting to --limit 50")

    count = classify_batch(
        limit=args.limit,
        app_id=args.app_id,
        provider_name=args.provider,
        retry_failed=args.retry_failed,
    )
    print(f"CLASSIFY_SUMMARY method=llm classified={count}")


if __name__ == "__main__":
    main()
