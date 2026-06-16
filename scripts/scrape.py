"""CLI to scrape Google Play reviews into the local database."""
import argparse
import logging

from google_play_scraper import Sort

from src.config import DEFAULT_APP_ID, SCRAPE_COUNTRY, SCRAPE_LANG
from src.scraping.play_store import scrape_reviews


SORT_OPTIONS = {
    "newest": Sort.NEWEST,
    "most_relevant": Sort.MOST_RELEVANT,
}


def main():
    parser = argparse.ArgumentParser(description="Scrape Google Play reviews")
    parser.add_argument("--app-id", default=DEFAULT_APP_ID, help="Package name")
    parser.add_argument("--count", type=int, default=500, help="Number of reviews")
    parser.add_argument("--lang", default=SCRAPE_LANG)
    parser.add_argument("--country", default=SCRAPE_COUNTRY)
    parser.add_argument(
        "--sort",
        default="newest",
        choices=list(SORT_OPTIONS.keys()),
        help="Sort order (default: newest)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = scrape_reviews(
        app_id=args.app_id,
        lang=args.lang,
        country=args.country,
        count=args.count,
        sort=SORT_OPTIONS[args.sort],
    )
    print(result.summary_line())


if __name__ == "__main__":
    main()
