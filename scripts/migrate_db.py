"""Idempotent migration: adds new columns to classifications and backfills method='llm'."""

import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DB_PATH

logger = logging.getLogger(__name__)


def _column_exists(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def migrate():
    if not DB_PATH.exists():
        logger.info("No database found at %s — nothing to migrate.", DB_PATH)
        return

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    new_columns = {
        "method": "TEXT NOT NULL DEFAULT 'llm'",
        "confidence": "REAL",
        "lda_topic_id": "INTEGER",
        "lda_topic_words": "TEXT",
    }

    for col, typedef in new_columns.items():
        if not _column_exists(cur, "classifications", col):
            cur.execute(f"ALTER TABLE classifications ADD COLUMN {col} {typedef}")
            logger.info("Added column classifications.%s", col)
        else:
            logger.info("Column classifications.%s already exists — skipping.", col)

    cur.execute("UPDATE classifications SET method = 'llm' WHERE method IS NULL OR method = ''")
    logger.info("Backfilled method='llm' on existing rows.")

    # If the old single-column UNIQUE constraint still exists, rebuild the table.
    # SQLite cannot DROP CONSTRAINT, so we do a table-swap.
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='classifications'")
    row = cur.fetchone()
    if row and "uq_classification_review UNIQUE (review_id)" in (row[0] or ""):
        logger.info("Old single-column UNIQUE(review_id) detected — rebuilding table...")
        cur.executescript("""
            PRAGMA foreign_keys=OFF;
            CREATE TABLE classifications_new (
                id INTEGER NOT NULL,
                review_id VARCHAR NOT NULL,
                method TEXT NOT NULL DEFAULT 'llm',
                sentiment VARCHAR NOT NULL,
                confidence REAL,
                topics JSON NOT NULL,
                justification TEXT,
                model_name VARCHAR,
                raw_response TEXT,
                lda_topic_id INTEGER,
                lda_topic_words TEXT,
                classified_at DATETIME,
                PRIMARY KEY (id),
                CONSTRAINT uq_classification_review_method UNIQUE (review_id, method),
                FOREIGN KEY(review_id) REFERENCES reviews (review_id)
            );
            INSERT INTO classifications_new
                SELECT id, review_id, method, sentiment, confidence, topics, justification,
                       model_name, raw_response, lda_topic_id, lda_topic_words, classified_at
                FROM classifications;
            DROP TABLE classifications;
            ALTER TABLE classifications_new RENAME TO classifications;
            CREATE INDEX IF NOT EXISTS ix_classifications_review_id ON classifications (review_id);
            PRAGMA foreign_keys=ON;
        """)
        logger.info("Table rebuilt with composite UNIQUE(review_id, method).")
    else:
        # Ensure composite index exists even if table was already correct
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='uq_classification_review_method'")
        if not cur.fetchone():
            cur.execute(
                "CREATE UNIQUE INDEX uq_classification_review_method ON classifications(review_id, method)"
            )
            logger.info("Created composite unique index (review_id, method).")

    conn.commit()
    conn.close()
    logger.info("Migration complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    migrate()
