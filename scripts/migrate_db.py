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


def _column_not_null(cursor: sqlite3.Cursor, table: str, column: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table})")
    for row in cursor.fetchall():
        if row[1] == column:
            return bool(row[3])
    return False


def _rebuild_classifications_table(cursor: sqlite3.Cursor) -> None:
    """Recreate classifications with nullable sentiment/topics and composite unique key."""
    cursor.executescript("""
        PRAGMA foreign_keys=OFF;
        CREATE TABLE classifications_new (
            id INTEGER NOT NULL,
            review_id VARCHAR NOT NULL,
            method TEXT NOT NULL DEFAULT 'llm',
            sentiment VARCHAR,
            confidence REAL,
            topics JSON,
            justification TEXT,
            model_name VARCHAR,
            raw_response TEXT,
            lda_topic_id INTEGER,
            lda_topic_words TEXT,
            error_msg TEXT,
            failed_at DATETIME,
            classified_at DATETIME,
            PRIMARY KEY (id),
            CONSTRAINT uq_classification_review_method UNIQUE (review_id, method),
            FOREIGN KEY(review_id) REFERENCES reviews (review_id)
        );
        INSERT INTO classifications_new
            SELECT id, review_id,
                   COALESCE(NULLIF(method, ''), 'llm'),
                   sentiment, confidence, topics, justification,
                   model_name, raw_response, lda_topic_id, lda_topic_words,
                   error_msg, failed_at, classified_at
            FROM classifications;
        DROP TABLE classifications;
        ALTER TABLE classifications_new RENAME TO classifications;
        CREATE INDEX IF NOT EXISTS ix_classifications_review_id ON classifications (review_id);
        PRAGMA foreign_keys=ON;
    """)


def migrate():
    if not DB_PATH.exists():
        logger.info("No database found at %s — nothing to migrate.", DB_PATH)
        return

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    try:
        new_columns = {
            "method": "TEXT NOT NULL DEFAULT 'llm'",
            "confidence": "REAL",
            "lda_topic_id": "INTEGER",
            "lda_topic_words": "TEXT",
            "error_msg": "TEXT",
            "failed_at": "DATETIME",
        }

        for col, typedef in new_columns.items():
            if not _column_exists(cur, "classifications", col):
                cur.execute(f"ALTER TABLE classifications ADD COLUMN {col} {typedef}")
                logger.info("Added column classifications.%s", col)
            else:
                logger.info("Column classifications.%s already exists — skipping.", col)

        cur.execute("UPDATE classifications SET method = 'llm' WHERE method IS NULL OR method = ''")
        logger.info("Backfilled method='llm' on existing rows.")

        needs_rebuild = False
        cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='classifications'")
        row = cur.fetchone()
        if row and "uq_classification_review UNIQUE (review_id)" in (row[0] or ""):
            logger.info("Old single-column UNIQUE(review_id) detected — will rebuild table.")
            needs_rebuild = True
        if _column_not_null(cur, "classifications", "sentiment") or _column_not_null(
            cur, "classifications", "topics"
        ):
            logger.info(
                "classifications.sentiment/topics must allow NULL for failed rows — will rebuild table."
            )
            needs_rebuild = True

        if needs_rebuild:
            _rebuild_classifications_table(cur)
            logger.info("Table rebuilt (nullable sentiment/topics, composite unique key).")
        else:
            cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='uq_classification_review_method'")
            if not cur.fetchone():
                cur.execute(
                    "CREATE UNIQUE INDEX uq_classification_review_method ON classifications(review_id, method)"
                )
                logger.info("Created composite unique index (review_id, method).")

        conn.commit()
        logger.info("Migration complete.")
    except Exception:
        conn.rollback()
        logger.exception("Migration failed — rolled back.")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    migrate()
