"""Query helpers that power the dashboard and evaluation scripts.

No Streamlit dependency here — caching is applied by callers (streamlit_app.py).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sqlalchemy import text

from src.db.models import engine, init_db


def _parse_topics_cell(val: Any) -> list[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(val, list):
        return val
    return []


def _merge_topics_json(llm_topics: Any, nlp_topics: Any) -> str:
    merged = list(
        dict.fromkeys(_parse_topics_cell(llm_topics) + _parse_topics_cell(nlp_topics))
    )
    return json.dumps(merged, ensure_ascii=False)


def _ensure_db():
    init_db()


def _method_clause(alias: str = "c") -> str:
    """Return SQL fragment for optional method filtering."""
    return f"AND ({alias}.method = :method OR :method IS NULL)"


# ---------------------------------------------------------------------------
# Core data
# ---------------------------------------------------------------------------

def get_reviews_df(app_id: str | None = None, method: str | None = None) -> pd.DataFrame:
    """Return all reviews joined with their classification (if any).

    When ``method`` is None (dashboard "Both" mode), returns **one row per review**:
    sentiment and LLM fields prefer the LLM row, then NLP; topics are the union of
    both methods' topic lists; NLP-only fields (confidence, LDA) come from NLP.
    """
    _ensure_db()
    if method:
        query = text(f"""
            SELECT
                r.review_id,
                r.app_id,
                r.username,
                r.content,
                r.score,
                r.thumbs_up,
                r.app_version,
                r.review_date,
                r.language,
                r.reply_content,
                c.method,
                c.sentiment,
                c.confidence,
                c.topics,
                c.justification,
                c.model_name,
                c.lda_topic_id,
                c.lda_topic_words,
                c.classified_at
            FROM reviews r
            LEFT JOIN classifications c
                ON r.review_id = c.review_id AND c.method = :method
            WHERE (:app_id IS NULL OR r.app_id = :app_id)
            ORDER BY r.review_date DESC
        """)
    else:
        query = text("""
            SELECT
                r.review_id,
                r.app_id,
                r.username,
                r.content,
                r.score,
                r.thumbs_up,
                r.app_version,
                r.review_date,
                r.language,
                r.reply_content,
                'both' AS method,
                COALESCE(llm.sentiment, nlp.sentiment) AS sentiment,
                nlp.confidence AS confidence,
                llm.topics AS topics_llm,
                nlp.topics AS topics_nlp,
                llm.justification AS justification,
                llm.model_name AS model_name,
                nlp.lda_topic_id AS lda_topic_id,
                nlp.lda_topic_words AS lda_topic_words,
                COALESCE(llm.classified_at, nlp.classified_at) AS classified_at
            FROM reviews r
            LEFT JOIN classifications llm
                ON r.review_id = llm.review_id AND llm.method = 'llm'
            LEFT JOIN classifications nlp
                ON r.review_id = nlp.review_id AND nlp.method = 'nlp'
            WHERE (:app_id IS NULL OR r.app_id = :app_id)
            ORDER BY r.review_date DESC
        """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"app_id": app_id, "method": method})
    if method is None and not df.empty:
        df["topics"] = [
            _merge_topics_json(lm, nm) for lm, nm in zip(df["topics_llm"], df["topics_nlp"])
        ]
        df = df.drop(columns=["topics_llm", "topics_nlp"])
    return df


def get_app_ids() -> list[str]:
    """Return distinct app ids in the database."""
    _ensure_db()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT app_id FROM reviews ORDER BY app_id")).fetchall()
    return [r[0] for r in rows]


def get_versions(app_id: str) -> list[str]:
    """Return distinct app versions for a given app, ordered newest first (semver-aware)."""
    _ensure_db()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT app_version FROM reviews WHERE app_id = :app_id AND app_version IS NOT NULL"),
            {"app_id": app_id},
        ).fetchall()
    raw = [r[0] for r in rows]

    def _semver_key(v: str):
        parts = []
        for segment in v.split("."):
            digits = "".join(c for c in segment if c.isdigit())
            parts.append(int(digits) if digits else 0)
        return parts

    try:
        return sorted(raw, key=_semver_key, reverse=True)
    except Exception:
        return sorted(raw, reverse=True)


# ---------------------------------------------------------------------------
# Per-method aggregations
# ---------------------------------------------------------------------------

def sentiment_by_version(app_id: str, method: str | None = None) -> pd.DataFrame:
    _ensure_db()
    if method is None:
        query = text("""
            SELECT
                r.app_version AS version,
                COALESCE(llm.sentiment, nlp.sentiment) AS sentiment,
                COUNT(*) AS count
            FROM reviews r
            LEFT JOIN classifications llm
                ON r.review_id = llm.review_id AND llm.method = 'llm'
            LEFT JOIN classifications nlp
                ON r.review_id = nlp.review_id AND nlp.method = 'nlp'
            WHERE r.app_id = :app_id
              AND r.app_version IS NOT NULL
              AND COALESCE(llm.sentiment, nlp.sentiment) IS NOT NULL
            GROUP BY r.app_version, COALESCE(llm.sentiment, nlp.sentiment)
            ORDER BY r.app_version
        """)
        params = {"app_id": app_id}
    else:
        query = text(f"""
            SELECT
                r.app_version AS version,
                c.sentiment,
                COUNT(*) AS count
            FROM reviews r
            JOIN classifications c ON r.review_id = c.review_id
            WHERE r.app_id = :app_id
              AND r.app_version IS NOT NULL
              AND c.sentiment IS NOT NULL
              {_method_clause()}
            GROUP BY r.app_version, c.sentiment
            ORDER BY r.app_version
        """)
        params = {"app_id": app_id, "method": method}
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)


def topics_by_version(app_id: str, method: str | None = None) -> pd.DataFrame:
    _ensure_db()
    if method is None:
        query = text("""
            SELECT
                version,
                topic,
                COUNT(*) AS count
            FROM (
                SELECT DISTINCT
                    r.review_id,
                    r.app_version AS version,
                    je.value AS topic
                FROM reviews r
                JOIN classifications c ON r.review_id = c.review_id AND c.method = 'llm',
                     json_each(c.topics) je
                WHERE r.app_id = :app_id AND r.app_version IS NOT NULL
                UNION
                SELECT DISTINCT
                    r.review_id,
                    r.app_version,
                    je.value
                FROM reviews r
                JOIN classifications c ON r.review_id = c.review_id AND c.method = 'nlp',
                     json_each(c.topics) je
                WHERE r.app_id = :app_id AND r.app_version IS NOT NULL
            ) x
            GROUP BY version, topic
            ORDER BY version
        """)
        params = {"app_id": app_id}
    else:
        query = text(f"""
            SELECT
                r.app_version AS version,
                je.value AS topic,
                COUNT(*) AS count
            FROM reviews r
            JOIN classifications c ON r.review_id = c.review_id,
                 json_each(c.topics) je
            WHERE r.app_id = :app_id
              AND r.app_version IS NOT NULL
              AND c.topics IS NOT NULL
              {_method_clause()}
            GROUP BY r.app_version, je.value
            ORDER BY r.app_version
        """)
        params = {"app_id": app_id, "method": method}
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)


def avg_score_by_version(app_id: str) -> pd.DataFrame:
    _ensure_db()
    query = text("""
        SELECT
            app_version AS version,
            AVG(score) AS avg_score,
            COUNT(*) AS review_count
        FROM reviews
        WHERE app_id = :app_id AND app_version IS NOT NULL
        GROUP BY app_version
        ORDER BY app_version
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"app_id": app_id})


def sentiment_over_time(app_id: str, method: str | None = None) -> pd.DataFrame:
    _ensure_db()
    if method is None:
        query = text("""
            SELECT
                strftime('%Y-%W', r.review_date) AS week,
                COALESCE(llm.sentiment, nlp.sentiment) AS sentiment,
                COUNT(*) AS count
            FROM reviews r
            LEFT JOIN classifications llm
                ON r.review_id = llm.review_id AND llm.method = 'llm'
            LEFT JOIN classifications nlp
                ON r.review_id = nlp.review_id AND nlp.method = 'nlp'
            WHERE r.app_id = :app_id
              AND r.review_date IS NOT NULL
              AND COALESCE(llm.sentiment, nlp.sentiment) IS NOT NULL
            GROUP BY week, COALESCE(llm.sentiment, nlp.sentiment)
            ORDER BY week
        """)
        params = {"app_id": app_id}
    else:
        query = text(f"""
            SELECT
                strftime('%Y-%W', r.review_date) AS week,
                c.sentiment,
                COUNT(*) AS count
            FROM reviews r
            JOIN classifications c ON r.review_id = c.review_id
            WHERE r.app_id = :app_id
              AND r.review_date IS NOT NULL
              AND c.sentiment IS NOT NULL
              {_method_clause()}
            GROUP BY week, c.sentiment
            ORDER BY week
        """)
        params = {"app_id": app_id, "method": method}
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)


def sentiment_vs_score(app_id: str, method: str | None = None) -> pd.DataFrame:
    _ensure_db()
    if method is None:
        query = text("""
            SELECT
                r.score,
                COALESCE(llm.sentiment, nlp.sentiment) AS sentiment,
                COUNT(*) AS count
            FROM reviews r
            LEFT JOIN classifications llm
                ON r.review_id = llm.review_id AND llm.method = 'llm'
            LEFT JOIN classifications nlp
                ON r.review_id = nlp.review_id AND nlp.method = 'nlp'
            WHERE r.app_id = :app_id
              AND COALESCE(llm.sentiment, nlp.sentiment) IS NOT NULL
            GROUP BY r.score, COALESCE(llm.sentiment, nlp.sentiment)
            ORDER BY r.score
        """)
        params = {"app_id": app_id}
    else:
        query = text(f"""
            SELECT
                r.score,
                c.sentiment,
                COUNT(*) AS count
            FROM reviews r
            JOIN classifications c ON r.review_id = c.review_id
            WHERE r.app_id = :app_id
              AND c.sentiment IS NOT NULL
              {_method_clause()}
            GROUP BY r.score, c.sentiment
            ORDER BY r.score
        """)
        params = {"app_id": app_id, "method": method}
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=params)


# ---------------------------------------------------------------------------
# Comparison queries (LLM vs NLP)
# ---------------------------------------------------------------------------

def sentiment_comparison(app_id: str) -> pd.DataFrame:
    """Side-by-side sentiment counts for LLM vs NLP."""
    _ensure_db()
    query = text("""
        SELECT
            c.method,
            c.sentiment,
            COUNT(*) AS count
        FROM classifications c
        JOIN reviews r ON r.review_id = c.review_id
        WHERE r.app_id = :app_id AND c.sentiment IS NOT NULL
        GROUP BY c.method, c.sentiment
        ORDER BY c.method, c.sentiment
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"app_id": app_id})


def agreement_matrix(app_id: str) -> pd.DataFrame:
    """Cross-tab of LLM sentiment vs NLP sentiment for reviews classified by both."""
    _ensure_db()
    query = text("""
        SELECT
            llm.sentiment AS llm_sentiment,
            nlp.sentiment AS nlp_sentiment,
            COUNT(*) AS count
        FROM classifications llm
        JOIN classifications nlp
            ON llm.review_id = nlp.review_id
            AND llm.method = 'llm' AND nlp.method = 'nlp'
        JOIN reviews r ON r.review_id = llm.review_id
        WHERE r.app_id = :app_id
          AND llm.sentiment IS NOT NULL
          AND nlp.sentiment IS NOT NULL
        GROUP BY llm.sentiment, nlp.sentiment
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"app_id": app_id})


def agreement_rate(app_id: str) -> dict:
    """Percentage of reviews where both methods agree on sentiment."""
    _ensure_db()
    query = text("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN llm.sentiment = nlp.sentiment THEN 1 ELSE 0 END) AS agreed
        FROM classifications llm
        JOIN classifications nlp
            ON llm.review_id = nlp.review_id
            AND llm.method = 'llm' AND nlp.method = 'nlp'
        JOIN reviews r ON r.review_id = llm.review_id
        WHERE r.app_id = :app_id
          AND llm.sentiment IS NOT NULL
          AND nlp.sentiment IS NOT NULL
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"app_id": app_id}).fetchone()
    if not row or row[0] == 0:
        return {"total": 0, "agreed": 0, "rate": 0.0}
    return {"total": row[0], "agreed": row[1], "rate": round(row[1] / row[0] * 100, 1)}


def comparison_reviews_df(app_id: str) -> pd.DataFrame:
    """Reviews with both LLM and NLP classifications side-by-side."""
    _ensure_db()
    query = text("""
        SELECT
            r.review_id,
            r.content,
            r.score,
            r.app_version,
            r.review_date,
            llm.sentiment AS llm_sentiment,
            llm.topics AS llm_topics,
            llm.justification AS llm_justification,
            llm.confidence AS llm_confidence,
            nlp.sentiment AS nlp_sentiment,
            nlp.confidence AS nlp_confidence,
            nlp.topics AS nlp_topics,
            nlp.lda_topic_id,
            nlp.lda_topic_words
        FROM reviews r
        JOIN classifications llm
            ON r.review_id = llm.review_id AND llm.method = 'llm'
        JOIN classifications nlp
            ON r.review_id = nlp.review_id AND nlp.method = 'nlp'
        WHERE r.app_id = :app_id
          AND llm.sentiment IS NOT NULL
          AND nlp.sentiment IS NOT NULL
        ORDER BY r.review_date DESC
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"app_id": app_id})


def get_embedding_clusters(app_id: str | None = None) -> pd.DataFrame | None:
    """Load pre-computed UMAP coordinates and cluster labels for the given app.

    Returns a DataFrame with columns: review_id, x, y, cluster.
    Returns None if embeddings have not been computed yet for this app.
    """
    from src.nlp.embeddings import load_embeddings  # noqa: PLC0415

    result = load_embeddings(app_id=app_id)
    if result is None:
        return None

    df = pd.DataFrame({
        "review_id": result.review_ids,
        "x": result.umap_2d[:, 0],
        "y": result.umap_2d[:, 1],
        "cluster": result.cluster_labels.astype(str),
    })

    return df if not df.empty else None


def lda_topic_distribution(app_id: str) -> pd.DataFrame:
    """Count of reviews per LDA topic ID (for NLP method only)."""
    _ensure_db()
    query = text("""
        SELECT
            c.lda_topic_id AS topic_id,
            c.lda_topic_words AS topic_words,
            COUNT(*) AS count
        FROM classifications c
        JOIN reviews r ON r.review_id = c.review_id
        WHERE r.app_id = :app_id AND c.method = 'nlp' AND c.lda_topic_id IS NOT NULL
        GROUP BY c.lda_topic_id, c.lda_topic_words
        ORDER BY count DESC
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"app_id": app_id})
