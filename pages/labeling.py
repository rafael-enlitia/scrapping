"""Interactive labeling page to build a gold dataset for evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.db.queries import get_app_ids, get_reviews_df
from src.llm.taxonomy import SENTIMENT_VALUES, TOPIC_VALUES

GOLD_PATH = Path("data/gold.jsonl")


def load_gold() -> dict[str, dict]:
    """Load existing gold labels keyed by review_id."""
    labels: dict[str, dict] = {}
    if GOLD_PATH.exists():
        with open(GOLD_PATH) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    labels[item["review_id"]] = item
                except (json.JSONDecodeError, KeyError):
                    st.warning(f"Skipped corrupt gold label on line {lineno}.")
    return labels


def save_gold(labels: dict[str, dict]):
    GOLD_PATH.parent.mkdir(exist_ok=True)
    with open(GOLD_PATH, "w") as f:
        for item in labels.values():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


st.title("Review Labeling Tool")
st.caption("Label reviews to build a gold dataset for evaluation. Labels are saved to `data/gold.jsonl`.")

app_ids = get_app_ids()
if not app_ids:
    st.warning("No reviews in the database yet.")
    st.stop()

selected_app = st.sidebar.selectbox("App", app_ids, key="label_app")
show_only = st.sidebar.radio("Show", ["Classified (to verify)", "Unclassified", "All"])
st.sidebar.divider()

gold_labels = load_gold()
st.sidebar.metric("Gold labels collected", len(gold_labels))

df = get_reviews_df(selected_app)
if df.empty:
    st.info("No reviews found.")
    st.stop()

# Deduplicate: when both LLM and NLP rows exist, prefer LLM for labeling reference
df = df.sort_values("method", ascending=True, na_position="last")
df = df.drop_duplicates(subset="review_id", keep="first")

if show_only == "Classified (to verify)":
    df = df[df["sentiment"].notna()]
elif show_only == "Unclassified":
    df = df[df["sentiment"].isna()]

if df.empty:
    st.info("No reviews match the selected filter.")
    st.stop()

# Pagination
page_size = 10
total_pages = max(1, (len(df) - 1) // page_size + 1)
page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1)
start = (page - 1) * page_size
page_df = df.iloc[start : start + page_size]

st.caption(f"Page {page}/{total_pages} — {len(df)} reviews total")

for idx, row in page_df.iterrows():
    rid = row["review_id"]
    existing = gold_labels.get(rid, {})
    is_labeled = bool(existing)

    stars = "⭐" * int(row["score"]) if pd.notna(row["score"]) else ""
    badge = "✅ labeled" if is_labeled else ""
    llm_sent = row["sentiment"] if pd.notna(row["sentiment"]) else "—"

    with st.expander(f"{stars} {badge} | {rid[:12]}… | LLM: {llm_sent}", expanded=not is_labeled):
        st.write(row["content"])

        if pd.notna(row.get("justification")):
            st.caption(f"**LLM justification:** {row['justification']}")

        col1, col2 = st.columns(2)
        with col1:
            default_sent_idx = SENTIMENT_VALUES.index(existing["sentiment"]) if existing.get("sentiment") in SENTIMENT_VALUES else (SENTIMENT_VALUES.index(llm_sent) if llm_sent in SENTIMENT_VALUES else 0)
            human_sentiment = st.selectbox(
                "Sentiment",
                SENTIMENT_VALUES,
                index=default_sent_idx,
                key=f"sent_{rid}",
            )

        with col2:
            llm_topics = []
            if pd.notna(row.get("topics")):
                try:
                    llm_topics = json.loads(row["topics"]) if isinstance(row["topics"], str) else row["topics"]
                except (json.JSONDecodeError, TypeError):
                    pass
            default_topics = existing.get("topics", llm_topics)
            human_topics = st.multiselect(
                "Topics",
                TOPIC_VALUES,
                default=[t for t in default_topics if t in TOPIC_VALUES],
                key=f"topics_{rid}",
            )

        if st.button("Save label", key=f"save_{rid}"):
            gold_labels[rid] = {
                "review_id": rid,
                "sentiment": human_sentiment,
                "topics": human_topics,
            }
            save_gold(gold_labels)
            st.success(f"Saved label for {rid[:12]}…")
            st.rerun()
