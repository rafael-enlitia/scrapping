"""Interactive labeling page to build a gold dataset for evaluation."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import DATA_DIR
from src.db.queries import get_app_ids, get_reviews_df
from src.llm.taxonomy import SENTIMENT_VALUES, TOPIC_VALUES

GOLD_PATH = DATA_DIR / "gold.jsonl"


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


def save_gold(labels: dict[str, dict]) -> None:
    """Rewrite full gold file (used for deletes and compaction)."""
    GOLD_PATH.parent.mkdir(exist_ok=True)
    with open(GOLD_PATH, "w") as f:
        for item in labels.values():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_gold_label(existing_labels: dict[str, dict], rid: str, item: dict) -> None:
    """Save one label efficiently: append if new (fast), rewrite if update (necessary)."""
    if rid not in existing_labels:
        GOLD_PATH.parent.mkdir(exist_ok=True)
        with open(GOLD_PATH, "a") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        existing_labels[rid] = item
        save_gold(existing_labels)


st.title("Review Labeling Tool")
st.caption("Label reviews to build a gold dataset for evaluation. Labels are saved to `data/gold.jsonl`.")

app_ids = get_app_ids()
if not app_ids:
    st.warning("No reviews in the database yet.")
    st.stop()

_shared_app = st.session_state.get("shared_app_id", "")
_default_app_idx = app_ids.index(_shared_app) if _shared_app in app_ids else 0
selected_app = st.sidebar.selectbox("App", app_ids, index=_default_app_idx, key="label_app")
show_only = st.sidebar.radio("Show", ["Classified (to verify)", "Unclassified", "All"])
st.sidebar.divider()

gold_labels = load_gold()
st.sidebar.metric("Gold labels collected", len(gold_labels))

df = get_reviews_df(selected_app)
if df.empty:
    st.info("No reviews found.")
    st.stop()

df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

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

page_size = 10
total_pages = max(1, (len(df) - 1) // page_size + 1)

# Reset page to 1 whenever the app or filter changes
filter_key = f"{selected_app}_{show_only}"
if st.session_state.get("_label_filter_key") != filter_key:
    st.session_state["_label_filter_key"] = filter_key
    st.session_state["_label_page"] = 1

page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=st.session_state.get("_label_page", 1), key="label_page_input")
st.session_state["_label_page"] = page

start = (page - 1) * page_size
page_df = df.iloc[start : start + page_size]

st.caption(f"Page {page}/{total_pages} — {len(df)} reviews total")

for idx, row in page_df.iterrows():
    rid = row["review_id"]
    existing = gold_labels.get(rid, {})
    is_labeled = bool(existing)

    score_val = row["score"]
    stars = "⭐" * int(score_val) if pd.notna(score_val) and 1 <= int(score_val) <= 5 else "☆"
    badge = "✅" if is_labeled else "🔲"
    method_shown = row.get("method", "unknown")
    pred_sent = row["sentiment"] if pd.notna(row["sentiment"]) else "—"
    _content_preview = str(row.get("content", ""))[:80].replace("\n", " ")
    _date_val = row.get("review_date")
    _date_str = _date_val.strftime("%Y-%m-%d") if pd.notna(_date_val) else ""

    header = f"{badge} {stars}  {pred_sent}  |  {_date_str}  |  {_content_preview}…"
    with st.expander(header, expanded=not is_labeled):
        st.write(row["content"])

        if pd.notna(row.get("justification")):
            st.caption(f"**LLM justification:** {row['justification']}")

        col1, col2 = st.columns(2)
        with col1:
            default_sent_idx = SENTIMENT_VALUES.index(existing["sentiment"]) if existing.get("sentiment") in SENTIMENT_VALUES else (SENTIMENT_VALUES.index(pred_sent) if pred_sent in SENTIMENT_VALUES else 0)
            human_sentiment = st.selectbox(
                "Sentiment",
                SENTIMENT_VALUES,
                index=default_sent_idx,
                key=f"sent_{rid}",
            )

        with col2:
            pred_topics = []
            if pd.notna(row.get("topics")):
                try:
                    pred_topics = json.loads(row["topics"]) if isinstance(row["topics"], str) else row["topics"]
                except (json.JSONDecodeError, TypeError):
                    pass
            default_topics = existing.get("topics", pred_topics)
            human_topics = st.multiselect(
                "Topics",
                TOPIC_VALUES,
                default=[t for t in default_topics if t in TOPIC_VALUES],
                key=f"topics_{rid}",
            )

        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("Save label", key=f"save_{rid}"):
                new_item = {"review_id": rid, "sentiment": human_sentiment, "topics": human_topics}
                save_gold_label(gold_labels, rid, new_item)
                gold_labels[rid] = new_item
                st.success(f"Saved label for {rid[:12]}…")
                st.rerun()
        with btn_col2:
            if is_labeled and st.button("Delete label", key=f"del_{rid}"):
                gold_labels.pop(rid, None)
                save_gold(gold_labels)
                st.warning(f"Deleted label for {rid[:12]}…")
                st.rerun()
