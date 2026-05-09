"""Interactive dashboard for App Feedback Monitor — LLM vs NLP comparison."""

from __future__ import annotations

import io
import json

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

from src.db.queries import (
    agreement_matrix,
    agreement_rate,
    avg_score_by_version,
    comparison_reviews_df,
    get_app_ids,
    get_reviews_df,
    get_versions,
    lda_topic_distribution,
    sentiment_by_version,
    sentiment_comparison,
    sentiment_over_time,
    sentiment_vs_score,
    topics_by_version,
)
from src.llm.taxonomy import SENTIMENT_VALUES, TOPIC_VALUES
from src.nlp.preprocessing import tokenize_for_lda

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="App Feedback Monitor", page_icon="📱", layout="wide")

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
    "mixed": "#f39c12",
}

SCORE_COLORS = {1: "#e74c3c", 2: "#e67e22", 3: "#f1c40f", 4: "#27ae60", 5: "#2ecc71"}

METHOD_LABELS = {"llm": "LLM (GPT / Ollama)", "nlp": "NLP (BERT + LDA)"}

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

app_ids = get_app_ids()
if not app_ids:
    st.warning("No reviews in the database yet. Run the scraper first:")
    st.code("python -m scripts.scrape --app-id com.whatsapp --count 500", language="bash")
    st.stop()

selected_app = st.sidebar.selectbox("App", app_ids)

# Method selector
method_option = st.sidebar.radio(
    "Classification method",
    ["LLM", "NLP", "Both (comparison)"],
    index=0,
)
method_filter: str | None = None
if method_option == "LLM":
    method_filter = "llm"
elif method_option == "NLP":
    method_filter = "nlp"

versions = get_versions(selected_app)
selected_versions = st.sidebar.multiselect("Versions", versions, default=versions)

# NLP cannot produce "mixed" — show only the sentiments that apply to the active method
_nlp_sentiments = [s for s in SENTIMENT_VALUES if s != "mixed"]
_sentiment_options = SENTIMENT_VALUES if method_filter != "nlp" else _nlp_sentiments
_sentiment_defaults = _sentiment_options  # all selected by default
selected_sentiments = st.sidebar.multiselect("Sentiments", _sentiment_options, default=_sentiment_defaults)

selected_topics = st.sidebar.multiselect("Topics", TOPIC_VALUES, default=TOPIC_VALUES)
if method_filter == "nlp":
    st.sidebar.caption(
        "ℹ️ In NLP mode, topics are discovered by LDA and mapped to this taxonomy. "
        "Not every taxonomy label may be present in every run."
    )

# Date range
raw_df_full = get_reviews_df(selected_app, method=method_filter)
if raw_df_full.empty:
    st.info("No reviews found for the selected app.")
    st.stop()

raw_df_full["review_date"] = pd.to_datetime(raw_df_full["review_date"], errors="coerce")
min_date = raw_df_full["review_date"].min()
max_date = raw_df_full["review_date"].max()
if pd.notna(min_date) and pd.notna(max_date):
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
else:
    date_range = None

st.sidebar.divider()
if st.sidebar.button("Clear cache & reload"):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Parse and filter
# ---------------------------------------------------------------------------
def parse_topics(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(val, list):
        return val
    return []


raw_df_full["topics_list"] = raw_df_full["topics"].apply(parse_topics)

df = raw_df_full.copy()

if selected_versions:
    df = df[df["app_version"].isin(selected_versions) | df["app_version"].isna()]
if selected_sentiments:
    df = df[df["sentiment"].isin(selected_sentiments) | df["sentiment"].isna()]
if selected_topics:
    df = df[
        (df["topics_list"].apply(lambda ts: any(t in selected_topics for t in ts)))
        | (df["topics_list"].apply(len) == 0)
    ]
if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    mask = df["review_date"].notna()
    df = df[~mask | ((df["review_date"].dt.date >= start) & (df["review_date"].dt.date <= end))]

# ---------------------------------------------------------------------------
# Header metrics
# ---------------------------------------------------------------------------
st.title("App Feedback Monitor")
method_badge = METHOD_LABELS.get(method_filter, "LLM + NLP")
st.caption(f"Showing data for **{selected_app}** — method: **{method_badge}**")
if method_option == "Both (comparison)":
    st.caption(
        "One row per review. Sentiment uses LLM when present, otherwise NLP. "
        "Topics combine both methods. Use the **Comparison** tab for side-by-side LLM vs NLP."
    )

# Warn if NLP mode is active but "mixed" was somehow selected (NLP never produces it)
if method_filter == "nlp" and "mixed" in selected_sentiments:
    st.warning(
        "⚠️ The **NLP pipeline** never produces a `mixed` sentiment — "
        "only `positive`, `negative` and `neutral` are possible. "
        "Remove `mixed` from the Sentiments filter to see all NLP reviews."
    )

classified_mask = df["sentiment"].notna()
unclassified_count = int((~classified_mask).sum())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total reviews", len(df))
col2.metric("Classified", int(classified_mask.sum()))
col3.metric("Unclassified", unclassified_count, delta=f"-{unclassified_count}" if unclassified_count else None, delta_color="inverse")
_avg_score = df["score"].mean()
col4.metric("Avg rating", f"{_avg_score:.1f}" if pd.notna(_avg_score) else "—")
col5.metric("Versions", df["app_version"].nunique())

if unclassified_count > 0:
    if method_filter == "nlp":
        st.info(f"{unclassified_count} reviews pending NLP classification. Run: `python -m scripts.classify_nlp --limit {unclassified_count}`")
    elif method_filter == "llm":
        st.info(f"{unclassified_count} reviews pending LLM classification. Run: `python -m scripts.classify --limit {unclassified_count}`")
    else:
        st.info(f"{unclassified_count} reviews pending. Run both classifiers to compare.")

# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------
csv_data = df.drop(columns=["topics"], errors="ignore").rename(columns={"topics_list": "topics"}).to_csv(index=False)
st.download_button("Export CSV", csv_data, f"{selected_app}_reviews.csv", "text/csv", width="content")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_names = ["Sentiment", "Topics", "Evolution", "Score Analysis", "Reviews"]
if method_option == "Both (comparison)":
    tab_names.insert(4, "Comparison")

tabs = st.tabs(tab_names)
tab_idx = {name: i for i, name in enumerate(tab_names)}

# -- Sentiment tab ----------------------------------------------------------
with tabs[tab_idx["Sentiment"]]:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Sentiment distribution")
        sent_counts = df[df["sentiment"].notna()]["sentiment"].value_counts()
        if not sent_counts.empty:
            fig = px.pie(
                names=sent_counts.index,
                values=sent_counts.values,
                color=sent_counts.index,
                color_discrete_map=SENTIMENT_COLORS,
            )
            fig.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig, width="stretch")

    with c2:
        st.subheader("Sentiment by version")
        sv = sentiment_by_version(selected_app, method=method_filter)
        if not sv.empty:
            if selected_versions and len(selected_versions) < len(versions):
                sv = sv[sv["version"].isin(selected_versions)]
            if not sv.empty:
                fig = px.bar(
                    sv,
                    x="version",
                    y="count",
                    color="sentiment",
                    color_discrete_map=SENTIMENT_COLORS,
                    barmode="stack",
                )
                fig.update_layout(xaxis_tickangle=-45, margin=dict(t=20, b=20))
                st.plotly_chart(fig, width="stretch")

# -- Topics tab -------------------------------------------------------------
with tabs[tab_idx["Topics"]]:
    # Compute all_topics once here — used by frequency, co-occurrence and word cloud
    all_topics: list[str] = [t for topics in df["topics_list"] for t in topics]

    if method_filter == "nlp":
        st.subheader("LDA-discovered topics")
        lda_dist = lda_topic_distribution(selected_app)

        # Apply the active version filter to the LDA distribution
        if not lda_dist.empty and selected_versions and len(selected_versions) < len(versions):
            # lda_topic_distribution doesn't carry version info — use df to derive
            # the set of lda_topic_ids that appear in the filtered reviews
            visible_lda_ids = df["lda_topic_id"].dropna().unique().tolist()
            lda_dist = lda_dist[lda_dist["topic_id"].isin(visible_lda_ids)]

        if not lda_dist.empty:
            # Recompute counts from the filtered df so they match other charts
            if selected_versions and len(selected_versions) < len(versions):
                id_counts = df["lda_topic_id"].value_counts().rename_axis("topic_id").reset_index(name="count")
                lda_dist = lda_dist.drop(columns=["count"], errors="ignore").merge(id_counts, on="topic_id", how="inner")

            for _, row in lda_dist.iterrows():
                st.markdown(f"**Topic {int(row['topic_id'])}** ({int(row['count'])} reviews): _{row['topic_words']}_")
            fig = px.bar(lda_dist, x="topic_id", y="count", text="count")
            fig.update_layout(
                xaxis_title="LDA Topic ID",
                yaxis_title="Reviews",
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, width="stretch")
            if selected_versions and len(selected_versions) < len(versions):
                st.caption("ℹ️ Counts reflect the active version filter.")
        else:
            st.info("No LDA topic data yet. Run: `python -m scripts.classify_nlp`")

        st.divider()
        st.subheader("Mapped taxonomy topics")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Topic frequency")
        if all_topics:
            topic_series = pd.Series(all_topics).value_counts()
            fig = px.bar(
                x=topic_series.index,
                y=topic_series.values,
                labels={"x": "Topic", "y": "Count"},
                color=topic_series.index,
            )
            fig.update_layout(xaxis_tickangle=-45, margin=dict(t=20, b=20), showlegend=False)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No topic data for the current filter selection.")

    with c2:
        st.subheader("Topics by version")
        # Filter topics_by_version to respect the active version filter
        tv = topics_by_version(selected_app, method=method_filter)
        if not tv.empty:
            if selected_versions and len(selected_versions) < len(versions):
                tv = tv[tv["version"].isin(selected_versions)]
            if not tv.empty:
                fig = px.bar(tv, x="version", y="count", color="topic", barmode="stack")
                fig.update_layout(xaxis_tickangle=-45, margin=dict(t=20, b=20))
                st.plotly_chart(fig, width="stretch")

    st.subheader("Topic co-occurrence")
    if all_topics:
        topic_pairs: dict[str, int] = {}
        for topics in df["topics_list"]:
            if len(topics) >= 2:
                for i, t1 in enumerate(sorted(topics)):
                    for t2 in sorted(topics)[i + 1 :]:
                        pair = f"{t1} + {t2}"
                        topic_pairs[pair] = topic_pairs.get(pair, 0) + 1
        if topic_pairs:
            pairs_df = pd.DataFrame(sorted(topic_pairs.items(), key=lambda x: -x[1])[:15], columns=["Pair", "Count"])
            fig = px.bar(pairs_df, x="Count", y="Pair", orientation="h")
            fig.update_layout(margin=dict(t=20, b=20), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Not enough multi-topic reviews to compute co-occurrence.")
    else:
        st.info("No topic data available.")

    st.subheader("Word Cloud")
    # Use preprocessed tokens (stopwords removed, stemmed) for a meaningful cloud
    _wc_lang = "portuguese" if not method_filter or method_filter in ("llm", "nlp") else "portuguese"
    _wc_tokens: list[str] = []
    for text in df["content"].dropna():
        _wc_tokens.extend(tokenize_for_lda(str(text), language=_wc_lang))
    wc_text = " ".join(_wc_tokens)
    if wc_text.strip():
        wc = WordCloud(
            width=900,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=150,
            collocations=False,
        ).generate(wc_text)
        fig_wc, ax_wc = plt.subplots(figsize=(12, 5))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        fig_wc.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig_wc)
        buf.seek(0)
        st.image(buf, width="stretch")
        st.caption("Words shown after stopword removal and stemming.")
    else:
        st.info("No review text available for word cloud.")

# -- Evolution tab ----------------------------------------------------------
with tabs[tab_idx["Evolution"]]:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Average score by version")
        asv = avg_score_by_version(selected_app)
        if not asv.empty:
            if selected_versions and len(selected_versions) < len(versions):
                asv = asv[asv["version"].isin(selected_versions)]
            if not asv.empty:
                fig = px.line(asv, x="version", y="avg_score", markers=True)
                fig.update_layout(xaxis_tickangle=-45, margin=dict(t=20, b=20))
                st.plotly_chart(fig, width="stretch")

    with c2:
        st.subheader("Sentiment over time (weekly)")
        sot = sentiment_over_time(selected_app, method=method_filter)
        if not sot.empty:
            fig = px.area(
                sot,
                x="week",
                y="count",
                color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
            )
            fig.update_layout(xaxis_tickangle=-45, margin=dict(t=20, b=20))
            st.plotly_chart(fig, width="stretch")

# -- Comparison tab (only when "Both" is selected) -------------------------
if "Comparison" in tab_idx:
    with tabs[tab_idx["Comparison"]]:
        st.subheader("LLM vs NLP — Sentiment Comparison")

        agree = agreement_rate(selected_app)
        m1, m2, m3 = st.columns(3)
        m1.metric("Reviews compared", agree["total"])
        m2.metric("Sentiment agreement", f"{agree['rate']}%")
        m3.metric("Disagreements", agree["total"] - agree["agreed"])

        if agree["total"] == 0:
            st.warning("No reviews have been classified by both methods yet. Run both pipelines first.")
        else:
            c1, c2 = st.columns(2)

            # Side-by-side pie charts
            sc = sentiment_comparison(selected_app)
            with c1:
                st.markdown("**LLM sentiment**")
                llm_sc = sc[sc["method"] == "llm"]
                if not llm_sc.empty:
                    fig = px.pie(llm_sc, names="sentiment", values="count", color="sentiment", color_discrete_map=SENTIMENT_COLORS)
                    fig.update_layout(margin=dict(t=20, b=20))
                    st.plotly_chart(fig, width="stretch")

            with c2:
                st.markdown("**NLP sentiment**")
                nlp_sc = sc[sc["method"] == "nlp"]
                if not nlp_sc.empty:
                    fig = px.pie(nlp_sc, names="sentiment", values="count", color="sentiment", color_discrete_map=SENTIMENT_COLORS)
                    fig.update_layout(margin=dict(t=20, b=20))
                    st.plotly_chart(fig, width="stretch")

            # Agreement matrix
            st.subheader("Agreement matrix (LLM vs NLP)")
            am = agreement_matrix(selected_app)
            if not am.empty:
                pivot = am.pivot_table(index="llm_sentiment", columns="nlp_sentiment", values="count", fill_value=0)
                fig = px.imshow(
                    pivot,
                    labels=dict(x="NLP Sentiment", y="LLM Sentiment", color="Count"),
                    color_continuous_scale="Blues",
                    text_auto=True,
                    aspect="auto",
                )
                fig.update_layout(margin=dict(t=20, b=20))
                st.plotly_chart(fig, width="stretch")

            # Topic comparison
            st.subheader("Topic distribution comparison")
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown("**LLM topics (taxonomy)**")
                tv_llm = topics_by_version(selected_app, method="llm")
                if not tv_llm.empty:
                    llm_topic_counts = tv_llm.groupby("topic")["count"].sum().sort_values(ascending=False)
                    fig = px.bar(x=llm_topic_counts.index, y=llm_topic_counts.values, labels={"x": "Topic", "y": "Count"})
                    fig.update_layout(xaxis_tickangle=-45, margin=dict(t=20, b=20))
                    st.plotly_chart(fig, width="stretch")

            with tc2:
                st.markdown("**NLP topics (LDA-discovered)**")
                lda_dist = lda_topic_distribution(selected_app)
                if not lda_dist.empty:
                    fig = px.bar(lda_dist, x="topic_id", y="count", text="topic_words")
                    fig.update_layout(xaxis_title="LDA Topic ID", margin=dict(t=20, b=20))
                    st.plotly_chart(fig, width="stretch")

            # Disagreement samples
            st.subheader("Sample disagreements")
            comp_df = comparison_reviews_df(selected_app)
            disagree_df = comp_df[comp_df["llm_sentiment"] != comp_df["nlp_sentiment"]]
            for _, row in disagree_df.head(20).iterrows():
                stars = "⭐" * int(row["score"]) if pd.notna(row["score"]) else ""
                header = f"{stars} LLM: **{row['llm_sentiment']}** | NLP: **{row['nlp_sentiment']}** (conf {row.get('nlp_confidence', '—')})"
                with st.expander(header, expanded=False):
                    st.write(row["content"])
                    if pd.notna(row.get("llm_justification")):
                        st.caption(f"**LLM justification:** {row['llm_justification']}")
                    st.caption(f"**LDA topic:** {row.get('lda_topic_words', '—')}")

# -- Score Analysis tab -----------------------------------------------------
with tabs[tab_idx["Score Analysis"]]:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Score distribution")
        if not df.empty:
            score_counts = df["score"].value_counts().sort_index()
            fig = px.bar(
                x=score_counts.index.astype(str),
                y=score_counts.values,
                labels={"x": "Stars", "y": "Count"},
                color=score_counts.index.astype(str),
                color_discrete_map={str(k): v for k, v in SCORE_COLORS.items()},
            )
            fig.update_layout(margin=dict(t=20, b=20), showlegend=False)
            st.plotly_chart(fig, width="stretch")

    with c2:
        st.subheader("Sentiment vs Star Rating")
        svs = sentiment_vs_score(selected_app, method=method_filter)
        if not svs.empty:
            pivot = svs.pivot_table(index="score", columns="sentiment", values="count", fill_value=0)
            fig = px.imshow(
                pivot,
                labels=dict(x="Sentiment", y="Stars", color="Count"),
                color_continuous_scale="YlOrRd",
                aspect="auto",
            )
            fig.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig, width="stretch")

# -- Reviews tab ------------------------------------------------------------
with tabs[tab_idx["Reviews"]]:
    st.subheader("Review details")

    search_query = st.text_input("Search reviews", placeholder="Type to filter by content...")

    display_df = df.copy()
    if search_query:
        display_df = display_df[display_df["content"].str.contains(search_query, case=False, na=False, regex=False)]

    st.caption(f"Showing {len(display_df)} reviews")

    # If in comparison mode, load side-by-side data
    comp_data: dict = {}
    if method_option == "Both (comparison)":
        comp_full = comparison_reviews_df(selected_app)
        comp_data = {row["review_id"]: row for _, row in comp_full.iterrows()}

    for _, row in display_df.head(100).iterrows():
        score_stars = "⭐" * int(row["score"]) if pd.notna(row["score"]) else ""
        sentiment_badge = row["sentiment"] if pd.notna(row["sentiment"]) else "unclassified"
        version_text = row["app_version"] if pd.notna(row["app_version"]) else "unknown"
        date_text = row["review_date"].strftime("%Y-%m-%d") if pd.notna(row["review_date"]) else ""

        header = f"{score_stars} — **{sentiment_badge}** | v{version_text} | {date_text}"
        with st.expander(header, expanded=False):
            st.write(row["content"])

            if method_option == "Both (comparison)" and row["review_id"] in comp_data:
                cr = comp_data[row["review_id"]]
                bc1, bc2 = st.columns(2)
                with bc1:
                    st.markdown(f"**LLM:** {cr['llm_sentiment']}")
                    if pd.notna(cr.get("llm_justification")):
                        st.caption(cr["llm_justification"])
                with bc2:
                    st.markdown(f"**NLP:** {cr['nlp_sentiment']} (conf {cr.get('nlp_confidence', '—')})")
                    if pd.notna(cr.get("lda_topic_words")):
                        st.caption(f"LDA: {cr['lda_topic_words']}")
            else:
                if pd.notna(row.get("justification")):
                    st.caption(f"**LLM justification:** {row['justification']}")
                if pd.notna(row.get("lda_topic_words")):
                    st.caption(f"**LDA topic:** {row['lda_topic_words']}")

            topics = row.get("topics_list", [])
            if topics:
                st.caption(f"**Topics:** {', '.join(topics)}")
            if pd.notna(row.get("reply_content")):
                st.caption(f"**Dev reply:** {row['reply_content']}")

    if len(display_df) > 100:
        st.info(f"Showing first 100 of {len(display_df)} reviews. Use the search filter to narrow down.")
