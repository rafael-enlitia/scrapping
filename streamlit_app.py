"""Interactive dashboard for App Feedback Monitor — LLM vs NLP comparison."""

from __future__ import annotations

import io
import json

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud

import src.db.queries as _q
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

METHOD_LABELS = {"llm": "LLM (GPT / Ollama / IAEDU)", "nlp": "NLP (BERT + LDA)"}


# ---------------------------------------------------------------------------
# Cached wrappers — caching lives here, not in queries.py
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def get_app_ids() -> list[str]:
    return _q.get_app_ids()


@st.cache_data(ttl=60, show_spinner=False)
def get_versions(app_id: str) -> list[str]:
    return _q.get_versions(app_id)


@st.cache_data(ttl=60, show_spinner=False)
def get_reviews_df(app_id: str | None = None, method: str | None = None) -> pd.DataFrame:
    return _q.get_reviews_df(app_id, method)


@st.cache_data(ttl=60, show_spinner=False)
def sentiment_by_version(app_id: str, method: str | None = None) -> pd.DataFrame:
    return _q.sentiment_by_version(app_id, method)


@st.cache_data(ttl=60, show_spinner=False)
def topics_by_version(app_id: str, method: str | None = None) -> pd.DataFrame:
    return _q.topics_by_version(app_id, method)


@st.cache_data(ttl=60, show_spinner=False)
def avg_score_by_version(app_id: str) -> pd.DataFrame:
    return _q.avg_score_by_version(app_id)


@st.cache_data(ttl=60, show_spinner=False)
def sentiment_over_time(app_id: str, method: str | None = None) -> pd.DataFrame:
    return _q.sentiment_over_time(app_id, method)


@st.cache_data(ttl=60, show_spinner=False)
def sentiment_vs_score(app_id: str, method: str | None = None) -> pd.DataFrame:
    return _q.sentiment_vs_score(app_id, method)


@st.cache_data(ttl=60, show_spinner=False)
def sentiment_comparison(app_id: str) -> pd.DataFrame:
    return _q.sentiment_comparison(app_id)


@st.cache_data(ttl=60, show_spinner=False)
def agreement_matrix(app_id: str) -> pd.DataFrame:
    return _q.agreement_matrix(app_id)


@st.cache_data(ttl=60, show_spinner=False)
def agreement_rate(app_id: str) -> dict:
    return _q.agreement_rate(app_id)


@st.cache_data(ttl=60, show_spinner=False)
def comparison_reviews_df(app_id: str) -> pd.DataFrame:
    return _q.comparison_reviews_df(app_id)


@st.cache_data(ttl=60, show_spinner=False)
def lda_topic_distribution(app_id: str) -> pd.DataFrame:
    return _q.lda_topic_distribution(app_id)


@st.cache_data(ttl=300, show_spinner=False)
def get_embedding_clusters(app_id: str | None = None) -> pd.DataFrame | None:
    return _q.get_embedding_clusters(app_id)


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")
st.sidebar.subheader("Data")

app_ids = get_app_ids()
if not app_ids:
    st.warning("No reviews in the database yet.")
    st.info("**Getting started:** Go to ⚙️ Pipeline Control in the sidebar, enter an App ID (e.g. `com.whatsapp`), then run Scrape → LLM Classify → NLP Classify in order.")
    st.page_link("pages/pipeline_control.py", label="Go to Pipeline Control →", icon="⚙️")
    st.stop()

selected_app = st.sidebar.selectbox("App", app_ids)
st.session_state["shared_app_id"] = selected_app

st.sidebar.subheader("Classification")

# Method selector
method_option = st.sidebar.radio(
    "Classification method",
    ["LLM", "NLP", "Both (comparison)"],
    index=0,
    help="'Both' adds a Comparison tab with LLM vs NLP agreement metrics, side-by-side pies and disagreement samples.",
)
method_filter: str | None = None
if method_option == "LLM":
    method_filter = "llm"
elif method_option == "NLP":
    method_filter = "nlp"
elif method_option == "Both (comparison)":
    st.sidebar.caption("ℹ️ **Both** mode adds a Comparison tab. Charts use LLM sentiment where available, NLP otherwise.")

versions = get_versions(selected_app)
selected_versions = st.sidebar.multiselect("Versions", versions, default=versions)

# NLP cannot produce "mixed" — show only the sentiments that apply to the active method
_nlp_sentiments = [s for s in SENTIMENT_VALUES if s != "mixed"]
_sentiment_options = SENTIMENT_VALUES if method_filter != "nlp" else _nlp_sentiments
selected_sentiments = st.sidebar.multiselect("Sentiments", _sentiment_options, default=list(_sentiment_options))

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
if st.sidebar.button("🔄 Clear cache & reload", help="Force-refresh all charts from the database (cache expires every 60 s automatically)"):
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
classified_count = int(classified_mask.sum())
unclassified_count = int((~classified_mask).sum())
total_count = len(df)
pct_classified = classified_count / total_count if total_count else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total reviews", total_count)
col2.metric("Classified", classified_count)
col3.metric("Unclassified", unclassified_count)
_avg_score = df["score"].mean()
col4.metric("Avg rating", f"{_avg_score:.1f}" if pd.notna(_avg_score) else "—")
col5.metric("Versions", df["app_version"].nunique())

if total_count > 0:
    st.progress(
        pct_classified,
        text=f"{pct_classified:.0%} classified ({classified_count} of {total_count} reviews)",
    )

if unclassified_count > 0:
    st.info(f"{unclassified_count} reviews still need **{method_badge}** classification.")
    st.page_link("pages/pipeline_control.py", label="Go to Pipeline Control to classify them", icon="⚙️")

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
            if selected_sentiments:
                sv = sv[sv["sentiment"].isin(selected_sentiments)]
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
    all_topics: list[str] = [t for topics in df["topics_list"] for t in topics]

    if method_filter == "nlp":
        st.subheader("LDA-discovered topics")
        lda_dist = lda_topic_distribution(selected_app)

        if not lda_dist.empty and selected_versions and len(selected_versions) < len(versions):
            visible_lda_ids = df["lda_topic_id"].dropna().unique().tolist()
            lda_dist = lda_dist[lda_dist["topic_id"].isin(visible_lda_ids)]

        if not lda_dist.empty:
            if selected_versions and len(selected_versions) < len(versions):
                id_counts = df["lda_topic_id"].value_counts().rename_axis("topic_id").reset_index(name="count")
                lda_dist = lda_dist.drop(columns=["count"], errors="ignore").merge(id_counts, on="topic_id", how="inner")

            for _, row in lda_dist.iterrows():
                st.markdown(f"**Topic {int(row['topic_id'])}** ({int(row['count'])} reviews): _{row['topic_words']}_")
            # Use top-3 keywords as x-axis tick labels so bars are self-explanatory
            lda_dist = lda_dist.copy()
            lda_dist["label"] = lda_dist.apply(
                lambda r: ", ".join(str(r["topic_words"]).split(", ")[:3]) if pd.notna(r["topic_words"]) else f"T{int(r['topic_id'])}",
                axis=1,
            )
            fig = px.bar(lda_dist, x="label", y="count", text="count", hover_data={"topic_words": True, "topic_id": True})
            fig.update_layout(xaxis_title="Topic keywords", yaxis_title="Reviews", xaxis_tickangle=-30, margin=dict(t=20, b=60))
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
    # Use the scrape language if stored on reviews, otherwise default to portuguese
    _review_langs = df["language"].dropna().unique().tolist() if "language" in df.columns else []
    _wc_lang = _review_langs[0] if len(_review_langs) == 1 else "portuguese"
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
            # Apply sentiment filter
            if selected_sentiments:
                sot = sot[sot["sentiment"].isin(selected_sentiments)]
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
                lda_dist_cmp = lda_topic_distribution(selected_app)
                if not lda_dist_cmp.empty:
                    lda_dist_cmp = lda_dist_cmp.copy()
                    lda_dist_cmp["label"] = lda_dist_cmp.apply(
                        lambda r: ", ".join(str(r["topic_words"]).split(", ")[:3]) if pd.notna(r["topic_words"]) else f"T{int(r['topic_id'])}",
                        axis=1,
                    )
                    fig = px.bar(lda_dist_cmp, x="label", y="count", hover_data={"topic_words": True})
                    fig.update_layout(xaxis_title="Topic keywords", xaxis_tickangle=-30, margin=dict(t=20, b=60))
                    st.plotly_chart(fig, width="stretch")

            st.subheader("Sample disagreements")
            comp_df = comparison_reviews_df(selected_app)
            disagree_df = comp_df[comp_df["llm_sentiment"] != comp_df["nlp_sentiment"]]
            for _, row in disagree_df.head(20).iterrows():
                score_val = row["score"]
                stars = "⭐" * int(score_val) if pd.notna(score_val) and 1 <= int(score_val) <= 5 else ""
                llm_conf = f" (conf {row.get('llm_confidence', '—')})" if pd.notna(row.get("llm_confidence")) else ""
                nlp_conf = f" (conf {row.get('nlp_confidence', '—')})" if pd.notna(row.get("nlp_confidence")) else ""
                header = f"{stars} LLM: **{row['llm_sentiment']}**{llm_conf} | NLP: **{row['nlp_sentiment']}**{nlp_conf}"
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
            # Apply sentiment filter
            if selected_sentiments:
                svs = svs[svs["sentiment"].isin(selected_sentiments)]
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

    rc1, rc2, rc3 = st.columns([3, 2, 1])
    with rc1:
        search_query = st.text_input("Search reviews", placeholder="Type to filter by content...", label_visibility="collapsed")
    with rc2:
        sort_by = st.selectbox(
            "Sort by",
            ["Date (newest)", "Date (oldest)", "Score (high→low)", "Score (low→high)", "Sentiment"],
            label_visibility="collapsed",
        )
    with rc3:
        csv_data = df.drop(columns=["topics"], errors="ignore").rename(columns={"topics_list": "topics"}).to_csv(index=False)
        st.download_button("⬇ CSV", csv_data, f"{selected_app}_reviews.csv", "text/csv", use_container_width=True)

    display_df = df.copy()
    if search_query:
        display_df = display_df[display_df["content"].str.contains(search_query, case=False, na=False, regex=False)]

    _sort_map = {
        "Date (newest)": ("review_date", False),
        "Date (oldest)": ("review_date", True),
        "Score (high→low)": ("score", False),
        "Score (low→high)": ("score", True),
        "Sentiment": ("sentiment", True),
    }
    _sort_col, _sort_asc = _sort_map[sort_by]
    if _sort_col in display_df.columns:
        display_df = display_df.sort_values(_sort_col, ascending=_sort_asc, na_position="last")

    total_display = len(display_df)
    page_size = 20
    total_pages = max(1, (total_display - 1) // page_size + 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * page_size
    page_df = display_df.iloc[start_idx : start_idx + page_size]

    st.caption(f"Page {page}/{total_pages} — {total_display} reviews match filters")

    comp_data: dict = {}
    if method_option == "Both (comparison)":
        comp_full = comparison_reviews_df(selected_app)
        comp_data = {row["review_id"]: row for _, row in comp_full.iterrows()}

    for _, row in page_df.iterrows():
        score_val = row["score"]
        score_stars = "⭐" * int(score_val) if pd.notna(score_val) and 1 <= int(score_val) <= 5 else ""
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
                    llm_conf = f" (conf {cr['llm_confidence']:.2f})" if pd.notna(cr.get("llm_confidence")) else ""
                    st.markdown(f"**LLM:** {cr['llm_sentiment']}{llm_conf}")
                    if pd.notna(cr.get("llm_justification")):
                        st.caption(cr["llm_justification"])
                with bc2:
                    nlp_conf = f" (conf {cr['nlp_confidence']:.2f})" if pd.notna(cr.get("nlp_confidence")) else ""
                    st.markdown(f"**NLP:** {cr['nlp_sentiment']}{nlp_conf}")
                    if pd.notna(cr.get("lda_topic_words")):
                        st.caption(f"LDA: {cr['lda_topic_words']}")
            else:
                if pd.notna(row.get("justification")):
                    st.caption(f"**LLM justification:** {row['justification']}")
                if pd.notna(row.get("confidence")):
                    st.caption(f"**Confidence:** {row['confidence']:.2f}")
                if pd.notna(row.get("lda_topic_words")):
                    st.caption(f"**LDA topic:** {row['lda_topic_words']}")

            topics = row.get("topics_list", [])
            if topics:
                st.caption(f"**Topics:** {', '.join(topics)}")
            if pd.notna(row.get("reply_content")):
                st.caption(f"**Dev reply:** {row['reply_content']}")
