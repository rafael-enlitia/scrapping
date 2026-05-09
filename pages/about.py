"""About / Help — comprehensive guide to the App Feedback Monitor."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="About & Help", page_icon="📖", layout="wide")

# ── Hero ─────────────────────────────────────────────────────────────────────
st.title("📖 About & Help")
st.markdown(
    """
    **App Feedback Monitor** is an open-source platform that automatically collects, classifies
    and visualises Google Play Store reviews. It runs two independent classification pipelines —
    a generative AI (LLM) pipeline and a traditional NLP (BERT + LDA) pipeline — so you can
    compare their outputs side by side and build a deeper understanding of what your users are saying.
    """
)

st.divider()

# ── What the app does ─────────────────────────────────────────────────────────
st.header("🚀 What this app does")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### 🕷️ Collect")
    st.markdown(
        "Scrapes any Google Play app's reviews via the Play Store API. "
        "Filters spam / very short reviews automatically. "
        "Deduplicates on every run so you never store the same review twice."
    )
with col2:
    st.markdown("#### 🤖 Classify")
    st.markdown(
        "Two independent pipelines classify each review for **sentiment** "
        "(positive / negative / neutral / mixed) and **topic** "
        "(performance, bugs, UI, pricing, …). "
        "Run one or both and compare them."
    )
with col3:
    st.markdown("#### 📊 Visualise")
    st.markdown(
        "Interactive dashboard with sentiment trends, topic frequency, "
        "word cloud, UMAP cluster map, LLM vs NLP comparison, "
        "and a full searchable review table with CSV export."
    )

st.divider()

# ── Quickstart ────────────────────────────────────────────────────────────────
st.header("⚡ Quickstart")

st.markdown(
    """
    The fastest way to get started is through the **⚙️ Pipeline Control** page in the sidebar —
    no terminal required after the initial `streamlit run`.

    If you prefer the terminal, here is the full workflow:
    """
)

with st.expander("Terminal workflow (click to expand)", expanded=False):
    st.code(
        """\
# 1. Scrape reviews
python -m scripts.scrape --app-id com.whatsapp --count 500

# 2. Classify with both pipelines
python -m scripts.classify --app-id com.whatsapp --limit 500
python -m scripts.classify_nlp --app-id com.whatsapp

# 3. (Optional) Compute semantic embeddings + clusters
python -m scripts.embed --app-id com.whatsapp --n-clusters 8

# 4. Open the dashboard
streamlit run streamlit_app.py
""",
        language="bash",
    )

st.markdown("**Browser-only workflow:**")
st.markdown(
    """
    1. Go to **⚙️ Pipeline Control** in the sidebar
    2. Enter your App ID (e.g. `com.whatsapp`) in the sidebar
    3. Run **Scrape → LLM Classify → NLP Classify** in sequence
    4. Optionally run **Embeddings** for the cluster visualisation
    5. Return to the main **📱 App Feedback Monitor** page to explore results
    """
)

st.divider()

# ── Pages ─────────────────────────────────────────────────────────────────────
st.header("🗂️ Pages in this app")

pages = [
    ("📱 App Feedback Monitor", "Main dashboard with all charts, filters and the review table."),
    ("⚙️ Pipeline Control", "Launch scraping, classification and embedding pipelines from the browser. Live log streaming with stop/cancel support."),
    ("🏷️ Review Labeling", "Manually label reviews to create a gold evaluation dataset saved to `data/gold.jsonl`."),
    ("📖 About & Help", "This page — full guide, FAQ and taxonomy reference."),
]
for icon_title, desc in pages:
    st.markdown(f"**{icon_title}** — {desc}")

st.divider()

# ── Classification methods ────────────────────────────────────────────────────
st.header("🔬 Classification methods")

tab_llm, tab_nlp = st.tabs(["🤖 LLM (GPT / Ollama)", "🧬 NLP (BERT + LDA)"])

with tab_llm:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### How it works")
        st.markdown(
            """
            Each review is sent to a large language model (OpenAI GPT or a local Ollama model)
            with a carefully crafted prompt that instructs it to return a structured JSON response
            containing sentiment, confidence, topics, and a brief justification.

            The prompt includes the full topic taxonomy so the model maps directly to the same
            labels used across the app.
            """
        )
        st.markdown("#### Providers")
        st.markdown(
            """
            | Provider | Setup | Cost |
            |----------|-------|------|
            | **OpenAI** (GPT-4o-mini, etc.) | Set `OPENAI_API_KEY` in `.env` | Per-call billing |
            | **Ollama** (llama3, mistral, …) | Run Ollama locally | Free |

            Switch providers with `LLM_PROVIDER=openai` or `LLM_PROVIDER=ollama` in `.env`,
            or override per-run with `--provider` in Pipeline Control.
            """
        )
    with col2:
        st.markdown("#### Strengths")
        st.markdown(
            """
            - Understands context, sarcasm and irony
            - Can return `mixed` sentiment
            - Provides a human-readable justification for each classification
            - Works well with short, colloquial text
            """
        )
        st.markdown("#### Limitations")
        st.markdown(
            """
            - Costs money (OpenAI) or requires a local server (Ollama)
            - Non-deterministic — same review may produce different labels on re-run
            - Exponential backoff handles rate limits (up to 5 retries), but sustained
              bursts may still exhaust the retry budget
            """
        )

with tab_nlp:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### How it works")
        st.markdown(
            """
            The NLP pipeline has two independent stages:

            **Sentiment — BERT**
            Uses `nlptown/bert-base-multilingual-uncased-sentiment`, a multilingual BERT model
            fine-tuned on product reviews to predict a 1–5 star rating, which is then mapped to
            positive / negative / neutral.

            **Topics — LDA**
            Review text is cleaned (stopwords removed, stemmed), then fed to a Latent Dirichlet
            Allocation model that discovers recurring themes unsupervised. LDA topics are mapped
            heuristically to the fixed taxonomy.
            """
        )
        st.markdown("#### First-run behaviour")
        st.markdown(
            """
            - Downloads the BERT model from HuggingFace (~440 MB, cached in `~/.cache/`)
            - Downloads NLTK stopwords and RSLP stemmer
            - Trains LDA on the full review corpus and saves to `data/lda_model.pkl`

            Subsequent runs reuse the cached model — only new reviews are classified.
            """
        )
    with col2:
        st.markdown("#### Strengths")
        st.markdown(
            """
            - Completely free — no API key required
            - Deterministic — same text always produces the same result
            - Fast batch inference with GPU support (auto-detected)
            - Discovers topics unsupervised — useful for exploratory analysis
            """
        )
        st.markdown("#### Limitations")
        st.markdown(
            """
            - Cannot output `mixed` sentiment (only positive / negative / neutral)
            - BERT truncates reviews > 512 tokens (~300–400 words)
            - LDA needs ≥ 10 documents to train; topics may be noisy with < 100 reviews
            - Language-dependent — use `--language english` for English reviews
            """
        )

st.divider()

# ── Dashboard tabs ────────────────────────────────────────────────────────────
st.header("📊 Dashboard tabs explained")

tabs = st.tabs(["Sentiment", "Topics", "Evolution", "Comparison", "Score Analysis", "Reviews"])

with tabs[0]:
    st.markdown(
        """
        **What you see:**
        - Pie chart — overall sentiment distribution across all filtered reviews
        - Stacked bar — sentiment breakdown per app version

        **Useful question:** *"Did version 3.1 generate more negative reviews than 3.0?"*

        **Tip:** Filter by specific versions in the sidebar to compare them directly.
        """
    )

with tabs[1]:
    st.markdown(
        """
        **What you see:**
        - **Topic frequency** bar chart — which topics are most mentioned
        - **Topics by version** stacked bar — how topic distribution shifts across versions
        - **Topic co-occurrence** horizontal bar — which topic pairs appear together most
        - **Word Cloud** — generated from preprocessed (stopword-free, stemmed) review text
        - **LDA topics** (NLP mode only) — raw LDA topic IDs with their top words

        **Useful question:** *"Since the last update, are users talking more about bugs or performance?"*

        **Tip:** The word cloud updates live as you change sidebar filters.
        """
    )

with tabs[2]:
    st.markdown(
        """
        **What you see:**
        - **Average score by version** line chart — how the star rating evolves
        - **Sentiment over time** area chart — weekly positive / negative / neutral counts

        **Useful question:** *"The rating dropped in January — was the sentiment also worse?"*

        **Tip:** Narrow the date range in the sidebar to zoom in on a specific period.
        """
    )

with tabs[3]:
    st.markdown(
        """
        Only visible when **Both (comparison)** is selected as the classification method.

        **What you see:**
        - Agreement rate — % of reviews where both methods returned the same sentiment
        - Side-by-side pie charts — LLM vs NLP sentiment distribution
        - Agreement matrix heatmap — cross-tab of LLM labels vs NLP labels
        - Topic distribution comparison — taxonomy (LLM) vs LDA topics (NLP)
        - Sample disagreements — expandable cards for reviews where the two methods disagree,
          including LLM justification and LDA topic words

        **Useful question:** *"Where do LLM and NLP most commonly disagree, and why?"*
        """
    )

with tabs[4]:
    st.markdown(
        """
        **What you see:**
        - **Score distribution** bar chart — count of 1–5 star reviews
        - **Sentiment vs Star Rating** heatmap — how sentiment labels map to star counts

        **Useful question:** *"How many 4-star reviews were classified as negative?"*

        **Tip:** A high count of 3-star reviews with negative sentiment often indicates
        users who are disappointed but unwilling to leave a 1-star review.
        """
    )

with tabs[5]:
    st.markdown(
        """
        **What you see:**
        - Searchable list of all reviews matching the current filters
        - Each card shows: stars, sentiment badge, version, date
        - Expand a card to see: full text, topics, LLM justification, LDA topic words,
          developer reply, and (in Both mode) side-by-side LLM vs NLP outputs
        - **Export CSV** button above the tabs exports all filtered reviews

        **Limits:** Only the first 100 reviews are rendered to keep the page fast.
        Use the search box or sidebar filters to narrow down to the reviews you need.
        """
    )

st.divider()

# ── Embeddings & clusters ─────────────────────────────────────────────────────
st.header("🗺️ Embeddings & Cluster visualisation")

st.markdown(
    """
    Available in **⚙️ Pipeline Control → 📊 Embeddings**.

    The embedding pipeline runs three stages:

    | Stage | What it does |
    |-------|-------------|
    | **BERT embeddings** | Encodes each review as a dense vector (mean-pooled from the base BERT model's last hidden layer) |
    | **UMAP** | Reduces the high-dimensional vectors to 2-D for visualisation, preserving local neighbourhood structure |
    | **KMeans** | Groups the 2-D points into *k* clusters (you choose *k*) |

    The resulting scatter plot lets you visually explore groups of semantically similar reviews.
    Hover over any point to see the review text, star rating and app version.

    Results are saved to `data/embeddings.npz` and reload automatically next time you open
    the Embeddings tab.

    **When to use this:**
    - To discover natural groupings in user feedback that the fixed taxonomy might miss
    - To spot outlier clusters (e.g. a sudden burst of identical complaints)
    - As an exploratory step before deciding on the number of LDA topics
    """
)

st.divider()

# ── Taxonomy ──────────────────────────────────────────────────────────────────
st.header("🏷️ Topic taxonomy")

st.markdown("Both pipelines classify into the same 10 topic categories:")

taxonomy = [
    ("performance", "Speed, lag, battery drain, loading times, crashes"),
    ("ui_ux", "Design, layout, visual appearance, navigation, accessibility"),
    ("bugs", "Errors, glitches, broken features, unexpected behaviour"),
    ("features", "Missing features, feature requests, praise for specific features"),
    ("pricing", "Cost, subscriptions, in-app purchases, ads, value for money"),
    ("privacy_security", "Data privacy, permissions, account security, data breaches"),
    ("customer_support", "Support responsiveness, help quality, response time"),
    ("updates", "Effects of recent updates, changelog, version regressions"),
    ("usability", "Ease of use, learning curve, onboarding, documentation"),
    ("other", "Topics that don't clearly fit any of the above categories"),
]

col1, col2 = st.columns(2)
for i, (key, desc) in enumerate(taxonomy):
    target = col1 if i % 2 == 0 else col2
    with target:
        st.markdown(f"**`{key}`** — {desc}")

st.divider()

# ── Filters explained ─────────────────────────────────────────────────────────
st.header("🔍 Sidebar filters explained")

st.markdown(
    """
    All filters on the main dashboard work **inclusively** — every filter starts with
    all values selected so you see the complete dataset by default. Remove values to narrow down.

    | Filter | Default | Effect when narrowed |
    |--------|---------|---------------------|
    | **App** | First app in DB | Shows data for a single app |
    | **Classification method** | LLM | Switches all charts and the review table to that method's data |
    | **Versions** | All versions | Restricts **all** charts — including aggregation charts (sentiment by version, topics by version, average score) — and the review table |
    | **Sentiments** | All sentiments | Hides reviews with non-selected sentiments from the table and pie chart. In **NLP mode**, `mixed` is removed automatically (NLP can never produce it). |
    | **Topics** | All topics | Shows only reviews that mention at least one selected topic |
    | **Date range** | Full range | Restricts the review table and time-series charts to the selected period |

    > **Note:** The date-range filter applies to the review table and time-series charts but
    > not to the per-version aggregation queries (which are cached per-app for performance).
    """
)

st.divider()

# ── Gold dataset & evaluation ─────────────────────────────────────────────────
st.header("🥇 Gold dataset & evaluation")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### What is the gold dataset?")
    st.markdown(
        """
        The gold dataset (`data/gold.jsonl`) is a set of reviews that have been **manually
        labelled** by a human with the correct sentiment and topics. It is used as a ground
        truth to measure how accurate the automatic classifiers are.

        Each line is a JSON object:
        ```json
        {"review_id": "abc123", "sentiment": "negative", "topics": ["bugs", "performance"]}
        ```

        Build the gold dataset using the **🏷️ Review Labeling** page in the sidebar.
        """
    )
with col2:
    st.markdown("#### Running evaluation")
    st.markdown(
        """
        Once you have gold labels, evaluate either pipeline from the terminal:
        ```bash
        # Evaluate both and compare
        python -m scripts.evaluate --gold data/gold.jsonl

        # Evaluate only LLM, save plots
        python -m scripts.evaluate --gold data/gold.jsonl --method llm --save-plots
        ```

        **Output includes:**
        - Overall accuracy
        - Macro F1 score
        - Per-class classification report
        - Confusion matrix
        - Per-topic F1 (LLM only)
        - Side-by-side comparison when `--method both`
        """
    )

st.divider()

# ── FAQ / Troubleshooting ─────────────────────────────────────────────────────
st.header("❓ Frequently asked questions")

faqs = [
    (
        "The dashboard is empty — no reviews show up",
        "You need to run the scraper first. Go to **⚙️ Pipeline Control → 🕷️ Scrape**, "
        "enter your App ID and click **▶ Start Scraping**. Once reviews are collected, "
        "run at least one classifier before the charts will show sentiment data.",
    ),
    (
        "NLP classifies 0 reviews and shows a UNIQUE constraint error",
        "Your database has an old schema that only allows one classification per review. "
        "Run `python -m scripts.migrate_db` from the terminal to fix it. "
        "This is a one-time migration for databases created before dual-method support was added.",
    ),
    (
        "LLM classification is failing with rate limit errors",
        "The system automatically retries up to 5 times with exponential backoff. "
        "If it still fails, wait a few minutes and try again with a smaller `--limit`. "
        "Consider switching to **Ollama** for unlimited free local inference.",
    ),
    (
        "The dashboard doesn't show new data after I ran a classifier",
        "Streamlit caches query results for 60 seconds. Click **Clear cache & reload** "
        "at the bottom of the sidebar to force a fresh fetch immediately.",
    ),
    (
        "The first NLP run is very slow",
        "On the first run, the BERT model (~440 MB) is downloaded from HuggingFace and cached. "
        "Subsequent runs are much faster. If you have a GPU, it will be used automatically.",
    ),
    (
        "Embeddings take a long time to compute",
        "BERT inference is CPU-bound. Expect 30–60 seconds for 500 reviews on a modern laptop "
        "without a GPU. Use the **Limit reviews** field in the Embeddings tab to test with fewer "
        "reviews first.",
    ),
    (
        "The word cloud is dominated by irrelevant short words",
        "The word cloud uses the same preprocessing as the NLP pipeline — stopwords are removed "
        "and words are stemmed. If you still see noise, try narrowing the filters to a specific "
        "sentiment or topic to focus the cloud on a subset of reviews.",
    ),
    (
        "The `mixed` sentiment option disappeared in NLP mode",
        "This is expected. The BERT model used by the NLP pipeline predicts a 1–5 star rating "
        "which maps to positive / negative / neutral only — it cannot output `mixed`. "
        "The dashboard removes `mixed` from the Sentiments filter automatically when NLP mode "
        "is active to avoid empty results. Switch to **LLM** or **Both** mode to see `mixed` again.",
    ),
    (
        "The Pipeline Control log is not updating",
        "Logs auto-refresh every 3 seconds while a pipeline is running. "
        "If the log appears stuck, click **🔄 Refresh log** to force an immediate update. "
        "If the pipeline has already finished, the log stops auto-refreshing but remains visible.",
    ),
    (
        "I stopped a pipeline but it seems to still be running",
        "The stop button sends SIGTERM to the subprocess. If the process is in the middle of a "
        "heavy operation (e.g. downloading the BERT model), it may take a few seconds to terminate. "
        "If it does not stop, restart the Streamlit server.",
    ),
    (
        "How do I analyse reviews in English instead of Portuguese?",
        "Pass `--language english` when running NLP classification: "
        "`python -m scripts.classify_nlp --language english`. "
        "Also pass `--lang en --country us` when scraping to fetch English reviews.",
    ),
    (
        "Can I analyse multiple apps at the same time?",
        "Yes. Scrape and classify each app separately using its own `--app-id`. "
        "All apps are stored in the same database and you can switch between them "
        "with the **App** selector in the sidebar.",
    ),
]

for question, answer in faqs:
    with st.expander(f"**{question}**"):
        st.markdown(answer)

st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption(
    "App Feedback Monitor — built with Streamlit, SQLAlchemy, HuggingFace Transformers, "
    "scikit-learn, OpenAI, Plotly and UMAP. "
    "See the README for full technical documentation."
)
