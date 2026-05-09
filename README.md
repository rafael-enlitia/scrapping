# App Feedback Monitor

Platform for sentiment analysis and topic discovery on mobile app reviews from the Google Play Store. Two independent classification pipelines run on the same data, and the dashboard lets you compare them side-by-side:

| Pipeline | Sentiment | Topics |
|----------|-----------|--------|
| **LLM** (OpenAI / Ollama) | Prompted classification (positive, negative, neutral, mixed) | Fixed taxonomy via prompt |
| **NLP** (traditional) | BERT multilingual model (`nlptown/bert-base-multilingual-uncased-sentiment`) | Unsupervised LDA topic modelling |

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Commands Reference](#commands-reference)
  - [Scrape reviews](#1-scrape-reviews)
  - [Classify with LLM](#2-classify-with-llm)
  - [Classify with NLP](#3-classify-with-nlp-bert--lda)
  - [Compute embeddings & clusters](#4-compute-embeddings--clusters)
  - [Run the dashboard](#5-run-the-dashboard)
  - [Label reviews (gold dataset)](#6-label-reviews-gold-dataset)
  - [Evaluate accuracy](#7-evaluate-accuracy)
  - [Migrate database](#8-migrate-database)
- [Dashboard Guide](#dashboard-guide)
- [Project Structure](#project-structure)
- [Taxonomy](#taxonomy)
- [Gold Label Format](#gold-label-format)
- [Known Limitations](#known-limitations)

---

## Requirements

- Python 3.10+
- ~3 GB disk for PyTorch + BERT model weights (downloaded on first NLP run)
- An OpenAI API key **or** a local [Ollama](https://ollama.com) server (only needed for the LLM pipeline)
- Internet connection for scraping and for the first BERT model download

---

## Installation

```bash
# Clone and enter the project
cd scrapping

# Create virtual environment
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and edit the environment file
cp .env.example .env
```

If you are upgrading from a version that only had the LLM pipeline, run the database migration (see [Migrate database](#7-migrate-database)).

---

## Configuration

All settings are read from a `.env` file in the project root. Available variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM backend: `openai` or `ollama` |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `BERT_MODEL` | `nlptown/bert-base-multilingual-uncased-sentiment` | HuggingFace model for NLP sentiment |
| `LDA_NUM_TOPICS` | `8` | Number of LDA topics to discover |
| `NLP_BATCH_SIZE` | `32` | Batch size for BERT inference |
| `DEFAULT_APP_ID` | `com.whatsapp` | Default app to scrape when `--app-id` is omitted |
| `SCRAPE_LANG` | `pt` | Review language filter |
| `SCRAPE_COUNTRY` | `pt` | Country store to scrape from |

---

## Commands Reference

All scripts are run as Python modules from the project root.

**Quick end-to-end workflow:**

```bash
# 1. Scrape reviews
python -m scripts.scrape --app-id com.whatsapp --count 500

# 2. Classify with both pipelines
python -m scripts.classify --app-id com.whatsapp --limit 500
python -m scripts.classify_nlp --app-id com.whatsapp

# 3. (Optional) Compute embeddings + clusters
python -m scripts.embed --app-id com.whatsapp --n-clusters 8

# 4. Open the dashboard
streamlit run streamlit_app.py
```

> **Tip:** You can also trigger scraping, classification and embedding computation directly from the **Pipeline Control** page in the dashboard — no terminal required.

### 1. Scrape reviews

Pull reviews from the Google Play Store into the local SQLite database.

```bash
python -m scripts.scrape [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--app-id` | string | `DEFAULT_APP_ID` from `.env` | Google Play package name (e.g. `com.whatsapp`) |
| `--count` | int | `500` | Number of reviews to fetch |
| `--lang` | string | `SCRAPE_LANG` from `.env` | Language code (`pt`, `en`, ...) |
| `--country` | string | `SCRAPE_COUNTRY` from `.env` | Country code (`pt`, `us`, ...) |
| `--sort` | choice | `newest` | Sort order: `newest` or `most_relevant` |

**Examples:**

```bash
# Use all defaults from .env (quick sanity check)
python -m scripts.scrape

# Scrape 500 newest Portuguese reviews of WhatsApp
python -m scripts.scrape --app-id com.whatsapp --count 500

# Scrape a small batch just to test the connection
python -m scripts.scrape --app-id com.spotify.music --count 50

# Scrape 1000 most relevant English reviews of Instagram from the US store
python -m scripts.scrape --app-id com.instagram.android --count 1000 --lang en --country us --sort most_relevant

# Scrape two apps back-to-back
python -m scripts.scrape --app-id com.whatsapp --count 500 && \
python -m scripts.scrape --app-id com.instagram.android --count 500
```

Duplicate reviews (same `review_id`) are automatically skipped.

---

### 2. Classify with LLM

Send unclassified reviews to an LLM (OpenAI or Ollama) for sentiment and topic classification.

```bash
python -m scripts.classify [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--limit` | int | all | Max reviews to classify in this run |
| `--app-id` | string | all apps | Filter to a specific app |
| `--provider` | choice | `LLM_PROVIDER` from `.env` | Override provider: `openai` or `ollama` |
| `--retry-failed` | flag | off | Retry reviews that failed before (defaults `--limit` to 50) |

**Examples:**

```bash
# Classify everything (no limit) — use only on small datasets
python -m scripts.classify

# Classify up to 100 reviews using the configured provider
python -m scripts.classify --limit 100

# Classify only WhatsApp reviews using Ollama
python -m scripts.classify --app-id com.whatsapp --provider ollama --limit 50

# Force OpenAI even if .env says ollama
python -m scripts.classify --provider openai --limit 200

# Classify new WhatsApp reviews with OpenAI, capped at 300
python -m scripts.classify --app-id com.whatsapp --provider openai --limit 300

# Retry reviews that previously failed
python -m scripts.classify --retry-failed

# Retry failed reviews for one specific app only
python -m scripts.classify --retry-failed --app-id com.whatsapp
```

Each review costs one LLM API call. Progress and failures are logged to the terminal.

---

### 3. Classify with NLP (BERT + LDA)

Run the traditional NLP pipeline: BERT for sentiment, LDA for topics. No API key needed.

```bash
python -m scripts.classify_nlp [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--limit` | int | all | Max reviews to classify |
| `--app-id` | string | all apps | Filter to a specific app |
| `--num-topics` | int | `LDA_NUM_TOPICS` from `.env` (default 8) | Number of LDA topics |
| `--retrain-lda` | flag | off | Force retrain the LDA model even if a saved one exists |
| `--language` | string | `portuguese` | Language for stopword removal and stemming |

**Examples:**

```bash
# Classify all unclassified reviews
python -m scripts.classify_nlp

# Classify 200 reviews with 12 LDA topics
python -m scripts.classify_nlp --limit 200 --num-topics 12

# Retrain LDA after scraping more data
python -m scripts.classify_nlp --retrain-lda

# Classify only one app and force retrain LDA after new data was added
python -m scripts.classify_nlp --app-id com.whatsapp --retrain-lda

# Tune topic granularity and retrain at the same time
python -m scripts.classify_nlp --num-topics 6 --retrain-lda

# Use English stopwords/stemming for an English-language app
python -m scripts.classify_nlp --app-id com.spotify.music --language english

# Full explicit run: app, topics, language, retrain
python -m scripts.classify_nlp \
  --app-id com.spotify.music \
  --num-topics 10 \
  --language english \
  --retrain-lda
```

**First run behaviour:**
1. Downloads the BERT model from HuggingFace (~440 MB, cached in `~/.cache/huggingface/`)
2. Downloads NLTK data (stopwords, RSLP stemmer)
3. Trains LDA on the entire review corpus and saves it to `data/lda_model.pkl`

Subsequent runs reuse the cached BERT model and saved LDA model.

---

### 4. Compute embeddings & clusters

Generate BERT sentence embeddings for all reviews, reduce them to 2-D with UMAP, and group them into semantic clusters with KMeans. Results are saved to `data/embeddings.npz` and visualised in the **Pipeline Control** page of the dashboard.

```bash
python -m scripts.embed [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--app-id` | string | all apps | Filter to a specific app |
| `--n-clusters` | int | `8` | Number of KMeans clusters |
| `--limit` | int | all | Max reviews to embed |

**Examples:**

```bash
# Embed all reviews with 8 clusters
python -m scripts.embed

# Embed WhatsApp reviews into 12 clusters
python -m scripts.embed --app-id com.whatsapp --n-clusters 12

# Quick test: embed only the first 100 reviews
python -m scripts.embed --limit 100 --n-clusters 5
```

> **Note:** BERT embedding is CPU-bound. Expect ~30–60 s for 500 reviews on a modern laptop without a GPU. The UMAP + KMeans step is fast (< 5 s).

---

### 5. Run the dashboard

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501` by default. The sidebar navigation includes three additional pages:

- **⚙️ Pipeline Control** — trigger scrape, LLM/NLP classification and embedding directly from the browser
- **🏷️ Review Labeling** — manually label reviews to build a gold evaluation dataset
- **📖 About & Help** — full in-app guide, FAQ, filter reference and taxonomy

```bash
# Run on a different port (e.g. if 8501 is in use)
streamlit run streamlit_app.py --server.port 8502

# Expose on all interfaces (for remote / VM access)
streamlit run streamlit_app.py --server.address 0.0.0.0
```

---

### 6. Label reviews (gold dataset)

Open the dashboard and navigate to **Review Labeling** in the sidebar. This page lets you manually assign sentiment and topics to build a gold-standard evaluation set, saved to `data/gold.jsonl`.

---

### 7. Evaluate accuracy

Compare predicted classifications against the human-labeled gold dataset.

```bash
python -m scripts.evaluate [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--gold` | string | **required** | Path to the gold JSONL file |
| `--method` | choice | `both` | Which pipeline to evaluate: `llm`, `nlp`, or `both` |
| `--save-plots` | flag | off | Save confusion matrix and F1 charts as HTML + PNG to `data/` |

**Examples:**

```bash
# Evaluate both pipelines and print a comparison summary
python -m scripts.evaluate --gold data/gold.jsonl

# Evaluate only the LLM pipeline
python -m scripts.evaluate --gold data/gold.jsonl --method llm

# Evaluate only the NLP pipeline
python -m scripts.evaluate --gold data/gold.jsonl --method nlp

# Evaluate only the LLM pipeline and save visual output
python -m scripts.evaluate --gold data/gold.jsonl --method llm --save-plots

# Evaluate both and save plots
python -m scripts.evaluate --gold data/gold.jsonl --method both --save-plots
```

Output includes: accuracy, macro F1, per-class classification report, confusion matrix, and (for LLM) per-topic F1 scores. When `--method both`, a side-by-side comparison summary is printed at the end.

> **Note:** Topic evaluation is only available for the LLM pipeline. LDA discovers topics unsupervised and they don't map 1:1 to the fixed taxonomy.

---

### 8. Migrate database

If you already had a database from the LLM-only version, run the migration to:

- Add new columns: `method`, `confidence`, `lda_topic_id`, `lda_topic_words`
- Backfill `method='llm'` on existing rows
- **Rebuild the `classifications` table** if the old schema had a single-column `UNIQUE(review_id)` constraint (which would block NLP rows from being stored alongside LLM rows). The migration replaces it with a composite `UNIQUE(review_id, method)` constraint, allowing both an `llm` and an `nlp` row per review.

```bash
python -m scripts.migrate_db
```

This is idempotent — safe to run multiple times. Does nothing if no database exists.

> **Important:** If the NLP pipeline reports `0 / N classified` with `UNIQUE constraint failed: classifications.review_id` errors, it means the old constraint is still in place. Run `python -m scripts.migrate_db` to fix it.

---

## Dashboard Guide

The sidebar provides filters for app, date range, version, sentiment, and topics. The **Classification method** radio button controls what data is shown:

| Mode | Behaviour |
|------|-----------|
| **LLM** | Shows only LLM-classified data. |
| **NLP** | Shows only BERT+LDA data. The Topics tab shows LDA-discovered topic groups with their top words. |
| **Both (comparison)** | Enables a **Comparison** tab with: side-by-side sentiment pie charts, agreement matrix heatmap, topic distribution comparison, and sample disagreements. The Reviews tab shows both methods' outputs per review. |

**Tabs:**

- **Sentiment** — pie chart + stacked bar by version
- **Topics** — frequency bar, per-version breakdown, co-occurrence pairs, **word cloud** generated from filtered review text, and (in NLP mode) LDA topic list with top words
- **Evolution** — average score by version, weekly sentiment area chart
- **Comparison** — only visible in "Both" mode (see above)
- **Score Analysis** — star distribution, sentiment vs star rating heatmap
- **Reviews** — searchable, expandable review cards with full classification details

**Additional pages (sidebar navigation):**

- **⚙️ Pipeline Control** — run scraping, LLM classification, NLP classification and BERT embedding pipelines directly from the browser. Each pipeline runs in a background thread with live log auto-refresh (every 3 s) and a stop button. The Embeddings tab shows an interactive 2-D UMAP scatter plot where hovering reveals review text, score and version.
- **🏷️ Review Labeling** — manually label reviews to build the gold evaluation dataset.
- **📖 About & Help** — comprehensive in-app reference: quickstart, tab-by-tab explanation, filter guide, taxonomy, gold dataset format and FAQ.

**Filter behaviour notes:**

- All filters default to **all-selected** — you see complete data on first load and narrow down by removing values.
- In **NLP mode**, `mixed` is automatically excluded from the Sentiments filter (NLP can never produce it). A warning appears if `mixed` somehow remains selected.
- The **Versions** filter is applied consistently to all charts including the aggregation queries (sentiment by version, topics by version, LDA distribution, average score).
- The sidebar **Max reviews per classify run** applies only to LLM and NLP classify tabs. Scrape uses **Number of reviews to fetch** and Embeddings has its own limit.

---

## Project Structure

```
scrapping/
├── streamlit_app.py              # Main dashboard
├── pages/
│   ├── about.py                  # In-app help & FAQ page
│   ├── labeling.py               # Human labeling page
│   └── pipeline_control.py       # Browser-based pipeline launcher
├── scripts/
│   ├── scrape.py                 # Google Play scraper CLI
│   ├── classify.py               # LLM classification CLI
│   ├── classify_nlp.py           # NLP (BERT + LDA) classification CLI
│   ├── embed.py                  # BERT embedding + UMAP/KMeans CLI
│   ├── evaluate.py               # Evaluation against gold labels
│   └── migrate_db.py             # Database migration
├── src/
│   ├── config.py                 # Environment variables & paths
│   ├── db/
│   │   ├── models.py             # SQLAlchemy models (Review, Classification)
│   │   └── queries.py            # Cached SQL helpers for dashboard + evaluation
│   ├── llm/                      # LLM classification pipeline
│   │   ├── classifier.py         # Batch classifier with retries
│   │   ├── prompts.py            # System + user prompts
│   │   ├── schemas.py            # Pydantic response schema
│   │   ├── taxonomy.py           # Sentiment & topic enums
│   │   └── providers/
│   │       ├── base.py           # Abstract LLM provider
│   │       ├── openai_client.py  # OpenAI — exponential backoff on rate limits
│   │       └── ollama_client.py  # Ollama — exponential backoff on connection errors
│   ├── nlp/                      # Traditional NLP pipeline
│   │   ├── preprocessing.py      # Text cleaning, stopwords, stemming
│   │   ├── sentiment.py          # BERT multilingual sentiment
│   │   ├── topics.py             # LDA topic modelling + taxonomy mapping
│   │   ├── embeddings.py         # BERT mean-pool embeddings + UMAP + KMeans
│   │   └── pipeline.py           # Orchestrator
│   └── scraping/
│       └── play_store.py         # Google Play scraper
├── tests/                        # Automated test suite (pytest)
│   ├── conftest.py               # Shared fixtures (in-memory DB, mock providers…)
│   ├── test_preprocessing.py     # NLP text cleaning tests
│   ├── test_db_models.py         # ORM constraint tests
│   ├── test_queries.py           # SQL query helper tests
│   ├── test_llm_providers.py     # Backoff / retry tests
│   ├── test_scraping.py          # Scraper persistence tests
│   ├── test_config.py            # Environment variable parsing tests
│   ├── test_embeddings.py        # UMAP + KMeans + save/load tests
│   ├── test_classifier.py        # LLM batch classifier tests
│   └── test_lda_topics.py        # LDA model tests
├── data/
│   ├── reviews.db                # SQLite database (gitignored)
│   ├── lda_model.pkl             # Persisted LDA model (gitignored)
│   ├── embeddings.npz            # Pre-computed BERT embeddings (gitignored)
│   ├── gold.jsonl                # Human labels (gitignored)
│   └── gold_example.jsonl        # Example gold format
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Taxonomy

Both pipelines classify into the same sentiment labels. Topics use a fixed taxonomy for the LLM; the NLP pipeline discovers topics via LDA and maps them heuristically.

**Sentiments:** `positive`, `negative`, `neutral`, `mixed`

**Topics:**

| Key | Description |
|-----|-------------|
| `performance` | Speed, lag, battery drain, crashes |
| `ui_ux` | Design, layout, visual appearance, navigation |
| `bugs` | Errors, glitches, broken features |
| `features` | Missing features, feature requests, feature praise |
| `pricing` | Cost, subscriptions, in-app purchases, ads |
| `privacy_security` | Data privacy, permissions, security concerns |
| `customer_support` | Support responsiveness, help quality |
| `updates` | Effects of recent updates, version changes |
| `usability` | Ease of use, learning curve, accessibility |
| `other` | Topics that don't fit above categories |

---

## Gold Label Format

The gold dataset is a JSONL file (one JSON object per line):

```json
{"review_id": "abc123", "sentiment": "negative", "topics": ["bugs", "performance"]}
{"review_id": "def456", "sentiment": "positive", "topics": ["features", "ui_ux"]}
```

- `review_id` must match a review in the database
- `sentiment` must be one of: `positive`, `negative`, `neutral`, `mixed`
- `topics` is a list of zero or more topic keys from the taxonomy above

See `data/gold_example.jsonl` for a sample.

---

## Known Limitations

### Scraping

- **Unofficial API** -- `google-play-scraper` uses undocumented Google Play endpoints. It can break without notice if Google changes their frontend.
- **No Apple App Store** -- only Google Play is supported. Adding iOS reviews would require a different scraper or an API like [App Store Connect](https://developer.apple.com/documentation/appstoreconnectapi).
- **Rate limiting** -- scraping too aggressively may result in temporary IP blocks. The scraper uses continuation tokens and batching, but very large counts (>5000) in a single run may fail.
- **Short reviews filtered** -- very short reviews (e.g. "good", "ok") are filtered out during scraping to improve classification quality. This means the dataset skews toward longer, more descriptive reviews.

### LLM Pipeline

- **Cost** -- each review requires one API call. At scale (thousands of reviews), OpenAI costs add up. Ollama is free but slower and quality varies by model.
- **Rate limits** -- OpenAI and Ollama rate limits are handled with exponential backoff (up to 5 retries, delay doubles each attempt). Very sustained bursts may still exhaust retries.
- **Non-deterministic** -- LLM outputs are stochastic. Running the same review twice may produce different sentiment/topic labels.
- **Mixed sentiment** -- the LLM can return `mixed` sentiment, which BERT cannot (it only predicts positive/negative/neutral). This creates a slight asymmetry when comparing the two methods.

### NLP Pipeline (BERT + LDA)

- **Model size** -- the BERT model requires ~440 MB download and ~1.5 GB RAM. On machines without a GPU, inference is CPU-bound and significantly slower.
- **512-token limit** -- BERT truncates reviews longer than 512 tokens (~300-400 words). Very long reviews lose context from the end.
- **Coarse sentiment mapping** -- the BERT model predicts 1-5 stars, which are mapped to 3 sentiment categories (1-2 = negative, 3 = neutral, 4-5 = positive). Borderline reviews (e.g. 3 stars with negative text) may be misclassified.
- **No "mixed" sentiment** -- unlike the LLM, BERT cannot output `mixed`. Reviews with genuinely mixed feelings get forced into one category.
- **LDA requires corpus** -- the LDA model needs enough data to find meaningful topics. With fewer than ~100 reviews, topics will be noisy. Retrain after adding significant new data.
- **LDA-to-taxonomy mapping is heuristic** -- the keyword-overlap mapping from LDA topics to the fixed taxonomy is best-effort. Some LDA topics may map to `other` even when they have a clear theme.
- **Language-dependent** -- stopwords and stemming default to Portuguese. If your reviews are in a different language, pass `--language english` (or another NLTK-supported language) to `classify_nlp`.

### Dashboard

- **100-review display limit** -- the Reviews tab only renders the first 100 matching reviews to keep the page responsive. Use search/filters to narrow down.
- **No real-time updates** -- the dashboard caches queries for 60 seconds. Click "Clear cache & reload" after running a classifier to see new results immediately.
- **SQLite concurrency** -- SQLite does not support concurrent writers. Do not run a classifier script while the dashboard is writing gold labels.
- **Pipeline Control background threads** -- pipelines launched from the browser run in Python threads inside the Streamlit process. They are not supervised processes; restarting Streamlit will abort any running pipeline. Use the **⏹ Stop** button to terminate a running pipeline gracefully.
- **Pipeline Control log refresh** -- logs auto-refresh every 3 seconds while a pipeline is running. No manual click is needed unless the page becomes stale.

### Embeddings

- **BERT base model for embeddings** -- `embeddings.py` loads a separate base BERT model (no classification head) for mean-pool embeddings; this is a second model download (~440 MB) on top of the sentiment model.
- **UMAP is non-deterministic without a seed** -- results are fixed with `random_state=42` by default, so re-running with the same data produces identical coordinates.
- **KMeans cluster count is manual** -- choose `--n-clusters` based on domain knowledge or elbow-method inspection. There is no automatic optimal-k selection.

### Evaluation

- **Gold set size** -- metrics are only as reliable as the gold set. With fewer than ~50 labeled reviews, precision/recall numbers have high variance.
- **Topic evaluation only for LLM** -- because LDA discovers topics unsupervised, there is no 1:1 mapping to the fixed taxonomy, so per-topic F1 is only computed for the LLM pipeline.
- **Kaleido for PNG export** -- saving plots as PNG requires the `kaleido` package, which may have installation issues on some platforms (especially Apple Silicon). HTML export always works.
