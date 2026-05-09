"""Pipeline Control — trigger scrape, LLM/NLP classification and embeddings from the UI."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
_PYTHON = sys.executable
_ENV = {**os.environ, "PYTHONPATH": str(_ROOT)}

# How often (seconds) the log auto-refreshes while a pipeline is running
_AUTO_REFRESH_INTERVAL = 3


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _run(cmd: list[str]) -> tuple[int, str]:
    """Run a subprocess synchronously and return (exit_code, combined_output)."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
        env=_ENV,
    )
    return result.returncode, ((result.stdout or "") + (result.stderr or "")).strip()


def _run_async(cmd: list[str], log_key: str) -> None:
    """Spawn *cmd* in a background thread; stream output to session_state."""
    st.session_state[log_key] = ""
    st.session_state[f"{log_key}_running"] = True
    st.session_state[f"{log_key}_returncode"] = None
    st.session_state[f"{log_key}_pid"] = None

    def _worker():
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(_ROOT),
            env=_ENV,
        )
        st.session_state[f"{log_key}_pid"] = proc.pid
        for line in proc.stdout:
            st.session_state[log_key] += line
        proc.wait()
        st.session_state[f"{log_key}_running"] = False
        st.session_state[f"{log_key}_returncode"] = proc.returncode
        st.session_state[f"{log_key}_pid"] = None

    threading.Thread(target=_worker, daemon=True).start()


def _stop_pipeline(log_key: str) -> None:
    """Send SIGTERM to the running pipeline process, if known."""
    pid = st.session_state.get(f"{log_key}_pid")
    if pid:
        try:
            os.kill(pid, 15)  # SIGTERM
            st.session_state[f"{log_key}_running"] = False
            st.session_state[log_key] += "\n[Stopped by user]"
        except ProcessLookupError:
            pass  # already finished


def _build_cmd(script: str, args: list[str]) -> list[str]:
    return [_PYTHON, str(_ROOT / "scripts" / script)] + args


def _init_log_state(log_key: str) -> None:
    """Ensure all session_state keys for a pipeline exist."""
    for suffix, default in [("", ""), ("_running", False), ("_returncode", None), ("_pid", None)]:
        key = f"{log_key}{suffix}"
        if key not in st.session_state:
            st.session_state[key] = default


def _render_log(log_key: str) -> None:
    """Render the running/stopped state + log output for a pipeline."""
    is_running = st.session_state[f"{log_key}_running"]
    log = st.session_state[log_key]
    rc = st.session_state.get(f"{log_key}_returncode")

    if is_running:
        col_status, col_stop, col_refresh = st.columns([3, 1, 1])
        with col_status:
            st.warning("Pipeline is running in the background…")
        with col_stop:
            if st.button("⏹ Stop", key=f"stop_{log_key}"):
                _stop_pipeline(log_key)
                st.rerun()
        with col_refresh:
            if st.button("🔄 Refresh", key=f"refresh_{log_key}"):
                st.rerun()
        # Auto-refresh while running
        time.sleep(_AUTO_REFRESH_INTERVAL)
        st.rerun()
    elif log:
        if rc is not None and rc != 0:
            st.error(f"Process exited with code {rc}")
        elif rc == 0:
            st.success("Pipeline completed successfully.")

    if log:
        st.code(log, language="text")


# --------------------------------------------------------------------------
# Page layout
# --------------------------------------------------------------------------

st.set_page_config(page_title="Pipeline Control", page_icon="⚙️", layout="wide")
st.title("⚙️ Pipeline Control")
st.caption("Launch scraping, classification and embedding pipelines without leaving the browser.")

# ── Shared sidebar inputs ─────────────────────────────────────────────────
with st.sidebar:
    st.header("App")
    app_id_input = st.text_input("Package name (App ID)", placeholder="com.example.app")
    st.caption("Used by every tab when you run a pipeline.")

    st.divider()
    st.subheader("Classify batch size")
    st.caption(
        "Only **LLM Classify** and **NLP Classify** read this value. "
        "**Scrape** ignores it — set how many reviews to download in that tab. "
        "**Embeddings** has its own limit on that tab. "
        "**0** = process every pending review in one run."
    )
    limit_input = st.number_input(
        "Max reviews per classify run",
        min_value=0,
        value=0,
        step=50,
        help="Caps LLM/NLP classification only. Not used for scraping or embedding.",
    )
    limit_val = int(limit_input) if limit_input else None

# ── Tab layout ────────────────────────────────────────────────────────────
tab_scrape, tab_llm, tab_nlp, tab_embed = st.tabs(
    ["🕷️ Scrape", "🤖 LLM Classify", "🧬 NLP Classify", "📊 Embeddings"]
)


# ══════════════════════════════════════════════════════════════════════════
# Scrape tab
# ══════════════════════════════════════════════════════════════════════════
with tab_scrape:
    st.subheader("Scrape Google Play Reviews")
    st.info(
        "How many reviews to **download** from the store is set below (**Number of reviews to fetch**). "
        "The sidebar **Max reviews per classify run** does **not** apply here."
    )

    col1, col2 = st.columns(2)
    with col1:
        scrape_count = st.number_input("Number of reviews to fetch", min_value=1, max_value=5000, value=500, step=100)
        scrape_lang = st.text_input("Language code", value="pt")
    with col2:
        scrape_country = st.text_input("Country code", value="pt")
        scrape_sort = st.selectbox("Sort order", ["newest", "most_relevant"])

    if not app_id_input:
        st.info("Enter an App ID in the sidebar to enable scraping.")
    elif st.button("▶ Start Scraping", type="primary"):
        args = [
            "--app-id", app_id_input,
            "--count", str(scrape_count),
            "--lang", scrape_lang,
            "--country", scrape_country,
            "--sort", scrape_sort,
        ]
        with st.spinner("Scraping — this may take a moment…"):
            code, out = _run(_build_cmd("scrape.py", args))
        if code == 0:
            st.success(out or "Scraping completed.")
        else:
            st.error(f"Scraping failed (exit {code})")
            st.code(out, language="text")


# ══════════════════════════════════════════════════════════════════════════
# LLM Classification tab
# ══════════════════════════════════════════════════════════════════════════
with tab_llm:
    st.subheader("LLM Classification")

    _llm_log_key = "llm_log"
    _init_log_state(_llm_log_key)

    col1, col2 = st.columns(2)
    with col1:
        llm_provider = st.selectbox("Provider", ["(env default)", "openai", "ollama"])
        llm_retry = st.checkbox("Retry failed reviews only (--retry-failed)")
    with col2:
        st.markdown("**Batch size:** sidebar → *Max reviews per classify run*")
        if limit_val:
            st.caption(f"This run will classify at most **{limit_val}** reviews.")
        else:
            st.caption("**0** in sidebar → classify **all** unclassified reviews.")

    if not st.session_state[f"{_llm_log_key}_running"]:
        if st.button("▶ Start LLM Classification", type="primary"):
            args = []
            if app_id_input:
                args += ["--app-id", app_id_input]
            if limit_val:
                args += ["--limit", str(limit_val)]
            if llm_provider != "(env default)":
                args += ["--provider", llm_provider]
            if llm_retry:
                args.append("--retry-failed")
            _run_async(_build_cmd("classify.py", args), _llm_log_key)
            st.rerun()

    _render_log(_llm_log_key)


# ══════════════════════════════════════════════════════════════════════════
# NLP Classification tab
# ══════════════════════════════════════════════════════════════════════════
with tab_nlp:
    st.subheader("NLP Classification (BERT + LDA)")

    _nlp_log_key = "nlp_log"
    _init_log_state(_nlp_log_key)

    col1, col2 = st.columns(2)
    with col1:
        nlp_topics = st.number_input("Number of LDA topics", min_value=2, max_value=30, value=8)
        nlp_lang = st.text_input("Language (stopwords)", value="portuguese")
    with col2:
        nlp_retrain = st.checkbox("Force retrain LDA model")
        st.markdown("**Batch size:** sidebar → *Max reviews per classify run*")
        if limit_val:
            st.caption(f"This run will classify at most **{limit_val}** reviews.")
        else:
            st.caption("**0** in sidebar → classify **all** unclassified reviews.")

    if not st.session_state[f"{_nlp_log_key}_running"]:
        if st.button("▶ Start NLP Classification", type="primary"):
            args = ["--num-topics", str(nlp_topics), "--language", nlp_lang]
            if app_id_input:
                args += ["--app-id", app_id_input]
            if limit_val:
                args += ["--limit", str(limit_val)]
            if nlp_retrain:
                args.append("--retrain-lda")
            _run_async(_build_cmd("classify_nlp.py", args), _nlp_log_key)
            st.rerun()

    _render_log(_nlp_log_key)


# ══════════════════════════════════════════════════════════════════════════
# Embeddings tab
# ══════════════════════════════════════════════════════════════════════════
with tab_embed:
    st.subheader("Review Embeddings + Clustering")
    st.caption(
        "Computes BERT sentence embeddings, reduces them with UMAP to 2-D, "
        "then groups reviews into clusters with KMeans. "
        "Hover over any point to read the review text."
    )
    st.info(
        "The sidebar **Max reviews per classify run** is for LLM/NLP only. "
        "This tab uses **Max reviews to embed** below."
    )

    _emb_log_key = "emb_log"
    _init_log_state(_emb_log_key)

    col1, col2 = st.columns(2)
    with col1:
        embed_clusters = st.number_input("Number of clusters (KMeans)", min_value=2, max_value=30, value=8)
    with col2:
        embed_limit = st.number_input(
            "Max reviews to embed (0 = all in DB / app filter)",
            min_value=0,
            value=0,
            step=100,
            help="Separate from the sidebar classify batch size.",
        )

    if not st.session_state[f"{_emb_log_key}_running"]:
        if st.button("▶ Compute Embeddings", type="primary"):
            args = ["--n-clusters", str(embed_clusters)]
            if app_id_input:
                args += ["--app-id", app_id_input]
            if embed_limit:
                args += ["--limit", str(embed_limit)]
            _run_async(_build_cmd("embed.py", args), _emb_log_key)
            st.rerun()

    _render_log(_emb_log_key)

    # ── Cluster visualisation ──────────────────────────────────────────────
    from src.nlp.embeddings import load_embeddings
    from src.db.models import Review, get_session, init_db

    emb = load_embeddings()
    if emb is not None:
        import pandas as pd
        import plotly.express as px

        st.divider()
        st.subheader("Cluster visualisation (UMAP 2-D)")

        # Load review content to enrich hover labels
        init_db()
        session = get_session()
        rows = session.query(Review.review_id, Review.content, Review.score, Review.app_version).all()
        session.close()
        content_map = {r.review_id: r for r in rows}

        plot_df = pd.DataFrame({
            "x": emb.umap_2d[:, 0],
            "y": emb.umap_2d[:, 1],
            "cluster": emb.cluster_labels.astype(str),
            "review_id": emb.review_ids,
        })

        # Enrich with content for hover (truncated to 120 chars)
        plot_df["preview"] = plot_df["review_id"].map(
            lambda rid: (content_map[rid].content[:120] + "…") if rid in content_map else ""
        )
        plot_df["score"] = plot_df["review_id"].map(
            lambda rid: content_map[rid].score if rid in content_map else None
        )
        plot_df["version"] = plot_df["review_id"].map(
            lambda rid: content_map[rid].app_version if rid in content_map else None
        )

        fig = px.scatter(
            plot_df,
            x="x", y="y",
            color="cluster",
            hover_data={
                "x": False,
                "y": False,
                "review_id": True,
                "score": True,
                "version": True,
                "preview": True,
            },
            labels={"x": "UMAP-1", "y": "UMAP-2", "preview": "Review"},
            title=f"{len(plot_df)} reviews · {plot_df['cluster'].nunique()} clusters",
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(legend_title_text="Cluster")
        st.plotly_chart(fig, width="stretch")

        counts = plot_df["cluster"].value_counts().reset_index()
        counts.columns = ["Cluster", "Reviews"]
        st.dataframe(counts.sort_values("Cluster"), width="stretch", hide_index=True)
    else:
        st.info("No embeddings computed yet. Run the pipeline above to generate them.")
