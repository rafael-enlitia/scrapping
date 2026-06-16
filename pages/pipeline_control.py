"""Pipeline Control — trigger scrape, LLM/NLP classification and embeddings from the UI."""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
_PYTHON = sys.executable
_ENV = {**os.environ, "PYTHONPATH": str(_ROOT), "PYTHONUNBUFFERED": "1"}

# How often (seconds) the page polls for new log output
_POLL_INTERVAL = 2


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _run_async(cmd: list[str], log_key: str) -> None:
    """Spawn *cmd* in a background thread; communicate via a queue to the main thread."""
    q: queue.Queue = queue.Queue()
    st.session_state[f"{log_key}_queue"] = q

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
        q.put(("pid", proc.pid))
        for line in proc.stdout:
            q.put(("log", line))
        proc.wait()
        q.put(("done", proc.returncode))

    threading.Thread(target=_worker, daemon=True).start()


def _drain_queue(log_key: str) -> None:
    """Drain any pending messages from the worker queue into session_state (main thread only)."""
    q = st.session_state.get(f"{log_key}_queue")
    if q is None:
        return
    while True:
        try:
            msg_type, msg_data = q.get_nowait()
            if msg_type == "pid":
                st.session_state[f"{log_key}_pid"] = msg_data
            elif msg_type == "log":
                st.session_state[log_key] += msg_data
            elif msg_type == "done":
                st.session_state[f"{log_key}_running"] = False
                st.session_state[f"{log_key}_returncode"] = msg_data
                st.session_state.pop(f"{log_key}_queue", None)
        except queue.Empty:
            break


def _stop_pipeline(log_key: str) -> None:
    """Send SIGTERM to the running pipeline process, if known."""
    pid = st.session_state.get(f"{log_key}_pid")
    if pid:
        try:
            os.kill(pid, 15)  # SIGTERM
            st.session_state[log_key] += "\n[Stopped by user]"
            st.session_state[f"{log_key}_running"] = False
            st.session_state.pop(f"{log_key}_queue", None)
        except ProcessLookupError:
            pass


def _build_cmd(script: str, args: list[str]) -> list[str]:
    module = script.removesuffix(".py")
    return [_PYTHON, "-u", "-m", f"scripts.{module}", *args]


def _init_log_state(log_key: str) -> None:
    for suffix, default in [("", ""), ("_running", False), ("_returncode", None), ("_pid", None)]:
        key = f"{log_key}{suffix}"
        if key not in st.session_state:
            st.session_state[key] = default


def _parse_progress(log: str) -> tuple[int, int] | None:
    """Return latest (current, target) from PROGRESS or [n/m] lines in the log."""
    current, target = 0, 0
    patterns = [
        r"PROGRESS (\d+)/(\d+)",
        r"\[(\d+)/(\d+)\] Classified",
        r"\[(\d+)/(\d+)\] Failed",
    ]
    for line in log.splitlines():
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                current, target = int(match.group(1)), int(match.group(2))
    return (current, target) if target else None


def _parse_found_total(log: str) -> int | None:
    """Infer batch size from 'Found N …' or 'Computing embeddings for N …' log lines."""
    patterns = [
        r"Found (\d+) unclassified reviews",
        r"Found (\d+) reviews to classify",
        r"Computing embeddings for (\d+) reviews",
    ]
    for line in log.splitlines():
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return int(match.group(1))
    return None


def _parse_latest_status(log: str) -> str | None:
    """Return the most recent INFO log message for the progress label."""
    for line in reversed(log.splitlines()):
        if "INFO:" in line:
            return line.split("INFO:", 1)[-1].strip()
    return None


def _parse_scrape_summary(log: str) -> dict[str, int] | None:
    """Parse SCRAPE_SUMMARY line emitted by scripts/scrape.py."""
    for line in log.splitlines():
        if not line.startswith("SCRAPE_SUMMARY"):
            continue
        stats: dict[str, int] = {}
        for part in line.split()[1:]:
            key, _, val = part.partition("=")
            if val.isdigit():
                stats[key] = int(val)
        return stats if stats else None
    return None


def _format_scrape_summary(stats: dict[str, int]) -> str:
    saved = stats.get("saved", 0)
    fetched = stats.get("fetched", 0)
    parts = [f"**{saved}** new reviews saved"]
    if stats.get("duplicates"):
        parts.append(f"**{stats['duplicates']}** already in database (skipped)")
    if stats.get("skipped_short"):
        parts.append(f"**{stats['skipped_short']}** too short (skipped)")
    if stats.get("no_id"):
        parts.append(f"**{stats['no_id']}** without review ID (skipped)")
    detail = " · ".join(parts)
    return f"{detail} — {fetched} eligible from store (requested up to {stats.get('store', fetched)})"


def _parse_kv_summary(log: str, prefix: str) -> dict[str, str] | None:
    """Parse lines like 'CLASSIFY_SUMMARY method=llm classified=42' (last match wins)."""
    last: dict[str, str] | None = None
    for line in log.splitlines():
        if not line.startswith(prefix):
            continue
        stats: dict[str, str] = {}
        for part in line.split()[1:]:
            key, _, val = part.partition("=")
            if key:
                stats[key] = val
        if stats:
            last = stats
    return last


def _format_classify_summary(stats: dict[str, str]) -> str:
    method = stats.get("method", "unknown").upper()
    count = stats.get("classified", "0")
    return f"**{count}** reviews classified with **{method}** pipeline"


def _format_embed_summary(stats: dict[str, str]) -> str:
    embedded = stats.get("embedded", stats.get("target", "0"))
    clusters = stats.get("clusters", "?")
    return f"**{embedded}** reviews embedded into **{clusters}** clusters"


def _pipeline_summary_parser(log: str, kind: str) -> str | None:
    """Parse completion summary and clear dashboard cache on success."""
    if kind == "scrape":
        stats = _parse_scrape_summary(log)
        if stats:
            st.cache_data.clear()
            return _format_scrape_summary(stats)
    elif kind == "classify":
        stats = _parse_kv_summary(log, "CLASSIFY_SUMMARY")
        if stats and "classified" in stats:
            st.cache_data.clear()
            return _format_classify_summary(stats)
    elif kind == "embed":
        stats = _parse_kv_summary(log, "EMBED_SUMMARY")
        if stats and "embedded" in stats:
            st.cache_data.clear()
            return _format_embed_summary(stats)
    return None


def _render_log(
    log_key: str,
    *,
    progress_target: int | None = None,
    progress_label: str = "Working",
    summary_parser: Callable[[str], str | None] | None = None,
) -> None:
    """Drain the queue, then render running/stopped state + log output."""
    _drain_queue(log_key)

    is_running = st.session_state[f"{log_key}_running"]
    log = st.session_state[log_key]
    rc = st.session_state.get(f"{log_key}_returncode")

    if is_running:
        progress = _parse_progress(log)
        current, target = (progress if progress else (0, 0))
        if not target:
            target = progress_target or _parse_found_total(log) or 0

        status = _parse_latest_status(log)
        if target > 0:
            pct = min(current / target, 1.0)
            label = status or f"{progress_label}… {current}/{target}"
            st.progress(pct, text=label)
        elif status:
            st.progress(0, text=status)
        else:
            st.progress(0, text=f"{progress_label}… starting")

        col_stop, col_refresh = st.columns([1, 1])
        with col_stop:
            if st.button("⏹ Stop", key=f"stop_{log_key}"):
                _stop_pipeline(log_key)
                st.rerun()
        with col_refresh:
            if st.button("🔄 Refresh", key=f"refresh_{log_key}"):
                st.rerun()
    elif log:
        summary_msg = summary_parser(log) if summary_parser else None
        if rc is not None and rc != 0:
            st.error(f"Process exited with code {rc}")
        elif rc == 0:
            st.success(summary_msg or "Pipeline completed successfully.")

    if log:
        st.subheader("Live log" if is_running else "Log")
        st.code(log, language="text")
    elif is_running:
        st.caption("Waiting for output…")

    if is_running:
        time.sleep(_POLL_INTERVAL)
        st.rerun()


# --------------------------------------------------------------------------
# Page layout
# --------------------------------------------------------------------------

st.set_page_config(page_title="Pipeline Control", page_icon="⚙️", layout="wide")
st.title("⚙️ Pipeline Control")
st.caption("Launch scraping, classification and embedding pipelines without leaving the browser.")

# ── Shared sidebar inputs ─────────────────────────────────────────────────
with st.sidebar:
    st.header("App")

    from src.db.queries import get_app_ids  # noqa: PLC0415
    _known_app_ids = get_app_ids()
    _shared = st.session_state.get("shared_app_id", "")
    if _known_app_ids:
        _default_idx = _known_app_ids.index(_shared) if _shared in _known_app_ids else 0
        app_id_input = st.selectbox("Package name (App ID)", _known_app_ids, index=_default_idx)
        _custom = st.text_input("Or enter a new App ID", placeholder="com.example.app")
        if _custom:
            app_id_input = _custom
    else:
        app_id_input = st.text_input("Package name (App ID)", value=_shared, placeholder="com.example.app")

    if not app_id_input:
        st.warning("Enter an App ID to enable pipeline actions.")
    st.caption("Used by every tab when you run a pipeline.")

    st.divider()
    st.subheader("Classify batch size")
    st.caption(
        "Only **LLM Classify** and **NLP Classify** read this value. "
        "**0** = process every pending review."
    )
    limit_input = st.number_input(
        "Max reviews per classify run",
        min_value=0,
        value=0,
        step=50,
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

    _scrape_log_key = "scrape_log"
    _init_log_state(_scrape_log_key)

    col1, col2 = st.columns(2)
    with col1:
        scrape_count = st.number_input("Number of reviews to fetch", min_value=1, max_value=5000, value=500, step=100)
        scrape_lang = st.text_input("Language code", value="pt")
    with col2:
        scrape_country = st.text_input("Country code", value="pt")
        scrape_sort = st.selectbox("Sort order", ["newest", "most_relevant"])

    if not app_id_input:
        st.info("Enter an App ID in the sidebar to enable scraping.")
    elif not st.session_state[f"{_scrape_log_key}_running"]:
        if st.button("▶ Start Scraping", type="primary", disabled=not app_id_input):
            args = [
                "--app-id", app_id_input,
                "--count", str(scrape_count),
                "--lang", scrape_lang,
                "--country", scrape_country,
                "--sort", scrape_sort,
            ]
            st.session_state["scrape_target"] = int(scrape_count)
            _run_async(_build_cmd("scrape.py", args), _scrape_log_key)
            st.rerun()

    def _scrape_summary(log: str) -> str | None:
        return _pipeline_summary_parser(log, "scrape")

    _render_log(
        _scrape_log_key,
        progress_target=int(st.session_state.get("scrape_target", scrape_count)),
        progress_label="Fetching reviews",
        summary_parser=_scrape_summary,
    )


# ══════════════════════════════════════════════════════════════════════════
# LLM Classification tab
# ══════════════════════════════════════════════════════════════════════════
with tab_llm:
    st.subheader("LLM Classification")

    _llm_log_key = "llm_log"
    _init_log_state(_llm_log_key)

    col1, col2 = st.columns(2)
    with col1:
        llm_provider = st.selectbox("Provider", ["(env default)", "openai", "ollama", "iaedu"])
        llm_retry = st.checkbox("Retry failed reviews only (--retry-failed)")
    with col2:
        st.markdown("**Batch size:** sidebar → *Max reviews per classify run*")
        if limit_val:
            st.caption(f"This run will classify at most **{limit_val}** reviews.")
        else:
            st.caption("**0** in sidebar → classify **all** unclassified reviews.")

    if not st.session_state[f"{_llm_log_key}_running"]:
        if st.button("▶ Start LLM Classification", type="primary", disabled=not app_id_input):
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

    _render_log(
        _llm_log_key,
        progress_target=limit_val,
        progress_label="LLM classification",
        summary_parser=lambda log: _pipeline_summary_parser(log, "classify"),
    )


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
        if limit_val:
            st.caption(f"This run will classify at most **{limit_val}** reviews.")

    if not st.session_state[f"{_nlp_log_key}_running"]:
        if st.button("▶ Start NLP Classification", type="primary", disabled=not app_id_input):
            args = ["--num-topics", str(nlp_topics), "--language", nlp_lang]
            if app_id_input:
                args += ["--app-id", app_id_input]
            if limit_val:
                args += ["--limit", str(limit_val)]
            if nlp_retrain:
                args.append("--retrain-lda")
            _run_async(_build_cmd("classify_nlp.py", args), _nlp_log_key)
            st.rerun()

    _render_log(
        _nlp_log_key,
        progress_target=limit_val,
        progress_label="NLP classification",
        summary_parser=lambda log: _pipeline_summary_parser(log, "classify"),
    )


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

    _emb_log_key = "emb_log"
    _init_log_state(_emb_log_key)

    col1, col2 = st.columns(2)
    with col1:
        embed_clusters = st.number_input("Number of clusters (KMeans)", min_value=2, max_value=30, value=8)
    with col2:
        embed_limit = st.number_input(
            "Max reviews to embed (0 = all)",
            min_value=0,
            value=0,
            step=100,
        )

    if not st.session_state[f"{_emb_log_key}_running"]:
        if st.button("▶ Compute Embeddings", type="primary", disabled=not app_id_input):
            args = ["--n-clusters", str(embed_clusters)]
            if app_id_input:
                args += ["--app-id", app_id_input]
            if embed_limit:
                args += ["--limit", str(embed_limit)]
            st.session_state["embed_target"] = int(embed_limit) if embed_limit else None
            _run_async(_build_cmd("embed.py", args), _emb_log_key)
            st.rerun()

    _embed_target = st.session_state.get("embed_target")
    if _embed_target is None:
        _embed_target = int(embed_limit) if embed_limit else None

    _render_log(
        _emb_log_key,
        progress_target=_embed_target,
        progress_label="Embeddings pipeline",
        summary_parser=lambda log: _pipeline_summary_parser(log, "embed"),
    )

    # ── Cluster visualisation ──────────────────────────────────────────────
    from src.db.queries import get_embedding_clusters  # noqa: PLC0415
    from src.db.models import Review, get_session, init_db  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415
    import plotly.express as px  # noqa: PLC0415

    _embed_app = app_id_input or None
    cluster_df = get_embedding_clusters(_embed_app)

    if cluster_df is not None and not cluster_df.empty:
        st.divider()
        st.subheader("Cluster visualisation (UMAP 2-D)")
        if _embed_app:
            st.caption(f"Showing embeddings for: **{_embed_app}**")

        # Enrich with review content for hover labels
        init_db()
        _session = get_session()
        try:
            _rows = _session.query(Review.review_id, Review.content, Review.score, Review.app_version).all()
        finally:
            _session.close()
        content_map = {r.review_id: r for r in _rows}

        plot_df = cluster_df.copy()
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
        st.info(
            "No embeddings computed yet for "
            + (f"**{_embed_app}**" if _embed_app else "this app")
            + ". Run the pipeline above to generate them."
        )
