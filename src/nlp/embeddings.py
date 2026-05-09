"""Sentence embeddings and UMAP+KMeans clustering for review exploration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

_EMBEDDINGS_PATH = DATA_DIR / "embeddings.npz"
_CLUSTER_PATH = DATA_DIR / "cluster_model.pkl"

# --------------------------------------------------------------------------
# Lazy UMAP import — umap-learn is optional at module load time
# --------------------------------------------------------------------------
_umap_reducer = None


def _get_umap(**kwargs):
    global _umap_reducer
    try:
        import umap  # noqa: PLC0415
        return umap.UMAP(**kwargs)
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for clustering. Install it with: pip install umap-learn"
        ) from exc


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------

class EmbeddingResult(NamedTuple):
    review_ids: list[str]
    embeddings: np.ndarray      # shape (n, hidden_size)
    umap_2d: np.ndarray         # shape (n, 2)
    cluster_labels: np.ndarray  # shape (n,)


# --------------------------------------------------------------------------
# Core functions
# --------------------------------------------------------------------------

_emb_tokenizer = None
_emb_model = None
_emb_device = None


def _load_embedding_model():
    """Lazy-load a base BERT model (without classification head) for pooling."""
    global _emb_tokenizer, _emb_model, _emb_device
    if _emb_tokenizer is None:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
        from src.config import BERT_MODEL  # noqa: PLC0415

        logger.info("Loading base BERT for embeddings: %s", BERT_MODEL)
        _emb_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        _emb_model = AutoModel.from_pretrained(BERT_MODEL)
        _emb_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _emb_model.to(_emb_device)
        _emb_model.eval()
    return _emb_tokenizer, _emb_model, _emb_device


def compute_embeddings(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Compute mean-pooled sentence embeddings using a base BERT model."""
    import torch  # noqa: PLC0415

    tokenizer, model, device = _load_embedding_model()
    all_vecs: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean-pool over token dimension, masked by attention
        hidden = outputs.last_hidden_state  # (B, seq, hidden)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_vecs.append(pooled.cpu().numpy())
        logger.info("Embeddings: batch %d–%d / %d", i + 1, min(i + batch_size, len(texts)), len(texts))

    return np.vstack(all_vecs)


def cluster_and_reduce(
    embeddings: np.ndarray,
    n_clusters: int = 8,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (umap_2d, cluster_labels) for the given embeddings."""
    normed = normalize(embeddings)

    reducer = _get_umap(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=random_state,
        verbose=False,
    )
    umap_2d = reducer.fit_transform(normed)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(normed)

    return umap_2d.astype(np.float32), labels.astype(np.int32)


def save_embeddings(review_ids: list[str], embeddings: np.ndarray, umap_2d: np.ndarray, labels: np.ndarray) -> None:
    from src.config import ensure_data_dir  # noqa: PLC0415
    ensure_data_dir()
    np.savez_compressed(
        _EMBEDDINGS_PATH,
        review_ids=np.array(review_ids),
        embeddings=embeddings,
        umap_2d=umap_2d,
        cluster_labels=labels,
    )
    logger.info("Embeddings saved to %s", _EMBEDDINGS_PATH)


def load_embeddings() -> EmbeddingResult | None:
    if not _EMBEDDINGS_PATH.exists():
        return None
    data = np.load(_EMBEDDINGS_PATH, allow_pickle=True)
    return EmbeddingResult(
        review_ids=data["review_ids"].tolist(),
        embeddings=data["embeddings"],
        umap_2d=data["umap_2d"],
        cluster_labels=data["cluster_labels"],
    )


def run_embedding_pipeline(
    reviews: list[tuple[str, str]],  # list of (review_id, content)
    n_clusters: int = 8,
) -> EmbeddingResult:
    """Full pipeline: embed → reduce → cluster → save → return result."""
    review_ids = [r[0] for r in reviews]
    texts = [r[1] for r in reviews]

    logger.info("Computing embeddings for %d reviews…", len(texts))
    embeddings = compute_embeddings(texts)

    logger.info("Running UMAP + KMeans (k=%d)…", n_clusters)
    umap_2d, labels = cluster_and_reduce(embeddings, n_clusters=n_clusters)

    save_embeddings(review_ids, embeddings, umap_2d, labels)
    return EmbeddingResult(review_ids=review_ids, embeddings=embeddings, umap_2d=umap_2d, cluster_labels=labels)
