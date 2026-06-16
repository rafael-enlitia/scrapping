"""Sentence embeddings and UMAP+KMeans clustering for review exploration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from src.config import DATA_DIR
from src.nlp.model_manager import model_manager

logger = logging.getLogger(__name__)


def _embeddings_path(app_id: str | None = None) -> Path:
    """Return the per-app or global embeddings file path."""
    if app_id:
        safe = app_id.replace("/", "_").replace("\\", "_")
        return DATA_DIR / f"embeddings_{safe}.npz"
    return DATA_DIR / "embeddings.npz"


def _get_umap(**kwargs):
    try:
        import umap  # noqa: PLC0415
        return umap.UMAP(**kwargs)
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for clustering. Install it with: pip install umap-learn"
        ) from exc


class EmbeddingResult(NamedTuple):
    review_ids: list[str]
    embeddings: np.ndarray      # shape (n, hidden_size)
    umap_2d: np.ndarray         # shape (n, 2)
    cluster_labels: np.ndarray  # shape (n,)


def compute_embeddings(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Compute mean-pooled sentence embeddings using the shared base BERT model."""
    if not texts:
        raise ValueError("texts must not be empty")

    import torch  # noqa: PLC0415

    tokenizer = model_manager.get_tokenizer()
    model, device = model_manager.get_embedding_model()

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
        hidden = outputs.last_hidden_state  # (B, seq, hidden)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_vecs.append(pooled.cpu().numpy())
        done = min(i + batch_size, len(texts))
        logger.info("PROGRESS %d/%d", done, len(texts))

    return np.vstack(all_vecs)


def cluster_and_reduce(
    embeddings: np.ndarray,
    n_clusters: int = 8,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (umap_2d, cluster_labels) for the given embeddings."""
    n_samples = len(embeddings)
    if n_samples == 0:
        raise ValueError("embeddings must not be empty")

    if n_clusters > n_samples:
        logger.warning(
            "n_clusters=%d > n_samples=%d — reducing n_clusters to %d",
            n_clusters, n_samples, n_samples,
        )
        n_clusters = n_samples

    normed = normalize(embeddings)

    reducer = _get_umap(
        n_components=2,
        n_neighbors=min(umap_n_neighbors, n_samples - 1),
        min_dist=umap_min_dist,
        random_state=random_state,
        verbose=False,
    )
    umap_2d = reducer.fit_transform(normed)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(normed)

    return umap_2d.astype(np.float32), labels.astype(np.int32)


def save_embeddings(
    review_ids: list[str],
    embeddings: np.ndarray,
    umap_2d: np.ndarray,
    labels: np.ndarray,
    app_id: str | None = None,
) -> None:
    from src.config import ensure_data_dir  # noqa: PLC0415
    ensure_data_dir()
    path = _embeddings_path(app_id)
    np.savez_compressed(
        path,
        review_ids=np.array(review_ids),
        embeddings=embeddings,
        umap_2d=umap_2d,
        cluster_labels=labels,
    )
    logger.info("Embeddings saved to %s", path)


def load_embeddings(app_id: str | None = None) -> EmbeddingResult | None:
    path = _embeddings_path(app_id)
    if not path.exists():
        return None
    data = np.load(path)
    return EmbeddingResult(
        review_ids=[str(r) for r in data["review_ids"].tolist()],
        embeddings=data["embeddings"],
        umap_2d=data["umap_2d"],
        cluster_labels=data["cluster_labels"],
    )


def run_embedding_pipeline(
    reviews: list[tuple[str, str]],  # list of (review_id, content)
    n_clusters: int = 8,
    app_id: str | None = None,
) -> EmbeddingResult:
    """Full pipeline: embed → reduce → cluster → save → return result."""
    if not reviews:
        raise ValueError("reviews must not be empty")

    review_ids = [r[0] for r in reviews]
    texts = [r[1] for r in reviews]
    total = len(texts)

    logger.info("Computing embeddings for %d reviews…", total)
    logger.info("PROGRESS 0/%d", total)
    embeddings = compute_embeddings(texts)

    logger.info("Running UMAP + KMeans (k=%d)…", n_clusters)
    logger.info("PROGRESS %d/%d", max(1, total * 85 // 100), total)
    umap_2d, labels = cluster_and_reduce(embeddings, n_clusters=n_clusters)
    logger.info("PROGRESS %d/%d", max(1, total * 95 // 100), total)

    logger.info("Saving embeddings…")
    save_embeddings(review_ids, embeddings, umap_2d, labels, app_id=app_id)
    logger.info("PROGRESS %d/%d", total, total)
    return EmbeddingResult(review_ids=review_ids, embeddings=embeddings, umap_2d=umap_2d, cluster_labels=labels)
