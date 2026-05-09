"""LDA topic modelling with sklearn."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.config import LDA_MODEL_PATH, LDA_NUM_TOPICS
from src.llm.taxonomy import Topic

logger = logging.getLogger(__name__)

# Best-effort keyword map from the fixed taxonomy to help label LDA topics
_TAXONOMY_KEYWORDS: dict[str, list[str]] = {
    Topic.PERFORMANCE.value: ["slow", "fast", "lag", "crash", "battery", "speed", "lent", "rapid", "trava"],
    Topic.UI_UX.value: ["design", "layout", "interface", "visual", "icon", "theme", "bonit", "fei"],
    Topic.BUGS.value: ["bug", "error", "glitch", "broken", "fix", "fail", "erro", "falh"],
    Topic.FEATURES.value: ["feature", "miss", "add", "need", "want", "funcionalidad", "falt"],
    Topic.PRICING.value: ["price", "pay", "subscri", "free", "cost", "ad", "preco", "pag", "anunci"],
    Topic.PRIVACY_SECURITY.value: ["privacy", "secur", "permiss", "data", "privacidad", "segur"],
    Topic.CUSTOMER_SUPPORT.value: ["support", "help", "contact", "response", "suport", "ajud"],
    Topic.UPDATES.value: ["update", "version", "new", "change", "atualiz", "vers"],
    Topic.USABILITY.value: ["easy", "hard", "simple", "confus", "intuit", "facil", "dific"],
    Topic.OTHER.value: [],
}


class TopicResult(NamedTuple):
    topic_id: int
    topic_words: str
    mapped_labels: list[str]


class LDAModel:
    """Wrapper around sklearn LDA with persistence and taxonomy mapping."""

    def __init__(self, n_topics: int = LDA_NUM_TOPICS, model_path: str | Path = LDA_MODEL_PATH):
        self.n_topics = n_topics
        self.model_path = Path(model_path)
        self.vectorizer: CountVectorizer | None = None
        self.lda: LatentDirichletAllocation | None = None
        self.feature_names: list[str] = []

    def fit(self, documents: list[str]) -> "LDAModel":
        """Train LDA on pre-processed (stemmed/tokenized) documents."""
        if len(documents) < 10:
            raise ValueError(
                f"LDA requires at least 10 documents to train, got {len(documents)}. "
                "Scrape more reviews before running the NLP pipeline."
            )
        logger.info("Fitting LDA with %d topics on %d documents", self.n_topics, len(documents))
        # Use min_df=1 for tiny corpora to avoid an empty vocabulary
        min_df = 2 if len(documents) >= 50 else 1
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=min_df, max_features=5000)
        dtm = self.vectorizer.fit_transform(documents)
        vocab_size = len(self.vectorizer.get_feature_names_out())
        if vocab_size == 0:
            raise ValueError(
                "LDA vocabulary is empty after preprocessing. "
                "Try scraping more reviews or lowering LDA_NUM_TOPICS."
            )
        self.feature_names = list(self.vectorizer.get_feature_names_out())

        self.lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=20,
            learning_method="online",
            random_state=42,
        )
        self.lda.fit(dtm)
        logger.info("LDA training complete.")
        return self

    def save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vectorizer": self.vectorizer, "lda": self.lda, "feature_names": self.feature_names, "n_topics": self.n_topics},
            self.model_path,
        )
        logger.info("LDA model saved to %s", self.model_path)

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        data = joblib.load(self.model_path)
        self.vectorizer = data["vectorizer"]
        self.lda = data["lda"]
        self.feature_names = data["feature_names"]
        self.n_topics = data["n_topics"]
        logger.info("LDA model loaded from %s (%d topics)", self.model_path, self.n_topics)
        return True

    def top_words(self, topic_id: int, n: int = 8) -> list[str]:
        if self.lda is None:
            return []
        topic_dist = self.lda.components_[topic_id]
        top_indices = topic_dist.argsort()[-n:][::-1]
        return [self.feature_names[i] for i in top_indices]

    def _map_topic_to_taxonomy(self, topic_id: int) -> list[str]:
        """Heuristic: overlap between LDA top words and taxonomy keywords."""
        words = set(self.top_words(topic_id, n=15))
        scores: dict[str, int] = {}
        for label, keywords in _TAXONOMY_KEYWORDS.items():
            if not keywords:
                continue
            score = sum(1 for kw in keywords if any(kw in w or w in kw for w in words))
            if score > 0:
                scores[label] = score

        if not scores:
            return ["other"]
        max_score = max(scores.values())
        return sorted(label for label, s in scores.items() if s >= max(1, max_score - 1))

    def predict(self, documents: list[str]) -> list[TopicResult]:
        """Assign dominant topic to each document."""
        if self.lda is None or self.vectorizer is None:
            raise RuntimeError("LDA model not fitted or loaded.")

        dtm = self.vectorizer.transform(documents)
        topic_distributions = self.lda.transform(dtm)

        results: list[TopicResult] = []
        for dist in topic_distributions:
            topic_id = int(np.argmax(dist))
            words = ", ".join(self.top_words(topic_id))
            mapped = self._map_topic_to_taxonomy(topic_id)
            results.append(TopicResult(topic_id=topic_id, topic_words=words, mapped_labels=mapped))

        return results

    def all_topic_summaries(self, n_words: int = 8) -> list[dict]:
        """Return a summary of all discovered topics for dashboard display."""
        if self.lda is None:
            return []
        summaries = []
        for tid in range(self.n_topics):
            summaries.append({
                "topic_id": tid,
                "words": self.top_words(tid, n_words),
                "mapped_labels": self._map_topic_to_taxonomy(tid),
            })
        return summaries
