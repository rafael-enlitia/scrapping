"""Text preprocessing for the NLP pipeline (BERT + LDA)."""

from __future__ import annotations

import re
import logging

import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer, SnowballStemmer

logger = logging.getLogger(__name__)

_NLTK_READY = False


def _ensure_nltk():
    global _NLTK_READY
    if _NLTK_READY:
        return
    for resource in ("stopwords", "rslp"):
        try:
            nltk.data.find(f"corpora/{resource}" if resource == "stopwords" else f"stemmers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    _NLTK_READY = True


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U00002700-\U000027bf"
    "\U0000fe00-\U0000fe0f"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "]+",
    flags=re.UNICODE,
)
_NON_ALPHA_RE = re.compile(r"[^a-záàâãéèêíïóôõúüçñ\s]", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Basic cleaning for BERT input (preserves sentence structure)."""
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_for_lda(text: str, language: str = "portuguese") -> list[str]:
    """Aggressive cleaning + stemming for bag-of-words / LDA."""
    _ensure_nltk()

    text = clean_text(text).lower()
    text = _NON_ALPHA_RE.sub(" ", text)
    tokens = text.split()

    try:
        stop_words = set(stopwords.words(language))
    except OSError:
        stop_words = set(stopwords.words("english"))
    stop_words |= set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    if language == "portuguese":
        stemmer = RSLPStemmer()
    else:
        stemmer = SnowballStemmer(language)

    return [stemmer.stem(t) for t in tokens]


def preprocess_batch(
    texts: list[str],
    language: str = "portuguese",
) -> tuple[list[str], list[str]]:
    """Return (cleaned_texts for BERT, joined_token_strings for LDA vectorizer)."""
    cleaned = [clean_text(t) for t in texts]
    lda_docs = [" ".join(tokenize_for_lda(t, language)) for t in texts]
    return cleaned, lda_docs
