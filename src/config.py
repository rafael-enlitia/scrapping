import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {name}={raw!r} is not a valid integer. "
            f"Please set it to a whole number (e.g. {default})."
        ) from None


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

DB_PATH = DATA_DIR / "reviews.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# IAEDU agent-chat (https://api.iaedu.pt)
IAEDU_API_KEY = os.getenv("IAEDU_API_KEY", os.getenv("OPENAI_API_KEY", ""))
IAEDU_CHANNEL_ID = os.getenv("IAEDU_CHANNEL_ID", "")
IAEDU_ENDPOINT = os.getenv(
    "IAEDU_ENDPOINT",
    "https://api.iaedu.pt/agent-chat/api/v1/agent/cmor5objoex9gfp01vm7p95jh/stream",
)

# NLP pipeline (BERT + LDA)
BERT_MODEL = os.getenv("BERT_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment")
LDA_NUM_TOPICS = _int_env("LDA_NUM_TOPICS", 8)
LDA_MODEL_PATH = DATA_DIR / "lda_model.pkl"
NLP_BATCH_SIZE = _int_env("NLP_BATCH_SIZE", 32)

# Scraping
DEFAULT_APP_ID = os.getenv("DEFAULT_APP_ID", "com.whatsapp")
SCRAPE_LANG = os.getenv("SCRAPE_LANG", "pt")
SCRAPE_COUNTRY = os.getenv("SCRAPE_COUNTRY", "pt")


def ensure_data_dir() -> None:
    """Create the data directory if it does not exist. Call explicitly when needed."""
    DATA_DIR.mkdir(exist_ok=True)
