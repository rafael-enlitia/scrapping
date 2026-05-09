from datetime import datetime, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import DATABASE_URL, ensure_data_dir


class Base(DeclarativeBase):
    pass


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String, unique=True, nullable=False, index=True)
    app_id = Column(String, nullable=False, index=True)
    username = Column(String)
    content = Column(Text, nullable=False)
    score = Column(Integer)
    thumbs_up = Column(Integer, default=0)
    app_version = Column(String, index=True)
    review_date = Column(DateTime)
    language = Column(String)
    reply_content = Column(Text)
    reply_date = Column(DateTime)
    scraped_at = Column(DateTime, default=_utcnow)


class Classification(Base):
    __tablename__ = "classifications"
    __table_args__ = (
        UniqueConstraint("review_id", "method", name="uq_classification_review_method"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String, ForeignKey("reviews.review_id"), nullable=False, index=True)
    method = Column(String, nullable=False, default="llm")
    sentiment = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    topics = Column(JSON, nullable=False)
    justification = Column(Text)
    model_name = Column(String)
    raw_response = Column(Text)
    lda_topic_id = Column(Integer, nullable=True)
    lda_topic_words = Column(String, nullable=True)
    classified_at = Column(DateTime, default=_utcnow)


engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    ensure_data_dir()
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()
