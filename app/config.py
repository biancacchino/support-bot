"""Environment-driven settings for the support bot."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "support_kb"

    redis_url: str = "redis://redis:6379/0"

    # Escalate when the top reranker score falls below this. Scored on the
    # cross-encoder, never on raw cosine similarity: an off-topic query can
    # still land a high embedding similarity, which is the exact failure this
    # gate exists to catch.
    confidence_threshold: float = 0.5

    conversation_ttl_seconds: int = 3600

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Stage 1 vector search breadth, then what survives reranking.
    retrieval_top_n: int = 10
    rerank_top_k: int = 4

    kb_dir: str = "kb"
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 60

    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()
