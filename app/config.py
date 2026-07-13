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

    # The task list asked for gemini-2.5-flash-lite. Google has since closed that
    # model to new API keys - it still appears in models.list(), but calling it
    # returns 404 "no longer available to new users", so the spec's choice is not
    # buildable as written. gemini-3.1-flash-lite is the current model at the same
    # tier and works on this key.
    #
    # Pinned to an explicit version rather than the gemini-flash-lite-latest
    # alias: the alias moves under you, and Phase 11's benchmark numbers are only
    # worth showing if the model that produced them can be named.
    gemini_model: str = "gemini-3.1-flash-lite"

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "support_kb"

    redis_url: str = "redis://redis:6379/0"

    # Escalate when the top reranker score falls below this. Scored on the
    # cross-encoder, never on raw cosine similarity: an off-topic query can
    # still land a high embedding similarity, which is the exact failure this
    # gate exists to catch.
    #
    # 0.35 is measured, not guessed. Against the 8 in-scope smoke queries and 9
    # off-topic ones, the worst genuine query scores 0.455 and the best impostor
    # scores 0.084, so anything in that gap separates them; 0.35 sits inside it
    # with room on both sides. The 0.5 this started as escalated 2 of the 8
    # genuine queries, which would have quietly eaten the deflection target.
    #
    # This is tuned on 17 queries. Phase 11 re-tunes it on 200-300 Bitext ones,
    # which is the number to trust.
    confidence_threshold: float = 0.35

    conversation_ttl_seconds: int = 3600

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Stage 1 vector search breadth, then what survives reranking.
    retrieval_top_n: int = 10
    rerank_top_k: int = 4

    kb_dir: str = "kb"

    # all-MiniLM-L6-v2 reads at most 256 tokens and silently truncates the rest,
    # so the chunk budget plus its "title > heading" prefix has to fit inside
    # that. Ingestion refuses to run if this is set too high rather than let the
    # tail of every long chunk be stored but never embedded.
    chunk_size_tokens: int = 224
    chunk_overlap_tokens: int = 40

    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()
