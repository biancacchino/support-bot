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

    # Per-caller limits. About fairness and abuse, not about the Gemini quota -
    # see below, they protect different things and conflating them is how a free
    # tier gets exhausted by callers who were all individually well behaved.
    rate_limit_per_minute: int = 10
    rate_limit_per_day: int = 200

    # The project-wide Gemini budget. Google no longer publishes a fixed free-tier
    # number per model - the docs say limits depend on the account and are shown
    # live in AI Studio (https://aistudio.google.com/rate-limit), and third-party
    # trackers disagree with each other (1,000 vs 1,500 RPD). 15 RPM is the one
    # figure they agree on for flash-lite, and 1,000 RPD is the conservative end of
    # the range.
    #
    # So these are deliberately overridable rather than baked in: check the real
    # number for the key in use and set it, rather than trusting this default.
    gemini_rpm: int = 15
    gemini_rpd: int = 1000

    # One turn can cost two Gemini calls: condensing the follow-up, then writing
    # the answer. The upstream budget is therefore sized in turns, not requests -
    # spending a 15-RPM budget as 15 turns would overrun it by 2x on any
    # conversation past the first turn.
    gemini_calls_per_turn: int = 2

    @property
    def upstream_turns_per_minute(self) -> int:
        return max(1, self.gemini_rpm // self.gemini_calls_per_turn)

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

    # When set, /admin/metrics requires it in an X-Admin-Key header. Empty means
    # open, which is fine locally and is not fine on the internet: the endpoint
    # exposes no message content, but "how often does this bot fail" is still not a
    # number to hand out by default.
    admin_api_key: str = ""

    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()
