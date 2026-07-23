"""Env-driven settings for the serving layer."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class Settings:
    def __init__(self) -> None:
        self.pg_host = os.environ.get("POSTGRES_HOST", "localhost")
        self.pg_port = os.environ.get("POSTGRES_PORT", "5433")
        self.pg_db = os.environ["POSTGRES_DB"]
        self.pg_user = os.environ["POSTGRES_USER"]
        self.pg_password = os.environ["POSTGRES_PASSWORD"]

        # Recommendation tuning: query = mean(liked) - dislike_weight * mean(disliked)
        self.dislike_weight = float(os.environ.get("DISLIKE_WEIGHT", "0.5"))

        # Async pool sizing
        self.pool_min_size = int(os.environ.get("POOL_MIN_SIZE", "1"))
        self.pool_max_size = int(os.environ.get("POOL_MAX_SIZE", "10"))

        # LLM reranker (Week 3). The API key is read from the environment by the
        # Anthropic SDK directly (ANTHROPIC_API_KEY); load_dotenv above pulls it
        # in from .env. Everything else here is a tunable.
        self.anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
        # How many cheap cosine candidates to hand the reranker (N). The LLM then
        # picks/reorders the top k. Larger N = better recall, more input tokens.
        self.rerank_candidates = int(os.environ.get("RERANK_CANDIDATES", "40"))
        # Hard cap on the reranker's output — it only emits an id list, so small.
        self.rerank_max_tokens = int(os.environ.get("RERANK_MAX_TOKENS", "1024"))
        self.anthropic_key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))

    @property
    def conninfo(self) -> str:
        return (
            f"host={self.pg_host} port={self.pg_port} dbname={self.pg_db} "
            f"user={self.pg_user} password={self.pg_password}"
        )


settings = Settings()
