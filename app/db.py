"""Async Postgres connection pool.

A single pool is created per process and shared across requests (see main.py
lifespan) — never connect per-request, which would dominate latency. Each pooled
connection has the pgvector type adapter registered once via the `configure` hook,
so `vector` columns round-trip as numpy arrays.
"""

from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async

from .config import settings


async def _configure(conn) -> None:
    """Run once per new pooled connection."""
    await register_vector_async(conn)


def build_pool() -> AsyncConnectionPool:
    """Construct the pool (unopened — main.py opens it in the lifespan)."""
    return AsyncConnectionPool(
        conninfo=settings.conninfo,
        min_size=settings.pool_min_size,
        max_size=settings.pool_max_size,
        configure=_configure,
        open=False,
    )
