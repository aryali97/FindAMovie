"""FastAPI serving layer: swipe ingestion + retrieval endpoints.

The async connection pool is opened once in the lifespan and shared across
requests. Each request borrows a pooled connection via the get_conn dependency
(which commits on successful exit, so POST /swipes persists).
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from psycopg import AsyncConnection
from psycopg.errors import ForeignKeyViolation

from . import retrieval as R
from . import reranker as Rr
from .config import settings
from .db import build_pool
from .models import Movie, MovieDetail, ScoredMovie, SwipeAck, SwipeIn


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = build_pool()
    await pool.open()
    app.state.pool = pool
    # One Anthropic client per process, shared across requests like the pool.
    # None when no API key is configured — the rerank path 503s in that case.
    app.state.anthropic = Rr.build_client() if settings.anthropic_key_present else None
    try:
        yield
    finally:
        await pool.close()
        if app.state.anthropic is not None:
            await app.state.anthropic.close()


app = FastAPI(title="FindAMovie", lifespan=lifespan)


async def get_conn(request: Request) -> AsyncConnection:
    """Borrow a connection from the pool for the duration of a request."""
    async with request.app.state.pool.connection() as conn:
        yield conn


@app.get("/healthz")
async def healthz(conn: AsyncConnection = Depends(get_conn)) -> dict:
    await conn.execute("SELECT 1")
    return {"status": "ok"}


@app.post("/swipes", response_model=SwipeAck)
async def post_swipe(
    swipe: SwipeIn, conn: AsyncConnection = Depends(get_conn)
) -> SwipeAck:
    try:
        await R.record_swipe(conn, swipe.user_id, swipe.movie_id, swipe.liked)
    except ForeignKeyViolation:
        raise HTTPException(404, f"movie_id {swipe.movie_id} not found")
    return SwipeAck(
        user_id=swipe.user_id, movie_id=swipe.movie_id, liked=swipe.liked
    )


@app.get("/recommendations", response_model=list[ScoredMovie])
async def get_recommendations(
    request: Request,
    user_id: str,
    k: int = Query(10, ge=1, le=100),
    rerank: bool = Query(True, description="LLM rerank the candidates; false = cosine baseline"),
    dislike_weight: float | None = Query(None, ge=0),
    conn: AsyncConnection = Depends(get_conn),
) -> list[ScoredMovie]:
    try:
        if not rerank:
            # Cheap cosine baseline (the "before" for Week 4 instrumentation).
            return await R.recommend_for_user(
                conn, user_id, k=k, dislike_weight=dislike_weight
            )

        # Reranked path: over-fetch cheap cosine candidates, then one Claude call
        # picks/reorders the top-k using the candidate metadata + the user's taste.
        client = request.app.state.anthropic
        if client is None:
            raise HTTPException(
                503,
                "reranker unavailable: set ANTHROPIC_API_KEY in .env to use rerank=true "
                "(or call with rerank=false for the cosine baseline)",
            )
        candidates = await R.recommend_for_user(
            conn, user_id, k=settings.rerank_candidates, dislike_weight=dislike_weight
        )
        details = await R.get_movies_by_ids(conn, [c.movie_id for c in candidates])
        liked = await R.liked_movie_titles(conn, [user_id])
        result = await Rr.rerank(
            client,
            settings.anthropic_model,
            liked,
            details,
            k=k,
            max_tokens=settings.rerank_max_tokens,
        )
        # Reorder the cosine candidates by the LLM's ranking; keep each movie's
        # original cosine distance in the response so the two paths stay comparable.
        by_id = {c.movie_id: c for c in candidates}
        return [by_id[i] for i in result.order if i in by_id]
    except R.NoLikesError:
        raise HTTPException(
            422, f"user {user_id!r} has no liked movies; like at least one to get recommendations"
        )


@app.get("/movies", response_model=list[Movie])
async def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    conn: AsyncConnection = Depends(get_conn),
) -> list[Movie]:
    return await R.search_movies(conn, q, limit=limit)


@app.get("/movies/{movie_id}", response_model=MovieDetail)
async def movie_detail(
    movie_id: int, conn: AsyncConnection = Depends(get_conn)
) -> MovieDetail:
    movie = await R.get_movie(conn, movie_id)
    if movie is None:
        raise HTTPException(404, f"movie_id {movie_id} not found")
    return movie


@app.get("/movies/{movie_id}/similar", response_model=list[ScoredMovie])
async def similar(
    movie_id: int,
    k: int = Query(10, ge=1, le=100),
    conn: AsyncConnection = Depends(get_conn),
) -> list[ScoredMovie]:
    try:
        return await R.similar_to(conn, movie_id, k=k)
    except ValueError:
        raise HTTPException(404, f"movie_id {movie_id} not found")
