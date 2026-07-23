"""Async data-access + cosine retrieval over the movies/swipes tables.

Every function takes an open async connection from the pool (see main.py) and
reuses *stored* embeddings — no embedding model runs in this process.

Retrieval query shape:  ORDER BY embedding <=> :q LIMIT k
where <=> is pgvector cosine distance (smaller = closer).

recommend_for_users takes a SET of users so the same code serves solo recs today
(a one-element set) and group recs later (a bigger set): the query vector is
    mean(liked vectors) - dislike_weight * mean(disliked vectors)
built from every swipe across the set, excluding movies anyone in the set has
already swiped.
"""

import numpy as np
from psycopg import AsyncConnection

from .config import settings
from .models import Movie, MovieDetail, ScoredMovie


class NoLikesError(Exception):
    """Raised when recommendations are requested but no liked movies exist."""


# --- swipe ingestion -------------------------------------------------------

async def record_swipe(
    conn: AsyncConnection, user_id: str, movie_id: int, liked: bool
) -> None:
    """Upsert a swipe: one opinion per (user, movie); re-swiping overwrites."""
    await conn.execute(
        """
        INSERT INTO swipes (user_id, movie_id, liked)
        VALUES (%s, %s, %s)
        ON CONFLICT (user_id, movie_id)
        DO UPDATE SET liked = EXCLUDED.liked, created_at = now()
        """,
        (user_id, movie_id, liked),
    )


# --- movie lookups ---------------------------------------------------------

async def get_movie(conn: AsyncConnection, movie_id: int) -> MovieDetail | None:
    row = await (
        await conn.execute(
            """
            SELECT movie_id, tmdb_id, title, year, genres, director,
                   "cast", tagline, overview
            FROM movies WHERE movie_id = %s
            """,
            (movie_id,),
        )
    ).fetchone()
    if row is None:
        return None
    return MovieDetail(
        movie_id=row[0], tmdb_id=row[1], title=row[2], year=row[3],
        genres=row[4], director=row[5], cast=row[6], tagline=row[7],
        overview=row[8],
    )


async def get_movies_by_ids(
    conn: AsyncConnection, movie_ids: list[int]
) -> list[MovieDetail]:
    """Full metadata for a set of ids, returned in the order given.

    Used to hydrate cheap cosine candidates (which carry only id/title/etc.) with
    the overview/cast/tagline the LLM reranker needs. One query, then re-ordered
    in Python to match movie_ids since SQL makes no ordering promise.
    """
    if not movie_ids:
        return []
    rows = await (
        await conn.execute(
            """
            SELECT movie_id, tmdb_id, title, year, genres, director,
                   "cast", tagline, overview
            FROM movies WHERE movie_id = ANY(%s::int[])
            """,
            (movie_ids,),
        )
    ).fetchall()
    by_id = {
        r[0]: MovieDetail(
            movie_id=r[0], tmdb_id=r[1], title=r[2], year=r[3], genres=r[4],
            director=r[5], cast=r[6], tagline=r[7], overview=r[8],
        )
        for r in rows
    }
    return [by_id[i] for i in movie_ids if i in by_id]


async def liked_movie_titles(
    conn: AsyncConnection, user_ids: list[str], limit: int = 50
) -> list[str]:
    """Titles the user set has liked — the taste signal handed to the reranker."""
    rows = await (
        await conn.execute(
            """
            SELECT m.title, m.year
            FROM swipes s JOIN movies m ON m.movie_id = s.movie_id
            WHERE s.user_id = ANY(%s) AND s.liked = true
            ORDER BY s.created_at DESC
            LIMIT %s
            """,
            (user_ids, limit),
        )
    ).fetchall()
    return [f"{t} ({y})" if y else t for t, y in rows]


async def search_movies(
    conn: AsyncConnection, q: str, limit: int = 20
) -> list[Movie]:
    rows = await (
        await conn.execute(
            """
            SELECT movie_id, title, year, genres, director
            FROM movies WHERE title ILIKE %s ORDER BY title LIMIT %s
            """,
            (f"%{q}%", limit),
        )
    ).fetchall()
    return [
        Movie(movie_id=r[0], title=r[1], year=r[2], genres=r[3], director=r[4])
        for r in rows
    ]


# --- cosine retrieval ------------------------------------------------------

async def _nearest(
    conn: AsyncConnection, query_vec, k: int, exclude_ids: list[int]
) -> list[ScoredMovie]:
    """Top-k movies by cosine distance to query_vec, excluding exclude_ids."""
    rows = await (
        await conn.execute(
            """
            SELECT movie_id, title, year, genres, director,
                   embedding <=> %s AS distance
            FROM movies
            WHERE movie_id <> ALL(%s::int[])
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_vec, exclude_ids, query_vec, k),
        )
    ).fetchall()
    return [
        ScoredMovie(
            movie_id=r[0], title=r[1], year=r[2], genres=r[3],
            director=r[4], distance=float(r[5]),
        )
        for r in rows
    ]


async def similar_to(
    conn: AsyncConnection, movie_id: int, k: int = 10
) -> list[ScoredMovie]:
    """Movies most similar to one seed (reuses its stored embedding)."""
    seed = await (
        await conn.execute(
            "SELECT embedding FROM movies WHERE movie_id = %s", (movie_id,)
        )
    ).fetchone()
    if seed is None:
        raise ValueError(f"movie_id {movie_id} not found")
    return await _nearest(conn, seed[0], k, [movie_id])


async def _swipe_vectors(
    conn: AsyncConnection, user_ids: list[str], liked: bool
) -> list[np.ndarray]:
    rows = await (
        await conn.execute(
            """
            SELECT m.embedding
            FROM swipes s JOIN movies m ON m.movie_id = s.movie_id
            WHERE s.user_id = ANY(%s) AND s.liked = %s
            """,
            (user_ids, liked),
        )
    ).fetchall()
    return [r[0].to_numpy() for r in rows]


async def _swiped_ids(conn: AsyncConnection, user_ids: list[str]) -> list[int]:
    rows = await (
        await conn.execute(
            "SELECT DISTINCT movie_id FROM swipes WHERE user_id = ANY(%s)",
            (user_ids,),
        )
    ).fetchall()
    return [r[0] for r in rows]


async def recommend_for_users(
    conn: AsyncConnection,
    user_ids: list[str],
    k: int = 10,
    dislike_weight: float | None = None,
) -> list[ScoredMovie]:
    """Recommend for a set of users via a taste centroid.

    query = mean(liked) - dislike_weight * mean(disliked)
    Likes are required; dislikes are an optional modifier. Movies already
    swiped by anyone in the set are excluded.
    """
    if dislike_weight is None:
        dislike_weight = settings.dislike_weight

    liked = await _swipe_vectors(conn, user_ids, liked=True)
    if not liked:
        raise NoLikesError("no liked movies for the given user(s)")

    query = np.mean(liked, axis=0)
    disliked = await _swipe_vectors(conn, user_ids, liked=False)
    if disliked:
        query = query - dislike_weight * np.mean(disliked, axis=0)

    exclude = await _swiped_ids(conn, user_ids)
    return await _nearest(conn, query, k, exclude)


async def recommend_for_user(
    conn: AsyncConnection,
    user_id: str,
    k: int = 10,
    dislike_weight: float | None = None,
) -> list[ScoredMovie]:
    """Solo convenience over recommend_for_users (a one-element set)."""
    return await recommend_for_users(conn, [user_id], k, dislike_weight)
