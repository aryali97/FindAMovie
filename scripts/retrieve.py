"""Cosine-similarity retrieval over the movies table (pgvector).

Two primitives Week 2's API will call:
  - similar_to(movie_id):    "movies like this one"
  - recommend_for(movie_ids): average the liked movies' vectors into a taste
                              centroid, return nearest movies (excluding seeds)

Both run the same SQL shape:  ORDER BY embedding <=> :query_vector LIMIT k
where <=> is pgvector's cosine distance (smaller = more similar).

Run directly for a sanity check:  ./env/bin/python scripts/retrieve.py
"""

import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parent.parent


def dsn() -> str:
    return (
        f"host={os.environ['POSTGRES_HOST']} port={os.environ['POSTGRES_PORT']} "
        f"dbname={os.environ['POSTGRES_DB']} user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


def _connect() -> psycopg.Connection:
    conn = psycopg.connect(dsn())
    register_vector(conn)
    return conn


def _nearest(conn, query_vec, k, exclude_ids):
    """Top-k movies by cosine distance to query_vec, excluding exclude_ids."""
    rows = conn.execute(
        """
        SELECT movie_id, title, year, genres, director,
               embedding <=> %s AS distance
        FROM movies
        WHERE movie_id <> ALL(%s)
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (query_vec, exclude_ids, query_vec, k),
    ).fetchall()
    return rows


def similar_to(movie_id: int, k: int = 10, conn=None):
    own = conn is None
    conn = conn or _connect()
    try:
        seed = conn.execute(
            "SELECT embedding FROM movies WHERE movie_id = %s", (movie_id,)
        ).fetchone()
        if seed is None:
            raise ValueError(f"movie_id {movie_id} not found")
        return _nearest(conn, seed[0], k, [movie_id])
    finally:
        if own:
            conn.close()


def recommend_for(movie_ids: list[int], k: int = 10, conn=None):
    own = conn is None
    conn = conn or _connect()
    try:
        vecs = conn.execute(
            "SELECT embedding FROM movies WHERE movie_id = ANY(%s)", (movie_ids,)
        ).fetchall()
        if not vecs:
            raise ValueError("none of the given movie_ids were found")
        # pgvector returns Vector objects; use .to_numpy() before averaging
        centroid = np.mean([v[0].to_numpy() for v in vecs], axis=0)  # taste centroid
        return _nearest(conn, centroid, k, movie_ids)
    finally:
        if own:
            conn.close()


def _title_to_id(conn, title: str) -> int:
    row = conn.execute(
        "SELECT movie_id FROM movies WHERE title = %s ORDER BY year LIMIT 1", (title,)
    ).fetchone()
    if row is None:
        raise ValueError(f"title {title!r} not found")
    return row[0]


def _print(header, rows):
    print(f"\n{header}")
    for movie_id, title, year, genres, director, dist in rows:
        g = ", ".join((genres or [])[:3])
        print(f"  {dist:.3f}  {title} ({year})  [{g}]  dir: {director}")


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    with _connect() as conn:
        # --- single-seed: "movies like X" ---
        for title in ("Inception", "Dune", "Bridesmaids"):
            try:
                mid = _title_to_id(conn, title)
            except ValueError as e:
                print(f"(skip: {e})")
                continue
            _print(f"Similar to {title!r}:", similar_to(mid, k=8, conn=conn))

        # --- multi-seed: taste centroid over a few sci-fi picks ---
        seeds = []
        for t in ("Inception", "Dune", "Interstellar"):
            try:
                seeds.append(_title_to_id(conn, t))
            except ValueError:
                pass
        if seeds:
            _print(
                f"Recommended for someone who liked {len(seeds)} sci-fi films:",
                recommend_for(seeds, k=8, conn=conn),
            )
