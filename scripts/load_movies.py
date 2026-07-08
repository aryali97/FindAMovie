"""Embed the subset movies and load them into Postgres (pgvector).

Offline build-once path:
  1. read TMDB metadata + subset movies (for year)
  2. keep only rows with a usable overview (status == "ok")
  3. build the embedding text blob per movie (embedding_text.build_from_row)
  4. embed with bge-small-en-v1.5 (batched, on MPS if available)
  5. upsert into the movies table (idempotent via ON CONFLICT)

Run:  ./env/bin/python scripts/load_movies.py
"""

import os
from pathlib import Path

import pandas as pd
import psycopg
import torch
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from embedding_text import build_from_row

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "tmdb" / "metadata.parquet"
SUBSET = ROOT / "data" / "subset" / "movies.parquet"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dsn() -> str:
    return (
        f"host={os.environ['POSTGRES_HOST']} port={os.environ['POSTGRES_PORT']} "
        f"dbname={os.environ['POSTGRES_DB']} user={os.environ['POSTGRES_USER']} "
        f"password={os.environ['POSTGRES_PASSWORD']}"
    )


def load_frame() -> pd.DataFrame:
    meta = pd.read_parquet(META)
    subset = pd.read_parquet(SUBSET, columns=["movieId", "title", "year"])
    df = meta.merge(subset, on="movieId", how="left", suffixes=("", "_ml"))

    usable = df[df["status"] == "ok"].copy()
    print(f"metadata rows: {len(df):,} | usable (status=ok): {len(usable):,}")
    usable["embedding_text"] = usable.apply(build_from_row, axis=1)
    return usable


def main() -> None:
    load_dotenv(ROOT / ".env")
    df = load_frame()

    device = pick_device()
    print(f"embedding {len(df):,} movies with {MODEL_NAME} on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    embeddings = model.encode(
        df["embedding_text"].tolist(),
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,     # unit vectors; cosine-friendly
        show_progress_bar=True,
    )
    print(f"embeddings shape: {embeddings.shape}")

    rows = [
        (
            int(r.movieId),
            int(r.tmdbId),
            r.tmdb_title or r.title,
            int(r.year) if pd.notna(r.year) else None,
            list(r.genres) if r.genres is not None and len(r.genres) else None,
            r.director,
            list(r.cast) if r.cast is not None and len(r.cast) else None,
            r.tagline,
            r.overview,
            r.embedding_text,
            emb,
        )
        for r, emb in zip(df.itertuples(index=False), embeddings)
    ]

    upsert = """
        INSERT INTO movies (movie_id, tmdb_id, title, year, genres, director,
                            "cast", tagline, overview, embedding_text, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (movie_id) DO UPDATE SET
            tmdb_id        = EXCLUDED.tmdb_id,
            title          = EXCLUDED.title,
            year           = EXCLUDED.year,
            genres         = EXCLUDED.genres,
            director       = EXCLUDED.director,
            "cast"         = EXCLUDED."cast",
            tagline        = EXCLUDED.tagline,
            overview       = EXCLUDED.overview,
            embedding_text = EXCLUDED.embedding_text,
            embedding      = EXCLUDED.embedding;
    """

    with psycopg.connect(dsn()) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.executemany(upsert, rows)
        conn.commit()
        n = conn.execute("SELECT count(*) FROM movies").fetchone()[0]

    print(f"done. movies table now has {n:,} rows.")


if __name__ == "__main__":
    main()
