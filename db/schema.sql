-- Movie retrieval schema: metadata + embedding vectors in one table.
-- Applied against the pgvector container (see docker-compose.yml).
-- Idempotent: safe to re-run.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS movies (
    movie_id        INTEGER PRIMARY KEY,        -- MovieLens id (joins to ratings)
    tmdb_id         INTEGER NOT NULL,           -- TMDB id (source of metadata)
    title           TEXT    NOT NULL,
    year            INTEGER,
    genres          TEXT[],                     -- native array, not a delimited string
    director        TEXT,
    "cast"          TEXT[],                     -- top-6 billed actors
    tagline         TEXT,
    overview        TEXT,
    embedding_text  TEXT    NOT NULL,           -- exact string we embedded (reproducibility)
    embedding       VECTOR(384) NOT NULL        -- bge-small-en-v1.5 output
);

-- HNSW graph index for approximate nearest-neighbor search under cosine distance.
-- Matches the `<=>` operator used by the retrieval query.
CREATE INDEX IF NOT EXISTS movies_embedding_hnsw
    ON movies USING hnsw (embedding vector_cosine_ops);
