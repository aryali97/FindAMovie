-- Swipe ingestion: one row per (user, movie) opinion.
-- Apply after schema.sql. Idempotent: safe to re-run.

CREATE TABLE IF NOT EXISTS swipes (
    user_id     TEXT        NOT NULL,
    movie_id    INTEGER     NOT NULL REFERENCES movies(movie_id),
    liked       BOOLEAN     NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- one opinion per user per movie; re-swiping upserts (see POST /swipes)
    PRIMARY KEY (user_id, movie_id)
);
