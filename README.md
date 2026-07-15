# FindAMovie

Building movie recommendation infra as well as LLM optimization and experimentation.

The deliverable is an **LLM reranker behind a purpose-built serving layer**. Modeling
is deliberately simple (cosine-similarity retrieval); the interesting work is the
serving stack. A full architecture design doc lands later — this README is the
practical run guide for the current state.

## Current state

Retrieval pipeline + async serving API working end-to-end.

- 4,759 movies (MovieLens subset, 2010–2023) with TMDB metadata, embedded and stored
  in Postgres/pgvector.
- **FastAPI serving layer** (async, pooled Postgres connections): swipe ingestion +
  recommendations. No embedding model runs in the API process — every endpoint reuses
  stored vectors. **No LLM reranker yet** (that's next).

## Prerequisites

- **Docker** (for Postgres + pgvector)
- **conda** — this project uses a directory-local, native-arm64 env at `./env`
  (built with [miniforge](https://github.com/conda-forge/miniforge) on Apple Silicon)
- A **TMDB API token** (free): https://www.themoviedb.org/settings/api

## Setup

### 1. Environment variables

```bash
cp .env.example .env
# then edit .env: set TMDB_READ_TOKEN, and adjust POSTGRES_* if needed
```

`POSTGRES_PORT` defaults to `5433` (5432 is often taken by another local Postgres).

### 2. Python env (directory-local, native arm64)

```bash
# create the env
conda create -p ./env python=3.11 -y

# heavy scientific stack + serving deps via conda-forge (native arm64)
conda install -p ./env -c conda-forge \
  pytorch sentence-transformers pandas pyarrow requests python-dotenv tqdm \
  fastapi uvicorn psycopg-pool -y

# DB adapters via pip
./env/bin/python -m pip install "psycopg[binary]" pgvector
```

> Note: `pytorch`/`sentence-transformers` are only needed for the **offline**
> embedding step (`scripts/load_movies.py`). The serving API itself does not import
> torch.

Run everything with `./env/bin/python …` — no activation needed.

### 3. Database

```bash
docker compose up -d                    # start Postgres + pgvector on localhost:5433
# apply schema + swipes migration (both idempotent)
docker exec -i -e PGPASSWORD=$POSTGRES_PASSWORD movie-recommender-db \
  psql -U $POSTGRES_USER -d $POSTGRES_DB < db/schema.sql
docker exec -i -e PGPASSWORD=$POSTGRES_PASSWORD movie-recommender-db \
  psql -U $POSTGRES_USER -d $POSTGRES_DB < db/swipes.sql
```

`docker compose down` stops it and keeps data; `down -v` wipes the volume.

## Data pipeline

Data lives under `data/` (gitignored — reproducible via these scripts).

```bash
# 1. MovieLens: download ml-32m into data/ml-32m/ (https://grouplens.org/datasets/movielens/)

# 2. Build a smaller, denser subset (writes data/subset/*.parquet)
./env/bin/python scripts/subset.py

# 3. Fetch TMDB metadata for the subset (writes data/tmdb/metadata.parquet)
#    resumable + rate-limited; ~4 min for the full subset
./env/bin/python scripts/fetch_tmdb.py

# 4. Embed + load into Postgres (idempotent upsert)
./env/bin/python scripts/load_movies.py
```

## Retrieval

```bash
./env/bin/python scripts/retrieve.py    # runs a sanity demo over a few seed movies
```

The module exposes two primitives (both take MovieLens movie ids):

- `similar_to(movie_id)` — nearest movies to one seed ("movies like this")
- `recommend_for([movie_ids])` — nearest movies to the average of several liked
  vectors (a "taste centroid")

Both run `ORDER BY embedding <=> :q LIMIT k`, where `<=>` is pgvector cosine distance.

## Serving API

```bash
./env/bin/python -m uvicorn app.main:app --port 8000   # interactive docs at /docs
```

Async FastAPI over a pooled Postgres connection. Endpoints:

| Method | Route | Purpose |
|--------|-------|---------|
| `POST` | `/swipes` | record a like/dislike (`{user_id, movie_id, liked}`); upserts |
| `GET` | `/recommendations?user_id=&k=&dislike_weight=` | taste-centroid recs, excludes swiped |
| `GET` | `/movies/{id}/similar?k=` | "more like this" |
| `GET` | `/movies?q=&limit=` | title search |
| `GET` | `/movies/{id}` | movie detail |
| `GET` | `/healthz` | liveness |

Recommendations use a taste centroid `mean(liked) − w·mean(disliked)` (`w` =
`DISLIKE_WEIGHT`, default 0.5). Likes are required (a user with no likes gets a 422);
dislikes are an optional modifier. The centroid logic takes a *set* of users, so group
recommendations are a future extension (a bigger set), not a rewrite.

```bash
# example loop
curl -X POST localhost:8000/swipes -H 'content-type: application/json' \
  -d '{"user_id":"me","movie_id":79132,"liked":true}'
curl "localhost:8000/recommendations?user_id=me&k=5"
```

## Layout

```
app/main.py              FastAPI app + routes
app/db.py                async psycopg connection pool
app/retrieval.py         async data-access + cosine retrieval (dislike-subtraction)
app/models.py            pydantic schemas
app/config.py            env-driven settings
db/schema.sql            movies table (vector(384)) + HNSW cosine index
db/swipes.sql            swipes table (user opinions)
docker-compose.yml       Postgres + pgvector
scripts/subset.py        MovieLens -> dense subset
scripts/fetch_tmdb.py    TMDB metadata fetch (resumable, rate-limited)
scripts/embedding_text.py  builds the per-movie text blob that gets embedded
scripts/load_movies.py   embed (bge-small on MPS) + upsert into Postgres
scripts/retrieve.py      cosine retrieval primitives (offline/CLI)
```
