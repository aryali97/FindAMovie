# FindAMovie

Building movie recommendation infra as well as LLM optimization and experimentation.

The deliverable is an **LLM reranker behind a purpose-built serving layer**. Modeling
is deliberately simple (cosine-similarity retrieval); the interesting work is the
serving stack. A full architecture design doc lands later — this README is the
practical run guide for the current state.

## Current state

Retrieval pipeline works end-to-end: **embeddings → pgvector → cosine similarity**.

- 4,759 movies (MovieLens subset, 2010–2023) with TMDB metadata, embedded and stored
  in Postgres/pgvector.
- Retrieval primitives run as a script/function. **No HTTP API yet** (that's next).

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

# heavy scientific stack via conda-forge (native arm64)
conda install -p ./env -c conda-forge \
  pytorch sentence-transformers pandas pyarrow requests python-dotenv tqdm -y

# DB adapters via pip
./env/bin/python -m pip install "psycopg[binary]" pgvector
```

Run everything with `./env/bin/python …` — no activation needed.

### 3. Database

```bash
docker compose up -d                    # start Postgres + pgvector on localhost:5433
docker exec -i -e PGPASSWORD=$POSTGRES_PASSWORD movie-recommender-db \
  psql -U $POSTGRES_USER -d $POSTGRES_DB < db/schema.sql   # apply schema (idempotent)
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

## Layout

```
db/schema.sql            movies table (vector(384)) + HNSW cosine index
docker-compose.yml       Postgres + pgvector
scripts/subset.py        MovieLens -> dense subset
scripts/fetch_tmdb.py    TMDB metadata fetch (resumable, rate-limited)
scripts/embedding_text.py  builds the per-movie text blob that gets embedded
scripts/load_movies.py   embed (bge-small on MPS) + upsert into Postgres
scripts/retrieve.py      cosine retrieval primitives
```
