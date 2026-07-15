"""FastAPI serving layer for the movie recommender.

Offline pipeline lives in scripts/; this package is the request path:
  config.py     env-driven settings
  db.py         async psycopg connection pool
  models.py     pydantic request/response schemas
  retrieval.py  cosine retrieval primitives (async)
  main.py       FastAPI app + routes
"""
