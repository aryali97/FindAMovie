"""Pydantic request/response schemas."""

from pydantic import BaseModel, Field


class SwipeIn(BaseModel):
    user_id: str = Field(..., description="opaque user identifier (no auth)")
    movie_id: int
    liked: bool


class SwipeAck(BaseModel):
    user_id: str
    movie_id: int
    liked: bool
    recorded: bool = True


class Movie(BaseModel):
    movie_id: int
    title: str
    year: int | None = None
    genres: list[str] | None = None
    director: str | None = None


class MovieDetail(Movie):
    tmdb_id: int | None = None
    cast: list[str] | None = None
    tagline: str | None = None
    overview: str | None = None


class ScoredMovie(Movie):
    """A movie plus its cosine distance to the query vector (smaller = closer)."""

    distance: float
