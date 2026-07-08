"""Build the text blob that gets embedded for each movie.

This is the single source of truth for what the embedding model "sees".
The load script persists the exact output in movies.embedding_text so that
retrieval behavior is always reproducible and debuggable.

Design:
  - A labeled, multi-line format. Labels ("Genres:", "Director:") give the
    model light structure without dominating the plot signal.
  - Overview (the plot) goes last and unlabeled-heavy weight is carried by it
    being the longest field — it's the strongest similarity signal.
  - Empty/missing fields are omitted entirely (no "Director: None" noise).
  - bge-small-en-v1.5 is used symmetrically here (movie<->movie), so we add
    NO query instruction prefix; that's only for asymmetric query->passage use.
"""

from collections.abc import Sequence


def build_embedding_text(
    *,
    title: str,
    year: int | None = None,
    genres: Sequence[str] | None = None,
    director: str | None = None,
    cast: Sequence[str] | None = None,
    tagline: str | None = None,
    overview: str | None = None,
) -> str:
    lines: list[str] = []

    head = title if not year else f"{title} ({year})"
    lines.append(f"Title: {head}")

    if genres:
        lines.append(f"Genres: {', '.join(genres)}")
    if director:
        lines.append(f"Director: {director}")
    if cast:
        lines.append(f"Cast: {', '.join(cast)}")
    if tagline:
        lines.append(f"Tagline: {tagline}")
    if overview:
        lines.append(f"Overview: {overview}")

    return "\n".join(lines)


def build_from_row(row) -> str:
    """Adapter for a pandas row / namedtuple from the TMDB metadata parquet."""
    def _clean(x):
        # parquet list columns come back as numpy arrays / lists; normalize
        return list(x) if x is not None and len(x) else None

    return build_embedding_text(
        title=row["tmdb_title"] or row.get("title"),
        year=int(row["year"]) if row.get("year") is not None else None,
        genres=_clean(row["genres"]),
        director=row["director"],
        cast=_clean(row["cast"]),
        tagline=row["tagline"],
        overview=row["overview"],
    )


if __name__ == "__main__":
    # Render a few real movies from the TMDB cache for inspection.
    from pathlib import Path

    import pandas as pd

    ROOT = Path(__file__).resolve().parent.parent
    meta = pd.read_parquet(ROOT / "data" / "tmdb" / "metadata.parquet")
    movies = pd.read_parquet(
        ROOT / "data" / "subset" / "movies.parquet", columns=["movieId", "year"]
    )
    df = meta.merge(movies, left_on="movieId", right_on="movieId", how="left")

    for title in ("Inception", "Dune"):
        row = df[df["tmdb_title"] == title]
        if row.empty:
            continue
        print("=" * 72)
        print(build_from_row(row.iloc[0]))
        print()
