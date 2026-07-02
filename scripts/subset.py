"""Build a smaller, denser MovieLens subset for iteration.

Filters the full ml-32m dataset down to:
  - movies released in MIN_YEAR or later (year parsed from the title string)
  - movies with at least MIN_RATINGS_PER_MOVIE ratings (after the year filter)
  - users with at least MIN_RATINGS_PER_USER ratings (on the surviving movies)

Writes the trimmed movies/ratings to data/subset/ as Parquet.
"""

import re
from pathlib import Path

import pandas as pd

# --- tunable knobs ---------------------------------------------------------
MIN_YEAR = 2010
MIN_RATINGS_PER_MOVIE = 50
MIN_RATINGS_PER_USER = 5
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "ml-32m"
OUT = ROOT / "data" / "subset"
OUT.mkdir(parents=True, exist_ok=True)

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def parse_year(title: str):
    m = YEAR_RE.search(title.strip())
    return int(m.group(1)) if m else None


def main():
    print("Loading movies...")
    movies = pd.read_csv(SRC / "movies.csv")
    movies["year"] = movies["title"].map(parse_year)
    n_total = len(movies)
    n_no_year = movies["year"].isna().sum()

    recent = movies[movies["year"] >= MIN_YEAR].copy()
    print(f"  movies total: {n_total:,}  (no parseable year: {n_no_year:,})")
    print(f"  movies >= {MIN_YEAR}: {len(recent):,}")

    recent_ids = set(recent["movieId"])

    print("Loading ratings (this is the big one, ~877MB)...")
    ratings = pd.read_csv(SRC / "ratings.csv")
    print(f"  ratings total: {len(ratings):,}")

    ratings = ratings[ratings["movieId"].isin(recent_ids)]
    print(f"  ratings on >= {MIN_YEAR} movies: {len(ratings):,}")

    # min ratings per movie
    movie_counts = ratings["movieId"].value_counts()
    keep_movies = movie_counts[movie_counts >= MIN_RATINGS_PER_MOVIE].index
    ratings = ratings[ratings["movieId"].isin(keep_movies)]
    print(f"  movies with >= {MIN_RATINGS_PER_MOVIE} ratings: {len(keep_movies):,}")

    # min ratings per user (on surviving movies)
    user_counts = ratings["userId"].value_counts()
    keep_users = user_counts[user_counts >= MIN_RATINGS_PER_USER].index
    ratings = ratings[ratings["userId"].isin(keep_users)]
    print(f"  users with >= {MIN_RATINGS_PER_USER} ratings: {len(keep_users):,}")

    final_movies = recent[recent["movieId"].isin(ratings["movieId"].unique())].copy()

    print("\n=== FINAL SUBSET ===")
    print(f"  ratings: {len(ratings):,}")
    print(f"  movies:  {len(final_movies):,}")
    print(f"  users:   {ratings['userId'].nunique():,}")
    density = len(ratings) / (len(final_movies) * ratings["userId"].nunique())
    print(f"  density: {density:.4%}  (fraction of user-movie cells filled)")
    print(f"  avg ratings/user:  {len(ratings) / ratings['userId'].nunique():.1f}")
    print(f"  avg ratings/movie: {len(ratings) / len(final_movies):.1f}")

    ratings.to_parquet(OUT / "ratings.parquet", index=False)
    final_movies.to_parquet(OUT / "movies.parquet", index=False)
    print(f"\nWrote {OUT}/ratings.parquet and movies.parquet")


if __name__ == "__main__":
    main()
