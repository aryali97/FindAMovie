"""Fetch TMDB metadata for the subset movies and cache it for embedding.

For each movie in data/subset/movies.parquet we resolve its tmdbId via
data/ml-32m/links.csv and pull, in a single `append_to_response=credits` call:

  - overview   (plot text — the primary embedding signal)
  - tagline
  - genres     (list of genre names)
  - director   (crew member with job == "Director")
  - cast       (top TOP_CAST billed actors, by TMDB's `order` field)

Design notes:
  - Auth uses the TMDB v4 read token as a Bearer header (TMDB_READ_TOKEN).
  - Requests run on a small thread pool but share ONE global token-bucket
    rate limiter, so we never exceed RATE_LIMIT_RPS no matter the worker count.
  - Results are appended to an append-only JSONL cache as they arrive, so a
    crash or Ctrl-C is fully resumable: a re-run skips tmdbIds already cached.
  - A final consolidation step writes data/tmdb/metadata.parquet from the JSONL.
  - 429 responses honor the Retry-After header; 5xx/timeouts get capped
    exponential backoff.

Run:  env/bin/python scripts/fetch_tmdb.py
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# --- tunable knobs ---------------------------------------------------------
RATE_LIMIT_RPS = 20        # global cap, comfortably under TMDB's ~50/s soft ceiling
WORKERS = 8                # concurrent in-flight requests
TOP_CAST = 6               # keep director + this many billed actors
REQUEST_TIMEOUT = 15       # seconds per HTTP request
MAX_RETRIES = 5            # for 429 / 5xx / network errors
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SUBSET_MOVIES = ROOT / "data" / "subset" / "movies.parquet"
LINKS = ROOT / "data" / "ml-32m" / "links.csv"
OUT_DIR = ROOT / "data" / "tmdb"
CACHE_JSONL = OUT_DIR / "metadata.jsonl"     # append-only, resumable
OUT_PARQUET = OUT_DIR / "metadata.parquet"   # consolidated final artifact

API_URL = "https://api.themoviedb.org/3/movie/{id}"


class RateLimiter:
    """Simple thread-safe token bucket: caps calls to `rps` per second."""

    def __init__(self, rps: float):
        self._min_interval = 1.0 / rps
        self._lock = threading.Lock()
        self._next_time = time.monotonic()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_time - now
            # schedule this call's slot; if we're behind, go now but push the slot forward
            start = max(now, self._next_time)
            self._next_time = start + self._min_interval
        if wait > 0:
            time.sleep(wait)


def load_done_ids() -> set[int]:
    """tmdbIds already present in the JSONL cache (for resume)."""
    done: set[int] = set()
    if CACHE_JSONL.exists():
        with CACHE_JSONL.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(int(json.loads(line)["tmdbId"]))
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue
    return done


def parse_movie(movie_id: int, tmdb_id: int, payload: dict) -> dict:
    """Reduce a TMDB movie payload to the fields we cache."""
    genres = [g["name"] for g in payload.get("genres", []) if g.get("name")]

    credits = payload.get("credits", {}) or {}
    director = next(
        (c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"),
        None,
    )
    cast_sorted = sorted(
        credits.get("cast", []), key=lambda c: c.get("order", 1_000_000)
    )
    cast = [c["name"] for c in cast_sorted[:TOP_CAST] if c.get("name")]

    overview = (payload.get("overview") or "").strip()
    return {
        "movieId": int(movie_id),
        "tmdbId": int(tmdb_id),
        "tmdb_title": payload.get("title"),
        "overview": overview,
        "tagline": (payload.get("tagline") or "").strip() or None,
        "genres": genres,
        "director": director,
        "cast": cast,
        "status": "ok" if overview else "no_overview",
    }


def fetch_one(session: requests.Session, limiter: RateLimiter,
              movie_id: int, tmdb_id: int) -> dict:
    """Fetch + parse one movie, with retry/backoff. Always returns a record."""
    url = API_URL.format(id=tmdb_id)
    params = {"append_to_response": "credits"}

    for attempt in range(MAX_RETRIES):
        limiter.acquire()
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            time.sleep(min(2 ** attempt, 30))  # network hiccup: backoff + retry
            continue

        if resp.status_code == 200:
            return parse_movie(movie_id, tmdb_id, resp.json())
        if resp.status_code == 404:
            return {"movieId": int(movie_id), "tmdbId": int(tmdb_id),
                    "tmdb_title": None, "overview": "", "tagline": None,
                    "genres": [], "director": None, "cast": [],
                    "status": "not_found"}
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 1))
            time.sleep(retry_after + 0.25)
            continue
        if 500 <= resp.status_code < 600:
            time.sleep(min(2 ** attempt, 30))
            continue
        # unexpected 4xx: don't hammer, record and move on
        return {"movieId": int(movie_id), "tmdbId": int(tmdb_id),
                "tmdb_title": None, "overview": "", "tagline": None,
                "genres": [], "director": None, "cast": [],
                "status": f"error_{resp.status_code}"}

    return {"movieId": int(movie_id), "tmdbId": int(tmdb_id),
            "tmdb_title": None, "overview": "", "tagline": None,
            "genres": [], "director": None, "cast": [],
            "status": "error_retries_exhausted"}


def build_todo() -> pd.DataFrame:
    """Subset movieIds joined to tmdbIds, minus anything already cached."""
    movies = pd.read_parquet(SUBSET_MOVIES, columns=["movieId"])
    links = pd.read_csv(LINKS, usecols=["movieId", "tmdbId"])
    todo = movies.merge(links, on="movieId", how="left")

    missing_tmdb = todo["tmdbId"].isna().sum()
    if missing_tmdb:
        print(f"  {missing_tmdb} subset movies have no tmdbId in links.csv (skipped)")
    todo = todo.dropna(subset=["tmdbId"]).copy()
    todo["tmdbId"] = todo["tmdbId"].astype(int)

    done = load_done_ids()
    if done:
        before = len(todo)
        todo = todo[~todo["tmdbId"].isin(done)]
        print(f"  resuming: {len(done)} already cached, {before - len(todo)} skipped")
    return todo


def consolidate() -> None:
    """Read the JSONL cache and write the final Parquet artifact."""
    rows = []
    with CACHE_JSONL.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows).drop_duplicates(subset=["tmdbId"], keep="last")
    df.to_parquet(OUT_PARQUET, index=False)

    counts = df["status"].value_counts()
    print("\n=== TMDB CACHE ===")
    print(f"  total cached: {len(df):,}")
    for status, n in counts.items():
        print(f"    {status:24} {n:,}")
    usable = (df["status"] == "ok").sum()
    print(f"  usable (has overview): {usable:,}  ({usable / len(df):.1%})")
    print(f"\nWrote {OUT_PARQUET}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    token = os.environ.get("TMDB_READ_TOKEN")
    if not token:
        raise SystemExit("TMDB_READ_TOKEN not set — copy .env.example to .env and fill it in")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building work list...")
    todo = build_todo()
    if todo.empty:
        print("Nothing to fetch — cache is already complete.")
        consolidate()
        return
    print(f"  fetching {len(todo):,} movies at up to {RATE_LIMIT_RPS} req/s "
          f"across {WORKERS} workers")

    limiter = RateLimiter(RATE_LIMIT_RPS)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}",
                            "accept": "application/json"})

    # Workers fetch; the main thread does all cache writes (no write lock needed).
    with CACHE_JSONL.open("a") as cache, \
            ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(fetch_one, session, limiter,
                        row.movieId, row.tmdbId): row.tmdbId
            for row in todo.itertuples(index=False)
        }
        try:
            for fut in tqdm(as_completed(futures), total=len(futures), unit="movie"):
                record = fut.result()
                cache.write(json.dumps(record) + "\n")
                cache.flush()
        except KeyboardInterrupt:
            print("\nInterrupted — progress saved to cache, re-run to resume.")
            raise

    consolidate()


if __name__ == "__main__":
    main()
