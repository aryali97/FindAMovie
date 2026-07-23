"""LLM reranker stage (Week 3) — the artifact's centerpiece.

Cosine retrieval (pgvector) pulls N cheap candidates; this module makes ONE Claude
call that reorders them to the best top-k for the user's taste. It sits between
retrieval and the response in GET /recommendations.

Deliberate structure so later weeks wrap cleanly:
  - The prompt lives here and nowhere else.
  - The SYSTEM block is fully static (no per-request interpolation) so it becomes
    the prefix-cache target in Week 5. Everything that varies per request — the
    liked titles, the candidate list, the requested k — goes in the user turn.
  - rerank() returns a RerankResult carrying token counts + latency, so the Week 4
    instrumentation reads fields off it instead of re-plumbing the call.
  - Structured output via a single FORCED tool call: the model can only answer by
    emitting {"order": [ids]}, which parses deterministically — no prose scraping.

The model is claude-haiku-4-5 (see config): cheapest Claude model, fastest (helps
the latency story), and reranking ~40 candidates is well within its ability.
"""

import time
from dataclasses import dataclass

from anthropic import AsyncAnthropic

from .models import MovieDetail

# Static system prompt — MUST stay byte-stable across requests to cache in Week 5.
# The "how many" and the data are supplied in the user turn, never here.
SYSTEM_PROMPT = (
    "You are a reranking stage in a movie recommendation system. You are given a "
    "user's liked movies and a list of candidate movies that were already "
    "retrieved by semantic similarity. Reorder the candidates so the ones that "
    "best match the user's taste come first, judging by theme, tone, genre, and "
    "sensibility rather than surface popularity. Then return the best matches, "
    "best first, by calling the submit_ranking tool with their movie ids.\n\n"
    "Rules:\n"
    "- Only use movie ids that appear in the candidate list. Never invent ids.\n"
    "- Return exactly the number of ids the user asks for, or fewer if there are "
    "fewer candidates.\n"
    "- Order matters: index 0 is the single best recommendation."
)

RANKING_TOOL = {
    "name": "submit_ranking",
    "description": "Submit the reranked candidate movie ids, best match first.",
    "input_schema": {
        "type": "object",
        "properties": {
            "order": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "Candidate movie ids ordered best-first. Only ids from the "
                    "candidate list; length must not exceed the requested count."
                ),
            }
        },
        "required": ["order"],
    },
}


@dataclass
class RerankResult:
    """Reranked ids plus the numbers Week 4 instrumentation needs."""

    order: list[int]  # candidate ids, best-first, already filtered to valid ids
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


def build_client() -> AsyncAnthropic:
    """One client per process, opened in the app lifespan (see main.py).

    Reads ANTHROPIC_API_KEY from the environment (loaded from .env by config).
    """
    return AsyncAnthropic()


def _candidate_block(m: MovieDetail) -> str:
    """Compact, token-bounded description of one candidate for the prompt."""
    parts = [f"[{m.movie_id}] {m.title}" + (f" ({m.year})" if m.year else "")]
    if m.genres:
        parts.append("genres: " + ", ".join(m.genres))
    if m.director:
        parts.append("dir: " + m.director)
    if m.cast:
        parts.append("cast: " + ", ".join(m.cast[:4]))
    line = " | ".join(parts)
    if m.overview:
        overview = m.overview.strip()
        if len(overview) > 300:
            overview = overview[:297].rstrip() + "..."
        line += "\n" + overview
    return line


def _build_user_message(
    liked_titles: list[str], candidates: list[MovieDetail], k: int
) -> str:
    liked = "\n".join(f"- {t}" for t in liked_titles) or "(none provided)"
    cand = "\n\n".join(_candidate_block(m) for m in candidates)
    return (
        f"Movies the user has liked:\n{liked}\n\n"
        f"Candidate movies to rerank:\n{cand}\n\n"
        f"Return the {k} best-matching candidate ids, best first, "
        "by calling submit_ranking."
    )


async def rerank(
    client: AsyncAnthropic,
    model: str,
    liked_titles: list[str],
    candidates: list[MovieDetail],
    k: int,
    max_tokens: int = 1024,
) -> RerankResult:
    """One Claude call: reorder `candidates` to the best top-k for the user.

    Returns the ordered candidate ids (filtered to real candidate ids, de-duped,
    truncated to k) plus token/latency stats. On no valid tool output, falls back
    to the incoming cosine order so the endpoint always returns something.
    """
    valid_ids = {m.movie_id for m in candidates}
    user_message = _build_user_message(liked_titles, candidates, k)

    started = time.perf_counter()
    resp = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        tools=[RANKING_TOOL],
        tool_choice={"type": "tool", "name": "submit_ranking"},
        messages=[{"role": "user", "content": user_message}],
    )
    latency_ms = (time.perf_counter() - started) * 1000.0

    order: list[int] = []
    seen: set[int] = set()
    for block in resp.content:
        if block.type == "tool_use" and block.name == "submit_ranking":
            for mid in block.input.get("order", []):
                if mid in valid_ids and mid not in seen:
                    order.append(mid)
                    seen.add(mid)
    # Fallback: if the model returned nothing usable, keep the cosine order.
    if not order:
        order = [m.movie_id for m in candidates]
    order = order[:k]

    u = resp.usage
    return RerankResult(
        order=order,
        model=resp.model,
        latency_ms=latency_ms,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )
