"""Web search -- a plain, no-key way for agents to look things up.

Used by Forge (datasheets, techniques), Scout (schools, coaches), and Iris
(news). Returns a normalized list of results regardless of backend:

    [{"title": ..., "url": ..., "snippet": ...}, ...]

Backend is config-selected (``SEARCH_BACKEND``). Today only 'duckduckgo' is
implemented -- no API key, real web results. The seam is deliberate: when you
later want higher reliability, add a Brave/SerpAPI backend keyed off
``config.search_api_key`` and the ``search()`` signature agents call never
changes.
"""

from __future__ import annotations

from ddgs import DDGS

from skills.config import config
from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)


def _duckduckgo(query: str, max_results: int, region: str | None) -> list[dict]:
    """DuckDuckGo backend. Normalizes ddgs' title/href/body to our shape."""
    raw = DDGS().text(query, max_results=max_results, region=region or "wt-wt")
    return [
        {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
        for r in raw
    ]


@skill
def search(query: str, max_results: int = 5, region: str | None = None) -> Result:
    """Search the web. Returns up to ``max_results`` {title, url, snippet} dicts.

    ``region`` is an optional locale like 'us-en'; defaults to worldwide. An
    empty result list is a success, not a failure -- the search ran, it just
    found nothing.
    """
    backend = config.search_backend
    if backend == "duckduckgo":
        results = _duckduckgo(query, max_results, region)
    else:
        return Result.failure(
            f"unknown SEARCH_BACKEND '{backend}' (only 'duckduckgo' is implemented)"
        )

    log.info("web search '%s' -> %d results (%s)", query, len(results), backend)
    return Result.success(results)
