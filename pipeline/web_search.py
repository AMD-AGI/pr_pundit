"""
Web search utility shared across the pipeline.

Uses DuckDuckGo (via ddgs) — free, no API key required.

Returns a list of result dicts:
  [{"title": "...", "href": "https://...", "body": "...snippet..."}, ...]
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULTS = 5


def web_search(query: str, max_results: int = _DEFAULT_MAX_RESULTS) -> list[dict]:
    """Search the web with DuckDuckGo.

    Args:
        query:       Search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        List of dicts with keys: title, href, body.
        Returns [] on failure — never raises.
    """
    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=max_results))
        logger.debug("web_search(%r) → %d results", query, len(results))
        return results
    except Exception as exc:
        logger.warning("web_search failed for %r: %s", query, exc)
        return []


def format_search_results(results: list[dict], max_chars: int = 3000) -> str:
    """Format search results as a compact readable string for LLM context."""
    if not results:
        return "(no results)"
    lines: list[str] = []
    total = 0
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        href = r.get("href", "")
        body = r.get("body", "")
        entry = f"[{i}] {title}\n    {href}\n    {body}"
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry)
    return "\n\n".join(lines)
