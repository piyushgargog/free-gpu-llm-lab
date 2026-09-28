"""DDGS (DuckDuckGo Search) wrapper.

Network calls only happen inside `search()`. URL normalization/dedup logic
is factored out into pure functions so it can be unit-tested without any
network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from free_gpu_llm.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


def normalize_url(url: str) -> str:
    """Normalize a URL for dedup purposes: lowercase scheme/host, strip
    trailing slash, drop fragment. Does not touch query params (some sites
    are meaningfully different per query string)."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower() or "https"
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def dedup_results(results: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicate results by normalized URL, preserving first-seen order."""
    seen: set[str] = set()
    deduped: list[SearchResult] = []
    for r in results:
        key = normalize_url(r.url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped


def search(query: str, max_results: int = 5, timeout: float = 10.0) -> list[SearchResult]:
    """Run a DDGS text search. Returns [] on any failure instead of raising —
    a failed search must never crash the chatbot."""
    query = (query or "").strip()
    if not query:
        return []

    try:
        from ddgs import DDGS
    except ImportError:
        logger.error("ddgs is not installed; install the 'web' extra to enable search.")
        return []

    try:
        with DDGS(timeout=timeout) as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        logger.exception("Web search failed for query=%r", query)
        return []

    results = [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("href", r.get("url", "")),
            snippet=r.get("body", ""),
        )
        for r in raw_results
        if r.get("href") or r.get("url")
    ]
    return dedup_results(results)[:max_results]
