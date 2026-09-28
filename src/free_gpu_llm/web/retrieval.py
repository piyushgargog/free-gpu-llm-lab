"""Combine search + extraction into retrieved context documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from free_gpu_llm.logging_utils import get_logger
from free_gpu_llm.web.extract import extract_text
from free_gpu_llm.web.search import SearchResult, search

logger = get_logger(__name__)


@dataclass
class RetrievedDocument:
    url: str
    title: str
    text: str
    snippet: str = ""


def retrieve(query: str, max_results: int = 5, extract_timeout: float = 10.0) -> list[RetrievedDocument]:
    """Search the web for `query` and extract readable text from each hit.

    Pages that fail to fetch/extract are skipped (not raised); an empty
    search result set simply returns []. Never crashes the caller.
    """
    results = search(query, max_results=max_results)
    if not results:
        logger.info("No search results for query=%r", query)
        return []

    documents: list[RetrievedDocument] = []
    for result in results:
        text = extract_text(result.url, timeout=extract_timeout)
        if text is None:
            logger.info("Skipping unusable page: %s", result.url)
            continue
        documents.append(
            RetrievedDocument(url=result.url, title=result.title, text=text, snippet=result.snippet)
        )
    return documents


def format_context(documents: list[RetrievedDocument]) -> str:
    """Render retrieved documents into a context block to prepend/inject
    into the model prompt. Returns "" if there's nothing retrieved."""
    if not documents:
        return ""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        title = doc.title or doc.url
        blocks.append(f"[{i}] {title} ({doc.url})\n{doc.text}")
    return "\n\n".join(blocks)
