"""trafilatura-based webpage text extraction."""

from __future__ import annotations

from typing import Optional

from free_gpu_llm.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_CHARS = 4000


def truncate_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Truncate extracted text to a character budget, on a word boundary
    where possible, and mark truncation explicitly rather than silently
    cutting mid-sentence with no indication."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars * 0.5:
        cut = cut[:last_space]
    return cut.rstrip() + " [...truncated]"


def extract_text(url: str, timeout: float = 10.0, max_chars: int = DEFAULT_MAX_CHARS) -> Optional[str]:
    """Fetch and extract readable text from a webpage.

    Returns None on any failure (network error, non-HTML, extraction
    yielding no content) instead of raising — a single failed page must
    never crash the chatbot or the rest of the retrieval batch.
    """
    if not url or not url.strip():
        return None

    try:
        import trafilatura
    except ImportError:
        logger.error("trafilatura is not installed; install the 'web' extra to enable extraction.")
        return None

    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=False)
    except Exception:
        logger.exception("Failed to fetch %s", url)
        return None

    if not downloaded:
        logger.warning("No content downloaded from %s", url)
        return None

    try:
        extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    except Exception:
        logger.exception("Failed to extract text from %s", url)
        return None

    if not extracted or not extracted.strip():
        logger.warning("Extraction produced no usable text for %s", url)
        return None

    return truncate_text(extracted.strip(), max_chars=max_chars)
