"""Search-trigger heuristic + end-to-end retrieval-augmentation pipeline.

This is a keyword heuristic for deciding when to augment a prompt with web
search results — it is NOT the model deciding to call a tool. Kept as a
single small function so it's easy to replace with real tool-calling later.
"""

from __future__ import annotations

from free_gpu_llm.web.retrieval import format_context, retrieve

# Historical trigger keywords used in the original Gemma web-augmented chatbot.
SEARCH_TRIGGER_KEYWORDS = [
    "latest",
    "recent",
    "today",
    "yesterday",
    "tomorrow",
    "current",
    "now",
    "news",
    "price",
    "stock",
    "weather",
    "score",
    "results",
    "update",
    "updates",
    "2026",
    "2025",
    "who is",
    "what happened",
    "search",
    "look up",
    "find",
    "according to",
]


def should_search(message: str) -> bool:
    """Return True if `message` contains a search-trigger keyword.

    Simple substring heuristic (not NLU) — matches the original
    experiment's behavior. False positives/negatives are expected; this is
    documented as a heuristic, not a classifier.
    """
    if not message:
        return False
    lowered = message.lower()
    return any(keyword in lowered for keyword in SEARCH_TRIGGER_KEYWORDS)


def build_augmented_context(message: str, max_results: int = 5) -> str:
    """If `message` looks like it needs fresh web info, retrieve it and
    return a formatted context block; otherwise return "" (no augmentation).
    Never raises — retrieval failures degrade to no augmentation.
    """
    if not should_search(message):
        return ""
    documents = retrieve(message, max_results=max_results)
    return format_context(documents)
