"""Lightweight tests for the web retrieval pure-logic helpers.

No network calls — search()/extract_text() are not exercised here since
they require live network access; only the pure functions are tested.
"""

from free_gpu_llm.web.extract import truncate_text
from free_gpu_llm.web.pipeline import should_search
from free_gpu_llm.web.search import SearchResult, dedup_results, normalize_url


def test_normalize_url_strips_trailing_slash_and_fragment():
    assert normalize_url("https://Example.com/Path/") == "https://example.com/Path"
    assert normalize_url("https://example.com/path#section") == "https://example.com/path"


def test_normalize_url_keeps_query_string():
    assert normalize_url("https://example.com/search?q=abc") == "https://example.com/search?q=abc"


def test_dedup_results_removes_duplicate_urls_keeps_first():
    results = [
        SearchResult(title="A", url="https://example.com/page/"),
        SearchResult(title="A duplicate", url="https://example.com/page"),
        SearchResult(title="B", url="https://other.com/page"),
    ]
    deduped = dedup_results(results)
    assert len(deduped) == 2
    assert deduped[0].title == "A"
    assert deduped[1].url == "https://other.com/page"


def test_truncate_text_short_text_unchanged():
    text = "short text"
    assert truncate_text(text, max_chars=100) == text


def test_truncate_text_long_text_marked_truncated():
    text = "word " * 1000
    truncated = truncate_text(text, max_chars=50)
    assert truncated.endswith("[...truncated]")
    assert len(truncated) <= 50 + len(" [...truncated]")


def test_should_search_triggers_on_keyword():
    assert should_search("what is the latest news on GPUs") is True
    assert should_search("What's the current stock price of NVDA?") is True


def test_should_search_false_for_generic_question():
    assert should_search("Explain what a transformer neural network is") is False


def test_should_search_handles_empty_input():
    assert should_search("") is False
    assert should_search(None) is False
