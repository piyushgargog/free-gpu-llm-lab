"""Lightweight tests for chat history normalization — no model/GPU involved."""

from free_gpu_llm.generation import build_messages, normalize_history, validate_logits


def test_normalize_history_tuple_format():
    history = [("hello", "hi there"), ("how are you", "good")]
    turns = normalize_history(history)
    assert len(turns) == 4
    assert turns[0].role == "user" and turns[0].content == "hello"
    assert turns[1].role == "assistant" and turns[1].content == "hi there"


def test_normalize_history_drops_in_flight_none_assistant():
    history = [("hello", None)]
    turns = normalize_history(history)
    assert len(turns) == 1
    assert turns[0].role == "user"


def test_normalize_history_dict_format():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    turns = normalize_history(history)
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"


def test_normalize_history_drops_empty_strings():
    history = [("  ", "hello"), ("hi", "  ")]
    turns = normalize_history(history)
    assert len(turns) == 2
    assert turns[0].content == "hello"
    assert turns[1].content == "hi"


def test_normalize_history_empty_and_none():
    assert normalize_history(None) == []
    assert normalize_history([]) == []


def test_build_messages_includes_system_history_and_new_message():
    messages = build_messages(
        system_prompt="You are a helpful assistant.",
        history=[("earlier question", "earlier answer")],
        user_message="new question",
    )
    assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert messages[1] == {"role": "user", "content": "earlier question"}
    assert messages[2] == {"role": "assistant", "content": "earlier answer"}
    assert messages[3] == {"role": "user", "content": "new question"}


def test_build_messages_no_system_prompt():
    messages = build_messages(system_prompt=None, history=[], user_message="hi")
    assert messages == [{"role": "user", "content": "hi"}]


def test_validate_logits_flags_nan():
    result = validate_logits([1.0, 2.0, float("nan"), 3.0])
    assert result.total == 4
    assert result.nan_count == 1
    assert result.inf_count == 0
    assert result.valid is False


def test_validate_logits_flags_inf():
    result = validate_logits([1.0, float("inf"), float("-inf")])
    assert result.nan_count == 0
    assert result.inf_count == 2
    assert result.valid is False


def test_validate_logits_all_clean_is_valid():
    result = validate_logits([1.0, 2.0, 3.0, -4.5])
    assert result.valid is True
