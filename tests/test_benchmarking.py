"""Lightweight tests for benchmarking.py — no GPU, no models, no network."""

import pytest

from free_gpu_llm.benchmarking import (
    BenchmarkResult,
    append_result,
    load_results,
    parse_llama_cpp_timings,
    tokens_per_second,
)


def test_tokens_per_second_basic():
    assert tokens_per_second(256, 20.0) == pytest.approx(12.8)


def test_tokens_per_second_rejects_non_positive_elapsed():
    with pytest.raises(ValueError):
        tokens_per_second(100, 0)
    with pytest.raises(ValueError):
        tokens_per_second(100, -1)


def test_benchmark_result_to_dict_roundtrip():
    result = BenchmarkResult(
        model="Muse-Glimmer-30B-Q4_K_M",
        hardware="2x Tesla T4",
        tokens_per_second=17.9,
        quantization="Q4_K_M",
        context_length=2048,
        valid=True,
        notes="DFlash speculative decoding",
    )
    d = result.to_dict()
    assert d["model"] == "Muse-Glimmer-30B-Q4_K_M"
    assert d["tokens_per_second"] == 17.9
    assert d["valid"] is True
    assert "timestamp" in d


def test_append_and_load_results_preserves_history(tmp_path):
    path = tmp_path / "results.jsonl"

    r1 = BenchmarkResult(model="gemma-3-4b-bf16", hardware="1x Tesla T4", tokens_per_second=10.04, valid=True)
    r2 = BenchmarkResult(model="gemma-3-4b-fp16", hardware="1x Tesla T4", tokens_per_second=0.0, valid=False,
                          notes="NaN logits — invalid benchmark")

    append_result(r1, path)
    append_result(r2, path)

    loaded = load_results(path)
    assert len(loaded) == 2
    assert loaded[0]["model"] == "gemma-3-4b-bf16"
    assert loaded[0]["valid"] is True
    assert loaded[1]["valid"] is False


def test_load_results_missing_file_returns_empty_list(tmp_path):
    assert load_results(tmp_path / "does_not_exist.jsonl") == []


def test_parse_llama_cpp_timings_extracts_prompt_and_eval_tps():
    sample_output = """
llama_print_timings:        load time =    500.12 ms
llama_print_timings:      sample time =     50.00 ms /   256 runs   (    0.20 ms per token,  5120.00 tokens per second)
llama_print_timings: prompt eval time =    200.00 ms /    21 tokens (    9.52 ms per token,   104.70 tokens per second)
llama_print_timings:        eval time =  18613.14 ms /   255 runs   (   73.00 ms per token,    13.70 tokens per second)
llama_print_timings:       total time =  18863.26 ms /   276 tokens
"""
    parsed = parse_llama_cpp_timings(sample_output)
    assert parsed["prompt_tokens_per_second"] == pytest.approx(104.70)
    assert parsed["eval_tokens_per_second"] == pytest.approx(13.70)


def test_parse_llama_cpp_timings_missing_lines_returns_none():
    parsed = parse_llama_cpp_timings("some unrelated log output\nno timings here")
    assert parsed["prompt_tokens_per_second"] is None
    assert parsed["eval_tokens_per_second"] is None
