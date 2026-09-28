#!/usr/bin/env python
"""Benchmark harness for Gemma 4 E4B Q4_0 GGUF via llama-cpp-python.

*** REMOTE GPU ONLY. NOT YET RUN — no result exists for this experiment.
When this is completed, its result should be appended to results.jsonl
via free_gpu_llm.benchmarking and the README updated with the real number. ***

Usage:
    python experiments/gemma4-e4b/benchmark.py --model-path /path/to/gemma-4-E4B_q4_0-it.gguf
"""

from __future__ import annotations

import argparse
import time

from free_gpu_llm.benchmarking import BenchmarkResult, append_result, get_git_commit, tokens_per_second


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default="Explain what a transformer neural network is in simple technical terms. Give a concise technical explanation.")
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", default="experiments/gemma4-e4b/results.jsonl")
    args = parser.parse_args()

    from llama_cpp import Llama

    llm = Llama(model_path=args.model_path, n_gpu_layers=args.n_gpu_layers, n_ctx=args.context)

    start = time.perf_counter()
    output = llm(args.prompt, max_tokens=args.max_tokens)
    elapsed = time.perf_counter() - start

    generated_tokens = output["usage"]["completion_tokens"]

    result = BenchmarkResult(
        model="gemma-4-E4B-it-q4_0",
        hardware="1x Tesla T4",
        model_path=args.model_path,
        tokens_per_second=tokens_per_second(generated_tokens, elapsed),
        quantization="Q4_0",
        context_length=args.context,
        prompt=args.prompt,
        generated_tokens=generated_tokens,
        elapsed_seconds=elapsed,
        git_commit=get_git_commit(),
        valid=True,
        notes="First completed Gemma 4 E4B benchmark run.",
    )
    append_result(result, args.output)
    print(f"{result.tokens_per_second:.2f} tok/s -> recorded to {args.output}")


if __name__ == "__main__":
    main()
