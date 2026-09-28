#!/usr/bin/env python
"""Benchmark harness for the Qwen3-30B-A3B manual-split run.

*** REMOTE GPU ONLY. NOT VERIFIED end-to-end (see README.md). ***

Historical observed result for this configuration: ~1.8 tokens/sec.
"""

from __future__ import annotations

import argparse
import time

from run import load_model_and_tokenizer

from free_gpu_llm.benchmarking import BenchmarkResult, append_result, get_git_commit, tokens_per_second
from free_gpu_llm.config import get_hf_token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Explain what a transformer neural network is in simple technical terms.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="experiments/qwen3-30b-a3b/results.jsonl")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(hf_token=get_hf_token())
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    elapsed = time.perf_counter() - start

    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    generated_count = len(new_tokens)

    result = BenchmarkResult(
        model="Qwen3-30B-A3B-GPTQ-Int4",
        hardware="2x Tesla T4",
        model_path="Qwen/Qwen3-30B-A3B-GPTQ-Int4",
        tokens_per_second=tokens_per_second(generated_count, elapsed),
        quantization="GPTQ-Int4 (backend=torch)",
        dtype="float16",
        generated_tokens=generated_count,
        elapsed_seconds=elapsed,
        prompt=args.prompt,
        git_commit=get_git_commit(),
        valid=True,
        notes="Manual 2xT4 layer split (GPU0=28 layers, GPU1=20 layers), GPTQ backend=torch (Marlin incompatible with T4)",
    )
    append_result(result, args.output)
    print(f"{result.tokens_per_second:.2f} tok/s -> recorded to {args.output}")


if __name__ == "__main__":
    main()
